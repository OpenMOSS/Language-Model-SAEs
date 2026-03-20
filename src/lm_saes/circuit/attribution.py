"""
Build an **attribution graph** that captures the *direct*, *linear* effects
between features and next-token logits for a *prompt-specific*
**local replacement model**.

High-level algorithm (matches the 2025 ``Attribution Graphs`` paper):
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

1. **Local replacement model** - we configure gradients to flow only through
   linear components of the network, effectively bypassing attention mechanisms,
   MLP non-linearities, and layer normalization scales.
2. **Forward pass** - record residual-stream activations and mark every active
   feature.
3. **Backward passes** - for each source node (feature or logit), inject a
   *custom* gradient that selects its encoder/decoder direction.  Because the
   model is linear in the residual stream under our freezes, this contraction
   equals the *direct effect* A_{s->t}.
4. **Assemble graph** - store edge weights in a dense matrix and package a
   ``Graph`` object.  Downstream utilities can *prune* the graph to the subset
   needed for interpretation.
"""

import contextlib
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from einops import einsum
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from lm_saes.models.clt import CrossLayerTranscoder
from lm_saes.utils.logging import get_distributed_logger

from .graph import Graph
from .replacement_model import ReplacementModel
from .utils.attn_scores_attribution import compute_attn_scores_attribution
from .utils.attribution_utils import (
    compute_partial_influences,
    compute_salient_logits,
    ensure_tokenized,
    select_encoder_rows,
    select_encoder_rows_lorsa,
    select_feature_activations,
    select_scaled_decoder_vecs_lorsa,
    select_scaled_decoder_vecs_transcoder,
)
from .utils.disk_offload import offload_modules
from .utils.transcoder_set import TranscoderSet

logger = get_distributed_logger("attribution")

TranscoderType = TranscoderSet | CrossLayerTranscoder


def _normalize_runtime_device(device: torch.device | str) -> torch.device:
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and device_obj.index is None:
        device_obj = torch.device(f"cuda:{torch.cuda.current_device()}")
    return device_obj


def _device_context(device: torch.device):
    return torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()


@dataclass
class PreparedAttributionState:
    model: ReplacementModel
    ctx: "AttributionContext"
    input_ids: torch.Tensor
    logits: torch.Tensor
    lorsa_activation_matrix: torch.Tensor | None
    lorsa_attention_score: torch.Tensor | None
    lorsa_attention_pattern: torch.Tensor | None
    z_attention_pattern: torch.Tensor | None
    clt_activation_matrix: torch.Tensor
    error_vecs: torch.Tensor
    token_vecs: torch.Tensor
    lorsa_encoder_rows: torch.Tensor | None
    lorsa_attention_patterns: torch.Tensor | None
    z_attention_patterns: torch.Tensor | None
    clt_encoder_rows: torch.Tensor
    use_lorsa: bool
    device: torch.device
    lorsa_feat_layer: torch.Tensor | None
    lorsa_feat_pos: torch.Tensor | None
    lorsa_feat_idx: torch.Tensor | None
    clt_feat_layer: torch.Tensor
    clt_feat_pos: torch.Tensor
    clt_feat_idx: torch.Tensor
    ov_group_sizes: torch.Tensor | None

    @property
    def n_layers(self) -> int:
        return self.clt_activation_matrix.shape[0]

    @property
    def n_pos(self) -> int:
        return self.clt_activation_matrix.shape[1]

    @property
    def lorsa_feature_count(self) -> int:
        return 0 if self.lorsa_activation_matrix is None else self.lorsa_activation_matrix._nnz()

    @property
    def clt_feature_count(self) -> int:
        return self.clt_activation_matrix._nnz()

    @property
    def total_active_feats(self) -> int:
        return self.lorsa_feature_count + self.clt_feature_count

    @property
    def logit_offset(self) -> int:
        if self.use_lorsa:
            return self.total_active_feats + 2 * self.n_layers * self.n_pos + self.n_pos
        return self.total_active_feats + self.n_layers * self.n_pos + self.n_pos

    def idx_to_layer(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to(self.device)
        if not self.use_lorsa:
            return self.clt_feat_layer[idx]

        assert self.lorsa_feat_layer is not None
        result = torch.empty_like(idx)
        is_lorsa = idx < self.lorsa_feature_count
        if is_lorsa.any():
            result[is_lorsa] = 2 * self.lorsa_feat_layer[idx[is_lorsa]]
        if (~is_lorsa).any():
            clt_idx = idx[~is_lorsa] - self.lorsa_feature_count
            result[~is_lorsa] = 2 * self.clt_feat_layer[clt_idx] + 1
        return result

    def idx_to_pos(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to(self.device)
        if not self.use_lorsa:
            return self.clt_feat_pos[idx]

        assert self.lorsa_feat_pos is not None
        result = torch.empty_like(idx)
        is_lorsa = idx < self.lorsa_feature_count
        if is_lorsa.any():
            result[is_lorsa] = self.lorsa_feat_pos[idx[is_lorsa]]
        if (~is_lorsa).any():
            clt_idx = idx[~is_lorsa] - self.lorsa_feature_count
            result[~is_lorsa] = self.clt_feat_pos[clt_idx]
        return result

    def idx_to_encoder_rows(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to(self.device)
        if not self.use_lorsa:
            return self.clt_encoder_rows[idx]

        assert self.lorsa_encoder_rows is not None
        result = torch.empty((idx.shape[0], self.token_vecs.shape[-1]), device=self.device, dtype=self.token_vecs.dtype)
        is_lorsa = idx < self.lorsa_feature_count
        if is_lorsa.any():
            result[is_lorsa] = self.lorsa_encoder_rows[idx[is_lorsa]]
        if (~is_lorsa).any():
            clt_idx = idx[~is_lorsa] - self.lorsa_feature_count
            result[~is_lorsa] = self.clt_encoder_rows[clt_idx]
        return result

    def idx_to_pattern(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to(self.device)
        pattern_dtype = self.token_vecs.dtype
        if not self.use_lorsa:
            return torch.nn.functional.one_hot(self.clt_feat_pos[idx], num_classes=self.n_pos).to(pattern_dtype)

        assert self.lorsa_attention_patterns is not None
        result = torch.empty((idx.shape[0], self.n_pos), device=self.device, dtype=pattern_dtype)
        is_lorsa = idx < self.lorsa_feature_count
        if is_lorsa.any():
            result[is_lorsa] = self.lorsa_attention_patterns[idx[is_lorsa]].to(pattern_dtype)
        if (~is_lorsa).any():
            clt_idx = idx[~is_lorsa] - self.lorsa_feature_count
            result[~is_lorsa] = torch.nn.functional.one_hot(
                self.clt_feat_pos[clt_idx], num_classes=self.n_pos
            ).to(pattern_dtype)
        return result

    def idx_to_z_pattern(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to(self.device)
        pattern_dtype = self.token_vecs.dtype
        if not self.use_lorsa:
            return torch.nn.functional.one_hot(self.clt_feat_pos[idx], num_classes=self.n_pos).to(pattern_dtype)

        assert self.z_attention_patterns is not None
        result = torch.empty((idx.shape[0], self.n_pos), device=self.device, dtype=pattern_dtype)
        is_lorsa = idx < self.lorsa_feature_count
        if is_lorsa.any():
            result[is_lorsa] = self.z_attention_patterns[idx[is_lorsa]].to(pattern_dtype)
        if (~is_lorsa).any():
            clt_idx = idx[~is_lorsa] - self.lorsa_feature_count
            result[~is_lorsa] = torch.nn.functional.one_hot(
                self.clt_feat_pos[clt_idx], num_classes=self.n_pos
            ).to(pattern_dtype)
        return result

    def idx_to_qk_idx(self, idx: torch.Tensor) -> torch.Tensor:
        assert self.use_lorsa, "QK tracing requires lorsa features"
        assert self.lorsa_feat_layer is not None and self.lorsa_feat_idx is not None and self.ov_group_sizes is not None
        idx = idx.to(self.device)
        return self.lorsa_feat_idx[idx] // self.ov_group_sizes[self.lorsa_feat_layer[idx]]


@dataclass
class FeatureBatchResult:
    idx_batch: torch.Tensor
    rows: torch.Tensor
    lorsa_pattern: torch.Tensor
    z_pattern: torch.Tensor


@dataclass
class AttributionReplica:
    state: PreparedAttributionState
    offload_handles: list[Callable]
    owns_model: bool = False

    @property
    def device(self) -> torch.device:
        return self.state.device

    def compute_feature_rows(self, idx_batch: torch.Tensor, *, retain_graph: bool = True) -> FeatureBatchResult:
        with _device_context(self.device):
            local_idx = idx_batch.to(self.device, non_blocking=self.device.type == "cuda")
            attention_patterns = self.state.idx_to_pattern(local_idx)
            rows = self.state.ctx.compute_batch(
                layers=self.state.idx_to_layer(local_idx),
                positions=self.state.idx_to_pos(local_idx),
                inject_values=self.state.idx_to_encoder_rows(local_idx),
                attention_patterns=attention_patterns,
                retain_graph=retain_graph,
            )
            return FeatureBatchResult(
                idx_batch=idx_batch.cpu(),
                rows=rows.cpu(),
                lorsa_pattern=attention_patterns.cpu(),
                z_pattern=self.state.idx_to_z_pattern(local_idx).cpu(),
            )

    def compute_qk_trace(self, idx: int, topk: int) -> tuple[int, Any]:
        assert self.state.use_lorsa, "QK tracing requires lorsa features"
        assert self.state.lorsa_activation_matrix is not None and self.state.lorsa_feat_layer is not None
        with _device_context(self.device):
            local_idx = torch.tensor([idx], device=self.device, dtype=torch.long)
            layer = int(self.state.lorsa_feat_layer[local_idx].item())
            q_pos = int(self.state.idx_to_pos(local_idx).item())
            k_pos = int(self.state.idx_to_z_pattern(local_idx).argmax(dim=-1).item())
            qk_idx = int(self.state.idx_to_qk_idx(local_idx).item())
            result = compute_attn_scores_attribution(
                self.state.model,
                self.state.lorsa_activation_matrix,
                self.state.clt_activation_matrix,
                layer,
                q_pos,
                k_pos,
                qk_idx,
                self.state.token_vecs,
                self.state.error_vecs,
                self.state.input_ids,
                topk,
            )
        return idx, result

    def cleanup(self) -> None:
        for reload_handle in self.offload_handles:
            reload_handle()
        for _, param in self.state.model._get_requires_grad_bias_params():
            param.grad = None


def _prepare_attribution_state(
    model: ReplacementModel,
    prompt: Union[str, torch.Tensor, List[int]],
    *,
    batch_size: int,
    use_lorsa: bool,
    offload: Literal["cpu", "disk", None],
    offload_handles: list[Callable],
) -> PreparedAttributionState:
    model.sync_runtime_device()
    device = _normalize_runtime_device(model.runtime_device)
    input_ids = ensure_tokenized(prompt, model.tokenizer)
    (
        logits,
        lorsa_activation_matrix,
        lorsa_attention_score,
        lorsa_attention_pattern,
        z_attention_pattern,
        clt_activation_matrix,
        error_vecs,
        token_vecs,
    ) = model.setup_attribution(input_ids, sparse=True)

    if use_lorsa:
        assert lorsa_activation_matrix is not None
        lorsa_decoder_vecs = select_scaled_decoder_vecs_lorsa(lorsa_activation_matrix, model.lorsas)
        lorsa_encoder_rows, lorsa_attention_patterns, z_attention_patterns = select_encoder_rows_lorsa(
            lorsa_activation_matrix,
            lorsa_attention_pattern,
            z_attention_pattern,
            model.lorsas,
        )
    else:
        lorsa_decoder_vecs = None
        lorsa_encoder_rows = None
        lorsa_attention_patterns = None
        z_attention_patterns = None

    clt_decoder_vecs = select_scaled_decoder_vecs_transcoder(clt_activation_matrix, model.transcoders)
    clt_encoder_rows = select_encoder_rows(clt_activation_matrix, model.transcoders)

    ctx = AttributionContext(
        lorsa_activation_matrix,
        clt_activation_matrix,
        error_vecs,
        token_vecs,
        lorsa_decoder_vecs,
        clt_decoder_vecs,
        model.attn_output_hook,
        model.mlp_output_hook,
        use_lorsa=use_lorsa,
        use_clt=isinstance(model.transcoders, CrossLayerTranscoder),
    )

    if offload:
        offload_handles += offload_modules(model.transcoders, offload)

    with ctx.install_hooks(model), _device_context(device):
        residual = model.forward(input_ids.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
        ctx._resid_activations[-1] = model.ln_final(residual)

    if offload:
        offload_handles += offload_modules([block.mlp for block in model.blocks], offload)
        if use_lorsa:
            offload_handles += offload_modules([block.attn for block in model.blocks], offload)

    lorsa_feat_layer, lorsa_feat_pos, lorsa_feat_idx = (
        lorsa_activation_matrix.indices() if use_lorsa and lorsa_activation_matrix is not None else (None, None, None)
    )
    clt_feat_layer, clt_feat_pos, clt_feat_idx = clt_activation_matrix.indices()
    ov_group_sizes = (
        torch.tensor([model.lorsas[i].cfg.ov_group_size for i in range(model.cfg.n_layers)], device=device)
        if use_lorsa
        else None
    )

    return PreparedAttributionState(
        model=model,
        ctx=ctx,
        input_ids=input_ids,
        logits=logits,
        lorsa_activation_matrix=lorsa_activation_matrix,
        lorsa_attention_score=lorsa_attention_score,
        lorsa_attention_pattern=lorsa_attention_pattern,
        z_attention_pattern=z_attention_pattern,
        clt_activation_matrix=clt_activation_matrix,
        error_vecs=error_vecs,
        token_vecs=token_vecs,
        lorsa_encoder_rows=lorsa_encoder_rows,
        lorsa_attention_patterns=lorsa_attention_patterns,
        z_attention_patterns=z_attention_patterns,
        clt_encoder_rows=clt_encoder_rows,
        use_lorsa=use_lorsa,
        device=device,
        lorsa_feat_layer=lorsa_feat_layer,
        lorsa_feat_pos=lorsa_feat_pos,
        lorsa_feat_idx=lorsa_feat_idx,
        clt_feat_layer=clt_feat_layer,
        clt_feat_pos=clt_feat_pos,
        clt_feat_idx=clt_feat_idx,
        ov_group_sizes=ov_group_sizes,
    )


def _build_parallel_replicas(
    model: ReplacementModel,
    prompt: Union[str, torch.Tensor, List[int]],
    *,
    batch_size: int,
    use_lorsa: bool,
    offload: Literal["cpu", "disk", None],
    parallel_devices: list[str] | None,
) -> list[AttributionReplica]:
    primary_device = str(_normalize_runtime_device(model.runtime_device))
    requested_devices = parallel_devices or []

    devices: list[str] = []
    for device in [primary_device, *requested_devices]:
        normalized = str(_normalize_runtime_device(device))
        if normalized not in devices:
            devices.append(normalized)

    replicas: list[AttributionReplica] = []
    models = [model]
    for device in devices[1:]:
        models.append(model.clone_to_device(device))

    for idx, replica_model in enumerate(models):
        offload_handles: list[Callable] = []
        state = _prepare_attribution_state(
            replica_model,
            prompt,
            batch_size=batch_size,
            use_lorsa=use_lorsa,
            offload=offload,
            offload_handles=offload_handles,
        )
        replicas.append(AttributionReplica(state=state, offload_handles=offload_handles, owns_model=idx > 0))

    return replicas


class AttributionContext:
    """Manage hooks for computing attribution rows.

    This helper caches residual-stream activations **(forward pass)** and then
    registers backward hooks that populate a write-only buffer with
    *direct-effect rows* **(backward pass)**.

    The buffer layout concatenates rows for **feature nodes**, **error nodes**,
    **token-embedding nodes**

    Args:
        activation_matrix (torch.sparse.Tensor):
            Sparse `(n_layers, n_pos, n_features)` tensor indicating **which**
            features fired at each layer/position.
        error_vectors (torch.Tensor):
            `(n_layers, n_pos, d_model)` - *residual* the CLT / PLT failed to
            reconstruct ("error nodes").
        token_vectors (torch.Tensor):
            `(n_pos, d_model)` - embeddings of the prompt tokens.
        decoder_vectors (torch.Tensor):
            `(total_active_features, d_model)` - decoder rows **only for active
            features**, already multiplied by feature activations so they
            represent a_s * W^dec.
    """

    def __init__(
        self,
        lorsa_activation_matrix: torch.sparse.Tensor,
        clt_activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        lorsa_decoder_vecs: torch.Tensor,
        clt_decoder_vecs: torch.Tensor,
        attn_output_hook: str,
        mlp_output_hook: str,
        use_lorsa: bool = True,
        use_clt: bool = True,
    ) -> None:
        if use_lorsa:
            assert lorsa_activation_matrix.shape[:-1] == clt_activation_matrix.shape[:-1], (
                "Lorsas and CLTs must have the same shape"
            )
        n_layers, n_pos, _ = clt_activation_matrix.shape

        # Forward-pass cache
        # L0Ainput, L0Minput, ... L-1Ainput, L-1Minput, pre_unembed
        if use_lorsa:
            self._resid_activations: List[torch.Tensor | None] = [None] * (2 * n_layers + 1)
        else:
            self._resid_activations: List[torch.Tensor | None] = [None] * (n_layers + 1)

        self._batch_buffer: torch.Tensor | None = None
        self.n_layers: int = n_layers
        self.use_lorsa = use_lorsa

        # Assemble all backward hooks up-front
        self._attribution_hooks = self._make_attribution_hooks(
            lorsa_activation_matrix,
            clt_activation_matrix,
            error_vectors,
            token_vectors,
            lorsa_decoder_vecs,
            clt_decoder_vecs,
            attn_output_hook,
            mlp_output_hook,
            use_lorsa,
            use_clt,
        )

        if use_lorsa:
            total_active_feats = lorsa_activation_matrix._nnz() + clt_activation_matrix._nnz()
            # total_active_feats + error_vectors + token_vectors
            self._row_size: int = total_active_feats + 2 * n_layers * n_pos + n_pos  # + logits later
        else:
            total_active_feats = clt_activation_matrix._nnz()
            # total_active_feats + error_vectors + token_vectors
            self._row_size: int = total_active_feats + n_layers * n_pos + n_pos  # + logits later

    def _caching_hooks(
        self, attn_input_hook: str, mlp_input_hook: str, model: ReplacementModel
    ) -> List[Tuple[str, Callable]]:
        """Return hooks that store residual activations layer-by-layer."""

        proxy = weakref.proxy(self)

        def _cache(acts: torch.Tensor, hook: HookPoint, *, index: int) -> torch.Tensor:
            proxy._resid_activations[index] = acts
            return acts

        hooks = []

        for layer in range(self.n_layers):
            if self.use_lorsa:
                hooks.append((f"blocks.{layer}.{attn_input_hook}", partial(_cache, index=layer * 2)))
                hooks.append((f"blocks.{layer}.{mlp_input_hook}", partial(_cache, index=layer * 2 + 1)))
            else:
                hooks.append((f"blocks.{layer}.{mlp_input_hook}", partial(_cache, index=layer)))
        hooks.append(("unembed.hook_pre", partial(_cache, index=2 * self.n_layers)))

        return hooks

    def _compute_score_hook(
        self,
        hook_name: str,
        output_vecs: torch.Tensor,
        write_index: slice,
        read_index: slice | np.ndarray = np.s_[:],
    ) -> Tuple[str, Callable]:
        """
        Factory that contracts *gradients* with an **output vector set**.
        The hook computes A_{s->t} and writes the result into an in-place buffer row.
        """

        proxy = weakref.proxy(self)

        def _hook_fn(grads: torch.Tensor, hook: HookPoint) -> None:
            proxy._batch_buffer[write_index] += einsum(
                grads.to(output_vecs.dtype)[read_index],
                output_vecs,
                "batch position d_model, position d_model -> position batch",
            )

        return hook_name, _hook_fn

    def _make_attribution_hooks(
        self,
        lorsa_activation_matrix: torch.sparse.Tensor,
        clt_activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        lorsa_decoder_vecs: torch.Tensor,
        clt_decoder_vecs: torch.Tensor,
        attn_output_hook: str,
        mlp_output_hook: str,
        use_lorsa: bool = True,
        use_clt=True,
    ) -> List[Tuple[str, Callable]]:
        """
        Create the complete backward-hook for computing attribution scores.
        """
        _, n_pos, _ = clt_activation_matrix.shape

        lorsa_error_vectors = error_vectors[: self.n_layers] if self.use_lorsa else None
        clt_error_vectors = error_vectors[self.n_layers :] if self.use_lorsa else error_vectors

        # Token-embedding nodes
        # lorsa_offset + clt_offset + attn_error_offset + mlp_error_offset
        token_offset = (
            (lorsa_activation_matrix._nnz() + clt_activation_matrix._nnz() + 2 * self.n_layers * n_pos)
            if self.use_lorsa
            else (clt_activation_matrix._nnz() + self.n_layers * n_pos)
        )

        token_hook = [
            self._compute_score_hook(
                "hook_embed",
                token_vectors,
                write_index=np.s_[token_offset : token_offset + n_pos],
            )
        ]

        mlp_hook_fn = self._make_attribution_hooks_clt if use_clt else self._make_attribution_hooks_plt

        if use_lorsa:
            out = (
                mlp_hook_fn(
                    clt_activation_matrix,
                    clt_error_vectors,
                    clt_decoder_vecs,
                    mlp_output_hook,
                    lorsa_offset=lorsa_activation_matrix._nnz(),
                )
                + self._make_attribution_hooks_lorsa(
                    lorsa_activation_matrix,
                    lorsa_error_vectors,
                    lorsa_decoder_vecs,
                    attn_output_hook,
                    clt_offset=clt_activation_matrix._nnz(),
                )
                + token_hook
            )
        else:
            out = (
                mlp_hook_fn(
                    clt_activation_matrix,
                    clt_error_vectors,
                    clt_decoder_vecs,
                    mlp_output_hook,
                    lorsa_offset=0,
                )
                + token_hook
            )

        return out

    def _make_attribution_hooks_lorsa(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        attn_output_hook: str,
        clt_offset: int,
    ) -> List[Tuple[str, Callable]]:
        """
        Create the complete backward-hook for computing attribution scores.
        activation_matrix:
            size (n_layers, n_pos, n_features)
            indices: (3, n_active_features)
            values: (n_active_features,)
        error_vectors:
            size (n_layers, n_pos, d_model)
        token_vectors:
            size (n_pos, d_model)
        decoder_vecs:
            size (n_active_features, d_model)
        """
        n_layers, n_pos, _ = activation_matrix.shape
        nnz_layers, nnz_positions, _ = activation_matrix.indices()

        # Map each layer → slice in flattened active-feature list
        _, counts = torch.unique_consecutive(nnz_layers, return_counts=True)
        edges = [0] + counts.cumsum(0).tolist()
        layer_spans = list(zip(edges[:-1], edges[1:]))

        # Feature nodes
        feature_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{attn_output_hook}",
                decoder_vecs[start:end],
                write_index=np.s_[start:end],
                read_index=np.s_[:, nnz_positions[start:end]],
            )
            for layer, (start, end) in enumerate(layer_spans)
            if start != end
        ]

        # Error nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            return activation_matrix._nnz() + clt_offset + layer * n_pos

        error_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{attn_output_hook}",
                error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
            )
            for layer in range(n_layers)
        ]

        return feature_hooks + error_hooks

    def _make_attribution_hooks_plt(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        mlp_output_hook: str,
        lorsa_offset: int,
    ) -> List[Tuple[str, Callable]]:
        """
        Create the complete backward-hook for computing attribution scores.
        activation_matrix:
            size (n_layers, n_pos, n_features)
            indices: (3, n_active_features)
            values: (n_active_features,)
        error_vectors:
            size (n_layers, n_pos, d_model)
        token_vectors:
            size (n_pos, d_model)
        decoder_vecs:
            size (n_active_features, d_model)
        """
        n_layers, n_pos, _ = activation_matrix.shape
        nnz_layers, nnz_positions, _ = activation_matrix.indices()

        # Map each layer → slice in flattened active-feature list
        _, counts = torch.unique_consecutive(nnz_layers, return_counts=True)
        edges = [0] + counts.cumsum(0).tolist()
        layer_spans = list(zip(edges[:-1], edges[1:]))

        # Feature nodes
        feature_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{mlp_output_hook}",
                decoder_vecs[start:end],
                write_index=np.s_[lorsa_offset + start : lorsa_offset + end],
                read_index=np.s_[:, nnz_positions[start:end]],
            )
            for layer, (start, end) in enumerate(layer_spans)
            if start != end
        ]

        # Error nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            # lorsa_offset + clt_offset + attn_error_offset + layer_offset
            if lorsa_offset == 0:
                return activation_matrix._nnz() + layer * n_pos
            else:
                return lorsa_offset + activation_matrix._nnz() + n_layers * n_pos + layer * n_pos

        error_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{mlp_output_hook}",
                error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
            )
            for layer in range(n_layers)
        ]

        return feature_hooks + error_hooks

    def _make_attribution_hooks_clt(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        mlp_output_hook: str,
        lorsa_offset: int,
    ) -> List[Tuple[str, Callable]]:
        """
        Create the complete backward-hook for computing attribution scores.
        activation_matrix:
            size (n_layers, n_pos, n_features)
            indices: (3, n_active_features)
            values: (n_active_features,)
        error_vectors:
            size (n_layers, n_pos, d_model)
        token_vectors:
            size (n_pos, d_model)
        decoder_vecs:
            size ((\sum_{i=0}^{n_layers} \sum_{j=0}^{i} n_active_features_layer_j), d_model)
        """

        n_layers, n_pos, _ = activation_matrix.shape
        nnz_layers, nnz_positions, _ = activation_matrix.indices()

        # Map each layer → slice in flattened active-feature list
        def _maybe_pad_for_inactive_layers(layers: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
            result = torch.zeros(n_layers, device=layers.device, dtype=counts.dtype)
            result[layers] = counts
            return result

        layers, counts = torch.unique_consecutive(nnz_layers, return_counts=True)  # active features per layer
        counts = _maybe_pad_for_inactive_layers(layers, counts)
        edges = counts.cumsum(0)  # n_layers
        decoder_layer_spans = [0] + edges.cumsum(0).tolist()
        assert edges[-1] == activation_matrix._nnz(), f"got {edges[-1]} but expected {activation_matrix._nnz()}"
        assert decoder_layer_spans[-1] == decoder_vecs.size(0), (
            f"got {decoder_layer_spans[-1]} but expected {decoder_vecs.size(0)}"
        )
        decoder_layer_spans = [
            slice(start, end) for start, end in zip(decoder_layer_spans[:-1], decoder_layer_spans[1:])
        ]

        # Feature nodes
        feature_hooks: list[Tuple[str, Callable[..., Any]]] = [
            self._compute_score_hook(
                f"blocks.{layer}.{mlp_output_hook}",
                decoder_vecs[decoder_layer_spans[layer]],
                write_index=np.s_[lorsa_offset : lorsa_offset + edges[layer]],
                read_index=np.s_[:, nnz_positions[: edges[layer]]],
            )
            for layer in range(n_layers)
        ]

        # Error nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            # lorsa_offset + clt_offset + attn_error_offset + layer_offset
            if lorsa_offset == 0:
                return activation_matrix._nnz() + layer * n_pos
            else:
                return lorsa_offset + activation_matrix._nnz() + n_layers * n_pos + layer * n_pos

        error_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{mlp_output_hook}",
                error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
            )
            for layer in range(n_layers)
        ]

        return feature_hooks + error_hooks

    @contextlib.contextmanager
    def install_hooks(self, model: "ReplacementModel"):
        """Context manager instruments the hooks for the forward and backward passes."""
        with model.hooks(
            fwd_hooks=self._caching_hooks(model.attn_input_hook, model.mlp_input_hook, model),
            bwd_hooks=self._attribution_hooks,
        ):
            yield

    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        attention_patterns: torch.Tensor | None = None,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """Return attribution rows for a batch of (layer, pos) nodes.

        The routine overrides gradients at **exact** residual-stream locations
        triggers one backward pass, and copies the rows from the internal buffer.

        Args:
            layers: 1-D tensor of layer indices *l* for the source nodes.
            positions: 1-D tensor of token positions *c* for the source nodes.
            inject_values: `(batch, d_model)` tensor with outer product
                a_s * W^(enc/dec) to inject as custom gradient.

        Returns:
            torch.Tensor: ``(batch, row_size)`` matrix - one row per node.
        """

        for resid_activation in self._resid_activations:
            assert resid_activation is not None, "Residual activations are not cached"

        batch_size = self._resid_activations[0].shape[0]
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=inject_values.device,
        )

        # Custom gradient injection (per-layer registration)
        batch_idx = torch.arange(len(layers), device=layers.device)

        def _inject(grads, *, batch_indices, pos_indices, patterns, values):
            grads_out = grads.clone().to(values.dtype)
            if patterns is not None:
                grads_out.index_put_((batch_indices,), values[:, None, :] * patterns[:, :, None])
            else:
                grads_out.index_put_((batch_indices, pos_indices), values)
            return grads_out.to(grads.dtype)

        handles = []
        layers_in_batch = layers.unique().tolist()

        for layer in layers_in_batch:
            mask = layers == layer
            if not mask.any():
                continue
            fn = partial(
                _inject,
                batch_indices=batch_idx[mask],
                pos_indices=positions[mask],
                patterns=attention_patterns[mask] if attention_patterns is not None else None,
                values=inject_values[mask],
            )
            handles.append(self._resid_activations[int(layer)].register_hook(fn))

        try:
            last_layer = max(layers_in_batch)
            sum(self._resid_activations[: last_layer + 1]).backward(
                gradient=torch.zeros_like(self._resid_activations[0]),
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        # debug: batch size > 1
        # if len(layers_in_batch) > 1:
        #     print('layers', layers)
        #     print(buf.T[0].nonzero().shape)
        #     print(buf.T[1].nonzero().shape)
        return buf.T[: len(layers)]


def attribute(
    prompt: Union[str, torch.Tensor, List[int]],
    model: ReplacementModel,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_feature_nodes: Optional[int] = None,
    offload: Literal["cpu", "disk", None] = None,
    update_interval: int = 4,
    slug: str = "untitled",
    sae_series: Optional[Union[str, List[str]]] = None,
    use_lorsa: bool = True,
    qk_tracing_topk: int = 10,
    list_of_features: Optional[List[Tuple[int, int, int, bool]]] = None,
    parallel_devices: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[float, float, str], None]] = None,
) -> Graph:
    """Compute an attribution graph for *prompt*.

    Args:
        prompt: Text, token ids, or tensor - will be tokenized if str.
        model: Frozen ``ReplacementModel``
        max_n_logits: Max number of logit nodes.
        desired_logit_prob: Keep logits until cumulative prob >= this value.
        batch_size: How many source nodes to process per backward pass.
        max_feature_nodes: Max number of feature nodes to include in the graph.
        offload: Method for offloading model parameters to save memory.
                 Options are "cpu" (move to CPU), "disk" (save to disk),
                 or None (no offloading).
        verbose: Whether to show progress information.
        update_interval: Number of batches to process before updating the feature ranking.
        list_of_features: list of (layer, feature_idx, pos, is_lorsa) tuples
        parallel_devices: Optional list of devices such as ["cuda:0", "cuda:1"].
            When at least two distinct devices are provided, the expensive
            feature-attribution and QK-tracing phases are sharded across replicas.
        progress_callback: Optional callback for tracking progress (current, total, phase).

    Returns:
        Graph: Fully dense adjacency (unpruned).
    """

    offload_handles = []
    try:
        return _run_attribution(
            model=model,
            prompt=prompt,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            offload_handles=offload_handles,
            update_interval=update_interval,
            sae_series=sae_series,
            use_lorsa=use_lorsa,
            qk_tracing_topk=qk_tracing_topk,
            list_of_features=list_of_features,
            parallel_devices=parallel_devices,
            progress_callback=progress_callback,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()


def _run_attribution(
    model,
    prompt,
    max_n_logits,
    desired_logit_prob,
    batch_size,
    max_feature_nodes,
    offload,
    offload_handles,
    update_interval=4,
    sae_series=None,
    use_lorsa: bool = True,
    qk_tracing_topk: int = 10,
    list_of_features=None,
    parallel_devices: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[float, float, str], None]] = None,
):
    start_time = time.time()
    logger.info("Phase 0/1: Preparing attribution states")
    phase_start = time.time()
    replicas = _build_parallel_replicas(
        model,
        prompt,
        batch_size=batch_size,
        use_lorsa=use_lorsa,
        offload=offload,
        parallel_devices=parallel_devices,
    )
    primary = replicas[0]
    state = primary.state
    offload_handles.extend(primary.offload_handles)
    primary.offload_handles = []

    input_ids = state.input_ids
    logits = state.logits
    lorsa_activation_matrix = state.lorsa_activation_matrix
    lorsa_attention_score = state.lorsa_attention_score
    lorsa_attention_pattern = state.lorsa_attention_pattern
    z_attention_pattern = state.z_attention_pattern
    clt_activation_matrix = state.clt_activation_matrix
    error_vecs = state.error_vecs
    token_vecs = state.token_vecs
    ctx = state.ctx
    n_layers = state.n_layers
    n_pos = state.n_pos
    total_active_feats = state.total_active_feats
    logger.info(
        f"Prepared {len(replicas)} attribution replica(s) in {time.time() - phase_start:.2f}s"
    )
    if use_lorsa and lorsa_activation_matrix is not None:
        logger.info(f"Found {total_active_feats} active features")

    # Phase 2: build input vector list
    logger.info("Phase 2: Building input vectors")
    phase_start = time.time()

    if list_of_features is not None:
        # Feature tracing: start from specific features instead of logits
        n_features = len(list_of_features)
        feature_activations = select_feature_activations(
            list_of_features, lorsa_activation_matrix, clt_activation_matrix
        )
        logger.info(f"Selected {n_features} features for tracing with activations: {feature_activations}")

        # Find the node indices of the features we want to trace
        feature_node_indices = []
        for layer, feature_idx, pos, is_lorsa in list_of_features:
            if is_lorsa:
                indices = lorsa_activation_matrix.indices()
                mask = (indices[0] == layer) & (indices[1] == pos) & (indices[2] == feature_idx)
                if mask.any():
                    feature_node_idx = torch.where(mask)[0][0]
                    feature_node_indices.append(feature_node_idx.item())
            else:
                indices = clt_activation_matrix.indices()
                mask = (indices[0] == layer) & (indices[1] == pos) & (indices[2] == feature_idx)
                if mask.any():
                    feature_node_idx = torch.where(mask)[0][0] + (lorsa_activation_matrix._nnz() if use_lorsa else 0)
                    feature_node_indices.append(feature_node_idx.item())

        logger.info(f"Feature node indices to trace: {feature_node_indices}")

        # Use feature activations as starting weights (normalized)
        logit_p = (
            feature_activations / feature_activations.sum()
            if feature_activations.sum() > 0
            else torch.ones(n_features, device=feature_activations.device, dtype=feature_activations.dtype)
        )

        if offload:
            offload_handles += offload_modules([state.model.unembed, state.model.embed], offload)

        logit_offset = state.logit_offset
        n_logits = n_features
        total_nodes = logit_offset + n_logits

        max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
        logger.info(f"Will include {max_feature_nodes} of {total_active_feats} feature nodes")

        edge_matrix = torch.zeros(max_feature_nodes + n_features, total_nodes)
        # if use_lorsa:
        lorsa_pattern = torch.zeros(max_feature_nodes + n_features, n_pos)
        z_pattern = torch.zeros(max_feature_nodes + n_features, n_pos)
        # Maps row indices in edge_matrix to original feature/node indices
        # First populated with feature node IDs (dummy), then feature IDs in attribution order
        row_to_node_index = torch.zeros(max_feature_nodes + n_features, dtype=torch.int32)
    else:
        logit_idx, logit_p, logit_vecs = compute_salient_logits(
            logits[0, -1],
            state.model.unembed.W_U,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
        )
        logger.info(f"Selected {len(logit_idx)} logits with cumulative probability {logit_p.sum().item():.4f}")

        if offload:
            offload_handles += offload_modules([state.model.unembed, state.model.embed], offload)

        logit_offset = state.logit_offset
        n_logits = len(logit_idx)
        total_nodes = logit_offset + n_logits

        max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
        logger.info(f"Will include {max_feature_nodes} of {total_active_feats} feature nodes")

        edge_matrix = torch.zeros(max_feature_nodes + n_logits, total_nodes)
        # if use_lorsa:
        lorsa_pattern = torch.zeros(max_feature_nodes + n_logits, n_pos)
        z_pattern = torch.zeros(max_feature_nodes + n_logits, n_pos)
        # Maps row indices in edge_matrix to original feature/node indices
        # First populated with logit node IDs, then feature IDs in attribution order
        row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)
    logger.info(f"Input vectors built in {time.time() - phase_start:.2f}s")

    # Phase 3: logit/feature attribution
    if list_of_features is not None:
        logger.info("Phase 3: Computing feature attributions (dummy logit)")
        phase_start = time.time()
        # Create dummy logit row with feature activations
        for i, (layer, feature_idx, pos, is_lorsa) in enumerate(list_of_features):
            # Find the row index for this feature in the total nodes
            if is_lorsa:
                assert lorsa_activation_matrix is not None
                # For LORSA, find the index in lorsa_activation_matrix
                indices = lorsa_activation_matrix.indices()
                mask = (indices[0] == layer) & (indices[1] == pos) & (indices[2] == feature_idx)
                if mask.any():
                    feature_row_idx = torch.where(mask)[0][0]
                    edge_matrix[i, feature_row_idx] = feature_activations[i].cpu()
            else:
                # For CLT, find the index in clt_activation_matrix
                indices = clt_activation_matrix.indices()
                mask = (indices[0] == layer) & (indices[1] == pos) & (indices[2] == feature_idx)
                if mask.any():
                    feature_row_idx = torch.where(mask)[0][0] + (lorsa_activation_matrix._nnz() if use_lorsa else 0)
                    edge_matrix[i, feature_row_idx] = feature_activations[i].cpu()
            row_to_node_index[i] = logit_offset + i

            # # '''notations begin'''
            # print(f'Feature {i}: layer={layer}, feature_idx={feature_idx}, pos={pos}, is_lorsa={is_lorsa}')
            # print(f'Feature activation value: {feature_activations[i]}')
            # print(f'Feature row index in edge_matrix: {feature_row_idx}')
            # print(f'Edge matrix value set: {edge_matrix[i, feature_row_idx]}')
            # print(f'Logit offset for this feature: {logit_offset + i}')

            # # Debug token sequence
            # input_ids = ensure_tokenized(prompt, model.tokenizer)
            # print(f'Input tokens: {input_ids}')
            # print(f'Token at pos {pos}: {input_ids[pos] if pos < len(input_ids) else "OUT_OF_RANGE"}')
            # print(f'Decoded token at pos {pos}: {model.tokenizer.decode(input_ids[pos]) if pos < len(input_ids) else "OUT_OF_RANGE"}')
            # print("--------------------------------")

            # # Assert that the edge matrix value was set correctly
            assert torch.allclose(edge_matrix[i, feature_row_idx], feature_activations[i].float().cpu(), rtol=1e-6), (
                f"Edge matrix value {edge_matrix[i, feature_row_idx]} != feature activation {feature_activations[i].float().cpu()}"
            )

            # Assert that feature_row_idx is within valid range
            assert 0 <= feature_row_idx < total_active_feats, (
                f"Feature row index {feature_row_idx} out of range [0, {total_active_feats})"
            )

            # Assert that logit offset is correct
            assert row_to_node_index[i] == logit_offset + i, (
                f"Row to node index mismatch: {row_to_node_index[i]} != {logit_offset + i}"
            )

            # '''notations end'''

        logger.info(f"Feature attributions completed in {time.time() - phase_start:.2f}s")
    else:
        logger.info("Phase 3: Computing logit attributions")
        phase_start = time.time()
        for i in range(0, len(logit_idx), batch_size):
            batch = logit_vecs[i : i + batch_size]
            rows = ctx.compute_batch(
                layers=torch.full((batch.shape[0],), 2 * n_layers if use_lorsa else n_layers),
                positions=torch.full((batch.shape[0],), n_pos - 1),
                inject_values=batch,
            )

            # '''notations begin'''
            # bias_attributions = []
            # for param in model._get_requires_grad_bias_params():
            #     try:
            #         attribution = (param[1].data * param[1].grad).sum()
            #         bias_attributions.append(attribution)
            #     except TypeError as e:
            #         pass
            # print(f'bias contribution: {sum(bias_attributions)}')
            # print(f'feature contribution: {rows[0, :total_active_feats].sum()}')
            # print(f'error contribution: {rows[0, total_active_feats: total_active_feats + 2 * n_layers * n_pos].sum()}')
            # print(f'token contribution: {rows[0, total_active_feats + 2 * n_layers * n_pos: logit_offset].sum()}')
            # print(f'logits[0, -1].max() - logits[0, -1].mean(): {logits[0, -1].max() - logits[0, -1].mean()}')
            # print("--------------------------------")
            # if n_logits == 1:
            #     assert torch.allclose(sum(bias_attributions) + rows[0].sum(), logits[0, -1].max() - logits[0, -1].mean(), rtol=1e-3), f"{sum(bias_attributions) + rows[0].sum()} != {logits[0, -1].max() - logits[0, -1].mean()}"
            # assert total_active_feats + (2 * n_layers + 1) * n_pos == rows.shape[1]
            # for param in model._get_requires_grad_bias_params():
            #     param[1].grad = None
            # '''notations end'''

            edge_matrix[i : i + batch.shape[0], :logit_offset] = rows.cpu()
            row_to_node_index[i : i + batch.shape[0]] = torch.arange(i, i + batch.shape[0]) + logit_offset
        logger.info(f"Logit attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 4: feature attribution
    logger.info("Phase 4: Computing feature attributions")

    # DEBUGGING:
    # def idx_to_activation_values(idx: torch.Tensor) -> torch.Tensor:
    #     is_lorsa = idx < len(lorsa_feat_layer)
    #     if is_lorsa.squeeze().item():
    #         layer, feat_idx = lorsa_feat_layer[idx], lorsa_feat_idx[idx]
    #         return lorsa_activation_matrix.values()[idx]
    #     else:
    #         layer, feat_idx = clt_feat_layer[idx - len(lorsa_feat_layer)], clt_feat_idx[idx - len(lorsa_feat_layer)]
    #         print(f'tc activation: {clt_activation_matrix.values()[idx - len(lorsa_feat_layer)]}')
    #         return clt_activation_matrix.values()[idx - len(lorsa_feat_layer)] - model.transcoders[layer].b_E[feat_idx]

    # END OF DEBUGGING:

    phase_start = time.time()
    st = n_logits
    visited = torch.zeros(total_active_feats, dtype=torch.bool)
    n_visited = 0

    pbar = tqdm(total=max_feature_nodes, desc="Feature influence computation")

    while n_visited < max_feature_nodes:
        if max_feature_nodes >= total_active_feats:
            pending = torch.arange(total_active_feats)
        else:
            influences = compute_partial_influences(edge_matrix[:st], logit_p, row_to_node_index[:st])
            feature_rank = torch.argsort(influences[:total_active_feats], descending=True).cpu()
            queue_size = min(update_interval * batch_size, max_feature_nodes - n_visited)
            pending = feature_rank[~visited[feature_rank]][:queue_size]

        queue = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
        if len(replicas) == 1:
            feature_results = [primary.compute_feature_rows(idx_batch, retain_graph=True) for idx_batch in queue]
        else:
            with ThreadPoolExecutor(max_workers=len(replicas)) as executor:
                futures = [
                    executor.submit(replicas[i % len(replicas)].compute_feature_rows, idx_batch, retain_graph=True)
                    for i, idx_batch in enumerate(queue)
                ]
                feature_results = [future.result() for future in futures]

        for result in feature_results:
            idx_batch = result.idx_batch
            n_visited += len(idx_batch)
            n_rows = min(idx_batch.shape[0], result.rows.shape[0])
            end = st + n_rows
            edge_matrix[st:end, :logit_offset] = result.rows[:n_rows]
            lorsa_pattern[st:end, :] = result.lorsa_pattern[:n_rows]
            z_pattern[st:end, :] = result.z_pattern[:n_rows]
            row_to_node_index[st:end] = idx_batch[:n_rows]
            visited[idx_batch[:n_rows]] = True
            st = end
            pbar.update(int(n_rows))
            if progress_callback:
                progress_callback(n_visited, max_feature_nodes, "Feature influence computation")

    pbar.close()
    logger.info(f"Feature attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 5: packaging graph
    selected_features = torch.where(visited)[0]
    if max_feature_nodes < total_active_feats:
        non_feature_nodes = torch.arange(total_active_feats, total_nodes)
        col_read = torch.cat([selected_features, non_feature_nodes])
        edge_matrix = edge_matrix[:, col_read]

    # ***** New Phase Begin *****
    # Phase 6: attribute attention scores
    # trace attention scores for every visited lorsa feature
    if use_lorsa and qk_tracing_topk > 0:
        assert lorsa_activation_matrix is not None
        selected_lorsa_feature = torch.where(visited[: lorsa_activation_matrix._nnz()])[0]
        qk_tracing_results = {}
        if len(replicas) == 1:
            tracing_results = [primary.compute_qk_trace(int(idx.item()), qk_tracing_topk) for idx in selected_lorsa_feature]
        else:
            with ThreadPoolExecutor(max_workers=len(replicas)) as executor:
                futures = [
                    executor.submit(replicas[i % len(replicas)].compute_qk_trace, int(idx.item()), qk_tracing_topk)
                    for i, idx in enumerate(selected_lorsa_feature)
                ]
                tracing_results = []
                for i, future in enumerate(
                    tqdm(futures, desc="Computing attention scores attribution")
                ):
                    tracing_results.append(future.result())
                    if progress_callback:
                        progress_callback(i + 1, len(selected_lorsa_feature), "Computing attention scores attribution")

        for idx, result in tracing_results:
            qk_tracing_results[idx] = result
        if progress_callback and len(replicas) == 1:
            for i in range(len(tracing_results)):
                progress_callback(i + 1, len(selected_lorsa_feature), "Computing attention scores attribution")
    else:
        qk_tracing_results = None

    # ***** New Phase End *****

    # sort rows such that features are in order
    edge_matrix = edge_matrix[row_to_node_index.argsort()]
    # if use_lorsa:
    lorsa_pattern = lorsa_pattern[row_to_node_index.argsort()]
    z_pattern = z_pattern[row_to_node_index.argsort()]
    final_node_count = edge_matrix.shape[1]
    full_edge_matrix = torch.zeros(final_node_count, final_node_count)
    full_edge_matrix[:max_feature_nodes] = edge_matrix[:max_feature_nodes]
    full_edge_matrix[-n_logits:] = edge_matrix[max_feature_nodes:]

    # modified for list_of_features
    graph = Graph(
        input_string=model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_tokens=torch.tensor(list_of_features) if list_of_features is not None else logit_idx,
        logit_probabilities=feature_activations if list_of_features is not None else logit_p,
        lorsa_active_features=lorsa_activation_matrix.indices().T if use_lorsa else None,
        lorsa_activation_values=lorsa_activation_matrix.values() if use_lorsa else None,
        clt_active_features=clt_activation_matrix.indices().T,
        clt_activation_values=clt_activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edge_matrix,
        qk_tracing_results=qk_tracing_results,
        cfg=model.cfg,
        sae_series=sae_series,
        use_lorsa=use_lorsa,
        lorsa_pattern=lorsa_pattern,
        z_pattern=z_pattern,
    )

    total_time = time.time() - start_time
    logger.info(f"Attribution completed in {total_time:.2f}s")

    for replica in replicas:
        replica.cleanup()

    return graph
