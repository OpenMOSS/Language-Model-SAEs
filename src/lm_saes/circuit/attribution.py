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
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from einops import einsum
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from lm_saes.clt import CrossLayerTranscoder
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
            self._resid_activations[last_layer].backward(
                gradient=torch.zeros_like(self._resid_activations[last_layer]),
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
    progress_callback: Optional[Callable[[float, float, str], None]] = None,
):
    start_time = time.time()
    # Phase 0: precompute
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.time()
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
    # we do not run forward passes in setup_attribution. Just assign some zero-inited tensors and set hooks for them
    # they only have informative values set in Phase 1: forward pass later

    lorsa_decoder_vecs = select_scaled_decoder_vecs_lorsa(lorsa_activation_matrix, model.lorsas) if use_lorsa else None
    lorsa_encoder_rows, lorsa_attention_patterns, z_attention_patterns = (
        select_encoder_rows_lorsa(lorsa_activation_matrix, lorsa_attention_pattern, z_attention_pattern, model.lorsas)
        if use_lorsa
        else (None, None, None)
    )

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
    logger.info(f"Precomputation completed in {time.time() - phase_start:.2f}s")
    if use_lorsa:
        logger.info(f"Found {lorsa_activation_matrix._nnz() + clt_activation_matrix._nnz()} active features")

    if offload:
        offload_handles += offload_modules(model.transcoders, offload)

    # Phase 1: forward pass
    logger.info("Phase 1: Running forward pass")
    phase_start = time.time()
    with ctx.install_hooks(model):
        residual = model.forward(input_ids.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
        ctx._resid_activations[-1] = model.ln_final(residual)
    logger.info(f"Forward pass completed in {time.time() - phase_start:.2f}s")

    if offload:
        offload_handles += offload_modules(
            [block.mlp for block in model.blocks] + [block.attn for block in model.blocks],
            offload,
        )

    # Phase 2: build input vector list
    logger.info("Phase 2: Building input vectors")
    phase_start = time.time()
    n_layers, n_pos, _ = clt_activation_matrix.shape
    if use_lorsa:
        total_active_feats = lorsa_activation_matrix._nnz() + clt_activation_matrix._nnz()
    else:
        total_active_feats = clt_activation_matrix._nnz()

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
            offload_handles += offload_modules([model.unembed, model.embed], offload)

        if use_lorsa:
            logit_offset = total_active_feats + 2 * n_layers * n_pos + n_pos
        else:
            logit_offset = total_active_feats + n_layers * n_pos + n_pos
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
            model.unembed.W_U,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
        )
        logger.info(f"Selected {len(logit_idx)} logits with cumulative probability {logit_p.sum().item():.4f}")

        if offload:
            offload_handles += offload_modules([model.unembed, model.embed], offload)

        if use_lorsa:
            logit_offset = total_active_feats + 2 * n_layers * n_pos + n_pos
        else:
            logit_offset = total_active_feats + n_layers * n_pos + n_pos
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
            assert torch.allclose(edge_matrix[i, feature_row_idx], feature_activations[i].cpu(), rtol=1e-6), (
                f"Edge matrix value {edge_matrix[i, feature_row_idx]} != feature activation {feature_activations[i].cpu()}"
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

    lorsa_feat_layer, lorsa_feat_pos = lorsa_activation_matrix.indices()[:2] if use_lorsa else (None, None)
    clt_feat_layer, clt_feat_pos = clt_activation_matrix.indices()[:2]

    lorsa_feat_layer, lorsa_feat_pos, lorsa_feat_idx = (
        lorsa_activation_matrix.indices() if use_lorsa else (None, None, None)
    )
    clt_feat_layer, clt_feat_pos, clt_feat_idx = clt_activation_matrix.indices()

    def idx_to_layer(idx: torch.Tensor) -> torch.Tensor:
        if use_lorsa:
            is_lorsa = idx < len(lorsa_feat_layer)
            return torch.where(
                is_lorsa.to(lorsa_feat_layer.device),
                2 * lorsa_feat_layer[idx * is_lorsa],
                2 * clt_feat_layer[(idx - len(lorsa_feat_layer)) * ~is_lorsa] + 1,
            )
        else:
            return clt_feat_layer[idx]

    def idx_to_pos(idx: torch.Tensor) -> torch.Tensor:
        if use_lorsa:
            is_lorsa = idx < len(lorsa_feat_layer)
            return torch.where(
                is_lorsa.to(lorsa_feat_pos.device),
                lorsa_feat_pos[idx * is_lorsa],
                clt_feat_pos[(idx - len(lorsa_feat_layer)) * ~is_lorsa],
            )
        else:
            return clt_feat_pos[idx]

    def idx_to_encoder_rows(idx: torch.Tensor) -> torch.Tensor:
        if use_lorsa:
            is_lorsa = idx < len(lorsa_feat_layer)
            return torch.where(
                is_lorsa.to(lorsa_encoder_rows.device)[:, None],
                lorsa_encoder_rows[idx * is_lorsa],
                clt_encoder_rows[(idx - len(lorsa_feat_layer)) * ~is_lorsa],
            )
        else:
            return clt_encoder_rows[idx]

    def idx_to_pattern(idx: torch.Tensor) -> torch.Tensor:
        if use_lorsa:
            is_lorsa = idx < len(lorsa_feat_layer)
            return torch.where(
                is_lorsa.to(lorsa_attention_patterns.device)[:, None],
                lorsa_attention_patterns[idx * is_lorsa],
                torch.nn.functional.one_hot(clt_feat_pos[(idx - len(lorsa_feat_layer)) * ~is_lorsa], num_classes=n_pos),
            )
        else:
            return torch.nn.functional.one_hot(clt_feat_pos[idx], num_classes=n_pos)

    def idx_to_z_pattern(idx: torch.Tensor) -> torch.Tensor:
        if use_lorsa:
            is_lorsa = idx < len(lorsa_feat_layer)
            return torch.where(
                is_lorsa.to(z_attention_patterns.device)[:, None],
                z_attention_patterns[idx * is_lorsa],
                torch.nn.functional.one_hot(clt_feat_pos[(idx - len(lorsa_feat_layer)) * ~is_lorsa], num_classes=n_pos),
            )
        else:
            return torch.nn.functional.one_hot(clt_feat_pos[idx], num_classes=n_pos)

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

        for idx_batch in queue:
            n_visited += len(idx_batch)

            rows = ctx.compute_batch(
                layers=idx_to_layer(idx_batch),
                positions=idx_to_pos(idx_batch),
                inject_values=idx_to_encoder_rows(idx_batch),
                attention_patterns=idx_to_pattern(idx_batch),
                retain_graph=n_visited < max_feature_nodes or qk_tracing_topk > 0,
            )

            # DEBUGGING:

            # Print feature information before attention patterns
            # for i, idx in enumerate(idx_batch):
            #     layer = idx_to_layer(idx.unsqueeze(0)).item()
            #     pos = idx_to_pos(idx.unsqueeze(0)).item()
            #     is_lorsa = idx < len(lorsa_feat_layer) if use_lorsa else False

            # print(f'attention_patterns {idx_batch} {idx_to_pattern(idx_batch)}')
            # bias_attributions = []
            # for param in model._get_requires_grad_bias_params():
            #     try:
            #         attribution = (param[1].data * param[1].grad).sum()
            #         bias_attributions.append(attribution)
            #     except TypeError as e:
            #         pass
            # print(f'bias_contribution: {sum(bias_attributions)}')
            # print(f'feature_contribution: {rows[0, :total_active_feats].sum()}')
            # print(f'error_contribution: {rows[0, total_active_feats: total_active_feats + 2 * n_layers * n_pos].sum()}')
            # print(f'token_contribution: {rows[0, total_active_feats + 2 * n_layers * n_pos: logit_offset].sum()}')
            # print(f'overall_activation: {idx_to_activation_values(idx_batch)}')
            # print(idx_batch.squeeze().item() < len(lorsa_feat_layer), torch.allclose(idx_to_activation_values(idx_batch), rows[0, :logit_offset].sum() + sum(bias_attributions), rtol=1e-1))
            # print("--------------------------------")
            # for param in model._get_requires_grad_bias_params():
            #     param[1].grad = None

            # END OF DEBUGGING:

            end = min(st + batch_size, st + rows.shape[0])
            edge_matrix[st:end, :logit_offset] = rows.cpu()
            lorsa_pattern[st:end, :] = idx_to_pattern(idx_batch)
            z_pattern[st:end, :] = idx_to_z_pattern(idx_batch)
            row_to_node_index[st:end] = idx_batch
            visited[idx_batch] = True
            st = end
            pbar.update(len(idx_batch))
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
    if qk_tracing_topk > 0:

        def idx_to_qk_idx(idx: torch.Tensor) -> torch.Tensor:
            ov_idx = lorsa_activation_matrix.indices()[2][idx]
            ov_group_sizes = torch.tensor(
                [model.lorsas[i].cfg.ov_group_size for i in range(model.cfg.n_layers)], device=ov_idx.device
            )
            return ov_idx // ov_group_sizes[lorsa_activation_matrix.indices()[0][idx]]

        selected_lorsa_feature = torch.where(visited[: lorsa_activation_matrix._nnz()])[0]
        selected_lorsa_feature_layer = lorsa_activation_matrix.indices()[0][selected_lorsa_feature]
        selected_lorsa_feature_qpos = idx_to_pos(selected_lorsa_feature)
        # currently we are only interested in the most prominent z pattern contributor for each lorsa feature
        # these k_poses are not even necessarily ones with the largest attention pattern due to effect from V
        # This is not enough for explaining the attention scores as a whole.
        # For now we only use qk tracing to get some sense of how lorsa attention pattern forms
        # In practice this should tell us something
        selected_lorsa_feature_kpos = idx_to_z_pattern(selected_lorsa_feature).argmax(dim=-1)
        selected_lorsa_feature_qk_idx = idx_to_qk_idx(selected_lorsa_feature)

        qk_tracing_results = {}
        for i, idx in enumerate(tqdm(selected_lorsa_feature, desc="Computing attention scores attribution")):
            qk_tracing_results[idx.item()] = compute_attn_scores_attribution(
                model,
                lorsa_activation_matrix,
                clt_activation_matrix,
                selected_lorsa_feature_layer[i],
                selected_lorsa_feature_qpos[i],
                selected_lorsa_feature_kpos[i],
                selected_lorsa_feature_qk_idx[i],
                token_vecs,
                error_vecs,
                input_ids,
                qk_tracing_topk,
            )
            if progress_callback:
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

    return graph
