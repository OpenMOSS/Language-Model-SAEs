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
import logging
import time
import weakref
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union, Dict
from lm_saes.sae import SparseAutoEncoder
import numpy as np
import torch
from einops import einsum
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from .graph import Graph
from .replacement_lc0_model import ReplacementModel
from .utils.disk_offload import offload_modules
from ..utils.logging import get_distributed_logger

from .leela_board import *

logger = get_distributed_logger("attribution")

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
        # lorsa_activation_matrix: torch.sparse.Tensor,
        tc_activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor, 
        # lorsa_decoder_vecs: torch.Tensor,
        tc_decoder_vecs: torch.Tensor,
        # attn_output_hook: str,
        mlp_output_hook: str,
    ) -> None:
        # assert lorsa_activation_matrix.shape[:-1] == tc_activation_matrix.shape[:-1], "LORSAs and CLTs must have the same shape"
        n_layers, n_pos, _ = tc_activation_matrix.shape

        # Forward-pass cache
        # # L0Ainput, L0Minput, ... L-1Ainput, L-1Minput, pre_unembed
        # L0Minput, L1Minput, ... L-1Minput, policy_head_input for transcoder only tracing
        self._resid_activations: List[torch.Tensor | None] = [None] * (n_layers + 1)
        # 添加policy head的q和k activations缓存
        self._policy_q_activations: torch.Tensor | None = None
        self._policy_k_activations: torch.Tensor | None = None
        self._batch_buffer: torch.Tensor | None = None # (row_size, batch_size, 1)
        self.n_layers: int = n_layers

        # Assemble all backward hooks up-front
        self._attribution_hooks = self._make_attribution_hooks(
            # lorsa_activation_matrix,
            tc_activation_matrix,
            error_vectors,
            token_vectors,
            # lorsa_decoder_vecs,
            tc_decoder_vecs,
            # attn_output_hook,
            mlp_output_hook
        )
        
        total_active_feats = tc_activation_matrix._nnz()
        # total_active_feats + error_vectors + token_vectors
        self._row_size: int = total_active_feats + n_layers * n_pos + n_pos  # + logits later

    # def _caching_hooks(self, attn_input_hook: str, mlp_input_hook: str) -> List[Tuple[str, Callable]]:
    #     """Return hooks that store residual activations layer-by-layer."""

    #     proxy = weakref.proxy(self)

    #     def _cache(acts: torch.Tensor, hook: HookPoint, *, index: int) -> torch.Tensor:
    #         proxy._resid_activations[index] = acts
    #         return acts

    #     hooks = []
    #     for layer in range(self.n_layers):
    #         hooks.append((f"blocks.{layer}.{attn_input_hook}", partial(_cache, index=layer * 2)))
    #         hooks.append((f"blocks.{layer}.{mlp_input_hook}", partial(_cache, index=layer * 2 + 1)))
    #     hooks.append(("unembed.hook_pre", partial(_cache, index=2 * self.n_layers)))

    #     return hooks

    def _caching_hooks(self, mlp_input_hook: str) -> List[Tuple[str, Callable]]:
        """Return hooks that store residual activations layer-by-layer."""

        proxy = weakref.proxy(self)

        def _cache(acts: torch.Tensor, hook: HookPoint, *, index: int) -> torch.Tensor:
            proxy._resid_activations[index] = acts
            print(f"DEBUG: _cache: {acts.shape}")
            return acts

        def _cache_q(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            proxy._policy_q_activations = acts
            print(f"DEBUG: _cache_q: {acts.shape}")
            return acts

        def _cache_k(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            proxy._policy_k_activations = acts
            print(f"DEBUG: _cache_k: {acts.shape}")
            return acts

        hooks = []
        for layer in range(self.n_layers):
            hooks.append((f"blocks.{layer}.{mlp_input_hook}", partial(_cache, index=layer)))
        
        hooks.append(("policy_head.hook_pre", partial(_cache, index=self.n_layers)))
        # 添加policy head的q和k缓存hooks
        hooks.append(("policy_head.hook_q", _cache_q))
        hooks.append(("policy_head.hook_k", _cache_k))

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
            print(f"DEBUG: Hook '{hook_name}' executed")
            print(f"DEBUG: grads shape: {grads.shape}")
            print(f"DEBUG: grads[0][6]: {grads[0][6].flatten()[:5].tolist()}")
            print(f"DEBUG: grads[0][46]: {grads[0][46].flatten()[:5].tolist()}")
            print(f"DEBUG: output_vecs shape: {output_vecs.shape}")
            print(f"DEBUG: {output_vecs.flatten()[:10].tolist() = }")
            print(f"DEBUG: write_index: {write_index}")
            print(f"DEBUG: read_index: {read_index}")
            
            # 计算einsum前的形状信息
            grads_read = grads.to(output_vecs.dtype)[read_index] #[1, 2240, 768]
            print(f"DEBUG: grads[read_index] shape: {grads_read.shape}")
            
            # 执行einsum计算
            result = einsum(
                grads_read,
                output_vecs,
                "batch position d_model, position d_model -> position batch",
            )
            print(f"DEBUG: grads_read.shape = {grads_read.shape}")
            print(f"DEBUG: output_vecs.shape = {output_vecs.shape}")
            print(f"DEBUG: einsum result shape: {result.shape}")
            print(f"DEBUG: einsum result sum: {result.sum().item()}")
            
            # 写入缓冲区
            proxy._batch_buffer[write_index] += result
            print(f"DEBUG: Updated _batch_buffer[{write_index}]")
            print(f"DEBUG: _batch_buffer[{write_index}] sum: {proxy._batch_buffer[write_index].sum().item()}")
            print("---")

        return hook_name, _hook_fn


    def _make_attribution_hooks(
        self,
        # lorsa_activation_matrix: torch.sparse.Tensor,
        tc_activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        # lorsa_decoder_vecs: torch.Tensor,
        tc_decoder_vecs: torch.Tensor,
        # attn_output_hook: str,
        mlp_output_hook: str,
    ) -> List[Tuple[str, Callable]]:
        """
        Create the complete backward-hook for computing attribution scores.
        """
        _, n_pos, _ = tc_activation_matrix.shape
        print(f"DEBUG_make_attribution_hooks: tc_activation_matrix.shape = {tc_activation_matrix.shape}")
        print(f"DEBUG_make_attribution_hooks: tc_activation_matrix.nnz() = {tc_activation_matrix._nnz()}")
        print(f"DEBUG_make_attribution_hooks: error_vectors.shape = {error_vectors.shape}")
        print(f"DEBUG_make_attribution_hooks: token_vectors.shape = {token_vectors.shape}")
        print(f"DEBUG_make_attribution_hooks: token_vectors = {token_vectors}")
        print(f"DEBUG_make_attribution_hooks: tc_decoder_vecs.shape = {tc_decoder_vecs.shape}")
        print(f"DEBUG_make_attribution_hooks: tc_decoder_vecs[31360] = {tc_decoder_vecs[31360]}")
        print(f"DEBUG_make_attribution_hooks: mlp_output_hook = {mlp_output_hook}")
        
        tc_error_vectors = error_vectors

        # Token-embedding nodes
        # tc_offset + mlp_error_offset (no lorsa for LC0 model)
        token_offset = tc_activation_matrix._nnz() + self.n_layers * n_pos
        print(f"{token_offset = }")
        token_hook = [
            self._compute_score_hook(
                "hook_embed",
                token_vectors,
                write_index=np.s_[token_offset : token_offset + n_pos],
            )
        ]
        return token_hook + self._make_attribution_hooks_tc(
            tc_activation_matrix,
            tc_error_vectors,
            tc_decoder_vecs,
            mlp_output_hook,
        ) 
        
        
        # return self._make_attribution_hooks_clt(
        #     tc_activation_matrix,
        #     tc_error_vectors,
        #     tc_decoder_vecs,
        #     mlp_output_hook,
        #     # lorsa_offset=lorsa_activation_matrix._nnz()
        # ) + self._make_attribution_hooks_lorsa( 
        #     lorsa_activation_matrix,
        #     lorsa_error_vectors,
        #     lorsa_decoder_vecs,
        #     attn_output_hook,
        #     clt_offset=tc_activation_matrix._nnz()
        # ) + token_hook
    
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

    def _make_attribution_hooks_tc(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        mlp_output_hook: str,
        # lorsa_offset: int,
    ) -> List[Tuple[str, Callable]]:
        """
        Create attribution hooks for single-layer transcoders.
        activation_matrix:
            size (n_layers, n_pos, n_features)
            indices: (3, n_active_features)
            values: (n_active_features,)
        error_vectors:
            size (n_layers, n_pos, d_model)
        decoder_vecs:
            size (n_active_features, d_model) - for single-layer transcoders
        """
        n_layers, n_pos, _ = activation_matrix.shape
        nnz_layers, nnz_positions, _ = activation_matrix.indices()

        # Map each layer → slice in flattened active-feature list
        if activation_matrix._nnz() == 0:
            return []  # Return empty list if no features are active
            
        _, counts = torch.unique_consecutive(nnz_layers, return_counts=True)
        edges = [0] + counts.cumsum(0).tolist()
        layer_spans = list(zip(edges[:-1], edges[1:]))

        # Simple assertion: decoder_vecs should match total active features
        assert edges[-1] == activation_matrix._nnz(), f'got {edges[-1]} but expected {activation_matrix._nnz()}'
        assert decoder_vecs.size(0) == activation_matrix._nnz(), f'got {decoder_vecs.size(0)} but expected {activation_matrix._nnz()}'

        # Feature nodes
        feature_hooks = []
        for layer, (start, end) in enumerate(layer_spans):
            if start != end:
                print(f"blocks.{layer}.{mlp_output_hook}")
                hook = self._compute_score_hook(
                    f"blocks.{layer}.{mlp_output_hook}",
                    decoder_vecs[start:end],
                    write_index=np.s_[start:end],
                    read_index=np.s_[:, nnz_positions[start:end]],
                )
                feature_hooks.append(hook)

        # Error nodes
        def error_offset(layer: int) -> int:
            # tc_offset + layer_offset
            return activation_matrix._nnz() + layer * n_pos
        
        error_hooks = []
        for layer in range(n_layers):
            hook = self._compute_score_hook(
                f"blocks.{layer}.{mlp_output_hook}",
                error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
            )
            error_hooks.append(hook)

        return feature_hooks + error_hooks

    # def _make_attribution_hooks_clt(
    #     self,
    #     activation_matrix: torch.sparse.Tensor,
    #     error_vectors: torch.Tensor,
    #     decoder_vecs: torch.Tensor,
    #     mlp_output_hook: str,
    #     lorsa_offset: int,
    # ) -> List[Tuple[str, Callable]]:
    #     """
    #     Create the complete backward-hook for computing attribution scores.
    #     activation_matrix:
    #         size (n_layers, n_pos, n_features)
    #         indices: (3, n_active_features)
    #         values: (n_active_features,)
    #     error_vectors:
    #         size (n_layers, n_pos, d_model)
    #     token_vectors:
    #         size (n_pos, d_model)
    #     decoder_vecs:
    #         size ((\sum_{i=0}^{n_layers} \sum_{j=0}^{i} n_active_features_layer_j), d_model)
    #     """
    #     n_layers, n_pos, _ = activation_matrix.shape
    #     nnz_layers, nnz_positions, _ = activation_matrix.indices()

    #     # Map each layer → slice in flattened active-feature list
    #     _, counts = torch.unique_consecutive(nnz_layers, return_counts=True)  # active features per layer
    #     edges = counts.cumsum(0)  # n_layers
    #     decoder_layer_spans = [0] + edges.cumsum(0).tolist()
    #     assert edges[-1] == activation_matrix._nnz(), f'got {edges[-1]} but expected {activation_matrix._nnz()}'
    #     assert decoder_layer_spans[-1] == decoder_vecs.size(0), f'got {len(decoder_layer_spans)} but expected {decoder_vecs.size(0)}'
    #     decoder_layer_spans = [slice(start, end) for start, end in zip(decoder_layer_spans[:-1], decoder_layer_spans[1:])]

    #     # Feature nodes
    #     feature_hooks: list[Tuple[str, Callable[..., Any]]] = [
    #         self._compute_score_hook(
    #             f"blocks.{layer}.{mlp_output_hook}",
    #             decoder_vecs[decoder_layer_spans[layer]],
    #             write_index=np.s_[lorsa_offset: lorsa_offset + edges[layer]],
    #             read_index=np.s_[:, nnz_positions[:edges[layer]]],
    #         )
    #         for layer in range(n_layers)
    #     ]

    #     # Error nodes
    #     def error_offset(layer: int) -> int:  # starting row for this layer
    #         # lorsa_offset + clt_offset + attn_error_offset + layer_offset
    #         return lorsa_offset + activation_matrix._nnz() + self.n_layers * n_pos + layer * n_pos
        
    #     error_hooks = [
    #         self._compute_score_hook(
    #             f"blocks.{layer}.{mlp_output_hook}",
    #             error_vectors[layer],
    #             write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
    #         )
    #         for layer in range(n_layers)
    #     ]

    #     return feature_hooks + error_hooks

    @contextlib.contextmanager
    def install_hooks(self, model: "ReplacementModel"):
        """Context manager instruments the hooks for the forward and backward passes."""
        with model.hooks(
            fwd_hooks=self._caching_hooks(model.mlp_input_hook),
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
        print(f"DEBUG: batch_size = {batch_size}")
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=inject_values.device,
        )

        # Custom gradient injection (per-layer registration)
        batch_idx = torch.arange(len(layers), device=layers.device)
        
        def _inject(grads, *, batch_indices, pos_indices, patterns, values):
            if batch_indices.max() >= grads.shape[0]:
                raise IndexError(f"Batch indices max ({batch_indices.max()}) >= grads batch size ({grads.shape[0]})")
            if pos_indices.max() >= grads.shape[1]:
                raise IndexError(f"Position indices max ({pos_indices.max()}) >= grads seq length ({grads.shape[1]})")
            
            grads_out = grads.clone().to(values.dtype)
            
            if patterns is not None:
                if patterns.shape[1] > grads.shape[1]:
                    raise IndexError(f"Patterns seq_len ({patterns.shape[1]}) > grads seq_len ({grads.shape[1]})")
                
                distributed_values = values[:, None, :] * patterns[:, :, None]
                grads_out.index_put_((batch_indices,), distributed_values)
            else:
                grads_out.index_put_((batch_indices, pos_indices), values)
            
            return grads_out.to(grads.dtype)

        handles = []
        layers_in_batch = layers.unique().tolist()

        for layer in layers_in_batch:
            mask = layers == layer
            if not mask.any():
                continue
            
            if int(layer) >= len(self._resid_activations): # len = 15, resid of every layer
                raise IndexError(f"Layer {layer} out of range")
            
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
            gradient = torch.zeros_like(self._resid_activations[last_layer])
            
            self._resid_activations[last_layer].backward(
                gradient=gradient,
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        result = buf.T[: len(layers)]
        return result
    
    def compute_start_end_batch_lc0(
        self,
        layers: torch.Tensor,
        move_positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """Return attribution rows for start and end positions of moves in single backward pass.

        This function performs one backward pass with gradients injected at both
        start and end positions simultaneously for each move pair.

        Args:
            layers: 1-D tensor of layer indices for the source nodes.
            move_positions: `(batch, 2)` tensor where move_positions[i, 0] is start pos
                and move_positions[i, 1] is end pos for the i-th move.
            inject_values: `(batch, seq_len, d_model)` tensor where we extract
                start and end position values for injection.

        Returns:
            torch.Tensor: `(batch, row_size)` matrix where each row corresponds to 
                the combined attribution of one start-end move pair.
        """
        
        for resid_activation in self._resid_activations:
            assert resid_activation is not None, "Residual activations are not cached"

        k_batch = move_positions.shape[0]
        device = inject_values.device
        
        # Ensure all tensors are on the same device
        start_pos = move_positions[:, 0].to(dtype=torch.long, device=device)
        end_pos = move_positions[:, 1].to(dtype=torch.long, device=device)
        layers = layers.to(device=device)

        batch_size = self._resid_activations[0].shape[0]
        print(f"DEBUG: batch_size = {batch_size}")
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=device,
        )

        # batch indices correspond to grads batch dim; grads batch size is 1 here
        batch_idx = torch.arange(len(layers), device=layers.device)

        def _inject_move_pair(grads, *, batch_indices, start_positions, end_positions, start_values, end_values):
            """Inject both start and end gradients for each move pair simultaneously"""
            grads_out = grads.clone().to(start_values.dtype)
            
            # Inject start positions
            grads_out.index_put_((batch_indices, start_positions), start_values)
            # Inject end positions (additive for same batch if positions overlap)
            grads_out.index_put_((batch_indices, end_positions), end_values, accumulate=True)
            
            return grads_out.to(grads.dtype)

        handles = []
        layers_in_batch = layers.unique().tolist()

        for layer in layers_in_batch:
            mask = layers == layer
            if not mask.any():
                continue
            
            # Extract start and end injection values for this layer
            layer_start_inject = torch.stack([
                inject_values[i, start_pos[i], :] for i in range(k_batch) if mask[i]
            ]) if mask.any() else torch.empty(0, inject_values.shape[-1], device=device)
            
            layer_end_inject = torch.stack([
                inject_values[i, end_pos[i], :] for i in range(k_batch) if mask[i]
            ]) if mask.any() else torch.empty(0, inject_values.shape[-1], device=device)
            
            if layer_start_inject.shape[0] > 0:  # Only register if there are items for this layer
                fn = partial(
                    _inject_move_pair,
                    batch_indices=batch_idx[mask],
                    start_positions=start_pos[mask],
                    end_positions=end_pos[mask],
                    start_values=layer_start_inject,
                    end_values=layer_end_inject,
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
        return buf.T[:k_batch]  # Return k_batch rows, one per move pair


# @torch.no_grad()
def compute_salient_logits_for_lc0(
    fen: str,
    logits: torch.Tensor,
    unembed_proj: torch.Tensor = None,  # 改为可选
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    model=None,
    residual_input=None,
    demean: bool = True,
    move_idx: int = None,  # 新增参数：指定要处理的move索引
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算LC0模型中显著的logits，并返回对应的move位置
    
    Args:
        fen: FEN字符串，表示当前棋盘状态
        logits: 策略logits
        unembed_proj: 可选的unembed投影矩阵
        max_n_logits: 最大选择的logits数量
        desired_logit_prob: 期望的累积概率阈值
        model: LC0模型（用于计算Jacobian）
        residual_input: 残差输入（用于计算Jacobian）
        demean: 是否进行去均值化操作，默认为True
        move_idx: 指定要处理的move索引，如果提供则直接处理该索引，忽略其他参数
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            * top_idx - 选中的logit索引，形状为(k,)
            * top_p - 对应的概率值，形状为(k,)
            * demeaned_vecs - 向量，形状为(k, seq_len, d_model)，如果demean=True则去均值化，否则为原始值
            * move_positions - 对应的move位置，形状为(k, 2)，每行包含[起点位置, 终点位置]
    """
    
    lboard = LeelaBoard.from_fen(fen)
    
    
    if logits.numel() == 0:
        raise ValueError("Input logits tensor is empty")
    
    # 如果提供了model和residual_input，计算Jacobian矩阵作为等效unembed_proj
    if unembed_proj is None and (model is None or residual_input is None):
        raise ValueError("Either unembed_proj or (model + residual_input) must be provided")

    if unembed_proj is not None and model is not None:
        logger.warning("Both unembed_proj and model provided, using model for Jacobian calculation")

    # 确保logits是1D张量
    if logits.dim() > 1:
        logits = logits.flatten()
    
    # 如果指定了move_idx，直接处理该索引
    if move_idx is not None:
        if move_idx < 0 or move_idx >= logits.size(0):
            raise ValueError(f"move_idx {move_idx} 超出logits范围 [0, {logits.size(0)-1}]")
        
        top_idx = torch.tensor([move_idx], device=logits.device)
        probs = torch.softmax(logits, dim=-1)
        top_p = probs[move_idx].unsqueeze(0)
    else:
        # 原有的top logits选择逻辑
        # 检查max_n_logits是否超过logits长度
        actual_max_logits = min(max_n_logits, logits.size(0))
        
        probs = torch.softmax(logits, dim=-1)
        top_p, top_idx = torch.topk(probs, actual_max_logits)
        cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
        top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]
    

    # 计算选出的logits对应的move位置
    move_positions = []
    for idx in top_idx:
        try:
            uci_move = lboard.idx2uci(idx.item())
            positions = lboard.uci_to_positions(uci_move)
            move_positions.append(positions)
        except Exception as e:
            # 如果无法获取move，使用默认值
            logger.warning(f"无法获取索引 {idx.item()} 对应的move位置: {e}")
            move_positions.append(torch.tensor([0, 0]))
    
    move_positions_tensor = torch.stack(move_positions)

    if model is not None and residual_input is not None and hasattr(model, 'policy_head'):
        device = residual_input.device
        d_model = residual_input.shape[-1]
        
        # 确保residual_input需要梯度
        # if not residual_input.requires_grad:
        #     residual_input = residual_input.detach().requires_grad_(True)
        residual_input = residual_input.detach().requires_grad_(True)
        # 前向传播获取policy logits
        policy_logits = model.policy_head(residual_input)
        
        # 计算选定logits的Jacobian矩阵 - 对所有位置求导
        batch_size, seq_len = residual_input.shape[:2]
        policy_dim = policy_logits.shape[-1]
        
        full_jacobian_matrix = torch.zeros(policy_dim, seq_len, d_model, device=device)
        
        for i in range(policy_dim):
            
            if residual_input.grad is not None:
                residual_input.grad.zero_()

            policy_logits[0, i].backward(retain_graph=True)
            if residual_input.grad is not None:
                # residual_input.grad shape: (batch_size, seq_len, d_model)
                grad = residual_input.grad[0, :, :].clone()  # shape: (seq_len, d_model)
                full_jacobian_matrix[i, :, :] = grad
                
        mean_jacobian = full_jacobian_matrix.mean(dim=0, keepdim=True)  # (1, seq_len, d_model)
        
        selected_jacobian_matrix = full_jacobian_matrix[top_idx]  # (k, seq_len, d_model)

        unembed_proj = selected_jacobian_matrix[:, -1, :].T.detach()  # shape: (d_model, k)
    
        if demean:
            result_matrix = selected_jacobian_matrix - mean_jacobian  # (k, seq_len, d_model)
        else:
            # 不进行去均值化，直接使用原始值
            result_matrix = selected_jacobian_matrix
        
        # 返回选中的logit索引、概率、Jacobian矩阵和move位置
        print(f"{top_idx = }")
        return top_idx, top_p, result_matrix.detach(), move_positions_tensor
    
    elif unembed_proj is not None:
        # 使用现有的unembed_proj计算
        cols = unembed_proj[:, top_idx]
        if demean:
            result = cols - unembed_proj.mean(dim=-1, keepdim=True)
        else:
            result = cols
        return top_idx, top_p, result.T, move_positions_tensor
    else:
        raise ValueError("Neither valid model nor unembed_proj provided")


@torch.no_grad()  # modified
def select_scaled_decoder_vecs_tc(
    activations: torch.sparse.Tensor,
    transcoders: Dict[int, SparseAutoEncoder]
) -> torch.Tensor:
    """Return decoder rows for **active** features only.

    The return value is already scaled by the feature activation, making it
    suitable as ``inject_values`` during gradient overrides.
    
    For transcoders, each layer has its own independent encoder/decoder,
    unlike CLT where features can span multiple layers.
    """
    # Assert that the values in transcoders are of type SparseAutoEncoder
    assert all(isinstance(t, SparseAutoEncoder) for t in transcoders.values())

    rows: List[torch.Tensor] = []
    
    # Convert activations to coalesced sparse tensors for each layer
    feature_act_rows = [activations[layer].coalesce() for layer in range(len(transcoders))]
    
    for layer in range(len(transcoders)):
        _, feat_idx = feature_act_rows[layer].indices()
        
        # Retrieve the decoder weights from the current layer's transcoder
        W_D = transcoders[str(layer)].W_D  # Shape: [d_sae, d_model]
        # Scale the decoder row by the feature activations
        # W_D[feat_idx]: [n_active_features, d_model]
        # feature_act_rows[layer].values(): [n_active_features]
        scaled_row = W_D[feat_idx] * feature_act_rows[layer].values()[:, None]
        
        rows.append(scaled_row)
    
    # Concatenate all the scaled rows
    return torch.cat(rows)


# @torch.no_grad()
# def select_scaled_decoder_vecs_lorsa(
#     activation_matrix: torch.sparse.Tensor,
#     lorsas: LowRankSparseAttention
# ) -> torch.Tensor:
#     """Return encoder rows for **active** features only."""
#     rows: List[torch.Tensor] = []
#     for layer, row in enumerate(activation_matrix):
#         _, head_idx = row.coalesce().indices()
#         rows.append(lorsas[layer].W_O[head_idx])
#     return torch.cat(rows) * activation_matrix.values()[:, None]

# @torch.no_grad()
# def select_encoder_rows_clt(
#     activation_matrix: torch.sparse.Tensor, transcoders: List[SparseAutoEncoder]
# ) -> torch.Tensor:
#     """Return encoder rows for **active** features only."""
#     rows: List[torch.Tensor] = []
#     for layer, row in enumerate(activation_matrix):
#         _, feat_idx = row.coalesce().indices()
#         rows.append(transcoders.W_E[layer].T[feat_idx])
#     return torch.cat(rows)

@torch.no_grad()
def select_encoder_rows_tc(
    activation_matrix: torch.sparse.Tensor, 
    transcoders: Dict[int, SparseAutoEncoder]
) -> torch.Tensor:
    """Return encoder rows for **active** features only.
    
    For transcoders, each layer has its own independent encoder/decoder,
    unlike CLT where features can span multiple layers.
    """
    rows: List[torch.Tensor] = []
    
    # 遍历每一层的激活矩阵
    for layer, row in enumerate(activation_matrix):
        _, feat_idx = row.coalesce().indices()
        
        # Use string key to access transcoder for this layer
        # W_E.T[feat_idx]: [n_active_features, d_model] 
        rows.append(transcoders[str(layer)].W_E.T[feat_idx]) 
        
    return torch.cat(rows)

# @torch.no_grad()
# def select_encoder_rows_lorsa(
#     activation_matrix: torch.sparse.Tensor,
#     lorsas: LowRankSparseAttention
# ) -> torch.Tensor:
#     """Return encoder rows for **active** features only."""
#     rows: List[torch.Tensor] = []
#     for layer, row in enumerate(activation_matrix):
#         _, head_idx = row.coalesce().indices()
#         rows.append(lorsas[layer].W_V[head_idx])
#     return torch.cat(rows)

def compute_partial_influences(edge_matrix, logit_p, row_to_node_index, max_iter=128, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalized_matrix = torch.empty_like(edge_matrix, device=device).copy_(edge_matrix)
    normalized_matrix = normalized_matrix.abs_()
    normalized_matrix /= normalized_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)

    influences = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod[-len(logit_p) :] = logit_p

    for _ in range(max_iter):
        prod = prod[row_to_node_index] @ normalized_matrix
        if not prod.any():
            break
        influences += prod
    else:
        raise RuntimeError("Failed to converge")

    return influences


def ensure_tokenized(prompt: Union[str, torch.Tensor, List[int]], tokenizer) -> torch.Tensor:
    """Convert *prompt* → 1-D tensor of token ids (no batch dim)."""

    # if isinstance(prompt, str):
    #     return tokenizer(prompt, return_tensors="pt").input_ids[0]
    if isinstance(prompt, str):
        return tokenizer(prompt).squeeze(0)
    if isinstance(prompt, torch.Tensor):
        return prompt.squeeze(0) if prompt.ndim == 2 else prompt
    if isinstance(prompt, list):
        return torch.tensor(prompt, dtype=torch.long)
    raise TypeError(f"Unsupported prompt type: {type(prompt)}")


def attribute(
    prompt: Union[str, torch.Tensor, List[int]],
    model: ReplacementModel,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_feature_nodes: Optional[int] = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
    use_legal_moves_only: bool = False,
    fen: str = None,
    lboard = None,
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
            use_legal_moves_only=use_legal_moves_only,
            fen=fen,
            lboard=lboard,
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
    use_legal_moves_only=True,
    fen=None,
    lboard=None,
):
    start_time = time.time()
    # Phase 0: precompute
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.time()
    input_ids = ensure_tokenized(prompt, model.tokenizer)
    logits, lorsa_activation_matrix, tc_activation_matrix, error_vecs, token_vecs = model.setup_attribution(
        input_ids, sparse=True
    )
    lorsa_decoder_vecs = select_scaled_decoder_vecs_lorsa(lorsa_activation_matrix, model.lorsas)
    lorsa_encoder_rows = select_encoder_rows_lorsa(lorsa_activation_matrix, model.lorsas)

    tc_decoder_vecs = select_scaled_decoder_vecs_tc(tc_activation_matrix, model.transcoders)
    clt_encoder_rows = select_encoder_rows_tc(tc_activation_matrix, model.transcoders)

    ctx = AttributionContext(
        lorsa_activation_matrix,
        tc_activation_matrix,
        error_vecs,
        token_vecs,
        lorsa_decoder_vecs,
        tc_decoder_vecs,
        model.attn_output_hook,
        model.mlp_output_hook
    )
    logger.info(f"Precomputation completed in {time.time() - phase_start:.2f}s")
    logger.info(f"Found {lorsa_activation_matrix._nnz() + tc_activation_matrix._nnz()} active features")

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
    n_layers, n_pos, _ = lorsa_activation_matrix.shape
    total_active_feats = lorsa_activation_matrix._nnz() + tc_activation_matrix._nnz()

    # 获取残差流用于Jacobian计算
    if hasattr(model, 'policy_head') and not hasattr(model, 'unembed'):
        # LC0模型：使用residual作为policy_head的输入计算Jacobian
        unembed_matrix = None
        residual_for_jacobian = residual
    else:
        # 标准模型：使用现有的unembed矩阵
        unembed_matrix = model.unembed.W_U
        residual_for_jacobian = None
    
    if use_legal_moves_only and fen is not None and lboard is not None:
        logit_idx, logit_p, logit_vecs = compute_salient_legal_logits_for_lc0(
            logits[0, -1],
            unembed_matrix,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            model=model,
            residual_input=residual_for_jacobian,
            fen=fen,
            lboard=lboard,
        )
    else:
        logit_idx, logit_p, logit_vecs = compute_salient_logits_for_lc0(
            logits[0, -1],
            unembed_matrix,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            model=model,
            residual_input=residual_for_jacobian,
        )
    logger.info(
        f"Selected {len(logit_idx)} logits with cumulative probability {logit_p.sum().item():.4f}"
    )

    if offload:
        offload_handles += offload_modules([model.unembed, model.embed], offload)

    logit_offset = total_active_feats + 2 * n_layers * n_pos + n_pos
    n_logits = len(logit_idx)
    total_nodes = logit_offset + n_logits

    max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
    logger.info(f"Will include {max_feature_nodes} of {total_active_feats} feature nodes")

    edge_matrix = torch.zeros(max_feature_nodes + n_logits, total_nodes)
    # Maps row indices in edge_matrix to original feature/node indices
    # First populated with logit node IDs, then feature IDs in attribution order
    row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)
    logger.info(f"Input vectors built in {time.time() - phase_start:.2f}s")

    # Phase 3: logit attribution
    logger.info("Phase 3: Computing logit attributions")
    phase_start = time.time()
    for i in range(0, len(logit_idx), batch_size):
        batch = logit_vecs[i : i + batch_size] # (bs, k, d_model)
        # print(logits[0, -1].max() - logits[0, -1].mean(), logits[0, -1].max())
        rows = ctx.compute_batch(
            layers=torch.full((batch.shape[0],), 2 * n_layers),
            positions=torch.full((batch.shape[0],), n_pos - 1),
            inject_values=batch,
        )
        # print(rows[0].sum())
        # print("lorsa", rows[0, :lorsa_activation_matrix._nnz()].sum())
        # print("clt", rows[0, lorsa_activation_matrix._nnz(): total_active_feats].sum())
        # print("lorsa error", rows[0, total_active_feats: total_active_feats + n_layers * n_pos].sum())
        # print("clt error", rows[0, total_active_feats + n_layers * n_pos: total_active_feats + 2 * n_layers * n_pos].sum())
        # print("token", rows[0, total_active_feats + 2 * n_layers * n_pos:].sum())

        bias_attributions = []
        for param in model._get_requires_grad_bias_params():
            try:
                attribution = (param[1].data * param[1].grad).sum()
                bias_attributions.append(attribution)
            
            except TypeError as e:
                pass
        assert torch.allclose(sum(bias_attributions) + rows[0].sum(), logits[0, -1].max() - logits[0, -1].mean(), atol=1e-3)
        assert total_active_feats + (2 * n_layers + 1) * n_pos == rows.shape[1]
        # print(sum(bias_attributions))

        edge_matrix[i : i + batch.shape[0], :logit_offset] = rows.cpu()
        row_to_node_index[i : i + batch.shape[0]] = (
            torch.arange(i, i + batch.shape[0]) + logit_offset
        )
    logger.info(f"Logit attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 4: feature attribution
    logger.info("Phase 4: Computing feature attributions")
    
    lorsa_feat_layer, lorsa_feat_pos, _ = lorsa_activation_matrix.indices()
    clt_feat_layer, clt_feat_pos, _ = tc_activation_matrix.indices()

    def idx_to_layer(idx: torch.Tensor) -> torch.Tensor:
        is_lorsa = idx < len(lorsa_feat_layer)
        return torch.where(
            is_lorsa.to(lorsa_feat_layer.device),
            2 * lorsa_feat_layer[idx * is_lorsa],
            2 * clt_feat_layer[(idx - len(lorsa_feat_layer)) * ~is_lorsa] + 1
        )
    def idx_to_pos(idx: torch.Tensor) -> torch.Tensor:
        is_lorsa = idx < len(lorsa_feat_layer)
        return torch.where(
            is_lorsa.to(lorsa_feat_pos.device),
            lorsa_feat_pos[idx * is_lorsa],
            clt_feat_pos[(idx - len(lorsa_feat_layer)) * ~is_lorsa]
        )
    def idx_to_encoder_rows(idx: torch.Tensor) -> torch.Tensor:
        is_lorsa = idx < len(lorsa_feat_layer)
        return torch.where(
            is_lorsa.to(lorsa_encoder_rows.device)[:, None],
            lorsa_encoder_rows[idx * is_lorsa],
            clt_encoder_rows[(idx - len(lorsa_feat_layer)) * ~is_lorsa]
        )

    def idx_to_activation_values(idx: torch.Tensor) -> torch.Tensor:
        """Get activation values for transcoder feature nodes.
        
        Args:
            idx: Global feature indices for transcoder nodes only
            
        Returns:
            Activation values for the specified transcoder features
        """
        # 只处理transcoder节点，直接索引
        return tc_activation_matrix.values()[idx]

    phase_start = time.time()
    st = n_logits
    visited = torch.zeros(total_active_feats, dtype=torch.bool)
    n_visited = 0

    pbar = tqdm(total=max_feature_nodes, desc="Feature influence computation")

    while n_visited < max_feature_nodes:
        if max_feature_nodes == total_active_feats:
            pending = torch.arange(total_active_feats)
        else:
            influences = compute_partial_influences(
                edge_matrix[:st], logit_p, row_to_node_index[:st]
            )
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
                retain_graph=n_visited < max_feature_nodes,
            )

            end = min(st + batch_size, st + rows.shape[0])
            edge_matrix[st:end, :logit_offset] = rows.cpu()
            row_to_node_index[st:end] = idx_batch
            visited[idx_batch] = True
            st = end
            pbar.update(len(idx_batch))

    pbar.close()
    logger.info(f"Feature attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 5: packaging graph
    selected_features = torch.where(visited)[0]
    if max_feature_nodes < total_active_feats:
        non_feature_nodes = torch.arange(total_active_feats, total_nodes)
        col_read = torch.cat([selected_features, non_feature_nodes])
        edge_matrix = edge_matrix[:, col_read]

    # sort rows such that features are in order
    edge_matrix = edge_matrix[row_to_node_index.argsort()]
    final_node_count = edge_matrix.shape[1]
    full_edge_matrix = torch.zeros(final_node_count, final_node_count)
    full_edge_matrix[:max_feature_nodes] = edge_matrix[:max_feature_nodes]
    full_edge_matrix[-n_logits:] = edge_matrix[max_feature_nodes:]

    graph = Graph(
        input_string=model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_tokens=logit_idx,
        logit_probabilities=logit_p,
        lorsa_active_features=lorsa_activation_matrix.indices().T,
        lorsa_activation_values=lorsa_activation_matrix.values(),
        clt_active_features=tc_activation_matrix.indices().T,
        clt_activation_values=tc_activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edge_matrix,
        cfg=model.cfg,
        scan=None,
    )

    total_time = time.time() - start_time
    logger.info(f"Attribution completed in {total_time:.2f}s")

    return graph