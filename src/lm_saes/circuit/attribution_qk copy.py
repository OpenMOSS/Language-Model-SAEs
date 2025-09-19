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
from lm_saes import SparseAutoEncoder, LowRankSparseAttention

import numpy as np
import torch
from einops import einsum
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from .graph_lc0 import Graph
from .replacement_lc0_model import ReplacementModel
from .utils.disk_offload import offload_modules
from .utils.create_graph_files import create_graph_files

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
        lorsa_activation_matrix: torch.sparse.Tensor,
        tc_activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor, 
        lorsa_decoder_vecs: torch.Tensor,
        tc_decoder_vecs: torch.Tensor,
        attn_output_hook: str,
        mlp_output_hook: str,
    ) -> None:
        # assert lorsa_activation_matrix.shape[:-1] == tc_activation_matrix.shape[:-1], "LORSAs and TCs must have the same shape"
        n_layers, n_pos, _ = tc_activation_matrix.shape # tc_activation_matrix.shape = torch.Size([15, 64, 12288])
        # Forward-pass cache
        # # L0Ainput, L0Minput, ... L-1Ainput, L-1Minput, pre_unembed
        # L0Minput, L1Minput, ... L-1Minput, policy_head_input for transcoder only tracing
        self._resid_activations: List[torch.Tensor | None] = [None] * (2 * n_layers + 1)
        # 添加policy head的q和k activations缓存
        self._policy_q_activations: torch.Tensor | None = None
        self._policy_k_activations: torch.Tensor | None = None
        # (row_size, batch_size, 1)
        self._batch_buffer: torch.Tensor | None = None
        self.n_layers: int = n_layers

        # Assemble all backward hooks up-front
        self._attribution_hooks = self._make_attribution_hooks(
            lorsa_activation_matrix,
            tc_activation_matrix,
            error_vectors,
            token_vectors,
            lorsa_decoder_vecs,
            tc_decoder_vecs,
            attn_output_hook,
            mlp_output_hook
        )
        
        total_active_feats = lorsa_activation_matrix._nnz() + tc_activation_matrix._nnz()
        # total_active_feats + error_vectors + token_vectors
        self._row_size: int = total_active_feats + 2 * n_layers * n_pos + n_pos  # + logits later

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

    def _caching_hooks(self, attn_input_hook: str, mlp_input_hook: str) -> List[Tuple[str, Callable]]:
        """Return hooks that store residual activations layer-by-layer."""

        proxy = weakref.proxy(self)

        def _cache(acts: torch.Tensor, hook: HookPoint, *, index: int) -> torch.Tensor:
            proxy._resid_activations[index] = acts
            # 为非叶子张量设置retain_grad以便检查梯度传播
            if acts.requires_grad:
                acts.retain_grad()
            # print(f"DEBUG: _cache: {acts.shape}, retain_grad set")
            return acts

        def _cache_q(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            proxy._policy_q_activations = acts
            # 为q activations设置retain_grad
            if acts.requires_grad:
                acts.retain_grad()
            # print(f"DEBUG: _cache_q: {acts.shape}, retain_grad set")
            return acts

        def _cache_k(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            proxy._policy_k_activations = acts
            # 为k activations设置retain_grad  
            if acts.requires_grad:
                acts.retain_grad()
            # print(f"DEBUG: _cache_k: {acts.shape}, retain_grad set")
            return acts

        hooks = []
        for layer in range(self.n_layers):
            hooks.append((f"blocks.{layer}.{attn_input_hook}", partial(_cache, index=layer * 2)))
            hooks.append((f"blocks.{layer}.{mlp_input_hook}", partial(_cache, index=layer * 2 + 1)))
        
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
            # print(f"DEBUG: Hook '{hook_name}' executed")
            # print(f"DEBUG: grads shape: {grads.shape}")
            grads_non_zero_row_idx = (grads[0] != 0).any(dim=1).nonzero(as_tuple=True)[0]
            # print(f"DEBUG: grads[0][6]: {grads[0][6].flatten()[:5].tolist()}")
            # print(f"DEBUG: grads[0][46]: {grads[0][46].flatten()[:5].tolist()}")
            # print(f"DEBUG: output_vecs shape: {output_vecs.shape}")
            # print(f"DEBUG: {output_vecs.flatten()[:10].tolist() = }")
            # print(f"DEBUG: write_index: {write_index}")
            # print(f"DEBUG: read_index: {read_index}")
            
            # 计算einsum前的形状信息
            grads_read = grads.to(output_vecs.dtype)[read_index] #[1, 2240, 768]
            # print(f"DEBUG: grads[read_index] shape: {grads_read.shape}")
            # print(f"DEBUG: grads_read.shape = {grads_read.shape}")
            # print(f"DEBUG: output_vecs.shape = {output_vecs.shape}")
            # 执行einsum计算
            result = einsum(
                grads_read,
                output_vecs,
                "batch position d_model, position d_model -> position batch",
            )
            # print(f"DEBUG: grads_read.shape = {grads_read.shape}")
            # print(f"DEBUG: output_vecs.shape = {output_vecs.shape}")
            # print(f"DEBUG: einsum result shape: {result.shape}")
            # print(f"DEBUG: einsum result sum: {result.sum().item()}")
            
            # 写入缓冲区
            # print(f"DEBUG: Updated _batch_buffer[{write_index}]")
            # print(f"DEBUG: _batch_buffer[{write_index}] sum: {proxy._batch_buffer[write_index].sum().item()}")
            # print(f'{result.shape = }')
            # print(f'{proxy._batch_buffer[write_index].shape = }')
            # print("---")
            proxy._batch_buffer[write_index] += result
            # print(f"DEBUG: Updated _batch_buffer[{write_index}]")
            # print(f"DEBUG: _batch_buffer[{write_index}] sum: {proxy._batch_buffer[write_index].sum().item()}")


        return hook_name, _hook_fn


    def _make_attribution_hooks(
        self,
        lorsa_activation_matrix: torch.sparse.Tensor,
        tc_activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        lorsa_decoder_vecs: torch.Tensor,
        tc_decoder_vecs: torch.Tensor,
        attn_output_hook: str,
        mlp_output_hook: str,
    ) -> List[Tuple[str, Callable]]:
        """
        Create the complete backward-hook for computing attribution scores.
        """
        _, n_pos, _ = tc_activation_matrix.shape
        lorsa_error_vectors = error_vectors[:self.n_layers]
        tc_error_vectors = error_vectors[self.n_layers:]
        # Token-embedding nodes
        # lorsa_offset + tc_offset + mlp_error_offset + lorsa_error_offset
        token_offset = lorsa_activation_matrix._nnz() + tc_activation_matrix._nnz() + 2 * self.n_layers * n_pos
        token_hook = [
            self._compute_score_hook(
                "hook_embed",
                token_vectors,
                write_index=np.s_[token_offset : token_offset + n_pos],
            )
        ]
        return token_hook + self._make_attribution_hooks_lorsa( 
            lorsa_activation_matrix,
            lorsa_error_vectors,
            lorsa_decoder_vecs,
            attn_output_hook,
            tc_offset=tc_activation_matrix._nnz() 
        ) + self._make_attribution_hooks_tc(
            tc_activation_matrix,
            tc_error_vectors,
            tc_decoder_vecs,
            mlp_output_hook,
            lorsa_offset=lorsa_activation_matrix._nnz()  # TC 从 LoRSA 结束位置开始
        )

    def _make_attribution_hooks_lorsa(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        attn_output_hook: str,
        tc_offset: int,
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
        print(f'in _make_attribution_hooks_lorsa : {layer_spans = }')
        
        
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
            return activation_matrix._nnz() + tc_offset + layer * n_pos
        
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
        lorsa_offset: int,
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
        print(f'in _make_attribution_hooks_tc : {layer_spans = }')
        

        # Simple assertion: decoder_vecs should match total active features
        assert edges[-1] == activation_matrix._nnz(), f'got {edges[-1]} but expected {activation_matrix._nnz()}'
        assert decoder_vecs.size(0) == activation_matrix._nnz(), f'got {decoder_vecs.size(0)} but expected {activation_matrix._nnz()}'

        print(f'{lorsa_offset = }')
        # Feature nodes
        feature_hooks = []
        for layer, (start, end) in enumerate(layer_spans):
            if start != end:
                print(f'{start = }, {end = }')
                hook = self._compute_score_hook(
                    f"blocks.{layer}.{mlp_output_hook}",
                    decoder_vecs[start:end],
                    write_index=np.s_[lorsa_offset+start:lorsa_offset+end],
                    read_index=np.s_[:, nnz_positions[start:end]],
                )
                feature_hooks.append(hook)

        # Error nodes
        def error_offset(layer: int) -> int:
            # lorsa_offset + tc_offset + attn_error_offset + layer_offset
            return lorsa_offset + activation_matrix._nnz() + self.n_layers * n_pos + layer * n_pos

        error_hooks = []
        for layer in range(n_layers):
            hook = self._compute_score_hook(
                f"blocks.{layer}.{mlp_output_hook}",
                error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
            )
            error_hooks.append(hook)

        return feature_hooks + error_hooks

    @contextlib.contextmanager
    def install_hooks(self, model: "ReplacementModel"):
        """Context manager instruments the hooks for the forward and backward passes."""
        with model.hooks(
            fwd_hooks=self._caching_hooks(model.attn_input_hook, model.mlp_input_hook),
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
        # print(f"DEBUG: batch_size = {batch_size}") [1]
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
                # print(f'use pattern method')
                if patterns.shape[1] > grads.shape[1]:
                    raise IndexError(f"Patterns seq_len ({patterns.shape[1]}) > grads seq_len ({grads.shape[1]})")
                
                distributed_values = values[:, None, :] * patterns[:, :, None]
                grads_out.index_put_((batch_indices,), distributed_values)
            else:
                grads_out.index_put_((batch_indices, pos_indices), values)
            
            return grads_out.to(grads.dtype)

        handles = []
        layers_in_batch = layers.unique().tolist()
        # print(f'{layers_in_batch = }')
        for layer in layers_in_batch:
            mask = layers == layer
            if not mask.any():
                continue
            
            if int(layer) >= len(self._resid_activations):
                raise IndexError(f"Layer {layer} out of range")
            
            # print(f'{batch_idx[mask] = }')
            # print(f'{positions[mask] = }')
            # print(f'{attention_patterns[mask] = }')
            # print(f'{inject_values.shape = }')
            # print(f'{inject_values[mask].flatten()[:5].tolist() = }')
            
            fn = partial(
                _inject,
                batch_indices=batch_idx[mask],
                pos_indices=positions[mask],
                patterns=attention_patterns[mask] if attention_patterns is not None else None,
                values=inject_values[mask],
            )
            # print(f"{len(self._resid_activations) = }")
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


    def compute_start_end_batch_from_q(
        self,
        move_positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """Return attribution rows for start positions of moves in single backward pass, starting from policy head q.

        This function performs one backward pass with gradients injected at
        start positions only for each move, starting the backward
        pass from the cached policy head q activations instead of the last layer.

        Args:
            layers: 1-D tensor of layer indices for the source nodes.
            move_positions: `(batch, 1)` tensor where move_positions[i] is start pos
                for the i-th move.
            inject_values: `(batch, seq_len, d_model)` tensor where we extract
                start position values for injection.

        Returns:
            torch.Tensor: `(batch, row_size)` matrix where each row corresponds to 
                the attribution of one start position.
        """
        
        for resid_activation in self._resid_activations:
            assert resid_activation is not None, "Residual activations are not cached"
        
        assert self._policy_q_activations is not None, "Policy head q activations are not cached"

        # Detach policy head k activations to isolate q tracing
        def detach_k_hook(acts, hook):
            """Detach k activations to prevent gradient flow"""
            return acts.detach()
        
        # # 添加detach k的hook
        # k_detach_handle = None
        # if self._policy_k_activations is not None and hasattr(self._policy_k_activations, 'grad'):
        #     # 如果k激活存在，直接detach它
        #     if self._policy_k_activations.requires_grad:
        #         self._policy_k_activations = self._policy_k_activations.detach()
        #         print("DEBUG: Detached policy head k activations")

        k_batch = move_positions.shape[0]
        device = inject_values.device
        
        # Ensure all tensors are on the same device
        start_pos = move_positions.to(dtype=torch.long, device=device)

            
        batch_size = self._policy_q_activations[0].shape[0]
        # print(f"DEBUG: batch_size = {batch_size}")
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=device,
        )

        # batch indices correspond to grads batch dim; grads batch size is 1 here
        batch_idx = torch.arange(len(start_pos), device=start_pos.device)

        def _inject_start_only(grads, *, batch_indices, start_positions, start_values):
            """Inject gradients only at start positions"""
            grads_out = grads.clone().to(start_values.dtype)
            
            # Only inject start positions, other positions remain 0
            grads_out.index_put_((batch_indices, start_positions), start_values)
            
            return grads_out.to(grads.dtype)

        handles = []

        layer_start_inject = torch.stack([
            inject_values[i, start_pos[i], :] for i in range(k_batch)
        ]) if k_batch > 0 else torch.empty(0, inject_values.shape[-1], device=device)
        

        if layer_start_inject.shape[0] > 0:  # Only register if there are items
            fn = partial(
                _inject_start_only,
                batch_indices=batch_idx,  # 所有batch indices
                start_positions=start_pos,  # 所有start positions
                start_values=layer_start_inject,  # 所有injection values
            )
            handles.append(self._policy_q_activations.register_hook(fn))
        
        try:
            self._policy_q_activations.backward(
                gradient=torch.zeros_like(self._policy_q_activations),
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        return buf.T[:k_batch]  # Return k_batch rows, one per start position

    def compute_start_end_batch_from_k(
        self,
        move_positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
        castle_tensor: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return attribution rows for end positions of moves in single backward pass, starting from policy head k.

        This function performs one backward pass with gradients injected at
        end positions only for each move, starting the backward
        pass from the cached policy head k activations instead of the last layer.

        Args:
            move_positions: `(batch, 1)` tensor where move_positions[i] is end pos
                for the i-th move.
            inject_values: `(batch, seq_len, d_model)` tensor where we extract
                end position values for injection.
            castle_tensor: `(batch,)` bool tensor indicating which moves are castling moves.
                If None, will auto-detect castling moves.

        Returns:
            torch.Tensor: `(batch, row_size)` matrix where each row corresponds to 
                the attribution of one end position.
        """
        
        for resid_activation in self._resid_activations:
            assert resid_activation is not None, "Residual activations are not cached"
        
        assert self._policy_k_activations is not None, "Policy head k activations are not cached"

        k_batch = move_positions.shape[0]
        device = inject_values.device
        
        # print(f'{inject_values.shape = }') # [1, 64, 768]
        # rows = torch.nonzero((inject_values[0].abs() > 0).any(dim=1), as_tuple=True)[0]
        # print(f'non zero rows in inject_values,{rows.cpu().tolist()}')

        # 如果未提供is_castle，则默认不异位
        if castle_tensor is None:
            castle_tensor = torch.zeros(k_batch, dtype=torch.bool, device=device)
        else:
            castle_tensor = castle_tensor.to(device=device, dtype=torch.bool)
        
        # Ensure all tensors are on the same device
        end_pos = move_positions.to(dtype=torch.long, device=device)

        # 王车易位检测和K位置调整
        adjusted_end_pos = end_pos.clone()
        
        for i in range(k_batch):
            if castle_tensor[i]:
                # 王车易位情况下的位置调整逻辑
                # start_row, start_col = start_pos[i] // 8, start_pos[i] % 8
                end_row, end_col = end_pos[i] // 8, end_pos[i] % 8
                if end_col == 6:  # 短异位 (e->g)
                    adjusted_end_pos[i] = end_row * 8 + 7  # h列车的位置
                    print(f"检测到短异位:  end={end_pos[i].item()} -> 调整K位置为: {adjusted_end_pos[i].item()}")
                elif end_col == 2:  # 长异位 (e->c)
                    adjusted_end_pos[i] = end_row * 8 + 0  # a列车的位置
                    print(f"检测到长异位: nd={end_pos[i].item()} -> 调整K位置为: {adjusted_end_pos[i].item()}")
                else:
                    print(f"警告: is_castle为True但移动不符合王车易位模式:end={end_pos[i].item()}")

        batch_size = self._policy_k_activations[0].shape[0]
        # print(f"DEBUG: batch_size = {batch_size}")
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=device,
        )
        # batch indices correspond to grads batch dim; grads batch size is 1 here
        batch_idx = torch.arange(len(adjusted_end_pos), device=adjusted_end_pos.device)

        def _inject_end_only(grads, *, batch_indices, end_positions, end_values):
            """Inject gradients only at end positions"""
            grads_out = grads.clone().to(end_values.dtype)

            grads_out.index_put_((batch_indices, end_positions), end_values)

            return grads_out.to(grads.dtype)
        handles = []
        layer_end_inject = torch.stack([
            inject_values[i, adjusted_end_pos[i], :] for i in range(k_batch)
        ]) if k_batch > 0 else torch.empty(0, inject_values.shape[-1], device=device)
        
        if layer_end_inject.shape[0] > 0:  # Only register if there are items
            fn = partial(
                _inject_end_only,
                batch_indices=batch_idx,  # 所有batch indices
                end_positions=adjusted_end_pos,  # 使用调整后的位置
                end_values=layer_end_inject,  # 所有injection values
            )

            handles.append(self._policy_k_activations.register_hook(fn))
        
        try:
            self._policy_k_activations.backward(
                gradient=torch.zeros_like(self._policy_k_activations),
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        return buf.T[:k_batch]  # Return k_batch rows, one per end position

def compute_logit_gradients_wrt_q(
    fen: str,
    logits: torch.Tensor,
    model=None,
    residual_input=None,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    demean: bool = True,
    move_idx: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算LC0模型中policy logits对q activations的梯度
    
    Args:
        fen: FEN字符串，表示当前棋盘状态
        logits: 策略logits, [1858]
        model: LC0模型（必须提供）
        residual_input: 残差输入（必须提供）
        max_n_logits: 最大选择的logits数量
        desired_logit_prob: 期望的累积概率阈值
        demean: 是否进行去均值化操作，默认为True
        move_idx: 指定要处理的move索引，如果提供则直接处理该索引
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            * top_idx - 选中的logit索引，形状为(k,)
            * top_p - 对应的概率值，形状为(k,)
            * gradient_matrix - 梯度矩阵，形状为(k, seq_len, d_model)
            * move_positions - 对应的move位置，形状为(k, 2)
    """
    
    if model is None or residual_input is None:
        raise ValueError("Both model and residual_input must be provided")
    
    if not hasattr(model, 'policy_head'):
        raise ValueError("Model must have policy_head attribute")
    
    lboard = LeelaBoard.from_fen(fen)
    
    if logits.numel() == 0:
        raise ValueError("Input logits tensor is empty")
    
    # 确保logits是1D张量
    if logits.dim() > 1:
        logits = logits.flatten()
    
    # 选择要处理的logit索引
    if move_idx is not None:
        if move_idx < 0 or move_idx >= logits.size(0):
            raise ValueError(f"move_idx {move_idx} 超出logits范围 [0, {logits.size(0)-1}]")
        
        top_idx = torch.tensor([move_idx], device=logits.device)
        probs = torch.softmax(logits, dim=-1)
        top_p = probs[move_idx].unsqueeze(0)
    else:
        # 原有的top logits选择逻辑
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
            logger.warning(f"无法获取索引 {idx.item()} 对应的move位置: {e}")
            move_positions.append(torch.tensor([0, 0]))
    
    move_positions_tensor = torch.stack(move_positions)
    
    # 准备计算梯度
    device = residual_input.device
    n_selected = len(top_idx)
    
    # 通过hook捕获q activations
    q_activations = None
    hook_handle = None
    
    def capture_q_hook(acts, hook):
        nonlocal q_activations
        # 创建新的叶子变量，这样可以设置requires_grad
        q_activations = acts.detach().clone().requires_grad_(True)
        return q_activations  # 返回我们的叶子变量，这样它就在计算图中
    
    try:
        # 注册hook到policy_head.hook_q
        hook_handle = model.policy_head.hook_q.add_hook(capture_q_hook)
        
        residual_input = residual_input.detach().clone().requires_grad_(True)

        print("residual_input requires_grad:", residual_input.requires_grad)  # True
        
        # 进行前向传播以捕获q activations
        policy_logits = model.policy_head(residual_input)
        
        # 确保q_activations被正确捕获
        if q_activations is None:
            raise ValueError("Failed to capture q activations through hook")
        
        # 计算选定logits对q的Jacobian矩阵
        batch_size, seq_len, d_model = q_activations.shape
        gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)
        
        for i, logit_idx in enumerate(top_idx):
            if q_activations.grad is not None:
                q_activations.grad.zero_()
            
            # 对选定的policy logit求梯度
            policy_logits[0, logit_idx].backward(retain_graph=True)
            
            if q_activations.grad is not None:
                grad = q_activations.grad[0, :, :].clone()  # shape: (seq_len, d_model)
                gradient_matrix[i, :, :] = grad
        
    finally:
        # 移除hook
        if hook_handle is not None:
            hook_handle.remove()
    
    # 去均值化处理
    if demean:
        mean_gradient = gradient_matrix.mean(dim=0, keepdim=True)
        result_matrix = gradient_matrix - mean_gradient
    else:
        result_matrix = gradient_matrix
    
    return top_idx, top_p, result_matrix.detach(), move_positions_tensor

def compute_logit_gradients_wrt_k(
    fen: str,
    logits: torch.Tensor,
    model=None,
    residual_input=None,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    demean: bool = True,
    move_idx: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算LC0模型中policy logits对k activations的梯度
    
    Args:
        fen: FEN字符串，表示当前棋盘状态
        logits: 策略logits
        model: LC0模型（必须提供）
        residual_input: 残差输入（必须提供）
        max_n_logits: 最大选择的logits数量
        desired_logit_prob: 期望的累积概率阈值
        demean: 是否进行去均值化操作，默认为True
        move_idx: 指定要处理的move索引，如果提供则直接处理该索引
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            * top_idx - 选中的logit索引，形状为(k,)
            * top_p - 对应的概率值，形状为(k,)
            * gradient_matrix - 梯度矩阵，形状为(k, seq_len, d_model)
            * move_positions - 对应的move位置，形状为(k, 2)
    """
    
    if model is None or residual_input is None:
        raise ValueError("Both model and residual_input must be provided")
    
    if not hasattr(model, 'policy_head'):
        raise ValueError("Model must have policy_head attribute")
    
    lboard = LeelaBoard.from_fen(fen)
    
    if logits.numel() == 0:
        raise ValueError("Input logits tensor is empty")
    
    # 确保logits是1D张量
    if logits.dim() > 1:
        logits = logits.flatten()
    
    # 选择要处理的logit索引
    if move_idx is not None:
        if move_idx < 0 or move_idx >= logits.size(0):
            raise ValueError(f"move_idx {move_idx} 超出logits范围 [0, {logits.size(0)-1}]")
        
        top_idx = torch.tensor([move_idx], device=logits.device)
        probs = torch.softmax(logits, dim=-1)
        top_p = probs[move_idx].unsqueeze(0)
    else:
        # 原有的top logits选择逻辑
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
            logger.warning(f"无法获取索引 {idx.item()} 对应的move位置: {e}")
            move_positions.append(torch.tensor([0, 0]))
    
    move_positions_tensor = torch.stack(move_positions)
    
    # 准备计算梯度
    device = residual_input.device
    n_selected = len(top_idx)
    
    # 通过hook捕获k activations
    k_activations = None
    hook_handle = None
    
    def capture_k_hook(acts, hook):
        nonlocal k_activations
        # 创建新的叶子变量，这样可以设置requires_grad
        k_activations = acts.detach().clone().requires_grad_(True)
        return k_activations  # 返回我们的叶子变量，这样它就在计算图中
    
    try:
        # 注册hook到policy_head.hook_k
        hook_handle = model.policy_head.hook_k.add_hook(capture_k_hook)
        
        residual_input = residual_input.detach().clone().requires_grad_(True)

        print("residual_input requires_grad:", residual_input.requires_grad)  # True
    
        policy_logits = model.policy_head(residual_input)
    
        if k_activations is None:
            raise ValueError("Failed to capture k activations through hook")

        batch_size, seq_len, d_model = k_activations.shape
        gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)
        
        for i, logit_idx in enumerate(top_idx):
            if k_activations.grad is not None:
                k_activations.grad.zero_()
            
            policy_logits[0, logit_idx].backward(retain_graph=True)
            if k_activations.grad is not None:
                grad = k_activations.grad[0, :, :].clone()
                gradient_matrix[i, :, :] = grad
        
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    
    if demean:
        mean_gradient = gradient_matrix.mean(dim=0, keepdim=True)
        result_matrix = gradient_matrix - mean_gradient
    else:
        result_matrix = gradient_matrix
    
    return top_idx, top_p, result_matrix.detach(), move_positions_tensor


# @torch.no_grad()
def compute_salient_logits_for_lc0(
    fen: str,
    logits: torch.Tensor,
    model=None,
    unembed_proj: torch.Tensor = None,  # 改为可选
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
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

    if unembed_proj is None and (model is None or residual_input is None):
        raise ValueError("Either unembed_proj or (model + residual_input) must be provided")

    if unembed_proj is not None and model is not None:
        logger.warning("Both unembed_proj and model provided, using model for Jacobian calculation")

    if logits.dim() > 1:
        logits = logits.flatten()
    
    if move_idx is not None:
        if move_idx < 0 or move_idx >= logits.size(0):
            raise ValueError(f"move_idx {move_idx} 超出logits范围 [0, {logits.size(0)-1}]")
        
        top_idx = torch.tensor([move_idx], device=logits.device)
        probs = torch.softmax(logits, dim=-1)
        top_p = probs[move_idx].unsqueeze(0)
    else:
        actual_max_logits = min(max_n_logits, logits.size(0))
        
        probs = torch.softmax(logits, dim=-1)
        top_p, top_idx = torch.topk(probs, actual_max_logits)
        cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
        top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

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

def compute_logit_gradients_wrt_group_k(
    fen: str,
    logits: torch.Tensor,
    model=None,
    residual_input=None,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    demean: bool = True,
    move_idx: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    group 模式：针对给定起点的"正样本走子 - 同起点的所有其他合法走子(负样本)"进行梯度差分。

    返回的 move_positions_tensor 为 LongTensor，形状 **[B, 2, M]**：
      - B = 1（单组），第 0 维是 batch
      - 第 1 维大小为 2：第 0 行是 Q 端起点，第 1 行是 K 端所有终点
      - 第 2 维大小为 M：列 0 放正样本终点，后续列放所有负样本终点
        * Q 行：只在第 0 列放起点，其余列用 -1 填充（下游可用 mask 过滤）
        * K 行：依次填 [终点_pos, 终点_neg1, 终点_neg2, ...]
    下游用法示例：
        batch_move_positions = move_positions[i:i+bs]      # [bs, 2, M]
        batch_move_positions_q = batch_move_positions[:, 0:1, :1]   # [bs,1,1] 只取起点
        batch_move_positions_k = batch_move_positions[:, 1:2, :]    # [bs,1,M] 全部 K 端终点
        valid_k_mask = (batch_move_positions_k >= 0)                 # 过滤 -1 的填充值

    其余返回与常规 compute_logit_gradients_wrt_qk 一致：
      top_idx, top_p, q_result_matrix, k_result_matrix, move_positions_tensor, residual_result_matrix
    """
    if model is None or residual_input is None:
        raise ValueError("Both model and residual_input must be provided")
    if not hasattr(model, 'policy_head'):
        raise ValueError("Model must have policy_head attribute")

    lboard = LeelaBoard.from_fen(fen)

    if logits.numel() == 0:
        raise ValueError("Input logits tensor is empty")
    if logits.dim() > 1:
        logits = logits.flatten()

    assert move_idx is not None, "move_idx must be given to compute_logit_gradients_wrt_group in group mode"
    if move_idx < 0 or move_idx >= logits.size(0):
        raise ValueError(f"move_idx {move_idx} 超出logits范围 [0, {logits.size(0)-1}]")

    top_idx = torch.tensor([move_idx], device=logits.device)
    probs = torch.softmax(logits, dim=-1)
    top_p = probs[move_idx].unsqueeze(0)

    # --- 组建正/负样本的 UCI 列表 ---
    chosen_uci = lboard.idx2uci(int(move_idx))
    start_sq = chosen_uci[:2]  # 同一起点
    legal_uci_all: List[str] = [mv.uci() for mv in lboard.generate_legal_moves()]
    negative_move_ucis = [u for u in legal_uci_all if u.startswith(start_sq) and u != chosen_uci]

    # --- 提取 Q 起点 & K 终点位置 ---
    def uci_to_qkpos(uci: str) -> tuple[int, int]:
        pos = lboard.uci_to_positions(uci)  # 期望返回 [q_pos, k_pos] 或 torch.Tensor([q,k])
        if isinstance(pos, torch.Tensor):
            return int(pos[0].item()), int(pos[1].item())
        return int(pos[0]), int(pos[1])

    qpos_pos, kpos_pos = uci_to_qkpos(chosen_uci)
    kpos_negs = [uci_to_qkpos(u)[1] for u in negative_move_ucis]

    # 先做 2×M：第0行 Q 起点(其余 -1)，第1行所有 K 终点
    M = 1 + len(kpos_negs)
    move_pos_2d = torch.full((2, M), -1, dtype=torch.long, device=logits.device)
    move_pos_2d[0, 0] = qpos_pos
    move_pos_2d[1, 0] = kpos_pos
    if len(kpos_negs) > 0:
        move_pos_2d[1, 1:1+len(kpos_negs)] = torch.tensor(kpos_negs, dtype=torch.long, device=logits.device)

    # 再包一层 batch 维度 => [1, 2, M]
    move_positions_tensor = move_pos_2d.unsqueeze(0)

    # ====== "正样本 − 负样本"梯度差分 ======
    device = residual_input.device
    n_selected = 1
    q_activations = None
    k_activations = None
    q_hook_handle = None
    k_hook_handle = None

    def capture_q_hook(acts, hook):
        nonlocal q_activations
        q_activations = acts
        q_activations.retain_grad()
        return q_activations

    def capture_k_hook(acts, hook):
        nonlocal k_activations
        k_activations = acts
        k_activations.retain_grad()
        return k_activations

    try:
        q_hook_handle = model.policy_head.hook_q.add_hook(capture_q_hook)
        k_hook_handle = model.policy_head.hook_k.add_hook(capture_k_hook)

        residual_input = residual_input.detach().clone().requires_grad_(True)
        policy_logits = model.policy_head(residual_input)

        if q_activations is None:
            raise ValueError("Failed to capture q activations through hook")
        if k_activations is None:
            raise ValueError("Failed to capture k activations through hook")

        _, seq_len, d_model = q_activations.shape
        q_gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)
        k_gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)
        residual_gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)

        # ---- 正样本 ----
        pos_idx = int(top_idx[0].item())
        model.zero_grad(set_to_none=True)
        if q_activations.grad is not None: q_activations.grad.zero_()
        if k_activations.grad is not None: k_activations.grad.zero_()
        if residual_input.grad is not None: residual_input.grad.zero_()

        policy_logits[0, pos_idx].backward(retain_graph=True)
        q_accum   = q_activations.grad[0].detach().clone()
        k_accum   = k_activations.grad[0].detach().clone()
        res_accum = residual_input.grad[0].detach().clone()

        # ---- 负样本（同起点）----
        neg_indices: List[int] = [lboard.uci2idx(u) for u in negative_move_ucis]
        n_neg = len(neg_indices)
        neg_weight = (1.0 / n_neg) if n_neg > 0 else 0.0   # 也可以改为 1.0 代表"简单相减"

        for j, neg_idx in enumerate(neg_indices):
            model.zero_grad(set_to_none=True)
            if q_activations.grad is not None: q_activations.grad.zero_()
            if k_activations.grad is not None: k_activations.grad.zero_()
            if residual_input.grad is not None: residual_input.grad.zero_()

            retain = (j < n_neg - 1)
            policy_logits[0, int(neg_idx)].backward(retain_graph=retain)
            
            # x = k_activations.grad[0]        # 形状 [64, 768]
            # mask = (x != 0).any(dim=1)       # [64] 的 bool
            # row_idx = mask.nonzero(as_tuple=True)[0]   # LongTensor，非零行下标
            # print(row_idx.tolist())
            
            q_accum   -= neg_weight * q_activations.grad[0].detach()
            k_accum   -= neg_weight * k_activations.grad[0].detach()
            res_accum -= neg_weight * residual_input.grad[0].detach()

        q_gradient_matrix[0] = q_accum
        k_gradient_matrix[0] = k_accum
        residual_gradient_matrix[0] = res_accum

    finally:
        if q_hook_handle is not None:
            q_hook_handle.remove()
        if k_hook_handle is not None:
            k_hook_handle.remove()

    # 如需去均值可在此处打开 demean 分支
    q_result_matrix = q_gradient_matrix
    k_result_matrix = k_gradient_matrix
    residual_result_matrix = residual_gradient_matrix

    return (
        top_idx,
        top_p,
        q_result_matrix.detach(),
        k_result_matrix.detach(),
        move_positions_tensor,           # [1, 2, M]
        residual_result_matrix.detach(),
    )


def compute_logit_gradients_wrt_qk(
    fen: str,
    logits: torch.Tensor,
    model=None,
    residual_input=None,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    demean: bool = True,
    move_idx: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算LC0模型中policy logits对q和k activations的梯度
    
    Args:
        fen: FEN字符串，表示当前棋盘状态
        logits: 策略logits
        model: LC0模型（必须提供）
        residual_input: 残差输入（必须提供）
        max_n_logits: 最大选择的logits数量
        desired_logit_prob: 期望的累积概率阈值
        demean: 是否进行去均值化操作，默认为True
        move_idx: 指定要处理的move索引，如果提供则直接处理该索引
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            * top_idx - 选中的logit索引，形状为(k,)
            * top_p - 对应的概率值，形状为(k,)
            * q_gradient_matrix - q的梯度矩阵，形状为(k, seq_len, d_model)
            * k_gradient_matrix - k的梯度矩阵，形状为(k, seq_len, d_model)
            * move_positions - 对应的move位置，形状为(k, 2)
            * residual_gradient_matrix - residual_input的梯度矩阵，形状为(k, seq_len, d_model)
    """
    
    if model is None or residual_input is None:
        raise ValueError("Both model and residual_input must be provided")
    
    if not hasattr(model, 'policy_head'):
        raise ValueError("Model must have policy_head attribute")
    
    lboard = LeelaBoard.from_fen(fen)
    
    if logits.numel() == 0:
        raise ValueError("Input logits tensor is empty")
    
    # 确保logits是1D张量
    if logits.dim() > 1:
        logits = logits.flatten()
    
    # 选择要处理的logit索引
    if move_idx is not None:
        if move_idx < 0 or move_idx >= logits.size(0):
            raise ValueError(f"move_idx {move_idx} 超出logits范围 [0, {logits.size(0)-1}]")
        
        top_idx = torch.tensor([move_idx], device=logits.device)
        probs = torch.softmax(logits, dim=-1)
        top_p = probs[move_idx].unsqueeze(0)
    else:
        # 原有的top logits选择逻辑
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
            logger.warning(f"无法获取索引 {idx.item()} 对应的move位置: {e}")
            move_positions.append(torch.tensor([0, 0]))
    
    move_positions_tensor = torch.stack(move_positions)
    
    # 准备计算梯度
    device = residual_input.device
    n_selected = len(top_idx)
    
    # 通过hook捕获q和k activations
    q_activations = None
    k_activations = None
    q_hook_handle = None
    k_hook_handle = None
    
    def capture_q_hook(acts, hook):
        nonlocal q_activations
        # 使用retain_grad()来保留梯度，而不是创建叶子节点
        q_activations = acts
        q_activations.retain_grad()
        return q_activations
    
    def capture_k_hook(acts, hook):
        nonlocal k_activations
        # 使用retain_grad()来保留梯度，而不是创建叶子节点
        k_activations = acts
        k_activations.retain_grad()
        return k_activations
    
    try:
        # 注册hook到policy_head.hook_q和hook_k
        q_hook_handle = model.policy_head.hook_q.add_hook(capture_q_hook)
        k_hook_handle = model.policy_head.hook_k.add_hook(capture_k_hook)
        
        # 将residual_input设置为叶子节点
        residual_input = residual_input.detach().clone().requires_grad_(True)

        print("residual_input requires_grad:", residual_input.requires_grad)  # True
        
        # 进行前向传播以捕获q和k activations
        policy_logits = model.policy_head(residual_input)
        
        # 确保q和k activations被正确捕获
        if q_activations is None:
            raise ValueError("Failed to capture q activations through hook")
        if k_activations is None:
            raise ValueError("Failed to capture k activations through hook")
        
        # 计算选定logits对q、k和residual_input的Jacobian矩阵
        batch_size, seq_len, d_model = q_activations.shape
        q_gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)
        k_gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)
        residual_gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)
        
        for i, logit_idx in enumerate(top_idx):
            # 清零所有梯度
            if q_activations.grad is not None:
                q_activations.grad.zero_()
            if k_activations.grad is not None:
                k_activations.grad.zero_()
            if residual_input.grad is not None:
                residual_input.grad.zero_()
            
            # 对选定的policy logit求梯度
            policy_logits[0, logit_idx].backward(retain_graph=True)
            
            # 收集q的梯度
            if q_activations.grad is not None:
                grad = q_activations.grad[0, :, :].clone()  # shape: (seq_len, d_model)
                q_gradient_matrix[i, :, :] = grad
            
            # 收集k的梯度
            if k_activations.grad is not None:
                grad = k_activations.grad[0, :, :].clone()  # shape: (seq_len, d_model)
                k_gradient_matrix[i, :, :] = grad
            
            # 收集residual_input的梯度
            if residual_input.grad is not None:
                grad = residual_input.grad[0, :, :].clone()  # shape: (seq_len, d_model)
                residual_gradient_matrix[i, :, :] = grad
        
    finally:
        # 移除hooks
        if q_hook_handle is not None:
            q_hook_handle.remove()
        if k_hook_handle is not None:
            k_hook_handle.remove()
    
    # 去均值化处理
    if demean:
        q_mean_gradient = q_gradient_matrix.mean(dim=0, keepdim=True)
        k_mean_gradient = k_gradient_matrix.mean(dim=0, keepdim=True)
        residual_mean_gradient = residual_gradient_matrix.mean(dim=0, keepdim=True)
        q_result_matrix = q_gradient_matrix - q_mean_gradient
        k_result_matrix = k_gradient_matrix - k_mean_gradient
        residual_result_matrix = residual_gradient_matrix - residual_mean_gradient
    else:
        q_result_matrix = q_gradient_matrix
        k_result_matrix = k_gradient_matrix
        residual_result_matrix = residual_gradient_matrix
    
    return top_idx, top_p, q_result_matrix.detach(), k_result_matrix.detach(), move_positions_tensor, residual_result_matrix.detach()



@torch.no_grad()  # modified
def select_scaled_decoder_vecs_tc(
    activations: torch.sparse.Tensor,
    transcoders: Dict[int, SparseAutoEncoder]
) -> torch.Tensor:
    """Return decoder rows for **active** features only.

    The return value is already scaled by the feature activation, making it
    suitable as ``inject_values`` during gradient overrides.
    
    For transcoders, each layer has its own independent encoder/decoder,
    unlike TC where features can span multiple layers.
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

@torch.no_grad()
def select_scaled_decoder_vecs_lorsa(
    activation_matrix: torch.sparse.Tensor,
    lorsas: LowRankSparseAttention
) -> torch.Tensor:
    """Return encoder rows for **active** features only."""
    rows: List[torch.Tensor] = []
    for layer, row in enumerate(activation_matrix):
        _, head_idx = row.coalesce().indices()
        rows.append(lorsas[layer].W_O[head_idx])
    return torch.cat(rows) * activation_matrix.values()[:, None]


@torch.no_grad()
def select_encoder_rows_tc(
    activation_matrix: torch.sparse.Tensor, 
    transcoders: Dict[int, SparseAutoEncoder]
) -> torch.Tensor:
    """Return encoder rows for **active** features only.
    
    For transcoders, each layer has its own independent encoder/decoder,
    unlike TC where features can span multiple layers.
    """
    rows: List[torch.Tensor] = []
    
    # 遍历每一层的激活矩阵
    for layer, row in enumerate(activation_matrix):
        _, feat_idx = row.coalesce().indices()
        
        # Use string key to access transcoder for this layer
        # W_E.T[feat_idx]: [n_active_features, d_model] 
        rows.append(transcoders[str(layer)].W_E.T[feat_idx]) 
        
    return torch.cat(rows)

@torch.no_grad()
def select_encoder_rows_lorsa(
    activation_matrix: torch.sparse.Tensor,
    attention_pattern: torch.Tensor,
    lorsas: LowRankSparseAttention
) -> torch.Tensor:
    """Return encoder rows for **active** features only."""
    rows: List[torch.Tensor] = []
    patterns: List[torch.Tensor] = []
    torch.cuda.synchronize()
    for layer, row in enumerate(activation_matrix):
        qpos, head_idx = row.coalesce().indices()
        # qk_idx = head_idx // lorsas[layer].cfg.d_qk_head
        qk_idx: Tensor = head_idx // (lorsas[layer].cfg.n_ov_heads // lorsas[layer].cfg.n_qk_heads)
        # torch.cuda.synchronize()
        # print(f'{attention_pattern.shape = }, {layer = }, {qk_idx = }, {qpos = }')
        pattern = attention_pattern[layer, qk_idx, qpos]
        patterns.append(pattern)
        rows.append(lorsas[layer].W_V[head_idx])
    return torch.cat(rows), torch.cat(patterns)

@torch.no_grad()
def select_encoder_bias_tc(
    activation_matrix: torch.sparse.Tensor,
    transcoders: Dict[str, "SparseAutoEncoder"],  # 与你 rows 版本一致，用 str(layer) 作为键
) -> torch.Tensor:
    rows: List[torch.Tensor] = []

    for layer, row in enumerate(activation_matrix):
        idx2d = row.coalesce().indices()
        if idx2d.numel() == 0:
            continue 

        _, feat_idx = idx2d  
        tc = transcoders[str(layer)]

        if getattr(tc, "b_E", None) is None:
            dev, dt = tc.W_E.device, tc.W_E.dtype
            layer_bias = torch.zeros(feat_idx.numel(), device=dev, dtype=dt)
        else:
            layer_bias = tc.b_E.index_select(0, feat_idx.to(device=tc.b_E.device))

        rows.append(layer_bias)

    if not rows:
        if len(transcoders) > 0:
            any_tc = next(iter(transcoders.values()))
            return torch.empty(0, device=any_tc.W_E.device, dtype=any_tc.W_E.dtype)
        return torch.empty(0, device=activation_matrix.device, dtype=activation_matrix.dtype)

    return torch.cat(rows, dim=0)

@torch.no_grad()
def select_encoder_bias_lorsa(
    activation_matrix: torch.sparse.Tensor,
    lorsas: LowRankSparseAttention,
) -> torch.Tensor:
    """
    Return encoder bias terms for LoRSA active features only.

    For each layer, gather the bias vector entries corresponding to the
    active heads (same indexing as select_encoder_rows_lorsa). If the
    LoRSA layer has no encoder bias attribute (e.g., b_V), return zeros
    for that layer's active heads to keep alignment with rows.
    """
    rows: List[torch.Tensor] = []

    for layer, row in enumerate(activation_matrix):
        idx2d = row.coalesce().indices()
        if idx2d.numel() == 0:
            continue

        _, head_idx = idx2d  # active head indices for this layer
        lrs = lorsas[layer]

        # Prefer b_V if present; otherwise produce zeros matching dtype/device
        bias_tensor = getattr(lrs, "b_V", None)
        if bias_tensor is None:
            dev, dt = lrs.W_V.device, lrs.W_V.dtype
            layer_bias = torch.zeros(head_idx.numel(), device=dev, dtype=dt)
        else:
            layer_bias = bias_tensor.index_select(0, head_idx.to(device=bias_tensor.device))

        rows.append(layer_bias)

    if not rows:
        # Fall back to an empty tensor on a reasonable device/dtype
        if len(lorsas) > 0:
            any_layer = next(iter(range(len(lorsas))))
            dev, dt = lorsas[any_layer].W_V.device, lorsas[any_layer].W_V.dtype
            return torch.empty(0, device=dev, dtype=dt)
        return torch.empty(0, device=activation_matrix.device, dtype=activation_matrix.dtype)

    return torch.cat(rows, dim=0)

# def compute_partial_influences(edge_matrix, logit_p, row_to_node_index, max_iter=128, device=None):
#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     normalized_matrix = torch.empty_like(edge_matrix, device=device).copy_(edge_matrix)
#     normalized_matrix = normalized_matrix.abs_()
#     normalized_matrix /= normalized_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)

#     influences = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
#     prod = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
#     prod[-len(logit_p) :] = logit_p

#     for _ in range(max_iter):
#         prod = prod[row_to_node_index] @ normalized_matrix
#         if not prod.any():
#             break
#         influences += prod
#     else:
#         raise RuntimeError("Failed to converge")

#     return influences

def compute_partial_influences(edge_matrix, logit_p, row_to_node_index,
                               max_iter=128, device=None, sign_mode="abs"):  # 'abs' | 'signed'
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = edge_matrix.to(device)

    if sign_mode == "abs":
        W = W.abs()
        W = W / W.sum(dim=1, keepdim=True).clamp(min=1e-8)
    elif sign_mode == "signed":
        # print('partial influence computed in signed mode')
        W = W / W.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
    else:
        raise ValueError("sign_mode must be 'abs' or 'signed'")

    influences = torch.zeros(W.shape[1], device=W.device)
    prod = torch.zeros(W.shape[1], device=W.device)
    prod[-len(logit_p):] = logit_p.to(W.device)

    for _ in range(max_iter):
        prod = prod[row_to_node_index.to(W.device)] @ W
        if prod.abs().sum() < 1e-12:  # 收敛/耗尽
            break
        influences += prod

    return influences


def attribute(
    prompt: Union[str, torch.Tensor, List[int]],
    model: ReplacementModel,
    is_castle: bool = False,
    *,
    max_n_logits: int = 10,
    side: str = 'k',                     # 'q' | 'k' | 'both'
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_feature_nodes: Optional[int] = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
    use_legal_moves_only: bool = False,
    fen: Optional[str] = None,
    lboard: Optional[Any] = None,
    move_idx: int | tuple[int, int] | None = None, 
    encoder_demean: bool = False,
    act_times_max: Optional[int] = 60_000_000,
    mongo_client = None,
    sae_series: str = 'lc0-tc',
    analysis_name: str = 'default',
    order_mode: str = 'positive' # ['positive', 'negative', 'move_pair', 'group']
) -> Dict[str, Any]:
    """Compute an attribution graph for *prompt* and return a structured bundle."""
    offload_handles = []


    # ---- 预处理 prompt -> input_ids ----
    # if isinstance(prompt, str):
    #     # 假设模型里有 tokenizer；按你项目的 tokenizer API 微调
    #     input_ids = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    # elif isinstance(prompt, list):
    #     input_ids = torch.tensor([prompt], dtype=torch.long, device=model.device)
    # elif isinstance(prompt, torch.Tensor):
    #     input_ids = prompt.to(model.device)
    #     if input_ids.dim() == 1:
    #         input_ids = input_ids.unsqueeze(0)  # [seq] -> [1, seq]
    #     input_ids = input_ids.long()
    # else:
    #     raise TypeError("prompt must be str | List[int] | torch.Tensor")
    input_ids = prompt
    
    try:
        return _run_attribution(
            model=model,
            prompt=input_ids,
            max_n_logits=max_n_logits,
            side=side,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            offload_handles=offload_handles,
            update_interval=update_interval,
            use_legal_moves_only=use_legal_moves_only,
            fen=fen,
            lboard=lboard,
            is_castle=is_castle,          # ★ 传递
            move_idx=move_idx,            # ★ 传递
            verbose=verbose,              # ★ 传递（当前未使用，可扩展日志）
            encoder_demean = encoder_demean,
            act_times_max = act_times_max,
            mongo_client = mongo_client,
            sae_series = sae_series,
            analysis_name = analysis_name,
            order_mode = order_mode,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()

def _run_attribution(
    model,
    prompt: torch.Tensor,                # 预处理后的 [1, seq]
    max_n_logits: int,
    side: str,                           # 'q' | 'k' | 'both'
    desired_logit_prob: float,
    batch_size: int,
    max_feature_nodes: Optional[int],
    offload: Literal["cpu", "disk", None],
    offload_handles: list,
    update_interval: int = 4,
    use_legal_moves_only: bool = True,
    fen: Optional[str] = None,
    lboard: Optional[Any] = None,
    is_castle: bool = False,             # Phase 3 的 castle_tensor
    move_idx: Optional[int] = None,      # 传给 compute_logit_gradients_wrt_qk
    verbose: bool = False,               # 可用于增强日志
    encoder_demean: bool = False,
    # 过滤相关
    act_times_max: Optional[int] = 60_000_000,
    mongo_client = None,
    sae_series: str = 'lc0-tc',
    analysis_name: str = 'default',
    order_mode: str = 'positive', # ['positive', 'negative', 'move_pair', 'group']
) -> Dict[str, Any]:
    start_time = time.time()

    # ========== 类型检查 ============
    if order_mode == 'move_pair':
        assert isinstance(move_idx, tuple), f"move_idx must be a tuple in movepair mode, now it is {type(move_idx)}"
        move_idx, negative_move_idx = move_idx[0], move_idx[1]
    elif order_mode == 'group':
        assert side =='k', f"side must be k during attributing in the group mode"
        negative_move_idx = None
    else:
        negative_move_idx = None
        
    # ========== Phase 0: 预计算 ==========
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.time()

    input_ids = prompt
    model_out, lorsa_activation_matrix, lorsa_attention_pattern, tc_activation_matrix, error_vecs, token_vecs = model.setup_attribution(
        input_ids, sparse=True
    )
    print("set up attribution! ")
    
    lorsa_decoder_vecs = select_scaled_decoder_vecs_lorsa(lorsa_activation_matrix, model.lorsas)
    lorsa_encoder_rows, lorsa_attention_patterns = select_encoder_rows_lorsa(lorsa_activation_matrix, lorsa_attention_pattern, model.lorsas)
    lorsa_encoder_bias = select_encoder_bias_lorsa(lorsa_activation_matrix, model.lorsas)
    
    tc_decoder_vecs = select_scaled_decoder_vecs_tc(tc_activation_matrix, model.transcoders)
    tc_encoder_rows = select_encoder_rows_tc(tc_activation_matrix, model.transcoders)
    tc_encoder_bias = select_encoder_bias_tc(tc_activation_matrix, model.transcoders)

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
    logger.info(f"Found {tc_activation_matrix._nnz()} active features")

    if offload:
        offload_handles += offload_modules(model.transcoders, offload)

    # ========== Phase 1: 前向 ==========
    logger.info("Phase 1: Running forward pass")
    print("Phase 1: Running forward pass")
    phase_start = time.time()
    with ctx.install_hooks(model):
        residual = model.forward(input_ids, stop_at_layer=model.cfg.n_layers)
        ctx._resid_activations[-1] = residual
        if hasattr(model, 'policy_head'):
            _ = model.policy_head(residual)
    print(f"Forward pass completed in {time.time() - phase_start:.2f}s")
    logger.info(f"Forward pass completed in {time.time() - phase_start:.2f}s")

    if offload:
        offload_handles += offload_modules(
            [block.mlp for block in model.blocks] + [block.attn for block in model.blocks],
            offload,
        )

    # ========== Phase 2: 准备 logit 相关 ==========
    logger.info("Phase 2: Building input vectors")
    phase_start = time.time()

    policy_out = model_out[0]
    n_layers, n_pos, _ = tc_activation_matrix.shape
    total_active_feats = lorsa_activation_matrix._nnz() + tc_activation_matrix._nnz()

    if order_mode == 'group':
        print('compute logit info in group mode')
        logit_idx, logit_p, logit_vecs_q, logit_vecs_k, move_positions, logit_vecs = compute_logit_gradients_wrt_group_k(
            fen=fen,
            logits=policy_out[0],
            model=model,
            residual_input=residual,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            demean=False,
            move_idx=move_idx,      
        )
    else:
        # 计算与 Q/K 的 logit 梯度向量
        logit_idx, logit_p, logit_vecs_q, logit_vecs_k, move_positions, logit_vecs = compute_logit_gradients_wrt_qk(
            fen=fen,
            logits=policy_out[0],
            model=model,
            residual_input=residual,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            demean=False,
            move_idx=move_idx,
        )
    print(f'{move_positions = }')
    
    if negative_move_idx is not None:
        logit_idx_negative, logit_p_negative, logit_vecs_q_negative, logit_vecs_k_negative, move_positions_negative, logit_vecs_negative = compute_logit_gradients_wrt_qk(
            fen=fen,
            logits=policy_out[0],
            model=model,
            residual_input=residual,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            demean=False,
            move_idx=negative_move_idx,
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

    # 预分配容器（q/k 共用）
    edge_matrix_q = torch.zeros(max_feature_nodes + n_logits, total_nodes)
    edge_matrix_k = torch.zeros(max_feature_nodes + n_logits, total_nodes)
    row_to_node_index_q = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)
    row_to_node_index_k = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)

    # ========== Phase 3: logit attribution（写入前 n_logits 行） ==========
    def bias_attr_now(model):
        vals = []
        for name, b in model._get_requires_grad_bias_params():
            if b.grad is not None and 'input' not in name:
                vals.append((b.detach() * b.grad).sum())
        return torch.stack(vals).sum() if vals else b.new_zeros(())

    logger.info("Phase 3: Computing logit attributions")
    phase_start = time.time()
    model.zero_grad(set_to_none=True)

    rows_q_last = None
    rows_k_last = None

    for i in range(0, len(logit_idx), batch_size):
        batch_move_positions = move_positions[i : i + batch_size]
        if order_mode == 'group':
            batch_move_positions_q = batch_move_positions[:,0]
            batch_move_positions_k = batch_move_positions[:,1]
        else:      
            batch_move_positions_k = batch_move_positions[:, 1:2]
            batch_move_positions_q = batch_move_positions[:, 0:1]
            batch_move_positions_q = batch_move_positions_q[0, 0].unsqueeze(0)

        batch_q = logit_vecs_q[i : i + batch_size]
        batch_k = logit_vecs_k[i : i + batch_size]
        
        if negative_move_idx is not None:
            batch_negative_k = logit_vecs_k_negative[i : i + batch_size]
            batch_negative_q = logit_vecs_q_negative[i : i + batch_size]
            batch_k = batch_k - batch_negative_k
            batch_q = batch_q - batch_negative_q
            
            
            batch_move_position_negative = move_positions_negative[i : i + batch_size]
            batch_move_positions_k_negative = batch_move_position_negative[:, 1:2]
            batch_move_positions_q_negative = batch_move_position_negative[:, 0:1]
            batch_move_positions_k = torch.cat([batch_move_positions_k, batch_move_positions_k_negative], dim=1)
            batch_move_positions_q = torch.cat([batch_move_positions_q, batch_move_positions_q_negative], dim=1)

        non_zero_row = (batch_k[0] != 0).any(dim=-1)

        rows_q = ctx.compute_start_end_batch_from_q(
            move_positions=batch_move_positions_q,
            inject_values=batch_q,
        )
        bias_q = bias_attr_now(model)
        model.zero_grad(set_to_none=True)
        non_zero_row = (batch_k[0] != 0).any(dim=-1)
  
        castle_tensor = torch.tensor([[True]]) if is_castle else None
        rows_k = ctx.compute_start_end_batch_from_k(
            move_positions=batch_move_positions_k,
            inject_values=batch_k,
            castle_tensor=castle_tensor,
        )
        bias_k = bias_attr_now(model)

        # 一致性检查
        idx = batch_move_positions[0]
        if negative_move_idx is not None:
            idx_negative = batch_move_position_negative[0]
        print(f'{idx.shape = }, {idx[0] = }, {idx[1] = }')
        if is_castle:
            if idx[1] == 2: idx[1] = 0
            elif idx[1] == 6: idx[1] = 7

        q_row_dots = (ctx._policy_q_activations[0][idx[0]] * logit_vecs_q[0][idx[0]]).sum()
        k_row_dots = (ctx._policy_k_activations[0][idx[1]] * logit_vecs_k[0][idx[1]]).sum()
        
        if negative_move_idx is not None:
            q_row_dots_negative = (ctx._policy_q_activations[0][idx_negative[0]] * logit_vecs_q_negative[0][idx_negative[0]]).sum()
            k_row_dots_negative = (ctx._policy_k_activations[0][idx_negative[1]] * logit_vecs_k_negative[0][idx_negative[1]]).sum()
            # print(f'q_row_dots_negative:{q_row_dots_negative = }')
            # print(f'k_row_dots_negative:{k_row_dots_negative = }')
            assert torch.allclose(bias_q + rows_q[0].sum(), q_row_dots - q_row_dots_negative, rtol=1e-3), f'{bias_q + rows_q[0].sum() = }, {q_row_dots = }, {q_row_dots_negative = }'
            assert torch.allclose(bias_k + rows_k[0].sum(), k_row_dots - k_row_dots_negative, rtol=1e-3), f'{bias_k + rows_k[0].sum() = }, {k_row_dots = }, {k_row_dots_negative = }'
        elif order_mode == 'group':
            print(f'verify in group mode')
            # idx[1]: [pos_k, neg_k1, neg_k2, ...]
            k_pos = idx[1][0]
            k_negs = idx[1][1:]
            k_dot_pos = (ctx._policy_k_activations[0][k_pos] *
                        logit_vecs_k[0][k_pos]).sum()
            k_neg_component = (ctx._policy_k_activations[0].index_select(0, k_negs) *
                            logit_vecs_k[0].index_select(0, k_negs)).sum()

            lhs = bias_k + rows_k[0].sum()
            rhs = k_dot_pos + k_neg_component   # 注意是 +，因为 component 已含负号
            assert torch.allclose(lhs, rhs, rtol=1e-3, atol=1e-6), (lhs, rhs)
        else:
            print(f'{k_row_dots = }')
            assert torch.allclose(bias_q + rows_q[0].sum(), q_row_dots, rtol=1e-3), f'{bias_q + rows_q[0].sum() = }, {q_row_dots = }'
            assert torch.allclose(bias_k + rows_k[0].sum(), k_row_dots, rtol=1e-3), f'{bias_k + rows_k[0].sum() = }, {k_row_dots = }'

        for param in model._get_requires_grad_bias_params():
            param[1].grad = None

        # 写入 logit 行
        bs = batch_q.shape[0]
        edge_matrix_q[i : i + bs, :logit_offset] = rows_q.cpu()
        edge_matrix_k[i : i + bs, :logit_offset] = rows_k.cpu()
        row_to_node_index_q[i : i + bs] = torch.arange(i, i + bs) + logit_offset
        row_to_node_index_k[i : i + bs] = torch.arange(i, i + bs) + logit_offset

        rows_q_last = rows_q  # 暂存最后一个 batch，作为"rows_*"返回
        rows_k_last = rows_k
    logger.info(f"Logit attributions completed in {time.time() - phase_start:.2f}s")

    # ========== Phase 4: feature attribution（按 side） ==========
    logger.info("Phase 4: Computing feature attributions")

    # 逐层均值（用于 encoder_demean）
    with torch.no_grad():
        layer_means: List[torch.Tensor] = []
        for l in range(n_layers):
            tc = model.transcoders[str(l)]
            # W_E: [d_model, d_sae]  ->  W_E.T: [d_sae, d_model]
            mean_vec = tc.W_E.T.mean(dim=0)  # [d_model]
            layer_means.append(mean_vec)
        layer_means = torch.stack(layer_means, dim=0)  # [n_layers, d_model]
        layer_means = layer_means.to(device=tc_encoder_rows.device, dtype=tc_encoder_rows.dtype)

    model.zero_grad(set_to_none=True)
    if hasattr(ctx, 'clear'):
        ctx.clear()
    elif hasattr(ctx, 'reset'):
        ctx.reset()

    # 稀疏索引
    lorsa_feat_layer, lorsa_feat_pos, lorsa_feat_idx = lorsa_activation_matrix.indices()
    tc_feat_layer, tc_feat_pos, tc_feat_idx = tc_activation_matrix.indices()

    # —— 构建允许掩码：True=保留, False=剔除 —— #
    allow_mask = torch.ones(total_active_feats, dtype=torch.bool, device='cpu')
    if mongo_client is not None and act_times_max is not None:
        print('wash dense nodes')
        cache = {}
        
        # 处理TC features
        tc_Ls = tc_feat_layer.cpu().tolist()
        tc_Fs = tc_feat_idx.cpu().tolist()
        tc_offset = lorsa_activation_matrix._nnz()
        
        print(f'{len(tc_Ls) = }, {len(tc_Fs) = }')
        for i, (L, F) in enumerate(zip(tc_Ls, tc_Fs)):
            gid = tc_offset + i  # TC features start after LoRSA features
            key = (int(L), int(F))
            if key not in cache:
                try:
                    sae_name = f"lc0_L{L}M_16x_k30_lr2e-03_auxk_sparseadam"
                    fr = mongo_client.get_feature(sae_name, sae_series, F)
                    at = None
                    if fr:
                        for ana in fr.analyses:
                            if ana.name == analysis_name:
                                at = ana.act_times
                                break
                    cache[key] = at
                except Exception:
                    cache[key] = None
            at = cache[key]
            if at is not None and at > act_times_max:
                allow_mask[gid] = False
    # print(f'{allow_mask = }')
    # 一些打印信息
    masked_idx = (~allow_mask).nonzero(as_tuple=True)[0]          # LongTensor
    num_masked = masked_idx.numel()

    print(f"not allow feature idx: {((~allow_mask).nonzero(as_tuple=True)[0]).tolist()}")
    # if num_masked > 0:
    #     # 2) 保证下标和索引目标在同一 device
    #     if masked_idx.device != tc_feat_layer.device:
    #         masked_idx = masked_idx.to(tc_feat_layer.device)

    #     # 3) 取出对应的 layer / feat（以及 pos 如需）
    #     masked_layers = tc_feat_layer.index_select(0, masked_idx) # 或 tc_feat_layer[masked_idx]
    #     masked_feats  = tc_feat_idx.index_select(0, masked_idx)   # 或 tc_feat_idx[masked_idx]
    #     # 可选：位置
    #     # masked_pos    = tc_feat_pos.index_select(0, masked_idx)

    #     # 4) 打印前若干个
    #     show_k = min(20, num_masked)
    #     print(f'first {show_k} masked gids/layer/feat:')
    #     for i in range(show_k):
    #         gid   = int(masked_idx[i].item())
    #         layer = int(masked_layers[i].item())
    #         feat  = int(masked_feats[i].item())
    #         print(f'  gid={gid:6d}  layer={layer:2d}  feat={feat:6d}')
    # else:
    #     print('no masked features')


    # —— 索引函数（给 run_feature_attribution） —— #
    def idx_to_layer(idx: torch.Tensor) -> torch.Tensor:
        is_lorsa = idx < len(lorsa_feat_layer)
        return torch.where(
            is_lorsa.to(lorsa_feat_layer.device),
            2 * lorsa_feat_layer[idx * is_lorsa],
            2 * tc_feat_layer[(idx - len(lorsa_feat_layer)) * ~is_lorsa] + 1
        )

    def idx_to_pos(idx: torch.Tensor) -> torch.Tensor:
        is_lorsa = idx < len(lorsa_feat_layer)
        return torch.where(
            is_lorsa.to(lorsa_feat_pos.device),
            lorsa_feat_pos[idx * is_lorsa],
            tc_feat_pos[(idx - len(lorsa_feat_layer)) * ~is_lorsa]
        )

    def idx_to_feature_id(idx: torch.Tensor) -> torch.Tensor:
        is_lorsa = (idx < len(lorsa_feat_layer))
        l_idx = (idx * is_lorsa).to(torch.long)
        t_idx = ((idx - len(lorsa_feat_layer)) * (~is_lorsa)).to(torch.long)

        return torch.where(
            is_lorsa.to(lorsa_feat_layer.device),
            lorsa_feat_idx[l_idx],
            tc_feat_idx[t_idx],
        )

    def idx_to_encoder_rows(idx: torch.Tensor) -> torch.Tensor:
        is_lorsa = idx < len(lorsa_feat_layer)
        rows = torch.where(
            is_lorsa.to(lorsa_encoder_rows.device)[:, None],
            lorsa_encoder_rows[idx * is_lorsa],
            tc_encoder_rows[(idx - len(lorsa_feat_layer)) * ~is_lorsa]
        )
        if encoder_demean:
            # Apply demean only to TC features
            layers = torch.where(
                is_lorsa.to(tc_feat_layer.device),
                torch.zeros_like(tc_feat_layer[0]),  # dummy for LoRSA
                tc_feat_layer[(idx - len(lorsa_feat_layer)) * ~is_lorsa]
            ).to(torch.long)     # [B]
            means = layer_means.index_select(0, layers)    # [B, d_model]
            # Only subtract mean for TC features
            means = torch.where(
                is_lorsa.to(means.device)[:, None],
                torch.zeros_like(means),
                means
            )
            rows = rows - means
        return rows

    def idx_to_encoder_bias(idx: torch.Tensor) -> torch.Tensor:
        is_lorsa = (idx < len(lorsa_feat_layer))
        l_idx = (idx * is_lorsa).to(torch.long)
        t_idx = ((idx - len(lorsa_feat_layer)) * (~is_lorsa)).to(torch.long)

        return torch.where(
            is_lorsa.to(lorsa_encoder_bias.device),
            lorsa_encoder_bias[l_idx],
            tc_encoder_bias[t_idx],
        )

    def idx_to_pattern(idx: torch.Tensor) -> torch.Tensor:
        is_lorsa = idx < len(lorsa_feat_layer)
        res = torch.where(
            is_lorsa.to(lorsa_attention_patterns.device)[:, None],
            lorsa_attention_patterns[idx * is_lorsa],
            torch.nn.functional.one_hot(
                tc_feat_pos[(idx - len(lorsa_feat_layer)) * ~is_lorsa],
                num_classes=n_pos
            )
        )
        return res

    def idx_to_activation_values(idx: torch.Tensor) -> torch.Tensor:
        is_lorsa = idx < len(lorsa_feat_layer)
        if is_lorsa.squeeze().item():
            return lorsa_activation_matrix.values()[idx]
        else:
            local_idx = (idx - len(lorsa_feat_layer)).to(torch.long)
            layer = tc_feat_layer[local_idx]
            feat_idx = tc_feat_idx[local_idx]

            if torch.is_tensor(layer):
                layer_key = str(int(layer.item()))
            else:
                layer_key = str(int(layer))

            tc = model.transcoders[layer_key]

            # 若无 b_E，用 0 占位
            b_E = getattr(tc, "b_E", None)
            if b_E is None:
                bias_val = 0.0
            else:
                bias_val = b_E[feat_idx.to(device=b_E.device, dtype=torch.long)]

            return tc_activation_matrix.values()[local_idx] - bias_val

    fa_result = run_feature_attribution(
        side=side,
        ctx=ctx,
        model=model,
        tc_activation_matrix=tc_activation_matrix,
        total_active_feats=total_active_feats,
        max_feature_nodes=max_feature_nodes,
        update_interval=update_interval,
        batch_size=batch_size,
        n_logits=n_logits,
        logit_p=logit_p,
        logit_offset=logit_offset,
        idx_to_layer=idx_to_layer,
        idx_to_pos=idx_to_pos,
        idx_to_encoder_rows=idx_to_encoder_rows,
        idx_to_encoder_bias=idx_to_encoder_bias,
        idx_to_pattern=idx_to_pattern,
        compute_partial_influences=compute_partial_influences,  # 假设在外层已定义
        bias_attr_now=bias_attr_now,
        edge_matrix_q=edge_matrix_q,
        row_to_node_index_q=row_to_node_index_q,
        edge_matrix_k=edge_matrix_k,
        row_to_node_index_k=row_to_node_index_k,
        logger=logger,
        order_mode = order_mode,
    )

    # ========== Phase 5: 打包（每个 side） ==========
    def package_side(
        visited: torch.Tensor,
        edge_matrix: torch.Tensor,
        row_to_node_index: torch.Tensor,
        *,
        allow_mask: Optional[torch.Tensor] = None,   # ★ 新增：允许的 feature（gid 级）
    ) -> Dict[str, Any]:
        total_nodes = logit_offset + n_logits

        # 1) 选中的 feature（全局 gid），并在"列"层面过滤掉不允许的
        if max_feature_nodes < total_active_feats:
            selected_features = torch.where(visited)[0].to(edge_matrix.device)
        else:
            selected_features = torch.arange(total_active_feats, device=edge_matrix.device)

        # print(f'{selected_features.shape = }') # 1024
            
        if allow_mask is not None:
            am = allow_mask.to(device=selected_features.device, dtype=torch.bool)
            keep_cols = am.index_select(0, selected_features)
            selected_features = selected_features[keep_cols]
            
        non_feature_nodes = torch.arange(total_active_feats, total_nodes, device=edge_matrix.device)
        col_read = torch.cat([selected_features, non_feature_nodes], dim=0)

        # 应用列选择（此处就已经把不允许的 feature 整列移除了）
        edge_matrix_read = edge_matrix[:, col_read]

        # 2) 行排序：先把行按 row_to_node_index 的自然顺序排好（稳定），然后再把"允许的 feature 行"提前
        sort_idx = row_to_node_index.argsort()
        edge_matrix_sorted = edge_matrix_read.index_select(0, sort_idx)
        r2n_sorted = row_to_node_index.index_select(0, sort_idx)

        # 标出 feature 行与 logit 行
        is_feature_row = (r2n_sorted < total_active_feats)
        is_logit_row   = (r2n_sorted >= logit_offset)

        # 计算 feature 行对应的 gid，并据 allow_mask 过滤"行"
        if allow_mask is not None:
            feat_row_gids = r2n_sorted.masked_select(is_feature_row).to(torch.long)    # [n_feature_rows_sorted]
            allow_on_rows = allow_mask.to(r2n_sorted.device, dtype=torch.bool).index_select(0, feat_row_gids)
        else:
            # 全允许
            allow_on_rows = torch.ones(int(is_feature_row.sum().item()), dtype=torch.bool, device=r2n_sorted.device)

        # 在"已排序"的坐标系里拿到：允许的 feature 行索引、不允许的 feature 行索引、logit 行索引
        feat_row_idx_sorted = torch.nonzero(is_feature_row, as_tuple=True)[0]          # 全部 feature 行（已排序坐标）
        allow_feat_rows_sorted = feat_row_idx_sorted[allow_on_rows]                    # 允许的 feature 行
        deny_feat_rows_sorted  = feat_row_idx_sorted[~allow_on_rows]                   # 不允许的 feature 行
        logit_rows_sorted      = torch.nonzero(is_logit_row, as_tuple=True)[0]         # logit 行

        # 目标：让前 max_feature_nodes 行全是"允许的 feature 行"
        # 先构造一个新的行置换：允许的 feature 行 -> 不允许的 feature 行 -> 其它（这里仅 logit 行）
        # 这样 edge_matrix_perm[:max_feature_nodes] 一定是允许的 feature 行（若数量足够）
        row_perm = torch.cat([allow_feat_rows_sorted, deny_feat_rows_sorted, logit_rows_sorted], dim=0)

        edge_matrix_perm = edge_matrix_sorted.index_select(0, row_perm)
        r2n_perm = r2n_sorted.index_select(0, row_perm)

        # debugging
        n_feature_rows_total = int(is_feature_row.sum().item())
        n_feature_rows_allowed = int(allow_feat_rows_sorted.numel())
        n_feature_cols_selected = int(selected_features.numel())

        print(f"[dbg] feature rows: total={n_feature_rows_total}, allowed={n_feature_rows_allowed}, "
            f"selected feature cols={n_feature_cols_selected}")

        if allow_mask is not None:
            present_feat_gids = torch.unique(r2n_sorted[is_feature_row].to(torch.long))
            allowed_feat_gids = torch.nonzero(allow_mask, as_tuple=True)[0].to(present_feat_gids.device)
            missing = allowed_feat_gids[~torch.isin(allowed_feat_gids, present_feat_gids)]
            print(f"[dbg] allowed-but-no-row gids (allowed in mask but absent as rows): {missing.numel()}")
        # end of debugging   
        
        

        # 统计可用的允许行数，决定实际填充几行
        allowed_rows_available = int(min(max_feature_nodes, allow_feat_rows_sorted.numel()))
        if allowed_rows_available < max_feature_nodes:
            print(f"[info] allowed feature rows = {allow_feat_rows_sorted.numel()}, "
                f"less than max_feature_nodes={max_feature_nodes}; "
                f"top block will use {allowed_rows_available} rows.")

        # 3) 组装方阵（列数 = 节点总数；行我们只把"允许的 feature 行（最多 K）+ 全部 logit 行"填进去）
        final_node_count = edge_matrix_perm.shape[1]
        full_edge_matrix = torch.zeros(
            final_node_count, final_node_count,
            device=edge_matrix_perm.device, dtype=edge_matrix_perm.dtype
        )

        # 顶部：允许的 feature 行（至多 K 行）
        if allowed_rows_available > 0:
            full_edge_matrix[:allowed_rows_available] = edge_matrix_perm[:allowed_rows_available]

        # 底部：logit 行（从置换后矩阵里单独取出，保证不受上面重排影响）
        full_edge_matrix[-n_logits:] = edge_matrix_perm.index_select(0, logit_rows_sorted)

        # 4) 同步返回"已置换后的" row_to_node_index，确保 DFS 能按新行序正确解码 gid
        row_to_node_index_final = r2n_perm.clone()

        # 元信息里记录一下实际使用的"有意义的 feature 行数"
        meta = {
            "n_logits": int(n_logits),
            "logit_offset": int(logit_offset),
            "final_node_count": int(final_node_count),
            "max_feature_rows": int(allowed_rows_available),   # ★ 实际填入的允许行数
            "filtered_feature_cols": int(selected_features.numel()),
        }

        return {
            "selected_features": selected_features,       # 已过滤（列）
            "col_read": col_read,
            "edge_matrix": edge_matrix_perm,              # 行已重排
            "row_to_node_index": row_to_node_index_final, # 与 edge_matrix 同步
            "full_edge_matrix": full_edge_matrix,         # 方阵（上=允许的 feature 行，下=logit 行）
            "meta": meta,
        }

    packaged_q = None
    packaged_k = None
    if 'q' in fa_result:
        packaged_q = package_side(
            visited=fa_result['q']['visited'],
            edge_matrix=fa_result['q']['edge_matrix'],
            row_to_node_index=fa_result['q']['row_to_node_index'],
            allow_mask=allow_mask,   # ★
        )
    if 'k' in fa_result:
        packaged_k = package_side(
            visited=fa_result['k']['visited'],
            edge_matrix=fa_result['k']['edge_matrix'],
            row_to_node_index=fa_result['k']['row_to_node_index'],
            allow_mask=allow_mask,   # ★
        )

    # —— 同步对 rows_q / rows_k 做掩码（不让被剔除的 gid 参与根选择） —— #
    # 如果没有 rows_*（n_logits=0），就置为 0 长
    if rows_q_last is None:
        rows_q_last = torch.zeros(1, logit_offset, device=tc_feat_layer.device)
    if rows_k_last is None:
        rows_k_last = torch.zeros(1, logit_offset, device=tc_feat_layer.device)

    rows_q_raw = rows_q_last
    rows_k_raw = rows_k_last

    mask_float = allow_mask.to(dtype=rows_q_raw.dtype, device=rows_q_raw.device).view(1, -1)
    rows_q_filtered = rows_q_raw.clone()
    rows_k_filtered = rows_k_raw.clone()
    # 仅屏蔽 feature 段（0..total_active_feats-1）
    rows_q_filtered[:, :total_active_feats] *= mask_float
    rows_k_filtered[:, :total_active_feats] *= mask_float
    # ========== 统一返回 ==========
    graph_bundle = {
        "meta": {
            "time_sec": float(time.time() - start_time),
            "side": side,
            "verbose": verbose,
            "use_legal_moves_only": use_legal_moves_only,
            "offload": offload,
        },
        "input": {
            "input_ids": prompt,
            "input_embedding": token_vecs,  # 添加input_embedding (hook_embed)
        },
        "logits": {
            "indices": logit_idx,
            "probabilities": logit_p,
            "move_positions": move_positions,
            "n_logits": int(n_logits),
        },
        "dims": {
            "n_layers": int(n_layers),
            "n_pos": int(n_pos),
            "logit_offset": int(logit_offset),
            "total_active_feats": int(total_active_feats),
            "max_feature_nodes": int(max_feature_nodes),
        },
        "tc_activations": {
            "indices": tc_activation_matrix.indices().T,   # [nnz, 3]
            "values": tc_activation_matrix.values(),       # [nnz]
            "tc_activation_matrix": tc_activation_matrix,      
        },
        "q": packaged_q,   # 或 None
        "k": packaged_k,   # 或 None

        # rows_*：已过滤版（供 DFS 选根）；同时附带 raw 方便调试
        "rows_q": rows_q_filtered,
        "rows_k": rows_k_filtered,
        "rows_q_raw": rows_q_raw,
        "rows_k_raw": rows_k_raw,

        # 便于下游调试/复用
        "feature_allow_mask": allow_mask,
    }

    return graph_bundle



    # graph = Graph(
    #     input_string=model.tokenizer.decode(input_ids),
    #     input_tokens=input_ids,
    #     logit_tokens=logit_idx,
    #     logit_probabilities=logit_p,
    #     lorsa_active_features=lorsa_activation_matrix.indices().T,
    #     lorsa_activation_values=lorsa_activation_matrix.values(),
    #     clt_active_features=tc_activation_matrix.indices().T,
    #     clt_activation_values=tc_activation_matrix.values(),
    #     selected_features=selected_features,
    #     adjacency_matrix=full_edge_matrix,
    #     cfg=model.cfg,
    #     scan=None,
    # )

    # total_time = time.time() - start_time
    # logger.info(f"Attribution completed in {total_time:.2f}s")

    # return graph


def run_feature_attribution(
    *,
    side: str,                           # 'q' | 'k' | 'both'
    ctx,
    model,
    tc_activation_matrix: torch.Tensor,
    total_active_feats: int,
    max_feature_nodes: int,
    update_interval: int,
    batch_size: int,
    n_logits: int,
    logit_p: torch.Tensor,               # 长度 = n_logits
    logit_offset: int,
    # 这些函数/映射需在外部已定义
    idx_to_layer,
    idx_to_pos,
    idx_to_encoder_rows,
    idx_to_encoder_bias,
    idx_to_pattern,
    compute_partial_influences,
    bias_attr_now,
    # 为 q / k 分别提供缓冲矩阵与行映射（会被原地写入）
    edge_matrix_q: torch.Tensor,
    row_to_node_index_q: torch.Tensor,
    edge_matrix_k: torch.Tensor,
    row_to_node_index_k: torch.Tensor,
    logger=None,
    order_mode: str = 'positive'
) -> dict:
    """
    通过 side 控制仅计算 q 或 k，或两者都算。
    返回: {"q": {...}, "k": {...}}（按请求的 side 填充）
    每个 side 的返回值:
      - visited: [total_active_feats] 的 bool 张量（哪些 feature 被访问/入列）
      - edge_matrix: 计算后（行=feature+logit，列=所有节点）的矩阵（原地同传入对象）
      - row_to_node_index: 计算后（行→全局 gid）的映射（原地同传入对象）
    """
    rank_logits_signed = (order_mode == 'negative')
    if rank_logits_signed is True:
        print('order: from most negative')
    
    def _phase(side_tag: str,
               edge_matrix: torch.Tensor,
               row_to_node_index: torch.Tensor,
               desc: str):
        nonlocal ctx, model

        if logger:
            logger.info(f"Phase: Computing feature attributions in {side_tag}")

        print("清空 ctx 中的计算状态…")
        model.zero_grad(set_to_none=True)
        if hasattr(ctx, 'clear'):
            ctx.clear()
        elif hasattr(ctx, 'reset'):
            ctx.reset()

        phase_start = time.time()
        st = n_logits  # 行起点：先放 logit 行

        visited = torch.zeros(total_active_feats, dtype=torch.bool)
        n_visited = 0

        pbar = tqdm(total=max_feature_nodes, desc=desc)

        feature_descending: bool = not rank_logits_signed
        influence_sign_mode = "signed" if rank_logits_signed else "abs"  # 需要你把 compute_partial_influences 加上这个开关

        while n_visited < max_feature_nodes:
            if max_feature_nodes == total_active_feats:
                pending = torch.arange(total_active_feats)
            else:
                influences = compute_partial_influences(
                    edge_matrix[:st],
                    logit_p,
                    row_to_node_index[:st],
                    sign_mode=influence_sign_mode,   # <-- 保留正负号时用 "signed"
                )
                feature_rank = torch.argsort(
                    influences[:total_active_feats],
                    descending=feature_descending     # <-- 统一用这个布尔量
                ).cpu()
                queue_size = min(update_interval * batch_size, max_feature_nodes - n_visited)
                pending = feature_rank[~visited[feature_rank]][:queue_size]

            queue = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]

            for idx_batch in queue:
                if idx_batch.numel() == 0:
                    continue
                
                n_visited += len(idx_batch)
                # print(f'{idx_batch = }')

                # ------- 准备注入 -------
                layers = idx_to_layer(idx_batch)
                positions = idx_to_pos(idx_batch)
                inject_values = idx_to_encoder_rows(idx_batch).detach()
                encoder_bias = idx_to_encoder_bias(idx_batch)
                attn_patterns = idx_to_pattern(idx_batch)
                if isinstance(attn_patterns, torch.Tensor):
                    attn_patterns = attn_patterns.detach()

                model.zero_grad(set_to_none=True)

                # 还有未完成批次则保留图
                has_more_in_this_phase = (n_visited < max_feature_nodes)
                rows_feature = ctx.compute_batch(
                    layers=layers,
                    positions=positions,
                    inject_values=inject_values,
                    attention_patterns=attn_patterns,
                    retain_graph=has_more_in_this_phase,
                )

                _ = bias_attr_now(model) + encoder_bias

                # ------- 写回矩阵 -------
                bs = rows_feature.shape[0]
                end = st + bs
                edge_matrix[st:end, :logit_offset] = rows_feature.detach().cpu()
                row_to_node_index[st:end] = idx_batch
                visited[idx_batch] = True
                st = end
                pbar.update(len(idx_batch))

        pbar.close()
        print(f"[{side_tag}] Feature attributions completed in {time.time() - phase_start:.2f}s")

        # 把当前 side 的结果返回
        return {
            "visited": visited,                     # [total_active_feats] bool
            "edge_matrix": edge_matrix,             # 原地对象
            "row_to_node_index": row_to_node_index  # 原地对象
        }

    side = side.lower()
    out: dict = {}
    if side in ('q', 'both'):
        out['q'] = _phase('q', edge_matrix_q, row_to_node_index_q, desc="Feature influence computation (q)")
    if side in ('k', 'both'):
        out['k'] = _phase('k', edge_matrix_k, row_to_node_index_k, desc="Feature influence computation (k)")
        
        
    out['tc_activation_matrix'] = tc_activation_matrix
        
    return out

# def extract_feature_subgraph(
#     graph_bundle: Dict[str, Any],
#     target_feature_gid: int,
#     *,
#     max_depth: int = 3,
#     min_edge_weight: float = 0.0,
#     include_self: bool = True,
#     side: str = 'k',  # 'q', 'k', 'both'
#     verbose: bool = False,
# ) -> Dict[str, Any]:
#     """
#     从完整的attribution graph中提取指定feature的子图
    
#     Args:
#         graph_bundle: 完整的attribution graph bundle，来自attribute()函数的输出
#         target_feature_gid: 目标feature的全局ID (gid)
#         max_depth: 最大搜索深度，控制子图的大小
#         min_edge_weight: 最小边权重阈值，过滤掉权重太小的边
#         include_self: 是否包含目标feature本身
#         side: 选择使用哪个side的图 ('q', 'k', 'both')
#         verbose: 是否打印详细信息
        
#     Returns:
#         Dict[str, Any]: 包含子图信息的字典
#             - subgraph: 子图的邻接矩阵
#             - node_mapping: 从子图节点索引到原图节点索引的映射
#             - reverse_mapping: 从原图节点索引到子图节点索引的映射
#             - target_feature_idx: 目标feature在子图中的索引
#             - meta: 元信息
#     """
    
#     def _extract_side_subgraph(side_data: Dict[str, Any], side_name: str) -> Dict[str, Any]:
#         """为单个side提取子图，逻辑与package_side保持一致"""
#         if side_data is None:
#             return None
            
#         # 获取与package_side相同的数据结构
#         edge_matrix = side_data['edge_matrix']  # 行已重排的矩阵
#         row_to_node_index = side_data['row_to_node_index']  # 与edge_matrix同步的行映射
#         selected_features = side_data['selected_features']  # 已过滤的feature gids
#         full_edge_matrix = side_data['full_edge_matrix']  # 方阵
#         meta = side_data['meta']
        
#         n_rows = edge_matrix.shape[0]
#         n_nodes = edge_matrix.shape[1]
        
#         # 检查目标feature是否在selected_features中
#         if target_feature_gid not in selected_features:
#             if verbose:
#                 print(f"Warning: Target feature {target_feature_gid} not found in {side_name} selected_features")
#             return None
            
#         # 找到目标feature在row_to_node_index中的行索引
#         target_row_idx = None
#         for i, node_idx in enumerate(row_to_node_index):
#             if node_idx == target_feature_gid:
#                 target_row_idx = i
#                 break
                
#         if target_row_idx is None:
#             if verbose:
#                 print(f"Warning: Target feature {target_feature_gid} not found as a row in {side_name} graph")
#             return None
            
#         # 使用BFS找到相关的节点（基于edge_matrix）
#         visited_nodes = set()
#         queue = [(target_row_idx, 0)]  # (row_idx, depth)
        
#         # 添加进度条
#         max_possible_visits = min(n_rows, max_depth * (n_rows + n_nodes))  # 估算最大访问次数
#         with tqdm(total=max_possible_visits, desc=f"BFS search ({side_name})", disable=not verbose) as pbar:
#             while queue:
#                 current_row, depth = queue.pop(0)
                
#                 if current_row in visited_nodes or depth > max_depth:
#                     continue
                    
#                 visited_nodes.add(current_row)
#                 pbar.update(1)
#                 pbar.set_postfix({
#                     'depth': depth,
#                     'visited': len(visited_nodes),
#                     'queue_size': len(queue)
#                 })
                
#                 if depth < max_depth:
#                     # 检查出边（当前行影响的其他节点）
#                     for j in range(n_nodes):
#                         if edge_matrix[current_row, j] > min_edge_weight:
#                             # 找到对应的行索引
#                             for i, node_idx in enumerate(row_to_node_index):
#                                 if node_idx == j and i not in visited_nodes:
#                                     queue.append((i, depth + 1))
#                                     break
                    
#                     # 检查入边（影响当前行的其他节点）
#                     for i in range(n_rows):
#                         if edge_matrix[i, current_row] > min_edge_weight:
#                             if i not in visited_nodes:
#                                 queue.append((i, depth + 1))
        
#         # 转换为有序列表
#         visited_rows_list = sorted(list(visited_nodes))
        
#         if not include_self and target_row_idx in visited_rows_list:
#             visited_rows_list.remove(target_row_idx)
            
#         if not visited_rows_list:
#             if verbose:
#                 print(f"Warning: No nodes found for target feature {target_feature_gid} in {side_name} graph")
#             return None
            
#         # 创建子图（基于edge_matrix的行和列）
#         subgraph = edge_matrix[visited_rows_list][:, visited_rows_list]
        
#         # 创建行映射
#         row_mapping = torch.tensor(visited_rows_list, dtype=torch.long)
#         reverse_row_mapping = torch.full((n_rows,), -1, dtype=torch.long)
#         reverse_row_mapping[row_mapping] = torch.arange(len(visited_rows_list))
        
#         # 找到目标feature在子图中的索引
#         target_subgraph_idx = reverse_row_mapping[target_row_idx].item()
        
#         # 获取对应的row_to_node_index
#         subgraph_row_to_node = row_to_node_index[row_mapping]
        
#         # 统计信息
#         n_subgraph_nodes = len(visited_rows_list)
#         n_subgraph_edges = (subgraph > min_edge_weight).sum().item()
        
#         if verbose:
#             print(f"{side_name} subgraph: {n_subgraph_nodes} nodes, {n_subgraph_edges} edges")
#             print(f"Target feature {target_feature_gid} at index {target_subgraph_idx}")
        
#         return {
#             'subgraph': subgraph,
#             'row_mapping': row_mapping,
#             'reverse_row_mapping': reverse_row_mapping,
#             'row_to_node_index': subgraph_row_to_node,
#             'target_feature_idx': target_subgraph_idx,
#             'meta': {
#                 'n_nodes': n_subgraph_nodes,
#                 'n_edges': n_subgraph_edges,
#                 'max_depth': max_depth,
#                 'min_edge_weight': min_edge_weight,
#                 'target_feature_gid': target_feature_gid,
#                 'original_meta': meta,
#             }
#         }
    
#     # 根据side参数选择要处理的图
#     result = {}
    
#     if side in ('q', 'both') and 'q' in graph_bundle:
#         result['q'] = _extract_side_subgraph(graph_bundle['q'], 'q')
        
#     if side in ('k', 'both') and 'k' in graph_bundle:
#         result['k'] = _extract_side_subgraph(graph_bundle['k'], 'k')
    
#     # 添加一些全局信息
#     result['target_feature_gid'] = target_feature_gid
#     result['extraction_params'] = {
#         'max_depth': max_depth,
#         'min_edge_weight': min_edge_weight,
#         'include_self': include_self,
#         'side': side,
#     }
    
#     return result

def find_feature_gid(attribution_result, layer, feature_id, position):
    tc_activations = attribution_result['tc_activations']
    indices = tc_activations['indices']  # [nnz, 3]
    values = tc_activations['values']    # [nnz]
    mask = (indices[:, 0] == layer) & (indices[:, 1] == position) & (indices[:, 2] == feature_id)
    if mask.any():
        matching_idx = mask.nonzero(as_tuple=True)[0][0]
        gid = matching_idx.item()
        activation_value = values[matching_idx].item()
        return gid, activation_value
    else:
        return None, None