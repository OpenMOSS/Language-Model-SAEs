"""
Low Rank Sparse Attention SAE Implementation

This module implements a Low Rank Sparse Attention layer as a Sparse Autoencoder,
inheriting from AbstractSparseAutoEncoder and supporting head parallelization.
"""

import math
from torch._tensor import Tensor
from torch._tensor import Tensor
from torch._tensor import Tensor
from typing import Dict, Optional, Tuple, Union, Any, Literal, overload, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.distributed.tensor
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from transformer_lens.hook_points import HookPoint
from typing_extensions import override
import einops
from jaxtyping import Float, Int
from transformer_lens.components.leela_encoder import SmolGen
from .abstract_sae import AbstractSparseAutoEncoder
from .config import LorsaConfig
from .utils.distributed import DimMap
from .utils.logging import get_distributed_logger
from .utils.timer import timer

logger = get_distributed_logger("lorsa")


class SmolGenLorsa(nn.Module):
    """SmolGen module adapted for LORSA with configurable n_qk_heads and d_qk_head."""
    
    def __init__(self, d_model: int, n_qk_heads: int, d_qk_head: int, n_ctx: int):
        super().__init__()
        self.d_model = d_model
        self.n_qk_heads = n_qk_heads
        self.d_qk_head = d_qk_head
        self.n_ctx = n_ctx
        
        # Compress input to smaller dimension
        self.compress = nn.Linear(d_model, 32, bias=False)
        
        # Calculate the flattened size after compression
        compressed_size = 32 * n_ctx  # 32 * 64 = 2048
        
        # Dense layers for processing
        self.dense1 = nn.Linear(compressed_size, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dense2 = nn.Linear(256, n_qk_heads * 256)  # Output size for n_qk_heads
        self.ln2 = nn.LayerNorm(n_qk_heads * 256)
        
        # Generate attention weights for each head
        self.smol_weight_gen = nn.Linear(256, n_ctx * n_ctx, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, n_ctx, d_model]
        Returns:
            weights: [batch_size, n_qk_heads, n_ctx, n_ctx]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compress input
        compressed = self.compress(x)  # [batch_size, n_ctx, 32]
        
        # Flatten for dense processing
        x_flat = compressed.view(batch_size, -1)  # [batch_size, 32 * n_ctx]
        
        # Dense layers
        x = self.dense1(x_flat)  # [batch_size, 256]
        x = F.silu(x)
        x = self.ln1(x)
        
        x = self.dense2(x)  # [batch_size, n_qk_heads * 256]
        x = F.silu(x)
        x = self.ln2(x)
        
        # Reshape for per-head processing
        x = x.view(batch_size, self.n_qk_heads, 256)  # [batch_size, n_qk_heads, 256]
        
        # Generate attention weights for each head
        weights = self.smol_weight_gen(x)  # [batch_size, n_qk_heads, n_ctx * n_ctx]
        weights = weights.view(batch_size, self.n_qk_heads, self.n_ctx, self.n_ctx)
        
        return weights

class LowRankSparseAttention(AbstractSparseAutoEncoder):
    def __init__(self, cfg: LorsaConfig, device_mesh: Optional[DeviceMesh] = None):
        super().__init__(cfg, device_mesh=device_mesh)
        self.cfg = cfg
        
        if self.cfg.attn_scale is None:
            self.cfg.attn_scale = self.cfg.d_qk_head ** 0.5
        assert self.cfg.attn_scale is not None
    
        if device_mesh is None:
            # Local parameters
            def _get_param_with_shape(shape: tuple[int, ...]) -> nn.Parameter:
                return nn.Parameter(
                    torch.empty(
                        shape,
                        dtype=self.cfg.dtype,
                        device=self.cfg.device,
                    )
                )
            
            self.W_Q = _get_param_with_shape((self.cfg.n_qk_heads, self.cfg.d_model, self.cfg.d_qk_head))
            self.W_K = _get_param_with_shape((self.cfg.n_qk_heads, self.cfg.d_model, self.cfg.d_qk_head))
            self.W_V = _get_param_with_shape((self.cfg.n_ov_heads, self.cfg.d_model))
            self.W_O = _get_param_with_shape((self.cfg.n_ov_heads, self.cfg.d_model))
            self.b_Q = _get_param_with_shape((self.cfg.n_qk_heads, self.cfg.d_qk_head))
            self.b_K = _get_param_with_shape((self.cfg.n_qk_heads, self.cfg.d_qk_head))
            self.b_V = _get_param_with_shape((self.cfg.n_ov_heads,))
            if self.cfg.use_decoder_bias:
                self.b_D = _get_param_with_shape((self.cfg.d_model,))
        else:
            # Distributed parameters with head sharding
            dim_maps = self.dim_maps()
            def _get_param_with_shape(shape: tuple[int, ...], placements: Sequence[Any]) -> nn.Parameter:
                return nn.Parameter(
                    torch.distributed.tensor.empty(
                        shape,
                        dtype=self.cfg.dtype,
                        device_mesh=device_mesh,
                        placements=placements,
                    )
                )
            self.W_Q = _get_param_with_shape((self.cfg.n_qk_heads, self.cfg.d_model, self.cfg.d_qk_head), placements=dim_maps["W_Q"].placements(device_mesh))
            self.W_K = _get_param_with_shape((self.cfg.n_qk_heads, self.cfg.d_model, self.cfg.d_qk_head), placements=dim_maps["W_K"].placements(device_mesh))
            self.W_V = _get_param_with_shape((self.cfg.n_ov_heads, self.cfg.d_model), placements=dim_maps["W_V"].placements(device_mesh))
            self.W_O = _get_param_with_shape((self.cfg.n_ov_heads, self.cfg.d_model), placements=dim_maps["W_O"].placements(device_mesh))
            self.b_Q = _get_param_with_shape((self.cfg.n_qk_heads, self.cfg.d_qk_head), placements=dim_maps["b_Q"].placements(device_mesh))
            self.b_K = _get_param_with_shape((self.cfg.n_qk_heads, self.cfg.d_qk_head), placements=dim_maps["b_K"].placements(device_mesh))
            self.b_V = _get_param_with_shape((self.cfg.n_ov_heads,), placements=dim_maps["b_V"].placements(device_mesh))
            if self.cfg.use_decoder_bias:
                self.b_D = _get_param_with_shape((self.cfg.d_model,), placements=dim_maps["b_D"].placements(device_mesh))
        
         # Attention mask
        mask = torch.tril(
            torch.ones(
                (self.cfg.n_ctx, self.cfg.n_ctx),
                device=self.cfg.device,
                dtype=self.cfg.dtype,
            ).bool(),
        )
        if self.device_mesh is not None:
            mask = DimMap({}).distribute(mask, self.device_mesh)
        self.register_buffer("mask", mask)

        if self.device_mesh is not None:
            IGNORE = DimMap({}).distribute(torch.tensor(-torch.inf, device=self.cfg.device), self.device_mesh)
        else:
            IGNORE = torch.tensor(-torch.inf, device=self.cfg.device)
        self.register_buffer("IGNORE", IGNORE)

        if self.cfg.positional_embedding_type == "rotary":
            print("use rotary positional embedding !!!!!")
            # Applies a rotation to each two-element chunk of keys and queries pre dot producting to bake in relative position.
            if self.cfg.rotary_dim is None:  # keep mypy happy
                raise ValueError("Rotary dim must be provided for rotary positional embeddings")
            sin, cos = self._calculate_sin_cos_rotary(
                self.cfg.rotary_dim,
                self.cfg.n_ctx,
                base=self.cfg.rotary_base,
                dtype=self.cfg.dtype,
                device=self.cfg.device,
            )
            if self.device_mesh is not None:
                sin = DimMap({}).distribute(sin, self.device_mesh)
                cos = DimMap({}).distribute(cos, self.device_mesh)
            self.register_buffer("rotary_sin", sin)
            self.register_buffer("rotary_cos", cos)

        if self.cfg.use_smolgen:
            self.smolgen = SmolGenLorsa(
                d_model=self.cfg.d_model,
                n_qk_heads=self.cfg.n_qk_heads,
                d_qk_head=self.cfg.d_qk_head,
                n_ctx=self.cfg.n_ctx
            ).to(self.cfg.device)
            self.register_buffer("smolgen_score_scale", torch.tensor(1.0))

        if self.cfg.use_learnable_attn_scale:
            # 创建可学习的 attn_scale 参数
            if device_mesh is None:
                self.attn_scale = nn.Parameter(
                    torch.tensor(
                        self.cfg.attn_scale,  # 使用配置中的初始值
                        dtype=self.cfg.dtype,
                        device=self.cfg.device,
                    )
                )
            else:
                # 分布式情况下的处理
                self.attn_scale = nn.Parameter(
                    torch.distributed.tensor.empty(
                        (),
                        dtype=self.cfg.dtype,
                        device_mesh=device_mesh,
                        placements=[],
                    )
                )
                self.attn_scale.data.fill_(self.cfg.attn_scale)

        # Initialize dead latents tracking buffers after all parameters are set up
        if self.cfg.use_auxk and self.cfg.act_fn == "topk":
            if device_mesh is None:
                self.register_buffer('tokens_since_last_activation', torch.zeros(self.cfg.d_sae, device=cfg.device, dtype=torch.long))
                self.register_buffer('is_dead', torch.zeros(self.cfg.d_sae, device=cfg.device, dtype=torch.bool))
            else:
                self.register_buffer('tokens_since_last_activation', torch.distributed.tensor.zeros(
                    self.cfg.d_sae,
                    dtype=torch.long,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["tokens_since_last_activation"].placements(device_mesh),
                ))
                self.register_buffer('is_dead', torch.distributed.tensor.zeros(
                    self.cfg.d_sae,
                    dtype=torch.bool,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["is_dead"].placements(device_mesh),
                ))
    
    
    def init_parameters(self, **kwargs):
        """Initialize parameters."""
        super().init_parameters(**kwargs)

        torch.nn.init.xavier_uniform_(self.W_Q)
        torch.nn.init.xavier_uniform_(self.W_K)
        
        W_V_bound = 1 / math.sqrt(self.cfg.d_sae)
        torch.nn.init.uniform_(self.W_V, -W_V_bound, W_V_bound)

        W_O_bound = 1 / math.sqrt(self.cfg.d_model)
        torch.nn.init.uniform_(self.W_O, -W_O_bound, W_O_bound)

        torch.nn.init.zeros_(self.b_Q)
        torch.nn.init.zeros_(self.b_K)
        torch.nn.init.zeros_(self.b_V)

        if self.cfg.use_decoder_bias:
            torch.nn.init.zeros_(self.b_D)
    
        # 对于smolgen的初始化
        if hasattr(self, "smolgens"):
            for sm in self.smolgens:
                for mod in sm.modules():
                    if isinstance(mod, nn.Linear):
                        if mod.bias is not None:
                            nn.init.zeros_(mod.bias)
                        nn.init.xavier_uniform_(mod.weight)
                    elif isinstance(mod, nn.LayerNorm):
                        nn.init.constant_(mod.weight, 1.0)
                        nn.init.zeros_(mod.bias)
    
    @torch.no_grad()
    def init_lorsa_with_mhsa(self, encoder_layer):
        """Initialize Lorsa with Original Multi Head Sparse Attention for LeelaChess"""
        assert self.cfg.n_qk_heads % encoder_layer.n_heads == 0
        input_norm_factor = math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_in]
        output_norm_factor = math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_out]
        
        qk_exp_factor = self.cfg.n_qk_heads // encoder_layer.n_heads
        ov_exp_factor = self.cfg.n_ov_heads // encoder_layer.n_heads
        
        d_head = encoder_layer.d_model // encoder_layer.n_heads
        
        q_weights = encoder_layer.mha.q_proj.weight.view(encoder_layer.d_model, encoder_layer.n_heads, d_head).permute(1, 0, 2)
        k_weights = encoder_layer.mha.k_proj.weight.view(encoder_layer.d_model, encoder_layer.n_heads, d_head).permute(1, 0, 2)
        
        expanded_q = torch.repeat_interleave(q_weights, qk_exp_factor, dim=0).to(self.cfg.dtype)
        expanded_k = torch.repeat_interleave(k_weights, qk_exp_factor, dim=0).to(self.cfg.dtype)
        def _first10_param(p):
            try:
                return p.detach().reshape(-1)[:10]
            except Exception:
                return p.reshape(-1)[:10]
        print(f"[mhsa init pre-scale] W_Q expanded first10: {_first10_param(expanded_q)}")
        print(f"[mhsa init pre-scale] W_K expanded first10: {_first10_param(expanded_k)}")
        self.W_Q.data = nn.Parameter(expanded_q / input_norm_factor)
        self.W_K.data = nn.Parameter(expanded_k / input_norm_factor)
 
        # Initialize biases b_Q and b_K from encoder biases (or zeros), with pre-scale print
        if encoder_layer.mha.q_proj.bias is not None:
            q_bias = encoder_layer.mha.q_proj.bias.view(encoder_layer.n_heads, d_head)
        else:
            q_bias = torch.zeros((encoder_layer.n_heads, d_head), device=self.cfg.device, dtype=self.cfg.dtype)
        if encoder_layer.mha.k_proj.bias is not None:
            k_bias = encoder_layer.mha.k_proj.bias.view(encoder_layer.n_heads, d_head)
        else:
            k_bias = torch.zeros((encoder_layer.n_heads, d_head), device=self.cfg.device, dtype=self.cfg.dtype)

        expanded_bq = torch.repeat_interleave(q_bias, qk_exp_factor, dim=0).to(self.cfg.dtype)
        expanded_bk = torch.repeat_interleave(k_bias, qk_exp_factor, dim=0).to(self.cfg.dtype)
        print(f"[mhsa init pre-scale] b_Q expanded first10: {_first10_param(expanded_bq)}")
        print(f"[mhsa init pre-scale] b_K expanded first10: {_first10_param(expanded_bk)}")
        self.b_Q.data = (expanded_bq / input_norm_factor)
        self.b_K.data = (expanded_bk / input_norm_factor)
        
        # if self.cfg.use_post_qk_ln and self.cfg.normalization_type == 'RMS':
        #     pass
        
    
    @override
    @torch.no_grad()
    def init_W_D_with_active_subspace(self, activation_batch: dict[str, torch.Tensor], d_active_subspace: int):
        """Initialize W_D with the active subspace.
        
        Args:
            activation_batch: The activation batch.
            d_active_subspace: The dimension of the active subspace.
        """
        label = self.prepare_label(activation_batch)
        flattened_label = label.flatten(0, 1)
        demeaned_label = flattened_label - flattened_label.mean(dim=0)
        U, S, V = torch.svd(demeaned_label.T.to(torch.float32))
        proj_weight = U[:, :d_active_subspace] # [d_model, d_active_subspace]
        self.W_O.data = self.W_O.data[:, :d_active_subspace] @ proj_weight.T.to(self.cfg.dtype)


    # 用于测试
    @torch.no_grad()
    def init_lorsa_W_D_with_mhsa(self, encoder_layer):  # 一般不用这个
        """Initialize Lorsa W_V and W_O with Original Multi Head Sparse Attention for LeelaChess"""
        assert self.cfg.n_ov_heads % encoder_layer.n_heads == 0
        input_norm_factor = math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_in]
        output_norm_factor = math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_out]
        
        ov_exp_factor = self.cfg.n_ov_heads // encoder_layer.n_heads
        
        def _first10_param(p):
            try:
                return p.detach().reshape(-1)[:10]
            except Exception:
                return p.reshape(-1)[:10]

        v_proj_weight = encoder_layer.mha.v_proj.weight  # [d_model, n_heads * d_head]
        d_head = encoder_layer.d_model // encoder_layer.n_heads
        v_weights_reshaped = v_proj_weight.view(
            encoder_layer.d_model, encoder_layer.n_heads, d_head
        ).permute(1, 0, 2)  # [n_heads, d_model, d_head]

        v_weights_for_lorsa = []
        for head_idx in range(encoder_layer.n_heads):
            head_v_weight = v_weights_reshaped[head_idx]  # [d_model, d_head]
            if d_head == encoder_layer.d_model:
                v_weights_for_lorsa.append(head_v_weight)
            else:
                repeat_count = encoder_layer.d_model // d_head
                v_weights_for_lorsa.append(head_v_weight.repeat_interleave(repeat_count, dim=0))
        
        self.W_V.data = torch.stack(v_weights_for_lorsa, dim=0) # [n_ov_heads, d_model]

        o_proj_weight = encoder_layer.mha.out_proj.weight  # [d_model, n_heads * d_head]
        o_weights_reshaped = o_proj_weight.view(
            encoder_layer.d_model, encoder_layer.n_heads, d_head
        ).permute(1, 0, 2) # [n_heads, d_model, d_head]

        o_weights_for_lorsa = []
        for head_idx in range(encoder_layer.n_heads):
            head_o_weight = o_weights_reshaped[head_idx]  # [d_model, d_head]
            if d_head == encoder_layer.d_model:
                o_weights_for_lorsa.append(head_o_weight)
            else:
                repeat_count = encoder_layer.d_model // d_head
                o_weights_for_lorsa.append(head_o_weight.repeat_interleave(repeat_count, dim=0))
        self.W_O.data = torch.stack(o_weights_for_lorsa, dim=0) # [n_ov_heads, d_model]
        if encoder_layer.mha.v_proj.bias is not None:
            v_bias = encoder_layer.mha.v_proj.bias.view(encoder_layer.n_heads, d_head)
        else:
            v_bias = torch.zeros((encoder_layer.n_heads, d_head), device=self.cfg.device, dtype=self.cfg.dtype)
        
        expanded_bv = torch.repeat_interleave(v_bias, ov_exp_factor, dim=0).to(self.cfg.dtype)
        print(f"[ov init pre-scale] b_V expanded first10: {_first10_param(expanded_bv)}")
        self.b_V.data = expanded_bv / input_norm_factor
        if self.cfg.use_decoder_bias:
            if encoder_layer.mha.out_proj.bias is not None:
                o_bias = encoder_layer.mha.out_proj.bias
            else:
                o_bias = torch.zeros(encoder_layer.d_model, device=self.cfg.device, dtype=self.cfg.dtype)
            print(f"[ov init pre-scale] b_D first10: {_first10_param(o_bias)}")
            self.b_D.data = o_bias / output_norm_factor



    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def init_W_D_with_active_subspace_per_head(self, activation_batch: dict[str, torch.Tensor], encoder_layer):
        """
        Initialize W_D with the active subspace for each head.
        """
        x, _ = self.prepare_input(activation_batch)
        
        d_model = self.W_O.shape[1]

        leela_d_head = encoder_layer.mha.d_k
        leela_n_heads = encoder_layer.mha.n_heads
        
        leela_W_O = encoder_layer.mha.out_proj.weight.view(leela_n_heads, leela_d_head, d_model)  # [n_head, d_head, d_model]

        leela_W_V = encoder_layer.mha.v_proj.weight.view(d_model, leela_n_heads, leela_d_head)  # [d_model, n_head, d_head]
        leela_W_V = leela_W_V.permute(1, 0, 2)    # [n_head, d_model, d_head]
                
        captured_z = None
        def capture_hook(z, hook):
            nonlocal captured_z
            captured_z = z.clone().detach()
            return z
        handle = encoder_layer.mha.hook_z.add_hook(capture_hook)
        _ = encoder_layer.forward(x)
        # print(f'{captured_z.shape = }') #[256, 24, 64, 32]
        # print(f'{leela_W_O.shape = }') #[32, n_heads, d_model]
        output_per_head = torch.einsum('b n s h, n h d -> b s n d', captured_z, leela_W_O)
        n_ov_per_orig_head = self.cfg.n_ov_heads // encoder_layer.n_heads
        for orig_head_index in range(encoder_layer.n_heads):
            output = output_per_head[:, :, orig_head_index, :]
            output_flattened = output.flatten(0, 1)
            demeaned_output = output_flattened - output_flattened.mean(dim=0)
            U, S, V = torch.svd(demeaned_output.T.to(torch.float32))
            proj_weight = U[:, :self.cfg.d_qk_head]
            self.W_O.data[orig_head_index*n_ov_per_orig_head:(orig_head_index+1)*n_ov_per_orig_head] = (
                self.W_O.data[orig_head_index*n_ov_per_orig_head:(orig_head_index+1)*n_ov_per_orig_head, :self.cfg.d_qk_head] @
                proj_weight.T
            )
            self.W_V.data[orig_head_index*n_ov_per_orig_head:(orig_head_index+1)*n_ov_per_orig_head] = (
                self.W_O.data[orig_head_index*n_ov_per_orig_head:(orig_head_index+1)*n_ov_per_orig_head] @
                (leela_W_V[orig_head_index] @ leela_W_O[orig_head_index]).T
            )
        self.W_V.data = self.W_V.data / self.W_V.data.norm(dim=1, keepdim=True)
        self.W_O.data = self.W_O.data / self.W_O.data.norm(dim=1, keepdim=True)
        
        feature_acts, hidden_pre = self.encode(x, return_hidden_pre=True)
        hidden_pre = hidden_pre.flatten(0, 1)
        self.b_V.data = -hidden_pre.mean(dim=0)


    # # TODO:搞下leela的mhsa,把这些适配下
    # @torch.no_grad()
    # def init_W_D_with_active_subspace_per_head(self, activation_batch: dict[str, torch.Tensor], encoder_layer):
    #     """
    #     Initialize W_D with the active subspace for each head.
    #     Adapted for LeelaChess model structure.
    #     """
    #     x, _ = self.prepare_input(activation_batch)
    #     captured_z = None
        
    #     def capture_hook(z, hook):
    #         nonlocal captured_z
    #         captured_z = z.clone().detach()
    #         return z

    #     handle = encoder_layer.hook_z.add_hook(capture_hook)
    #     _ = encoder_layer(x)
    
    #     d_head = encoder_layer.d_model // encoder_layer.n_heads
    #     out_proj_weights = []
        
    #     for head_idx in range(encoder_layer.n_heads):
    #         start_dim = head_idx * d_head
    #         end_dim = start_dim + d_head
    #         head_out_proj = encoder_layer.mha.out_proj.weight[:, start_dim:end_dim].T
    #         out_proj_weights.append(head_out_proj)
        
    #     mhsa_W_O = torch.stack(out_proj_weights, dim=0)  # [n_heads, d_head, d_model]
        
    #     # 使用原始代码的 einsum 方式（captured_z: [b, n, s, h]）
    #     # print(f'{captured_z.shape = }, {mhsa_W_O.shape = }') # 例如 [32, 24, 64, 32] 与 [24, 32, 768]
    #     output_per_head = torch.einsum('b n s h, n h d -> b s n d', captured_z, mhsa_W_O)
        
    #     n_ov_per_orig_head = self.cfg.n_ov_heads // encoder_layer.n_heads
        
    #     for orig_head_index in range(encoder_layer.n_heads):
    #         output = output_per_head[:, :, orig_head_index, :].to(torch.float32)
    #         output_flattened = output.flatten(0, 1)
    #         demeaned_output = output_flattened - output_flattened.mean(dim=0)
            
    #         U, S, V = torch.svd(demeaned_output.T)
    #         proj_weight = U[:, :self.cfg.d_qk_head].to(self.W_O.device, dtype=torch.float32)
            
    #         start_idx = orig_head_index * n_ov_per_orig_head
    #         end_idx = (orig_head_index + 1) * n_ov_per_orig_head
    #         # 用活动子空间投影更新 W_O
    #         self.W_O.data[start_idx:end_idx] = (
    #             self.W_O.data[start_idx:end_idx, :self.cfg.d_qk_head].to(torch.float32) @ proj_weight.T
    #         ).to(self.W_O.dtype)
            
    #         # 从 LeelaChess 的 v_proj 中提取每个头的权重（列切块）
    #         start_dim = orig_head_index * d_head
    #         end_dim = start_dim + d_head
    #         v_proj_head = encoder_layer.mha.v_proj.weight[:, start_dim:end_dim]  # [d_model, d_head]
    #         # 组合权（与注释版一致的方向）：C = W_V^{(n)} @ W_O^{(n)} -> [d_model, d_model]
    #         W_O_head   = mhsa_W_O[orig_head_index]                                   # [d_head, d_model]
    #         combined_weight = v_proj_head.to(torch.float32) @ W_O_head.to(torch.float32)  # [d_model, d_model]
            
    #         # 用组合权更新 W_V：W_V_block = W_O_block @ C^T
    #         self.W_V.data[start_idx:end_idx] = (
    #             self.W_O.data[start_idx:end_idx].to(torch.float32) @ combined_weight.T
    #         ).to(self.W_V.dtype)
        
    #     # 行归一化
    #     self.W_V.data = self.W_V.data / self.W_V.data.norm(dim=1, keepdim=True)
    #     self.W_O.data = self.W_O.data / self.W_O.data.norm(dim=1, keepdim=True)
    #     def _first10_param(p):
    #         try:
    #             return p.detach().reshape(-1)[:10]
    #         except Exception:
    #             return p.reshape(-1)[:10]
    #     print(f"[active_subspace per-head] W_V first10: {_first10_param(self.W_V)}")
    #     print(f"[active_subspace per-head] W_O first10: {_first10_param(self.W_O)}")
        
    #     feature_acts, hidden_pre = self.encode(x, return_hidden_pre=True)
    #     hidden_pre = hidden_pre.flatten(0, 1)
    #     # Initialize b_V using encoder pre-activation mean
    #     self.b_V.data = -hidden_pre.mean(dim=0)
    #     # Initialize decoder bias b_D to minimize mean residual: adjust by mean(label - reconstructed)
    #     if getattr(self.cfg, 'use_decoder_bias', False):
    #         label = self.prepare_label(activation_batch)
    #         reconstructed = self.decode(feature_acts)
    #         b_d_delta = (label - reconstructed).mean(dim=(0, 1)).to(self.b_D.dtype)
    #         self.b_D.data = (self.b_D.data + b_d_delta)
    #     # Print biases first 10 values
    #     def _first10_param(p):
    #         try:
    #             return p.detach().reshape(-1)[:10]
    #         except Exception:
    #             return p.reshape(-1)[:10]
    #     print(f"[active_subspace per-head] b_V first10: {_first10_param(self.b_V)}")
    #     if getattr(self.cfg, 'use_decoder_bias', False):
    #         print(f"[active_subspace per-head] b_D first10: {_first10_param(self.b_D)}")

    def _calculate_sin_cos_rotary(
        self,
        rotary_dim: int,
        n_ctx: int,
        base: int = 10000,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> Tuple[
        Float[torch.Tensor, "n_ctx rotary_dim"],
        Float[torch.Tensor, "n_ctx rotary_dim"],
    ]:
        """
        Calculate the sine and cosine waves to use in a rotary embedding. See https://blog.eleuther.ai/rotary-embeddings/ for details

        Note: For some inexplicable reason, in GPT-J each ADJACENT pair of elements in k and q are rotated, in GPT-NeoX the pair of elements at k and k+n//2 are rotated (ie folding the full length in half, and then looking at pairs accordingly). I have absolutely no clue why, it should be completely equivalent.
        To resolve this, I've coded it to default to the GPT-J mode, but to explicitly check whether it's GPT-NeoX and then do the GPT-NeoX thing if it is.
        """
        high_precision = torch.float32 if dtype != torch.float64 else torch.float64
        pos = torch.arange(n_ctx, dtype=high_precision, device=device)
        dim = torch.arange(rotary_dim // 2, dtype=high_precision, device=device)

        # Llama-3.1 uses NTK-by-Parts Rotary Embedding introduced in Section 3.2 in https://arxiv.org/pdf/2309.00071
        # Implementation copied from https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/modeling_rope_utils.py#L310
        if self.cfg.use_NTK_by_parts_rope:
            inv_freq = 1.0 / (
                base ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64, device=device).float() / rotary_dim)
            )
            factor = self.cfg.NTK_by_parts_factor
            low_freq_factor = self.cfg.NTK_by_parts_low_freq_factor
            high_freq_factor = self.cfg.NTK_by_parts_high_freq_factor
            old_context_len = self.cfg.old_context_len

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            wavelen = 2 * math.pi / inv_freq
            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            smoothed_inv_freq = (
                1 - smooth_factor
            ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
            freq = 1 / inv_freq_llama
        else:
            freq = base ** (dim / (rotary_dim / 2))
        if self.cfg.rotary_adjacent_pairs:
            freq = einops.repeat(freq, "d -> (d 2)")
        else:
            freq = einops.repeat(freq, "d -> (2 d)")
        # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
        angles = pos[:, None] / freq[None, :]
        return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)

    @override
    def encoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Norm of encoder (Q/K weights)."""
        if not isinstance(self.W_V, DTensor):
            return torch.norm(self.W_V, p=2, dim=1, keepdim=keepdim).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            return DTensor.from_local(
                torch.norm(self.W_V.to_local(), p=2, dim=1, keepdim=keepdim),
                device_mesh=self.device_mesh,
                placements=self.dim_maps()["W_V"].placements(self.device_mesh),
            )

    @override
    def decoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Norm of decoder (O weights)."""
        if not isinstance(self.W_O, DTensor):
            return torch.norm(self.W_O, p=2, dim=1, keepdim=keepdim).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            return DTensor.from_local(
                torch.norm(self.W_O.to_local(), p=2, dim=1, keepdim=keepdim),
                device_mesh=self.device_mesh,
                placements=self.dim_maps()["W_O"].placements(self.device_mesh),
            )

    @override
    def decoder_bias_norm(self) -> torch.Tensor:
        """Norm of decoder bias."""
        if not self.cfg.use_decoder_bias:
            raise ValueError("Decoder bias not used")
        return torch.norm(self.b_D, p=2, dim=0, keepdim=True)

    @override
    @torch.no_grad()
    def transform_to_unit_decoder_norm(self):
        """Transform to unit decoder norm."""
        norm = self.decoder_norm(keepdim=True)
        self.W_O /= norm
        self.W_V *= norm.unsqueeze(-1)

    @override
    @torch.no_grad()
    def standardize_parameters_of_dataset_norm(self):
        """Standardize parameters for dataset norm."""
        assert self.cfg.norm_activation == "dataset-wise"
        assert self.dataset_average_activation_norm is not None

        hook_point_in = self.cfg.hook_point_in
        hook_point_out = self.cfg.hook_point_out

        input_norm_factor = math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[hook_point_in]
        output_norm_factor = math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[hook_point_out]

        self.W_Q.data *= input_norm_factor
        self.W_K.data *= input_norm_factor 

        self.W_O.data = self.W_O.data * input_norm_factor / output_norm_factor
        self.b_D.data = self.b_D.data / output_norm_factor
        # self.W_V.data *= input_norm_factor
        self.b_V.data /= input_norm_factor

        if hasattr(self, 'smolgen') and self.cfg.use_smolgen:
            self.smolgen.compress.weight.data *= input_norm_factor
        
        self.cfg.norm_activation = "inference"

    @override
    @torch.no_grad()
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0):
        raise NotImplementedError(
            "It does not make sense to init the encoder with the decoder transpose for Lorsa"
        )
    
    @overload
    def encode(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
        return_hidden_pre: Literal[False] = False,
        **kwargs,
    ) -> Float[torch.Tensor, "batch seq_len d_sae"]: ...

    @overload
    def encode(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
        return_hidden_pre: Literal[True],
        **kwargs,
    ) -> Tuple[
        Float[torch.Tensor, "batch seq_len d_sae"],
        Float[torch.Tensor, "batch seq_len d_sae"],
    ]: ...

    @override
    def encode(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
        return_hidden_pre: bool = False,
        return_attention_pattern: bool = False,
        **kwargs
    ) -> Union[
        Float[torch.Tensor, "batch seq_len d_sae"],
        Tuple[
            Float[torch.Tensor, "batch seq_len d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        Tuple[
            Float[torch.Tensor, "batch seq_len d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
            Float[torch.Tensor, "batch n_qk_heads q_pos k_pos"],
        ],
    ]:
        """Encode to sparse head activations."""
        # Compute Q, K, V
        q, k, v = self._compute_qkv(x)
        
        # Prepare tensors for attention math
        query = q.permute(0, 2, 1, 3)   # [B, H, S, d]
        key = k.permute(0, 2, 1, 3)     # [B, H, S, d]
        value = v.reshape(*k.shape[:3], -1).permute(0, 2, 1, 3)  # [B, H, S, dv]

        # Optional SmolGen additive bias as attention mask: [B, H, S, S]
        attn_mask = None
        if getattr(self.cfg, "use_smolgen", True):
            if hasattr(self, "smolgen"):
                bias = self.smolgen(x)
                # Normalize to [B,H,S,S] using repeat_interleave to align head dim to n_qk_heads
                if bias.dim() == 3:
                    # [B,S,S] -> [B,H,S,S]
                    bias = bias.unsqueeze(1).repeat_interleave(self.cfg.n_qk_heads, dim=1)
                elif bias.dim() == 4:
                    # [B,1,S,S] or [B,H,S,S]
                    h = bias.shape[1]
                    if h == 1:
                        bias = bias.repeat_interleave(self.cfg.n_qk_heads, dim=1)
                    elif h == self.cfg.n_qk_heads:
                        pass
                    else:
                        # If heads divide evenly, tile by factor
                        if self.cfg.n_qk_heads % h == 0:
                            factor = self.cfg.n_qk_heads // h
                            bias = bias.repeat_interleave(factor, dim=1)
                        else:
                            raise ValueError(f"SmolGen output head dim {h} cannot be expanded to n_qk_heads {self.cfg.n_qk_heads}")
                else:
                    raise ValueError("SmolGen output must be [B,S,S] or [B,1,S,S] or [B,H,S,S]")
                attn_mask = bias.to(query.dtype) * float(self.smolgen_score_scale.item())
            else:
                print('no SmolGen')
                attn_mask = None

        # Combine scale - match encoder behavior (LORSA uses division so we pass inverse into SDPA)
        if hasattr(self, 'attn_scale'):
            scale = 1 / self.attn_scale.item()
        else:
            scale = 1 / self.cfg.attn_scale

        # Compute attention outputs using fused kernels
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            z = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,  # None or [B,H,S,S]
                scale=scale,
                is_causal=False,
                enable_gqa=True,
            )
        hidden_pre = z.permute(0, 2, 1, 3).reshape(*v.shape)

        # Post-activation feature acts
        # modified here
        # originally: feature_acts = self.activation_function(hidden_pre)
        # print(f'{self.activation_function(hidden_pre) = }')
        feature_acts = hidden_pre * self.activation_function(hidden_pre)

        # Optionally compute attention pattern for return
        pattern: Optional[torch.Tensor] = None
        if return_attention_pattern:
            # raw scores: [B,H,S,S]
            raw_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            if attn_mask is not None:
                raw_scores = raw_scores + attn_mask
            pattern = torch.softmax(raw_scores, dim=-1)
        

        # Assemble return values according to flags
        if return_hidden_pre and return_attention_pattern:
            assert pattern is not None
            return feature_acts, hidden_pre, pattern
        elif return_hidden_pre:
            return feature_acts, hidden_pre
        elif return_attention_pattern:
            assert pattern is not None
            return feature_acts, pattern
        else:
            return feature_acts

    @torch.no_grad()
    def init_attn_scale_from_encoder(self, encoder_layer: nn.Module):
        """Initialize this LORSA's attn_scale from an existing encoder's attn_scale."""
        if getattr(self.cfg, "use_learnable_attn_scale", False):
            return
        if not hasattr(self, "attn_scale"):
            raise ValueError("This LORSA has no attn_scale to initialize.")
        if not hasattr(encoder_layer.mha, "qk_scale"):
            raise ValueError("Encoder layer has no qk_scale to initialize from.")

        self.attn_scale.data = 1.0 / encoder_layer.mha.qk_scale.data


    @torch.no_grad()
    def init_smolgen_from_encoder(self, encoder_layer: nn.Module):
        """Initialize this LORSA's SmolGen module from an existing encoder's SmolGen."""
        if not getattr(self.cfg, "use_smolgen", False):
            return
        if not hasattr(self, "smolgen"):
            raise ValueError("This LORSA has no SmolGen to initialize.")
        if not hasattr(encoder_layer, "smolgen"):
            raise ValueError("Encoder layer has no SmolGen to initialize from.")
        
        input_norm_factor = math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_in]
        
        # 复制 SmolGen 的状态
        qk_exp_factor = self.cfg.n_qk_heads // encoder_layer.n_heads
        
        # 复制基础层的权重
        self.smolgen.compress.load_state_dict(encoder_layer.smolgen.compress.state_dict())
        self.smolgen.dense1.load_state_dict(encoder_layer.smolgen.dense1.state_dict())
        self.smolgen.ln1.load_state_dict(encoder_layer.smolgen.ln1.state_dict())
        
        original_dense2_weight = encoder_layer.smolgen.dense2.weight  # [6144, 256]
        original_dense2_bias = encoder_layer.smolgen.dense2.bias  # [6144]
        expanded_dense2_weight = torch.repeat_interleave(
            original_dense2_weight, qk_exp_factor, dim=0
        ).to(self.cfg.dtype)
        expanded_dense2_bias = torch.repeat_interleave(
            original_dense2_bias, qk_exp_factor, dim=0
        ).to(self.cfg.dtype)
        
        self.smolgen.dense2.weight.data = expanded_dense2_weight
        self.smolgen.dense2.bias.data = expanded_dense2_bias
        
        # 同样处理ln2
        original_ln2_weight = encoder_layer.smolgen.ln2.weight  # [6144]
        original_ln2_bias = encoder_layer.smolgen.ln2.bias  # [6144]
        
        expanded_ln2_weight = torch.repeat_interleave(
            original_ln2_weight, qk_exp_factor, dim=0
        ).to(self.cfg.dtype)
        expanded_ln2_bias = torch.repeat_interleave(
            original_ln2_bias, qk_exp_factor, dim=0
        ).to(self.cfg.dtype)
        
        self.smolgen.ln2.weight.data = expanded_ln2_weight
        self.smolgen.ln2.bias.data = expanded_ln2_bias
        self.smolgen.smol_weight_gen.load_state_dict(encoder_layer.smolgen.smol_weight_gen.state_dict())
        
        # Debug: print first 10 values of each SmolGen parameter before scaling

        # print(f"[smolgen init pre-scale] compress.W: {_first10_param(self.smolgen.compress.weight)}")
        # if getattr(self.smolgen.compress, 'bias', None) is not None:
        #     print(f"[smolgen init pre-scale] compress.b: {_first10_param(self.smolgen.compress.bias)}")
        # print(f"[smolgen init pre-scale] dense1.W: {_first10_param(self.smolgen.dense1.weight)}")
        # if getattr(self.smolgen.dense1, 'bias', None) is not None:
        #     print(f"[smolgen init pre-scale] dense1.b: {_first10_param(self.smolgen.dense1.bias)}")
        # print(f"[smolgen init pre-scale] ln1.weight: {_first10_param(self.smolgen.ln1.weight)}")
        # print(f"[smolgen init pre-scale] ln1.bias: {_first10_param(self.smolgen.ln1.bias)}")
        # print(f"[smolgen init pre-scale] dense2.W: {_first10_param(self.smolgen.dense2.weight)}")
        # if getattr(self.smolgen.dense2, 'bias', None) is not None:
        #     print(f"[smolgen init pre-scale] dense2.b: {_first10_param(self.smolgen.dense2.bias)}")
        # print(f"[smolgen init pre-scale] ln2.weight: {_first10_param(self.smolgen.ln2.weight)}")
        # print(f"[smolgen init pre-scale] ln2.bias: {_first10_param(self.smolgen.ln2.bias)}")
        # print(f"[smolgen init pre-scale] smol_weight_gen.W: {_first10_param(self.smolgen.smol_weight_gen.weight)}")
        # if getattr(self.smolgen.smol_weight_gen, 'bias', None) is not None:
        #     print(f"[smolgen init pre-scale] smol_weight_gen.b: {_first10_param(self.smolgen.smol_weight_gen.bias)}")
        
        self.smolgen_score_scale.data = self.smolgen_score_scale.data * (input_norm_factor ** 2)

    @override
    def decode(self, feature_acts, **kwargs):
        """Decode head activations to output."""
        if feature_acts.layout == torch.sparse_coo:
            return torch.sparse.mm(
                feature_acts.to(torch.float32),
                self.W_O.to(torch.float32),
            ).to(self.cfg.dtype) + self.b_D
        out = torch.einsum("bps,sd->bpd", feature_acts, self.W_O)
        if self.cfg.use_decoder_bias:
            out = out + self.b_D
        if isinstance(out, DTensor):
            out = out.full_tensor()
        if self.cfg.skip_bos:
            out = out[:, 1:]
        return out

    def _compute_qkv(self, x: Float[torch.Tensor, "batch seq_len d_model"]) -> tuple[
        Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
        Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
        Float[torch.Tensor, "batch seq_len n_ov_heads"],
    ]:
        """Compute queries, keys, values."""
        q = torch.einsum(
            "bsd,Qdq->bsQq",
            x,
            self.W_Q,
        ) + self.b_Q
        k = torch.einsum(
            "bsd,Kdk->bsKk",
            x,
            self.W_K,
        ) + self.b_K
        v = torch.einsum(
            "bsd,Vd->bsV",
            x,
            self.W_V,
        ) + self.b_V
        
        # Apply positional embedding
        if self.cfg.positional_embedding_type == "rotary":
            q = self._apply_rotary(q)
            k = self._apply_rotary(k)

        return q, k, v
    
    def encode_z_pattern_for_head(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
        head_idx: Int[torch.Tensor, "n_active_features"],
    ) -> Float[torch.Tensor, "n_active_features k_pos"]:
        assert x.size(0) == 1, f"x must be of shape (1, seq_len, d_model), but got {x.shape}"
        qk_idx: Tensor = head_idx // (self.cfg.n_ov_heads // self.cfg.n_qk_heads)
        q, k, v = self._compute_qkv(x)
        # (n_active_features, q_pos, k_pos)
        pattern = self._compute_attention_pattern(q, k)[qk_idx, 0]
        return pattern.mul_(v[0, :, head_idx, None].permute(1, 2, 0))
    
    def _apply_rotary(
        self,
        x: Float[torch.Tensor, "batch pos head_index d_head"],
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        # x = x.repeat(1, 1, 1, self.cfg.rotary_scale)
        x = x.repeat_interleave(self.cfg.rotary_scale, dim=-1)
        
        x_pos = x.size(1)
        x_rot = x[:, :, :, :self.cfg.rotary_dim]
        x_pass = x[:, :, :, self.cfg.rotary_dim :]
        x_flip = self._rotate_every_two(x_rot)

        rotary_cos = self.rotary_cos[  # type: ignore
            None, : x_pos, None, :
        ]
        rotary_sin = self.rotary_sin[  # type: ignore
            None, : x_pos, None, :
        ]
        x_rotated = x_rot * rotary_cos + x_flip * rotary_sin

        return torch.cat([x_rotated, x_pass], dim=-1)
    
    def _rotate_every_two(
        self, x: Float[torch.Tensor, "... rotary_dim"]
    ) -> Float[torch.Tensor, "... rotary_dim"]:
        """
        Rotary helper function, splits x into blocks of size 2 along the final axis and maps [x0, x1] to [-x1, x0]

        The final axis of x must have even length.

        GPT-NeoX and GPT-J do rotary subtly differently, see calculate_sin_cos_rotary for details.
        """
        rot_x = x.clone()
        if self.cfg.rotary_adjacent_pairs:
            rot_x[..., ::2] = -x[..., 1::2]
            rot_x[..., 1::2] = x[..., ::2]
        else:
            n = x.size(-1) // 2
            rot_x[..., :n] = -x[..., n:]
            rot_x[..., n:] = x[..., :n]

        return rot_x

    def _compute_attention_pattern(
        self,
        q: Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
        k: Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
    ) -> Float[torch.Tensor, "n_qk_heads batch q_pos k_pos"]:
        """Compute attention patterns."""
        q = q.permute(2, 0, 1, 3) # (n_qk_heads, batch, seq_len, d_qk_head)
        k = k.permute(2, 0, 3, 1) # (n_qk_heads, batch, d_qk_head, seq_len)
        # torch.cuda.synchronize()
        scores = torch.einsum("sbqd,sbdk->sbqk", q, k) / self.attn_scale.data
        # scores = self._apply_causal_mask(scores)
        # torch.cuda.synchronize()
        return F.softmax(scores, dim=-1)

    def _compute_head_outputs(
        self,
        pattern: Float[torch.Tensor, "n_qk_heads batch q_pos k_pos"],
        v: Float[torch.Tensor, "batch seq_len n_ov_heads"],
    ) -> Float[torch.Tensor, "batch n_qk_heads d_qk_head seq_len"]:
        """Compute head outputs from pattern and values."""
        v = einops.rearrange(v, "b seq h -> b h seq")
        if self.cfg.n_qk_heads != self.cfg.n_ov_heads:
            # (batch, n_qk_heads, d_qk_head, seq_len)
            v = v.view(v.shape[0], self.cfg.n_qk_heads, self.cfg.n_ov_heads // self.cfg.n_qk_heads, v.shape[2])
            v = v.permute(1, 0, 2, 3) # (n_qk_heads, batch, d_qk_head, seq_len)
            z = torch.einsum('hbqk,hbrk->hbrq', pattern, v)
            z = z.permute(1, 0, 2, 3) # (batch, n_qk_heads, d_qk_head, seq_len)
            z = z.flatten(start_dim=1, end_dim=2)
        else:
            z = torch.einsum('bhqk,bhk->bhq', pattern, v)
        return z.permute(0, 2, 1)

    def _apply_causal_mask(self, scores):
        """Apply causal mask to attention scores."""
        seq_len = scores.size(-2)
        mask = self.mask[None, None, -seq_len:, -seq_len:]  # type: ignore
        ignore_value = self.IGNORE.to(scores.device)
        return torch.where(mask, scores, ignore_value)  # type: ignore
    
    @override
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool):
        raise NotImplementedError("set_decoder_to_fixed_norm does not make sense for lorsa")

    @override
    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float):
        """Set encoder weights to fixed norm."""
        raise NotImplementedError("set_encoder_to_fixed_norm does not make sense for lorsa")
    
    @torch.no_grad()
    def update_dead_latents(self, feature_acts: torch.Tensor):
        """Update the dead latents tracking based on current feature activations.
        
        Args:
            feature_acts: Feature activations tensor of shape (batch, d_sae) or (batch, seq_len, d_sae)
        """
        if not (self.cfg.use_auxk and self.cfg.act_fn == "topk"):
            return
            
        # Calculate batch size (number of tokens in this batch)
        if feature_acts.dim() == 3:  # (batch, seq_len, d_sae)
            batch_size = feature_acts.size(0) * feature_acts.size(1)  # batch * seq_len
        else:  # (batch, d_sae)
            batch_size = feature_acts.size(0)  # batch
            
        # Check which features were activated in this batch
        if feature_acts.dim() == 3:  # (batch, seq_len, d_sae)
            activated = feature_acts.gt(0).any(dim=(0, 1))  # (d_sae,)
        else:  # (batch, d_sae)
            activated = feature_acts.gt(0).any(dim=0)  # (d_sae,)
            
        # Update tokens since last activation
        # If a feature was activated, reset to 0; otherwise, add batch_size
        self.tokens_since_last_activation = torch.where(
            activated,
            torch.zeros_like(self.tokens_since_last_activation),
            self.tokens_since_last_activation + batch_size
        )
        
        # Mark as dead if tokens since last activation exceeds threshold
        self.is_dead = self.tokens_since_last_activation >= self.cfg.dead_threshold
    
    
    @overload
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_batch_norm_mse: bool = False,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        p: int = 1,
        l1_coefficient: float = 1.0,
        return_aux_data: Literal[True] = True,
        **kwargs,
    ) -> tuple[
        Float[torch.Tensor, " batch"],
        tuple[dict[str, Optional[torch.Tensor]], dict[str, torch.Tensor]],
    ]: ...

    @overload
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_batch_norm_mse: bool = False,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        p: int = 1,
        l1_coefficient: float = 1.0,
        return_aux_data: Literal[False],
        **kwargs,
    ) -> Float[torch.Tensor, " batch"]: ...

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        label: (
            Union[
                Float[torch.Tensor, "batch d_model"],
                Float[torch.Tensor, "batch seq_len d_model"],
            ]
            | None
        ) = None,
        *,
        use_batch_norm_mse: bool = False,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        frequency_scale: float = 0.01,
        p: int = 1,
        l1_coefficient: float = 1.0,
        return_aux_data: bool = True,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch"],
        tuple[
            Float[torch.Tensor, "batch"],
            tuple[dict[str, Optional[torch.Tensor]], dict[str, torch.Tensor]],
        ],
    ]:
        """Compute the loss for the autoencoder.
        Ensure that the input activations are normalized by calling `normalize_activations` before calling this method.
        """
        x, encoder_kwargs = self.prepare_input(batch)
        label = self.prepare_label(batch, **kwargs)

        feature_acts, hidden_pre = self.encode(x, return_hidden_pre=True, **encoder_kwargs)
        reconstructed = self.decode(feature_acts, **kwargs)

        # Update dead latents tracking
        self.update_dead_latents(feature_acts)

        l_rec = (reconstructed - label).pow(2)
        if use_batch_norm_mse:
            l_rec = (
                l_rec
                / (label - label.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()
            )
        l_rec = l_rec.sum(dim=-1)
        if isinstance(l_rec, DTensor):
            l_rec = l_rec.full_tensor()
        loss_dict: dict[str, Optional[torch.Tensor]] = {
            "l_rec": l_rec,
        }
        loss = l_rec.mean()

        if sparsity_loss_type is not None:
            if sparsity_loss_type == "power":
                l_s = torch.norm(feature_acts * self.decoder_norm(), p=p, dim=-1)
            elif sparsity_loss_type == "tanh":
                l_s = torch.tanh(tanh_stretch_coefficient * feature_acts * self.decoder_norm()).sum(dim=-1)
            elif sparsity_loss_type == "tanh-quad":
                approx_frequency = einops.reduce(
                    torch.tanh(tanh_stretch_coefficient * feature_acts * self.decoder_norm()),
                    "... d_sae -> d_sae",
                    "mean",
                )
                l_s = (approx_frequency * (1 + approx_frequency / frequency_scale)).sum(dim=-1)
            else:
                raise ValueError(f"sparsity_loss_type f{sparsity_loss_type} not supported.")
            if isinstance(l_s, DTensor):
                l_s = l_s.full_tensor()
            l_s = l1_coefficient * l_s
            # WARNING: Some DTensor bugs make if l1_coefficient * l_s goes before full_tensor, the l1_coefficient value will be internally cached. Furthermore, it will cause the backward pass to fail with redistribution error. See https://github.com/pytorch/pytorch/issues/153603 and https://github.com/pytorch/pytorch/issues/153615 .
            loss_dict["l_s"] = l_s
            loss = loss + l_s.mean()
        else:
            loss_dict["l_s"] = None

        # Add AuxK auxiliary loss if enabled
        if self.cfg.use_auxk and self.cfg.act_fn == "topk":
            with timer.time("auxk_loss_calculation"):
                # Get reconstruction error
                e = label - reconstructed  # (batch, d_model) or (batch, seq_len, d_model)
                
                # Get the top-k_aux dead latents based on their activation values
                current_k = self.current_k
                if self.device_mesh is not None:
                    self.current_k = min(self.cfg.k_aux, self.is_dead.full_tensor().sum())
                else:
                    self.current_k = min(self.cfg.k_aux, self.is_dead.sum())
                
                # print(f'{self.current_k = }')
                
                if self.current_k > 0:
                    # Scale feature activations by decoder norm if configured
                    if self.cfg.sparsity_include_decoder_norm:
                        dead_sparsity_scores = hidden_pre * self.is_dead * self.decoder_norm()
                    else:
                        dead_sparsity_scores = hidden_pre * self.is_dead

                    dead_activation_mask = self.activation_function(dead_sparsity_scores)
                    dead_feature_acts = torch.clamp(hidden_pre * dead_activation_mask * self.is_dead, min=0.0)
                    
                    # Decode auxiliary feature activations
                    aux_reconstructed = dead_feature_acts @ self.W_O
                    if isinstance(aux_reconstructed, DTensor):
                        aux_reconstructed = DimMap({}).redistribute(aux_reconstructed)
                    # print(f'{torch.norm(e, dim=-1) = }')
                    # print(f'{torch.norm(aux_reconstructed, dim=-1) = }')
                    l_aux = (e - aux_reconstructed).pow(2).sum(dim=-1)
                else:
                    l_aux = torch.zeros_like(l_rec)
                
                # print(f'{torch.norm(l_aux,dim = -1) = }')
                
                if isinstance(l_aux, DTensor):
                    l_aux = l_aux.full_tensor()
                loss_dict["l_aux"] = l_aux
                loss = loss + self.cfg.aux_coefficient * l_aux.mean()
                
                self.current_k = current_k

        if return_aux_data:
            aux_data = {
                "feature_acts": feature_acts,
                "reconstructed": reconstructed,
                "hidden_pre": hidden_pre,
            }
            return loss, (loss_dict, aux_data)
        return loss

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        strict_loading: bool = True,
        fold_activation_scale: bool = True,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        """Load pretrained model."""
        cfg = LorsaConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        model = cls.from_config(cfg, fold_activation_scale=fold_activation_scale, device_mesh=device_mesh)
        return model    
    @override
    def dim_maps(self) -> dict[str, DimMap]:
        """Return dimension maps for head parallelization."""
        base_maps = super().dim_maps()
        return {
            **base_maps,
            "W_Q": DimMap({"model": 0}),
            "W_K": DimMap({"model": 0}), 
            "W_V": DimMap({"model": 0}),
            "W_O": DimMap({"model": 0}),
            "b_Q": DimMap({"model": 0}),
            "b_K": DimMap({"model": 0}),
            "b_D": DimMap({}),
        }
    
    @override
    def prepare_input(self, batch, **kwargs):
        """Prepare input tensor."""
        if self.device_mesh is not None:
            x = DimMap({}).distribute(batch[self.cfg.hook_point_in], self.device_mesh)
        else:
            x = batch[self.cfg.hook_point_in]
        return x, {}

    @override
    def prepare_label(self, batch, **kwargs):
        """Prepare label tensor."""
        label = batch[self.cfg.hook_point_out]
        if self.cfg.skip_bos:
            label = label[:, 1:]
        return label 

    @torch.no_grad()
    def print_all_parameters(self, max_elements: int = 10):
        """打印所有参数的详细信息"""
        print("=" * 80)
        print("LORSA 模型参数信息")
        print("=" * 80)
        
        def _print_param_info(name: str, param: torch.Tensor, max_elements: int = 10):
            """打印单个参数的信息"""
            print(f"\n{name}:")
            print(f"  形状: {param.shape}")
            print(f"  数据类型: {param.dtype}")
            print(f"  设备: {param.device}")
            print(f"  是否可学习: {param.requires_grad}")
            
            # 计算统计信息
            param_flat = param.detach().flatten()
            
            # 对于布尔类型，使用特殊的统计方法
            if param.dtype == torch.bool:
                print(f"  True值数量: {param_flat.sum().item()}")
                print(f"  False值数量: {(~param_flat).sum().item()}")
                print(f"  True值比例: {param_flat.float().mean().item():.6f}")
                print(f"  False值比例: {(~param_flat).float().mean().item():.6f}")
            else:
                # 对于数值类型，计算常规统计信息
                print(f"  均值: {param_flat.mean().item():.6f}")
                print(f"  标准差: {param_flat.std().item():.6f}")
                print(f"  最小值: {param_flat.min().item():.6f}")
                print(f"  最大值: {param_flat.max().item():.6f}")
                print(f"  L2范数: {param.norm().item():.6f}")
            
            # 打印前几个元素
            if param.numel() > 0:
                elements_to_show = min(max_elements, param.numel())
                elements = param_flat[:elements_to_show]
                
                # 对于布尔类型，转换为字符串显示
                if param.dtype == torch.bool:
                    elements_str = [str(x.item()) for x in elements]
                    print(f"  前{elements_to_show}个元素: {elements_str}")
                else:
                    print(f"  前{elements_to_show}个元素: {elements.tolist()}")
                    
                if param.numel() > max_elements:
                    print(f"  ... (总共 {param.numel()} 个元素)")
        
        # 打印主要权重参数
        print("\n【主要权重参数】")
        _print_param_info("W_Q (Query权重)", self.W_Q, max_elements)
        _print_param_info("W_K (Key权重)", self.W_K, max_elements)
        _print_param_info("W_V (Value权重)", self.W_V, max_elements)
        _print_param_info("W_O (Output权重)", self.W_O, max_elements)
        
        # 打印偏置参数
        print("\n【偏置参数】")
        _print_param_info("b_Q (Query偏置)", self.b_Q, max_elements)
        _print_param_info("b_K (Key偏置)", self.b_K, max_elements)
        _print_param_info("b_V (Value偏置)", self.b_V, max_elements)
        
        if hasattr(self, 'b_D') and self.b_D is not None:
            _print_param_info("b_D (Decoder偏置)", self.b_D, max_elements)
        
        # 打印SmolGen参数（如果存在）
        if hasattr(self, 'smolgen') and self.smolgen is not None:
            print("\n【SmolGen参数】")
            for name, module in self.smolgen.named_modules():
                if len(list(module.children())) == 0:  # 叶子模块
                    for param_name, param in module.named_parameters():
                        full_name = f"smolgen.{name}.{param_name}"
                        _print_param_info(full_name, param, max_elements)
        
        # 打印其他缓冲区
        print("\n【缓冲区参数】")
        for name, buffer in self.named_buffers():
            if buffer.numel() > 0:  # 只打印非空缓冲区
                _print_param_info(f"buffer.{name}", buffer, max_elements)
        
        # 打印配置信息
        print("\n【配置信息】")
        print(f"  d_model: {self.cfg.d_model}")
        print(f"  n_qk_heads: {self.cfg.n_qk_heads}")
        print(f"  n_ov_heads: {self.cfg.n_ov_heads}")
        print(f"  d_qk_head: {self.cfg.d_qk_head}")
        print(f"  use_smolgen: {getattr(self.cfg, 'use_smolgen', False)}")
        print(f"  use_decoder_bias: {getattr(self.cfg, 'use_decoder_bias', False)}")
        
        # 打印可学习的 attn_scale 参数（如果存在）
        if hasattr(self, 'attn_scale') and self.attn_scale is not None:
            print("\n【可学习参数】")
            _print_param_info("attn_scale (注意力缩放)", self.attn_scale, max_elements)
        
        print("\n" + "=" * 80)
