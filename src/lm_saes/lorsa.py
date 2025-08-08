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
import torch.distributed.tensor
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from transformer_lens.hook_points import HookPoint
from typing_extensions import override
import einops
from jaxtyping import Float, Int

from .abstract_sae import AbstractSparseAutoEncoder
from .config import LorsaConfig
from .utils.distributed import DimMap
from .utils.logging import get_distributed_logger

logger = get_distributed_logger("lorsa")


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
            if self.cfg.use_decoder_bias:
                self.b_O = _get_param_with_shape((self.cfg.d_model,))
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
            if self.cfg.use_decoder_bias:
                self.b_O = _get_param_with_shape((self.cfg.d_model,), placements=dim_maps["b_O"].placements(device_mesh))
        
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

        if self.cfg.use_decoder_bias:
            torch.nn.init.zeros_(self.b_O)
    
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
        return torch.norm(self.b_O, p=2, dim=0, keepdim=True)

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
        self.b_Q.data *= input_norm_factor
        self.b_K.data *= input_norm_factor

        self.W_O.data = self.W_O.data * input_norm_factor / output_norm_factor
        self.b_O.data = self.b_O.data / output_norm_factor
        
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
        
        # Attention pattern
        # n_qk_heads batch q_pos k_pos
        pattern = self._compute_attention_pattern(q, k)
        
        # Head outputs
        hidden_pre = self._compute_head_outputs(pattern, v)

        feature_acts = self.activation_function(hidden_pre)

        return_values = [feature_acts]
        if return_hidden_pre:
            return_values.append(hidden_pre)
        if return_attention_pattern:
            return_values.append(pattern.permute(1, 0, 2, 3))
        return tuple(return_values) if len(return_values) > 1 else return_values[0]

    @override
    def decode(self, feature_acts, **kwargs):
        """Decode head activations to output."""
        if feature_acts.layout == torch.sparse_coo:
            return torch.sparse.mm(feature_acts, self.W_O) + self.b_O
        out = torch.einsum("bps,sd->bpd", feature_acts, self.W_O)
        if self.cfg.use_decoder_bias:
            out = out + self.b_O
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
        )
        
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
        qk_idx: Tensor = head_idx // self.cfg.d_qk_head
        q, k, v = self._compute_qkv(x)

        # (n_active_features, q_pos, k_pos)
        pattern = self._compute_attention_pattern(q, k)[qk_idx, 0]
        return pattern * v[0, :, head_idx, None].permute(1, 2, 0)
    
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
        scores = torch.einsum("sbqd,sbdk->sbqk", q, k) / self.cfg.attn_scale
        scores = self._apply_causal_mask(scores)
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
        **kwargs,
    ):
        """Load pretrained model."""
        cfg = LorsaConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        model = cls.from_config(cfg, fold_activation_scale=fold_activation_scale)
        assert model.cfg.dtype == torch.float32
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
            "b_O": DimMap({}),
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