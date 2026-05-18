# Copyright (c) 2024, Michael Poli.
# Simplified: removed ground_truth_activations_path / print_activations debug paths.

import math
import logging
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MHA
from .engine import ShortDepthwiseFIR, ParallelInnerFIR, ParallelIIR
from .layers import (
    ParallelGatedMLP,
    RMSNorm,
    VocabParallelEmbedding,
    VocabParallelUnembedding,
    HAS_TE,
)
from .utils import (
    Lambda,
    column_split,
    interleave,
    print_rank_0,
    move_to_device,
    fixup_te_workspace,
)
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint, HookedRootModule

try:
    from .positional_embeddings import swap_mha_rope
except ImportError:
    pass


def _cuda_device_context(device):
    """Return a cuda device context manager, or nullcontext for cpu."""
    if device == "cpu" or (isinstance(device, torch.device) and device.type == "cpu"):
        return nullcontext()
    return torch.cuda.device(device)


# ---------------------------------------------------------------------------
# AttentionBlock
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    """Transformer (MHA + MLP) block used at attn_layer_idxs.

    Hook naming follows transformer_block.py conventions:
      hook_resid_pre   – input residual stream
      hook_attn_in     – normalised input fed into MHA
      hook_attn_out    – MHA output (before residual add)
      hook_resid_mid   – residual stream between attention and MLP
      hook_mlp_in      – normalised input fed into MLP
      hook_mlp_out     – MLP output (before residual add)
      hook_resid_post  – output residual stream
    """

    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.proj_groups = config.get("proj_groups", 1)
        dtype = config.get("attn_block_dtype", torch.bfloat16)
        mlp_dtype = config.get("mlp_dtype", torch.bfloat16)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.hidden_size_per_attention_head = config.hidden_size // config.num_attention_heads

        self.pre_norm = RMSNorm(config)
        self.post_norm = RMSNorm(config)

        self.inner_mha_cls = MHA(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_heads_kv=config.num_attention_heads // self.proj_groups,
            rotary_emb_dim=config.hidden_size // config.num_attention_heads,
            qkv_proj_bias=config.get("qkv_proj_bias", True),
            rotary_emb_base=config.get("rotary_emb_base", 1000000),
            causal=True,
            layer_idx=layer_idx,
            out_proj_bias=config.get("mha_out_proj_bias", True),
            use_flash_attn=self.config.use_flash_attn,
        ).to(dtype=dtype)

        if config.get("use_interpolated_rotary_pos_emb", False):
            swap_mha_rope(
                mha=self.inner_mha_cls,
                kwargs_new_rope={"scaling_factor": config.get("rotary_emb_scaling_factor", 1.0)},
            )

        if self.config.get("smeared_gqa", False):
            self.inner_mha_cls.num_heads_kv = self.inner_mha_cls.num_heads
        self.inner_mha_cls.rotary_emb.register_buffer(
            "inv_freq", self.inner_mha_cls.rotary_emb.inv_freq
        )

        self.mlp = ParallelGatedMLP(config, layer_idx).to(dtype=mlp_dtype)

        # HookPoints --------------------------------------------------------
        self.hook_resid_pre  = HookPoint()  # [batch, pos, d_model]
        self.hook_attn_in    = HookPoint()  # [batch, pos, d_model]
        self.hook_attn_out   = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid  = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_in     = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out    = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        u = self.hook_resid_pre(u)  # [batch, pos, d_model]

        if type(padding_mask) == torch.Tensor:
            u = u * padding_mask[..., None]

        attn_in  = self.hook_attn_in(self.pre_norm(u))                         # [batch, pos, d_model]
        attn_out = self.hook_attn_out(
            self.inner_mha_cls(attn_in, inference_params=inference_params)
        )                                                                       # [batch, pos, d_model]
        resid_mid = self.hook_resid_mid(u + attn_out)                          # [batch, pos, d_model]

        if type(padding_mask) == torch.Tensor:
            resid_mid = resid_mid * padding_mask[..., None]

        mlp_in    = self.hook_mlp_in(self.post_norm(resid_mid))                # [batch, pos, d_model]
        mlp_out   = self.hook_mlp_out(self.mlp(mlp_in))                        # [batch, pos, d_model]
        resid_post = self.hook_resid_post(resid_mid + mlp_out)                 # [batch, pos, d_model]
        return resid_post, None


# ---------------------------------------------------------------------------
# HyenaCascade  (inner mixer used inside ParallelGatedConvBlock)
# ---------------------------------------------------------------------------

class HyenaCascade(nn.Module):
    """Hyena long-convolution mixer (IIR or short-FIR inner path).

    Sub-modules:
      short_fir            – ShortDepthwiseFIR (causal depthwise conv, kernel=3)
      inner_fir            – ParallelInnerFIR  (HCS kernel=7 or HCM kernel=128)
      iir                  – ParallelIIR       (HCL long IIR / FFT filter)

    HookPoints:
      hook_fir_out         – after short FIR              [batch, 3*d_model, pos]
      hook_x2              – output gate x2               [batch, d_model,   pos]
      hook_x1              – input gate x1                [batch, d_model,   pos]
      hook_v               – value stream v               [batch, d_model,   pos]
      hook_x1v             – x1 ⊙ v  (filter input)      [batch, d_model,   pos]
      inner_fir.hook_out   – HCS/HCM filter, pre-x2 gate [batch, d_model,   pos]
      iir.hook_out         – HCL filter, pre-x2 gate     [batch, d_model,   pos]
      hook_filter_out      – after x2 gate                [batch, pos, d_model]

    Note: only one of inner_fir / iir exists depending on the layer type.
    """

    def __init__(
        self, config, layer_idx, hyena_filter_groups=None, fir_inner_filter_length=None
    ) -> None:
        super().__init__()
        self.config     = config
        self.hyena_filter_groups = hyena_filter_groups
        self.use_flashfft        = config.get("use_flashfft", False)
        self.state_size          = config.state_size
        self.hidden_size         = config.hidden_size
        self.num_filters         = config.num_filters
        self.column_split_hyena  = config.get("column_split_hyena", True)
        self.hyena_flip_x1x2     = config.get("hyena_flip_x1x2", False)
        self.long_fir_threshold  = config.get("long_fir_threshold", None)
        self.fir_inner_filter_length = fir_inner_filter_length

        assert self.hidden_size % self.num_filters == 0 and self.num_filters <= self.hidden_size

        self.num_attention_heads            = config.num_attention_heads
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads

        # ── Short depthwise FIR (always present) ───────────────────────────
        self.short_fir = ShortDepthwiseFIR(
            channels    = 3 * config.hidden_size,
            kernel_size = config.short_filter_length,
            has_bias    = config.short_filter_bias,
        )

        # ── Main filter: HCS/HCM → ParallelInnerFIR, HCL → ParallelIIR ───
        if fir_inner_filter_length is not None:
            self.inner_fir = ParallelInnerFIR(
                hyena_filter_groups = self.hyena_filter_groups,
                filter_length       = fir_inner_filter_length,
                hidden_size         = config.hidden_size,
            )
        else:
            self.iir = ParallelIIR(
                num_systems  = self.hyena_filter_groups,
                state_size   = config.state_size,
                hidden_size  = config.hidden_size,
            )

        # HookPoints ─────────────────────────────────────────────────────────
        self.hook_fir_out    = HookPoint()  # [batch, 3*d_model, pos] – after short FIR
        self.hook_x2         = HookPoint()  # [batch, d_model,   pos] – output gate
        self.hook_x1         = HookPoint()  # [batch, d_model,   pos] – input gate
        self.hook_v          = HookPoint()  # [batch, d_model,   pos] – value stream
        self.hook_x1v        = HookPoint()  # [batch, d_model,   pos] – x1 ⊙ v
        self.hook_x2y        = HookPoint()  # [batch, d_model,   pos] – x2 ⊙ filter_out (pre-permute)
        self.hook_filter_out = HookPoint()  # [batch, pos, d_model]   – after x2 gate + permute

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        return self.parallel_forward(u, padding_mask)

    def parallel_forward(self, u, padding_mask=None):
        # ── 1. Short FIR ────────────────────────────────────────────────────
        z_pre = self.short_fir(u, padding_mask)
        if self.config.interleave:
            z_pre = interleave(z_pre)
        z_pre = self.hook_fir_out(z_pre)                            # [B, 3D, L]

        # ── 2. Split x2 / x1 / v and hook each ─────────────────────────────
        if not self.column_split_hyena:
            _x2, _x1, _v = z_pre.split([self.hidden_size] * 3, dim=1)
            if self.hyena_flip_x1x2:
                _x1, _x2 = _x2, _x1
            _x2  = self.hook_x2(_x2)                                # [B, D, L]
            _x1  = self.hook_x1(_x1)                                # [B, D, L]
            _v   = self.hook_v(_v)                                   # [B, D, L]
            _x1v = self.hook_x1v(_x1 * _v)                          # [B, D, L]
        else:
            # column_split: observe-only (non-invertible permutation)
            _x2, _x1, _v = column_split(z_pre, self.num_attention_heads, self.hidden_size_per_attention_head)
            if self.hyena_flip_x1x2:
                _x1, _x2 = _x2, _x1
            self.hook_x2(_x2)
            self.hook_x1(_x1)
            self.hook_v(_v)
            _x1v = self.hook_x1v(_x1 * _v)                          # patching propagates to filter

        # ── 3. Main filter → hook_out inside submodule ───────────────────────
        if self.fir_inner_filter_length is not None:
            y = self.inner_fir(_x1v)
        else:
            y = self.iir(
                _x1v,
                prefill_style     = self.config.get("prefill_style", "fft"),
                use_flashfft      = self.use_flashfft,
                long_fir_threshold= self.long_fir_threshold,
            )
        # y: [B, D, L] pre-gate  (inner_fir.hook_out / iir.hook_out already applied)

        # ── 4. x2 gate + permute → hook_filter_out ──────────────────────────
        y = self.hook_x2y(_x2 * y)                                  # [B, D, L]
        y = self.hook_filter_out(y.permute(0, 2, 1))                # [B, L, D]
        return y, None



# ---------------------------------------------------------------------------
# ParallelGatedConvBlock  (Hyena block at hcl / hcm / hcs layer indices)
# ---------------------------------------------------------------------------

class ParallelGatedConvBlock(nn.Module):
    """Parallel gated conv (Hyena) block used at hcl/hcm/hcs_layer_idxs.

    Data-flow:
      u  →  pre_norm  →  projections  →  [HyenaCascade]  →  out_filter_dense  →  + u
         →  post_norm  →  mlp  →  + (conv output + u)

    HookPoints (custom names for the conv path):
      hook_resid_pre   – input u                               [batch, pos, d_model]
      hook_conv_in     – after pre_norm+projection, into filter [batch, pos, 3*d_model]
      hook_conv_out    – out_filter_dense output (before +u)   [batch, pos, d_model]
      hook_resid_mid   – after conv branch + residual           [batch, pos, d_model]
      hook_mlp_in      – after post_norm, into MLP              [batch, pos, d_model]
      hook_mlp_out     – MLP output (before residual add)       [batch, pos, d_model]
      hook_resid_post  – final output                           [batch, pos, d_model]
    """

    def __init__(
        self, config, layer_idx, hyena_filter_groups=None, fir_inner_filter_length=None
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.fir_inner_filter_length = fir_inner_filter_length
        self.hyena_filter_groups = (
            hyena_filter_groups if hyena_filter_groups is not None else config.hidden_size
        )
        dtype = config.get("hyena_block_dtype", torch.bfloat16)
        mlp_dtype = config.get("mlp_dtype", torch.bfloat16)

        self.pre_norm  = RMSNorm(config).to(dtype=dtype)
        self.post_norm = RMSNorm(config).to(dtype=dtype)

        self.filter = HyenaCascade(
            config,
            layer_idx,
            hyena_filter_groups=self.hyena_filter_groups,
            fir_inner_filter_length=fir_inner_filter_length,
        ).to(dtype=dtype)

        # For FP8 models (40b+) replace with TELinear(..., use_fp8=True).
        self.projections = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=config.qkv_proj_bias,
        ).to(dtype=dtype)
        self.out_filter_dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.hyena_out_proj_bias
        ).to(dtype)
        self.mlp = ParallelGatedMLP(config, layer_idx).to(dtype=mlp_dtype)

        # HookPoints --------------------------------------------------------
        self.hook_resid_pre  = HookPoint()  # [batch, pos, d_model]
        self.hook_conv_in    = HookPoint()  # [batch, pos, 3*d_model]
        self.hook_conv_out   = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid  = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_in     = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out    = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        u = self.hook_resid_pre(u)  # [batch, pos, d_model]

        # --- mixer: pre_norm → project → filter → out_dense ---------------
        with _cuda_device_context(u.device):
            z = self.projections(self.pre_norm(u))

        if type(padding_mask) == torch.Tensor:
            z = z * padding_mask[..., None]

        z = self.hook_conv_in(z)                                               # [batch, pos, 3*d_model]
        z, inference_params = self.filter(
            z, inference_params=inference_params, padding_mask=padding_mask
        )

        conv_out  = self.hook_conv_out(self.out_filter_dense(z))               # [batch, pos, d_model]
        resid_mid = self.hook_resid_mid(conv_out + u)                          # [batch, pos, d_model]

        if type(padding_mask) == torch.Tensor:
            resid_mid = resid_mid * padding_mask[..., None]

        # --- MLP -----------------------------------------------------------
        mlp_in    = self.hook_mlp_in(self.post_norm(resid_mid))                # [batch, pos, d_model]
        mlp_out   = self.hook_mlp_out(self.mlp(mlp_in))                        # [batch, pos, d_model]
        resid_post = self.hook_resid_post(resid_mid + mlp_out)                 # [batch, pos, d_model]
        return resid_post, inference_params


# ---------------------------------------------------------------------------
# Block factory
# ---------------------------------------------------------------------------

def get_block(config, layer_idx, flash_fft=None):
    if layer_idx in config.attn_layer_idxs:
        return AttentionBlock(config, layer_idx)
    elif layer_idx in config.hcl_layer_idxs:
        block = ParallelGatedConvBlock(config, layer_idx)
        if config.get("use_flashfft", False):
            block.filter.iir.fftconv_fn = flash_fft   # inject into ParallelIIR submodule
        return block
    elif layer_idx in config.hcm_layer_idxs:
        return ParallelGatedConvBlock(
            config,
            layer_idx,
            hyena_filter_groups=config.hcm_filter_groups,
            fir_inner_filter_length=config.hcm_filter_length,
        )
    elif layer_idx in config.hcs_layer_idxs:
        return ParallelGatedConvBlock(
            config,
            layer_idx,
            hyena_filter_groups=config.hcs_filter_groups,
            fir_inner_filter_length=config.hcs_filter_length,
        )
    else:
        raise NotImplementedError(f"No block type defined for layer_idx={layer_idx}")


# ---------------------------------------------------------------------------
# StripedHyena  (top-level model)
# ---------------------------------------------------------------------------

class StripedHyena(HookedRootModule):
    """StripedHyena 2 model with TransformerLens HookPoints throughout.

    Extends HookedRootModule so that ``run_with_cache`` / ``add_hook`` work
    the same way as on ``HookedTransformer``.

    Top-level HookPoints:
      hook_embed   – after embedding lookup  [batch, pos, d_model]
      hook_logits  – after unembed           [batch, pos, vocab]

    Per-block HookPoints live inside each block under ``blocks[i]``.
    """

    def __init__(self, config):
        super().__init__()
        if HAS_TE:
            fixup_te_workspace()

        if config.get("use_fp8_input_projections", False) and not HAS_TE:
            raise ImportError(
                "This model requires FP8 input projections (use_fp8_input_projections=True) "
                "which depends on Transformer Engine, but TE is not installed.\n"
                "Only 7b models (8k, 262k, 1m) can run without Transformer Engine."
            )

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.config.get("evo2_style_activations", False):
            self.logger.warning(
                "Not using Evo2-style activations. "
                "Set evo2_style_activations=True in config when loading Evo 2 checkpoints."
            )

        with torch.device("cuda:0" if torch.cuda.is_available() else "cpu"):
            self.embedding_layer = VocabParallelEmbedding(config)

        self.flash_fft = None
        if config.get("use_flashfft", False):
            try:
                from flashfftconv import FlashFFTConv

                self.flash_fft = FlashFFTConv(config.seqlen, dtype=torch.bfloat16)
            except ImportError:
                pass

        self.blocks = nn.ModuleList()
        self.block_idx_to_device: dict[int, str] = {}

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        layers_per_gpu = math.ceil(config.num_layers / num_gpus)

        for layer_idx in tqdm(range(config.num_layers)):
            device_idx = min(layer_idx // layers_per_gpu, num_gpus - 1)
            device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
            with torch.device(device):
                with _cuda_device_context(device):
                    block = get_block(config, layer_idx, flash_fft=self.flash_fft)
                    move_to_device(block, device)
            self.blocks.append(block)
            self.block_idx_to_device[layer_idx] = device

        with torch.device(self.block_idx_to_device[0]):
            with _cuda_device_context(self.block_idx_to_device[0]):
                self.norm = RMSNorm(config) if config.get("final_norm", True) else None
                if config.tie_embeddings:
                    self.unembed = Lambda(self.embedding_layer.unembed)
                else:
                    self.unembed = VocabParallelUnembedding(config)

        # Top-level HookPoints --------------------------------------------
        self.hook_embed  = HookPoint()  # [batch, pos, d_model]
        self.hook_logits = HookPoint()  # [batch, pos, vocab]

        # Register all hook points so run_with_cache / add_hook work ------
        self.setup()

