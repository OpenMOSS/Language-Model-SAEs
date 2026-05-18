# Copyright (c) 2024, Michael Poli.
# Refactored: HyenaInferenceEngine replaced by three independent nn.Module subclasses.

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens.hook_points import HookPoint


# ---------------------------------------------------------------------------
# Standalone FFT-conv utility  (used by ParallelInnerFIR and ParallelIIR)
# ---------------------------------------------------------------------------

def fftconv_func(u, k, D, dropout_mask, gelu=True, k_rev=None, bidirectional=False, **kwargs):
    """FFT-based convolution of u with filter k, plus optional D skip connection."""
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    # broadcast filter shape to match u
    k_f = k_f.squeeze()
    if len(u.shape) > len(k_f.shape):
        k_f = k_f.unsqueeze(0)

    if bidirectional:
        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
        k1, k2 = k.squeeze().split(k.squeeze().shape[0] // 2, dim=0)
        k2_f = torch.fft.rfft(k2, n=fft_size) / fft_size
        y = torch.fft.irfft(u_f * k_f + u_f.conj() * k2_f.conj(), n=fft_size, norm="forward")[..., :seqlen]
    else:
        if k_rev is not None:
            k_f = k_f + torch.fft.rfft(k_rev, n=fft_size).conj() / fft_size
        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
        y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    return (y + u * D.unsqueeze(-1)).to(dtype=u.dtype)


# ---------------------------------------------------------------------------
# ShortDepthwiseFIR  –  causal depthwise conv on the full 3·D projection
# ---------------------------------------------------------------------------

class ShortDepthwiseFIR(nn.Module):
    """Causal depthwise short FIR applied to the full 3·D projection.

    Parameters
    ----------
    channels   : int   3 * hidden_size
    kernel_size: int   short_filter_length  (e.g. 3)
    has_bias   : bool  whether to add a learnable additive bias

    forward input : u  [B, L, 3D]  channels-last
    forward output: z  [B, 3D, L]  channels-first  (same layout as classic z_pre)
    """

    def __init__(self, channels: int, kernel_size: int, has_bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size))
        self.bias   = nn.Parameter(torch.randn(channels)) if has_bias else None

    def forward(self, u, padding_mask=None):
        L    = u.shape[1]
        u_cf = u.permute(0, 2, 1)                           # [B, 3D, L]
        z = F.conv1d(
            u_cf.to(torch.float32),
            self.weight.to(torch.float32),
            bias=None,
            stride=1,
            padding=self.kernel_size - 1,
            groups=u_cf.shape[1],
        )[..., :L].to(u.dtype)

        if self.bias is not None:
            z = z + self.bias[None, :, None]
        if isinstance(padding_mask, torch.Tensor):
            z = z * padding_mask[:, None]

        return z                                             # [B, 3D, L]


# ---------------------------------------------------------------------------
# ParallelInnerFIR  –  inner FIR for HCS (kernel=7) and HCM (kernel=128)
# ---------------------------------------------------------------------------

class ParallelInnerFIR(nn.Module):
    """Inner FIR filter applied to x1v for HCS and HCM Hyena layers.

    HCS  filter_length=7:   direct conv1d,  no D skip
    HCM  filter_length=128: FFT-based conv, gated D skip (fftconv_func includes D)

    Parameters
    ----------
    hyena_filter_groups : int   number of independent filter groups
    filter_length       : int   FIR kernel length  (7 for HCS, 128 for HCM)
    hidden_size         : int   d_model

    forward input / output : x1v  [B, d_model, pos]  channels-first,  BEFORE x2 gate

    hook_out captures the pre-gate filter response (observe / patch via run_with_cache).
    """

    def __init__(self, hyena_filter_groups: int, filter_length: int, hidden_size: int):
        super().__init__()
        self.hyena_filter_groups = hyena_filter_groups
        self.filter_length       = filter_length
        self.hidden_size         = hidden_size

        self.h = nn.Parameter(torch.randn(hyena_filter_groups, 1, filter_length))
        # D is a gated skip connection used for HCM only
        self.D = nn.Parameter(torch.zeros(hidden_size)) if filter_length >= 128 else None

        self.hook_out = HookPoint()   # [batch, d_model, pos] — before x2 gate

    def _expand_h(self):
        if self.hyena_filter_groups > 1:
            return self.h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)
        return self.h

    def forward(self, x1v):
        """x1v: [B, D, L] → z: [B, D, L]  (pre-gate)"""
        L = x1v.shape[2]
        h = self._expand_h()

        if self.filter_length >= 128:
            # HCM: FFT conv; fftconv_func adds D·x1v skip internally
            with torch.autocast("cuda"):
                z = fftconv_func(
                    x1v.to(torch.float32),
                    h[:, :, :L].to(torch.float32),
                    self.D,
                    None,
                    gelu=False,
                    bidirectional=False,
                ).to(x1v.dtype)
        else:
            # HCS: direct causal conv1d (no D skip)
            z = F.conv1d(
                x1v.to(torch.float32),
                h.to(torch.float32),
                bias=None,
                stride=1,
                padding=self.filter_length - 1,
                groups=x1v.shape[1],
            )[..., :L].to(x1v.dtype)

        return self.hook_out(z)


# ---------------------------------------------------------------------------
# ParallelIIR  –  long IIR / FFT filter for HCL layers
# ---------------------------------------------------------------------------

class ParallelIIR(nn.Module):
    """Parallel IIR (modal / FFT) filter for HCL Hyena layers.

    Owns all IIR parameters: log_poles, residues, D.

    Parameters
    ----------
    num_systems : int   hyena_filter_groups  (IIR filter groups)
    state_size  : int   number of IIR state poles per group
    hidden_size : int   d_model

    forward input / output : x1v  [B, d_model, pos]  channels-first,  BEFORE x2 gate

    hook_out captures the pre-gate filter response (observe / patch via run_with_cache).
    """

    def __init__(self, num_systems: int, state_size: int, hidden_size: int):
        super().__init__()
        self.num_systems = num_systems
        self.state_size  = state_size
        self.hidden_size = hidden_size

        self.log_poles = nn.Parameter(
            torch.randn(num_systems, state_size, 1, dtype=torch.float32)
        )
        self.residues  = nn.Parameter(
            torch.randn(num_systems, state_size, dtype=torch.float32)
        )
        self.D         = nn.Parameter(torch.zeros(hidden_size))

        # fftconv_fn is injected by StripedHyena when use_flashfft=True
        self.fftconv_fn = None

        # time-axis buffer (recomputed lazily, not a persistent parameter)
        self.t = None

        self.hook_out = HookPoint()   # [batch, d_model, pos] — before x2 gate

    # ------------------------------------------------------------------
    # Filter computation helpers
    # ------------------------------------------------------------------

    def _update_time(self, L: int, device):
        if self.t is None or self.t.shape[-1] < L:
            self.t = torch.arange(L, device=device)[None, None]
        else:
            self.t = self.t[..., :L]

    def _compute_filter(self, L: int, device):
        """Compute the IIR impulse response h from log_poles and residues."""
        self._update_time(L, device)
        residues  = self.residues.to(torch.float32)
        log_poles = self.log_poles.to(torch.float32)
        h = (residues[..., None] * (log_poles * self.t).exp()).sum(1)[None]  # [1, D, L]
        return h, log_poles, residues

    # ------------------------------------------------------------------
    # Parallel (prefill / training) forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x1v,
        inference_params=None,
        prefill_style: str = "fft",
        use_flashfft: bool = False,
        long_fir_threshold=None,
    ):
        """x1v: [B, D, L] → y: [B, D, L]  (pre-gate, hook_out applied)"""
        L        = x1v.shape[2]
        fft_size = 2 * L
        h, _, _  = self._compute_filter(L, x1v.device)
        D        = self.D

        if use_flashfft and (L % 2) == 0 and self.fftconv_fn is not None:
            y = self.fftconv_fn(x1v.to(dtype=torch.bfloat16).contiguous(), h.to(dtype=torch.float32))
        elif long_fir_threshold is None:
            H = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
            X = torch.fft.fft(x1v.to(dtype=torch.float32), n=fft_size)[..., : H.shape[-1]]
            y = torch.fft.irfft(X * H, n=fft_size, norm="forward")[..., :L]
        else:
            assert h.shape[0] == 1
            h_fir = h[0][:, None, :long_fir_threshold]
            y = F.conv1d(
                x1v, h_fir.to(dtype=x1v.dtype),
                stride=1, groups=x1v.shape[1], padding=h_fir.shape[-1] - 1,
            )[..., :L]

        return self.hook_out(y.to(dtype=x1v.dtype) + x1v * D.unsqueeze(-1))
