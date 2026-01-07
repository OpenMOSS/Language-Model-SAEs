import torch
from torch import nn
from einops import rearrange
from typing import Iterable, List, Optional

from lm_saes.backend.dino import dinov3
from lm_saes.config import CNNSAEConfig, LanguageModelConfig
from lm_saes.cnnsae import CNNSparseAutoEncoder


def _map_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(dtype_str.lower(), torch.float32)


class SAEFeatureClassifier(nn.Module):
    """
    Wrap DINOv3 + CNNSAE into a classifier-like module that outputs
    [B, d_sae] feature activations. This makes each SAE feature act
    like a "class logit" so pytorch-grad-cam targets can select a
    feature index.

    Expected input: image tensor in [-1, 1], shape [B, 3, H, W].
    """

    def __init__(
        self,
        *,
        sae_path: str,
        dino_ckpt_path: str,
        hook_point: str,
        device: torch.device,
        dtype: str = "float32",
        reduce: str = "mean",  # max | mean | mix | lse
        mix_alpha: float = 0.7,
        mix_alpha_start: Optional[float] = None,
        mix_alpha_end: Optional[float] = None,
        mix_hold_frac: float = 0.3,
        lse_temperature: float = 0.2,
        lse_temperature_start: Optional[float] = None,
        lse_temperature_end: Optional[float] = None,
        lse_hold_frac: float = 0.3,
        center_frac: float = 1.0,
        center_mode: str = "crop",
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = _map_dtype(dtype) if isinstance(dtype, str) else dtype
        self.reduce = reduce
        self.mix_alpha = float(mix_alpha)
        self.mix_alpha_start = mix_alpha_start
        self.mix_alpha_end = mix_alpha_end
        self.mix_hold_frac = mix_hold_frac
        self.lse_temperature = float(lse_temperature)
        self.lse_temperature_start = lse_temperature_start
        self.lse_temperature_end = lse_temperature_end
        self.lse_hold_frac = lse_hold_frac
        self.center_frac = float(center_frac)
        self.center_mode = center_mode

        device_str = "cuda" if device.type == "cuda" else "cpu"

        # Load DINOv3 (ConvNeXt backbone with HookPoints)
        lm_cfg = LanguageModelConfig(
            model_name="dinov3_large",
            model_from_pretrained_path=dino_ckpt_path,
            device=device_str,
            dtype=str(self.dtype),
        )
        _dino_lm = dinov3(lm_cfg)
        # Register the actual DINO model as a submodule so pytorch-grad-cam can find its layers
        self.dino_backbone = _dino_lm.model
        self.dino_backbone.eval()

        # Load CNNSAE and register as submodule
        sae_cfg = CNNSAEConfig.from_pretrained(sae_path, device=device_str, dtype=self.dtype)
        self.sae = CNNSparseAutoEncoder.from_config(sae_cfg)
        self.sae.eval()

        # Hook capture
        modules = dict(self.dino_backbone.named_modules())
        if hook_point not in modules:
            raise KeyError(f"hook_point not found in DINO model: {hook_point}")
        self._hook_point = hook_point
        self._captured = None

        def _fwd_hook(_module, _inp, out):
            self._captured = out

        modules[hook_point].register_forward_hook(_fwd_hook)

        # ImageNet mean/std for DINO
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    # ------------------------------------------------------------------ #
    # Helpers for scheduling alpha / temperature (optional)
    # ------------------------------------------------------------------ #
    def _interp_with_hold(self, start: float, end: float, hold_frac: float, progress: float) -> float:
        if progress <= hold_frac or hold_frac >= 1.0:
            return start
        p = (progress - hold_frac) / max(1e-8, 1.0 - hold_frac)
        return start + p * (end - start)

    def _get_mix_alpha(self, step_idx: Optional[int], num_steps: Optional[int]) -> float:
        if self.mix_alpha_start is None or self.mix_alpha_end is None or num_steps is None or num_steps <= 1:
            return self.mix_alpha
        progress = float(step_idx or 0) / float(max(num_steps - 1, 1))
        return float(self._interp_with_hold(self.mix_alpha_start, self.mix_alpha_end, self.mix_hold_frac, progress))

    def _get_lse_temperature(self, step_idx: Optional[int], num_steps: Optional[int]) -> float:
        if self.lse_temperature_start is None or self.lse_temperature_end is None or num_steps is None or num_steps <= 1:
            return self.lse_temperature
        progress = float(step_idx or 0) / float(max(num_steps - 1, 1))
        return float(self._interp_with_hold(self.lse_temperature_start, self.lse_temperature_end, self.lse_hold_frac, progress))

    # ------------------------------------------------------------------ #
    # Core forward
    # ------------------------------------------------------------------ #
    def _preprocess(self, img_m11: torch.Tensor) -> torch.Tensor:
        img01 = (img_m11 + 1.0) / 2.0
        img01 = img01.clamp(0.0, 1.0)
        return (img01 - self._mean.to(img01.device)) / self._std.to(img01.device)

    def _center_mask(self, h: int, w: int, device: torch.device) -> Optional[torch.Tensor]:
        frac = self.center_frac
        if frac >= 1.0:
            return None
        if frac <= 0.0:
            frac = 1.0 / float(max(min(h, w), 1))
        ch = max(int(round(h * frac)), 1)
        cw = max(int(round(w * frac)), 1)
        hs = (h - ch) // 2
        ws = (w - cw) // 2
        mask = torch.zeros((h, w), dtype=torch.bool, device=device)
        mask[hs : hs + ch, ws : ws + cw] = True
        return mask

    def _reduce_feats(
        self,
        feats_bdhw: torch.Tensor,
        mask_hw: Optional[torch.Tensor],
        *,
        step_idx: Optional[int],
        num_steps: Optional[int],
    ) -> torch.Tensor:
        """
        feats_bdhw: [B, d_sae, h, w]
        returns [B, d_sae]
        """
        reduce = self.reduce.lower()
        if reduce == "max":
            x = feats_bdhw
            if mask_hw is not None:
                x = x.masked_fill(~mask_hw.to(x.device).unsqueeze(0).unsqueeze(0), -torch.inf)
            return x.flatten(2).amax(dim=-1)

        if reduce == "mean":
            x = feats_bdhw
            if mask_hw is None:
                return x.flatten(2).mean(dim=-1)
            mask = mask_hw.to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
            denom = mask.sum(dim=(2, 3), keepdim=False).clamp_min(1)
            return (x * mask).flatten(2).sum(dim=-1) / denom

        if reduce == "mix":
            alpha = self._get_mix_alpha(step_idx, num_steps)
            if mask_hw is None:
                peak = feats_bdhw.flatten(2).amax(dim=-1)
                cover = torch.relu(feats_bdhw).flatten(2).mean(dim=-1)
            else:
                mask = mask_hw.to(feats_bdhw.device)
                peak = feats_bdhw.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -torch.inf).flatten(2).amax(dim=-1)
                xr = torch.relu(feats_bdhw) * mask.to(dtype=feats_bdhw.dtype).unsqueeze(0).unsqueeze(0)
                denom = mask.sum().clamp_min(1).to(dtype=feats_bdhw.dtype)
                cover = xr.flatten(2).sum(dim=-1) / denom
            return alpha * peak + (1.0 - alpha) * cover

        if reduce == "lse":
            T = self._get_lse_temperature(step_idx, num_steps)
            if T <= 0:
                raise ValueError("lse_temperature must be > 0")
            x = torch.relu(feats_bdhw)
            if mask_hw is not None:
                mask = mask_hw.to(feats_bdhw.device).view(1, 1, -1)
                x = x.flatten(2)
                x = x.masked_fill(~mask, -torch.inf)
            else:
                x = x.flatten(2)
            return T * torch.logsumexp(x / T, dim=-1)

        raise ValueError(f"unknown reduce={self.reduce}, expected one of: max | mean | mix | lse")

    def forward(self, img_m11: torch.Tensor, *, step_idx: Optional[int] = None, num_steps: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            img_m11: [B,3,H,W] in [-1,1]
            step_idx/num_steps: optional scheduling hints for mix/lse reductions.
        Returns:
            logits: [B, d_sae] SAE feature activations (post spatial reduction).
        """
        # Ensure gradients are enabled for this pass
        with torch.set_grad_enabled(True):
            img_norm = self._preprocess(img_m11)
            self._captured = None
            _ = self.dino_backbone(img_norm)
            acts_hw = self._captured
            if acts_hw is None:
                raise RuntimeError(f"failed to capture activation at hook_point={self._hook_point}")

            # Check if captured activation has gradients
            # if not acts_hw.requires_grad:
            #    print(f"Warning: Activation at {self._hook_point} does not require grad!")

            acts_seq = rearrange(acts_hw, "b d h w -> b (h w) d")
            batch = {self.sae.cfg.hook_point_in: acts_seq}
            x_sae, encoder_kwargs, _ = self.sae.prepare_input(batch)
            _, feature_acts = self.sae.encode(x_sae, return_hidden_pre=True, **encoder_kwargs)  # [B, seq, d_sae]

            # reshape to [B, d_sae, h, w]
            _, _, h, w = acts_hw.shape
            feats_bdhw = feature_acts.permute(0, 2, 1).contiguous().view(feature_acts.shape[0], feature_acts.shape[2], h, w)
            mask_hw = self._center_mask(h, w, device=feats_bdhw.device)
            logits = self._reduce_feats(feats_bdhw, mask_hw, step_idx=step_idx, num_steps=num_steps)
            return logits

    # ------------------------------------------------------------------ #
    # Target layer helpers for CAM
    # ------------------------------------------------------------------ #
    def list_candidate_layers(self, prefix: str = "") -> List[str]:
        """
        Return names of modules in the DINO model that can be used as CAM target layers.
        Names are prefixed with 'dino_backbone.' so they match the module hierarchy.
        """
        names = []
        for name, module in self.dino_backbone.named_modules():
            if prefix and not name.startswith(prefix):
                continue
            # Heuristic: only keep modules with parameters or HookPoints
            if list(module.children()) or any(p.requires_grad for p in module.parameters(recurse=False)):
                # Return full path for pytorch-grad-cam to find
                names.append(f"dino_backbone.{name}" if name else "dino_backbone")
        return names

    def resolve_target_layers(self, names: Iterable[str]) -> List[nn.Module]:
        """
        Given module names, resolve them to modules for pytorch-grad-cam.
        Names can be either:
        - Short names within DINO (e.g. 'stages.2.20.hook_resid_pre')
        - Full paths in SAEFeatureClassifier (e.g. 'dino_backbone.stages.2.20.hook_resid_pre')
        """
        # Build lookup from both short (DINO-internal) and full (SAEFeatureClassifier) paths
        dino_modules = dict(self.dino_backbone.named_modules())
        full_modules = dict(self.named_modules())
        
        layers: List[nn.Module] = []
        for n in names:
            n = n.strip()
            if not n:
                continue
            # Try full path first (e.g. 'dino_backbone.stages.2.20')
            if n in full_modules:
                layers.append(full_modules[n])
            # Then try short path (e.g. 'stages.2.20')
            elif n in dino_modules:
                layers.append(dino_modules[n])
            else:
                raise KeyError(f"target layer not found: {n}. Use --list_layers to see available modules.")
        if not layers:
            raise ValueError("No target layers resolved. Please provide at least one valid module name.")
        return layers

