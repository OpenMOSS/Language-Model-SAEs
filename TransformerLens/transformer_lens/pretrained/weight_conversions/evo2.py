from __future__ import annotations

"""Weight conversion helpers for Evo 2 / StripedHyena2 checkpoints."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


_CHECKPOINT_WRAPPER_KEYS = (
    "state_dict",
    "model_state_dict",
    "model",
    "module",
    "net",
    "weights",
)

_PREFIXES_TO_STRIP = (
    "module.",
    "model.",
    "net.",
    "evo2.",
    "inner_model.",
    "wrapped_model.",
)


def _is_tensor_like(value: Any) -> bool:
    return torch.is_tensor(value) or isinstance(value, torch.nn.Parameter)


def _unwrap_checkpoint(obj: Any) -> Any:
    """Recursively unwrap common checkpoint containers until we reach tensors."""
    if hasattr(obj, "state_dict") and not isinstance(obj, Mapping):
        return obj.state_dict()

    if not isinstance(obj, Mapping):
        return obj

    for key in _CHECKPOINT_WRAPPER_KEYS:
        value = obj.get(key)
        if isinstance(value, Mapping):
            return _unwrap_checkpoint(value)

    if len(obj) == 1:
        value = next(iter(obj.values()))
        if isinstance(value, Mapping):
            return _unwrap_checkpoint(value)

    return obj


def _strip_prefix(key: str) -> str:
    while True:
        stripped = False
        for prefix in _PREFIXES_TO_STRIP:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                stripped = True
                break
        if not stripped:
            return key


def _load_checkpoint(checkpoint: str | Path | Any) -> Any:
    if isinstance(checkpoint, (str, Path)):
        checkpoint = Path(checkpoint)
        try:
            return torch.load(
                checkpoint,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        except Exception:
            return torch.load(checkpoint, map_location="cpu", weights_only=False)
    return checkpoint


def _remap_hyena_cascade_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Translate flat HyenaCascade parameter names from old checkpoints to the
    refactored submodule layout used by the current StripedHyena implementation.

    Old layout (Savanna / vortex checkpoints)        New submodule layout
    ──────────────────────────────────────────────   ──────────────────────────────────
    blocks.N.filter.short_filter_weight          →   blocks.N.filter.short_fir.weight
    blocks.N.filter.short_filter_bias            →   blocks.N.filter.short_fir.bias
    blocks.N.filter.h                            →   blocks.N.filter.inner_fir.h
    blocks.N.filter.D   (HCM layer)              →   blocks.N.filter.inner_fir.D
    blocks.N.filter.log_poles                    →   blocks.N.filter.iir.log_poles
    blocks.N.filter.residues                     →   blocks.N.filter.iir.residues
    blocks.N.filter.D   (HCL layer)              →   blocks.N.filter.iir.D

    The HCL vs HCM distinction for the ambiguous ``D`` key is inferred from the
    presence of ``log_poles`` (HCL) or ``h`` (HCS/HCM) in the same block.
    """
    import re

    # Collect which block indices have which filter sub-keys so we can
    # distinguish HCL (has log_poles) from HCM (has h + D, no log_poles).
    hcl_blocks: set[int] = set()
    for key in state_dict:
        m = re.match(r'^blocks\.(\d+)\.filter\.log_poles$', key)
        if m:
            hcl_blocks.add(int(m.group(1)))

    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        m = re.match(r'^(blocks\.(\d+)\.filter\.)(.+)$', key)
        if m:
            prefix, n_str, suffix = m.group(1), m.group(2), m.group(3)
            n = int(n_str)
            if   suffix == "short_filter_weight":  key = prefix + "short_fir.weight"
            elif suffix == "short_filter_bias":    key = prefix + "short_fir.bias"
            elif suffix == "h":                    key = prefix + "inner_fir.h"
            elif suffix == "log_poles":            key = prefix + "iir.log_poles"
            elif suffix == "residues":             key = prefix + "iir.residues"
            elif suffix == "D":
                key = prefix + ("iir.D" if n in hcl_blocks else "inner_fir.D")
        remapped[key] = value
    return remapped


def convert_evo2_weights(checkpoint: Any, cfg: HookedTransformerConfig | None = None):
    """Normalize Evo 2 checkpoints into a plain state_dict.

    Steps applied in order:
    1. Load from path / unwrap checkpoint containers (``state_dict``, ``model``, …)
    2. Strip distributed-training prefixes (``module.``, ``evo2.``, …)
    3. Remap legacy flat HyenaCascade parameter names to the refactored submodule
       layout (``short_filter_weight`` → ``short_fir.weight``, etc.)
    """
    checkpoint = _load_checkpoint(checkpoint)
    checkpoint = _unwrap_checkpoint(checkpoint)

    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported Evo 2 checkpoint type: {type(checkpoint)!r}")

    state_dict: dict[str, Any] = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str):
            continue
        if not (_is_tensor_like(value) or key.endswith("._extra_state")) and "." not in key:
            # Skip checkpoint metadata (epoch counters, iteration labels, etc.)
            continue
        state_dict[_strip_prefix(key)] = value

    return _remap_hyena_cascade_keys(state_dict)
