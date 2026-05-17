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


def convert_evo2_weights(checkpoint: Any, cfg: HookedTransformerConfig | None = None):
    """Normalize Evo 2 checkpoints into a plain state_dict.

    Evo 2 checkpoints are usually already stored using module names that match the
    local `components.evo2` implementation. The only work we need here is:

    - unwrap common checkpoint containers such as `state_dict` or `model`
    - strip distributed-training prefixes such as `module.`
    - keep tensor values and `_extra_state` payloads intact
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
            # Skip checkpoint metadata like epoch counters and iteration labels.
            continue
        normalized_key = _strip_prefix(key)
        state_dict[normalized_key] = value

    return state_dict
