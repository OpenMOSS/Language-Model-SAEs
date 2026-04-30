from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from .bt4_transfer import (
    HAS_ONNX_SUPPORT as BT4_HAS_ONNX_SUPPORT,
    CleanLC0Model as BT4CleanLC0Model,
    convert_bt4_onnx_to_clean_state_dict,
)
from .t82_transfer import (
    HAS_ONNX_SUPPORT as T82_HAS_ONNX_SUPPORT,
    CleanLC0Model as T82CleanLC0Model,
    convert_t82_onnx_to_clean_state_dict,
)


@dataclass(frozen=True)
class ModelConversionSpec:
    name: str
    model_cls: type[nn.Module]
    convert_fn: Callable[..., Path]
    has_onnx_support: bool


_SPECS: dict[str, ModelConversionSpec] = {
    "bt4": ModelConversionSpec(
        name="bt4",
        model_cls=BT4CleanLC0Model,
        convert_fn=convert_bt4_onnx_to_clean_state_dict,
        has_onnx_support=BT4_HAS_ONNX_SUPPORT,
    ),
    "t82": ModelConversionSpec(
        name="t82",
        model_cls=T82CleanLC0Model,
        convert_fn=convert_t82_onnx_to_clean_state_dict,
        has_onnx_support=T82_HAS_ONNX_SUPPORT,
    ),
}


def list_supported_models() -> tuple[str, ...]:
    return tuple(sorted(_SPECS))


def get_conversion_spec(model_name: str) -> ModelConversionSpec:
    key = model_name.strip().lower()
    if key not in _SPECS:
        supported = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model `{model_name}`. Supported models: {supported}")
    return _SPECS[key]


def build_clean_model(model_name: str, *, device: str | torch.device | None = None) -> nn.Module:
    spec = get_conversion_spec(model_name)
    model = spec.model_cls()
    if device is not None:
        model = model.to(device)
    return model


def load_clean_model(
    model_name: str,
    *,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    map_location: str | torch.device | None = "cpu",
) -> nn.Module:
    model = build_clean_model(model_name, device=device)
    if checkpoint_path is None:
        return model
    state_dict = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    model.load_state_dict(state_dict)
    return model


def convert_onnx_to_clean_state_dict(
    model_name: str,
    *,
    onnx_model_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
    verbose: bool = True,
) -> Path:
    spec = get_conversion_spec(model_name)
    return spec.convert_fn(
        onnx_model_path=onnx_model_path,
        output_path=output_path,
        device=device,
        verbose=verbose,
    )


def supports_onnx_conversion(model_name: str) -> bool:
    return get_conversion_spec(model_name).has_onnx_support


def conversion_support_status() -> dict[str, bool]:
    return {name: spec.has_onnx_support for name, spec in sorted(_SPECS.items())}


__all__ = [
    "BT4CleanLC0Model",
    "ModelConversionSpec",
    "T82CleanLC0Model",
    "build_clean_model",
    "conversion_support_status",
    "convert_onnx_to_clean_state_dict",
    "get_conversion_spec",
    "list_supported_models",
    "load_clean_model",
    "supports_onnx_conversion",
]
