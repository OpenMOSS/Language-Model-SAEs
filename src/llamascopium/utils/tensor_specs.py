import functools

import einops
import torch
from torch import Tensor


class TensorSpecs:
    """Infer the specs of a tensor.

    Specs are a tuple of dimension names. Length of the tuple should match the number of dimensions of the tensor.
    """

    @staticmethod
    def feature_acts(tensor: torch.Tensor) -> tuple[str, ...]:
        if tensor.ndim == 2:
            return ("batch", "sae")
        elif tensor.ndim == 3:
            return ("batch", "context", "sae")
        else:
            raise ValueError(f"Cannot infer tensor specs for tensor with {tensor.ndim} dimensions.")

    @staticmethod
    def reconstructed(tensor: torch.Tensor) -> tuple[str, ...]:
        if tensor.ndim == 2:
            return ("batch", "model")
        elif tensor.ndim == 3:
            return ("batch", "context", "model")
        else:
            raise ValueError(f"Cannot infer tensor specs for tensor with {tensor.ndim} dimensions.")

    @staticmethod
    def label(tensor: torch.Tensor) -> tuple[str, ...]:
        return TensorSpecs.reconstructed(tensor)

    @staticmethod
    def loss(tensor: torch.Tensor) -> tuple[str, ...]:
        if tensor.ndim == 1:
            return ("batch",)
        elif tensor.ndim == 2:
            return ("batch", "context")
        else:
            raise ValueError(f"Cannot infer tensor specs for tensor with {tensor.ndim} dimensions.")


def h(names: tuple[str, ...]) -> str:
    return " ".join(names)


def reduce(tensor: Tensor, specs: tuple[str, ...], reduction_map: dict[str, str]) -> tuple[Tensor, tuple[str, ...]]:
    """Reduce the tensor by the mapping of dimension names to reduction functions."""

    assert tensor.ndim == len(specs), f"Tensor has {tensor.ndim} dimensions, but specs have {len(specs)} dimensions"

    def _reduce(tensor: Tensor, specs: tuple[str, ...], dim: str, reduction: str) -> tuple[Tensor, tuple[str, ...]]:
        target_specs = tuple(filter(lambda x: x != dim, specs))
        if specs == target_specs:
            return tensor, target_specs
        return einops.reduce(tensor, f"{h(specs)} -> {h(target_specs)}", reduction), target_specs

    return functools.reduce(lambda acc, item: _reduce(*acc, *item), reduction_map.items(), (tensor, specs))


def apply_token_mask(
    tensor: Tensor, specs: tuple[str, ...], mask: Tensor | None = None, reduction: str = "sum"
) -> tuple[Tensor, tuple[str, ...]]:
    """Apply the token mask to the tensor. Mask should be a 0/1 tensor with the same shape as the token part of the tensor, i.e. the shape of batch and context dimensions."""

    assert tensor.ndim == len(specs), f"Tensor has {tensor.ndim} dimensions, but specs have {len(specs)} dimensions"

    if mask is None:
        return reduce(tensor, specs, {"batch": reduction, "context": reduction})

    token_specs = tuple(filter(lambda x: x in ["batch", "context"], specs))
    token_shape = tuple([size for size, spec in zip(tensor.shape, specs) if spec in token_specs])
    assert mask.shape == token_shape, (
        f"Mask has shape {mask.shape}, but input tensor has token part of shape {token_shape}"
    )

    target_specs = tuple(filter(lambda x: x not in token_specs, specs))
    result = einops.einsum(tensor, mask.to(tensor), f"{h(specs)}, {h(token_specs)} -> {h(target_specs)}")
    if reduction == "mean":
        result = result / mask.sum()
    return result, target_specs
