"""Backward-compat shim for :mod:`llamascopium.circuits.hooks`."""

from llamascopium.circuits.hooks import (
    apply_saes,
    detach_at,
    replace_model_biases_with_leaves,
    replace_sae_biases_with_leaves,
)

__all__ = [
    "apply_saes",
    "detach_at",
    "replace_model_biases_with_leaves",
    "replace_sae_biases_with_leaves",
]
