"""Backward-compat shim for :mod:`lm_saes.circuits.hooks`."""

from lm_saes.circuits.hooks import (
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
