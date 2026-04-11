"""Backward-compat shim for :mod:`lm_saes.circuits.attribution`."""

from lm_saes.circuits.attribution import (
    AttributionResult,
    NodeInfoQueue,
    NodeInfoRef,
    attribute,
    compute_intermediates_attribution,
    get_normalized_matrix,
    prune_attribution,
    qk_trace,
)

__all__ = [
    "AttributionResult",
    "NodeInfoQueue",
    "NodeInfoRef",
    "attribute",
    "compute_intermediates_attribution",
    "get_normalized_matrix",
    "prune_attribution",
    "qk_trace",
]
