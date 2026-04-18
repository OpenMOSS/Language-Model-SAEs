"""Backward-compat shim for :mod:`lm_saes.circuits.indexed_tensor`."""

from lm_saes.circuits.indexed_tensor import (
    Node,
    NodeDimension,
    NodeIndexed,
    NodeIndexedMatrix,
    NodeIndexedTensor,
    NodeIndexedVector,
    NodeInfo,
)

Dimension = NodeDimension
Dimensioned = NodeIndexed

__all__ = [
    "Dimension",
    "Dimensioned",
    "Node",
    "NodeDimension",
    "NodeIndexed",
    "NodeIndexedMatrix",
    "NodeIndexedTensor",
    "NodeIndexedVector",
    "NodeInfo",
]
