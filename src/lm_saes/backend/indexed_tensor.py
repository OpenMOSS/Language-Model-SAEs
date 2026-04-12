"""Backward-compat shim for :mod:`lm_saes.circuits.indexed_tensor`."""

from lm_saes.circuits.indexed_tensor import (
    Node,
    NodeAxis,
    NodeIndexed,
    NodeIndexedMatrix,
    NodeIndexedTensor,
    NodeIndexedVector,
    NodeInfo,
)

Dimension = NodeAxis
Dimensioned = NodeIndexed

__all__ = [
    "Dimension",
    "Dimensioned",
    "Node",
    "NodeAxis",
    "NodeIndexed",
    "NodeIndexedMatrix",
    "NodeIndexedTensor",
    "NodeIndexedVector",
    "NodeInfo",
]
