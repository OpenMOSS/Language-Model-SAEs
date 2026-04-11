"""Backward-compat shim for :mod:`lm_saes.circuits.indexed_tensor`."""

from lm_saes.circuits.indexed_tensor import (
    Dimensioned,
    DimensionedMatrix,
    DimensionedTensor,
    DimensionedVector,
    Node,
    NodeDimension,
    NodeInfo,
)

Dimension = NodeDimension
NodeIndexedTensor = DimensionedTensor
NodeIndexedVector = DimensionedVector
NodeIndexedMatrix = DimensionedMatrix

__all__ = [
    "Dimension",
    "Dimensioned",
    "DimensionedMatrix",
    "DimensionedTensor",
    "DimensionedVector",
    "Node",
    "NodeDimension",
    "NodeIndexedMatrix",
    "NodeIndexedTensor",
    "NodeIndexedVector",
    "NodeInfo",
]
