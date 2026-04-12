"""Unit tests for PyTree generic type resolution in tree_map."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from lm_saes.core.pytree import PyTree

V = TypeVar("V")


@dataclass
class Dim(PyTree):
    """Mirrors NodeDimension: a non-generic PyTree used as a typed tuple element."""

    data: torch.Tensor
    tag: str


@dataclass
class Container(Generic[V], PyTree):
    """Mirrors Dimensioned[V]: generic PyTree with a typed-tuple PyTree field."""

    value: V
    dimensions: tuple[Dim, ...]


def test_generic_pytree_tree_map_tensor():
    """tree_map_tensor should round-trip a generic PyTree with a Tensor value."""
    dim = Dim(data=torch.tensor([1.0]), tag="d")
    original = Container(value=torch.tensor([1.0, 2.0, 3.0]), dimensions=(dim,))
    result = original.tree_map_tensor(lambda t: t * 2)
    assert torch.equal(result.value, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.equal(result.dimensions[0].data, torch.tensor([2.0]))


def test_generic_pytree_to_device():
    """to() (uses tree_map) should work on generic PyTree subclasses."""
    dim = Dim(data=torch.tensor([1.0]), tag="d")
    original = Container(value=torch.tensor([1.0]), dimensions=(dim,))
    result = original.to("cpu")
    assert torch.equal(result.value, original.value)
    assert torch.equal(result.dimensions[0].data, dim.data)


def test_generic_pytree_full_tensor():
    """full_tensor (uses tree_map) should work on generic PyTree subclasses."""
    dim = Dim(data=torch.tensor([1.0]), tag="d")
    original = Container(value=torch.tensor([1.0]), dimensions=(dim,))
    result = original.full_tensor()
    assert torch.equal(result.value, original.value)
    assert result.dimensions[0].tag == "d"


def test_generic_pytree_nested_list_full_tensor():
    """full_tensor on Container[list[Container[Tensor]]] — mirrors Dimensioned[list[Dimensioned[Tensor]]]."""
    inner_dim = Dim(data=torch.tensor([10.0]), tag="inner_d")
    inner = Container(value=torch.tensor([1.0, 2.0]), dimensions=(inner_dim,))

    outer_dim = Dim(data=torch.tensor([20.0]), tag="outer_d")
    outer = Container(value=[inner], dimensions=(outer_dim,))

    result = outer.full_tensor()

    assert isinstance(result.value, list)
    assert len(result.value) == 1
    assert torch.equal(result.value[0].value, torch.tensor([1.0, 2.0]))
    assert result.value[0].dimensions[0].tag == "inner_d"
    assert result.dimensions[0].tag == "outer_d"


def test_attribution_result_full_tensor():
    """full_tensor on real AttributionResult with qk_trace_results (parameterized generic field)."""
    from lm_saes.circuits.attribution import AttributionResult
    from lm_saes.circuits.indexed_tensor import (
        Dimensioned,
        DimensionedMatrix,
        DimensionedVector,
        NodeDimension,
        NodeInfo,
    )

    targets = NodeDimension.from_node_infos([NodeInfo(key="layer_0", indices=torch.arange(3).unsqueeze(1))])
    sources = NodeDimension.from_node_infos([NodeInfo(key="layer_1", indices=torch.arange(4).unsqueeze(1))])

    sample = AttributionResult(
        activations=DimensionedVector.from_data(torch.randn(3), dimensions=(targets,)),
        attribution=DimensionedMatrix.from_data(torch.randn(3, 4), dimensions=(targets, sources)),
        logits=torch.randn(5),
        probs=torch.softmax(torch.randn(5), dim=0),
        prompt_token_ids=[1, 2, 3],
        prompt_tokens=["a", "b", "c"],
        logit_token_ids=[4, 5],
        logit_tokens=["d", "e"],
        qk_trace_results=Dimensioned(
            value=[
                Dimensioned(
                    value=torch.randn(2, 3),
                    dimensions=(
                        NodeDimension.from_node_infos([NodeInfo(key="q", indices=torch.arange(2).unsqueeze(1))]),
                        NodeDimension.from_node_infos([NodeInfo(key="k", indices=torch.arange(3).unsqueeze(1))]),
                    ),
                ),
            ],
            dimensions=(NodeDimension.from_node_infos([NodeInfo(key="target", indices=torch.arange(1).unsqueeze(1))]),),
        ),
    )

    result = sample.full_tensor()

    assert isinstance(result, AttributionResult)
    assert torch.equal(result.activations.data, sample.activations.data)
    assert isinstance(result.qk_trace_results, Dimensioned)
    assert torch.equal(result.qk_trace_results.value[0].value, sample.qk_trace_results.value[0].value)
