from __future__ import annotations

import torch

from lm_saes.circuits.attribution import AttributionResult
from lm_saes.circuits.indexed_tensor import (
    NodeDimension,
    NodeIndexed,
    NodeIndexedMatrix,
    NodeIndexedVector,
    NodeInfo,
)
from lm_saes.core.serialize import dump, load


def _deep_equal(a: object, b: object) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def test_dump_attribution_result() -> None:
    targets = NodeDimension.from_node_infos([NodeInfo(key="layer_0", indices=torch.arange(3).unsqueeze(1))])
    sources = NodeDimension.from_node_infos([NodeInfo(key="layer_1", indices=torch.arange(4).unsqueeze(1))])
    activations_data = torch.arange(3, dtype=torch.float32)
    attribution_data = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    logits = torch.arange(5, dtype=torch.float32)
    probs = torch.arange(5, dtype=torch.float32) / 10.0
    qk_inner = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    sample = AttributionResult(
        activations=NodeIndexedVector.from_data(activations_data, dimensions=(targets,)),
        attribution=NodeIndexedMatrix.from_data(attribution_data, dimensions=(targets, sources)),
        logits=logits,
        probs=probs,
        prompt_token_ids=[1, 2, 3],
        prompt_tokens=["<bos>", "a", "b"],
        logit_token_ids=[4, 5],
        logit_tokens=["c", "d"],
        qk_trace_results=NodeIndexed(
            value=[
                NodeIndexed(
                    value=qk_inner,
                    dimensions=(
                        NodeDimension.from_node_infos([NodeInfo(key="q", indices=torch.arange(2).unsqueeze(1))]),
                        NodeDimension.from_node_infos([NodeInfo(key="k", indices=torch.arange(3).unsqueeze(1))]),
                    ),
                ),
            ],
            dimensions=(NodeDimension.from_node_infos([NodeInfo(key="target", indices=torch.arange(1).unsqueeze(1))]),),
        ),
    )

    expected = {
        "_version": "1",
        "data": {
            "activations": {
                "n_dims": 1,
                "data": activations_data,
                "dimensions": [
                    {
                        "layer_0": {
                            "key": "layer_0",
                            "indices": torch.arange(3).unsqueeze(1),
                            "offsets": torch.arange(3),
                        },
                    },
                ],
            },
            "attribution": {
                "n_dims": 2,
                "data": attribution_data,
                "dimensions": [
                    {
                        "layer_0": {
                            "key": "layer_0",
                            "indices": torch.arange(3).unsqueeze(1),
                            "offsets": torch.arange(3),
                        },
                    },
                    {
                        "layer_1": {
                            "key": "layer_1",
                            "indices": torch.arange(4).unsqueeze(1),
                            "offsets": torch.arange(4),
                        },
                    },
                ],
            },
            "logits": logits,
            "probs": probs,
            "prompt_token_ids": [1, 2, 3],
            "prompt_tokens": ["<bos>", "a", "b"],
            "logit_token_ids": [4, 5],
            "logit_tokens": ["c", "d"],
            "qk_trace_results": {
                "value": [
                    {
                        "value": qk_inner,
                        "dimensions": [
                            {
                                "q": {
                                    "key": "q",
                                    "indices": torch.arange(2).unsqueeze(1),
                                    "offsets": torch.arange(2),
                                },
                            },
                            {
                                "k": {
                                    "key": "k",
                                    "indices": torch.arange(3).unsqueeze(1),
                                    "offsets": torch.arange(3),
                                },
                            },
                        ],
                    },
                ],
                "dimensions": [
                    {
                        "target": {
                            "key": "target",
                            "indices": torch.arange(1).unsqueeze(1),
                            "offsets": torch.arange(1),
                        },
                    },
                ],
            },
        },
    }

    assert _deep_equal(dump(sample), expected)


def test_load_attribution_result() -> None:
    targets = NodeDimension.from_node_infos([NodeInfo(key="layer_0", indices=torch.arange(3).unsqueeze(1))])
    sources = NodeDimension.from_node_infos([NodeInfo(key="layer_1", indices=torch.arange(4).unsqueeze(1))])
    sample = AttributionResult(
        activations=NodeIndexedVector.from_data(torch.randn(3), dimensions=(targets,)),
        attribution=NodeIndexedMatrix.from_data(torch.randn(3, 4), dimensions=(targets, sources)),
        logits=torch.randn(5),
        probs=torch.softmax(torch.randn(5), dim=0),
        prompt_token_ids=[1, 2, 3],
        prompt_tokens=["<bos>", "a", "b"],
        logit_token_ids=[4, 5],
        logit_tokens=["c", "d"],
        qk_trace_results=NodeIndexed(
            value=[
                NodeIndexed(
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

    restored = load(dump(sample), AttributionResult)

    assert torch.equal(sample.logits, restored.logits)
    assert torch.equal(sample.probs, restored.probs)
    assert torch.equal(sample.activations.data, restored.activations.data)
    assert torch.equal(sample.attribution.data, restored.attribution.data)
    assert sample.prompt_token_ids == restored.prompt_token_ids
    assert sample.prompt_tokens == restored.prompt_tokens
    assert sample.logit_token_ids == restored.logit_token_ids
    assert sample.logit_tokens == restored.logit_tokens
    assert sample.qk_trace_results is not None and restored.qk_trace_results is not None
    for x, y in zip(sample.qk_trace_results.value, restored.qk_trace_results.value):
        assert torch.equal(x.value, y.value)
