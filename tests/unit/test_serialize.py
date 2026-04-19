from __future__ import annotations

import torch

from llamascopium.circuits.attribution import AttributionResult, QKTracingResult
from llamascopium.circuits.indexed_tensor import (
    NodeDimension,
    NodeIndexed,
    NodeIndexedMatrix,
    NodeIndexedVector,
    NodeInfo,
)
from llamascopium.core.serialize import dump, load


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
    qk_pair_inner = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    qk_q_marginal_inner = torch.arange(2, dtype=torch.float32)
    qk_k_marginal_inner = torch.arange(3, dtype=torch.float32)
    target_dim = NodeDimension.from_node_infos([NodeInfo(key="target", indices=torch.arange(1).unsqueeze(1))])
    sample = AttributionResult(
        activations=NodeIndexedVector.from_data(activations_data, dimensions=(targets,)),
        attribution=NodeIndexedMatrix.from_data(attribution_data, dimensions=(targets, sources)),
        logits=logits,
        probs=probs,
        prompt_token_ids=[1, 2, 3],
        prompt_tokens=["<bos>", "a", "b"],
        logit_token_ids=[4, 5],
        logit_tokens=["c", "d"],
        qk_trace_results=QKTracingResult(
            q_marginal=NodeIndexed(
                value=[
                    NodeIndexed(
                        value=qk_q_marginal_inner,
                        dimensions=(
                            NodeDimension.from_node_infos([NodeInfo(key="q", indices=torch.arange(2).unsqueeze(1))]),
                        ),
                    ),
                ],
                dimensions=(target_dim,),
            ),
            k_marginal=NodeIndexed(
                value=[
                    NodeIndexed(
                        value=qk_k_marginal_inner,
                        dimensions=(
                            NodeDimension.from_node_infos([NodeInfo(key="k", indices=torch.arange(3).unsqueeze(1))]),
                        ),
                    ),
                ],
                dimensions=(target_dim,),
            ),
            pairs=NodeIndexed(
                value=[
                    NodeIndexed(
                        value=qk_pair_inner,
                        dimensions=(
                            NodeDimension.from_node_infos([NodeInfo(key="q", indices=torch.arange(2).unsqueeze(1))]),
                            NodeDimension.from_node_infos([NodeInfo(key="k", indices=torch.arange(3).unsqueeze(1))]),
                        ),
                    ),
                ],
                dimensions=(target_dim,),
            ),
        ),
    )

    expected = {
        "_version": "2",
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
            "targets": None,
            "logits": logits,
            "probs": probs,
            "prompt_token_ids": [1, 2, 3],
            "prompt_tokens": ["<bos>", "a", "b"],
            "logit_token_ids": [4, 5],
            "logit_tokens": ["c", "d"],
            "qk_trace_results": {
                "q_marginal": {
                    "value": [
                        {
                            "value": qk_q_marginal_inner,
                            "dimensions": [
                                {
                                    "q": {
                                        "key": "q",
                                        "indices": torch.arange(2).unsqueeze(1),
                                        "offsets": torch.arange(2),
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
                "k_marginal": {
                    "value": [
                        {
                            "value": qk_k_marginal_inner,
                            "dimensions": [
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
                "pairs": {
                    "value": [
                        {
                            "value": qk_pair_inner,
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
        },
    }

    assert _deep_equal(dump(sample), expected)


def test_load_attribution_result() -> None:
    targets = NodeDimension.from_node_infos([NodeInfo(key="layer_0", indices=torch.arange(3).unsqueeze(1))])
    sources = NodeDimension.from_node_infos([NodeInfo(key="layer_1", indices=torch.arange(4).unsqueeze(1))])
    target_dim = NodeDimension.from_node_infos([NodeInfo(key="target", indices=torch.arange(1).unsqueeze(1))])
    sample = AttributionResult(
        activations=NodeIndexedVector.from_data(torch.randn(3), dimensions=(targets,)),
        attribution=NodeIndexedMatrix.from_data(torch.randn(3, 4), dimensions=(targets, sources)),
        logits=torch.randn(5),
        probs=torch.softmax(torch.randn(5), dim=0),
        prompt_token_ids=[1, 2, 3],
        prompt_tokens=["<bos>", "a", "b"],
        logit_token_ids=[4, 5],
        logit_tokens=["c", "d"],
        qk_trace_results=QKTracingResult(
            q_marginal=NodeIndexed(
                value=[
                    NodeIndexed(
                        value=torch.randn(2),
                        dimensions=(
                            NodeDimension.from_node_infos([NodeInfo(key="q", indices=torch.arange(2).unsqueeze(1))]),
                        ),
                    ),
                ],
                dimensions=(target_dim,),
            ),
            k_marginal=NodeIndexed(
                value=[
                    NodeIndexed(
                        value=torch.randn(3),
                        dimensions=(
                            NodeDimension.from_node_infos([NodeInfo(key="k", indices=torch.arange(3).unsqueeze(1))]),
                        ),
                    ),
                ],
                dimensions=(target_dim,),
            ),
            pairs=NodeIndexed(
                value=[
                    NodeIndexed(
                        value=torch.randn(2, 3),
                        dimensions=(
                            NodeDimension.from_node_infos([NodeInfo(key="q", indices=torch.arange(2).unsqueeze(1))]),
                            NodeDimension.from_node_infos([NodeInfo(key="k", indices=torch.arange(3).unsqueeze(1))]),
                        ),
                    ),
                ],
                dimensions=(target_dim,),
            ),
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
    for x, y in zip(sample.qk_trace_results.pairs.value, restored.qk_trace_results.pairs.value):
        assert torch.equal(x.value, y.value)
    for x, y in zip(sample.qk_trace_results.q_marginal.value, restored.qk_trace_results.q_marginal.value):
        assert torch.equal(x.value, y.value)
    for x, y in zip(sample.qk_trace_results.k_marginal.value, restored.qk_trace_results.k_marginal.value):
        assert torch.equal(x.value, y.value)


def test_load_legacy_attribution_result_with_bare_nodeindexed_qk() -> None:
    """Legacy MongoDB blobs store ``qk_trace_results`` as a bare
    ``NodeIndexed[list[NodeIndexed[torch.Tensor]]]`` (pre-``QKTracingResult``
    refactor). Verify such blobs still load, with the pair data promoted into
    ``QKTracingResult.pairs`` and empty Q/K marginals filled in.
    """
    targets = NodeDimension.from_node_infos([NodeInfo(key="layer_0", indices=torch.arange(3).unsqueeze(1))])
    sources = NodeDimension.from_node_infos([NodeInfo(key="layer_1", indices=torch.arange(4).unsqueeze(1))])
    legacy_pairs = NodeIndexed(
        value=[
            NodeIndexed(
                value=torch.tensor([1.5, 2.5]),
                dimensions=(
                    NodeDimension.from_node_infos([NodeInfo(key="q", indices=torch.arange(2).unsqueeze(1))]),
                    NodeDimension.from_node_infos([NodeInfo(key="k", indices=torch.arange(2).unsqueeze(1))]),
                ),
            ),
            NodeIndexed(
                value=torch.tensor([3.5]),
                dimensions=(
                    NodeDimension.from_node_infos([NodeInfo(key="q", indices=torch.arange(1).unsqueeze(1))]),
                    NodeDimension.from_node_infos([NodeInfo(key="k", indices=torch.arange(1).unsqueeze(1))]),
                ),
            ),
        ],
        dimensions=(NodeDimension.from_node_infos([NodeInfo(key="target", indices=torch.arange(2).unsqueeze(1))]),),
    )

    # Build a dump by unstructuring the legacy shape directly and wrapping
    # it in a ``{"_version": "1", "data": ...}`` envelope — the pre-refactor
    # FORMAT_VERSION. The migration framework picks up ``_version`` < current
    # and invokes the ``QKTracingResult._v1_to_v2`` hook.
    from llamascopium.core.serialize import unstructure

    legacy_blob = {
        "_version": "1",
        "data": {
            "activations": unstructure(NodeIndexedVector.from_data(torch.randn(3), dimensions=(targets,))),
            "attribution": unstructure(NodeIndexedMatrix.from_data(torch.randn(3, 4), dimensions=(targets, sources))),
            "logits": torch.randn(5),
            "probs": torch.softmax(torch.randn(5), dim=0),
            "prompt_token_ids": [1, 2, 3],
            "prompt_tokens": ["<bos>", "a", "b"],
            "logit_token_ids": [4, 5],
            "logit_tokens": ["c", "d"],
            "qk_trace_results": unstructure(legacy_pairs),  # ← the pre-refactor shape
        },
    }

    restored = load(legacy_blob, AttributionResult)

    assert restored.qk_trace_results is not None
    assert isinstance(restored.qk_trace_results, QKTracingResult)
    # Pairs are preserved.
    assert len(restored.qk_trace_results.pairs.value) == 2
    assert torch.equal(
        restored.qk_trace_results.pairs.value[0].value,
        torch.tensor([1.5, 2.5]),
    )
    assert torch.equal(
        restored.qk_trace_results.pairs.value[1].value,
        torch.tensor([3.5]),
    )
    # Marginals are zero-valued placeholders: each slot reuses the pair's
    # Q-role / K-role NodeDimension but fills the attribution tensor with
    # zeros (no authoritative marginal existed in the legacy format).
    assert len(restored.qk_trace_results.q_marginal.value) == 2
    assert len(restored.qk_trace_results.k_marginal.value) == 2
    for legacy_slot, q_slot, k_slot in zip(
        legacy_pairs.value,
        restored.qk_trace_results.q_marginal.value,
        restored.qk_trace_results.k_marginal.value,
    ):
        assert q_slot.value.shape == (len(legacy_slot.dimensions[0]),)
        assert k_slot.value.shape == (len(legacy_slot.dimensions[1]),)
        assert torch.all(q_slot.value == 0)
        assert torch.all(k_slot.value == 0)
