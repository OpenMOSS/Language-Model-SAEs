"""Unit tests for NodeIndexedTensor, NodeIndexedVector, and NodeIndexedMatrix.

Index convention
----------------
``NodeInfo.indices`` must be a 2-D tensor of shape ``(n_elements, d_index)``.
For the common 1-D logical-key case (e.g. a token position) that means shape
``(n, 1)``, which is what the helper :func:`_ni` produces.  For a 2-D logical
key (e.g. (token_pos, feature_idx)) use :func:`_ni2`.
"""

import pytest
import torch

from lm_saes.backend.language_model import (
    NodeIndexedMatrix,
    NodeIndexedTensor,
    NodeIndexedVector,
    NodeInfo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ni(key: str, *positions: int) -> NodeInfo:
    """Build a NodeInfo with 2-D indices ``(n_elements, 1)`` from plain ints."""
    return NodeInfo(key=key, indices=torch.tensor([[p] for p in positions], dtype=torch.long))


def _ni2(key: str, *pairs: tuple[int, int]) -> NodeInfo:
    """Build a NodeInfo with 2-D indices ``(n_elements, 2)`` from (a, b) pairs."""
    return NodeInfo(key=key, indices=torch.tensor(list(pairs), dtype=torch.long))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_from_node_infos_allocates_zeros_of_correct_size():
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0, 3), _ni("b", 1)],),
    )
    assert t.data.shape == (3,)
    assert torch.all(t.data == 0)


def test_from_data_stores_values_exactly():
    data = torch.tensor([10.0, 20.0, 30.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        node_infos=([_ni("a", 0, 3), _ni("b", 7)],),
    )
    assert torch.equal(t.data, data)


def test_node_mappings_store_correct_offsets():
    # "a" is registered first → data offsets 0, 1
    # "b" is registered next  → data offset  2
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0, 3), _ni("b", 7)],),
    )
    assert torch.equal(t.node_mappings[0]["a"].offsets, torch.tensor([0, 1]))
    assert torch.equal(t.node_mappings[0]["b"].offsets, torch.tensor([2]))


def test_node_mappings_store_correct_indices():
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0, 3), _ni("b", 7)],),
    )
    assert torch.equal(t.node_mappings[0]["a"].indices, torch.tensor([[0], [3]]))
    assert torch.equal(t.node_mappings[0]["b"].indices, torch.tensor([[7]]))


# ---------------------------------------------------------------------------
# _nodes_to_offsets
# ---------------------------------------------------------------------------


def test_nodes_to_offsets_full_first_node():
    # "a" → offsets 0, 1 ;  "b" → offsets 2, 3
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    assert torch.equal(t._nodes_to_offsets([_ni("a", 0, 3)], 0), torch.tensor([0, 1]))


def test_nodes_to_offsets_full_second_node():
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    assert torch.equal(t._nodes_to_offsets([_ni("b", 1, 2)], 0), torch.tensor([2, 3]))


def test_nodes_to_offsets_partial_first_element():
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    assert torch.equal(t._nodes_to_offsets([_ni("a", 0)], 0), torch.tensor([0]))


def test_nodes_to_offsets_partial_second_element():
    # logical index 3 is the *second* element of "a" → data offset 1
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    assert torch.equal(t._nodes_to_offsets([_ni("a", 3)], 0), torch.tensor([1]))


def test_nodes_to_offsets_cross_node_selection():
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    result = t._nodes_to_offsets([_ni("a", 3), _ni("b", 1)], 0)
    assert torch.equal(result, torch.tensor([1, 2]))


# ---------------------------------------------------------------------------
# _offsets_to_nodes
# ---------------------------------------------------------------------------


def test_offsets_to_nodes_full_first_node():
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    node_infos = t._offsets_to_nodes(torch.tensor([0, 1]), 0)
    assert len(node_infos) == 1
    assert node_infos[0].key == "a"
    assert torch.equal(node_infos[0].indices, torch.tensor([[0], [3]]))


def test_offsets_to_nodes_full_second_node():
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    node_infos = t._offsets_to_nodes(torch.tensor([2, 3]), 0)
    assert len(node_infos) == 1
    assert node_infos[0].key == "b"
    assert torch.equal(node_infos[0].indices, torch.tensor([[1], [2]]))


def test_offsets_to_nodes_single_offset_second_element():
    # Offset 1 is the second element of "a": logical index 3
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    node_infos = t._offsets_to_nodes(torch.tensor([1]), 0)
    assert len(node_infos) == 1
    assert node_infos[0].key == "a"
    assert torch.equal(node_infos[0].indices, torch.tensor([[3]]))


def test_offsets_to_nodes_across_two_nodes():
    # Offset 0 → a→0,  offset 2 → b→1
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    node_infos = t._offsets_to_nodes(torch.tensor([0, 2]), 0)
    assert len(node_infos) == 2
    assert node_infos[0].key == "a"
    assert torch.equal(node_infos[0].indices, torch.tensor([[0]]))
    assert node_infos[1].key == "b"
    assert torch.equal(node_infos[1].indices, torch.tensor([[1]]))


# ---------------------------------------------------------------------------
# Duality: _nodes_to_offsets ↔ _offsets_to_nodes
# ---------------------------------------------------------------------------


def test_nodes_to_offsets_then_to_nodes_full_round_trip():
    # x: offsets 0,1,2 ; y: offsets 3,4
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("x", 5, 7, 9), _ni("y", 1, 3)],),
    )
    original = [_ni("x", 5, 7, 9), _ni("y", 1, 3)]
    offsets = t._nodes_to_offsets(original, 0)
    recovered = t._offsets_to_nodes(offsets, 0)
    assert len(recovered) == 2
    assert recovered[0].key == "x"
    assert torch.equal(recovered[0].indices, torch.tensor([[5], [7], [9]]))
    assert recovered[1].key == "y"
    assert torch.equal(recovered[1].indices, torch.tensor([[1], [3]]))


def test_nodes_to_offsets_then_to_nodes_partial_round_trip():
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("x", 5, 7, 9), _ni("y", 1, 3)],),
    )
    original = [_ni("x", 7), _ni("y", 1)]
    offsets = t._nodes_to_offsets(original, 0)
    recovered = t._offsets_to_nodes(offsets, 0)
    assert len(recovered) == 2
    assert recovered[0].key == "x"
    assert torch.equal(recovered[0].indices, torch.tensor([[7]]))
    assert recovered[1].key == "y"
    assert torch.equal(recovered[1].indices, torch.tensor([[1]]))


def test_offsets_to_nodes_then_to_offsets_round_trip():
    # x[5]=0, x[7]=1, x[9]=2, y[1]=3, y[3]=4
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("x", 5, 7, 9), _ni("y", 1, 3)],),
    )
    original_offsets = torch.tensor([0, 2, 3])  # x→5, x→9, y→1
    node_infos = t._offsets_to_nodes(original_offsets, 0)
    recovered_offsets = t._nodes_to_offsets(node_infos, 0)
    assert torch.equal(recovered_offsets, original_offsets)


def test_duality_with_merged_node():
    # Register "a" in two separate add_elements calls so the merge path is exercised.
    t = NodeIndexedTensor.from_node_infos(node_infos=([_ni("a", 0, 3)],))
    t._add_elements([_ni("a", 7)], dim=0)
    # a: offsets 0,1,2 for logical 0,3,7
    offsets = t._nodes_to_offsets([_ni("a", 0, 3, 7)], 0)
    assert torch.equal(offsets, torch.tensor([0, 1, 2]))
    recovered = t._offsets_to_nodes(offsets, 0)
    assert len(recovered) == 1
    assert recovered[0].key == "a"
    assert torch.equal(recovered[0].indices, torch.tensor([[0], [3], [7]]))


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------


def test_getitem_first_node_exact_values():
    data = torch.tensor([10.0, 20.0, 30.0, 40.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    result = t[[_ni("a", 0, 3)]]
    assert torch.equal(result.data, torch.tensor([10.0, 20.0]))


def test_getitem_second_node_exact_values():
    data = torch.tensor([10.0, 20.0, 30.0, 40.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    result = t[[_ni("b", 1, 2)]]
    assert torch.equal(result.data, torch.tensor([30.0, 40.0]))


def test_getitem_reordered_cross_node_exact_values():
    # Ask for "b"→2 before "a"→0 to verify arbitrary ordering is respected
    data = torch.tensor([10.0, 20.0, 30.0, 40.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    result = t[[_ni("b", 2), _ni("a", 0)]]
    assert torch.equal(result.data, torch.tensor([40.0, 10.0]))


def test_getitem_single_element_from_multi_element_node():
    data = torch.tensor([10.0, 20.0, 30.0, 40.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    # "a"→3 is the *second* element of "a" → data offset 1 → value 20
    result = t[[_ni("a", 3)]]
    assert torch.equal(result.data, torch.tensor([20.0]))


def test_getitem_none_returns_full_tensor():
    data = torch.tensor([1.0, 2.0, 3.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        node_infos=([_ni("a", 0), _ni("b", 5), _ni("c", 2)],),
    )
    result = t[None]
    assert torch.equal(result.data, data)


# ---------------------------------------------------------------------------
# __setitem__
# ---------------------------------------------------------------------------


def test_setitem_full_node_exact_values():
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    t[[_ni("a", 0, 3)]] = torch.tensor([7.0, 8.0])
    assert t.data[0].item() == 7.0  # offset 0  →  "a"→0
    assert t.data[1].item() == 8.0  # offset 1  →  "a"→3
    assert t.data[2].item() == 0.0  # offset 2  →  "b"→1  (untouched)
    assert t.data[3].item() == 0.0  # offset 3  →  "b"→2  (untouched)


def test_setitem_partial_node_leaves_rest_unchanged():
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    t[[_ni("a", 3)]] = torch.tensor([99.0])
    assert t.data[0].item() == 0.0  # "a"→0 (offset 0) untouched
    assert t.data[1].item() == 99.0  # "a"→3 (offset 1) written


def test_setitem_none_sets_all():
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0), _ni("b", 5)],),
    )
    t[None] = torch.tensor([3.0, 7.0])
    assert torch.equal(t.data, torch.tensor([3.0, 7.0]))


def test_setitem_then_getitem_round_trip():
    t = NodeIndexedTensor.from_node_infos(
        node_infos=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    t[[_ni("b", 1, 2)]] = torch.tensor([11.0, 22.0])
    result = t[[_ni("b", 1, 2)]]
    assert torch.equal(result.data, torch.tensor([11.0, 22.0]))


# ---------------------------------------------------------------------------
# _add_elements
# ---------------------------------------------------------------------------


def test_add_elements_expands_size():
    t = NodeIndexedTensor.from_node_infos(node_infos=([_ni("a", 0, 1)],))
    t._add_elements([_ni("b", 5, 6)], dim=0)
    assert t.data.shape == (4,)


def test_add_elements_new_node_gets_correct_offsets_and_indices():
    t = NodeIndexedTensor.from_node_infos(node_infos=([_ni("a", 0, 1)],))
    t._add_elements([_ni("b", 5, 6)], dim=0)
    assert torch.equal(t.node_mappings[0]["b"].offsets, torch.tensor([2, 3]))
    assert torch.equal(t.node_mappings[0]["b"].indices, torch.tensor([[5], [6]]))


def test_add_elements_existing_node_is_merged():
    t = NodeIndexedTensor.from_node_infos(node_infos=([_ni("a", 0)],))
    t._add_elements([_ni("a", 7)], dim=0)
    assert t.data.shape == (2,)
    assert torch.equal(t.node_mappings[0]["a"].offsets, torch.tensor([0, 1]))
    assert torch.equal(t.node_mappings[0]["a"].indices, torch.tensor([[0], [7]]))


def test_add_elements_preserves_existing_values():
    t = NodeIndexedTensor.from_data(
        data=torch.tensor([5.0, 6.0]),
        node_infos=([_ni("a", 0, 1)],),
    )
    t._add_elements([_ni("b", 9)], dim=0)
    assert torch.equal(t.data[:2], torch.tensor([5.0, 6.0]))
    assert t.data[2].item() == 0.0


def test_add_elements_newly_added_node_is_addressable():
    t = NodeIndexedTensor.from_data(
        data=torch.tensor([5.0, 6.0]),
        node_infos=([_ni("a", 0, 1)],),
    )
    t._add_elements([_ni("b", 9)], dim=0)
    t[[_ni("b", 9)]] = torch.tensor([42.0])
    result = t[[_ni("b", 9)]]
    assert torch.equal(result.data, torch.tensor([42.0]))


# ---------------------------------------------------------------------------
# NodeIndexedMatrix (2-D)
# ---------------------------------------------------------------------------


def test_matrix_shape():
    m = NodeIndexedMatrix.from_node_infos(
        node_infos=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    assert m.data.shape == (2, 3)


def test_matrix_getitem_row_slice_exact_values():
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m = NodeIndexedMatrix.from_data(
        data=data,
        node_infos=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    result = m[[_ni("r", 0)], None]
    assert torch.equal(result.data, torch.tensor([[1.0, 2.0, 3.0]]))


def test_matrix_getitem_col_slice_exact_values():
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m = NodeIndexedMatrix.from_data(
        data=data,
        node_infos=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    result = m[None, [_ni("c", 1)]]
    assert torch.equal(result.data, torch.tensor([[2.0], [5.0]]))


def test_matrix_getitem_submatrix_exact_values():
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m = NodeIndexedMatrix.from_data(
        data=data,
        node_infos=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    # Row 1 ("r"→1), cols 0 and 2 ("c"→0, "c"→2) → [[4, 6]]
    result = m[[_ni("r", 1)], [_ni("c", 0, 2)]]
    assert torch.equal(result.data, torch.tensor([[4.0, 6.0]]))


def test_matrix_getitem_none_none_returns_full_data():
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    m = NodeIndexedMatrix.from_data(
        data=data,
        node_infos=([_ni("r", 0, 1)], [_ni("c", 0, 1)]),
    )
    result = m[None, None]
    assert torch.equal(result.data, data)


def test_matrix_setitem_submatrix_exact_values():
    m = NodeIndexedMatrix.from_node_infos(
        node_infos=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    m[[_ni("r", 0)], [_ni("c", 1, 2)]] = torch.tensor([[99.0, 88.0]])
    assert m.data[0, 0].item() == 0.0  # "c"→0 untouched
    assert m.data[0, 1].item() == 99.0  # "c"→1 written
    assert m.data[0, 2].item() == 88.0  # "c"→2 written
    assert m.data[1, 0].item() == 0.0  # row "r"→1 entirely untouched
    assert m.data[1, 1].item() == 0.0
    assert m.data[1, 2].item() == 0.0


def test_matrix_setitem_then_getitem_round_trip():
    m = NodeIndexedMatrix.from_node_infos(
        node_infos=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    m[[_ni("r", 1)], [_ni("c", 0, 2)]] = torch.tensor([[11.0, 22.0]])
    result = m[[_ni("r", 1)], [_ni("c", 0, 2)]]
    assert torch.equal(result.data, torch.tensor([[11.0, 22.0]]))


def test_matrix_add_target_then_addressable():
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m = NodeIndexedMatrix.from_data(
        data=data,
        node_infos=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    m.add_target([_ni("r2", 10)])
    m[[_ni("r2", 10)], None] = torch.tensor([[7.0, 8.0, 9.0]])
    result = m[[_ni("r2", 10)], None]
    assert torch.equal(result.data, torch.tensor([[7.0, 8.0, 9.0]]))


# ---------------------------------------------------------------------------
# NodeIndexedVector – topk exercises _offsets_to_nodes
# ---------------------------------------------------------------------------


def test_vector_topk_values_and_nodes_same_key():
    # offsets: p→5=0, p→7=1, p→9=2, q→1=3
    data = torch.tensor([3.0, 1.0, 4.0, 2.0])
    v = NodeIndexedVector.from_data(
        data=data,
        node_infos=([_ni("p", 5, 7, 9), _ni("q", 1)],),
    )
    values, node_infos = v.topk(k=2)
    # top-2: offset 2 (4.0 → p→9) then offset 0 (3.0 → p→5)
    assert torch.equal(values, torch.tensor([4.0, 3.0]))
    # both belong to "p" (consecutive), so they collapse into one NodeInfo
    assert len(node_infos) == 1
    assert node_infos[0].key == "p"
    assert torch.equal(node_infos[0].indices, torch.tensor([[9], [5]]))


def test_vector_topk_values_and_nodes_different_keys():
    # offsets: a→0=0, b→1=1, a→2=2
    data = torch.tensor([1.0, 4.0, 3.0])
    v = NodeIndexedVector.from_data(
        data=data,
        node_infos=([_ni("a", 0), _ni("b", 1), _ni("a", 2)],),
    )
    values, node_infos = v.topk(k=2)
    # top-2: offset 1 (4.0 → b→1) then offset 2 (3.0 → a→2)
    assert torch.equal(values, torch.tensor([4.0, 3.0]))
    assert len(node_infos) == 2
    assert node_infos[0].key == "b"
    assert torch.equal(node_infos[0].indices, torch.tensor([[1]]))
    assert node_infos[1].key == "a"
    assert torch.equal(node_infos[1].indices, torch.tensor([[2]]))


# ---------------------------------------------------------------------------
# d_index = 2 (2-D logical keys, e.g. (token_pos, feature_idx))
# ---------------------------------------------------------------------------
#
# Setup:
#   node "feat": 3 active (token_pos, feature_idx) pairs → (0,1), (2,3), (2,5)
#                  offsets 0, 1, 2
#   node "err":  2 pairs → (0,0), (1,0)
#                  offsets 3, 4
#
# inv_indices["feat"] has shape (3, 6):
#   [0,1]→0,  [2,3]→1,  [2,5]→2
# inv_indices["err"]  has shape (2, 1):
#   [0,0]→0,  [1,0]→1
# ---------------------------------------------------------------------------


def _make_2d_tensor() -> NodeIndexedTensor:
    return NodeIndexedTensor.from_node_infos(
        node_infos=(
            [
                _ni2("feat", (0, 1), (2, 3), (2, 5)),
                _ni2("err", (0, 0), (1, 0)),
            ],
        ),
    )


def test_2d_node_mappings_store_correct_indices():
    t = _make_2d_tensor()
    assert torch.equal(
        t.node_mappings[0]["feat"].indices,
        torch.tensor([[0, 1], [2, 3], [2, 5]]),
    )
    assert torch.equal(
        t.node_mappings[0]["err"].indices,
        torch.tensor([[0, 0], [1, 0]]),
    )


def test_2d_node_mappings_store_correct_offsets():
    t = _make_2d_tensor()
    assert torch.equal(t.node_mappings[0]["feat"].offsets, torch.tensor([0, 1, 2]))
    assert torch.equal(t.node_mappings[0]["err"].offsets, torch.tensor([3, 4]))


def test_2d_nodes_to_offsets_full_node():
    t = _make_2d_tensor()
    offsets = t._nodes_to_offsets([_ni2("feat", (0, 1), (2, 3), (2, 5))], 0)
    assert torch.equal(offsets, torch.tensor([0, 1, 2]))


def test_2d_nodes_to_offsets_partial_selection():
    t = _make_2d_tensor()
    # (2, 5) is the third element of "feat" → offset 2
    offsets = t._nodes_to_offsets([_ni2("feat", (2, 5))], 0)
    assert torch.equal(offsets, torch.tensor([2]))


def test_2d_nodes_to_offsets_cross_node():
    t = _make_2d_tensor()
    # feat→(2,3)=offset 1, err→(1,0)=offset 4
    offsets = t._nodes_to_offsets([_ni2("feat", (2, 3)), _ni2("err", (1, 0))], 0)
    assert torch.equal(offsets, torch.tensor([1, 4]))


def test_2d_offsets_to_nodes_full_feat_node():
    t = _make_2d_tensor()
    node_infos = t._offsets_to_nodes(torch.tensor([0, 1, 2]), 0)
    assert len(node_infos) == 1
    assert node_infos[0].key == "feat"
    assert torch.equal(node_infos[0].indices, torch.tensor([[0, 1], [2, 3], [2, 5]]))


def test_2d_offsets_to_nodes_partial_feat():
    t = _make_2d_tensor()
    # offset 2 → feat→(2,5)
    node_infos = t._offsets_to_nodes(torch.tensor([2]), 0)
    assert len(node_infos) == 1
    assert node_infos[0].key == "feat"
    assert torch.equal(node_infos[0].indices, torch.tensor([[2, 5]]))


def test_2d_offsets_to_nodes_full_err_node():
    t = _make_2d_tensor()
    node_infos = t._offsets_to_nodes(torch.tensor([3, 4]), 0)
    assert len(node_infos) == 1
    assert node_infos[0].key == "err"
    assert torch.equal(node_infos[0].indices, torch.tensor([[0, 0], [1, 0]]))


def test_2d_duality_nodes_to_offsets_to_nodes():
    t = _make_2d_tensor()
    original = [_ni2("feat", (0, 1), (2, 5)), _ni2("err", (0, 0))]
    offsets = t._nodes_to_offsets(original, 0)
    recovered = t._offsets_to_nodes(offsets, 0)
    assert len(recovered) == 2
    assert recovered[0].key == "feat"
    assert torch.equal(recovered[0].indices, torch.tensor([[0, 1], [2, 5]]))
    assert recovered[1].key == "err"
    assert torch.equal(recovered[1].indices, torch.tensor([[0, 0]]))


def test_2d_duality_offsets_to_nodes_to_offsets():
    t = _make_2d_tensor()
    # feat→(0,1)=0, feat→(2,5)=2, err→(0,0)=3
    original_offsets = torch.tensor([0, 2, 3])
    node_infos = t._offsets_to_nodes(original_offsets, 0)
    recovered = t._nodes_to_offsets(node_infos, 0)
    assert torch.equal(recovered, original_offsets)


def test_2d_getitem_exact_values():
    data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        node_infos=(
            [
                _ni2("feat", (0, 1), (2, 3), (2, 5)),
                _ni2("err", (0, 0), (1, 0)),
            ],
        ),
    )
    # feat→(0,1)=offset 0=10, feat→(2,5)=offset 2=30
    result = t[[_ni2("feat", (0, 1), (2, 5))]]
    assert torch.equal(result.data, torch.tensor([10.0, 30.0]))


def test_2d_setitem_exact_values():
    t = NodeIndexedTensor.from_node_infos(
        node_infos=(
            [
                _ni2("feat", (0, 1), (2, 3), (2, 5)),
                _ni2("err", (0, 0), (1, 0)),
            ],
        ),
    )
    t[[_ni2("err", (0, 0), (1, 0))]] = torch.tensor([7.0, 8.0])
    assert t.data[0].item() == 0.0  # feat (untouched)
    assert t.data[3].item() == 7.0  # err→(0,0)
    assert t.data[4].item() == 8.0  # err→(1,0)


def test_2d_topk_exact_values_and_nodes():
    data = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
    v = NodeIndexedVector.from_data(
        data=data,
        node_infos=(
            [
                _ni2("feat", (0, 1), (2, 3), (2, 5)),
                _ni2("err", (0, 0), (1, 0)),
            ],
        ),
    )
    values, node_infos = v.topk(k=3)
    # top-3: offset 1 (5.0→feat(2,3)), offset 3 (4.0→err(0,0)), offset 2 (3.0→feat(2,5))
    assert torch.equal(values, torch.tensor([5.0, 4.0, 3.0]))
    assert len(node_infos) == 3
    assert node_infos[0].key == "feat"
    assert torch.equal(node_infos[0].indices, torch.tensor([[2, 3]]))
    assert node_infos[1].key == "err"
    assert torch.equal(node_infos[1].indices, torch.tensor([[0, 0]]))
    assert node_infos[2].key == "feat"
    assert torch.equal(node_infos[2].indices, torch.tensor([[2, 5]]))


# ---------------------------------------------------------------------------
# matmul
# ---------------------------------------------------------------------------
#
# Abbreviations used in comments below:
#   v  = NodeIndexedVector  (shape (n,))
#   M  = NodeIndexedMatrix  (shape (r, c))
# ---------------------------------------------------------------------------


def _make_vector(key: str, *positions: int, values: list[float]) -> NodeIndexedVector:
    return NodeIndexedVector.from_data(
        data=torch.tensor(values, dtype=torch.float32),
        node_infos=([_ni(key, *positions)],),
    )


def _make_matrix(
    row_key: str,
    row_positions: list[int],
    col_key: str,
    col_positions: list[int],
    data: list[list[float]],
) -> NodeIndexedMatrix:
    return NodeIndexedMatrix.from_data(
        data=torch.tensor(data, dtype=torch.float32),
        node_infos=([_ni(row_key, *row_positions)], [_ni(col_key, *col_positions)]),
    )


# ---- vector @ matrix → vector -------------------------------------------


def test_vector_matmul_matrix_data_exact():
    # v = [1, 2, 3], M = [[1,0],[0,1],[1,1]]
    # v @ M = [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
    v = _make_vector("a", 0, 1, 2, values=[1.0, 2.0, 3.0])
    M = _make_matrix("a", [0, 1, 2], "b", [0, 1], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = v.matmul(M)
    assert isinstance(result, NodeIndexedVector)
    assert torch.equal(result.data, torch.tensor([4.0, 5.0]))


def test_vector_matmul_matrix_inherits_col_node_infos():
    v = _make_vector("a", 0, 1, 2, values=[1.0, 2.0, 3.0])
    M = _make_matrix("a", [0, 1, 2], "b", [10, 20], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = v.matmul(M)
    assert result.node_infos[0] == M.node_infos[1]


def test_vector_matmul_matrix_operator():
    v = _make_vector("a", 0, 1, values=[3.0, 4.0])
    M = _make_matrix("a", [0, 1], "b", [0, 1], [[1.0, 2.0], [3.0, 4.0]])
    result = v @ M
    assert isinstance(result, NodeIndexedVector)
    # [3,4] @ [[1,2],[3,4]] = [3+12, 6+16] = [15, 22]
    assert torch.equal(result.data, torch.tensor([15.0, 22.0]))


# ---- vector @ vector → scalar -------------------------------------------


def test_vector_matmul_vector_dot_product():
    u = _make_vector("a", 0, 1, 2, values=[1.0, 2.0, 3.0])
    v = _make_vector("a", 0, 1, 2, values=[4.0, 5.0, 6.0])
    result = u.matmul(v)
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert isinstance(result, float)
    assert result == pytest.approx(32.0)


def test_vector_matmul_vector_orthogonal():
    u = _make_vector("a", 0, 1, values=[1.0, 0.0])
    v = _make_vector("a", 0, 1, values=[0.0, 1.0])
    assert u.matmul(v) == pytest.approx(0.0)


# ---- matrix @ vector → vector -------------------------------------------


def test_matrix_matmul_vector_data_exact():
    # M = [[1,2,3],[4,5,6]], v = [1,0,1]
    # M @ v = [1+0+3, 4+0+6] = [4, 10]
    M = _make_matrix("r", [0, 1], "c", [0, 1, 2], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v = _make_vector("c", 0, 1, 2, values=[1.0, 0.0, 1.0])
    result = M.matmul(v)
    assert isinstance(result, NodeIndexedVector)
    assert torch.equal(result.data, torch.tensor([4.0, 10.0]))


def test_matrix_matmul_vector_inherits_row_node_infos():
    M = _make_matrix("r", [5, 7], "c", [0, 1], [[1.0, 0.0], [0.0, 1.0]])
    v = _make_vector("c", 0, 1, values=[2.0, 3.0])
    result = M.matmul(v)
    assert result.node_infos[0] == M.node_infos[0]


def test_matrix_matmul_vector_operator():
    M = _make_matrix("r", [0, 1], "c", [0, 1], [[2.0, 0.0], [0.0, 3.0]])
    v = _make_vector("c", 0, 1, values=[4.0, 5.0])
    result = M @ v
    # [[2,0],[0,3]] @ [4,5] = [8, 15]
    assert isinstance(result, NodeIndexedVector)
    assert torch.equal(result.data, torch.tensor([8.0, 15.0]))


# ---- matrix @ matrix → matrix -------------------------------------------


def test_matrix_matmul_matrix_data_exact():
    # A (2×3) @ B (3×2) → C (2×2)
    # A = [[1,0,0],[0,1,0]], B = [[1,2],[3,4],[5,6]]
    # C = [[1,2],[3,4]]
    A = _make_matrix("r", [0, 1], "m", [0, 1, 2], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    B = _make_matrix("m", [0, 1, 2], "c", [0, 1], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = A.matmul(B)
    assert isinstance(result, NodeIndexedMatrix)
    assert torch.equal(result.data, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_matrix_matmul_matrix_inherits_outer_node_infos():
    A = _make_matrix("r", [10, 20], "m", [0, 1, 2], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    B = _make_matrix("m", [0, 1, 2], "c", [30, 40], [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    result = A @ B
    assert result.node_infos[0] == A.node_infos[0]  # row nodes from A
    assert result.node_infos[1] == B.node_infos[1]  # col nodes from B


def test_matrix_matmul_matrix_identity():
    I = _make_matrix("r", [0, 1], "c", [0, 1], [[1.0, 0.0], [0.0, 1.0]])
    M = _make_matrix("c", [0, 1], "d", [0, 1], [[3.0, 4.0], [5.0, 6.0]])
    result = I @ M
    assert torch.equal(result.data, M.data)


def test_matrix_matmul_matrix_operator():
    A = _make_matrix("r", [0, 1], "m", [0, 1], [[1.0, 2.0], [3.0, 4.0]])
    B = _make_matrix("m", [0, 1], "c", [0, 1], [[5.0, 6.0], [7.0, 8.0]])
    result = A @ B
    # [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    assert torch.equal(result.data, torch.tensor([[19.0, 22.0], [43.0, 50.0]]))


# ---- _check_node_matching -----------------------------------------------


def test_matmul_check_node_matching_passes_when_equal():
    v = _make_vector("a", 0, 1, values=[1.0, 2.0])
    M = _make_matrix("a", [0, 1], "b", [0, 1], [[1.0, 0.0], [0.0, 1.0]])
    # Should not raise
    result = v.matmul(M, _check_node_matching=True)
    assert isinstance(result, NodeIndexedVector)


def test_matmul_check_node_matching_raises_when_mismatched():
    v = _make_vector("a", 0, 1, values=[1.0, 2.0])
    # Matrix row key is "x" not "a" → mismatch
    M = _make_matrix("x", [0, 1], "b", [0, 1], [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="Node matching failed"):
        v.matmul(M, _check_node_matching=True)


def test_matrix_matmul_vector_check_node_matching_raises_when_mismatched():
    M = _make_matrix("r", [0, 1], "c", [0, 1], [[1.0, 0.0], [0.0, 1.0]])
    # Vector key is "x", not "c" → col/row mismatch
    v = _make_vector("x", 0, 1, values=[1.0, 2.0])
    with pytest.raises(ValueError, match="Node matching failed"):
        M.matmul(v, _check_node_matching=True)
