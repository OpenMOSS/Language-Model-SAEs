import pytest
import torch

from lm_saes.backend.language_model import Matrix


def _build_sample_matrix() -> Matrix:
    matrix = Matrix(torch.device("cpu"))
    matrix.add_source([("s1", torch.tensor([0, 2])), ("s2", torch.tensor([1]))])
    matrix.add_target([("t1", torch.tensor([10, 11])), ("t2", torch.tensor([12]))])
    matrix.update_matrix(
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.float32,
        )
    )
    return matrix


def test_add_source_and_target_shapes_and_node_mappings():
    matrix = Matrix(torch.device("cpu"))
    matrix.add_source(("src", torch.tensor([3, 7])))
    matrix.add_target(("tgt", torch.tensor([11, 13, 17])))

    assert matrix.matrix.shape == (2, 3)
    assert torch.equal(matrix.source["src"].indices, torch.tensor([3, 7]))
    assert torch.equal(matrix.source["src"].matrix_indices, torch.tensor([0, 1]))
    assert torch.equal(matrix.target["tgt"].indices, torch.tensor([11, 13, 17]))
    assert torch.equal(matrix.target["tgt"].matrix_indices, torch.tensor([0, 1, 2]))


def test_add_source_merges_same_key():
    matrix = Matrix(torch.device("cpu"))
    matrix.add_source(("src", torch.tensor([0, 2])))
    matrix.add_source(("src", torch.tensor([5])))

    assert matrix.matrix.shape == (3, 0)
    assert torch.equal(matrix.source["src"].indices, torch.tensor([0, 2, 5]))
    assert torch.equal(matrix.source["src"].matrix_indices, torch.tensor([0, 1, 2]))


def test_update_matrix_overwrites_values():
    matrix = Matrix(torch.device("cpu"))
    matrix.add_source(("src", torch.tensor([0, 1])))
    matrix.add_target(("tgt", torch.tensor([0, 1])))

    new_values = torch.tensor([[1.5, -2.0], [3.25, 4.0]], dtype=torch.float32)
    matrix.update_matrix(new_values)

    assert torch.equal(matrix.matrix, new_values)


def test_get_submatrix_selects_expected_rows_and_columns():
    matrix = _build_sample_matrix()

    submatrix = matrix.get_submatrix(
        source_node_infos=[("s1", torch.tensor([2]))],
        target_node_infos=[("t1", torch.tensor([11])), ("t2", torch.tensor([12]))],
    )

    expected = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
    assert torch.equal(submatrix.matrix, expected)
    assert torch.equal(submatrix.source["s1"].indices, torch.tensor([2]))
    assert torch.equal(submatrix.source["s1"].matrix_indices, torch.tensor([0]))
    assert torch.equal(submatrix.target["t1"].indices, torch.tensor([11]))
    assert torch.equal(submatrix.target["t1"].matrix_indices, torch.tensor([0]))
    assert torch.equal(submatrix.target["t2"].indices, torch.tensor([12]))
    assert torch.equal(submatrix.target["t2"].matrix_indices, torch.tensor([1]))


def test_add_multiple_nodes_assigns_contiguous_matrix_indices():
    matrix = Matrix(torch.device("cpu"))
    matrix.add_source([("s1", torch.tensor([0, 2])), ("s2", torch.tensor([1]))])
    matrix.add_target([("t1", torch.tensor([10, 11])), ("t2", torch.tensor([12]))])

    assert torch.equal(matrix.source["s1"].matrix_indices, torch.tensor([0, 1]))
    assert torch.equal(matrix.source["s2"].matrix_indices, torch.tensor([2]))
    assert torch.equal(matrix.target["t1"].matrix_indices, torch.tensor([0, 1]))
    assert torch.equal(matrix.target["t2"].matrix_indices, torch.tensor([2]))


def test_top_k_returns_grouped_source_node_infos():
    matrix = _build_sample_matrix()

    topk_values, node_infos = matrix.top_k(k=2, dim=0, reduction_weights=torch.tensor([1.0, 1.0, 1.0]))

    assert torch.equal(topk_values, torch.tensor([24.0, 15.0]))
    assert len(node_infos) == 2
    assert node_infos[0][0] == "s2"
    assert torch.equal(node_infos[0][1], torch.tensor([1]))
    assert node_infos[1][0] == "s1"
    assert torch.equal(node_infos[1][1], torch.tensor([2]))


def test_top_k_returns_grouped_target_node_infos():
    matrix = _build_sample_matrix()

    topk_values, node_infos = matrix.top_k(k=2, dim=1, reduction_weights=torch.tensor([1.0, 1.0, 1.0]))

    assert torch.equal(topk_values, torch.tensor([18.0, 15.0]))
    assert len(node_infos) == 2
    assert node_infos[0][0] == "t2"
    assert torch.equal(node_infos[0][1], torch.tensor([12]))
    assert node_infos[1][0] == "t1"
    assert torch.equal(node_infos[1][1], torch.tensor([11]))


def test_top_k_raises_for_invalid_dim():
    matrix = _build_sample_matrix()

    with pytest.raises(ValueError, match="Invalid dimension: 2"):
        matrix.top_k(k=1, dim=2, reduction_weights=torch.tensor([1.0, 1.0, 1.0]))


def test_end_to_end_mixed_add_submatrix_and_topk():
    matrix = Matrix(torch.device("cpu"))
    matrix.add_source([("s1", torch.tensor([0, 2])), ("s2", torch.tensor([1]))])
    matrix.add_source(("s1", torch.tensor([4])))
    matrix.add_target([("t1", torch.tensor([10, 11])), ("t2", torch.tensor([12]))])
    matrix.add_target(("t1", torch.tensor([13])))
    matrix.update_matrix(
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            dtype=torch.float32,
        )
    )

    submatrix = matrix.get_submatrix(
        source_node_infos=[("s1", torch.tensor([2, 4])), ("s2", torch.tensor([1]))],
        target_node_infos=[("t1", torch.tensor([11, 13])), ("t2", torch.tensor([12]))],
    )
    assert torch.equal(
        submatrix.matrix,
        torch.tensor(
            [
                [6.0, 8.0, 7.0],
                [14.0, 16.0, 15.0],
                [10.0, 12.0, 11.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(submatrix.source["s1"].indices, torch.tensor([2, 4]))
    assert torch.equal(submatrix.source["s1"].matrix_indices, torch.tensor([0, 1]))
    assert torch.equal(submatrix.source["s2"].indices, torch.tensor([1]))
    assert torch.equal(submatrix.source["s2"].matrix_indices, torch.tensor([2]))
    assert torch.equal(submatrix.target["t1"].indices, torch.tensor([11, 13]))
    assert torch.equal(submatrix.target["t1"].matrix_indices, torch.tensor([0, 1]))
    assert torch.equal(submatrix.target["t2"].indices, torch.tensor([12]))
    assert torch.equal(submatrix.target["t2"].matrix_indices, torch.tensor([2]))

    source_topk_values, source_node_infos = submatrix.top_k(
        k=2,
        dim=0,
        reduction_weights=torch.tensor([1.0, 1.0, 1.0]),
    )
    assert torch.equal(source_topk_values, torch.tensor([45.0, 33.0]))
    assert source_node_infos[0][0] == "s1"
    assert torch.equal(source_node_infos[0][1], torch.tensor([4]))
    assert source_node_infos[1][0] == "s2"
    assert torch.equal(source_node_infos[1][1], torch.tensor([1]))

    target_topk_values, target_node_infos = submatrix.top_k(
        k=2,
        dim=1,
        reduction_weights=torch.tensor([1.0, 1.0, 1.0]),
    )
    assert torch.equal(target_topk_values, torch.tensor([36.0, 33.0]))
    assert len(target_node_infos) == 2
    assert target_node_infos[0][0] == "t1"
    assert torch.equal(target_node_infos[0][1], torch.tensor([13]))
    assert target_node_infos[1][0] == "t2"
    assert torch.equal(target_node_infos[1][1], torch.tensor([12]))


def test_submatrix_chain_keeps_mapping_consistent_for_topk():
    matrix = Matrix(torch.device("cpu"))
    matrix.add_source([("s1", torch.tensor([0, 2])), ("s2", torch.tensor([1]))])
    matrix.add_target([("t1", torch.tensor([10, 11])), ("t2", torch.tensor([12]))])
    matrix.update_matrix(
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.float32,
        )
    )

    first_submatrix = matrix.get_submatrix(
        source_node_infos=[("s1", torch.tensor([0, 2]))],
        target_node_infos=[("t1", torch.tensor([10, 11]))],
    )
    second_submatrix = first_submatrix.get_submatrix(
        source_node_infos=[("s1", torch.tensor([2]))],
        target_node_infos=[("t1", torch.tensor([11]))],
    )

    assert torch.equal(second_submatrix.matrix, torch.tensor([[5.0]], dtype=torch.float32))

    topk_values, node_infos = second_submatrix.top_k(
        k=1,
        dim=0,
        reduction_weights=torch.tensor([1.0]),
    )
    assert torch.equal(topk_values, torch.tensor([5.0]))
    assert len(node_infos) == 1
    assert node_infos[0][0] == "s1"
    assert torch.equal(node_infos[0][1], torch.tensor([2]))
