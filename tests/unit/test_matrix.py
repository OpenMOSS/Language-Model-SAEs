import torch

from lm_saes.backend.language_model import Matrix


def _build_sample_matrix() -> Matrix:
    matrix = Matrix()
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
    matrix = Matrix()
    matrix.add_source(("src", torch.tensor([3, 7])))
    matrix.add_target(("tgt", torch.tensor([11, 13, 17])))

    assert matrix.matrix.shape == (2, 3)
    assert torch.equal(matrix.source["src"].indices, torch.tensor([3, 7]))
    assert torch.equal(matrix.source["src"].matrix_indices, torch.tensor([0, 1]))
    assert torch.equal(matrix.target["tgt"].indices, torch.tensor([11, 13, 17]))
    assert torch.equal(matrix.target["tgt"].matrix_indices, torch.tensor([0, 1, 2]))


def test_add_source_merges_same_key():
    matrix = Matrix()
    matrix.add_source(("src", torch.tensor([0, 2])))
    matrix.add_source(("src", torch.tensor([5])))

    assert matrix.matrix.shape == (3, 0)
    assert torch.equal(matrix.source["src"].indices, torch.tensor([0, 2, 5]))
    assert torch.equal(matrix.source["src"].matrix_indices, torch.tensor([0, 1, 2]))


def test_update_matrix_overwrites_values():
    matrix = Matrix()
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
    matrix = Matrix()
    matrix.add_source([("s1", torch.tensor([0, 2])), ("s2", torch.tensor([1]))])
    matrix.add_target([("t1", torch.tensor([10, 11])), ("t2", torch.tensor([12]))])

    assert torch.equal(matrix.source["s1"].matrix_indices, torch.tensor([0, 1]))
    assert torch.equal(matrix.source["s2"].matrix_indices, torch.tensor([2]))
    assert torch.equal(matrix.target["t1"].matrix_indices, torch.tensor([0, 1]))
    assert torch.equal(matrix.target["t2"].matrix_indices, torch.tensor([2]))
