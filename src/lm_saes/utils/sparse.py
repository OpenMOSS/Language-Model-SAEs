import torch


def sort_topk_result(topk_indices: torch.Tensor, topk_values: torch.Tensor):
    topk_indices_sorted, sort_perm = torch.sort(topk_indices, dim=-1)
    topk_values_sorted = torch.gather(topk_values, dim=-1, index=sort_perm)
    return topk_indices_sorted, topk_values_sorted


def build_sparse_csr_from_topk(topk_indices_sorted: torch.Tensor, topk_values_sorted: torch.Tensor, n_cols: int):
    n_rows: int = topk_indices_sorted.shape[0]
    nnz_per_row: int = topk_indices_sorted.shape[1]

    crow_indices: torch.Tensor = torch.arange(
        0,
        (n_rows + 1) * nnz_per_row,
        nnz_per_row,
        device=topk_indices_sorted.device,
        dtype=topk_indices_sorted.dtype,
    )
    col_indices: torch.Tensor = topk_indices_sorted.reshape(-1)
    values: torch.Tensor = topk_values_sorted.reshape(-1)

    sparse_csr_tensor: torch.Tensor = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(n_rows, n_cols))

    return sparse_csr_tensor


def build_sparse_coo_from_topk(topk_indices_sorted: torch.Tensor, topk_values_sorted: torch.Tensor, n_cols: int):
    n_rows: int = topk_indices_sorted.shape[0]
    nnz_per_row: int = topk_indices_sorted.shape[1]

    row_indices: torch.Tensor = torch.arange(
        0, n_rows, device=topk_indices_sorted.device, dtype=topk_indices_sorted.dtype
    ).repeat_interleave(nnz_per_row)
    col_indices: torch.Tensor = topk_indices_sorted.reshape(-1)
    values: torch.Tensor = topk_values_sorted.reshape(-1)
    sparse_coo_tensor: torch.Tensor = torch.sparse_coo_tensor(
        torch.stack([row_indices, col_indices], dim=0),
        values,
        size=(n_rows, n_cols),
    )
    return sparse_coo_tensor
