import torch
import triton
import triton.language as tl
from jaxtyping import Float



@triton.jit
def spmm_kernel(
    W_ptr, x_vals_ptr, x_idx_ptr, x_nnz_ptr, y_ptr, 
    M, N, B, nnz_max, 
    stride_wm, stride_wn, stride_xb, stride_yb, stride_y,
    BLOCK_SIZE_M: tl.constexpr
):
    row_idx = tl.program_id(0)  # Each program processes one row of W
    batch_idx = tl.program_id(1)  # Each program processes one batch element

    if row_idx >= M or batch_idx >= B:
        return

    # Load nnz for this batch element
    nnz = tl.load(x_nnz_ptr + batch_idx)

    acc = tl.zeros((1,), dtype=tl.float32)  # Accumulator for dot product

    # Iterate over sparse indices
    for i in range(nnz):
        col_idx = tl.load(x_idx_ptr + batch_idx * stride_xb + i)  # Load sparse index
        x_val = tl.load(x_vals_ptr + batch_idx * stride_xb + i)  # Load sparse value
        w_val = tl.load(W_ptr + row_idx * stride_wm + col_idx)  # Load corresponding W entry
        acc += w_val * x_val  # Accumulate result

    # Store result in output matrix y[B, M]
    tl.store(y_ptr + batch_idx * stride_yb + row_idx * stride_y, acc)

def sparse_dense_batched_mul(W, x_vals, x_idx, x_nnz, M, N, B):
    """
    W: (M, N) Dense Matrix
    x_vals: (B, nnz_max) Non-zero values of sparse vectors
    x_idx: (B, nnz_max) Column indices of non-zero values
    x_nnz: (B,) Number of non-zero elements per batch element
    """
    assert W.shape == (M, N)
    
    nnz_max = x_vals.shape[1]  # Max number of non-zero elements per vector
    y = torch.zeros((B, M), device=W.device, dtype=W.dtype)  # Output shape (B, M)

    grid = (M, B)  # Parallelize over rows and batch elements
    spmm_kernel[grid](
        W, x_vals, x_idx, x_nnz, y, 
        M, N, B, nnz_max, 
        W.stride(0), W.stride(1), x_vals.stride(0), y.stride(0), y.stride(1),
        BLOCK_SIZE_M=1
    )
    return y


def get_sparse_representation(x, pad_val=0):
    """
    Extracts sparse indices and values from a batched dense tensor x.
    
    Args:
        x (torch.Tensor): (B, N) dense tensor with sparsity.
        pad_val (int, optional): Value to use for padding (default: 0).
        
    Returns:
        sparse_indices (torch.Tensor): (B, nnz_max) Tensor containing column indices of nonzero elements.
        sparse_values (torch.Tensor): (B, nnz_max) Tensor containing corresponding nonzero values.
        nnz_per_row (torch.Tensor): (B,) Number of nonzero elements per row.
    """
    B, N = x.shape
    
    # Get nonzero indices
    nonzero_idx = (x != 0).nonzero(as_tuple=True)  # Tuple of (batch_idx, col_idx)
    batch_indices, col_indices = nonzero_idx
    values = x[nonzero_idx]  # Extract corresponding values

    # Compute number of nonzero entries per row
    nnz_per_row = torch.bincount(batch_indices, minlength=B)
    nnz_max = nnz_per_row.max().item()  # Find the max nnz across batch

    # Create padded tensors
    sparse_indices = torch.full((B, nnz_max), pad_val, dtype=torch.long, device=x.device)
    sparse_values = torch.full((B, nnz_max), 0.0, dtype=x.dtype, device=x.device)

    # Fill sparse representation efficiently
    for b in range(B):
        num_values = nnz_per_row[b]
        if num_values > 0:
            sparse_indices[b, :num_values] = col_indices[batch_indices == b]
            sparse_values[b, :num_values] = values[batch_indices == b]

    return sparse_indices, sparse_values, nnz_per_row



class TritonDecoderAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_indices, sparse_values, decoder_weight):
        """
        sparse_indices: (B, nnz_max) - Indices of non-zero elements per batch.
        sparse_values: (B, nnz_max) - Non-zero values per batch.
        decoder_weight: (M, N) - Dense weight matrix.
        
        Returns:
            output: (B, M) - Result of sparse-dense multiplication.
        """
        ctx.save_for_backward(sparse_indices, sparse_values, decoder_weight)
        return sparse_dense_batched_mul(decoder_weight.T, sparse_values, sparse_indices, 
                                        torch.tensor([sparse_values.shape[1]], device=sparse_values.device), 
                                        decoder_weight.shape[1], decoder_weight.shape[0], sparse_indices.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes gradients w.r.t. inputs during backpropagation.
        """
        sparse_indices, sparse_values, decoder_weight = ctx.saved_tensors

        assert grad_output.is_contiguous(), "grad_output must be contiguous; this is probably because the subsequent op was a .sum() or something like that, which returns a non-contiguous gradient"

        # Compute gradient w.r.t. decoder_weight
        decoder_grad = triton_sparse_transpose_dense_matmul(
            sparse_indices, sparse_values, grad_output, N=decoder_weight.shape[1]
        ).T  # Transpose to match expected layout

        # Compute gradient w.r.t. sparse_values
        sparse_values_grad = triton_dense_dense_sparseout_matmul(
            grad_output, decoder_weight, sparse_indices
        )

        return None, sparse_values_grad, decoder_grad


def decode_with_triton(feature_acts: Float[torch.Tensor, "batch d_sae"]):
    sparse_indices, sparse_values, nnz_per_row = get_sparse_representation(feature_acts)
    return 
    
