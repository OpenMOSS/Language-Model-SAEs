"""Triton kernels for sparse matrix operations in SAE training."""

from typing import cast

import torch
import torch.sparse
import triton
import triton.language as tl

from lm_saes.utils.logging import get_logger
from lm_saes.utils.sparse import build_sparse_csr_from_topk, sort_topk_result

logger = get_logger("kernels")


def get_sparse_representation(x, pad_val=0):
    """
    Efficiently extracts sparse indices and values from a batched dense tensor x.

    Args:
        x (torch.Tensor): (B, d_sae) dense tensor with sparsity.
        pad_val (int, optional): Value to use for padding (default: 0).

    Returns:
        sparse_indices (torch.Tensor): (B, nnz_max) Tensor containing column indices of nonzero elements.
        sparse_values (torch.Tensor): (B, nnz_max) Tensor containing corresponding nonzero values.
        nnz_per_row (torch.Tensor): (B,) Number of nonzero elements per row.
    """
    B, d_sae = x.shape

    # Get nonzero indices and values
    nonzero_idx = (x != 0).nonzero(as_tuple=True)  # (batch_idx, col_idx)
    batch_indices, col_indices = nonzero_idx
    values = x[nonzero_idx]

    # Count nonzero elements per row
    nnz_per_row = torch.bincount(batch_indices, minlength=B)
    nnz_max = nnz_per_row.max().item()

    # Prepare output tensors (handle case where there are no nonzero elements)
    if nnz_max == 0:
        # No nonzero elements - return minimal zerotensors
        sparse_indices = torch.zeros((B, 1), dtype=torch.long, device=x.device)
        sparse_values = torch.zeros((B, 1), dtype=x.dtype, device=x.device)
    else:
        sparse_indices = torch.full((B, cast(int, nnz_max)), pad_val, dtype=torch.long, device=x.device)
        sparse_values = torch.zeros((B, cast(int, nnz_max)), dtype=x.dtype, device=x.device)

    # Only perform scattering if we have nonzero elements
    if nnz_max > 0 and len(batch_indices) > 0:
        # Compute position indices for scattering
        cum_nnzs = torch.cumsum(nnz_per_row, dim=0)
        row_offsets = torch.cat([torch.tensor([0], device=x.device), cum_nnzs[:-1]])  # Shift right for offsets
        positions = torch.arange(len(batch_indices), device=x.device) - row_offsets[batch_indices]

        # Scatter indices and values
        sparse_indices[batch_indices, positions] = col_indices
        sparse_values[batch_indices, positions] = values

    return sparse_indices, sparse_values


def triton_sparse_transpose_dense_matmul(
    sparse_indices: torch.Tensor,
    sparse_values: torch.Tensor,
    dense: torch.Tensor,
    N: int,
    BLOCK_SIZE_AK=128,
) -> torch.Tensor:
    """
    calculates sparse.T @ dense (i.e reducing along the collated dimension of sparse)
    dense must be contiguous along dim 0 (in other words, dense.T is contiguous)

    sparse_indices is shape (A, k)
    sparse_values is shape (A, k)
    dense is shape (A, B)

    output is shape (N, B)
    """

    assert sparse_indices.shape == sparse_values.shape
    assert sparse_indices.is_contiguous()
    assert sparse_values.is_contiguous()
    assert dense.is_contiguous()  # contiguous along B

    K = sparse_indices.shape[1]
    A = dense.shape[0]
    assert sparse_indices.shape[0] == A

    # Validate sparse indices are within bounds
    if sparse_indices.numel() > 0:
        max_idx = sparse_indices.max().item()
        if max_idx >= N:
            raise ValueError(f"Sparse indices out of bounds: max_idx={max_idx}, N={N}")

    # COO-format and sorted
    sorted_indices = sparse_indices.view(-1).sort()
    coo_indices = torch.stack(
        [
            torch.arange(A, device=sparse_indices.device).repeat_interleave(K)[sorted_indices.indices],
            sorted_indices.values,
        ]
    )  # shape (2, A * K)
    coo_values = sparse_values.view(-1)[sorted_indices.indices]  # shape (A * K,)
    return triton_coo_sparse_dense_matmul(coo_indices, coo_values, dense, N, BLOCK_SIZE_AK)


def triton_coo_sparse_dense_matmul(
    coo_indices: torch.Tensor,
    coo_values: torch.Tensor,
    dense: torch.Tensor,
    N: int,
    BLOCK_SIZE_AK=128,
) -> torch.Tensor:
    AK = coo_indices.shape[1]
    B = dense.shape[1]

    out = torch.zeros(N, B, device=dense.device, dtype=torch.float32)

    def grid(META):
        return triton.cdiv(AK, META["BLOCK_SIZE_AK"]), 1

    triton_sparse_transpose_dense_matmul_kernel[grid](  # type: ignore
        coo_indices,
        coo_values,
        dense,
        out,
        stride_da=dense.stride(0),
        stride_db=dense.stride(1),
        B=B,
        N=N,
        AK=AK,
        BLOCK_SIZE_AK=BLOCK_SIZE_AK,  # type: ignore
        BLOCK_SIZE_B=triton.next_power_of_2(B),  # type: ignore
    )
    return out.to(coo_values.dtype)


@triton.jit
def triton_sparse_transpose_dense_matmul_kernel(
    coo_indices_ptr,
    coo_values_ptr,
    dense_ptr,
    out_ptr,
    stride_da,
    stride_db,
    B,
    N,
    AK,
    BLOCK_SIZE_AK: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    """
    coo_indices is shape (2, AK)
    coo_values is shape (AK,)
    dense is shape (A, B), contiguous along B
    out is shape (N, B)
    """

    pid_ak = tl.program_id(0)
    pid_b = tl.program_id(1)

    coo_offsets = tl.arange(0, BLOCK_SIZE_AK)
    b_offsets = tl.arange(0, BLOCK_SIZE_B)

    A_coords = tl.load(
        coo_indices_ptr + pid_ak * BLOCK_SIZE_AK + coo_offsets,
        mask=pid_ak * BLOCK_SIZE_AK + coo_offsets < AK,
    )
    K_coords = tl.load(
        coo_indices_ptr + pid_ak * BLOCK_SIZE_AK + coo_offsets + AK,
        mask=pid_ak * BLOCK_SIZE_AK + coo_offsets < AK,
    )
    values = tl.load(
        coo_values_ptr + pid_ak * BLOCK_SIZE_AK + coo_offsets,
        mask=pid_ak * BLOCK_SIZE_AK + coo_offsets < AK,
    )

    last_k = tl.min(K_coords)
    accum = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)

    for ind in range(BLOCK_SIZE_AK):
        if ind + pid_ak * BLOCK_SIZE_AK < AK:
            # workaround to do A_coords[ind]
            a = tl.sum(
                tl.where(
                    tl.arange(0, BLOCK_SIZE_AK) == ind,
                    A_coords,
                    tl.zeros((BLOCK_SIZE_AK,), dtype=tl.int64),
                )
            )

            k = tl.sum(
                tl.where(
                    tl.arange(0, BLOCK_SIZE_AK) == ind,
                    K_coords,
                    tl.zeros((BLOCK_SIZE_AK,), dtype=tl.int64),
                )
            )

            v = tl.sum(
                tl.where(
                    tl.arange(0, BLOCK_SIZE_AK) == ind,
                    values,
                    tl.zeros((BLOCK_SIZE_AK,), dtype=tl.float32),
                )
            )

            tl.device_assert(k < N)

            if k != last_k:
                tl.atomic_add(
                    out_ptr + last_k * B + BLOCK_SIZE_B * pid_b + b_offsets,
                    accum,
                    mask=BLOCK_SIZE_B * pid_b + b_offsets < B,
                )
                accum *= 0
                last_k = k

            if v != 0:
                accum += v * tl.load(dense_ptr + a * stride_da + b_offsets, mask=b_offsets < B)

    tl.atomic_add(
        out_ptr + last_k * B + BLOCK_SIZE_B * pid_b + b_offsets,
        accum,
        mask=BLOCK_SIZE_B * pid_b + b_offsets < B,
    )


def triton_sparse_dense_matmul(
    sparse_indices: torch.Tensor,
    sparse_values: torch.Tensor,
    dense: torch.Tensor,
) -> torch.Tensor:
    """
    calculates sparse @ dense (i.e reducing along the uncollated dimension of sparse)
    dense must be contiguous along dim 0 (in other words, dense.T is contiguous)

    sparse_indices is shape (batch_size, k)
    sparse_values is shape (batch_size, k)
    dense is shape (d_sae, d_model)   ## TODO: check if this is correct, this is compatible with current situation

    output is shape (batch_size, d_model)
    """
    N = dense.shape[0]
    assert sparse_indices.shape == sparse_values.shape
    assert sparse_indices.is_contiguous()
    assert sparse_values.is_contiguous()
    assert dense.is_contiguous()  # contiguous along B

    A = sparse_indices.shape[0]
    K = sparse_indices.shape[1]
    B = dense.shape[1]

    out = torch.zeros(A, B, device=dense.device, dtype=sparse_values.dtype)

    triton_sparse_dense_matmul_kernel[(A,)](  # type: ignore
        sparse_indices,
        sparse_values,
        dense,
        out,
        stride_dn=dense.stride(0),
        stride_db=dense.stride(1),
        A=A,
        B=B,
        N=N,
        K=K,
        BLOCK_SIZE_K=triton.next_power_of_2(K),  # type: ignore
        BLOCK_SIZE_B=triton.next_power_of_2(B),  # type: ignore
    )
    return out


def triton_dense_dense_sparseout_matmul(
    dense1: torch.Tensor,
    dense2: torch.Tensor,
    at_indices: torch.Tensor,
) -> torch.Tensor:
    """
    dense1: shape (batch_size, d_model)
    dense2: shape (d_model, d_sae)
    at_indices: shape (batch_size, K)
    out values: shape (batch_size, K)
    calculates dense1 @ dense2 only for the indices in at_indices

    equivalent to (dense1 @ dense2).gather(1, at_indices)
    """
    A, B = dense1.shape
    N = dense2.shape[1]
    assert dense2.shape[0] == B
    assert at_indices.shape[0] == A
    K = at_indices.shape[1]
    assert at_indices.is_contiguous()

    assert dense1.stride(1) == 1, "dense1 must be contiguous along B"
    assert dense2.stride(0) == 1, "dense2 must be contiguous along B"

    if K > 2048:
        logger.warning(f"Using naive matmul for large K: {K}")
        # naive is more efficient for large K
        return (dense1 @ dense2).gather(1, at_indices)

    out = torch.zeros(A, K, device=dense1.device, dtype=dense1.dtype)

    # grid = lambda META: (triton.cdiv(A, META['BLOCK_SIZE_A']),)

    triton_dense_dense_sparseout_matmul_kernel[(A,)](  # type: ignore
        dense1,
        dense2,
        at_indices,
        out,
        stride_d1a=dense1.stride(0),
        stride_d1b=dense1.stride(1),
        stride_d2b=dense2.stride(0),
        stride_d2n=dense2.stride(1),
        A=A,
        B=B,
        N=N,
        K=K,
        BLOCK_SIZE_B=triton.next_power_of_2(B),  # type: ignore
        BLOCK_SIZE_N=triton.next_power_of_2(N),  # type: ignore
        BLOCK_SIZE_K=triton.next_power_of_2(K),  # type: ignore
    )

    return out


@triton.jit
def triton_dense_dense_sparseout_matmul_kernel(
    dense1_ptr,
    dense2_ptr,
    at_indices_ptr,
    out_ptr,
    stride_d1a,
    stride_d1b,
    stride_d2b,
    stride_d2n,
    A,
    B,
    N,
    K,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    dense1: shape (A, B)
    dense2: shape (B, N)
    at_indices: shape (A, K)
    out values: shape (A, K)
    """

    pid = tl.program_id(0)

    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    at_indices = tl.load(at_indices_ptr + pid * K + offsets_k, mask=offsets_k < K)  # shape (K,)

    offsets_b = tl.arange(0, BLOCK_SIZE_B)
    dense1 = tl.load(dense1_ptr + pid * stride_d1a + offsets_b * stride_d1b, mask=offsets_b < B)  # shape (B,)

    accum = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)

    for k in range(K):
        # workaround to do at_indices[b]
        i = tl.sum(
            tl.where(
                tl.arange(0, BLOCK_SIZE_K) == k,
                at_indices,
                tl.zeros((BLOCK_SIZE_K,), dtype=tl.int64),
            )
        )
        tl.device_assert(i < N)

        dense2col = tl.load(dense2_ptr + offsets_b * stride_d2b + i * stride_d2n, mask=offsets_b < B)  # shape (B,)
        accum += tl.where(
            tl.arange(0, BLOCK_SIZE_K) == k,
            tl.sum(dense1 * dense2col),
            tl.zeros((BLOCK_SIZE_K,), dtype=tl.int64),
        )

    tl.store(out_ptr + pid * K + offsets_k, accum, mask=offsets_k < K)


@triton.jit
def triton_sparse_dense_matmul_kernel(
    sparse_indices_ptr,
    sparse_values_ptr,
    dense_ptr,
    out_ptr,
    stride_dn,
    stride_db,
    A,
    B,
    N,
    K,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    """
    sparse_indices is shape (A, K)
    sparse_values is shape (A, K)
    dense is shape (N, B), contiguous along B
    out is shape (A, B)
    """

    pid = tl.program_id(0)

    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    sparse_indices = tl.load(sparse_indices_ptr + pid * K + offsets_k, mask=offsets_k < K)  # shape (K,)
    sparse_values = tl.load(sparse_values_ptr + pid * K + offsets_k, mask=offsets_k < K)  # shape (K,)

    accum = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)

    offsets_b = tl.arange(0, BLOCK_SIZE_B)

    for k in range(K):
        # workaround to do sparse_indices[k]
        i = tl.sum(
            tl.where(
                tl.arange(0, BLOCK_SIZE_K) == k,
                sparse_indices,
                tl.zeros((BLOCK_SIZE_K,), dtype=tl.int64),
            )
        )
        # workaround to do sparse_values[k]
        v = tl.sum(
            tl.where(
                tl.arange(0, BLOCK_SIZE_K) == k,
                sparse_values,
                tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32),
            )
        )

        tl.device_assert(i < N)
        if v != 0:
            accum += v * tl.load(dense_ptr + i * stride_dn + offsets_b * stride_db, mask=offsets_b < B)

    tl.store(out_ptr + pid * B + offsets_b, accum.to(sparse_values.dtype), mask=offsets_b < B)


class TritonEncoderAutogradDynamicK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W_E, b_E, sparsity_threshold):
        """
        Forward pass: x @ W_E + b_E using dense operations.

        Args:
            ctx: autograd context
            x: (batch, d_model) - Input activations
            W_E: (d_model, d_sae) - Encoder weights
            b_E: (d_sae,) - Encoder bias
            sparsity_threshold: (float) - Sparsity threshold for using sparse kernels.

        Returns:
            output: (batch, d_sae) - Encoded activations
        """
        # Save tensors for backward pass
        ctx.save_for_backward(x, W_E, b_E)
        # Save non-tensor values separately
        ctx.sparsity_threshold = sparsity_threshold

        # Forward pass using dense operations (no triton kernel needed)
        output = torch.mm(x, W_E) + b_E
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass: Use triton sparse kernels when grad_output is sparse (after topk).

        Args:
            grad_outputs: Tuple containing grad_output (batch, d_sae) - Gradient w.r.t. output (sparse after topk)

        Returns:
            grad_x: (batch, d_model) - Gradient w.r.t. input
            grad_W_E: (d_model, d_sae) - Gradient w.r.t. encoder weights
            grad_b_E: (d_sae,) - Gradient w.r.t. encoder bias
            None: For sparsity_threshold (no gradient needed)
        """
        assert len(grad_outputs) == 1, "Expected exactly one gradient output"
        grad_output = grad_outputs[0]

        x, W_E, b_E = ctx.saved_tensors
        sparsity_threshold = ctx.sparsity_threshold

        assert grad_output.is_contiguous(), "grad_output must be contiguous for triton kernels"

        # Check if grad_output is sparse (after topk activation)
        sparsity_ratio = (grad_output == 0).float().mean().item()
        use_sparse_kernels = (
            sparsity_ratio > sparsity_threshold
        )  # Use sparse kernels if sparsity_ratio > sparsity_threshold

        if use_sparse_kernels:
            # Use triton sparse kernels for backward pass
            sparse_indices, sparse_values = get_sparse_representation(grad_output)

            # Validate sparse indices are within bounds (skip if no active features)
            if sparse_indices.numel() > 0:
                max_sparse_idx = sparse_indices.max().item()
                if max_sparse_idx >= W_E.size(1):
                    raise ValueError(
                        f"Sparse indices out of bounds: max_idx={max_sparse_idx}, expected < {W_E.size(1)}"
                    )

            # grad_x = grad_output @ W_E.T (sparse @ dense)
            grad_x = triton_sparse_dense_matmul(sparse_indices, sparse_values, W_E.T.contiguous())

            # grad_W_E = x.T @ grad_output (dense.T @ sparse)
            # This is equivalent to sparse.T @ dense, so we can use transpose matmul
            grad_W_E = triton_sparse_transpose_dense_matmul(
                sparse_indices, sparse_values, x.contiguous(), N=W_E.size(1)
            ).T

            # grad_b_E = grad_output.sum(dim=0) for sparse tensor
            grad_b_E = torch.zeros_like(b_E)

            # sparse_indices: (batch, k) - feature indices for each batch sample
            # sparse_values: (batch, k) - corresponding values
            # We need to sum all values that correspond to the same feature index

            # Flatten to get all (batch_idx, feature_idx) pairs and their values
            batch_size, k = sparse_indices.shape

            # Get valid (non-padded) entries
            if sparse_indices.numel() > 0:
                valid_mask = sparse_indices < W_E.size(1)  # Valid feature indices
                valid_feature_indices = sparse_indices[valid_mask]
                valid_values = sparse_values[valid_mask]

                # Sum values by feature index to compute bias gradient (only if we have valid features)
                if valid_feature_indices.numel() > 0:
                    grad_b_E.index_add_(0, valid_feature_indices, valid_values)

        else:
            # Fall back to dense operations when not sparse enough
            grad_x = grad_output @ W_E.T
            grad_W_E = x.T @ grad_output
            grad_b_E = grad_output.sum(dim=0)

        return grad_x, grad_W_E, grad_b_E, None


class TritonDecoderAutogradTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_indices, sparse_values, decoder_weight):
        ctx.save_for_backward(sparse_indices, sparse_values, decoder_weight)
        return triton_sparse_dense_matmul(sparse_indices, sparse_values, decoder_weight.T)

    @staticmethod
    def backward(ctx, *grad_outputs, **args):
        assert len(grad_outputs) == 1, "grad_outputs must be a single tensor"
        grad_output = grad_outputs[0]
        sparse_indices, sparse_values, decoder_weight = ctx.saved_tensors

        assert grad_output.is_contiguous(), (
            "grad_output must be contiguous; this is probably because the subsequent op was a .sum() or something like that, which returns a non contiguous gradient"
        )

        decoder_grad = triton_sparse_transpose_dense_matmul(
            sparse_indices, sparse_values, grad_output, N=decoder_weight.size(1)
        ).T

        feature_acts_grad = triton_dense_dense_sparseout_matmul(grad_output, decoder_weight, sparse_indices)
        feature_acts_grad *= sparse_values.ne(0).to(feature_acts_grad.dtype)

        # decoder is contiguous when transposed so this is a matching layout
        return None, feature_acts_grad, decoder_grad


@triton.jit
def masked_matmul_kernel(
    # ptr
    output_ptr,  # [batch_size, k]
    a_ptr,  # [batch_size, d_model]
    b_ptr,  # [n_cols, d_model]  (e.g. W_D: [d_sae, d_model])
    indices_ptr,  # [batch_size, k]
    # shape
    batch_size,
    k,
    d_model,
    n_cols,
    # strides
    stride_a_batch,
    stride_a_d,
    stride_b_row,
    stride_b_d,
    stride_idx_batch,
    stride_idx_k,
    stride_out_batch,
    stride_out_k,
    # block size
    BLOCK_SIZE_D: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // k
    k_idx = pid % k

    if batch_id >= batch_size:
        return

    col_idx = tl.load(indices_ptr + batch_id * stride_idx_batch + k_idx * stride_idx_k)

    accumulator = 0.0

    for d_start in range(0, d_model, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask = d_offsets < d_model

        a_vals = tl.load(
            a_ptr + batch_id * stride_a_batch + d_offsets * stride_a_d,
            mask=mask,
            other=0.0,
        )

        b_vals = tl.load(
            b_ptr + col_idx * stride_b_row + d_offsets * stride_b_d,
            mask=mask,
            other=0.0,
        )

        accumulator += tl.sum(a_vals * b_vals)

    tl.store(output_ptr + batch_id * stride_out_batch + k_idx * stride_out_k, accumulator)


def masked_matmul(a: torch.Tensor, b: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda, f"a must be on CUDA, got {a.device}"
    assert b.is_cuda, f"b must be on CUDA, got {b.device}"
    assert indices.is_cuda, f"indices must be on CUDA, got {indices.device}"
    assert a.dim() == 2 and b.dim() == 2 and indices.dim() == 2
    assert a.shape[0] == indices.shape[0]
    assert a.shape[1] == b.shape[1]

    batch_size, d_model = a.shape
    n_cols = b.shape[0]
    k = indices.shape[1]

    a = a.contiguous()
    b = b.contiguous()
    indices = indices.contiguous().to(torch.int64)

    output = torch.empty((batch_size, k), device=a.device, dtype=a.dtype)

    BLOCK_SIZE_D = triton.next_power_of_2(min(d_model, 1024))

    grid = (batch_size * k,)

    masked_matmul_kernel[grid](
        output,
        a,
        b,
        indices,
        batch_size,
        k,
        d_model,
        n_cols,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        indices.stride(0),
        indices.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return output


@triton.jit
def masked_matmul_with_col_sum_kernel(
    # ptr
    output_ptr,  # [batch_size, k]
    col_sum_ptr,  # [n_cols]
    a_ptr,  # [batch_size, d_model]
    b_ptr,  # [n_cols, d_model]
    indices_ptr,  # [batch_size, k]
    # shape
    batch_size,
    k,
    d_model,
    n_cols,
    # strides
    stride_a_batch,
    stride_a_d,
    stride_b_row,
    stride_b_d,
    stride_idx_batch,
    stride_idx_k,
    stride_out_batch,
    stride_out_k,
    # block size
    BLOCK_SIZE_D: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // k
    k_idx = pid % k

    if batch_id >= batch_size:
        return

    col_idx = tl.load(indices_ptr + batch_id * stride_idx_batch + k_idx * stride_idx_k)

    accumulator = 0.0

    for d_start in range(0, d_model, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask = d_offsets < d_model

        a_vals = tl.load(
            a_ptr + batch_id * stride_a_batch + d_offsets * stride_a_d,
            mask=mask,
            other=0.0,
        )

        b_vals = tl.load(
            b_ptr + col_idx * stride_b_row + d_offsets * stride_b_d,
            mask=mask,
            other=0.0,
        )

        accumulator += tl.sum(a_vals * b_vals)

    # 写入 output
    tl.store(output_ptr + batch_id * stride_out_batch + k_idx * stride_out_k, accumulator)

    # 用 atomic add 累加到 col_sum[col_idx]
    tl.atomic_add(col_sum_ptr + col_idx, accumulator)


def masked_matmul_with_col_sum(
    a: torch.Tensor, b: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    assert a.is_cuda, f"a must be on CUDA, got {a.device}"
    assert b.is_cuda, f"b must be on CUDA, got {b.device}"
    assert indices.is_cuda, f"indices must be on CUDA, got {indices.device}"
    assert a.dim() == 2 and b.dim() == 2 and indices.dim() == 2
    assert a.shape[0] == indices.shape[0]
    assert a.shape[1] == b.shape[1]

    batch_size, d_model = a.shape
    n_cols = b.shape[0]
    k = indices.shape[1]

    a = a.contiguous()
    b = b.contiguous()
    indices = indices.contiguous().to(torch.int64)

    output = torch.empty((batch_size, k), device=a.device, dtype=a.dtype)
    col_sum = torch.zeros(n_cols, device=a.device, dtype=a.dtype)

    BLOCK_SIZE_D = triton.next_power_of_2(min(d_model, 1024))

    grid = (batch_size * k,)

    masked_matmul_with_col_sum_kernel[grid](
        output,
        col_sum,
        a,
        b,
        indices,
        batch_size,
        k,
        d_model,
        n_cols,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        indices.stride(0),
        indices.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return output, col_sum


class TopKSparseFusedSAE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        W_E: torch.Tensor,
        b_E: torch.Tensor,
        W_D: torch.Tensor,
        b_D: torch.Tensor,
        k: int,
        sparsity_include_decoder_norm: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_sae = W_E.shape[1]

        hidden_pre = x @ W_E + b_E

        if sparsity_include_decoder_norm:
            _, topk_indices = torch.topk(hidden_pre * W_D.norm(dim=-1, keepdim=True), k=k, dim=-1, sorted=False)
            topk_values = torch.gather(hidden_pre, dim=1, index=topk_indices)
        else:
            topk_values, topk_indices = torch.topk(hidden_pre, k=k, dim=-1, sorted=False)

        # build feature acts, only for log
        feature_acts = torch.zeros_like(hidden_pre)
        feature_acts.scatter_(dim=1, index=topk_indices, src=topk_values)

        # build sparse feature acts
        topk_indices_sorted, topk_values_sorted = sort_topk_result(topk_indices, topk_values)
        feature_acts_sparse = build_sparse_csr_from_topk(topk_indices_sorted, topk_values_sorted, d_sae)

        # decode
        reconstructed = torch.sparse.mm(feature_acts_sparse, W_D) + b_D

        ctx.save_for_backward(
            x,
            W_E,
            b_E,
            W_D,
            b_D,
            feature_acts_sparse,
            topk_indices_sorted,
            topk_values_sorted,
        )
        ctx.k = k

        return reconstructed, feature_acts, hidden_pre

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
        grad_feature_acts: torch.Tensor | None = None,
        grad_hidden_pre: torch.Tensor | None = None,
    ) -> tuple[None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        (
            x,
            W_E,
            b_E,
            W_D,
            b_D,
            feature_acts_sparse,
            topk_indices_sorted,
            topk_values_sorted,
        ) = ctx.saved_tensors
        k = ctx.k

        batch_size, d_model = x.shape
        d_sae = W_E.shape[1]

        grad_W_E = grad_b_E = grad_W_D = grad_b_D = None

        # grad_b_D
        grad_b_D = grad_output.sum(dim=0)

        # grad_W_D
        feature_acts_sparse_T = feature_acts_sparse.to_sparse_coo().T
        feature_acts_sparse_T = feature_acts_sparse_T.to_sparse_csr()
        grad_W_D = feature_acts_sparse_T @ grad_output

        grad_topk_values, grad_b_E = masked_matmul_with_col_sum(grad_output, W_D, topk_indices_sorted)

        crow_indices = torch.arange(
            0,
            (batch_size + 1) * k,
            k,
            device=grad_topk_values.device,
            dtype=torch.int32,
        )
        col_indices = topk_indices_sorted.reshape(-1).to(torch.int32)
        grad_hidden_pre = torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            grad_topk_values.reshape(-1),
            size=(batch_size, d_sae),
        )

        # grad_W_E
        grad_hidden_pre_T = grad_hidden_pre.to_sparse_coo().T
        grad_hidden_pre_T = grad_hidden_pre_T.to_sparse_csr()
        grad_W_E = torch.sparse.mm(grad_hidden_pre_T, x.to(grad_hidden_pre_T.dtype)).T

        return None, grad_W_E, grad_b_E, grad_W_D, grad_b_D, None, None
