"""Triton kernels for sparse matrix operations in SAE training."""

import math
from typing import cast

import torch
import torch.sparse
import triton
import triton.language as tl
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from lm_saes.utils.distributed import DimMap
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
    @torch.amp.custom_fwd(device_type="cuda")
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
            _, topk_indices = torch.topk(hidden_pre * W_D.norm(dim=-1), k=k, dim=-1, sorted=False)
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
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx,
        grad_output: torch.Tensor,
        grad_feature_acts_none: torch.Tensor | None = None,
        grad_hidden_pre_none: torch.Tensor | None = None,
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

        # import os

        # if os.environ.get("LOCAL_RANK") == "0":
        #     print("grad_b_D:", grad_b_D)
        #     print("grad_W_D:", grad_W_D)
        #     print("grad_W_E:", grad_W_E)
        #     print("grad_b_E:", grad_b_E)
        # import time

        # time.sleep(5)
        # if os.environ.get("LOCAL_RANK") == "1":
        #     print("grad_b_D:", grad_b_D)
        #     print("grad_W_D:", grad_W_D)
        #     print("grad_W_E:", grad_W_E)
        #     print("grad_b_E:", grad_b_E)
        # time.sleep(5)
        # exit()

        return None, grad_W_E, grad_b_E, grad_W_D, grad_b_D, None, None


class DPTopKSparseFusedSAE(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: torch.Tensor,
        W_E: torch.Tensor,
        b_E: torch.Tensor,
        W_D: torch.Tensor,
        b_D: torch.Tensor,
        k: int,
        sparsity_include_decoder_norm: bool,
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_local, W_E_local, b_E_local, W_D_local, b_D_local = (
            x.to_local(),
            W_E.to_local(),
            b_E.to_local(),
            W_D.to_local(),
            b_D.to_local(),
        )

        d_sae = W_E_local.shape[1]

        hidden_pre_local = x_local @ W_E_local + b_E_local

        if sparsity_include_decoder_norm:
            _, topk_indices_local = torch.topk(hidden_pre_local * W_D_local.norm(dim=-1), k=k, dim=-1, sorted=False)
            topk_values_local = torch.gather(hidden_pre_local, dim=1, index=topk_indices_local)
        else:
            topk_values_local, topk_indices_local = torch.topk(hidden_pre_local, k=k, dim=-1, sorted=False)

        # build feature acts, only for log
        feature_acts_local = torch.zeros_like(hidden_pre_local)
        feature_acts_local.scatter_(dim=1, index=topk_indices_local, src=topk_values_local)

        # build sparse feature acts
        topk_indices_sorted_local, topk_values_sorted_local = sort_topk_result(topk_indices_local, topk_values_local)
        feature_acts_sparse_local = build_sparse_csr_from_topk(
            topk_indices_sorted_local, topk_values_sorted_local, d_sae
        )

        # decode
        reconstructed_local = torch.sparse.mm(feature_acts_sparse_local, W_D_local) + b_D_local

        reconstructed = DTensor.from_local(reconstructed_local, device_mesh=device_mesh, placements=x.placements)
        feature_acts = DTensor.from_local(feature_acts_local, device_mesh=device_mesh, placements=x.placements)
        hidden_pre = DTensor.from_local(hidden_pre_local, device_mesh=device_mesh, placements=x.placements)

        ctx.save_for_backward(
            x,
            W_E,
            b_E,
            W_D,
            b_D,
            feature_acts_sparse_local,
            topk_indices_sorted_local,
            topk_values_sorted_local,
        )
        ctx.k = k
        ctx.device_mesh = device_mesh

        return reconstructed, feature_acts, hidden_pre

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx,
        grad_output: torch.Tensor,
        grad_feature_acts_none: torch.Tensor | None = None,
        grad_hidden_pre_none: torch.Tensor | None = None,
    ) -> tuple[None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        (
            x,
            W_E,
            b_E,
            W_D,
            b_D,
            feature_acts_sparse_local,
            topk_indices_sorted_local,
            topk_values_sorted_local,
        ) = ctx.saved_tensors
        k = ctx.k
        device_mesh = ctx.device_mesh
        x_local, W_E_local, W_D_local, grad_output_local = (
            x.to_local(),
            W_E.to_local(),
            W_D.to_local(),
            grad_output.to_local(),
        )

        batch_size_local, d_model = x_local.shape
        d_sae = W_E_local.shape[1]

        grad_W_E_local = grad_b_E_local = grad_W_D_local = grad_b_D_local = None

        # grad_b_D
        grad_b_D_local = grad_output_local.sum(dim=0)

        # grad_W_D
        feature_acts_sparse_T_local = feature_acts_sparse_local.to_sparse_coo().T
        feature_acts_sparse_T_local = feature_acts_sparse_T_local.to_sparse_csr()
        grad_W_D_local = feature_acts_sparse_T_local @ grad_output_local

        grad_topk_values_local, grad_b_E_local = masked_matmul_with_col_sum(
            grad_output_local, W_D_local, topk_indices_sorted_local
        )

        crow_indices = torch.arange(
            0,
            (batch_size_local + 1) * k,
            k,
            device=grad_topk_values_local.device,
            dtype=torch.int32,
        )
        col_indices = topk_indices_sorted_local.reshape(-1).to(torch.int32)
        grad_hidden_pre_local = torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            grad_topk_values_local.reshape(-1),
            size=(batch_size_local, d_sae),
        )

        # grad_W_E
        grad_hidden_pre_T_local = grad_hidden_pre_local.to_sparse_coo().T
        grad_hidden_pre_T_local = grad_hidden_pre_T_local.to_sparse_csr()
        grad_W_E_local = torch.sparse.mm(grad_hidden_pre_T_local, x_local.to(grad_hidden_pre_T_local.dtype)).T

        grad_b_D = DTensor.from_local(
            grad_b_D_local.unsqueeze(-1),
            device_mesh=device_mesh,
            placements=DimMap({"data": 1}).placements(device_mesh),
        )
        grad_b_D = grad_b_D.sum(dim=1)

        grad_W_D = DTensor.from_local(
            grad_W_D_local.unsqueeze(-1),
            device_mesh=device_mesh,
            placements=DimMap({"data": 2, "model": 0}).placements(device_mesh),
        )
        grad_W_D = grad_W_D.sum(dim=2)

        grad_W_E = DTensor.from_local(
            grad_W_E_local.unsqueeze(-1),
            device_mesh=device_mesh,
            placements=DimMap({"data": 2, "model": 1}).placements(device_mesh),
        )
        grad_W_E = grad_W_E.sum(dim=2)

        grad_b_E = DTensor.from_local(
            grad_b_E_local.unsqueeze(-1),
            device_mesh=device_mesh,
            placements=DimMap({"data": 1, "model": 0}).placements(device_mesh),
        )
        grad_b_E = grad_b_E.sum(dim=1)

        # import os

        # if os.environ.get("LOCAL_RANK") == "0":
        #     print("grad_b_D:", grad_b_D)
        #     print("grad_W_D:", grad_W_D)
        #     print("grad_W_E:", grad_W_E)
        #     print("grad_b_E:", grad_b_E)
        # import time

        # time.sleep(5)
        # if os.environ.get("LOCAL_RANK") == "1":
        #     print("grad_b_D:", grad_b_D)
        #     print("grad_W_D:", grad_W_D)
        #     print("grad_W_E:", grad_W_E)
        #     print("grad_b_E:", grad_b_E)
        # time.sleep(5)
        # exit()

        return None, grad_W_E, grad_b_E, grad_W_D, grad_b_D, None, None, None


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        "EVEN_HEADDIM_V": lambda args: args["headdim_v"] == args["BLOCK_HEADDIM_V"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    headdim_v,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_HEADDIM_V: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    EVEN_HEADDIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_d_v = tl.arange(0, BLOCK_HEADDIM_V)
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d_v[None, :])
    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM_V], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=offs_d[None, :] < headdim,
                    other=0.0,
                )
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k), input_precision="tf32x3")
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # -- update output accumulator --
        acc_o = acc_o * acc_o_scale[:, None]
        # load v with headdim_v
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM_V:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=offs_d_v[None, :] < headdim_v,
                    other=0.0,
                )
        else:
            if EVEN_HEADDIM_V:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d_v[None, :] < headdim_v),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v, input_precision="tf32x3")

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output (using headdim_v)
    offs_d_v = tl.arange(0, BLOCK_HEADDIM_V)
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d_v[None, :])
    if EVEN_M:
        if EVEN_HEADDIM_V:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d_v[None, :] < headdim_v)
    else:
        if EVEN_HEADDIM_V:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs,
                acc_o,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d_v[None, :] < headdim_v),
            )


@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim_v,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM_V: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d_v = tl.arange(0, BLOCK_HEADDIM_V)
    # load (o and do have headdim_v dimension)
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d_v[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d_v[None, :] < headdim_v),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d_v[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d_v[None, :] < headdim_v),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    offs_d_v,
    seqlen_k,
    headdim,
    headdim_v,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    EVEN_HEADDIM_V: tl.constexpr,
):
    # Store dk (with headdim) and dv (with headdim_v) separately
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
        if EVEN_HEADDIM_V:
            tl.store(dv_ptrs, dv)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d_v[None, :] < headdim_v)
    else:
        if EVEN_HEADDIM:
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(
                dk_ptrs,
                dk,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
            )
        if EVEN_HEADDIM_V:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(
                dv_ptrs,
                dv,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d_v[None, :] < headdim_v),
            )


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    seqlen_q,
    seqlen_k,
    headdim,
    headdim_v,
    ATOMIC_ADD: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_HEADDIM_V: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    EVEN_HEADDIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_d_v = tl.arange(0, BLOCK_HEADDIM_V)
    # initialize pointers to value-like data
    # q, k, dq, dk use headdim; v, do, dv use headdim_v
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d_v[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d_v[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    # initialize dv (headdim_v) and dk (headdim)
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d_v[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            offs_d,
            offs_d_v,
            seqlen_k,
            headdim,
            headdim_v,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            EVEN_HEADDIM_V=EVEN_HEADDIM_V,
        )
        return
    # k (headdim) and v (headdim_v) stay in SRAM throughout
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        if EVEN_HEADDIM_V:
            v = tl.load(v_ptrs)
        else:
            v = tl.load(v_ptrs, mask=offs_d_v[None, :] < headdim_v, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
            )
        if EVEN_HEADDIM_V:
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            v = tl.load(
                v_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d_v[None, :] < headdim_v),
                other=0.0,
            )
    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        # load q (headdim)
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, tl.trans(k), input_precision="tf32x3")
        if not EVEN_N:
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        p = tl.exp(qk - lse_i[:, None])
        # load do (headdim_v)
        if EVEN_M & EVEN_HEADDIM_V:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d_v[None, :] < headdim_v),
                other=0.0,
            )
        # compute dv = p^T @ do (headdim_v)
        dv += tl.dot(tl.trans(p).to(do.dtype), do, input_precision="tf32x3")
        # compute dp = do @ v^T (using headdim_v)
        if not (EVEN_M & EVEN_HEADDIM_V):
            tl.debug_barrier()
        dp = tl.dot(do, tl.trans(v), input_precision="tf32x3")
        if not EVEN_HEADDIM_V:
            tl.debug_barrier()
        # compute ds = p * (dp - delta[:, None])
        Di = tl.load(D + offs_m_curr)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        # compute dk = ds^T @ q (headdim)
        dk += tl.dot(tl.trans(ds), q, input_precision="tf32x3")
        # compute dq = ds @ k (headdim)
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds, k, input_precision="tf32x3")
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k, input_precision="tf32x3")
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k, input_precision="tf32x3")
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
        else:
            dq = tl.dot(ds, k, input_precision="tf32x3")
            if EVEN_M & EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq)
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    )
        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d_v[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        offs_d_v,
        seqlen_k,
        headdim,
        headdim_v,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
        EVEN_HEADDIM_V=EVEN_HEADDIM_V,
    )


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        # Other configs seem to give wrong results when seqlen_q % 128 != 0, disabling them for now
        # # Kernel is buggy (give wrong result) if we set BLOCK_m=128, BLOCK_n=64, num_warps=*4*
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
    ],
    key=[
        "CACHE_KEY_SEQLEN_Q",
        "CACHE_KEY_SEQLEN_K",
        "IS_CAUSAL",
        "BLOCK_HEADDIM",
    ],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        "EVEN_HEADDIM_V": lambda args: args["headdim_v"] == args["BLOCK_HEADDIM_V"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    headdim_v,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_HEADDIM_V: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    EVEN_HEADDIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                headdim_v,
                ATOMIC_ADD=False,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                BLOCK_HEADDIM_V=BLOCK_HEADDIM_V,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                EVEN_HEADDIM_V=EVEN_HEADDIM_V,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            DO,
            DQ,
            DK,
            DV,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            seqlen_q,
            seqlen_k,
            headdim,
            headdim_v,
            ATOMIC_ADD=True,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            BLOCK_HEADDIM_V=BLOCK_HEADDIM_V,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            EVEN_HEADDIM_V=EVEN_HEADDIM_V,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )


def flash_attn_forward(q, k, v, causal=False, softmax_scale=None):
    # shape constraints
    # q, k: (batch, seqlen, nheads, d)
    # v: (batch, seqlen_k, nheads, d_v) where d_v can be different from d
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, d_v = v.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape[:3] == (batch, seqlen_k, nheads)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    # output has shape (batch, seqlen_q, nheads, d_v)
    o = torch.empty((batch, seqlen_q, nheads, d_v), device=q.device, dtype=q.dtype)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK_HEADDIM_V = max(triton.next_power_of_2(d_v), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8

    def grid(META):
        return (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)

    _fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        d_v,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        causal,
        BLOCK_HEADDIM,
        BLOCK_HEADDIM_V,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o, lse, softmax_scale


def flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv, causal=False, softmax_scale=None):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, d_v = v.shape
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK_HEADDIM_V = max(triton.next_power_of_2(d_v), 16)

    def grid(META):
        return (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)

    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d_v,
        BLOCK_M=128,
        BLOCK_HEADDIM_V=BLOCK_HEADDIM_V,
    )

    def grid(META):
        return (
            triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
            batch * nheads,
        )

    _bwd_kernel[grid](
        q,
        k,
        v,
        do,
        dq_accum,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        d_v,
        seqlen_q // 32,
        seqlen_k // 32,
        causal,
        BLOCK_HEADDIM,
        BLOCK_HEADDIM_V,
    )
    dq.copy_(dq_accum)


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, q, k, v, causal=False, softmax_scale=None):
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k: (batch_size, seqlen_k, nheads, headdim)
        v: (batch_size, seqlen_k, nheads, headdim_v)  # headdim_v can be different from headdim
        Returns: (batch_size, seqlen_q, nheads, headdim_v)
        """
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        o, lse, ctx.softmax_scale = flash_attn_forward(q, k, v, causal=causal, softmax_scale=softmax_scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            flash_attn_backward(
                do,  # [batch_size, seq_len_q, n_heads, headdim_v]
                q,
                k,
                v,
                o,
                lse,
                dq,
                dk,
                dv,
                causal=ctx.causal,
                softmax_scale=ctx.softmax_scale,
            )
        return dq, dk, dv, None, None, None


class TopKSparseFusedDecode(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        hidden_pre: torch.Tensor,
        W_D: torch.Tensor,
        b_D: torch.Tensor,
        topk: int,
        sparsity_include_decoder_norm: bool,
    ) -> torch.Tensor:
        d_sae = W_D.shape[0]

        if sparsity_include_decoder_norm:
            _, topk_indices = torch.topk(hidden_pre * W_D.norm(dim=-1), k=topk, dim=-1, sorted=False)
            topk_values = torch.gather(hidden_pre, dim=1, index=topk_indices)
        else:
            topk_values, topk_indices = torch.topk(hidden_pre, k=topk, dim=-1, sorted=False)

        feature_acts = torch.zeros_like(hidden_pre)
        feature_acts.scatter_(dim=1, index=topk_indices, src=topk_values)

        topk_indices_sorted, topk_values_sorted = sort_topk_result(topk_indices, topk_values)

        feature_acts_sparse = build_sparse_csr_from_topk(topk_indices_sorted, topk_values_sorted, d_sae)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            feature_acts_sparse = feature_acts_sparse.to(W_D.dtype)
            output = torch.sparse.mm(feature_acts_sparse, W_D) + b_D

        ctx.save_for_backward(
            hidden_pre,
            W_D,
            b_D,
            feature_acts_sparse,
            topk_indices_sorted,
            topk_values_sorted,
        )

        return output, feature_acts

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx,
        grad_output: torch.Tensor,
        grad_feature_acts_none: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        (
            hidden_pre,
            W_D,
            b_D,
            feature_acts_sparse,
            topk_indices_sorted,
            topk_values_sorted,
        ) = ctx.saved_tensors

        grad_hidden_pre = grad_W_D = grad_b_D = None

        # grad_b_D
        grad_b_D = grad_output.sum(dim=0)

        # grad_W_D
        feature_acts_sparse_T = feature_acts_sparse.to_sparse_coo().T

        feature_acts_sparse_T = feature_acts_sparse_T.to_sparse_csr()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            grad_W_D = torch.sparse.mm(feature_acts_sparse_T, grad_output)

        grad_topk_values = masked_matmul(grad_output, W_D, topk_indices_sorted)

        grad_hidden_pre = torch.zeros_like(hidden_pre, dtype=grad_topk_values.dtype)
        grad_hidden_pre.scatter_(dim=1, index=topk_indices_sorted, src=grad_topk_values)

        return grad_hidden_pre, grad_W_D, grad_b_D, None, None


class DPTopKSparseFusedDecode(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        hidden_pre: torch.Tensor,
        W_D: torch.Tensor,
        b_D: torch.Tensor,
        topk: int,
        sparsity_include_decoder_norm: bool,
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_pre_local, W_D_local, b_D_local = (
            hidden_pre.to_local(),
            W_D.to_local(),
            b_D.to_local(),
        )

        d_sae = W_D_local.shape[0]

        if sparsity_include_decoder_norm:
            _, topk_indices_local = torch.topk(hidden_pre_local * W_D_local.norm(dim=-1), k=topk, dim=-1, sorted=False)
            topk_values_local = torch.gather(hidden_pre_local, dim=1, index=topk_indices_local)
        else:
            topk_values_local, topk_indices_local = torch.topk(hidden_pre_local, k=topk, dim=-1, sorted=False)

        feature_acts_local = torch.zeros_like(hidden_pre_local)
        feature_acts_local.scatter_(dim=1, index=topk_indices_local, src=topk_values_local)

        topk_indices_sorted_local, topk_values_sorted_local = sort_topk_result(topk_indices_local, topk_values_local)

        feature_acts_sparse_local = build_sparse_csr_from_topk(
            topk_indices_sorted_local, topk_values_sorted_local, d_sae
        )

        with torch.amp.autocast(device_type="cuda", enabled=False):
            feature_acts_sparse_local = feature_acts_sparse_local.to(W_D_local.dtype)
            output_local = torch.sparse.mm(feature_acts_sparse_local, W_D_local) + b_D_local

        output = DTensor.from_local(output_local, device_mesh=device_mesh, placements=hidden_pre.placements)
        feature_acts = DTensor.from_local(feature_acts_local, device_mesh=device_mesh, placements=hidden_pre.placements)

        ctx.save_for_backward(
            hidden_pre,
            W_D,
            b_D,
            feature_acts_sparse_local,
            topk_indices_sorted_local,
            topk_values_sorted_local,
        )
        ctx.device_mesh = device_mesh

        return output, feature_acts

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx,
        grad_output: torch.Tensor,
        grad_feature_acts_none: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        (
            hidden_pre,
            W_D,
            b_D,
            feature_acts_sparse_local,
            topk_indices_sorted_local,
            topk_values_sorted_local,
        ) = ctx.saved_tensors
        device_mesh = ctx.device_mesh

        hidden_pre_local, W_D_local, grad_output_local = (
            hidden_pre.to_local(),
            W_D.to_local(),
            grad_output.to_local(),
        )

        grad_hidden_pre_local = grad_W_D_local = grad_b_D_local = None

        # grad_b_D
        grad_b_D_local = grad_output_local.sum(dim=0)

        # grad_W_D
        feature_acts_sparse_T_local = feature_acts_sparse_local.to_sparse_coo().T
        feature_acts_sparse_T_local = feature_acts_sparse_T_local.to_sparse_csr()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            grad_W_D_local = torch.sparse.mm(feature_acts_sparse_T_local, grad_output_local)

        grad_topk_values_local = masked_matmul(grad_output_local, W_D_local, topk_indices_sorted_local)

        grad_hidden_pre_local = torch.zeros_like(hidden_pre_local, dtype=grad_topk_values_local.dtype)
        grad_hidden_pre_local.scatter_(dim=1, index=topk_indices_sorted_local, src=grad_topk_values_local)

        # Aggregate gradients across data parallel dimension
        grad_b_D = DTensor.from_local(
            grad_b_D_local.unsqueeze(-1),
            device_mesh=device_mesh,
            placements=DimMap({"data": 1}).placements(device_mesh),
        )
        grad_b_D = grad_b_D.sum(dim=1)

        grad_W_D = DTensor.from_local(
            grad_W_D_local.unsqueeze(-1),
            device_mesh=device_mesh,
            placements=DimMap({"data": 2, "model": 0}).placements(device_mesh),
        )
        grad_W_D = grad_W_D.sum(dim=2)

        grad_hidden_pre = DTensor.from_local(
            grad_hidden_pre_local,
            device_mesh=device_mesh,
            placements=hidden_pre.placements,
        )

        return grad_hidden_pre, grad_W_D, grad_b_D, None, None, None
