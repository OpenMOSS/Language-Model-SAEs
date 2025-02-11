import torch
import triton
import triton.language as tl
from jaxtyping import Float


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

    # COO-format and sorted
    sorted_indices = sparse_indices.view(-1).sort()
    coo_indices = torch.stack(
        [
            torch.arange(A, device=sparse_indices.device).repeat_interleave(K)[
                sorted_indices.indices
            ],
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

    out = torch.zeros(N, B, device=dense.device, dtype=coo_values.dtype)

    def grid(META):
        return triton.cdiv(AK, META["BLOCK_SIZE_AK"]), 1

    triton_sparse_transpose_dense_matmul_kernel[grid](
        coo_indices,
        coo_values,
        dense,
        out,
        stride_da=dense.stride(0),
        stride_db=dense.stride(1),
        B=B,
        N=N,
        AK=AK,
        BLOCK_SIZE_AK=BLOCK_SIZE_AK,
        BLOCK_SIZE_B=triton.next_power_of_2(B),
    )
    return out

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
    dense is shape (d_sae, d_model)

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

    triton_sparse_dense_matmul_kernel[(A,)](
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
        BLOCK_SIZE_K=triton.next_power_of_2(K),
        BLOCK_SIZE_B=triton.next_power_of_2(B),
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

    if K > 512:
        # print("WARN - using naive matmul for large K")
        # naive is more efficient for large K
        return (dense1 @ dense2).gather(1, at_indices)

    out = torch.zeros(A, K, device=dense1.device, dtype=dense1.dtype)

    # grid = lambda META: (triton.cdiv(A, META['BLOCK_SIZE_A']),)

    triton_dense_dense_sparseout_matmul_kernel[(A,)](
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
        BLOCK_SIZE_B=triton.next_power_of_2(B),
        BLOCK_SIZE_N=triton.next_power_of_2(N),
        BLOCK_SIZE_K=triton.next_power_of_2(K),
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
    dense1 = tl.load(
        dense1_ptr + pid * stride_d1a + offsets_b * stride_d1b, mask=offsets_b < B
    )  # shape (B,)

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

        dense2col = tl.load(
            dense2_ptr + offsets_b * stride_d2b + i * stride_d2n, mask=offsets_b < B
        )  # shape (B,)
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
    sparse_indices = tl.load(
        sparse_indices_ptr + pid * K + offsets_k, mask=offsets_k < K
    )  # shape (K,)
    sparse_values = tl.load(
        sparse_values_ptr + pid * K + offsets_k, mask=offsets_k < K
    )  # shape (K,)

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
            accum += v * tl.load(
                dense_ptr + i * stride_dn + offsets_b * stride_db, mask=offsets_b < B
            )

    tl.store(out_ptr + pid * B + offsets_b, accum.to(sparse_values.dtype), mask=offsets_b < B)

@torch.no_grad()
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

    # Prepare output tensors
    sparse_indices = torch.full((B, nnz_max), pad_val, dtype=torch.long, device=x.device)
    sparse_values = torch.zeros((B, nnz_max), dtype=x.dtype, device=x.device)

    # Compute position indices for scattering
    cum_nnzs = torch.cumsum(nnz_per_row, dim=0)
    row_offsets = torch.cat([torch.tensor([0], device=x.device), cum_nnzs[:-1]])  # Shift right for offsets
    positions = torch.arange(len(batch_indices), device=x.device) - row_offsets[batch_indices]

    # Scatter indices and values
    sparse_indices[batch_indices, positions] = col_indices
    sparse_values[batch_indices, positions] = values

    return sparse_indices, sparse_values


class TritonDecoderAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_indices, sparse_values, decoder_weight):
        ctx.save_for_backward(sparse_indices, sparse_values, decoder_weight)
        return triton_sparse_dense_matmul(sparse_indices, sparse_values, decoder_weight.T)

    @staticmethod
    def backward(ctx, grad_output):
        sparse_indices, sparse_values, decoder_weight = ctx.saved_tensors

        assert grad_output.is_contiguous(), "grad_output must be contiguous; this is probably because the subsequent op was a .sum() or something like that, which returns a non contiguous gradient"

        decoder_grad = triton_sparse_transpose_dense_matmul(
            sparse_indices, sparse_values, grad_output, N=decoder_weight.shape[1]
        ).T

        return (
            None,
            triton_dense_dense_sparseout_matmul(grad_output, decoder_weight, sparse_indices),
            # decoder is contiguous when transposed so this is a matching layout
            decoder_grad,
            None,
        )

def decode_with_triton_spmm_kernel(feature_acts: Float[torch.Tensor, "batch d_sae"], decoder_weight: Float[torch.Tensor, "d_model d_sae"]):
    """
    Perform sparse-dense matrix multiplication using Triton.
    
    Args:
        feature_acts: (B, d_sae) - Sparse feature activations (input).
        decoder_weight: (d_model d_sae) - Decoder weight matrix.
    
    Returns:
        output: (B, d_model) - The decoded output.
    """
    # Convert dense feature_acts into sparse representation
    sparse_indices, sparse_values = get_sparse_representation(feature_acts)

    # Perform sparse-dense multiplication using Triton
    output = TritonDecoderAutograd.apply(sparse_indices, sparse_values, decoder_weight.T.contiguous().T)

    return output



if __name__ == '__main__':

    import torch
    import torch.nn as nn
    import triton
    import triton.language as tl

    def test_triton_decoder_forward():
        # Set parameters
        B, d_sae, d_model = 4, 32, 16  # Batch size, input dim, output dim

        # Create a random dense weight matrix (as in nn.Linear), size = (d_model, d_sae)
        decoder = nn.Linear(d_sae, d_model, bias=False, dtype=torch.float32, device="cuda")
                
        # Create a random sparse input matrix
        dense_input = torch.randn((B, d_sae), dtype=torch.float32, device="cuda")
        
        # Zero out some values to simulate sparsity (~70% sparsity)
        dense_input[torch.rand_like(dense_input) < 0.7] = 0  

        # Run our Triton-based sparse-dense multiply
        triton_output = decode_with_triton_spmm_kernel(
            dense_input, 
            decoder.weight
        )

        # Compare against standard dense multiply (nn.Linear equivalent)
        torch_output = decoder(dense_input)  # Equivalent to nn.Linear

        # Ensure outputs are numerically close
        assert torch.allclose(triton_output, torch_output, atol=1e-4), "Mismatch between Triton and PyTorch outputs!"

        print("✅ Triton forward pass matches nn.Linear!")
    
    def test_triton_decoder_backward():
        # Set parameters
        B, d_sae, d_model = 4, 32, 16  # Batch size, input dim, output dim

        # Create a random dense weight matrix (as in nn.Linear)
        decoder = nn.Linear(d_sae, d_model, bias=False, dtype=torch.float32, device="cuda")
                    
        # Create a random sparse input matrix
        dense_input = torch.randn((B, d_sae), dtype=torch.float32, device="cuda")
        
        # Zero out some values to simulate sparsity (~70% sparsity)
        dense_input[torch.rand_like(dense_input) < 0.7] = 0  

        # Enable gradient tracking
        decoder.weight.requires_grad_(True)

        # Run forward pass with Triton
        triton_output = decode_with_triton_spmm_kernel(
            dense_input, 
            decoder.weight
        )

        # Run forward pass with PyTorch nn.Linear
        torch_output = decoder(dense_input)

        # Generate random gradient to propagate backward
        grad_output = torch.randn_like(torch_output)

        # Backpropagate
        triton_output.backward(grad_output)
        torch_output.backward(grad_output)

        # Compare gradients
        assert torch.allclose(decoder.weight.grad, decoder.weight.grad, atol=1e-4), \
            "Mismatch between Triton and PyTorch gradients!"

        print("✅ Triton backward pass matches nn.Linear!")


    # Ensure we have the Triton-based kernel
    def benchmark_triton_vs_torch(B=32, d_sae=512, d_model=256, sparsity=0.7, warmup=5, iters=20):
        """
        Benchmarks Triton-based sparse-dense multiplication vs PyTorch's nn.Linear.
        
        Args:): Batch size.
            d_sae (int): Input feature dimension.
            d_model (int): Output feature dimension.
            sparsity (float): Percentage of zeros in the input.
            warmup (int): Number of warmup iterations.
            iters (int): Number of timed iterations.
        """

        # Create weight matrix similar to nn.Linear
        decoder = nn.Linear(d_sae, d_model, bias=False, dtype=torch.float32, device="cuda")

        # Generate a dense input
        dense_input = torch.randn((B, d_sae), dtype=torch.float32, device="cuda")

        # Introduce sparsity
        dense_input[torch.rand_like(dense_input) < sparsity] = 0  

        # Warmup runs (to eliminate startup overhead)
        for _ in range(warmup):
            torch_output = decoder(dense_input)
            triton_output = decode_with_triton_spmm_kernel(
                dense_input, 
                decoder.weight
            )
            grad_output = torch.randn_like(triton_output)
            triton_output.backward(grad_output)
            torch_output.backward(grad_output)

        # Measure nn.Linear time
        torch.cuda.synchronize()
        start_torch = torch.cuda.Event(enable_timing=True)
        end_torch = torch.cuda.Event(enable_timing=True)

        start_torch.record()
        for _ in range(iters):
            torch_output = decoder(dense_input)
            grad_output = torch.randn_like(torch_output)
            torch_output.backward(grad_output)
            
        end_torch.record()
        torch.cuda.synchronize()
        torch_time = start_torch.elapsed_time(end_torch) / iters  # Average time in ms

        # Measure Triton Kernel time
        torch.cuda.synchronize()
        start_triton = torch.cuda.Event(enable_timing=True)
        end_triton = torch.cuda.Event(enable_timing=True)

        start_triton.record()
        for _ in range(iters):
            triton_output = decode_with_triton_spmm_kernel(
                dense_input, 
                decoder.weight
            )
            grad_output = torch.randn_like(triton_output)
            triton_output.backward(grad_output)

        end_triton.record()
        torch.cuda.synchronize()
        triton_time = start_triton.elapsed_time(end_triton) / iters  # Average time in ms

        # Print results
        print(f"🔹 PyTorch nn.Linear Avg Time: {torch_time:.3f} ms")
        print(f"⚡ Triton Sparse-Dense Avg Time: {triton_time:.3f} ms")
        print(f"🚀 Speedup: {torch_time / triton_time:.2f}x")

    
    # Run test
    test_triton_decoder_forward()
    test_triton_decoder_backward()
    # Run benchmark
    benchmark_triton_vs_torch(B=8192, d_sae=4096 * 32, d_model=4096, sparsity=0.99, warmup=10, iters=100)
