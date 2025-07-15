from typing import Union

import torch
from jaxtyping import Float

from .kernels import TritonDecoderAutogradDynamicK, TritonDecoderAutogradTopK

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
    sparse_indices = torch.full((B, cast(int, nnz_max)), pad_val, dtype=torch.long, device=x.device)
    sparse_values = torch.zeros((B, cast(int, nnz_max)), dtype=x.dtype, device=x.device)

    # Compute position indices for scattering
    cum_nnzs = torch.cumsum(nnz_per_row, dim=0)
    row_offsets = torch.cat([torch.tensor([0], device=x.device), cum_nnzs[:-1]])  # Shift right for offsets
    positions = torch.arange(len(batch_indices), device=x.device) - row_offsets[batch_indices]

    # Scatter indices and values
    sparse_indices[batch_indices, positions] = col_indices
    sparse_values[batch_indices, positions] = values

    return sparse_indices, sparse_values


def decode_with_triton_spmm_kernel(
    feature_acts: Union[
        Float[torch.Tensor, "batch d_sae"],
        Float[torch.sparse.Tensor, "batch d_sae"]
    ],
    decoder_weight: Float[torch.Tensor, "d_model d_sae"],
    dynamic_k: bool = True,
) -> Union[
    Float[torch.Tensor, "batch d_model"],
    Float[torch.Tensor, "batch n_layers d_model"],
    Float[torch.Tensor, "batch seq_len d_model"],
]:
    """
    Perform sparse-dense matrix multiplication using Triton.

    Args:
        feature_acts: (B, d_sae) - Sparse feature activations (input).
        decoder_weight: (d_sae, d_model) - Decoder weight matrix.

    Returns:
        output: (B, d_model) - The decoded output.
    """
    if isinstance(feature_acts, torch.sparse.Tensor):
        sparse_indices, sparse_values = feature_acts.indices(), feature_acts.values()
        output = TritonDecoderAutogradTopK.apply(sparse_indices, sparse_values, decoder_weight.contiguous().T)
        return output  # type: ignore[return-value]
    
    if dynamic_k:
        output = TritonDecoderAutogradDynamicK.apply(feature_acts, decoder_weight.contiguous().T)
    else:
        sparse_indices, sparse_values = get_sparse_representation(feature_acts)
        output = TritonDecoderAutogradTopK.apply(sparse_indices, sparse_values, decoder_weight.contiguous().T)
    return output  # type: ignore[return-value]



if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import triton
    import triton.language as tl

    def test_triton_decoder(
        B, d_sae, d_model, sparsity=0.9, dtype=torch.float32, require_precise_feature_acts_grad=True
    ):
        # Set parameters

        # Create a random dense weight matrix (as in nn.Linear)
        decoder = nn.Linear(d_sae, d_model, bias=False, dtype=dtype, device="cuda")

        # Create a random sparse input matrix
        dense_input = torch.randn((B, d_sae), dtype=dtype, device="cuda")

        # Zero out some values to simulate sparsity
        dense_input[torch.rand_like(dense_input) < sparsity] = 0

        # Enable gradient tracking
        decoder.weight.requires_grad_(True)
        dense_input.requires_grad_(True)

        grad_output = torch.randn((B, d_model), dtype=dtype, device="cuda")

        # Run forward pass with Triton
        triton_output = decode_with_triton_spmm_kernel(dense_input, decoder.weight, require_precise_feature_acts_grad)
        assert isinstance(triton_output, torch.Tensor), "triton_output is not a torch.Tensor"

        triton_output.backward(grad_output)

        triton_decoder_weight_grad, triton_dense_input_grad = decoder.weight.grad.clone(), dense_input.grad.clone()  # pyright: ignore

        decoder.weight.grad.zero_()  # pyright: ignore
        dense_input.grad.zero_()  # pyright: ignore

        torch_output = decoder(dense_input)
        torch_output.backward(grad_output)

        torch_decoder_weight_grad, torch_dense_input_grad = decoder.weight.grad.clone(), dense_input.grad.clone()  # pyright: ignore

        # Compare gradients
        assert decoder.weight.grad is not None, "decoder.weight.grad is None"
        assert torch.allclose(triton_output, torch_output, atol=1e-5), "Mismatch between Triton and PyTorch outputs!"
        assert torch.allclose(triton_decoder_weight_grad, torch_decoder_weight_grad, atol=1e-5), (
            f"Mismatch between Triton and PyTorch gradients on decoder weights! {triton_decoder_weight_grad=}, {torch_decoder_weight_grad=}"
        )

        if require_precise_feature_acts_grad:
            assert torch.allclose(triton_dense_input_grad, torch_dense_input_grad, atol=1e-5), (
                f"Mismatch between Triton and PyTorch gradients on dense input! {triton_dense_input_grad=}, {torch_dense_input_grad=}"
            )
        else:
            assert torch.allclose(
                triton_dense_input_grad[dense_input.ne(0)], torch_dense_input_grad[dense_input.ne(0)], atol=1e-5
            ), (
                f"Mismatch between Triton and PyTorch gradients on dense input! {triton_dense_input_grad=}, {torch_dense_input_grad=}"
            )

        logger.info("✅ Triton forward and backward pass matches nn.Linear!")

    # Ensure we have the Triton-based kernel
    def benchmark_triton_vs_torch(
        B=32,
        d_sae=512,
        d_model=256,
        sparsity=0.7,
        warmup=5,
        iters=20,
        dtype=torch.float32,
        require_precise_feature_acts_grad=True,
    ):
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
        decoder = nn.Linear(d_sae, d_model, bias=False, dtype=dtype, device="cuda")

        # Generate a dense input
        dense_input = torch.randn((B, d_sae), dtype=dtype, device="cuda")

        # Introduce sparsity
        dense_input[torch.rand_like(dense_input) < sparsity] = 0

        # Warmup runs (to eliminate startup overhead)
        for _ in range(warmup):
            torch_output = decoder(dense_input)
            triton_output = decode_with_triton_spmm_kernel(
                dense_input, decoder.weight, require_precise_feature_acts_grad
            )
            assert isinstance(triton_output, torch.Tensor), "triton_output is not a torch.Tensor"
            grad_output = torch.randn_like(triton_output)
            triton_output.backward(grad_output)
            torch_output.backward(grad_output)

        # Measure nn.Linear time
        torch.cuda.synchronize()
        start_torch = torch.cuda.Event(enable_timing=True)
        end_torch = torch.cuda.Event(enable_timing=True)

        start_torch.record()  # type: ignore ; There should be a stream argument and we don't know why
        for _ in range(iters):
            torch_output = decoder(dense_input)
            grad_output = torch.randn_like(torch_output)
            torch_output.backward(grad_output)

        end_torch.record()  # type: ignore
        torch.cuda.synchronize()
        torch_time = start_torch.elapsed_time(end_torch) / iters  # Average time in ms

        # Measure Triton Kernel time
        torch.cuda.synchronize()
        start_triton = torch.cuda.Event(enable_timing=True)
        end_triton = torch.cuda.Event(enable_timing=True)

        start_triton.record()  # type: ignore
        for _ in range(iters):
            triton_output = decode_with_triton_spmm_kernel(
                dense_input, decoder.weight, require_precise_feature_acts_grad
            )
            assert isinstance(triton_output, torch.Tensor), "triton_output is not a torch.Tensor"
            grad_output = torch.randn_like(triton_output)
            triton_output.backward(grad_output)

        end_triton.record()  # type: ignore
        torch.cuda.synchronize()
        triton_time = start_triton.elapsed_time(end_triton) / iters  # Average time in ms

        # Print results
        logger.info(f"🔹 PyTorch nn.Linear Avg Time: {torch_time:.3f} ms")
        logger.info(f"⚡ Triton Sparse-Dense Avg Time: {triton_time:.3f} ms")
        logger.info(f"🚀 Speedup: {torch_time / triton_time:.2f}x")

    # Run test
    test_triton_decoder(B=16, d_sae=4096, d_model=256, sparsity=0.9, require_precise_feature_acts_grad=True)
    # Run benchmark
    benchmark_triton_vs_torch(
        B=8192,
        d_sae=4096 * 32,
        d_model=4096,
        sparsity=0.999,
        warmup=10,
        iters=10,
        require_precise_feature_acts_grad=True,
    )
