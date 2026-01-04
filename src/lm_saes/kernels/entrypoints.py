from typing import Union

import torch
from jaxtyping import Float

from lm_saes.kernels.kernels import TritonDecoderAutogradTopK, TritonEncoderAutogradDynamicK, get_sparse_representation
from lm_saes.utils.logging import get_logger

logger = get_logger("kernels")


def decode_with_triton_spmm_kernel(
    feature_acts: Union[Float[torch.Tensor, "batch d_sae"], Float[torch.sparse.Tensor, "batch d_sae"]],
    decoder_weight: Float[torch.Tensor, "d_sae d_model"],
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
        dynamic_k: bool - Whether to use dynamic k selection.
        sparsity_threshold: float - Sparsity threshold for using sparse kernels.

    Returns:
        output: (B, d_model) - The decoded output.
    """
    if feature_acts.is_sparse:
        sparse_indices, sparse_values = feature_acts.indices(), feature_acts.values()
    else:
        sparse_indices, sparse_values = get_sparse_representation(feature_acts)
    return TritonDecoderAutogradTopK.apply(sparse_indices, sparse_values, decoder_weight.contiguous().T)  # type: ignore[return-value]


def encode_with_triton_spmm_kernel(
    x: Float[torch.Tensor, "batch n_layers d_model"],
    W_E: Float[torch.Tensor, "n_layers d_model d_sae"],
    b_E: Float[torch.Tensor, "n_layers d_sae"],
    sparsity_threshold: float = 0.996,
) -> Float[torch.Tensor, "batch n_layers d_sae"]:
    """
    Perform sparse-dense matrix multiplication using Triton for encoding.

    This function implements the "bld,lds->bls" operation by processing each layer
    separately and stacking the results, as requested. Provides only for-loop implementations.

    Args:
        x: (batch, n_layers, d_model) - Input activations from all layers.
        W_E: (n_layers, d_model, d_sae) - Encoder weight matrices for each layer.
        b_E: (n_layers, d_sae) - Encoder bias vectors for each layer.
        sparsity_threshold: (float) - Sparsity threshold for using sparse kernels.

    Returns:
        output: (batch, n_layers, d_sae) - The encoded output for all layers.
    """
    batch_size, n_layers, _ = x.shape
    d_sae = W_E.shape[2]

    # For-loop implementation as originally requested
    output = torch.zeros(batch_size, n_layers, d_sae, device=x.device, dtype=x.dtype)

    # Process each layer separately using the approach suggested by the user
    for layer_idx in range(n_layers):
        # Extract activations for this layer: (batch, d_model)
        x_layer = x[:, layer_idx, :]

        # Extract encoder weights for this layer: (d_model, d_sae)
        W_E_layer = W_E[layer_idx, :, :]

        # Extract bias for this layer: (d_sae,)
        b_E_layer = b_E[layer_idx, :]

        # Compute: x_layer @ W_E_layer + b_E_layer
        # Result shape: (batch, d_sae)
        layer_output = TritonEncoderAutogradDynamicK.apply(x_layer, W_E_layer, b_E_layer, sparsity_threshold)
        assert isinstance(layer_output, torch.Tensor), "TritonEncoderAutogradDynamicK must return a tensor"

        # Store result in output tensor
        output[:, layer_idx, :] = layer_output

    return output


if __name__ == "__main__":
    import torch
    import torch.nn as nn

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
        triton_output = decode_with_triton_spmm_kernel(dense_input, decoder.weight)
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
            triton_output = decode_with_triton_spmm_kernel(dense_input, decoder.weight)
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
            triton_output = decode_with_triton_spmm_kernel(dense_input, decoder.weight)
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
