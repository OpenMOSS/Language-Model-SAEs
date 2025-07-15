"""Triton kernels for sparse matrix operations in SAE training."""

from typing import cast

import torch
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.device_mesh import DeviceMesh
import triton
import triton.language as tl
from jaxtyping import Float
from typing import Union

from lm_saes.utils.logging import get_logger

logger = get_logger("kernels")


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

    if K > 512:
        logger.warning("Using naive matmul for large K")
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

class TritonDecoderAutogradDynamicK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature_acts, decoder_weight):
        sparse_indices, sparse_values = get_sparse_representation(feature_acts)
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

        # decoder is contiguous when transposed so this is a matching layout
        return grad_output @ decoder_weight, decoder_grad


def triton_clt_sparse_backward_x(grad_hidden_pre, W_E):
    """
    Compute grad_x = einsum("...ls,lds->...ld", grad_hidden_pre, W_E)
    Accelerate computation using sparsity in grad_hidden_pre
    
    Args:
        grad_hidden_pre: (..., n_layers, d_sae) - Sparse gradients
        W_E: (n_layers, d_model, d_sae) - Encoder weights
    
    Returns:
        grad_x: (..., n_layers, d_model) - Input gradients
    """
    batch_dims = grad_hidden_pre.shape[:-2]
    n_layers, d_sae = grad_hidden_pre.shape[-2:]
    d_model = W_E.shape[-2]
    
    # Initialize output
    grad_x = torch.zeros(*batch_dims, n_layers, d_model, device=grad_hidden_pre.device, dtype=grad_hidden_pre.dtype)
    
    # Process layer by layer, each layer is independent
    for layer in range(n_layers):
        grad_hidden_layer = grad_hidden_pre[..., layer, :]  # (..., d_sae)
        W_E_layer = W_E[layer]  # (d_model, d_sae)
        
        # Use triton sparse kernel (caller has already checked sparsity condition)
        # Reshape to 2D for sparse matrix multiplication
        batch_size = grad_hidden_layer.flatten(end_dim=-2).shape[0]
        grad_hidden_flat = grad_hidden_layer.flatten(end_dim=-2).contiguous()  # (batch_flattened, d_sae)
        
        # Get sparse representation
        sparse_indices, sparse_values = get_sparse_representation(grad_hidden_flat)
        
        # Sparse matrix multiplication: grad_hidden_flat @ W_E_layer.T
        # Make W_E_layer.T contiguous to satisfy kernel requirements
        W_E_layer_T_contiguous = W_E_layer.T.contiguous()
        grad_x_flat = triton_sparse_dense_matmul(sparse_indices, sparse_values, W_E_layer_T_contiguous)
        
        # Reshape back to original shape
        grad_x[..., layer, :] = grad_x_flat.view(*batch_dims, d_model)
    
    return grad_x


def triton_clt_sparse_backward_W_E(x, grad_hidden_pre):
    """
    Compute grad_W_E = einsum("...ld,...ls->lds", x, grad_hidden_pre)
    Accelerate computation using sparsity in grad_hidden_pre
    
    Args:
        x: (..., n_layers, d_model) - Input activations
        grad_hidden_pre: (..., n_layers, d_sae) - Sparse gradients
    
    Returns:
        grad_W_E: (n_layers, d_model, d_sae) - Encoder weight gradients
    """
    batch_dims = x.shape[:-2]
    n_layers, d_model = x.shape[-2:]
    d_sae = grad_hidden_pre.shape[-1]
    
    # Initialize output
    grad_W_E = torch.zeros(n_layers, d_model, d_sae, device=x.device, dtype=x.dtype)
    
    # Process layer by layer
    for layer in range(n_layers):
        x_layer = x[..., layer, :]  # (..., d_model)
        grad_hidden_layer = grad_hidden_pre[..., layer, :]  # (..., d_sae)
        
        # Use triton sparse kernel (caller has already checked sparsity condition)
        # Reshape to 2D
        batch_size = x_layer.flatten(end_dim=-2).shape[0]
        x_flat = x_layer.flatten(end_dim=-2).contiguous()  # (batch_flattened, d_model)
        grad_hidden_flat = grad_hidden_layer.flatten(end_dim=-2).contiguous()  # (batch_flattened, d_sae)
        
        # Get sparse representation
        sparse_indices, sparse_values = get_sparse_representation(grad_hidden_flat)
        
        # Sparse matrix multiplication: x_flat.T @ grad_hidden_sparse
        grad_W_E_layer = triton_sparse_transpose_dense_matmul(
            sparse_indices, sparse_values, x_flat, N=d_sae
        ).T  # (d_model, d_sae)
        
        grad_W_E[layer] = grad_W_E_layer
    
    return grad_W_E


def encode_with_triton_clt_kernel(
    x: Float[torch.Tensor, "... n_layers d_model"],
    W_E: Float[torch.Tensor, "n_layers d_model d_sae"],
    b_E: Float[torch.Tensor, "n_layers d_sae"],
    k: int,
    device_mesh: DeviceMesh,
) -> Float[torch.Tensor, "... n_layers d_sae"]:
    """
    Triton-accelerated version of CLT encoding with distributed training support
    
    Args:
        x: Input activations (..., n_layers, d_model)
        W_E: Encoder weights (n_layers, d_model, d_sae)
        b_E: Encoder biases (n_layers, d_sae)
        k: Top-k activation count
        device_mesh: Distributed device mesh
    
    Returns:
        feature_acts: CLT feature activations (..., n_layers, d_sae)
    """
    # We need to modify TritonEncoderAutogradDynamicK to accept bias parameter
    # or compute einsum here first then pass to the modified autograd function
    
    # Compute hidden_pre = einsum("...ld,lds->...ls", x, W_E) + b_E
    hidden_pre = torch.einsum("...ld,lds->...ls", x, W_E) + b_E
    
    # Use modified autograd function to handle batchtopk and backward
    result = TritonEncoderAutogradFromHiddenPre.apply(hidden_pre, x, W_E, k, device_mesh)
    assert isinstance(result, torch.Tensor), "TritonEncoderAutogradFromHiddenPre should return a tensor"
    return result


class TritonEncoderAutogradFromHiddenPre(torch.autograd.Function):
    """
    CLT triton autograd function starting from computed hidden_pre
    Specialized for handling batchtopk activation and sparse backward
    """
    @staticmethod
    def forward(ctx, hidden_pre, x, W_E, k, device_mesh):
        # hidden_pre: (..., n_layers, d_sae) - Pre-activation including bias
        # x: (..., n_layers, d_model) - Original input
        # W_E: (n_layers, d_model, d_sae) - Encoder weights
        
        assert isinstance(hidden_pre, DTensor)
        
        # Apply batchtopk activation function
        from lm_saes.utils.distributed import distributed_batch_kthvalue_clt_binary_search
        threshold, _ = distributed_batch_kthvalue_clt_binary_search(hidden_pre, (k-1, k), device_mesh=device_mesh)
        mask = hidden_pre >= threshold
        feature_acts = hidden_pre * mask
        
        # Save information needed for backward
        ctx.save_for_backward(x, W_E, mask)
        ctx.device_mesh = device_mesh
        return feature_acts

    @staticmethod
    def backward(ctx, *grad_outputs, **args):
        assert len(grad_outputs) == 1, "grad_outputs must be a single tensor"
        grad_output = grad_outputs[0]
        x, W_E, mask = ctx.saved_tensors
        grad_hidden_pre = grad_output * mask  # Only topk positions have gradients
        
        # Use our optimized backward functions
        assert isinstance(grad_hidden_pre, DTensor)
        
        # Convert to local tensors for triton operations
        grad_hidden_pre_local = grad_hidden_pre.to_local()
        x_local = x.to_local()
        W_E_local = W_E.to_local()
        
        grad_x_local = triton_clt_sparse_backward_x(grad_hidden_pre_local, W_E_local)
        grad_W_E_local = triton_clt_sparse_backward_W_E(x_local, grad_hidden_pre_local)
        
        # Convert back to DTensor
        from lm_saes.utils.distributed import DimMap
        
        grad_x = DTensor.from_local(
            grad_x_local,
            device_mesh=ctx.device_mesh,
            placements=[Replicate()]
        )
        grad_W_E = DTensor.from_local(
            grad_W_E_local,
            device_mesh=ctx.device_mesh,
            placements=DimMap({"model": 2}).placements(ctx.device_mesh)
        )
        
        # For b_E gradient (bias gradient equals sum of grad_hidden_pre over batch dimensions)
        grad_b_E_local = grad_hidden_pre_local.sum(dim=tuple(range(len(grad_hidden_pre_local.shape) - 2)))
        grad_b_E = DTensor.from_local(
            grad_b_E_local,
            device_mesh=ctx.device_mesh,
            placements=DimMap({"model": 1}).placements(ctx.device_mesh)
        )
        
        # Return order: hidden_pre, x, W_E, k, device_mesh
        return grad_hidden_pre, grad_x, grad_W_E, None, None


class TritonDecoderAutogradTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_indices, sparse_values, decoder_weight):
        ctx.save_for_backward(sparse_indices, sparse_values, decoder_weight)
        print(f"sparse_indices: {sparse_indices.shape}, sparse_values: {sparse_values.shape}, decoder_weight: {decoder_weight.shape}")
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

        # decoder is contiguous when transposed so this is a matching layout
        return None, triton_dense_dense_sparseout_matmul(grad_output, decoder_weight, sparse_indices), decoder_grad
