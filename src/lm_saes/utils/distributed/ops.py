from typing import Literal, Tuple, Union, overload

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard


def full_tensor(x: Tensor) -> Tensor:
    """Convert DTensor to regular Tensor if needed."""
    if isinstance(x, DTensor):
        return x.full_tensor()
    return x


def to_local(x: Tensor) -> Tensor:
    """Convert DTensor to local Tensor if needed."""
    if isinstance(x, DTensor):
        return x.to_local()
    return x


def item(x: Tensor) -> float:
    """Extract item from a Tensor. A dedicated function is necessary because DTensor.item() silently returns the local value."""
    return full_tensor(x).item()


@overload
def distributed_topk(
    x: Float[DTensor, "batch n_layers d_sae"],
    k: int,
    device_mesh: DeviceMesh,
    dim: Union[int, Tuple[int, ...]] = -1,
    tolerance: int = 1,
    max_iterations: int = 50,
    mesh_dim_name: str = "model",
    *,
    return_threshold: Literal[False] = False,
) -> Float[DTensor, "batch n_layers d_sae"]: ...


@overload
def distributed_topk(
    x: Float[DTensor, "batch n_layers d_sae"],
    k: int,
    device_mesh: DeviceMesh,
    dim: Union[int, Tuple[int, ...]] = -1,
    tolerance: int = 1,
    max_iterations: int = 50,
    mesh_dim_name: str = "model",
    *,
    return_threshold: Literal[True],
) -> Tuple[Float[DTensor, "batch n_layers d_sae"], Float[Tensor, ""]]: ...


def distributed_topk(
    x: Float[DTensor, "batch n_layers d_sae"],
    k: int,
    device_mesh: DeviceMesh,
    dim: Union[int, Tuple[int, ...]] = -1,
    tolerance: int = 1,
    max_iterations: int = 50,
    mesh_dim_name: str = "model",
    *,
    return_threshold: bool = False,
) -> Union[Float[DTensor, "batch n_layers d_sae"], Tuple[Float[DTensor, "batch n_layers d_sae"], Float[Tensor, ""]]]:
    """
    Perform distributed batch kthvalue operation on a DTensor using binary search.

    Args:
        x: Input tensor of shape (batch, n_layers, d_sae)
        k: Target number of top elements to keep
        device_mesh: Device mesh for distributed training
        dim: Dimension(s) along which to perform topk operation
        tolerance: Acceptable range for the number of elements above threshold
        max_iterations: Maximum number of binary search iterations
        mesh_dim_name: Name of the mesh dimension to shard along
        return_threshold: If True, return both the result tensor and the threshold value

    Returns:
        If return_threshold is False, returns the filtered DTensor.
        If return_threshold is True, returns a tuple of (filtered DTensor, threshold).
    """
    local_tensor = x.to_local()
    placements = x.placements

    with torch.no_grad():
        mesh_dim_idx = None
        if device_mesh.mesh_dim_names is not None:
            try:
                mesh_dim_idx = device_mesh.mesh_dim_names.index(mesh_dim_name)
            except ValueError:
                raise ValueError(f"Mesh dimension '{mesh_dim_name}' not found in device mesh")

        # Check if the tensor is sharded along the specified dimensions
        if mesh_dim_idx is None or not isinstance(placements[mesh_dim_idx], Shard):
            raise ValueError("x must be sharded along the specified dimension")

        shard_dim: Tuple[int] = (placements[mesh_dim_idx].dim,)  # type: ignore
        if isinstance(dim, int):
            dim = (dim,)

        def _ensure_positive_dim(dim: Tuple[int, ...]) -> Tuple[int, ...]:
            """We want to ensure that the dims are positive"""
            return tuple(d if d >= 0 else d + local_tensor.ndim for d in dim)

        dim = _ensure_positive_dim(dim)

        if not any(d in shard_dim for d in dim):
            raise ValueError("At least one of the specified dimensions must be sharded")

        constant_dims = tuple(d for d in range(local_tensor.ndim) if d not in dim)
        constant_dim_size = tuple(local_tensor.size(d) for d in constant_dims)

        k_lower_bound, k_upper_bound = k - tolerance, k + tolerance
        search_low_val = torch.zeros(constant_dim_size, device=local_tensor.device)
        search_high_val = torch.full(constant_dim_size, local_tensor.max().item(), device=local_tensor.device)

        local_tensor_flat = local_tensor.flatten(start_dim=len(constant_dims))

        group = device_mesh.get_group(mesh_dim_name)

        threshold = (search_low_val + search_high_val) / 2
        for _ in range(max_iterations):
            threshold = (search_low_val + search_high_val) / 2
            torch.distributed.all_reduce(threshold, group=group, op=torch.distributed.ReduceOp.AVG)

            count_above_threshold = (local_tensor_flat > threshold.unsqueeze(-1)).sum(-1)
            # All-reduce to get total count across all ranks
            torch.distributed.all_reduce(count_above_threshold, group=group)

            if ((k_lower_bound <= count_above_threshold) * (count_above_threshold <= k_upper_bound)).all():
                break

            to_increase = count_above_threshold > k_upper_bound
            to_decrease = count_above_threshold < k_lower_bound

            if to_increase.any():
                search_low_val = torch.where(to_increase, threshold, search_low_val)
            if to_decrease.any():
                search_high_val = torch.where(to_decrease, threshold, search_high_val)

            # Check for convergence across all devices
            local_converged = (search_high_val - search_low_val < 1e-6).all()
            converged_tensor = local_converged.float().detach().clone()
            torch.distributed.all_reduce(converged_tensor, group=group, op=torch.distributed.ReduceOp.MIN)
            if converged_tensor.item() > 0:  # All devices have converged
                break

        while threshold.ndim < local_tensor.ndim:
            threshold = threshold[..., None]

    local_tensor = local_tensor * local_tensor.ge(threshold)

    result = DTensor.from_local(
        local_tensor,
        device_mesh=device_mesh,
        placements=placements,
    )

    if return_threshold:
        return result, threshold
    else:
        return result


def masked_fill(x: Float[Tensor, "..."], mask: Float[Tensor, "..."], value: float) -> Float[Tensor, "..."]:
    """
    Perform masked fill operation on a Tensor.
    """
    if isinstance(x, DTensor):
        assert isinstance(mask, DTensor), "mask must be a DTensor"
        x_local = x.to_local()
        mask_local = mask.to_local()
        x_local[mask_local] = value
        return DTensor.from_local(x_local, device_mesh=x.device_mesh, placements=x.placements)
    else:
        assert isinstance(mask, Tensor), "mask must be a Tensor"
        x[mask] = value
        return x


def slice_fill(
    x: Float[Tensor, "..."],
    slice_tuple: Union[int, slice, Tuple[Union[int, slice, None], ...]],
    value: Union[float, int, Tensor],
) -> Float[Tensor, "..."]:
    """
    Fill a slice of a Tensor or DTensor with a value.
    """
    if isinstance(x, DTensor):
        x_local = x.to_local()
        x_local[slice_tuple] = value
        return DTensor.from_local(x_local, device_mesh=x.device_mesh, placements=x.placements)
    else:
        x[slice_tuple] = value
        return x
