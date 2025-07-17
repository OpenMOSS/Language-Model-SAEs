from typing import Any, Optional, Union

import torch
from jaxtyping import Float
from torch._tensor import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from lm_saes.utils.timer import timer


class DimMap:
    """A class representing the mapping between tensor dimensions and device mesh dimensions.

    This class replaces the previous dictionary-based implementation of dimension maps
    with a more structured approach that provides helper methods for common operations.
    """

    def __init__(self, dim_map: dict[str, int]):
        """Initialize a DimMap with a dictionary mapping mesh dimension names to tensor dimensions.

        Args:
            dim_map: A dictionary mapping mesh dimension names (e.g., "head", "model")
                    to shardable tensor dimensions (e.g., 0, 1, 2).
        """
        self._dim_map = dim_map.copy()

    def __getitem__(self, key: str) -> int:
        """Get the tensor dimension for a given mesh dimension name."""
        return self._dim_map[key]

    def __contains__(self, key: str) -> bool:
        """Check if a mesh dimension name is in the dimension map."""
        return key in self._dim_map

    def __iter__(self):
        """Iterate over the mesh dimension names in the dimension map."""
        return iter(self._dim_map)

    def items(self):
        """Return the items of the dimension map."""
        return self._dim_map.items()

    def values(self):
        """Return the tensor dimensions in the dimension map."""
        return self._dim_map.values()

    def keys(self):
        """Return the mesh dimension names in the dimension map."""
        return self._dim_map.keys()

    def to_dict(self) -> dict[str, int]:
        """Return the dimension map as a dictionary."""
        return self._dim_map.copy()

    @classmethod
    def from_dict(cls, dim_map: dict[str, int]) -> "DimMap":
        """Create a DimMap from a dictionary."""
        return cls(dim_map)

    def reverse(self) -> dict[int, str]:
        """Return a reversed dimension map mapping tensor dimensions to mesh dimension names."""
        # Validate that the dimension map values are unique
        if len(set(self._dim_map.values())) != len(self._dim_map):
            raise ValueError("Dimension map values must be unique to reverse.")

        return {v: k for k, v in self._dim_map.items()}

    def placements(self, device_mesh: DeviceMesh) -> list[Placement]:
        """Get the placements for a tensor based on the dimension map and device mesh.

        Args:
            device_mesh: The device mesh to get placements for.

        Returns:
            A list of placements for the tensor.

        Raises:
            ValueError: If the device mesh does not have mesh dimension names.
        """
        if device_mesh.mesh_dim_names is None:
            raise ValueError("Device mesh does not have mesh dimension names.")

        return [
            Shard(self._dim_map[dim_name]) if dim_name in self._dim_map else Replicate()
            for dim_name in device_mesh.mesh_dim_names
        ]

    def local_slices(self, shape: tuple[int, ...], device_mesh: DeviceMesh) -> tuple[slice, ...]:
        """Get the local slices for a tensor based on the dimension map and device mesh.

        Args:
            shape: The shape of the tensor.
            device_mesh: The device mesh to get local slices for.

        Returns:
            A tuple of slices for the tensor.

        Raises:
            ValueError: If the dimension map values are not unique.
        """
        # Check if the dim map can be safely reversed
        if len(set(self._dim_map.values())) != len(self._dim_map):
            raise ValueError("Dimension map values must be unique to reverse.")

        reverse_dim_map = self.reverse()

        def get_slice(mesh_dim_name: str, dim_size: int) -> slice:
            assert device_mesh.mesh_dim_names is not None

            if mesh_dim_name not in device_mesh.mesh_dim_names:
                return slice(None)
            else:
                assert dim_size % device_mesh.get_group(mesh_dim_name).size() == 0
                step = dim_size // device_mesh.get_group(mesh_dim_name).size()
                local_rank = device_mesh.get_local_rank(mesh_dim_name)
                return slice(local_rank * step, (local_rank + 1) * step)

        return tuple(
            slice(None) if i not in reverse_dim_map else get_slice(reverse_dim_map[i], dim_size)
            for i, dim_size in enumerate(shape)
        )

    def distribute(self, tensor: torch.Tensor, device_mesh: DeviceMesh) -> DTensor:
        """Distribute a tensor on the device mesh according to the dimension map.

        Args:
            tensor: The tensor to distribute.
            device_mesh: The device mesh to distribute the tensor on.

        Returns:
            A distributed tensor.
        """
        placements = self.placements(device_mesh)
        return distribute_tensor(tensor, device_mesh, placements)

    def redistribute(self, tensor: DTensor) -> DTensor:
        """Redistribute a distributed tensor on the device mesh according to the dimension map.

        Args:
            tensor: The distributed tensor to redistribute.
        """
        placements = self.placements(tensor.device_mesh)
        return tensor.redistribute(placements=placements)

    def __or__(self, other: "DimMap") -> "DimMap":
        """Merge this DimMap with another DimMap or dictionary."""
        return DimMap(self.to_dict() | other.to_dict())

@torch.no_grad()
def distributed_batch_kthvalue_clt(
    x: Float[DTensor, "batch n_layers d_sae"],
    k: int,
    device_mesh: Optional[DeviceMesh] = None,  ## TODO: type
    mesh_dim_name: str = "model",
) -> tuple[DTensor, Optional[DTensor]]:
    """
    Perform distributed batch kthvalue operation on a DTensor sharded along the specified dimension.
    """
    if not isinstance(x, DTensor):
        # For regular tensors, use standard kthvalue
        raise ValueError("x must be a DTensor")
    
    if x.dim() != 3:
        raise ValueError("x must be a 3D DTensor")
    
    if device_mesh is None:
        raise ValueError("device_mesh must be provided when x is a DTensor")
    
    # Get local tensor and placements
    local_tensor = x.to_local()
    placements = x.placements
    
    # Find the placement indices for the specified dimensions
    mesh_dim_idx = None
    if device_mesh.mesh_dim_names is not None:
        try:
            mesh_dim_idx = device_mesh.mesh_dim_names.index(mesh_dim_name)
        except ValueError:
            raise ValueError(f"Mesh dimension '{mesh_dim_name}' not found in device mesh")
    
    # Check if the tensor is sharded along the specified dimensions
    if mesh_dim_idx is None or not isinstance(placements[mesh_dim_idx], Shard):
        # If not sharded along the specified dimensions, use standard kthvalue
        raise ValueError("x must be sharded along the specified dimension")
    
    shard_dim: Any = placements[mesh_dim_idx].dim  # type: ignore
    if shard_dim != 1 and shard_dim != 2:  # also consider negative dims
        # If sharded along a different dimension, use standard kthvalue
        raise ValueError("x must be sharded along the specified dimension")
    
    # Get the groups for the mesh dimensions
    group = device_mesh.get_group(mesh_dim_name)
    world_size = group.size()
    
    # approximate top-k based on divide and conquer
    # hardcoded parameters
    with timer.time("kthvalue"):
        divide_batch = 65536
        bucket_size = 8192
        batch_k = 16
        # divide_batch = 131072
        # bucket_size = 8192
        # batch_k = 16
        batch_size, n_layers, d_sae = local_tensor.shape
        # local_tensor = local_tensor[local_tensor.gt(0)]  ## TODO

        # local_tensor = local_tensor.reshape((divide_batch, bucket_size))
        local_tensor = local_tensor.reshape((bucket_size, divide_batch)).transpose(0, 1)
        # Top-t in each row
        topkbs, _ = torch.topk(local_tensor, k=batch_k, dim=1)
        # Flatten to 1D
        topkbs = topkbs.flatten()
        # kthvalue among these elements
        candidates, _ = torch.kthvalue(topkbs, k=topkbs.shape[-1] - k * batch_size * n_layers // world_size + 1, dim=-1)
        candidates = candidates.unsqueeze(0)


    # rigorous top-k
    # with timer.time("kthvalue"):
    #     # Compute kthvalue on local tensor
    #     batch_size, n_layers, d_sae = local_tensor.shape
    #     local_tensor = local_tensor.flatten()
    #     # local_tensor = local_tensor[local_tensor.gt(0)]

    #     candidates, _ = torch.kthvalue(local_tensor, k=local_tensor.shape[-1] - k * batch_size * n_layers // world_size + 1, dim=-1)
    #     candidates = candidates.unsqueeze(0)

    # Gather all local kthvalue results
    gathered_values = [torch.empty_like(candidates) for _ in range(world_size)]
    # print(candidates)
    # print(gathered_values[0])
    torch.distributed.all_gather(gathered_values, candidates, group=group)
    
    # Concatenate all gathered results along the specified dimension
    all_values = torch.cat(gathered_values, dim=-1)
    
    # Compute global kthvalue on the concatenated results
    # result_values, _ = torch.kthvalue(all_values, k=all_values.shape[-1] - k * batch_size * n_layers + 1, dim=-1)
    result_values = all_values.mean()
    # place the result_values back to DTensor (Replicate)
    result_values = DTensor.from_local(result_values, device_mesh=device_mesh, placements=[Replicate() for _ in range(len(placements))])
    return result_values, None

@torch.no_grad()
def distributed_batch_kthvalue_clt_binary_search(
    x: Float[DTensor, "batch n_layers d_sae"],
    k_range: tuple[int, int],
    device_mesh: Optional[DeviceMesh] = None,  
    mesh_dim_name: str = "model",
) -> float:
    """
    Perform distributed batch kthvalue operation on a DTensor using binary search.
    
    Args:
        x: Input tensor of shape (batch, n_layers, d_sae)
        k_range: Acceptable range for the number of elements above threshold (lower_bound, upper_bound)
        device_mesh: Device mesh for distributed training
        mesh_dim_name: Name of the mesh dimension to shard along
        
    Returns:
        Threshold value that gives acceptable k elements within the specified range
    """
    if not isinstance(x, DTensor):
        raise ValueError("x must be a DTensor")

    if x.dim() != 3:
        raise ValueError("x must be a 3D DTensor")
    
    if device_mesh is None:
        raise ValueError("device_mesh must be provided when x is a DTensor")
    
    # Get local tensor and placements
    local_tensor = x.to_local()
    placements = x.placements
    
    # Find the placement indices for the specified dimensions
    mesh_dim_idx = None
    if device_mesh.mesh_dim_names is not None:
        try:
            mesh_dim_idx = device_mesh.mesh_dim_names.index(mesh_dim_name)
        except ValueError:
            raise ValueError(f"Mesh dimension '{mesh_dim_name}' not found in device mesh")
    
    # Check if the tensor is sharded along the specified dimensions
    if mesh_dim_idx is None or not isinstance(placements[mesh_dim_idx], Shard):
        raise ValueError("x must be sharded along the specified dimension")
    
    shard_dim: Any = placements[mesh_dim_idx].dim  # type: ignore
    if shard_dim != 1 and shard_dim != 2:  # also consider negative dims
        raise ValueError("x must be sharded along the specified dimension")
    
    # Get the groups for the mesh dimensions
    group = device_mesh.get_group(mesh_dim_name)
    world_size = group.size()

    batch_size, n_layers, d_sae = local_tensor.shape
    k_range_overall = (k_range[0] * batch_size * n_layers, k_range[1] * batch_size * n_layers)
    
    with timer.time("kthvalue"):
        # Flatten the local tensor for easier processing
        local_tensor_flat = local_tensor.flatten()
        
        # Binary search parameters
        lower_bound, upper_bound = k_range_overall
        search_low = 0.0
        search_high = 10.0
        max_iterations = 50
        tolerance = 1e-6
        
        # First check if 10 is a reasonable upper bound
        count_above_high = (local_tensor_flat > search_high).sum()
        torch.distributed.all_reduce(count_above_high, group=group)
        
        if count_above_high > upper_bound:
            # Need to increase search space
            search_high *= 2
            count_above_high = (local_tensor_flat > search_high).sum()
            torch.distributed.all_reduce(count_above_high, group=group)
        
        # Check if we can directly use 0 as threshold
        count_above_0 = (local_tensor_flat > 0).sum()
        torch.distributed.all_reduce(count_above_0, group=group)
        
        if count_above_0 <= upper_bound:
            # Directly use 0 as threshold
            threshold = torch.tensor(0.0, device=local_tensor.device, dtype=local_tensor.dtype)
        else:
            # Binary search for the optimal threshold
            threshold = None
            for iteration in range(max_iterations):
                threshold = (search_low + search_high) / 2
                
                # Count elements above threshold on this rank
                count_above_threshold = (local_tensor_flat > threshold).sum()
                
                # All-reduce to get total count across all ranks
                torch.distributed.all_reduce(count_above_threshold, group=group)
                
                if lower_bound <= count_above_threshold <= upper_bound:
                    # Found acceptable threshold
                    break
                elif count_above_threshold > upper_bound:
                    # Too many elements above threshold, increase threshold
                    search_low = threshold
                else:
                    # Too few elements above threshold, decrease threshold
                    search_high = threshold
                
                # Check for convergence
                if search_high - search_low < tolerance:
                    break
            
            # If we didn't find a threshold in range, use the best approximation
            if threshold is None:
                threshold = search_low  # Use the lower bound as a fallback
    
    # Ensure threshold is a scalar value
    if isinstance(threshold, torch.Tensor):
        threshold_value = threshold.item()
    else:
        threshold_value = threshold
    
    return threshold_value