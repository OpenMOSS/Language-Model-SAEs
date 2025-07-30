from typing import Any, Optional, Union

import torch
from jaxtyping import Float
from torch._tensor import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from typing import Any, Union, Optional, Tuple
from jaxtyping import Float


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

def distributed_topk(
    x: Float[DTensor, "batch n_layers d_sae"],
    k: int,
    device_mesh: DeviceMesh,
    dim: Union[int, Tuple[int, ...]] = -1,
    tolerance: int = 1,
    max_iterations: int = 50,
    mesh_dim_name: str = "model",
) -> DTensor:
    """
    Perform distributed batch kthvalue operation on a DTensor using binary search.
    
    Args:
        x: Input tensor of shape (batch, n_layers, d_sae)
        k: Target number of top elements to keep
        k_range: Acceptable range for the number of elements above threshold (lower_bound, upper_bound)
        device_mesh: Device mesh for distributed training
        mesh_dim_name: Name of the mesh dimension to shard along
        
    Returns:
        Tuple of (threshold, None) where threshold is a scalar value that gives acceptable k elements
    """
    if not isinstance(x, DTensor) or device_mesh is None:
        raise ValueError("x must be a DTensor and device_mesh must be provided")
    
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
        search_high_val = torch.full(constant_dim_size, local_tensor.max(), device=local_tensor.device)
        
        local_tensor_flat = local_tensor.flatten(start_dim=len(constant_dims))

        group = device_mesh.get_group(mesh_dim_name)
        
        
        for _ in range(max_iterations):
            threshold = (search_low_val + search_high_val) / 2
            torch.distributed.all_reduce(threshold, group=group, op=torch.distributed.ReduceOp.AVG)
            
            count_above_threshold = (local_tensor_flat > threshold.unsqueeze(-1)).sum(-1)
            # All-reduce to get total count across all ranks
            torch.distributed.all_reduce(count_above_threshold, group=group)
            
            if (
                (k_lower_bound <= count_above_threshold) * (count_above_threshold <= k_upper_bound)
            ).all():
                break

            to_increase = count_above_threshold > k_upper_bound
            to_decrease = count_above_threshold < k_lower_bound

            if to_increase.any():
                search_low_val = torch.where(to_increase, threshold, search_low_val)
            if to_decrease.any():
                search_high_val = torch.where(to_decrease, threshold, search_high_val)
            
            # Check for convergence
            if (search_high_val - search_low_val < 1e-6).all():
                break
        
        while threshold.ndim < local_tensor.ndim:
            threshold = threshold[..., None]
    
    local_tensor = local_tensor * local_tensor.ge(threshold)
    
    result = DTensor.from_local(
        local_tensor,
        device_mesh=device_mesh,
        placements=placements,
    )
    
    return result