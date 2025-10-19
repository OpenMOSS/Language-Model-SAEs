from typing import cast

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard


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

    @classmethod
    def from_placements(cls, placements: tuple[Placement, ...], device_mesh: DeviceMesh) -> "DimMap":
        """Create a DimMap from a list of placements and a device mesh."""
        assert device_mesh.mesh_dim_names is not None, "Device mesh must have mesh dimension names"
        assert all(isinstance(placement, Shard) or isinstance(placement, Replicate) for placement in placements), (
            "All placements must be Shard or Replicate"
        )
        return cls(
            {
                dim_name: cast(Shard, placements[i]).dim
                for i, dim_name in enumerate(device_mesh.mesh_dim_names)
                if isinstance(placements[i], Shard)
            }
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DimMap):
            return False
        """Check if two DimMaps are equal."""
        return self.to_dict() == other.to_dict()

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, DimMap):
            return True
        """Check if two DimMaps are not equal."""
        return self.to_dict() != other.to_dict()
