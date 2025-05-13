import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard


def placements_from_dim_map(dim_map: dict[str, int], device_mesh: DeviceMesh) -> list[Placement]:
    if device_mesh.mesh_dim_names is None:
        raise ValueError("Device mesh does not have mesh dimension names.")
    return [Shard(dim_map[dim_name]) if dim_name in dim_map else Replicate() for dim_name in device_mesh.mesh_dim_names]


def distribute_tensor_on_dim(tensor: torch.Tensor, device_mesh: DeviceMesh, dim_map: dict[str, int]) -> DTensor:
    """Distribute a tensor on a specific dimension of the device mesh.

    Args:
        tensor: The tensor to distribute.
        device_mesh: The device mesh to distribute the tensor on.
        dim_map: The dimension map of the device mesh."""

    if device_mesh.mesh_dim_names is None:
        raise ValueError("Device mesh does not have mesh dimension names.")

    placements = [
        Shard(dim_map[dim_name]) if dim_name in dim_map else Replicate() for dim_name in device_mesh.mesh_dim_names
    ]
    return distribute_tensor(tensor, device_mesh, placements)


def local_slices_from_dim_map(
    shape: tuple[int, ...], dim_map: dict[str, int], device_mesh: DeviceMesh
) -> tuple[slice, ...]:
    """Get the local slices from the dimension map."""

    # Check if the dim map can be safely reversed
    if len(set(dim_map.values())) != len(dim_map):
        raise ValueError("Dimension map values must be unique to reverse.")

    reverse_dim_map = {v: k for k, v in dim_map.items()}

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
