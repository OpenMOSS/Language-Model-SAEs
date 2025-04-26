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
