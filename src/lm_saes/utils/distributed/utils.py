import functools
import operator
from typing import Any, Optional, cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement


def all_gather_dict(
    data: dict[str, Any],
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> list[dict[str, Any]]:
    """
    All-gather a dictionary across all ranks. For each key, if the value is a tensor, use torch.distributed.all_gather
    (supporting uneven sized tensors); otherwise, use all_gather_object. Returns a list of dicts, one per rank.

    Args:
        data: Dictionary to all-gather. Tensor values should be on the correct device.
        group: Optional process group for communication.

    Returns:
        List of dictionaries, one per rank, with gathered values.
    """
    world_size = dist.get_world_size(group=group)
    keys = list(data.keys())
    gathered_dicts: list[dict[str, Any]] = [dict() for _ in range(world_size)]

    # Gather each key separately
    for k in keys:
        v = data[k]
        if isinstance(v, torch.Tensor):
            # First, gather tensor metadata (shape, dtype) from all ranks
            tensor_meta = {"shape": v.shape, "dtype": v.dtype}
            meta_list: list[dict[str, Any] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(meta_list, tensor_meta, group=group)

            # Create output tensors with correct shapes for each rank
            output = [
                torch.empty(rank_meta["shape"], dtype=rank_meta["dtype"], device=v.device)
                for rank_meta in cast(list[dict[str, Any]], meta_list)
            ]
            # Now perform all_gather with correctly sized tensors
            dist.all_gather(output, v, group=group)
            for i, t in enumerate(output):
                gathered_dicts[i][k] = t
        else:
            # Use all_gather_object for non-tensor values
            object_list = [None for _ in range(world_size)]
            dist.all_gather_object(object_list, v, group=group)
            for i, obj in enumerate(object_list):
                gathered_dicts[i][k] = obj
    return gathered_dicts


def mesh_dim_size(device_mesh: DeviceMesh | None, mesh_dim: str) -> int:
    if device_mesh is None:
        return 1
    assert device_mesh is not None
    assert device_mesh.mesh_dim_names is not None, "Device mesh does not have mesh dimension names"
    return device_mesh.get_group(mesh_dim).size() if mesh_dim in device_mesh.mesh_dim_names else 1


def mesh_rank(device_mesh: DeviceMesh | None) -> int:
    """Get the rank of the current process in the device mesh. Computed through the coordinate of the device mesh.

    Args:
        device_mesh: Device mesh to get the rank from.

    Returns:
        Rank of the current process in the device mesh.
    """
    if device_mesh is None:
        return 0
    coord = device_mesh.get_coordinate()
    shape = device_mesh.shape
    assert coord is not None, "Device mesh does not have coordinate"
    return sum(coord[i] * functools.reduce(operator.mul, shape[i + 1 :], 1) for i in range(len(coord)))


def replace_placements(
    placements: tuple[Placement, ...], device_mesh: DeviceMesh, mesh_dim: str, new_placement: Placement
) -> tuple[Placement, ...]:
    assert device_mesh.mesh_dim_names is not None, "Device mesh does not have mesh dimension names"
    if mesh_dim in device_mesh.mesh_dim_names:
        return tuple(
            new_placement if i == device_mesh.mesh_dim_names.index(mesh_dim) else p for i, p in enumerate(placements)
        )
    return placements
