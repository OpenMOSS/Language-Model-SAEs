import functools
import operator
from typing import Any, Callable, Optional, cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement

from lm_saes.utils.misc import is_primary_rank


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


def all_gather_list(
    data: list[Any],
    group: Optional[torch.distributed.ProcessGroup] = None,
    flatten: bool = True,
) -> list[Any] | list[list[Any]]:
    """
    All-gather a list across all ranks in a process group.

    Args:
        data: List to all-gather from the current rank.
        group: Optional process group for communication. If None, uses the default group.
        flatten: If True, returns a single flattened list containing all items from all ranks.
                 If False, returns a list of lists, one per rank.

    Returns:
        If flatten=True: A single list containing all items from all ranks, concatenated in rank order.
        If flatten=False: A list of lists, where each element is the list from the corresponding rank.
        If distributed is not initialized or world size is 1, returns the input data unchanged.
    """
    if not dist.is_initialized() or dist.get_world_size(group=group) == 1:
        return data

    world_size = dist.get_world_size(group=group)

    # Gather all lists using all_gather_object
    gathered_lists = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_lists, data, group=group)
    gathered_lists = cast(list[list[Any]], gathered_lists)
    if flatten:
        # Flatten the list of lists into a single list
        return [item for rank_list in gathered_lists for item in rank_list]
    return gathered_lists


def broadcast_object(
    object: Any | None,
    group_src: int = 0,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> Any:
    """
    Broadcast an object across all ranks in a process group.

    Args:
        object: Object to broadcast from the source rank. Can be any picklable object.
        group_src: Source rank for the broadcast operation (global rank ID within the group).
        group: Optional process group for communication. If None, uses the default group.

    Returns:
        The broadcasted object. All ranks will receive the object from the source rank.
    """
    object_list = [object]
    dist.broadcast_object_list(object_list, group_src=group_src, group=group)
    return object_list[0]


def mesh_dim_size(device_mesh: DeviceMesh | None, mesh_dim: str) -> int:
    if device_mesh is None:
        return 1
    assert device_mesh is not None
    assert device_mesh.mesh_dim_names is not None, "Device mesh does not have mesh dimension names"
    return device_mesh.get_group(mesh_dim).size() if mesh_dim in device_mesh.mesh_dim_names else 1


def mesh_dim_rank(device_mesh: DeviceMesh | None, mesh_dim: str) -> int:
    if device_mesh is None:
        return 0
    assert device_mesh is not None
    assert device_mesh.mesh_dim_names is not None, "Device mesh does not have mesh dimension names"
    return device_mesh.get_group(mesh_dim).rank() if mesh_dim in device_mesh.mesh_dim_names else 0


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


# Cache for process groups to avoid recreating them
_process_group_cache: dict[tuple[int, tuple[str, ...]], dist.ProcessGroup] = {}


def get_mesh_ranks(device_mesh: DeviceMesh, mesh_dim: list[str]) -> list[int]:
    """Get global ranks for processes spanning specified mesh dimensions.

    Fixes dimensions not in mesh_dim to current coordinate, iterates over mesh_dim dimensions.

    Args:
        device_mesh: Device mesh to query.
        mesh_dim: List of dimension names to span.

    Returns:
        List of global ranks participating in the group.

    Example:
        For a 2×4 mesh [[0,1,2,3], [4,5,6,7]] with dims ["dp", "tp"]:
        - coord (0,1), mesh_dim=["dp"] -> [1, 5]
        - coord (0,1), mesh_dim=["tp"] -> [0, 1, 2, 3]
        - coord (0,1), mesh_dim=["dp","tp"] -> [0, 1, 2, 3, 4, 5, 6, 7]
    """
    assert device_mesh.mesh_dim_names is not None, "Device mesh does not have mesh dimension names"
    assert all(dim in device_mesh.mesh_dim_names for dim in mesh_dim), (
        f"Mesh dimensions {mesh_dim} not found in device mesh"
    )

    coord = device_mesh.get_coordinate()
    assert coord is not None, "Device mesh does not have coordinate"

    mesh_dim_indices = [device_mesh.mesh_dim_names.index(dim) for dim in mesh_dim]

    # Fix non-mesh_dim dimensions to current coord, span mesh_dim dimensions
    indices = [slice(None) if i in mesh_dim_indices else coord[i] for i in range(len(coord))]

    ranks_tensor = device_mesh.mesh[tuple(indices)]
    print(f"rank: {dist.get_rank()}, ranks: {ranks_tensor.flatten().tolist()}")
    return ranks_tensor.flatten().tolist()


def get_process_group(
    device_mesh: DeviceMesh, mesh_dim: str | list[str] | None = None
) -> torch.distributed.ProcessGroup:
    """Get process group for specified mesh dimensions.

    Args:
        device_mesh: Device mesh to query.
        mesh_dim: Dimension name(s). If None, uses all dimensions.
                  If str, uses device_mesh.get_group().
                  If list, creates new group via get_mesh_ranks() and caches the result.

    Returns:
        Process group for the specified dimensions.
    """
    assert device_mesh.mesh_dim_names is not None, "Device mesh does not have mesh dimension names"
    if isinstance(mesh_dim, str):
        assert mesh_dim in device_mesh.mesh_dim_names, f"Mesh dimension {mesh_dim} not found in device mesh"
        return device_mesh.get_group(mesh_dim)
    else:
        mesh_dim = mesh_dim if mesh_dim is not None else list(device_mesh.mesh_dim_names)

        # Use cache to avoid recreating process groups
        cache_key = (id(device_mesh), tuple(sorted(mesh_dim)))
        if cache_key in _process_group_cache:
            return _process_group_cache[cache_key]

        ranks = get_mesh_ranks(device_mesh, mesh_dim)
        pg = cast(dist.ProcessGroup, dist.new_group(ranks=ranks))
        _process_group_cache[cache_key] = pg
        return pg


def get_primary_process_group(device_mesh: DeviceMesh, mesh_dim_name: str) -> dist.ProcessGroup:
    """Get the process group that divides the device mesh along the given dimension. This should be the complement of `get_process_group` on the given dimension."""

    assert device_mesh.mesh_dim_names is not None, "Device mesh does not have mesh dimension names"
    mesh_dim_names = [name for name in device_mesh.mesh_dim_names if name != mesh_dim_name]
    return get_process_group(device_mesh, mesh_dim_names)


def execute_and_broadcast(
    func: Callable[..., Any],
    device_mesh: DeviceMesh | None,
    mesh_dim_name: str = "sweep",
    **kwargs,
) -> Any:
    """Execute a function in the primary rank and broadcast the result to all ranks. If the device mesh is None, the function simply executes the function and returns the result."""
    if device_mesh is None:
        return func(**kwargs)
    group = get_primary_process_group(device_mesh, mesh_dim_name)
    return broadcast_object(
        func(**kwargs) if is_primary_rank(device_mesh, dim_name=mesh_dim_name) else None, group_src=0, group=group
    )
