import os
import warnings
from typing import Any, Iterable, Optional, cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_reduce

from lm_saes.utils.distributed import DimMap

from .logging import get_distributed_logger

logger = get_distributed_logger("utils.misc")


def is_master() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def is_primary_rank(device_mesh: DeviceMesh | None, dim_name: str = "sweep") -> bool:
    if device_mesh is None:
        return True
    coord = device_mesh.get_coordinate()
    mesh_dim_names = device_mesh.mesh_dim_names
    if coord is None or mesh_dim_names is None:
        return False
    coord = [c for i, c in enumerate(coord) if dim_name not in mesh_dim_names or i != mesh_dim_names.index(dim_name)]
    return all(c == 0 for c in coord)


def print_once(
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n",
) -> None:
    """Print only from the master process in distributed training.

    Note: This function is deprecated. Use logger.info() instead.
    """
    if is_master():
        message = sep.join(str(v) for v in values) if sep else " ".join(str(v) for v in values)
        logger.info(message)


def check_file_path_unused(file_path):
    # Check if the file path is None
    if file_path is None:
        logger.error("File path is empty.")
        exit()

    # Check if the file already exists
    if os.path.exists(file_path):
        logger.error(f"File {file_path} already exists. Please choose a different file path.")
        exit()


str_dtype_map = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float": torch.float,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "int": torch.int,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.bool": torch.bool,
    "torch.bfloat16": torch.bfloat16,
    "torch.float": torch.float,
    "torch.int": torch.int,
}


def convert_str_to_torch_dtype(str_dtype: str) -> torch.dtype:
    if str_dtype in str_dtype_map:
        return str_dtype_map[str_dtype]
    else:
        raise ValueError(f"Unsupported data type: {str_dtype}. Supported data types: {list(str_dtype_map.keys())}.")


def convert_torch_dtype_to_str(dtype: torch.dtype) -> str:
    dtype_str_map = {v: k for k, v in str_dtype_map.items()}
    if dtype in dtype_str_map:
        return dtype_str_map[dtype]
    else:
        raise ValueError(f"Unsupported data type: {dtype}. Supported data types: {list(dtype_str_map.values())}.")


def all_gather_tensor(tensor, group: Optional[torch.distributed.ProcessGroup] = None):
    world_size = dist.get_world_size(group=group)
    tensor_meta = {"shape": tensor.shape, "dtype": tensor.dtype}
    meta_list: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(meta_list, tensor_meta, group=group)
    gathered_tensors = [
        torch.empty(rank_meta["shape"], dtype=rank_meta["dtype"], device=tensor.device)
        for rank_meta in cast(list[dict[str, Any]], meta_list)
    ]
    dist.all_gather(gathered_tensors, tensor, group=group)
    return gathered_tensors


def get_tensor_from_specific_rank(tensor, src=0):
    dist.broadcast(tensor, src=src)
    return tensor


def all_reduce_tensor(tensor, aggregate: str, group: Optional[torch.distributed.ProcessGroup] = None):
    _OP_MAP = {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.SUM,  # Use SUM for mean, but will need to divide by world size
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
        "product": dist.ReduceOp.PRODUCT,
    }
    assert aggregate in _OP_MAP, f"Unsupported aggregate: {aggregate}. Supported aggregates: {list(_OP_MAP.keys())}."

    # gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    tensor = all_reduce(tensor, op=_OP_MAP[aggregate])
    assert tensor is not None, "All reduce failed"
    if aggregate == "mean":
        tensor = tensor / dist.get_world_size(group=group)
    return tensor


def assert_tensor_consistency(tensor):
    flat_tensor = tensor.flatten()

    local_checksum = flat_tensor.sum().item()
    checksum_tensor = torch.tensor(local_checksum).to(tensor.device)

    dist.all_reduce(checksum_tensor, op=dist.ReduceOp.SUM)

    world_size = dist.get_world_size()
    expected_checksum = local_checksum * world_size

    # Step 5: Assert that the checksums match across all ranks
    assert checksum_tensor.item() == expected_checksum, "Inconsistent tensor data across ranks. Checksum mismatch."


def calculate_activation_norm(
    activation_stream: Iterable[dict[str, torch.Tensor]],
    hook_points: list[str],
    batch_num: int = 8,
    device_mesh: DeviceMesh | None = None,
) -> dict[str, float]:
    activation_norm = {}
    stream_iter = iter(activation_stream)
    if device_mesh is not None and "head" in cast(tuple[str, ...], device_mesh.mesh_dim_names):
        hook_points = hook_points[DimMap({"head": 0}).local_slices((len(hook_points),), device_mesh)[0]]
    assert len(hook_points) > 0, "No hook points provided"
    while batch_num > 0:
        try:
            batch = next(stream_iter)
        except StopIteration:
            warnings.warn(f"Activation stream ended prematurely. {batch_num} batches not processed.")
            break
        for key in hook_points:
            if key not in activation_norm:
                activation_norm[key] = batch[key].norm(p=2, dim=1)
            else:
                activation_norm[key] = torch.cat((activation_norm[key], batch[key].norm(p=2, dim=1)), dim=0)
        batch_num -= 1
    for key in activation_norm:
        activation_norm[key] = activation_norm[key].mean().item()
    if device_mesh is not None and "head" in cast(tuple[str, ...], device_mesh.mesh_dim_names):
        object_list = [None] * device_mesh.get_group("head").size()
        dist.all_gather_object(object_list, activation_norm, group=device_mesh.get_group("head"))
        activation_norm = {k: v for d in cast(list[dict[str, float]], object_list) for k, v in d.items()}
    return activation_norm


def pad_and_truncate_tokens(
    tokens: torch.Tensor,
    seq_len: int,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Pad tokens to desired sequence length.

    Args:
        tokens: Input tokens tensor or list of token tensors to pad
        seq_len: Desired sequence length after padding
        pad_token_id: Token ID to use for padding (default: 0)

    Returns:
        torch.Tensor: Padded token tensor with shape (batch_size, seq_len)
    """
    if tokens.size(-1) > seq_len:
        return tokens[..., :seq_len]

    pad_len = seq_len - tokens.size(-1)

    padding = torch.full(
        (*tokens.shape[:-1], pad_len),
        pad_token_id,
        dtype=torch.long,
        device=tokens.device,
    )
    return torch.cat([tokens, padding], dim=-1)


def get_slice_length(s: slice, length: int):
    start, stop, step = s.indices(length)
    length = (stop - start + step - 1) // step
    return length


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
            expected_device = torch.device(f"cuda:{dist.get_rank(group=group)}")
            v = v.to(expected_device)

            # First, gather tensor metadata (shape, dtype) from all ranks
            tensor_meta = {"shape": v.shape, "dtype": v.dtype}
            meta_list: list[dict[str, Any] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(meta_list, tensor_meta, group=group)

            # Create output tensors with correct shapes for each rank
            output = [
                torch.empty(rank_meta["shape"], dtype=rank_meta["dtype"], device=expected_device)
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


def get_device_mesh_dim_size(device_mesh: DeviceMesh | None, mesh_dim: str) -> int:
    if device_mesh is None:
        return 1
    assert device_mesh is not None
    assert device_mesh.mesh_dim_names is not None, "Device mesh does not have mesh dimension names"
    return device_mesh.get_group(mesh_dim).size() if mesh_dim in device_mesh.mesh_dim_names else 1
