import os
import warnings
from typing import Iterable, Union, cast

import torch
import torch.distributed as dist
from jaxtyping import Float
from torch.distributed.device_mesh import DeviceMesh

from lm_saes.utils.distributed import DimMap
from lm_saes.utils.distributed.ops import item

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
    activation_stream: Iterable[
        dict[
            str,
            Union[
                Float[torch.Tensor, "batch seq_len d_model"],
                Float[torch.Tensor, "batch d_model"],
            ],
        ]
    ],
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
                activation_norm[key] = batch[key].norm(p=2, dim=-1)
            else:
                activation_norm[key] = torch.cat((activation_norm[key], batch[key].norm(p=2, dim=-1)), dim=0)
        batch_num -= 1
    for key in activation_norm:
        activation_norm[key] = item(activation_norm[key].mean())
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
