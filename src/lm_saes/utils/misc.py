import os
import warnings
from typing import Iterable

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce
from transformers import PreTrainedTokenizerBase


def is_master() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def print_once(
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n",
) -> None:
    if is_master():
        print(*values, sep=sep, end=end)


def check_file_path_unused(file_path):
    # Check if the file path is None
    if file_path is None:
        print("Error: File path is empty.")
        exit()

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"Error: File {file_path} already exists. Please choose a different file path.")
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


def gather_tensors_from_specific_rank(tensor, dst=0):
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)] if dist.get_rank() == dst else None
    dist.gather(tensor, gather_list=gathered_tensors, dst=dst)
    return gathered_tensors if dist.get_rank() == dst else None


def get_tensor_from_specific_rank(tensor, src=0):
    dist.broadcast(tensor, src=src)
    return tensor


def all_reduce_tensor(tensor, aggregate="none"):
    _OP_MAP = {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.SUM,  # Use SUM for mean, but will need to divide by world size
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
        "product": dist.ReduceOp.PRODUCT,
    }

    # gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    tensor = all_reduce(tensor, op=_OP_MAP[aggregate])
    assert tensor is not None, "All reduce failed"
    if aggregate == "mean":
        tensor = tensor / dist.get_world_size()
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
    activation_stream: Iterable[dict[str, torch.Tensor]], hook_points: list[str], batch_num: int = 8
) -> dict[str, float]:
    activation_norm = {}
    stream_iter = iter(activation_stream)
    hook_points = list(set(hook_points))
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
    return activation_norm


def get_modality_tokens(tokenizer: PreTrainedTokenizerBase, model_name: str) -> dict[str, torch.Tensor]:
    modality_tokens = {}
    if model_name == "facebook/chameleon-7b":
        for token_name, token_id in tokenizer.get_vocab().items():
            if token_name.startswith("IMGIMG"):
                modality_tokens["image"] = (
                    [token_id] if "image" not in modality_tokens else modality_tokens["image"] + [token_id]
                )
            else:
                modality_tokens["text"] = (
                    [token_id] if "text" not in modality_tokens else modality_tokens["text"] + [token_id]
                )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    for modality in modality_tokens:
        modality_tokens[modality] = torch.tensor(modality_tokens[modality], dtype=torch.long)
    return modality_tokens
