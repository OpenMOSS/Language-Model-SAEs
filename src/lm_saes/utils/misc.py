import os

import torch
import torch.distributed as dist

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