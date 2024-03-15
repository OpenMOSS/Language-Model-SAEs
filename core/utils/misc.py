import torch
import torch.distributed as dist

def print_once(
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n",
) -> None:
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*values, sep=sep, end=end)
    else:
        print(*values, sep=sep, end=end)
