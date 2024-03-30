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

def norm_ratio(a, b):
    a_norm = torch.norm(a, 2, dim=0).mean()
    b_norm = torch.norm(b, 2, dim=0).mean()
    return a_norm/b_norm