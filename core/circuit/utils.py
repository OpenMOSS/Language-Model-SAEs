import copy
from typing import Callable

from einops import repeat
import torch

from core.circuit.computation_graph import Meta, NodeInfo, padding


def concat(x: torch.Tensor, x_meta: list[list[Meta]], y: torch.Tensor, y_meta: list[list[Meta]]):
    return torch.cat([x, y], dim=1), [m1 + m2 for m1, m2 in zip(x_meta, y_meta)]

def concat_bias(x: torch.Tensor, x_meta: list[list[Meta]], bias: torch.Tensor, bias_meta: Meta):
    return concat(x, x_meta, repeat(bias, "... -> pos 1 ...", pos=x.size(0)), [[bias_meta] for _ in range(x.size(0))])

def compact(x: torch.Tensor, x_meta: list[list[Meta]], norm_threshold: float = 0.03):
    max_passed = max(*[(x[i].norm(dim=-1) > norm_threshold).sum().item() for i in range(x.size(0))])
    original = x.sum(1)
    for i in range(x.size(0)):
        passed = x[i].norm(dim=-1) > norm_threshold
        x[i, :passed.sum()] = x[i, passed]
        x[i, passed.sum():] = 0
        x_meta[i] = [m for m, p in zip(x_meta[i], passed) if p] + [padding for _ in range(max_passed - passed.sum())]
    x = x[:, :max_passed]
    x, x_meta = concat(x, x_meta, (original - x.sum(1)).unsqueeze(1), [[(NodeInfo("compact_rest", pos=i), None)] for i in range(x.size(0))])
    assert torch.allclose(x.sum(1), original, atol=1e-4, rtol=1e-3)
    return x, x_meta

def equal_shape(x: torch.Tensor, x_meta: list[list[Meta]]):
    return x.size(0) == len(x_meta) and all(len(m) == x.size(1) for m in x_meta)

def ablate(x: torch.Tensor, meta: list[list[Meta]], condition: Callable[[Meta], bool]):
    x = x.clone()
    meta = copy.deepcopy(meta)
    for pos in range(x.size(0)):
        for i in range(x.size(1)):
            if condition(meta[pos][i]):
                x[pos, i] = 0
                meta[pos][i] = padding
    return x, meta