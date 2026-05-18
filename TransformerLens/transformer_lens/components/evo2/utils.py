import torch
import logging

log = logging.getLogger(__name__)


def get_dim_for_local_rank(dim: int, world_size: int, local_rank: int, multiple_of: int = 1) -> int:
    """Return the slice of `dim` assigned to `local_rank` when split across `world_size`."""
    multiple = dim // multiple_of
    div = multiple // world_size
    mod = multiple % world_size
    local_multiple = div + int(local_rank < mod)
    return local_multiple * multiple_of


def grab_first_if_tuple(x):
    return x[0] if x.__class__.__name__ == "tuple" else x


def interleave(z_pre):
    if len(z_pre.shape) == 3:  # non-cached: [B, L, D]
        x1 = z_pre[:, 0::3, :]
        x2 = z_pre[:, 1::3, :]
        v  = z_pre[:, 2::3, :]
        return torch.cat([x1, x2, v], dim=1)
    else:                       # cached: [..., D]
        x1 = z_pre[..., 0::3]
        x2 = z_pre[..., 1::3]
        v  = z_pre[..., 2::3]
        return torch.concat([x1, x2, v], dim=-1)


def column_split(x, num_heads, head_size):
    """Split a tensor across heads (column-wise) for three projections."""
    if len(x.shape) == 2:
        x = x.reshape(x.shape[0], num_heads, 3 * head_size)
        x2 = x[..., :head_size].reshape(x.shape[0], -1)
        x1 = x[..., head_size:2 * head_size].reshape(x.shape[0], -1)
        v  = x[..., 2 * head_size:].reshape(x.shape[0], -1)
    else:
        x = x.reshape(x.shape[0], num_heads, 3 * head_size, x.shape[2])
        x2 = x[:, :, :head_size].reshape(x.shape[0], -1, x.shape[-1])
        x1 = x[:, :, head_size:2 * head_size].reshape(x.shape[0], -1, x.shape[-1])
        v  = x[:, :, 2 * head_size:].reshape(x.shape[0], -1, x.shape[-1])
    return x2, x1, v


def move_to_device(module, device):
    """Recursively move all parameters and buffers to `device`."""
    for child in module.children():
        move_to_device(child, device)
    for param in module.parameters(recurse=False):
        if param.device != device:
            param.data = param.data.to(device)
    for buf in module.buffers(recurse=False):
        if buf.device != device:
            buf.data = buf.data.to(device)
    module.to(device)


def fixup_fp8_extra_states(module):
    """Recursively fix device location of Transformer Engine fp8 extra states.

    No-op when Transformer Engine is not installed.
    """
    if not hasattr(module, "fp8_meta"):
        for child in module.children():
            fixup_fp8_extra_states(child)
        return

    for child in module.children():
        fixup_fp8_extra_states(child)

    log.debug(f"Reloading fp8 extra state to proper device for {module}")
    module.fp8_meta_tensors_initialized = False
    device = next(module.parameters()).device
    with torch.cuda.device(device):
        module.set_extra_state(module.get_extra_state())

    for k in ["scaling_fwd", "scaling_bwd"]:
        for attr in ["amax_history", "scale"]:
            tensor = getattr(module.fp8_meta[k], attr)
            assert tensor.device == device, (k, tensor, device)


def fixup_te_workspace():
    """Patch Transformer Engine to use per-device cuBLAS workspaces for multi-GPU.

    No-op when Transformer Engine is not installed.
    """
    try:
        import transformer_engine  # noqa: F401
    except ImportError:
        return

    from functools import lru_cache

    @lru_cache
    def te_cublas_get_workspace_per_device(device):
        log.info(f"Fixup applied: Allocating cublas workspace for {device=}")
        import transformer_engine.pytorch.module.base as tebase

        with torch.cuda.device(device):
            tebase._cublas_workspace = None
            return tebase.get_workspace()

    def get_workspace():
        return te_cublas_get_workspace_per_device(torch.cuda.current_device())

    import transformer_engine.pytorch.module.linear as telinear
    telinear.get_workspace = get_workspace


def print_rank_0(message, debug=False, end="\n"):
    """Print only from rank 0 (or always when not running distributed)."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True, end=end)
    else:
        print(message, flush=True, end=end)


class dotdict(dict):
    """Dictionary with attribute-style access."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
