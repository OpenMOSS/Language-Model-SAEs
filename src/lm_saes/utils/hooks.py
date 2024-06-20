import torch
from transformer_lens.hook_points import HookPoint

def compose_hooks(*hooks):
    """
    Compose multiple hooks into a single hook by executing them in order.
    """
    def composed_hook(tensor: torch.Tensor, hook: HookPoint):
        for hook_fn in hooks:
            tensor = hook_fn(tensor, hook)
        return tensor
    return composed_hook

def retain_grad_hook(tensor: torch.Tensor, hook: HookPoint):
    """
    Retain the gradient of the tensor at the given hook point.
    """
    tensor.retain_grad()
    return tensor

def detach_hook(tensor: torch.Tensor, hook: HookPoint):
    """
    Detach the tensor at the given hook point.
    """
    return tensor.detach().requires_grad_(True)