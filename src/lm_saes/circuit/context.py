from contextlib import contextmanager
from typing import Callable, Tuple, Union
import torch
from transformer_lens.hook_points import HookPoint, HookedRootModule

from lm_saes.sae import SparseAutoEncoder

@contextmanager
def apply_sae(
    model: HookedRootModule,
    saes: list[SparseAutoEncoder]
):
    """
    Apply the sparse autoencoders to the model.
    """
    fwd_hooks: list[Tuple[Union[str, Callable], Callable]] = []
    def get_fwd_hooks(sae: SparseAutoEncoder) -> list[Tuple[Union[str, Callable], Callable]]:
        if sae.cfg.hook_point_in == sae.cfg.hook_point_out:
            def hook(tensor: torch.Tensor, hook: HookPoint):
                reconstructed = sae.forward(tensor)
                return reconstructed + (tensor - reconstructed).detach()
            return [(sae.cfg.hook_point_in, hook)]
        else:
            x = None
            def hook_in(tensor: torch.Tensor, hook: HookPoint):
                nonlocal x
                x = tensor
                return tensor
            def hook_out(tensor: torch.Tensor, hook: HookPoint):
                nonlocal x
                assert x is not None, "hook_in must be called before hook_out."
                reconstructed = sae.forward(x, label=tensor)
                x = None
                return reconstructed + (tensor - reconstructed).detach()
            return [(sae.cfg.hook_point_in, hook_in), (sae.cfg.hook_point_out, hook_out)]
    for sae in saes:
        hooks = get_fwd_hooks(sae)
        fwd_hooks.extend(hooks)
    with model.mount_hooked_modules([(sae.cfg.hook_point_out, "sae", sae) for sae in saes]):
        with model.hooks(fwd_hooks):
            yield model

@contextmanager
def detach_on(
    model: HookedRootModule,
    hook_points: list[str],
):
    """
    Detach the gradients on the given hook points.
    """

    def generate_hook():
        hook_pre = HookPoint()
        hook_post = HookPoint()
        def hook(tensor: torch.Tensor, hook: HookPoint):
            return hook_post(hook_pre(tensor).detach().requires_grad_())
        return hook_pre, hook_post, hook
    
    hooks = {hook_point: generate_hook() for hook_point in hook_points}
    fwd_hooks = [(hook_point, hook) for hook_point, (_, _, hook) in hooks.items()]
    with model.mount_hooked_modules([(hook_point, "pre", hook) for hook_point, (hook, _, _) in hooks.items()] + [(hook_point, "post", hook) for hook_point, (_, hook, _) in hooks.items()]):
        with model.hooks(fwd_hooks):
            yield model
