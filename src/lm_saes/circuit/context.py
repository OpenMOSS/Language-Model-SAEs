from contextlib import contextmanager
from typing import Callable, Tuple, Union
import torch
from transformer_lens.hook_points import HookPoint, HookedRootModule

from lm_saes.sae import SparseAutoEncoder

@contextmanager
def apply_sae(
    model: HookedRootModule,
    saes: list[SparseAutoEncoder],
    keep_sae_errors: bool | list[bool] = True,
):
    """
    Apply the sparse autoencoders to the model.
    """
    fwd_hooks: list[Tuple[Union[str, Callable], Callable]] = []
    def get_fwd_hooks(sae: SparseAutoEncoder, keep_sae_errors: bool) -> list[Tuple[Union[str, Callable], Callable]]:
        if sae.cfg.hook_point_in == sae.cfg.hook_point_out:
            def hook(tensor: torch.Tensor, hook: HookPoint):
                reconstructed = sae.forward(tensor)
                if not keep_sae_errors and reconstructed.norm() == 0:
                    raise ValueError(f"The norm of the reconstructed tensor is 0, in hook point {hook.name}.")
                return reconstructed + (tensor - reconstructed).detach() if keep_sae_errors else reconstructed / reconstructed.norm() * tensor.norm()
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
                if not keep_sae_errors and reconstructed.norm() == 0:
                    raise ValueError(f"The norm of the reconstructed tensor is 0, in hook point {hook.name}.")
                return reconstructed + (tensor - reconstructed).detach() if keep_sae_errors else reconstructed / reconstructed.norm() * tensor.norm()
            return [(sae.cfg.hook_point_in, hook_in), (sae.cfg.hook_point_out, hook_out)]
        
    keep_sae_errors = [keep_sae_errors] * len(saes) if isinstance(keep_sae_errors, bool) else keep_sae_errors
    for sae, keep_sae_errors in zip(saes, keep_sae_errors):
        hooks = get_fwd_hooks(sae, keep_sae_errors)
        fwd_hooks.extend(hooks)
    with model.mount_hooked_modules([(sae.cfg.hook_point_out, "sae", sae) for sae in saes]):
        with model.hooks(fwd_hooks):
            yield model