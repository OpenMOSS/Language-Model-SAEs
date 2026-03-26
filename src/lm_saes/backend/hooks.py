from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, cast

import torch
from transformer_lens.hook_points import HookedRootModule, HookPoint

from lm_saes.backend.tl_addons import mount_hooked_modules

if TYPE_CHECKING:
    from lm_saes.models.sparse_dictionary import SparseDictionary


@contextmanager
def apply_saes(model: HookedRootModule, saes: list["SparseDictionary"]):
    from lm_saes.models.lorsa import LowRankSparseAttention
    from lm_saes.models.molt import MixtureOfLinearTransform
    from lm_saes.models.sae import SparseAutoEncoder

    """
    Apply the sparse dictionaries to the model.
    """
    # Rule out CLTs and crosscoders. Ideally CLTs should be supported.
    assert all(
        [isinstance(sae, SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform) for sae in saes]
    ), "Currently only support sparse dictionaries that guarantee the hook points in happen before the hook points out."

    saes: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform] = cast(
        list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform], saes
    )

    fwd_hooks: list[tuple[str | Callable, Callable]] = []
    hook_errors: list[tuple[str, str, HookPoint]] = []

    def get_fwd_hooks(
        sae: SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform,
    ) -> tuple[list[tuple[str | Callable, Callable]], HookPoint]:
        x = None
        hook_error = HookPoint()

        def hook_in(tensor: torch.Tensor, hook: HookPoint):
            nonlocal x
            x = sae.encode(tensor, hook_attn_scores=True)
            return tensor

        def hook_out(tensor: torch.Tensor, hook: HookPoint):
            nonlocal x
            assert x is not None, "hook_in must be called before hook_out."
            reconstructed = sae.decode(x)
            x = None
            return reconstructed + hook_error(tensor - reconstructed)

        return [(sae.cfg.hook_point_in, hook_in), (sae.cfg.hook_point_out, hook_out)], hook_error

    for sae in saes:
        hooks, hook_error = get_fwd_hooks(sae)
        fwd_hooks.extend(hooks)
        hook_errors.append((sae.cfg.hook_point_out, "error", hook_error))
    with mount_hooked_modules(model, [(sae.cfg.hook_point_out, "sae", sae) for sae in saes] + hook_errors):
        with model.hooks(fwd_hooks):
            yield model


@contextmanager
def detach_at(
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
    fwd_hooks: list[tuple[str | Callable, Callable]] = [
        (hook_point, hook) for hook_point, (_, _, hook) in hooks.items()
    ]
    with mount_hooked_modules(
        model,
        [(hook_point, "pre", hook) for hook_point, (hook, _, _) in hooks.items()]
        + [(hook_point, "post", hook) for hook_point, (_, hook, _) in hooks.items()],
    ):
        with model.hooks(fwd_hooks):
            yield model
