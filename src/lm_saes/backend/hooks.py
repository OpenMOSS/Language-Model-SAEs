from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Union, cast

import torch
import torch.nn as nn
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


def _make_bias_leaf(bias_data: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    """Expand a bias tensor to ``(batch_size, seq_len, *bias_shape)`` as a detached leaf."""
    leaf = bias_data.detach().unsqueeze(0).unsqueeze(0)  # (1, 1, *bias_shape)
    leaf = leaf.expand(batch_size, seq_len, *bias_data.shape).clone()
    return leaf.requires_grad_(True)


@contextmanager
def replace_biases_with_leaves(
    model: HookedRootModule,
    saes: list["SparseDictionary"],
    batch_size: int,
    seq_len: int,
):

    from lm_saes.models.lorsa import LowRankSparseAttention
    from lm_saes.models.molt import MixtureOfLinearTransform
    from lm_saes.models.sae import SparseAutoEncoder

    bias_leaves: dict[str, torch.Tensor] = {}
    # (module, attr_name, original_value, is_param_swap)
    saved_state: list[tuple[nn.Module, str, Union[torch.Tensor, nn.Parameter], bool]] = []
    fwd_hooks: list[tuple[str | Callable, Callable]] = []

    def _save_zero_and_hook(
        module: nn.Module,
        attr_name: str,
        cache_key: str,
        hook_point: str,
    ) -> None:
        """Zero a base-model bias and register a hook that adds the leaf back."""
        param = getattr(module, attr_name, None)
        if param is None:
            return
        saved_state.append((module, attr_name, param.data.clone(), False))
        leaf = _make_bias_leaf(param.data, batch_size, seq_len)
        bias_leaves[cache_key] = leaf
        param.zero_()

        def _hook(tensor: torch.Tensor, hook: HookPoint, _leaf: torch.Tensor = leaf) -> torch.Tensor:
            return tensor + _leaf

        fwd_hooks.append((hook_point, _hook))

    def _save_and_replace_param(
        module: nn.Module,
        attr_name: str,
        cache_key: str,
    ) -> None:
        """Swap an ``nn.Parameter`` with an expanded leaf."""
        original = module._parameters.get(attr_name)
        if original is None:
            return
        module._parameters.pop(attr_name)
        saved_state.append((module, attr_name, original, True))
        leaf = _make_bias_leaf(original.data, batch_size, seq_len)
        bias_leaves[cache_key] = leaf
        object.__setattr__(module, attr_name, leaf)

    try:
        for i, block in enumerate(model.blocks):  # type: ignore[arg-type]
            if hasattr(block, "attn") and hasattr(block.attn, "b_O") and getattr(block.attn, "b_O") is not None:
                _save_zero_and_hook(block.attn, "b_O", f"blocks.{i}.attn.b_O", f"blocks.{i}.hook_attn_out")
            if hasattr(block, "mlp") and hasattr(block.mlp, "b_out") and getattr(block.mlp, "b_out") is not None:
                _save_zero_and_hook(block.mlp, "b_out", f"blocks.{i}.mlp.b_out", f"blocks.{i}.hook_mlp_out")

        # ---- SAE / MOLT / LORSA biases (parameter replacement) ----
        for sae in saes:
            if not isinstance(sae, SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform):
                continue
            if isinstance(sae, LowRankSparseAttention):
                for bias_name in ("b_Q", "b_K"):
                    if hasattr(sae, bias_name) and getattr(sae, bias_name) is not None:
                        _save_and_replace_param(sae, bias_name, f"{sae.cfg.hook_point_out}.sae.{bias_name}")
            # decoder bias b_D (SAE / MOLT / LORSA)
            if sae.cfg.use_decoder_bias and hasattr(sae, "b_D"):
                _save_and_replace_param(sae, "b_D", f"{sae.cfg.hook_point_out}.sae.b_D")

        with model.hooks(fwd_hooks):
            yield bias_leaves

    finally:
        for module, attr_name, original, is_param_swap in saved_state:
            if is_param_swap:
                if attr_name in module.__dict__:
                    del module.__dict__[attr_name]
                module._parameters[attr_name] = cast(nn.Parameter, original)
            else:
                getattr(module, attr_name).data.copy_(original)
