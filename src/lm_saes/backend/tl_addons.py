"""TransformerLens addon utilities.

Standalone functions supplementing TransformerLens' HookedRootModule:
  - run_with_cache_until: cache activations up to a specific hook, then stop.
  - run_with_ref_cache: cache non-detached activation references (supports retain_grad).
  - mount_hooked_modules: context manager to temporarily mount sub-modules at hook points.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.utils import Slice, SliceInput

NamesFilter = Optional[Union[Callable[[str], bool], Sequence[str], str]]
DeviceType = Optional[torch.device]


class _ModuleStop(Exception):
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


def run_with_cache_until(
    model: Any,
    *model_args: Any,
    names_filter: NamesFilter = None,
    until: str | None = None,
    device: DeviceType = None,
    remove_batch_dim: bool = False,
    reset_hooks_end: bool = True,
    clear_contexts: bool = False,
    pos_slice: Slice | SliceInput | None = None,
    **model_kwargs: Any,
) -> tuple[Any, dict[str, torch.Tensor]]:
    """Run model and return output plus cached activations, stopping at a given hook.

    Stops the forward pass when the `until` hook is reached, avoiding computation
    of later layers. Use this instead of run_with_cache when only activations
    up to a specific layer are needed.

    Args:
        model: A TransformerLens model with get_caching_hooks and hooks (e.g. HookedTransformer).
        *model_args: Positional arguments for the model.
        names_filter: Filter for which activations to cache. None, str, list[str], or callable.
        until: Hook name to stop at. Defaults to the last cached hook.
        device: Device to cache activations on. None uses model device.
        remove_batch_dim: If True, remove batch dim when caching (batch_size=1 only).
        reset_hooks_end: If True, remove hooks when done.
        clear_contexts: If True, clear hook contexts when resetting.
        pos_slice: Slice to apply to cached positions.
        **model_kwargs: Keyword arguments for the model.

    Returns:
        (model_output, cache_dict) where model_output is the activation at `until`
        (or full output if no early stop) and cache_dict contains cached activations.
    """
    pos_slice = Slice.unwrap(pos_slice)

    cache_dict, fwd, _ = model.get_caching_hooks(
        names_filter,
        False,
        device,
        remove_batch_dim=remove_batch_dim,
        pos_slice=pos_slice,
    )

    if until is None:
        until = fwd[-1][0]

    def stop_hook(tensor: torch.Tensor, *, hook: HookPoint) -> None:
        if hook.name == until:
            raise _ModuleStop(tensor)

    with model.hooks(
        fwd_hooks=fwd + [(until, stop_hook)],
        reset_hooks_end=reset_hooks_end,
        clear_contexts=clear_contexts,
    ):
        try:
            model_out = model(*model_args, **model_kwargs)
        except _ModuleStop as e:
            model_out = e.tensor

    return model_out, cache_dict


def run_with_ref_cache(
    model: Any,
    *model_args: Any,
    names_filter: NamesFilter = None,
    retain_grad: bool = False,
    reset_hooks_end: bool = True,
    clear_contexts: bool = False,
    **model_kwargs: Any,
) -> tuple[Any, dict[str, torch.Tensor]]:
    """Run the model and return a reference cache of (non-detached) activations.

    Unlike run_with_cache, activations are stored by reference and are not moved
    to a different device. This allows gradient flow through the cached tensors
    when retain_grad=True.

    Args:
        model: A TransformerLens HookedRootModule (e.g. HookedTransformer).
        *model_args: Positional arguments for the model.
        names_filter: Filter for which activations to cache. None, str, list[str], or callable.
        retain_grad: If True, call retain_grad() on each cached activation.
        reset_hooks_end: If True, remove hooks when done.
        clear_contexts: If True, clear hook contexts when resetting.
        **model_kwargs: Keyword arguments for the model.

    Returns:
        (model_output, cache_dict) where cache_dict maps hook names to live tensors.
    """
    if names_filter is None:

        def _filter(name: str) -> bool:
            return True
    elif isinstance(names_filter, str):
        _str = names_filter

        def _filter(name: str) -> bool:
            return name == _str
    elif isinstance(names_filter, list):
        _list = names_filter

        def _filter(name: str) -> bool:
            return name in _list
    elif callable(names_filter):
        _fn = names_filter

        def _filter(name: str) -> bool:
            return _fn(name)
    else:
        raise ValueError("names_filter must be None, a string, list of strings, or callable")

    cache: dict[str, torch.Tensor] = {}

    def save_hook(tensor: torch.Tensor, *, hook: HookPoint) -> None:
        if retain_grad:
            tensor.retain_grad()
        cache[hook.name] = tensor  # type: ignore[index]

    fwd_hooks = [(name, save_hook) for name, hp in model.hook_dict.items() if _filter(name)]

    with model.hooks(
        fwd_hooks=fwd_hooks,
        reset_hooks_end=reset_hooks_end,
        clear_contexts=clear_contexts,
    ):
        model_out = model(*model_args, **model_kwargs)

    return model_out, cache


@contextmanager
def mount_hooked_modules(
    model: HookedRootModule,
    hooked_modules: Sequence[Tuple[str, str, nn.Module]],
) -> Generator[HookedRootModule, None, None]:
    """Context manager to temporarily mount child modules at named hook points.

    Attaches each module as a sub-module of the specified hook point, calls
    model.setup() so the new modules are registered in hook_dict/mod_dict, and
    cleans everything up on exit.

    Args:
        model: A TransformerLens HookedRootModule (e.g. HookedTransformer).
        hooked_modules: Sequence of (hook_point_name, child_name, module) triples.
            hook_point_name – name of an existing HookPoint in model.mod_dict.
            child_name      – attribute name to mount the module under.
            module          – the nn.Module to mount.

    Yields:
        The model with the modules mounted.
    """
    for hook_point_name, child_name, module in hooked_modules:
        hook_point = model.mod_dict[hook_point_name]
        assert isinstance(hook_point, HookPoint)
        hook_point.add_module(child_name, module)

    model.setup()

    try:
        yield model
    finally:
        for hook_point_name, child_name, module in hooked_modules:
            hook_point = model.mod_dict[hook_point_name]
            delattr(hook_point, child_name)
            if isinstance(module, HookedRootModule):
                module.setup()
        model.setup()
