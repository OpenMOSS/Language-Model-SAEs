"""Early-stop activation caching for TransformerLens models.

Provides run_with_cache_until as a faster replacement for run_with_cache when
only activations up to a specific hook point are needed.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

import torch
from transformer_lens.hook_points import HookPoint
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
