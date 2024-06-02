
from typing import Callable, Optional, Tuple, cast
import transformer_lens as tl
tl.HookedSAETransformer

class HookedRootModule(tl.hook_points.HookedRootModule):
    def get_ref_caching_hooks(
        self,
        names_filter: tl.hook_points.NamesFilter = None,
        retain_grad: bool = False,
        cache: Optional[dict] = None,
    ) -> Tuple[dict, list, list]:
        """Creates hooks to keep references to activations. Note: It does not add the hooks to the model.

        Args:
            names_filter (NamesFilter, optional): Which activations to cache. Can be a list of strings (hook names) or a filter function mapping hook names to booleans. Defaults to lambda name: True.
            retain_grad (bool, optional): Whether to retain gradients for the activations. Defaults to False.
            cache (Optional[dict], optional): The cache to store activations in, a new dict is created by default. Defaults to None.

        Returns:
            cache (dict): The cache where activations will be stored.
            fwd_hooks (list): The forward hooks.
        """
        if cache is None:
            cache = {}

        if names_filter is None:
            names_filter = lambda name: True
        elif isinstance(names_filter, str):
            filter_str = names_filter
            names_filter = lambda name: name == filter_str
        elif isinstance(names_filter, list):
            filter_list = names_filter
            names_filter = lambda name: name in filter_list

        # mypy can't seem to infer this
        names_filter = cast(Callable[[str], bool], names_filter)

        def save_hook(tensor, hook):
            if retain_grad:
                tensor.retain_grad()
            cache[hook.name] = tensor

        fwd_hooks = []
        for name, hp in self.hook_dict.items():
            if names_filter(name):
                fwd_hooks.append((name, save_hook))

        return cache, fwd_hooks

    def run_with_ref_cache(
        self,
        *model_args,
        names_filter: tl.hook_points.NamesFilter = None,
        retain_grad: bool = False,
        reset_hooks_end=True,
        clear_contexts=False,
        **model_kwargs,
    ):
        """
        Runs the model and returns the model output and a reference cache.

        Args:
            *model_args: Positional arguments for the model.
            names_filter (NamesFilter, optional): A filter for which activations to cache. Accepts None, str,
                list of str, or a function that takes a string and returns a bool. Defaults to None, which
                means cache everything.
            retain_grad (bool, optional): Whether to retain gradients for the activations. Defaults to False.
            **model_kwargs: Keyword arguments for the model.

        Returns:
            tuple: A tuple containing the model output and the reference cache.

        """
        cache_dict, fwd = self.get_ref_caching_hooks(
            names_filter,
            retain_grad=retain_grad,
        )

        with self.hooks(
            fwd_hooks=fwd,
            reset_hooks_end=reset_hooks_end,
            clear_contexts=clear_contexts,
        ):
            model_out = self(*model_args, **model_kwargs)

        return model_out, cache_dict

class HookedTransformer(tl.HookedTransformer, HookedRootModule):
    pass