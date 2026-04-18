import inspect
from functools import cache
from typing import Callable, Self, TypeVar, get_type_hints

import cattrs
import torch
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_map

converter = cattrs.Converter()

converter.register_unstructure_hook(torch.Tensor, lambda v: v)
converter.register_structure_hook(torch.Tensor, lambda v, _t: v)

T = TypeVar("T", bound="PyTree")

_PYTREE_CLS_KEY = "__pytree_cls__"

_unstructure_fn = cache(lambda cls: make_dict_unstructure_fn(cls, converter))


def _structure_fn(cls: type[T]) -> Callable[..., T]:
    structure_fn = make_dict_structure_fn(cls, converter)
    return lambda d: structure_fn(d, cls)


def _resolve_generic_cls(instance: "PyTree") -> type:
    """For generic PyTree subclasses, resolve TypeVar params to concrete types."""
    cls = type(instance)
    params = getattr(cls, "__type_params__", None) or getattr(cls, "__parameters__", ())
    if not params:
        return cls
    hints = get_type_hints(cls)
    tvar_to_type: dict[TypeVar, type] = {}
    for name, hint in hints.items():
        if isinstance(hint, TypeVar) and hasattr(instance, name):
            tvar_to_type[hint] = type(getattr(instance, name))
    concrete = tuple(tvar_to_type.get(p, p) for p in params)
    return cls[concrete[0]] if len(concrete) == 1 else cls.__class_getitem__(concrete)  # type: ignore


class PyTree:
    def tree_map_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        pytree_dict = self.__to_pytree__()
        mapped = tree_map(
            lambda x: x.tree_map_tensor(fn) if isinstance(x, PyTree) else fn(x) if isinstance(x, Tensor) else x,
            pytree_dict,
        )
        return self.__from_pytree__(mapped)

    def tree_map(self, fn: Callable[["Tensor | PyTree"], "Tensor | PyTree"]) -> Self:
        pytree_dict = self.__to_pytree__()
        mapped = tree_map(
            lambda x: fn(x) if isinstance(x, Tensor | PyTree) else x,
            pytree_dict,
        )
        return self.__from_pytree__(mapped)

    def to(self, device: torch.device | str) -> Self:
        return self.tree_map(lambda t: t.to(device))

    def full_tensor(self) -> Self:
        return self.tree_map(lambda t: t.full_tensor() if isinstance(t, DTensor | PyTree) else t)

    def __to_pytree__(self) -> dict:
        d = _unstructure_fn(type(self))(self)
        resolved = _resolve_generic_cls(self)
        if resolved is not type(self):
            d[_PYTREE_CLS_KEY] = resolved
        return d

    @classmethod
    def __from_pytree__(cls, data: dict) -> Self:
        resolved_cls = data.pop(_PYTREE_CLS_KEY, cls)
        return _structure_fn(resolved_cls)(data)


def _is_pytree_type(t: type) -> bool:
    cls = inspect.getattr_static(t, "__origin__", t)
    return isinstance(cls, type) and issubclass(cls, PyTree)


converter.register_unstructure_hook_func(_is_pytree_type, lambda v: v)
converter.register_structure_hook_func(_is_pytree_type, lambda v, _t: v)
