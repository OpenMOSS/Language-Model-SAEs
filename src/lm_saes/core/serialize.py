from __future__ import annotations

from typing import Any, Callable, TypeVar

import cattrs
import torch
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn, override
from cattrs.strategies import use_class_methods

FORMAT_VERSION = "1"

T = TypeVar("T")
C = TypeVar("C", bound=type)

converter = cattrs.Converter()

converter.register_unstructure_hook(torch.Tensor, lambda v: v)
converter.register_structure_hook(torch.Tensor, lambda v, _t: v)

use_class_methods(converter, "__structure__", "__unstructure__")


def register_overrides(**overrides: Any) -> Callable[[C], C]:
    def decorator(cls: C) -> C:
        converter.register_unstructure_hook(cls, make_dict_unstructure_fn(cls, converter, **overrides))
        converter.register_structure_hook(cls, make_dict_structure_fn(cls, converter, **overrides))
        return cls

    return decorator


structure = converter.structure

unstructure = converter.unstructure


def dump(obj: Any) -> dict[str, Any]:
    """Serialize *obj* to a version-tagged, ``torch.save``-friendly dict."""
    return {"_version": FORMAT_VERSION, "data": converter.unstructure(obj)}


def load(blob: Any, cls: type[T]) -> T:
    """Rehydrate a value of type *cls* from a :func:`dump` blob."""
    if not is_current_format(blob):
        raise ValueError(f"Expected serialized blob with _version={FORMAT_VERSION!r}, got {blob!r}")
    return converter.structure(blob["data"], cls)


def is_current_format(blob: Any) -> bool:
    return isinstance(blob, dict) and blob.get("_version") == FORMAT_VERSION


__all__ = [
    "converter",
    "dump",
    "is_current_format",
    "load",
    "override",
    "register_overrides",
    "structure",
    "unstructure",
]
