from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable, TypeVar

import cattrs
import torch
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn, override
from cattrs.strategies import use_class_methods

FORMAT_VERSION = "2"

T = TypeVar("T")
C = TypeVar("C", bound=type)

converter = cattrs.Converter()

converter.register_unstructure_hook(torch.Tensor, lambda v: v)
converter.register_structure_hook(torch.Tensor, lambda v, _t: v)

use_class_methods(converter, "__structure__", "__unstructure__")

_loading_source_version: ContextVar[str | None] = ContextVar("_loading_source_version", default=None)
"""The ``_version`` value of the blob currently being loaded."""


def migrate(*, before: str) -> Callable[[Callable[[Any], Any]], Any]:
    """Mark a static function as a version migration.

    The migration fires during :func:`load` whenever the blob's stored
    ``_version`` is strictly less than ``before``. It maps raw dict to raw dict.
    Multiple migrations are applied in ascending ``before`` order.

    Usage:

        @dataclass
        class MyClass:
            new_field: int

            @migrate(before="2")
            def _v1_to_v2(data: dict) -> dict:
                return {"new_field": data["old_field"]}
    """

    def decorator(func: Callable[[Any], Any]) -> Any:
        setattr(func, "__migrate_before__", before)
        return staticmethod(func)

    return decorator


def _collect_migrations(cls: type) -> list[tuple[str, Callable[[Any], Any]]]:
    """Gather every ``@migrate``-annotated static method declared directly on
    ``cls`` (inherited migrations are not considered)."""
    collected: list[tuple[str, Callable[[Any], Any]]] = []
    for attr in vars(cls).values():
        if isinstance(attr, staticmethod):
            func = attr.__func__
            before = getattr(func, "__migrate_before__", None)
            if before is not None:
                collected.append((before, func))
    collected.sort(key=lambda entry: int(entry[0]))
    return collected


def _has_migrations(cls: Any) -> bool:
    return isinstance(cls, type) and bool(_collect_migrations(cls))


def _make_migrating_structure_hook(cls: type) -> Callable[[Any, type], Any]:
    migrations = _collect_migrations(cls)

    class_structure = cls.__dict__.get("__structure__")
    if class_structure is not None:

        def delegate(data: Any, _t: type) -> Any:
            return class_structure.__func__(cls, data)
    else:
        base_hook = make_dict_structure_fn(cls, converter)

        def delegate(data: Any, t: type) -> Any:
            return base_hook(data, t)

    def migrating_hook(data: Any, t: type) -> Any:
        source_version = _loading_source_version.get()
        if source_version is not None:
            for before_v, func in migrations:
                if int(source_version) < int(before_v):
                    data = func(data)
        return delegate(data, t)

    return migrating_hook


converter.register_structure_hook_factory(_has_migrations, _make_migrating_structure_hook)


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
    """Rehydrate a value of type *cls* from a :func:`dump` blob. Blobs with
    ``_version`` older than :data:`FORMAT_VERSION` are migrated on the fly via
    any :func:`migrate`-annotated static methods their classes declare."""
    source_version: str = blob["_version"]
    token = _loading_source_version.set(source_version)
    try:
        return converter.structure(blob["data"], cls)
    finally:
        _loading_source_version.reset(token)


__all__ = [
    "FORMAT_VERSION",
    "converter",
    "dump",
    "load",
    "migrate",
    "override",
    "register_overrides",
    "structure",
    "unstructure",
]
