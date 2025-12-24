import re
import threading
from functools import wraps
from typing import Any, Callable, Generic, ParamSpec, TypeVar

import numpy as np
import torch

P = ParamSpec("P")
R = TypeVar("R")


class synchronized(Generic[P, R]):
    """Decorator to ensure sequential execution of a function based on parameters.

    Different parameters can be acquired in parallel, but the same parameters
    will be executed sequentially.
    """

    _func: Callable[P, R]

    def __init__(self, func: Callable[P, R]) -> None:
        self._func = func
        self._locks: dict[frozenset[tuple[str, Any]], threading.Lock] = {}
        self._global_lock = threading.Lock()
        wraps(func)(self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        assert len(args) == 0, "Positional arguments are not supported"
        key = frozenset(kwargs.items())

        # The lock creation is locked by the global lock to avoid race conditions on locks.
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            lock = self._locks[key]

        with lock:
            return self._func(*args, **kwargs)  # type: ignore[call-arg]

    def __getattr__(self, name: str):
        return getattr(self._func, name)


def make_serializable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable formats."""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def trim_minimum(
    origins: list[dict[str, Any] | None],
    feature_acts_indices: np.ndarray,
    feature_acts_values: np.ndarray,
) -> tuple[list[dict[str, Any] | None], np.ndarray, np.ndarray]:
    """Trim multiple arrays to the length of the shortest non-None array.

    Args:
        origins: Origins
        feature_acts_indices: Feature acts indices
        feature_acts_values: Feature acts values

    Returns:
        list: List of trimmed arrays
    """

    min_length = min(len(origins), feature_acts_indices[-1] + 10)
    feature_acts_indices_mask = feature_acts_indices <= min_length
    return (
        origins[: int(min_length)],
        feature_acts_indices[feature_acts_indices_mask],
        feature_acts_values[feature_acts_indices_mask],
    )


def natural_sort_key(name: str) -> list[tuple[int, int | str]]:
    """Convert a string into a sort key for natural sorting.

    Splits the string into alternating number and string parts, returning
    a list of tuples where numbers are (0, int) and strings are (1, str).
    This allows numbers to be sorted numerically and strings alphabetically.
    """
    parts = re.split(r"(\d+)", name)
    key: list[tuple[int, int | str]] = []
    for part in parts:
        if part:
            if part.isdigit():
                key.append((0, int(part)))
            else:
                key.append((1, part.lower()))
    return key
