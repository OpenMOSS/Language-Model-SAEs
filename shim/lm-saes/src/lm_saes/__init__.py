"""Backward-compatibility shim: lm-saes has been renamed to llamascopium."""

import warnings

warnings.warn(
    "The 'lm-saes' package has been renamed to 'llamascopium'. Please update your dependency: pip install llamascopium",
    DeprecationWarning,
    stacklevel=2,
)

from llamascopium import *  # noqa: F403, E402
from llamascopium import __all__  # noqa: F401, E402


def _cli_entrypoint() -> None:
    """Shim CLI entry point that delegates to llamascopium."""
    from llamascopium.cli import entrypoint

    entrypoint()
