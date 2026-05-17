from __future__ import annotations

from pathlib import Path
import sys

DEFAULT_EVO2_CHECKPOINTS = {
    "evo2_7b": Path("/inspire/hdd/global_user/hezhengfu-240208120186/models/evo2_7b/evo2_7b.pt"),
}


def _get_vendor_evo2_class():
    try:
        from evo2 import Evo2 as vendor_evo2
    except ModuleNotFoundError:
        # Allow local development before `uv sync` installs the vendored package.
        vendored_root = Path(__file__).resolve().parents[3] / "third_party" / "evo2"
        if vendored_root.exists():
            sys.path.insert(0, str(vendored_root))
        from evo2 import Evo2 as vendor_evo2
    return vendor_evo2


def resolve_evo2_checkpoint(model_name: str = "evo2_7b", local_path: str | Path | None = None) -> str | None:
    if local_path is not None:
        return str(Path(local_path).expanduser().resolve())
    default_path = DEFAULT_EVO2_CHECKPOINTS.get(model_name)
    if default_path is not None and default_path.exists():
        return str(default_path)
    return None


class Evo2:
    def __new__(cls, model_name: str = "evo2_7b", local_path: str | Path | None = None, *args, **kwargs):
        vendor_evo2 = _get_vendor_evo2_class()
        return vendor_evo2(
            model_name=model_name,
            local_path=resolve_evo2_checkpoint(model_name, local_path),
            *args,
            **kwargs,
        )


def load_evo2(model_name: str = "evo2_7b", local_path: str | Path | None = None) -> Evo2:
    return Evo2(model_name=model_name, local_path=local_path)
