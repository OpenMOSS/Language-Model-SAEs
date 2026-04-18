import importlib.util
import json
from pathlib import Path
from typing import Any

import tomlkit
import typer
import yaml


def load_config(config_file: str | Path) -> dict[str, Any]:
    """Load configuration from a file.

    Args:
        config_file: Path to the configuration file.

    Returns:
        The loaded configuration as a dictionary.

    Raises:
        typer.BadParameter: If the file format is not supported or file doesn't exist.
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise typer.BadParameter(f"Configuration file not found: {config_file}")

    if config_path.suffix == ".json":
        with open(config_path) as f:
            return json.load(f)

    elif config_path.suffix in {".yaml", ".yml"}:
        with open(config_path) as f:
            return yaml.safe_load(f)

    elif config_path.suffix == ".toml":
        with open(config_path) as f:
            return tomlkit.load(f).unwrap()

    elif config_path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("__lm_sae_config__", config_path)
        if spec is None or spec.loader is None:
            raise typer.BadParameter(f"Failed to load configuration file: {config_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config

    raise typer.BadParameter(
        f"Unsupported configuration file format: {config_path.suffix}. "
        "Supported formats: .json, .yaml, .yml, .toml, .py"
    )
