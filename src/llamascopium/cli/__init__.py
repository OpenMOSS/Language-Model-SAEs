import datasets
import torch
import typer

from . import create, remove, run, show
from .common import (
    DEFAULT_MONGO_DB,
    DEFAULT_MONGO_URI,
    DEFAULT_SAE_SERIES,
    MongoDBOption,
    MongoURIOption,
    SAESeriesOption,
    console,
)
from .utils import load_config

app = typer.Typer(
    name="lm-saes",
    help="CLI for Language Model Sparse Autoencoders - training, analysis, and database management.",
    no_args_is_help=True,
)

# Register run commands directly on main app
for command in run.app.registered_commands:
    app.registered_commands.append(command)

# Register sub-apps
app.add_typer(create.app, name="create")
app.add_typer(remove.app, name="remove")
app.add_typer(show.app, name="show")


def entrypoint() -> None:
    """Main entrypoint for the CLI application."""
    datasets.disable_progress_bar()

    try:
        app()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


__all__ = [
    "app",
    "console",
    "entrypoint",
    "load_config",
    "DEFAULT_MONGO_URI",
    "DEFAULT_MONGO_DB",
    "DEFAULT_SAE_SERIES",
    "MongoURIOption",
    "MongoDBOption",
    "SAESeriesOption",
]
