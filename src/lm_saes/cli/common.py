import os
from typing import Annotated

import typer
from rich.console import Console

# Environment variable defaults
DEFAULT_MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DEFAULT_MONGO_DB = os.environ.get("MONGO_DB", "mechinterp")
DEFAULT_SAE_SERIES = os.environ.get("SAE_SERIES", "default")

# Shared console instance
console = Console()

# Common options
MongoURIOption = Annotated[
    str,
    typer.Option(
        "--mongo-uri",
        envvar="MONGO_URI",
        help="MongoDB connection URI.",
    ),
]

MongoDBOption = Annotated[
    str,
    typer.Option(
        "--mongo-db",
        envvar="MONGO_DB",
        help="MongoDB database name.",
    ),
]

SAESeriesOption = Annotated[
    str,
    typer.Option(
        "--series",
        "-s",
        envvar="SAE_SERIES",
        help="SAE series identifier.",
    ),
]
