from typing import Annotated

import typer

from lm_saes.database import MongoClient, MongoDBConfig

from .common import (
    DEFAULT_MONGO_DB,
    DEFAULT_MONGO_URI,
    DEFAULT_SAE_SERIES,
    MongoDBOption,
    MongoURIOption,
    SAESeriesOption,
)

app = typer.Typer(help="Remove database records.")


@app.command("sae")
def remove_sae(
    name: Annotated[str, typer.Argument(help="Name of the SAE to remove.")],
    series: SAESeriesOption = DEFAULT_SAE_SERIES,
    mongo_uri: MongoURIOption = DEFAULT_MONGO_URI,
    mongo_db: MongoDBOption = DEFAULT_MONGO_DB,
) -> None:
    """Remove an SAE record and its associated features and analyses."""
    client = MongoClient(MongoDBConfig(mongo_uri=mongo_uri, mongo_db=mongo_db))
    client.remove_sae(sae_name=name, sae_series=series)
    typer.echo(f"SAE '{name}' (series: {series}) removed successfully.")


@app.command("analysis")
def remove_analysis(
    sae_name: Annotated[str, typer.Argument(help="Name of the SAE.")],
    name: Annotated[str, typer.Argument(help="Name of the analysis to remove.")],
    sae_series: SAESeriesOption = DEFAULT_SAE_SERIES,
    mongo_uri: MongoURIOption = DEFAULT_MONGO_URI,
    mongo_db: MongoDBOption = DEFAULT_MONGO_DB,
) -> None:
    """Remove an analysis record from an SAE."""
    client = MongoClient(MongoDBConfig(mongo_uri=mongo_uri, mongo_db=mongo_db))
    client.remove_feature_analysis(name=name, sae_name=sae_name, sae_series=sae_series)
    typer.echo(f"Analysis '{name}' removed from SAE '{sae_name}' (series: {sae_series}).")
