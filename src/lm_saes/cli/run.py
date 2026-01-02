from pathlib import Path
from typing import Annotated

import typer

from lm_saes.runners import (
    AnalyzeSAESettings,
    GenerateActivationsSettings,
    TrainSAESettings,
)
from lm_saes.runners import (
    analyze_sae as run_analyze,
)
from lm_saes.runners import (
    generate_activations as run_generate,
)
from lm_saes.runners import (
    train_sae as run_train,
)

from .utils import load_config

app = typer.Typer()


@app.command("generate")
def generate_activations(
    config: Annotated[Path, typer.Argument(help="Path to GenerateActivationsSettings configuration file.")],
) -> None:
    """Generate activations from a language model."""
    cfg = load_config(config)
    settings = GenerateActivationsSettings.model_validate(cfg)
    run_generate(settings)


@app.command("train")
def train(
    config: Annotated[Path, typer.Argument(help="Path to TrainSAESettings configuration file.")],
) -> None:
    """Train a Sparse Autoencoder."""
    cfg = load_config(config)
    settings = TrainSAESettings.model_validate(cfg)
    run_train(settings)


@app.command("analyze")
def analyze(
    config: Annotated[Path, typer.Argument(help="Path to AnalyzeSAESettings configuration file.")],
) -> None:
    """Analyze a trained Sparse Autoencoder."""
    cfg = load_config(config)
    settings = AnalyzeSAESettings.model_validate(cfg)
    run_analyze(settings)
