import itertools
from functools import lru_cache
from typing import Annotated, Any, Generator

import numpy as np
import torch
import typer
from datasets import Dataset
from rich.panel import Panel
from rich.text import Text

from lm_saes.analysis.samples import TokenizedSample
from lm_saes.cli.common import (
    DEFAULT_MONGO_DB,
    DEFAULT_MONGO_URI,
    DEFAULT_SAE_SERIES,
    MongoDBOption,
    MongoURIOption,
    SAESeriesOption,
    console,
)
from lm_saes.config import MongoDBConfig
from lm_saes.database import FeatureAnalysisSampling, MongoClient
from lm_saes.resource_loaders import LanguageModel, load_dataset_shard, load_model

app = typer.Typer(help="Display feature activation samples and statistics.")


def _get_activation_color(activation: float, max_activation: float) -> str:
    """Get a color based on activation intensity (yellow to red gradient)."""
    if max_activation <= 0:
        return "white"
    intensity = min(activation / max_activation, 1.0)
    if intensity < 0.01:
        return "white"
    if intensity < 0.33:
        return "yellow"
    elif intensity < 0.66:
        return "orange1"
    else:
        return "red"


def _extract_samples(
    sampling: FeatureAnalysisSampling,
    client: MongoClient,
    max_activation: float,
) -> Generator[TokenizedSample, Any, Any]:
    """Extract a single sample from sampling data and convert to TokenizedSample."""
    # Get sparse feature acts for this sample
    feature_acts_indices = sampling.feature_acts_indices
    feature_acts_values = sampling.feature_acts_values
    feature_acts = torch.sparse_coo_tensor(
        torch.tensor(feature_acts_indices),
        torch.tensor(feature_acts_values),
        (int(np.max(feature_acts_indices[0]) + 1), 2048),
    )
    feature_acts = feature_acts.to_dense()

    @lru_cache(maxsize=8)
    def _load_model(model_name: str) -> LanguageModel:
        cfg = client.get_model_cfg(name=model_name)
        assert cfg is not None, f"Model {model_name} not found"
        cfg.tokenizer_only = True
        return load_model(cfg)

    @lru_cache(maxsize=8)
    def _load_dataset_shard(dataset_name: str, shard_idx: int, n_shards: int) -> Dataset:
        cfg = client.get_dataset_cfg(name=dataset_name)
        assert cfg is not None, f"Dataset {dataset_name} not found"
        return load_dataset_shard(cfg, shard_idx, n_shards)

    for feature_acts_i, context_idx, dataset_name, model_name, shard_idx, n_shards in zip(
        feature_acts,
        sampling.context_idx,
        sampling.dataset_name,
        sampling.model_name,
        sampling.shard_idx if sampling.shard_idx is not None else [0] * len(sampling.dataset_name),
        sampling.n_shards if sampling.n_shards is not None else [1] * len(sampling.dataset_name),
    ):
        dataset = _load_dataset_shard(dataset_name, shard_idx, n_shards)
        model = _load_model(model_name)

        data = dataset[context_idx]
        origins = model.trace({k: [v] for k, v in data.items()}, n_context=2048)[0]

        tokenized_sample = TokenizedSample.construct(data["text"], feature_acts_i, origins, max_activation)

        yield tokenized_sample


@app.command("feature")
def show_feature(
    sae_name: Annotated[str, typer.Argument(help="Name of the SAE.")],
    feature_index: Annotated[int, typer.Argument(help="Index of the feature to display.")],
    sae_series: SAESeriesOption = DEFAULT_SAE_SERIES,
    analysis_name: Annotated[str | None, typer.Option("--analysis", "-a", help="Name of the analysis.")] = None,
    num_samples: Annotated[int, typer.Option("--num-samples", "-n", help="Number of samples to display.")] = 5,
    visible_range: Annotated[
        int, typer.Option("--visible-range", "-r", help="Number of tokens around max activation to show.")
    ] = 20,
    mongo_uri: MongoURIOption = DEFAULT_MONGO_URI,
    mongo_db: MongoDBOption = DEFAULT_MONGO_DB,
) -> None:
    """Display top activation samples for a feature with highlighted tokens."""
    client = MongoClient(MongoDBConfig(mongo_uri=mongo_uri, mongo_db=mongo_db))

    # Fetch feature data
    features = client.list_features(sae_name=sae_name, sae_series=sae_series, indices=[feature_index])
    if not features:
        console.print(f"[red]Feature {feature_index} not found in SAE '{sae_name}'[/red]")
        raise typer.Exit(1)

    feature = features[0]

    # Find the analysis
    analysis = None
    if analysis_name:
        analysis = next((a for a in feature.analyses if a.name == analysis_name), None)
    else:
        analysis = next((a for a in feature.analyses if a.name == "default"), None)
        if analysis is None:
            analysis = next((a for a in feature.analyses), None)

    if analysis is None:
        console.print(f"[red]No analysis found for feature {feature_index}[/red]")
        raise typer.Exit(1)

    # Print feature header
    console.print()
    header = Text()
    header.append(f"Feature #{feature_index}", style="bold cyan")
    header.append(f" @ {sae_name}", style="dim")
    if feature.interpretation:
        header.append(f"\n{feature.interpretation.get('text', '')}", style="italic")
    console.print(Panel(header, title="Feature Info", border_style="cyan"))

    # Print statistics
    stats = Text()
    stats.append("Activation Times: ", style="bold")
    stats.append(f"{analysis.act_times:,}\n")
    stats.append("Max Activation: ", style="bold")
    stats.append(f"{analysis.max_feature_acts:.4f}\n")
    if analysis.n_analyzed_tokens:
        stats.append("Analyzed Tokens: ", style="bold")
        stats.append(f"{analysis.n_analyzed_tokens:,}")
    console.print(Panel(stats, title="Statistics", border_style="green"))

    # Find top_activations sampling
    sampling = next((s for s in analysis.samplings if s.name == "top_activations"), None)
    if sampling is None:
        console.print("[yellow]No top activation samples available.[/yellow]")
        raise typer.Exit(0)

    # Display samples with color legend
    legend = Text()
    legend.append(f"Top {num_samples} Activation Samples", style="bold")
    legend.append("  (")
    legend.append("Weak", style="yellow on grey23")
    legend.append(" | ")
    legend.append("Medium", style="orange1 on grey23")
    legend.append(" | ")
    legend.append("Strong", style="red on grey23")
    legend.append(")")
    console.print()
    console.print(legend)
    console.print()

    for i, sample in enumerate(
        itertools.islice(_extract_samples(sampling, client, analysis.max_feature_acts), num_samples)
    ):
        sample = sample.to_max_activation_surrounding(visible_range)
        rich_text = Text()
        for seg in sample.segments:
            if seg.activation > 0.01:
                rich_text.append(
                    seg.text, style=f"bold {_get_activation_color(seg.activation, analysis.max_feature_acts)} on grey23"
                )
            else:
                rich_text.append(seg.text)

        sample_title = f"Sample {i + 1} (max act: {sample.sample_max_activation:.4f})"
        console.print(Panel(rich_text, title=sample_title, border_style="blue"))
