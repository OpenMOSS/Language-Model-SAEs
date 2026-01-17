"""Module for automatic interpretation of SAE features."""

import asyncio
from functools import lru_cache
from typing import Optional

from datasets import Dataset
from pydantic_settings import BaseSettings
from tqdm.asyncio import tqdm

from lm_saes.analysis.autointerp import AutoInterpConfig, FeatureInterpreter
from lm_saes.backend.language_model import LanguageModelConfig
from lm_saes.database import MongoClient, MongoDBConfig
from lm_saes.resource_loaders import load_dataset_shard, load_model
from lm_saes.utils.logging import get_logger

logger = get_logger("runners.autointerp")


class AutoInterpSettings(BaseSettings):
    """Settings for automatic interpretation of SAE features."""

    sae_name: str
    """Name of the SAE model to interpret. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model to interpret. Use as identifier for the SAE model in the database."""

    model: LanguageModelConfig
    """Configuration for the language model used to generate activations."""

    model_name: str
    """Name of the model to load."""

    auto_interp: AutoInterpConfig
    """Configuration for the auto-interpretation process."""

    mongo: MongoDBConfig
    """Configuration for the MongoDB database."""

    features: Optional[list[int]] = None
    """List of specific feature indices to interpret. If None, will interpret all features."""

    analysis_name: str = "default"
    """Name of the analysis to use for interpretation."""

    max_workers: int = 10
    """Maximum number of workers to use for interpretation."""


async def interpret_feature(settings: AutoInterpSettings, show_progress: bool = True):
    """Interpret features using async API calls for maximum concurrency.

    Args:
        settings: Configuration for feature interpretation
        show_progress: Whether to show progress bar (requires tqdm)
    """

    @lru_cache(maxsize=None)
    def get_dataset(dataset_name: str, shard_idx: int, n_shards: int) -> Dataset:
        dataset_cfg = mongo_client.get_dataset_cfg(dataset_name)
        assert dataset_cfg is not None, f"Dataset {dataset_name} not found"
        dataset = load_dataset_shard(dataset_cfg, shard_idx, n_shards)
        return dataset

    mongo_client = MongoClient(settings.mongo)
    language_model = load_model(settings.model)
    interpreter = FeatureInterpreter(settings.auto_interp, mongo_client)

    # Set up progress tracking
    progress_bar = None
    processed_count = 0
    total_count = None

    def progress_callback(processed: int, total: int, current_feature: int) -> None:
        """Update progress bar and log progress.

        Args:
            processed: Number of features processed (completed + skipped + failed)
            total: Total number of features to process
            current_feature: Index of the feature currently being processed
        """
        nonlocal processed_count, total_count, progress_bar
        processed_count = processed
        if total_count is None:
            total_count = total
            if show_progress:
                progress_bar = tqdm(
                    total=total,
                    desc="Interpreting features",
                    unit="feature",
                    dynamic_ncols=True,
                    initial=0,
                )

        if progress_bar is not None:
            progress_bar.n = processed
            progress_bar.refresh()
            progress_bar.set_postfix({"current": current_feature})

    async for result in interpreter.interpret_features(
        sae_name=settings.sae_name,
        sae_series=settings.sae_series,
        model=language_model,
        datasets=get_dataset,
        analysis_name=settings.analysis_name,
        feature_indices=settings.features,
        max_concurrent=settings.max_workers,
        progress_callback=progress_callback,
    ):
        if result["explanation"] is not None:
            interpretation = {
                "text": result["explanation"],
            }
        else:
            interpretation = None
        mongo_client.update_feature(
            settings.sae_name, result["feature_index"], {"interpretation": interpretation}, settings.sae_series
        )
    if progress_bar is not None:
        progress_bar.close()
        logger.info(f"Completed interpretation: {processed_count}/{total_count} features processed")


def auto_interp(settings: AutoInterpSettings):
    """Synchronous wrapper for interpret_feature.

    Args:
        settings: Configuration for feature interpretation
    """
    asyncio.run(interpret_feature(settings))
