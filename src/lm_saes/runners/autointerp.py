"""Module for automatic interpretation of SAE features."""

import concurrent.futures
from functools import lru_cache
from typing import Any, Optional

from datasets import Dataset
from pydantic_settings import BaseSettings

from lm_saes.analysis.feature_interpreter import AutoInterpConfig, FeatureInterpreter
from lm_saes.config import LanguageModelConfig, MongoDBConfig
from lm_saes.database import MongoClient
from lm_saes.resource_loaders import load_dataset_shard, load_model


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

    feature_range: Optional[list[int]] = None
    """Range of feature indices to interpret [start, end]. If None, will interpret all features."""

    top_k_features: Optional[int] = None
    """Number of top activating features to interpret. If None, will use the features or feature_range."""

    analysis_name: str = "default"
    """Name of the analysis to use for interpretation."""

    max_workers: int = 10
    """Maximum number of workers to use for interpretation."""


def interpret_feature(args: dict[str, Any]):
    settings: AutoInterpSettings = args["settings"]
    feature_indices: list[int] = args["feature_indices"]
    print(f"Interpreting {len(feature_indices)} features")

    @lru_cache(maxsize=None)
    def get_dataset(dataset_name: str, shard_idx: int, n_shards: int) -> Dataset:
        dataset_cfg = mongo_client.get_dataset_cfg(dataset_name)
        assert dataset_cfg is not None, f"Dataset {dataset_name} not found"
        dataset = load_dataset_shard(dataset_cfg, shard_idx, n_shards)
        return dataset

    mongo_client = MongoClient(settings.mongo)
    language_model = load_model(settings.model)
    interpreter = FeatureInterpreter(settings.auto_interp, mongo_client)
    for result in interpreter.interpret_features(
        sae_name=settings.sae_name,
        sae_series=settings.sae_series,
        feature_indices=feature_indices,
        model=language_model,
        datasets=get_dataset,
        analysis_name=settings.analysis_name,
    ):
        interpretation = {
            "text": result["explanation"],
            "validation": [
                {"method": eval_result["method"], "passed": eval_result["passed"], "detail": eval_result}
                for eval_result in result["evaluations"]
            ],
            "complexity": result["complexity"],
            "consistency": result["consistency"],
            "detail": result["explanation_details"],
            "passed": result["passed"],
            "time": result["time"],
        }
        print(
            f"Updating feature {result['feature_index']}\nTime: {result['time']}\nExplanation: {interpretation['text']}\nComplexity: {interpretation['complexity']}\nConsistency: {interpretation['consistency']}\nPassed: {interpretation['passed']}\n\n"
        )
        mongo_client.update_feature(
            settings.sae_name, result["feature_index"], {"interpretation": interpretation}, settings.sae_series
        )


def auto_interp(settings: AutoInterpSettings) -> None:
    """Automatically interpret SAE features using LLMs.

    Args:
        settings: Configuration settings for auto-interpretation
    """
    # Set up MongoDB client
    mongo_client = MongoClient(settings.mongo)

    # Determine which features to interpret
    if settings.top_k_features:
        # Get top k most frequently activating features
        act_times = mongo_client.get_feature_act_times(settings.sae_name, settings.sae_series, settings.analysis_name)
        if not act_times:
            raise ValueError(f"No feature activation times found for {settings.sae_name}/{settings.sae_series}")
        sorted_features = sorted(act_times.items(), key=lambda x: x[1], reverse=True)
        feature_indices = [idx for idx, _ in sorted_features[: settings.top_k_features]]
    elif settings.feature_range:
        # Use feature range
        feature_indices = list(range(settings.feature_range[0], settings.feature_range[1] + 1))
    elif settings.features:
        # Use specific features
        feature_indices = settings.features
    else:
        # Use all features (be careful, this could be a lot!)
        max_feature_acts = mongo_client.get_max_feature_acts(
            settings.sae_name, settings.sae_series, settings.analysis_name
        )
        if not max_feature_acts:
            raise ValueError(f"No feature activations found for {settings.sae_name}/{settings.sae_series}")
        feature_indices = list(max_feature_acts.keys())

    # Load resources
    print(f"Loading SAE model: {settings.sae_name}/{settings.sae_series}")
    print(f"Loading language model: {settings.model_name}")

    chunk_size = len(feature_indices) // settings.max_workers + 1
    feature_batches = [feature_indices[i : i + chunk_size] for i in range(0, len(feature_indices), chunk_size)]
    args_batches = [{"feature_indices": feature_indices, "settings": settings} for feature_indices in feature_batches]

    with concurrent.futures.ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        list(executor.map(interpret_feature, args_batches))

    print("Done!")
