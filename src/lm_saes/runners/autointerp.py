"""Module for automatic interpretation of SAE features."""

from functools import lru_cache
from typing import Optional

from datasets import Dataset
from pydantic_settings import BaseSettings
from tqdm import tqdm

from lm_saes.analysis.auto_interp import AutoInterpConfig, FeatureInterpreter
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
    model = load_model(settings.model)

    # Load the dataset for non-activating examples
    @lru_cache(maxsize=None)
    def get_dataset(dataset_name: str, shard_idx: int, n_shards: int) -> Dataset:
        dataset_cfg = mongo_client.get_dataset_cfg(dataset_name)
        assert dataset_cfg is not None, f"Dataset {dataset_name} not found"
        dataset = load_dataset_shard(dataset_cfg, shard_idx, n_shards)
        return dataset

    # Create the interpreter
    interpreter = FeatureInterpreter(settings.auto_interp, mongo_client)

    # Call interpret_feature with all features at once
    print(f"Interpreting {len(feature_indices)} features...")
    total_interpreted = 0
    proc_bar = tqdm(total=len(feature_indices), desc="Interpreting")
    for result in interpreter.interpret_features(
        sae_name=settings.sae_name,
        sae_series=settings.sae_series,
        feature_indices=feature_indices,
        model=model,
        datasets=get_dataset,
        analysis_name=settings.analysis_name,
        max_workers=settings.max_workers,
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
        }
        print(
            f"Updating feature {result['feature_index']}\nExplanation: {interpretation['text']}\nComplexity: {interpretation['complexity']}\nConsistency: {interpretation['consistency']}\nPassed: {interpretation['passed']}\n\n"
        )
        mongo_client.update_feature(
            settings.sae_name, result["feature_index"], {"interpretation": interpretation}, settings.sae_series
        )
        total_interpreted += 1
        proc_bar.update(1)
        proc_bar.set_postfix(feature_index=result["feature_index"])
    proc_bar.close()
    print(f"Completed interpreting {total_interpreted} features")
