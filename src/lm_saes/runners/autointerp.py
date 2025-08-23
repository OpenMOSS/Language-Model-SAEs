"""Module for automatic interpretation of SAE features."""

import os
from functools import lru_cache
from typing import Optional

from datasets import Dataset
from pydantic_settings import BaseSettings
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from lm_saes.analysis.feature_interpreter import AutoInterpConfig, FeatureInterpreter
from lm_saes.config import LanguageModelConfig, MongoDBConfig
from lm_saes.database import MongoClient
from lm_saes.resource_loaders import load_dataset_shard, load_model
from lm_saes.utils.logging import get_logger, setup_logging

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

    feature_range: Optional[tuple[int, int]] = None
    """Range of feature indices to interpret [start, end]. If None, will interpret all features."""

    top_k_features: Optional[int] = None
    """Number of top activating features to interpret. If None, will use the features or feature_range."""

    analysis_name: str = "default"
    """Name of the analysis to use for interpretation."""


def interpret_feature(
    settings: AutoInterpSettings,
    feature_indices: list[int],
    mongo_client: MongoClient,
    device_mesh: Optional[DeviceMesh],
) -> None:
    """Interpret a feature using the language model.

    Args:
        settings: Configuration settings for auto-interpretation
        feature_indices: List of feature indices to interpret
        mongo_client: MongoDB client
        device_mesh: Device mesh
    """

    @lru_cache(maxsize=None)
    def get_dataset(dataset_name: str, shard_idx: int, n_shards: int) -> Dataset:
        dataset_cfg = mongo_client.get_dataset_cfg(dataset_name)
        assert dataset_cfg is not None, f"Dataset {dataset_name} not found"
        dataset = load_dataset_shard(dataset_cfg, shard_idx, n_shards)
        return dataset

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
        logger.info(
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
    setup_logging(level="INFO")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device_mesh = (
        init_device_mesh(
            device_type="cpu",
            mesh_shape=(world_size,),
            mesh_dim_names=("model",),
        )
        if world_size > 1
        else None
    )

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
    logger.info(f"Loading SAE model: {settings.sae_name}/{settings.sae_series}")
    logger.info(f"Loading language model: {settings.model_name}")

    if device_mesh is not None:
        local_feature_indices = feature_indices[device_mesh.get_rank() :: world_size]
    else:
        local_feature_indices = feature_indices

    interpret_feature(settings, local_feature_indices, mongo_client, device_mesh)

    logger.info("Done!")
