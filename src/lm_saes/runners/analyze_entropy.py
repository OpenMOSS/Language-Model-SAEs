"""Module for analyzing feature activation entropy in SAE models."""

from typing import Any, Optional
import json

import torch
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from torch.distributed.device_mesh import init_device_mesh

from lm_saes.activation.factory import ActivationFactory
from lm_saes.analysis.feature_activation_entropy import FeatureEntropyAnalyzer
from lm_saes.config import (
    ActivationFactoryConfig,
    BaseSAEConfig,
    FeatureAnalyzerConfig,
    DatasetConfig,
    LanguageModelConfig,
    MongoDBConfig,
)
from lm_saes.resource_loaders import load_dataset, load_model
from lm_saes.runners.utils import load_config
from lm_saes.database import MongoClient
from lm_saes.sae import SparseAutoEncoder
from lm_saes.clt import CrossLayerTranscoder
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.utils.logging import get_distributed_logger, setup_logging

logger = get_distributed_logger("runners.analyze_entropy")


class AnalyzeEntropySettings(BaseSettings):
    """Settings for analyzing feature activation entropy in a Sparse Autoencoder."""

    model_config = ConfigDict(extra='ignore')

    sae: BaseSAEConfig
    """Configuration for the SAE model architecture and parameters"""

    sae_name: str
    """Name of the SAE model. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model. Use as identifier for the SAE model in the database."""

    activation_factory: ActivationFactoryConfig
    """Configuration for generating activations"""

    model: Optional[LanguageModelConfig] = None
    """Configuration for the language model. Required if using dataset sources."""

    model_name: Optional[str] = None
    """Name of the tokenizer to load. LORSA may require a tokenizer to get the modality indices."""

    datasets: Optional[dict[str, Optional[DatasetConfig]]] = None
    """Name to dataset config mapping. Required if using dataset sources."""

    dataset: Optional[DatasetConfig] = None
    """Single dataset configuration (alternative to datasets dict)"""

    dataset_name: Optional[str] = None
    """Name of the dataset (used with dataset parameter)"""

    analyzer: FeatureAnalyzerConfig
    """Configuration for feature analysis"""

    amp_dtype: torch.dtype = torch.bfloat16
    """The dtype to use for automatic mixed precision"""

    mongo: MongoDBConfig
    """Configuration for the MongoDB database."""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""

    output_file: Optional[str] = None
    """Optional file path to save entropy analysis results as JSON"""


@torch.no_grad()
def analyze_entropy(settings: AnalyzeEntropySettings) -> None:
    """Analyze feature activation entropy in a SAE model.

    This function computes entropy metrics for each feature by analyzing how
    feature activations distribute across different chess piece categories.

    Args:
        settings: Configuration settings for entropy analysis
    """
    # Set up logging
    setup_logging(level="INFO")

    device_mesh = (
        init_device_mesh(
            device_type=settings.device_type,
            mesh_shape=(settings.model_parallel_size,),
            mesh_dim_names=("model",),
        )
        if settings.model_parallel_size > 1
        else None
    )

    logger.info(f"Device mesh initialized: {device_mesh}")

    mongo_client = MongoClient(settings.mongo)
    logger.info("MongoDB client initialized")

    # Load configurations
    logger.info(f"Loading model configuration: {settings.model_name}")
    model_cfg = load_config(
        config=settings.model,
        name=settings.model_name,
        mongo_client=mongo_client,
        config_type="model",
        required=False,
    )

    # Handle both single dataset and datasets dict
    if settings.dataset is not None and settings.dataset_name is not None:
        # Single dataset mode
        dataset_cfg = load_config(
            config=settings.dataset,
            name=settings.dataset_name,
            mongo_client=mongo_client,
            config_type="dataset",
        )
        dataset_cfgs = {settings.dataset_name: dataset_cfg}
    elif settings.datasets is not None:
        # Multiple datasets mode
        dataset_cfgs = {
            dataset_name: load_config(
                config=dataset_cfg,
                name=dataset_name,
                mongo_client=mongo_client,
                config_type="dataset",
            )
            for dataset_name, dataset_cfg in settings.datasets.items()
        }
    else:
        dataset_cfgs = None

    model = load_model(model_cfg) if model_cfg is not None else None
    datasets = (
        {
            dataset_name: load_dataset(dataset_cfg, device_mesh=device_mesh)
            for dataset_name, dataset_cfg in dataset_cfgs.items()
        }
        if dataset_cfgs is not None
        else None
    )
    
    # Get dataset path from the dataset config
    dataset_path = None
    if settings.dataset is not None and settings.dataset.is_dataset_on_disk:
        dataset_path = settings.dataset.dataset_name_or_path
    elif dataset_cfgs is not None and len(dataset_cfgs) > 0:
        # Use the first dataset's path if available
        first_cfg = next(iter(dataset_cfgs.values()))
        if first_cfg.is_dataset_on_disk:
            dataset_path = first_cfg.dataset_name_or_path

    activation_factory = ActivationFactory(settings.activation_factory)

    logger.info(f"Loading {settings.sae.sae_type} model")
    sae_cls = {
        "sae": SparseAutoEncoder,
        "clt": CrossLayerTranscoder,
        "lorsa": LowRankSparseAttention,
    }[settings.sae.sae_type]
    sae = sae_cls.from_config(settings.sae, device_mesh=device_mesh)

    logger.info(f"{settings.sae.sae_type} model loaded: {type(sae).__name__}")

    analyzer = FeatureEntropyAnalyzer(settings.analyzer)
    logger.info("Feature entropy analyzer initialized")

    logger.info("Processing activations for entropy analysis")

    with torch.amp.autocast(device_type=settings.device_type, dtype=settings.amp_dtype):
        result = analyzer.analyze_entropy(
            activation_factory,
            sae=sae,
            device_mesh=device_mesh,
            activation_factory_process_kwargs={
                "model": model,
                "model_name": settings.model_name,
                "datasets": datasets,
            },
            dataset_path=dataset_path,
        )

    logger.info("Entropy analysis completed")
    logger.info(f"Mean entropy across all features: {result['mean_entropy']:.4f}")
    logger.info(f"Number of features analyzed: {result['n_features']}")

    # Save to MongoDB
    logger.info("Saving entropy analysis results to MongoDB")
    start_idx = 0 if device_mesh is None else device_mesh.get_local_rank("model") * result["n_features"]
    
    # Store entropy results in MongoDB
    mongo_client.db[f"entropy_analysis_{settings.sae_series}"].insert_one({
        "sae_name": settings.sae_name,
        "sae_series": settings.sae_series,
        "mean_entropy": result["mean_entropy"],
        "n_features": result["n_features"],
        "per_feature_entropy": result["per_feature_entropy"],
        "start_idx": start_idx,
    })

    # Optionally save detailed results to file
    if settings.output_file:
        logger.info(f"Saving detailed results to {settings.output_file}")
        with open(settings.output_file, "w") as f:
            json.dump(
                {
                    "sae_name": settings.sae_name,
                    "sae_series": settings.sae_series,
                    "mean_entropy": result["mean_entropy"],
                    "n_features": result["n_features"],
                    "per_feature_entropy": result["per_feature_entropy"],  # All features' entropy values
                    "feature_details": result["feature_details"],  # All features' detailed information
                },
                f,
                indent=2,
            )

    logger.info(f"{settings.sae.sae_type} entropy analysis completed successfully")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_entropy.py <config_file>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config from file
    import yaml
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    settings = AnalyzeEntropySettings(**config_dict)
    analyze_entropy(settings)

