"""Module for analyzing SAE models."""

from typing import Optional

import torch
from pydantic_settings import BaseSettings
from torch.distributed.device_mesh import init_device_mesh

from lm_saes.activation.factory import ActivationFactory
from lm_saes.analysis.feature_analyzer import FeatureAnalyzer
from lm_saes.config import (
    ActivationFactoryConfig,
    BaseSAEConfig,
    CrossCoderConfig,
    FeatureAnalyzerConfig,
    LanguageModelConfig,
    MongoDBConfig,
)
from lm_saes.crosscoder import CrossCoder
from lm_saes.database import MongoClient
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.logging import get_distributed_logger, setup_logging

logger = get_distributed_logger("runners.analyze")


class AnalyzeSAESettings(BaseSettings):
    """Settings for analyzing a Sparse Autoencoder."""

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

    analyzer: FeatureAnalyzerConfig
    """Configuration for feature analysis"""

    mongo: MongoDBConfig
    """Configuration for the MongoDB database."""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


def analyze_sae(settings: AnalyzeSAESettings) -> None:
    """Analyze a SAE model.

    Args:
        settings: Configuration settings for SAE analysis
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

    activation_factory = ActivationFactory(settings.activation_factory)

    logger.info("Loading SAE model")
    if isinstance(settings.sae, CrossCoderConfig):
        sae = CrossCoder.from_config(settings.sae, device_mesh=device_mesh)
    else:
        sae = SparseAutoEncoder.from_config(settings.sae, device_mesh=device_mesh)

    logger.info(f"SAE model loaded: {type(sae).__name__}")

    analyzer = FeatureAnalyzer(settings.analyzer)
    logger.info("Feature analyzer initialized")

    logger.info("Processing activations for analysis")
    activations = activation_factory.process()
    result = analyzer.analyze_chunk(activations, sae=sae, device_mesh=device_mesh)

    logger.info("Analysis completed, saving results to MongoDB")
    start_idx = 0 if device_mesh is None else device_mesh.get_local_rank("model") * len(result)
    mongo_client.add_feature_analysis(
        name="default", sae_name=settings.sae_name, sae_series=settings.sae_series, analysis=result, start_idx=start_idx
    )

    logger.info("SAE analysis completed successfully")


class AnalyzeCrossCoderSettings(BaseSettings):
    """Settings for analyzing a CrossCoder model."""

    sae: CrossCoderConfig
    """Configuration for the CrossCoder model architecture and parameters"""

    sae_name: str
    """Name of the SAE model. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model. Use as identifier for the SAE model in the database."""

    activation_factories: list[ActivationFactoryConfig]
    """Configuration for generating activations"""

    analyzer: FeatureAnalyzerConfig
    """Configuration for feature analysis"""

    feature_analysis_name: str = "default"
    """Name of the feature analysis."""

    mongo: MongoDBConfig
    """Configuration for the MongoDB database."""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


@torch.no_grad()
def analyze_crosscoder(settings: AnalyzeCrossCoderSettings) -> None:
    """Analyze a CrossCoder model. The key difference to analyze_sae is that the activation factories are a list of ActivationFactoryConfig, one for each head; and the analyzing contains a device mesh transformation from head parallelism to model (feature) parallelism.

    Args:
        settings: Configuration settings for CrossCoder analysis
    """
    # Set up logging
    setup_logging(level="INFO")

    assert (
        len(settings.activation_factories) * len(settings.activation_factories[0].hook_points) == settings.sae.n_heads
    ), "Total number of hook points must match the number of heads in the CrossCoder"

    parallel_size = len(settings.activation_factories)

    logger.info(f"Analyzing CrossCoder with {settings.sae.n_heads} heads, {parallel_size} parallel size")

    crosscoder_device_mesh = init_device_mesh(
        device_type=settings.device_type,
        mesh_shape=(parallel_size,),
        mesh_dim_names=("head",),
    )

    device_mesh = init_device_mesh(
        device_type=settings.device_type,
        mesh_shape=(parallel_size,),
        mesh_dim_names=("model",),
    )

    logger.info("Device meshes initialized for CrossCoder analysis")

    mongo_client = MongoClient(settings.mongo)
    logger.info("MongoDB client initialized")

    logger.info("Setting up activation factory for CrossCoder head")
    activation_factory = ActivationFactory(settings.activation_factories[crosscoder_device_mesh.get_local_rank("head")])

    logger.info("Loading CrossCoder model")
    sae = CrossCoder.from_config(settings.sae, device_mesh=crosscoder_device_mesh)

    logger.info("Feature analyzer initialized")
    analyzer = FeatureAnalyzer(settings.analyzer)

    logger.info("Processing activations for CrossCoder analysis")
    activations = activation_factory.process()
    result = analyzer.analyze_chunk(activations, sae=sae, device_mesh=device_mesh)

    logger.info("CrossCoder analysis completed, saving results to MongoDB")
    start_idx = 0 if device_mesh is None else device_mesh.get_local_rank("model") * len(result)
    mongo_client.add_feature_analysis(
        name=settings.feature_analysis_name,
        sae_name=settings.sae_name,
        sae_series=settings.sae_series,
        analysis=result,
        start_idx=start_idx,
    )

    logger.info("CrossCoder analysis completed successfully")
