"""Module for generating activations from language models."""

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings
from torch.distributed.device_mesh import init_device_mesh

from lm_saes.activation.factory import ActivationFactory
from lm_saes.activation.writer import ActivationWriter
from lm_saes.config import (
    ActivationFactoryConfig,
    ActivationFactoryDatasetSource,
    ActivationFactoryTarget,
    ActivationWriterConfig,
    BufferShuffleConfig,
    DatasetConfig,
    LanguageModelConfig,
    MongoDBConfig,
)
from lm_saes.database import MongoClient
from lm_saes.resource_loaders import load_dataset, load_model
from lm_saes.runners.utils import load_config
from lm_saes.utils.logging import get_distributed_logger, setup_logging

logger = get_distributed_logger("runners.generate")


class GenerateActivationsSettings(BaseSettings):
    """Settings for activation generation."""

    model: Optional[LanguageModelConfig] = None
    """Configuration for loading the language model. If `None`, will read from the database."""

    model_name: str
    """Name of the model to load. Use as identifier for the model in the database."""

    dataset: Optional[DatasetConfig] = None
    """Configuration for loading the dataset. If `None`, will read from the database."""

    dataset_name: str
    """Name of the dataset. Use as identifier for the dataset in the database."""

    hook_points: list[str]
    """List of model hook points to capture activations from"""

    output_dir: Path
    """Directory to save activation files"""

    target: ActivationFactoryTarget = ActivationFactoryTarget.ACTIVATIONS_2D
    """Target type for activation generation"""

    model_batch_size: int = 1
    """Batch size for model forward"""

    batch_size: Optional[int] = None
    """Size of the batch for activation generation"""

    buffer_size: Optional[int] = None
    """Size of the buffer for activation generation"""

    buffer_shuffle: Optional[BufferShuffleConfig] = None
    """"Manual seed and device of generator for generating randomperm in buffer"""

    total_tokens: Optional[int] = None
    """Optional total number of tokens to generate"""

    context_size: int = 128
    """Context window size for tokenization"""

    n_samples_per_chunk: Optional[int] = None
    """Number of samples per saved chunk"""

    num_workers: Optional[int] = None
    """Number of workers for parallel writing"""

    format: Literal["pt", "safetensors"] = "safetensors"
    """Format to save activations in ('pt' or 'safetensors')"""

    n_shards: Optional[int] = None
    """Number of shards to split the dataset into. If None, the dataset is split to the world size. Must be larger than the world size."""

    start_shard: int = 0
    """The shard to start writing from"""

    mongo: Optional[MongoDBConfig] = None
    """Configuration for the MongoDB database. If `None`, will not use the database."""

    ignore_token_ids: Optional[list[int]] = None
    """ Tokens to ignore in the activations. """

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""

    def model_post_init(self, __context: dict) -> None:
        """Validate configuration after initialization."""
        if self.mongo is not None:
            assert self.model is not None, "Database not provided. Must manually provide model config."
            assert self.dataset is not None, "Database not provided. Must manually provide dataset config."


def generate_activations(settings: GenerateActivationsSettings) -> None:
    """Generate and save model activations from a dataset.

    Args:
        settings: Configuration settings for activation generation
    """
    # Set up logging
    setup_logging(level="INFO")

    # Initialize device mesh
    device_mesh = (
        init_device_mesh(
            device_type=settings.device_type,
            mesh_shape=(int(os.environ.get("WORLD_SIZE", 1)), 1),
            mesh_dim_names=("data", "model"),
        )
        if os.environ.get("WORLD_SIZE") is not None
        else None
    )

    logger.info(f"Device mesh initialized: {device_mesh}")

    mongo_client = MongoClient(settings.mongo) if settings.mongo is not None else None
    if mongo_client:
        logger.info("MongoDB client initialized")

    # Load configurations
    logger.info("Loading model and dataset configurations")
    model_cfg = load_config(
        config=settings.model, name=settings.model_name, mongo_client=mongo_client, config_type="model"
    )

    dataset_cfg = load_config(
        config=settings.dataset, name=settings.dataset_name, mongo_client=mongo_client, config_type="dataset"
    )

    # Load model and dataset
    logger.info("Loading model and dataset")
    model = load_model(model_cfg)
    dataset, metadata = load_dataset(
        dataset_cfg,
        device_mesh=device_mesh,
        n_shards=settings.n_shards,
        start_shard=settings.start_shard,
    )

    logger.info(f"Model loaded: {settings.model_name}")
    logger.info(f"Dataset loaded: {settings.dataset_name}")

    # Configure activation generation
    logger.info("Configuring activation factory")
    factory_cfg = ActivationFactoryConfig(
        sources=[ActivationFactoryDatasetSource(name=settings.dataset_name)],
        target=settings.target,
        hook_points=settings.hook_points,
        context_size=settings.context_size,
        model_batch_size=settings.model_batch_size,
        batch_size=settings.batch_size,
        buffer_size=settings.buffer_size,
        buffer_shuffle=settings.buffer_shuffle,
        ignore_token_ids=settings.ignore_token_ids,
    )

    # Configure activation writer
    logger.info("Configuring activation writer")
    writer_cfg = ActivationWriterConfig(
        hook_points=settings.hook_points,
        total_generating_tokens=settings.total_tokens,
        n_samples_per_chunk=settings.n_samples_per_chunk,
        cache_dir=settings.output_dir,
        format=settings.format,
        num_workers=settings.num_workers,
    )

    # Create factory and writer
    factory = ActivationFactory(factory_cfg)
    writer = ActivationWriter(writer_cfg)

    logger.info("Starting activation generation and writing")
    # Generate and write activations
    activations = factory.process(
        model=model, model_name=settings.model_name, datasets={settings.dataset_name: (dataset, metadata)}
    )
    writer.process(activations, device_mesh=device_mesh, start_shard=settings.start_shard)

    logger.info("Activation generation completed successfully")
