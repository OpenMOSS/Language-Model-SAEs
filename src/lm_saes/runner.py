import os
from pathlib import Path
from typing import Literal, Optional, TypeVar, overload

import wandb
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer

from lm_saes.activation.factory import ActivationFactory
from lm_saes.activation.writer import ActivationWriter
from lm_saes.analysis.feature_analyzer import FeatureAnalyzer
from lm_saes.config import (
    ActivationFactoryConfig,
    ActivationFactoryDatasetSource,
    ActivationFactoryTarget,
    ActivationWriterConfig,
    BaseSAEConfig,
    BufferShuffleConfig,
    DatasetConfig,
    FeatureAnalyzerConfig,
    InitializerConfig,
    LanguageModelConfig,
    MongoDBConfig,
    TrainerConfig,
    WandbConfig,
)
from lm_saes.crosscoder import CrossCoder
from lm_saes.database import MongoClient
from lm_saes.initializer import Initializer
from lm_saes.mixcoder import MixCoder
from lm_saes.resource_loaders import load_dataset, load_model
from lm_saes.sae import SparseAutoEncoder
from lm_saes.trainer import Trainer

T = TypeVar("T")


@overload
def load_config(
    config: Optional[T],
    name: Optional[str],
    mongo_client: Optional[MongoClient],
    config_type: str,
    required: Literal[True] = True,
) -> T: ...


@overload
def load_config(
    config: Optional[T],
    name: Optional[str],
    mongo_client: Optional[MongoClient],
    config_type: str,
    required: Literal[False] = False,
) -> Optional[T]: ...


def load_config(
    config: Optional[T],
    name: Optional[str],
    mongo_client: Optional[MongoClient],
    config_type: str,
    required: bool = True,
) -> Optional[T]:
    """Load configuration from settings or database.

    Args:
        config: Configuration provided directly in settings
        name: Name of the config to load from database
        mongo_client: Optional MongoDB client for database operations
        config_type: String identifier for error messages ('model' or 'dataset')
        required: Whether the config must be present

    Returns:
        Loaded configuration or None if not required and not found

    Raises:
        AssertionError: If config is required but not found
    """
    if mongo_client is not None and name is not None:
        if config is None:
            config = getattr(mongo_client, f"get_{config_type}_cfg")(name)
            print(f"Loaded {config_type} config from database: {name}")
        else:
            getattr(mongo_client, f"add_{config_type}")(name, config)
            print(f"Added {config_type} config to database: {name}")

    if required:
        assert config is not None, f"{config_type} config not provided and not found in database"
    return config


class GenerateActivationsSettings(BaseSettings):
    """Settings for activation generation."""

    model_config = SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

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

    model_batch_size: Optional[int] = None
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

    @model_validator(mode="after")
    def validate_cfg(self) -> "GenerateActivationsSettings":
        if self.mongo is not None:
            assert self.model is not None, "Database not provided. Must manually provide model config."
            assert self.dataset is not None, "Database not provided. Must manually provide dataset config."
        return self


def generate_activations(settings: GenerateActivationsSettings) -> None:
    """Generate and save model activations from a dataset."""
    # Initialize device mesh
    device_mesh = (
        init_device_mesh(
            device_type="cuda",
            mesh_shape=(int(os.environ.get("WORLD_SIZE", 1)), 1),
            mesh_dim_names=("data", "model"),
        )
        if os.environ.get("WORLD_SIZE") is not None
        else None
    )

    mongo_client = MongoClient(settings.mongo) if settings.mongo is not None else None

    # Load configurations
    model_cfg = load_config(
        config=settings.model, name=settings.model_name, mongo_client=mongo_client, config_type="model"
    )

    dataset_cfg = load_config(
        config=settings.dataset, name=settings.dataset_name, mongo_client=mongo_client, config_type="dataset"
    )

    # Load model and dataset
    model = load_model(model_cfg)
    dataset, metadata = load_dataset(
        dataset_cfg,
        device_mesh=device_mesh,
        n_shards=settings.n_shards,
        start_shard=settings.start_shard,
    )

    # Configure activation generation
    factory_cfg = ActivationFactoryConfig(
        sources=[ActivationFactoryDatasetSource(name=settings.dataset_name)],
        target=settings.target,
        hook_points=settings.hook_points,
        context_size=settings.context_size,
        model_batch_size=settings.model_batch_size,
        batch_size=settings.batch_size,
        buffer_size=settings.buffer_size,
        buffer_shuffle=settings.buffer_shuffle,
    )

    # Configure activation writer
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

    # Generate and write activations
    activations = factory.process(
        model=model, model_name=settings.model_name, datasets={settings.dataset_name: (dataset, metadata)}
    )
    writer.process(activations, device_mesh=device_mesh, start_shard=settings.start_shard)


class TrainSAESettings(BaseSettings):
    """Settings for training a Sparse Autoencoder (SAE)."""

    sae: BaseSAEConfig
    """Configuration for the SAE model architecture and parameters"""

    sae_name: str
    """Name of the SAE model. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model. Use as identifier for the SAE model in the database."""

    initializer: InitializerConfig
    """Configuration for model initialization"""

    trainer: TrainerConfig
    """Configuration for training process"""

    activation_factory: ActivationFactoryConfig
    """Configuration for generating activations"""

    wandb: Optional[WandbConfig] = None
    """Configuration for Weights & Biases logging"""

    eval: bool = False
    """Whether to run in evaluation mode"""

    data_parallel_size: int = 1
    """Size of data parallel mesh"""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

    mongo: Optional[MongoDBConfig] = None
    """Configuration for MongoDB"""

    model: Optional[LanguageModelConfig] = None
    """Configuration for the language model. Required if using dataset sources."""

    model_name: Optional[str] = None
    """Name of the tokenizer to load. Mixcoder requires a tokenizer to get the modality indices."""

    datasets: Optional[dict[str, Optional[DatasetConfig]]] = None
    """Name to dataset config mapping. Required if using dataset sources."""


def train_sae(settings: TrainSAESettings) -> None:
    """Train a SAE model.

    Args:
        settings: Configuration settings for SAE training
    """
    device_mesh = (
        init_device_mesh(
            device_type="cuda",
            mesh_shape=(settings.data_parallel_size, settings.model_parallel_size),
            mesh_dim_names=("data", "model"),
        )
        if settings.data_parallel_size > 1 or settings.model_parallel_size > 1
        else None
    )

    mongo_client = MongoClient(settings.mongo) if settings.mongo is not None else None

    # Load configurations
    model_cfg = load_config(
        config=settings.model,
        name=settings.model_name,
        mongo_client=mongo_client,
        config_type="model",
        required=False,
    )

    dataset_cfgs = (
        {
            dataset_name: load_config(
                config=dataset_cfg,
                name=dataset_name,
                mongo_client=mongo_client,
                config_type="dataset",
            )
            for dataset_name, dataset_cfg in settings.datasets.items()
        }
        if settings.datasets is not None
        else None
    )

    # Load model and datasets
    model = load_model(model_cfg) if model_cfg is not None else None
    datasets = (
        {
            dataset_name: load_dataset(dataset_cfg, device_mesh=device_mesh)
            for dataset_name, dataset_cfg in dataset_cfgs.items()
        }
        if dataset_cfgs is not None
        else None
    )

    activation_factory = ActivationFactory(settings.activation_factory)
    activations_stream = activation_factory.process(
        model=model,
        model_name=settings.model_name,
        datasets=datasets,
    )
    initializer = Initializer(settings.initializer)

    if settings.sae.sae_type == "mixcoder":
        assert settings.model_name is not None, "Model name is required for mixcoder SAE"
        tokenizer = AutoTokenizer.from_pretrained(settings.model_name, trust_remote_code=True)
        mixcoder_settings = {
            "model_name": settings.model_name,
            "tokenizer": tokenizer,
        }
        sae = initializer.initialize_sae_from_config(
            settings.sae,
            activation_stream=activations_stream,
            device_mesh=device_mesh,
            mixcoder_settings=mixcoder_settings,
        )
    else:
        sae = initializer.initialize_sae_from_config(
            settings.sae, activation_stream=activations_stream, device_mesh=device_mesh
        )

    wandb_logger = (
        wandb.init(
            project=settings.wandb.wandb_project,
            config=settings.model_dump(),
            name=settings.wandb.exp_name,
            entity=settings.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),
        )
        if settings.wandb is not None and (device_mesh is None or device_mesh.get_rank() == 0)
        else None
    )
    if wandb_logger is not None:
        wandb_logger.watch(sae, log="all")

    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    trainer = Trainer(settings.trainer)
    trainer.fit(sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger)
    sae.save_pretrained(
        save_path=settings.trainer.exp_result_path,
        sae_name=settings.sae_name,
        sae_series=settings.sae_series,
        mongo_client=mongo_client,
    )

    if wandb_logger is not None:
        wandb_logger.finish()


class AnalyzeSAESettings(BaseSettings):
    sae: BaseSAEConfig
    """Configuration for the SAE model architecture and parameters"""

    sae_name: str
    """Name of the SAE model. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model. Use as identifier for the SAE model in the database."""

    activation_factory: ActivationFactoryConfig
    """Configuration for generating activations"""

    analyzer: FeatureAnalyzerConfig
    """Configuration for feature analysis"""

    mongo: MongoDBConfig
    """Configuration for the MongoDB database."""


def analyze_sae(settings: AnalyzeSAESettings) -> None:
    """Analyze a SAE model."""
    mongo_client = MongoClient(settings.mongo)
    activation_factory = ActivationFactory(settings.activation_factory)

    if settings.sae.sae_type == "sae":
        sae = SparseAutoEncoder.from_config(settings.sae)
    elif settings.sae.sae_type == "crosscoder":
        sae = CrossCoder.from_config(settings.sae)
    elif settings.sae.sae_type == "mixcoder":
        sae = MixCoder.from_config(settings.sae)
    else:
        # TODO: add support for different SAE config types, e.g. MixCoderConfig, CrossCoderConfig, etc.
        raise ValueError(f"SAE type {settings.sae.sae_type} not supported.")

    analyzer = FeatureAnalyzer(settings.analyzer)

    activations = activation_factory.process()
    result = analyzer.analyze_chunk(activations, sae=sae)

    mongo_client.add_feature_analysis(
        name="default", sae_name=settings.sae_name, sae_series=settings.sae_series, analysis=result
    )
