import os
from pathlib import Path
from typing import Literal, Optional

import wandb
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.distributed.device_mesh import init_device_mesh

from lm_saes.activation.factory import ActivationFactory
from lm_saes.activation.writer import ActivationWriter
from lm_saes.config import (
    ActivationFactoryConfig,
    ActivationFactoryDatasetSource,
    ActivationFactoryTarget,
    ActivationWriterConfig,
    DatasetConfig,
    InitializerConfig,
    LanguageModelConfig,
    SAEConfig,
    TrainerConfig,
    WandbConfig,
)
from lm_saes.initializer import Initializer
from lm_saes.resource_loaders import load_dataset, load_model
from lm_saes.trainer import Trainer
from lm_saes.utils.misc import is_master


class GenerateActivationsSettings(BaseSettings):
    """Settings for activation generation."""

    model_config = SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    model: LanguageModelConfig
    """Configuration for loading the language model"""

    model_name: str
    """Name of the model to load"""

    dataset: DatasetConfig
    """Configuration for loading the dataset"""

    dataset_name: str
    """Name of the dataset"""

    hook_points: list[str]
    """List of model hook points to capture activations from"""

    output_dir: Path
    """Directory to save activation files"""

    total_tokens: Optional[int] = None
    """Optional total number of tokens to generate"""

    context_size: int = 128
    """Context window size for tokenization"""

    n_samples_per_chunk: int = 16
    """Number of samples per saved chunk"""

    format: Literal["pt", "safetensors"] = "safetensors"
    """Format to save activations in ('pt' or 'safetensors')"""

    n_shards: Optional[int] = None
    """Number of shards to split the dataset into. If None, the dataset is split to the world size. Must be larger than the world size."""

    start_shard: int = 0
    """The shard to start writing from"""


def generate_activations(settings: GenerateActivationsSettings) -> None:
    """Generate and save model activations from a dataset.

    Args:
        settings: Configuration settings for activation generation
    """
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

    # Load model and dataset
    model = load_model(settings.model)
    dataset, metadata = load_dataset(
        settings.dataset,
        device_mesh=device_mesh,
        n_shards=settings.n_shards,
        start_shard=settings.start_shard,
    )

    # Configure activation generation
    factory_cfg = ActivationFactoryConfig(
        sources=[ActivationFactoryDatasetSource(name=settings.dataset_name)],
        target=ActivationFactoryTarget.ACTIVATIONS_2D,
        hook_points=settings.hook_points,
        context_size=settings.context_size,
        batch_size=None,
        buffer_size=None,
    )

    # Configure activation writer
    writer_cfg = ActivationWriterConfig(
        hook_points=settings.hook_points,
        total_generating_tokens=settings.total_tokens,
        n_samples_per_chunk=settings.n_samples_per_chunk,
        cache_dir=settings.output_dir,
        format=settings.format,
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

    sae: SAEConfig
    """Configuration for the SAE model architecture and parameters"""

    initializer: InitializerConfig
    """Configuration for model initialization"""

    trainer: TrainerConfig
    """Configuration for training process"""

    activation_factory: ActivationFactoryConfig
    """Configuration for generating activations"""

    wandb: WandbConfig
    """Configuration for Weights & Biases logging"""

    eval: bool = False
    """Whether to run in evaluation mode"""

    data_parallel_size: int = 1
    """Size of data parallel mesh"""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""


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
    activation_factory = ActivationFactory(settings.activation_factory)
    activations_stream = activation_factory.process()
    # TODO: get activation norm from activation_factory
    # activation_norm = activation_factory.get_activation_norm()
    activation_norm = {settings.sae.hook_point_in: 1.0, settings.sae.hook_point_out: 1.0}
    initializer = Initializer(settings.initializer)
    sae = initializer.initialize_sae_from_config(
        settings.sae, activation_stream=activations_stream, activation_norm=activation_norm, device_mesh=device_mesh
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
        if settings.wandb.log_to_wandb and is_master()
        else None
    )
    if wandb_logger is not None:
        wandb_logger.watch(sae, log="all")

    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    trainer = Trainer(settings.trainer)
    trainer.fit(sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger)
