import os
import wandb
from pathlib import Path
from typing import Literal, Optional

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
    """Settings for activation generation.

    Attributes:
        model_cfg: Configuration for loading the language model
        dataset_cfg: Configuration for loading the dataset
        dataset_name: Name of the dataset
        hook_points: List of model hook points to capture activations from
        output_dir: Directory to save activation files
        total_tokens: Optional total number of tokens to generate
        context_size: Context window size for tokenization
        n_samples_per_chunk: Number of samples per saved chunk
        n_shards: Number of shards to split the dataset into. If None, the dataset is split to the world size.
        start_shard: The shard to start writing from
        format: Format to save activations in ('pt' or 'safetensors')
    """

    model_config = SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    model: LanguageModelConfig
    dataset: DatasetConfig
    dataset_name: str
    hook_points: list[str]
    output_dir: Path
    total_tokens: Optional[int] = None
    context_size: int = 128
    n_samples_per_chunk: int = 16
    format: Literal["pt", "safetensors"] = "safetensors"
    n_shards: Optional[int] = None
    start_shard: int = 0


class TrainSAESettings(BaseSettings):
    sae: SAEConfig
    initializer: InitializerConfig
    trainer: TrainerConfig
    activation_factory: ActivationFactoryConfig
    wandb: WandbConfig
    eval: bool = False
    data_parallel_size: int = 1
    model_parallel_size: int = 1


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
    activations = factory.process(model=model, datasets={settings.dataset_name: (dataset, metadata)})
    writer.process(activations, device_mesh=device_mesh, start_shard=settings.start_shard)


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
    sae = initializer.initialize_sae_from_config(settings.sae, activation_stream=activations_stream, activation_norm=activation_norm, device_mesh=device_mesh)

    wandb_logger = wandb.init(
            project=settings.wandb.wandb_project,
            config=settings.model_dump(),
            name=settings.wandb.exp_name,
            entity=settings.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),
        ) if settings.wandb.log_to_wandb and is_master() else None
    if wandb_logger is not None:
        wandb_logger.watch(sae, log="all")
    
    # TODO: implement eval_fn
    eval_fn = lambda x: None if settings.eval else None

    trainer = Trainer(settings.trainer)
    trainer.fit(sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger)