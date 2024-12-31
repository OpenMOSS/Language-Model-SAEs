import os
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
    LanguageModelConfig,
)
from lm_saes.resource_loaders import load_dataset, load_model


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

    model_config = SettingsConfigDict(cli_parse_args=True)

    model_cfg: LanguageModelConfig
    dataset_cfg: DatasetConfig
    dataset_name: str
    hook_points: list[str]
    output_dir: Path
    total_tokens: Optional[int] = None
    context_size: int = 128
    n_samples_per_chunk: int = 16
    format: Literal["pt", "safetensors"] = "safetensors"
    n_shards: Optional[int] = None
    start_shard: int = 0


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
    model = load_model(settings.model_cfg)
    dataset = load_dataset(
        settings.dataset_cfg,
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
    activations = factory.process(model=model, datasets={settings.dataset_name: dataset})
    writer.process(activations, device_mesh=device_mesh, start_shard=settings.start_shard)
