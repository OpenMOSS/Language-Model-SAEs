import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, TypeVar, cast, overload

import torch
import torch.distributed as dist
import wandb
from datasets import Dataset
from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings
from torch.distributed.device_mesh import init_device_mesh

from lm_saes.activation.factory import ActivationFactory
from lm_saes.activation.writer import ActivationWriter
from lm_saes.analysis.auto_interp import (
    AutoInterpConfig,
    FeatureInterpreter,
)
from lm_saes.analysis.feature_analyzer import FeatureAnalyzer
from lm_saes.config import (
    ActivationFactoryConfig,
    ActivationFactoryDatasetSource,
    ActivationFactoryTarget,
    ActivationWriterConfig,
    BaseSAEConfig,
    BufferShuffleConfig,
    CrossCoderConfig,
    DatasetConfig,
    FeatureAnalyzerConfig,
    InitializerConfig,
    LanguageModelConfig,
    MixCoderConfig,
    MongoDBConfig,
    TrainerConfig,
    WandbConfig,
)
from lm_saes.crosscoder import CrossCoder
from lm_saes.database import MongoClient
from lm_saes.initializer import Initializer
from lm_saes.mixcoder import MixCoder
from lm_saes.resource_loaders import load_dataset, load_dataset_shard, load_model
from lm_saes.sae import SparseAutoEncoder
from lm_saes.trainer import Trainer
from lm_saes.utils.misc import get_modality_tokens, is_primary_rank

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
        ignore_token_ids=settings.ignore_token_ids,
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

    if isinstance(settings.sae, MixCoderConfig):
        modality_names = settings.sae.modality_names
        if "text" in modality_names:  # Multimodal mixcoder SAE
            from transformers.models.auto.tokenization_auto import AutoTokenizer

            assert (
                model_cfg is not None
            ), "Model cfg is required for multimodal mixcoder SAE for inferring text/image tokens"
            tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name, trust_remote_code=True)
            modality_tokens = get_modality_tokens(tokenizer, model_cfg.model_name)
            for modality in modality_tokens.keys():
                modality_tokens[modality] = modality_tokens[modality].to(settings.sae.device)
            assert list(sorted(modality_tokens.keys())) == list(
                sorted(modality_names)
            ), "Modality names must match the keys of modality_tokens"

            def activation_interceptor(
                activations: dict[str, torch.Tensor], source_idx: int
            ) -> dict[str, torch.Tensor]:
                assert (
                    "tokens" in activations
                ), "Tokens are required for multimodal mixcoder SAE for inferring text/image tokens"
                modalities = torch.zeros_like(activations["tokens"], dtype=torch.int)
                for i, modality in enumerate(modality_names):
                    mask = torch.isin(activations["tokens"], modality_tokens[modality])
                    modalities[mask] = i
                activations = activations | {"modalities": modalities}
                return activations
        else:  # Multi-lingual mixcoder SAE
            assert [
                source.name for source in settings.activation_factory.sources
            ] == modality_names, "Modality names must match the names of the activation sources"

            def activation_interceptor(
                activations: dict[str, torch.Tensor], source_idx: int
            ) -> dict[str, torch.Tensor]:
                assert "tokens" in activations, "Tokens are required for inferring shape of activations"
                modalities = torch.ones_like(activations["tokens"], dtype=torch.int) * source_idx
                activations = activations | {"modalities": modalities}
                return activations

        activation_factory = ActivationFactory(
            settings.activation_factory, before_aggregation_interceptor=activation_interceptor
        )
    else:
        activation_factory = ActivationFactory(settings.activation_factory)

    activations_stream = activation_factory.process(
        model=model,
        model_name=settings.model_name,
        datasets=datasets,
    )
    initializer = Initializer(settings.initializer)

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
            mode=os.getenv("WANDB_MODE", "online"),  # type: ignore
        )
        if settings.wandb is not None and (device_mesh is None or device_mesh.get_rank() == 0)
        else None
    )
    if wandb_logger is not None:
        wandb_logger.watch(sae, log="all")

    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    trainer = Trainer(settings.trainer)
    sae.cfg.save_hyperparameters(settings.trainer.exp_result_path)
    trainer.fit(sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger)
    sae.save_pretrained(
        save_path=settings.trainer.exp_result_path,
        sae_name=settings.sae_name,
        sae_series=settings.sae_series,
        mongo_client=mongo_client,
    )

    if wandb_logger is not None:
        wandb_logger.finish()


class TrainCrossCoderSettings(BaseSettings):
    """Settings for training a CrossCoder. The main difference to TrainSAESettings is that the activation factory is a list of ActivationFactoryConfig, one for each head."""

    sae: CrossCoderConfig
    """Configuration for the CrossCoder model architecture and parameters"""

    sae_name: str
    """Name of the SAE model. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model. Use as identifier for the SAE model in the database."""

    initializer: InitializerConfig
    """Configuration for model initialization"""

    trainer: TrainerConfig
    """Configuration for training process"""

    activation_factories: list[ActivationFactoryConfig]
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


def train_crosscoder(settings: TrainCrossCoderSettings) -> None:
    """Train a CrossCoder.

    Args:
        settings: Configuration settings for SAE training
    """
    assert isinstance(settings.sae, CrossCoderConfig), "CrossCoderConfig is required for training a CrossCoder"
    assert (
        len(settings.activation_factories) == settings.sae.n_heads
    ), "Number of activation factories must match the number of heads in the CrossCoder"

    device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(settings.sae.n_heads, settings.data_parallel_size, settings.model_parallel_size),
        mesh_dim_names=("head", "data", "model"),
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

    activation_factory = ActivationFactory(settings.activation_factories[device_mesh.get_local_rank("head")])

    activations_stream = activation_factory.process(
        model=model,
        model_name=settings.model_name,
        datasets=datasets,
    )
    initializer = Initializer(settings.initializer)

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
            mode=os.getenv("WANDB_MODE", "online"),  # type: ignore
        )
        if settings.wandb is not None and (device_mesh is None or device_mesh.get_rank() == 0)
        else None
    )

    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    trainer = Trainer(settings.trainer)
    sae.cfg.save_hyperparameters(settings.trainer.exp_result_path)
    trainer.fit(sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger)
    sae.save_pretrained(
        save_path=settings.trainer.exp_result_path,
        sae_name=settings.sae_name,
        sae_series=settings.sae_series,
        mongo_client=mongo_client,
    )

    if wandb_logger is not None:
        wandb_logger.finish()


class SweepingItem(BaseModel):
    """A single item in a sweeping configuration."""

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

    wandb: Optional[WandbConfig] = None
    """Configuration for Weights & Biases logging"""


class SweepSAESettings(BaseSettings):
    """Settings for sweeping a Sparse Autoencoder (SAE)."""

    items: list[SweepingItem]
    """List of sweeping items"""

    activation_factory: ActivationFactoryConfig
    """Configuration for generating activations"""

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


def sweep_sae(settings: SweepSAESettings) -> None:
    """Sweep experiments for training SAE models.

    Args:
        settings: Configuration settings for SAE sweeping
    """

    n_sweeps = len(settings.items)

    device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(n_sweeps, settings.data_parallel_size, settings.model_parallel_size),
        mesh_dim_names=("sweep", "data", "model"),
    )

    mongo_client = MongoClient(settings.mongo) if settings.mongo is not None else None

    if device_mesh.get_local_rank("sweep") == 0:
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

        if isinstance(settings.items[0].sae, MixCoderConfig):
            modality_names = settings.items[0].sae.modality_names
            assert all(
                isinstance(item.sae, MixCoderConfig) and item.sae.modality_names == modality_names
                for item in settings.items
            ), "All items must have the same modality names"
            if "text" in modality_names:  # Multimodal mixcoder SAE
                from transformers.models.auto.tokenization_auto import AutoTokenizer

                assert (
                    model_cfg is not None
                ), "Model cfg is required for multimodal mixcoder SAE for inferring text/image tokens"
                tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name, trust_remote_code=True)
                modality_tokens = get_modality_tokens(tokenizer, model_cfg.model_name)
                for modality in modality_tokens.keys():
                    modality_tokens[modality] = modality_tokens[modality].to(settings.items[0].sae.device)
                assert list(sorted(modality_tokens.keys())) == list(
                    sorted(modality_names)
                ), "Modality names must match the keys of modality_tokens"

                def activation_interceptor(
                    activations: dict[str, torch.Tensor], source_idx: int
                ) -> dict[str, torch.Tensor]:
                    assert (
                        "tokens" in activations
                    ), "Tokens are required for multimodal mixcoder SAE for inferring text/image tokens"
                    modalities = torch.zeros_like(activations["tokens"], dtype=torch.int)
                    for i, modality in enumerate(modality_names):
                        mask = torch.isin(activations["tokens"], modality_tokens[modality])
                        modalities[mask] = i
                    activations = activations | {"modalities": modalities}
                    return activations
            else:  # Multi-lingual mixcoder SAE
                assert [
                    source.name for source in settings.activation_factory.sources
                ] == modality_names, "Modality names must match the names of the activation sources"

                def activation_interceptor(
                    activations: dict[str, torch.Tensor], source_idx: int
                ) -> dict[str, torch.Tensor]:
                    assert "tokens" in activations, "Tokens are required for inferring shape of activations"
                    modalities = torch.ones_like(activations["tokens"], dtype=torch.int) * source_idx
                    activations = activations | {"modalities": modalities}
                    return activations

            activation_factory = ActivationFactory(
                settings.activation_factory, before_aggregation_interceptor=activation_interceptor
            )
        else:
            activation_factory = ActivationFactory(settings.activation_factory)

        activations_stream = activation_factory.process(
            model=model,
            model_name=settings.model_name,
            datasets=datasets,
        )
    else:
        activations_stream = None

    def broadcast_activations_stream(activations_stream: Iterable[dict[str, torch.Tensor]] | None):
        if device_mesh.get_local_rank("sweep") == 0:
            assert activations_stream is not None, "Activations stream must be provided on rank 0 of sweep dimension"
            for activations in activations_stream:
                dist.broadcast_object_list([activations], group=device_mesh.get_group("sweep"), group_src=0)
                yield activations
            dist.broadcast_object_list([None], group=device_mesh.get_group("sweep"), group_src=0)
        else:
            while True:
                objs = [None]
                dist.broadcast_object_list(objs, group=device_mesh.get_group("sweep"), group_src=0)
                if objs[0] is None:
                    break
                activations = {
                    k: v.to(torch.device("cuda", int(os.environ["LOCAL_RANK"]))) if isinstance(v, torch.Tensor) else v
                    for k, v in cast(dict[str, Any], objs[0]).items()  # noqa: F821
                }
                yield activations

    activations_stream = broadcast_activations_stream(activations_stream)

    item = settings.items[device_mesh.get_local_rank("sweep")]
    initializer = Initializer(item.initializer)

    sae = initializer.initialize_sae_from_config(
        item.sae, activation_stream=activations_stream, device_mesh=device_mesh
    )

    wandb_logger = (
        wandb.init(
            project=item.wandb.wandb_project,
            config=item.model_dump(),
            name=item.wandb.exp_name,
            entity=item.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),  # type: ignore
        )
        if item.wandb is not None and is_primary_rank(device_mesh)
        else None
    )
    if wandb_logger is not None:
        wandb_logger.watch(sae, log="all")

    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    trainer = Trainer(item.trainer)
    sae.cfg.save_hyperparameters(item.trainer.exp_result_path)
    trainer.fit(sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger)
    sae.save_pretrained(
        save_path=item.trainer.exp_result_path,
        sae_name=item.sae_name,
        sae_series=item.sae_series,
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

    model: Optional[LanguageModelConfig] = None
    """Configuration for the language model. Required if using dataset sources."""

    model_name: Optional[str] = None
    """Name of the tokenizer to load. Mixcoder requires a tokenizer to get the modality indices."""

    analyzer: FeatureAnalyzerConfig
    """Configuration for feature analysis"""

    mongo: MongoDBConfig
    """Configuration for the MongoDB database."""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""


def analyze_sae(settings: AnalyzeSAESettings) -> None:
    """Analyze a SAE model."""

    device_mesh = (
        init_device_mesh(
            device_type="cuda",
            mesh_shape=(settings.model_parallel_size,),
            mesh_dim_names=("model",),
        )
        if settings.model_parallel_size > 1
        else None
    )

    mongo_client = MongoClient(settings.mongo)

    model_cfg = load_config(
        config=settings.model,
        name=settings.model_name,
        mongo_client=mongo_client,
        config_type="model",
        required=False,
    )

    if isinstance(settings.sae, MixCoderConfig):
        modality_names = settings.sae.modality_names
        if "text" in modality_names:  # Multimodal mixcoder SAE
            from transformers.models.auto.tokenization_auto import AutoTokenizer

            assert (
                model_cfg is not None
            ), "Model cfg is required for multimodal mixcoder SAE for inferring text/image tokens"
            tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name, trust_remote_code=True)
            modality_tokens = get_modality_tokens(tokenizer, model_cfg.model_name)
            for modality in modality_tokens.keys():
                modality_tokens[modality] = modality_tokens[modality].to(settings.sae.device)

            assert list(sorted(modality_tokens.keys())) == list(
                sorted(modality_names)
            ), "Modality names must match the keys of modality_tokens"

            def activation_interceptor(
                activations: dict[str, torch.Tensor], source_idx: int
            ) -> dict[str, torch.Tensor]:
                assert (
                    "tokens" in activations
                ), "Tokens are required for multimodal mixcoder SAE for inferring text/image tokens"
                modalities = torch.zeros_like(activations["tokens"], dtype=torch.int)
                for i, modality in enumerate(modality_names):
                    mask = torch.isin(activations["tokens"], modality_tokens[modality])
                    modalities[mask] = i
                activations = activations | {"modalities": modalities}
                return activations
        else:  # Multi-lingual mixcoder SAE
            assert [
                source.name for source in settings.activation_factory.sources
            ] == modality_names, "Modality names must match the names of the activation sources"

            def activation_interceptor(
                activations: dict[str, torch.Tensor], source_idx: int
            ) -> dict[str, torch.Tensor]:
                assert "tokens" in activations, "Tokens are required for inferring shape of activations"
                modalities = torch.ones_like(activations["tokens"], dtype=torch.int) * source_idx
                activations = activations | {"modalities": modalities}
                return activations

        activation_factory = ActivationFactory(
            settings.activation_factory, before_aggregation_interceptor=activation_interceptor
        )
    else:
        activation_factory = ActivationFactory(settings.activation_factory)

    if isinstance(settings.sae, MixCoderConfig):
        sae = MixCoder.from_config(settings.sae)
    elif isinstance(settings.sae, CrossCoderConfig):
        sae = CrossCoder.from_config(settings.sae)
    else:
        sae = SparseAutoEncoder.from_config(settings.sae)

    if device_mesh is not None:
        sae.tensor_parallel(device_mesh)

    analyzer = FeatureAnalyzer(settings.analyzer)

    def rebatch_activations(activations: Iterable[dict[str, torch.Tensor]]):
        n_ctx = None
        for activation in activations:
            batch_size = activation["tokens"].size(0)
            if n_ctx is None:
                n_ctx = activation["tokens"].size(1)
            if n_ctx is not None and activation["tokens"].size(1) != n_ctx:
                warnings.warn(f"Context size mismatch: {n_ctx} != {activation['tokens'].size(1)}. Skipping batch.")
                continue
            for i in range(4):
                start = i * batch_size // 4
                end = (i + 1) * batch_size // 4
                yield {k: v[start:end] for k, v in activation.items()}

    activations = activation_factory.process()
    activations = rebatch_activations(activations)
    result = analyzer.analyze_chunk(activations, sae=sae, device_mesh=device_mesh)

    start_idx = 0 if device_mesh is None else device_mesh.get_local_rank("model") * len(result)
    mongo_client.add_feature_analysis(
        name="default", sae_name=settings.sae_name, sae_series=settings.sae_series, analysis=result, start_idx=start_idx
    )


class AutoInterpSettings(BaseSettings):
    """Settings for automatic interpretation of SAE features."""

    sae: BaseSAEConfig
    """Configuration for the SAE model architecture and parameters"""

    sae_name: str
    """Name of the SAE model to interpret. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model to interpret. Use as identifier for the SAE model in the database."""

    model: LanguageModelConfig
    """Configuration for the language model used to generate activations."""

    model_name: str
    """Name of the model to load."""

    dataset_name: str
    """Name of the dataset to use for non-activating examples."""

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
    if isinstance(settings.sae, MixCoderConfig):
        sae = MixCoder.from_config(settings.sae)
    elif isinstance(settings.sae, CrossCoderConfig):
        sae = CrossCoder.from_config(settings.sae)
    else:
        sae = SparseAutoEncoder.from_config(settings.sae)

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
    results = interpreter.interpret_feature(
        sae_name=settings.sae_name,
        sae_series=settings.sae_series,
        feature_indices=feature_indices,
        model=model,
        sae=sae,
        datasets=get_dataset,
        dataset_name=settings.dataset_name,
        analysis_name=settings.analysis_name,
    )

    # Save results to database
    print("Saving results to database...")
    for feature_idx, result in results.items():
        interpretation = {
            "text": result["explanation"],
            "validation": [
                {"method": eval_result["method"], "passed": eval_result["passed"], "detail": eval_result}
                for eval_result in result["evaluations"]
            ],
            "detail": result["explanation_details"],
        }

        mongo_client.update_feature(
            settings.sae_name, feature_idx, {"interpretation": interpretation}, settings.sae_series
        )
    print(f"Completed interpreting {len(results)} features")
