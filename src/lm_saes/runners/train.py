"""Module for sweeping SAE experiments."""

import os
from typing import Any, Iterable, Optional, cast

import torch
import torch.distributed as dist
import wandb
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from torch.distributed.device_mesh import init_device_mesh

from lm_saes.activation.factory import ActivationFactory
from lm_saes.config import (
    ActivationFactoryConfig,
    BaseSAEConfig,
    CLTConfig,
    CrossCoderConfig,
    DatasetConfig,
    InitializerConfig,
    LanguageModelConfig,
    MongoDBConfig,
    TrainerConfig,
    WandbConfig,
    LorsaConfig,
)
from lm_saes.database import MongoClient
from lm_saes.initializer import Initializer
from lm_saes.resource_loaders import load_dataset, load_model
from lm_saes.runners.utils import load_config
from lm_saes.trainer import Trainer
from lm_saes.utils.logging import get_distributed_logger, setup_logging
from lm_saes.utils.misc import is_primary_rank

logger = get_distributed_logger("runners.train")


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

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


def train_sae(settings: TrainSAESettings) -> None:
    """Train a SAE model.

    Args:
        settings: Configuration settings for SAE training
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

    mongo_client = MongoClient(settings.mongo) if settings.mongo is not None else None
    if mongo_client:
        logger.info("MongoDB client initialized")

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
    logger.info("Loading model and datasets")
    model = load_model(model_cfg) if model_cfg is not None else None
    datasets = (
        {
            dataset_name: load_dataset(dataset_cfg, device_mesh=device_mesh)
            for dataset_name, dataset_cfg in dataset_cfgs.items()
        }
        if dataset_cfgs is not None
        else None
    )

    activation_factory = ActivationFactory(settings.activation_factory, device_mesh=device_mesh)

    logger.info("Processing activations stream")
    activations_stream = activation_factory.process(
        model=model,
        model_name=settings.model_name,
        datasets=datasets,
    )

    logger.info("Initializing SAE")
    initializer = Initializer(settings.initializer)

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

    sae = initializer.initialize_sae_from_config(
        settings.sae, activation_stream=activations_stream, device_mesh=device_mesh, wandb_logger=wandb_logger
    )

    logger.info(f"SAE initialized: {type(sae).__name__}")

    if wandb_logger is not None:
        logger.info("WandB logger initialized")

    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    logger.info("Starting training")
    trainer = Trainer(settings.trainer)
    sae.cfg.save_hyperparameters(settings.trainer.exp_result_path)
    trainer.fit(sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger)

    logger.info("Training completed, saving model")
    sae.save_pretrained(
        save_path=settings.trainer.exp_result_path,
        sae_name=settings.sae_name,
        sae_series=settings.sae_series,
        mongo_client=mongo_client,
    )

    if wandb_logger is not None:
        wandb_logger.finish()
        logger.info("WandB session closed")

    logger.info("SAE training completed successfully")


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

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


def train_crosscoder(settings: TrainCrossCoderSettings) -> None:
    """Train a CrossCoder.

    Args:
        settings: Configuration settings for SAE training
    """
    # Set up logging
    setup_logging(level="INFO")

    assert isinstance(settings.sae, CrossCoderConfig), "CrossCoderConfig is required for training a CrossCoder"
    assert all(
        len(activation_factory.hook_points) == len(settings.activation_factories[0].hook_points)
        for activation_factory in settings.activation_factories
    ), "Number of hook points of activation factories must be the same"
    assert (
        len(settings.activation_factories) * len(settings.activation_factories[0].hook_points) == settings.sae.n_heads
    ), "Total number of hook points must match the number of heads in the CrossCoder"
    head_parallel_size = len(settings.activation_factories)

    device_mesh = init_device_mesh(
        device_type=settings.device_type,
        mesh_shape=(head_parallel_size, settings.data_parallel_size, settings.model_parallel_size),
        mesh_dim_names=("head", "data", "model"),
    )

    logger.info(
        f"Device mesh initialized with {settings.sae.n_heads} heads, {head_parallel_size} head parallel size, {settings.data_parallel_size} data parallel size, {settings.model_parallel_size} model parallel size"
    )

    mongo_client = MongoClient(settings.mongo) if settings.mongo is not None else None
    if mongo_client:
        logger.info("MongoDB client initialized")

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
    logger.info("Loading model and datasets")
    model = load_model(model_cfg) if model_cfg is not None else None
    datasets = (
        {
            dataset_name: load_dataset(dataset_cfg, device_mesh=device_mesh)
            for dataset_name, dataset_cfg in dataset_cfgs.items()
        }
        if dataset_cfgs is not None
        else None
    )

    logger.info("Setting up activation factory for CrossCoder")
    activation_factory = ActivationFactory(settings.activation_factories[device_mesh.get_local_rank("head")])

    logger.info("Processing activations stream")
    activations_stream = activation_factory.process(
        model=model,
        model_name=settings.model_name,
        datasets=datasets,
    )

    logger.info("Initializing CrossCoder")
    initializer = Initializer(settings.initializer)

    sae = initializer.initialize_sae_from_config(
        settings.sae, activation_stream=activations_stream, device_mesh=device_mesh
    )

    logger.info("CrossCoder initialized")

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
        logger.info("WandB logger initialized")

    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    logger.info("Starting CrossCoder training")
    trainer = Trainer(settings.trainer)
    sae.cfg.save_hyperparameters(settings.trainer.exp_result_path)
    trainer.fit(sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger)

    logger.info("Training completed, saving CrossCoder")
    sae.save_pretrained(
        save_path=settings.trainer.exp_result_path,
        sae_name=settings.sae_name,
        sae_series=settings.sae_series,
        mongo_client=mongo_client,
    )

    if wandb_logger is not None:
        wandb_logger.finish()
        logger.info("WandB session closed")

    logger.info("CrossCoder training completed successfully")


class TrainCLTSettings(BaseSettings):
    """Settings for training a Cross Layer Transcoder (CLT). CLT works with multiple layers and their corresponding hook points."""

    sae: CLTConfig
    """Configuration for the CLT model architecture and parameters"""

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

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

    mongo: Optional[MongoDBConfig] = None
    """Configuration for MongoDB"""

    model: Optional[LanguageModelConfig] = None
    """Configuration for the language model. Required if using dataset sources."""

    model_name: Optional[str] = None
    """Name of the tokenizer to load. CLT requires a tokenizer to get the modality indices."""

    datasets: Optional[dict[str, Optional[DatasetConfig]]] = None
    """Name to dataset config mapping. Required if using dataset sources."""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


def train_clt(settings: TrainCLTSettings) -> None:
    """Train a Cross Layer Transcoder (CLT) model.

    Args:
        settings: Configuration settings for CLT training
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

    mongo_client = MongoClient(settings.mongo) if settings.mongo is not None else None
    if mongo_client:
        logger.info("MongoDB client initialized")

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
    logger.info("Loading model and datasets")
    model = load_model(model_cfg) if model_cfg is not None else None
    datasets = (
        {
            dataset_name: load_dataset(dataset_cfg, device_mesh=device_mesh)
            for dataset_name, dataset_cfg in dataset_cfgs.items()
        }
        if dataset_cfgs is not None
        else None
    )

    activation_factory = ActivationFactory(settings.activation_factory, device_mesh=device_mesh)

    logger.info("Processing activations stream")
    activations_stream = activation_factory.process(
        model=model,
        model_name=settings.model_name,
        datasets=datasets,
    )

    logger.info("Initializing CLT")
    initializer = Initializer(settings.initializer)

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

    sae = initializer.initialize_sae_from_config(
        settings.sae, activation_stream=activations_stream, device_mesh=device_mesh, wandb_logger=wandb_logger
    )

    logger.info(f"CLT initialized: {type(sae).__name__}")

    if wandb_logger is not None:
        logger.info("WandB logger initialized")

    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    logger.info("Starting CLT training")
    trainer = Trainer(settings.trainer)
    sae.cfg.save_hyperparameters(settings.trainer.exp_result_path)
    trainer.fit(sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger)

    logger.info("Training completed, saving CLT model")
    sae.save_pretrained(
        save_path=settings.trainer.exp_result_path,
        sae_name=settings.sae_name,
        sae_series=settings.sae_series,
        mongo_client=mongo_client,
    )

    if wandb_logger is not None:
        wandb_logger.finish()
        logger.info("WandB session closed")

    logger.info("CLT training completed successfully")


class TrainLorsaSettings(BaseSettings):
    """Settings for training a LORSA (Low-Rank Sparse Autoencoder) model."""

    sae: LorsaConfig
    """Configuration for the LORSA model architecture and parameters"""

    sae_name: str
    """Name of the LORSA model. Use as identifier for the LORSA model in the database."""

    sae_series: str
    """Series of the LORSA model. Use as identifier for the LORSA model in the database."""

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

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

    mongo: Optional[MongoDBConfig] = None
    """Configuration for MongoDB"""

    model: Optional[LanguageModelConfig] = None
    """Configuration for the language model. Required if using dataset sources."""

    model_name: Optional[str] = None
    """Name of the tokenizer to load. LORSA may require a tokenizer to get the modality indices."""

    datasets: Optional[dict[str, Optional[DatasetConfig]]] = None
    """Name to dataset config mapping. Required if using dataset sources."""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


def train_lorsa(settings: TrainLorsaSettings) -> None:
    """Train a LORSA (Low-Rank Sparse Autoencoder) model.

    Args:
        settings: Configuration settings for LORSA training
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

    mongo_client = MongoClient(settings.mongo) if settings.mongo is not None else None
    if mongo_client:
        logger.info("MongoDB client initialized")

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
    logger.info("Loading model and datasets")
    model = load_model(model_cfg) if model_cfg is not None else None
    datasets = (
        {
            dataset_name: load_dataset(dataset_cfg, device_mesh=device_mesh)
            for dataset_name, dataset_cfg in dataset_cfgs.items()
        }
        if dataset_cfgs is not None
        else None
    )

    activation_factory = ActivationFactory(settings.activation_factory, device_mesh=device_mesh)

    logger.info("Processing activations stream")
    activations_stream = activation_factory.process(
        model=model,
        model_name=settings.model_name,
        datasets=datasets,
    )

    logger.info("Initializing lorsa")
    initializer = Initializer(settings.initializer)

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

    sae = initializer.initialize_sae_from_config(
        settings.sae, activation_stream=activations_stream, device_mesh=device_mesh, wandb_logger=wandb_logger
    )

    logger.info(f"lorsa initialized: {type(sae).__name__}")

    if wandb_logger is not None:
        logger.info("WandB logger initialized")

    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    logger.info("Starting LORSA training")
    trainer = Trainer(settings.trainer)
    sae.cfg.save_hyperparameters(settings.trainer.exp_result_path)
    trainer.fit(sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger)

    logger.info("Training completed, saving LORSA model")
    sae.save_pretrained(
        save_path=settings.trainer.exp_result_path,
        sae_name=settings.sae_name,
        sae_series=settings.sae_series,
        mongo_client=mongo_client,
    )

    if wandb_logger is not None:
        wandb_logger.finish()
        logger.info("WandB session closed")

    logger.info("LORSA training completed successfully")


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

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


def sweep_sae(settings: SweepSAESettings) -> None:
    """Sweep experiments for training SAE models.

    Args:
        settings: Configuration settings for SAE sweeping
    """
    # Set up logging
    setup_logging(level="INFO")

    n_sweeps = len(settings.items)

    device_mesh = init_device_mesh(
        device_type=settings.device_type,
        mesh_shape=(n_sweeps, settings.data_parallel_size, settings.model_parallel_size),
        mesh_dim_names=("sweep", "data", "model"),
    )

    logger.info(f"Device mesh initialized for sweep with {n_sweeps} configurations")

    mongo_client = MongoClient(settings.mongo) if settings.mongo is not None else None

    if device_mesh.get_local_rank("sweep") == 0:
        logger.info("Loading configurations on rank 0")
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
        logger.info("Loading model and datasets on rank 0")
        model = load_model(model_cfg) if model_cfg is not None else None
        datasets = (
            {
                dataset_name: load_dataset(dataset_cfg, device_mesh=device_mesh)
                for dataset_name, dataset_cfg in dataset_cfgs.items()
            }
            if dataset_cfgs is not None
            else None
        )

        activation_factory = ActivationFactory(settings.activation_factory, device_mesh=device_mesh)

        logger.info("Processing activations stream on rank 0")
        activations_stream = activation_factory.process(
            model=model,
            model_name=settings.model_name,
            datasets=datasets,
        )
    else:
        activations_stream = None

    def broadcast_activations_stream(activations_stream: Optional[Iterable[dict[str, torch.Tensor]]]):
        if device_mesh.get_local_rank("sweep") == 0:
            assert activations_stream is not None, "Activations stream must be provided on rank 0 of sweep dimension"
            for activations in activations_stream:
                dist.broadcast_object_list([activations], group=device_mesh.get_group("sweep"), src=0)
                yield activations
            dist.broadcast_object_list([None], group=device_mesh.get_group("sweep"), src=0)
        else:
            while True:
                objs = [None]
                dist.broadcast_object_list(objs, group=device_mesh.get_group("sweep"), src=0)
                if objs[0] is None:
                    break
                activations = {
                    k: v.to(torch.device("cuda", int(os.environ["LOCAL_RANK"]))) if isinstance(v, torch.Tensor) else v
                    for k, v in cast(dict[str, Any], objs[0]).items()  # noqa: F821
                }
                yield activations

    activations_stream = broadcast_activations_stream(activations_stream)

    item = settings.items[device_mesh.get_local_rank("sweep")]
    logger.info(f"Processing sweep item: {item.sae_name}/{item.sae_series}")

    initializer = Initializer(item.initializer)

    logger.info("Initializing SAE for sweep item")
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
        logger.info("WandB logger initialized for sweep item")
        wandb_logger.watch(sae, log="all")

    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    logger.info("Starting training for sweep item")
    trainer = Trainer(item.trainer)
    sae.cfg.save_hyperparameters(item.trainer.exp_result_path)
    trainer.fit(sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger)

    logger.info("Training completed, saving sweep item")
    sae.save_pretrained(
        save_path=item.trainer.exp_result_path,
        sae_name=item.sae_name,
        sae_series=item.sae_series,
        mongo_client=mongo_client,
    )

    if wandb_logger is not None:
        wandb_logger.finish()
        logger.info("WandB session closed for sweep item")

    logger.info(f"Sweep item completed: {item.sae_name}/{item.sae_series}")
