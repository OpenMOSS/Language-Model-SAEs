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
    CrossCoderConfig,
    DatasetConfig,
    InitializerConfig,
    LanguageModelConfig,
    MixCoderConfig,
    MongoDBConfig,
    TrainerConfig,
    WandbConfig,
)
from lm_saes.database import MongoClient
from lm_saes.initializer import Initializer
from lm_saes.resource_loaders import load_dataset, load_model
from lm_saes.runners.utils import load_config
from lm_saes.trainer import Trainer
from lm_saes.utils.misc import get_modality_tokens, is_primary_rank


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

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


def train_sae(settings: TrainSAESettings) -> None:
    """Train a SAE model.

    Args:
        settings: Configuration settings for SAE training
    """
    device_mesh = (
        init_device_mesh(
            device_type=settings.device_type,
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

            assert model_cfg is not None, (
                "Model cfg is required for multimodal mixcoder SAE for inferring text/image tokens"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name, trust_remote_code=True)
            modality_tokens = get_modality_tokens(tokenizer, model_cfg.model_name)
            for modality in modality_tokens.keys():
                modality_tokens[modality] = modality_tokens[modality].to(settings.sae.device)
            assert list(sorted(modality_tokens.keys())) == list(sorted(modality_names)), (
                "Modality names must match the keys of modality_tokens"
            )

            def activation_interceptor(
                activations: dict[str, torch.Tensor], source_idx: int
            ) -> dict[str, torch.Tensor]:
                assert "tokens" in activations, (
                    "Tokens are required for multimodal mixcoder SAE for inferring text/image tokens"
                )
                modalities = torch.zeros_like(activations["tokens"], dtype=torch.int)
                for i, modality in enumerate(modality_names):
                    mask = torch.isin(activations["tokens"], modality_tokens[modality])
                    modalities[mask] = i
                activations = activations | {"modalities": modalities}
                return activations
        else:  # Multi-lingual mixcoder SAE
            assert [source.name for source in settings.activation_factory.sources] == modality_names, (
                "Modality names must match the names of the activation sources"
            )

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

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


def train_crosscoder(settings: TrainCrossCoderSettings) -> None:
    """Train a CrossCoder.

    Args:
        settings: Configuration settings for SAE training
    """
    assert isinstance(settings.sae, CrossCoderConfig), "CrossCoderConfig is required for training a CrossCoder"
    assert len(settings.activation_factories) == settings.sae.n_heads, (
        "Number of activation factories must match the number of heads in the CrossCoder"
    )

    device_mesh = init_device_mesh(
        device_type=settings.device_type,
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

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


def sweep_sae(settings: SweepSAESettings) -> None:
    """Sweep experiments for training SAE models.

    Args:
        settings: Configuration settings for SAE sweeping
    """

    n_sweeps = len(settings.items)

    device_mesh = init_device_mesh(
        device_type=settings.device_type,
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

                assert model_cfg is not None, (
                    "Model cfg is required for multimodal mixcoder SAE for inferring text/image tokens"
                )
                tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name, trust_remote_code=True)
                modality_tokens = get_modality_tokens(tokenizer, model_cfg.model_name)
                for modality in modality_tokens.keys():
                    modality_tokens[modality] = modality_tokens[modality].to(settings.items[0].sae.device)
                assert list(sorted(modality_tokens.keys())) == list(sorted(modality_names)), (
                    "Modality names must match the keys of modality_tokens"
                )

                def activation_interceptor(
                    activations: dict[str, torch.Tensor], source_idx: int
                ) -> dict[str, torch.Tensor]:
                    assert "tokens" in activations, (
                        "Tokens are required for multimodal mixcoder SAE for inferring text/image tokens"
                    )
                    modalities = torch.zeros_like(activations["tokens"], dtype=torch.int)
                    for i, modality in enumerate(modality_names):
                        mask = torch.isin(activations["tokens"], modality_tokens[modality])
                        modalities[mask] = i
                    activations = activations | {"modalities": modalities}
                    return activations
            else:  # Multi-lingual mixcoder SAE
                assert [source.name for source in settings.activation_factory.sources] == modality_names, (
                    "Modality names must match the names of the activation sources"
                )

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
