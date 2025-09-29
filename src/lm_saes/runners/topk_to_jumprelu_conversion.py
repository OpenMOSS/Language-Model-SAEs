"""Module for sweeping SAE experiments."""

from typing import Optional

import torch
from pydantic_settings import BaseSettings
from torch.distributed.device_mesh import init_device_mesh

from lm_saes.activation.factory import ActivationFactory
from lm_saes.clt import CrossLayerTranscoder
from lm_saes.config import (
    ActivationFactoryConfig,
    CLTConfig,
    DatasetConfig,
    LanguageModelConfig,
    MongoDBConfig,
)
from lm_saes.database import MongoClient
from lm_saes.resource_loaders import load_dataset, load_model
from lm_saes.runners.utils import load_config
from lm_saes.utils.logging import get_distributed_logger, setup_logging
from lm_saes.utils.topk_to_jumprelu_conversion import topk_to_jumprelu_conversion

logger = get_distributed_logger("runners.topk_to_jumprelu_conversion")


class ConvertCLTSettings(BaseSettings):
    """Settings for converting a CLT model from topk to jumprelu."""

    sae: CLTConfig
    """Configuration for the CLT model architecture and parameters"""

    sae_name: str
    """Name of the SAE model. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model. Use as identifier for the SAE model in the database."""

    activation_factory: ActivationFactoryConfig
    """Configuration for generating activations"""

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

    exp_result_path: str
    """Path to save the converted CLT model"""


@torch.no_grad()
def convert_clt(settings: ConvertCLTSettings) -> None:
    """Train a Cross Layer Transcoder (CLT) model.

    Args:
        settings: Configuration settings for CLT conversion
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

    activation_factory = ActivationFactory(
        settings.activation_factory,
        device_mesh=device_mesh,
    )

    logger.info("Processing activations stream")
    activations_stream = activation_factory.process(
        model=model,
        model_name=settings.model_name,
        datasets=datasets,
    )

    logger.info("Loading CLT")
    sae = CrossLayerTranscoder.from_config(
        settings.sae,
        device_mesh=device_mesh,
        fold_activation_scale=False,
    )

    logger.info(f"CLT loaded from {settings.sae.sae_pretrained_name_or_path}")
    logger.info("Starting CLT conversion")

    sae = topk_to_jumprelu_conversion(
        sae,
        activations_stream=activations_stream,
        device_mesh=device_mesh,
    )

    logger.info("Conversion completed, saving CLT model")
    sae.save_pretrained(
        save_path=settings.exp_result_path,
        sae_name=settings.sae_name,
        sae_series=settings.sae_series,
        mongo_client=mongo_client,
    )
    sae.cfg.save_hyperparameters(settings.exp_result_path)

    logger.info("CLT conversion completed successfully")
