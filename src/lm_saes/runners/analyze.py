"""Module for analyzing SAE models."""

from typing import Optional

import torch
from pydantic_settings import BaseSettings
from torch.distributed.device_mesh import init_device_mesh

from lm_saes.activation.factory import ActivationFactory
from lm_saes.analysis.direct_logit_attributor import DirectLogitAttributor
from lm_saes.analysis.feature_analyzer import FeatureAnalyzer
from lm_saes.backend.language_model import TransformerLensLanguageModel
from lm_saes.clt import CrossLayerTranscoder
from lm_saes.config import (
    ActivationFactoryConfig,
    BaseSAEConfig,
    CLTConfig,
    CrossCoderConfig,
    DatasetConfig,
    DirectLogitAttributorConfig,
    FeatureAnalyzerConfig,
    LanguageModelConfig,
    LorsaConfig,
    MOLTConfig,
    MongoDBConfig,
    SAEConfig,
)
from lm_saes.crosscoder import CrossCoder
from lm_saes.database import MongoClient
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.molt import MixtureOfLinearTransform
from lm_saes.resource_loaders import load_dataset, load_model
from lm_saes.runners.utils import load_config
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

    model_name: Optional[str] = None
    """Name of the tokenizer to load. LORSA may require a tokenizer to get the modality indices."""

    datasets: Optional[dict[str, Optional[DatasetConfig]]] = None
    """Name to dataset config mapping. Required if using dataset sources."""

    analyzer: FeatureAnalyzerConfig
    """Configuration for feature analysis"""

    amp_dtype: torch.dtype = torch.bfloat16

    mongo: MongoDBConfig
    """Configuration for the MongoDB database."""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

    data_parallel_size: int = 1
    """Size of data parallel mesh"""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


@torch.no_grad()
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
            mesh_shape=(settings.model_parallel_size, settings.data_parallel_size),
            mesh_dim_names=("model", "data"),
        )
        if settings.model_parallel_size > 1 or settings.data_parallel_size > 1
        else None
    )

    logger.info(f"Device mesh initialized: {device_mesh}")

    mongo_client = MongoClient(settings.mongo)
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

    logger.info(f"Loading {settings.sae.sae_type} model")
    sae_cls = {
        "sae": SparseAutoEncoder,
        "clt": CrossLayerTranscoder,
        "lorsa": LowRankSparseAttention,
        "molt": MixtureOfLinearTransform,
    }[settings.sae.sae_type]
    sae = sae_cls.from_config(settings.sae, device_mesh=device_mesh)

    logger.info(f"{settings.sae.sae_type} model loaded: {type(sae).__name__}")

    analyzer = FeatureAnalyzer(settings.analyzer)
    logger.info("Feature analyzer initialized")

    logger.info("Processing activations for analysis")

    with torch.amp.autocast(device_type=settings.device_type, dtype=settings.amp_dtype):
        result = analyzer.analyze_chunk(
            activation_factory,
            sae=sae,
            device_mesh=device_mesh,
            activation_factory_process_kwargs={
                "model": model,
                "model_name": settings.model_name,
                "datasets": datasets,
            },
        )

    logger.info("Analysis completed, saving results to MongoDB")
    start_idx = 0 if device_mesh is None else device_mesh.get_local_rank("model") * len(result)
    if device_mesh is None or settings.data_parallel_size == 1 or device_mesh.get_local_rank("data") == 0:
        mongo_client.add_feature_analysis(
            name="default",
            sae_name=settings.sae_name,
            sae_series=settings.sae_series,
            analysis=result,
            start_idx=start_idx,
        )

    logger.info(f"{settings.sae.sae_type} analysis completed successfully")


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

    amp_dtype: torch.dtype = torch.bfloat16
    """ The dtype to use for outputting activations. If `None`, will not override the dtype. """

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

    with torch.amp.autocast(device_type=settings.device_type, dtype=settings.amp_dtype):
        result = analyzer.analyze_chunk(
            activation_factory,
            sae=sae,
            device_mesh=device_mesh,
        )

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


class DirectLogitAttributeSettings(BaseSettings):
    """Settings for analyzing a CrossCoder model."""

    sae: BaseSAEConfig
    """Configuration for the SAE model architecture and parameters"""

    sae_name: str
    """Name of the SAE model. Use as identifier for the SAE model in the database."""

    layer_idx: Optional[int | None] = None
    """The index of layer to DLA."""

    sae_series: str
    """Series of the SAE model. Use as identifier for the SAE model in the database."""

    model: Optional[LanguageModelConfig] = None
    """Configuration for the language model."""

    model_name: str
    """Name of the language model."""

    direct_logit_attributor: DirectLogitAttributorConfig
    """Configuration for the direct logit attributor."""

    mongo: MongoDBConfig
    """Configuration for the MongoDB database."""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""

    # model_parallel_size: int = 1
    # """Size of model parallel (tensor parallel) mesh"""

    # data_parallel_size: int = 1
    # """Size of data parallel mesh"""

    # head_parallel_size: int = 1
    # """Size of head parallel mesh"""


@torch.no_grad()
def direct_logit_attribute(settings: DirectLogitAttributeSettings) -> None:
    """Direct logit attribute a SAE model.

    Args:
        settings: Configuration settings for DirectLogitAttributor
    """
    # Set up logging
    setup_logging(level="INFO")

    # device_mesh = (
    #     init_device_mesh(
    #         device_type=settings.device_type,
    #         mesh_shape=(settings.head_parallel_size, settings.data_parallel_size, settings.model_parallel_size),
    #         mesh_dim_names=("head", "data", "model"),
    #     )
    #     if settings.head_parallel_size > 1 or settings.data_parallel_size > 1 or settings.model_parallel_size > 1
    #     else None
    # )

    mongo_client = MongoClient(settings.mongo)
    logger.info("MongoDB client initialized")

    logger.info("Loading SAE model")
    if isinstance(settings.sae, CrossCoderConfig):
        sae = CrossCoder.from_config(settings.sae)
    elif isinstance(settings.sae, SAEConfig):
        sae = SparseAutoEncoder.from_config(settings.sae)
    elif isinstance(settings.sae, CLTConfig):
        sae = CrossLayerTranscoder.from_config(settings.sae)
    elif isinstance(settings.sae, LorsaConfig):
        sae = LowRankSparseAttention.from_config(settings.sae)
    else:
        raise ValueError(f"Unsupported SAE config type: {type(settings.sae)}")

    # Load configurations
    model_cfg = load_config(
        config=settings.model,
        name=settings.model_name,
        mongo_client=mongo_client,
        config_type="model",
        required=True,
    )
    model_cfg.device = settings.device_type
    model_cfg.dtype = sae.cfg.dtype

    model = load_model(model_cfg)
    assert isinstance(model, TransformerLensLanguageModel), (
        "DirectLogitAttributor only supports TransformerLensLanguageModel as the model backend"
    )

    logger.info("Direct logit attribution")
    direct_logit_attributor = DirectLogitAttributor(settings.direct_logit_attributor)
    results = direct_logit_attributor.direct_logit_attribute(sae, model, settings.layer_idx)

    # if is_master():
    logger.info("Direct logit attribution completed, saving results to MongoDB")
    mongo_client.update_features(
        sae_name=settings.sae_name,
        sae_series=settings.sae_series,
        update_data=[{"logits": result} for result in results],
        start_idx=0,
    )
    logger.info("Direct logit attribution completed successfully")
