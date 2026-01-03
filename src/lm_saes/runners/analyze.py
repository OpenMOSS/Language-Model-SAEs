"""Module for analyzing SAE models."""

import pickle
import shutil
from pathlib import Path
from typing import Any, Optional

import torch
from pydantic_settings import BaseSettings
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.activation.factory import ActivationFactory, ActivationFactoryConfig
from lm_saes.analysis.direct_logit_attributor import DirectLogitAttributor, DirectLogitAttributorConfig
from lm_saes.analysis.feature_analyzer import FeatureAnalyzer, FeatureAnalyzerConfig
from lm_saes.backend.language_model import LanguageModelConfig, TransformerLensLanguageModel
from lm_saes.config import DatasetConfig
from lm_saes.crosscoder import CrossCoder
from lm_saes.database import MongoClient, MongoDBConfig
from lm_saes.resource_loaders import load_dataset, load_model
from lm_saes.utils.distributed.utils import broadcast_object
from lm_saes.utils.logging import get_distributed_logger, setup_logging

from .utils import PretrainedSAE, load_config

logger = get_distributed_logger("runners.analyze")


def save_analysis_to_file(
    output_dir: Path,
    analysis_name: str,
    sae_name: str,
    sae_series: str,
    analysis: list[dict[str, Any]],
    start_idx: int,
    device_mesh: DeviceMesh | None = None,
) -> Path:
    """Save analysis results to file.

    Args:
        output_dir: Output directory for analysis results.
        analysis_name: Name of the analysis.
        sae_name: Name of the SAE model.
        sae_series: Series of the SAE model.
        analysis: List of analysis results (should be already gathered if in distributed mode).
        start_idx: Starting index for feature indexing.
        device_mesh: Optional device mesh for distributed execution. If provided, results will be
            sharded across ranks and then merged by rank 0.

    Returns:
        Path to the saved pickle file.
    """
    analysis_dir = output_dir / sae_series / sae_name
    for idx, analysis_item in enumerate(analysis):
        analysis_item["feature_idx"] = idx + start_idx

    if device_mesh is None:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        pickle_path = analysis_dir / f"{analysis_name}_analysis_{len(analysis)}_features.pkl"

        with open(pickle_path, "wb") as f:
            pickle.dump(analysis, f)
        return pickle_path
    else:
        shards_dir = analysis_dir / f"{analysis_name}_shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        shard_pickle_path = shards_dir / f"{start_idx}_{start_idx + len(analysis) - 1}.pkl"
        with open(shard_pickle_path, "wb") as f:
            pickle.dump(analysis, f)

        torch.distributed.barrier(group=device_mesh.get_group("model"))

        merged_pickle_path = None
        if device_mesh.get_local_rank("model") == 0:
            # get all file in the shards directory
            pickle_paths = [p for p in shards_dir.glob("*.pkl")]
            # merge the pickles
            merged_analysis = []
            for pickle_path in pickle_paths:
                with open(pickle_path, "rb") as f:
                    merged_analysis.extend(pickle.load(f))
            # save the merged analysis
            merged_analysis.sort(key=lambda x: x["feature_idx"])
            merged_pickle_path = analysis_dir / f"{analysis_name}_analysis_{len(merged_analysis)}_features.pkl"
            with open(merged_pickle_path, "wb") as f:
                pickle.dump(merged_analysis, f)
            # remove the shards directory
            shutil.rmtree(shards_dir)

        merged_pickle_path = broadcast_object(merged_pickle_path, group_src=0, group=device_mesh.get_group("model"))
        return merged_pickle_path


class AnalyzeSAESettings(BaseSettings):
    """Settings for analyzing a Sparse Autoencoder."""

    sae: PretrainedSAE
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

    feature_analysis_name: str = "default"
    """Name of the feature analysis."""

    mongo: MongoDBConfig | None = None
    """Configuration for the MongoDB database."""

    output_dir: Optional[Path] = None
    """Directory to save analysis results. Only used if MongoDB client is not provided."""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

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
            mesh_shape=(settings.model_parallel_size,),
            mesh_dim_names=("model",),
        )
        if settings.model_parallel_size > 1
        else None
    )

    logger.info(f"Device mesh initialized: {device_mesh}")

    mongo_client = None
    if settings.mongo is not None:
        mongo_client = MongoClient(settings.mongo)
        logger.info("MongoDB client initialized")
    else:
        assert settings.output_dir is not None, "Output directory must be provided if MongoDB client is not provided"
        logger.info(f"Analysis results will be saved to {settings.output_dir}")

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

    activation_factory = ActivationFactory(settings.activation_factory, device_mesh=device_mesh)

    sae = AbstractSparseAutoEncoder.from_pretrained(
        settings.sae.pretrained_name_or_path,
        device_mesh=device_mesh,
        device=settings.sae.device,
        dtype=settings.sae.dtype,
        fold_activation_scale=settings.sae.fold_activation_scale,
        strict_loading=settings.sae.strict_loading,
    )

    logger.info(f"SAE model loaded: {type(sae).__name__}")

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

    logger.info("Analysis completed, saving results")
    start_idx = 0 if device_mesh is None else device_mesh.get_local_rank("model") * len(result)
    if mongo_client is not None:
        logger.info("Saving results to MongoDB")
        mongo_client.add_feature_analysis(
            name=settings.feature_analysis_name,
            sae_name=settings.sae_name,
            sae_series=settings.sae_series,
            analysis=result,
            start_idx=start_idx,
        )
        logger.info("Results saved to MongoDB")
    else:
        assert settings.output_dir is not None, "Output directory must be set when MongoDB is not used"
        logger.info(f"Saving results to output directory: {settings.output_dir}")
        pickle_path = save_analysis_to_file(
            output_dir=settings.output_dir,
            analysis_name=settings.feature_analysis_name,
            sae_name=settings.sae_name,
            sae_series=settings.sae_series,
            analysis=result,
            start_idx=start_idx,
            device_mesh=device_mesh,
        )
        logger.info(f"Results saved to: {pickle_path}")

    logger.info("SAE analysis completed successfully")


class AnalyzeCrossCoderSettings(BaseSettings):
    """Settings for analyzing a CrossCoder model."""

    sae: PretrainedSAE
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

    mongo: MongoDBConfig | None = None
    """Configuration for the MongoDB database."""

    output_dir: Optional[Path] = None
    """Directory to save analysis results. Only used if MongoDB client is not provided."""

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

    parallel_size = len(settings.activation_factories)

    logger.info(f"Analyzing CrossCoder with {parallel_size} parallel size")

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

    mongo_client = None
    if settings.mongo is not None:
        mongo_client = MongoClient(settings.mongo)
        logger.info("MongoDB client initialized")
    else:
        assert settings.output_dir is not None, "Output directory must be provided if MongoDB client is not provided"
        logger.info(f"Analysis results will be saved to: {settings.output_dir}")

    logger.info("Setting up activation factory for CrossCoder head")
    activation_factory = ActivationFactory(settings.activation_factories[crosscoder_device_mesh.get_local_rank("head")])

    logger.info("Loading CrossCoder model")
    sae = CrossCoder.from_pretrained(
        settings.sae.pretrained_name_or_path,
        device_mesh=crosscoder_device_mesh,
        device=settings.sae.device,
        dtype=settings.sae.dtype,
        fold_activation_scale=settings.sae.fold_activation_scale,
        strict_loading=settings.sae.strict_loading,
    )

    assert len(settings.activation_factories) * len(settings.activation_factories[0].hook_points) == sae.cfg.n_heads, (
        "Total number of hook points must match the number of heads in the CrossCoder"
    )

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
    if mongo_client is not None:
        mongo_client.add_feature_analysis(
            name=settings.feature_analysis_name,
            sae_name=settings.sae_name,
            sae_series=settings.sae_series,
            analysis=result,
            start_idx=start_idx,
        )
    else:
        assert settings.output_dir is not None, "Output directory must be set when MongoDB is not used"
        logger.info(f"Saving results to output directory: {settings.output_dir}")
        pickle_path = save_analysis_to_file(
            output_dir=settings.output_dir,
            analysis_name=settings.feature_analysis_name,
            sae_name=settings.sae_name,
            sae_series=settings.sae_series,
            analysis=result,
            start_idx=start_idx,
            device_mesh=device_mesh,
        )
        logger.info(f"Results saved to: {pickle_path}")

    logger.info("CrossCoder analysis completed successfully")


class DirectLogitAttributeSettings(BaseSettings):
    """Settings for analyzing a CrossCoder model."""

    sae: PretrainedSAE
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

    mongo: MongoDBConfig | None = None
    """Configuration for the MongoDB database."""

    analysis_file: Path | None = None
    """The analysis results file to be updated. Only used if MongoDB client is not provided."""

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

    mongo_client = None
    if settings.mongo is not None:
        mongo_client = MongoClient(settings.mongo)
        logger.info("MongoDB client initialized")
    else:
        assert settings.analysis_file is not None, (
            "Analysis directory must be provided if MongoDB client is not provided"
        )
        # the analysis directory should contain the analysis results to be updated
        logger.info(f"Analysis results to be updated: {settings.analysis_file}")

    logger.info("Loading SAE model")
    sae = AbstractSparseAutoEncoder.from_pretrained(
        settings.sae.pretrained_name_or_path,
        device=settings.sae.device,
        dtype=settings.sae.dtype,
        fold_activation_scale=settings.sae.fold_activation_scale,
        strict_loading=settings.sae.strict_loading,
    )

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
    if mongo_client is not None:
        logger.info("Direct logit attribution completed, saving results to MongoDB")
        mongo_client.update_features(
            sae_name=settings.sae_name,
            sae_series=settings.sae_series,
            update_data=[{"logits": result} for result in results],
            start_idx=0,
        )
    else:
        assert settings.analysis_file is not None, "analysis_file must be set when MongoDB is not used"
        logger.info(f"Loading analysis results from: {settings.analysis_file}")

        # Load existing analysis results
        with open(settings.analysis_file, "rb") as f:
            analysis_data = pickle.load(f)

        # Update each feature with logits
        assert len(analysis_data) == len(results), (
            f"Number of features in analysis file ({len(analysis_data)}) does not match "
            f"number of results from direct logit attribution ({len(results)})"
        )

        for i, result in enumerate(results):
            assert analysis_data[i]["feature_idx"] == i, "Feature index mismatch"
            analysis_data[i]["logits"] = result

        # Save updated analysis back to file
        logger.info(f"Saving updated analysis results to: {settings.analysis_file}")
        with open(settings.analysis_file, "wb") as f:
            pickle.dump(analysis_data, f)

        logger.info(f"Updated {len(results)} features with logit attributions")

    logger.info("Direct logit attribution completed successfully")
