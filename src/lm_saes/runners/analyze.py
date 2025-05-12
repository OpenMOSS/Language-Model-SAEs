"""Module for analyzing SAE models."""

from typing import Optional

import torch
from pydantic_settings import BaseSettings
from torch.distributed.device_mesh import init_device_mesh

from lm_saes.activation.factory import ActivationFactory
from lm_saes.analysis.feature_analyzer import FeatureAnalyzer
from lm_saes.config import (
    ActivationFactoryConfig,
    BaseSAEConfig,
    CrossCoderConfig,
    FeatureAnalyzerConfig,
    LanguageModelConfig,
    MixCoderConfig,
    MongoDBConfig,
)
from lm_saes.crosscoder import CrossCoder
from lm_saes.database import MongoClient
from lm_saes.mixcoder import MixCoder
from lm_saes.runners.utils import load_config
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.misc import get_modality_tokens


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
    """Name of the tokenizer to load. Mixcoder requires a tokenizer to get the modality indices."""

    analyzer: FeatureAnalyzerConfig
    """Configuration for feature analysis"""

    mongo: MongoDBConfig
    """Configuration for the MongoDB database."""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


def analyze_sae(settings: AnalyzeSAESettings) -> None:
    """Analyze a SAE model.

    Args:
        settings: Configuration settings for SAE analysis
    """
    device_mesh = (
        init_device_mesh(
            device_type=settings.device_type,
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

    if isinstance(settings.sae, MixCoderConfig):
        sae = MixCoder.from_config(settings.sae, device_mesh=device_mesh)
    elif isinstance(settings.sae, CrossCoderConfig):
        sae = CrossCoder.from_config(settings.sae, device_mesh=device_mesh)
    else:
        sae = SparseAutoEncoder.from_config(settings.sae, device_mesh=device_mesh)

    analyzer = FeatureAnalyzer(settings.analyzer)

    activations = activation_factory.process()
    result = analyzer.analyze_chunk(activations, sae=sae, device_mesh=device_mesh)

    start_idx = 0 if device_mesh is None else device_mesh.get_local_rank("model") * len(result)
    mongo_client.add_feature_analysis(
        name="default", sae_name=settings.sae_name, sae_series=settings.sae_series, analysis=result, start_idx=start_idx
    )


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

    feature_analysis_name: str = "default"
    """Name of the feature analysis."""

    mongo: MongoDBConfig
    """Configuration for the MongoDB database."""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


@torch.no_grad()
def analyze_crosscoder(settings: AnalyzeCrossCoderSettings) -> None:
    """Analyze a CrossCoder model.

    Args:
        settings: Configuration settings for CrossCoder analysis
    """
    parallel_size = len(settings.activation_factories)
    assert parallel_size == settings.sae.n_heads, "Number of activation factories must match the number of heads"

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

    mongo_client = MongoClient(settings.mongo)

    activation_factory = ActivationFactory(settings.activation_factories[crosscoder_device_mesh.get_local_rank("head")])

    sae = CrossCoder.from_config(settings.sae, device_mesh=crosscoder_device_mesh)

    analyzer = FeatureAnalyzer(settings.analyzer)

    activations = activation_factory.process()
    result = analyzer.analyze_chunk(activations, sae=sae, device_mesh=device_mesh)

    start_idx = 0 if device_mesh is None else device_mesh.get_local_rank("model") * len(result)
    mongo_client.add_feature_analysis(
        name=settings.feature_analysis_name,
        sae_name=settings.sae_name,
        sae_series=settings.sae_series,
        analysis=result,
        start_idx=start_idx,
    )
