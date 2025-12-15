"""Module for evaluating SAE models."""

import os
from typing import Optional

import torch
import wandb
from pydantic_settings import BaseSettings
from torch.distributed.device_mesh import init_device_mesh

from lm_saes import ReplacementModel
from lm_saes.activation.factory import ActivationFactory
from lm_saes.clt import CrossLayerTranscoder
from lm_saes.config import (
    ActivationFactoryConfig,
    BaseSAEConfig,
    CrossCoderConfig,
    EvalConfig,
    GraphEvalConfig,
    LanguageModelConfig,
    WandbConfig,
)
from lm_saes.crosscoder import CrossCoder
from lm_saes.evaluator import Evaluator, GraphEval
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.distributed import mesh_rank
from lm_saes.utils.logging import get_distributed_logger, setup_logging

logger = get_distributed_logger("runners.eval")


class EvalGraphSettings(BaseSettings):
    model_cfg: LanguageModelConfig
    """Configuration for the language model."""

    transcoders_path: str
    """The save path of CLT."""

    lorsas_path: list
    """The save path of lorsa."""

    dataset_path: str
    """The path of evaluation json file."""

    eval: GraphEvalConfig
    """Configuration for the GraphEval"""

    device: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""

    show: bool = False

    use_lorsa: bool = True


class EvaluateSAESettings(BaseSettings):
    """Settings for evaluating a Sparse Autoencoder."""

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

    eval: EvalConfig
    """Configuration for evaluation"""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

    fold_activation_scale: bool = False
    """Whether to fold the activation scale."""

    wandb: Optional[WandbConfig] = None
    """Configuration for Weights & Biases logging"""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


def evaluate_sae(settings: EvaluateSAESettings) -> None:
    """Evaluate a SAE model.

    Args:
        settings: Configuration settings for SAE evaluation
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

    activation_factory = ActivationFactory(settings.activation_factory)

    logger.info("Loading SAE model")

    cls = {
        "crosscoder": CrossCoder,
        "sae": SparseAutoEncoder,
        "clt": CrossLayerTranscoder,
        "lorsa": LowRankSparseAttention,
    }[settings.sae.sae_type]

    sae = cls.from_config(
        settings.sae,
        device_mesh=device_mesh,
        fold_activation_scale=settings.fold_activation_scale,
    )

    logger.info(f"SAE model loaded: {type(sae).__name__}")

    wandb_logger = (
        wandb.init(
            project=settings.wandb.wandb_project,
            config=settings.model_dump(),
            name=settings.wandb.exp_name,
            entity=settings.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),  # type: ignore
        )
        if settings.wandb is not None and (device_mesh is None or mesh_rank(device_mesh) == 0)
        else None
    )

    if wandb_logger is not None:
        logger.info("WandB logger initialized")

    logger.info("Processing activations for evaluation")
    activations = activation_factory.process()
    evaluator = Evaluator(settings.eval)
    evaluator.evaluate(sae, activations, wandb_logger)
    logger.info("Evaluation completed")


def eval_graph(settings: EvalGraphSettings) -> None:
    # Set up logging
    setup_logging(level="INFO")

    logger.info("Loading transcoder and lorsa")
    transcoders = CrossLayerTranscoder.from_pretrained(
        settings.transcoders_path,
        device=settings.device,
    )

    if settings.use_lorsa:
        lorsas = [
            LowRankSparseAttention.from_pretrained(lorsa_cfg, device=settings.device)
            for lorsa_cfg in settings.lorsas_path
        ]
        # for lorsa in lorsas:
        #     lorsa.cfg.skip_bos = False
    else:
        lorsas = None

    logger.info("Loading replacement model")
    replacement_model = ReplacementModel.from_pretrained(
        settings.model_cfg,
        transcoders,
        lorsas,  # pyright: ignore[reportArgumentType]
        use_lorsa=settings.use_lorsa,
    )

    grapheval = GraphEval(settings.eval)

    grapheval.eval(
        replacement_model,
        settings.dataset_path,
        use_lorsa=settings.use_lorsa,
    )


class EvaluateCrossCoderSettings(BaseSettings):
    """Settings for evaluating a CrossCoder model."""

    sae: CrossCoderConfig
    """Configuration for the CrossCoder model architecture and parameters"""

    sae_name: str
    """Name of the SAE model. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model. Use as identifier for the SAE model in the database."""

    activation_factories: list[ActivationFactoryConfig]
    """Configuration for generating activations"""

    eval: EvalConfig
    """Configuration for evaluation"""

    wandb: Optional[WandbConfig] = None
    """Configuration for Weights & Biases logging"""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


@torch.no_grad()
def evaluate_crosscoder(settings: EvaluateCrossCoderSettings) -> None:
    """Evaluate a CrossCoder model. The key difference to evaluate_sae is that the activation factories are a list of ActivationFactoryConfig, one for each head; and the evaluating contains a device mesh transformation from head parallelism to model (feature) parallelism.

    Args:
        settings: Configuration settings for CrossCoder evaluation
    """
    # Set up logging
    setup_logging(level="INFO")

    assert (
        len(settings.activation_factories) * len(settings.activation_factories[0].hook_points) == settings.sae.n_heads
    ), "Total number of hook points must match the number of heads in the CrossCoder"

    parallel_size = len(settings.activation_factories)

    logger.info(f"Analyzing CrossCoder with {settings.sae.n_heads} heads, {parallel_size} parallel size")

    device_mesh = init_device_mesh(
        device_type=settings.device_type,
        mesh_shape=(parallel_size,),
        mesh_dim_names=("head",),
    )

    logger.info("Device meshes initialized for CrossCoder analysis")

    logger.info("Setting up activation factory for CrossCoder head")
    activation_factory = ActivationFactory(settings.activation_factories[device_mesh.get_local_rank("head")])

    logger.info("Loading CrossCoder model")
    sae = CrossCoder.from_config(settings.sae, device_mesh=device_mesh)

    wandb_logger = (
        wandb.init(
            project=settings.wandb.wandb_project,
            config=settings.model_dump(),
            name=settings.wandb.exp_name,
            entity=settings.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),  # type: ignore
        )
        if settings.wandb is not None and (device_mesh is None or mesh_rank(device_mesh) == 0)
        else None
    )

    if wandb_logger is not None:
        logger.info("WandB logger initialized")

    logger.info("Processing activations for CrossCoder evaluation")
    activations = activation_factory.process()
    evaluator = Evaluator(settings.eval)
    evaluator.evaluate(sae, activations, wandb_logger)

    logger.info("CrossCoder evaluation completed successfully")
