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

from lm_saes import ReplacementModel, LanguageModelConfig, CrossLayerTranscoder, LowRankSparseAttention
from lm_saes.database import MongoClient
from lm_saes.initializer import Initializer
from lm_saes.resource_loaders import load_dataset, load_model
from lm_saes.runners.utils import load_config
from lm_saes.trainer import Trainer
from lm_saes.utils.logging import get_distributed_logger, setup_logging
from lm_saes.utils.misc import is_primary_rank

logger = get_distributed_logger("runners.eval_trace")

class EvalGraphSettings(BaseSettings):
    """Settings for evaluating a attribution graph."""
    
    model: LanguageModelConfig
    """Configuration for the language model."""
    
    model_name: Optional[str] = None
    """Name of the tokenizer to load. Mixcoder requires a tokenizer to get the modality indices."""
    
    clt: str
    """Save path of CrossLayerTranscoder"""
    
    lorsas: list
    """Save path of LORSA"""
    
    activation_factory: DatasetConfig
    """Configuration for the dataset"""
    
    device_type: str = 'cuda'
    """Device type to use for distributed training ('cuda' or 'cpu')"""
    
    


def evaluate(settings: EvalGraphSettings):
    # Set up logging
    setup_logging(level="INFO")
    
    # Initialize device mesh
    device_mesh = (
        init_device_mesh(
            device_type=settings.device_type,
            mesh_shape=(int(os.environ.get("WORLD_SIZE", 1)), 1),
            mesh_dim_names=("data", "model"),
        )
        if os.environ.get("WORLD_SIZE") is not None
        else None
    )
    
    logger.info(f"Device mesh initialized: {device_mesh}")
    
    activation_factory = ActivationFactory(settings.activation_factory)
    
    # loading clt model
    logger.info(f"Loading clt model")
    clt = CrossLayerTranscoder.from_config(settings.clt, device_mesh=device_mesh)
    
    # loading lorsa model
    logger.info(f"Loading lorsa model")
    lorsas = [
        LowRankSparseAttention.from_pretrained(lorsa, device_mesh=device_mesh)
        for lorsa in settings.lorsas
    ]
    
    # loading replacementmodel
    logger.info("Loading model")
    model = ReplacementModel.from_pretrained(settings.model, clt, lorsas)
    
    

    