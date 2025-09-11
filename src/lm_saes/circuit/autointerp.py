from lm_saes import AutoInterpSettings, LanguageModelConfig, MongoDBConfig
from lm_saes.analysis.feature_interpreter import AutoInterpConfig
from lm_saes import auto_interp
import concurrent.futures
from functools import lru_cache
from typing import Any, Optional

from datasets import Dataset
from pydantic_settings import BaseSettings

from lm_saes.analysis.feature_interpreter import AutoInterpConfig, FeatureInterpreter
from lm_saes.config import LanguageModelConfig, MongoDBConfig
from lm_saes.database import MongoClient
from lm_saes.resource_loaders import load_dataset_shard, load_model
from lm_saes.utils.logging import get_logger, setup_logging

logger = get_logger("runners.autointerp4graph")

class AutoInterp4GraphSettings(BaseSettings):
    """Settings for automatic interpretation of SAE features."""

    graph_path: str
    """The json file path of graph to demonstrate."""
    
    sae_name: str
    """Name of the SAE model to interpret. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model to interpret. Use as identifier for the SAE model in the database."""

    model: LanguageModelConfig
    """Configuration for the language model used to generate activations."""

    model_name: str
    """Name of the model to load."""

    auto_interp: AutoInterpConfig
    """Configuration for the auto-interpretation process."""

    mongo: MongoDBConfig
    """Configuration for the MongoDB database."""

    analysis_name: str = "default"
    """Name of the analysis to use for interpretation."""

    max_workers: int = 10
    """Maximum number of workers to use for interpretation."""

def auto_interp4graph(settings: AutoInterp4GraphSettings):
    """Automatically interpret features using LLMs.
    
    Args:
        settings: Configuration
    """
    setup_logging(level="INFO")
    
    # Set up MongoDB client
    mongo_client = MongoClient(settings.mongo)

    