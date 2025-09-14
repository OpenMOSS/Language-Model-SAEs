from lm_saes import AutoInterpSettings, LanguageModelConfig, MongoDBConfig
from lm_saes.analysis.feature_interpreter import AutoInterpConfig
from lm_saes import auto_interp
import concurrent.futures
from functools import lru_cache
from typing import Any, Optional
import json

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
    
    # sae_name: str
    # """Name of the SAE model to interpret. Use as identifier for the SAE model in the database."""

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
    
    cover: bool = False
    """Whether cover the generated interp"""

def get_feature(graph_path):
    """
    Process node data from JSON file and organize into specified list based on feature_type
    
    Args:
        graph_path (str): Path to the JSON file
        
    Returns:
        list: List containing dictionaries, each with format {"sae_name": str, "feature_idx": int}
    """
    try:
        # Read JSON file
        with open(graph_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {graph_path} does not exist")
        return []
    except json.JSONDecodeError:
        print(f"Error: File {graph_path} is not a valid JSON format")
        return []
    
    # Extract metadata and nodes from data
    metadata = data.get('metadata', {})
    nodes = data.get('nodes', [])
    
    # Get analysis names from metadata
    lorsa_analysis_name = metadata.get('lorsa_analysis_name', 'L{}Lorsa')
    clt_analysis_name = metadata.get('clt_analysis_name', 'L{}CLT-k1024')
    
    result_list = []
    
    for node in nodes:
        node_id = node.get('node_id')
        feature_type = node.get('feature_type')
        
        # Skip nodes missing required fields
        if not node_id or not feature_type:
            continue
        
        # Split node_id into components
        parts = node_id.split('_')
        if len(parts) < 3:
            continue  # Skip if node_id format is invalid
        
        try:
            # Extract layer_id and feature_id from node_id parts
            layer_id = int(parts[0])
            feature_id = int(parts[1])
        except ValueError:
            continue  # Skip if conversion to integer fails
        
        # Process based on feature_type
        if feature_type == 'lorsa':
            # For lorsa: layer_id = layer_id // 2, use clt_analysis_name
            new_layer_id = layer_id // 2
            sae_name = lorsa_analysis_name.format(new_layer_id)
        elif feature_type == 'cross layer transcoder':
            # For cross layer transcoder: layer_id = (layer_id - 1) // 2, use lorsa_analysis_name
            new_layer_id = (layer_id - 1) // 2
            sae_name = clt_analysis_name.format(new_layer_id)
        else:
            continue  # Skip other feature types
        
        # Add processed result to the list
        result_list.append({
            'sae_name': sae_name,
            'feature_idx': feature_id
        })
    
    return result_list

def auto_interp4graph(settings: AutoInterp4GraphSettings):
    """Automatically interpret features using LLMs.
    
    Args:
        settings: Configuration
    """
    setup_logging(level="INFO")
    
    # Set up MongoDB client
    mongo_client = MongoClient(settings.mongo)
    
    language_model = load_model(settings.model)
    
    feature_list = get_feature(settings.graph_path)
    
    interpreter = FeatureInterpreter(settings.auto_interp, mongo_client)
    
    @lru_cache(maxsize=None)
    def get_dataset(dataset_name: str, shard_idx: int, n_shards: int) -> Dataset:
        dataset_cfg = mongo_client.get_dataset_cfg(dataset_name)
        assert dataset_cfg is not None, f"Dataset {dataset_name} not found"
        dataset = load_dataset_shard(dataset_cfg, shard_idx, n_shards)
        return dataset
    
    for todo_feature in feature_list:
        sae_name = todo_feature['sae_name']
        feature_idx = todo_feature['feature_idx']
        print(feature_idx, sae_name)
        # continue
        feature = mongo_client.get_feature(sae_name, settings.sae_series, feature_idx)
        if feature is not None:
            if feature.interpretation is None or settings.cover:
                result = {
                        "feature_index": feature.index,
                        "sae_name": sae_name,
                        "sae_series": settings.sae_series,
                    } | interpreter.interpret_single_feature(feature, language_model, get_dataset, settings.analysis_name)
                
                interpretation = {
                    "text": result["explanation"],
                    "validation": [
                        {"method": eval_result["method"], "passed": eval_result["passed"], "detail": eval_result}
                        for eval_result in result["evaluations"]
                    ],
                    "complexity": result["complexity"],
                    "consistency": result["consistency"],
                    "detail": result["explanation_details"],
                    "passed": result["passed"],
                    "time": result["time"],
                }
                logger.info(
                    f"Updating feature {result['feature_index']} in {sae_name}\nTime: {result['time']}\nExplanation: {interpretation['text']}"
                )
                mongo_client.update_feature(
                    sae_name, result["feature_index"], {"interpretation": interpretation}, settings.sae_series
                )
            elif feature is not None:
                logger.info(
                    f"Already interp feature {feature_idx} in {sae_name}\nExplanation: {feature.interpretation}"
                )
            else:
                logger.info(
                    f"Feature {feature_idx} in {sae_name} does not exist. Please check it."
                )
        