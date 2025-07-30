"""Post-analysis processors for different SAE types.

This module provides post-processing functionality for analysis results,
allowing different SAE implementations to customize how their analysis
results are formatted and processed.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
from einops import rearrange
from torch.distributed.device_mesh import DeviceMesh

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.activation.factory import ActivationFactory
from lm_saes.utils.discrete import KeyedDiscreteMapper


class PostAnalysisProcessor(ABC):
    """Abstract base class for post-analysis processors.
    
    Each SAE type can implement its own post-analysis processor to customize
    how analysis results are formatted and processed.
    """
    
    def process(
        self,
        sae: AbstractSparseAutoEncoder,
        act_times: torch.Tensor,
        n_analyzed_tokens: int,
        max_feature_acts: torch.Tensor,
        sample_result: dict[str, dict[str, torch.Tensor]],
        mapper: KeyedDiscreteMapper,
        device_mesh: DeviceMesh | None = None,
        activation_factory: ActivationFactory | None = None,
    ) -> list[dict[str, Any]]:
        """Process analysis results into the final per-feature format.
        
        This method implements the template pattern:
        1. Calls the subclass-specific _process_tensors method
        2. Applies the standard rearrangement and conversion to lists
        
        Args:
            sae: The sparse autoencoder model
            act_times: Tensor of activation times for each feature
            n_analyzed_tokens: Number of tokens analyzed
            max_feature_acts: Tensor of maximum activation values for each feature
            sample_result: Dictionary of sampling results
            mapper: KeyedDiscreteMapper for encoding/decoding metadata
            device_mesh: Device mesh for distributed tensors
            
        Returns:
            List of dictionaries containing per-feature analysis results
        """
        # Step 1: Let subclasses process tensors (add any SAE-specific data)
        sample_result = self._process_tensors(
            sae,
            act_times,
            n_analyzed_tokens,
            max_feature_acts,
            sample_result,
            mapper,
            device_mesh,
            activation_factory
        )
        
        # Step 2: Apply standard rearrangement and conversion
        sample_result = {k: v for k, v in sample_result.items() if v is not None}
        sample_result = {
            k1: {k2: rearrange(v2, "n_samples d_sae ... -> d_sae n_samples ...") for k2, v2 in v1.items()}
            for k1, v1 in sample_result.items()
        }
        
        # Step 3: Convert to final format
        results = []
        for i in range(len(act_times)):
            feature_result = {
                "act_times": act_times[i].item(),
                "n_analyzed_tokens": n_analyzed_tokens,
                "max_feature_acts": max_feature_acts[i].item(),
                "samplings": [
                    {
                        "name": k,
                        **self._sparsify_feature_acts(v["feature_acts"][i]),
                        **self._extra_info(v, i),
                        # TODO: Filter out meta that is not string
                        **{k2: mapper.decode(k2, v[k2][i].tolist()) for k2 in mapper.keys()},
                    }
                    for k, v in sample_result.items()
                ],
            }
            results.append(feature_result)
        
        return results

    def _sparsify_feature_acts(self, feature_acts: torch.Tensor) -> dict[str, Any]:
        """Sparsify the feature acts.
        """
        feature_acts = feature_acts.to_sparse()
        return {
            "feature_acts_indices": feature_acts.indices().tolist(),
            "feature_acts_values": feature_acts.values().tolist(),
        }
    
    def _extra_info(self, v: dict[str, Any], i: int) -> dict[str, Any]:
        """Extra information to add to the feature result.
        """
        return {}

    @abstractmethod
    def _process_tensors(
        self,
        sae: AbstractSparseAutoEncoder,
        act_times: torch.Tensor,
        n_analyzed_tokens: int,
        max_feature_acts: torch.Tensor,
        sample_result: dict[str, dict[str, torch.Tensor]],
        mapper: KeyedDiscreteMapper,
        device_mesh: DeviceMesh | None = None,
        activation_factory: ActivationFactory | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Process tensors and add SAE-specific data to sample_result.
        
        This is the method that subclasses should override to add their specific processing.
        
        Args:
            sae: The sparse autoencoder model
            act_times: Tensor of activation times for each feature
            n_analyzed_tokens: Number of tokens analyzed
            max_feature_acts: Tensor of maximum activation values for each feature
            sample_result: Dictionary of sampling results
            mapper: KeyedDiscreteMapper for encoding/decoding metadata
            device_mesh: Device mesh for distributed tensors
            
        Returns:
            Updated sample_result with any additional tensor data
        """
        return sample_result


# Registry for post-analysis processors
_post_analysis_registry: dict[str, type[PostAnalysisProcessor]] = {}


def register_post_analysis_processor(sae_type: str, processor_class: type[PostAnalysisProcessor]) -> None:
    """Register a post-analysis processor for a specific SAE type.
    
    Args:
        sae_type: The SAE type identifier
        processor_class: The processor class to register
    """
    _post_analysis_registry[sae_type] = processor_class


def get_post_analysis_processor(sae_type: str) -> PostAnalysisProcessor:
    """Get the post-analysis processor for a specific SAE type.
    
    Args:
        sae_type: The SAE type identifier
        
    Returns:
        The post-analysis processor instance
        
    Raises:
        KeyError: If no processor is registered for the given SAE type
    """
    if sae_type not in _post_analysis_registry:
        raise KeyError(f"No post-analysis processor registered for SAE type: {sae_type}")
    
    return _post_analysis_registry[sae_type]()


# Import processors to register them
from .crosscoder import CrossCoderPostAnalysisProcessor
from .generic import GenericPostAnalysisProcessor
from .lorsa import LorsaPostAnalysisProcessor

__all__ = [
    "PostAnalysisProcessor",
    "register_post_analysis_processor", 
    "get_post_analysis_processor",
    "CrossCoderPostAnalysisProcessor",
    "GenericPostAnalysisProcessor",
    "LorsaPostAnalysisProcessor",
] 