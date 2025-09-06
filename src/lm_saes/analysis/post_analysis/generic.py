"""Generic post-analysis processor for standard SAE types.

This module provides a generic post-processing functionality for SAE types
that don't require special handling beyond the basic analysis results.
"""
import torch
from torch.distributed.device_mesh import DeviceMesh
from typing import Any

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.activation.factory import ActivationFactory
from lm_saes.utils.discrete import KeyedDiscreteMapper

from . import PostAnalysisProcessor, register_post_analysis_processor


class GenericPostAnalysisProcessor(PostAnalysisProcessor):
    """Generic post-analysis processor for standard SAE types.
    
    This processor provides basic post-processing functionality for SAE types
    that don't require special handling beyond the standard analysis results.
    """
    
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
        activation_factory_process_kwargs: dict[str, Any] = {},
    ) -> tuple[dict[str, dict[str, torch.Tensor]], list[dict[str, Any]] | None]:
        """Generic processor doesn't add any additional tensor data."""
        return sample_result, None


# Register the processor for generic SAE types
register_post_analysis_processor("sae", GenericPostAnalysisProcessor)
register_post_analysis_processor("generic", GenericPostAnalysisProcessor) 