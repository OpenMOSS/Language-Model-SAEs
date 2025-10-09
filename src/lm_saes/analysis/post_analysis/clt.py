"""Post-analysis processor for CLT SAE.

This module provides post-processing functionality specific to CLT SAEs,
including handling of decoder norms for cross-layer transcoders.
"""

from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.activation.factory import ActivationFactory
from lm_saes.clt import CrossLayerTranscoder
from lm_saes.utils.discrete import KeyedDiscreteMapper

from .base import PostAnalysisProcessor, register_post_analysis_processor


class CLTPostAnalysisProcessor(PostAnalysisProcessor):
    """Post-analysis processor for CLT SAE.

    This processor handles CLT-specific analysis results including:
    - Decoder norms for the specified CLT layer
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
        """Process CLT-specific tensors and add decoder norms.

        Args:
            sae: The sparse autoencoder model
            act_times: Tensor of activation times for each feature
            n_analyzed_tokens: Number of tokens analyzed
            max_feature_acts: Tensor of maximum activation values for each feature
            sample_result: Dictionary of sampling results
            mapper: KeyedDiscreteMapper for encoding/decoding metadata
            device_mesh: Device mesh for distributed tensors
            activation_factory: Activation factory for generating activations

        Returns:
            Updated sample_result with CLT-specific data
        """
        assert isinstance(sae, CrossLayerTranscoder), "CLTPostAnalysisProcessor only supports CrossLayerTranscoder SAE"

        # Compute decoder norms for the specified layer
        decoder_norms = sae.decoder_norm_per_feature()
        decoder_norms = [{"decoder_norms": dn.cpu().float().numpy()} for dn in decoder_norms.t()]

        return sample_result, decoder_norms


# Register the processor for CLT SAE type
register_post_analysis_processor("clt", CLTPostAnalysisProcessor)
