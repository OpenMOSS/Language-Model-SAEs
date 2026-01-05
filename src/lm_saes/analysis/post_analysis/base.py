"""Base class and registry for post-analysis processors.

This module defines the abstract base class for post-analysis processors
and the registration mechanism for different SAE types.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from tqdm import tqdm

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.activation.factory import ActivationFactory
from lm_saes.utils.discrete import KeyedDiscreteMapper
from lm_saes.utils.logging import get_logger

# Set up logger for this module
logger = get_logger(__name__)


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
        sum_feature_acts: torch.Tensor,
        sample_result: dict[str, dict[str, torch.Tensor]],
        mapper: KeyedDiscreteMapper,
        device_mesh: DeviceMesh | None = None,
        activation_factory: ActivationFactory | None = None,
        activation_factory_process_kwargs: dict[str, Any] = {},
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
        logger.info("[PostAnalysisProcessor] Processing tensors with sae type specific logic.")
        sample_result, decoder_info = self._process_tensors(
            sae,
            act_times,
            n_analyzed_tokens,
            max_feature_acts,
            sum_feature_acts,
            sample_result,
            mapper,
            device_mesh,
            activation_factory,
            activation_factory_process_kwargs=activation_factory_process_kwargs,
        )

        # Step 2: Apply standard rearrangement and conversion
        sample_result = {k: v for k, v in sample_result.items() if v is not None}
        # sample_result = {
        #     k1: {k2: rearrange(v2, "n_samples d_sae ... -> d_sae n_samples ...") for k2, v2 in v1.items()}
        #     for k1, v1 in sample_result.items()
        # }

        # Step 3: Convert to final format
        logger.info("[PostAnalysisProcessor] Converting results to final per-feature format.")
        results = []
        
        for i in tqdm(range(len(act_times)), desc="Converting results to final per-feature format"):
            samplings = []
            for k, v in sample_result.items():
                # Filter out non-activating samples for this feature (marked as -inf in elt)
                mask = None
                if "elt" in v and isinstance(v["elt"], torch.Tensor) and v["elt"].dim() == 2:
                    mask = v["elt"][:, i] != -torch.inf

                def _maybe_mask_tensor(t: Any) -> Any:
                    if mask is None or not isinstance(t, torch.Tensor):
                        return t
                    # Only mask tensors whose first dimension matches n_samples
                    if t.dim() >= 1 and t.shape[0] == mask.shape[0]:
                        return t[mask]
                    return t

                v_view = {kk: _maybe_mask_tensor(vv) for kk, vv in v.items()}

                tmp_dict = {}
                for k2 in mapper.keys():
                    if len(v_view[k2].shape)==2:
                        tmp_dict[k2] = mapper.decode(k2, v_view[k2][:, i].tolist())
                    else:
                        tmp_dict[k2] = mapper.decode(k2, v_view[k2][:].tolist())
                samplings.append({
                    "name": k,
                    **self._sparsify_feature_acts(v_view["feature_acts"][:, i]),
                    **self._extra_info(v_view, i),
                    **tmp_dict,
                })
            feature_result = {
                "act_times": act_times[i].item(),
                "n_analyzed_tokens": n_analyzed_tokens,
                "max_feature_acts": max_feature_acts[i].item(),
                "sum_feature_acts": sum_feature_acts[i].item(),
                **(decoder_info[i] if decoder_info is not None else {}),
                "samplings": samplings,
            }
            results.append(feature_result)

        return results

    def _sparsify_feature_acts(self, feature_acts: torch.Tensor) -> dict[str, Any]:
        """Sparsify the feature acts."""
        feature_acts = feature_acts.to_sparse()
        return {
            "feature_acts_indices": feature_acts.indices().cpu().float().numpy(),
            "feature_acts_values": feature_acts.values().cpu().float().numpy(),
        }

    def _extra_info(self, sampling_data: dict[str, Any], i: int) -> dict[str, Any]:
        """Extra information to add to the feature result."""
        return {
            "context_idx": sampling_data["context_idx"][:, i].cpu().numpy(),
            "shard_idx": sampling_data["shard_idx"][:, i].cpu().numpy()
            if "shard_idx" in sampling_data
            else torch.zeros_like(sampling_data["context_idx"][:, i].cpu(), dtype=torch.int64).numpy(),
            "n_shards": sampling_data["n_shards"][:, i].cpu().numpy()
            if "n_shards" in sampling_data
            else torch.ones_like(sampling_data["context_idx"][:, i].cpu(), dtype=torch.int64).numpy(),
        }

    @abstractmethod
    def _process_tensors(
        self,
        sae: AbstractSparseAutoEncoder,
        act_times: torch.Tensor,
        n_analyzed_tokens: int,
        max_feature_acts: torch.Tensor,
        sum_feature_acts: torch.Tensor,
        sample_result: dict[str, dict[str, torch.Tensor]],
        mapper: KeyedDiscreteMapper,
        device_mesh: DeviceMesh | None = None,
        activation_factory: ActivationFactory | None = None,
        activation_factory_process_kwargs: dict[str, Any] = {},
    ) -> tuple[dict[str, dict[str, torch.Tensor]], list[dict[str, Any]] | None]:
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
            activation_factory: Activation factory
            activation_factory_process_kwargs: Keyword arguments for activation factory process
        Returns:
            Updated sample_result with any additional tensor data
        """
        return sample_result, None


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
