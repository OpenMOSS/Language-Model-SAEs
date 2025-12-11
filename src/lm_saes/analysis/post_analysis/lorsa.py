"""Post-analysis processor for LoRSA (Low-Rank Sparse Attention) SAE.

This module provides post-processing functionality specific to LoRSA SAEs,
including handling of z patterns (attention patterns) in sparse layout.

The z patterns represent the attention patterns for each active feature in the LoRSA model.
They are stored as sparse tensors with shape (batch, context, feature, n_ctx) where:
- batch: batch dimension
- context: sequence length dimension
- feature: feature dimension (d_sae)
- n_ctx: context length for attention patterns

Example:
    When analyzing LoRSA features, the post processor extracts z patterns for each feature
    and returns them in sparse layout format:

    {
        "top_z_patterns": {
            "indices": [[0, 0, 1], [0, 1, 2], [1, 0, 1]],  # batch, context, n_ctx
            "values": [0.1, 0.2, 0.3],  # attention pattern values
            "size": [2, 3, 5]  # batch, context, n_ctx dimensions
        }
    }
"""

from typing import Any

import torch
from einops import repeat
from torch.distributed.device_mesh import DeviceMesh
from tqdm import tqdm

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.activation.factory import ActivationFactory
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.utils.discrete import KeyedDiscreteMapper
from lm_saes.utils.distributed.ops import item
from lm_saes.utils.logging import get_distributed_logger
from lm_saes.utils.misc import is_primary_rank

from .base import PostAnalysisProcessor, register_post_analysis_processor

logger = get_distributed_logger("lorsa_post_analysis")


class LorsaPostAnalysisProcessor(PostAnalysisProcessor):
    """Post-analysis processor for LoRSA SAE.

    This processor handles LoRSA-specific analysis results including:
    - Z patterns (attention patterns) in sparse layout
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
        """Process tensors and add LoRSA-specific data to sample_result.

        Args:
            sae: The sparse autoencoder model
            act_times: Tensor of activation times for each feature
            n_analyzed_tokens: Number of tokens analyzed
            max_feature_acts: Tensor of maximum activation values for each feature
            sample_result: Dictionary of sampling results
            mapper: KeyedDiscreteMapper for encoding/decoding metadata
            device_mesh: Device mesh for distributed tensors
            activation_factory: Factory for re-initializing activation stream
            activation_factory_process_kwargs: Keyword arguments for activation factory process
        Returns:
            Updated sample_result with any additional tensor data
        """
        assert isinstance(sae, LowRankSparseAttention)
        assert activation_factory is not None

        # Extract interested (shard_idx, context_idx) pairs from sample_result
        interested_pairs = []
        head_indices = []

        _feature_acts = []
        for _, sampling_data in sample_result.items():
            if sampling_data is None:
                continue

            head_index = (
                torch.arange(sampling_data["feature_acts"].shape[1], device=sae.cfg.device, dtype=torch.long)[None, :]
                .expand(sampling_data["feature_acts"].shape[0], -1)
                .flatten()
            )  # [n_samples * d_sae]
            head_indices.append(head_index)

            _feature_acts.append(sampling_data["feature_acts"])

            # Get shard_idx and context_idx from metadata
            # n_samples * d_sae
            shard_indices = sampling_data.get(
                "shard_idx", torch.zeros_like(sampling_data["feature_acts"][:, :, 0], dtype=torch.int64)
            ).flatten()
            context_indices = sampling_data["context_idx"].flatten()

            interested_pairs.append(torch.stack([shard_indices, context_indices], dim=1))
        interested_pairs = torch.cat(interested_pairs)
        head_indices = torch.cat(head_indices)
        _feature_acts = torch.cat(_feature_acts).flatten(0, 1)
        n_ctx = _feature_acts.size(-1)

        # Initialize z pattern storage
        z_pattern_data = torch.sparse_coo_tensor(
            size=(interested_pairs.shape[0], n_ctx * n_ctx),
            dtype=_feature_acts.dtype,
            device=sae.cfg.device,
        )  # type: ignore

        # Re-initialize activation stream
        activation_stream = activation_factory.process(
            **activation_factory_process_kwargs,
        )

        visited = 0
        active_head_mask = act_times.ne(0)
        pbar = tqdm(
            total=interested_pairs.shape[0],
            desc="Processing LoRSA z patterns",
            disable=not is_primary_rank(device_mesh),
        )
        # Iterate through activation stream
        for batch_data in activation_stream:
            # Extract metadata from batch
            meta = batch_data["meta"]

            for i, m in enumerate(meta):
                data_idx = torch.tensor(
                    [m.get("shard_idx", int(0)), m["context_idx"]], device=interested_pairs.device, dtype=torch.long
                )

                interested_pairs_idx = (data_idx == interested_pairs).all(dim=1)
                n_unfiltered_interested_pairs = interested_pairs_idx.sum()
                visited += n_unfiltered_interested_pairs

                interested_pairs_idx &= repeat(
                    tensor=active_head_mask,
                    pattern="d_sae -> (n_samples d_sae)",
                    n_samples=interested_pairs_idx.size(0) // active_head_mask.size(0),
                )

                if not interested_pairs_idx.any():
                    continue

                interested_heads = head_indices[interested_pairs_idx]
                interested_feature_acts = _feature_acts[interested_pairs_idx].to(torch.float32)

                z_pattern = sae.encode_z_pattern_for_head(
                    batch_data[sae.cfg.hook_point_in][i : i + 1],
                    interested_heads,
                )
                z_pattern *= interested_feature_acts.ne(0)[..., None]
                small_zp_mask = z_pattern.abs() < 1e-2 * interested_feature_acts[..., None]
                z_pattern.masked_fill_(small_zp_mask, 0.0)

                z_pattern = z_pattern.to_sparse()

                z_pattern_data += torch.sparse_coo_tensor(
                    indices=torch.cat(
                        [
                            interested_pairs_idx.nonzero().squeeze(1)[None, z_pattern.indices()[0]],
                            z_pattern.indices()[1:2] * n_ctx + z_pattern.indices()[2:],
                        ],
                    ),
                    values=z_pattern.values(),
                    size=z_pattern_data.size(),
                )

                pbar.update(item(n_unfiltered_interested_pairs))

            if visited == interested_pairs.shape[0]:
                break

        # put z_pattern_data into sample_result
        st = 0
        z_pattern_data = z_pattern_data.coalesce()
        for sampling_name, sampling_data in sample_result.items():
            # TODO: Fix type hint errors
            sampling_data["z_pattern_indices"] = []  # pyright: ignore[reportArgumentType]
            sampling_data["z_pattern_values"] = []  # pyright: ignore[reportArgumentType]

            n_samples, d_sae, n_ctx = sampling_data["feature_acts"].shape
            zp_data_mask = (z_pattern_data.indices()[0] >= st) & (z_pattern_data.indices()[0] < st + n_samples * d_sae)
            sample_feature_indices = z_pattern_data.indices()[0][zp_data_mask]
            qk_indices = z_pattern_data.indices()[1][zp_data_mask]
            zp_values = z_pattern_data.values()[zp_data_mask]

            for feature_idx in range(d_sae):
                feature_mask = (sample_feature_indices % d_sae).eq(feature_idx)
                sample_indices = sample_feature_indices[feature_mask] // d_sae
                q_idx_of_feature = qk_indices[feature_mask] // n_ctx
                k_idx_of_feature = qk_indices[feature_mask] % n_ctx

                sampling_data["z_pattern_indices"].append(
                    torch.stack([sample_indices, q_idx_of_feature, k_idx_of_feature]).cpu().numpy()
                )
                sampling_data["z_pattern_values"].append(zp_values[feature_mask].cpu().float().numpy())

            st += n_samples * d_sae

        assert st == z_pattern_data.size(0)
        return sample_result, None

    def _extra_info(self, sampling_data: dict[str, Any], i: int) -> dict[str, Any]:
        """Extra information to add to the feature result."""
        base_extra_info = super()._extra_info(sampling_data, i)
        z_pattern_indices = sampling_data["z_pattern_indices"][i]
        z_pattern_values = sampling_data["z_pattern_values"][i]

        return {
            **base_extra_info,
            "z_pattern_indices": z_pattern_indices,
            "z_pattern_values": z_pattern_values,
        }


# Register the processor for LoRSA SAE type
register_post_analysis_processor("lorsa", LorsaPostAnalysisProcessor)
