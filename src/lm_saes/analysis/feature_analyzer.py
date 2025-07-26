import warnings
from typing import Any, Iterable, Optional, cast

import torch
from einops import rearrange, repeat
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from tqdm import tqdm

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.config import FeatureAnalyzerConfig
from lm_saes.crosscoder import CrossCoder
from lm_saes.utils.discrete import KeyedDiscreteMapper
from lm_saes.utils.distributed import DimMap
from lm_saes.utils.logging import get_distributed_logger
from lm_saes.utils.misc import (
    all_gather_dict,
    all_reduce_tensor,
    get_mesh_dim_size,
    is_primary_rank,
)
from lm_saes.utils.tensor_dict import concat_dict_of_tensor, sort_dict_of_tensor
from lm_saes.utils.timer import timer

logger = get_distributed_logger(__name__)


@timer.time("update_mask_ratio_stats")
def update_mask_ratio_stats(
    mask_ratio_stats: dict[str, torch.Tensor],
    meta: dict[str, Any],
    feature_acts: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Update the mask ratio statistics. Only works for LLaDA SAEs with mask_ratio meta in 2D activations.

    Args:
        mask_ratio_stats: Dictionary mapping mask ratios to activation counts per feature
        meta: Dictionary containing the mask ratio
        feature_acts: Feature activations

    Returns:
        Updated mask ratio statistics
    """
    mask_ratios = torch.tensor(meta["mask_ratio"], device=feature_acts.device)
    unique_mask_ratios = torch.unique(mask_ratios)
    # Create all masks at once for parallel processing
    masks = mask_ratios.unsqueeze(0) == unique_mask_ratios.unsqueeze(1)  # [n_unique, batch_size]

    # Parallel computation for all mask ratios at once
    for i, mask_ratio in enumerate(unique_mask_ratios):
        mask_ratio_key = str(mask_ratio.item())
        mask = masks[i]  # [batch_size]

        if mask.any():  # Only process if there are samples with this mask ratio
            # Vectorized computation of activations for selected samples
            # Use broadcasting to avoid explicit selection
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
            masked_feature_acts = feature_acts * mask_expanded  # [batch_size, context_size, d_sae_local]
            current_activations = masked_feature_acts.gt(0.0).sum(dim=[0, 1])  # [d_sae_local]

            # Initialize tensor for this mask_ratio if first time seeing it
            if mask_ratio_key not in mask_ratio_stats:
                mask_ratio_stats[mask_ratio_key] = torch.zeros(
                    (feature_acts.size(-1),), dtype=torch.long, device=feature_acts.device
                )

            # Add current batch activations to the running total
            mask_ratio_stats[mask_ratio_key] += current_activations

    return mask_ratio_stats


class FeatureAnalyzer:
    """Analyzes feature activations from a sparse autoencoder.

    This class processes activation data from a sparse autoencoder to:
    1. Track activation statistics like frequency and magnitude
    2. Sample and store representative activations
    3. Organize results by feature for analysis
    """

    def __init__(
        self,
        cfg: FeatureAnalyzerConfig,
    ):
        """Initialize the feature analyzer.

        Args:
            cfg: Analysis configuration specifying parameters like sample sizes and thresholds
        """
        self.cfg = cfg

    def _merge_sample_results(
        self, sample_results: list[dict[str, torch.Tensor] | None], max_feature_acts: torch.Tensor, name: str
    ) -> dict[str, torch.Tensor] | None:
        def calculate_elt(feature_acts: torch.Tensor) -> torch.Tensor:
            elt = feature_acts.clamp(min=0.0).max(dim=1).values
            elt[elt > max_feature_acts.unsqueeze(0) * self.cfg.subsamples[name]["proportion"]] = -torch.inf
            return elt

        sample_results: list[dict[str, torch.Tensor]] = [
            sample_result for sample_result in sample_results if sample_result is not None
        ]
        if len(sample_results) == 0:
            return None
        merged_sample_result = None
        for sample_result in sample_results:
            if "elt" not in sample_result.keys():
                # this sample result is not sorted, so we need to truncate the feature_acts to max_length,
                # calculate the elt for each sample,
                # and then rearrange the feature_acts to [batch_size, d_sae, context_size],
                if "max_length" in self.cfg.subsamples[name].keys():
                    sample_result["feature_acts"] = sample_result["feature_acts"][
                        :, : self.cfg.subsamples[name]["max_length"], :
                    ]
                sample_result["elt"] = calculate_elt(sample_result["feature_acts"])
                sample_result["feature_acts"] = rearrange(
                    sample_result["feature_acts"], "batch_size context_size d_sae -> batch_size d_sae context_size"
                )

            if merged_sample_result is None:
                merged_sample_result = {
                    k: torch.zeros((0, *v.shape[1:]), device=v.device, dtype=v.dtype) for k, v in sample_result.items()
                }
            merged_sample_result = concat_dict_of_tensor(sample_result, merged_sample_result, dim=0)
            merged_sample_result = sort_dict_of_tensor(
                merged_sample_result, sort_dim=0, sort_key="elt", descending=True
            )
            merged_sample_result = {
                k: v[
                    : min(
                        self.cfg.subsamples[name]["n_samples"], (merged_sample_result["elt"] != -torch.inf).sum().item()
                    )
                ]
                for k, v in merged_sample_result.items()
            }
        return merged_sample_result

    @timer.time("compute_ignore_token_masks")
    def compute_ignore_token_masks(
        self, tokens: torch.Tensor, ignore_token_ids: Optional[list[int]] = None
    ) -> torch.Tensor:
        """Compute ignore token masks for the given tokens.

        Args:
            tokens: The tokens to compute the ignore token masks for
            ignore_token_ids: The token IDs to ignore
        """
        if ignore_token_ids is None:
            warnings.warn(
                "ignore_token_ids are not provided. No tokens (including pad tokens) will be filtered out. If this is intentional, set ignore_token_ids explicitly to an empty list to avoid this warning.",
                UserWarning,
                stacklevel=2,
            )
            ignore_token_ids = []
        mask = torch.ones_like(tokens, dtype=torch.bool)  # TODO: check if this is correct when tokens is a DTensor
        for token_id in ignore_token_ids:
            mask &= tokens != token_id
        return mask

    @torch.no_grad()
    def analyze_chunk(
        self,
        activation_stream: Iterable[dict[str, Any]],
        sae: AbstractSparseAutoEncoder,
        device_mesh: DeviceMesh | None = None,
    ) -> list[dict[str, Any]]:
        """Analyze feature activations for a chunk of the SAE.

        Processes activation data to:
        1. Track activation statistics
        2. Sample representative activations
        3. Organize results by feature

        Args:
            activation_stream: Iterator yielding activation batches with metadata
            sae: The sparse autoencoder model

        Returns:
            List of dictionaries containing per-feature analysis results:
            - Activation counts and maximums
            - Sampled activations with metadata
        """
        n_tokens = n_analyzed_tokens = 0

        # Progress tracking
        pbar = tqdm(
            total=self.cfg.total_analyzing_tokens,
            desc="Analyzing SAE",
            smoothing=0.01,
            disable=not is_primary_rank(device_mesh),
        )

        d_sae_local = sae.cfg.d_sae // get_mesh_dim_size(device_mesh, "model")

        # Initialize tracking variables
        sample_results = cast(
            dict[str, dict[str, torch.Tensor] | None], {name: None for name in self.cfg.subsamples.keys()}
        )
        act_times = torch.zeros((d_sae_local,), dtype=torch.long, device=sae.cfg.device)
        max_feature_acts = torch.zeros((d_sae_local,), dtype=sae.cfg.dtype, device=sae.cfg.device)
        mapper = KeyedDiscreteMapper()
        mask_ratio_stats = {}
        # Process activation batches

        for batch in activation_stream:
            # Reshape meta to zip outer dimensions to inner
            meta = {k: [m[k] for m in batch["meta"]] for k in batch["meta"][0].keys()}

            batch = sae.normalize_activations(batch)

            # Get feature activations from SAE
            x, kwargs = sae.prepare_input(batch)
            feature_acts: torch.Tensor = sae.encode(x, **kwargs)
            if isinstance(feature_acts, DTensor):
                assert device_mesh is not None, "Device mesh is required for DTensor feature activations"
                if device_mesh is not feature_acts.device_mesh:
                    feature_acts = DTensor.from_local(
                        feature_acts.redistribute(
                            placements=DimMap({"head": -1, "model": -1}).placements(feature_acts.device_mesh)
                        ).to_local(),
                        device_mesh,
                        placements=DimMap({"model": -1}).placements(device_mesh),
                    )
                    # TODO: Remove this once redistributing across device meshes is supported
                feature_acts = feature_acts.redistribute(
                    placements=DimMap({"model": -1, "data": 0}).placements(device_mesh)
                ).to_local()
            if isinstance(sae, CrossCoder):
                feature_acts = feature_acts.max(dim=-2).values
            assert feature_acts.shape == (
                batch["tokens"].shape[0] // get_mesh_dim_size(device_mesh, "data"),
                batch["tokens"].shape[1],
                d_sae_local,
            ), (
                f"feature_acts.shape: {feature_acts.shape}, expected: {(batch['tokens'].shape[0] // get_mesh_dim_size(device_mesh, 'data'), batch['tokens'].shape[1], d_sae_local)}"
            )

            # Compute ignore token masks
            ignore_token_masks = self.compute_ignore_token_masks(batch["tokens"], self.cfg.ignore_token_ids)
            if isinstance(ignore_token_masks, DTensor):
                ignore_token_masks = ignore_token_masks.to_local()
            feature_acts *= repeat(ignore_token_masks, "batch_size n_ctx -> batch_size n_ctx 1")

            # Update activation statistics
            act_times += feature_acts.gt(0.0).sum(dim=[0, 1])
            max_feature_acts = torch.max(max_feature_acts, feature_acts.max(dim=0).values.max(dim=0).values)

            if "mask_ratio" in meta.keys():
                meta["mask_ratio"] = [round(float(m), 2) for m in meta["mask_ratio"]]
                mask_ratio_stats = update_mask_ratio_stats(mask_ratio_stats, meta, feature_acts)
            # TODO: Filter out meta that is not string
            data_group = (
                device_mesh.get_group("data")
                if device_mesh is not None and get_mesh_dim_size(device_mesh, "data") > 1
                else None
            )
            discrete_meta = {
                k: torch.tensor(mapper.encode(k, v, group=data_group), device=sae.cfg.device, dtype=torch.int32)
                for k, v in meta.items()
            }
            for name in self.cfg.subsamples.keys():
                sample_results[name] = self._merge_sample_results(
                    [
                        sample_results[name],
                        {
                            "feature_acts": feature_acts,  # [batch_size, context_size, d_sae]
                            **{
                                k: repeat(v, "batch_size -> batch_size d_sae", d_sae=d_sae_local)  # [batch_size, d_sae]
                                for k, v in discrete_meta.items()
                            },
                        },
                    ],
                    max_feature_acts,
                    name,
                )

            # Update progress
            n_tokens_current = batch[
                "tokens"
            ].numel()  # n_tokens_current is the number of tokens in the current batch across all ranks, which equals to batch_size * group_size
            n_tokens += n_tokens_current
            n_analyzed_tokens += cast(int, ignore_token_masks.int().sum().item())
            pbar.update(n_tokens_current)

            if n_tokens >= self.cfg.total_analyzing_tokens:
                break

        pbar.close()

        if device_mesh is not None and get_mesh_dim_size(device_mesh, "data") > 1:
            max_feature_acts = all_reduce_tensor(max_feature_acts, "max", device_mesh.get_group("data"))
            act_times = all_reduce_tensor(act_times, "sum", device_mesh.get_group("data"))
            for name in self.cfg.subsamples.keys():
                assert sample_results[name] is not None, "Sample results are not collected for all ranks"
                gathered_sample_results = all_gather_dict(
                    cast(dict[str, torch.Tensor], sample_results[name]), device_mesh.get_group("data")
                )
                sample_results[name] = self._merge_sample_results(
                    cast(list[dict[str, torch.Tensor] | None], gathered_sample_results),
                    max_feature_acts,
                    name,
                )
            for mask_ratio, tensor in mask_ratio_stats.items():
                gathered_tensor = all_reduce_tensor(tensor, "sum", device_mesh.get_group("data"))
                mask_ratio_stats[mask_ratio] = gathered_tensor
        # Filter and rearrange results
        sample_results = {k: v for k, v in sample_results.items() if v is not None}
        sample_results = {
            name: {k: rearrange(v, "n_samples d_sae ... -> d_sae n_samples ...") for k, v in sample_result.items()}
            for name, sample_result in sample_results.items()
        }

        # Format final per-feature results
        return self._format_analysis_results(
            act_times=act_times,
            n_analyzed_tokens=n_analyzed_tokens,
            max_feature_acts=max_feature_acts,
            sample_results=sample_results,
            mapper=mapper,
            mask_ratio_stats=mask_ratio_stats,
        )

    def _format_analysis_results(
        self,
        act_times: torch.Tensor,
        n_analyzed_tokens: int,
        max_feature_acts: torch.Tensor,
        sample_results: dict[str, dict[str, torch.Tensor]],
        mapper: KeyedDiscreteMapper,
        mask_ratio_stats: dict[str, torch.Tensor] | None = None,
    ) -> list[dict[str, Any]]:
        """Format the analysis results into the final per-feature format.

        Args:
            act_times: Tensor of activation times for each feature
            n_analyzed_tokens: Number of tokens analyzed
            max_feature_acts: Tensor of maximum activation values for each feature
            sample_result: Dictionary of sampling results
            mapper: MetaMapper for encoding/decoding metadata

        Returns:
            List of dictionaries containing per-feature analysis results
        """
        results = []

        for i in range(len(act_times)):
            feature_result = {
                "act_times": act_times[i].item(),
                "n_analyzed_tokens": n_analyzed_tokens,
                "max_feature_acts": max_feature_acts[i].item(),
                "samplings": [
                    {
                        "name": name,
                        "feature_acts": sample_result["feature_acts"][i].tolist(),
                        # TODO: Filter out meta that is not string
                        **{
                            meta_key: mapper.decode(meta_key, sample_result[meta_key][i].tolist())
                            for meta_key in mapper.keys()
                        },
                    }
                    for name, sample_result in sample_results.items()
                ],
            }
            if mask_ratio_stats is not None:
                feature_result["mask_ratio_stats"] = {
                    mask_ratio: tensor[i].item() for mask_ratio, tensor in mask_ratio_stats.items()
                }

            results.append(feature_result)

        return results
