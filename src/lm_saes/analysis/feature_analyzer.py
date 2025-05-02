import warnings
from typing import Any, Iterable, Mapping, Optional, cast

import torch
from einops import rearrange, repeat
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from tqdm import tqdm

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.config import FeatureAnalyzerConfig
from lm_saes.crosscoder import CrossCoder
from lm_saes.mixcoder import MixCoder
from lm_saes.utils.discrete import KeyedDiscreteMapper
from lm_saes.utils.distributed import placements_from_dim_map
from lm_saes.utils.misc import is_primary_rank
from lm_saes.utils.tensor_dict import concat_dict_of_tensor, sort_dict_of_tensor


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

    def _process_batch(
        self,
        feature_acts: torch.Tensor,
        discrete_meta: dict[str, torch.Tensor],
        sample_result: Mapping[str, dict[str, torch.Tensor] | None],
        max_feature_acts: torch.Tensor,
    ) -> Mapping[str, dict[str, torch.Tensor] | None]:
        """Process a batch of activations to update sampling results.

        For each subsample type:
        1. Computes sampling weights if enabled
        2. Filters activations based on configured thresholds
        3. Updates running sample collections
        4. Maintains top N samples by activation magnitude

        Args:
            feature_acts: Feature activation values for current batch
            discrete_meta: Metadata tensors like dataset/context IDs
            sample_result: Current sampling results to update
            max_feature_acts: Maximum activation seen so far per feature

        Returns:
            Updated sampling results with new batch incorporated
        """
        # Compute exponential lottery ticket values for sampling if enabled
        if self.cfg.enable_sampling:
            weights = feature_acts.clamp(min=0.0).pow(self.cfg.sample_weight_exponent).max(dim=1).values
            elt = torch.rand_like(weights).log() / weights
            elt[weights == 0.0] = -torch.inf
        else:
            elt = feature_acts.clamp(min=0.0).max(dim=1).values

        # Process each subsample type (e.g. top activations)
        for name in self.cfg.subsamples.keys():
            elt_cur = elt.clone()
            # Zero out samples above the subsample threshold
            elt_cur[
                feature_acts.max(dim=1).values > max_feature_acts.unsqueeze(0) * self.cfg.subsamples[name]["proportion"]
            ] = -torch.inf

            sample_result_cur = sample_result[name]

            # Prepare batch data with proper dimensions
            batch_data = {
                "elt": elt_cur,
                "feature_acts": rearrange(
                    feature_acts,
                    "batch_size context_size d_sae -> batch_size d_sae context_size",
                ),
                **{
                    k: repeat(
                        v,
                        "batch_size -> batch_size d_sae",
                        d_sae=feature_acts.size(-1),
                    )
                    for k, v in discrete_meta.items()
                },
            }

            # Initialize or update sample collection
            if sample_result_cur is None:
                sample_result_cur = batch_data
            elif (
                sample_result_cur["elt"].size(0) > 0
                and (elt_cur.max(dim=0).values > sample_result_cur["elt"][-1]).any()
            ):
                sample_result_cur = concat_dict_of_tensor(
                    sample_result_cur,
                    batch_data,
                )
            else:  # Skip if all activations are below the threshold
                continue

            # Sort and keep top N samples
            sample_result_cur = sort_dict_of_tensor(sample_result_cur, sort_dim=0, sort_key="elt", descending=True)
            sample_result_cur = {
                k: v[
                    : min(self.cfg.subsamples[name]["n_samples"], (sample_result_cur["elt"] != -torch.inf).sum().item())
                ]
                for k, v in sample_result_cur.items()
            }

            # Update main sample result with current batch results out of place
            sample_result = {**sample_result, name: sample_result_cur}

        return sample_result

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
        mask = torch.ones_like(tokens, dtype=torch.bool)
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

        if device_mesh is not None and device_mesh.mesh_dim_names is not None and "model" in device_mesh.mesh_dim_names:
            d_sae_local = sae.cfg.d_sae // device_mesh["model"].size()
        else:
            d_sae_local = sae.cfg.d_sae

        # Initialize tracking variables
        sample_result = {k: None for k in self.cfg.subsamples.keys()}
        act_times = torch.zeros((d_sae_local,), dtype=torch.long, device=sae.cfg.device)
        max_feature_acts = torch.zeros((d_sae_local,), dtype=sae.cfg.dtype, device=sae.cfg.device)
        mapper = KeyedDiscreteMapper()

        if isinstance(sae, MixCoder):
            act_times_modalities = {
                k: torch.zeros((d_sae_local,), dtype=torch.long, device=sae.cfg.device) for k in sae.cfg.modality_names
            }
            max_feature_acts_modalities = {
                k: torch.zeros((d_sae_local,), dtype=sae.cfg.dtype, device=sae.cfg.device)
                for k in sae.cfg.modality_names
            }
        else:
            act_times_modalities = None
            max_feature_acts_modalities = None

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
                        feature_acts.full_tensor(), device_mesh, run_check=True
                    )  # Convert to full tensor and distribute on the new device mesh since redistributing across device meshes is not supported yet
                    # TODO: Remove this once redistributing across device meshes is supported
                feature_acts = feature_acts.redistribute(
                    placements=placements_from_dim_map({"model": -1}, device_mesh)
                ).to_local()
            if isinstance(sae, CrossCoder):
                feature_acts = feature_acts.max(dim=-2).values
            assert feature_acts.shape == (batch["tokens"].shape[0], batch["tokens"].shape[1], d_sae_local), (
                f"feature_acts.shape: {feature_acts.shape}, expected: {(batch['tokens'].shape[0], batch['tokens'].shape[1], d_sae_local)}"
            )

            # Compute ignore token masks
            ignore_token_masks = self.compute_ignore_token_masks(batch["tokens"], self.cfg.ignore_token_ids)
            feature_acts *= repeat(ignore_token_masks, "batch_size n_ctx -> batch_size n_ctx 1")

            # Update activation statistics
            act_times += feature_acts.gt(0.0).sum(dim=[0, 1])
            max_feature_acts = torch.max(max_feature_acts, feature_acts.max(dim=0).values.max(dim=0).values)

            if isinstance(sae, MixCoder):
                assert act_times_modalities is not None and max_feature_acts_modalities is not None
                for i, k in enumerate(sae.cfg.modality_names):
                    feature_acts_modality = feature_acts * (batch["modalities"] == i).long().unsqueeze(-1)
                    act_times_modalities[k] += feature_acts_modality.gt(0.0).sum(dim=[0, 1])
                    max_feature_acts_modalities[k] = torch.max(
                        max_feature_acts_modalities[k], feature_acts_modality.max(dim=0).values.max(dim=0).values
                    )

            # TODO: Filter out meta that is not string
            discrete_meta = {
                k: torch.tensor(mapper.encode(k, v), device=sae.cfg.device, dtype=torch.int32) for k, v in meta.items()
            }
            sample_result = self._process_batch(feature_acts, discrete_meta, sample_result, max_feature_acts)

            # Update progress
            n_tokens_current = batch["tokens"].numel()
            n_tokens += n_tokens_current
            n_analyzed_tokens += cast(int, ignore_token_masks.int().sum().item())
            pbar.update(n_tokens_current)

            if n_tokens >= self.cfg.total_analyzing_tokens:
                break

        pbar.close()

        # Filter and rearrange results
        sample_result = {k: v for k, v in sample_result.items() if v is not None}
        sample_result = {
            k1: {k2: rearrange(v2, "n_samples d_sae ... -> d_sae n_samples ...") for k2, v2 in v1.items()}
            for k1, v1 in sample_result.items()
        }

        # Format final per-feature results
        return self._format_analysis_results(
            sae=sae,
            act_times=act_times,
            n_analyzed_tokens=n_analyzed_tokens,
            max_feature_acts=max_feature_acts,
            sample_result=sample_result,
            mapper=mapper,
            act_times_modalities=act_times_modalities,
            max_feature_acts_modalities=max_feature_acts_modalities,
            device_mesh=device_mesh,
        )

    def _format_analysis_results(
        self,
        sae: AbstractSparseAutoEncoder,
        act_times: torch.Tensor,
        n_analyzed_tokens: int,
        max_feature_acts: torch.Tensor,
        sample_result: dict[str, dict[str, torch.Tensor]],
        mapper: KeyedDiscreteMapper,
        act_times_modalities: dict[str, torch.Tensor] | None = None,
        max_feature_acts_modalities: dict[str, torch.Tensor] | None = None,
        device_mesh: DeviceMesh | None = None,
    ) -> list[dict[str, Any]]:
        """Format the analysis results into the final per-feature format.

        Args:
            sae: The sparse autoencoder model
            act_times: Tensor of activation times for each feature
            n_analyzed_tokens: Number of tokens analyzed
            max_feature_acts: Tensor of maximum activation values for each feature
            sample_result: Dictionary of sampling results
            mapper: MetaMapper for encoding/decoding metadata
            act_times_modalities: Optional dictionary of activation times per modality (for MixCoder)
            max_feature_acts_modalities: Optional dictionary of maximum activation values per modality (for MixCoder)

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
                        "name": k,
                        "feature_acts": v["feature_acts"][i].tolist(),
                        # TODO: Filter out meta that is not string
                        **{k2: mapper.decode(k2, v[k2][i].tolist()) for k2 in mapper.keys()},
                    }
                    for k, v in sample_result.items()
                ],
            }

            if isinstance(sae, CrossCoder):
                decoder_norms = sae.decoder_norm()
                if isinstance(decoder_norms, DTensor):
                    assert device_mesh is not None, "Device mesh is required for DTensor decoder norms"
                    if decoder_norms.device_mesh is not device_mesh:
                        decoder_norms = DTensor.from_local(
                            decoder_norms.full_tensor(), device_mesh, run_check=True
                        )  # Convert to full tensor and distribute on the new device mesh since redistributing across device meshes is not supported yet
                        # TODO: Remove this once redistributing across device meshes is supported

                    decoder_norms = decoder_norms.redistribute(
                        placements=placements_from_dim_map({"model": -1}, device_mesh)
                    ).to_local()
                assert decoder_norms.shape[-1] == len(act_times), (
                    f"decoder_norms.shape: {decoder_norms.shape}, expected d_sae dim to match act_times length: {len(act_times)}"
                )
                feature_result["decoder_norms"] = decoder_norms[:, i].tolist()

            # Add modality-specific metrics for MixCoder
            if (
                isinstance(sae, MixCoder)
                and act_times_modalities is not None
                and max_feature_acts_modalities is not None
            ):
                feature_result["act_times_modalities"] = {k: v[i].item() for k, v in act_times_modalities.items()}
                feature_result["max_feature_acts_modalities"] = {
                    k: v[i].item() for k, v in max_feature_acts_modalities.items()
                }

            results.append(feature_result)

        return results
