from typing import Any, Iterable, Mapping

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from lm_saes.config import FeatureAnalyzerConfig
from lm_saes.mixcoder import MixCoder
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.discrete import KeyedDiscreteMapper
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

    @torch.no_grad()
    def analyze_chunk(
        self,
        activation_stream: Iterable[dict[str, Any]],
        sae: SparseAutoEncoder,
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
        n_training_tokens = 0

        # Progress tracking
        pbar = tqdm(
            total=self.cfg.total_analyzing_tokens,
            desc="Analyzing SAE",
            smoothing=0.01,
        )

        # Initialize tracking variables
        sample_result = {k: None for k in self.cfg.subsamples.keys()}
        act_times = torch.zeros((sae.cfg.d_sae,), dtype=torch.long, device=sae.cfg.device)
        max_feature_acts = torch.zeros((sae.cfg.d_sae,), dtype=sae.cfg.dtype, device=sae.cfg.device)
        mapper = KeyedDiscreteMapper()

        # Process activation batches
        for batch in activation_stream:
            # Reshape meta to zip outer dimensions to inner
            meta = {k: [m[k] for m in batch["meta"]] for k in batch["meta"][0].keys()}

            # Get feature activations from SAE
            if isinstance(sae, MixCoder):
                feature_acts = sae.encode(batch[sae.cfg.hook_point_in], modalities=batch["modalities"])
            else:
                feature_acts = sae.encode(batch[sae.cfg.hook_point_in])
            # Update activation statistics
            act_times += feature_acts.gt(0.0).sum(dim=[0, 1])
            max_feature_acts = torch.max(max_feature_acts, feature_acts.max(dim=0).values.max(dim=0).values)

            # TODO: Filter out meta that is not string
            discrete_meta = {
                k: torch.tensor(mapper.encode(k, v), device=sae.cfg.device, dtype=torch.int32) for k, v in meta.items()
            }
            sample_result = self._process_batch(feature_acts, discrete_meta, sample_result, max_feature_acts)

            # Update progress
            n_tokens_current = batch["tokens"].numel()
            n_training_tokens += n_tokens_current
            pbar.update(n_tokens_current)

            if n_training_tokens >= self.cfg.total_analyzing_tokens:
                break

        pbar.close()

        # Filter and rearrange results
        sample_result = {k: v for k, v in sample_result.items() if v is not None}
        sample_result = {
            k1: {k2: rearrange(v2, "n_samples d_sae ... -> d_sae n_samples ...") for k2, v2 in v1.items()}
            for k1, v1 in sample_result.items()
        }

        # Format final per-feature results
        return [
            {
                "act_times": act_times[i].item(),
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
            for i in range(sae.cfg.d_sae)
        ]
