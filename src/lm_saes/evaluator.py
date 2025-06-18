from typing import Iterable

import torch
import torch.distributed.tensor
from einops import reduce
from torch.distributed.tensor import DTensor
from tqdm import tqdm
from transformer_lens import HookedTransformer
from wandb.sdk.wandb_run import Run

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.config import EvalConfig
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.distributed import DimMap
from lm_saes.utils.logging import get_distributed_logger, log_metrics

logger = get_distributed_logger("evaluator")


def item(x: torch.Tensor) -> float:
    return x.item() if not isinstance(x, DTensor) else x.full_tensor().item()


class Evaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.cur_step = 0
        self.cur_tokens = 0
        self.metrics = {}

    @torch.no_grad()
    def _evaluate_activations(
        self,
        sae: AbstractSparseAutoEncoder,
        log_info: dict[str, torch.Tensor],
        activation_dict: dict[str, torch.Tensor],
        useful_token_mask: torch.Tensor,
    ) -> None:
        """Evaluate SAE activations and compute various metrics.

        Args:
            sae: Sparse autoencoder model
            log_info: Dictionary containing logging information and feature activations
            activation_dict: Dictionary of activation tensors at different hook points
            useful_token_mask: Boolean mask indicating valid tokens
        """

        def log_metric(metric: str, value: float) -> None:
            """Add or append a metric value to self.metrics."""
            self.metrics[metric] = (
                torch.tensor([value], device=self.cfg.device)
                if metric not in self.metrics
                else torch.cat([self.metrics[metric], torch.tensor([value], device=self.cfg.device)])
            )

        # 1. Basic token and loss metrics
        log_metric("n_tokens", item(useful_token_mask.sum()))
        for loss_key in [
            "loss_mean",
            "loss_reconstruction_mean",
        ]:  # when using token input loss_mean & loss_reconstruction_mean are expected to be present
            if loss_key in log_info:
                log_metric(loss_key, item(log_info.pop(loss_key)))

        # 2. Get activations and compute reconstructions
        batch = sae.normalize_activations(activation_dict)
        x, encode_kwargs = sae.prepare_input(batch)
        label = sae.prepare_label(batch)
        feature_acts = sae.encode(x, **encode_kwargs)
        reconstructed = sae.decode(feature_acts)

        # 3. Compute sparsity metrics
        l0 = (feature_acts > 0).float().sum(-1).mean()
        log_metric("l0", item(l0))

        # 4. Compute reconstruction quality metrics
        # L2 reconstruction error
        per_token_l2_loss = (reconstructed - label).pow(2).sum(dim=-1)
        l2_norm_error = per_token_l2_loss.sqrt().mean()
        log_metric("l2_norm_error", item(l2_norm_error))

        # Normalized metrics
        l2_norm_error_ratio = l2_norm_error / label.norm(p=2, dim=-1).mean()
        log_metric("l2_norm_error_ratio", item(l2_norm_error_ratio))

        activation_variance = (label - label.mean(0, keepdim=True)).pow(2)
        l_rec = (reconstructed - label).pow(2) / activation_variance.sum(dim=-1, keepdim=True).clamp(
            min=1e-8
        ).sqrt().mean()
        log_metric("l_rec", item(l_rec.mean()))

        # Explained variance
        total_variance = (label - label.mean(0)).pow(2).sum(dim=-1)
        explained_variance = 1 - per_token_l2_loss / total_variance
        log_metric("explained_variance", item(explained_variance.mean()))

        # 5. Update feature activation tracking
        log_info["act_freq_scores"] += reduce(
            (feature_acts.abs() > 0).float(),
            "... d_sae -> d_sae",
            "sum",
        )
        log_info["n_frac_active_tokens"] += item(useful_token_mask.sum())

        # 6. Periodic feature sparsity logging
        if (self.cur_step + 1) % self.cfg.feature_sampling_window == 0:
            feature_sparsity = log_info["act_freq_scores"] / log_info["n_frac_active_tokens"]

            # Log sparsity distribution
            log_metric("mean_log10_feature_sparsity", item(torch.log10(feature_sparsity + 1e-10).mean()))

            # Log sparsity thresholds
            for threshold, name in [
                (1e-1, "above_1e-1"),
                (1e-2, "above_1e-2"),
                (1e-5, "below_1e-5"),
                (1e-6, "below_1e-6"),
            ]:
                comparison = feature_sparsity > threshold if "above" in name else feature_sparsity < threshold
                log_metric(name, item(comparison.sum()))

            # Reset tracking counters
            log_info["act_freq_scores"].zero_()
            log_info["n_frac_active_tokens"].zero_()

    @torch.no_grad()
    def process_metrics(self, wandb_logger: Run | None = None) -> None:
        def calc_mean(metric: str) -> float:
            return item(
                torch.sum(self.metrics[metric] * self.metrics["n_tokens"]) / torch.sum(self.metrics["n_tokens"])
            )

        self.metrics["mean_log10_feature_sparsity"] = item(self.metrics["mean_log10_feature_sparsity"].mean())
        self.metrics["above_1e-1"] = item(self.metrics["above_1e-1"].mean())
        self.metrics["above_1e-2"] = item(self.metrics["above_1e-2"].mean())
        self.metrics["below_1e-5"] = item(self.metrics["below_1e-5"].mean())
        self.metrics["below_1e-6"] = item(self.metrics["below_1e-6"].mean())

        self.metrics["l_rec"] = calc_mean("l_rec")
        self.metrics["l0"] = calc_mean("l0")
        self.metrics["l2_norm_error"] = calc_mean("l2_norm_error")
        self.metrics["l2_norm_error_ratio"] = calc_mean("l2_norm_error_ratio")
        self.metrics["explained_variance"] = calc_mean("explained_variance")
        for loss_key in [
            "loss_mean",
            "loss_reconstruction_mean",
        ]:  # when using token input, loss_mean & loss_reconstruction_mean are expected to be present
            if loss_key in self.metrics:
                self.metrics[loss_key] = calc_mean(loss_key)

        if wandb_logger is not None:
            wandb_logger.log(self.metrics)

        log_metrics(logger.logger, self.metrics, title="Evaluation Metrics")

    @torch.no_grad()
    def _evaluate_tokens(
        self, sae: SparseAutoEncoder, batch: dict[str, torch.Tensor], model: HookedTransformer
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        input_ids = batch["tokens"]  # shape: (seq_len)
        assert model.tokenizer is not None, "Tokenizer is required for token input"
        filter_tokens = [model.tokenizer.eos_token_id, model.tokenizer.bos_token_id, model.tokenizer.pad_token_id]
        useful_token_mask = torch.isin(input_ids, torch.tensor(filter_tokens), invert=True)  # shape: (seq_len)

        loss, cache = model.run_with_cache(
            input_ids,
            return_type="loss",
            loss_per_token=True,
            names_filter=[sae.cfg.hook_point_in, sae.cfg.hook_point_out],
            return_cache_object=False,
        )
        # TODO: check normalization
        reconstructed_activations = sae.forward(cache[sae.cfg.hook_point_in], tokens=input_ids)

        def replace_hook(activations: torch.Tensor, hook_point: str) -> torch.Tensor:
            return torch.where(useful_token_mask, reconstructed_activations, activations)

        reconstructed_loss: torch.Tensor = model.run_with_hooks(
            input_ids,
            return_type="loss",
            fwd_hooks=[(sae.cfg.hook_point_out, replace_hook)],
            loss_per_token=True,
        )

        def get_useful_token_loss(loss: torch.Tensor):
            return loss[useful_token_mask].mean()

        loss_dict = {
            "loss_mean": get_useful_token_loss(loss),  # type: ignore
            "loss_reconstruction_mean": get_useful_token_loss(reconstructed_loss),
        }
        return cache, loss_dict, useful_token_mask

    def evaluate(
        self,
        sae: AbstractSparseAutoEncoder,
        data_stream: Iterable[dict[str, torch.Tensor]],
        wandb_logger: Run | None = None,
        model: HookedTransformer | None = None,
    ) -> None:
        log_info = {
            "act_freq_scores": torch.zeros(sae.cfg.d_sae, device=sae.cfg.device, dtype=sae.cfg.dtype)
            if sae.device_mesh is None
            else torch.distributed.tensor.zeros(
                sae.cfg.d_sae,
                dtype=sae.cfg.dtype,
                device_mesh=sae.device_mesh,
                placements=DimMap({"model": 0}).placements(sae.device_mesh),
            ),
            "n_frac_active_tokens": torch.tensor([0], device=sae.cfg.device, dtype=torch.int),
        }
        proc_bar = tqdm(total=self.cfg.total_eval_tokens)
        for batch in data_stream:
            if not self.cfg.use_cached_activations:
                assert model is not None, "Model is required for token input"
                assert isinstance(sae, SparseAutoEncoder), "Must be a SparseAutoEncoder for token input"
                activation_dict, loss_reconstruction_dict, useful_token_mask = self._evaluate_tokens(sae, batch, model)
                log_info.update(loss_reconstruction_dict)
            else:
                activation_dict = batch
                useful_token_mask: torch.Tensor = torch.ones(
                    batch["tokens"].shape[0], device=sae.cfg.device, dtype=torch.bool
                )
            self._evaluate_activations(sae, log_info, activation_dict, useful_token_mask)
            proc_bar.update(item(useful_token_mask.sum()))
            self.cur_tokens += item(useful_token_mask.sum())
            self.cur_step += 1
            if self.cur_tokens > self.cfg.total_eval_tokens:
                break
        self.process_metrics(wandb_logger)
