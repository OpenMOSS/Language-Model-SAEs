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
        batch: dict[str, torch.Tensor],
        useful_token_mask: torch.Tensor,
    ) -> None:
        """Evaluate SAE activations and compute various metrics.

        Args:
            sae: Sparse autoencoder model
            log_info: Dictionary containing logging information and feature activations
            batch: Dictionary of activation tensors at different hook points
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
        x, encode_kwargs = sae.prepare_input(batch)
        label = sae.prepare_label(batch)
        feature_acts = sae.encode(x, **encode_kwargs)
        reconstructed = sae.decode(feature_acts)

        # 3. Compute sparsity metrics
        l0 = (feature_acts > 0).float().sum(-1)
        if sae.device_mesh is not None:
            l0 = l0.full_tensor()
        if sae.cfg.sae_type == "clt":
            label = label.permute(1, 0, 2)
            reconstructed = reconstructed.permute(1, 0, 2)
            l0_dict = {
                f"l0_layer{l}": l0[:, l].mean().item() for l in range(l0.size(1))
            }
            for key, value in l0_dict.items():
                log_metric(key, value)
            
            l0 = l0.sum(-1)  # for clt, l0 is the sum of l0s of all layers

        log_metric("l0", item(l0.mean()))

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

        if sae.cfg.sae_type == "clt":
            per_layer_ev = explained_variance.mean(0)
            per_layer_ev_dict = {
                f"explained_variance_layer{l}": per_layer_ev[l].item()
                for l in range(per_layer_ev.size(0))
            }
            for key, value in per_layer_ev_dict.items():
                log_metric(key, value)
            
        # 5. Update feature activation tracking
        if sae.cfg.sae_type == "clt":
            reduce_str = "... layers d_sae -> layers d_sae"
        else:
            reduce_str = "... d_sae -> d_sae"

        log_info["act_freq_scores"] += reduce(
            (feature_acts.abs() > 0).float(),
            reduce_str,
            "sum",
        )
        log_info["n_frac_active_tokens"] += item(useful_token_mask.sum())

        # 6. Periodic feature sparsity logging
        if (self.cur_step + 1) % self.cfg.feature_sampling_window == 0:
            feature_sparsity = log_info["act_freq_scores"] / log_info["n_frac_active_tokens"]
            if sae.cfg.sae_type == "clt":
                above_1e_1 = (feature_sparsity > 1e-1).sum(-1)
                above_1e_2 = (feature_sparsity > 1e-2).sum(-1)
                below_1e_5 = (feature_sparsity < 1e-5).sum(-1)
                below_1e_6 = (feature_sparsity < 1e-6).sum(-1)
                sparsity_results = {}
                
                for l in range(sae.cfg.n_layers):
                    sparsity_results[f"above_1e-1_layer{l}"] = above_1e_1[l].item()
                    sparsity_results[f"above_1e-2_layer{l}"] = above_1e_2[l].item()

                for l in range(sae.cfg.n_layers):
                    sparsity_results[f"below_1e-5_layer{l}"] = below_1e_5[l].item()
                    sparsity_results[f"below_1e-6_layer{l}"] = below_1e_6[l].item()

                sparsity_results["above_1e-1"] = above_1e_1.sum().item()
                sparsity_results["above_1e-2"] = above_1e_2.sum().item()
                sparsity_results["below_1e-5"] = below_1e_5.sum().item()
                sparsity_results["below_1e-6"] = below_1e_6.sum().item()
            
            else:
                sparsity_results = {
                    "above_1e-1": (feature_sparsity > 1e-1).sum(-1).item(),
                    "above_1e-2": (feature_sparsity > 1e-2).sum(-1).item(),
                    "below_1e-5": (feature_sparsity < 1e-5).sum(-1).item(),
                    "below_1e-6": (feature_sparsity < 1e-6).sum(-1).item(),
                }
            
            for key, value in sparsity_results.items():
                log_metric(key, value)

            # Reset tracking counters
            log_info["act_freq_scores"].zero_()
            log_info["n_frac_active_tokens"].zero_()

    @torch.no_grad()
    def process_metrics(self, wandb_logger: Run | None = None) -> None:
        def calc_mean(metric: str) -> float:
            return item(
                torch.sum(self.metrics[metric] * self.metrics["n_tokens"]) / torch.sum(self.metrics["n_tokens"])
            )
        
        sparsity_metrics = [k for k in self.metrics.keys() if "above_1e" in k or "below_1e" in k]
        for metric in sparsity_metrics:
            self.metrics[metric] = self.metrics[metric].float().mean().item()

        self.metrics["l_rec"] = calc_mean("l_rec")
        l0_metrics = [k for k in self.metrics.keys() if "l" in k]
        for metric in l0_metrics:
            self.metrics[metric] = calc_mean(metric)

        self.metrics["l2_norm_error"] = calc_mean("l2_norm_error")
        self.metrics["l2_norm_error_ratio"] = calc_mean("l2_norm_error_ratio")
        ev_metrics = [k for k in self.metrics.keys() if "explained_variance" in k]
        for metric in ev_metrics:
            self.metrics[metric] = calc_mean(metric)

        for loss_key in [
            "loss_mean",
            "loss_reconstruction_mean",
        ]:  # when using token input, loss_mean & loss_reconstruction_mean are expected to be present
            if loss_key in self.metrics:
                self.metrics[loss_key] = calc_mean(loss_key)

        if wandb_logger is not None:
            wandb_logger.log(self.metrics)
        
        self.metrics.pop("n_tokens")
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
        act_freq_scores_shape = (sae.cfg.n_layers, sae.cfg.d_sae) if sae.cfg.sae_type == "clt" else (sae.cfg.d_sae,)  # type: ignore
        log_info = {
            "act_freq_scores": torch.zeros(act_freq_scores_shape, device=sae.cfg.device, dtype=sae.cfg.dtype)
            if sae.device_mesh is None
            else torch.distributed.tensor.zeros(
                act_freq_scores_shape,
                dtype=sae.cfg.dtype,
                device_mesh=sae.device_mesh,
                placements=DimMap({"model": -1}).placements(sae.device_mesh),
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
