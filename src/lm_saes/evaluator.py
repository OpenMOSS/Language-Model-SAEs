from typing import Iterable

import torch
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer
from wandb.sdk.wandb_run import Run

from lm_saes.config import EvalConfig
from lm_saes.sae import SparseAutoEncoder


class Evaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.cur_step = 0
        self.cur_tokens = 0
        self.metrics = {}

    @torch.no_grad()
    def _evaluate_activations(
        self,
        sae: SparseAutoEncoder,
        log_info: dict[str, Tensor],
        activation_dict: dict[str, Tensor],
        n_tokens: int,
    ) -> None:
        def log_metric(metric: str, value: float) -> None:
            self.metrics[metric] = (
                torch.tensor([value], device=self.cfg.device)
                if metric not in self.metrics
                else torch.cat([self.metrics[metric], torch.tensor([value], device=self.cfg.device)])
            )

        log_metric("n_tokens", n_tokens)
        if log_info["reconstructed"] is None:
            log_info["reconstructed"] = sae.forward(activation_dict[sae.cfg.hook_point_in])
        log_info["feature_acts"] = sae.encode(activation_dict[sae.cfg.hook_point_in])
        l_rec = (log_info["reconstructed"] - activation_dict[sae.cfg.hook_point_out]).pow(2) / (
            activation_dict[sae.cfg.hook_point_out] - activation_dict[sae.cfg.hook_point_out].mean(0, keepdim=True)
        ).pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt().mean()
        log_metric("l_rec", l_rec.mean().item())
        l0 = (log_info["feature_acts"] > 0).float().sum(-1).mean()
        log_metric("l0", l0.item())
        per_token_l2_loss = (log_info["reconstructed"] - activation_dict[sae.cfg.hook_point_out]).pow(2).sum(dim=-1)
        l2_norm_error = per_token_l2_loss.sqrt().mean()
        log_metric("l2_norm_error", l2_norm_error.item())
        l2_norm_error_ratio = l2_norm_error / activation_dict[sae.cfg.hook_point_out].norm(p=2, dim=-1).mean()
        log_metric("l2_norm_error_ratio", l2_norm_error_ratio.item())
        total_variance = (
            (activation_dict[sae.cfg.hook_point_out] - activation_dict[sae.cfg.hook_point_out].mean(0))
            .pow(2)
            .sum(dim=-1)
        )
        explained_variance = 1 - per_token_l2_loss / total_variance
        log_metric("explained_variance", explained_variance.mean().item())

        did_fire = (log_info["feature_acts"] > 0).float().sum(0) > 0
        log_info["n_forward_passes_since_fired"] += 1
        log_info["n_forward_passes_since_fired"][did_fire] = 0
        log_info["act_freq_scores"] += (log_info["feature_acts"].abs() > 0).float().sum(0)
        log_info["n_frac_active_tokens"] += n_tokens
        if (self.cur_step + 1) % self.cfg.feature_sampling_window == 0:
            feature_sparsity = log_info["act_freq_scores"] / log_info["n_frac_active_tokens"]
            log_metric("mean_log10_feature_sparsity", torch.log10(feature_sparsity + 1e-10).mean().item())
            log_metric("above_1e-1", (feature_sparsity > 1e-1).sum().item())
            log_metric("above_1e-2", (feature_sparsity > 1e-2).sum().item())
            log_metric("below_1e-5", (feature_sparsity < 1e-5).sum().item())
            log_metric("below_1e-6", (feature_sparsity < 1e-6).sum().item())
            log_info["act_freq_scores"] = torch.zeros_like(log_info["act_freq_scores"])
            log_info["n_frac_active_tokens"] = torch.zeros_like(log_info["n_frac_active_tokens"])

    @torch.no_grad()
    def process_metrics(self, wandb_logger: Run) -> None:
        def calc_mean(metric: str) -> float:
            return (
                torch.sum(self.metrics[metric] * self.metrics["n_tokens"]) / torch.sum(self.metrics["n_tokens"])
            ).item()

        self.metrics["mean_log10_feature_sparsity"] = (
            torch.tensor(self.metrics["mean_log10_feature_sparsity"]).mean().item()
        )
        self.metrics["above_1e-1"] = self.metrics["above_1e-1"].mean().item()
        self.metrics["above_1e-2"] = self.metrics["above_1e-2"].mean().item()
        self.metrics["below_1e-5"] = self.metrics["below_1e-5"].mean().item()
        self.metrics["below_1e-6"] = self.metrics["below_1e-6"].mean().item()

        self.metrics["l_rec"] = calc_mean("l_rec")
        self.metrics["l0"] = calc_mean("l0")
        self.metrics["l2_norm_error"] = calc_mean("l2_norm_error")
        self.metrics["l2_norm_error_ratio"] = calc_mean("l2_norm_error_ratio")
        self.metrics["explained_variance"] = calc_mean("explained_variance")

    @torch.no_grad()
    def _evaluate_tokens(
        self, sae: SparseAutoEncoder, batch: dict[str, Tensor], model: HookedTransformer
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], int]:
        input_ids = batch["tokens"]  # shape: (seq_len)
        assert model.tokenizer is not None, "Tokenizer is required for token input"
        filter_tokens = [model.tokenizer.eos_token_id, model.tokenizer.bos_token_id, model.tokenizer.pad_token_id]
        useful_token_mask = torch.isin(input_ids, torch.tensor(filter_tokens), invert=True)  # shape: (seq_len)
        input_ids = input_ids[useful_token_mask]  # shape: (seq_len)
        loss, cache = model.run_with_cache(
            input_ids,
            return_type="loss",
            loss_per_token=True,
            names_filter=[sae.cfg.hook_point_in, sae.cfg.hook_point_out],
            return_cache_object=False,
        )
        reconstructed_activations = sae.forward(cache[sae.cfg.hook_point_in]).to(
            cache[sae.cfg.hook_point_out].dtype
        )  # shape: (seq_len, d_model)

        def replace_hook(activations: Tensor, hook_point: str) -> Tensor:
            return torch.where(useful_token_mask, reconstructed_activations, activations)

        reconstructed_loss: Tensor = model.run_with_hooks(
            input_ids,
            return_type="loss",
            fwd_hooks=[(sae.cfg.hook_point_out, replace_hook)],
            loss_per_token=True,
        )

        def get_useful_token_loss(loss: Tensor):
            return loss[useful_token_mask].sum()

        loss_dict = {
            "loss_sum": get_useful_token_loss(loss),  # type: ignore
            "loss_reconstruction_sum": get_useful_token_loss(reconstructed_loss),
            "reconstructed": reconstructed_activations,
        }

        return cache, loss_dict, int(useful_token_mask.sum().item())

    def evaluate(
        self,
        sae: SparseAutoEncoder,
        data_stream: Iterable[dict[str, Tensor]],
        wandb_logger: Run,
        model: HookedTransformer | None = None,
    ) -> None:
        log_info = {
            "n_forward_passes_since_fired": torch.zeros(sae.cfg.d_sae, device=sae.cfg.device, dtype=sae.cfg.dtype),
            "act_freq_scores": torch.zeros(sae.cfg.d_sae, device=sae.cfg.device, dtype=sae.cfg.dtype),
            "n_frac_active_tokens": torch.tensor([0], device=sae.cfg.device, dtype=torch.int),
        }
        proc_bar = tqdm(total=self.cfg.total_eval_tokens)
        for batch in data_stream:
            if not self.cfg.use_cached_activations:
                assert model is not None, "Model is required for token input"
                activation_dict, loss_reconstruction_dict, n_tokens = self._evaluate_tokens(sae, batch, model)
                log_info.update(loss_reconstruction_dict)
            else:
                activation_dict = batch
                n_tokens = activation_dict[sae.cfg.hook_point_out].shape[0]
            self._evaluate_activations(sae, log_info, activation_dict, n_tokens)
            proc_bar.update(n_tokens)
            self.cur_tokens += n_tokens
            self.cur_step += 1
            if self.cur_tokens > self.cfg.total_eval_tokens:
                break
        self.process_metrics(wandb_logger)
