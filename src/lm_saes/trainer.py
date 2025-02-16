import math
import os
from typing import Callable, Iterable

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor
from torch.optim import Adam, Optimizer
from tqdm import tqdm

from lm_saes.config import TrainerConfig
from lm_saes.mixcoder import MixCoder
from lm_saes.optim import get_scheduler
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.misc import all_reduce_tensor
from wandb.sdk.wandb_run import Run


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.checkpoint_thresholds: list[int] = []
        self.total_training_steps: int = 0
        self.lr_warm_up_steps: int = 0
        self.lr_cool_down_steps: int = 0
        self.k_warmup_steps: int = 0
        self.l1_coefficient_warmup_steps: int = 0
        self.cur_step: int = 0
        self.cur_tokens: int = 0
        self.optimizer: Optimizer | None = None
        self.scheduler: lr_scheduler.LRScheduler | None = None
        self.wandb_logger: Run | None = None

    def _initialize_trainer(
        self, sae: SparseAutoEncoder, activation_stream: Iterable[dict[str, Tensor]], wandb_logger: Run | None = None
    ):
        batch_size = next(iter(activation_stream))[sae.cfg.hook_point_in].shape[0]
        self.total_training_steps = self.cfg.total_training_tokens // batch_size

        def calculate_warmup_steps(warmup_steps: float | int) -> int:
            if isinstance(warmup_steps, float):
                assert 0 <= warmup_steps <= 1.0
                return int(warmup_steps * self.total_training_steps)
            return warmup_steps

        self.lr_warm_up_steps = calculate_warmup_steps(self.cfg.lr_warm_up_steps)
        self.lr_cool_down_steps = calculate_warmup_steps(self.cfg.lr_cool_down_steps)
        self.k_warmup_steps = calculate_warmup_steps(self.cfg.k_warmup_steps)
        self.l1_coefficient_warmup_steps = calculate_warmup_steps(self.cfg.l1_coefficient_warmup_steps)
        if self.cfg.n_checkpoints > 0:
            if self.cfg.check_point_save_mode == "linear":
                self.checkpoint_thresholds = list(
                    range(0, self.cfg.total_training_tokens, self.cfg.total_training_tokens // self.cfg.n_checkpoints)
                )[1:]
            elif self.cfg.check_point_save_mode == "log":
                self.checkpoint_thresholds = [
                    math.ceil(2 ** (i / self.cfg.n_checkpoints * math.log2(self.total_training_steps))) * batch_size
                    for i in range(1, self.cfg.n_checkpoints)
                ]
        self.wandb_logger = wandb_logger

    def _initialize_optimizer(self, sae: SparseAutoEncoder):
        # TODO: check if this is correct
        if isinstance(self.cfg.lr, float):
            optimizer = Adam(sae.get_parameters(), lr=self.cfg.lr, betas=self.cfg.betas)
        else:
            assert isinstance(self.cfg.lr, dict)
            assert sae.cfg.sae_type == "mixcoder"
            params = sae.get_parameters()
            assert len(params) == len(self.cfg.lr)
            for param_group in params:
                param_group["lr"] = self.cfg.lr[param_group["modality"]]
            optimizer = Adam(params, betas=self.cfg.betas)
        scheduler = get_scheduler(
            scheduler_name=self.cfg.lr_scheduler_name,
            optimizer=optimizer,
            warm_up_steps=self.lr_warm_up_steps,
            cool_down_steps=self.lr_cool_down_steps,
            training_steps=self.total_training_steps,
            lr_end_ratio=self.cfg.lr_end_ratio,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _training_step(
        self,
        sae: SparseAutoEncoder,
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        if "topk" not in sae.cfg.act_fn and self.l1_coefficient_warmup_steps > 0:
            assert self.cfg.l1_coefficient is not None
            sae.set_current_l1_coefficient(
                min(1.0, self.cur_step / self.l1_coefficient_warmup_steps) * self.cfg.l1_coefficient
            )
        elif "topk" in sae.cfg.act_fn and self.k_warmup_steps > 0:
            assert self.cfg.initial_k is not None, "initial_k must be provided"
            assert self.cfg.initial_k >= sae.cfg.top_k, "initial_k must be greater than or equal to top_k"
            sae.set_current_k(
                max(
                    sae.cfg.top_k,
                    math.ceil(
                        self.cfg.initial_k + (sae.cfg.top_k - self.cfg.initial_k) / self.k_warmup_steps * self.cur_step,
                    ),
                )
            )

        loss, (loss_data, aux_data) = sae.compute_loss(
            batch,
            sparsity_loss_type=self.cfg.sparsity_loss_type,
            tanh_stretch_coefficient=self.cfg.tanh_stretch_coefficient,
            p=self.cfg.p,
            use_batch_norm_mse=self.cfg.use_batch_norm_mse,
            return_aux_data=True,
            tokens=batch["tokens"],
        )
        loss_dict = {"loss": loss, "batch_size": batch[sae.cfg.hook_point_in].shape[0]} | loss_data | aux_data
        return loss_dict

    @torch.no_grad()
    def _log(self, sae: SparseAutoEncoder, log_info: dict, batch: dict[str, Tensor]):
        assert self.optimizer is not None, "Optimizer must be initialized"
        assert self.wandb_logger is not None, "Wandb logger must be provided"
        activation_out = batch["output"]
        did_fire = (log_info["feature_acts"] > 0).float().sum(0) > 0
        log_info["n_forward_passes_since_fired"] += 1
        log_info["n_forward_passes_since_fired"][did_fire] = 0
        log_info["act_freq_scores"] += (log_info["feature_acts"].abs() > 0).float().sum(0)
        log_info["n_frac_active_tokens"] += log_info["batch_size"]
        if (self.cur_step + 1) % self.cfg.feature_sampling_window == 0:
            feature_sparsity = log_info["act_freq_scores"] / log_info["n_frac_active_tokens"]

            log_feature_sparsity = torch.log10(feature_sparsity + 1e-10)
            wandb_log_dict = {
                "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
                "sparsity/above_1e-1": (feature_sparsity > 1e-1).sum().item(),
                "sparsity/above_1e-2": (feature_sparsity > 1e-2).sum().item(),
                "sparsity/below_1e-5": (feature_sparsity < 1e-5).sum().item(),
                "sparsity/below_1e-6": (feature_sparsity < 1e-6).sum().item(),
            }
            if sae.cfg.sae_type == "crosscoder":
                overall_act_freq_scores = all_reduce_tensor(feature_sparsity, aggregate="max")
                wandb_log_dict.update(
                    {
                        "sparsity/overall_above_1e-1": (overall_act_freq_scores > 1e-1).sum().item(),
                        "sparsity/overall_above_1e-2": (overall_act_freq_scores > 1e-2).sum().item(),
                        "sparsity/overall_below_1e-5": (overall_act_freq_scores < 1e-5).sum().item(),
                        "sparsity/overall_below_1e-6": (overall_act_freq_scores < 1e-6).sum().item(),
                    }
                )

            self.wandb_logger.log(wandb_log_dict, step=self.cur_step + 1)
            log_info["act_freq_scores"] = torch.zeros_like(log_info["act_freq_scores"])
            log_info["n_frac_active_tokens"] = torch.zeros_like(log_info["n_frac_active_tokens"])

        if (self.cur_step + 1) % self.cfg.log_frequency == 0:
            l0 = (log_info["feature_acts"] > 0).float().sum(-1).mean()
            l_rec = log_info["l_rec"].mean()
            per_token_l2_loss = (log_info["reconstructed"] - activation_out).pow(2).sum(dim=-1)
            total_variance = (activation_out - activation_out.mean(0)).pow(2).sum(dim=-1)
            l2_norm_error = per_token_l2_loss.sqrt().mean()
            l2_norm_error_ratio = l2_norm_error / activation_out.norm(p=2, dim=-1).mean()
            explained_variance = 1 - per_token_l2_loss / total_variance
            wandb_log_dict = {
                # losses
                "losses/mse_loss": l_rec.item(),
                **(
                    {"losses/sparsity_loss": log_info["l_s"].mean().item()}
                    if log_info.get("l_s", None) is not None
                    else {}
                ),
                "losses/overall_loss": log_info["loss"].item(),
                # variance explained
                "metrics/explained_variance": explained_variance.mean().item(),
                "metrics/explained_variance_std": explained_variance.std().item(),
                # sparsity
                "metrics/l0": l0.item(),
                "metrics/mean_feature_act": log_info["feature_acts"][log_info["feature_acts"].gt(0)].mean().item(),
                "metrics/l2_norm_error": l2_norm_error.item(),
                "metrics/l2_norm_error_ratio": l2_norm_error_ratio.item(),
                # norm
                "metrics/gradients_norm": log_info["grad_norm"].item(),
                # sparsity
                "sparsity/mean_passes_since_fired": log_info["n_forward_passes_since_fired"].mean().item(),
                "details/current_learning_rate": self.optimizer.param_groups[0]["lr"],
                "details/n_training_tokens": self.cur_tokens,
            }
            wandb_log_dict.update(sae.log_statistics())
            if sae.cfg.sae_type == "crosscoder":
                wandb_log_dict.update(
                    {
                        "metrics/overall_l0": all_reduce_tensor(log_info["feature_acts"], aggregate="max")
                        .gt(0)
                        .float()
                        .sum(-1)
                        .mean()
                    }
                )
            elif sae.cfg.sae_type == "mixcoder":
                assert isinstance(sae, MixCoder)
                for modality, (start, end) in sae.modality_index.items():
                    if modality == "shared":
                        continue
                    shared_start, shared_end = sae.modality_index["shared"]
                    mask = sae.get_modality_token_mask(batch["tokens"], modality)
                    feature_acts_modality = log_info["feature_acts"][mask]
                    reconstructed_modality = log_info["reconstructed"][mask]
                    activation_out_modality = activation_out[mask]
                    explained_variance_modality = 1 - (reconstructed_modality - activation_out_modality).pow(2).sum(
                        dim=-1
                    ) / (activation_out_modality - activation_out_modality.mean(0)).pow(2).sum(dim=-1)
                    token_num = mask.sum().item()
                    wandb_log_dict.update(
                        {
                            f"mixcoder_metrics/{modality}_l0": (feature_acts_modality[:, start:end] > 0)
                            .float()
                            .sum(-1)
                            .mean()
                            .item(),
                            f"mixcoder_metrics/{modality}_shared_l0": (
                                feature_acts_modality[:, shared_start:shared_end] > 0
                            )
                            .float()
                            .sum(-1)
                            .mean()
                            .item(),
                            f"mixcoder_metrics/{modality}_token_num": token_num,
                            f"mixcoder_metrics/{modality}_ev": explained_variance_modality.float().mean().item(),
                        }
                    )

            self.wandb_logger.log(wandb_log_dict, step=self.cur_step + 1)

    def _save_checkpoint(self, sae: SparseAutoEncoder):
        if len(self.checkpoint_thresholds) > 0 and self.cur_tokens >= self.checkpoint_thresholds[0]:
            path = os.path.join(
                self.cfg.exp_result_path,
                "checkpoints",
                f"{self.cur_step}.safetensors",
            )
            sae.save_checkpoint(path)
            self.checkpoint_thresholds.pop(0)

    def fit(
        self,
        sae: SparseAutoEncoder,
        activation_stream: Iterable[dict[str, Tensor]],
        eval_fn: Callable[[SparseAutoEncoder], None] | None = None,
        wandb_logger: Run | None = None,
    ):
        self._initialize_trainer(sae, activation_stream, wandb_logger)
        self._initialize_optimizer(sae)
        assert self.optimizer is not None
        assert self.scheduler is not None
        log_info = {
            "act_freq_scores": torch.zeros(sae.cfg.d_sae, device=sae.cfg.device, dtype=sae.cfg.dtype),
            "n_forward_passes_since_fired": torch.zeros(sae.cfg.d_sae, device=sae.cfg.device, dtype=sae.cfg.dtype),
            "n_frac_active_tokens": torch.tensor([0], device=sae.cfg.device, dtype=torch.int),
        }
        proc_bar = tqdm(total=self.total_training_steps, smoothing=0.001)
        for batch in activation_stream:
            proc_bar.update(1)
            sae.train()
            self.optimizer.zero_grad()
            loss_dict = self._training_step(sae, batch)
            loss_dict["loss"].backward()
            # TODO: add support for mixcoder to use different clip_grad_norm for each modality
            loss_dict["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                sae.parameters(),
                max_norm=self.cfg.clip_grad_norm if self.cfg.clip_grad_norm > 0 else math.inf,
            )
            self.optimizer.step()
            self.scheduler.step()
            if sae.cfg.force_unit_decoder_norm:
                sae.set_decoder_to_fixed_norm(value=1.0, force_exact=True)
            log_info.update(loss_dict)
            proc_bar.set_description(f"loss: {log_info['loss'].item()}")
            """
            log_info is a dict with the following keys:
            - act_freq_scores: Tensor[d_sae]
            - n_forward_passes_since_fired: Tensor[d_sae]
            - n_frac_active_tokens: Tensor[1]
            - loss: Tensor[1]
            - l_rec: Tensor[batch_size]
            - l_s: Tensor[batch_size] | None
            - feature_acts: Tensor[batch_size, d_sae]
            - reconstructed: Tensor[batch_size, d_sae]
            - hidden_pre: Tensor[batch_size, d_sae]
            """

            activation_in, activation_out = batch[sae.cfg.hook_point_in], batch[sae.cfg.hook_point_out]

            if self.wandb_logger is not None:
                self._log(sae, log_info, {"input": activation_in, "output": activation_out, "tokens": batch["tokens"]})

            if eval_fn is not None and (self.cur_step + 1) % self.cfg.eval_frequency == 0:
                eval_fn(sae)

            self._save_checkpoint(sae)

            self.cur_step += 1
            self.cur_tokens += batch[sae.cfg.hook_point_in].shape[0]
            if self.cur_tokens >= self.cfg.total_training_tokens:
                break
