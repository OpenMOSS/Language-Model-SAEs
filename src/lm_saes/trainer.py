import math
import os
from typing import Callable, Iterable

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor
from torch.optim import Adam, Optimizer
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from lm_saes.config import TrainerConfig
from lm_saes.optim import get_scheduler
from lm_saes.sae import SparseAutoEncoder


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
        optimizer = Adam(sae.parameters(), lr=self.cfg.lr, betas=self.cfg.betas)
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
        if (not sae.cfg.act_fn == "topk") and self.l1_coefficient_warmup_steps > 0:
            assert self.cfg.l1_coefficient is not None
            sae.set_current_l1_coefficient(
                min(1.0, self.cur_step / self.l1_coefficient_warmup_steps) * self.cfg.l1_coefficient
            )
        elif self.k_warmup_steps > 0:
            assert self.cfg.initial_k is not None, "initial_k must be provided"
            sae.set_current_k(
                math.ceil(
                    max(
                        1.0,
                        self.cfg.initial_k
                        + (1 - self.cfg.initial_k) / self.k_warmup_steps * self.cur_step,  # d_model / top_k
                    )
                    * sae.cfg.top_k
                )
            )

        activation_in, activation_out = batch[sae.cfg.hook_point_in], batch[sae.cfg.hook_point_out]
        loss, (loss_data, aux_data) = sae.compute_loss(
            x=activation_in,
            label=activation_out,
            lp=self.cfg.lp,
            use_batch_norm_mse=self.cfg.use_batch_norm_mse,
            return_aux_data=True,
        )
        loss_dict = {"loss": loss, "batch_size": activation_in.shape[0]} | loss_data | aux_data
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
            decoder_norm = sae.decoder_norm().mean()
            encoder_norm = sae.encoder_norm().mean()
            wandb_log_dict = {
                # losses
                "losses/mse_loss": l_rec.item(),
                "losses/overall_loss": log_info["loss"].item(),
                # variance explained
                "metrics/explained_variance": explained_variance.mean().item(),
                "metrics/explained_variance_std": explained_variance.std().item(),
                # sparsity
                "metrics/l0": l0.item(),
                "metrics/l2_norm_error": l2_norm_error.item(),
                "metrics/l2_norm_error_ratio": l2_norm_error_ratio.item(),
                # norm
                "metrics/decoder_norm": decoder_norm.item(),
                "metrics/encoder_norm": encoder_norm.item(),
                "metrics/encoder_bias_norm": sae.encoder.bias.norm().item(),
                "metrics/gradients_norm": log_info["grad_norm"].item(),
                # sparsity
                "sparsity/mean_passes_since_fired": log_info["n_forward_passes_since_fired"].mean().item(),
                "details/current_learning_rate": self.optimizer.param_groups[0]["lr"],
                "details/n_training_tokens": self.cur_tokens,
            }
            if sae.cfg.use_decoder_bias:
                wandb_log_dict["metrics/decoder_bias_norm"] = sae.decoder.bias.norm().item()
            if sae.cfg.act_fn == "topk":
                wandb_log_dict["sparsity/k"] = sae.current_k
            else:
                wandb_log_dict["sparsity/l1_coefficient"] = sae.current_l1_coefficient

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
            loss_dict["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                sae.parameters(),
                max_norm=self.cfg.clip_grad_norm if self.cfg.clip_grad_norm > 0 else math.inf,
            )
            self.optimizer.step()
            self.scheduler.step()
            if not sae.cfg.sparsity_include_decoder_norm:
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
            - l_lp: Tensor[batch_size] | None
            - feature_acts: Tensor[batch_size, d_sae]
            - reconstructed: Tensor[batch_size, d_sae]
            - hidden_pre: Tensor[batch_size, d_sae]
            """

            activation_in, activation_out = batch[sae.cfg.hook_point_in], batch[sae.cfg.hook_point_out]

            if self.wandb_logger is not None:
                self._log(sae, log_info, {"input": activation_in, "output": activation_out})

            if eval_fn is not None and (self.cur_step + 1) % self.cfg.eval_frequency == 0:
                eval_fn(sae)

            self._save_checkpoint(sae)

            self.cur_step += 1
            self.cur_tokens += batch[sae.cfg.hook_point_in].shape[0]
            if self.cur_tokens >= self.cfg.total_training_tokens:
                break
