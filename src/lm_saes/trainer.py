import math
import os
from typing import Callable, Iterable

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim import Adam, Optimizer
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.config import TrainerConfig
from lm_saes.crosscoder import CrossCoder
from lm_saes.optim import get_scheduler
from lm_saes.utils.logging import get_distributed_logger, log_metrics
from lm_saes.utils.misc import is_primary_rank
from lm_saes.utils.tensor_dict import batch_size
from lm_saes.utils.timer import timer

logger = get_distributed_logger("trainer")


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

    @timer.time("initialize_trainer")
    def _initialize_trainer(
        self,
        sae: AbstractSparseAutoEncoder,
        activation_stream: Iterable[dict[str, Tensor]],
        wandb_logger: Run | None = None,
    ):
        bs = batch_size(next(iter(activation_stream)))
        self.total_training_steps = self.cfg.total_training_tokens // bs

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
                    math.ceil(2 ** (i / self.cfg.n_checkpoints * math.log2(self.total_training_steps))) * bs
                    for i in range(1, self.cfg.n_checkpoints)
                ]
        self.wandb_logger = wandb_logger

    @timer.time("initialize_optimizer")
    def _initialize_optimizer(self, sae: AbstractSparseAutoEncoder):
        assert isinstance(self.cfg.lr, float)
        optimizer = Adam(sae.get_parameters(), lr=self.cfg.lr, betas=self.cfg.betas)
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

    @timer.time("training_step")
    def _training_step(
        self,
        sae: AbstractSparseAutoEncoder,
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        if "topk" in sae.cfg.act_fn and self.k_warmup_steps > 0:
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

        l1_coefficient = (
            min(1.0, self.cur_step / self.l1_coefficient_warmup_steps) * self.cfg.l1_coefficient
            if self.cfg.l1_coefficient is not None
            else 1.0
        )

        loss, (loss_data, aux_data) = sae.compute_loss(
            batch,
            sparsity_loss_type=self.cfg.sparsity_loss_type,
            tanh_stretch_coefficient=self.cfg.tanh_stretch_coefficient,
            p=self.cfg.p,
            use_batch_norm_mse=self.cfg.use_batch_norm_mse,
            return_aux_data=True,
            l1_coefficient=l1_coefficient,
        )
        loss_dict = (
            {"loss": loss, "batch_size": batch_size(batch), "l1_coefficient": l1_coefficient} | loss_data | aux_data
        )
        return loss_dict

    @torch.no_grad()
    @timer.time("log")
    def _log(self, sae: AbstractSparseAutoEncoder, log_info: dict, batch: dict[str, Tensor]):
        # TODO: add full distributed support
        assert self.optimizer is not None, "Optimizer must be initialized"
        label = sae.prepare_label(batch)
        act_freq_scores = (log_info["feature_acts"] > 0).float().sum(0)
        if sae.cfg.sae_type == "crosscoder":
            if not isinstance(act_freq_scores, DTensor):
                act_freq_scores = act_freq_scores.amax(dim=0)
            else:
                # Operator aten.amax.default does not have a sharding strategy registered.
                act_freq_scores = act_freq_scores.full_tensor().amax(dim=0)
        if isinstance(act_freq_scores, DTensor):
            act_freq_scores = act_freq_scores.full_tensor()

        log_info["n_forward_passes_since_fired"] += 1
        log_info["n_forward_passes_since_fired"][act_freq_scores > 0] = 0
        log_info["act_freq_scores"] += act_freq_scores
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
            if self.wandb_logger is not None:
                self.wandb_logger.log(wandb_log_dict, step=self.cur_step + 1)
            log_info["act_freq_scores"] = torch.zeros_like(log_info["act_freq_scores"])
            log_info["n_frac_active_tokens"] = torch.zeros_like(log_info["n_frac_active_tokens"])

        if (self.cur_step + 1) % self.cfg.log_frequency == 0:
            feature_acts = log_info["feature_acts"]
            act_feature_counts = feature_acts.gt(0).float().sum()
            mean_feature_act = feature_acts.sum() / act_feature_counts
            if isinstance(mean_feature_act, DTensor):
                mean_feature_act = mean_feature_act.full_tensor()

            l0 = (feature_acts > 0).float().sum(-1)  # [batch_size] for normal sae, [batch_size, n_heads] for crosscoder
            if isinstance(l0, DTensor):
                l0 = l0.full_tensor()

            l_rec = log_info["l_rec"]
            if isinstance(l_rec, DTensor):
                l_rec = l_rec.full_tensor()

            l_s = log_info["l_s"]
            if isinstance(l_s, DTensor):
                l_s = l_s.full_tensor()

            per_token_l2_loss = (
                (log_info["reconstructed"] - label).pow(2).sum(dim=-1)
            )  # [batch_size] for normal sae, [batch_size, n_heads] for crosscoder
            total_variance = (
                (label - label.mean(0)).pow(2).sum(dim=-1)
            )  # [batch_size] for normal sae, [batch_size, n_heads] for crosscoder
            l2_norm_error = per_token_l2_loss.sqrt().mean()
            l2_norm_error_ratio = l2_norm_error / label.norm(p=2, dim=-1).mean()
            explained_variance = (
                1 - per_token_l2_loss / total_variance
            )  # [batch_size] for normal sae, [batch_size, n_heads] for crosscoder
            if isinstance(explained_variance, DTensor):
                explained_variance = explained_variance.full_tensor()
            if isinstance(l2_norm_error, DTensor):
                l2_norm_error = l2_norm_error.full_tensor()
            if isinstance(l2_norm_error_ratio, DTensor):
                l2_norm_error_ratio = l2_norm_error_ratio.full_tensor()

            grad_norm = log_info["grad_norm"]
            if isinstance(grad_norm, DTensor):
                grad_norm = grad_norm.full_tensor()

            wandb_log_dict = {
                # losses
                "losses/mse_loss": l_rec.mean().item(),
                **({"losses/sparsity_loss": l_s.mean().item()} if log_info.get("l_s", None) is not None else {}),
                "losses/overall_loss": log_info["loss"].item(),
                # variance explained
                "metrics/explained_variance": explained_variance.mean().item(),
                "metrics/explained_variance_std": explained_variance.std().item(),
                # sparsity
                "metrics/l0": l0.mean().item(),
                "metrics/mean_feature_act": mean_feature_act.item(),
                "metrics/l2_norm_error": l2_norm_error.item(),
                "metrics/l2_norm_error_ratio": l2_norm_error_ratio.item(),
                # norm
                "metrics/gradients_norm": grad_norm.item(),
                # sparsity
                "sparsity/mean_passes_since_fired": log_info["n_forward_passes_since_fired"].mean().item(),
                "details/current_learning_rate": self.optimizer.param_groups[0]["lr"],
                "details/n_training_tokens": self.cur_tokens,
                "details/l1_coefficient": log_info["l1_coefficient"],
            }
            # Add timer information
            timer_data = {f"time/{name}": time_value for name, time_value in timer.get_all_timers().items()}
            timer_avg_data = {f"time_avg/{name}": avg_time for name, avg_time in timer.get_all_average_times().items()}
            wandb_log_dict.update(timer_data)
            wandb_log_dict.update(timer_avg_data)

            wandb_log_dict.update(sae.log_statistics())

            if isinstance(sae, CrossCoder):
                assert explained_variance.ndim == 2 and explained_variance.shape[1] == len(sae.cfg.hook_points)
                for i, k in enumerate(sae.cfg.hook_points):
                    wandb_log_dict.update(
                        {
                            f"crosscoder_metrics/{k}/explained_variance": explained_variance[:, i].mean().item(),
                            f"crosscoder_metrics/{k}/explained_variance_std": explained_variance[:, i].std().item(),
                            f"crosscoder_metrics/{k}/l0": l0[:, i].mean().item(),
                            f"crosscoder_metrics/{k}/l_rec": l_rec[:, i].mean().item(),
                        }
                    )

            if is_primary_rank(sae.device_mesh):
                log_metrics(logger.logger, wandb_log_dict, step=self.cur_step + 1, title="Training Metrics")

            if timer.enabled:
                logger.info(f"\nTimer Summary:\n{timer.summary()}\n")

            if self.wandb_logger is not None:
                self.wandb_logger.log(wandb_log_dict, step=self.cur_step + 1)

    @timer.time("save_checkpoint")
    def _save_checkpoint(self, sae: AbstractSparseAutoEncoder):
        if len(self.checkpoint_thresholds) > 0 and self.cur_tokens >= self.checkpoint_thresholds[0]:
            suffix = "safetensors" if sae.device_mesh is None else "dcp"
            path = os.path.join(
                self.cfg.exp_result_path,
                "checkpoints",
                f"{self.cur_step}.{suffix}",
            )
            sae.save_checkpoint(path)
            self.checkpoint_thresholds.pop(0)

    def fit(
        self,
        sae: AbstractSparseAutoEncoder,
        activation_stream: Iterable[dict[str, Tensor]],
        eval_fn: Callable[[AbstractSparseAutoEncoder], None] | None = None,
        wandb_logger: Run | None = None,
    ):
        # Reset timer at the start of training
        timer.reset()

        self._initialize_trainer(sae, activation_stream, wandb_logger)
        self._initialize_optimizer(sae)
        assert self.optimizer is not None
        assert self.scheduler is not None
        log_info = {
            "act_freq_scores": torch.zeros(sae.cfg.d_sae, device=sae.cfg.device, dtype=sae.cfg.dtype),
            "n_forward_passes_since_fired": torch.zeros(sae.cfg.d_sae, device=sae.cfg.device, dtype=sae.cfg.dtype),
            "n_frac_active_tokens": torch.tensor([0], device=sae.cfg.device, dtype=torch.int),
        }
        proc_bar = tqdm(total=self.total_training_steps, smoothing=0.001, disable=not is_primary_rank(sae.device_mesh))
        for batch in activation_stream:
            with timer.time("training_iteration"):
                proc_bar.update(1)

                batch = sae.normalize_activations(batch)

                sae.train()

                self.optimizer.zero_grad()

                loss_dict = self._training_step(sae, batch)

                with timer.time("backward"):
                    loss_dict["loss"].backward()

                with timer.time("clip_grad_norm"):
                    loss_dict["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                        sae.parameters(),
                        max_norm=self.cfg.clip_grad_norm if self.cfg.clip_grad_norm > 0 else math.inf,
                    )

                with timer.time("optimizer_step"):
                    self.optimizer.step()

                log_info.update(loss_dict)
                proc_bar.set_description(f"loss: {log_info['loss'].item()}")

                self._log(sae, log_info, batch)

                if eval_fn is not None and (self.cur_step + 1) % self.cfg.eval_frequency == 0:
                    with timer.time("evaluation"):
                        eval_fn(sae)

                self._save_checkpoint(sae)
                with timer.time("scheduler_step"):
                    self.scheduler.step()

                self.cur_step += 1
                self.cur_tokens += batch_size(batch)
                if self.cur_tokens >= self.cfg.total_training_tokens:
                    break
