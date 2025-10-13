import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.distributed.checkpoint as dcp
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.tensor import DTensor
from torch.optim import Adam, Optimizer
from tqdm import tqdm

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.clt import CrossLayerTranscoder
from lm_saes.config import TrainerConfig
from lm_saes.crosscoder import CrossCoder
from lm_saes.molt import MixtureOfLinearTransform
from lm_saes.optim import SparseAdam, get_scheduler
from lm_saes.utils.logging import get_distributed_logger, log_metrics
from lm_saes.utils.misc import is_primary_rank
from lm_saes.utils.tensor_dict import batch_size
from lm_saes.utils.timer import timer
from wandb.sdk.wandb_run import Run

logger = get_distributed_logger("trainer")


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.checkpoint_thresholds: list[int] = []
        self.total_training_steps: int = 0
        self.lr_warm_up_steps: int = 0
        self.lr_cool_down_steps: int = 0
        self.k_warmup_steps: int = 0
        self.k_cold_booting_steps: int = 0
        self.l1_coefficient_warmup_steps: int = 0
        self.cur_step: int = 0
        self.cur_tokens: int = 0
        self.optimizer: Optimizer | None = None
        self.scheduler: lr_scheduler.LRScheduler | None = None
        self.wandb_logger: Run | None = None

    def save_checkpoint(self, sae: AbstractSparseAutoEncoder, checkpoint_path: Path | str) -> None:
        """
        Save a complete checkpoint including model, optimizer, scheduler, and
        trainer state.

        Args:
            sae: The sparse autoencoder model to save
            checkpoint_path: Path where to save the checkpoint (without extension)
        """

        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = Path(checkpoint_path) / "checkpoints" / f"step_{self.cur_step}"

        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        sae.cfg.save_hyperparameters(checkpoint_dir)
        # Save model state
        if sae.device_mesh is None:
            sae.save_checkpoint(Path(checkpoint_dir) / "sae_weights.safetensors")
        else:
            sae.save_checkpoint(Path(checkpoint_dir) / "sae_weights.dcp")

        if is_primary_rank(sae.device_mesh):
            # Prepare trainer state
            trainer_state = {
                "cur_step": self.cur_step,
                "cur_tokens": self.cur_tokens,
                "total_training_steps": self.total_training_steps,
                "lr_warm_up_steps": self.lr_warm_up_steps,
                "lr_cool_down_steps": self.lr_cool_down_steps,
                "k_warmup_steps": self.k_warmup_steps,
                "k_cold_booting_steps": self.k_cold_booting_steps,
                "l1_coefficient_warmup_steps": self.l1_coefficient_warmup_steps,
                "checkpoint_thresholds": self.checkpoint_thresholds,
                "cfg": self.cfg,
            }

            # Save trainer state
            trainer_path = checkpoint_dir / "trainer.pt"
            torch.save(trainer_state, trainer_path)

        # Save optimizer state - handle distributed tensors
        if self.optimizer is not None:
            if sae.device_mesh is None:
                if is_primary_rank(sae.device_mesh):
                    optimizer_path = checkpoint_dir / "optimizer.pt"
                    optimizer_state = self.optimizer.state_dict()
                    torch.save(optimizer_state, optimizer_path)
            else:
                optimizer_path = checkpoint_dir / "optimizer.dcp"
                optimizer_state = self.optimizer.state_dict()
                fs_writer = FileSystemWriter(optimizer_path)
                dcp.save(optimizer_state, storage_writer=fs_writer)

        # Save scheduler state - handle distributed tensors
        if self.scheduler is not None:
            if sae.device_mesh is None:
                if is_primary_rank(sae.device_mesh):
                    scheduler_path = checkpoint_dir / "scheduler.pt"
                    scheduler_state = self.scheduler.state_dict()
                    torch.save(scheduler_state, scheduler_path)
            else:
                scheduler_path = checkpoint_dir / "scheduler.dcp"
                scheduler_state = self.scheduler.state_dict()
                fs_writer = FileSystemWriter(scheduler_path)
                dcp.save(scheduler_state, storage_writer=fs_writer)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    @classmethod
    def from_checkpoint(
        cls,
        sae: AbstractSparseAutoEncoder,
        checkpoint_path: str,
    ) -> "Trainer":
        """
        Load a complete checkpoint including model, optimizer, scheduler, and
        trainer state.

        Args:
            device_mesh: The device mesh to load the model into
            checkpoint_path: Path where the checkpoint was saved (without extension)

        Returns:
            Trainer: A new trainer instance with loaded state
        """
        # Load trainer state first to get the config
        checkpoint_dir = Path(checkpoint_path)
        trainer_path = checkpoint_dir / "trainer.pt"
        if os.path.exists(trainer_path):
            trainer_state = torch.load(trainer_path, map_location="cpu", weights_only=False)
            cfg = trainer_state.get("cfg")
            if cfg is None:
                raise ValueError("Checkpoint does not contain trainer config")

            # Create trainer instance with loaded config
            trainer = cls(cfg)
            trainer.cfg.from_pretrained_path = checkpoint_path

            # Restore trainer state variables
            trainer.cur_step = trainer_state["cur_step"]
            trainer.cur_tokens = trainer_state["cur_tokens"]
            trainer.total_training_steps = trainer_state["total_training_steps"]
            trainer.lr_warm_up_steps = trainer_state["lr_warm_up_steps"]
            trainer.lr_cool_down_steps = trainer_state["lr_cool_down_steps"]
            trainer.k_warmup_steps = trainer_state["k_warmup_steps"]
            trainer.k_cold_booting_steps = trainer_state["k_cold_booting_steps"]
            trainer.l1_coefficient_warmup_steps = trainer_state["l1_coefficient_warmup_steps"]
            trainer.checkpoint_thresholds = trainer_state["checkpoint_thresholds"]

            logger.info(f"Loaded trainer state from step {trainer.cur_step}")
        else:
            raise ValueError(f"Trainer checkpoint not found at {trainer_path}")

        trainer._initialize_optimizer(sae)
        assert trainer.optimizer is not None and trainer.scheduler is not None, (
            "Optimizer and scheduler should be already initialized"
        )

        # Load optimizer state
        if sae.device_mesh is None:
            optimizer_path = checkpoint_dir / "optimizer.pt"
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            trainer.optimizer.load_state_dict(optimizer_state)
            logger.info("Loaded optimizer state")
        else:
            optimizer_path = checkpoint_dir / "optimizer.dcp"
            fs_reader = FileSystemReader(str(optimizer_path))
            optimizer_state = trainer.optimizer.state_dict()
            dcp.load(optimizer_state, storage_reader=fs_reader)
            trainer.optimizer.load_state_dict(optimizer_state)
            logger.info("Loaded optimizer state")
            logger.info(f"trainer.optimizer.state_dict(): {trainer.optimizer.state_dict()}")

        # Load scheduler state
        if sae.device_mesh is None:
            scheduler_path = checkpoint_dir / "scheduler.pt"
            scheduler_state = torch.load(scheduler_path, map_location="cpu")
            trainer.scheduler.load_state_dict(scheduler_state)
            logger.info("Loaded scheduler state")
        else:
            scheduler_path = checkpoint_dir / "scheduler.dcp"
            fs_reader = FileSystemReader(str(scheduler_path))
            scheduler_state = trainer.scheduler.state_dict()
            dcp.load(scheduler_state, storage_reader=fs_reader)
            trainer.scheduler.load_state_dict(scheduler_state)
            logger.info("Loaded scheduler state")
            logger.info(f"trainer.scheduler.state_dict(): {trainer.scheduler.state_dict()}")

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return trainer

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
        self.k_cold_booting_steps = calculate_warmup_steps(self.cfg.k_cold_booting_steps)
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

        def _apply_lr(parameters: dict[str, Any]):
            assert isinstance(self.cfg.lr, float)
            if parameters["name"] == "jumprelu":
                return {**parameters, "lr": self.cfg.jumprelu_lr_factor * self.cfg.lr}
            return parameters

        params = [_apply_lr(parameters) for parameters in sae.get_parameters()]

        def _format_parameters(parameters: dict[str, Any]) -> str:
            param_info = f"{parameters['name']}:"
            for i, param in enumerate(parameters["params"]):
                param_info += f"\n    [{i}] shape={list(param.shape)}, dtype={param.dtype}"
                if param.requires_grad:
                    param_info += ", trainable"
                else:
                    param_info += ", frozen"
            if "lr" in parameters:
                param_info += f"\n    lr={parameters['lr']}"
            return param_info

        param_str = "\n".join([_format_parameters(p) for p in params])
        logger.info(f"\nParameter Groups: \n{param_str}\n")

        optim_cls = {
            "adam": Adam,
            "sparseadam": SparseAdam,
        }[self.cfg.optimizer_class]

        # 构建optimizer参数
        optimizer_kwargs = {
            "params": params,
            "lr": self.cfg.lr,
            "betas": self.cfg.betas,
        }

        # 只有adam optimizer才支持foreach参数
        if self.cfg.optimizer_class == "adam":
            optimizer_kwargs["foreach"] = self.cfg.optimizer_foreach

        optimizer = optim_cls(**optimizer_kwargs)
        # TODO: make this elegant

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
            if self.cur_step < self.k_cold_booting_steps:
                sae.set_current_k(int(self.cfg.initial_k))
            else:
                warmup_progress = (self.cur_step - self.k_cold_booting_steps) / self.k_warmup_steps
                warmup_progress = min(1.0, warmup_progress)

                if self.cfg.k_schedule_type == "exponential":
                    exp_factor = self.cfg.k_exponential_factor
                    decay_factor = (1.0 - math.exp(-exp_factor * warmup_progress)) / (1.0 - math.exp(-exp_factor))
                    current_k = self.cfg.initial_k - (self.cfg.initial_k - sae.cfg.top_k) * decay_factor
                elif self.cfg.k_schedule_type == "linear":
                    current_k = self.cfg.initial_k + (sae.cfg.top_k - self.cfg.initial_k) * warmup_progress
                else:
                    current_k = self.cfg.initial_k + (sae.cfg.top_k - self.cfg.initial_k) * warmup_progress

                sae.set_current_k(
                    max(
                        sae.cfg.top_k,
                        math.ceil(current_k),
                    )
                )

        l1_coefficient = (
            min(1.0, self.cur_step / self.l1_coefficient_warmup_steps) * self.cfg.l1_coefficient
            if self.cfg.l1_coefficient is not None
            else 1.0
        )

        lp_coefficient = self.cfg.lp_coefficient if self.cfg.lp_coefficient is not None else 0.0

        result = sae.compute_loss(
            batch,
            sparsity_loss_type=self.cfg.sparsity_loss_type,
            tanh_stretch_coefficient=self.cfg.tanh_stretch_coefficient,
            p=self.cfg.p,
            use_batch_norm_mse=self.cfg.use_batch_norm_mse,
            return_aux_data=not self.cfg.skip_metrics_calculation,
            l1_coefficient=l1_coefficient,
            lp_coefficient=lp_coefficient,
        )

        loss, (loss_data, aux_data) = result if not self.cfg.skip_metrics_calculation else (result, ({}, {}))
        loss_dict = (
            {
                "loss": loss,
                "batch_size": batch_size(batch),
                "l1_coefficient": l1_coefficient,
                "lp_coefficient": lp_coefficient,
            }
            | loss_data
            | aux_data
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
        elif sae.cfg.sae_type == "clt":
            log_info["reconstructed"] = log_info["reconstructed"].permute(1, 0, 2)
            label = label.permute(1, 0, 2)
        elif sae.cfg.sae_type == "lorsa":
            act_freq_scores = act_freq_scores.mean(0)
        if isinstance(act_freq_scores, DTensor):
            act_freq_scores = act_freq_scores.full_tensor()

        log_info["act_freq_scores"] += act_freq_scores
        log_info["n_frac_active_tokens"] += log_info["batch_size"]
        if (self.cur_step + 1) % self.cfg.feature_sampling_window == 0:
            feature_sparsity = log_info["act_freq_scores"] / log_info["n_frac_active_tokens"]
            if isinstance(sae, CrossLayerTranscoder):
                above_1e_1 = (feature_sparsity > 1e-1).sum(-1)
                above_1e_2 = (feature_sparsity > 1e-2).sum(-1)
                below_1e_5 = (feature_sparsity < 1e-5).sum(-1)
                below_1e_6 = (feature_sparsity < 1e-6).sum(-1)
                below_1e_7 = (feature_sparsity < 1e-7).sum(-1)
                wandb_log_dict = {}

                for l in range(sae.cfg.n_layers):
                    wandb_log_dict[f"sparsity/above_1e-1_layer{l}"] = above_1e_1[l].item()
                    wandb_log_dict[f"sparsity/above_1e-2_layer{l}"] = above_1e_2[l].item()

                for l in range(sae.cfg.n_layers):
                    wandb_log_dict[f"sparsity/below_1e-5_layer{l}"] = below_1e_5[l].item()
                    wandb_log_dict[f"sparsity/below_1e-6_layer{l}"] = below_1e_6[l].item()
                    wandb_log_dict[f"sparsity/below_1e-7_layer{l}"] = below_1e_7[l].item()

                wandb_log_dict["sparsity/above_1e-1"] = above_1e_1.sum().item()
                wandb_log_dict["sparsity/above_1e-2"] = above_1e_2.sum().item()
                wandb_log_dict["sparsity/below_1e-5"] = below_1e_5.sum().item()
                wandb_log_dict["sparsity/below_1e-6"] = below_1e_6.sum().item()
                wandb_log_dict["sparsity/below_1e-7"] = below_1e_7.sum().item()

            else:
                wandb_log_dict = {
                    "sparsity/above_1e-1": (feature_sparsity > 1e-1).sum(-1).item(),
                    "sparsity/above_1e-2": (feature_sparsity > 1e-2).sum(-1).item(),
                    "sparsity/below_1e-5": (feature_sparsity < 1e-5).sum(-1).item(),
                    "sparsity/below_1e-6": (feature_sparsity < 1e-6).sum(-1).item(),
                    "sparsity/below_1e-7": (feature_sparsity < 1e-7).sum(-1).item(),
                }
            if is_primary_rank(sae.device_mesh):
                log_metrics(logger.logger, wandb_log_dict, step=self.cur_step + 1, title="Sparsity Metrics")
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

            l_s = log_info.get("l_s", None)
            if isinstance(l_s, DTensor):
                l_s = l_s.full_tensor()

            l_p = log_info["l_p"]
            if isinstance(l_p, DTensor):
                l_p = l_p.full_tensor()

            if sae.cfg.sae_type == "lorsa":
                label = label.flatten(0, 1)
                log_info["reconstructed"] = log_info["reconstructed"].flatten(0, 1)
            per_token_l2_loss = (
                (log_info["reconstructed"] - label).pow(2).sum(dim=-1)
            )  # [batch_size] for normal sae, [batch_size, n_heads] for crosscoder
            total_variance = (
                (label - label.mean(dim=0)).pow(2).sum(dim=-1)
            )  # [batch_size] for normal sae, [batch_size, n_heads] for crosscoder
            l2_norm_error = per_token_l2_loss.sqrt().mean()
            l2_norm_error_ratio = l2_norm_error / label.norm(p=2, dim=-1).mean()
            explained_variance_legacy = (
                1 - per_token_l2_loss / total_variance
            )  # [batch_size] for normal sae, [batch_size, n_heads] for crosscoder
            if isinstance(explained_variance_legacy, DTensor):
                explained_variance_legacy = explained_variance_legacy.full_tensor()
            l2_loss_mean = per_token_l2_loss.mean(dim=0)
            if isinstance(l2_loss_mean, DTensor):
                l2_loss_mean = l2_loss_mean.full_tensor()
            total_variance_mean = total_variance.mean(dim=0)
            if isinstance(total_variance_mean, DTensor):
                total_variance_mean = total_variance_mean.full_tensor()
            explained_variance = 1 - l2_loss_mean / total_variance_mean
            if sae.cfg.sae_type == "clt":
                per_layer_ev = explained_variance_legacy.mean(0)
                clt_per_layer_ev_dict = {
                    f"metrics/explained_variance_L{l}": per_layer_ev[l].item() for l in range(per_layer_ev.size(0))
                }
                clt_per_layer_l0_dict = {f"metrics/l0_layer{l}": l0[:, l].mean().item() for l in range(l0.size(1))}
                l0 = l0.sum(-1)  # [batch_size]
                ####
                # per_decoder_norm = sae.decoder_norm_per_decoder()
                # if isinstance(per_decoder_norm, DTensor):
                #     per_decoder_norm = per_decoder_norm.full_tensor()  ## TODO: check if this is correct
                # clt_per_decoder_norm_dict = {
                #     f"metrics/decoder_norm_per_decoder_{i}": per_decoder_norm[i].item() for i in range(per_decoder_norm.shape[0])
                # }
            else:
                clt_per_layer_ev_dict = {}
                clt_per_layer_l0_dict = {}
                # clt_per_decoder_norm_dict = {}

            if isinstance(l2_norm_error, DTensor):
                l2_norm_error = l2_norm_error.full_tensor()
            if isinstance(l2_norm_error_ratio, DTensor):
                l2_norm_error_ratio = l2_norm_error_ratio.full_tensor()

            # grad_norm = log_info["grad_norm"]
            # if isinstance(grad_norm, DTensor):
            #     grad_norm = grad_norm.full_tensor()

            wandb_log_dict = {
                # losses
                "losses/mse_loss": l_rec.mean().item(),
                **({"losses/sparsity_loss": l_s.mean().item()} if log_info.get("l_s", None) is not None else {}),  # pyright: ignore[reportOptionalMemberAccess]
                **({"losses/lp_loss": l_p.mean().item()} if log_info.get("l_p", None) is not None else {}),
                "losses/overall_loss": log_info["loss"].item(),
                # variance explained
                **clt_per_layer_ev_dict,
                "metrics/explained_variance": explained_variance.mean().item(),
                "metrics/explained_variance_legacy": explained_variance_legacy.mean().item(),
                # sparsity
                "metrics/l0": l0.mean().item(),
                **clt_per_layer_l0_dict,
                # **clt_per_decoder_norm_dict,
                "metrics/mean_feature_act": mean_feature_act.item(),
                "metrics/l2_norm_error": l2_norm_error.item(),
                "metrics/l2_norm_error_ratio": l2_norm_error_ratio.item(),
                # norm
                # "metrics/gradients_norm": grad_norm.item(),
                "details/current_learning_rate": self.optimizer.param_groups[0]["lr"],
                "details/n_training_tokens": self.cur_tokens,
                "details/l1_coefficient": log_info["l1_coefficient"],
                "details/lp_coefficient": log_info["lp_coefficient"],
            }

            # Add timer information
            timer_data = {f"time/{name}": time_value for name, time_value in timer.get_all_timers().items()}
            timer_avg_data = {f"time_avg/{name}": avg_time for name, avg_time in timer.get_all_average_times().items()}
            wandb_log_dict.update(timer_data)
            wandb_log_dict.update(timer_avg_data)
            wandb_log_dict.update(sae.log_statistics())

            if isinstance(sae, CrossCoder):
                assert explained_variance.ndim == 1 and len(explained_variance) == len(sae.cfg.hook_points)
                for i, k in enumerate(sae.cfg.hook_points):
                    wandb_log_dict.update(
                        {
                            f"crosscoder_metrics/{k}/explained_variance": explained_variance[i].mean().item(),
                            f"crosscoder_metrics/{k}/l0": l0[:, i].mean().item(),
                            f"crosscoder_metrics/{k}/l_rec": l_rec[:, i].mean().item(),
                        }
                    )
            elif isinstance(sae, MixtureOfLinearTransform):
                # MOLT sparsity metrics: track activation per rank group
                feature_acts_for_rank = feature_acts
                if isinstance(feature_acts_for_rank, DTensor):
                    feature_acts_for_rank = feature_acts_for_rank.full_tensor()

                # Calculate l0 per rank group and total rank sum
                feature_idx = 0
                total_rank_sum = 0.0

                for rank in sae.cfg.available_ranks:
                    rank_str = str(rank)
                    if rank_str in sae.U_matrices:
                        # Get global count for this rank group
                        if hasattr(sae, "_global_rank_count_map"):
                            # In distributed case, use global count
                            global_count = sae._global_rank_count_map[rank]
                        else:
                            # Non-distributed case
                            global_count = sae.U_matrices[rank_str].shape[0]

                        if global_count > 0:
                            # Extract features for this rank group
                            end_idx = feature_idx + global_count
                            rank_features = feature_acts_for_rank[..., feature_idx:end_idx]

                            # Count active transforms (l0) for this rank group
                            rank_l0 = (rank_features > 0).float().sum(-1)
                            rank_l0_mean = rank_l0.mean().item()

                            # Record metrics
                            wandb_log_dict[f"molt_metrics/l0_rank{rank}"] = rank_l0_mean
                            wandb_log_dict[f"molt_metrics/l0_rank{rank}_ratio"] = rank_l0_mean / global_count
                            total_rank_sum += rank_l0_mean * rank

                            feature_idx += global_count

                # Record total rank sum
                wandb_log_dict["molt_metrics/total_rank_sum"] = total_rank_sum

            if is_primary_rank(sae.device_mesh):
                log_metrics(logger.logger, wandb_log_dict, step=self.cur_step + 1, title="Training Metrics")

            if timer.enabled:
                logger.info(f"\nTimer Summary:\n{timer.summary()}\n")

            if self.wandb_logger is not None:
                self.wandb_logger.log(wandb_log_dict, step=self.cur_step + 1)

    @timer.time("save_checkpoint")
    def _maybe_save_sae_checkpoint(self, sae: AbstractSparseAutoEncoder):
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
    ) -> bool | None:
        # Reset timer at the start of training
        timer.reset()

        if self.cfg.from_pretrained_path is None:
            logger.info("Initializing trainer and optimizer")
            self._initialize_trainer(sae, activation_stream, wandb_logger)
            self._initialize_optimizer(sae)

        assert self.optimizer is not None and self.scheduler is not None, (
            "Optimizer and scheduler should be already initialized"
        )

        maybe_local_d_sae = sae.cfg.d_sae  # if sae.device_mesh is None else sae.cfg.d_sae // sae.device_mesh.size()
        if sae.cfg.sae_type == "clt":
            act_freq_scores_shape = (
                sae.cfg.n_layers,  # type: ignore
                maybe_local_d_sae,
            )
        else:
            act_freq_scores_shape = (maybe_local_d_sae,)  # type: ignore
        log_info = {
            "act_freq_scores": torch.zeros(act_freq_scores_shape, device=sae.cfg.device, dtype=sae.cfg.dtype),
            "n_frac_active_tokens": torch.tensor([0], device=sae.cfg.device, dtype=torch.int),
        }
        proc_bar = tqdm(total=self.total_training_steps, smoothing=0.001, disable=not is_primary_rank(sae.device_mesh))
        proc_bar.update(self.cur_step)

        try:
            activation_stream = iter(activation_stream)
            batch = next(activation_stream)
            while True:
                with timer.time("training_iteration"):
                    proc_bar.update(1)

                    batch = sae.normalize_activations(batch)

                    sae.train()

                    with torch.autocast(device_type=sae.cfg.device, dtype=self.cfg.amp_dtype):
                        loss_dict = self._training_step(sae, batch)

                    log_info.update(loss_dict)
                    proc_bar.set_description(
                        f"loss: {log_info['loss'].item():.2f}, learning rate: {self.optimizer.param_groups[0]['lr']:.2e}"
                    )

                    if not self.cfg.skip_metrics_calculation:
                        self._log(sae, log_info, batch)

                    with timer.time("refresh_batch"):
                        del batch
                        batch = next(activation_stream)

                    with timer.time("backward"):
                        loss_dict["loss"].backward()

                    with timer.time("clip_grad_norm"):
                        # exclude the grad of the jumprelu threshold
                        loss_dict["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                            [
                                param
                                for name, param in sae.named_parameters()
                                if param.grad is not None and "log_jumprelu_threshold" not in name
                            ],
                            max_norm=self.cfg.clip_grad_norm if self.cfg.clip_grad_norm > 0 else math.inf,
                        )

                    with timer.time("optimizer_step"):
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if eval_fn is not None and (self.cur_step + 1) % self.cfg.eval_frequency == 0:
                        with timer.time("evaluation"):
                            eval_fn(sae)

                    self._maybe_save_sae_checkpoint(sae)
                    with timer.time("scheduler_step"):
                        self.scheduler.step()

                    self.cur_step += 1
                    self.cur_tokens += batch_size(batch)
                    if self.cur_tokens >= self.cfg.total_training_tokens:
                        break
        except StopIteration:
            logger.info("the current stream has ended")
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise e
