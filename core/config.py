from abc import ABC
from dataclasses import dataclass
from typing import Any, Optional, cast

import torch
import torch.distributed as dist

import wandb

import os 

from core.utils import print_once

@dataclass
class RunnerConfig(ABC):
    """
    The config that's shared across all runners.
    """

    # Data Generating Function (Model + Training Distibuion)
    model_name: str = "gpt2"
    hook_point: str = "blocks.{layer}.hook_mlp_out"
    dataset_path: str = "openwebtext"
    is_dataset_tokenized: bool = False
    is_dataset_on_disk: bool = False
    context_size: int = 128
    use_cached_activations: bool = False
    cached_activations_path: Optional[str] = (
        None  # Defaults to "activations/{dataset}/{model}/{full_hook_name}_{hook_point_head_index}"
    )

    # SAE Parameters
    d_model: int = 768

    # Activation Store Parameters
    n_tokens_in_buffer: int = 500_000
    total_training_tokens: int = 300_000_000
    store_batch_size: int = 64

    # Misc
    use_ddp: bool = False
    device: str = "cpu"
    seed: int = 42
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        # Autofill cached_activations_path unless the user overrode it
        if self.cached_activations_path is None:
            self.cached_activations_path = f"activations/{self.dataset_path.replace('/', '_')}/{self.model_name.replace('/', '_')}/{self.hook_point}"

        # Set rank, world_size, and device if using DDP
        if self.use_ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = f"cuda:{self.rank}"

        


@dataclass
class LanguageModelSAERunnerConfig(RunnerConfig):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    # SAE Parameters
    decoder_bias_init_method: str = "geometric_median" # The method to initialize the decoder bias. Options: geometric_median, mean, zeros
    geometric_median_max_iter: Optional[int] = 1000 # The maximum number of iterations for the geometric median algorithm. Required if decoder_bias_init_method is geometric_median
    expansion_factor: int = 32
    from_pretrained_path: Optional[str] = None
    d_sae: Optional[int] = None # The dimension of the SAE, i.e. the number of dictionary components (or features). If None, it will be set to d_model * expansion_factor
    norm_activation: bool = True

    # Training Parameters
    l1_coefficient: float = 0.00008
    lp_norm: float = 1
    lr: float = 0.0004
    lr_scheduler_name: str = (
        "constantwithwarmup"  # constant, constantwithwarmup, linearwarmupdecay, cosineannealing, cosineannealingwarmup
    )
    lr_warm_up_steps: int = 5000
    train_batch_size: int = 4096

    # Resampling protocol args
    use_ghost_grads: bool = True  # want to change this to true on some timeline.
    feature_sampling_window: int = 1000
    dead_feature_window: int = 5000  # unless this window is larger feature sampling,

    dead_feature_threshold: float = 1e-6

    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "gpt2-sae-training"
    run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_log_frequency: int = 10

    # Evaluation
    eval_frequency: int = 1000

    # Misc
    n_checkpoints: int = 10
    checkpoint_path: str = "checkpoints"

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.expansion_factor, list):
            self.d_sae = self.d_model * self.expansion_factor

        if self.run_name is None:
            self.run_name = f"{self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"

        if self.decoder_bias_init_method not in ["geometric_median", "mean", "zeros"]:
            raise ValueError(
                f"b_dec_init_method must be geometric_median, mean, or zeros. Got {self.decoder_bias_init_method}"
            )
        if self.decoder_bias_init_method == "zeros":
            print_once(
                "Warning: We are initializing b_dec to zeros. This is probably not what you want."
            )

        unique_id = cast(
            Any, wandb
        ).util.generate_id()  # not sure why this type is erroring

        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.checkpoint_path = f"{self.checkpoint_path}/{unique_id}"
        if not self.use_ddp or self.rank == 0:
            os.makedirs(self.checkpoint_path)

        print_once(
            f"Run name: {self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"
        )

        self.effective_batch_size = self.train_batch_size * self.world_size if self.use_ddp else self.train_batch_size
        print_once(f"Effective batch size: {self.effective_batch_size}")

        total_training_steps = self.total_training_tokens // self.effective_batch_size
        print_once(f"Total training steps: {total_training_steps}")

        total_wandb_updates = total_training_steps // self.wandb_log_frequency
        print_once(f"Total wandb updates: {total_wandb_updates}")

        # how many times will we sample dead neurons?
        # assert self.dead_feature_window <= self.feature_sampling_window, "dead_feature_window must be smaller than feature_sampling_window"
        n_feature_window_samples = total_training_steps // self.feature_sampling_window
        print_once(
            f"n_tokens_per_feature_sampling_window (millions): {(self.feature_sampling_window * self.context_size * self.effective_batch_size) / 10 **6}"
        )
        print_once(
            f"n_tokens_per_dead_feature_window (millions): {(self.dead_feature_window * self.context_size * self.effective_batch_size) / 10 **6}"
        )

        if self.use_ghost_grads:
            print_once("Using Ghost Grads.")

        print_once(
            f"We will reset the sparsity calculation {n_feature_window_samples} times."
        )
        print_once(
            f"Number tokens in sparsity calculation window: {self.feature_sampling_window * self.effective_batch_size:.2e}"
        )