from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, cast

import torch
import torch.distributed as dist

import wandb

import os 

from core.utils import print_once

from transformer_lens.loading_from_pretrained import get_official_model_name

@dataclass
class RunnerConfig:
    # Misc
    use_ddp: bool = False
    device: str = "cpu"
    seed: int = 42
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        # Set rank, world_size, and device if using DDP
        if self.use_ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = f"cuda:{self.rank}"

@dataclass
class LanguageModelConfig(RunnerConfig):
    model_name: str = "gpt2"
    cache_dir: Optional[str] = None
    d_model: int = 768
    local_files_only: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.model_name = get_official_model_name(self.model_name)

@dataclass
class TextDatasetConfig(RunnerConfig):
    dataset_path: str = "openwebtext"
    cache_dir: Optional[str] = None
    is_dataset_tokenized: bool = False
    is_dataset_on_disk: bool = False
    context_size: int = 128
    store_batch_size: int = 64

@dataclass
class ActivationStoreConfig(LanguageModelConfig, TextDatasetConfig):
    hook_point: str = "blocks.0.hook_mlp_out"
    use_cached_activations: bool = False
    cached_activations_path: Optional[str] = (
        None  # Defaults to "activations/{self.dataset_path.split('/')[-1]}/{self.model_name.replace('/', '_')}_{self.context_size}"
    )

    # Activation Store Parameters
    n_tokens_in_buffer: int = 500_000

    def __post_init__(self):
        super().__post_init__()
        # Autofill cached_activations_path unless the user overrode it
        if self.cached_activations_path is None:
            self.cached_activations_path = f"activations/{self.dataset_path.split('/')[-1]}/{self.model_name.replace('/', '_')}_{self.context_size}"

@dataclass
class WandbConfig(RunnerConfig):
    log_to_wandb: bool = True
    wandb_project: str = "gpt2-sae-training"
    run_name: Optional[str] = None
    wandb_entity: Optional[str] = None

@dataclass
class SAEConfig(RunnerConfig):
    """
    Configuration for training or running a sparse autoencoder.
    """
    from_pretrained_path: Optional[str] = None

    use_decoder_bias: bool = True
    decoder_bias_init_method: str = "geometric_median"
    geometric_median_max_iter: Optional[int] = 1000 # The maximum number of iterations for the geometric median algorithm. Required if decoder_bias_init_method is geometric_median
    expansion_factor: int = 32
    d_model: int = 768
    d_sae: Optional[int] = None # The dimension of the SAE, i.e. the number of dictionary components (or features). If None, it will be set to d_model * expansion_factor
    norm_activation: str = "token-wise" # none, token-wise, batch-wise
    decoder_exactly_unit_norm: bool = True

    l1_coefficient: float = 0.00008
    lp: int = 1

    use_ghost_grads: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.d_sae is None:
            self.d_sae = self.d_model * self.expansion_factor

        print_once(f"Sparse Autoencoder dimension: {self.d_sae}")

@dataclass
class LanguageModelSAEConfig(SAEConfig, WandbConfig, ActivationStoreConfig):
    pass

@dataclass
class LanguageModelSAETrainingConfig(LanguageModelSAEConfig):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    # Training Parameters
    total_training_tokens: int = 300_000_000
    lr: float = 0.0004
    betas: Tuple[float, float] = (0.9, 0.999)
    lr_scheduler_name: str = (
        "constantwithwarmup"  # constant, constantwithwarmup, linearwarmupdecay, cosineannealing, cosineannealingwarmup, exponentialwarmup
    )
    lr_end: Optional[float] = 1 / 32
    lr_warm_up_steps: int = 5000
    lr_cool_down_steps: int = 10000
    train_batch_size: int = 4096

    # Resampling protocol args
    feature_sampling_window: int = 1000
    dead_feature_window: int = 5000  # unless this window is larger feature sampling,

    dead_feature_threshold: float = 1e-6    

    # Evaluation
    eval_frequency: int = 1000

    # Misc
    log_frequency: int = 10

    n_checkpoints: int = 10
    checkpoint_path: str = "checkpoints"

    def __post_init__(self):
        super().__post_init__()

        if self.run_name is None:
            self.run_name = f"{self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"

        print_once(
            f"Run name: {self.run_name}"
        )

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

        self.effective_batch_size = self.train_batch_size * self.world_size if self.use_ddp else self.train_batch_size
        print_once(f"Effective batch size: {self.effective_batch_size}")

        total_training_steps = self.total_training_tokens // self.effective_batch_size
        print_once(f"Total training steps: {total_training_steps}")

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

@dataclass
class ActivationGenerationConfig(LanguageModelConfig, TextDatasetConfig):
    hook_points: list[str] = field(default_factory=list)

    activation_save_path: Optional[str] = None # Defaults to "activations/{dataset}/{model}_{context_size}"

    total_generating_tokens: int = 300_000_000
    chunk_size: int = int(0.5 * 2 ** 30) # 0.5 GB

    def __post_init__(self):
        super().__post_init__()

        if self.activation_save_path is None:
            self.activation_save_path = f"activations/{self.dataset_path.split('/')[-1]}/{self.model_name.replace('/', '_')}_{self.context_size}"
        os.makedirs(self.activation_save_path, exist_ok=True)

@dataclass
class LanguageModelSAEAnalysisConfig(SAEConfig, ActivationStoreConfig):
    """
    Configuration for analyzing a sparse autoencoder on a language model.
    """

    total_analyzing_tokens: int = 300_000_000
    analysis_batch_size: int = 4096
    n_samples: int = 1000
    analysis_save_path: str = "analysis/test.pt"

    def __post_init__(self):
        super().__post_init__()

        self.effective_batch_size = self.analysis_batch_size * self.world_size if self.use_ddp else self.analysis_batch_size
        print_once(f"Effective batch size: {self.effective_batch_size}")

        total_training_steps = self.total_analyzing_tokens // self.effective_batch_size
        print_once(f"Total steps: {total_training_steps}")