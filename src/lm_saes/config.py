import json
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import deprecated

import torch
import torch.distributed as dist

import os

from lm_saes.utils.huggingface import parse_pretrained_name_or_path
from lm_saes.utils.misc import print_once

from transformer_lens.loading_from_pretrained import get_official_model_name


@dataclass
class BaseModelConfig:
    device: str = "cpu"
    seed: int = 42
    dtype: torch.dtype = torch.float32

    def to_dict(self) -> Dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in [base_field.name for base_field in fields(BaseModelConfig)]
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any], **kwargs):
        d = {k: v for k, v in d.items() if k in [field.name for field in fields(cls)]}
        return cls(**d, **kwargs)
    
@dataclass
class RunnerConfig:
    use_ddp: bool = False

    exp_name: str = "test"
    exp_series: Optional[str] = None
    exp_result_dir: str = "results"

    def __post_init__(self):
        # Set rank, world_size, and device if using DDP
        if self.use_ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            if isinstance(self, BaseModelConfig):
                self.device = f"cuda:{self.rank}"

        if not self.use_ddp or self.rank == 0:
            os.makedirs(self.exp_result_dir, exist_ok=True)
            os.makedirs(os.path.join(self.exp_result_dir, self.exp_name), exist_ok=True)


@dataclass
class LanguageModelConfig(BaseModelConfig):
    model_name: str = "gpt2"
    model_from_pretrained_path: Optional[str] = None
    cache_dir: Optional[str] = None
    d_model: int = 768
    local_files_only: bool = False

    def __post_init__(self):
        self.model_name = get_official_model_name(self.model_name)

    @staticmethod
    def from_pretrained_sae(pretrained_name_or_path: str, **kwargs):
        """Load the LanguageModelConfig from a pretrained SAE name or path. Config is read from <pretrained_name_or_path>/lm_config.json.

        Args:
            sae_path (str): The path to the pretrained SAE.
            **kwargs: Additional keyword arguments to pass to the LanguageModelConfig constructor.
        """
        path = parse_pretrained_name_or_path(pretrained_name_or_path)
        with open(os.path.join(path, "lm_config.json"), "r") as f:
            lm_config = json.load(f)
        return LanguageModelConfig.from_dict(lm_config, **kwargs)

    def save_lm_config(self, sae_path: Optional[str] = None):
        if sae_path is None:
            if isinstance(self, RunnerConfig):
                sae_path = os.path.join(self.exp_result_dir, self.exp_name)
            else:
                raise ValueError("sae_path must be specified if not called from a RunnerConfig.")
        assert os.path.exists(sae_path), f"{sae_path} does not exist. Unable to save LanguageModelConfig."
        with open(os.path.join(sae_path, "lm_config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=4)        


@dataclass
class TextDatasetConfig:
    dataset_path: str = "openwebtext"
    cache_dir: Optional[str] = None
    is_dataset_tokenized: bool = False
    is_dataset_on_disk: bool = False
    concat_tokens: bool = True
    context_size: int = 128
    store_batch_size: int = 64


@dataclass
class ActivationStoreConfig(LanguageModelConfig, TextDatasetConfig):
    hook_points: List[str] = field(default_factory=lambda: ["blocks.0.hook_resid_pre"])

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
class WandbConfig:
    log_to_wandb: bool = True
    wandb_project: str = "gpt2-sae-training"
    run_name: Optional[str] = None
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.run_name is None and isinstance(self, RunnerConfig):
            self.run_name = self.exp_name


@dataclass
class SAEConfig(BaseModelConfig):
    """
    Configuration for training or running a sparse autoencoder.
    """
    hook_point_in: str = "blocks.0.hook_resid_pre"
    hook_point_out: str = None # If None, it will be set to hook_point_in

    sae_pretrained_name_or_path: Optional[str] = None
    strict_loading: bool = True

    use_decoder_bias: bool = False
    apply_decoder_bias_to_pre_encoder: bool = True  # set to False when training transcoders
    decoder_bias_init_method: str = "geometric_median"
    expansion_factor: int = 32
    d_model: int = 768
    d_sae: Optional[int] = (
        None  # The dimension of the SAE, i.e. the number of dictionary components (or features). If None, it will be set to d_model * expansion_factor
    )
    norm_activation: str = "token-wise"  # none, token-wise, batch-wise
    decoder_exactly_unit_norm: bool = True
    use_glu_encoder: bool = False

    l1_coefficient: float = 0.00008
    lp: int = 1

    use_ghost_grads: bool = True

    def __post_init__(self):
        if self.hook_point_out is None:
            self.hook_point_out = self.hook_point_in
        if self.d_sae is None:
            self.d_sae = self.d_model * self.expansion_factor

    @staticmethod
    def from_pretrained(pretrained_name_or_path: str, strict_loading: bool = True, **kwargs):
        """Load the SAEConfig from a pretrained SAE name or path. Config is read from <pretrained_name_or_path>/hyperparams.json.
        
        Args:
            sae_path (str): The path to the pretrained SAE.
            **kwargs: Additional keyword arguments to pass to the SAEConfig constructor.
        """
        path = parse_pretrained_name_or_path(pretrained_name_or_path)
        with open(os.path.join(path, "hyperparams.json"), "r") as f:
            sae_config = json.load(f)
        sae_config["sae_pretrained_name_or_path"] = pretrained_name_or_path
        sae_config["strict_loading"] = strict_loading
        return SAEConfig.from_dict(sae_config, **kwargs)
    
    @deprecated("Use from_pretrained and to_dict instead.")
    @staticmethod
    def get_hyperparameters(
        exp_name: str, exp_result_dir: str, ckpt_name: str, strict_loading: bool = True
    ) -> dict[str, Any]:
        with open(os.path.join(exp_result_dir, exp_name, "hyperparams.json"), "r") as f:
            hyperparams = json.load(f)
        hyperparams["sae_pretrained_name_or_path"] = os.path.join(
            exp_result_dir, exp_name, "checkpoints", ckpt_name
        )
        hyperparams["strict_loading"] = strict_loading
        # Remove non-hyperparameters from the dict
        hyperparams = {
            k: v
            for k, v in hyperparams.items()
            if k in SAEConfig.__dataclass_fields__.keys()
        }
        return hyperparams
    
    def save_hyperparameters(self, sae_path: Optional[str] = None, remove_loading_info: bool = True):
        if sae_path is None:
            if isinstance(self, RunnerConfig):
                sae_path = os.path.join(self.exp_result_dir, self.exp_name)
            else:
                raise ValueError("sae_path must be specified if not called from a RunnerConfig.")
        assert os.path.exists(sae_path), f"{sae_path} does not exist. Unable to save hyperparameters."
        d = self.to_dict()
        if remove_loading_info:
            d.pop("sae_pretrained_name_or_path", None)
            d.pop("strict_loading", None)
        with open(os.path.join(sae_path, "hyperparams.json"), "w") as f:
            json.dump(d, f, indent=4)
    
@dataclass
class OpenAIConfig:
    openai_api_key: str
    openai_base_url: str

@dataclass
class AutoInterpConfig(SAEConfig, LanguageModelConfig, OpenAIConfig):
    num_sample: int = 10
    p: float = 0.7
    num_left_token: int = 10
    num_right_token: int = 5


@dataclass
class LanguageModelSAERunnerConfig(SAEConfig, WandbConfig, ActivationStoreConfig, RunnerConfig):
    pass


@dataclass
class LanguageModelSAETrainingConfig(LanguageModelSAERunnerConfig):
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

    finetuning: bool = False

    # Resampling protocol args
    feature_sampling_window: int = 1000
    dead_feature_window: int = 5000  # unless this window is larger feature sampling,

    dead_feature_threshold: float = 1e-6

    # Evaluation
    eval_frequency: int = 1000

    # Misc
    log_frequency: int = 10

    n_checkpoints: int = 10

    def __post_init__(self):
        super().__post_init__()

        if not self.use_ddp or self.rank == 0:
            if os.path.exists(
                os.path.join(self.exp_result_dir, self.exp_name, "checkpoints")
            ):
                raise ValueError(
                    f"Checkpoints for experiment {self.exp_name} already exist. Consider changing the experiment name."
                )
            os.makedirs(os.path.join(self.exp_result_dir, self.exp_name, "checkpoints"))


        self.effective_batch_size = (
            self.train_batch_size * self.world_size
            if self.use_ddp
            else self.train_batch_size
        )
        print_once(f"Effective batch size: {self.effective_batch_size}")

        total_training_steps = self.total_training_tokens // self.effective_batch_size
        print_once(f"Total training steps: {total_training_steps}")

        if self.use_ghost_grads:
            print_once("Using Ghost Grads.")


@dataclass
class LanguageModelSAEPruningConfig(LanguageModelSAERunnerConfig):
    """
    Configuration for pruning a sparse autoencoder on a language model.
    """

    total_training_tokens: int = 10_000_000
    train_batch_size: int = 4096

    dead_feature_threshold: float = 1e-6
    dead_feature_max_act_threshold: float = 1.0
    decoder_norm_threshold: float = 0.99


@dataclass
class ActivationGenerationConfig(LanguageModelConfig, TextDatasetConfig):
    hook_points: list[str] = field(default_factory=list)

    activation_save_path: Optional[str] = (
        None  # Defaults to "activations/{dataset}/{model}_{context_size}"
    )

    total_generating_tokens: int = 300_000_000
    chunk_size: int = int(0.5 * 2**30)  # 0.5 GB

    def __post_init__(self):
        super().__post_init__()

        if self.activation_save_path is None:
            self.activation_save_path = f"activations/{self.dataset_path.split('/')[-1]}/{self.model_name.replace('/', '_')}_{self.context_size}"
        os.makedirs(self.activation_save_path, exist_ok=True)

@dataclass
class MongoConfig:
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "mechinterp"

@dataclass
class LanguageModelSAEAnalysisConfig(SAEConfig, ActivationStoreConfig, MongoConfig, RunnerConfig):
    """
    Configuration for analyzing a sparse autoencoder on a language model.
    """

    total_analyzing_tokens: int = 300_000_000
    enable_sampling: bool = (
        False  # If True, we will sample the activations based on weights. Otherwise, top n_samples activations will be used.
    )
    sample_weight_exponent: float = 2.0
    subsample: Dict[str, Dict[str, Any]] = field(default_factory=lambda: { "top_activations": {"proportion": 1.0, "n_samples": 10} })

    n_sae_chunks: int = 1  # Number of chunks to split the SAE into for analysis. For large models and SAEs, this can be useful to avoid memory issues.

    def __post_init__(self):
        super().__post_init__()

        assert self.d_sae % self.n_sae_chunks == 0, f"d_sae ({self.d_sae}) must be divisible by n_sae_chunks ({self.n_sae_chunks})"


@dataclass
class FeaturesDecoderConfig(SAEConfig, LanguageModelConfig, MongoConfig, RunnerConfig):
    top: int = 10
