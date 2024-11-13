import json
import math
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from transformer_lens.loading_from_pretrained import get_official_model_name
from typing_extensions import deprecated

from lm_saes.utils.config import FlattenableModel
from lm_saes.utils.huggingface import parse_pretrained_name_or_path
from lm_saes.utils.misc import (
    convert_str_to_torch_dtype,
    convert_torch_dtype_to_str,
    is_master,
    print_once,
)


@dataclass(kw_only=True)
class BaseConfig(FlattenableModel):
    def __post_init__(self):
        pass


@dataclass(kw_only=True)
class BaseModelConfig(BaseConfig):
    device: str = "cpu"
    seed: int = 42
    dtype: torch.dtype = torch.float32

    def to_dict(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any], **kwargs):
        d = {k: v for k, v in d.items() if k in [field.name for field in fields(cls)]}
        return cls(**d, **kwargs)

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.dtype, str):
            self.dtype = convert_str_to_torch_dtype(self.dtype)
        if dist.is_initialized() and self.device == "cuda":
            self.device = f"cuda:{dist.get_rank()}"


@dataclass(kw_only=True)
class RunnerConfig(BaseConfig):
    exp_name: str = "test"
    exp_series: Optional[str] = None
    exp_result_path: str = "results"

    def __post_init__(self):
        super().__post_init__()
        if is_master():
            os.makedirs(self.exp_result_path, exist_ok=True)


@dataclass(kw_only=True)
class LanguageModelConfig(BaseModelConfig):
    model_name: str = "gpt2"
    model_from_pretrained_path: Optional[str] = None
    use_flash_attn: bool = False
    cache_dir: Optional[str] = None
    d_model: int = 768
    local_files_only: bool = False

    def __post_init__(self):
        super().__post_init__()
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

    def save_lm_config(self, sae_path: str):
        assert os.path.exists(sae_path), f"{sae_path} does not exist. Unable to save LanguageModelConfig."

        d = self.to_dict()
        for k, v in d.items():
            if isinstance(v, torch.dtype):
                d[k] = convert_torch_dtype_to_str(v)

        with open(os.path.join(sae_path, "lm_config.json"), "w") as f:
            json.dump(d, f, indent=4)


@dataclass(kw_only=True)
class TextDatasetConfig(RunnerConfig):
    dataset_path: List[str] = "openwebtext"  # type: ignore
    cache_dir: Optional[str] = None
    is_dataset_tokenized: bool = False
    is_dataset_on_disk: bool = False
    concat_tokens: List[bool] = False  # type: ignore
    context_size: int = 128
    store_batch_size: int = 64
    sample_probs: List[float] = field(default_factory=lambda: [1.0])
    prepend_bos: List[bool] = field(default_factory=lambda: [True])

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.dataset_path, str):
            self.dataset_path = [self.dataset_path]

        if isinstance(self.concat_tokens, bool):
            self.concat_tokens = [self.concat_tokens]

        if isinstance(self.prepend_bos, bool):
            self.prepend_bos = [self.prepend_bos]

        if False in self.prepend_bos:
            print(
                "Warning: prepend_bos is set to False for some datasets. This setting might not be suitable for most modern models."
            )

        self.sample_probs = [p / sum(self.sample_probs) for p in self.sample_probs]

        assert len(self.sample_probs) == len(
            self.dataset_path
        ), "Number of sample_probs must match number of dataset paths"
        assert len(self.concat_tokens) == len(
            self.dataset_path
        ), "Number of concat_tokens must match number of dataset paths"
        assert len(self.prepend_bos) == len(
            self.dataset_path
        ), "Number of prepend_bos must match number of dataset paths"


@dataclass(kw_only=True)
class ActivationStoreConfig(BaseModelConfig, RunnerConfig):
    lm: LanguageModelConfig
    dataset: TextDatasetConfig
    hook_points: List[str] = field(default_factory=lambda: ["blocks.0.hook_resid_pre"])
    """ Hook points to store activations from, i.e. the layer output of which is used for training/evaluating the dictionary. Will run until the last hook point in the list, so make sure to order them correctly. """

    use_cached_activations: bool = False
    cached_activations_path: List[str] = None  # type: ignore

    n_tokens_in_buffer: int = 500_000
    tp_size: int = 1
    ddp_size: int = 1

    def __post_init__(self):
        super().__post_init__()
        # Autofill cached_activations_path unless the user overrode it
        if self.cached_activations_path is None:
            self.cached_activations_path = [
                f"activations/{path.split('/')[-1]}/{self.lm.model_name.replace('/', '_')}_{self.dataset.context_size}"
                for path in self.dataset.dataset_path
            ]


@dataclass(kw_only=True)
class WandbConfig(BaseConfig):
    log_to_wandb: bool = True
    wandb_project: str = "gpt2-sae-training"
    exp_name: Optional[str] = None
    wandb_entity: Optional[str] = None


@dataclass(kw_only=True)
class SAEConfig(BaseModelConfig):
    """
    Configuration for training or running a sparse autoencoder.
    """

    hook_point_in: str = "blocks.0.hook_resid_pre"
    """ The hook point to use as input to the SAE. """
    hook_point_out: str = None  # type: ignore
    """ The hook point to use as label of the SAE. If None, it will be set to hook_point_in. """

    sae_pretrained_name_or_path: Optional[str] = None
    strict_loading: bool = True

    use_decoder_bias: bool = True
    apply_decoder_bias_to_pre_encoder: bool = True  # set to False when training transcoders
    expansion_factor: int = 128
    d_model: int = 768
    d_sae: int = None  # type: ignore
    bias_init_method: str = "all_zero"
    act_fn: str = "relu"
    jump_relu_threshold: float = 0.0

    """ The dimension of the SAE, i.e. the number of dictionary components (or features). If None, it will be set to d_model * expansion_factor """
    norm_activation: str = "token-wise"  # none, token-wise, batch-wise, dataset-wise
    dataset_average_activation_norm: Dict[str, float] | None = None
    decoder_exactly_fixed_norm: bool = False
    sparsity_include_decoder_norm: bool = True  # set to True: sparsity loss = sum(act * corresponding_decoder_norm), otherwise loss = sum(act). Incompatible with decoder_exactly_fixed_norm
    use_glu_encoder: bool = False
    init_decoder_norm: float | None = None  # type: ignore
    init_encoder_norm: float | None = None  # type: ignore
    init_encoder_with_decoder_transpose: bool = True

    lp: int = 1
    l1_coefficient: float = 0.00008
    l1_coefficient_warmup_steps: int | float = 0.1
    top_k: int = 50
    k_warmup_steps: int | float = 0.1

    use_batch_norm_mse: bool = True  # will set to False in the future

    use_ghost_grads: bool = False

    tp_size: int = 1
    ddp_size: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.hook_point_out is None:
            self.hook_point_out = self.hook_point_in
        if self.d_sae is None:
            self.d_sae = self.d_model * self.expansion_factor
        if self.norm_activation == "dataset-wise" and self.dataset_average_activation_norm is None:
            print(
                'dataset_average_activation_norm is None and norm_activation is "dataset-wise". Will be computed automatically from the dataset.'
            )
        if self.sparsity_include_decoder_norm and self.decoder_exactly_fixed_norm:
            raise ValueError("sparsity_include_decoder_norm and decoder_exactly_fixed_norm are incompatible.")
        if self.sparsity_include_decoder_norm and self.use_ghost_grads:
            raise ValueError("sparsity_include_decoder_norm and use_ghost_grads are incompatible.")
        if self.init_encoder_with_decoder_transpose and isinstance(self.init_encoder_norm, float):
            raise ValueError("init_encoder_with_decoder_transpose and init_encoder_norm with float are incompatible.")

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
    def get_hyperparameters(exp_result_path: str, ckpt_name: str, strict_loading: bool = True) -> dict[str, Any]:
        with open(os.path.join(exp_result_path, "hyperparams.json"), "r") as f:
            hyperparams = json.load(f)
        hyperparams["sae_pretrained_name_or_path"] = os.path.join(exp_result_path, "checkpoints", ckpt_name)
        hyperparams["strict_loading"] = strict_loading
        # Remove non-hyperparameters from the dict
        hyperparams = {k: v for k, v in hyperparams.items() if k in SAEConfig.__dataclass_fields__.keys()}
        return hyperparams

    def save_hyperparameters(self, sae_path: str, remove_loading_info: bool = True):
        assert os.path.exists(sae_path), f"{sae_path} does not exist. Unable to save hyperparameters."
        d = self.to_dict()
        if remove_loading_info:
            d.pop("sae_pretrained_name_or_path", None)
            d.pop("strict_loading", None)

        for k, v in d.items():
            if isinstance(v, torch.dtype):
                d[k] = convert_torch_dtype_to_str(v)

        with open(os.path.join(sae_path, "hyperparams.json"), "w") as f:
            json.dump(d, f, indent=4)


@dataclass(kw_only=True)
class OpenAIConfig(BaseConfig):
    openai_api_key: str
    openai_base_url: str


@dataclass(kw_only=True)
class AutoInterpConfig(BaseConfig):
    sae: SAEConfig
    lm: LanguageModelConfig
    openai: OpenAIConfig
    num_sample: int = 10
    p: float = 0.7
    num_left_token: int = 20
    num_right_token: int = 5


@dataclass(kw_only=True)
class LanguageModelSAERunnerConfig(RunnerConfig):
    sae: SAEConfig
    lm: LanguageModelConfig
    act_store: ActivationStoreConfig
    wandb: WandbConfig


@dataclass(kw_only=True)
class LanguageModelSAETrainingConfig(LanguageModelSAERunnerConfig):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    # Training Parameters
    total_training_tokens: int = 300_000_000
    lr: float = 0.0004
    betas: Tuple[float, float] = (0.9, 0.999)
    lr_scheduler_name: str = "constantwithwarmup"  # constant, constantwithwarmup, linearwarmupdecay, cosineannealing, cosineannealingwarmup, exponentialwarmup
    lr_end_ratio: float = 1 / 32
    lr_warm_up_steps: int | float = 0.1
    lr_cool_down_steps: int | float = 0.1
    train_batch_size: int = 4096
    clip_grad_norm: float = 0.0
    remove_gradient_parallel_to_decoder_directions: bool = False

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
    check_point_save_mode: str = "log"  # 'log' or 'linear'
    checkpoint_thresholds: list = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()

        if is_master():
            if os.path.exists(os.path.join(self.exp_result_path, "checkpoints")):
                raise ValueError(
                    f"Checkpoints for experiment {self.exp_result_path} already exist. Consider changing the experiment name."
                )
            os.makedirs(os.path.join(self.exp_result_path, "checkpoints"))

        self.effective_batch_size = self.train_batch_size * self.sae.ddp_size
        print_once(f"Effective batch size: {self.effective_batch_size}")

        total_training_steps = self.total_training_tokens // self.effective_batch_size
        print_once(f"Total training steps: {total_training_steps}")
        if self.lr_scheduler_name == "constantwithwarmup" and isinstance(self.lr_warm_up_steps, float):
            assert 0 <= self.lr_warm_up_steps <= 1.0
            self.lr_warm_up_steps = int(self.lr_warm_up_steps * total_training_steps)
            print_once(f"Learning rate warm up steps: {self.lr_warm_up_steps}")
        if isinstance(self.sae.l1_coefficient_warmup_steps, float) and self.sae.act_fn != "topk":
            assert 0 <= self.sae.l1_coefficient_warmup_steps <= 1.0
            self.sae.l1_coefficient_warmup_steps = int(self.sae.l1_coefficient_warmup_steps * total_training_steps)
            print_once(f"L1 coefficient warm up steps: {self.sae.l1_coefficient_warmup_steps}")
        if isinstance(self.sae.k_warmup_steps, float) and self.sae.act_fn == "topk":
            assert 0 <= self.sae.k_warmup_steps <= 1.0
            self.sae.k_warmup_steps = int(self.sae.k_warmup_steps * total_training_steps)
            print_once(f"K warm up steps: {self.sae.k_warmup_steps}")
        if isinstance(self.lr_cool_down_steps, float):
            assert 0 <= self.lr_cool_down_steps <= 1.0
            self.lr_cool_down_steps = int(self.lr_cool_down_steps * total_training_steps)
            print_once(f"Learning rate cool down steps: {self.lr_cool_down_steps}")

        if self.lr_scheduler_name == "constantwithwarmup" and isinstance(self.lr_warm_up_steps, float):
            assert 0 <= self.lr_warm_up_steps <= 1.0
            self.lr_warm_up_steps = int(self.lr_warm_up_steps * total_training_steps)
            print_once(f"Learning rate warm up steps: {self.lr_warm_up_steps}")
        if isinstance(self.sae.l1_coefficient_warmup_steps, float):
            assert 0 <= self.sae.l1_coefficient_warmup_steps <= 1.0
            self.sae.l1_coefficient_warmup_steps = int(self.sae.l1_coefficient_warmup_steps * total_training_steps)
            print_once(f"L1 coefficient warm up steps: {self.sae.l1_coefficient_warmup_steps}")
        if isinstance(self.lr_cool_down_steps, float):
            assert 0 <= self.lr_cool_down_steps <= 1.0
            self.lr_cool_down_steps = int(self.lr_cool_down_steps * total_training_steps)
            print_once(f"Learning rate cool down steps: {self.lr_cool_down_steps}")
        if self.finetuning:
            assert self.sae.l1_coefficient == 0.0, "L1 coefficient must be 0.0 for finetuning."

        if self.n_checkpoints > 0:
            if self.check_point_save_mode == "linear":
                self.checkpoint_thresholds = list(
                    range(0, self.total_training_tokens, self.total_training_tokens // self.n_checkpoints)
                )[1:]
            elif self.check_point_save_mode == "log":
                self.checkpoint_thresholds = [
                    math.ceil(2 ** (i / self.n_checkpoints * math.log2(total_training_steps)))
                    * self.effective_batch_size
                    for i in range(1, self.n_checkpoints)
                ]
            else:
                raise ValueError(f"Unknown checkpoint save mode: {self.check_point_save_mode}")

        assert 0 <= self.lr_end_ratio <= 1, "lr_end_ratio must be in 0 to 1 (inclusive)."


@dataclass(kw_only=True)
class LanguageModelSAEPruningConfig(LanguageModelSAERunnerConfig):
    """
    Configuration for pruning a sparse autoencoder on a language model.
    """

    total_training_tokens: int = 10_000_000
    train_batch_size: int = 4096

    dead_feature_threshold: float = 1e-6
    dead_feature_max_act_threshold: float = 1.0
    decoder_norm_threshold: float = 0.99

    def __post_init__(self):
        super().__post_init__()

        if is_master():
            os.makedirs(
                os.path.join(self.exp_result_path, "checkpoints"),
                exist_ok=True,
            )


@dataclass(kw_only=True)
class ActivationGenerationConfig(RunnerConfig):
    lm: LanguageModelConfig
    dataset: TextDatasetConfig

    hook_points: list[str] = field(default_factory=list)

    activation_save_path: str = None  # type: ignore

    total_generating_tokens: int = 300_000_000
    chunk_size: int = int(0.5 * 2**30)  # 0.5 GB
    ddp_size: int = 1
    tp_size: int = 1

    def __post_init__(self):
        super().__post_init__()

        if self.activation_save_path is None:
            assert (
                isinstance(self.dataset.dataset_path, list) and len(self.dataset.dataset_path) == 1
            ), "Only one dataset path is supported for activation generation."
            self.activation_save_path = f"activations/{self.dataset.dataset_path[0].split('/')[-1]}/{self.lm.model_name.replace('/', '_')}_{self.dataset.context_size}"
        os.makedirs(self.activation_save_path, exist_ok=True)


@dataclass(kw_only=True)
class MongoConfig(BaseConfig):
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "mechinterp"


@dataclass(kw_only=True)
class LanguageModelSAEAnalysisConfig(RunnerConfig):
    """
    Configuration for analyzing a sparse autoencoder on a language model.
    """

    sae: SAEConfig
    lm: LanguageModelConfig
    act_store: ActivationStoreConfig
    mongo: MongoConfig

    total_analyzing_tokens: int = 300_000_000
    enable_sampling: bool = False  # If True, we will sample the activations based on weights. Otherwise, top n_samples activations will be used.
    sample_weight_exponent: float = 2.0
    subsample: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {"top_activations": {"proportion": 1.0, "n_samples": 10}}
    )

    n_sae_chunks: int = 1  # Number of chunks to split the SAE into for analysis. For large models and SAEs, this can be useful to avoid memory issues.

    def __post_init__(self):
        super().__post_init__()
        assert (
            self.sae.d_sae % self.n_sae_chunks == 0
        ), f"d_sae ({self.sae.d_sae}) must be divisible by n_sae_chunks ({self.n_sae_chunks})"


@dataclass(kw_only=True)
class FeaturesDecoderConfig(RunnerConfig):
    sae: SAEConfig
    lm: LanguageModelConfig
    mongo: MongoConfig
    top: int = 10
