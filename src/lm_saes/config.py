import json
import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Optional, Tuple

import torch
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
)

from .utils.huggingface import parse_pretrained_name_or_path
from .utils.misc import convert_str_to_torch_dtype, convert_torch_dtype_to_str


class BaseConfig(BaseModel):
    pass


class BaseModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # allow parsing torch.dtype

    device: str = Field(default="cpu", exclude=True)

    dtype: Annotated[
        torch.dtype,
        BeforeValidator(lambda v: convert_str_to_torch_dtype(v) if isinstance(v, str) else v),
        PlainSerializer(convert_torch_dtype_to_str),
        WithJsonSchema(
            {
                "type": "string",
            },
            mode="serialization",
        ),
    ] = Field(default=torch.bfloat16, exclude=True, validate_default=False)


class BaseSAEConfig(BaseModelConfig):
    """
    Base class for SAE configs.
    Initializer will initialize SAE based on config type.
    So this class should not be used directly but only as a base config class for other SAE variants like SAEConfig, MixCoderConfig, CrossCoderConfig, etc.
    """

    sae_type: Literal["sae", "crosscoder", "mixcoder"]
    hook_point_in: str
    hook_point_out: str = Field(default_factory=lambda validated_model: validated_model["hook_point_in"])
    d_model: int
    expansion_factor: int
    use_decoder_bias: bool = True
    use_glu_encoder: bool = False
    act_fn: Literal["relu", "jumprelu", "topk", "batchtopk"] = "relu"
    jump_relu_threshold: float = 0.0
    apply_decoder_bias_to_pre_encoder: bool = False
    norm_activation: str = "dataset-wise"
    sparsity_include_decoder_norm: bool = True
    force_unit_decoder_norm: bool = False
    top_k: int = 50
    sae_pretrained_name_or_path: Optional[str] = None
    strict_loading: bool = True
    use_triton_kernel: bool = False
    sparsity_threshold_for_triton_spmm_kernel: float = 0.99

    # anthropic jumprelu
    jumprelu_threshold_window: float = 2.0

    @property
    def d_sae(self) -> int:
        return self.d_model * self.expansion_factor

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, **kwargs):
        """Load the SAEConfig from a pretrained SAE name or path. Config is read from <pretrained_name_or_path>/hyperparams.json.

        Args:
            sae_path (str): The path to the pretrained SAE.
            **kwargs: Additional keyword arguments to pass to the SAEConfig constructor.
        """
        path = parse_pretrained_name_or_path(pretrained_name_or_path)
        with open(os.path.join(path, "config.json"), "r") as f:
            sae_config = json.load(f)
        sae_config["sae_pretrained_name_or_path"] = pretrained_name_or_path
        sae_config["strict_loading"] = strict_loading
        return cls.model_validate({**sae_config, **kwargs})

    def save_hyperparameters(self, sae_path: Path | str, remove_loading_info: bool = True):
        assert os.path.exists(sae_path), f"{sae_path} does not exist. Unable to save hyperparameters."
        d = self.model_dump()
        if remove_loading_info:
            d.pop("sae_pretrained_name_or_path", None)
            d.pop("strict_loading", None)

        with open(os.path.join(sae_path, "config.json"), "w") as f:
            json.dump(d, f, indent=4)


class SAEConfig(BaseSAEConfig):
    sae_type: Literal["sae", "crosscoder", "mixcoder"] = "sae"


class CrossCoderConfig(BaseSAEConfig):
    sae_type: Literal["sae", "crosscoder", "mixcoder"] = "crosscoder"


class MixCoderConfig(BaseSAEConfig):
    sae_type: Literal["sae", "crosscoder", "mixcoder"] = "mixcoder"
    modalities: dict[str, int]
    loss_weights: dict[str, float]

    @property
    def d_sae(self) -> int:
        return sum(self.modalities.values())

    @property
    def modality_names(self) -> list[str]:
        return [k for k in self.modalities.keys() if k != "shared"]

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if "shared" in self.modalities:
            assert list(self.modalities.keys())[-1] == "shared", "Shared modality must be the last modality"


class InitializerConfig(BaseConfig):
    bias_init_method: str = "all_zero"
    const_times_for_init_b_e: int = 10000
    init_decoder_norm: float | None = None
    decoder_uniform_bound: float = 1.0
    init_encoder_norm: float | None = None
    encoder_uniform_bound: float = 1.0
    init_encoder_with_decoder_transpose: bool = True
    init_encoder_with_decoder_transpose_factor: float = 1.0
    init_log_jumprelu_threshold_value: float | None = None
    init_search: bool = False
    state: Literal["training", "inference"] = "training"
    l1_coefficient: float | None = 0.00008


class TrainerConfig(BaseConfig):
    l1_coefficient: float | None = 0.00008
    l1_coefficient_warmup_steps: int | float = 0.1
    sparsity_loss_type: Literal["power", "tanh", None] = None
    tanh_stretch_coefficient: float = 4.0
    p: int = 1
    initial_k: int | float | None = None
    k_warmup_steps: int | float = 0.1
    use_batch_norm_mse: bool = True

    lr: float | dict[str, float] = 0.0004
    betas: Tuple[float, float] = (0.9, 0.999)
    lr_scheduler_name: Literal[
        "constant",
        "constantwithwarmup",
        "linearwarmupdecay",
        "cosineannealing",
        "cosineannealingwarmup",
        "exponentialwarmup",
    ] = "constantwithwarmup"
    lr_end_ratio: float = 1 / 32
    lr_warm_up_steps: int | float = 5000
    lr_cool_down_steps: int | float = 0.2
    clip_grad_norm: float = 0.0
    feature_sampling_window: int = 1000
    total_training_tokens: int = 300_000_000

    log_frequency: int = 1000
    eval_frequency: int = 1000
    n_checkpoints: int = 10
    check_point_save_mode: Literal["log", "linear"] = "log"

    exp_result_path: Path = Path("results")

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.exp_result_path.exists():
            raise ValueError(
                f"Checkpoints for experiment {self.exp_result_path} already exist. Consider changing the experiment name."
            )
        self.exp_result_path.mkdir(parents=True, exist_ok=True)
        self.exp_result_path.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
        assert self.lr_end_ratio <= 1, "lr_end_ratio must be in 0 to 1 (inclusive)."


class EvalConfig(BaseConfig):
    feature_sampling_window: int = 1000
    total_eval_tokens: int = 1000000
    use_cached_activations: bool = False
    device: str = "cpu"


class DatasetConfig(BaseConfig):
    dataset_name_or_path: str = "openwebtext"
    cache_dir: Optional[str] = None
    is_dataset_on_disk: bool = False


class ActivationFactorySource(BaseModel):
    type: str
    """ The type of the source. Used to determine the source of activations in deserialization. """
    name: str
    sample_weights: float = 1.0
    """ The sample weights to use for the source. Will be used to randomly pull tokens from the source when multiple sources are present. """


class ActivationFactoryDatasetSource(ActivationFactorySource):
    type: str = "dataset"
    is_dataset_tokenized: bool = False
    """ Whether the dataset is tokenized. Non-tokenized datasets should have records with fields `text`, `images`, etc. Tokenized datasets should have records with fields `tokens`, which could contain either padded or non-padded tokens. """
    prepend_bos: bool = True
    """ Whether to prepend the BOS token to each record when tokenizing. """


class ActivationFactoryActivationsSource(ActivationFactorySource):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # allow parsing torch.dtype

    type: str = "activations"
    path: str
    """ The path to the cached activations. """
    device: str = "cpu"
    """ The device to load the activations on. """
    dtype: Optional[
        Annotated[
            torch.dtype,
            BeforeValidator(lambda v: convert_str_to_torch_dtype(v) if isinstance(v, str) else v),
            PlainSerializer(convert_torch_dtype_to_str),
            WithJsonSchema(
                {
                    "type": "string",
                },
                mode="serialization",
            ),
        ]
    ] = None
    """ We might want to convert presaved bf16 activations to fp32"""
    num_workers: int = 4
    """ The number of workers to use for loading the activations. """
    prefetch: Optional[int] = 8
    """ The number of chunks to prefetch."""


class ActivationFactoryTarget(Enum):
    TOKENS = "tokens"
    """ Output non-padded and non-truncated tokens. """
    ACTIVATIONS_2D = "activations-2d"
    """ Output activations in `(seq_len, d_model)` shape. Tokens are padded and truncated to the same length. """
    ACTIVATIONS_1D = "activations-1d"
    """ Output activations in `(n_filtered_tokens, d_model)` shape. Tokens are filtered in this stage. """
    BATCHED_ACTIVATIONS_1D = "batched-activations-1d"
    """ Output batched activations in `(batch_size, d_model)` shape. Tokens may be shuffled in this stage. """

    @property
    def stage(self) -> int:
        return {
            ActivationFactoryTarget.TOKENS: 0,
            ActivationFactoryTarget.ACTIVATIONS_2D: 1,
            ActivationFactoryTarget.ACTIVATIONS_1D: 2,
            ActivationFactoryTarget.BATCHED_ACTIVATIONS_1D: 3,
        }[self]

    def __lt__(self, other: "ActivationFactoryTarget") -> bool:
        return self.stage < other.stage

    def __le__(self, other: "ActivationFactoryTarget") -> bool:
        return self.stage <= other.stage


class BufferShuffleConfig(BaseConfig):
    perm_seed: int = 42
    """ Perm seed for aligned permutation for generating activations. If `None`, will not use manual seed for Generator. """
    generator_device: Optional[str] = None
    """ The device to be assigned for the torch.Generator. If 'None', generator will be initialized on cpu as pytorch default. """


class ActivationFactoryConfig(BaseConfig):
    sources: list[ActivationFactoryDatasetSource | ActivationFactoryActivationsSource]
    """ List of sources to use for activations. Can be a dataset or a path to activations. """
    target: ActivationFactoryTarget
    """ The target to produce. """
    hook_points: list[str]
    """ The hook points to capture activations from. """
    num_workers: int = 4
    """ The number of workers to use for loading the dataset. """
    context_size: int = 128
    """ The context size to use for generating activations. All tokens will be padded or truncated to this size. """
    model_batch_size: int = 1
    """ The batch size to use for generating activations. """
    batch_size: Optional[int] = Field(
        default_factory=lambda validated_model: 64
        if validated_model["target"] == ActivationFactoryTarget.BATCHED_ACTIVATIONS_1D
        else None
    )
    """ The batch size to use for outputting `batched-activations-1d`. """
    buffer_size: Optional[int] = Field(
        default_factory=lambda validated_model: 500_000
        if validated_model["target"] == ActivationFactoryTarget.BATCHED_ACTIVATIONS_1D
        else None
    )
    """ Buffer size for online shuffling. If `None`, no shuffling will be performed. """
    buffer_shuffle: Optional[BufferShuffleConfig] = None
    """" Manual seed and device of generator for generating randomperm in buffer. """
    ignore_token_ids: Optional[list[int]] = None
    """ Tokens to ignore in the activations. """


class LanguageModelConfig(BaseModelConfig):
    model_name: str = "gpt2"
    """ The name of the model to use. """
    model_from_pretrained_path: Optional[str] = None
    """ The path to the pretrained model. If `None`, will use the model from HuggingFace. """
    use_flash_attn: bool = False
    """ Whether to use Flash Attention. """
    cache_dir: Optional[str] = None
    """ The directory of the HuggingFace cache. Should have the same effect as `HF_HOME`. """
    d_model: int = 768
    """ The dimension of the model. """
    local_files_only: bool = False
    """ Whether to only load the model from the local files. Should have the same effect as `HF_HUB_OFFLINE=1`. """
    max_length: int = 2048
    """ The maximum length of the input. """
    backend: Literal["huggingface", "transformer_lens", "auto"] = "auto"
    """ The backend to use for the language model. """

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
        return LanguageModelConfig.model_validate(lm_config, **kwargs)

    def save_lm_config(self, sae_path: str):
        assert os.path.exists(sae_path), f"{sae_path} does not exist. Unable to save LanguageModelConfig."

        d = self.model_dump()
        with open(os.path.join(sae_path, "lm_config.json"), "w") as f:
            json.dump(d, f, indent=4)


class ActivationWriterConfig(BaseConfig):
    hook_points: list[str]
    """ The hook points to capture activations from. """
    total_generating_tokens: Optional[int] = None
    """ The total number of tokens to generate. If `None`, will write all activations to disk. """
    n_samples_per_chunk: Optional[int] = None
    """ The number of samples to write to disk per chunk. If `None`, will not further batch the activations. """
    cache_dir: str | Path = Path("activations")
    """ The directory to save the activations. """
    format: Literal["pt", "safetensors"] = "safetensors"
    num_workers: Optional[int] = None
    """ The number of workers to use for writing the activations. If `None`, will not use multi-threaded writing. """


class FeatureAnalyzerConfig(BaseConfig):
    total_analyzing_tokens: int
    """ Total number of tokens to analyze """

    enable_sampling: bool = False
    """ Whether to use weighted sampling for selecting activations. 
        If `False`, will only keep top activations (below the subsample threshold). 
    """

    sample_weight_exponent: float = 2.0
    """ Exponent for weighting samples by activation value """

    ignore_token_ids: Optional[list[int | None]] = None
    """ Tokens to ignore in the activations. """

    subsamples: dict[str, dict[str, int | float]] = Field(
        default_factory=lambda: {"top_activations": {"proportion": 1.0, "n_samples": 10}}
    )
    """ Dictionary mapping subsample names to their parameters:
        - `proportion`: Proportion of max activation to consider
        - `n_samples`: Number of samples to keep
    """


class WandbConfig(BaseConfig):
    wandb_project: str = "gpt2-sae-training"
    exp_name: Optional[str] = None
    wandb_entity: Optional[str] = None


class MongoDBConfig(BaseConfig):
    mongo_uri: str = Field(default_factory=lambda: os.environ.get("MONGO_URI", "mongodb://localhost:27017/"))
    mongo_db: str = Field(default_factory=lambda: os.environ.get("MONGO_DB", "mechinterp"))
