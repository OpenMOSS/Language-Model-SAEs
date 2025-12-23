import json
import os
from abc import ABC, abstractmethod
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
from typing_extensions import override

from .utils.huggingface import parse_pretrained_name_or_path
from .utils.misc import (
    convert_str_to_torch_dtype,
    convert_torch_dtype_to_str,
)

SAE_TYPE_TO_CONFIG_CLASS = {}


def register_sae_config(name):
    def _register(cls):
        SAE_TYPE_TO_CONFIG_CLASS[name] = cls
        return cls

    return _register


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
    ] = Field(default=torch.float32, exclude=True, validate_default=False)


class BaseSAEConfig(BaseModelConfig, ABC):
    """
    Base class for SAE configs.
    Initializer will initialize SAE based on config type.
    So this class should not be used directly but only as a base config class for other SAE variants like SAEConfig, CrossCoderConfig, etc.
    """

    sae_type: Literal["sae", "crosscoder", "clt", "lorsa", "molt"]
    d_model: int
    expansion_factor: float
    use_decoder_bias: bool = True
    act_fn: Literal["relu", "jumprelu", "topk", "batchtopk", "batchlayertopk", "layertopk"] = "relu"
    norm_activation: str = "dataset-wise"
    sparsity_include_decoder_norm: bool = True
    top_k: int = 50
    sae_pretrained_name_or_path: str | None = None
    strict_loading: bool = True
    use_triton_kernel: bool = False
    sparsity_threshold_for_triton_spmm_kernel: float = 0.996
    sparsity_threshold_for_csr: float = 0.05
    # anthropic jumprelu
    jumprelu_threshold_window: float = 2.0
    promote_act_fn_dtype: Annotated[
        torch.dtype | None,
        BeforeValidator(lambda v: convert_str_to_torch_dtype(v) if isinstance(v, str) else v),
        PlainSerializer(convert_torch_dtype_to_str),
        WithJsonSchema(
            {
                "type": ["string", "null"],
            },
            mode="serialization",
        ),
    ] = Field(default=None, exclude=True, validate_default=False)

    @property
    def d_sae(self) -> int:
        d_sae = int(self.d_model * self.expansion_factor)
        return d_sae

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

        if cls is BaseSAEConfig:
            cls = SAE_TYPE_TO_CONFIG_CLASS[sae_config["sae_type"]]
        return cls.model_validate({**sae_config, **kwargs})

    def save_hyperparameters(self, sae_path: str | Path, remove_loading_info: bool = True):
        assert os.path.exists(sae_path), f"{sae_path} does not exist. Unable to save hyperparameters."
        d = self.model_dump()
        if remove_loading_info:
            d.pop("sae_pretrained_name_or_path", None)
            d.pop("strict_loading", None)

        with open(os.path.join(sae_path, "config.json"), "w") as f:
            json.dump(d, f, indent=4)

    @property
    @abstractmethod
    def associated_hook_points(self) -> list[str]:
        pass


@register_sae_config("sae")
class SAEConfig(BaseSAEConfig):
    sae_type: Literal["sae", "crosscoder", "clt", "lorsa", "molt"] = "sae"
    hook_point_in: str
    hook_point_out: str
    use_glu_encoder: bool = False

    @property
    def associated_hook_points(self) -> list[str]:
        return [self.hook_point_in, self.hook_point_out]


@register_sae_config("lorsa")
class LorsaConfig(BaseSAEConfig):
    """Configuration for Low Rank Sparse Attention."""

    sae_type: Literal["sae", "crosscoder", "clt", "lorsa", "molt"] = "lorsa"

    hook_point_in: str
    hook_point_out: str

    # Attention dimensions
    n_qk_heads: int
    d_qk_head: int
    positional_embedding_type: Literal["rotary", "none"] = "rotary"
    rotary_dim: int
    rotary_base: int = 10000
    rotary_adjacent_pairs: bool = True
    rotary_scale: int = 1
    use_NTK_by_parts_rope: bool = False
    NTK_by_parts_factor: float = 1.0
    NTK_by_parts_low_freq_factor: float = 1.0
    NTK_by_parts_high_freq_factor: float = 1.0
    old_context_len: int = 2048
    n_ctx: int

    # Attention settings
    attn_scale: float | None = None
    use_post_qk_ln: bool = False
    normalization_type: Literal["LN", "RMS"] | None = None
    eps: float = 1e-6

    @property
    def n_ov_heads(self) -> int:
        return self.d_sae

    @property
    def ov_group_size(self) -> int:
        return self.n_ov_heads // self.n_qk_heads

    @property
    def associated_hook_points(self) -> list[str]:
        """All hook points used by Lorsa."""
        return [self.hook_point_in, self.hook_point_out]

    def model_post_init(self, __context):
        super().model_post_init(__context)
        assert self.hook_point_in is not None and self.hook_point_out is not None, (
            "hook_point_in and hook_point_out must be set"
        )
        assert self.hook_point_in != self.hook_point_out, "hook_point_in and hook_point_out must be different"
        assert self.n_ov_heads % self.n_qk_heads == 0, "n_ov_heads must be divisible by n_qk_heads"

        if self.attn_scale is None:
            self.attn_scale = self.d_qk_head**0.5


@register_sae_config("clt")
class CLTConfig(BaseSAEConfig):
    """Configuration for Cross Layer Transcoder (CLT).

    A CLT consists of L encoders and L(L+1)/2 decoders where each encoder at layer L
    reads from the residual stream at that layer and can decode to layers L through L-1.
    """

    sae_type: Literal["sae", "crosscoder", "clt", "lorsa", "molt"] = "clt"
    act_fn: Literal["relu", "jumprelu", "topk", "batchtopk", "batchlayertopk", "layertopk"] = "relu"
    init_cross_layer_decoder_all_zero: bool = False
    hook_points_in: list[str]
    """List of hook points to capture input activations from, one for each layer."""
    hook_points_out: list[str]
    """List of hook points to capture output activations from, one for each layer."""
    decode_with_csr: bool = False
    """Whether to decode with CSR matrices. If `True`, will use CSR matrices for decoding. If `False`, will use dense matrices for decoding."""

    @property
    def n_layers(self) -> int:
        """Number of layers in the CLT."""
        return len(self.hook_points_in)

    @property
    def n_decoders(self) -> int:
        """Number of decoders in the CLT."""
        return self.n_layers * (self.n_layers + 1) // 2

    @property
    def associated_hook_points(self) -> list[str]:
        """All hook points used by the CLT."""
        return self.hook_points_in + self.hook_points_out

    def model_post_init(self, __context):
        super().model_post_init(__context)
        assert len(self.hook_points_in) == len(self.hook_points_out), (
            "Number of input and output hook points must match"
        )


@register_sae_config("molt")
class MOLTConfig(BaseSAEConfig):
    """Configuration for Mixture of Linear Transforms (MOLT).

    MOLT is a more efficient alternative to transcoders that sparsely replaces
    MLP computation in transformers. It converts dense MLP layers into sparse,
    interpretable linear transforms.
    """

    sae_type: Literal["sae", "crosscoder", "clt", "lorsa", "molt"] = "molt"
    hook_point_in: str
    """Hook point to capture input activations from."""
    hook_point_out: str
    """Hook point to output activations to."""
    rank_counts: dict[int, int]
    """Dictionary mapping rank values to their integer counts.
    Example: {4: 128, 8: 256, 16: 128} means 128 transforms of rank 4, 256 transforms of rank 8, and 128 transforms of rank 16.
    """

    def model_post_init(self, __context):
        super().model_post_init(__context)
        # Validate counts
        assert self.rank_counts, "rank_counts cannot be empty"

        for rank, count in self.rank_counts.items():
            assert rank > 0, f"Rank must be positive, got {rank}"
            assert count > 0, f"Count for rank {rank} must be positive, got {count}"

        # Workaround: expansion_factor is not used in MOLT, but we keep it for consistency with other SAE variants.
        assert abs(self.expansion_factor - self.d_sae / self.d_model) < 0.1, (
            f"Expansion factor {self.expansion_factor} is not close to d_sae / d_model {self.d_sae / self.d_model}"
        )

    def generate_rank_assignments(self) -> list[int]:
        """Generate rank assignment for each of the d_sae linear transforms.

        Returns:
            List of rank assignments for each transform.
            For example: [1, 1, 1, 1, 2, 2, 4].
        """
        assignments = []
        for rank in sorted(self.rank_counts.keys()):
            assignments.extend([rank] * self.rank_counts[rank])
        return assignments

    def get_local_rank_assignments(self, model_parallel_size: int) -> list[int]:
        """Get rank assignments for a specific local device in distributed running.

        Each device gets all rank groups, with each group evenly divided across devices.
        This ensures consistent encoder/decoder sharding without feature_acts redistribution.

        Args:
            model_parallel_size: Number of model parallel devices for training and inference.

        Returns:
            List of rank assignments for this local device
            For example:
            global_rank_assignments = [1, 1, 2, 2], model_parallel_size = 2 -> local_rank_assignments = [1, 2]
        """
        local_assignments = []
        for rank in sorted(self.rank_counts.keys()):
            global_count = self.rank_counts[rank]

            # Verify even division
            assert global_count % model_parallel_size == 0, (
                f"Transform rank {rank} global count {global_count} not divisible by "
                f"model_parallel_size {model_parallel_size}"
            )

            local_count = global_count // model_parallel_size
            local_assignments.extend([rank] * local_count)

        return local_assignments

    @property
    @override
    def d_sae(self) -> int:
        """Calculate d_sae based on total rank counts."""
        return sum(self.rank_counts.values())

    @property
    def available_ranks(self) -> list[int]:
        """Get sorted list of available ranks."""
        return sorted(self.rank_counts.keys())

    @property
    def num_rank_types(self) -> int:
        """Number of different rank types."""
        return len(self.rank_counts)

    @property
    def associated_hook_points(self) -> list[str]:
        return [self.hook_point_in, self.hook_point_out]


@register_sae_config("crosscoder")
class CrossCoderConfig(BaseSAEConfig):
    sae_type: Literal["sae", "crosscoder", "clt", "lorsa", "molt"] = "crosscoder"
    hook_points: list[str]

    @property
    def associated_hook_points(self) -> list[str]:
        return self.hook_points

    @property
    def n_heads(self) -> int:
        return len(self.hook_points)


class InitializerConfig(BaseConfig):
    bias_init_method: Literal["all_zero", "geometric_median"] = "all_zero"
    decoder_uniform_bound: float = 1.0
    encoder_uniform_bound: float = 1.0
    init_encoder_with_decoder_transpose: bool = True
    init_encoder_with_decoder_transpose_factor: float = 1.0
    init_log_jumprelu_threshold_value: float | None = None
    grid_search_init_norm: bool = False
    initialize_W_D_with_active_subspace: bool = False
    d_active_subspace: int | None = None
    initialize_lorsa_with_mhsa: bool | None = None
    initialize_tc_with_mlp: bool | None = None
    model_layer: int | None = None
    init_encoder_bias_with_mean_hidden_pre: bool = False


class TrainerConfig(BaseConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    l1_coefficient: float | None = 0.00008
    l1_coefficient_warmup_steps: int | float = 0.1
    lp_coefficient: float | None = None
    amp_dtype: Annotated[
        torch.dtype | None,
        BeforeValidator(lambda v: convert_str_to_torch_dtype(v) if isinstance(v, str) else v),
        PlainSerializer(convert_torch_dtype_to_str),
        WithJsonSchema(
            {
                "type": ["string", "null"],
            },
            mode="serialization",
        ),
    ] = Field(default=torch.bfloat16, exclude=True, validate_default=False)
    sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None
    tanh_stretch_coefficient: float = 4.0
    frequency_scale: float = 0.01
    p: int = 1
    initial_k: int | float | None = None
    k_warmup_steps: int | float = 0.1
    k_cold_booting_steps: int | float = 0
    k_schedule_type: Literal["linear", "exponential"] = "linear"
    k_exponential_factor: float = 3.0
    skip_metrics_calculation: bool = False
    gradient_accumulation_steps: int = 1

    lr: float | dict[str, float] = 0.0004
    betas: Tuple[float, float] = (0.9, 0.999)
    optimizer_class: Literal["adam", "sparseadam"] = "adam"
    optimizer_foreach: bool = True
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
    jumprelu_lr_factor: float = 1.0
    clip_grad_norm: float = 0.0
    feature_sampling_window: int = 1000
    total_training_tokens: int = 300_000_000

    log_frequency: int = 1000
    eval_frequency: int = 1000
    n_checkpoints: int = 10
    check_point_save_mode: Literal["log", "linear"] = "log"

    from_pretrained_path: str | None = None
    exp_result_path: str = "results"

    def model_post_init(self, __context):
        super().model_post_init(__context)
        Path(self.exp_result_path).mkdir(parents=True, exist_ok=True)
        Path(self.exp_result_path, "checkpoints").mkdir(parents=True, exist_ok=True)
        assert self.lr_end_ratio <= 1, "lr_end_ratio must be in 0 to 1 (inclusive)."

        if self.from_pretrained_path is not None:
            assert os.path.exists(self.from_pretrained_path), (
                f"from_pretrained_path {self.from_pretrained_path} does not exist"
            )


class EvalConfig(BaseConfig):
    total_eval_tokens: int = 1000000


class GraphEvalConfig(BaseConfig):
    max_n_logits: int = 2
    # How many logits to attribute from, max. We attribute to min(max_n_logits, n_logits_to_reach_desired_log_prob); see below for the latter

    desired_logit_prob: float = 0.95
    # Attribution will attribute from the minimum number of logits needed to reach this probability mass (or max_n_logits, whichever is lower)

    max_feature_nodes: int = 1024
    # Only attribute from this number of feature nodes, max. Lower is faster, but you will lose more of the graph. None means no limit.

    batch_size: int = 2
    # Batch size when attributing

    offload: Literal[None, "disk", "cpu"] = None
    # Offload various parts of the model during attribution to save memory. Can be 'disk', 'cpu', or None (keep on GPU)

    start_from: int = 0


class DatasetConfig(BaseConfig):
    dataset_name_or_path: str = "openwebtext"
    cache_dir: str | None = None
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
    path: str | dict[str, str]
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
    prefetch: int | None = 8
    """ The number of chunks to prefetch."""


class ActivationFactoryTarget(Enum):
    TOKENS = "tokens"
    """ Output non-padded and non-truncated tokens. """
    ACTIVATIONS_2D = "activations-2d"
    """ Output activations in `(batch_size, seq_len, d_model)` shape. Tokens are padded and truncated to the same length. """
    ACTIVATIONS_1D = "activations-1d"
    """ Output activations in `(n_filtered_tokens, d_model)` shape. Tokens are filtered in this stage. """

    @property
    def stage(self) -> int:
        return {
            ActivationFactoryTarget.TOKENS: 0,
            ActivationFactoryTarget.ACTIVATIONS_2D: 1,
            ActivationFactoryTarget.ACTIVATIONS_1D: 2,
        }[self]

    def __lt__(self, other: "ActivationFactoryTarget") -> bool:
        return self.stage < other.stage

    def __le__(self, other: "ActivationFactoryTarget") -> bool:
        return self.stage <= other.stage


class BufferShuffleConfig(BaseConfig):
    perm_seed: int = 42
    """ Perm seed for aligned permutation for generating activations. If `None`, will not use manual seed for Generator. """
    generator_device: str | None = None
    """ The device to be assigned for the torch.Generator. If 'None', generator will be initialized on cpu as pytorch default. """


class ActivationFactoryConfig(BaseConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # allow parsing torch.dtype

    sources: list[ActivationFactoryDatasetSource | ActivationFactoryActivationsSource]
    """ List of sources to use for activations. Can be a dataset or a path to activations. """
    target: ActivationFactoryTarget
    """ The target to produce. """
    hook_points: list[str]
    """ The hook points to capture activations from. """
    batch_size: int
    """ The batch size to use for outputting activations. """
    num_workers: int = 4
    """ The number of workers to use for loading the dataset. """
    context_size: int | None = None
    """ The context size to use for generating activations. All tokens will be padded or truncated to this size. If `None`, will not pad or truncate tokens. This may lead to some error when re-batching activations of different context sizes."""
    model_batch_size: int = 1
    """ The batch size to use for model forward pass when generating activations. """
    override_dtype: Optional[
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
    """ The dtype to use for outputting activations. If `None`, will not override the dtype. """
    buffer_size: int | None = None
    """ Buffer size for online shuffling. If `None`, no shuffling will be performed. """
    buffer_shuffle: BufferShuffleConfig | None = None
    """" Manual seed and device of generator for generating randomperm in buffer. """
    ignore_token_ids: list[int] | None = None
    """ Tokens to ignore in the activations. """


class LanguageModelConfig(BaseModelConfig):
    model_name: str = "gpt2"
    """ The name of the model to use. """
    model_from_pretrained_path: str | None = None
    """ The path to the pretrained model. If `None`, will use the model from HuggingFace. """
    use_flash_attn: bool = False
    """ Whether to use Flash Attention. """
    cache_dir: str | None = None
    """ The directory of the HuggingFace cache. Should have the same effect as `HF_HOME`. """
    local_files_only: bool = False
    """ Whether to only load the model from the local files. Should have the same effect as `HF_HUB_OFFLINE=1`. """
    max_length: int = 2048
    """ The maximum length of the input. """
    backend: Literal["huggingface", "transformer_lens", "auto"] = "auto"
    """ The backend to use for the language model. """
    load_ckpt: bool = True
    tokenizer_only: bool = False
    """ Whether to only load the tokenizer. """
    prepend_bos: bool = True
    """ Whether to prepend the BOS token to the input. """
    bos_token_id: int | None = None
    """ The ID of the BOS token. If `None`, will use the default BOS token. """
    eos_token_id: int | None = None
    """ The ID of the EOS token. If `None`, will use the default EOS token. """
    pad_token_id: int | None = None
    """ The ID of the padding token. If `None`, will use the default padding token. """

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


class LLaDAConfig(LanguageModelConfig):
    mask_ratio: float = 0.0
    mdm_mask_token_id: int = 126336
    prepend_bos: bool = False
    calculate_logits: bool = False


class ActivationWriterConfig(BaseConfig):
    hook_points: list[str]
    """ The hook points to capture activations from. """
    total_generating_tokens: int | None = None
    """ The total number of tokens to generate. If `None`, will write all activations to disk. """
    n_samples_per_chunk: int | None = None
    """ The number of samples to write to disk per chunk. If `None`, will not further batch the activations. """
    cache_dir: str = "activations"
    """ The directory to save the activations. """
    format: Literal["pt", "safetensors"] = "safetensors"
    num_workers: int | None = None
    """ The number of workers to use for writing the activations. If `None`, will not use multi-threaded writing. """


class FeatureAnalyzerConfig(BaseConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # allow parsing torch.dtype
    total_analyzing_tokens: int
    """ Total number of tokens to analyze """

    ignore_token_ids: list[int] | None = None
    """ Tokens to ignore in the activations. """

    subsamples: dict[str, dict[str, int | float]] = Field(
        default_factory=lambda: {
            "top_activations": {"proportion": 1.0, "n_samples": 10},
            "non_activating": {
                "proportion": 0.3,
                "n_samples": 20,
                "max_length": 50,
            },
        }
    )
    """ Dictionary mapping subsample names to their parameters:
        - `proportion`: Proportion of max activation to consider
        - `n_samples`: Number of samples to keep
        - `max_length`: Maximum length of the sample
    """

    clt_layer: int | None = None
    """ Layer to analyze for CLT. Provided iff analyzing CLT. """


class DirectLogitAttributorConfig(BaseConfig):
    top_k: int = 10
    """ The number of top tokens to attribute to. """

    clt_layer: int | None = None
    """ Layer to analyze for CLT. Provided iff analyzing CLT. """


class WandbConfig(BaseConfig):
    wandb_project: str = "gpt2-sae-training"
    exp_name: str | None = None
    wandb_entity: str | None = None
    wandb_run_id: str | None = None
    wandb_resume: Literal["allow", "must", "never", "auto"] = "never"


class MongoDBConfig(BaseConfig):
    mongo_uri: str = Field(default_factory=lambda: os.environ.get("MONGO_URI", "mongodb://localhost:27017/"))
    mongo_db: str = Field(default_factory=lambda: os.environ.get("MONGO_DB", "mechinterp"))
