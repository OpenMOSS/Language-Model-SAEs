from typing import Annotated

import torch
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
)

from lm_saes.utils.misc import (
    convert_str_to_torch_dtype,
    convert_torch_dtype_to_str,
)


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
