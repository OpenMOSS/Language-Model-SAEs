import math
from typing import Any, Literal, Union, overload

import torch
import torch.distributed.tensor
from jaxtyping import Float
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from transformer_lens.hook_points import HookPoint
from typing_extensions import override

from lm_saes.utils.distributed import DimMap
from lm_saes.utils.logging import get_distributed_logger
from lm_saes.utils.timer import timer

from .abstract_sae import AbstractSparseAutoEncoder
from .config import SAEConfig

logger = get_distributed_logger("sae")


class SparseAutoEncoder(AbstractSparseAutoEncoder):
    def __init__(self, cfg: SAEConfig, device_mesh: DeviceMesh | None = None):
        super(SparseAutoEncoder, self).__init__(cfg, device_mesh=device_mesh)
        self.cfg = cfg

        if device_mesh is None:
            self.W_E = nn.Parameter(torch.empty(cfg.d_model, cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
            self.b_E = nn.Parameter(torch.empty(cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
            self.W_D = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_model, device=cfg.device, dtype=cfg.dtype))
            if cfg.use_decoder_bias:
                self.b_D = nn.Parameter(torch.empty(cfg.d_model, device=cfg.device, dtype=cfg.dtype))

            if cfg.use_glu_encoder:
                self.W_E_glu = nn.Parameter(torch.empty(cfg.d_model, cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
                self.b_E_glu = nn.Parameter(torch.empty(cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
        else:
            self.W_E = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.d_model,
                    cfg.d_sae,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["W_E"].placements(device_mesh),
                )
            )
            self.b_E = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.d_sae,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["b_E"].placements(device_mesh),
                )
            )
            self.W_D = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.d_sae,
                    cfg.d_model,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["W_D"].placements(device_mesh),
                )
            )
            if cfg.use_decoder_bias:
                self.b_D = nn.Parameter(
                    torch.distributed.tensor.empty(
                        cfg.d_model,
                        dtype=cfg.dtype,
                        device_mesh=device_mesh,
                        placements=self.dim_maps()["b_D"].placements(device_mesh),
                    )
                )
            if cfg.use_glu_encoder:
                self.W_E_glu = nn.Parameter(
                    torch.distributed.tensor.empty(
                        cfg.d_model,
                        cfg.d_sae,
                        dtype=cfg.dtype,
                        device_mesh=device_mesh,
                        placements=self.dim_maps()["W_E_glu"].placements(device_mesh),
                    )
                )
                self.b_E_glu = nn.Parameter(
                    torch.distributed.tensor.empty(
                        cfg.d_sae,
                        dtype=cfg.dtype,
                        device_mesh=device_mesh,
                        placements=self.dim_maps()["b_E_glu"].placements(device_mesh),
                    )
                )

        self.hook_hidden_pre = HookPoint()
        self.hook_feature_acts = HookPoint()
        self.hook_reconstructed = HookPoint()
        
        # Initialize dead latents tracking buffers after all parameters are set up
        if self.cfg.use_auxk and self.cfg.act_fn == "topk":
            print("======== use auxk loss set up ========")
            if device_mesh is None:
                self.register_buffer('tokens_since_last_activation', torch.zeros(self.cfg.d_sae, device=cfg.device, dtype=torch.long))
                self.register_buffer('is_dead', torch.zeros(self.cfg.d_sae, device=cfg.device, dtype=torch.bool))
            else:
                self.register_buffer('tokens_since_last_activation', torch.distributed.tensor.zeros(
                    self.cfg.d_sae,
                    dtype=torch.long,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["tokens_since_last_activation"].placements(device_mesh),
                ))
                self.register_buffer('is_dead', torch.distributed.tensor.zeros(
                    self.cfg.d_sae,
                    dtype=torch.bool,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["is_dead"].placements(device_mesh),
                ))

    @override
    def encoder_norm(self, keepdim: bool = False):
        """Compute the norm of the encoder weight."""
        if not isinstance(self.W_E, DTensor):
            return torch.norm(self.W_E, p=2, dim=0, keepdim=keepdim).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            return DTensor.from_local(
                torch.norm(self.W_E.to_local(), p=2, dim=0, keepdim=keepdim),
                device_mesh=self.device_mesh,
                placements=DimMap({"model": 1 if keepdim else 0}).placements(self.device_mesh),
            )

    @override
    def decoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the norm of the decoder weight."""
        if not isinstance(self.W_D, DTensor):
            return torch.norm(self.W_D, p=2, dim=1, keepdim=keepdim).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            return DTensor.from_local(
                torch.norm(self.W_D.to_local(), p=2, dim=1, keepdim=keepdim),
                device_mesh=self.device_mesh,
                placements=DimMap({"model": 0}).placements(self.device_mesh),
            )

    @override
    def decoder_bias_norm(self) -> torch.Tensor:
        if not self.cfg.use_decoder_bias:
            raise ValueError("Decoder bias is not used")
        if not isinstance(self.b_D, DTensor):
            return torch.norm(self.b_D, p=2, dim=0, keepdim=True).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            return DTensor.from_local(
                torch.norm(self.b_D.to_local(), p=2, dim=0, keepdim=True),
                device_mesh=self.device_mesh,
                placements=DimMap({}).placements(self.device_mesh),
            )

    @override
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool):
        if force_exact:
            self.W_D.mul_(value / self.decoder_norm(keepdim=True))
        else:
            self.W_D.mul_(value / torch.clamp(self.decoder_norm(keepdim=True), min=value))

    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float):
        self.W_E.mul_(value / self.encoder_norm(keepdim=True))

    def dim_maps(self) -> dict[str, DimMap]:
        """Return a dictionary mapping parameter names to dimension maps.

        Returns:
            A dictionary mapping parameter names to DimMap objects.
        """
        parent_maps = super().dim_maps()
        sae_maps = {
            "W_E": DimMap({"model": 1}),
            "W_D": DimMap({"model": 0}),
            "b_E": DimMap({"model": 0}),
        }
        if self.cfg.use_decoder_bias:
            sae_maps["b_D"] = DimMap({})
        if self.cfg.use_glu_encoder:
            sae_maps["W_E_glu"] = DimMap({"model": 1})
            sae_maps["b_E_glu"] = DimMap({"model": 0})
        if self.cfg.use_auxk and self.cfg.act_fn == "topk":
            sae_maps["tokens_since_last_activation"] = DimMap({"model": 0})
            sae_maps["is_dead"] = DimMap({"model": 0})
        return parent_maps | sae_maps

    @override
    @torch.no_grad()
    def transform_to_unit_decoder_norm(self):
        self.W_D.mul_(1 / self.decoder_norm(keepdim=False))

    @torch.no_grad()
    def standardize_parameters_of_dataset_norm(
        self, dataset_average_activation_norm: dict[str, float] | None = None
    ):  # should be overridden by subclasses due to side effects
        """
        Standardize the parameters of the model to account for dataset_norm during inference.
        This function should be called during inference by the Initializer.

        During training, the activations correspond to an input `x` where the norm is sqrt(d_model).
        However, during inference, the norm of the input `x` corresponds to the dataset_norm.
        To ensure consistency between training and inference, the activations during inference
        are scaled by the factor:

            scaled_activation = training_activation * (dataset_norm / sqrt(d_model))

        Args:
            dataset_average_activation_norm (dict[str, float]):
                A dictionary where keys represent in or out and values
                specify the average activation norm of the dataset during inference.

                dataset_average_activation_norm = {
                    self.cfg.hook_point_in: 1.0,
                    self.cfg.hook_point_out: 1.0,
                }

        Returns:
            None: Updates the internal parameters to reflect the standardized activations and change the norm_activation to "inference" mode.
        """
        assert self.cfg.norm_activation == "dataset-wise"
        assert self.dataset_average_activation_norm is not None or dataset_average_activation_norm is not None
        if dataset_average_activation_norm is not None:
            self.set_dataset_average_activation_norm(dataset_average_activation_norm)
        assert self.dataset_average_activation_norm is not None
        input_norm_factor: float = (
            math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_in]
        )
        output_norm_factor: float = (
            math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_out]
        )
        
        # print(f'{self.cfg.hook_point_in = }')
        # print(f'{self.cfg.hook_point_out = }')
        # print(f'{self.dataset_average_activation_norm[self.cfg.hook_point_in] = }')
        # print(f'{self.dataset_average_activation_norm[self.cfg.hook_point_out] = }')
        # print(f'{input_norm_factor = }')
        # print(f'{output_norm_factor = }')
        
        self.b_E.div_(input_norm_factor)
        if self.cfg.use_decoder_bias:
            assert self.b_D is not None, "Decoder bias should exist if use_decoder_bias is True"
            self.b_D.div_(output_norm_factor)
        self.W_D.mul_(input_norm_factor / output_norm_factor)
        self.cfg.norm_activation = "inference"

    # @torch.no_grad()
    # def standardize_parameters_of_dataset_norm(
    #     self, dataset_average_activation_norm: dict[str, float] | None = None
    # ):
    #     assert self.cfg.norm_activation == "dataset-wise"
    #     assert self.dataset_average_activation_norm is not None or dataset_average_activation_norm is not None
    #     if dataset_average_activation_norm is not None:
    #         self.set_dataset_average_activation_norm(dataset_average_activation_norm)
    #     assert self.dataset_average_activation_norm is not None

    #     # 1) 计算标量（Python float）
    #     input_norm_factor: float = (
    #         math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_in]
    #     )
    #     output_norm_factor: float = (
    #         math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_out]
    #     )

    #     # 2) 选择目标 dtype 和 device —— 强制使用 torch.float32
    #     target_dtype = torch.float32

    #     # 尽量取一个已存在参数的 device（优先 b_E，再 fallback 到 model parameters，再 fallback cpu）
    #     device = None
    #     if hasattr(self, "b_E") and getattr(self.b_E, "device", None) is not None:
    #         device = self.b_E.device
    #     else:
    #         try:
    #             # next(self.parameters()) 在没有参数时会抛异常
    #             device = next(self.parameters()).device
    #         except StopIteration:
    #             device = torch.device("cpu")

    #     # 3) 把标量转成 tensor（float32）
    #     input_norm_factor_t = torch.tensor(input_norm_factor, dtype=target_dtype, device=device)
    #     output_norm_factor_t = torch.tensor(output_norm_factor, dtype=target_dtype, device=device)

    #     # debug 打印（现在不会报错）
    #     print(f'{self.cfg.hook_point_in = }')
    #     print(f'{self.cfg.hook_point_out = }')
    #     print(f'{self.dataset_average_activation_norm[self.cfg.hook_point_in] = }')
    #     print(f'{self.dataset_average_activation_norm[self.cfg.hook_point_out] = }')
    #     print(f'{input_norm_factor_t = }')
    #     print(f'{output_norm_factor_t = }')  # tensor 会打印 dtype & device

    #     # 4) 用 tensor 做原地修改（广播会生效）
    #     # 注意：如果 self.b_E / self.b_D / self.W_D 是 DTensor，需要做 DTensor 专用处理（见下方说明）。
    #     # 一般场景（nn.Parameter / torch.Tensor）下如下写法是安全的：
    #     self.b_E.div_(input_norm_factor_t)
    #     if self.cfg.use_decoder_bias:
    #         assert self.b_D is not None, "Decoder bias should exist if use_decoder_bias is True"
    #         self.b_D.div_(output_norm_factor_t)
    #     self.W_D.mul_(input_norm_factor_t / output_norm_factor_t)

    #     # 最后切换模式标志
    #     self.cfg.norm_activation = "inference"

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        return_hidden_pre: Literal[False] = False,
        **kwargs,
    ) -> Union[Float[torch.Tensor, "batch d_sae"], Float[torch.Tensor, "batch seq_len d_sae"]]: ...

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        return_hidden_pre: Literal[True],
        **kwargs,
    ) -> tuple[
        Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
    ]: ...

    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        return_hidden_pre: bool = False,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch d_sae"],
        Float[torch.Tensor, "batch seq_len d_sae"],
        tuple[
            Union[
                Float[torch.Tensor, "batch d_sae"],
                Float[torch.Tensor, "batch seq_len d_sae"],
            ],
            Union[
                Float[torch.Tensor, "batch d_sae"],
                Float[torch.Tensor, "batch seq_len d_sae"],
            ],
        ],
    ]:
        """Encode input tensor through the sparse autoencoder.

        Args:
            x: Input tensor of shape (batch, d_model) or (batch, seq_len, d_model)
            return_hidden_pre: If True, also return the pre-activation hidden states

        Returns:
            If return_hidden_pre is False:
                Feature activations tensor of shape (batch, d_sae) or (batch, seq_len, d_sae)
            If return_hidden_pre is True:
                Tuple of (feature_acts, hidden_pre) where both have shape (batch, d_sae) or (batch, seq_len, d_sae)
        """
        # Pass through encoder
        # x = x.to(torch.float64)
        hidden_pre = x @ self.W_E + self.b_E

        # Apply GLU if configured
        if self.cfg.use_glu_encoder:
            hidden_pre_glu = torch.sigmoid(x @ self.W_E_glu + self.b_E_glu)
            hidden_pre = hidden_pre * hidden_pre_glu

        hidden_pre = self.hook_hidden_pre(hidden_pre)

        # Scale feature activations by decoder norm if configured
        if self.cfg.sparsity_include_decoder_norm:
            sparsity_scores = hidden_pre * self.decoder_norm()
        else:
            sparsity_scores = hidden_pre

        # Apply activation function. The activation function here differs from a common activation function,
        # since it computes a scaling of the input tensor, which is, suppose the common activation function
        # is $f(x)$, then here it computes $f(x) / x$. For simple ReLU case, it computes a mask of 1s and 0s.
        activation_mask = self.activation_function(sparsity_scores)
        

        nonzero_indices = (activation_mask != 0).nonzero(as_tuple=False)

        # 只取前 10 个
        first10_idx = nonzero_indices[:10]

        # 对应的值
        first10_vals = activation_mask[first10_idx[:, 0], *first10_idx[:, 1:].T]

        # 打印结果
        # for idx, val in zip(first10_idx, first10_vals):
        #     print('activation_mask')
        #     print(f"Index: {tuple(idx.tolist())}, Value: {val.item()}")    
    
        # end of debugging
        
        feature_acts = self.hook_feature_acts(hidden_pre * activation_mask)

        if return_hidden_pre:
            return feature_acts, hidden_pre
        return feature_acts

    def decode(
        self,
        feature_acts: Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:  # may be overridden by subclasses
        # print(f"Indices: {feature_acts.indices().shape}") # [2,2240]
        # print(f"Values: {feature_acts.values().shape}") # [2240]
        # max_l0_in_batch = feature_acts.gt(0).to(feature_acts).sum(dim=-1).max()
        
        # 检查feature_acts是否为稀疏张量，如果不是则转换为稀疏张量
        if not feature_acts.is_sparse:
            feature_acts = feature_acts.to_sparse()
        
        non_zero_values = feature_acts.values()
        
        max_l0_in_batch = non_zero_values.gt(0).sum().max()
        sparsity_threshold = self.cfg.d_sae * (1 - self.cfg.sparsity_threshold_for_triton_spmm_kernel)
        if (
            self.cfg.use_triton_kernel and 0 < max_l0_in_batch < sparsity_threshold
        ):  # triton kernel cannot handle empty feature_acts
            from .kernels import decode_with_triton_spmm_kernel

            require_precise_feature_acts_grad = "topk" not in self.cfg.act_fn
            reconstructed = decode_with_triton_spmm_kernel(
                feature_acts, self.W_D.T.contiguous(), require_precise_feature_acts_grad
            )  # TODO: remove the transpose
        else:
            # reconstructed = feature_acts @ self.W_D
            reconstructed = torch.matmul(feature_acts.to_dense(), self.W_D)

        assert reconstructed is not None, "Reconstructed cannot be None"
        if self.cfg.use_decoder_bias:
            reconstructed = reconstructed + self.b_D
        reconstructed = self.hook_reconstructed(reconstructed)

        if isinstance(reconstructed, DTensor):
            reconstructed = DimMap({"data": 0}).redistribute(reconstructed)

        return reconstructed

    def forward(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:
        feature_acts = self.encode(x, **kwargs)
        reconstructed = self.decode(feature_acts, **kwargs)
        return reconstructed

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, fold_activation_scale: bool = True, **kwargs):
        cfg = SAEConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        return cls.from_config(cfg, fold_activation_scale=fold_activation_scale)

    @torch.no_grad()
    def _init_encoder_with_decoder_transpose(
        self, encoder: torch.nn.Linear, decoder: torch.nn.Linear, factor: float = 1.0
    ):
        encoder.weight.data = decoder.weight.data.T.clone().contiguous() * factor

    @override
    @torch.no_grad()
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0):
        self.W_E.copy_(self.W_D.contiguous().T * factor)

    @override
    @torch.no_grad()
    def init_parameters(self, **kwargs):
        super().init_parameters(**kwargs)

        W_E = torch.empty(self.cfg.d_model, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(
            -kwargs["encoder_uniform_bound"], kwargs["encoder_uniform_bound"]
        )
        W_D = torch.empty(self.cfg.d_sae, self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(
            -kwargs["decoder_uniform_bound"], kwargs["decoder_uniform_bound"]
        )
        b_E = torch.zeros(self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype)

        if self.device_mesh is not None:
            W_E = self.dim_maps()["W_E"].distribute(W_E, self.device_mesh)
            W_D = self.dim_maps()["W_D"].distribute(W_D, self.device_mesh)
            b_E = self.dim_maps()["b_E"].distribute(b_E, self.device_mesh)

        self.W_E.copy_(W_E)
        self.W_D.copy_(W_D)
        self.b_E.copy_(b_E)

        if self.cfg.use_decoder_bias:
            b_D = torch.zeros(self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype)

            if self.device_mesh is not None:
                b_D = self.dim_maps()["b_D"].distribute(b_D, self.device_mesh)

            self.b_D.copy_(b_D)

        if self.cfg.use_glu_encoder:
            W_E_glu = torch.empty(
                self.cfg.d_model, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype
            ).uniform_(-kwargs["encoder_uniform_bound"], kwargs["encoder_uniform_bound"])
            if self.device_mesh is not None:
                W_E_glu = self.dim_maps()["W_E_glu"].distribute(W_E_glu, self.device_mesh)
            self.W_E_glu.copy_(W_E_glu)

    @override
    def prepare_input(self, batch: dict[str, torch.Tensor], **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        x = batch[self.cfg.hook_point_in]
        return x, {}

    @override
    def prepare_label(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        label = batch[self.cfg.hook_point_out]
        return label
    
    @torch.no_grad()
    def update_dead_latents(self, feature_acts: torch.Tensor):
        """Update the dead latents tracking based on current feature activations.
        
        Args:
            feature_acts: Feature activations tensor of shape (batch, d_sae) or (batch, seq_len, d_sae)
        """
        if not (self.cfg.use_auxk and self.cfg.act_fn == "topk"):
            return
            
        # Calculate batch size (number of tokens in this batch)
        if feature_acts.dim() == 3:  # (batch, seq_len, d_sae)
            batch_size = feature_acts.size(0) * feature_acts.size(1)  # batch * seq_len
        else:  # (batch, d_sae)
            batch_size = feature_acts.size(0)  # batch
            
        # Check which features were activated in this batch
        if feature_acts.dim() == 3:  # (batch, seq_len, d_sae)
            activated = feature_acts.gt(0).any(dim=(0, 1))  # (d_sae,)
        else:  # (batch, d_sae)
            activated = feature_acts.gt(0).any(dim=0)  # (d_sae,)
            
        # Update tokens since last activation
        # If a feature was activated, reset to 0; otherwise, add batch_size
        self.tokens_since_last_activation = torch.where(
            activated,
            torch.zeros_like(self.tokens_since_last_activation),
            self.tokens_since_last_activation + batch_size
        )
        
        # Mark as dead if tokens since last activation exceeds threshold
        self.is_dead = self.tokens_since_last_activation >= self.cfg.dead_threshold
    
    @override
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        label: (
            Union[
                Float[torch.Tensor, "batch d_model"],
                Float[torch.Tensor, "batch seq_len d_model"],
            ]
            | None
        ) = None,
        *,
        use_batch_norm_mse: bool = False,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        frequency_scale: float = 0.01,
        p: int = 1,
        l1_coefficient: float = 1.0,
        return_aux_data: bool = True,
        log_info: dict = {},
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, " batch"],
        tuple[
            Float[torch.Tensor, " batch"],
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        ],
    ]:
        """Compute the loss for the autoencoder with optional AuxK auxiliary loss.
        Ensure that the input activations are normalized by calling `normalize_activations` before calling this method.
        """
        x, encoder_kwargs = self.prepare_input(batch)
        label = self.prepare_label(batch, **kwargs)

        feature_acts, hidden_pre = self.encode(x, return_hidden_pre=True, **encoder_kwargs)
        reconstructed = self.decode(feature_acts, **kwargs)

        # newly added
        # Update dead latents tracking
        self.update_dead_latents(feature_acts)

        with timer.time("loss_calculation"):
            l_rec = (reconstructed - label).pow(2)
            if use_batch_norm_mse:
                l_rec = (
                    l_rec
                    / (label - label.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()
                )
            l_rec = l_rec.sum(dim=-1)
            if isinstance(l_rec, DTensor):
                l_rec = l_rec.full_tensor()
            loss_dict = {
                "l_rec": l_rec,
            }
            loss = l_rec.mean()

            if sparsity_loss_type is not None:
                with timer.time("sparsity_loss_calculation"):
                    decoder_norm = self.decoder_norm() if self.cfg.sparsity_include_decoder_norm else 1.0
                    if sparsity_loss_type == "power":
                        l_s = torch.norm(feature_acts * decoder_norm, p=p, dim=-1)
                    elif sparsity_loss_type == "tanh":
                        l_s = torch.tanh(tanh_stretch_coefficient * feature_acts * decoder_norm).sum(dim=-1)
                    elif sparsity_loss_type == "tanh-quad":
                        approx_frequency = einops.reduce(
                            torch.tanh(tanh_stretch_coefficient * feature_acts * decoder_norm),
                            "... d_sae -> d_sae",
                            "mean",
                        )
                        l_s = (approx_frequency * (1 + approx_frequency / frequency_scale)).sum(dim=-1)
                    else:
                        raise ValueError(f"sparsity_loss_type f{sparsity_loss_type} not supported.")
                    if isinstance(l_s, DTensor):
                        l_s = l_s.full_tensor()
                    l_s = l1_coefficient * l_s
                    # WARNING: Some DTensor bugs make if l1_coefficient * l_s goes before full_tensor, the l1_coefficient value will be internally cached. Furthermore, it will cause the backward pass to fail with redistribution error. See https://github.com/pytorch/pytorch/issues/153603 and https://github.com/pytorch/pytorch/issues/153615 .
                    loss_dict["l_s"] = l_s
                    loss = loss + l_s.mean()

            # Add AuxK auxiliary loss if enabled
            if self.cfg.use_auxk and self.cfg.act_fn == "topk":
                print("======== use auxk loss ========")
                with timer.time("auxk_loss_calculation"):
                    # Get reconstruction error
                    e = label - reconstructed  # (batch, d_model) or (batch, seq_len, d_model)
                    
                    # Get the top-k_aux dead latents based on their activation values
                    current_k = self.current_k
                    if self.device_mesh is not None:
                        self.current_k = min(self.cfg.k_aux, self.is_dead.full_tensor().sum())
                    else:
                        self.current_k = min(self.cfg.k_aux, self.is_dead.sum())
                    
                    if self.current_k > 0:
                        # Scale feature activations by decoder norm if configured
                        if self.cfg.sparsity_include_decoder_norm:
                            dead_sparsity_scores = hidden_pre * self.is_dead * self.decoder_norm()
                        else:
                            dead_sparsity_scores = hidden_pre * self.is_dead

                        dead_activation_mask = self.activation_function(dead_sparsity_scores)
                        dead_feature_acts = torch.clamp(hidden_pre * dead_activation_mask * self.is_dead, min=0.0)
                        
                        # Decode auxiliary feature activations
                        aux_reconstructed = dead_feature_acts @ self.W_D

                        if isinstance(aux_reconstructed, DTensor):
                            aux_reconstructed = DimMap({}).redistribute(aux_reconstructed)
                        
                        l_aux = (e - aux_reconstructed).pow(2).sum(dim=-1)
                    else:
                        l_aux = torch.zeros_like(l_rec)
                    
                    if isinstance(l_aux, DTensor):
                        l_aux = l_aux.full_tensor()
                    loss_dict["l_aux"] = l_aux
                    loss = loss + self.cfg.aux_coefficient * l_aux.mean()
                    
                    self.current_k = current_k
        if return_aux_data:
            aux_data = {
                "feature_acts": feature_acts,
                "reconstructed": reconstructed,
                "hidden_pre": hidden_pre,
            }
            return loss, (loss_dict, aux_data)
        return loss
