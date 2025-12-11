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
import torch.nn.functional as F

from lm_saes.utils.distributed import DimMap
from lm_saes.utils.logging import get_distributed_logger

from .abstract_sae import AbstractSparseAutoEncoder
from .config import CNNSAEConfig

from einops import einsum, rearrange

logger = get_distributed_logger("cnnsae")

def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))

class CNNSparseAutoEncoder(AbstractSparseAutoEncoder):
    def __init__(self, cfg: CNNSAEConfig, device_mesh: DeviceMesh | None = None):
        super(CNNSparseAutoEncoder, self).__init__(cfg, device_mesh=device_mesh)
        assert device_mesh is None
        self.cfg = cfg
                
        self.W_conv = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_model, 7, 7, device=cfg.device, dtype=cfg.dtype))
        self.b_conv = nn.Parameter(torch.zeros(cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
        self.layer_norm_weight = nn.Parameter(torch.ones(cfg.d_model, device=cfg.device, dtype=cfg.dtype))
        self.layer_norm_bias = nn.Parameter(torch.zeros(cfg.d_model, device=cfg.device, dtype=cfg.dtype))
        # self.W_E = nn.Parameter(torch.empty(cfg.d_model, cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
        # self.b_E = nn.Parameter(torch.empty(cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
        
        self.W_D = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_model, device=cfg.device, dtype=cfg.dtype))
        if cfg.use_decoder_bias:
            self.b_D = nn.Parameter(torch.empty(cfg.d_model, device=cfg.device, dtype=cfg.dtype))

        self.hook_hidden_pre = HookPoint()
        self.hook_feature_acts = HookPoint()
        self.hook_reconstructed = HookPoint()

    @override
    def encoder_norm(self, keepdim: bool = False):
        """Compute the norm of the encoder weight."""
        return torch.norm(self.W_conv, p=2, dim=(1,2,3), keepdim=keepdim)
        # return torch.norm(self.W_E, p=2, dim=1, keepdim=keepdim).to(self.cfg.device)
    
    # def encoder_norm2(self, keepdim: bool = False):
    #     """Compute the norm of the encoder weight."""
    #     return torch.norm(self.W_E2, p=2, dim=(1,2,3), keepdim=keepdim).to(self.cfg.device)

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
        # self.W_E.mul_(value / self.encoder_norm(keepdim=True))
        self.W_conv.mul_(value/self.encoder_norm(keepdim=True))
        # self.W_E2.mul_(value/self.encoder_norm2(keepdim=True))

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
        return parent_maps | sae_maps

    @override
    @torch.no_grad()
    def transform_to_unit_decoder_norm(self):
        self.W_D.mul_(1 / self.decoder_norm(keepdim=False))

    @torch.no_grad()
    def standardize_parameters_of_dataset_norm(self):  # should be overridden by subclasses due to side effects
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
        assert self.dataset_average_activation_norm is not None
        input_norm_factor: float = (
            math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_in]
        )
        output_norm_factor: float = (
            math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_out]
        )
        # self.b_E.div_(input_norm_factor)
        self.b_conv.div_(input_norm_factor)
        if self.cfg.use_decoder_bias:
            assert self.b_D is not None, "Decoder bias should exist if use_decoder_bias is True"
            self.b_D.div_(output_norm_factor)
        self.W_D.mul_(input_norm_factor / output_norm_factor)
        self.cfg.norm_activation = "inference"

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
            Float[torch.Tensor, "batch d_model H W"],
        ],
        return_hidden_pre: bool = False,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch d_sae H W"],
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
        # hidden_pre = x @ self.W_E + self.b_E
        # print("input", x.shape)
        b, S, dm = x.shape
        h = int(math.sqrt(S))
        x = rearrange(x, "b (h w) d -> b d h w", h=h)
        # hidden_pre = self.Encoder(x)
        hidden_pre = F.conv2d(x, self.W_conv, self.b_conv, padding=3)
        # hidden_pre_u = hidden_pre.mean(dim=1, keepdim=True)
        # hidden_pre_s = (hidden_pre - hidden_pre_u).pow(2).mean(dim=1, keepdim=True)
        # hidden_pre = (hidden_pre-hidden_pre_u) / torch.sqrt(hidden_pre_s + 1e-6)
        # hidden_pre = F.conv2d(hidden_pre, self.W_E2, self.b_E2, padding=2)
        # hidden_pre = einsum(hidden_pre, self.W_E, "b c h w, c d -> b d h w") + self.b_E.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # hidden_pre = hidden_pre.reshape(b,ds,-1).permute(0,2,1)
        hidden_pre = rearrange(hidden_pre, "b d h w -> b (h w) d")

        # Apply GLU if configured
        # if self.cfg.use_glu_encoder:
        #     hidden_pre_glu = torch.sigmoid(x @ self.W_E_glu + self.b_E_glu)
        #     hidden_pre = hidden_pre * hidden_pre_glu

        hidden_pre = self.hook_hidden_pre(hidden_pre)

        # Scale feature activations by decoder norm if configured
        if self.cfg.sparsity_include_decoder_norm:
            # print("hidden_pre", hidden_pre.shape, "decoder_norm", self.decoder_norm().shape)
            hidden_pre = hidden_pre * self.decoder_norm()

        feature_acts = self.activation_function(hidden_pre)
        feature_acts = self.hook_feature_acts(feature_acts)

        if self.cfg.sparsity_include_decoder_norm:
            feature_acts = feature_acts / self.decoder_norm()
            hidden_pre = hidden_pre / self.decoder_norm()

        if return_hidden_pre:
            return feature_acts, hidden_pre
        return feature_acts

    def decode(
        self,
        feature_acts: Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch d_sae H W"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch d_model H W"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:  # may be overridden by subclasses
        max_l0_in_batch = feature_acts.gt(0).to(feature_acts).sum(dim=-1).max()
        sparsity_threshold = self.cfg.d_sae * (1 - self.cfg.sparsity_threshold_for_triton_spmm_kernel)
        if (
            self.cfg.use_triton_kernel and 0 < max_l0_in_batch < sparsity_threshold
        ):  # triton kernel cannot handle empty feature_acts
            from .kernels import decode_with_triton_spmm_kernel

            reconstructed = decode_with_triton_spmm_kernel(feature_acts, self.W_D.T.contiguous())
        else:
            reconstructed = feature_acts @ self.W_D

        assert reconstructed is not None, "Reconstructed cannot be None"
        if self.cfg.use_decoder_bias:
            reconstructed = reconstructed + self.b_D# .unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        reconstructed = self.hook_reconstructed(reconstructed)

        if isinstance(reconstructed, DTensor):
            reconstructed = DimMap({"data": 0}).redistribute(reconstructed)

        # b, c, h, w = reconstructed.shape
        # reconstructed = reconstructed.reshape(b, c, -1).permute(0,2,1)
        
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
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, **kwargs):
        cfg = CNNSAEConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        return cls.from_config(cfg)

    @torch.no_grad()
    def _init_encoder_with_decoder_transpose(
        self, encoder: torch.nn.Linear, decoder: torch.nn.Linear, factor: float = 1.0
    ):
        encoder.weight.data = decoder.weight.data.T.clone().contiguous() * factor

    @override
    @torch.no_grad()
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0):
        RuntimeError("Not support for cnnsae")

    @override
    @torch.no_grad()
    def init_parameters(self, **kwargs):
        super().init_parameters(**kwargs)

        W_conv = torch.empty(self.cfg.d_sae, self.cfg.d_model, 7, 7, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(-(self.cfg.d_sae)**(-0.5), (self.cfg.d_sae)**(-0.5))
        # W_E = torch.empty(self.cfg.d_model, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(
        #     -kwargs["encoder_uniform_bound"], kwargs["encoder_uniform_bound"]
        # )
        # W_E2 = torch.empty(self.cfg.d_sae, self.cfg.d_model*4, 5, 5, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(
        #     -kwargs["encoder_uniform_bound"], kwargs["encoder_uniform_bound"]
        # )
        W_D = torch.empty(self.cfg.d_sae, self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(
            -kwargs["decoder_uniform_bound"], kwargs["decoder_uniform_bound"]
        )
        # b_E = torch.zeros(self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype)
        # b_E2 = torch.zeros(self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype)

        # if self.device_mesh is not None:
        #     W_E = self.dim_maps()["W_E"].distribute(W_E, self.device_mesh)
        #     W_D = self.dim_maps()["W_D"].distribute(W_D, self.device_mesh)
        #     b_E = self.dim_maps()["b_E"].distribute(b_E, self.device_mesh)

        self.W_conv.copy_(W_conv)
        # self.W_E.copy_(W_E)
        # self.W_E2.copy_(W_E2)
        self.W_D.copy_(W_D)
        # self.b_E.copy_(b_E)
        # self.b_E2.copy_(b_E2)

        if self.cfg.use_decoder_bias:
            b_D = torch.zeros(self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype)

            if self.device_mesh is not None:
                b_D = self.dim_maps()["b_D"].distribute(b_D, self.device_mesh)

            self.b_D.copy_(b_D)


    @override
    def prepare_input(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        x = batch[self.cfg.hook_point_in]
        return x, {}, {}

    @override
    def prepare_label(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        label = batch[self.cfg.hook_point_out]
        return label

    @override
    @torch.no_grad()
    def init_W_D_with_active_subspace(self, activation_batch: dict[str, torch.Tensor], d_active_subspace: int):
        """Initialize W_D with the active subspace.

        Args:
            activation_batch: The activation batch.
            d_active_subspace: The dimension of the active subspace.
        """
        label = self.prepare_label(activation_batch)
        demeaned_label = label - label.mean(dim=0)
        U, S, V = torch.svd(demeaned_label.T.to(torch.float32))
        proj_weight = U[:, :d_active_subspace]  # [d_model, d_active_subspace]
        self.W_D.copy_(self.W_D.data[:, :d_active_subspace] @ proj_weight.T.to(self.cfg.dtype))

    @torch.no_grad()
    def init_encoder_bias_with_mean_hidden_pre(self, activation_batch: dict[str, torch.Tensor]):
        x = self.prepare_input(activation_batch)[0]
        _, hidden_pre = self.encode(x, return_hidden_pre=True)
        self.b_E.copy_(-hidden_pre.mean(dim=0))
    
    @torch.no_grad()
    def compute_activation_frequency_scores(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Compute activation frequency scores for feature sparsity tracking.

        Args:
            feature_acts: Feature activations tensor

        Returns:
            Activation frequency scores tensor, aggregated appropriately for the model type.
            Default implementation returns sum over batch dimension.
        """
        return (feature_acts > 0).float().sum(dim=(0,1))