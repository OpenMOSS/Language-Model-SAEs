import math
from typing import Any, Literal, Union, overload

import torch
from jaxtyping import Float
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from transformer_lens.hook_points import HookPoint
from typing_extensions import override

from .abstract_sae import AbstractSparseAutoEncoder
from .config import SAEConfig


class SparseAutoEncoder(AbstractSparseAutoEncoder):
    def __init__(self, cfg: SAEConfig, device_mesh: DeviceMesh | None = None):
        super(SparseAutoEncoder, self).__init__(cfg)
        self.cfg = cfg

        assert device_mesh is None, "SparseAutoEncoder currently does not support multi-GPU."

        self.encoder = torch.nn.Linear(cfg.d_model, cfg.d_sae, bias=True, device=cfg.device, dtype=cfg.dtype)
        self.decoder = torch.nn.Linear(
            cfg.d_sae, cfg.d_model, bias=cfg.use_decoder_bias, device=cfg.device, dtype=cfg.dtype
        )
        if cfg.use_glu_encoder:
            self.encoder_glu = torch.nn.Linear(cfg.d_model, cfg.d_sae, bias=True, device=cfg.device, dtype=cfg.dtype)

        self.hook_hidden_pre = HookPoint()
        self.hook_feature_acts = HookPoint()
        self.hook_reconstructed = HookPoint()

    @override
    def encoder_norm(self, keepdim: bool = False):
        """Compute the norm of the encoder weight."""
        if not isinstance(self.encoder.weight, DTensor):
            return torch.norm(self.encoder.weight, p=2, dim=1, keepdim=keepdim).to(self.cfg.device)
        else:
            # We suspect that using torch.norm on dtensor may lead to some bugs
            # during the backward process that are difficult to pinpoint and resolve.
            # Therefore, we first convert the decoder weight from dtensor to tensor for norm calculation,
            # and then redistribute it to different nodes.
            assert self.device_mesh is not None
            encoder_norm = torch.norm(self.encoder.weight.to_local(), p=2, dim=1, keepdim=keepdim)
            encoder_norm = DTensor.from_local(
                encoder_norm, device_mesh=self.device_mesh["model"], placements=[Shard(0)]
            )
            encoder_norm = encoder_norm.redistribute(placements=[Replicate()], async_op=True).to_local()
            return encoder_norm

    @override
    def decoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the norm of the decoder weight."""
        if not isinstance(self.decoder.weight, DTensor):
            return torch.norm(self.decoder.weight, p=2, dim=0, keepdim=keepdim).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            decoder_norm = torch.norm(self.decoder.weight.to_local(), p=2, dim=0, keepdim=keepdim)
            decoder_norm = DTensor.from_local(
                decoder_norm, device_mesh=self.device_mesh["model"], placements=[Shard(int(keepdim))]
            )
            decoder_norm = decoder_norm.redistribute(placements=[Replicate()], async_op=True).to_local()
            return decoder_norm

    @override
    def decoder_bias_norm(self) -> torch.Tensor:
        if not self.cfg.use_decoder_bias:
            raise ValueError("Decoder bias is not used")
        if not isinstance(self.decoder.bias, DTensor):
            return torch.norm(self.decoder.bias, p=2, dim=0, keepdim=True).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            decoder_bias_norm = torch.norm(self.decoder.bias.to_local(), p=2, dim=0, keepdim=True)
            decoder_bias_norm = DTensor.from_local(
                decoder_bias_norm, device_mesh=self.device_mesh["model"], placements=[Shard(0)]
            )
            decoder_bias_norm = decoder_bias_norm.redistribute(placements=[Replicate()], async_op=True).to_local()
            return decoder_bias_norm

    @torch.no_grad()
    def _set_decoder_to_fixed_norm(self, decoder: torch.nn.Linear, value: float, force_exact: bool):
        """Set the decoder to a fixed norm.
        Args:
            value (float): The target norm value.
            force_exact (bool): If True, the decoder weight will be scaled to exactly match the target norm.
                If False, the decoder weight will be scaled to match the target norm up to a small tolerance.
            device_mesh (DeviceMesh | None): The device mesh to use for distributed training.
        """
        decoder_norm = self.decoder_norm(keepdim=True)
        if self.device_mesh:
            # TODO: check if this is correct
            # guess that norm should be distributed as the decoder weight
            decoder_norm = distribute_tensor(decoder_norm, device_mesh=self.device_mesh["model"], placements=[Shard(0)])
        if force_exact:
            decoder.weight.data *= value / decoder_norm
        else:
            decoder.weight.data *= value / torch.clamp(decoder_norm, min=value)

    @override
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool):
        self._set_decoder_to_fixed_norm(self.decoder, value, force_exact)

    @torch.no_grad()
    def _set_encoder_to_fixed_norm(self, encoder: torch.nn.Linear, value: float):
        """Set the encoder to a fixed norm.
        Args:
            value (float): The target norm value.
            device_mesh (DeviceMesh | None): The device mesh to use for distributed training.
        """
        assert not self.cfg.use_glu_encoder, "GLU encoder not supported"
        encoder_norm = self.encoder_norm(keepdim=True)
        if self.device_mesh:
            # TODO: check if this is correct
            encoder_norm = distribute_tensor(encoder_norm, device_mesh=self.device_mesh["model"], placements=[Shard(0)])
        encoder.weight.data *= value / encoder_norm

    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float):
        self._set_encoder_to_fixed_norm(self.encoder, value)

    @override
    @torch.no_grad()
    def full_state_dict(self):  # should be overridden by subclasses
        state_dict = super().full_state_dict()

        # If force_unit_decoder_norm is True, we need to normalize the decoder weight before saving
        # We use a deepcopy to avoid modifying the original weight to avoid affecting the training progress
        if self.cfg.force_unit_decoder_norm:
            state_dict["decoder.weight"] = self.decoder.weight.data.clone()  # deepcopy
            decoder_norm = torch.norm(state_dict["decoder.weight"], p=2, dim=0, keepdim=True)
            state_dict["decoder.weight"] = state_dict["decoder.weight"] / decoder_norm

        return state_dict

    @staticmethod
    @torch.no_grad()
    def _transform_to_unit_decoder_norm(encoder: torch.nn.Linear, decoder: torch.nn.Linear):
        decoder_weight = decoder.weight.to_local() if isinstance(decoder.weight, DTensor) else decoder.weight
        decoder_norm = torch.norm(decoder_weight, p=2, dim=0, keepdim=False)
        decoder.weight.data = decoder.weight.data / decoder_norm
        encoder.weight.data = encoder.weight.data * decoder_norm[:, None]
        encoder.bias.data = encoder.bias.data * decoder_norm

    @override
    @torch.no_grad()
    def transform_to_unit_decoder_norm(self):
        self._transform_to_unit_decoder_norm(self.encoder, self.decoder)

    @torch.no_grad()
    def standardize_parameters_of_dataset_norm(
        self, dataset_average_activation_norm: dict[str, float] | None
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
        self.encoder.bias.data = self.encoder.bias.data / input_norm_factor
        if self.cfg.use_decoder_bias:
            self.decoder.bias.data = self.decoder.bias.data / output_norm_factor
        self.decoder.weight.data = self.decoder.weight.data * input_norm_factor / output_norm_factor
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
        # Optionally subtract decoder bias before encoding
        if self.cfg.use_decoder_bias and self.cfg.apply_decoder_bias_to_pre_encoder:
            x = x - self.decoder.bias

        # Pass through encoder
        hidden_pre = self.encoder(x)
        # Apply GLU if configured
        if self.cfg.use_glu_encoder:
            hidden_pre_glu = torch.sigmoid(self.encoder_glu(x))
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
        max_l0_in_batch = feature_acts.gt(0).to(feature_acts).sum(dim=-1).max()
        sparsity_threshold = self.cfg.d_sae * (1 - self.cfg.sparsity_threshold_for_triton_spmm_kernel)
        if (
            self.cfg.use_triton_kernel and 0 < max_l0_in_batch < sparsity_threshold
        ):  # triton kernel cannot handle empty feature_acts
            from .kernels import decode_with_triton_spmm_kernel

            require_precise_feature_acts_grad = "topk" not in self.cfg.act_fn
            reconstructed = decode_with_triton_spmm_kernel(
                feature_acts, self.decoder.weight, require_precise_feature_acts_grad
            )
        else:
            reconstructed = self.decoder(feature_acts)
        reconstructed = self.hook_reconstructed(reconstructed)

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
        cfg = SAEConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        return cls.from_config(cfg)

    @torch.no_grad()
    def _init_encoder_with_decoder_transpose(
        self, encoder: torch.nn.Linear, decoder: torch.nn.Linear, factor: float = 1.0
    ):
        encoder.weight.data = decoder.weight.data.T.clone().contiguous() * factor

    @override
    @torch.no_grad()
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0):
        self._init_encoder_with_decoder_transpose(self.encoder, self.decoder, factor)

    @override
    @torch.no_grad()
    def init_parameters(self, **kwargs):
        super().init_parameters(**kwargs)
        torch.nn.init.uniform_(
            self.encoder.weight,
            a=-kwargs["encoder_uniform_bound"],
            b=kwargs["encoder_uniform_bound"],
        )
        torch.nn.init.uniform_(
            self.decoder.weight,
            a=-kwargs["decoder_uniform_bound"],
            b=kwargs["decoder_uniform_bound"],
        )
        torch.nn.init.zeros_(self.encoder.bias)
        if self.cfg.use_decoder_bias:
            torch.nn.init.zeros_(self.decoder.bias)
        if self.cfg.use_glu_encoder:
            torch.nn.init.kaiming_uniform_(self.encoder_glu.weight)
            torch.nn.init.zeros_(self.encoder_glu.bias)

    @override
    def prepare_input(self, batch: dict[str, torch.Tensor], **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        return batch[self.cfg.hook_point_in], {}

    @override
    def prepare_label(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        return batch[self.cfg.hook_point_out]

    def get_parameters(self) -> list[dict[str, Any]]:
        return [
            # all parameters but decoder weight, use filter to avoid modifying the decoder weight
            {
                "params": [
                    param for name, param in self.named_parameters() if "decoder" not in name or "weight" not in name
                ],
                "name": "exclude_decoder_weight",
            },
            {
                "params": [self.decoder.weight],
                "name": "decoder_weight",
            },
        ]
