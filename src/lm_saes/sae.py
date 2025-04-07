import math
from typing import Literal, Union, cast, overload, override

import torch
from jaxtyping import Float
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from transformer_lens.hook_points import HookPoint

from .abstract_sae import AbstractSparseAutoEncoder
from .config import BaseSAEConfig, SAEConfig
from .kernels import decode_with_triton_spmm_kernel


class SparseAutoEncoder(AbstractSparseAutoEncoder):
    def __init__(self, cfg: BaseSAEConfig):
        super(SparseAutoEncoder, self).__init__(cfg)
        self.cfg = cfg
        # should be set by Trainer during training
        self.current_k = cfg.top_k
        self.encoder = torch.nn.Linear(cfg.d_model, cfg.d_sae, bias=True, device=cfg.device, dtype=cfg.dtype)
        self.decoder = torch.nn.Linear(
            cfg.d_sae, cfg.d_model, bias=cfg.use_decoder_bias, device=cfg.device, dtype=cfg.dtype
        )
        if cfg.use_glu_encoder:
            self.encoder_glu = torch.nn.Linear(cfg.d_model, cfg.d_sae, bias=True, device=cfg.device, dtype=cfg.dtype)
        self.hook_hidden_pre = HookPoint()
        self.hook_feature_acts = HookPoint()
        self.hook_reconstructed = HookPoint()
        # if cfg.norm_activation == "dataset-wise", the dataset average activation norm should be
        # calculated by the initializer before training starts and set by standardize_parameters_of_dataset_activation_scaling
        self.dataset_average_activation_norm: dict[str, float] | None = None

    @torch.no_grad()
    def set_dataset_average_activation_norm(self, dataset_average_activation_norm: dict[str, float]):
        """Set the dataset average activation norm for training or inference.
        dataset_average_activation_norm is set by the Initializer and only used when cfg.norm_activation == "dataset-wise".
        """
        self.dataset_average_activation_norm = dataset_average_activation_norm

    @torch.no_grad()
    def set_current_k(self, current_k: int):
        """Set the current k for topk activation function.
        This should be set by the Trainer during training.
        """
        self.current_k = current_k

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

    def compute_norm_factor(self, x: torch.Tensor, hook_point: str) -> torch.Tensor:
        """Compute the normalization factor for the activation vectors.
        This should be called during forward pass.
        There are four modes for norm_activation:
        - "token-wise": normalize by the token-wise norm calculated with the input x
        - "batch-wise": normalize by the batch-wise norm calculated with the input x
        - "dataset-wise": normalize by the dataset-wise norm initialized by the Initializer with sae.set_dataset_average_activation_norm method
        - "inference": no normalization, the weights are already normalized by the Initializer with sae.standardize_parameters_of_dataset_norm method
        """
        assert self.cfg.norm_activation in [
            "token-wise",
            "batch-wise",
            "dataset-wise",
            "inference",
        ], f"Not implemented norm_activation {self.cfg.norm_activation}"
        if self.cfg.norm_activation == "token-wise":
            return math.sqrt(self.cfg.d_model) / torch.norm(x, 2, dim=-1, keepdim=True)
        if self.cfg.norm_activation == "batch-wise":
            return math.sqrt(self.cfg.d_model) / torch.norm(x, 2, dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        if self.cfg.norm_activation == "dataset-wise":
            assert (
                self.dataset_average_activation_norm is not None
            ), "dataset_average_activation_norm must be provided from Initializer"
            return torch.tensor(
                math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[hook_point],
                device=x.device,
                dtype=x.dtype,
            )
        if self.cfg.norm_activation == "inference":
            return torch.tensor(1.0, device=x.device, dtype=x.dtype)
        raise ValueError(f"Not implemented norm_activation {self.cfg.norm_activation}")

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

    @torch.no_grad()
    def full_state_dict(self):  # should be overridden by subclasses
        state_dict = self.state_dict()
        if self.device_mesh and self.device_mesh["model"].size(0) > 1:
            state_dict = {k: v.full_tensor() if isinstance(v, DTensor) else v for k, v in state_dict.items()}

        # Add dataset_average_activation_norm to state dict
        if self.dataset_average_activation_norm is not None:
            for hook_point, value in self.dataset_average_activation_norm.items():
                state_dict[f"dataset_average_activation_norm.{hook_point}"] = torch.tensor(value)

        # If force_unit_decoder_norm is True, we need to normalize the decoder weight before saving
        # We use a deepcopy to avoid modifying the original weight to avoid affecting the training progress
        if self.cfg.force_unit_decoder_norm:
            state_dict["decoder.weight"] = self.decoder.weight.data.clone()  # deepcopy
            decoder_norm = torch.norm(state_dict["decoder.weight"], p=2, dim=0, keepdim=True)
            state_dict["decoder.weight"] = state_dict["decoder.weight"] / decoder_norm

        return cast(dict[str, torch.Tensor], state_dict)

    @staticmethod
    @torch.no_grad()
    def _transform_to_unit_decoder_norm(encoder: torch.nn.Linear, decoder: torch.nn.Linear):
        decoder_weight = decoder.weight.to_local() if isinstance(decoder.weight, DTensor) else decoder.weight
        decoder_norm = torch.norm(decoder_weight, p=2, dim=0, keepdim=False)
        decoder.weight.data = decoder.weight.data / decoder_norm
        encoder.weight.data = encoder.weight.data * decoder_norm[:, None]
        encoder.bias.data = encoder.bias.data * decoder_norm

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
        # Apply input normalization based on config
        input_norm_factor = self.compute_norm_factor(x, hook_point=self.cfg.hook_point_in)
        x = x * input_norm_factor
        # Optionally subtract decoder bias before encoding
        if self.cfg.use_decoder_bias and self.cfg.apply_decoder_bias_to_pre_encoder:
            # We need to convert decoder bias to a tensor before subtracting
            bias = self.decoder.bias.to_local() if isinstance(self.decoder.bias, DTensor) else self.decoder.bias
            x = x - bias

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
        feature_acts = hidden_pre * activation_mask
        feature_acts = self.hook_feature_acts(feature_acts)

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

    @overload
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_batch_norm_mse: bool = False,
        sparsity_loss_type: Literal["power", "tanh", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        p: int = 1,
        l1_coefficient: float = 1.0,
        return_aux_data: Literal[True] = True,
        **kwargs,
    ) -> tuple[
        Float[torch.Tensor, " batch"],
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    ]: ...

    @overload
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_batch_norm_mse: bool = False,
        sparsity_loss_type: Literal["power", "tanh", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        p: int = 1,
        l1_coefficient: float = 1.0,
        return_aux_data: Literal[False],
        **kwargs,
    ) -> Float[torch.Tensor, " batch"]: ...

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
        sparsity_loss_type: Literal["power", "tanh", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        p: int = 1,
        l1_coefficient: float = 1.0,
        return_aux_data: bool = True,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, " batch"],
        tuple[
            Float[torch.Tensor, " batch"],
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        ],
    ]:  # may be overridden by subclasses
        x: torch.Tensor = batch[self.cfg.hook_point_in].to(self.cfg.dtype)
        label: torch.Tensor = batch[self.cfg.hook_point_out].to(self.cfg.dtype)
        feature_acts, hidden_pre = self.encode(x, return_hidden_pre=True, **kwargs)
        reconstructed = self.decode(feature_acts, **kwargs)
        label_norm_factor: torch.Tensor = self.compute_norm_factor(label, hook_point=self.cfg.hook_point_out)
        label_normed = label * label_norm_factor
        l_rec = (reconstructed - label_normed).pow(2)
        if use_batch_norm_mse:
            l_rec = (
                l_rec
                / (label_normed - label_normed.mean(dim=0, keepdim=True))
                .pow(2)
                .sum(dim=-1, keepdim=True)
                .clamp(min=1e-8)
                .sqrt()
            )
        loss = l_rec.sum(dim=-1).mean()
        loss_dict = {
            "l_rec": l_rec,
        }

        if sparsity_loss_type == "power":
            l_s = torch.norm(feature_acts * self.decoder_norm(), p=p, dim=-1)
            loss_dict["l_s"] = l1_coefficient * l_s.mean()
            loss = loss + l1_coefficient * l_s.mean()
        elif sparsity_loss_type == "tanh":
            l_s = torch.tanh(tanh_stretch_coefficient * feature_acts * self.decoder_norm()).sum(dim=-1)
            loss_dict["l_s"] = l1_coefficient * l_s.mean()
            loss = loss + l1_coefficient * l_s.mean()
        elif sparsity_loss_type is None:
            pass
        else:
            raise ValueError(f"sparsity_loss_type f{sparsity_loss_type} not supported.")

        if return_aux_data:
            aux_data = {
                "feature_acts": feature_acts,
                "reconstructed": reconstructed / label_norm_factor,
                "hidden_pre": hidden_pre,
            }
            return loss, (loss_dict, aux_data)
        return loss

    def load_full_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        # Extract and set dataset_average_activation_norm if present
        norm_keys = [k for k in state_dict.keys() if k.startswith("dataset_average_activation_norm.")]
        if norm_keys:
            dataset_norm = {key.split(".", 1)[1]: state_dict[key].item() for key in norm_keys}
            self.set_dataset_average_activation_norm(dataset_norm)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dataset_average_activation_norm.")}
        self.load_state_dict(state_dict, strict=self.cfg.strict_loading)

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, **kwargs):
        cfg = SAEConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        return cls.from_config(cfg)

    @torch.no_grad()
    def _init_encoder_with_decoder_transpose(
        self, encoder: torch.nn.Linear, decoder: torch.nn.Linear, factor: float = 1.0
    ):
        encoder.weight.data = decoder.weight.data.T.clone().contiguous() * factor

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
