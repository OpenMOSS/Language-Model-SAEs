import math
import os
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Literal, Union, cast, overload

import safetensors.torch as safe
import torch
from jaxtyping import Float
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from transformer_lens.hook_points import HookedRootModule, HookPoint

from lm_saes.database import MongoClient
from lm_saes.utils.huggingface import parse_pretrained_name_or_path

from .config import BaseSAEConfig, SAEConfig


class SparseAutoEncoder(HookedRootModule):
    def __init__(self, cfg: BaseSAEConfig):
        super(SparseAutoEncoder, self).__init__()
        self.cfg = cfg
        # should be set by Trainer during training
        self.current_k = cfg.top_k
        # should be initialized by Initializer and set by Trainer during training
        self.current_l1_coefficient = None
        self.encoder = torch.nn.Linear(cfg.d_model, cfg.d_sae, bias=True, device=cfg.device, dtype=cfg.dtype)
        self.decoder = torch.nn.Linear(
            cfg.d_sae, cfg.d_model, bias=cfg.use_decoder_bias, device=cfg.device, dtype=cfg.dtype
        )
        if cfg.use_glu_encoder:
            self.encoder_glu = torch.nn.Linear(cfg.d_model, cfg.d_sae, bias=True, device=cfg.device, dtype=cfg.dtype)
        self.activation_function: Callable[[torch.Tensor], torch.Tensor] = self.activation_function_factory(cfg)
        self.hook_hidden_pre = HookPoint()
        self.hook_feature_acts = HookPoint()
        self.hook_reconstructed = HookPoint()
        # if cfg.norm_activation == "dataset-wise", the dataset average activation norm should be
        # calculated by the initializer before training starts and set by standardize_parameters_of_dataset_activation_scaling
        self.dataset_average_activation_norm: dict[str, float] | None = None
        self.device_mesh: DeviceMesh | None = None

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

    @torch.no_grad()
    def set_current_l1_coefficient(self, current_l1_coefficient: float):
        """Set the current l1 coefficient for the topk activation function.
        This should be set by the Trainer during training.
        """
        self.current_l1_coefficient = current_l1_coefficient

    def _encoder_norm(self, encoder: torch.nn.Linear, keepdim: bool = False):
        """Compute the norm of the encoder weight."""
        if not isinstance(encoder.weight, DTensor):
            return torch.norm(encoder.weight, p=2, dim=1, keepdim=keepdim).to(self.cfg.device)
        else:
            # We suspect that using torch.norm on dtensor may lead to some bugs
            # during the backward process that are difficult to pinpoint and resolve.
            # Therefore, we first convert the decoder weight from dtensor to tensor for norm calculation,
            # and then redistribute it to different nodes.
            assert self.device_mesh is not None
            encoder_norm = torch.norm(encoder.weight.to_local(), p=2, dim=1, keepdim=keepdim)
            encoder_norm = DTensor.from_local(
                encoder_norm, device_mesh=self.device_mesh["model"], placements=[Shard(0)]
            )
            encoder_norm = encoder_norm.redistribute(placements=[Replicate()], async_op=True).to_local()
            return encoder_norm

    def _decoder_norm(self, decoder: torch.nn.Linear, keepdim: bool = False):
        """Compute the norm of the decoder weight."""
        if not isinstance(decoder.weight, DTensor):
            return torch.norm(decoder.weight, p=2, dim=0, keepdim=keepdim).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            decoder_norm = torch.norm(decoder.weight.to_local(), p=2, dim=0, keepdim=keepdim)
            decoder_norm = DTensor.from_local(
                decoder_norm, device_mesh=self.device_mesh["model"], placements=[Shard(int(keepdim))]
            )
            decoder_norm = decoder_norm.redistribute(placements=[Replicate()], async_op=True).to_local()
            return decoder_norm

    def activation_function_factory(self, cfg: BaseSAEConfig) -> Callable[[torch.Tensor], torch.Tensor]:  # type: ignore
        assert cfg.act_fn.lower() in ["relu", "topk", "jumprelu"], f"Not implemented activation function {cfg.act_fn}"
        if cfg.act_fn.lower() == "relu":
            return lambda x: torch.where(x > 0, 1, 0)
        if cfg.act_fn.lower() == "jumprelu":
            return lambda x: torch.where(x > cfg.jump_relu_threshold, 1, 0)
        if cfg.act_fn.lower() == "topk":

            def topk_activation(x: torch.Tensor):
                x = torch.clamp(x, min=0.0)
                k = x.shape[-1] - self.current_k + 1
                k_th_value, _ = torch.kthvalue(x, k=k, dim=-1)
                return torch.where(x >= k_th_value, 1, 0)

            return topk_activation

    def compute_norm_factor(self, x: torch.Tensor, hook_point: str) -> torch.Tensor:  # type: ignore
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

    @torch.no_grad()
    def _set_decoder_to_fixed_norm(self, decoder: torch.nn.Linear, value: float, force_exact: bool):
        """Set the decoder to a fixed norm.
        Args:
            value (float): The target norm value.
            force_exact (bool): If True, the decoder weight will be scaled to exactly match the target norm.
                If False, the decoder weight will be scaled to match the target norm up to a small tolerance.
            device_mesh (DeviceMesh | None): The device mesh to use for distributed training.
        """
        decoder_norm = self._decoder_norm(decoder, keepdim=True)
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
        encoder_norm = self._encoder_norm(encoder, keepdim=True)
        if self.device_mesh:
            # TODO: check if this is correct
            encoder_norm = distribute_tensor(encoder_norm, device_mesh=self.device_mesh["model"], placements=[Shard(0)])
        encoder.weight.data *= value / encoder_norm

    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float):
        self._set_encoder_to_fixed_norm(self.encoder, value)

    @torch.no_grad()
    def _get_full_state_dict(self):  # should be overridden by subclasses
        state_dict = self.state_dict()
        if self.device_mesh and self.device_mesh["model"].size(0) > 1:
            state_dict = {k: v.full_tensor() if isinstance(v, DTensor) else v for k, v in state_dict.items()}

        # If sparsity_include_decoder_norm is False, we need to normalize the decoder weight before saving
        # We use a deepcopy to avoid modifying the original weight to avoid affecting the training progress
        if not self.cfg.sparsity_include_decoder_norm:
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

    @torch.no_grad()
    def save_checkpoint(self, ckpt_path: Path | str) -> None:
        # TODO: save the config to MongoDB
        """
        {
            "name": sae_name,
            "config": sae_config,
            "path": final_ckpt_path,
        }
        """
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "sae_weights.safetensors")
        state_dict = self._get_full_state_dict()
        if self.device_mesh is None or self.device_mesh.get_rank() == 0:
            if Path(ckpt_path).suffix == ".safetensors":
                safe.save_file(state_dict, ckpt_path, {"version": version("lm-saes")})
            elif Path(ckpt_path).suffix == ".pt":
                torch.save({"sae": state_dict, "version": version("lm-saes")}, ckpt_path)
            else:
                raise ValueError(
                    f"Invalid checkpoint path {ckpt_path}. Currently only supports .safetensors and .pt formats."
                )

    @torch.no_grad()
    def save_pretrained(
        self,
        save_path: Path | str,
        sae_name: str | None = None,
        sae_series: str | None = None,
        mongo_client: MongoClient | None = None,
    ) -> None:
        # TODO: save dataset_average_activation_norm
        self.save_checkpoint(save_path)
        if self.device_mesh is None or self.device_mesh.get_rank() == 0:
            self.cfg.save_hyperparameters(save_path)
            if mongo_client is not None:
                assert (
                    sae_name is not None and sae_series is not None
                ), "sae_name and sae_series must be provided when saving to MongoDB"
                mongo_client.create_sae(
                    name=sae_name, series=sae_series, path=str(Path(save_path).absolute()), cfg=self.cfg
                )

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        return_hidden_pre: Literal[False] = False,
    ) -> Union[Float[torch.Tensor, "batch d_sae"], Float[torch.Tensor, "batch seq_len d_sae"]]: ...

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        return_hidden_pre: Literal[True],
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
    ]:  # should be overridden by subclasses
        input_norm_factor = self.compute_norm_factor(x, hook_point=self.cfg.hook_point_in)
        x = x * input_norm_factor
        if self.cfg.use_decoder_bias and self.cfg.apply_decoder_bias_to_pre_encoder:
            # We need to convert decoder bias to a tensor before subtracting
            bias = self.decoder.bias.to_local() if isinstance(self.decoder.bias, DTensor) else self.decoder.bias
            x = x - bias
        hidden_pre = self.encoder(x)
        if self.cfg.use_glu_encoder:
            hidden_pre_glu = torch.sigmoid(self.encoder_glu(x))
            hidden_pre = hidden_pre * hidden_pre_glu

        hidden_pre = self.hook_hidden_pre(hidden_pre)
        if self.cfg.sparsity_include_decoder_norm:
            true_feature_acts = hidden_pre * self._decoder_norm(decoder=self.decoder)
        else:
            true_feature_acts = hidden_pre
        activation_mask = self.activation_function(true_feature_acts)
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
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:  # may be overridden by subclasses
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
        feature_acts = self.encode(x)
        reconstructed = self.decode(feature_acts)
        return reconstructed

    @overload
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_batch_norm_mse: bool = False,
        lp: int = 1,
        return_aux_data: Literal[True] = True,
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
        lp: int = 1,
        return_aux_data: Literal[False],
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
        lp: int = 1,
        return_aux_data: bool = True,
    ) -> Union[
        Float[torch.Tensor, " batch"],
        tuple[
            Float[torch.Tensor, " batch"],
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        ],
    ]:  # may be overridden by subclasses
        x: torch.Tensor = batch[self.cfg.hook_point_in]
        label: torch.Tensor = batch[self.cfg.hook_point_out]
        feature_acts, hidden_pre = self.encode(x, return_hidden_pre=True)
        reconstructed = self.decode(feature_acts)
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
        loss = l_rec.mean()
        loss_dict = {
            "l_rec": l_rec,
        }

        if not self.cfg.act_fn == "topk":
            l_lp = torch.norm(feature_acts, p=lp, dim=-1)
            loss_dict["l_lp"] = l_lp
            assert self.current_l1_coefficient is not None
            loss = loss + self.current_l1_coefficient * l_lp.mean()

        if return_aux_data:
            aux_data = {
                "feature_acts": feature_acts,
                "reconstructed": reconstructed / label_norm_factor,
                "hidden_pre": hidden_pre,
            }
            return loss, (loss_dict, aux_data)
        return loss

    @classmethod
    def from_config(cls, cfg: SAEConfig) -> "SparseAutoEncoder":
        if cfg.sae_pretrained_name_or_path is None:
            return cls(cfg)
        path = parse_pretrained_name_or_path(cfg.sae_pretrained_name_or_path)
        if path.endswith(".pt") or path.endswith(".safetensors"):
            ckpt_path = path
        else:
            ckpt_prioritized_paths = [
                f"{path}/sae_weights.safetensors",
                f"{path}/sae_weights.pt",
                f"{path}/checkpoints/pruned.safetensors",
                f"{path}/checkpoints/pruned.pt",
                f"{path}/checkpoints/final.safetensors",
                f"{path}/checkpoints/final.pt",
            ]
            for ckpt_path in ckpt_prioritized_paths:
                if os.path.exists(ckpt_path):
                    break
            else:
                raise FileNotFoundError(f"Pretrained model not found at {cfg.sae_pretrained_name_or_path}")

        if ckpt_path.endswith(".safetensors"):
            state_dict = safe.load_file(ckpt_path, device=cfg.device)
        else:
            state_dict = torch.load(ckpt_path, map_location=cfg.device)["sae"]

        model = cls(cfg)
        model.load_state_dict(state_dict, strict=cfg.strict_loading)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, **kwargs):
        cfg = SAEConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        return cls.from_config(cfg)

    @torch.no_grad()
    def log_statistics(self):
        log_dict = {
            "metrics/encoder_norm": self._encoder_norm(self.encoder).mean().item(),
            "metrics/encoder_bias_norm": self.encoder.bias.norm().item(),
            "metrics/decoder_norm": self._decoder_norm(self.decoder).mean().item(),
        }
        if self.cfg.use_decoder_bias:
            log_dict["metrics/decoder_bias_norm"] = self.decoder.bias.norm().item()
        if self.cfg.act_fn == "topk":
            log_dict["sparsity/k"] = self.current_k
        else:
            log_dict["sparsity/l1_coefficient"] = self.current_l1_coefficient
        return log_dict

    @torch.no_grad()
    def _init_encoder_with_decoder_transpose(self, encoder: torch.nn.Linear, decoder: torch.nn.Linear):
        encoder.weight.data = decoder.weight.data.T.clone().contiguous()

    @torch.no_grad()
    def init_encoder_with_decoder_transpose(self):
        self._init_encoder_with_decoder_transpose(self.encoder, self.decoder)
