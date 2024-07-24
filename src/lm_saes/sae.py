from importlib.metadata import version
import os
from typing import Dict, Literal, Union, overload, List
import torch
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.device_mesh import init_device_mesh
import torch.nn as nn
import math
from einops import einsum
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint, HookedRootModule

import safetensors.torch as safe

from lm_saes.config import SAEConfig, LanguageModelSAETrainingConfig
from lm_saes.activation.activation_store import ActivationStore
from lm_saes.utils.huggingface import parse_pretrained_name_or_path
import torch.distributed._functional_collectives as funcol
from torch.distributed._tensor import DTensor
import torch.distributed as dist
from torch.distributed._tensor import (
    DTensor,
    Shard,
    Replicate,
    distribute_module,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
    loss_parallel,
)

from lm_saes.utils.misc import is_master


class SparseAutoEncoder(HookedRootModule):
    """Sparse AutoEncoder model.

    An autoencoder model that learns to compress the input activation tensor into a high-dimensional but sparse feature activation tensor.

    Can also act as a transcoder model, which learns to compress the input activation tensor into a feature activation tensor, and then reconstruct a label activation tensor from the feature activation tensor.
    """

    def __init__(self, cfg: SAEConfig):
        """Initialize the SparseAutoEncoder model.

        Args:
            cfg (SAEConfig): The configuration of the model.
        """

        super(SparseAutoEncoder, self).__init__()

        self.cfg = cfg
        self.current_l1_coefficient = cfg.l1_coefficient

        self.encoder = torch.nn.Linear(
            cfg.d_model, cfg.d_sae, bias=True, device=cfg.device, dtype=cfg.dtype
        )
        torch.nn.init.kaiming_uniform_(self.encoder.weight)
        torch.nn.init.zeros_(self.encoder.bias)
        self.device_mesh = init_device_mesh(
            "cuda", (cfg.ddp_size, cfg.tp_size), mesh_dim_names=("ddp", "tp")
        )

        if cfg.use_glu_encoder:

            self.encoder_glu = torch.nn.Linear(
                cfg.d_model, cfg.d_sae, bias=True, device=cfg.device, dtype=cfg.dtype
            )
            torch.nn.init.kaiming_uniform_(self.encoder_glu.weight)
            torch.nn.init.zeros_(self.encoder_glu.bias)

        self.feature_act_mask = torch.nn.Parameter(
            torch.ones((cfg.d_sae,), dtype=cfg.dtype, device=cfg.device)
        )
        self.feature_act_scale = torch.nn.Parameter(
            torch.ones((cfg.d_sae,), dtype=cfg.dtype, device=cfg.device)
        )

        self.decoder = torch.nn.Linear(
            cfg.d_sae,
            cfg.d_model,
            bias=cfg.use_decoder_bias,
            device=cfg.device,
            dtype=cfg.dtype,
        )
        torch.nn.init.kaiming_uniform_(self.decoder.weight)
        self.set_decoder_norm_to_fixed_norm(during_init=True)

        self.train_base_parameters()

        self.hook_hidden_pre = HookPoint()
        self.hook_feature_acts = HookPoint()
        self.hook_reconstructed = HookPoint()

        self.initialize_parameters()

    def initialize_parameters(self):
        torch.nn.init.kaiming_uniform_(self.encoder.weight)

        if self.cfg.use_glu_encoder:
            torch.nn.init.kaiming_uniform_(self.encoder_glu.weight)
            torch.nn.init.zeros_(self.encoder_glu.bias)

        torch.nn.init.kaiming_uniform_(self.decoder.weight)
        self.set_decoder_norm_to_fixed_norm(
            self.cfg.init_decoder_norm, force_exact=True, during_init=True
        )

        if self.cfg.use_decoder_bias:
            torch.nn.init.zeros_(self.decoder.bias)
        torch.nn.init.zeros_(self.encoder.bias)

        if self.cfg.init_encoder_with_decoder_transpose:
            self.encoder.weight.data = self.decoder.weight.data.T.clone().contiguous()
        else:
            self.set_encoder_norm_to_fixed_norm(self.cfg.init_encoder_norm)

    def train_base_parameters(self):
        """Set the base parameters to be trained."""

        base_parameters = [
            self.encoder.weight,
            self.decoder.weight,
            self.encoder.bias,
        ]
        if self.cfg.use_glu_encoder:
            base_parameters.extend([self.encoder_glu.weight, self.encoder_glu.bias])
        if self.cfg.use_decoder_bias:
            base_parameters.append(self.decoder.bias)
        for p in self.parameters():
            p.requires_grad_(False)
        for p in base_parameters:
            p.requires_grad_(True)

    def train_finetune_for_suppression_parameters(self):
        """Set the parameters to be trained for feature suppression."""

        finetune_for_suppression_parameters = [
            self.feature_act_scale,
            self.decoder.weight,
        ]
        if self.cfg.use_decoder_bias:
            finetune_for_suppression_parameters.append(self.decoder.bias)
        for p in self.parameters():
            p.requires_grad_(False)
        for p in finetune_for_suppression_parameters:
            p.requires_grad_(True)

    def compute_norm_factor(
        self, x: torch.Tensor, hook_point: str
    ) -> float | torch.Tensor:
        """Compute the normalization factor for the activation vectors."""

        # Normalize the activation vectors to have L2 norm equal to sqrt(d_model)
        if self.cfg.norm_activation == "token-wise":
            return math.sqrt(self.cfg.d_model) / torch.norm(x, 2, dim=-1, keepdim=True)
        elif self.cfg.norm_activation == "batch-wise":
            return math.sqrt(self.cfg.d_model) / torch.norm(
                x, 2, dim=-1, keepdim=True
            ).mean(dim=-2, keepdim=True)
        elif self.cfg.norm_activation == "dataset-wise":
            assert (
                self.cfg.dataset_average_activation_norm is not None
            ), "dataset_average_activation_norm must be provided for dataset-wise normalization"
            return (
                math.sqrt(self.cfg.d_model)
                / self.cfg.dataset_average_activation_norm[hook_point]
            )
        else:
            return torch.tensor(1.0, dtype=self.cfg.dtype, device=self.cfg.device)

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        label: (
            Union[
                Float[torch.Tensor, "batch d_model"],
                Float[torch.Tensor, "batch seq_len d_model"],
            ]
            | None
        ) = None,
        return_hidden_pre: Literal[False] = False,
    ) -> Union[
        Float[torch.Tensor, "batch d_sae"], Float[torch.Tensor, "batch seq_len d_sae"]
    ]: ...

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        label: (
            Union[
                Float[torch.Tensor, "batch d_model"],
                Float[torch.Tensor, "batch seq_len d_model"],
            ]
            | None
        ),
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
        label: (
            Union[
                Float[torch.Tensor, "batch d_model"],
                Float[torch.Tensor, "batch seq_len d_model"],
            ]
            | None
        ) = None,
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
    ]:
        """Encode the model activation x into feature activations.

        Args:
            x (torch.Tensor): The input activation tensor.
            label (torch.Tensor, optional): The label activation tensor in transcoder training. Used for normalizing the feature activations. Defaults to None, which means using x as the label.
            return_hidden_pre (bool, optional): Whether to return the hidden pre-activation. Defaults to False.

        Returns:
            torch.Tensor: The feature activations.

        """

        if label is None:
            label = x

        if self.cfg.use_decoder_bias and self.cfg.apply_decoder_bias_to_pre_encoder:
            x = (
                x - self.decoder.bias.to_local() # type: ignore
                if self.cfg.tp_size > 1
                else x - self.decoder.bias
            )

        x = x * self.compute_norm_factor(x, hook_point="in")

        hidden_pre = self.encoder(x)

        if self.cfg.use_glu_encoder:
            hidden_pre_glu = torch.sigmoid(self.encoder_glu(x))

            hidden_pre = hidden_pre * hidden_pre_glu

        hidden_pre = hidden_pre / self.compute_norm_factor(label, hook_point="in")
        hidden_pre = self.hook_hidden_pre(hidden_pre)

        feature_acts = (
            self.feature_act_mask
            * self.feature_act_scale
            * torch.clamp(hidden_pre, min=0.0)
        )

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
    ]:
        """Decode the feature activations into the reconstructed model activation in the label space.

        Args:
            feature_acts (torch.Tensor): The feature activations. Should not be normalized.

        Returns:
            torch.Tensor: The reconstructed model activation. Not normalized.
        """

        reconstructed = self.decoder(feature_acts)
        reconstructed = self.hook_reconstructed(reconstructed)

        return reconstructed

    def compute_loss(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        dead_feature_mask: Float[torch.Tensor, "d_sae"] | None = None,
        label: (
            Union[
                Float[torch.Tensor, "batch d_model"],
                Float[torch.Tensor, "batch seq_len d_model"],
            ]
            | None
        ) = None,
        return_aux_data: bool = True,
    ) -> Union[
        Float[torch.Tensor, "batch"],
        tuple[
            Float[torch.Tensor, "batch"],
            tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        ],
    ]:
        """Compute the loss of the model.

        Args:
            x (torch.Tensor): The input activation tensor.
            label (torch.Tensor, optional): The label activation tensor in transcoder training. Defaults to None, which means using x as the label.
            return_aux_data (bool, optional): Whether to return the auxiliary data. Defaults to False.

        Returns:
            torch.Tensor: The loss value.
        """

        if label is None:
            label = x

        label_norm_factor = self.compute_norm_factor(label, hook_point="out")

        feature_acts, hidden_pre = self.encode(x, label, return_hidden_pre=True)
        feature_acts_normed = feature_acts * label_norm_factor  # (batch, d_sae)
        # hidden_pre_normed = hidden_pre * label_norm_factor

        reconstructed = self.decode(feature_acts)
        reconstructed_normed = reconstructed * label_norm_factor

        label_normed = label * label_norm_factor

        # l_rec: (batch, d_model)
        l_rec = (reconstructed_normed - label_normed).pow(2) / (
            label_normed - label_normed.mean(dim=0, keepdim=True)
        ).pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()

        # l_l1: (batch,)
        if self.cfg.sparsity_include_decoder_norm:

            l_l1 = torch.norm(
                feature_acts_normed * self.decoder_norm(),
                p=self.cfg.lp,
                dim=-1,
            )
        else:
            l_l1 = torch.norm(feature_acts_normed, p=self.cfg.lp, dim=-1)

        l_ghost_resid = torch.tensor(0.0, dtype=self.cfg.dtype, device=self.cfg.device)

        if (
            self.cfg.use_ghost_grads
            and self.training
            and dead_feature_mask is not None
            and dead_feature_mask.sum() > 0
        ):
            # ghost protocol
            assert (
                self.cfg.tp_size == 1
            ), "Ghost protocol not supported in tensor parallel training"
            # 1.
            residual = label_normed - reconstructed_normed
            residual_centred = residual - residual.mean(dim=0, keepdim=True)
            l2_norm_residual = torch.norm(residual, dim=-1)

            # 2.
            feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_feature_mask])
            ghost_out = (
                feature_acts_dead_neurons_only
                @ self.decoder.weight[dead_feature_mask, :]
            )
            l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
            norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
            ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

            # 3.
            l_ghost_resid = (
                torch.pow((ghost_out - residual.detach().float()), 2)
                / (residual_centred.detach() ** 2).sum(dim=-1, keepdim=True).sqrt()
            )
            mse_rescaling_factor = (l_rec / (l_ghost_resid + 1e-6)).detach()
            l_ghost_resid = mse_rescaling_factor * l_ghost_resid

        loss = (
            l_rec.mean()
            + self.current_l1_coefficient * l_l1.mean()
            + l_ghost_resid.mean()
        )

        if return_aux_data:
            aux_data = {
                "feature_acts": feature_acts,
                "reconstructed": reconstructed,
                "hidden_pre": hidden_pre,
            }
            return loss, (
                {"l_rec": l_rec, "l_l1": l_l1, "l_ghost_resid": l_ghost_resid},
                aux_data,
            )

        return loss

    def forward(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        label: (
            Union[
                Float[torch.Tensor, "batch d_model"],
                Float[torch.Tensor, "batch seq_len d_model"],
            ]
            | None
        ) = None,
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:
        """Encode and then decode the input activation tensor, outputting the reconstructed activation tensor."""

        if label is None:
            label = x

        feature_acts = self.encode(x, label)
        reconstructed = self.decode(feature_acts)

        return reconstructed

    @torch.no_grad()
    def update_l1_coefficient(self, training_step):
        if self.cfg.l1_coefficient_warmup_steps <= 0:
            return
        self.current_l1_coefficient = (
            min(1.0, training_step / self.cfg.l1_coefficient_warmup_steps)
            * self.cfg.l1_coefficient
        )

    @torch.no_grad()
    def set_decoder_norm_to_fixed_norm(
        self,
        value: float | None = 1.0,
        force_exact: bool | None = None,
        during_init: bool = False,
    ):
        if value is None:
            return
        decoder_norm = self.decoder_norm(keepdim=True, during_init=during_init)
        if force_exact is None:
            force_exact = self.cfg.decoder_exactly_fixed_norm
        if force_exact:
            self.decoder.weight.data = self.decoder.weight.data * value / decoder_norm
        else:
            # Set the norm of the decoder to not exceed value
            self.decoder.weight.data = (
                self.decoder.weight.data * value / torch.clamp(decoder_norm, min=value)
            )

    @torch.no_grad()
    def set_encoder_norm_to_fixed_norm(self, value: float | None = 1.0):
        if self.cfg.use_glu_encoder:
            raise NotImplementedError("GLU encoder not supported")
        if value is None:
            print(
                f"Encoder norm is not set to a fixed value, using random initialization."
            )
            return
        encoder_norm = self.encoder_norm(keepdim=True)
        self.encoder.weight.data = self.encoder.weight.data * value / encoder_norm

    @torch.no_grad()
    def transform_to_unit_decoder_norm(self):
        """
        If we include decoder norm in the sparsity loss, the final decoder norm is not guaranteed to be 1.
        We make an equivalent transformation to the decoder to make it unit norm.
        See https://transformer-circuits.pub/2024/april-update/index.html#training-saes
        """
        assert (
            self.cfg.sparsity_include_decoder_norm
        ), "Decoder norm is not included in the sparsity loss"
        if self.cfg.use_glu_encoder:
            raise NotImplementedError("GLU encoder not supported")

        decoder_norm = self.decoder_norm()  # (d_sae,)
        self.encoder.weight.data = self.encoder.weight.data * decoder_norm[:, None]
        self.decoder.weight.data = self.decoder.weight.data.T / decoder_norm

        self.encoder.bias.data = self.encoder.bias.data * decoder_norm

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
        to the decoder directions.
        """

        parallel_component = einsum(
            self.decoder.weight.grad,
            self.decoder.weight.data,
            "d_sae d_model, d_sae d_model -> d_sae",
        )

        assert (
            self.decoder.weight.grad is not None
        ), "No gradient to remove parallel component from"

        self.decoder.weight.grad -= einsum(
            parallel_component,
            self.decoder.weight.data,
            "d_sae, d_sae d_model -> d_sae d_model",
        )

    @torch.no_grad()
    def compute_thomson_potential(self):
        dist = (
            torch.cdist(self.decoder.weight, self.decoder.weight, p=2)
            .flatten()[1:]
            .view(self.cfg.d_sae - 1, self.cfg.d_sae + 1)[:, :-1]
        )
        mean_thomson_potential = (1 / dist).mean()
        return mean_thomson_potential

    @staticmethod
    def from_config(cfg: SAEConfig) -> "SparseAutoEncoder":
        """Load the SparseAutoEncoder model from the pretrained configuration.

        Args:
            cfg (SAEConfig): The configuration of the model, containing the sae_pretrained_name_or_path.

        Returns:
            SparseAutoEncoder: The pretrained SparseAutoEncoder model.
        """
        pretrained_name_or_path = cfg.sae_pretrained_name_or_path
        if pretrained_name_or_path is None:
            return SparseAutoEncoder(cfg)

        path = parse_pretrained_name_or_path(pretrained_name_or_path)

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
                raise FileNotFoundError(
                    f"Pretrained model not found at {pretrained_name_or_path}"
                )

        if ckpt_path.endswith(".safetensors"):
            state_dict = safe.load_file(ckpt_path, device=cfg.device)
        else:
            state_dict = torch.load(ckpt_path, map_location=cfg.device)["sae"]

        model = SparseAutoEncoder(cfg)
        model.load_state_dict(state_dict, strict=cfg.strict_loading)

        return model

    @staticmethod
    def from_pretrained(
        pretrained_name_or_path: str, strict_loading: bool = True, **kwargs
    ) -> "SparseAutoEncoder":
        """Load the SparseAutoEncoder model from the pretrained configuration.

        Args:
            pretrained_name_or_path (str): The name or path of the pretrained model.
            strict_loading (bool, optional): Whether to load the model strictly. Defaults to True.
            **kwargs: Additional keyword arguments as BaseModelConfig.

        Returns:
            SparseAutoEncoder: The pretrained SparseAutoEncoder model.
        """
        cfg = SAEConfig.from_pretrained(
            pretrained_name_or_path, strict_loading=strict_loading, **kwargs
        )

        return SparseAutoEncoder.from_config(cfg)

    @torch.no_grad()
    @staticmethod
    def from_initialization_searching(
        activation_store: ActivationStore,
        cfg: LanguageModelSAETrainingConfig,
    ):
        test_batch = activation_store.next(
            batch_size=cfg.train_batch_size
        )
        activation_in, activation_out = test_batch[cfg.sae.hook_point_in], test_batch[cfg.sae.hook_point_out]  # type: ignore

        if (
            cfg.sae.norm_activation == "dataset-wise"
            and cfg.sae.dataset_average_activation_norm is None
        ):
            print(
                f"SAE: Computing average activation norm on the first {cfg.train_batch_size * 8} samples."
            )

            average_in_norm, average_out_norm = (
                activation_in.norm(p=2, dim=1).mean().item(),
                activation_out.norm(p=2, dim=1).mean().item(),
            )

            print(
                f"Average input activation norm: {average_in_norm}\nAverage output activation norm: {average_out_norm}"
            )
            cfg.sae.dataset_average_activation_norm = {
                "in": average_in_norm,
                "out": average_out_norm,
            }

        if cfg.sae.init_decoder_norm is None:
            assert (
                cfg.sae.sparsity_include_decoder_norm
            ), "Decoder norm must be included in sparsity loss"
            if (
                not cfg.sae.init_encoder_with_decoder_transpose
                or cfg.sae.hook_point_in != cfg.sae.hook_point_out
            ):
                raise NotImplementedError(
                    "Transcoders cannot be initialized automatically."
                )
        print("SAE: Starting grid search for initial decoder norm.")

        test_sae = SparseAutoEncoder.from_config(cfg=cfg.sae)

        def grid_search_best_init_norm(search_range: List[float]) -> float:
            losses: Dict[float, float] = {}

            for norm in search_range:
                test_sae.set_decoder_norm_to_fixed_norm(
                    norm, force_exact=True, during_init=True
                )
                test_sae.encoder.weight.data = (
                    test_sae.decoder.weight.data.T.clone().contiguous()
                )
                mse = test_sae.compute_loss(x=activation_in, label=activation_out)[1][0]["l_rec"].mean().item()  # type: ignore
                losses[norm] = mse
            best_norm = min(losses, key=losses.get)  # type: ignore
            return best_norm

        best_norm_coarse = grid_search_best_init_norm(
            torch.linspace(0.1, 1, 10).numpy().tolist()
        )
        best_norm_fine_grained = grid_search_best_init_norm(
            torch.linspace(best_norm_coarse - 0.09, best_norm_coarse + 0.1, 20)
            .numpy()
            .tolist()
        )
        print(
            f"The best (i.e. lowest MSE) initialized norm is {best_norm_fine_grained}"
        )

        test_sae.set_decoder_norm_to_fixed_norm(
            best_norm_fine_grained, force_exact=True
        )
        test_sae.encoder.weight.data = (
            test_sae.decoder.weight.data.T.clone().contiguous()
        )

        return test_sae

    def get_full_state_dict(self) -> dict:
        state_dict = self.state_dict()
        if self.cfg.tp_size > 1:
            state_dict = {
                k: v.full_tensor() if isinstance(v, DTensor) else v
                for k, v in state_dict.items()
            }
        return state_dict

    def save_pretrained(self, ckpt_path: str) -> None:
        """Save the model to the checkpoint path.

        Args:
            ckpt_path (str): The path to save the model. If a directory, the model will be saved to the directory with the default filename `sae_weights.safetensors`.
        """
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "sae_weights.safetensors")
        state_dict = self.get_full_state_dict()
        if is_master():
            if ckpt_path.endswith(".safetensors"):
                safe.save_file(
                    state_dict, ckpt_path, {"version": version("lm-saes")}
                )
            elif ckpt_path.endswith(".pt"):
                torch.save(
                    {"sae": state_dict, "version": version("lm-saes")}, ckpt_path
                )
            else:
                raise ValueError(
                    f"Invalid checkpoint path {ckpt_path}. Currently only supports .safetensors and .pt formats."
                )

    def decoder_norm(self, keepdim: bool = False, during_init: bool = False):
        # We suspect that using torch.norm on dtensor may lead to some bugs during the backward process that are difficult to pinpoint and resolve. Therefore, we first convert the decoder weight from dtensor to tensor for norm calculation, and then redistribute it to different nodes.
        if self.cfg.tp_size == 1 or during_init:
            return torch.norm(self.decoder.weight, p=2, dim=0, keepdim=keepdim)
        else:
            decoder_norm = torch.norm(
                self.decoder.weight.to_local(), p=2, dim=0, keepdim=keepdim # type: ignore
            ) 
            decoder_norm = DTensor.from_local(
                decoder_norm,
                device_mesh=self.device_mesh["tp"],
                placements=[Shard(int(keepdim))],
            )
            decoder_norm = decoder_norm.redistribute(
                placements=[Replicate()], async_op=True
            ).to_local()
            return decoder_norm

    def encoder_norm(
        self,
        keepdim: bool = False,
        during_init: bool = False,
    ):
        if self.cfg.tp_size == 1 or during_init:
            return torch.norm(self.encoder.weight, p=2, dim=1, keepdim=keepdim)
        else:
            encoder_norm = torch.norm(
                self.encoder.weight.to_local(), p=2, dim=1, keepdim=keepdim # type: ignore
            ) 
            encoder_norm = DTensor.from_local(
                encoder_norm, device_mesh=self.device_mesh["tp"], placements=[Shard(0)]
            )
            encoder_norm = encoder_norm.redistribute(
                placements=[Replicate()], async_op=True
            ).to_local()
            return encoder_norm
