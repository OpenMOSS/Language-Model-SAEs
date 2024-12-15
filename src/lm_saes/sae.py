import math
import os
from builtins import print
from importlib.metadata import version
from typing import Dict, List, Literal, Union, overload

import safetensors.torch as safe
import torch
from einops import einsum
from jaxtyping import Float
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from transformer_lens.hook_points import HookedRootModule, HookPoint

from .activation.activation_store import ActivationStore
from .config import LanguageModelSAETrainingConfig, SAEConfig
from .utils.huggingface import parse_pretrained_name_or_path
from .utils.misc import is_master, print_once


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
        self.current_k = cfg.top_k
        self.tensor_paralleled = False

        self.encoder = torch.nn.Linear(cfg.d_model, cfg.d_sae, bias=True, device=cfg.device, dtype=cfg.dtype)
        torch.nn.init.kaiming_uniform_(self.encoder.weight)
        torch.nn.init.zeros_(self.encoder.bias)

        if cfg.tp_size > 1 or cfg.ddp_size > 1:
            self.device_mesh = init_device_mesh("cuda", (cfg.ddp_size, cfg.tp_size), mesh_dim_names=("ddp", "tp"))

        if cfg.use_glu_encoder:
            self.encoder_glu = torch.nn.Linear(cfg.d_model, cfg.d_sae, bias=True, device=cfg.device, dtype=cfg.dtype)
            torch.nn.init.kaiming_uniform_(self.encoder_glu.weight)
            torch.nn.init.zeros_(self.encoder_glu.bias)

        self.activation_function = self.activation_function_factory(cfg)

        self.decoder = torch.nn.Linear(
            cfg.d_sae,
            cfg.d_model,
            bias=cfg.use_decoder_bias,
            device=cfg.device,
            dtype=cfg.dtype,
        )
        torch.nn.init.kaiming_uniform_(self.decoder.weight)
        self.set_decoder_norm_to_fixed_norm()

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
        self.set_decoder_norm_to_fixed_norm(self.cfg.init_decoder_norm, force_exact=True)

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
        """Set the parameters to be trained against feature suppression."""

        finetune_for_suppression_parameters = [self.decoder.weight]

        if self.cfg.use_decoder_bias:
            finetune_for_suppression_parameters.append(self.decoder.bias)
        for p in self.parameters():
            p.requires_grad_(False)
        for p in finetune_for_suppression_parameters:
            p.requires_grad_(True)

    def activation_function_factory(self, cfg: SAEConfig):
        if self.cfg.act_fn.lower() == "relu":
            return lambda hidden_pre: torch.clamp(hidden_pre, min=0.0)
        if self.cfg.act_fn.lower() == "topk":

            def topk_activation(hidden_pre):
                feature_acts = torch.clamp(hidden_pre, min=0.0)
                if self.cfg.sparsity_include_decoder_norm:
                    true_feature_acts = feature_acts * self.decoder_norm()
                else:
                    true_feature_acts = feature_acts
                topk = torch.topk(true_feature_acts, k=self.current_k, dim=-1)
                result = torch.zeros_like(feature_acts)
                result.scatter_(-1, topk.indices, feature_acts.gather(-1, topk.indices))
                return result

            return topk_activation

        elif cfg.act_fn.lower() == "jumprelu":
            return lambda hidden_pre: hidden_pre.where(hidden_pre > cfg.jump_relu_threshold, 0)
        else:
            raise NotImplementedError(f"Not implemented activation function {cfg.act_fn}")

    def compute_norm_factor(self, x: torch.Tensor, hook_point: str) -> float | torch.Tensor:
        """Compute the normalization factor for the activation vectors."""

        # Normalize the activation vectors to have L2 norm equal to sqrt(d_model)
        if self.cfg.norm_activation == "token-wise":
            return math.sqrt(self.cfg.d_model) / torch.norm(x, 2, dim=-1, keepdim=True)
        elif self.cfg.norm_activation == "batch-wise":
            return math.sqrt(self.cfg.d_model) / torch.norm(x, 2, dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        elif self.cfg.norm_activation == "dataset-wise":
            assert (
                self.cfg.dataset_average_activation_norm is not None
            ), "dataset_average_activation_norm must be provided for dataset-wise normalization"
            return math.sqrt(self.cfg.d_model) / self.cfg.dataset_average_activation_norm[hook_point]
        else:
            return torch.tensor(1.0, dtype=self.cfg.dtype, device=self.cfg.device)

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
    ]:
        """Encode the model activation x into feature activations.

        Args:
            x (torch.Tensor): The input activation tensor.
            return_hidden_pre (bool, optional): Whether to return the hidden pre-activation. Defaults to False.

        Returns:
            torch.Tensor: The feature activations.

        """

        input_norm_factor = self.compute_norm_factor(x, hook_point="in")
        x = x * input_norm_factor

        if self.cfg.use_decoder_bias and self.cfg.apply_decoder_bias_to_pre_encoder:
            x = (
                x - self.decoder.bias.to_local()  # type: ignore
                if self.cfg.tp_size > 1
                else x - self.decoder.bias
            )

        hidden_pre = self.encoder(x)

        if self.cfg.use_glu_encoder:
            hidden_pre_glu = torch.sigmoid(self.encoder_glu(x))

            hidden_pre = hidden_pre * hidden_pre_glu

        hidden_pre = self.hook_hidden_pre(hidden_pre)

        feature_acts = self.activation_function(hidden_pre)

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

    @overload
    def compute_loss(  # type: ignore . I have no idea why these overloads are overlapping
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        dead_feature_mask: Float[torch.Tensor, " d_sae"] | None = None,
        label: (
            Union[
                Float[torch.Tensor, "batch d_model"],
                Float[torch.Tensor, "batch seq_len d_model"],
            ]
            | None
        ) = None,
        return_aux_data: Literal[True] = True,
    ) -> tuple[
        Float[torch.Tensor, " batch"],
        tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ]: ...

    @overload
    def compute_loss(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        dead_feature_mask: Float[torch.Tensor, " d_sae"] | None = None,
        label: (
            Union[
                Float[torch.Tensor, "batch d_model"],
                Float[torch.Tensor, "batch seq_len d_model"],
            ]
            | None
        ) = None,
        return_aux_data: Literal[False] = False,
    ) -> Float[torch.Tensor, " batch"]: ...

    def compute_loss(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        dead_feature_mask: Float[torch.Tensor, " d_sae"] | None = None,
        label: (
            Union[
                Float[torch.Tensor, "batch d_model"],
                Float[torch.Tensor, "batch seq_len d_model"],
            ]
            | None
        ) = None,
        return_aux_data: bool = True,
    ) -> Union[
        Float[torch.Tensor, " batch"],
        tuple[
            Float[torch.Tensor, " batch"],
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

        feature_acts, hidden_pre = self.encode(x, return_hidden_pre=True)
        reconstructed = self.decode(feature_acts)

        label_norm_factor = self.compute_norm_factor(label, hook_point="out")

        label_normed = label * label_norm_factor

        # l_rec: (batch, d_model)
        l_rec = (reconstructed - label_normed).pow(2)

        if self.cfg.use_batch_norm_mse:
            l_rec = (
                l_rec
                / (label_normed - label_normed.mean(dim=0, keepdim=True))
                .pow(2)
                .sum(dim=-1, keepdim=True)
                .clamp(min=1e-8)
                .sqrt()
            )

        # l_l1: (batch,)
        if self.cfg.sparsity_include_decoder_norm:
            feature_acts = feature_acts * self.decoder_norm()

        loss = l_rec.mean()
        loss_dict = {
            "l_rec": l_rec,
        }

        if not self.cfg.act_fn == "topk":
            l_l1 = torch.norm(feature_acts, p=self.cfg.lp, dim=-1)
            loss_dict["l_l1"] = l_l1
            loss = loss + self.current_l1_coefficient * l_l1.mean()

        if self.cfg.use_ghost_grads and self.training and dead_feature_mask is not None and dead_feature_mask.sum() > 0:
            l_ghost_resid = self.compute_ghost_grad_loss(
                reconstructed,
                hidden_pre,
                label_normed,
                dead_feature_mask,
                l_rec,
            )
            loss = loss + l_ghost_resid.mean()
            loss_dict["l_ghost_resid"] = l_ghost_resid

        if return_aux_data:
            aux_data = {
                "feature_acts": feature_acts,
                "reconstructed": reconstructed / label_norm_factor,
                "hidden_pre": hidden_pre,
            }
            return loss, (
                loss_dict,
                aux_data,
            )

        return loss

    def forward(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:
        """Encode and then decode the input activation tensor, outputting the reconstructed activation tensor."""

        feature_acts = self.encode(x)
        reconstructed = self.decode(feature_acts)

        return reconstructed

    @torch.no_grad()
    def update_l1_coefficient(self, training_step):
        if self.cfg.l1_coefficient_warmup_steps <= 0:
            return
        self.current_l1_coefficient = (
            min(1.0, training_step / self.cfg.l1_coefficient_warmup_steps) * self.cfg.l1_coefficient
        )

    @torch.no_grad()
    def update_k(self, training_step):
        if self.cfg.k_warmup_steps <= 0:
            return

        assert self.cfg.initial_k is not None, "initial_k must be provided"

        self.current_k = math.ceil(
            max(
                1.0,
                self.cfg.initial_k + (1 - self.cfg.initial_k) / self.cfg.k_warmup_steps * training_step,
            )
            * self.cfg.top_k
        )

    @torch.no_grad()
    def set_decoder_norm_to_fixed_norm(
        self,
        value: float | None = 1.0,
        force_exact: bool | None = None,
    ):
        if value is None:
            return
        decoder_norm = self.decoder_norm(keepdim=True)
        if force_exact is None:
            force_exact = self.cfg.decoder_exactly_fixed_norm

        if self.cfg.tp_size > 1 and self.tensor_paralleled:
            decoder_norm = distribute_tensor(
                decoder_norm,
                device_mesh=self.device_mesh["tp"],
                placements=[Shard(0)],
            )

        if force_exact:
            self.decoder.weight.data *= value / decoder_norm
        else:
            # Set the norm of the decoder to not exceed value
            self.decoder.weight.data *= value / torch.clamp(decoder_norm, min=value)

    @torch.no_grad()
    def set_encoder_norm_to_fixed_norm(self, value: float | None = 1.0):
        if self.cfg.use_glu_encoder:
            raise NotImplementedError("GLU encoder not supported")
        if value is None:
            print("Encoder norm is not set to a fixed value, using random initialization.")
            return
        encoder_norm = self.encoder_norm(keepdim=True)
        if self.cfg.tp_size > 1 and not self.tensor_paralleled:
            encoder_norm = distribute_tensor(
                encoder_norm,
                device_mesh=self.device_mesh["tp"],
                placements=[Shard(0)],
            )
        self.encoder.weight.data *= value / encoder_norm

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

        assert self.decoder.weight.grad is not None, "No gradient to remove parallel component from"

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

    @classmethod
    def from_config(cls, cfg: SAEConfig) -> "SparseAutoEncoder":
        """Load the SparseAutoEncoder model from the pretrained configuration.

        Args:
            cfg (SAEConfig): The configuration of the model, containing the sae_pretrained_name_or_path.

        Returns:
            SparseAutoEncoder: The pretrained SparseAutoEncoder model.
        """
        pretrained_name_or_path = cfg.sae_pretrained_name_or_path
        if pretrained_name_or_path is None:
            return cls(cfg)

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
                raise FileNotFoundError(f"Pretrained model not found at {pretrained_name_or_path}")

        if ckpt_path.endswith(".safetensors"):
            state_dict = safe.load_file(ckpt_path, device=cfg.device)
        else:
            state_dict = torch.load(ckpt_path, map_location=cfg.device)["sae"]

        model = cls(cfg)

        if cfg.norm_activation == "dataset-wise":
            state_dict = model.standardize_parameters_of_dataset_activation_scaling(state_dict)

        if cfg.sparsity_include_decoder_norm:
            state_dict = model.transform_to_unit_decoder_norm(state_dict)

        if cfg.act_fn == "topk" and cfg.jump_relu_threshold > 0:
            print("Converting topk activation to jumprelu for inference. Features are set independent to each other.")
            model.cfg.act_fn = "jumprelu"

        model.load_state_dict(state_dict, strict=cfg.strict_loading)

        return model

    @staticmethod
    def from_pretrained(pretrained_name_or_path: str, strict_loading: bool = True, **kwargs) -> "SparseAutoEncoder":
        """Load the SparseAutoEncoder model from the pretrained configuration.

        Args:
            pretrained_name_or_path (str): The name or path of the pretrained model.
            strict_loading (bool, optional): Whether to load the model strictly. Defaults to True.
            **kwargs: Additional keyword arguments as BaseModelConfig.

        Returns:
            SparseAutoEncoder: The pretrained SparseAutoEncoder model.
        """
        cfg = SAEConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)

        return SparseAutoEncoder.from_config(cfg)

    @classmethod
    @torch.no_grad()
    def from_initialization_searching(
        cls,
        activation_store: ActivationStore,
        cfg: LanguageModelSAETrainingConfig,
    ):
        test_batch = activation_store.next(batch_size=cfg.train_batch_size * 8)
        activation_in, activation_out = test_batch[cfg.sae.hook_point_in], test_batch[cfg.sae.hook_point_out]  # type: ignore

        if cfg.sae.norm_activation == "dataset-wise" and cfg.sae.dataset_average_activation_norm is None:
            print_once(f"SAE: Computing average activation norm on the first {cfg.train_batch_size * 8} samples.")

            average_in_norm, average_out_norm = (
                activation_in.norm(p=2, dim=1).mean().item(),
                activation_out.norm(p=2, dim=1).mean().item(),
            )

            print_once(
                f"Average input activation norm: {average_in_norm}\nAverage output activation norm: {average_out_norm}"
            )
            cfg.sae.dataset_average_activation_norm = {
                "in": average_in_norm,
                "out": average_out_norm,
            }

        if cfg.sae.init_decoder_norm is None:
            assert cfg.sae.sparsity_include_decoder_norm, "Decoder norm must be included in sparsity loss"
            if not cfg.sae.init_encoder_with_decoder_transpose or cfg.sae.hook_point_in != cfg.sae.hook_point_out:
                sae = SparseAutoEncoder.from_config(cfg=cfg.sae)
                print_once(
                    f"Transcoders cannot be initialized automatically. Skipping.\nEncoder norm: {sae.encoder_norm().mean().item()}\nDecoder norm: {sae.decoder_norm().mean().item()}"
                )
                return sae

            print_once("SAE: Starting grid search for initial decoder norm.")

            sae = cls.from_config(cfg=cfg.sae)

            def grid_search_best_init_norm(search_range: List[float]) -> float:
                losses: Dict[float, float] = {}

                for norm in search_range:
                    sae.set_decoder_norm_to_fixed_norm(norm, force_exact=True)
                    sae.encoder.weight.data = sae.decoder.weight.data.T.clone().contiguous()
                    mse = (
                        sae.compute_loss(
                            x=activation_in[: cfg.train_batch_size],
                            label=activation_out[: cfg.train_batch_size],
                        )[1][0]["l_rec"]
                        .mean()
                        .item()
                    )  # type: ignore

                    losses[norm] = mse
                best_norm = min(losses, key=losses.get)  # type: ignore
                return best_norm

            best_norm_coarse = grid_search_best_init_norm(torch.linspace(0.1, 1, 10).numpy().tolist())  # type: ignore
            best_norm_fine_grained = grid_search_best_init_norm(
                torch.linspace(best_norm_coarse - 0.09, best_norm_coarse + 0.1, 20).numpy().tolist()  # type: ignore
            )

            print(f"The best (i.e. lowest MSE) initialized norm is {best_norm_fine_grained}")

            sae.set_decoder_norm_to_fixed_norm(best_norm_fine_grained, force_exact=True)

        else:
            sae = cls.from_config(cfg=cfg.sae)

        if cfg.sae.init_encoder_with_decoder_transpose:
            sae.encoder.weight.data = sae.decoder.weight.data.T.clone().contiguous()

        if cfg.sae.bias_init_method == "geometric_median":
            sae.decoder.bias.data = (sae.compute_norm_factor(activation_out, hook_point="out") * activation_out).mean(0)

            if not cfg.sae.apply_decoder_bias_to_pre_encoder:
                normalized_input = sae.compute_norm_factor(activation_in, hook_point="in") * activation_in
                normalized_median = normalized_input.mean(0)
                sae.encoder.bias.data = -normalized_median @ sae.encoder.weight.data.T

        return sae

    def get_full_state_dict(self) -> dict:
        state_dict = self.state_dict()
        if self.cfg.tp_size > 1:
            state_dict = {k: v.full_tensor() if isinstance(v, DTensor) else v for k, v in state_dict.items()}
        return state_dict

    @torch.no_grad()
    def transform_to_unit_decoder_norm(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        decoder_norm = torch.norm(state_dict["decoder.weight"], p=2, dim=0, keepdim=False)
        state_dict["decoder.weight"] = state_dict["decoder.weight"] / decoder_norm
        state_dict["encoder.weight"] = state_dict["encoder.weight"] * decoder_norm[:, None]
        state_dict["encoder.bias"] = state_dict["encoder.bias"] * decoder_norm
        return state_dict

    @torch.no_grad()
    def standardize_parameters_of_dataset_activation_scaling(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        assert self.cfg.norm_activation == "dataset-wise"
        assert self.cfg.dataset_average_activation_norm is not None

        input_norm_factor = math.sqrt(self.cfg.d_model) / self.cfg.dataset_average_activation_norm["in"]
        output_norm_factor = math.sqrt(self.cfg.d_model) / self.cfg.dataset_average_activation_norm["out"]

        state_dict["encoder.bias"] = state_dict["encoder.bias"] / input_norm_factor
        state_dict["decoder.bias"] = state_dict["decoder.bias"] / output_norm_factor
        state_dict["decoder.weight"] = state_dict["decoder.weight"] * input_norm_factor / output_norm_factor

        self.cfg.norm_activation = "inference"

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
                safe.save_file(state_dict, ckpt_path, {"version": version("lm-saes")})
            elif ckpt_path.endswith(".pt"):
                torch.save({"sae": state_dict, "version": version("lm-saes")}, ckpt_path)
            else:
                raise ValueError(
                    f"Invalid checkpoint path {ckpt_path}. Currently only supports .safetensors and .pt formats."
                )

    def decoder_norm(self, keepdim: bool = False):
        # We suspect that using torch.norm on dtensor may lead to some bugs during the backward process that are difficult to pinpoint and resolve. Therefore, we first convert the decoder weight from dtensor to tensor for norm calculation, and then redistribute it to different nodes.
        if self.cfg.tp_size == 1 or not self.tensor_paralleled:
            return torch.norm(self.decoder.weight, p=2, dim=0, keepdim=keepdim)
        else:
            assert isinstance(self.decoder.weight, DTensor)
            decoder_norm = torch.norm(
                self.decoder.weight.to_local(),
                p=2,
                dim=0,
                keepdim=keepdim,  # type: ignore
            )
            decoder_norm = DTensor.from_local(
                decoder_norm,
                device_mesh=self.device_mesh["tp"],
                placements=[Shard(int(keepdim))],
            )
            decoder_norm = decoder_norm.redistribute(placements=[Replicate()], async_op=True).to_local()
            return decoder_norm

    def encoder_norm(
        self,
        keepdim: bool = False,
    ):
        if self.cfg.tp_size == 1 or not self.tensor_paralleled:
            return torch.norm(self.encoder.weight, p=2, dim=1, keepdim=keepdim)
        else:
            assert isinstance(self.encoder.weight, DTensor)
            encoder_norm = torch.norm(
                self.encoder.weight.to_local(),
                p=2,
                dim=1,
                keepdim=keepdim,  # type: ignore
            )
            encoder_norm = DTensor.from_local(encoder_norm, device_mesh=self.device_mesh["tp"], placements=[Shard(0)])
            encoder_norm = encoder_norm.redistribute(placements=[Replicate()], async_op=True).to_local()
            return encoder_norm

    def compute_ghost_grad_loss(self, reconstructed, hidden_pre, label_normed, dead_feature_mask, l_rec):
        # ghost protocol
        assert self.cfg.tp_size == 1, "Ghost protocol not supported in tensor parallel training"
        # 1.
        residual = label_normed - reconstructed
        residual_centred = residual - residual.mean(dim=0, keepdim=True)
        l2_norm_residual = torch.norm(residual, dim=-1)

        # 2.
        feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_feature_mask])
        ghost_out = feature_acts_dead_neurons_only @ self.decoder.weight[dead_feature_mask, :]
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

        return l_ghost_resid
