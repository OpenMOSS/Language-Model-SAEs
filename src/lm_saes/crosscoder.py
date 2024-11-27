import math
import os
from builtins import print
from importlib.metadata import version
from typing import Dict, List, Literal, Union, overload

import safetensors.torch as safe
import torch
import torch.distributed as dist
from einops import einsum
from jaxtyping import Float
from transformer_lens.hook_points import HookedRootModule, HookPoint

from .activation.activation_store import ActivationStore
from .config import LanguageModelSAETrainingConfig, SAEConfig
from .utils.huggingface import parse_pretrained_name_or_path
from .utils.misc import is_master, gather_tensors_from_specific_rank, all_gather_tensor, print_once, get_tensor_from_specific_rank, assert_tensor_consistency
from .sae import SparseAutoEncoder


class CrossCoder(SparseAutoEncoder):
    """Sparse AutoEncoder model.

    An autoencoder model that learns to compress the input activation tensor into a high-dimensional but sparse feature activation tensor.

    Can also act as a transcoder model, which learns to compress the input activation tensor into a feature activation tensor, and then reconstruct a label activation tensor from the feature activation tensor.
    """

    def __init__(self, cfg: SAEConfig):
        """Initialize the CrossCoder model.

        Args:
            cfg (SAEConfig): The configuration of the model.
        """

        super(CrossCoder, self).__init__(cfg)

        self.cfg = cfg
        self.current_l1_coefficient = cfg.l1_coefficient
        self.current_k = cfg.top_k
        self.tensor_paralleled = False

        self.encoder = torch.nn.Linear(cfg.d_model, cfg.d_sae, bias=True, device=cfg.device, dtype=cfg.dtype)
        torch.nn.init.kaiming_uniform_(self.encoder.weight)
        torch.nn.init.zeros_(self.encoder.bias)

        if cfg.tp_size > 1 or cfg.ddp_size > 1:
            raise NotImplementedError('TODO: currently do not support further distributing each layer')
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
    

    def decoder_norm(self, keepdim: bool = False, local_only=True, aggregate='none'):
        decoder_norm = torch.norm(self.decoder.weight, p=2, dim=0, keepdim=keepdim)
        if not local_only:
            decoder_norm = all_gather_tensor(decoder_norm, aggregate=aggregate)
        return decoder_norm


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
            average_in_norm, average_out_norm = (
                activation_in.norm(p=2, dim=1).mean().item(),
                activation_out.norm(p=2, dim=1).mean().item(),
            )

            cfg.sae.dataset_average_activation_norm = {
                "in": average_in_norm,
                "out": average_out_norm,
            }
        
        sae = cls.from_config(cfg=cfg.sae)
        return sae
    
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

        hidden_pre = all_gather_tensor(hidden_pre, aggregate='sum')
        hidden_pre = self.hook_hidden_pre(hidden_pre)

        feature_acts = self.activation_function(hidden_pre)
        feature_acts = self.hook_feature_acts(feature_acts)

        if return_hidden_pre:
            return feature_acts, hidden_pre
        return feature_acts
    
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
        
        l_rec = l_rec.mean()
        l_rec = all_gather_tensor(l_rec, aggregate='mean')

        loss = l_rec
        loss_dict = {
            "l_rec": l_rec,
        }

        # l_l1: (batch,)
        feature_acts = feature_acts * self.decoder_norm(local_only=False, aggregate='mean')

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
    
    def initialize_with_same_weight_across_layers(self):
        self.encoder.weight.data = get_tensor_from_specific_rank(self.encoder.weight.data.clone(), src=0)
        self.encoder.bias.data = get_tensor_from_specific_rank(self.encoder.bias.data.clone(), src=0)
        self.decoder.weight.data = get_tensor_from_specific_rank(self.decoder.weight.data.clone(), src=0)
        self.decoder.bias.data = get_tensor_from_specific_rank(self.decoder.bias.data.clone(), src=0)

    def search_for_enc_dec_norm_with_lowest_mse(self, activation_store, cfg):

        test_batch = activation_store.next(batch_size=cfg.train_batch_size * 8)
        activation_in, activation_out = test_batch[self.cfg.hook_point_in], test_batch[self.cfg.hook_point_out]

        def grid_search_best_init_norm(search_range: List[float]) -> float:
                losses: Dict[float, float] = {}

                for norm in search_range:
                    self.set_decoder_norm_to_fixed_norm(norm, force_exact=True)
                    self.set_encoder_norm_to_fixed_norm(norm)
                    mse = (
                        self.compute_loss(
                            x=activation_in[: cfg.train_batch_size],
                            label=activation_out[: cfg.train_batch_size],
                        )[1][0]["l_rec"]
                        .mean()
                        .item()
                    )  # type: ignore

                    losses[norm] = mse
                best_norm = min(losses, key=losses.get)  # type: ignore
                return best_norm

        best_norm_coarse = grid_search_best_init_norm(torch.linspace(0.1, 1, 10).numpy().tolist())
        best_norm_fine_grained = grid_search_best_init_norm(
            torch.linspace(best_norm_coarse - 0.09, best_norm_coarse + 0.1, 20).numpy().tolist()
        )

        print_once(f"The best (i.e. lowest MSE) initialized norm is {best_norm_fine_grained}")

        self.set_decoder_norm_to_fixed_norm(best_norm_fine_grained, force_exact=True)

        return self