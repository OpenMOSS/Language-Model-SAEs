from typing import Dict, List, Literal, Union, overload

import torch
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint

from .activation.activation_store import ActivationStore
from .config import LanguageModelSAETrainingConfig, SAEConfig
from .sae import SparseAutoEncoder
from .utils.misc import all_reduce_tensor, get_tensor_from_specific_rank, print_once


class CrossCoder(SparseAutoEncoder):
    """Sparse AutoEncoder model.

    An autoencoder model that learns to compress the input activation tensor into a high-dimensional but sparse feature activation tensor.

    Can also act as a transcoder model, which learns to compress the input activation tensor into a feature activation tensor, and then reconstruct a label activation tensor from the feature activation tensor.
    """

    def __init__(self, cfg: BaseSAEConfig):
        super(CrossCoder, self).__init__(cfg)

        if cfg.tp_size > 1 or cfg.ddp_size > 1:
            raise NotImplementedError("TODO: currently do not support further distributing each layer for Crosscoders.")
            # self.device_mesh = init_device_mesh("cuda", (cfg.ddp_size, cfg.tp_size), mesh_dim_names=("ddp", "tp"))

        if cfg.use_glu_encoder:
            self.encoder_glu = torch.nn.Linear(cfg.d_model, cfg.d_sae, bias=True, device=cfg.device, dtype=cfg.dtype)
            torch.nn.init.kaiming_uniform_(self.encoder_glu.weight)
            torch.nn.init.zeros_(self.encoder_glu.bias)

    def _decoder_norm(
        self, 
        decoder: torch.nn.Linear, 
        keepdim: bool = False, 
        local_only=True, 
        aggregate="none"
    ):
        decoder_norm = super()._decoder_norm(
            decoder=decoder,
            keepdim=keepdim,
        )
        if not local_only:
            decoder_norm = all_reduce_tensor(
                decoder_norm, 
                aggregate=aggregate,
            )
        return decoder_norm

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

        input_norm_factor = self.compute_norm_factor(x, hook_point=self.cfg.hook_point_in)
        x = x * input_norm_factor

        if self.cfg.use_decoder_bias and self.cfg.apply_decoder_bias_to_pre_encoder:
            x = (
                x - self.decoder.bias.to_local()  # type: ignore
                if self.cfg.tp_size > 1
                else x - self.decoder.bias
            )

        hidden_pre = self.encoder(x)

        hidden_pre = all_reduce_tensor(hidden_pre, aggregate="sum")
        hidden_pre = self.hook_hidden_pre(hidden_pre)
        
        if self.cfg.sparsity_include_decoder_norm:
            true_feature_acts = hidden_pre * self._decoder_norm(
                decoder=self.decoder,
                local_only=True,
            )
        else:
            true_feature_acts = hidden_pre

        feature_acts = self.activation_function(true_feature_acts)
        feature_acts = hidden_pre * activation_mask

        feature_acts = self.hook_feature_acts(feature_acts)

        if return_hidden_pre:
            return feature_acts, hidden_pre
        return feature_acts

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
    ]:
        x: torch.Tensor = batch[self.cfg.hook_point_in]
        label: torch.Tensor = batch[self.cfg.hook_point_out]

        feature_acts, hidden_pre = self.encode(x, return_hidden_pre=True)
        reconstructed = self.decode(feature_acts)

        label_norm_factor = self.compute_norm_factor(label, hook_point=self.cfg.hook_point_out)
        label_normed = label * label_norm_factor

        # l_rec: (batch, d_model)
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

        l_rec = l_rec.mean()
        l_rec = all_reduce_tensor(l_rec, aggregate="mean")

        loss = l_rec
        loss_dict = {
            "l_rec": l_rec,
        }

        # l_l1: (batch,)
        feature_acts = feature_acts * self.decoder_norm(local_only=False, aggregate="mean")

        if not ("topk" in self.cfg.act_fn):
            l_lp = torch.norm(feature_acts, p=lp, dim=-1)
            loss_dict["l_lp"] = l_l1
            assert self.current_l1_coefficient is not None
            loss = loss + self.current_l1_coefficient * l_l1.mean()

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

