from typing import Literal, Union, overload

import torch
from jaxtyping import Float
from torch.distributed.tensor import DTensor

from .config import BaseSAEConfig
from .sae import SparseAutoEncoder
from .utils.misc import all_reduce_tensor, get_tensor_from_specific_rank


class CrossCoder(SparseAutoEncoder):
    """Sparse AutoEncoder model.

    An autoencoder model that learns to compress the input activation tensor into a high-dimensional but sparse feature activation tensor.

    Can also act as a transcoder model, which learns to compress the input activation tensor into a feature activation tensor, and then reconstruct a label activation tensor from the feature activation tensor.
    """

    def __init__(self, cfg: BaseSAEConfig):
        super(CrossCoder, self).__init__(cfg)

    def _decoder_norm(self, decoder: torch.nn.Linear, keepdim: bool = False, local_only=True, aggregate="none"):
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

        input_norm_factor = self.compute_norm_factor(x, hook_point=self.cfg.hook_point_in)
        x = x * input_norm_factor

        if self.cfg.use_decoder_bias and self.cfg.apply_decoder_bias_to_pre_encoder:
            bias = self.decoder.bias.to_local() if isinstance(self.decoder.bias, DTensor) else self.decoder.bias
            x = x - bias

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

        activation_mask = self.activation_function(true_feature_acts)
        feature_acts = hidden_pre * activation_mask

        feature_acts = self.hook_feature_acts(feature_acts)

        if return_hidden_pre:
            return feature_acts, hidden_pre
        return feature_acts

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

        if "topk" not in self.cfg.act_fn:
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
