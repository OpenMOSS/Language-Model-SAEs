from typing import Callable, Literal, Union, cast, overload

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

    def activation_function_factory(self) -> Callable[[torch.Tensor], torch.Tensor]:
        assert self.cfg.act_fn.lower() in [
            "relu",
            "topk",
            "jumprelu",
            "batchtopk",
        ], f"Not implemented activation function {self.cfg.act_fn}"
        if self.cfg.act_fn.lower() == "jumprelu":

            class STEFunction(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input: torch.Tensor, log_jumprelu_threshold: torch.Tensor):
                    jumprelu_threshold = log_jumprelu_threshold.exp()
                    jumprelu_threshold = all_reduce_tensor(jumprelu_threshold, aggregate="sum")
                    ctx.save_for_backward(input, jumprelu_threshold)
                    return input.gt(jumprelu_threshold).to(input.dtype)

                @staticmethod
                def backward(ctx, *grad_outputs: torch.Tensor):
                    assert len(grad_outputs) == 1
                    grad_output = grad_outputs[0]

                    input, jumprelu_threshold = ctx.saved_tensors
                    grad_input = torch.zeros_like(input)
                    grad_log_jumprelu_threshold_unscaled = torch.where(
                        (input - jumprelu_threshold).abs() < self.cfg.jumprelu_threshold_window * 0.5,
                        -jumprelu_threshold / self.cfg.jumprelu_threshold_window,
                        0.0,
                    )
                    grad_log_jumprelu_threshold = (
                        grad_log_jumprelu_threshold_unscaled
                        / torch.where(
                            ((input - jumprelu_threshold).abs() < self.cfg.jumprelu_threshold_window * 0.5)
                            * (input != 0.0),
                            input,
                            1.0,
                        )
                        * grad_output
                    )
                    grad_log_jumprelu_threshold = grad_log_jumprelu_threshold.sum(
                        dim=tuple(range(grad_log_jumprelu_threshold.ndim - 1))
                    )

                    return grad_input, grad_log_jumprelu_threshold

            return lambda x: cast(torch.Tensor, STEFunction.apply(x, self.log_jumprelu_threshold))

        else:
            return super().activation_function_factory()

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
            sparsity_scores = hidden_pre * self._decoder_norm(
                decoder=self.decoder,
                local_only=True,
            )
        else:
            sparsity_scores = hidden_pre

        activation_mask = self.activation_function(sparsity_scores)
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
        sparsity_loss_type: Literal["power", "tanh", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        p: int = 1,
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
        return_aux_data: bool = True,
        **kwargs,
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

        l_rec = l_rec.sum(dim=-1).mean()

        loss = l_rec
        loss_dict = {
            "l_rec": l_rec,
        }

        if sparsity_loss_type == "power":
            l_s = torch.norm(feature_acts * self._decoder_norm(decoder=self.decoder), p=p, dim=-1)
            loss_dict["l_s"] = self.current_l1_coefficient * l_s.mean()
            assert self.current_l1_coefficient is not None
            loss = loss + self.current_l1_coefficient * l_s.mean()
        elif sparsity_loss_type == "tanh":
            l_s = torch.tanh(tanh_stretch_coefficient * feature_acts * self._decoder_norm(decoder=self.decoder)).sum(
                dim=-1
            )
            loss_dict["l_s"] = self.current_l1_coefficient * l_s.mean()
            assert self.current_l1_coefficient is not None
            loss = loss + self.current_l1_coefficient * l_s.mean()
        elif sparsity_loss_type is None:
            pass
        else:
            raise ValueError(f"sparsity_loss_type f{sparsity_loss_type} not supported.")

        loss = all_reduce_tensor(loss, aggregate="mean")

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

    @torch.no_grad()
    def log_statistics(self):
        assert self.dataset_average_activation_norm is not None
        return {f"info/{k}": v for k, v in self.dataset_average_activation_norm.items()}

    def initialize_with_same_weight_across_layers(self):
        self.encoder.weight.data = get_tensor_from_specific_rank(self.encoder.weight.data.clone(), src=0)
        self.encoder.bias.data = get_tensor_from_specific_rank(self.encoder.bias.data.clone(), src=0)
        self.decoder.weight.data = get_tensor_from_specific_rank(self.decoder.weight.data.clone(), src=0)
        self.decoder.bias.data = get_tensor_from_specific_rank(self.decoder.bias.data.clone(), src=0)
