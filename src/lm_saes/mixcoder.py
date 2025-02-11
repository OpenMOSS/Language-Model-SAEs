import math
from typing import Literal, MutableMapping, Union, cast, overload

import torch
from jaxtyping import Float
from torch import nn
from torch.distributed.tensor import DTensor
from transformer_lens.hook_points import HookPoint

from lm_saes.config import MixCoderConfig
from lm_saes.sae import SparseAutoEncoder


class MixCoder(SparseAutoEncoder):
    def __init__(self, cfg: MixCoderConfig):
        """A multi-modal sparse autoencoder that handles different modalities with separate encoder/decoder pairs.

        This class extends the base SparseAutoEncoder to support multiple modalities, where each modality
        has its own encoder and decoder, plus a shared component. The architecture allows for modality-specific
        feature extraction while maintaining shared representations.

        Attributes:
            modality_index (dict[str, tuple[int, int]]): Maps modality names to their index ranges in the feature space
            modality_indices (dict[str, torch.Tensor]): Maps modality names to their token indices
            encoder (MutableMapping[str, nn.Linear]): Dictionary of encoders for each modality
            decoder (MutableMapping[str, nn.Linear]): Dictionary of decoders for each modality
            encoder_glu (MutableMapping[str, nn.Linear]): Optional GLU gates for encoders
        """
        super().__init__(cfg)
        # remove encoder and decoder initialized by super()
        del self.encoder
        del self.decoder
        if cfg.use_glu_encoder:
            del self.encoder_glu

        # initialize new encoder and decoder
        self.cfg = cfg
        self.modality_index = {}
        self.modality_indices = {}
        self.encoder = cast(MutableMapping[str, nn.Linear], nn.ModuleDict())
        self.decoder = cast(MutableMapping[str, nn.Linear], nn.ModuleDict())
        self.encoder_glu = cast(MutableMapping[str, nn.Linear], nn.ModuleDict())
        self.hook_hidden_pre = HookPoint()
        self.hook_feature_acts = HookPoint()
        self.hook_reconstructed = HookPoint()
        for modality, d_modality in cfg.modalities.items():
            self.encoder[modality] = nn.Linear(cfg.d_model, d_modality, bias=True, device=cfg.device, dtype=cfg.dtype)
            self.decoder[modality] = nn.Linear(
                d_modality, cfg.d_model, bias=cfg.use_decoder_bias, device=cfg.device, dtype=cfg.dtype
            )
            if cfg.use_glu_encoder:
                self.encoder_glu[modality] = nn.Linear(
                    cfg.d_model, d_modality, bias=True, device=cfg.device, dtype=cfg.dtype
                )

        index = 0
        for modality, d_modality in cfg.modalities.items():
            if modality == "shared":
                continue
            self.modality_index[modality] = (index, index + d_modality)
            index += d_modality

        assert index + cfg.modalities["shared"] == cfg.d_sae
        self.modality_index["shared"] = (index, cfg.d_sae)

    @torch.no_grad()
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool):
        for modality in self.cfg.modalities.keys():
            self._set_decoder_to_fixed_norm(self.decoder[modality], value, force_exact)

    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float):
        for modality in self.cfg.modalities.keys():
            self._set_encoder_to_fixed_norm(self.encoder[modality], value)

    @torch.no_grad()
    def _get_full_state_dict(self):
        state_dict = self.state_dict()
        if self.device_mesh and self.device_mesh["model"].size(0) > 1:
            state_dict = {k: v.full_tensor() if isinstance(v, DTensor) else v for k, v in state_dict.items()}

        if self.dataset_average_activation_norm is not None:
            for hook_point, value in self.dataset_average_activation_norm.items():
                state_dict[f"dataset_average_activation_norm.{hook_point}"] = torch.tensor(value)

        for modality, indices in self.modality_indices.items():
            state_dict[f"modality_indices.{modality}"] = indices

        if not self.cfg.sparsity_include_decoder_norm:
            for modality in self.cfg.modalities.keys():
                state_dict[f"decoder.{modality}.weight"] = self.decoder[modality].weight.data.clone()
                decoder_norm = torch.norm(state_dict[f"decoder.{modality}.weight"], p=2, dim=0, keepdim=True)
                state_dict[f"decoder.{modality}.weight"] = state_dict[f"decoder.{modality}.weight"] / decoder_norm

        return cast(dict[str, torch.Tensor], state_dict)

    @torch.no_grad()
    def _load_full_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        super()._load_full_state_dict(state_dict)
        modality_indices_keys = [k for k in state_dict.keys() if k.startswith("modality_indices.")]
        assert len(modality_indices_keys) == len(self.cfg.modalities) - 1  # shared modality is not included
        self.modality_indices = {key.split(".", 1)[1]: state_dict[key] for key in modality_indices_keys}

    @torch.no_grad()
    def transform_to_unit_decoder_norm(self):
        for modality in self.cfg.modalities.keys():
            self._transform_to_unit_decoder_norm(self.encoder[modality], self.decoder[modality])

    @torch.no_grad()
    def standardize_parameters_of_dataset_norm(self, dataset_average_activation_norm: dict[str, float] | None):
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

        for modality in self.cfg.modalities.keys():
            self.encoder[modality].bias.data = self.encoder[modality].bias.data / input_norm_factor
            if self.cfg.use_decoder_bias:
                self.decoder[modality].bias.data = self.decoder[modality].bias.data / output_norm_factor
            self.decoder[modality].weight.data = (
                self.decoder[modality].weight.data * input_norm_factor / output_norm_factor
            )
        self.cfg.norm_activation = "inference"

    @torch.no_grad()
    def init_encoder_with_decoder_transpose(self, factor: float = 1.):
        for modality in self.cfg.modalities.keys():
            self._init_encoder_with_decoder_transpose(self.encoder[modality], self.decoder[modality], factor)

    @torch.no_grad()
    def log_statistics(self):
        log_dict = {}
        for modality in self.cfg.modalities.keys():
            log_dict[f"metrics/{modality}.encoder_norm"] = self._encoder_norm(self.encoder[modality]).mean().item()
            log_dict[f"metrics/{modality}.encoder_bias_norm"] = self.encoder[modality].bias.norm().item()
            log_dict[f"metrics/{modality}.decoder_norm"] = self._decoder_norm(self.decoder[modality]).mean().item()
            if self.cfg.use_decoder_bias:
                log_dict[f"metrics/{modality}.decoder_bias_norm"] = self.decoder[modality].bias.norm().item()
        if "topk" in self.cfg.act_fn:
            log_dict["sparsity/k"] = self.current_k
        else:
            log_dict["sparsity/l1_coefficient"] = self.current_l1_coefficient
        return log_dict

    def get_modality_index(self) -> dict[str, tuple[int, int]]:
        """Get the mapping from modality names to their index ranges in the feature space.

        Returns:
            dict[str, tuple[int, int]]: A dictionary mapping modality names (e.g. 'text', 'image')
                to tuples of (start_idx, end_idx) that define the index range in the feature space.
                The shared modality should be the last one.
        """
        return self.modality_index

    def get_modality_token_mask(
        self,
        tokens: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        modality: str,
    ) -> Union[Float[torch.Tensor, "batch d_model"], Float[torch.Tensor, "batch seq_len d_model"]]:
        """Get the activation of a specific modality.

        Args:
            activation: The activation tensor to be masked
            tokens: The token tensor to use for masking
            modality: The name of the modality to get the activation for

        Returns:
            The activation of the specified modality. The shape is the same as the input activation.
        """
        activation_mask = torch.isin(tokens, self.modality_indices[modality])
        return activation_mask

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        return_hidden_pre: Literal[False] = False,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch d_sae"],
        Float[torch.Tensor, "batch seq_len d_sae"],
    ]: ...

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
        dict[str, Union[Float[torch.Tensor, "batch d_sae"], Float[torch.Tensor, "batch seq_len d_sae"]]],
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
        """Encodes input tensors into sparse feature representations for each modality.

        The encoding process:
        1. Separates input by modality using token masks
        2. Applies modality-specific encoding to each modality activation ('shared' modality included)
        3. Combines modality-specific and shared features

        Args:
            x: Input tensor to encode
            return_hidden_pre: If True, returns both feature activations and pre-activation values
            **kwargs: Must contain 'tokens' for modality masking

        Returns:
            Either feature activations alone, or tuple of (feature_acts, hidden_pre) if return_hidden_pre=True
        """
        assert "tokens" in kwargs
        tokens = kwargs["tokens"]
        feature_acts = torch.zeros(x.shape[0], self.cfg.d_sae, device=x.device, dtype=x.dtype)
        hidden_pre = torch.zeros(x.shape[0], self.cfg.d_sae, device=x.device, dtype=x.dtype)
        input_norm_factor = self.compute_norm_factor(x, hook_point=self.cfg.hook_point_in)
        x = x * input_norm_factor
        for modality, (start, end) in self.modality_index.items():
            if modality == "shared":
                # shared modality is not encoded directly but summed up during other modalities' encoding
                continue
            activation_mask = self.get_modality_token_mask(tokens, modality).unsqueeze(1)
            if self.cfg.use_decoder_bias and self.cfg.apply_decoder_bias_to_pre_encoder:
                modality_bias = (
                    self.decoder[modality].bias.to_local()  # TODO: check if this is correct # type: ignore
                    if isinstance(self.decoder[modality].bias, DTensor)
                    else self.decoder[modality].bias
                )
                shared_bias = (
                    self.decoder["shared"].bias.to_local()
                    if isinstance(self.decoder["shared"].bias, DTensor)
                    else self.decoder["shared"].bias
                )
                x = x - modality_bias - shared_bias

            hidden_pre_modality = self.encoder[modality](x)
            hidden_pre_shared = self.encoder["shared"](x)

            if self.cfg.use_glu_encoder:
                hidden_pre_modality_glu = torch.sigmoid(self.encoder_glu[modality](x))
                hidden_pre_modality = hidden_pre_modality * hidden_pre_modality_glu
                hidden_pre_shared_glu = torch.sigmoid(self.encoder_glu["shared"](x))
                hidden_pre_shared = hidden_pre_shared * hidden_pre_shared_glu

            if self.cfg.sparsity_include_decoder_norm:
                true_feature_acts_modality = hidden_pre_modality * self._decoder_norm(decoder=self.decoder[modality])
                true_feature_acts_shared = hidden_pre_shared * self._decoder_norm(decoder=self.decoder["shared"])
            else:
                true_feature_acts_modality = hidden_pre_modality
                true_feature_acts_shared = hidden_pre_shared

            true_feature_acts_concat = (
                torch.cat([true_feature_acts_modality, true_feature_acts_shared], dim=1) * activation_mask
            )
            activation_mask_concat = self.activation_function(true_feature_acts_concat)
            feature_acts_concat = true_feature_acts_concat * activation_mask_concat

            feature_acts_modality = feature_acts_concat[:, : self.cfg.modalities[modality]]
            feature_acts_shared = feature_acts_concat[:, self.cfg.modalities[modality] :]
            assert feature_acts_shared.shape[1] == self.cfg.modalities["shared"]

            feature_acts[:, start:end] += feature_acts_modality
            hidden_pre[:, start:end] += hidden_pre_modality

            shared_start, shared_end = self.modality_index["shared"]
            feature_acts[:, shared_start:shared_end] += feature_acts_shared
            hidden_pre[:, shared_start:shared_end] += hidden_pre_shared

        hidden_pre = self.hook_hidden_pre(hidden_pre)
        feature_acts = self.hook_feature_acts(feature_acts)
        # assert torch.all((feature_acts > 0).sum(-1) <= self.current_k)
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
    ]:
        reconstructed = torch.zeros(
            feature_acts.shape[0], self.cfg.d_model, device=feature_acts.device, dtype=feature_acts.dtype
        )
        for modality, (start, end) in self.modality_index.items():
            feature_acts_modality = feature_acts[:, start:end]  # batch x d_modality
            reconstructed_modality = self.decoder[modality](feature_acts_modality)  # batch x d_model
            reconstructed += reconstructed_modality
        reconstructed = self.hook_reconstructed(reconstructed)
        return reconstructed

    @torch.no_grad()
    def init_parameters(self, **kwargs):
        assert "modality_indices" in kwargs
        modality_indices: dict[str, torch.Tensor] = kwargs["modality_indices"]
        for modality in self.cfg.modalities.keys():
            torch.nn.init.kaiming_uniform_(self.encoder[modality].weight)
            torch.nn.init.kaiming_uniform_(self.decoder[modality].weight)
            torch.nn.init.zeros_(self.encoder[modality].bias)
            if self.cfg.use_decoder_bias:
                torch.nn.init.zeros_(self.decoder[modality].bias)
            if self.cfg.use_glu_encoder:
                torch.nn.init.kaiming_uniform_(self.encoder_glu[modality].weight)
                torch.nn.init.zeros_(self.encoder_glu[modality].bias)

        for modality, indices in modality_indices.items():
            self.modality_indices[modality] = indices.to(self.cfg.device)

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, **kwargs):
        cfg = MixCoderConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        return cls.from_config(cfg)

    def get_parameters(self):
        params = []
        for modality in self.cfg.modalities.keys():
            modality_params = list(self.encoder[modality].parameters()) + list(self.decoder[modality].parameters())
            if self.cfg.use_glu_encoder:
                modality_params += list(self.encoder_glu[modality].parameters())
            params.append({"params": modality_params, "modality": modality})
        return params
