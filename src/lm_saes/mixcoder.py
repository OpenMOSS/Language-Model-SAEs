from typing import Any, Literal, Union, overload

import torch
from jaxtyping import Float, Int
from typing_extensions import override

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
            encoder (MutableMapping[str, nn.Linear]): Dictionary of encoders for each modality
            decoder (MutableMapping[str, nn.Linear]): Dictionary of decoders for each modality
            encoder_glu (MutableMapping[str, nn.Linear]): Optional GLU gates for encoders
        """
        super().__init__(cfg)

        # initialize new encoder and decoder
        self.cfg = cfg
        self.modality_index = {}
        index = 0
        for modality, d_modality in cfg.modalities.items():
            self.modality_index[modality] = (index, index + d_modality)
            index += d_modality

        assert self.cfg.act_fn == "jumprelu", "Only jumprelu is supported for MixCoder"
        assert self.log_jumprelu_threshold is not None, "log_jumprelu_threshold must be set for MixCoder"

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
        modality_indices: Union[Int[torch.Tensor, "batch d_model"], Int[torch.Tensor, "batch seq_len d_model"]],
        modality: str,
    ) -> Union[Float[torch.Tensor, "batch d_model"], Float[torch.Tensor, "batch seq_len d_model"]]:
        if modality == "shared":
            return torch.ones_like(modality_indices, dtype=torch.bool)
        index = self.cfg.modality_names.index(modality)
        return modality_indices == index

    def _get_penalty_mask(self, feature_acts: torch.Tensor, modality_focus: str) -> torch.Tensor:
        # given a modality, return a mask that is 1 for the modality and shared modality and penalty_coefficient for all other modalities
        penalty_mask = torch.ones_like(feature_acts)
        for modality, (start, end) in self.modality_index.items():
            if modality == "shared" or modality == modality_focus:
                continue
            penalty_mask[start:end] = self.cfg.penalty_coefficient[modality]
        return penalty_mask

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

    def compute_loss(  # type: ignore
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
    ]:
        x, label = batch[self.cfg.hook_point_in], batch[self.cfg.hook_point_out]

        feature_acts, hidden_pre = self.encode(x, return_hidden_pre=True, **kwargs)
        reconstructed = self.decode(feature_acts, **kwargs)

        loss_list = []
        loss_list = []
        loss_dict = {}

        for modality, (start, end) in self.modality_index.items():
            if modality == "shared":
                continue
            token_mask = self.get_modality_token_mask(modality_indices=batch["modalities"], modality=modality)
            token_num = token_mask.sum(dim=-1)
            l_rec = (reconstructed[token_mask] - label[token_mask]).pow(2)
            if use_batch_norm_mse:
                l_rec = (
                    l_rec
                    / (label[token_mask] - label[token_mask].mean(dim=0, keepdim=True))
                    .pow(2)
                    .sum(dim=-1, keepdim=True)
                    .clamp(min=1e-8)
                    .sqrt()
                )
            l_rec = l_rec * self.cfg.loss_weights[modality]
            loss_list.append(l_rec)
            loss_dict[f"{modality}_loss"] = l_rec.mean()
            loss_dict[f"{modality}_token_num"] = token_num.item()

        l_rec = torch.concat(loss_list)
        loss = l_rec.sum(dim=-1).mean()
        loss_dict["l_rec"] = l_rec

        assert self.current_l1_coefficient is not None, "current_l1_coefficient is not initialized"
        penalty_feature_acts = torch.zeros_like(feature_acts)
        for modality, (start, end) in self.modality_index.items():
            if modality == "shared":
                continue
            token_mask = self.get_modality_token_mask(modality_indices=batch["modalities"], modality=modality)
            penalty_feature_acts[token_mask] = feature_acts[token_mask] * self._get_penalty_mask(
                feature_acts[token_mask], modality
            )
        if sparsity_loss_type is not None:
            if sparsity_loss_type == "power":
                l_s = torch.norm(penalty_feature_acts * self.decoder_norm(), p=p, dim=-1)
            elif sparsity_loss_type == "tanh":
                l_s = torch.tanh(tanh_stretch_coefficient * penalty_feature_acts * self.decoder_norm()).sum(dim=-1)
            else:
                raise ValueError(f"sparsity_loss_type f{sparsity_loss_type} not supported.")
            loss_dict["l_s"] = l1_coefficient * l_s.mean()
            loss = loss + l1_coefficient * l_s.mean()

        if return_aux_data:
            aux_data = {
                "feature_acts": feature_acts,
                "reconstructed": reconstructed,
                "hidden_pre": hidden_pre,
            }
            return loss, (loss_dict, aux_data)

        return loss

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, **kwargs):
        cfg = MixCoderConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        return cls.from_config(cfg)

    @override
    def prepare_input(self, batch: dict[str, torch.Tensor], **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        return batch[self.cfg.hook_point_in], {"modalities": batch["modalities"]}
