from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple, Union

import torch
from torch import nn

from lm_saes.clt import CrossLayerTranscoder
from lm_saes.sae import SparseAutoEncoder


@dataclass
class TranscoderSetConfig:
    n_layers: int
    d_sae: int
    feature_input_hook: str
    feature_output_hook: str
    scan: str | list[str] | None = None
    skip_connection: bool = False


class TranscoderSet(nn.Module):
    """
    A collection of per-layer transcoders that enable construction of a replacement model.

    We want to pack a list of PerLayerTranscoders (implemented with class SparseAutoEncoder)
    into a compatible interface with CrossLayerTranscoder, so that we can use the same interface
    to construct a ReplacementModel.

    We do this in circuits as we do not see this needed elsewhere.

    Attributes:
        transcoders: ModuleList of SparseAutoEncoder instances, one per layer
        n_layers: Total number of layers covered
        d_sae: Common feature dimension across all transcoders
        feature_input_hook: Hook point where features read from (e.g., "hook_resid_mid")
        feature_output_hook: Hook point where features write to (e.g., "hook_mlp_out")
        scan: Optional identifier to identify corresponding feature visualization
        skip_connection: Whether transcoders include learned skip connections (always False for SparseAutoEncoder)
    """

    def __init__(
        self,
        config: TranscoderSetConfig,
        transcoders: dict[int, SparseAutoEncoder],
    ):
        super().__init__()
        # Validate that we have continuous layers from 0 to max
        assert set(transcoders.keys()) == set(range(max(transcoders.keys()) + 1)), (
            f"Each layer should have a transcoder, but got transcoders for layers {set(transcoders.keys())}"
        )
        self.cfg = config
        self.transcoders = nn.ModuleList([transcoders[i] for i in range(len(transcoders))])

        # Verify all transcoders have the same d_sae
        for transcoder in self.transcoders:
            assert transcoder.cfg.d_sae == self.cfg.d_sae, (
                f"All transcoders must have the same d_sae, but got {transcoder.cfg.d_sae} != {self.cfg.d_sae}"
            )

    def __len__(self):
        return self.cfg.n_layers

    def __getitem__(self, idx: int) -> SparseAutoEncoder:
        return self.transcoders[idx]  # type: ignore

    def __iter__(self) -> Iterator[SparseAutoEncoder]:
        return iter(self.transcoders)  # type: ignore

    def apply_activation_function(self, layer_id: int, features: torch.Tensor) -> torch.Tensor:
        """Apply activation function for a specific layer.

        Args:
            layer_id: Layer index
            features: Feature activations to apply activation function to

        Returns:
            Activated features
        """
        return self.transcoders[layer_id].activation_function(features)  # type: ignore

    def encode(self, input_acts):
        return torch.stack(
            [transcoder.encode(input_acts[i]) for i, transcoder in enumerate(self.transcoders)],  # type: ignore
            dim=0,
        )

    def _get_decoder_vectors(self, layer_id: int, feature_indices: torch.Tensor) -> torch.Tensor:
        """Get decoder weight vectors for specific feature indices.

        Args:
            layer_id: Layer index
            feature_indices: Feature indices to get decoder vectors for

        Returns:
            Decoder weight vectors of shape (n_features, d_model)
        """
        transcoder = self.transcoders[layer_id]
        # W_D has shape (d_sae, d_model), so we index along the first dimension
        return transcoder.W_D[feature_indices]  # type: ignore

    def select_decoder_vectors(self, features):
        if not features.is_sparse:
            features = features.to_sparse()

        all_layer_idx, all_pos_idx, all_feat_idx = features.indices()
        all_activations = features.values()
        all_scaled_decoder_vectors = []
        for unique_layer in all_layer_idx.unique():
            layer_mask = all_layer_idx == unique_layer
            feat_idx = all_feat_idx[layer_mask]
            activations = all_activations[layer_mask]

            decoder_vectors = self._get_decoder_vectors(unique_layer.item(), feat_idx)

            # Multiply each activation by its corresponding decoder vector
            scaled_decoder_vectors = activations.unsqueeze(-1) * decoder_vectors
            all_scaled_decoder_vectors.append(scaled_decoder_vectors)

        all_scaled_decoder_vectors = torch.cat(all_scaled_decoder_vectors)
        encoder_mapping = torch.arange(features._nnz(), device=features.device)

        return all_pos_idx, all_layer_idx, all_feat_idx, all_scaled_decoder_vectors, encoder_mapping

    def decode(
        self,
        feature_acts: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Decode feature activations through all transcoders.

        Args:
            acts: Feature activations of shape (n_layers, batch, d_sae) or (n_layers, batch, seq_len, d_sae)
            **kwargs: Additional arguments passed to decode

        Returns:
            Reconstructed activations of shape (n_layers, batch, d_model) or (n_layers, batch, seq_len, d_model)
        """
        assert (
            isinstance(feature_acts, list)
            and isinstance(feature_acts[0], torch.Tensor)
            and feature_acts[0].layout == torch.sparse_coo
        ), "feature_acts must be a list of sparse tensors"

        return torch.stack(
            [transcoder.decode_coo(feature_acts[i], **kwargs) for i, transcoder in enumerate(self.transcoders)],
            dim=0,
        )

    def compute_attribution_components(
        self,
        mlp_inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extract active features and their encoder/decoder vectors for attribution.

        Args:
            mlp_inputs: (n_layers, n_pos, d_model) tensor of MLP inputs

        Returns:
            Dict containing all components needed for AttributionContext:
                - activation_matrix: Sparse (n_layers, n_pos, d_sae) activations
                - reconstruction: (n_layers, n_pos, d_model) reconstructed outputs
                - encoder_vecs: Concatenated encoder vectors for active features
                - decoder_vecs: Concatenated decoder vectors (scaled by activations)
                - encoder_to_decoder_map: Mapping from encoder to decoder indices
        """
        device = mlp_inputs.device

        reconstruction = torch.zeros_like(mlp_inputs)
        encoder_vectors = []
        decoder_vectors = []
        sparse_acts_list = []

        for layer in range(self.cfg.n_layers):
            sparse_acts, active_encoders = self.encode_sparse(mlp_inputs[layer], layer_id=layer, zero_first_pos=True)
            layer_reconstruction, active_decoders = self.decode_sparse(sparse_acts, layer_id=layer)
            reconstruction[layer] = layer_reconstruction
            encoder_vectors.append(active_encoders)
            decoder_vectors.append(active_decoders)
            sparse_acts_list.append(sparse_acts)

        activation_matrix = torch.stack(sparse_acts_list).coalesce()
        encoder_to_decoder_map = torch.arange(activation_matrix._nnz(), device=device)

        return {
            "activation_matrix": activation_matrix,
            "reconstruction": reconstruction,
            "encoder_vecs": torch.cat(encoder_vectors, dim=0),
            "decoder_vecs": torch.cat(decoder_vectors, dim=0),
            "encoder_to_decoder_map": encoder_to_decoder_map,
            "decoder_locations": activation_matrix.indices()[:2],
        }

    def encode_layer(
        self,
        x: torch.Tensor,
        layer_id: int,
        apply_activation_function: bool = True,
        return_hidden_pre: bool | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode input for a specific layer.

        Args:
            x: Input tensor of shape (batch, d_model) or (batch, seq_len, d_model)
            layer_id: Layer index
            apply_activation_function: If False, return hidden_pre instead of activated features.
                This parameter is provided for compatibility with CrossLayerTranscoder interface.
            return_hidden_pre: If True, also return pre-activation hidden states.
                If None, determined by apply_activation_function (False -> True, True -> False).
            **kwargs: Additional arguments passed to encode

        Returns:
            If return_hidden_pre is False or None and apply_activation_function is True:
                Feature activations tensor
            If return_hidden_pre is True or apply_activation_function is False:
                Tuple of (feature_acts, hidden_pre) or just hidden_pre if apply_activation_function is False
        """
        # Map apply_activation_function to return_hidden_pre for compatibility
        if return_hidden_pre is None:
            return_hidden_pre = not apply_activation_function

        result = self.transcoders[layer_id].encode(x, return_hidden_pre=return_hidden_pre, **kwargs)

        # If apply_activation_function is False, return hidden_pre instead of feature_acts
        if not apply_activation_function and isinstance(result, tuple):
            return result[1]  # Return hidden_pre
        elif not apply_activation_function:
            # Fallback: if we didn't get hidden_pre, encode again with return_hidden_pre=True
            _, hidden_pre = self.transcoders[layer_id].encode(x, return_hidden_pre=True, **kwargs)
            return hidden_pre

        return result

    def encode_sparse(
        self,
        x: torch.Tensor,
        layer_id: int,
        zero_first_pos: bool = False,
    ) -> tuple[torch.sparse.Tensor, torch.Tensor]:
        """Encode input and return sparse activations with active encoder vectors.

        Args:
            x: Input tensor of shape (n_pos, d_model) or (batch, n_pos, d_model)
            layer_id: Layer index
            zero_first_pos: If True, zero out the first position

        Returns:
            Tuple of (sparse_acts, active_encoders) where:
                - sparse_acts: Sparse tensor of activations
                - active_encoders: Encoder weight vectors for active features
        """
        transcoder = self.transcoders[layer_id]
        feature_acts = transcoder.encode(x)

        if zero_first_pos:
            if feature_acts.ndim == 2:
                feature_acts[0] = 0
            else:
                feature_acts[:, 0] = 0

        # Convert to sparse and extract active features
        sparse_acts = feature_acts.to_sparse()
        active_indices = sparse_acts.indices()[-1]  # Feature indices
        active_encoders = transcoder.W_E[:, active_indices].T  # type: ignore

        return sparse_acts, active_encoders

    def decode_sparse(
        self,
        sparse_acts: torch.sparse.Tensor,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode sparse activations and return reconstruction with active decoder vectors.

        Args:
            sparse_acts: Sparse tensor of activations
            layer_id: Layer index

        Returns:
            Tuple of (reconstruction, active_decoders) where:
                - reconstruction: Reconstructed output
                - active_decoders: Decoder weight vectors scaled by activations
        """
        transcoder = self.transcoders[layer_id]
        dense_acts = sparse_acts.to_dense()
        reconstruction = transcoder.decode(dense_acts)

        # Extract active features and their decoder vectors
        active_indices = sparse_acts.indices()[-1]  # Feature indices
        active_values = sparse_acts.values()
        decoder_vectors = transcoder.W_D[active_indices]  # type: ignore
        scaled_decoders = active_values.unsqueeze(-1) * decoder_vectors

        return reconstruction, scaled_decoders

    @property
    def b_D(self) -> torch.Tensor:
        return torch.stack([transcoder.b_D for transcoder in self.transcoders], dim=0)

    @property
    def W_D(self) -> torch.Tensor:
        return torch.stack([transcoder.W_D for transcoder in self.transcoders], dim=0)

    @property
    def W_E(self) -> torch.Tensor:
        return torch.stack([transcoder.W_E for transcoder in self.transcoders], dim=0)


def load_transcoder_set(
    transcoder_set: Union[str, List[str]],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Union[CrossLayerTranscoder, Dict[int, SparseAutoEncoder]], str, str]:
    if isinstance(transcoder_set, str):
        clt = CrossLayerTranscoder.from_pretrained(
            transcoder_set,
            device=device,
            dtype=dtype,
        )

        assert len(set(".".join(clt.cfg.hook_points_in[l].split(".")[2:]) for l in range(clt.cfg.n_layers))) == 1, (
            "CLT must have exactly one hook point position for input and output"
        )
        # feature_input_hook = ".".join(clt.cfg.hook_points_in[0].split(".")[2:])  # 2: is to remove the "blocks.L" prefix
        # feature_output_hook = ".".join(clt.cfg.hook_points_out[0].split(".")[2:])
        return clt, "mlp.hook_in", "mlp.hook_out"

    elif isinstance(transcoder_set, List):
        transcoders = {
            i: SparseAutoEncoder.from_pretrained(transcoder, device=device, dtype=dtype)
            for i, transcoder in enumerate(transcoder_set)
        }
        # Verify all transcoders have the same hook points
        first_cfg = list(transcoders.values())[0].cfg
        for transcoder in transcoders.values():
            assert transcoder.cfg.hook_point_in == first_cfg.hook_point_in, (
                f"All transcoders must have the same hook_point_in, but got "
                f"{transcoder.cfg.hook_point_in} != {first_cfg.hook_point_in}"
            )
            assert transcoder.cfg.hook_point_out == first_cfg.hook_point_out, (
                f"All transcoders must have the same hook_point_out, but got "
                f"{transcoder.cfg.hook_point_out} != {first_cfg.hook_point_out}"
            )
        feature_input_hook = ".".join(first_cfg.hook_point_in.split(".")[2:])  # 2: is to remove the "blocks.L" prefix
        feature_output_hook = ".".join(first_cfg.hook_point_out.split(".")[2:])
        return transcoders, feature_input_hook, feature_output_hook

    else:
        raise ValueError(
            f"Transcoder set {transcoder_set} is not a string (loading CLTs) or list (loading a set of transcoders)"
        )
