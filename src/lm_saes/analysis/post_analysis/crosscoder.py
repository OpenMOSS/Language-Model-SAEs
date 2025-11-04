"""Post-analysis processor for CrossCoder SAE.

This module provides post-processing functionality specific to CrossCoder SAEs,
including handling of decoder norms, similarity matrices, and inner product matrices.
"""

from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.activation.factory import ActivationFactory
from lm_saes.crosscoder import CrossCoder
from lm_saes.utils.discrete import KeyedDiscreteMapper
from lm_saes.utils.distributed import DimMap

from .base import PostAnalysisProcessor, register_post_analysis_processor


class CrossCoderPostAnalysisProcessor(PostAnalysisProcessor):
    """Post-analysis processor for CrossCoder SAE.

    This processor handles CrossCoder-specific analysis results including:
    - Decoder norms
    - Decoder similarity matrices
    - Decoder inner product matrices
    """

    def _process_tensors(
        self,
        sae: AbstractSparseAutoEncoder,
        act_times: torch.Tensor,
        n_analyzed_tokens: int,
        max_feature_acts: torch.Tensor,
        sample_result: dict[str, dict[str, torch.Tensor]],
        mapper: KeyedDiscreteMapper,
        device_mesh: DeviceMesh | None = None,
        activation_factory: ActivationFactory | None = None,
        activation_factory_process_kwargs: dict[str, Any] = {},
    ) -> tuple[dict[str, dict[str, torch.Tensor]], list[dict[str, Any]]]:
        """Format the analysis results into the final per-feature format for CrossCoder.

        Args:
            sae: The sparse autoencoder model
            act_times: Tensor of activation times for each feature
            n_analyzed_tokens: Number of tokens analyzed
            max_feature_acts: Tensor of maximum activation values for each feature
            sample_result: Dictionary of sampling results
            mapper: KeyedDiscreteMapper for encoding/decoding metadata
            device_mesh: Device mesh for distributed tensors

        Returns:
            List of dictionaries containing per-feature analysis results
        """
        assert isinstance(sae, CrossCoder), "CrossCoderPostAnalysisProcessor only supports CrossCoder SAE"
        decoder_norms = sae.decoder_norm()
        if isinstance(decoder_norms, DTensor):
            assert device_mesh is not None, "Device mesh is required for DTensor decoder norms"
            if decoder_norms.device_mesh is not device_mesh:
                decoder_norms = DTensor.from_local(
                    decoder_norms.redistribute(
                        placements=DimMap({"head": -1, "model": -1}).placements(decoder_norms.device_mesh)
                    ).to_local(),
                    device_mesh,
                    placements=DimMap({"model": -1}).placements(device_mesh),
                )
                # TODO: Remove this once redistributing across device meshes is supported

            decoder_norms = decoder_norms.redistribute(
                placements=DimMap({"model": -1}).placements(device_mesh)
            ).to_local()
        assert decoder_norms.shape[-1] == len(act_times), (
            f"decoder_norms.shape: {decoder_norms.shape}, expected d_sae dim to match act_times length: {len(act_times)}"
        )

        decoder_similarity_matrices = sae.decoder_similarity_matrices()
        if isinstance(decoder_similarity_matrices, DTensor):
            assert device_mesh is not None, "Device mesh is required for DTensor decoder similarity matrices"
            if decoder_similarity_matrices.device_mesh is not device_mesh:
                decoder_similarity_matrices = DTensor.from_local(
                    decoder_similarity_matrices.redistribute(
                        placements=DimMap({"head": 0, "model": 0}).placements(decoder_similarity_matrices.device_mesh)
                    ).to_local(),
                    device_mesh,
                    placements=DimMap({"model": 0}).placements(device_mesh),
                )
                # TODO: Remove this once redistributing across device meshes is supported

            decoder_similarity_matrices = decoder_similarity_matrices.redistribute(
                placements=DimMap({"model": 0}).placements(device_mesh)
            ).to_local()
        assert decoder_similarity_matrices.shape[0] == len(act_times), (
            f"decoder_similarity_matrices.shape: {decoder_similarity_matrices.shape}, expected d_sae dim to match act_times length: {len(act_times)}"
        )

        decoder_inner_product_matrices = sae.decoder_inner_product_matrices()
        if isinstance(decoder_inner_product_matrices, DTensor):
            assert device_mesh is not None, "Device mesh is required for DTensor decoder inner product matrices"
            if decoder_inner_product_matrices.device_mesh is not device_mesh:
                decoder_inner_product_matrices = DTensor.from_local(
                    decoder_inner_product_matrices.redistribute(
                        placements=DimMap({"head": 0, "model": 0}).placements(
                            decoder_inner_product_matrices.device_mesh
                        )
                    ).to_local(),
                    device_mesh,
                    placements=DimMap({"model": 0}).placements(device_mesh),
                )
                # TODO: Remove this once redistributing across device meshes is supported

            decoder_inner_product_matrices = decoder_inner_product_matrices.redistribute(
                placements=DimMap({"model": 0}).placements(device_mesh)
            ).to_local()
        assert decoder_inner_product_matrices.shape[0] == len(act_times), (
            f"decoder_inner_product_matrices.shape: {decoder_inner_product_matrices.shape}, expected d_sae dim to match act_times length: {len(act_times)}"
        )

        return sample_result, [
            {
                "decoder_norms": decoder_norms[:, i].tolist(),
                "decoder_similarity_matrices": decoder_similarity_matrices[i, :, :].tolist(),
                "decoder_inner_product_matrices": decoder_inner_product_matrices[i, :, :].tolist(),
            }
            for i in range(len(act_times))
        ]


# Register the processor for CrossCoder SAE type
register_post_analysis_processor("crosscoder", CrossCoderPostAnalysisProcessor)
