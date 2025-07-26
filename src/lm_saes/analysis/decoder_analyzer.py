"""Decoder norm analysis for CrossCoder SAEs."""

from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from lm_saes.crosscoder import CrossCoder
from lm_saes.utils.distributed import DimMap
from lm_saes.utils.misc import get_mesh_dim_size


def compute_decoder_norms(sae: CrossCoder, device_mesh: DeviceMesh | None = None) -> torch.Tensor:
    """Compute decoder norms for CrossCoder SAE.

    Args:
        sae: CrossCoder SAE model
        device_mesh: Device mesh for distributed computation

    Returns:
        Decoder norms tensor
    """
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

        decoder_norms = decoder_norms.redistribute(placements=DimMap({"model": -1}).placements(device_mesh)).to_local()

    return decoder_norms


def compute_decoder_similarity_matrices(sae: CrossCoder, device_mesh: DeviceMesh | None = None) -> torch.Tensor:
    """Compute decoder similarity matrices for CrossCoder SAE.

    Args:
        sae: CrossCoder SAE model
        device_mesh: Device mesh for distributed computation

    Returns:
        Decoder similarity matrices tensor
    """
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

    return decoder_similarity_matrices


def compute_decoder_inner_product_matrices(sae: CrossCoder, device_mesh: DeviceMesh | None = None) -> torch.Tensor:
    """Compute decoder inner product matrices for CrossCoder SAE.

    Args:
        sae: CrossCoder SAE model
        device_mesh: Device mesh for distributed computation

    Returns:
        Decoder inner product matrices tensor
    """
    decoder_inner_product_matrices = sae.decoder_inner_product_matrices()

    if isinstance(decoder_inner_product_matrices, DTensor):
        assert device_mesh is not None, "Device mesh is required for DTensor decoder inner product matrices"
        if decoder_inner_product_matrices.device_mesh is not device_mesh:
            decoder_inner_product_matrices = DTensor.from_local(
                decoder_inner_product_matrices.redistribute(
                    placements=DimMap({"head": 0, "model": 0}).placements(decoder_inner_product_matrices.device_mesh)
                ).to_local(),
                device_mesh,
                placements=DimMap({"model": 0}).placements(device_mesh),
            )
            # TODO: Remove this once redistributing across device meshes is supported

        decoder_inner_product_matrices = decoder_inner_product_matrices.redistribute(
            placements=DimMap({"model": 0}).placements(device_mesh)
        ).to_local()

    return decoder_inner_product_matrices


def compute_decoder_projection_matrices(sae: CrossCoder, device_mesh: DeviceMesh | None = None) -> torch.Tensor:
    """Compute decoder projection matrices for CrossCoder SAE.

    Args:
        sae: CrossCoder SAE model
        device_mesh: Device mesh for distributed computation

    Returns:
        Decoder projection matrices tensor
    """
    decoder_projection_matrices = sae.decoder_projection_matrices()

    if isinstance(decoder_projection_matrices, DTensor):
        assert device_mesh is not None, "Device mesh is required for DTensor decoder projection matrices"
        if decoder_projection_matrices.device_mesh is not device_mesh:
            decoder_projection_matrices = DTensor.from_local(
                decoder_projection_matrices.redistribute(
                    placements=DimMap({"head": 0, "model": 0}).placements(decoder_projection_matrices.device_mesh)
                ).to_local(),
                device_mesh,
                placements=DimMap({"model": 0}).placements(device_mesh),
            )
            # TODO: Remove this once redistributing across device meshes is supported

        decoder_projection_matrices = decoder_projection_matrices.redistribute(
            placements=DimMap({"model": 0}).placements(device_mesh)
        ).to_local()

    return decoder_projection_matrices


def compute_crosscoder_decoder_metrics(
    sae: CrossCoder, act_times: torch.Tensor, device_mesh: DeviceMesh | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute all decoder metrics for CrossCoder SAE.

    Args:
        sae: CrossCoder SAE model
        act_times: Activation times tensor for validation
        device_mesh: Device mesh for distributed computation

    Returns:
        Tuple of (decoder_norms, decoder_similarity_matrices, decoder_projection_matrices)
    """
    decoder_norms = compute_decoder_norms(sae, device_mesh)
    assert decoder_norms.shape[-1] == len(act_times), (
        f"decoder_norms.shape: {decoder_norms.shape}, expected d_sae dim to match act_times length: {len(act_times)}"
    )

    decoder_similarity_matrices = compute_decoder_similarity_matrices(sae, device_mesh)
    assert decoder_similarity_matrices.shape[0] == len(act_times), (
        f"decoder_similarity_matrices.shape: {decoder_similarity_matrices.shape}, expected d_sae dim to match act_times length: {len(act_times)}"
    )

    decoder_projection_matrices = compute_decoder_projection_matrices(sae, device_mesh)
    assert decoder_projection_matrices.shape[0] == len(act_times), (
        f"decoder_projection_matrices.shape: {decoder_projection_matrices.shape}, expected d_sae dim to match act_times length: {len(act_times)}"
    )

    return decoder_norms, decoder_similarity_matrices, decoder_projection_matrices


class DecoderAnalyzer:
    """Analyzer for CrossCoder decoder norms and similarity matrices.

    This analyzer computes decoder-related metrics for CrossCoder SAEs without
    requiring additional dataset sampling.
    """

    def __init__(self):
        """Initialize the decoder analyzer."""
        pass

    def analyze(
        self,
        sae: CrossCoder,
        d_sae: int,
        device_mesh: DeviceMesh | None = None,
    ) -> list[dict[str, Any]]:
        """Analyze decoder norms and similarity matrices for CrossCoder SAE.

        Args:
            sae: CrossCoder SAE model
            d_sae: SAE dimension (total across all model parallel ranks)
            device_mesh: Device mesh for distributed computation

        Returns:
            List of per-feature dictionaries containing decoder metrics
        """
        # Create dummy act_times for validation
        d_sae_local = d_sae // get_mesh_dim_size(device_mesh, "model")
        act_times = torch.zeros((d_sae_local,), dtype=torch.long, device=sae.cfg.device)

        # Compute decoder metrics
        decoder_norms, decoder_similarity_matrices, decoder_projection_matrices = compute_crosscoder_decoder_metrics(
            sae, act_times, device_mesh
        )

        # Format results per feature
        results = []
        for i in range(d_sae_local):
            feature_result = {
                "decoder_norms": decoder_norms[:, i].tolist(),
                "decoder_similarity_matrix": decoder_similarity_matrices[i, :, :].tolist(),
                "decoder_projection_matrix": decoder_projection_matrices[i, :, :].tolist(),
            }
            results.append(feature_result)

        return results
