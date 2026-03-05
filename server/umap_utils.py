from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.sae import SparseAutoEncoder


@lru_cache(maxsize=16)
def get_sae_decoder_weights_umap(
    sae: AbstractSparseAutoEncoder,
) -> dict[str, Any]:
    """Compute and cache a 2D UMAP embedding over SAE decoder weights.

    This helper is type-aware:

    - For Transcoder-style SAEs (``SparseAutoEncoder`` / CLT / etc.), the decoder
      matrix is ``W_D``.
    - For Lorsa (``LowRankSparseAttention``), the decoder matrix is ``W_O``.

    The returned embedding is purely over the decoder rows; it does not depend
    on the particular layer or feature activations.
    """
    try:
        import umap  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("umap-learn is not installed; please install it to use the UMAP endpoint.") from exc

    # Select the correct decoder matrix depending on SAE type.
    if isinstance(sae, LowRankSparseAttention):
        # Lorsa: decoder is W_O with shape [n_features, d_model]
        decoder_weights = sae.W_O
    else:
        # Transcoder / vanilla SAE / CLT: decoder is W_D
        # Many subclasses (e.g. CLT) expose W_D as either a Tensor or nn.ParameterList.
        decoder_weights = getattr(sae, "W_D", None)
        if decoder_weights is None:
            raise RuntimeError("SAE does not expose a decoder weight matrix named 'W_D'.")

    # Handle potential distributed tensor case or ParameterList.
    if isinstance(decoder_weights, torch.distributed.tensor.DTensor):
        decoder_weights_local = decoder_weights.to_local()
    elif isinstance(decoder_weights, torch.nn.ParameterList):
        # For multi-layer decoders (e.g. CLT), flatten all layers along feature dimension.
        decoder_weights_local = torch.cat(
            [p.detach().reshape(-1, p.shape[-1]) for p in decoder_weights],
            dim=0,
        )
    else:
        decoder_weights_local = decoder_weights

    decoder_weights_np = decoder_weights_local.detach().cpu().numpy()

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    embedding = reducer.fit_transform(decoder_weights_np)

    feature_ids: list[int] = list(range(decoder_weights_np.shape[0]))
    return {"embedding": embedding, "feature_ids": feature_ids}


def compute_decoder_weights_umap_for_name(
    name: str,
    get_sae_fn,
) -> dict[str, Any]:
    """Resolve SAE by name and compute its decoder-weights UMAP.

    This is a thin wrapper so that ``app.py`` does not need to know how
    decoder weights are stored for different SAE types.
    """
    sae: AbstractSparseAutoEncoder = get_sae_fn(name)
    return get_sae_decoder_weights_umap(sae)

