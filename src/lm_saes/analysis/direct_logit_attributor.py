from functools import singledispatch

import einops
import torch
from transformer_lens import HookedTransformer

from lm_saes.backend import LanguageModel
from lm_saes.backend.language_model import TransformerLensLanguageModel
from lm_saes.clt import CrossLayerTranscoder
from lm_saes.config import BaseConfig
from lm_saes.crosscoder import CrossCoder
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.sae import SparseAutoEncoder


class DirectLogitAttributorConfig(BaseConfig):
    top_k: int = 10
    """ The number of top tokens to attribute to. """

    clt_layer: int | None = None
    """ Layer to analyze for CLT. Provided iff analyzing CLT. """


def _compute_logits_from_residual(residual: torch.Tensor, model: HookedTransformer) -> torch.Tensor:
    """Helper function to compute logits from a residual tensor.

    Args:
        residual: Residual tensor with shape (d_sae, d_model).
        model: The HookedTransformer model.

    Returns:
        Logits tensor with shape (d_sae, d_vocab).
    """
    residual = einops.rearrange(residual, "batch d_model -> batch 1 d_model")
    if model.cfg.normalization_type is not None:
        residual = model.ln_final(residual)
    logits = model.unembed(residual)
    logits = einops.rearrange(logits, "batch 1 d_vocab -> batch d_vocab")
    return logits


@singledispatch
def compute_logits_and_d_sae(sae, model: HookedTransformer, layer_idx: int | None = None) -> tuple[torch.Tensor, int]:
    """Compute logits and d_sae for a given SAE type.

    Args:
        sae: The SAE model.
        model: The HookedTransformer model.
        layer_idx: The layer index (required for CLT).

    Returns:
        A tuple of (logits, d_sae) where logits has shape (d_sae, d_vocab).
    """
    raise NotImplementedError(f"Unsupported SAE type: {type(sae)}")


@compute_logits_and_d_sae.register
def _(sae: CrossCoder, model: HookedTransformer, layer_idx: int | None = None) -> tuple[torch.Tensor, int]:
    """Compute logits and d_sae for CrossCoder."""
    residual = sae.W_D[-1]
    d_sae = sae.cfg.d_sae
    logits = _compute_logits_from_residual(residual, model)
    return logits, d_sae


@compute_logits_and_d_sae.register
def _(sae: SparseAutoEncoder, model: HookedTransformer, layer_idx: int | None = None) -> tuple[torch.Tensor, int]:
    """Compute logits and d_sae for SparseAutoEncoder."""
    residual = sae.W_D
    d_sae = sae.cfg.d_sae
    logits = _compute_logits_from_residual(residual, model)
    return logits, d_sae


@compute_logits_and_d_sae.register
def _(sae: LowRankSparseAttention, model: HookedTransformer, layer_idx: int | None = None) -> tuple[torch.Tensor, int]:
    """Compute logits and d_sae for LowRankSparseAttention."""
    residual = sae.W_O
    d_sae = sae.cfg.n_ov_heads
    logits = _compute_logits_from_residual(residual, model)
    return logits, d_sae


@compute_logits_and_d_sae.register
def _(sae: CrossLayerTranscoder, model: HookedTransformer, layer_idx: int | None = None) -> tuple[torch.Tensor, int]:
    """Compute logits and d_sae for CrossLayerTranscoder."""
    assert layer_idx is not None, "layer_idx is required for CrossLayerTranscoder"

    d_sae = sae.cfg.d_sae
    logits = None

    for i in range(layer_idx, sae.cfg.n_layers):
        residual = sae.W_D[i][layer_idx]
        logits_layer = _compute_logits_from_residual(residual, model)

        if logits is None:
            logits = logits_layer
        else:
            logits = logits + logits_layer

    assert logits is not None, "Logits should not be None after computation"
    return logits, d_sae


class DirectLogitAttributor:
    def __init__(self, cfg: DirectLogitAttributorConfig):
        self.cfg = cfg

    @torch.no_grad()
    def direct_logit_attribute(self, sae, model: LanguageModel, layer_idx: int | None = None):
        """Compute direct logit attribution for the given SAE.

        Args:
            sae: The SAE model to attribute.
            model: The language model backend.
            layer_idx: The layer index (required for some SAE types like CrossLayerTranscoder).

        Returns:
            A list of dictionaries containing top positive and negative logits for each feature.
        """
        assert isinstance(model, TransformerLensLanguageModel), (
            "DirectLogitAttributor only supports TransformerLensLanguageModel as the model backend"
        )
        hooked_model: HookedTransformer | None = model.model
        assert hooked_model is not None, "Model ckpt must be loaded for direct logit attribution"

        # Use singledispatch to compute logits and d_sae based on SAE type
        logits, d_sae = compute_logits_and_d_sae(sae, hooked_model, layer_idx)

        # Select the top k tokens
        top_k_logits, top_k_indices = torch.topk(logits, self.cfg.top_k, dim=-1)
        top_k_tokens = [hooked_model.to_str_tokens(top_k_indices[i]) for i in range(d_sae)]

        assert top_k_logits.shape == top_k_indices.shape == (d_sae, self.cfg.top_k), (
            f"Top k logits and indices should have shape (d_sae, top_k), but got {top_k_logits.shape} and {top_k_indices.shape}"
        )
        assert (len(top_k_tokens), len(top_k_tokens[0])) == (d_sae, self.cfg.top_k), (
            f"Top k tokens should have shape (d_sae, top_k), but got {len(top_k_tokens)} and {len(top_k_tokens[0])}"
        )

        # Select the bottom k tokens
        bottom_k_logits, bottom_k_indices = torch.topk(logits, self.cfg.top_k, dim=-1, largest=False)
        bottom_k_tokens = [hooked_model.to_str_tokens(bottom_k_indices[i]) for i in range(d_sae)]

        assert bottom_k_logits.shape == bottom_k_indices.shape == (d_sae, self.cfg.top_k), (
            f"Bottom k logits and indices should have shape (d_sae, top_k), but got {bottom_k_logits.shape} and {bottom_k_indices.shape}"
        )
        assert (len(bottom_k_tokens), len(bottom_k_tokens[0])) == (d_sae, self.cfg.top_k), (
            f"Bottom k tokens should have shape (d_sae, top_k), but got {len(bottom_k_tokens)} and {len(bottom_k_tokens[0])}"
        )

        result = [
            {
                "top_positive": [
                    {"token": token, "logit": logit} for token, logit in zip(top_k_tokens[i], top_k_logits[i].tolist())
                ],
                "top_negative": [
                    {"token": token, "logit": logit}
                    for token, logit in zip(bottom_k_tokens[i], bottom_k_logits[i].tolist())
                ],
            }
            for i in range(d_sae)
        ]
        return result
