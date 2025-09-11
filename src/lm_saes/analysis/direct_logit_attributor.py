import einops
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.backend import LanguageModel
from lm_saes.backend.language_model import TransformerLensLanguageModel
from lm_saes.config import DirectLogitAttributorConfig
from lm_saes.crosscoder import CrossCoder
from lm_saes.sae import SparseAutoEncoder
from lm_saes import CrossLayerTranscoder

from lm_saes import ReplacementModel, LanguageModelConfig, CrossLayerTranscoder, LowRankSparseAttention



class DirectLogitAttributor:
    def __init__(self, cfg: DirectLogitAttributorConfig):
        self.cfg = cfg
    
    @torch.no_grad()
    def direct_logit_attribute(self, sae, model:LanguageModel, layer_idx=None):
        # model = self.model
        assert isinstance(model, TransformerLensLanguageModel), (
            "DirectLogitAttributor only supports TransformerLensLanguageModel as the model backend"
        )
        model: HookedTransformer | None = model.model
        assert model is not None, "Model ckpt must be loaded for direct logit attribution"

        if isinstance(sae, CrossCoder):
            residual = sae.W_D[-1]
            d_sae = sae.cfg.d_sae
        elif isinstance(sae, SparseAutoEncoder):
            residual = sae.W_D
            d_sae = sae.cfg.d_sae
        elif isinstance(sae, LowRankSparseAttention):
            residual = sae.W_O
            d_sae = sae.cfg.n_ov_heads
        elif isinstance(sae, CrossLayerTranscoder):
            d_sae = sae.cfg.d_sae
        else:
            raise ValueError(f"Unsupported SAE type: {type(sae)}")

        if not isinstance(sae, CrossLayerTranscoder):
            residual = einops.rearrange(residual, "batch d_model -> batch 1 d_model")  # Add a context dimension

            if model.cfg.normalization_type is not None:
                residual = model.ln_final(residual)  # [batch, pos, d_model]
            logits = model.unembed(residual)  # [batch, pos, d_vocab]
            logits = einops.rearrange(logits, "batch 1 d_vocab -> batch d_vocab")  # Remove the context dimension
        else:
            with torch.no_grad():
                logits = None
                for i in range(layer_idx, sae.cfg.n_layers):
                    residual = sae.W_D[i][layer_idx]
                    
                    residual = einops.rearrange(residual, "batch d_model -> batch 1 d_model")  # Add a context dimension
                    # print(f'{residual.shape=}')
                
                    if model.cfg.normalization_type is not None:
                        residual = model.ln_final(residual)  # [batch, pos, d_model]
                    logits_ = model.unembed(residual)  # [batch, pos, d_vocab]
                    if logits is None:
                        logits = einops.rearrange(logits_, "batch 1 d_vocab -> batch d_vocab")  # Remove the context dimension
                    else:
                        logits = logits + einops.rearrange(logits_, "batch 1 d_vocab -> batch d_vocab")
                    del logits_, residual
                    torch.cuda.empty_cache()
        
        # Select the top k tokens
        top_k_logits, top_k_indices = torch.topk(logits, self.cfg.top_k, dim=-1)
        top_k_tokens = [model.to_str_tokens(top_k_indices[i]) for i in range(d_sae)]

        assert top_k_logits.shape == top_k_indices.shape == (d_sae, self.cfg.top_k), (
            f"Top k logits and indices should have shape (d_sae, top_k), but got {top_k_logits.shape} and {top_k_indices.shape}"
        )
        assert (len(top_k_tokens), len(top_k_tokens[0])) == (d_sae, self.cfg.top_k), (
            f"Top k tokens should have shape (d_sae, top_k), but got {len(top_k_tokens)} and {len(top_k_tokens[0])}"
        )

        # Select the bottom k tokens
        bottom_k_logits, bottom_k_indices = torch.topk(logits, self.cfg.top_k, dim=-1, largest=False)
        bottom_k_tokens = [model.to_str_tokens(bottom_k_indices[i]) for i in range(d_sae)]

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
