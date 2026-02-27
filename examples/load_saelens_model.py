"""
Load a Sparse Autoencoder from HuggingFace (using SAELens format) and use it with lm-saes.

Requires: uv add "lm-saes[sae_lens]"
"""

import torch
from transformer_lens import HookedTransformer

from lm_saes.abstract_sae import AbstractSparseAutoEncoder

# Load Gemma Scope 2 SAE from HuggingFace
sae = AbstractSparseAutoEncoder.from_pretrained("gemma-scope-2-1b-pt-res-all:layer_12_width_16k_l0_small").to("cpu")

print(f"Loaded SAE: {sae.cfg}")

# Load Gemma 3 with TransformerLens
model = HookedTransformer.from_pretrained("google/gemma-3-1b-pt")
model.to("cpu")
model.eval()

prompt = "The capital of France is"
tokens = model.to_tokens(prompt)
_, cache = model.run_with_cache(tokens, names_filter=[sae.cfg.hook_point_in])
activations = cache[sae.cfg.hook_point_in]

with torch.no_grad():
    feature_acts = sae.encode(activations)
    reconstructed = sae.decode(feature_acts)

l0 = (feature_acts > 0).sum(dim=-1).float().mean()
mse = (activations.to(sae.cfg.dtype) - reconstructed).pow(2).mean()
print(f"Prompt: {prompt}")
print(f"Average L0: {l0.item():.1f}")
print(f"Reconstruction MSE: {mse.item():.6f}")
