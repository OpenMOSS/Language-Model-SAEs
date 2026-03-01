"""
Load a Transcoder from HuggingFace.
"""

import torch
from transformer_lens import HookedTransformer

from lm_saes.abstract_sae import AbstractSparseAutoEncoder

# Load Gemma Scope 2 SAE from HuggingFace
sae = AbstractSparseAutoEncoder.from_pretrained(
    "OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B:transcoder/8x/k128/layer12_transcoder_8x_k128",
    fold_activation_scale=False,
).to("cpu")

print(f"Loaded SAE: {sae.cfg}")

# Load Gemma 3 with TransformerLens
model = HookedTransformer.from_pretrained("Qwen/Qwen3-1.7B")
model.to("cpu")
model.eval()

prompt = "The capital of France is"
tokens = model.to_tokens(prompt)
_, cache = model.run_with_cache(tokens, names_filter=[sae.cfg.hook_point_in, sae.cfg.hook_point_out])
x = cache[sae.cfg.hook_point_in]
label = cache[sae.cfg.hook_point_out]

with torch.no_grad():
    feature_acts = sae.encode(x)
    reconstructed = sae.decode(feature_acts)

l0 = (feature_acts > 0).sum(dim=-1).float().mean()
mse = (x.to(sae.cfg.dtype) - reconstructed).pow(2).mean()
print(f"Prompt: {prompt}")
print(f"Average L0: {l0.item():.1f}")
print(f"Reconstruction MSE: {mse.item():.6f}")
