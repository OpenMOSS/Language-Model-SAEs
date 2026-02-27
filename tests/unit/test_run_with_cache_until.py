import torch
from transformer_lens import HookedTransformer

from lm_saes.backend.run_with_cache_until import run_with_cache_until


def test_run_with_cache_until_matches_run_with_cache():
    """run_with_cache_until produces same cache as run_with_cache when stopping at last hook."""
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    tokens = torch.tensor([[1, 2, 3, 4, 5]])

    hook_points = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
    _, cache_full = model.run_with_cache(tokens, names_filter=hook_points)
    out_until, cache_until = run_with_cache_until(model, tokens, names_filter=hook_points, until=hook_points[-1])

    for name in hook_points:
        assert torch.allclose(cache_full[name], cache_until[name]), f"Mismatch at {name}"
    assert torch.allclose(cache_full[hook_points[-1]], out_until)
