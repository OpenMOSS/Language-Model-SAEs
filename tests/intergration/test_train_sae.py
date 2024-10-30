import torch
from einops import rearrange
from torch.optim import Adam
from transformer_lens import HookedTransformer, HookedTransformerConfig

from lm_saes.config import SAEConfig
from lm_saes.sae import SparseAutoEncoder


def test_train_sae():
    ### Traing setup ###
    batch_size = 2
    hook_point = "blocks.0.hook_resid_pre"
    device = "cpu"
    dtype = torch.float32
    torch.manual_seed(42)

    ### Model setup ###
    model_cfg = HookedTransformerConfig(
        n_layers=2,
        d_mlp=2,
        d_model=5,
        d_head=5,
        n_heads=2,
        n_ctx=10,
        d_vocab=50,
        act_fn="relu",
    )
    model = HookedTransformer(
        cfg=model_cfg,
    )

    ### SAE setup ###
    sae_cfg = SAEConfig(
        hook_point_in=hook_point,
        expansion_factor=2,
        d_model=5,
        # top_k=5,
    )
    sae = SparseAutoEncoder.from_config(sae_cfg)

    ### Get activations ###
    tokens = torch.randint(0, 50, (batch_size, 10))
    with torch.no_grad():
        _, cache = model.run_with_cache_until(tokens, names_filter=hook_point, until=hook_point)
        batch = {
            hook_point: rearrange(
                cache[hook_point].to(dtype=dtype, device=device),
                "b l d -> (b l) d",
            )
        }

    ### Train SAE ###
    optimizer = Adam(sae.parameters(), lr=0.001)
    sae.train()
    activation_in, activation_out = batch[hook_point], batch[hook_point]
    loss, _ = sae.compute_loss(activation_in, label=activation_out)
    loss.backward()
    optimizer.step()
