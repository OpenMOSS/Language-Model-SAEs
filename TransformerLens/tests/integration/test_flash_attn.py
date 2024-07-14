import einops
import torch

from transformer_lens.components import Attention, GroupedQueryAttention
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def test_flash_attention_output_is_correct():
    """
    Verify if flash attention output is correct.
    """
    d_model = 512
    d_head = 32
    n_heads = 16
    n_ctx = 128
    n_key_value_heads = 4
    n_layers = 1
    dtype = torch.bfloat16
    device = torch.device('cuda')

    cfg_dict = {
        'use_flash_attn': False,
        'd_model': d_model,
        'd_head': d_head,
        'n_heads': n_heads,
        'n_ctx': n_ctx,
        'n_key_value_heads': n_key_value_heads,
        'n_layers': n_layers,
        'act_fn': "silu",
        'dtype': torch.bfloat16,
    }
    regular_attention_cfg = HookedTransformerConfig.from_dict(cfg_dict)
    cfg_dict['use_flash_attn'] = True
    flash_attention_cfg = HookedTransformerConfig.from_dict(cfg_dict)
    flash_gqa_attention_cfg = HookedTransformerConfig.from_dict(cfg_dict)

    regular_attention = Attention(regular_attention_cfg)

    assert not hasattr(regular_attention, 'flash_attn_func'), "AbstractAttention should not have 'flash_attn_func' if set `use_flash_attn=False`"

    flash_attention = Attention(flash_attention_cfg)

    assert hasattr(flash_attention, 'flash_attn_func'), "AbstractAttention should have 'flash_attn_func' if set `use_flash_attn=True`"

    flash_gqa_attention = GroupedQueryAttention(flash_gqa_attention_cfg)

    # Variables started with `_` mean that the GQA key/value parameters
    W_Q = torch.rand((n_heads, d_model, d_head), dtype=dtype)
    b_Q = torch.rand((n_heads, d_head), dtype=dtype)
    _W_K = torch.rand((n_key_value_heads, d_model, d_head), dtype=dtype)
    W_K = torch.repeat_interleave(_W_K, dim=0, repeats=n_heads // n_key_value_heads)
    _b_K = torch.rand((n_key_value_heads, d_head), dtype=dtype)
    b_K = torch.repeat_interleave(_b_K, dim=0, repeats=n_heads // n_key_value_heads)
    _W_V = torch.rand((n_key_value_heads, d_model, d_head), dtype=dtype)
    W_V = torch.repeat_interleave(_W_V, dim=0, repeats=n_heads // n_key_value_heads)
    _b_V = torch.rand((n_key_value_heads, d_head), dtype=dtype)
    b_V = torch.repeat_interleave(_b_V, dim=0, repeats=n_heads // n_key_value_heads)
    W_O = torch.rand((n_heads, d_head, d_model), dtype=dtype)
    b_O = torch.rand(d_model, dtype=dtype)

    regular_attention_state_dict = {
        "W_Q": W_Q,
        "b_Q": b_Q,
        "W_O": W_O,
        "b_O": b_O,
        "W_K": W_K,
        "b_K": b_K,
        "W_V": W_V,
        "b_V": b_V,
        "mask": regular_attention.state_dict()["mask"],
        "IGNORE": regular_attention.state_dict()["IGNORE"],
    }
    flash_attention_state_dict = {
        "W_Q": W_Q,
        "b_Q": b_Q,
        "W_O": W_O,
        "b_O": b_O,
        "W_K": W_K,
        "b_K": b_K,
        "W_V": W_V,
        "b_V": b_V,
        "mask": flash_attention.state_dict()["mask"],
        "IGNORE": flash_attention.state_dict()["IGNORE"],
    }
    flash_gqa_attention_state_dict = {
        "W_Q": W_Q,
        "b_Q": b_Q,
        "W_O": W_O,
        "b_O": b_O,
        "_W_K": _W_K,
        "_b_K": _b_K,
        "_W_V": _W_V,
        "_b_V": _b_V,
        "mask": flash_attention.state_dict()["mask"],
        "IGNORE": flash_attention.state_dict()["IGNORE"],
    }

    regular_attention.load_state_dict(regular_attention_state_dict)
    regular_attention.to(device)
    flash_attention.load_state_dict(flash_attention_state_dict)
    flash_attention.to(device)
    flash_gqa_attention.load_state_dict(flash_gqa_attention_state_dict)
    flash_gqa_attention.to(device)

    query_input = torch.rand((1, 5, d_model), dtype=dtype).to(device)
    key_input = torch.rand((1, 5, d_model), dtype=dtype).to(device)
    value_input = torch.rand((1, 5, d_model), dtype=dtype).to(device)

    # Test regular attention and attention with FlashAttentionV2 
    regular_attn_output = regular_attention(query_input, key_input, value_input)
    flash_attn_output = flash_attention(query_input, key_input, value_input)

    assert torch.allclose(regular_attn_output, flash_attn_output, rtol=1e-2)

    # Test FlashAttention behaves correctly when use_split_qkv_input is True
    flash_gqa_attention.cfg.use_split_qkv_input = True
    split_query_input = einops.repeat(query_input, "b n d -> b n h d", h=n_heads).clone()
    split_key_input = einops.repeat(key_input, "b n d -> b n h d", h=n_key_value_heads).clone()
    split_value_input = einops.repeat(value_input, "b n d -> b n h d", h=n_key_value_heads).clone()

    split_flash_attn_output = flash_gqa_attention(
        split_query_input, split_key_input, split_value_input
    )

    assert torch.allclose(regular_attn_output, split_flash_attn_output, rtol=1e-2)