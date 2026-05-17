import torch

from transformer_lens.loading_from_pretrained import get_official_model_name
from transformer_lens.pretrained.weight_conversions.evo2 import convert_evo2_weights


def test_convert_evo2_weights_unwraps_checkpoint_and_strips_prefixes():
    checkpoint = {
        "iter_12500": {
            "module.blocks.0.attn.Wqkv.weight": torch.ones(2, 3),
            "model.blocks.0.attn.Wqkv.bias": torch.zeros(3),
            "module.blocks.0.attn._extra_state": {"recipe": "keep"},
            "epoch": 12,
        }
    }

    state_dict = convert_evo2_weights(checkpoint)

    assert "blocks.0.attn.Wqkv.weight" in state_dict
    assert "blocks.0.attn.Wqkv.bias" in state_dict
    assert "blocks.0.attn._extra_state" in state_dict
    assert "epoch" not in state_dict
    assert torch.equal(state_dict["blocks.0.attn.Wqkv.weight"], torch.ones(2, 3))


def test_arcinstitute_evo2_alias_resolves():
    assert get_official_model_name("arcinstitute/evo2-7b") == "arcinstitute/evo2_7b"
