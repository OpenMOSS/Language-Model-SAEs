import pytest
import torch

from lm_saes import (
    CLTConfig,
    DirectLogitAttributeSettings,
    DirectLogitAttributorConfig,
    LanguageModelConfig,
    LorsaConfig,
    MongoDBConfig,
    direct_logit_attribute,
)


class TestDLA:
    @pytest.fixture
    def simple_clt_config(self):
        """a simple CLT setting for qwen3-0.6b"""
        return CLTConfig(
            d_model=1024,
            expansion_factor=2,  # d_sae = 8
            hook_points_in=["layer_0_in", "layer_1_in"],
            hook_points_out=["layer_0_out", "layer_1_out"],
            use_decoder_bias=True,
            act_fn="relu",
            apply_decoder_bias_to_pre_encoder=False,
            norm_activation="inference",  # No normalization for simplicity
            sparsity_include_decoder_norm=False,
            force_unit_decoder_norm=False,
            device="cuda",
            dtype=torch.float32,
        )

    @pytest.fixture
    def simple_lorsa_config(self):
        """a simple lorsa setting for qwen3-0.6b"""
        return LorsaConfig(
            d_qk_head=6,
            d_model=1024,
            n_qk_heads=4,
            expansion_factor=2,
            act_fn="relu",
            hook_point_in="blocks.0.ln1.hook_normalized",
            hook_point_out="blocks.0.ln1.hook_attn_out",
            rotary_base=1_000_000,
            n_ctx=10,
            skip_bos=True,
            device="cuda",
            dtype=torch.float32,
            normalization_type="RMS",
            use_post_qk_ln=True,
            rotary_dim=6,
            rotary_adjacent_pairs=False,
        )

    def test_dla_clt(self, simple_clt_config):
        """test the DLA for CLT"""
        layer_idx = 1
        settings = DirectLogitAttributeSettings(
            sae=simple_clt_config,
            sae_name="test",
            layer_idx=layer_idx,
            sae_series="test",
            model=LanguageModelConfig(
                model_name="Qwen/Qwen3-0.6B",
                device="cuda",
                dtype="torch.bfloat16",
                prepend_bos=False,
            ),
            model_name="Qwen3-0.6B",
            direct_logit_attributor=DirectLogitAttributorConfig(
                top_k=5,
            ),
            mongo=MongoDBConfig(
                mongo_db="mechinterp_test",
            ),
        )

        direct_logit_attribute(settings)

    def test_dla_lorsa(self, simple_lorsa_config):
        """test the DLA for Lorsa"""
        layer_idx = 1
        settings = DirectLogitAttributeSettings(
            sae=simple_lorsa_config,
            sae_name="test",
            layer_idx=layer_idx,
            sae_series="test",
            model=LanguageModelConfig(
                model_name="Qwen/Qwen3-0.6B",
                device="cuda",
                dtype="torch.bfloat16",
                prepend_bos=False,
            ),
            model_name="Qwen3-0.6B",
            direct_logit_attributor=DirectLogitAttributorConfig(
                top_k=5,
            ),
            mongo=MongoDBConfig(
                mongo_db="mechinterp_test",
            ),
        )

        direct_logit_attribute(settings)
