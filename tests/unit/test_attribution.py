import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from lm_saes import CLTConfig, CrossLayerTranscoder, LorsaConfig, LowRankSparseAttention, ReplacementModel
from lm_saes.circuit.attribution import attribute
from lm_saes.circuit.graph import compute_influence, normalize_matrix


class TestAttribution:
    """Test suite for attribution graph with lorsa and clt."""

    @pytest.fixture
    def clt_model(self):
        simple_clt_config = CLTConfig(
            sae_type="clt",
            d_model=4,
            expansion_factor=2,  # d_sae = 8
            hook_points_in=["layer_0_in", "layer_1_in"],
            hook_points_out=["layer_0_out", "layer_1_out"],
            use_decoder_bias=True,
            act_fn="relu",
            apply_decoder_bias_to_pre_encoder=False,
            norm_activation="inference",  # No normalization for simplicity
            sparsity_include_decoder_norm=False,
            force_unit_decoder_norm=False,
            device="cpu",
            dtype=torch.float32,
        )

        _clt_model = CrossLayerTranscoder(simple_clt_config)

        # Initialize model weights
        with torch.no_grad():
            for param in _clt_model.parameters():
                if param.requires_grad:
                    torch.nn.init.uniform_(param, a=-0.1, b=0.1)

        return _clt_model

    @pytest.fixture
    def lorsa_models(self):
        simple_lorsa_configs = [
            LorsaConfig(
                d_qk_head=6,
                d_model=4,
                n_qk_heads=4,
                expansion_factor=2,
                act_fn="relu",
                hook_point_in="blocks.0.ln1.hook_normalized",
                hook_point_out="blocks.0.ln1.hook_attn_out",
                rotary_base=1_000_000,
                n_ctx=10,
                skip_bos=True,
                device="cpu",
                dtype=torch.float32,
                normalization_type="RMS",
                use_post_qk_ln=True,
                rotary_dim=6,
                rotary_adjacent_pairs=False,
            ),
            LorsaConfig(
                d_qk_head=6,
                d_model=4,
                n_qk_heads=4,
                expansion_factor=2,
                act_fn="relu",
                hook_point_in="blocks.1.ln1.hook_normalized",
                hook_point_out="blocks.1.ln1.hook_attn_out",
                rotary_base=1_000_000,
                n_ctx=10,
                skip_bos=True,
                device="cpu",
                dtype=torch.float32,
                normalization_type="RMS",
                use_post_qk_ln=True,
                rotary_dim=6,
                rotary_adjacent_pairs=False,
            ),
        ]
        _lorsa_models = [
            LowRankSparseAttention(simple_lorsa_configs[0]),
            LowRankSparseAttention(simple_lorsa_configs[1]),
        ]
        for lorsa in _lorsa_models:
            lorsa.cfg.skip_bos = False

        # Initialize model weights
        with torch.no_grad():
            for lorsa in _lorsa_models:
                for param in lorsa.parameters():
                    if param.requires_grad:
                        torch.nn.init.uniform_(param, a=-0.1, b=0.1)
        return _lorsa_models

    @pytest.fixture
    def model(self, clt_model, lorsa_models):
        transformer_cfg = HookedTransformerConfig(
            n_layers=2,
            d_model=4,
            d_head=2,
            n_heads=1,
            d_mlp=4,
            d_vocab=10,
            n_ctx=50,
            act_fn="relu",
            device="cpu",
            positional_embedding_type="rotary",
            rotary_base=1_000_000,
            rotary_dim=2,
            rotary_adjacent_pairs=False,
            tokenizer_name="Qwen3-0.6B",
        )
        _model = HookedTransformer(transformer_cfg)
        # Initialize model weights
        with torch.no_grad():
            for param in _model.parameters():
                if param.requires_grad:
                    torch.nn.init.uniform_(param, a=-0.1, b=0.1)

        _model.__class__ = ReplacementModel
        _model._configure_replacement_model(
            transcoders=clt_model,
            lorsas=lorsa_models,
            mlp_input_hook="mlp.hook_in",
            mlp_output_hook="mlp.hook_out",
            attn_input_hook="attn.hook_in",
            attn_output_hook="attn.hook_out",
        )
        return _model

    @pytest.fixture
    def prompt(self):
        return torch.tensor([[1, 2, 3, 4, 5, 6]])

    @pytest.fixture
    def graph(self, model, prompt):
        return attribute(prompt, model, max_n_logits=5, desired_logit_prob=0.8, batch_size=1)

    def test_intervention(self, graph, model):
        n_features = graph.selected_features.size(0)

        s = graph.input_tokens
        adjacency_matrix = graph.adjacency_matrix.to("cpu")
        logit_tokens = graph.logit_tokens.to("cpu")

        logits, activation_cache, lorsa_attention_pattern = model.get_activations(s)
        logits = logits.squeeze(0)

        relevant_logits = logits[-1, logit_tokens]
        demeaned_relevant_logits = relevant_logits - logits[-1].mean()

        def verify_intervention(
            expected_effects,
            layer: int | torch.Tensor,
            pos: int | torch.Tensor,
            feature_idx: int | torch.Tensor,
            new_activation: float | torch.Tensor,
            sae_type: str,
            logit_atol=5e-4,
            logit_rtol=1e-5,
        ):
            new_logits, new_activation_cache = model.feature_intervention(
                s,
                [(layer, pos, feature_idx, new_activation, sae_type)],
                constrained_layers=range(model.cfg.n_layers),
                apply_activation_function=False,
            )
            new_logits = new_logits.squeeze(0)

            new_relevant_logits = new_logits[-1, logit_tokens]
            new_demeaned_relevant_logits = new_relevant_logits - new_logits[-1].mean()

            expected_logit_difference = expected_effects[-len(logit_tokens) :]

            assert torch.allclose(
                new_demeaned_relevant_logits,
                demeaned_relevant_logits - expected_logit_difference,
                atol=logit_atol,
                rtol=logit_rtol,
            )

        for node_idx in range(n_features):
            orig_feature_idx = graph.selected_features[node_idx]
            is_lorsa = orig_feature_idx < len(graph.lorsa_active_features)
            if is_lorsa:
                layer, pos, feat_idx = graph.lorsa_active_features[orig_feature_idx].tolist()
            else:
                orig_feature_idx = orig_feature_idx - len(graph.lorsa_active_features)
                layer, pos, feat_idx = graph.clt_active_features[orig_feature_idx].tolist()

            if is_lorsa:
                new_activation = 0
                expected_effects = adjacency_matrix[:, node_idx]
                verify_intervention(
                    expected_effects=expected_effects,
                    layer=layer,
                    pos=pos,
                    feature_idx=feat_idx,
                    new_activation=new_activation,
                    sae_type="lorsa",
                )
            else:
                new_activation = 0
                expected_effects = adjacency_matrix[:, node_idx]
                verify_intervention(
                    expected_effects=expected_effects,
                    layer=layer,
                    pos=pos,
                    feature_idx=feat_idx,
                    new_activation=new_activation,
                    sae_type="clt",
                )

    def test_influence(self, graph):
        """test the calculation of node influence"""
        n_logits = len(graph.logit_tokens)
        n_features = len(graph.selected_features)
        layers = graph.cfg.n_layers
        error_end_idx = n_features + 2 * graph.n_pos * layers
        token_end_idx = error_end_idx + len(graph.input_tokens)

        logit_weights = torch.zeros(graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device)
        logit_weights[-n_logits:] = graph.logit_probabilities
        normalized_matrix = normalize_matrix(graph.adjacency_matrix)
        node_influence = compute_influence(normalized_matrix, logit_weights)
        emb_err_influence = torch.sum(node_influence[n_features:token_end_idx])
        logit_influence = torch.sum(graph.logit_probabilities)
        assert torch.allclose(emb_err_influence, logit_influence, rtol=2e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
