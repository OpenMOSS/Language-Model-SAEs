"""Integration tests for attribution functionality with minimal models."""

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from lm_saes.clt import CrossLayerTranscoder
from lm_saes.config import CLTConfig
from lm_saes.circuit.attribution import attribute, AttributionContext, compute_salient_logits
from lm_saes.circuit.replacement_model import ReplacementModel


class TestAttribution:
    """Test suite for attribution functionality with minimal models."""

    @pytest.fixture(scope="class")
    def device(self):
        """Device for testing."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture(scope="class")
    def transformer_config(self, device):
        """Create a minimal 2-layer HookedTransformer configuration."""
        return HookedTransformerConfig(
            n_layers=2,
            d_model=2,
            d_head=2,
            n_heads=1,
            d_mlp=4,
            d_vocab=10,
            n_ctx=5,
            act_fn="relu",
            attn_only=False,
            device=device,
        )

    @pytest.fixture(scope="class")
    def transformer_model(self, transformer_config):
        """Create a minimal HookedTransformer model."""
        model = HookedTransformer(transformer_config)
        
        # Initialize with simple values for reproducibility
        with torch.no_grad():
            # Initialize embeddings
            model.embed.W_E.data.fill_(0.1)
            model.pos_embed.W_pos.data.fill_(0.05)
            
            # Initialize attention weights
            for block in model.blocks:
                block.attn.W_Q.data.fill_(0.1)
                block.attn.W_K.data.fill_(0.1)
                block.attn.W_V.data.fill_(0.1)
                block.attn.W_O.data.fill_(0.1)
                block.attn.b_Q.data.fill_(0.01)
                block.attn.b_K.data.fill_(0.01)
                block.attn.b_V.data.fill_(0.01)
                block.attn.b_O.data.fill_(0.01)
                
                # Initialize MLP weights
                block.mlp.W_in.data.fill_(0.1)
                block.mlp.W_out.data.fill_(0.1)
                block.mlp.b_in.data.fill_(0.01)
                block.mlp.b_out.data.fill_(0.01)
                
                # Initialize layer norms
                block.ln1.w.data.fill_(1.0)
                block.ln1.b.data.fill_(0.0)
                block.ln2.w.data.fill_(1.0)
                block.ln2.b.data.fill_(0.0)
            
            # Initialize final layer norm and unembed
            model.ln_final.w.data.fill_(1.0)
            model.ln_final.b.data.fill_(0.0)
            model.unembed.W_U.data.fill_(0.1)
            model.unembed.b_U.data.fill_(0.01)
        
        return model

    @pytest.fixture(scope="class")
    def clt_config(self, device):
        """Create a CLT configuration matching the transformer."""
        return CLTConfig(
            sae_type="clt",
            d_model=2,
            expansion_factor=2,  # d_sae = 4
            hook_points_in=["blocks.0.ln2.hook_normalized", "blocks.1.ln2.hook_normalized"],
            hook_points_out=["blocks.0.hook_mlp_out", "blocks.1.hook_mlp_out"],
            use_decoder_bias=True,
            sparsity_include_decoder_norm=True,
            act_fn="relu",
            device=device,
            dtype=torch.float32,
            # Add required fields for CLT
            normalize_sae_decoder=False,
            scale_sparsity_penalty_by_decoder_norm=False,
            decoder_heuristic_init=True,
            init_encoder_as_decoder_transpose=False,
            activation_fn_str="relu",
            activation_fn_kwargs={},
        )

    @pytest.fixture(scope="class")
    def clt_model(self, clt_config):
        """Create a CLT model instance."""
        model = CrossLayerTranscoder(clt_config)
        
        # Initialize with simple values for reproducibility
        with torch.no_grad():
            # Initialize encoder weights: (n_layers, d_model, d_sae)
            model.W_E.data.fill_(0.1)
            model.b_E.data.fill_(0.01)
            
            # Initialize decoder weights for each layer
            for layer_to in range(clt_config.n_layers):
                model.W_D[layer_to].data.fill_(0.1)
                model.b_D[layer_to].data.fill_(0.01)
        
        model.init_parameters()
        return model

    @pytest.fixture(scope="class")
    def replacement_model(self, transformer_model, clt_model):
        """Create a replacement model combining transformer and CLT."""
        # Convert HookedTransformer to ReplacementModel by changing its class
        replacement_model = transformer_model
        replacement_model.__class__ = ReplacementModel
        
        # Mark as CLT model and configure
        replacement_model.is_clt = True
        replacement_model._configure_replacement_model(
            transcoders=clt_model,
            feature_input_hook="ln2.hook_normalized",
            feature_output_hook="hook_mlp_out",
        )
        
        # Add a simple tokenizer for testing
        replacement_model.tokenizer = None  # For simple tensor tests
        
        return replacement_model

    @pytest.fixture
    def simple_prompt(self):
        """Create a simple test prompt."""
        return torch.tensor([1, 2, 3], dtype=torch.long)

    def test_replacement_model_creation(self, replacement_model, clt_model):
        """Test that replacement model is created correctly."""
        assert isinstance(replacement_model, ReplacementModel)
        assert replacement_model.cfg.n_layers == 2
        assert replacement_model.cfg.d_model == 2
        assert replacement_model.d_transcoder == clt_model.cfg.d_sae
        assert replacement_model.feature_input_hook == "ln2.hook_normalized"
        assert replacement_model.feature_output_hook == "hook_mlp_out.hook_out_grad"

    def test_replacement_model_forward(self, replacement_model, simple_prompt):
        """Test that replacement model can perform forward pass."""
        with torch.no_grad():
            # Test forward pass
            output = replacement_model(simple_prompt.unsqueeze(0))
            assert output.shape == (1, 3, 10)  # (batch, seq_len, vocab_size)
            
            # Test setup_attribution
            logits, activation_matrix, error_vecs, token_vecs = replacement_model.setup_attribution(
                simple_prompt, sparse=True
            )
            
            assert logits.shape == (1, 3, 10)
            assert activation_matrix.shape == (2, 3, 4)  # (n_layers, seq_len, d_sae)
            assert error_vecs.shape == (2, 3, 2)  # (n_layers, seq_len, d_model)
            assert token_vecs.shape == (3, 2)  # (seq_len, d_model)

    def test_compute_salient_logits(self, replacement_model, simple_prompt):
        """Test computation of salient logits."""
        with torch.no_grad():
            # Get logits from model
            logits = replacement_model(simple_prompt.unsqueeze(0))
            final_logits = logits[0, -1, :]  # Last position logits
            
            # Test compute_salient_logits
            logit_idx, logit_probs, demeaned_vecs = compute_salient_logits(
                final_logits,
                replacement_model.unembed.W_U,
                max_n_logits=5,
                desired_logit_prob=0.8,
            )
            
            assert len(logit_idx) <= 5
            assert len(logit_probs) == len(logit_idx)
            assert demeaned_vecs.shape == (len(logit_idx), 2)  # (n_logits, d_model)
            assert torch.all(logit_probs >= 0)
            assert torch.all(logit_probs <= 1)

    def test_attribution_context_creation(self, replacement_model, simple_prompt):
        """Test creation of AttributionContext."""
        with torch.no_grad():
            # Setup attribution
            logits, activation_matrix, error_vecs, token_vecs = replacement_model.setup_attribution(
                simple_prompt, sparse=True
            )
            
            # Select scaled decoder vectors (dummy implementation for testing)
            decoder_vecs = torch.randn(activation_matrix._nnz(), 2)
            
            # Create AttributionContext
            ctx = AttributionContext(
                activation_matrix=activation_matrix,
                error_vectors=error_vecs,
                token_vectors=token_vecs,
                decoder_vecs=decoder_vecs,
                feature_output_hook=replacement_model.feature_output_hook,
            )
            
            assert ctx.n_layers == 2
            assert ctx._row_size == activation_matrix._nnz() + (2 + 1) * 3  # features + error + token nodes

    def test_attribution_small_example(self, replacement_model, simple_prompt):
        """Test attribution computation with a small example."""
        with torch.no_grad():
            # Run attribution with minimal parameters
            graph = attribute(
                prompt=simple_prompt,
                model=replacement_model,
                max_n_logits=3,
                desired_logit_prob=0.7,
                batch_size=4,
                max_feature_nodes=8,
                verbose=False,
            )
            
            # Verify graph properties
            assert graph.input_tokens.shape == (3,)
            assert len(graph.logit_tokens) <= 3
            assert len(graph.logit_probabilities) == len(graph.logit_tokens)
            assert graph.adjacency_matrix.shape[0] == graph.adjacency_matrix.shape[1]
            assert graph.cfg.n_layers == 2
            assert graph.cfg.d_model == 2

    def test_attribution_with_string_prompt(self, replacement_model):
        """Test attribution with string prompt (requires tokenizer)."""
        pytest.skip("No tokenizer available for string prompts in test model")

    def test_attribution_different_batch_sizes(self, replacement_model, simple_prompt):
        """Test attribution with different batch sizes."""
        for batch_size in [1, 2, 4]:
            with torch.no_grad():
                graph = attribute(
                    prompt=simple_prompt,
                    model=replacement_model,
                    max_n_logits=2,
                    desired_logit_prob=0.5,
                    batch_size=batch_size,
                    max_feature_nodes=4,
                    verbose=False,
                )
                
                # Should produce consistent results regardless of batch size
                assert graph.input_tokens.shape == (3,)
                assert len(graph.logit_tokens) <= 2

    def test_attribution_edge_cases(self, replacement_model, simple_prompt):
        """Test attribution with edge cases."""
        with torch.no_grad():
            # Test with very small max_feature_nodes
            graph = attribute(
                prompt=simple_prompt,
                model=replacement_model,
                max_n_logits=1,
                desired_logit_prob=0.3,
                batch_size=2,
                max_feature_nodes=1,
                verbose=False,
            )
            
            assert len(graph.logit_tokens) <= 1
            assert len(graph.selected_features) <= 1

    def test_attribution_gradient_flow(self, replacement_model, simple_prompt):
        """Test that attribution preserves gradient flow properties."""
        with torch.no_grad():
            # Run attribution
            graph = attribute(
                prompt=simple_prompt,
                model=replacement_model,
                max_n_logits=2,
                desired_logit_prob=0.5,
                batch_size=2,
                max_feature_nodes=4,
                verbose=False,
            )
            
            # Check that adjacency matrix has reasonable values
            adj_matrix = graph.adjacency_matrix
            assert not torch.isnan(adj_matrix).any(), "Adjacency matrix should not contain NaN"
            assert not torch.isinf(adj_matrix).any(), "Adjacency matrix should not contain Inf"
            
            # Check that some connections exist (non-zero entries)
            assert adj_matrix.abs().sum() > 0, "Adjacency matrix should have non-zero entries"

    def test_attribution_deterministic(self, replacement_model, simple_prompt):
        """Test that attribution produces deterministic results."""
        torch.manual_seed(42)
        
        with torch.no_grad():
            graph1 = attribute(
                prompt=simple_prompt,
                model=replacement_model,
                max_n_logits=2,
                desired_logit_prob=0.5,
                batch_size=2,
                max_feature_nodes=4,
                verbose=False,
            )
            
            torch.manual_seed(42)
            graph2 = attribute(
                prompt=simple_prompt,
                model=replacement_model,
                max_n_logits=2,
                desired_logit_prob=0.5,
                batch_size=2,
                max_feature_nodes=4,
                verbose=False,
            )
            
            # Results should be identical
            assert torch.allclose(graph1.adjacency_matrix, graph2.adjacency_matrix, atol=1e-6)
            assert torch.equal(graph1.input_tokens, graph2.input_tokens)
            assert torch.equal(graph1.logit_tokens, graph2.logit_tokens) 