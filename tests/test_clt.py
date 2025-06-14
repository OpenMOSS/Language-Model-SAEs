"""Unit tests for Cross Layer Transcoder (CLT) implementation."""

import pytest
import torch
import torch.nn as nn

from lm_saes.clt import CrossLayerTranscoder
from lm_saes.config import CLTConfig


class TestCrossLayerTranscoder:
    """Test suite for CrossLayerTranscoder."""

    @pytest.fixture
    def simple_config(self):
        """Create a simple CLT configuration for testing."""
        return CLTConfig(
            sae_type="clt",
            d_model=4,
            expansion_factor=2,  # d_sae = 8
            hook_points_in=["layer_0_in", "layer_1_in"],
            hook_points_out=["layer_0_out", "layer_1_out"],
            use_decoder_bias=True,
            act_fn="relu",
            apply_decoder_bias_to_pre_encoder=False,
            norm_activation="inference",  # No normalization for simplicity
            sparsity_include_decoder_norm=True,
            force_unit_decoder_norm=False,
            device="cpu",
            dtype=torch.float32,
        )

    @pytest.fixture
    def clt_model(self, simple_config):
        """Create a CLT model instance."""
        return CrossLayerTranscoder(simple_config)

    @pytest.fixture
    def simple_batch(self):
        """Create simple test data with only 1's and 0's."""
        batch_size = 2
        seq_len = 3
        d_model = 4
        
        # Create simple binary data
        batch = {
            "layer_0_in": torch.ones(batch_size, seq_len, d_model),
            "layer_1_in": torch.zeros(batch_size, seq_len, d_model),
            "layer_0_out": torch.ones(batch_size, seq_len, d_model),
            "layer_1_out": torch.zeros(batch_size, seq_len, d_model),
        }
        return batch

    def test_clt_initialization(self, clt_model, simple_config):
        """Test that CLT model initializes correctly."""
        # Check basic properties
        assert clt_model.cfg.n_layers == 2
        assert clt_model.cfg.d_model == 4
        assert clt_model.cfg.d_sae == 8
        assert clt_model.n_decoder_matrices == 3  # 2*(2+1)/2 = 3
        
        # Check parameter shapes
        assert clt_model.W_E.shape == (2, 4, 8)  # (n_layers, d_model, d_sae)
        assert clt_model.b_E.shape == (2, 8)     # (n_layers, d_sae)
        assert clt_model.W_D.shape == (3, 8, 4)  # (n_decoder_matrices, d_sae, d_model)
        assert clt_model.b_D.shape == (2, 4)     # (n_layers, d_model)

    def test_prepare_input_and_label(self, clt_model, simple_batch):
        """Test input and label preparation."""
        # Test input preparation (uses hook_points_in)
        x, kwargs = clt_model.prepare_input(simple_batch)
        assert x.shape == (2, 3, 2, 4)  # (batch_size, seq_len, n_layers, d_model)
        assert torch.allclose(x[:, :, 0, :], torch.ones(2, 3, 4))  # layer_0_in
        assert torch.allclose(x[:, :, 1, :], torch.zeros(2, 3, 4))  # layer_1_in
        
        # Test label preparation (uses hook_points_out)
        labels = clt_model.prepare_label(simple_batch)
        assert labels.shape == (2, 3, 2, 4)  # (batch_size, seq_len, n_layers, d_model)
        assert torch.allclose(labels[:, :, 0, :], torch.ones(2, 3, 4))  # layer_0_out
        assert torch.allclose(labels[:, :, 1, :], torch.zeros(2, 3, 4))  # layer_1_out

    def test_encoder_decoder_shapes(self, clt_model, simple_batch):
        """Test encoder and decoder forward pass shapes."""
        x, _ = clt_model.prepare_input(simple_batch)
        
        # Test encoding
        feature_acts = clt_model.encode(x)
        assert feature_acts.shape == (2, 3, 2, 8)  # (batch_size, seq_len, n_layers, d_sae)
        
        # Test encoding with hidden_pre
        feature_acts, hidden_pre = clt_model.encode(x, return_hidden_pre=True)
        assert feature_acts.shape == (2, 3, 2, 8)
        assert hidden_pre.shape == (2, 3, 2, 8)
        
        # Test decoding
        reconstructed = clt_model.decode(feature_acts)
        assert reconstructed.shape == (2, 3, 2, 4)  # (batch_size, seq_len, n_layers, d_model)

    def test_forward_pass(self, clt_model, simple_batch):
        """Test full forward pass."""
        x, _ = clt_model.prepare_input(simple_batch)
        
        # Test forward method
        reconstructed = clt_model.forward(x)
        assert reconstructed.shape == (2, 3, 2, 4)

    def test_loss_computation(self, clt_model, simple_batch):
        """Test loss computation."""
        # Test basic loss computation
        loss = clt_model.compute_loss(simple_batch, return_aux_data=False)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar loss
        
        # Test loss computation with auxiliary data
        loss, (loss_dict, aux_data) = clt_model.compute_loss(simple_batch, return_aux_data=True)
        assert isinstance(loss, torch.Tensor)
        assert "l_rec" in loss_dict
        assert "feature_acts" in aux_data
        assert "reconstructed" in aux_data
        assert "hidden_pre" in aux_data

    def test_decoder_indexing(self, clt_model):
        """Test decoder weight indexing."""
        # Test getting decoder weights for different layer pairs
        # For n_layers=2, we should have decoders for:
        # (0,0), (0,1), (1,1) - total of 3 decoders
        
        decoder_00 = clt_model.get_decoder_weights(0, 0)
        decoder_01 = clt_model.get_decoder_weights(0, 1)
        decoder_11 = clt_model.get_decoder_weights(1, 1)
        
        assert decoder_00.shape == (8, 4)  # (d_sae, d_model)
        assert decoder_01.shape == (8, 4)
        assert decoder_11.shape == (8, 4)
        
        # Test that we can't access invalid decoder combinations
        with pytest.raises(AssertionError):
            clt_model.get_decoder_weights(1, 0)  # layer_from > layer_to

    def test_parameter_initialization_methods(self, clt_model):
        """Test parameter initialization and normalization methods."""
        # Test encoder initialization with decoder transpose
        clt_model.init_encoder_with_decoder_transpose(factor=1.0)
        
        # Test setting decoder to fixed norm
        clt_model.set_decoder_to_fixed_norm(value=1.0, force_exact=True)
        
        # Test setting encoder to fixed norm
        clt_model.set_encoder_to_fixed_norm(value=1.0)
        
        # Test transform to unit decoder norm
        clt_model.transform_to_unit_decoder_norm()

    def test_norm_computations(self, clt_model):
        """Test norm computation methods."""
        # Test encoder norm
        encoder_norm = clt_model.encoder_norm()
        assert encoder_norm.shape == (8,)  # (d_sae,)
        
        # Test decoder norm
        decoder_norm = clt_model.decoder_norm()
        assert decoder_norm.shape == (8,)  # (d_sae,)
        
        # Test decoder bias norm
        decoder_bias_norm = clt_model.decoder_bias_norm()
        assert decoder_bias_norm.shape == (1, 4)  # (1, d_model)

    def test_missing_hook_points(self, clt_model):
        """Test error handling for missing hook points."""
        incomplete_batch = {
            "layer_0_in": torch.ones(2, 3, 4),
            # Missing other hook points
        }
        
        with pytest.raises(ValueError, match="Missing hook point"):
            clt_model.prepare_input(incomplete_batch)
        
        with pytest.raises(ValueError, match="Missing hook point"):
            clt_model.prepare_label(incomplete_batch)

    def test_cross_layer_attribution(self, clt_model, simple_batch):
        """Test cross-layer attribution functionality."""
        x, _ = clt_model.prepare_input(simple_batch)
        
        attribution_data = clt_model.cross_layer_attribution(x)
        
        # Check that attribution data contains expected keys
        assert "feature_activations" in attribution_data
        assert "layer_outputs" in attribution_data
        assert "contributions_to_layer_0" in attribution_data
        assert "contributions_to_layer_1" in attribution_data
        
        # Check shapes
        assert attribution_data["feature_activations"].shape == (2, 3, 2, 8)
        assert attribution_data["layer_outputs"].shape == (2, 3, 2, 4)

    # TODO: Add your parameter initialization tests here
    def test_custom_parameter_initialization(self, clt_model, simple_batch):
        """Test with custom parameter initialization.
        
        Fill in this test with specific parameter values to verify
        that CLT is working correctly with known inputs and expected outputs.
        """
        # Example: Initialize parameters to specific values
        # with torch.no_grad():
        #     # Initialize encoders
        #     clt_model.W_E.data.fill_(0.1)
        #     clt_model.b_E.data.fill_(0.0)
        #     
        #     # Initialize decoders
        #     clt_model.W_D.data.fill_(0.1)
        #     clt_model.b_D.data.fill_(0.0)
        
        # TODO: Add your specific tests here
        pass


if __name__ == "__main__":
    pytest.main([__file__]) 