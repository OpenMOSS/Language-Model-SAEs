"""Unit tests for Cross Layer Transcoder (CLT) implementation."""

import math

import pytest
import torch

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
            sparsity_include_decoder_norm=False,
            force_unit_decoder_norm=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float32,
        )

    @pytest.fixture
    def clt_model(self, simple_config):
        """Create a CLT model instance."""
        return CrossLayerTranscoder(simple_config)

    @pytest.fixture
    def simple_batch(self, clt_model):
        """Create simple test data with only 1's and 0's."""
        batch_size = 2
        seq_len = 3
        d_model = 4

        # Create simple binary data
        batch = {
            "layer_0_in": torch.ones(batch_size, seq_len, d_model, device=clt_model.cfg.device),
            "layer_1_in": torch.zeros(batch_size, seq_len, d_model, device=clt_model.cfg.device),
            "layer_0_out": torch.ones(batch_size, seq_len, d_model, device=clt_model.cfg.device),
            "layer_1_out": torch.zeros(batch_size, seq_len, d_model, device=clt_model.cfg.device),
        }
        return batch

    def test_clt_initialization(self, clt_model, simple_config):
        """Test that CLT model initializes correctly."""
        # Check basic properties
        assert clt_model.cfg.n_layers == 2
        assert clt_model.cfg.d_model == 4
        assert clt_model.cfg.d_sae == 8
        # Check parameter shapes
        assert clt_model.W_E.shape == (2, 4, 8)  # (n_layers, d_model, d_sae)
        assert clt_model.b_E.shape == (2, 8)  # (n_layers, d_sae)
        for layer_to in range(clt_model.cfg.n_layers):
            assert clt_model.W_D[layer_to].shape == (layer_to + 1, 8, 4)  # (1, d_sae, d_model)
            assert clt_model.b_D[layer_to].shape == (4,)  # (d_model,)

    def test_prepare_input_and_label(self, clt_model, simple_batch):
        """Test input and label preparation."""
        # Test input preparation (uses hook_points_in)
        x, kwargs = clt_model.prepare_input(simple_batch)
        assert x.shape == (2, 3, 2, 4)  # (batch_size, seq_len, n_layers, d_model)
        assert torch.allclose(x[:, :, 0, :], torch.ones(2, 3, 4, device=clt_model.cfg.device))  # layer_0_in
        assert torch.allclose(x[:, :, 1, :], torch.zeros(2, 3, 4, device=clt_model.cfg.device))  # layer_1_in

        # Test label preparation (uses hook_points_out)
        labels = clt_model.prepare_label(simple_batch)
        assert labels.shape == (2, 3, 2, 4)  # (batch_size, seq_len, n_layers, d_model)
        assert torch.allclose(labels[:, :, 0, :], torch.ones(2, 3, 4, device=clt_model.cfg.device))  # layer_0_out
        assert torch.allclose(labels[:, :, 1, :], torch.zeros(2, 3, 4, device=clt_model.cfg.device))  # layer_1_out

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
        """Test decoder weight and bias indexing."""
        # Test getting decoder weights for different layers
        # For n_layers=2, we should have:
        # W_D[0] contains decoder for layer 0 -> layer 0  (shape: 1, d_sae, d_model)
        # W_D[1] contains decoders for layers 0,1 -> layer 1  (shape: 2, d_sae, d_model)

        decoder_weights_0 = clt_model.get_decoder_weights(0)  # layer 0 decoders
        decoder_weights_1 = clt_model.get_decoder_weights(1)  # layer 1 decoders

        assert decoder_weights_0.shape == (1, 8, 4)  # (1, d_sae, d_model) - only layer 0->0
        assert decoder_weights_1.shape == (2, 8, 4)  # (2, d_sae, d_model) - layers 0,1->1

        # Test getting decoder biases
        bias_0 = clt_model.get_decoder_bias(0)  # layer 0 bias
        bias_1 = clt_model.get_decoder_bias(1)  # layer 1 bias

        assert bias_0.shape == (4,)  # (d_model,)
        assert bias_1.shape == (4,)  # (d_model,)

    def test_norm_computations(self, clt_model):
        """Test norm computation methods."""
        # Test encoder norm - should return norm for each layer
        encoder_norm = clt_model.encoder_norm()
        assert encoder_norm.shape == (2,)  # (n_layers,) - averaged across layers

        # Test decoder norm - should return norm for each decoder
        decoder_norm = clt_model.decoder_norm()
        assert decoder_norm.shape == (3,)  # (n_decoder_matrices,)

        # Test decoder bias norm - should return norm for each layer
        decoder_bias_norm = clt_model.decoder_bias_norm()
        assert decoder_bias_norm.shape == (2,)  # (n_layers,)

    def test_missing_hook_points(self, clt_model):
        """Test error handling for missing hook points."""
        incomplete_batch = {
            "layer_0_in": torch.ones(2, 3, 4, device=clt_model.cfg.device),
            # Missing other hook points
        }

        with pytest.raises(ValueError, match="Missing hook point"):
            clt_model.prepare_input(incomplete_batch)

        with pytest.raises(ValueError, match="Missing hook point"):
            clt_model.prepare_label(incomplete_batch)

    @pytest.fixture
    def large_config(self):
        """Create a larger CLT configuration for testing initialization behavior."""
        return CLTConfig(
            sae_type="clt",
            d_model=1024,
            expansion_factor=16,  # d_sae = 16384
            hook_points_in=["layer_0_in", "layer_1_in", "layer_2_in"],
            hook_points_out=["layer_0_out", "layer_1_out", "layer_2_out"],
            use_decoder_bias=True,
            act_fn="relu",
            apply_decoder_bias_to_pre_encoder=False,
            norm_activation="inference",
            sparsity_include_decoder_norm=False,
            force_unit_decoder_norm=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float32,
        )

    @pytest.fixture
    def large_clt_model(self, large_config):
        """Create a large CLT model instance."""
        return CrossLayerTranscoder(large_config)

    def test_clt_initialization_behavior_large_model(self, large_clt_model):
        """Test CLT model with custom initialization to observe activation and output norms with larger dimensions."""
        config = large_clt_model.cfg

        # Initialize with custom CLT initialization
        large_clt_model.init_parameters(
            encoder_uniform_bound=0.1,  # Will be ignored by CLT
            decoder_uniform_bound=0.1,  # Will be ignored by CLT
            init_log_jumprelu_threshold_value=0.0,
        )

        # Create random input with 2-norm of sqrt(d_model)
        batch_size = 8
        seq_len = 16
        d_model = config.d_model
        n_layers = config.n_layers

        # Generate random input for each layer and normalize to have 2-norm of sqrt(d_model)
        target_norm = torch.sqrt(torch.tensor(d_model, dtype=torch.float32, device=large_clt_model.cfg.device))

        random_batch = {}
        for layer_idx in range(n_layers):
            # Generate random tensor
            random_tensor = torch.randn(batch_size, seq_len, d_model, device=large_clt_model.cfg.device)

            # Normalize each example to have the desired norm
            for b in range(batch_size):
                for s in range(seq_len):
                    current_norm = torch.norm(random_tensor[b, s, :])
                    if current_norm > 0:
                        random_tensor[b, s, :] = random_tensor[b, s, :] * target_norm / current_norm

            random_batch[f"layer_{layer_idx}_in"] = random_tensor
            random_batch[f"layer_{layer_idx}_out"] = random_tensor  # Use same for out (just for testing)

        # Prepare input and run forward pass
        input_tensor, _ = large_clt_model.prepare_input(random_batch)

        # Get feature activations and final output
        feature_acts, hidden_pre = large_clt_model.encode(input_tensor, return_hidden_pre=True)
        output = large_clt_model.decode(feature_acts)

        print("\n=== CLT Custom Initialization Test Results (Large Model) ===")
        print(f"Model config: d_model={config.d_model}, d_sae={config.d_sae}, n_layers={config.n_layers}")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Target input norm per token: {target_norm:.4f}")

        # Check actual input norms
        input_norms_per_layer = []
        for layer_idx in range(n_layers):
            layer_input = input_tensor[:, :, layer_idx, :]  # (batch, seq, d_model)
            layer_norms = torch.norm(layer_input, dim=-1)  # (batch, seq)
            avg_norm = layer_norms.mean().item()
            input_norms_per_layer.append(avg_norm)
            print(
                f"Layer {layer_idx} input - Average norm: {avg_norm:.4f}, Min: {layer_norms.min().item():.4f}, Max: {layer_norms.max().item():.4f}"
            )

        print("\n--- Feature Activations ---")
        for layer_idx in range(n_layers):
            layer_features = feature_acts[:, :, layer_idx, :]  # (batch, seq, d_sae)
            layer_feature_norms = torch.norm(layer_features, dim=-1)  # (batch, seq)
            layer_sparsity = (layer_features > 0).float().mean().item()

            print(
                f"Layer {layer_idx} features - Average norm: {layer_feature_norms.mean().item():.4f}, "
                f"Min: {layer_feature_norms.min().item():.4f}, Max: {layer_feature_norms.max().item():.4f}, "
                f"Sparsity: {layer_sparsity:.4f}"
            )

        print("\n--- Hidden Pre-Activations ---")
        for layer_idx in range(n_layers):
            layer_hidden_pre = hidden_pre[:, :, layer_idx, :]  # (batch, seq, d_sae)
            layer_hidden_norms = torch.norm(layer_hidden_pre, dim=-1)  # (batch, seq)

            print(
                f"Layer {layer_idx} hidden_pre - Average norm: {layer_hidden_norms.mean().item():.4f}, "
                f"Min: {layer_hidden_norms.min().item():.4f}, Max: {layer_hidden_norms.max().item():.4f}"
            )

        print("\n--- Reconstruction Outputs ---")
        for layer_idx in range(n_layers):
            layer_output = output[:, :, layer_idx, :]  # (batch, seq, d_model)
            layer_output_norms = torch.norm(layer_output, dim=-1)  # (batch, seq)

            print(
                f"Layer {layer_idx} output - Average norm: {layer_output_norms.mean().item():.4f}, "
                f"Min: {layer_output_norms.min().item():.4f}, Max: {layer_output_norms.max().item():.4f}"
            )

        print("\n--- Parameter Initialization Stats ---")

        # Encoder stats
        expected_encoder_bound = 1.0 / math.sqrt(config.d_sae)
        encoder_norm = torch.norm(large_clt_model.W_E)
        encoder_max = torch.max(torch.abs(large_clt_model.W_E))
        encoder_std = torch.std(large_clt_model.W_E)
        print(f"Encoder weights - Expected bound: ±{expected_encoder_bound:.6f}")
        print(
            f"  Actual norm: {encoder_norm.item():.4f}, Max abs value: {encoder_max.item():.6f}, Std: {encoder_std.item():.6f}"
        )

        # Decoder stats per layer
        for layer_to in range(n_layers):
            expected_decoder_bound = 1.0 / math.sqrt((layer_to + 1) * config.d_model)
            decoder_weights = large_clt_model.W_D[layer_to]
            decoder_norm = torch.norm(decoder_weights)
            decoder_max = torch.max(torch.abs(decoder_weights))
            decoder_std = torch.std(decoder_weights)
            print(f"Layer {layer_to} decoder weights - Expected bound: ±{expected_decoder_bound:.6f}")
            print(
                f"  Actual norm: {decoder_norm.item():.4f}, Max abs value: {decoder_max.item():.6f}, Std: {decoder_std.item():.6f}"
            )

        # Basic assertions to ensure things are working
        assert not torch.isnan(feature_acts).any(), "Feature activations contain NaN"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(feature_acts).any(), "Feature activations contain Inf"
        assert not torch.isinf(output).any(), "Output contains Inf"

        # Test loss computation
        loss = large_clt_model.compute_loss(random_batch, return_aux_data=False)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar loss
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"

        print(f"\nLoss: {loss.item():.6f}")

        # Test statistical properties with larger model
        # For uniform distribution U(-a,a), variance = a^2/3
        expected_encoder_var = (expected_encoder_bound**2) / 3
        actual_encoder_var = torch.var(large_clt_model.W_E).item()
        print(f"\nEncoder variance - Expected: {expected_encoder_var:.8f}, Actual: {actual_encoder_var:.8f}")

        for layer_to in range(n_layers):
            expected_decoder_bound = 1.0 / math.sqrt((layer_to + 1) * config.d_model)
            expected_decoder_var = (expected_decoder_bound**2) / 3
            actual_decoder_var = torch.var(large_clt_model.W_D[layer_to]).item()
            print(
                f"Layer {layer_to} decoder variance - Expected: {expected_decoder_var:.8f}, Actual: {actual_decoder_var:.8f}"
            )

        print("=== Test completed successfully ===\n")


if __name__ == "__main__":
    pytest.main([__file__])
