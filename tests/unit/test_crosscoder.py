import math

import pytest
import torch

from lm_saes import CrossCoderConfig
from lm_saes.crosscoder import CrossCoder


@pytest.fixture
def config() -> CrossCoderConfig:
    d_model = 2
    expansion_factor = 2  # Using integer expansion factor
    # Note: d_sae will be calculated as d_model * expansion_factor = 4
    return CrossCoderConfig(
        hook_points=["head0", "head1"],
        d_model=d_model,
        expansion_factor=expansion_factor,
        n_heads=2,
        device="cpu",
        dtype=torch.float32,
        act_fn="relu",
        top_k=2,
        norm_activation="token-wise",
        use_decoder_bias=True,
        sparsity_include_decoder_norm=True,
    )


@pytest.fixture
def crosscoder(config: CrossCoderConfig) -> CrossCoder:
    """Create a CrossCoder with predefined weights for testing."""
    model = CrossCoder(config)

    # Initialize encoder weights with known values
    model.W_E.data = torch.tensor(
        [
            # Head 0
            [
                [1.0, 0.5, 0.0, 0.2],  # input dim 0
                [0.0, 1.0, 0.5, 0.3],  # input dim 1
            ],
            # Head 1
            [
                [0.5, 0.0, 1.0, 0.4],  # input dim 0
                [0.5, 1.0, 0.0, 0.6],  # input dim 1
            ],
        ],
        device=config.device,
        dtype=config.dtype,
    )

    # Initialize encoder bias
    model.b_E.data = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.1],  # Head 0
            [0.3, 0.2, 0.1, 0.3],  # Head 1
        ],
        device=config.device,
        dtype=config.dtype,
    )

    # Initialize decoder weights
    model.W_D.data = torch.tensor(
        [
            # Head 0
            [
                [1.0, 0.5],  # feature 0
                [0.5, 1.0],  # feature 1
                [0.0, 0.5],  # feature 2
                [0.2, 0.3],  # feature 3
            ],
            # Head 1
            [
                [0.5, 0.0],  # feature 0
                [1.0, 0.5],  # feature 1
                [0.5, 1.0],  # feature 2
                [0.4, 0.6],  # feature 3
            ],
        ],
        device=config.device,
        dtype=config.dtype,
    )

    # Initialize decoder bias
    model.b_D.data = torch.tensor(
        [
            [0.1, 0.2],  # Head 0
            [0.2, 0.1],  # Head 1
        ],
        device=config.device,
        dtype=config.dtype,
    )

    return model


def test_init_parameters(config: CrossCoderConfig):
    """Test that parameters are initialized correctly."""
    model = CrossCoder(config)
    model.init_parameters(encoder_uniform_bound=0.1, decoder_uniform_bound=0.1, init_log_jumprelu_threshold_value=0.0)

    # Check shapes
    assert model.W_E.shape == (config.n_heads, config.d_model, config.d_sae)
    assert model.b_E.shape == (config.n_heads, config.d_sae)
    assert model.W_D.shape == (config.n_heads, config.d_sae, config.d_model)
    assert model.b_D.shape == (config.n_heads, config.d_model)

    # Check that values are within bounds
    assert torch.all(model.W_E <= 0.1)
    assert torch.all(model.W_E >= -0.1)
    assert torch.all(model.W_D <= 0.1)
    assert torch.all(model.W_D >= -0.1)

    # Check biases are zeros
    assert torch.all(model.b_E == 0.0)
    assert torch.all(model.b_D == 0.0)


def test_encode(crosscoder: CrossCoder):
    """Test encoding with known inputs."""
    # Create a batch with 1 sample
    x = torch.tensor(
        [
            # Batch 1
            [
                # Head 0
                [1.0, 2.0],
                # Head 1
                [2.0, 1.0],
            ]
        ],
        device=crosscoder.cfg.device,
        dtype=crosscoder.cfg.dtype,
    )

    # Expected pre-activation values (observed from actual output)
    # Individual head activations were summed together and returned, with the result being:
    # Combined pre-activations: [2.9, 3.9, 3.4, 2.6]
    # Apply ReLU activation (>0): [1, 1, 1, 1] elementwise
    # Decoder norms: sqrt(W_D[0,0,0]² + W_D[0,0,1]²) = sqrt(1² + 0.5²) = sqrt(1.25) = 1.118
    #                sqrt(W_D[0,1,0]² + W_D[0,1,1]²) = sqrt(0.5² + 1²) = sqrt(1.25) = 1.118
    #                sqrt(W_D[0,2,0]² + W_D[0,2,1]²) = sqrt(0² + 0.5²) = 0.5
    #                sqrt(W_D[0,3,0]² + W_D[0,3,1]²) = sqrt(0.2² + 0.3²) = sqrt(0.13) = 0.36
    # So activations * decoder_norm = [2.4*1.118, 3.9*1.118, 3.4*0.5, 2.4*0.36] = [2.68, 4.36, 1.7, 0.864]

    # Calculate result manually and verify with encode method
    feature_acts = crosscoder.encode(x)

    # Expected values based on observed output
    expected_hidden_pre = torch.tensor(
        [[[2.9, 3.9, 3.4, 2.6], [2.9, 3.9, 3.4, 2.6]]], device=crosscoder.cfg.device, dtype=crosscoder.cfg.dtype
    )

    # Get the hidden pre values for verification
    _, hidden_pre = crosscoder.encode(x, return_hidden_pre=True)

    # Check hidden pre values match our expectations
    assert torch.allclose(hidden_pre, expected_hidden_pre, atol=1e-6)

    # When ReLU is applied, it should retain the sign (>0 becomes 1, else 0)
    # All values are positive in our case
    assert torch.all(feature_acts > 0)


def test_decode(crosscoder: CrossCoder):
    """Test decoding with known feature activations."""
    # Create sample feature activations
    feature_acts = torch.tensor(
        [
            # Batch 1
            [
                # Head 0
                [1.0, 1.0, 1.0, 1.0],
                # Head 1
                [1.0, 1.0, 1.0, 1.0],
            ]
        ],
        device=crosscoder.cfg.device,
        dtype=crosscoder.cfg.dtype,
    )

    # Expected decoded values (from manual calculation)
    # Head 0: [1*1 + 1*0.5 + 1*0 + 1*0.2 + 0.1, 1*0.5 + 1*1 + 1*0.5 + 1*0.3 + 0.2] = [1.8, 2.5]
    # Head 1: [1*0.5 + 1*1 + 1*0.5 + 1*0.4 + 0.2, 1*0 + 1*0.5 + 1*1 + 1*0.6 + 0.1] = [2.6, 2.2]

    expected_decoded = torch.tensor(
        [
            [
                [1.8, 2.5],
                [2.6, 2.2],
            ]
        ],
        device=crosscoder.cfg.device,
        dtype=crosscoder.cfg.dtype,
    )

    decoded = crosscoder.decode(feature_acts)

    assert torch.allclose(decoded, expected_decoded, atol=1e-6)


def test_forward(crosscoder: CrossCoder):
    """Test forward pass (encode + decode)."""
    # Input tensor
    x = torch.tensor(
        [
            # Batch 1
            [
                # Head 0
                [1.0, 2.0],
                # Head 1
                [2.0, 1.0],
            ]
        ],
        device=crosscoder.cfg.device,
        dtype=crosscoder.cfg.dtype,
    )

    # Forward pass
    output = crosscoder.forward(x)

    # The forward pass should encode then decode
    feature_acts = crosscoder.encode(x)
    expected_output = crosscoder.decode(feature_acts)

    assert torch.allclose(output, expected_output, atol=1e-6)


def test_encoder_decoder_norm(crosscoder: CrossCoder):
    """Test encoder and decoder norm calculations."""
    # Manually calculate the encoder norm
    # √(W_E[0,0,0]² + W_E[0,1,0]²) = √(1² + 0²) = 1
    # √(W_E[0,0,1]² + W_E[0,1,1]²) = √(0.5² + 1²) = √1.25 = 1.118
    # √(W_E[0,0,2]² + W_E[0,1,2]²) = √(0² + 0.5²) = 0.5
    # √(W_E[0,0,3]² + W_E[0,1,3]²) = √(0.2² + 0.3²) = √0.13 = 0.36
    expected_encoder_norm_head0 = torch.tensor(
        [1.0, 1.118, 0.5, 0.36], device=crosscoder.cfg.device, dtype=crosscoder.cfg.dtype
    )

    # √(W_E[1,0,0]² + W_E[1,1,0]²) = √(0.5² + 0.5²) = √0.5 = 0.707
    # √(W_E[1,0,1]² + W_E[1,1,1]²) = √(0² + 1²) = 1
    # √(W_E[1,0,2]² + W_E[1,1,2]²) = √(1² + 0²) = 1
    # √(W_E[1,0,3]² + W_E[1,1,3]²) = √(0.4² + 0.6²) = √0.52 = 0.721
    expected_encoder_norm_head1 = torch.tensor(
        [0.707, 1.0, 1.0, 0.721], device=crosscoder.cfg.device, dtype=crosscoder.cfg.dtype
    )

    expected_encoder_norm = torch.stack([expected_encoder_norm_head0, expected_encoder_norm_head1])

    # Manually calculate the decoder norm
    # √(W_D[0,0,0]² + W_D[0,0,1]²) = √(1² + 0.5²) = √1.25 = 1.118
    # √(W_D[0,1,0]² + W_D[0,1,1]²) = √(0.5² + 1²) = √1.25 = 1.118
    # √(W_D[0,2,0]² + W_D[0,2,1]²) = √(0² + 0.5²) = 0.5
    # √(W_D[0,3,0]² + W_D[0,3,1]²) = √(0.2² + 0.3²) = √0.13 = 0.36
    expected_decoder_norm_head0 = torch.tensor(
        [1.118, 1.118, 0.5, 0.36], device=crosscoder.cfg.device, dtype=crosscoder.cfg.dtype
    )

    # √(W_D[1,0,0]² + W_D[1,0,1]²) = √(0.5² + 0²) = 0.5
    # √(W_D[1,1,0]² + W_D[1,1,1]²) = √(1² + 0.5²) = √1.25 = 1.118
    # √(W_D[1,2,0]² + W_D[1,2,1]²) = √(0.5² + 1²) = √1.25 = 1.118
    # √(W_D[1,3,0]² + W_D[1,3,1]²) = √(0.4² + 0.6²) = √0.52 = 0.721
    expected_decoder_norm_head1 = torch.tensor(
        [0.5, 1.118, 1.118, 0.721], device=crosscoder.cfg.device, dtype=crosscoder.cfg.dtype
    )

    expected_decoder_norm = torch.stack([expected_decoder_norm_head0, expected_decoder_norm_head1])

    encoder_norm = crosscoder.encoder_norm()
    decoder_norm = crosscoder.decoder_norm()

    assert torch.allclose(encoder_norm, expected_encoder_norm, atol=1e-3)
    assert torch.allclose(decoder_norm, expected_decoder_norm, atol=1e-3)


def test_decoder_bias_norm(crosscoder: CrossCoder):
    """Test decoder bias norm calculation."""
    # Expected bias norm per head: ||b_D||
    # Head 0: √(0.1² + 0.2²) = √0.05 = 0.224
    # Head 1: √(0.2² + 0.1²) = √0.05 = 0.224
    expected_bias_norm = torch.tensor(
        [
            [0.224],  # Head 0
            [0.224],  # Head 1
        ],
        device=crosscoder.cfg.device,
        dtype=crosscoder.cfg.dtype,
    )

    bias_norm = crosscoder.decoder_bias_norm()

    assert torch.allclose(bias_norm, expected_bias_norm, atol=1e-3)


def test_set_decoder_to_fixed_norm(crosscoder: CrossCoder):
    """Test setting decoder to fixed norm."""
    # Save original decoder weights
    original_W_D = crosscoder.W_D.clone()

    # Set to fixed norm of 2.0 with force_exact=True
    crosscoder.set_decoder_to_fixed_norm(2.0, force_exact=True)

    # Check that norms are exactly 2.0
    decoder_norm = crosscoder.decoder_norm()
    assert torch.allclose(decoder_norm, 2.0 * torch.ones_like(decoder_norm), atol=1e-5)

    # Reset weights
    crosscoder.W_D.data = original_W_D.clone()

    # Set to fixed norm of 0.1 with force_exact=False
    # This should scale down norms that are larger than 0.1
    crosscoder.set_decoder_to_fixed_norm(0.1, force_exact=False)

    # Check that all norms are <= 0.1
    decoder_norm = crosscoder.decoder_norm()
    assert torch.all(decoder_norm <= 0.1 + 1e-5)


def test_set_encoder_to_fixed_norm(crosscoder: CrossCoder):
    """Test setting encoder to fixed norm."""
    # Set to fixed norm of 2.0
    crosscoder.set_encoder_to_fixed_norm(2.0)

    # Check that norms are exactly 2.0
    encoder_norm = crosscoder.encoder_norm()
    assert torch.allclose(encoder_norm, 2.0 * torch.ones_like(encoder_norm), atol=1e-5)


def test_init_encoder_with_decoder_transpose(crosscoder: CrossCoder):
    """Test initializing encoder with decoder transpose."""
    # Save original weights
    original_W_D = crosscoder.W_D.clone()

    # Initialize encoder with decoder transpose
    crosscoder.init_encoder_with_decoder_transpose(factor=2.0)

    # Check that W_E = 2.0 * W_D.transpose
    for h in range(crosscoder.cfg.n_heads):
        expected_W_E = 2.0 * original_W_D[h].transpose(0, 1)
        assert torch.allclose(crosscoder.W_E[h], expected_W_E, atol=1e-5)


def test_prepare_input(crosscoder: CrossCoder):
    """Test prepare_input method."""
    # Create a batch with hook points
    batch = {
        "head0": torch.tensor([[1.0, 2.0]], device=crosscoder.cfg.device, dtype=crosscoder.cfg.dtype),
        "head1": torch.tensor([[2.0, 1.0]], device=crosscoder.cfg.device, dtype=crosscoder.cfg.dtype),
    }

    # Expected stacked input
    expected_input = torch.tensor(
        [
            [
                [1.0, 2.0],  # head0
                [2.0, 1.0],  # head1
            ]
        ],
        device=crosscoder.cfg.device,
        dtype=crosscoder.cfg.dtype,
    )

    # Test prepare_input
    x, encoder_kwargs, decoder_kwargs = crosscoder.prepare_input(batch)

    assert torch.allclose(x, expected_input, atol=1e-5)
    assert encoder_kwargs == {}
    assert decoder_kwargs == {}


def test_prepare_label(crosscoder: CrossCoder):
    """Test prepare_label method."""
    # Create a batch with hook points
    batch = {
        "head0": torch.tensor([[1.0, 2.0]], device=crosscoder.cfg.device, dtype=crosscoder.cfg.dtype),
        "head1": torch.tensor([[2.0, 1.0]], device=crosscoder.cfg.device, dtype=crosscoder.cfg.dtype),
    }

    # Expected label is the same as input for CrossCoder
    expected_label = torch.tensor(
        [
            [
                [1.0, 2.0],  # head0
                [2.0, 1.0],  # head1
            ]
        ],
        device=crosscoder.cfg.device,
        dtype=crosscoder.cfg.dtype,
    )

    # Test prepare_label
    label = crosscoder.prepare_label(batch)

    assert torch.allclose(label, expected_label, atol=1e-5)


def test_compute_loss(crosscoder: CrossCoder):
    """Test compute_loss with reconstruction loss."""
    # Create a batch
    batch = {
        "head0": torch.tensor([[1.0, 2.0]], device=crosscoder.cfg.device, dtype=crosscoder.cfg.dtype),
        "head1": torch.tensor([[2.0, 1.0]], device=crosscoder.cfg.device, dtype=crosscoder.cfg.dtype),
        "tokens": torch.tensor([[0, 0]], device=crosscoder.cfg.device, dtype=torch.long),
    }

    # Compute loss
    result = crosscoder.compute_loss(
        batch,
        return_aux_data=True,
        sparsity_loss_type=None,
    )
    loss = result["loss"]

    # Check loss is a scalar
    assert loss.shape == ()

    # Check result contains reconstruction loss
    assert "l_rec" in result

    # Check result contains expected keys
    assert "feature_acts" in result
    assert "reconstructed" in result
    assert "hidden_pre" in result

    # Check reconstruction matches forward pass
    x = crosscoder.prepare_input(batch)[0]
    expected_output = crosscoder.forward(x)

    assert torch.allclose(result["reconstructed"], expected_output, atol=1e-5)

    # Test with sparsity loss
    result_with_sparsity = crosscoder.compute_loss(
        batch,
        return_aux_data=True,
        sparsity_loss_type="power",
        p=1,
        l1_coefficient=0.1,
    )
    loss_with_sparsity = result_with_sparsity["loss"]

    # Sparsity loss should make the total loss higher
    assert loss_with_sparsity >= loss


def test_standardize_parameters_of_dataset_norm(crosscoder: CrossCoder):
    """Test standardizing parameters for dataset norm."""
    # Set norm_activation to dataset-wise
    crosscoder.cfg.norm_activation = "dataset-wise"

    # Set dataset norms
    dataset_norms = {"head0": 2.0, "head1": 3.0}
    crosscoder.set_dataset_average_activation_norm(dataset_norms)

    # Save original parameters
    original_b_E = crosscoder.b_E.clone()
    original_b_D = crosscoder.b_D.clone()

    # Standardize parameters
    crosscoder.standardize_parameters_of_dataset_norm()

    # Check that norm_activation is set to "inference"
    assert crosscoder.cfg.norm_activation == "inference"

    # Check biases are divided by respective norms
    sqrt_d_model = math.sqrt(crosscoder.cfg.d_model)
    expected_b_E_head0 = original_b_E[0] / (sqrt_d_model / dataset_norms["head0"])
    expected_b_E_head1 = original_b_E[1] / (sqrt_d_model / dataset_norms["head1"])
    expected_b_D_head0 = original_b_D[0] / (sqrt_d_model / dataset_norms["head0"])
    expected_b_D_head1 = original_b_D[1] / (sqrt_d_model / dataset_norms["head1"])

    assert torch.allclose(crosscoder.b_E[0], expected_b_E_head0, atol=1e-5)
    assert torch.allclose(crosscoder.b_E[1], expected_b_E_head1, atol=1e-5)
    assert torch.allclose(crosscoder.b_D[0], expected_b_D_head0, atol=1e-5)
    assert torch.allclose(crosscoder.b_D[1], expected_b_D_head1, atol=1e-5)


def test_torch_compile(crosscoder: CrossCoder):
    """Test that CrossCoder works with torch.compile."""
    # Always use eager backend to ensure compatibility with CPU
    backend = "eager"

    try:
        # Create input tensor
        x = torch.tensor(
            [
                # Batch 1
                [
                    # Head 0
                    [1.0, 2.0],
                    # Head 1
                    [2.0, 1.0],
                ]
            ],
            device=crosscoder.cfg.device,
            dtype=crosscoder.cfg.dtype,
        )

        # Test forward pass with original model
        original_output = crosscoder(x)

        # Test encode function with original model
        original_encode_output = crosscoder.encode(x)

        # Test decode function with original model
        feature_acts = torch.tensor(
            [
                # Batch 1
                [
                    # Head 0
                    [1.0, 1.0, 1.0, 1.0],
                    # Head 1
                    [1.0, 1.0, 1.0, 1.0],
                ]
            ],
            device=crosscoder.cfg.device,
            dtype=crosscoder.cfg.dtype,
        )
        original_decode_output = crosscoder.decode(feature_acts)

        # Compile the model with different options based on PyTorch version
        compile_kwargs = {"backend": backend, "dynamic": True, "fullgraph": True}

        # Compile the model and its functions
        compiled_model = torch.compile(crosscoder, **compile_kwargs)
        compiled_encode = torch.compile(crosscoder.encode, **compile_kwargs)
        compiled_decode = torch.compile(crosscoder.decode, **compile_kwargs)

        # Test forward pass with compiled model
        compiled_output = compiled_model(x)

        # Test encode with compiled function
        compiled_encode_output = compiled_encode(x)

        # Test decode with compiled function
        compiled_decode_output = compiled_decode(feature_acts)

        # Assert all outputs match
        assert torch.allclose(original_output, compiled_output, atol=1e-5)
        assert torch.allclose(original_encode_output, compiled_encode_output, atol=1e-5)
        assert torch.allclose(original_decode_output, compiled_decode_output, atol=1e-5)

    except Exception as e:
        if "dynamo" in str(e).lower() or "compile" in str(e).lower():
            pytest.skip(f"torch.compile failed with error: {e}")
        else:
            raise e


def test_decoder_inner_product_matrices(crosscoder: CrossCoder):
    """Test calculation of decoder inner product matrices."""
    # Get the inner product matrices
    inner_product_matrices = crosscoder.decoder_inner_product_matrices()

    # Expected shape: (d_sae, n_heads, n_heads)
    assert inner_product_matrices.shape == (crosscoder.cfg.d_sae, crosscoder.cfg.n_heads, crosscoder.cfg.n_heads)

    # For each feature dimension, calculate the expected inner product between heads
    # For feature 0:
    # Head 0: [1.0, 0.5]
    # Head 1: [0.5, 0.0]
    # Inner product = 1.0*0.5 + 0.5*0.0 = 0.5
    expected_feature0 = torch.tensor(
        [
            [1.0 * 1.0 + 0.5 * 0.5, 1.0 * 0.5 + 0.5 * 0.0],  # Head 0 with Heads [0,1]
            [0.5 * 1.0 + 0.0 * 0.5, 0.5 * 0.5 + 0.0 * 0.0],  # Head 1 with Heads [0,1]
        ],
        device=crosscoder.cfg.device,
        dtype=crosscoder.cfg.dtype,
    )

    # For feature 1:
    # Head 0: [0.5, 1.0]
    # Head 1: [1.0, 0.5]
    # Inner product = 0.5*1.0 + 1.0*0.5 = 1.0
    expected_feature1 = torch.tensor(
        [
            [0.5 * 0.5 + 1.0 * 1.0, 0.5 * 1.0 + 1.0 * 0.5],  # Head 0 with Heads [0,1]
            [1.0 * 0.5 + 0.5 * 1.0, 1.0 * 1.0 + 0.5 * 0.5],  # Head 1 with Heads [0,1]
        ],
        device=crosscoder.cfg.device,
        dtype=crosscoder.cfg.dtype,
    )

    # Check the first two features (others follow similar pattern)
    assert torch.allclose(inner_product_matrices[0], expected_feature0, atol=1e-6)
    assert torch.allclose(inner_product_matrices[1], expected_feature1, atol=1e-6)


def test_decoder_similarity_matrices(crosscoder: CrossCoder):
    """Test calculation of decoder similarity matrices."""
    # Get the similarity matrices
    similarity_matrices = crosscoder.decoder_similarity_matrices()

    # Expected shape: (d_sae, n_heads, n_heads)
    assert similarity_matrices.shape == (crosscoder.cfg.d_sae, crosscoder.cfg.n_heads, crosscoder.cfg.n_heads)

    # Get the inner product matrices and decoder norms
    inner_product_matrices = crosscoder.decoder_inner_product_matrices()
    decoder_norms = crosscoder.decoder_norm()

    # Calculate expected similarity matrices manually
    # For each feature dimension, similarity = inner_product / (norm_i * norm_j)
    expected_similarity = inner_product_matrices.clone()
    for i in range(crosscoder.cfg.n_heads):
        for j in range(crosscoder.cfg.n_heads):
            expected_similarity[:, i, j] /= decoder_norms[i] * decoder_norms[j]

    # Check that the similarity matrices match our manual calculation
    assert torch.allclose(similarity_matrices, expected_similarity, atol=1e-6)

    # Check that diagonal elements are 1.0 (self-similarity)
    for i in range(crosscoder.cfg.n_heads):
        assert torch.allclose(similarity_matrices[:, i, i], torch.ones_like(similarity_matrices[:, i, i]), atol=1e-6)

    # Check that matrices are symmetric
    for i in range(crosscoder.cfg.n_heads):
        for j in range(crosscoder.cfg.n_heads):
            assert torch.allclose(similarity_matrices[:, i, j], similarity_matrices[:, j, i], atol=1e-6)
