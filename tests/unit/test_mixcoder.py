import pytest
import torch

from lm_saes.config import MixCoderConfig
from lm_saes.mixcoder import MixCoder


@pytest.fixture
def config():
    return MixCoderConfig(
        d_model=3,
        modalities={"text": 2, "image": 3, "shared": 4},
        device="cpu",
        dtype=torch.float32,
        use_glu_encoder=True,
        use_decoder_bias=True,
        hook_point_in="hook_point_in",
        hook_point_out="hook_point_out",
        expansion_factor=1.0,
        top_k=2,
        act_fn="topk",
    )


@pytest.fixture
def modality_indices():
    return {
        "text": torch.tensor([1, 2, 3, 4]),
        "image": torch.tensor([5, 6, 7, 8]),
    }


@pytest.fixture
def mixcoder(config, modality_indices):
    model = MixCoder(config)
    model.init_parameters(modality_indices=modality_indices)
    model.decoder["text"].bias.data = torch.rand_like(model.decoder["text"].bias.data)
    model.decoder["image"].bias.data = torch.rand_like(model.decoder["image"].bias.data)
    model.decoder["shared"].bias.data = torch.rand_like(model.decoder["shared"].bias.data)
    model.encoder["text"].bias.data = torch.rand_like(model.encoder["text"].bias.data)
    model.encoder["image"].bias.data = torch.rand_like(model.encoder["image"].bias.data)
    model.encoder["shared"].bias.data = torch.rand_like(model.encoder["shared"].bias.data)
    return model


def test_init_parameters(mixcoder, config):
    assert mixcoder.modality_index == {"text": (0, 2), "image": (2, 5), "shared": (5, 9)}
    assert torch.allclose(mixcoder.modality_indices["text"], torch.tensor([1, 2, 3, 4]))
    assert torch.allclose(mixcoder.modality_indices["image"], torch.tensor([5, 6, 7, 8]))


def test_encode_decode(mixcoder, config):
    """Test the encoding and decoding process."""
    mixcoder.set_dataset_average_activation_norm({"hook_point_in": 1.0, "hook_point_out": 1.0})
    batch_size = 8
    x = torch.randn(batch_size, config.d_model)  # batch, d_model
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    x_text = torch.cat([x[:4, :], torch.zeros(4, config.d_model)], dim=0)
    x_image = torch.cat([torch.zeros(4, config.d_model), x[4:, :]], dim=0)
    tokens_text = torch.tensor([1, 2, 3, 4, 0, 0, 0, 0])
    tokens_image = torch.tensor([0, 0, 0, 0, 5, 6, 7, 8])
    # Test encode
    feature_acts = mixcoder.encode(x, tokens=tokens)
    assert feature_acts.shape == (batch_size, config.d_sae)  # batch, d_sae

    feature_acts_text = mixcoder.encode(x_text, tokens=tokens_text)
    assert feature_acts_text.shape == (batch_size, config.d_sae)
    feature_acts_image = mixcoder.encode(x_image, tokens=tokens_image)
    assert feature_acts_image.shape == (batch_size, config.d_sae)
    modality_index = mixcoder.get_modality_index()

    assert torch.allclose(
        feature_acts_text[:4, slice(*modality_index["text"])],
        feature_acts[:4, slice(*modality_index["text"])],
    )
    assert torch.allclose(
        feature_acts_image[4:, slice(*modality_index["image"])],
        feature_acts[4:, slice(*modality_index["image"])],
    )

    assert torch.allclose(
        torch.cat(
            [
                feature_acts_text[:4, slice(*modality_index["shared"])],
                feature_acts_image[4:, slice(*modality_index["shared"])],
            ],
            dim=0,
        ),
        feature_acts[:, slice(*modality_index["shared"])],
    )
    print(feature_acts)

    # Test decode
    reconstructed = mixcoder.decode(feature_acts)
    assert reconstructed.shape == (batch_size, config.d_model)

    reconstructed_text = mixcoder.decode(feature_acts_text)
    assert reconstructed_text.shape == (batch_size, config.d_model)

    reconstructed_image = mixcoder.decode(feature_acts_image)
    assert reconstructed_image.shape == (batch_size, config.d_model)

    assert torch.allclose(reconstructed_text[:4, :], reconstructed[:4, :])
    assert torch.allclose(reconstructed_image[4:, :], reconstructed[4:, :])


def test_get_modality_activation_mask(mixcoder, config):
    """Test the _get_modality_activation method."""
    batch_size = 8
    x = torch.ones(batch_size, config.d_model)
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])

    # Test text modality
    text_activation_mask = mixcoder._get_modality_activation_mask(x, tokens, "text")
    assert torch.all(text_activation_mask[0, :4] == 1)  # First 4 positions should be 1
    assert torch.all(text_activation_mask[0, 4:] == 0)  # Last 4 positions should be 0

    # Test image modality
    image_activation_mask = mixcoder._get_modality_activation_mask(x, tokens, "image")
    assert torch.all(image_activation_mask[1, :4] == 0)  # First 4 positions should be 0
    assert torch.all(image_activation_mask[1, 4:] == 1)  # Last 4 positions should be 1
