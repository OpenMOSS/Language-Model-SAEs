import pytest
import torch
from pytest_mock import MockerFixture

from lm_saes.config import InitializerConfig, MixCoderConfig, SAEConfig
from lm_saes.initializer import Initializer


@pytest.fixture
def sae_config() -> SAEConfig:
    return SAEConfig(
        hook_point_in="in",
        hook_point_out="out",
        d_model=2,
        expansion_factor=2,
        device="cpu",
        dtype=torch.float32,  # the precision of bfloat16 is not enough for the tests
    )


@pytest.fixture
def mixcoder_config() -> MixCoderConfig:
    return MixCoderConfig(
        hook_point_in="in",
        hook_point_out="out",
        d_model=2,
        expansion_factor=2,
        device="cpu",
        dtype=torch.float32,  # the precision of bfloat16 is not enough for the tests
        modalities={"text": 4, "image": 4, "shared": 2},
    )


@pytest.fixture
def initializer_config() -> InitializerConfig:
    return InitializerConfig(
        state="training",
        init_search=False,
    )


def test_initialize_sae_from_config(
    sae_config: SAEConfig, mixcoder_config: MixCoderConfig, initializer_config: InitializerConfig, mocker: MockerFixture
):
    initializer = Initializer(initializer_config)
    initializer_config.state = "training"
    sae_config.norm_activation = "token-wise"
    sae = initializer.initialize_sae_from_config(sae_config)
    sae_config.norm_activation = "dataset-wise"
    sae = initializer.initialize_sae_from_config(sae_config, activation_norm={"in": 3.0, "out": 2.0})
    assert sae.dataset_average_activation_norm == {"in": 3.0, "out": 2.0}

    initializer_config.state = "inference"
    sae_config.norm_activation = "dataset-wise"
    initializer = Initializer(initializer_config)
    sae = initializer.initialize_sae_from_config(sae_config, activation_norm={"in": 3.0, "out": 2.0})
    assert sae.cfg.norm_activation == "inference"
    assert sae.dataset_average_activation_norm == {"in": 3.0, "out": 2.0}
    sae_config.sparsity_include_decoder_norm = False
    sae_config.act_fn = "topk"
    sae_config.jump_relu_threshold = 2.0
    sae = initializer.initialize_sae_from_config(sae_config)
    assert sae.cfg.act_fn == "jumprelu"

    initializer_config.state = "training"
    tokenizer = mocker.Mock()
    tokenizer.get_vocab.return_value = {
        "IMGIMG1": 1,
        "IMGIMG2": 2,
        "IMGIMG3": 3,
        "IMGIMG4": 4,
        "TEXT1": 5,
        "TEXT2": 6,
        "TEXT3": 7,
        "TEXT4": 8,
    }
    model_name = "facebook/chameleon-7b"

    mixcoder_settings = {"tokenizer": tokenizer, "model_name": model_name}

    mixcoder_config.norm_activation = "token-wise"
    mixcoder = initializer.initialize_sae_from_config(mixcoder_config, mixcoder_settings=mixcoder_settings)
    mixcoder_config.norm_activation = "dataset-wise"
    mixcoder = initializer.initialize_sae_from_config(
        mixcoder_config, activation_norm={"in": 3.0, "out": 2.0}, mixcoder_settings=mixcoder_settings
    )
    assert mixcoder.dataset_average_activation_norm == {"in": 3.0, "out": 2.0}

    initializer_config.state = "inference"
    mixcoder_config.norm_activation = "dataset-wise"
    initializer = Initializer(initializer_config)
    mixcoder = initializer.initialize_sae_from_config(mixcoder_config, activation_norm={"in": 3.0, "out": 2.0})
    assert mixcoder.cfg.norm_activation == "inference"
    assert mixcoder.dataset_average_activation_norm == {"in": 3.0, "out": 2.0}
    mixcoder_config.sparsity_include_decoder_norm = False
    mixcoder_config.act_fn = "topk"
    mixcoder_config.jump_relu_threshold = 2.0
    mixcoder = initializer.initialize_sae_from_config(mixcoder_config)
    assert mixcoder.cfg.act_fn == "topk"


def test_initialize_search(
    mocker: MockerFixture, sae_config: SAEConfig, mixcoder_config: MixCoderConfig, initializer_config: InitializerConfig
):
    def stream_generator():
        # Create 10 batches of activations
        for _ in range(20):
            yield {
                "in": torch.ones(4, sae_config.d_model),  # norm will be sqrt(16)
                "out": torch.ones(4, sae_config.d_model) * 2,  # norm will be sqrt(16) * 2
                "tokens": torch.tensor([2, 3, 4, 5]),
            }

    sae_config.hook_point_out = sae_config.hook_point_in
    initializer_config.init_search = True
    initializer_config.l1_coefficient = 0.0008
    activation_stream_iter = stream_generator()
    initializer = Initializer(initializer_config)
    sae = initializer.initialize_sae_from_config(sae_config, activation_stream=activation_stream_iter)
    assert torch.allclose(sae._decoder_norm(sae.decoder), sae._decoder_norm(sae.decoder).mean(), atol=1e-4, rtol=1e-5)

    initializer_config.bias_init_method = "geometric_median"
    initializer_config.init_encoder_with_decoder_transpose = True
    sae_config.apply_decoder_bias_to_pre_encoder = False
    sae = initializer.initialize_sae_from_config(sae_config, activation_stream=activation_stream_iter)
    assert torch.allclose(sae._decoder_norm(sae.decoder), sae._decoder_norm(sae.decoder).mean(), atol=1e-4, rtol=1e-5)

    tokenizer = mocker.Mock()
    tokenizer.get_vocab.return_value = {
        "IMGIMG1": 1,
        "IMGIMG2": 2,
        "IMGIMG3": 3,
        "IMGIMG4": 4,
        "TEXT1": 5,
        "TEXT2": 6,
        "TEXT3": 7,
        "TEXT4": 8,
    }
    model_name = "facebook/chameleon-7b"

    mixcoder_settings = {"tokenizer": tokenizer, "model_name": model_name}

    mixcoder_config.hook_point_out = mixcoder_config.hook_point_in
    initializer_config.init_search = True
    initializer_config.l1_coefficient = 0.0008
    activation_stream_iter = stream_generator()
    initializer = Initializer(initializer_config)
    mixcoder = initializer.initialize_sae_from_config(
        mixcoder_config, mixcoder_settings=mixcoder_settings, activation_stream=activation_stream_iter
    )
    assert torch.allclose(
        mixcoder._decoder_norm(mixcoder.decoder["image"]),
        mixcoder._decoder_norm(mixcoder.decoder["image"]).mean(),
        atol=1e-4,
        rtol=1e-5,
    )

    initializer_config.bias_init_method = "geometric_median"
    initializer_config.init_encoder_with_decoder_transpose = True
    mixcoder_config.apply_decoder_bias_to_pre_encoder = False
    mixcoder = initializer.initialize_sae_from_config(
        mixcoder_config, mixcoder_settings=mixcoder_settings, activation_stream=activation_stream_iter
    )
    assert torch.allclose(
        mixcoder._decoder_norm(mixcoder.decoder["text"]),
        mixcoder._decoder_norm(mixcoder.decoder["text"]).mean(),
        atol=1e-4,
        rtol=1e-5,
    )
