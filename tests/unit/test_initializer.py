import pytest
import torch
from pytest_mock import MockerFixture

from lm_saes.config import InitializerConfig, SAEConfig
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
def generator(sae_config: SAEConfig) -> torch.Generator:
    gen = torch.Generator(device=sae_config.device)
    gen.manual_seed(42)
    return gen


@pytest.fixture
def initializer_config() -> InitializerConfig:
    return InitializerConfig(
        state="training",
    )


def test_initialize_sae_from_config(sae_config: SAEConfig, initializer_config: InitializerConfig):
    initializer = Initializer(initializer_config)
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


def test_initialize_search(
    mocker: MockerFixture, sae_config: SAEConfig, initializer_config: InitializerConfig, generator: torch.Generator
):
    def stream_generator():
        # Create 10 batches of activations
        for _ in range(20):
            yield {
                "in": torch.ones(4, sae_config.d_model),  # norm will be sqrt(16)
                "out": torch.ones(4, sae_config.d_model) * 2,  # norm will be sqrt(16) * 2
            }

    sae_config.hook_point_out = sae_config.hook_point_in
    initializer_config.init_search = True
    initializer_config.l1_coefficient = 0.0008
    activation_stream_iter = mocker.Mock()
    activation_stream_iter = stream_generator()
    initializer = Initializer(initializer_config)
    sae = initializer.initialize_sae_from_config(sae_config, activation_stream=activation_stream_iter)
    assert torch.allclose(sae.decoder_norm(), sae.decoder_norm().mean(), atol=1e-4, rtol=1e-5)

    initializer_config.bias_init_method = "geometric_median"
    initializer_config.init_encoder_with_decoder_transpose = True
    sae_config.apply_decoder_bias_to_pre_encoder = False
    sae = initializer.initialize_sae_from_config(sae_config, activation_stream=activation_stream_iter)
    assert torch.allclose(sae.decoder_norm(), sae.decoder_norm().mean(), atol=1e-4, rtol=1e-5)
