import pytest
import torch
from pytest_mock import MockerFixture

from lm_saes import InitializerConfig, SAEConfig
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
def initializer_config() -> InitializerConfig:
    return InitializerConfig(
        init_search=False,
    )


def test_initialize_sae_from_config(
    sae_config: SAEConfig, initializer_config: InitializerConfig, mocker: MockerFixture
):
    initializer = Initializer(initializer_config)
    sae_config.norm_activation = "token-wise"
    sae = initializer.initialize_sae_from_config(
        sae_config, activation_stream=[{"in": torch.ones(4, 2), "out": torch.ones(4, 2), "tokens": torch.zeros(4)}]
    )
    sae_config.norm_activation = "dataset-wise"
    sae = initializer.initialize_sae_from_config(
        sae_config,
        activation_norm={"in": 3.0, "out": 2.0},
        activation_stream=[{"in": torch.ones(4, 2), "out": torch.ones(4, 2), "tokens": torch.zeros(4)}],
    )
    assert sae.dataset_average_activation_norm == {"in": 3.0, "out": 2.0}

    sae_config.norm_activation = "dataset-wise"
    initializer = Initializer(initializer_config)
    sae = initializer.initialize_sae_from_config(
        sae_config,
        activation_norm={"in": 3.0, "out": 2.0},
        activation_stream=[{"in": torch.ones(4, 2), "out": torch.ones(4, 2), "tokens": torch.zeros(4)}],
    )
    # The config is not modified by the initializer anymore, it's passed to SAE.from_config
    assert sae.cfg.norm_activation == "dataset-wise"
    assert sae.dataset_average_activation_norm == {"in": 3.0, "out": 2.0}
    sae_config.sparsity_include_decoder_norm = False
    sae_config.act_fn = "topk"
    # jumprelu_threshold is not a field in SAEConfig anymore, it's jumprelu_threshold_window
    sae_config.jumprelu_threshold_window = 2.0
    sae = initializer.initialize_sae_from_config(
        sae_config, activation_stream=[{"in": torch.ones(4, 2), "out": torch.ones(4, 2), "tokens": torch.zeros(4)}]
    )
    assert sae.cfg.act_fn == "topk"


def test_initialize_search(mocker: MockerFixture, sae_config: SAEConfig, initializer_config: InitializerConfig):
    def stream_generator():
        # Create 10 batches of activations
        for _ in range(20):
            yield {
                "in": torch.ones(4, sae_config.d_model),  # norm will be sqrt(2)
                "out": torch.ones(4, sae_config.d_model) * 2,  # norm will be 2 * sqrt(2)
                "tokens": torch.tensor([2, 3, 4, 5]),
            }

    sae_config.hook_point_out = sae_config.hook_point_in
    initializer_config.grid_search_init_norm = True
    activation_stream_iter = stream_generator()
    initializer = Initializer(initializer_config)
    sae = initializer.initialize_sae_from_config(sae_config, activation_stream=activation_stream_iter)
    # _decoder_norm takes the module now, and returns norm of W_D.
    # With direct parameter access, it might be sae.decoder_norm()
    assert torch.allclose(sae.decoder_norm(), sae.decoder_norm().mean(), atol=1e-4, rtol=1e-5)

    initializer_config.bias_init_method = "geometric_median"
    initializer_config.init_encoder_with_decoder_transpose = True
    sae = initializer.initialize_sae_from_config(sae_config, activation_stream=activation_stream_iter)
    assert torch.allclose(sae.decoder_norm(), sae.decoder_norm().mean(), atol=1e-4, rtol=1e-5)
