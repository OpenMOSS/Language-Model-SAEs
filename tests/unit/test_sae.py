import math

import pytest
import torch

from lm_saes.config import SAEConfig
from lm_saes.sae import SparseAutoEncoder


@pytest.fixture
def sae_config() -> SAEConfig:
    return SAEConfig(
        hook_point_in="in",
        hook_point_out="out",
        d_model=2,
        expansion_factor=2,
        device="cpu",
        dtype=torch.float32,
        act_fn="topk",
        jump_relu_threshold=2.0,
        top_k=2,
    )


@pytest.fixture
def generator(sae_config: SAEConfig) -> torch.Generator:
    gen = torch.Generator(device=sae_config.device)
    gen.manual_seed(42)
    return gen


@pytest.fixture
def sae(sae_config: SAEConfig, generator: torch.Generator) -> SparseAutoEncoder:
    sae = SparseAutoEncoder(sae_config)
    sae.encoder.weight.data = torch.randn(
        sae_config.d_sae, sae_config.d_model, generator=generator, device=sae_config.device, dtype=sae_config.dtype
    )
    sae.decoder.weight.data = torch.randn(
        sae_config.d_model, sae_config.d_sae, generator=generator, device=sae_config.device, dtype=sae_config.dtype
    )
    if sae_config.use_decoder_bias:
        sae.decoder.bias.data = torch.randn(
            sae_config.d_model, generator=generator, device=sae_config.device, dtype=sae_config.dtype
        )
    if sae_config.use_glu_encoder:
        sae.encoder_glu.weight.data = torch.randn(
            sae_config.d_sae, sae_config.d_model, generator=generator, device=sae_config.device, dtype=sae_config.dtype
        )
        sae.encoder_glu.bias.data = torch.randn(
            sae_config.d_sae, generator=generator, device=sae_config.device, dtype=sae_config.dtype
        )
    return sae


def test_set_norm(sae: SparseAutoEncoder):
    def set_decoder_norm(norm: float, force_exact: bool):
        model = sae
        model.set_decoder_to_fixed_norm(norm, force_exact=force_exact)
        if force_exact:
            assert torch.allclose(
                model._decoder_norm(model.decoder, keepdim=False),
                norm * torch.ones(size=(model.cfg.d_sae,), device=model.cfg.device, dtype=model.cfg.dtype),
                atol=1e-4,
                rtol=1e-5,
            )
        else:
            assert torch.all(
                model._decoder_norm(model.decoder, keepdim=False)
                <= norm * torch.ones(size=(model.cfg.d_sae,), device=model.cfg.device, dtype=model.cfg.dtype) + 1e-4
            )

    def set_encoder_norm(norm: float):
        model = sae
        model.set_encoder_to_fixed_norm(norm)
        assert torch.allclose(
            model._encoder_norm(model.encoder, keepdim=False),
            norm * torch.ones(size=(model.cfg.d_sae,), device=model.cfg.device, dtype=model.cfg.dtype),
            atol=1e-4,
            rtol=1e-5,
        )

    set_decoder_norm(7.3, force_exact=True)
    set_decoder_norm(3.7, force_exact=False)
    set_decoder_norm(7.3, force_exact=False)
    set_encoder_norm(7.3)
    set_encoder_norm(3.7)


def test_sae_activate_fn(sae_config: SAEConfig, sae: SparseAutoEncoder):
    sae.current_k = 2
    assert torch.allclose(
        sae.activation_function(
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], device=sae_config.device, dtype=sae_config.dtype)
        ).to(sae_config.device, sae_config.dtype),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], device=sae_config.device, dtype=sae_config.dtype),
        atol=1e-4,
        rtol=1e-5,
    )


def test_transform_to_unit_decoder_norm(sae_config: SAEConfig, sae: SparseAutoEncoder):
    sae.transform_to_unit_decoder_norm()
    assert torch.allclose(
        sae._decoder_norm(sae.decoder, keepdim=False),
        torch.ones(size=(sae_config.d_sae,), device=sae_config.device, dtype=sae_config.dtype),
        atol=1e-4,
        rtol=1e-5,
    )


def test_compute_norm_factor(sae_config: SAEConfig, sae: SparseAutoEncoder):
    sae_config.norm_activation = "token-wise"
    assert torch.allclose(
        sae.compute_norm_factor(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=sae_config.device, dtype=sae_config.dtype), hook_point="in"
        ),
        torch.tensor(
            [[math.sqrt(2) / math.sqrt(5)], [math.sqrt(2) / 5]], device=sae_config.device, dtype=sae_config.dtype
        ),
        atol=1e-4,
        rtol=1e-5,
    )
    sae_config.norm_activation = "batch-wise"
    assert torch.allclose(
        sae.compute_norm_factor(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=sae_config.device, dtype=sae_config.dtype), hook_point="in"
        ),
        torch.tensor([[2 * math.sqrt(2) / (math.sqrt(5) + 5)]], device=sae_config.device, dtype=sae_config.dtype),
        atol=1e-4,
        rtol=1e-5,
    )
    sae_config.norm_activation = "dataset-wise"
    sae.set_dataset_average_activation_norm(
        {
            "in": 3.0,
            "out": 2.0,
        }
    )
    assert torch.allclose(
        sae.compute_norm_factor(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=sae_config.device, dtype=sae_config.dtype), hook_point="in"
        ),
        torch.tensor([[math.sqrt(2) / 3]], device=sae_config.device, dtype=sae_config.dtype),
        atol=1e-4,
        rtol=1e-5,
    )
    assert torch.allclose(
        sae.compute_norm_factor(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=sae_config.device, dtype=sae_config.dtype), hook_point="out"
        ),
        torch.tensor([[math.sqrt(2) / 2]], device=sae_config.device, dtype=sae_config.dtype),
        atol=1e-4,
        rtol=1e-5,
    )


def test_persistent_dataset_average_activation_norm(sae_config: SAEConfig, sae: SparseAutoEncoder):
    sae.set_dataset_average_activation_norm({"in": 3.0, "out": 2.0})
    assert sae.dataset_average_activation_norm == {"in": 3.0, "out": 2.0}
    state_dict = sae._get_full_state_dict()
    assert state_dict["dataset_average_activation_norm.in"] == 3.0
    assert state_dict["dataset_average_activation_norm.out"] == 2.0

    new_sae = SparseAutoEncoder(sae_config)
    new_sae._load_full_state_dict(state_dict)
    assert new_sae.cfg == sae.cfg
    assert all(torch.allclose(p, q, atol=1e-4, rtol=1e-5) for p, q in zip(new_sae.parameters(), sae.parameters()))
    assert new_sae.dataset_average_activation_norm == {"in": 3.0, "out": 2.0}


def test_get_full_state_dict(sae_config: SAEConfig, sae: SparseAutoEncoder):
    sae_config.sparsity_include_decoder_norm = False
    state_dict = sae._get_full_state_dict()
    assert "decoder.weight" in state_dict
    assert not torch.allclose(state_dict["decoder.weight"], sae.decoder.weight.data, atol=1e-4, rtol=1e-5)
    sae.set_decoder_to_fixed_norm(1.0, force_exact=True)
    assert torch.allclose(state_dict["decoder.weight"], sae.decoder.weight.data, atol=1e-4, rtol=1e-5)


def test_standardize_parameters_of_dataset_norm(sae_config: SAEConfig, sae: SparseAutoEncoder):
    sae_config.norm_activation = "dataset-wise"
    sae.encoder.bias.data = torch.tensor(
        [[1.0, 2.0]],
        requires_grad=True,
        dtype=sae_config.dtype,
        device=sae_config.device,
    )
    encoder_bias_data = sae.encoder.bias.data.clone()
    sae.decoder.weight.data = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        requires_grad=True,
        dtype=sae_config.dtype,
        device=sae_config.device,
    )
    decoder_weight_data = sae.decoder.weight.data.clone()
    if sae_config.use_decoder_bias:
        sae.decoder.bias.data = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0]],
            requires_grad=True,
            dtype=sae_config.dtype,
            device=sae_config.device,
        )
    decoder_bias_data = sae.decoder.bias.data.clone()
    sae.standardize_parameters_of_dataset_norm({"in": 3.0, "out": 2.0})
    assert sae.cfg.norm_activation == "inference"
    assert sae.dataset_average_activation_norm == {"in": 3.0, "out": 2.0}
    assert torch.allclose(
        sae.encoder.bias.data, encoder_bias_data / math.sqrt(sae_config.d_model) * 3.0, atol=1e-4, rtol=1e-5
    )
    assert torch.allclose(
        sae.decoder.weight.data,
        decoder_weight_data / 3.0 * 2.0,
        atol=1e-4,
        rtol=1e-5,
    )
    if sae_config.use_decoder_bias:
        assert torch.allclose(
            sae.decoder.bias.data, decoder_bias_data / math.sqrt(sae_config.d_model) * 2.0, atol=1e-4, rtol=1e-5
        )


def test_forward(sae_config: SAEConfig, sae: SparseAutoEncoder):
    sae.set_dataset_average_activation_norm({"in": 3.0, "out": 2.0})
    output = sae.forward(torch.tensor([[1.0, 2.0]], device=sae_config.device, dtype=sae_config.dtype))
    assert output.shape == (1, 2)
