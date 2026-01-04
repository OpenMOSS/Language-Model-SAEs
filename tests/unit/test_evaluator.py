import pytest
import torch
from pytest_mock import MockerFixture
from transformer_lens import HookedTransformer

from lm_saes import EvalConfig, SAEConfig
from lm_saes.evaluator import Evaluator
from lm_saes.sae import SparseAutoEncoder


@pytest.fixture
def eval_config() -> EvalConfig:
    return EvalConfig(
        total_eval_tokens=10,
    )


@pytest.fixture
def sae_config(eval_config: EvalConfig):
    config = SAEConfig(
        hook_point_in="hook_point_in",
        hook_point_out="hook_point_out",
        d_sae=4,
        device="cpu",
        dtype=torch.float32,
        d_model=2,
        expansion_factor=2,
    )
    return config


@pytest.fixture
def sae(sae_config: SAEConfig, mocker: MockerFixture) -> SparseAutoEncoder:
    batch_size = 4
    sae = mocker.Mock(spec=SparseAutoEncoder)
    sae.cfg = sae_config

    # Mock specs
    sae.specs = mocker.Mock()
    sae.specs.feature_acts.return_value = ("batch", "sae")
    sae.specs.label.return_value = ("batch", "model")

    # Mock compute_loss to return what evaluate() expects
    def mock_compute_loss(batch, return_aux_data=False):
        if not return_aux_data:
            return torch.tensor(0.5)
        return {
            "loss": torch.tensor(0.5),
            "l_rec": torch.zeros(batch_size),
            "l_s": torch.zeros(batch_size),
            "feature_acts": torch.zeros(batch_size, sae_config.d_sae),
            "reconstructed": torch.zeros(batch_size, sae_config.d_model),
            "label": torch.zeros(batch_size, sae_config.d_model),
            "hidden_pre": torch.zeros(batch_size, sae_config.d_sae),
            "n_tokens": batch_size,
        }

    sae.compute_loss.side_effect = mock_compute_loss
    sae.normalize_activations.side_effect = lambda x: x
    sae.compute_training_metrics.return_value = {}
    return sae


@pytest.fixture
def mock_model(mocker, sae_config: SAEConfig):
    model = mocker.Mock(spec=HookedTransformer)
    model.tokenizer = mocker.Mock()
    return model


def test_evaluator_initialization(eval_config):
    evaluator = Evaluator(eval_config)
    assert evaluator.cfg == eval_config


def test_evaluate(eval_config, sae, mock_model):
    evaluator = Evaluator(eval_config)

    # Create a mock data stream
    batch_size = 4
    data_stream = [
        {
            "hook_point_in": torch.zeros(batch_size, sae.cfg.d_model),
            "hook_point_out": torch.zeros(batch_size, sae.cfg.d_model),
            "tokens": torch.zeros(batch_size, 1, dtype=torch.long),
        }
        for _ in range(5)
    ]

    # Run evaluate
    results = evaluator.evaluate(sae, data_stream, model=None)

    # Check that results contains some expected metrics
    assert "losses/overall_loss" in results
    assert "metrics/l0" in results
    assert "metrics/explained_variance" in results
    assert results["losses/overall_loss"] == 0.5
