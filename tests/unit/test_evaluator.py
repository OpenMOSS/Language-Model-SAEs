from typing import Iterable

import pytest
import torch
from pytest_mock import MockerFixture
from transformer_lens import HookedTransformer

from lm_saes.config import EvalConfig, SAEConfig
from lm_saes.evaluator import Evaluator
from lm_saes.sae import SparseAutoEncoder


@pytest.fixture
def eval_config() -> EvalConfig:
    return EvalConfig(
        total_eval_tokens=60,
        feature_sampling_window=2,
        use_cached_activations=False,
        device="cpu",
    )


@pytest.fixture
def sae_config(mocker: MockerFixture, eval_config: EvalConfig):
    config = mocker.Mock(spec=SAEConfig)
    config.hook_point_in = "hook_point_in"
    config.hook_point_out = "hook_point_out"
    config.d_sae = 4
    config.device = eval_config.device
    config.dtype = torch.float32
    config.d_model = 2
    return config


@pytest.fixture
def activation_stream(sae_config: SAEConfig) -> Iterable[dict[str, torch.Tensor]]:
    """Creates a mock activation stream with known values for testing."""
    batch_size = 4

    def stream_generator():
        # Create 20 batches of activations
        for i in range(20):
            yield {
                sae_config.hook_point_in: torch.ones(
                    (batch_size, sae_config.d_model), device=sae_config.device, dtype=sae_config.dtype
                )
                * i,
                sae_config.hook_point_out: torch.ones(
                    (batch_size, sae_config.d_model), device=sae_config.device, dtype=sae_config.dtype
                )
                * (i + 1),
                "tokens": torch.arange(start=1, end=5, device=sae_config.device, dtype=torch.int64),
            }

    return stream_generator()


@pytest.fixture
def token_stream(sae_config: SAEConfig) -> Iterable[dict[str, torch.Tensor]]:
    def stream_generator():
        for _ in range(20):
            yield {"tokens": torch.arange(start=1, end=5, device=sae_config.device, dtype=torch.int64)}

    return stream_generator()


@pytest.fixture
def sae(sae_config: SAEConfig, mocker: MockerFixture) -> SparseAutoEncoder:
    batch_size = 4
    sae = mocker.Mock(spec=SparseAutoEncoder)
    sae.cfg = sae_config
    sae.forward.return_value = (
        torch.ones(batch_size, sae_config.d_model, device=sae_config.device, dtype=sae_config.dtype) * 3
    )
    sae.encode.return_value = (
        torch.ones(batch_size, sae_config.d_sae, device=sae_config.device, dtype=sae_config.dtype) * 4
    )
    sae.normalize_activations.side_effect = lambda x: x
    return sae


@pytest.fixture
def mock_model(mocker, sae_config: SAEConfig):
    model = mocker.Mock(spec=HookedTransformer)
    # Mock the tokenizer attributes
    model.tokenizer = mocker.Mock()
    model.tokenizer.eos_token_id = 0
    model.tokenizer.bos_token_id = 1
    model.tokenizer.pad_token_id = 2

    # Mock run_with_cache method
    def mock_run_with_cache(*args, **kwargs):
        batch_size = args[0].shape[0]  # input_ids shape
        d_model = sae_config.d_model  # example dimension

        # Create mock loss and cache
        mock_loss = torch.ones(batch_size) * 10
        mock_cache = {
            sae_config.hook_point_in: torch.randn(
                (batch_size, d_model), device=sae_config.device, dtype=sae_config.dtype
            ),
            sae_config.hook_point_out: torch.randn(
                (batch_size, d_model), device=sae_config.device, dtype=sae_config.dtype
            ),
        }
        return mock_loss, mock_cache

    def mock_run_with_hooks(*args, **kwargs):
        input_ids = args[0]
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1] if len(input_ids.shape) > 1 else 1

        # Verify hook arguments
        fwd_hooks = kwargs.get("fwd_hooks", [])
        assert len(fwd_hooks) == 1  # Should have one hook
        hook_point, replace_hook = fwd_hooks[0]

        # Create mock loss tensor
        mock_loss = (
            torch.ones((batch_size, seq_len), device=sae_config.device, dtype=sae_config.dtype) * 5
            if kwargs.get("loss_per_token", False)
            else torch.ones(1, device=sae_config.device, dtype=sae_config.dtype) * 5
        )
        return mock_loss

    model.run_with_cache.side_effect = mock_run_with_cache
    model.run_with_hooks.side_effect = mock_run_with_hooks
    return model


def test_evaluator_initialization(eval_config):
    evaluator = Evaluator(eval_config)
    assert evaluator.cfg == eval_config
    assert evaluator.cur_step == 0
    assert evaluator.cur_tokens == 0
    assert evaluator.metrics == {}


def test_evaluate_tokens(eval_config, sae, token_stream, mock_model):
    evaluator = Evaluator(eval_config)

    batch = next(iter(token_stream))  # batch = {"tokens": torch.tensor([1, 2, 3, 4], dtype=torch.int64)}
    cache, loss_dict, useful_token_mask = evaluator._evaluate_tokens(sae, batch, mock_model)
    n_tokens = useful_token_mask.sum().item()
    assert n_tokens == 2
    assert loss_dict["loss_mean"] == torch.tensor(10, device=sae.cfg.device, dtype=sae.cfg.dtype)
    assert loss_dict["loss_reconstruction_mean"] == torch.tensor(5, device=sae.cfg.device, dtype=sae.cfg.dtype)
    assert loss_dict["reconstructed"].shape == (4, 2)
    assert torch.allclose(loss_dict["reconstructed"], torch.ones(4, 2) * 3)
    assert cache["hook_point_in"].shape == (4, 2)
    assert cache["hook_point_out"].shape == (4, 2)


def test_evaluate_activations(eval_config, sae, activation_stream, mock_model):
    evaluator = Evaluator(eval_config)
    log_info = {
        "act_freq_scores": torch.zeros(sae.cfg.d_sae, device=sae.cfg.device, dtype=sae.cfg.dtype),
        "n_frac_active_tokens": torch.tensor([0], device=sae.cfg.device, dtype=torch.int),
    }
    for i, batch in enumerate(activation_stream):
        # batch: (4, 2)

        batch_size = batch[sae.cfg.hook_point_out].shape[0]
        loss_dict = {
            "loss_mean": torch.tensor(10 * i, device=sae.cfg.device, dtype=sae.cfg.dtype),
            "loss_reconstruction_mean": torch.tensor(5 * i, device=sae.cfg.device, dtype=sae.cfg.dtype),
            "reconstructed": torch.ones(batch_size, sae.cfg.d_model, device=sae.cfg.device, dtype=sae.cfg.dtype) * i,
        }
        log_info.update(loss_dict)
        useful_token_mask = torch.ones(batch_size, device=sae.cfg.device, dtype=torch.bool)
        useful_token_mask[0] = False
        evaluator._evaluate_activations(sae, log_info, batch, useful_token_mask)
        evaluator.cur_step += 1


def test_process_metrics(eval_config, mocker, sae):
    evaluator = Evaluator(eval_config)
    wandb_logger = mocker.Mock()
    evaluator.metrics["n_tokens"] = torch.arange(1, 21, device=sae.cfg.device, dtype=sae.cfg.dtype)
    evaluator.metrics["l_rec"] = torch.arange(1, 21, device=sae.cfg.device, dtype=sae.cfg.dtype)
    evaluator.metrics["l0"] = torch.arange(1, 21, device=sae.cfg.device, dtype=sae.cfg.dtype)
    evaluator.metrics["l2_norm_error"] = torch.arange(1, 21, device=sae.cfg.device, dtype=sae.cfg.dtype)
    evaluator.metrics["l2_norm_error_ratio"] = torch.arange(1, 21, device=sae.cfg.device, dtype=sae.cfg.dtype)
    evaluator.metrics["explained_variance"] = torch.arange(1, 21, device=sae.cfg.device, dtype=sae.cfg.dtype)
    evaluator.metrics["mean_log10_feature_sparsity"] = torch.arange(1, 21, device=sae.cfg.device, dtype=sae.cfg.dtype)
    evaluator.metrics["above_1e-1"] = torch.arange(1, 21, device=sae.cfg.device, dtype=sae.cfg.dtype)
    evaluator.metrics["above_1e-2"] = torch.arange(1, 21, device=sae.cfg.device, dtype=sae.cfg.dtype)
    evaluator.metrics["below_1e-5"] = torch.arange(1, 21, device=sae.cfg.device, dtype=sae.cfg.dtype)
    evaluator.metrics["below_1e-6"] = torch.arange(1, 21, device=sae.cfg.device, dtype=sae.cfg.dtype)
    evaluator.process_metrics(wandb_logger)
    print(evaluator.metrics)
