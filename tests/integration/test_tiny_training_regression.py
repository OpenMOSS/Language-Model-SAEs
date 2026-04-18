"""End-to-end regression test for tiny SAE training metrics.

This test runs training through the public llamascopium training API and compares
logged metrics against a hardcoded baseline.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

import llamascopium.trainer as trainer_module
from llamascopium import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    InitializerConfig,
    SAEConfig,
    TrainerConfig,
    TrainSAESettings,
    train_sae,
)
from tests.integration.tiny_training_regression_baseline import (
    BASELINE_METRICS_4A5D190,
    EVAL_STEPS,
    METRIC_THRESHOLDS,
)

HOOK_POINT = "blocks.4.hook_resid_post"
EXPECTED_LOG_KEYS: dict[str, str] = {
    "explained_variance": "metrics/explained_variance",
    "l0": "metrics/l0",
    "below_1e-5": "sparsity/below_1e-5",
    "above_1e-2": "sparsity/above_1e-2",
    "current_learning_rate": "details/current_learning_rate",
    "l1_coefficient": "details/l1_coefficient",
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this training regression test")
def test_tiny_training_metrics_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    activation_path = os.environ.get("LM_SAES_TINY_ACTIVATION_PATH")
    if activation_path is None:
        pytest.skip("Set LM_SAES_TINY_ACTIVATION_PATH to the activation cache root before running this test.")

    hook_dir = Path(activation_path) / HOOK_POINT
    if not hook_dir.exists():
        pytest.skip(f"Activation path not found for hook point: {hook_dir}")

    # Align seeds with the tiny training setup used for baseline generation.
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    result_path = tmp_path / "tiny_training_regression" / "4x" / "pytest_tiny_regression"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    captured_metrics: dict[int, dict[str, float]] = {}
    original_log_metrics = trainer_module.log_metrics

    def _capture_log_metrics(logger, metrics, step=None, title="Training Metrics"):
        if step in EVAL_STEPS:
            step_metrics: dict[str, float] = {}
            for metric_name, log_key in EXPECTED_LOG_KEYS.items():
                if log_key in metrics:
                    step_metrics[metric_name] = float(metrics[log_key])
            captured_metrics[int(step)] = step_metrics
        return original_log_metrics(logger, metrics, step=step, title=title)

    monkeypatch.setattr(trainer_module, "log_metrics", _capture_log_metrics)

    settings = TrainSAESettings(
        sae=SAEConfig(
            hook_point_in=HOOK_POINT,
            hook_point_out=HOOK_POINT,
            d_model=64,
            expansion_factor=4,
            act_fn="topk",
            top_k=8,
            dtype=torch.float32,
            device="cuda",
            use_triton_kernel=False,
        ),
        sae_name="pytest_tiny_regression",
        sae_series="tiny-regression",
        initializer=InitializerConfig(
            grid_search_init_norm=True,
            init_encoder_with_decoder_transpose=True,
        ),
        trainer=TrainerConfig(
            lr=2e-2,
            lr_warm_up_steps=100,
            total_training_tokens=2_000_000,
            initial_k=64,
            k_warmup_steps=200,
            k_schedule_type="linear",
            log_frequency=10,
            eval_frequency=200,
            n_checkpoints=0,
            use_batch_norm_mse=False,
            check_point_save_mode="linear",
            exp_result_path=str(result_path),
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path=activation_path,
                    name="act-1d",
                    device="cuda",
                    dtype=torch.float32,
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_1D,
            hook_points=[HOOK_POINT],
            batch_size=2048,
            buffer_size=None,
        ),
        wandb=None,
        data_parallel_size=1,
        model_parallel_size=1,
        eval=False,
    )

    failures: list[str] = []
    try:
        train_sae(settings)
    except Exception as exc:
        pytest.fail(f"llamascopium.train_sae failed: {exc!r}")

    for step in EVAL_STEPS:
        actual_step = captured_metrics.get(step, {})
        baseline_step = BASELINE_METRICS_4A5D190.get(step, {})

        for metric_name, threshold in METRIC_THRESHOLDS.items():
            actual = actual_step.get(metric_name)
            expected = baseline_step.get(metric_name)

            if actual is None:
                failures.append(f"step={step} metric={metric_name}: missing in current run")
                continue
            if expected is None:
                failures.append(f"step={step} metric={metric_name}: missing in baseline")
                continue

            diff = abs(actual - expected)
            if diff > threshold:
                failures.append(
                    f"step={step} metric={metric_name}: actual={actual:.6f}, expected={expected:.6f}, "
                    f"diff={diff:.6f}, threshold={threshold}"
                )

    assert not failures, (
        f"Tiny SAE training regression detected metric drift.\nresult_path: {result_path}\n" + "\n".join(failures)
    )
