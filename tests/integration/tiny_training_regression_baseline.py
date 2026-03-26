"""Baseline metrics for tiny SAE training regression.

This baseline is generated from commit `4a5d190617ff28d328799953504179f4cdfae9c0`
with the tiny training regression configuration in tests using:
    - hook point: blocks.4.hook_resid_post
    - activation path: TinyStories-1M-1d
    - total_training_tokens: 2_000_000
    - batch_size: 2048
    - topk: 8
"""

from __future__ import annotations

EVAL_STEPS = [200, 400, 600, 800, 970]

METRIC_THRESHOLDS: dict[str, float] = {
    "explained_variance": 1e-4,
    "l0": 1,
    "below_1e-5": 1,
    "above_1e-2": 1,
    "current_learning_rate": 1e-6,
    "l1_coefficient": 1e-6,
}

BASELINE_METRICS_4A5D190: dict[int, dict[str, float]] = {
    200: {
        "above_1e-2": 156.0,
        "below_1e-5": 46.0,
        "explained_variance": 0.7969,
        "l0": 10.0001,
        "current_learning_rate": 0.02,
        "l1_coefficient": 8e-05,
    },
    400: {
        "above_1e-2": 179.0,
        "below_1e-5": 38.0,
        "explained_variance": 0.8516,
        "l0": 7.9998,
        "current_learning_rate": 0.02,
        "l1_coefficient": 8e-05,
    },
    600: {
        "above_1e-2": 184.0,
        "below_1e-5": 31.0,
        "explained_variance": 0.8633,
        "l0": 8.0002,
        "current_learning_rate": 0.02,
        "l1_coefficient": 8e-05,
    },
    800: {
        "above_1e-2": 182.0,
        "below_1e-5": 30.0,
        "explained_variance": 0.8828,
        "l0": 8.0007,
        "current_learning_rate": 0.018212,
        "l1_coefficient": 8e-05,
    },
    970: {
        "above_1e-2": 190.0,
        "below_1e-5": 31.0,
        "explained_variance": 0.8828,
        "l0": 8.0006,
        "current_learning_rate": 0.001321,
        "l1_coefficient": 8e-05,
    },
}
