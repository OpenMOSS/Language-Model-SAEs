from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from llamascopium import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    AnalyzeSAESettings,
    FeatureAnalyzerConfig,
    SAEConfig,
    analyze_sae,
)

from llamascopium.utils.evo2_hooks import default_activation_dir, default_result_dir, tc_hook_points_for_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Evo2 TC features.")
    parser.add_argument("--layer", type=int, default=31, help="Layer to analyze.")
    parser.add_argument("--k", type=int, default=30, help="Top-k used during TC training.")
    parser.add_argument("--exp_factor", type=int, default=16, help="Expansion factor used during TC training.")
    parser.add_argument("--n_tokens", type=int, default=100_000_000, help="Number of tokens to analyze.")
    return parser.parse_args()


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    args = parse_args()
    model_name = "evo2_7b"
    lr = 1e-4
    hook_points = tc_hook_points_for_layer(args.layer)
    run_name = f"{model_name}_L{args.layer}_tc_{args.exp_factor}x_k{args.k}_lr{lr:.0e}"
    sae_path = Path(default_result_dir("tc")) / run_name
    activation_path = Path(default_activation_dir("tc_2d"))
    analysis_output_dir = sae_path / "analysis"

    print(f"[INFO] layer={args.layer} hook_points={hook_points}")
    print(f"[INFO] sae_path={sae_path}")
    print(f"[INFO] activation_path={activation_path}")
    print(f"[INFO] analysis_output_dir={analysis_output_dir}")

    settings = AnalyzeSAESettings(
        sae=SAEConfig.from_pretrained(
            str(sae_path),
            device="cuda",
            dtype=torch.float32,
        ),
        analyzer=FeatureAnalyzerConfig(
            total_analyzing_tokens=args.n_tokens,
            subsamples={
                "top_activations": {"proportion": 1.0, "n_samples": 16},
                "sampling_0.7": {"proportion": 0.7, "n_samples": 16},
                "sampling_0.5": {"proportion": 0.5, "n_samples": 16},
                "sampling_0.2": {"proportion": 0.2, "n_samples": 16},
                "non_activating": {"proportion": 0.3, "n_samples": 20, "max_length": 50},
            },
        ),
        sae_name=run_name,
        sae_series=f"{model_name}-tc",
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path=str(activation_path),
                    name="evo2-tc-2d",
                    device="cuda",
                    dtype=torch.float32,
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_2D,
            hook_points=hook_points,
            batch_size=16,
            buffer_size=None,
            ignore_token_ids=[],
        ),
        feature_analysis_name="default",
        output_dir=analysis_output_dir,
        device_type="cuda",
    )
    analyze_sae(settings)
