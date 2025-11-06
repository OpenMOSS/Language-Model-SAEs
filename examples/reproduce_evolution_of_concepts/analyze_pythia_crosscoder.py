import argparse
import os
import re
from pathlib import Path

import torch
from more_itertools import batched

from lm_saes import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    AnalyzeCrossCoderSettings,
    CrossCoderConfig,
    FeatureAnalyzerConfig,
    MongoDBConfig,
    analyze_crosscoder,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="160m")
    parser.add_argument("--name", type=str, default="L6R-lr5e-05-l1c0.5-32heads-8x-jlr0.1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--analysis-name", type=str, default="default")
    return parser.parse_args()


d_model_map = {
    "70m": 512,
    "160m": 768,
    "410m": 1024,
    "1b": 2048,
    "1.4b": 2048,
    "2.8b": 2048,
    "6.9b": 4096,
    "12b": 5120,
}

n_layers_map = {
    "70m": 6,
    "160m": 12,
    "410m": 24,
    "1b": 16,
    "1.4b": 24,
    "2.8b": 32,
    "6.9b": 32,
    "12b": 36,
}

steps = [
    0,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
    14000,
    20000,
    27000,
    34000,
    47000,
    60000,
    74000,
    87000,
    100000,
    114000,
    127000,
    143000,
]

if __name__ == "__main__":
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE"))
    if world_size is None:
        raise ValueError("WORLD_SIZE is not set")
    assert len(steps) % world_size == 0, f"Head count {len(steps)} is not divisible by world size {world_size}"

    head_per_device = len(steps) // world_size
    layer = int(re.search(r"L(\d+)R", args.name).group(1))
    print(f"Analyzing {args.name} at layer {layer}")
    settings = AnalyzeCrossCoderSettings(
        sae=CrossCoderConfig.from_pretrained(
            os.path.expanduser(f"~/results/{args.name}"),
            device="cuda",
            dtype=torch.float16,
        ),
        analyzer=FeatureAnalyzerConfig(
            total_analyzing_tokens=100_000_000,
            subsamples={
                "top_activations": {"proportion": 1.0, "n_samples": 20},
                "non_activating": {"proportion": 0.3, "n_samples": 20, "max_length": 50},
            },
            ignore_token_ids=[0],
        ),
        sae_name=args.name,
        sae_series="pythia-crosscoder",
        activation_factories=[
            ActivationFactoryConfig(
                sources=[
                    ActivationFactoryActivationsSource(
                        path={
                            f"step{step}": Path(
                                os.path.expanduser(
                                    f"~/activations/SlimPajama-3B-activations-pythia-{args.size}-2d-all-fp16/step{step}/blocks.{layer}.hook_resid_post"
                                )
                            )
                            for step in per_device_steps
                        },
                        sample_weights=1.0,
                        name="SlimPajama-3B",
                        device="cuda",
                        dtype=torch.float16,
                    )
                ],
                target=ActivationFactoryTarget.ACTIVATIONS_2D,
                hook_points=[f"step{step}" for step in per_device_steps],
                batch_size=args.batch_size,
            )
            for per_device_steps in batched(steps, head_per_device)
        ],
        mongo=MongoDBConfig(),
        feature_analysis_name=args.analysis_name,
        device_type="cuda",
    )
    analyze_crosscoder(settings)
