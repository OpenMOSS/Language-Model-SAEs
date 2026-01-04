import argparse
import os

import torch

from lm_saes import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    AnalyzeSAESettings,
    FeatureAnalyzerConfig,
    MongoDBConfig,
    PretrainedSAE,
    analyze_sae,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--activation_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    args = parse_args()

    analyze_sae(
        AnalyzeSAESettings(
            sae=PretrainedSAE(
                pretrained_name_or_path=os.path.expanduser(args.sae_path),
                device="cuda",
                dtype=torch.float16,
            ),
            sae_name="pythia-160m-sae",
            sae_series="pythia-sae",
            activation_factory=ActivationFactoryConfig(
                sources=[
                    ActivationFactoryActivationsSource(
                        path=os.path.expanduser(args.activation_path),
                        name="pythia-160m-2d",
                        device="cuda",
                        dtype=torch.float16,
                    )
                ],
                target=ActivationFactoryTarget.ACTIVATIONS_2D,
                hook_points=["blocks.6.hook_resid_post"],
                batch_size=16,
                context_size=2048,
            ),
            analyzer=FeatureAnalyzerConfig(
                total_analyzing_tokens=100_000_000,
                subsamples={
                    "top_activations": {"proportion": 1.0, "n_samples": 20},
                    "subsampling_80%": {"proportion": 0.8, "n_samples": 10},
                    "subsampling_60%": {"proportion": 0.6, "n_samples": 10},
                    "subsampling_40%": {"proportion": 0.4, "n_samples": 10},
                    "non_activating": {"proportion": 0.3, "n_samples": 20, "max_length": 50},
                },
            ),
            mongo=MongoDBConfig(),
            device_type="cuda",
        )
    )
