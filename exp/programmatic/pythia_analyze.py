import torch

from lm_saes import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    AnalyzeSAESettings,
    FeatureAnalyzerConfig,
    MongoDBConfig,
    SAEConfig,
    analyze_sae,
)

if __name__ == "__main__":
    settings = AnalyzeSAESettings(
        sae=SAEConfig.from_pretrained(
            "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/sae/pythia-160m_lr0.0004-updated/blocks.3.ln1.hook_normalized",
            device="cuda",
            dtype=torch.float32,
        ),
        analyzer=FeatureAnalyzerConfig(
            total_analyzing_tokens=100_000_000,
            batch_size=16,
            enable_sampling=False,
            subsamples={
                "top_activations": {"proportion": 1.0, "n_samples": 10},
                "subsample-0.7": {"proportion": 0.7, "n_samples": 5},
                "subsample-0.5": {"proportion": 0.5, "n_samples": 5},
            },
        ),
        sae_name="pythia-160m-test-L3",
        sae_series="pythia-160m-test",
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/data/SlimPajama-3B-activations-pythia-2d",
                    type="activations",
                    name="SlimPajama-3B-activations-pythia-2d",
                    device="cuda",
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_2D,
            hook_points=["blocks.3.ln1.hook_normalized"],
        ),
        mongo=MongoDBConfig(
            mongo_uri="mongodb://10.244.170.184:27017/",
            mongo_db="sweeplr",
        ),
    )
    analyze_sae(settings)
