import argparse

import torch

from lm_saes import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    InitializerConfig,
    SAEConfig,
    TrainerConfig,
    TrainSAESettings,
    WandbConfig,
    train_sae,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAE with configurable learning rate.")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for the trainer.")
    args = parser.parse_args()

    lr = args.lr

    settings = TrainSAESettings(
        sae=SAEConfig(
            hook_point_in="blocks.3.ln1.hook_normalized",
            hook_point_out="blocks.3.ln1.hook_normalized",
            d_model=768,
            expansion_factor=8,
            act_fn="topk",
            norm_activation="dataset-wise",
            sparsity_include_decoder_norm=True,
            top_k=64,
            dtype=torch.float32,
            device="cuda",
        ),
        sae_name="pythia-160m-test-L3-updated",
        sae_series="pythia-160m-test-updated",
        initializer=InitializerConfig(
            init_search=True,
            state="training",
        ),
        trainer=TrainerConfig(
            lp=1,
            initial_k=768 / (2 * 64),
            lr=lr,
            lr_scheduler_name="constantwithwarmup",
            total_training_tokens=760_000_000,
            log_frequency=1000,
            eval_frequency=1000,
            n_checkpoints=10,
            check_point_save_mode="linear",
            exp_result_path=f"sae/pythia-160m_lr{lr}-updated/blocks.3.ln1.hook_normalized",
        ),
        wandb=WandbConfig(
            log_to_wandb=True,
            wandb_project="pythia-160m-lrsweep-updated",
            exp_name=f"pythia-160m-lr{lr}-updated",
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/data/SlimPajama-3B-activations-pythia",
                    type="activations",
                    name="pythia160m-activations",
                    device="cuda",
                    num_workers=16,
                )
            ],
            target=ActivationFactoryTarget.BATCHED_ACTIVATIONS_1D,
            hook_points=["blocks.3.ln1.hook_normalized"],
            batch_size=2048,
            buffer_size=500_000,
            ignore_token_ids=[0],
        ),
        eval=False,
        data_parallel_size=1,
        model_parallel_size=1,
    )
    train_sae(settings)
