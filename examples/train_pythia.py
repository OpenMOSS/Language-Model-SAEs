import torch

from lm_saes import (
    ActivationFactoryConfig,
    ActivationFactoryDatasetSource,
    ActivationFactoryTarget,
    DatasetConfig,
    InitializerConfig,
    LanguageModelConfig,
    SAEConfig,
    TrainerConfig,
    TrainSAESettings,
    WandbConfig,
    train_sae,
)

if __name__ == "__main__":
    settings = TrainSAESettings(
        sae=SAEConfig(
            hook_point_in="blocks.3.ln1.hook_normalized",
            hook_point_out="blocks.3.ln1.hook_normalized",
            d_model=768,
            expansion_factor=8,
            act_fn="topk",
            norm_activation="token-wise",
            sparsity_include_decoder_norm=True,
            top_k=50,
            dtype=torch.float32,
            device="cuda",
        ),
        initializer=InitializerConfig(
            init_search=True,
            state="training",
        ),
        trainer=TrainerConfig(
            lp=1,
            initial_k=768 / 2,
            lr=4e-4,
            lr_scheduler_name="constantwithwarmup",
            total_training_tokens=600_000_000,
            log_frequency=1000,
            eval_frequency=1000000,
            n_checkpoints=5,
            check_point_save_mode="linear",
            exp_result_path="results",
        ),
        wandb=WandbConfig(
            wandb_project="pythia-160m-test",
            exp_name="pythia-160m-test",
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryDatasetSource(
                    name="openwebtext",
                )
            ],
            target=ActivationFactoryTarget.BATCHED_ACTIVATIONS_1D,
            hook_points=["blocks.3.ln1.hook_normalized"],
            batch_size=2048,
            buffer_size=None,
            ignore_token_ids=[],
        ),
        sae_name="pythia-160m-test-L3",
        sae_series="pythia-160m-test",
        model=LanguageModelConfig(
            model_name="EleutherAI/pythia-160m",
            device="cuda",
            dtype="torch.float32",
        ),
        model_name="pythia-160m",
        datasets={
            "openwebtext": DatasetConfig(
                dataset_name_or_path="Skylion007/openwebtext",
            )
        },
    )
    train_sae(settings)
