import os

import torch

from lm_saes import (
    ActivationFactoryConfig,
    ActivationFactoryDatasetSource,
    ActivationFactoryTarget,
    BufferShuffleConfig,
    CLTConfig,
    DatasetConfig,
    InitializerConfig,
    LanguageModelConfig,
    TrainCLTSettings,
    TrainerConfig,
    WandbConfig,
    train_clt,
)

if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    settings = TrainCLTSettings(
        sae=CLTConfig(
            hook_points_in=[f"blocks.{i}.ln2.hook_normalized" for i in range(12)],
            hook_points_out=[f"blocks.{i}.hook_mlp_out" for i in range(12)],
            d_model=768,
            expansion_factor=8,
            act_fn="layertopk",
            top_k=256,
            dtype=torch.float32,
            device="cuda",
        ),
        initializer=InitializerConfig(
            grid_search_init_norm=False,
            init_encoder_with_decoder_transpose=False,
        ),
        trainer=TrainerConfig(
            initial_k=768 * 12 * 8 // 2,
            k_warmup_steps=1.0,
            k_schedule_type="exponential",
            k_exponential_factor=30,
            lr_warm_up_steps=1000,
            use_batch_norm_mse=False,
            lr=5e-5,
            optimizer_class="adam",
            total_training_tokens=800_000_000,
            log_frequency=1000,
            eval_frequency=1000000000,
            n_checkpoints=0,
            check_point_save_mode="log",
            exp_result_path="results",
        ),
        model=LanguageModelConfig(
            model_name="EleutherAI/pythia-160m",
            device="cuda",
            dtype="torch.float16",
        ),
        model_name="pythia-160m",
        datasets={
            "SlimPajama-3B": DatasetConfig(
                dataset_name_or_path="Hzfinfdu/SlimPajama-3B",
            )
        },
        wandb=WandbConfig(
            wandb_project="lm-saes",
            exp_name="pythia-160m-clt",
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryDatasetSource(
                    name="SlimPajama-3B",
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_1D,
            hook_points=[f"blocks.{i}.ln2.hook_normalized" for i in range(12)]
            + [f"blocks.{i}.hook_mlp_out" for i in range(12)],
            batch_size=4096,
            buffer_size=4096 * 4,
            buffer_shuffle=BufferShuffleConfig(
                perm_seed=42,
                generator_device="cuda",
            ),
        ),
        sae_name="pythia-160m-clt",
        sae_series="pythia-clt",
    )
    train_clt(settings)
