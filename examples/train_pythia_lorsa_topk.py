import math
import os

import torch

from lm_saes import (
    ActivationFactoryConfig,
    ActivationFactoryDatasetSource,
    ActivationFactoryTarget,
    BufferShuffleConfig,
    DatasetConfig,
    InitializerConfig,
    LanguageModelConfig,
    LorsaConfig,
    TrainerConfig,
    TrainLorsaSettings,
    WandbConfig,
    train_lorsa,
)

if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    settings = TrainLorsaSettings(
        sae=LorsaConfig(
            hook_point_in="blocks.6.ln1.hook_normalized",
            hook_point_out="blocks.6.hook_attn_out",
            d_model=768,
            expansion_factor=8,
            n_qk_heads=48,
            act_fn="topk",
            d_qk_head=64,
            rotary_dim=64 // 4,
            rotary_adjacent_pairs=False,
            rotary_base=10_000,
            n_ctx=1024,
            top_k=64,
            dtype=torch.float32,
            device="cuda",
            skip_bos=True,
            use_post_qk_ln=False,
        ),
        initializer=InitializerConfig(
            grid_search_init_norm=False,
            init_encoder_with_decoder_transpose=False,
            decoder_uniform_bound=1 / math.sqrt(768),
            encoder_uniform_bound=1 / math.sqrt(768 * 8),
        ),
        trainer=TrainerConfig(
            amp_dtype=torch.float32,
            lr=2e-4,
            initial_k=64,
            k_warmup_steps=0.1,
            k_schedule_type="linear",
            total_training_tokens=800_000_000 // 1024,
            log_frequency=1000,
            eval_frequency=1000000,
            n_checkpoints=0,
            check_point_save_mode="linear",
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
            exp_name="pythia-160m-lorsa",
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryDatasetSource(
                    name="SlimPajama-3B",
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_2D,
            hook_points=["blocks.6.ln1.hook_normalized", "blocks.6.hook_attn_out"],
            batch_size=32,
            buffer_size=None,
            buffer_shuffle=BufferShuffleConfig(
                perm_seed=42,
                generator_device="cuda",
            ),
        ),
        sae_name="pythia-160m-lorsa",
        sae_series="pythia-lorsa",
    )
    train_lorsa(settings)
