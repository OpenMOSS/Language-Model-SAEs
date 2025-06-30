import math

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

l = 15

if __name__ == "__main__":
    settings = TrainSAESettings(
        sae=SAEConfig(
            sae_type="sae",
            hook_point_in=f"blocks.{l}.hook_resid_post",
            hook_point_out=f"blocks.{l}.hook_resid_post",
            d_model=4096,
            expansion_factor=8,
            act_fn="jumprelu",
            jumprelu_threshold_window=2.0,
            norm_activation="dataset-wise",
            sparsity_include_decoder_norm=False,
            force_unit_decoder_norm=False,
            dtype=torch.float32,
            device="cuda",
        ),
        initializer=InitializerConfig(
            init_log_jumprelu_threshold_value=math.log(0.1),
            init_encoder_with_decoder_transpose_factor=1 / 8,
            encoder_uniform_bound=1 / (4096 * 8),
            decoder_uniform_bound=1 / 4096,
            init_search=False,
            state="training",
            bias_init_method="init_b_e_for_const_fire_times",
            const_times_for_init_b_e=10000,
        ),
        trainer=TrainerConfig(
            use_batch_norm_mse=False,
            use_triton_kernel=True,
            l1_coefficient=3.5,
            l1_coefficient_warmup_steps=1.0,
            sparsity_loss_type="tanh",
            tanh_stretch_coefficient=4.0,
            lr=2e-4,
            lr_scheduler_name="constantwithwarmup",
            lr_warm_up_steps=5000,
            total_training_tokens=800_000_000,
            log_frequency=100,
            feature_sampling_window=5000,
            eval_frequency=1000000,
            n_checkpoints=0,
            check_point_save_mode="linear",
            exp_result_path=f"jumprelu_test/L{l}R-init_threshold-0.1-l1-3.5-epsilon-2.0-bf16",
            # exp_result_path=f"jumprelu_test/debug",
        ),
        wandb=WandbConfig(
            log_to_wandb=True,
            wandb_project="Jan_Update_Jumprelu_Test",
            exp_name=f"jumprelu_test/L{l}R-init_threshold-0.1-l1-3.5-epsilon-2.0-bf16",
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/zf_projects/Language-Model-SAEs/LlamaScopeCross_LXR_activations",
                    type="activations",
                    name="SlimPajama-3B-LXR-800M",
                    device="cuda",
                    num_workers=None,
                ),
            ],
            target=ActivationFactoryTarget.BATCHED_ACTIVATIONS_1D,
            hook_points=[f"blocks.{l}.hook_resid_post"],
            batch_size=16384,
            buffer_size=None,
            ignore_token_ids=[],
        ),
        sae_name="Jan_Update_Jumprelu_Test",
        sae_series="Jan_Update_Jumprelu_Test",
        mongo=None,
        eval=False,
        data_parallel_size=1,
        model_parallel_size=1,
    )
    train_sae(settings)
