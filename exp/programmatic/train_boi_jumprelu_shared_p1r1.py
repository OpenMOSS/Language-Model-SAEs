import argparse
import math
import os

import torch
import torch.distributed as dist

from lm_saes import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    BufferShuffleConfig,
    CrossCoderConfig,
    InitializerConfig,
    TrainerConfig,
    TrainSAESettings,
    WandbConfig,
    train_sae,
)

dist.init_process_group(backend="nccl")
torch.cuda.set_device(f'cuda:{os.environ["LOCAL_RANK"]}')

parser = argparse.ArgumentParser(description="Train SAE Model")
parser.add_argument("--lr", type=float, required=True, help="Learning rate for training")
parser.add_argument("--l", type=int, required=True, help="Learning rate for training")
parser.add_argument("--l1coef", type=float, required=True, help="l1_coefficient")
parser.add_argument("--shared_decoder_sparsity_factor", type=float, required=False, help="l1_coefficient", default=0.1)
parser.add_argument("--expfactor", type=int, required=True, help="Learning rate for training")
args = parser.parse_args()
lr = args.lr
l = args.l
l1_coefficient = args.l1coef
expfactor = args.expfactor

if __name__ == "__main__":
    path_dict = {
        0: "/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/reasondata-b-2d-801M-f",
        1: "/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/reasondata-i-2d-801M-f",
        2: "/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/reasondata-o-2d-801M-f",
    }
    p_path_dict = {
        0: "/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/pretraindata-b-2d-1001M",
        1: "/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/pretraindata-i-2d-1001M",
        2: "/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/pretraindata-o-2d-1001M",
    }
    # 0, 1, 2, is for b i o respectively
    rank = dist.get_rank()
    path = path_dict.get(rank)
    p_path = p_path_dict.get(rank)

    modelname_dict = {0: "base_llama", 1: "Instruct_llama", 2: "o_llama"}
    subject_model = modelname_dict.get(rank)

    settings = TrainSAESettings(
        sae=CrossCoderConfig(
            sae_type="crosscoder",
            hook_point_in=f"blocks.{l}.hook_resid_post",
            hook_point_out=f"blocks.{l}.hook_resid_post",
            d_model=4096,
            expansion_factor=expfactor,
            act_fn="jumprelu",
            jumprelu_threshold_window=2.0,
            norm_activation="dataset-wise",
            sparsity_include_decoder_norm=False,
            force_unit_decoder_norm=False,
            dtype=torch.float32,
            device="cuda",
            use_shared_decoder=True,
            d_shared_decoder=2048,
            shared_decoder_sparsity_factor=args.shared_decoder_sparsity_factor,
            use_triton_kernel=True,
            sparsity_threshold_for_triton_spmm_kernel=0.99,
        ),
        initializer=InitializerConfig(
            init_search=True,
            state="training",
            init_log_jumprelu_threshold_value=math.log(0.03),
            init_encoder_with_decoder_transpose_factor=1.0,
            init_encoder_with_decoder_transpose=True,
            # decoder_uniform_bound=1/4096,
            bias_init_method="all_zero",
            # const_times_for_init_b_e=10000,
        ),
        trainer=TrainerConfig(
            use_batch_norm_mse=False,
            l1_coefficient=l1_coefficient,
            l1_coefficient_warmup_steps=1.0,
            sparsity_loss_type="tanh",
            tanh_stretch_coefficient=1.0,
            lr=lr,
            lr_scheduler_name="constantwithwarmup",
            lr_warm_up_steps=1000,
            total_training_tokens=900_000_000,
            log_frequency=1000,
            feature_sampling_window=1000,
            eval_frequency=1000000,
            n_checkpoints=2,
            check_point_save_mode="linear",
            exp_result_path=f"/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/JR_CC_tests_shared/shared/{subject_model}_l{l}_{expfactor}x_lr{lr}_jumprelu_l1coef{l1_coefficient}_shared_decoder_sparsity_factor{args.shared_decoder_sparsity_factor}_p1r1_fulltokens",
        ),
        wandb=WandbConfig(
            log_to_wandb=True,
            wandb_project="boi-sweep",
            exp_name=f"shared_{subject_model}_l{l}_{expfactor}x_lr{lr}_jumprelu_l1coef{l1_coefficient}_shared_factor{args.shared_decoder_sparsity_factor}_p1r1_fulltokens",
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path=f"{path}",
                    type="activations",
                    name="OpenR1-Math-220k-L15R-800M",
                    device="cuda",
                    dtype=torch.float32,
                    num_workers=4,
                ),
                ActivationFactoryActivationsSource(
                    path=f"{p_path}",
                    type="activations",
                    name="SlimPajama-3B-LXR-800M",
                    device="cuda",
                    dtype=torch.float32,
                    num_workers=4,
                ),
            ],
            target=ActivationFactoryTarget.BATCHED_ACTIVATIONS_1D,
            hook_points=[f"blocks.{l}.hook_resid_post"],
            batch_size=4096 * 4,
            buffer_size=4096 * 8,
            buffer_shuffle=BufferShuffleConfig(perm_seed=42, generator_device="cuda"),
            ignore_token_ids=[128001, 128009],
        ),
        sae_name=f"BOI-shared-{subject_model}-l1{l1_coefficient}",
        sae_series="BOI-shared",
        mongo=None,
        eval=False,
        data_parallel_size=1,
        model_parallel_size=1,
    )
    train_sae(settings)
