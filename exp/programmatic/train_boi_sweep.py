import argparse
import os

import torch
import torch.distributed as dist

from lm_saes import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
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
args = parser.parse_args()
lr = args.lr
l = args.l

if __name__ == "__main__":
    path_dict = {
        0: "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/gen_act/b-1d",
        1: "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/gen_act/i-1d",
        2: "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/gen_act/o-1d",
    }
    # 0, 1, 2, is for b i o repectively
    rank = dist.get_rank()
    path = path_dict.get(rank)
    modelname_dict = {0: "base_llama", 1: "Instruct_llama", 2: "o_llama"}
    subject_model = modelname_dict.get(rank)

    settings = TrainSAESettings(
        sae=CrossCoderConfig(
            sae_type="crosscoder",
            hook_point_in=f"blocks.{l}.hook_resid_post",
            hook_point_out=f"blocks.{l}.hook_resid_post",
            d_model=4096,
            expansion_factor=32,
            act_fn="topk",
            norm_activation="dataset-wise",
            sparsity_include_decoder_norm=True,
            top_k=64,
            dtype=torch.bfloat16,
            device="cuda",
        ),
        initializer=InitializerConfig(
            init_search=True,
            state="training",
        ),
        trainer=TrainerConfig(
            initial_k=2048,
            lr=lr,
            lr_scheduler_name="constantwithwarmup",
            lr_warm_up_steps=5000,
            total_training_tokens=800_000_000,
            log_frequency=200,
            feature_sampling_window=500,
            eval_frequency=1000000,
            n_checkpoints=0,
            check_point_save_mode="linear",
            exp_result_path=f"/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/result/{subject_model}_l{l}_lr{lr}_stdtoken",
        ),
        wandb=WandbConfig(
            log_to_wandb=True,
            wandb_project="boi-sweep",
            exp_name=f"{subject_model}_l{l}_lr{lr}_stdtoken",
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path=f"{path}",
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
        sae_name="BOI-sweeplr",
        sae_series="BOI-sweeplr",
        mongo=None,
        eval=False,
        data_parallel_size=1,
        model_parallel_size=1,
    )
    train_sae(settings)
