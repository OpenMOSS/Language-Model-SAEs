import math
import time
import os

import torch
import numpy as np

from lm_saes import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    InitializerConfig,
    SAEConfig,
    TrainerConfig,
    TrainSAESettings,
    MongoDBConfig,
    WandbConfig,
    train_sae,
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--layer', type=int, default=0, help='layer (default: 0)')
parser.add_argument('--k', type=int, default=50, help='top_k (default: 50)')
parser.add_argument('--d_feature', type=int, default=4096, help='d_feature (default: 4096)')
parser.add_argument('--exp_factor', type=int, default=8, help='expasion factor (default: 8)')
parser.add_argument('--init_with_svd', action='store_true', help='init with svd (default: False)')
args = parser.parse_args()
l=args.layer
lr=args.lr
k=args.k
d_feature=args.d_feature
exp_factor=args.exp_factor
init_with_svd=args.init_with_svd
if init_with_svd:
    exp_name = f"L{l}A-{exp_factor}x-k{k}-lr{lr:.0e}-d_feature{d_feature}-svd-topk"
else:
    exp_name = f"L{l}A-{exp_factor}x-k{k}-lr{lr:.0e}-default-topk"
    
exp_result_path = "./results"

if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    # seed = int(time.time())
    seed = 42
    print(f"[INFO] Using seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    settings = TrainSAESettings(
        sae=SAEConfig(
            sae_type="sae",
            d_model=4096,
            expansion_factor=exp_factor,
            act_fn='topk',
            top_k=k,
            norm_activation="dataset-wise",
            sparsity_include_decoder_norm=True,
            force_unit_decoder_norm=False,
            dtype=torch.float32,
            device="cuda",
            hook_point_in=f"blocks.{l}.hook_attn_out",
            hook_point_out=f"blocks.{l}.hook_attn_out",
        ),
        initializer=InitializerConfig(
            state="training",
            grid_search_init_norm=True,
            bias_init_method="geometric_median",
            initialize_W_D_with_active_subspace=init_with_svd,
            init_encoder_bias_with_mean_hidden_pre=True,
            d_active_subspace=d_feature,
        ),
        trainer=TrainerConfig(
            amp_dtype = torch.bfloat16,
            initial_k=k,
            k_warmup_steps=0.1,
            k_schedule_type="linear",
            optimizer_type="adam",
            use_batch_norm_mse=False,
            lr=lr,
            lr_scheduler_name="constantwithwarmup",
            lr_warm_up_steps=500,
            lr_cool_down_steps=0.2,
            total_training_tokens=2_500_000_000,
            clip_grad_norm=1.0,
            log_frequency=100,
            feature_sampling_window=1000,
            eval_frequency=1000000,
            n_checkpoints=0,
            check_point_save_mode="linear",
            exp_result_path=exp_result_path,
        ),
        wandb=WandbConfig(
            log_to_wandb=True,
            wandb_project="Llama-3.1-8B-LXA-SAE-Sweep-ActFn-PaperExp",
            exp_name=exp_name,
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path="./activations",
                    type="activations",
                    name="SlimPajama",
                    device="cuda",
                    dtype=torch.float32,
                    num_workers=8,
                    prefetch=4,
                ),
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_1D,
            hook_points=[f"blocks.{l}.hook_attn_out"],
            batch_size=32768,
            buffer_size=None,
            ignore_token_ids=[],
        ),
        sae_name=exp_name,
        # sae_series="Llama-3.1-8B-LXA-SAE-Scaling-Law-PaperExp",
        sae_series="Llama-3.1-8B-LXA-SAE-Sweep-ActFn-PaperExp",
        # sae_series="Llama-3.1-8B-LXA-SAE",
        model_name="meta-llama/llama-3.1-8B",
        # mongo=MongoDBConfig(
        #     mongo_uri="mongodb://localhost:27017/",
        #     mongodb="sae_analysis"
        # ),
        model_parallel_size=os.environ.get("WORLD_SIZE", 1),
    )
    train_sae(settings)