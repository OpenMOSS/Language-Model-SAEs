import torch
import os
import torch.distributed as dist

from core.config import ActivationGenerationConfig
from core.runner import language_model_sae_runner

use_ddp = False

if use_ddp:
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

cfg = ActivationGenerationConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name = "gpt2",
    hook_points = ['hook_pos_embed'] + [f'blocks.{i}.hook_attn_out' for i in range(12)] + [f'blocks.{i}.hook_mlp_out' for i in range(12)],
    d_model = 768,
    dataset_path = "Skylion007/openwebtext",
    is_dataset_tokenized=False,
    is_dataset_on_disk=False,
    
    use_ddp = use_ddp,
    device = "cuda",
    seed = 42,
    n_checkpoints = 10,
    checkpoint_path = "checkpoints",
    dtype = torch.float32,
)

sparse_autoencoder = language_model_sae_runner(cfg)

if use_ddp:
    dist.destroy_process_group()