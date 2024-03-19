import torch
import os
import torch.distributed as dist

from core.config import ActivationGenerationConfig
from core.runner import activation_generation_runner

use_ddp = False

if use_ddp:
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

cfg = ActivationGenerationConfig(
    model_name = "gpt2",
    hook_points = ['hook_pos_embed'] + [f'blocks.{i}.hook_attn_out' for i in range(12)] + [f'blocks.{i}.hook_mlp_out' for i in range(12)],
    d_model = 768,
    dataset_path = "Skylion007/openwebtext",
    is_dataset_tokenized=False,
    is_dataset_on_disk=False,
    
    use_ddp = use_ddp,
    device = "cuda",
    dtype = torch.float32,
)

activation_generation_runner(cfg)

if use_ddp:
    dist.destroy_process_group()