import torch
import os
import torch.distributed as dist
import sys

sys.path.insert(0, os.getcwd())

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from core.config import LanguageModelSAEAnalysisConfig, SAEConfig
from core.runner import sample_feature_activations_runner

use_ddp = False

if use_ddp:
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

cfg = LanguageModelSAEAnalysisConfig(
    # LanguageModelConfig
    model_name = "gpt2",

    # TextDatasetConfig
    dataset_path = "data/openwebtext",
    is_dataset_tokenized = False,
    is_dataset_on_disk = True,
    concat_tokens = False,
    context_size = 256,
    store_batch_size = 16,

    # ActivationStoreConfig
    hook_point = f"blocks.9.hook_mlp_out",
    
    # SAEConfig
    **SAEConfig.get_hyperparameters("test", "results", "final.pt"),

    # LanguageModelSAEAnalysisConfig
    total_analyzing_tokens = 10_000_000,
    n_samples = 150,
    n_bins = 640,
    analysis_name = "test",

    # RunnerConfig
    use_ddp = use_ddp,
    device = "cuda",
    seed = 42,
    dtype = torch.float32,

    exp_name = "test",
)


sample_feature_activations_runner(cfg)

if use_ddp:
    dist.destroy_process_group()
    torch.cuda.empty_cache()