import torch
import os
import torch.distributed as dist

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY"

from core.config import LanguageModelSAETrainingConfig
from core.runner import language_model_sae_runner

use_ddp = False

if use_ddp:
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

cfg = LanguageModelSAETrainingConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name = "gpt2-small",
    hook_point = "blocks.0.hook_resid_pre",
    d_model = 768,
    dataset_path = "Skylion007/openwebtext",
    is_dataset_tokenized=False,
    is_dataset_on_disk=False,
    
    # SAE Parameters
    expansion_factor = 32,
    decoder_bias_init_method = "geometric_median",
    
    # Training Parameters
    lr = 0.0004,
    l1_coefficient = 0.00008,
    lr_scheduler_name="constantwithwarmup",
    train_batch_size = 4096,
    context_size = 128,
    lr_warm_up_steps= 5000,
    
    # Activation Store Parameters
    n_tokens_in_buffer = 500_000,
    total_training_tokens = 300_000_000,
    store_batch_size = 32,
    
    # Dead Neurons and Sparsity
    use_ghost_grads = True,
    feature_sampling_window = 1000,
    dead_feature_window = 5000,
    dead_feature_threshold = 1e-6,
    
    # WANDB
    log_to_wandb = True,
    wandb_project= "gpt2-sae",
    wandb_entity = None,

    # Evaluation
    eval_frequency=1000,
    
    # Misc
    log_frequency=100,
    
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