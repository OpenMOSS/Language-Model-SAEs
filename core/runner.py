from typing import Any, cast

import wandb

from transformers import AutoModelForCausalLM

from transformer_lens import HookedTransformer

from core.config import ActivationGenerationConfig, LanguageModelSAETrainingConfig
from core.sae import SparseAutoEncoder
from core.activation.activation_dataset import make_activation_dataset
from core.activation.activation_store import ActivationStore
from core.sae_training import train_sae

def language_model_sae_runner(cfg: LanguageModelSAETrainingConfig):
    if cfg.from_pretrained_path is not None:
        # TODO: Implement this
        raise NotImplementedError
    else:
        hf_model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir=cfg.cache_dir, local_files_only=cfg.local_files_only)
        model = HookedTransformer.from_pretrained('gpt2', device=cfg.device, cache_dir=cfg.cache_dir, hf_model=hf_model)
        model.eval()
        sae = SparseAutoEncoder(cfg).to(cfg.device)
        activation_store = ActivationStore.from_config(model=model, cfg=cfg)
        
    if cfg.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
        wandb.init(project=cfg.wandb_project, config=cast(Any, cfg), name=cfg.run_name, entity=cfg.wandb_entity)

    # train SAE
    sparse_autoencoder = train_sae(
        model,
        sae,
        activation_store,
        cfg,
    )

    if cfg.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
        wandb.finish()

    return sparse_autoencoder

def activation_generation_runner(cfg: ActivationGenerationConfig):
    model = HookedTransformer.from_pretrained('gpt2', device=cfg.device, cache_dir=cfg.cache_dir)
    model.eval()
    
    make_activation_dataset(model, cfg)