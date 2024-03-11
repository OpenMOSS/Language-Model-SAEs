from typing import Any, cast

import wandb

import torch

from transformers import AutoModelForCausalLM

from transformer_lens import HookedTransformer

from core.config import ActivationGenerationConfig, LanguageModelSAEAnalysisConfig, LanguageModelSAETrainingConfig
from core.sae import SparseAutoEncoder
from core.activation.activation_dataset import make_activation_dataset
from core.activation.activation_store import ActivationStore
from core.sae_training import train_sae
from core.analysis.sample_feature_activations import sample_feature_activations

def language_model_sae_runner(cfg: LanguageModelSAETrainingConfig):
    sae = SparseAutoEncoder(cfg=cfg)
    if cfg.from_pretrained_path is not None:
        sae.load_state_dict(torch.load(cfg.from_pretrained_path, map_location=cfg.device)["sae"])
    hf_model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir=cfg.cache_dir, local_files_only=cfg.local_files_only)
    model = HookedTransformer.from_pretrained('gpt2', device=cfg.device, cache_dir=cfg.cache_dir, hf_model=hf_model)
    model.eval()
    activation_store = ActivationStore.from_config(model=model, cfg=cfg)
        
    if cfg.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
        wandb.init(project=cfg.wandb_project, config=cast(Any, cfg), name=cfg.run_name, entity=cfg.wandb_entity)

    # train SAE
    sae = train_sae(
        model,
        sae,
        activation_store,
        cfg,
    )

    if cfg.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
        wandb.finish()

    return sae

def activation_generation_runner(cfg: ActivationGenerationConfig):
    model = HookedTransformer.from_pretrained('gpt2', device=cfg.device, cache_dir=cfg.cache_dir)
    model.eval()
    
    make_activation_dataset(model, cfg)

def sample_feature_activations_runner(cfg: LanguageModelSAEAnalysisConfig):
    sae = SparseAutoEncoder(cfg=cfg)
    if cfg.from_pretrained_path is not None:
        sae.load_state_dict(torch.load(cfg.from_pretrained_path, map_location=cfg.device)["sae"])

    hf_model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir=cfg.cache_dir, local_files_only=cfg.local_files_only)
    model = HookedTransformer.from_pretrained('gpt2', device=cfg.device, cache_dir=cfg.cache_dir, hf_model=hf_model)
    model.eval()

    activation_store = ActivationStore.from_config(model=model, cfg=cfg)
    sample_feature_activations(sae, activation_store, cfg)

    if not cfg.use_ddp or cfg.rank == 0:
        feature_activations = torch.load(cfg.analysis_save_path, map_location=cfg.device)
        act_times = feature_activations["act_times"][0]
        elt = feature_activations["elt"][0][0]
        feature_act = feature_activations["feature_acts"][0][0]
        context = feature_activations["contexts"][0][0]
        position = feature_activations["positions"][0][0]
        print(f"act_times: {act_times}")
        print(f"elt: {elt}")
        print(f"feature_act: {feature_act}")
        print(f"context: {context}")
        print(f"position: {position}")
        print(f"context str: {model.tokenizer.decode(context)}")
        print(f"token str: {model.tokenizer.decode([context[position]])}")