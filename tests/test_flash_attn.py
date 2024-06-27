from typing import Any, cast
import os
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/remote-home/fkzhu/zfk/engineering/Language-Model-SAEs/src")

import wandb
import logging

from dataclasses import asdict

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import convert_gpt2_weights

from lm_saes.config import (
    ActivationGenerationConfig,
    LanguageModelSAEAnalysisConfig,
    LanguageModelSAETrainingConfig,
    LanguageModelSAERunnerConfig,
    LanguageModelSAEPruningConfig,
    FeaturesDecoderConfig,
)
from lm_saes.database import MongoClient
from lm_saes.evals import run_evals
from lm_saes.sae import SparseAutoEncoder
from lm_saes.activation.activation_dataset import make_activation_dataset
from lm_saes.activation.activation_store import ActivationStore
from lm_saes.sae_training import prune_sae, train_sae
from lm_saes.analysis.sample_feature_activations import sample_feature_activations
from lm_saes.analysis.features_to_logits import features_to_logits
from lm_saes.activation.activation_source import TokenActivationSource
from lm_saes.activation.token_source import TokenSource

from datasets import load_dataset
from transformer_lens import HookedTransformer

import pytest

@pytest.fixture
def dataset():
    return load_dataset("Skylion007/openwebtext", split="train")

@pytest.fixture
def dataloader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=32)

@pytest.fixture
def model():
    return HookedTransformer.from_pretrained('gpt2')

@pytest.mark.parametrize(
        'config', [(15, 'M')],
        indirect=['config'])
def test_language_model_sae_runner(config: LanguageModelSAETrainingConfig):
    cfg = config
    cfg.sae.save_hyperparameters(os.path.join(cfg.exp_result_dir, cfg.exp_name))
    cfg.lm.save_lm_config(os.path.join(cfg.exp_result_dir, cfg.exp_name))
    sae = SparseAutoEncoder.from_config(cfg=cfg.sae)

    if cfg.finetuning:
        # Fine-tune SAE with frozen encoder weights and bias
        sae.train_finetune_for_suppression_parameters()

    hf_model = AutoModelForCausalLM.from_pretrained(
        (
            cfg.lm.model_name
            if cfg.lm.model_from_pretrained_path is None
            else cfg.lm.model_from_pretrained_path
        ),
        cache_dir=cfg.lm.cache_dir,
        local_files_only=cfg.lm.local_files_only,
        torch_dtype=cfg.lm.dtype,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(
        (
            cfg.lm.model_name
            if cfg.lm.model_from_pretrained_path is None
            else cfg.lm.model_from_pretrained_path
        ),
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=True,
    )
    
    model = HookedTransformer.from_pretrained(
        cfg.lm.model_name,
        use_flash_attn=cfg.lm.use_flash_attn,
        device=cfg.lm.device,
        cache_dir=cfg.lm.cache_dir,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        dtype=cfg.lm.dtype,
    )

    model.eval()
    logging.info(model.eval())
    activation_store = ActivationStore.from_config(model=model, cfg=cfg.act_store)

    if cfg.wandb.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
        wandb_config: dict = {
            **asdict(cfg),
            **asdict(cfg.sae),
            **asdict(cfg.lm),
        }
        del wandb_config["sae"]
        del wandb_config["lm"]
        wandb_run = wandb.init(
            project=cfg.wandb.wandb_project,
            config=wandb_config,
            name=cfg.wandb.exp_name,
            entity=cfg.wandb.wandb_entity,
        )
        with open(
            os.path.join(cfg.exp_result_dir, cfg.exp_name, "train_wandb_id.txt"), "w"
        ) as f:
            f.write(wandb_run.id)
        wandb.watch(sae, log="all")

    # # train SAE
    # sae = train_sae(
    #     model,
    #     sae,
    #     activation_store,
    #     cfg,
    # )

    # if cfg.wandb.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
    #     wandb.finish()

    # bfloat16 dtype test
    if cfg.lm.dtype == torch.bfloat16:
        for name, obj in vars(HookedTransformer).items():
            if isinstance(obj, property):
                try:
                    param = model.__getattribute__(name)
                    assert (param.dtype == torch.bfloat16)
                except:
                    logging.warning(f"Does not have attribute {name}")