import os
from dataclasses import asdict
from typing import cast

import torch
import wandb
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_saes.activation.activation_dataset import make_activation_dataset
from lm_saes.activation.activation_store import ActivationStore
from lm_saes.analysis.features_to_logits import features_to_logits
from lm_saes.analysis.sample_feature_activations import sample_feature_activations
from lm_saes.config import (
    ActivationGenerationConfig,
    FeaturesDecoderConfig,
    LanguageModelSAEAnalysisConfig,
    LanguageModelSAEPruningConfig,
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingConfig,
)
from lm_saes.database import MongoClient
from lm_saes.evals import run_evals
from lm_saes.post_processing import post_process_topk_to_jumprelu_for_inference
from lm_saes.sae import SparseAutoEncoder
from lm_saes.sae_training import prune_sae, train_sae
from lm_saes.utils.misc import is_master


def language_model_sae_runner(cfg: LanguageModelSAETrainingConfig):
    hf_model = AutoModelForCausalLM.from_pretrained(
        (cfg.lm.model_name if cfg.lm.model_from_pretrained_path is None else cfg.lm.model_from_pretrained_path),
        cache_dir=cfg.lm.cache_dir,
        local_files_only=cfg.lm.local_files_only,
        torch_dtype=cfg.lm.dtype,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(
        (cfg.lm.model_name if cfg.lm.model_from_pretrained_path is None else cfg.lm.model_from_pretrained_path),
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=True,
    )

    model = HookedTransformer.from_pretrained_no_processing(
        cfg.lm.model_name,
        use_flash_attn=cfg.lm.use_flash_attn,
        device=cfg.lm.device,
        cache_dir=cfg.lm.cache_dir,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        dtype=cfg.lm.dtype,
    )
    model.offload_params_after(cfg.act_store.hook_points[-1], torch.tensor([[0]], device=cfg.lm.device))
    model.eval()
    activation_store = ActivationStore.from_config(model=model, cfg=cfg.act_store)

    if not cfg.finetuning and (
        cfg.sae.norm_activation == "dataset-wise"
        and cfg.sae.dataset_average_activation_norm is None
        or cfg.sae.init_decoder_norm is None
    ):
        sae = SparseAutoEncoder.from_initialization_searching(
            activation_store=activation_store,
            cfg=cfg,
        )
    else:
        sae = SparseAutoEncoder.from_config(cfg=cfg.sae)

        if cfg.finetuning:
            # Fine-tune SAE with frozen encoder weights and bias
            sae.train_finetune_for_suppression_parameters()

    if is_master():
        cfg.sae.save_hyperparameters(cfg.exp_result_path)
        cfg.lm.save_lm_config(cfg.exp_result_path)

    if cfg.wandb.log_to_wandb and is_master():
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
        with open(os.path.join(cfg.exp_result_path, "train_wandb_id.txt"), "w") as f:
            f.write(wandb_run.id)
        wandb.watch(sae, log="all")

    # train SAE
    sae = train_sae(
        model,
        sae,
        activation_store,
        cfg,
    )

    if cfg.wandb.log_to_wandb and is_master():
        wandb.finish()

    return sae


def language_model_sae_prune_runner(cfg: LanguageModelSAEPruningConfig):
    cfg.sae.save_hyperparameters(os.path.join(cfg.exp_result_path))
    cfg.lm.save_lm_config(os.path.join(cfg.exp_result_path))
    sae = SparseAutoEncoder.from_config(cfg=cfg.sae)
    hf_model = AutoModelForCausalLM.from_pretrained(
        (cfg.lm.model_name if cfg.lm.model_from_pretrained_path is None else cfg.lm.model_from_pretrained_path),
        cache_dir=cfg.lm.cache_dir,
        local_files_only=cfg.lm.local_files_only,
        torch_dtype=cfg.lm.dtype,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(
        (cfg.lm.model_name if cfg.lm.model_from_pretrained_path is None else cfg.lm.model_from_pretrained_path),
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=True,
    )
    model = HookedTransformer.from_pretrained_no_processing(
        cfg.lm.model_name,
        use_flash_attn=cfg.lm.use_flash_attn,
        device=cfg.lm.device,
        cache_dir=cfg.lm.cache_dir,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        dtype=cfg.lm.dtype,
    )
    model.eval()
    activation_store = ActivationStore.from_config(model=model, cfg=cfg.act_store)
    if cfg.wandb.log_to_wandb and is_master():
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
        with open(os.path.join(cfg.exp_result_path, "prune_wandb_id.txt"), "w") as f:
            f.write(wandb_run.id)

    sae = prune_sae(
        sae,
        activation_store,
        cfg,
    )

    result = run_evals(model, sae, activation_store, cfg, 0)

    # Print results in tabular format
    if is_master():
        for key, value in result.items():
            print(f"{key}: {value}")

    if cfg.wandb.log_to_wandb and is_master():
        wandb.finish()


def language_model_sae_eval_runner(cfg: LanguageModelSAERunnerConfig):
    sae = SparseAutoEncoder.from_config(cfg=cfg.sae)
    hf_model = AutoModelForCausalLM.from_pretrained(
        (cfg.lm.model_name if cfg.lm.model_from_pretrained_path is None else cfg.lm.model_from_pretrained_path),
        cache_dir=cfg.lm.cache_dir,
        local_files_only=cfg.lm.local_files_only,
    )

    hf_tokenizer = AutoTokenizer.from_pretrained(
        (cfg.lm.model_name if cfg.lm.model_from_pretrained_path is None else cfg.lm.model_from_pretrained_path),
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=True,
    )
    model = HookedTransformer.from_pretrained_no_processing(
        cfg.lm.model_name,
        use_flash_attn=cfg.lm.use_flash_attn,
        device=cfg.lm.device,
        cache_dir=cfg.lm.cache_dir,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        dtype=cfg.lm.dtype,
    )

    model.eval()
    activation_store = ActivationStore.from_config(model=model, cfg=cfg.act_store)

    if cfg.wandb.log_to_wandb and is_master():
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
        with open(os.path.join(cfg.exp_result_path, "eval_wandb_id.txt"), "w") as f:
            f.write(wandb_run.id)

    result = run_evals(model, sae, activation_store, cfg, 0)

    # Print results in tabular format
    if is_master():
        for key, value in result.items():
            print(f"{key}: {value}")

    if cfg.wandb.log_to_wandb and is_master():
        wandb.finish()

    return sae


def activation_generation_runner(cfg: ActivationGenerationConfig):
    hf_model = AutoModelForCausalLM.from_pretrained(
        (cfg.lm.model_name if cfg.lm.model_from_pretrained_path is None else cfg.lm.model_from_pretrained_path),
        cache_dir=cfg.lm.cache_dir,
        local_files_only=cfg.lm.local_files_only,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(
        (cfg.lm.model_name if cfg.lm.model_from_pretrained_path is None else cfg.lm.model_from_pretrained_path),
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=True,
    )
    model = HookedTransformer.from_pretrained_no_processing(
        cfg.lm.model_name,
        use_flash_attn=cfg.lm.use_flash_attn,
        device=cfg.lm.device,
        cache_dir=cfg.lm.cache_dir,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        dtype=cfg.lm.dtype,
    )
    model.eval()

    make_activation_dataset(model, cfg)


def sample_feature_activations_runner(cfg: LanguageModelSAEAnalysisConfig):
    sae = SparseAutoEncoder.from_config(cfg=cfg.sae)

    if cfg.sae.tp_size > 1:
        plan = {
            "encoder": ColwiseParallel(output_layouts=Replicate()),
            "decoder": RowwiseParallel(output_layouts=Replicate()),
        }
        if cfg.sae.use_glu_encoder:
            plan["encoder_glu"] = ColwiseParallel(output_layouts=Replicate())
        sae = cast(SparseAutoEncoder, parallelize_module(sae, device_mesh=sae.device_mesh["tp"], parallelize_plan=plan))

    sae.decoder.weight = None  # type: ignore[assignment]
    torch.cuda.empty_cache()

    hf_model = AutoModelForCausalLM.from_pretrained(
        (cfg.lm.model_name if cfg.lm.model_from_pretrained_path is None else cfg.lm.model_from_pretrained_path),
        cache_dir=cfg.lm.cache_dir,
        local_files_only=cfg.lm.local_files_only,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(
        (cfg.lm.model_name if cfg.lm.model_from_pretrained_path is None else cfg.lm.model_from_pretrained_path),
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=True,
    )
    model = HookedTransformer.from_pretrained_no_processing(
        cfg.lm.model_name,
        use_flash_attn=cfg.lm.use_flash_attn,
        device=cfg.lm.device,
        cache_dir=cfg.lm.cache_dir,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        dtype=cfg.lm.dtype,
    )
    model.eval()
    client = MongoClient(cfg.mongo.mongo_uri, cfg.mongo.mongo_db)
    if is_master():
        client.create_dictionary(cfg.exp_name, cfg.exp_result_path, cfg.sae.d_sae, cfg.exp_series)

    for chunk_id in range(cfg.n_sae_chunks):
        activation_store = ActivationStore.from_config(model=model, cfg=cfg.act_store)
        result = sample_feature_activations(sae, model, activation_store, cfg, chunk_id, cfg.n_sae_chunks)
        for i in range(len(result["index"].cpu().numpy().tolist())):
            client.update_feature(
                cfg.exp_name,
                result["index"][i].item(),
                {
                    "act_times": result["act_times"][i].item(),
                    "max_feature_acts": result["max_feature_acts"][i].item(),
                    "feature_acts_all": result["feature_acts_all"][i]
                    .cpu()
                    .float()
                    .numpy(),  # use .float() to convert bfloat16 to float32
                    "analysis": [
                        {
                            "name": v["name"],
                            "feature_acts": v["feature_acts"][i].cpu().float().numpy(),
                            "contexts": v["contexts"][i].cpu().numpy(),
                        }
                        for v in result["analysis"]
                    ],
                },
                dictionary_series=cfg.exp_series,
            )

        del result
        del activation_store
        torch.cuda.empty_cache()


@torch.no_grad()
def features_to_logits_runner(cfg: FeaturesDecoderConfig):
    sae = SparseAutoEncoder.from_config(cfg=cfg.sae)

    hf_model = AutoModelForCausalLM.from_pretrained(
        (cfg.lm.model_name if cfg.lm.model_from_pretrained_path is None else cfg.lm.model_from_pretrained_path),
        cache_dir=cfg.lm.cache_dir,
        local_files_only=cfg.lm.local_files_only,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(
        (cfg.lm.model_name if cfg.lm.model_from_pretrained_path is None else cfg.lm.model_from_pretrained_path),
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=True,
    )
    model = HookedTransformer.from_pretrained_no_processing(
        cfg.lm.model_name,
        use_flash_attn=cfg.lm.use_flash_attn,
        device=cfg.lm.device,
        cache_dir=cfg.lm.cache_dir,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        dtype=cfg.lm.dtype,
    )
    model.eval()

    result_dict = features_to_logits(sae, model, cfg)

    client = MongoClient(cfg.mongo.mongo_uri, cfg.mongo.mongo_db)

    for feature_index, logits in result_dict.items():
        sorted_indeces = torch.argsort(logits)
        top_negative_logits = logits[sorted_indeces[: cfg.top]].cpu().tolist()
        top_positive_logits = logits[sorted_indeces[-cfg.top :]].cpu().tolist()
        top_negative_ids = sorted_indeces[: cfg.top].tolist()
        top_positive_ids = sorted_indeces[-cfg.top :].tolist()
        top_negative_tokens = model.to_str_tokens(torch.tensor(top_negative_ids), prepend_bos=False)
        top_positive_tokens = model.to_str_tokens(torch.tensor(top_positive_ids), prepend_bos=False)
        counts, edges = torch.histogram(
            logits.cpu(), bins=60, range=(-60.0, 60.0)
        )  # Why logits.cpu():Could not run 'aten::histogram.bin_ct' with arguments from the 'CUDA' backend
        client.update_feature(
            cfg.exp_name,
            int(feature_index),
            {
                "logits": {
                    "top_negative": [
                        {"token_id": id, "logit": logit, "token": token}
                        for id, logit, token in zip(top_negative_ids, top_negative_logits, top_negative_tokens)
                    ],
                    "top_positive": [
                        {"token_id": id, "logit": logit, "token": token}
                        for id, logit, token in zip(top_positive_ids, top_positive_logits, top_positive_tokens)
                    ],
                    "histogram": {
                        "counts": counts.cpu().tolist(),
                        "edges": edges.cpu().tolist(),
                    },
                }
            },
            dictionary_series=cfg.exp_series,
        )


@torch.no_grad()
def post_process_topk_to_jumprelu_runner(cfg: LanguageModelSAERunnerConfig):
    sae = SparseAutoEncoder.from_config(cfg=cfg.sae)
    hf_model = AutoModelForCausalLM.from_pretrained(
        (cfg.lm.model_name if cfg.lm.model_from_pretrained_path is None else cfg.lm.model_from_pretrained_path),
        cache_dir=cfg.lm.cache_dir,
        local_files_only=cfg.lm.local_files_only,
    )

    hf_tokenizer = AutoTokenizer.from_pretrained(
        (cfg.lm.model_name if cfg.lm.model_from_pretrained_path is None else cfg.lm.model_from_pretrained_path),
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=True,
    )
    model = HookedTransformer.from_pretrained_no_processing(
        cfg.lm.model_name,
        use_flash_attn=cfg.lm.use_flash_attn,
        device=cfg.lm.device,
        cache_dir=cfg.lm.cache_dir,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        dtype=cfg.lm.dtype,
    )

    model.eval()
    activation_store = ActivationStore.from_config(model=model, cfg=cfg.act_store)
    post_process_topk_to_jumprelu_for_inference(sae, activation_store, cfg)
