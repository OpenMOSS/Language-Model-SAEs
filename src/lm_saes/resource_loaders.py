from typing import Any, Optional, cast

import datasets
import torch
from torch.distributed.device_mesh import DeviceMesh
from transformer_lens import HookedTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    ChameleonForConditionalGeneration,
    PreTrainedModel,
)

from lm_saes.config import DatasetConfig, LanguageModelConfig


def load_dataset_shard(
    cfg: DatasetConfig,
    shard_index: int,
    n_shards: int,
) -> datasets.Dataset:
    if not cfg.is_dataset_on_disk:
        dataset = datasets.load_dataset(cfg.dataset_name_or_path, split="train", cache_dir=cfg.cache_dir)
    else:
        dataset = datasets.load_from_disk(cfg.dataset_name_or_path)
    dataset = cast(datasets.Dataset, dataset)
    dataset = dataset.shard(num_shards=n_shards, index=shard_index, contiguous=True)
    return dataset


def load_dataset(
    cfg: DatasetConfig,
    device_mesh: Optional[DeviceMesh] = None,
    n_shards: Optional[int] = None,
    start_shard: int = 0,
) -> tuple[datasets.Dataset, Optional[dict[str, Any]]]:
    if not cfg.is_dataset_on_disk:
        dataset = datasets.load_dataset(cfg.dataset_name_or_path, split="train", cache_dir=cfg.cache_dir)
    else:
        dataset = datasets.load_from_disk(cfg.dataset_name_or_path)
    dataset = cast(datasets.Dataset, dataset)
    if device_mesh is not None:
        shard = dataset.shard(
            num_shards=n_shards or device_mesh.get_group("data").size(),
            index=start_shard + device_mesh.get_group("data").rank(),
            contiguous=True,
        )
        shard_metadata = {
            "shard_index": start_shard + device_mesh.get_group("data").rank(),
            "n_shards": n_shards or device_mesh.get_group("data").size(),
        }
    else:
        shard = dataset
        shard_metadata = None
    shard = shard.with_format("torch")
    return shard, shard_metadata


def load_model(cfg: LanguageModelConfig):
    if cfg.device == "cuda":
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device(cfg.device)

    if "chameleon" in cfg.model_name:
        hf_model = ChameleonForConditionalGeneration.from_pretrained(
            (cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path),
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            torch_dtype=cfg.dtype,
        ).to(device)  # type: ignore
    else:
        hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            (cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path),
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            torch_dtype=cfg.dtype,
        ).to(device)
    if "chameleon" in cfg.model_name:
        hf_processor = AutoProcessor.from_pretrained(
            (cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path),
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=True,
            local_files_only=cfg.local_files_only,
        )
        hf_tokenizer = None
    else:
        hf_tokenizer = AutoTokenizer.from_pretrained(
            (cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path),
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=True,
            local_files_only=cfg.local_files_only,
        )
        hf_processor = None

    model = HookedTransformer.from_pretrained_no_processing(
        cfg.model_name,
        use_flash_attn=cfg.use_flash_attn,
        device=device,
        cache_dir=cfg.cache_dir,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        processor=hf_processor,
        dtype=cfg.dtype,
    )
    model.eval()
    return model
