from typing import Any, Literal, Optional, cast

import datasets
import torch
from torch.distributed.device_mesh import DeviceMesh

from lm_saes.backend.language_model import (
    LanguageModel,
    QwenLanguageModel,
    QwenVLLanguageModel,
    TransformerLensLanguageModel,
    LLaDALanguageModel,
)
from lm_saes.config import DatasetConfig, LanguageModelConfig, LLaDAConfig


def dataset_transform(data):
    if "image" in data:
        # Rename image to images
        data["images"] = data["image"]
        del data["image"]

        data["images"] = [[torch.tensor(image) for image in images] for images in data["images"]]
    return data


def load_dataset_shard(
    cfg: DatasetConfig,
    shard_idx: int,
    n_shards: int,
) -> datasets.Dataset:
    if not cfg.is_dataset_on_disk:
        dataset = datasets.load_dataset(cfg.dataset_name_or_path, split="train", cache_dir=cfg.cache_dir)
    else:
        dataset = datasets.load_from_disk(cfg.dataset_name_or_path)
    dataset = cast(datasets.Dataset, dataset)
    dataset = dataset.shard(num_shards=n_shards, index=shard_idx, contiguous=True)
    dataset.set_transform(dataset_transform)
    return dataset


def load_dataset(
    cfg: DatasetConfig,
    device_mesh: Optional[DeviceMesh] = None,
    n_shards: Optional[int] = None,
    start_shard: int = 0,
) -> tuple[datasets.Dataset, Optional[dict[str, Any]]]:
    if not cfg.is_dataset_on_disk:
        dataset = datasets.load_dataset(
            cfg.dataset_name_or_path, split="train", cache_dir=cfg.cache_dir, trust_remote_code=True
        )
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
            "shard_idx": start_shard + device_mesh.get_group("data").rank(),
            "n_shards": n_shards or device_mesh.get_group("data").size(),
        }
    else:
        shard = dataset
        shard_metadata = None
    shard.set_transform(dataset_transform)
    return shard, shard_metadata


def infer_model_backend(model_name: str) -> Literal["huggingface", "transformer_lens"]:
    if model_name.startswith("Qwen/Qwen2.5-VL"):
        return "huggingface"
    elif model_name.startswith("Qwen/Qwen2.5"):
        return "huggingface"
    elif model_name.startswith("GSAI-ML/LLaDA"):
        return "huggingface"
    else:
        return "transformer_lens"


def load_model(cfg: LanguageModelConfig) -> LanguageModel:
    backend = infer_model_backend(cfg.model_name) if cfg.backend == "auto" else cfg.backend
    if backend == "huggingface":
        if cfg.model_name.startswith("Qwen/Qwen2.5-VL"):
            return QwenVLLanguageModel(cfg)
        elif cfg.model_name.startswith("Qwen/Qwen2.5"):
            return QwenLanguageModel(cfg)
        elif cfg.model_name.startswith("GSAI-ML/LLaDA"):
            assert isinstance(cfg, LLaDAConfig)
            return LLaDALanguageModel(cfg)
        else:
            raise NotImplementedError(f"Model {cfg.model_name} not supported in HuggingFace backend.")
    elif backend == "transformer_lens":
        return TransformerLensLanguageModel(cfg)
    else:
        raise NotImplementedError(f"Backend {backend} not supported.")
