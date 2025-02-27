from typing import Any, Optional, cast

import datasets
from torch.distributed.device_mesh import DeviceMesh

from lm_saes.backend.language_model import LanguageModel
from lm_saes.config import DatasetConfig, LanguageModelConfig


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
    dataset = dataset.with_format("torch")
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
    shard = shard.with_format("torch")
    return shard, shard_metadata


def load_model(cfg: LanguageModelConfig) -> LanguageModel:
    raise NotImplementedError
