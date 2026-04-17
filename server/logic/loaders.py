from functools import lru_cache

import torch
from datasets import Dataset
from torch.distributed.device_mesh import DeviceMesh

from lm_saes import (
    LanguageModel,
    SparseDictionary,
    SparseDictionaryConfig,
    load_dataset_shard,
    load_model,
)
from lm_saes.utils.timer import timer
from server.config import (
    LRU_CACHE_SIZE_DATASETS,
    LRU_CACHE_SIZE_MODELS,
    LRU_CACHE_SIZE_SAES,
    client,
    device,
    sae_series,
    tokenizer_only,
)
from server.utils.common import synchronized


@synchronized
@lru_cache(maxsize=LRU_CACHE_SIZE_MODELS)
@timer.time("get_model")
def get_model(*, name: str, device_mesh: DeviceMesh | None = None) -> LanguageModel:
    """Load and cache a language model."""
    cfg = client.get_model_cfg(name)
    if cfg is None:
        raise ValueError(f"Model {name} not found")
    if tokenizer_only:
        cfg.backend = "tokenizer_only"
    cfg.device = device
    cfg.dtype = torch.bfloat16
    return load_model(cfg, device_mesh=device_mesh)


@synchronized
@lru_cache(maxsize=LRU_CACHE_SIZE_DATASETS)
@timer.time("get_dataset")
def get_dataset(*, name: str, shard_idx: int = 0, n_shards: int = 1) -> Dataset:
    """Load and cache a dataset shard."""
    cfg = client.get_dataset_cfg(name)
    assert cfg is not None, f"Dataset {name} not found"
    return load_dataset_shard(cfg, shard_idx, n_shards)


@synchronized
@lru_cache(maxsize=LRU_CACHE_SIZE_SAES)
@timer.time("get_sae")
def get_sae(*, name: str, device_mesh: DeviceMesh | None = None) -> SparseDictionary:
    """Load and cache a sparse autoencoder."""
    path = client.get_sae_path(name, sae_series)
    assert path is not None, f"SAE {name} not found"
    sae = SparseDictionary.from_pretrained(path, device=device, dtype=torch.bfloat16, device_mesh=device_mesh)
    sae.eval()
    return sae


@synchronized
@lru_cache(maxsize=LRU_CACHE_SIZE_SAES)
@timer.time("get_sae_cfg")
def get_sae_cfg(*, name: str) -> SparseDictionaryConfig:
    sae_record = client.get_sae(name, sae_series)
    assert sae_record is not None, f"SAE {name} not found"
    return sae_record.cfg
