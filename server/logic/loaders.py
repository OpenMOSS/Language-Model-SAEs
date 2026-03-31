from functools import lru_cache

from datasets import Dataset

from lm_saes import SparseDictionaryConfig
from lm_saes.backend import LanguageModel
from lm_saes.models.sparse_dictionary import SparseDictionary
from lm_saes.resource_loaders import load_dataset_shard, load_model
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
def get_model(*, name: str) -> LanguageModel:
    """Load and cache a language model."""
    cfg = client.get_model_cfg(name)
    if cfg is None:
        raise ValueError(f"Model {name} not found")
    cfg.tokenizer_only = tokenizer_only
    cfg.device = device
    return load_model(cfg)


@synchronized
@lru_cache(maxsize=LRU_CACHE_SIZE_DATASETS)
def get_dataset(*, name: str, shard_idx: int = 0, n_shards: int = 1) -> Dataset:
    """Load and cache a dataset shard."""
    cfg = client.get_dataset_cfg(name)
    assert cfg is not None, f"Dataset {name} not found"
    return load_dataset_shard(cfg, shard_idx, n_shards)


@synchronized
@lru_cache(maxsize=LRU_CACHE_SIZE_SAES)
def get_sae(*, name: str) -> SparseDictionary:
    """Load and cache a sparse autoencoder."""
    path = client.get_sae_path(name, sae_series)
    assert path is not None, f"SAE {name} not found"
    sae = SparseDictionary.from_pretrained(path, device=device)
    sae.eval()
    return sae


@synchronized
@lru_cache(maxsize=LRU_CACHE_SIZE_SAES)
def get_sae_cfg(*, name: str) -> SparseDictionaryConfig:
    sae_record = client.get_sae(name, sae_series)
    assert sae_record is not None, f"SAE {name} not found"
    return sae_record.cfg
