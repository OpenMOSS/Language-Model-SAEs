from functools import lru_cache

from datasets import Dataset

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.backend import LanguageModel
from lm_saes.resource_loaders import load_dataset_shard, load_model
from server.config import client, device, sae_series, tokenizer_only
from server.utils.common import synchronized


@synchronized
@lru_cache(maxsize=8)
def get_model(*, name: str) -> LanguageModel:
    """Load and cache a language model."""
    cfg = client.get_model_cfg(name)
    if cfg is None:
        raise ValueError(f"Model {name} not found")
    cfg.tokenizer_only = tokenizer_only
    cfg.device = device
    return load_model(cfg)


@synchronized
@lru_cache(maxsize=16)
def get_dataset(*, name: str, shard_idx: int = 0, n_shards: int = 1) -> Dataset:
    """Load and cache a dataset shard."""
    cfg = client.get_dataset_cfg(name)
    assert cfg is not None, f"Dataset {name} not found"
    return load_dataset_shard(cfg, shard_idx, n_shards)


@synchronized
@lru_cache(maxsize=8)
def get_sae(*, name: str) -> AbstractSparseAutoEncoder:
    """Load and cache a sparse autoencoder."""
    path = client.get_sae_path(name, sae_series)
    assert path is not None, f"SAE {name} not found"
    sae = AbstractSparseAutoEncoder.from_pretrained(path, device=device)
    sae.eval()
    return sae
