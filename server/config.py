import os

import torch

from llamascopium import MongoClient, MongoDBConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
client = MongoClient(MongoDBConfig())
sae_series = os.environ.get("SAE_SERIES", "default")
tokenizer_only = os.environ.get("IS_WORKER") is None and os.environ.get("NUM_WORKERS") != "0"

# LRU cache sizes (configurable via environment variables)
LRU_CACHE_SIZE_SAMPLES = int(os.environ.get("LRU_CACHE_SIZE_SAMPLES", "128"))
LRU_CACHE_SIZE_MODELS = int(os.environ.get("LRU_CACHE_SIZE_MODELS", "8"))
LRU_CACHE_SIZE_DATASETS = int(os.environ.get("LRU_CACHE_SIZE_DATASETS", "16"))
LRU_CACHE_SIZE_SAES = int(os.environ.get("LRU_CACHE_SIZE_SAES", "64"))
LRU_CACHE_SIZE_CIRCUITS = int(os.environ.get("LRU_CACHE_SIZE_CIRCUITS", "64"))
