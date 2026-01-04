import os

import torch

from lm_saes import MongoDBConfig
from lm_saes.database import MongoClient

device = "cuda" if torch.cuda.is_available() else "cpu"
client = MongoClient(MongoDBConfig())
sae_series = os.environ.get("SAE_SERIES", "default")
tokenizer_only = os.environ.get("TOKENIZER_ONLY", "false").lower() == "true"
