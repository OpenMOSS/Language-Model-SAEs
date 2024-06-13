import torch
import os
import torch.distributed as dist
from lm_saes.config import LanguageModelSAEAnalysisConfig, SAEConfig
from lm_saes.runner import sample_feature_activations_runner

cfg = LanguageModelSAEAnalysisConfig(
    # LanguageModelConfig
    model_name = "gpt2",

    # TextDatasetConfig
    dataset_path = "data/openwebtext",
    is_dataset_tokenized = False,
    is_dataset_on_disk = True,
    concat_tokens = False,
    context_size = 256,
    store_batch_size = 16,

    # ActivationStoreConfig
    hook_points = ["blocks.3.hook_mlp_out"],
    
    # SAEConfig
    **SAEConfig.from_pretrained("result/L3M").to_dict(),  # Load the hyperparameters from the trained model.

    # LanguageModelSAEAnalysisConfig
    total_analyzing_tokens = 20_000_000,
    subsample = {                           # The subsample configuration. The key is the name of the subsample, and the value is the proportion of activation and number of samples.
        "top_activations": {"proportion": 1.0, "n_samples": 80},
        "subsample-0.9": {"proportion": 0.9, "n_samples": 20},
        "subsample-0.8": {"proportion": 0.8, "n_samples": 20},
        "subsample-0.7": {"proportion": 0.7, "n_samples": 20},
        "subsample-0.5": {"proportion": 0.5, "n_samples": 20},
    },

    # MongoConfig
    mongo_db="mechinterp",                  # MongoDB database name. We use MongoDB to store the activation samples.
    mongo_uri="mongodb://localhost:27017",  # MongoDB URI.

    # RunnerConfig
    device = "cuda",
    seed = 42,
    dtype = torch.float32,

    exp_name = "L3M",
    exp_series = "default",
    exp_result_dir = "results",
)

sample_feature_activations_runner(cfg)