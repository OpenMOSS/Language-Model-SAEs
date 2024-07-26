import torch
import sys
sys.path.append('/remote-home/jxwang/project/Language-Model-SAEs/src')
sys.path.append('/remote-home/jxwang/project/Language-Model-SAEs/TransformerLens')
from lm_saes.config import LanguageModelSAEAnalysisConfig, SAEConfig
from lm_saes.runner import sample_feature_activations_runner

layer = 24
cfg = LanguageModelSAEAnalysisConfig.from_flattened(dict(
    # LanguageModelConfig
    model_name = "meta-llama/Meta-Llama-3-8B",
    model_from_pretrained_path="/remote-home/share/models/llama3_hf/Meta-Llama-3-8B",
    use_flash_attn = True,

    # TextDatasetConfig
    dataset_path = "/remote-home/share/research/mechinterp/redpajama_2B",
    is_dataset_tokenized = False,
    is_dataset_on_disk = True,
    concat_tokens = False,
    context_size = 1024,
    store_batch_size = 16,
    prepend_bos = False,

    # ActivationStoreConfig
    hook_points = [f"blocks.{layer}.hook_resid_post"],
    
    # SAEConfig
    **SAEConfig.from_pretrained(f"/remote-home/share/research/mechinterp/Llama3.0SAE_for_dev/LlamaBase-32x/LlamaBase-L{layer}R-32x-2048bs-lr-0.0006-l1-1.5e-05-fix_norm-None-clip-1.0-").to_dict(),  # Load the hyperparameters from the trained model.

    # LanguageModelSAEAnalysisConfig
    total_analyzing_tokens = 200_000_000,
    n_sae_chunks = 5,
    subsample = {                           # The subsample configuration. The key is the name of the subsample, and the value is the proportion of activation and number of samples.
        "top_activations": {"proportion": 1.0, "n_samples": 32},
        "subsample-0.9": {"proportion": 0.9, "n_samples": 8},
        "subsample-0.8": {"proportion": 0.8, "n_samples": 8},
        "subsample-0.7": {"proportion": 0.7, "n_samples": 8},
        "subsample-0.6": {"proportion": 0.6, "n_samples": 8},
        "subsample-0.5": {"proportion": 0.5, "n_samples": 8},
        "subsample-0.4": {"proportion": 0.4, "n_samples": 8},
        "subsample-0.3": {"proportion": 0.3, "n_samples": 8},
        "subsample-0.2": {"proportion": 0.2, "n_samples": 8},
        "subsample-0.1": {"proportion": 0.1, "n_samples": 8},
    },

    # MongoConfig
    mongo_db="mechinterp",                  # MongoDB database name. We use MongoDB to store the activation samples.
    mongo_uri="mongodb://10.176.52.106:20276",  # MongoDB URI.

    # RunnerConfig
    device = "cuda",
    seed = 42,
    dtype = torch.bfloat16,

    exp_name = f"LlamaBase-L{layer}R-32x-2048bs-lr-0.0006-l1-1.5e-05-fix_norm-None-clip-1.0-",
    exp_series = "LlamaBase-dictionary",
    exp_result_dir = "results",
))

sample_feature_activations_runner(cfg)