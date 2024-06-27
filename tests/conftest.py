import torch
import pytest

from lm_saes.config import LanguageModelSAETrainingConfig
from lm_saes.runner import language_model_sae_runner

def pytest_addoption(parser):
    parser.addoption("--layer", nargs="*", type=int, required=False, help='Layer number')
    parser.addoption("--batch_size", type=int, required=False, default=4096, help='Batchsize, default 4096')
    parser.addoption("--lr", type=float, required=False, default=8e-5, help='Learning rate, default 8e-5')
    parser.addoption("--expdir", type=str, required=False, default="/remote-home/fkzhu/zfk/engineering/Language-Model-SAEs/results", help='Export directory, default zfk/ftresults_KL')
    parser.addoption("--useddp", type=bool, required=False, default=False, help='If using distributed method, default False')
    parser.addoption('--attn_type', type=str, required=True, choices=['flash', 'normal'], default="flash", help='Use or not use log of wandb, default True')
    parser.addoption('--dtype', type=str, required=False, choices=['fp32', 'bfp16'], default="fp32", help='Dtype, default fp32')
 
@pytest.fixture
def args(request):
    return {"layer":request.config.getoption("--layer"),
            "batch_size":request.config.getoption("--batch_size"),
            "lr":request.config.getoption("--lr"),
            "expdir":request.config.getoption("--expdir"),
            "useddp":request.config.getoption("--useddp"),
            "attn_type":request.config.getoption("--attn_type"),
            "dtype":request.config.getoption("--dtype"),
            }

@pytest.fixture
def config(args, request):
    layer, hook_suffix_abbr = request.param
    HOOK_SUFFIX={"M":"hook_mlp_out", "A":"hook_attn_out", "R":"hook_resid_post"}
    LR = args['lr'] 
    TRAIN_BATCH_SIZE = args['batch_size']
    FLASH_ATTN = "FA" if args['attn_type'] == 'flash' else "noFA"
    EXPORT_DIR = args['expdir']
    DTYPE = torch.float32 if args['dtype'] == 'fp32' else torch.bfloat16
    COEF = f"{FLASH_ATTN}-bs-{TRAIN_BATCH_SIZE}-32x-{args['dtype']}"
    cfg = LanguageModelSAETrainingConfig.from_flattened(dict(
        # LanguageModelConfig
        model_name = "meta-llama/Meta-Llama-3-8B",                            # The model name or path for the pre-trained model.
        model_from_pretrained_path = "/remote-home/share/models/llama3_hf/Meta-Llama-3-8B",
        use_flash_attn = args['attn_type'],
        d_model = 4096,                                  # The hidden size of the model.

        # TextDatasetConfig
        dataset_path = "/remote-home/share/research/mechinterp/gpt2-dictionary/data/openwebtext",                   # The corpus name or path. Each of a data record should contain (and may only contain) a "text" field.
        is_dataset_tokenized = False,                   # Whether the dataset is tokenized.
        is_dataset_on_disk = True,                      # Whether the dataset is on disk. If not on disk, `datasets.load_dataset`` will be used to load the dataset, and the train split will be used for training.
        concat_tokens = False,                          # Whether to concatenate tokens into a single sequence. If False, only data record with length of non-padding tokens larger than `context_size` will be used.
        context_size = 256,                             # The sequence length of the text dataset.
        store_batch_size = 32,                          # The batch size for loading the corpus.

        # ActivationStoreConfig
        hook_points = [f"blocks.{layer}.{HOOK_SUFFIX[hook_suffix_abbr]}"],        # Hook points to store activations from, i.e. the layer output of which is used for training/evaluating the dictionary. Will run until the last hook point in the list, so make sure to order them correctly.
        use_cached_activations = False,                 # Whether to use cached activations. Caching activation is now not recommended, as it may consume extremely large disk space. (May be tens of TBs for corpus like `openwebtext`)
        n_tokens_in_buffer = 500_000,                   # The number of tokens to store in the activation buffer. The buffer is used to shuffle the activations before training the dictionary.
        
        # SAEConfig
        hook_point_in = f"blocks.{layer}.{HOOK_SUFFIX[hook_suffix_abbr]}",
        hook_point_out = f"blocks.{layer}.{HOOK_SUFFIX[hook_suffix_abbr]}",
        expansion_factor = 32,                          # The expansion factor of the dictionary. d_sae = expansion_factor * d_model.
        norm_activation = "token-wise",                 # The normalization method for the activations. Can be "token-wise", "batch-wise" or "none".
        decoder_exactly_unit_norm = False,              # Whether to enforce the decoder to have exactly unit norm. If False, the decoder will have less than or equal to unit norm.
        use_glu_encoder = False,                        # Whether to use the Gated Linear Unit (GLU) for the encoder.
        l1_coefficient = 1.2e-4,                        # The L1 regularization coefficient for the feature activations.
        lp = 1,                                         # The p-norm to use for the L1 regularization.
        use_ghost_grads = True,                         # Whether to use the ghost gradients for saving dead features.

        # LanguageModelSAETrainingConfig
        total_training_tokens = 320_000_000,          # The total number of tokens to train the dictionary.
        lr = 4e-4,                                      # The learning rate for the dictionary training.
        betas = (0, 0.9999),                            # The betas for the Adam optimizer.
        lr_scheduler_name = "constantwithwarmup",       # The learning rate scheduler name. Can be "constant", "constantwithwarmup", "linearwarmupdecay", "cosineannealing", "cosineannealingwarmup" or "exponentialwarmup".
        lr_warm_up_steps = 5000,                        # The number of warm-up steps for the learning rate.
        lr_cool_down_steps = 10000,                     # The number of cool-down steps for the learning rate. Currently only used for the "constantwithwarmup" scheduler.
        train_batch_size = TRAIN_BATCH_SIZE,                        # The batch size for training the dictionary, i.e. the number of token activations in a batch.
        feature_sampling_window = 1000,                 # The window size for sampling the feature activations.
        dead_feature_window = 5000,                     # The window size for detecting the dead features.
        dead_feature_threshold = 1e-6,                  # The threshold for detecting the dead features.
        eval_frequency = 1000,                          # The step frequency for evaluating the dictionary.
        log_frequency = 100,                            # The step frequency for logging the training information (to wandb).
        n_checkpoints = 10,                             # The number of checkpoints to save during the training.

        # WandbConfig
        log_to_wandb = False,                            # Whether to log the training information to wandb.
        wandb_project= "flashattn",                      # The wandb project name.
        
        # RunnerConfig
        device = "cuda",                                # The device to place all torch tensors.
        seed = 42,                                      # The random seed.
        # dtype = torch.bfloat16,                          # The torch data type of non-integer tensors.
        dtype = DTYPE,                          # The torch data type of non-integer tensors.

        exp_name = f"test-L{layer}{hook_suffix_abbr}-{COEF}",
        exp_series = "default",
        exp_result_dir = EXPORT_DIR,
    ))
    return cfg