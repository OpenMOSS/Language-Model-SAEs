import os
import sys

sys.path.insert(0, os.getcwd())
import argparse

import torch

from lm_saes.config import LanguageModelSAETrainingConfig
from lm_saes.runner import language_model_sae_runner

parser = argparse.ArgumentParser(description="Process hyparameters")
parser.add_argument("-l", "--layer", nargs="*", required=False, type=int, help="Layer number")
parser.add_argument("-b", "--l1_coeff", type=float, required=False, default=2e-4, help="L1 Coefficient, default 2e-4")
parser.add_argument("-s", "--batch_size", type=int, required=False, default=2048, help="Batchsize, default 2048")
parser.add_argument("-r", "--lr", type=float, required=False, default=6e-4, help="Learning rate, default 8e-5")
parser.add_argument("-d", "--expdir", type=str, required=False, default="./results", help="Export directory")
parser.add_argument("--ddp_size", type=int, required=False, default=1, help="Distributed data parallel size, default 1")
parser.add_argument("--tp_size", type=int, required=False, default=1, help="Tensor parallel size, default 1")
parser.add_argument("--dtype", type=str, required=False, default="bfloat16", help="Dtype, default bfloat16")
parser.add_argument("--fix_norm", type=float, required=False, default=None, help="Fixed decoder norm, default None")
parser.add_argument("--exp_factor", type=int, required=False, default=32, help="Expansion factor, default 32")
parser.add_argument("--name", type=str, required=False, default="", help="Experiment name, default ")
parser.add_argument("--clip_grad_norm", type=float, required=False, default=0.0, help="Clip grad_norm, default 0.")
parser.add_argument("--tc_in_abbr", type=str, required=False, default="R", help="Input layer type, default None")
parser.add_argument("--tc_out_abbr", type=str, required=False, default="R", help="Output layer type, default None")
parser.add_argument("--buffer_size", type=int, required=False, default=500_000, help="Buffer size, default 500_000")
parser.add_argument("--k", type=int, required=False, default=50, help="Top-K")
parser.add_argument("--store_batch_size", type=int, required=False, default=32)
parser.add_argument("--log_to_wandb", type=bool, required=False, default=False)
parser.add_argument("--total_training_tokens", type=int, required=False, default=1_000_000)
args = parser.parse_args()

if args.tp_size > 1 or args.ddp_size > 1:
    print("Distributed training")
    import os

    import torch.distributed as dist

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

HOOK_SUFFIX = {
    "M": "hook_mlp_out",
    "A": "hook_attn_out",
    "R": "hook_resid_post",
    "TC": "ln2.hook_normalized",
    "Emb": "hook_resid_pre",
}
LR = args.lr
print(f"Learning rate: {LR}")
LAYER = args.layer
TC_IN_ABBR = args.tc_in_abbr
TC_OUT_ABBR = args.tc_out_abbr
L1_COEFF = args.l1_coeff
TRAIN_BATCH_SIZE = args.batch_size
EXPORT_DIR = args.expdir
DTYPE = torch.bfloat16
FIX_DECODER_NORM = args.fix_norm
EXP_FACTOR = args.exp_factor
COEF = f"{EXP_FACTOR}x-{TRAIN_BATCH_SIZE}bs-lr-{LR}-l1-{L1_COEFF}-fix_norm-{FIX_DECODER_NORM}-clip-{args.clip_grad_norm}-{args.name}"
# for layer, hook_suffix_abbr in LAYER_RANGE:
# for layer, hook_in_suffix_abbr, hook_out_suffix_abbr in zip(LAYER, TC_IN_ABBR, TC_OUT_ABBR):
layer = LAYER[0]
hook_in_suffix_abbr = TC_IN_ABBR
hook_out_suffix_abbr = TC_OUT_ABBR
if hook_in_suffix_abbr == "Emb":
    assert layer == 0
if args.tc_in_abbr is not None:
    hook_in_suffix_abbr = args.tc_in_abbr
    hook_out_suffix_abbr = args.tc_out_abbr

EXPORT_PATH = f"./Llama3_1Base-LX{hook_in_suffix_abbr}-{EXP_FACTOR}x-topk/Llama3_1Base-L{layer}{hook_in_suffix_abbr}-{args.exp_factor}x-lr{LR}-l1{L1_COEFF}"

cfg = LanguageModelSAETrainingConfig.from_flattened(
    dict(
        # LanguageModelConfig
        model_name="meta-llama/Llama-3.1-8B",  # The model name or path for the pre-trained model.
        model_from_pretrained_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/hezhengfu-240208120186/models/Llama-3.1-8B",
        d_model=4096,  # The hidden size of the model.
        # TextDatasetConfig
        dataset_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/hezhengfu-240208120186/data/SlimPajama-3B",
        # The corpus name or path. Each of a data record should contain (and may only contain) a "text" field.
        is_dataset_tokenized=False,  # Whether the dataset is tokenized.
        is_dataset_on_disk=True,
        # Whether the dataset is on disk. If not on disk, `datasets.load_dataset`` will be used to load the dataset, and the train split will be used for training.
        concat_tokens=False,
        # Whether to concatenate tokens into a single sequence. If False, only data record with length of non-padding tokens larger than `context_size` will be used.
        context_size=1024,  # The sequence length of the text dataset.
        store_batch_size=args.store_batch_size,  # The batch size for loading the corpus.
        # ActivationStoreConfig
        hook_points=[
            f"blocks.{layer}.{HOOK_SUFFIX[hook_in_suffix_abbr]}",
            f"blocks.{layer}.{HOOK_SUFFIX[hook_out_suffix_abbr]}",
        ],
        tp_size=args.tp_size,
        # Hook points to store activations from, i.e. the layer output of which is used for training/evaluating the dictionary. Will run until the last hook point in the list, so make sure to order them correctly.
        use_cached_activations=False,
        # Whether to use cached activations. Caching activation is now not recommended, as it may consume extremely large disk space. (May be tens of TBs for corpus like `openwebtext`)
        n_tokens_in_buffer=args.buffer_size,
        # The number of tokens to store in the activation buffer. The buffer is used to shuffle the activations before training the dictionary.
        # SAEConfig
        hook_point_in=f"blocks.{layer}.{HOOK_SUFFIX[hook_in_suffix_abbr]}",
        hook_point_out=f"blocks.{layer}.{HOOK_SUFFIX[hook_out_suffix_abbr]}",
        expansion_factor=EXP_FACTOR,  # The expansion factor of the dictionary. d_sae = expansion_factor * d_model.
        norm_activation="dataset-wise",
        # The normalization method for the activations. Can be "token-wise", "batch-wise" or "none".
        decoder_exactly_unit_norm=False,
        # Whether to enforce the decoder to have exactly unit norm. If False, the decoder will have less than or equal to unit norm.
        bias_init_method="all_zero",
        use_glu_encoder=False,  # Whether to use the Gated Linear Unit (GLU) for the encoder.
        lp=1,  # The p-norm to use for the L1 regularization.
        use_ghost_grads=False,  # Whether to use the ghost gradients for saving dead features.
        use_decoder_bias=True,
        sparsity_include_decoder_norm=True,
        remove_gradient_parallel_to_decoder_directions=False,
        apply_decoder_bias_to_pre_encoder=False,
        init_encoder_with_decoder_transpose=hook_in_suffix_abbr != "TC",
        init_decoder_norm=args.fix_norm,
        init_encoder_norm=None if hook_in_suffix_abbr != "TC" else args.fix_norm,
        act_fn="topk",
        top_k=args.k,
        k_warmup_steps=0.1,  # if hook_in_suffix_abbr != "TC" else 0.,
        use_batch_norm_mse=True,
        # LanguageModelSAETrainingConfig
        total_training_tokens=args.total_training_tokens,  # The total number of tokens to train the dictionary.
        lr=LR,  # The learning rate for the dictionary training.
        betas=(0.9, 0.9999),  # The betas for the Adam optimizer.
        lr_scheduler_name="constantwithwarmup",
        # The learning rate scheduler name. Can be "constant", "constantwithwarmup", "linearwarmupdecay", "cosineannealing", "cosineannealingwarmup" or "exponentialwarmup".
        lr_warm_up_steps=5000,  # The number of warm-up steps for the learning rate.
        lr_cool_down_steps=0.2,  # The number of cool-down steps for the learning rate.
        lr_end_ratio=0.01,  # The ratio of the end learning rate to the initial learning rate.
        train_batch_size=TRAIN_BATCH_SIZE,
        # The batch size for training the dictionary, i.e. the number of token activations in a batch.
        feature_sampling_window=1000,  # The window size for sampling the feature activations.
        dead_feature_window=5000,  # The window size for detecting the dead features.
        dead_feature_threshold=1e-6,  # The threshold for detecting the dead features.
        eval_frequency=10000000000,  # The step frequency for evaluating the dictionary.
        log_frequency=1000,  # The step frequency for logging the training information (to wandb).
        n_checkpoints=0,  # The number of checkpoints to save during the training.
        clip_grad_norm=args.clip_grad_norm,
        # WandbConfig
        log_to_wandb=args.log_to_wandb,  # Whether to log the training information to wandb.
        wandb_project="LlamaScope",  # The wandb project name.
        # RunnerConfig
        device="cuda",  # The device to place all torch tensors.
        seed=42,  # The random seed.
        # dtype = torch.bfloat16,                          # The torch data type of non-integer tensors.
        dtype=DTYPE,  # The torch data type of non-integer tensors.
        exp_name=f"Llama3_1Base-L{layer}{hook_in_suffix_abbr}-{EXP_FACTOR}x-lr{args.lr}-k{args.k}",
        exp_series="LlamaScope",
        exp_result_path=EXPORT_PATH,
    )
)
sparse_autoencoder = language_model_sae_runner(cfg)
