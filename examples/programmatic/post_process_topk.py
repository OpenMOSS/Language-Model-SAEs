import os
import argparse

from lm_saes import LanguageModelSAERunnerConfig, SAEConfig, post_process_topk_to_jumprelu_runner

parser = argparse.ArgumentParser(description="Process hyparameters")
parser.add_argument("-l", "--layer", nargs="*", required=False, type=int, help="Layer number")
parser.add_argument("-d", "--expdir", type=str, required=False, default="./results", help="Export directory")
parser.add_argument("--dtype", type=str, required=False, default="bfloat16", help="Dtype, default bfloat16")
parser.add_argument("--exp_factor", type=int, required=False, default=32, help="Expansion factor, default 32")
parser.add_argument("--tc_in_abbr", type=str, required=False, default="R", help="Input layer type, default None")
parser.add_argument("--tc_out_abbr", type=str, required=False, default="R", help="Output layer type, default None")
parser.add_argument("--store_batch_size", type=int, required=False, default=8)
args = parser.parse_args()

layer = args.layer[0]

hook_point_in = args.tc_in_abbr
hook_point_out = args.tc_out_abbr
exp_factor = args.exp_factor

HOOK_SUFFIX = {
    "M": "hook_mlp_out",
    "A": "hook_attn_out",
    "R": "hook_resid_post",
    "TC": "ln2.hook_normalized",
    "Emb": "hook_resid_pre",
}


hook_suffix_in = HOOK_SUFFIX[hook_point_in]
hook_suffix_out = HOOK_SUFFIX[hook_point_out]
ckpt_path = f"<base_path>/Llama3_1Base-LX{hook_point_in}-{exp_factor}x"
ckpt_path = os.path.join(ckpt_path, f"Llama3_1Base-L{layer}{hook_point_in}-{exp_factor}x")
sae_config = SAEConfig.from_pretrained(ckpt_path).to_dict()


model_name = "meta-llama/Llama-3.1-8B"
# model_from_pretrained_path = "<local_model_path>"

hook_points = [
    f"blocks.{layer}.{hook_suffix_in}",
    f"blocks.{layer}.{hook_suffix_out}",
]

cfg = LanguageModelSAERunnerConfig.from_flattened(
    dict(
        **sae_config,
        model_name=model_name,
        # model_from_pretrained_path=model_from_pretrained_path,
        # d_model=4096,
        dataset_path="<local_dataset_path>",
        is_dataset_tokenized=False,
        is_dataset_on_disk=True,
        concat_tokens=False,
        context_size=1024,
        store_batch_size=args.store_batch_size,
        hook_points=hook_points,
        use_cached_activations=False,
        hook_points_in=hook_points[0],
        hook_points_out=hook_points[1],
        # norm_activation="token-wise",
        decoder_exactly_unit_norm=False,
        decoder_bias_init_method="geometric_median",
        # use_glu_encoder=False,
        # use_ghost_grads=False,  # Whether to use the ghost gradients for saving dead features.
        # use_decoder_bias=True,
        # sparsity_include_decoder_norm=True,
        remove_gradient_parallel_to_decoder_directions=False,
        # apply_decoder_bias_to_pre_encoder=True,
        # init_encoder_with_decoder_transpose=True,
        # expansion_factor=8,
        # train_batch_size=2048,
        log_to_wandb=False,
        # device="cuda",
        # seed=44,
        # dtype=args.dtype,
        exp_name="eval",
        exp_result_path=ckpt_path,
    )
)

post_process_topk_to_jumprelu_runner(cfg)
