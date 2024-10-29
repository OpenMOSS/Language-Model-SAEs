from transformer_lens import hook_points
from lm_saes import post_process_topk_to_jumprelu_runner, LanguageModelSAERunnerConfig, SAEConfig
import os
import torch
import jsonlines

layer = 15

hook_point_in = 'R'
hook_point_out = hook_point_in if hook_point_in != 'TC' else 'M'
exp_factor = 8

HOOK_SUFFIX = {"M": "hook_mlp_out", "A": "hook_attn_out", "R": "hook_resid_post", "TC": "ln2.hook_normalized",
               "Emb": "hook_resid_pre"}


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
        model_from_pretrained_path=model_from_pretrained_path,
        # d_model=4096,
        dataset_path="<local_dataset_path>",
        is_dataset_tokenized=False,
        is_dataset_on_disk=True,
        concat_tokens=False,
        context_size=1024,
        store_batch_size=4,
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
        train_batch_size=2048,
        log_to_wandb=False,
        # device="cuda",
        # seed=44,
        # dtype=torch.bfloat16,
        exp_name="eval",
        exp_result_dir=f"./result/{layer}_{hook_point_in}",
    )
)

post_process_topk_to_jumprelu_runner(cfg)