import logging
import random

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import HookedTransformer

from lm_saes.config import (
    LanguageModelConfig,
)

from datasets import load_dataset, load_from_disk
from transformer_lens import HookedTransformer

import pytest

HOOK_SUFFIX={"mlp":"hook_mlp_out", "self_attn":"hook_attn_out", "resid":"hook_resid_post"}
model_name = 'meta-llama/Meta-Llama-3-8B'
model_path = 'path/to/model'
d_model = 4096

@pytest.fixture
def dataset():
    return load_dataset("Skylion007/openwebtext", split="train")

@pytest.fixture
def dataloader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=32)

@pytest.fixture
def model():
    return HookedTransformer.from_pretrained('gpt2')

@pytest.fixture
def prepare_config():
    cfg = LanguageModelConfig.from_flattened(dict(
        # LanguageModelConfig
        model_name = model_name,                            # The model name or path for the pre-trained model.
        model_from_pretrained_path = model_path,
        d_model = d_model,                                  # The hidden size of the model.
        
        # RunnerConfig
        device = "cuda",                                # The device to place all torch tensors.
        seed = 42,                                      # The random seed.
        dtype = torch.bfloat16,                          # The torch data type of non-integer tensors.

        exp_name = f"test",
        exp_series = "default",
        exp_result_dir = "results",
    ))
    return cfg

@pytest.fixture
def prepare_llama3_models():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    hf_no_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                       attn_implementation="eager", 
                                                       torch_dtype=torch.bfloat16)
    hf_no_model.eval()
    hf_fa_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                       attn_implementation="flash_attention_2", 
                                                       torch_dtype=torch.bfloat16)
    hf_fa_model.eval()
    hf_fa_model.to(device)
    return hf_no_model, hf_fa_model

@pytest.fixture
def prepare_models(prepare_config):
    cfg = prepare_config
    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_from_pretrained_path,
        cache_dir=cfg.cache_dir,
        local_files_only=cfg.local_files_only,
        torch_dtype=cfg.dtype,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_from_pretrained_path,
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=True,
    )

    # FlashAttention only allow dtype of bfp16 and fp16
    assert cfg.dtype in [torch.bfloat16, torch.float16] 

    fa_model = HookedTransformer.from_pretrained(
        cfg.model_name,
        use_flash_attn=True,
        device=cfg.device,
        cache_dir=cfg.cache_dir,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        dtype=cfg.dtype,
    )
    fa_model.eval()
    no_model = HookedTransformer.from_pretrained(
        cfg.model_name,
        use_flash_attn=False,
        device=cfg.device,
        cache_dir=cfg.cache_dir,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        dtype=cfg.dtype,
    )
    no_model.eval()
    logging.warning("Model loaded!")
    return fa_model, no_model, cfg

def test_flash_attn_dtype(prepare_models):
    fa_model, no_model, cfg = prepare_models
    # bfloat16 dtype test
    if cfg.dtype == torch.bfloat16:
        for model in [fa_model, no_model]:
            for name, obj in vars(HookedTransformer).items():
                if isinstance(obj, property):
                    try:
                        param = model.__getattribute__(name)
                        assert (param.dtype == torch.bfloat16)
                    except:
                        logging.warning(f"Does not have attribute {name}")


def test_flash_attn_correctness(prepare_models, prepare_llama3_models, dataset):
    """
    This test function is only for Llama3-8B
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    test_input_list = []
    for _ in range(10):
        text = ''.join(next(iter(dataloader))['text'])
        idx = random.randrange(0, len(text)-64)
        test_input_list.append(text[idx:idx+64])
    fa_model, no_model, cfg = prepare_models
    hf_no_model, hf_fa_model = prepare_llama3_models
    hf_models = {'flash_attn':hf_fa_model, 'no_flash_attn':hf_no_model}

    module_names = [f"model.layers.{i}" + e for e in [".self_attn", '.mlp', ''] for i in range(no_model.cfg.n_layers)]+['lm_head']
    hf_output_cache = {'flash_attn':{}, 'no_flash_attn':{}}
    hf_handles = []

    def get_hook(model_type, module_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hf_output_cache[hook.model_type][hook.module_name] = output[0]
            else:
                hf_output_cache[hook.model_type][hook.module_name] = output
        hook.model_type = model_type
        hook.module_name = module_name
        return hook
    for model_type in ['flash_attn', 'no_flash_attn']:
        for (module_name, module) in hf_models[model_type].named_modules():
            if module_name in module_names:
                hf_handles.append(module.register_forward_hook(hook=get_hook(model_type, module_name)))
    
    for test_input in test_input_list:
        tokens = no_model.to_tokens(test_input, prepend_bos=not False).to(cfg.device)
        fa_logits, fa_cache = fa_model.run_with_cache(tokens, return_type="logits")
        no_logits, no_cache = no_model.run_with_cache(tokens, return_type="logits")
        _ = hf_models['flash_attn'](tokens, use_cache=False)
        _ = hf_models['no_flash_attn'](tokens.cpu(), use_cache=False)
        for layer in range(no_model.cfg.n_layers):
            for abbr, hook_suffix_abbr in HOOK_SUFFIX.items():
                fa_value = fa_cache[f'blocks.{layer}.{hook_suffix_abbr}']
                no_value = no_cache[f'blocks.{layer}.{hook_suffix_abbr}']

                hf_fa_value = hf_output_cache['flash_attn'][f'model.layers.{layer}' if abbr == 'resid' 
                                                            else f'model.layers.{layer}.{abbr}']
                hf_no_value = hf_output_cache['no_flash_attn'][f'model.layers.{layer}' if abbr == 'resid' 
                                                            else f'model.layers.{layer}.{abbr}']

                delta_max_fa_no = torch.abs(fa_value.cpu() - no_value.cpu()).max().item()
                delta_max_hf_fa_no = torch.abs(hf_fa_value.cpu() - hf_no_value).max().item()
                logging.warning(f"L{layer}{abbr}\ttl:{delta_max_fa_no}\thf:{delta_max_hf_fa_no}")
                assert (delta_max_fa_no < delta_max_hf_fa_no * 5)
        d_logits_fa_no = torch.abs(fa_logits.cpu() - no_logits.cpu()).max().item()
        d_logits_hf_fa_no = torch.abs(hf_output_cache['flash_attn']['lm_head'].cpu() - hf_output_cache['no_flash_attn']['lm_head']).max().item()

        logging.warning(f"Logits\ttl:{d_logits_fa_no}\thf:{d_logits_hf_fa_no}")
        assert (d_logits_fa_no < d_logits_hf_fa_no * 5)
        for handle in hf_handles:
            handle.remove()