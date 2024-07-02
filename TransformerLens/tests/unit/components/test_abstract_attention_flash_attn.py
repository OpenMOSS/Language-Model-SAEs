from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch

MODEL_NAMES = {
    'gpt2':'gpt2',
    'llama3-base':'meta-llama/Meta-Llama-3-8B',
    'llama3-instruct':'meta-llama/Meta-Llama-3-8B-Instruct',
}
MODEL_PATHS = {
    'gpt2':'path/to/gpt2',
    'llama3':'path/to/llama3-base',
    'llama3-instruct':'path/to/llama3-instruct',
}


def test_hooked_transformer():
    model_name = 'gpt2'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS[model_name],
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
    )

    hf_tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS[model_name],
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=True,
    )
    model = HookedTransformer.from_pretrained(
        MODEL_NAMES[model_name],
        use_flash_attn=False,
        device=device,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        dtype=dtype,
    )

    assert not hasattr(model.blocks[0].attn, 'flash_attn_func'), "AbstractAttention should not have 'flash_attn_func' if set `use_flash_attn=False`"
