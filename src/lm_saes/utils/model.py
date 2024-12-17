from transformer_lens import HookedTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    ChameleonForConditionalGeneration,
    PreTrainedModel,
)

from lm_saes.config import LanguageModelConfig


def load_model(cfg: LanguageModelConfig):
    if "chameleon" in cfg.model_name:
        hf_model = ChameleonForConditionalGeneration.from_pretrained(
            (cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path),
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            torch_dtype=cfg.dtype,
        ).to(cfg.device)  # type: ignore
        print(f"Model loaded on device {cfg.device}")
    else:
        hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            (cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path),
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            torch_dtype=cfg.dtype,
        ).to(cfg.device)  # type: ignore
    if "chameleon" in cfg.model_name:
        hf_processor = AutoProcessor.from_pretrained(
            (cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path),
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=True,
            local_files_only=cfg.local_files_only,
        )
        hf_tokenizer = None
    else:
        hf_tokenizer = AutoTokenizer.from_pretrained(
            (cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path),
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=True,
            local_files_only=cfg.local_files_only,
        )
        hf_processor = None

    model = HookedTransformer.from_pretrained_no_processing(
        cfg.model_name,
        use_flash_attn=cfg.use_flash_attn,
        device=cfg.device,
        cache_dir=cfg.cache_dir,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        processor=hf_processor,
        dtype=cfg.dtype,
    )
    model.eval()
    return model
