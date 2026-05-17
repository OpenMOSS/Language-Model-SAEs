from __future__ import annotations

import importlib.util
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from transformer_lens.hook_points import HookPoint, HookedRootModule
from transformer_lens.pretrained.weight_conversions.evo2 import convert_evo2_weights

from vortex.model.generation import generate as vortex_generate
from vortex.model.model import StripedHyena
from vortex.model.tokenizer import CharLevelTokenizer
from vortex.model.utils import dotdict
import vortex.model.model as vortex_model_module

T = TypeVar("T", bound="HookedEvo2")

DEFAULT_EVO2_7B_PATH = Path(
    "/inspire/hdd/global_user/hezhengfu-240208120186/models/evo2_7b/evo2_7b.pt"
)


def _has_flash_attn_cuda() -> bool:
    return importlib.util.find_spec("flash_attn_2_cuda") is not None


def _evo2_7b_config() -> dotdict:
    return dotdict(
        {
            "model_name": "shc-evo2-7b-8k-2T-v2",
            "vocab_size": 512,
            "hidden_size": 4096,
            "num_filters": 4096,
            "hcl_layer_idxs": [2, 6, 9, 13, 16, 20, 23, 27, 30],
            "hcm_layer_idxs": [1, 5, 8, 12, 15, 19, 22, 26, 29],
            "hcs_layer_idxs": [0, 4, 7, 11, 14, 18, 21, 25, 28],
            "attn_layer_idxs": [3, 10, 17, 24, 31],
            "hcm_filter_length": 128,
            "hcl_filter_groups": 4096,
            "hcm_filter_groups": 256,
            "hcs_filter_groups": 256,
            "hcs_filter_length": 7,
            "num_layers": 32,
            "short_filter_length": 3,
            "num_attention_heads": 32,
            "short_filter_bias": False,
            "mlp_init_method": torch.nn.init.zeros_,
            "mlp_output_init_method": torch.nn.init.zeros_,
            "eps": 1e-6,
            "state_size": 16,
            "rotary_emb_base": 10000,
            "rotary_emb_scaling_factor": 128,
            "use_interpolated_rotary_pos_emb": True,
            "make_vocab_size_divisible_by": 8,
            "inner_size_multiple_of": 16,
            "inner_mlp_size": 11264,
            "log_intermediate_values": False,
            "proj_groups": 1,
            "hyena_filter_groups": 1,
            "column_split_hyena": False,
            "column_split": True,
            "interleave": True,
            "evo2_style_activations": True,
            "model_parallel_size": 1,
            "pipe_parallel_size": 1,
            "tie_embeddings": True,
            "mha_out_proj_bias": True,
            "hyena_out_proj_bias": True,
            "hyena_flip_x1x2": False,
            "qkv_proj_bias": False,
            "use_fp8_input_projections": False,
            "max_seqlen": 1048576,
            "max_batch_size": 1,
            "final_norm": True,
            "use_flash_attn": False,
            "use_flash_rmsnorm": False,
            "use_flash_depthwise": False,
            "use_flashfft": False,
            "use_laughing_hyena": False,
            "inference_mode": True,
            "tokenizer_type": "CharLevelTokenizer",
            "prefill_style": "fft",
            "mlp_activation": "gelu",
            "print_activations": False,
        }
    )


@contextmanager
def _suppress_evo2_init_logs():
    root_logger = logging.getLogger()
    striped_logger = logging.getLogger("StripedHyena")
    previous_root_level = root_logger.level
    previous_striped_level = striped_logger.level
    previous_tqdm = vortex_model_module.tqdm

    def _no_tqdm(iterable, *args, **kwargs):
        return iterable

    root_logger.setLevel(logging.WARNING)
    striped_logger.setLevel(logging.WARNING)
    vortex_model_module.tqdm = _no_tqdm
    try:
        yield
    finally:
        vortex_model_module.tqdm = previous_tqdm
        root_logger.setLevel(previous_root_level)
        striped_logger.setLevel(previous_striped_level)


class HookedEvo2(HookedRootModule):
    """Hook-enabled wrapper around the Evo 2 / StripedHyena2 architecture."""

    def __init__(
        self,
        model_name: str = "arcinstitute/evo2_7b",
        local_path: str | Path | None = None,
        dtype: torch.dtype = torch.float32,
        move_to_device: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.dtype = dtype
        self.cfg = _evo2_7b_config()
        self.cfg.dtype = dtype
        self.tokenizer = CharLevelTokenizer(self.cfg.vocab_size)
        self.cfg.tokenizer_prepends_bos = False

        if not self.cfg.get("use_flash_attn", False) or not _has_flash_attn_cuda():
            self.cfg.use_flash_attn = False
        if self.cfg.get("use_fp8_input_projections", False):
            try:
                import transformer_engine  # noqa: F401
            except ImportError:
                # 7B checkpoints can run without TE, just slower.
                self.cfg.use_fp8_input_projections = False

        with _suppress_evo2_init_logs():
            self.model = StripedHyena(self.cfg)

        self.hook_embed = HookPoint()
        self.hook_resid_pre = nn.ModuleList([HookPoint() for _ in range(self.cfg.num_layers)])
        self.hook_resid_post = nn.ModuleList([HookPoint() for _ in range(self.cfg.num_layers)])
        self.hook_logits = HookPoint()

        self.setup()

        if local_path is not None:
            with _suppress_evo2_init_logs():
                self.load_pretrained_weights(local_path)

        self.model.to(dtype=self.dtype)

        # StripedHyena already handles its own device placement across layers.
        _ = move_to_device

    def load_pretrained_weights(self, checkpoint_path: str | Path | Any):
        state_dict = convert_evo2_weights(checkpoint_path, None)
        self.model.custom_load_state_dict(state_dict, strict=True)
        self.model.to(dtype=self.dtype)

    def to_tokens(self, input: Union[str, list[str], torch.Tensor], move_to_device: bool = True):
        if isinstance(input, torch.Tensor):
            tokens = input
        elif isinstance(input, str):
            tokens = torch.tensor(self.tokenizer.tokenize(input), dtype=torch.long).unsqueeze(0)
        elif isinstance(input, list):
            token_lists = [self.tokenizer.tokenize(text) for text in input]
            max_len = max((len(tokens) for tokens in token_lists), default=0)
            padded = [tokens + [self.tokenizer.eod] * (max_len - len(tokens)) for tokens in token_lists]
            tokens = torch.tensor(padded, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported input type for to_tokens: {type(input)!r}")

        if move_to_device:
            tokens = tokens.to(self.model.block_idx_to_device[0])
        return tokens

    def _run_blocks(self, x, padding_mask=None, inference_params_dict=None):
        if type(padding_mask) == torch.Tensor:
            x = x * padding_mask[..., None]

        for block_idx, block in enumerate(self.model.blocks):
            block_dtype = next(block.parameters()).dtype
            x = x.to(dtype=block_dtype)
            if inference_params_dict is None:
                inference_params = None
            else:
                inference_params = inference_params_dict[self.model.block_idx_to_name(block_idx)]

            x = self.hook_resid_pre[block_idx](x)
            x = self.model.cross_device_transfer(x, block_idx)
            x, _ = block(x, inference_params=inference_params, padding_mask=padding_mask)
            x = self.hook_resid_post[block_idx](x)

        return x, inference_params_dict

    def forward(self, x, inference_params_dict=None, padding_mask=None):
        if isinstance(x, (str, list)):
            x = self.to_tokens(x)

        x = self.hook_embed(self.model.embedding_layer(x))
        x = x.to(dtype=next(self.model.parameters()).dtype)

        if inference_params_dict is not None:
            x, inference_params_dict = self._run_blocks(
                x,
                padding_mask=padding_mask,
                inference_params_dict=inference_params_dict,
            )
        else:
            x, inference_params_dict = self._run_blocks(
                x,
                padding_mask=padding_mask,
                inference_params_dict=None,
            )

        x = x.to(self.model.block_idx_to_device[0])
        x = x.to(dtype=next(self.model.norm.parameters()).dtype)
        x = self.model.norm(x)
        x = self.model.unembed(x)
        x = self.hook_logits(x)
        return x, inference_params_dict

    def __call__(self, x, inference_params_dict=None, padding_mask=None):
        return self.forward(x, inference_params_dict=inference_params_dict, padding_mask=padding_mask)

    def generate(
        self,
        prompt_seqs,
        n_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        batched: bool = True,
        cached_generation: bool = True,
        verbose: int = 1,
        force_prompt_threshold: int | None = None,
    ):
        with torch.no_grad():
            return vortex_generate(
                prompt_seqs=prompt_seqs,
                model=self.model,
                tokenizer=self.tokenizer,
                n_tokens=n_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                batched=batched,
                cached_generation=cached_generation,
                verbose=verbose,
                force_prompt_threshold=force_prompt_threshold,
            )

    @classmethod
    def from_pretrained(
        cls: Type[T],
        model_name: str,
        local_path: str | Path | None = None,
        dtype: torch.dtype = torch.float32,
        move_to_device: bool = True,
    ) -> T:
        if local_path is None:
            if model_name == "arcinstitute/evo2-7b" or model_name == "arcinstitute/evo2_7b":
                local_path = DEFAULT_EVO2_7B_PATH
            elif Path(model_name).exists():
                local_path = model_name
            else:
                local_path = hf_hub_download(
                    repo_id="arcinstitute/evo2_7b",
                    filename="evo2_7b.pt",
                    token=None,
                )
            logging.info("Loading Evo2 weights from %s", local_path)

        return cls(
            model_name=model_name,
            local_path=local_path,
            dtype=dtype,
            move_to_device=move_to_device,
        )
