# Copyright (c) 2024, Michael Poli.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable

from .utils import grab_first_if_tuple

try:
    from transformer_engine.pytorch import Linear as _TELinearBase
    from transformer_engine.common.recipe import Format, DelayedScaling
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False


def set_format_recipe():
    if not HAS_TE:
        raise RuntimeError(
            "Transformer Engine is required for FP8 recipes but is not installed. "
            "Install it with: pip install transformer_engine>=2.0.0"
        )
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    return fp8_format, fp8_recipe


if HAS_TE:

    class TELinear(_TELinearBase):
        """Transformer-Engine Linear with optional FP8.  Used for larger (40b+) models."""

        def __init__(
            self,
            input_size: int,
            output_size: int,
            init_method: Callable = None,
            bias: bool = True,
            skip_bias_add: bool = False,
            use_fp8: bool = False,
            **kwargs,
        ):
            params_dtype = torch.bfloat16
            self.te_return_bias = skip_bias_add and bias
            self.use_fp8_input_projections = use_fp8
            if use_fp8:
                self.fp8_format, self.fp8_recipe = set_format_recipe()
            super().__init__(
                in_features=input_size,
                out_features=output_size,
                sequence_parallel=False,
                fuse_wgrad_accumulation=False,
                tp_group=None,
                tp_size=1,
                init_method=init_method,
                params_dtype=params_dtype,
                parallel_mode=None,
                bias=bias,
                return_bias=self.te_return_bias,
                **kwargs,
            )

        def forward(self, x):
            if self.use_fp8_input_projections:
                with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                    out = super().forward(x)
            else:
                out = super().forward(x)
            if self.te_return_bias:
                return out
            return out, None

else:

    class TELinear(nn.Module):
        """Pure-PyTorch fallback for TELinear (no Transformer Engine).

        Matches TE's state-dict key names (weight / bias) and returns
        (output, bias_or_None) so callers written for TELinear still work.
        Only needed for 40b+ models that use FP8 projections.
        """

        def __init__(
            self,
            input_size: int,
            output_size: int,
            init_method: Callable = None,
            bias: bool = True,
            skip_bias_add: bool = False,
            use_fp8: bool = False,
            **kwargs,
        ):
            super().__init__()
            if use_fp8:
                raise RuntimeError(
                    "FP8 requires Transformer Engine, which is not installed. "
                    "Install it with: pip install transformer_engine>=2.0.0"
                )
            self.te_return_bias = skip_bias_add and bias
            self.use_fp8_input_projections = False
            self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=torch.bfloat16))
            if bias:
                self.bias = nn.Parameter(torch.zeros(output_size, dtype=torch.bfloat16))
            else:
                self.register_parameter("bias", None)
            if init_method is not None:
                init_method(self.weight)
            else:
                nn.init.xavier_uniform_(self.weight)

        def forward(self, x):
            out = F.linear(x.to(self.weight.dtype), self.weight, self.bias)
            if self.te_return_bias:
                return out, self.bias
            return out, None


class RMSNorm(torch.nn.Module):
    def __init__(self, config):
        super(RMSNorm, self).__init__()
        self.eps, self.hidden_size = config.eps, config.hidden_size
        self.scale = torch.nn.Parameter(torch.ones(self.hidden_size, dtype=config.params_dtype))
        self.register_parameter("scale", self.scale)
        self.use_flash_rmsnorm = config.get("use_flash_rmsnorm", False)

        if self.use_flash_rmsnorm:
            from flash_attn.ops.rms_norm import rms_norm as rmsnorm_func
            self.rmsnorm_func = rmsnorm_func

    def forward(self, x):
        if self.use_flash_rmsnorm:
            return self.rmsnorm_func(x, self.scale, self.eps)
        y = x / (x.norm(2, dim=-1, keepdim=True) * self.hidden_size ** (-1.0 / 2) + self.eps)
        return self.scale * y


class ParallelGatedMLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        multiple_of = config.get("inner_size_multiple_of", 64)
        self.act_type = config.get("mlp_activation", "gelu")
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError

        if self.layer_idx > 0 and config.get("evo2_style_activations", False):
            self.act = nn.Identity()

        self.multiple_of = multiple_of * config.model_parallel_size

        inner_size = int(2 * config.hidden_size * 4 / 3)
        inner_size = self.multiple_of * ((inner_size + self.multiple_of - 1) // self.multiple_of)
        inner_size = config.get("inner_mlp_size", inner_size)

        self.l1 = nn.Linear(config.hidden_size, inner_size, bias=False)
        self.l2 = nn.Linear(config.hidden_size, inner_size, bias=False)
        self.l3 = nn.Linear(inner_size, config.hidden_size, bias=False)

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        z1, z2 = grab_first_if_tuple(z1), grab_first_if_tuple(z2)
        y = self.l3(self.act(z1) * z2)
        return grab_first_if_tuple(y)


class VocabParallelEmbedding(nn.Embedding):
    "Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/embedding.py"

    def __init__(self, config):
        vocab_size, process_group, padding_idx = (
            config.vocab_size,
            config.get("process_group", None),
            config.get("padding_idx", None),
        )
        self.process_group = process_group
        if process_group is not None:
            world_size = torch.distributed.get_world_size(process_group)
            if vocab_size % world_size != 0:
                raise ValueError(f"vocab_size ({vocab_size}) must be divisible by world_size ({world_size})")
            if world_size > 1 and padding_idx is not None:
                raise RuntimeError("ParallelEmbedding does not support padding_idx")
        else:
            world_size = 1
        super().__init__(vocab_size // world_size, embedding_dim=config.hidden_size, padding_idx=padding_idx)

    def forward(self, input: Tensor) -> Tensor:
        if self.process_group is None:
            return super().forward(input)
        rank = torch.distributed.get_rank(self.process_group)
        vocab_size = self.num_embeddings
        vocab_start_index, vocab_end_index = rank * vocab_size, (rank + 1) * vocab_size
        input_ids_mask = (input < vocab_start_index) | (input >= vocab_end_index)
        input = input - vocab_start_index
        input[input_ids_mask] = 0
        embeddings = self.forward(input)
        embeddings[input_ids_mask] = 0.0
        torch.distributed.all_reduce(embeddings, group=self.process_group)
        return embeddings

    def unembed(self, u: Tensor) -> Tensor:
        if self.process_group is None:
            return u @ self.weight.T
        raise NotImplementedError


class VocabParallelUnembedding(VocabParallelEmbedding):
    def forward(self, input: Tensor) -> Tensor:
        return self.unembed(input)
