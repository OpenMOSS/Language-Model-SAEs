"""Hooked Transformer Gated MLP Component.

This module contains all the component :class:`GatedMLP`.
"""
from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float
from transformers.utils import is_bitsandbytes_available

from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.addmm import batch_addmm
from transformer_lens.factories.activation_function_factory import (
    ActivationFunctionFactory,
)


if is_bitsandbytes_available():
    pass


class GatedMLP(nn.Module):
    """Gated MLP

    This MLP matches the implementation for Mixtral on HuggingFace. It is meant to stay within our
    MoE, since the format of this MLP is different from the standard MLPs throughout
    TransformerLens.

    It may be possible to rework this to follow the same interface as other MLPs, but for the
    time being it is being left as is to ensure accuracy.
    """

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.d_mlp = self.cfg.d_mlp

        if self.d_mlp is None:
            raise ValueError("d_mlp must be set to use an MLP")

        self.W_in = nn.Linear(self.cfg.d_model, self.d_mlp, bias=False)
        self.W_out = nn.Linear(self.d_mlp, self.cfg.d_model, bias=False)
        self.W_gate = nn.Linear(self.cfg.d_model, self.d_mlp, bias=False)

        # hook on gate output but before act_fn
        self.hook_gate = HookPoint()  # [batch, pos, d_mlp]
        # hook on the linear component of the input
        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        # hook on act_fn(gate_output) * W_in(x) + b_in
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        self.act_fn = ActivationFunctionFactory.pick_activation_function(self.cfg)

    def forward(self, x: Float[torch.Tensor, "pos d_model"]) -> Float[torch.Tensor, "pos d_model"]:
        gated_x = self.hook_gate(self.W_gate(x))
        pre_act = self.hook_pre(self.W_in(x))
        post_act = self.hook_post(self.act_fn(gated_x) * pre_act)
        return self.W_out(post_act)
