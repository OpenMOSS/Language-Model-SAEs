import os

import torch
from transformer_lens import HookedTransformer

from datasets import Dataset

from core.sae import SparseAutoEncoder
from core.config import FeaturesDecoderConfig

from core.utils.misc import check_file_path_unused

import plotly.graph_objects as go
import numpy as np

@torch.no_grad()
def features_to_logits(sae: SparseAutoEncoder, model: HookedTransformer, cfg: FeaturesDecoderConfig):

    num_ones = int(torch.sum(sae.feature_act_mask).item())

    feature_acts = torch.zeros(num_ones, 24576).to(cfg.device)

    index = 0
    for i in range(len(sae.feature_act_mask)):
        if sae.feature_act_mask[i] == 1:
            feature_acts[index, i] = 1
            index += 1

    feature_acts = torch.unsqueeze(feature_acts, dim=1)
    
    # print(feature_acts.shape)
    residual = sae.features_decoder(feature_acts)
    # print(residual.shape)
    
    if model.cfg.normalization_type is not None:
        residual = model.ln_final(residual)  # [batch, pos, d_model]
    logits = model.unembed(residual)  # [batch, pos, d_vocab]

    # print(logits.shape)
    active_indices = [i for i, val in enumerate(sae.feature_act_mask) if val == 1]
    # print(len(active_indices))
    result_dict = {str(feature_index): logits[idx][0] for idx, feature_index in enumerate(active_indices)}
    # print(result_dict)
    return result_dict