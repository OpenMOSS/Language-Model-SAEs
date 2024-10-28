import torch

from lm_saes.sae import SparseAutoEncoder


@torch.no_grad()
def merge_pre_enc_bias_to_enc_bias(sae: SparseAutoEncoder):
    assert sae.cfg.apply_decoder_bias_to_pre_encoder

    sae.cfg.apply_decoder_bias_to_pre_encoder = False
    sae.encoder.bias.data = sae.encoder.bias.data - sae.encoder.weight.data @ sae.decoder.bias.data

    return sae
