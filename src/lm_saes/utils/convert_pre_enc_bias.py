import torch

from ..sae import SparseAutoEncoder


@torch.no_grad()
def merge_pre_enc_bias_to_enc_bias(sae: SparseAutoEncoder):
    assert sae.cfg.apply_decoder_bias_to_pre_encoder

    sae.cfg.apply_decoder_bias_to_pre_encoder = False
    sae.b_E.copy_(sae.b_E - sae.b_D @ sae.W_E)

    return sae
