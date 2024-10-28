import torch

from lm_saes.config import SAEConfig
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.convert_pre_enc_bias import merge_pre_enc_bias_to_enc_bias

cfg = SAEConfig(
    d_model=512,
    expansion_factor=4,
    apply_decoder_bias_to_pre_encoder=True,
)

sae = SparseAutoEncoder(cfg)
sample = torch.randn(4, cfg.d_model)

assert (sae(sample) == merge_pre_enc_bias_to_enc_bias(sae)(sample)).all()
