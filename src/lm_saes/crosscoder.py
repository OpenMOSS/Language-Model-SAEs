from lm_saes.abstract_sae import AbstractSparseAutoEncoder

from .config import BaseSAEConfig


class CrossCoder(AbstractSparseAutoEncoder):
    """Sparse AutoEncoder model.

    An autoencoder model that learns to compress the input activation tensor into a high-dimensional but sparse feature activation tensor.

    Can also act as a transcoder model, which learns to compress the input activation tensor into a feature activation tensor, and then reconstruct a label activation tensor from the feature activation tensor.
    """

    def __init__(self, cfg: BaseSAEConfig):
        super(CrossCoder, self).__init__(cfg)
