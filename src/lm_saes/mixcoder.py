from lm_saes.config import MixCoderConfig
from lm_saes.sae import SparseAutoEncoder


class MixCoder(SparseAutoEncoder):
    def __init__(self, cfg: MixCoderConfig):
        super().__init__(cfg)
        pass
