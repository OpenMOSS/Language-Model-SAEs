"""Model definitions for Language Model SAEs.

This package contains the sparse dictionary base class, protocol definitions,
and all concrete model implementations.
"""

from lm_saes.models.protocols import (
    ActiveSubspaceInitializable,
    DatasetNormStandardizable,
    EncoderBiasInitializable,
    EncoderInitializable,
    NormComputing,
    NormConstrainable,
)
from lm_saes.models.sparse_dictionary import (
    SparseDictionary,
    SparseDictionaryConfig,
    register_sae_config,
    register_sae_model,
)

__all__ = [
    "SparseDictionary",
    "SparseDictionaryConfig",
    "register_sae_config",
    "register_sae_model",
    "NormComputing",
    "NormConstrainable",
    "DatasetNormStandardizable",
    "EncoderInitializable",
    "ActiveSubspaceInitializable",
    "EncoderBiasInitializable",
]
