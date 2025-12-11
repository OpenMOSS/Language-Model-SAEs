from typing import Any
from typing import Any, Mapping, Optional, cast
from lm_saes.utils.distributed import DimMap, masked_fill, to_local


import torch
from einops import repeat
from torch.distributed.device_mesh import DeviceMesh
from tqdm import tqdm
from torch.distributed.tensor import DTensor

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.activation.factory import ActivationFactory
from lm_saes.cnnsae import CNNSparseAutoEncoder
from lm_saes.utils.discrete import KeyedDiscreteMapper
from lm_saes.utils.logging import get_distributed_logger

from .generic import GenericPostAnalysisProcessor
from .base import PostAnalysisProcessor, register_post_analysis_processor
from lm_saes.utils.misc import is_primary_rank
from lm_saes.utils.tensor_dict import concat_dict_of_tensor, sort_dict_of_tensor
from einops import rearrange, repeat


logger = get_distributed_logger("lorsa_post_analysis")


class CNNSAEPostAnalysisProcessor(GenericPostAnalysisProcessor):
    
    def _extra_info(self, sampling_data: dict[str, Any], i: int) -> dict[str, Any]:
        """Extra information to add to the feature result."""
        return {
            # 和通用实现保持一致：为“当前 feature”取对应列的样本索引
            # 这样每个 feature 都会拿到自己 top 样本的 context/shard，而不是整组共享
            "context_idx": sampling_data["context_idx"][:, i].cpu().numpy(),
            "shard_idx": sampling_data["shard_idx"][:, i].cpu().numpy()
            if "shard_idx" in sampling_data
            else torch.zeros_like(sampling_data["context_idx"][:, i].cpu(), dtype=torch.int64).numpy(),
            "n_shards": sampling_data["n_shards"][:, i].cpu().numpy()
            if "n_shards" in sampling_data
            else torch.ones_like(sampling_data["context_idx"][:, i].cpu(), dtype=torch.int64).numpy(),
        }


# Register the processor for CNN SAE type
register_post_analysis_processor("cnnsae", CNNSAEPostAnalysisProcessor)
