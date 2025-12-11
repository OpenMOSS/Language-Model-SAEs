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
        # `sampling_data["context_idx"]` 在当前实现中通常是 1D（top 样本维度），
        # 若将来改成与通用实现一致的 2D（样本 x 特征），这里优先取对应列；
        # 若仍为 1D，则直接返回同一批样本（保持兼容，不再触发越界）。
        context_idx = sampling_data["context_idx"]
        if context_idx.dim() == 2:
            ctx = context_idx[:, i]
        else:  # 1D fallback
            ctx = context_idx

        shard_idx_tensor = sampling_data.get("shard_idx")
        if shard_idx_tensor is not None:
            if shard_idx_tensor.dim() == 2:
                shard = shard_idx_tensor[:, i]
            else:
                shard = shard_idx_tensor
        else:
            shard = torch.zeros_like(ctx, dtype=torch.int64)

        n_shards_tensor = sampling_data.get("n_shards")
        if n_shards_tensor is not None:
            if n_shards_tensor.dim() == 2:
                n_shards = n_shards_tensor[:, i]
            else:
                n_shards = n_shards_tensor
        else:
            n_shards = torch.ones_like(ctx, dtype=torch.int64)

        return {
            "context_idx": ctx.cpu().numpy(),
            "shard_idx": shard.cpu().numpy(),
            "n_shards": n_shards.cpu().numpy(),
        }


# Register the processor for CNN SAE type
register_post_analysis_processor("cnnsae", CNNSAEPostAnalysisProcessor)
