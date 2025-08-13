from torch._tensor import Tensor


import math
import torch
from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.activation_functions import JumpReLU
from torch.distributed.tensor import DTensor
from typing import Any, Iterable, Optional
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from lm_saes.utils.logging import get_distributed_logger
from lm_saes.utils.math import topk
from torch.distributed.tensor.placement_types import Shard
from lm_saes.utils.distributed import distributed_topk

logger = get_distributed_logger("utils.topk_to_jumprelu_conversion")

@torch.no_grad()
def topk_to_jumprelu_conversion(
    sae: AbstractSparseAutoEncoder,
    activations_stream: Iterable[dict[str, Tensor]],
    device_mesh: Optional[DeviceMesh] = None,
) -> AbstractSparseAutoEncoder:
    """Convert a CLT model from topk to jumprelu.

    Args:
        sae: The CLT model to convert
        activations_stream: The activations stream to convert
        device_mesh: The device mesh to use
    """
    assert "topk" in sae.cfg.act_fn, "CLT model must use topk activation function"

    activation_stream = iter(activations_stream)
    activation_batch = next(activation_stream)
    activation_batch = sae.normalize_activations(activation_batch)
    x, kwargs = sae.prepare_input(activation_batch)
    _, hidden_pre = sae.encode(x, **kwargs, return_hidden_pre=True)
    hidden_pre = torch.clamp(hidden_pre, min=0.0)

    topk_func = topk if device_mesh is None else distributed_topk
    kwargs = {} if device_mesh is None else {
        "device_mesh": device_mesh,
        "mesh_dim_name": "model",
    }

    topk_acts, threshold = topk_func(
        hidden_pre,
        k=sae.cfg.top_k * hidden_pre.size(0),
        dim=(-3, -2, -1),
        return_threshold=True,
        **kwargs,
    )
    
    origin_rec = sae(x)

    threshold = threshold.squeeze().item()
    logger.info(f"Computed threshold: {threshold}")

    sae.cfg.act_fn = "jumprelu"
    sae.activation_function = sae.activation_function_factory(device_mesh)
    assert isinstance(sae.activation_function, JumpReLU)

    if sae.cfg.sparsity_include_decoder_norm:
        decoder_norm_per_feature = sae.decoder_norm_per_feature()
        for layer in range(sae.cfg.n_layers):
            sae.activation_function.log_jumprelu_threshold.data[layer] = (
                threshold / decoder_norm_per_feature[layer]
            ).log()
        sae.cfg.sparsity_include_decoder_norm = False
        logger.info("Also converting sparsity_include_decoder_norm to False so we do not need decoders to get encode results.")
    else:
        sae.activation_function.log_jumprelu_threshold.data.fill_(math.log(threshold.item()))

    converted_rec = sae(x)

    print(f'{origin_rec=}')
    print(f'{converted_rec=}')
    print(f'{origin_rec - converted_rec=}')

    validation_batch = next(activation_stream)
    validation_batch = sae.normalize_activations(validation_batch)
    x, kwargs = sae.prepare_input(validation_batch)
    feature_acts = sae.encode(x, **kwargs)
    
    l0 = feature_acts.gt(0).float().sum() / feature_acts.size(0)
    logger.info(f"converted sae got L0 of {l0.item()}, should be {sae.cfg.top_k}")

    return sae
