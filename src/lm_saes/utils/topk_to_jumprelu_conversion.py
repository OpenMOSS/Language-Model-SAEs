import math
import torch
from lm_saes import AbstractSparseAutoEncoder
from lm_saes.activation_functions import JumpReLU
from typing import Iterable, Optional
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from lm_saes.utils.logging import get_distributed_logger
from lm_saes.utils.math import topk
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
    _, threshold = topk_func(
        hidden_pre,
        k=sae.cfg.top_k * hidden_pre.size(0),
        dim=(-3, -2, -1),
        return_threshold=True,
        **kwargs,
    )

    print(threshold)

    sae.cfg.act_fn = "jumprelu"
    sae.activation_function = sae.activation_function_factory(device_mesh)
    assert isinstance(sae.activation_function, JumpReLU)
    sae.activation_function.log_jumprelu_threshold.data.fill_(math.log(threshold.item()))

    return sae
