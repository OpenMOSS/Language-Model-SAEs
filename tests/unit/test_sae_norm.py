import torch
import torch.distributed as dist
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from lm_saes.config import SAEConfig
from lm_saes.sae import SparseAutoEncoder

# init distributed environment
dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank())
sae_config = SAEConfig(
    hook_point_in="in",
    hook_point_out="out",
    d_model=2,
    expansion_factor=2,
    device="cuda",
)
device_mesh = DeviceMesh(
    mesh_dim_names=("ddp", "tp"),
    mesh=[1, 2],
    device_type="cuda",
)
# print(sae_config)
sae = SparseAutoEncoder(sae_config)

assert sae.encoder.weight.shape == (4, 2)
assert sae.decoder.weight.shape == (2, 4)

generator = torch.Generator()

sae.encoder.weight = torch.nn.Parameter(torch.randn(4, 2, generator=generator) * 3)
sae.decoder.weight = torch.nn.Parameter(torch.randn(2, 4, generator=generator) * 5)

# print(sae.encoder.weight)
# print(sae.decoder.weight)


def calculate_base_norm(sae: SparseAutoEncoder, x: torch.Tensor, keepdim: bool = False):
    encoder_norm = sae.encoder_norm(keepdim=keepdim)
    decoder_norm = sae.decoder_norm(keepdim=keepdim)
    y = torch.sum(encoder_norm * x).squeeze()
    y.backward()
    z = torch.sum(decoder_norm * x).squeeze()
    z.backward()
    return encoder_norm.to("cpu"), decoder_norm.to("cpu")


def calculate_tensor_parallel_norm(sae: SparseAutoEncoder, x: torch.Tensor, keepdim: bool = False):
    encoder_norm = sae.encoder_norm(keepdim=keepdim, device_mesh=device_mesh)
    decoder_norm = sae.decoder_norm(keepdim=keepdim, device_mesh=device_mesh)
    sae.set_tensor_paralleled(True)
    plan = {
        "encoder": ColwiseParallel(output_layouts=Replicate()),
        "decoder": RowwiseParallel(input_layouts=Replicate()),
    }
    sae = parallelize_module(sae, device_mesh=sae.device_mesh["tp"], parallelize_plan=plan)
    y = torch.sum(sae.encoder_norm() * x).squeeze()
    y.backward()
    z = torch.sum(sae.decoder_norm() * x).squeeze()
    z.backward()
    return encoder_norm.to("cpu"), decoder_norm.to("cpu")


def test_tp_norm(keepdim: bool = False):
    x = torch.randn(1).to(sae.cfg.device)
    encoder_norm_base, decoder_norm_base = calculate_base_norm(sae, x, keepdim=keepdim)
    encoder_norm_tp, decoder_norm_tp = calculate_tensor_parallel_norm(sae, x, keepdim=keepdim)
    assert torch.allclose(encoder_norm_base, encoder_norm_tp)
    assert torch.allclose(decoder_norm_base, decoder_norm_tp)
    if dist.get_rank() == 0:
        print(
            f"test_tp_norm passed\nencoder norm base {encoder_norm_base}, encoder norm tp {encoder_norm_tp}\ndecoder norm base {decoder_norm_base}, decoder norm tp {decoder_norm_tp}"
        )


if __name__ == "__main__":
    test_tp_norm(keepdim=False)
