import os

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from llamascopium.backend.language_model import LanguageModelConfig, TransformerLensLanguageModel
from llamascopium.models.sparse_dictionary import SparseDictionary
from llamascopium.resource_loaders import load_model


def load_replacement_modules(
    layers: list[int], exp_factor: int, topk: int, include_lorsa: bool = True, device_mesh: DeviceMesh | None = None
):
    replacement_modules = []
    sae_types = ["lorsa", "transcoder"] if include_lorsa else ["transcoder"]
    for layer in layers:
        for sae_type in sae_types:
            local_sae_path = f"OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B:{sae_type}/{exp_factor}x/k{topk}/layer{layer}_{sae_type}_{exp_factor}x_k{topk}"
            replacement_modules.append(
                SparseDictionary.from_pretrained(
                    local_sae_path,
                    device="cuda",
                    dtype="torch.float32",
                    fold_activation_scale=False,
                    device_mesh=device_mesh,
                )
            )
    return replacement_modules


def load_language_model(device_mesh: DeviceMesh | None = None):
    model_cfg = LanguageModelConfig(
        model_name="Qwen/Qwen3-1.7B",
        device="cuda",
        dtype="torch.float32",
        prepend_bos=False,
    )
    return load_model(model_cfg, device_mesh)


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(int(os.environ.get("WORLD_SIZE", "1")),), mesh_dim_names=("model",)
    )
    model: TransformerLensLanguageModel = load_language_model(device_mesh)
    replacement_modules = load_replacement_modules(
        layers=list(range(model.cfg.n_layers)), exp_factor=8, topk=64, include_lorsa=True, device_mesh=device_mesh
    )

    attribute_result = model.attribute(
        "The National Digital ",
        replacement_modules=replacement_modules,
        max_n_logits=10,
        desired_logit_prob=0.95,
        batch_size=64,
        max_features=4096,
        enable_qk_tracing=True,
        qk_top_fraction=0.6,
    )

    print(attribute_result)
