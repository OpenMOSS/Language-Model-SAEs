import os
from typing import Dict, Optional

import torch
from safetensors.torch import load_file


@torch.no_grad()
def list_activation_chunks(activation_path: str, hook_point: str) -> list[str]:
    print(f"Reading cached activations from {os.path.join(activation_path, hook_point)}")
    return sorted(
        [
            os.path.join(activation_path, hook_point, f)
            for f in os.listdir(os.path.join(activation_path, hook_point))
            if f.endswith(".pt")
        ]
    )


@torch.no_grad()
def load_activation_chunk(chunk_path: str, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
    if chunk_path.endswith(".safetensors"):
        return load_file(chunk_path, device=device or "cpu")
    else:
        return torch.load(chunk_path, weights_only=True, map_location=device)
