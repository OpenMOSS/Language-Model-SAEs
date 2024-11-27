import os
from typing import Dict

import torch



@torch.no_grad()
def list_activation_chunks(activation_path: str, hook_point: str) -> list[str]:
    print(f'Reading cached activations from {os.path.join(activation_path, hook_point)}')
    return sorted(
        [
            os.path.join(activation_path, hook_point, f)
            for f in os.listdir(os.path.join(activation_path, hook_point))
            if f.endswith(".pt")
        ]
    )


@torch.no_grad()
def load_activation_chunk(chunk_path: str) -> Dict[str, torch.Tensor]:
    return torch.load(chunk_path, weights_only=True)