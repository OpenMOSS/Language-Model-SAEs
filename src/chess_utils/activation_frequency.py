from typing import Iterable, Literal, Tuple

import torch
from tqdm.auto import tqdm

from .utils import get_piece_type_pos, get_pos_from_square


FeatureType = Literal["transcoder", "lorsa"]


def feature_frequency_with_piece_type(
    model: torch.nn.Module,
    lorsas: list,
    transcoders: dict[int, torch.nn.Module],
    fen_list: list[str],
    feature_type: FeatureType,
    layer: int,
    feature_id: int,
    piece_type: str | None = None,
    max_samples: int | None = None,
    device: torch.device | str = "cuda",
) -> tuple[float, int, int]:
    if piece_type is not None:
        assert piece_type in [
            "my p",
            "my n",
            "my b",
            "my r",
            "my q",
            "my k",
            "opponent's p",
            "opponent's n",
            "opponent's b",
            "opponent's r",
            "opponent's q",
            "opponent's k",
        ]
    assert feature_type in ["transcoder", "lorsa"]
    assert layer in range(model.cfg.n_layers)

    if max_samples is not None:
        fen_list = fen_list[:max_samples]

    if feature_type == "transcoder":
        sae = transcoders[layer]
        input_hook_name = f"blocks.{layer}.resid_mid_after_ln"
    else:
        sae = lorsas[layer]
        input_hook_name = f"blocks.{layer}.hook_attn_in"
    sae = sae.to(device)

    total_positions = 0
    active_positions = 0

    for fen in tqdm(fen_list, desc="Computing feature frequency"):
        _, cache = model.run_with_cache(fen, prepend_bos=False)
        input_tensor = cache[input_hook_name]
        feature_acts = sae.encode(input_tensor).to(device)

        if feature_acts.dim() == 3:
            acts_all = feature_acts[0, :, feature_id]
        elif feature_acts.dim() == 2:
            acts_all = feature_acts[:, feature_id]
        else:
            raise ValueError(f"Unexpected feature_acts shape: {feature_acts.shape}")

        if piece_type is not None:
            positions = get_piece_type_pos(fen, piece_type)
            if not positions:
                continue
            acts = acts_all[positions]
        else:
            acts = acts_all

        total_positions += len(acts)
        active_positions += int((acts > 0).sum().item())

    if total_positions == 0:
        return 0.0, 0, 0

    freq = active_positions / total_positions
    return freq, active_positions, total_positions


def feature_frequency_with_keypoint(
    model: torch.nn.Module,
    lorsas: list,
    transcoders: dict[int, torch.nn.Module],
    fen_list: list[str],
    keypoint_list: Iterable[str],
    feature_type: FeatureType,
    layer: int,
    feature_id: int,
    max_samples: int | None = None,
    device: torch.device | str = "cuda",
) -> tuple[float, int, int]:

    assert feature_type in ["transcoder", "lorsa"]
    assert layer in range(model.cfg.n_layers)

    keypoint_list = list(keypoint_list)
    if max_samples is not None:
        fen_list = fen_list[:max_samples]
        keypoint_list = keypoint_list[:max_samples]

    if feature_type == "transcoder":
        sae = transcoders[layer]
        input_hook_name = f"blocks.{layer}.resid_mid_after_ln"
    else:
        sae = lorsas[layer]
        input_hook_name = f"blocks.{layer}.hook_attn_in"
    sae = sae.to(device)

    total_positions = 0
    active_positions = 0

    for fen, keypoint in tqdm(
        zip(fen_list, keypoint_list), desc="Computing feature frequency at keypoints"
    ):
        _, cache = model.run_with_cache(fen, prepend_bos=False)
        input_tensor = cache[input_hook_name]
        feature_acts = sae.encode(input_tensor).to(device)

        if feature_acts.dim() == 3:
            acts_all = feature_acts[0, :, feature_id]
        elif feature_acts.dim() == 2:
            acts_all = feature_acts[:, feature_id]
        else:
            raise ValueError(f"Unexpected feature_acts shape: {feature_acts.shape}")

        pos = get_pos_from_square(fen, keypoint)
        act = acts_all[pos]

        total_positions += 1
        if act > 0:
            active_positions += 1

    if total_positions == 0:
        return 0.0, 0, 0

    freq = active_positions / total_positions
    return freq, active_positions, total_positions