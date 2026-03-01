from __future__ import annotations

import json
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from lm_saes import SparseAutoEncoder, LowRankSparseAttention
from tqdm.auto import tqdm
from src.path_evaluation.apply_layernorm import apply_layernorm_path_with_feature_types

from src.chess_utils import get_feature_vector, get_feature_encoder_vector


def compute_virtual_weight_single(
    layer1: int,
    feature_id1: int,
    feature_type1: str,
    layer2: int,
    feature_id2: int,
    feature_type2: str,
    transcoders: dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    apply_layernorm_path: bool = True,
) -> Optional[float]:
    feature_type1 = feature_type1.lower()
    feature_type2 = feature_type2.lower()

    if layer1 == layer2 and feature_type1 == feature_type2:
        return None

    if layer1 == layer2 and feature_type1 != feature_type2:
        if feature_type1 == "lorsa" and feature_type2 == "transcoder":
            upstream = (layer1, feature_id1, feature_type1)
            downstream = (layer2, feature_id2, feature_type2)
        elif feature_type1 == "transcoder" and feature_type2 == "lorsa":
            upstream = (layer2, feature_id2, feature_type2)
            downstream = (layer1, feature_id1, feature_type1)
        else:
            return None
    else:
        if layer1 < layer2:
            upstream = (layer1, feature_id1, feature_type1)
            downstream = (layer2, feature_id2, feature_type2)
        elif layer2 < layer1:
            upstream = (layer2, feature_id2, feature_type2)
            downstream = (layer1, feature_id1, feature_type1)
        else:
            return None

    up_layer, up_fid, up_type = upstream
    down_layer, down_fid, down_type = downstream

    dec_vec = get_feature_vector(
        lorsas=lorsas,
        transcoders=transcoders,  # type: ignore[arg-type]
        feature_type=up_type,  # type: ignore[arg-type]
        layer=up_layer,
        feature_id=up_fid,
    )

    # Consider the position of the feature types in the path propagation function
    if apply_layernorm_path:
        dec_vec = apply_layernorm_path_with_feature_types(
            dec_vec,
            src_layer=up_layer,
            src_feature_type=up_type,  # type: ignore[arg-type]
            tgt_layer=down_layer,
            tgt_feature_type=down_type,  # type: ignore[arg-type]
        )

    enc_vec = get_feature_encoder_vector(
        lorsas=lorsas,
        transcoders=transcoders,  # type: ignore[arg-type]
        feature_type=down_type,  # type: ignore[arg-type]
        layer=down_layer,
        feature_id=down_fid,
    )

    dec_norm = dec_vec.norm()
    enc_norm = enc_vec.norm()
    if dec_norm == 0 or enc_norm == 0:
        return None

    cos_sim = torch.dot(dec_vec, enc_vec) / (dec_norm * enc_norm)
    return float(cos_sim.item())


def _virtual_weights_within_csv(
    csv_path: str,
    transcoders: dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    apply_layernorm_path: bool = True,
) -> List[float]:
    df = pd.read_csv(csv_path)
    weights: List[float] = []
    n = len(df)
    for i in tqdm(range(n), desc=f"virtual_weight_within:{Path(csv_path).name}"):
        row_i = df.iloc[i]
        layer1 = int(row_i["layer"])
        fid1 = int(row_i["feature_id"])
        type1 = str(row_i["feature_type"])
        for j in range(i + 1, n):
            row_j = df.iloc[j]
            layer2 = int(row_j["layer"])
            fid2 = int(row_j["feature_id"])
            type2 = str(row_j["feature_type"])
            vw = compute_virtual_weight_single(
                layer1=layer1,
                feature_id1=fid1,
                feature_type1=type1,
                layer2=layer2,
                feature_id2=fid2,
                feature_type2=type2,
                transcoders=transcoders,
                lorsas=lorsas,
                apply_layernorm_path=apply_layernorm_path,
            )
            if vw is not None:
                weights.append(vw)
    return weights


def virtual_weight_between_two_csv(
    csv_path_src: str,
    csv_path_tgt: str,
    transcoders: dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    apply_layernorm_path: bool = True,
) -> pd.DataFrame:
    df_src = pd.read_csv(csv_path_src)
    df_tgt = pd.read_csv(csv_path_tgt)
    records: List[Dict[str, object]] = []
    for i in tqdm(
        range(len(df_src)),
        desc=f"virtual_weight_between:src={Path(csv_path_src).name},tgt={Path(csv_path_tgt).name}",
    ):
        row_i = df_src.iloc[i]
        layer1 = int(row_i["layer"])
        fid1 = int(row_i["feature_id"])
        type1 = str(row_i["feature_type"])
        for j in range(len(df_tgt)):
            row_j = df_tgt.iloc[j]
            layer2 = int(row_j["layer"])
            fid2 = int(row_j["feature_id"])
            type2 = str(row_j["feature_type"])
            vw = compute_virtual_weight_single(
                layer1=layer1,
                feature_id1=fid1,
                feature_type1=type1,
                layer2=layer2,
                feature_id2=fid2,
                feature_type2=type2,
                transcoders=transcoders,
                lorsas=lorsas,
                apply_layernorm_path=apply_layernorm_path,
            )
            if vw is None:
                continue
            records.append(
                {
                    "src_layer": layer1,
                    "src_feature_id": fid1,
                    "src_feature_type": type1,
                    "tgt_layer": layer2,
                    "tgt_feature_id": fid2,
                    "tgt_feature_type": type2,
                    "virtual_weight": vw,
                }
            )
    return pd.DataFrame(records)


def _mean_from_list(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def compute_weighted_virtual_weight_from_json(
    json_path: Path,
    transcoders: dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    apply_layernorm_path: bool = True,
) -> Tuple[float, int, int]:
    """Read feature pairs from a JSON file, compute the weighted average virtual_weight.

    Parameters:
    json_path:
        JSON file path, containing feature pairs data.
    transcoders:
        Transcoder SAE dictionary.
    lorsas:
        Lorsa list.
    apply_layernorm_path:
        Whether to apply LayerNorm path propagation.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON file {json_path} should contain a list of feature pairs")

    virtual_weights: List[float] = []
    weights: List[float] = []

    for pair in tqdm(data, desc=f"Processing {json_path.name}", leave=False):
        source_layer = int(pair["source_layer"])
        source_feature = int(pair["source_feature"])
        source_type = str(pair["source_type"]).lower()

        target_layer = int(pair["target_layer"])
        target_feature = int(pair["target_feature"])
        target_type = str(pair["target_type"]).lower()

        reduction_ratio = float(pair.get("reduction_ratio", 0.0))
        weight = -reduction_ratio
        vw = compute_virtual_weight_single(
            layer1=source_layer,
            feature_id1=source_feature,
            feature_type1=source_type,  # type: ignore[arg-type]
            layer2=target_layer,
            feature_id2=target_feature,
            feature_type2=target_type,  # type: ignore[arg-type]
            transcoders=transcoders,  # type: ignore[arg-type]
            lorsas=lorsas,
            apply_layernorm_path=apply_layernorm_path,
        )

        if vw is not None:
            virtual_weights.append(vw)
            weights.append(weight)

    total_pairs = len(data)
    successful_pairs = len(virtual_weights)

    if successful_pairs == 0:
        return float("nan"), 0, total_pairs

    total_weight = sum(weights)
    if total_weight == 0:
        weighted_mean = sum(virtual_weights) / successful_pairs
    else:
        weighted_mean = sum(vw * w for vw, w in zip(virtual_weights, weights)) / total_weight

    return float(weighted_mean), successful_pairs, total_pairs


def _is_within_path(json_filename: str) -> bool:
    """Check if the JSON file name represents a virtual_weight within a path.
    
    Within path: source and target come from the same path (e.g. f6g7_f6g7.json)
    Between path: source and target come from different paths (e.g. f6g7_h3d7.json)
    
    File name format: move1_move2.json, where move1 and move2 are 4-character chess moves (e.g. f6g7, h3d7)
    If move1 == move2, it is within path; otherwise it is between path.
    """
    # Extract the file name (without extension)
    name = json_filename.replace(".json", "")
    parts = name.split("_")
    
    # Simple check: if there are only two parts and they are the same, it is within path
    if len(parts) == 2:
        return parts[0] == parts[1]
    
    # If there are more than two parts, try to extract the first two parts and the last two parts
    # e.g. f6g7_f6g7.json -> ['f6g7', 'f6g7']
    # or: move1_move2_move3_move4.json -> check if the first two are equal to the last two
    if len(parts) >= 2:
        # Check if the first part (usually a 4-character chess move) is equal to the second part
        if len(parts[0]) == 4 and len(parts[1]) == 4:
            return parts[0] == parts[1]
    
    return False


def compute_folder_virtual_weight(
    folder_path: str,
    transcoders: dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    apply_layernorm_path: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    """Compute the virtual_weight of CSV or JSON files in a folder.

    If the folder contains JSON files (containing feature pairs and reduction_ratio),
    use weighted average (weight = -reduction_ratio) to compute virtual_weight.
    It will distinguish between within path and between path virtual_weight.

    Returns:
    Tuple[Dict[str, float], Dict[str, float], float, float]
        (within path virtual_weight dictionary, between path virtual_weight dictionary,
         within path mean, between path mean)
    """
    folder = Path(folder_path)
    
    # Check if there are JSON files (exclude infl_all_feature.json etc.)
    json_files = [
        f for f in sorted(folder.glob("*.json"), key=lambda p: p.name)
        if not f.name.startswith("infl_")
    ]
    
    if json_files:
        # Use JSON file mode: compute weighted average virtual_weight
        # Distinguish between within path and between path
        within_path_means: Dict[str, float] = {}  # within path virtual_weight
        between_path_means: Dict[str, float] = {}  # between path virtual_weight
        
        for json_file in tqdm(json_files, desc="virtual_weight_from_json"):
            weighted_vw, successful, total = compute_weighted_virtual_weight_from_json(
                json_file,
                transcoders,
                lorsas,
                apply_layernorm_path=apply_layernorm_path,
            )
            
            # Check if the file name represents within path or between path
            if _is_within_path(json_file.name):
                within_path_means[json_file.name] = weighted_vw
            else:
                between_path_means[json_file.name] = weighted_vw
        
        # Compute the mean of within path and between path
        within_path_mean = (
            _mean_from_list([v for v in within_path_means.values() if not pd.isna(v)])
            if within_path_means
            else float("nan")
        )
        between_path_mean = (
            _mean_from_list([v for v in between_path_means.values() if not pd.isna(v)])
            if between_path_means
            else float("nan")
        )
        
        return within_path_means, between_path_means, within_path_mean, between_path_mean
    
    else:
        # Use the traditional CSV file mode
        # For CSV mode, we still return the format of within path and between path
        # But in CSV mode, all files are regarded as within path
        csv_files = sorted(folder.glob("*top100_features.csv"), key=lambda p: p.name)
        within_path_means: Dict[str, float] = {}
        for csv in tqdm(csv_files, desc="virtual_weight_within_folder"):
            ws = _virtual_weights_within_csv(
                str(csv),
                transcoders,
                lorsas,
                apply_layernorm_path=apply_layernorm_path,
            )
            within_path_means[csv.name] = _mean_from_list(ws)

        # CSV mode has no concept of between path, return empty dictionary
        between_path_means: Dict[str, float] = {}

        within_path_mean = (
            _mean_from_list([v for v in within_path_means.values() if not pd.isna(v)])
            if within_path_means
            else float("nan")
        )
        between_path_mean = float("nan")

        return within_path_means, between_path_means, within_path_mean, between_path_mean