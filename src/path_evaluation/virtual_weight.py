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

    # 使用考虑 feature 类型位置的路径传播函数
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
    """从 JSON 文件读取 feature pairs，计算加权平均 virtual_weight。

    参数
    ----
    json_path:
        JSON 文件路径，包含 feature pairs 数据。
    transcoders:
        Transcoder SAE 字典。
    lorsas:
        Lorsa 列表。
    apply_layernorm_path:
        是否应用 LayerNorm 路径传播。

    返回
    ----
    Tuple[float, int, int]
        (加权平均 virtual_weight, 成功计算的 pair 数量, 总 pair 数量)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON file {json_path} should contain a list of feature pairs")

    virtual_weights: List[float] = []
    weights: List[float] = []  # 使用 -reduction_ratio 作为权重

    for pair in tqdm(data, desc=f"Processing {json_path.name}", leave=False):
        source_layer = int(pair["source_layer"])
        source_feature = int(pair["source_feature"])
        source_type = str(pair["source_type"]).lower()

        target_layer = int(pair["target_layer"])
        target_feature = int(pair["target_feature"])
        target_type = str(pair["target_type"]).lower()

        reduction_ratio = float(pair.get("reduction_ratio", 0.0))
        weight = -reduction_ratio  # 使用 reduction_ratio 的相反数作为权重

        # 计算 virtual_weight
        # 注意：这里直接传递字符串，compute_virtual_weight_single 内部会处理类型转换
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

    # 计算加权平均
    total_weight = sum(weights)
    if total_weight == 0:
        # 如果所有权重为 0，则使用简单平均
        weighted_mean = sum(virtual_weights) / successful_pairs
    else:
        weighted_mean = sum(vw * w for vw, w in zip(virtual_weights, weights)) / total_weight

    return float(weighted_mean), successful_pairs, total_pairs


def _is_within_path(json_filename: str) -> bool:
    """判断 JSON 文件名是否表示 path 内的 virtual_weight。
    
    path 内：source 和 target 来自同一个 path（例如 f6g7_f6g7.json）
    path 间：source 和 target 来自不同的 path（例如 f6g7_h3d7.json）
    
    文件名格式：move1_move2.json，其中 move1 和 move2 是 4 个字符的棋步（例如 f6g7, h3d7）
    如果 move1 == move2，则是 path 内；否则是 path 间。
    """
    # 提取文件名（不含扩展名）
    name = json_filename.replace(".json", "")
    parts = name.split("_")
    
    # 简单判断：如果只有两个部分，且它们相同，则是 path 内
    if len(parts) == 2:
        return parts[0] == parts[1]
    
    # 如果有多于两个部分，尝试提取前两个部分和后两个部分
    # 例如：f6g7_f6g7.json -> ['f6g7', 'f6g7']
    # 或者：move1_move2_move3_move4.json -> 检查前两个是否等于后两个
    if len(parts) >= 2:
        # 检查第一个部分（通常是 4 个字符的棋步）是否等于第二个部分
        if len(parts[0]) == 4 and len(parts[1]) == 4:
            return parts[0] == parts[1]
    
    return False


def compute_folder_virtual_weight(
    folder_path: str,
    transcoders: dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    apply_layernorm_path: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    """计算文件夹中 CSV 或 JSON 文件的 virtual_weight。

    如果文件夹中包含 JSON 文件（包含 feature pairs 和 reduction_ratio），
    则使用加权平均（权重为 -reduction_ratio）计算 virtual_weight。
    会区分 path 内和 path 间的 virtual_weight。

    参数
    ----
    folder_path:
        文件夹路径，包含 CSV 或 JSON 文件。
    transcoders:
        Transcoder SAE 字典。
    lorsas:
        Lorsa 列表。
    apply_layernorm_path:
        是否应用 LayerNorm 路径传播。

    返回
    ----
    Tuple[Dict[str, float], Dict[str, float], float, float]
        (path 内 virtual_weight 字典, path 间 virtual_weight 字典, 
         path 内整体平均, path 间整体平均)
    """
    folder = Path(folder_path)
    
    # 检查是否有 JSON 文件（排除 infl_all_feature.json 等特殊文件）
    json_files = [
        f for f in sorted(folder.glob("*.json"), key=lambda p: p.name)
        if not f.name.startswith("infl_")
    ]
    
    if json_files:
        # 使用 JSON 文件模式：计算加权平均 virtual_weight
        # 区分 path 内和 path 间
        within_path_means: Dict[str, float] = {}  # path 内的 virtual_weight
        between_path_means: Dict[str, float] = {}  # path 间的 virtual_weight
        
        for json_file in tqdm(json_files, desc="virtual_weight_from_json"):
            weighted_vw, successful, total = compute_weighted_virtual_weight_from_json(
                json_file,
                transcoders,
                lorsas,
                apply_layernorm_path=apply_layernorm_path,
            )
            
            # 根据文件名判断是 path 内还是 path 间
            if _is_within_path(json_file.name):
                within_path_means[json_file.name] = weighted_vw
            else:
                between_path_means[json_file.name] = weighted_vw
        
        # 计算 path 内和 path 间的整体平均
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
        # 使用传统的 CSV 文件模式
        # 对于 CSV 模式，我们仍然返回 path 内和 path 间的格式
        # 但 CSV 模式下，所有文件都视为 path 内
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

        # CSV 模式下没有 path 间的概念，返回空字典
        between_path_means: Dict[str, float] = {}

        within_path_mean = (
            _mean_from_list([v for v in within_path_means.values() if not pd.isna(v)])
            if within_path_means
            else float("nan")
        )
        between_path_mean = float("nan")  # CSV 模式下没有 path 间数据

        return within_path_means, between_path_means, within_path_mean, between_path_mean