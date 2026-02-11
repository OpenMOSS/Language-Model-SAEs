
from __future__ import annotations

from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from lm_saes import SparseAutoEncoder, LowRankSparseAttention
from tqdm.auto import tqdm
from src.path_evaluation.apply_layernorm import apply_layernorm_path_with_feature_types

from src.chess_utils import get_feature_vector


def compute_decoder_cos_sim_single(
    layer1: int,
    feature_id1: int,
    feature_type1: str,
    layer2: int,
    feature_id2: int,
    feature_type2: str,
    transcoders: dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
) -> Optional[float]:
    """Compute cosine similarity between decoder vectors of two features.

    The feature may come from either a transcoder SAE or a LORSA block, as
    specified by ``feature_type*``. This mirrors the logic of
    ``compute_virtual_weight_single`` in ``virtual_weight.py``, but compares
    decoder vectors on both sides instead of decoder vs encoder.
    """
    feature_type1 = feature_type1.lower()
    feature_type2 = feature_type2.lower()

    if (
        layer1 == layer2
        and feature_id1 == feature_id2
        and feature_type1 == feature_type2
    ):
        return None

    dec_vec1 = get_feature_vector(
        lorsas=lorsas,
        transcoders=transcoders,
        feature_type=feature_type1,
        layer=layer1,
        feature_id=feature_id1,
    )
    dec_vec2 = get_feature_vector(
        lorsas=lorsas,
        transcoders=transcoders,
        feature_type=feature_type2,
        layer=layer2,
        feature_id=feature_id2,
    )

    if layer1 != layer2:
        if layer1 < layer2:
            # 将 dec_vec1 从 layer1 的 feature_type1 位置传播到 layer2 的 feature_type2 位置
            dec_vec1 = apply_layernorm_path_with_feature_types(
                dec_vec1,
                src_layer=layer1,
                src_feature_type=feature_type1,
                tgt_layer=layer2,
                tgt_feature_type=feature_type2,
            )
        else:
            # 将 dec_vec2 从 layer2 的 feature_type2 位置传播到 layer1 的 feature_type1 位置
            dec_vec2 = apply_layernorm_path_with_feature_types(
                dec_vec2,
                src_layer=layer2,
                src_feature_type=feature_type2,
                tgt_layer=layer1,
                tgt_feature_type=feature_type1,
            )

    norm1 = dec_vec1.norm()
    norm2 = dec_vec2.norm()
    if norm1 == 0 or norm2 == 0:
        return None

    cos_sim = torch.dot(dec_vec1, dec_vec2) / (norm1 * norm2)
    return float(cos_sim.item())


def _decoder_cos_sims_within_csv(
    csv_path: str,
    transcoders: dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
) -> List[float]:
    """Compute all pairwise decoder cosine similarities within a single CSV."""
    df = pd.read_csv(csv_path)
    sims: List[float] = []
    n = len(df)
    for i in tqdm(range(n), desc=f"decoder_cos_within:{Path(csv_path).name}"):
        row_i = df.iloc[i]
        layer1 = int(row_i["layer"])
        fid1 = int(row_i["feature_id"])
        type1 = str(row_i["feature_type"])
        for j in range(i + 1, n):
            row_j = df.iloc[j]
            layer2 = int(row_j["layer"])
            fid2 = int(row_j["feature_id"])
            type2 = str(row_j["feature_type"])
            sim = compute_decoder_cos_sim_single(
                layer1=layer1,
                feature_id1=fid1,
                feature_type1=type1,
                layer2=layer2,
                feature_id2=fid2,
                feature_type2=type2,
                transcoders=transcoders,
                lorsas=lorsas,
            )
            if sim is not None:
                sims.append(sim)
    return sims


def decoder_cos_sim_between_two_csv(
    csv_path_src: str,
    csv_path_tgt: str,
    transcoders: dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
) -> pd.DataFrame:
    """Compute decoder cosine similarity for all pairs between two CSV feature lists."""
    df_src = pd.read_csv(csv_path_src)
    df_tgt = pd.read_csv(csv_path_tgt)
    records: List[Dict[str, object]] = []
    for i in tqdm(
        range(len(df_src)),
        desc=f"decoder_cos_between:src={Path(csv_path_src).name},tgt={Path(csv_path_tgt).name}",
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
            sim = compute_decoder_cos_sim_single(
                layer1=layer1,
                feature_id1=fid1,
                feature_type1=type1,
                layer2=layer2,
                feature_id2=fid2,
                feature_type2=type2,
                transcoders=transcoders,
                lorsas=lorsas,
            )
            if sim is None:
                continue
            records.append(
                {
                    "src_layer": layer1,
                    "src_feature_id": fid1,
                    "src_feature_type": type1,
                    "tgt_layer": layer2,
                    "tgt_feature_id": fid2,
                    "tgt_feature_type": type2,
                    "decoder_cos_sim": sim,
                }
            )
    return pd.DataFrame(records)


def _mean_from_list(values: List[float]) -> float:
    """Return the arithmetic mean, or NaN if the list is empty."""
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def compute_folder_decoder_cos_sim(
    folder_path: str,
    transcoders: dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
) -> Tuple[Dict[str, float], List[Tuple[str, str, float]], float, float]:
    """Compute within-file and cross-file mean decoder cosine similarities in a folder.

    The function mirrors ``compute_folder_virtual_weight`` but uses decoder
    cosine similarity as the metric. It assumes all relevant CSVs match
    ``*top100_features.csv``.
    """
    folder = Path(folder_path)
    csv_files = sorted(folder.glob("*top100_features.csv"), key=lambda p: p.name)
    within_means: Dict[str, float] = {}
    for csv in tqdm(csv_files, desc="decoder_cos_within_folder"):
        sims = _decoder_cos_sims_within_csv(
            str(csv),
            transcoders,
            lorsas,
        )
        within_means[csv.name] = _mean_from_list(sims)

    cross_means: List[Tuple[str, str, float]] = []
    for a, b in tqdm(
        list(combinations(csv_files, 2)),
        desc="decoder_cos_between_files",
    ):
        df = decoder_cos_sim_between_two_csv(
            str(a),
            str(b),
            transcoders=transcoders,
            lorsas=lorsas,
        )
        mean_sim = (
            float(df["decoder_cos_sim"].mean()) if not df.empty else float("nan")
        )
        cross_means.append((a.name, b.name, mean_sim))

    within_mean = (
        _mean_from_list([v for v in within_means.values() if not pd.isna(v)])
        if within_means
        else float("nan")
    )
    cross_mean = (
        _mean_from_list(
            [v for _, _, v in cross_means if not pd.isna(v)]
        )
        if cross_means
        else float("nan")
    )

    return within_means, cross_means, within_mean, cross_mean
