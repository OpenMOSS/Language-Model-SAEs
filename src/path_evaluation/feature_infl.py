from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.feature_and_steering.interact import analyze_node_activation_impact


FeatureNode = Tuple[int, int, int, str]


def precompute_activations_and_weights(
    fen: str,
    model: Any,
    transcoders: list[Any],
    lorsas: list[Any],
) -> Dict[str, Any]:
    with torch.no_grad():
        _, cache = model.run_with_cache(fen, prepend_bos=False)
    
    model_device = next(model.parameters()).device
    tc_activations = []
    lorsa_activations = []
    for layer in range(15):
        lorsa_input = cache[f'blocks.{layer}.hook_attn_in']
        if lorsas is not None and layer < len(lorsas) and lorsas[layer] is not None:
            lorsa_dense_activation = lorsas[layer].encode(lorsa_input)
            lorsa_sparse_activation = lorsa_dense_activation.to_sparse_coo()
        else:
            lorsa_sparse_activation = torch.sparse_coo_tensor(
                torch.empty(3, 0, dtype=torch.long),
                torch.empty(0),
                (1, 64, 128)
            ).to(model_device)
        lorsa_activations.append(lorsa_sparse_activation)
        tc_input = cache[f'blocks.{layer}.resid_mid_after_ln']
        if transcoders is not None and layer < len(transcoders) and transcoders[layer] is not None:
            tc_dense_activation = transcoders[layer].encode(tc_input)
            tc_sparse_activation = tc_dense_activation.to_sparse_coo()
        else:
            tc_sparse_activation = torch.sparse_coo_tensor(
                torch.empty(3, 0, dtype=torch.long),
                torch.empty(0),
                (1, 64, 128)
            ).to(model_device)
        tc_activations.append(tc_sparse_activation)
    tc_WDs = []
    lorsa_WDs = []
    for layer in range(15):
        if transcoders is not None and layer < len(transcoders) and transcoders[layer] is not None:
            tc_WDs.append(transcoders[layer].W_D.detach().to(model_device))
        else:
            tc_WDs.append(None)
        if lorsas is not None and layer < len(lorsas) and lorsas[layer] is not None:
            lorsa_WDs.append(lorsas[layer].W_O.detach().to(model_device))
        else:
            lorsa_WDs.append(None)
    return {
        "cache": cache,
        "tc_activations": tc_activations,
        "lorsa_activations": lorsa_activations,
        "tc_WDs": tc_WDs,
        "lorsa_WDs": lorsa_WDs,
        "model": model,
        "transcoders": transcoders,
        "lorsas": lorsas,
    }


def batch_analyze_node_interactions(
    precomputed_data: Dict[str, Any],
    feature_pairs: List[Tuple[FeatureNode, FeatureNode]],
    steering_scale: float = 0.0,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for source_node, target_node in tqdm(
        feature_pairs, desc="Analyzing feature interactions"
    ):
        try:
            result = analyze_node_activation_impact(
                steering_nodes=[source_node],
                target_nodes=target_node,
                steering_scale=steering_scale,
                cache=precomputed_data["cache"],
                model=precomputed_data["model"],
                tc_activations=precomputed_data["tc_activations"],
                lorsa_activations=precomputed_data["lorsa_activations"],
                tc_WDs=precomputed_data["tc_WDs"],
                lorsa_WDs=precomputed_data["lorsa_WDs"],
                transcoders=precomputed_data["transcoders"],
                lorsas=precomputed_data["lorsas"],
            )
            result["source_node"] = source_node
            result["target_node_input"] = target_node
            results.append(result)
        except Exception as e:
            print(f"Error processing {source_node} -> {target_node}: {e}")
            continue
    return results


def _build_feature_pairs_within(unique_features: pd.DataFrame) -> List[Tuple[FeatureNode, FeatureNode]]:
    feature_pairs: List[Tuple[FeatureNode, FeatureNode]] = []
    for i in range(len(unique_features)):
        for j in range(i + 1, len(unique_features)):
            f1 = unique_features.iloc[i]
            f2 = unique_features.iloc[j]

            layer1, layer2 = int(f1["layer"]), int(f2["layer"])
            type1, type2 = f1["feature_type"], f2["feature_type"]
            pos1, pos2 = int(f1["position_idx"]), int(f2["position_idx"])
            fid1, fid2 = int(f1["feature_id"]), int(f2["feature_id"])

            # 排除相同的 feature（self-connection 不应该包含相同的 feature）
            if layer1 == layer2 and pos1 == pos2 and fid1 == fid2 and type1 == type2:
                continue

            if layer1 == layer2:
                if type1 == "lorsa" and type2 == "transcoder":
                    source, target = f1, f2
                elif type1 == "transcoder" and type2 == "lorsa":
                    source, target = f2, f1
                else:
                    continue
            else:
                if layer1 < layer2:
                    source, target = f1, f2
                else:
                    source, target = f2, f1

            source_node: FeatureNode = (
                int(source["layer"]),
                int(source["position_idx"]),
                int(source["feature_id"]),
                str(source["feature_type"]),
            )

            target_node: FeatureNode = (
                int(target["layer"]),
                int(target["position_idx"]),
                int(target["feature_id"]),
                str(target["feature_type"]),
            )

            feature_pairs.append((source_node, target_node))
    return feature_pairs


def _build_feature_pairs_between(
    unique_features_src: pd.DataFrame,
    unique_features_tgt: pd.DataFrame,
) -> List[Tuple[FeatureNode, FeatureNode]]:
    feature_pairs: List[Tuple[FeatureNode, FeatureNode]] = []
    for _, f1 in unique_features_src.iterrows():
        for _, f2 in unique_features_tgt.iterrows():
            layer1, layer2 = int(f1["layer"]), int(f2["layer"])
            type1, type2 = f1["feature_type"], f2["feature_type"]
            pos1, pos2 = int(f1["position_idx"]), int(f2["position_idx"])
            fid1, fid2 = int(f1["feature_id"]), int(f2["feature_id"])

            # 排除相同的 feature（cross-connection 不应该包含相同的 feature）
            if layer1 == layer2 and pos1 == pos2 and fid1 == fid2 and type1 == type2:
                continue

            if layer1 == layer2:
                if type1 == "lorsa" and type2 == "transcoder":
                    source, target = f1, f2
                else:
                    continue
            else:
                if layer1 < layer2:
                    source, target = f1, f2
                else:
                    continue

            source_node: FeatureNode = (
                int(source["layer"]),
                int(source["position_idx"]),
                int(source["feature_id"]),
                str(source["feature_type"]),
            )

            target_node: FeatureNode = (
                int(target["layer"]),
                int(target["position_idx"]),
                int(target["feature_id"]),
                str(target["feature_type"]),
            )

            feature_pairs.append((source_node, target_node))

    return feature_pairs


def _process_interaction_results(
    results: List[Dict[str, Any]],
    steering_scale: float,
    output_prefix: Optional[str] = None,
    threshold: float = 0.1,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    processed_results: List[Dict[str, Any]] = []
    for result in results:
        if "target_nodes" in result and len(result["target_nodes"]) > 0:
            target_result = result["target_nodes"][0]
            original_activation = target_result["original_activation"]
            modified_activation = target_result["modified_activation"]
            activation_change = target_result["activation_change"]
        else:
            original_activation = result.get("original_activation", 0.0)
            modified_activation = result.get("modified_activation", 0.0)
            activation_change = result.get("activation_change", 0.0)

        if original_activation != 0:
            reduction_ratio = activation_change / original_activation
        else:
            reduction_ratio = 0.0

        source_node = result["source_node"]
        target_node = result["target_node_input"]

        processed_result: Dict[str, Any] = {
            "source_layer": source_node[0],
            "source_pos": source_node[1],
            "source_feature": source_node[2],
            "source_type": source_node[3],
            "target_layer": target_node[0],
            "target_pos": target_node[1],
            "target_feature": target_node[2],
            "target_type": target_node[3],
            "original_activation": original_activation,
            "modified_activation": modified_activation,
            "activation_change": activation_change,
            "steering_scale": result.get("steering_scale", steering_scale),
            "reduction_ratio": reduction_ratio,
            "target_node_str": target_result.get("target_node", "")
            if "target_nodes" in result and len(result["target_nodes"]) > 0
            else result.get("target_node", ""),
            "steering_nodes_count": result.get("steering_nodes_count", 1),
        }

        processed_results.append(processed_result)

    json_output_file: Optional[str] = None
    csv_output_file: Optional[str] = None

    if output_prefix is not None:
        json_output_file = f"{output_prefix}.json"
        with open(json_output_file, "w") as f:
            json.dump(processed_results, f, indent=2)

        results_df = pd.DataFrame(processed_results)
        filtered_df = results_df[results_df["reduction_ratio"] <= -threshold]
        csv_output_file = f"{output_prefix}_reduction_{threshold}.csv"
        filtered_df.to_csv(csv_output_file, index=False)

    return processed_results, json_output_file, csv_output_file


def compute_feature_interactions_batch(
    csv_file_path: str,
    fen: str,
    model: Any,
    transcoders: list[Any],
    lorsas: list[Any],
    steering_scale: float = 0.0,
    output_prefix: str = "feature_interaction_results",
) -> Tuple[List[Dict[str, Any]], str, str]:
    df = pd.read_csv(csv_file_path)
    unique_features = df[
        ["position_name", "position_idx", "layer", "feature_id", "feature_type"]
    ].drop_duplicates()

    print(f"找到 {len(unique_features)} 个unique features")
    print("预计算激活值和权重...")
    precomputed_data = precompute_activations_and_weights(
        fen, model, transcoders, lorsas
    )

    feature_pairs = _build_feature_pairs_within(unique_features)
    print(f"将计算 {len(feature_pairs)} 个不同层feature组合")

    print("开始批量计算feature interactions...")
    raw_results = batch_analyze_node_interactions(
        precomputed_data, feature_pairs, steering_scale
    )

    processed_results, json_file, csv_file = _process_interaction_results(
        raw_results, steering_scale, output_prefix, threshold=0.1
    )

    print(f"成功计算了 {len(processed_results)} 个feature组合")
    if json_file:
        print(f"JSON结果已保存到 {json_file}")
    if csv_file:
        print(f"CSV结果已保存到 {csv_file}")

    return processed_results, json_file, csv_file


def _mean_infl_from_results(results: List[Dict[str, Any]]) -> float:
    """计算平均下降比例（reduction_ratio），而不是绝对差值（activation_change）"""
    infls = [float(r.get("reduction_ratio", 0.0)) for r in results]
    if not infls:
        return float("nan")
    return float(sum(infls) / len(infls))


def compute_self_conn_for_csv(
    csv_file_path: str,
    precomputed_data: Dict[str, Any],
    steering_scale: float = 0.0,
    output_prefix: Optional[str] = None,
) -> Tuple[float, Optional[str], Optional[str]]:
    df = pd.read_csv(csv_file_path)
    unique_features = df[
        ["position_name", "position_idx", "layer", "feature_id", "feature_type"]
    ].drop_duplicates()

    feature_pairs = _build_feature_pairs_within(unique_features)
    raw_results = batch_analyze_node_interactions(
        precomputed_data, feature_pairs, steering_scale
    )
    processed_results, json_file, csv_file = _process_interaction_results(
        raw_results, steering_scale, output_prefix, threshold=0.1
    )
    self_conn = _mean_infl_from_results(processed_results)
    return self_conn, json_file, csv_file


def compute_cross_conn_between_csvs(
    csv_file_src: str,
    csv_file_tgt: str,
    precomputed_data: Dict[str, Any],
    steering_scale: float = 0.0,
    output_prefix: Optional[str] = None,
) -> Tuple[float, Optional[str], Optional[str]]:
    df_src = pd.read_csv(csv_file_src)
    df_tgt = pd.read_csv(csv_file_tgt)

    unique_src = df_src[
        ["position_name", "position_idx", "layer", "feature_id", "feature_type"]
    ].drop_duplicates()
    unique_tgt = df_tgt[
        ["position_name", "position_idx", "layer", "feature_id", "feature_type"]
    ].drop_duplicates()

    feature_pairs = _build_feature_pairs_between(unique_src, unique_tgt)
    raw_results = batch_analyze_node_interactions(
        precomputed_data, feature_pairs, steering_scale
    )
    processed_results, json_file, csv_file = _process_interaction_results(
        raw_results, steering_scale, output_prefix, threshold=0.1
    )
    cross_conn = _mean_infl_from_results(processed_results)
    return cross_conn, json_file, csv_file


def compute_folder_csr(
    folder_path: str,
    fen: str,
    model: Any,
    transcoders: list[Any],
    lorsas: list[Any],
    steering_scale: float = 0.0,
    output_dir: Optional[str] = None,
) -> Tuple[List[Tuple[str, str, float]], float]:
    folder = Path(folder_path)
    csv_files = sorted(folder.glob("*_top*_features.csv"), key=lambda p: p.name)
    if len(csv_files) < 2:
        return [], float("nan")

    out_root = Path(output_dir) if output_dir is not None else folder

    print("预计算激活值和权重（全局一次）...")
    precomputed_data = precompute_activations_and_weights(
        fen, model, transcoders, lorsas
    )

    move_names: Dict[Path, str] = {}
    for csv_path in csv_files:
        stem = csv_path.stem
        move_names[csv_path] = stem.split("_top")[0]

    self_conn: Dict[str, float] = {}
    for csv_path in csv_files:
        move = move_names[csv_path]
        prefix = str(out_root / f"{move}_{move}")
        sc, _, _ = compute_self_conn_for_csv(
            str(csv_path),
            precomputed_data=precomputed_data,
            steering_scale=steering_scale,
            output_prefix=prefix,
        )
        self_conn[csv_path.name] = sc

    csr_pairs: List[Tuple[str, str, float]] = []
    from itertools import combinations
    import math

    for csv_a, csv_b in combinations(csv_files, 2):
        move_a = move_names[csv_a]
        move_b = move_names[csv_b]
        prefix = str(out_root / f"{move_a}_{move_b}")
        cross_conn, _, _ = compute_cross_conn_between_csvs(
            str(csv_a),
            str(csv_b),
            precomputed_data=precomputed_data,
            steering_scale=steering_scale,
            num_workers=num_workers,
            output_prefix=prefix,
        )

        sc_a = self_conn.get(csv_a.name, float("nan"))
        sc_b = self_conn.get(csv_b.name, float("nan"))
        denom = min(sc_a, sc_b)
        if denom == 0.0 or math.isnan(denom):
            csr = float("nan")
        else:
            csr = cross_conn / denom

        csr_pairs.append((csv_a.name, csv_b.name, csr))

    valid_csrs = [v for _, _, v in csr_pairs if not math.isnan(v)]
    if not valid_csrs:
        mean_csr = float("nan")
    else:
        mean_csr = float(sum(valid_csrs) / len(valid_csrs))

    return csr_pairs, mean_csr
