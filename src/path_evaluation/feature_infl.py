from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.feature_and_steering.interact import analyze_node_activation_impact


FeatureNode = Tuple[int, int, int, str]


def _feature_type_order(feature_type: str) -> int:
    feature_type_lower = feature_type.lower()
    if feature_type_lower == "lorsa":
        return 0
    if feature_type_lower == "transcoder":
        return 1
    raise ValueError(f"Unknown feature_type: {feature_type}")


def _feature_sort_key(node: FeatureNode) -> Tuple[int, int, int, int]:
    layer, pos, feature_id, feature_type = node
    return (layer, _feature_type_order(feature_type), pos, feature_id)


def build_feature_list_from_df(unique_features: pd.DataFrame) -> List[FeatureNode]:
    feature_nodes = {
        (
            int(row["layer"]),
            int(row["position_idx"]),
            int(row["feature_id"]),
            str(row["feature_type"]).lower(),
        )
        for _, row in unique_features.iterrows()
    }
    return sorted(feature_nodes, key=_feature_sort_key)


def _is_valid_causal_edge(source_node: FeatureNode, target_node: FeatureNode) -> bool:
    if source_node == target_node:
        return False

    source_layer, _, _, source_type = source_node
    target_layer, _, _, target_type = target_node

    if source_layer < target_layer:
        return True

    if source_layer == target_layer:
        return source_type.lower() == "lorsa" and target_type.lower() == "transcoder"

    return False


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
            result["target_node_inputs"] = [target_node]
            results.append(result)
        except Exception as e:
            print(f"Error processing {source_node} -> {target_node}: {e}")
            continue
    return results


def _build_feature_pairs_within(unique_features: pd.DataFrame) -> List[Tuple[FeatureNode, FeatureNode]]:
    feature_nodes = build_feature_list_from_df(unique_features)
    feature_pairs: List[Tuple[FeatureNode, FeatureNode]] = []
    for i, source_node in enumerate(feature_nodes):
        for target_node in feature_nodes[i + 1 :]:
            if _is_valid_causal_edge(source_node, target_node):
                feature_pairs.append((source_node, target_node))
    return feature_pairs


def _build_feature_pairs_between(
    unique_features_src: pd.DataFrame,
    unique_features_tgt: pd.DataFrame,
) -> List[Tuple[FeatureNode, FeatureNode]]:
    source_features = build_feature_list_from_df(unique_features_src)
    target_features = build_feature_list_from_df(unique_features_tgt)
    feature_pairs: List[Tuple[FeatureNode, FeatureNode]] = []
    for source_node in source_features:
        for target_node in target_features:
            if _is_valid_causal_edge(source_node, target_node):
                feature_pairs.append((source_node, target_node))

    return feature_pairs


def _build_target_groups_within_feature_list(
    feature_list: List[FeatureNode],
) -> List[Tuple[FeatureNode, List[FeatureNode]]]:
    ordered_features = sorted(set(feature_list), key=_feature_sort_key)
    grouped_targets: List[Tuple[FeatureNode, List[FeatureNode]]] = []
    for i, source_node in enumerate(ordered_features):
        target_nodes = [
            target_node
            for target_node in ordered_features[i + 1 :]
            if _is_valid_causal_edge(source_node, target_node)
        ]
        if target_nodes:
            grouped_targets.append((source_node, target_nodes))
    return grouped_targets


def _build_target_groups_between_feature_lists(
    source_feature_list: List[FeatureNode],
    target_feature_list: List[FeatureNode],
) -> List[Tuple[FeatureNode, List[FeatureNode]]]:
    ordered_sources = sorted(set(source_feature_list), key=_feature_sort_key)
    ordered_targets = sorted(set(target_feature_list), key=_feature_sort_key)
    grouped_targets: List[Tuple[FeatureNode, List[FeatureNode]]] = []
    for source_node in ordered_sources:
        target_nodes = [
            target_node
            for target_node in ordered_targets
            if _is_valid_causal_edge(source_node, target_node)
        ]
        if target_nodes:
            grouped_targets.append((source_node, target_nodes))
    return grouped_targets


def _batch_analyze_grouped_node_interactions(
    precomputed_data: Dict[str, Any],
    grouped_targets: List[Tuple[FeatureNode, List[FeatureNode]]],
    steering_scale: float = 0.0,
    desc: str = "Analyzing feature interactions",
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for source_node, target_nodes in tqdm(grouped_targets, desc=desc):
        try:
            result = analyze_node_activation_impact(
                steering_nodes=[source_node],
                target_nodes=target_nodes,
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
            result["target_node_inputs"] = list(target_nodes)
            if len(target_nodes) == 1:
                result["target_node_input"] = target_nodes[0]
            results.append(result)
        except Exception as e:
            print(f"Error processing {source_node} -> {len(target_nodes)} targets: {e}")
            continue
    return results


def batch_analyze_feature_list_interactions(
    precomputed_data: Dict[str, Any],
    feature_list: List[FeatureNode],
    steering_scale: float = 0.0,
) -> List[Dict[str, Any]]:
    grouped_targets = _build_target_groups_within_feature_list(feature_list)
    return _batch_analyze_grouped_node_interactions(
        precomputed_data,
        grouped_targets,
        steering_scale=steering_scale,
        desc="Analyzing feature interactions by source",
    )


def batch_analyze_feature_list_cross_interactions(
    precomputed_data: Dict[str, Any],
    source_feature_list: List[FeatureNode],
    target_feature_list: List[FeatureNode],
    steering_scale: float = 0.0,
) -> List[Dict[str, Any]]:
    grouped_targets = _build_target_groups_between_feature_lists(
        source_feature_list,
        target_feature_list,
    )
    return _batch_analyze_grouped_node_interactions(
        precomputed_data,
        grouped_targets,
        steering_scale=steering_scale,
        desc="Analyzing cross-feature interactions by source",
    )


def _iter_target_entries(
    result: Dict[str, Any],
) -> List[Tuple[Optional[FeatureNode], Dict[str, Any]]]:
    target_results = result.get("target_nodes")
    if isinstance(target_results, list) and target_results:
        target_node_inputs = result.get("target_node_inputs")
        if not isinstance(target_node_inputs, list):
            single_target = result.get("target_node_input")
            target_node_inputs = [single_target] if single_target is not None else []

        entries: List[Tuple[Optional[FeatureNode], Dict[str, Any]]] = []
        for idx, target_result in enumerate(target_results):
            target_node = (
                target_node_inputs[idx]
                if idx < len(target_node_inputs)
                else result.get("target_node_input")
            )
            entries.append((target_node, target_result))
        return entries

    target_node = result.get("target_node_input")
    if target_node is None:
        return []

    return [
        (
            target_node,
            {
                "target_node": result.get("target_node", ""),
                "original_activation": result.get("original_activation", 0.0),
                "modified_activation": result.get("modified_activation", 0.0),
                "activation_change": result.get("activation_change", 0.0),
            },
        )
    ]


def _process_interaction_results(
    results: List[Dict[str, Any]],
    steering_scale: float,
    output_prefix: Optional[str] = None,
    threshold: float = 0.1,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    processed_results: List[Dict[str, Any]] = []
    for result in results:
        source_node = result.get("source_node")
        if source_node is None:
            continue

        for target_node, target_result in _iter_target_entries(result):
            if target_node is None:
                continue

            original_activation = float(target_result.get("original_activation", 0.0))
            modified_activation = float(target_result.get("modified_activation", 0.0))
            activation_change = float(target_result.get("activation_change", 0.0))

            if original_activation != 0:
                reduction_ratio = activation_change / original_activation
            else:
                reduction_ratio = 0.0

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
                "target_node_str": target_result.get("target_node", ""),
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
        if results_df.empty:
            filtered_df = results_df
        else:
            filtered_df = results_df[results_df["reduction_ratio"] <= -threshold]
        csv_output_file = f"{output_prefix}_reduction_{threshold}.csv"
        filtered_df.to_csv(csv_output_file, index=False)

    return processed_results, json_output_file, csv_output_file


def compute_feature_list_interactions(
    feature_list: List[FeatureNode],
    precomputed_data: Dict[str, Any],
    steering_scale: float = 0.0,
    output_prefix: Optional[str] = None,
    threshold: float = 0.1,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    raw_results = batch_analyze_feature_list_interactions(
        precomputed_data,
        feature_list,
        steering_scale=steering_scale,
    )
    return _process_interaction_results(
        raw_results,
        steering_scale,
        output_prefix,
        threshold=threshold,
    )


def compute_cross_feature_list_interactions(
    source_feature_list: List[FeatureNode],
    target_feature_list: List[FeatureNode],
    precomputed_data: Dict[str, Any],
    steering_scale: float = 0.0,
    output_prefix: Optional[str] = None,
    threshold: float = 0.1,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    raw_results = batch_analyze_feature_list_cross_interactions(
        precomputed_data,
        source_feature_list,
        target_feature_list,
        steering_scale=steering_scale,
    )
    return _process_interaction_results(
        raw_results,
        steering_scale,
        output_prefix,
        threshold=threshold,
    )


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
    feature_list = build_feature_list_from_df(unique_features)

    precomputed_data = precompute_activations_and_weights(
        fen, model, transcoders, lorsas
    )

    processed_results, json_file, csv_file = compute_feature_list_interactions(
        feature_list,
        precomputed_data,
        steering_scale=steering_scale,
        output_prefix=output_prefix,
        threshold=0.1,
    )

    return processed_results, json_file, csv_file


def _mean_infl_from_results(results: List[Dict[str, Any]]) -> float:
    infls = [float(r.get("reduction_ratio", 0.0)) for r in results]
    if not infls:
        return float("nan")
    return float(sum(infls) / len(infls))


def compute_self_conn_for_csv(
    csv_file_path: str,
    precomputed_data: Dict[str, Any],
    steering_scale: float = 0.0,
    output_prefix: Optional[str] = None,
    threshold: float = 0.1,
) -> Tuple[float, Optional[str], Optional[str]]:
    df = pd.read_csv(csv_file_path)
    unique_features = df[
        ["position_name", "position_idx", "layer", "feature_id", "feature_type"]
    ].drop_duplicates()
    feature_list = build_feature_list_from_df(unique_features)

    processed_results, json_file, csv_file = compute_feature_list_interactions(
        feature_list,
        precomputed_data,
        steering_scale=steering_scale,
        output_prefix=output_prefix,
        threshold=threshold,
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
    source_feature_list = build_feature_list_from_df(unique_src)
    target_feature_list = build_feature_list_from_df(unique_tgt)

    processed_results, json_file, csv_file = compute_cross_feature_list_interactions(
        source_feature_list,
        target_feature_list,
        precomputed_data,
        steering_scale=steering_scale,
        output_prefix=output_prefix,
        threshold=0.1,
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
