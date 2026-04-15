from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from lm_saes import LowRankSparseAttention, SparseAutoEncoder
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.chess_utils import get_move_from_policy_output_with_prob
from src.feature_and_steering import analyze_position_features_comprehensive
from src.path_evaluation.feature_infl import (
    build_feature_list_from_df,
    compute_cross_feature_list_interactions,
    compute_feature_list_interactions,
    precompute_activations_and_weights,
)

DEFAULT_MODEL_NAME = "lc0/BT4-1024x15x32h"
DEFAULT_DEVICE = "cuda"
DEFAULT_TC_ROOT = "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/tc/k_30_e_16"
DEFAULT_LORSA_ROOT = "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/lorsa/k_30_e_16"

INTERACTION_COLUMNS = [
    "source_layer",
    "source_pos",
    "source_feature",
    "source_type",
    "target_layer",
    "target_pos",
    "target_feature",
    "target_type",
    "original_activation",
    "modified_activation",
    "activation_change",
    "steering_scale",
    "reduction_ratio",
    "target_node_str",
    "steering_nodes_count",
]

_MODEL_BUNDLE_CACHE: dict[
    tuple[str, str, str, str],
    tuple[
        HookedTransformer,
        dict[int, SparseAutoEncoder],
        list[LowRankSparseAttention],
    ],
] = {}


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def load_model_bundle(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    tc_root: str | Path = DEFAULT_TC_ROOT,
    lorsa_root: str | Path = DEFAULT_LORSA_ROOT,
) -> tuple[
    HookedTransformer,
    dict[int, SparseAutoEncoder],
    list[LowRankSparseAttention],
]:
    resolved_device = resolve_device(device)
    tc_root = Path(tc_root).resolve()
    lorsa_root = Path(lorsa_root).resolve()
    cache_key = (
        model_name,
        resolved_device,
        str(tc_root),
        str(lorsa_root),
    )

    cached_bundle = _MODEL_BUNDLE_CACHE.get(cache_key)
    if cached_bundle is not None:
        return cached_bundle

    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        dtype=torch.float32,
    ).eval()
    if resolved_device != "cpu":
        model = model.to(resolved_device)

    transcoders = {
        layer: SparseAutoEncoder.from_pretrained(
            str(tc_root / f"L{layer}"),
            dtype=torch.float32,
            device=resolved_device,
        )
        for layer in range(15)
    }
    lorsas = [
        LowRankSparseAttention.from_pretrained(
            str(lorsa_root / f"L{layer}"),
            dtype=torch.float32,
            device=resolved_device,
        )
        for layer in range(15)
    ]

    bundle = (model, transcoders, lorsas)
    _MODEL_BUNDLE_CACHE[cache_key] = bundle
    return bundle


def sanitize_fen_for_path(fen: str, max_length: int = 100) -> str:
    safe_fen = fen.replace("/", "_").replace(" ", "_").replace("-", "_").replace(":", "_")
    if len(safe_fen) > max_length:
        return safe_fen[:max_length]
    return safe_fen


def build_fen_output_dir(output_dir: str | Path, fen: str) -> Path:
    return Path(output_dir).resolve() / f"fen_{sanitize_fen_for_path(fen)}"


def get_top_k_moves(
    fen: str,
    model: HookedTransformer,
    top_k_moves: int,
) -> dict[str, float]:
    if top_k_moves < 1:
        raise ValueError("top_k_moves must be at least 1")

    output, _ = model.run_with_cache(fen, prepend_bos=False)
    policy_output = output[0]
    legal_moves_with_probs = get_move_from_policy_output_with_prob(
        policy_output,
        fen,
        return_list=True,
    )
    if not legal_moves_with_probs:
        return {}

    sorted_moves = sorted(
        legal_moves_with_probs,
        key=lambda item: item[2],
        reverse=True,
    )[:top_k_moves]
    return {move_uci: prob for move_uci, _, prob in sorted_moves}


def _item_if_possible(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _to_serializable_analysis_payload(
    all_results: dict[str, Any],
    *,
    fen: str,
) -> dict[str, Any]:
    serializable_result = dict(all_results)

    for analysis_result in serializable_result.values():
        if not isinstance(analysis_result, dict):
            continue
        results = analysis_result.get("results")
        if not isinstance(results, list):
            continue

        for result in results:
            if not isinstance(result, dict):
                continue

            if "activation_value" in result:
                result["activation_value"] = _item_if_possible(result["activation_value"])

            for key in ("original_value", "modified_value", "value_diff", "steering_scale"):
                if key in result:
                    result[key] = _item_if_possible(result[key])

            move_probabilities = result.get("move_probabilities")
            if not isinstance(move_probabilities, dict):
                continue

            for prob_info in move_probabilities.values():
                if not isinstance(prob_info, dict):
                    continue
                for prob_key in ("original_prob", "modified_prob", "prob_diff"):
                    if prob_key in prob_info:
                        prob_info[prob_key] = _item_if_possible(prob_info[prob_key])

    serializable_result["fen"] = fen
    return serializable_result


def collect_move_feature_analysis(
    *,
    fen: str,
    model: HookedTransformer,
    transcoders: dict[int, SparseAutoEncoder],
    lorsas: list[LowRankSparseAttention],
    move_probabilities: dict[str, float],
    steering_factor: float = 0.0,
    activation_threshold: float = 0.0,
    max_features_per_type: Optional[int] = None,
    max_steering_features: Optional[int] = None,
) -> dict[str, Any]:
    if not move_probabilities:
        raise ValueError("move_probabilities cannot be empty")

    feature_types = ["transcoder", "lorsa"]
    position_map = {f"pos_{index}": index for index in range(64)}
    all_results: dict[str, Any] = {}

    for position_idx in tqdm(range(64), desc="Analyzing positions"):
        position_name = f"pos_{position_idx}"
        analysis_result = analyze_position_features_comprehensive(
            pos_dict=position_map,
            position_name=position_name,
            model=model,
            transcoders=transcoders,
            lorsas=lorsas,
            fen=fen,
            moves_tracing=dict(move_probabilities),
            feature_types=feature_types,
            steering_scale=steering_factor,
            activation_threshold=activation_threshold,
            max_features_per_type=max_features_per_type,
            max_steering_features=max_steering_features,
        )
        analysis_result["moves_tracing"] = dict(move_probabilities)
        all_results[position_name] = analysis_result

    return _to_serializable_analysis_payload(all_results, fen=fen)


def build_top_feature_tables(
    *,
    analysis_payload: dict[str, Any],
    move_probabilities: dict[str, float],
    n_features: int,
) -> dict[str, pd.DataFrame]:
    if n_features < 1:
        raise ValueError("n_features must be at least 1")

    feature_tables: dict[str, pd.DataFrame] = {}

    for move_uci in move_probabilities:
        feature_rows: list[dict[str, Any]] = []

        for position_name, analysis_result in analysis_payload.items():
            if not isinstance(analysis_result, dict):
                continue

            results = analysis_result.get("results")
            if not isinstance(results, list):
                continue

            for result in results:
                if not isinstance(result, dict):
                    continue

                move_probability_dict = result.get("move_probabilities")
                if not isinstance(move_probability_dict, dict):
                    continue

                move_probs = move_probability_dict.get(move_uci)
                if not isinstance(move_probs, dict):
                    continue

                prob_diff = move_probs.get("prob_diff")
                if prob_diff is None:
                    continue

                position_idx = 0
                if "_" in str(position_name):
                    try:
                        position_idx = int(str(position_name).split("_")[1])
                    except ValueError:
                        position_idx = 0

                feature_rows.append(
                    {
                        "position_name": position_name,
                        "position_idx": position_idx,
                        "layer": result.get("layer", 0),
                        "feature_id": result.get("feature_id", 0),
                        "feature_type": result.get("feature_type", "unknown"),
                        "activation_value": result.get("activation_value", 0.0),
                        "steering_scale": result.get("steering_scale", 0.0),
                        "move": move_uci,
                        "prob_diff": prob_diff,
                        "original_prob": move_probs.get("original_prob", 0.0),
                        "modified_prob": move_probs.get("modified_prob", 0.0),
                    }
                )

        if not feature_rows:
            continue

        feature_df = pd.DataFrame(feature_rows)
        feature_tables[move_uci] = feature_df.nsmallest(n_features, "prob_diff").reset_index(drop=True)

    return feature_tables


def _empty_interaction_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=INTERACTION_COLUMNS)


def _unique_feature_frame(feature_table: pd.DataFrame) -> pd.DataFrame:
    return feature_table[
        ["position_name", "position_idx", "layer", "feature_id", "feature_type"]
    ].drop_duplicates()


def _interaction_results_to_frame(
    processed_results: list[dict[str, Any]],
    *,
    reduction_ratio: float,
) -> pd.DataFrame:
    if not processed_results:
        return _empty_interaction_frame()

    results_df = pd.DataFrame(processed_results)
    if results_df.empty:
        return _empty_interaction_frame()

    filtered_df = results_df[results_df["reduction_ratio"] <= -reduction_ratio]
    if filtered_df.empty:
        return _empty_interaction_frame()

    return filtered_df.reset_index(drop=True)


def compute_self_interaction_frame(
    *,
    feature_table: pd.DataFrame,
    precomputed_data: dict[str, Any],
    steering_factor: float = 0.0,
    reduction_ratio: float = 0.1,
) -> pd.DataFrame:
    unique_features = _unique_feature_frame(feature_table)
    feature_list = build_feature_list_from_df(unique_features)
    if not feature_list:
        return _empty_interaction_frame()

    processed_results, _, _ = compute_feature_list_interactions(
        feature_list,
        precomputed_data,
        steering_scale=steering_factor,
        output_prefix=None,
        threshold=reduction_ratio,
    )
    return _interaction_results_to_frame(
        processed_results,
        reduction_ratio=reduction_ratio,
    )


def compute_cross_interaction_frame(
    *,
    source_feature_table: pd.DataFrame,
    target_feature_table: pd.DataFrame,
    precomputed_data: dict[str, Any],
    steering_factor: float = 0.0,
    reduction_ratio: float = 0.1,
) -> pd.DataFrame:
    source_features = build_feature_list_from_df(_unique_feature_frame(source_feature_table))
    target_features = build_feature_list_from_df(_unique_feature_frame(target_feature_table))
    if not source_features or not target_features:
        return _empty_interaction_frame()

    processed_results, _, _ = compute_cross_feature_list_interactions(
        source_features,
        target_features,
        precomputed_data,
        steering_scale=steering_factor,
        output_prefix=None,
        threshold=reduction_ratio,
    )
    return _interaction_results_to_frame(
        processed_results,
        reduction_ratio=reduction_ratio,
    )


def _feature_selection_metadata(feature_table: pd.DataFrame) -> list[dict[str, Any]]:
    metadata_columns = [
        "position_name",
        "position_idx",
        "layer",
        "feature_id",
        "feature_type",
        "activation_value",
        "prob_diff",
    ]
    if feature_table.empty:
        return []

    unique_features = feature_table[metadata_columns].drop_duplicates(
        subset=["position_name", "position_idx", "layer", "feature_id", "feature_type"]
    )
    records = unique_features.to_dict(orient="records")
    normalized_records: list[dict[str, Any]] = []
    for record in records:
        normalized_records.append(
            {
                "position_name": str(record["position_name"]),
                "position_idx": int(record["position_idx"]),
                "layer": int(record["layer"]),
                "feature_id": int(record["feature_id"]),
                "feature_type": str(record["feature_type"]),
                "activation_value": float(record["activation_value"]),
                "prob_diff": float(record["prob_diff"]),
            }
        )
    return normalized_records


def write_interaction_csv_with_metadata(
    *,
    output_csv: str | Path,
    interaction_frame: pd.DataFrame,
    fen: str,
    source_move: str,
    target_move: str,
    source_feature_table: pd.DataFrame,
    target_feature_table: pd.DataFrame,
    top_k_moves: int,
    n_features: int,
    reduction_ratio: float,
    steering_factor: float,
) -> Path:
    output_csv = Path(output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "fen": fen,
        "source_move": source_move,
        "target_move": target_move,
        "top_k_moves": top_k_moves,
        "n_features": n_features,
        "reduction_ratio": reduction_ratio,
        "steering_factor": steering_factor,
        "source_features_json": _feature_selection_metadata(source_feature_table),
        "target_features_json": _feature_selection_metadata(target_feature_table),
    }

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        for key, value in metadata.items():
            serialized_value = (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else str(value)
            )
            handle.write(f"# {key}: {serialized_value}\n")
        interaction_frame.to_csv(handle, index=False, lineterminator="\n")

    return output_csv


def read_interaction_csv_with_metadata(
    input_csv: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    input_csv = Path(input_csv).resolve()
    metadata: dict[str, Any] = {}

    with input_csv.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("# "):
                break
            key, _, raw_value = line[2:].partition(": ")
            value = raw_value.rstrip("\n")
            try:
                metadata[key] = json.loads(value)
            except json.JSONDecodeError:
                metadata[key] = value

    frame = pd.read_csv(input_csv, comment="#")
    return metadata, frame


def generate_path_csvs(
    *,
    fen: str,
    output_dir: str | Path,
    top_k_moves: int = 1,
    n_features: int = 200,
    reduction_ratio: float = 0.1,
    steering_factor: float = 0.0,
    activation_threshold: float = 0.0,
    max_features_per_type: Optional[int] = None,
    max_steering_features: Optional[int] = None,
    device: str = DEFAULT_DEVICE,
    model_name: str = DEFAULT_MODEL_NAME,
    tc_root: str | Path = DEFAULT_TC_ROOT,
    lorsa_root: str | Path = DEFAULT_LORSA_ROOT,
    save_analysis_json: bool = False,
) -> dict[str, Any]:
    if reduction_ratio < 0:
        raise ValueError("reduction_ratio must be non-negative")

    model, transcoders, lorsas = load_model_bundle(
        model_name=model_name,
        device=device,
        tc_root=tc_root,
        lorsa_root=lorsa_root,
    )

    move_probabilities = get_top_k_moves(
        fen=fen,
        model=model,
        top_k_moves=top_k_moves,
    )
    if not move_probabilities:
        raise ValueError("No legal moves were found for the provided FEN")

    analysis_payload = collect_move_feature_analysis(
        fen=fen,
        model=model,
        transcoders=transcoders,
        lorsas=lorsas,
        move_probabilities=move_probabilities,
        steering_factor=steering_factor,
        activation_threshold=activation_threshold,
        max_features_per_type=max_features_per_type,
        max_steering_features=max_steering_features,
    )

    feature_tables = build_top_feature_tables(
        analysis_payload=analysis_payload,
        move_probabilities=move_probabilities,
        n_features=n_features,
    )
    if not feature_tables:
        raise RuntimeError("No top-feature tables could be built for the selected moves")

    fen_output_dir = build_fen_output_dir(output_dir, fen)
    fen_output_dir.mkdir(parents=True, exist_ok=True)
    (fen_output_dir / "fen.txt").write_text(fen, encoding="utf-8")

    analysis_json_path: Optional[Path] = None
    if save_analysis_json:
        analysis_json_path = fen_output_dir / "infl_all_feature.json"
        analysis_json_path.write_text(
            json.dumps(analysis_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    precomputed_data = precompute_activations_and_weights(
        fen,
        model,
        transcoders,
        lorsas,
    )

    generated_csvs: dict[str, Path] = {}
    ordered_moves = [move for move in move_probabilities if move in feature_tables]

    for move in ordered_moves:
        feature_table = feature_tables[move]
        interaction_frame = compute_self_interaction_frame(
            feature_table=feature_table,
            precomputed_data=precomputed_data,
            steering_factor=steering_factor,
            reduction_ratio=reduction_ratio,
        )
        output_csv = fen_output_dir / f"{move}_{move}_reduction_{reduction_ratio}.csv"
        generated_csvs[f"{move}_{move}"] = write_interaction_csv_with_metadata(
            output_csv=output_csv,
            interaction_frame=interaction_frame,
            fen=fen,
            source_move=move,
            target_move=move,
            source_feature_table=feature_table,
            target_feature_table=feature_table,
            top_k_moves=top_k_moves,
            n_features=n_features,
            reduction_ratio=reduction_ratio,
            steering_factor=steering_factor,
        )

    for source_move, target_move in combinations(ordered_moves, 2):
        source_feature_table = feature_tables[source_move]
        target_feature_table = feature_tables[target_move]
        interaction_frame = compute_cross_interaction_frame(
            source_feature_table=source_feature_table,
            target_feature_table=target_feature_table,
            precomputed_data=precomputed_data,
            steering_factor=steering_factor,
            reduction_ratio=reduction_ratio,
        )
        output_csv = fen_output_dir / f"{source_move}_{target_move}_reduction_{reduction_ratio}.csv"
        generated_csvs[f"{source_move}_{target_move}"] = write_interaction_csv_with_metadata(
            output_csv=output_csv,
            interaction_frame=interaction_frame,
            fen=fen,
            source_move=source_move,
            target_move=target_move,
            source_feature_table=source_feature_table,
            target_feature_table=target_feature_table,
            top_k_moves=top_k_moves,
            n_features=n_features,
            reduction_ratio=reduction_ratio,
            steering_factor=steering_factor,
        )

    return {
        "fen": fen,
        "fen_output_dir": fen_output_dir,
        "move_probabilities": move_probabilities,
        "selected_feature_counts": {
            move: len(_unique_feature_frame(feature_table))
            for move, feature_table in feature_tables.items()
        },
        "generated_csvs": generated_csvs,
        "analysis_json": analysis_json_path,
    }
