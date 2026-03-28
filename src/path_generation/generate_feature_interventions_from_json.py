from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

JSON_FILENAME = "infl_all_feature.json"
FIXED_K = 1
TOP_N_PER_MOVE = 400
FOLDERS = [
    Path(
        "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/exp/53ICMLnew/2generate_significant_features/output/K2/pmin0.0_pmax0.2_K2"
    ),
    Path(
        "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/exp/53ICMLnew/2generate_significant_features/output/K2/pmin0.0_pmax1.0_K2"
    ),
    Path(
        "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/exp/53ICMLnew/2generate_significant_features/output/K2/pmin0.8_pmax1.0_K2"
    ),
    Path(
        "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/exp/53ICMLnew/2generate_significant_features/output/K2samestart/samestart_K2"
    ),
]

CSV_COLUMNS = [
    "position_name",
    "position_idx",
    "layer",
    "feature_id",
    "feature_type",
    "activation_value",
    "steering_scale",
    "move",
    "prob_diff",
    "original_prob",
    "modified_prob",
    "fen",
]

def load_json(json_path: Path) -> dict[str, Any]:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object in: {json_path}")
    return payload


def iter_position_entries(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    for position_name, analysis_result in payload.items():
        if not isinstance(analysis_result, dict):
            continue
        if "results" not in analysis_result:
            continue
        entries.append((position_name, analysis_result))
    return entries


def infer_move_order(position_entries: list[tuple[str, dict[str, Any]]]) -> list[str]:
    ordered_moves: list[str] = []
    seen_moves: set[str] = set()

    for _, analysis_result in position_entries:
        moves_tracing = analysis_result.get("moves_tracing")
        if not isinstance(moves_tracing, dict):
            continue
        for move_uci in moves_tracing:
            if isinstance(move_uci, str) and move_uci not in seen_moves:
                ordered_moves.append(move_uci)
                seen_moves.add(move_uci)

    if ordered_moves:
        return ordered_moves

    for _, analysis_result in position_entries:
        results = analysis_result.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            move_probabilities = result.get("move_probabilities")
            if not isinstance(move_probabilities, dict):
                continue
            for move_uci in move_probabilities:
                if isinstance(move_uci, str) and move_uci not in seen_moves:
                    ordered_moves.append(move_uci)
                    seen_moves.add(move_uci)

    return ordered_moves


def infer_position_idx(position_name: str) -> int:
    if "_" not in position_name:
        return 0
    suffix = position_name.rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return 0


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_top_rows_for_move(
    *,
    position_entries: list[tuple[str, dict[str, Any]]],
    move_uci: str,
    n: int,
    fen: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for fallback_position_name, analysis_result in position_entries:
        results = analysis_result.get("results")
        if not isinstance(results, list):
            continue

        for result in results:
            if not isinstance(result, dict):
                continue

            move_probabilities = result.get("move_probabilities")
            if not isinstance(move_probabilities, dict):
                continue

            move_probs = move_probabilities.get(move_uci)
            if not isinstance(move_probs, dict):
                continue

            prob_diff = as_float(move_probs.get("prob_diff"))
            if prob_diff is None:
                continue

            position_name = result.get("position_name", fallback_position_name)
            position_idx = as_int(
                result.get("position_idx"),
                infer_position_idx(str(position_name)),
            )

            rows.append(
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
                    "fen": fen,
                }
            )

    rows.sort(
        key=lambda row: (
            row["prob_diff"],
            row["position_idx"],
            row["layer"],
            row["feature_id"],
        )
    )
    return rows[:n]


def write_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_fen(output_dir: Path, fen: str) -> None:
    if not fen:
        return
    fen_path = output_dir / "fen.txt"
    fen_path.write_text(fen, encoding="utf-8")
    print(f"Saved fen to: {fen_path}")


def generate_csvs_for_json(*, json_path: Path, output_dir: Path, n: int) -> int:
    payload = load_json(json_path)
    position_entries = iter_position_entries(payload)
    if not position_entries:
        raise ValueError(f"No position entries with results found in: {json_path}")

    fen = str(payload.get("fen", ""))
    move_order = infer_move_order(position_entries)
    if not move_order:
        raise ValueError(f"No moves found in: {json_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    maybe_write_fen(output_dir, fen)

    print(f"Loaded {len(position_entries)} positions from: {json_path}")
    print(f"Found {len(move_order)} moves in JSON")

    generated_files = 0
    for move_uci in move_order:
        rows = collect_top_rows_for_move(
            position_entries=position_entries,
            move_uci=move_uci,
            n=n,
            fen=fen,
        )
        if not rows:
            print(f"Skipping {move_uci}: no rows with valid prob_diff found.")
            continue

        csv_path = output_dir / f"{move_uci}_top{n}_features.csv"
        write_csv(csv_path, rows)
        generated_files += 1
        print(f"Saved {len(rows)} rows to: {csv_path}")

    if generated_files == 0:
        raise RuntimeError(f"No CSV files were generated from: {json_path}")

    return generated_files


def discover_batch_jobs(folder: Path) -> list[tuple[Path, Path]]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a directory: {folder}")

    jobs: list[tuple[Path, Path]] = []

    root_json_path = folder / JSON_FILENAME
    if root_json_path.is_file():
        jobs.append((root_json_path, folder))

    for child in sorted(folder.iterdir()):
        if not child.is_dir():
            continue
        json_path = child / JSON_FILENAME
        if json_path.is_file():
            jobs.append((json_path, child))

    return jobs


def run_batch_generation(*, folders: list[Path], k: int, n: int) -> None:
    print(f"Running batch generation with K={k}, top-{n} features")
    print(f"Scanning {len(folders)} top-level folders")

    total_json_files = 0
    total_generated_csvs = 0
    failures: list[str] = []

    for folder in folders:
        resolved_folder = folder.resolve()
        jobs = discover_batch_jobs(resolved_folder)
        if not jobs:
            failures.append(f"{resolved_folder}: no {JSON_FILENAME} files found")
            print(f"Skipping {resolved_folder}: no {JSON_FILENAME} files found.")
            continue

        total_json_files += len(jobs)
        print(f"Found {len(jobs)} JSON files in: {resolved_folder}")

        for json_path, output_dir in jobs:
            try:
                total_generated_csvs += generate_csvs_for_json(
                    json_path=json_path,
                    output_dir=output_dir,
                    n=n,
                )
            except Exception as exc:
                failures.append(f"{json_path}: {exc}")
                print(f"Failed {json_path}: {exc}")

    if total_generated_csvs == 0:
        raise RuntimeError("No CSV files were generated from the provided folders.")

    print(
        f"Processed {total_json_files} JSON files and generated "
        f"{total_generated_csvs} CSV files in total."
    )

    if failures:
        failure_preview = "\n".join(failures[:10])
        if len(failures) > 10:
            failure_preview += f"\n... and {len(failures) - 10} more"
        raise RuntimeError(
            "Batch generation completed with failures:\n"
            f"{failure_preview}"
        )


def main() -> None:
    run_batch_generation(
        folders=FOLDERS,
        k=FIXED_K,
        n=TOP_N_PER_MOVE,
    )


if __name__ == "__main__":
    main()
