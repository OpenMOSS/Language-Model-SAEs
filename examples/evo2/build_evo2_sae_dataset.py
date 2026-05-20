from __future__ import annotations

import gzip
import json
import shutil
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import DatasetInfo, Features, Value
from datasets.arrow_writer import ArrowWriter


DATA_ROOT = Path("/inspire/hdd/project/reasoning/public/activations/evo2_7b/opengenome2/json")
OUTPUT_ROOT = Path("/inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Evo2/evo2_gtdb_v220_stitched_sae")

DATASET_PRESET = "gtdb-v220-stitched-sae"
DATASET_NAME = "evo2_gtdb_v220_stitched_sae"
BUILDER_NAME = "evo2_gtdb_v220_stitched_sae"
CONFIG_NAME = "evo2_gtdb_v220_stitched_sae"
DESCRIPTION = (
    "Filtered OpenGenome2 dataset for Evo2 SAE training using only the "
    "midtraining_specific/gtdb_v220_stitched source. "
    "By default it uses train chunks only and does not re-trim sequences."
)
HOMEPAGE = ""
LICENSE = ""
CITATION = ""

INCLUDE_TEST_SPLIT = False
APPLY_PAPER_BP_BUDGETS = False
PROGRESS_ROW_INTERVAL = 100_000


@dataclass(frozen=True)
class SourceRule:
    alias: str
    paper_role: str
    domain: str
    rel_dir: str


PRESETS: dict[str, list[SourceRule]] = {
    "gtdb-v220-stitched-sae": [
        SourceRule(
            alias="gtdb_v220_stitched",
            paper_role="GTDB v220 stitched prokaryotic genomes",
            domain="prokaryote",
            rel_dir="midtraining_specific/gtdb_v220_stitched",
        )
    ],
}


PAPER_BP_BUDGETS = {
    "gtdb-v220-stitched-sae": {
        "gtdb_v220_stitched": None,
    },
}


FEATURES = Features(
    {
        "text": Value("string"),
        "source_dataset": Value("string"),
        "paper_role": Value("string"),
        "domain": Value("string"),
        "split": Value("string"),
        "source_file": Value("string"),
        "row_in_file": Value("int64"),
        "sequence_length": Value("int64"),
    }
)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def detect_split(path: Path) -> str:
    name = path.name.lower()
    if "train" in name:
        return "train"
    if "test" in name:
        return "test"
    if "valid" in name or "val" in name:
        return "validation"
    return "unknown"


def extract_text(record: dict[str, Any]) -> str | None:
    text = record.get("text")
    if isinstance(text, str):
        return text

    nested = record.get("record")
    if isinstance(nested, dict):
        nested_text = nested.get("text")
        if isinstance(nested_text, str):
            return nested_text
    return None


def collect_files(rule: SourceRule, include_test: bool) -> list[Path]:
    source_dir = DATA_ROOT / rule.rel_dir
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    files = sorted(source_dir.rglob("*.jsonl.gz"))
    if not include_test:
        files = [path for path in files if "test" not in path.name.lower()]
    return files


def get_bp_budgets(dataset_preset: str, apply_paper_bp_budgets: bool) -> dict[str, int | None]:
    if not apply_paper_bp_budgets:
        return {rule.alias: None for rule in PRESETS[dataset_preset]}
    return {rule.alias: PAPER_BP_BUDGETS[dataset_preset].get(rule.alias) for rule in PRESETS[dataset_preset]}


def feature_schema_json() -> dict[str, Any]:
    return FEATURES.to_dict()


def dataset_config_json(output_root: Path) -> dict[str, Any]:
    return {
        "dataset_name_or_path": str(output_root),
        "is_dataset_on_disk": True,
        "builder_name": BUILDER_NAME,
        "citation": CITATION,
        "config_name": CONFIG_NAME,
        "dataset_name": DATASET_NAME,
        "description": DESCRIPTION,
        "features": feature_schema_json(),
        "homepage": HOMEPAGE,
        "license": LICENSE,
    }


def dataset_info() -> DatasetInfo:
    return DatasetInfo(
        builder_name=BUILDER_NAME,
        citation=CITATION,
        config_name=CONFIG_NAME,
        dataset_name=DATASET_NAME,
        description=DESCRIPTION,
        features=FEATURES,
        homepage=HOMEPAGE,
        license=LICENSE,
    )


def build_selected_sources(dataset_preset: str, include_test: bool, apply_paper_bp_budgets: bool) -> list[dict[str, Any]]:
    budgets = get_bp_budgets(dataset_preset, apply_paper_bp_budgets)
    selected_sources: list[dict[str, Any]] = []
    for rule in PRESETS[dataset_preset]:
        files = collect_files(rule, include_test=include_test)
        log(
            f"selected source `{rule.alias}`: files={len(files)} "
            f"dir={DATA_ROOT / rule.rel_dir} budget={budgets.get(rule.alias)}"
        )
        selected_sources.append(
            {
                "alias": rule.alias,
                "paper_role": rule.paper_role,
                "domain": rule.domain,
                "source_dir": str(DATA_ROOT / rule.rel_dir),
                "files": files,
                "bp_budget": budgets.get(rule.alias),
            }
        )
    return selected_sources


def init_summary(selected_sources: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_rows": 0,
        "total_sequence_length": 0,
        "skipped_missing_text": 0,
        "per_source": {
            source["alias"]: {
                "rows": 0,
                "sequence_length": 0,
                "files_selected": len(source["files"]),
                "files_touched": set(),
                "paper_role": source["paper_role"],
                "domain": source["domain"],
            }
            for source in selected_sources
        },
    }


def materialize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_rows": summary["total_rows"],
        "total_sequence_length": summary["total_sequence_length"],
        "skipped_missing_text": summary["skipped_missing_text"],
        "per_source": {
            alias: {
                **{k: v for k, v in stats.items() if k != "files_touched"},
                "files_touched": sorted(stats["files_touched"]),
            }
            for alias, stats in summary["per_source"].items()
        },
    }


def iter_selected_rows(
    selected_sources: list[dict[str, Any]],
    max_rows: int | None,
    summary: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    total_rows = 0

    for source in selected_sources:
        log(
            f"processing source `{source['alias']}` with {len(source['files'])} files "
            f"(budget={source['bp_budget']})"
        )
        budget = source["bp_budget"]
        consumed_bp = 0

        for file_idx, file_path in enumerate(source["files"], start=1):
            log(
                f"opening file {file_idx}/{len(source['files'])} for `{source['alias']}`: {file_path.name}"
            )
            file_rows = 0
            file_bp = 0
            with gzip.open(file_path, "rt", encoding="utf-8") as handle:
                for row_idx, line in enumerate(handle):
                    if max_rows is not None and total_rows >= max_rows:
                        log(f"reached max_rows={max_rows}; stopping row iteration")
                        return

                    record = json.loads(line)
                    text = extract_text(record)
                    if text is None:
                        summary["skipped_missing_text"] += 1
                        continue

                    sequence_length = len(text)
                    if budget is not None and consumed_bp + sequence_length > budget:
                        break

                    consumed_bp += sequence_length
                    total_rows += 1
                    file_rows += 1
                    file_bp += sequence_length

                    source_stats = summary["per_source"][source["alias"]]
                    source_stats["rows"] += 1
                    source_stats["sequence_length"] += sequence_length
                    source_stats["files_touched"].add(str(file_path))
                    summary["total_rows"] += 1
                    summary["total_sequence_length"] += sequence_length

                    if file_rows % PROGRESS_ROW_INTERVAL == 0:
                        log(
                            f"progress `{source['alias']}` {file_path.name}: "
                            f"file_rows={file_rows} file_bp={file_bp} "
                            f"total_rows={summary['total_rows']} total_bp={summary['total_sequence_length']}"
                        )

                    yield {
                        "text": text,
                        "source_dataset": source["alias"],
                        "paper_role": source["paper_role"],
                        "domain": source["domain"],
                        "split": detect_split(file_path),
                        "source_file": str(file_path),
                        "row_in_file": row_idx,
                        "sequence_length": sequence_length,
                    }

            log(
                f"finished file {file_idx}/{len(source['files'])} for `{source['alias']}`: "
                f"rows={file_rows} bp={file_bp} "
                f"source_total_rows={summary['per_source'][source['alias']]['rows']} "
                f"source_total_bp={summary['per_source'][source['alias']]['sequence_length']}"
            )
            if budget is not None and consumed_bp >= budget:
                log(
                    f"reached source budget for `{source['alias']}`: "
                    f"consumed_bp={consumed_bp} budget={budget}"
                )
                break


def write_sidecar_metadata(
    output_root: Path,
    selected_sources: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    config_path = output_root / "dataset_config.json"
    manifest_path = output_root / "selection_manifest.json"

    config_path.write_text(json.dumps(dataset_config_json(output_root), indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_preset": DATASET_PRESET,
                "data_root": str(DATA_ROOT),
                "include_test_split": INCLUDE_TEST_SPLIT,
                "apply_paper_bp_budgets": APPLY_PAPER_BP_BUDGETS,
                "sources": [
                    {
                        "alias": source["alias"],
                        "paper_role": source["paper_role"],
                        "domain": source["domain"],
                        "source_dir": source["source_dir"],
                        "bp_budget": source["bp_budget"],
                        "file_count": len(source["files"]),
                        "files": [str(path) for path in source["files"]],
                    }
                    for source in selected_sources
                ],
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_hf_dataset_metadata(output_root: Path) -> None:
    dataset_info().write_to_directory(str(output_root), pretty_print=True)
    state = {
        "_data_files": [{"filename": "data-00000-of-00001.arrow"}],
        "_fingerprint": f"{DATASET_NAME}_manual_arrow",
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": None,
    }
    (output_root / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def build_and_save_dataset(output_root: Path = OUTPUT_ROOT, max_rows: int | None = None) -> dict[str, Any]:
    log(
        f"starting dataset build: preset={DATASET_PRESET} output={output_root} "
        f"include_test={INCLUDE_TEST_SPLIT} paper_budgets={APPLY_PAPER_BP_BUDGETS}"
    )
    selected_sources = build_selected_sources(
        dataset_preset=DATASET_PRESET,
        include_test=INCLUDE_TEST_SPLIT,
        apply_paper_bp_budgets=APPLY_PAPER_BP_BUDGETS,
    )
    summary = init_summary(selected_sources)
    build_dir = output_root.parent / f".{output_root.name}_build"
    arrow_path = build_dir / "data-00000-of-00001.arrow"

    if build_dir.exists():
        log(f"removing stale build dir: {build_dir}")
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    log(f"writing temporary Arrow data to: {arrow_path}")

    writer = ArrowWriter(features=FEATURES, path=str(arrow_path))
    for row in iter_selected_rows(
        selected_sources=selected_sources,
        max_rows=max_rows,
        summary=summary,
    ):
        writer.write(row)
    writer.finalize()
    log(
        f"finished Arrow writing: rows={summary['total_rows']} "
        f"bp={summary['total_sequence_length']}"
    )

    final_summary = materialize_summary(summary)
    log("writing HuggingFace dataset metadata")
    write_hf_dataset_metadata(build_dir)
    write_sidecar_metadata(output_root=build_dir, selected_sources=selected_sources, summary=final_summary)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        log(f"removing existing output dataset: {output_root}")
        shutil.rmtree(output_root)
    log(f"moving prepared HuggingFace dataset to final path: {output_root}")
    shutil.move(str(build_dir), str(output_root))
    log(f"wrote HuggingFace dataset and metadata under: {output_root}")
    return final_summary


def main() -> None:
    summary = build_and_save_dataset()
    print("=" * 80)
    print(f"saved dataset to: {OUTPUT_ROOT}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
