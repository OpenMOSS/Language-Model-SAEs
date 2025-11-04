import json
from pathlib import Path
from typing import Any

from lm_saes.config import MongoDBConfig
from lm_saes.database import FeatureRecord, MongoClient


def augment_graph_with_feature_data(
    json_path: str | Path,
    mongo_cfg: MongoDBConfig,
    sae_series: str,
) -> str:
    """Augment a circuit graph JSON by injecting logits and interpretation for feature nodes.

    Args:
            json_path: Path to the input graph JSON file. Must match the format produced by create_graph_files.
            mongo_cfg: MongoDB configuration used to instantiate a client.
            sae_series: SAE series identifier used to query feature records.

    Returns:
            Path to the newly written augmented JSON file as a string.

    Behavior:
            - Parse each node's `node_id` of the form "{layer_idx}_{feature_idx}_{ctx_id}".
            - Determine `sae_name` based on `feature_type`:
              - feature_type == "lorsa": `sae_name = metadata.lorsa_analysis_name.format(layer_idx // 2)`
              - feature_type == "cross layer transcoder": `sae_name = metadata.clt_analysis_name.format((layer_idx - 1) // 2)`
            - Fetch `FeatureRecord` via `MongoClient.get_feature(sae_name, sae_series, feature_idx)`.
            - If found, write `logits` and `interpretation` back into the node.
            - Write a copy of the JSON alongside the original with suffix `_augmented.json`.
    """
    input_path = Path(json_path)
    with input_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    metadata: dict[str, Any] = data.get("metadata", {})
    lorsa_analysis_name: str | None = metadata.get("lorsa_analysis_name")
    clt_analysis_name: str | None = metadata.get("clt_analysis_name")

    # Instantiate client
    client = MongoClient(mongo_cfg)

    nodes: list[dict[str, Any]] = data.get("nodes", [])
    for node in nodes:
        feature_type = node.get("feature_type")
        if feature_type not in ("lorsa", "cross layer transcoder"):
            continue

        node_id = node.get("node_id")
        if not isinstance(node_id, str):
            continue

        parts = node_id.split("_")
        if len(parts) != 3:
            # Unexpected format; skip silently
            continue
        try:
            layer_idx = int(parts[0])
            feature_idx = int(parts[1])
        except ValueError:
            continue

        # Resolve SAE name based on feature type
        if feature_type == "lorsa":
            if not lorsa_analysis_name:
                continue
            resolved_sae_name = lorsa_analysis_name.format(layer_idx // 2)
        elif feature_type == "cross layer transcoder":
            if not clt_analysis_name:
                continue
            resolved_sae_name = clt_analysis_name.format((layer_idx - 1) // 2)
        else:
            continue

        # Query feature record
        feature: FeatureRecord | None = client.get_feature(resolved_sae_name, sae_series, feature_idx)
        if feature is None:
            continue

        # Inject fields if present
        if feature.logits is not None:
            node["logits"] = feature.logits
        if feature.interpretation is not None:
            node["interpretation"] = feature.interpretation

    # Write augmented copy next to original
    output_path = input_path.with_name(f"{input_path.stem}_augmented.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return str(output_path)
