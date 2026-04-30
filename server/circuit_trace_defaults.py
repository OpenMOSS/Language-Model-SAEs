from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "ui" / "src" / "config" / "circuit-trace-defaults.json"

with _DEFAULTS_PATH.open("r", encoding="utf-8") as handle:
    _RAW_DEFAULTS: dict[str, Any] = json.load(handle)

CIRCUIT_TRACE_DEFAULTS = {
    "max_feature_nodes": int(_RAW_DEFAULTS["max_feature_nodes"]),
    "node_threshold": float(_RAW_DEFAULTS["node_threshold"]),
    "edge_threshold": float(_RAW_DEFAULTS["edge_threshold"]),
    "save_activation_info": bool(_RAW_DEFAULTS["save_activation_info"]),
}

DEFAULT_MAX_FEATURE_NODES = CIRCUIT_TRACE_DEFAULTS["max_feature_nodes"]
DEFAULT_NODE_THRESHOLD = CIRCUIT_TRACE_DEFAULTS["node_threshold"]
DEFAULT_EDGE_THRESHOLD = CIRCUIT_TRACE_DEFAULTS["edge_threshold"]
DEFAULT_SAVE_ACTIVATION_INFO = CIRCUIT_TRACE_DEFAULTS["save_activation_info"]
