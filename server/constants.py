from __future__ import annotations
from typing import Dict, TypedDict


class SaeComboConfig(TypedDict):
    """Configuration for a single SAE combo."""

    id: str  # ID, e.g. "k_30_e_16"
    label: str  # frontend display text
    tc_base_path: str  # Transcoder SAE base path
    lorsa_base_path: str  # Lorsa SAE base path
    tc_sae_name_template: str  # Transcoder SAE name template (for building dictionary name)
    lorsa_sae_name_template: str  # Lorsa SAE name template (for building dictionary name)
    tc_analysis_name: str  # Transcoder analysis_name (saved to JSON metadata)
    lorsa_analysis_name: str  # Lorsa analysis_name (saved to JSON metadata)


# BT4 model name (Leela Chess Zero BT4 strength model)
BT4_MODEL_NAME: str = "lc0/BT4-1024x15x32h"

# ===== Available BT4 SAE combos (from exp/38mongoanalyses/combos.txt) =====

def _make_bt4_sae_combo(k: int, e: int) -> SaeComboConfig:
    combo_id = f"k_{k}_e_{e}"
    suffix = f"k{k}_e{e}"
    return {
        "id": combo_id,
        "label": combo_id,
        "tc_base_path": (
            "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
            f"rlin_projects/chess-SAEs-N/result_BT4/tc/{combo_id}"
        ),
        "lorsa_base_path": (
            "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
            f"rlin_projects/chess-SAEs-N/result_BT4/lorsa/{combo_id}"
        ),
        "tc_sae_name_template": f"BT4_tc_L{{layer}}M_{suffix}",
        "lorsa_sae_name_template": f"BT4_lorsa_L{{layer}}A_{suffix}",
        "tc_analysis_name": f"BT4_tc_{suffix}",
        "lorsa_analysis_name": f"BT4_lorsa_{suffix}",
    }


BT4_SAE_COMBOS: Dict[str, SaeComboConfig] = {
    combo["id"]: combo
    for combo in (
        _make_bt4_sae_combo(30, 16),
        _make_bt4_sae_combo(30, 32),
        _make_bt4_sae_combo(60, 16),
        _make_bt4_sae_combo(60, 32),
        _make_bt4_sae_combo(90, 16),
        _make_bt4_sae_combo(90, 32),
    )
}

# Default SAE combo used as initial frontend option
BT4_DEFAULT_SAE_COMBO: str = "k_30_e_16"


def get_bt4_sae_combo(combo_id: str | None) -> SaeComboConfig:
    """Get SAE config by combo ID; None or unknown IDs fall back to the default combo."""

    if combo_id is None:
        return BT4_SAE_COMBOS[BT4_DEFAULT_SAE_COMBO]
    return BT4_SAE_COMBOS.get(combo_id, BT4_SAE_COMBOS[BT4_DEFAULT_SAE_COMBO])


# ===== Backward-compatible path constants =====
# Keep BT4_TC_BASE_PATH / BT4_LORSA_BASE_PATH pointing to the default combo
# for compatibility with older code; new code should prefer get_bt4_sae_combo.

_default_combo = get_bt4_sae_combo(BT4_DEFAULT_SAE_COMBO)

BT4_TC_BASE_PATH: str = _default_combo["tc_base_path"]
BT4_LORSA_BASE_PATH: str = _default_combo["lorsa_base_path"]
