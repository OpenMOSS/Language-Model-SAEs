from __future__ import annotations
from typing import Dict, TypedDict


class SaeComboConfig(TypedDict):
    """Configuration for a single SAE combo."""

    id: str  # ID, e.g. "k_64_e_32"
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

BT4_SAE_COMBOS: Dict[str, SaeComboConfig] = {
    "k_30_e_16": {
        "id": "k_30_e_16",
        "label": "k_30_e_16 (small capacity)",
        "tc_base_path": (
            "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
            "rlin_projects/chess-SAEs-N/result_BT4/tc/k_30_e_16"
        ),
        "lorsa_base_path": (
            "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
            "rlin_projects/chess-SAEs-N/result_BT4/lorsa/k_30_e_16"
        ),
        "tc_sae_name_template": "BT4_tc_L{layer}M_k30_e16",
        "lorsa_sae_name_template": "BT4_lorsa_L{layer}A_k30_e16",
        "tc_analysis_name": "BT4_tc_k30_e16",
        "lorsa_analysis_name": "BT4_lorsa_k30_e16",
    },
    "k_64_e_32": {
        "id": "k_64_e_32",
        "label": "k_64_e_32 (default medium capacity)",
        "tc_base_path": (
            "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
            "rlin_projects/chess-SAEs-N/result_BT4/tc/k_64_e_32"
        ),
        "lorsa_base_path": (
            "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
            "rlin_projects/chess-SAEs-N/result_BT4/lorsa/k_64_e_32"
        ),
        "tc_sae_name_template": "BT4_tc_L{layer}M_k64_e32",
        "lorsa_sae_name_template": "BT4_lorsa_L{layer}A_k64_e32",
        "tc_analysis_name": "BT4_tc_k64_e32",
        "lorsa_analysis_name": "BT4_lorsa_k64_e32",
    },
    "k_128_e_64": {
        "id": "k_128_e_64",
        "label": "k_128_e_64 (larger capacity)",
        "tc_base_path": (
            "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
            "rlin_projects/chess-SAEs-N/result_BT4/tc/k_128_e_64"
        ),
        "lorsa_base_path": (
            "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
            "rlin_projects/chess-SAEs-N/result_BT4/lorsa/k_128_e_64"
        ),
        "tc_sae_name_template": "BT4_tc_L{layer}M_k128_e64",
        "lorsa_sae_name_template": "BT4_lorsa_L{layer}A_k128_e64",
        "tc_analysis_name": "BT4_tc_k128_e64",
        "lorsa_analysis_name": "BT4_lorsa_k128_e64",
    },
    "k_256_e_128": {
        "id": "k_256_e_128",
        "label": "k_256_e_128 (largest capacity)",
        "tc_base_path": (
            "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
            "rlin_projects/chess-SAEs-N/result_BT4/tc/k_256_e_128"
        ),
        "lorsa_base_path": (
            "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
            "rlin_projects/chess-SAEs-N/result_BT4/lorsa/k_256_e_128"
        ),
        "tc_sae_name_template": "BT4_tc_L{layer}M_k256_e128",
        "lorsa_sae_name_template": "BT4_lorsa_L{layer}A_k256_e128",
        "tc_analysis_name": "BT4_tc_k256_e128",
        "lorsa_analysis_name": "BT4_lorsa_k256_e128",
    },
    "k_128_e_128": {
        "id": "k_128_e_128",
        "label": "k_128_e_128 (original default combo)",
        "tc_base_path": (
            "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
            "rlin_projects/chess-SAEs-N/result_BT4/tc/k_128_e_128"
        ),
        "lorsa_base_path": (
            "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
            "rlin_projects/chess-SAEs-N/result_BT4/lorsa/k_128_e_128"
        ),
        # Under this combo, SAE names are consistent with the old version, without k/e suffix
        "tc_sae_name_template": "BT4_tc_L{layer}M",
        "lorsa_sae_name_template": "BT4_lorsa_L{layer}A",
        "tc_analysis_name": "BT4_tc",
        "lorsa_analysis_name": "BT4_lorsa",
    },
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
