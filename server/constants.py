from __future__ import annotations

"""
全局常量配置，用于在不同模块之间共享模型与路径设置。

目前支持多种 BT4 SAE 组合（不同的 k / e 配置），前端只能在这些
预定义的组合中进行选择。
"""

from typing import Dict, TypedDict


class SaeComboConfig(TypedDict):
    """单个 SAE 组合的配置"""

    id: str  # 组合 ID，例如 "k_64_e_32"
    label: str  # 前端展示用文案
    tc_base_path: str  # Transcoder SAE 基础路径
    lorsa_base_path: str  # LoRSA SAE 基础路径
    tc_sae_name_template: str  # Transcoder SAE 名称模板（用于构建字典名）
    lorsa_sae_name_template: str  # LoRSA SAE 名称模板（用于构建字典名）
    tc_analysis_name: str  # Transcoder analysis_name（保存到 JSON 的 metadata 中）
    lorsa_analysis_name: str  # LoRSA analysis_name（保存到 JSON 的 metadata 中）


# BT4 模型名称（Leela Chess Zero BT4 棋力模型）
BT4_MODEL_NAME: str = "lc0/BT4-1024x15x32h"

# ===== BT4 可选 SAE 组合（来自 exp/38mongoanalyses/组合.txt） =====

BT4_SAE_COMBOS: Dict[str, SaeComboConfig] = {
    "k_30_e_16": {
        "id": "k_30_e_16",
        "label": "k_30_e_16（小容量）",
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
        "label": "k_64_e_32（默认中等容量）",
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
        "label": "k_128_e_64（较大容量）",
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
        "label": "k_256_e_128（最大容量）",
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
        "label": "k_128_e_128（原始默认组合）",
        "tc_base_path": (
            "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
            "rlin_projects/chess-SAEs-N/result_BT4/tc/k_128_e_128"
        ),
        "lorsa_base_path": (
            "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
            "rlin_projects/chess-SAEs-N/result_BT4/lorsa/k_128_e_128"
        ),
        # 这一组合下 SAE 名称与老版本一致，无 k/e 后缀
        "tc_sae_name_template": "BT4_tc_L{layer}M",
        "lorsa_sae_name_template": "BT4_lorsa_L{layer}A",
        "tc_analysis_name": "BT4_tc",
        "lorsa_analysis_name": "BT4_lorsa",
    },
}

# 默认使用的 SAE 组合（前端初始选项）
BT4_DEFAULT_SAE_COMBO: str = "k_30_e_16"


def get_bt4_sae_combo(combo_id: str | None) -> SaeComboConfig:
    """根据组合 ID 获取 SAE 配置，None 或未知 ID 都回退到默认组合。"""

    if combo_id is None:
        return BT4_SAE_COMBOS[BT4_DEFAULT_SAE_COMBO]
    return BT4_SAE_COMBOS.get(combo_id, BT4_SAE_COMBOS[BT4_DEFAULT_SAE_COMBO])


# ===== 向后兼容的路径常量 =====
# 依然保留 BT4_TC_BASE_PATH / BT4_LORSA_BASE_PATH，指向默认组合，
# 以兼容旧代码；新的代码应尽量通过 get_bt4_sae_combo 获取路径。

_default_combo = get_bt4_sae_combo(BT4_DEFAULT_SAE_COMBO)

BT4_TC_BASE_PATH: str = _default_combo["tc_base_path"]
BT4_LORSA_BASE_PATH: str = _default_combo["lorsa_base_path"]
