from __future__ import annotations

"""
全局常量配置，用于在不同模块之间共享模型与路径设置。
"""

# BT4 模型名称（Leela Chess Zero BT4 棋力模型）
BT4_MODEL_NAME: str = "lc0/BT4-1024x15x32h"

# BT4 Transcoder SAE 的基础路径
BT4_TC_BASE_PATH: str = (
    "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
    "rlin_projects/chess-SAEs-N/result_BT4/tc/k_128_e_128"
)

# BT4 LoRSA SAE 的基础路径
BT4_LORSA_BASE_PATH: str = (
    "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
    "rlin_projects/chess-SAEs-N/result_BT4/lorsa/k_128_e_128"
)


