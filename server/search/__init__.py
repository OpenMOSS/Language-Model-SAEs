"""
DAG Classic MCTS Search Package

这个包实现了基于 DAG（有向无环图）的 MCTS 搜索算法，用于国际象棋引擎。
"""

# 从 node 模块导入核心数据结构
from .node import (
    Edge,
    Node,
    LowNode,
    EdgeAndNode,
)

# 从 search 模块导入搜索相关类
from .search import (
    SearchParams,
    MEvaluator,
    Backend,
    SimpleBackend,
    TranspositionTable,
    Search,
    SearchTracer,
    PositionHistory,
    NodeToProcess,
    MakeRootMoveFilter,
    get_fpu,
    compute_cpuct,
)

# 从 model_interface 模块导入模型接口函数
from .model_interface import (
    DEFAULT_MODEL_NAME,
    set_model_getter,
    get_policy,
    get_value,
    get_wl,
    get_d,
    get_q,
    get_m,
    policy_tensor_to_move_dict,
    evaluate_position,
    evaluate_position_for_search,
    create_search_backend_eval_fn,
)

__all__ = [
    # 核心数据结构
    "Edge",
    "Node",
    "LowNode",
    "EdgeAndNode",
    
    # 搜索相关
    "SearchParams",
    "MEvaluator",
    "Backend",
    "SimpleBackend",
    "TranspositionTable",
    "Search",
    "SearchTracer",
    "PositionHistory",
    "NodeToProcess",
    "MakeRootMoveFilter",
    "get_fpu",
    "compute_cpuct",
    
    # 模型接口
    "DEFAULT_MODEL_NAME",
    "set_model_getter",
    "get_policy",
    "get_value",
    "get_wl",
    "get_d",
    "get_q",
    "get_m",
    "policy_tensor_to_move_dict",
    "evaluate_position",
    "evaluate_position_for_search",
    "create_search_backend_eval_fn",
]

__version__ = "0.1.0"
