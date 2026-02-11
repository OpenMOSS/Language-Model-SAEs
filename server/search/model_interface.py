from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Optional, Callable, Any

import sys
import torch
import chess

# 为了能够导入 leela_interp，需要将其所在目录加入 sys.path
# 该路径与本地 notebook 中保持一致：
# PROJECT_ROOT = "/inspire/.../chess-SAEs/exp/leela-interp/src"
PROJECT_ROOT = "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/exp/leela-interp/src"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_MODEL_NAME = "lc0/BT4-1024x15x32h"

# 可选的外部模型获取函数，用于与 app.py 共享缓存
_external_model_getter: Optional[Callable[[str], object]] = None


def set_model_getter(getter: Callable[[str], object]) -> None:
    """设置外部模型获取函数，用于共享缓存的模型
    
    Args:
        getter: 一个函数，接受 model_name 并返回 HookedTransformer 模型
    """
    global _external_model_getter
    _external_model_getter = getter


def _try_get_from_circuits_service(model_name: str) -> Optional[object]:
    """尝试从 circuits_service 获取缓存的模型"""
    try:
        import sys
        import os
        # 添加 server 目录到路径
        server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if server_dir not in sys.path:
            sys.path.insert(0, server_dir)
        
        from circuits_service import get_cached_models
        cached_model, _, _, _ = get_cached_models(model_name)
        if cached_model is not None:
            print(f"✅ [model_interface] 使用 circuits_service 缓存的模型: {model_name}")
            return cached_model
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ [model_interface] 从 circuits_service 获取模型失败: {e}")
    return None


@lru_cache(maxsize=4)
def _get_model_internal(model_name: str = DEFAULT_MODEL_NAME) -> object:
    """内部模型加载函数（仅当外部 getter 不可用时使用）"""
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        dtype=torch.float32,
    )
    return model.eval()


def _get_model(model_name: str = DEFAULT_MODEL_NAME) -> object:
    """获取模型，优先使用外部缓存"""
    # 1. 优先使用外部 getter（由 app.py 设置）
    if _external_model_getter is not None:
        return _external_model_getter(model_name)
    
    # 2. 尝试从 circuits_service 获取缓存
    cached = _try_get_from_circuits_service(model_name)
    if cached is not None:
        return cached
    
    # 3. 回退到内部加载
    return _get_model_internal(model_name)


@lru_cache(maxsize=256)
def _run_model_outputs(fen: str, model_name: str) -> tuple[torch.Tensor, ...]:
    if not fen:
        raise ValueError("FEN string should not be empty")
    model = _get_model(model_name)
    with torch.no_grad():
        output, _ = model.run_with_cache(fen, prepend_bos=False)
    if not isinstance(output, Sequence) or len(output) < 3:
        raise RuntimeError("Output format isn't correct, it should contain policy、value、m")
    # lru_cache 要求返回值是可哈希的
    return tuple(output)


def get_policy(fen: str, model_name: str = DEFAULT_MODEL_NAME) -> torch.Tensor:
    outputs = _run_model_outputs(fen, model_name)
    return outputs[0]


def get_value(fen: str, model_name: str = DEFAULT_MODEL_NAME) -> torch.Tensor:
    """获取价值输出（WDL 格式）
    
    Args:
        fen: FEN 字符串
        model_name: 模型名称
        
    Returns:
        Tensor of shape [1, 3]，包含 [win_rate, draw_rate, lose_rate]
        - value[0][0]: win rate（胜率）
        - value[0][1]: draw rate（和棋率）
        - value[0][2]: lose rate（败率）
    """
    outputs = _run_model_outputs(fen, model_name)
    return outputs[1]


def get_wl(fen: str, model_name: str = DEFAULT_MODEL_NAME) -> float:
    """获取 Win-Loss 值（wl_）
    
    wl_ = win_rate - lose_rate
    
    Args:
        fen: FEN 字符串
        model_name: 模型名称
        
    Returns:
        Win-Loss 值（范围通常在 [-1, 1]）
    """
    value = get_value(fen, model_name)
    # value[0][0] 是 win rate, value[0][2] 是 lose rate
    win_rate = value[0][0].item()
    lose_rate = value[0][2].item()
    return win_rate - lose_rate


def get_d(fen: str, model_name: str = DEFAULT_MODEL_NAME) -> float:
    """获取 Draw 概率（d_）
    
    d_ = draw_rate
    
    Args:
        fen: FEN 字符串
        model_name: 模型名称
        
    Returns:
        Draw 概率（范围 [0, 1]）
    """
    value = get_value(fen, model_name)
    # value[0][1] 是 draw rate
    return value[0][1].item()


def get_q(fen: str, model_name: str = DEFAULT_MODEL_NAME, draw_score: float = 0.0) -> float:
    """获取 Q 值
    
    Q = wl + draw_score × d
    其中：
    - wl = win_rate - lose_rate
    - d = draw_rate
    - draw_score: 和棋得分（默认 0.0）
    
    Args:
        fen: FEN 字符串
        model_name: 模型名称
        draw_score: 和棋得分，默认为 0.0
        
    Returns:
        Q 值（范围通常在 [-1, 1] 附近，取决于 draw_score）
    """
    wl = get_wl(fen, model_name)
    d = get_d(fen, model_name)
    return wl + draw_score * d


def get_m(fen: str, model_name: str = DEFAULT_MODEL_NAME) -> torch.Tensor:
    """获取 Moves left 值（m_）
    
    Args:
        fen: FEN 字符串
        model_name: 模型名称
        
    Returns:
        Moves left Tensor
    """
    outputs = _run_model_outputs(fen, model_name)
    return outputs[2]


def policy_tensor_to_move_dict(
    policy_tensor: torch.Tensor,
    fen: str,
    legal_moves: list[chess.Move] | None = None,
) -> dict[str, float]:
    """将 policy tensor 转换为移动字典（move_uci -> probability）
    
    使用 LeelaBoard 将 policy tensor 的索引映射到 UCI 移动。
    
    Args:
        policy_tensor: Policy tensor，形状为 [1, 1858] 或 [1858]
        fen: FEN 字符串，用于创建 LeelaBoard
        legal_moves: 合法移动列表，如果为 None 则从 FEN 生成
        
    Returns:
        字典，键为 UCI 移动字符串，值为对应的概率（已归一化到合法移动）
    """
    import chess
    from leela_interp import LeelaBoard
    
    # 确保 policy_tensor 是一维的
    if policy_tensor.dim() > 1:
        policy_logits = policy_tensor[0] if policy_tensor.dim() == 2 else policy_tensor
    else:
        policy_logits = policy_tensor
    
    # 转换为 numpy 或保持为 tensor（用于索引）
    if isinstance(policy_logits, torch.Tensor):
        policy_logits = policy_logits.cpu()
    
    # 创建 LeelaBoard
    lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
    
    # 获取合法移动
    if legal_moves is None:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
    
    legal_uci_set = set(move.uci() for move in legal_moves)
    
    # 将 policy logits 转换为概率（softmax）
    if isinstance(policy_logits, torch.Tensor):
        policy_probs = torch.softmax(policy_logits, dim=0)
    else:
        import numpy as np
        policy_probs = torch.softmax(torch.from_numpy(policy_logits), dim=0)
    
    # 构建移动字典
    policy_dict: dict[str, float] = {}
    total_prob = 0.0
    
    # 遍历所有可能的索引，找到对应的合法移动
    for idx in range(len(policy_probs)):
        try:
            uci = lboard.idx2uci(idx)
            if uci in legal_uci_set:
                prob = policy_probs[idx].item() if isinstance(policy_probs[idx], torch.Tensor) else float(policy_probs[idx])
                policy_dict[uci] = prob
                total_prob += prob
        except (KeyError, IndexError, ValueError):
            # 索引可能无效，跳过
            continue
    
    # 归一化：确保合法移动的概率和为 1
    if total_prob > 0:
        policy_dict = {uci: prob / total_prob for uci, prob in policy_dict.items()}
    else:
        # 如果没有找到任何合法移动的概率，使用均匀分布
        uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0.0
        policy_dict = {move.uci(): uniform_prob for move in legal_moves}
    
    return policy_dict


def evaluate_position(fen: str, model_name: str = DEFAULT_MODEL_NAME) -> dict[str, object]:
    """完整评估一个局面，返回 q, d, m, p（用于搜索 Backend）
    
    Args:
        fen: FEN 字符串
        model_name: 模型名称
        
    Returns:
        字典，包含:
        - q: Win-Loss 值
        - d: Draw 概率
        - m: Moves left 值
        - p: 策略字典 {move_uci: probability}
    """
    # 获取 WDL
    value = get_value(fen, model_name)
    win_rate = value[0][0].item()
    draw_rate = value[0][1].item()
    lose_rate = value[0][2].item()
    
    wl = win_rate - lose_rate
    
    # 获取 M
    m_tensor = get_m(fen, model_name)
    m_value = m_tensor[0][0].item() if m_tensor.dim() > 1 else m_tensor[0].item()
    
    # 获取 Policy
    policy_tensor = get_policy(fen, model_name)
    policy_dict = policy_tensor_to_move_dict(policy_tensor, fen)
    
    return {
        'q': wl,
        'd': draw_rate,
        'm': m_value,
        'p': policy_dict,
    }


def evaluate_position_for_search(fen: str, model_name: str = DEFAULT_MODEL_NAME) -> dict[str, object]:
    """为 MCTS 搜索提供局面评估
    
    返回包含 q, d, m, p 的字典，供 SimpleBackend 使用。
    
    Args:
        fen: FEN 字符串
        model_name: 模型名称
        
    Returns:
        字典，包含：
        - q: float, Win-Loss 值
        - d: float, Draw 概率
        - m: float, Moves left 值
        - p: dict[str, float], 移动概率字典（move_uci -> probability）
    """
    # 获取模型输出
    policy_tensor = get_policy(fen, model_name)
    value_tensor = get_value(fen, model_name)
    m_tensor = get_m(fen, model_name)
    
    # 计算 WL 和 D
    win_rate = value_tensor[0][0].item()
    draw_rate = value_tensor[0][1].item()
    lose_rate = value_tensor[0][2].item()
    wl = win_rate - lose_rate
    
    # 获取 M 值
    m_value = m_tensor[0][0].item() if m_tensor.dim() > 1 else m_tensor.item()
    
    # 获取策略字典
    policy_dict = policy_tensor_to_move_dict(policy_tensor, fen)
    
    return {
        'q': wl,
        'd': draw_rate,
        'm': m_value,
        'p': policy_dict,
    }


def create_search_backend_eval_fn(model_name: str = DEFAULT_MODEL_NAME) -> Callable[[str], dict[str, object]]:
    """创建用于 SimpleBackend 的评估函数
    
    Args:
        model_name: 模型名称
        
    Returns:
        评估函数，接受 FEN 字符串，返回评估结果字典
    """
    def eval_fn(fen: str) -> dict[str, object]:
        return evaluate_position_for_search(fen, model_name)
    return eval_fn


def run_mcts_search(
    fen: str,
    max_playouts: int = 100,
    target_minibatch_size: int = 8,
    cpuct: float = 1.0,
    max_depth: int = 10,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """运行 MCTS 搜索并返回最佳移动
    
    Args:
        fen: FEN 字符串
        max_playouts: 最大模拟次数
        target_minibatch_size: 目标 minibatch 大小
        cpuct: UCT 探索系数
        max_depth: 最大搜索深度
        model_name: 模型名称
        
    Returns:
        包含最佳移动和搜索统计信息的字典
    """
    from .search import SearchParams, Search, SimpleBackend, Node
    
    # 创建搜索参数
    params = SearchParams(
        max_playouts=max_playouts,
        target_minibatch_size=target_minibatch_size,
        cpuct=cpuct,
        max_depth=max_depth,
    )
    
    # 创建模型评估函数
    eval_fn = create_search_backend_eval_fn(model_name)
    
    # 创建后端
    backend = SimpleBackend(model_eval_fn=eval_fn)
    
    # 创建根节点
    root_node = Node(fen=fen)
    
    # 创建搜索实例
    search = Search(
        root_node=root_node,
        backend=backend,
        params=params,
    )
    
    # 运行搜索
    search.run_blocking()
    
    # 获取最佳移动
    best_move = search.get_best_move()
    
    # 收集搜索统计信息
    result: dict[str, Any] = {
        'best_move': best_move.uci() if best_move else None,
        'total_playouts': search.get_total_playouts(),
        'max_depth_reached': search.get_current_max_depth(),
        'root_visits': root_node.get_n(),
    }
    
    # 收集根节点子节点的访问统计
    if root_node.has_children():
        children_stats = []
        for i in range(root_node.get_num_edges()):
            edge = root_node.get_edge_at_index(i)
            child = root_node.get_child_at_index(i)
            if edge:
                move = edge.get_move()
                visits = child.get_n() if child else 0
                q = child.get_q(0.0) if child and child.get_n() > 0 else 0.0
                children_stats.append({
                    'move': move.uci() if move else None,
                    'visits': visits,
                    'q': q,
                    'policy': edge.get_p(),
                })
        # 按访问次数排序
        children_stats.sort(key=lambda x: x['visits'], reverse=True)
        result['top_moves'] = children_stats[:10]  # 只返回前10个
    
    return result