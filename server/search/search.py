import json
import chess
import math
import random
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from collections import deque
import time
import threading
import weakref
from pathlib import Path

try:
    from .node import Node, Edge, LowNode, EdgeAndNode
except ImportError:
    # 支持直接导入（如在 Jupyter notebook 中）
    from node import Node, Edge, LowNode, EdgeAndNode


def MakeRootMoveFilter(fen: str, searchmoves: List[str]) -> List[chess.Move]:
    """过滤根节点的合法移动"""
    board = chess.Board(fen)
    root_moves = []
    for move_str in searchmoves:
        try:
            move = chess.Move.from_uci(move_str)
            if board.is_legal(move):
                root_moves.append(move)
        except ValueError:
            continue
    return root_moves


def fast_sign(x: float) -> float:
    """快速符号函数"""
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return -1.0
    else:
        return 0.0


def fast_log(x: float) -> float:
    """快速对数函数"""
    return math.log(x) if x > 0.0 else float('-inf')


def fast_logistic(x: float) -> float:
    """快速逻辑函数"""
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class SearchParams:
    """搜索参数"""
    # M 评估参数
    moves_left_slope: float = 0.0
    moves_left_max_effect: float = 0.0
    moves_left_constant_factor: float = 0.0
    moves_left_scaled_factor: float = 0.0
    moves_left_quadratic_factor: float = 0.0
    moves_left_threshold: float = 0.0
    
    # UCT 参数
    cpuct: float = 3.0
    cpuct_factor: float = 0.0
    cpuct_base: float = 19652.0
    fpu_value: float = 0.0
    fpu_absolute: bool = False
    
    # 低Q值探索增强参数（用于发现弃后连杀等隐藏走法）
    low_q_exploration_enabled: bool = False  # 是否启用低Q值探索增强
    low_q_threshold: float = 0.3  # Q值阈值，低于此值认为是"低Q值"
    low_q_exploration_bonus: float = 0.1  # 探索奖励的基础值
    low_q_visit_threshold: int = 5  # 访问次数阈值，低于此值认为是"未充分探索"
    
    # 搜索控制
    max_playouts: int = 10000
    target_minibatch_size: int = 8
    max_collision_visits: int = 1
    max_out_of_order: int = 10
    max_depth: int = 0  # 最大搜索深度，0 表示不限制
    # 注意：较小的max_depth会在相同的max_playouts下探索更多分支（因为每条路径更短）
    # 但每条路径的评估可能不够深入，需要在深度和广度之间平衡
    
    # 温度参数
    temperature: float = 0.0
    temperature_cutoff_move: int = 0
    temperature_endgame: float = 0.0
    temperature_visit_offset: float = 0.0
    temperature_winpct_cutoff: float = 0.0
    
    # 其他
    draw_score: float = 0.0
    multipv: int = 1


class MEvaluator:
    """M 评估器，用于计算 Moves Left 的效用值"""
    
    def __init__(self, params: Optional[SearchParams] = None, parent: Optional[Node] = None):
        if params is None:
            self.enabled = False
            self.m_slope = 0.0
            self.m_cap = 0.0
            self.a_constant = 0.0
            self.a_linear = 0.0
            self.a_square = 0.0
            self.q_threshold = 0.0
            self.parent_m = 0.0
            self.parent_within_threshold = False
        else:
            self.enabled = True
            self.m_slope = params.moves_left_slope
            self.m_cap = params.moves_left_max_effect
            self.a_constant = params.moves_left_constant_factor
            self.a_linear = params.moves_left_scaled_factor
            self.a_square = params.moves_left_quadratic_factor
            self.q_threshold = params.moves_left_threshold
            self.parent_m = parent.GetM() if parent else 0.0
            self.parent_within_threshold = (
                self._within_threshold(parent, self.q_threshold) 
                if parent else False
            )
    
    def set_parent(self, parent: Node) -> None:
        """设置父节点"""
        assert parent is not None, "Parent node cannot be None"
        if self.enabled:
            self.parent_m = parent.GetM()
            self.parent_within_threshold = self._within_threshold(parent, self.q_threshold)
    
    def get_m_utility(self, child: Node, q: float) -> float:
        """计算子节点的 M 效用值"""
        if not self.enabled or not self.parent_within_threshold:
            return 0.0
        
        child_m = child.GetM()
        m = max(-self.m_cap, min(self.m_cap, self.m_slope * (child_m - self.parent_m)))
        m *= fast_sign(-q)
        
        if 0.0 < self.q_threshold < 1.0:
            q = max(0.0, (abs(q) - self.q_threshold)) / (1.0 - self.q_threshold)
        
        m *= self.a_constant + self.a_linear * abs(q) + self.a_square * q * q
        return m
    
    def get_m_utility_edge(self, edge: EdgeAndNode, q: float) -> float:
        """计算边的 M 效用值"""
        if not self.enabled or not self.parent_within_threshold:
            return 0.0
        if edge.get_n() == 0:
            return self.get_default_m_utility()
        return self.get_m_utility(edge.node(), q)
    
    def get_default_m_utility(self) -> float:
        """返回未访问节点的默认 M 效用值"""
        return 0.0
    
    @staticmethod
    def _within_threshold(parent: Optional[Node], q_threshold: float) -> bool:
        """检查父节点的 Q 值是否在阈值内"""
        if parent is None:
            return False
        return abs(parent.GetQ(0.0)) > q_threshold


# 辅助函数
def get_fpu(params: SearchParams, node: Node, is_root_node: bool, draw_score: float) -> float:
    """计算 First Play Urgency (FPU)"""
    value = params.fpu_value
    if params.fpu_absolute:
        return value
    else:
        return -node.get_q(-draw_score) - value * math.sqrt(node.get_visited_policy())


def compute_cpuct(params: SearchParams, n: int, is_root_node: bool) -> float:
    """计算 UCT 系数"""
    init = params.cpuct
    k = params.cpuct_factor
    base = params.cpuct_base
    if k == 0.0:
        return init
    return init + k * fast_log((n + base) / base)


@dataclass
class NodeToProcess:
    """待处理的节点"""
    path: List[Tuple[Node, int, int]]  # (节点, 重复次数, 剩余步数)
    node: Node
    multivisit: int = 1 # 一次回传视为几次访问，默认1次，C++中支持在并行搜索collision时被设置成更大，撞车直接记账
    is_collision_flag: bool = False  # 重命名避免与方法冲突
    maxvisit: int = 0
    ooo_completed: bool = False # NodeToProcess 对象是否已经完成了乱序评估
    nn_queried: bool = False # NodeToProcess 对象是否已经查询了神经网络
    is_tt_hit: bool = False # NodeToProcess 对象是否命中置换表
    is_cache_hit: bool = False # NodeToProcess 对象是否命中缓存
    hash: Optional[int] = None
    tt_low_node: Optional[LowNode] = None
    eval: Optional[Dict[str, Any]] = None  # {'q': float, 'd': float, 'm': float, 'p': List[float]}
    history: Optional[Any] = None  # PositionHistory
    repetitions: int = 0
    moves_left: int = 0
    
    @staticmethod
    def visit(path: List[Tuple[Node, int, int]], history: Any) -> 'NodeToProcess':
        """创建访问节点"""
        node, repetitions, moves_left = path[-1]
        return NodeToProcess(
            path=path,
            node=node,
            multivisit=1,
            is_collision_flag=False,
            history=history,
            eval={'q': 0.0, 'd': 0.0, 'm': 0.0, 'p': []},
            repetitions=repetitions,
            moves_left=moves_left
        )
    
    @staticmethod
    def collision(path: List[Tuple[Node, int, int]], multivisit: int, maxvisit: int = 0) -> 'NodeToProcess':
        """创建碰撞节点"""
        node, repetitions, moves_left = path[-1]
        return NodeToProcess(
            path=path,
            node=node,
            multivisit=multivisit,
            is_collision_flag=True,
            maxvisit=maxvisit,
            repetitions=repetitions,
            moves_left=moves_left
        )
    
    def is_collision(self) -> bool:
        """检查是否为碰撞"""
        return self.is_collision_flag
    
    def is_extendable(self) -> bool:
        """检查节点是否可扩展"""
        return not self.is_collision_flag and not self.node.is_terminal() and self.node.get_n() == 0
    
    def can_eval_out_of_order(self) -> bool:
        """检查是否可以乱序评估"""
        # 简化实现，总是返回 False
        return False


@dataclass
class TraceEdgeRecord:
    """用于记录一次边选择的追踪信息"""
    parent_fen: str
    child_fen: Optional[str]
    move_uci: Optional[str]
    score: float
    q: float
    u: float
    m: float
    visits: int
    stage: str
    timestamp: float


@dataclass
class TraceExpansionRecord:
    """用于记录一次节点扩展的追踪信息"""
    node_fen: str
    move_uci_list: List[str]
    policies: List[float]
    is_tt_hit: bool
    timestamp: float


class SearchTracer:
    """用于记录搜索过程中选择与扩展的辅助工具"""

    def __init__(self) -> None:
        self._edge_records: List[TraceEdgeRecord] = []
        self._expansion_records: List[TraceExpansionRecord] = []
        self._lock = threading.Lock()

    def log_selection(
        self,
        parent: Node,
        edge: EdgeAndNode,
        score: float,
        q: float,
        u: float,
        m: float,
        stage: str,
    ) -> None:
        """记录一次边选择"""
        move = edge.get_move() if edge and edge.edge() else None
        child = edge.node() if edge else None
        record = TraceEdgeRecord(
            parent_fen=parent.get_fen(),
            child_fen=child.get_fen() if child else None,
            move_uci=move.uci() if move else None,
            score=score,
            q=q,
            u=u,
            m=m,
            visits=edge.get_n() if edge else 0,
            stage=stage,
            timestamp=time.time(),
        )
        with self._lock:
            self._edge_records.append(record)

    def log_expansion(
        self,
        node: Node,
        legal_moves: List[chess.Move],
        policies: List[float],
        is_tt_hit: bool,
    ) -> None:
        """记录一次节点扩展的信息"""
        record = TraceExpansionRecord(
            node_fen=node.get_fen(),
            move_uci_list=[move.uci() for move in legal_moves],
            policies=policies.copy(),
            is_tt_hit=is_tt_hit,
            timestamp=time.time(),
        )
        with self._lock:
            self._expansion_records.append(record)

    def get_edge_records(self) -> List[TraceEdgeRecord]:
        """获取当前记录的边选择信息（浅拷贝）"""
        with self._lock:
            return self._edge_records.copy()

    def get_expansion_records(self) -> List[TraceExpansionRecord]:
        """获取当前记录的扩展信息（浅拷贝）"""
        with self._lock:
            return self._expansion_records.copy()

    def export_graphviz(
        self,
        output_path: str,
        max_edges: int = 200,
    ) -> None:
        """将当前追踪记录导出为 graphviz 图文件"""
        try:
            import graphviz
        except ImportError as exc:
            raise RuntimeError("graphviz 未安装，无法导出搜索图。") from exc

        dot = graphviz.Digraph(comment="Search Trace Export")
        with self._lock:
            edge_records = self._edge_records[:max_edges]

        for record in edge_records:
            if not record.child_fen or not record.move_uci:
                continue
            label = f"{record.move_uci}\\nscore={record.score:.2f}\\nq={record.q:.2f}"
            dot.edge(record.parent_fen, record.child_fen, label=label)

        dot.render(output_path, format="pdf", cleanup=True)

    def export_json(
        self,
        output_path: str,
        max_edges: Optional[int] = 1000,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        将追踪数据导出为包含节点/边列表的 JSON 文件
        
        Args:
            output_path: 输出文件路径
            max_edges: 最大边记录数，0 或 None 表示不限制（保存完整搜索树）
            metadata: 可选的元数据字典，包含搜索参数、统计信息等
        """
        data: Dict[str, Any] = {
            "nodes": [],
            "edges": [],
            "metadata": metadata if metadata is not None else {},
        }
        with self._lock:
            # 如果 max_edges 为 0 或 None，则不限制，保存所有边记录
            if max_edges is None or max_edges == 0:
                edge_records = self._edge_records
            else:
                edge_records = self._edge_records[:max_edges]
            expansions = {record.node_fen: record for record in self._expansion_records}

        seen_nodes: set[str] = set()

        def add_node(fen: Optional[str]) -> None:
            if not fen or fen in seen_nodes:
                return
            expansion = expansions.get(fen)
            data["nodes"].append({
                "fen": fen,
                "moves": expansion.move_uci_list if expansion else [],
                "policies": expansion.policies if expansion else [],
                "is_tt_hit": expansion.is_tt_hit if expansion else None,
                "timestamp": expansion.timestamp if expansion else None,
            })
            seen_nodes.add(fen)

        for record in edge_records:
            add_node(record.parent_fen)
            add_node(record.child_fen)
            if record.child_fen and record.move_uci:
                data["edges"].append({
                    "parent": record.parent_fen,
                    "child": record.child_fen,
                    "move": record.move_uci,
                    "score": record.score,
                    "q": record.q,
                    "u": record.u,
                    "m": record.m,
                    "visits": record.visits,
                    "stage": record.stage,
                    "timestamp": record.timestamp,
                })

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)


class PositionHistory:
    """位置历史，用于跟踪游戏历史"""
    
    def __init__(self, initial_fen: str = chess.Board().fen()):
        self.fens: List[str] = [initial_fen]
        self.moves: List[chess.Move] = []
    
    def append(self, move: chess.Move) -> None:
        """添加移动"""
        board = chess.Board(self.fens[-1])
        board.push(move)
        self.fens.append(board.fen())
        self.moves.append(move)
    
    def pop(self) -> None:
        """移除最后一个移动"""
        if self.fens:
            self.fens.pop()
        if self.moves:
            self.moves.pop()
    
    def last(self) -> chess.Board:
        """获取最后一个位置"""
        return chess.Board(self.fens[-1])
    
    def get_length(self) -> int:
        """获取历史长度"""
        return len(self.fens)
    
    def trim(self, length: int) -> None:
        """修剪历史到指定长度"""
        if length < len(self.fens):
            self.fens = self.fens[:length]
            self.moves = self.moves[:length-1] if length > 0 else []
    
    def get_positions(self) -> List[str]:
        """获取所有位置"""
        return self.fens.copy()
    
    def is_black_to_move(self) -> bool:
        """检查是否轮到黑方"""
        return self.last().turn == chess.BLACK


class Backend:
    """后端接口，用于神经网络评估"""
    
    def evaluate(self, fen: str, legal_moves: List[chess.Move]) -> Dict[str, Any]:
        """
        评估位置
        
        Returns:
            {'q': float, 'd': float, 'm': float, 'p': List[float]}
        """
        raise NotImplementedError


class SimpleBackend(Backend):
    """简单后端实现，使用模型接口"""
    
    def __init__(self, model_eval_fn: Optional[Callable[[str], Dict[str, Any]]] = None):
        self.model_eval_fn = model_eval_fn
        self.cache: Dict[str, Dict[str, Any]] = {}
        # self.cache:  {'8/5B2/2R5/1pP1p1k1/1P2Pb2/r5pK/8/8 b - - 2 50': {'q': 0.004424240440130234, 'd': 0.9480566382408142, 'm': 163.16165161132812, 'p': {'f4e3': 0.0625, 'f4d2': 0.0625...
    
    def evaluate(self, fen: str, legal_moves: List[chess.Move]) -> Dict[str, Any]:
        """评估位置"""
        # print("self.model_eval_fn: ", self.model_eval_fn)
        # print("self.cache: ", self.cache)
        # print("fen: ", fen)
        # print("legal_moves: ", legal_moves)
        if fen in self.cache:
            cached = self.cache[fen]
            
            # 提取合法移动的策略
            policies = [cached.get('p', {}).get(move.uci(), 0.0) for move in legal_moves]
            return {
                'q': cached.get('q', 0.0),
                'd': cached.get('d', 0.0),
                'm': cached.get('m', 0.0),
                'p': policies
            }
        
        if self.model_eval_fn:
            result = self.model_eval_fn(fen)
            self.cache[fen] = result
            policies = [result.get('p', {}).get(move.uci(), 0.0) for move in legal_moves]
            return {
                'q': result.get('q', 0.0),
                'd': result.get('d', 0.0),
                'm': result.get('m', 0.0),
                'p': policies
            }
        
        # 默认返回值
        return {
            'q': 0.0,
            'd': 0.0,
            'm': 0.0,
            'p': [1.0 / len(legal_moves)] * len(legal_moves) if legal_moves else []
        }


class TranspositionTable:
    """置换表，用于存储和重用 LowNode"""
    
    def __init__(self):
        # 使用弱引用字典，当 LowNode 不再被引用时自动清理
        self.table: Dict[int, LowNode] = {}  # {position_hash: LowNode}
    
    def get(self, position_hash: int) -> Optional[LowNode]:
        """获取位置的 LowNode"""
        return self.table.get(position_hash)
    
    def put(self, position_hash: int, low_node: LowNode) -> None:
        """存储位置的 LowNode"""
        self.table[position_hash] = low_node
    
    def clear_expired(self) -> None:
        """清理过期的条目（简化实现，Python 的垃圾回收会自动处理）"""
        # 在完整实现中，可以使用 weakref 来检测过期的条目
        pass


class Search:
    def __init__(
        self,
        root_node: Node,
        backend: Backend,
        params: SearchParams,
        root_move_filter: Optional[List[chess.Move]] = None,
        start_time: Optional[float] = None,
        transposition_table: Optional[TranspositionTable] = None,
        tracer: Optional[SearchTracer] = None,
    ):
        self.root_node_ = root_node
        self.backend_ = backend
        self.params_ = params
        self.root_move_filter_ = root_move_filter or []
        self.start_time_ = start_time if start_time else time.time()
        self.tt_ = transposition_table if transposition_table else TranspositionTable()
        
        self.stop_ = False
        self.total_playouts_ = 0
        self.initial_visits_ = root_node.get_n()
        self.cum_depth_ = 0
        self.max_depth_ = 0
        
        self.nodes_mutex_ = threading.Lock()
        self.current_best_edge_: Optional[EdgeAndNode] = None
        self.tracer_: Optional[SearchTracer] = tracer
        
        # 位置历史
        board = chess.Board(root_node.get_fen())
        self.played_history_ = PositionHistory(root_node.get_fen())
    
    def get_draw_score(self, is_odd_depth: bool) -> float:
        """获取和棋得分"""
        is_black_to_move = self.played_history_.is_black_to_move()
        is_root = not is_odd_depth
        if (is_odd_depth == is_black_to_move) == is_root:
            return self.params_.draw_score
        else:
            return -self.params_.draw_score
    
    def get_best_child_no_temperature(
        self, 
        parent: Node, 
        depth: int
    ) -> EdgeAndNode:
        """
        获取最终落子使用的最佳子节点（尽量对齐 C++ 版的无温度选子逻辑）
        
        - 优先级顺序与 C++ `GetBestChildrenNoTemperature` 一致：
          1. 已证明的胜局（先非 tablebase 将杀，再 tablebase 胜）
          2. 非终局分支：优先高访问数，其次高 Q，最后高先验 P
          3. 已证明的败局（先 tablebase 败，再普通败），且偏好“拖长败局”的走法
        - 不再在最终选招中加入 U 项，避免因为探索欲导致放弃已经找到的必杀线。
        """
        best_list = self.get_best_children_no_temperature(parent, 1, depth)
        if not best_list:
            return EdgeAndNode()
        best_edge = best_list[0]
        # 记录一次“最终选子” trace，便于在 JSON 里对齐 C++ 的 bestmove 决策
        is_odd_depth = (depth % 2) == 1
        draw_score = self.get_draw_score(is_odd_depth)
        fpu = get_fpu(self.params_, parent, parent == self.root_node_, draw_score)
        m_evaluator = (
            MEvaluator(self.params_, parent)
            if self.params_.moves_left_slope != 0.0
            else MEvaluator()
        )
        q = best_edge.get_q(fpu, draw_score)
        m = m_evaluator.get_m_utility_edge(best_edge, q)
        # 最终选子阶段不再有 U 项
        u = 0.0
        score = q + m
        self._trace_edge_selection(parent, best_edge, score, (q, u, m), "best_child")
        return best_edge
     
    def get_best_children_no_temperature( # no use
        self,
        parent: Node,
        count: int,
        depth: int
    ) -> List[EdgeAndNode]:
        """
        按 C++ `GetBestChildrenNoTemperature` 的逻辑返回访问数前几名的候选边。
        
        排序规则（从好到坏）：
        1. 终局胜（普通将杀 > tablebase 将杀，偏好更快将杀，M 更小）
        2. 非终局：优先访问数 N 更大，其次 Q 更大，最后先验 P 更大
        3. 终局败（tablebase 败 > 普通败），偏好“拖长失败”，即 M 更大
        """
        if parent.get_n() == 0:
            return []

        is_odd_depth = (depth % 2) == 1
        draw_score = self.get_draw_score(is_odd_depth)
        fpu = get_fpu(self.params_, parent, parent == self.root_node_, draw_score)

        # 收集候选边
        edges: List[EdgeAndNode] = []
        for i in range(parent.get_num_edges()):
            edge = parent.get_edge_at_index(i)
            child = parent.get_child_at_index(i)

            # 根节点过滤：只允许在 searchmoves 里的招法
            if parent == self.root_node_ and self.root_move_filter_:
                if edge.get_move() not in self.root_move_filter_:
                    continue

            edges.append(EdgeAndNode(edge, child))

        if not edges:
            return []

        def edge_rank(e: EdgeAndNode) -> int:
            """
            与 C++ 中 EdgeRank 含义对应：
            0: 终局失败
            1: tablebase 失败
            2: 非终局 / 和棋 / 未访问
            3: tablebase 胜
            4: 终局胜
            """
            n = e.get_n()
            wl = e.get_wl(0.0)
            if n == 0 or (not e.is_terminal()) or wl == 0.0:
                return 2  # 非终局 / 暂未分出胜负
            if e.is_tb_terminal():
                return 1 if wl < 0.0 else 3
            return 0 if wl < 0.0 else 4

        def sort_key(e: EdgeAndNode) -> Tuple[float, ...]:
            kind = edge_rank(e)
            n = e.get_n()
            q = e.get_q(0.0, draw_score)
            p = e.get_p()
            m = e.get_m(0.0)

            # 非终局：优先访问数、再 Q、再先验 P（全部按大优先）
            if kind == 2:
                return (kind, float(n), q, p)

            # 胜局：kind 越大越好，且偏好更快将杀（M 更小）
            if kind >= 3:
                return (kind, -m)

            # 败局：kind 越大越“不差”，且偏好拖长失败（M 更大）
            return (kind, m)

        # 与 C++ 一样，按“更好”的在前排序
        edges.sort(key=sort_key, reverse=True)
        if count < len(edges):
            edges = edges[:count]
        return edges
    
    def pick_node_to_extend(self, node: Node, path: List[Tuple[Node, int, int]], history: PositionHistory) -> Optional[NodeToProcess]:
        """选择要扩展的节点（UCT 选择，递归实现）"""
        # 检查节点是否为终端节点或未访问节点
        if node.is_terminal():
            return None
        
        # 检查最大深度限制
        current_depth = len(path)
        if self.params_.max_depth > 0 and current_depth >= self.params_.max_depth: # search stopper 4, 不过 这个是一条路径达到深度不再扩展，所有路径都不扩展的时候才停止
            # 达到最大深度，不再扩展
            return None
        
        if node.get_n() == 0:
            # 未访问节点，尝试标记为正在扩展
            if node.try_start_score_update():
                return NodeToProcess.visit(path, history)
            return None
        
        # 检查是否有子节点
        if not node.has_children():
            # 需要扩展节点
            if node.try_start_score_update():
                return NodeToProcess.visit(path, history)
            return None
        
        # 使用 UCT 选择最佳子节点
        is_root = node == self.root_node_
        is_odd_depth = len(path) % 2 == 1
        draw_score = self.get_draw_score(is_odd_depth) 
        fpu = get_fpu(self.params_, node, is_root, draw_score)
        cpuct = compute_cpuct(self.params_, node.get_total_visits(), is_root)
        u_coeff = cpuct * math.sqrt(max(node.get_children_visits(), 1))
        
        m_evaluator = MEvaluator(self.params_, node) if self.params_.moves_left_slope != 0.0 else MEvaluator()
        
        candidates: list[Tuple[float, EdgeAndNode, int, Tuple[float, float, float]]] = []
        for i in range(node.get_num_edges()):
            edge = node.get_edge_at_index(i)
            child = node.get_child_at_index(i)
            
            if is_root and self.root_move_filter_:
                if edge.get_move() not in self.root_move_filter_:
                    continue
            
            edge_and_node = EdgeAndNode(edge, child)
            q = edge_and_node.get_q(fpu, draw_score)
            m = m_evaluator.get_m_utility_edge(edge_and_node, q)
            u = edge_and_node.get_u(u_coeff)
            
            # 低Q值探索增强：对低Q值但未充分探索的走法给予额外探索奖励
            # 这有助于发现弃后连杀等隐藏走法（模型先验Q值不高，但实际可能是好走法）
            low_q_bonus = 0.0
            if self.params_.low_q_exploration_enabled:
                n_started = edge_and_node.get_n_started()
                threshold = self.params_.low_q_threshold
                p = edge_and_node.get_p()  # 获取先验概率
                
                # 对于低Q值走法，即使访问次数超过阈值，只要访问次数相对较少，仍然给予奖励
                # 这样可以持续探索弃后连杀等需要深度搜索才能发现的走法
                if q < threshold:
                    # 计算Q值因子：当Q=threshold时因子为0，当Q<<threshold时因子接近1
                    if threshold > 0:
                        # 正数阈值：线性映射，Q值越低因子越大
                        # 当q=threshold时，因子=0；当q<<threshold（如负数）时，因子接近1
                        q_factor = max(0.0, min(1.0, (threshold - q) / threshold))
                    elif threshold == 0:
                        # 零阈值：只对负数Q值给予奖励，因子与Q的绝对值成正比
                        # 使用一个合理的范围（如[-1, 0]）来归一化
                        q_factor = min(1.0, abs(q)) if q < 0 else 0.0
                    else:
                        # 负数阈值：对更负的Q值给予更大奖励
                        # 当q=threshold时，因子=0；当q<<threshold时，因子接近1
                        # 使用阈值到-1的范围来归一化
                        q_range = abs(threshold) + 1.0  # 假设Q值最小为-1
                        q_factor = max(0.0, min(1.0, (threshold - q) / q_range))
                    
                    # 访问次数因子：访问越少，因子越大
                    # 使用平滑衰减，即使超过阈值也给予一定奖励（但会衰减）
                    visit_threshold = self.params_.low_q_visit_threshold
                    if n_started < visit_threshold:
                        visit_factor = 1.0 - (n_started / max(visit_threshold, 1))
                    else:
                        # 超过阈值后使用指数衰减，确保即使访问次数较多也能获得一些奖励
                        # 衰减速度：访问次数每增加阈值的一半，因子减半
                        excess = n_started - visit_threshold
                        decay_rate = visit_threshold * 0.5  # 每增加阈值的一半，因子减半
                        visit_factor = 0.5 ** (excess / max(decay_rate, 1))
                    
                    # 先验概率补偿因子：P越小，补偿越大
                    # 对于低先验概率的走法，需要更大的奖励来补偿其U项的不足
                    # 使用反比例关系：p越小，补偿因子越大（但限制上限避免过度补偿）
                    p_compensation = 1.0 / max(p, 0.001)  # 避免除零，限制最小P为0.001
                    p_compensation = min(p_compensation, 10.0)  # 限制最大补偿为10倍
                    
                    # 访问次数差距补偿：如果当前走法访问次数远少于其他走法，给予额外奖励
                    # 这有助于在访问次数差距很大时（如11 vs 9441）仍然能够探索低Q值走法
                    max_sibling_visits = 0
                    for j in range(node.get_num_edges()):
                        if j != i:
                            sibling_edge = node.get_edge_at_index(j)
                            sibling_child = node.get_child_at_index(j)
                            sibling_edge_and_node = EdgeAndNode(sibling_edge, sibling_child)
                            max_sibling_visits = max(max_sibling_visits, sibling_edge_and_node.get_n_started())
                    
                    # 如果当前走法访问次数远少于其他走法，给予额外奖励
                    # 访问次数差距越大，奖励越大（使用对数缩放避免过度补偿）
                    visit_gap_bonus = 0.0
                    if max_sibling_visits > 0 and max_sibling_visits > n_started * 2:  # 至少差距2倍才给予奖励
                        visit_gap_ratio = max_sibling_visits / max(n_started, 1)
                        # 使用对数缩放：差距越大，奖励越大，但增长放缓
                        # 对于11 vs 9441的情况，ratio = 9441/11 ≈ 858，log10(859) ≈ 2.93
                        visit_gap_factor = min(3.0, math.log(visit_gap_ratio + 1) / math.log(10))  # 限制在3倍以内，适应更大的差距
                        visit_gap_bonus = self.params_.low_q_exploration_bonus * 0.5 * visit_gap_factor
                    
                    # 基础奖励：考虑Q值因子和访问次数因子
                    base_bonus = self.params_.low_q_exploration_bonus * q_factor * visit_factor
                    # 最终奖励：基础奖励乘以先验概率补偿因子，再加上访问次数差距奖励
                    # 但为了平衡，先验概率补偿只应用一部分（比如平方根），避免过度补偿
                    low_q_bonus = base_bonus * math.sqrt(p_compensation) + visit_gap_bonus
            
            u += low_q_bonus
            score = q + u + m
            candidates.append((score, edge_and_node, i, (q, u, m)))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda item: item[0], reverse=True)
        
        # 智能分配访问次数：计算应该为最佳节点分配多少次访问
        # 这可以减少重复的UCT计算，提高效率（类似C++代码的批量分配机制）
        best_score, best_edge_and_node, best_edge_idx, best_components = candidates[0]
        best_q, best_u, best_m = best_components
        best_without_u = best_q + best_m  # Q + M，不含U项
        
        # 计算应该分配多少次访问
        estimated_visits = 1  # 默认分配1次
        if len(candidates) > 1:
            second_best_score, _, _, _ = candidates[1]
            # 只有当最佳节点的Q+M小于第二佳节点的总分数时，才需要计算估计访问次数
            # 因为如果best_without_u >= second_best_score，说明即使U项衰减，最佳节点仍然会保持最佳
            if best_without_u < second_best_score:
                p = best_edge_and_node.get_p()
                n_started = best_edge_and_node.get_n_started()
                
                # 公式：k ≈ P * U_coeff / (second_best - best_without_u) - n_started #这个很容易推出来，必须是>1且为整数
                # 其中 U_coeff = cpuct * sqrt(N_parent)
                score_diff = second_best_score - best_without_u
                if score_diff > 1e-9:  # 避免除零
                    estimated_visits = max(1, int(
                        p * u_coeff / score_diff - n_started + 1
                    ))
                    # 限制最大分配次数，避免过度分配（最多不超过target_minibatch_size）
                    estimated_visits = min(estimated_visits, self.params_.target_minibatch_size)
        
        # 选择最佳节点，并记录估计的访问次数
        for score, edge_and_node, edge_idx, components in candidates:
            move = edge_and_node.get_move()
            if move is None:
                continue
            
            # 只为最佳节点使用智能分配的访问次数
            multivisit = estimated_visits if edge_idx == best_edge_idx else 1
            
            self._trace_edge_selection(node, edge_and_node, score, components, 'pick_node')
            history.append(move)
            
            if edge_and_node.node() is None:
                # 子节点尚未创建，需要创建
                board = chess.Board(node.get_fen())
                if move not in board.legal_moves:
                    history.pop()
                    continue
                board.push(move)
                child_fen = board.fen()
                child_node = Node(fen=child_fen, parent=node, index=edge_idx)
                node.children_[edge_idx] = child_node
                edge_and_node.edge_.node_ = child_node
                
                new_path = path + [(child_node, 0, 0)]
                history.pop()
                
                if self.params_.max_depth > 0 and len(new_path) >= self.params_.max_depth:
                    continue
                
                if child_node.try_start_score_update():
                    # 创建NodeToProcess时，设置multivisit（智能分配的访问次数）
                    node_to_process = NodeToProcess.visit(new_path, history)
                    node_to_process.multivisit = multivisit
                    return node_to_process
                continue
            
            # 子节点已存在，递归选择
            child_node = edge_and_node.node()
            new_path = path + [(child_node, 0, 0)]
            result = self.pick_node_to_extend(child_node, new_path, history)
            history.pop()
            if result is not None:
                # 如果是最佳节点，设置multivisit（智能分配的访问次数）
                # 注意：这里只在找到叶子节点时才设置，因为递归调用发生在子节点上
                if edge_idx == best_edge_idx:
                    result.multivisit = multivisit
                return result
        
        return None
    
    def _compute_position_hash(self, fen: str, cache_history_length: int = 1) -> int:
        """计算位置的哈希值（简化实现，使用 FEN 字符串的哈希）"""
        # 在完整实现中，应该使用 Zobrist 哈希或类似方法
        # 这里使用 FEN 字符串的哈希作为简化实现
        return hash(fen)
    
    def extend_node(self, node_to_process: NodeToProcess) -> None:
        """扩展节点，创建边和获取神经网络评估"""
        node = node_to_process.node
        board = chess.Board(node.get_fen())
        
        # 生成合法移动
        legal_moves = list(board.legal_moves)
        
        # 检查游戏结束
        if not legal_moves:
            if board.is_checkmate():
                # 输棋：如果当前是白方，结果对白方是 -1.0；如果是黑方，结果对黑方是 -1.0
                # 由于节点存储的是从"刚移动的一方"的视角，所以需要翻转
                result = -1.0 if board.turn == chess.WHITE else 1.0
                node.make_terminal(result, 0.0, 'end_of_game')
            else:
                # 和棋
                node.make_terminal(0.0, 0.0, 'end_of_game')
            return
        
        # 检查重复（简化实现）
        if node_to_process.repetitions >= 2:
            node.make_terminal(0.0, node_to_process.moves_left, 'twofold')
            return
        
        # 检查置换表
        position_hash = self._compute_position_hash(node.get_fen())
        node_to_process.hash = position_hash
        
        # 首先检查置换表
        tt_low_node = self.tt_.get(position_hash)
        if tt_low_node:
            # 命中置换表，重用 LowNode
            node_to_process.is_tt_hit = True
            node_to_process.tt_low_node = tt_low_node
            node_to_process.nn_queried = True
            
            # 从 LowNode 获取边，建立移动 UCI 到策略的映射
            move_to_policy = {}
            for edge in tt_low_node.get_edges():
                move_to_policy[edge.get_move().uci()] = edge.get_p()
            
            # 根据当前节点的合法移动，从映射中获取策略
            # 如果某个移动不在映射中，使用默认策略（均匀分布）
            policies = []
            default_policy = 1.0 / len(legal_moves) if legal_moves else 0.0
            for move in legal_moves:
                policy = move_to_policy.get(move.uci(), default_policy)
                policies.append(policy)
            
            # 归一化策略（确保总和为1）åå
            total_policy = sum(policies)
            if total_policy > 0:
                policies = [p / total_policy for p in policies]
            else:
                policies = [1.0 / len(legal_moves)] * len(legal_moves)
            
            # 创建边（使用当前节点的合法移动和匹配的策略）
            node.create_edges(legal_moves, policies)
            node.set_low_node(tt_low_node)
            
            # 从 LowNode 获取评估结果
            node_to_process.eval = {
                'q': tt_low_node.get_wl(),
                'd': tt_low_node.get_d(),
                'm': tt_low_node.get_m(),
                'p': policies
            }
            self._trace_expansion(node, legal_moves, policies, True)
            return
        
        # 未命中置换表，需要计算神经网络评估
        node_to_process.is_tt_hit = False
        
        # 获取神经网络评估
        eval_result = self.backend_.evaluate(node.get_fen(), legal_moves)
        
        # 创建新的 LowNode
        low_node = LowNode(legal_moves)
        policies = eval_result.get('p', [1.0 / len(legal_moves)] * len(legal_moves))
        
        # 确保策略长度匹配
        if len(policies) != len(legal_moves):
            policies = [1.0 / len(legal_moves)] * len(legal_moves)
        
        low_node.set_nneval(
            eval_result.get('q', 0.0),
            eval_result.get('d', 0.0),
            eval_result.get('m', 0.0),
            policies
        )
        
        # 创建边
        node.create_edges(legal_moves, policies)
        node.set_low_node(low_node)
        
        # 存储到置换表
        self.tt_.put(position_hash, low_node)
        
        # 存储评估结果
        node_to_process.eval = eval_result
        node_to_process.tt_low_node = low_node
        node_to_process.nn_queried = True
        self._trace_expansion(node, legal_moves, policies, False)
    
    def do_backup_update(self, node_to_process: NodeToProcess) -> None:
        """反向传播更新"""
        path = node_to_process.path
        multivisit = node_to_process.multivisit
        
        if not path:
            return
        
        # 获取节点的评估值
        node = node_to_process.node
        
        if node.is_terminal():
            # 终端节点：使用节点存储的值
            v = node.get_wl()
            d = node.get_d()
            m = node.get_m()
        elif node_to_process.tt_low_node:
            # 从 LowNode 获取（需要翻转，因为 LowNode 是从对手视角）
            # LowNode 存储的是从"即将移动的一方"的视角
            # Node 存储的是从"刚移动的一方"的视角
            low_node = node_to_process.tt_low_node
            v = -low_node.get_wl()  # 翻转 WL
            d = low_node.get_d()
            m = low_node.get_m() + 1.0  # 增加一步
        else:
            # 改成直接报错好了
            raise RuntimeError(
                f"无法获取节点评估值：节点不是终端节点，且没有 LowNode。"
                f"节点 FEN: {node.get_fen()}, "
                f"节点访问次数: {node.get_n()}, "
                f"是否可扩展: {node_to_process.is_extendable()}, "
                f"是否命中置换表: {node_to_process.is_tt_hit}, "
                f"是否查询神经网络: {node_to_process.nn_queried}"
            )#  
        
        # 向上传播（从叶子节点到根节点）
        for i in range(len(path) - 1, -1, -1):
            n, repetitions, moves_left = path[i]
            
            # 更新节点
            n.finalize_score_update(v, d, m, multivisit)
            
            # 如果有 LowNode，也需要更新
            low_node = n.get_low_node()
            if low_node:
                # 从节点视角转换到 LowNode 视角（翻转）
                low_node.finalize_score_update(-v, d, max(0.0, m - 1.0), multivisit)
            
            # 翻转值（因为从对手视角）
            if i > 0:
                v = -v
                m = m + 1.0
        
        self.total_playouts_ += multivisit
        self.cum_depth_ += len(path) * multivisit
        self.max_depth_ = max(self.max_depth_, len(path))
    
    def execute_one_iteration(self) -> None:
        """执行一次迭代"""
        if self.stop_:
            return
        
        # 1. 收集一批节点
        minibatch: List[NodeToProcess] = []
        
        with self.nodes_mutex_:
            history = PositionHistory(self.root_node_.get_fen())
            target_batch_size = max(1, self.params_.target_minibatch_size)
            max_pick_attempts = max(target_batch_size * 4, 32)
            pick_attempts = 0
            while (
                not self.stop_
                and len(minibatch) < target_batch_size
                and pick_attempts < max_pick_attempts
            ):
                node_to_process = self.pick_node_to_extend( # 先选一个节点扩展，用score = q+u+m。每次都是从root开始
                    self.root_node_,
                    [(self.root_node_, 0, 0)],
                    history,
                )
                pick_attempts += 1
                if node_to_process is None:
                    continue
                minibatch.append(node_to_process)


        # search stopper 3
        # 如果没有任何可扩展的节点，说明在达到max_depth限制
        # stop here
        if not minibatch:
            self.stop_ = True
            return
        
        # 2. 扩展节点
        for node_to_process in minibatch:
            if node_to_process.is_extendable():
                self.extend_node(node_to_process)
        
        # 3. 反向传播
        with self.nodes_mutex_:
            for node_to_process in minibatch:
                if not node_to_process.is_collision():
                    self.do_backup_update(node_to_process)
                    
                    # 更新当前最佳边
                    if self.root_node_.get_n() > 0:
                        self.current_best_edge_ = self.get_best_child_no_temperature(self.root_node_, 0)
        
        # 4. 检查停止条件
        if self.total_playouts_ >= self.params_.max_playouts:
            self.stop_ = True
    
    def run_blocking(self) -> None: 
        while not self.stop_:
            self.execute_one_iteration()
            if self.total_playouts_ >= self.params_.max_playouts: # search stopper 2
                break
    
    def stop(self) -> None: # search stopper 1, externally triggered
        """停止搜索"""
        self.stop_ = True
    
    def get_best_move(self) -> Optional[chess.Move]:
        """获取最佳移动"""
        with self.nodes_mutex_:
            if self.current_best_edge_ and self.current_best_edge_.edge():
                return self.current_best_edge_.get_move(self.played_history_.is_black_to_move())
            elif self.root_node_.has_children():
                best = self.get_best_child_no_temperature(self.root_node_, 0)
                if best and best.edge():
                    return best.get_move(self.played_history_.is_black_to_move())
        return None
    
    def get_total_playouts(self) -> int:
        """获取总访问次数"""
        return self.total_playouts_
    
    def get_max_depth_limit(self) -> int:
        """获取配置的最大深度限制（0 表示不限制）"""
        return self.params_.max_depth
    
    def get_current_max_depth(self) -> int:
        """获取当前实际达到的最大深度"""
        return self.max_depth_

    def export_trace_json(
        self,
        output_dir: str,
        max_edges: Optional[int] = 1000,
        slug: Optional[str] = None,
    ) -> str:
        """
        导出搜索追踪数据到 JSON 文件，自动包含 metadata
        
        Args:
            output_dir: 输出目录；文件名会根据 slug 自动生成
            max_edges: 最大边记录数，0 或 None 表示不限制（保存完整搜索树）
            slug: 用于元数据/文件名的标识，未提供时默认使用 root FEN

        Returns:
            实际写入的文件路径
        """
        if not self.tracer_:
            raise RuntimeError("SearchTracer 未设置，无法导出追踪数据")
        
        if not output_dir:
            raise ValueError("输出目录 output_dir 不能为空")
        def _sanitize(text: str) -> str:
            return (
                text.strip()
                .replace(" ", "_")
                .replace("/", "_")
                .replace(":", "_")
            )

        safe_slug = _sanitize(slug) if slug else _sanitize(self.root_node_.get_fen())
        depth_component = self.params_.max_depth if self.params_.max_depth > 0 else "inf"
        filename = f"search_trace_{safe_slug}_p{self.params_.max_playouts}_d{depth_component}.json"
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir_path / filename)
        
        # 自动收集 metadata
        best_move = self.get_best_move()
        all_edge_records = self.tracer_.get_edge_records()
        expansion_records = self.tracer_.get_expansion_records()
        
        # 计算实际保存的边数（考虑 max_edges 限制）
        if max_edges is None or max_edges == 0:
            actual_saved_edges = len(all_edge_records)
        else:
            actual_saved_edges = min(len(all_edge_records), max_edges)
        
        metadata = {
            "root_fen": self.root_node_.get_fen(),
            "search_params": {
                "max_playouts": self.params_.max_playouts,
                "target_minibatch_size": self.params_.target_minibatch_size,
                "cpuct": self.params_.cpuct,
                "max_depth": self.params_.max_depth,
                "moves_left_slope": self.params_.moves_left_slope,
                "moves_left_max_effect": self.params_.moves_left_max_effect,
                "fpu_value": self.params_.fpu_value,
                "fpu_absolute": self.params_.fpu_absolute,
                "draw_score": self.params_.draw_score,
            },
            "search_results": {
                "best_move": str(best_move) if best_move else None,
                "total_playouts": self.total_playouts_,
                "max_depth": self.max_depth_,  # 实际达到的最大深度
                "max_depth_limit": self.params_.max_depth,  # 配置的最大深度限制
            },
            "trace_stats": {
                "total_edge_records": len(all_edge_records),  # 总记录数
                "saved_edge_records": actual_saved_edges,  # 实际保存的边数
                "num_expansion_records": len(expansion_records),
                "max_edges_limit": max_edges if (max_edges is not None and max_edges > 0) else None,  # 边数限制（None表示无限制）
            },
            "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        
        # 调用 tracer 的 export_json 方法
        self.tracer_.export_json(output_path, max_edges=max_edges, metadata=metadata)
        return output_path

    def _trace_edge_selection(
        self,
        parent: Node,
        edge: EdgeAndNode,
        score: float,
        components: Tuple[float, float, float],
        stage: str,
    ) -> None:
        """记录一次边选择"""
        if not self.tracer_:
            return
        q, u, m = components
        self.tracer_.log_selection(parent, edge, score, q, u, m, stage)

    def _trace_expansion(
        self,
        node: Node,
        legal_moves: List[chess.Move],
        policies: List[float],
        is_tt_hit: bool,
    ) -> None:
        """记录一次节点扩展"""
        if not self.tracer_:
            return
        self.tracer_.log_expansion(node, legal_moves, policies, is_tt_hit)
