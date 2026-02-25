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
    from node import Node, Edge, LowNode, EdgeAndNode


def MakeRootMoveFilter(fen: str, searchmoves: List[str]) -> List[chess.Move]:
    """filter the legal moves of the root node"""
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
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return -1.0
    else:
        return 0.0


def fast_log(x: float) -> float:
    return math.log(x) if x > 0.0 else float('-inf')


def fast_logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class SearchParams:
    """search parameters"""
    # M evaluation parameters
    moves_left_slope: float = 0.0
    moves_left_max_effect: float = 0.0
    moves_left_constant_factor: float = 0.0
    moves_left_scaled_factor: float = 0.0
    moves_left_quadratic_factor: float = 0.0
    moves_left_threshold: float = 0.0
    
    # UCT parameters
    cpuct: float = 3.0
    cpuct_factor: float = 0.0
    cpuct_base: float = 19652.0
    fpu_value: float = 0.0
    fpu_absolute: bool = False
    
    # low Q value exploration enhancement parameters (for discovering hidden moves like suicide attacks)
    low_q_exploration_enabled: bool = False  # whether to enable low Q value exploration enhancement
    low_q_threshold: float = 0.3  # Q value threshold, below which is considered "low Q value"
    low_q_exploration_bonus: float = 0.1  # exploration bonus base value
    low_q_visit_threshold: int = 5  # visit threshold, below which is considered "not fully explored"
    
    # search control
    max_playouts: int = 10000
    target_minibatch_size: int = 8
    max_collision_visits: int = 1
    max_out_of_order: int = 10
    max_depth: int = 0  # maximum search depth, 0 means unlimited
    # note: smaller max_depth will explore more branches in the same max_playouts (because each path is shorter)
    # but the evaluation of each path may not be deep enough, so we need to balance the depth and breadth
    
    # temperature parameters
    temperature: float = 0.0
    temperature_cutoff_move: int = 0
    temperature_endgame: float = 0.0
    temperature_visit_offset: float = 0.0
    temperature_winpct_cutoff: float = 0.0
    
    # other
    draw_score: float = 0.0
    multipv: int = 1


class MEvaluator:
    """M evaluator, for calculating the utility value of Moves Left"""
    
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
        """set the parent node"""
        assert parent is not None, "Parent node cannot be None"
        if self.enabled:
            self.parent_m = parent.GetM()
            self.parent_within_threshold = self._within_threshold(parent, self.q_threshold)
    
    def get_m_utility(self, child: Node, q: float) -> float:
        """calculate the M utility value of the child node"""
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
        """calculate the M utility value of the edge"""
        if not self.enabled or not self.parent_within_threshold:
            return 0.0
        if edge.get_n() == 0:
            return self.get_default_m_utility()
        return self.get_m_utility(edge.node(), q)
    
    def get_default_m_utility(self) -> float:
        """return the default M utility value of the unvisited node"""
        return 0.0
    
    @staticmethod
    def _within_threshold(parent: Optional[Node], q_threshold: float) -> bool:
        """check if the Q value of the parent node is within the threshold"""
        if parent is None:
            return False
        return abs(parent.GetQ(0.0)) > q_threshold


# auxiliary functions
def get_fpu(params: SearchParams, node: Node, is_root_node: bool, draw_score: float) -> float:
    """calculate the First Play Urgency (FPU)"""
    value = params.fpu_value
    if params.fpu_absolute:
        return value
    else:
        return -node.get_q(-draw_score) - value * math.sqrt(node.get_visited_policy())


def compute_cpuct(params: SearchParams, n: int, is_root_node: bool) -> float:
    """calculate the UCT coefficient"""
    init = params.cpuct
    k = params.cpuct_factor
    base = params.cpuct_base
    if k == 0.0:
        return init
    return init + k * fast_log((n + base) / base)


@dataclass
class NodeToProcess:
    """node to process"""
    path: List[Tuple[Node, int, int]]  # (node, repetitions, moves left)
    node: Node
    multivisit: int = 1 # one return is considered as several visits, default 1, C++ supports setting larger when searching in parallel with collision, direct account for collision
    is_collision_flag: bool = False  # avoid naming conflict with method
    maxvisit: int = 0
    ooo_completed: bool = False # whether the NodeToProcess object has completed out-of-order evaluation
    nn_queried: bool = False # whether the NodeToProcess object has queried the neural network
    is_tt_hit: bool = False # whether the NodeToProcess object has hit the transposition table
    is_cache_hit: bool = False # whether the NodeToProcess object has hit the cache
    hash: Optional[int] = None
    tt_low_node: Optional[LowNode] = None
    eval: Optional[Dict[str, Any]] = None  # {'q': float, 'd': float, 'm': float, 'p': List[float]}
    history: Optional[Any] = None  # history of the position
    repetitions: int = 0
    moves_left: int = 0
    
    @staticmethod
    def visit(path: List[Tuple[Node, int, int]], history: Any) -> 'NodeToProcess':
        """create a visited node"""
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
        """create a collision node"""
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
        """check if it is a collision"""
        return self.is_collision_flag
    
    def is_extendable(self) -> bool:
        """check if the node is extendable"""
        return not self.is_collision_flag and not self.node.is_terminal() and self.node.get_n() == 0
    
    def can_eval_out_of_order(self) -> bool:
        """check if it can be evaluated out of order"""
        return False


@dataclass
class TraceEdgeRecord:
    """record the information of one edge selection"""
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
    """record the information of one node expansion"""
    node_fen: str
    move_uci_list: List[str]
    policies: List[float]
    is_tt_hit: bool
    timestamp: float


class SearchTracer:
    """Helper for recording selection and expansion during search."""

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
        """Log one edge selection."""
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
        """Log information for one node expansion."""
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
        """Get recorded edge selections (shallow copy)."""
        with self._lock:
            return self._edge_records.copy()

    def get_expansion_records(self) -> List[TraceExpansionRecord]:
        """Get recorded expansion info (shallow copy)."""
        with self._lock:
            return self._expansion_records.copy()

    def export_graphviz(
        self,
        output_path: str,
        max_edges: int = 200,
    ) -> None:
        """Export current trace to a graphviz graph file."""
        try:
            import graphviz
        except ImportError as exc:
            raise RuntimeError("graphviz is not installed; cannot export search graph.") from exc

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
        Export trace data to a JSON file with node/edge lists.

        Args:
            output_path: Output file path.
            max_edges: Max edge records; 0 or None means no limit (full tree).
            metadata: Optional metadata dict (search params, stats, etc.).
        """
        data: Dict[str, Any] = {
            "nodes": [],
            "edges": [],
            "metadata": metadata if metadata is not None else {},
        }
        with self._lock:
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
    """Position history for tracking game history."""

    def __init__(self, initial_fen: str = chess.Board().fen()):
        self.fens: List[str] = [initial_fen]
        self.moves: List[chess.Move] = []
    
    def append(self, move: chess.Move) -> None:
        """Append a move."""
        board = chess.Board(self.fens[-1])
        board.push(move)
        self.fens.append(board.fen())
        self.moves.append(move)
    
    def pop(self) -> None:
        """Remove the last move."""
        if self.fens:
            self.fens.pop()
        if self.moves:
            self.moves.pop()
    
    def last(self) -> chess.Board:
        """Get the last position."""
        return chess.Board(self.fens[-1])
    
    def get_length(self) -> int:
        """Get history length."""
        return len(self.fens)
    
    def trim(self, length: int) -> None:
        """Trim history to the given length."""
        if length < len(self.fens):
            self.fens = self.fens[:length]
            self.moves = self.moves[:length-1] if length > 0 else []
    
    def get_positions(self) -> List[str]:
        """Get all positions."""
        return self.fens.copy()
    
    def is_black_to_move(self) -> bool:
        """Check if it is Black's turn."""
        return self.last().turn == chess.BLACK


class Backend:
    """Backend interface for neural network evaluation."""

    def evaluate(self, fen: str, legal_moves: List[chess.Move]) -> Dict[str, Any]:
        """
        Evaluate a position.

        Returns:
            {'q': float, 'd': float, 'm': float, 'p': List[float]}
        """
        raise NotImplementedError


class SimpleBackend(Backend):
    """Simple backend implementation using the model interface."""

    def __init__(self, model_eval_fn: Optional[Callable[[str], Dict[str, Any]]] = None):
        self.model_eval_fn = model_eval_fn
        self.cache: Dict[str, Dict[str, Any]] = {}
        # self.cache:  {'8/5B2/2R5/1pP1p1k1/1P2Pb2/r5pK/8/8 b - - 2 50': {'q': 0.004424240440130234, 'd': 0.9480566382408142, 'm': 163.16165161132812, 'p': {'f4e3': 0.0625, 'f4d2': 0.0625...
    
    def evaluate(self, fen: str, legal_moves: List[chess.Move]) -> Dict[str, Any]:
        """Evaluate a position."""
        # print("self.model_eval_fn: ", self.model_eval_fn)
        # print("self.cache: ", self.cache)
        # print("fen: ", fen)
        # print("legal_moves: ", legal_moves)
        if fen in self.cache:
            cached = self.cache[fen]
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
        return {
            'q': 0.0,
            'd': 0.0,
            'm': 0.0,
            'p': [1.0 / len(legal_moves)] * len(legal_moves) if legal_moves else []
        }


class TranspositionTable:
    """Transposition table for storing and reusing LowNodes."""

    def __init__(self):
        self.table: Dict[int, LowNode] = {}
    
    def get(self, position_hash: int) -> Optional[LowNode]:
        """Get the LowNode for a position."""
        return self.table.get(position_hash)
    
    def put(self, position_hash: int, low_node: LowNode) -> None:
        """Store the LowNode for a position."""
        self.table[position_hash] = low_node
    
    def clear_expired(self) -> None:
        """Clear expired entries (simplified; Python GC handles cleanup)."""
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
        board = chess.Board(root_node.get_fen())
        self.played_history_ = PositionHistory(root_node.get_fen())
    
    def get_draw_score(self, is_odd_depth: bool) -> float:
        """Get draw score."""
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
        Get the best child for final move (aligned with C++ no-temperature selection).
        Priority: 1) proven wins (mate then TB win), 2) non-terminal by N then Q then P,
        3) proven losses (TB loss then normal), preferring longer losses. No U term in final pick.
        """
        best_list = self.get_best_children_no_temperature(parent, 1, depth)
        if not best_list:
            return EdgeAndNode()
        best_edge = best_list[0]
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
        u = 0.0
        score = q + m
        self._trace_edge_selection(parent, best_edge, score, (q, u, m), "best_child")
        return best_edge
     
    def get_best_children_no_temperature(  # no use
        self,
        parent: Node,
        count: int,
        depth: int
    ) -> List[EdgeAndNode]:
        """
        Return top candidates by visit count (C++ GetBestChildrenNoTemperature logic).
        Sort order: 1) terminal win (mate > TB, prefer faster mate), 2) non-terminal by N, Q, P,
        3) terminal loss (TB > normal), prefer longer loss (larger M).
        """
        if parent.get_n() == 0:
            return []

        is_odd_depth = (depth % 2) == 1
        draw_score = self.get_draw_score(is_odd_depth)
        fpu = get_fpu(self.params_, parent, parent == self.root_node_, draw_score)

        edges: List[EdgeAndNode] = []
        for i in range(parent.get_num_edges()):
            edge = parent.get_edge_at_index(i)
            child = parent.get_child_at_index(i)
            if parent == self.root_node_ and self.root_move_filter_:
                if edge.get_move() not in self.root_move_filter_:
                    continue

            edges.append(EdgeAndNode(edge, child))

        if not edges:
            return []

        def edge_rank(e: EdgeAndNode) -> int:
            """Match C++ EdgeRank: 0=terminal loss, 1=TB loss, 2=non-terminal/draw/unvisited, 3=TB win, 4=terminal win."""
            n = e.get_n()
            wl = e.get_wl(0.0)
            if n == 0 or (not e.is_terminal()) or wl == 0.0:
                return 2
            if e.is_tb_terminal():
                return 1 if wl < 0.0 else 3
            return 0 if wl < 0.0 else 4

        def sort_key(e: EdgeAndNode) -> Tuple[float, ...]:
            kind = edge_rank(e)
            n = e.get_n()
            q = e.get_q(0.0, draw_score)
            p = e.get_p()
            m = e.get_m(0.0)

            if kind == 2:
                return (kind, float(n), q, p)
            if kind >= 3:
                return (kind, -m)
            return (kind, m)

        edges.sort(key=sort_key, reverse=True)
        if count < len(edges):
            edges = edges[:count]
        return edges
    
    def pick_node_to_extend(self, node: Node, path: List[Tuple[Node, int, int]], history: PositionHistory) -> Optional[NodeToProcess]:
        """Pick node to extend (UCT selection, recursive)."""
        if node.is_terminal():
            return None
        current_depth = len(path)
        if self.params_.max_depth > 0 and current_depth >= self.params_.max_depth:
            return None
        if node.get_n() == 0:
            if node.try_start_score_update():
                return NodeToProcess.visit(path, history)
            return None
        if not node.has_children():
            if node.try_start_score_update():
                return NodeToProcess.visit(path, history)
            return None
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
            
            low_q_bonus = 0.0
            if self.params_.low_q_exploration_enabled:
                n_started = edge_and_node.get_n_started()
                threshold = self.params_.low_q_threshold
                p = edge_and_node.get_p()
                if q < threshold:
                    if threshold > 0:
                        q_factor = max(0.0, min(1.0, (threshold - q) / threshold))
                    elif threshold == 0:
                        q_factor = min(1.0, abs(q)) if q < 0 else 0.0
                    else:
                        q_range = abs(threshold) + 1.0
                        q_factor = max(0.0, min(1.0, (threshold - q) / q_range))
                    visit_threshold = self.params_.low_q_visit_threshold
                    if n_started < visit_threshold:
                        visit_factor = 1.0 - (n_started / max(visit_threshold, 1))
                    else:
                        excess = n_started - visit_threshold
                        decay_rate = visit_threshold * 0.5
                        visit_factor = 0.5 ** (excess / max(decay_rate, 1))
                    p_compensation = 1.0 / max(p, 0.001)
                    p_compensation = min(p_compensation, 10.0)
                    max_sibling_visits = 0
                    for j in range(node.get_num_edges()):
                        if j != i:
                            sibling_edge = node.get_edge_at_index(j)
                            sibling_child = node.get_child_at_index(j)
                            sibling_edge_and_node = EdgeAndNode(sibling_edge, sibling_child)
                            max_sibling_visits = max(max_sibling_visits, sibling_edge_and_node.get_n_started())
                    visit_gap_bonus = 0.0
                    if max_sibling_visits > 0 and max_sibling_visits > n_started * 2:
                        visit_gap_ratio = max_sibling_visits / max(n_started, 1)
                        visit_gap_factor = min(3.0, math.log(visit_gap_ratio + 1) / math.log(10))
                        visit_gap_bonus = self.params_.low_q_exploration_bonus * 0.5 * visit_gap_factor
                    base_bonus = self.params_.low_q_exploration_bonus * q_factor * visit_factor
                    low_q_bonus = base_bonus * math.sqrt(p_compensation) + visit_gap_bonus
            u += low_q_bonus
            score = q + u + m
            candidates.append((score, edge_and_node, i, (q, u, m)))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_edge_and_node, best_edge_idx, best_components = candidates[0]
        best_q, best_u, best_m = best_components
        best_without_u = best_q + best_m
        estimated_visits = 1
        if len(candidates) > 1:
            second_best_score, _, _, _ = candidates[1]
            if best_without_u < second_best_score:
                p = best_edge_and_node.get_p()
                n_started = best_edge_and_node.get_n_started()
                score_diff = second_best_score - best_without_u
                if score_diff > 1e-9:
                    estimated_visits = max(1, int(
                        p * u_coeff / score_diff - n_started + 1
                    ))
                    estimated_visits = min(estimated_visits, self.params_.target_minibatch_size)
        for score, edge_and_node, edge_idx, components in candidates:
            move = edge_and_node.get_move()
            if move is None:
                continue
            multivisit = estimated_visits if edge_idx == best_edge_idx else 1
            self._trace_edge_selection(node, edge_and_node, score, components, 'pick_node')
            history.append(move)
            if edge_and_node.node() is None:
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
                    node_to_process = NodeToProcess.visit(new_path, history)
                    node_to_process.multivisit = multivisit
                    return node_to_process
                continue
            child_node = edge_and_node.node()
            new_path = path + [(child_node, 0, 0)]
            result = self.pick_node_to_extend(child_node, new_path, history)
            history.pop()
            if result is not None:
                if edge_idx == best_edge_idx:
                    result.multivisit = multivisit
                return result
        
        return None
    
    def _compute_position_hash(self, fen: str, cache_history_length: int = 1) -> int:
        """Compute position hash (simplified: hash of FEN string)."""
        return hash(fen)

    def extend_node(self, node_to_process: NodeToProcess) -> None:
        """Extend node: create edges and get neural network evaluation."""
        node = node_to_process.node
        board = chess.Board(node.get_fen())
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            if board.is_checkmate():
                result = -1.0 if board.turn == chess.WHITE else 1.0
                node.make_terminal(result, 0.0, 'end_of_game')
            else:
                node.make_terminal(0.0, 0.0, 'end_of_game')
            return
        if node_to_process.repetitions >= 2:
            node.make_terminal(0.0, node_to_process.moves_left, 'twofold')
            return
        position_hash = self._compute_position_hash(node.get_fen())
        node_to_process.hash = position_hash
        tt_low_node = self.tt_.get(position_hash)
        if tt_low_node:
            node_to_process.is_tt_hit = True
            node_to_process.tt_low_node = tt_low_node
            node_to_process.nn_queried = True
            move_to_policy = {}
            for edge in tt_low_node.get_edges():
                move_to_policy[edge.get_move().uci()] = edge.get_p()
            policies = []
            default_policy = 1.0 / len(legal_moves) if legal_moves else 0.0
            for move in legal_moves:
                policy = move_to_policy.get(move.uci(), default_policy)
                policies.append(policy)
            total_policy = sum(policies)
            if total_policy > 0:
                policies = [p / total_policy for p in policies]
            else:
                policies = [1.0 / len(legal_moves)] * len(legal_moves)
            node.create_edges(legal_moves, policies)
            node.set_low_node(tt_low_node)
            node_to_process.eval = {
                'q': tt_low_node.get_wl(),
                'd': tt_low_node.get_d(),
                'm': tt_low_node.get_m(),
                'p': policies
            }
            self._trace_expansion(node, legal_moves, policies, True)
            return
        node_to_process.is_tt_hit = False
        eval_result = self.backend_.evaluate(node.get_fen(), legal_moves)
        low_node = LowNode(legal_moves)
        policies = eval_result.get('p', [1.0 / len(legal_moves)] * len(legal_moves))
        if len(policies) != len(legal_moves):
            policies = [1.0 / len(legal_moves)] * len(legal_moves)
        
        low_node.set_nneval(
            eval_result.get('q', 0.0),
            eval_result.get('d', 0.0),
            eval_result.get('m', 0.0),
            policies
        )
        node.create_edges(legal_moves, policies)
        node.set_low_node(low_node)
        self.tt_.put(position_hash, low_node)
        node_to_process.eval = eval_result
        node_to_process.tt_low_node = low_node
        node_to_process.nn_queried = True
        self._trace_expansion(node, legal_moves, policies, False)
    
    def do_backup_update(self, node_to_process: NodeToProcess) -> None:
        """Backup update (backpropagation)."""
        path = node_to_process.path
        multivisit = node_to_process.multivisit
        if not path:
            return
        node = node_to_process.node
        if node.is_terminal():
            v = node.get_wl()
            d = node.get_d()
            m = node.get_m()
        elif node_to_process.tt_low_node:
            low_node = node_to_process.tt_low_node
            v = -low_node.get_wl()
            d = low_node.get_d()
            m = low_node.get_m() + 1.0
        else:
            raise RuntimeError(
                f"Cannot get node evaluation: node is not terminal and has no LowNode. "
                f"FEN: {node.get_fen()}, n: {node.get_n()}, "
                f"extendable: {node_to_process.is_extendable()}, "
                f"tt_hit: {node_to_process.is_tt_hit}, nn_queried: {node_to_process.nn_queried}"
            )
        for i in range(len(path) - 1, -1, -1):
            n, repetitions, moves_left = path[i]
            n.finalize_score_update(v, d, m, multivisit)
            low_node = n.get_low_node()
            if low_node:
                low_node.finalize_score_update(-v, d, max(0.0, m - 1.0), multivisit)
            if i > 0:
                v = -v
                m = m + 1.0
        
        self.total_playouts_ += multivisit
        self.cum_depth_ += len(path) * multivisit
        self.max_depth_ = max(self.max_depth_, len(path))
    
    def execute_one_iteration(self) -> None:
        """Execute one iteration."""
        if self.stop_:
            return
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
                node_to_process = self.pick_node_to_extend(
                    self.root_node_,
                    [(self.root_node_, 0, 0)],
                    history,
                )
                pick_attempts += 1
                if node_to_process is None:
                    continue
                minibatch.append(node_to_process)
        if not minibatch:
            self.stop_ = True
            return
        for node_to_process in minibatch:
            if node_to_process.is_extendable():
                self.extend_node(node_to_process)
        with self.nodes_mutex_:
            for node_to_process in minibatch:
                if not node_to_process.is_collision():
                    self.do_backup_update(node_to_process)
                    if self.root_node_.get_n() > 0:
                        self.current_best_edge_ = self.get_best_child_no_temperature(self.root_node_, 0)
        if self.total_playouts_ >= self.params_.max_playouts:
            self.stop_ = True
    
    def run_blocking(self) -> None: 
        while not self.stop_:
            self.execute_one_iteration()
            if self.total_playouts_ >= self.params_.max_playouts: # search stopper 2
                break
    
    def stop(self) -> None:
        """Stop the search."""
        self.stop_ = True

    def get_best_move(self) -> Optional[chess.Move]:
        """Get the best move."""
        with self.nodes_mutex_:
            if self.current_best_edge_ and self.current_best_edge_.edge():
                return self.current_best_edge_.get_move(self.played_history_.is_black_to_move())
            elif self.root_node_.has_children():
                best = self.get_best_child_no_temperature(self.root_node_, 0)
                if best and best.edge():
                    return best.get_move(self.played_history_.is_black_to_move())
        return None
    
    def get_total_playouts(self) -> int:
        """Get total playouts."""
        return self.total_playouts_

    def get_max_depth_limit(self) -> int:
        """Get configured max depth limit (0 means unlimited)."""
        return self.params_.max_depth

    def get_current_max_depth(self) -> int:
        """Get current maximum depth reached."""
        return self.max_depth_

    def export_trace_json(
        self,
        output_dir: str,
        max_edges: Optional[int] = 1000,
        slug: Optional[str] = None,
    ) -> str:
        """
        Export search trace to a JSON file with metadata.

        Args:
            output_dir: Output directory; filename is derived from slug.
            max_edges: Max edge records; 0 or None means no limit.
            slug: Optional slug for metadata/filename; defaults to root FEN.

        Returns:
            Path of the written file.
        """
        if not self.tracer_:
            raise RuntimeError("SearchTracer not set; cannot export trace data.")
        if not output_dir:
            raise ValueError("output_dir must not be empty.")
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
        best_move = self.get_best_move()
        all_edge_records = self.tracer_.get_edge_records()
        expansion_records = self.tracer_.get_expansion_records()
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
                "max_depth": self.max_depth_,
                "max_depth_limit": self.params_.max_depth,
            },
            "trace_stats": {
                "total_edge_records": len(all_edge_records),
                "saved_edge_records": actual_saved_edges,
                "num_expansion_records": len(expansion_records),
                "max_edges_limit": max_edges if (max_edges is not None and max_edges > 0) else None,
            },
            "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
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
        """Log one edge selection."""
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
        """Log one node expansion."""
        if not self.tracer_:
            return
        self.tracer_.log_expansion(node, legal_moves, policies, is_tt_hit)
