
import chess
from typing import Optional, List
import math


class Edge:

    def __init__(self, move: chess.Move, p: float = 0.0):
        self.move_ = move
        self.p_ = max(0.0, min(1.0, p))
        self.node_: Optional['Node'] = None
    
    def get_move(self, as_opponent: bool = False) -> chess.Move:
        return self.move_
    
    def get_p(self) -> float:
        return self.p_
    
    def set_p(self, val: float) -> None:
        self.p_ = max(0.0, min(1.0, val))
    
    @staticmethod
    def sort_edges(edges: List['Edge']) -> None:
        edges.sort(key=lambda e: e.get_p(), reverse=True)


class Node:
    def __init__(
        self,
        fen: str,
        parent: Optional['Node'] = None,
        index: int = 0,
        wl: float = 0.0,
        d: float = 0.0,
        m: float = 0.0,
    ):
        self.fen_ = fen
        self.parent_ = parent # parent is a node
        self.index_ = index # index is the index of the edge in the parent's edges list
        
        self.wl_ = wl # win-loss value
        self.d_ = d # draw probability
        self.m_ = m # moves left estimation
        
        self.n_ = 0 # visit count
        self.n_in_flight_ = 0
        
        self.edges_: list[Edge] = []
        self.children_: list[Optional['Node']] = []
        
        self.terminal_type_: Optional[str] = None # None, 'end_of_game', 'tablebase', 'twofold'
        self.lower_bound_: Optional[float] = None # lower bound of the node
        self.upper_bound_: Optional[float] = None # upper bound of the node
        
        self.low_node_: Optional['LowNode'] = None  # LowNode is a low-level node that is used to store the value of the node
    
    def get_fen(self) -> str:
        return self.fen_
    
    def get_parent(self) -> Optional['Node']:
        return self.parent_
    
    def has_children(self) -> bool:
        return len(self.edges_) > 0
    
    def get_n(self) -> int:
        return self.n_
    
    def get_n_in_flight(self) -> int:
        return self.n_in_flight_
    
    def get_children_visits(self) -> int:
        return self.n_ - 1 if self.n_ > 0 else 0
    
    def get_n_started(self) -> int:
        return self.n_ + self.n_in_flight_
    
    def get_q(self, draw_score: float = 0.0) -> float:
        return self.wl_ + draw_score * self.d_
    
    def get_wl(self) -> float:
        return self.wl_
    
    def get_d(self) -> float:
        return self.d_
    
    def get_m(self) -> float:
        return self.m_
    
    def is_terminal(self) -> bool:
        return self.terminal_type_ is not None
    
    def get_num_edges(self) -> int:
        return len(self.edges_)
    
    def create_edges(self, moves: list[chess.Move], policies: Optional[list[float]] = None) -> None:
        assert len(self.edges_) == 0, "Node already has edges, cannot create edges"
        
        self.edges_ = []
        self.children_ = []
        
        if policies is None:
            policies = [0.0] * len(moves)
        
        assert len(moves) == len(policies), "Move list and policy list length must be the same"
        
        for move, p in zip(moves, policies):
            edge = Edge(move, p)
            edge.node_ = None  # initially the child node is not created
            self.edges_.append(edge)
            self.children_.append(None)
    
    def create_single_child_node(self, move: chess.Move) -> 'Node':
        assert len(self.edges_) == 0, "Node already has edges, cannot create single child node"
        
        # create edge
        edge = Edge(move, 0.0)
        self.edges_ = [edge]
        
        # create child node's FEN
        board = chess.Board(self.fen_)
        board.push(move)
        child_fen = board.fen()
        
        # create child node
        child = Node(
            fen=child_fen,
            parent=self,
            index=0,
        )
        
        self.children_ = [child]
        edge.node_ = child
        
        return child
    
    def get_edge_to_node(self, node: 'Node') -> Optional[Edge]:
        """get the edge pointing to the specified child node"""
        for i, child in enumerate(self.children_):
            if child == node:
                return self.edges_[i]
        return None
    
    def get_own_edge(self) -> Optional[Edge]:
        """get the edge pointing to the current node (from the parent node)"""
        if self.parent_ is None:
            return None
        if self.index_ < len(self.parent_.edges_):
            return self.parent_.edges_[self.index_]
        return None
    
    def finalize_score_update(self, v: float, d: float, m: float, multivisit: int = 1) -> None:
        """update the node's evaluation value and visit count"""
        # update the evaluation value (weighted average)
        if self.n_ > 0:
            self.wl_ = (self.wl_ * self.n_ + v * multivisit) / (self.n_ + multivisit)
            self.d_ = (self.d_ * self.n_ + d * multivisit) / (self.n_ + multivisit)
            self.m_ = (self.m_ * self.n_ + m * multivisit) / (self.n_ + multivisit)
        else:
            self.wl_ = v
            self.d_ = d
            self.m_ = m
        
        # update the visit count
        self.n_ += multivisit
        self.n_in_flight_ = max(0, self.n_in_flight_ - multivisit)
    
    def increment_n_in_flight(self, multivisit: int = 1) -> None:
        """increment the number of in-flight visits (virtual loss)"""
        self.n_in_flight_ += multivisit
    
    def try_start_score_update(self) -> bool:
        """try to start score update"""
        if self.n_ == 0 and self.n_in_flight_ == 0:
            self.n_in_flight_ = 1
            return True
        return False
    
    def cancel_score_update(self, multivisit: int = 1) -> None:
        """cancel score update"""
        self.n_in_flight_ = max(0, self.n_in_flight_ - multivisit)
    
    def make_terminal(self, result: float, plies_left: float = 0.0, terminal_type: str = 'end_of_game') -> None:
        """mark the node as terminal and set the score"""
        self.terminal_type_ = terminal_type
        self.wl_ = result
        self.d_ = 1.0 if result == 0.0 else 0.0
        self.m_ = plies_left
    
    def make_not_terminal(self) -> None:
        """mark the node as non-terminal"""
        self.terminal_type_ = None
    
    def release_children(self) -> None:
        """release all children"""
        self.edges_ = []
        self.children_ = []
    
    def get_m(self) -> float:
        """get moves left value"""
        return self.m_
    
    def get_total_visits(self) -> int:
        """get total visits (including in-flight)"""
        return self.n_ + self.n_in_flight_
    
    def get_visited_policy(self) -> float:
        """get the sum of the policy probabilities of the visited edges"""
        total = 0.0
        for i, child in enumerate(self.children_):
            if child is not None and child.get_n() > 0:
                total += self.edges_[i].get_p()
        return total
    
    def set_bounds(self, lower: float, upper: float) -> None:
        """set the bounds"""
        self.lower_bound_ = lower
        self.upper_bound_ = upper
    
    def get_bounds(self) -> tuple[Optional[float], Optional[float]]:
        """get the bounds"""
        return (self.lower_bound_, self.upper_bound_)
    
    def is_tb_terminal(self) -> bool:
        """check if the node is a tablebase terminal node"""
        return self.terminal_type_ == 'tablebase'
    
    def get_low_node(self) -> Optional['LowNode']:
        """get the LowNode"""
        return self.low_node_
    
    def set_low_node(self, low_node: Optional['LowNode']) -> None:
        """set the LowNode"""
        if self.low_node_:
            self.low_node_.remove_parent()
        self.low_node_ = low_node
        if low_node:
            low_node.add_parent()
    
    def unset_low_node(self) -> None:
        """clear the LowNode"""
        if self.low_node_:
            self.low_node_.remove_parent()
            self.low_node_ = None
    
    def zero_n_in_flight(self) -> bool:
        """check if all nodes' n_in_flight is 0 (for debugging)"""
        if self.n_in_flight_ != 0:
            return False
        for child in self.children_:
            if child is not None and not child.zero_n_in_flight():
                return False
        return True
    
    def get_child_at_index(self, index: int) -> Optional['Node']:
        """get the child node at the specified index"""
        if 0 <= index < len(self.children_):
            return self.children_[index]
        return None
    
    def get_edge_at_index(self, index: int) -> Optional[Edge]:
        """get the edge at the specified index"""
        if 0 <= index < len(self.edges_):
            return self.edges_[index]
        return None
    
    def sort_edges(self) -> None:
        """sort the edges by policy probability"""
        indexed_edges = list(enumerate(self.edges_))
        # sort by policy probability in descending order
        indexed_edges.sort(key=lambda x: x[1].get_p(), reverse=True)
        # rearrange the edges and children
        sorted_edges = []
        sorted_children = []
        for idx, edge in indexed_edges:
            sorted_edges.append(edge)
            sorted_children.append(self.children_[idx])
            # update the index of the child node
            if sorted_children[-1] is not None:
                sorted_children[-1].index_ = len(sorted_children) - 1
        self.edges_ = sorted_edges
        self.children_ = sorted_children


class LowNode:
    """low node class, stores the neural network evaluation results and the array of edges"""
    
    def __init__(
        self,
        moves: Optional[List[chess.Move]] = None,
        edges: Optional[List[Edge]] = None,
    ):
        self.edges_: List[Edge] = edges if edges else []
        if moves:
            self.edges_ = [Edge(move, 0.0) for move in moves]
        
        self.wl_: float = 0.0  # Win-Loss value
        self.d_: float = 0.0   # Draw probability
        self.m_: float = 0.0   # Moves left
        
        self.n_: int = 0  # visit count
        self.num_parents_: int = 0  # parent node count
        self.is_transposition: bool = False
        
        self.terminal_type_: Optional[str] = None
        self.lower_bound_: Optional[float] = None
        self.upper_bound_: Optional[float] = None
        
        self.child_: Optional[Node] = None  # the first child node
    
    def get_num_edges(self) -> int:
        """get the number of edges"""
        return len(self.edges_)
    
    def get_edges(self) -> List[Edge]:
        """get the list of edges"""
        return self.edges_
    
    def get_wl(self) -> float:
        """get the Win-Loss value"""
        return self.wl_
    
    def get_d(self) -> float:
        """get the Draw probability"""
        return self.d_
    
    def get_m(self) -> float:
        """get the Moves left"""
        return self.m_
    
    def get_n(self) -> int:
        """get the number of visits"""
        return self.n_
    
    def get_children_visits(self) -> int:
        """get the number of visits to the children nodes"""
        return self.n_ - 1 if self.n_ > 0 else 0
    
    def is_terminal(self) -> bool:
        """check if the node is a terminal node"""
        return self.terminal_type_ is not None
    
    def get_bounds(self) -> tuple[Optional[float], Optional[float]]:
        """get the bounds"""
        return (self.lower_bound_, self.upper_bound_)
    
    def get_terminal_type(self) -> Optional[str]:
        """get the terminal type"""
        return self.terminal_type_
    
    def is_transposition(self) -> bool:
        """check if the node is a transposition"""
        return self.is_transposition
    
    def set_nneval(self, q: float, d: float, m: float, policies: List[float]) -> None:
        """set the neural network evaluation results"""
        assert self.n_ == 0, "LowNode already has visits, cannot set evaluation results"
        assert len(policies) == len(self.edges_), "the number of policies does not match the number of edges"
        
        for i, p in enumerate(policies):
            self.edges_[i].set_p(p)
        
        self.wl_ = q
        self.d_ = d
        self.m_ = m
    
    def make_terminal(self, result: float, plies_left: float = 0.0, terminal_type: str = 'end_of_game') -> None:
        """mark the node as a terminal node"""
        self.terminal_type_ = terminal_type
        self.wl_ = result
        self.d_ = 1.0 if result == 0.0 else 0.0
        self.m_ = plies_left
    
    def set_bounds(self, lower: float, upper: float) -> None:
        """set the bounds"""
        self.lower_bound_ = lower
        self.upper_bound_ = upper
    
    def finalize_score_update(self, v: float, d: float, m: float, multivisit: int = 1) -> None: # important logic, when to use this function?
        """finalize the score update"""
        if self.n_ > 0:
            self.wl_ = (self.wl_ * self.n_ + v * multivisit) / (self.n_ + multivisit)
            self.d_ = (self.d_ * self.n_ + d * multivisit) / (self.n_ + multivisit)
            self.m_ = (self.m_ * self.n_ + m * multivisit) / (self.n_ + multivisit)
        else:
            self.wl_ = v
            self.d_ = d
            self.m_ = m
        self.n_ += multivisit
    
    def cancel_score_update(self, multivisit: int = 1) -> None:
        """cancel the score update (empty implementation, LowNode does not need virtual loss)"""
        pass
    
    def add_parent(self) -> None:
        """add a parent node"""
        self.num_parents_ += 1
        if self.num_parents_ > 1:
            self.is_transposition = True
    
    def remove_parent(self) -> None:
        """remove a parent node"""
        assert self.num_parents_ > 0
        self.num_parents_ -= 1
    
    def sort_edges(self) -> None:
        """sort the edges by policy probability"""
        Edge.sort_edges(self.edges_)
    
    def get_child(self) -> Optional[Node]:
        """get the first child node"""
        return self.child_
    
    def set_child(self, child: Optional[Node]) -> None:
        """set the first child node"""
        self.child_ = child


class EdgeAndNode:
    """edge and node combination class, simplified access"""
    
    def __init__(self, edge: Optional[Edge] = None, node: Optional[Node] = None):
        self.edge_ = edge
        self.node_ = node
    
    def reset(self) -> None:
        """reset to empty"""
        self.edge_ = None
        self.node_ = None
    
    def __bool__(self) -> bool:
        """check if valid"""
        return self.edge_ is not None
    
    def __eq__(self, other) -> bool:
        """compare equal"""
        if not isinstance(other, EdgeAndNode):
            return False
        return self.edge_ == other.edge_ and self.node_ == other.node_
    
    def has_node(self) -> bool:
        """check if has node"""
        return self.node_ is not None
    
    def edge(self) -> Optional[Edge]:
        """get the edge"""
        return self.edge_
    
    def node(self) -> Optional[Node]:
        """get the node"""
        return self.node_
    
    def get_q(self, default_q: float, draw_score: float) -> float:
        """get the Q value"""
        if self.node_ and self.node_.get_n() > 0:
            return self.node_.get_q(draw_score)
        return default_q
    
    def get_wl(self, default_wl: float) -> float:
        """get the Win-Loss value"""
        if self.node_ and self.node_.get_n() > 0:
            return self.node_.get_wl()
        return default_wl
    
    def get_d(self, default_d: float) -> float:
        """get the Draw probability"""
        if self.node_ and self.node_.get_n() > 0:
            return self.node_.get_d()
        return default_d
    
    def get_m(self, default_m: float) -> float:
        """get the Moves left"""
        if self.node_ and self.node_.get_n() > 0:
            return self.node_.get_m()
        return default_m
    
    def get_n(self) -> int:
        """get the number of visits"""
        return self.node_.get_n() if self.node_ else 0
    
    def get_n_started(self) -> int:
        """get the number of started visits"""
        return self.node_.get_n_started() if self.node_ else 0
    
    def get_n_in_flight(self) -> int:
        """get the number of in-flight visits"""
        return self.node_.get_n_in_flight() if self.node_ else 0
    
    def is_terminal(self) -> bool:
        """check if the node is a terminal node"""
        return self.node_.is_terminal() if self.node_ else False
    
    def is_tb_terminal(self) -> bool:
        """check if the node is a tablebase terminal node"""
        return self.node_.is_tb_terminal() if self.node_ else False
    
    def get_bounds(self) -> tuple[Optional[float], Optional[float]]:
        """get the bounds"""
        if self.node_:
            return self.node_.get_bounds()
        return (None, None)
    
    def get_p(self) -> float:
        """get the policy probability (from the edge)"""
        if self.edge_:
            return self.edge_.get_p()
        # if only node, get the edge from the parent node
        if self.node_ and self.node_.get_parent():
            parent = self.node_.get_parent()
            edge = parent.get_edge_to_node(self.node_)
            if edge:
                return edge.get_p()
        return 0.0
    
    def get_move(self, flip: bool = False) -> Optional[chess.Move]:
        """get the move"""
        if self.edge_:
            return self.edge_.get_move(flip)
        return None
    
    def get_u(self, numerator: float) -> float:
        """get the U value = numerator * p / (1 + n_started)"""
        p = self.get_p()
        n_started = self.get_n_started()
        return numerator * p / (1 + n_started)
