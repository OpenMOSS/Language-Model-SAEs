"""国际象棋规则判断模块

提供各种国际象棋局面分析的规则函数。
"""

from .my_rook_under_attack import is_rook_under_attack, rook_under_attack_details
from .my_knight_under_attack import is_knight_under_attack, knight_under_attack_details
from .my_bishop_under_attack import is_bishop_under_attack, bishop_under_attack_details
from .my_queen_under_attack import is_queen_under_attack, queen_under_attack_details

# 可以攻击对方棋子的规则
from .my_can_capture_rook import is_can_capture_rook, can_capture_rook_details
from .my_can_capture_knight import is_can_capture_knight, can_capture_knight_details
from .my_can_capture_bishop import is_can_capture_bishop, can_capture_bishop_details
from .my_can_capture_queen import is_can_capture_queen, can_capture_queen_details
from .my_king_check import is_king_in_check, king_check_details, is_checkmate, is_stalemate

from .my_bishop_pair import is_bishop_pair, get_bishop_positions

# 战术分析
from .my_pin import is_piece_pinned, get_pinned_pieces, get_pin_details
from .my_fork import has_fork_move, get_fork_moves, get_fork_details, is_in_fork, get_fork_threat_details
from .my_skewer import has_skewer_move, get_skewer_moves, get_skewer_details

# 兵结构分析
from .my_pawn_structure import (
    get_isolated_pawns, get_doubled_pawns, get_passed_pawns, 
    get_backward_pawns, analyze_pawn_structure
)

# 中心控制和子力活动度
from .my_center_control import (
    get_center_control, get_mobility_score, analyze_piece_activity,
    get_king_safety_score
)

# 威胁分析
from .my_threat_analysis import (
    analyze_threats_by_lesser_pieces, get_threat_details, get_threat_summary
)


__all__ = [
    # 棋子被抓相关
    "is_rook_under_attack",
    "rook_under_attack_details",
    "is_knight_under_attack",
    "knight_under_attack_details",
    "is_bishop_under_attack",
    "bishop_under_attack_details",
    "is_queen_under_attack",
    "queen_under_attack_details",
    
    # 可以攻击对方棋子相关
    "is_can_capture_rook",
    "can_capture_rook_details",
    "is_can_capture_knight",
    "can_capture_knight_details",
    "is_can_capture_bishop",
    "can_capture_bishop_details",
    "is_can_capture_queen",
    "can_capture_queen_details",
    
    # 王被将军相关
    "is_king_in_check", 
    "king_check_details",
    "is_checkmate",
    "is_stalemate",
    
    # 双象相关
    "is_bishop_pair",
    "get_bishop_positions",
    
    # 战术分析
    "is_piece_pinned",
    "get_pinned_pieces", 
    "get_pin_details",
    "has_fork_move",
    "get_fork_moves",
    "get_fork_details",
    "is_in_fork",
    "get_fork_threat_details",
    "has_skewer_move",
    "get_skewer_moves", 
    "get_skewer_details",
    
    # 兵结构分析
    "get_isolated_pawns",
    "get_doubled_pawns", 
    "get_passed_pawns",
    "get_backward_pawns",
    "analyze_pawn_structure",
    
    # 中心控制和子力活动度
    "get_center_control",
    "get_mobility_score",
    "analyze_piece_activity",
    "get_king_safety_score",
    
    # 威胁分析
    "analyze_threats_by_lesser_pieces",
    "get_threat_details",
    "get_threat_summary",
]
