import chess
from typing import Dict, Set

__all__ = [
    "get_center_control", "get_mobility_score", "analyze_piece_activity",
    "get_king_safety_score"
]

# 中心格子定义
CENTER_SQUARES = [chess.E4, chess.E5, chess.D4, chess.D5]
EXTENDED_CENTER = [
    chess.C3, chess.C4, chess.C5, chess.C6,
    chess.D3, chess.D4, chess.D5, chess.D6,
    chess.E3, chess.E4, chess.E5, chess.E6,
    chess.F3, chess.F4, chess.F5, chess.F6
]


def get_center_control(fen: str) -> dict:
    """分析中心控制情况。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        包含中心控制分析的字典
    """
    board = chess.Board(fen)
    result = {
        'white': {'center_attacks': 0, 'extended_center_attacks': 0, 'center_pieces': 0},
        'black': {'center_attacks': 0, 'extended_center_attacks': 0, 'center_pieces': 0}
    }
    
    for color in [chess.WHITE, chess.BLACK]:
        color_name = "white" if color == chess.WHITE else "black"
        
        # 统计中心格子上的己方棋子
        for square in CENTER_SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                result[color_name]['center_pieces'] += 1
        
        # 统计对中心格子的攻击
        for square in CENTER_SQUARES:
            if _is_square_attacked_by_color(board, square, color):
                result[color_name]['center_attacks'] += 1
        
        # 统计对扩展中心的攻击
        for square in EXTENDED_CENTER:
            if _is_square_attacked_by_color(board, square, color):
                result[color_name]['extended_center_attacks'] += 1
    
    # 计算控制优势
    result['white_advantage'] = (
        result['white']['center_attacks'] - result['black']['center_attacks'] +
        result['white']['center_pieces'] * 2 - result['black']['center_pieces'] * 2
    )
    
    return result


def get_mobility_score(fen: str, color: str) -> dict:
    """计算子力活动度得分。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        color: 要分析的颜色（'white' 或 'black'）
        
    Returns:
        包含活动度分析的字典
    """
    board = chess.Board(fen)
    chess_color = chess.WHITE if color.lower() == 'white' else chess.BLACK
    total_moves = 0
    piece_mobility = {}
    
    # 统计各种棋子的合法着法数量
    for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        piece_mobility[chess.piece_name(piece_type)] = 0
    
    # 只统计指定颜色的着法
    current_turn = board.turn
    if current_turn != chess_color:
        # 如果不是该颜色行棋，临时切换
        board.turn = chess_color
        legal_moves = list(board.legal_moves)
        board.turn = current_turn
    else:
        legal_moves = list(board.legal_moves)
    
    total_moves = len(legal_moves)
    
    # 按棋子类型分类统计
    for move in legal_moves:
        piece = board.piece_at(move.from_square)
        if piece and piece.color == chess_color:
            piece_name = chess.piece_name(piece.piece_type)
            if piece_name in piece_mobility:
                piece_mobility[piece_name] += 1
    
    return {
        'total_mobility': total_moves,
        'piece_mobility': piece_mobility,
        'mobility_score': _calculate_mobility_score(piece_mobility)
    }


def analyze_piece_activity(fen: str) -> dict:
    """全面分析双方子力活动度。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        包含双方活动度对比的字典
    """
    result = {}
    
    for color in ['white', 'black']:
        mobility = get_mobility_score(fen, color)
        
        result[color] = {
            'mobility': mobility,
            'piece_development': _get_development_score(fen, color),
            'piece_coordination': _get_coordination_score(fen, color)
        }
    
    # 计算活动度优势
    white_total = result['white']['mobility']['mobility_score']
    black_total = result['black']['mobility']['mobility_score']
    result['mobility_advantage'] = white_total - black_total
    
    return result


def get_king_safety_score(fen: str, color: str) -> dict:
    """评估王的安全性。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        color: 要评估的王的颜色（'white' 或 'black'）
        
    Returns:
        包含王安全性分析的字典
    """
    board = chess.Board(fen)
    chess_color = chess.WHITE if color.lower() == 'white' else chess.BLACK
    king_square = board.king(chess_color)
    if king_square is None:
        return {'safety_score': -1000, 'threats': [], 'pawn_shield': 0}
    
    safety_score = 0
    threats = []
    
    # 检查王周围的兵盾
    pawn_shield_score = _evaluate_pawn_shield(board, king_square, chess_color)
    safety_score += pawn_shield_score
    
    # 检查王周围被敌方控制的格子
    enemy_controlled = 0
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    
    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            if df == 0 and dr == 0:
                continue
            
            new_file = king_file + df
            new_rank = king_rank + dr
            
            if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
                check_square = chess.square(new_file, new_rank)
                if _is_square_attacked_by_color(board, check_square, not chess_color):
                    enemy_controlled += 1
    
    safety_score -= enemy_controlled * 10
    
    # 检查直接威胁
    if board.is_check():
        threats.append('check')
        safety_score -= 50
    
    # 检查王前是否有逃脱路线
    escape_squares = 0
    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            if df == 0 and dr == 0:
                continue
            
            new_file = king_file + df
            new_rank = king_rank + dr
            
            if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
                escape_square = chess.square(new_file, new_rank)
                # 检查这个格子是否安全且合法
                piece = board.piece_at(escape_square)
                if (not piece or piece.color != chess_color) and not _is_square_attacked_by_color(board, escape_square, not chess_color):
                    escape_squares += 1
    
    safety_score += escape_squares * 5
    
    return {
        'safety_score': safety_score,
        'threats': threats,
        'pawn_shield': pawn_shield_score,
        'enemy_controlled_squares': enemy_controlled,
        'escape_squares': escape_squares
    }


def _is_square_attacked_by_color(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    """检查指定格子是否被指定颜色攻击。"""
    # 获取指定颜色的所有棋子
    for piece_square in chess.SQUARES:
        piece = board.piece_at(piece_square)
        if piece and piece.color == color:
            attacks = board.attacks(piece_square)
            if square in attacks:
                return True
    return False


def _calculate_mobility_score(piece_mobility: dict) -> int:
    """根据棋子活动度计算得分。"""
    weights = {
        'queen': 1,
        'rook': 2,
        'bishop': 2,
        'knight': 4  # 马的每步移动价值更高
    }
    
    total_score = 0
    for piece_name, moves in piece_mobility.items():
        weight = weights.get(piece_name, 1)
        total_score += moves * weight
    
    return total_score


def _get_development_score(fen: str, color: str) -> int:
    """评估子力发展情况。"""
    board = chess.Board(fen)
    chess_color = chess.WHITE if color.lower() == 'white' else chess.BLACK
    score = 0
    starting_rank = 0 if chess_color == chess.WHITE else 7
    
    # 检查轻子是否移动
    for piece_type in [chess.KNIGHT, chess.BISHOP]:
        pieces = board.pieces(piece_type, chess_color)
        for piece_square in pieces:
            rank = chess.square_rank(piece_square)
            if rank != starting_rank:
                score += 10  # 轻子发展奖励
    
    # 检查易位
    if chess_color == chess.WHITE:
        king_square = board.king(chess.WHITE)
        if king_square and chess.square_file(king_square) in [2, 6]:  # 易位后王的位置
            score += 20
    else:
        king_square = board.king(chess.BLACK)
        if king_square and chess.square_file(king_square) in [2, 6]:
            score += 20
    
    return score


def _get_coordination_score(fen: str, color: str) -> int:
    """评估子力协调性。"""
    board = chess.Board(fen)
    chess_color = chess.WHITE if color.lower() == 'white' else chess.BLACK
    score = 0
    
    # 检查同种棋子的协同（如重车）
    for piece_type in [chess.ROOK, chess.BISHOP]:
        pieces = list(board.pieces(piece_type, chess_color))
        if len(pieces) >= 2:
            # 检查是否在同一线上
            for i in range(len(pieces)):
                for j in range(i + 1, len(pieces)):
                    sq1, sq2 = pieces[i], pieces[j]
                    file1, rank1 = chess.square_file(sq1), chess.square_rank(sq1)
                    file2, rank2 = chess.square_file(sq2), chess.square_rank(sq2)
                    
                    if piece_type == chess.ROOK:
                        if file1 == file2 or rank1 == rank2:
                            score += 15  # 重车奖励
                    elif piece_type == chess.BISHOP:
                        if abs(file1 - file2) == abs(rank1 - rank2):
                            score += 10  # 双象对角线奖励
    
    return score


def _evaluate_pawn_shield(board: chess.Board, king_square: chess.Square, color: chess.Color) -> int:
    """评估王前兵盾。"""
    score = 0
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    
    # 检查王前三列的兵盾
    direction = 1 if color == chess.WHITE else -1
    
    for df in [-1, 0, 1]:
        check_file = king_file + df
        if 0 <= check_file <= 7:
            # 检查王前的兵
            for dr in [1, 2]:
                pawn_rank = king_rank + (dr * direction)
                if 0 <= pawn_rank <= 7:
                    pawn_square = chess.square(check_file, pawn_rank)
                    piece = board.piece_at(pawn_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        score += 15 - (dr * 5)  # 越近的兵保护价值越高
                        break
    
    return score 