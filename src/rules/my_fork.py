import chess
from typing import List, Dict, Set

__all__ = ["has_fork_move", "get_fork_moves", "get_fork_details", "is_in_fork", "get_fork_threat_details"]


def has_fork_move(fen: str) -> bool:
    """检查当前行棋方是否有叉攻着法（同时攻击两个或更多非兵棋子）。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        如果存在叉攻着法返回True，否则返回False
    """
    board = chess.Board(fen)
    for move in board.legal_moves:
        if _is_fork_move(board, move):
            return True
    return False


def get_fork_moves(fen: str) -> list[str]:
    """获取所有可能的叉攻着法（同时攻击两个或更多非兵棋子）。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        所有叉攻着法的UCI字符串列表
    """
    board = chess.Board(fen)
    fork_moves = []
    
    for move in board.legal_moves:
        if _is_fork_move(board, move):
            fork_moves.append(move.uci())
    
    return fork_moves


def get_fork_details(fen: str) -> list[dict]:
    """获取叉攻的详细信息（只考虑攻击非兵棋子）。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        包含叉攻详情的字典列表，每个字典包含：
        - move: 叉攻着法
        - forking_piece: 形成叉攻的棋子类型
        - targets: 被攻击的非兵目标位置列表
        - target_values: 被攻击目标的价值列表
        - total_value: 目标总价值
    """
    board = chess.Board(fen)
    fork_details = []
    
    for move in board.legal_moves:
        targets = _get_fork_targets(board, move)
        if len(targets) >= 2:
            board_copy = board.copy()
            board_copy.push(move)
            
            forking_piece = board.piece_at(move.from_square)
            target_values = []
            target_positions = []
            
            for target_square in targets:
                target_piece = board.piece_at(target_square)
                if target_piece:
                    target_values.append(_piece_value(target_piece))
                    target_positions.append(chess.square_name(target_square))
            
            fork_details.append({
                'move': move.uci(),
                'forking_piece': chess.piece_name(forking_piece.piece_type) if forking_piece else 'unknown',
                'targets': target_positions,
                'target_values': target_values,
                'total_value': sum(target_values)
            })
    
    return fork_details


def is_in_fork(fen: str) -> bool:
    """检查当前行棋方是否面临叉攻威胁。
    
    检查当前棋盘状态下，对手是否有棋子正在同时攻击己方的两个或更多非兵棋子。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        如果当前行棋方面临叉攻威胁返回True，否则返回False
    """
    board = chess.Board(fen)
    current_player_color = board.turn
    opponent_color = not current_player_color
    
    # 检查对手的每个棋子
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == opponent_color:
            # 获取这个棋子能攻击的所有位置
            attacks = board.attacks(square)
            
            # 统计被攻击的己方非兵棋子
            attacked_own_pieces = []
            for target_square in attacks:
                target_piece = board.piece_at(target_square)
                if (target_piece and 
                    target_piece.color == current_player_color and
                    target_piece.piece_type != chess.PAWN):  # 忽略兵
                    attacked_own_pieces.append(target_square)
            
            # 如果同时攻击己方2个或更多非兵棋子，则构成叉攻威胁
            if len(attacked_own_pieces) >= 2:
                return True
    
    return False


def get_fork_threat_details(fen: str) -> list[dict]:
    """获取当前行棋方面临的叉攻威胁详细信息。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        包含叉攻威胁详情的字典列表，每个字典包含：
        - forking_piece: 对手叉攻棋子的位置和类型
        - attacked_pieces: 己方被攻击的非兵棋子位置和类型列表
        - total_threat_value: 被威胁棋子的总价值
    """
    board = chess.Board(fen)
    threat_details = []
    current_player_color = board.turn
    opponent_color = not current_player_color
    
    # 检查对手的每个棋子
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == opponent_color:
            # 获取这个棋子能攻击的所有位置
            attacks = board.attacks(square)
            
            # 统计被攻击的己方非兵棋子
            attacked_pieces = []
            total_value = 0
            
            for target_square in attacks:
                target_piece = board.piece_at(target_square)
                if (target_piece and 
                    target_piece.color == current_player_color and
                    target_piece.piece_type != chess.PAWN):  # 忽略兵
                    piece_info = {
                        'position': chess.square_name(target_square),
                        'piece_type': chess.piece_name(target_piece.piece_type),
                        'value': _piece_value(target_piece)
                    }
                    attacked_pieces.append(piece_info)
                    total_value += _piece_value(target_piece)
            
            # 如果同时攻击己方2个或更多非兵棋子，则记录这个威胁
            if len(attacked_pieces) >= 2:
                threat_details.append({
                    'forking_piece': {
                        'position': chess.square_name(square),
                        'piece_type': chess.piece_name(piece.piece_type)
                    },
                    'attacked_pieces': attacked_pieces,
                    'total_threat_value': total_value
                })
    
    return threat_details


def _is_fork_move(board: chess.Board, move: chess.Move) -> bool:
    """检查一个着法是否构成叉攻。
    
    Args:
        board: 当前棋盘状态
        move: 要检查的着法
        
    Returns:
        如果是叉攻着法返回True，否则返回False
    """
    targets = _get_fork_targets(board, move)
    return len(targets) >= 2


def _get_fork_targets(board: chess.Board, move: chess.Move) -> list[chess.Square]:
    """获取着法执行后能攻击到的敌方棋子（不包括兵）。
    
    Args:
        board: 当前棋盘状态
        move: 要检查的着法
        
    Returns:
        能攻击到的敌方非兵棋子位置列表
    """
    # 创建棋盘副本并执行着法
    board_copy = board.copy()
    board_copy.push(move)
    
    # 获取移动后棋子的位置
    to_square = move.to_square
    piece = board_copy.piece_at(to_square)
    
    if not piece:
        return []
    
    # 获取这个棋子能攻击的所有位置
    attacks = board_copy.attacks(to_square)
    
    # 筛选出敌方非兵棋子
    targets = []
    for target_square in attacks:
        target_piece = board_copy.piece_at(target_square)
        if (target_piece and 
            target_piece.color != piece.color and 
            target_piece.piece_type != chess.PAWN):  # 忽略兵
            targets.append(target_square)
    
    return targets


def _piece_value(piece: chess.Piece) -> int:
    """获取棋子的基本价值。"""
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 100
    }
    return values.get(piece.piece_type, 0)


def get_knight_fork_opportunities(fen: str) -> list[dict]:
    """专门检查马的叉攻机会（马是最常见的叉攻棋子），忽略兵。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        马的叉攻机会详情列表（只计算攻击非兵棋子的情况）
    """
    board = chess.Board(fen)
    knight_forks = []
    
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.KNIGHT:
            targets = _get_fork_targets(board, move)  # 这个函数已经过滤了兵
            if len(targets) >= 2:
                # 检查是否攻击到王
                has_king_attack = False
                target_info = []
                
                for target_square in targets:
                    target_piece = board.piece_at(target_square)
                    if target_piece:
                        if target_piece.piece_type == chess.KING:
                            has_king_attack = True
                        target_info.append({
                            'position': chess.square_name(target_square),
                            'piece': chess.piece_name(target_piece.piece_type),
                            'value': _piece_value(target_piece)
                        })
                
                knight_forks.append({
                    'move': move.uci(),
                    'targets': target_info,
                    'has_king_attack': has_king_attack,
                    'total_value': sum(info['value'] for info in target_info)
                })
    
    return knight_forks 