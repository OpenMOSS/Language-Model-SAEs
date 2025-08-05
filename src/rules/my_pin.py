import chess
from typing import List, Tuple, Optional

__all__ = ["is_piece_pinned", "get_pinned_pieces", "get_pin_details"]


def is_piece_pinned(fen: str, square_name: str) -> bool:
    """检查指定位置的棋子是否被钉住。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        square_name: 要检查的棋子位置（如'e4'）
        
    Returns:
        如果棋子被钉住返回True，否则返回False
    """
    board = chess.Board(fen)
    square = chess.parse_square(square_name)
    piece = board.piece_at(square)
    if not piece:
        return False
    
    # 找到己方王的位置
    king_square = board.king(piece.color)
    if king_square is None:
        return False
    
    # 检查被测试棋子是否在王和敌方攻击子之间
    for attacker_square in chess.SQUARES:
        attacker = board.piece_at(attacker_square)
        if not attacker or attacker.color == piece.color:
            continue
        
        # 只考虑能进行直线攻击的棋子（车、象、后）
        if attacker.piece_type not in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
            continue
        
        # 检查攻击者、被测试棋子、王是否在同一条线上
        if not _is_on_attack_line(attacker_square, square, king_square, attacker.piece_type):
            continue
        
        # 检查被测试棋子是否真的在攻击者和王之间
        if not _is_between(attacker_square, square, king_square):
            continue
        
        # 检查移除被测试棋子后，攻击者是否能攻击到王
        board_copy = board.copy()
        board_copy.remove_piece_at(square)
        
        if king_square in board_copy.attacks(attacker_square):
            return True
    
    return False


def get_pinned_pieces(fen: str) -> list[str]:
    """获取当前棋盘上所有被钉住的棋子位置。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        被钉住的棋子位置列表（标准代数记号法）
    """
    board = chess.Board(fen)
    pinned_pieces = []
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            square_name = chess.square_name(square)
            if is_piece_pinned(fen, square_name):
                pinned_pieces.append(square_name)
    
    return pinned_pieces


def get_pin_details(fen: str) -> list[dict]:
    """获取钉住的详细信息。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        包含钉住详情的字典列表，每个字典包含：
        - pinned_piece: 被钉住的棋子位置
        - pinning_piece: 形成钉住的棋子位置  
        - target_piece: 被保护的目标棋子位置
        - line_type: 钉住类型（'rank', 'file', 'diagonal'）
    """
    board = chess.Board(fen)
    pin_details = []
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
        
        # 找到己方王的位置
        king_square = board.king(piece.color)
        if king_square is None:
            continue
        
        # 检查是否被钉住到王
        for attacker_square in chess.SQUARES:
            attacker = board.piece_at(attacker_square)
            if not attacker or attacker.color == piece.color:
                continue
            
            # 只考虑能进行直线攻击的棋子
            if attacker.piece_type not in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
                continue
            
            # 检查是否在攻击线上
            if not _is_on_attack_line(attacker_square, square, king_square, attacker.piece_type):
                continue
            
            # 检查位置关系
            if not _is_between(attacker_square, square, king_square):
                continue
            
            # 验证移除后确实会暴露王
            board_copy = board.copy()
            board_copy.remove_piece_at(square)
            
            if king_square in board_copy.attacks(attacker_square):
                pin_details.append({
                    'pinned_piece': chess.square_name(square),
                    'pinning_piece': chess.square_name(attacker_square),
                    'target_piece': chess.square_name(king_square),
                    'line_type': _get_line_type(attacker_square, king_square)
                })
                break  # 每个棋子只记录一次钉住
    
    return pin_details


def _is_on_attack_line(attacker: chess.Square, middle: chess.Square, target: chess.Square, piece_type: chess.PieceType) -> bool:
    """检查三个格子是否在指定棋子类型的攻击线上。"""
    file1, rank1 = chess.square_file(attacker), chess.square_rank(attacker)
    file2, rank2 = chess.square_file(middle), chess.square_rank(middle)
    file3, rank3 = chess.square_file(target), chess.square_rank(target)
    
    # 检查是否在同一条直线上
    if piece_type in [chess.ROOK, chess.QUEEN]:
        # 横线或竖线
        if (file1 == file2 == file3) or (rank1 == rank2 == rank3):
            return True
    
    if piece_type in [chess.BISHOP, chess.QUEEN]:
        # 对角线
        if (abs(file1 - file2) == abs(rank1 - rank2) and 
            abs(file2 - file3) == abs(rank2 - rank3) and
            abs(file1 - file3) == abs(rank1 - rank3)):
            return True
    
    return False


def _is_between(square1: chess.Square, square2: chess.Square, square3: chess.Square) -> bool:
    """检查square2是否在square1和square3之间。"""
    file1, rank1 = chess.square_file(square1), chess.square_rank(square1)
    file2, rank2 = chess.square_file(square2), chess.square_rank(square2)
    file3, rank3 = chess.square_file(square3), chess.square_rank(square3)
    
    # 同一列
    if file1 == file2 == file3:
        min_rank, max_rank = min(rank1, rank3), max(rank1, rank3)
        return min_rank < rank2 < max_rank
    
    # 同一行
    if rank1 == rank2 == rank3:
        min_file, max_file = min(file1, file3), max(file1, file3)
        return min_file < file2 < max_file
    
    # 对角线
    if abs(file1 - file3) == abs(rank1 - rank3):
        min_file, max_file = min(file1, file3), max(file1, file3)
        min_rank, max_rank = min(rank1, rank3), max(rank1, rank3)
        return (min_file < file2 < max_file and min_rank < rank2 < max_rank)
    
    return False


def _get_line_type(square1: chess.Square, square2: chess.Square) -> str:
    """确定两个方格之间的线型。"""
    file1, rank1 = chess.square_file(square1), chess.square_rank(square1)
    file2, rank2 = chess.square_file(square2), chess.square_rank(square2)
    
    if rank1 == rank2:
        return 'rank'
    elif file1 == file2:
        return 'file'
    else:
        return 'diagonal' 