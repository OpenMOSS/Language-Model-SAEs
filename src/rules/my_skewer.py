import chess
from typing import List, Dict, Optional

__all__ = ["has_skewer_move", "get_skewer_moves", "get_skewer_details"]


def has_skewer_move(fen: str) -> bool:
    """检查当前行棋方是否有牵制着法。
    
    牵制（Skewer）是指攻击一个高价值棋子，迫使其移动后暴露后方的低价值棋子。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        如果存在牵制着法返回True，否则返回False
    """
    board = chess.Board(fen)
    for move in board.legal_moves:
        if _is_skewer_move(board, move):
            return True
    return False


def get_skewer_moves(fen: str) -> list[str]:
    """获取所有可能的牵制着法。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        所有牵制着法的UCI字符串列表
    """
    board = chess.Board(fen)
    skewer_moves = []
    
    for move in board.legal_moves:
        if _is_skewer_move(board, move):
            skewer_moves.append(move.uci())
    
    return skewer_moves


def get_skewer_details(fen: str) -> list[dict]:
    """获取牵制的详细信息。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        包含牵制详情的字典列表，每个字典包含：
        - move: 牵制着法
        - skewering_piece: 形成牵制的棋子类型
        - target_piece: 被攻击的高价值目标
        - behind_piece: 被暴露的后方棋子
        - line_type: 牵制线型（'rank', 'file', 'diagonal'）
        - value_gain: 可能获得的价值
    """
    board = chess.Board(fen)
    skewer_details = []
    
    for move in board.legal_moves:
        skewer_info = _get_skewer_info(board, move)
        if skewer_info:
            skewer_details.append({
                'move': move.uci(),
                'skewering_piece': chess.piece_name(board.piece_at(move.from_square).piece_type),
                'target_piece': skewer_info['target_piece'],
                'behind_piece': skewer_info['behind_piece'],
                'line_type': skewer_info['line_type'],
                'value_gain': skewer_info['value_gain']
            })
    
    return skewer_details


def _is_skewer_move(board: chess.Board, move: chess.Move) -> bool:
    """检查一个着法是否构成牵制。
    
    Args:
        board: 当前棋盘状态
        move: 要检查的着法
        
    Returns:
        如果是牵制着法返回True，否则返回False
    """
    return _get_skewer_info(board, move) is not None


def _get_skewer_info(board: chess.Board, move: chess.Move) -> Optional[dict]:
    """获取牵制的详细信息。
    
    Args:
        board: 当前棋盘状态
        move: 要检查的着法
        
    Returns:
        如果是牵制则返回详细信息字典，否则返回None
    """
    # 执行着法
    board_copy = board.copy()
    board_copy.push(move)
    
    # 获取移动后的棋子位置
    attacking_piece = board_copy.piece_at(move.to_square)
    if not attacking_piece:
        return None
    
    # 检查这个棋子能攻击的直线上的敌方棋子
    attacks = board_copy.attacks(move.to_square)
    
    for target_square in attacks:
        target_piece = board_copy.piece_at(target_square)
        if not target_piece or target_piece.color == attacking_piece.color:
            continue
        
        # 检查是否在同一条直线上
        if not _is_line_attack(attacking_piece.piece_type, move.to_square, target_square):
            continue
        
        # 检查这条直线上目标后方是否有己方可以攻击的棋子
        behind_square = _get_square_behind(move.to_square, target_square)
        if behind_square is None:
            continue
        
        behind_piece = board_copy.piece_at(behind_square)
        if not behind_piece or behind_piece.color == attacking_piece.color:
            continue
        
        # 检查价值关系：目标应该比后方棋子更有价值（这是牵制的特征）
        target_value = _piece_value(target_piece)
        behind_value = _piece_value(behind_piece)
        
        if target_value > behind_value:
            return {
                'target_piece': chess.square_name(target_square),
                'behind_piece': chess.square_name(behind_square),
                'line_type': _get_line_type(move.to_square, target_square),
                'value_gain': behind_value,
                'target_value': target_value
            }
    
    return None


def _is_line_attack(piece_type: chess.PieceType, from_square: chess.Square, to_square: chess.Square) -> bool:
    """检查棋子类型是否能进行直线攻击。"""
    if piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP]:
        file1, rank1 = chess.square_file(from_square), chess.square_rank(from_square)
        file2, rank2 = chess.square_file(to_square), chess.square_rank(to_square)
        
        # 车和后的横竖线攻击
        if piece_type in [chess.QUEEN, chess.ROOK]:
            if file1 == file2 or rank1 == rank2:
                return True
        
        # 象和后的对角线攻击
        if piece_type in [chess.QUEEN, chess.BISHOP]:
            if abs(file1 - file2) == abs(rank1 - rank2):
                return True
    
    return False


def _get_square_behind(attacker_square: chess.Square, target_square: chess.Square) -> Optional[chess.Square]:
    """获取目标后方的格子。"""
    file1, rank1 = chess.square_file(attacker_square), chess.square_rank(attacker_square)
    file2, rank2 = chess.square_file(target_square), chess.square_rank(target_square)
    
    # 计算方向
    file_diff = file2 - file1
    rank_diff = rank2 - rank1
    
    # 标准化方向
    if file_diff != 0:
        file_diff = file_diff // abs(file_diff)
    if rank_diff != 0:
        rank_diff = rank_diff // abs(rank_diff)
    
    # 计算后方位置
    behind_file = file2 + file_diff
    behind_rank = rank2 + rank_diff
    
    # 检查是否在棋盘范围内
    if 0 <= behind_file <= 7 and 0 <= behind_rank <= 7:
        return chess.square(behind_file, behind_rank)
    
    return None


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


def get_discovered_attacks(board: chess.Board) -> list[dict]:
    """检测闪击（发现攻击）。
    
    闪击是指移动前方棋子后，暴露后方棋子对目标的攻击。
    
    Args:
        board: 当前棋盘状态
        
    Returns:
        闪击详情列表
    """
    discovered_attacks = []
    
    for move in board.legal_moves:
        # 检查移动前后的攻击变化
        original_attacks = set()
        
        # 记录移动前的攻击
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                original_attacks.update(board.attacks(square))
        
        # 执行移动
        board_copy = board.copy()
        board_copy.push(move)
        
        # 记录移动后的攻击
        new_attacks = set()
        for square in chess.SQUARES:
            piece = board_copy.piece_at(square)
            if piece and piece.color == board.turn:
                new_attacks.update(board_copy.attacks(square))
        
        # 找出新增的攻击
        discovered = new_attacks - original_attacks
        
        # 筛选出有价值的发现攻击
        for target_square in discovered:
            target_piece = board_copy.piece_at(target_square)
            if target_piece and target_piece.color != board.turn:
                # 确认这是由于移开遮挡造成的
                if _is_discovered_attack(board, move, target_square):
                    discovered_attacks.append({
                        'move': move.uci(),
                        'moved_piece': chess.piece_name(board.piece_at(move.from_square).piece_type),
                        'discovering_piece': _find_discovering_piece(board_copy, target_square, move.from_square),
                        'target': chess.square_name(target_square),
                        'target_value': _piece_value(target_piece)
                    })
    
    return discovered_attacks


def _is_discovered_attack(board: chess.Board, move: chess.Move, target_square: chess.Square) -> bool:
    """检查是否是真正的发现攻击。"""
    # 移动前目标不被攻击
    if target_square in board.attacks_mask:
        return False
    
    # 移动后目标被攻击
    board_copy = board.copy()
    board_copy.push(move)
    
    return target_square in board_copy.attacks_mask


def _find_discovering_piece(board: chess.Board, target_square: chess.Square, moved_from: chess.Square) -> str:
    """找到进行发现攻击的棋子。"""
    for square in chess.SQUARES:
        if square == moved_from:  # 跳过移动的棋子
            continue
        
        piece = board.piece_at(square)
        if piece and piece.color == board.turn:
            attacks = board.attacks(square)
            if target_square in attacks:
                # 检查这条攻击线是否经过移动的起始位置
                if _line_passes_through(square, target_square, moved_from):
                    return chess.piece_name(piece.piece_type)
    
    return "unknown"


def _line_passes_through(start: chess.Square, end: chess.Square, middle: chess.Square) -> bool:
    """检查从start到end的直线是否经过middle。"""
    file1, rank1 = chess.square_file(start), chess.square_rank(start)
    file2, rank2 = chess.square_file(end), chess.square_rank(end)
    file3, rank3 = chess.square_file(middle), chess.square_rank(middle)
    
    # 检查是否在同一条直线上
    if file1 == file2 == file3:  # 同一列
        return min(rank1, rank2) < rank3 < max(rank1, rank2)
    elif rank1 == rank2 == rank3:  # 同一行
        return min(file1, file2) < file3 < max(file1, file2)
    elif abs(file1 - file2) == abs(rank1 - rank2):  # 对角线
        # 检查是否在对角线上并且在中间
        if abs(file1 - file3) == abs(rank1 - rank3) and abs(file2 - file3) == abs(rank2 - rank3):
            return (min(file1, file2) < file3 < max(file1, file2) and 
                    min(rank1, rank2) < rank3 < max(rank1, rank2))
    
    return False 