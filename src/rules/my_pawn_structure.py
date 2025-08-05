import chess
from typing import List, Dict, Set

__all__ = [
    "get_isolated_pawns", "get_doubled_pawns", "get_passed_pawns", 
    "get_backward_pawns", "analyze_pawn_structure"
]


def get_isolated_pawns(fen: str, color: str) -> list[str]:
    """获取孤兵位置。
    
    孤兵是指在相邻列上没有同色兵的兵。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        color: 要检查的兵的颜色（'white' 或 'black'）
        
    Returns:
        孤兵位置列表（标准代数记号法）
    """
    board = chess.Board(fen)
    chess_color = chess.WHITE if color.lower() == 'white' else chess.BLACK
    pawns = board.pieces(chess.PAWN, chess_color)
    isolated_pawns = []
    
    for pawn_square in pawns:
        file = chess.square_file(pawn_square)
        
        # 检查相邻列是否有同色兵
        has_adjacent_pawn = False
        for adjacent_file in [file - 1, file + 1]:
            if 0 <= adjacent_file <= 7:  # 确保在棋盘范围内
                for rank in range(8):
                    adjacent_square = chess.square(adjacent_file, rank)
                    piece = board.piece_at(adjacent_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess_color:
                        has_adjacent_pawn = True
                        break
                if has_adjacent_pawn:
                    break
        
        if not has_adjacent_pawn:
            isolated_pawns.append(chess.square_name(pawn_square))
    
    return isolated_pawns


def get_doubled_pawns(fen: str, color: str) -> dict[int, list[str]]:
    """获取叠兵位置。
    
    叠兵是指同一列上有多个同色兵。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        color: 要检查的兵的颜色（'white' 或 'black'）
        
    Returns:
        字典，键为列号，值为该列上的兵位置列表（仅包含有多个兵的列）
    """
    board = chess.Board(fen)
    chess_color = chess.WHITE if color.lower() == 'white' else chess.BLACK
    pawns = board.pieces(chess.PAWN, chess_color)
    files_pawns = {}
    
    # 按列分组
    for pawn_square in pawns:
        file = chess.square_file(pawn_square)
        if file not in files_pawns:
            files_pawns[file] = []
        files_pawns[file].append(chess.square_name(pawn_square))
    
    # 只返回有多个兵的列
    doubled_pawns = {file: squares for file, squares in files_pawns.items() if len(squares) > 1}
    
    return doubled_pawns


def get_passed_pawns(fen: str, color: str) -> list[str]:
    """获取通路兵位置。
    
    通路兵是指前进路线上没有敌方兵阻挡的兵。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        color: 要检查的兵的颜色（'white' 或 'black'）
        
    Returns:
        通路兵位置列表（标准代数记号法）
    """
    board = chess.Board(fen)
    chess_color = chess.WHITE if color.lower() == 'white' else chess.BLACK
    pawns = board.pieces(chess.PAWN, chess_color)
    enemy_pawns = board.pieces(chess.PAWN, not chess_color)
    passed_pawns = []
    
    for pawn_square in pawns:
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # 确定前进方向
        direction = 1 if chess_color == chess.WHITE else -1
        
        # 检查前进路线（本列和相邻列）
        is_passed = True
        for check_file in [file - 1, file, file + 1]:
            if 0 <= check_file <= 7:
                # 检查从当前位置到底线的所有格子
                start_rank = rank + direction
                end_rank = 8 if chess_color == chess.WHITE else -1
                
                for check_rank in range(start_rank, end_rank, direction):
                    if 0 <= check_rank <= 7:
                        check_square = chess.square(check_file, check_rank)
                        if check_square in enemy_pawns:
                            is_passed = False
                            break
                if not is_passed:
                    break
        
        if is_passed:
            passed_pawns.append(chess.square_name(pawn_square))
    
    return passed_pawns


def get_backward_pawns(fen: str, color: str) -> list[str]:
    """获取倒挂兵位置。
    
    倒挂兵是指落后于相邻列的己方兵且无法安全前进的兵。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        color: 要检查的兵的颜色（'white' 或 'black'）
        
    Returns:
        倒挂兵位置列表（标准代数记号法）
    """
    board = chess.Board(fen)
    chess_color = chess.WHITE if color.lower() == 'white' else chess.BLACK
    pawns = board.pieces(chess.PAWN, chess_color)
    enemy_pawns = board.pieces(chess.PAWN, not chess_color)
    backward_pawns = []
    
    for pawn_square in pawns:
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # 检查相邻列的己方兵是否都在前面
        is_backward = True
        has_adjacent_pawn = False
        
        for adjacent_file in [file - 1, file + 1]:
            if 0 <= adjacent_file <= 7:
                for check_rank in range(8):
                    check_square = chess.square(adjacent_file, check_rank)
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess_color:
                        has_adjacent_pawn = True
                        # 检查这个相邻兵是否在前面
                        if chess_color == chess.WHITE and check_rank <= rank:
                            is_backward = False
                        elif chess_color == chess.BLACK and check_rank >= rank:
                            is_backward = False
        
        # 如果没有相邻兵，就不是倒挂兵
        if not has_adjacent_pawn:
            is_backward = False
        
        # 检查是否能安全前进
        if is_backward:
            direction = 1 if chess_color == chess.WHITE else -1
            next_rank = rank + direction
            if 0 <= next_rank <= 7:
                next_square = chess.square(file, next_rank)
                # 检查前进格子是否被敌方兵控制
                for enemy_pawn in enemy_pawns:
                    enemy_file = chess.square_file(enemy_pawn)
                    enemy_rank = chess.square_rank(enemy_pawn)
                    # 检查敌方兵是否能攻击到前进格子
                    if abs(enemy_file - file) == 1:
                        enemy_attack_rank = enemy_rank + (1 if not chess_color else -1)
                        if enemy_attack_rank == next_rank:
                            # 前进会被攻击，确实是倒挂兵
                            backward_pawns.append(chess.square_name(pawn_square))
                            break
    
    return backward_pawns


def analyze_pawn_structure(fen: str) -> dict:
    """完整分析兵结构。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        包含各种兵结构分析结果的字典
    """
    result = {}
    
    for color in ['white', 'black']:
        isolated = get_isolated_pawns(fen, color)
        doubled = get_doubled_pawns(fen, color)
        passed = get_passed_pawns(fen, color)
        backward = get_backward_pawns(fen, color)
        
        result[color] = {
            'isolated_pawns': isolated,
            'doubled_pawns': {
                f'file_{file}': squares 
                for file, squares in doubled.items()
            },
            'passed_pawns': passed,
            'backward_pawns': backward,
            'weaknesses_count': len(isolated) + sum(len(squares) - 1 for squares in doubled.values()) + len(backward),
            'strengths_count': len(passed)
        }
    
    return result


def get_pawn_structure_score(fen: str, color: str) -> int:
    """计算兵结构得分。
    
    正分表示良好的兵结构，负分表示有弱点。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        color: 要评估的颜色（'white' 或 'black'）
        
    Returns:
        兵结构得分
    """
    score = 0
    
    # 惩罚弱点
    isolated = get_isolated_pawns(fen, color)
    doubled = get_doubled_pawns(fen, color)
    backward = get_backward_pawns(fen, color)
    
    score -= len(isolated) * 20  # 孤兵惩罚
    score -= sum(len(squares) - 1 for squares in doubled.values()) * 15  # 叠兵惩罚
    score -= len(backward) * 10  # 倒挂兵惩罚
    
    # 奖励优势
    passed = get_passed_pawns(fen, color)
    score += len(passed) * 25  # 通路兵奖励
    
    return score 