import chess

__all__ = ["is_bishop_pair", "get_bishop_positions"]


def is_bishop_pair(fen: str) -> bool:
    """检查指定颜色的玩家是否拥有双象（bishop pair）。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        如果当前行棋方拥有两个象，返回True，否则返回False
    """
    board = chess.Board(fen)
    # 获取当前行棋方的颜色
    color = board.turn
    
    # 统计当前行棋方的象的数量
    bishop_count = len(board.pieces(chess.BISHOP, color))
    return bishop_count == 2


def get_bishop_positions(fen: str) -> list[str]:
    """获取当前行棋方所有象的位置。
    
    Args:
        fen: 标准的FEN字符串表示的棋盘状态
        
    Returns:
        包含象位置的列表，位置以标准代数记号法表示（如['c1', 'f1']）
        如果有一个象，列表长度为1；如果有两个象，列表长度为2；如果没有象，列表为空
    """
    board = chess.Board(fen)
    # 获取当前行棋方的颜色
    color = board.turn
    
    # 获取当前行棋方所有象的位置
    bishop_squares = board.pieces(chess.BISHOP, color)
    
    # 将棋盘方格转换为标准代数记号法
    positions = [chess.square_name(square) for square in bishop_squares]
    
    # 按字母顺序排序，保证输出的一致性
    return sorted(positions)