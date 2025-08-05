import chess

__all__ = ["is_can_capture_knight", "can_capture_knight_details"]


def is_can_capture_knight(fen: str) -> bool:
    """判断在给定 FEN 局面下，当前行动方 (side to move) 是否可以攻击（吃）对方的任意一只马。

    *可以攻击* 定义：若己方可以在下一步用以下棋子直接吃掉对方的马，则认为可以攻击该马。允许的攻击
    棋子类型包括：车、马、兵、象、后（斜抓）、王，即 *python-chess* 中的所有常规棋子。

    Args:
        fen: 合法的 FEN 字符串。

    Returns:
        bool: 只要存在至少一只对方马可以被己方攻击，返回 ``True``；否则返回 ``False``。

    Raises:
        ValueError: 当 ``fen`` 字符串无法被解析时抛出。
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"无效的 FEN: {fen}") from exc

    own_color = board.turn            # 当前行棋方
    opponent_color = not own_color

    # 找到对方所有马所在的格子
    opponent_knight_squares = board.pieces(chess.KNIGHT, opponent_color)

    for knight_sq in opponent_knight_squares:
        # 检查己方是否可以攻击这个马
        if board.is_attacked_by(own_color, knight_sq):
            return True
    return False


def can_capture_knight_details(fen: str) -> list[tuple[str, list[str]]]:
    """返回更详细的可以攻击对方马的信息。

    Args:
        fen: FEN 字符串。

    Returns:
        list[tuple[str, list[str]]]: 每个元素包含两个部分：
            1. 可以被攻击的对方马所在格子（如 ``"b8"``）
            2. 一个列表，列出所有可以攻击该马的己方棋子格子（如 ``["a6", "c6"]``）
        如果没有任何对方马可以被攻击，则返回空列表。
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"无效的 FEN: {fen}") from exc

    own_color = board.turn
    opponent_color = not own_color

    results: list[tuple[str, list[str]]] = []
    for knight_sq in board.pieces(chess.KNIGHT, opponent_color):
        attackers_bb = board.attackers(own_color, knight_sq)
        if attackers_bb:
            knight_square_name = chess.square_name(knight_sq)
            attacker_squares = [chess.square_name(sq) for sq in attackers_bb]
            results.append((knight_square_name, attacker_squares))
    return results 