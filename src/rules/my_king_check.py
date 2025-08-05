import chess

__all__ = ["is_king_in_check", "king_check_details", "is_checkmate", "is_stalemate"]


def is_king_in_check(fen: str) -> bool:
    """判断在给定 FEN 局面下，当前行动方 (side to move) 的王是否正被对方棋子攻击（将军）。

    *将军* 定义：若对方可以在下一步用以下棋子直接吃掉该王，则认为该王被将军。允许的攻击
    棋子类型包括：车、马、兵、象、后、王，即 *python-chess* 中的所有常规棋子。

    Args:
        fen: 合法的 FEN 字符串。

    Returns:
        bool: 如果己方王被对方攻击，返回 ``True``；否则返回 ``False``。

    Raises:
        ValueError: 当 ``fen`` 字符串无法被解析时抛出。
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"无效的 FEN: {fen}") from exc

    # 直接使用 python-chess 的内置方法检查将军
    return board.is_check()


def king_check_details(fen: str) -> tuple[str | None, list[str]]:
    """返回更详细的将军信息。

    Args:
        fen: FEN 字符串。

    Returns:
        tuple[str | None, list[str]]: 包含两个部分：
            1. 被攻击的己方王所在格子（如 ``"e1"``），如果没有被将军则为 ``None``
            2. 一个列表，列出所有对该王构成直接威胁的对方棋子格子（如 ``["d8", "f6"]``）
        如果王没有被将军，则返回 ``(None, [])``。
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"无效的 FEN: {fen}") from exc

    own_color = board.turn
    opponent_color = not own_color

    # 找到己方王的位置
    king_square = board.king(own_color)
    
    if king_square is None:
        # 理论上不应该发生，但为了健壮性
        return (None, [])
    
    # 检查是否有对方棋子攻击王
    attackers_bb = board.attackers(opponent_color, king_square)
    
    if attackers_bb:
        king_square_name = chess.square_name(king_square)
        attacker_squares = [chess.square_name(sq) for sq in attackers_bb]
        return (king_square_name, attacker_squares)
    else:
        return (None, [])


def is_checkmate(fen: str) -> bool:
    """判断在给定 FEN 局面下，当前行动方是否被将死。

    Args:
        fen: 合法的 FEN 字符串。

    Returns:
        bool: 如果当前行动方被将死，返回 ``True``；否则返回 ``False``。

    Raises:
        ValueError: 当 ``fen`` 字符串无法被解析时抛出。
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"无效的 FEN: {fen}") from exc

    return board.is_checkmate()


def is_stalemate(fen: str) -> bool:
    """判断在给定 FEN 局面下，当前行动方是否逼和（无子可动但未被将军）。

    Args:
        fen: 合法的 FEN 字符串。

    Returns:
        bool: 如果当前行动方被逼和，返回 ``True``；否则返回 ``False``。

    Raises:
        ValueError: 当 ``fen`` 字符串无法被解析时抛出。
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"无效的 FEN: {fen}") from exc

    return board.is_stalemate() 