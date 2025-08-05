import chess

__all__ = ["is_queen_under_attack", "queen_under_attack_details"]


def is_queen_under_attack(fen: str) -> bool:
    """判断在给定 FEN 局面下，当前行动方 (side to move) 的任意一只后是否正被对方棋子攻击。

    *被抓* 定义：若对方可以在下一步用以下棋子直接吃掉该后，则认为该后被抓。允许的攻击
    棋子类型包括：车、马、兵、象、后、王，即 *python-chess* 中的所有常规棋子。

    Args:
        fen: 合法的 FEN 字符串。

    Returns:
        bool: 只要存在至少一只己方后被对方攻击，返回 ``True``；否则返回 ``False``。

    Raises:
        ValueError: 当 ``fen`` 字符串无法被解析时抛出。
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"无效的 FEN: {fen}") from exc

    own_color = board.turn            # 当前行棋方
    opponent_color = not own_color

    # 找到己方所有后所在的格子
    queen_squares = board.pieces(chess.QUEEN, own_color)

    for queen_sq in queen_squares:
        # 任何来自对方的攻击都算
        if board.is_attacked_by(opponent_color, queen_sq):
            return True
    return False


def queen_under_attack_details(fen: str) -> list[tuple[str, list[str]]]:
    """返回更详细的被抓信息。

    Args:
        fen: FEN 字符串。

    Returns:
        list[tuple[str, list[str]]]: 每个元素包含两个部分：
            1. 被攻击的己方后所在格子（如 ``"d1"``）
            2. 一个列表，列出所有对该后构成直接威胁的对方棋子格子（如 ``["e2", "c4"]``）
        如果没有任何后被抓，则返回空列表。
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"无效的 FEN: {fen}") from exc

    own_color = board.turn
    opponent_color = not own_color

    results: list[tuple[str, list[str]]] = []
    for queen_sq in board.pieces(chess.QUEEN, own_color):
        attackers_bb = board.attackers(opponent_color, queen_sq)
        if attackers_bb:
            queen_square_name = chess.square_name(queen_sq)
            attacker_squares = [chess.square_name(sq) for sq in attackers_bb]
            results.append((queen_square_name, attacker_squares))
    return results 