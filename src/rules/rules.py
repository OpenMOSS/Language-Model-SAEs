import chess
import chess.engine

ENGINE_PATH = "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/Stockfish/src/stockfish"

def safe_board(fen: str):
    try:
        return chess.Board(fen)
    except ValueError:
        print(f"Warning: Invalid FEN string: {fen}")
        return None

def can_capture_opponent_rook(fen: str) -> bool:
    board = safe_board(fen)
    if board is None:
        return False

    opponent_rook = 'R' if board.turn == chess.BLACK else 'r'

    for move in board.legal_moves:
        target_piece = board.piece_at(move.to_square)
        if target_piece and target_piece.symbol() == opponent_rook:
            return True

    return False


def is_own_knight_under_attack(fen: str) -> bool:
    board = safe_board(fen)
    if board is None:
        return False

    own_knight = 'N' if board.turn == chess.WHITE else 'n'

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if (
            piece
            and piece.symbol() == own_knight
            and board.is_attacked_by(not board.turn, square)
        ):
            return True

    return False

def is_own_rook_under_attack(fen: str) -> bool:
    board = safe_board(fen)
    if board is None:
        return False
    own_rook = 'R' if board.turn == chess.WHITE else 'r'

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if (
            piece
            and piece.symbol() == own_rook
            and board.is_attacked_by(not board.turn, square)
        ):
            return True

    return False


def can_checkmate_in_1_move(fen: str) -> bool:
    board = safe_board(fen)
    if board is None:
        return False
    if board.is_checkmate():
        return True
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return True
        board.pop()

    return False
