import chess

# 1 fen, square -> pos
def get_pos_from_square(fen: str, square: str) -> int:
    board = chess.Board(fen)
    sq = chess.parse_square(square)
    if board.turn == chess.BLACK:
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        sq = chess.square(file_idx, 7 - rank_idx)
    return sq

# 2 fen, piece_type -> bool
def has_piece_of_type(fen: str, piece_type: str) -> bool:
    assert piece_type in ['my p', 'my n', 'my b', 'my r', 'my q', 'my k', "opponent's p", "opponent's n", "opponent's b", "opponent's r", "opponent's q", "opponent's k"]
    board = chess.Board(fen)
    side_to_move = board.turn
    owner_str, piece_str = piece_type.split()
    if owner_str == 'my':
        color = side_to_move
    else:
        color = not side_to_move
    piece_map = {
        'p': chess.PAWN,
        'n': chess.KNIGHT,
        'b': chess.BISHOP,
        'r': chess.ROOK,
        'q': chess.QUEEN,
        'k': chess.KING,
    }
    piece_type_enum = piece_map[piece_str]
    return len(board.pieces(piece_type_enum, color)) > 0


# 3 fen, piece_type -> pos list
def get_piece_type_pos(fen: str, piece_type: str) -> list[int]:
    assert piece_type in [
        "my p",
        "my n",
        "my b",
        "my r",
        "my q",
        "my k",
        "opponent's p",
        "opponent's n",
        "opponent's b",
        "opponent's r",
        "opponent's q",
        "opponent's k",
    ]
    if not has_piece_of_type(fen, piece_type):
        return []
    board = chess.Board(fen)
    side_to_move = board.turn
    owner_str, piece_str = piece_type.split()
    if owner_str == "my":
        color = side_to_move
    else:
        color = not side_to_move
    piece_map = {
        "p": chess.PAWN,
        "n": chess.KNIGHT,
        "b": chess.BISHOP,
        "r": chess.ROOK,
        "q": chess.QUEEN,
        "k": chess.KING,
    }
    piece_type_enum = piece_map[piece_str]
    pos_list = list(board.pieces(piece_type_enum, color))
    if board.turn == chess.BLACK:
        flipped = []
        for sq in pos_list:
            file_idx = chess.square_file(sq)
            rank_idx = chess.square_rank(sq)
            flipped.append(chess.square(file_idx, 7 - rank_idx))
        pos_list = flipped
    return pos_list


# 4 fen, move_uci -> {start_pos: int, end_pos: int}
def get_start_end_pos_from_move_uci(fen: str, move_uci: str) -> tuple[int, int]:
    board = chess.Board(fen)
    start_square = move_uci[:2]
    end_square = move_uci[2:4]
    start_pos = get_pos_from_square(fen, start_square)
    end_pos = get_pos_from_square(fen, end_square)
    return start_pos, end_pos

# 5 fen, move_uci -> bool
def is_valid_move_uci(fen: str, move_uci: str) -> bool:
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    return board.is_legal(move)
