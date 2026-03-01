"""Implements LeelaBoard, which is a wrapper around the python-chess board that can
produce Leela-formatted inputs and has other useful methods.

Based on https://github.com/so-much-meta/lczero_tools/blob/master/src/lcztools/_leela_board.py
(GPL-3.0). Updated for newer versions of Leela and added plotting + interp helpers.
"""

import collections
import io
import struct

import chess
import chess.pgn
import chess.svg
import numpy as np
import pandas as pd
import torch
from chess import Move
from matplotlib.cm import ScalarMappable

# from .iceberg_board import IcebergBoard
from .uci_to_idx import idx_to_uci as _idx_to_uci
from .uci_to_idx import uci_to_idx as _uci_to_idx


'''from utils begins'''
_LETTERS = {letter: i for i, letter in enumerate("abcdefgh")}


def sq2idx(sq: str, turn: bool):
    file, row = sq
    file = _LETTERS[file]
    if not turn:
        # Black's turn
        row = 9 - int(row)
    return (int(row) - 1) * 8 + file


_SQUARES = [f"{file}{rank}" for file in _LETTERS for rank in range(1, 9)]

_IDX2SQ = {
    True: {sq2idx(sq, True): sq for sq in _SQUARES},
    False: {sq2idx(sq, False): sq for sq in _SQUARES},
}


def idx2sq(idx: int, turn: bool):
    return _IDX2SQ[turn][idx]
'''from utils ends'''




flat_planes = []
for i in range(256):
    flat_planes.append(np.ones((8, 8), dtype=np.uint8) * i)

LeelaBoardData = collections.namedtuple(
    "LeelaBoardData",
    "plane_bytes repetition "
    "transposition_key us_ooo us_oo them_ooo them_oo "
    "side_to_move rule50_count",
)


def pc_board_property(propertyname):
    """Create a property based on self.pc_board"""

    def prop(self):
        return getattr(self.pc_board, propertyname)

    return property(prop)


class LeelaBoard:
    turn = pc_board_property("turn")
    move_stack = pc_board_property("move_stack")
    _plane_bytes_struct = struct.Struct(">Q")

    def __init__(self):
        """If leela_board is passed as an argument, return a copy"""
        self.pc_board = chess.Board()
        self.lcz_stack = []
        self._lcz_transposition_counter = collections.Counter()
        self._lcz_push()
        self.is_game_over = self.pc_method("is_game_over")
        self.can_claim_draw = self.pc_method("can_claim_draw")
        self.generate_legal_moves = self.pc_method("generate_legal_moves")

    def fen(self) -> str:
        return self.pc_board.fen()

    def sq2idx(self, square: str) -> int:
        return sq2idx(square, self.turn)

    def idx2sq(self, idx: int) -> str:
        return idx2sq(idx, self.turn)

    def chess_sq2idx(self, square: chess.Square) -> int:
        return self.sq2idx(chess.square_name(square))

    def idx2chess_sq(self, idx: int) -> chess.Square:
        return chess.parse_square(self.idx2sq(idx))

    def uci2idx(self, uci: str) -> int:
        return self._uci_to_idx_dict()[uci]

    def idx2uci(self, idx: int) -> str:
        return self._idx_to_uci_dict()[idx]

    def plot(
        self,
        heatmap: torch.Tensor
        | np.ndarray
        | list[str]
        | dict[str, str | float]
        | None = None,
        moves: str | list[str] | None = None,
        highlight: str | None = None,
        caption: str | None = None,
        cmap: str = "YlOrRd",
        mappable: ScalarMappable | None = None,
        zero_center: bool = False,
        arrows: dict[str, str] | None = None,
        attn_map: torch.Tensor | np.ndarray | None = None,
        show_lastmove: bool = True,
    ):
        return IcebergBoard(
            board=self.pc_board,
            heatmap=heatmap,
            next_moves=moves,
            highlight=highlight,
            caption=caption,
            cmap=cmap,
            mappable=mappable,
            zero_center=zero_center,
            arrows=arrows,
            attn_map=attn_map,
            show_lastmove=show_lastmove,
        )

    @classmethod
    def from_uci(cls, uci_moves: list[str]):
        """Create a LeelaBoard from a list of UCI moves"""
        board = cls()
        for uci_move in uci_moves:
            board.push_uci(uci_move)
        return board

    @classmethod
    def from_pgn(cls, pgn: str):
        """Create a LeelaBoard from a PGN string"""
        game = chess.pgn.read_game(io.StringIO(pgn))
        uci_moves = [move.uci() for move in game.mainline_moves()]
        return cls.from_uci(uci_moves)

    @classmethod
    def from_fen(
        cls,
        fen: str,
        moves: list[str] | None = None,
        uci: bool = False,
        history_synthesis: bool = False,
    ):
        """Create a LeelaBoard from a FEN string.

        If `moves` is not None, apply the moves starting from the FEN position,
        then return the board *after* the moves have been applied. This means Lc0
        will have access to the move history.

        If `history_synthesis` is set, repeat the current position 8 times to fill
        Leela's buffer. The model we use is finetuned to ignore history, and zeros
        out any history it does get, so for that one it doesn't make a difference.
        When using original versions of Leela, we recommend setting this to True.

        Moves are expected to be SAN, not UCI, unless uci=True.
        """
        board = chess.Board(fen)

        leela_board = cls()
        leela_board.pc_board = board
        # HACK: these will be references to the old pc_board, which was just
        # the initial position. Need to reset them.
        # TODO: maybe should just re-implement __init__ and use __new__?
        leela_board.is_game_over = leela_board.pc_method("is_game_over")
        leela_board.can_claim_draw = leela_board.pc_method("can_claim_draw")
        leela_board.generate_legal_moves = leela_board.pc_method("generate_legal_moves")
        # HACK: this will have the initial board state already after initialization,
        # we want to get rid of that
        leela_board.lcz_stack = []
        # Now push the correct board state
        leela_board._lcz_push()

        if history_synthesis:
            # Repeat the initial board state
            n_moves = len(moves) if moves is not None else 0
            while len(leela_board.lcz_stack) + n_moves < 8:
                leela_board._lcz_push()

        if moves is not None:
            for move in moves:
                if uci:
                    leela_board.push_uci(move)
                else:
                    leela_board.push_san(move)

        return leela_board

    @classmethod
    def from_puzzle(cls, puzzle: pd.Series, fast: bool = True):
        """Load a board from the Lichess puzzle pandas DataFrame.

        Note that the FEN field in the puzzle DataFrame is the position one ply before
        the main puzzle position, so don't just use `from_fen(puzzle["FEN"])`,
        use this method instead!
        """
        fen = puzzle["FEN"]

        if fast:
            # Don't play through the whole game, just play the first move to get the
            # actual puzzle position. This is significantly faster, but Lc0 won't have
            # access to the full move history.
            # (Anecdotally, this doesn't cause problems.)
            return LeelaBoard.from_fen(fen, puzzle["Moves"].split(" ")[:1], uci=True)

        fen_board = chess.Board(fen)

        game = chess.pgn.read_game(io.StringIO(puzzle["PGN"]))
        moves = list(game.mainline_moves())
        uci_moves = [move.uci() for move in moves]
        leela_board = cls()
        moves_so_far = []
        for move in uci_moves:
            if leela_board.pc_board == fen_board:
                break
            leela_board.push_uci(move)
            moves_so_far.append(move)

        next_moves = puzzle["Moves"].split(" ")

        moves_so_far.append(next_moves[0])
        leela_board.push_uci(next_moves[0])

        return leela_board

    def copy(self, history=7):
        """Note! Currently the copy constructor uses pc_board.copy(stack=False), which
        makes pops impossible
        """
        cls = type(self)
        copied = cls.__new__(cls)
        copied.pc_board = self.pc_board.copy(stack=False)
        # copied.pc_board.stack[:] = self.pc_board.stack[-history:]
        copied.pc_board.move_stack[:] = self.pc_board.move_stack[-history:]
        copied.lcz_stack = self.lcz_stack[-history:]
        copied._lcz_transposition_counter = self._lcz_transposition_counter.copy()
        copied.is_game_over = copied.pc_method("is_game_over")
        copied.can_claim_draw = copied.pc_method("can_claim_draw")
        copied.generate_legal_moves = copied.pc_method("generate_legal_moves")
        return copied

    def pc_method(self, methodname):
        """Return attribute of self.pc_board, useful for copying method bindings"""
        return getattr(self.pc_board, methodname)

    def is_threefold(self):
        transposition_key = self.pc_board._transposition_key()
        return self._lcz_transposition_counter[transposition_key] >= 3

    def is_fifty_moves(self):
        return self.pc_board.halfmove_clock >= 100

    def is_draw(self):
        return self.is_threefold() or self.is_fifty_moves()

    def push(self, move):
        self.pc_board.push(move)
        self._lcz_push()

    def push_uci(self, uci):
        # don't check for legality - it takes much longer to run...
        # self.pc_board.push_uci(uci)
        self.pc_board.push(Move.from_uci(uci))
        self._lcz_push()

    def push_san(self, san):
        self.pc_board.push_san(san)
        self._lcz_push()

    def pop(self):
        result = self.pc_board.pop()
        _lcz_data = self.lcz_stack.pop()
        self._lcz_transposition_counter.subtract((_lcz_data.transposition_key,))
        return result

    def _plane_bytes_iter(self):
        """Get plane bytes... used for _lcz_push"""
        pack = self._plane_bytes_struct.pack
        pieces_mask = self.pc_board.pieces_mask
        for color in (True, False):
            for piece_type in range(1, 7):
                byts = pack(pieces_mask(piece_type, color))
                yield byts

    def _lcz_push(self):
        """Push data onto the lcz data stack after pushing board moves"""
        transposition_key = self.pc_board._transposition_key()
        self._lcz_transposition_counter.update((transposition_key,))
        repetitions = self._lcz_transposition_counter[transposition_key] - 1
        # side_to_move = 0 if we're white, 1 if we're black
        side_to_move = 0 if self.pc_board.turn else 1
        rule50_count = self.pc_board.halfmove_clock
        # Figure out castling rights
        if not side_to_move:
            # we're white
            _c = self.pc_board.castling_rights
            us_ooo, us_oo = (_c >> chess.A1) & 1, (_c >> chess.H1) & 1
            them_ooo, them_oo = (_c >> chess.A8) & 1, (_c >> chess.H8) & 1
        else:
            # We're black
            _c = self.pc_board.castling_rights
            us_ooo, us_oo = (_c >> chess.A8) & 1, (_c >> chess.H8) & 1
            them_ooo, them_oo = (_c >> chess.A1) & 1, (_c >> chess.H1) & 1
        # Create 13 planes... 6 us, 6 them, repetitions>=1
        plane_bytes = b"".join(self._plane_bytes_iter())
        repetition = repetitions >= 1
        lcz_data = LeelaBoardData(
            plane_bytes,
            repetition=repetition,
            transposition_key=transposition_key,
            us_ooo=us_ooo,
            us_oo=us_oo,
            them_ooo=them_ooo,
            them_oo=them_oo,
            side_to_move=side_to_move,
            rule50_count=rule50_count,
        )
        self.lcz_stack.append(lcz_data)

    def serialize_features(self):
        """Get compacted bytes representation of input planes"""
        curdata = self.lcz_stack[-1]
        bytes_false_true = bytes([False]), bytes([True])
        bytes_per_history = 97
        total_plane_bytes = bytes_per_history * 8

        def bytes_iter():
            plane_bytes_yielded = 0
            for data in self.lcz_stack[-1:-9:-1]:
                yield data.plane_bytes
                yield bytes_false_true[data.repetition]
                plane_bytes_yielded += bytes_per_history
            # 104 total piece planes... fill in missing with 0s
            yield bytes(total_plane_bytes - plane_bytes_yielded)
            # Yield the rest of the constant planes
            yield np.packbits(
                (
                    curdata.us_ooo,
                    curdata.us_oo,
                    curdata.them_ooo,
                    curdata.them_oo,
                    curdata.side_to_move,
                )
            ).tobytes()
            yield chr(curdata.rule50_count).encode()

        return b"".join(bytes_iter())

    @classmethod
    def deserialize_features(cls, serialized):
        planes_stack = []
        rule50_count = serialized[-1]  # last byte is rule 50
        board_attrs = np.unpackbits(
            memoryview(serialized[-2:-1])
        )  # second to last byte
        us_ooo, us_oo, them_ooo, them_oo, side_to_move = board_attrs[:5]
        bytes_per_history = 97
        for history_idx in range(0, bytes_per_history * 8, bytes_per_history):
            plane_bytes = serialized[history_idx : history_idx + 96]
            repetition = serialized[history_idx + 96]
            if not side_to_move:
                # we're white
                planes = np.unpackbits(memoryview(plane_bytes))[::-1].reshape(12, 8, 8)[
                    ::-1
                ]
            else:
                # We're black
                planes = (
                    np.unpackbits(memoryview(plane_bytes))[::-1]
                    .reshape(12, 8, 8)[::-1]
                    .reshape(2, 6, 8, 8)[::-1, :, ::-1]
                    .reshape(12, 8, 8)
                )
            planes_stack.append(planes)
            planes_stack.append([flat_planes[repetition]])
        planes_stack.append(
            [
                flat_planes[us_ooo],
                flat_planes[us_oo],
                flat_planes[them_ooo],
                flat_planes[them_oo],
                flat_planes[side_to_move],
                flat_planes[rule50_count],
                flat_planes[0],
                flat_planes[1],
            ]
        )
        planes = np.concatenate(planes_stack)
        return planes

    def lcz_features(self, no_history=False):
        """Get neural network input planes as uint8"""
        # print(list(self._planes_iter()))
        planes_stack = []
        curdata = self.lcz_stack[-1]
        planes_yielded = 0
        for data in self.lcz_stack[-1:-9:-1]:
            plane_bytes = data.plane_bytes
            if not curdata.side_to_move:
                # we're white
                planes = np.unpackbits(memoryview(plane_bytes))[::-1].reshape(12, 8, 8)[
                    ::-1
                ]
            else:
                # We're black
                planes = (
                    np.unpackbits(memoryview(plane_bytes))[::-1]
                    .reshape(12, 8, 8)[::-1]
                    .reshape(2, 6, 8, 8)[::-1, :, ::-1]
                    .reshape(12, 8, 8)
                )
            planes_stack.append(planes)
            planes_stack.append([flat_planes[data.repetition]])
            planes_yielded += 13
        empty_planes = [flat_planes[0] for _ in range(104 - planes_yielded)]
        if empty_planes:
            planes_stack.append(empty_planes)
        # Yield the rest of the constant planes
        planes_stack.append(
            [
                flat_planes[curdata.us_ooo],
                flat_planes[curdata.us_oo],
                flat_planes[curdata.them_ooo],
                flat_planes[curdata.them_oo],
                flat_planes[curdata.side_to_move],
                flat_planes[curdata.rule50_count],
                flat_planes[0],
                flat_planes[1],
            ]
        )
        planes = np.concatenate(planes_stack)

        if no_history:
            # If no history is allowed then we zero out the history planes.
            planes[12:104] = 0
        print("12,0,0,:", planes[12,0,0])
        return planes

    def _uci_to_idx_dict(self):
        data = self.lcz_stack[-1]
        # uci_to_idx_index =
        #  White, no-castling => 0
        #  White, castling => 1
        #  Black, no-castling => 2
        #  Black, castling => 3
        uci_to_idx_index = (data.us_ooo | data.us_oo) + 2 * data.side_to_move
        return _uci_to_idx[uci_to_idx_index]

    def _idx_to_uci_dict(self):
        data = self.lcz_stack[-1]
        uci_to_idx_index = (data.us_ooo | data.us_oo) + 2 * data.side_to_move
        return _idx_to_uci[uci_to_idx_index]

    def batch_uci2idx(self, uci_list):
        # Return list of NN policy output indexes for this board position, given uci_list

        # TODO: Perhaps it's possible to just add the uci knight promotion move to the index dict
        # currently knight promotions are not in the dict
        uci_list = [uci.rstrip("n") for uci in uci_list]

        uci_idx_dct = self._uci_to_idx_dict()
        return [uci_idx_dct[m] for m in uci_list]

    def __repr__(self):
        return "LeelaBoard('{}')".format(self.pc_board.fen())

    def _repr_svg_(self):
        return chess.svg.board(
            board=self.pc_board,
            size=390,
            lastmove=self.pc_board.peek() if self.pc_board.move_stack else None,
            check=self.pc_board.king(self.pc_board.turn)
            if self.pc_board.is_check()
            else None,
            colors={
                "square light": "#f5f5f5",
                "square dark": "#cfcfcf",
                "square light lastmove": "#cfcfff",
                "square dark lastmove": "#a0a0ff",
            },
        )

    def __str__(self):
        if self.pc_board.is_game_over() or self.is_draw():
            result = self.pc_board.result(claim_draw=True)
            turnstring = "Result: {}".format(result)
        else:
            turnstring = "Turn: {}".format("White" if self.pc_board.turn else "Black")
        boardstr = self.pc_board.__str__() + "\n" + turnstring
        return boardstr

    def __eq__(self, other):
        return self.get_hash_key() == other.get_hash_key()

    def __hash__(self):
        return hash(self.get_hash_key())

    def get_hash_key(self):
        transposition_key = self.pc_board._transposition_key()
        return (
            transposition_key
            + (
                self._lcz_transposition_counter[transposition_key],
                self.pc_board.halfmove_clock,
            )
            + tuple(self.pc_board.move_stack[-7:])
        )

    def uci_to_positions(self, move_uci: str) -> torch.Tensor:
        """
        convert UCI move to start and end positions
        """
        
        def uci_to_coords(uci: str) -> tuple[int, int]:
            """convert UCI coordinate to board coordinate"""
            file = ord(uci[0]) - ord('a')  # a->0, b->1, ..., h->7
            rank = int(uci[1]) - 1         # 1->0, 2->1, ..., 8->7
            return rank, file
        
        def coords_to_1d(rank: int, file: int) -> int:
            """convert board coordinate to 1D index"""
            return rank * 8 + file
        
        def flip_coords(rank: int, file: int) -> tuple[int, int]:
            """flip coordinates (black to white perspective) - only flip rank, file remains unchanged"""
            flipped_rank = 7 - rank  # 0->7, 1->6, ..., 7->0
            return flipped_rank, file
        
        # get side to move from current board state
        side = 'w' if self.pc_board.turn else 'b'
        
        start_uci = move_uci[:2]
        end_uci = move_uci[2:4]
        start_rank, start_file = uci_to_coords(start_uci)
        end_rank, end_file = uci_to_coords(end_uci)
        
        # if black, need to flip coordinates
        if side == 'b':
            start_rank, start_file = flip_coords(start_rank, start_file)
            end_rank, end_file = flip_coords(end_rank, end_file)
        
        start_pos = coords_to_1d(start_rank, start_file)
        end_pos = coords_to_1d(end_rank, end_file)
        
        return torch.tensor([start_pos, end_pos])
