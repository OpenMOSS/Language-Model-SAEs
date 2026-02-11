"""Chess plotting utilities for creating interactive chessboard heatmaps."""

from __future__ import annotations

from typing import Literal

import numpy as np
import plotly.graph_objects as go

N_POS: int = 64
PIECE_SYMBOLS: dict[str, str] = {"K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙", "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}


def parse_fen_board(fen_str: str) -> tuple[list[list[str | None]], str]:
    """Return (8x8 board, active_color)."""
    parts = fen_str.strip().split(" ")
    if len(parts) < 2:
        raise ValueError(f"Invalid FEN: {fen_str}")
    board_part, active_color = parts[0], parts[1]
    rows = board_part.split("/")
    if len(rows) != 8:
        raise ValueError(f"Invalid board in FEN: {fen_str}")
    board: list[list[str | None]] = []
    for row_str in rows:
        row: list[str | None] = []
        for ch in row_str:
            row.extend([None] * int(ch) if ch.isdigit() else [ch])
        if len(row) != 8:
            raise ValueError(f"Invalid row in FEN: {row_str}")
        board.append(row)
    return board, active_color


def vals64_to_board(vals64: np.ndarray, *, active_color: str = "w", flip_for_black: bool = True) -> np.ndarray:
    """Reshape (64,) -> (8,8) with rank8 at row0. Flip board for black to move if flip_for_black=True."""
    if vals64.shape != (64,):
        raise ValueError(f"expected (64,), got {vals64.shape}")
    board = vals64.reshape(8, 8)[::-1, :]
    if active_color.lower() == "b" and flip_for_black:
        board = board[::-1, :]
    return board


def make_board_fig(z: np.ndarray, *, board_2d: list[list[str | None]], active_color: str, title: str, colorbar_title: str) -> go.Figure:
    """Create chessboard heatmap with pieces overlay."""
    files, ranks = list("abcdefgh"), ["8", "7", "6", "5", "4", "3", "2", "1"]

    # Flip board visualization for black to move
    if active_color.lower() == "b":
        ranks = ranks[::-1]
        board_2d = board_2d[::-1]

    checker = np.fromfunction(lambda r, c: ((r + c) % 2 == 0).astype(int), (8, 8))
    checker_cs = [[0.0, "#B58863"], [0.4999, "#B58863"], [0.5, "#F0D9B5"], [1.0, "#F0D9B5"]]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=checker, x=list(range(8)), y=list(range(8)), colorscale=checker_cs, showscale=False, hoverinfo="skip", xgap=1, ygap=1))
    fig.add_trace(go.Heatmap(z=z, x=list(range(8)), y=list(range(8)), colorscale="RdBu", zmid=0.0, opacity=0.78, colorbar=dict(title=colorbar_title), xgap=1, ygap=1, hovertemplate="%{customdata}<br>%{z:.6f}<extra></extra>", customdata=[[f"{files[c]}{ranks[r]}" for c in range(8)] for r in range(8)]))
    for r in range(8):
        for c in range(8):
            p = board_2d[r][c]
            if p is None:
                continue
            fig.add_annotation(x=c, y=r, text=PIECE_SYMBOLS.get(p, p), showarrow=False, font=dict(size=26, color=("#111111" if p.islower() else "#FAFAFA")))
    fig.update_layout(title=f"{title}（to-move={active_color}）", width=650, height=650, margin=dict(l=30, r=30, t=60, b=30))
    fig.update_xaxes(tickmode="array", tickvals=list(range(8)), ticktext=files, showgrid=False, zeroline=False, constrain="domain")
    fig.update_yaxes(tickmode="array", tickvals=list(range(8)), ticktext=ranks, autorange="reversed", showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    return fig


def plot_board_heatmap(fen: str, values: np.ndarray, title: str = "Chess Board Heatmap", colorbar_title: str = "Value") -> go.Figure:
    """Create a chessboard heatmap directly from FEN string and values array.

    Args:
        fen: FEN string representing the board position
        values: Array of 64 values, one for each square (0=a1, 1=b1, ..., 63=h8)
        title: Plot title
        colorbar_title: Colorbar title

    Returns:
        Plotly figure with chessboard heatmap

    Note:
        The values array should be indexed as: 0=a1, 1=b1, 2=c1, ..., 7=h1, 8=a2, ..., 63=h8
        The visualization will automatically flip the board for black to move to match
        the correct perspective.
    """
    if values.shape != (64,):
        raise ValueError(f"Values must be shape (64,), got {values.shape}")

    # Parse FEN to get board layout and active color
    board_2d, active_color = parse_fen_board(fen)

    # Convert values to board layout
    board_values = vals64_to_board(values, active_color=active_color, flip_for_black=False)

    # For black to move, flip both data and board layout to match visualization
    if active_color.lower() == "b":
        board_values = board_values[::-1]  # Flip the data
        board_2d = board_2d[::-1]  # Flip the board layout

    # Create the heatmap figure
    fig = make_board_fig(
        board_values,
        board_2d=board_2d,
        active_color=active_color,
        title=title,
        colorbar_title=colorbar_title
    )

    return fig