import chess
import chess.engine
from typing import Optional, Dict, Any, Tuple
ENGINE_PATH = "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/Stockfish/src/stockfish"
ENGINE_TIME_LIMIT = 0.2


def score_to_cp(s: chess.engine.PovScore, mate_cp_val: int) -> float:
    try:
        if s.is_mate():
            m = s.mate()
            if m is None:
                return 0.0
            return float(mate_cp_val if m > 0 else -mate_cp_val)
        v = s.score()
        return 0.0 if v is None else float(v)
    except Exception:
        return 0.0


def expected_score_from_wdl(wdl: Tuple[float, float, float]) -> Optional[float]:
    try:
        if wdl is None:
            return None
        p_win, p_draw, _ = wdl
        if p_win is None or p_draw is None:
            return None
        return float(p_win) + 0.5 * float(p_draw)
    except Exception:
        return None


def map_cp_loss_to_score(cp_loss: Optional[float], scale: float = 80.0) -> Optional[float]:
    try:
        if cp_loss is None:
            return None
        if cp_loss <= 0:
            return 100.0
        return 100.0 / (1.0 + (cp_loss / max(1e-6, scale)))
    except Exception:
        return None


def compute_move_score(
    best_cp: Optional[float],
    best_wdl: Optional[Tuple[float, float, float]],
    my_cp: Optional[float],
    my_wdl: Optional[Tuple[float, float, float]],
    *,
    weight_wdl: float = 0.6,
    cp_scale: float = 80.0,
    min_best_E: float = 0.05,
) -> Tuple[float, dict]:
    E_best = expected_score_from_wdl(best_wdl)
    E_my = expected_score_from_wdl(my_wdl)

    wdl_score = None
    if E_best is not None and E_my is not None:
        denom = max(E_best, min_best_E)
        ratio = max(0.0, min(1.0, E_my / denom))
        wdl_score = 100.0 * ratio

    cp_score = None
    if best_cp is not None and my_cp is not None:
        cp_loss = best_cp - my_cp
        cp_score = map_cp_loss_to_score(cp_loss, scale=cp_scale)
    else:
        cp_loss = None

    if wdl_score is not None and cp_score is not None:
        final_score = float(weight_wdl * wdl_score + (1.0 - weight_wdl) * cp_score)
    elif wdl_score is not None:
        final_score = float(wdl_score)
    elif cp_score is not None:
        final_score = float(cp_score)
    else:
        final_score = 0.0

    details = {
        "E_best": E_best,
        "E_my": E_my,
        "wdl_score": wdl_score,
        "cp_score": cp_score,
        "cp_loss": cp_loss,
        "weight_wdl": weight_wdl,
        "cp_scale": cp_scale,
    }
    return final_score, details


def analyze_board_state(fen: str, time_limit: float = ENGINE_TIME_LIMIT, mate_cp: int = 10000, pov_color: Optional[chess.Color] = None):
    try:
        board = chess.Board(fen)
    except ValueError:
        return None, None

    if pov_color is None:
        pov_color = board.turn

    try:
        with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
            opts = getattr(engine, "options", {})
            if "UCI_AnalyseMode" in opts:
                engine.configure({"UCI_AnalyseMode": True})
            if "UCI_ShowWDL" in opts:
                engine.configure({"UCI_ShowWDL": True})

            info = engine.analyse(board, chess.engine.Limit(depth=12), info=chess.engine.INFO_ALL)
            score = info["score"].pov(pov_color)
            cp_val = score_to_cp(score, mate_cp)

            p_win = p_draw = p_loss = None
            if "wdl" in info:
                try:
                    pov_wdl = info["wdl"]
                    try:
                        wdl_obj = pov_wdl.pov(pov_color)
                    except Exception:
                        wdl_obj = getattr(pov_wdl, "wdl", pov_wdl)
                    p_win, p_draw, p_loss = (
                        wdl_obj.wins/1000.0,
                        wdl_obj.draws/1000.0,
                        wdl_obj.losses/1000.0,
                    )
                except Exception:
                    p_win = p_draw = p_loss = None

            if p_win is None:
                try:
                    if score.is_mate():
                        m = score.mate()
                        if m and m > 0:
                            p_win, p_draw, p_loss = 1.0, 0.0, 0.0
                        else:
                            p_win, p_draw, p_loss = 0.0, 0.0, 1.0
                    else:
                        model = chess.engine.CpWdlModel()
                        wdl_est = score.wdl(model=model)
                        p_win, p_draw, p_loss = (
                            wdl_est.wins/1000.0,
                            wdl_est.draws/1000.0,
                            wdl_est.losses/1000.0,
                        )
                except Exception:
                    p_win, p_draw, p_loss = 0.0, 0.0, 0.0

            return cp_val, (p_win, p_draw, p_loss)
    except Exception:
        return None, None


def evaluate_move_quality(
    fen: str,
    move_str: str,
    time_limit: float = ENGINE_TIME_LIMIT,
    mate_cp: int = 10000,
    multipv: int = 3,
) -> Optional[Dict[str, Any]]:
    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    move: Optional[chess.Move] = None
    try:
        move = chess.Move.from_uci(move_str)
        if move not in board.legal_moves:
            move = None
    except Exception:
        move = None
    if move is None:
        try:
            move = board.parse_san(move_str)
        except Exception:
            return None

    if move not in board.legal_moves:
        return None

    pov_color = board.turn

    try:
        with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
            opts = getattr(engine, "options", {})
            if "UCI_AnalyseMode" in opts:
                engine.configure({"UCI_AnalyseMode": True})
            if "UCI_ShowWDL" in opts:
                engine.configure({"UCI_ShowWDL": True})

            limit = chess.engine.Limit(depth=12)
            infos = engine.analyse(board, limit, multipv=max(1, multipv), info=chess.engine.INFO_ALL)
            if isinstance(infos, dict):
                infos = [infos]
            def _score_value(d):
                try:
                    s = d["score"].pov(pov_color)
                    return score_to_cp(s, mate_cp)
                except Exception:
                    return float('-inf')
            infos = sorted(infos, key=_score_value, reverse=True)

            best_info = infos[0]
            best_move = None
            if "pv" in best_info and len(best_info["pv"]) > 0:
                best_move = best_info["pv"][0].uci()

            pov = board.turn
            def _analyse_board(b: chess.Board):
                info = engine.analyse(b, limit, info=chess.engine.INFO_ALL)
                score = info["score"].pov(pov)
                cp_val = score_to_cp(score, mate_cp)
                p_win = p_draw = p_loss = None
                if "wdl" in info:
                    try:
                        pov_wdl = info["wdl"]
                        try:
                            wdl_obj = pov_wdl.pov(pov)
                        except Exception:
                            wdl_obj = getattr(pov_wdl, "wdl", pov_wdl)
                        p_win, p_draw, p_loss = (
                            wdl_obj.wins/1000.0,
                            wdl_obj.draws/1000.0,
                            wdl_obj.losses/1000.0,
                        )
                    except Exception:
                        p_win = p_draw = p_loss = None
                if p_win is None:
                    try:
                        if score.is_mate():
                            m = score.mate()
                            if m and m > 0:
                                p_win, p_draw, p_loss = 1.0, 0.0, 0.0
                            else:
                                p_win, p_draw, p_loss = 0.0, 0.0, 1.0
                        else:
                            model = chess.engine.CpWdlModel()
                            wdl_est = score.wdl(model=model)
                            p_win, p_draw, p_loss = (
                                wdl_est.wins/1000.0,
                                wdl_est.draws/1000.0,
                                wdl_est.losses/1000.0,
                            )
                    except Exception:
                        p_win, p_draw, p_loss = 0.0, 0.0, 0.0
                return cp_val, (p_win, p_draw, p_loss)

            root_cp, root_wdl = _analyse_board(board)

            if best_move is not None:
                b_best = board.copy()
                b_best.push(chess.Move.from_uci(best_move))
                best_cp, best_wdl = _analyse_board(b_best)
            else:
                best_cp, best_wdl = root_cp, root_wdl

            b_my = board.copy()
            b_my.push(move)
            my_cp, my_wdl = _analyse_board(b_my)

            cp_loss = None if (best_cp is None or my_cp is None) else (best_cp - my_cp)

            score_100, score_details = compute_move_score(
                best_cp=best_cp,
                best_wdl=best_wdl,
                my_cp=my_cp,
                my_wdl=my_wdl,
                weight_wdl=0.6,
                cp_scale=80.0,
                min_best_E=0.05,
            )

            return {
                "is_legal": True,
                "best_move": best_move,
                "root_cp": root_cp,
                "root_wdl": root_wdl,
                "best_cp": best_cp,
                "best_wdl": best_wdl,
                "my_move": move.uci(),
                "my_cp": my_cp,
                "my_wdl": my_wdl,
                "cp_loss": cp_loss,
                "score_100": score_100,
                "score_details": score_details,
            }
    except Exception:
        return None
