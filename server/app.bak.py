import io
import json
import os
import sys
import logging
import threading
import time
import uuid
from functools import lru_cache
from typing import Any, Optional, List, Tuple, Dict
from asyncio import to_thread

import chess
import chess.engine
import msgpack
import numpy as np
import plotly.graph_objects as go
import torch
from datasets import Dataset
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from torchvision import transforms

from lm_saes.backend import LanguageModel
from lm_saes.config import MongoDBConfig, SAEConfig
from lm_saes.database import MongoClient
from lm_saes.resource_loaders import load_dataset_shard, load_model
from lm_saes.sae import SparseAutoEncoder
from lm_saes.lc0_mapping import uci_to_idx_mappings, idx_to_uci_mappings, get_mapping_index

# 导入Transformer模型相关
try:
    from transformer_lens import HookedTransformer
    TRANSFORMER_LENS_AVAILABLE = True
    logging.info("✓ transformer_lens 模块加载成功")
except ImportError as e:
    logging.warning(f"⚠️ 无法加载 transformer_lens 模块: {e}")
    TRANSFORMER_LENS_AVAILABLE = False

# 全局模型实例
global_chess_model = None

# 添加规则模块路径
# 尝试多种路径方式
possible_paths = [
    os.path.join(os.path.dirname(__file__), '../exp/07rule'),
    os.path.join(os.getcwd(), 'exp/07rule'),
    os.path.join(os.getcwd(), '../exp/07rule'),
    '/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/exp/07rule'
]

rules_path = None
for path in possible_paths:
    if os.path.exists(path):
        rules_path = path
        break

if rules_path:
    # 使用绝对路径，避免相对路径问题
    rules_path = os.path.abspath(rules_path)
    sys.path.append(rules_path)
    print(f"🔍 使用规则模块路径: {rules_path}")
    print(f"🔍 当前工作目录: {os.getcwd()}")
    print(f"🔍 规则目录内容: {os.listdir(rules_path)}")
    print(f"🔍 rules子目录内容: {os.listdir(os.path.join(rules_path, 'rules'))}")
else:
    print(f"❌ 未找到规则模块目录")
    print(f"🔍 尝试的路径: {possible_paths}")

# 导入国际象棋规则分析函数
if rules_path:
    try:
        # 尝试直接导入
        from rules.my_rook_under_attack import is_rook_under_attack
        from rules.my_knight_under_attack import is_knight_under_attack  
        from rules.my_bishop_under_attack import is_bishop_under_attack
        from rules.my_queen_under_attack import is_queen_under_attack
        from rules.my_can_capture_rook import is_can_capture_rook
        from rules.my_can_capture_knight import is_can_capture_knight
        from rules.my_can_capture_bishop import is_can_capture_bishop
        from rules.my_can_capture_queen import is_can_capture_queen
        from rules.my_king_check import is_king_in_check, is_checkmate, is_stalemate
        from rules.my_bishop_pair import is_bishop_pair
        from rules.my_pin import get_pinned_pieces
        from rules.my_fork import is_in_fork
        from rules.my_pawn_structure import get_isolated_pawns, get_doubled_pawns, get_passed_pawns
        from rules.my_center_control import get_center_control
        from rules.my_threat_analysis import analyze_threats_by_lesser_pieces, get_threat_summary
        RULES_AVAILABLE = True
        logging.info("✓ 国际象棋规则模块加载成功")
    except ImportError as e:
        logging.warning(f"⚠️ 无法加载国际象棋规则模块: {e}")
        print(f"🔍 详细错误信息: {e}")
        print(f"🔍 Python路径: {sys.path}")
        RULES_AVAILABLE = False
else:
    logging.warning("⚠️ 规则模块路径未找到，跳过规则模块加载")
    RULES_AVAILABLE = False

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global global_engine, global_chess_model
    try:
        global_engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        logging.info(f"✓ Stockfish引擎成功加载: {ENGINE_PATH}")
        # Configure global engine options once
        for option, value in ENGINE_OPTIONS.items():
            global_engine.configure({option: value})
    except Exception as e:
        logging.error(f"❌ Stockfish引擎加载失败: {e}")
        global_engine = None  # Ensure it's None if failed
    
    # 加载象棋模型
    if TRANSFORMER_LENS_AVAILABLE:
        try:
            global_chess_model = HookedTransformer.from_pretrained_no_processing(
                'lc0/T82-768x15x24h',
                dtype=torch.float32,
            ).eval()
            if torch.cuda.is_available():
                global_chess_model = global_chess_model.cuda()
            logging.info("✓ 象棋模型成功加载: lc0/T82-768x15x24h")
        except Exception as e:
            logging.error(f"❌ 象棋模型加载失败: {e}")
            global_chess_model = None
    else:
        logging.warning("⚠️ transformer_lens 不可用，跳过象棋模型加载")

@app.on_event("shutdown")
async def shutdown_event():
    global global_engine, global_chess_model
    if global_engine:
        global_engine.quit()
        logging.info("✓ Stockfish引擎已关闭")
    
    if global_chess_model:
        del global_chess_model
        global_chess_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("✓ 象棋模型已清理")

app.add_middleware(GZipMiddleware, minimum_size=1000)

client = MongoClient(MongoDBConfig())
sae_series = os.environ.get("SAE_SERIES", "default")

# Stockfish引擎配置
ENGINE_PATH = "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/Stockfish/src/stockfish"

# 验证Stockfish引擎是否可用
try:
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        logging.info(f"✓ Stockfish引擎成功加载: {ENGINE_PATH}")
        engine_info = engine.id
        logging.info(f"  引擎名称: {engine_info.get('name', 'unknown')}")
        logging.info(f"  引擎作者: {engine_info.get('author', 'unknown')}")
except Exception as e:
    logging.error(f"❌ Stockfish引擎加载失败: {e}")
    logging.error(f"  请确认路径是否正确: {ENGINE_PATH}")
    # 不抛出异常，让服务器继续运行，只是引擎分析功能不可用

# 优化的引擎配置 - 更快的分析速度
ENGINE_TIME_LIMIT = 0.3  # 0.3秒快速分析
ENGINE_OPTIONS = {
    "Threads": 2,  # 减少线程数
    "Hash": 256,   # 减少哈希表大小
    "Skill Level": 15,  # 稍微降低技能等级以提高速度
}

# 会话管理：跟踪活跃的分析会话
active_sessions = {}
session_lock = threading.Lock()

# 全局引擎实例
global_engine: Optional[chess.engine.SimpleEngine] = None

def register_session(session_id: str, cancel_event: threading.Event):
    """注册一个新的分析会话"""
    with session_lock:
        active_sessions[session_id] = {
            'cancel_event': cancel_event,
            'created_at': time.time()
        }
        logging.info(f"注册新会话: {session_id}")

def cancel_session(session_id: str) -> bool:
    """取消指定的分析会话"""
    with session_lock:
        if session_id in active_sessions:
            active_sessions[session_id]['cancel_event'].set()
            del active_sessions[session_id]
            logging.info(f"取消会话: {session_id}")
            return True
        else:
            logging.warning(f"会话不存在或已结束: {session_id}")
            return False

def cleanup_session(session_id: str):
    """清理完成的会话"""
    with session_lock:
        if session_id in active_sessions:
            del active_sessions[session_id]
            logging.info(f"清理会话: {session_id}")

def is_session_cancelled(session_id: str) -> bool:
    """检查会话是否被取消"""
    if not session_id:
        return False  # 如果没有session_id，不进行取消检查
        
    with session_lock:
        if session_id in active_sessions:
            return active_sessions[session_id]['cancel_event'].is_set()
        return False  # 如果会话不存在，认为未被取消（允许继续执行）

def analyze_chess_rules(fen: str) -> dict:
    """
    分析FEN局面的所有国际象棋规则
    
    Args:
        fen: 棋局的FEN字符串
        
    Returns:
        dict: 包含所有规则分析结果的字典
    """
    if not RULES_AVAILABLE:
        return {"error": "规则模块未加载"}
    
    try:
        rules_result = {}
        
        # 基本检查 - 己方棋子被抓
        rules_result["is_rook_under_attack"] = is_rook_under_attack(fen)
        rules_result["is_knight_under_attack"] = is_knight_under_attack(fen)
        rules_result["is_bishop_under_attack"] = is_bishop_under_attack(fen)
        rules_result["is_queen_under_attack"] = is_queen_under_attack(fen)
        
        # 可以攻击对方棋子
        rules_result["is_can_capture_rook"] = is_can_capture_rook(fen)
        rules_result["is_can_capture_knight"] = is_can_capture_knight(fen)
        rules_result["is_can_capture_bishop"] = is_can_capture_bishop(fen)
        rules_result["is_can_capture_queen"] = is_can_capture_queen(fen)
        
        # 王的状态
        rules_result["is_king_in_check"] = is_king_in_check(fen)
        rules_result["is_checkmate"] = is_checkmate(fen)
        rules_result["is_stalemate"] = is_stalemate(fen)
        
        # 棋子配置
        rules_result["is_bishop_pair"] = is_bishop_pair(fen)
        
        # 战术分析
        pinned_pieces = get_pinned_pieces(fen)
        rules_result["has_pinned_pieces"] = len(pinned_pieces) > 0
        rules_result["pinned_pieces"] = pinned_pieces
        
        rules_result["is_in_fork"] = is_in_fork(fen)
        
        # 兵结构分析 - 分别分析白方和黑方
        isolated_pawns_white = get_isolated_pawns(fen, 'white')
        isolated_pawns_black = get_isolated_pawns(fen, 'black')
        rules_result["isolated_pawns_white"] = isolated_pawns_white
        rules_result["isolated_pawns_black"] = isolated_pawns_black
        rules_result["has_isolated_pawns"] = len(isolated_pawns_white) > 0 or len(isolated_pawns_black) > 0
        
        doubled_pawns_white = get_doubled_pawns(fen, 'white')
        doubled_pawns_black = get_doubled_pawns(fen, 'black')
        rules_result["doubled_pawns_white"] = doubled_pawns_white
        rules_result["doubled_pawns_black"] = doubled_pawns_black
        rules_result["has_doubled_pawns"] = len(doubled_pawns_white) > 0 or len(doubled_pawns_black) > 0
        
        passed_pawns_white = get_passed_pawns(fen, 'white')
        passed_pawns_black = get_passed_pawns(fen, 'black')
        rules_result["passed_pawns_white"] = passed_pawns_white
        rules_result["passed_pawns_black"] = passed_pawns_black
        rules_result["has_passed_pawns"] = len(passed_pawns_white) > 0 or len(passed_pawns_black) > 0
        
        # 中心控制分析
        center_control = get_center_control(fen)
        rules_result["center_control"] = center_control
        
        # 威胁分析 - 棋子受到较小棋子的威胁
        threat_analysis = analyze_threats_by_lesser_pieces(fen)
        threat_summary = get_threat_summary(fen)
        rules_result["threat_analysis"] = threat_analysis
        rules_result["threat_summary"] = threat_summary
        
        logging.info(f"🧩 规则分析完成: {len(rules_result)} 项规则")
        return rules_result
        
    except Exception as e:
        logging.error(f"规则分析失败: {e}")
        return {"error": f"规则分析失败: {str(e)}"}


def calculate_material_balance(fen: str) -> dict:
    """
    计算棋盘上双方的物质力量
    
    Args:
        fen: 棋局的FEN字符串
        
    Returns:
        dict: 包含双方物质力量分析的字典
    """
    try:
        board = chess.Board(fen)
        
        # 标准棋子价值
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # 王不计入物质价值
        }
        
        # 初始化计数器
        white_material = 0
        black_material = 0
        white_pieces = {piece_type: 0 for piece_type in piece_values.keys()}
        black_pieces = {piece_type: 0 for piece_type in piece_values.keys()}
        
        # 遍历棋盘统计棋子
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_type = piece.piece_type
                piece_value = piece_values[piece_type]
                
                if piece.color == chess.WHITE:
                    white_material += piece_value
                    white_pieces[piece_type] += 1
                else:
                    black_material += piece_value
                    black_pieces[piece_type] += 1
        
        # 计算物质差距
        material_difference = white_material - black_material
        
        # 物质力量分析结果
        material_analysis = {
            "white_material": white_material,
            "black_material": black_material,
            "material_difference": material_difference,  # 正数表示白方优势，负数表示黑方优势
            "material_advantage": "white" if material_difference > 0 else "black" if material_difference < 0 else "equal",
            "white_pieces": {
                "pawns": white_pieces[chess.PAWN],
                "knights": white_pieces[chess.KNIGHT],
                "bishops": white_pieces[chess.BISHOP],
                "rooks": white_pieces[chess.ROOK],
                "queens": white_pieces[chess.QUEEN],
                "king": white_pieces[chess.KING]
            },
            "black_pieces": {
                "pawns": black_pieces[chess.PAWN],
                "knights": black_pieces[chess.KNIGHT],
                "bishops": black_pieces[chess.BISHOP],
                "rooks": black_pieces[chess.ROOK],
                "queens": black_pieces[chess.QUEEN],
                "king": black_pieces[chess.KING]
            }
        }
        
        logging.info(f"💰 物质分析完成: 白方{white_material} vs 黑方{black_material} (差距: {material_difference})")
        return material_analysis
        
    except Exception as e:
        logging.error(f"物质分析失败: {e}")
        return {"error": f"物质分析失败: {str(e)}"}


async def get_wdl_evaluation(fen: str) -> dict:
    """
    使用Stockfish引擎获取WDL (Win/Draw/Loss) 评估
    
    Args:
        fen: 棋局的FEN字符串
        
    Returns:
        dict: 包含WDL评估的字典
    """
    try:
        board = chess.Board(fen)
        
        def run_wdl_analysis():
            try:
                with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
                    # 配置引擎参数，确保启用WDL评估
                    engine_options = ENGINE_OPTIONS.copy()
                    # 一些Stockfish版本需要特殊设置来启用WDL
                    try:
                        engine.configure(engine_options)
                    except:
                        pass  # 如果配置失败，继续使用默认设置
                    
                    # 进行更深入的分析以获得准确的WDL
                    info = engine.analyse(
                        board,
                        chess.engine.Limit(
                            time=ENGINE_TIME_LIMIT * 2,  # 稍长时间以获得更准确的WDL
                            depth=12,
                            nodes=1_000_000
                        ),
                        info=chess.engine.INFO_ALL
                    )
                    
                    return info
            except Exception as e:
                logging.error(f"WDL分析内部错误: {e}")
                return None
        
        info = await to_thread(run_wdl_analysis)
        
        if info:
            wdl_analysis = {}
            
            # 获取评估分数
            if "score" in info:
                score = info["score"]
                
                try:
                    # 相对于当前行棋方的分数
                    if score.is_mate():
                        # 如果是将死局面
                        # 正确的方式是检查relative属性
                        if hasattr(score.relative, 'mate') and score.relative.mate() is not None:
                            mate_in = score.relative.mate()
                        elif hasattr(score, 'mate') and score.mate() is not None:
                            mate_in = score.mate()
                        else:
                            # 备用方案：从字符串解析
                            try:
                                score_str = str(score)
                                if '#' in score_str:
                                    mate_in = int(score_str.split('#')[1])
                                else:
                                    mate_in = 1 if 'mate' in score_str.lower() else None
                            except:
                                mate_in = None
                        
                        wdl_analysis["evaluation_type"] = "mate"
                        wdl_analysis["mate_in"] = mate_in
                        wdl_analysis["evaluation_cp"] = None
                        
                        # 对于将死局面，WDL是确定的
                        if mate_in is not None and mate_in > 0:  # 当前行棋方可以将死对方
                            wdl_analysis["win_probability"] = 1.0
                            wdl_analysis["draw_probability"] = 0.0
                            wdl_analysis["loss_probability"] = 0.0
                        else:  # 当前行棋方将被将死
                            wdl_analysis["win_probability"] = 0.0
                            wdl_analysis["draw_probability"] = 0.0
                            wdl_analysis["loss_probability"] = 1.0
                    else:
                        # 普通局面评估
                        try:
                            # 尝试不同的方式获取分数
                            if hasattr(score.relative, 'score'):
                                cp_score = score.relative.score(mate_score=10000)
                            elif hasattr(score, 'score'):
                                cp_score = score.score(mate_score=10000)
                            else:
                                cp_score = None
                        except Exception as e:
                            logging.warning(f"获取分数失败: {e}")
                            cp_score = None
                        
                        wdl_analysis["evaluation_type"] = "centipawn"
                        wdl_analysis["evaluation_cp"] = cp_score
                        wdl_analysis["mate_in"] = None
                        
                        # 将分数转换为胜率（使用logistic函数）
                        # 这是一个常用的转换公式，400分约等于50%的胜率差
                        if cp_score is not None:
                            # 转换为胜率（相对于当前行棋方）
                            win_prob = 1 / (1 + 10 ** (-cp_score / 400))
                            
                            # 简化的WDL计算（实际的WDL需要更复杂的模型）
                            # 这里使用一个启发式方法
                            if abs(cp_score) < 50:  # 接近平局的局面
                                draw_prob = 0.4 - abs(cp_score) / 500  # 最高40%平局概率
                                draw_prob = max(0.1, min(0.4, draw_prob))
                            else:
                                draw_prob = 0.1  # 基础平局概率
                            
                            # 调整胜负概率
                            remaining_prob = 1.0 - draw_prob
                            win_prob_adjusted = win_prob * remaining_prob
                            loss_prob_adjusted = (1 - win_prob) * remaining_prob
                            
                            wdl_analysis["win_probability"] = round(win_prob_adjusted, 3)
                            wdl_analysis["draw_probability"] = round(draw_prob, 3)
                            wdl_analysis["loss_probability"] = round(loss_prob_adjusted, 3)
                        else:
                            # 无法评估的情况
                            wdl_analysis["win_probability"] = 0.33
                            wdl_analysis["draw_probability"] = 0.34
                            wdl_analysis["loss_probability"] = 0.33
                            
                except Exception as e:
                    logging.error(f"分析分数对象时出错: {e}")
                    # 回退到简单评估
                    wdl_analysis["evaluation_type"] = "unknown"
                    wdl_analysis["evaluation_cp"] = None
                    wdl_analysis["mate_in"] = None
                    wdl_analysis["win_probability"] = 0.33
                    wdl_analysis["draw_probability"] = 0.34
                    wdl_analysis["loss_probability"] = 0.33
            
            # 获取WDL信息（如果引擎支持）
            if "wdl" in info:
                wdl = info["wdl"]
                total = sum(wdl)
                if total > 0:
                    wdl_analysis["engine_win_probability"] = round(wdl[0] / total, 3)
                    wdl_analysis["engine_draw_probability"] = round(wdl[1] / total, 3)
                    wdl_analysis["engine_loss_probability"] = round(wdl[2] / total, 3)
            
            # 添加当前行棋方信息
            wdl_analysis["active_color"] = "white" if board.turn == chess.WHITE else "black"
            
            # 获取主要变着
            if "pv" in info and info["pv"]:
                pv_moves = []
                temp_board = board.copy()
                for move in info["pv"][:5]:  # 取前5步主要变着
                    pv_moves.append(temp_board.san(move))
                    temp_board.push(move)
                wdl_analysis["principal_variation"] = pv_moves
            
            logging.info(f"📊 WDL分析完成: W{wdl_analysis.get('win_probability', '?')} D{wdl_analysis.get('draw_probability', '?')} L{wdl_analysis.get('loss_probability', '?')}")
            return wdl_analysis
        else:
            logging.warning("WDL分析失败: 无法获取引擎信息")
            return {"error": "WDL分析失败: 无法获取引擎评估"}
            
    except Exception as e:
        logging.error(f"WDL分析异常: {e}")
        return {"error": f"WDL分析异常: {str(e)}"}


# 使用LC0映射表
logging.info(f"✓ LC0映射表加载完成: 白方无易位 {len(uci_to_idx_mappings[0])}, 白方有易位 {len(uci_to_idx_mappings[1])}, 黑方无易位 {len(uci_to_idx_mappings[2])}, 黑方有易位 {len(uci_to_idx_mappings[3])}")

def find_best_moves_from_legal_moves(policy_logits, legal_moves, board):
    """从合法移动中直接查找最佳移动的备用方法"""
    try:
        best_moves = []
        policy_probs = torch.softmax(policy_logits, dim=0)
        
        # 确定使用哪个映射表
        mapping_idx = get_mapping_index(board)
        uci_to_idx = uci_to_idx_mappings[mapping_idx]
        
        # 为每个合法移动查找对应的索引和分数
        move_scores = []
        for legal_move in legal_moves:
            uci = legal_move.uci()
            if uci in uci_to_idx:
                idx = uci_to_idx[uci]
                if idx < len(policy_logits):
                    score = policy_logits[idx].item()
                    prob = policy_probs[idx].item()
                    move_scores.append({
                        "move": legal_move,
                        "uci": uci,
                        "san": board.san(legal_move),
                        "score": score,
                        "probability": prob * 100,
                        "index": idx
                    })
        
        # 按分数排序
        move_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # 返回前5个
        return move_scores[:5]
        
    except Exception as e:
        logging.error(f"备用移动查找失败: {e}")
        return []

async def analyze_position_with_model(fen: str) -> dict:
    """
    使用自己的模型分析FEN局面
    
    Args:
        fen: 棋局的FEN字符串
        
    Returns:
        dict: 包含模型分析结果的字典
    """
    global global_chess_model
    
    if not global_chess_model or not TRANSFORMER_LENS_AVAILABLE:
        return {"error": "模型未加载或不可用"}
    
    try:
        def run_model_inference():
            try:
                # 使用模型进行推理
                with torch.no_grad():
                    outputs, cache = global_chess_model.run_with_cache(fen, prepend_bos=False)
                
                # 解析输出
                policy_logits = outputs[0][0] if len(outputs) > 0 else None  # 策略logits
                wdl_probs = outputs[1][0] if len(outputs) > 1 else None      # WDL概率
                value_score = outputs[2][0] if len(outputs) > 2 else None    # 价值评估
                
                result = {
                    "raw_outputs": {
                        "policy_logits_shape": list(policy_logits.shape) if policy_logits is not None else None,
                        "wdl_probs_shape": list(wdl_probs.shape) if wdl_probs is not None else None,
                        "value_shape": list(value_score.shape) if value_score is not None else None,
                    }
                }
                
                # 解析策略输出得到最佳移动
                if policy_logits is not None:
                    try:
                        board = chess.Board(fen)
                        legal_moves_list = list(board.legal_moves)
                        
                        logging.info(f"开始策略分析: FEN={fen[:20]}..., 合法移动数={len(legal_moves_list)}, 策略输出维度={policy_logits.shape}")
                        
                        # 计算softmax概率
                        policy_probs = torch.softmax(policy_logits, dim=0)
                        
                        # 确定使用哪个映射表
                        mapping_idx = get_mapping_index(board)
                        idx_to_uci = idx_to_uci_mappings[mapping_idx]
                        
                        logging.info(f"使用映射表 {mapping_idx}: {'白方' if mapping_idx < 2 else '黑方'}, {'有王车易位权' if mapping_idx % 2 == 1 else '无王车易位权'}")
                        
                        # 方法1：通过索引排序查找
                        sorted_indices = torch.argsort(policy_logits, descending=True)
                        best_moves = []
                        checked_moves = 0
                        max_check = min(500, len(sorted_indices))  # 减少检查数量
                        
                        for idx in sorted_indices[:max_check]:
                            try:
                                move_idx = idx.item()
                                if move_idx in idx_to_uci:
                                    uci_move = idx_to_uci[move_idx]
                                    
                                    try:
                                        move = chess.Move.from_uci(uci_move)
                                        
                                        # 检查移动是否合法
                                        if move in legal_moves_list:
                                            move_score = policy_logits[idx].item()
                                            move_prob = policy_probs[idx].item()
                                            
                                            best_moves.append({
                                                "uci": uci_move,
                                                "san": board.san(move),
                                                "score": round(move_score, 4),
                                                "probability": round(move_prob * 100, 2),
                                                "move_index": move_idx
                                            })
                                            
                                            if len(best_moves) >= 5:
                                                break
                                    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                                        continue
                                
                                checked_moves += 1
                            except Exception:
                                continue
                        
                        # 方法2：如果方法1失败，使用备用方案
                        if not best_moves:
                            logging.warning(f"主要方法未找到合法移动，使用备用方法...")
                            backup_moves = find_best_moves_from_legal_moves(policy_logits, legal_moves_list, board)
                            if backup_moves:
                                best_moves = backup_moves
                                logging.info(f"备用方法找到 {len(backup_moves)} 个移动")
                        
                        result["policy_analysis"] = {
                            "best_moves": best_moves,
                            "total_legal_moves": len(legal_moves_list),
                            "checked_indices": checked_moves,
                            "policy_shape": list(policy_logits.shape),
                            "mapping_size": len(idx_to_uci),
                            "mapping_index": mapping_idx,
                            "method_used": "primary" if checked_moves > 0 else "backup"
                        }
                        
                        if best_moves:
                            result["best_move"] = best_moves[0]["uci"]
                            result["best_move_san"] = best_moves[0]["san"]
                            result["best_move_probability"] = best_moves[0]["probability"]
                            logging.info(f"找到最佳移动: {best_moves[0]['san']} ({best_moves[0]['probability']:.1f}%)")
                        else:
                            # 最后的备用方案：随机选择一个合法移动
                            if legal_moves_list:
                                random_move = legal_moves_list[0]  # 选择第一个合法移动
                                result["best_move"] = random_move.uci()
                                result["best_move_san"] = board.san(random_move)
                                result["best_move_probability"] = 0.0
                                result["policy_analysis"]["fallback_move"] = True
                                logging.warning(f"使用备用移动: {board.san(random_move)}")
                            else:
                                logging.error("没有找到任何合法移动")
                        
                    except Exception as e:
                        logging.error(f"策略分析失败: {e}")
                        result["policy_error"] = str(e)
                
                # 解析WDL概率
                if wdl_probs is not None:
                    try:
                        # 假设WDL顺序是 [win, draw, loss]
                        win_prob = wdl_probs[0].item() if len(wdl_probs) > 0 else 0
                        draw_prob = wdl_probs[1].item() if len(wdl_probs) > 1 else 0
                        loss_prob = wdl_probs[2].item() if len(wdl_probs) > 2 else 0
                        
                        result["wdl_analysis"] = {
                            "win_probability": round(win_prob, 4),
                            "draw_probability": round(draw_prob, 4),
                            "loss_probability": round(loss_prob, 4),
                            "win_percent": round(win_prob * 100, 1),
                            "draw_percent": round(draw_prob * 100, 1),
                            "loss_percent": round(loss_prob * 100, 1)
                        }
                    except Exception as e:
                        logging.error(f"WDL分析失败: {e}")
                        result["wdl_error"] = str(e)
                
                # 解析价值评估
                if value_score is not None:
                    try:
                        value = value_score.item() if hasattr(value_score, 'item') else float(value_score)
                        result["value_analysis"] = {
                            "raw_value": round(value, 4),
                            "normalized_value": round(value / 100, 4)  # 假设需要归一化
                        }
                    except Exception as e:
                        logging.error(f"价值分析失败: {e}")
                        result["value_error"] = str(e)
                
                return result
                
            except Exception as e:
                logging.error(f"模型推理内部错误: {e}")
                return {"error": f"模型推理失败: {str(e)}"}
        
        # 在线程中运行推理以避免阻塞
        result = await to_thread(run_model_inference)
        
        logging.info(f"🤖 模型分析完成 for {fen[:20]}...")
        return result
        
    except Exception as e:
        logging.error(f"模型分析异常: {e}")
        return {"error": f"模型分析异常: {str(e)}"}


async def analyze_position_with_stockfish_simple(fen: str) -> Optional[tuple[str, Optional[str]]]:
    """
    简化的Stockfish分析函数，无会话管理
    
    Args:
        fen: 棋局的FEN字符串
        
    Returns:
        tuple[str, Optional[str]]: (最佳走法, ponder走法)；如果分析失败则返回None
    """
    logging.info(f"⚡ 开始分析FEN: {fen[:30]}...")

    if not fen or len(fen.strip()) == 0:
        logging.error("收到空的FEN字符串")
        return None

    try:
        board = chess.Board(fen)
        if board.is_game_over():
            logging.info(f"游戏已结束: {fen}")
            return None

        logging.info(f"🔍 开始Stockfish计算: {fen[:30]}...")

        # 使用新的引擎实例避免状态冲突
        def run_engine_analysis():
            try:
                with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
                    # 配置引擎参数
                    for option, value in ENGINE_OPTIONS.items():
                        engine.configure({option: value})
                    
                    # 分析局面
                    result = engine.play(
                        board,
                        chess.engine.Limit(
                            time=ENGINE_TIME_LIMIT,
                            depth=10,         # 降低深度以提高速度
                            nodes=500_000     # 降低节点数
                        )
                    )
                    return result
            except Exception as e:
                logging.error(f"引擎分析内部错误: {e}")
                return None

        result = await to_thread(run_engine_analysis)

        if result and result.move:
            best_move = str(result.move)
            ponder = str(result.ponder) if result.ponder else None
            logging.info(f"🎯 分析成功: {best_move} [FEN:{fen[:20]}...]")
            return best_move, ponder
        else:
            logging.warning(f"引擎未返回着法: {fen}")
            return None

    except ValueError as e:
        logging.error(f"无效的FEN: {fen} - {str(e)}")
        return None
    except Exception as e:
        logging.error(f"分析异常 {fen}: {e}")
        return None

# Remove global caches in favor of LRU cache
# sae_cache: dict[str, SparseAutoEncoder] = {}
# lm_cache: dict[str, LanguageModel] = {}
# dataset_cache: dict[tuple[str, int, int], Dataset] = {}


# def fen_to_longfen(fen: str, move: str) -> str:
#     '''
#     input: fen,move
#     output: longfen(wrnbqkbnrpppppppp................................PPPPPPPPRNBQKBNRKQkq..0..1..e2e40)
#     '''
#     parts = fen.split()
#     board_fen = parts[0]  # 棋盘部分
#     active_color = parts[1]  # 当前走棋方 (w/b)
#     castling = parts[2]  # 王车易位权利
#     en_passant = parts[3]  # 过路兵目标格
#     halfmove = parts[4]  # 半回合计数
#     fullmove = parts[5]  # 全回合计数
    
#     # 转换棋盘部分 (8x8 = 64个字符)
#     longfen_board = ""
#     for char in board_fen:
#         if char == '/':
#             continue  # 跳过行分隔符
#         elif char.isdigit():
#             # 数字表示连续的空格数
#             longfen_board += '.' * int(char)
#         else:
#             # 棋子字符直接添加
#             longfen_board += char
    
#     # 确保棋盘部分正好64个字符
#     assert len(longfen_board) == 64, f"棋盘应该有64个字符，实际有{len(longfen_board)}个"
    
#     # 处理王车易位权利 (4个字符位置：KQkq)
#     castling_longfen = ""
#     for right in ['K', 'Q', 'k', 'q']:
#         if right in castling:
#             castling_longfen += right
#         else:
#             castling_longfen += '.'
    
#     # 处理过路兵 (2个字符)
#     if en_passant == '-':
#         en_passant_longfen = ".."
#     else:
#         en_passant_longfen = en_passant
    
#     # 处理半回合和全回合计数 (各1个字符)
#     halfmove_padded = halfmove.ljust(3, '.')  # 左对齐，右侧填充.
#     fullmove_padded = fullmove.ljust(3, '.')  # 左对齐，右侧填充.
    
#     # 组装longfen字符串
#     longfen = (
#         active_color +  # 当前走棋方 (1字符)
#         longfen_board +  # 棋盘 (64字符)
#         castling_longfen +  # 王车易位 (4字符)
#         en_passant_longfen +  # 过路兵 (2字符)
#         halfmove_padded +  # 半回合 (3字符)
#         fullmove_padded +  # 全回合 (3字符)
#         move +  # 走法
#         "0"  # 结束标记
#     )
    
#     return longfen

# def fen_to_longfen_behavioral_cloning(fen: str) -> str:
#     '''
#     input: fen
#     output: longfen(wrnbqkbnrpppppppp................................PPPPPPPPRNBQKBNRKQkq..0..1..0)
#     '''
#     parts = fen.split()
#     board_fen = parts[0]  # 棋盘部分
#     active_color = parts[1]  # 当前走棋方 (w/b)
#     castling = parts[2]  # 王车易位权利
#     en_passant = parts[3]  # 过路兵目标格
#     halfmove = parts[4]  # 半回合计数
#     fullmove = parts[5]  # 全回合计数
    
#     # 转换棋盘部分 (8x8 = 64个字符)
#     longfen_board = ""
#     for char in board_fen:
#         if char == '/':
#             continue  # 跳过行分隔符
#         elif char.isdigit():
#             # 数字表示连续的空格数
#             longfen_board += '.' * int(char)
#         else:
#             # 棋子字符直接添加
#             longfen_board += char
    
#     # 确保棋盘部分正好64个字符
#     assert len(longfen_board) == 64, f"棋盘应该有64个字符，实际有{len(longfen_board)}个"
    
#     # 处理王车易位权利 (4个字符位置：KQkq)
#     castling_longfen = ""
#     for right in ['K', 'Q', 'k', 'q']:
#         if right in castling:
#             castling_longfen += right
#         else:
#             castling_longfen += '.'
    
#     # 处理过路兵 (2个字符)
#     if en_passant == '-':
#         en_passant_longfen = ".."
#     else:
#         en_passant_longfen = en_passant
    
#     # 处理半回合和全回合计数 (各3个字符)
#     # 使用左对齐，右侧填充.
#     halfmove_padded = halfmove.ljust(3, '.')  # 左对齐，右侧填充.
#     fullmove_padded = fullmove.ljust(3, '.')  # 左对齐，右侧填充.
    
#     # 组装longfen字符串
#     longfen_behavioral_cloning = (
#         active_color +  # 当前走棋方 (1字符)
#         longfen_board +  # 棋盘 (64字符)
#         castling_longfen +  # 王车易位 (4字符)
#         en_passant_longfen +  # 过路兵 (2字符)
#         halfmove_padded +  # 半回合 (3字符)
#         fullmove_padded +  # 全回合 (3字符)
#         # move +  # 走法
#         "0"  # 结束标记
#     )
    
#     return longfen_behavioral_cloning



def fen_to_board_str(fen: str) -> str:
    '''
    input: fen
    output: board_str(rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR)
    '''
    parts = fen.split()
    board_fen = parts[0]  # 棋盘部分    
    
    board_str = ""
    for char in board_fen:
        if char == '/':
            continue  # 跳过行分隔符
        elif char.isdigit():
            # 数字表示连续的空格数
            board_str += '.' * int(char)
        else:
            # 棋子字符直接添加
            board_str += char   
    return board_str

@lru_cache(maxsize=8)
def get_model(name: str) -> LanguageModel:
    """Load and cache a language model.

    Args:
        name: Name of the model to load

    Returns:
        LanguageModel: The loaded model

    Raises:
        ValueError: If the model is not found
    """
    cfg = client.get_model_cfg(name)
    if cfg is None:
        raise ValueError(f"Model {name} not found")
    return load_model(cfg)


@lru_cache(maxsize=16)
def get_dataset(name: str, shard_idx: int = 0, n_shards: int = 1) -> Dataset:
    """Load and cache a dataset shard.

    Args:
        name: Name of the dataset
        shard_idx: Index of the shard to load
        n_shards: Total number of shards

    Returns:
        Dataset: The loaded dataset shard

    Raises:
        AssertionError: If the dataset is not found
    """
    cfg = client.get_dataset_cfg(name)
    assert cfg is not None, f"Dataset {name} not found"
    return load_dataset_shard(cfg, shard_idx, n_shards)


@lru_cache(maxsize=8)
def get_sae(name: str) -> SparseAutoEncoder:
    """Load and cache a sparse autoencoder.

    Args:
        name: Name of the SAE to load

    Returns:
        SparseAutoEncoder: The loaded SAE

    Raises:
        AssertionError: If the SAE is not found
    """
    path = client.get_sae_path(name, sae_series)
    assert path is not None, f"SAE {name} not found"
    cfg = SAEConfig.from_pretrained(path)
    sae = SparseAutoEncoder.from_config(cfg)
    sae.eval()
    return sae


def make_serializable(obj):
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def trim_minimum(*arrs: list[Any] | None) -> list[list[Any] | None]:
    """Trim multiple arrays to the length of the shortest non-None array.

    Args:
        *arrs: Arrays to trim

    Returns:
        list: List of trimmed arrays

    Example:
        >>> a = [1, 2, 3, 4]
        >>> b = [5, 6, 7]
        >>> c = [8, 9, 10, 11, 12]
        >>> trim_minimum(a, b, c)
        [[1, 2, 3], [5, 6, 7], [8, 9, 10]]
    """
    min_length = min(len(arr) for arr in arrs if arr is not None)
    return [arr[:min_length] if arr is not None else None for arr in arrs]


@app.exception_handler(AssertionError)
async def assertion_error_handler(request, exc):
    return Response(content=str(exc), status_code=400)


@app.exception_handler(torch.cuda.OutOfMemoryError)
async def oom_error_handler(request, exc):
    print("CUDA Out of memory. Clearing cache.")
    # Clear LRU caches
    get_model.cache_clear()
    get_dataset.cache_clear()
    get_sae.cache_clear()
    return Response(content="CUDA Out of memory", status_code=500)


@app.get("/dictionaries")
def list_dictionaries():
    return client.list_saes(sae_series=sae_series, has_analyses=True)


@app.get("/images/{dataset_name}")
def get_image(dataset_name: str, context_idx: int, image_idx: int, shard_idx: int = 0, n_shards: int = 1):
    dataset = get_dataset(dataset_name, shard_idx, n_shards)
    data = dataset[int(context_idx)]

    image_key = next((key for key in ["image", "images"] if key in data), None)
    if image_key is None:
        return Response(content="Image not found", status_code=404)

    if len(data[image_key]) <= image_idx:
        return Response(content="Image not found", status_code=404)

    image_tensor = data[image_key][image_idx]

    # Convert tensor to PIL Image and then to bytes
    image = transforms.ToPILImage()(image_tensor.to(torch.uint8))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")

    return Response(content=img_byte_arr.getvalue(), media_type="image/png")


@app.get("/dictionaries/{name}/metrics")
def get_available_metrics(name: str):
    """Get available metrics for a dictionary.

    Args:
        name: Name of the dictionary/SAE

    Returns:
        List of available metric names
    """
    metrics = client.get_available_metrics(name, sae_series=sae_series)
    return {"metrics": metrics}


@app.get("/dictionaries/{name}/features/count")
def count_features_with_filters(
    name: str,
    feature_analysis_name: str | None = None,
    metric_filters: str | None = None,
):
    """Count features that match the given filters.

    Args:
        name: Name of the dictionary/SAE
        feature_analysis_name: Optional analysis name
        metric_filters: Optional JSON string of metric filters

    Returns:
        Count of features matching the filters
    """
    # Parse metric filters if provided
    parsed_metric_filters = None
    if metric_filters:
        try:
            parsed_metric_filters = json.loads(metric_filters)
        except (json.JSONDecodeError, TypeError):
            return Response(
                content=f"Invalid metric_filters format: {metric_filters}",
                status_code=400,
            )

    count = client.count_features_with_filters(
        sae_name=name, sae_series=sae_series, name=feature_analysis_name, metric_filters=parsed_metric_filters
    )

    return {"count": count}


@app.get("/dictionaries/{name}/features/{feature_index}")
def get_feature(
    name: str,
    feature_index: str | int,
    feature_analysis_name: str | None = None,
    metric_filters: str | None = None,
):
    # Parse feature_index if it's a string
    if isinstance(feature_index, str) and feature_index != "random":
        try:
            feature_index = int(feature_index)
        except ValueError:
            return Response(
                content=f"Feature index {feature_index} is not a valid integer",
                status_code=400,
            )

    # Parse metric filters if provided
    parsed_metric_filters = None
    if metric_filters:
        try:
            parsed_metric_filters = json.loads(metric_filters)
        except (json.JSONDecodeError, TypeError):
            return Response(
                content=f"Invalid metric_filters format: {metric_filters}",
                status_code=400,
            )

    # Get feature data
    feature = (
        client.get_random_alive_feature(
            sae_name=name, sae_series=sae_series, name=feature_analysis_name, metric_filters=parsed_metric_filters
        )
        if feature_index == "random"
        else client.get_feature(sae_name=name, sae_series=sae_series, index=feature_index)
    )

    if feature is None:
        return Response(
            content=f"Feature {feature_index} not found in SAE {name}",
            status_code=404,
        )

    analysis = next(
        (a for a in feature.analyses if a.name == feature_analysis_name or feature_analysis_name is None), None
    )
    if analysis is None:
        return Response(
            content=f"Feature analysis {feature_analysis_name} not found in SAE {name}"
            if feature_analysis_name is not None
            else f"No feature analysis found in SAE {name}",
            status_code=404,
        )

    def process_sample(*, feature_acts, context_idx, dataset_name, model_name, shard_idx=None, n_shards=None):
        """Process a sample to extract and format feature data.

        Args:
            feature_acts: Feature activations
            decoder_norms: Decoder norms
            context_idx: Context index in the dataset
            dataset_name: Name of the dataset
            model_name: Name of the model
            shard_idx: Index of the dataset shard, defaults to 0
            n_shards: Total number of shards, defaults to 1

        Returns:
            dict: Processed sample data
        """  # Get model and dataset
        
        # TODO 
        # 打印一下data
    
        # print(f"{dataset_name=}")
        # print(f"{model_name = }")
        model = get_model(model_name)
        data = get_dataset(dataset_name, shard_idx, n_shards)[context_idx]

        # Get origins for the features  
        origins = model.trace({k: [v] for k, v in data.items()})[0]

        # Process image data if present
        # TODO: check here 
        image_key = next((key for key in ["image", "images"] if key in data), None)
        if image_key is not None:
            image_urls = [
                f"/images/{dataset_name}?context_idx={context_idx}&shard_idx={shard_idx}&n_shards={n_shards}&image_idx={img_idx}"
                for img_idx in range(len(data[image_key]))
            ]
            del data[image_key]
            data["images"] = image_urls
        
        # TODO: 
        # Trim to matching lengths
        origins, feature_acts = trim_minimum(origins, feature_acts)
        assert origins is not None and feature_acts is not None, "Origins and feature acts must not be None"

        # board_state = data["fen"]
        
        if "chess" in model_name or "lc0" in model_name:
            pass # 直接用data['fen']
            # data["text"] = board_state
            data['text'] = data["fen"]
            
            # 移除自动stockfish分析，让前端可以先显示棋盘再单独分析
            # 原本的自动分析代码已移除以提高响应速度
                    
        # Process text data if present
        if "text" in data and "lc0" not in model_name:
            text_ranges = [origin["range"] for origin in origins if origin is not None and origin["key"] == "text"]
            if text_ranges:
                max_text_origin = max(text_ranges, key=lambda x: x[1])
                data["text"] = data["text"][: max_text_origin[1]]
        
        print(f"{data = }after process_sample")
        # print(f"{shard_idx = }")
        # print(f"{n_shards = }")
        # print(f"{context_idx = }")
        # print(f"{feature_acts = }")
        # return {**data, "origins": origins, "feature_acts":context_idx feature_acts}
        return {**data, "origins": origins, "feature_acts": feature_acts, "context_idx": context_idx}

    # Process all samples for each sampling
    sample_groups = []
    for sampling in analysis.samplings:
        # Using zip to process correlated data instead of indexing
        samples = [
            process_sample(
                feature_acts=feature_acts,
                context_idx=context_idx,
                dataset_name=dataset_name,
                model_name=model_name,
                shard_idx=shard_idx,
                n_shards=n_shards,
            )
            for feature_acts, context_idx, dataset_name, model_name, shard_idx, n_shards in zip(
                sampling.feature_acts,
                sampling.context_idx,
                sampling.dataset_name,
                sampling.model_name,
                sampling.shard_idx if sampling.shard_idx is not None else [0] * len(sampling.feature_acts),
                sampling.n_shards if sampling.n_shards is not None else [1] * len(sampling.feature_acts),
            )
        ]

        sample_groups.append(
            {
                "analysis_name": sampling.name,
                "samples": samples,
            }
        )

    print(f"{analysis.decoder_norms = }")
    # Prepare response
    response_data = {
        "feature_index": feature.index,
        "analysis_name": analysis.name,
        "interpretation": feature.interpretation,
        "dictionary_name": feature.sae_name,
        "logits": feature.logits,
        "decoder_norms": analysis.decoder_norms,
        "decoder_similarity_matrix": analysis.decoder_similarity_matrix,
        "decoder_inner_product_matrix": analysis.decoder_inner_product_matrix,
        "act_times": analysis.act_times,
        "max_feature_act": analysis.max_feature_acts,
        "n_analyzed_tokens": analysis.n_analyzed_tokens,
        "sample_groups": sample_groups,
        "is_bookmarked": client.is_bookmarked(sae_name=name, sae_series=sae_series, feature_index=feature.index),
    }

    return Response(
        content=msgpack.packb(make_serializable(response_data)),
        media_type="application/x-msgpack",
    )


@app.get("/dictionaries/{name}")
def get_dictionary(name: str):
    # Get feature activation times
    feature_activation_times = client.get_feature_act_times(name, sae_series=sae_series)
    if feature_activation_times is None:
        return Response(content=f"Dictionary {name} not found", status_code=404)

    # Create histogram of log activation times
    log_act_times = np.log10(np.array(list(feature_activation_times.values())))
    feature_activation_times_histogram = go.Histogram(
        x=log_act_times,
        nbinsx=100,
        hovertemplate="Count: %{y}<br>Range: %{x}<extra></extra>",
        marker_color="#636EFA",
        showlegend=False,
    ).to_plotly_json()

    # Get alive feature count
    alive_feature_count = client.get_alive_feature_count(name, sae_series=sae_series)
    if alive_feature_count is None:
        return Response(content=f"SAE {name} not found", status_code=404)

    # Prepare and return response
    response_data = {
        "dictionary_name": name,
        "feature_activation_times_histogram": [feature_activation_times_histogram],
        "alive_feature_count": alive_feature_count,
    }

    return Response(
        content=msgpack.packb(make_serializable(response_data)),
        media_type="application/x-msgpack",
    )


@app.get("/dictionaries/{name}/analyses")
def get_analyses(name: str):
    """Get all available analyses for a dictionary.

    Args:
        name: Name of the dictionary/SAE

    Returns:
        List of analysis names
    """
    # Get a random feature to check its available analyses
    feature = client.get_random_alive_feature(sae_name=name, sae_series=sae_series)
    if feature is None:
        return Response(content=f"Dictionary {name} not found", status_code=404)

    # Extract unique analysis names from feature
    analyses = list(set(analysis.name for analysis in feature.analyses))
    return analyses


@app.post("/dictionaries/{name}/features/{feature_index}/bookmark")
def add_bookmark(name: str, feature_index: int):
    """Add a bookmark for a feature.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature to bookmark

    Returns:
        Success response or error
    """
    try:
        success = client.add_bookmark(sae_name=name, sae_series=sae_series, feature_index=feature_index)
        if success:
            return {"message": "Bookmark added successfully"}
        else:
            return Response(content="Feature is already bookmarked", status_code=409)
    except ValueError as e:
        return Response(content=str(e), status_code=404)


@app.delete("/dictionaries/{name}/features/{feature_index}/bookmark")
def remove_bookmark(name: str, feature_index: int):
    """Remove a bookmark for a feature.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature to remove bookmark from

    Returns:
        Success response or error
    """
    success = client.remove_bookmark(sae_name=name, sae_series=sae_series, feature_index=feature_index)
    if success:
        return {"message": "Bookmark removed successfully"}
    else:
        return Response(content="Bookmark not found", status_code=404)


@app.get("/dictionaries/{name}/features/{feature_index}/bookmark")
def check_bookmark(name: str, feature_index: int):
    """Check if a feature is bookmarked.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature

    Returns:
        Bookmark status
    """
    is_bookmarked = client.is_bookmarked(sae_name=name, sae_series=sae_series, feature_index=feature_index)
    return {"is_bookmarked": is_bookmarked}


@app.get("/bookmarks")
def list_bookmarks(sae_name: Optional[str] = None, sae_series: Optional[str] = None, limit: int = 100, skip: int = 0):
    """List bookmarks with optional filtering.

    Args:
        sae_name: Optional SAE name filter
        sae_series: Optional SAE series filter
        limit: Maximum number of bookmarks to return
        skip: Number of bookmarks to skip (for pagination)

    Returns:
        List of bookmarks
    """
    bookmarks = client.list_bookmarks(sae_name=sae_name, sae_series=sae_series, limit=limit, skip=skip)

    # Convert to dict for JSON serialization
    bookmark_data = []
    for bookmark in bookmarks:
        bookmark_dict = bookmark.model_dump()
        # Convert datetime to ISO string for JSON
        bookmark_dict["created_at"] = bookmark.created_at.isoformat()
        bookmark_data.append(bookmark_dict)

    return {
        "bookmarks": bookmark_data,
        "total_count": client.get_bookmark_count(sae_name=sae_name, sae_series=sae_series),
    }


@app.put("/dictionaries/{name}/features/{feature_index}/bookmark")
def update_bookmark(name: str, feature_index: int, tags: Optional[list[str]] = None, notes: Optional[str] = None):
    """Update a bookmark with new tags or notes.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature
        tags: Optional new tags for the bookmark
        notes: Optional new notes for the bookmark

    Returns:
        Success response or error
    """
    success = client.update_bookmark(
        sae_name=name, sae_series=sae_series, feature_index=feature_index, tags=tags, notes=notes
    )
    if success:
        return {"message": "Bookmark updated successfully"}
    else:
        return Response(content="Bookmark not found", status_code=404)


@app.post("/cancel_all_sessions")
async def cancel_all_analysis_sessions():
    """取消所有正在进行的分析会话
    
    这个端点用于前端切换feature时快速清理所有旧的分析任务，
    避免后台继续进行无用的Stockfish计算。
    
    Returns:
        取消操作的结果统计
    """
    try:
        cancelled_count = 0
        session_ids = []
        
        with session_lock:
            # 获取所有活跃会话的ID
            session_ids = list(active_sessions.keys())
            
            # 标记所有会话为取消状态
            for session_id in session_ids:
                if session_id in active_sessions:
                    active_sessions[session_id]['cancel_event'].set()
                    cancelled_count += 1
            
            # 清空活跃会话列表
            active_sessions.clear()
        
        logging.info(f"🧹 批量取消 {cancelled_count} 个分析会话: {session_ids[:5]}{'...' if len(session_ids) > 5 else ''}")
        
        return {
            "status": "success",
            "message": f"已取消 {cancelled_count} 个分析会话",
            "cancelled_count": cancelled_count,
            "cancelled_sessions": session_ids[:10]  # 只返回前10个，避免响应过大
        }
        
    except Exception as e:
        logging.error(f"批量取消会话时出错: {e}")
        return Response(
            content=f"批量取消会话出错: {str(e)}", 
            status_code=500
        )


@app.post("/cancel_session")
async def cancel_analysis_session(request_data: dict):
    """取消指定的分析会话
    
    Args:
        request_data: 包含session_id的请求数据
        
    Returns:
        取消操作的结果
    """
    try:
        session_id = request_data.get("session_id")
        if not session_id:
            return Response(content="缺少session_id参数", status_code=400)
        
        success = cancel_session(session_id)
        
        if success:
            return {
                "status": "success",
                "message": f"会话 {session_id} 已取消",
                "session_id": session_id
            }
        else:
            return {
                "status": "not_found",
                "message": f"会话 {session_id} 不存在或已结束",
                "session_id": session_id
            }
            
    except Exception as e:
        logging.error(f"取消会话时出错: {e}")
        return Response(
            content=f"取消会话出错: {str(e)}", 
            status_code=500
        )


@app.post("/analyze/stockfish")
async def analyze_stockfish_simple(request_data: dict):
    """简化的异步分析FEN局面的Stockfish最佳走法、规则分析、物质力量、WDL评估和模型推理
    
    Args:
        request_data: 包含FEN字符串的请求数据
        
    Returns:
        Stockfish分析结果、规则分析结果、物质力量分析、WDL评估和模型推理结果
    """
    try:
        fen = request_data.get("fen")
        include_rules = request_data.get("include_rules", True)  # 默认包含规则分析
        include_material = request_data.get("include_material", True)  # 默认包含物质分析
        include_wdl = request_data.get("include_wdl", True)  # 默认包含WDL评估
        include_model = request_data.get("include_model", True)  # 默认包含模型推理
        
        logging.info(f"📥 收到分析请求 FEN: {fen[:30] if fen else 'None'}...")
        
        if not fen:
            logging.error("❌ 缺少FEN参数")
            return Response(content="缺少FEN参数", status_code=400)
        
        # 执行简化的Stockfish分析
        logging.info(f"🚀 开始调用Stockfish分析: {fen[:30]}...")
        analysis_result = await analyze_position_with_stockfish_simple(fen)
        
        if analysis_result is not None:
            best_move, ponder = analysis_result
            logging.info(f"✅ API成功返回结果: {best_move}")
            
            stockfish_analysis = {
                "best_move": best_move if best_move else None,
                "ponder": ponder if ponder else None,
                "status": "success" if best_move else "no_move",
                "error": None,
                "fen": fen
            }
            
            # 如果找到最佳着法，添加额外信息
            if best_move:
                try:
                    board = chess.Board(fen)
                    move = chess.Move.from_uci(best_move)
                    
                    # 添加走法的SAN表示
                    stockfish_analysis["move_san"] = board.san(move)
                    
                    # 添加起始和目标格子
                    stockfish_analysis["from_square"] = chess.square_name(move.from_square)
                    stockfish_analysis["to_square"] = chess.square_name(move.to_square)
                    
                    # 如果是升变着法，添加升变信息
                    if move.promotion:
                        stockfish_analysis["promotion"] = chess.piece_name(move.promotion)
                    
                    # 检查是否将军
                    board.push(move)
                    stockfish_analysis["is_check"] = board.is_check()
                    
                except Exception as e:
                    logging.warning(f"生成走法额外信息时出错: {e}")
        else:
            logging.info(f"❌ API分析无结果: {fen[:30]}...")
            stockfish_analysis = {
                "best_move": None,
                "ponder": None,
                "status": "no_move", 
                "error": "分析失败",
                "fen": fen
            }
        
        # 添加规则分析
        if include_rules:
            logging.info(f"🧩 开始规则分析: {fen[:30]}...")
            rules_analysis = analyze_chess_rules(fen)
            stockfish_analysis["rules"] = rules_analysis
            logging.info(f"🧩 规则分析完成: {len(rules_analysis)} 项规则")
        
        # 添加物质力量分析
        if include_material:
            logging.info(f"💰 开始物质分析: {fen[:30]}...")
            material_analysis = calculate_material_balance(fen)
            stockfish_analysis["material"] = material_analysis
            if "error" not in material_analysis:
                logging.info(f"💰 物质分析完成: 白方{material_analysis['white_material']} vs 黑方{material_analysis['black_material']}")
        
        # 添加WDL评估
        if include_wdl:
            logging.info(f"📊 开始WDL评估: {fen[:30]}...")
            wdl_analysis = await get_wdl_evaluation(fen)
            stockfish_analysis["wdl"] = wdl_analysis
            if "error" not in wdl_analysis:
                logging.info(f"📊 WDL评估完成: W{wdl_analysis.get('win_probability', '?')} D{wdl_analysis.get('draw_probability', '?')} L{wdl_analysis.get('loss_probability', '?')}")
        
        # 添加模型推理分析
        if include_model:
            include_selfplay = request_data.get("include_selfplay", False)  # 新增参数控制是否包含self-play
            logging.info(f"🤖 开始模型推理: {fen[:30]}...")
            
            if include_selfplay:
                model_analysis = await analyze_position_with_model_and_selfplay(fen)
            else:
                model_analysis = await analyze_position_with_model(fen)
                
            stockfish_analysis["model"] = model_analysis
            if "error" not in model_analysis:
                best_move_model = model_analysis.get("best_move", "无")
                model_wdl = model_analysis.get("wdl_analysis", {})
                selfplay_info = ""
                if "selfplay" in model_analysis:
                    selfplay_moves = model_analysis["selfplay"].get("total_moves", 0)
                    selfplay_info = f", Self-play: {selfplay_moves}步"
                logging.info(f"🤖 模型推理完成: 最佳走法={best_move_model}, WDL={model_wdl.get('win_percent', '?')}/{model_wdl.get('draw_percent', '?')}/{model_wdl.get('loss_percent', '?')}{selfplay_info}")
            
        logging.info(f"📤 API返回响应: status={stockfish_analysis['status']}")
        return stockfish_analysis
        
    except Exception as e:
        logging.error(f"💥 Stockfish分析API错误: {e}")
        return Response(
            content=f"分析出错: {str(e)}", 
            status_code=500
        )


@app.post("/analyze/rules")
async def analyze_chess_rules_api(request_data: dict):
    """分析FEN局面的国际象棋规则
    
    Args:
        request_data: 包含FEN字符串的请求数据
        
    Returns:
        规则分析结果
    """
    try:
        fen = request_data.get("fen")
        
        logging.info(f"📥 收到规则分析请求 FEN: {fen[:30] if fen else 'None'}...")
        
        if not fen:
            logging.error("❌ 缺少FEN参数")
            return Response(content="缺少FEN参数", status_code=400)
        
        # 执行规则分析
        logging.info(f"🧩 开始规则分析: {fen[:30]}...")
        rules_result = analyze_chess_rules(fen)
        
        response_data = {
            "fen": fen,
            "rules": rules_result,
            "status": "success" if "error" not in rules_result else "error"
        }
        
        logging.info(f"📤 规则分析API返回响应: {len(rules_result)} 项规则")
        return response_data
        
    except Exception as e:
        logging.error(f"💥 规则分析API错误: {e}")
        return Response(
            content=f"规则分析出错: {str(e)}", 
            status_code=500
        )


@app.post("/analyze/model")
async def analyze_model_api(request_data: dict):
    """使用自己的模型分析FEN局面
    
    Args:
        request_data: 包含FEN字符串的请求数据
        
    Returns:
        模型推理结果
    """
    try:
        fen = request_data.get("fen")
        include_selfplay = request_data.get("include_selfplay", False)
        
        logging.info(f"📥 收到模型推理请求 FEN: {fen[:30] if fen else 'None'}...")
        
        if not fen:
            logging.error("❌ 缺少FEN参数")
            return Response(content="缺少FEN参数", status_code=400)
        
        # 执行模型推理
        logging.info(f"🤖 开始模型推理: {fen[:30]}...")
        if include_selfplay:
            model_result = await analyze_position_with_model_and_selfplay(fen)
        else:
            model_result = await analyze_position_with_model(fen)
        
        response_data = {
            "fen": fen,
            "model": model_result,
            "status": "success" if "error" not in model_result else "error"
        }
        
        logging.info(f"📤 模型推理API返回响应")
        return response_data
        
    except Exception as e:
        logging.error(f"💥 模型推理API错误: {e}")
        return Response(
            content=f"模型推理出错: {str(e)}", 
            status_code=500
        )


@app.post("/analyze/selfplay")
async def analyze_selfplay_api(request_data: dict):
    """使用模型进行self-play分析
    
    Args:
        request_data: 包含FEN字符串和self-play参数的请求数据
        
    Returns:
        Self-play分析结果
    """
    try:
        fen = request_data.get("fen")
        max_moves = request_data.get("max_moves", 10)
        temperature = request_data.get("temperature", 1.0)
        
        logging.info(f"📥 收到Self-play请求 FEN: {fen[:30] if fen else 'None'}...")
        
        if not fen:
            logging.error("❌ 缺少FEN参数")
            return Response(content="缺少FEN参数", status_code=400)
        
        if not global_chess_model or not TRANSFORMER_LENS_AVAILABLE:
            return Response(content="模型未加载或不可用", status_code=503)
        
        # 执行self-play
        def run_selfplay():
            try:
                logging.info(f"🎮 开始Self-play分析: {fen[:30]}...")
                selfplay_engine = HookedSelfPlayEngine(global_chess_model, temperature=temperature)
                selfplay_result = selfplay_engine.play_game(initial_fen=fen, max_moves=max_moves)
                logging.info(f"🎮 Self-play完成: {selfplay_result['total_moves']} 步")
                return selfplay_result
            except Exception as e:
                logging.error(f"Self-play分析失败: {e}")
                return {"error": f"Self-play分析失败: {str(e)}"}
        
        selfplay_result = await to_thread(run_selfplay)
        
        response_data = {
            "fen": fen,
            "selfplay": selfplay_result,
            "status": "success" if "error" not in selfplay_result else "error",
            "parameters": {
                "max_moves": max_moves,
                "temperature": temperature
            }
        }
        
        logging.info(f"📤 Self-play API返回响应")
        return response_data
        
    except Exception as e:
        logging.error(f"💥 Self-play API错误: {e}")
        return Response(
            content=f"Self-play分析出错: {str(e)}", 
            status_code=500
        )


@app.get("/analyze/model/status")
async def get_model_status():
    """获取模型状态信息
    
    Returns:
        模型状态和信息
    """
    global global_chess_model
    
    try:
        status_info = {
            "transformer_lens_available": TRANSFORMER_LENS_AVAILABLE,
            "model_loaded": global_chess_model is not None,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if global_chess_model is not None:
            try:
                # 获取模型基本信息
                status_info.update({
                    "model_name": "lc0/T82-768x15x24h",
                    "model_device": str(next(global_chess_model.parameters()).device),
                    "model_dtype": str(next(global_chess_model.parameters()).dtype),
                })
            except Exception as e:
                logging.warning(f"获取模型详细信息失败: {e}")
                status_info["model_info_error"] = str(e)
        
        return {
            "status": "ready" if status_info["model_loaded"] else "unavailable",
            "info": status_info
        }
        
    except Exception as e:
        logging.error(f"获取模型状态失败: {e}")
        return Response(
            content=f"获取模型状态出错: {str(e)}", 
            status_code=500
        )


@app.post("/analyze/stockfish/batch")
async def analyze_stockfish_batch(request_data: dict):
    """批量分析多个FEN局面
    
    Args:
        request_data: 包含FEN列表的请求数据
        
    Returns:
        批量Stockfish分析结果
    """
    try:
        fens = request_data.get("fens", [])
        if not fens:
            return Response(content="缺少FEN列表", status_code=400)
        
        results = {}
        for fen in fens:
            if isinstance(fen, str) and fen.strip():
                analysis_result = await analyze_position_with_stockfish_simple(fen)
                
                if analysis_result is not None:
                    best_move, ponder = analysis_result
                    results[fen] = {
                        "best_move": best_move if best_move else None,
                        "ponder": ponder if ponder else None,
                        "status": "success" if best_move else "no_move",
                        "error": None,
                        "fen": fen
                    }
                    
                    # 添加额外信息
                    if best_move:
                        try:
                            board = chess.Board(fen)
                            move = chess.Move.from_uci(best_move)
                            
                            results[fen]["move_san"] = board.san(move)
                            results[fen]["from_square"] = chess.square_name(move.from_square)
                            results[fen]["to_square"] = chess.square_name(move.to_square)
                            
                            if move.promotion:
                                results[fen]["promotion"] = chess.piece_name(move.promotion)
                            
                            board.push(move)
                            results[fen]["is_check"] = board.is_check()
                            
                        except Exception as e:
                            logging.warning(f"生成走法额外信息时出错: {e}")
                else:
                    results[fen] = {
                        "best_move": None,
                        "ponder": None,
                        "status": "no_move",
                        "error": "分析失败",
                        "fen": fen
                    }
        
        return {"results": results}
        
    except Exception as e:
        logging.error(f"批量Stockfish分析API错误: {e}")
        return Response(
            content=f"批量分析出错: {str(e)}", 
            status_code=500
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 在模型相关导入之后添加 self-play 引擎类
class HookedSelfPlayEngine:
    """使用HookTransformer进行self-play的引擎"""
    
    def __init__(self, model, temperature: float = 1.0):
        """
        初始化self-play引擎
        
        Args:
            model: HookedTransformer模型
            temperature: 温度参数，控制随机性
        """
        self.model = model
        self.temperature = temperature
        self.model.eval()
        
        # 记录游戏历史
        self.game_history = []
        self.move_probabilities = []
    
    def get_top_moves(self, fen: str, top_k: int = 5) -> List[Tuple[str, float, int]]:
        """
        获取前k个最佳移动
        
        Args:
            fen: FEN字符串
            top_k: 返回前k个移动
            
        Returns:
            移动列表: [(uci_move, probability, index), ...]
        """
        try:
            # 运行模型推理
            output, cache = self.model.run_with_cache(fen, prepend_bos=False)
            policy_output = output[0]  # [1, 1858]
            
            # 获取合法移动
            chess_board = chess.Board(fen)
            legal_moves_list = list(chess_board.legal_moves)
            legal_uci_set = set(move.uci() for move in legal_moves_list)
            
            # 应用温度缩放
            scaled_logits = policy_output / self.temperature
            
            # 计算softmax概率
            probabilities = torch.softmax(scaled_logits, dim=-1)[0]  # [1858]
            
            # 获取所有移动的索引和概率
            all_moves = []
            
            # 确定使用哪个映射表
            mapping_idx = get_mapping_index(chess_board)
            idx_to_uci = idx_to_uci_mappings[mapping_idx]
            
            for idx in range(len(policy_output[0])):
                try:
                    if idx in idx_to_uci:
                        uci = idx_to_uci[idx]
                        if uci in legal_uci_set:
                            prob = probabilities[idx].item()
                            all_moves.append((uci, prob, idx))
                except:
                    continue
            
            # 按概率排序并返回前k个
            all_moves.sort(key=lambda x: x[1], reverse=True)
            return all_moves[:top_k]
            
        except Exception as e:
            logging.error(f"获取最佳移动时出错: {e}")
            return []
    
    def select_move(self, fen: str) -> Tuple[str, float]:
        """
        根据概率分布选择移动
        
        Args:
            fen: FEN字符串
            
        Returns:
            (选择的移动, 移动概率)
        """
        top_moves = self.get_top_moves(fen, top_k=10)  # 获取前10个作为候选
        
        if not top_moves:
            # 如果没有合法移动，随机选择一个
            chess_board = chess.Board(fen)
            legal_moves = list(chess_board.legal_moves)
            if legal_moves:
                move = np.random.choice(legal_moves)
                return move.uci(), 1.0 / len(legal_moves)
            else:
                raise ValueError("没有合法移动")
        
        # 根据概率分布选择移动
        moves, probs, indices = zip(*top_moves)
        chosen_idx = np.random.choice(len(moves), p=np.array(probs) / sum(probs))
        chosen_move = moves[chosen_idx]
        chosen_prob = probs[chosen_idx]
        
        return chosen_move, chosen_prob
    
    def play_game(self, initial_fen: str = None, max_moves: int = 10) -> Dict:
        """
        进行一局self-play游戏
        
        Args:
            initial_fen: 初始FEN字符串，None表示标准开局
            max_moves: 最大移动数
            
        Returns:
            游戏结果字典
        """
        # 初始化棋盘
        if initial_fen:
            try:
                board = chess.Board(initial_fen)
            except ValueError:
                board = chess.Board()
        else:
            board = chess.Board()
        
        # 重置游戏历史
        self.game_history = []
        self.move_probabilities = []
        
        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            move_count += 1
            current_player = "white" if board.turn == chess.WHITE else "black"
            current_fen = board.fen()
            
            try:
                # 获取前5个最佳移动
                top_moves = self.get_top_moves(current_fen, top_k=5)
                
                # 选择移动
                chosen_move, chosen_prob = self.select_move(current_fen)
                
                # 记录游戏历史
                self.game_history.append({
                    'move_number': move_count,
                    'player': current_player,
                    'move': chosen_move,
                    'probability': chosen_prob,
                    'fen_before': current_fen,
                    'fen_after': None,  # 将在执行移动后填充
                    'top_moves': top_moves,
                    'move_san': None  # 将在执行移动后填充
                })
                self.move_probabilities.append(chosen_prob)
                
                # 执行移动
                move_obj = chess.Move.from_uci(chosen_move)
                move_san = board.san(move_obj)
                board.push(move_obj)
                
                # 更新历史记录中的信息
                self.game_history[-1]['fen_after'] = board.fen()
                self.game_history[-1]['move_san'] = move_san
                
            except Exception as e:
                logging.error(f"Self-play移动选择错误: {e}")
                break
        
        # 游戏结束
        result = board.result()
        outcome = board.outcome()
        
        # 返回游戏结果
        game_result = {
            'initial_fen': initial_fen or chess.Board().fen(),
            'final_fen': board.fen(),
            'result': result,
            'total_moves': len(self.game_history),
            'termination': outcome.termination.name if outcome else None,
            'game_history': self.game_history,
            'move_probabilities': self.move_probabilities,
            'temperature': self.temperature
        }
        
        return game_result

async def analyze_position_with_model_and_selfplay(fen: str) -> dict:
    """
    使用自己的模型分析FEN局面并进行self-play
    
    Args:
        fen: 棋局的FEN字符串
        
    Returns:
        dict: 包含模型分析结果和self-play结果的字典
    """
    global global_chess_model
    
    if not global_chess_model or not TRANSFORMER_LENS_AVAILABLE:
        return {"error": "模型未加载或不可用"}
    
    try:
        def run_model_inference_with_selfplay():
            try:
                # 先进行常规模型分析
                with torch.no_grad():
                    outputs, cache = global_chess_model.run_with_cache(fen, prepend_bos=False)
                
                # 解析输出
                policy_logits = outputs[0][0] if len(outputs) > 0 else None  # 策略logits
                wdl_probs = outputs[1][0] if len(outputs) > 1 else None      # WDL概率
                value_score = outputs[2][0] if len(outputs) > 2 else None    # 价值评估
                
                result = {
                    "raw_outputs": {
                        "policy_logits_shape": list(policy_logits.shape) if policy_logits is not None else None,
                        "wdl_probs_shape": list(wdl_probs.shape) if wdl_probs is not None else None,
                        "value_shape": list(value_score.shape) if value_score is not None else None,
                    }
                }
                
                # 解析策略输出得到最佳移动（保持原有逻辑）
                if policy_logits is not None:
                    try:
                        board = chess.Board(fen)
                        legal_moves_list = list(board.legal_moves)
                        
                        # 计算softmax概率
                        policy_probs = torch.softmax(policy_logits, dim=0)
                        
                        # 确定使用哪个映射表
                        mapping_idx = get_mapping_index(board)
                        idx_to_uci = idx_to_uci_mappings[mapping_idx]
                        
                        # 方法1：通过索引排序查找
                        sorted_indices = torch.argsort(policy_logits, descending=True)
                        best_moves = []
                        checked_moves = 0
                        max_check = min(500, len(sorted_indices))
                        
                        for idx in sorted_indices[:max_check]:
                            try:
                                move_idx = idx.item()
                                if move_idx in idx_to_uci:
                                    uci_move = idx_to_uci[move_idx]
                                    
                                    try:
                                        move = chess.Move.from_uci(uci_move)
                                        
                                        # 检查移动是否合法
                                        if move in legal_moves_list:
                                            move_score = policy_logits[idx].item()
                                            move_prob = policy_probs[idx].item()
                                            
                                            best_moves.append({
                                                "uci": uci_move,
                                                "san": board.san(move),
                                                "score": round(move_score, 4),
                                                "probability": round(move_prob * 100, 2),
                                                "move_index": move_idx
                                            })
                                            
                                            if len(best_moves) >= 5:
                                                break
                                    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                                        continue
                                
                                checked_moves += 1
                            except Exception:
                                continue
                        
                        # 方法2：如果方法1失败，使用备用方案
                        if not best_moves:
                            backup_moves = find_best_moves_from_legal_moves(policy_logits, legal_moves_list, board)
                            if backup_moves:
                                best_moves = backup_moves
                        
                        result["policy_analysis"] = {
                            "best_moves": best_moves,
                            "total_legal_moves": len(legal_moves_list),
                            "checked_indices": checked_moves,
                            "policy_shape": list(policy_logits.shape),
                            "mapping_size": len(idx_to_uci),
                            "mapping_index": mapping_idx,
                            "method_used": "primary" if checked_moves > 0 else "backup"
                        }
                        
                        if best_moves:
                            result["best_move"] = best_moves[0]["uci"]
                            result["best_move_san"] = best_moves[0]["san"]
                            result["best_move_probability"] = best_moves[0]["probability"]
                        else:
                            # 最后的备用方案：随机选择一个合法移动
                            if legal_moves_list:
                                random_move = legal_moves_list[0]
                                result["best_move"] = random_move.uci()
                                result["best_move_san"] = board.san(random_move)
                                result["best_move_probability"] = 0.0
                                result["policy_analysis"]["fallback_move"] = True
                        
                    except Exception as e:
                        logging.error(f"策略分析失败: {e}")
                        result["policy_error"] = str(e)
                
                # 解析WDL概率
                if wdl_probs is not None:
                    try:
                        win_prob = wdl_probs[0].item() if len(wdl_probs) > 0 else 0
                        draw_prob = wdl_probs[1].item() if len(wdl_probs) > 1 else 0
                        loss_prob = wdl_probs[2].item() if len(wdl_probs) > 2 else 0
                        
                        result["wdl_analysis"] = {
                            "win_probability": round(win_prob, 4),
                            "draw_probability": round(draw_prob, 4),
                            "loss_probability": round(loss_prob, 4),
                            "win_percent": round(win_prob * 100, 1),
                            "draw_percent": round(draw_prob * 100, 1),
                            "loss_percent": round(loss_prob * 100, 1)
                        }
                    except Exception as e:
                        logging.error(f"WDL分析失败: {e}")
                        result["wdl_error"] = str(e)
                
                # 解析价值评估
                if value_score is not None:
                    try:
                        value = value_score.item() if hasattr(value_score, 'item') else float(value_score)
                        result["value_analysis"] = {
                            "raw_value": round(value, 4),
                            "normalized_value": round(value / 100, 4)
                        }
                    except Exception as e:
                        logging.error(f"价值分析失败: {e}")
                        result["value_error"] = str(e)
                
                # 进行self-play
                try:
                    logging.info(f"🎮 开始self-play分析: {fen[:30]}...")
                    selfplay_engine = HookedSelfPlayEngine(global_chess_model, temperature=1.0)
                    selfplay_result = selfplay_engine.play_game(initial_fen=fen, max_moves=10)
                    result["selfplay"] = selfplay_result
                    logging.info(f"🎮 Self-play完成: {selfplay_result['total_moves']} 步")
                except Exception as e:
                    logging.error(f"Self-play分析失败: {e}")
                    result["selfplay_error"] = str(e)
                
                return result
                
            except Exception as e:
                logging.error(f"模型推理内部错误: {e}")
                return {"error": f"模型推理失败: {str(e)}"}
        
        # 在线程中运行推理以避免阻塞
        result = await to_thread(run_model_inference_with_selfplay)
        
        logging.info(f"🤖 模型分析+Self-play完成 for {fen[:20]}...")
        return result
        
    except Exception as e:
        logging.error(f"模型分析异常: {e}")
        return {"error": f"模型分析异常: {str(e)}"}