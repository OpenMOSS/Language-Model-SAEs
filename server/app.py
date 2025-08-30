import io
import json
import os
import signal
import sys
import logging
import threading
import time
import hashlib
from functools import lru_cache
from typing import Any, Optional, List, Tuple, Dict
from asyncio import to_thread
import importlib.util

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
from fastapi.responses import StreamingResponse

# 新增：缺失导入
import asyncio
from collections import deque

from lm_saes.backend import LanguageModel
from lm_saes.config import MongoDBConfig, SAEConfig
from lm_saes.database import MongoClient
from lm_saes.resource_loaders import load_dataset_shard, load_model
from lm_saes.sae import SparseAutoEncoder
from lm_saes.lc0_mapping import (
    uci_to_idx_mappings, idx_to_uci_mappings, get_mapping_index
)

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


# 分析结果缓存管理器
class AnalysisCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def _make_key(self, fen, analysis_type, **params):
        """生成缓存键"""
        params_str = json.dumps(params, sort_keys=True)
        key_data = f"{fen}_{analysis_type}_{params_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, fen, analysis_type, **params):
        """获取缓存的分析结果"""
        key = self._make_key(fen, analysis_type, **params)
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def set(self, fen, analysis_type, result, **params):
        """设置缓存的分析结果"""
        key = self._make_key(fen, analysis_type, **params)
        with self.lock:
            # 如果缓存已满，删除最旧的条目
            if len(self.cache) >= self.max_size:
                oldest_key = min(
                    self.access_times.keys(),
                    key=lambda k: self.access_times[k]
                )
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = result
            self.access_times[key] = time.time()
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()


# 全局分析缓存实例
global_analysis_cache = AnalysisCache(max_size=500)


# 任务管理器
class TaskManager:
    def __init__(self):
        self.active_tasks = {}  # task_id -> task_info
        self.task_queue = []  # 优先级队列
        self.lock = threading.Lock()
        self.next_task_id = 0
    
    def create_task(self, task_type, priority=0, **kwargs):
        """创建新任务"""
        with self.lock:
            task_id = f"{task_type}_{int(time.time())}_{self.next_task_id}"
            self.next_task_id += 1
            
            task_info = {
                'task_id': task_id,
                'task_type': task_type,
                'priority': priority,
                'status': 'pending',
                'created_at': time.time(),
                'cancel_event': threading.Event(),
                'kwargs': kwargs
            }
            
            self.active_tasks[task_id] = task_info
            return task_id
    
    def cancel_task(self, task_id):
        """取消任务"""
        with self.lock:
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['cancel_event'].set()
                self.active_tasks[task_id]['status'] = 'cancelled'
                logging.info(f"⚡ 任务已取消: {task_id}")
                return True
            return False
    
    def cancel_tasks_by_pattern(self, pattern):
        """根据模式取消任务"""
        cancelled_count = 0
        with self.lock:
            active_task_ids = list(self.active_tasks.keys())
            logging.info(f"🔍 检查任务取消模式 '{pattern}': 当前活跃任务 {len(active_task_ids)} 个")
            
            # 批量取消所有匹配的任务
            matching_tasks = []
            for task_id in active_task_ids:
                if pattern in task_id:
                    matching_tasks.append(task_id)
            
            logging.info(f"🔍 找到 {len(matching_tasks)} 个匹配的任务: {matching_tasks}")
            
            # 快速批量设置取消标志
            for task_id in matching_tasks:
                if task_id in self.active_tasks:
                    self.active_tasks[task_id]['cancel_event'].set()
                    self.active_tasks[task_id]['status'] = 'cancelled'
                    cancelled_count += 1
                    
            if cancelled_count > 0:
                logging.info(f"⚡ 批量取消完成: {cancelled_count} 个任务已设置取消标志")
            
        return cancelled_count
    
    def is_cancelled(self, task_id):
        """检查任务是否被取消"""
        with self.lock:
            if task_id in self.active_tasks:
                return self.active_tasks[task_id]['cancel_event'].is_set()
            return False
    
    def complete_task(self, task_id):
        """标记任务完成"""
        with self.lock:
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = 'completed'
                # 清理老任务
                self._cleanup_old_tasks()
                
    def get_all_tasks(self):
        """获取所有活跃任务信息"""
        with self.lock:
            return {
                task_id: {
                    'task_type': info['task_type'],
                    'priority': info['priority'],
                    'status': info['status'],
                    'created_at': info['created_at'],
                    'is_cancelled': info['cancel_event'].is_set()
                }
                for task_id, info in self.active_tasks.items()
            }
            
    def force_clear_all_tasks(self):
        """强制清理所有任务（紧急情况使用）"""
        with self.lock:
            cleared_count = len(self.active_tasks)
            # 设置所有任务为取消状态
            for task_info in self.active_tasks.values():
                task_info['cancel_event'].set()
                task_info['status'] = 'force_cancelled'
            # 清空任务列表
            self.active_tasks.clear()
            logging.warning(f"🚨 强制清理了 {cleared_count} 个活跃任务")
            return cleared_count
    
    def _cleanup_old_tasks(self):
        """清理超过10分钟的老任务"""
        current_time = time.time()
        old_tasks = [
            task_id for task_id, task_info in self.active_tasks.items()
            if current_time - task_info['created_at'] > 600  # 10分钟
        ]
        for task_id in old_tasks:
            del self.active_tasks[task_id]


# 全局任务管理器
global_task_manager = TaskManager()

# 添加规则模块路径
# 尝试多种路径方式
possible_paths = [
    os.path.join(os.path.dirname(__file__), '../exp/07rule'),
    os.path.join(os.getcwd(), 'exp/07rule'),
    os.path.join(os.getcwd(), '../exp/07rule'),
    '/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/'
    'rlin_projects/chess-SAEs/exp/07rule'
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
    print("❌ 未找到规则模块目录")
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
    
    # 关闭所有活跃的引擎实例
    global_engine_manager.close_all_engines()
    
    # 清理全局引擎引用
    if global_engine:
        try:
            global_engine = None
            logging.info("✓ Stockfish引擎引用已清理")
        except Exception as e:
            logging.warning(f"⚠️ 清理Stockfish引擎时出错: {e}")
    
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

# 引擎管理器 - 提供更安全的引擎操作
class EngineManager:
    def __init__(self):
        self.engine_lock = threading.Lock()
        self.active_engines = set()
    
    def create_engine(self) -> Optional[chess.engine.SimpleEngine]:
        """创建新的引擎实例"""
        try:
            engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
            with self.engine_lock:
                self.active_engines.add(engine)
            return engine
        except Exception as e:
            logging.error(f"创建引擎失败: {e}")
            return None
    
    def close_engine(self, engine: chess.engine.SimpleEngine):
        """安全关闭引擎实例"""
        if engine is None:
            return
        
        with self.engine_lock:
            if engine in self.active_engines:
                self.active_engines.remove(engine)
        
        try:
            engine.quit()
        except Exception as e:
            logging.warning(f"关闭引擎时出错: {e}")
    
    def close_all_engines(self):
        """关闭所有活跃的引擎"""
        with self.engine_lock:
            engines_to_close = list(self.active_engines)
            self.active_engines.clear()
        
        for engine in engines_to_close:
            try:
                engine.quit()
            except Exception as e:
                logging.warning(f"关闭引擎时出错: {e}")
        
        logging.info(f"已关闭 {len(engines_to_close)} 个引擎实例")

# 全局引擎管理器
global_engine_manager = EngineManager()

# 信号处理器 - 优雅地处理Ctrl+C
def signal_handler(signum, frame):
    """处理Ctrl+C信号，优雅地关闭所有资源"""
    logging.info(f"🛑 收到信号 {signum}，开始优雅关闭...")
    
    try:
        # 关闭所有活跃的引擎实例
        global_engine_manager.close_all_engines()
        
        # 强制清理所有任务
        global_task_manager.force_clear_all_tasks()
        
        # 清理所有会话
        with session_lock:
            for session_id, session_info in active_sessions.items():
                session_info['cancel_event'].set()
            active_sessions.clear()
        
        logging.info("✅ 优雅关闭完成")
        
    except Exception as e:
        logging.error(f"❌ 优雅关闭时出错: {e}")
    
    # 退出程序
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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
            engine = None
            try:
                engine = global_engine_manager.create_engine()
                if engine is None:
                    logging.error("无法创建WDL引擎实例")
                    return None
                
                # 配置引擎参数，确保启用WDL评估
                engine_options = ENGINE_OPTIONS.copy()
                # 一些Stockfish版本需要特殊设置来启用WDL
                try:
                    engine.configure(engine_options)
                except Exception as config_error:
                    logging.warning(f"WDL引擎配置失败: {config_error}")
                
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
            except chess.engine.EngineTerminatedError as e:
                logging.error(f"WDL引擎意外终止: {e}")
                return None
            except chess.engine.EngineError as e:
                logging.error(f"WDL引擎错误: {e}")
                return None
            except Exception as e:
                logging.error(f"WDL分析内部错误: {e}")
                return None
            finally:
                # 使用引擎管理器安全关闭引擎
                if engine:
                    global_engine_manager.close_engine(engine)
        
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
            engine = None
            start_time = time.time()
            max_analysis_time = ENGINE_TIME_LIMIT * 3  # 最大分析时间
            
            try:
                engine = global_engine_manager.create_engine()
                if engine is None:
                    logging.error("无法创建引擎实例")
                    return None
                
                # 配置引擎参数
                for option, value in ENGINE_OPTIONS.items():
                    try:
                        engine.configure({option: value})
                    except Exception as config_error:
                        logging.warning(f"引擎配置失败 {option}={value}: {config_error}")
                
                # 检查是否超时
                if time.time() - start_time > max_analysis_time:
                    logging.warning(f"引擎分析超时: {fen[:20]}...")
                    return None
                
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
            except chess.engine.EngineTerminatedError as e:
                logging.error(f"引擎意外终止: {e}")
                return None
            except chess.engine.EngineError as e:
                logging.error(f"引擎错误: {e}")
                return None
            except Exception as e:
                logging.error(f"引擎分析内部错误: {e}")
                return None
            finally:
                # 使用引擎管理器安全关闭引擎
                if engine:
                    global_engine_manager.close_engine(engine)

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
        
        # 同时取消所有任务
        task_cancelled_count = global_task_manager.force_clear_all_tasks()
        
        # 关闭所有活跃的引擎实例
        global_engine_manager.close_all_engines()
        
        logging.info(f"🧹 批量取消 {cancelled_count} 个分析会话和 {task_cancelled_count} 个任务: {session_ids[:5]}{'...' if len(session_ids) > 5 else ''}")
        
        return {
            "status": "success",
            "message": f"已取消 {cancelled_count} 个分析会话和 {task_cancelled_count} 个任务",
            "cancelled_count": cancelled_count + task_cancelled_count,
            "cancelled_sessions": session_ids[:10],  # 只返回前10个，避免响应过大
            "cancelled_tasks": task_cancelled_count
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
        
        # 检查缓存
        cache_key_params = {
            "include_rules": include_rules,
            "include_material": include_material, 
            "include_wdl": include_wdl,
            "include_model": include_model
        }
        
        cached_result = global_analysis_cache.get(fen, "stockfish", **cache_key_params)
        if cached_result:
            logging.info(f"💰 使用缓存结果: {fen[:30]}...")
            return cached_result
        
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
        
        # 缓存结果
        if stockfish_analysis.get('status') == 'success':
            global_analysis_cache.set(fen, "stockfish", stockfish_analysis, **cache_key_params)
            logging.info(f"💾 分析结果已缓存: {fen[:30]}...")
        
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
        
        # 参数验证
        if max_moves <= 0 or max_moves > 20:
            logging.error(f"❌ 无效的max_moves参数: {max_moves}")
            return Response(content="max_moves必须在1-20之间", status_code=400)
        
        if temperature <= 0 or temperature > 5.0:
            logging.error(f"❌ 无效的temperature参数: {temperature}")
            return Response(content="temperature必须在0-5.0之间", status_code=400)
        
        # 生成会话ID用于跟踪
        session_id = f"selfplay_{int(time.time())}_{hash(fen) % 10000}"
        logging.info(f"🎮 [会话:{session_id}] 开始Self-play分析")

        # 创建任务并执行self-play
        task_id = global_task_manager.create_task(
            "selfplay", 
            priority=1,  # 高优先级
            fen=fen, 
            max_moves=max_moves, 
            temperature=temperature
        )
        
        def run_selfplay():
            try:
                # 检查FEN格式
                try:
                    chess.Board(fen)
                except Exception as e:
                    logging.error(f"❌ [任务:{task_id}] 无效的FEN格式: {e}")
                    return {"error": f"无效的FEN格式: {str(e)}"}
                
                start_time = time.time()
                logging.info(f"🎮 [任务:{task_id}] 开始Self-play分析: {fen[:30]}...")
                
                # 使用任务ID创建引擎，支持取消
                selfplay_engine = HookedSelfPlayEngine(
                    global_chess_model, 
                    temperature=temperature,
                    task_id=task_id
                )
                
                # 优化超时设置：基础120秒 + 每步8秒
                dynamic_timeout = max(20.0, max_moves * 8.0)
                logging.info(f"⏰ [任务:{task_id}] 设置超时时间: {dynamic_timeout}秒 (最大步数: {max_moves})")
                
                selfplay_result = selfplay_engine.play_game(
                    initial_fen=fen, 
                    max_moves=max_moves, 
                    timeout=dynamic_timeout
                )
                
                elapsed_time = time.time() - start_time
                logging.info(f"✅ [任务:{task_id}] Self-play完成: {selfplay_result['total_moves']} 步, 耗时: {elapsed_time:.2f}秒")
                
                # 标记任务完成
                global_task_manager.complete_task(task_id)
                
                selfplay_result['session_id'] = session_id
                selfplay_result['task_id'] = task_id
                selfplay_result['processing_time'] = elapsed_time
                return selfplay_result
                
            except Exception as e:
                error_msg = str(e)
                if "任务已被取消" in error_msg:
                    logging.info(f"⚡ [任务:{task_id}] Self-play任务被用户取消")
                    return {"error": "任务被取消", "task_id": task_id, "cancelled": True}
                else:
                    logging.error(f"💥 [任务:{task_id}] Self-play分析失败: {e}")
                    import traceback
                    logging.error(f"💥 [任务:{task_id}] 详细错误: {traceback.format_exc()}")
                    return {"error": f"Self-play分析失败: {error_msg}", "task_id": task_id}
        
        selfplay_result = await to_thread(run_selfplay)
        
        response_data = {
            "fen": fen,
            "selfplay": selfplay_result,
            "status": "success" if "error" not in selfplay_result else "error",
            "session_id": session_id,
            "timestamp": int(time.time()),
            "parameters": {
                "max_moves": max_moves,
                "temperature": temperature
            },
            "processing_time": selfplay_result.get('processing_time', 0) if "error" not in selfplay_result else None
        }
        
        logging.info(f"📤 [会话:{session_id}] Self-play API返回响应，状态: {response_data['status']}")
        logging.info(f"📤 [会话:{session_id}] Self-play结果: {selfplay_result}")
        return response_data
        
    except Exception as e:
        session_id = session_id if 'session_id' in locals() else 'unknown'
        logging.error(f"💥 [会话:{session_id}] Self-play API错误: {e}")
        import traceback
        logging.error(f"💥 详细错误堆栈: {traceback.format_exc()}")
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

@app.get("/health")
async def health_check():
    """系统健康检查"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": int(time.time()),
            "services": {
                "api": "healthy",
                "model": "healthy" if global_chess_model else "not_loaded",
                "transformer_lens": "available" if TRANSFORMER_LENS_AVAILABLE else "unavailable"
            },
            "memory": {
                "available": True  # 可以添加更详细的内存检查
            }
        }
        
        # 简单的模型健康检查
        if global_chess_model and TRANSFORMER_LENS_AVAILABLE:
            try:
                # 快速测试模型是否响应
                test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                board = chess.Board(test_fen)
                # 这里不实际运行推理，只检查模型是否可访问
                health_status["services"]["model"] = "healthy"
            except Exception as e:
                health_status["services"]["model"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": int(time.time())
        }


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
    
    def __init__(self, model, temperature: float = 1.0, task_id: str = None):
        """
        初始化self-play引擎
        
        Args:
            model: HookedTransformer模型
            temperature: 温度参数，控制随机性
            task_id: 任务ID，用于取消控制
            
        Note:
            引擎将在每步都执行完整的WDL分析以确保精度
        """
        self.model = model
        self.temperature = temperature
        self.task_id = task_id
        self.model.eval()
        
        # 记录游戏历史
        self.game_history = []
        self.move_probabilities = []
        
        # 性能优化设置
        self.max_inference_time = 8.0  # 单次推理最大时间
    
    def validate_move_legality(self, fen: str, uci_move: str) -> bool:
        """
        验证移动的合法性
        
        Args:
            fen: FEN字符串
            uci_move: UCI格式的移动字符串
            
        Returns:
            移动是否合法
        """
        try:
            chess_board = chess.Board(fen)
            legal_moves = list(chess_board.legal_moves)
            legal_uci_set = set(move.uci() for move in legal_moves)
            return uci_move in legal_uci_set
        except Exception as e:
            logging.warning(f"验证移动合法性时出错: {e}")
            return False
    
    def get_legal_moves_with_probabilities(self, fen: str) -> List[Tuple[str, float]]:
        """
        获取所有合法移动及其概率（修复版本）
        
        Args:
            fen: FEN字符串
            
        Returns:
            合法移动列表: [(uci_move, probability), ...]
        """
        try:
            # 运行模型推理
            output, cache = self.model.run_with_cache(fen, prepend_bos=False)
            policy_output = output[0]  # [1, 1858]
            
            # 获取所有合法移动
            chess_board = chess.Board(fen)
            legal_moves = list(chess_board.legal_moves)
            legal_uci_set = set(move.uci() for move in legal_moves)
            
            # 计算概率
            probabilities = torch.softmax(policy_output / self.temperature, dim=-1)[0]
            
            # 获取映射表
            mapping_idx = get_mapping_index(chess_board)
            idx_to_uci = idx_to_uci_mappings[mapping_idx]
            
            move_probs = []
            total_prob = 0.0
            
            for move in legal_moves:
                try:
                    # 转换为模型索引
                    uci_move = move.uci()
                    if uci_move in idx_to_uci.values():
                        # 找到对应的索引
                        for idx, uci in idx_to_uci.items():
                            if uci == uci_move:
                                prob = probabilities[idx].item()
                                move_probs.append((uci_move, prob))
                                total_prob += prob
                                break
                    else:
                        # 如果转换失败，给一个很小的概率
                        move_probs.append((uci_move, 0.001))
                        total_prob += 0.001
                        
                except Exception as e:
                    logging.warning(f"处理移动 {move.uci()} 时出错: {e}")
                    # 如果转换失败，给一个很小的概率
                    move_probs.append((move.uci(), 0.001))
                    total_prob += 0.001
            
            # 重新归一化概率
            if total_prob > 0:
                move_probs = [(move, prob / total_prob) for move, prob in move_probs]
            
            # 按概率排序
            move_probs.sort(key=lambda x: x[1], reverse=True)
            
            return move_probs
            
        except Exception as e:
            logging.error(f"获取合法移动概率时出错: {e}")
            return []
    
    def get_best_move_simple(self, fen: str) -> Tuple[str, float]:
        """
        简单方法：获取最大概率的移动，如果合法就返回，否则找下一个
        
        Args:
            fen: FEN字符串
            
        Returns:
            (最佳移动, 概率)
        """
        try:
            # 运行模型推理
            output, cache = self.model.run_with_cache(fen, prepend_bos=False)
            policy_output = output[0]  # [1, 1858]
            
            # 获取所有合法移动
            chess_board = chess.Board(fen)
            legal_moves = list(chess_board.legal_moves)
            legal_uci_set = set(move.uci() for move in legal_moves)
            
            # 计算概率
            probabilities = torch.softmax(policy_output / self.temperature, dim=-1)[0]
            
            # 获取映射表
            mapping_idx = get_mapping_index(chess_board)
            idx_to_uci = idx_to_uci_mappings[mapping_idx]
            
            # 按概率排序所有可能的移动
            all_moves_with_probs = []
            for idx in range(len(policy_output[0])):
                if idx in idx_to_uci:
                    uci_move = idx_to_uci[idx]
                    prob = probabilities[idx].item()
                    all_moves_with_probs.append((uci_move, prob, idx))
            
            # 按概率排序
            all_moves_with_probs.sort(key=lambda x: x[1], reverse=True)
            
            # 找到第一个合法的移动
            for uci_move, prob, idx in all_moves_with_probs:
                if uci_move in legal_uci_set:
                    logging.debug(f"✅ 找到最佳合法移动: {uci_move} (概率:{prob:.4f}, 索引:{idx})")
                    return uci_move, prob
            
            # 如果没有找到合法移动，随机选择一个
            if legal_moves:
                random_move = np.random.choice(legal_moves)
                return random_move.uci(), 1.0 / len(legal_moves)
            else:
                raise ValueError("没有合法移动")
                
        except Exception as e:
            logging.error(f"获取最佳移动时出错: {e}")
            raise
    
    def get_best_move_with_validation(self, fen: str) -> Tuple[str, float, int]:
        """
        获取最佳移动并验证合法性（使用合法移动概率方法）
        
        Args:
            fen: FEN字符串
            
        Returns:
            (最佳移动, 概率, 索引)
        """
        try:
            # 使用新的合法移动概率方法
            legal_moves_with_probs = self.get_legal_moves_with_probabilities(fen)
            
            if not legal_moves_with_probs:
                raise ValueError("没有找到合法移动")
            
            # 返回概率最高的合法移动
            best_move, best_prob = legal_moves_with_probs[0]
            
            # 获取对应的索引（用于调试）
            chess_board = chess.Board(fen)
            mapping_idx = get_mapping_index(chess_board)
            idx_to_uci = idx_to_uci_mappings[mapping_idx]
            
            best_idx = None
            for idx, uci in idx_to_uci.items():
                if uci == best_move:
                    best_idx = idx
                    break
            
            logging.debug(f"✅ 最佳合法移动: {best_move} (概率:{best_prob:.4f}, 索引:{best_idx})")
            return best_move, best_prob, best_idx or 0
                
        except Exception as e:
            logging.error(f"获取最佳移动时出错: {e}")
            raise
    
    def get_top_moves(self, fen: str, top_k: int = 5) -> List[Tuple[str, float, int]]:
        """
        获取前k个最佳移动（带超时和取消检查）
        
        Args:
            fen: FEN字符串
            top_k: 返回前k个移动
            
        Returns:
            移动列表: [(uci_move, probability, index), ...]
        """
        try:
            # 检查任务是否被取消
            if self.task_id and global_task_manager.is_cancelled(self.task_id):
                logging.info(f"⚡ [任务:{self.task_id}] 推演被取消，停止执行")
                raise RuntimeError("任务已被取消")
            
            # 运行模型推理（带超时）
            inference_start = time.time()
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
        选择最佳移动（使用简单验证方法）
        
        Args:
            fen: FEN字符串
            
        Returns:
            (选择的移动, 移动概率)
        """
        try:
            # 使用简单方法：获取最大概率的移动，如果合法就返回
            best_move, best_prob = self.get_best_move_simple(fen)
            return best_move, best_prob
            
        except Exception as e:
            logging.warning(f"简单方法失败，回退到传统方法: {e}")
            
            # 回退到传统方法
            top_moves = self.get_top_moves(fen, top_k=5)
        
        if not top_moves:
            # 如果没有合法移动，随机选择一个
            chess_board = chess.Board(fen)
            legal_moves = list(chess_board.legal_moves)
            if legal_moves:
                move = np.random.choice(legal_moves)
                return move.uci(), 1.0 / len(legal_moves)
            else:
                raise ValueError("没有合法移动")
        
        # 始终选择概率最高的走法（第一名）
        best_move, best_prob, best_idx = top_moves[0]
        return best_move, best_prob
    
    def get_wdl_analysis(self, fen: str) -> Dict:
        """
        获取当前局面的WDL分析
        
        Args:
            fen: FEN字符串
            
        Returns:
            WDL分析结果（始终以白方视角显示）
        """
        try:
            # 检查任务是否被取消
            if self.task_id and global_task_manager.is_cancelled(self.task_id):
                logging.info(f"⚡ [任务:{self.task_id}] WDL分析被取消")
                raise RuntimeError("任务已被取消")
                
            with torch.no_grad():
                outputs, cache = self.model.run_with_cache(fen, prepend_bos=False)
            
            # 尝试不同的方式获取WDL输出
            wdl_probs = None
            if len(outputs) > 1:
                # 方式1：尝试outputs[1][0]
                try:
                    wdl_probs = outputs[1][0]
                    if hasattr(wdl_probs, 'shape') and len(wdl_probs.shape) > 0:
                        logging.debug(f"✓ 方式1成功: WDL shape={wdl_probs.shape}")
                    else:
                        logging.debug(f"✓ 方式1成功: WDL type={type(wdl_probs)}")
                except Exception as e:
                    logging.debug(f"✗ 方式1失败: {e}")
                    
                    # 方式2：尝试outputs[1]
                    try:
                        wdl_candidate = outputs[1]
                        if hasattr(wdl_candidate, 'shape') and len(wdl_candidate.shape) >= 2:
                            wdl_probs = wdl_candidate[0]
                        elif hasattr(wdl_candidate, 'shape') and len(wdl_candidate.shape) == 1:
                            wdl_probs = wdl_candidate
                        else:
                            wdl_probs = wdl_candidate
                        logging.debug(f"✓ 方式2成功: WDL shape={wdl_probs.shape if hasattr(wdl_probs, 'shape') else type(wdl_probs)}")
                    except Exception as e2:
                        logging.debug(f"✗ 方式2失败: {e2}")
                        
                        # 方式3：尝试其他可能的输出位置
                        for i in range(len(outputs)):
                            try:
                                candidate = outputs[i]
                                if hasattr(candidate, 'shape') and len(candidate.shape) >= 1:
                                    candidate_size = candidate.shape[-1] if len(candidate.shape) > 0 else 0
                                    if candidate_size == 3:  # WDL应该是3个值
                                        if len(candidate.shape) == 2:
                                            wdl_probs = candidate[0]
                                        else:
                                            wdl_probs = candidate
                                        logging.debug(f"✓ 方式3成功在outputs[{i}]: shape={wdl_probs.shape}")
                                        break
                            except Exception:
                                continue
            
            if wdl_probs is not None:
                # 确保WDL概率是有效的
                try:
                    # 尝试获取3个WDL值
                    values = None
                    if hasattr(wdl_probs, 'shape'):
                        if len(wdl_probs.shape) == 1 and wdl_probs.shape[0] >= 3:
                            # 一维张量，直接取前3个值
                            values = [wdl_probs[i] for i in range(3)]
                        elif len(wdl_probs.shape) == 0:
                            # 标量，可能需要其他处理
                            logging.warning(f"WDL输出是标量: {wdl_probs}")
                        elif wdl_probs.shape[-1] >= 3:
                            # 多维张量，取最后一维的前3个值
                            if len(wdl_probs.shape) == 2:
                                values = [wdl_probs[0][i] for i in range(3)]
                            else:
                                values = [wdl_probs[..., i] for i in range(3)]
                    elif hasattr(wdl_probs, '__len__') and len(wdl_probs) >= 3:
                        # 列表或类似序列
                        values = wdl_probs[:3]
                    
                    if values is not None:
                        # 转换为浮点数
                        try:
                            win_prob = float(values[0].item() if hasattr(values[0], 'item') else values[0])
                            draw_prob = float(values[1].item() if hasattr(values[1], 'item') else values[1])
                            loss_prob = float(values[2].item() if hasattr(values[2], 'item') else values[2])
                        except Exception:
                            # 如果item()失败，尝试直接转换
                            win_prob = float(values[0])
                            draw_prob = float(values[1])
                            loss_prob = float(values[2])
                        
                        # 检查概率是否合理
                        total_prob = win_prob + draw_prob + loss_prob
                        if total_prob > 0:
                            # 归一化概率
                            win_prob /= total_prob
                            draw_prob /= total_prob
                            loss_prob /= total_prob
                                                
                            # 解析FEN以确定当前行棋方
                            fen_parts = fen.split()
                            current_player = fen_parts[1] if len(fen_parts) > 1 else 'w'
                            
                            # 模型输出的WDL是相对于当前行棋方的
                            if current_player == 'w':  # 白方行棋
                                white_win_prob = win_prob
                                black_win_prob = loss_prob
                            else:  # 黑方行棋
                                white_win_prob = loss_prob
                                black_win_prob = win_prob
                            
                            logging.debug(f"✓ WDL解析成功: W{white_win_prob:.3f} D{draw_prob:.3f} B{black_win_prob:.3f}")
                            return {
                                "white_win_prob": round(white_win_prob * 100, 2),
                                "draw_prob": round(draw_prob * 100, 2),
                                "black_win_prob": round(black_win_prob * 100, 2),
                                "current_player": current_player,
                                "evaluation_type": "model_wdl"
                            }
                        else:
                            logging.warning(f"WDL概率总和为0: [{win_prob}, {draw_prob}, {loss_prob}]")
                    else:
                        wdl_shape = wdl_probs.shape if hasattr(wdl_probs, 'shape') else "unknown"
                        logging.warning(f"无法从WDL输出提取3个值: shape={wdl_shape}, type={type(wdl_probs)}")
                        
                except Exception as parse_error:
                    logging.error(f"解析WDL概率失败: {parse_error}")
                    import traceback
                    logging.debug(f"WDL解析错误详情: {traceback.format_exc()}")
            
            # 如果无法获取有效的WDL，使用合理的估计值
            logging.debug("无法获取有效的WDL输出，使用估计值")
            return {
                "white_win_prob": 45.0,
                "draw_prob": 10.0,
                "black_win_prob": 45.0,
                "evaluation_type": "estimated",
                "note": "模型WDL解析失败，使用估计值"
            }
                
        except Exception as e:
            logging.error(f"WDL分析失败: {e}")
            return {
                "white_win_prob": 33.0,
                "draw_prob": 34.0,
                "black_win_prob": 33.0,
                "evaluation_type": "error",
                "error": str(e)
            }
    
    def play_game(self, initial_fen: str = None, max_moves: int = 10, timeout: float = 120.0) -> Dict:
        """
        进行一局self-play游戏（带超时控制）
        
        Args:
            initial_fen: 初始FEN字符串，None表示标准开局
            max_moves: 最大移动数
            timeout: 超时时间（秒），默认120秒
            
        Returns:
            游戏结果字典
        """
        start_time = time.time()
        
        def check_timeout():
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Self-play超时 ({timeout}秒)")
            
        check_timeout()  # 初始检查
        
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
        wdl_history = []  # 记录WDL变化历史
        
        # 分析初始局面的WDL
        initial_wdl = self.get_wdl_analysis(board.fen())
        wdl_history.append({
            'move_number': 0,
            'fen': board.fen(),
            'wdl': initial_wdl
        })
        
        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            check_timeout()  # 每步检查超时
            
            # 检查任务是否被取消
            if self.task_id and global_task_manager.is_cancelled(self.task_id):
                logging.info(f"⚡ Self-play任务被取消: {self.task_id}")
                break
            
            move_count += 1
            current_player = "white" if board.turn == chess.WHITE else "black"
            current_fen = board.fen()
            
            try:
                # 获取前5个最佳移动（耗时操作，需要检查取消）
                if self.task_id and global_task_manager.is_cancelled(self.task_id):
                    logging.info(f"⚡ [任务:{self.task_id}] 在获取最佳移动前被取消")
                    break
                top_moves = self.get_top_moves(current_fen, top_k=5)
                
                # 选择移动（再次检查取消）
                if self.task_id and global_task_manager.is_cancelled(self.task_id):
                    logging.info(f"⚡ [任务:{self.task_id}] 在选择移动前被取消")
                    break
                chosen_move, chosen_prob = self.select_move(current_fen)
                
                # 执行移动
                move_obj = chess.Move.from_uci(chosen_move)
                move_san = board.san(move_obj)
                board.push(move_obj)
                
                # WDL分析前检查取消（WDL分析比较耗时）
                if self.task_id and global_task_manager.is_cancelled(self.task_id):
                    logging.info(f"⚡ [任务:{self.task_id}] 在WDL分析前被取消")
                    break
                    
                # 每步都执行完整的WDL分析
                logging.debug(f"🎯 步骤{move_count}: 执行WDL分析")
                after_move_wdl = self.get_wdl_analysis(board.fen())
                logging.debug(f"🎯 步骤{move_count}: WDL结果: {after_move_wdl.get('evaluation_type', 'unknown')}")
                
                # 记录游戏历史（包含WDL信息）
                self.game_history.append({
                    'move_number': move_count,
                    'player': current_player,
                    'move': chosen_move,
                    'probability': chosen_prob,
                    'fen_before': current_fen,
                    'fen_after': board.fen(),
                    'top_moves': top_moves,
                    'move_san': move_san,
                    'wdl_before': wdl_history[-1]['wdl'] if wdl_history else None,  # 移动前的WDL
                    'wdl_after': after_move_wdl  # 移动后的WDL
                })
                self.move_probabilities.append(chosen_prob)
                
                # 记录WDL历史
                wdl_history.append({
                    'move_number': move_count,
                    'fen': board.fen(),
                    'wdl': after_move_wdl
                })
                
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
            'wdl_history': wdl_history,  # 新增：WDL变化历史
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

@app.post("/analyze/selfplay/branch")
async def analyze_selfplay_branch_api(request_data: dict):
    """从指定步骤使用指定走法重新推演
    
    Args:
        request_data: 包含以下参数的请求数据
            - initial_fen: 初始FEN字符串
            - game_history: 之前的推演历史
            - branch_step: 要分支的步骤编号（从1开始）
            - selected_move: 选择的走法（UCI格式）
            - max_moves: 最大推演步数（默认10）
            - temperature: 温度参数（默认1.0）
        
    Returns:
        新的推演结果
    """
    try:
        initial_fen = request_data.get("initial_fen")
        game_history = request_data.get("game_history", [])
        branch_step = request_data.get("branch_step")
        selected_move = request_data.get("selected_move")
        max_moves = request_data.get("max_moves", 10)
        temperature = request_data.get("temperature", 1.0)
        
        logging.info(f"📥 收到分支推演请求: branch_step={branch_step}, selected_move={selected_move}")
        
        if not initial_fen:
            logging.error("❌ 缺少initial_fen参数")
            return Response(content="缺少initial_fen参数", status_code=400)
        
        if branch_step is None:
            logging.error("❌ 缺少branch_step参数")
            return Response(content="缺少branch_step参数", status_code=400)
        
        if not selected_move:
            logging.error("❌ 缺少selected_move参数")
            return Response(content="缺少selected_move参数", status_code=400)
        
        if not global_chess_model or not TRANSFORMER_LENS_AVAILABLE:
            return Response(content="模型未加载或不可用", status_code=503)
        
        def run_branch_selfplay():
            try:
                logging.info(f"🎯 开始分支推演: 从第{branch_step}步使用走法{selected_move}")
                
                # 重建到分支点的棋盘状态
                board = chess.Board(initial_fen)
                
                # 重播到分支点之前的所有走法
                for i in range(branch_step - 1):
                    if i < len(game_history):
                        move_uci = game_history[i]["move"]
                        try:
                            move = chess.Move.from_uci(move_uci)
                            board.push(move)
                        except Exception as e:
                            logging.error(f"重播走法失败 {move_uci}: {e}")
                            return {"error": f"重播走法失败: {str(e)}"}
                
                # 验证选择的走法是否合法
                try:
                    selected_move_obj = chess.Move.from_uci(selected_move)
                    if selected_move_obj not in board.legal_moves:
                        return {"error": f"选择的走法 {selected_move} 不合法"}
                except Exception as e:
                    return {"error": f"无效的走法格式 {selected_move}: {str(e)}"}
                
                # 执行选择的走法
                move_san = board.san(selected_move_obj)
                board.push(selected_move_obj)
                branch_fen = board.fen()
                
                # 计算剩余可推演步数（总共最多10步）
                remaining_moves = max(0, 10 - branch_step)
                
                # 从新位置开始self-play
                selfplay_engine = HookedSelfPlayEngine(global_chess_model, temperature=temperature)
                selfplay_result = selfplay_engine.play_game(initial_fen=branch_fen, max_moves=remaining_moves, timeout=20.0)
                
                # 获取分支点之前的WDL历史
                wdl_history = []
                temp_board = chess.Board(initial_fen)
                
                # 添加初始WDL
                initial_wdl = selfplay_engine.get_wdl_analysis(initial_fen)
                wdl_history.append({
                    'move_number': 0,
                    'fen': initial_fen,
                    'wdl': initial_wdl
                })
                
                # 构建完整的新游戏历史
                # 1. 保留分支点之前的历史，但需要重新计算WDL
                new_game_history = []
                for i in range(branch_step - 1):
                    if i < len(game_history):
                        step = game_history[i].copy()
                        # 重新计算WDL
                        move = chess.Move.from_uci(step["move"])
                        before_wdl = selfplay_engine.get_wdl_analysis(temp_board.fen())
                        temp_board.push(move)
                        after_wdl = selfplay_engine.get_wdl_analysis(temp_board.fen())
                        
                        step['wdl_before'] = before_wdl
                        step['wdl_after'] = after_wdl
                        new_game_history.append(step)
                        
                        # 添加到WDL历史
                        wdl_history.append({
                            'move_number': i + 1,
                            'fen': temp_board.fen(),
                            'wdl': after_wdl
                        })
                
                # 2. 获取分支点原始步骤的top_moves（保持原有候选走法显示）
                original_branch_step = game_history[branch_step - 1] if branch_step <= len(game_history) else None
                original_top_moves = original_branch_step.get('top_moves', []) if original_branch_step else []
                
                # 计算选择走法在原候选中的概率
                selected_move_prob = 1.0
                for move_info in original_top_moves:
                    if move_info[0] == selected_move:
                        selected_move_prob = move_info[1]
                        break
                
                # 3. 添加分支点的选择走法
                before_move_wdl = selfplay_engine.get_wdl_analysis(temp_board.fen())
                after_move_wdl = selfplay_engine.get_wdl_analysis(branch_fen)
                
                branch_step_info = {
                    'move_number': branch_step,
                    'player': "white" if temp_board.turn else "black",
                    'move': selected_move,
                    'probability': selected_move_prob,
                    'fen_before': temp_board.fen(),
                    'fen_after': branch_fen,
                    'move_san': move_san,
                    'top_moves': original_top_moves,  # 保持原有候选走法
                    'wdl_before': before_move_wdl,
                    'wdl_after': after_move_wdl,
                    'is_branch_point': True  # 标记这是分支点
                }
                
                new_game_history.append(branch_step_info)
                
                # 添加分支点WDL到历史
                wdl_history.append({
                    'move_number': branch_step,
                    'fen': branch_fen,
                    'wdl': after_move_wdl
                })
                
                # 4. 添加从新位置开始的self-play历史
                if selfplay_result.get('game_history'):
                    for i, step in enumerate(selfplay_result['game_history']):
                        # 调整步骤编号
                        adjusted_step = step.copy()
                        adjusted_step['move_number'] = branch_step + 1 + i
                        new_game_history.append(adjusted_step)
                
                # 5. 合并WDL历史（包含新推演的WDL）
                if selfplay_result.get('wdl_history'):
                    for wdl_step in selfplay_result['wdl_history']:
                        if wdl_step['move_number'] > 0:  # 跳过初始状态，避免重复
                            adjusted_wdl_step = wdl_step.copy()
                            adjusted_wdl_step['move_number'] = branch_step + adjusted_wdl_step['move_number']
                            wdl_history.append(adjusted_wdl_step)
                
                # 构建分支推演结果
                branch_result = {
                    'initial_fen': initial_fen,
                    'branch_step': branch_step,
                    'selected_move': selected_move,
                    'final_fen': selfplay_result.get('final_fen', branch_fen),
                    'result': selfplay_result.get('result', '*'),
                    'total_moves': len(new_game_history),
                    'termination': selfplay_result.get('termination'),
                    'game_history': new_game_history,
                    'move_probabilities': [selected_move_prob] + selfplay_result.get('move_probabilities', []),
                    'wdl_history': wdl_history,  # 包含完整的WDL历史
                    'temperature': temperature,
                    'is_branch': True,
                    'original_total_moves': len(game_history),
                    'branch_info': {
                        'branch_step': branch_step,
                        'selected_move': selected_move,
                        'selected_move_san': move_san,
                        'new_moves_count': len(selfplay_result.get('game_history', [])),
                        'remaining_moves_used': remaining_moves
                    }
                }
                
                logging.info(f"🎯 分支推演完成: 从第{branch_step}步分支，新增{len(selfplay_result.get('game_history', []))}步")
                return branch_result
                
            except Exception as e:
                logging.error(f"分支推演失败: {e}")
                return {"error": f"分支推演失败: {str(e)}"}
        
        branch_result = await to_thread(run_branch_selfplay)
        
        response_data = {
            "initial_fen": initial_fen,
            "branch": branch_result,
            "status": "success" if "error" not in branch_result else "error",
            "parameters": {
                "branch_step": branch_step,
                "selected_move": selected_move,
                "max_moves": max_moves,
                "temperature": temperature
            }
        }
        
        logging.info(f"📤 分支推演API返回响应")
        return response_data
        
    except Exception as e:
        logging.error(f"💥 分支推演API错误: {e}")
        return Response(
            content=f"分支推演出错: {str(e)}", 
            status_code=500
        )

@app.post("/analyze/batch")
async def analyze_batch(request_data: dict):
    """批量分析多个FEN局面
    
    Args:
        request_data: 包含FEN列表和分析选项的请求数据
        {
            "fens": ["fen1", "fen2", ...],
            "analysis_types": ["stockfish", "selfplay"],  // 可选的分析类型
            "include_rules": True,
            "include_material": True,
            "include_wdl": True,
            "include_model": True,
            "max_moves": 10,  // 仅用于selfplay
            "temperature": 1.0  // 仅用于selfplay
        }
        
    Returns:
        批量分析结果
    """
    try:
        fens = request_data.get("fens", [])
        analysis_types = request_data.get("analysis_types", ["stockfish"])
        include_rules = request_data.get("include_rules", True)
        include_material = request_data.get("include_material", True)
        include_wdl = request_data.get("include_wdl", True)
        include_model = request_data.get("include_model", True)
        max_moves = request_data.get("max_moves", 10)
        temperature = request_data.get("temperature", 1.0)
        
        logging.info(f"📥 收到批量分析请求: {len(fens)} 个局面, 类型: {analysis_types}")
        
        if not fens:
            logging.error("❌ 缺少FEN列表")
            return Response(content="缺少FEN列表", status_code=400)
        
        results = {}
        
        # 处理每个FEN
        for i, fen in enumerate(fens):
            if not fen:
                continue
                
            logging.info(f"🔄 处理第 {i+1}/{len(fens)} 个局面: {fen[:30]}...")
            fen_results = {}
            
            # Stockfish分析
            if "stockfish" in analysis_types:
                cache_key_params = {
                    "include_rules": include_rules,
                    "include_material": include_material,
                    "include_wdl": include_wdl,
                    "include_model": include_model
                }
                
                # 检查缓存
                cached_result = global_analysis_cache.get(fen, "stockfish", **cache_key_params)
                if cached_result:
                    logging.info(f"💰 使用缓存的Stockfish结果: {fen[:30]}...")
                    fen_results["stockfish"] = cached_result
                else:
                    # 执行分析
                    stockfish_request = {
                        "fen": fen,
                        "include_rules": include_rules,
                        "include_material": include_material,
                        "include_wdl": include_wdl,
                        "include_model": include_model
                    }
                    
                    stockfish_result = await analyze_stockfish_simple(stockfish_request)
                    fen_results["stockfish"] = stockfish_result
            
            # Self-play分析
            if "selfplay" in analysis_types:
                cache_key_params = {
                    "max_moves": max_moves,
                    "temperature": temperature
                }
                
                # 检查缓存
                cached_result = global_analysis_cache.get(fen, "selfplay", **cache_key_params)
                if cached_result:
                    logging.info(f"💰 使用缓存的Self-play结果: {fen[:30]}...")
                    fen_results["selfplay"] = cached_result
                else:
                    # 执行self-play分析
                    selfplay_request = {
                        "fen": fen,
                        "max_moves": max_moves,
                        "temperature": temperature
                    }
                    
                    # 为了避免超时，我们需要调用实际的self-play函数
                    try:
                        session_id = f"batch_{int(time.time())}_{i}"
                        
                        def run_selfplay():
                            try:
                                # 验证FEN格式
                                chess.Board(fen)
                            except Exception as e:
                                logging.error(f"❌ [批量:{session_id}] 无效的FEN格式: {e}")
                                return {"error": f"无效的FEN格式: {str(e)}"}
                            
                            start_time = time.time()
                            logging.info(f"🎮 [批量:{session_id}] 开始Self-play分析: {fen[:30]}...")
                            
                            selfplay_engine = HookedSelfPlayEngine(global_chess_model, temperature=temperature)
                            dynamic_timeout = max(20.0, max_moves * 15.0)
                            selfplay_result = selfplay_engine.play_game(initial_fen=fen, max_moves=max_moves, timeout=dynamic_timeout)
                            
                            elapsed_time = time.time() - start_time
                            logging.info(f"✅ [批量:{session_id}] Self-play完成: {selfplay_result['total_moves']} 步, 耗时: {elapsed_time:.2f}秒")
                            
                            selfplay_result['session_id'] = session_id
                            selfplay_result['processing_time'] = elapsed_time
                            return selfplay_result
                        
                        selfplay_result = await to_thread(run_selfplay)
                        
                        # 缓存结果
                        if "error" not in selfplay_result:
                            global_analysis_cache.set(fen, "selfplay", selfplay_result, **cache_key_params)
                        
                        fen_results["selfplay"] = selfplay_result
                        
                    except Exception as e:
                        logging.error(f"💥 [批量:{session_id}] Self-play分析失败: {e}")
                        fen_results["selfplay"] = {"error": f"Self-play分析失败: {str(e)}"}
            
            results[fen] = fen_results
        
        response_data = {
            "status": "success",
            "total_fens": len(fens),
            "processed_fens": len(results),
            "results": results,
            "timestamp": int(time.time()),
            "analysis_types": analysis_types
        }
        
        logging.info(f"📤 批量分析完成: {len(results)}/{len(fens)} 个局面已处理")
        return response_data
        
    except Exception as e:
        logging.error(f"💥 批量分析API错误: {e}")
        import traceback
        logging.error(f"💥 详细错误堆栈: {traceback.format_exc()}")
        return Response(
            content=f"批量分析出错: {str(e)}",
            status_code=500
        )


@app.post("/analyze/unified")
async def analyze_unified(request_data: dict):
    """统一分析接口，智能选择分析类型
    
    Args:
        request_data: 包含FEN和分析偏好的请求数据
        {
            "fen": "fen_string",
            "preferred_type": "auto|stockfish|selfplay",  // 首选分析类型
            "fallback": True,  // 是否启用备用分析
            "include_rules": True,
            "include_material": True,
            "include_wdl": True,
            "include_model": True,
            "max_moves": 10,
            "temperature": 1.0
        }
        
    Returns:
        统一分析结果
    """
    try:
        fen = request_data.get("fen")
        preferred_type = request_data.get("preferred_type", "auto")
        fallback = request_data.get("fallback", True)
        include_rules = request_data.get("include_rules", True)
        include_material = request_data.get("include_material", True)
        include_wdl = request_data.get("include_wdl", True)
        include_model = request_data.get("include_model", True)
        max_moves = request_data.get("max_moves", 10)
        temperature = request_data.get("temperature", 1.0)
        
        logging.info(f"📥 收到统一分析请求: {fen[:30] if fen else 'None'}..., 类型: {preferred_type}")
        
        if not fen:
            logging.error("❌ 缺少FEN参数")
            return Response(content="缺少FEN参数", status_code=400)
        
        result = {
            "fen": fen,
            "preferred_type": preferred_type,
            "timestamp": int(time.time())
        }
        
        # 智能选择分析类型
        if preferred_type == "auto":
            # 自动模式：优先使用缓存中的结果
            stockfish_cache = global_analysis_cache.get(fen, "stockfish", 
                include_rules=include_rules, include_material=include_material, 
                include_wdl=include_wdl, include_model=include_model)
            
            selfplay_cache = global_analysis_cache.get(fen, "selfplay",
                max_moves=max_moves, temperature=temperature)
            
            if stockfish_cache:
                result["analysis"] = stockfish_cache
                result["analysis_type"] = "stockfish"
                result["cache_hit"] = True
                logging.info(f"💰 使用缓存的Stockfish结果: {fen[:30]}...")
            elif selfplay_cache:
                result["analysis"] = selfplay_cache
                result["analysis_type"] = "selfplay"
                result["cache_hit"] = True
                logging.info(f"💰 使用缓存的Self-play结果: {fen[:30]}...")
            else:
                # 没有缓存，执行Stockfish分析（更快）
                preferred_type = "stockfish"
        
        # 执行指定类型的分析
        if preferred_type == "stockfish" and "analysis" not in result:
            stockfish_request = {
                "fen": fen,
                "include_rules": include_rules,
                "include_material": include_material,
                "include_wdl": include_wdl,
                "include_model": include_model
            }
            
            stockfish_result = await analyze_stockfish_simple(stockfish_request)
            result["analysis"] = stockfish_result
            result["analysis_type"] = "stockfish"
            result["cache_hit"] = False
            
        elif preferred_type == "selfplay" and "analysis" not in result:
            # 执行self-play分析（使用批量分析的逻辑）
            batch_request = {
                "fens": [fen],
                "analysis_types": ["selfplay"],
                "max_moves": max_moves,
                "temperature": temperature
            }
            
            batch_result = await analyze_batch(batch_request)
            if batch_result.get("status") == "success" and fen in batch_result.get("results", {}):
                result["analysis"] = batch_result["results"][fen].get("selfplay")
                result["analysis_type"] = "selfplay"
                result["cache_hit"] = False
            else:
                if fallback:
                    # 回退到Stockfish分析
                    stockfish_request = {
                        "fen": fen,
                        "include_rules": include_rules,
                        "include_material": include_material,
                        "include_wdl": include_wdl,
                        "include_model": include_model
                    }
                    
                    stockfish_result = await analyze_stockfish_simple(stockfish_request)
                    result["analysis"] = stockfish_result
                    result["analysis_type"] = "stockfish"
                    result["cache_hit"] = False
                    result["fallback_used"] = True
                    logging.info(f"⚠️ Self-play分析失败，回退到Stockfish: {fen[:30]}...")
                else:
                    result["error"] = "Self-play分析失败且未启用回退"
        
        result["status"] = "success" if "analysis" in result else "error"
        
        logging.info(f"📤 统一分析完成: 类型={result.get('analysis_type', 'none')}, 缓存命中={result.get('cache_hit', False)}")
        return result
        
    except Exception as e:
        logging.error(f"💥 统一分析API错误: {e}")
        import traceback
        logging.error(f"💥 详细错误堆栈: {traceback.format_exc()}")
        return Response(
            content=f"统一分析出错: {str(e)}",
            status_code=500
        )

@app.post("/tasks/cancel")
async def cancel_task_api(request_data: dict):
    """取消指定任务
    
    Args:
        request_data: 包含task_id的请求数据
        
    Returns:
        取消结果
    """
    try:
        task_id = request_data.get("task_id")
        pattern = request_data.get("pattern")  # 可选：按模式取消
        
        logging.info(f"🔍 [任务取消API] 收到取消请求: task_id={task_id}, pattern={pattern}")
        
        if task_id:
            success = global_task_manager.cancel_task(task_id)
            if success:
                logging.info(f"⚡ 任务取消成功: {task_id}")
                return {"status": "success", "message": f"任务 {task_id} 已取消"}
            else:
                return {"status": "not_found", "message": f"任务 {task_id} 不存在"}
        
        elif pattern:
            cancelled_count = global_task_manager.cancel_tasks_by_pattern(pattern)
            logging.info(f"⚡ 批量取消任务: {cancelled_count} 个任务包含模式 '{pattern}'")
            return {"status": "success", "message": f"已取消 {cancelled_count} 个任务", "cancelled_count": cancelled_count}
        
        else:
            return {"status": "error", "message": "需要提供 task_id 或 pattern 参数"}
            
    except Exception as e:
        logging.error(f"💥 取消任务API错误: {e}")
        return Response(content=f"取消任务出错: {str(e)}", status_code=500)


@app.get("/tasks/status")
async def get_tasks_status():
    """获取当前活跃任务状态
    
    Returns:
        活跃任务列表
    """
    try:
        active_tasks = global_task_manager.get_all_tasks()
        for task_info in active_tasks.values():
            task_info["elapsed_time"] = time.time() - task_info["created_at"]
        
        return {
            "status": "success",
            "active_tasks": active_tasks,
            "total_count": len(active_tasks),
            "timestamp": int(time.time())
        }
        
    except Exception as e:
        logging.error(f"💥 查询任务状态API错误: {e}")
        return Response(content=f"查询任务状态出错: {str(e)}", status_code=500)


@app.post("/tasks/priority")
async def set_task_priority(request_data: dict):
    """设置任务优先级（用于分支推演等高优先级任务）
    
    Args:
        request_data: 包含task_pattern和new_priority的请求数据
        
    Returns:
        设置结果
    """
    try:
        task_pattern = request_data.get("task_pattern")
        new_priority = request_data.get("new_priority", 10)  # 默认高优先级
        
        if not task_pattern:
            return {"status": "error", "message": "需要提供 task_pattern 参数"}
        
        # 取消低优先级的相关任务
        cancelled_count = global_task_manager.cancel_tasks_by_pattern(task_pattern)
        
        logging.info(f"🔥 高优先级任务请求: 取消了 {cancelled_count} 个相关任务，模式: '{task_pattern}'")
        
        return {
            "status": "success", 
            "message": f"已为高优先级任务清理了 {cancelled_count} 个相关任务",
            "cancelled_count": cancelled_count,
            "new_priority": new_priority
        }
        
    except Exception as e:
        logging.error(f"💥 设置任务优先级API错误: {e}")
        return Response(content=f"设置任务优先级出错: {str(e)}", status_code=500)


@app.post("/tasks/force_clear")
async def force_clear_all_tasks():
    """强制清理所有任务（紧急情况使用）
    
    Returns:
        清理结果
    """
    try:
        cleared_count = global_task_manager.force_clear_all_tasks()
        
        logging.warning(f"🚨 强制清理API: 清理了 {cleared_count} 个任务")
        
        return {
            "status": "success", 
            "message": f"强制清理了 {cleared_count} 个任务",
            "cleared_count": cleared_count
        }
        
    except Exception as e:
        logging.error(f"💥 强制清理任务API错误: {e}")
        return Response(content=f"强制清理任务出错: {str(e)}", status_code=500)

@app.post("/circuits/generate")
async def circuits_generate(request_data: dict):
    """根据FEN与可选的move生成电路归因图，并返回JSON（并可保存到/circuits）。

    请求参数：
      - fen: 必填，FEN字符串
      - move_uci: 选填，UCI走法；缺省时将自动推理
      - side: 选填，"k"|"q"|"both"，默认"k"
      - node_threshold: 选填，默认0.5
      - edge_threshold: 选填，默认0.3
      - sae_series: 选填，默认"lc0-tc"
      - save: 选填，是否保存到/circuits，默认True
    返回：
      { graph, saved_path, slug, fen, move_uci, inferred }
    """
    try:
        if not CIRCUITS_SERVICE_AVAILABLE:
            return Response(content="circuits_service 未就绪", status_code=503)

        fen = request_data.get("fen")
        move_uci = request_data.get("move_uci")
        side = request_data.get("side", "k")
        node_threshold = float(request_data.get("node_threshold", 0.5))
        edge_threshold = float(request_data.get("edge_threshold", 0.3))
        sae_series_req = request_data.get("sae_series") or os.environ.get("SAE_SERIES", "lc0-tc")
        save = bool(request_data.get("save", False))
        output_dir = request_data.get("output_dir")  # 可选自定义输出路径

        if not fen:
            return Response(content="缺少fen参数", status_code=400)

        # 若未提供move，尝试用模型推理，否则回退Stockfish
        inferred = False
        if not move_uci:
            inferred = True
            best_move = None
            # 优先使用自有模型
            try:
                if global_chess_model and TRANSFORMER_LENS_AVAILABLE:
                    engine = HookedSelfPlayEngine(global_chess_model, temperature=1.0)
                    best_move, _prob = engine.get_best_move_simple(fen)
            except Exception as e:
                logging.warning(f"模型推理best move失败，将回退Stockfish: {e}")
                best_move = None

            if not best_move:
                try:
                    sf_res = await analyze_position_with_stockfish_simple(fen)
                    if sf_res is not None:
                        best_move, _ponder = sf_res
                except Exception as e:
                    logging.error(f"回退Stockfish获取best move失败: {e}")

            if not best_move:
                return Response(content="无法推理到可用的best move，请手动提供move_uci", status_code=422)
            move_uci = best_move

        graph_json, saved_path, slug = await to_thread(
            run_trace_to_graph,
            fen=fen,
            move_uci=move_uci,
            side=side,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
            sae_series=sae_series_req,
            output_dir=output_dir if save else None,
            save=save,
        )

        if not graph_json:
            return Response(content="图生成失败", status_code=500)

        # 对返回的JSON做最小化适配（保证前端transformCircuitData可以识别）
        # 若已有nodes/edges则直接返回；否则透传
        resp = {
            "status": "success",
            "graph": graph_json,
            "saved_path": saved_path,
            "slug": slug,
            "fen": fen,
            "move_uci": move_uci,
            "inferred": inferred,
        }
        return resp

    except Exception as e:
        logging.error(f"💥 circuits_generate 错误: {e}")
        return Response(content=f"生成电路图出错: {str(e)}", status_code=500)

@app.get("/logs/recent")
async def get_recent_logs(limit: int = 200):
    try:
        limit = max(1, min(2000, int(limit)))
    except Exception:
        limit = 200
    return {"logs": log_handler.get_recent(limit)}

@app.get("/logs/stream")
async def stream_logs():
    async def event_gen():
        last_len = 0
        while True:
            await asyncio.sleep(0.5)
            logs = log_handler.get_recent(2000)
            if len(logs) > last_len:
                # 只发送新增部分
                for item in logs[last_len:]:
                    data = json.dumps(item)
                    yield f"data: {data}\n\n"
                last_len = len(logs)
    return StreamingResponse(event_gen(), media_type="text/event-stream")