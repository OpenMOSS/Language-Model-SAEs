import io
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, Optional
import threading
import time

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

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

client = MongoClient(MongoDBConfig())
sae_series = os.environ.get("SAE_SERIES", "default")

# 会话管理：跟踪活跃的分析会话
active_sessions = {}
session_lock = threading.Lock()

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

# Remove global caches in favor of LRU cache
# sae_cache: dict[str, SparseAutoEncoder] = {}
# lm_cache: dict[str, LanguageModel] = {}
# dataset_cache: dict[tuple[str, int, int], Dataset] = {}

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
ENGINE_TIME_LIMIT = 0.1  # 减少到0.1秒，更快响应
ENGINE_OPTIONS = {
    "Threads": 2,  # 减少线程数
    "Hash": 256,   # 减少哈希表大小
    "Skill Level": 15,  # 稍微降低技能等级以提高速度
}

# 内部缓存函数 - 只缓存FEN，不包含session_id
@lru_cache(maxsize=1000)
def _cached_stockfish_analysis(fen: str) -> Optional[tuple[str, Optional[str]]]:
    """
    内部缓存的Stockfish分析函数，只基于FEN进行缓存
    
    Args:
        fen: 棋局的FEN字符串
        
    Returns:
        tuple[str, Optional[str]]: (最佳走法, ponder走法)；如果分析失败则返回None
    """
    import time
    import random
    
    # 小延迟模拟计算时间
    delay = random.uniform(0.01, 0.1)  # 10-100ms随机延迟
    time.sleep(delay)
    
    if not fen or len(fen.strip()) == 0:
        logging.error("收到空的FEN字符串")
        return None
        
    try:
        # 验证FEN字符串
        board = chess.Board(fen)
        if board.is_game_over():
            logging.info(f"游戏已结束: {fen}")
            return None
            
        # 初始化引擎
        with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
            # 配置引擎参数
            for option, value in ENGINE_OPTIONS.items():
                engine.configure({option: value})
            
            # 分析局面
            try:
                result = engine.play(
                    board, 
                    chess.engine.Limit(
                        time=ENGINE_TIME_LIMIT,
                        depth=12,         # 适中的深度平衡速度和质量
                        nodes=800_000     # 适中的节点数
                    )
                )
                
                # 处理分析结果
                if result.move:
                    best_move = str(result.move)
                    ponder = str(result.ponder) if result.ponder else None
                    logging.info(f"🎯 缓存分析完成: {best_move} [FEN:{fen[:15]}...]")
                    return best_move, ponder
                else:
                    logging.warning(f"引擎未返回着法: {fen}")
                    return None
                    
            except chess.engine.EngineTerminatedError:
                logging.error(f"引擎意外终止: {fen}")
                return None
            except chess.engine.EngineError as e:
                logging.error(f"引擎错误: {e} (FEN: {fen})")
                return None
                
    except ValueError as e:
        logging.error(f"无效的FEN: {fen} - {str(e)}")
        return None
    except Exception as e:
        logging.error(f"分析异常 {fen}: {e}")
        return None
            
    return None


def analyze_position_with_stockfish(fen: str, session_id: Optional[str] = None) -> Optional[tuple[str, Optional[str]]]:
    """
    使用Stockfish分析单个位置，支持session管理和取消
    
    Args:
        fen: 棋局的FEN字符串
        session_id: 会话ID（用于取消控制和日志追踪）
        
    Returns:
        tuple[str, Optional[str]]: (最佳走法, ponder走法)；如果分析失败或被取消则返回None
        
    Note:
        - 分离缓存和session管理，确保取消检查始终生效
        - 内部使用缓存提高性能，外层处理会话控制
        - 包含详细的日志追踪
    """
    logging.info(f"⚡ 开始分析请求: {fen[:20]}... [会话:{session_id}]")
    
    # 第一次取消检查：分析开始前
    if session_id and is_session_cancelled(session_id):
        logging.info(f"🛑 分析开始前被取消 [会话:{session_id}]: {fen[:20]}...")
        return None
    
    try:
        # 第二次取消检查：调用缓存分析前
        if session_id and is_session_cancelled(session_id):
            logging.info(f"🛑 缓存分析前被取消 [会话:{session_id}]: {fen[:20]}...")
            return None
        
        logging.info(f"🔍 开始Stockfish计算 [会话:{session_id}]: {fen[:20]}...")
        
        # 调用内部缓存函数
        result = _cached_stockfish_analysis(fen)
        
        # 第三次取消检查：分析完成后
        if session_id and is_session_cancelled(session_id):
            logging.info(f"🛑 分析完成后被取消 [会话:{session_id}]: {fen[:20]}...")
            return None
        
        if result:
            best_move, ponder = result
            logging.info(f"🎯 分析成功返回: {best_move} [会话:{session_id}]")
            return result
        else:
            logging.info(f"❌ 分析无结果 [会话:{session_id}]: {fen[:20]}...")
            return None
            
    except Exception as e:
        logging.error(f"分析异常 [会话:{session_id}] {fen}: {e}")
        return None
    finally:
        # 清理会话（如果提供了session_id）
        if session_id:
            cleanup_session(session_id)
            logging.info(f"🧹 会话已清理 [会话:{session_id}]")


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
    
        print(f"{dataset_name=}")
        print(f"{model_name = }")
        model = get_model(model_name)
        data = get_dataset(dataset_name, shard_idx, n_shards)[context_idx]

        # Get origins for the features - 对于象棋模型，跳过耗时的trace调用以提高响应速度
        # if "chess" in model_name or "lc0" in model_name:
        #     # 为象棋数据创建简化的origins，避免耗时的模型trace
        #     if "text" in data and data["text"]:
        #         text_length = len(data["text"])
        #         origins = [{"key": "text", "range": [0, text_length]}] * text_length
        #     else:
        #         origins = []
        # else:
        #     # 对于其他模型，保持原有的trace行为
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
        # Add debug info before trimming
        print(f"Before trim - origins length: {len(origins) if origins else 'None'}")
        print(f"Before trim - feature_acts length: {len(feature_acts) if feature_acts is not None else 'None'}")
        print(f"Before trim - feature_acts type: {type(feature_acts)}")
        print(f"Before trim - feature_acts sample: {feature_acts[:5] if feature_acts and len(feature_acts) > 0 else 'Empty or None'}")
        
        # Trim to matching lengths
        origins, feature_acts = trim_minimum(origins, feature_acts)
        assert origins is not None and feature_acts is not None, "Origins and feature acts must not be None"
        
        # Add debug info after trimming
        print(f"After trim - origins length: {len(origins) if origins else 'None'}")
        print(f"After trim - feature_acts length: {len(feature_acts) if feature_acts is not None else 'None'}")
        print(f"After trim - feature_acts sample: {feature_acts[:5] if feature_acts and len(feature_acts) > 0 else 'Empty or None'}")

        # board_state = data["fen"]
        
        if "chess" in model_name or "lc0" in model_name:
            # 直接用data['fen']
            data['text'] = data["fen"]
            print(f"len(data['text']) = {len(data['text'])}")
            
            # 只标记为需要Stockfish分析，不在这里执行分析以提高响应速度
            if "fen" in data:
                data["needs_stockfish_analysis"] = True
                data["stockfish_analysis"] = {
                    "status": "pending",
                    "fen": data["fen"]
                }
            else:
                data["needs_stockfish_analysis"] = False
                data["stockfish_analysis"] = {
                    "status": "no_fen",
                    "error": "无FEN数据",
                    "fen": None
                }
        # Process text data if present
        if "text" in data and "lc0" not in model_name:
            text_ranges = [origin["range"] for origin in origins if origin is not None and origin["key"] == "text"]
            if text_ranges:
                max_text_origin = max(text_ranges, key=lambda x: x[1])
                data["text"] = data["text"][: max_text_origin[1]]
        
        print(f"Final check before return:")
        print(f"  - origins length: {len(origins) if origins else 'None'}")
        print(f"  - feature_acts length: {len(feature_acts) if feature_acts is not None else 'None'}")
        print(f"  - feature_acts is None: {feature_acts is None}")
        print(f"  - context_idx: {context_idx}")
        
        # Create return dictionary
        result = {**data, "origins": origins, "feature_acts": feature_acts, "context_idx": context_idx}
        
        print(f"Return dict keys: {list(result.keys())}")
        print(f"Return dict feature_acts: {result.get('feature_acts', 'NOT_FOUND')}")
        
        return result

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
async def analyze_stockfish(request_data: dict):
    """异步分析FEN局面的Stockfish最佳走法
    
    Args:
        request_data: 包含FEN字符串和session_id的请求数据
        
    Returns:
        Stockfish分析结果
    """
    try:
        fen = request_data.get("fen")
        session_id = request_data.get("session_id")
        
        logging.info(f"📥 收到分析请求 [会话:{session_id}] FEN: {fen[:20] if fen else 'None'}...")
        
        if not fen:
            logging.error(f"❌ 缺少FEN参数 [会话:{session_id}]")
            return Response(content="缺少FEN参数", status_code=400)
        
        # 如果提供了session_id，先检查是否已经被取消
        if session_id and is_session_cancelled(session_id):
            logging.info(f"🛑 API层检测到会话已被取消 [会话:{session_id}]: {fen[:20]}...")
            return {
                "best_move": None,
                "ponder": None,
                "status": "cancelled",
                "error": "分析被取消",
                "fen": fen,
                "session_id": session_id
            }
        
        # 如果提供了session_id，注册会话以支持取消功能
        cancel_event = None
        if session_id:
            cancel_event = threading.Event()
            register_session(session_id, cancel_event)
            logging.info(f"📝 注册分析会话: {session_id} for FEN: {fen[:20]}...")
        
        # 执行Stockfish分析（现在支持通过session_id取消）
        logging.info(f"🚀 开始调用Stockfish分析 [会话:{session_id}]: {fen[:20]}...")
        analysis_result = analyze_position_with_stockfish(fen, session_id)
        
        # 再次检查是否在分析过程中被取消
        if session_id and is_session_cancelled(session_id):
            logging.info(f"🛑 API层分析完成后检测到已取消 [会话:{session_id}]: {fen[:20]}...")
            return {
                "best_move": None,
                "ponder": None,
                "status": "cancelled",
                "error": "分析被取消",
                "fen": fen,
                "session_id": session_id
            }
        
        if analysis_result is not None:
            best_move, ponder = analysis_result
            logging.info(f"✅ API成功返回结果 [会话:{session_id}]: {best_move}")
            
            stockfish_analysis = {
                "best_move": best_move if best_move else None,
                "ponder": ponder if ponder else None,
                "status": "success" if best_move else "no_move",
                "error": None,
                "fen": fen,
                "session_id": session_id
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
            logging.info(f"❌ API分析无结果 [会话:{session_id}]: {fen[:20]}...")
            stockfish_analysis = {
                "best_move": None,
                "ponder": None,
                "status": "no_move", 
                "error": "分析失败",
                "fen": fen,
                "session_id": session_id
            }
            
        logging.info(f"📤 API返回响应 [会话:{session_id}]: status={stockfish_analysis['status']}")
        return stockfish_analysis
        
    except Exception as e:
        logging.error(f"💥 Stockfish分析API错误 [会话:{session_id if 'session_id' in locals() else 'unknown'}]: {e}")
        return Response(
            content=f"分析出错: {str(e)}", 
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
                analysis_result = analyze_position_with_stockfish(fen)
                
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