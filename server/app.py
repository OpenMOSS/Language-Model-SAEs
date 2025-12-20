# NEW HEADER
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io
from functools import lru_cache
from typing import Any, Optional, Tuple, List, Dict
from pathlib import Path

import msgpack
import numpy as np
import plotly.graph_objects as go
import torch
from datasets import Dataset
from fastapi import FastAPI, Response, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# 全局 BT4 常量（模型与 SAE 路径），兼容脚本运行和 package 导入
try:
    from .constants import (
        BT4_TC_BASE_PATH,
        BT4_LORSA_BASE_PATH,
        BT4_SAE_COMBOS,
        BT4_DEFAULT_SAE_COMBO,
        get_bt4_sae_combo,
    )
except ImportError:
    from constants import (
        BT4_TC_BASE_PATH,
        BT4_LORSA_BASE_PATH,
        BT4_SAE_COMBOS,
        BT4_DEFAULT_SAE_COMBO,
        get_bt4_sae_combo,
    )
try:
    from torchvision import transforms
except ImportError:
    transforms = None
    print("WARNING: torchvision not found, image processing will be disabled")

from lm_saes.backend import LanguageModel
from lm_saes.config import MongoDBConfig, SAEConfig
from lm_saes.database import MongoClient
from lm_saes.resource_loaders import load_dataset_shard, load_model
from lm_saes.sae import SparseAutoEncoder
import subprocess
import json
import tempfile
import os
import time

import random
import chess

# 添加HookedTransformer导入
try:
    from transformer_lens import HookedTransformer
    HOOKED_TRANSFORMER_AVAILABLE = True
except ImportError:
    HookedTransformer = None
    HOOKED_TRANSFORMER_AVAILABLE = False
    print("WARNING: transformer_lens not found, HookedTransformer will not be available")

from lm_saes.lc0_mapping.lc0_mapping import (
    idx_to_uci_mappings,
    get_mapping_index,
)
from lm_saes.circuit.leela_board import LeelaBoard
from move_evaluation import evaluate_move_quality  # 引入走法评测

# 导入战术特征分析
try:
    from tactic_features import analyze_tactic_features, validate_fens
    from lm_saes import LowRankSparseAttention
    TACTIC_FEATURES_AVAILABLE = True
except ImportError:
    analyze_tactic_features = None
    validate_fens = None
    LowRankSparseAttention = None
    TACTIC_FEATURES_AVAILABLE = False
    print("WARNING: tactic_features not found, tactic analysis will not be available")

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

SEARCH_TRACE_OUTPUT_DIR = Path("search_trace_outputs")

client = MongoClient(MongoDBConfig())
sae_series = os.environ.get("SAE_SERIES", "default")
tokenizer_only = os.environ.get("TOKENIZER_ONLY", "false").lower() == "true"
if tokenizer_only:
    print("WARNING: Tokenizer only mode is enabled, some features may not be available")

# Remove global caches in favor of LRU cache
# sae_cache: dict[str, SparseAutoEncoder] = {}
# lm_cache: dict[str, LanguageModel] = {}
# dataset_cache: dict[tuple[str, int, int], Dataset] = {}


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
    cfg.tokenizer_only = tokenizer_only
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


###############################################################################
# 全局模型 / SAE 组合缓存与加载状态
###############################################################################

# 当前使用的 BT4 SAE 组合 ID（例如 "k_64_e_32"），默认使用常量中的默认组合
CURRENT_BT4_SAE_COMBO_ID: str = BT4_DEFAULT_SAE_COMBO


def _make_combo_cache_key(model_name: str, combo_id: str | None) -> str:
    """为缓存/日志生成键：同一模型不同组合使用不同 key。"""

    if not combo_id:
        return model_name
    return f"{model_name}::{combo_id}"


# 添加全局模型缓存（先初始化本地缓存，circuits_service 导入后会更新）
_hooked_models: Dict[str, Any] = {}
_transcoders_cache: Dict[str, Dict[int, SparseAutoEncoder]] = {}
_lorsas_cache: Dict[str, Any] = {}  # combo_key -> List[LowRankSparseAttention]
_replacement_models_cache: Dict[str, Any] = {}  # combo_key -> ReplacementModel

# 添加全局加载日志缓存（用于前端显示）
_loading_logs: Dict[str, list] = {}  # combo_key -> [log1, log2, ...]

# 添加全局加载状态跟踪（用于避免重复加载）
import threading

# 全局锁：确保同一时间只能加载一个配置（避免GPU内存同时被多个配置占用）
_global_loading_lock = threading.Lock()

_loading_locks: Dict[str, threading.Lock] = {}  # combo_key -> Lock
_loading_status: Dict[str, dict] = {}  # combo_key -> {"is_loading": bool}
_cancel_loading: Dict[str, bool] = {}  # combo_key -> should_cancel (用于中断加载)

# Circuit tracing日志存储（用于前端显示）
_circuit_trace_logs: Dict[str, list] = {}  # trace_key -> [log1, log2, ...]
_circuit_trace_status: Dict[str, dict] = {}  # trace_key -> {"is_tracing": bool}
_circuit_trace_results: Dict[str, dict] = {}  # trace_key -> {"graph_data": ..., "finished_at": ts}

# Trace结果持久化目录
TRACE_RESULTS_DIR = Path(__file__).parent.parent / "circuit_trace_results"
TRACE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _save_trace_result_to_disk(trace_key: str, result: dict) -> None:
    """将trace结果保存到磁盘"""
    try:
        # 使用安全的文件名（替换特殊字符）
        safe_key = trace_key.replace("::", "_").replace("/", "_").replace(" ", "_")
        file_path = TRACE_RESULTS_DIR / f"{safe_key}.json"
        
        # 保存结果（包含trace_key以便恢复）
        save_data = {
            "trace_key": trace_key,
            "result": result,
            "saved_at": time.time()
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Trace结果已保存到磁盘: {file_path}")
    except Exception as e:
        print(f"⚠️ 保存trace结果到磁盘失败: {e}")

def _load_trace_results_from_disk() -> None:
    """从磁盘加载已保存的trace结果"""
    global _circuit_trace_results
    
    try:
        if not TRACE_RESULTS_DIR.exists():
            return
        
        loaded_count = 0
        for file_path in TRACE_RESULTS_DIR.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    save_data = json.load(f)
                
                trace_key = save_data.get("trace_key")
                result = save_data.get("result")
                
                if trace_key and result:
                    # 只加载最近30天的结果（避免加载过多旧数据）
                    saved_at = save_data.get("saved_at", 0)
                    if time.time() - saved_at < 30 * 24 * 3600:
                        _circuit_trace_results[trace_key] = result
                        loaded_count += 1
                    else:
                        # 删除过期文件
                        file_path.unlink()
                        print(f"🗑️ 删除过期的trace结果: {file_path}")
            except Exception as e:
                print(f"⚠️ 加载trace结果文件失败 {file_path}: {e}")
        
        if loaded_count > 0:
            print(f"✅ 从磁盘加载了 {loaded_count} 个trace结果")
    except Exception as e:
        print(f"⚠️ 加载trace结果失败: {e}")

# 服务器启动时加载已保存的结果
_load_trace_results_from_disk()

# 使用统一的持久化存储（已在上方定义）
def _load_trace_result_from_disk(trace_key: str) -> dict | None:
    """从磁盘加载trace结果（使用统一的存储格式）"""
    import urllib.parse
    try:
        # 使用安全的文件名
        safe_key = trace_key.replace("::", "_").replace("/", "_").replace(" ", "_")
        file_path = TRACE_RESULTS_DIR / f"{safe_key}.json"
        
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                save_data = json.load(f)
            
            # 检查trace_key是否匹配
            saved_trace_key = save_data.get("trace_key")
            if saved_trace_key == trace_key:
                result = save_data.get("result")
                if result:
                    print(f"✅ 从磁盘加载trace结果: {file_path}")
                    return result
        
        # 如果精确匹配失败，尝试遍历所有文件查找匹配的trace_key（处理编码差异）
        if TRACE_RESULTS_DIR.exists():
            for storage_file in TRACE_RESULTS_DIR.glob("*.json"):
                try:
                    with open(storage_file, "r", encoding="utf-8") as f:
                        save_data = json.load(f)
                    
                    saved_trace_key = save_data.get("trace_key")
                    # 尝试解码比较（处理可能的编码差异）
                    if saved_trace_key == trace_key:
                        result = save_data.get("result")
                        if result:
                            print(f"✅ 从磁盘加载trace结果（通过遍历查找）: {storage_file}")
                            return result
                except Exception as e:
                    continue
        
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"⚠️ 从磁盘加载trace结果失败 ({trace_key}): {e}")
        return None

def get_hooked_model(model_name: str = 'lc0/BT4-1024x15x32h'):
    """获取或加载HookedTransformer模型 - 仅支持BT4（带全局缓存）"""
    global _hooked_models
    
    # 强制使用BT4模型
    model_name = 'lc0/BT4-1024x15x32h'
    
    # 先检查circuits_service的缓存（只对模型本身，不区分 SAE 组合）
    if CIRCUITS_SERVICE_AVAILABLE and get_cached_models is not None:
        cached_hooked_model, _, _, _ = get_cached_models(model_name)
        if cached_hooked_model is not None:
            return cached_hooked_model
    
    # 检查本地缓存
    if model_name not in _hooked_models:
        if not HOOKED_TRANSFORMER_AVAILABLE:
            raise ValueError("HookedTransformer不可用，请安装transformer_lens")
        
        print(f"🔍 正在加载HookedTransformer模型: {model_name}")
        model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            dtype=torch.float32,
        ).eval()
        _hooked_models[model_name] = model
        
        # 如果circuits_service可用，也更新共享缓存
        if CIRCUITS_SERVICE_AVAILABLE and set_cached_models is not None:
            # 需要transcoders和lorsas才能调用set_cached_models，这里只缓存模型
            _global_hooked_models[model_name] = model
        
        print(f"✅ HookedTransformer模型 {model_name} 加载成功")
    
    return _hooked_models[model_name]

def get_cached_transcoders_and_lorsas(
    model_name: str,
    sae_combo_id: str | None = None,
) -> Tuple[Optional[Dict[int, SparseAutoEncoder]], Optional[List[LowRankSparseAttention]]]:
    """获取缓存的 transcoders 和 lorsas（优先使用 circuits_service 的共享缓存）"""

    combo_id = sae_combo_id or CURRENT_BT4_SAE_COMBO_ID
    cache_key = _make_combo_cache_key(model_name, combo_id)

    # 先检查circuits_service的缓存
    if CIRCUITS_SERVICE_AVAILABLE and get_cached_models is not None:
        _, cached_transcoders, cached_lorsas, _ = get_cached_models(cache_key)
        if cached_transcoders is not None and cached_lorsas is not None:
            return cached_transcoders, cached_lorsas

    # 检查本地缓存
    global _transcoders_cache, _lorsas_cache
    return _transcoders_cache.get(cache_key), _lorsas_cache.get(cache_key)

def get_available_models():
    """获取可用的模型列表 - 仅支持BT4"""
    return [
        {'name': 'lc0/BT4-1024x15x32h', 'display_name': 'BT4-1024x15x32h'},
    ]


def make_serializable(obj):
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def trim_minimum(
    origins: list[dict[str, Any] | None],
    feature_acts_indices: np.ndarray,
    feature_acts_values: np.ndarray,
) -> tuple[list[dict[str, Any] | None], np.ndarray, np.ndarray]:
    """Trim multiple arrays to the length of the shortest non-None array.

    Args:
        origins: Origins
        feature_acts_indices: Feature acts indices
        feature_acts_values: Feature acts values

    Returns:
        list: List of trimmed arrays
    """
    # 检查是否为国际象棋模型（通过检查origins中是否包含FEN数据）
    has_fen_data = any(
        origin is not None and origin.get("key") == "fen" 
        for origin in origins if origin is not None
    )

    if has_fen_data:
        # 对于国际象棋模型，强制最小长度为64（棋盘格子数）
        min_length = max(64, feature_acts_indices[-1] + 10)
    else:
        # 对于其他模型，使用原有逻辑
        min_length = min(len(origins), feature_acts_indices[-1] + 10)
    
    feature_acts_indices_mask = feature_acts_indices <= min_length
    return origins[:int(min_length)], feature_acts_indices[feature_acts_indices_mask], feature_acts_values[feature_acts_indices_mask]


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


###############################################################################
# BT4 SAE 组合相关 API
###############################################################################


@app.get("/sae/combos")
def list_sae_combos() -> Dict[str, Any]:
    """
    返回可选的 BT4 SAE 组合列表及默认组合。

    这些组合来自 `exp/38mongoanalyses/组合.txt`，前端只能在这些组合中选择。
    """

    combos = [
        {
            "id": cfg["id"],
            "label": cfg["label"],
            "tc_base_path": cfg["tc_base_path"],
            "lorsa_base_path": cfg["lorsa_base_path"],
        }
        for cfg in BT4_SAE_COMBOS.values()
    ]

    return {
        "default_id": BT4_DEFAULT_SAE_COMBO,
        "current_id": CURRENT_BT4_SAE_COMBO_ID,
        "combos": combos,
    }


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


@app.get("/dictionaries/{name}/features/{feature_index}")
def get_feature(
    name: str,
    feature_index: str | int,
    feature_analysis_name: str | None = None,
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
    print(f'{feature_analysis_name = }')
    print(f'{name = }')
    # Get feature data
    feature = (
        client.get_random_alive_feature(
            sae_name=name,
            sae_series=sae_series,
            name=feature_analysis_name,
        )
        if feature_index == "random"
        else client.get_feature(
            sae_name=name,
            sae_series=sae_series,
            index=feature_index)
    )

    if feature is None:
        return Response(
            content=f"Feature {feature_index} not found in SAE {name}",
            status_code=404,
        )

    analysis = next(
        (
            a for a in feature.analyses
            if a.name == feature_analysis_name or feature_analysis_name is None
        ),
        None,
    )
    print(f'{analysis = }')
    if analysis is None:
        return Response(
            content=f"Feature analysis {feature_analysis_name} not found in SAE {name}"
            if feature_analysis_name is not None
            else f"No feature analysis found in SAE {name}",
            status_code=404,
        )

    def process_sample(
        *,
        sparse_feature_acts,
        context_idx,
        dataset_name,
        model_name,
        shard_idx=None,
        n_shards=None,
    ):
        """Process a sample to extract and format feature data.

        Args:
            sparse_feature_acts: Sparse feature activations,
                optional z pattern activations
            decoder_norms: Decoder norms
            context_idx: Context index in the dataset
            dataset_name: Name of the dataset
            model_name: Name of the model
            shard_idx: Index of the dataset shard, defaults to 0
            n_shards: Total number of shards, defaults to 1

        Returns:
            dict: Processed sample data
        """
        model = get_model(model_name)
        
        data = get_dataset(dataset_name, shard_idx, n_shards)[context_idx.item()]

        # Get origins for the features
        origins = model.trace({k: [v] for k, v in data.items()})[0]

        # Process image data if present
        image_key = next(
            (key for key in ["image", "images"] if key in data),
            None,
        )
        if image_key is not None:
            image_urls = [
                f"/images/{dataset_name}?context_idx={context_idx}&"
                f"shard_idx={shard_idx}&n_shards={n_shards}&"
                f"image_idx={img_idx}"
                for img_idx in range(len(data[image_key]))
            ]
            del data[image_key]
            data["images"] = image_urls

        # Trim to matching lengths
        (
            feature_acts_indices,
            feature_acts_values,
            z_pattern_indices,
            z_pattern_values,
        ) = sparse_feature_acts

        origins, feature_acts_indices, feature_acts_values = trim_minimum(
            origins,
            feature_acts_indices,
            feature_acts_values,
        )
        assert (
            origins is not None
            and feature_acts_indices is not None
            and feature_acts_values is not None
        ), "Origins and feature acts must not be None"

        # 检查是否为国际象棋模型（多种检测方式）
        has_fen_data = any(
            origin is not None and origin.get("key") == "fen" 
            for origin in origins if origin is not None
        )
        
        # 通过模型名称或数据集名称判断是否为棋类模型
        is_chess_model = (
            has_fen_data or 
            "chess" in model_name.lower() or 
            "lc0" in model_name.lower() or
            "chess" in dataset_name.lower() or
            "lc0" in dataset_name.lower()
        )
        
        if is_chess_model:
            # 对于国际象棋模型，创建长度为64的密集激活数组
            dense_feature_acts = np.zeros(64)
            
            # 强制类型
            feature_acts_indices = np.asarray(feature_acts_indices, dtype=np.int64)
            feature_acts_values = np.asarray(feature_acts_values, dtype=np.float32)

            # 可选：过滤非法索引
            valid_mask = (feature_acts_indices >= 0) & (feature_acts_indices < 64)
            feature_acts_indices = feature_acts_indices[valid_mask]
            feature_acts_values = feature_acts_values[valid_mask]

            # 然后再 zip 循环或直接向量化写入
            for idx, val in zip(feature_acts_indices, feature_acts_values):
                        dense_feature_acts[idx] = val
            
            # 确保FEN数据存在
            if "fen" not in data:
                # 如果没有FEN数据，尝试从origins中提取
                fen_origins = [origin for origin in origins if origin is not None and origin.get("key") == "fen"]
                if fen_origins:
                    # 使用第一个FEN origin的范围来提取文本
                    fen_origin = fen_origins[0]
                    if "range" in fen_origin and "text" in data:
                        start, end = fen_origin["range"]
                        data["fen"] = data["text"][start:end]
                    else:
                        # 如果没有range信息，使用整个文本
                        data["fen"] = data.get("text", "")
                else:
                    # 如果完全没有FEN信息，创建一个默认的
                    data["fen"] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            
        else:
            # 对于其他模型，使用原有逻辑
            dense_feature_acts = np.zeros(len(origins))
            
            for i, (idx, val) in enumerate(zip(feature_acts_indices, feature_acts_values)):
                try:
                    # 确保idx是有效的整数
                    if hasattr(idx, 'item'):
                        idx = idx.item()
                    elif hasattr(idx, '__int__'):
                        idx = int(idx)
                    else:
                        idx = int(float(idx))
                    
                    # 确保val是有效的数值
                    if hasattr(val, 'item'):
                        val = val.item()
                    elif hasattr(val, '__float__'):
                        val = float(val)
                    else:
                        val = float(val)
                    
                    # 检查索引范围
                    if 0 <= idx < len(origins):
                        dense_feature_acts[idx] = val
                        
                except (ValueError, TypeError, IndexError):
                    continue

        # Process text data if present
        if "text" in data:
            text_ranges = [
                origin["range"] for origin in origins
                if origin is not None and origin["key"] == "text"
            ]
            if text_ranges:
                max_text_origin = max(text_ranges, key=lambda x: x[1])
                data["text"] = data["text"][: max_text_origin[1]]

        # 对于国际象棋模型，使用FEN作为文本
        if is_chess_model:
            data["text"] = data.get("fen", "No FEN data")

        return {
            **data,
            "origins": origins,
            "feature_acts": dense_feature_acts,  # 返回密集激活数组
            "feature_acts_indices": feature_acts_indices,
            "feature_acts_values": feature_acts_values,
            "z_pattern_indices": z_pattern_indices,
            "z_pattern_values": z_pattern_values,
        }
    
    def process_sparse_feature_acts(
        feature_acts_indices: np.ndarray,
        feature_acts_values: np.ndarray,
        z_pattern_indices: np.ndarray | None = None,
        z_pattern_values: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]]:
        """Process sparse feature acts.
        
        Args:
            feature_acts_indices: Feature acts indices
            feature_acts_values: Feature acts values
            z_pattern_indices: Z pattern indices
            z_pattern_values: Z pattern values
        
        TODO: This is really ugly, we should find a better way to do this.
        """

        if feature_acts_indices.size == 0 or feature_acts_indices.shape[1] == 0:
            return


        _, feature_acts_counts = np.unique(
            feature_acts_indices[0],
            return_counts=True,
        )

        _, z_pattern_counts = (
            np.unique(z_pattern_indices[0], return_counts=True)
            if z_pattern_indices is not None
            else (None, None)
        )

        feature_acts_sample_ranges = np.concatenate(
            [[0], np.cumsum(feature_acts_counts)]
        )

        z_pattern_sample_ranges = (
            np.concatenate([[0], np.cumsum(z_pattern_counts)])
            if z_pattern_counts is not None
            else None
        )

        feature_acts_sample_ranges = list(
            zip(feature_acts_sample_ranges[:-1], feature_acts_sample_ranges[1:])
        )

        if z_pattern_sample_ranges is not None:
            z_pattern_sample_ranges = list(
                zip(z_pattern_sample_ranges[:-1], z_pattern_sample_ranges[1:])
            )
            if len(feature_acts_sample_ranges) != len(z_pattern_sample_ranges):
                z_pattern_sample_ranges = [(None, None)] * len(feature_acts_sample_ranges)
        else:
            z_pattern_sample_ranges = [(None, None)] * len(feature_acts_sample_ranges)

        for (feature_acts_start, feature_acts_end), (z_pattern_start, z_pattern_end) in zip(feature_acts_sample_ranges, z_pattern_sample_ranges):
            feature_acts_indices_i = feature_acts_indices[1, feature_acts_start:feature_acts_end]
            feature_acts_values_i = feature_acts_values[feature_acts_start:feature_acts_end]
            z_pattern_indices_i = z_pattern_indices[1:, z_pattern_start:z_pattern_end] if z_pattern_indices is not None else None
            z_pattern_values_i = z_pattern_values[z_pattern_start:z_pattern_end] if z_pattern_values is not None else None

            yield feature_acts_indices_i, feature_acts_values_i, z_pattern_indices_i, z_pattern_values_i


    sample_groups = []
    for sampling in analysis.samplings:
        try:
            # Using zip to process correlated data instead of indexing
            samples = [
                process_sample(
                    sparse_feature_acts=sparse_feature_acts,
                    context_idx=context_idx,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    shard_idx=shard_idx,
                    n_shards=n_shards,
                )
                for sparse_feature_acts, context_idx, dataset_name, model_name, shard_idx, n_shards in zip(
                    process_sparse_feature_acts(
                        sampling.feature_acts_indices,
                        sampling.feature_acts_values,
                        sampling.z_pattern_indices,
                        sampling.z_pattern_values,
                    ),
                    sampling.context_idx,
                    sampling.dataset_name,
                    sampling.model_name,
                    sampling.shard_idx if sampling.shard_idx is not None else [0] * len(sampling.feature_acts_indices),
                    sampling.n_shards if sampling.n_shards is not None else [1] * len(sampling.feature_acts_indices),
                )
            ]
            

            sample_groups.append(
                {
                    "analysis_name": sampling.name,
                    "samples": samples,
                }
            )
        except Exception as e:
            # 返回400错误响应
            return Response(
                content=f"处理sampling '{sampling.name}' 时出错: {str(e)}", 
                status_code=400
            )

    # Prepare response
    response_data = {
        "feature_index": feature.index,
        "analysis_name": analysis.name,
        "interpretation": feature.interpretation,
        "dictionary_name": feature.sae_name,
        "decoder_norms": analysis.decoder_norms,
        "decoder_similarity_matrices": analysis.decoder_similarity_matrices,
        "decoder_inner_product_matrices": analysis.decoder_inner_product_matrices,
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


@app.post("/circuit/sync_clerps_to_interpretations")
def sync_clerps_to_interpretations(request: dict):

    try:
        nodes = request.get("nodes", [])
        lorsa_analysis_name = request.get("lorsa_analysis_name")
        tc_analysis_name = request.get("tc_analysis_name")
        
        if not isinstance(nodes, list):
            raise HTTPException(status_code=400, detail="nodes must be a list")
        
        # 根据analysis_name找到对应的组合配置
        combo_cfg = None
        if lorsa_analysis_name or tc_analysis_name:
            for combo_id, cfg in BT4_SAE_COMBOS.items():
                if (lorsa_analysis_name and cfg.get("lorsa_analysis_name") == lorsa_analysis_name) or \
                   (tc_analysis_name and cfg.get("tc_analysis_name") == tc_analysis_name):
                    combo_cfg = cfg
                    break
        
        print(f"🔄 开始同步clerps到interpretations:")
        print(f"   - 节点数量: {len(nodes)}")
        print(f"   - LoRSA analysis_name: {lorsa_analysis_name}")
        print(f"   - TC analysis_name: {tc_analysis_name}")
        if combo_cfg:
            print(f"   - 找到组合配置: {combo_cfg.get('id')}")
            print(f"   - LoRSA模板: {combo_cfg.get('lorsa_sae_name_template')}")
            print(f"   - TC模板: {combo_cfg.get('tc_sae_name_template')}")
        
        synced_count = 0
        skipped_count = 0
        error_count = 0
        results = []
        
        for node in nodes:
            node_id = node.get('node_id')
            clerp = node.get('clerp')
            feature_idx = node.get('feature')
            layer = node.get('layer')
            feature_type = node.get('feature_type', '').lower()
            
            # 跳过没有clerp或clerp为空的节点
            if not clerp or not isinstance(clerp, str) or clerp.strip() == '':
                skipped_count += 1
                continue
            
            # 构建SAE名称（使用模板）
            sae_name = None
            if 'lorsa' in feature_type:
                if combo_cfg and combo_cfg.get('lorsa_sae_name_template'):
                    # 使用模板，替换{layer}为实际层号
                    sae_name = combo_cfg['lorsa_sae_name_template'].format(layer=layer)
                elif lorsa_analysis_name:
                    # 向后兼容：如果没有找到组合配置，尝试使用旧的方式
                    sae_name = lorsa_analysis_name.replace("{}", str(layer))
                else:
                    sae_name = f"BT4_lorsa_L{layer}A"
            elif 'transcoder' in feature_type or 'cross layer transcoder' in feature_type:
                if combo_cfg and combo_cfg.get('tc_sae_name_template'):
                    # 使用模板，替换{layer}为实际层号
                    sae_name = combo_cfg['tc_sae_name_template'].format(layer=layer)
                elif tc_analysis_name:
                    # 向后兼容：如果没有找到组合配置，尝试使用旧的方式
                    sae_name = tc_analysis_name.replace("{}", str(layer))
                else:
                    sae_name = f"BT4_tc_L{layer}M"
            
            if not sae_name or feature_idx is None:
                skipped_count += 1
                continue
            
            try:
                # 解码clerp（如果是URL编码的）
                import urllib.parse
                decoded_clerp = urllib.parse.unquote(clerp)
                
                # 创建interpretation字典
                interpretation_dict = {
                    "text": decoded_clerp,
                    "method": "circuit_clerp",
                    "validation": []
                }
                
                # 保存到MongoDB
                client.update_feature(
                    sae_name=sae_name,
                    sae_series=sae_series,
                    feature_index=feature_idx,
                    update_data={"interpretation": interpretation_dict}
                )
                
                synced_count += 1
                results.append({
                    "node_id": node_id,
                    "sae_name": sae_name,
                    "feature_index": feature_idx,
                    "status": "synced"
                })
                
                print(f"✅ 已同步节点 {node_id}: {sae_name}[{feature_idx}]")
                
            except Exception as e:
                error_count += 1
                results.append({
                    "node_id": node_id,
                    "sae_name": sae_name,
                    "feature_index": feature_idx,
                    "status": "error",
                    "error": str(e)
                })
                print(f"❌ 同步节点 {node_id} 失败: {e}")
        
        summary = {
            "total_nodes": len(nodes),
            "synced": synced_count,
            "skipped": skipped_count,
            "errors": error_count,
            "results": results[:50]  # 只返回前50个详细结果
        }
        
        print(f"✅ 同步完成: {synced_count} 成功, {skipped_count} 跳过, {error_count} 失败")
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"同步失败: {str(e)}")


@app.post("/circuit/sync_interpretations_to_clerps")
def sync_interpretations_to_clerps(request: dict):
    try:
        nodes = request.get("nodes", [])
        lorsa_analysis_name = request.get("lorsa_analysis_name")
        tc_analysis_name = request.get("tc_analysis_name")
        
        if not isinstance(nodes, list):
            raise HTTPException(status_code=400, detail="nodes must be a list")
        
        # 根据analysis_name找到对应的组合配置
        combo_cfg = None
        if lorsa_analysis_name or tc_analysis_name:
            for combo_id, cfg in BT4_SAE_COMBOS.items():
                if (lorsa_analysis_name and cfg.get("lorsa_analysis_name") == lorsa_analysis_name) or \
                   (tc_analysis_name and cfg.get("tc_analysis_name") == tc_analysis_name):
                    combo_cfg = cfg
                    break
        
        print(f"🔄 开始从interpretations同步到clerps:")
        print(f"   - 节点数量: {len(nodes)}")
        print(f"   - LoRSA analysis_name: {lorsa_analysis_name}")
        print(f"   - TC analysis_name: {tc_analysis_name}")
        if combo_cfg:
            print(f"   - 找到组合配置: {combo_cfg.get('id')}")
            print(f"   - LoRSA模板: {combo_cfg.get('lorsa_sae_name_template')}")
            print(f"   - TC模板: {combo_cfg.get('tc_sae_name_template')}")
        
        updated_nodes = []
        found_count = 0
        not_found_count = 0
        
        for node in nodes:
            node_id = node.get('node_id')
            feature_idx = node.get('feature')
            layer = node.get('layer')
            feature_type = node.get('feature_type', '').lower()
            
            # 构建SAE名称（使用模板）
            sae_name = None
            if 'lorsa' in feature_type:
                if combo_cfg and combo_cfg.get('lorsa_sae_name_template'):
                    # 使用模板，替换{layer}为实际层号
                    sae_name = combo_cfg['lorsa_sae_name_template'].format(layer=layer)
                elif lorsa_analysis_name:
                    # 向后兼容：如果没有找到组合配置，尝试使用旧的方式
                    sae_name = lorsa_analysis_name.replace("{}", str(layer))
                else:
                    sae_name = f"BT4_lorsa_L{layer}A"
            elif 'transcoder' in feature_type or 'cross layer transcoder' in feature_type:
                if combo_cfg and combo_cfg.get('tc_sae_name_template'):
                    # 使用模板，替换{layer}为实际层号
                    sae_name = combo_cfg['tc_sae_name_template'].format(layer=layer)
                elif tc_analysis_name:
                    # 向后兼容：如果没有找到组合配置，尝试使用旧的方式
                    sae_name = tc_analysis_name.replace("{}", str(layer))
                else:
                    sae_name = f"BT4_tc_L{layer}M"
            
            updated_node = {**node}  # 复制原节点数据
            
            if sae_name and feature_idx is not None:
                try:
                    # 从MongoDB读取feature
                    feature = client.get_feature(
                        sae_name=sae_name,
                        sae_series=sae_series,
                        index=feature_idx
                    )
                    
                    if feature and feature.interpretation:
                        interp = feature.interpretation
                        if isinstance(interp, dict):
                            clerp_text = interp.get("text", "")
                        else:
                            clerp_text = getattr(interp, "text", "")
                        
                        if clerp_text:
                            updated_node["clerp"] = clerp_text
                            found_count += 1
                            print(f"✅ 找到节点 {node_id} 的interpretation: {sae_name}[{feature_idx}]")
                        else:
                            not_found_count += 1
                    else:
                        not_found_count += 1
                        
                except Exception as e:
                    print(f"⚠️ 读取节点 {node_id} 的interpretation失败: {e}")
                    not_found_count += 1
            else:
                not_found_count += 1
            
            updated_nodes.append(updated_node)
        
        summary = {
            "total_nodes": len(nodes),
            "found": found_count,
            "not_found": not_found_count,
            "updated_nodes": updated_nodes
        }
        
        print(f"✅ 同步完成: {found_count} 找到, {not_found_count} 未找到")
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"同步失败: {str(e)}")


@app.post("/dictionaries/{name}/features/{feature_index}/interpret")
def interpret_feature(
    name: str,
    feature_index: int,
    type: str,
    custom_interpretation: Optional[str] = None
):
    """
    处理特征解释：自动生成、自定义保存或验证
    
    Args:
        name: SAE名称
        feature_index: 特征索引
        type: 解释类型 (auto/custom/validate)
        custom_interpretation: 自定义解释文本（type=custom时需要）
    
    Returns:
        Interpretation对象（字典格式）
    """
    try:
        # 获取特征
        feature = client.get_feature(
            sae_name=name,
            sae_series=sae_series,
            index=feature_index
        )
        
        if feature is None:
            raise HTTPException(
                status_code=404,
                detail=f"Feature {feature_index} not found in SAE {name}"
            )
        
        if type == "custom":
            # 保存自定义解释
            if not custom_interpretation:
                raise HTTPException(
                    status_code=400,
                    detail="custom_interpretation is required for type=custom"
                )
            
            # FastAPI应该已经自动解码了URL编码的参数
            # 如果仍有问题，可以使用 urllib.parse.unquote 解码
            import urllib.parse
            decoded_interpretation = urllib.parse.unquote(custom_interpretation)
            
            print(f"📝 收到解释文本:")
            print(f"   - 原始: {custom_interpretation}")
            print(f"   - 解码: {decoded_interpretation}")
            
            # 创建解释字典（只包含必需字段，其他字段不返回以符合前端schema的optional定义）
            interpretation_dict = {
                "text": decoded_interpretation,
                "method": "custom",
                "validation": []
            }
            
            # 保存到数据库
            try:
                client.update_feature(
                    sae_name=name,
                    sae_series=sae_series,
                    feature_index=feature_index,
                    update_data={"interpretation": interpretation_dict}
                )
            except Exception as update_error:
                print(f"Failed to update feature interpretation: {update_error}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save interpretation: {str(update_error)}"
                )
            
            return interpretation_dict
        
        elif type == "auto":
            raise HTTPException(
                status_code=501,
                detail="Automatic interpretation is not yet implemented. Please use custom interpretation."
            )
        
        elif type == "validate":
            if not feature.interpretation:
                raise HTTPException(
                    status_code=400,
                    detail="No interpretation available to validate"
                )
            
            interp = feature.interpretation
            print(f"📖 读取解释文本: {interp.get('text', '') if isinstance(interp, dict) else getattr(interp, 'text', '')}")
            
            if isinstance(interp, dict):
                result = {
                    "text": interp.get("text", ""),
                    "method": interp.get("method", "unknown"),
                    "validation": interp.get("validation", [])
                }
                if interp.get("passed") is not None:
                    result["passed"] = interp.get("passed")
                if interp.get("complexity") is not None:
                    result["complexity"] = interp.get("complexity")
                if interp.get("consistency") is not None:
                    result["consistency"] = interp.get("consistency")
                return result
            else:
                # 如果是对象，尝试访问属性
                result = {
                    "text": getattr(interp, "text", ""),
                    "method": getattr(interp, "method", "unknown"),
                    "validation": getattr(interp, "validation", [])
                }
                # 只有当值不是None时才添加可选字段
                passed = getattr(interp, "passed", None)
                if passed is not None:
                    result["passed"] = passed
                complexity = getattr(interp, "complexity", None)
                if complexity is not None:
                    result["complexity"] = complexity
                consistency = getattr(interp, "consistency", None)
                if consistency is not None:
                    result["consistency"] = consistency
                return result
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid type: {type}. Must be 'auto', 'custom', or 'validate'"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process interpretation: {str(e)}"
        )


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


# LC0 引擎类
class LC0Engine:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def play(self, chess_board):
        try:
            # 使用 notebook 同款接口进行推理
            fen = chess_board.fen()
            print(f"🔍 处理FEN: {fen}")

            # 创建 LeelaBoard 实例来处理映射
            lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
            lboard.pc_board = chess_board  # 使用现有的棋盘状态

            with torch.no_grad():
                output, cache = self.model.run_with_cache(fen, prepend_bos=False)
                if isinstance(output, (list, tuple)) and len(output) >= 1:
                    policy_output = output[0]
                else:
                    policy_output = output
                if policy_output.dim() == 2:
                    policy_logits = policy_output[0]
                else:
                    policy_logits = policy_output

            legal_moves = list(chess_board.legal_moves)
            legal_uci_set = set(move.uci() for move in legal_moves)
            sorted_indices = torch.argsort(policy_logits, descending=True)

            top10 = []
            for idx in sorted_indices[:10].tolist():
                uci = lboard.idx2uci(idx)
                logit = float(policy_logits[idx].item())
                top10.append((uci, logit))
            
            print("🔍 模型输出调试信息:")
            print(f"   - policy_logits shape: {tuple(policy_logits.shape)}")
            print(f"   - 合法移动数量: {len(legal_moves)}")
            print("   - 前10个最高概率move (uci, logit):")
            print("     " + ", ".join([f"{uci}:{logit:.4f}" for uci, logit in top10]))

            # 依次尝试最高概率索引对应的 UCI，选择第一个合法移动
            for rank, idx in enumerate(sorted_indices.tolist(), start=1):
                uci = lboard.idx2uci(idx)
                if uci in legal_uci_set:
                    move = chess.Move.from_uci(uci)
                    print(f"✅ 选择最大概率合法移动: {uci} (概率排名: {rank}, logit: {policy_logits[idx].item():.4f})")
                    return move

            # 如果未找到合法移动，打印报错并抛异常
            print("❌ 错误：模型未能找到任何合法移动！")
            print(f"   - 当前局面 FEN: {fen}")
            print(f"   - 示例合法移动: {[m.uci() for m in legal_moves[:10]]}")
            print(f"   - 尝试了前 {min(len(sorted_indices), 50)} 个最高概率的token")
            raise ValueError("模型未能找到任何合法移动")

        except Exception as e:
            print(f"❌ LC0Engine.play() 出错: {e}")
            raise e


@app.post("/play_game")
def play_game(request: dict):
    """
    与模型对战：输入当前局面 FEN，返回模型建议的下一步移动 (UCI 格式)
    
    支持两种模式：
    1. 直接使用神经网络策略输出（use_search=False，默认）
    2. 使用 MCTS 搜索（use_search=True）
    
    Args:
        request: 包含以下字段：
            - fen: FEN 字符串（必需）
            - use_search: 是否使用 MCTS 搜索（可选，默认 False）
            - search_params: 搜索参数（可选，use_search=True 时有效）
                - max_playouts: 最大模拟次数（默认 100）
                - target_minibatch_size: minibatch 大小（默认 8）
                - cpuct: UCT 探索系数（默认 1.0）
                - max_depth: 最大搜索深度（默认 10）
    """
    fen = request.get("fen")
    use_search = request.get("use_search", False)
    search_params = request.get("search_params", {})
    # 强制使用BT4模型
    model_name = "lc0/BT4-1024x15x32h"
    
    save_trace = bool(request.get("save_trace", False))
    trace_output_dir = request.get("trace_output_dir") or str(SEARCH_TRACE_OUTPUT_DIR)
    # trace_max_edges: 0 或 None 表示不限制（保存完整搜索树），其他值表示最大边数
    trace_max_edges_raw = request.get("trace_max_edges", 1000)
    trace_max_edges = None if (trace_max_edges_raw == 0 or trace_max_edges_raw is None) else int(trace_max_edges_raw)

    if not fen:
        raise HTTPException(status_code=400, detail="FEN 字符串不能为空")
    
    try:
        board = chess.Board(fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail="无效的 FEN 字符串")
    
    try:
        # 检查HookedTransformer是否可用
        if not HOOKED_TRANSFORMER_AVAILABLE:
            print("❌ 错误：HookedTransformer不可用")
            raise HTTPException(status_code=503, detail="HookedTransformer不可用，请安装transformer_lens")
        
        if use_search:
            # 使用 MCTS 搜索
            print(f"🔍 使用 MCTS 搜索模式: {fen[:50]}...")
            
            # 导入搜索模块
            try:
                from search.model_interface import run_mcts_search, set_model_getter
                # 设置模型获取器以复用缓存
                set_model_getter(get_hooked_model)
            except ImportError as e:
                print(f"❌ 导入搜索模块失败: {e}")
                raise HTTPException(status_code=503, detail="MCTS 搜索模块不可用")
            
            # 解析搜索参数，使用默认值
            max_playouts = search_params.get("max_playouts", 100)
            target_minibatch_size = search_params.get("target_minibatch_size", 8)
            cpuct = search_params.get("cpuct", 1.0)
            max_depth = search_params.get("max_depth", 10)
            
            print(f"   搜索参数: max_playouts={max_playouts}, cpuct={cpuct}, max_depth={max_depth}")
            
            # 运行搜索
            search_result = run_mcts_search(
                fen=fen,
                max_playouts=max_playouts,
                target_minibatch_size=target_minibatch_size,
                cpuct=cpuct,
                max_depth=max_depth,
                model_name=model_name,
            )
            
            best_move = search_result.get("best_move")
            if not best_move:
                raise ValueError("MCTS 搜索未能找到合法移动")
            
            print(f"✅ MCTS 搜索完成: {best_move}, playouts={search_result.get('total_playouts')}")
            
            return {
                "move": best_move,
                "model_used": model_name,
                "search_used": True,
                "search_stats": {
                    "total_playouts": search_result.get("total_playouts"),
                    "max_depth_reached": search_result.get("max_depth_reached"),
                    "root_visits": search_result.get("root_visits"),
                    "top_moves": search_result.get("top_moves", [])[:5],  # 只返回前5个
                }
            }
        else:
            # 直接使用神经网络策略输出
            model = get_hooked_model(model_name)
            engine = LC0Engine(model)
            move = engine.play(board)
            return {"move": move.uci(), "model_used": model_name, "search_used": False}
        
    except ValueError as e:
        print(f"❌ 模型找不到合法移动: {e}")
        raise HTTPException(status_code=400, detail=f"模型找不到合法移动: {str(e)}")
    except Exception as e:
        print(f"❌ 处理移动时出错: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理移动时出错: {str(e)}")


@app.post("/play_game_with_search")
def play_game_with_search(request: dict):
    """
    与模型对战（使用 MCTS 搜索）：输入当前局面 FEN 和搜索参数，返回模型建议的下一步移动 (UCI 格式)
    
    请求参数:
        - fen: FEN 字符串
        - max_playouts: 最大模拟次数（默认 100）
        - target_minibatch_size: 目标 minibatch 大小（默认 8）
        - cpuct: UCT 探索系数（默认 1.0）
        - max_depth: 最大搜索深度（默认 10，0 表示不限制）
        - low_q_exploration_enabled: 是否启用低Q值探索增强（默认 False）
        - low_q_threshold: Q值阈值，低于此值认为是"低Q值"（默认 0.3）
        - low_q_exploration_bonus: 探索奖励的基础值（默认 0.1）
        - low_q_visit_threshold: 访问次数阈值，低于此值认为是"未充分探索"（默认 5）
    """
    fen = request.get("fen")
    # 强制使用BT4模型
    model_name = "lc0/BT4-1024x15x32h"
    
    # 搜索参数（使用默认值）
    max_playouts = request.get("max_playouts", 100)
    target_minibatch_size = request.get("target_minibatch_size", 8)
    cpuct = request.get("cpuct", 1.0)
    max_depth = request.get("max_depth", 10)
    
    # 低Q值探索增强参数（用于发现弃后连杀等隐藏走法）
    low_q_exploration_enabled = request.get("low_q_exploration_enabled", False)
    low_q_threshold = request.get("low_q_threshold", 0.3)
    low_q_exploration_bonus = request.get("low_q_exploration_bonus", 0.1)
    low_q_visit_threshold = request.get("low_q_visit_threshold", 5)
    
    save_trace = bool(request.get("save_trace", False))
    trace_slug = request.get("trace_slug")
    trace_output_dir = request.get("trace_output_dir") or str(SEARCH_TRACE_OUTPUT_DIR)
    # trace_max_edges: 0 或 None 表示不限制（保存完整搜索树），其他值表示最大边数
    trace_max_edges_raw = request.get("trace_max_edges", 1000)
    trace_max_edges = None if (trace_max_edges_raw == 0 or trace_max_edges_raw is None) else int(trace_max_edges_raw)
    
    if not fen:
        raise HTTPException(status_code=400, detail="FEN 字符串不能为空")
    
    try:
        board = chess.Board(fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail="无效的 FEN 字符串")
    
    try:
        # 检查HookedTransformer是否可用
        if not HOOKED_TRANSFORMER_AVAILABLE:
            print("❌ 错误：HookedTransformer不可用")
            raise HTTPException(status_code=503, detail="HookedTransformer不可用，请安装transformer_lens")
        
        # 导入搜索模块
        from search import (
            SearchParams, Search, SimpleBackend, Node, SearchTracer,
            get_wl, get_d, get_m, get_policy,
            policy_tensor_to_move_dict, set_model_getter,
        )
        
        # 设置模型获取函数，使用共享缓存
        set_model_getter(get_hooked_model)
        
        # 创建模型评估函数
        def model_eval_fn(fen_str: str) -> dict:
            """模型评估函数，返回 q, d, m, p"""
            wl = get_wl(fen_str, model_name)
            d = get_d(fen_str, model_name)
            m_tensor = get_m(fen_str, model_name)
            m_value = m_tensor.item() if hasattr(m_tensor, 'item') else float(m_tensor)
            
            # 获取策略
            policy_tensor = get_policy(fen_str, model_name)
            policy_dict = policy_tensor_to_move_dict(policy_tensor, fen_str)
            
            return {
                'q': wl,
                'd': d,
                'm': m_value,
                'p': policy_dict
            }
        
        # 创建搜索参数
        params = SearchParams(
            max_playouts=max_playouts,
            target_minibatch_size=target_minibatch_size,
            cpuct=cpuct,
            max_depth=max_depth,
            low_q_exploration_enabled=low_q_exploration_enabled,
            low_q_threshold=low_q_threshold,
            low_q_exploration_bonus=low_q_exploration_bonus,
            low_q_visit_threshold=low_q_visit_threshold,
        )
        
        # 创建后端和根节点
        backend = SimpleBackend(model_eval_fn)
        root_node = Node(fen=fen)
        
        tracer = SearchTracer() if save_trace else None
        # 创建搜索对象并运行
        search = Search(
            root_node=root_node,
            backend=backend,
            params=params,
            tracer=tracer,
        )
        
        print(f"🔍 开始 MCTS 搜索: max_playouts={max_playouts}, max_depth={max_depth}")
        search.run_blocking()
        
        # 获取最佳移动
        best_move = search.get_best_move()
        total_playouts = search.get_total_playouts()
        current_max_depth = search.get_current_max_depth()
        
        if best_move is None:
            raise ValueError("搜索未能找到合法移动")
        
        print(f"✅ MCTS 搜索完成: playouts={total_playouts}, depth={current_max_depth}, best_move={best_move.uci()}")
        
        trace_file_path = None
        if save_trace and tracer:
            trace_file_path = search.export_trace_json(
                output_dir=trace_output_dir,
                max_edges=trace_max_edges,
            )

        response_data = {
            "move": best_move.uci(),
            "model_used": model_name,
            "search_info": {
                "total_playouts": total_playouts,
                "max_depth_reached": current_max_depth,
                "max_depth_limit": max_depth,
            }
        }
        if trace_file_path:
            response_data["trace_file_path"] = trace_file_path
            response_data["trace_filename"] = Path(trace_file_path).name

        return response_data
        
    except ValueError as e:
        print(f"❌ 搜索找不到合法移动: {e}")
        raise HTTPException(status_code=400, detail=f"搜索找不到合法移动: {str(e)}")
    except Exception as e:
        print(f"❌ 搜索处理时出错: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"搜索处理时出错: {str(e)}")


@app.get("/search_trace/files/{filename}")
def download_search_trace_file(filename: str):
    """下载保存的MCTS搜索trace文件"""
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    target_path = SEARCH_TRACE_OUTPUT_DIR / safe_name
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Trace file not found")
    return FileResponse(
        path=target_path,
        media_type="application/json",
        filename=safe_name,
    )


# 在play_game接口后添加局面分析接口
@app.post("/analyze/board")
def analyze_board(request: dict):
    """使用HookedTransformer模型分析当前局面，并返回行棋方胜率、和棋率及对方胜率"""
    fen = request.get("fen")
    # 强制使用BT4模型
    model_name = "lc0/BT4-1024x15x32h"
    
    if not fen:
        raise HTTPException(status_code=400, detail="FEN字符串不能为空")
    try:
        if not HOOKED_TRANSFORMER_AVAILABLE:
            raise HTTPException(status_code=503, detail="HookedTransformer不可用，请安装transformer_lens")
        
        # 使用指定的模型
        model = get_hooked_model(model_name)
        
        with torch.no_grad():
            output, _ = model.run_with_cache(fen, prepend_bos=False)
        
        # 模型输出是一个列表，包含三个元素：
        # output[0]: logits, shape [1, 1858]
        # output[1]: WDL, shape [1, 3] - [当前行棋方胜率, 和棋率, 当前行棋方败率]
        # output[2]: 其他输出, shape [1, 1]
        
        if isinstance(output, (list, tuple)) and len(output) >= 2:
            wdl_tensor = output[1]  # 获取WDL输出
            if wdl_tensor.shape == torch.Size([1, 3]):
                # WDL已经是概率分布，不需要softmax
                current_player_win = wdl_tensor[0][0].item()  # 当前行棋方胜率
                draw_prob = wdl_tensor[0][1].item()  # 和棋率
                current_player_loss = wdl_tensor[0][2].item()  # 当前行棋方败率
                
                # 直接返回当前行棋方的胜率信息，不进行翻转
                # [当前行棋方胜率, 和棋率, 对方胜率]
                evaluation = [current_player_win, draw_prob, current_player_loss]
            else:
                print(f"WDL输出形状不正确: {wdl_tensor.shape}, 期望 [1, 3]")
                evaluation = [0.5, 0.2, 0.3]
        else:
            print(f"模型输出格式不正确，期望包含至少2个元素的列表，实际得到: {type(output)}")
            evaluation = [0.5, 0.2, 0.3]
        
        return {"evaluation": evaluation, "model_used": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"局面分析出错: {str(e)}")


@app.get("/models")
def get_models():
    """获取可用的模型列表"""
    return {"models": get_available_models()}


# 导入circuits_service
try:
    from circuits_service import (
        run_circuit_trace, 
        check_dense_features, 
        load_model_and_transcoders,
        get_cached_models,
        set_cached_models,
        _global_hooked_models,
        _global_transcoders_cache,
        _global_lorsas_cache,
        _global_replacement_models_cache
    )
    from lm_saes.circuit.replacement_lc0_model import ReplacementModel
    CIRCUITS_SERVICE_AVAILABLE = True
    # 如果circuits_service可用，将本地缓存指向共享缓存
    _hooked_models = _global_hooked_models
    _transcoders_cache = _global_transcoders_cache
    _lorsas_cache = _global_lorsas_cache
    _replacement_models_cache = _global_replacement_models_cache
except ImportError as e:
    run_circuit_trace = None
    check_dense_features = None
    load_model_and_transcoders = None
    get_cached_models = None
    set_cached_models = None
    _global_hooked_models = {}
    _global_transcoders_cache = {}
    _global_lorsas_cache = {}
    _global_replacement_models_cache = {}
    ReplacementModel = None
    CIRCUITS_SERVICE_AVAILABLE = False
    print(f"WARNING: circuits_service not found, circuit tracing will not be available: {e}")

# 导入patching服务
try:
    from patching import run_patching_analysis
    PATCHING_SERVICE_AVAILABLE = True
except ImportError:
    run_patching_analysis = None
    PATCHING_SERVICE_AVAILABLE = False
    print("WARNING: patching service not found, patching analysis will not be available")

# 导入intervention服务
try:
    from intervention import run_feature_steering_analysis
    INTERVENTION_SERVICE_AVAILABLE = True
except ImportError:
    run_feature_steering_analysis = None
    INTERVENTION_SERVICE_AVAILABLE = False
    print("WARNING: intervention service not found, steering analysis will not be available")

# 导入自对弈服务
try:
    from self_play import run_self_play, analyze_game_positions
    SELF_PLAY_SERVICE_AVAILABLE = True
except ImportError:
    run_self_play = None
    analyze_game_positions = None
    SELF_PLAY_SERVICE_AVAILABLE = False
    print("WARNING: self-play service not found, self-play functionality will not be available")

# 导入Logit Lens服务
try:
    from logit_lens import IntegratedPolicyLens
    LOGIT_LENS_AVAILABLE = True
except ImportError:
    IntegratedPolicyLens = None
    LOGIT_LENS_AVAILABLE = False
    print("WARNING: logit_lens not found, logit lens functionality will not be available")

# 全局Logit Lens缓存
_logit_lens_instances = {}

# Circuit tracing进程跟踪（防止同时运行多个trace）
_circuit_tracing_lock = threading.Lock()
_is_circuit_tracing = False


@app.post("/circuit/preload_models")
def preload_circuit_models(request: dict):
    """
    预加载 transcoders 和 lorsas 模型，以便后续的 circuit trace 能够快速使用。

    Args:
        request: 包含模型信息的请求体
            - model_name: 模型名称 (可选，默认: "lc0/BT4-1024x15x32h")
            - sae_combo_id: SAE 组合 ID（例如 "k_64_e_32"，可选，默认使用后端当前组合）

    行为：
        - 如果选择了与当前不同的组合，会先清理之前组合的 SAE 缓存并尝试释放显存；
        - 同一组合在已加载且完整时直接返回 already_loaded；
        - 加载过程中的进度日志会写入全局 _loading_logs，前端可轮询查看。
    """

    global CURRENT_BT4_SAE_COMBO_ID, _loading_locks, _loading_status, _loading_logs, _cancel_loading
    global _transcoders_cache, _lorsas_cache, _replacement_models_cache, _global_loading_lock

    model_name = request.get("model_name", "lc0/BT4-1024x15x32h")
    
    # URL解码，处理可能的编码问题（与 /circuit/loading_logs 保持一致）
    import urllib.parse
    
    decoded_model_name = urllib.parse.unquote(model_name)
    if "%" in decoded_model_name:
        decoded_model_name = urllib.parse.unquote(decoded_model_name)
    
    requested_combo_id = request.get("sae_combo_id") or CURRENT_BT4_SAE_COMBO_ID

    # 归一化组合配置（如果传入了未知 ID，会回退到默认组合）
    combo_cfg = get_bt4_sae_combo(requested_combo_id)
    combo_id = combo_cfg["id"]
    # 使用解码后的 model_name 生成缓存键
    combo_key = _make_combo_cache_key(decoded_model_name, combo_id)
    
    # 如果切换组合，先中断当前正在加载的其他组合
    if combo_id != CURRENT_BT4_SAE_COMBO_ID:
        # 中断所有其他组合的加载
        for other_combo_key in list(_cancel_loading.keys()):
            if other_combo_key != combo_key:
                _cancel_loading[other_combo_key] = True
                print(f"🛑 标记中断加载: {other_combo_key}")
                # 如果该组合正在加载，也在日志中记录
                if other_combo_key in _loading_logs:
                    _loading_logs[other_combo_key].append({
                        "timestamp": time.time(),
                        "message": f"🛑 加载被中断（切换到新组合 {combo_id}）",
                    })

    try:
        if not CIRCUITS_SERVICE_AVAILABLE or load_model_and_transcoders is None:
            raise HTTPException(status_code=503, detail="Circuit tracing service not available")

        # 如果切换组合，则清空之前组合的 SAE 缓存并尝试释放显存
        if combo_id != CURRENT_BT4_SAE_COMBO_ID:
            print(f"🔁 棋类 SAE 组合切换: {CURRENT_BT4_SAE_COMBO_ID} -> {combo_id}，开始清理旧缓存")

            # 清空所有 SAE 缓存（包括 circuits_service 的全局缓存），仅保留 HookedTransformer 模型本身
            for cache_name, cache in [
                ("_transcoders_cache", _transcoders_cache),
                ("_lorsas_cache", _lorsas_cache),
                ("_replacement_models_cache", _replacement_models_cache),
            ]:
                try:
                    for cache_key, v in list(cache.items()):
                        # 尝试把 SAE 挪到 CPU，再删除引用
                        if isinstance(v, dict):
                            for sae in v.values():
                                try:
                                    if hasattr(sae, "to"):
                                        sae.to("cpu")
                                except Exception:
                                    continue
                        elif isinstance(v, list):
                            for sae in v:
                                try:
                                    if hasattr(sae, "to"):
                                        sae.to("cpu")
                                except Exception:
                                    continue
                        del cache[cache_key]
                    print(f"   - 已清空缓存 {cache_name}")
                except Exception as clear_err:
                    print(f"   ⚠️ 清理缓存 {cache_name} 时出错: {clear_err}")
            
            # 同时清理 circuits_service 的全局缓存
            if CIRCUITS_SERVICE_AVAILABLE:
                try:
                    for cache_key in list(_global_transcoders_cache.keys()):
                        if cache_key != decoded_model_name:  # 保留 HookedTransformer 的缓存键（只有 model_name）
                            del _global_transcoders_cache[cache_key]
                    for cache_key in list(_global_lorsas_cache.keys()):
                        if cache_key != decoded_model_name:
                            del _global_lorsas_cache[cache_key]
                    for cache_key in list(_global_replacement_models_cache.keys()):
                        if cache_key != decoded_model_name:
                            del _global_replacement_models_cache[cache_key]
                    print("   - 已清空 circuits_service 全局缓存")
                except Exception as clear_err:
                    print(f"   ⚠️ 清理 circuits_service 全局缓存时出错: {clear_err}")

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("   - 已调用 torch.cuda.empty_cache() 释放显存")
            except Exception as e:
                print(f"   ⚠️ 调用 empty_cache 失败: {e}")

            # 清理旧的patching分析器
            try:
                from intervention import clear_patching_analyzer
                clear_patching_analyzer(CURRENT_BT4_SAE_COMBO_ID)
                print("   - 已清理旧的patching分析器")
            except (ImportError, Exception) as e:
                print(f"   ⚠️ 清理patching分析器失败: {e}")

            CURRENT_BT4_SAE_COMBO_ID = combo_id

        # 为当前组合创建/获取加载锁
        if combo_key not in _loading_locks:
            _loading_locks[combo_key] = threading.Lock()

        # 检查是否已经预加载
        cached_transcoders, cached_lorsas = get_cached_transcoders_and_lorsas(decoded_model_name, combo_id)
        if cached_transcoders is not None and cached_lorsas is not None:
            if len(cached_transcoders) == 15 and len(cached_lorsas) == 15:
                print(f"✅ Transcoders 和 LoRSAs 已经预加载: {decoded_model_name} @ {combo_id}")
                return {
                    "status": "already_loaded",
                    "message": f"模型 {decoded_model_name} 组合 {combo_id} 的 transcoders 和 lorsas 已经预加载",
                    "model_name": decoded_model_name,
                    "sae_combo_id": combo_id,
                    "n_layers": len(cached_lorsas),
                    "transcoders_count": len(cached_transcoders),
                    "lorsas_count": len(cached_lorsas),
                }

        # 使用全局锁确保同一时间只加载一个配置（避免GPU内存同时被多个配置占用）
        # 然后再使用组合锁避免同一组合的并发加载
        with _global_loading_lock:
            with _loading_locks[combo_key]:
                # 再次检查是否已经加载（可能在等待锁的过程中已经加载完成）
                cached_transcoders, cached_lorsas = get_cached_transcoders_and_lorsas(decoded_model_name, combo_id)
                if cached_transcoders is not None and cached_lorsas is not None:
                    if len(cached_transcoders) == 15 and len(cached_lorsas) == 15:
                        print(f"✅ Transcoders 和 LoRSAs 已经预加载（在锁内检查）: {decoded_model_name} @ {combo_id}")
                        return {
                            "status": "already_loaded",
                            "message": f"模型 {decoded_model_name} 组合 {combo_id} 的 transcoders 和 lorsas 已经预加载",
                            "model_name": decoded_model_name,
                            "sae_combo_id": combo_id,
                            "n_layers": len(cached_lorsas),
                            "transcoders_count": len(cached_transcoders),
                            "lorsas_count": len(cached_lorsas),
                        }

                # 标记正在加载，并清除中断标志（在全局锁内设置，确保其他请求能检测到）
                _loading_status[combo_key] = {"is_loading": True}
                _cancel_loading[combo_key] = False
                print(f"🔍 开始预加载 transcoders 和 lorsas: {decoded_model_name} @ {combo_id} (全局锁已获取)")

                try:
                    # 获取 HookedTransformer 模型
                    hooked_model = get_hooked_model(decoded_model_name)

                    # 仅支持 BT4
                    if "BT4" not in decoded_model_name:
                        raise HTTPException(status_code=400, detail="Unsupported Model!")

                    tc_base_path = combo_cfg["tc_base_path"]
                    lorsa_base_path = combo_cfg["lorsa_base_path"]
                    n_layers = 15

                    # 初始化加载日志
                    if combo_key not in _loading_logs:
                        _loading_logs[combo_key] = []
                    loading_logs = _loading_logs[combo_key]
                    loading_logs.clear()
                    # 添加初始日志
                    loading_logs.append({
                        "timestamp": time.time(),
                        "message": f"🔍 开始预加载 transcoders 和 lorsas: {decoded_model_name} @ {combo_id}",
                    })
                    print(f"📝 初始化加载日志列表: combo_key={combo_key}, 列表ID={id(loading_logs)}")

                    # 加载 transcoders 和 lorsas
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    # 创建取消标志字典（通过引用传递，可以在循环中检查）
                    # 使用一个包装函数来定期检查取消标志
                    def check_cancel():
                        return _cancel_loading.get(combo_key, False)
                    
                    cancel_flag = {"should_cancel": False, "combo_key": combo_key, "check_fn": check_cancel}
                    replacement_model, transcoders, lorsas = load_model_and_transcoders(
                        model_name=decoded_model_name,
                        device=device,
                        tc_base_path=tc_base_path,
                        lorsa_base_path=lorsa_base_path,
                        n_layers=n_layers,
                        hooked_model=hooked_model,
                        loading_logs=loading_logs,
                        cancel_flag=cancel_flag,
                        cache_key=combo_key,  # 传递 cache_key 以区分不同组合
                    )

                    print(f"📝 加载完成后的日志数量: {len(loading_logs)}")

                    # 缓存 transcoders 和 lorsas（同时更新共享缓存和本地缓存）
                    _transcoders_cache[combo_key] = transcoders
                    _lorsas_cache[combo_key] = lorsas
                    _replacement_models_cache[combo_key] = replacement_model

                    # 如果 circuits_service 可用，也更新共享缓存（使用 combo_key 作为缓存键）
                    if CIRCUITS_SERVICE_AVAILABLE and set_cached_models is not None:
                        set_cached_models(combo_key, hooked_model, transcoders, lorsas, replacement_model)

                    print(f"✅ 预加载完成: {model_name} @ {combo_id}")
                    print(f"   - Transcoders: {len(transcoders)} 层")
                    print(f"   - LoRSAs: {len(lorsas)} 层")

                    # 添加完成日志
                    if combo_key in _loading_logs:
                        _loading_logs[combo_key].append(
                            {
                                "timestamp": time.time(),
                                "message": f"✅ 预加载完成: {model_name} @ {combo_id}",
                            }
                        )
                        _loading_logs[combo_key].append(
                            {
                                "timestamp": time.time(),
                                "message": f"   - Transcoders: {len(transcoders)} 层",
                            }
                        )
                        _loading_logs[combo_key].append(
                            {
                                "timestamp": time.time(),
                                "message": f"   - LoRSAs: {len(lorsas)} 层",
                            }
                        )

                    _loading_status[combo_key] = {"is_loading": False}

                    return {
                        "status": "loaded",
                        "message": f"成功预加载模型 {decoded_model_name} 组合 {combo_id} 的 transcoders 和 lorsas",
                        "model_name": decoded_model_name,
                        "sae_combo_id": combo_id,
                        "n_layers": n_layers,
                        "transcoders_count": len(transcoders),
                        "lorsas_count": len(lorsas),
                        "device": device,
                    }
                except InterruptedError as e:
                    # 加载被中断，清空已加载的部分缓存
                    _loading_status[combo_key] = {"is_loading": False}
                    _cancel_loading[combo_key] = False
                    # 清空该组合的缓存
                    if combo_key in _transcoders_cache:
                        del _transcoders_cache[combo_key]
                    if combo_key in _lorsas_cache:
                        del _lorsas_cache[combo_key]
                    if combo_key in _replacement_models_cache:
                        del _replacement_models_cache[combo_key]
                    if combo_key in _loading_logs:
                        _loading_logs[combo_key].append({
                            "timestamp": time.time(),
                            "message": f"🛑 加载已中断并清空缓存: {str(e)}",
                        })
                    print(f"🛑 加载被中断，已清空缓存: {combo_key}")
                    raise HTTPException(status_code=499, detail=f"加载被中断: {str(e)}")
                except Exception:
                    _loading_status[combo_key] = {"is_loading": False}
                    raise

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        if combo_key in _loading_logs:
            _loading_logs[combo_key].append(
                {
                    "timestamp": time.time(),
                    "message": f"❌ 预加载失败: {str(e)}",
                }
            )
        if combo_key in _loading_status:
            _loading_status[combo_key] = {"is_loading": False}
        raise HTTPException(status_code=500, detail=f"预加载失败: {str(e)}")


@app.post("/circuit/cancel_loading")
def cancel_loading(request: dict):
    """
    中断正在进行的模型加载
    
    Args:
        request: 包含模型信息的请求体
            - model_name: 模型名称 (可选，默认: "lc0/BT4-1024x15x32h")
            - sae_combo_id: SAE 组合 ID（可选，如果不提供则中断所有正在加载的组合）
    
    Returns:
        中断结果
    """
    global _cancel_loading, _loading_status, _loading_logs
    global _transcoders_cache, _lorsas_cache, _replacement_models_cache
    
    model_name = request.get("model_name", "lc0/BT4-1024x15x32h")
    requested_combo_id = request.get("sae_combo_id")
    
    if requested_combo_id:
        # 中断指定的组合
        combo_cfg = get_bt4_sae_combo(requested_combo_id)
        combo_id = combo_cfg["id"]
        combo_key = _make_combo_cache_key(model_name, combo_id)
        
        if combo_key in _loading_status and _loading_status[combo_key].get("is_loading", False):
            _cancel_loading[combo_key] = True
            print(f"🛑 标记中断加载: {combo_key}")
            return {
                "status": "cancelled",
                "message": f"已标记中断组合 {combo_id} 的加载",
                "model_name": model_name,
                "sae_combo_id": combo_id,
            }
        else:
            return {
                "status": "not_loading",
                "message": f"组合 {combo_id} 当前没有正在加载",
                "model_name": model_name,
                "sae_combo_id": combo_id,
            }
    else:
        # 中断所有正在加载的组合
        cancelled_keys = []
        for combo_key, status in _loading_status.items():
            if status.get("is_loading", False):
                _cancel_loading[combo_key] = True
                cancelled_keys.append(combo_key)
                print(f"🛑 标记中断加载: {combo_key}")
        
        return {
            "status": "cancelled" if cancelled_keys else "no_loading",
            "message": f"已标记中断 {len(cancelled_keys)} 个组合的加载" if cancelled_keys else "当前没有正在加载的组合",
            "cancelled_keys": cancelled_keys,
        }


@app.get("/circuit/loading_logs")
def get_loading_logs(
    model_name: str = "lc0/BT4-1024x15x32h",
    sae_combo_id: str | None = None,
):
    """
    获取模型加载日志
    
    Args:
        model_name: 模型名称 (查询参数，默认: "lc0/BT4-1024x15x32h")
        sae_combo_id: SAE组合ID (查询参数，可选)
    
    Returns:
        加载日志列表
    """

    global _loading_logs, _loading_status

    # URL解码，处理可能的双重编码问题
    import urllib.parse

    decoded_model_name = urllib.parse.unquote(model_name)
    if "%" in decoded_model_name:
        decoded_model_name = urllib.parse.unquote(decoded_model_name)

    combo_id = sae_combo_id or CURRENT_BT4_SAE_COMBO_ID
    combo_cfg = get_bt4_sae_combo(combo_id)
    normalized_combo_id = combo_cfg["id"]
    combo_key = _make_combo_cache_key(decoded_model_name, normalized_combo_id)

    logs = _loading_logs.get(combo_key, [])
    is_loading = _loading_status.get(combo_key, {}).get("is_loading", False)
    
    # 调试信息
    print(f"📊 GET /circuit/loading_logs: combo_key={combo_key}, logs_count={len(logs)}, is_loading={is_loading}")

    return {
        "model_name": decoded_model_name,
        "sae_combo_id": normalized_combo_id,
        "logs": logs,
        "total_count": len(logs),
        "is_loading": is_loading,
    }



@app.post("/circuit_trace")
def circuit_trace(request: dict):
    """
    运行circuit trace分析并返回graph数据
    
    Args:
        request: 包含分析参数的请求体
            - fen: FEN字符串 (必需)
            - move_uci: 要分析的UCI移动 (必需)
            - side: 分析侧 (q/k/both, 默认: "k")
            - max_feature_nodes: 最大特征节点数 (默认: 4096)
            - node_threshold: 节点阈值 (默认: 0.73)
            - edge_threshold: 边阈值 (默认: 0.57)
            - max_n_logits: 最大logit数量 (默认: 1)
            - desired_logit_prob: 期望logit概率 (默认: 0.95)
            - batch_size: 批处理大小 (默认: 1)
            - order_mode: 排序模式 (positive/negative, 默认: "positive")
            - encoder_demean: 是否对encoder进行demean (默认: False)
            - save_activation_info: 是否保存激活信息 (默认: False)
    
    Returns:
        Graph数据 (JSON格式)
    """
    global _is_circuit_tracing
    
    try:
        # 检查circuits_service是否可用
        if not CIRCUITS_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Circuit tracing service not available")
        
        # 检查是否有正在进行的circuit tracing进程
        with _circuit_tracing_lock:
            if _is_circuit_tracing:
                raise HTTPException(status_code=409, detail="另一个circuit tracing进程正在进行中，请等待完成后再试")
            _is_circuit_tracing = True
        
        try:
            # 提取参数
            fen = request.get("fen")
            if not fen:
                raise HTTPException(status_code=400, detail="FEN string is required")
            
            # 解码FEN以确保trace_key的一致性
            fen = _decode_fen(fen)
            
            move_uci = request.get("move_uci")
            if move_uci:
                move_uci = _decode_fen(move_uci)  # move_uci也可能被编码
            negative_move_uci = request.get("negative_move_uci", None)  # 新增negative_move_uci参数
            if negative_move_uci:
                negative_move_uci = _decode_fen(negative_move_uci)  # negative_move_uci也可能被编码
            
            side = request.get("side", "k")
            max_feature_nodes = request.get("max_feature_nodes", 4096)
            node_threshold = request.get("node_threshold", 0.73)
            edge_threshold = request.get("edge_threshold", 0.57)
            max_n_logits = request.get("max_n_logits", 1)
            desired_logit_prob = request.get("desired_logit_prob", 0.95)
            batch_size = request.get("batch_size", 1)
            order_mode = request.get("order_mode", "positive")
            encoder_demean = request.get("encoder_demean", False)
            save_activation_info = request.get("save_activation_info", True)  # 默认启用激活信息保存
            max_act_times = request.get("max_act_times", None)  # 添加最大激活次数参数
            # 强制使用BT4模型
            model_name = "lc0/BT4-1024x15x32h"
            
            print(f"🔍 Circuit Trace 请求参数:")
            print(f"   - FEN: {fen}")
            print(f"   - Move UCI: {move_uci}")
            print(f"   - Negative Move UCI: {negative_move_uci}")
            print(f"   - Model Name: {model_name}")
            print(f"   - Side: {side}")
            print(f"   - Order Mode: {order_mode}")
            print(f"   - Max Act Times: {max_act_times}")
            
            # 验证 side 参数
            if side not in ["q", "k", "both"]:
                raise HTTPException(status_code=400, detail="side must be 'q', 'k', or 'both'")
            
            # 验证 order_mode 参数和处理both模式
            if order_mode == "both":
                # Both模式：需要positive move和negative move
                if not move_uci:
                    raise HTTPException(status_code=400, detail="move_uci (positive move) is required for 'both' mode")
                if not negative_move_uci:
                    raise HTTPException(status_code=400, detail="negative_move_uci is required for 'both' mode")
                # Both模式强制side为both
                side = "both"
                # 将order_mode转换为move_pair，以便后端处理
                order_mode = "move_pair"
            elif order_mode not in ["positive", "negative"]:
                raise HTTPException(status_code=400, detail="order_mode must be 'positive', 'negative', or 'both'")
            
            # 验证move_uci
            if not move_uci:
                raise HTTPException(status_code=400, detail="move_uci is required")
            
            # 获取已缓存的HookedTransformer模型
            hooked_model = get_hooked_model(model_name)
            
            # 检查是否有缓存的transcoders和lorsas
            cached_transcoders, cached_lorsas = get_cached_transcoders_and_lorsas(model_name)
            cached_replacement_model = _replacement_models_cache.get(model_name)
            
            # 检查是否正在加载
            global _loading_status, _loading_locks
            is_loading = _loading_status.get(model_name, {}).get("is_loading", False)
            
            # 如果缓存不完整且正在加载，等待加载完成
            cache_complete = (cached_transcoders is not None and cached_lorsas is not None and 
                             cached_replacement_model is not None and
                             len(cached_transcoders) == 15 and len(cached_lorsas) == 15)
            
            if not cache_complete and is_loading:
                print(f"⏳ 检测到正在加载TC/LoRSA，等待加载完成（避免重复加载）: {model_name}")
                # 获取加载锁（等待加载完成）
                if model_name not in _loading_locks:
                    _loading_locks[model_name] = threading.Lock()
                
                # 等待加载完成（最多等待10分钟，因为加载可能需要较长时间）
                max_wait_time = 600  # 10分钟
                wait_start = time.time()
                wait_interval = 1  # 每秒检查一次
                while (time.time() - wait_start) < max_wait_time:
                    is_loading = _loading_status.get(model_name, {}).get("is_loading", False)
                    # 重新检查缓存
                    cached_transcoders, cached_lorsas = get_cached_transcoders_and_lorsas(model_name)
                    cached_replacement_model = _replacement_models_cache.get(model_name)
                    cache_complete = (cached_transcoders is not None and cached_lorsas is not None and 
                                     cached_replacement_model is not None and
                                     len(cached_transcoders) == 15 and len(cached_lorsas) == 15)
                    if cache_complete:
                        print(f"✅ 等待加载完成，已获取完整缓存: {model_name} (等待时间: {time.time() - wait_start:.1f}秒)")
                        break
                    if not is_loading and not cache_complete:
                        # 加载已完成但缓存不完整，可能是加载失败
                        print(f"⚠️ 加载已完成但缓存不完整，可能需要重新加载: {model_name}")
                        break
                    time.sleep(wait_interval)
                    elapsed = time.time() - wait_start
                    if int(elapsed) % 10 == 0 and int(elapsed) > 0:  # 每10秒打印一次
                        print(f"⏳ 仍在等待加载完成... (已等待 {elapsed:.1f}秒, TC: {len(cached_transcoders) if cached_transcoders else 0}, LoRSA: {len(cached_lorsas) if cached_lorsas else 0})")
                
                if not cache_complete:
                    elapsed = time.time() - wait_start
                    if elapsed >= max_wait_time:
                        print(f"⚠️ 等待加载超时（{elapsed:.1f}秒），但将继续使用当前缓存或报错: {model_name}")
                    else:
                        print(f"⚠️ 加载完成但缓存不完整，将使用当前缓存或报错: {model_name}")
            
            # 获取当前使用的SAE组合ID（从请求中获取，如果没有则使用当前全局组合）
            sae_combo_id = request.get("sae_combo_id") or CURRENT_BT4_SAE_COMBO_ID
            combo_cfg = get_bt4_sae_combo(sae_combo_id)
            normalized_combo_id = combo_cfg["id"]
            
            # 根据组合ID设置正确的路径（即使使用缓存，也需要路径用于兼容性）
            if 'BT4' in model_name:
                tc_base_path = combo_cfg["tc_base_path"]
                lorsa_base_path = combo_cfg["lorsa_base_path"]
            else:
                raise HTTPException(status_code=400, detail="Unsupported Model!")
            
            # 使用组合ID获取正确的缓存（因为不同组合使用不同的缓存键）
            combo_key = _make_combo_cache_key(model_name, normalized_combo_id)
            cached_transcoders = _transcoders_cache.get(combo_key)
            cached_lorsas = _lorsas_cache.get(combo_key)
            cached_replacement_model = _replacement_models_cache.get(combo_key)
            
            # 重新检查缓存完整性
            cache_complete = (cached_transcoders is not None and cached_lorsas is not None and 
                             cached_replacement_model is not None and
                             len(cached_transcoders) == 15 and len(cached_lorsas) == 15)
            
            if cache_complete:
                # 使用缓存的transcoders和lorsas，不需要重新加载
                print(f"✅ 使用缓存的transcoders、lorsas和replacement_model: {model_name} @ {normalized_combo_id}")
            else:
                # 检查是否仍在加载
                is_still_loading = _loading_status.get(combo_key, {}).get("is_loading", False)
                if is_still_loading:
                    # 如果仍在加载，继续等待
                    print(f"⏳ 缓存不完整但仍在使用中加载，将继续等待...")
                    raise HTTPException(status_code=503, detail=f"模型 {model_name} 组合 {normalized_combo_id} 正在加载中，请稍后重试。当前进度: TC {len(cached_transcoders) if cached_transcoders else 0}/15, LoRSA {len(cached_lorsas) if cached_lorsas else 0}/15")
                elif cached_transcoders is None or cached_lorsas is None:
                    # 完全没有缓存，需要加载
                    print(f"⚠️ 未找到缓存，将重新加载transcoders和lorsas: {model_name} @ {normalized_combo_id}")
                    print("   提示：建议先调用 /circuit/preload_models 进行预加载以加速")
                else:
                    # 有部分缓存但不完整，也重新加载（这种情况不应该发生，因为应该等待加载完成）
                    print(f"⚠️ 缓存不完整（TC: {len(cached_transcoders)}, LoRSA: {len(cached_lorsas)}），将重新加载: {model_name} @ {normalized_combo_id}")
            
            # 创建trace_key用于日志存储（确保使用解码后的FEN和move_uci）
            # fen和move_uci已经在前面被解码了
            trace_key = f"{model_name}::{normalized_combo_id}::{fen}::{move_uci}"
            
            # 初始化日志列表
            if trace_key not in _circuit_trace_logs:
                _circuit_trace_logs[trace_key] = []
            trace_logs = _circuit_trace_logs[trace_key]
            trace_logs.clear()  # 清空之前的日志
            
            # 设置tracing状态
            _circuit_trace_status[trace_key] = {"is_tracing": True}
            
            # 添加初始日志
            trace_logs.append({
                "timestamp": time.time(),
                "message": f"🔍 开始Circuit Trace: FEN={fen}, Move={move_uci}, Side={side}, OrderMode={order_mode}"
            })
            
            try:
                # 运行circuit trace，传递已缓存的模型和transcoders/lorsas以及日志列表
                graph_data = run_circuit_trace(
                    prompt=fen,
                    move_uci=move_uci,
                    negative_move_uci=negative_move_uci,  # 传递negative_move_uci
                    model_name=model_name,  # 添加模型名称参数
                    tc_base_path=tc_base_path,  # 传递正确的TC路径
                    lorsa_base_path=lorsa_base_path,  # 传递正确的LORSA路径
                    side=side,
                    max_feature_nodes=max_feature_nodes,
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                    max_n_logits=max_n_logits,
                    desired_logit_prob=desired_logit_prob,
                    batch_size=batch_size,
                    order_mode=order_mode,
                    encoder_demean=encoder_demean,
                    save_activation_info=save_activation_info,
                    act_times_max=max_act_times,  # 传递最大激活次数参数
                    log_level="INFO",
                    hooked_model=hooked_model,  # 传递已缓存的模型
                    cached_transcoders=cached_transcoders,  # 传递缓存的transcoders
                    cached_lorsas=cached_lorsas,  # 传递缓存的lorsas
                    cached_replacement_model=cached_replacement_model,  # 传递缓存的replacement_model
                    sae_combo_id=normalized_combo_id,  # 传递归一化后的SAE组合ID，用于生成正确的analysis_name模板
                    trace_logs=trace_logs  # 传递日志列表
                )
                
                # 添加完成日志
                finished_ts = time.time()
                trace_logs.append({
                    "timestamp": finished_ts,
                    "message": "✅ Circuit Trace完成!"
                })

                result_data = {
                    "graph_data": graph_data,
                    "finished_at": finished_ts,
                    "logs": list(trace_logs),
                }
                
                # 保存到内存
                _circuit_trace_results[trace_key] = result_data
                
                # 持久化到磁盘（确保即使服务器重启也能恢复）
                try:
                    _save_trace_result_to_disk(trace_key, result_data)
                except Exception as e:
                    print(f"⚠️ 持久化trace结果失败（但结果已保存在内存中）: {e}")
                
            except Exception as trace_error:
                # 即使trace失败，也尝试保存部分结果（如果有的话）
                print(f"⚠️ Circuit trace过程中出现异常: {trace_error}")
                # 如果有部分结果，尝试保存
                if trace_logs:
                    try:
                        partial_result = {
                            "graph_data": None,
                            "finished_at": time.time(),
                            "logs": list(trace_logs),
                            "error": str(trace_error)
                        }
                        _circuit_trace_results[trace_key] = partial_result
                        _save_trace_result_to_disk(trace_key, partial_result)
                    except:
                        pass
                # 重新抛出异常
                raise
            finally:
                # 更新tracing状态
                _circuit_trace_status[trace_key] = {"is_tracing": False}
            
            return graph_data
        
        finally:
            # 无论成功还是失败，都要清除标志
            with _circuit_tracing_lock:
                _is_circuit_tracing = False
        
    except HTTPException:
        # HTTPException需要重新抛出（标志已在finally中清除）
        raise
    except Exception as e:
        # 其他异常转换为HTTPException（标志已在finally中清除）
        raise HTTPException(status_code=500, detail=f"Circuit trace analysis failed: {str(e)}")


@app.get("/circuit_trace/status")
def circuit_trace_status():
    """检查circuit trace服务的状态"""
    global _is_circuit_tracing
    return {
        "available": CIRCUITS_SERVICE_AVAILABLE,
        "hooked_transformer_available": HOOKED_TRANSFORMER_AVAILABLE,
        "is_tracing": _is_circuit_tracing
    }


@app.get("/circuit_trace/result")
def circuit_trace_result(
    model_name: str = "lc0/BT4-1024x15x32h",
    sae_combo_id: str | None = None,
    fen: str | None = None,
    move_uci: str | None = None,
):
    """
    获取最近一次完成的circuit trace结果
    如果内存中没有，会尝试从磁盘加载
    """
    global _circuit_trace_results

    if fen and move_uci:
        # 解码FEN和move_uci以确保trace_key的一致性
        decoded_fen = _decode_fen(fen)
        decoded_move_uci = _decode_fen(move_uci)
        decoded_model_name = _decode_fen(model_name)
        
        combo_id = sae_combo_id or CURRENT_BT4_SAE_COMBO_ID
        combo_cfg = get_bt4_sae_combo(combo_id)
        normalized_combo_id = combo_cfg["id"]
        trace_key = f"{decoded_model_name}::{normalized_combo_id}::{decoded_fen}::{decoded_move_uci}"
        
        # 先尝试从内存加载
        result = _circuit_trace_results.get(trace_key)
        
        # 如果内存中没有，尝试从磁盘加载
        if not result:
            print(f"🔍 内存中未找到trace结果，尝试从磁盘加载: {trace_key}")
            disk_result = _load_trace_result_from_disk(trace_key)
            if disk_result:
                # 加载到内存中以便后续快速访问
                _circuit_trace_results[trace_key] = disk_result
                result = disk_result
                print(f"✅ 成功从磁盘恢复trace结果: {trace_key}")
    else:
        # 如果没有提供fen和move_uci，返回最近的结果
        latest_key = None
        latest_ts = -1
        for key, payload in _circuit_trace_results.items():
            ts = payload.get("finished_at", 0)
            if ts > latest_ts:
                latest_ts = ts
                latest_key = key
        
        result = _circuit_trace_results.get(latest_key) if latest_key else None
        
        # 如果内存中没有，尝试从磁盘查找最新的
        if not result:
            print("🔍 内存中未找到最近的trace结果，尝试从磁盘查找...")
            # 遍历磁盘上的所有trace文件，找到最新的
            if TRACE_RESULTS_DIR.exists():
                latest_disk_result = None
                latest_disk_ts = -1
                latest_trace_key = None
                for storage_file in TRACE_RESULTS_DIR.glob("*.json"):
                    try:
                        with open(storage_file, "r", encoding="utf-8") as f:
                            save_data = json.load(f)
                        saved_trace_key = save_data.get("trace_key")
                        disk_result = save_data.get("result")
                        if disk_result:
                            ts = disk_result.get("finished_at", 0)
                            if ts > latest_disk_ts:
                                latest_disk_ts = ts
                                latest_disk_result = disk_result
                                latest_trace_key = saved_trace_key
                    except Exception as e:
                        print(f"⚠️ 加载trace文件失败 {storage_file}: {e}")
                        continue
                
                if latest_disk_result and latest_trace_key:
                    # 加载到内存中
                    _circuit_trace_results[latest_trace_key] = latest_disk_result
                    result = latest_disk_result
                    print(f"✅ 成功从磁盘恢复最新的trace结果: {latest_trace_key}")

    if not result:
        raise HTTPException(status_code=404, detail="未找到trace结果")

    return result


@app.get("/circuit_trace/logs")
def get_circuit_trace_logs(
    model_name: str = "lc0/BT4-1024x15x32h",
    sae_combo_id: str | None = None,
    fen: str | None = None,
    move_uci: str | None = None,
):
    """
    获取circuit tracing的日志
    
    Args:
        model_name: 模型名称 (查询参数，默认: "lc0/BT4-1024x15x32h")
        sae_combo_id: SAE组合ID (查询参数，可选)
        fen: FEN字符串 (查询参数，可选)
        move_uci: UCI移动 (查询参数，可选)
    
    Returns:
        Circuit tracing日志列表
    """
    global _circuit_trace_logs, _circuit_trace_status
    
    # 如果提供了所有参数，使用精确匹配
    if fen and move_uci:
        # 解码FEN和move_uci以确保trace_key的一致性
        decoded_fen = _decode_fen(fen)
        decoded_move_uci = _decode_fen(move_uci)
        decoded_model_name = _decode_fen(model_name)
        
        combo_id = sae_combo_id or CURRENT_BT4_SAE_COMBO_ID
        combo_cfg = get_bt4_sae_combo(combo_id)
        normalized_combo_id = combo_cfg["id"]
        trace_key = f"{decoded_model_name}::{normalized_combo_id}::{decoded_fen}::{decoded_move_uci}"
        logs = _circuit_trace_logs.get(trace_key, [])
        is_tracing = _circuit_trace_status.get(trace_key, {}).get("is_tracing", False)
    else:
        # 否则返回最近的日志（按时间戳排序）
        all_logs = []
        for trace_key, log_list in _circuit_trace_logs.items():
            if log_list:
                # 获取最后一条日志的时间戳
                last_log_time = log_list[-1]["timestamp"] if log_list else 0
                all_logs.append((last_log_time, trace_key, log_list))
        
        # 按时间戳降序排序
        all_logs.sort(key=lambda x: x[0], reverse=True)
        
        # 返回最近一条trace的日志
        if all_logs:
            _, trace_key, logs = all_logs[0]
            is_tracing = _circuit_trace_status.get(trace_key, {}).get("is_tracing", False)
        else:
            logs = []
            is_tracing = False
    
    return {
        "model_name": model_name,
        "sae_combo_id": sae_combo_id or CURRENT_BT4_SAE_COMBO_ID,
        "logs": logs,
        "total_count": len(logs),
        "is_tracing": is_tracing,
    }


@app.post("/circuit/check_dense_features")
def check_dense_features_api(request: dict):
    """
    检查circuit中哪些节点是dense feature（激活次数超过阈值）
    
    Args:
        request: 包含检查参数的请求体
            - nodes: 节点列表
            - threshold: 激活次数阈值（可选，None表示无限大）
            - sae_series: SAE系列名称（可选，默认: BT4-exp128）
            - lorsa_analysis_name: LoRSA分析名称模板（可选）
            - tc_analysis_name: TC分析名称模板（可选）
    
    Returns:
        dense节点的ID列表
    """
    try:
        # 检查circuits_service是否可用
        if not CIRCUITS_SERVICE_AVAILABLE or check_dense_features is None:
            raise HTTPException(status_code=503, detail="Dense feature check service not available")
        
        # 提取参数
        nodes = request.get("nodes", [])
        if not isinstance(nodes, list):
            raise HTTPException(status_code=400, detail="nodes must be a list")
        
        threshold = request.get("threshold")
        if threshold is not None:
            try:
                threshold = int(threshold)
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail="threshold must be an integer or null")
        
        sae_series = request.get("sae_series", "BT4-exp128")
        lorsa_analysis_name = request.get("lorsa_analysis_name")
        tc_analysis_name = request.get("tc_analysis_name")
        
        print(f"🔍 检查dense features: {len(nodes)} 个节点, 阈值={threshold}")
        print(f"   - LoRSA模板: {lorsa_analysis_name}")
        print(f"   - TC模板: {tc_analysis_name}")
        
        # 设置MongoDB连接
        mongo_config = MongoDBConfig()
        mongo_client_instance = MongoClient(mongo_config)
        
        # 调用检查函数
        dense_node_ids = check_dense_features(
            nodes=nodes,
            threshold=threshold,
            mongo_client=mongo_client_instance,
            sae_series=sae_series,
            lorsa_analysis_name=lorsa_analysis_name,
            tc_analysis_name=tc_analysis_name
        )
        
        print(f"✅ 找到 {len(dense_node_ids)} 个dense节点")
        
        return {
            "dense_nodes": dense_node_ids,
            "total_nodes": len(nodes),
            "threshold": threshold
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Dense feature check failed: {str(e)}")


@app.post("/patching_analysis")
def patching_analysis(request: dict):
    """
    运行patching分析并返回Token Predictions结果
    
    Args:
        request: 包含分析参数的请求体
            - fen: FEN字符串 (必需)
            - feature_type: 特征类型 ('transcoder' 或 'lorsa') (必需)
            - layer: 层数 (必需)
            - pos: 位置 (必需)
            - feature: 特征索引 (必需)
    
    Returns:
        Token Predictions分析结果 (JSON格式)
    """
    try:
        # 检查patching服务是否可用
        if not PATCHING_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Patching service not available")
        
        # 提取参数
        fen = request.get("fen")
        if not fen:
            raise HTTPException(status_code=400, detail="FEN string is required")
        
        feature_type = request.get("feature_type")
        if feature_type not in ['transcoder', 'lorsa']:
            raise HTTPException(status_code=400, detail="feature_type must be 'transcoder' or 'lorsa'")
        
        layer = request.get("layer")
        if layer is None or not isinstance(layer, int):
            raise HTTPException(status_code=400, detail="layer must be an integer")
        
        pos = request.get("pos")
        if pos is None or not isinstance(pos, int):
            raise HTTPException(status_code=400, detail="pos must be an integer")
        
        feature = request.get("feature")
        if feature is None or not isinstance(feature, int):
            raise HTTPException(status_code=400, detail="feature must be an integer")
        
        print(f"🔍 运行patching分析: {feature_type} L{layer} pos{pos} feature{feature}")
        
        # 运行patching分析
        result = run_patching_analysis(
            fen=fen,
            feature_type=feature_type,
            layer=layer,
            pos=pos,
            feature=feature
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        print(f"✅ Patching分析完成，找到 {result['statistics']['total_legal_moves']} 个合法移动")
        
        return result
        
    except Exception as e:
        print(f"❌ Patching分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"Patching analysis failed: {str(e)}")


@app.get("/patching_analysis/status")
def patching_analysis_status():
    """检查patching分析服务的状态"""
    return {
        "available": PATCHING_SERVICE_AVAILABLE,
        "hooked_transformer_available": HOOKED_TRANSFORMER_AVAILABLE
    }


@app.post("/steering_analysis")
def steering_analysis(request: dict):
    """
    运行steering分析并返回Token Predictions结果，支持可调的steering_scale
    
    Args:
        request: 包含分析参数的请求体
            - fen: FEN字符串 (必需)
            - feature_type: 特征类型 ('transcoder' 或 'lorsa') (必需)
            - layer: 层数 (必需)
            - pos: 位置 (必需)
            - feature: 特征索引 (必需)
            - steering_scale: 放大系数 (可选，默认 1)
    
    Returns:
        Token Predictions分析结果 (JSON格式)
    """
    try:
        if not INTERVENTION_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Steering service not available")

        fen = request.get("fen")
        if not fen:
            raise HTTPException(status_code=400, detail="FEN string is required")

        feature_type = request.get("feature_type")
        if feature_type not in ['transcoder', 'lorsa']:
            raise HTTPException(status_code=400, detail="feature_type must be 'transcoder' or 'lorsa'")

        layer = request.get("layer")
        if layer is None or not isinstance(layer, int):
            raise HTTPException(status_code=400, detail="layer must be an integer")

        pos = request.get("pos")
        if pos is None or not isinstance(pos, int):
            raise HTTPException(status_code=400, detail="pos must be an integer")

        feature = request.get("feature")
        if feature is None or not isinstance(feature, int):
            raise HTTPException(status_code=400, detail="feature must be an integer")

        steering_scale = request.get("steering_scale", 1)
        if not isinstance(steering_scale, (int, float)):
            raise HTTPException(status_code=400, detail="steering_scale must be a number")

        # 获取metadata信息
        metadata = request.get("metadata", {})

        print(f"🔍 运行steering分析: {feature_type} L{layer} pos{pos} feature{feature} scale{steering_scale}")
        print(f"📋 Metadata: {metadata}")

        result = run_feature_steering_analysis(
            fen=fen,
            feature_type=feature_type,
            layer=layer,
            pos=pos,
            feature=feature,
            steering_scale=steering_scale,
            metadata=metadata
        )

        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])

        print(f"✅ Steering分析完成，找到 {result['statistics']['total_legal_moves']} 个合法移动")
        return result

    except Exception as e:
        print(f"❌ Steering分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"Steering analysis failed: {str(e)}")


@app.get("/steering_analysis/status")
def steering_analysis_status():
    """检查steering分析服务的状态"""
    return {
        "available": INTERVENTION_SERVICE_AVAILABLE,
        "hooked_transformer_available": HOOKED_TRANSFORMER_AVAILABLE
    }


@app.post("/self_play")
def start_self_play(request: dict):
    """
    开始自对弈并返回游戏数据
    
    Args:
        request: 包含游戏参数的请求体
            - initial_fen: 初始FEN字符串 (可选，默认起始局面)
            - max_moves: 最大移动数 (默认: 10)
            - temperature: 温度参数 (默认: 1.0)
    
    Returns:
        自对弈游戏数据 (JSON格式)
    """
    try:
        # 检查自对弈服务是否可用
        if not SELF_PLAY_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Self-play service not available")
        
        # 提取参数
        initial_fen = request.get("initial_fen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        max_moves = request.get("max_moves", 10)
        temperature = request.get("temperature", 1.0)
        
        # 验证参数
        if not isinstance(max_moves, int) or max_moves <= 0:
            raise HTTPException(status_code=400, detail="max_moves must be a positive integer")
        
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise HTTPException(status_code=400, detail="temperature must be a non-negative number")
        
        print(f"🎮 开始自对弈: {initial_fen[:50]}..., 最大移动数: {max_moves}, 温度: {temperature}")
        
        # 强制使用BT4模型
        model_name = "lc0/BT4-1024x15x32h"
        hooked_model = get_hooked_model(model_name)
        
        # 运行自对弈
        game_result = run_self_play(
            initial_fen=initial_fen,
            max_moves=max_moves,
            temperature=temperature,
            model=hooked_model
        )
        
        print(f"✅ 自对弈完成，共进行了 {len(game_result['moves'])} 步")
        
        return game_result
        
    except Exception as e:
        print(f"❌ 自对弈失败: {e}")
        raise HTTPException(status_code=500, detail=f"Self-play failed: {str(e)}")


@app.post("/self_play/analyze")
def analyze_self_play_positions(request: dict):
    """
    分析自对弈中的位置序列
    
    Args:
        request: 包含位置序列的请求体
            - positions: FEN字符串列表
    
    Returns:
        位置分析结果 (JSON格式)
    """
    try:
        # 检查自对弈服务是否可用
        if not SELF_PLAY_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Self-play service not available")
        
        # 提取参数
        positions = request.get("positions", [])
        
        if not isinstance(positions, list) or not positions:
            raise HTTPException(status_code=400, detail="positions must be a non-empty list of FEN strings")
        
        print(f"🔍 分析位置序列，共 {len(positions)} 个位置")
        
        # 获取已缓存的HookedTransformer模型
        hooked_model = get_hooked_model()
        
        # 分析位置序列
        analysis_result = analyze_game_positions(
            positions=positions,
            model=hooked_model
        )
        
        print(f"✅ 位置分析完成")
        
        return {
            "positions_analysis": analysis_result,
            "total_positions": len(positions)
        }
        
    except Exception as e:
        print(f"❌ 位置分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"Position analysis failed: {str(e)}")


@app.get("/self_play/status")
def self_play_status():
    """检查自对弈服务的状态"""
    return {
        "available": SELF_PLAY_SERVICE_AVAILABLE,
        "hooked_transformer_available": HOOKED_TRANSFORMER_AVAILABLE
    }


@app.post("/logit_lens/analyze")
def logit_lens_analyze(request: dict):
    """
    运行Logit Lens分析
    
    Args:
        request: 包含分析参数的请求体
            - fen: FEN字符串 (必需)
            - target_move: 目标移动UCI (可选)
            - topk_vocab: 考虑的顶部词汇数量 (可选，默认: 2000)
    
    Returns:
        Logit Lens分析结果 (JSON格式)
    """
    try:
        # 检查Logit Lens服务是否可用
        if not LOGIT_LENS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Logit Lens service not available")
        
        # 提取参数
        fen = request.get("fen")
        if not fen:
            raise HTTPException(status_code=400, detail="FEN string is required")
        
        # 强制使用BT4模型
        model_name = "lc0/BT4-1024x15x32h"
        target_move = request.get("target_move")
        topk_vocab = request.get("topk_vocab", 2000)
        
        print(f"🔍 运行Logit Lens分析: FEN={fen[:50]}..., model={model_name}, target={target_move}")
        
        # 获取或创建Logit Lens实例
        global _logit_lens_instances
        if model_name not in _logit_lens_instances:
            # 获取模型
            hooked_model = get_hooked_model(model_name)
            # 创建Logit Lens实例
            _logit_lens_instances[model_name] = IntegratedPolicyLens(hooked_model)
        
        lens = _logit_lens_instances[model_name]
        
        # 运行分析
        result = lens.analyze_single_fen(
            fen=fen,
            target_move=target_move,
            topk_vocab=topk_vocab
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        print(f"✅ Logit Lens分析完成，分析了 {result['num_layers']} 层")
        
        return {
            **result,
            "model_used": model_name
        }
        
    except Exception as e:
        print(f"❌ Logit Lens分析失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Logit Lens analysis failed: {str(e)}")


@app.get("/logit_lens/status")
def logit_lens_status():
    """检查Logit Lens服务的状态"""
    return {
        "available": LOGIT_LENS_AVAILABLE,
        "hooked_transformer_available": HOOKED_TRANSFORMER_AVAILABLE
    }


@app.post("/logit_lens/mean_ablation")
def logit_lens_mean_ablation(request: dict):
    """
    运行Mean Ablation分析
    
    Args:
        request: 包含分析参数的请求体
            - fen: FEN字符串 (必需)
            - hook_types: hook类型列表 (可选，默认: ['attn_out', 'mlp_out'])
            - target_move: 目标移动UCI (可选)
            - topk_vocab: 考虑的顶部词汇数量 (可选，默认: 2000)
    
    Returns:
        Mean Ablation分析结果 (JSON格式)
    """
    try:
        # 检查Logit Lens服务是否可用
        if not LOGIT_LENS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Logit Lens service not available")
        
        # 提取参数
        fen = request.get("fen")
        if not fen:
            raise HTTPException(status_code=400, detail="FEN string is required")
        
        # 强制使用BT4模型
        model_name = "lc0/BT4-1024x15x32h"
        hook_types = request.get("hook_types", ['attn_out', 'mlp_out'])
        target_move = request.get("target_move")
        topk_vocab = request.get("topk_vocab", 2000)
        
        print(f"🔍 运行Mean Ablation分析: FEN={fen[:50]}..., model={model_name}, hooks={hook_types}, target={target_move}")
        
        # 获取或创建Logit Lens实例
        global _logit_lens_instances
        if model_name not in _logit_lens_instances:
            # 获取模型
            hooked_model = get_hooked_model(model_name)
            # 创建Logit Lens实例
            _logit_lens_instances[model_name] = IntegratedPolicyLens(hooked_model)
        
        lens = _logit_lens_instances[model_name]
        
        # 运行Mean Ablation分析
        result = lens.analyze_mean_ablation(
            fen=fen,
            hook_types=hook_types,
            target_move=target_move,
            topk_vocab=topk_vocab
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        print(f"✅ Mean Ablation分析完成，分析了 {result['num_layers']} 层，{len(result['hook_types'])} 种hook类型")
        
        return {
            **result,
            "model_used": model_name
        }
        
    except Exception as e:
        print(f"❌ Mean Ablation分析失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Mean Ablation analysis failed: {str(e)}")


# 新增：走法评测接口（基于Stockfish）
@app.post("/evaluate_move")
def evaluate_move(request: dict):
    """
    评测一次移动：输入上一步之前的FEN与该步UCI，返回0-100评分、cp差、WDL等。

    body: { "fen": str, "move": str, "time_limit": float? }
    """
    fen = request.get("fen")
    move = request.get("move")
    time_limit = request.get("time_limit", 0.2)
    if not fen or not move:
        raise HTTPException(status_code=400, detail="fen与move必填")
    try:
        _ = chess.Board(fen)
    except Exception:
        raise HTTPException(status_code=400, detail="无效的FEN")

    res = evaluate_move_quality(fen, move, time_limit=time_limit)
    if res is None:
        raise HTTPException(status_code=400, detail="评测失败或走法不合法")
    return res


# 战术特征分析接口
@app.post("/tactic_features/analyze")
async def analyze_tactic_features_api(
    file: UploadFile = File(...),
    n_random: int = Form(200),
    n_fens: int = Form(200),
    top_k_lorsa: int = Form(10),
    top_k_tc: int = Form(10),
    specific_layer: Optional[str] = Form(None),
    specific_layer_top_k: int = Form(20),
):
    """
    分析战术特征：上传FEN文件，与随机FEN比较，找出最相关的特征
    
    Args:
        file: 上传的txt文件，每行一个FEN
        model_name: 模型名称
        n_random: 随机FEN数量（兼容旧参数）
        n_fens: FEN数量（新参数，优先使用）
        top_k_lorsa: 显示top k个LoRSA特征
        top_k_tc: 显示top k个TC特征
        specific_layer: 指定层号（可选），如果提供则额外返回该层的详细特征
        specific_layer_top_k: 指定层的top k特征数
    """
    if not TACTIC_FEATURES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tactic features analysis not available")
    
    if not HOOKED_TRANSFORMER_AVAILABLE:
        raise HTTPException(status_code=503, detail="HookedTransformer not available")
    
    try:
        # 强制使用BT4模型
        model_name = "lc0/BT4-1024x15x32h"
        
        # ========== 调试信息：函数开始 ==========
        print("=" * 80)
        print("🚀 开始处理战术特征分析请求")
        print(f"📥 接收到的原始参数:")
        print(f"   - model_name: {model_name} (强制使用BT4)")
        print(f"   - n_random: {n_random}")
        print(f"   - n_fens: {n_fens}")
        print(f"   - top_k_lorsa: {top_k_lorsa}")
        print(f"   - top_k_tc: {top_k_tc}")
        print(f"   - specific_layer (原始): {specific_layer} (类型: {type(specific_layer)})")
        print(f"   - specific_layer_top_k: {specific_layer_top_k}")
        print("=" * 80)
        
        # 解析specific_layer参数
        parsed_specific_layer = None
        print(f"🔍 开始解析 specific_layer 参数...")
        print(f"   - specific_layer is None: {specific_layer is None}")
        if specific_layer is not None:
            print(f"   - specific_layer 值: '{specific_layer}'")
            print(f"   - specific_layer.strip() 后: '{specific_layer.strip() if isinstance(specific_layer, str) else specific_layer}'")
        
        if specific_layer is not None and isinstance(specific_layer, str) and specific_layer.strip():
            try:
                parsed_specific_layer = int(specific_layer.strip())
                print(f"✅ 成功解析指定层参数: {parsed_specific_layer} (原始值: '{specific_layer}')")
            except (ValueError, TypeError) as e:
                print(f"❌ 解析层号参数失败: {e}")
                print(f"⚠️ 无效的层号参数: '{specific_layer}'，将忽略指定层分析")
                parsed_specific_layer = None
        elif specific_layer is None:
            print(f"ℹ️ 未提供 specific_layer 参数，将不进行指定层分析")
        else:
            print(f"⚠️ specific_layer 参数为空字符串或无效，将忽略")
        
        # 使用n_fens参数（如果提供），否则使用n_random
        actual_n_fens = n_fens if n_fens != 200 or n_random == 200 else n_random
        print(f"📊 实际使用的FEN数量: {actual_n_fens}")
        
        print(f"🎯 最终解析结果:")
        print(f"   - parsed_specific_layer: {parsed_specific_layer}")
        print(f"   - specific_layer_top_k: {specific_layer_top_k}")
        print(f"   - actual_n_fens: {actual_n_fens}")
        if parsed_specific_layer is not None:
            print(f"✅ 将分析指定层: Layer {parsed_specific_layer}")
        else:
            print(f"ℹ️ 不进行指定层分析")
        print("=" * 80)
        
        # 读取文件内容
        contents = await file.read()
        text = contents.decode('utf-8')
        tactic_fens = [line.strip() for line in text.strip().split('\n') if line.strip()]
        
        if not tactic_fens:
            raise HTTPException(status_code=400, detail="文件为空或没有有效的FEN行")
        
        # 验证FEN格式
        valid_fens, invalid_fens = validate_fens(tactic_fens)
        
        # 限制FEN数量：如果文件中的FEN多于设置的数量，取前n条；否则全部使用
        if len(valid_fens) > actual_n_fens:
            print(f"📊 文件中有 {len(valid_fens)} 个有效FEN，取前 {actual_n_fens} 个")
            valid_fens = valid_fens[:actual_n_fens]
        else:
            print(f"📊 文件中有 {len(valid_fens)} 个有效FEN，全部使用")
        
        if len(valid_fens) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"没有有效的FEN字符串。无效FEN示例: {invalid_fens[:5]}"
            )
        
        # 加载模型（使用缓存）
        hooked_model = get_hooked_model(model_name)
        
        # 检查缓存的transcoders和lorsas
        cached_transcoders, cached_lorsas = get_cached_transcoders_and_lorsas(model_name)
        
        num_layers = 15
        if cached_transcoders is not None and cached_lorsas is not None:
            if len(cached_transcoders) == num_layers and len(cached_lorsas) == num_layers:
                print(f"✅ 使用缓存的transcoders和lorsas: {model_name}")
                transcoders = cached_transcoders
                lorsas = cached_lorsas
            else:
                # 缓存不完整，需要加载
                print(f"⚠️ 缓存不完整，重新加载: {model_name}")
                transcoders = None
                lorsas = None
        else:
            transcoders = None
            lorsas = None
        
        # 如果缓存不可用，则加载
        if transcoders is None or lorsas is None:
            if 'BT4' in model_name:
                tc_base_path = BT4_TC_BASE_PATH
                lorsa_base_path = BT4_LORSA_BASE_PATH
            else:
                raise ValueError("Unsupported Model!")
            
            transcoders = {}
            lorsas = []
            
            for layer in range(num_layers):
                # 加载Transcoder
                tc_path = f"{tc_base_path}/L{layer}"
                if os.path.exists(tc_path):
                    transcoders[layer] = SparseAutoEncoder.from_pretrained(
                        tc_path,
                        dtype=torch.float32,
                        device=device,
                    )
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Transcoder not found at {tc_path}"
                    )
                
                # 加载LoRSA
                lorsa_path = f"{lorsa_base_path}/L{layer}"
                if os.path.exists(lorsa_path):
                    lorsas.append(LowRankSparseAttention.from_pretrained(
                        lorsa_path,
                        device=device,
                    ))
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"LoRSA not found at {lorsa_path}"
                    )
            
            # 缓存加载的transcoders和lorsas
            if CIRCUITS_SERVICE_AVAILABLE and set_cached_models is not None:
                # 需要创建replacement_model才能缓存，这里先缓存transcoders和lorsas
                _global_transcoders_cache[model_name] = transcoders
                _global_lorsas_cache[model_name] = lorsas
                _global_hooked_models[model_name] = hooked_model
        
        # 执行分析
        print("=" * 80)
        print(f"🔬 开始执行特征分析")
        print(f"   - 战术FEN数量: {len(valid_fens)}条")
        print(f"   - 随机FEN数量: {actual_n_fens}条")
        print(f"   - 模型层数: {num_layers}层 (0-{num_layers-1})")
        if parsed_specific_layer is not None:
            print(f"   ✅ 指定层分析已启用:")
            print(f"      - 层号: Layer {parsed_specific_layer}")
            print(f"      - Top K: {specific_layer_top_k}")
            if parsed_specific_layer < 0 or parsed_specific_layer >= num_layers:
                print(f"      ⚠️ 警告: 层号 {parsed_specific_layer} 超出有效范围!")
        else:
            print(f"   ℹ️ 未指定层，将只返回所有层的Top K特征")
        print("=" * 80)
        
        result = analyze_tactic_features(
            tactic_fens=valid_fens,
            model=hooked_model,
            lorsas=lorsas,
            transcoders=transcoders,
            n_random=actual_n_fens,
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # 排序并取top k
        lorsa_diffs = sorted(result["lorsa_diffs"], key=lambda x: x[2], reverse=True)[:top_k_lorsa]
        tc_diffs = sorted(result["tc_diffs"], key=lambda x: x[2], reverse=True)[:top_k_tc]
        
        # 格式化结果
        def format_diff(diff_tuple):
            layer, feature, diff, p_random, p_tactic, kind = diff_tuple
            return {
                "layer": layer,
                "feature": feature,
                "diff": float(diff),
                "p_random": float(p_random),
                "p_tactic": float(p_tactic),
                "kind": kind
            }
        
        response_data = {
            "valid_tactic_fens": result["valid_tactic_fens"],
            "invalid_tactic_fens": result["invalid_tactic_fens"],
            "random_fens": result["random_fens"],
            "tactic_fens": result["tactic_fens"],
            "top_lorsa_features": [format_diff(d) for d in lorsa_diffs],
            "top_tc_features": [format_diff(d) for d in tc_diffs],
            "invalid_fens_sample": result.get("invalid_fens_list", [])
        }
        
        # 如果指定了层号，返回该层的详细特征
        print("=" * 80)
        print(f"🔍 检查是否需要返回指定层特征...")
        print(f"   - parsed_specific_layer: {parsed_specific_layer}")
        print(f"   - num_layers: {num_layers}")
        print(f"   - 条件检查: parsed_specific_layer is not None = {parsed_specific_layer is not None}")
        if parsed_specific_layer is not None:
            print(f"   - 条件检查: 0 <= {parsed_specific_layer} < {num_layers} = {0 <= parsed_specific_layer < num_layers}")
        
        if parsed_specific_layer is not None and 0 <= parsed_specific_layer < num_layers:
            print(f"✅ 开始筛选 Layer {parsed_specific_layer} 的特征...")
            
            # 打印所有特征的总数（用于调试）
            total_lorsa_diffs = len(result["lorsa_diffs"])
            total_tc_diffs = len(result["tc_diffs"])
            print(f"   - 总LoRSA特征数: {total_lorsa_diffs}")
            print(f"   - 总TC特征数: {total_tc_diffs}")
            
            # 筛选出指定层的特征
            specific_lorsa = [d for d in result["lorsa_diffs"] if d[0] == parsed_specific_layer]
            specific_tc = [d for d in result["tc_diffs"] if d[0] == parsed_specific_layer]
            
            print(f"📊 Layer {parsed_specific_layer} 特征统计:")
            print(f"   - LoRSA特征: {len(specific_lorsa)}个")
            print(f"   - TC特征: {len(specific_tc)}个")
            
            if len(specific_lorsa) == 0:
                print(f"   ⚠️ 警告: Layer {parsed_specific_layer} 没有找到任何 LoRSA 特征!")
            if len(specific_tc) == 0:
                print(f"   ⚠️ 警告: Layer {parsed_specific_layer} 没有找到任何 TC 特征!")
            
            # 排序并取top k
            specific_lorsa_sorted = sorted(specific_lorsa, key=lambda x: x[2], reverse=True)[:specific_layer_top_k]
            specific_tc_sorted = sorted(specific_tc, key=lambda x: x[2], reverse=True)[:specific_layer_top_k]
            
            print(f"   - 排序后取Top {specific_layer_top_k}:")
            print(f"     * LoRSA: {len(specific_lorsa_sorted)}个")
            print(f"     * TC: {len(specific_tc_sorted)}个")
            
            # 打印前3个特征的详细信息（用于调试）
            if len(specific_lorsa_sorted) > 0:
                print(f"   - LoRSA Top 3 特征示例:")
                for i, feat in enumerate(specific_lorsa_sorted[:3]):
                    print(f"     [{i+1}] Layer={feat[0]}, Feature={feat[1]}, Diff={feat[2]:.6f}")
            
            if len(specific_tc_sorted) > 0:
                print(f"   - TC Top 3 特征示例:")
                for i, feat in enumerate(specific_tc_sorted[:3]):
                    print(f"     [{i+1}] Layer={feat[0]}, Feature={feat[1]}, Diff={feat[2]:.6f}")
            
            response_data["specific_layer"] = parsed_specific_layer
            response_data["specific_layer_lorsa"] = [format_diff(d) for d in specific_lorsa_sorted]
            response_data["specific_layer_tc"] = [format_diff(d) for d in specific_tc_sorted]
            
            print(f"✅ 已添加指定层特征到响应数据:")
            print(f"   - specific_layer: {response_data.get('specific_layer')}")
            print(f"   - specific_layer_lorsa: {len(response_data.get('specific_layer_lorsa', []))}个")
            print(f"   - specific_layer_tc: {len(response_data.get('specific_layer_tc', []))}个")
        elif parsed_specific_layer is not None:
            print(f"❌ 指定的层号 {parsed_specific_layer} 超出有效范围 (0-{num_layers-1})")
            print(f"   将忽略指定层分析")
        else:
            print(f"ℹ️ 未指定层号，跳过指定层特征筛选")
        
        print("=" * 80)
        print(f"📤 准备返回响应数据:")
        print(f"   - 基础统计: valid_tactic_fens={response_data.get('valid_tactic_fens')}, tactic_fens={response_data.get('tactic_fens')}")
        print(f"   - Top LoRSA特征: {len(response_data.get('top_lorsa_features', []))}个")
        print(f"   - Top TC特征: {len(response_data.get('top_tc_features', []))}个")
        print(f"   - 指定层: {response_data.get('specific_layer', '未指定')}")
        if response_data.get('specific_layer') is not None:
            print(f"   - 指定层LoRSA: {len(response_data.get('specific_layer_lorsa', []))}个")
            print(f"   - 指定层TC: {len(response_data.get('specific_layer_tc', []))}个")
        print("=" * 80)
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.get("/tactic_features/status")
def tactic_features_status():
    """检查战术特征分析服务的状态"""
    return {
        "available": TACTIC_FEATURES_AVAILABLE,
        "hooked_transformer_available": HOOKED_TRANSFORMER_AVAILABLE
    }


# Graph Feature Diffing API
try:
    from graph_feature_diffing import compare_fen_activations, parse_node_id
    GRAPH_FEATURE_DIFFING_AVAILABLE = True
except ImportError:
    compare_fen_activations = None
    parse_node_id = None
    GRAPH_FEATURE_DIFFING_AVAILABLE = False
    print("WARNING: graph_feature_diffing not found, graph feature diffing will not be available")


@app.post("/circuit/compare_fen_activations")
def compare_fen_activations_api(request: dict):
    """
    比较两个FEN的激活差异，找出在perturbed FEN中未激活的节点
    
    请求体:
    {
        "graph_json": {...},  # 原始图的JSON数据
        "original_fen": "2k5/4Q3/3P4/8/6p1/4p3/q1pbK3/1R6 b - - 0 32",
        "perturbed_fen": "2k5/4Q3/3P4/8/6p1/8/q1pbK3/1R6 b - - 0 32",
        "model_name": "lc0/BT4-1024x15x32h",
        "activation_threshold": 0.0
    }
    """
    if not GRAPH_FEATURE_DIFFING_AVAILABLE or not CIRCUITS_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Graph feature diffing service is not available")
    
    try:
        graph_json = request.get('graph_json')
        original_fen = request.get('original_fen')
        perturbed_fen = request.get('perturbed_fen')
        model_name = request.get('model_name', 'lc0/BT4-1024x15x32h')
        activation_threshold = request.get('activation_threshold', 0.0)
        
        if not graph_json:
            raise HTTPException(status_code=400, detail="graph_json is required")
        if not original_fen:
            raise HTTPException(status_code=400, detail="original_fen is required")
        if not perturbed_fen:
            raise HTTPException(status_code=400, detail="perturbed_fen is required")
        
        # 验证FEN格式
        try:
            chess.Board(original_fen)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid original FEN: {original_fen}")
        
        try:
            chess.Board(perturbed_fen)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid perturbed FEN: {perturbed_fen}")
        
        print(f"🔍 开始比较FEN激活差异:")
        print(f"   - 原始FEN: {original_fen}")
        print(f"   - 扰动FEN: {perturbed_fen}")
        print(f"   - 模型: {model_name}")
        print(f"   - 激活阈值: {activation_threshold}")
        
        # 获取或加载模型和 transcoders/lorsas
        # 优先使用预加载的缓存，并在有加载锁时禁止重新加载
        n_layers = 15

        # 统一使用当前组合 ID（与 SaeComboLoader / circuit_trace 保持一致）
        sae_combo_id = request.get("sae_combo_id") or CURRENT_BT4_SAE_COMBO_ID
        combo_cfg = get_bt4_sae_combo(sae_combo_id)
        normalized_combo_id = combo_cfg["id"]
        combo_key = _make_combo_cache_key(model_name, normalized_combo_id)

        # 获取 HookedTransformer 模型（自身有缓存）
        hooked_model = get_hooked_model(model_name)

        # 先从本地缓存中取（按 combo_key 区分不同组合）
        global _transcoders_cache, _lorsas_cache, _replacement_models_cache, _loading_status
        cached_transcoders = _transcoders_cache.get(combo_key)
        cached_lorsas = _lorsas_cache.get(combo_key)
        cached_replacement_model = _replacement_models_cache.get(combo_key)

        cache_complete = (
            cached_transcoders is not None
            and cached_lorsas is not None
            and cached_replacement_model is not None
            and len(cached_transcoders) == n_layers
            and len(cached_lorsas) == n_layers
        )

        is_loading = _loading_status.get(combo_key, {}).get("is_loading", False)

        # 如果当前组合正在加载，直接报错，禁止在锁未释放时重复加载
        if not cache_complete and is_loading:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Transcoders/LoRSAs for model {model_name} combo {normalized_combo_id} "
                    f"are still loading. 请等待加载完成或取消后再比较激活差异。"
                ),
            )

        if cache_complete:
            # 正常使用已预加载好的模型与 SAE
            print(f"✅ 使用预加载的 transcoders/LoRSAs: {model_name} @ {normalized_combo_id}")
            replacement_model = cached_replacement_model
            transcoders = cached_transcoders
            lorsas = cached_lorsas
        else:
            # 【严格模式】完全禁止在 compare 接口里主动加载 LoRSA / TC
            # 要求调用方必须先通过 /circuit/preload_models 预加载相应组合
            msg = (
                f"No cached transcoders/LoRSAs for model {model_name} combo {normalized_combo_id}. "
                "请先调用 /circuit/preload_models 预加载该组合后再比较激活差异。"
            )
            print(f"❌ {msg}")
            raise HTTPException(status_code=503, detail=msg)
        
        # 执行比较
        result = compare_fen_activations(
            graph_json=graph_json,
            original_fen=original_fen,
            perturbed_fen=perturbed_fen,
            model_name=model_name,
            transcoders=transcoders,
            lorsas=lorsas,
            replacement_model=replacement_model,
            activation_threshold=activation_threshold,
            n_layers=n_layers
        )
        
        print(f"✅ 比较完成:")
        print(f"   - 总节点数: {result['total_nodes']}")
        print(f"   - 未激活节点数: {result['inactive_nodes_count']}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"比较失败: {str(e)}")


def _decode_fen(fen: str) -> str:
    """解码FEN字符串（支持多次解码，处理双重编码）"""
    import urllib.parse
    decoded = fen
    while "%" in decoded:
        new_decoded = urllib.parse.unquote(decoded)
        if new_decoded == decoded:
            break  # 没有更多编码了
        decoded = new_decoded
    return decoded


# 导入 global_weight 模块
try:
    from .global_weight import (
        load_max_activations,
        tc_global_weight_in,
        lorsa_global_weight_in,
        tc_global_weight_out,
        lorsa_global_weight_out,
    )
except ImportError:
    from global_weight import (
        load_max_activations,
        tc_global_weight_in,
        lorsa_global_weight_in,
        tc_global_weight_out,
        lorsa_global_weight_out,
    )


@app.get("/global_weight")
def get_global_weight(
    model_name: str = "lc0/BT4-1024x15x32h",
    sae_combo_id: str | None = None,
    feature_type: str = "tc",  # "tc" or "lorsa"
    layer_idx: int = 0,
    feature_idx: int = 0,
    k: int = 100,
):
    """
    获取feature的全局权重（输入和输出）
    
    Args:
        model_name: 模型名称
        sae_combo_id: SAE组合ID
        feature_type: 特征类型 ("tc" 或 "lorsa")
        layer_idx: 层索引
        feature_idx: 特征索引
        k: 返回的top k数量
    
    Returns:
        包含输入和输出全局权重的字典
    """
    try:
        # URL解码，处理可能的编码问题（与 /circuit/loading_logs 保持一致）
        import urllib.parse
        
        decoded_model_name = urllib.parse.unquote(model_name)
        if "%" in decoded_model_name:
            decoded_model_name = urllib.parse.unquote(decoded_model_name)
        
        # 获取SAE组合配置
        combo_id = sae_combo_id or CURRENT_BT4_SAE_COMBO_ID
        combo_cfg = get_bt4_sae_combo(combo_id)
        normalized_combo_id = combo_cfg["id"]
        
        # 使用 get_cached_transcoders_and_lorsas 获取缓存的transcoders和lorsas
        # 这个函数会先检查 circuits_service 的缓存，然后再检查本地缓存
        # 使用解码后的 model_name
        cached_transcoders, cached_lorsas = get_cached_transcoders_and_lorsas(decoded_model_name, normalized_combo_id)
        
        if cached_transcoders is None or cached_lorsas is None:
            # 提供更详细的错误信息，包括请求的组合ID和当前服务器端的组合ID
            # 使用解码后的 model_name 生成缓存键
            cache_key = _make_combo_cache_key(decoded_model_name, normalized_combo_id)
            error_detail = (
                f"Transcoders/LoRSAs未加载，请先调用 /circuit/preload_models 预加载。"
                f"请求的组合ID: {normalized_combo_id}, "
                f"缓存键: {cache_key}, "
                f"当前服务器端组合ID: {CURRENT_BT4_SAE_COMBO_ID}"
            )
            print(f"⚠️ /global_weight 请求失败: {error_detail}")
            print(f"   原始model_name参数: {model_name!r}")
            print(f"   解码后model_name: {decoded_model_name!r}")
            # 打印当前缓存键列表以帮助调试
            if CIRCUITS_SERVICE_AVAILABLE:
                from circuits_service import _global_transcoders_cache, _global_lorsas_cache
                print(f"   circuits_service 缓存键: transcoders={list(_global_transcoders_cache.keys())}, lorsas={list(_global_lorsas_cache.keys())}")
                # 检查是否存在类似的缓存键（使用原始或解码后的model_name）
                for key in list(_global_transcoders_cache.keys()) + list(_global_lorsas_cache.keys()):
                    if normalized_combo_id in key:
                        print(f"     找到相关缓存键: {key!r}")
            print(f"   本地缓存键: transcoders={list(_transcoders_cache.keys())}, lorsas={list(_lorsas_cache.keys())}")
            # 检查是否存在类似的缓存键
            for key in list(_transcoders_cache.keys()) + list(_lorsas_cache.keys()):
                if normalized_combo_id in key:
                    print(f"     找到相关缓存键: {key!r}")
            raise HTTPException(
                status_code=503,
                detail=error_detail
            )
        
        # 加载max activations数据
        tc_max_acts, lorsa_max_acts = load_max_activations(
            normalized_combo_id, device=device, get_bt4_sae_combo=get_bt4_sae_combo
        )
        
        # 验证参数
        if layer_idx < 0 or layer_idx >= len(cached_transcoders):
            raise HTTPException(status_code=400, detail=f"layer_idx必须在0-{len(cached_transcoders)-1}之间")
        
        if feature_type == "tc":
            if feature_idx < 0 or feature_idx >= cached_transcoders[layer_idx].cfg.d_sae:
                raise HTTPException(
                    status_code=400,
                    detail=f"feature_idx必须在0-{cached_transcoders[layer_idx].cfg.d_sae-1}之间"
                )
            
            # 计算TC的全局权重
            features_in = tc_global_weight_in(
                cached_transcoders, cached_lorsas, layer_idx, feature_idx,
                tc_max_acts, lorsa_max_acts, k=k
            )
            features_out = tc_global_weight_out(
                cached_transcoders, cached_lorsas, layer_idx, feature_idx,
                tc_max_acts, lorsa_max_acts, k=k
            )
        elif feature_type == "lorsa":
            if feature_idx < 0 or feature_idx >= cached_lorsas[layer_idx].cfg.d_sae:
                raise HTTPException(
                    status_code=400,
                    detail=f"feature_idx必须在0-{cached_lorsas[layer_idx].cfg.d_sae-1}之间"
                )
            
            # 计算LoRSA的全局权重
            features_in = lorsa_global_weight_in(
                cached_transcoders, cached_lorsas, layer_idx, feature_idx,
                tc_max_acts, lorsa_max_acts, k=k
            )
            features_out = lorsa_global_weight_out(
                cached_transcoders, cached_lorsas, layer_idx, feature_idx,
                tc_max_acts, lorsa_max_acts, k=k
            )
        else:
            raise HTTPException(status_code=400, detail="feature_type必须是'tc'或'lorsa'")
        
        return {
            "feature_type": feature_type,
            "layer_idx": layer_idx,
            "feature_idx": feature_idx,
            "feature_name": f"BT4_{feature_type}_L{layer_idx}{'M' if feature_type == 'tc' else 'A'}_k30_e16#{feature_idx}",
            "features_in": [{"name": name, "weight": weight} for name, weight in features_in],
            "features_out": [{"name": name, "weight": weight} for name, weight in features_out],
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"计算全局权重失败: {str(e)}")


###############################################################################
# Circuit Annotation API
###############################################################################


@app.post("/circuit_annotations")
def create_circuit_annotation(request: dict):
    """
    创建新的circuit标注
    
    Args:
        request: 包含以下字段：
            - circuit_interpretation: 回路的整体解释
            - sae_combo_id: SAE组合ID
            - features: 特征列表，每个特征包含：
                - sae_name: SAE名称
                - sae_series: SAE系列
                - layer: 层号
                - feature_index: 特征索引
                - feature_type: 特征类型 ("transcoder" 或 "lorsa")
                - interpretation: 该特征的解释（可选）
            - metadata: 可选的元数据字典
    
    Returns:
        创建的circuit标注信息
    """
    try:
        circuit_interpretation = request.get("circuit_interpretation", "")
        sae_combo_id = request.get("sae_combo_id")
        features = request.get("features", [])
        metadata = request.get("metadata")
        
        if not sae_combo_id:
            raise HTTPException(status_code=400, detail="sae_combo_id is required")
        
        if not isinstance(features, list) or len(features) == 0:
            raise HTTPException(status_code=400, detail="features must be a non-empty list")
        
        # 生成唯一的circuit_id
        import uuid
        circuit_id = str(uuid.uuid4())
        
        # 添加调试日志
        print(f"[DEBUG] create_circuit_annotation: circuit_id={circuit_id}, sae_combo_id={sae_combo_id}")
        print(f"[DEBUG] create_circuit_annotation: features={features}")
        
        # 创建circuit标注
        success = client.create_circuit_annotation(
            circuit_id=circuit_id,
            circuit_interpretation=circuit_interpretation,
            sae_combo_id=sae_combo_id,
            features=features,
            metadata=metadata,
        )
        
        print(f"[DEBUG] create_circuit_annotation: success={success}")
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create circuit annotation")
        
        # 获取创建的circuit标注
        circuit = client.get_circuit_annotation(circuit_id)
        if circuit is None:
            raise HTTPException(status_code=500, detail="Failed to retrieve created circuit annotation")
        
        # 转换为字典格式
        circuit_dict = circuit.model_dump()
        circuit_dict["created_at"] = circuit.created_at.isoformat()
        circuit_dict["updated_at"] = circuit.updated_at.isoformat()
        
        return circuit_dict
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"创建circuit标注失败: {str(e)}")


@app.get("/circuit_annotations/by_feature")
def get_circuits_by_feature(
    sae_name: str,
    sae_series: Optional[str] = None,
    layer: int = 0,
    feature_index: int = 0,
    feature_type: Optional[str] = None,
):
    """
    获取包含指定特征的所有circuit标注
    
    Args:
        sae_name: SAE名称
        sae_series: SAE系列（可选，默认使用全局sae_series）
        layer: 层号
        feature_index: 特征索引
        feature_type: 可选的特征类型过滤器 ("transcoder" 或 "lorsa")
    
    Returns:
        包含该特征的所有circuit标注列表
    """
    try:
        sae_series_param = sae_series if sae_series is not None else globals()['sae_series']
        
        # 添加调试日志
        print(f"[DEBUG] get_circuits_by_feature: sae_name={sae_name}, sae_series={sae_series_param}, layer={layer}, feature_index={feature_index}, feature_type={feature_type}")
        
        circuits = client.get_circuits_by_feature(
            sae_name=sae_name,
            sae_series=sae_series_param,
            layer=layer,
            feature_index=feature_index,
            feature_type=feature_type,
        )
        
        print(f"[DEBUG] get_circuits_by_feature: found {len(circuits)} circuits")
        
        circuit_list = []
        for circuit in circuits:
            circuit_dict = circuit.model_dump()
            circuit_dict["created_at"] = circuit.created_at.isoformat()
            circuit_dict["updated_at"] = circuit.updated_at.isoformat()
            circuit_list.append(circuit_dict)
        
        return {"circuits": circuit_list}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取circuit标注失败: {str(e)}")


@app.get("/circuit_annotations/{circuit_id}")
def get_circuit_annotation(circuit_id: str):
    """
    获取指定的circuit标注
    
    Args:
        circuit_id: Circuit标注的唯一ID
    
    Returns:
        Circuit标注信息
    """
    try:
        circuit = client.get_circuit_annotation(circuit_id)
        if circuit is None:
            raise HTTPException(status_code=404, detail=f"Circuit annotation {circuit_id} not found")
        
        circuit_dict = circuit.model_dump()
        circuit_dict["created_at"] = circuit.created_at.isoformat()
        circuit_dict["updated_at"] = circuit.updated_at.isoformat()
        
        return circuit_dict
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取circuit标注失败: {str(e)}")


@app.get("/circuit_annotations")
def list_circuit_annotations(
    sae_combo_id: Optional[str] = None,
    limit: int = 100,
    skip: int = 0,
):
    """
    列出所有circuit标注
    
    Args:
        sae_combo_id: 可选的SAE组合ID过滤器
        limit: 返回的最大数量
        skip: 跳过的数量（用于分页）
    
    Returns:
        Circuit标注列表
    """
    try:
        circuits = client.list_circuit_annotations(
            sae_combo_id=sae_combo_id,
            limit=limit,
            skip=skip,
        )
        
        circuit_list = []
        for circuit in circuits:
            circuit_dict = circuit.model_dump()
            circuit_dict["created_at"] = circuit.created_at.isoformat()
            circuit_dict["updated_at"] = circuit.updated_at.isoformat()
            circuit_list.append(circuit_dict)
        
        return {
            "circuits": circuit_list,
            "total_count": client.get_circuit_annotation_count(sae_combo_id=sae_combo_id),
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"列出circuit标注失败: {str(e)}")


@app.put("/circuit_annotations/{circuit_id}/interpretation")
def update_circuit_interpretation(circuit_id: str, request: dict):
    """
    更新circuit的整体解释
    
    Args:
        circuit_id: Circuit标注的唯一ID
        request: 包含 circuit_interpretation 字段
    
    Returns:
        成功消息
    """
    try:
        circuit_interpretation = request.get("circuit_interpretation", "")
        
        success = client.update_circuit_interpretation(
            circuit_id=circuit_id,
            circuit_interpretation=circuit_interpretation,
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Circuit annotation {circuit_id} not found")
        
        return {"message": "Circuit interpretation updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"更新circuit解释失败: {str(e)}")


@app.post("/circuit_annotations/{circuit_id}/features")
def add_feature_to_circuit(circuit_id: str, request: dict):
    """
    向circuit添加一个特征
    
    Args:
        circuit_id: Circuit标注的唯一ID
        request: 包含以下字段：
            - sae_name: SAE名称
            - sae_series: SAE系列（可选，默认使用全局sae_series）
            - layer: 层号
            - feature_index: 特征索引
            - feature_type: 特征类型 ("transcoder" 或 "lorsa")
            - interpretation: 该特征的解释（可选）
    
    Returns:
        成功消息
    """
    try:
        sae_name = request.get("sae_name")
        sae_series_param = request.get("sae_series", sae_series)
        layer = request.get("layer")
        feature_index = request.get("feature_index")
        feature_type = request.get("feature_type")
        interpretation = request.get("interpretation", "")
        
        if not all([sae_name, layer is not None, feature_index is not None, feature_type]):
            raise HTTPException(
                status_code=400,
                detail="sae_name, layer, feature_index, and feature_type are required"
            )
        
        success = client.add_feature_to_circuit(
            circuit_id=circuit_id,
            sae_name=sae_name,
            sae_series=sae_series_param,
            layer=layer,
            feature_index=feature_index,
            feature_type=feature_type,
            interpretation=interpretation,
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Circuit annotation {circuit_id} not found or feature already exists"
            )
        
        return {"message": "Feature added to circuit successfully"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"添加特征到circuit失败: {str(e)}")


@app.delete("/circuit_annotations/{circuit_id}/features")
def remove_feature_from_circuit(circuit_id: str, request: dict):
    """
    从circuit中删除一个特征
    
    Args:
        circuit_id: Circuit标注的唯一ID
        request: 包含以下字段：
            - sae_name: SAE名称
            - sae_series: SAE系列（可选，默认使用全局sae_series）
            - layer: 层号
            - feature_index: 特征索引
            - feature_type: 特征类型 ("transcoder" 或 "lorsa")
    
    Returns:
        成功消息
    """
    try:
        sae_name = request.get("sae_name")
        sae_series_param = request.get("sae_series", sae_series)
        layer = request.get("layer")
        feature_index = request.get("feature_index")
        feature_type = request.get("feature_type")
        
        if not all([sae_name, layer is not None, feature_index is not None, feature_type]):
            raise HTTPException(
                status_code=400,
                detail="sae_name, layer, feature_index, and feature_type are required"
            )
        
        success = client.remove_feature_from_circuit(
            circuit_id=circuit_id,
            sae_name=sae_name,
            sae_series=sae_series_param,
            layer=layer,
            feature_index=feature_index,
            feature_type=feature_type,
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Circuit annotation {circuit_id} not found or feature not in circuit"
            )
        
        return {"message": "Feature removed from circuit successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"从circuit删除特征失败: {str(e)}")


@app.put("/circuit_annotations/{circuit_id}/features/interpretation")
def update_feature_interpretation_in_circuit(circuit_id: str, request: dict):
    """
    更新circuit中某个特征的解释
    
    Args:
        circuit_id: Circuit标注的唯一ID
        request: 包含以下字段：
            - sae_name: SAE名称
            - sae_series: SAE系列（可选，默认使用全局sae_series）
            - layer: 层号
            - feature_index: 特征索引
            - feature_type: 特征类型 ("transcoder" 或 "lorsa")
            - interpretation: 新的解释文本
    
    Returns:
        成功消息
    """
    try:
        sae_name = request.get("sae_name")
        sae_series_param = request.get("sae_series", sae_series)
        layer = request.get("layer")
        feature_index = request.get("feature_index")
        feature_type = request.get("feature_type")
        interpretation = request.get("interpretation", "")
        
        if not all([sae_name, layer is not None, feature_index is not None, feature_type]):
            raise HTTPException(
                status_code=400,
                detail="sae_name, layer, feature_index, and feature_type are required"
            )
        
        success = client.update_feature_interpretation_in_circuit(
            circuit_id=circuit_id,
            sae_name=sae_name,
            sae_series=sae_series_param,
            layer=layer,
            feature_index=feature_index,
            feature_type=feature_type,
            interpretation=interpretation,
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Circuit annotation {circuit_id} not found or feature not in circuit"
            )
        
        return {"message": "Feature interpretation updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"更新特征解释失败: {str(e)}")


@app.delete("/circuit_annotations/{circuit_id}")
def delete_circuit_annotation(circuit_id: str):
    """
    删除circuit标注
    
    Args:
        circuit_id: Circuit标注的唯一ID
    
    Returns:
        成功消息
    """
    try:
        success = client.delete_circuit_annotation(circuit_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Circuit annotation {circuit_id} not found")
        
        return {"message": "Circuit annotation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"删除circuit标注失败: {str(e)}")


# 添加CORS中间件 - 必须在所有路由定义之后
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
