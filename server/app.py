# NEW HEADER
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io
from functools import lru_cache
from typing import Any, Optional, Tuple, List, Dict

import msgpack
import numpy as np
import plotly.graph_objects as go
import torch
from datasets import Dataset
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
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

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

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


# 添加全局模型缓存
_hooked_models = {}

def get_hooked_model(model_name: str = 'lc0/T82-768x15x24h'):
    """获取或加载HookedTransformer模型"""
    global _hooked_models
    
    if model_name not in _hooked_models:
        if not HOOKED_TRANSFORMER_AVAILABLE:
            raise ValueError("HookedTransformer不可用，请安装transformer_lens")
        
        print(f"🔍 正在加载HookedTransformer模型: {model_name}")
        _hooked_models[model_name] = HookedTransformer.from_pretrained_no_processing(
            model_name,
            dtype=torch.float32,
        ).eval()
        print(f"✅ HookedTransformer模型 {model_name} 加载成功")
    
    return _hooked_models[model_name]

def get_available_models():
    """获取可用的模型列表"""
    return [
        {'name': 'lc0/T82-768x15x24h', 'display_name': 'T82-768x15x24h'},
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
        """  # Get model and dataset
        print(f'{model_name = } in process_sample')
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

        print("function 1")
        _, feature_acts_counts = np.unique(
            feature_acts_indices[0],
            return_counts=True,
        )
        print("function 2")
        _, z_pattern_counts = (
            np.unique(z_pattern_indices[0], return_counts=True)
            if z_pattern_indices is not None
            else (None, None)
        )
        print("function 3")
        feature_acts_sample_ranges = np.concatenate(
            [[0], np.cumsum(feature_acts_counts)]
        )
        print("function 4")
        z_pattern_sample_ranges = (
            np.concatenate([[0], np.cumsum(z_pattern_counts)])
            if z_pattern_counts is not None
            else None
        )
        print("function 5")
        feature_acts_sample_ranges = list(
            zip(feature_acts_sample_ranges[:-1], feature_acts_sample_ranges[1:])
        )
        print("function 6")
        if z_pattern_sample_ranges is not None:
            z_pattern_sample_ranges = list(
                zip(z_pattern_sample_ranges[:-1], z_pattern_sample_ranges[1:])
            )
            if len(feature_acts_sample_ranges) != len(z_pattern_sample_ranges):
                z_pattern_sample_ranges = [(None, None)] * len(feature_acts_sample_ranges)
        else:
            z_pattern_sample_ranges = [(None, None)] * len(feature_acts_sample_ranges)
        print("function 7")
        for (feature_acts_start, feature_acts_end), (z_pattern_start, z_pattern_end) in zip(feature_acts_sample_ranges, z_pattern_sample_ranges):
            feature_acts_indices_i = feature_acts_indices[1, feature_acts_start:feature_acts_end]
            feature_acts_values_i = feature_acts_values[feature_acts_start:feature_acts_end]
            z_pattern_indices_i = z_pattern_indices[1:, z_pattern_start:z_pattern_end] if z_pattern_indices is not None else None
            z_pattern_values_i = z_pattern_values[z_pattern_start:z_pattern_end] if z_pattern_values is not None else None

            yield feature_acts_indices_i, feature_acts_values_i, z_pattern_indices_i, z_pattern_values_i


    sample_groups = []
    for sampling in analysis.samplings:
        try:
            print(f'go into process_sample loop')
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
    """
    fen = request.get("fen")
    model_name = request.get("model_name", "lc0/T82-768x15x24h")
    
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
        
        # 使用指定的模型
        model = get_hooked_model(model_name)
        
        # 创建引擎并获取移动（不做随机回退）
        engine = LC0Engine(model)
        move = engine.play(board)
        return {"move": move.uci(), "model_used": model_name}
        
    except ValueError as e:
        print(f"❌ 模型找不到合法移动: {e}")
        raise HTTPException(status_code=400, detail=f"模型找不到合法移动: {str(e)}")
    except Exception as e:
        print(f"❌ 处理移动时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理移动时出错: {str(e)}")


# 在play_game接口后添加局面分析接口
@app.post("/analyze/board")
def analyze_board(request: dict):
    """使用HookedTransformer模型分析当前局面，并返回行棋方胜率、和棋率及对方胜率"""
    fen = request.get("fen")
    model_name = request.get("model_name", "lc0/T82-768x15x24h")
    
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
    from circuits_service import run_circuit_trace
    CIRCUITS_SERVICE_AVAILABLE = True
except ImportError:
    run_circuit_trace = None
    CIRCUITS_SERVICE_AVAILABLE = False
    print("WARNING: circuits_service not found, circuit tracing will not be available")

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



@app.post("/circuit_trace")
def circuit_trace(request: dict):
    """
    运行circuit trace分析并返回graph数据
    
    Args:
        request: 包含分析参数的请求体
            - fen: FEN字符串 (必需)
            - move_uci: 要分析的UCI移动 (必需)
            - side: 分析侧 (q/k/both, 默认: "k")
            - max_feature_nodes: 最大特征节点数 (默认: 1024)
            - node_threshold: 节点阈值 (默认: 0.9)
            - edge_threshold: 边阈值 (默认: 0.69)
            - max_n_logits: 最大logit数量 (默认: 1)
            - desired_logit_prob: 期望logit概率 (默认: 0.95)
            - batch_size: 批处理大小 (默认: 1)
            - order_mode: 排序模式 (positive/negative, 默认: "positive")
            - encoder_demean: 是否对encoder进行demean (默认: False)
            - save_activation_info: 是否保存激活信息 (默认: False)
    
    Returns:
        Graph数据 (JSON格式)
    """
    try:
        # 检查circuits_service是否可用
        if not CIRCUITS_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Circuit tracing service not available")
        
        # 提取参数
        fen = request.get("fen")
        if not fen:
            raise HTTPException(status_code=400, detail="FEN string is required")
        
        move_uci = request.get("move_uci")
        if not move_uci:
            raise HTTPException(status_code=400, detail="move_uci is required")
        
        side = request.get("side", "k")
        max_feature_nodes = request.get("max_feature_nodes", 1024)
        node_threshold = request.get("node_threshold", 0.9)
        edge_threshold = request.get("edge_threshold", 0.69)
        max_n_logits = request.get("max_n_logits", 1)
        desired_logit_prob = request.get("desired_logit_prob", 0.95)
        batch_size = request.get("batch_size", 1)
        order_mode = request.get("order_mode", "positive")
        encoder_demean = request.get("encoder_demean", False)
        save_activation_info = request.get("save_activation_info", False)
        
        # 验证 side 参数
        if side not in ["q", "k", "both"]:
            raise HTTPException(status_code=400, detail="side must be 'q', 'k', or 'both'")
        
        # 验证 order_mode 参数
        if order_mode not in ["positive", "negative"]:
            raise HTTPException(status_code=400, detail="order_mode must be 'positive' or 'negative'")
        
        # 获取已缓存的HookedTransformer模型
        hooked_model = get_hooked_model()
        
        # 运行circuit trace，传递已缓存的模型
        graph_data = run_circuit_trace(
            prompt=fen,
            move_uci=move_uci,
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
            log_level="INFO",
            hooked_model=hooked_model  # 传递已缓存的模型
        )
        
        return graph_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Circuit trace analysis failed: {str(e)}")


@app.get("/circuit_trace/status")
def circuit_trace_status():
    """检查circuit trace服务的状态"""
    return {
        "available": CIRCUITS_SERVICE_AVAILABLE,
        "hooked_transformer_available": HOOKED_TRANSFORMER_AVAILABLE
    }


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

        print(f"🔍 运行steering分析: {feature_type} L{layer} pos{pos} feature{feature} scale{steering_scale}")

        result = run_feature_steering_analysis(
            fen=fen,
            feature_type=feature_type,
            layer=layer,
            pos=pos,
            feature=feature,
            steering_scale=steering_scale,
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
        
        # 获取指定的模型
        model_name = request.get("model_name", "lc0/T82-768x15x24h")
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
            - model_name: 模型名称 (可选，默认: lc0/T82-768x15x24h)
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
        
        model_name = request.get("model_name", "lc0/T82-768x15x24h")
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


# 添加CORS中间件 - 必须在所有路由定义之后
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
