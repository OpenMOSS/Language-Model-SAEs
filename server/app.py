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
from fastapi import FastAPI, Response, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
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


@app.post("/circuit/sync_clerps_to_interpretations")
def sync_clerps_to_interpretations(request: dict):

    try:
        nodes = request.get("nodes", [])
        lorsa_analysis_name = request.get("lorsa_analysis_name")
        tc_analysis_name = request.get("tc_analysis_name")
        
        if not isinstance(nodes, list):
            raise HTTPException(status_code=400, detail="nodes must be a list")
        
        print(f"🔄 开始同步clerps到interpretations:")
        print(f"   - 节点数量: {len(nodes)}")
        print(f"   - LoRSA模板: {lorsa_analysis_name}")
        print(f"   - TC模板: {tc_analysis_name}")
        
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
            
            # 构建SAE名称
            sae_name = None
            if 'lorsa' in feature_type:
                if lorsa_analysis_name:
                    sae_name = lorsa_analysis_name.replace("{}", str(layer))
                else:
                    sae_name = f"lc0-lorsa-L{layer}"
            elif 'transcoder' in feature_type or 'cross layer transcoder' in feature_type:
                if tc_analysis_name:
                    sae_name = tc_analysis_name.replace("{}", str(layer))
                else:
                    sae_name = f"lc0_L{layer}M_16x_k30_lr2e-03_auxk_sparseadam"
            
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
        
        print(f"🔄 开始从interpretations同步到clerps:")
        print(f"   - 节点数量: {len(nodes)}")
        print(f"   - LoRSA模板: {lorsa_analysis_name}")
        print(f"   - TC模板: {tc_analysis_name}")
        
        updated_nodes = []
        found_count = 0
        not_found_count = 0
        
        for node in nodes:
            node_id = node.get('node_id')
            feature_idx = node.get('feature')
            layer = node.get('layer')
            feature_type = node.get('feature_type', '').lower()
            
            # 构建SAE名称
            sae_name = None
            if 'lorsa' in feature_type:
                if lorsa_analysis_name:
                    sae_name = lorsa_analysis_name.replace("{}", str(layer))
                else:
                    sae_name = f"lc0-lorsa-L{layer}"
            elif 'transcoder' in feature_type or 'cross layer transcoder' in feature_type:
                if tc_analysis_name:
                    sae_name = tc_analysis_name.replace("{}", str(layer))
                else:
                    sae_name = f"lc0_L{layer}M_16x_k30_lr2e-03_auxk_sparseadam"
            
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
    from circuits_service import run_circuit_trace, check_dense_features
    CIRCUITS_SERVICE_AVAILABLE = True
except ImportError:
    run_circuit_trace = None
    check_dense_features = None
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
            - model_name: 模型名称 (默认: "lc0/T82-768x15x24h")
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
        negative_move_uci = request.get("negative_move_uci", None)  # 新增negative_move_uci参数
        
        side = request.get("side", "k")
        max_feature_nodes = request.get("max_feature_nodes", 1024)
        node_threshold = request.get("node_threshold", 0.9)
        edge_threshold = request.get("edge_threshold", 0.69)
        max_n_logits = request.get("max_n_logits", 1)
        desired_logit_prob = request.get("desired_logit_prob", 0.95)
        batch_size = request.get("batch_size", 1)
        order_mode = request.get("order_mode", "positive")
        encoder_demean = request.get("encoder_demean", False)
        save_activation_info = request.get("save_activation_info", True)  # 默认启用激活信息保存
        max_act_times = request.get("max_act_times", None)  # 添加最大激活次数参数
        model_name = request.get("model_name", "lc0/T82-768x15x24h")  # 添加模型名称参数
        
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
        
        # 根据模型名称设置正确的路径
        base_path = "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs"
        if 'BT4' in model_name:
            tc_base_path = f"{base_path}/result_BT4/tc"
            lorsa_base_path = f"{base_path}/result_BT4/lorsa"
        else:
            tc_base_path = f"{base_path}/result/tc"
            lorsa_base_path = f"{base_path}/result/lorsa"
        
        # 运行circuit trace，传递已缓存的模型
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


@app.post("/circuit/check_dense_features")
def check_dense_features_api(request: dict):
    """
    检查circuit中哪些节点是dense feature（激活次数超过阈值）
    
    Args:
        request: 包含检查参数的请求体
            - nodes: 节点列表
            - threshold: 激活次数阈值（可选，None表示无限大）
            - sae_series: SAE系列名称（可选，默认: lc0-circuit-tracing）
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
        
        sae_series = request.get("sae_series", "lc0-circuit-tracing")
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


@app.post("/logit_lens/mean_ablation")
def logit_lens_mean_ablation(request: dict):
    """
    运行Mean Ablation分析
    
    Args:
        request: 包含分析参数的请求体
            - fen: FEN字符串 (必需)
            - model_name: 模型名称 (可选，默认: lc0/T82-768x15x24h)
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
        
        model_name = request.get("model_name", "lc0/T82-768x15x24h")
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
    model_name: str = Form("lc0/BT4-1024x15x32h"),
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
        # ========== 调试信息：函数开始 ==========
        print("=" * 80)
        print("🚀 开始处理战术特征分析请求")
        print(f"📥 接收到的原始参数:")
        print(f"   - model_name: {model_name}")
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
        
        # 加载模型
        hooked_model = get_hooked_model(model_name)
        
        # 加载LoRSA和Transcoders
        num_layers = 15
        base_path = "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs"
        if 'BT4' in model_name:
            tc_base_path = f"{base_path}/result_BT4/tc"
            lorsa_base_path = f"{base_path}/result_BT4/lorsa"
        else:
            tc_base_path = f"{base_path}/result/tc"
            lorsa_base_path = f"{base_path}/result/lorsa"
        
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
            # lorsa_path = f"{lorsa_base_path}/lc0_L{layer}_bidirectional_lr0.0002_k_aux4096_coefficient0.125_dead_threshold1000000"
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

# 添加CORS中间件 - 必须在所有路由定义之后
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
