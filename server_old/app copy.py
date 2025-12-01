# NEW HEADER
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io
from functools import lru_cache
from typing import Any, Optional

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
_hooked_model = None

def get_hooked_model():
    """获取或加载HookedTransformer模型"""
    global _hooked_model
    if _hooked_model is None:
        if not HOOKED_TRANSFORMER_AVAILABLE:
            raise ValueError("HookedTransformer不可用，请安装transformer_lens")
        
        print("🔍 正在加载HookedTransformer模型...")
        _hooked_model = HookedTransformer.from_pretrained_no_processing(
            'lc0/T82-768x15x24h',
            dtype=torch.float32,
        ).eval()
        print("✅ HookedTransformer模型加载成功")
    
    return _hooked_model


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
    
    # 如果没有找到FEN数据，检查模型名称或数据集名称来判断是否为棋类模型
    # 这里可以添加更多的棋类模型检测逻辑
    
    if has_fen_data:
        # 对于国际象棋模型，强制最小长度为64（棋盘格子数）
        min_length = max(64, feature_acts_indices[-1] + 10)
        print(f"🔍 检测到国际象棋模型（通过FEN数据），强制最小长度: {min_length}")
    else:
        # 对于其他模型，使用原有逻辑
        min_length = min(len(origins), feature_acts_indices[-1] + 10)
        print(f"🔍 通用模型，计算最小长度: {min_length}")
    
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
            
            # 添加调试信息
            print(f"🔍 处理国际象棋模型数据:")
            print(f"   - 模型名称: {model_name}")
            print(f"   - 数据集名称: {dataset_name}")
            print(f"   - 检测方式: {'FEN数据' if has_fen_data else '模型/数据集名称'}")
            print(f"   - feature_acts_indices shape: {feature_acts_indices.shape}")
            print(f"   - feature_acts_values shape: {feature_acts_values.shape}")
            print(f"   - 前几个索引: {feature_acts_indices[:5] if len(feature_acts_indices) > 0 else 'empty'}")
            print(f"   - 前几个值: {feature_acts_values[:5] if len(feature_acts_values) > 0 else 'empty'}")
            
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
            
            print(f" 国际象棋模型: 创建64长度激活数组，非零激活数: {np.count_nonzero(dense_feature_acts)}")
            print(f" FEN数据: {data.get('fen', 'Not found')[:50]}...")
        else:
            # 对于其他模型，使用原有逻辑
            dense_feature_acts = np.zeros(len(origins))
            
            # 添加调试信息
            print(f"🔍 处理通用模型数据:")
            print(f"   - 模型名称: {model_name}")
            print(f"   - 数据集名称: {dataset_name}")
            print(f"   - feature_acts_indices shape: {feature_acts_indices.shape}")
            print(f"   - feature_acts_values shape: {feature_acts_values.shape}")
            print(f"   - origins length: {len(origins)}")
            
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
                    else:
                        print(f"⚠️ 索引 {idx} 超出范围 [0, {len(origins)})，跳过")
                        
                except (ValueError, TypeError, IndexError) as e:
                    print(f"⚠️ 处理索引 {idx} 和值 {val} 时出错: {e}")
                    continue
            
            print(f"🔍 通用模型: 创建{len(origins)}长度激活数组，非零激活数: {np.count_nonzero(dense_feature_acts)}")

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

        # 打印 z_pattern 的形状与前几项
        print(f"   - z_pattern_indices shape: {z_pattern_indices.shape if z_pattern_indices is not None else 'None'}")
        print(f"   - z_pattern_values shape: {z_pattern_values.shape if z_pattern_values is not None else 'None'}")

        if z_pattern_indices is not None and z_pattern_values is not None and z_pattern_values.size > 0:
            n_head = min(5, z_pattern_values.shape[0])
            if z_pattern_indices.ndim == 2:
                z_head_idx = z_pattern_indices[:, :n_head].T.tolist()  # 例如 [ [i,j], ... ]
            else:
                z_head_idx = z_pattern_indices[:n_head].tolist()
            print(f"   - z 前几个索引: {z_head_idx}")
            print(f"   - z 前几个值: {np.round(z_pattern_values[:n_head], 6).tolist()}")

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
        # 添加详细的原始数据结构调试
        print(f"🔍 原始数据结构调试:")
        print(f"   - feature_acts_indices shape: {feature_acts_indices.shape}")
        print(f"   - feature_acts_values shape: {feature_acts_values.shape}")
        print(f"   - z_pattern_indices shape: {z_pattern_indices.shape if z_pattern_indices is not None else 'None'}")
        print(f"   - z_pattern_values shape: {z_pattern_values.shape if z_pattern_values is not None else 'None'}")
        if z_pattern_indices is None or z_pattern_values is None:
            print("   - 说明: z_pattern 为空（对 transcoder 分析属正常情况）")
        
        # 分析第一维的唯一值（样本数量）
        if feature_acts_indices.size == 0 or feature_acts_indices.shape[1] == 0:
            print("   - feature_acts 无样本（空稀疏索引）")
            return
        unique_samples = np.unique(feature_acts_indices[0])
        print(f"   - feature_acts 第一维唯一值: {len(unique_samples)} 个样本")
        if unique_samples.size > 0:
            print(f"   - feature_acts 样本范围: {unique_samples.min()} 到 {unique_samples.max()}")
        
        if z_pattern_indices is not None and z_pattern_indices.size > 0 and z_pattern_indices.shape[1] > 0:
            unique_z_samples = np.unique(z_pattern_indices[0])
            print(f"   - z_pattern 第一维唯一值: {len(unique_z_samples)} 个样本")
            if unique_z_samples.size > 0:
                print(f"   - z_pattern 样本范围: {unique_z_samples.min()} 到 {unique_z_samples.max()}")
 
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
 
        # 组装为区间对列表
        feature_acts_sample_ranges = list(
            zip(feature_acts_sample_ranges[:-1], feature_acts_sample_ranges[1:])
        )
        if z_pattern_sample_ranges is not None:
            z_pattern_sample_ranges = list(
                zip(z_pattern_sample_ranges[:-1], z_pattern_sample_ranges[1:])
            )
            # 长度不匹配时，退化为 None 区间
            if len(feature_acts_sample_ranges) != len(z_pattern_sample_ranges):
                print("❌ 数据不匹配：使用 feature_acts 长度对齐 z_pattern")
                z_pattern_sample_ranges = [(None, None)] * len(feature_acts_sample_ranges)
        else:
            # z_pattern 为空时，构造等长的 None 区间占位
            z_pattern_sample_ranges = [(None, None)] * len(feature_acts_sample_ranges)
 
        for (feature_acts_start, feature_acts_end), (z_pattern_start, z_pattern_end) in zip(feature_acts_sample_ranges, z_pattern_sample_ranges):
            feature_acts_indices_i = feature_acts_indices[1, feature_acts_start:feature_acts_end]
            feature_acts_values_i = feature_acts_values[feature_acts_start:feature_acts_end]
            z_pattern_indices_i = z_pattern_indices[1:, z_pattern_start:z_pattern_end] if z_pattern_indices is not None else None
            z_pattern_values_i = z_pattern_values[z_pattern_start:z_pattern_end] if z_pattern_values is not None else None
            
            # 稀疏 z_pattern 每样本完整打印
            if z_pattern_indices_i is not None and z_pattern_values_i is not None:
                nnz_z = z_pattern_values_i.shape[0]
                print(f"🧪 z_pattern sample: nnz={nnz_z}")
                if nnz_z > 0:
                    if z_pattern_indices_i.ndim == 2:
                        # 例如形状 [2, N]，逐列即每个坐标
                        z_all_idx = z_pattern_indices_i.T.tolist()
                    else:
                        # 例如形状 [N]
                        z_all_idx = z_pattern_indices_i.tolist()
                    z_all_val = z_pattern_values_i.tolist()

                    # 分别打印所有索引与所有值
                    print(f"   z all idx: {z_all_idx}")
                    print(f"   z all val: {z_all_val}")

                    # 若你想按"(idx, val)"配对一起打印（更直观），可用：
                    if z_pattern_indices_i.ndim == 2:
                        z_pairs = list(zip(z_all_idx, z_all_val))  # [([i,j], val), ...]
                    else:
                        z_pairs = list(zip(z_all_idx, z_all_val))  # [(i, val), ...]
                    print(f"   z pairs: {z_pairs}")

            yield feature_acts_indices_i, feature_acts_values_i, z_pattern_indices_i, z_pattern_values_i

    # Process all samples for each sampling
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
            print(f"❌ 处理sampling '{sampling.name}' 时出错:")
            print(f"   - 错误类型: {type(e).__name__}")
            print(f"   - 错误信息: {str(e)}")
            import traceback
            print(f"   - 详细堆栈:")
            traceback.print_exc()
            
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


@app.post("/analyze/stockfish")
def analyze_stockfish(request: dict):
    """使用 Stockfish 分析国际象棋位置
    
    Args:
        request: 包含 FEN 字符串的请求体
        
    Returns:
        Stockfish 分析结果
    """
    try:
        fen = request.get("fen")
        if not fen:
            return Response(content="FEN string is required", status_code=400)
        
        # 验证 FEN 格式（基本检查）
        parts = fen.split()
        if len(parts) < 4:
            return Response(content="Invalid FEN format", status_code=400)
        
        # 检查 Stockfish 是否可用
        try:
            result = subprocess.run(
                ["stockfish", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode != 0:
                return Response(content="Stockfish not available", status_code=503)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return Response(content="Stockfish not found or not responding", status_code=503)
        
        # 创建临时文件用于 Stockfish 输入
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(f"position fen {fen}\n")
            f.write("go depth 15\n")
            f.write("quit\n")
            temp_file = f.name
        
        try:
            # 运行 Stockfish 分析
            result = subprocess.run(
                ["stockfish"],
                stdin=open(temp_file, 'r'),
                capture_output=True,
                text=True,
                timeout=30  # 30秒超时
            )
            
            if result.returncode != 0:
                return Response(content=f"Stockfish analysis failed: {result.stderr}", status_code=500)
            
            # 解析 Stockfish 输出
            output_lines = result.stdout.strip().split('\n')
            
            # 提取最佳走法
            best_move = None
            ponder = None
            evaluation = None
            depth = None
            nodes = None
            
            for line in output_lines:
                if line.startswith('bestmove'):
                    parts = line.split()
                    if len(parts) >= 2:
                        best_move = parts[1]
                    if len(parts) >= 4 and parts[2] == 'ponder':
                        ponder = parts[3]
                elif 'info' in line and 'depth' in line:
                    # 解析评估信息
                    info_parts = line.split()
                    for i, part in enumerate(info_parts):
                        if part == 'depth' and i + 1 < len(info_parts):
                            depth = int(info_parts[i + 1])
                        elif part == 'score' and i + 1 < len(info_parts):
                            if info_parts[i + 1] == 'cp':
                                evaluation = int(info_parts[i + 2]) / 100.0  # 转换为兵值
                            elif info_parts[i + 1] == 'mate':
                                evaluation = float('inf') if int(info_parts[i + 2]) > 0 else float('-inf')
                        elif part == 'nodes' and i + 1 < len(info_parts):
                            nodes = int(info_parts[i + 1])
            
            # 计算胜率（基于评估值）
            wdl = None
            if evaluation is not None and evaluation != float('inf') and evaluation != float('-inf'):
                # 使用简单的 sigmoid 函数估算胜率
                import math
                win_prob = 1 / (1 + math.exp(-evaluation / 0.7))
                draw_prob = 0.1  # 简化假设
                loss_prob = 1 - win_prob - draw_prob
                
                wdl = {
                    "winProb": max(0, min(1, win_prob)),
                    "drawProb": max(0, min(1, draw_prob)),
                    "lossProb": max(0, min(1, loss_prob))
                }
            
            # 检查将军状态
            is_check = False
            if 'check' in result.stdout.lower():
                is_check = True
            
            # 计算物质力量（简化版本）
            material = {
                "white_material": 0,
                "black_material": 0
            }
            
            # 从 FEN 中计算物质力量
            board_part = fen.split()[0]
            piece_values = {
                'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0,
                'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0
            }
            
            for char in board_part:
                if char in piece_values:
                    if char.isupper():
                        material["white_material"] += piece_values[char]
                    else:
                        material["black_material"] += piece_values[char]
            
            response_data = {
                "status": "success",
                "fen": fen,
                "bestMove": best_move,
                "ponder": ponder,
                "evaluation": evaluation,
                "depth": depth,
                "nodes": nodes,
                "wdl": wdl,
                "isCheck": is_check,
                "material": material,
                "rules": "Standard chess rules apply"
            }
            
            return response_data
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                
    except subprocess.TimeoutExpired:
        return Response(content="Stockfish analysis timeout", status_code=504)
    except Exception as e:
        return Response(content=f"Analysis error: {str(e)}", status_code=500)


# 在文件末尾添加 LC0Engine 类和新的后端接口 /play_game

class LC0Engine:
    """简化版 LC0 模型引擎包装器，用于与模型对战"""
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def play(self, chess_board):
        try:
            # 使用 notebook 同款接口进行推理
            fen = chess_board.fen()
            print(f"🔍 处理FEN: {fen}")

            with torch.no_grad():
                output, cache = self.model.run_with_cache(fen, prepend_bos=False)
                
                # LC0模型输出格式：outputs[0]是策略，outputs[1]是价值，outputs[2]是MLH
                if isinstance(output, (list, tuple)) and len(output) >= 1:
                    policy_output = output[0]  # 形状应该是 (1, 1858)
                else:
                    policy_output = output

                # 取策略输出的logits，形状应该是 (1858,)
                if policy_output.dim() == 2:
                    policy_logits = policy_output[0]  # 从 (1, 1858) 取出 (1858,)
                else:
                    policy_logits = policy_output

            legal_moves = list(chess_board.legal_moves)
            legal_uci_set = set(move.uci() for move in legal_moves)
            sorted_token_ids = torch.argsort(policy_logits, descending=True)

            # 使用 LC0 映射将索引转换为 UCI
            try:
                mapping_index = get_mapping_index(chess_board)
                idx_to_uci = idx_to_uci_mappings[mapping_index]
            except Exception as e:
                print(f"❌ 获取LC0映射失败: {e}")
                idx_to_uci = {}

            print("🔍 模型输出调试信息:")
            print(f"   - policy_logits shape: {tuple(policy_logits.shape)}")
            print(f"   - 合法移动数量: {len(legal_moves)}")
            
            # 打印前10个最高概率的 UCI 及其logit
            top10 = []
            for idx in sorted_token_ids[:10].tolist():
                uci = idx_to_uci.get(idx)
                logit = float(policy_logits[idx].item())
                top10.append((uci, logit))
            print("   - 前10个最高概率move (uci, logit):")
            print("     " + ", ".join([f"{uci if uci is not None else 'None'}:{logit:.4f}" for uci, logit in top10]))

            # 依次尝试最高概率索引对应的 UCI，选择第一个合法移动
            for rank, idx in enumerate(sorted_token_ids.tolist(), start=1):
                uci = idx_to_uci.get(idx)
                if not uci:
                    continue
                if uci in legal_uci_set:
                    move = chess.Move.from_uci(uci)
                    print(f"✅ 选择最大概率合法移动: {uci} (概率排名: {rank}, logit: {policy_logits[idx].item():.4f})")
                    return move

            # 如果未找到合法移动，打印报错并抛异常
            print("❌ 错误：模型未能找到任何合法移动！")
            print(f"   - 当前局面 FEN: {fen}")
            print(f"   - 示例合法移动: {[m.uci() for m in legal_moves[:10]]}")
            print(f"   - 尝试了前 {min(len(sorted_token_ids), 50)} 个最高概率的token")
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
        
        # 使用缓存模型（仅首次加载）
        model = get_hooked_model()
        
        # 创建引擎并获取移动（不做随机回退）
        engine = LC0Engine(model)
        move = engine.play(board)
        return {"move": move.uci()}
        
    except ValueError as e:
        print(f"❌ 模型找不到合法移动: {e}")
        raise HTTPException(status_code=400, detail=f"模型找不到合法移动: {str(e)}")
    except Exception as e:
        print(f"❌ 处理移动时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理移动时出错: {str(e)}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 导入circuits_service
try:
    from circuits_service import run_circuit_trace
    CIRCUITS_SERVICE_AVAILABLE = True
except ImportError:
    run_circuit_trace = None
    CIRCUITS_SERVICE_AVAILABLE = False
    print("WARNING: circuits_service not found, circuit tracing will not be available")



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
            - order_mode: 排序模式 (默认: "positive")
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
        
        print(f"🔍 开始circuit trace分析:")
        print(f"   - FEN: {fen}")
        print(f"   - Move UCI: {move_uci}")
        print(f"   - Side: {side}")
        print(f"   - Max feature nodes: {max_feature_nodes}")
        
        # 运行circuit trace
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
            log_level="INFO"
        )
        
        print(f"✅ Circuit trace分析完成，返回graph数据")
        return graph_data
        
    except Exception as e:
        print(f"❌ Circuit trace分析出错: {e}")
        raise HTTPException(status_code=500, detail=f"Circuit trace analysis failed: {str(e)}")


@app.get("/circuit_trace/status")
def circuit_trace_status():
    """检查circuit trace服务的状态"""
    return {
        "available": CIRCUITS_SERVICE_AVAILABLE,
        "hooked_transformer_available": HOOKED_TRANSFORMER_AVAILABLE
    }
