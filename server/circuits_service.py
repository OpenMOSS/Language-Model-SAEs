#!/usr/bin/env python3
"""
Fast tracing test script for chess SAE attribution.
This script can be run with torchrun for distributed execution.
"""

import argparse
import json
import logging
import sys
import time
import io
import contextlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch
import chess
from transformer_lens import HookedTransformer
from tqdm import tqdm

# 全局 BT4 常量（模型与 SAE 路径）
# 兼容直接运行 server 目录和作为 package 导入两种方式
try:
    from .constants import BT4_TC_BASE_PATH, BT4_LORSA_BASE_PATH
except ImportError:
    from constants import BT4_TC_BASE_PATH, BT4_LORSA_BASE_PATH

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

# 导入项目模块
from lm_saes import ReplacementModel, LowRankSparseAttention, SparseAutoEncoder
from lm_saes.circuit.attribution_qk_for_feature_attribution import attribute
from lm_saes.circuit.graph_lc0 import Graph
from lm_saes.circuit.utils.create_graph_files import create_graph_files, build_model, create_nodes, create_used_nodes_and_edges, prune_graph
from lm_saes.circuit.leela_board import LeelaBoard
from src.lm_saes.config import MongoDBConfig
from src.lm_saes.database import (
    MongoClient,
    SAERecord,
    DatasetRecord,
    ModelRecord,
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志记录"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class TeeWriter:
    """一个同时写入多个目标的writer类"""
    def __init__(self, *targets):
        self.targets = targets
    
    def write(self, text):
        for target in self.targets:
            target.write(text)
    
    def flush(self):
        for target in self.targets:
            if hasattr(target, 'flush'):
                target.flush()


class LogCapture:
    """捕获print、logger和tqdm输出的上下文管理器"""
    
    def __init__(self, log_list: list):
        """
        Args:
            log_list: 用于存储日志的列表，每个元素为 {"timestamp": float, "message": str}
        """
        self.log_list = log_list
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_buffer = io.StringIO()
        self.log_handlers = []
        
    def _log_message(self, message: str):
        """将消息添加到日志列表"""
        if message.strip():  # 只添加非空消息
            self.log_list.append({
                "timestamp": time.time(),
                "message": message.strip()
            })
    
    def _write_and_log(self, text: str, original_stream):
        """写入原始流并记录日志"""
        original_stream.write(text)  # 先写入原始流
        # 按行分割并记录
        if text:
            for line in text.rstrip('\n').split('\n'):
                if line.strip():
                    self._log_message(line)
    
    def _setup_logger_handler(self):
        """设置logger处理器"""
        class LogListHandler(logging.Handler):
            def __init__(self, log_list):
                super().__init__()
                self.log_list = log_list
                
            def emit(self, record):
                log_entry = self.format(record)
                self.log_list.append({
                    "timestamp": time.time(),
                    "message": log_entry
                })
        
        # 为attribution logger添加handler
        attribution_logger = logging.getLogger("attribution")
        handler = LogListHandler(self.log_list)
        handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        attribution_logger.addHandler(handler)
        self.log_handlers.append((attribution_logger, handler))
        
        # 也为root logger添加handler（捕获所有日志）
        root_logger = logging.getLogger()
        root_handler = LogListHandler(self.log_list)
        root_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(root_handler)
        self.log_handlers.append((root_logger, root_handler))
    
    def _setup_tqdm_handler(self):
        """设置tqdm的写入函数"""
        # 保存原始的tqdm.write
        self.original_tqdm_write = tqdm.write
        
        def custom_tqdm_write(s, file=None, end="\n", nolock=False):
            # 调用原始函数
            self.original_tqdm_write(s, file=file, end=end, nolock=nolock)
            # 记录日志
            if s.strip():
                self._log_message(s.strip())
        
        tqdm.write = custom_tqdm_write
    
    def __enter__(self):
        # 创建一个包装的stdout，同时写入原始stdout和记录日志
        class LoggingStdout:
            def __init__(self, original, log_capture):
                self.original = original
                self.log_capture = log_capture
            
            def write(self, text):
                self.log_capture._write_and_log(text, self.original)
            
            def flush(self):
                self.original.flush()
            
            def __getattr__(self, name):
                return getattr(self.original, name)
        
        class LoggingStderr:
            def __init__(self, original, log_capture):
                self.original = original
                self.log_capture = log_capture
            
            def write(self, text):
                self.log_capture._write_and_log(text, self.original)
            
            def flush(self):
                self.original.flush()
            
            def __getattr__(self, name):
                return getattr(self.original, name)
        
        # 替换stdout和stderr
        sys.stdout = LoggingStdout(self.original_stdout, self)
        sys.stderr = LoggingStderr(self.original_stderr, self)
        
        # 设置logger handler
        self._setup_logger_handler()
        
        # 设置tqdm handler
        self._setup_tqdm_handler()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复stdout和stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # 移除logger handlers
        for logger, handler in self.log_handlers:
            logger.removeHandler(handler)
        
        # 恢复tqdm.write
        if hasattr(self, 'original_tqdm_write'):
            tqdm.write = self.original_tqdm_write
        
        return False  # 不抑制异常


# 全局缓存（与app.py共享）
_global_hooked_models: Dict[str, HookedTransformer] = {}
_global_transcoders_cache: Dict[str, Dict[int, SparseAutoEncoder]] = {}
_global_lorsas_cache: Dict[str, List[LowRankSparseAttention]] = {}
_global_replacement_models_cache: Dict[str, ReplacementModel] = {}

# 加载锁，防止并发加载导致重复加载
import threading
_loading_lock = threading.Lock()
_is_loading: Dict[str, bool] = {}  # model_name -> is_loading


def get_cached_models(cache_key: str) -> Tuple[Optional[HookedTransformer], Optional[Dict[int, SparseAutoEncoder]], Optional[List[LowRankSparseAttention]], Optional[ReplacementModel]]:
    """
    获取缓存的模型、transcoders和lorsas
    
    Args:
        cache_key: 缓存键，格式为 "model_name" 或 "model_name::combo_id"
    """
    global _global_hooked_models, _global_transcoders_cache, _global_lorsas_cache, _global_replacement_models_cache
    
    # HookedTransformer 模型不依赖 combo_id，只使用 model_name
    model_name = cache_key.split("::")[0] if "::" in cache_key else cache_key
    hooked_model = _global_hooked_models.get(model_name)
    
    # transcoders, lorsas, replacement_model 使用完整的 cache_key（包含 combo_id）
    transcoders = _global_transcoders_cache.get(cache_key)
    lorsas = _global_lorsas_cache.get(cache_key)
    replacement_model = _global_replacement_models_cache.get(cache_key)
    
    return hooked_model, transcoders, lorsas, replacement_model


def set_cached_models(
    cache_key: str,
    hooked_model: HookedTransformer,
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    replacement_model: ReplacementModel
):
    """
    设置缓存的模型、transcoders和lorsas
    
    Args:
        cache_key: 缓存键，格式为 "model_name" 或 "model_name::combo_id"
    """
    global _global_hooked_models, _global_transcoders_cache, _global_lorsas_cache, _global_replacement_models_cache
    
    # HookedTransformer 模型不依赖 combo_id，只使用 model_name
    model_name = cache_key.split("::")[0] if "::" in cache_key else cache_key
    _global_hooked_models[model_name] = hooked_model
    
    # transcoders, lorsas, replacement_model 使用完整的 cache_key（包含 combo_id）
    _global_transcoders_cache[cache_key] = transcoders
    _global_lorsas_cache[cache_key] = lorsas
    _global_replacement_models_cache[cache_key] = replacement_model


def load_model_and_transcoders(
    model_name: str,
    device: str,
    tc_base_path: str,
    lorsa_base_path: str,
    n_layers: int = 15,
    hooked_model: Optional[HookedTransformer] = None,  # 新增参数
    loading_logs: Optional[list] = None,  # 新增参数：用于收集加载日志
    cancel_flag: Optional[dict] = None,  # 新增参数：用于检查是否应该中断加载 {"combo_key": should_cancel}
    cache_key: Optional[str] = None  # 新增参数：缓存键，格式为 "model_name::combo_id"，如果不提供则使用 model_name
) -> Tuple[ReplacementModel, Dict[int, SparseAutoEncoder], List[LowRankSparseAttention]]:
    """
    加载模型和transcoders（带全局缓存和加载锁，防止重复加载）
    
    Args:
        cache_key: 缓存键，格式为 "model_name::combo_id"。如果不提供，则使用 model_name（向后兼容）
    """
    global _global_hooked_models, _global_transcoders_cache, _global_lorsas_cache, _global_replacement_models_cache
    global _loading_lock, _is_loading
    
    logger = logging.getLogger(__name__)
    
    # 确定缓存键
    if cache_key is None:
        cache_key = model_name
    
    # 辅助函数：添加日志（同时打印到控制台和收集到日志列表）
    def add_log(message: str):
        print(message)
        logger.info(message)
        if loading_logs is not None:
            log_entry = {
                "timestamp": time.time(),
                "message": message
            }
            loading_logs.append(log_entry)
            # 调试：打印日志列表的长度
            if len(loading_logs) % 5 == 0:  # 每5条日志打印一次
                print(f"📝 当前日志列表长度: {len(loading_logs)}")
    
    # 先检查全局缓存（无锁快速检查）
    cached_hooked_model, cached_transcoders, cached_lorsas, cached_replacement_model = get_cached_models(cache_key)
    
    # 检查缓存是否完整（有transcoders和lorsas，且层数正确）
    if cached_transcoders is not None and cached_lorsas is not None:
        if len(cached_transcoders) == n_layers and len(cached_lorsas) == n_layers:
            if cached_replacement_model is not None:
                add_log(f"✅ 使用缓存的模型、transcoders和lorsas: {model_name}")
                logger.info(f"✅ 从缓存加载: {model_name} (transcoders={len(cached_transcoders)}层, lorsas={len(cached_lorsas)}层)")
                return cached_replacement_model, cached_transcoders, cached_lorsas
    
    # 获取加载锁，防止并发加载
    with _loading_lock:
        # 再次检查缓存（双重检查锁定模式）
        cached_hooked_model, cached_transcoders, cached_lorsas, cached_replacement_model = get_cached_models(cache_key)
        if cached_transcoders is not None and cached_lorsas is not None:
            if len(cached_transcoders) == n_layers and len(cached_lorsas) == n_layers:
                if cached_replacement_model is not None:
                    add_log(f"✅ 使用缓存的模型、transcoders和lorsas（双重检查）: {cache_key}")
                    return cached_replacement_model, cached_transcoders, cached_lorsas
        
        # 检查是否正在加载（使用 cache_key 作为加载状态键）
        if _is_loading.get(cache_key, False):
            add_log(f"⏳ 模型 {cache_key} 正在被其他线程加载，等待...")
            # 释放锁并等待
    
    # 如果正在加载，等待加载完成
    wait_count = 0
    max_wait = 600  # 最多等待600秒
    while _is_loading.get(cache_key, False) and wait_count < max_wait:
        time.sleep(1)
        wait_count += 1
        if wait_count % 10 == 0:
            add_log(f"⏳ 等待模型加载中... ({wait_count}秒)")
    
    # 再次检查缓存
    cached_hooked_model, cached_transcoders, cached_lorsas, cached_replacement_model = get_cached_models(cache_key)
    if cached_transcoders is not None and cached_lorsas is not None:
        if len(cached_transcoders) == n_layers and len(cached_lorsas) == n_layers:
            if cached_replacement_model is not None:
                add_log(f"✅ 使用缓存的模型、transcoders和lorsas（等待后）: {cache_key}")
                return cached_replacement_model, cached_transcoders, cached_lorsas
    
    # 获取加载锁并标记为正在加载
    with _loading_lock:
        # 最终检查
        cached_hooked_model, cached_transcoders, cached_lorsas, cached_replacement_model = get_cached_models(cache_key)
        if cached_transcoders is not None and cached_lorsas is not None:
            if len(cached_transcoders) == n_layers and len(cached_lorsas) == n_layers:
                if cached_replacement_model is not None:
                    add_log(f"✅ 使用缓存的模型、transcoders和lorsas（最终检查）: {cache_key}")
                    return cached_replacement_model, cached_transcoders, cached_lorsas
        
        # 标记为正在加载（使用 cache_key）
        _is_loading[cache_key] = True
        add_log(f"🔒 获取加载锁，开始加载模型: {cache_key}")
    
    try:
        # 如果缓存不完整或不存在，则加载
        add_log(f"🔍 开始加载模型和transcoders: {model_name}")
        
        # 使用传入的模型或从缓存获取或加载新模型
        if hooked_model is not None:
            add_log("使用传入的HookedTransformer模型")
            model = hooked_model
        elif cached_hooked_model is not None:
            add_log("使用缓存的HookedTransformer模型")
            model = cached_hooked_model
        else:
            add_log("加载新的HookedTransformer模型...")
            model = HookedTransformer.from_pretrained_no_processing(
                model_name,
                dtype=torch.float32,
            ).eval()
            # 缓存模型
            _global_hooked_models[model_name] = model
            add_log("✅ HookedTransformer模型加载完成")
        
        # 初始化或获取已有的transcoders缓存（使用 cache_key）
        if cache_key not in _global_transcoders_cache:
            _global_transcoders_cache[cache_key] = {}
        transcoders = _global_transcoders_cache[cache_key]
        
        # 加载transcoders（逐层检查，避免重复加载）
        add_log(f"🔍 开始加载Transcoders，共{n_layers}层...")
        for layer in range(n_layers):
            # 检查是否应该中断加载
            if cancel_flag is not None:
                # 如果有检查函数，调用它；否则直接检查 should_cancel
                if "check_fn" in cancel_flag and callable(cancel_flag["check_fn"]):
                    should_cancel = cancel_flag["check_fn"]()
                else:
                    should_cancel = cancel_flag.get("should_cancel", False)
                if should_cancel:
                    add_log(f"🛑 加载被中断（TC Layer {layer}/{n_layers-1}）")
                    raise InterruptedError("加载被用户中断")
            
            # 检查该层是否已经加载
            if layer in transcoders:
                add_log(f"  [TC Layer {layer}/{n_layers-1}] ✅ 已缓存，跳过加载")
                continue
            
            tc_path = f"{tc_base_path}/L{layer}"
            add_log(f"  [TC Layer {layer}/{n_layers-1}] 开始加载: {tc_path}")
            logger.info(f"📁 加载TC L{layer}: {tc_path}")
            start_time = time.time()
            transcoders[layer] = SparseAutoEncoder.from_pretrained(
                tc_path,
                dtype=torch.float32,
                device=device,
            )
            load_time = time.time() - start_time
            add_log(f"  [TC Layer {layer}/{n_layers-1}] ✅ 加载完成，耗时: {load_time:.2f}秒")
        
        add_log(f"✅ 所有Transcoders加载完成，共{len(transcoders)}层")
        
        # 初始化或获取已有的lorsas缓存（使用 cache_key）
        if cache_key not in _global_lorsas_cache:
            _global_lorsas_cache[cache_key] = []
        lorsas = _global_lorsas_cache[cache_key]
        
        # 加载LORSA（逐层检查，避免重复加载）
        add_log(f"🔍 开始加载LoRSAs，共{n_layers}层...")
        for layer in range(n_layers):
            # 检查是否应该中断加载
            if cancel_flag is not None:
                # 如果有检查函数，调用它；否则直接检查 should_cancel
                if "check_fn" in cancel_flag and callable(cancel_flag["check_fn"]):
                    should_cancel = cancel_flag["check_fn"]()
                else:
                    should_cancel = cancel_flag.get("should_cancel", False)
                if should_cancel:
                    add_log(f"🛑 加载被中断（LoRSA Layer {layer}/{n_layers-1}）")
                    raise InterruptedError("加载被用户中断")
            
            # 检查该层是否已经加载
            if layer < len(lorsas):
                add_log(f"  [LoRSA Layer {layer}/{n_layers-1}] ✅ 已缓存，跳过加载")
                continue
            
            lorsa_path = f"{lorsa_base_path}/L{layer}"
            add_log(f"  [LoRSA Layer {layer}/{n_layers-1}] 开始加载: {lorsa_path}")
            logger.info(f"📁 加载LORSA L{layer}: {lorsa_path}")
            start_time = time.time()
            lorsas.append(LowRankSparseAttention.from_pretrained(
                lorsa_path,
                device=device
            ))
            load_time = time.time() - start_time
            add_log(f"  [LoRSA Layer {layer}/{n_layers-1}] ✅ 加载完成，耗时: {load_time:.2f}秒")
        
        add_log(f"✅ 所有LoRSAs加载完成，共{len(lorsas)}层")
        
        # 创建替换模型
        add_log("🔍 创建ReplacementModel...")
        replacement_model = ReplacementModel.from_pretrained_model(
            model, transcoders, lorsas
        )
        add_log("✅ ReplacementModel创建完成")
        
        # 缓存所有加载的模型（使用 cache_key）
        set_cached_models(cache_key, model, transcoders, lorsas, replacement_model)
        add_log(f"✅ 模型、transcoders和lorsas已缓存: {cache_key}")
        
        return replacement_model, transcoders, lorsas
    except Exception as e:
        # 任何异常（包括 OOM）时，清理当前 cache_key 下已加载的 SAE，避免占用显存
        add_log(f"❌ 加载过程中出错，将清空缓存 {cache_key}: {e}")
        try:
            # 将已加载的 SAE 挪到 CPU 再删除引用
            if cache_key in _global_transcoders_cache:
                for sae in _global_transcoders_cache[cache_key].values():
                    try:
                        if hasattr(sae, "to"):
                            sae.to("cpu")
                    except Exception:
                        continue
                del _global_transcoders_cache[cache_key]
            if cache_key in _global_lorsas_cache:
                for sae in _global_lorsas_cache[cache_key]:
                    try:
                        if hasattr(sae, "to"):
                            sae.to("cpu")
                    except Exception:
                        continue
                del _global_lorsas_cache[cache_key]
            if cache_key in _global_replacement_models_cache:
                del _global_replacement_models_cache[cache_key]
        finally:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    add_log("🧹 已在异常后调用 torch.cuda.empty_cache() 释放显存")
            except Exception:
                pass
        # 将异常继续抛出，让上层处理 HTTP 错误码等
        raise
    
    finally:
        # 释放加载锁（使用 cache_key）
        with _loading_lock:
            _is_loading[cache_key] = False
            add_log(f"🔓 释放加载锁: {cache_key}")


def setup_mongodb(mongo_uri: str, mongo_db: str) -> Optional[MongoClient]:
    """设置MongoDB连接"""
    logger = logging.getLogger(__name__)
    
    try:
        mongo_config = MongoDBConfig(
            mongo_uri=mongo_uri,
            mongo_db=mongo_db
        )
        mongo_client = MongoClient(mongo_config)
        logger.info(f"MongoDB连接成功: {mongo_config.mongo_db}")
        return mongo_client
    except Exception as e:
        logger.warning(f"MongoDB连接失败: {e}")
        return None


def run_attribution(
    model: ReplacementModel,
    prompt: str,
    fen: str,
    move_uci: str,
    side: str,
    max_n_logits: int,
    desired_logit_prob: float,
    max_feature_nodes: int,
    batch_size: int,
    order_mode: str,
    mongo_client: Optional[MongoClient],
    sae_series: str,
    act_times_max: Optional[int] = None,
    encoder_demean: bool = False,
    save_activation_info: bool = False,
    negative_move_uci: Optional[str] = None  # 新增negative_move_uci参数
) -> Dict[str, Any]:
    """运行attribution分析"""
    logger = logging.getLogger(__name__)
    
    # 设置棋盘
    lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
    is_castle = False  # 可以根据需要调整
    
    # 处理move_idx：根据order_mode和negative_move_uci决定
    if order_mode == 'move_pair':
        # move_pair模式：需要positive和negative move
        if not negative_move_uci:
            raise ValueError("negative_move_uci is required for move_pair mode")
        positive_move_idx = lboard.uci2idx(move_uci)
        negative_move_idx = lboard.uci2idx(negative_move_uci)
        move_idx = (positive_move_idx, negative_move_idx)
        logger.info(f"Move pair mode: positive_move_idx={positive_move_idx}, negative_move_idx={negative_move_idx}")
    else:
        # positive或negative模式：只有一个move
        move_idx = lboard.uci2idx(move_uci)
    
    # 设置梯度
    torch.set_grad_enabled(True)
    model.reset_hooks()
    model.zero_grad(set_to_none=True)
    
    # 运行attribution
    logger.info(f"开始attribution分析: {prompt}")
    start_time = time.time()
    
    attribution_result = attribute(
        prompt=prompt,
        model=model,
        is_castle=is_castle,
        side=side,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=None,
        update_interval=4,
        use_legal_moves_only=False,
        fen=fen,
        lboard=lboard,
        move_idx=move_idx,
        encoder_demean=encoder_demean,
        act_times_max=act_times_max,
        mongo_client=mongo_client,
        sae_series=sae_series,
        analysis_name='default',
        order_mode=order_mode,
        save_activation_info=save_activation_info,
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Attribution分析完成，耗时: {elapsed_time:.2f}s")
    
    return attribution_result


def create_graph_from_attribution(
    model,
    attribution_result: Dict[str, Any],
    prompt: str,
    side: str,
    slug: str,  # 将 slug 移到前面
    sae_series: Optional[str] = None,
) -> Graph:
    """
    从attribution结果创建Graph对象
    
    Args:
        model: 替换模型实例
        attribution_result: Attribution结果字典
        prompt: 输入提示
        side: 分析侧 ('q', 'k', 或 'both')
        slug: 图的标识符
        sae_series: SAE系列名称
    
    Returns:
        Graph: 创建的图对象
    """
    logger = logging.getLogger(__name__)
    logger.info(f"正在为侧'{side}'创建图对象...")
    try:
        # 提取公共数据
        lorsa_activation_matrix = attribution_result['lorsa_activations']['lorsa_activation_matrix']
        tc_activation_matrix = attribution_result['tc_activations']['tc_activation_matrix']
        input_embedding = attribution_result['input']['input_embedding']
        logit_idx = attribution_result['logits']['indices']
        logit_p = attribution_result['logits']['probabilities']
        lorsa_active_features = attribution_result['lorsa_activations']['indices']
        lorsa_activation_values = attribution_result['lorsa_activations']['values']
        tc_active_features = attribution_result['tc_activations']['indices']
        tc_activation_values = attribution_result['tc_activations']['values']
        
        # 根据side选择对应的数据
        if side == 'q':
            q_data = attribution_result.get('q')
            if q_data is None:
                raise ValueError("Attribution结果中没有找到'q'侧数据")
            full_edge_matrix = q_data['full_edge_matrix']
            selected_features = q_data['selected_features']
            side_logit_position = q_data.get('move_positions')
            activation_info = attribution_result.get('activation_info', {}).get('q')
            
        elif side == 'k':
            k_data = attribution_result.get('k')
            if k_data is None:
                raise ValueError("Attribution结果中没有找到'k'侧数据")
            full_edge_matrix = k_data['full_edge_matrix']
            selected_features = k_data['selected_features']
            side_logit_position = k_data.get('move_positions')
            activation_info = attribution_result.get('activation_info', {}).get('k')
            
        elif side == 'both':
            # 处理both情况，需要合并q和k侧的数据
            q_data = attribution_result.get('q')
            k_data = attribution_result.get('k')
            if q_data is None or k_data is None:
                raise ValueError("Attribution结果中没有找到'q'或'k'侧数据，无法进行both模式合并")
            
            # 导入merge_qk_graph函数
            from lm_saes.circuit.attribution_qk_for_feature_attribution import merge_qk_graph
            
            logger.info("开始合并q和k侧数据...")
            merged = merge_qk_graph(attribution_result)
            
            full_edge_matrix = merged["adjacency_matrix"]
            selected_features = merged["selected_features"]
            side_logit_position = merged["logit_position"]
            
            # 使用merge_qk_graph返回的合并激活信息
            activation_info = merged.get("activation_info")
            logger.info(f"合并完成，包含 {len(selected_features)} 个选中特征")
            
        else:
            raise ValueError(f"不支持的侧: {side}")
        
        # 创建Graph对象
        graph = Graph(
            input_string=prompt,
            input_tokens=input_embedding,
            logit_tokens=logit_idx,
            logit_probabilities=logit_p,
            logit_position=side_logit_position,
            lorsa_active_features=lorsa_active_features,
            lorsa_activation_values=lorsa_activation_values,
            tc_active_features=tc_active_features,
            tc_activation_values=tc_activation_values,
            selected_features=selected_features,
            adjacency_matrix=full_edge_matrix,
            cfg=model.cfg,
            sae_series=sae_series,
            slug=slug,
            activation_info=activation_info,
        )
        
        logger.info(f"成功创建图对象，包含 {len(selected_features)} 个选中特征")
        return graph
        
    except Exception as e:
        logger.error(f"创建图对象时出错: {e}")
        raise


def create_graph_json_data(
    graph: Graph,
    slug: str,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    sae_series: Optional[str] = None,
    lorsa_analysis_name: str = "",
    tc_analysis_name: str = "",
) -> Dict[str, Any]:
    """创建graph的JSON数据，不保存到文件"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"开始创建graph JSON数据: {slug}")
    start_time = time.time()
    
    if sae_series is None:
        if graph.sae_series is None:
            raise ValueError(
                "Neither sae_series nor graph.sae_series was set. One must be set to identify "
                "which transcoders were used when creating the graph."
            )
        sae_series = graph.sae_series

    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph.to(device)
    
    fen = graph.input_string
    lboard = None
    if fen:
        print(f'in graph input_string {fen = }')
        lboard = LeelaBoard.from_fen(fen)
    else:
        print('[Warning] fen is none')
        
    to_uci = lboard.idx2uci if lboard is not None else None 
    
    if isinstance(graph.logit_tokens, torch.Tensor):
        _logit_idxs = graph.logit_tokens.view(-1).tolist()
    else:
        _logit_idxs = list(graph.logit_tokens)
    
    
    logit_moves = [
        (to_uci(int(i)) if to_uci is not None else f"idx:{int(i)}")
        for i in _logit_idxs
    ]
    target_move = logit_moves[0] if logit_moves else None
    
    print(f'{target_move = }') 
    print(f'{graph.adjacency_matrix.shape = }')
    
    node_mask, edge_mask, cumulative_scores = (
        el.to(device) for el in prune_graph(graph, node_threshold, edge_threshold)
    )

    nodes = create_nodes(graph, node_mask, cumulative_scores, to_uci = to_uci)
    used_nodes, used_edges = create_used_nodes_and_edges(graph, nodes, edge_mask)
    model = build_model(
        graph=graph,
        used_nodes=used_nodes,
        used_edges=used_edges,
        slug=slug,
        sae_series=sae_series,
        node_threshold=node_threshold,
        lorsa_analysis_name=lorsa_analysis_name,
        tc_analysis_name=tc_analysis_name,
        logit_moves = logit_moves,
        target_move = target_move,
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Graph JSON数据创建完成，耗时: {elapsed_time:.2f}s")
    
    return model.model_dump()


def run_circuit_trace(
    prompt: str,
    move_uci: str,
    negative_move_uci: Optional[str] = None,  # 新增negative_move_uci参数
    model_name: str = "lc0/BT4-1024x15x32h",
    device: str = "cuda",
    tc_base_path: str = BT4_TC_BASE_PATH,
    lorsa_base_path: str = BT4_LORSA_BASE_PATH,
    n_layers: int = 15,
    side: str = "both",
    max_n_logits: int = 1,
    desired_logit_prob: float = 0.95,
    max_feature_nodes: int = 4096,
    batch_size: int = 1,
    order_mode: str = "positive",
    mongo_uri: str = "mongodb://10.245.40.143:27017",
    mongo_db: str = "mechinterp",
    sae_series: str = "BT4-exp128",
    act_times_max: Optional[int] = None,
    encoder_demean: bool = False,
    save_activation_info: bool = False,
    node_threshold: float = 0.73,
    edge_threshold: float = 0.57,
    log_level: str = "INFO",
    hooked_model: Optional[HookedTransformer] = None,  # 新增参数
    cached_transcoders: Optional[Dict[int, SparseAutoEncoder]] = None,  # 新增：缓存的transcoders
    cached_lorsas: Optional[List[LowRankSparseAttention]] = None,  # 新增：缓存的lorsas
    cached_replacement_model: Optional[ReplacementModel] = None,  # 新增：缓存的replacement_model
    sae_combo_id: Optional[str] = None,  # 新增：SAE组合ID，用于生成正确的analysis_name模板
    trace_logs: Optional[list] = None  # 新增：用于存储日志的列表
) -> Dict[str, Any]:
    """运行circuit trace并返回graph数据"""
    logger = setup_logging(log_level)
    
    # 如果提供了trace_logs，使用日志捕获
    if trace_logs is not None:
        log_capture = LogCapture(trace_logs)
        log_capture.__enter__()
    else:
        log_capture = None
    
    # 设置设备
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA不可用，切换到CPU")
        device = "cpu"
    
    try:
        # 合法性检测：验证move_uci在prompt fen下是否合法
        board = chess.Board(prompt)
        legal_uci_moves = [move.uci() for move in board.legal_moves]
        if move_uci not in legal_uci_moves:
            logger.error(f"❌ 移动 {move_uci} 在fen {prompt} 下不合法！")
            raise Exception(f"不合法的UCI移动: {move_uci} 不在fen {prompt}的合法走法中。\n合法走法列表: {legal_uci_moves}")

        # 加载模型（如果已有缓存则使用缓存）
        if cached_replacement_model is not None and cached_transcoders is not None and cached_lorsas is not None:
            logger.info("使用缓存的模型、transcoders和lorsas...")
            model = cached_replacement_model
            transcoders = cached_transcoders
            lorsas = cached_lorsas
        else:
            print("加载模型和transcoders...")
            print(f'{lorsa_base_path = }')
            print(f'{tc_base_path = }')
            
            logger.info("加载模型和transcoders...")
            model, transcoders, lorsas = load_model_and_transcoders(
                model_name, device, tc_base_path, 
                lorsa_base_path, n_layers, hooked_model  # 传递hooked_model
            )
        
        # 设置MongoDB
        mongo_client = setup_mongodb(mongo_uri, mongo_db)
        print(f'DEBUG: mongo_client = {mongo_client}')
        # 生成slug
        slug = f'circuit_trace_{order_mode}_{side}_{max_feature_nodes}'
        
        # 运行attribution
        attribution_result = run_attribution(
            model=model,
            prompt=prompt,
            fen=prompt,
            move_uci=move_uci,
            side=side,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            max_feature_nodes=max_feature_nodes,
            batch_size=batch_size,
            order_mode=order_mode,
            mongo_client=mongo_client,
            sae_series=sae_series,
            act_times_max=act_times_max,
            encoder_demean=encoder_demean,
            save_activation_info=True,  # 强制设置为True以获取激活信息
            negative_move_uci=negative_move_uci  # 传递negative_move_uci
        )
        
        # 创建Graph
        logger.info("创建Graph对象...")
        graph = create_graph_from_attribution(
            model=model,
            attribution_result=attribution_result,
            prompt=prompt,
            side=side,
            slug=slug,
            sae_series=sae_series
        )
        
        # 根据sae_combo_id获取对应的analysis_name（从BT4_SAE_COMBOS配置中读取）
        # 如果sae_combo_id为None，尝试从tc_base_path和lorsa_base_path推断
        try:
            try:
                from .constants import get_bt4_sae_combo, BT4_SAE_COMBOS
            except ImportError:
                from constants import get_bt4_sae_combo, BT4_SAE_COMBOS
            
            # 如果sae_combo_id为None，尝试从路径推断
            if sae_combo_id is None:
                # 从路径中提取组合ID（例如：/path/to/tc/k_30_e_16 -> k_30_e_16）
                import os
                tc_path_parts = os.path.normpath(tc_base_path).split(os.sep)
                lorsa_path_parts = os.path.normpath(lorsa_base_path).split(os.sep)
                
                # 查找路径中的组合ID（通常在路径的最后几部分）
                inferred_combo_id = None
                for combo_id in BT4_SAE_COMBOS.keys():
                    if combo_id in tc_path_parts or combo_id in lorsa_path_parts:
                        inferred_combo_id = combo_id
                        break
                
                if inferred_combo_id:
                    sae_combo_id = inferred_combo_id
                    logger.info(f"从路径推断SAE组合ID: {sae_combo_id}")
                else:
                    logger.warning(f"无法从路径推断SAE组合ID，使用默认组合")
            
            combo_cfg = get_bt4_sae_combo(sae_combo_id)
            # 直接从配置中读取analysis_name字段（如果存在），否则回退到模板字段
            lorsa_analysis_name = combo_cfg.get("lorsa_analysis_name", combo_cfg.get("lorsa_sae_name_template", ""))
            tc_analysis_name = combo_cfg.get("tc_analysis_name", combo_cfg.get("tc_sae_name_template", ""))
            logger.info(f"使用SAE组合 {combo_cfg['id']} 的analysis_name: LoRSA={lorsa_analysis_name}, TC={tc_analysis_name}")
        except Exception as e:
            logger.warning(f"无法获取SAE组合配置，使用空字符串: {e}")
            import traceback
            traceback.print_exc()
            lorsa_analysis_name = ""
            tc_analysis_name = ""
        
        # 创建JSON数据，传递analysis_name（从BT4_SAE_COMBOS配置中读取）
        graph_data = create_graph_json_data(
            graph, slug, node_threshold, edge_threshold, 
            sae_series, lorsa_analysis_name, tc_analysis_name
        )
        
        logger.info("Circuit trace分析完成!")
        
        # 退出日志捕获
        if log_capture is not None:
            log_capture.__exit__(None, None, None)
        
        return graph_data
        
    except Exception as e:
        logger.error(f"有点问题: {e}")
        # logger.error(f"执行过程中发生错误: {e}")
        
        # 确保退出日志捕获
        if log_capture is not None:
            log_capture.__exit__(None, None, None)
        
        raise


def save_graph_files(
    graph: Graph,
    slug: str,
    output_path: str,
    node_threshold: float = 0.9,
    edge_threshold: float = 0.69
) -> None:
    """保存graph文件"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"开始保存graph文件到: {output_path}")
    start_time = time.time()
    
    create_graph_files(
        graph=graph,
        slug=slug,
        output_path=output_path,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Graph文件保存完成，耗时: {elapsed_time:.2f}s")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Fast tracing test for chess SAE attribution")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="lc0/BT4-1024x15x32h",
                       help="模型名称")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备 (cuda/cpu)")
    parser.add_argument("--n_layers", type=int, default=15,
                       help="模型层数")
    
    # 路径参数
    parser.add_argument("--tc_base_path", type=str, 
                       default="/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/tc/k_128_e_128",
                       help="TC模型基础路径")
    parser.add_argument("--lorsa_base_path", type=str,
                       default="/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/lorsa/k_128_e_128",
                       help="LORSA模型基础路径")
    parser.add_argument("--output_path", type=str,
                       default="/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/graphs/fast_tracing",
                       help="输出路径")
    
    # 分析参数
    parser.add_argument("--prompt", type=str, default="2k5/4Q3/3P4/8/6p1/4p3/q1pbK3/1R6 b - - 0 32",
                       help="FEN字符串")
    parser.add_argument("--move_uci", type=str, default="a2c4",
                       help="要分析的UCI移动")
    parser.add_argument("--side", type=str, default="k", choices=["q", "k", "both"],
                       help="分析侧 (q/k/both)")
    parser.add_argument("--max_n_logits", type=int, default=1,
                       help="最大logit数量")
    parser.add_argument("--desired_logit_prob", type=float, default=0.95,
                       help="期望logit概率")
    parser.add_argument("--max_feature_nodes", type=int, default=1024,
                       help="最大特征节点数")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批处理大小")
    parser.add_argument("--order_mode", type=str, default="positive",
                       choices=["positive", "negative", "move_pair", "group"],
                       help="排序模式")
    
    # MongoDB参数
    parser.add_argument("--mongo_uri", type=str, default="mongodb://10.245.40.143:27017",
                       help="MongoDB URI")
    parser.add_argument("--mongo_db", type=str, default="mechinterp",
                       help="MongoDB数据库名")
    parser.add_argument("--sae_series", type=str, default="BT4",
                       help="SAE系列名")
    parser.add_argument("--act_times_max", type=lambda x: int(x) if x.lower() != "none" else None, default=None, help="最大激活次数 (可选)")
    
    # 其他参数
    parser.add_argument("--encoder_demean", action="store_true",
                       help="是否对encoder进行demean")
    parser.add_argument("--save_activation_info", action="store_true",
                       help="是否保存激活信息")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    parser.add_argument("--node_threshold", type=float, default=0.73,
                       help="节点阈值")
    parser.add_argument("--edge_threshold", type=float, default=0.57,
                       help="边阈值")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    # 设置设备
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA不可用，切换到CPU")
        args.device = "cpu"
    
    try:
        # 加载模型
        logger.info("加载模型和transcoders...")
        model, transcoders, lorsas = load_model_and_transcoders(
            args.model_name, args.device, args.tc_base_path, 
            args.lorsa_base_path, args.n_layers
        )
        
        # 设置MongoDB
        mongo_client = setup_mongodb(args.mongo_uri, args.mongo_db)
        
        # 生成slug
        slug = f'fast_tracing_{args.side}_{args.max_feature_nodes}'
        
        # 运行attribution
        attribution_result = run_attribution(
            model=model,
            prompt=args.prompt,
            fen=args.prompt,
            move_uci=args.move_uci,
            side=args.side,
            max_n_logits=args.max_n_logits,
            desired_logit_prob=args.desired_logit_prob,
            max_feature_nodes=args.max_feature_nodes,
            batch_size=args.batch_size,
            order_mode=args.order_mode,
            mongo_client=mongo_client,
            sae_series=args.sae_series,
            act_times_max=args.act_times_max,
            encoder_demean=args.encoder_demean,
            save_activation_info=args.save_activation_info
        )
        
        # 创建Graph
        logger.info("创建Graph对象...")
        graph = create_graph_from_attribution(
            model=model,
            attribution_result=attribution_result,
            prompt=args.prompt,
            side=args.side,
            slug=slug,
            sae_series=args.sae_series
        )
        
        # 保存文件
        save_graph_files(
            graph, slug, args.output_path, 
            args.node_threshold, args.edge_threshold
        )
        
        logger.info("分析完成!")
        
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}")
        raise


def check_dense_features(
    nodes: List[Dict[str, Any]],
    threshold: Optional[int],
    mongo_client: Optional[MongoClient],
    sae_series: str = "BT4-exp128",
    lorsa_analysis_name: Optional[str] = None,
    tc_analysis_name: Optional[str] = None
) -> List[str]:
    """
    检查哪些节点是dense feature（激活次数超过阈值）
    
    Args:
        nodes: 节点列表，每个节点包含node_id, feature, layer, feature_type等信息
        threshold: 激活次数阈值，None表示无限大（所有节点都不是dense）
        mongo_client: MongoDB客户端
        sae_series: SAE系列名称
        lorsa_analysis_name: LoRSA分析名称模板（如 ""）
        tc_analysis_name: TC分析名称模板（如 "BT4_tc_L{}M"）BT4_lorsa_L{}A
    
    Returns:
        dense节点的node_id列表
    """
    logger = logging.getLogger(__name__)
    
    if threshold is None:
        # 阈值为None，所有节点都不是dense
        return []
    
    if mongo_client is None:
        logger.warning("MongoDB客户端不可用，无法检查dense features")
        return []
    
    # 打印传入的模板参数
    logger.info(f"🔍 Dense检查参数: lorsa_analysis_name={lorsa_analysis_name}, tc_analysis_name={tc_analysis_name}, threshold={threshold}")
    
    dense_node_ids = []
    not_dense_nodes = []  # 记录非dense节点用于调试
    
    for node in nodes:
        try:
            node_id = node.get('node_id')
            feature_idx = node.get('feature')
            layer = node.get('layer')
            feature_type = node.get('feature_type', '').lower()
            
            if node_id is None or feature_idx is None or layer is None:
                logger.debug(f"跳过节点 {node_id}: 缺少必要信息")
                continue
            
            # 构建SAE名称
            sae_name = None
            if 'lorsa' in feature_type:
                if lorsa_analysis_name:
                    # 使用提供的模板
                    sae_name = lorsa_analysis_name.replace("{}", str(layer))
                else:
                    # 默认格式
                    sae_name = f"lc0-lorsa-L{layer}"
            elif 'transcoder' in feature_type or 'cross layer transcoder' in feature_type:
                if tc_analysis_name:
                    # 使用提供的模板
                    sae_name = tc_analysis_name.replace("{}", str(layer))
                else:
                    # 默认格式
                    sae_name = f"lc0_L{layer}M_16x_k30_lr2e-03_auxk_sparseadam"
            else:
                logger.debug(f"跳过节点 {node_id}: 未知特征类型 {feature_type}")
                continue
            
            # 详细打印每个节点的analysis_name
            logger.info(f"📋 节点 {node_id}: feature_type={feature_type}, layer={layer}, feature={feature_idx}, sae_name={sae_name}")
            
            # 从MongoDB获取该特征的激活次数
            feature_data = mongo_client.get_feature(
                sae_name=sae_name,
                sae_series=sae_series,
                index=feature_idx
            )
            
            if feature_data is None:
                logger.warning(f"❌ 节点 {node_id}: 在MongoDB中未找到特征数据 (sae={sae_name}, sae_series={sae_series}, idx={feature_idx})")
                not_dense_nodes.append({
                    'node_id': node_id,
                    'reason': 'MongoDB中未找到',
                    'sae_name': sae_name,
                    'sae_series': sae_series,
                    'feature_idx': feature_idx
                })
                continue
            
            # 获取该特征的激活次数
            if feature_data.analyses:
                analysis = feature_data.analyses[0]
                act_times = getattr(analysis, 'act_times', 0)
                
                logger.info(f"📊 节点 {node_id}: act_times={act_times}, threshold={threshold}, sae_name={sae_name}")
                
                if act_times > threshold:
                    dense_node_ids.append(node_id)
                    logger.info(f"✅ Dense节点: {node_id} (act_times={act_times} > threshold={threshold})")
                else:
                    not_dense_nodes.append({
                        'node_id': node_id,
                        'reason': f'act_times={act_times} <= threshold={threshold}',
                        'sae_name': sae_name,
                        'act_times': act_times
                    })
                    logger.info(f"⚪ 非Dense节点: {node_id} (act_times={act_times} <= threshold={threshold})")
            else:
                logger.warning(f"❌ 节点 {node_id}: 没有分析数据")
                not_dense_nodes.append({
                    'node_id': node_id,
                    'reason': '没有分析数据',
                    'sae_name': sae_name
                })
            
        except Exception as e:
            logger.warning(f"检查节点 {node.get('node_id')} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"📈 统计: 总节点={len(nodes)}, Dense节点={len(dense_node_ids)}, 非Dense节点={len(not_dense_nodes)}")
    if not_dense_nodes:
        logger.info(f"🔍 非Dense节点详情（前10个）:")
        for node_info in not_dense_nodes[:10]:
            logger.info(f"  - {node_info}")
    
    return dense_node_ids


if __name__ == "__main__":
    main()
