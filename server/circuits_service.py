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
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch
import chess
from transformer_lens import HookedTransformer

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

# 导入项目模块
from lm_saes import ReplacementModel, LowRankSparseAttention, SparseAutoEncoder
from lm_saes.circuit.attribution_qk import attribute
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


# 全局缓存（与app.py共享）
_global_hooked_models: Dict[str, HookedTransformer] = {}
_global_transcoders_cache: Dict[str, Dict[int, SparseAutoEncoder]] = {}
_global_lorsas_cache: Dict[str, List[LowRankSparseAttention]] = {}
_global_replacement_models_cache: Dict[str, ReplacementModel] = {}


def get_cached_models(model_name: str) -> Tuple[Optional[HookedTransformer], Optional[Dict[int, SparseAutoEncoder]], Optional[List[LowRankSparseAttention]], Optional[ReplacementModel]]:
    """获取缓存的模型、transcoders和lorsas"""
    global _global_hooked_models, _global_transcoders_cache, _global_lorsas_cache, _global_replacement_models_cache
    
    hooked_model = _global_hooked_models.get(model_name)
    transcoders = _global_transcoders_cache.get(model_name)
    lorsas = _global_lorsas_cache.get(model_name)
    replacement_model = _global_replacement_models_cache.get(model_name)
    
    return hooked_model, transcoders, lorsas, replacement_model


def set_cached_models(
    model_name: str,
    hooked_model: HookedTransformer,
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    replacement_model: ReplacementModel
):
    """设置缓存的模型、transcoders和lorsas"""
    global _global_hooked_models, _global_transcoders_cache, _global_lorsas_cache, _global_replacement_models_cache
    
    _global_hooked_models[model_name] = hooked_model
    _global_transcoders_cache[model_name] = transcoders
    _global_lorsas_cache[model_name] = lorsas
    _global_replacement_models_cache[model_name] = replacement_model


def load_model_and_transcoders(
    model_name: str,
    device: str,
    tc_base_path: str,
    lorsa_base_path: str,
    n_layers: int = 15,
    hooked_model: Optional[HookedTransformer] = None,  # 新增参数
    loading_logs: Optional[list] = None  # 新增参数：用于收集加载日志
) -> Tuple[ReplacementModel, Dict[int, SparseAutoEncoder], List[LowRankSparseAttention]]:
    """加载模型和transcoders（带全局缓存）"""
    logger = logging.getLogger(__name__)
    
    # 辅助函数：添加日志（同时打印到控制台和收集到日志列表）
    def add_log(message: str):
        print(message)
        logger.info(message)
        if loading_logs is not None:
            loading_logs.append({
                "timestamp": time.time(),
                "message": message
            })
    
    # 先检查全局缓存
    cached_hooked_model, cached_transcoders, cached_lorsas, cached_replacement_model = get_cached_models(model_name)
    
    # 检查缓存是否完整（有transcoders和lorsas，且层数正确）
    if cached_transcoders is not None and cached_lorsas is not None:
        if len(cached_transcoders) == n_layers and len(cached_lorsas) == n_layers:
            if cached_replacement_model is not None:
                add_log(f"✅ 使用缓存的模型、transcoders和lorsas: {model_name}")
                logger.info(f"✅ 从缓存加载: {model_name} (transcoders={len(cached_transcoders)}层, lorsas={len(cached_lorsas)}层)")
                return cached_replacement_model, cached_transcoders, cached_lorsas
    
    # 如果缓存不完整或不存在，则加载
    add_log(f"🔍 开始加载模型和transcoders: {model_name}")
    
    # 使用传入的模型或从缓存获取或加载新模型
    if hooked_model is not None:
        logger.info("使用传入的HookedTransformer模型")
        model = hooked_model
    elif cached_hooked_model is not None:
        logger.info("使用缓存的HookedTransformer模型")
        model = cached_hooked_model
    else:
        logger.info("加载新的HookedTransformer模型")
        model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            dtype=torch.float32,
        ).eval()
        # 缓存模型
        _global_hooked_models[model_name] = model
    
    # 加载transcoders
    add_log(f"🔍 开始加载Transcoders，共{n_layers}层...")
    transcoders = {}
    for layer in range(n_layers):
        # 根据模型名称选择不同的路径格式
        # if 'BT4' in model_name:
        #     # BT4路径格式: L{layer}
        #     tc_path = f"{tc_base_path}/L{layer}"
        # else:
        #     # 默认T82路径格式
        #     tc_path = f"{tc_base_path}/lc0_L{layer}M_16x_k30_lr2e-03_auxk_sparseadam"
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
    
    # 加载LORSA
    add_log(f"🔍 开始加载LoRSAs，共{n_layers}层...")
    lorsas = []
    for layer in range(n_layers):
        # 根据模型名称选择不同的路径格式
        # if 'BT4' in model_name:
        #     # BT4路径格式: L{layer}
        #     lorsa_path = f"{lorsa_base_path}/lc0_L{layer}_bidirectional_lr0.0002_k_aux4096_coefficient0.125_dead_threshold1000000"
        # else:
        #     # 默认T82路径格式
        #     lorsa_path = f"{lorsa_base_path}/lc0_L{layer}_bidirectional_lr8e-05_k_aux4096_coefficient0.0625_dead_threshold1000000"
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
    replacement_model = ReplacementModel.from_pretrained_model(
        model, transcoders, lorsas
    )
    
    # 缓存所有加载的模型
    set_cached_models(model_name, model, transcoders, lorsas, replacement_model)
    add_log(f"✅ 模型、transcoders和lorsas已缓存: {model_name}")
    
    return replacement_model, transcoders, lorsas


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
            from lm_saes.circuit.attribution_qk import merge_qk_graph
            
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
    tc_base_path: str = "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/tc",
    lorsa_base_path: str = "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/lorsa",
    n_layers: int = 15,
    side: str = "both",
    max_n_logits: int = 1,
    desired_logit_prob: float = 0.95,
    max_feature_nodes: int = 4096,
    batch_size: int = 1,
    order_mode: str = "positive",
    mongo_uri: str = "mongodb://10.244.94.234:27017",
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
    cached_replacement_model: Optional[ReplacementModel] = None  # 新增：缓存的replacement_model
) -> Dict[str, Any]:
    """运行circuit trace并返回graph数据"""
    logger = setup_logging(log_level)
    
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
            print("✅ 使用缓存的模型、transcoders和lorsas...")
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
        
        # 创建JSON数据
        graph_data = create_graph_json_data(
            graph, slug, node_threshold, edge_threshold, 
            sae_series, "", ""
        )
        
        logger.info("Circuit trace分析完成!")
        return graph_data
        
    except Exception as e:
        logger.error(f"有点问题: {e}")
        # logger.error(f"执行过程中发生错误: {e}")
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
                       default="/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/tc",
                       help="TC模型基础路径")
    parser.add_argument("--lorsa_base_path", type=str,
                       default="/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/lorsa",
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
    parser.add_argument("--mongo_uri", type=str, default="mongodb://10.244.94.234:27017",
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
        lorsa_analysis_name: LoRSA分析名称模板（如 "BT4_lorsa_L{}A"）
        tc_analysis_name: TC分析名称模板（如 "BT4_tc_L{}M"）
    
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
