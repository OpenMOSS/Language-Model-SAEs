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


def load_model_and_transcoders(
    model_name: str,
    device: str,
    tc_base_path: str,
    lorsa_base_path: str,
    n_layers: int = 15,
    hooked_model: Optional[HookedTransformer] = None  # 新增参数
) -> Tuple[ReplacementModel, Dict[int, SparseAutoEncoder], List[LowRankSparseAttention]]:
    """加载模型和transcoders"""
    logger = logging.getLogger(__name__)
    
    # 使用传入的模型或加载新模型
    if hooked_model is not None:
        logger.info("使用传入的HookedTransformer模型")
        model = hooked_model
    else:
        logger.info("加载新的HookedTransformer模型")
        model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            dtype=torch.float32,
        ).eval()
    
    # 加载transcoders
    transcoders = {}
    for layer in range(n_layers):
        tc_path = f"{tc_base_path}/lc0_L{layer}M_16x_k30_lr2e-03_auxk_sparseadam"
        transcoders[layer] = SparseAutoEncoder.from_pretrained(
            tc_path,
            dtype=torch.float32,
            device=device,
        )
    
    # 加载LORSA
    lorsas = []
    for layer in range(n_layers):
        lorsa_path = f"{lorsa_base_path}/lc0_L{layer}_bidirectional_lr8e-05_k_aux4096_coefficient0.0625_dead_threshold1000000"
        lorsas.append(LowRankSparseAttention.from_pretrained(
            lorsa_path,
            device=device
        ))
    
    # 创建替换模型
    replacement_model = ReplacementModel.from_pretrained_model(
        model, transcoders, lorsas
    )
    
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
    save_activation_info: bool = False
) -> Dict[str, Any]:
    """运行attribution分析"""
    logger = logging.getLogger(__name__)
    
    # 设置棋盘
    lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
    move_idx = lboard.uci2idx(move_uci)
    is_castle = False  # 可以根据需要调整
    
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
    """从attribution结果创建Graph对象"""
    logger = logging.getLogger(__name__)
    
    # 提取数据
    lorsa_activation_matrix = attribution_result['lorsa_activations']['lorsa_activation_matrix']
    tc_activation_matrix = attribution_result['tc_activations']['tc_activation_matrix']
    input_embedding = attribution_result['input']['input_embedding']
    logit_idx = attribution_result['logits']['indices']
    logit_p = attribution_result['logits']['probabilities']
    lorsa_active_features = attribution_result['lorsa_activations']['indices']
    lorsa_activation_values = attribution_result['lorsa_activations']['values']
    tc_active_features = attribution_result['tc_activations']['indices']
    tc_activation_values = attribution_result['tc_activations']['values']
    full_edge_matrix = attribution_result[side]['full_edge_matrix']
    selected_features = attribution_result[side]['selected_features']
    side_logit_position = attribution_result[side]['move_positions']
    
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
        activation_info=attribution_result.get('activation_info'),
    )
    
    logger.info(f"Graph创建完成: {slug}")
    return graph


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
    model_name: str = "lc0/T82-768x15x24h",
    device: str = "cuda",
    tc_base_path: str = "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/result/tc",
    lorsa_base_path: str = "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/result/lorsa",
    n_layers: int = 15,
    side: str = "k",
    max_n_logits: int = 1,
    desired_logit_prob: float = 0.95,
    max_feature_nodes: int = 1024,
    batch_size: int = 1,
    order_mode: str = "positive",
    mongo_uri: str = "mongodb://10.244.136.216:27017",
    mongo_db: str = "mechinterp",
    sae_series: str = "lc0-circuit-tracing",
    act_times_max: Optional[int] = None,
    encoder_demean: bool = False,
    save_activation_info: bool = False,
    node_threshold: float = 0.9,
    edge_threshold: float = 0.69,
    log_level: str = "INFO",
    hooked_model: Optional[HookedTransformer] = None  # 新增参数
) -> Dict[str, Any]:
    """运行circuit trace并返回graph数据"""
    logger = setup_logging(log_level)
    
    # 设置设备
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA不可用，切换到CPU")
        device = "cpu"
    
    try:
        # 加载模型
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
            save_activation_info=save_activation_info
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
        logger.error(f"执行过程中发生错误: {e}")
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
    parser.add_argument("--model_name", type=str, default="lc0/T82-768x15x24h",
                       help="模型名称")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备 (cuda/cpu)")
    parser.add_argument("--n_layers", type=int, default=15,
                       help="模型层数")
    
    # 路径参数
    parser.add_argument("--tc_base_path", type=str, 
                       default="/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/result/tc",
                       help="TC模型基础路径")
    parser.add_argument("--lorsa_base_path", type=str,
                       default="/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/result/lorsa",
                       help="LORSA模型基础路径")
    parser.add_argument("--output_path", type=str,
                       default="/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/graphs/fast_tracing",
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
    parser.add_argument("--mongo_uri", type=str, default="mongodb://10.244.136.216:27017",
                       help="MongoDB URI")
    parser.add_argument("--mongo_db", type=str, default="mechinterp",
                       help="MongoDB数据库名")
    parser.add_argument("--sae_series", type=str, default="lc0-circuit-tracing",
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
    parser.add_argument("--node_threshold", type=float, default=0.9,
                       help="节点阈值")
    parser.add_argument("--edge_threshold", type=float, default=0.69,
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


if __name__ == "__main__":
    main()
