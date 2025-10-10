# run this script:
'''
python scripts/trace_to_graph.py \
  --fen "2k5/4Q3/3P4/8/6p1/4p3/q1pbK3/1R6 b - - 0 32" \
  --move_uci a2c4 \
  --cuda_visible_devices 1 \
  --device cuda \
  --max_n_logits 1 \
  --desired_logit_prob 0.95 \
  --batch_size 1 \
  --max_feature_nodes 4096 \
  --side k \
  --slug win_or_go_home_k_4096_B \
  --output_path /path/to/graphs \
  --node_threshold 0.86 \
  --edge_threshold 0.51 \
  --sae_series lc0-circuit-tracing \
  --analysis_name default
'''
import argparse
import os
import sys
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

# 项目根目录加入 sys.path，便于相对导入
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from lm_saes import SparseAutoEncoder, LowRankSparseAttention
from lm_saes import ReplacementModel
from lm_saes.config import MongoDBConfig
from lm_saes.database import MongoClient
from src.lm_saes.circuit.attribution_qk import attribute
from src.lm_saes.circuit.graph_lc0 import Graph
from src.lm_saes.circuit.utils.create_graph_files import create_graph_files
from src.lm_saes.circuit.leela_board import LeelaBoard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run attribution and export graph JSON for lc0 model"
    )

    # 基础参数
    parser.add_argument(
        "--model_name", type=str, default="lc0/T82-768x15x24h",
        help="HookedTransformer model name"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device for inference"
    )

    # SAE/Transcoder 路径模板（按层展开）
    transcoder_default = (
        "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
        "rlin_projects/chess-SAEs/result/tc/"
        "lc0_L{layer}M_16x_k30_lr2e-03_auxk_sparseadam"
    )
    parser.add_argument(
        "--transcoder_dir_tpl",
        type=str,
        default=transcoder_default,
        help="Transcoder directory template with {layer} placeholder",
    )
    
    lorsa_default = (
        "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
        "rlin_projects/chess-SAEs/result/lorsa/"
        "lc0_L{layer}_bidirectional_lr8e-05_k_aux4096_coefficient0.0625_"
        "dead_threshold1000000"
    )
    parser.add_argument(
        "--lorsa_dir_tpl",
        type=str,
        default=lorsa_default,
        help="LoRSA directory template with {layer} placeholder",
    )
    parser.add_argument(
        "--n_layers", type=int, default=15,
        help="Number of layers to load SAEs for"
    )

    # 归因/图相关参数
    parser.add_argument(
        "--fen", type=str, required=True,
        help="FEN string of the chess position"
    )
    parser.add_argument(
        "--move_uci", type=str, required=True,
        help="Target move in UCI format, e.g. a2c4"
    )
    parser.add_argument(
        "--side", type=str, default="k", choices=["k", "q", "both"],
        help="Which side attribution to compute"
    )
    parser.add_argument(
        "--is_castle", action="store_true",
        help="Whether target move is castle (affects k side)"
    )
    parser.add_argument(
        "--max_n_logits", type=int, default=1,
        help="Max number of logits to attribute from"
    )
    parser.add_argument(
        "--desired_logit_prob", type=float, default=0.95,
        help="Cumulative logit prob mass to cover"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Attribution batch size"
    )
    parser.add_argument(
        "--max_feature_nodes", type=int, default=4096,
        help="Max feature nodes to attribute"
    )
    parser.add_argument(
        "--update_interval", type=int, default=4,
        help="Ranking update interval"
    )
    parser.add_argument(
        "--use_legal_moves_only", action="store_true",
        help="Restrict logits to legal moves only"
    )
    parser.add_argument(
        "--encoder_demean", action="store_true",
        help="Enable encoder de-mean in attribution"
    )
    parser.add_argument(
        "--act_times_max", type=int, default=70_000_000,
        help="Max act times for filtering features"
    )
    parser.add_argument(
        "--sae_series", type=str, default="lc0-circuit-tracing",
        help="Identifier for the SAE series"
    )
    parser.add_argument(
        "--analysis_name", type=str, default="default",
        help="Analysis tag"
    )
    parser.add_argument(
        "--order_mode",
        type=str,
        default="positive",
        choices=["positive", "negative", "move_pair", "group"],
        help="Ordering strategy for features",
    )
    parser.add_argument(
        "--save_activation_info", action="store_true",
        help="Save activation info in attribution result"
    )

    # 输出参数
    parser.add_argument(
        "--slug", type=str, default="win_or_go_home_k_4096",
        help="Output slug"
    )
    parser.add_argument(
        "--output_path", type=str, default=str(PROJECT_ROOT / "graphs"),
        help="Output directory for JSON"
    )
    parser.add_argument(
        "--node_threshold", type=float, default=0.86,
        help="Pruning node threshold"
    )
    parser.add_argument(
        "--edge_threshold", type=float, default=0.51,
        help="Pruning edge threshold"
    )

    # MongoDB 参数（可选）
    parser.add_argument(
        "--mongo_uri", type=str, default=None,
        help="MongoDB connection URI, e.g. mongodb://host:27017"
    )
    parser.add_argument(
        "--mongo_db", type=str, default=None,
        help="MongoDB database name, e.g. mechinterp"
    )
    parser.add_argument(
        "--disable_mongo", action="store_true",
        help="Disable MongoDB even if URI/DB provided or env available"
    )

    # CUDA 可见设备（可选）
    parser.add_argument(
        "--cuda_visible_devices", type=str, default=None,
        help="Set CUDA_VISIBLE_DEVICES before running"
    )

    return parser


def load_model_transcoders_and_lorsas(
    model_name: str, n_layers: int, transcoder_dir_tpl: str,
    lorsa_dir_tpl: str, device: str
):
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        dtype=torch.float32,
    ).eval()

    transcoders = {
        layer: SparseAutoEncoder.from_pretrained(
            transcoder_dir_tpl.format(layer=layer),
            dtype=torch.float32,
            device=device,
            fold_activation_scale=True,
        )
        for layer in range(n_layers)
    }

    lorsas = [
        LowRankSparseAttention.from_pretrained(
            lorsa_dir_tpl.format(layer=layer),
            device=device
        )
        for layer in range(n_layers)
    ]

    replaced = ReplacementModel.from_pretrained_model(
        model, transcoders, lorsas
    )
    return replaced


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    device = args.device

    # 准备输入：将 FEN 直接作为 prompt（lc0 项目下游会将其解析为棋盘）
    prompt = args.fen

    # 加载模型与 SAEs
    model = load_model_transcoders_and_lorsas(
        model_name=args.model_name,
        n_layers=args.n_layers,
        transcoder_dir_tpl=args.transcoder_dir_tpl,
        lorsa_dir_tpl=args.lorsa_dir_tpl,
        device=device,
    )

    # 构建 LeelaBoard 与 move_idx
    lboard = LeelaBoard.from_fen(args.fen, history_synthesis=True)
    move_idx = lboard.uci2idx(args.move_uci)

    # 初始化 Mongo 客户端（可选）
    mongo_client = None
    if not args.disable_mongo:
        # 允许从参数或环境变量读取配置
        mongo_uri = args.mongo_uri or os.environ.get("MONGO_URI")
        mongo_db = args.mongo_db or os.environ.get("MONGO_DB")
        if mongo_uri or mongo_db:
            try:
                cfg = MongoDBConfig(
                    mongo_uri=mongo_uri or "mongodb://localhost:27017/",
                    mongo_db=mongo_db or "mechinterp",
                )
                mongo_client = MongoClient(cfg)
                # 连接探测
                mongo_client.client.admin.command("ping")
                print(f"✓ MongoDB连接成功 | db={cfg.mongo_db} uri={cfg.mongo_uri}")
            except Exception as e:
                print(f"❌ MongoDB连接失败: {e}. 将以本地模式继续运行。")
                mongo_client = None

    # 归因
    torch.set_grad_enabled(True)
    model.reset_hooks()
    model.zero_grad(set_to_none=True)

    attribution_result = attribute(
        prompt=prompt,
        model=model,
        is_castle=args.is_castle,
        side=args.side,
        max_n_logits=args.max_n_logits,
        desired_logit_prob=args.desired_logit_prob,
        batch_size=args.batch_size,
        max_feature_nodes=args.max_feature_nodes,
        offload=None,
        update_interval=args.update_interval,
        use_legal_moves_only=args.use_legal_moves_only,
        fen=args.fen,
        lboard=lboard,
        move_idx=move_idx,
        encoder_demean=args.encoder_demean,
        act_times_max=args.act_times_max,
        mongo_client=mongo_client,
        sae_series=args.sae_series,
        analysis_name=args.analysis_name,
        order_mode=args.order_mode,
        save_activation_info=args.save_activation_info,
    )

    # 从 attribution_result 构造 Graph 对象
    input_embedding = attribution_result["input"]["input_embedding"]
    logit_idx = attribution_result["logits"]["indices"]
    logit_p = attribution_result["logits"]["probabilities"]

    lorsa_active_features = attribution_result["lorsa_activations"]["indices"]
    lorsa_activation_values = attribution_result["lorsa_activations"]["values"]
    tc_active_features = attribution_result["tc_activations"]["indices"]
    tc_activation_values = attribution_result["tc_activations"]["values"]

    side_key = args.side
    full_edge_matrix = attribution_result[side_key]["full_edge_matrix"]
    selected_features = attribution_result[side_key]["selected_features"]
    side_logit_position = attribution_result[side_key]["move_positions"]

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
        sae_series=args.sae_series,
        slug=args.slug,
        activation_info=attribution_result.get("activation_info", None),
    )

    # 写出 JSON
    create_graph_files(
        graph=graph,
        slug=args.slug,
        output_path=args.output_path,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
    )


if __name__ == "__main__":
    main()