import os
import json
import time
import subprocess
import uuid
from typing import Optional, Tuple, Dict, Any

from pathlib import Path
import tempfile

# 计算项目根目录与默认输出目录（ui/public/circuits）
SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ui" / "public" / "circuits"
SCRIPTS_TRACE_TO_GRAPH = PROJECT_ROOT / "scripts" / "trace_to_graph.py"


def ensure_output_dir(path: Optional[str] = None) -> Path:
    out_dir = Path(path) if path else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def make_slug(prefix: str = "gen") -> str:
    ts = int(time.time())
    uid = uuid.uuid4().hex[:6]
    return f"{prefix}_{ts}_{uid}"


def run_trace_to_graph(
    *,
    fen: str,
    move_uci: str,
    side: str = "k",
    node_threshold: float = 0.5,
    edge_threshold: float = 0.3,
    sae_series: str = "lc0-tc",
    model_name: str = "lc0/T82-768x15x24h",
    n_layers: int = 15,
    transcoder_dir_tpl: Optional[str] = None,
    output_dir: Optional[str] = None,
    slug: Optional[str] = None,
    save: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """调用 scripts/trace_to_graph.py 生成电路图文件，并加载返回JSON内容。

    Returns:
      (graph_json, saved_path, slug)
    """
    # 选择输出目录：保存则使用指定/默认目录；不保存则使用临时目录
    tempdir_ctx = None
    if save:
        out_dir = ensure_output_dir(output_dir)
    else:
        tempdir_ctx = tempfile.TemporaryDirectory()
        out_dir = Path(tempdir_ctx.name)

    slug = slug or make_slug("circuit")

    # 默认的transcoder路径模板（与脚本默认保持一致）
    if transcoder_dir_tpl is None:
        transcoder_dir_tpl = (
            "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/"
            "rlin_projects/chess-SAEs/result/tc/"
            "lc0_L{layer}M_16x_k30_lr2e-03_auxk_sparseadam"
        )

    # 组织命令行参数
    cmd = [
        os.environ.get("PYTHON", "python"),
        str(SCRIPTS_TRACE_TO_GRAPH),
        "--fen",
        fen,
        "--move_uci",
        move_uci,
        "--device",
        "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu",
        "--max_n_logits",
        "1",
        "--desired_logit_prob",
        "0.95",
        "--batch_size",
        "1",
        "--max_feature_nodes",
        "2048",
        "--side",
        side,
        "--slug",
        slug,
        "--output_path",
        str(out_dir),
        "--node_threshold",
        str(node_threshold),
        "--edge_threshold",
        str(edge_threshold),
        "--sae_series",
        sae_series,
        "--analysis_name",
        "default",
        "--model_name",
        model_name,
        "--n_layers",
        str(n_layers),
        "--transcoder_dir_tpl",
        transcoder_dir_tpl,
    ]

    # 执行脚本
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        if tempdir_ctx is not None:
            tempdir_ctx.cleanup()
        return None, None, None

    # 约定脚本会写出 {slug}.json
    json_path = out_dir / f"{slug}.json"
    if not json_path.exists():
        # 兼容可能不同命名的输出（如包含前缀）——粗略回退：寻找最近修改的json
        json_files = sorted(
            out_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if json_files:
            json_path = json_files[0]
        else:
            if tempdir_ctx is not None:
                tempdir_ctx.cleanup()
            return None, None, None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        saved_path = str(json_path) if save else None
        return content, saved_path, slug
    except Exception:
        return None, None, None
    finally:
        if tempdir_ctx is not None:
            tempdir_ctx.cleanup() 