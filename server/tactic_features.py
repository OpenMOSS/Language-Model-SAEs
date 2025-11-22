import chess
import torch
import collections
from typing import Dict, List, Tuple, Set
from datasets import load_from_disk

try:
    from transformer_lens import HookedTransformer
    from lm_saes.sae import SparseAutoEncoder
    from lm_saes import LowRankSparseAttention
    HOOKED_TRANSFORMER_AVAILABLE = True
except ImportError:
    HookedTransformer = None
    SparseAutoEncoder = None
    LowRankSparseAttention = None
    HOOKED_TRANSFORMER_AVAILABLE = False


def validate_fens(fens: List[str]) -> Tuple[List[str], List[str]]:
    """验证FEN字符串列表，返回有效和无效的FEN"""
    valid_fens = []
    invalid_fens = []
    
    for fen in fens:
        fen = fen.strip()
        if not fen:
            continue
        try:
            board = chess.Board(fen)
            valid_fens.append(fen)
        except ValueError:
            invalid_fens.append(fen)
    
    return valid_fens, invalid_fens


@torch.no_grad()
def collect_activation_stats(
    fens: List[str],
    model: HookedTransformer,
    lorsas: List,
    transcoders: Dict[int, any],
    log_every: int = 25
) -> Tuple[List[collections.Counter], List[collections.Counter], int]:
    """统计FEN数据集上各层LoRSA/TC特征是否激活"""
    num_layers = len(lorsas)
    lorsa_counts: List[collections.Counter] = [collections.Counter() for _ in range(num_layers)]
    tc_counts: List[collections.Counter] = [collections.Counter() for _ in range(num_layers)]
    num_ok = 0

    for idx, fen in enumerate(fens):
        try:
            _, cache = model.run_with_cache(fen, prepend_bos=False)
        except Exception as e:
            continue

        num_ok += 1

        for layer in range(num_layers):
            # LoRSA 激活
            try:
                lorsa_input = cache[f'blocks.{layer}.hook_attn_in']
                lorsa_dense = lorsas[layer].encode(lorsa_input)
                lorsa_sparse = lorsa_dense.to_sparse_coo()
                if lorsa_sparse._nnz() > 0:
                    feat_idx = lorsa_sparse.indices()[2]
                    for f in torch.unique(feat_idx).tolist():
                        lorsa_counts[layer][int(f)] += 1
            except Exception:
                pass

            # TC 激活
            try:
                tc_input = cache[f'blocks.{layer}.resid_mid_after_ln']
                tc_dense = transcoders[layer].encode(tc_input)
                tc_sparse = tc_dense.to_sparse_coo()
                if tc_sparse._nnz() > 0:
                    feat_idx = tc_sparse.indices()[2]
                    for f in torch.unique(feat_idx).tolist():
                        tc_counts[layer][int(f)] += 1
            except Exception:
                pass

        if (idx + 1) % log_every == 0:
            print(f"processed {idx+1}/{len(fens)} (valid {num_ok})")

    return lorsa_counts, tc_counts, num_ok




def compute_all_diffs(
    rand_lorsa_counts: List[collections.Counter],
    rand_tc_counts: List[collections.Counter],
    tactic_lorsa_counts: List[collections.Counter],
    tactic_tc_counts: List[collections.Counter],
    n_random: int,
    n_tactic: int,
) -> Tuple[List[Tuple[int, int, float, float, float, str]], List[Tuple[int, int, float, float, float, str]]]:
    """计算LoRSA和TC的所有差异"""
    lorsa_results = []
    tc_results = []
    num_layers = len(rand_lorsa_counts)
    
    for layer in range(num_layers):
        # LoRSA
        c_r = rand_lorsa_counts[layer]
        c_t = tactic_lorsa_counts[layer]
        keys = set(c_r.keys()) | set(c_t.keys())
        for f in keys:
            pr = (c_r.get(f, 0) / max(1, n_random)) if n_random > 0 else 0.0
            pt = (c_t.get(f, 0) / max(1, n_tactic)) if n_tactic > 0 else 0.0
            lorsa_results.append((layer, int(f), pt - pr, pr, pt, 'LoRSA'))
        
        # TC
        c_r = rand_tc_counts[layer]
        c_t = tactic_tc_counts[layer]
        keys = set(c_r.keys()) | set(c_t.keys())
        for f in keys:
            pr = (c_r.get(f, 0) / max(1, n_random)) if n_random > 0 else 0.0
            pt = (c_t.get(f, 0) / max(1, n_tactic)) if n_tactic > 0 else 0.0
            tc_results.append((layer, int(f), pt - pr, pr, pt, 'TC'))
    
    return lorsa_results, tc_results


def get_random_fens(n: int = 500, dataset_path: str = None) -> List[str]:
    """从数据集中随机采样FEN"""
    if dataset_path is None:
        dataset_path = "/inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/chess_master_data"
    
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)
        dataset_size = len(dataset)
        random_indices = torch.randperm(dataset_size)[:min(n, dataset_size)].tolist()
        random_data = [dataset[i]['fen'] for i in random_indices]
        return random_data
    except Exception as e:
        print(f"Warning: Could not load random dataset: {e}")
        return []


def analyze_tactic_features(
    tactic_fens: List[str],
    model: HookedTransformer,
    lorsas: List,
    transcoders: Dict[int, any],
    n_random: int = 500,
    dataset_path: str = None,
) -> Dict[str, any]:
    """分析战术特征，返回差异最大的特征"""
    # 验证FEN
    valid_tactic_fens, invalid_tactic_fens = validate_fens(tactic_fens)
    
    if not valid_tactic_fens:
        return {
            "error": "No valid FENs found",
            "invalid_fens": invalid_tactic_fens
        }
    
    # 获取随机FEN
    random_fens = get_random_fens(n_random, dataset_path)
    if not random_fens:
        return {
            "error": "Could not load random FEN dataset"
        }
    
    # 收集激活统计
    print("Collecting activation stats on random_fens...")
    rand_lorsa_counts, rand_tc_counts, rand_ok = collect_activation_stats(
        random_fens, model, lorsas, transcoders
    )
    
    print("Collecting activation stats on tactic_fens...")
    tactic_lorsa_counts, tactic_tc_counts, tactic_ok = collect_activation_stats(
        valid_tactic_fens, model, lorsas, transcoders
    )
    
    # 计算差异
    lorsa_diffs, tc_diffs = compute_all_diffs(
        rand_lorsa_counts, rand_tc_counts,
        tactic_lorsa_counts, tactic_tc_counts,
        rand_ok, tactic_ok
    )
    
    return {
        "valid_tactic_fens": len(valid_tactic_fens),
        "invalid_tactic_fens": len(invalid_tactic_fens),
        "random_fens": rand_ok,
        "tactic_fens": tactic_ok,
        "lorsa_diffs": lorsa_diffs,
        "tc_diffs": tc_diffs,
        "invalid_fens_list": invalid_tactic_fens[:10]  # 只返回前10个无效FEN作为示例
    }
