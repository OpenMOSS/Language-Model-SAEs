from lm_saes import SparseAutoEncoder, LowRankSparseAttention
from transformer_lens import HookedTransformer

import torch
import sys
from pathlib import Path
from collections import defaultdict
import json
from tqdm.auto import tqdm
from typing import Dict, Any, Optional, List
import pandas as pd
import argparse
from multiprocessing import Pool, cpu_count, set_start_method
import traceback
import os

import chess
project_root = Path.cwd().parent.parent.parent
sys.path.append(str(project_root))
from src.chess_utils import get_move_from_policy_output_with_prob
from src.feature_and_steering import analyze_position_features_comprehensive

model_name = 'lc0/BT4-1024x15x32h'

_model = None
_transcoders = None
_lorsas = None

def get_model():
    global _model
    if _model is None:
        _model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    dtype=torch.float32,
).eval()
    return _model

def get_transcoders():
    global _transcoders
    if _transcoders is None:
        _transcoders = {
    layer: SparseAutoEncoder.from_pretrained(
        f'/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/tc/k_30_e_16/L{layer}',
        dtype=torch.float32,
        device='cuda',
    )
    for layer in range(15)
}
    return _transcoders

def get_lorsas():
    global _lorsas
    if _lorsas is None:
        _lorsas = [
            LowRankSparseAttention.from_pretrained(
                f'/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/lorsa/k_30_e_16/L{layer}', 
                dtype=torch.float32,
                device='cuda',
            )
            for layer in range(15)
        ]
    return _lorsas

model = get_model()
transcoders = get_transcoders()
lorsas = get_lorsas()


def get_top_K_moves(fen: str, model: HookedTransformer, K: int) -> Dict[str, float]:
    output, _ = model.run_with_cache(fen, prepend_bos=False)
    policy_output = output[0]
    legal_moves_with_probs = get_move_from_policy_output_with_prob(
        policy_output, 
        fen, 
        return_list=True
    )
    
    if not legal_moves_with_probs:
        return {}
    
    sorted_moves = sorted(legal_moves_with_probs, key=lambda x: x[2], reverse=True)[:K]
    result = {move_uci: prob for move_uci, _, prob in sorted_moves}
    return result


def features_for_all_move_to_json(
    fen: str,
    model: HookedTransformer,
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    move_prob_dict: Dict[str, float],
    steering_scale: float = 0.0,
    activation_threshold: float = 0.0,
    max_features_per_type: Optional[int] = None,
    max_steering_features: Optional[int] = None,
) -> Dict[str, Any]:
    MOVES_TRACING = {move_uci: move_uci for move_uci in move_prob_dict.keys()}
    FEATURE_TYPES = ['transcoder', 'lorsa']
    FEN = fen
    POS_DICT = {f"pos_{i}": i for i in range(64)}
    
    all_results = {}
    total_features_count = 0
    
    for pos_idx in tqdm(range(64), desc="分析位置"):
        position_name = f"pos_{pos_idx}"
        moves_to_trace = MOVES_TRACING.copy()
        
        analysis_result = analyze_position_features_comprehensive(
            pos_dict=POS_DICT,
            position_name=position_name,
            model=model,
            transcoders=transcoders,
            lorsas=lorsas,
            fen=FEN,
            moves_tracing=moves_to_trace,
            feature_types=FEATURE_TYPES,
            steering_scale=steering_scale,
            activation_threshold=activation_threshold,
            max_features_per_type=max_features_per_type,
            max_steering_features=max_steering_features
        )
        analysis_result['moves_tracing'] = move_prob_dict.copy()
        
        all_results[position_name] = analysis_result
        total_features_count += len(analysis_result['results'])
    
    serializable_result = all_results.copy()
    for position_name, analysis_result in serializable_result.items():
        for result in analysis_result['results']:
            if 'activation_value' in result and hasattr(result['activation_value'], 'item'):
                result['activation_value'] = result['activation_value'].item()
            numeric_fields = ['original_value', 'modified_value', 'value_diff', 'steering_scale']
            for key in numeric_fields:
                if key in result and hasattr(result[key], 'item'):
                    result[key] = result[key].item()
            
            if 'move_probabilities' in result:
                for move_name, prob_info in result['move_probabilities'].items():
                    if isinstance(prob_info, dict):
                        for prob_key in ['original_prob', 'modified_prob', 'prob_diff']:
                            if prob_key in prob_info and hasattr(prob_info[prob_key], 'item'):
                                prob_info[prob_key] = prob_info[prob_key].item()
    
    serializable_result['fen'] = FEN
    return serializable_result


def get_top_n_features_for_each_move(
    all_results: Dict[str, Any],
    move_prob_dict: Dict[str, float],
    n: int,
    fen: str
) -> Dict[str, pd.DataFrame]:
    result_dfs = {}
    
    for move_uci, prob in move_prob_dict.items():
        all_features_data = []
        
        for position_name, analysis_result in all_results.items():
            if 'results' not in analysis_result:
                continue
            
            for result in analysis_result['results']:
                if 'move_probabilities' not in result:
                    continue
                
                move_probs = result['move_probabilities'].get(move_uci, {})
                if not isinstance(move_probs, dict):
                    continue
                
                prob_diff = move_probs.get('prob_diff')
                if prob_diff is None:
                    continue
                
                feature_data = {
                    'position_name': position_name,
                    'position_idx': int(position_name.split('_')[1]) if '_' in position_name else 0,
                    'layer': result.get('layer', 0),
                    'feature_id': result.get('feature_id', 0),
                    'feature_type': result.get('feature_type', 'unknown'),
                    'activation_value': result.get('activation_value', 0.0),
                    'steering_scale': result.get('steering_scale', 0.0),
                    'move': move_uci,
                    'prob_diff': prob_diff,
                    'original_prob': move_probs.get('original_prob', 0.0),
                    'modified_prob': move_probs.get('modified_prob', 0.0),
                    'fen': fen,
                }
                
                all_features_data.append(feature_data)
        
        if not all_features_data:
            continue
        
        df = pd.DataFrame(all_features_data)
        top_n = df.nsmallest(n, 'prob_diff')
        result_dfs[move_uci] = top_n
    
    return result_dfs


def process_single_fen(
    fen: str,
    K: int,
    n: int,
    output_dir: Path,
    steering_scale: float = 0.0,
    activation_threshold: float = 0.0,
    max_features_per_type: Optional[int] = None,
    max_steering_features: Optional[int] = None,
):
    local_model = get_model()
    local_transcoders = get_transcoders()
    local_lorsas = get_lorsas()
    
    fen_safe = fen.replace('/', '_').replace(' ', '_').replace('-', '_').replace(':', '_')
    if len(fen_safe) > 100:
        fen_safe = fen_safe[:100]
    fen_output_dir = output_dir / f"fen_{fen_safe}"
    fen_output_dir.mkdir(parents=True, exist_ok=True)
    
    top_K_moves = get_top_K_moves(fen, local_model, K)
    if not top_K_moves:
        return
    
    all_results = features_for_all_move_to_json(
        fen=fen,
        model=local_model,
        transcoders=local_transcoders,
        lorsas=local_lorsas,
        move_prob_dict=top_K_moves,
        steering_scale=steering_scale,
        activation_threshold=activation_threshold,
        max_features_per_type=max_features_per_type,
        max_steering_features=max_steering_features
    )
    
    output_json = fen_output_dir / "infl_all_feature.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    feature_dfs = get_top_n_features_for_each_move(
        all_results=all_results,
        move_prob_dict=top_K_moves,
        n=n,
        fen=fen
    )
    
    for move_uci, df in feature_dfs.items():
        output_csv = fen_output_dir / f"{move_uci}_top{n}_features.csv"
        df.to_csv(output_csv, index=False)
    
    with open(fen_output_dir / "fen.txt", 'w', encoding='utf-8') as f:
        f.write(fen)


def process_single_fen_wrapper(args):
    fen, K, n, output_dir_str, steering_scale, activation_threshold, max_features_per_type, max_steering_features = args
    try:
        import traceback
        process_single_fen(
            fen=fen,
            K=K,
            n=n,
            output_dir=Path(output_dir_str),
            steering_scale=steering_scale,
            activation_threshold=activation_threshold,
            max_features_per_type=max_features_per_type,
            max_steering_features=max_steering_features
        )
        return {'fen': fen, 'status': 'success'}
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return {'fen': fen, 'status': 'failed', 'error': error_msg}


def process_txt_file(
    txt_file: str,
    output_dir: str,
    K: int,
    n: int,
    steering_scale: float = 0.0,
    activation_threshold: float = 0.0,
    max_features_per_type: Optional[int] = None,
    max_steering_features: Optional[int] = None,
    num_workers: Optional[int] = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fens = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('FEN:'):
            fen = line[4:].strip()
            if fen:
                fens.append(fen)
    
    total_fens = len(fens)
    print(f"找到 {total_fens} 个FEN需要处理")
    
    if num_workers is None:
        num_workers = min(cpu_count(), 4)
    
    if num_workers > 1:
        print(f"使用 {num_workers} 个进程并行处理")
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        args_list = [
            (fen, K, n, str(output_path), steering_scale, activation_threshold, max_features_per_type, max_steering_features)
            for fen in fens
        ]
        
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_fen_wrapper, args_list),
                total=total_fens,
                desc="处理FEN"
            ))
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        failed_count = sum(1 for r in results if r['status'] == 'failed')
        print(f"处理完成！成功: {success_count}, 失败: {failed_count}")
        
        if failed_count > 0:
            failed_results = [r for r in results if r['status'] == 'failed']
            print(f"失败的FEN (前10个):")
            for r in failed_results[:10]:
                print(f"  {r['fen']}")
                if 'error' in r:
                    error_lines = r['error'].split('\n')
                    print(f"    错误: {error_lines[0]}")
            if len(failed_results) > 10:
                print(f"  ... 还有 {len(failed_results) - 10} 个失败的FEN")
    else:
        print("使用单进程处理")
        for idx, fen in enumerate(tqdm(fens, desc="处理FEN")):
            try:
                process_single_fen(
                    fen=fen,
                    K=K,
                    n=n,
                    output_dir=output_path,
                    steering_scale=steering_scale,
                    activation_threshold=activation_threshold,
                    max_features_per_type=max_features_per_type,
                    max_steering_features=max_steering_features
                )
            except Exception as e:
                print(f"处理FEN失败 (索引 {idx}): {e}")
                traceback.print_exc()
                continue
    
    print(f"处理完成！结果保存在: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_file", type=str, required=True, help="Path to txt file containing FEN positions")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--K", type=int, required=True, help="Number of top moves to consider")
    parser.add_argument("--num_features", type=int, required=True, help="Number of top features to save for each move")
    parser.add_argument("--steering_scale", type=float, default=0.0, help="Steering scale")
    parser.add_argument("--activation_threshold", type=float, default=0.0, help="Activation threshold")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers (default: min(cpu_count, 4))")
    args = parser.parse_args()
    
    process_txt_file(
        txt_file=args.txt_file,
        output_dir=args.output_dir,
        K=args.K,
        n=args.num_features,
        steering_scale=args.steering_scale,
        activation_threshold=args.activation_threshold,
        max_features_per_type=None,
        max_steering_features=None,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
