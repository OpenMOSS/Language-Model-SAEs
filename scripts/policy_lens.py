# run this script:
'''
python scripts/policy_lens.py \
  --fen "2k5/4Q3/3P4/8/6p1/4p3/q1pbK3/1R6 b - - 0 32" \
  --target_move a2c4 \
  --model_name lc0/T82-768x15x24h \
  --device cuda \
  --output_dir ./policy_lens_results \
  --save_plots \
  --save_data
'''
import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import chess
import chess.svg

from transformer_lens import HookedTransformer

# 项目根目录加入 sys.path，便于相对导入
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# 添加 leela-interp 路径
sys.path.insert(0, str(PROJECT_ROOT / "exp" / "leela-interp" / "src"))

try:
    from leela_interp import LeelaBoard  # type: ignore
except ImportError:
    print("Warning: leela_interp not found. Make sure the path is correct.")
    LeelaBoard = None


class Lc0LayerNorm(nn.Module):
    """LayerNorm implementation for lc0 model"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.w = nn.Parameter(torch.ones(normalized_shape))
        self.b = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.w * (x - mean) / torch.sqrt(var + self.eps) + self.b


class IntegratedPolicyLens:
    def __init__(self, model_name: str = 'lc0/T82-768x15x24h'):
        print("Loading model...")
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            dtype=torch.float32,
        ).eval()
        
        self.policy_head = self.model.policy_head
        self.num_layers = len(getattr(self.model, "blocks", [])) or 15
        print(f"Detected {self.num_layers} layers.")
        
        # cache for layernorm copies and alpha
        self.ln_copies: List[Tuple[Lc0LayerNorm, Lc0LayerNorm, nn.Parameter, nn.Parameter]] = []  # noqa: E501
        self.ln_orig: List[Tuple[Lc0LayerNorm, Lc0LayerNorm]] = []
        
        # create non-destructive copies of layernorms (with b zeroed) to use in logit-lens
        self.cache_layernorm()
        
        print("Model loaded successfully!")
        
    def cache_layernorm(self):
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise RuntimeError("Model does not have 'blocks' attribute!")
        
        self.ln_copies = []
        self.ln_orig = []
        
        first_param = next(self.model.parameters())
        target_device = first_param.device
        target_dtype = first_param.dtype
        
        for i, layer in enumerate(blocks):
            ln1 = getattr(layer, "ln1", None)
            ln2 = getattr(layer, "ln2", None)
            alpha_input = getattr(layer, "alpha_input", None)
            alpha_out1 = getattr(layer, "alpha_out1", None)
            
            if ln1 is None or ln2 is None or alpha_input is None or alpha_out1 is None:
                raise RuntimeError(f"Layer {i} missing expected attributes.")
            # keep reference to original LN modules (not modified)
            self.ln_orig.append((ln1, ln2))
            
            # create deep copies to be used for post-ln logit-lens transforms
            ln1_copy = copy.deepcopy(ln1)
            ln2_copy = copy.deepcopy(ln2)
    
            ln1_copy.to(device=target_device, dtype=target_dtype)
            ln2_copy.to(device=target_device, dtype=target_dtype)
            # zero the bias on the copies only (original model untouched)
            with torch.no_grad():
                ln1_copy.b.data.zero_()
                ln2_copy.b.data.zero_()
            # store (ln1_copy, ln2_copy, alpha_input, alpha_out1)
            self.ln_copies.append((ln1_copy, ln2_copy, alpha_input, alpha_out1))
        
        print(f"Created LayerNorm copies for {len(self.ln_copies)} layers (orig model untouched).")
                
    def apply_postln_truncation(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Given h = resid_post_after_ln at layer_idx (i.e. after ln2 of that layer),
        apply subsequent layers' alpha and their LN copies (with b=0) while zeroing sublayer outputs.
        Returns transformed representation suitable for feeding to policy_head.
        """
        # ensure we don't modify input in-place
        x = h
        for layer in range(layer_idx + 1, self.num_layers):
            ln1_copy, ln2_copy, alpha_in, alpha_out = self.ln_copies[layer]
            # multiply by scalar alphas (ensure dtype/device broadcasting)
            x = x * alpha_in  # attn branch zero + alpha_in * x
            x = ln1_copy(x)   # apply copy with b=0
            x = x * alpha_out  # mlp branch zero + alpha_out * x
            x = ln2_copy(x)
        return x

    def analyze_single_fen(self, fen: str, target_move: Optional[str] = None,
                          topk_vocab: int = 2000) -> Optional[Dict[str, Any]]:
        """
        Analyze a FEN position. If `target_move` is provided (UCI string), 
        compute its rank/score per layer.
        topk_vocab: how many top logits to consider when searching for legal moves
        (for efficiency).
        """
        print(f"Analyzing FEN: {fen}  target_move: {target_move}")
        try:
            board = chess.Board(fen)
            if board.is_game_over():
                print("Warning: Game is over for this position")
        except ValueError as e:
            print(f"Invalid FEN: {e}")
            return None

        with torch.no_grad():
            output, cache = self.model.run_with_cache(fen, prepend_bos=False)
        
        # Final layer predictions (unchanged behavior)
        final_layer_input = cache[f'blocks.{self.num_layers-1}.resid_post_after_ln']  # [1, seq, d]  # noqa: E501
        final_policy_output = self.policy_head(final_layer_input)
        
        if LeelaBoard is None:
            raise RuntimeError("LeelaBoard not available. Please check leela_interp import.")  # noqa: E501

        lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
        chess_board = chess.Board(fen)
        legal_uci_set = set(move.uci() for move in chess_board.legal_moves)
        
        # Compute final layer top legal moves (for reference)
        if final_policy_output.dim() == 3:
            final_logits_last = final_policy_output[0, -1, :]
        elif final_policy_output.dim() == 2:
            final_logits_last = final_policy_output[0, :]
        else:
            raise RuntimeError("Unexpected final_policy_output shape")
        
        # get sorted final logits (for ranking)
        final_vals, final_idxs = torch.sort(final_logits_last, descending=True)
        final_vals = final_vals.detach().cpu().numpy()
        final_idxs = final_idxs.detach().cpu().numpy()
        final_top_legal_moves = []
        # collect top 10 legal moves based on final logits
        for idx_val, idx in zip(final_vals, final_idxs):
            uci = lboard.idx2uci(int(idx))
            if uci in legal_uci_set:
                final_top_legal_moves.append((int(idx), uci, float(idx_val)))
            if len(final_top_legal_moves) >= 10:
                break
        
        # Analyze each layer using Post-LN logit-lens (apply_postln_truncation)
        layer_analysis = {}
        
        for layer in range(self.num_layers):
            print(f"Analyzing layer {layer}...")
            
            layer_input = cache[f'blocks.{layer}.resid_post_after_ln']  # after ln2 at that layer  # noqa: E501
            truncated_rep = self.apply_postln_truncation(layer_input, layer)
            
            # Now compute logits for this truncated representation
            layer_policy_output = self.policy_head(truncated_rep)
            
            # obtain logits_last (shape [V])
            if layer_policy_output.dim() == 3:
                logits_last = layer_policy_output[0, -1, :]
            elif layer_policy_output.dim() == 2:
                logits_last = layer_policy_output[0, :]
            else:
                raise RuntimeError("Unexpected layer_policy_output shape")
            
            # Top legal moves (fast): pick topk_vocab logits and filter legal ones for display  # noqa: E501
            topk = min(topk_vocab, logits_last.numel())
            vals, idxs = torch.topk(logits_last, k=topk)
            vals_np = vals.detach().cpu().numpy()
            idxs_np = idxs.detach().cpu().numpy()
            
            current_top_legal_moves = []
            # collect first up to 10 legal moves from topk
            for score_val, idx_val in zip(vals_np, idxs_np):
                uci = lboard.idx2uci(int(idx_val))
                if uci in legal_uci_set:
                    current_top_legal_moves.append((int(idx_val), uci, float(score_val)))
                if len(current_top_legal_moves) >= 10:
                    break
            
            # For target_move ranking we need to compute full ranking among legal moves.  # noqa: E501
            logits_last_cpu = logits_last.detach().cpu()
            legal_scores = []
            # Build uci->idx by scanning all indices once
            V = logits_last_cpu.numel()
            idx_to_uci = [lboard.idx2uci(i) for i in range(V)]
            # collect legal moves with their scores
            for idx_val, uci in enumerate(idx_to_uci):
                if uci in legal_uci_set:
                    score_val = float(logits_last_cpu[idx_val].item())
                    legal_scores.append((uci, score_val, idx_val))
            # sort legal_scores by score desc
            legal_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Compute target rank if requested
            target_info = None
            if target_move is not None:
                # ensure target_move is a legal move
                if target_move in legal_uci_set:
                    # find in legal_scores
                    rank = None
                    t_score = None
                    for i, (uci, score_val, idx_val) in enumerate(legal_scores):
                        if uci == target_move:
                            rank = i + 1  # 1-based
                            t_score = score_val
                            break
                    target_info = {'uci': target_move, 'rank': rank, 'score': t_score}
                else:
                    target_info = {'uci': target_move, 'rank': None, 'score': None}
            
            # Build move ranking info to return
            move_rankings = []
            for move_idx, uci, layer_score in current_top_legal_moves:
                # find final layer rank if present
                final_rank = None
                final_score = None
                for rank, (f_idx, f_uci, f_score) in enumerate(final_top_legal_moves):
                    if f_uci == uci:
                        final_rank = rank + 1
                        final_score = f_score
                        break
                move_rankings.append({
                    'move': uci,
                    'layer_score': float(layer_score),
                    'final_rank': final_rank,
                    'final_score': float(final_score) if final_score is not None else None,
                    'rank_change': (final_rank - 1) if final_rank is not None else None
                })
            
            layer_analysis[f'layer_{layer}'] = {
                'top_legal_moves': current_top_legal_moves,
                'move_rankings': move_rankings,
                'target': target_info,
            }
        
        return {
            'fen': fen,
            'final_layer_predictions': final_top_legal_moves,
            'layer_analysis': layer_analysis,
            'target_move': target_move
        }
    
    def create_layer_table(self, position_data: dict, fig, gs, start_row: int = 1, target_move: Optional[str] = None):
        """
        Create layer analysis table. If target_move is provided, include a column
        showing the target move's rank per layer.
        """
        layer_analysis = position_data.get('layer_analysis', {})
        returned_target_move = position_data.get('target_move', None)
        if target_move is None:
            target_move = returned_target_move

        num_layers = len(layer_analysis)
        
        # Create table data
        headers = ['Layer', 'Top 3 Moves', 'Scores', 'Final Ranks', 'Rank Changes']
        if target_move:
            headers.append(f"Target '{target_move}' Rank")
        
        table_data = []

        for layer_idx in range(num_layers):
            layer_name = f'layer_{layer_idx}'
            if layer_name not in layer_analysis:
                continue
            layer_data = layer_analysis[layer_name]
            
            # Top 3 moves
            top_moves = layer_data.get('top_legal_moves', [])[:3]
            move_ucis = []
            move_scores = []
            final_ranks = []
            rank_changes = []

            for top_move in top_moves:
                move_uci = top_move[1]
                move_score = top_move[2]
                move_ucis.append(move_uci)
                move_scores.append(f'{move_score:.3f}')

                # Get ranking information
                move_rankings = layer_data.get('move_rankings', [])
                final_rank = None
                rank_change = None
                for ranking in move_rankings:
                    if ranking['move'] == move_uci:
                        final_rank = ranking.get('final_rank', None)
                        rank_change = ranking.get('rank_change', None)  # noqa: E501
                        break
                final_ranks.append(str(final_rank) if final_rank else 'N/A')
                rank_changes.append(str(rank_change) if rank_change is not None else 'N/A')  # noqa: E501
            
            # Pad with empty strings if less than 3 moves
            while len(move_ucis) < 3:
                move_ucis.append('')
                move_scores.append('')
                final_ranks.append('')
                rank_changes.append('')

            # Target move rank column
            target_rank = ''
            if target_move:
                target_info = layer_data.get('target', {})
                if target_info and target_info.get('uci') == target_move:
                    rank = target_info.get('rank')
                    target_rank = str(rank) if rank is not None else 'N/A'

            row = [
                f'Layer {layer_idx}',
                ', '.join(move_ucis),
                ', '.join(move_scores),
                ', '.join(final_ranks),
                ', '.join(rank_changes)
            ]
            if target_move:
                row.append(target_rank)

            table_data.append(row)
        
        # Create table
        ax = fig.add_subplot(gs[start_row:, :])
        ax.axis('tight')
        ax.axis('off')
        
        col_widths = [0.15, 0.30, 0.18, 0.14, 0.12]
        if target_move:
            col_widths.append(0.12)

        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colWidths=col_widths
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Set header style
        for i in range(len(headers)):
            try:
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            except Exception:
                pass
        
        # Set data row style - alternating colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                try:
                    cell = table[(i, j)]
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                except KeyError:
                    pass
        
        ax.set_title('Layer Top Move Analysis', fontsize=14, fontweight='bold', pad=20)

    def visualize_analysis(self, position_data: dict, target_move: Optional[str] = None,
                          save_path: Optional[str] = None):
        fen = position_data['fen']
        
        # Create chess board visualization (for CLI mode, just print FEN)
        print(f"Chess position FEN: {fen}")
        
        # Print final predictions
        final_predictions = position_data.get('final_layer_predictions', [])
        print("Final Layer Top 3 Predictions:")
        for i, (idx, uci, score) in enumerate(final_predictions[:3]):
            print(f"  {i+1}. {uci} (score: {score:.3f})")
        print()
        
        # Create layer analysis table
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(1, 1)
        
        # Pass target_move through if provided
        self.create_layer_table(position_data, fig, gs, start_row=0, target_move=target_move)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()

    def analyze_and_visualize(self, fen: str, target_move: Optional[str] = None,
                            save_path: Optional[str] = None):
        analysis_result = self.analyze_single_fen(fen, target_move=target_move)
        if analysis_result is not None:
            self.visualize_analysis(analysis_result, target_move=target_move, 
                                  save_path=save_path)
            return analysis_result
        else:
            print("Analysis failed. Please check the FEN string.")
            return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run policy lens analysis on lc0 model"
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

    # 分析参数
    parser.add_argument(
        "--fen", type=str, required=True,
        help="FEN string of the chess position"
    )
    parser.add_argument(
        "--target_move", type=str, default=None,
        help="Target move in UCI format, e.g. a2c4"
    )
    parser.add_argument(
        "--topk_vocab", type=int, default=2000,
        help="Number of top logits to consider when searching for legal moves"
    )

    # 输出参数
    parser.add_argument(
        "--output_dir", type=str, default="./policy_lens_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save_plots", action="store_true",
        help="Save analysis plots"
    )
    parser.add_argument(
        "--save_data", action="store_true",
        help="Save analysis data as JSON"
    )
    parser.add_argument(
        "--plot_filename", type=str, default=None,
        help="Custom filename for saved plot"
    )

    # CUDA 可见设备（可选）
    parser.add_argument(
        "--cuda_visible_devices", type=str, default=None,
        help="Set CUDA_VISIBLE_DEVICES before running"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化分析器
    print("Initializing Policy Lens...")
    policy_lens = IntegratedPolicyLens(model_name=args.model_name)

    # 生成输出文件名
    fen_clean = args.fen.replace(" ", "_").replace("/", "_")
    target_suffix = f"_{args.target_move}" if args.target_move else ""  # noqa: E501
    
    plot_path = None
    if args.save_plots:
        if args.plot_filename:
            plot_path = output_dir / args.plot_filename
        else:
            plot_path = output_dir / f"policy_lens_{fen_clean}{target_suffix}.png"  # noqa: E501

    # 运行分析
    print("Running analysis...")
    result = policy_lens.analyze_and_visualize(
        fen=args.fen,
        target_move=args.target_move,
        save_path=str(plot_path) if plot_path else None
    )

    # 保存数据
    if args.save_data and result is not None:
        data_path = output_dir / f"policy_lens_data_{fen_clean}{target_suffix}.json"  # noqa: E501
        
        # 转换 numpy 数组为列表以便 JSON 序列化
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}  # noqa: E501
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj

        json_result = convert_for_json(result)
        
        with open(data_path, 'w') as f:
            json.dump(json_result, f, indent=2)
        
        print(f"Analysis data saved to: {data_path}")

    print("Analysis complete!")


if __name__ == "__main__":
    main()