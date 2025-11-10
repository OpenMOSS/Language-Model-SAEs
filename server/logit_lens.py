import torch
import torch.nn as nn
import copy
from typing import List, Tuple, Optional, Dict, Any
import chess
from lm_saes.circuit.leela_board import LeelaBoard


class IntegratedPolicyLens:
    """集成的Policy Lens分析器，用于分析模型每一层对移动预测的贡献"""
    
    def __init__(self, model):
        """
        初始化Policy Lens
        
        Args:
            model: HookedTransformer模型实例
        """
        self.model = model
        self.policy_head = self.model.policy_head
        self.num_layers = len(getattr(self.model, "blocks", [])) or 15
        print(f"Detected {self.num_layers} layers.")
        
        # cache for layernorm copies and alpha
        self.ln_copies: List[Tuple[Any, Any, nn.Parameter, nn.Parameter]] = []
        self.ln_orig: List[Tuple[Any, Any]] = []
        
        # create non-destructive copies of layernorms (with b zeroed) to use in logit-lens
        self.cache_layernorm()
        
        print("Policy Lens initialized successfully!")
        
    def cache_layernorm(self):
        """缓存LayerNorm的副本用于logit lens分析"""
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
        给定h = resid_post_after_ln at layer_idx (即该层ln2之后),
        应用后续层的alpha和其LN副本（b=0）同时将子层输出置零。
        返回适合送入policy_head的转换表示。
        
        Args:
            h: 残差激活
            layer_idx: 当前层索引
            
        Returns:
            转换后的表示
        """
        x = h
        for l in range(layer_idx + 1, self.num_layers):
            ln1_copy, ln2_copy, alpha_in, alpha_out = self.ln_copies[l]
            # multiply by scalar alphas (ensure dtype/device broadcasting)
            x = x * alpha_in  # attn branch zero + alpha_in * x
            x = ln1_copy(x)   # apply copy with b=0
            x = x * alpha_out  # mlp branch zero + alpha_out * x
            x = ln2_copy(x)
        return x   
    
    def analyze_single_fen(
        self, 
        fen: str, 
        target_move: Optional[str] = None, 
        topk_vocab: int = 2000
    ) -> Dict[str, Any]:
        """
        分析单个FEN位置。如果提供了`target_move`（UCI字符串），
        则计算其在每一层的排名/分数。
        
        Args:
            fen: FEN字符串
            target_move: 目标移动（UCI格式，可选）
            topk_vocab: 考虑多少个顶部logits来搜索合法移动（用于效率）
            
        Returns:
            分析结果字典
        """
        print(f"Analyzing FEN: {fen}, target_move: {target_move}")
        
        try:
            board = chess.Board(fen)
            if board.is_game_over():
                print("Warning: Game is over for this position")
        except ValueError as e:
            print(f"Invalid FEN: {e}")
            return {"error": f"Invalid FEN: {str(e)}"}
        
        with torch.no_grad():
            output, cache = self.model.run_with_cache(fen, prepend_bos=False)
        
        # Final layer predictions
        final_layer_input = cache[f'blocks.{self.num_layers-1}.resid_post_after_ln']
        final_policy_output = self.policy_head(final_layer_input)
        
        lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
        chess_board = chess.Board(fen)
        legal_uci_set = set(move.uci() for move in chess_board.legal_moves)
        
        # Compute final layer top legal moves
        if final_policy_output.dim() == 3:
            final_logits_last = final_policy_output[0, -1, :]
        elif final_policy_output.dim() == 2:
            final_logits_last = final_policy_output[0, :]
        else:
            raise RuntimeError("Unexpected final_policy_output shape")
        
        # get sorted final logits
        final_vals, final_idxs = torch.sort(final_logits_last, descending=True)
        final_vals = final_vals.detach().cpu().numpy()
        final_idxs = final_idxs.detach().cpu().numpy()
        final_top_legal_moves = []
        
        # collect top 10 legal moves based on final logits
        for idx_val, idx in zip(final_vals, final_idxs):
            uci = lboard.idx2uci(int(idx))
            if uci in legal_uci_set:
                final_top_legal_moves.append({
                    'idx': int(idx),
                    'uci': uci,
                    'score': float(idx_val)
                })
            if len(final_top_legal_moves) >= 10:
                break
        
        # Analyze each layer using Post-LN logit-lens
        layer_analysis = {}
        # 选择最终层的ground truth: final_top_legal_moves的第一个
        final_top_move_uci = final_top_legal_moves[0]['uci'] if len(final_top_legal_moves) > 0 else None
        
        for layer in range(self.num_layers):
            print(f"Analyzing layer {layer}...")
            
            layer_input = cache[f'blocks.{layer}.resid_post_after_ln']
            truncated_rep = self.apply_postln_truncation(layer_input, layer)
            
            # compute logits for this truncated representation
            layer_policy_output = self.policy_head(truncated_rep)
            
            # obtain logits_last (shape [V])
            if layer_policy_output.dim() == 3:
                logits_last = layer_policy_output[0, -1, :]
            elif layer_policy_output.dim() == 2:
                logits_last = layer_policy_output[0, :]
            else:
                raise RuntimeError("Unexpected layer_policy_output shape")
            
            # Top legal moves (fast): pick topk_vocab logits and filter legal ones
            topk = min(topk_vocab, logits_last.numel())
            vals, idxs = torch.topk(logits_last, k=topk)
            vals_np = vals.detach().cpu().numpy()
            idxs_np = idxs.detach().cpu().numpy()
            
            current_top_legal_moves = []
            # collect first up to 10 legal moves from topk
            for score_val, idx_val in zip(vals_np, idxs_np):
                uci = lboard.idx2uci(int(idx_val))
                if uci in legal_uci_set:
                    current_top_legal_moves.append({
                        'idx': int(idx_val),
                        'uci': uci,
                        'score': float(score_val)
                    })
                if len(current_top_legal_moves) >= 10:
                    break
            
            # For target_move ranking, build full legal scores
            logits_last_cpu = logits_last.detach().cpu()
            legal_scores = []
            V = logits_last_cpu.numel()
            
            # Build uci->idx mapping
            idx_to_uci = [lboard.idx2uci(i) for i in range(V)]
            
            # collect legal moves with their scores
            for idx_val, uci in enumerate(idx_to_uci):
                if uci in legal_uci_set:
                    score_val = float(logits_last_cpu[idx_val].item())
                    legal_scores.append((uci, score_val, idx_val))
            
            # sort legal_scores by score desc
            legal_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 计算softmax概率（在所有合法移动上）
            if len(legal_scores) > 0:
                import math
                max_logit = max(s for _, s, _ in legal_scores)
                exp_vals = [math.exp(s - max_logit) for _, s, _ in legal_scores]
                sum_exp = sum(exp_vals)
                probs = [ev / sum_exp for ev in exp_vals]
            else:
                probs = []

            # 计算当前层在合法移动集合上的logit熵（概率越集中熵越小）
            def _entropy(p_list):
                import math
                eps = 1e-12
                if not p_list:
                    return None
                return float(-sum(p * math.log(max(p, eps)) for p in p_list))
            logit_entropy = _entropy(probs)

            # Compute target rank if requested
            target_info = None
            if target_move is not None:
                if target_move in legal_uci_set:
                    rank = None
                    t_score = None
                    t_prob = None
                    for i, (uci, score_val, idx_val) in enumerate(legal_scores):
                        if uci == target_move:
                            rank = i + 1  # 1-based
                            t_score = score_val
                            t_prob = probs[i] if i < len(probs) else None
                            break
                    target_info = {
                        'uci': target_move,
                        'rank': rank,
                        'score': t_score,
                        'prob': t_prob
                    }
                else:
                    target_info = {
                        'uci': target_move,
                        'rank': None,
                        'score': None,
                        'error': 'Target move is not legal'
                    }

            # 计算“最终层Top移动”在本层的rank/score/prob
            final_top_info = None
            if final_top_move_uci is not None:
                f_rank = None
                f_score = None
                f_prob = None
                for i, (uci, score_val, idx_val) in enumerate(legal_scores):
                    if uci == final_top_move_uci:
                        f_rank = i + 1
                        f_score = score_val
                        f_prob = probs[i] if i < len(probs) else None
                        break
                final_top_info = {
                    'uci': final_top_move_uci,
                    'rank': f_rank,
                    'score': f_score,
                    'prob': f_prob,
                }
            
            # Build move rankings
            move_rankings = []
            for move_data in current_top_legal_moves:
                uci = move_data['uci']
                layer_score = move_data['score']
                
                # find final layer rank
                final_rank = None
                final_score = None
                for rank, f_move in enumerate(final_top_legal_moves):
                    if f_move['uci'] == uci:
                        final_rank = rank + 1
                        final_score = f_move['score']
                        break
                
                move_rankings.append({
                    'move': uci,
                    'layer_score': float(layer_score),
                    'final_rank': final_rank,
                    'final_score': float(final_score) if final_score is not None else None,
                    'rank_change': (final_rank - (len([m for m in current_top_legal_moves if m['score'] > layer_score]) + 1)) if final_rank is not None else None
                })
            
            layer_analysis[f'layer_{layer}'] = {
                'top_legal_moves': current_top_legal_moves,
                'move_rankings': move_rankings,
                'target': target_info,
                'final_top_move': final_top_info,
                'logit_entropy': logit_entropy,
            }
        
        return {
            'fen': fen,
            'final_layer_predictions': final_top_legal_moves,
            'layer_analysis': layer_analysis,
            'target_move': target_move,
            'num_layers': self.num_layers,
            'final_top_move_uci': final_top_move_uci
        }
