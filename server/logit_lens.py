import torch
import torch.nn as nn
import copy
from typing import List, Tuple, Optional, Dict, Any
import chess
from lm_saes.circuit.leela_board import LeelaBoard
from pathlib import Path
from safetensors.torch import load_file


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
                'legal_scores': legal_scores,  # 保存完整分数列表用于KL散度计算
                'probs': probs,  # 保存概率分布用于KL散度计算
            }
        
        # 计算层间KL散度和JSD（用于检测相变）
        def _kl_divergence(p_probs, q_probs, p_legal_scores, q_legal_scores):
            """
            计算KL散度 KL(P||Q)
            
            Args:
                p_probs: P分布的概率列表（与p_legal_scores对应）
                q_probs: Q分布的概率列表（与q_legal_scores对应）
                p_legal_scores: P分布的(uci, score, idx)列表
                q_legal_scores: Q分布的(uci, score, idx)列表
            
            Returns:
                KL散度值
            """
            import math
            eps = 1e-12
            
            # 构建uci到概率的映射
            p_dict = {uci: prob for (uci, _, _), prob in zip(p_legal_scores, p_probs)}
            q_dict = {uci: prob for (uci, _, _), prob in zip(q_legal_scores, q_probs)}
            
            # 获取所有合法移动的并集
            all_moves = set(p_dict.keys()) | set(q_dict.keys())
            
            if not all_moves:
                return None
            
            kl_sum = 0.0
            for uci in all_moves:
                p_val = p_dict.get(uci, eps)  # 如果移动不在P中，使用很小的值
                q_val = q_dict.get(uci, eps)  # 如果移动不在Q中，使用很小的值
                
                # 避免log(0)和除以0
                if p_val > eps and q_val > eps:
                    kl_sum += p_val * math.log(p_val / q_val)
            
            return float(kl_sum)
        
        def _js_divergence(p_probs, q_probs, p_legal_scores, q_legal_scores):
            """
            计算Jensen-Shannon散度 JSD(P||Q)
            
            JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
            其中 M = 0.5 * (P + Q)
            
            JSD是一个真正的度量（满足对称性、非负性、三角不等式），范围在[0, 1]之间（如果使用log2）或[0, ln(2)]之间（如果使用自然对数）
            
            Args:
                p_probs: P分布的概率列表（与p_legal_scores对应）
                q_probs: Q分布的概率列表（与q_legal_scores对应）
                p_legal_scores: P分布的(uci, score, idx)列表
                q_legal_scores: Q分布的(uci, score, idx)列表
            
            Returns:
                JSD值
            """
            import math
            eps = 1e-12
            
            # 构建uci到概率的映射
            p_dict = {uci: prob for (uci, _, _), prob in zip(p_legal_scores, p_probs)}
            q_dict = {uci: prob for (uci, _, _), prob in zip(q_legal_scores, q_probs)}
            
            # 获取所有合法移动的并集
            all_moves = set(p_dict.keys()) | set(q_dict.keys())
            
            if not all_moves:
                return None
            
            # 计算平均分布 M = 0.5 * (P + Q)
            m_dict = {}
            for uci in all_moves:
                p_val = p_dict.get(uci, eps)
                q_val = q_dict.get(uci, eps)
                m_dict[uci] = 0.5 * (p_val + q_val)
            
            # 计算 JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
            kl_pm = 0.0
            kl_qm = 0.0
            
            for uci in all_moves:
                p_val = p_dict.get(uci, eps)
                q_val = q_dict.get(uci, eps)
                m_val = m_dict[uci]
                
                # 计算 KL(P||M)
                if p_val > eps and m_val > eps:
                    kl_pm += p_val * math.log(p_val / m_val)
                
                # 计算 KL(Q||M)
                if q_val > eps and m_val > eps:
                    kl_qm += q_val * math.log(q_val / m_val)
            
            jsd = 0.5 * kl_pm + 0.5 * kl_qm
            return float(jsd)
        
        # 计算相邻层之间的KL散度
        layer_kl_divergences = []
        for layer in range(1, self.num_layers):
            prev_layer_data = layer_analysis.get(f'layer_{layer-1}')
            curr_layer_data = layer_analysis.get(f'layer_{layer}')
            
            if prev_layer_data and curr_layer_data:
                prev_probs = prev_layer_data.get('probs', [])
                curr_probs = curr_layer_data.get('probs', [])
                prev_legal_scores = prev_layer_data.get('legal_scores', [])
                curr_legal_scores = curr_layer_data.get('legal_scores', [])
                
                if prev_probs and curr_probs:
                    kl_forward = _kl_divergence(
                        curr_probs, prev_probs,
                        curr_legal_scores, prev_legal_scores
                    )  # KL(当前层||前一层的分布)
                    kl_backward = _kl_divergence(
                        prev_probs, curr_probs,
                        prev_legal_scores, curr_legal_scores
                    )  # KL(前一层的分布||当前层)
                    
                    # 计算真正的Jensen-Shannon散度（JSD）
                    # JSD是一个真正的度量，满足对称性、非负性和三角不等式
                    jsd = _js_divergence(
                        prev_probs, curr_probs,
                        prev_legal_scores, curr_legal_scores
                    )
                    
                    layer_kl_divergences.append({
                        'from_layer': layer - 1,
                        'to_layer': layer,
                        'kl_forward': kl_forward,  # KL(当前||前一层)
                        'kl_backward': kl_backward,  # KL(前一层||当前)
                        'jsd': jsd,  # Jensen-Shannon散度（真正的度量）
                    })
                else:
                    layer_kl_divergences.append({
                        'from_layer': layer - 1,
                        'to_layer': layer,
                        'kl_forward': None,
                        'kl_backward': None,
                        'jsd': None,
                    })
            else:
                layer_kl_divergences.append({
                    'from_layer': layer - 1,
                    'to_layer': layer,
                    'kl_forward': None,
                    'kl_backward': None,
                    'jsd': None,
                })
        
        # 计算每层相对于最终层的KL散度
        final_layer_data = layer_analysis.get(f'layer_{self.num_layers-1}')
        layer_to_final_kl = []
        if final_layer_data:
            final_probs = final_layer_data.get('probs', [])
            final_legal_scores = final_layer_data.get('legal_scores', [])
            
            for layer in range(self.num_layers):
                layer_data = layer_analysis.get(f'layer_{layer}')
                if layer_data:
                    layer_probs = layer_data.get('probs', [])
                    layer_legal_scores = layer_data.get('legal_scores', [])
                    
                    if layer_probs and final_probs:
                        kl_to_final = _kl_divergence(
                            final_probs, layer_probs,
                            final_legal_scores, layer_legal_scores
                        )  # KL(最终层||当前层)
                        kl_from_final = _kl_divergence(
                            layer_probs, final_probs,
                            layer_legal_scores, final_legal_scores
                        )  # KL(当前层||最终层)
                        jsd = _js_divergence(
                            layer_probs, final_probs,
                            layer_legal_scores, final_legal_scores
                        )  # JSD(当前层||最终层)
                        
                        layer_to_final_kl.append({
                            'layer': layer,
                            'kl_to_final': kl_to_final,
                            'kl_from_final': kl_from_final,
                            'jsd': jsd,  # Jensen-Shannon散度
                        })
                    else:
                        layer_to_final_kl.append({
                            'layer': layer,
                            'kl_to_final': None,
                            'kl_from_final': None,
                            'jsd': None,
                        })
                else:
                    layer_to_final_kl.append({
                        'layer': layer,
                        'kl_to_final': None,
                        'kl_from_final': None,
                        'jsd': None,
                    })
        
        return {
            'fen': fen,
            'final_layer_predictions': final_top_legal_moves,
            'layer_analysis': layer_analysis,
            'target_move': target_move,
            'num_layers': self.num_layers,
            'final_top_move_uci': final_top_move_uci,
            'layer_kl_divergences': layer_kl_divergences,  # 相邻层之间的KL散度
            'layer_to_final_kl': layer_to_final_kl,  # 每层相对于最终层的KL散度
        }
    
    def load_mean_activation(self, hook_point: str, base_dir: str = None) -> Tuple[torch.Tensor, Optional[int]]:
        """
        加载指定hook点的平均激活值
        
        Args:
            hook_point: hook点名称，如 "blocks.0.hook_attn_out"
            base_dir: 平均值文件的基础目录（可选）
            
        Returns:
            (mean_activation, n_tokens): 平均激活值和token数量
        """
        model_name = self.model.cfg.model_name
        
        if base_dir is None:
            if 'BT4' not in model_name:
                raise ValueError(f"仅支持BT4模型，收到: {model_name}")
            model_prefix = 'BT4'
            
            # 根据hook点类型确定子目录
            if 'attn_out' in hook_point:
                subdir = 'attn_out_mean'
            elif 'mlp_out' in hook_point:
                subdir = 'mlp_out_mean'
            else:
                raise ValueError(f"未知的hook点类型: {hook_point}")
            
            base_dir = f"/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/activations/{model_prefix}/{subdir}"

        safe_name = hook_point.replace(".", "_")
        file_path = Path(base_dir) / f"{safe_name}_mean.safetensors"
        
        if not file_path.exists():
            raise FileNotFoundError(f"平均值文件不存在: {file_path}")

        data = load_file(str(file_path))
        mean_activation = data["mean"]
        n_tokens = data["n_tokens"].item() if "n_tokens" in data else None
        
        return mean_activation, n_tokens
    
    def analyze_mean_ablation(
        self, 
        fen: str, 
        hook_types: List[str] = None,
        target_move: Optional[str] = None,
        topk_vocab: int = 2000
    ) -> Dict[str, Any]:
        """
        对每一层进行Mean Ablation分析
        
        Args:
            fen: FEN字符串
            hook_types: 要分析的hook类型列表，如 ['attn_out', 'mlp_out']，默认两者都分析
            target_move: 目标移动（UCI格式，可选）
            topk_vocab: 考虑多少个顶部logits来搜索合法移动
            
        Returns:
            分析结果字典
        """
        if hook_types is None:
            hook_types = ['attn_out', 'mlp_out']
        
        print(f"Analyzing Mean Ablation for FEN: {fen}, target_move: {target_move}")
        
        try:
            board = chess.Board(fen)
            if board.is_game_over():
                print("Warning: Game is over for this position")
        except ValueError as e:
            print(f"Invalid FEN: {e}")
            return {"error": f"Invalid FEN: {str(e)}"}
        
        # 获取原始输出
        with torch.no_grad():
            original_output, original_cache = self.model.run_with_cache(fen, prepend_bos=False)
        
        # 获取原始的top移动
        lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
        chess_board = chess.Board(fen)
        legal_uci_set = set(move.uci() for move in chess_board.legal_moves)
        
        # 计算原始输出的top移动
        # original_output是一个list，output[0]是logits tensor，shape为[batch, vocab_size]
        original_logits = original_output[0]  # shape: [1, 1858]
        if original_logits.dim() == 2:
            original_logits_last = original_logits[0, :]  # shape: [1858]
        elif original_logits.dim() == 1:
            original_logits_last = original_logits
        else:
            raise RuntimeError(f"Unexpected original logits shape: {original_logits.shape}")
        
        original_vals, original_idxs = torch.sort(original_logits_last, descending=True)
        original_vals = original_vals.detach().cpu().numpy()
        original_idxs = original_idxs.detach().cpu().numpy()
        original_top_legal_moves = []
        
        for idx_val, idx in zip(original_vals, original_idxs):
            uci = lboard.idx2uci(int(idx))
            if uci in legal_uci_set:
                original_top_legal_moves.append({
                    'idx': int(idx),
                    'uci': uci,
                    'score': float(idx_val)
                })
            if len(original_top_legal_moves) >= 10:
                break
        
        original_top_move_uci = original_top_legal_moves[0]['uci'] if len(original_top_legal_moves) > 0 else None
        
        # 对每一层每个hook类型进行ablation
        ablation_results = {}
        
        for hook_type in hook_types:
            for layer in range(self.num_layers):
                hook_point = f"blocks.{layer}.hook_{hook_type}"
                print(f"Analyzing {hook_point}...")
                
                try:
                    # 加载平均激活值
                    mean_activation, n_tokens = self.load_mean_activation(hook_point)
                    mean_activation = mean_activation.to(self.model.cfg.device)
                    
                    # 获取原始激活的shape来扩展mean
                    if hook_point in original_cache:
                        original_activation = original_cache[hook_point]
                        batch_size, seq_len = original_activation.shape[:2]
                        mean_expanded = mean_activation.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                        mean_expanded = mean_expanded.to(original_activation.device)
                    else:
                        print(f"Warning: {hook_point} not in cache, skipping")
                        continue
                    
                    # 定义hook函数
                    def mean_ablation_hook(activation, hook):
                        return mean_expanded.to(activation.device)
                    
                    # 运行ablation
                    with self.model.hooks(fwd_hooks=[(hook_point, mean_ablation_hook)]):
                        with torch.no_grad():
                            ablated_output, _ = self.model.run_with_cache(fen, prepend_bos=False)
                    
                    # 获取ablated输出的logits
                    # ablated_output是一个list，output[0]是logits tensor
                    ablated_logits = ablated_output[0]  # shape: [1, 1858]
                    if ablated_logits.dim() == 2:
                        ablated_logits_last = ablated_logits[0, :]  # shape: [1858]
                    elif ablated_logits.dim() == 1:
                        ablated_logits_last = ablated_logits
                    else:
                        raise RuntimeError(f"Unexpected ablated logits shape: {ablated_logits.shape}")
                    
                    # 计算logit差异
                    logit_diff = original_logits_last - ablated_logits_last
                    
                    # 获取ablated后的top移动
                    topk = min(topk_vocab, ablated_logits_last.numel())
                    vals, idxs = torch.topk(ablated_logits_last, k=topk)
                    vals_np = vals.detach().cpu().numpy()
                    idxs_np = idxs.detach().cpu().numpy()
                    
                    ablated_top_legal_moves = []
                    for score_val, idx_val in zip(vals_np, idxs_np):
                        uci = lboard.idx2uci(int(idx_val))
                        if uci in legal_uci_set:
                            ablated_top_legal_moves.append({
                                'idx': int(idx_val),
                                'uci': uci,
                                'score': float(score_val)
                            })
                        if len(ablated_top_legal_moves) >= 10:
                            break
                    
                    # 构建完整的合法移动分数列表
                    ablated_logits_cpu = ablated_logits_last.detach().cpu()
                    V = ablated_logits_cpu.numel()
                    idx_to_uci = [lboard.idx2uci(i) for i in range(V)]
                    
                    legal_scores = []
                    for idx_val, uci in enumerate(idx_to_uci):
                        if uci in legal_uci_set:
                            score_val = float(ablated_logits_cpu[idx_val].item())
                            legal_scores.append((uci, score_val, idx_val))
                    
                    legal_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # 计算softmax概率
                    if len(legal_scores) > 0:
                        import math
                        max_logit = max(s for _, s, _ in legal_scores)
                        exp_vals = [math.exp(s - max_logit) for _, s, _ in legal_scores]
                        sum_exp = sum(exp_vals)
                        probs = [ev / sum_exp for ev in exp_vals]
                    else:
                        probs = []
                    
                    # 计算熵
                    def _entropy(p_list):
                        import math
                        eps = 1e-12
                        if not p_list:
                            return None
                        return float(-sum(p * math.log(max(p, eps)) for p in p_list))
                    
                    logit_entropy = _entropy(probs)
                    
                    # 目标移动信息
                    target_info = None
                    if target_move is not None and target_move in legal_uci_set:
                        rank = None
                        t_score = None
                        t_prob = None
                        for i, (uci, score_val, idx_val) in enumerate(legal_scores):
                            if uci == target_move:
                                rank = i + 1
                                t_score = score_val
                                t_prob = probs[i] if i < len(probs) else None
                                break
                        target_info = {
                            'uci': target_move,
                            'rank': rank,
                            'score': t_score,
                            'prob': t_prob
                        }
                    
                    # 原始top移动在ablated后的排名
                    original_top_info = None
                    if original_top_move_uci is not None:
                        o_rank = None
                        o_score = None
                        o_prob = None
                        for i, (uci, score_val, idx_val) in enumerate(legal_scores):
                            if uci == original_top_move_uci:
                                o_rank = i + 1
                                o_score = score_val
                                o_prob = probs[i] if i < len(probs) else None
                                break
                        original_top_info = {
                            'uci': original_top_move_uci,
                            'rank': o_rank,
                            'score': o_score,
                            'prob': o_prob
                        }
                    
                    # 统计信息
                    logit_diff_stats = {
                        'mean': float(logit_diff.mean().item()),
                        'std': float(logit_diff.std().item()),
                        'max': float(logit_diff.max().item()),
                        'min': float(logit_diff.min().item()),
                        'l2_norm': float(torch.norm(logit_diff).item())
                    }
                    
                    ablation_results[hook_point] = {
                        'hook_point': hook_point,
                        'layer': layer,
                        'hook_type': hook_type,
                        'top_legal_moves': ablated_top_legal_moves,
                        'logit_diff_stats': logit_diff_stats,
                        'target': target_info,
                        'original_top_move': original_top_info,
                        'logit_entropy': logit_entropy,
                        'n_tokens_in_mean': n_tokens
                    }
                    
                except Exception as e:
                    print(f"Error analyzing {hook_point}: {e}")
                    import traceback
                    traceback.print_exc()
                    ablation_results[hook_point] = {
                        'hook_point': hook_point,
                        'layer': layer,
                        'hook_type': hook_type,
                        'error': str(e)
                    }
        
        return {
            'fen': fen,
            'original_top_legal_moves': original_top_legal_moves,
            'original_top_move_uci': original_top_move_uci,
            'ablation_results': ablation_results,
            'target_move': target_move,
            'num_layers': self.num_layers,
            'hook_types': hook_types
        }
