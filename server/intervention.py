import torch
from typing import Dict, Any, Optional, List, Tuple
from lm_saes import SparseAutoEncoder, LowRankSparseAttention
from transformer_lens import HookedTransformer
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from lm_saes.circuit.leela_board import LeelaBoard
    import chess
except ImportError:
    print("WARNING: leela_interp not found, chess functionality will be limited")
    LeelaBoard = None
    chess = None

try:
    # 统一“合法走法概率”的口径：复用 src/chess/move.py 的实现
    from src.chess import get_move_from_policy_output_with_prob
except Exception:
    get_move_from_policy_output_with_prob = None

# 全局 BT4 配置常量（兼容脚本运行和 package 导入）
try:
    from .constants import BT4_MODEL_NAME, BT4_TC_BASE_PATH, BT4_LORSA_BASE_PATH, get_bt4_sae_combo
except ImportError:
    from constants import BT4_MODEL_NAME, BT4_TC_BASE_PATH, BT4_LORSA_BASE_PATH, get_bt4_sae_combo


class PatchingAnalyzer:
    """消融分析器，用于分析特征对模型输出的影响"""
    
    def __init__(self, model: HookedTransformer, 
                 transcoders: Dict[int, SparseAutoEncoder], 
                 lorsas: List[LowRankSparseAttention]):
        self.model = model
        self.transcoders = transcoders
        self.lorsas = lorsas
        
        # 预计算WD权重
        self.tc_WDs = {}
        self.lorsa_WDs = {}
        
        for layer in range(15):
            self.tc_WDs[layer] = transcoders[layer].W_D
            self.lorsa_WDs[layer] = lorsas[layer].W_O
    
    def _get_cache(self, fen: str) -> dict:
        """运行模型并返回 cache。"""
        _, cache = self.model.run_with_cache(fen, prepend_bos=False)
        return cache

    def _get_lorsa_sparse_acts(self, cache: dict, layer: int) -> torch.Tensor:
        """获取指定层的 LoRSA sparse activations: [batch,pos,feature] 的 sparse_coo 形式。"""
        lorsa_hook = f"blocks.{layer}.hook_attn_in"
        if lorsa_hook not in cache:
            available_hooks = [k for k in cache.keys() if f"blocks.{layer}" in str(k)]
            raise KeyError(
                f"Missing hook '{lorsa_hook}' in cache. Available (sample): {available_hooks[:20]}"
            )
        lorsa_input = cache[lorsa_hook]
        lorsa_dense_activation = self.lorsas[layer].encode(lorsa_input)
        return lorsa_dense_activation.to_sparse_coo()

    def _get_tc_sparse_acts(self, cache: dict, layer: int) -> torch.Tensor:
        """获取指定层的 Transcoder sparse activations: [batch,pos,feature] 的 sparse_coo 形式。"""
        tc_hook = f"blocks.{layer}.resid_mid_after_ln"
        if tc_hook not in cache:
            available_hooks = [k for k in cache.keys() if f"blocks.{layer}" in str(k)]
            raise KeyError(
                f"Missing hook '{tc_hook}' in cache. Available (sample): {available_hooks[:20]}"
            )
        tc_input = cache[tc_hook]
        tc_dense_activation = self.transcoders[layer].encode(tc_input)
        return tc_dense_activation.to_sparse_coo()
    
    def steering_analysis(self, feature_type: str, layer: int, 
                                   pos: int, feature: int, steering_scale: int, 
                                   fen: str) -> Optional[Dict[str, Any]]:
        """使用hook进行消融分析"""
        
        # 确保无残留hook
        try:
            self.model.reset_hooks()
        except Exception:
            pass
        
        # 获取激活值：只计算当前 feature_type + 当前 layer，避免访问无关 hook
        cache = self._get_cache(fen)
        if feature_type == 'transcoder':
            activations = self._get_tc_sparse_acts(cache, layer)
            WDs = self.tc_WDs[layer]
        elif feature_type == 'lorsa':
            activations = self._get_lorsa_sparse_acts(cache, layer)
            WDs = self.lorsa_WDs[layer]
        else:
            raise ValueError("feature_type必须是'transcoder'或'lorsa'")
        
        # 查找激活值
        target_indices = torch.tensor([0, pos, feature], 
                                    device=activations.indices().device)
        matches = (activations.indices() == 
                  target_indices.unsqueeze(1)).all(dim=0)
        
        if not matches.any():
            print('该位置没有激活值，无法进行消融分析')
            return None
        
        activation_value = activations.values()[matches].item()
        
        # 计算特征贡献
        feature_contribution = activation_value * WDs[feature]  # [768]
        
        # 确定要修改的hook位置
        if feature_type == 'transcoder':
            hook_name = f'blocks.{layer}.hook_mlp_out'
        else:  # lorsa
            hook_name = f'blocks.{layer}.hook_attn_out'
        
        # 再次确保无hook并获取原始输出（无修改）
        try:
            self.model.reset_hooks()
        except Exception:
            pass
        original_output, cache = self.model.run_with_cache(fen, prepend_bos=False)
        
        # 定义hook修改函数
        def modify_hook(tensor, hook):
            modified_activation = tensor.clone()
            modified_activation[0, pos] = modified_activation[0, pos] + (steering_scale - 1) * feature_contribution
            return modified_activation
        
        # 运行修改后的模型（仅本次生效的hook）
        self.model.add_hook(hook_name, modify_hook)
        modified_output, _ = self.model.run_with_cache(
            fen, prepend_bos=False)
        # 清理hook，避免影响后续请求
        try:
            self.model.reset_hooks()
        except Exception:
            pass
        
        # 计算logit差异
        logit_diff = original_output[0] - modified_output[0]
        
        return {
            'feature_type': feature_type,
            'layer': layer,
            'pos': pos,
            'feature': feature,
            'activation_value': activation_value,
            'feature_contribution': feature_contribution.detach().cpu().numpy().tolist(),
            'original_output': original_output[0].detach().cpu().numpy().tolist(),
            'modified_output': modified_output[0].detach().cpu().numpy().tolist(),
            'logit_diff': logit_diff.detach().cpu().numpy().tolist(),
            'hook_name': hook_name
        }

    def multi_steering_analysis(
        self,
        fen: str,
        feature_type: str,
        layer: int,
        nodes: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        # 确保无残留hook
        try:
            self.model.reset_hooks()
        except Exception:
            pass

        # 参数校验与规范化
        if not isinstance(nodes, list) or len(nodes) == 0:
            raise ValueError("nodes must be a non-empty list")
        normalized_nodes: List[Dict[str, Any]] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            pos = node.get("pos")
            feature = node.get("feature")
            steering_scale = node.get("steering_scale", 1)
            if not isinstance(pos, int) or not isinstance(feature, int):
                continue
            if not isinstance(steering_scale, (int, float)):
                steering_scale = 1
            normalized_nodes.append(
                {"pos": pos, "feature": feature, "steering_scale": float(steering_scale)}
            )
        if len(normalized_nodes) == 0:
            raise ValueError("nodes is empty after validation")

        # 获取激活值：只计算当前 feature_type + 当前 layer，避免访问无关 hook
        cache = self._get_cache(fen)
        if feature_type == "transcoder":
            activations = self._get_tc_sparse_acts(cache, layer)
            WDs = self.tc_WDs[layer]
            hook_name = f"blocks.{layer}.hook_mlp_out"
        elif feature_type == "lorsa":
            activations = self._get_lorsa_sparse_acts(cache, layer)
            WDs = self.lorsa_WDs[layer]
            hook_name = f"blocks.{layer}.hook_attn_out"
        else:
            raise ValueError("feature_type必须是'transcoder'或'lorsa'")

        # 将需要的 (pos, feature) 做成集合，用一次遍历 sparse idx/value 查找激活值
        targets = {(n["pos"], n["feature"]) for n in normalized_nodes}
        found_acts: Dict[Tuple[int, int], float] = {}
        try:
            idx = activations.indices()  # [3, nnz]
            val = activations.values()
            # idx[0] = batch index, idx[1] = pos, idx[2] = feature
            for j in range(idx.shape[1]):
                if int(idx[0, j].item()) != 0:
                    continue
                key = (int(idx[1, j].item()), int(idx[2, j].item()))
                if key in targets:
                    found_acts[key] = float(val[j].item())
        except Exception:
            # 如果 sparse 结构不符合预期，直接失败
            raise ValueError("failed to parse sparse activations")

        # 要求每个 node 都能在对应 pos 取到激活值，否则返回 None（与单 feature 行为一致）
        missing = [(p, f) for (p, f) in targets if (p, f) not in found_acts]
        if missing:
            print(f"该位置没有激活值，无法进行多 feature steering: missing={missing}")
            return None

        # 计算每个 pos 的总 delta（多个 feature 可能落在同一个 pos）
        pos_to_delta: Dict[int, torch.Tensor] = {}
        node_details: List[Dict[str, Any]] = []
        for n in normalized_nodes:
            pos = n["pos"]
            feature = n["feature"]
            scale = n["steering_scale"]
            activation_value = found_acts[(pos, feature)]
            feature_contribution = activation_value * WDs[feature]  # [d_model]
            delta = (scale - 1.0) * feature_contribution
            if pos not in pos_to_delta:
                pos_to_delta[pos] = delta
            else:
                pos_to_delta[pos] = pos_to_delta[pos] + delta
            node_details.append(
                {
                    "pos": pos,
                    "feature": feature,
                    "steering_scale": scale,
                    "activation_value": activation_value,
                }
            )

        # 原始输出（无修改）
        try:
            self.model.reset_hooks()
        except Exception:
            pass
        original_output, _ = self.model.run_with_cache(fen, prepend_bos=False)

        # Hook：在指定 pos 上加上 delta
        def modify_hook(tensor, hook):
            modified_activation = tensor.clone()
            for pos, delta in pos_to_delta.items():
                modified_activation[0, pos] = modified_activation[0, pos] + delta
            return modified_activation

        self.model.add_hook(hook_name, modify_hook)
        modified_output, _ = self.model.run_with_cache(fen, prepend_bos=False)
        try:
            self.model.reset_hooks()
        except Exception:
            pass

        logit_diff = original_output[0] - modified_output[0]
        return {
            "feature_type": feature_type,
            "layer": layer,
            "nodes": node_details,
            "original_output": original_output[0].detach().cpu().numpy().tolist(),
            "modified_output": modified_output[0].detach().cpu().numpy().tolist(),
            "logit_diff": logit_diff.detach().cpu().numpy().tolist(),
            "hook_name": hook_name,
        }
    
    def analyze_steering_results(self, ablation_result: Dict[str, Any], 
                               fen: str) -> Dict[str, Any]:
        """分析消融结果，返回对合法移动的影响"""
        if ablation_result is None:
            return None
        
        if LeelaBoard is None or chess is None:
            return {'error': 'Chess functionality not available'}
        
        logit_diff = torch.tensor(ablation_result['logit_diff'])
        original_output = torch.tensor(ablation_result['original_output'])
        modified_output = torch.tensor(ablation_result['modified_output'])
        
        # 获取合法移动
        lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
        chess_board = chess.Board(fen)
        legal_uci_set = set(move.uci() for move in chess_board.legal_moves)
        
        # 收集所有合法移动的 idx / uci / logit
        legal_moves: list[dict[str, Any]] = []
        for idx in range(1858):
            try:
                uci = lboard.idx2uci(idx)
            except Exception:
                continue
            if uci not in legal_uci_set:
                continue
            legal_moves.append(
                {
                    "idx": idx,
                    "uci": uci,
                    "original_logit": float(original_output[0, idx].item()),
                    "modified_logit": float(modified_output[0, idx].item()),
                }
            )

        # 统一概率口径：全部通过 src/chess/move.py 的 get_move_from_policy_output_with_prob 获取 all-legal softmax 概率
        original_prob_by_uci: dict[str, float] = {}
        modified_prob_by_uci: dict[str, float] = {}
        if get_move_from_policy_output_with_prob is not None:
            try:
                orig_logits_last = original_output[0, :].detach().cpu()
                mod_logits_last = modified_output[0, :].detach().cpu()
                orig_all = get_move_from_policy_output_with_prob(orig_logits_last.unsqueeze(0), fen, return_list=True)
                mod_all = get_move_from_policy_output_with_prob(mod_logits_last.unsqueeze(0), fen, return_list=True)
                if isinstance(orig_all, list):
                    original_prob_by_uci = {uci: float(prob) for (uci, _logit, prob) in orig_all}
                if isinstance(mod_all, list):
                    modified_prob_by_uci = {uci: float(prob) for (uci, _logit, prob) in mod_all}
            except Exception:
                original_prob_by_uci = {}
                modified_prob_by_uci = {}
        
        # 获取所有合法移动的logit差异和概率差异
        # 计算 top-k 展示口径：在 all-legal softmax 概率上取 top-k 再归一化（仅用于“Top Moves”展示）
        topk = 5
        def _topk_renorm(prob_by_uci: dict[str, float], k: int) -> dict[str, float]:
            if not prob_by_uci:
                return {}
            items = sorted(prob_by_uci.items(), key=lambda x: x[1], reverse=True)[: max(1, int(k))]
            s = sum(p for _, p in items)
            if s <= 0:
                return {uci: 0.0 for uci, _ in items}
            return {uci: float(p / s) for uci, p in items}

        original_prob_topk_by_uci = _topk_renorm(original_prob_by_uci, topk)
        modified_prob_topk_by_uci = _topk_renorm(modified_prob_by_uci, topk)

        legal_moves_with_diff: list[dict[str, Any]] = []
        for m in legal_moves:
            uci = m["uci"]
            idx = m["idx"]
            original_prob = float(original_prob_by_uci.get(uci, 0.0))
            modified_prob = float(modified_prob_by_uci.get(uci, 0.0))
            original_prob_topk = float(original_prob_topk_by_uci.get(uci, 0.0))
            modified_prob_topk = float(modified_prob_topk_by_uci.get(uci, 0.0))
            legal_moves_with_diff.append(
                {
                    "uci": uci,
                    "diff": float(logit_diff[0, idx].item()),
                    "original_logit": m["original_logit"],
                    "modified_logit": m["modified_logit"],
                    "prob_diff": float(modified_prob - original_prob),
                    "original_prob": original_prob,
                    "modified_prob": modified_prob,
                    "prob_diff_topk": float(modified_prob_topk - original_prob_topk),
                    "original_prob_topk": original_prob_topk,
                    "modified_prob_topk": modified_prob_topk,
                    "idx": idx,
                }
            )
        
        # 注意：这里区分三种排序口径，避免“Top Moves by Prob”语义混乱
        # 1) prob_diff 排序：找“概率提升/下降最多”的走法（更适合 promoting/inhibiting）
        sorted_by_prob_diff = sorted(
            legal_moves_with_diff,
            key=lambda x: x.get("prob_diff", 0.0),
            reverse=True,
        )
        # 2) prob 排序：找“修改后概率最高”的走法（更适合 top moves by prob）
        sorted_by_modified_prob = sorted(
            legal_moves_with_diff,
            key=lambda x: x.get("modified_prob", 0.0),
            reverse=True,
        )
        # 3) top-k prob 排序：匹配 logit-lens 的展示口径
        sorted_by_modified_prob_topk = sorted(
            legal_moves_with_diff,
            key=lambda x: x.get("modified_prob_topk", 0.0),
            reverse=True,
        )
        # 仍保留logit差异排序以备需要
        sorted_by_logit = sorted(legal_moves_with_diff, key=lambda x: x['diff'], reverse=True)
        
        # 基于概率差异的前后5个（正向促进=概率提升最多；抑制=概率下降最多）
        promoting_moves = sorted_by_prob_diff[:5]
        inhibiting_moves = list(reversed(sorted_by_prob_diff[-5:]))
        
        # 统计信息
        total_legal_moves = len(legal_moves_with_diff)
        if total_legal_moves > 0:
            avg_logit_diff = (sum(x['diff'] for x in legal_moves_with_diff) / 
                            total_legal_moves)
            max_logit_diff = max(x['diff'] for x in legal_moves_with_diff)
            min_logit_diff = min(x['diff'] for x in legal_moves_with_diff)
            
            # 概率差异统计（修改后 - 原始）
            avg_prob_diff = (sum((x['modified_prob'] - x['original_prob']) for x in legal_moves_with_diff) / 
                           total_legal_moves)
            max_prob_diff = max((x['modified_prob'] - x['original_prob']) for x in legal_moves_with_diff)
            min_prob_diff = min((x['modified_prob'] - x['original_prob']) for x in legal_moves_with_diff)
        else:
            avg_logit_diff = max_logit_diff = min_logit_diff = 0
            avg_prob_diff = max_prob_diff = min_prob_diff = 0
        
        return {
            # 特征缺失促进的移动（logit下降）
            'promoting_moves': promoting_moves,
            # 特征缺失抑制的移动（logit上升）
            'inhibiting_moves': inhibiting_moves,
            'statistics': {
                'total_legal_moves': total_legal_moves,
                'avg_logit_diff': avg_logit_diff,
                'max_logit_diff': max_logit_diff,
                'min_logit_diff': min_logit_diff,
                'avg_prob_diff': avg_prob_diff,
                'max_prob_diff': max_prob_diff,
                'min_prob_diff': min_prob_diff
            },
            'ablation_info': {
                'feature_type': ablation_result.get('feature_type'),
                'layer': ablation_result.get('layer'),
                # 单 feature 时这些字段存在；多 feature 时用 nodes 替代
                'pos': ablation_result.get('pos'),
                'feature': ablation_result.get('feature'),
                'activation_value': ablation_result.get('activation_value'),
                'nodes': ablation_result.get('nodes'),
                'hook_name': ablation_result.get('hook_name')
            },
            # 返回两套“Top moves”：
            # - top_moves_by_prob: 按修改后概率（all-legal softmax）排序
            # - top_moves_by_prob_topk: 按修改后概率（top-k legal softmax，匹配 logit-lens）排序
            'top_moves_by_prob': sorted_by_modified_prob[:10],
            'top_moves_by_prob_topk': sorted_by_modified_prob_topk[:10],
            # 保留：按 prob_diff 排序的前10个（用于诊断/对齐）
            'top_moves_by_prob_diff': sorted_by_prob_diff[:10],
        }


# 全局分析器实例（延迟初始化，仅支持BT4）
# 使用字典存储不同组合的分析器，key为combo_id
_patching_analyzers: Dict[str, PatchingAnalyzer] = {}
_current_combo_id: Optional[str] = None

def clear_patching_analyzer(combo_id: Optional[str] = None):
    """清理指定组合的patching分析器，如果combo_id为None则清理所有"""
    global _patching_analyzers, _current_combo_id
    if combo_id is None:
        _patching_analyzers.clear()
        _current_combo_id = None
        print("🧹 已清理所有patching分析器")
    elif combo_id in _patching_analyzers:
        del _patching_analyzers[combo_id]
        if _current_combo_id == combo_id:
            _current_combo_id = None
        print(f"🧹 已清理组合 {combo_id} 的patching分析器")

def get_patching_analyzer(metadata: Optional[Dict[str, Any]] = None, combo_id: Optional[str] = None) -> PatchingAnalyzer:
    """
    获取或创建仅支持BT4的patching分析器实例。
    
    Args:
        metadata: 保留参数以保证兼容性，已弃用
        combo_id: SAE组合ID（例如 "k_128_e_128"），如果不提供则从app.py获取当前组合
    
    Returns:
        PatchingAnalyzer: 分析器实例
    """
    global _patching_analyzers, _current_combo_id
    
    # 优先从 metadata 中获取组合ID（前端会传 sae_combo_id）
    if combo_id is None and isinstance(metadata, dict):
        meta_combo_id = metadata.get("sae_combo_id")
        if isinstance(meta_combo_id, str) and meta_combo_id.strip():
            combo_id = meta_combo_id.strip()

    # 获取当前组合ID
    if combo_id is None:
        try:
            # 尝试从app.py获取当前组合
            import sys
            if 'app' in sys.modules:
                from app import CURRENT_BT4_SAE_COMBO_ID
                combo_id = CURRENT_BT4_SAE_COMBO_ID
            else:
                # 如果app模块未加载，使用默认组合
                combo_id = "k_30_e_16"
        except (ImportError, AttributeError):
            # 如果无法获取，使用默认组合
            combo_id = "k_30_e_16"
    
    # 如果已经有该组合的分析器，直接返回
    if combo_id in _patching_analyzers:
        return _patching_analyzers[combo_id]
    
    try:
        from transformer_lens import HookedTransformer
        from lm_saes import SparseAutoEncoder, LowRankSparseAttention
        
        # 获取当前组合的配置
        combo_cfg = get_bt4_sae_combo(combo_id)
        tc_base_path = combo_cfg["tc_base_path"]
        lorsa_base_path = combo_cfg["lorsa_base_path"]
        
        print(f"🔍 正在初始化BT4 Patching分析器（组合: {combo_id}）...")
        print(f"📁 TC路径: {tc_base_path}")
        print(f"📁 LORSA路径: {lorsa_base_path}")
        print(f"🔍 使用模型: {BT4_MODEL_NAME}")
        
        # 构建cache_key（与preload_circuit_models保持一致）
        cache_key = f"{BT4_MODEL_NAME}::{combo_id}"
        
        # 尝试从circuits_service获取缓存的模型（使用cache_key）
        try:
            from circuits_service import get_cached_models
            cached_hooked_model, cached_transcoders, cached_lorsas, _ = get_cached_models(cache_key)
            
            if cached_hooked_model is not None and cached_transcoders is not None and cached_lorsas is not None:
                if len(cached_transcoders) == 15 and len(cached_lorsas) == 15:
                    print(f"✅ 使用缓存的模型、transcoders和lorsas（组合: {combo_id}）")
                    model = cached_hooked_model
                    transcoders = cached_transcoders
                    lorsas = cached_lorsas
                else:
                    raise ValueError(f"缓存不完整: transcoders={len(cached_transcoders)}, lorsas={len(cached_lorsas)}")
            else:
                raise ValueError(f"缓存不存在: cache_key={cache_key}")
        except (ImportError, ValueError) as e:
            print(f"⚠️ 无法使用缓存，需要等待预加载完成: {e}")
            print(f"💡 提示: 请先调用 /circuit/preload_models 预加载组合 {combo_id} 的模型")
            raise RuntimeError(
                f"组合 {combo_id} 的模型尚未预加载。请先调用 /circuit/preload_models 接口预加载模型，"
                f"或等待预加载完成后再使用patching分析功能。"
            )
        
        # 创建分析器并缓存
        analyzer = PatchingAnalyzer(model, transcoders, lorsas)
        _patching_analyzers[combo_id] = analyzer
        _current_combo_id = combo_id
        print(f"✅ BT4 Patching分析器初始化成功（组合: {combo_id}）")
        return analyzer
    except Exception as e:
        print(f"❌ Patching分析器初始化失败: {e}")
        raise


def run_feature_steering_analysis(fen: str, feature_type: str, layer: int, 
                         pos: int, feature: int, steering_scale: int, 
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """运行patching分析的公共接口"""
    analyzer = get_patching_analyzer(metadata)
    
    # 运行消融分析
    # ablation_result = analyzer.hook_based_ablation_analysis(
    ablation_result = analyzer.steering_analysis(
        feature_type=feature_type,
        layer=layer,
        pos=pos,
        feature=feature,
        steering_scale=steering_scale,
        fen=fen
    )
    
    if ablation_result is None:
        return {'error': '该位置没有激活值，无法进行消融分析'}
    
    # 分析结果
    analysis_result = analyzer.analyze_steering_results(ablation_result, fen)
    
    return analysis_result


def run_multi_feature_steering_analysis(
    fen: str,
    feature_type: str,
    layer: int,
    nodes: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    运行多 feature steering 分析（每个 feature 对应一个 position）并返回结果。

    Args:
        fen: FEN 字符串
        feature_type: 'transcoder' 或 'lorsa'
        layer: 层号
        nodes: node 列表，每个元素至少包含 pos/feature/steering_scale
        metadata: 保留参数以保证兼容性（目前不使用）

    Returns:
        与 run_feature_steering_analysis 同结构的分析结果。
    """
    analyzer = get_patching_analyzer(metadata)
    ablation_result = analyzer.multi_steering_analysis(
        fen=fen,
        feature_type=feature_type,
        layer=layer,
        nodes=nodes,
    )
    if ablation_result is None:
        return {"error": "至少有一个 node 在对应 pos 上没有激活值，无法进行 steering"}
    return analyzer.analyze_steering_results(ablation_result, fen)