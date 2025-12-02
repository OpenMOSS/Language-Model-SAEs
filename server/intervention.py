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
    
    def get_activations(self, fen: str) -> Tuple[List, List]:
        """获取给定FEN的激活值"""
        output, cache = self.model.run_with_cache(fen, prepend_bos=False)
        
        lorsa_activations, tc_activations = [], []
        
        for layer in range(15):
            # LoRSA激活
            lorsa_input = cache[f'blocks.{layer}.hook_attn_in']
            lorsa_dense_activation = self.lorsas[layer].encode(lorsa_input)
            lorsa_sparse_activation = lorsa_dense_activation.to_sparse_coo()
            lorsa_activations.append(lorsa_sparse_activation)
            
            # TC激活
            tc_input = cache[f'blocks.{layer}.resid_mid_after_ln']
            tc_dense_activation = self.transcoders[layer].encode(tc_input)
            tc_sparse_activation = tc_dense_activation.to_sparse_coo()
            tc_activations.append(tc_sparse_activation)
        
        return lorsa_activations, tc_activations
    
    def steering_analysis(self, feature_type: str, layer: int, 
                                   pos: int, feature: int, steering_scale: int, 
                                   fen: str) -> Optional[Dict[str, Any]]:
        """使用hook进行消融分析"""
        
        # 确保无残留hook
        try:
            self.model.reset_hooks()
        except Exception:
            pass
        
        # 获取激活值
        lorsa_activations, tc_activations = self.get_activations(fen)
        
        if feature_type == 'transcoder':
            activations = tc_activations[layer]
            WDs = self.tc_WDs[layer]
        elif feature_type == 'lorsa':
            activations = lorsa_activations[layer]
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
        
        # 收集所有合法移动的索引和logit值
        legal_indices = []
        legal_original_logits = []
        legal_modified_logits = []
        
        for idx in range(1858):
            try:
                uci = lboard.idx2uci(idx)
                if uci in legal_uci_set:
                    legal_indices.append(idx)
                    legal_original_logits.append(original_output[0, idx].item())
                    legal_modified_logits.append(modified_output[0, idx].item())
            except Exception:
                continue
        
        # 计算原始和修改后的概率（对合法移动的logit做softmax）
        if legal_indices and legal_original_logits and legal_modified_logits:
            # 确保是float tensor并放到同一device
            device = original_output.device if hasattr(original_output, 'device') else torch.device('cpu')
            
            # 直接从original_output和modified_output中提取合法移动的logit
            legal_indices_tensor = torch.tensor(legal_indices, device=device)
            orig_logits_legals = original_output[0, legal_indices_tensor].to(device).float()
            mod_logits_legals = modified_output[0, legal_indices_tensor].to(device).float()
            
            # Softmax归一化（对所有合法移动）
            original_probs = torch.softmax(orig_logits_legals, dim=0)
            modified_probs = torch.softmax(mod_logits_legals, dim=0)
            
            # 计算概率差异（修改后 - 原始）
            prob_diff = modified_probs - original_probs
        else:
            original_probs = torch.tensor([])
            modified_probs = torch.tensor([])
            prob_diff = torch.tensor([])
        
        # 获取所有合法移动的logit差异和概率差异
        legal_moves_with_diff = []
        for i, idx in enumerate(legal_indices):
            try:
                uci = lboard.idx2uci(idx)
                logit_diff_value = logit_diff[0, idx].item()
                original_logit = legal_original_logits[i]
                modified_logit = legal_modified_logits[i]
                
                # 获取概率差异
                prob_diff_value = prob_diff[i].item() if i < len(prob_diff) else 0.0
                original_prob = original_probs[i].item() if i < len(original_probs) else 0.0
                modified_prob = modified_probs[i].item() if i < len(modified_probs) else 0.0
                
                legal_moves_with_diff.append({
                    'uci': uci,
                    'diff': logit_diff_value,
                    'original_logit': original_logit,
                    'modified_logit': modified_logit,
                    'prob_diff': prob_diff_value,
                    'original_prob': original_prob,
                    'modified_prob': modified_prob,
                    'idx': idx
                })
            except Exception:
                continue
        
        # 生成按概率差异排序的列表（降序）
        sorted_by_prob = sorted(
            legal_moves_with_diff,
            key=lambda x: x['modified_prob'] - x['original_prob'],
            reverse=True
        )
        # 仍保留logit差异排序以备需要
        sorted_by_logit = sorted(legal_moves_with_diff, key=lambda x: x['diff'], reverse=True)
        
        # 基于概率差异的前后5个（正向促进=概率提升最多；抑制=概率下降最多）
        promoting_moves = sorted_by_prob[:5]
        inhibiting_moves = list(reversed(sorted_by_prob[-5:]))
        
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
                'feature_type': ablation_result['feature_type'],
                'layer': ablation_result['layer'],
                'pos': ablation_result['pos'],
                'feature': ablation_result['feature'],
                'activation_value': ablation_result['activation_value'],
                'hook_name': ablation_result['hook_name']
            },
            # 返回按概率差异排序的前10个结果，便于前端直接展示
            'top_moves_by_prob': sorted_by_prob[:10]
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
                combo_id = "k_128_e_128"
        except (ImportError, AttributeError):
            # 如果无法获取，使用默认组合
            combo_id = "k_128_e_128"
    
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