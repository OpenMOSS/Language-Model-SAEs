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


# 全局分析器实例（延迟初始化）
_patching_analyzer = None
_current_model_type = None  # 跟踪当前模型类型

def get_patching_analyzer(metadata: Optional[Dict[str, Any]] = None) -> PatchingAnalyzer:
    """获取或创建patching分析器实例，支持根据metadata动态选择模型路径"""
    global _patching_analyzer, _current_model_type
    
    # 根据metadata确定模型路径
    if metadata:
        lorsa_analysis_name = metadata.get('lorsa_analysis_name', '')
        tc_analysis_name = metadata.get('tc_analysis_name', '')
        
        # 检查是否提供了analysis_name且包含BT4
        if (lorsa_analysis_name and 'BT4' in lorsa_analysis_name) or (tc_analysis_name and 'BT4' in tc_analysis_name):
            print("🔍 检测到BT4模型，使用BT4路径...")
            tc_base_path = '/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/result_BT4/tc'
            lorsa_base_path = '/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/result_BT4/lorsa'
            model_type = 'BT4'
        else:
            print("🔍 使用默认T82模型路径...")
            tc_base_path = '/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/result/tc'
            lorsa_base_path = '/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/result/lorsa'
            model_type = 'T82'
    else:
        print("🔍 无metadata信息，使用默认T82模型路径...")
        tc_base_path = '/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/result/tc'
        lorsa_base_path = '/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/result/lorsa'
        model_type = 'T82'
    
    # 检查是否需要重新初始化分析器（模型类型变化时）
    if _patching_analyzer is None or _current_model_type != model_type:
        if _patching_analyzer is not None:
            print(f"🔄 模型类型从 {_current_model_type} 切换到 {model_type}，重新初始化分析器...")
            # 清理旧的模型资源
            del _patching_analyzer
            _patching_analyzer = None
        try:
            from transformer_lens import HookedTransformer
            from lm_saes import SparseAutoEncoder, LowRankSparseAttention
            
            print("🔍 正在初始化Patching分析器...")
            print(f"📁 TC路径: {tc_base_path}")
            print(f"📁 LORSA路径: {lorsa_base_path}")
            
            # 根据路径类型选择不同的模型
            if 'result_BT4' in tc_base_path:
                # BT4模型
                model_name = 'lc0/BT4-1024x15x32h'
                print("🔍 使用BT4模型: lc0/BT4-1024x15x32h")
            else:
                # 默认T82模型
                model_name = 'lc0/T82-768x15x24h'
                print("🔍 使用T82模型: lc0/T82-768x15x24h")
            
            # 加载模型
            model = HookedTransformer.from_pretrained_no_processing(
                model_name,
                dtype=torch.float32,
            ).eval()
            
            # 加载transcoders
            transcoders = {}
            for layer in range(15):
                # 根据路径类型选择不同的路径格式
                if 'result_BT4' in tc_base_path:
                    # BT4路径格式: _L{layer}M
                    tc_path = f"{tc_base_path}/L{layer}"
                else:
                    # 默认T82路径格式
                    tc_path = f"{tc_base_path}/lc0_L{layer}M_16x_k30_lr2e-03_auxk_sparseadam"
                
                print(f"📁 加载TC L{layer}: {tc_path}")
                transcoders[layer] = SparseAutoEncoder.from_pretrained(
                    tc_path,
                    dtype=torch.float32,
                    device='cuda',
                )
            
            # 加载lorsas
            lorsas = []
            for layer in range(15):
                # 根据路径类型选择不同的路径格式
                if 'result_BT4' in lorsa_base_path:
                    # BT4路径格式: BT4_lorsa_L{layer}A
                    lorsa_path = f"{lorsa_base_path}/lc0_L{layer}_bidirectional_lr0.0002_k_aux4096_coefficient0.125_dead_threshold1000000"
                else:
                    # 默认T82路径格式
                    lorsa_path = f"{lorsa_base_path}/lc0_L{layer}_bidirectional_lr8e-05_k_aux4096_coefficient0.0625_dead_threshold1000000"
                
                print(f"📁 加载LORSA L{layer}: {lorsa_path}")
                lorsas.append(LowRankSparseAttention.from_pretrained(
                    lorsa_path, 
                    device='cuda'
                ))
            
            _patching_analyzer = PatchingAnalyzer(model, transcoders, lorsas)
            _current_model_type = model_type  # 更新当前模型类型
            print("✅ Patching分析器初始化成功")
            
        except Exception as e:
            print(f"❌ Patching分析器初始化失败: {e}")
            raise
    
    return _patching_analyzer


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