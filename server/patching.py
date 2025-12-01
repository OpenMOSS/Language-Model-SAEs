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
    
    def hook_based_ablation_analysis(self, feature_type: str, layer: int, 
                                   pos: int, feature: int, 
                                   fen: str) -> Optional[Dict[str, Any]]:
        """使用hook进行消融分析"""
        
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
        
        # 获取原始输出（无修改）
        original_output, cache = self.model.run_with_cache(fen, prepend_bos=False)
        
        # 定义hook修改函数
        def modify_hook(tensor, hook):
            modified_activation = tensor.clone()
            modified_activation[0, pos] -= feature_contribution
            return modified_activation
        
        # 运行修改后的模型（使用hook修改）
        self.model.add_hook(hook_name, modify_hook)
        modified_output, modified_cache = self.model.run_with_cache(
            fen, prepend_bos=False)
        
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
    
    def analyze_ablation_results(self, ablation_result: Dict[str, Any], 
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
        
        # 获取所有合法移动的logit差异
        legal_moves_with_diff = []
        for idx in range(1858):
            try:
                uci = lboard.idx2uci(idx)
                if uci in legal_uci_set:
                    diff_value = logit_diff[0, idx].item()
                    original_logit = original_output[0, idx].item()
                    modified_logit = modified_output[0, idx].item()
                    legal_moves_with_diff.append({
                        'uci': uci,
                        'diff': diff_value,
                        'original_logit': original_logit,
                        'modified_logit': modified_logit,
                        'idx': idx
                    })
            except Exception:
                continue
        
        # 按logit差异排序
        legal_moves_with_diff.sort(key=lambda x: x['diff'], reverse=True)
        
        # 找出logit降低最多的5个移动（促进这些移动）
        promoting_moves = legal_moves_with_diff[:5]
        
        # 找出logit提升最多的5个移动（抑制这些移动）
        inhibiting_moves = legal_moves_with_diff[-5:][::-1]
        
        # 统计信息
        total_legal_moves = len(legal_moves_with_diff)
        if total_legal_moves > 0:
            avg_logit_diff = (sum(x['diff'] for x in legal_moves_with_diff) / 
                            total_legal_moves)
            max_logit_diff = max(x['diff'] for x in legal_moves_with_diff)
            min_logit_diff = min(x['diff'] for x in legal_moves_with_diff)
        else:
            avg_logit_diff = max_logit_diff = min_logit_diff = 0
        
        return {
            # 特征缺失促进的移动（logit下降）
            'promoting_moves': promoting_moves,
            # 特征缺失抑制的移动（logit上升）
            'inhibiting_moves': inhibiting_moves,
            'statistics': {
                'total_legal_moves': total_legal_moves,
                'avg_logit_diff': avg_logit_diff,
                'max_logit_diff': max_logit_diff,
                'min_logit_diff': min_logit_diff
            },
            'ablation_info': {
                'feature_type': ablation_result['feature_type'],
                'layer': ablation_result['layer'],
                'pos': ablation_result['pos'],
                'feature': ablation_result['feature'],
                'activation_value': ablation_result['activation_value'],
                'hook_name': ablation_result['hook_name']
            }
        }


# 全局分析器实例（延迟初始化）
_patching_analyzer = None

def get_patching_analyzer() -> PatchingAnalyzer:
    """获取或创建patching分析器实例（使用全局缓存）"""
    global _patching_analyzer
    
    if _patching_analyzer is None:
        try:
            from transformer_lens import HookedTransformer
            from lm_saes import SparseAutoEncoder, LowRankSparseAttention
            
            print("🔍 正在初始化Patching分析器...")
            
            model_name = 'lc0/BT4-1024x15x32h'
            
            # 尝试从circuits_service获取缓存的模型
            try:
                from circuits_service import get_cached_models
                cached_hooked_model, cached_transcoders, cached_lorsas, _ = get_cached_models(model_name)
                
                if cached_hooked_model is not None and cached_transcoders is not None and cached_lorsas is not None:
                    if len(cached_transcoders) == 15 and len(cached_lorsas) == 15:
                        print("✅ 使用缓存的模型、transcoders和lorsas")
                        model = cached_hooked_model
                        transcoders = cached_transcoders
                        lorsas = cached_lorsas
                    else:
                        raise ValueError("缓存不完整")
                else:
                    raise ValueError("缓存不存在")
            except (ImportError, ValueError) as e:
                print(f"⚠️ 无法使用缓存，重新加载: {e}")
                
                # 加载模型 - 强制使用BT4
                model = HookedTransformer.from_pretrained_no_processing(
                    model_name,
                    dtype=torch.float32,
                ).eval()
                
                # 加载transcoders
                transcoders = {}
                for layer in range(15):
                    transcoders[layer] = SparseAutoEncoder.from_pretrained(
                        (f'/inspire/hdd/global_user/hezhengfu-240208120186/'
                         f'rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/tc/'
                         f'L{layer}'),
                        dtype=torch.float32,
                        device='cuda',
                    )
                
                # 加载lorsas
                lorsas = []
                for layer in range(15):
                    lorsas.append(LowRankSparseAttention.from_pretrained(
                        (f'/inspire/hdd/global_user/hezhengfu-240208120186/'
                         f'rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/lorsa/'
                         f'L{layer}'), 
                        device='cuda'
                    ))
            
            _patching_analyzer = PatchingAnalyzer(model, transcoders, lorsas)
            print("✅ Patching分析器初始化成功")
            
        except Exception as e:
            print(f"❌ Patching分析器初始化失败: {e}")
            raise
    
    return _patching_analyzer


def run_patching_analysis(fen: str, feature_type: str, layer: int, 
                         pos: int, feature: int) -> Dict[str, Any]:
    """运行patching分析的公共接口"""
    analyzer = get_patching_analyzer()
    
    # 运行消融分析
    ablation_result = analyzer.hook_based_ablation_analysis(
        feature_type=feature_type,
        layer=layer,
        pos=pos,
        feature=feature,
        fen=fen
    )
    
    if ablation_result is None:
        return {'error': '该位置没有激活值，无法进行消融分析'}
    
    # 分析结果
    analysis_result = analyzer.analyze_ablation_results(ablation_result, fen)
    
    return analysis_result