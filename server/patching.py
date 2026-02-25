import torch
from typing import Dict, Any, Optional, List, Tuple
from lm_saes import SparseAutoEncoder, LowRankSparseAttention
from transformer_lens import HookedTransformer
import sys
from pathlib import Path

# add project root to Python path
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
    """ablation analyzer, for analyzing the impact of features on model output"""
    
    def __init__(self, model: HookedTransformer, 
                 transcoders: Dict[int, SparseAutoEncoder], 
                 lorsas: List[LowRankSparseAttention]):
        self.model = model
        self.transcoders = transcoders
        self.lorsas = lorsas
        
        # pre-calculate WD weights
        self.tc_WDs = {}
        self.lorsa_WDs = {}
        
        for layer in range(15):
            self.tc_WDs[layer] = transcoders[layer].W_D
            self.lorsa_WDs[layer] = lorsas[layer].W_O
    
    def get_activations(self, fen: str) -> Tuple[List, List]:
        """get the activations of the given FEN"""
        output, cache = self.model.run_with_cache(fen, prepend_bos=False)
        
        lorsa_activations, tc_activations = [], []
        
        for layer in range(15):
            # Lorsa activations
            lorsa_input = cache[f'blocks.{layer}.hook_attn_in']
            lorsa_dense_activation = self.lorsas[layer].encode(lorsa_input)
            lorsa_sparse_activation = lorsa_dense_activation.to_sparse_coo()
            lorsa_activations.append(lorsa_sparse_activation)
            
            # TC activations
            tc_input = cache[f'blocks.{layer}.resid_mid_after_ln']
            tc_dense_activation = self.transcoders[layer].encode(tc_input)
            tc_sparse_activation = tc_dense_activation.to_sparse_coo()
            tc_activations.append(tc_sparse_activation)
        
        return lorsa_activations, tc_activations
    
    def hook_based_ablation_analysis(self, feature_type: str, layer: int, 
                                   pos: int, feature: int, 
                                   fen: str) -> Optional[Dict[str, Any]]:
        """use hook to perform ablation analysis"""
        
        # get activations
        lorsa_activations, tc_activations = self.get_activations(fen)
        
        if feature_type == 'transcoder':
            activations = tc_activations[layer]
            WDs = self.tc_WDs[layer]
        elif feature_type == 'lorsa':
            activations = lorsa_activations[layer]
            WDs = self.lorsa_WDs[layer]
        else:
            raise ValueError("feature_type must be 'transcoder' or 'lorsa'")
        
        # find activations
        target_indices = torch.tensor([0, pos, feature], 
                                    device=activations.indices().device)
        matches = (activations.indices() == 
                  target_indices.unsqueeze(1)).all(dim=0)
        
        if not matches.any():
            print('no activation value at this position, cannot perform ablation analysis')
            return None
        
        activation_value = activations.values()[matches].item()
        
        # calculate feature contribution
        feature_contribution = activation_value * WDs[feature]  # [768]
        
        # determine the hook position to modify
        if feature_type == 'transcoder':
            hook_name = f'blocks.{layer}.hook_mlp_out'
        else:  # lorsa
            hook_name = f'blocks.{layer}.hook_attn_out'
        
        # get the original output (without modification)
        original_output, cache = self.model.run_with_cache(fen, prepend_bos=False)
        
        # define the hook modification function
        def modify_hook(tensor, hook):
            modified_activation = tensor.clone()
            modified_activation[0, pos] -= feature_contribution
            return modified_activation
        
        # run the modified model (using hook modification)
        self.model.add_hook(hook_name, modify_hook)
        modified_output, modified_cache = self.model.run_with_cache(
            fen, prepend_bos=False)
        
        # calculate the logit difference
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
        """analyze the ablation results, return the impact on legal moves"""
        if ablation_result is None:
            return None
        
        if LeelaBoard is None or chess is None:
            return {'error': 'Chess functionality not available'}
        
        logit_diff = torch.tensor(ablation_result['logit_diff'])
        original_output = torch.tensor(ablation_result['original_output'])
        modified_output = torch.tensor(ablation_result['modified_output'])
        
        # get legal moves
        lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
        chess_board = chess.Board(fen)
        legal_uci_set = set(move.uci() for move in chess_board.legal_moves)
        
        # get the logit difference of all legal moves
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
        
        # sort by logit difference
        legal_moves_with_diff.sort(key=lambda x: x['diff'], reverse=True)
        
        # find the 5 moves with the most logit decrease (promoting these moves)
        promoting_moves = legal_moves_with_diff[:5]
        
        # find the 5 moves with the most logit increase (inhibiting these moves)
        inhibiting_moves = legal_moves_with_diff[-5:][::-1]
        
        # get statistics
        total_legal_moves = len(legal_moves_with_diff)
        if total_legal_moves > 0:
            avg_logit_diff = (sum(x['diff'] for x in legal_moves_with_diff) / 
                            total_legal_moves)
            max_logit_diff = max(x['diff'] for x in legal_moves_with_diff)
            min_logit_diff = min(x['diff'] for x in legal_moves_with_diff)
        else:
            avg_logit_diff = max_logit_diff = min_logit_diff = 0
        
        return {
            # promoting moves (logit decrease)
            'promoting_moves': promoting_moves,
            # inhibiting moves (logit increase)
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


# global analyzer instance (lazy initialization)
_patching_analyzer = None

def get_patching_analyzer() -> PatchingAnalyzer:
    """get or create patching analyzer instance (using global cache)"""
    global _patching_analyzer
    
    if _patching_analyzer is None:
        try:
            from transformer_lens import HookedTransformer
            from lm_saes import SparseAutoEncoder, LowRankSparseAttention
            
            print("initializing Patching analyzer...")
            
            model_name = 'lc0/BT4-1024x15x32h'
            
            # try to get the cached model from circuits_service
            try:
                from circuits_service import get_cached_models
                cached_hooked_model, cached_transcoders, cached_lorsas, _ = get_cached_models(model_name)
                
                if cached_hooked_model is not None and cached_transcoders is not None and cached_lorsas is not None:
                    if len(cached_transcoders) == 15 and len(cached_lorsas) == 15:
                        print("using cached model, transcoders and lorsas")
                        model = cached_hooked_model
                        transcoders = cached_transcoders
                        lorsas = cached_lorsas
                    else:
                        raise ValueError("cache incomplete")
                else:
                    raise ValueError("cache not found")
            except (ImportError, ValueError) as e:
                print(f"cannot use cache, need to reload: {e}")
                
                # load model - force using BT4
                model = HookedTransformer.from_pretrained_no_processing(
                    model_name,
                    dtype=torch.float32,
                ).eval()
                
                # load transcoders
                transcoders = {}
                for layer in range(15):
                    transcoders[layer] = SparseAutoEncoder.from_pretrained(
                        (f'/inspire/hdd/global_user/hezhengfu-240208120186/'
                         f'rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/tc/'
                         f'L{layer}'),
                        dtype=torch.float32,
                        device='cuda',
                    )
                
                # load lorsas
                lorsas = []
                for layer in range(15):
                    lorsas.append(LowRankSparseAttention.from_pretrained(
                        (f'/inspire/hdd/global_user/hezhengfu-240208120186/'
                         f'rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/lorsa/'
                         f'L{layer}'), 
                        device='cuda'
                    ))
            
            _patching_analyzer = PatchingAnalyzer(model, transcoders, lorsas)
            print("Patching analyzer initialized successfully")
            
        except Exception as e:
            print(f"Patching analyzer initialization failed: {e}")
            raise
    
    return _patching_analyzer


def run_patching_analysis(fen: str, feature_type: str, layer: int, 
                         pos: int, feature: int) -> Dict[str, Any]:
    """run the public interface for patching analysis"""
    analyzer = get_patching_analyzer()
    
    # run the ablation analysis
    ablation_result = analyzer.hook_based_ablation_analysis(
        feature_type=feature_type,
        layer=layer,
        pos=pos,
        feature=feature,
        fen=fen
    )
    
    if ablation_result is None:
        return {'error': 'no activation value at this position, cannot perform ablation analysis'}
    
    # analysis result
    analysis_result = analyzer.analyze_ablation_results(ablation_result, fen)
    
    return analysis_result