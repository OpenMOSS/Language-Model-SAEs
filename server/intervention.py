import torch
from typing import Dict, Any, Optional, List, Tuple
from lm_saes import SparseAutoEncoder, LowRankSparseAttention
from transformer_lens import HookedTransformer
import sys
from pathlib import Path

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
    from src.chess_utils import get_move_from_policy_output_with_prob
    from src.chess_utils.move import get_value_from_output
except Exception:
    get_move_from_policy_output_with_prob = None
    get_value_from_output = None
try:
    from .constants import BT4_MODEL_NAME, BT4_TC_BASE_PATH, BT4_LORSA_BASE_PATH, get_bt4_sae_combo
except ImportError:
    from constants import BT4_MODEL_NAME, BT4_TC_BASE_PATH, BT4_LORSA_BASE_PATH, get_bt4_sae_combo


class PatchingAnalyzer:
    def __init__(self, model: HookedTransformer, 
                 transcoders: Dict[int, SparseAutoEncoder], 
                 lorsas: List[LowRankSparseAttention]):
        self.model = model
        self.transcoders = transcoders
        self.lorsas = lorsas
        
        self.tc_WDs = {}
        self.lorsa_WDs = {}
        
        for layer in range(15):
            self.tc_WDs[layer] = transcoders[layer].W_D
            self.lorsa_WDs[layer] = lorsas[layer].W_O
    
    def _get_cache(self, fen: str):
        _, cache = self.model.run_with_cache(fen, prepend_bos=False)
        return cache

    def _get_lorsa_sparse_acts(self, cache: dict, layer: int) -> torch.Tensor:
        """get the sparse activations of the specified layer of Lorsa: [batch,pos,feature] in sparse_coo format"""
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
        """get the sparse activations of the specified layer of Transcoder: [batch,pos,feature] in sparse_coo format"""
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
        """use hook to perform ablation analysis"""
        
        # ensure no residual hooks
        try:
            self.model.reset_hooks()
        except Exception:
            pass
        
        # get activations: only calculate current feature_type + current layer, avoid accessing irrelevant hooks
        cache = self._get_cache(fen)
        if feature_type == 'transcoder':
            activations = self._get_tc_sparse_acts(cache, layer)
            WDs = self.tc_WDs[layer]
        elif feature_type == 'lorsa':
            activations = self._get_lorsa_sparse_acts(cache, layer)
            WDs = self.lorsa_WDs[layer]
        else:
            raise ValueError("feature_type must be 'transcoder' or 'lorsa'")
        
        # find activations
        target_indices = torch.tensor([0, pos, feature], 
                                    device=activations.indices().device)
        matches = (activations.indices() == 
                  target_indices.unsqueeze(1)).all(dim=0)
        
        if not matches.any():
            print('No activations at this position, cannot perform ablation analysis')
            return None
        
        activation_value = activations.values()[matches].item()
        
        # calculate feature contribution
        feature_contribution = activation_value * WDs[feature]  # [768]
        
        # determine the hook position to modify
        if feature_type == 'transcoder':
            hook_name = f'blocks.{layer}.hook_mlp_out'
        else:  # lorsa
            hook_name = f'blocks.{layer}.hook_attn_out'
        
        # again ensure no hooks and get original output (no modification)
        try:
            self.model.reset_hooks()
        except Exception:
            pass
        print(f"call model.run_with_cache, fen: {fen}")
        print(f"fen type: {type(fen)}, length: {len(fen) if isinstance(fen, str) else 'N/A'}")

        original_output, cache = self.model.run_with_cache(fen, prepend_bos=False)

        print(f"model.run_with_cache returned:")
        print(f"original_output type: {type(original_output)}")
        print(f"original_output length: {len(original_output) if hasattr(original_output, '__len__') else 'N/A'}")
        if hasattr(original_output, '__getitem__'):
            for i in range(min(3, len(original_output))):
                item = original_output[i]
                print(f"original_output[{i}] type: {type(item)}")
                if hasattr(item, 'shape'):
                    print(f"original_output[{i}] shape: {item.shape}")
                elif hasattr(item, '__len__'):
                    print(f"original_output[{i}] length: {len(item)}")
                    if len(item) > 0 and isinstance(item, (list, tuple)):
                        print(f"original_output[{i}][0] type: {type(item[0])}")
                        if hasattr(item[0], 'shape'):
                            print(f"original_output[{i}][0] shape: {item[0].shape}")

        # check the specific shape of policy logits
        policy_logits = original_output[0]
        print(f"policy_logits shape: {policy_logits.shape}")
        print(f"policy_logits[:5]: {policy_logits[:5].tolist() if hasattr(policy_logits, 'tolist') else policy_logits[:5]}")
        
        # define the hook modification function
        def modify_hook(tensor, hook):
            modified_activation = tensor.clone()
            modified_activation[0, pos] = modified_activation[0, pos] + (steering_scale - 1) * feature_contribution
            return modified_activation
        
        # run the modified model (only the hooks that take effect this time)
        self.model.add_hook(hook_name, modify_hook)
        modified_output, _ = self.model.run_with_cache(
            fen, prepend_bos=False)
        # clean up hooks, avoid affecting subsequent requests
        try:
            self.model.reset_hooks()
        except Exception:
            pass
        
        # calculate the logit difference
        logit_diff = modified_output[0] - original_output[0]

        # calculate the value difference (Win - Loss)
        original_value = float(original_output[1][0][0] - original_output[1][0][2]) if get_value_from_output else 0.0
        modified_value = float(modified_output[1][0][0] - modified_output[1][0][2]) if get_value_from_output else 0.0
        value_diff = modified_value - original_value

        print(f"steering_analysis returned data:")
        print(f"original_output[0] shape: {original_output[0].shape}")
        print(f"modified_output[0] shape: {modified_output[0].shape}")
        print(f"logit_diff shape: {logit_diff.shape}")

        # ensure the policy logits are the correct shape
        policy_original = original_output[0]
        policy_modified = modified_output[0]

        # if [1, 1858], get [0] to get [1858]
        if policy_original.ndim == 2:
            policy_original = policy_original[0]
        if policy_modified.ndim == 2:
            policy_modified = policy_modified[0]

        print(f"processed policy_original shape: {policy_original.shape}")
        print(f"processed policy_modified shape: {policy_modified.shape}")

        result = {
            'feature_type': feature_type,
            'layer': layer,
            'pos': pos,
            'feature': feature,
            'activation_value': activation_value,
            'feature_contribution': feature_contribution.detach().cpu().numpy().tolist(),
            'original_output': policy_original.detach().cpu().numpy().tolist(),
            'modified_output': policy_modified.detach().cpu().numpy().tolist(),
            'logit_diff': logit_diff.detach().cpu().numpy().tolist(),
            'original_value': float(original_value),
            'modified_value': float(modified_value),
            'value_diff': float(value_diff),
            'hook_name': hook_name
        }

        print(f"returned original_output length: {len(result['original_output'])}")
        print(f"returned modified_output length: {len(result['modified_output'])}")
        print(f"returned logit_diff length: {len(result['logit_diff'])}")

        return result

    def multi_steering_analysis(
        self,
        fen: str,
        feature_type: str,
        layer: int,
        nodes: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        # ensure no residual hooks
        try:
            self.model.reset_hooks()
        except Exception:
            pass

        # for multi steering, temporarily do not calculate value (to avoid index errors)
        get_value_from_output = None

        # parameter validation and normalization
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

        # get activations: only calculate current feature_type + current layer, avoid accessing irrelevant hooks
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
            raise ValueError("feature_type must be 'transcoder' or 'lorsa'")

        # make the needed (pos, feature) into a set, use one traversal to find activations
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
            # if the sparse structure is not as expected, fail directly
            raise ValueError("failed to parse sparse activations")

        # require each node to get activations at the corresponding pos, otherwise return None (consistent with single feature behavior)
        missing = [(p, f) for (p, f) in targets if (p, f) not in found_acts]
        if missing:
            print(f"No activations at this position, cannot perform multi feature steering: missing={missing}")
            return None

        # calculate the total delta for each pos (multiple features may fall on the same pos)
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

        # original output (no modification)
        try:
            self.model.reset_hooks()
        except Exception:
            pass
        original_output, _ = self.model.run_with_cache(fen, prepend_bos=False)

        # Hook: add delta to the specified pos
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

        # calculate the logit difference, handle different output formats
        try:
            # handle original_output -可能是 tensor or list
            if isinstance(original_output, torch.Tensor):
                orig_logits = original_output
                if orig_logits.ndim == 2 and orig_logits.shape[0] == 1:
                    orig_logits = orig_logits[0]  # from [1, 1858] to [1858]
            elif isinstance(original_output, (list, tuple)) and len(original_output) > 0:
                if isinstance(original_output[0], torch.Tensor):
                    orig_logits = original_output[0]
                    if orig_logits.ndim == 2 and orig_logits.shape[0] == 1:
                        orig_logits = orig_logits[0]
                else:
                    # handle nested list cases
                    orig_logits = torch.tensor(original_output[0])
                    if orig_logits.ndim == 2 and orig_logits.shape[0] == 1:
                        orig_logits = orig_logits[0]
            else:
                raise ValueError(f"Unexpected original_output format: {type(original_output)}")

            # handle modified_output -可能是 tensor or list
            if isinstance(modified_output, torch.Tensor):
                mod_logits = modified_output
                if mod_logits.ndim == 2 and mod_logits.shape[0] == 1:
                    mod_logits = mod_logits[0]  # from [1, 1858] to [1858]
            elif isinstance(modified_output, (list, tuple)) and len(modified_output) > 0:
                if isinstance(modified_output[0], torch.Tensor):
                    mod_logits = modified_output[0]
                    if mod_logits.ndim == 2 and mod_logits.shape[0] == 1:
                        mod_logits = mod_logits[0]
                else:
                    # handle nested list cases
                    mod_logits = torch.tensor(modified_output[0])
                    if mod_logits.ndim == 2 and mod_logits.shape[0] == 1:
                        mod_logits = mod_logits[0]
            else:
                raise ValueError(f"Unexpected modified_output format: {type(modified_output)}")

            print(f"Original logits shape: {orig_logits.shape}")
            print(f"Modified logits shape: {mod_logits.shape}")

            logit_diff = mod_logits - orig_logits
            print(f"Logit diff shape: {logit_diff.shape}")
        except (RuntimeError, IndexError, TypeError) as e:
            print(f"Error computing logit difference: {e}")
            print(f"Original output type: {type(original_output)}, length: {len(original_output) if hasattr(original_output, '__len__') else 'N/A'}")
            print(f"Modified output type: {type(modified_output)}, length: {len(modified_output) if hasattr(modified_output, '__len__') else 'N/A'}")
            if len(original_output) > 0:
                print(f"original_output[0] type: {type(original_output[0])}, shape: {getattr(original_output[0], 'shape', 'no shape')}")
            if len(modified_output) > 0:
                print(f"modified_output[0] type: {type(modified_output[0])}, shape: {getattr(modified_output[0], 'shape', 'no shape')}")
            raise ValueError(f"Failed to compute logit difference: {e}")

        # calculate the value difference (Win - Loss), add safety check
        original_value = 0.0
        modified_value = 0.0
        value_diff = 0.0

        def safe_to_numpy(tensor_or_list):
            if isinstance(tensor_or_list, torch.Tensor):
                # Ensure it is 1D with shape [1858]
                if tensor_or_list.ndim == 2 and tensor_or_list.shape[0] == 1:
                    tensor_or_list = tensor_or_list[0]
                return tensor_or_list.detach().cpu().numpy().tolist()
            else:
                # Handle nested list cases
                tensor = torch.tensor(tensor_or_list)
                if tensor.ndim == 2 and tensor.shape[0] == 1:
                    tensor = tensor[0]
                return tensor.detach().cpu().numpy().tolist()

        return {
            "feature_type": feature_type,
            "layer": layer,
            "nodes": node_details,
            "original_output": safe_to_numpy(orig_logits),
            "modified_output": safe_to_numpy(mod_logits),
            "logit_diff": logit_diff.detach().cpu().numpy().tolist(),
            "original_value": float(original_value),
            "modified_value": float(modified_value),
            "value_diff": float(value_diff),
            "hook_name": hook_name,
        }
    
    def analyze_steering_results(self, ablation_result: Dict[str, Any],
                               fen: str) -> Dict[str, Any]:
        """analyze the ablation results, return the impact on legal moves"""
        if ablation_result is None:
            return None

        if LeelaBoard is None or chess is None:
            return {'error': 'Chess functionality not available'}

        print(f"analyze_steering_results started")
        orig_out = ablation_result.get('original_output', [])

        logit_diff = torch.tensor(ablation_result['logit_diff'])
        original_output = torch.tensor(ablation_result['original_output'])
        modified_output = torch.tensor(ablation_result['modified_output'])

        print(f"after creating tensor - original_output shape: {original_output.shape}")

        # ensure the output is the correct shape [1858] instead of [1, 1858]
        if original_output.ndim == 2 and original_output.shape[0] == 1:
            original_output = original_output[0]
        if modified_output.ndim == 2 and modified_output.shape[0] == 1:
            modified_output = modified_output[0]

        # ensure the logit_diff is the same shape as original_output
        if logit_diff.ndim == 2 and logit_diff.shape[0] == 1:
            logit_diff = logit_diff[0]

        lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
        chess_board = chess.Board(fen)
        legal_uci_set = set(move.uci() for move in chess_board.legal_moves)
        
        # collect all legal moves idx / uci / logit
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
                    "original_logit": float(original_output[idx].item()),
                    "modified_logit": float(modified_output[idx].item()),
                }
            )

        # Unified probability convention: calculate softmax probability over all legal moves
        original_prob_by_uci: dict[str, float] = {}
        modified_prob_by_uci: dict[str, float] = {}

        # extract the probability of legal moves from policy logits
        def get_legal_move_probs(policy_logits: torch.Tensor, fen: str) -> dict[str, float]:
            """calculate the softmax probability of all legal moves from policy logits"""
            print(f"start calculating probability, policy_logits shape: {policy_logits.shape}, type: {type(policy_logits)}")

            # Ensure policy_logits has the correct shape
            if policy_logits.ndim == 2:
                policy_logits = policy_logits[0]  # Remove batch dimension
                print(f"after removing batch dimension - policy_logits shape: {policy_logits.shape}")

            # get legal moves
            chess_board = chess.Board(fen)
            legal_uci_set = set(move.uci() for move in chess_board.legal_moves)
            print(f"number of legal moves: {len(legal_uci_set)}")

            # extract the logits of legal moves
            legal_logits = []
            legal_ucis = []
            for idx in range(1858):
                try:
                    uci = lboard.idx2uci(idx)
                    if uci in legal_uci_set:
                        logit_value = float(policy_logits[idx].item())
                        legal_logits.append(logit_value)
                        legal_ucis.append(uci)
                except Exception:
                    continue

            print(f"number of legal moves extracted: {len(legal_logits)}")
            if not legal_logits:
                return {}

            # calculate softmax probability
            legal_logits_tensor = torch.tensor(legal_logits)
            print(f"logits range: {min(legal_logits):.3f} - {max(legal_logits):.3f}")

            probs = torch.softmax(legal_logits_tensor, dim=0)
            prob_by_uci = {uci: float(prob.item()) for uci, prob in zip(legal_ucis, probs)}

            print(f"probability range: {min(prob_by_uci.values()):.6f} - {max(prob_by_uci.values()):.6f}")
            return prob_by_uci

        try:
            original_prob_by_uci = get_legal_move_probs(original_output, fen)
            modified_prob_by_uci = get_legal_move_probs(modified_output, fen)
            print(f"probability calculation successful: original_prob_by_uci has {len(original_prob_by_uci)} moves, modified_prob_by_uci has {len(modified_prob_by_uci)} moves")
            if original_prob_by_uci:
                sample_uci = list(original_prob_by_uci.keys())[0]
                print(f"example probability - {sample_uci}: original={original_prob_by_uci[sample_uci]:.6f}, modified={modified_prob_by_uci.get(sample_uci, 0):.6f}")
        except Exception as e:
            print(f"probability calculation failed: {e}")
            import traceback
            print(f"error details: {traceback.format_exc()}")
            original_prob_by_uci = {}
            modified_prob_by_uci = {}
        
        # get the logit difference and probability difference of all legal moves
        # get the top k moves with the highest probability (directly use the original probability, no re-normalization)
        topk = 5
        def _get_topk_probs(prob_by_uci: dict[str, float], k: int) -> dict[str, float]:
            if not prob_by_uci:
                return {}
            items = sorted(prob_by_uci.items(), key=lambda x: x[1], reverse=True)[: max(1, int(k))]
            return {uci: prob for uci, prob in items}

        original_prob_topk_by_uci = _get_topk_probs(original_prob_by_uci, topk)
        modified_prob_topk_by_uci = _get_topk_probs(modified_prob_by_uci, topk)

        legal_moves_with_diff: list[dict[str, Any]] = []
        for m in legal_moves:
            uci = m["uci"]
            idx = m["idx"]
            original_prob = float(original_prob_by_uci.get(uci, 0.0))
            modified_prob = float(modified_prob_by_uci.get(uci, 0.0))
            original_prob_topk = float(original_prob_topk_by_uci.get(uci, 0.0))
            modified_prob_topk = float(modified_prob_topk_by_uci.get(uci, 0.0))
            # select the correct index way based on the shape of logit_diff
            if logit_diff.ndim == 2:
                diff_value = float(logit_diff[0, idx].item())
            else:
                diff_value = float(logit_diff[idx].item())

            legal_moves_with_diff.append(
                {
                    "uci": uci,
                    "diff": diff_value,
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
        
        # Note: distinguish three sorting criteria to avoid semantic confusion of "Top Moves by Prob"
        # 1) prob_diff sorting: find the moves with the highest probability increase/decrease (best for promoting/inhibiting)
        sorted_by_prob_diff = sorted(
            legal_moves_with_diff,
            key=lambda x: x.get("prob_diff", 0.0),
            reverse=True,
        )
        # 2) prob sorting: find the moves with the highest probability after modification (best for "top moves by prob")
        sorted_by_modified_prob = sorted(
            legal_moves_with_diff,
            key=lambda x: x.get("modified_prob", 0.0),
            reverse=True,
        )
        # 3) top-k prob sorting: matches the display convention of logit-lens
        sorted_by_modified_prob_topk = sorted(
            legal_moves_with_diff,
            key=lambda x: x.get("modified_prob_topk", 0.0),
            reverse=True,
        )
        # still keep the logit difference sorting for future needs
        sorted_by_logit = sorted(legal_moves_with_diff, key=lambda x: x['diff'], reverse=True)
        
        # based on the probability difference, the front and back 5 (promoting=probability increase most; inhibiting=probability decrease most)
        promoting_moves = sorted_by_prob_diff[:5]
        inhibiting_moves = list(reversed(sorted_by_prob_diff[-5:]))
        
        # statistics information
        total_legal_moves = len(legal_moves_with_diff)
        if total_legal_moves > 0:
            avg_logit_diff = (sum(x['diff'] for x in legal_moves_with_diff) / 
                            total_legal_moves)
            max_logit_diff = max(x['diff'] for x in legal_moves_with_diff)
            min_logit_diff = min(x['diff'] for x in legal_moves_with_diff)
            
            # probability difference statistics (modified - original)
            avg_prob_diff = (sum((x['modified_prob'] - x['original_prob']) for x in legal_moves_with_diff) / 
                           total_legal_moves)
            max_prob_diff = max((x['modified_prob'] - x['original_prob']) for x in legal_moves_with_diff)
            min_prob_diff = min((x['modified_prob'] - x['original_prob']) for x in legal_moves_with_diff)
        else:
            avg_logit_diff = max_logit_diff = min_logit_diff = 0
            avg_prob_diff = max_prob_diff = min_prob_diff = 0
        
        # calculate the value statistics information
        original_value = ablation_result.get('original_value', 0.0)
        modified_value = ablation_result.get('modified_value', 0.0)
        value_diff = ablation_result.get('value_diff', 0.0)

        return {
            # feature missing promoting moves (logit decrease)
            'promoting_moves': promoting_moves,
            # feature missing inhibiting moves (logit increase)
            'inhibiting_moves': inhibiting_moves,
            'statistics': {
                'total_legal_moves': total_legal_moves,
                'avg_logit_diff': avg_logit_diff,
                'max_logit_diff': max_logit_diff,
                'min_logit_diff': min_logit_diff,
                'avg_prob_diff': avg_prob_diff,
                'max_prob_diff': max_prob_diff,
                'min_prob_diff': min_prob_diff,
                'original_value': original_value,
                'modified_value': modified_value,
                'value_diff': value_diff
            },
            'ablation_info': {
                'feature_type': ablation_result.get('feature_type'),
                'layer': ablation_result.get('layer'),
                # when there is single feature, these fields exist; when there are multiple features, use nodes instead
                'pos': ablation_result.get('pos'),
                'feature': ablation_result.get('feature'),
                'activation_value': ablation_result.get('activation_value'),
                'nodes': ablation_result.get('nodes'),
                'hook_name': ablation_result.get('hook_name')
            },
            # return two sets of "Top moves":
            # - top_moves_by_prob: sorted by modified probability (all-legal softmax)
            # - top_moves_by_prob_topk: sorted by modified probability (top-k legal softmax, match logit-lens)
            'top_moves_by_prob': sorted_by_modified_prob[:10],
            'top_moves_by_prob_topk': sorted_by_modified_prob_topk[:10],
            # keep: the top 10 sorted by prob_diff (for diagnosis/alignment)
            'top_moves_by_prob_diff': sorted_by_prob_diff[:10],
        }


# global analyzer instance (lazy initialization, only supports BT4)
# use a dictionary to store different analyzer instances, key is combo_id
_patching_analyzers: Dict[str, PatchingAnalyzer] = {}
_current_combo_id: Optional[str] = None

def clear_patching_analyzer(combo_id: Optional[str] = None):
    """clean up the specified patching analyzer, if combo_id is None, clean up all"""
    global _patching_analyzers, _current_combo_id
    if combo_id is None:
        _patching_analyzers.clear()
        _current_combo_id = None
        print("all patching analyzers have been cleaned up")
    elif combo_id in _patching_analyzers:
        del _patching_analyzers[combo_id]
        if _current_combo_id == combo_id:
            _current_combo_id = None
        print(f"patching analyzer for combo {combo_id} has been cleaned up")

def get_patching_analyzer(metadata: Optional[Dict[str, Any]] = None, combo_id: Optional[str] = None) -> PatchingAnalyzer:
    global _patching_analyzers, _current_combo_id
    
    # get the combo_id from metadata (frontend will pass sae_combo_id)
    if combo_id is None and isinstance(metadata, dict):
        meta_combo_id = metadata.get("sae_combo_id")
        if isinstance(meta_combo_id, str) and meta_combo_id.strip():
            combo_id = meta_combo_id.strip()

    # get the current combo_id
    if combo_id is None:
        try:
            # try to get the current combo_id from app.py
            import sys
            if 'app' in sys.modules:
                from app import CURRENT_BT4_SAE_COMBO_ID
                combo_id = CURRENT_BT4_SAE_COMBO_ID
            else:
                # if the app module is not loaded, use the default combo
                combo_id = "k_30_e_16"
        except (ImportError, AttributeError):
            # if cannot get, use the default combo
            combo_id = "k_30_e_16"
    
    # if the analyzer for this combo already exists, return it directly
    if combo_id in _patching_analyzers:
        return _patching_analyzers[combo_id]
    
    try:
        from transformer_lens import HookedTransformer
        from lm_saes import SparseAutoEncoder, LowRankSparseAttention
        
        # get the configuration of the current combo
        combo_cfg = get_bt4_sae_combo(combo_id)
        tc_base_path = combo_cfg["tc_base_path"]
        lorsa_base_path = combo_cfg["lorsa_base_path"]
        
        print(f"initializing BT4 Patching analyzer (combo: {combo_id})...")
        print(f"TC path: {tc_base_path}")
        print(f"LORSA path: {lorsa_base_path}")
        print(f"using model: {BT4_MODEL_NAME}")
        
        # build the cache_key (consistent with preload_circuit_models)
        cache_key = f"{BT4_MODEL_NAME}::{combo_id}"
        
        # try to get the cached model from circuits_service (using cache_key)
        try:
            from circuits_service import get_cached_models
            cached_hooked_model, cached_transcoders, cached_lorsas, _ = get_cached_models(cache_key)
            
            if cached_hooked_model is not None and cached_transcoders is not None and cached_lorsas is not None:
                if len(cached_transcoders) == 15 and len(cached_lorsas) == 15:
                    print(f"using cached model, transcoders and lorsas (combo: {combo_id})")
                    model = cached_hooked_model
                    transcoders = cached_transcoders
                    lorsas = cached_lorsas
                else:
                    raise ValueError(f"cache incomplete: transcoders={len(cached_transcoders)}, lorsas={len(cached_lorsas)}")
            else:
                raise ValueError(f"cache not found: cache_key={cache_key}")
        except (ImportError, ValueError) as e:
            print(f"cannot use cache, need to wait for preload to complete: {e}")
            print(f"tip: please call /circuit/preload_models to preload the model for combo {combo_id}")
            raise RuntimeError(
                f"model for combo {combo_id} is not preloaded. please call /circuit/preload_models to preload the model, "
                f"or wait for preload to complete before using the patching analysis functionality."
            )
        
        # create the analyzer and cache it
        analyzer = PatchingAnalyzer(model, transcoders, lorsas)
        _patching_analyzers[combo_id] = analyzer
        _current_combo_id = combo_id
        print(f"BT4 Patching analyzer initialized successfully (combo: {combo_id})")
        return analyzer
    except Exception as e:
        print(f"Patching analyzer initialization failed: {e}")
        raise


def run_feature_steering_analysis(fen: str, feature_type: str, layer: int, 
                         pos: int, feature: int, steering_scale: int, 
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """run the public interface for patching analysis"""
    analyzer = get_patching_analyzer(metadata)
    
    # run the ablation analysis
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
        return {'error': 'no activation value at this position, cannot perform ablation analysis'}
    
    # analysis result
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
    run multi feature steering analysis (each feature corresponds to one position) and return the result.

    Args:
        fen: FEN string
        feature_type: 'transcoder' or 'lorsa'
        layer: layer number
        nodes: node list, each element must contain pos/feature/steering_scale
        metadata: reserved parameter to ensure compatibility (currently not used)

    Returns:
        the same structure as run_feature_steering_analysis.
    """
    analyzer = get_patching_analyzer(metadata)
    ablation_result = analyzer.multi_steering_analysis(
        fen=fen,
        feature_type=feature_type,
        layer=layer,
        nodes=nodes,
    )
    if ablation_result is None:
        return {"error": "at least one node has no activation value at the corresponding pos, cannot perform steering"}
    return analyzer.analyze_steering_results(ablation_result, fen)