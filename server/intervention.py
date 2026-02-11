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
    from src.chess_utils import get_move_from_policy_output_with_prob
    from src.chess_utils.move import get_value_from_output
except Exception:
    get_move_from_policy_output_with_prob = None
    get_value_from_output = None

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
    
    def _get_cache(self, fen: str):
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
        print(f"🔍 调用 model.run_with_cache，fen: {fen}")
        print(f"🔍 fen 类型: {type(fen)}, 长度: {len(fen) if isinstance(fen, str) else 'N/A'}")

        original_output, cache = self.model.run_with_cache(fen, prepend_bos=False)

        print(f"🔍 model.run_with_cache 返回:")
        print(f"📊 original_output 类型: {type(original_output)}")
        print(f"📊 original_output 长度: {len(original_output) if hasattr(original_output, '__len__') else 'N/A'}")
        if hasattr(original_output, '__getitem__'):
            for i in range(min(3, len(original_output))):
                item = original_output[i]
                print(f"📊 original_output[{i}] 类型: {type(item)}")
                if hasattr(item, 'shape'):
                    print(f"📊 original_output[{i}] 形状: {item.shape}")
                elif hasattr(item, '__len__'):
                    print(f"📊 original_output[{i}] 长度: {len(item)}")
                    if len(item) > 0 and isinstance(item, (list, tuple)):
                        print(f"📊 original_output[{i}][0] 类型: {type(item[0])}")
                        if hasattr(item[0], 'shape'):
                            print(f"📊 original_output[{i}][0] 形状: {item[0].shape}")

        # 检查policy logits的具体形状
        policy_logits = original_output[0]
        print(f"📊 policy_logits 形状: {policy_logits.shape}")
        print(f"📊 policy_logits[:5]: {policy_logits[:5].tolist() if hasattr(policy_logits, 'tolist') else policy_logits[:5]}")
        
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
        logit_diff = modified_output[0] - original_output[0]

        # 计算value差异 (Win - Loss)
        original_value = float(original_output[1][0][0] - original_output[1][0][2]) if get_value_from_output else 0.0
        modified_value = float(modified_output[1][0][0] - modified_output[1][0][2]) if get_value_from_output else 0.0
        value_diff = modified_value - original_value

        print(f"🔍 steering_analysis 返回数据:")
        print(f"📊 original_output[0] 形状: {original_output[0].shape}")
        print(f"📊 modified_output[0] 形状: {modified_output[0].shape}")
        print(f"📊 logit_diff 形状: {logit_diff.shape}")

        # 确保policy logits是正确的形状
        policy_original = original_output[0]
        policy_modified = modified_output[0]

        # 如果是 [1, 1858]，取 [0] 得到 [1858]
        if policy_original.ndim == 2:
            policy_original = policy_original[0]
        if policy_modified.ndim == 2:
            policy_modified = policy_modified[0]

        print(f"📊 处理后的policy_original 形状: {policy_original.shape}")
        print(f"📊 处理后的policy_modified 形状: {policy_modified.shape}")

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

        print(f"📋 返回的 original_output 长度: {len(result['original_output'])}")
        print(f"📋 返回的 modified_output 长度: {len(result['modified_output'])}")
        print(f"📋 返回的 logit_diff 长度: {len(result['logit_diff'])}")

        return result

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

        # 对于multi steering，暂时不计算value（避免索引错误）
        get_value_from_output = None

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

        # 计算logit差异，处理不同格式的输出
        try:
            # 处理original_output - 可能是张量或列表
            if isinstance(original_output, torch.Tensor):
                orig_logits = original_output
                if orig_logits.ndim == 2 and orig_logits.shape[0] == 1:
                    orig_logits = orig_logits[0]  # 从 [1, 1858] 变为 [1858]
            elif isinstance(original_output, (list, tuple)) and len(original_output) > 0:
                if isinstance(original_output[0], torch.Tensor):
                    orig_logits = original_output[0]
                    if orig_logits.ndim == 2 and orig_logits.shape[0] == 1:
                        orig_logits = orig_logits[0]
                else:
                    # 处理嵌套列表的情况
                    orig_logits = torch.tensor(original_output[0])
                    if orig_logits.ndim == 2 and orig_logits.shape[0] == 1:
                        orig_logits = orig_logits[0]
            else:
                raise ValueError(f"Unexpected original_output format: {type(original_output)}")

            # 处理modified_output - 可能是张量或列表
            if isinstance(modified_output, torch.Tensor):
                mod_logits = modified_output
                if mod_logits.ndim == 2 and mod_logits.shape[0] == 1:
                    mod_logits = mod_logits[0]  # 从 [1, 1858] 变为 [1858]
            elif isinstance(modified_output, (list, tuple)) and len(modified_output) > 0:
                if isinstance(modified_output[0], torch.Tensor):
                    mod_logits = modified_output[0]
                    if mod_logits.ndim == 2 and mod_logits.shape[0] == 1:
                        mod_logits = mod_logits[0]
                else:
                    # 处理嵌套列表的情况
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

        # 计算value差异 (Win - Loss)，添加安全检查
        original_value = 0.0
        modified_value = 0.0
        value_diff = 0.0

        # 对于multi steering，暂时不计算value（避免索引错误）
        # if get_value_from_output and len(original_output) > 1 and len(modified_output) > 1:
        #     try:
        #         original_value = float(original_output[1][0][0] - original_output[1][0][2])
        #         modified_value = float(modified_output[1][0][0] - modified_output[1][0][2])
        #         value_diff = modified_value - original_value
        #     except (IndexError, TypeError):
        #         # 如果value输出格式不正确，保持默认值0.0
        #         pass 

        # 确保输出格式正确
        def safe_to_numpy(tensor_or_list):
            if isinstance(tensor_or_list, torch.Tensor):
                # 确保是一维的 [1858] 形状
                if tensor_or_list.ndim == 2 and tensor_or_list.shape[0] == 1:
                    tensor_or_list = tensor_or_list[0]
                return tensor_or_list.detach().cpu().numpy().tolist()
            else:
                # 处理嵌套列表的情况
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
        """分析消融结果，返回对合法移动的影响"""
        if ablation_result is None:
            return None

        if LeelaBoard is None or chess is None:
            return {'error': 'Chess functionality not available'}

        print(f"🔍 analyze_steering_results 开始")
        orig_out = ablation_result.get('original_output', [])

        logit_diff = torch.tensor(ablation_result['logit_diff'])
        original_output = torch.tensor(ablation_result['original_output'])
        modified_output = torch.tensor(ablation_result['modified_output'])

        print(f"🔧 创建tensor后 - original_output 形状: {original_output.shape}")

        # 确保输出是正确的形状 [1858] 而不是 [1, 1858]
        if original_output.ndim == 2 and original_output.shape[0] == 1:
            original_output = original_output[0]
        if modified_output.ndim == 2 and modified_output.shape[0] == 1:
            modified_output = modified_output[0]

        # 确保 logit_diff 与 original_output 形状一致
        if logit_diff.ndim == 2 and logit_diff.shape[0] == 1:
            logit_diff = logit_diff[0]

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
                    "original_logit": float(original_output[idx].item()),
                    "modified_logit": float(modified_output[idx].item()),
                }
            )

        # 统一概率口径：在所有合法移动中计算softmax概率
        original_prob_by_uci: dict[str, float] = {}
        modified_prob_by_uci: dict[str, float] = {}

        # 从policy logits中提取合法移动的概率
        def get_legal_move_probs(policy_logits: torch.Tensor, fen: str) -> dict[str, float]:
            """从policy logits中计算所有合法移动的softmax概率"""
            print(f"🔍 开始计算概率, policy_logits形状: {policy_logits.shape}, 类型: {type(policy_logits)}")

            # 确保policy_logits是正确的形状
            if policy_logits.ndim == 2:
                policy_logits = policy_logits[0]  # 移除batch维度
                print(f"📊 移除batch维度后形状: {policy_logits.shape}")

            # 获取合法移动
            chess_board = chess.Board(fen)
            legal_uci_set = set(move.uci() for move in chess_board.legal_moves)
            print(f"📋 合法移动数量: {len(legal_uci_set)}")

            # 提取合法移动的logits
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

            print(f"📊 提取到的合法移动数量: {len(legal_logits)}")
            if not legal_logits:
                return {}

            # 计算softmax概率
            legal_logits_tensor = torch.tensor(legal_logits)
            print(f"🔢 logits范围: {min(legal_logits):.3f} - {max(legal_logits):.3f}")

            probs = torch.softmax(legal_logits_tensor, dim=0)
            prob_by_uci = {uci: float(prob.item()) for uci, prob in zip(legal_ucis, probs)}

            print(f"📈 概率范围: {min(prob_by_uci.values()):.6f} - {max(prob_by_uci.values()):.6f}")
            return prob_by_uci

        try:
            original_prob_by_uci = get_legal_move_probs(original_output, fen)
            modified_prob_by_uci = get_legal_move_probs(modified_output, fen)
            print(f"✅ 概率计算成功: original_prob_by_uci 有 {len(original_prob_by_uci)} 个移动, modified_prob_by_uci 有 {len(modified_prob_by_uci)} 个移动")
            if original_prob_by_uci:
                sample_uci = list(original_prob_by_uci.keys())[0]
                print(f"示例概率 - {sample_uci}: 原始={original_prob_by_uci[sample_uci]:.6f}, 修改后={modified_prob_by_uci.get(sample_uci, 0):.6f}")
        except Exception as e:
            print(f"❌ 计算概率失败: {e}")
            import traceback
            print(f"❌ 错误详情: {traceback.format_exc()}")
            original_prob_by_uci = {}
            modified_prob_by_uci = {}
        
        # 获取所有合法移动的logit差异和概率差异
        # 取前k个最高概率的移动（直接使用原始概率，不重新归一化）
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
            # 根据 logit_diff 的形状选择正确的索引方式
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
        
        # 计算value统计信息
        original_value = ablation_result.get('original_value', 0.0)
        modified_value = ablation_result.get('modified_value', 0.0)
        value_diff = ablation_result.get('value_diff', 0.0)

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
                'min_prob_diff': min_prob_diff,
                'original_value': original_value,
                'modified_value': modified_value,
                'value_diff': value_diff
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
            # 返回两套"Top moves"：
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