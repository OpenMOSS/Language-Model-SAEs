from typing import Dict, List, Union, Callable, Tuple
from functools import partial
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from lm_saes import SparseAutoEncoder, LanguageModelConfig
import re


class ReplacementMLP(nn.Module):
    """Wrapper for a TransformerLens MLP layer that adds in extra hooks"""

    def __init__(self, old_mlp: nn.Module):
        super().__init__()
        self.old_mlp = old_mlp
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

    def forward(self, x):
        x = self.hook_in(x)
        mlp_out = self.old_mlp(x)
        return self.hook_out(mlp_out)

class ReplacementAttention(nn.Module):
    """Wrapper for a TransformerLens Attention layer that adds in extra hooks"""
    def __init__(self, old_attn: nn.Module):
        super().__init__()
        self.old_attn = old_attn
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()
    
    def forward(self, query_input, key_input, value_input, **kwargs):
        assert torch.allclose(query_input, key_input) and torch.allclose(query_input, value_input)
        query_input = self.hook_in(query_input)
        attn_out = self.old_attn(query_input, key_input, value_input, **kwargs)
        return self.hook_out(attn_out)

class ReplacementPolicyHead(nn.Module):
    """Wrapper for LC0 PolicyHead that adds in extra hooks"""

    def __init__(self, old_policy_head: nn.Module):
        super().__init__()
        self.old_policy_head = old_policy_head
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        
        # 为Q和K添加hook points
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()

    def forward(self, x):
        x = self.hook_pre(x)
        
        # 手动实现PolicyHead的前向传播，在Q和K处添加hooks
        # Dense1层
        x = self.old_policy_head.dense1(x)
        x = self.old_policy_head.mish(x)
        
        # Q和K投影，并添加hooks
        q = self.old_policy_head.q_proj(x)
        k = self.old_policy_head.k_proj(x)
        
        # 通过hook points
        q = self.hook_q(q)
        k = self.hook_k(k)
        
        # 计算注意力分数
        scores = self.old_policy_head.hook_policy_qk_score(
            torch.matmul(q, k.transpose(-2, -1))
        )
        scores = scores * self.old_policy_head.scale
        
        # Promotion 计算
        promotion_slice = k[:, 56:64, :]
        promotion_out = self.old_policy_head.promotion(promotion_slice)
        
        promotion_out = promotion_out.transpose(1, 2)
        promotion_out_part1, promotion_out_part2 = torch.split(
            promotion_out, [3, 1], dim=1
        )
        promotion_out = torch.add(promotion_out_part1, promotion_out_part2)
        promotion_out = promotion_out.transpose(1, 2)
        promotion_out = promotion_out.reshape(x.shape[0], 1, 24)
        
        promotion_slice2 = scores[:, 48:56, 56:64]
        promotion_out2 = promotion_slice2.reshape(-1, 64, 1)
        promotion_out2 = torch.cat(
            [promotion_out2, promotion_out2, promotion_out2], dim=-1
        )
        promotion_out2 = promotion_out2.reshape(-1, 8, 24)
        
        promotion = promotion_out2 + promotion_out
        promotion = promotion.reshape(-1, 3, 64)
        
        # 组合policy
        policy = torch.cat([scores, promotion], dim=1)
        policy = policy.reshape(-1, 4288)
        
        indices_long = self.old_policy_head.indices.detach().long()
        policy_logits = policy[:, indices_long]
        
        policy_out = self.old_policy_head.hook_policy_out(policy_logits)
        return self.hook_post(policy_out)

class ReplacementModel(HookedTransformer):    
    d_transcoder: int
    transcoders: Dict[int, SparseAutoEncoder]
    lorsas: nn.ModuleList
    mlp_input_hook: str
    mlp_output_hook: str
    original_mlp_output_hook: str
    feature_input_hook: str
    feature_output_hook: str
    attn_input_hook: str
    attn_output_hook: str
    

    @classmethod
    def from_pretrained_and_transcoders(
        cls,
        model_cfg: LanguageModelConfig,
        transcoders: Dict[int, SparseAutoEncoder],
        mlp_input_hook: str = "resid_mid_after_ln",
        mlp_output_hook: str = "hook_mlp_out",
        **kwargs,
    ) -> "ReplacementModel":
        
        if model_cfg.model_name.endswith('.pt'):
            model_data = torch.load(model_cfg.model_name, map_location='cpu')
            from .utils.convert_leela_weights import convert_leela_weights
            model = convert_leela_weights(model_data, model_cfg)
        else:
            model = super().from_pretrained(
                model_cfg.model_name,
                use_flash_attn=model_cfg.use_flash_attn,
                device=model_cfg.device,
                cache_dir=model_cfg.cache_dir,
                dtype=model_cfg.dtype,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
            )

        return cls.from_pretrained_model(
            model, transcoders, mlp_input_hook, mlp_output_hook
        )

    @classmethod
    def from_pretrained_model(
        cls,
        model: HookedTransformer,
        transcoders: Dict[int, SparseAutoEncoder],
        mlp_input_hook: str = "resid_mid_after_ln",
        mlp_output_hook: str = "hook_mlp_out",
    ) -> "ReplacementModel":
        
        replacement_model = cls(
            model.cfg,
            tokenizer=None,
            move_to_device=False,
        )
        
        print(f"{replacement_model = }")
        
        original_state_dict = model.state_dict()
        model_state_dict = {}
        
        for key, value in original_state_dict.items():
            if not key.startswith('transcoders.'):
                model_state_dict[key] = value
        
        replacement_model.load_state_dict(model_state_dict, strict=True)
        
        replacement_model._configure_replacement_model(
            transcoders, mlp_input_hook, mlp_output_hook
        )
        
        # 确保整个模型在正确的设备上
        replacement_model.to(replacement_model.cfg.device)
        
        return replacement_model

    def _configure_replacement_model(
        self,
        transcoders: Dict[int, SparseAutoEncoder],
        mlp_input_hook: str,
        mlp_output_hook: str,
    ):
        """配置replacement model - 基于原始replacement_model.py的逻辑"""
        
        for layer_idx, transcoder in transcoders.items():
            transcoder.to(self.cfg.device, self.cfg.dtype)

        self.transcoders = nn.ModuleDict({str(k): v for k, v in transcoders.items()})
        self.d_transcoder = list(transcoders.values())[0].cfg.d_sae

        self.mlp_input_hook = mlp_input_hook
        self.original_mlp_output_hook = mlp_output_hook
        self.mlp_output_hook = mlp_output_hook + ".hook_out_grad"
        
        # 设置feature hooks，用于intervention相关方法
        self.feature_input_hook = mlp_input_hook
        self.feature_output_hook = mlp_output_hook

        # 包装MLP层（类似原始代码包装mlp和attn）
        for block in self.blocks:
            block.mlp = ReplacementMLP(block.mlp)
        
        # 包装LC0模型的policy_head（如果存在）
        if hasattr(self, 'policy_head') and self.policy_head is not None:
            self.policy_head = ReplacementPolicyHead(self.policy_head)
            
        self._configure_gradient_flow()
        self.setup()

    def _configure_gradient_flow(self):
        """配置梯度流 - 基于原始replacement_model.py"""

        # 为每一层配置skip connection
        for layer in range(self.cfg.n_layers):
            self._configure_skip_connection(self.blocks[layer], layer)

        def stop_gradient(acts, hook):
            return acts.detach()

        # 为LayerNorm添加gradient停止hooks（如果存在）
        for block in self.blocks:
            if hasattr(block.ln1, 'hook_scale'):
                block.ln1.hook_scale.add_hook(stop_gradient, is_permanent=True)
            if hasattr(block.ln2, 'hook_scale'):
                block.ln2.hook_scale.add_hook(stop_gradient, is_permanent=True)
            # if hasattr(block, "ln1_post") and hasattr(block.ln1_post, 'hook_scale'):
            #     block.ln1_post.hook_scale.add_hook(stop_gradient, is_permanent=True)
            # if hasattr(block, "ln2_post") and hasattr(block.ln2_post, 'hook_scale'):
            #     block.ln2_post.hook_scale.add_hook(stop_gradient, is_permanent=True)
        
        if hasattr(self.policy_head.old_policy_head.mish, 'hook_weight'):
            self.policy_head.old_policy_head.mish.hook_weight.add_hook(
                stop_gradient, is_permanent=True
            )

        # 只需要停止attention weights的梯度（包含SmolGen效果的softmax输出）
        for block in self.blocks:
            if hasattr(block, 'hook_attn_pattern'):
                block.hook_attn_pattern.add_hook(
                    stop_gradient, is_permanent=True
                )
                
        for param in self.parameters():
            param.requires_grad = False
        
        for name, bias in self._get_requires_grad_bias_params():
            bias.requires_grad_(True)  # 推荐用下划线版本，原地修改

        def enable_gradient(acts, hook):            
            if acts.is_leaf:
                acts.requires_grad = True
            return acts

        if (hasattr(self, 'policy_head') and 
                hasattr(self.policy_head, 'hook_pre')):
            self.policy_head.hook_pre.add_hook(
                enable_gradient, is_permanent=True
            )

    def _configure_skip_connection(self, block, layer):
        """为指定层配置skip connection - 基于原始replacement_model.py"""
        
        def add_skip_connection(acts: torch.Tensor, hook: HookPoint, grad_hook: HookPoint, replacement_bias: torch.Tensor):
            # 和原始代码一样的逻辑
            assert replacement_bias.requires_grad, "Replacement bias must be a parameter"
            return grad_hook((acts - replacement_bias).detach() + replacement_bias)

        # 为MLP输出添加hook和特殊的grad hook（类似原始代码）
        output_hook_parts = self.original_mlp_output_hook.split(".")
        subblock = block
        for part in output_hook_parts:
            subblock = getattr(subblock, part)
        subblock.hook_out_grad = HookPoint()
        subblock.add_hook(
            partial(
                add_skip_connection,
                grad_hook=subblock.hook_out_grad,
                replacement_bias=self.transcoders[str(layer)].b_D,
            ),
            is_permanent=True,
        )

    def _get_activation_caching_hooks(
        self,
        zero_bos: bool = False,
        sparse: bool = False,
        apply_activation_function: bool = True,
    ) -> Tuple[List, List[Tuple[str, Callable]]]:
        """获取激活缓存hooks - 简化版，只处理MLP"""
        
        activation_matrix = [None] * self.cfg.n_layers

        def cache_activations_mlp(acts, hook, layer, zero_bos):
            # 使用individual SAE而不是CrossLayerTranscoder
            transcoder_acts = self.transcoders[str(layer)].encode(
                acts,
                return_hidden_pre=not apply_activation_function
            )

            if not apply_activation_function:
                transcoder_acts = transcoder_acts[1].detach().squeeze(0)
            else:
                transcoder_acts = transcoder_acts.detach().squeeze(0)
            
            if zero_bos:
                transcoder_acts[0] = 0
            if sparse:
                transcoder_acts = transcoder_acts.to_sparse()

            activation_matrix[layer] = transcoder_acts

        activation_hooks = [
            (
                f"blocks.{layer}.{self.mlp_input_hook}",
                partial(cache_activations_mlp, layer=layer, zero_bos=zero_bos),
            )
            for layer in range(self.cfg.n_layers)
        ]

        return activation_matrix, activation_hooks

    def get_activations(
        self,
        inputs: Union[str, torch.Tensor],
        sparse: bool = False,
        zero_bos: bool = False,
        apply_activation_function: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取transcoder激活 - 类似原始代码但简化版"""
        
        activation_cache, activation_hooks = self._get_activation_caching_hooks(
            sparse=sparse,
            zero_bos=zero_bos,
            apply_activation_function=apply_activation_function,
        )
        with torch.inference_mode(), self.hooks(activation_hooks):
            logits = self(inputs)
        activation_cache = torch.stack(activation_cache)
        if sparse:
            activation_cache = activation_cache.coalesce()
        return logits, activation_cache

    @torch.no_grad()
    def setup_attribution(
        self,
        inputs: Union[str, torch.Tensor],
        sparse: bool = False,
        zero_bos: bool = False,
    ):

        # 直接使用inputs，不需要tokenizer处理
        if isinstance(inputs, torch.Tensor):
            tokens = inputs.squeeze(0)
            assert tokens.ndim == 1, "Tokens must be a 1D tensor"
        else:
            # 假设inputs是字符串，模型内部会处理
            tokens = inputs

        # 缓存激活和MLP输出
        activation_matrix, activation_hooks = self._get_activation_caching_hooks(
            sparse=sparse, zero_bos=zero_bos
        )
        mlp_out_cache, mlp_out_caching_hooks, _ = self.get_caching_hooks(
            lambda name: self.mlp_output_hook in name
        )
        
        # 添加policy head q和k的缓存hooks（如果policy_head存在）
        policy_cache = {}
        policy_caching_hooks = []
        if hasattr(self, 'policy_head'):
            policy_cache, policy_caching_hooks, _ = self.get_caching_hooks(
                lambda name: ("policy_head.hook_q" in name or 
                             "policy_head.hook_k" in name)
            )

        seq_len = len(tokens) if isinstance(tokens, torch.Tensor) else 1
        error_vectors = torch.zeros(
            [self.cfg.n_layers, seq_len, self.cfg.d_model],
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )
        
        # 运行前向传播
        all_hooks = activation_hooks + mlp_out_caching_hooks + policy_caching_hooks
        logits = self.run_with_hooks(tokens, fwd_hooks=all_hooks)
        
        # 缓存policy head的q和k activations到模型属性中
        if hasattr(self, 'policy_head'):
            for hook_name, cached_value in policy_cache.items():
                if "hook_q" in hook_name:
                    self._policy_q_activations = cached_value
                elif "hook_k" in hook_name:
                    self._policy_k_activations = cached_value

        # 计算重构和误差（只有MLP，类似原始代码的clt部分）
        transcoder_reconstruction = torch.stack([
            self.transcoders[str(layer)].decode(activation_matrix[layer])
            for layer in range(self.cfg.n_layers)
        ])
        
        error_vectors = torch.cat(
            list(mlp_out_cache.values()),
            dim=0
        ) - transcoder_reconstruction

        if zero_bos and isinstance(tokens, torch.Tensor):
            error_vectors[:, 0] = 0

        activation_matrix = torch.stack(activation_matrix)
        if sparse:
            activation_matrix = activation_matrix.coalesce()

        # 从hook_embed获取实际的embedding值
        with torch.no_grad():
            # 运行前向传播获取embedding
            _, cache = self.run_with_cache(tokens, prepend_bos=False)
            token_vectors = cache['hook_embed'].detach()  # 形状: [batch, seq_len, d_model]
            # 如果是batch维度，取第一个batch
            if token_vectors.dim() == 3:
                token_vectors = token_vectors[0]  # 形状: [seq_len, d_model]

        return logits, activation_matrix, error_vectors, token_vectors

    def setup_intervention_with_freeze(
        self, inputs: Union[str, torch.Tensor], direct_effects: bool = False
    ) -> List[Tuple[str, Callable]]:
        """设置干预和冻结 - 简化版，只处理MLP相关"""
        
        if direct_effects:
            hookpoints_to_freeze = ["hook_scale", self.feature_output_hook]
        else:
            hookpoints_to_freeze = []  # 没有attention pattern需要冻结

        freeze_cache, cache_hooks, _ = self.get_caching_hooks(
            names_filter=lambda name: any(hookpoint in name for hookpoint in hookpoints_to_freeze)
        )
        self.run_with_hooks(inputs, fwd_hooks=cache_hooks)

        def freeze_hook(activations, hook):
            cached_values = freeze_cache[hook.name]

            # 处理序列长度不匹配的情况
            if (
                "hook_scale" in hook.name or self.feature_output_hook in hook.name
            ) and activations.shape[1] != cached_values.shape[1]:
                new_activations = activations.clone()
                new_activations[:, : cached_values.shape[1]] = cached_values
                return new_activations

            assert activations.shape == cached_values.shape, (
                f"Activations shape {activations.shape} does not match cached values"
                f" shape {cached_values.shape} at hook {hook.name}"
            )
            return cached_values

        fwd_hooks = [
            (hookpoint, freeze_hook)
            for hookpoint in freeze_cache.keys()
            if self.feature_input_hook not in hookpoint
        ]

        return fwd_hooks

    def _get_feature_intervention_hooks(
        self,
        inputs: Union[str, torch.Tensor],
        interventions: List[
            Tuple[int, Union[int, slice, torch.Tensor], int, Union[int, torch.Tensor]]
        ],
        direct_effects: bool = False,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
    ):
        """获取特征干预hooks - 简化版，只处理MLP"""
        
        from collections import defaultdict
        interventions_by_layer = defaultdict(list)
        for layer, pos, feature_idx, value in interventions:
            interventions_by_layer[layer].append((pos, feature_idx, value))

        # 激活缓存
        activation_cache, activation_hooks = self._get_activation_caching_hooks(
            apply_activation_function=apply_activation_function
        )

        def intervention_hook(activations, hook, layer, layer_interventions):
            transcoder_activations = activation_cache[layer]
            if not apply_activation_function:
                transcoder_activations = (
                    self.transcoders[str(layer)]
                    .activation_function(transcoder_activations.unsqueeze(0))
                    .squeeze(0)
                )
            transcoder_output = self.transcoders[str(layer)].decode(transcoder_activations)
            for pos, feature_idx, value in layer_interventions:
                transcoder_activations[pos, feature_idx] = value
            new_transcoder_output = self.transcoders[str(layer)].decode(transcoder_activations)
            steering_vector = new_transcoder_output - transcoder_output
            return activations + steering_vector

        intervention_hooks = [
            (
                f"blocks.{layer}.{self.feature_output_hook}",
                partial(intervention_hook, layer=layer, layer_interventions=layer_interventions),
            )
            for layer, layer_interventions in interventions_by_layer.items()
        ]

        all_hooks = (
            self.setup_intervention_with_freeze(inputs, direct_effects=direct_effects)
            if direct_effects  # 不需要freeze_attention了
            else []
        )
        all_hooks += activation_hooks + intervention_hooks

        return all_hooks, activation_cache

    @torch.no_grad()
    def feature_intervention(
        self,
        inputs: Union[str, torch.Tensor],
        interventions: List[
            Tuple[int, Union[int, slice, torch.Tensor], int, Union[int, torch.Tensor]]
        ],
        direct_effects: bool = False,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """特征干预 - 简化版，只处理MLP"""
        
        feature_intervention_hook_output = self._get_feature_intervention_hooks(
            inputs,
            interventions,
            direct_effects=direct_effects,
            freeze_attention=freeze_attention,
            apply_activation_function=apply_activation_function,
        )

        hooks, activation_cache = feature_intervention_hook_output

        with self.hooks(hooks):
            logits = self(inputs)

        activation_cache = torch.stack(activation_cache)

        return logits, activation_cache

    # def _get_requires_grad_bias_params(self):
    #     """获取需要梯度的bias参数 - 类似原始代码但简化"""
    #     bias_params = []
    #     for param in self.named_parameters():
    #         if ('.b' in param[0] and 
    #             'transcoders.b_E' not in param[0] and 
    #             'old' not in param[0]):
    #             bias_params.append(param)
    #     return bias_params
    
    def _get_requires_grad_bias_params(self):
        bias_params = []
        for name, p in self.named_parameters():
            if 'old' in name:
                if 'policy_head' in name:
                    pass
                else:
                    continue
            is_bias_like = (name.endswith('.bias') or re.search(r'\.b($|[_\.])', name))
            if not is_bias_like:
                continue
            # 排除任意层级索引下的 b_E（transcoders.0.b_E、transcoders.foo.b_E 都能命中）
            if re.search(r'^transcoders\.[^.]+\.b_E($|[_\.])', name):
                continue
            # if 'b_Q' in name or 'b_K' in name:
            #     continue

            bias_params.append((name, p))
        return bias_params