from functools import partial
from torch._tensor import Tensor
from typing import Callable, List, Tuple, Union
import torch
from jaxtyping import Float
from tqdm import tqdm

from lm_saes.circuit.tracing.replacement_model import ReplacementModel
from lm_saes import CrossLayerTranscoder
from lm_saes.circuit.utils.batched_features import BatchedFeatures, ConnectedFeatures


def init_empty_coo_tensor(size, device):
    return torch.sparse_coo_tensor(
        size=size,
        device=device,
    )


class ForwardOnlyReplacementModel(ReplacementModel):
    def _configure_gradient_flow(self):
        pass

    @torch.no_grad()
    def setup_attribution(
        self,
        inputs: Union[str, torch.Tensor],
        sparse: bool = False,
        zero_bos: bool = True,
    ):
        if isinstance(inputs, torch.Tensor):
            tokens = inputs.squeeze(0)
            assert tokens.ndim == 1, "Tokens must be a 1D tensor"
        else:
            assert isinstance(inputs, str), "Inputs must be a string"
            tokenized = self.tokenizer(inputs, return_tensors="pt").input_ids.to(self.cfg.device)
            tokens = tokenized.squeeze(0)
        tokens = tokens[:self.cfg.n_ctx]

        special_tokens = []
        for special_token in self.tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                special_tokens.extend(special_token)
            else:
                special_tokens.append(special_token)

        special_token_ids = self.tokenizer.convert_tokens_to_ids(special_tokens)
        zero_bos = zero_bos and tokens[0].cpu().item() in special_token_ids  # == self.tokenizer.bos_token_id

        # cache activations and MLP in
        (
            activation_matrix,
            lorsa_attention_pattern,
            activation_hooks,
            attn_hook_scales,
            mlp_hook_scales,
        ) = (
            self._get_activation_caching_hooks(sparse=sparse, zero_bos=zero_bos)
        )

        self.run_with_hooks(tokens, fwd_hooks=activation_hooks)

        if self.use_lorsa:
            lorsa_activation_matrix = activation_matrix[: self.cfg.n_layers]
            clt_activation_matrix = activation_matrix[self.cfg.n_layers :]
        else:
            lorsa_activation_matrix = None
            clt_activation_matrix = activation_matrix

        if self.use_lorsa:
            lorsa_activation_matrix = torch.stack(lorsa_activation_matrix)
            lorsa_attention_pattern = torch.stack(lorsa_attention_pattern)
            attn_hook_scales = torch.stack(attn_hook_scales)
        clt_activation_matrix = torch.stack(clt_activation_matrix)
        mlp_hook_scales = torch.stack(mlp_hook_scales)
        if sparse:
            if self.use_lorsa:
                lorsa_activation_matrix = lorsa_activation_matrix.coalesce()
            clt_activation_matrix = clt_activation_matrix.coalesce()

        return (
            lorsa_activation_matrix,
            lorsa_attention_pattern,
            clt_activation_matrix,
            attn_hook_scales,
            mlp_hook_scales,
        )
    
    def _get_activation_caching_hooks(
        self,
        zero_bos: bool = False,
        sparse: bool = False,
        apply_activation_function: bool = True,
    ) -> Tuple[List, List[Tuple[str, Callable]]]:
        activation_matrix = [None] * self.cfg.n_layers * 2
        lorsa_attention_pattern = [None] * self.cfg.n_layers
        attn_hook_scales = [None] * self.cfg.n_layers
        mlp_hook_scales = [None] * self.cfg.n_layers
        
        activation_hooks = []

        def cache_activations_attn(acts, hook, layer, zero_bos):
            encode_result = self.lorsas[layer].encode(
                acts,
                return_hidden_pre=not apply_activation_function,
                return_attention_pattern=True,
                return_attention_score=True,
            )

            if not apply_activation_function:
                lorsa_acts = encode_result[1].detach().squeeze(0)
                pattern = encode_result[2].detach().squeeze(0)
            else:
                lorsa_acts = encode_result[0].detach().squeeze(0)
                pattern = encode_result[1].detach().squeeze(0)

            if zero_bos:
                lorsa_acts[0] = 0
            if sparse:
                lorsa_acts = lorsa_acts.to_sparse()

            activation_matrix[layer] = lorsa_acts
            lorsa_attention_pattern[layer] = pattern

        activation_hooks.extend(
            [
                (
                    f"blocks.{layer}.{self.attn_input_hook}",
                    partial(cache_activations_attn, layer=layer, zero_bos=zero_bos),
                )
                for layer in range(self.cfg.n_layers)
            ]
        )

        def cache_activations_mlp(acts, hook, layer, zero_bos):
            # Use unified encode_layer interface for both TranscoderSet and CrossLayerTranscoder
            transcoder_acts = self.transcoders.encode_layer(
                acts, layer, apply_activation_function=apply_activation_function
            )

            # Handle tuple return (feature_acts, hidden_pre) - extract the appropriate one
            if isinstance(transcoder_acts, tuple):
                transcoder_acts = transcoder_acts[1 if not apply_activation_function else 0]

            transcoder_acts = transcoder_acts.detach().squeeze(0)

            if zero_bos:
                transcoder_acts[0] = 0
            if sparse:
                transcoder_acts = transcoder_acts.to_sparse()

            def mlp_offset(layer):
                return self.cfg.n_layers + layer

            activation_matrix[mlp_offset(layer)] = transcoder_acts

        activation_hooks.extend(
            [
                (
                    f"blocks.{layer}.{self.mlp_input_hook}",
                    partial(cache_activations_mlp, layer=layer, zero_bos=zero_bos),
                )
                for layer in range(self.cfg.n_layers)
            ]
        )
        
        def cache_activations_attn_ln_scales(acts, hook, layer):
            attn_hook_scales[layer] = acts.squeeze(0, -1)
        
        def cache_activations_mlp_ln_scales(acts, hook, layer):
            mlp_hook_scales[layer] = acts.squeeze(0, -1)
        
        activation_hooks.extend(
            [
                (
                    f"blocks.{layer}.ln1.hook_scale",
                    partial(cache_activations_attn_ln_scales, layer=layer),
                )
                for layer in range(self.cfg.n_layers)
            ]
        )
        
        activation_hooks.extend(
            [
                (
                    f"blocks.{layer}.ln2.hook_scale",
                    partial(cache_activations_mlp_ln_scales, layer=layer),
                )
                for layer in range(self.cfg.n_layers)
            ]
        )

        return (
            activation_matrix,
            lorsa_attention_pattern,
            activation_hooks,
            attn_hook_scales,
            mlp_hook_scales,
        )

    @torch.no_grad()
    def prepare_virtual_weights(self, interested_features):
        # With lorsa added to circuit tracing framework, a good property is that we can now
        # understand attention-mediated interactions only in the form of residual-only virtual weights.

        if isinstance(self.transcoders, CrossLayerTranscoder):
            raise NotImplementedError("CrossLayerTranscoder is not supported yet")
        
        self.interested_features = interested_features
        
        lorsa_encoders = torch.stack([
            self.lorsas[layer].W_E * self.blocks[layer].ln1.w[:, None] for layer in range(self.cfg.n_layers)
        ])  # n_layers, d_model, d_sae
        
        lorsa_decoders = torch.stack([
            self.lorsas[layer].W_D for layer in range(self.cfg.n_layers)
        ])  # n_layers, d_sae, d_model
        
        transcoder_encoders = torch.stack([
            self.transcoders[layer].W_E * self.blocks[layer].ln2.w[:, None] for layer in range(self.cfg.n_layers)
        ])  # n_layers, d_model, d_sae
        
        transcoder_decoders = self.transcoders.W_D  # n_layers, d_model, d_sae
        
        # pythia-only
        lorsa_decoders -= lorsa_decoders.mean(dim=1, keepdim=True)
        transcoder_decoders -= transcoder_decoders.mean(dim=1, keepdim=True)

        interested_encoders = torch.where(
            interested_features.is_lorsa[:, None],
            lorsa_encoders[interested_features.layer, :, interested_features.index],
            transcoder_encoders[interested_features.layer, :, interested_features.index]
        )  # batch_size d_model
        
        interested_decoders = torch.where(
            interested_features.is_lorsa[:, None],
            lorsa_decoders[interested_features.layer, interested_features.index],
            transcoder_decoders[interested_features.layer, interested_features.index]
        )  # batch_size d_model

        # pythia-only
        self.upstream_lorsa_vw = torch.where(
            # interested_features.sublayer_index()[:, None, None] > torch.arange(self.cfg.n_layers, device=interested_features.layer.device)[None, :, None],
            interested_features.layer[:, None, None] > torch.arange(self.cfg.n_layers, device=interested_features.layer.device)[None, :, None],
            torch.einsum("lsd,bd->bls", lorsa_decoders, interested_encoders),
            0,
        )  # batch n_layer d_sae

        self.upstream_transcoder_vw = torch.where(
            interested_features.layer[:, None, None] > torch.arange(self.cfg.n_layers, device=interested_features.layer.device)[None, :, None],
            torch.einsum("lsd,bd->bls", transcoder_decoders, interested_encoders),
            0,
        )  # batch n_layer d_sae

        self.downstream_lorsa_vw = torch.where(
            interested_features.layer[:, None, None] < torch.arange(self.cfg.n_layers, device=interested_features.layer.device)[None, :, None],
            torch.einsum("lds,bd->bls", lorsa_encoders, interested_decoders),
            0,
        )  # batch n_layer d_sae

        self.downstream_transcoder_vw = torch.where(
            # interested_features.sublayer_index()[:, None, None] <= torch.arange(self.cfg.n_layers, device=interested_features.layer.device)[None, :, None],
            interested_features.layer[:, None, None] < torch.arange(self.cfg.n_layers, device=interested_features.layer.device)[None, :, None],
            torch.einsum("lds,bd->bls", transcoder_encoders, interested_decoders),
            0,
        )  # batch n_layer d_sae
        
        
        batch_size = len(interested_features)
        self.upstream_lorsa_attribution = torch.zeros(
            batch_size,
            self.cfg.n_layers,
            self.lorsas[0].cfg.d_sae,
            device=self.cfg.device
        )
        self.upstream_transcoder_attribution = torch.zeros(
            batch_size,
            self.cfg.n_layers,
            self.transcoders[0].cfg.d_sae,
            device=self.cfg.device
        )
        self.downstream_lorsa_attribution = torch.zeros(
            batch_size,
            self.cfg.n_layers,
            self.lorsas[0].cfg.d_sae,
            device=self.cfg.device
        )
        self.downstream_transcoder_attribution = torch.zeros(
            batch_size,
            self.cfg.n_layers,
            self.transcoders[0].cfg.d_sae,
            device=self.cfg.device
        )
        
        self.accumulated_lorsa_acts = torch.zeros(
            self.cfg.n_layers,
            self.lorsas[0].cfg.d_sae,
            device=self.cfg.device
        )
        
        self.accumulated_transcoder_acts = torch.zeros(
            self.cfg.n_layers,
            self.transcoders[0].cfg.d_sae,
            device=self.cfg.device
        )
    
    @torch.no_grad()
    def parse_global_weight_results(self, top_k=10):
        connected_features: List[ConnectedFeatures] = []
        for i in range(len(self.interested_features)):
            upstream_lorsa_twera_values, upstream_lorsa_twera_indices = (self.upstream_lorsa_attribution[i] / self.accumulated_lorsa_acts).nan_to_num(0).flatten().topk(top_k)
            upstream_transcoder_twera_values, upstream_transcoder_twera_indices = (self.upstream_transcoder_attribution[i] / self.accumulated_transcoder_acts).nan_to_num(0).flatten().topk(top_k)
            downstream_lorsa_twera_values, downstream_lorsa_twera_indices = (self.downstream_lorsa_attribution[i] / self.accumulated_lorsa_acts).nan_to_num(0).flatten().topk(top_k)
            downstream_transcoder_twera_values, downstream_transcoder_twera_indices = (self.downstream_transcoder_attribution[i] / self.accumulated_transcoder_acts).nan_to_num(0).flatten().topk(top_k)

            def tensor_divmod(tensor, divisor):
                return torch.stack([tensor // divisor, tensor % divisor])
            
            upstream_lorsa_twera_indices = tensor_divmod(upstream_lorsa_twera_indices, self.lorsas[0].cfg.d_sae)
            upstream_transcoder_twera_indices = tensor_divmod(upstream_transcoder_twera_indices, self.transcoders[0].cfg.d_sae)
            downstream_lorsa_twera_indices = tensor_divmod(downstream_lorsa_twera_indices, self.lorsas[0].cfg.d_sae)
            downstream_transcoder_twera_indices = tensor_divmod(downstream_transcoder_twera_indices, self.transcoders[0].cfg.d_sae)

            # pythia-only
            upstream_features = BatchedFeatures(
                layer=torch.cat([upstream_lorsa_twera_indices[0], upstream_transcoder_twera_indices[0]]),
                index=torch.cat([upstream_lorsa_twera_indices[1], upstream_transcoder_twera_indices[1]]),
                is_lorsa=torch.cat([torch.ones_like(upstream_lorsa_twera_indices[0], dtype=torch.bool), torch.zeros_like(upstream_transcoder_twera_indices[0], dtype=torch.bool)]),
            ) if self.interested_features.layer[i] > 0 else BatchedFeatures.empty()

            downstream_features = BatchedFeatures(
                layer=torch.cat([downstream_lorsa_twera_indices[0], downstream_transcoder_twera_indices[0]]),
                index=torch.cat([downstream_lorsa_twera_indices[1], downstream_transcoder_twera_indices[1]]),
                is_lorsa=torch.cat([torch.ones_like(downstream_lorsa_twera_indices[0], dtype=torch.bool), torch.zeros_like(downstream_transcoder_twera_indices[0], dtype=torch.bool)]),
            ) if self.interested_features.layer[i] < self.cfg.n_layers else BatchedFeatures.empty()

            upstream_values = torch.cat(
                [upstream_lorsa_twera_values, upstream_transcoder_twera_values]
            ) if self.interested_features.layer[i] > 0 else torch.tensor(
                [], device=self.cfg.device
            )

            downstream_values = torch.cat(
                [downstream_lorsa_twera_values, downstream_transcoder_twera_values]
            ) if self.interested_features.layer[i] < self.cfg.n_layers else torch.tensor(
                [], device=self.cfg.device
            )

            assert len(upstream_features) == len(upstream_values)
            assert len(downstream_features) == len(downstream_values)

            connected_features.append(
                ConnectedFeatures(
                    upstream_features=upstream_features,
                    downstream_features=downstream_features,
                    upstream_values=upstream_values,
                    downstream_values=downstream_values,
                ).sort()
            )
            
        return connected_features


    @torch.no_grad()
    def record_activation_matrix(
        self,
        inputs: Union[str, torch.Tensor],
        lorsa_activation_matrix: Float[torch.sparse.Tensor, "n_layers n_pos d_sae"],
        clt_activation_matrix: Float[torch.sparse.Tensor, "n_layers n_pos d_sae"],
        attention_pattern: Float[torch.sparse.Tensor, "n_layers n_qk_heads n_pos n_pos"],
        attn_hook_scales: Float[torch.Tensor, "n_layers pos"],
        mlp_hook_scales: Float[torch.Tensor, "n_layers pos"]
    ):
        if isinstance(inputs, torch.Tensor):
            input_ids = inputs.squeeze(0)
            assert input_ids.ndim == 1, "Tokens must be a 1D tensor"
        else:
            assert isinstance(inputs, str), "Inputs must be a string"
            tokenized = self.tokenizer(inputs, return_tensors="pt").input_ids.to(self.cfg.device)
            input_ids = tokenized.squeeze(0)
        input_ids = input_ids[:self.cfg.n_ctx]
        n_ctx = input_ids.size(0)

        # first we record all : Tensoractivations
        # bos_removed_pattern = attention_pattern[..., 1:].sum(-1).repeat_interleave(self.lorsas[0].qk_exp_factor, dim=1).permute(0, 2, 1)
        self.accumulated_lorsa_acts += lorsa_activation_matrix.sum(1).to_dense()
        self.accumulated_transcoder_acts += clt_activation_matrix.sum(1).to_dense()

        # We can think of transcoder features as self-attending so each feature has an attention pattern
        
        # interested act: [batch pos] coo -> dense
        # vw: [batch n_layer d_sae] dense
        # other acts: [n_layer pos d_sae] coo
        # pattern: [batch qpos kpos] dense
        
        # 1. get activation matrix of interested features
        acts = []
        for i in range(len(self.interested_features)):
            if self.interested_features.is_lorsa[i]:
                acts.append(
                    lorsa_activation_matrix[
                        self.interested_features.layer[i],
                        :,
                        self.interested_features.index[i]
                    ]
                )
            else:
                acts.append(
                    clt_activation_matrix[
                        self.interested_features.layer[i],
                        :,
                        self.interested_features.index[i]
                    ]
                )
        acts = torch.stack(acts).to_dense()  # batch pos
        if acts.eq(0).all():
            # this should have no effect on either downstream or upstream attribution
            return
        
        # 2. get attn pattern of interested features
        patterns = []
        for i in range(len(self.interested_features)):
            if self.interested_features.is_lorsa[i]:
                patterns.append(
                    attention_pattern[
                        self.interested_features.layer[i],
                        self.interested_features.index[i] // self.lorsas[0].qk_exp_factor,
                    ]
                )
            else:
                patterns.append(
                    torch.eye(n_ctx).to(self.cfg.device)
                )
        patterns = torch.stack(patterns)  # batch qpos kpos
        
        hook_scales = []
        for i in range(len(self.interested_features)):
            if self.interested_features.is_lorsa[i]:
                hook_scales.append(
                    attn_hook_scales[self.interested_features.layer[i]]
                )
            else:
                hook_scales.append(
                    mlp_hook_scales[self.interested_features.layer[i]]
                )
        hook_scales = torch.stack(hook_scales)  # batch kpos
        
        
        # upstream connections
        # 1. we compute aij first to get rid of pos dim
        attribution_to_kpos = torch.einsum("bq,bqk->bk", acts, patterns)
        
        # conceptually hook scales should divide all upstream features
        # we do this to target features since it is easier to handle
        # this should only happen in computing aij and should not count to E[aj]
        attribution_to_kpos = attribution_to_kpos / hook_scales
        
        def compute_aij(
            act_matrix: Float[torch.sparse.Tensor, "n_layer pos d_sae"],
        ):
            assert act_matrix.ndim == 3
            return torch.stack([
                torch.sparse.mm(  # we use sparse mm to sum over pos dim - aij should be summed over token positions
                    attribution_to_kpos,
                    act_matrix[layer]
                )  # batch d_sae
                for layer in range(self.cfg.n_layers)
            ], dim=1).to_dense()  # batch n_layer d_sae
            
        upstream_lorsa_aij = compute_aij(lorsa_activation_matrix)
        upstream_transcoder_aij = compute_aij(clt_activation_matrix)
                
        # 2. multiply by Vij
        self.upstream_lorsa_attribution += upstream_lorsa_aij * self.upstream_lorsa_vw
        self.upstream_transcoder_attribution += upstream_transcoder_aij * self.upstream_transcoder_vw


        # downstream connections
        # we first expand every downstream lorsa feature to attention form ie q_pos k_pos
        # then we sum over q_pos to get contribution from k_pos, regardless which q_pos this feature is contributing to
        # these two steps can be done with sparse mm
        # then we use the same trick above to remove k_pos dim so we can easily do in dense format
        
        # 1. downstream lorsa features
        downstream_lorsa_aij = []
        for layer in range(self.cfg.n_layers):
            lorsa_k_pos_contribution = torch.einsum(
                "qs,sqk->sk",
                lorsa_activation_matrix[layer].to_dense(),
                attention_pattern[layer].repeat_interleave(self.lorsas[layer].qk_exp_factor, dim=0)
            )
            # same as above. we divide attn_hook_scale here
            lorsa_k_pos_contribution = lorsa_k_pos_contribution / attn_hook_scales[layer][None, :]
            
            
            downstream_lorsa_aij.append(
                torch.einsum(
                    "sk,bk->bs",
                    lorsa_k_pos_contribution,
                    acts
                )
            )
            
        downstream_lorsa_aij = torch.stack(downstream_lorsa_aij, dim=1)  # batch n_layer d_sae
        self.downstream_lorsa_attribution += downstream_lorsa_aij * self.downstream_lorsa_vw
        
        # 2. downstream transcoders
        downstream_transcoder_aij = torch.stack([
            torch.sparse.mm(
                acts / mlp_hook_scales[layer],  # this is normal handling of hook scale
                clt_activation_matrix[layer]
            )
            for layer in range(self.cfg.n_layers)
        ], dim=1)  # batch n_layer d_sae
        
        self.downstream_transcoder_attribution += downstream_transcoder_aij * self.downstream_transcoder_vw
