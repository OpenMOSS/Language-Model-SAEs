from functools import partial
from typing import Callable, List, Tuple, Union

import torch
from jaxtyping import Float
from tqdm import tqdm

from lm_saes.circuit.replacement_model import ReplacementModel
from lm_saes import CrossLayerTranscoder


@torch.no_grad()
def put_activation_to_diag(tensor, value_set_to_one=False):
    tensor = tensor.coalesce()
    assert tensor.ndim == 1
    return torch.sparse_coo_tensor(
        indices = torch.stack([tensor.indices()[0], tensor.indices()[0]], dim=0),
        values = torch.ones_like(tensor.values()) if value_set_to_one else tensor.values(),
        size = (tensor.size(0), tensor.size(0)),
        device=tensor.device
    )

@torch.no_grad()
def select_top_k_virtual_weights(
    feature_feature_virtual_weights0: Float[torch.sparse.Tensor, "n_layers d_sae d_sae"],
    feature_feature_virtual_weights1: Float[torch.sparse.Tensor, "n_layers d_sae d_sae"],
    token_feature_virtual_weights: Float[torch.Tensor, "d_vocab d_sae"],
    topk: int = 64
) -> Float[torch.Tensor, "d_sae d_sae"]:

    def filter_coo_tensor_with_threshold(
        tensor: torch.sparse.Tensor,
        top_threshold: float,
        bottom_threshold: float,
    ):
        tensor = tensor.coalesce()
        mask = torch.logical_or(
            tensor.values() > top_threshold,
            tensor.values() < bottom_threshold,
        )
        return torch.sparse_coo_tensor(
            indices=tensor.indices()[:, mask],
            values=tensor.values()[mask],
            size=tensor.size(),
            device=tensor.device,
        ).coalesce()

    all_values = torch.cat(
        [
            feature_feature_virtual_weights0.coalesce().values(),
            feature_feature_virtual_weights1.coalesce().values(),
            token_feature_virtual_weights.coalesce().values()
        ],
        dim=0
    )
    top_threshold = torch.kthvalue(all_values, all_values.numel() - topk).values
    bottom_threshold = torch.kthvalue(all_values, topk).values
    return (
        filter_coo_tensor_with_threshold(feature_feature_virtual_weights0, top_threshold, bottom_threshold),
        filter_coo_tensor_with_threshold(feature_feature_virtual_weights1, top_threshold, bottom_threshold),
        filter_coo_tensor_with_threshold(token_feature_virtual_weights, top_threshold, bottom_threshold),
    )

def init_empty_coo_tensor(size, device):
    return torch.sparse_coo_tensor(
        size=size,
        device=device,
    )


class ForwardOnlyReplacementModel(ReplacementModel):
    @classmethod
    def from_pretrained(
        cls,
        *args, **kwargs,
    ) -> "ForwardOnlyReplacementModel":
        assert "max_feature_acts" in kwargs, "max_feature_acts must be provided"
        model = super().from_pretrained(*args, **kwargs)
        model.max_feature_acts = kwargs.pop("max_feature_acts")
        
        assert model.use_lorsa

        # virtual weights
        model.token_lorsa_virtual_weights = [None] * model.cfg.n_layers
        model.token_transcoder_virtual_weights = [None] * model.cfg.n_layers
        
        model.lorsa_transcoder_virtual_weights = [[] for _ in range(model.cfg.n_layers)]
        model.transcoder_transcoder_virtual_weights = [[] for _ in range(model.cfg.n_layers)]
        model.lorsa_lorsa_virtual_weights = [[] for _ in range(model.cfg.n_layers)]
        model.transcoder_lorsa_virtual_weights = [[] for _ in range(model.cfg.n_layers)]
        
        # expected residual attributions
        d_sae = kwargs["plt_set"][0].cfg.d_sae
        d_vocab = model.cfg.d_vocab
        model.token_lorsa_era = [init_empty_coo_tensor((d_vocab, d_sae), model.cfg.device)] * model.cfg.n_layers
        model.token_transcoder_era = [init_empty_coo_tensor((d_vocab, d_sae), model.cfg.device)] * model.cfg.n_layers
        
        model.lorsa_transcoder_era = [[init_empty_coo_tensor((d_sae, d_sae), model.cfg.device) for source in range(target)] for target in range(model.cfg.n_layers)]
        model.transcoder_transcoder_era = [[init_empty_coo_tensor((d_sae, d_sae), model.cfg.device) for source in range(target)] for target in range(model.cfg.n_layers)]
        model.lorsa_lorsa_era = [[init_empty_coo_tensor((d_sae, d_sae), model.cfg.device) for source in range(target)] for target in range(model.cfg.n_layers)]
        model.transcoder_lorsa_era = [[init_empty_coo_tensor((d_sae, d_sae), model.cfg.device) for source in range(target)] for target in range(model.cfg.n_layers)]
        
        
        
        
        model._prepare_virtual_weights(kwargs.get("topk", 64))
        
        return model

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

        special_tokens = []
        for special_token in self.tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                special_tokens.extend(special_token)
            else:
                special_tokens.append(special_token)

        special_token_ids = self.tokenizer.convert_tokens_to_ids(special_tokens)
        zero_bos = zero_bos and tokens[0].cpu().item() in special_token_ids  # == self.tokenizer.bos_token_id

        # cache activations and MLP in
        (activation_matrix, lorsa_attention_pattern, activation_hooks) = (
            self._get_activation_caching_hooks(sparse=sparse, zero_bos=zero_bos)
        )

        logits = self.run_with_hooks(tokens, fwd_hooks=activation_hooks)

        if self.use_lorsa:
            lorsa_activation_matrix = activation_matrix[: self.cfg.n_layers]
            clt_activation_matrix = activation_matrix[self.cfg.n_layers :]
        else:
            lorsa_activation_matrix = None
            clt_activation_matrix = activation_matrix

        if self.use_lorsa:
            lorsa_activation_matrix = torch.stack(lorsa_activation_matrix)
            lorsa_attention_pattern = torch.stack(lorsa_attention_pattern)
        clt_activation_matrix = torch.stack(clt_activation_matrix)
        if sparse:
            if self.use_lorsa:
                lorsa_activation_matrix = lorsa_activation_matrix.coalesce()
            clt_activation_matrix = clt_activation_matrix.coalesce()

        token_vectors = self.W_E[tokens].detach()  # (n_pos, d_model)
        return (
            lorsa_activation_matrix,
            lorsa_attention_pattern,
            clt_activation_matrix,
            token_vectors,
        )
    
    def _get_activation_caching_hooks(
        self,
        zero_bos: bool = False,
        sparse: bool = False,
        apply_activation_function: bool = True,
    ) -> Tuple[List, List[Tuple[str, Callable]]]:
        if self.use_lorsa:
            activation_matrix = [None] * self.cfg.n_layers * 2
            lorsa_attention_pattern = [None] * self.cfg.n_layers
        else:
            activation_matrix = [None] * self.cfg.n_layers
        activation_hooks = []

        if self.use_lorsa:
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
        else:
            lorsa_attention_pattern = None

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
                    return layer

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

        return (
            activation_matrix,
            lorsa_attention_pattern,
            activation_hooks,
        )

    @torch.no_grad()
    def _prepare_virtual_weights(self, topk=64):
        # With lorsa added to circuit tracing framework, a good property is that we can now
        # understand attention-mediated interactions in the form of residual only virtual weights.

        # We start from transcoders as target, which is easier
        if isinstance(self.transcoders, CrossLayerTranscoder):
            raise NotImplementedError("CrossLayerTranscoder is not supported yet")
        
        # we maintain a virtual weight subset for all <upstream decoder, downstream encoder> pairs
        # recording every feature-feature pair would be intractable so we have to do pruning at
        # virtual weight stage.
        def get_decoder_encoder_virtual_weights(
            upstream_max_activation: Float[torch.Tensor, "d_sae"],
            W_E: Float[torch.Tensor, "d_model d_sae"],
            W_D: Float[torch.Tensor, "d_sae d_model"],
        ) -> Union[
            Float[torch.Tensor, "d_sae d_sae"],
            Float[torch.sparse.Tensor, "d_sae d_sae"],
        ]:  
            vw = W_D @ W_E
            vw_scaled = upstream_max_activation[:, None] * vw
            vw_scaled = vw_scaled.squeeze(0)  # [source, target]
            vw = torch.where(
                torch.logical_or(
                    vw_scaled > torch.kthvalue(vw_scaled, vw_scaled.size(0) - topk, dim=0).values,
                    vw_scaled < torch.kthvalue(vw_scaled, topk, dim=0).values
                ),
                vw,
                0,
            ).to_sparse().coalesce()
            return vw


        for target_layer in tqdm(range(self.cfg.n_layers), desc="preparing top virtual weights"):
            self.token_lorsa_virtual_weights[target_layer] = get_decoder_encoder_virtual_weights(
                upstream_max_activation=torch.ones(self.cfg.d_vocab, device=self.cfg.device),
                W_E=self.lorsas[target_layer].W_E,
                W_D=self.W_E,
            )
            self.token_transcoder_virtual_weights[target_layer] = get_decoder_encoder_virtual_weights(
                upstream_max_activation=torch.ones(self.cfg.d_vocab, device=self.cfg.device),
                W_E=self.transcoders[target_layer].W_E,
                W_D=self.W_E,
            )

            self.lorsa_transcoder_virtual_weights[target_layer] = torch.stack([
                get_decoder_encoder_virtual_weights(
                    upstream_max_activation=self.max_feature_acts['lorsa'][source_lorsa_layer],
                    W_E=self.transcoders.W_E[target_layer],
                    W_D=self.lorsas[source_lorsa_layer].W_D,
                )
                for source_lorsa_layer in range(target_layer + 1)
            ])

            self.transcoder_transcoder_virtual_weights[target_layer] = torch.stack([
                get_decoder_encoder_virtual_weights(
                    upstream_max_activation=self.max_feature_acts['transcoder'][source_transcoder_layer],
                    W_E=self.transcoders.W_E[target_layer],
                    W_D=self.transcoders.W_D[source_transcoder_layer],
                )
                for source_transcoder_layer in range(target_layer + 1)
            ])
        
            if target_layer == 0:
                # for target layer 0, we do not have any lorsa_lorsa_virtual_weights and transcoder_lorsa_virtual_weights
                self.lorsa_lorsa_virtual_weights[target_layer] = torch.sparse_coo_tensor(
                    size=(0, self.lorsas[target_layer].cfg.d_sae, self.lorsas[target_layer].cfg.d_sae)
                )
                self.transcoder_lorsa_virtual_weights[target_layer] = torch.sparse_coo_tensor(
                    size=(0, self.transcoders[target_layer].cfg.d_sae, self.transcoders[target_layer].cfg.d_sae)
                )
                continue

            self.lorsa_lorsa_virtual_weights[target_layer] = torch.stack([
                get_decoder_encoder_virtual_weights(
                    upstream_max_activation=self.max_feature_acts['lorsa'][source_lorsa_layer],
                    W_E=self.lorsas[target_layer].W_E,
                    W_D=self.lorsas[source_lorsa_layer].W_D,
                )
                for source_lorsa_layer in range(target_layer)
            ])

            self.transcoder_lorsa_virtual_weights[target_layer] = torch.stack([
                get_decoder_encoder_virtual_weights(
                    upstream_max_activation=self.max_feature_acts['transcoder'][source_transcoder_layer],
                    W_E=self.lorsas[target_layer].W_E,
                    W_D=self.transcoders.W_D[source_transcoder_layer],
                )
                for source_transcoder_layer in range(target_layer)
            ])

    @torch.no_grad()
    def record_activation_matrix(
        self,
        input_ids: Float[torch.Tensor, "n_pos"],
        lorsa_activation_matrix: Float[torch.sparse.Tensor, "n_layers n_pos d_sae"],
        clt_activation_matrix: Float[torch.sparse.Tensor, "n_layers n_pos d_sae"],
        attention_pattern: Float[torch.sparse.Tensor, "n_layers n_qk_heads n_pos n_pos"]
    ):
        # we do target=transcoders first which is easier.
        n_layer, n_pos, d_sae = lorsa_activation_matrix.size()
        for pos in range(1, n_pos):
            for target_layer in range(n_layer):
                target_activation_matrix = clt_activation_matrix[target_layer, pos]
                
                token_transcoder_era = torch.sparse.mm(
                    self.token_transcoder_virtual_weights[n_layer][input_ids[pos]],  # d_sae in coo
                    put_activation_to_diag(target_activation_matrix, value_set_to_one=True)
                )
                
                for source_layer in range(target_layer):
                    
                    