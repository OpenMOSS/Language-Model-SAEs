from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Callable, List, Tuple, Union

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_saes.backend.language_model import LanguageModelConfig
from lm_saes.clt import CrossLayerTranscoder
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.resource_loaders import load_model

from .utils.attribution_utils import ensure_tokenized
from .utils.transcoder_set import TranscoderSet

# Type definition for transcoders: per-layer (dict) or cross-layer (CLT)
TranscoderType = TranscoderSet | CrossLayerTranscoder

# Type definition for an intervention tuple (layer, position, feature_idx, value)
Intervention = tuple[int | torch.Tensor, int | slice | torch.Tensor, int | torch.Tensor, int | torch.Tensor, str]


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


class ReplacementUnembed(nn.Module):
    """Wrapper for a TransformerLens Unembed layer that adds in extra hooks"""

    def __init__(self, old_unembed: nn.Module):
        super().__init__()
        self.old_unembed = old_unembed
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    @property
    def W_U(self):
        return self.old_unembed.W_U

    @property
    def b_U(self):
        return self.old_unembed.b_U

    def forward(self, x):
        x = self.hook_pre(x)
        x = self.old_unembed(x)
        return self.hook_post(x)


class ReplacementModel(HookedTransformer):
    d_transcoder: int
    transcoders: TranscoderType  # Support both per-layer and cross-layer transcoders
    lorsas: nn.ModuleList
    mlp_input_hook: str
    mlp_output_hook: str
    attn_input_hook: str = None
    attn_output_hook: str = None

    @classmethod
    def from_config(
        cls,
        config: LanguageModelConfig,
        transcoders: TranscoderType,
        lorsas: List[LowRankSparseAttention],
        mlp_input_hook: str = "mlp.hook_in",
        mlp_output_hook: str = "mlp.hook_out",
        attn_input_hook: str = "attn.hook_in",
        attn_output_hook: str = "attn.hook_out",
        use_lorsa: bool = True,
        **kwargs,
    ) -> "ReplacementModel":
        """Create a ReplacementModel from a given LanguageModelConfig and dict of transcoders

        Args:
            config (LanguageModelConfig): the config of the HookedTransformer that this
                ReplacmentModel will inherit from
            transcoders: CrossLayerTranscoder: A CrossLayerTranscoder
            lorsas: List[LowRankSparseAttention]: A list of LowRankSparseAttention modules
            mlp_input_hook (List[str], optional): The hookpoints of the model that transcoders
                hook into. Defaults to "mlp.hook_in".
            mlp_output_hook (List[str], optional): The hookpoints of the model that transcoders
                hook out of. Defaults to "mlp.hook_out".
            attn_input_hook (List[str], optional): The hookpoints of the model that lorsa
                hook into. Defaults to "attn.hook_in".
            attn_output_hook (List[str], optional): The hookpoints of the model that lorsa
                hook out of. Defaults to "attn.hook_out".
            **kwargs: Additional keyword arguments to pass to the HookedTransformer constructor.

        Returns:
            ReplacementModel: The loaded ReplacementModel
        """

        model = load_model(config).model  # type: ignore[reportAttributeAccessIssue]
        model._configure_replacement_model(
            transcoders, lorsas, mlp_input_hook, mlp_output_hook, attn_input_hook, attn_output_hook, use_lorsa
        )
        return model

    @classmethod
    def from_pretrained_and_transcoders(
        cls,
        model_cfg: LanguageModelConfig,
        transcoders: TranscoderType,
        lorsas: List[LowRankSparseAttention],
        mlp_input_hook: str = "mlp.hook_in",
        mlp_output_hook: str = "mlp.hook_out",
        attn_input_hook: str = "attn.hook_in",
        attn_output_hook: str = "attn.hook_out",
        use_lorsa: bool = True,
        **kwargs,
    ) -> "ReplacementModel":
        """Create a ReplacementModel from the name of HookedTransformer and dict of transcoders

        Args:
            model_cfg (LanguageModelConfig): the config of the pretrained HookedTransformer that this
                ReplacmentModel will inherit from
            transcoders (Union[Dict[int, SparseAutoEncoder], CrossLayerTranscoder]): A dict that maps from layer -> Transcoder or a CrossLayerTranscoder
            mlp_input_hook (List[str], optional): The hookpoints of the model that transcoders
                hook into for inputs. Defaults to "mlp.hook_in".
            mlp_output_hook (List[str], optional): The hookpoints of the model that transcoders
                hook into for outputs. Defaults to "mlp.hook_out".
            attn_input_hook (List[str], optional): The hookpoints of the model that lorsa
                hook into. Defaults to "attn.hook_in".
            attn_output_hook (List[str], optional): The hookpoints of the model that lorsa
                hook out of. Defaults to "attn.hook_out".

        Returns:
            ReplacementModel: The loaded ReplacementModel
        """
        hf_model = (
            AutoModelForCausalLM.from_pretrained(
                (
                    model_cfg.model_name
                    if model_cfg.model_from_pretrained_path is None
                    else model_cfg.model_from_pretrained_path
                ),
                cache_dir=model_cfg.cache_dir,
                local_files_only=model_cfg.local_files_only,
                dtype=model_cfg.dtype,
                trust_remote_code=True,
            )
            if model_cfg.load_ckpt
            else None
        )
        hf_tokenizer = (
            AutoTokenizer.from_pretrained(
                (
                    model_cfg.model_name
                    if model_cfg.model_from_pretrained_path is None
                    else model_cfg.model_from_pretrained_path
                ),
                trust_remote_code=True,
                use_fast=True,
                add_bos_token=True,
                local_files_only=model_cfg.local_files_only,
            )
            if model_cfg.load_ckpt
            else None
        )
        model = super().from_pretrained(
            model_cfg.model_name,
            use_flash_attn=model_cfg.use_flash_attn,
            device=model_cfg.device,
            cache_dir=model_cfg.cache_dir,
            hf_model=hf_model,
            hf_config=hf_model.config if hf_model is not None else None,
            tokenizer=hf_tokenizer,
            dtype=model_cfg.dtype,  # type: ignore ; issue with transformer_lens
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )

        model._configure_replacement_model(
            transcoders,
            lorsas,
            mlp_input_hook,
            mlp_output_hook,
            attn_input_hook,
            attn_output_hook,
            use_lorsa,
        )
        return model

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        model_cfg: LanguageModelConfig,
        transcoders: TranscoderType,
        lorsas: List[LowRankSparseAttention],
        mlp_input_hook: str = "mlp.hook_in",
        mlp_output_hook: str = "mlp.hook_out",
        attn_input_hook: str = "attn.hook_in",
        attn_output_hook: str = "attn.hook_out",
        use_lorsa: bool = True,
        **kwargs,
    ) -> "ReplacementModel":
        """Create a ReplacementModel from the name of HookedTransformer and dict of transcoders

        Args:
            model_name (str): the name of the pretrained HookedTransformer that this
                ReplacmentModel will inherit from
            transcoders: (str): Either a predefined transcoder set name, or a config file
                defining where to load them from
            device (torch.device, Optional): the device onto which to load the transcoders
                and HookedTransformer.

        Returns:
            ReplacementModel: The loaded ReplacementModel
        """

        return cls.from_pretrained_and_transcoders(
            model_cfg,
            transcoders,
            lorsas,
            mlp_input_hook=mlp_input_hook,
            mlp_output_hook=mlp_output_hook,
            attn_input_hook=attn_input_hook,
            attn_output_hook=attn_output_hook,
            use_lorsa=use_lorsa,
            **kwargs,
        )

    def _configure_replacement_model(
        self,
        transcoders: TranscoderType,
        lorsas: List[LowRankSparseAttention],
        mlp_input_hook: str,
        mlp_output_hook: str,
        attn_input_hook: str,
        attn_output_hook: str,
        use_lorsa: bool = True,
    ):
        # Configure Transcoders
        transcoders.to(self.cfg.device, self.cfg.dtype)
        self.add_module("transcoders", transcoders)
        self.d_transcoder = transcoders.cfg.d_sae
        self.mlp_input_hook = mlp_input_hook
        self.original_mlp_output_hook = mlp_output_hook
        self.mlp_output_hook = mlp_output_hook + ".hook_out_grad"

        for block in self.blocks:
            block.mlp = ReplacementMLP(block.mlp)

        # Configure Lorsa if needed
        self.use_lorsa = use_lorsa
        if use_lorsa and lorsas is not None:
            for lorsa in lorsas:
                lorsa.to(self.cfg.device, self.cfg.dtype)
            self.add_module("lorsas", nn.ModuleList(lorsas))
            self.d_lorsa = lorsas[0].cfg.d_sae
            self.attn_input_hook = attn_input_hook
            self.original_attn_output_hook = attn_output_hook
            self.attn_output_hook = attn_output_hook + ".hook_out_grad"
            for block in self.blocks:
                block.attn = ReplacementAttention(block.attn)

        self.unembed = ReplacementUnembed(self.unembed)

        self._configure_gradient_flow()
        self._deduplicate_attention_buffers()
        self.setup()

    def _configure_gradient_flow(self):
        def stop_gradient(acts, hook):
            return acts.detach()

        for layer in range(self.cfg.n_layers):
            self._configure_skip_connection(self.blocks[layer], layer)
            if self.use_lorsa and self.lorsas[layer].cfg.use_post_qk_ln:
                self.lorsas[layer].ln_q.hook_scale.add_hook(stop_gradient, is_permanent=True)
                self.lorsas[layer].ln_k.hook_scale.add_hook(stop_gradient, is_permanent=True)

        for block in self.blocks:
            # We don't need to stop gradient for the attention pattern if we have lorsa
            # because the gradient will be stopped at attn.hook_out_grad
            # block.attn.hook_pattern.add_hook(stop_gradient, is_permanent=True)
            if not self.use_lorsa:
                block.attn.hook_pattern.add_hook(stop_gradient, is_permanent=True)
            block.ln1.hook_scale.add_hook(stop_gradient, is_permanent=True)
            block.ln2.hook_scale.add_hook(stop_gradient, is_permanent=True)
            if hasattr(block, "ln1_post"):
                block.ln1_post.hook_scale.add_hook(stop_gradient, is_permanent=True)
            if hasattr(block, "ln2_post"):
                block.ln2_post.hook_scale.add_hook(stop_gradient, is_permanent=True)
        self.ln_final.hook_scale.add_hook(stop_gradient, is_permanent=True)

        for param in self.parameters():
            param.requires_grad = False

        for param in self._get_requires_grad_bias_params():
            param[1].requires_grad = True

        def enable_gradient(acts, hook):
            acts.requires_grad = True
            return acts

        self.hook_embed.add_hook(enable_gradient, is_permanent=True)

    def _configure_skip_connection(self, block, layer):
        def add_skip_connection(
            acts: torch.Tensor, hook: HookPoint, grad_hook: HookPoint, replacement_bias: torch.Tensor
        ):
            # We add grad_hook because we need a way to hook into the gradients of the output
            # of this function. If we put the backwards hook here at hook, the grads will be 0
            # because we detached acts.
            assert replacement_bias.requires_grad, "Replacement bias must be a parameter"
            return grad_hook((acts - replacement_bias).detach() + replacement_bias)

        # add mlp output hook and special grad hook
        output_hook_parts = self.original_mlp_output_hook.split(".")
        subblock = block
        for part in output_hook_parts:
            subblock = getattr(subblock, part)
        subblock.hook_out_grad = HookPoint()

        subblock.add_hook(
            partial(
                add_skip_connection,
                grad_hook=subblock.hook_out_grad,
                replacement_bias=self.transcoders.b_D[layer],
            ),
            is_permanent=True,
        )

        if self.use_lorsa:
            # add attn output hook and special grad hook
            output_hook_parts = self.original_attn_output_hook.split(".")
            subblock = block
            for part in output_hook_parts:
                subblock = getattr(subblock, part)
            subblock.hook_out_grad = HookPoint()
            subblock.add_hook(
                partial(
                    add_skip_connection,
                    grad_hook=subblock.hook_out_grad,
                    replacement_bias=self.lorsas[layer].b_D,
                ),
                is_permanent=True,
            )

    def _deduplicate_attention_buffers(self):
        """
        Share attention buffers across layers to save memory.

        TransformerLens makes separate copies of the same masks and RoPE
        embeddings for each layer - This just keeps one copy
        of each and shares it across all layers.
        """

        attn_masks = {}
        if not self.use_lorsa:
            for block in self.blocks:
                attn_masks[block.attn.attn_type] = block.attn.mask  # type: ignore
                if hasattr(block.attn, "rotary_sin"):
                    attn_masks["rotary_sin"] = block.attn.rotary_sin  # type: ignore
                    attn_masks["rotary_cos"] = block.attn.rotary_cos  # type: ignore

            for block in self.blocks:
                block.attn.mask = attn_masks[block.attn.attn_type]  # type: ignore
                if hasattr(block.attn, "rotary_sin"):
                    block.attn.rotary_sin = attn_masks["rotary_sin"]  # type: ignore
                    block.attn.rotary_cos = attn_masks["rotary_cos"]  # type: ignore
        else:
            for block in self.blocks:
                attn_masks[block.attn.old_attn.attn_type] = block.attn.old_attn.mask
                attn_masks["rotary_sin"] = block.attn.old_attn.rotary_sin
                attn_masks["rotary_cos"] = block.attn.old_attn.rotary_cos

            for block in self.blocks:
                block.attn.old_attn.mask = attn_masks[block.attn.old_attn.attn_type]
                block.attn.old_attn.rotary_sin = attn_masks["rotary_sin"]
                block.attn.old_attn.rotary_cos = attn_masks["rotary_cos"]

            lorsa_masks = {}
            for lorsa in self.lorsas:
                lorsa_masks["causal_mask"] = lorsa.mask
                lorsa_masks["rotary_sin"] = lorsa.rotary_sin
                lorsa_masks["rotary_cos"] = lorsa.rotary_cos

            for lorsa in self.lorsas:
                lorsa.mask = lorsa_masks["causal_mask"]
                lorsa.rotary_sin = lorsa_masks["rotary_sin"]
                lorsa.rotary_cos = lorsa_masks["rotary_cos"]

    def _get_activation_caching_hooks(
        self,
        zero_bos: bool = False,
        sparse: bool = False,
        apply_activation_function: bool = True,
    ) -> Tuple[List, List[Tuple[str, Callable]]]:
        if self.use_lorsa:
            activation_matrix = [None] * self.cfg.n_layers * 2
            lorsa_attention_score = [None] * self.cfg.n_layers
            lorsa_attention_pattern = [None] * self.cfg.n_layers
            z_attention_pattern = [None] * self.cfg.n_layers
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
                z_pattern = self.lorsas[layer].encode_z_patterns(acts)

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
                lorsa_attention_score[layer] = encode_result[-1]
                lorsa_attention_pattern[layer] = pattern
                z_attention_pattern[layer] = z_pattern.detach()

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
            lorsa_attention_score = None
            lorsa_attention_pattern = None
            z_attention_pattern = None

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
            lorsa_attention_score,
            lorsa_attention_pattern,
            z_attention_pattern,
            activation_hooks,
        )

    def get_activations(
        self,
        inputs: Union[str, torch.Tensor],
        sparse: bool = False,
        zero_bos: bool = True,
        apply_activation_function: bool = True,
        use_lorsa: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the transcoder activations for a given prompt

        Args:
            inputs (Union[str, torch.Tensor]): The inputs you want to get activations over
            sparse (bool, optional): Whether to return a sparse tensor of activations.
                Useful if d_transcoder is large. Defaults to False.
            zero_bos (bool, optional): Whether to zero out activations / errors at the 0th
                position (<BOS>). Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the model logits on the inputs and the
                associated activation cache
        """

        (activation_cache, lorsa_attention_score, lorsa_attention_pattern, _, activation_hooks) = (
            self._get_activation_caching_hooks(
                sparse=sparse,
                zero_bos=zero_bos,
                apply_activation_function=apply_activation_function,
            )
        )
        with torch.inference_mode(), self.hooks(activation_hooks):
            logits = self(inputs)
        activation_cache = torch.stack(activation_cache)
        if use_lorsa:
            lorsa_attention_pattern = torch.stack(lorsa_attention_pattern)
        if sparse:
            activation_cache = activation_cache.coalesce()
        return logits, activation_cache, lorsa_attention_pattern

    @contextmanager
    def zero_softcap(self):
        current_softcap = self.cfg.output_logits_soft_cap
        try:
            self.cfg.output_logits_soft_cap = 0.0
            yield
        finally:
            self.cfg.output_logits_soft_cap = current_softcap

    def maybe_zero_bos(self, tokens: torch.Tensor) -> torch.Tensor:
        special_tokens = []
        for special_token in self.tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                special_tokens.extend(special_token)
            else:
                special_tokens.append(special_token)
        special_token_ids = self.tokenizer.convert_tokens_to_ids(special_tokens)
        return tokens[0].cpu().item() in special_token_ids

    @torch.no_grad()
    def setup_attribution(
        self,
        inputs: Union[str, torch.Tensor],
        sparse: bool = False,
        zero_bos: bool = True,
    ):
        """Precomputes the transcoder / lorsa activations and error vectors, saving them and the
        token embeddings.

        Args:
            inputs (str): the inputs to attribute - hard coded to be a single string (no
                batching) for now
            sparse (bool): whether to return activations as a sparse tensor or not
            zero_bos (bool): whether to zero out the activations and error vectors at the
                bos position
        """

        tokens = ensure_tokenized(inputs, self.tokenizer)
        zero_bos = zero_bos and self.maybe_zero_bos(tokens)

        # cache activations and MLP in
        (activation_matrix, lorsa_attention_score, lorsa_attention_pattern, z_attention_pattern, activation_hooks) = (
            self._get_activation_caching_hooks(sparse=sparse, zero_bos=zero_bos)
        )

        if self.use_lorsa:
            attn_out_cache, attn_out_caching_hooks, _ = self.get_caching_hooks(
                lambda name: self.attn_output_hook in name
            )
        else:
            attn_out_cache, attn_out_caching_hooks = None, None

        mlp_out_cache, mlp_out_caching_hooks, _ = self.get_caching_hooks(lambda name: self.mlp_output_hook in name)

        if self.use_lorsa:
            error_vectors = torch.zeros(
                [self.cfg.n_layers * 2, len(tokens), self.cfg.d_model],
                device=self.cfg.device,
                dtype=self.cfg.dtype,
            )
        else:
            error_vectors = torch.zeros(
                [self.cfg.n_layers, len(tokens), self.cfg.d_model],
                device=self.cfg.device,
                dtype=self.cfg.dtype,
            )

        # note: activation_hooks must come before error_hooks
        if self.use_lorsa:
            logits = self.run_with_hooks(
                tokens, fwd_hooks=(activation_hooks + attn_out_caching_hooks + mlp_out_caching_hooks)
            )
        else:
            logits = self.run_with_hooks(tokens, fwd_hooks=(activation_hooks + mlp_out_caching_hooks))

        if self.use_lorsa:
            lorsa_activation_matrix = activation_matrix[: self.cfg.n_layers]
            clt_activation_matrix = activation_matrix[self.cfg.n_layers :]
        else:
            lorsa_activation_matrix = None
            clt_activation_matrix = activation_matrix

        if self.use_lorsa:
            lorsa_reconstruction = torch.stack(
                [self.lorsas[layer].decode(lorsa_activation_matrix[layer]) for layer in range(self.cfg.n_layers)]
            )

        clt_reconstruction = self.transcoders.decode(clt_activation_matrix)

        if self.use_lorsa:
            error_vectors[: self.cfg.n_layers] = torch.cat(list(attn_out_cache.values()), dim=0) - lorsa_reconstruction
            error_vectors[self.cfg.n_layers :] = torch.cat(list(mlp_out_cache.values()), dim=0) - clt_reconstruction
        else:
            error_vectors = torch.cat(list(mlp_out_cache.values()), dim=0) - clt_reconstruction

        if zero_bos:
            error_vectors[:, 0] = 0

        if self.use_lorsa:
            lorsa_activation_matrix = torch.stack(lorsa_activation_matrix)
            lorsa_attention_pattern = torch.stack(lorsa_attention_pattern)
            z_attention_pattern = torch.stack(z_attention_pattern)
        clt_activation_matrix = torch.stack(clt_activation_matrix)
        if sparse:
            if self.use_lorsa:
                lorsa_activation_matrix = lorsa_activation_matrix.coalesce()
            clt_activation_matrix = clt_activation_matrix.coalesce()

        token_vectors = self.W_E[tokens].detach()  # (n_pos, d_model)
        return (
            logits,
            lorsa_activation_matrix,
            lorsa_attention_score,
            lorsa_attention_pattern,
            z_attention_pattern,
            clt_activation_matrix,
            error_vectors,
            token_vectors,
        )

    def setup_intervention_with_freeze(
        self,
        inputs: str | torch.Tensor,
        constrained_layers: range | None = None,
        use_lorsa: bool = True,
    ) -> tuple[torch.Tensor, list[tuple[str, Callable]]]:
        """Sets up an intervention with either frozen attention + LayerNorm(default) or frozen
        attention, LayerNorm, and MLPs, for constrained layers

        Args:
            inputs (Union[str, torch.Tensor]): The inputs to intervene on
            constrained_layers (range | None): whether to apply interventions only to a certain
                range. Mostly applicable to CLTs. If the given range includes all model layers,
                we also freeze layernorm denominators, computing direct effects. None means no
                constraints (iterative patching)

        Returns:
            list[tuple[str, Callable]]: The freeze hooks needed to run the desired intervention.
        """

        hookpoints_to_freeze = ["hook_pattern"]
        if constrained_layers:
            if set(range(self.cfg.n_layers)).issubset(set(constrained_layers)):
                hookpoints_to_freeze.append("hook_scale")
            hookpoints_to_freeze.append(self.mlp_input_hook)
            if use_lorsa:
                hookpoints_to_freeze.append(self.attn_input_hook)

        # only freeze outputs in constrained range
        selected_hook_points = []
        for hook_point, hook_obj in self.hook_dict.items():
            if any(hookpoint_to_freeze in hook_point for hookpoint_to_freeze in hookpoints_to_freeze):
                # don't freeze feature outputs if the layer is not in the constrained range
                if (
                    (self.mlp_input_hook in hook_point or (use_lorsa and self.attn_input_hook in hook_point))
                    and constrained_layers
                    and hook_obj.layer() not in constrained_layers
                ):
                    continue
                selected_hook_points.append(hook_point)

        freeze_cache, cache_hooks, _ = self.get_caching_hooks(names_filter=selected_hook_points)

        original_activations, _, _, _, activation_caching_hooks = self._get_activation_caching_hooks()
        self.run_with_hooks(inputs, fwd_hooks=cache_hooks + activation_caching_hooks)

        def freeze_hook(activations, hook):
            cached_values = freeze_cache[hook.name]

            assert activations.shape == cached_values.shape, (
                f"Activations shape {activations.shape} does not match cached values"
                f" shape {cached_values.shape} at hook {hook.name}"
            )
            return cached_values

        if use_lorsa:
            fwd_hooks = [
                (hookpoint, freeze_hook)
                for hookpoint in freeze_cache.keys()
                if self.mlp_input_hook not in hookpoint and self.attn_input_hook not in hookpoint
            ]
        else:
            fwd_hooks = [
                (hookpoint, freeze_hook) for hookpoint in freeze_cache.keys() if self.mlp_input_hook not in hookpoint
            ]

        return torch.stack(original_activations), fwd_hooks

    def _get_feature_intervention_hooks(
        self,
        inputs: str | torch.Tensor,
        interventions: list[Intervention],
        constrained_layers: range | None = None,
        apply_activation_function: bool = True,
        sparse: bool = False,
        using_past_kv_cache: bool = False,
        use_lorsa: bool = True,
    ):
        """Given the input, and a dictionary of features to intervene on, performs the
        intervention, allowing all effects to propagate (optionally allowing its effects to
        propagate through transcoders)

        Args:
            input (_type_): the input prompt to intervene on
            intervention_dict (List[Intervention]): A list of interventions to perform, formatted
                as a list of (layer, position, feature_idx, value)
            constrained_layers (range | None): whether to apply interventions only to a certain
                range, freezing all MLPs within the layer range before doing so. This is mostly
                applicable to CLTs. If the given range includes all model layers, we also freeze
                layernorm denominators, computing direct effects.nNone means no constraints
                (iterative patching)
            apply_activation_function (bool): whether to apply the activation function when
                recording the activations to be returned. This is useful to set to False for
                testing purposes, as attribution predicts the change in pre-activation
                feature values.
            sparse (bool): whether to sparsify the activations in the returned cache. Setting
                this to True will take up less memory, at the expense of slower interventions.
            using_past_kv_cache (bool): whether we are generating with past_kv_cache, meaning that
                n_pos is 1, and we must append onto the existing logit / activation cache if the
                hooks are run multiple times. Defaults to False
        """

        interventions_by_layer_mlp = defaultdict(list)
        interventions_by_layer_lorsa = defaultdict(list)
        for layer, pos, feature_idx, value, sae_type in interventions:
            if sae_type == "clt":
                interventions_by_layer_mlp[layer].append((pos, feature_idx, value))
            else:
                interventions_by_layer_lorsa[layer].append((pos, feature_idx, value))

        # We're generating one token at a time
        original_activations, freeze_hooks = self.setup_intervention_with_freeze(
            inputs,
            constrained_layers=constrained_layers,
            use_lorsa=use_lorsa,
        )
        n_pos = inputs.size(0)

        layer_deltas_mlp = torch.zeros(
            [self.cfg.n_layers, n_pos, self.cfg.d_model],
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )

        if use_lorsa:
            layer_deltas_lorsa = torch.zeros(
                [self.cfg.n_layers, n_pos, self.cfg.d_model],
                dtype=self.cfg.dtype,
                device=self.cfg.device,
            )

        # This activation cache will fill up during our forward intervention pass
        (activation_cache, lorsa_attention_score, lorsa_attention_pattern, _, activation_hooks) = (
            self._get_activation_caching_hooks(
                apply_activation_function=apply_activation_function,
                sparse=sparse,
            )
        )

        def calculate_delta_hook_mlp(activations, hook, layer: int, layer_interventions):
            if constrained_layers:
                # base deltas on original activations; don't let effects propagate
                if use_lorsa:
                    transcoder_activations = original_activations[self.cfg.n_layers :]
                else:
                    transcoder_activations = original_activations
                transcoder_activations = transcoder_activations.permute(1, 0, 2)
            for i in range(len(transcoder_activations)):
                if transcoder_activations[i] is None:
                    transcoder_activations[i] = torch.zeros_like(transcoder_activations[0])

            activation_deltas = transcoder_activations.clone().detach()
            for pos, feature_idx, value in layer_interventions:
                activation_deltas[pos, layer, feature_idx] = value

            # calculate delta value from the change of activation
            # Use unified decode interface
            if hasattr(self.transcoders, "decode"):
                # Both TranscoderSet and CrossLayerTranscoder have decode method
                reconstruct_new = self.transcoders.decode(activation_deltas)
                reconstruct_old = self.transcoders.decode(transcoder_activations)
                reconstruct = reconstruct_new - reconstruct_old
                layer_deltas_mlp[:, :, :] += reconstruct[:, :, :]
            else:
                # Fallback for dict of transcoders - decode each layer separately
                for layer_idx in range(self.cfg.n_layers):
                    if activation_deltas[layer_idx] is not None and transcoder_activations[layer_idx] is not None:
                        reconstruct_new = self.transcoders[layer_idx].decode(activation_deltas[layer_idx])
                        reconstruct_old = self.transcoders[layer_idx].decode(transcoder_activations[layer_idx])
                        reconstruct = reconstruct_new - reconstruct_old
                        layer_deltas_mlp[layer_idx] += reconstruct

        def calculate_delta_hook_lorsa(activations, hook, layer: int, layer_interventions):
            if constrained_layers:
                # base deltas on original activations; don't let effects propagate
                lorsa_activations = original_activations[layer]
            lorsa_activations = lorsa_activations.unsqueeze(0)
            activation_deltas = lorsa_activations.clone().detach()
            for pos, feature_idx, value in layer_interventions:
                activation_deltas[0, pos, feature_idx] = value

            # calculate delta value from the change of activation
            reconstruct_new = self.lorsas[layer].decode(activation_deltas)
            reconstruct_old = self.lorsas[layer].decode(lorsa_activations)
            reconstruct = reconstruct_new - reconstruct_old
            layer_deltas_lorsa[layer] += reconstruct[0]

        def intervention_hook_mlp(activations, hook, layer: int):
            new_acts = activations
            if layer in intervention_range:
                new_acts = new_acts + layer_deltas_mlp[layer]
            return new_acts

        def intervention_hook_lorsa(activations, hook, layer: int):
            new_acts = activations
            if layer in intervention_range:
                new_acts = new_acts + layer_deltas_lorsa[layer]
            return new_acts

        delta_hooks = [
            (
                f"blocks.{layer}.{self.mlp_output_hook}",
                partial(calculate_delta_hook_mlp, layer=layer, layer_interventions=layer_interventions),
            )
            for layer, layer_interventions in interventions_by_layer_mlp.items()
        ]
        if use_lorsa:
            delta_hooks = delta_hooks + [
                (
                    f"blocks.{layer}.{self.attn_output_hook}",
                    partial(calculate_delta_hook_lorsa, layer=layer, layer_interventions=layer_interventions),
                )
                for layer, layer_interventions in interventions_by_layer_lorsa.items()
            ]

        intervention_range = constrained_layers if constrained_layers else range(self.cfg.n_layers)
        intervention_hooks = [
            (f"blocks.{layer}.{self.mlp_output_hook}", partial(intervention_hook_mlp, layer=layer))
            for layer in range(self.cfg.n_layers)
        ]
        if use_lorsa:
            intervention_hooks = intervention_hooks + [
                (f"blocks.{layer}.{self.attn_output_hook}", partial(intervention_hook_lorsa, layer=layer))
                for layer in range(self.cfg.n_layers)
            ]

        all_hooks = freeze_hooks + activation_hooks + delta_hooks + intervention_hooks
        cached_logits = [] if using_past_kv_cache else [None]

        return all_hooks, cached_logits, activation_cache

    @torch.no_grad
    def feature_intervention(
        self,
        inputs: str | torch.Tensor,
        intervention: list[Intervention],
        constrained_layers: range | None = None,
        apply_activation_function: bool = True,
        sparse: bool = False,
        use_lorsa: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given the input, and a dictionary of features to intervene on, performs the
        intervention, and returns the logits and feature activations. If freeze_attention or
        constrained_layers is True, attention patterns will be frozen, along with MLPs and
        LayerNorms. If constrained_layers is set, the effects of intervention will not propagate
        through the constrained layers, and CLTs will write only to those layers. Otherwise, the
        effects of the intervention will propagate through transcoders / LayerNorms

        Args:
            input (_type_): the input prompt to intervene on
            interventions (list[tuple[int, Union[int, slice, torch.Tensor]], int,
                Union[int, torch.Tensor]]): A list of interventions to perform, formatted as
                a list of (layer, position, feature_idx, value)
            constrained_layers (range | None): whether to apply interventions only to a certain
                range. Mostly applicable to CLTs. If the given range includes all model layers,
                we also freeze layernorm denominators, computing direct effects. None means no
                constraints (iterative patching)
            freeze_attention (bool): whether to freeze all attention patterns an layernorms
            apply_activation_function (bool): whether to apply the activation function when
                recording the activations to be returned. This is useful to set to False for
                testing purposes, as attribution predicts the change in pre-activation
                feature values.
            sparse (bool): whether to sparsify the activations in the returned cache. Setting
                this to True will take up less memory, at the expense of slower interventions.
        """

        hooks, _, activation_cache = self._get_feature_intervention_hooks(
            inputs,
            intervention,
            constrained_layers=constrained_layers,
            apply_activation_function=apply_activation_function,
            sparse=sparse,
            use_lorsa=use_lorsa,
        )

        with self.hooks(hooks):  # type: ignore
            logits = self(inputs)

        activation_cache = torch.stack(activation_cache)

        return logits, activation_cache

    def _get_requires_grad_bias_params(self):
        bias_params = []
        for param in self.named_parameters():
            if (
                ".b" in param[0]
                and "b_Q" not in param[0]
                and "b_K" not in param[0]
                and "old" not in param[0]
                and "transcoders.b_E" not in param[0]
            ):
                bias_params.append(param)
            elif self.use_lorsa and "lorsas" in param[0] and ("b_Q" in param[0] or "b_K" in param[0]):
                bias_params.append(param)
        return bias_params
