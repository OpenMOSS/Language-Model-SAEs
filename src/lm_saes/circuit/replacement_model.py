from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Callable, List, Tuple, Union

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_saes.clt import CrossLayerTranscoder
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.config import LanguageModelConfig
from lm_saes.resource_loaders import load_model


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
    transcoders: CrossLayerTranscoder
    lorsas: nn.ModuleList
    mlp_input_hook: str
    mlp_output_hook: str
    attn_input_hook: str = None
    attn_output_hook: str = None

    @classmethod
    def from_config(
        cls,
        config: LanguageModelConfig,
        transcoders: CrossLayerTranscoder,
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
        transcoders: CrossLayerTranscoder,
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
                torch_dtype=model_cfg.dtype,
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
        transcoders: CrossLayerTranscoder,
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
        transcoders: CrossLayerTranscoder,
        lorsas: List[LowRankSparseAttention],
        mlp_input_hook: str,
        mlp_output_hook: str,
        attn_input_hook: str,
        attn_output_hook: str,
        use_lorsa: bool = True,
    ):
        self.use_lorsa = use_lorsa

        transcoders.to(self.cfg.device, self.cfg.dtype)
        if lorsas is not None:
            for lorsa in lorsas:
                assert not lorsa.cfg.skip_bos, "Lorsa must not skip bos, will be handled by replacement model"

        self.add_module("transcoders", transcoders)
        if use_lorsa:
            self.add_module("lorsas", nn.ModuleList(lorsas))

        self.d_transcoder = transcoders.cfg.d_sae
        if use_lorsa:
            self.d_lorsa = lorsas[0].cfg.d_sae

        self.mlp_input_hook = mlp_input_hook
        self.original_mlp_output_hook = mlp_output_hook
        self.mlp_output_hook = mlp_output_hook + ".hook_out_grad"

        if use_lorsa:
            self.attn_input_hook = attn_input_hook
            self.original_attn_output_hook = attn_output_hook
            self.attn_output_hook = attn_output_hook + ".hook_out_grad"

        for block in self.blocks:
            block.mlp = ReplacementMLP(block.mlp)
            if use_lorsa:
                block.attn = ReplacementAttention(block.attn)

        self.unembed = ReplacementUnembed(self.unembed)

        self._configure_gradient_flow()
        self._deduplicate_attention_buffers()
        self.setup()

    def _configure_gradient_flow(self):
        for layer in range(self.cfg.n_layers):
            self._configure_skip_connection(self.blocks[layer], layer)
            self.lorsas[layer]._configure_gradient_flow()

        def stop_gradient(acts, hook):
            return acts.detach()

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
            lorsa_attention_pattern = [None] * self.cfg.n_layers
        else:
            activation_matrix = [None] * self.cfg.n_layers
        activation_hooks = []

        if self.use_lorsa:

            def cache_activations_attn(acts, hook, layer, zero_bos):
                encode_result = self.lorsas[layer].encode(
                    acts, return_hidden_pre=not apply_activation_function, return_attention_pattern=True
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
                transcoder_acts = self.transcoders.encode_single_layer(
                    acts, layer, return_hidden_pre=not apply_activation_function
                )

                if not apply_activation_function:
                    transcoder_acts = transcoder_acts[1].detach().squeeze(0)
                else:
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
                transcoder_acts = self.transcoders.encode_single_layer(
                    acts, layer, return_hidden_pre=not apply_activation_function
                )

                if not apply_activation_function:
                    transcoder_acts = transcoder_acts[1].detach().squeeze(0)
                else:
                    transcoder_acts = transcoder_acts.detach().squeeze(0)

                if zero_bos:
                    transcoder_acts[0] = 0
                if sparse:
                    transcoder_acts = transcoder_acts.to_sparse()

                def mlp_offset(layer):
                    return layer

                activation_matrix[mlp_offset(layer)] = transcoder_acts
                # print('run')

            activation_hooks.extend(
                [
                    (
                        f"blocks.{layer}.{self.mlp_input_hook}",
                        partial(cache_activations_mlp, layer=layer, zero_bos=zero_bos),
                    )
                    for layer in range(self.cfg.n_layers)
                ]
            )

        return activation_matrix, lorsa_attention_pattern, activation_hooks

    def get_activations(
        self,
        inputs: Union[str, torch.Tensor],
        sparse: bool = False,
        zero_bos: bool = True,
        apply_activation_function: bool = True,
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

        activation_cache, lorsa_attention_pattern, activation_hooks = self._get_activation_caching_hooks(
            sparse=sparse,
            zero_bos=zero_bos,
            apply_activation_function=apply_activation_function,
        )
        with torch.inference_mode(), self.hooks(activation_hooks):
            logits = self(inputs)
        activation_cache = torch.stack(activation_cache)
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
        activation_matrix, lorsa_attention_pattern, activation_hooks = self._get_activation_caching_hooks(
            sparse=sparse, zero_bos=zero_bos
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
            print(len(activation_matrix))
            print(activation_matrix)

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
        clt_activation_matrix = torch.stack(clt_activation_matrix)
        if sparse:
            if self.use_lorsa:
                lorsa_activation_matrix = lorsa_activation_matrix.coalesce()
            clt_activation_matrix = clt_activation_matrix.coalesce()

        token_vectors = self.W_E[tokens].detach()  # (n_pos, d_model)
        return (
            logits,
            lorsa_activation_matrix,
            lorsa_attention_pattern,
            clt_activation_matrix,
            error_vectors,
            token_vectors,
        )

    def setup_intervention_with_freeze(
        self, inputs: Union[str, torch.Tensor], direct_effects: bool = False
    ) -> List[Tuple[str, Callable]]:
        """Sets up an intervention with either frozen attention (default) or frozen
        attention, LayerNorm, and MLPs, for direct effects

        Args:
            inputs (Union[str, torch.Tensor]): The inputs to intervene on
            direct_effects (bool, optional): Whether to freeze not just attention, but also
                LayerNorm and MLPs. Defaults to False.

        Returns:
            List[Tuple[str, Callable]]: The freeze hooks needed to run the desired intervention.
        """

        if direct_effects:
            hookpoints_to_freeze = ["hook_pattern", "hook_scale", self.feature_output_hook]
        else:
            hookpoints_to_freeze = ["hook_pattern"]

        freeze_cache, cache_hooks, _ = self.get_caching_hooks(
            names_filter=lambda name: any(hookpoint in name for hookpoint in hookpoints_to_freeze)
        )
        self.run_with_hooks(inputs, fwd_hooks=cache_hooks)

        def freeze_hook(activations, hook):
            cached_values = freeze_cache[hook.name]

            # if we're doing open-ended generation, the position dimensions won't match
            # so we'll just freeze the previous positions, and leave the new ones unfrozen
            if "hook_pattern" in hook.name and activations.shape[2:] != cached_values.shape[2:]:
                new_activations = activations.clone()
                new_activations[:, :, : cached_values.shape[2], : cached_values.shape[3]] = cached_values
                return new_activations

            elif ("hook_scale" in hook.name or self.feature_output_hook in hook.name) and activations.shape[
                1
            ] != cached_values.shape[1]:
                new_activations = activations.clone()
                new_activations[:, : cached_values.shape[1]] = cached_values
                return new_activations

            # if other positions don't match, that's no good
            assert activations.shape == cached_values.shape, (
                f"Activations shape {activations.shape} does not match cached values"
                f" shape {cached_values.shape} at hook {hook.name}"
            )
            return cached_values

        fwd_hooks = [
            (hookpoint, freeze_hook) for hookpoint in freeze_cache.keys() if self.feature_input_hook not in hookpoint
        ]

        if not direct_effects:
            return fwd_hooks

        return fwd_hooks

    def _get_feature_intervention_hooks(
        self,
        inputs: Union[str, torch.Tensor],
        interventions: List[Tuple[int, Union[int, slice, torch.Tensor], int, Union[int, torch.Tensor]]],
        direct_effects: bool = False,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
    ):
        """Given the input, and a dictionary of features to intervene on, performs the
        intervention, allowing all effects to propagate (optionally allowing its effects to
        propagate through transcoders)

        Args:
            input (_type_): the input prompt to intervene on
            intervention_dict (List[Tuple[int, Union[int, slice, torch.Tensor]], int,
                Union[int, torch.Tensor]]): A list of interventions to perform, formatted as
                a list of (layer, position, feature_idx, value)
            direct_effects (bool): whether to freeze all MLPs/transcoders / attn patterns /
                layernorm denominators
            apply_activation_function (bool): whether to apply the activation function when
                recording the activations to be returned. This is useful to set to False for
                testing purposes, as attribution predicts the change in pre-activation
                feature values.
        """

        interventions_by_layer = defaultdict(list)
        for layer, pos, feature_idx, value in interventions:
            interventions_by_layer[layer].append((pos, feature_idx, value))

        # This activation cache will fill up during our forward intervention pass
        activation_cache, lorsa_attention_pattern, activation_hooks = self._get_activation_caching_hooks(
            apply_activation_function=apply_activation_function
        )

        def intervention_hook(activations, hook, layer, layer_interventions):
            transcoder_activations = activation_cache[layer]
            if not apply_activation_function:
                transcoder_activations = (
                    self.transcoders[layer].activation_function(transcoder_activations.unsqueeze(0)).squeeze(0)
                )
            transcoder_output = self.transcoders[layer].decode(transcoder_activations)
            for pos, feature_idx, value in layer_interventions:
                transcoder_activations[pos, feature_idx] = value
            new_transcoder_output = self.transcoders[layer].decode(transcoder_activations)
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
            if freeze_attention or direct_effects
            else []
        )
        all_hooks += activation_hooks + intervention_hooks

        return all_hooks, activation_cache, lorsa_attention_pattern

    @torch.no_grad
    def feature_intervention(
        self,
        inputs: Union[str, torch.Tensor],
        interventions: List[Tuple[int, Union[int, slice, torch.Tensor], int, Union[int, torch.Tensor]]],
        direct_effects: bool = False,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given the input, and a dictionary of features to intervene on, performs the
        intervention, and returns the logits and feature activations. If direct_effects is
        True, attention patterns will be frozen, along with MLPs and LayerNorms. If it is
        False, the effects of the intervention will propagate through transcoders /
        LayerNorms

        Args:
            input (_type_): the input prompt to intervene on
            interventions (List[Tuple[int, Union[int, slice, torch.Tensor]], int,
                Union[int, torch.Tensor]]): A list of interventions to perform, formatted as
                a list of (layer, position, feature_idx, value)
            direct_effects (bool): whether to freeze all MLPs/transcoders / attn patterns /
                layernorm denominators
            apply_activation_function (bool): whether to apply the activation function when
                recording the activations to be returned. This is useful to set to False for
                testing purposes, as attribution predicts the change in pre-activation
                feature values.
        """

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
            elif self.use_lorsa is True and "lorsas" in param[0] and ("b_Q" in param[0] or "b_K" in param[0]):
                bias_params.append(param)
        return bias_params
