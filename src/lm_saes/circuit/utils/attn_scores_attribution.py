"""Utilities for computing attention score attributions."""

from dataclasses import dataclass, field

import torch

from ..replacement_model import ReplacementModel
from .attribution_utils import (
    select_scaled_decoder_vecs_lorsa,
    select_scaled_decoder_vecs_transcoder,
)


@dataclass
class QKTracingResults:
    """Results from QK tracing analysis.

    pair_wise_contributors: List of tuples containing (q_node, k_node, score) for top pairwise contributors.
    top_q_marginal_contributors: List of tuples containing (q_node, score) for top Q marginal contributors.
    top_k_marginal_contributors: List of tuples containing (k_node, score) for top K marginal contributors.
    """

    pair_wise_contributors: list[tuple]
    top_q_marginal_contributors: list[tuple]
    top_k_marginal_contributors: list[tuple]


@dataclass
class ResStreamComponents:
    lorsa_activation_matrix: torch.Tensor
    transcoder_activation_matrix: torch.Tensor
    components: torch.Tensor
    n_layers: int
    pos: int
    bias_names: list[str] = field(default_factory=list)

    def sum(self):
        return self.components.sum(0)

    def update_components(self, components):
        self.components = components
        return self

    def append(self, new_component, bias_name=None):
        if new_component.ndim == 1:
            new_component = new_component[None, :]
        new_res_stream_components = ResStreamComponents(
            lorsa_activation_matrix=self.lorsa_activation_matrix,
            transcoder_activation_matrix=self.transcoder_activation_matrix,
            bias_names=self.bias_names,
            n_layers=self.n_layers,
            pos=self.pos,
            components=torch.cat(
                [
                    self.components,
                    new_component,
                ],
                dim=0,
            ),
        )
        if bias_name is not None:
            new_res_stream_components.bias_names.append(bias_name)
        return new_res_stream_components

    def drop_bias_terms(self):
        # only do this to k side as it contributes equally to all attn scores
        # WARNING: this is not strictly safe as layernorm scale might be different across token positions
        # this may lead to bias terms slightly different (i.e. multiplied by different scales)
        # We choose to still apply this. We assume this wont matter a lot
        self.components = self.components[: -len(self.bias_names) - 2 * self.n_layers]
        assert self.components.size(0) == sum(
            [
                self.lorsa_activation_matrix._nnz(),  # lorsa features
                self.transcoder_activation_matrix._nnz(),  # transcoder features
                2 * self.n_layers,  # errors
                1,  # token
            ]
        ), "We might not want to drop bias terms for the second time."
        return self

    def map_idx_to_nodes(self, indices: torch.Tensor, input_ids: torch.Tensor):
        """Map component indices to node names.

        Args:
            indices: Tensor of component indices to map.
            input_ids: Input token IDs for generating node names.

        Returns:
            List of node name strings.
        """
        results = []
        lorsa_end_idx = self.lorsa_activation_matrix._nnz() + 1
        transcoder_end_idx = lorsa_end_idx + self.transcoder_activation_matrix._nnz()
        error_end_idx = transcoder_end_idx + 2 * self.n_layers
        decoder_bias_idx = error_end_idx + 2 * self.n_layers
        for idx in indices.cpu().tolist():
            if idx == 0:
                res = f"E_{input_ids[self.pos]}_{self.pos}"
            elif idx < lorsa_end_idx:
                lorsa_idx = idx - 1
                layer, pos, feature_idx = self.lorsa_activation_matrix.coalesce().indices()[:, lorsa_idx]
                res = f"{layer}_{feature_idx}_{pos}"
            elif idx < transcoder_end_idx:
                tc_idx = idx - lorsa_end_idx
                layer, pos, feature_idx = self.transcoder_activation_matrix.coalesce().indices()[:, tc_idx]
                res = f"{layer}_{feature_idx}_{pos}"
            elif idx < error_end_idx:
                res = f"{idx - transcoder_end_idx}_error_{self.pos}"
            elif idx < decoder_bias_idx:
                res = f"{idx - error_end_idx}_decoderbias_{self.pos}"
            else:
                res = self.bias_names[idx - decoder_bias_idx]
            results.append(res)
        return results


@torch.no_grad()
def get_residual_stream_components_single_module(
    activation_matrix: torch.Tensor,
    interested_pos: int,
    error_vecs: torch.Tensor,
    use_lorsa: bool,
    layer: int | torch.Tensor,
    model: ReplacementModel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get residual stream components for a single module (Lorsa or CLT).

    Args:
        activation_matrix: Sparse activation matrix.
        interested_pos: Position of interest.
        error_vecs: Error vectors from the model.
        use_lorsa: Whether to use Lorsa (True) or CLT (False).
        layer: Target layer index (int or scalar tensor).
        model: The replacement model.

    Returns:
        Tuple of (single_side_activation_matrix, feature_contribution_to_res_stream,
                  replacement_model_error_vecs, replacement_model_decoder_biases).
    """
    layer_int = int(layer.item() if isinstance(layer, torch.Tensor) else layer)
    layer_mask = activation_matrix.indices()[0] < layer_int
    pos_mask = activation_matrix.indices()[1] == interested_pos
    feature_mask = layer_mask & pos_mask

    single_side_activation_matrix = torch.sparse_coo_tensor(
        activation_matrix.indices()[:, feature_mask], activation_matrix.values()[feature_mask]
    )  # type: ignore
    decoder_vec_select_fn = select_scaled_decoder_vecs_lorsa if use_lorsa else select_scaled_decoder_vecs_transcoder
    feature_contribution_to_res_stream = decoder_vec_select_fn(
        single_side_activation_matrix.coalesce(), model.lorsas if use_lorsa else model.transcoders
    )

    replacement_model = model.lorsas if use_lorsa else model.transcoders
    replacement_model_decoder_biases = (
        torch.stack([replacement_model[i].b_D for i in range(layer_int)])
        if layer_int > 0
        else torch.zeros(0, replacement_model[0].b_D.shape[0], device=error_vecs.device)
    )

    replacement_model_error_vecs = error_vecs[: model.cfg.n_layers] if use_lorsa else error_vecs[model.cfg.n_layers :]
    replacement_model_error_vecs = replacement_model_error_vecs[
        :layer_int, interested_pos
    ]  # n_layer_below_target_layer d_model

    return (
        single_side_activation_matrix,
        feature_contribution_to_res_stream,
        replacement_model_error_vecs,
        replacement_model_decoder_biases,
    )


@torch.no_grad()
def get_residual_stream_components(
    lorsa_activation_matrix: torch.Tensor,
    clt_activation_matrix: torch.Tensor,
    pos: int,
    error_vecs: torch.Tensor,
    token_vecs: torch.Tensor,
    layer: int | torch.Tensor,
    model: ReplacementModel,
) -> ResStreamComponents:
    """Get residual stream components combining Lorsa and CLT contributions.

    Args:
        lorsa_activation_matrix: Lorsa activation matrix.
        clt_activation_matrix: CLT activation matrix.
        pos: Position of interest.
        error_vecs: Error vectors from the model.
        token_vecs: Token embedding vectors.
        layer: Target layer index (int or scalar tensor).
        model: The replacement model.

    Returns:
        ResStreamComponents containing all residual stream components.
    """
    layer_int = int(layer.item() if isinstance(layer, torch.Tensor) else layer)
    lorsa_components = get_residual_stream_components_single_module(
        lorsa_activation_matrix, pos, error_vecs, True, layer, model
    )
    clt_components = get_residual_stream_components_single_module(
        clt_activation_matrix, pos, error_vecs, False, layer, model
    )
    # must follow the order of tokenvecs, lorsa/clt features, lorsa/clt errors, lorsa/clt decoder biases
    # ln bias and qk bias are added later
    l = [
        token_vecs[pos : pos + 1],
        lorsa_components[1],
        clt_components[1],
        lorsa_components[2],
        clt_components[2],
        lorsa_components[3],
        clt_components[3],
    ]

    return ResStreamComponents(
        lorsa_activation_matrix=lorsa_components[0],
        transcoder_activation_matrix=clt_components[0],
        n_layers=layer_int,
        pos=pos,
        components=torch.cat(l, dim=0),
    )


def probe_linear_equivalent_for_ln(components: ResStreamComponents, ln: torch.nn.Module) -> ResStreamComponents:
    """Probe the linear equivalent of layer normalization.

    We use a small trick to "probe" the linear equivalent of layernorms given a res stream input.
    A ln is only linear in the sense of fixing the denominator (i.e. hook_scale in TL).
    This is already done in replacement model _configure_gradient_flow.
    We do this so we do not need to care about the actual implementation (i.e. RMS or LN).

    Args:
        components: Residual stream components.
        ln: Layer normalization module.

    Returns:
        Updated ResStreamComponents with linearized layer norm applied.
    """
    # a ln is only linear in the sense of fixing the denominator (i.e. hook_scale in TL)
    # this is already done in replacement model _configure_gradient_flow
    # we do this so we do not need to care about the actual implementation (i.e. RMS or LN)
    original_input_to_ln = components.sum().detach().clone()
    original_input_to_ln.requires_grad_()
    ln_out = ln(original_input_to_ln)
    probe_grads = torch.eye(ln_out.size(-1), device=ln_out.device)
    W_list = []
    for k in range(ln_out.size(-1)):
        b = ln(original_input_to_ln)
        b.backward(gradient=probe_grads[k])
        W_list.append(original_input_to_ln.grad)
        original_input_to_ln.grad = None
    W_recovered = torch.stack(W_list)
    b_recovered = ln_out - W_recovered @ original_input_to_ln
    post_ln_components = components.components @ W_recovered.T
    assert torch.allclose(post_ln_components.sum(0) + b_recovered, ln_out, atol=1e-4)
    components.update_components(post_ln_components)
    return components.append(b_recovered, bias_name="b_ln1")


def get_single_side_QK_components(
    lorsa_activation_matrix: torch.Tensor,
    clt_activation_matrix: torch.Tensor,
    pos: int,
    error_vecs: torch.Tensor,
    token_vecs: torch.Tensor,
    qk_idx: int | torch.Tensor,
    layer: int | torch.Tensor,
    model: ReplacementModel,
    q_side: bool = True,
) -> ResStreamComponents:
    """Get Q or K side components for attention score computation.

    Args:
        lorsa_activation_matrix: LoRSA activation matrix.
        clt_activation_matrix: CLT activation matrix.
        pos: Position of interest.
        error_vecs: Error vectors from the model.
        token_vecs: Token embedding vectors.
        qk_idx: QK head index (int or scalar tensor).
        layer: Target layer index (int or scalar tensor).
        model: The replacement model.
        q_side: Whether to compute Q side (True) or K side (False).

    Returns:
        ResStreamComponents for Q or K side.
    """
    layer_int = int(layer.item() if isinstance(layer, torch.Tensor) else layer)
    qk_idx_int = int(qk_idx.item() if isinstance(qk_idx, torch.Tensor) else qk_idx)
    components = get_residual_stream_components(
        lorsa_activation_matrix,
        clt_activation_matrix,
        pos,
        error_vecs,
        token_vecs,
        layer,
        model,
    )
    components = probe_linear_equivalent_for_ln(components, model.blocks[layer_int].ln1)
    with torch.no_grad():
        lorsa = model.lorsas[layer_int]
        W_qk = lorsa.W_Q[qk_idx_int] if q_side else lorsa.W_K[qk_idx_int]
        b_qk = lorsa.b_Q[qk_idx_int] if q_side else lorsa.b_K[qk_idx_int]
        q_or_k = torch.einsum(
            "bd,dq->bq",
            components.components,
            W_qk,
        )
        components = components.update_components(q_or_k)
        components = components.append(b_qk, bias_name="b_q" if q_side else "b_k")
        components = components.update_components(components.components / (lorsa.cfg.attn_scale**0.5))

        if lorsa.cfg.use_post_qk_ln:
            ln = lorsa.ln_q if q_side else lorsa.ln_k
            components = probe_linear_equivalent_for_ln(components, ln)

        # apply_rotary only works in 4-d.
        # have to expand to 4-d first and put the interested components to the right pos
        components = components.update_components(
            lorsa._apply_rotary(
                components.components[:, None, None, :].expand(-1, pos, -1, -1),
            )[:, -1, 0, :]
        )

        return components


def extract_QK_tracing_result(
    q_side: ResStreamComponents,
    k_side: ResStreamComponents,
    input_ids: torch.Tensor,
    topk: int = 10,
) -> QKTracingResults:
    """Extract QK tracing results from Q and K side components.

    Args:
        q_side: Q side residual stream components.
        k_side: K side residual stream components.
        input_ids: Input token IDs for node name generation.
        topk: Number of top contributors to return.

    Returns:
        QKTracingResults containing pairwise and marginal contributors.
    """
    attr_matrix = q_side.components @ k_side.components.T
    topk_pairwise_attr_entries = attr_matrix.flatten().topk(min(topk, attr_matrix.numel()))
    topk_pairwise_attr_indices = topk_pairwise_attr_entries.indices
    q_features, k_features = (
        topk_pairwise_attr_indices // attr_matrix.size(1),
        topk_pairwise_attr_indices % attr_matrix.size(1),
    )
    # pair-wise top contributors
    pair_wise_contributors = list(
        zip(
            q_side.map_idx_to_nodes(q_features, input_ids),
            k_side.map_idx_to_nodes(k_features, input_ids),
            topk_pairwise_attr_entries.values.cpu().tolist(),
        )
    )
    top_q_marginal_contributors, top_k_marginal_contributors = (
        attr_matrix.sum(1).topk(min(topk, attr_matrix.size(0))),
        attr_matrix.sum(0).topk(min(topk, attr_matrix.size(1))),
    )
    top_q_marginal_contributors = list(
        zip(
            q_side.map_idx_to_nodes(top_q_marginal_contributors.indices, input_ids),
            top_q_marginal_contributors.values.cpu().tolist(),
        )
    )
    top_k_marginal_contributors = list(
        zip(
            k_side.map_idx_to_nodes(top_k_marginal_contributors.indices, input_ids),
            top_k_marginal_contributors.values.cpu().tolist(),
        )
    )
    return QKTracingResults(
        pair_wise_contributors=pair_wise_contributors,
        top_q_marginal_contributors=top_q_marginal_contributors,
        top_k_marginal_contributors=top_k_marginal_contributors,
    )


def compute_attn_scores_attribution(
    model: ReplacementModel,
    lorsa_activation_matrix,
    clt_activation_matrix,
    layer: torch.Tensor,
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    qk_idx: torch.Tensor,
    token_vecs: torch.Tensor,
    error_vecs: torch.Tensor,
    input_ids: torch.Tensor,
    topk: int = 10,
) -> QKTracingResults:
    """Compute attention score attribution for QK attention patterns.

    Args:
        model: The replacement model used for computation.
        lorsa_activation_matrix: Activation matrix for Lorsa features.
        clt_activation_matrix: Activation matrix for cross-layer transcoder features.
        layer: Target layer index.
        q_pos: Query position indices.
        k_pos: Key position indices.
        qk_idx: QK head index.
        token_vecs: Token embedding vectors.
        error_vecs: Error vectors from the model.
        input_ids: Input token IDs.
        topk: Number of top contributors to return.

    Returns:
        QKTracingResults containing pairwise and marginal contributors.
    """
    assert q_pos > 0, "q_pos=0 should not appear in interested heads"
    if k_pos == 0:
        # we are currently not instrested why a token attend to bos
        return QKTracingResults(
            pair_wise_contributors=[],
            top_q_marginal_contributors=[],
            top_k_marginal_contributors=[],
        )
    q_side = get_single_side_QK_components(
        lorsa_activation_matrix, clt_activation_matrix, q_pos, error_vecs, token_vecs, qk_idx, layer, model, q_side=True
    )
    k_side = get_single_side_QK_components(
        lorsa_activation_matrix,
        clt_activation_matrix,
        k_pos,
        error_vecs,
        token_vecs,
        qk_idx,
        layer,
        model,
        q_side=False,
    ).drop_bias_terms()

    return extract_QK_tracing_result(q_side, k_side, input_ids, topk=topk)
