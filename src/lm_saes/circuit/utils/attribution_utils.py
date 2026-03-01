"""Utility functions for attribution computation."""

from typing import List, Tuple, Union

import torch

from lm_saes.clt import CrossLayerTranscoder
from lm_saes.lorsa import LowRankSparseAttention

from .transcoder_set import TranscoderSet

# Type definition for transcoders: per-layer (dict) or cross-layer (CLT)
TranscoderType = TranscoderSet | CrossLayerTranscoder


@torch.no_grad()
def compute_salient_logits(
    logits: torch.Tensor,
    unembed_proj: torch.Tensor,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick the smallest logit set whose cumulative prob >= *desired_logit_prob*.

    Args:
        logits: ``(d_vocab,)`` vector (single position).
        unembed_proj: ``(d_model, d_vocab)`` unembedding matrix.
        max_n_logits: Hard cap *k*.
        desired_logit_prob: Cumulative probability threshold *p*.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            * logit_indices - ``(k,)`` vocabulary ids.
            * logit_probs   - ``(k,)`` softmax probabilities.
            * demeaned_vecs - ``(k, d_model)`` unembedding columns, demeaned.
    """

    probs = torch.softmax(logits, dim=-1)
    top_p, top_idx = torch.topk(probs, max_n_logits)
    cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
    top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    cols = unembed_proj[:, top_idx]
    demeaned = cols - unembed_proj.mean(dim=-1, keepdim=True)
    return top_idx, top_p, demeaned.T


@torch.no_grad()
def select_scaled_decoder_vecs_transcoder(
    activations: torch.sparse.Tensor, transcoders: TranscoderType
) -> torch.Tensor:
    """Return decoder rows for **active** features only.

    activations: [layer, context, d_sae] in coo format.

    The return value is already scaled by the feature activation, making it
    suitable as ``inject_values`` during gradient overrides.
    """
    if activations._nnz() == 0:
        return torch.zeros(0, transcoders.W_D.shape[2], device=activations.device)
    if isinstance(transcoders, TranscoderSet):
        rows: List[torch.Tensor] = [
            transcoders.W_D[layer, row.coalesce().indices()[1]] for layer, row in enumerate(activations)
        ]
        return torch.cat(rows) * activations.values()[:, None]
    else:  # CLT
        rows: List[torch.Tensor] = []
        feature_act_rows = [activations[layer_from].coalesce() for layer_from in range(transcoders.cfg.n_layers)]
        for layer_to in range(transcoders.cfg.n_layers):
            for layer_from in range(layer_to + 1):
                _, feat_idx = feature_act_rows[layer_from].indices()
                rows.append(
                    transcoders.W_D[layer_to][layer_from, feat_idx] * feature_act_rows[layer_from].values()[:, None]
                )
        return torch.cat(rows)


@torch.no_grad()
def select_scaled_decoder_vecs_lorsa(activations: torch.sparse.Tensor, lorsas: LowRankSparseAttention) -> torch.Tensor:
    """Return encoder rows for **active** features only."""
    if activations._nnz() == 0:
        return torch.zeros(0, lorsas[0].W_O.shape[1], device=activations.device)
    rows: List[torch.Tensor] = [lorsas[layer].W_O[row.coalesce().indices()[1]] for layer, row in enumerate(activations)]
    return torch.cat(rows) * activations.values()[:, None]


@torch.no_grad()
def select_encoder_rows(activations: torch.sparse.Tensor, transcoders: TranscoderType) -> torch.Tensor:
    """Return encoder rows for **active** features only."""
    rows: List[torch.Tensor] = []
    for layer, row in enumerate(activations):
        _, feat_idx = row.coalesce().indices()
        rows.append(transcoders.W_E[layer].T[feat_idx])
    return torch.cat(rows)


@torch.no_grad()
def select_encoder_rows_lorsa(
    activation_matrix: torch.sparse.Tensor,
    attention_pattern: torch.Tensor,
    z_attention_pattern: torch.Tensor,
    lorsas: LowRankSparseAttention,
) -> torch.Tensor:
    """Return encoder rows for **active** features only."""
    rows: List[torch.Tensor] = []
    patterns: List[torch.Tensor] = []
    z_patterns: List[torch.Tensor] = []
    for layer, row in enumerate(activation_matrix):
        qpos, head_idx = row.coalesce().indices()

        qk_idx = head_idx // (lorsas[layer].cfg.n_ov_heads // lorsas[layer].cfg.n_qk_heads)
        pattern = attention_pattern[layer, qk_idx, qpos]
        patterns.append(pattern)

        z_pattern = z_attention_pattern[layer, head_idx, qpos]
        z_patterns.append(z_pattern)

        rows.append(lorsas[layer].W_V[head_idx])
    return torch.cat(rows), torch.cat(patterns), torch.cat(z_patterns)


def compute_partial_influences(edge_matrix, logit_p, row_to_node_index, max_iter=128, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalized_matrix = torch.empty_like(edge_matrix, device=device).copy_(edge_matrix)
    normalized_matrix = normalized_matrix.abs_()
    normalized_matrix /= normalized_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)

    influences = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod[-len(logit_p) :] = logit_p

    for _ in range(max_iter):
        prod = prod[row_to_node_index] @ normalized_matrix
        if not prod.any():
            break
        influences += prod
    else:
        raise RuntimeError("Failed to converge")

    return influences


@torch.no_grad()
def select_feature_activations(
    features_to_trace: List[Tuple[int, int, int, bool]],
    lorsa_activation_matrix: torch.sparse.Tensor,
    clt_activation_matrix: torch.sparse.Tensor,
) -> torch.Tensor:
    """Return activation values for specified features.

    Args:
        features_to_trace: List of (layer, feature_idx, pos, is_lorsa) tuples.
        lorsa_activation_matrix: Sparse tensor for LORSA activations.
        clt_activation_matrix: Sparse tensor for CLT activations.

    Returns:
        torch.Tensor: Activation values for the specified features.
    """
    activations = []
    for layer, feature_idx, pos, is_lorsa in features_to_trace:
        if is_lorsa:
            # Find activation in lorsa matrix
            indices = lorsa_activation_matrix.indices()
            values = lorsa_activation_matrix.values()
            mask = (indices[0] == layer) & (indices[1] == pos) & (indices[2] == feature_idx)
            if mask.any():
                activations.append(values[mask][0])
            else:
                activations.append(torch.tensor(0.0, device=values.device, dtype=values.dtype))
        else:
            # Find activation in clt matrix
            indices = clt_activation_matrix.indices()
            values = clt_activation_matrix.values()
            mask = (indices[0] == layer) & (indices[1] == pos) & (indices[2] == feature_idx)
            if mask.any():
                activations.append(values[mask][0])
            else:
                activations.append(torch.tensor(0.0, device=values.device, dtype=values.dtype))
    return torch.stack(activations)


def ensure_tokenized(prompt: Union[str, torch.Tensor, List[int]], tokenizer) -> torch.Tensor:
    """Convert *prompt* → 1-D tensor of token ids (no batch dim)."""

    if isinstance(prompt, str):
        return tokenizer(prompt, return_tensors="pt").input_ids[0]
    if isinstance(prompt, torch.Tensor):
        return prompt.squeeze(0) if prompt.ndim == 2 else prompt
    if isinstance(prompt, list):
        return torch.tensor(prompt, dtype=torch.long)
    raise TypeError(f"Unsupported prompt type: {type(prompt)}")
