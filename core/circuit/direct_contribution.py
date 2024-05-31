from typing import Dict, Optional
from einops import einsum, rearrange
import torch
from transformer_lens import HookedTransformer
import functools
import torch.nn.functional as F

from core.circuit.computation_graph import Meta, NodeInfo, PathInfo, padding
from core.circuit.utils import concat, concat_bias, equal_shape, compact
from core.sae import SparseAutoEncoder


def mlp_direct_contribution(x: torch.Tensor, meta: list[list[Meta]], layer: int, model: HookedTransformer, cache: Dict[str, torch.Tensor]):
    assert equal_shape(x, meta)

    original = cache[f"blocks.{layer}.hook_resid_mid"][0]
    assert torch.allclose(x.sum(1), original, atol=1e-4, rtol=1e-3)

    ln_scale_orig = ((original - original.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True) + model.cfg.eps).sqrt()
    mlp_in_orig = (original - original.mean(-1, keepdim=True)) / ln_scale_orig
    mlp_in = (x - x.mean(-1, keepdim=True)) / ln_scale_orig.unsqueeze(1)
    assert torch.allclose(mlp_in.sum(1), mlp_in_orig, atol=1e-4, rtol=1e-3)

    pre_act_orig = einsum(mlp_in_orig, model.blocks[layer].mlp.W_in, "pos d_model, d_model d_mlp -> pos d_mlp") + model.blocks[layer].mlp.b_in
    pre_act = einsum(mlp_in, model.blocks[layer].mlp.W_in, "pos part d_model, d_model d_mlp -> pos part d_mlp")
    pre_act, meta = concat_bias(pre_act, meta, model.blocks[layer].mlp.b_in, (NodeInfo("b_in", layer=layer, module=f"blocks.{layer}.mlp"), None))
    assert torch.allclose(pre_act.sum(1), pre_act_orig, atol=1e-4, rtol=1e-3)

    post_act_orig = model.blocks[layer].mlp.act_fn(pre_act_orig)
    post_act = pre_act * (post_act_orig / pre_act_orig).unsqueeze(1) 
    assert torch.allclose(post_act.sum(1), post_act_orig, atol=1e-4, rtol=1e-3)

    mlp_out_orig = einsum(post_act_orig, model.blocks[layer].mlp.W_out, "pos d_mlp, d_mlp d_model -> pos d_model") + model.blocks[0].mlp.b_out
    mlp_out = einsum(post_act, model.blocks[layer].mlp.W_out, "pos part d_mlp, d_mlp d_model -> pos part d_model")
    mlp_out, meta = concat_bias(mlp_out, meta, model.blocks[layer].mlp.b_out, (NodeInfo("b_out", layer=layer, module=f"blocks.{layer}.mlp"), None))
    assert torch.allclose(mlp_out.sum(1), mlp_out_orig, atol=1e-4, rtol=1e-3)

    assert equal_shape(mlp_out, meta)
    return mlp_out, meta

def attn_direct_contribution(x: torch.Tensor, meta: list[list[Meta]], layer: int, model: HookedTransformer, cache: Dict[str, torch.Tensor], separate_heads: bool = False):
    assert equal_shape(x, meta)

    original = cache[f"blocks.{layer}.hook_resid_pre"][0]
    assert torch.allclose(x.sum(1), original, atol=1e-4, rtol=1e-3)

    ln_scale_orig = ((original - original.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True) + model.cfg.eps).sqrt()
    attn_in_orig = (original - original.mean(-1, keepdim=True)) / ln_scale_orig
    attn_in = (x - x.mean(-1, keepdim=True)) / ln_scale_orig.unsqueeze(1)
    assert torch.allclose(attn_in.sum(1), attn_in_orig, atol=1e-4, rtol=1e-3)

    q_orig = einsum(attn_in_orig, model.blocks[layer].attn.W_Q, "pos d_model, head_index d_model d_head -> pos head_index d_head") + model.blocks[layer].attn.b_Q
    k_orig = einsum(attn_in_orig, model.blocks[layer].attn.W_K, "pos d_model, head_index d_model d_head -> pos head_index d_head") + model.blocks[layer].attn.b_K
    v_orig = einsum(attn_in_orig, model.blocks[layer].attn.W_V, "pos d_model, head_index d_model d_head -> pos head_index d_head") + model.blocks[layer].attn.b_V
    v = einsum(attn_in, model.blocks[layer].attn.W_V, "pos part d_model, head_index d_model d_head -> pos part head_index d_head")
    v, meta = concat_bias(v, meta, model.blocks[layer].attn.b_V, (NodeInfo("b_V", layer=layer, module=f"blocks.{layer}.attn"), None))
    assert torch.allclose(v.sum(1), v_orig, atol=1e-4, rtol=1e-3)

    attn_scores_orig = einsum(q_orig, k_orig, "query_pos head_index d_head, key_pos head_index d_head -> head_index query_pos key_pos") / model.blocks[layer].attn.attn_scale
    mask = model.blocks[layer].attn.mask[None, -attn_scores_orig.size(-2):, -attn_scores_orig.size(-1):]
    attn_scores_orig = torch.where(mask, attn_scores_orig, model.blocks[layer].attn.IGNORE)

    pattern_orig = F.softmax(attn_scores_orig, dim=-1)
    pattern_orig = torch.where(torch.isnan(pattern_orig), torch.zeros_like(pattern_orig), pattern_orig).to(model.cfg.dtype)

    z_orig = einsum(v_orig, pattern_orig, "key_pos head_index d_head, head_index query_pos key_pos -> query_pos head_index d_head")
    z = einsum(v, pattern_orig, "key_pos part head_index d_head, head_index query_pos key_pos -> query_pos key_pos part head_index d_head")
    z = rearrange(z, "query_pos key_pos part head_index d_head -> query_pos (key_pos part) head_index d_head")
    meta = [functools.reduce(lambda x, y: x + y, meta) for _ in meta]
    assert torch.allclose(z.sum(1), z_orig, atol=1e-4, rtol=1e-3)

    out_orig = einsum(z_orig, model.blocks[layer].attn.W_O, "pos head_index d_head, head_index d_head d_model -> pos d_model") + model.blocks[layer].attn.b_O
    head_cnt = model.cfg.n_heads
    if not separate_heads:
        out = einsum(z, model.blocks[layer].attn.W_O, "pos part head_index d_head, head_index d_head d_model -> pos part d_model")
    else:
        out = einsum(z, model.blocks[layer].attn.W_O, "pos part head_index d_head, head_index d_head d_model -> pos part head_index d_model")
        out = rearrange(out, "pos part head_index d_model -> pos (part head_index) d_model")
        meta = [[(m[0], PathInfo(head=head_idx)) for m in ms for head_idx in range(head_cnt)] for ms in meta]
    out, meta = compact(out, meta)
    out, meta = concat_bias(out, meta, model.blocks[layer].attn.b_O, (NodeInfo("b_O", layer=layer, module=f"blocks.{layer}.attn"), None))
    
    assert torch.allclose(out.sum(1), out_orig, atol=1e-4, rtol=1e-3)

    assert equal_shape(out, meta)
    return out, meta

def attn_score_direct_contribution(q_x: torch.Tensor, q_meta: list[list[Meta]], k_x: torch.Tensor, k_meta: list[list[Meta]], layer: int, model: HookedTransformer, cache: Dict[str, torch.Tensor]):
    assert equal_shape(q_x, q_meta) and equal_shape(k_x, k_meta)

    original = cache[f"blocks.{layer}.hook_resid_pre"][0]
    assert torch.allclose(q_x.sum(1), original, atol=1e-4, rtol=1e-3)
    assert torch.allclose(k_x.sum(1), original, atol=1e-4, rtol=1e-3)

    ln_scale_orig = ((original - original.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True) + model.cfg.eps).sqrt()
    attn_in_orig = (original - original.mean(-1, keepdim=True)) / ln_scale_orig
    q_attn_in = (q_x - q_x.mean(-1, keepdim=True)) / ln_scale_orig.unsqueeze(1)
    k_attn_in = (k_x - k_x.mean(-1, keepdim=True)) / ln_scale_orig.unsqueeze(1)
    assert torch.allclose(q_attn_in.sum(1), attn_in_orig, atol=1e-4, rtol=1e-3)
    assert torch.allclose(k_attn_in.sum(1), attn_in_orig, atol=1e-4, rtol=1e-3)

    q_orig = einsum(attn_in_orig, model.blocks[layer].attn.W_Q, "pos d_model, head_index d_model d_head -> pos head_index d_head") + model.blocks[layer].attn.b_Q
    k_orig = einsum(attn_in_orig, model.blocks[layer].attn.W_K, "pos d_model, head_index d_model d_head -> pos head_index d_head") + model.blocks[layer].attn.b_K
    q = einsum(q_attn_in, model.blocks[layer].attn.W_Q, "pos part d_model, head_index d_model d_head -> pos part head_index d_head")
    k = einsum(k_attn_in, model.blocks[layer].attn.W_K, "pos part d_model, head_index d_model d_head -> pos part head_index d_head")
    q, q_meta = concat_bias(q, q_meta, model.blocks[layer].attn.b_Q, (NodeInfo("b_Q", layer=layer, module=f"blocks.{layer}.attn"), None))
    k, k_meta = concat_bias(k, k_meta, model.blocks[layer].attn.b_K, (NodeInfo("b_K", layer=layer, module=f"blocks.{layer}.attn"), None))
    assert torch.allclose(q.sum(1), q_orig, atol=1e-4, rtol=1e-3)
    assert torch.allclose(k.sum(1), k_orig, atol=1e-4, rtol=1e-3)

    attn_scores_orig = einsum(q_orig, k_orig, "query_pos head_index d_head, key_pos head_index d_head -> head_index query_pos key_pos") / model.blocks[layer].attn.attn_scale
    mask = model.blocks[layer].attn.mask[None, -attn_scores_orig.size(-2):, -attn_scores_orig.size(-1):]
    attn_scores_orig = torch.where(mask, attn_scores_orig, model.blocks[layer].attn.IGNORE)
    attn_scores = einsum(q, k, "query_pos query_part head_index d_head, key_pos key_part head_index d_head -> head_index query_pos key_pos query_part key_part") / model.blocks[layer].attn.attn_scale
    mask = model.blocks[layer].attn.mask[None, -attn_scores_orig.size(-2):, -attn_scores_orig.size(-1):, None, None]
    attn_scores = torch.where(mask, attn_scores, model.blocks[layer].attn.IGNORE)
    assert torch.allclose(attn_scores.sum([3, 4]), attn_scores_orig, atol=1e-4, rtol=1e-3)

    return attn_scores, q_meta, k_meta

def logits_direct_contribution(x: torch.Tensor, meta: list[list[Meta]], model: HookedTransformer, cache: Dict[str, torch.Tensor]):
    assert equal_shape(x, meta)

    original = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"][0]
    assert torch.allclose(x.sum(1), original, atol=1e-4, rtol=1e-3)

    ln_scale_orig = ((original - original.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True) + model.cfg.eps).sqrt()
    logits_in_orig = (original - original.mean(-1, keepdim=True)) / ln_scale_orig
    logits_in = (x - x.mean(-1, keepdim=True)) / ln_scale_orig.unsqueeze(1)

    logits_orig = einsum(logits_in_orig, model.unembed.W_U, "pos d_model, d_model vocab -> pos vocab") + model.unembed.b_U
    logits = einsum(logits_in, model.unembed.W_U, "pos part d_model, d_model vocab -> pos part vocab")
    logits, meta = concat_bias(logits, meta, model.unembed.b_U, (NodeInfo("b_U", module="unembed"), None))
    assert torch.allclose(logits.sum(1), logits_orig, atol=1e-4, rtol=1e-3)
    
    assert equal_shape(logits, meta)
    return logits, meta


def sae_direct_contribution(x: torch.Tensor, meta: list[list[Meta]], dict_name: str, sae: SparseAutoEncoder, cache: Dict[str, torch.Tensor]):
    assert equal_shape(x, meta)

    activation_in, activation_out = cache[sae.cfg.hook_point_in][0], cache[sae.cfg.hook_point_out][0]
    assert torch.allclose(x.sum(1), activation_in, atol=1e-4, rtol=1e-3)
    hidden_pre = einsum(
        x * sae.compute_norm_factor(activation_in).unsqueeze(1),
        sae.encoder,
        "... d_model, d_model d_sae -> ... d_sae",
    )
    hidden_pre, meta = concat_bias(hidden_pre, meta, sae.encoder_bias, (NodeInfo("encoder_bias", module=dict_name), None))
    feature_acts = sae.feature_act_mask * sae.feature_act_scale * hidden_pre / sae.compute_norm_factor(activation_out).unsqueeze(1)

    return feature_acts, meta



def partition_activation(dict_name: str, sae: SparseAutoEncoder, cache: Dict[str, torch.Tensor]):
    activation_in, activation_out = cache[sae.cfg.hook_point_in][0], cache[sae.cfg.hook_point_out][0]
    feature_acts = sae.encode(activation_in, label=activation_out)
    max_feature_count = (feature_acts > 0).sum(-1).max()
    features = torch.zeros(activation_in.size(0), max_feature_count, activation_in.size(-1), dtype=activation_in.dtype, device=activation_in.device)
    meta = [[] for _ in range(activation_in.size(0))]
    for pos in range(activation_in.size(0)):
        features[pos, :(feature_acts[pos] > 0).sum(-1).item()] = sae.decoder[feature_acts[pos] > 0] * feature_acts[pos][feature_acts[pos] > 0].unsqueeze(-1)
        meta[pos] = [(NodeInfo(str(feature_idx.item()), module=dict_name, pos=pos, activation=feature_acts[pos][feature_idx].item()), None) for feature_idx in feature_acts[pos].nonzero(as_tuple=True)[0]]
        meta[pos] += [padding for _ in range(max_feature_count - len(meta[pos]))]
    # print(dict_name, (aux["x_hat"] - features.sum(1)).detach().reshape(-1).sort(descending=True).values)
    # assert torch.allclose(aux["x_hat"], features.sum(1), atol=1e-4, rtol=1e-3)
    sae_error = activation_out - features.sum(1)
    features, meta = concat(features, meta, sae_error.unsqueeze(1), [[(NodeInfo("sae_error", module=dict_name, pos=i), None)] for i in range(features.size(0))])
    # assert torch.allclose(activation_out, features.sum(1), atol=1e-4, rtol=1e-3)
    return features, meta