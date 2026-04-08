from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Generic,
    Iterator,
    Sequence,
    TypeVar,
    cast,
)

import einops
import torch
from torch.distributed.tensor import DTensor
from tqdm import tqdm

from lm_saes.backend.hooks import replace_biases_with_leaves
from lm_saes.backend.indexed_tensor import Dimension, NodeIndexedMatrix, NodeIndexedVector, NodeInfo
from lm_saes.utils.distributed import DimMap, full_tensor
from lm_saes.utils.distributed.ops import maybe_local_map, multi_batch_index, nonzero, searchsorted
from lm_saes.utils.misc import ensure_tokenized
from lm_saes.utils.timer import timer

if TYPE_CHECKING:
    from lm_saes.backend.language_model import TransformerLensLanguageModel
    from lm_saes.models.lorsa import LowRankSparseAttention
    from lm_saes.models.molt import MixtureOfLinearTransform
    from lm_saes.models.sae import SparseAutoEncoder
    from lm_saes.models.sparse_dictionary import SparseDictionary


@dataclass
class NodeInfoRef(NodeInfo):
    """NodeInfo with reference to node (tensor) in computation graph."""

    ref: torch.Tensor


NodeInfoT = TypeVar("NodeInfoT", bound=NodeInfo)


class NodeInfoQueue(Generic[NodeInfoT]):
    def __init__(self, node_infos: Sequence[NodeInfoT] = ()):
        self.queue = list(node_infos)

    def enqueue(self, node_info: Sequence[NodeInfoT]):
        self.queue.extend(node_info)

    def dequeue(self, batch_size: int) -> Sequence[NodeInfoT]:
        accumulated = 0
        results = []
        while accumulated < batch_size and len(self.queue) > 0:
            if accumulated + len(self.queue[0]) > batch_size:
                results.append(self.queue[0][: batch_size - accumulated])
                self.queue[0] = self.queue[0][batch_size - accumulated :]
                accumulated = batch_size
            else:
                results.append(self.queue.pop(0))
                accumulated += len(results[-1])
        return results

    def iter(self, batch_size: int) -> Iterator[Sequence[NodeInfoT]]:
        while len(self.queue) > 0:
            yield self.dequeue(batch_size)


@dataclass
class AttributionResult:
    activations: NodeIndexedVector
    attribution: NodeIndexedMatrix
    logits: torch.Tensor
    probs: torch.Tensor
    prompt_token_ids: list[int] = field(default_factory=list)
    prompt_tokens: list[str] = field(default_factory=list)
    logit_token_ids: list[int] = field(default_factory=list)
    logit_tokens: list[str] = field(default_factory=list)


def get_normalized_matrix(matrix: NodeIndexedMatrix) -> NodeIndexedMatrix:
    return NodeIndexedMatrix.from_data(
        data=torch.abs(matrix.data) / torch.abs(matrix.data).sum(dim=1, keepdim=True).clamp(min=1e-8),
        dimensions=matrix.dimensions,
    )


@timer.time("compute_intermediates_attribution")
def compute_intermediates_attribution(
    attribution: NodeIndexedMatrix,
    targets: Dimension,
    intermediates: Dimension,
    max_iter: int,
) -> NodeIndexedMatrix:
    attribution = get_normalized_matrix(attribution)
    influence = attribution[targets, None]
    if len(intermediates) == 0:
        return influence
    t2i: NodeIndexedMatrix = attribution[targets, intermediates]
    i2all: NodeIndexedMatrix = attribution[intermediates, None]
    i2i: NodeIndexedMatrix = attribution[intermediates, intermediates]
    for _ in range(max_iter):
        cur_influence = t2i @ i2all
        if not torch.any(cur_influence.data):
            break
        influence += cur_influence
        t2i = t2i @ i2i
    return influence


@timer.time("values")
def values(node_infos: Sequence[NodeInfoRef]) -> list[torch.Tensor]:
    return multi_batch_index(
        [node_info.ref for node_info in node_infos],
        [node_info.indices for node_info in node_infos],
        n_batch_dims=1,
    )


@timer.time("grads")
def grads(node_infos: Sequence[NodeInfoRef]) -> list[torch.Tensor]:
    return multi_batch_index(
        [
            node_info.ref.grad if node_info.ref.grad is not None else torch.zeros_like(node_info.ref)
            for node_info in node_infos
        ],
        [node_info.indices for node_info in node_infos],
        n_batch_dims=1,
    )


def clear_grads(node_infos: Sequence[NodeInfoRef]) -> None:
    for node_info in node_infos:
        node_info.ref.grad = None


def retrieval_from_intermediates(dimension: Dimension, intermediates: Sequence[tuple[NodeInfoRef, NodeInfoRef]]):
    return [
        NodeInfoRef(
            key=node.key,
            indices=node.indices
            if dimension.device_mesh is None
            else DimMap({}).from_local(node.indices, dimension.device_mesh),
            ref=intermediate[0].ref,
        )
        for node in dimension.node_mappings.values()
        for intermediate in intermediates
        if node.key == intermediate[0].key
    ]


@timer.time("greedily_collect_attribution")
def greedily_collect_attribution(
    targets: Sequence[NodeInfoRef],
    sources: Sequence[NodeInfoRef],
    intermediates: Sequence[tuple[NodeInfoRef, NodeInfoRef]],  # [up as target, down as source]
    max_intermediates: int,
    reduction_weight: torch.Tensor,
    max_iter: int = 100,
) -> tuple[NodeIndexedMatrix, Dimension]:
    """
    Greedily collect attribution from targets to sources through intermediates.
    """

    all_sources = list(sources) + [intermediate[1] for intermediate in intermediates]

    targets_dimension = Dimension.from_node_infos(targets)
    all_sources_dimension = Dimension.from_node_infos(all_sources)
    source_intermediates_dimension = Dimension.from_node_infos([intermediate[1] for intermediate in intermediates])
    attribution = NodeIndexedMatrix.from_dimensions(
        dimensions=(targets_dimension, all_sources_dimension),
        device=targets[0].ref.device,
        dtype=targets[0].ref.dtype,
        device_mesh=targets[0].ref.device_mesh if isinstance(targets[0].ref, DTensor) else None,
    )

    batch_size = targets[0].ref.shape[0]

    queue = NodeInfoQueue(targets)
    with torch.no_grad():
        source_values = [value.detach() for value in values(all_sources)]

    for target_batch in queue.iter(batch_size):
        clear_grads(all_sources)
        root = maybe_local_map(torch.diag)(torch.cat(values(target_batch), dim=1))

        with timer.time("backward"):
            root.sum().backward(retain_graph=True)

        attribution[Dimension.from_node_infos(target_batch), None] = torch.cat(
            [
                einops.einsum(
                    value[: root.shape[0]],
                    grad.detach()[: root.shape[0]],
                    "batch n_elements ..., batch n_elements ... -> batch n_elements",
                )
                for value, grad in zip(source_values, grads(all_sources))
            ],
            dim=1,
        ).to(attribution.data.dtype)

    collected_intermediates_dimension = Dimension.empty(
        device=targets[0].ref.device,
        device_mesh=targets[0].ref.device_mesh if isinstance(targets[0].ref, DTensor) else None,
    )
    reduction_weight_vec: NodeIndexedVector = NodeIndexedVector.from_data(
        reduction_weight, dimensions=(targets_dimension,)
    )
    for i in tqdm(range(0, max_intermediates, batch_size)):
        cur_batch_size = min(batch_size, max_intermediates - i)
        intermediates_attribution = compute_intermediates_attribution(
            attribution, targets_dimension, collected_intermediates_dimension, max_iter
        )

        influence = reduction_weight_vec @ intermediates_attribution[None, source_intermediates_dimension]

        _, selected_nodes = influence.topk(k=cur_batch_size, ignore_dimension=collected_intermediates_dimension)

        collected_intermediates_dimension = collected_intermediates_dimension + selected_nodes

        clear_grads(all_sources)
        node_refs = retrieval_from_intermediates(selected_nodes, intermediates)
        root = maybe_local_map(torch.diag)(torch.cat(values(node_refs), dim=1))

        with timer.time("backward"):
            root.sum().backward(retain_graph=True)

        attribution.add_targets(
            selected_nodes,
            torch.cat(
                [
                    einops.einsum(
                        value[: root.shape[0]],
                        grad.detach()[: root.shape[0]],
                        "batch n_elements ..., batch n_elements ... -> batch n_elements",
                    )
                    for value, grad in zip(source_values, grads(all_sources))
                ],
                dim=1,
            ).to(attribution.data.dtype),
        )

    return attribution, collected_intermediates_dimension


def ln_detach_hooks(models: TransformerLensLanguageModel) -> list[str]:
    assert models.model is not None, "model must be initialized"
    detach_hooks = []
    for i, block in enumerate(models.model.blocks):
        for module_name in ["ln1", "ln2", "ln1_post", "ln2_post"]:
            if hasattr(block, module_name) and isinstance(getattr(block, module_name), torch.nn.Module):
                detach_hooks.append(f"blocks.{i}.{module_name}.hook_scale")

    detach_hooks.append("ln_final.hook_scale")
    return detach_hooks


def attn_detach_hooks(models: TransformerLensLanguageModel) -> list[str]:
    assert models.model is not None, "model must be initialized"
    detach_hooks = []
    for i, block in enumerate(models.model.blocks):
        if hasattr(block, "attn") and isinstance(block.attn, torch.nn.Module):
            detach_hooks.append(f"blocks.{i}.attn.hook_pattern")
            if models.model.cfg.use_qk_norm:
                detach_hooks.append(f"blocks.{i}.attn.q_norm.hook_scale")
                detach_hooks.append(f"blocks.{i}.attn.k_norm.hook_scale")
    return detach_hooks


@timer.time("_find_influence_threshold")
def _find_influence_threshold(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Find score threshold that keeps the desired fraction of total influence."""
    if scores.numel() == 0:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    sorted_scores = torch.sort(scores.view(-1), descending=True).values
    cumulative_score = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores).clamp(min=1e-8)
    threshold_index = searchsorted(cumulative_score, threshold)
    threshold_index = min(int(threshold_index.item()), len(cumulative_score) - 1)
    return sorted_scores[threshold_index]


@timer.time("prune_attribution")
def prune_attribution(
    attribution: NodeIndexedMatrix,
    logit_weights: torch.Tensor,
    node_threshold: float = 0.6,
    edge_threshold: float = 0.8,
) -> NodeIndexedMatrix:
    """Prune an attribution NodeIndexedMatrix by removing low-influence nodes and edges.

    The attribution matrix is expected to have:
    - dim 0 (targets/rows): logit nodes (key="logits") + collected feature nodes
    - dim 1 (sources/cols): embed nodes (key="hook_embed") + error nodes (key ends with ".error")
                             + all (possibly uncollected) feature nodes

    Logit nodes and embed/error source nodes are always kept. Feature nodes are pruned based
    on their cumulative contribution to the weighted logit output.

    Args:
        attribution: NodeIndexedMatrix from the attribution computation.
        node_threshold: Retain feature nodes accounting for this fraction of total influence.
        edge_threshold: Retain edges accounting for this fraction of total edge influence.
        logit_weights: Per-logit scalar weights (shape ``[n_logits]``).

    Returns:
        Pruned NodeIndexedMatrix containing only kept nodes and edges.
    """
    if node_threshold > 1.0 or node_threshold < 0.0:
        raise ValueError("node_threshold must be between 0.0 and 1.0")
    if edge_threshold > 1.0 or edge_threshold < 0.0:
        raise ValueError("edge_threshold must be between 0.0 and 1.0")

    logits_dimension = attribution.dimensions[0].filter_keys(lambda key: key == "logits")
    intermediates_dimension = attribution.dimensions[0].filter_keys(lambda key: key != "logits")
    optional_sources_dimension = (
        attribution.dimensions[1].filter_keys(lambda key: key.endswith(".error")) + intermediates_dimension
    )

    node_scores = NodeIndexedVector.from_data(
        logit_weights, dimensions=(logits_dimension,)
    ) @ compute_intermediates_attribution(attribution, logits_dimension, intermediates_dimension, max_iter=100)
    influence = node_scores[intermediates_dimension]
    influence.add_nodes(logits_dimension, logit_weights)
    edge_scores = NodeIndexedMatrix.from_data(
        get_normalized_matrix(attribution).data * influence[attribution.dimensions[0]].data[:, None],
        dimensions=attribution.dimensions,
    )

    node_mask = node_scores.map(lambda x: x >= _find_influence_threshold(x, node_threshold))
    edge_mask = edge_scores.map(lambda x: x >= _find_influence_threshold(x, edge_threshold))

    old_node_mask = node_mask.clone()
    node_mask[optional_sources_dimension] = node_mask[optional_sources_dimension] & edge_mask[
        None, optional_sources_dimension
    ].any(0)
    node_mask[intermediates_dimension] = node_mask[intermediates_dimension] & edge_mask[
        intermediates_dimension, None
    ].any(1)

    while not torch.equal(node_mask.data, old_node_mask.data):
        old_node_mask = node_mask.clone()
        edge_mask.masked_fill_dim_(1, ~node_mask[optional_sources_dimension], False)
        edge_mask.masked_fill_dim_(0, ~node_mask[intermediates_dimension], False)
        node_mask[optional_sources_dimension] = node_mask[optional_sources_dimension] & edge_mask[
            None, optional_sources_dimension
        ].any(0)
        node_mask[intermediates_dimension] = node_mask[intermediates_dimension] & edge_mask[
            intermediates_dimension, None
        ].any(1)

    attribution = attribution.clone()
    attribution.masked_fill_dim_(1, ~node_mask[optional_sources_dimension], 0)
    attribution.masked_fill_dim_(0, ~node_mask[intermediates_dimension], 0)
    attribution.masked_fill_(~edge_mask, 0)
    return attribution


@dataclass
class QKTraceRequest:
    lorsa: LowRankSparseAttention
    head_idx: int
    q_pos: int
    k_pos: int

    @property
    def dedup_key(self) -> tuple[str, int, int, int]:
        return (self.lorsa.cfg.hook_point_out, self.head_idx, self.q_pos, self.k_pos)

    @classmethod
    def from_lorsa_feature(cls, node_info: NodeInfo, lorsa: LowRankSparseAttention, attn_scores: torch.Tensor):
        head_idx = node_info.indices[0][1] // lorsa.cfg.ov_group_size
        attn_score = attn_scores[head_idx, 0, :, :]  # (q_pos, k_pos)
        q_pos, k_pos = torch.unravel_index(attn_score.argmax(), attn_score.shape)
        return cls(lorsa, int(head_idx), int(q_pos.item()), int(k_pos.item()))


@dataclass
class QKTraceResult:
    nodes: tuple[NodeInfo, NodeInfo]
    attribution: float


@timer.time("collect_cache")
def collect_cache(
    model: TransformerLensLanguageModel,
    inputs: torch.Tensor | str,
    replacement_modules: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform],
    with_bias_leaves: bool = False,
):
    from contextlib import nullcontext

    from lm_saes.models.lorsa import LowRankSparseAttention
    from lm_saes.models.sparse_dictionary import SparseDictionary

    assert model.model is not None, "model must be initialized"
    tokens = ensure_tokenized(inputs, model.tokenizer, device=model.device)

    bias_ctx = (
        replace_biases_with_leaves(
            model.model,
            cast(list[SparseDictionary], replacement_modules),
            batch_size=tokens.shape[0],
            seq_len=tokens.shape[1],
        )
        if with_bias_leaves
        else nullcontext({})
    )

    with model.apply_saes(cast(list[SparseDictionary], replacement_modules)):
        with bias_ctx as bias_leaves:
            with model.detach_at(
                ["hook_embed"]
                + [replacement_module.cfg.hook_point_out + ".error" for replacement_module in replacement_modules]
                + [
                    replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts"
                    for replacement_module in replacement_modules
                ]
                + [
                    replacement_module.cfg.hook_point_out + ".sae.hook_attn_pattern"
                    for replacement_module in replacement_modules
                    if isinstance(replacement_module, LowRankSparseAttention)
                ]
                + [
                    replacement_module.cfg.hook_point_out + item
                    for replacement_module in replacement_modules
                    if isinstance(replacement_module, LowRankSparseAttention) and replacement_module.cfg.use_post_qk_ln
                    for item in (".sae.ln_q.hook_scale", ".sae.ln_k.hook_scale")
                ]
                + ln_detach_hooks(model)
                + attn_detach_hooks(model)
            ):
                logits, cache = model.run_with_ref_cache(
                    tokens,
                    names_filter=["hook_embed.post"]
                    + [
                        replacement_module.cfg.hook_point_out + ".error.post"
                        for replacement_module in replacement_modules
                    ]
                    + [
                        replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.pre"
                        for replacement_module in replacement_modules
                    ]
                    + [
                        replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.post"
                        for replacement_module in replacement_modules
                    ]
                    + [
                        replacement_module.cfg.hook_point_out + ".sae.hook_attn_score"
                        for replacement_module in replacement_modules
                        if isinstance(replacement_module, LowRankSparseAttention)
                    ]
                    + ln_detach_hooks(model),
                )

    cache.update(bias_leaves)
    return logits, cache


@timer.time("attribute")
def attribute(
    model: TransformerLensLanguageModel,
    inputs: torch.Tensor | str,
    replacement_modules: list[SparseDictionary],
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_features: int | None = None,
):
    from lm_saes.models.lorsa import LowRankSparseAttention
    from lm_saes.models.molt import MixtureOfLinearTransform
    from lm_saes.models.sae import SparseAutoEncoder

    tokens = ensure_tokenized(inputs, model.tokenizer, device=model.device)
    replacement_modules_cast: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform] = cast(
        list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform], replacement_modules
    )
    batch_logits, cache = collect_cache(
        model, einops.repeat(tokens, "n -> b n", b=batch_size), replacement_modules_cast
    )

    with torch.no_grad():
        probs = torch.softmax(batch_logits[0, -1], dim=-1)
        top_p, top_idx = torch.topk(probs, max_n_logits)
        cutoff = int(searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
        top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    seq_len = cache["hook_embed.post"].shape[1]

    targets: list[NodeInfoRef] = [
        NodeInfoRef(
            key="logits",
            ref=batch_logits[:, -1, :] - batch_logits[:, -1, :].mean(dim=-1, keepdim=True),
            indices=top_idx.unsqueeze(-1),
        )
    ]

    seq_indices = (
        torch.arange(seq_len, device=model.device).unsqueeze(-1)
        if model.device_mesh is None
        else DTensor.from_local(
            torch.arange(seq_len, device=model.device).unsqueeze(-1),
            device_mesh=model.device_mesh,
            placements=DimMap({}).placements(model.device_mesh),
        )
    )
    sources: list[NodeInfoRef] = [
        NodeInfoRef(
            key="hook_embed",
            ref=cache["hook_embed.post"],
            indices=seq_indices,
        )
    ] + [
        NodeInfoRef(
            key=replacement_module.cfg.hook_point_out + ".error",
            ref=cache[replacement_module.cfg.hook_point_out + ".error.post"],
            indices=seq_indices,
        )
        for (replacement_module) in replacement_modules_cast
    ]

    intermediates: list[tuple[NodeInfoRef, NodeInfoRef]] = [
        (
            NodeInfoRef(
                key=replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts",
                ref=cache[replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.pre"],
                indices=nonzero(cache[replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.pre"][0]),
            ),
            NodeInfoRef(
                key=replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts",
                ref=cache[replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.post"],
                indices=nonzero(cache[replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.post"][0]),
            ),
        )
        for replacement_module in replacement_modules_cast
    ]

    max_intermediates = max_features if max_features is not None else len(intermediates)
    max_iter = len(replacement_modules) + 10

    attribution, collected_intermediates = greedily_collect_attribution(
        targets=targets,
        sources=sources,
        intermediates=intermediates,
        max_intermediates=max_intermediates,
        reduction_weight=top_p,
        max_iter=max_iter,
    )

    sources_dimension = Dimension.from_node_infos(sources)
    attribution = attribution[None, sources_dimension + collected_intermediates]

    intermediate_ref_map = {node_info.key: node_info.ref.detach() for node_info, _ in intermediates}
    activations = torch.cat(
        multi_batch_index(
            [node_info.ref[0] for node_info in targets],
            [node_info.indices for node_info in targets],
        )
        + multi_batch_index(
            [intermediate_ref_map[node_info.key][0] for node_info in collected_intermediates],
            [node_info.indices for node_info in collected_intermediates],
        )
        + [torch.ones_like(node_info.indices[:, 0], dtype=node_info.ref.dtype) for node_info in sources],
        dim=0,
    )

    activations_vec = NodeIndexedVector.from_data(
        data=activations,
        dimensions=(Dimension.from_node_infos(targets) + collected_intermediates + Dimension.from_node_infos(sources),),
    )

    prompt_token_ids = full_tensor(tokens).detach().cpu().tolist()
    logit_token_ids = full_tensor(top_idx).detach().cpu().tolist()

    return AttributionResult(
        activations=activations_vec,
        attribution=attribution,
        logits=batch_logits[:, -1, top_idx].detach(),
        probs=top_p,
        prompt_token_ids=prompt_token_ids,
        prompt_tokens=[model.tokenizer.decode([token_id]) for token_id in prompt_token_ids],
        logit_token_ids=logit_token_ids,
        logit_tokens=[model.tokenizer.decode([token_id]) for token_id in logit_token_ids],
    )


@timer.profile("qk_trace")
def qk_trace(
    model: TransformerLensLanguageModel,
    inputs: torch.Tensor | str,
    replacement_modules: list[SparseDictionary],
    lorsa_features: list[NodeInfo],
    topk: int = 10,
    batch_size: int = 1,
):
    from lm_saes.models.lorsa import LowRankSparseAttention
    from lm_saes.models.molt import MixtureOfLinearTransform
    from lm_saes.models.sae import SparseAutoEncoder

    tokens = ensure_tokenized(inputs, model.tokenizer, device=model.device)
    replacement_modules_cast: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform] = cast(
        list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform], replacement_modules
    )
    _, cache = collect_cache(
        model, einops.repeat(tokens, "n -> b n", b=batch_size * topk), replacement_modules_cast, with_bias_leaves=True
    )
    # print(cache["blocks.24.hook_attn_out.sae.hook_feature_acts.post"][0][2].nonzero())
    rm_mapping = {
        replacement_module.cfg.hook_point_out: replacement_module for replacement_module in replacement_modules_cast
    }
    requests = [
        QKTraceRequest.from_lorsa_feature(
            lorsa_feature,
            cast(LowRankSparseAttention, rm_mapping[lorsa_feature.key]),
            cache[lorsa_feature.key + ".sae.hook_attn_score"],
        )
        for lorsa_feature in lorsa_features
    ]

    unique_map: dict[tuple[str, int, int, int], int] = {}
    unique_requests: list[QKTraceRequest] = []
    request_indices: list[int] = []
    for req in requests:
        key = req.dedup_key
        if key not in unique_map:
            unique_map[key] = len(unique_requests)
            unique_requests.append(req)
        request_indices.append(unique_map[key])

    unique_results = qk_trace_from_request(model, unique_requests, cache, replacement_modules_cast, topk)
    return [unique_results[idx] for idx in request_indices]


@timer.profile("qk_trace_from_request")
def qk_trace_from_request(
    model: TransformerLensLanguageModel,
    requests: list[QKTraceRequest],
    cache: dict[str, torch.Tensor],
    replacement_modules: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform],
    topk: int,
) -> list[list[QKTraceResult]]:  # pyright: ignore[reportReturnType]
    from lm_saes.utils.distributed.ops import to_local

    seq_len = cache["hook_embed.post"].shape[1]
    fwd_batch_size = cache["hook_embed.post"].shape[0]
    bwd_batch_size = fwd_batch_size // topk
    pos_indices = torch.arange(seq_len, device=model.device).unsqueeze(-1)
    sources: list[NodeInfoRef] = (
        [
            NodeInfoRef(
                key="hook_embed",
                ref=cache["hook_embed.post"],
                indices=pos_indices,
            )
        ]
        + [
            NodeInfoRef(
                key=replacement_module.cfg.hook_point_out + ".error",
                ref=cache[replacement_module.cfg.hook_point_out + ".error.post"],
                indices=pos_indices,
            )
            for replacement_module in replacement_modules
        ]
        + [
            NodeInfoRef(
                key=replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts",
                ref=cache[replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.post"],
                indices=to_local(
                    nonzero(cache[replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.post"][0])
                ),
            )
            for replacement_module in replacement_modules
        ]
    )
    assert model.model is not None
    for i in range(len(model.model.blocks)):
        for key in (f"blocks.{i}.attn.b_O", f"blocks.{i}.mlp.b_out"):
            if key in cache:
                sources.append(NodeInfoRef(key=key, ref=cache[key], indices=pos_indices))
    for replacement_module in replacement_modules:
        hp = replacement_module.cfg.hook_point_out
        for suffix in (".sae.b_Q", ".sae.b_K", ".sae.b_D"):
            key = hp + suffix
            if key in cache:
                sources.append(NodeInfoRef(key=key, ref=cache[key], indices=pos_indices))

    sources_dimension = Dimension.from_node_infos(sources)
    results = []
    for batch_start in range(0, len(requests), bwd_batch_size):
        request_batch = requests[batch_start : batch_start + bwd_batch_size]
        clear_grads(sources)
        bwd_scores = torch.stack(
            [
                cache[request.lorsa.cfg.hook_point_out + ".sae.hook_attn_score"][
                    request.head_idx, batch_idx * topk : (batch_idx + 1) * topk, request.q_pos, request.k_pos
                ]
                for batch_idx, request in enumerate(request_batch)
            ],
            dim=0,
        )
        bwd_scores.sum().backward(create_graph=True, retain_graph=True)
        first_order_gradients = torch.cat(
            [
                einops.einsum(
                    value.detach(),
                    grad,
                    "batch n_elements ..., batch n_elements ... -> batch n_elements",
                )
                for value, grad in zip(values(sources), grads(sources))
            ],
            dim=1,
        )  # fwd_batch_size, n_sources
        second_sources = []
        all_topk_indices = []
        for batch_idx in range(len(request_batch)):
            _, topk_indices = first_order_gradients[batch_idx * topk].topk(topk)  # get the first row of each request
            second_sources.extend(sources_dimension.offsets_to_nodes(topk_indices))
            all_topk_indices.append(topk_indices)

        all_topk_indices = torch.stack(all_topk_indices, dim=0)  # (len(request_batch), topk)
        batch_row_indices = torch.arange(len(request_batch), device=model.device).unsqueeze(1) * topk + torch.arange(
            topk, device=model.device
        ).unsqueeze(0)  # (len(request_batch), topk)
        second_bwd_values = first_order_gradients[batch_row_indices, all_topk_indices].reshape(-1)
        clear_grads(sources)
        second_bwd_values.sum().backward(retain_graph=True)

        second_order_gradients = torch.cat(
            [
                einops.einsum(
                    value.detach(),
                    grad,
                    "batch n_elements ..., batch n_elements ... -> batch n_elements",
                )
                for value, grad in zip(values(sources), grads(sources))
            ],
            dim=1,
        )  # fwd_batch_size, n_sources

        for batch_idx in range(len(request_batch)):
            batch_results = []
            for k_idx in range(topk):
                idx = batch_idx * topk + k_idx
                topk_values, topk_indices = second_order_gradients[idx].topk(topk)
                topk_nodes = list(sources_dimension.offsets_to_nodes(topk_indices))
                batch_results.append(
                    QKTraceResult(
                        nodes=(second_sources[idx], topk_nodes[k_idx]),
                        attribution=topk_values[k_idx].item(),
                    )
                )

            batch_results.sort(key=lambda x: x.attribution, reverse=True)
            results.append(batch_results[:topk])

    return results
