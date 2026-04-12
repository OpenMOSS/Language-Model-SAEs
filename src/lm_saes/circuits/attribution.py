from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
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

from lm_saes.backend.language_model import TransformerLensLanguageModel
from lm_saes.circuits.hooks import replace_biases_with_leaves
from lm_saes.circuits.indexed_tensor import (
    NodeDimension,
    NodeIndexed,
    NodeIndexedMatrix,
    NodeIndexedVector,
    NodeInfo,
)
from lm_saes.core.pytree import PyTree
from lm_saes.models.lorsa import LowRankSparseAttention
from lm_saes.models.molt import MixtureOfLinearTransform
from lm_saes.models.sae import SparseAutoEncoder
from lm_saes.models.sparse_dictionary import SparseDictionary
from lm_saes.utils.distributed import DimMap, full_tensor
from lm_saes.utils.distributed.ops import maybe_local_map, multi_batch_index, nonzero, searchsorted
from lm_saes.utils.misc import ensure_tokenized
from lm_saes.utils.timer import timer


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
class AttributionResult(PyTree):
    activations: NodeIndexedVector
    attribution: NodeIndexedMatrix
    logits: torch.Tensor
    probs: torch.Tensor
    prompt_token_ids: list[int] = field(default_factory=list)
    prompt_tokens: list[str] = field(default_factory=list)
    logit_token_ids: list[int] = field(default_factory=list)
    logit_tokens: list[str] = field(default_factory=list)
    qk_trace_results: NodeIndexed[list[NodeIndexed[torch.Tensor]]] | None = None


def get_normalized_matrix(matrix: NodeIndexedMatrix) -> NodeIndexedMatrix:
    return NodeIndexedMatrix.from_data(
        data=torch.abs(matrix.data) / torch.abs(matrix.data).sum(dim=1, keepdim=True).clamp(min=1e-8),
        dimensions=matrix.dimensions,
    )


@timer.time("compute_intermediates_attribution")
def compute_intermediates_attribution(
    attribution: NodeIndexedMatrix,
    targets: NodeDimension,
    intermediates: NodeDimension,
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


def retrieval_from_intermediates(dimension: NodeDimension, intermediates: Sequence[tuple[NodeInfoRef, NodeInfoRef]]):
    return [
        NodeInfoRef(
            key=node.key,
            indices=node.indices,
            ref=intermediate[0].ref,
        )
        for node in dimension.node_infos
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
) -> tuple[NodeIndexedMatrix, NodeDimension]:
    """
    Greedily collect attribution from targets to sources through intermediates.
    """

    all_sources = list(sources) + [intermediate[1] for intermediate in intermediates]

    targets_dimension = NodeDimension.from_node_infos(targets)
    all_sources_dimension = NodeDimension.from_node_infos(all_sources)
    source_intermediates_dimension = NodeDimension.from_node_infos([intermediate[1] for intermediate in intermediates])
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

        attribution[NodeDimension.from_node_infos(target_batch), None] = torch.cat(
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

    collected_intermediates_dimension = NodeDimension.empty(
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


@timer.time("collect_cache")
def collect_cache(
    model: TransformerLensLanguageModel,
    inputs: torch.Tensor | str,
    replacement_modules: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform],
    with_bias_leaves: bool = False,
):
    from contextlib import nullcontext

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
    enable_qk_tracing: bool = False,
    qk_top_fraction: float = 0.6,
    qk_topk: int = 10,
):
    assert not enable_qk_tracing or batch_size >= qk_topk, (
        f"attribute(batch_size={batch_size}) must be >= qk_topk={qk_topk} when enable_qk_tracing=True"
    )
    tokens = ensure_tokenized(inputs, model.tokenizer, device=model.device)
    replacement_modules_cast: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform] = cast(
        list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform], replacement_modules
    )
    batch_logits, cache = collect_cache(
        model,
        einops.repeat(tokens, "n -> b n", b=batch_size),
        replacement_modules_cast,
        with_bias_leaves=enable_qk_tracing,
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

    sources_dimension = NodeDimension.from_node_infos(sources)
    attribution = attribution[None, sources_dimension + collected_intermediates]

    intermediate_ref_map = {node_info.key: node_info.ref.detach() for node_info, _ in intermediates}
    activations = torch.cat(
        multi_batch_index(
            [node_info.ref[0].detach() for node_info in targets],
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
        dimensions=(
            NodeDimension.from_node_infos(targets) + collected_intermediates + NodeDimension.from_node_infos(sources),
        ),
    )

    prompt_token_ids = full_tensor(tokens).detach().cpu().tolist()
    logit_token_ids = full_tensor(top_idx).detach().cpu().tolist()

    qk_trace_results = None
    if enable_qk_tracing:
        lorsa_features = collected_intermediates.filter_keys(
            lambda key: any(
                isinstance(rm, LowRankSparseAttention) and key == rm.cfg.hook_point_out + ".sae.hook_feature_acts"
                for rm in replacement_modules_cast
            )
        )

        if len(lorsa_features) > 0:
            logits_dimension = NodeDimension.from_node_infos(targets)
            intermediates_attribution = compute_intermediates_attribution(
                attribution, logits_dimension, collected_intermediates, max_iter
            )

            reduction_weight_vec = NodeIndexedVector.from_data(top_p, dimensions=(logits_dimension,))
            lorsa_influence = reduction_weight_vec @ intermediates_attribution[None, lorsa_features]
            _, top_lorsa_features = lorsa_influence.topk(k=max(1, int(len(lorsa_features) * qk_top_fraction)))
            top_lorsa_node_infos = list(top_lorsa_features)

            rm_mapping = {rm.cfg.hook_point_out: rm for rm in replacement_modules_cast}
            # Dedup features that map to the same (hp, head, q, k): they would produce
            # identical Hessian computations, so compute once and fan out.
            unique_map: dict[tuple[str, int, int, int], int] = {}
            unique_qk_targets: list[NodeInfoRef] = []
            feature_to_unique: list[int] = []
            for feature_node in top_lorsa_node_infos:
                hp = feature_node.key.replace(".sae.hook_feature_acts", "")
                attn_score_ref = cache[hp + ".sae.hook_attn_score"]
                head_idx = (
                    int(full_tensor(feature_node.indices)[0, 1])
                    // cast(LowRankSparseAttention, rm_mapping[hp]).cfg.ov_group_size
                )
                q_pos, k_pos = torch.unravel_index(attn_score_ref[0, head_idx].argmax(), attn_score_ref.shape[-2:])
                dedup_key = (hp, head_idx, int(q_pos), int(k_pos))
                if dedup_key not in unique_map:
                    unique_map[dedup_key] = len(unique_qk_targets)
                    unique_qk_targets.append(
                        NodeInfoRef(
                            key=hp + ".sae.hook_attn_score",
                            ref=attn_score_ref,
                            indices=torch.tensor(
                                [[head_idx, int(q_pos), int(k_pos)]], device=model.device, dtype=torch.long
                            ),
                        )
                    )
                feature_to_unique.append(unique_map[dedup_key])

            from lm_saes.utils.distributed.ops import to_local

            assert model.model is not None
            bias_keys = [
                *(f"blocks.{i}.{kind}" for i in range(len(model.model.blocks)) for kind in ("attn.b_O", "mlp.b_out")),
                *(
                    rm.cfg.hook_point_out + suffix
                    for rm in replacement_modules_cast
                    for suffix in (".sae.b_Q", ".sae.b_K", ".sae.b_D")
                ),
            ]
            qk_sources: list[NodeInfoRef] = [
                *sources,
                *(
                    NodeInfoRef(
                        key=rm.cfg.hook_point_out + ".sae.hook_feature_acts",
                        ref=cache[rm.cfg.hook_point_out + ".sae.hook_feature_acts.post"],
                        indices=to_local(nonzero(cache[rm.cfg.hook_point_out + ".sae.hook_feature_acts.post"][0])),
                    )
                    for rm in replacement_modules_cast
                ),
                *(NodeInfoRef(key=k, ref=cache[k], indices=seq_indices) for k in bias_keys if k in cache),
            ]

            hessian = compute_hessian_matrix(unique_qk_targets, qk_sources, topk=qk_topk)
            qk_trace_results = NodeIndexed(
                value=[hessian.value[i] for i in feature_to_unique],
                dimensions=(NodeDimension.from_node_infos(top_lorsa_node_infos),),
            )

    return AttributionResult(
        activations=activations_vec,
        attribution=attribution,
        logits=batch_logits[:, -1, top_idx].detach(),
        probs=top_p,
        prompt_token_ids=prompt_token_ids,
        prompt_tokens=[model.tokenizer.decode([token_id]) for token_id in prompt_token_ids],
        logit_token_ids=logit_token_ids,
        logit_tokens=[model.tokenizer.decode([token_id]) for token_id in logit_token_ids],
        qk_trace_results=qk_trace_results,
    )


def qk_trace(
    model: TransformerLensLanguageModel,
    inputs: torch.Tensor | str,
    replacement_modules: list[SparseDictionary],
    lorsa_features: list[NodeInfo],
    topk: int = 10,
    batch_size: int = 1,
) -> NodeIndexed[list["NodeIndexed[torch.Tensor]"]]:
    from lm_saes.utils.distributed.ops import to_local

    tokens = ensure_tokenized(inputs, model.tokenizer, device=model.device)
    replacement_modules_cast: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform] = cast(
        list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform], replacement_modules
    )
    _, cache = collect_cache(
        model,
        einops.repeat(tokens, "n -> b n", b=batch_size * topk),
        replacement_modules_cast,
        with_bias_leaves=True,
    )

    assert model.model is not None
    seq_len = cache["hook_embed.post"].shape[1]
    pos_indices = torch.arange(seq_len, device=model.device).unsqueeze(-1)

    # sources: embed + errors + feature_acts + bias leaves
    sources: list[NodeInfoRef] = [NodeInfoRef(key="hook_embed", ref=cache["hook_embed.post"], indices=pos_indices)]
    for rm in replacement_modules_cast:
        hp = rm.cfg.hook_point_out
        sources.append(NodeInfoRef(key=hp + ".error", ref=cache[hp + ".error.post"], indices=pos_indices))
        sources.append(
            NodeInfoRef(
                key=hp + ".sae.hook_feature_acts",
                ref=cache[hp + ".sae.hook_feature_acts.post"],
                indices=to_local(nonzero(cache[hp + ".sae.hook_feature_acts.post"][0])),
            )
        )
    bias_keys = [
        *(f"blocks.{i}.{kind}" for i in range(len(model.model.blocks)) for kind in ("attn.b_O", "mlp.b_out")),
        *(
            rm.cfg.hook_point_out + suffix
            for rm in replacement_modules_cast
            for suffix in (".sae.b_Q", ".sae.b_K", ".sae.b_D")
        ),
    ]
    sources.extend(NodeInfoRef(key=k, ref=cache[k], indices=pos_indices) for k in bias_keys if k in cache)

    flat_features: list[NodeInfo] = [row for feat in lorsa_features for row in feat]

    rm_mapping = {rm.cfg.hook_point_out: rm for rm in replacement_modules_cast}

    unique_map: dict[tuple[str, int, int, int], int] = {}
    unique_qk_targets: list[NodeInfoRef] = []
    feature_to_unique: list[int] = []
    for feature_node in flat_features:
        hp = feature_node.key.replace(".sae.hook_feature_acts", "")
        attn_score_ref = cache[hp + ".sae.hook_attn_score"]
        head_idx = (
            int(full_tensor(feature_node.indices)[0, 1])
            // cast(LowRankSparseAttention, rm_mapping[hp]).cfg.ov_group_size
        )
        q_pos, k_pos = torch.unravel_index(attn_score_ref[0, head_idx].argmax(), attn_score_ref.shape[-2:])
        dedup_key = (hp, head_idx, int(q_pos), int(k_pos))
        if dedup_key not in unique_map:
            unique_map[dedup_key] = len(unique_qk_targets)
            unique_qk_targets.append(
                NodeInfoRef(
                    key=hp + ".sae.hook_attn_score",
                    ref=attn_score_ref,
                    indices=torch.tensor([[head_idx, int(q_pos), int(k_pos)]], device=model.device, dtype=torch.long),
                )
            )
        feature_to_unique.append(unique_map[dedup_key])

    hessian = compute_hessian_matrix(unique_qk_targets, sources, topk=topk)
    return NodeIndexed(
        value=[hessian.value[i] for i in feature_to_unique],
        dimensions=(NodeDimension.from_node_infos(flat_features),),
    )


def compute_hessian_matrix(
    targets: Sequence[NodeInfoRef],
    sources: Sequence[NodeInfoRef],
    topk: int,
) -> NodeIndexed[list["NodeIndexed[torch.Tensor]"]]:
    sources_dimension = NodeDimension.from_node_infos(sources)
    fwd_batch_size = sources[0].ref.shape[0]
    bwd_batch_size = fwd_batch_size // topk

    per_slot_results: list[NodeIndexed[torch.Tensor]] = []

    target_queue = NodeInfoQueue(list(targets))
    for target_batch in target_queue.iter(bwd_batch_size):
        stacked = torch.cat(values(target_batch), dim=1)  # (fwd_batch, n_slots)
        bwd_scores = torch.stack(
            [stacked[i * topk : (i + 1) * topk, i] for i in range(stacked.shape[1])],
            dim=0,
        )
        first_grad_refs = torch.autograd.grad(
            bwd_scores.sum(),
            [s.ref for s in sources],
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        first_grads = multi_batch_index(
            [g if g is not None else torch.zeros_like(s.ref) for g, s in zip(first_grad_refs, sources)],
            [s.indices for s in sources],
            n_batch_dims=1,
        )
        first_attributions = torch.cat(
            [
                einops.einsum(
                    value.detach(),
                    grad,
                    "batch n_elements ..., batch n_elements ... -> batch n_elements",
                )
                for value, grad in zip(values(sources), first_grads)
            ],
            dim=1,
        )  # (fwd_batch_size, n_sources)

        second_targets = NodeDimension.empty(device=bwd_scores.device)
        all_topk_indices = []
        for slot_idx in range(stacked.shape[1]):
            _, topk_indices = first_attributions[slot_idx * topk].topk(topk)
            second_targets = second_targets + sources_dimension.offsets_to_nodes(offsets=topk_indices)
            all_topk_indices.append(topk_indices)

        second_bwd_values = first_attributions[
            torch.arange(stacked.shape[1], device=bwd_scores.device).unsqueeze(1) * topk
            + torch.arange(topk, device=bwd_scores.device).unsqueeze(0),
            torch.stack(all_topk_indices, dim=0),
        ].reshape(-1)

        second_grads_refs = torch.autograd.grad(
            second_bwd_values.sum(),
            [s.ref for s in sources],
            retain_graph=True,
            allow_unused=True,
        )

        second_grads = multi_batch_index(
            [g if g is not None else torch.zeros_like(s.ref) for g, s in zip(second_grads_refs, sources)],
            [s.indices for s in sources],
            n_batch_dims=1,
        )
        second_attributions = torch.cat(
            [
                einops.einsum(
                    value.detach(),
                    grad,
                    "batch n_elements ..., batch n_elements ... -> batch n_elements",
                )
                for value, grad in zip(values(sources), second_grads)
            ],
            dim=1,
        )
        second_targets_list = list(second_targets)
        diag = torch.arange(topk, device=second_attributions.device)
        for slot_idx in range(stacked.shape[1]):
            topk_values, topk_indices = second_attributions[slot_idx * topk : (slot_idx + 1) * topk].topk(topk, dim=-1)
            pair_attrs = topk_values[diag, diag]
            second_nodes = list(sources_dimension.offsets_to_nodes(topk_indices[diag, diag]))
            first_nodes = second_targets_list[slot_idx * topk : (slot_idx + 1) * topk]

            order = torch.argsort(pair_attrs, descending=True).tolist()
            # Drop pairs whose attribution is exactly zero: they come from
            # ties inside `topk` over all-zero tails (e.g. sources whose
            # gradient path to this target is structurally zero), and the
            # specific pick among those ties is arbitrary and differs
            # between runs with different source orderings.
            order = [i for i in order if pair_attrs[i].item() != 0.0]
            device = pair_attrs.device
            per_slot_results.append(
                NodeIndexed(
                    value=pair_attrs[order] if len(order) > 0 else pair_attrs[:0],
                    dimensions=(
                        NodeDimension.from_node_infos([first_nodes[i] for i in order], device=device),
                        NodeDimension.from_node_infos([second_nodes[i] for i in order], device=device),
                    ),
                )
            )

    targets_dim = NodeDimension.from_node_infos(list(targets))
    return NodeIndexed(value=per_slot_results, dimensions=(targets_dim,))
