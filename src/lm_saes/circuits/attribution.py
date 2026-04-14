from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Iterator,
    Self,
    Sequence,
    cast,
)

import einops
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from tqdm import tqdm

from lm_saes.backend.language_model import TransformerLensLanguageModel
from lm_saes.circuits.hooks import replace_model_biases_with_leaves, replace_sae_biases_with_leaves
from lm_saes.circuits.indexed_tensor import (
    NodeDimension,
    NodeIndexed,
    NodeIndexedMatrix,
    NodeIndexedVector,
    NodeInfo,
)
from lm_saes.core.pytree import PyTree
from lm_saes.core.serialize import migrate
from lm_saes.models.lorsa import LowRankSparseAttention
from lm_saes.models.molt import MixtureOfLinearTransform
from lm_saes.models.sae import SparseAutoEncoder
from lm_saes.models.sparse_dictionary import SparseDictionary
from lm_saes.utils.distributed import DimMap, full_tensor
from lm_saes.utils.distributed.ops import maybe_local_map, multi_batch_index, nonzero, searchsorted
from lm_saes.utils.misc import ensure_tokenized
from lm_saes.utils.timer import timer


@dataclass
class NodeRefs:
    """Collection of named tensor references indexed by a :class:`NodeDimension`."""

    mapping: dict[str, torch.Tensor]
    dimension: NodeDimension

    @classmethod
    def from_nodes_and_refs(cls, entries: Sequence[tuple[str, torch.Tensor, torch.Tensor]]) -> Self:
        """Construct from ``(key, indices, ref)`` triples."""
        mapping: dict[str, torch.Tensor] = {}
        node_infos: list[NodeInfo] = []
        for key, indices, ref in entries:
            mapping[key] = ref
            node_infos.append(NodeInfo(key=key, indices=indices))
        if len(node_infos) == 0:
            raise ValueError("Cannot build NodeRefs from empty entries without an explicit device.")
        return cls(mapping=mapping, dimension=NodeDimension.from_node_infos(node_infos))

    @property
    def device(self) -> torch.device:
        return next(iter(self.mapping.values())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.mapping.values())).dtype

    @property
    def device_mesh(self) -> DeviceMesh | None:
        first = next(iter(self.mapping.values()))
        return first.device_mesh if isinstance(first, DTensor) else None

    @property
    def batch_size(self) -> int:
        return next(iter(self.mapping.values())).shape[0]

    def refs(self) -> list[torch.Tensor]:
        """Unique ref tensors in insertion order (for ``torch.autograd.grad``)."""
        return list(self.mapping.values())

    @timer.time("values")
    def values(self) -> list[torch.Tensor]:
        """Index each ref by its node indices: ``mapping[key][:, *indices]``."""
        return multi_batch_index([(ref, indices) for _, indices, ref in self], n_batch_dims=1)

    @timer.time("grads")
    def grads(self) -> list[torch.Tensor]:
        """Like :meth:`values` but on ``.grad``."""
        return multi_batch_index(
            [(grad if (grad := ref.grad) is not None else torch.zeros_like(ref), indices) for _, indices, ref in self],
            n_batch_dims=1,
        )

    def clear_grads(self) -> None:
        for ref in self.mapping.values():
            ref.grad = None

    def iter_batches(self, batch_size: int) -> Iterator[NodeRefs]:
        """Yield sub-:class:`NodeRefs` of at most `batch_size` elements."""
        queue = list(self.dimension.node_infos)
        while queue:
            accumulated = 0
            batch: list[NodeInfo] = []
            while accumulated < batch_size and queue:
                ni = queue[0]
                remaining = batch_size - accumulated
                if len(ni) > remaining:
                    batch.append(ni[:remaining])
                    queue[0] = ni[remaining:]
                    accumulated = batch_size
                else:
                    batch.append(queue.pop(0))
                    accumulated += len(batch[-1])
            yield self[NodeDimension.from_node_infos(batch, device=self.dimension.device)]

    def __add__(self, other: NodeRefs) -> NodeRefs:
        """Combine two :class:`NodeRefs` (merge mappings, concatenate dimensions)."""
        return NodeRefs(mapping={**self.mapping, **other.mapping}, dimension=self.dimension + other.dimension)

    def __getitem__(self, index: NodeDimension) -> NodeRefs:
        """Subset to a sub-dimension (filter mapping to relevant keys)."""
        relevant_keys = {node.key for node in index.node_mappings.values()}
        return NodeRefs(
            mapping={k: v for k, v in self.mapping.items() if k in relevant_keys},
            dimension=index,
        )

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
        for ni in self.dimension.node_infos:
            yield (ni.key, ni.indices, self.mapping[ni.key])


@dataclass
class IntermediateRefs:
    """Paired upstream/downstream refs for intermediate nodes."""

    upstream: NodeRefs
    downstream: NodeRefs


@dataclass
class QKTracingResult(PyTree):
    """Bundled output of :func:`compute_qk_tracing`.

    Each outer ``NodeIndexed`` is keyed by the Lorsa-feature target slot. Within
    a slot, ``q_marginal`` / ``k_marginal`` are 1-D top-k rankings of per-side
    first-order source attributions, and ``pairs`` is a 2-D ranking over
    (Q-role, K-role) pair attributions produced by a single second backward
    driven by a merged Q/K top-k pick list.
    """

    q_marginal: NodeIndexed[list[NodeIndexed[torch.Tensor]]]
    k_marginal: NodeIndexed[list[NodeIndexed[torch.Tensor]]]
    pairs: NodeIndexed[list[NodeIndexed[torch.Tensor]]]

    @migrate(before="2")
    def _v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:  # type: ignore[misc]
        """Legacy MongoDB blobs stored ``qk_trace_results`` as a bare
        ``NodeIndexed[list[NodeIndexed[torch.Tensor]]]`` (the pair-only shape
        predating the bundled return type). Promote them by wrapping the
        legacy value as ``pairs`` and populating ``q_marginal`` /
        ``k_marginal`` with structurally-valid placeholders: each slot
        reuses the pair slot's Q-role / K-role :class:`NodeDimension`
        respectively, and fills the attribution values with zeros. The
        zero-filled marginals are a deliberate lossy fallback — legacy
        circuits never computed a dedicated K-marginal pass, so there is no
        authoritative value to restore.
        """

        def _dim_size(dim_raw: dict[str, Any]) -> int:
            return sum(int(node["indices"].shape[0]) for node in dim_raw.values()) if dim_raw else 0

        q_marginal_slots: list[dict[str, Any]] = []
        k_marginal_slots: list[dict[str, Any]] = []
        for slot in data["value"]:
            q_dim_raw = slot["dimensions"][0]
            k_dim_raw = slot["dimensions"][1]
            q_marginal_slots.append({"value": torch.zeros(_dim_size(q_dim_raw)), "dimensions": [q_dim_raw]})
            k_marginal_slots.append({"value": torch.zeros(_dim_size(k_dim_raw)), "dimensions": [k_dim_raw]})

        return {
            "q_marginal": {"value": q_marginal_slots, "dimensions": data["dimensions"]},
            "k_marginal": {"value": k_marginal_slots, "dimensions": data["dimensions"]},
            "pairs": data,
        }


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
    qk_trace_results: QKTracingResult | None = None


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


@timer.time("greedily_collect_attribution")
def greedily_collect_attribution(
    targets: NodeRefs,
    sources: NodeRefs,
    intermediates: IntermediateRefs,
    max_intermediates: int,
    reduction_weight: torch.Tensor,
    max_iter: int = 100,
) -> tuple[NodeIndexedMatrix, NodeDimension]:
    """Greedily collect attribution from targets to sources through intermediates."""

    all_sources = sources + intermediates.downstream

    attribution = NodeIndexedMatrix.from_dimensions(
        dimensions=(targets.dimension, all_sources.dimension),
        device=targets.device,
        dtype=targets.dtype,
        device_mesh=targets.device_mesh,
    )

    batch_size = targets.batch_size

    with torch.no_grad():
        source_values = [value.detach() for value in all_sources.values()]

    for target_batch in targets.iter_batches(batch_size):
        all_sources.clear_grads()
        root = maybe_local_map(torch.diag)(torch.cat(target_batch.values(), dim=1))

        with timer.time("backward"):
            root.sum().backward(retain_graph=True)

        attribution[target_batch.dimension, None] = torch.cat(
            [
                einops.einsum(
                    value[: root.shape[0]],
                    grad.detach()[: root.shape[0]],
                    "batch n_elements ..., batch n_elements ... -> batch n_elements",
                )
                for value, grad in zip(source_values, all_sources.grads())
            ],
            dim=1,
        ).to(attribution.data.dtype)

    collected = NodeDimension.empty(
        device=targets.device,
        device_mesh=targets.device_mesh,
    )
    reduction_weight_vec: NodeIndexedVector = NodeIndexedVector.from_data(
        reduction_weight, dimensions=(targets.dimension,)
    )
    for i in tqdm(range(0, max_intermediates, batch_size)):
        cur_batch_size = min(batch_size, max_intermediates - i)
        intermediates_attribution = compute_intermediates_attribution(
            attribution, targets.dimension, collected, max_iter
        )

        influence = reduction_weight_vec @ intermediates_attribution[None, intermediates.downstream.dimension]

        _, selected_nodes = influence.topk(k=cur_batch_size, ignore_dimension=collected)

        collected = collected + selected_nodes

        all_sources.clear_grads()
        selected_refs = intermediates.upstream[selected_nodes]
        root = maybe_local_map(torch.diag)(torch.cat(selected_refs.values(), dim=1))

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
                    for value, grad in zip(source_values, all_sources.grads())
                ],
                dim=1,
            ).to(attribution.data.dtype),
        )

    return attribution, collected


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

    logits = attribution.dimensions[0].filter_keys(lambda key: key == "logits")
    intermediates = attribution.dimensions[0].filter_keys(lambda key: key != "logits")
    optional_sources = attribution.dimensions[1].filter_keys(lambda key: key.endswith(".error")) + intermediates

    node_scores = NodeIndexedVector.from_data(logit_weights, dimensions=(logits,)) @ compute_intermediates_attribution(
        attribution, logits, intermediates, max_iter=100
    )
    influence = node_scores[intermediates]
    influence.add_nodes(logits, logit_weights)
    edge_scores = NodeIndexedMatrix.from_data(
        get_normalized_matrix(attribution).data * influence[attribution.dimensions[0]].data[:, None],
        dimensions=attribution.dimensions,
    )

    node_mask = node_scores.map(lambda x: x >= _find_influence_threshold(x, node_threshold))
    edge_mask = edge_scores.map(lambda x: x >= _find_influence_threshold(x, edge_threshold))

    old_node_mask = node_mask.clone()
    node_mask[optional_sources] = node_mask[optional_sources] & edge_mask[None, optional_sources].any(0)
    node_mask[intermediates] = node_mask[intermediates] & edge_mask[intermediates, None].any(1)

    while not torch.equal(node_mask.data, old_node_mask.data):
        old_node_mask = node_mask.clone()
        edge_mask.masked_fill_dim_(1, ~node_mask[optional_sources], False)
        edge_mask.masked_fill_dim_(0, ~node_mask[intermediates], False)
        node_mask[optional_sources] = node_mask[optional_sources] & edge_mask[None, optional_sources].any(0)
        node_mask[intermediates] = node_mask[intermediates] & edge_mask[intermediates, None].any(1)

    attribution = attribution.clone()
    attribution.masked_fill_dim_(1, ~node_mask[optional_sources], 0)
    attribution.masked_fill_dim_(0, ~node_mask[intermediates], 0)
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
    batch_size, seq_len = tokens.shape[0], tokens.shape[1]

    model_bias_ctx = (
        replace_model_biases_with_leaves(model.model, batch_size, seq_len) if with_bias_leaves else nullcontext([])
    )
    sae_bias_ctx = (
        replace_sae_biases_with_leaves(
            model.model,
            cast(list[SparseDictionary], replacement_modules),
            batch_size,
            seq_len,
        )
        if with_bias_leaves
        else nullcontext([])
    )

    detach_hook_points = (
        ["hook_embed"]
        + [rm.cfg.hook_point_out + ".error" for rm in replacement_modules]
        + [rm.cfg.hook_point_out + ".sae.hook_feature_acts" for rm in replacement_modules]
        + [
            rm.cfg.hook_point_out + ".sae.hook_attn_pattern"
            for rm in replacement_modules
            if isinstance(rm, LowRankSparseAttention)
        ]
        + [
            rm.cfg.hook_point_out + item
            for rm in replacement_modules
            if isinstance(rm, LowRankSparseAttention) and rm.cfg.use_post_qk_ln
            for item in (".sae.ln_q.hook_scale", ".sae.ln_k.hook_scale")
        ]
        + ln_detach_hooks(model)
        + attn_detach_hooks(model)
    )

    with (
        model_bias_ctx as model_bias_names,
        model.apply_saes(cast(list[SparseDictionary], replacement_modules)),
        sae_bias_ctx as sae_bias_names,
        model.detach_at(detach_hook_points),
    ):
        logits, cache = model.run_with_ref_cache(
            tokens,
            names_filter=["hook_embed.post"]
            + [rm.cfg.hook_point_out + ".error.post" for rm in replacement_modules]
            + [rm.cfg.hook_point_out + ".sae.hook_feature_acts.pre" for rm in replacement_modules]
            + [rm.cfg.hook_point_out + ".sae.hook_feature_acts.post" for rm in replacement_modules]
            + [
                rm.cfg.hook_point_out + ".sae.hook_attn_score"
                for rm in replacement_modules
                if isinstance(rm, LowRankSparseAttention)
            ]
            + [
                rm.cfg.hook_point_out + suffix
                for rm in replacement_modules
                if isinstance(rm, LowRankSparseAttention)
                for suffix in (".sae.hook_q_post_rot", ".sae.hook_k_post_rot")
            ]
            + ln_detach_hooks(model)
            + model_bias_names
            + sae_bias_names,
        )

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
    replacement_modules: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform] = cast(
        list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform], replacement_modules
    )
    batch_logits, cache = collect_cache(
        model,
        einops.repeat(tokens, "n -> b n", b=batch_size),
        replacement_modules,
        with_bias_leaves=enable_qk_tracing,
    )

    with torch.no_grad():
        probs = torch.softmax(batch_logits[0, -1], dim=-1)
        top_p, top_idx = torch.topk(probs, max_n_logits)
        cutoff = int(searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
        top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    seq_len = cache["hook_embed.post"].shape[1]

    targets = NodeRefs.from_nodes_and_refs(
        [
            (
                "logits",
                top_idx.unsqueeze(-1),
                batch_logits[:, -1, :] - batch_logits[:, -1, :].mean(dim=-1, keepdim=True),
            ),
        ]
    )

    seq_indices = (
        torch.arange(seq_len, device=model.device).unsqueeze(-1)
        if model.device_mesh is None
        else DTensor.from_local(
            torch.arange(seq_len, device=model.device).unsqueeze(-1),
            device_mesh=model.device_mesh,
            placements=DimMap({}).placements(model.device_mesh),
        )
    )
    sources = NodeRefs.from_nodes_and_refs(
        [("hook_embed", seq_indices, cache["hook_embed.post"])]
        + [
            (rm.cfg.hook_point_out + ".error", seq_indices, cache[rm.cfg.hook_point_out + ".error.post"])
            for rm in replacement_modules
        ]
    )

    intermediate_entries = [
        (
            rm.cfg.hook_point_out + ".sae.hook_feature_acts",
            nonzero(cache[rm.cfg.hook_point_out + ".sae.hook_feature_acts.pre"][0]),
            cache[rm.cfg.hook_point_out + ".sae.hook_feature_acts.pre"],
            cache[rm.cfg.hook_point_out + ".sae.hook_feature_acts.post"],
        )
        for rm in replacement_modules
    ]
    intermediates = IntermediateRefs(
        upstream=NodeRefs.from_nodes_and_refs([(key, idx, pre) for key, idx, pre, _ in intermediate_entries]),
        downstream=NodeRefs.from_nodes_and_refs([(key, idx, post) for key, idx, _, post in intermediate_entries]),
    )

    max_intermediates = max_features if max_features is not None else len(replacement_modules)
    max_iter = len(replacement_modules) + 10

    attribution, collected_intermediates = greedily_collect_attribution(
        targets=targets,
        sources=sources,
        intermediates=intermediates,
        max_intermediates=max_intermediates,
        reduction_weight=top_p,
        max_iter=max_iter,
    )

    attribution = attribution[None, sources.dimension + collected_intermediates]

    activations = torch.cat(
        multi_batch_index(
            [(ref[0].detach(), indices) for _, indices, ref in targets],
        )
        + multi_batch_index(
            [(ref[0].detach(), indices) for key, indices, ref in intermediates.upstream[collected_intermediates]],
        )
        + [torch.ones_like(indices[:, 0], dtype=sources.dtype) for _, indices, _ in sources],
        dim=0,
    )

    activations_vec = NodeIndexedVector.from_data(
        data=activations,
        dimensions=(targets.dimension + collected_intermediates + sources.dimension,),
    )

    prompt_token_ids = full_tensor(tokens).detach().cpu().tolist()
    logit_token_ids = full_tensor(top_idx).detach().cpu().tolist()

    qk_trace_results = None
    if enable_qk_tracing:
        lorsa_features = collected_intermediates.filter_keys(
            lambda key: any(
                isinstance(rm, LowRankSparseAttention) and key == rm.cfg.hook_point_out + ".sae.hook_feature_acts"
                for rm in replacement_modules
            )
        )

        if len(lorsa_features) > 0:
            lorsa_influence = (
                NodeIndexedVector.from_data(top_p, dimensions=(targets.dimension,))
                @ compute_intermediates_attribution(attribution, targets.dimension, collected_intermediates, max_iter)[
                    None, lorsa_features
                ]
            )

            _, top_lorsa_features = lorsa_influence.topk(k=max(1, int(len(lorsa_features) * qk_top_fraction)))

            q_targets, k_targets, slot_to_unique = _retrieve_qk_vector_targets(
                top_lorsa_features,
                cache,
                {
                    rm.cfg.hook_point_out: rm.cfg.ov_group_size
                    for rm in replacement_modules
                    if isinstance(rm, LowRankSparseAttention)
                },
                {
                    rm.cfg.hook_point_out: rm.attn_scale
                    for rm in replacement_modules
                    if isinstance(rm, LowRankSparseAttention)
                },
            )

            assert model.model is not None
            bias_keys = [
                f"blocks.{i}.{kind}"
                for i in range(len(model.model.blocks))
                for kind in ("attn.hook_b_O", "mlp.hook_b_out")
            ] + [
                rm.cfg.hook_point_out + suffix
                for rm in replacement_modules
                for suffix in (".sae.hook_b_Q", ".sae.hook_b_K", ".sae.hook_b_D")
            ]
            qk_sources = sources + NodeRefs.from_nodes_and_refs(
                [
                    (
                        rm.cfg.hook_point_out + ".sae.hook_feature_acts",
                        nonzero(cache[rm.cfg.hook_point_out + ".sae.hook_feature_acts.post"][0]),
                        cache[rm.cfg.hook_point_out + ".sae.hook_feature_acts.post"],
                    )
                    for rm in replacement_modules
                ]
                + [(k, seq_indices, cache[k]) for k in bias_keys if k in cache]
            )

            tracing_results = compute_qk_tracing(q_targets, k_targets, qk_sources, topk=qk_topk)
            qk_trace_results = QKTracingResult(
                q_marginal=NodeIndexed(
                    value=[tracing_results.q_marginal.value[unique_idx] for unique_idx in slot_to_unique],
                    dimensions=(top_lorsa_features,),
                ),
                k_marginal=NodeIndexed(
                    value=[tracing_results.k_marginal.value[unique_idx] for unique_idx in slot_to_unique],
                    dimensions=(top_lorsa_features,),
                ),
                pairs=NodeIndexed(
                    value=[tracing_results.pairs.value[unique_idx] for unique_idx in slot_to_unique],
                    dimensions=(top_lorsa_features,),
                ),
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


def _retrieve_qk_vector_targets(
    features: NodeDimension,
    cache: dict[str, torch.Tensor],
    ov_group_sizes: dict[str, int],
    attn_scales: dict[str, float],
) -> tuple[NodeRefs, NodeRefs, list[int]]:
    """Map Lorsa features to ``Q(q_pos, h)`` / ``K(k_pos, h)`` vector targets.

    For each feature, pick its ``(q_pos, qk_head, k_pos)`` triple where ``k_pos``
    is the argmax key for that ``(qk_head, q_pos)``, and emit two parallel
    per-slot ``NodeRefs`` whose value at each slot is a ``(d_qk_head,)``-shaped
    slice of ``hook_q`` / ``hook_k``. Duplicate triples across features (which
    arise when ``ov_group_size > 1``) are collapsed — the returned ``NodeRefs``
    only contain unique targets. The ``slot_to_unique`` list maps each original
    feature offset to its index in the deduplicated slot list.

    The cached ``hook_q`` / ``hook_k`` tensors are **pre-scaled by**
    ``1 / sqrt(attn_scale)`` (per Lorsa module). This makes the implicit
    bilinear target ``Q·K`` evaluate to ``(Q·K) / attn_scale`` — the actual
    attention score fed into softmax — so the resulting pair attributions and
    marginals are on the same scale as the pre-refactor ``compute_hessian_matrix``
    path which targeted ``hook_attn_score`` directly.
    """
    slot_triples: list[tuple[str, int, int, int]] = []
    for feature_node in features:
        sae_path = feature_node.key.replace(".sae.hook_feature_acts", "")
        attn_score_ref = cache[f"{sae_path}.sae.hook_attn_score"]
        q_pos, ov_head_idx = full_tensor(feature_node.indices)[0].tolist()
        qk_head_idx = ov_head_idx // ov_group_sizes[sae_path]
        k_pos = int(attn_score_ref[0, qk_head_idx, q_pos].argmax().item())
        slot_triples.append((sae_path, int(qk_head_idx), int(q_pos), k_pos))

    seen: dict[tuple[str, int, int, int], int] = {}
    deduped: list[tuple[str, int, int, int]] = []
    slot_to_unique: list[int] = []
    for triple in slot_triples:
        unique_idx = seen.get(triple)
        if unique_idx is None:
            unique_idx = len(deduped)
            seen[triple] = unique_idx
            deduped.append(triple)
        slot_to_unique.append(unique_idx)

    # Memoize per-sae_path scaled refs — one scaled copy per Lorsa module,
    # shared across all slots that target the same module.
    scaled_q_refs: dict[str, torch.Tensor] = {}
    scaled_k_refs: dict[str, torch.Tensor] = {}

    def _scaled_refs(sae_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        if sae_path not in scaled_q_refs:
            inv_sqrt_scale = attn_scales[sae_path] ** -0.5
            scaled_q_refs[sae_path] = cache[f"{sae_path}.sae.hook_q_post_rot"] * inv_sqrt_scale
            scaled_k_refs[sae_path] = cache[f"{sae_path}.sae.hook_k_post_rot"] * inv_sqrt_scale
        return scaled_q_refs[sae_path], scaled_k_refs[sae_path]

    q_entries: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    k_entries: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for sae_path, qk_head_idx, q_pos, k_pos in deduped:
        q_ref, k_ref = _scaled_refs(sae_path)
        device = q_ref.device
        q_entries.append(
            (
                f"{sae_path}.sae.hook_q_post_rot",
                torch.tensor([[q_pos, qk_head_idx]], device=device),
                q_ref,
            )
        )
        k_entries.append(
            (
                f"{sae_path}.sae.hook_k_post_rot",
                torch.tensor([[k_pos, qk_head_idx]], device=device),
                k_ref,
            )
        )
    q_targets = NodeRefs.from_nodes_and_refs(q_entries)
    k_targets = NodeRefs.from_nodes_and_refs(k_entries)
    return q_targets, k_targets, slot_to_unique


def qk_trace(
    model: TransformerLensLanguageModel,
    inputs: torch.Tensor | str,
    replacement_modules: list[SparseDictionary],
    lorsa_features: NodeDimension,
    topk: int = 10,
    batch_size: int = 1,
) -> QKTracingResult:
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
    source_entries: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("hook_embed", pos_indices, cache["hook_embed.post"]),
    ]
    for rm in replacement_modules_cast:
        hp = rm.cfg.hook_point_out
        source_entries.append((hp + ".error", pos_indices, cache[hp + ".error.post"]))
        source_entries.append(
            (
                hp + ".sae.hook_feature_acts",
                nonzero(cache[hp + ".sae.hook_feature_acts.post"][0]),
                cache[hp + ".sae.hook_feature_acts.post"],
            )
        )
    bias_keys = [
        f"blocks.{i}.{kind}" for i in range(len(model.model.blocks)) for kind in ("attn.hook_b_O", "mlp.hook_b_out")
    ] + [
        rm.cfg.hook_point_out + suffix
        for rm in replacement_modules_cast
        for suffix in (".sae.hook_b_Q", ".sae.hook_b_K", ".sae.hook_b_D")
    ]
    source_entries.extend((k, pos_indices, cache[k]) for k in bias_keys if k in cache)
    qk_sources = NodeRefs.from_nodes_and_refs(source_entries)

    q_targets, k_targets, slot_to_unique = _retrieve_qk_vector_targets(
        lorsa_features,
        cache,
        {
            rm.cfg.hook_point_out: rm.cfg.ov_group_size
            for rm in replacement_modules_cast
            if isinstance(rm, LowRankSparseAttention)
        },
        {
            rm.cfg.hook_point_out: rm.attn_scale
            for rm in replacement_modules_cast
            if isinstance(rm, LowRankSparseAttention)
        },
    )
    tracing_results = compute_qk_tracing(q_targets, k_targets, qk_sources, topk=topk)
    return QKTracingResult(
        q_marginal=NodeIndexed(
            value=[tracing_results.q_marginal.value[unique_idx] for unique_idx in slot_to_unique],
            dimensions=(lorsa_features,),
        ),
        k_marginal=NodeIndexed(
            value=[tracing_results.k_marginal.value[unique_idx] for unique_idx in slot_to_unique],
            dimensions=(lorsa_features,),
        ),
        pairs=NodeIndexed(
            value=[tracing_results.pairs.value[unique_idx] for unique_idx in slot_to_unique],
            dimensions=(lorsa_features,),
        ),
    )


def compute_hessian_matrix(
    targets: NodeRefs,
    sources: NodeRefs,
    topk: int,
) -> NodeIndexed[list["NodeIndexed[torch.Tensor]"]]:
    fwd_batch_size = sources.batch_size
    bwd_batch_size = fwd_batch_size // topk

    per_slot_results: list[NodeIndexed[torch.Tensor]] = []

    for target_batch in targets.iter_batches(bwd_batch_size):
        stacked = torch.cat(target_batch.values(), dim=1)  # (fwd_batch, n_slots)
        bwd_scores = torch.stack(
            [stacked[i * topk : (i + 1) * topk, i] for i in range(stacked.shape[1])],
            dim=0,
        )
        first_grad_refs = torch.autograd.grad(
            bwd_scores.sum(),
            sources.refs(),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        first_grads = multi_batch_index(
            [
                (g if g is not None else torch.zeros_like(ref), indices)
                for g, (_, indices, ref) in zip(first_grad_refs, sources)
            ],
            n_batch_dims=1,
        )
        first_attributions = torch.cat(
            [
                einops.einsum(
                    value.detach(),
                    grad,
                    "batch n_elements ..., batch n_elements ... -> batch n_elements",
                )
                for value, grad in zip(sources.values(), first_grads)
            ],
            dim=1,
        )  # (fwd_batch_size, n_sources)

        second_targets = NodeDimension.empty(device=bwd_scores.device)
        all_topk_indices = []
        for slot_idx in range(stacked.shape[1]):
            _, topk_indices = first_attributions[slot_idx * topk].topk(topk)
            second_targets = second_targets + sources.dimension.offsets_to_nodes(offsets=topk_indices)
            all_topk_indices.append(topk_indices)

        second_bwd_values = first_attributions[
            torch.arange(stacked.shape[1], device=bwd_scores.device).unsqueeze(1) * topk
            + torch.arange(topk, device=bwd_scores.device).unsqueeze(0),
            torch.stack(all_topk_indices, dim=0),
        ].reshape(-1)

        second_grads_refs = torch.autograd.grad(
            second_bwd_values.sum(),
            sources.refs(),
            retain_graph=True,
            allow_unused=True,
        )

        second_grads = multi_batch_index(
            [
                (g if g is not None else torch.zeros_like(ref), indices)
                for g, (_, indices, ref) in zip(second_grads_refs, sources)
            ],
            n_batch_dims=1,
        )
        second_attributions = torch.cat(
            [
                einops.einsum(
                    value.detach(),
                    grad,
                    "batch n_elements ..., batch n_elements ... -> batch n_elements",
                )
                for value, grad in zip(sources.values(), second_grads)
            ],
            dim=1,
        )
        second_targets_list = list(second_targets)
        diag = torch.arange(topk, device=second_attributions.device)
        for slot_idx in range(stacked.shape[1]):
            topk_values, topk_indices = second_attributions[slot_idx * topk : (slot_idx + 1) * topk].topk(topk, dim=-1)
            pair_attrs = topk_values[diag, diag]
            second_nodes = list(sources.dimension.offsets_to_nodes(topk_indices[diag, diag]))
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

    return NodeIndexed(value=per_slot_results, dimensions=(targets.dimension,))


def _first_order_attributions(
    grad_refs: Sequence[torch.Tensor | None],
    sources: NodeRefs,
) -> torch.Tensor:
    """Turn per-ref raw gradient tensors into a ``(fwd_batch, n_sources)`` table
    of ``v_i · grad_i`` scalars, one per source element — the same pattern as
    :func:`compute_hessian_matrix` uses for its ``first_attributions`` tensor.
    """
    indexed = multi_batch_index(
        [(g if g is not None else torch.zeros_like(ref), indices) for g, (_, indices, ref) in zip(grad_refs, sources)],
        n_batch_dims=1,
    )
    return torch.cat(
        [
            einops.einsum(
                value.detach(),
                grad,
                "batch n_elements ..., batch n_elements ... -> batch n_elements",
            )
            for value, grad in zip(sources.values(), indexed)
        ],
        dim=1,
    )


def _emit_pair_slot(
    pair_rows: torch.Tensor,
    pick_is_q: torch.Tensor,
    pick_src: torch.Tensor,
    sources: NodeRefs,
    device: torch.device,
) -> NodeIndexed[torch.Tensor]:
    """Fold per-pick rows into a deduped ``(Q, K)`` NodeIndexed.

    Each row is ``T[pick, :]`` for a Q-pick or ``T[:, pick]`` for a K-pick;
    its argmax is the complementary partner. A pair ``(i, j)`` can surface
    twice — once from the Q side picking ``j`` as partner for ``i`` and once
    from the K side picking ``i`` as partner for ``j`` — so we dedup on
    ``(q_node, k_node)`` keeping the first (largest) occurrence.
    """
    partner_val, partner_idx = (t.squeeze(-1) for t in pair_rows.topk(1, dim=-1))
    q_src = torch.where(pick_is_q, pick_src, partner_idx)
    k_src = torch.where(pick_is_q, partner_idx, pick_src)
    q_nodes = list(sources.dimension.offsets_to_nodes(q_src))
    k_nodes = list(sources.dimension.offsets_to_nodes(k_src))

    def _key(ni: NodeInfo) -> tuple:
        return (ni.key, tuple(ni.indices.flatten().tolist()))

    seen: set[tuple] = set()
    kept: list[tuple[NodeInfo, NodeInfo, torch.Tensor]] = []
    for i in torch.argsort(partner_val, descending=True).tolist():
        if partner_val[i].item() == 0.0:
            continue
        pair_key = (_key(q_nodes[i]), _key(k_nodes[i]))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        kept.append((q_nodes[i], k_nodes[i], partner_val[i]))

    if not kept:
        empty = NodeDimension.empty(device=device)
        return NodeIndexed(value=partner_val[:0], dimensions=(empty, empty))

    q_kept, k_kept, vals_kept = zip(*kept)
    return NodeIndexed(
        value=torch.stack(list(vals_kept)),
        dimensions=(
            NodeDimension.from_node_infos(list(q_kept), device=device),
            NodeDimension.from_node_infos(list(k_kept), device=device),
        ),
    )


def compute_qk_tracing(
    q_targets: NodeRefs,
    k_targets: NodeRefs,
    sources: NodeRefs,
    topk: int,
) -> QKTracingResult:
    """Role-labeled QK tracing — Q-marginal, K-marginal, and Q→K pair
    attributions — for attention-score targets.

    Two first-order VJPs decompose the bilinear score ``s = (Q·K)/attn_scale``
    into per-source Q-role and K-role contributions:

    - ``a_Q[i] = v_i · Σ_d K[d]·∂Q[d]/∂s_i`` (VJP of Q with K as cotangent)
    - ``a_K[j] = v_j · Σ_d Q[d]·∂K[d]/∂s_j`` (VJP of K with Q as cotangent)

    A merged top-k over ``[a_Q, a_K]`` per slot picks the most influential
    role-tagged sources. A single second backward driven by the role-matched
    scalars yields, per pick, a row of the bilinear tensor
    ``T_{ij} := v_i·v_j·(∂Q/∂s_i)·(∂K/∂s_j)`` — ``T[i, :]`` for a Q-pick,
    ``T[:, j]`` for a K-pick — whose argmax gives the complementary partner.
    The second-order identity holds exactly because Q and K are linear in
    sources inside the circuit-trace context (SAE decoders, ``W_Q``/``W_K``,
    detached LN, and rotary embeddings are all linear).

    Returned pairs carry a structural role label: dim-0 is always the Q-side
    source, dim-1 the K-side source, regardless of which side picked them.
    """
    assert len(q_targets.dimension) == len(k_targets.dimension)
    device = sources.device
    bwd_batch_size = sources.batch_size // topk
    n_slots_total = len(q_targets.dimension)

    def _diag(stacked: torch.Tensor) -> torch.Tensor:
        n_slots = stacked.shape[1]
        return torch.stack([stacked[s * topk : (s + 1) * topk, s] for s in range(n_slots)], dim=0)

    def _first_order(outputs: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
        return _first_order_attributions(
            torch.autograd.grad(
                outputs,
                sources.refs(),
                grad_outputs=cotangent,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            ),
            sources,
        )

    def _topk_marginal(slot_row: torch.Tensor) -> NodeIndexed[torch.Tensor]:
        values, indices = slot_row.topk(topk)
        order = [i for i in torch.argsort(values, descending=True).tolist() if values[i].item() != 0.0]
        return NodeIndexed(
            value=values[order] if order else values[:0],
            dimensions=(sources.dimension.offsets_to_nodes(indices[order] if order else indices[:0]),),
        )

    q_marginal_slots: list[NodeIndexed[torch.Tensor]] = []
    k_marginal_slots: list[NodeIndexed[torch.Tensor]] = []
    pair_slots: list[NodeIndexed[torch.Tensor]] = []

    starts = range(0, n_slots_total, bwd_batch_size)
    for start in tqdm(starts, desc="qk_tracing", disable=len(starts) <= 1):
        end = min(start + bwd_batch_size, n_slots_total)
        offsets = torch.arange(start, end, device=q_targets.dimension.device)
        q_sub = q_targets[q_targets.dimension.offsets_to_nodes(offsets)]
        k_sub = k_targets[k_targets.dimension.offsets_to_nodes(offsets)]

        q_vecs = _diag(torch.cat(q_sub.values(), dim=1))
        k_vecs = _diag(torch.cat(k_sub.values(), dim=1))
        n_slots = q_vecs.shape[0]

        a_Q = _first_order(q_vecs, k_vecs)
        a_K = _first_order(k_vecs, q_vecs)
        n_sources = a_Q.shape[1]

        slot_first_rows = torch.arange(n_slots, device=device) * topk
        merged = torch.cat([a_Q[slot_first_rows].detach(), a_K[slot_first_rows].detach()], dim=-1)
        _, pick_flat = merged.topk(topk, dim=-1)
        is_q_pick = pick_flat < n_sources
        pick_src = torch.where(is_q_pick, pick_flat, pick_flat - n_sources)

        bwd_rows = slot_first_rows.unsqueeze(1) + torch.arange(topk, device=device).unsqueeze(0)
        second_bwd_values = torch.where(is_q_pick, a_Q[bwd_rows, pick_src], a_K[bwd_rows, pick_src]).reshape(-1)
        second_attributions = _first_order_attributions(
            torch.autograd.grad(
                second_bwd_values.sum(),
                sources.refs(),
                retain_graph=True,
                allow_unused=True,
            ),
            sources,
        )

        q_marginal_slots.extend(_topk_marginal(a_Q[s * topk]) for s in range(n_slots))
        k_marginal_slots.extend(_topk_marginal(a_K[s * topk]) for s in range(n_slots))
        pair_slots.extend(
            _emit_pair_slot(
                second_attributions[s * topk : (s + 1) * topk],
                is_q_pick[s],
                pick_src[s],
                sources,
                device,
            )
            for s in range(n_slots)
        )

    outer = (q_targets.dimension,)
    return QKTracingResult(
        q_marginal=NodeIndexed(value=q_marginal_slots, dimensions=outer),
        k_marginal=NodeIndexed(value=k_marginal_slots, dimensions=outer),
        pairs=NodeIndexed(value=pair_slots, dimensions=outer),
    )
