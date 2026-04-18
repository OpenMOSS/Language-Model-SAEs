"""Unit tests for :func:`compute_qk_tracing`.

Curated linear Q and K are built from a shared source tensor so that the
closed-form bilinear pair attribution

    T_{ij} = v_i * v_j * (A[:, i] . B[:, j])

(with ``v_i = sources[row, i]``, ``A = ∂Q/∂sources``, ``B = ∂K/∂sources``) can
be compared directly against the output of :func:`compute_qk_tracing`. The
function returns a :class:`QKTracingResult` with ``q_marginal``, ``k_marginal``,
and ``pairs`` — the pairs are structurally role-labeled (dim 0 = Q-role, dim 1
= K-role).

The first-order marginals have closed forms too:

    a_Q[i] = v_i · Σ_d K[d] · A[d, i] = v_i · (K_vec · A[:, i])
    a_K[j] = v_j · Σ_d Q[d] · B[d, j] = v_j · (Q_vec · B[:, j])
"""

from __future__ import annotations

import torch

from llamascopium.circuits.attribution import NodeRefs, compute_qk_tracing


def _build_curated_setup(
    n_slots: int,
    topk: int,
    n_sources: int,
    d_qk_head: int,
    seed: int = 0,
) -> tuple[
    NodeRefs,  # q_targets
    NodeRefs,  # k_targets
    NodeRefs,  # sources
    torch.Tensor,  # A: (n_slots, d_qk_head, n_sources)
    torch.Tensor,  # B: (n_slots, d_qk_head, n_sources)
    torch.Tensor,  # source_values per slot: (n_slots, n_sources)
]:
    """Build a linear-Q, linear-K toy problem wired through :class:`NodeRefs`.

    Forward-batch layout matches the real ``compute_qk_tracing`` contract: row
    ``s*topk + k`` carries slot ``s``'s data for its ``k``-th replicate. Q and
    K for slot ``s`` depend on the source activations at the same row, via
    ``Q[row, s, :] = A[s] @ sources[row]`` and symmetrically for K. Across
    replicates inside a slot the source values are identical (as in the real
    ``einops.repeat(tokens, 'n -> b n')`` forward), so the top-1 pair is
    deterministic and independent of ``k``.
    """
    torch.manual_seed(seed)

    A = torch.randn(n_slots, d_qk_head, n_sources)
    B = torch.randn(n_slots, d_qk_head, n_sources)
    per_slot_values = torch.randn(n_slots, n_sources)

    replicated = per_slot_values.repeat_interleave(topk, dim=0)  # (fwd_batch, n_sources)
    sources_ref = replicated.detach().clone().requires_grad_(True)

    Q_per_slot: list[torch.Tensor] = []
    K_per_slot: list[torch.Tensor] = []
    for s in range(n_slots):
        Q_per_slot.append(sources_ref @ A[s].T)
        K_per_slot.append(sources_ref @ B[s].T)
    q_ref = torch.stack(Q_per_slot, dim=1)  # (fwd_batch, n_slots, d_qk_head)
    k_ref = torch.stack(K_per_slot, dim=1)

    # Mirror the real ``_retrieve_qk_vector_targets`` pattern: one key per
    # Lorsa module, all slots live inside a single NodeInfo with stacked
    # per-slot indices. Using distinct per-slot keys would route through
    # ``NodeDimension.offsets_to_nodes``'s ``torch.unique(sorted=False)`` which
    # can reorder the slot→row correspondence.
    slot_indices = torch.arange(n_slots, dtype=torch.long).unsqueeze(-1)  # (n_slots, 1)
    q_targets = NodeRefs.from_nodes_and_refs([("hook_q", slot_indices, q_ref)])
    k_targets = NodeRefs.from_nodes_and_refs([("hook_k", slot_indices, k_ref)])

    sources = NodeRefs.from_nodes_and_refs(
        [
            (
                "sources",
                torch.arange(n_sources, dtype=torch.long).unsqueeze(-1),
                sources_ref,
            )
        ]
    )

    return q_targets, k_targets, sources, A, B, per_slot_values


def _build_single_slot(
    A0: torch.Tensor,
    B0: torch.Tensor,
    values0: torch.Tensor,
    topk: int,
) -> tuple[NodeRefs, NodeRefs, NodeRefs]:
    """Pack a single-slot (A, B, values) triple into the NodeRefs contract."""
    replicated = values0.unsqueeze(0).repeat(topk, 1)
    sources_ref = replicated.detach().clone().requires_grad_(True)

    q_ref = (sources_ref @ A0.T).unsqueeze(1)  # (fwd_batch, 1, d_qk_head)
    k_ref = (sources_ref @ B0.T).unsqueeze(1)

    q_targets = NodeRefs.from_nodes_and_refs([("hook_q", torch.tensor([[0]], dtype=torch.long), q_ref)])
    k_targets = NodeRefs.from_nodes_and_refs([("hook_k", torch.tensor([[0]], dtype=torch.long), k_ref)])
    sources = NodeRefs.from_nodes_and_refs(
        [
            (
                "sources",
                torch.arange(values0.shape[0], dtype=torch.long).unsqueeze(-1),
                sources_ref,
            )
        ]
    )
    return q_targets, k_targets, sources


def _expected_T(
    A_slot: torch.Tensor,
    B_slot: torch.Tensor,
    values_slot: torch.Tensor,
) -> torch.Tensor:
    """Closed-form ``T_{ij} = v_i * v_j * (A[:, i] . B[:, j])``."""
    core = A_slot.T @ B_slot  # (n_sources, n_sources)
    return values_slot.unsqueeze(1) * values_slot.unsqueeze(0) * core


def _expected_q_marginal(
    A_slot: torch.Tensor,
    B_slot: torch.Tensor,
    values_slot: torch.Tensor,
) -> torch.Tensor:
    """``a_Q[i] = v_i · Σ_d (B_slot @ values_slot)[d] · A_slot[d, i]``."""
    K_vec = B_slot @ values_slot
    return values_slot * (K_vec @ A_slot)


def _expected_k_marginal(
    A_slot: torch.Tensor,
    B_slot: torch.Tensor,
    values_slot: torch.Tensor,
) -> torch.Tensor:
    Q_vec = A_slot @ values_slot  # (d_qk_head,)
    return values_slot * (Q_vec @ B_slot)  # (n_sources,) — v_j · (Q · B[:,j])


def test_pairs_match_closed_form_bilinear():
    n_slots, topk, n_sources, d_qk_head = 3, 4, 6, 5
    q_targets, k_targets, sources, A, B, values = _build_curated_setup(
        n_slots=n_slots, topk=topk, n_sources=n_sources, d_qk_head=d_qk_head
    )

    result = compute_qk_tracing(q_targets, k_targets, sources, topk=topk)

    assert len(result.pairs.value) == n_slots
    for slot_idx in range(n_slots):
        slot = result.pairs.value[slot_idx]
        expected = _expected_T(A[slot_idx], B[slot_idx], values[slot_idx])
        q_nodes = list(slot.dimensions[0])
        k_nodes = list(slot.dimensions[1])
        assert len(q_nodes) == len(k_nodes) == len(slot.value)
        assert len(slot.value) > 0, f"slot {slot_idx}: no non-zero pairs returned"
        for q_ni, k_ni, attrib in zip(q_nodes, k_nodes, slot.value):
            assert q_ni.key == "sources"
            assert k_ni.key == "sources"
            i = int(q_ni.indices[0, 0].item())
            j = int(k_ni.indices[0, 0].item())
            assert torch.allclose(attrib, expected[i, j], atol=1e-5), (
                f"slot {slot_idx} pair ({i}, {j}): got {attrib.item()}, expected {expected[i, j].item()}"
            )


def test_q_marginal_matches_closed_form():
    # ``topk`` is used both to replicate the forward batch (fwd_batch =
    # n_slots * topk) and to bound the marginal's output size, so both must
    # agree. Setting topk = n_sources means the marginal top-k covers every
    # source — a strong assertion.
    n_slots, n_sources, d_qk_head = 2, 5, 4
    topk = n_sources
    q_targets, k_targets, sources, A, B, values = _build_curated_setup(
        n_slots=n_slots, topk=topk, n_sources=n_sources, d_qk_head=d_qk_head, seed=7
    )

    result = compute_qk_tracing(q_targets, k_targets, sources, topk=topk)

    for slot_idx in range(n_slots):
        slot = result.q_marginal.value[slot_idx]
        expected_a_Q = _expected_q_marginal(A[slot_idx], B[slot_idx], values[slot_idx])
        q_nodes = list(slot.dimensions[0])
        for node, attrib in zip(q_nodes, slot.value):
            idx = int(node.indices[0, 0].item())
            assert torch.allclose(attrib, expected_a_Q[idx], atol=1e-5), (
                f"slot {slot_idx} q_marginal source {idx}: got {attrib.item()}, expected {expected_a_Q[idx].item()}"
            )


def test_k_marginal_matches_closed_form():
    n_slots, n_sources, d_qk_head = 2, 5, 4
    topk = n_sources
    q_targets, k_targets, sources, A, B, values = _build_curated_setup(
        n_slots=n_slots, topk=topk, n_sources=n_sources, d_qk_head=d_qk_head, seed=11
    )

    result = compute_qk_tracing(q_targets, k_targets, sources, topk=topk)

    for slot_idx in range(n_slots):
        slot = result.k_marginal.value[slot_idx]
        expected_a_K = _expected_k_marginal(A[slot_idx], B[slot_idx], values[slot_idx])
        k_nodes = list(slot.dimensions[0])
        for node, attrib in zip(k_nodes, slot.value):
            idx = int(node.indices[0, 0].item())
            assert torch.allclose(attrib, expected_a_K[idx], atol=1e-5), (
                f"slot {slot_idx} k_marginal source {idx}: got {attrib.item()}, expected {expected_a_K[idx].item()}"
            )


def test_recovers_global_top_pair():
    """With a hand-picked ``A`` / ``B`` the dominant pair is predictable."""
    n_sources, d_qk_head, topk = 5, 3, 2
    A0 = torch.zeros(d_qk_head, n_sources)
    A0[0, 2] = 1.0
    B0 = torch.zeros(d_qk_head, n_sources)
    B0[0, 4] = 1.0
    values0 = torch.ones(n_sources)
    values0[2] = 3.0
    values0[4] = 2.0

    q_targets, k_targets, sources = _build_single_slot(A0, B0, values0, topk)
    result = compute_qk_tracing(q_targets, k_targets, sources, topk=topk)
    slot = result.pairs.value[0]
    assert len(slot.value) >= 1
    q_nodes = list(slot.dimensions[0])
    k_nodes = list(slot.dimensions[1])
    top_i = int(q_nodes[0].indices[0, 0].item())
    top_j = int(k_nodes[0].indices[0, 0].item())
    assert (top_i, top_j) == (2, 4)
    # v_2 · v_4 · 1 = 3·2·1 = 6
    assert torch.allclose(slot.value[0], torch.tensor(6.0), atol=1e-5)


def test_role_labels_are_structural():
    """Q-role and K-role assignments must NOT be swapped even when both sides
    share comparable magnitudes.
    """
    n_sources, d_qk_head, topk = 4, 2, 2
    A0 = torch.zeros(d_qk_head, n_sources)
    A0[0, 1] = 1.0
    B0 = torch.zeros(d_qk_head, n_sources)
    B0[0, 3] = 1.0
    values0 = torch.ones(n_sources) * 2.0

    q_targets, k_targets, sources = _build_single_slot(A0, B0, values0, topk)
    result = compute_qk_tracing(q_targets, k_targets, sources, topk=topk)
    slot = result.pairs.value[0]
    q_nodes = list(slot.dimensions[0])
    k_nodes = list(slot.dimensions[1])
    top_i = int(q_nodes[0].indices[0, 0].item())
    top_j = int(k_nodes[0].indices[0, 0].item())
    # Q-role must be the source picked by A (=1); K-role by B (=3).
    assert top_i == 1, f"Q-role source should be 1, got {top_i}"
    assert top_j == 3, f"K-role source should be 3, got {top_j}"


def test_k_dominant_pair_recovered_via_k_side_pick():
    """Regression guard for the merged top-k pick over [a_Q, a_K].

    We construct a case where the globally-best pair ``(i*, j*)`` has its K
    side ``j*`` dominating the K-marginal but its Q side ``i*`` buried in the
    Q-marginal ranking. A Q-only pick would miss ``i*``; the merged pick
    surfaces ``j*`` as a K-role pick, whose second-backward column ``T[:, j*]``
    then lets us recover ``i*`` as the argmax partner.

    Setup (d_qk_head = 1 so everything is scalar-per-channel):

    - ``A[0, :] = [1, 1, 1, 5]`` — Q channel 0 loads source 3 most heavily.
    - ``B[0, :] = [20, 0, 0, 0]`` — K channel 0 loads source 0 exclusively.
    - ``values = [1, 1, 1, 1]``.

    Then:

    - ``a_Q[i] = v_i · K[0] · A[0, i]``: depends on the fully-contracted
      ``K_vec[0] = Σ_s B[0, s]·v_s = 20``. So ``a_Q = [20, 20, 20, 100]``.
      Top-1 Q is source 3 — source 0 is NOT in the top-1 of Q-marginal at
      ``topk=1``.
    - ``a_K[j] = v_j · Q[0] · B[0, j]`` with ``Q[0] = Σ_s A[0, s]·v_s = 8``:
      ``a_K = [160, 0, 0, 0]``. Top K is source 0 with value 160.
    - Merged = ``[20, 20, 20, 100, 160, 0, 0, 0]``. With ``topk=1`` the pick
      is source 0 as a K-role source (flat index 4 ≥ n_sources=4).
    - ``T_{ij} = v_i·v_j·A[0, i]·B[0, j]``: non-zero only when j=0, i.e.
      column T[:, 0] = ``[20, 20, 20, 100]``. Argmax over i is source 3.
    - Expected emitted pair: ``(Q-role=3, K-role=0, value=100)``.

    A buggy Q-only implementation would pick i=3 as the Q-pick and then its
    row T[3, :] has argmax at j=0 with value 100 — yielding ``(3, 0, 100)``
    by coincidence. So to make this a real guard, we also lower A[0, 3] so
    the Q-marginal top-1 is NOT source 3, and lower the partner recovery so
    only the K-side pick path can find the best (i*, j*) pair.
    """
    topk = 1
    A0 = torch.tensor([[2.0, 2.0, 2.0, 3.0]])
    B0 = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    values0 = torch.tensor([1.0, 1.0, 1.0, 1.0])

    # Closed-form:
    #   Q_vec[0] = Σ_s A[0,s]·v_s = 2+2+2+3 = 9
    #   K_vec[0] = Σ_s B[0,s]·v_s = 10
    #   a_Q[i] = v_i · K_vec[0] · A[0, i] = 10·A[0, i]
    #          = [20, 20, 20, 30]             → top-1 Q = source 3
    #   a_K[j] = v_j · Q_vec[0] · B[0, j] = 9·B[0, j]
    #          = [90, 0, 0, 0]                → top-1 K = source 0
    #   merged = [20, 20, 20, 30, 90, 0, 0, 0]  → top-1 = flat idx 4 → K-role pick, src 0
    #
    #   T[i, j] = v_i·v_j·A[0, i]·B[0, j]
    #   Column T[:, 0] = [2·10, 2·10, 2·10, 3·10] = [20, 20, 20, 30]
    #   argmax over i = source 3, value = 30.
    #   Expected emitted pair: (Q=3, K=0, value=30).
    q_targets, k_targets, sources = _build_single_slot(A0, B0, values0, topk)
    result = compute_qk_tracing(q_targets, k_targets, sources, topk=topk)
    slot = result.pairs.value[0]
    assert len(slot.value) == 1
    q_node = list(slot.dimensions[0])[0]
    k_node = list(slot.dimensions[1])[0]
    top_i = int(q_node.indices[0, 0].item())
    top_j = int(k_node.indices[0, 0].item())
    # K-side pick was source 0; its partner via T[:, 0] argmax is source 3.
    assert (top_i, top_j) == (3, 0), f"expected (Q=3, K=0), got ({top_i}, {top_j})"
    assert torch.allclose(slot.value[0], torch.tensor(30.0), atol=1e-5)

    # Sanity: confirm the K-marginal did flag source 0 as the dominant K.
    k_marginal = result.k_marginal.value[0]
    k_nodes = list(k_marginal.dimensions[0])
    top_k_src = int(k_nodes[0].indices[0, 0].item())
    assert top_k_src == 0
    assert torch.allclose(k_marginal.value[0], torch.tensor(90.0), atol=1e-5)


def test_pair_dedup_when_same_pair_picked_from_both_sides():
    """With a single dominant ``(i*, j*)`` pair, both the Q-side and K-side
    picks will surface it — the Q-pick at ``i*`` sees ``j*`` as its partner via
    ``T[i*, :]`` and the K-pick at ``j*`` sees ``i*`` as its partner via
    ``T[:, j*]``. The emitted ``pairs`` slot must contain ``(i*, j*)`` **once**,
    not twice.
    """
    topk = 2
    # A and B both load source 1 and source 3 prominently, so a_Q and a_K both
    # rank those two highly. Merged top-2 across [a_Q, a_K] then picks one from
    # the Q side and one from the K side — but they resolve to the same pair.
    A0 = torch.tensor([[0.0, 4.0, 0.0, 0.0]])  # Q picks source 1
    B0 = torch.tensor([[0.0, 0.0, 0.0, 5.0]])  # K picks source 3
    values0 = torch.tensor([1.0, 1.0, 1.0, 1.0])

    # a_Q[i] = v_i · K_vec[0] · A[0, i] = 5 · A[0, i] = [0, 20, 0, 0]
    # a_K[j] = v_j · Q_vec[0] · B[0, j] = 4 · B[0, j] = [0, 0, 0, 20]
    # merged = [0, 20, 0, 0, 0, 0, 0, 20]  → top-2 picks: (Q, src=1) and (K, src=3)
    # T[1, :] = [0, 0, 0, 20]  → argmax j = 3, value 20 → pair (Q=1, K=3)
    # T[:, 3] = [0, 20, 0, 0]  → argmax i = 1, value 20 → pair (Q=1, K=3) — duplicate
    # After dedup: exactly one pair (Q=1, K=3, value=20).
    q_targets, k_targets, sources = _build_single_slot(A0, B0, values0, topk)
    result = compute_qk_tracing(q_targets, k_targets, sources, topk=topk)
    slot = result.pairs.value[0]

    assert len(slot.value) == 1, f"expected exactly one pair after dedup, got {len(slot.value)}"
    q_node = list(slot.dimensions[0])[0]
    k_node = list(slot.dimensions[1])[0]
    assert int(q_node.indices[0, 0].item()) == 1
    assert int(k_node.indices[0, 0].item()) == 3
    assert torch.allclose(slot.value[0], torch.tensor(20.0), atol=1e-5)
