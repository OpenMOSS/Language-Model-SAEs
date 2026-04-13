"""Unit tests for :func:`compute_qk_pair_attribution`.

Curated linear Q and K are built from a shared source tensor so that the
closed-form bilinear pair attribution

    T_{ij} = v_i * v_j * (A[:, i] . B[:, j])

(with ``v_i = sources[row, i]``, ``A = ∂Q/∂sources``, ``B = ∂K/∂sources``) can
be compared directly against the output of
:func:`compute_qk_pair_attribution`, which is advertised to return this exact
value with **dim 0 = Q-role source and dim 1 = K-role source**.
"""

from __future__ import annotations

import torch

from lm_saes.circuits.attribution import NodeRefs, compute_qk_pair_attribution


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

    Forward-batch layout matches the real ``compute_qk_pair_attribution``
    contract: row ``s*topk + k`` carries slot ``s``'s data for its ``k``-th
    replicate. Q and K for slot ``s`` depend on the source activations at the
    same row, via ``Q[row, s, :] = A[s] @ sources[row]`` and symmetrically for
    K. Across replicates inside a slot the source values are identical (as in
    the real ``einops.repeat(tokens, 'n -> b n')`` forward), so the top-1 pair
    is deterministic and independent of ``k``.
    """
    torch.manual_seed(seed)

    A = torch.randn(n_slots, d_qk_head, n_sources)
    B = torch.randn(n_slots, d_qk_head, n_sources)
    per_slot_values = torch.randn(n_slots, n_sources)

    replicated = per_slot_values.repeat_interleave(topk, dim=0)  # (fwd_batch, n_sources)
    sources_ref = replicated.detach().clone().requires_grad_(True)

    # Slot s uses source rows in [s*topk, (s+1)*topk). For each row, compute
    # ``Q[row, s_slot, :] = A[s_slot] @ sources[row]`` but only the rows
    # belonging to slot s_slot matter for s_slot's backward — other rows'
    # contributions never flow into s_slot's gradient because they live in a
    # different forward-batch row.
    Q_per_slot: list[torch.Tensor] = []
    K_per_slot: list[torch.Tensor] = []
    for s in range(n_slots):
        Q_per_slot.append(sources_ref @ A[s].T)  # (fwd_batch, d_qk_head)
        K_per_slot.append(sources_ref @ B[s].T)
    q_ref = torch.stack(Q_per_slot, dim=1)  # (fwd_batch, n_slots, d_qk_head)
    k_ref = torch.stack(K_per_slot, dim=1)

    q_entries = [
        (
            f"slot_{s}.hook_q",
            torch.tensor([[s]], dtype=torch.long),
            q_ref,
        )
        for s in range(n_slots)
    ]
    k_entries = [
        (
            f"slot_{s}.hook_k",
            torch.tensor([[s]], dtype=torch.long),
            k_ref,
        )
        for s in range(n_slots)
    ]
    q_targets = NodeRefs.from_nodes_and_refs(q_entries)
    k_targets = NodeRefs.from_nodes_and_refs(k_entries)

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


def _expected_T(
    A_slot: torch.Tensor,
    B_slot: torch.Tensor,
    values_slot: torch.Tensor,
) -> torch.Tensor:
    """Closed-form ``T_{ij} = v_i * v_j * (A[:, i] . B[:, j])``."""
    core = A_slot.T @ B_slot  # (n_sources, n_sources)
    return values_slot.unsqueeze(1) * values_slot.unsqueeze(0) * core


def test_compute_qk_pair_attribution_matches_closed_form_bilinear():
    n_slots, topk, n_sources, d_qk_head = 3, 4, 6, 5
    q_targets, k_targets, sources, A, B, values = _build_curated_setup(
        n_slots=n_slots, topk=topk, n_sources=n_sources, d_qk_head=d_qk_head
    )

    result = compute_qk_pair_attribution(q_targets, k_targets, sources, topk=topk)

    assert len(result.value) == n_slots
    for slot_idx in range(n_slots):
        slot = result.value[slot_idx]
        expected = _expected_T(A[slot_idx], B[slot_idx], values[slot_idx])
        q_dim, k_dim = slot.dimensions
        # Every returned pair must correspond to a non-zero T_{ij} entry and
        # its value must match closed-form exactly (up to fp tolerance).
        q_nodes = list(q_dim)
        k_nodes = list(k_dim)
        assert len(q_nodes) == len(k_nodes) == len(slot.value)
        assert len(slot.value) > 0, f"slot {slot_idx}: no non-zero pairs returned"
        for q_ni, k_ni, attrib in zip(q_nodes, k_nodes, slot.value):
            # Role invariant: dim 0 is Q-role, dim 1 is K-role. Both come from
            # the single "sources" key in this curated setup.
            assert q_ni.key == "sources"
            assert k_ni.key == "sources"
            i = int(q_ni.indices[0, 0].item())
            j = int(k_ni.indices[0, 0].item())
            assert torch.allclose(attrib, expected[i, j], atol=1e-5), (
                f"slot {slot_idx} pair ({i}, {j}): got {attrib.item()}, expected {expected[i, j].item()}"
            )


def test_compute_qk_pair_attribution_recovers_global_top_pair():
    """With a hand-picked ``A`` / ``B`` the dominant pair is predictable."""
    n_sources, d_qk_head, topk = 5, 3, 2
    # Slot 0: Q picks source 2 via A, K picks source 4 via B — so the dominant
    # bilinear pair should be (Q-role=2, K-role=4).
    A0 = torch.zeros(d_qk_head, n_sources)
    A0[0, 2] = 1.0
    B0 = torch.zeros(d_qk_head, n_sources)
    B0[0, 4] = 1.0
    values0 = torch.ones(n_sources)
    values0[2] = 3.0
    values0[4] = 2.0

    A = A0.unsqueeze(0)
    B = B0.unsqueeze(0)
    values = values0.unsqueeze(0)

    replicated = values.repeat_interleave(topk, dim=0)
    sources_ref = replicated.detach().clone().requires_grad_(True)

    q_ref = (sources_ref @ A[0].T).unsqueeze(1)  # (fwd_batch, 1, d_qk_head)
    k_ref = (sources_ref @ B[0].T).unsqueeze(1)

    q_targets = NodeRefs.from_nodes_and_refs([("slot_0.hook_q", torch.tensor([[0]], dtype=torch.long), q_ref)])
    k_targets = NodeRefs.from_nodes_and_refs([("slot_0.hook_k", torch.tensor([[0]], dtype=torch.long), k_ref)])
    sources = NodeRefs.from_nodes_and_refs(
        [
            (
                "sources",
                torch.arange(n_sources, dtype=torch.long).unsqueeze(-1),
                sources_ref,
            )
        ]
    )

    result = compute_qk_pair_attribution(q_targets, k_targets, sources, topk=topk)
    slot = result.value[0]
    assert len(slot.value) >= 1
    # The dominant pair should be (Q-role=2, K-role=4) with value
    # v_2 * v_4 * (A[:,2] . B[:,4]) = 3 * 2 * 1 = 6.
    q_nodes = list(slot.dimensions[0])
    k_nodes = list(slot.dimensions[1])
    top_i = int(q_nodes[0].indices[0, 0].item())
    top_j = int(k_nodes[0].indices[0, 0].item())
    assert (top_i, top_j) == (2, 4)
    assert torch.allclose(slot.value[0], torch.tensor(6.0), atol=1e-5)


def test_compute_qk_pair_attribution_role_labels_are_structural():
    """Q-role and K-role assignments must NOT be swapped even when both have
    comparable magnitudes. We verify this by constructing a case where A and B
    pick different source indices and checking the returned pair orientation.
    """
    n_sources, d_qk_head, topk = 4, 2, 2
    A0 = torch.zeros(d_qk_head, n_sources)
    A0[0, 1] = 1.0
    B0 = torch.zeros(d_qk_head, n_sources)
    B0[0, 3] = 1.0
    values0 = torch.ones(n_sources) * 2.0

    fwd_batch = topk
    replicated = values0.unsqueeze(0).repeat(fwd_batch, 1)
    sources_ref = replicated.detach().clone().requires_grad_(True)

    q_ref = (sources_ref @ A0.T).unsqueeze(1)
    k_ref = (sources_ref @ B0.T).unsqueeze(1)

    q_targets = NodeRefs.from_nodes_and_refs([("slot_0.hook_q", torch.tensor([[0]], dtype=torch.long), q_ref)])
    k_targets = NodeRefs.from_nodes_and_refs([("slot_0.hook_k", torch.tensor([[0]], dtype=torch.long), k_ref)])
    sources = NodeRefs.from_nodes_and_refs(
        [
            (
                "sources",
                torch.arange(n_sources, dtype=torch.long).unsqueeze(-1),
                sources_ref,
            )
        ]
    )

    result = compute_qk_pair_attribution(q_targets, k_targets, sources, topk=topk)
    slot = result.value[0]
    q_nodes = list(slot.dimensions[0])
    k_nodes = list(slot.dimensions[1])
    top_i = int(q_nodes[0].indices[0, 0].item())
    top_j = int(k_nodes[0].indices[0, 0].item())
    # Q-role must be the source picked by A (=1); K-role must be the source
    # picked by B (=3). A swap would yield (3, 1) and must fail this test.
    assert top_i == 1, f"Q-role source should be 1, got {top_i}"
    assert top_j == 3, f"K-role source should be 3, got {top_j}"
