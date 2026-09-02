from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from humanbrain.sparse_neuronal_role_filler import (
    SparseNeuronalRoleFiller,
    SparsePointerSuperposition,
    SparseRoleFillerAblation,
)


def test_fresh_opaque_projection_is_sparse_deterministic_and_immutable() -> None:
    substrate = SparseNeuronalRoleFiller()
    first = substrate.project(b"\x00fresh\xffidentity/one")
    second = substrate.project(b"\x00fresh\xffidentity/two")

    assert first == substrate.project(b"\x00fresh\xffidentity/one")
    assert first != second
    assert first.active_cell_count == 64
    assert len(first.block_cells) == 64
    assert tuple(cell // 64 for cell in first.global_cells) == tuple(range(64))
    assert all(0 <= cell < 64 for cell in first.block_cells)
    assert substrate.counts.pointer_cells == 4_096
    assert substrate.counts.fixed_cell_count == 12_418
    assert substrate.counts.fixed_synapse_count == 1_052_672
    with pytest.raises(FrozenInstanceError):
        first.sha256 = "0" * 64  # type: ignore[misc]


def test_ordered_binding_round_trips_and_ablation_physically_stops() -> None:
    substrate = SparseNeuronalRoleFiller()
    role = substrate.project(b"opaque-a")
    filler = substrate.project(b"opaque-b")

    bound = substrate.bind(role, filler)
    reversed_bound = substrate.bind(filler, role)
    assert not bound.stopped
    assert bound.pointer is not None
    assert reversed_bound.pointer is not None
    assert bound.pointer != reversed_bound.pointer
    assert bound.trace.active_synapses == 192
    assert bound.trace.physical_cell_steps == 64 * 65 * 600

    recovered = substrate.unbind(role, bound.pointer)
    assert recovered.pointer == filler
    assert recovered.trace.population == "unbinding"

    stopped = substrate.bind(
        role,
        filler,
        ablation=SparseRoleFillerAblation.BINDING_OFF,
    )
    assert stopped.pointer is None
    assert stopped.trace.stopped
    assert stopped.trace.stop_reason == "tonic_stop"
    assert set(stopped.trace.winner_cells) == {64}
    assert stopped.trace.active_synapses == 64


def test_nested_composition_can_be_unbound_in_reverse_order() -> None:
    substrate = SparseNeuronalRoleFiller()
    outer_role = substrate.project(b"outer-lane")
    inner_role = substrate.project(b"inner-lane")
    filler = substrate.project(b"opaque-payload")

    inner = substrate.bind(inner_role, filler)
    assert inner.pointer is not None
    outer = substrate.bind(outer_role, inner.pointer)
    assert outer.pointer is not None

    recovered_inner = substrate.unbind(outer_role, outer.pointer)
    assert recovered_inner.pointer == inner.pointer
    assert recovered_inner.pointer is not None
    recovered_filler = substrate.unbind(inner_role, recovered_inner.pointer)
    assert recovered_filler.pointer == filler


def test_compact_superposition_drives_noise_tolerant_live_cleanup() -> None:
    substrate = SparseNeuronalRoleFiller()
    prototype = substrate.project(b"unseen-live-prototype")
    distractor_a = substrate.project(b"unseen-live-distractor-a")
    distractor_b = substrate.project(b"unseen-live-distractor-b")

    state = SparsePointerSuperposition.empty()
    for observation in range(5):
        donor = substrate.project(f"noise-{observation}".encode())
        noisy_blocks = tuple((observation * 11 + step) % 64 for step in range(12))
        noisy = prototype.with_block_cells(
            tuple((block, donor.block_cells[block]) for block in noisy_blocks)
        )
        state = state.accumulate(noisy)

    assert state.observation_count == 5
    assert state.storage_bytes == 8_192
    assert state.allocated_synapse_count == 4_096
    assert state.plastic_synapse_events == 320
    assert 64 <= state.nonzero_synapse_count <= 124

    decision = substrate.cleanup(
        state,
        (distractor_a, prototype, distractor_b),
    )
    assert decision.pointer == prototype
    assert decision.trace.winner_cell == 1
    assert decision.trace.live_neural_scores[1] > decision.trace.tonic_stop_score
    assert decision.trace.live_neural_scores[1] > max(
        decision.trace.live_neural_scores[0],
        decision.trace.live_neural_scores[2],
    )
    assert decision.trace.physical_cell_steps == 4 * 600
    assert decision.trace.active_synapses > 1

    with pytest.raises(FrozenInstanceError):
        state.observation_count = 0  # type: ignore[misc]


def test_ambiguous_or_missing_support_selects_physical_stop() -> None:
    substrate = SparseNeuronalRoleFiller()
    first = substrate.project(b"symmetric-one")
    second = substrate.project(b"symmetric-two")
    ambiguous = SparsePointerSuperposition.empty().accumulate(first).accumulate(second)

    collision = substrate.cleanup(ambiguous, (first, second))
    assert collision.pointer is None
    assert collision.trace.stop_reason == "collision"
    assert collision.trace.live_neural_scores[0] == collision.trace.live_neural_scores[1]
    assert collision.trace.simultaneous_spike_cells == 2
    assert collision.trace.collision_stop_spike_step is not None
    assert collision.trace.physical_cell_steps == 4 * 600

    missing = substrate.cleanup(SparsePointerSuperposition.empty(), (first, second))
    assert missing.pointer is None
    assert missing.trace.stop_reason == "tonic_stop"
    assert missing.trace.winner_cell is None
    assert missing.trace.live_neural_scores == (0, 0)
    assert missing.trace.physical_cell_steps == 3 * 600


def test_cleanup_ablation_removes_support_and_selects_tonic_stop() -> None:
    substrate = SparseNeuronalRoleFiller()
    supported = substrate.project(b"supported-live-pointer")
    other = substrate.project(b"other-live-pointer")
    state = SparsePointerSuperposition.empty().accumulate(supported)

    intact = substrate.cleanup(state, (other, supported))
    assert intact.pointer == supported

    ablated = substrate.cleanup(
        state,
        (other, supported),
        ablation=SparseRoleFillerAblation.CLEANUP_OFF,
    )
    assert ablated.pointer is None
    assert ablated.trace.stop_reason == "tonic_stop"
    assert ablated.trace.live_neural_scores == (0, 0)
    assert ablated.trace.active_synapses == 1
