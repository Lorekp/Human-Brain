from __future__ import annotations

from experiments.r3_pilot.common_rooted_structured_event_graph_v1 import (
    project_rooted_structured_event_graph_v1,
)
from experiments.r3_pilot.common_structured_role_path_projection_v1 import (
    FACTOR_CUE_CELLS,
    RELAY_STEPS,
    TRANSIENT_NUMPY_ARRAY_BOUND_BYTES_PER_VERTEX,
    project_structured_role_paths_v1,
)


def _graph_a():
    return project_rooted_structured_event_graph_v1(
        (b"A" * 16, b"A" * 16, b"D" * 16),
        carrier_payloads=(b"0" * 16, b"1" * 16, b"2" * 16, b"3" * 16),
        root_carrier_indices=(3,),
        boundary_carrier_indices=(0,),
        incidences=(
            (1, 0, 11),
            (2, 0, 12),
            (4, 1, 7),
            (5, 2, 8),
            (6, 3, 9),
        ),
    )


def _renumbered_graph():
    return project_rooted_structured_event_graph_v1(
        (b"D" * 16, b"A" * 16, b"A" * 16),
        carrier_payloads=(b"2" * 16, b"3" * 16, b"0" * 16, b"1" * 16),
        root_carrier_indices=(1,),
        boundary_carrier_indices=(2,),
        incidences=(
            (0, 2, 202),
            (3, 2, 201),
            (4, 1, 203),
            (5, 0, 204),
            (6, 3, 205),
        ),
    )


def test_depth_three_cell_roles_retain_decoys_and_binary_deduplicate() -> None:
    graph = _graph_a()
    projected = project_structured_role_paths_v1(graph)
    projected.authenticate()

    assert projected.source_event_count == 3
    assert projected.factor_count == 2
    assert projected.collapsed_duplicate_count == 1
    assert projected.role_path_count == 2
    assert all(factor.presence_cell == 1 for factor in projected.factors)
    assert all(len(factor.cue_cells) == FACTOR_CUE_CELLS for factor in projected.factors)

    factors = {
        factor.event_root.semantic_sha256: factor for factor in projected.factors
    }
    reachable_root = graph.event_roots[0].semantic_sha256
    reachable = factors[reachable_root]
    decoy_root = graph.event_roots[2].semantic_sha256
    decoy = factors[decoy_root]
    assert reachable.role_path.relay_spike_bits == 1 << 3
    assert decoy.role_path.relay_spike_bits == 0

    # The relay path itself is invariant to absolute vertex numbering, event
    # order, carrier order, and incidence ordinals.  Only primitive topology
    # and anonymous carrier/event kinds can affect the factor cue.
    renumbered = project_structured_role_paths_v1(_renumbered_graph())
    renumbered.authenticate()
    assert renumbered.factors == projected.factors

    assert projected.capacity.relay_cell_count == 7 * 7
    assert projected.capacity.relay_synapse_count == 5 * 7
    assert projected.capacity.transient_numpy_array_bound_bytes == (
        7 * TRANSIENT_NUMPY_ARRAY_BOUND_BYTES_PER_VERTEX
    )
    assert projected.capacity.checkpoint_array_bytes == 0
    assert projected.work.boundary_seed_cell_steps == 7 * RELAY_STEPS
    assert projected.work.relay_cell_steps == 7 * 7 * RELAY_STEPS
    assert projected.work.reverse_synaptic_reads == 5 * 7
    assert projected.work.total > projected.work.relay_cell_steps
