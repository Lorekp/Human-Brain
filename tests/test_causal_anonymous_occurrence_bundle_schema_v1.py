from __future__ import annotations

from inspect import signature
from itertools import pairwise

import pytest
from experiments.r3_pilot import (
    causal_anonymous_occurrence_bundle_schema_v1 as occurrence,
)


def _groups(*indices: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(range(4 * index, 4 * index + 4)) for index in indices
    )


def _relabel(
    nodes: tuple[tuple[int, ...], ...], *, multiplier: int, offset: int
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(sorted((multiplier * cell + offset) % 64 for cell in node))
        for node in nodes
    )


def _chain(
    nodes: tuple[tuple[int, ...], ...],
) -> tuple[occurrence.PhysicalOccurrenceEvent, ...]:
    return (
        occurrence.PhysicalOccurrenceEvent(
            occurrence.OPEN_CUE_CELL, nodes[0], nodes[0]
        ),
        *(
            occurrence.PhysicalOccurrenceEvent(
                occurrence.EXTEND_CUE_CELL, source, target
            )
            for source, target in pairwise(nodes)
        ),
    )


def _fanout(
    nodes: tuple[tuple[int, ...], ...],
) -> tuple[occurrence.PhysicalOccurrenceEvent, ...]:
    return (
        occurrence.PhysicalOccurrenceEvent(
            occurrence.OPEN_CUE_CELL, nodes[0], nodes[0]
        ),
        *(
            occurrence.PhysicalOccurrenceEvent(
                occurrence.EXTEND_CUE_CELL, nodes[0], target
            )
            for target in nodes[1:]
        ),
    )


def _closed(
    events: tuple[occurrence.PhysicalOccurrenceEvent, ...],
) -> tuple[occurrence.PhysicalOccurrenceEvent, ...]:
    return (
        *events,
        occurrence.PhysicalOccurrenceEvent(occurrence.EOS_CUE_CELL, (), ()),
    )


def test_anonymous_occurrence_bundle_learns_reusable_graph_relays() -> None:
    assert "OccurrenceFactorStaticCertificate" not in vars(occurrence)
    hidden_authority_names = {
        "desired",
        "desired_action",
        "family",
        "schema_index",
        "target",
    }
    assert hidden_authority_names.isdisjoint(
        signature(
            occurrence.fit_causal_anonymous_occurrence_bundle_schema_v1
        ).parameters
    )
    assert hidden_authority_names.isdisjoint(
        signature(
            occurrence.FrozenAnonymousOccurrenceBundleSchemaV1.recognize
        ).parameters
    )

    a3 = _chain(_groups(0, 1, 2))
    b3 = _fanout(_groups(3, 4, 5))
    a4 = _chain(_groups(6, 7, 8, 9))
    checkpoint = occurrence.fit_causal_anonymous_occurrence_bundle_schema_v1(
        a3 + b3 + _closed(a4)
    )

    summary = checkpoint.training_summary
    assert summary.schema_assignments == (0, 1, 0)
    assert summary.allocated_schema_count == 2
    assert summary.excitatory_update_ticks == 13
    assert summary.inhibitory_update_ticks == 39
    assert summary.maximum_excitatory_tick == 3
    assert summary.maximum_inhibitory_tick == 3
    assert summary.action_policy_state_bytes == 160
    assert summary.persistent_state_bytes == 178
    assert summary.work.input_event_count == 11
    assert summary.work.bundle_count == 3
    assert summary.work.replay_event_count == 2
    assert summary.work.relation_equality_comparison_count == 30
    assert summary.work.consequence_equality_comparison_count == 12
    assert summary.work.validation_equality_comparison_count == 3
    assert summary.work.relation_race_count == 13
    assert summary.work.schema_race_count == 3
    assert summary.work.action_recall_race_count == 11
    assert summary.work.action_intervention_race_count == 13
    assert summary.work.completion_race_count == 3
    assert summary.work.total_physical_cell_steps == 15_405
    assert checkpoint.learned_state_shapes == (
        (2, 10, 4),
        (2, 10, 4),
        (2,),
        (8,),
        (8,),
    )
    assert occurrence.NEW_CELL_COUNT == 19
    assert occurrence.TOTAL_CELL_COUNT == 69_227
    assert occurrence.PERSISTENT_STATE_BYTES == 178
    assert occurrence.PUBLIC_TRAIN_INPUT_COUNT == 0
    assert occurrence.DEV_INPUT_COUNT == 0
    assert occurrence.M0_INPUT_COUNT == 0
    assert occurrence.C1_INPUT_COUNT == 0
    assert occurrence.G0_INPUT_COUNT == 0
    assert occurrence.PATH2_INPUT_COUNT == 0
    assert occurrence.PROTECTED_INPUT_COUNT == 0

    b4_nodes = _groups(10, 11, 12, 13)
    b4 = checkpoint.recognize(_closed(_fanout(b4_nodes)))
    b4_binding = occurrence.require_occurrence_graph_schema(b4)
    assert b4.status == occurrence.READY
    assert b4_binding.schema_index == 1
    assert b4_binding.schema_cell_id == occurrence.SCHEMA_CELL_OFFSET + 1
    assert b4_binding.selected_actions == (
        occurrence.OPEN_ROOT_ACTION,
        occurrence.ATTACH_SIBLING_ACTION,
        occurrence.ATTACH_SIBLING_ACTION,
        occurrence.ATTACH_SIBLING_ACTION,
        occurrence.COMPLETE_ACTION,
    )
    assert b4_binding.root_cells == b4_nodes[0]
    assert b4_binding.source_to_carrier == tuple(
        (node, index) for index, node in enumerate(b4_nodes)
    )
    assert b4_binding.ordered_edges == ((0, 1), (0, 2), (0, 3))
    assert b4_binding.occupancy == (3, 0, 0, 0)
    assert (b4_binding.root_arity, b4_binding.max_depth) == (3, 1)
    assert b4_binding.certificate_sha256 == b4.sha256

    a5_nodes = _relabel(_groups(0, 1, 2, 3, 4), multiplier=3, offset=1)
    a5 = checkpoint.recognize(_closed(_chain(a5_nodes)))
    a5_binding = occurrence.require_occurrence_graph_schema(a5)
    assert a5_binding.schema_index == 0
    assert a5_binding.selected_actions == (
        occurrence.OPEN_ROOT_ACTION,
        occurrence.ATTACH_DESCENDANT_ACTION,
        occurrence.ATTACH_DESCENDANT_ACTION,
        occurrence.ATTACH_DESCENDANT_ACTION,
        occurrence.ATTACH_DESCENDANT_ACTION,
        occurrence.COMPLETE_ACTION,
    )
    assert a5_binding.ordered_edges == ((0, 1), (1, 2), (2, 3), (3, 4))
    assert a5_binding.occupancy == (1, 1, 1, 1, 0)
    assert (a5_binding.root_arity, a5_binding.max_depth) == (1, 4)

    relabeled_nodes = _relabel(
        _groups(6, 7, 8, 9), multiplier=5, offset=9
    )
    relabeled = checkpoint.recognize(_closed(_chain(relabeled_nodes)))
    relabeled_binding = occurrence.require_occurrence_graph_schema(relabeled)
    assert relabeled_binding.schema_index == 0
    assert relabeled_binding.root_cells == relabeled_nodes[0]
    assert relabeled_binding.ordered_edges == ((0, 1), (1, 2), (2, 3))
    assert relabeled_binding.source_to_carrier != tuple(
        (node, index) for index, node in enumerate(_groups(6, 7, 8, 9))
    )
    assert checkpoint.recognize(_closed(_chain(relabeled_nodes))).sha256 == (
        relabeled.sha256
    )

    mixed_nodes = _relabel(_groups(0, 1, 2, 3), multiplier=7, offset=3)
    mixed_stream = _closed(
        (
            occurrence.PhysicalOccurrenceEvent(
                occurrence.OPEN_CUE_CELL, mixed_nodes[0], mixed_nodes[0]
            ),
            occurrence.PhysicalOccurrenceEvent(
                occurrence.EXTEND_CUE_CELL, mixed_nodes[0], mixed_nodes[1]
            ),
            occurrence.PhysicalOccurrenceEvent(
                occurrence.EXTEND_CUE_CELL, mixed_nodes[1], mixed_nodes[2]
            ),
            occurrence.PhysicalOccurrenceEvent(
                occurrence.EXTEND_CUE_CELL, mixed_nodes[0], mixed_nodes[3]
            ),
        )
    )
    mixed = checkpoint.recognize(mixed_stream)
    assert mixed.status == occurrence.STOP_TONIC
    assert mixed.schema_index is None

    relation_off = checkpoint.recognize(
        _closed(_fanout(b4_nodes)), relation_cell_gate=False
    )
    assert relation_off.status == occurrence.STOP_RELATION_ZERO
    assert relation_off.schema_index is None

    action_off = checkpoint.recognize(
        _closed(_fanout(b4_nodes)), action_cell_gate=False
    )
    assert action_off.status == occurrence.STOP_ACTION_ZERO
    assert action_off.schema_index is None

    with pytest.raises(occurrence.OccurrenceBundleFitStop) as replay_stop:
        occurrence.fit_causal_anonymous_occurrence_bundle_schema_v1(
            a3 + b3 + _closed(a4), replay_gate=False
        )
    assert replay_stop.value.status == occurrence.STOP_REPLAY
