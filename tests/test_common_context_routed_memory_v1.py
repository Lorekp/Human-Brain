from __future__ import annotations

import dataclasses

import pytest
from experiments.g0 import common_event_associative_memory_v1 as associative
from experiments.g0 import common_event_cells_v1 as output_projection
from experiments.g0 import common_event_cells_v2 as filler_projection
from experiments.r3_pilot import common_context_routed_memory_v1 as router
from experiments.r3_pilot import common_episodic_transform_consolidation_v2 as memory
from experiments.r3_pilot import common_one_byte_copy_relation_v2 as transforms

_COVER_VALUES = (0x00, 0x4A, 0x94, 0xDE, 0x25, 0x6B, 0xBD, 0xF3)
_QUERY_VALUE = 0x4A
_HISTORY_QUERY = b"opaque/query/history"
_CURRENT_QUERY = b"opaque/query/current"

_input_cells = filler_projection.CommonOneByteFillerCells()
_output_cells = output_projection.CommonEventCells()


def _digits(value: int) -> tuple[int, ...]:
    return tuple(
        (
            value
            // (
                filler_projection.FILLER_RADIX
                ** (filler_projection.FILLER_LANES - lane - 1)
            )
        )
        % filler_projection.FILLER_RADIX
        for lane in range(filler_projection.FILLER_LANES)
    )


def _value(digits: tuple[int, ...]) -> int:
    result = 0
    for digit in digits:
        result = result * filler_projection.FILLER_RADIX + digit
    return result


def _shift(value: int) -> int:
    return _value(tuple((digit + 1) % 4 for digit in _digits(value)))


def _teacher_cells(value: int) -> tuple[int, ...]:
    output = _output_cells.race_output(
        tuple(
            tuple(24 if candidate == digit else 0 for candidate in range(4))
            for digit in _digits(value)
        )
    )
    assert output.decision == "TOKEN"
    return output.active_cells


def _opaque(event: bytes) -> tuple[int, ...]:
    return _output_cells.project(event).trace_cells


def _relation_views(
    relation_event: bytes,
    *,
    exact_shifted: bool,
    semantic_shifted: bool,
) -> tuple[
    tuple[associative.CommonAssociationLesson, ...],
    tuple[transforms.RelationGatedOneByteLesson, ...],
    bytes,
]:
    exact_source = b"opaque/exact/source/" + relation_event
    exact_lessons = tuple(
        associative.CommonAssociationLesson(
            observation_events=(bytes((value,)), relation_event, exact_source),
            consequence_byte=bytes(
                ((_shift(value) if exact_shifted else value),)
            ),
        )
        for value in _COVER_VALUES
    )
    semantic_lessons: list[transforms.RelationGatedOneByteLesson] = []
    for context_index in range(2):
        context_cells = _opaque(
            b"opaque/semantic/context/"
            + relation_event
            + bytes((context_index,))
        )
        source_cells = _opaque(
            b"opaque/semantic/source/"
            + relation_event
            + bytes((context_index,))
        )
        for value in _COVER_VALUES:
            semantic_lessons.append(
                transforms.RelationGatedOneByteLesson(
                    input_frame=_input_cells.project_one_byte(bytes((value,))),
                    relation_cells=_opaque(relation_event),
                    context_cells=context_cells,
                    source_cells=source_cells,
                    teacher_output_cells=_teacher_cells(
                        _shift(value) if semantic_shifted else value
                    ),
                )
            )
    return exact_lessons, tuple(semantic_lessons), exact_source


def _route_lesson(
    *,
    query_event: bytes,
    relation_event: bytes,
    exact_source: bytes,
    evidence_context_event: bytes,
    evidence_source_event: bytes,
    observed_value: int,
    teacher_gate: bool = True,
    cell_gate: bool = True,
    comparator_gate: bool = True,
) -> router.PhysicalContextRetrievalLesson:
    return router.PhysicalContextRetrievalLesson(
        query_context_event=query_event,
        exact_observation_events=(
            bytes((_QUERY_VALUE,)),
            relation_event,
            exact_source,
        ),
        semantic_input_frame=_input_cells.project_one_byte(bytes((_QUERY_VALUE,))),
        relation_cells=_opaque(relation_event),
        evidence_context_cells=_opaque(evidence_context_event),
        evidence_source_cells=_opaque(evidence_source_event),
        observed_output_cells=_teacher_cells(observed_value),
        teacher_gate=teacher_gate,
        cell_gate=cell_gate,
        comparator_gate=comparator_gate,
    )


def test_context_router_transfers_stops_revises_and_never_replays() -> None:
    laws = {
        b"r1": (False, True),
        b"r2": (True, False),
        b"r3": (False, True),
        b"r4": (True, False),
        b"r5": (False, True),
        b"r6": (False, False),
    }
    exact_lessons: list[associative.CommonAssociationLesson] = []
    semantic_lessons: list[transforms.RelationGatedOneByteLesson] = []
    exact_sources: dict[bytes, bytes] = {}
    for relation_event, (exact_shifted, semantic_shifted) in laws.items():
        exact, semantic, exact_source = _relation_views(
            relation_event,
            exact_shifted=exact_shifted,
            semantic_shifted=semantic_shifted,
        )
        exact_lessons.extend(exact)
        semantic_lessons.extend(semantic)
        exact_sources[relation_event] = exact_source

    child = memory.fit_common_episodic_transform_consolidation_v2(
        tuple(exact_lessons),
        tuple(semantic_lessons),
    )
    child_root = child.checkpoint_sha256
    exact_root = child.episodic_checkpoint_sha256
    semantic_root = child.semantic_checkpoint_sha256

    history_r1 = _route_lesson(
        query_event=_HISTORY_QUERY,
        relation_event=b"r1",
        exact_source=exact_sources[b"r1"],
        evidence_context_event=b"router/context/history/r1",
        evidence_source_event=b"router/source/history/r1",
        observed_value=_QUERY_VALUE,
    )
    current_r1 = _route_lesson(
        query_event=_CURRENT_QUERY,
        relation_event=b"r1",
        exact_source=exact_sources[b"r1"],
        evidence_context_event=b"router/context/current/r1",
        evidence_source_event=b"router/source/current/r1",
        observed_value=_shift(_QUERY_VALUE),
    )
    history_r2 = _route_lesson(
        query_event=_HISTORY_QUERY,
        relation_event=b"r2",
        exact_source=exact_sources[b"r2"],
        evidence_context_event=b"router/context/history/r2",
        evidence_source_event=b"router/source/history/r2",
        observed_value=_shift(_QUERY_VALUE),
    )
    current_r2 = _route_lesson(
        query_event=_CURRENT_QUERY,
        relation_event=b"r2",
        exact_source=exact_sources[b"r2"],
        evidence_context_event=b"router/context/current/r2",
        evidence_source_event=b"router/source/current/r2",
        observed_value=_QUERY_VALUE,
    )
    fitted = router.fit_common_context_routed_memory_v1(
        child,
        (history_r1, current_r1, history_r2, current_r2),
    )
    fitted.authenticate()
    summary = fitted.training_summary
    assert summary.query_schema_count == 2
    assert summary.accepted_vote_count == 4
    assert summary.provenance_count == 4
    assert summary.excitatory_update_ticks == 4
    assert summary.inhibitory_update_ticks == 0
    assert summary.trainable_state_bytes == 104
    assert summary.provenance_state_bytes == 720
    assert summary.checkpoint_array_bytes == 824
    assert summary.added_cell_count == 65
    assert summary.router_learning_cell_steps == 31_800
    assert summary.exact_child_reported_work > 0
    assert summary.semantic_child_cell_steps > 0
    assert summary.parent_lesson_replay_count == 0
    assert summary.retained_router_parent_count == 0

    recall_args = {
        "exact_observation_events": (
            bytes((_QUERY_VALUE,)),
            b"r3",
            exact_sources[b"r3"],
        ),
        "semantic_input_frame": _input_cells.project_one_byte(
            bytes((_QUERY_VALUE,))
        ),
        "relation_cells": _opaque(b"r3"),
    }
    history = fitted.recall(_HISTORY_QUERY, **recall_args)
    current = fitted.recall(_CURRENT_QUERY, **recall_args)
    assert history.status == router.RECALLED
    assert history.selected_route == router.EXACT_ROUTE
    assert history.output.token == _QUERY_VALUE
    assert current.status == router.RECALLED
    assert current.selected_route == router.SEMANTIC_ROUTE
    assert current.output.token == _shift(_QUERY_VALUE)
    assert history.work.router_cell_steps == 44_400
    assert history.work.exact_child_reported_work > 0
    assert history.work.semantic_child_cell_steps > 0

    assert fitted.recall(
        b"opaque/query/unknown", **recall_args
    ).status == router.STOP_QUERY_UNKNOWN
    assert fitted.recall(
        _HISTORY_QUERY, router_gate=False, **recall_args
    ).status == router.STOP_ROUTE_UNSUPPORTED
    relay_off = fitted.recall(
        _HISTORY_QUERY, output_relay_gate=False, **recall_args
    )
    assert relay_off.selected_route == router.EXACT_ROUTE
    assert relay_off.status == router.STOP_OUTPUT_RELAY

    deranged_exact = fitted.recall(
        _HISTORY_QUERY,
        exact_observation_events=(
            bytes((_QUERY_VALUE,)),
            b"r3",
            b"opaque/deranged/source",
        ),
        semantic_input_frame=recall_args["semantic_input_frame"],
        relation_cells=recall_args["relation_cells"],
    )
    assert deranged_exact.selected_route == router.EXACT_ROUTE
    assert deranged_exact.status == router.STOP_SELECTED_CHILD
    semantic_gate_off = fitted.recall(
        _CURRENT_QUERY,
        semantic_relation_gate=False,
        **recall_args,
    )
    assert semantic_gate_off.selected_route == router.SEMANTIC_ROUTE
    assert semantic_gate_off.status == router.STOP_SELECTED_CHILD

    dependent = _route_lesson(
        query_event=_HISTORY_QUERY,
        relation_event=b"r3",
        exact_source=exact_sources[b"r3"],
        evidence_context_event=b"router/context/history/r1",
        evidence_source_event=b"router/source/dependent",
        observed_value=_QUERY_VALUE,
    )
    teacher_off = dataclasses.replace(
        _route_lesson(
            query_event=b"opaque/query/teacher-off",
            relation_event=b"r3",
            exact_source=exact_sources[b"r3"],
            evidence_context_event=b"router/context/teacher-off",
            evidence_source_event=b"router/source/teacher-off",
            observed_value=_QUERY_VALUE,
        ),
        teacher_gate=False,
    )
    cell_off = dataclasses.replace(teacher_off, teacher_gate=True, cell_gate=False)
    comparator_off = dataclasses.replace(
        teacher_off,
        teacher_gate=True,
        comparator_gate=False,
    )
    both_match = _route_lesson(
        query_event=b"opaque/query/both",
        relation_event=b"r6",
        exact_source=exact_sources[b"r6"],
        evidence_context_event=b"router/context/both",
        evidence_source_event=b"router/source/both",
        observed_value=_QUERY_VALUE,
    )
    neither_match = _route_lesson(
        query_event=b"opaque/query/neither",
        relation_event=b"r1",
        exact_source=exact_sources[b"r1"],
        evidence_context_event=b"router/context/neither",
        evidence_source_event=b"router/source/neither",
        observed_value=_shift(_shift(_QUERY_VALUE)),
    )
    fitted_root = fitted.checkpoint_sha256
    controlled = router.advance_common_context_routed_memory_v1(
        fitted,
        (
            history_r1,
            dependent,
            teacher_off,
            cell_off,
            comparator_off,
            both_match,
            neither_match,
        ),
    )
    controlled_summary = controlled.training_summary
    assert controlled_summary.accepted_vote_count == 4
    assert controlled_summary.duplicate_nonvote_count == 1
    assert controlled_summary.dependent_nonvote_count == 1
    assert controlled_summary.teacher_off_count == 1
    assert controlled_summary.cell_off_count == 1
    assert controlled_summary.comparator_off_count == 1
    assert controlled_summary.consequence_not_singleton_count == 2
    assert controlled_summary.query_schema_count == 2
    assert controlled_summary.provenance_count == 6
    assert controlled_summary.trainable_state_bytes == 104
    assert controlled_summary.provenance_state_bytes == 1_080
    assert controlled_summary.checkpoint_array_bytes == 1_184
    assert [row.disposition for row in controlled.provenance[-2:]] == [
        router.DUPLICATE_NO_VOTE,
        router.DEPENDENT_NO_VOTE,
    ]
    assert controlled.recall(_HISTORY_QUERY, **recall_args).output.token == _QUERY_VALUE
    assert fitted.checkpoint_sha256 == fitted_root

    contrary_r4 = _route_lesson(
        query_event=_HISTORY_QUERY,
        relation_event=b"r4",
        exact_source=exact_sources[b"r4"],
        evidence_context_event=b"router/context/revision/r4",
        evidence_source_event=b"router/source/revision/r4",
        observed_value=_QUERY_VALUE,
    )
    one_contradiction = router.advance_common_context_routed_memory_v1(
        controlled,
        (contrary_r4,),
    )
    collision = one_contradiction.recall(_HISTORY_QUERY, **recall_args)
    assert collision.status == router.STOP_ROUTE_AMBIGUOUS
    assert collision.route_raw_scores == (1, 1)
    assert collision.selected_route is None

    contrary_r5 = _route_lesson(
        query_event=_HISTORY_QUERY,
        relation_event=b"r5",
        exact_source=exact_sources[b"r5"],
        evidence_context_event=b"router/context/revision/r5",
        evidence_source_event=b"router/source/revision/r5",
        observed_value=_shift(_QUERY_VALUE),
    )
    revised = router.advance_common_context_routed_memory_v1(
        one_contradiction,
        (contrary_r5,),
    )
    revised_history = revised.recall(_HISTORY_QUERY, **recall_args)
    assert revised_history.status == router.RECALLED
    assert revised_history.selected_route == router.SEMANTIC_ROUTE
    assert revised_history.route_raw_scores == (0, 2)
    assert revised_history.output.token == _shift(_QUERY_VALUE)
    assert revised.recall(_CURRENT_QUERY, **recall_args).output.token == _shift(
        _QUERY_VALUE
    )
    assert revised.training_summary.accepted_vote_count == 6
    assert revised.training_summary.provenance_count == 8
    assert revised.training_summary.provenance_state_bytes == 1_440
    assert revised.training_summary.checkpoint_array_bytes == 1_544

    forged = dataclasses.replace(
        recall_args["semantic_input_frame"],
        sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="authentication"):
        revised.recall(
            _HISTORY_QUERY,
            exact_observation_events=recall_args["exact_observation_events"],
            semantic_input_frame=forged,
            relation_cells=recall_args["relation_cells"],
        )

    for state in (fitted, controlled, one_contradiction, revised):
        state.authenticate()
        assert state.memory_checkpoint_sha256 == child_root
        assert state.episodic_checkpoint_sha256 == exact_root
        assert state.semantic_checkpoint_sha256 == semantic_root
        assert not hasattr(state, "_parent")
    child.authenticate()
    assert child.checkpoint_sha256 == child_root
    assert child.episodic_checkpoint_sha256 == exact_root
    assert child.semantic_checkpoint_sha256 == semantic_root
    assert router.PUBLIC_INPUT_COUNT == 0
    assert router.DEV_INPUT_COUNT == 0
    assert router.M0_INPUT_COUNT == 0
    assert router.C1_INPUT_COUNT == 0
    assert router.G0_INPUT_COUNT == 0
    assert router.PATH2_INPUT_COUNT == 0
    assert router.PROTECTED_INPUT_COUNT == 0
    assert router.HELDOUT_INPUT_COUNT == 0
