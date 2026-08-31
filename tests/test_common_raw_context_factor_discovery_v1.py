from __future__ import annotations

import dataclasses

import pytest
from experiments.g0 import common_event_associative_memory_v1 as associative
from experiments.g0 import common_event_cells_v1 as output_projection
from experiments.g0 import common_event_cells_v2 as filler_projection
from experiments.r3_pilot import common_episodic_transform_consolidation_v2 as memory
from experiments.r3_pilot import common_factorized_context_routed_memory_v2 as router
from experiments.r3_pilot import common_one_byte_copy_relation_v2 as transforms
from experiments.r3_pilot import common_raw_context_factor_discovery_v1 as discovery

_COVER_VALUES = (0x00, 0x4A, 0x94, 0xDE, 0x25, 0x6B, 0xBD, 0xF3)
_QUERY_VALUE = 0x4A

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


def _raw_lesson(
    *,
    raw_context: bytes,
    relation_event: bytes,
    exact_source: bytes,
    evidence_ordinal: int,
) -> discovery.PhysicalRawContextRouteLesson:
    return discovery.PhysicalRawContextRouteLesson(
        raw_context_frame=_output_cells.project(raw_context),
        exact_observation_events=(
            bytes((_QUERY_VALUE,)),
            relation_event,
            exact_source,
        ),
        semantic_input_frame=_input_cells.project_one_byte(bytes((_QUERY_VALUE,))),
        relation_cells=_opaque(relation_event),
        evidence_context_cells=_opaque(
            b"opaque/raw-context/context/" + bytes((evidence_ordinal,))
        ),
        evidence_source_cells=_opaque(
            b"opaque/raw-context/source/" + bytes((evidence_ordinal,))
        ),
        observed_output_cells=_teacher_cells(_shift(_QUERY_VALUE)),
    )


def test_raw_context_discovers_unseen_ab_then_collides_and_revises() -> None:
    semantic_relations = (b"s1", b"s2", b"s3", b"s4")
    exact_relations = (b"x0", b"x1", b"x2", b"x3")
    laws = {
        **{relation: (False, True) for relation in semantic_relations},
        **{relation: (True, False) for relation in exact_relations},
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
        tuple(exact_lessons), tuple(semantic_lessons)
    )
    child_root = child.checkpoint_sha256
    exact_root = child.episodic_checkpoint_sha256
    semantic_root = child.semantic_checkpoint_sha256

    # A is the recurring position-0 binding in a0/a1; B is the recurring
    # position-1 binding in 0b/1b.  The two exact branches in 00 cancel the two
    # repeated background branches; the remaining backgrounds have one vote and
    # tie tonic.  The raw A+B frame is never fitted.
    initial_lessons = (
        _raw_lesson(
            raw_context=b"a0",
            relation_event=b"s1",
            exact_source=exact_sources[b"s1"],
            evidence_ordinal=1,
        ),
        _raw_lesson(
            raw_context=b"00",
            relation_event=b"x0",
            exact_source=exact_sources[b"x0"],
            evidence_ordinal=2,
        ),
        _raw_lesson(
            raw_context=b"0b",
            relation_event=b"s2",
            exact_source=exact_sources[b"s2"],
            evidence_ordinal=3,
        ),
        _raw_lesson(
            raw_context=b"a1",
            relation_event=b"s3",
            exact_source=exact_sources[b"s3"],
            evidence_ordinal=4,
        ),
        _raw_lesson(
            raw_context=b"1b",
            relation_event=b"s4",
            exact_source=exact_sources[b"s4"],
            evidence_ordinal=5,
        ),
    )
    fitted = discovery.fit_common_raw_context_factor_discovery_v1(
        child, initial_lessons
    )
    fitted.authenticate()
    summary = fitted.training_summary
    assert summary.parent_factor_count == 6
    assert summary.accepted_factor_vote_count == 10
    assert summary.admitted_factor_count == 2
    assert summary.new_admitted_factor_count == 2
    assert summary.admission_probe_count == 6
    assert summary.admission_state_bytes == 96
    assert summary.checkpoint_array_bytes == (
        summary.route_trainable_state_bytes
        + summary.provenance_state_bytes
        + summary.admission_state_bytes
    )
    assert summary.added_discovery_cell_count == 6
    assert summary.added_composite_cell_count == (
        router.FIXED_ROUTER_CELL_COUNT + summary.parent_factor_count + 6
    )
    assert summary.parent_lesson_replay_count == 0
    assert summary.raw_lesson_retention_count == 0

    recall_args = {
        "raw_context_frame": _output_cells.project(b"ab"),
        "exact_observation_events": (
            bytes((_QUERY_VALUE,)),
            b"s1",
            exact_sources[b"s1"],
        ),
        "semantic_input_frame": _input_cells.project_one_byte(
            bytes((_QUERY_VALUE,))
        ),
        "relation_cells": _opaque(b"s1"),
    }
    novel = fitted.recall(**recall_args)
    assert novel.status == discovery.RECALLED
    assert novel.routed_recall is not None
    assert novel.routed_recall.selected_route == router.SEMANTIC_ROUTE
    assert novel.routed_recall.route_raw_scores == (0, 4)
    assert novel.routed_recall.output.token == _shift(_QUERY_VALUE)
    assert len(novel.projected_factor_cells) == 2
    assert all(
        match.status == discovery.RECALLED for match in novel.branch_matches
    )
    assert novel.record()["caller_factor_tuple_input"] is False
    assert novel.record()["host_route_label_input"] is False

    assert fitted.recall(
        **recall_args, projection_gate=False
    ).status == discovery.STOP_PROJECTION_GATE
    unknown = fitted.recall(
        **{**recall_args, "raw_context_frame": _output_cells.project(b"az")}
    )
    assert unknown.status == discovery.STOP_FACTOR_UNKNOWN
    assert unknown.routed_recall is None

    projection_off = discovery.fit_common_raw_context_factor_discovery_v1(
        child,
        (
            initial_lessons[0],
            dataclasses.replace(initial_lessons[3], projection_gate=False),
        ),
    )
    plasticity_off = discovery.fit_common_raw_context_factor_discovery_v1(
        child,
        (
            initial_lessons[0],
            dataclasses.replace(initial_lessons[3], plasticity_gate=False),
        ),
    )
    assert projection_off.training_summary.admitted_factor_count == 0
    assert plasticity_off.training_summary.admitted_factor_count == 0
    assert projection_off.recall(**recall_args).status == (
        discovery.STOP_FACTOR_UNKNOWN
    )
    assert plasticity_off.recall(**recall_args).status == (
        discovery.STOP_FACTOR_UNKNOWN
    )

    contrary_1 = _raw_lesson(
        raw_context=b"a2",
        relation_event=b"x1",
        exact_source=exact_sources[b"x1"],
        evidence_ordinal=6,
    )
    contrary_2 = _raw_lesson(
        raw_context=b"a3",
        relation_event=b"x2",
        exact_source=exact_sources[b"x2"],
        evidence_ordinal=7,
    )
    contrary_3 = _raw_lesson(
        raw_context=b"a4",
        relation_event=b"x3",
        exact_source=exact_sources[b"x3"],
        evidence_ordinal=8,
    )
    one_contrary = discovery.advance_common_raw_context_factor_discovery_v1(
        fitted, (contrary_1,)
    )
    still_semantic = one_contrary.recall(**recall_args)
    assert still_semantic.routed_recall is not None
    assert still_semantic.routed_recall.route_raw_scores == (1, 3)

    collided = discovery.advance_common_raw_context_factor_discovery_v1(
        one_contrary, (contrary_2,)
    )
    collision = collided.recall(**recall_args)
    assert collision.status == router.STOP_ROUTE_AMBIGUOUS
    assert collision.routed_recall is not None
    assert collision.routed_recall.route_raw_scores == (2, 2)

    revised = discovery.advance_common_raw_context_factor_discovery_v1(
        collided, (contrary_3,)
    )
    revised_novel = revised.recall(**recall_args)
    assert revised_novel.status == discovery.RECALLED
    assert revised_novel.routed_recall is not None
    assert revised_novel.routed_recall.selected_route == router.EXACT_ROUTE
    assert revised_novel.routed_recall.route_raw_scores == (3, 2)
    assert revised_novel.routed_recall.output.token == _QUERY_VALUE
    assert revised.training_summary.admitted_factor_count == 2

    accepted_before = revised.training_summary.accepted_factor_vote_count
    duplicate_before = revised.training_summary.duplicate_nonvote_count
    replayed = discovery.advance_common_raw_context_factor_discovery_v1(
        revised, (initial_lessons[0],)
    )
    replayed_novel = replayed.recall(**recall_args)
    assert replayed_novel.routed_recall is not None
    assert replayed_novel.routed_recall.route_raw_scores == (3, 2)
    assert replayed.training_summary.accepted_factor_vote_count == accepted_before
    assert replayed.training_summary.duplicate_nonvote_count == duplicate_before + 2
    assert replayed.training_summary.admitted_factor_count == 2
    assert len({row.sha256 for row in replayed.provenance}) == len(
        replayed.provenance
    )

    forged = dataclasses.replace(
        recall_args["raw_context_frame"], sha256="0" * 64
    )
    with pytest.raises(ValueError, match="authentication"):
        replayed.recall(**{**recall_args, "raw_context_frame": forged})

    lesson_fields = {
        field.name for field in dataclasses.fields(initial_lessons[0])
    }
    assert "factor_cells" not in lesson_fields
    assert "route" not in lesson_fields
    assert "target_route" not in lesson_fields
    for state in (
        fitted,
        one_contrary,
        collided,
        revised,
        replayed,
        projection_off,
        plasticity_off,
    ):
        state.authenticate()
        assert state.memory_checkpoint_sha256 == child_root
        assert state.episodic_checkpoint_sha256 == exact_root
        assert state.semantic_checkpoint_sha256 == semantic_root
        assert not hasattr(state, "_parent")
        assert not hasattr(state, "_raw_lessons")
        assert not hasattr(state, "_whole_context_registry")
    child.authenticate()
    assert child.checkpoint_sha256 == child_root
    assert discovery.PUBLIC_INPUT_COUNT == 0
    assert discovery.DEV_INPUT_COUNT == 0
    assert discovery.M0_INPUT_COUNT == 0
    assert discovery.C1_INPUT_COUNT == 0
    assert discovery.G0_INPUT_COUNT == 0
    assert discovery.PATH2_INPUT_COUNT == 0
    assert discovery.PROTECTED_INPUT_COUNT == 0
    assert discovery.HELDOUT_INPUT_COUNT == 0
