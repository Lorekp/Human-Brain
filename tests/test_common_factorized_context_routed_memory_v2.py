from __future__ import annotations

import dataclasses

import pytest
from experiments.g0 import common_event_associative_memory_v1 as associative
from experiments.g0 import common_event_cells_v1 as output_projection
from experiments.g0 import common_event_cells_v2 as filler_projection
from experiments.r3_pilot import common_episodic_transform_consolidation_v2 as memory
from experiments.r3_pilot import common_factorized_context_routed_memory_v2 as router
from experiments.r3_pilot import common_one_byte_copy_relation_v2 as transforms

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


def _moved_factor(
    base: tuple[int, ...], lanes: tuple[int, ...], *, distance: int
) -> tuple[int, ...]:
    cells = list(base)
    for lane in lanes:
        start = (
            output_projection.TRACE_CELL_OFFSET
            + lane * output_projection.FEATURE_BANK_CELLS
        )
        cells[lane] = start + (
            (cells[lane] - start + distance)
            % output_projection.FEATURE_BANK_CELLS
        )
    return tuple(cells)


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


def _factor_lesson(
    *,
    factor_cells: tuple[tuple[int, ...], ...],
    relation_event: bytes,
    exact_source: bytes,
    evidence_ordinal: int,
    observed_value: int,
) -> router.PhysicalFactorRouteLesson:
    return router.PhysicalFactorRouteLesson(
        factor_cells=factor_cells,
        exact_observation_events=(
            bytes((_QUERY_VALUE,)),
            relation_event,
            exact_source,
        ),
        semantic_input_frame=_input_cells.project_one_byte(bytes((_QUERY_VALUE,))),
        relation_cells=_opaque(relation_event),
        evidence_context_cells=_opaque(
            b"opaque/router/context/" + bytes((evidence_ordinal,))
        ),
        evidence_source_cells=_opaque(
            b"opaque/router/source/" + bytes((evidence_ordinal,))
        ),
        observed_output_cells=_teacher_cells(observed_value),
    )


def test_factorized_router_composes_unseen_context_then_collides_and_revises() -> None:
    factor_a = _opaque(b"opaque/factor/base")
    factor_b = _moved_factor(factor_a, (0, 1), distance=1)
    ambiguous_factor = _moved_factor(factor_a, (0,), distance=1)
    unknown_factor = _moved_factor(
        factor_a, tuple(range(output_projection.TRACE_LANES)), distance=2
    )
    laws = {
        b"r1": (False, True),
        b"r2": (True, False),
        b"r3": (False, True),
        b"r4": (True, False),
        b"r5": (False, True),
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

    factor_a_semantic = _factor_lesson(
        factor_cells=(factor_a,),
        relation_event=b"r1",
        exact_source=exact_sources[b"r1"],
        evidence_ordinal=1,
        observed_value=_shift(_QUERY_VALUE),
    )
    factor_b_semantic = _factor_lesson(
        factor_cells=(factor_b,),
        relation_event=b"r3",
        exact_source=exact_sources[b"r3"],
        evidence_ordinal=2,
        observed_value=_shift(_QUERY_VALUE),
    )
    fitted = router.fit_common_factorized_context_routed_memory_v2(
        child, (factor_a_semantic, factor_b_semantic)
    )
    fitted.authenticate()
    summary = fitted.training_summary
    assert summary.factor_count == 2
    assert summary.accepted_factor_vote_count == 2
    assert summary.provenance_count == 2
    assert summary.excitatory_update_ticks == 2
    assert summary.inhibitory_update_ticks == 0
    assert summary.trainable_state_bytes == 104
    assert summary.provenance_state_bytes == 360
    assert summary.checkpoint_array_bytes == 464
    assert summary.added_cell_count == 65
    assert summary.router_learning_cell_steps == 15_000
    assert summary.parent_lesson_replay_count == 0
    assert summary.retained_router_parent_count == 0
    assert summary.raw_lesson_retention_count == 0

    recall_args = {
        "exact_observation_events": (
            bytes((_QUERY_VALUE,)),
            b"r5",
            exact_sources[b"r5"],
        ),
        "semantic_input_frame": _input_cells.project_one_byte(
            bytes((_QUERY_VALUE,))
        ),
        "relation_cells": _opaque(b"r5"),
    }
    assert fitted.recall((factor_a,), **recall_args).status == (
        router.STOP_ROUTE_AMBIGUOUS
    )
    assert fitted.recall((factor_b,), **recall_args).status == (
        router.STOP_ROUTE_AMBIGUOUS
    )

    # A and B were never presented together.  Their unordered physical traces
    # contribute one semantic tick each and jointly beat the tonic abstention.
    novel = fitted.recall((factor_a, factor_b), **recall_args)
    reordered = fitted.recall((factor_b, factor_a), **recall_args)
    assert novel.status == router.RECALLED
    assert novel.selected_route == router.SEMANTIC_ROUTE
    assert novel.route_raw_scores == (0, 2)
    assert novel.output.token == _shift(_QUERY_VALUE)
    assert novel.work.router_cell_steps == 46_200
    assert reordered.record() == novel.record()

    ambiguous = fitted.recall((ambiguous_factor,), **recall_args)
    assert ambiguous.status == router.STOP_FACTOR_AMBIGUOUS
    assert ambiguous.selected_route is None
    unknown = fitted.recall((factor_a, unknown_factor), **recall_args)
    assert unknown.status == router.STOP_FACTOR_UNKNOWN
    assert unknown.selected_route is None
    assert fitted.recall(
        (factor_a, factor_b), factor_gate=False, **recall_args
    ).status == router.STOP_FACTOR_GATE
    assert fitted.recall(
        (factor_a, factor_b), router_gate=False, **recall_args
    ).status == router.STOP_ROUTE_UNSUPPORTED
    selected_child_off = fitted.recall(
        (factor_a, factor_b), semantic_relation_gate=False, **recall_args
    )
    assert selected_child_off.selected_route == router.SEMANTIC_ROUTE
    assert selected_child_off.status == router.STOP_SELECTED_CHILD

    fitted_root = fitted.checkpoint_sha256
    factor_a_exact_1 = _factor_lesson(
        factor_cells=(factor_a,),
        relation_event=b"r2",
        exact_source=exact_sources[b"r2"],
        evidence_ordinal=3,
        observed_value=_shift(_QUERY_VALUE),
    )
    one_contradiction = router.advance_common_factorized_context_routed_memory_v2(
        fitted, (factor_a_exact_1,)
    )
    collision = one_contradiction.recall(
        (factor_a, factor_b), **recall_args
    )
    assert collision.status == router.STOP_ROUTE_AMBIGUOUS
    assert collision.route_raw_scores == (1, 1)
    assert collision.selected_route is None

    factor_a_exact_2 = _factor_lesson(
        factor_cells=(factor_a,),
        relation_event=b"r4",
        exact_source=exact_sources[b"r4"],
        evidence_ordinal=4,
        observed_value=_shift(_QUERY_VALUE),
    )
    revised = router.advance_common_factorized_context_routed_memory_v2(
        one_contradiction, (factor_a_exact_2,)
    )
    revised_novel = revised.recall((factor_b, factor_a), **recall_args)
    assert revised_novel.status == router.RECALLED
    assert revised_novel.selected_route == router.EXACT_ROUTE
    assert revised_novel.route_raw_scores == (2, 1)
    assert revised_novel.output.token == _QUERY_VALUE
    assert revised.training_summary.accepted_factor_vote_count == 4
    assert revised.training_summary.provenance_count == 4
    assert revised.training_summary.excitatory_update_ticks == 4
    assert revised.training_summary.inhibitory_update_ticks == 2
    assert revised.training_summary.provenance_state_bytes == 720
    assert revised.training_summary.checkpoint_array_bytes == 824
    assert revised.training_summary.router_learning_cell_steps == 31_800
    assert len({row.sha256 for row in revised.provenance}) == 4
    assert [row.disposition for row in revised.provenance] == [
        router.ACCEPTED_VOTE,
        router.ACCEPTED_VOTE,
        router.ACCEPTED_VOTE,
        router.ACCEPTED_VOTE,
    ]

    ablated_lesson = dataclasses.replace(
        _factor_lesson(
            factor_cells=(unknown_factor,),
            relation_event=b"r1",
            exact_source=exact_sources[b"r1"],
            evidence_ordinal=5,
            observed_value=_shift(_QUERY_VALUE),
        ),
        cell_gate=False,
    )
    controlled = router.advance_common_factorized_context_routed_memory_v2(
        revised, (ablated_lesson,)
    )
    assert controlled.training_summary.factor_count == 2
    assert controlled.training_summary.provenance_count == 4
    assert controlled.training_summary.cell_off_count == 1
    assert controlled.recall((factor_a, factor_b), **recall_args).output.token == (
        _QUERY_VALUE
    )

    forged = dataclasses.replace(
        recall_args["semantic_input_frame"], sha256="0" * 64
    )
    with pytest.raises(ValueError, match="authentication"):
        revised.recall(
            (factor_a, factor_b),
            exact_observation_events=recall_args["exact_observation_events"],
            semantic_input_frame=forged,
            relation_cells=recall_args["relation_cells"],
        )

    lesson_fields = {field.name for field in dataclasses.fields(factor_a_semantic)}
    assert "route" not in lesson_fields
    assert "target_route" not in lesson_fields
    assert fitted.checkpoint_sha256 == fitted_root
    for state in (fitted, one_contradiction, revised, controlled):
        state.authenticate()
        assert state.memory_checkpoint_sha256 == child_root
        assert state.episodic_checkpoint_sha256 == exact_root
        assert state.semantic_checkpoint_sha256 == semantic_root
        assert not hasattr(state, "_parent")
        assert not hasattr(state, "_factor_combinations")
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
