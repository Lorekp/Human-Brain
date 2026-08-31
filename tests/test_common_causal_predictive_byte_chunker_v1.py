from __future__ import annotations

import dataclasses
from hashlib import sha256

import pytest
from experiments.g0 import common_event_associative_memory_v1 as associative
from experiments.g0 import common_event_cells_v1 as output_projection
from experiments.g0 import common_event_cells_v2 as filler_projection
from experiments.r3_pilot import common_causal_predictive_byte_chunker_v1 as chunker
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


def _physical(tag: bytes, *, provenance: bool = False) -> tuple[int, ...]:
    offset = (
        chunker.PROVENANCE_CELL_OFFSET if provenance else chunker.INPUT_CELL_OFFSET
    )
    cells: list[int] = []
    for lane in range(chunker.TRACE_LANES):
        digest = sha256(tag + bytes((lane,))).digest()
        local = int.from_bytes(digest[:2], "little")
        cells.append(offset + lane * chunker.INPUT_LANE_BANK_CELLS + local)
    return tuple(cells)


def _motifs(namespace: bytes) -> dict[str, tuple[tuple[int, ...], ...]]:
    lengths = {"A": 2, "B": 3, "C": 4, "D": 5}
    return {
        name: tuple(
            _physical(namespace + name.encode("ascii") + bytes((ordinal,)))
            for ordinal in range(length)
        )
        for name, length in lengths.items()
    }


def _stream(
    names: tuple[str, ...],
    motifs: dict[str, tuple[tuple[int, ...], ...]],
    provenance_tag: bytes,
) -> chunker.PhysicalContinuousByteStream:
    events = tuple(event for name in names for event in motifs[name])
    return chunker.seal_physical_continuous_byte_stream(
        chunker.STREAM_BEGIN_CELLS,
        events,
        chunker.STREAM_END_CELLS,
        _physical(provenance_tag, provenance=True),
    )


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
            consequence_byte=bytes(((_shift(value) if exact_shifted else value),)),
        )
        for value in _COVER_VALUES
    )
    semantic_lessons: list[transforms.RelationGatedOneByteLesson] = []
    for context_index in range(2):
        context_cells = _opaque(
            b"opaque/semantic/context/" + relation_event + bytes((context_index,))
        )
        source_cells = _opaque(
            b"opaque/semantic/source/" + relation_event + bytes((context_index,))
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
    names: tuple[str, str],
    motifs: dict[str, tuple[tuple[int, ...], ...]],
    relation_event: bytes,
    exact_source: bytes,
    evidence_ordinal: int,
    namespace: bytes,
) -> chunker.PhysicalContinuousRouteLesson:
    return chunker.PhysicalContinuousRouteLesson(
        stream=_stream(
            names,
            motifs,
            namespace + b"/route-stream/" + bytes((evidence_ordinal,)),
        ),
        exact_observation_events=(
            bytes((_QUERY_VALUE,)),
            relation_event,
            exact_source,
        ),
        semantic_input_frame=_input_cells.project_one_byte(bytes((_QUERY_VALUE,))),
        relation_cells=_opaque(relation_event),
        evidence_context_cells=_opaque(
            namespace + b"/route-context/" + bytes((evidence_ordinal,))
        ),
        evidence_source_cells=_opaque(
            namespace + b"/route-source/" + bytes((evidence_ordinal,))
        ),
        observed_output_cells=_teacher_cells(_shift(_QUERY_VALUE)),
    )


def _corpus(
    motifs: dict[str, tuple[tuple[int, ...], ...]],
    namespace: bytes,
    exact_sources: dict[bytes, bytes],
) -> tuple[
    tuple[chunker.PhysicalContinuousByteStream, ...],
    tuple[chunker.PhysicalContinuousRouteLesson, ...],
]:
    patterns = (
        ("A", "B", "D", "C", "B"),
        ("B", "A", "C", "D", "B"),
        ("C", "B", "D", "A", "C"),
        ("A", "C", "B", "A", "B", "D", "C"),
    )
    predictor_streams = tuple(
        _stream(
            pattern,
            motifs,
            namespace + b"/predictor/" + bytes((pattern_index, copy_index)),
        )
        for pattern_index, pattern in enumerate(patterns)
        for copy_index in range(2)
    )
    # Repeating the same authenticated stream is a provenance duplicate, not a
    # third vote.  No corpus row ever contains A immediately followed by D.
    predictor_streams += (predictor_streams[0],)
    route_specs = (
        (("A", "B"), b"s1"),
        (("A", "C"), b"s2"),
        (("B", "D"), b"s3"),
        (("C", "D"), b"s4"),
        (("B", "C"), b"x0"),
        (("C", "B"), b"x0"),
    )
    lessons = tuple(
        _route_lesson(
            names,
            motifs,
            relation,
            exact_sources[relation],
            ordinal,
            namespace,
        )
        for ordinal, (names, relation) in enumerate(route_specs, 1)
    )
    return predictor_streams, lessons


def _recall_args(
    stream: chunker.PhysicalContinuousByteStream,
    exact_sources: dict[bytes, bytes],
) -> dict[str, object]:
    return {
        "stream": stream,
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


def test_causal_predictive_chunks_novel_ad_then_collision_revision() -> None:
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

    motifs = _motifs(b"opaque/original")
    predictor_streams, route_lessons = _corpus(
        motifs, b"opaque/original", exact_sources
    )
    assert all(stream.events[0] != motifs["D"][0] for stream in predictor_streams)
    assert all(lesson.stream.events[0] != motifs["D"][0] for lesson in route_lessons)
    fitted = chunker.fit_common_causal_predictive_byte_chunker_v1(
        child, predictor_streams, route_lessons
    )
    fitted.authenticate()
    summary = fitted.training_summary
    assert summary.predictor_duplicate_nonvote_count > 0
    assert summary.admitted_chunk_count == 4
    assert summary.routed_lesson_count == 6
    assert summary.trace_collision_count == 0
    assert summary.raw_stream_retention_count == 0
    assert summary.caller_span_retention_count == 0
    assert summary.parent_lesson_replay_count == 0
    assert summary.total_checkpoint_array_bytes == (
        summary.chunker_checkpoint_array_bytes
        + summary.router_checkpoint_array_bytes
    )
    assert not any(
        association.source_cells == motifs["A"][-1]
        and association.target_cells == motifs["D"][0]
        for association in fitted.predictor_associations
    )

    novel_stream = _stream(
        ("A", "D"), motifs, b"opaque/original/fresh/novel-ad"
    )
    recall_args = _recall_args(novel_stream, exact_sources)
    novel = fitted.recall(**recall_args)
    assert novel.status == chunker.RECALLED
    assert novel.cut_offsets == (len(motifs["A"]),)
    assert len(novel.admitted_chunk_traces) == 2
    assert novel.routed_recall is not None
    assert novel.routed_recall.selected_route == router.SEMANTIC_ROUTE
    assert novel.routed_recall.route_raw_scores == (0, 4)
    assert novel.routed_recall.output.token == _shift(_QUERY_VALUE)
    assert novel.record()["caller_factor_tuple_input"] is False
    assert novel.record()["caller_span_or_position_input"] is False
    assert novel.record()["host_slot_order_authority"] is False
    assert novel.record()["surprise_threshold_authority"] is False
    feature_three_rows = tuple(
        index
        for index, cell in enumerate(novel.boundary_feature_cells)
        if cell == chunker.FEATURE_CELLS[3]
    )
    assert feature_three_rows
    assert all(
        novel.boundary_action_cells[index] == chunker.ACTION_CELLS[chunker.CONTINUE]
        and novel.cut_inhibitory_consequence_spikes[index]
        for index in feature_three_rows
    )
    cut_row = novel.boundary_action_cells.index(
        chunker.ACTION_CELLS[chunker.CUT_REPLAY]
    )
    assert not novel.cut_inhibitory_consequence_spikes[cut_row]

    offset_stream = _stream(
        ("B", "A", "D", "C"), motifs, b"opaque/original/fresh/offset-badc"
    )
    offset = fitted.recall(**_recall_args(offset_stream, exact_sources))
    assert offset.status == chunker.RECALLED
    assert offset.cut_offsets == (
        len(motifs["B"]),
        len(motifs["B"]) + len(motifs["A"]),
        len(motifs["B"]) + len(motifs["A"]) + len(motifs["D"]),
    )
    assert offset.admitted_chunk_traces[1:3] == novel.admitted_chunk_traces

    assert fitted.recall(
        **recall_args, predictor_gate=False
    ).status == chunker.STOP_PREDICTOR_GATE
    assert fitted.recall(
        **recall_args, comparator_gate=False
    ).status == chunker.STOP_COMPARATOR_GATE
    assert fitted.recall(
        **recall_args, hypothesis_action_gate=False
    ).status == chunker.STOP_HYPOTHESIS_ACTION_GATE
    assert fitted.recall(
        **recall_args, reset_gate=False
    ).status == chunker.STOP_RESET_GATE
    assert fitted.recall(
        **recall_args, replay_gate=False
    ).status == chunker.STOP_REPLAY_GATE
    assert fitted.recall(
        **recall_args, admission_gate=False
    ).status == chunker.STOP_ADMISSION_GATE
    assert fitted.recall(
        **recall_args, router_gate=False
    ).status == router.STOP_ROUTE_UNSUPPORTED

    tie_stream = chunker.seal_physical_continuous_byte_stream(
        chunker.STREAM_BEGIN_CELLS,
        (motifs["C"][-1], motifs["A"][-1], motifs["B"][0], motifs["B"][1]),
        chunker.STREAM_END_CELLS,
        _physical(b"opaque/original/fresh/unidentifiable", provenance=True),
    )
    assert fitted.recall(
        **_recall_args(tie_stream, exact_sources)
    ).status == chunker.STOP_BOUNDARY_UNIDENTIFIABLE

    with pytest.raises(RuntimeError, match=chunker.STOP_PREDICTOR_GATE):
        chunker.fit_common_causal_predictive_byte_chunker_v1(
            child,
            predictor_streams,
            route_lessons,
            predictor_gate=False,
        )
    with pytest.raises(RuntimeError, match=chunker.STOP_PLASTICITY_GATE):
        chunker.fit_common_causal_predictive_byte_chunker_v1(
            child,
            predictor_streams,
            route_lessons,
            plasticity_gate=False,
        )

    contrary = tuple(
        _route_lesson(
            ("A", "B"),
            motifs,
            relation,
            exact_sources[relation],
            20 + ordinal,
            b"opaque/original",
        )
        for ordinal, relation in enumerate((b"x1", b"x2", b"x3"), 1)
    )
    one_contrary = chunker.advance_common_causal_predictive_byte_chunker_v1(
        fitted, contrary[:1]
    )
    still_semantic = one_contrary.recall(**recall_args)
    assert still_semantic.routed_recall is not None
    assert still_semantic.routed_recall.route_raw_scores == (1, 3)
    collided = chunker.advance_common_causal_predictive_byte_chunker_v1(
        one_contrary, contrary[1:2]
    )
    collision = collided.recall(**recall_args)
    assert collision.status == router.STOP_ROUTE_AMBIGUOUS
    assert collision.routed_recall is not None
    assert collision.routed_recall.route_raw_scores == (2, 2)
    revised = chunker.advance_common_causal_predictive_byte_chunker_v1(
        collided, contrary[2:]
    )
    revision = revised.recall(**recall_args)
    assert revision.status == chunker.RECALLED
    assert revision.routed_recall is not None
    assert revision.routed_recall.selected_route == router.EXACT_ROUTE
    assert revision.routed_recall.route_raw_scores == (3, 2)
    assert revision.routed_recall.output.token == _QUERY_VALUE

    duplicate_before = revised.router_training_summary.duplicate_nonvote_count
    replayed = chunker.advance_common_causal_predictive_byte_chunker_v1(
        revised, route_lessons[:1]
    )
    replayed_novel = replayed.recall(**recall_args)
    assert replayed_novel.routed_recall is not None
    assert replayed_novel.routed_recall.route_raw_scores == (3, 2)
    assert replayed.router_training_summary.duplicate_nonvote_count == (
        duplicate_before + 2
    )

    permuted_motifs = _motifs(b"opaque/permuted")
    permuted_streams, permuted_lessons = _corpus(
        permuted_motifs, b"opaque/permuted", exact_sources
    )
    permuted = chunker.fit_common_causal_predictive_byte_chunker_v1(
        child, permuted_streams, permuted_lessons
    )
    permuted_novel = permuted.recall(
        **_recall_args(
            _stream(
                ("A", "D"),
                permuted_motifs,
                b"opaque/permuted/fresh/novel-ad",
            ),
            exact_sources,
        )
    )
    assert permuted_novel.status == chunker.RECALLED
    assert permuted_novel.cut_offsets == (len(permuted_motifs["A"]),)
    assert permuted_novel.routed_recall is not None
    assert permuted_novel.routed_recall.route_raw_scores == (0, 4)

    with pytest.raises(ValueError, match="authentication"):
        dataclasses.replace(novel_stream, sha256="0" * 64)

    lesson_fields = {field.name for field in dataclasses.fields(route_lessons[0])}
    assert "factor_cells" not in lesson_fields
    assert "span" not in lesson_fields
    assert "route" not in lesson_fields
    assert "target_route" not in lesson_fields
    stream_fields = {field.name for field in dataclasses.fields(novel_stream)}
    assert "byte_count" not in stream_fields
    assert "position_cells" not in stream_fields
    assert "binding_cells" not in stream_fields
    for state in (
        fitted,
        one_contrary,
        collided,
        revised,
        replayed,
        permuted,
    ):
        state.authenticate()
        assert state.memory_checkpoint_sha256 == child_root
        assert state.episodic_checkpoint_sha256 == exact_root
        assert state.semantic_checkpoint_sha256 == semantic_root
        assert not hasattr(state, "_raw_streams")
        assert not hasattr(state, "_caller_spans")
        assert not hasattr(state, "_whole_context_registry")
        assert not hasattr(state, "_parent")
    child.authenticate()
    assert child.checkpoint_sha256 == child_root
    assert chunker.PUBLIC_INPUT_COUNT == 0
    assert chunker.DEV_INPUT_COUNT == 0
    assert chunker.M0_INPUT_COUNT == 0
    assert chunker.C1_INPUT_COUNT == 0
    assert chunker.G0_INPUT_COUNT == 0
    assert chunker.PATH2_INPUT_COUNT == 0
    assert chunker.PROTECTED_INPUT_COUNT == 0
    assert chunker.HELDOUT_INPUT_COUNT == 0
