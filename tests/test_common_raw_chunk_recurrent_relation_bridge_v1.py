from __future__ import annotations

import dataclasses
import inspect

from experiments.r3_pilot import common_causal_predictive_byte_chunker_v1 as chunker
from experiments.r3_pilot import common_episodic_transform_consolidation_v2 as memory
from experiments.r3_pilot import common_raw_chunk_recurrent_relation_bridge_v1 as bridge
from experiments.r3_pilot import causal_recurrent_fragment_relation_policy_v2 as recurrent
from experiments.r3_pilot import causal_reusable_program_fragment_library_v1 as fragments
from experiments.r3_pilot import causal_task_observation_candidate_plan_bank_v1 as bank
from experiments.r3_pilot import typed_semantic_packed_absolute_map_fold_v1 as absolute
from tests import test_causal_recurrent_fragment_relation_policy_v2 as policy_fixture
from tests import test_common_causal_predictive_byte_chunker_v1 as chunk_fixture


def _fit_chunk_child() -> tuple[
    memory.FrozenCommonEpisodicTransformMemoryV2,
    dict[bytes, bytes],
]:
    semantic_relations = (b"s1", b"s2", b"s3", b"s4")
    exact_relations = (b"x0", b"x1", b"x2", b"x3")
    laws = {
        **{relation: (False, True) for relation in semantic_relations},
        **{relation: (True, False) for relation in exact_relations},
    }
    exact_lessons = []
    semantic_lessons = []
    exact_sources: dict[bytes, bytes] = {}
    for relation_event, (exact_shifted, semantic_shifted) in laws.items():
        exact, semantic, exact_source = chunk_fixture._relation_views(
            relation_event,
            exact_shifted=exact_shifted,
            semantic_shifted=semantic_shifted,
        )
        exact_lessons.extend(exact)
        semantic_lessons.extend(semantic)
        exact_sources[relation_event] = exact_source
    return (
        memory.fit_common_episodic_transform_consolidation_v2(
            tuple(exact_lessons), tuple(semantic_lessons)
        ),
        exact_sources,
    )


def _fit_chunker(
    child: memory.FrozenCommonEpisodicTransformMemoryV2,
    exact_sources: dict[bytes, bytes],
    namespace: bytes,
) -> tuple[
    chunker.FrozenCommonCausalPredictiveByteChunkerV1,
    dict[str, tuple[tuple[int, ...], ...]],
]:
    motifs = chunk_fixture._motifs(namespace)
    predictor_streams, route_lessons = chunk_fixture._corpus(
        motifs, namespace, exact_sources
    )
    return (
        chunker.fit_common_causal_predictive_byte_chunker_v1(
            child, predictor_streams, route_lessons
        ),
        motifs,
    )


def _fit_policy() -> recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2:
    signature = absolute.list_int_to_int_signature()
    publication = bank.CausalTaskObservationCandidatePlanBankV1.build(
        policy_fixture._checkpoint(),
        signature,
        (([64],), ([-2],), ([4],), ([5],)),
    )
    library = fragments.fit_reusable_program_fragment_library_v1(
        publication,
        (
            policy_fixture._encoded((64, 2, 4, 5)),
            policy_fixture._encoded((0, -2, 4, 0)),
        ),
    )
    relation_cases = (
        ([-8, 7, -1],),
        ([6, 5, 4, -3],),
        ([-2, 3, 9],),
        ([12, -11, 0, 5],),
        ([-9, 2, 6],),
        ([5, -4, 3],),
        ([8, 7],),
        ([-1, 0, 12],),
    )
    lesson_frontier = library.build(signature, relation_cases)
    first = lesson_frontier.select(
        policy_fixture._encoded((-1, 1, 1, 0, 2, -1, 0, -1))
    )
    second = lesson_frontier.select(
        policy_fixture._encoded((9, 7, 5, 11, 11, 7, 0, 1))
    )
    return recurrent.fit_causal_recurrent_fragment_relation_policy_v2(
        library,
        (
            recurrent.FragmentRelationLesson(lesson_frontier, first),
            recurrent.FragmentRelationLesson(lesson_frontier, second),
        ),
    )


def _predicate_effect(cases: tuple[tuple[object, ...], ...]) -> tuple[int, ...]:
    return tuple(
        sum(item for item in case[0] if abs(item) < 5)  # type: ignore[union-attr]
        for case in cases
    )


def _action_effect(cases: tuple[tuple[object, ...], ...]) -> tuple[int, ...]:
    return tuple(
        sum(abs(item) for item in case[0] if item < 5)  # type: ignore[union-attr]
        for case in cases
    )


def _bridge_lessons(
    motifs: dict[str, tuple[tuple[int, ...], ...]],
    namespace: bytes,
    *,
    ambiguity_control: bool,
) -> tuple[bridge.PhysicalChunkRelationConsequenceLesson, ...]:
    cases = (
        ([-8, 7, -1],),
        ([6, 5, 4, -3],),
        ([-2, 3, 9],),
    )
    rows: list[bridge.PhysicalChunkRelationConsequenceLesson] = []
    for ordinal in range(2):
        rows.append(
            bridge.observe_physical_chunk_relation_consequence(
                chunk_fixture._stream(
                    ("A",),
                    motifs,
                    namespace + b"/bridge/a/" + bytes((ordinal,)),
                ),
                cases,
                _predicate_effect(cases),
            )
        )
        rows.append(
            bridge.observe_physical_chunk_relation_consequence(
                chunk_fixture._stream(
                    ("D",),
                    motifs,
                    namespace + b"/bridge/d/" + bytes((ordinal,)),
                ),
                cases,
                _action_effect(cases),
            )
        )
    rows.append(rows[0])
    if ambiguity_control:
        for ordinal in range(2):
            rows.append(
                bridge.observe_physical_chunk_relation_consequence(
                    chunk_fixture._stream(
                        ("C",),
                        motifs,
                        namespace + b"/bridge/c0/" + bytes((ordinal,)),
                    ),
                    cases,
                    _predicate_effect(cases),
                )
            )
            rows.append(
                bridge.observe_physical_chunk_relation_consequence(
                    chunk_fixture._stream(
                        ("C",),
                        motifs,
                        namespace + b"/bridge/c1/" + bytes((ordinal,)),
                    ),
                    cases,
                    _action_effect(cases),
                )
            )
    return tuple(rows)


def test_raw_chunks_activate_two_relations_and_drive_two_edit_rollout() -> None:
    child, exact_sources = _fit_chunk_child()
    policy = _fit_policy()
    fitted_chunker, motifs = _fit_chunker(
        child, exact_sources, b"opaque/raw-relation/original"
    )
    fitted = bridge.fit_common_raw_chunk_recurrent_relation_bridge_v1(
        fitted_chunker,
        policy,
        _bridge_lessons(
            motifs,
            b"opaque/raw-relation/original",
            ambiguity_control=True,
        ),
    )
    fitted.authenticate()
    state = fitted.state
    assert state.status == bridge.READY
    assert state.ready_relation_count == 2
    assert state.ready_trace_count == 2
    assert state.trace_count == 3
    assert state.work.duplicate_provenance_nonvotes == 1
    assert state.work.accepted_evidence_count == 8
    assert state.work.causal_relation_races == 8
    assert state.work.candidate_relation_executions == 16
    assert state.work.learned_relation_cells == 2
    assert state.work.learned_trace_cells == 3
    assert state.work.raw_stream_retention_count == 0
    assert state.work.caller_relation_id_retention_count == 0
    assert state.work.checkpoint_array_bytes == (
        fitted._evidence_traces.nbytes
        + fitted._evidence_provenances.nbytes
        + fitted._evidence_winners.nbytes
        + fitted._traces.nbytes
        + fitted.excitatory.nbytes
        + fitted.inhibitory.nbytes
    )

    fresh_cases = (
        ([-10, -4, 3, 8],),
        ([0, 5, -5, 2],),
        ([9, -1, -2, 4],),
        ([6, -7, 11],),
        ([-3, 3, 12, -4],),
    )
    raw_task = chunk_fixture._stream(
        ("A", "D"),
        motifs,
        b"opaque/raw-relation/original/fresh/unseen-a-d",
    )
    rollout = fitted.recall(raw_task, fresh_cases)
    assert rollout.status == recurrent.READY
    assert rollout.cut_offsets == (len(motifs["A"]),)
    assert len(rollout.admitted_chunk_traces) == 2
    assert len(set(rollout.activated_relation_cells)) == 2
    assert set(rollout.activated_relation_receipt_sha256s) == set(
        policy.state.relation_receipt_sha256s
    )
    assert rollout.frontier is not None
    assert rollout.frontier.state.desired_reads_during_build == 0
    assert rollout.frontier.state.source_materializations_during_build == 0
    assert rollout.frontier.source_materialization_count == 0
    assert rollout.work.activated_relation_cells == 2
    assert rollout.work.selected_actuations == 2
    assert rollout.work.committed_edits == 2
    assert rollout.work.unselected_splice_executions == 0
    assert rollout.frontier.state.work.affordance_races == 2
    assert rollout.frontier.state.work.policy_races == 5
    assert rollout.frontier.state.work.effect_races == 2

    selected = rollout.frontier.select(policy_fixture._encoded((7, 2, 7, 0, 10)))
    assert selected.status == recurrent.SELECTED
    assert selected.witness is not None
    assert len(selected.witness.applied_relation_receipt_sha256s) == 2
    assert "abs(item_0) < 5" in selected.witness.source
    assert selected.witness.source.count("abs(item_0)") == 2
    assert policy_fixture._execute(selected.witness.source, fresh_cases) == (
        7,
        2,
        7,
        0,
        10,
    )
    assert rollout.frontier.source_materialization_count == 1
    record = rollout.record()
    assert record["caller_chunk_span_input"] is False
    assert record["caller_factor_tuple_input"] is False
    assert record["caller_relation_id_or_order_input"] is False
    assert record["caller_edit_path_input"] is False
    assert record["desired_outputs_during_rollout"] is False
    assert record["program_label_input"] is False

    hypothesis_off = fitted.recall(
        raw_task, fresh_cases, hypothesis_action_gate=False
    )
    assert hypothesis_off.status == chunker.STOP_HYPOTHESIS_ACTION_GATE
    assert hypothesis_off.frontier is None
    admission_off = fitted.recall(raw_task, fresh_cases, admission_gate=False)
    assert admission_off.status == chunker.STOP_ADMISSION_GATE
    assert admission_off.frontier is None
    bridge_off = fitted.recall(raw_task, fresh_cases, bridge_cell_gate=False)
    assert bridge_off.status == bridge.STOP_BRIDGE_GATE
    assert bridge_off.frontier is None
    relation_off = fitted.recall(raw_task, fresh_cases, relation_cell_gate=False)
    assert relation_off.status == recurrent.STOP_RELATION
    assert relation_off.frontier is not None
    assert relation_off.work.selected_actuations == 0

    ambiguous = fitted.recall(
        chunk_fixture._stream(
            ("C", "D"),
            motifs,
            b"opaque/raw-relation/original/fresh/ambiguous-c-d",
        ),
        fresh_cases,
    )
    assert ambiguous.status == bridge.STOP_BRIDGE_AMBIGUOUS
    assert ambiguous.frontier is None
    unknown = fitted.recall(
        chunk_fixture._stream(
            ("B", "D"),
            motifs,
            b"opaque/raw-relation/original/fresh/unknown-b-d",
        ),
        fresh_cases,
    )
    assert unknown.status == bridge.STOP_BRIDGE_UNKNOWN
    assert unknown.frontier is None

    remapped_chunker, remapped_motifs = _fit_chunker(
        child, exact_sources, b"opaque/raw-relation/remapped"
    )
    remapped = bridge.fit_common_raw_chunk_recurrent_relation_bridge_v1(
        remapped_chunker,
        policy,
        _bridge_lessons(
            remapped_motifs,
            b"opaque/raw-relation/remapped",
            ambiguity_control=False,
        ),
    )
    remapped_rollout = remapped.recall(
        chunk_fixture._stream(
            ("A", "D"),
            remapped_motifs,
            b"opaque/raw-relation/remapped/fresh/unseen-a-d",
        ),
        fresh_cases,
    )
    assert remapped_rollout.status == recurrent.READY
    assert remapped_rollout.frontier is not None
    remapped_selected = remapped_rollout.frontier.select(
        policy_fixture._encoded((7, 2, 7, 0, 10))
    )
    assert remapped_selected.status == recurrent.SELECTED
    assert remapped_selected.witness is not None
    assert remapped_selected.witness.program.sha256 == selected.witness.program.sha256
    assert remapped.checkpoint_sha256 != fitted.checkpoint_sha256

    lesson_fields = {
        field.name for field in dataclasses.fields(_bridge_lessons(
            motifs,
            b"opaque/raw-relation/field-audit",
            ambiguity_control=False,
        )[0])
    }
    assert "relation" not in lesson_fields
    assert "relation_id" not in lesson_fields
    assert "relation_order" not in lesson_fields
    assert "chunk_span" not in lesson_fields
    assert "factor_tuple" not in lesson_fields
    assert "edit_path" not in lesson_fields
    assert "desired_outputs" not in lesson_fields
    recall_parameters = inspect.signature(fitted.recall).parameters
    assert tuple(recall_parameters)[:3] == ("stream", "cases", "predictor_gate")
    assert "desired_outputs" not in recall_parameters
    assert "relation_ids" not in recall_parameters
    assert "relation_order" not in recall_parameters
    assert not hasattr(fitted, "_raw_streams")
    assert not hasattr(fitted, "_caller_spans")
    assert not hasattr(fitted, "_caller_relation_ids")
    assert bridge.PUBLIC_INPUT_COUNT == 0
    assert bridge.DEV_INPUT_COUNT == 0
    assert bridge.M0_INPUT_COUNT == 0
    assert bridge.C1_INPUT_COUNT == 0
    assert bridge.G0_INPUT_COUNT == 0
    assert bridge.PATH2_INPUT_COUNT == 0
    assert bridge.PROTECTED_INPUT_COUNT == 0
    assert bridge.HELDOUT_INPUT_COUNT == 0
    child.authenticate()
    policy.authenticate()
