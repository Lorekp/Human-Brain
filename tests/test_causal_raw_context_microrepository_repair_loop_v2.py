from __future__ import annotations

import inspect
from hashlib import sha256
from pathlib import Path

from experiments.r3_pilot import causal_microrepository_repair_loop_v1 as repository
from experiments.r3_pilot import causal_raw_context_microrepository_repair_loop_v2 as raw_repository
from experiments.r3_pilot import causal_recurrent_fragment_relation_policy_v2 as recurrent
from experiments.r3_pilot import common_causal_predictive_byte_chunker_v1 as chunker
from experiments.r3_pilot import common_raw_chunk_recurrent_relation_bridge_v1 as bridge
from humanbrain import typed_program

from tests import test_causal_microrepository_repair_loop_v1 as repository_fixture
from tests import test_common_causal_predictive_byte_chunker_v1 as chunk_fixture
from tests import test_common_raw_chunk_recurrent_relation_bridge_v1 as bridge_fixture


def _root_source(
    policy: recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2,
) -> tuple[bytes, str]:
    receipts = policy.relation_receipts
    program = typed_program.Program(
        receipts[0].base_program_tokens,
        policy._library._signature,
    )
    return typed_program.compile(program).source.encode("utf-8"), program.sha256


def _spec(
    root: Path,
    *,
    source: bytes,
    program_sha256: str,
    cases: tuple[list[int], ...],
    expected: tuple[int, ...],
    force_test_failure: bool = False,
) -> repository.MicroRepositoryRepairSpecV1:
    return repository_fixture._write_microrepository(
        root,
        root_source=source,
        root_program_sha256=program_sha256,
        cases=cases,
        expected=expected,
        force_test_failure=force_test_failure,
    )


def _assert_no_repository_action(
    result: raw_repository.RawContextMicroRepositoryRepairResultV2,
) -> None:
    work = result.inner_result.work
    assert work.desired_manifest_reads == 0
    assert work.terminal_selection_races == 0
    assert work.source_materializations == 0
    assert work.selected_patch_sources == 0
    assert work.patch_writes == 0
    assert work.repository_tests == 0
    assert work.commits == 0
    assert work.revisions == 0
    assert work.reverts == 0


def test_raw_context_authorizes_one_microrepository_patch_test_commit_or_revert(
    tmp_path: Path,
) -> None:
    assert not tuple(tmp_path.iterdir())
    child, exact_sources = bridge_fixture._fit_chunk_child()
    policy = bridge_fixture._fit_policy()
    fitted_chunker, motifs = bridge_fixture._fit_chunker(
        child,
        exact_sources,
        b"opaque/raw-repository/fit",
    )
    bridge_lessons = bridge_fixture._bridge_lessons(
        motifs,
        b"opaque/raw-repository/fit",
        ambiguity_control=True,
    )
    fitted_bridge = bridge.fit_common_raw_chunk_recurrent_relation_bridge_v1(
        fitted_chunker,
        policy,
        bridge_lessons,
    )
    fitted_bridge.authenticate()
    fit_roots = {lesson.stream.sha256 for lesson in bridge_lessons}
    assert not tuple(tmp_path.iterdir())

    raw_task = chunk_fixture._stream(
        ("A", "D"),
        motifs,
        b"opaque/raw-repository/fresh/task-a-d",
    )
    assert raw_task.sha256 not in fit_roots
    assert all(
        raw_task.provenance_cells != lesson.stream.provenance_cells for lesson in bridge_lessons
    )
    assert not any(
        association.source_cells == motifs["A"][-1] and association.target_cells == motifs["D"][0]
        for association in fitted_chunker.predictor_associations
    )

    fresh_cases = (
        [-10, -4, 3, 8],
        [0, 5, -5, 2],
    )
    relation_cases = {
        (-8, 7, -1),
        (6, 5, 4, -3),
        (-2, 3, 9),
        (12, -11, 0, 5),
        (-9, 2, 6),
        (5, -4, 3),
        (8, 7),
        (-1, 0, 12),
    }
    assert not ({tuple(case) for case in fresh_cases} & relation_cases)
    frozen_checkpoints = (
        child.checkpoint_sha256,
        fitted_chunker.checkpoint_sha256,
        policy.checkpoint_sha256,
        fitted_bridge.checkpoint_sha256,
        raw_task.sha256,
    )

    # The answer and repository do not exist until every cognitive checkpoint
    # and the fresh raw task stream are frozen above.
    fresh_expected = (7, 2)
    source, root_program_sha256 = _root_source(policy)
    primary_spec = _spec(
        tmp_path / "primary",
        source=source,
        program_sha256=root_program_sha256,
        cases=fresh_cases,
        expected=fresh_expected,
    )
    prepared = raw_repository.prepare_raw_context_microrepository_repair_v2(
        fitted_bridge,
        policy,
        raw_task,
        primary_spec,
    )
    state = prepared.state
    assert state.status == raw_repository.PREPARED
    assert state.bridge_checkpoint_sha256 == frozen_checkpoints[3]
    assert state.policy_checkpoint_sha256 == frozen_checkpoints[2]
    assert state.task_stream_sha256 == frozen_checkpoints[4]
    assert state.case_count == len(fresh_cases)
    assert state.raw_context_recalls == 1
    assert state.cut_count == 1
    assert state.activated_relation_cell_count == 2
    assert state.distinct_relation_cell_count == 2
    assert state.activated_relation_receipt_count == 2
    assert state.distinct_relation_receipt_count == 2
    assert state.event_assembly_reads == 7 * chunker.TRACE_LANES
    assert state.bridge_trace_equality_reads == 2 * 3 * chunker.TRACE_LANES
    assert state.bridge_relation_cell_steps == 2 * (recurrent.RELATION_COUNT + 1)
    assert state.selected_policy_actuations == 2
    assert state.committed_edits == 2
    assert state.revisions == 0
    assert state.unselected_splice_executions == 0
    assert state.desired_reads_during_prepare == 0
    assert state.source_materializations_during_prepare == 0
    assert state.patch_writes_during_prepare == 0
    assert state.repository_tests_during_prepare == 0
    assert state.added_cell_count == 0
    assert state.added_trainable_bytes == 0
    assert state.added_checkpoint_bytes == 0
    assert state.caller_chunk_span_input is False
    assert state.caller_trace_id_input is False
    assert state.caller_relation_id_or_order_input is False
    assert state.caller_edit_path_input is False
    assert state.desired_or_program_label_input is False
    assert state.raw_stream_retention_count == 0

    result = prepared.repair()
    assert result.status == repository.COMMITTED
    inner = result.inner_result
    assert inner.status == repository.COMMITTED
    assert inner.selection_status == recurrent.SELECTED
    assert inner.exact_proof_count == 1
    assert len(set(inner.applied_relation_receipt_sha256s)) == 2
    assert inner.repository_test_exit_code == 0
    assert inner.work.source_reads == 2
    assert inner.work.case_manifest_reads == 1
    assert inner.work.desired_manifest_reads == 1
    assert inner.work.terminal_selection_races == 1
    assert inner.work.source_materializations == 1
    assert inner.work.selected_patch_sources == 1
    assert inner.work.unselected_patch_sources == 0
    assert inner.work.patch_writes == 1
    assert inner.work.repository_tests == 1
    assert inner.work.consequence_policy_races == 1
    assert inner.work.commits == 1
    assert inner.work.revisions == 0
    assert inner.work.reverts == 0
    assert state.selected_policy_actuations == 2
    assert (
        prepared._inner.state.root_rollouts + inner.work.repository_consequence_roots
        <= recurrent.MAX_ROOTS
    )

    causal_controls = (
        (
            {"hypothesis_action_gate": False},
            chunker.STOP_HYPOTHESIS_ACTION_GATE,
        ),
        ({"admission_gate": False}, chunker.STOP_ADMISSION_GATE),
        ({"bridge_cell_gate": False}, bridge.STOP_BRIDGE_GATE),
        ({"relation_cell_gate": False}, recurrent.STOP_RELATION),
    )
    for ordinal, (gates, expected_status) in enumerate(causal_controls):
        control_spec = _spec(
            tmp_path / f"causal-{ordinal}",
            source=source,
            program_sha256=root_program_sha256,
            cases=fresh_cases,
            expected=fresh_expected,
        )
        stopped = raw_repository.prepare_raw_context_microrepository_repair_v2(
            fitted_bridge,
            policy,
            raw_task,
            control_spec,
            **gates,
        )
        assert stopped.state.status == expected_status
        stopped_result = stopped.repair()
        assert stopped_result.status == expected_status
        _assert_no_repository_action(stopped_result)
        assert (control_spec.repo_root / control_spec.source_relative_path).read_bytes() == source

    raw_controls = (
        (
            ("C", "D"),
            bridge.STOP_BRIDGE_AMBIGUOUS,
            b"opaque/raw-repository/fresh/ambiguous-c-d",
        ),
        (
            ("B", "D"),
            bridge.STOP_BRIDGE_UNKNOWN,
            b"opaque/raw-repository/fresh/unknown-b-d",
        ),
    )
    for ordinal, (names, expected_status, provenance) in enumerate(raw_controls):
        control_spec = _spec(
            tmp_path / f"raw-{ordinal}",
            source=source,
            program_sha256=root_program_sha256,
            cases=fresh_cases,
            expected=fresh_expected,
        )
        stopped = raw_repository.prepare_raw_context_microrepository_repair_v2(
            fitted_bridge,
            policy,
            chunk_fixture._stream(names, motifs, provenance),
            control_spec,
        )
        assert stopped.state.status == expected_status
        stopped_result = stopped.repair()
        assert stopped_result.status == expected_status
        _assert_no_repository_action(stopped_result)

    pre_attestation_drift_spec = _spec(
        tmp_path / "pre-attestation-drift",
        source=source,
        program_sha256=root_program_sha256,
        cases=fresh_cases,
        expected=fresh_expected,
    )
    pre_attestation_drift_path = (
        pre_attestation_drift_spec.repo_root / pre_attestation_drift_spec.source_relative_path
    )
    pre_attestation_drift_path.write_bytes(source + b"# drift before recall\n")
    pre_attestation_drift = raw_repository.prepare_raw_context_microrepository_repair_v2(
        fitted_bridge,
        policy,
        raw_task,
        pre_attestation_drift_spec,
    )
    assert pre_attestation_drift.state.status == repository.STOP_SOURCE_DRIFT
    assert pre_attestation_drift.state.raw_context_recalls == 0
    pre_attestation_drift_result = pre_attestation_drift.repair()
    assert pre_attestation_drift_result.status == repository.STOP_SOURCE_DRIFT
    assert pre_attestation_drift_result.inner_result.work.source_reads == 1
    assert pre_attestation_drift_result.inner_result.work.case_manifest_reads == 0
    _assert_no_repository_action(pre_attestation_drift_result)

    no_match_spec = _spec(
        tmp_path / "terminal-no-match",
        source=source,
        program_sha256=root_program_sha256,
        cases=fresh_cases,
        expected=(1, 2),
    )
    no_match = raw_repository.prepare_raw_context_microrepository_repair_v2(
        fitted_bridge,
        policy,
        raw_task,
        no_match_spec,
    ).repair()
    assert no_match.status == recurrent.STOP_NO_EXACT
    assert no_match.inner_result.work.desired_manifest_reads == 1
    assert no_match.inner_result.work.terminal_selection_races == 1
    assert no_match.inner_result.work.source_materializations == 0
    assert no_match.inner_result.work.patch_writes == 0
    assert no_match.inner_result.work.repository_tests == 0

    ambiguous_spec = _spec(
        tmp_path / "terminal-ambiguous",
        source=source,
        program_sha256=root_program_sha256,
        cases=([],),
        expected=(0,),
    )
    ambiguous = raw_repository.prepare_raw_context_microrepository_repair_v2(
        fitted_bridge,
        policy,
        raw_task,
        ambiguous_spec,
    ).repair()
    assert ambiguous.status == recurrent.STOP_MULTIPLE_EXACT
    assert ambiguous.inner_result.work.desired_manifest_reads == 1
    assert ambiguous.inner_result.work.terminal_selection_races == 1
    assert ambiguous.inner_result.work.source_materializations == 0
    assert ambiguous.inner_result.work.patch_writes == 0
    assert ambiguous.inner_result.work.repository_tests == 0

    drift_spec = _spec(
        tmp_path / "drift",
        source=source,
        program_sha256=root_program_sha256,
        cases=fresh_cases,
        expected=fresh_expected,
    )
    drifted = raw_repository.prepare_raw_context_microrepository_repair_v2(
        fitted_bridge,
        policy,
        raw_task,
        drift_spec,
    )
    drift_path = drift_spec.repo_root / drift_spec.source_relative_path
    drift_path.write_bytes(source + b"# drift\n")
    drift_result = drifted.repair()
    assert drift_result.status == repository.STOP_SOURCE_DRIFT
    assert drift_result.inner_result.work.desired_manifest_reads == 0
    assert drift_result.inner_result.work.patch_writes == 0
    assert drift_result.inner_result.work.repository_tests == 0

    failure_spec = _spec(
        tmp_path / "forced-failure",
        source=source,
        program_sha256=root_program_sha256,
        cases=fresh_cases,
        expected=fresh_expected,
        force_test_failure=True,
    )
    failed = raw_repository.prepare_raw_context_microrepository_repair_v2(
        fitted_bridge,
        policy,
        raw_task,
        failure_spec,
    ).repair()
    assert failed.status == repository.STOP_TEST_FAILURE
    assert failed.inner_result.repository_test_exit_code == 17
    assert failed.inner_result.consequence_action_key == "action:revise"
    assert failed.inner_result.work.patch_writes == 1
    assert failed.inner_result.work.repository_tests == 1
    assert failed.inner_result.work.commits == 0
    assert failed.inner_result.work.revisions == 1
    assert failed.inner_result.work.reverts == 1
    assert (failure_spec.repo_root / failure_spec.source_relative_path).read_bytes() == source

    implementation = inspect.getsource(raw_repository)
    assert implementation.count("_read_expected_manifest") == 0
    assert implementation.count(".select(") == 0
    assert implementation.count("subprocess.run") == 0
    assert implementation.count("policy.build(") == 0
    v1_repair = inspect.getsource(repository.PreparedMicroRepositoryRepairV1.repair)
    assert v1_repair.count("_read_expected_manifest(") == 1
    assert v1_repair.count("self._frontier.select(") == 1
    assert v1_repair.count("subprocess.run(") == 1
    v1_prepare = inspect.getsource(repository.prepare_microrepository_repair_v1)
    assert v1_prepare.count("policy.build(") == 1
    assert v1_prepare.count("_snapshot_microrepository_repair_v1(") == 1
    assert v1_prepare.count("_finalize_microrepository_repair_from_snapshot_v1(") == 3
    assert sha256(source).hexdigest() == primary_spec.root_source_sha256
    assert (
        child.checkpoint_sha256,
        fitted_chunker.checkpoint_sha256,
        policy.checkpoint_sha256,
        fitted_bridge.checkpoint_sha256,
        raw_task.sha256,
    ) == frozen_checkpoints
    child.authenticate()
    fitted_chunker.authenticate()
    policy.authenticate()
    fitted_bridge.authenticate()
