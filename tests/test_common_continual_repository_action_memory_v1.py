from __future__ import annotations

import inspect
import json
import sys
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from experiments.r3_pilot import causal_microrepository_repair_loop_v2 as string_repair
from experiments.r3_pilot import causal_predicate_gated_fold_v1 as gated
from experiments.r3_pilot import causal_recurrent_fragment_relation_policy_v2 as recurrent
from experiments.r3_pilot import causal_reusable_program_fragment_library_v1 as fragments
from experiments.r3_pilot import causal_task_observation_candidate_plan_bank_v1 as plan_bank
from experiments.r3_pilot import common_continual_repository_action_memory_v1 as memory
from experiments.r3_pilot import common_microrepository_autonomous_correction_v1 as correction
from experiments.r3_pilot import common_repository_action_bank_v1 as action_bank
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import typed_semantic_packed_absolute_map_fold_v1 as absolute
from humanbrain import typed_program
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature


def _checkpoint() -> gated.FrozenPredicateGateV1:
    return gated.fit_causal_predicate_gate(
        (
            gated.observe_predicate_gate_lesson(
                accumulator_before=11,
                item=-10,
                threshold=3,
                observed_after=1,
            ),
            gated.observe_predicate_gate_lesson(
                accumulator_before=-5,
                item=7,
                threshold=-1,
                observed_after=-5,
            ),
            gated.observe_predicate_gate_lesson(
                accumulator_before=6,
                item=2,
                threshold=8,
                observed_after=8,
            ),
        ),
        feature_cell_ids=(911, 307),
        action_cell_ids=(701, 113),
    )


def _encoded_ints(
    values: Sequence[int],
) -> tuple[output_cells.TypedOutputEncoding, ...]:
    return tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.INT,
        )
        for value in values
    )


def _fit_recurrent_policy_once(
) -> recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2:
    signature = absolute.list_int_to_int_signature()
    publication = plan_bank.CausalTaskObservationCandidatePlanBankV1.build(
        _checkpoint(),
        signature,
        (([64],), ([-2],), ([4],), ([5],)),
    )
    library = fragments.fit_reusable_program_fragment_library_v1(
        publication,
        (
            _encoded_ints((64, 2, 4, 5)),
            _encoded_ints((0, -2, 4, 0)),
        ),
    )
    lesson_cases = (
        ([-8, 7, -1],),
        ([6, 5, 4, -3],),
        ([-2, 3, 9],),
        ([12, -11, 0, 5],),
        ([-9, 2, 6],),
        ([5, -4, 3],),
        ([8, 7],),
        ([-1, 0, 12],),
    )
    lesson_frontier = library.build(signature, lesson_cases)
    predicate_relation = lesson_frontier.select(
        _encoded_ints((-1, 1, 1, 0, 2, -1, 0, -1))
    )
    action_relation = lesson_frontier.select(
        _encoded_ints((9, 7, 5, 11, 11, 7, 0, 1))
    )
    return recurrent.fit_causal_recurrent_fragment_relation_policy_v2(
        library,
        (
            recurrent.FragmentRelationLesson(lesson_frontier, predicate_relation),
            recurrent.FragmentRelationLesson(lesson_frontier, action_relation),
        ),
    )


def _string_signature() -> TypedSignature:
    string = ProgramType.String()
    return TypedSignature(
        entry_point="clean_text",
        parameters=(ProgramParameter("payload", string),),
        result=string,
    )


def _build_string_frontier(
    policy: string_repair.FrozenStringRelationPolicyV2,
    signature: TypedSignature,
    visible_inputs: tuple[str, ...],
    visible_expected: tuple[str, ...],
) -> string_repair.StringRelationFrontierV2:
    frontier = policy.build(signature, tuple((value,) for value in visible_inputs))
    bank = frontier.repository_action_bank
    assert frontier.status == string_repair.READY
    assert {item.receipt_cell_support for item in bank.records} == {1, 2}
    expected = frontier.encode_visible_expected(list(visible_expected))
    assert all(item.physical_outputs == expected for item in bank.records)
    return frontier


def _build_list_frontier(
    policy: recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2,
    signature: TypedSignature,
    visible_inputs: tuple[list[int], ...],
    visible_expected: tuple[int, ...],
) -> recurrent.RecurrentFragmentRelationFrontierV2:
    frontier = policy.build(signature, tuple((value,) for value in visible_inputs))
    bank = frontier.repository_action_bank
    assert frontier.status == recurrent.READY
    assert {item.receipt_cell_support for item in bank.records} == {1, 2}
    expected = frontier.encode_visible_expected(list(visible_expected))
    assert all(item.physical_outputs == expected for item in bank.records)
    return frontier


def _stage_repository(root: Path, source_name: str, source: bytes) -> None:
    root.mkdir()
    (root / source_name).write_bytes(source)


def _control_spec(
    root: Path,
    source_name: str,
    source: bytes,
    root_program_sha256: str,
) -> correction.AutonomousCorrectionSpecV1:
    return correction.AutonomousCorrectionSpecV1(
        repo_root=root,
        source_relative_path=source_name,
        expected_manifest_relative_path="not-created.json",
        test_command=(sys.executable, "-B", "not-created.py"),
        root_program_sha256=root_program_sha256,
        root_source_sha256=sha256(source).hexdigest(),
    )


def _install_hidden_fixture(
    root: Path,
    *,
    source_name: str,
    source: bytes,
    root_program_sha256: str,
    visible_expected: Sequence[object],
    entry_point: str,
    hidden_input: object,
    hidden_expected: object,
    manifest_name: str,
    test_name: str,
) -> correction.AutonomousCorrectionSpecV1:
    case_name = f"{test_name}.case.json"
    (root / manifest_name).write_text(
        json.dumps({"expected": list(visible_expected)}, separators=(",", ":")),
        encoding="utf-8",
    )
    (root / case_name).write_text(
        json.dumps(
            {"input": hidden_input, "expected": hidden_expected},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (root / test_name).write_text(
        "from __future__ import annotations\n"
        "import json\n"
        "from pathlib import Path\n"
        "root = Path(__file__).parent\n"
        f"source = root / {source_name!r}\n"
        "namespace = {}\n"
        "exec(compile(source.read_text(encoding='utf-8'), str(source), 'exec'), namespace)\n"
        f"record = json.loads((root / {case_name!r}).read_text(encoding='utf-8'))\n"
        f"actual = namespace[{entry_point!r}](record['input'])\n"
        "if actual != record['expected']:\n"
        "    raise SystemExit(3)\n",
        encoding="utf-8",
    )
    return correction.AutonomousCorrectionSpecV1(
        repo_root=root,
        source_relative_path=source_name,
        expected_manifest_relative_path=manifest_name,
        test_command=(sys.executable, "-B", test_name),
        root_program_sha256=root_program_sha256,
        root_source_sha256=sha256(source).hexdigest(),
    )


def _memory_snapshot(
    value: memory.FrozenContinualRepositoryActionMemoryV1,
) -> tuple[str, tuple[str, ...]]:
    value.authenticate()
    return (
        value.checkpoint_sha256,
        tuple(
            sha256(array.tobytes(order="C")).hexdigest()
            for _, array in value._arrays()
        ),
    )


def _assert_preaction_stop(
    result: correction.AutonomousCorrectionResultV1,
    expected_status: str,
) -> None:
    result.authenticate()
    assert result.status == expected_status
    assert result.repository_actions == ()
    assert result.work.desired_manifest_reads == 0
    assert result.work.candidate_races == 0
    assert result.work.source_materializations == 0
    assert result.work.patch_writes == 0
    assert result.work.repository_tests == 0


def _stable_program_map(
    bank: action_bank.FrozenRepositoryActionBankV1,
) -> dict[str, str]:
    return {
        bank.action_sha256_for(record): record.program_sha256
        for record in bank.records
    }


def test_two_policy_replay_free_continual_repository_action_memory(
    tmp_path: Path,
) -> None:
    recurrent_policy = _fit_recurrent_policy_once()
    list_signature = absolute.list_int_to_int_signature()
    string_signature = _string_signature()
    string_training = string_repair.solve_string_relation_training_v2(
        recurrent_policy,
        string_signature,
        (
            ("A1e2Z9",),
            ("u3B4i8",),
            ("O5x6a0",),
            ("e7Y8U9",),
        ),
    )
    string_policy = string_repair.fit_string_relation_policy_v2(
        recurrent_policy,
        string_training,
    )

    string_root = string_training.identity_program
    list_root = typed_program.Program(
        recurrent_policy._records[0].base_program_tokens,
        list_signature,
    )
    string_source = typed_program.compile(string_root).source.encode("utf-8")
    list_source = typed_program.compile(list_root).source.encode("utf-8")
    assert string_source != list_source

    string_a_visible = ("a1b2c1", "x9y8z9", "zero00", "no-digits")
    string_a_expected = ("121", "989", "00", "")
    string_ap_visible = ("r1s0r1", "t2u3t2", "pair44", "void")
    string_ap_expected = ("101", "232", "44", "")
    list_b_visible = ([1, 2, 7], [4, 6], [5, 9], [3, 0, 8])
    list_b_expected = (3, 4, 0, 3)
    list_bp_visible = ([2, 2, 10], [1, 4, 5], [6, 7], [3, 3, 12])
    list_bp_expected = (4, 5, 0, 6)

    # All desired-free frontiers exist before any hidden consequence fixture.
    string_a = _build_string_frontier(
        string_policy,
        string_signature,
        string_a_visible,
        string_a_expected,
    )
    list_b = _build_list_frontier(
        recurrent_policy,
        list_signature,
        list_b_visible,
        list_b_expected,
    )
    string_ap = _build_string_frontier(
        string_policy,
        string_signature,
        string_ap_visible,
        string_ap_expected,
    )
    list_bp = _build_list_frontier(
        recurrent_policy,
        list_signature,
        list_bp_visible,
        list_bp_expected,
    )
    banks = tuple(
        frontier.repository_action_bank
        for frontier in (string_a, list_b, string_ap, list_bp)
    )
    string_a_bank, list_b_bank, string_ap_bank, list_bp_bank = banks
    assert string_a_bank.frontier_sha256 != string_ap_bank.frontier_sha256
    assert list_b_bank.frontier_sha256 != list_bp_bank.frontier_sha256
    assert {item.sha256 for item in string_a_bank.records}.isdisjoint(
        item.sha256 for item in string_ap_bank.records
    )
    assert {item.sha256 for item in list_b_bank.records}.isdisjoint(
        item.sha256 for item in list_bp_bank.records
    )
    assert _stable_program_map(string_a_bank) == _stable_program_map(string_ap_bank)
    assert _stable_program_map(list_b_bank) == _stable_program_map(list_bp_bank)
    assert set(string_a_bank.action_sha256s).isdisjoint(list_b_bank.action_sha256s)

    layouts = (
        (tmp_path / "coast", "solution.py", string_source),
        (tmp_path / "grove", "worker.py", list_source),
        (tmp_path / "relocated", "repaired_module.py", string_source),
        (tmp_path / "ledger", "handler.py", list_source),
    )
    for root, source_name, source in layouts:
        _stage_repository(root, source_name, source)

    spec_a = _install_hidden_fixture(
        layouts[0][0],
        source_name=layouts[0][1],
        source=string_source,
        root_program_sha256=string_root.sha256,
        visible_expected=string_a_expected,
        entry_point="clean_text",
        hidden_input="A1b2!",
        hidden_expected="21",
        manifest_name="visible.json",
        test_name="check.py",
    )
    result_a = correction.prepare_autonomous_correction_v1(
        string_a,
        spec_a,
    ).repair()
    result_a.authenticate()
    assert result_a.status == correction.COMMITTED
    assert [item.receipt_cell_support for item in result_a.attempts] == [2]
    memory_a = memory.start_continual_repository_action_memory_v1(
        result_a.repository_actions
    )
    parent_snapshot = _memory_snapshot(memory_a)

    list_context = memory.project_attested_repository_source_v1(list_source)
    foreign_bank = memory_a.recall_attested_context(list_context, list_b_bank)
    assert foreign_bank.status == memory.STOP_ACTION_BANK
    wrong_memory_result = correction.prepare_autonomous_correction_v1(
        list_b,
        _control_spec(
            layouts[1][0],
            layouts[1][1],
            list_source,
            list_root.sha256,
        ),
        action_memory=memory_a,
    ).repair()
    _assert_preaction_stop(wrong_memory_result, correction.STOP_LEARNED_RECALL)

    spec_b = _install_hidden_fixture(
        layouts[1][0],
        source_name=layouts[1][1],
        source=list_source,
        root_program_sha256=list_root.sha256,
        visible_expected=list_b_expected,
        entry_point="aggregate",
        hidden_input=[-2],
        hidden_expected=-2,
        manifest_name="observed.json",
        test_name="verify.py",
    )
    result_b = correction.prepare_autonomous_correction_v1(
        list_b,
        spec_b,
    ).repair()
    result_b.authenticate()
    assert result_b.status == correction.COMMITTED
    assert [item.receipt_cell_support for item in result_b.attempts] == [2, 1]
    assert result_b.attempts[0].rolled_back is True
    assert result_b.attempts[1].committed is True

    memory_ab = memory.advance_continual_repository_action_memory_v1(
        memory_a,
        result_b.repository_actions,
    )
    assert _memory_snapshot(memory_a) == parent_snapshot
    summary = memory_ab.training_summary
    assert summary.policy_count == 2
    assert summary.action_count == 4
    assert summary.context_count == 2
    assert summary.provenance_count == 3
    assert summary.trainable_state_bytes == 16
    assert summary.checkpoint_array_bytes == 768
    assert summary.parent_lesson_replay_count == 0
    assert summary.retained_parent_count == 0
    assert summary.raw_lesson_retention_count == 0
    assert [item.passed_cell for item in memory_ab.provenance] == [1, 0, 1]
    memory_ab_snapshot = _memory_snapshot(memory_ab)

    string_context = memory.project_attested_repository_source_v1(string_source)
    remembered_a = memory_ab.recall_attested_context(string_context, string_ap_bank)
    remembered_b = memory_ab.recall_attested_context(list_context, list_bp_bank)
    assert remembered_a.status == memory.RECALLED
    assert remembered_b.status == memory.RECALLED
    assert next(
        item.receipt_cell_support
        for item in string_ap_bank.records
        if string_ap_bank.action_sha256_for(item)
        == remembered_a.selected_candidate_action_sha256
    ) == 2
    assert next(
        item.receipt_cell_support
        for item in list_bp_bank.records
        if list_bp_bank.action_sha256_for(item)
        == remembered_b.selected_candidate_action_sha256
    ) == 1

    forged = string_ap_bank.records[0]._replace(sha256="0" * 64)
    forged_result = correction.prepare_autonomous_correction_v1(
        string_ap,
        _control_spec(
            layouts[2][0],
            layouts[2][1],
            string_source,
            string_root.sha256,
        ),
        candidate_records=(forged, string_ap_bank.records[1]),
    ).repair()
    _assert_preaction_stop(forged_result, correction.STOP_ATTESTATION)
    memory_gate_result = correction.prepare_autonomous_correction_v1(
        string_ap,
        _control_spec(
            layouts[2][0],
            layouts[2][1],
            string_source,
            string_root.sha256,
        ),
        action_memory=memory_ab,
        memory_recall_gate=False,
    ).repair()
    _assert_preaction_stop(memory_gate_result, correction.STOP_LEARNED_RECALL)
    gated_recall = memory_ab.recall_attested_context(
        string_context,
        string_ap_bank,
        candidate_gate=False,
    )
    assert gated_recall.status == memory.STOP_CANDIDATE_GATE
    candidate_gate_result = correction.prepare_autonomous_correction_v1(
        string_ap,
        _control_spec(
            layouts[2][0],
            layouts[2][1],
            string_source,
            string_root.sha256,
        ),
        action_memory=memory_ab,
        memory_candidate_gate=False,
    ).repair()
    _assert_preaction_stop(candidate_gate_result, correction.STOP_LEARNED_RECALL)

    spec_ap = _install_hidden_fixture(
        layouts[2][0],
        source_name=layouts[2][1],
        source=string_source,
        root_program_sha256=string_root.sha256,
        visible_expected=string_ap_expected,
        entry_point="clean_text",
        hidden_input="Q5r6!",
        hidden_expected="65",
        manifest_name="fresh.json",
        test_name="fresh_check.py",
    )
    result_ap = correction.prepare_autonomous_correction_v1(
        string_ap,
        spec_ap,
        candidate_records=tuple(reversed(string_ap_bank.records)),
        action_memory=memory_ab,
    ).repair()
    result_ap.authenticate()
    assert result_ap.status == correction.COMMITTED
    assert [item.receipt_cell_support for item in result_ap.attempts] == [2]
    assert result_ap.repository_actions

    spec_bp = _install_hidden_fixture(
        layouts[3][0],
        source_name=layouts[3][1],
        source=list_source,
        root_program_sha256=list_root.sha256,
        visible_expected=list_bp_expected,
        entry_point="aggregate",
        hidden_input=[-4, 2, 11],
        hidden_expected=-2,
        manifest_name="later.json",
        test_name="later_check.py",
    )
    result_bp = correction.prepare_autonomous_correction_v1(
        list_bp,
        spec_bp,
        candidate_records=tuple(reversed(list_bp_bank.records)),
        action_memory=memory_ab,
    ).repair()
    result_bp.authenticate()
    assert result_bp.status == correction.COMMITTED
    assert [item.receipt_cell_support for item in result_bp.attempts] == [1]
    assert result_bp.repository_actions

    results = (result_a, result_b, result_ap, result_bp)
    assert sum(item.work.source_materializations for item in results) == 5
    assert sum(item.work.repository_tests for item in results) == 5
    assert sum(item.work.unselected_source_materializations for item in results) == 0
    assert sum(item.work.unselected_repository_tests for item in results) == 0
    assert sum(
        frontier.source_materialization_count
        for frontier in (string_a, list_b, string_ap, list_bp)
    ) == 5
    assert _memory_snapshot(memory_ab) == memory_ab_snapshot

    forbidden_labels = {"type", "domain", "context_label", "action_label"}
    assert forbidden_labels.isdisjoint(
        inspect.signature(memory.FrozenContinualRepositoryActionMemoryV1.recall_attested_context).parameters
    )
    assert forbidden_labels.isdisjoint(
        inspect.signature(correction.prepare_autonomous_correction_v1).parameters
    )
