from __future__ import annotations

import inspect
import json
import sys
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import numpy as np
from experiments.r3_pilot import causal_microrepository_repair_loop_v2 as repair
from experiments.r3_pilot import causal_predicate_gated_fold_v1 as gated
from experiments.r3_pilot import causal_recurrent_fragment_relation_policy_v2 as recurrent
from experiments.r3_pilot import causal_reusable_program_fragment_library_v1 as fragments
from experiments.r3_pilot import causal_task_observation_candidate_plan_bank_v1 as bank
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import typed_semantic_packed_absolute_map_fold_v1 as absolute
from humanbrain import typed_program
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature


def _checkpoint() -> gated.FrozenPredicateGateV1:
    lessons = (
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
    )
    return gated.fit_causal_predicate_gate(
        lessons,
        feature_cell_ids=(911, 307),
        action_cell_ids=(701, 113),
    )


def _encoded_ints(values: Sequence[int]) -> tuple[output_cells.TypedOutputEncoding, ...]:
    return tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.INT,
        )
        for value in values
    )


def _encoded_strings(
    values: Sequence[str],
) -> tuple[output_cells.TypedOutputEncoding, ...]:
    return tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.STRING,
        )
        for value in values
    )


def _base_policy() -> recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2:
    signature = absolute.list_int_to_int_signature()
    publication = bank.CausalTaskObservationCandidatePlanBankV1.build(
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
    first = lesson_frontier.select(_encoded_ints((-1, 1, 1, 0, 2, -1, 0, -1)))
    second = lesson_frontier.select(_encoded_ints((9, 7, 5, 11, 11, 7, 0, 1)))
    return recurrent.fit_causal_recurrent_fragment_relation_policy_v2(
        library,
        (
            recurrent.FragmentRelationLesson(lesson_frontier, first),
            recurrent.FragmentRelationLesson(lesson_frontier, second),
        ),
    )


def _string_signature() -> TypedSignature:
    string = ProgramType.String()
    return TypedSignature(
        entry_point="clean_text",
        parameters=(ProgramParameter("payload", string),),
        result=string,
    )


def _write_repository(
    root: Path,
    *,
    root_source: bytes,
    root_program_sha256: str,
    cases: tuple[str, ...],
    expected: tuple[str, ...],
    force_test_failure: bool = False,
) -> repair.MicroRepositoryRepairSpecV2:
    root.mkdir()
    (root / "solution.py").write_bytes(root_source)
    (root / "cases.json").write_text(
        json.dumps({"cases": cases}, separators=(",", ":")),
        encoding="utf-8",
    )
    (root / "expected.json").write_text(
        json.dumps({"expected": expected}, separators=(",", ":")),
        encoding="utf-8",
    )
    forced_exit = 17 if force_test_failure else 0
    (root / "repo_test.py").write_text(
        "from __future__ import annotations\n"
        "import json\n"
        "from pathlib import Path\n"
        "from solution import clean_text\n"
        "root = Path(__file__).parent\n"
        "cases = json.loads((root / 'cases.json').read_text())['cases']\n"
        "expected = json.loads((root / 'expected.json').read_text())['expected']\n"
        "actual = [clean_text(value) for value in cases]\n"
        "if actual != expected:\n"
        "    raise SystemExit(3)\n"
        f"raise SystemExit({forced_exit})\n",
        encoding="utf-8",
    )
    return repair.MicroRepositoryRepairSpecV2(
        repo_root=root,
        source_relative_path="solution.py",
        case_manifest_relative_path="cases.json",
        expected_manifest_relative_path="expected.json",
        test_command=(sys.executable, "-B", "repo_test.py"),
        root_program_sha256=root_program_sha256,
        root_source_sha256=sha256(root_source).hexdigest(),
    )


def _assert_no_write_stop(result: repair.MicroRepositoryRepairResultV2) -> None:
    assert result.work.selected_patch_sources == 0
    assert result.work.unselected_patch_sources == 0
    assert result.work.patched_file_count == 0
    assert result.work.patch_writes == 0
    assert result.work.unselected_patch_writes == 0
    assert result.work.repository_tests == 0
    assert result.work.unselected_repository_tests == 0
    assert result.work.consequence_policy_races == 0
    assert result.work.commits == 0
    assert result.work.revisions == 0
    assert result.work.reverts == 0


def test_string_relations_transfer_to_one_fresh_microrepository(tmp_path: Path) -> None:
    base = _base_policy()
    signature = _string_signature()
    train_cases = (
        ("A1e2Z9",),
        ("u3B4i8",),
        ("O5x6a0",),
        ("e7Y8U9",),
    )
    solved = repair.solve_string_relation_training_v2(base, signature, train_cases)
    policy = repair.fit_string_relation_policy_v2(base, solved)
    reversed_policy = repair.fit_string_relation_policy_v2(
        base,
        solved,
        lesson_order=(1, 0),
    )

    assert solved.build_work.concrete_proof_witnesses == 0
    assert solved.build_work.materialized_witnesses == 0
    assert policy.state.work.frontier_relation_units == 1_080
    assert policy.state.work.frontier_predictions == 164
    assert policy.state.work.frontier_races == 380
    assert policy.state.work.train_desired_reads == 3
    assert policy.state.work.train_selected_witnesses == 3
    assert policy.state.work.selected_trace_program_executions == 4
    assert policy.state.work.selected_trace_case_evaluations == 16
    assert policy.state.work.unselected_trace_program_executions == 0
    assert policy.state.work.learned_relation_cells == 2
    assert policy.state.work.lineage_input_cells == 26
    assert policy.state.work.lineage_output_cells == 26
    assert policy.state.work.lineage_coincidence_cells == 676
    assert policy.state.work.transferred_abs_relation_bytes == 0
    assert policy.state.policy_arrays_reused_by_reference
    assert policy._policy_excitatory is base._policy_excitatory
    assert policy._policy_inhibitory is base._policy_inhibitory
    assert not np.shares_memory(policy._relation_excitatory, base._relation_excitatory)
    assert not np.shares_memory(policy._relation_inhibitory, base._relation_inhibitory)
    assert len(policy.relation_receipts) == 2
    assert {item.role for item in policy.relation_receipts} == {
        "selected-character-subsequence",
        "outer-one-hole-context",
    }
    context = next(
        item for item in policy.relation_receipts if item.actuation_kind == "one-hole-context"
    )
    assert context.context_prefix_tokens
    assert context.context_suffix_tokens == ()
    assert context.donor_tokens.count("<typed-hole>") == 1
    assert policy.state.fresh_artifact_hashes == ()

    # Fresh cases and expected cells come into existence only after both TRAIN
    # checkpoints are frozen.  The two policies share the one frozen frontier.
    fresh_cases = ("A1b2!", "007Bond", "no-digits", "9z8Y7")
    fresh_expected = ("21", "700", "", "789")
    fresh_rows = tuple((value,) for value in fresh_cases)
    reversed_frontier = reversed_policy.build(signature, fresh_rows)
    assert reversed_frontier.status == repair.READY
    reversed_selection = reversed_frontier.select(_encoded_strings(fresh_expected))
    assert reversed_selection.status == recurrent.SELECTED
    assert reversed_selection.witness is not None
    reversed_source = reversed_selection.witness.source

    root_program = solved.identity_program
    root_source = typed_program.compile(root_program).source.encode("utf-8")
    spec = _write_repository(
        tmp_path / "primary",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        cases=fresh_cases,
        expected=fresh_expected,
    )
    prepared = repair.prepare_microrepository_repair_v2(policy, signature, spec)
    assert prepared.state.status == repair.PREPARED
    assert prepared.state.case_count == len(fresh_cases)
    assert prepared.state.desired_reads_during_prepare == 0
    assert prepared.state.source_materializations_during_prepare == 0
    assert prepared.state.patch_writes_during_prepare == 0
    assert prepared.state.repository_tests_during_prepare == 0
    assert prepared.state.root_rollouts == 1
    assert prepared.state.affordance_races == 2
    assert prepared.state.policy_races == 4
    assert prepared.state.effect_races == 2
    assert prepared.state.selected_policy_actuations == 2
    assert prepared.state.selected_program_executions == 2
    assert prepared.state.committed_edits == 2
    assert prepared.state.revisions == 0
    assert prepared.state.unselected_splice_executions == 0
    assert len(prepared.state.relation_receipt_sha256s) == 2
    assert len(set(prepared.state.relation_receipt_sha256s)) == 2

    result = prepared.repair()
    assert result.status == repair.COMMITTED
    assert result.selection_status == recurrent.SELECTED
    assert result.exact_proof_count == 1
    assert len(result.applied_relation_receipt_sha256s) == 2
    assert result.consequence_action_key == "action:commit"
    assert result.repository_test_exit_code == 0
    assert result.work.source_reads == 2
    assert result.work.case_manifest_reads == 1
    assert result.work.desired_manifest_reads == 1
    assert result.work.root_source_renderings == 0
    assert result.work.terminal_selection_races == 1
    assert result.work.source_materializations == 1
    assert result.work.selected_policy_actuations == 2
    assert result.work.selected_program_executions == 2
    assert result.work.selected_patch_sources == 1
    assert result.work.unselected_patch_sources == 0
    assert result.work.patched_file_count == 1
    assert result.work.patch_writes == 1
    assert result.work.unselected_patch_writes == 0
    assert result.work.repository_consequence_roots == 1
    assert result.work.repository_tests == 1
    assert result.work.unselected_repository_tests == 0
    assert result.work.consequence_policy_races == 1
    assert result.work.unselected_splice_executions == 0
    assert result.work.commits == 1
    assert result.work.revisions == 0
    assert result.work.reverts == 0
    repaired_source = (spec.repo_root / spec.source_relative_path).read_text(
        encoding="utf-8"
    )
    assert "'0123456789'" in repaired_source
    assert repaired_source.endswith("[::-1]\n")
    assert repaired_source == reversed_source
    assert sha256(repaired_source.encode()).hexdigest() == result.source_after_sha256

    checkpoint_record = json.loads(policy.checkpoint_payload_bytes)

    def recorded_strings(value: object) -> set[str]:
        if isinstance(value, str):
            return {value}
        if isinstance(value, dict):
            return {
                item
                for key, child in value.items()
                for item in ({key} | recorded_strings(child))
            }
        if isinstance(value, list):
            return {item for child in value for item in recorded_strings(child)}
        return set()

    checkpoint_strings = recorded_strings(checkpoint_record)
    for fresh_value in (*fresh_cases, *(value for value in fresh_expected if value)):
        assert fresh_value not in checkpoint_strings
    assert result.winner_program_sha256 is not None
    assert result.winner_program_sha256 not in checkpoint_strings
    assert result.source_after_sha256 not in checkpoint_strings

    gate_spec = _write_repository(
        tmp_path / "gates",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        cases=fresh_cases,
        expected=fresh_expected,
    )
    gate_controls = (
        ({"context_cell_gate": False}, repair.STOP_CONTEXT),
        ({"lineage_cell_gate": False}, repair.STOP_LINEAGE),
        ({"policy_cell_gate": False}, repair.STOP_POLICY),
        ({"effect_cell_gate": False}, repair.STOP_EFFECT),
        ({"actuation_gate": False}, repair.STOP_ACTUATION),
    )
    for kwargs, expected_status in gate_controls:
        stopped = repair.prepare_microrepository_repair_v2(
            policy,
            signature,
            gate_spec,
            **kwargs,
        )
        assert stopped.state.status == expected_status
        stopped_result = stopped.repair()
        assert stopped_result.status == expected_status
        assert stopped_result.work.desired_manifest_reads == 0
        assert stopped_result.work.terminal_selection_races == 0
        assert stopped_result.work.source_materializations == 0
        _assert_no_write_stop(stopped_result)
    assert (gate_spec.repo_root / gate_spec.source_relative_path).read_bytes() == root_source

    ambiguous_spec = _write_repository(
        tmp_path / "ambiguous",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        cases=("a121b", "letters"),
        expected=("121", ""),
    )
    ambiguous = repair.prepare_microrepository_repair_v2(
        policy,
        signature,
        ambiguous_spec,
    ).repair()
    assert ambiguous.status == recurrent.STOP_MULTIPLE_EXACT
    assert ambiguous.work.desired_manifest_reads == 1
    assert ambiguous.work.terminal_selection_races == 1
    assert ambiguous.work.source_materializations == 0
    _assert_no_write_stop(ambiguous)

    no_match_spec = _write_repository(
        tmp_path / "no_match",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        cases=fresh_cases,
        expected=("not", "a", "terminal", "vector"),
    )
    no_match = repair.prepare_microrepository_repair_v2(
        policy,
        signature,
        no_match_spec,
    ).repair()
    assert no_match.status == recurrent.STOP_NO_EXACT
    assert no_match.work.desired_manifest_reads == 1
    assert no_match.work.terminal_selection_races == 1
    assert no_match.work.source_materializations == 0
    _assert_no_write_stop(no_match)

    drift_spec = _write_repository(
        tmp_path / "drift",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        cases=fresh_cases,
        expected=fresh_expected,
    )
    drifted = repair.prepare_microrepository_repair_v2(policy, signature, drift_spec)
    drift_path = drift_spec.repo_root / drift_spec.source_relative_path
    drift_path.write_bytes(root_source + b"# drift\n")
    drift_result = drifted.repair()
    assert drift_result.status == repair.STOP_SOURCE_DRIFT
    assert drift_result.work.desired_manifest_reads == 0
    assert drift_result.work.terminal_selection_races == 0
    _assert_no_write_stop(drift_result)

    failure_spec = _write_repository(
        tmp_path / "failure",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        cases=fresh_cases,
        expected=fresh_expected,
        force_test_failure=True,
    )
    failed = repair.prepare_microrepository_repair_v2(
        policy,
        signature,
        failure_spec,
    ).repair()
    assert failed.status == repair.STOP_TEST_FAILURE
    assert failed.selection_status == recurrent.SELECTED
    assert failed.exact_proof_count == 1
    assert failed.consequence_action_key == "action:revise"
    assert failed.repository_test_exit_code == 17
    assert failed.work.terminal_selection_races == 1
    assert failed.work.source_materializations == 1
    assert failed.work.selected_policy_actuations == 2
    assert failed.work.selected_program_executions == 2
    assert failed.work.patch_writes == 1
    assert failed.work.repository_tests == 1
    assert failed.work.consequence_policy_races == 1
    assert failed.work.commits == 0
    assert failed.work.revisions == 1
    assert failed.work.reverts == 1
    assert (failure_spec.repo_root / failure_spec.source_relative_path).read_bytes() == (
        root_source
    )

    implementation = inspect.getsource(repair.PreparedMicroRepositoryRepairV2)
    assert implementation.count("self._frontier.select(") == 1
    assert implementation.count("subprocess.run(") == 1
    assert implementation.count("recurrent._policy_race(") == 1
