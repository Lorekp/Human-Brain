from __future__ import annotations

import inspect
import json
import sys
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from experiments.r3_pilot import causal_microrepository_repair_loop_v1 as repair_loop
from experiments.r3_pilot import causal_predicate_gated_fold_v1 as gated
from experiments.r3_pilot import causal_recurrent_fragment_relation_policy_v2 as recurrent
from experiments.r3_pilot import causal_reusable_program_fragment_library_v1 as fragments
from experiments.r3_pilot import causal_task_observation_candidate_plan_bank_v1 as bank
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import typed_semantic_packed_absolute_map_fold_v1 as absolute
from humanbrain import typed_program
from humanbrain.typed_program_types import TypedSignature


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


def _encoded(values: Sequence[int]) -> tuple[output_cells.TypedOutputEncoding, ...]:
    return tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.INT,
        )
        for value in values
    )


def _fit_policy(
    signature: TypedSignature,
) -> recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2:
    publication = bank.CausalTaskObservationCandidatePlanBankV1.build(
        _checkpoint(),
        signature,
        (([64],), ([-2],), ([4],), ([5],)),
    )
    library = fragments.fit_reusable_program_fragment_library_v1(
        publication,
        (
            _encoded((64, 2, 4, 5)),
            _encoded((0, -2, 4, 0)),
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
    abs_at_predicate = lesson_frontier.select(_encoded((-1, 1, 1, 0, 2, -1, 0, -1)))
    abs_as_selected_action = lesson_frontier.select(_encoded((9, 7, 5, 11, 11, 7, 0, 1)))
    return recurrent.fit_causal_recurrent_fragment_relation_policy_v2(
        library,
        (
            recurrent.FragmentRelationLesson(lesson_frontier, abs_at_predicate),
            recurrent.FragmentRelationLesson(
                lesson_frontier,
                abs_as_selected_action,
            ),
        ),
    )


def _write_microrepository(
    root: Path,
    *,
    root_source: bytes,
    root_program_sha256: str,
    cases: tuple[list[int], ...],
    expected: tuple[int, ...],
    force_test_failure: bool = False,
) -> repair_loop.MicroRepositoryRepairSpecV1:
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
        "from solution import aggregate\n"
        "root = Path(__file__).parent\n"
        "cases = json.loads((root / 'cases.json').read_text())['cases']\n"
        "expected = json.loads((root / 'expected.json').read_text())['expected']\n"
        "actual = [aggregate(list(values)) for values in cases]\n"
        "if actual != expected:\n"
        "    raise SystemExit(3)\n"
        f"raise SystemExit({forced_exit})\n",
        encoding="utf-8",
    )
    return repair_loop.MicroRepositoryRepairSpecV1(
        repo_root=root,
        source_relative_path="solution.py",
        case_manifest_relative_path="cases.json",
        expected_manifest_relative_path="expected.json",
        test_command=(sys.executable, "-B", "repo_test.py"),
        root_program_sha256=root_program_sha256,
        root_source_sha256=sha256(root_source).hexdigest(),
    )


def _assert_no_write_stop(
    result: repair_loop.MicroRepositoryRepairResultV1,
) -> None:
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


def test_recurrent_policy_repairs_one_attested_microrepository(tmp_path: Path) -> None:
    signature = absolute.list_int_to_int_signature()
    policy = _fit_policy(signature)
    receipts = policy.relation_receipts
    assert len(receipts) == 2
    assert len({item.receipt_sha256 for item in receipts}) == 2
    assert len({item.base_program_sha256 for item in receipts}) == 1
    assert len({item.base_program_tokens for item in receipts}) == 1

    # Fixture provisioning is outside the measured repair loop.  It renders
    # the learned root once; the adapter itself subsequently uses hashes only.
    root_program = typed_program.Program(receipts[0].base_program_tokens, signature)
    assert root_program.sha256 == policy.state.base_program_sha256
    root_source = typed_program.compile(root_program).source.encode("utf-8")

    fresh_cases = (
        [-10, -4, 3, 8],
        [0, 5, -5, 2],
        [9, -1, -2, 4],
        [6, -7, 11],
        [-3, 3, 12, -4],
    )
    fresh_expected = (7, 2, 7, 0, 10)
    spec = _write_microrepository(
        tmp_path / "primary",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        cases=fresh_cases,
        expected=fresh_expected,
    )
    prepared = repair_loop.prepare_microrepository_repair_v1(
        policy,
        signature,
        spec,
    )
    assert prepared.state.status == repair_loop.PREPARED
    assert prepared.state.case_count == len(fresh_cases)
    assert prepared.state.desired_reads_during_prepare == 0
    assert prepared.state.source_materializations_during_prepare == 0
    assert prepared.state.patch_writes_during_prepare == 0
    assert prepared.state.repository_tests_during_prepare == 0
    assert prepared.state.root_rollouts == 1
    assert prepared.state.affordance_races == 2
    assert prepared.state.policy_races == 5
    assert prepared.state.effect_races == 2
    assert prepared.state.selected_policy_actuations == 2
    assert prepared.state.committed_edits == 2
    assert prepared.state.revisions == 0
    assert prepared.state.unselected_splice_executions == 0
    assert len(prepared.state.relation_receipt_sha256s) == 2
    assert len(set(prepared.state.relation_receipt_sha256s)) == 2

    result = prepared.repair()
    assert result.status == repair_loop.COMMITTED
    assert result.selection_status == recurrent.SELECTED
    assert result.exact_proof_count == 1
    assert len(result.applied_relation_receipt_sha256s) == 2
    assert len(set(result.applied_relation_receipt_sha256s)) == 2
    assert result.consequence_action_key == "action:commit"
    assert result.repository_test_exit_code == 0
    assert result.work.source_reads == 2
    assert result.work.case_manifest_reads == 1
    assert result.work.desired_manifest_reads == 1
    assert result.work.root_source_renderings == 0
    assert result.work.terminal_selection_races == 1
    assert result.work.source_materializations == 1
    assert result.work.selected_policy_actuations == 2
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
    assert (
        prepared.state.root_rollouts + result.work.repository_consequence_roots
        <= recurrent.MAX_ROOTS
    )
    assert prepared.state.policy_races <= recurrent.MAX_POLICY_RACES_PER_ROOT
    assert result.work.consequence_policy_races <= recurrent.MAX_POLICY_RACES_PER_ROOT
    repaired_source = (spec.repo_root / spec.source_relative_path).read_text(encoding="utf-8")
    assert "abs(item_0) < 5" in repaired_source
    assert repaired_source.count("abs(item_0)") == 2
    assert sha256(repaired_source.encode("utf-8")).hexdigest() == (result.source_after_sha256)

    gate_spec = _write_microrepository(
        tmp_path / "gates",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        cases=fresh_cases,
        expected=fresh_expected,
    )
    gate_controls = (
        ({"relation_cell_gate": False}, recurrent.STOP_RELATION),
        ({"policy_cell_gate": False}, recurrent.STOP_POLICY),
        ({"actuation_gate": False}, recurrent.STOP_ACTUATION),
    )
    for kwargs, expected_status in gate_controls:
        stopped = repair_loop.prepare_microrepository_repair_v1(
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
    assert (gate_spec.repo_root / gate_spec.source_relative_path).read_bytes() == (root_source)

    no_match_spec = _write_microrepository(
        tmp_path / "no_match",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        cases=fresh_cases,
        expected=(1, 2, 3, 4, 5),
    )
    no_match = repair_loop.prepare_microrepository_repair_v1(
        policy,
        signature,
        no_match_spec,
    ).repair()
    assert no_match.status == recurrent.STOP_NO_EXACT
    assert no_match.work.desired_manifest_reads == 1
    assert no_match.work.terminal_selection_races == 1
    assert no_match.work.source_materializations == 0
    _assert_no_write_stop(no_match)

    ambiguous_spec = _write_microrepository(
        tmp_path / "ambiguous",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        cases=([],),
        expected=(0,),
    )
    ambiguous = repair_loop.prepare_microrepository_repair_v1(
        policy,
        signature,
        ambiguous_spec,
    ).repair()
    assert ambiguous.status == recurrent.STOP_MULTIPLE_EXACT
    assert ambiguous.work.desired_manifest_reads == 1
    assert ambiguous.work.terminal_selection_races == 1
    assert ambiguous.work.source_materializations == 0
    _assert_no_write_stop(ambiguous)

    drift_spec = _write_microrepository(
        tmp_path / "drift",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        cases=fresh_cases,
        expected=fresh_expected,
    )
    drifted = repair_loop.prepare_microrepository_repair_v1(
        policy,
        signature,
        drift_spec,
    )
    assert drifted.state.status == repair_loop.PREPARED
    drift_path = drift_spec.repo_root / drift_spec.source_relative_path
    drift_path.write_bytes(root_source + b"# drift\n")
    drift_result = drifted.repair()
    assert drift_result.status == repair_loop.STOP_SOURCE_DRIFT
    assert drift_result.work.desired_manifest_reads == 0
    assert drift_result.work.terminal_selection_races == 0
    assert drift_result.work.source_materializations == 0
    _assert_no_write_stop(drift_result)

    failure_spec = _write_microrepository(
        tmp_path / "failure",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        cases=fresh_cases,
        expected=fresh_expected,
        force_test_failure=True,
    )
    failed = repair_loop.prepare_microrepository_repair_v1(
        policy,
        signature,
        failure_spec,
    ).repair()
    assert failed.status == repair_loop.STOP_TEST_FAILURE
    assert failed.selection_status == recurrent.SELECTED
    assert failed.exact_proof_count == 1
    assert failed.consequence_action_key == "action:revise"
    assert failed.repository_test_exit_code == 17
    assert failed.work.terminal_selection_races == 1
    assert failed.work.source_materializations == 1
    assert failed.work.selected_policy_actuations == 2
    assert failed.work.selected_patch_sources == 1
    assert failed.work.unselected_patch_sources == 0
    assert failed.work.patched_file_count == 1
    assert failed.work.patch_writes == 1
    assert failed.work.repository_tests == 1
    assert failed.work.unselected_repository_tests == 0
    assert failed.work.consequence_policy_races == 1
    assert failed.work.commits == 0
    assert failed.work.revisions == 1
    assert failed.work.revisions <= recurrent.MAX_REVISIONS_PER_ROOT
    assert failed.work.reverts == 1
    assert (failure_spec.repo_root / failure_spec.source_relative_path).read_bytes() == (
        root_source
    )

    implementation = inspect.getsource(repair_loop.PreparedMicroRepositoryRepairV1)
    assert implementation.count("self._frontier.select(") == 1
    assert implementation.count("subprocess.run(") == 1
    assert implementation.count("recurrent._policy_race(") == 1
