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
from experiments.r3_pilot import causal_task_observation_candidate_plan_bank_v1 as bank
from experiments.r3_pilot import common_microrepository_autonomous_correction_v1 as correction
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
    visible_expected: tuple[str, ...],
    repository_input: str = "A1b2!",
    repository_expected: str = "12",
    force_failure: bool = False,
    drift_source: bool = False,
) -> correction.AutonomousCorrectionSpecV1:
    root.mkdir()
    (root / "solution.py").write_bytes(root_source)
    (root / "visible_expected.json").write_text(
        json.dumps({"expected": visible_expected}, separators=(",", ":")),
        encoding="utf-8",
    )
    (root / "repository_case.json").write_text(
        json.dumps(
            {"payload": repository_input, "expected": repository_expected},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    forced_exit = 17 if force_failure else 0
    drift_line = (
        "source = root / 'solution.py'\nsource.write_bytes(source.read_bytes() + b'# drift\\n')\n"
        if drift_source
        else ""
    )
    (root / "repo_test.py").write_text(
        "from __future__ import annotations\n"
        "import json\n"
        "from pathlib import Path\n"
        "from solution import clean_text\n"
        "root = Path(__file__).parent\n"
        "record = json.loads((root / 'repository_case.json').read_text())\n"
        "actual = clean_text(record['payload'])\n"
        + drift_line
        + "if actual != record['expected']:\n"
        "    raise SystemExit(3)\n"
        f"raise SystemExit({forced_exit})\n",
        encoding="utf-8",
    )
    return correction.AutonomousCorrectionSpecV1(
        repo_root=root,
        source_relative_path="solution.py",
        expected_manifest_relative_path="visible_expected.json",
        test_command=(sys.executable, "-B", "repo_test.py"),
        root_program_sha256=root_program_sha256,
        root_source_sha256=sha256(root_source).hexdigest(),
    )


def _assert_one_failed_attempt(
    result: correction.AutonomousCorrectionResultV1,
) -> None:
    assert len(result.attempts) == 1
    assert result.attempts[0].repository_test_exit_code != 0
    assert result.attempts[0].rolled_back
    assert not result.attempts[0].committed
    assert result.work.source_materializations == 1
    assert result.work.patch_writes == 1
    assert result.work.repository_tests == 1
    assert result.work.unselected_repository_tests == 0
    assert result.work.reverts == 1
    assert result.work.rollback_verifications == 1
    assert result.work.commits == 0


def test_cell_state_replans_after_one_real_repository_failure(tmp_path: Path) -> None:
    base = _base_policy()
    signature = _string_signature()
    train_cases = (
        ("A1e2Z9",),
        ("u3B4i8",),
        ("O5x6a0",),
        ("e7Y8U9",),
    )
    solved = string_repair.solve_string_relation_training_v2(
        base,
        signature,
        train_cases,
    )
    policy = string_repair.fit_string_relation_policy_v2(base, solved)

    # These fresh visible digit subsequences are palindromes.  Both learned
    # candidates are therefore semantically eligible without a test label.
    visible_cases = ("a1b2c1", "x9y8z9", "zero00", "no-digits")
    visible_expected = ("121", "989", "00", "")
    frontier = policy.build(signature, tuple((value,) for value in visible_cases))
    assert frontier.status == string_repair.READY
    supported = tuple(item for item in frontier.candidate_records if item.receipt_cell_support > 0)
    assert len(supported) == 2
    assert {item.receipt_cell_support for item in supported} == {1, 2}
    assert supported[0].program_sha256 != supported[1].program_sha256

    root_program = solved.identity_program
    root_source = typed_program.compile(root_program).source.encode("utf-8")
    primary_spec = _write_repository(
        tmp_path / "primary",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        visible_expected=visible_expected,
    )
    materializations_before = frontier.source_materialization_count
    prepared = correction.prepare_autonomous_correction_v1(
        frontier,
        primary_spec,
    )
    assert prepared.state.status == correction.PREPARED
    assert prepared.state.candidate_count == 2
    assert prepared.state.source_reads_during_prepare == 1
    assert prepared.state.desired_reads_during_prepare == 0
    assert prepared.state.source_materializations_during_prepare == 0
    assert prepared.state.patch_writes_during_prepare == 0
    assert prepared.state.repository_tests_during_prepare == 0

    result = prepared.repair()
    assert result.status == correction.COMMITTED
    assert [item.receipt_cell_support for item in result.attempts] == [2, 1]
    assert [item.repository_test_exit_code for item in result.attempts] == [3, 0]
    assert [item.consequence_action_key for item in result.attempts] == [
        recurrent.ACTION_KEYS[recurrent.ACTION_REVISE],
        recurrent.ACTION_KEYS[recurrent.ACTION_COMMIT],
    ]
    assert [item.rolled_back for item in result.attempts] == [True, False]
    assert [item.committed for item in result.attempts] == [False, True]
    assert result.attempts[0].candidate_sha256 != result.attempts[1].candidate_sha256
    assert result.attempts[0].candidate_address != result.attempts[1].candidate_address
    assert result.attempts[0].program_sha256 != result.attempts[1].program_sha256
    assert result.work.desired_manifest_reads == 1
    assert result.work.eligibility_races == 1
    assert result.work.candidate_races == 2
    assert result.work.source_materializations == 2
    assert result.work.unselected_source_materializations == 0
    assert result.work.selected_patch_sources == 2
    assert result.work.patch_writes == 2
    assert result.work.repository_consequence_roots == 2
    assert result.work.consequence_policy_races == 2
    assert result.work.repository_tests == 2
    assert result.work.unselected_repository_tests == 0
    assert result.work.revisions == 1
    assert result.work.replan_updates == 1
    assert result.work.reverts == 1
    assert result.work.rollback_verifications == 1
    assert result.work.commits == 1
    assert result.work.candidate_work_cell_steps == 3_600
    assert frontier.source_materialization_count - materializations_before == 2
    assert result.initial_checkpoint_sha256 is not None
    assert result.final_checkpoint is not None
    assert result.final_checkpoint.trainable_state_bytes == 4
    assert result.final_checkpoint.address_state_bytes == 8
    assert result.final_checkpoint.provenance_state_bytes == 74
    assert result.final_checkpoint.checkpoint_array_bytes == 86
    assert result.final_checkpoint.canonical_metadata_bytes > 0
    assert result.final_checkpoint.filled_provenance_rows == 2
    assert result.final_checkpoint.candidate_races == 2
    assert result.final_checkpoint.replan_updates == 1
    assert result.final_checkpoint.candidate_work_cell_steps == 3_600
    assert len(result.final_checkpoint.physical_tier_sha256s) == 2
    assert len(set(result.final_checkpoint.physical_tier_sha256s)) == 2
    assert result.final_checkpoint.selector_projection_sha256
    assert result.initial_checkpoint_sha256 != result.final_checkpoint.sha256
    repaired_source = (primary_spec.repo_root / primary_spec.source_relative_path).read_bytes()
    assert repaired_source != root_source
    assert sha256(repaired_source).hexdigest() == result.source_after_sha256
    assert not repaired_source.rstrip().endswith(b"[::-1]")

    # Removing physical receipt support stops before desired, compile, or test.
    support_spec = _write_repository(
        tmp_path / "support_off",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        visible_expected=visible_expected,
    )
    before = frontier.source_materialization_count
    support_off = correction.prepare_autonomous_correction_v1(
        frontier,
        support_spec,
        relation_support_gate=False,
    ).repair()
    assert support_off.status == correction.STOP_RELATION_SUPPORT
    assert support_off.work.desired_manifest_reads == 0
    assert support_off.work.candidate_races == 0
    assert support_off.work.source_materializations == 0
    assert support_off.work.patch_writes == 0
    assert support_off.work.repository_tests == 0
    assert frontier.source_materialization_count == before
    assert (support_spec.repo_root / support_spec.source_relative_path).read_bytes() == (
        root_source
    )

    # A forged source-free record is rejected without compiling it.
    forged_record = supported[0]._replace(receipt_cell_support=9)
    forged_bank_spec = _write_repository(
        tmp_path / "forged_bank",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        visible_expected=visible_expected,
    )
    forged_bank = correction.prepare_autonomous_correction_v1(
        frontier,
        forged_bank_spec,
        candidate_records=(forged_record, supported[1]),
    ).repair()
    assert forged_bank.status == correction.STOP_ATTESTATION
    assert forged_bank.work.desired_manifest_reads == 0
    assert forged_bank.work.source_materializations == 0

    # Revision and replan cell ablations each stop after one exact rollback.
    revision_spec = _write_repository(
        tmp_path / "revision_off",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        visible_expected=visible_expected,
    )
    revision_off = correction.prepare_autonomous_correction_v1(
        frontier,
        revision_spec,
    ).repair(revision_cell_gate=False)
    assert revision_off.status == correction.STOP_REVISION_GATE
    _assert_one_failed_attempt(revision_off)
    assert revision_off.work.candidate_races == 1
    assert revision_off.work.revisions == 0
    assert revision_off.work.replan_updates == 0
    assert (revision_spec.repo_root / revision_spec.source_relative_path).read_bytes() == (
        root_source
    )

    replan_spec = _write_repository(
        tmp_path / "replan_off",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        visible_expected=visible_expected,
    )
    replan_off = correction.prepare_autonomous_correction_v1(
        frontier,
        replan_spec,
    ).repair(replan_cell_gate=False)
    assert replan_off.status == correction.STOP_REPLAN_GATE
    _assert_one_failed_attempt(replan_off)
    assert replan_off.work.candidate_races == 1
    assert replan_off.work.revisions == 1
    assert replan_off.work.replan_updates == 0
    assert (replan_spec.repo_root / replan_spec.source_relative_path).read_bytes() == (root_source)

    # Without the inhibitory candidate update, the physical rerace repeats the
    # failed winner and stops before a second compile, patch, or test.
    repeat_spec = _write_repository(
        tmp_path / "repeat",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        visible_expected=visible_expected,
    )
    repeated = correction.prepare_autonomous_correction_v1(
        frontier,
        repeat_spec,
    ).repair(inhibitory_update_gate=False)
    assert repeated.status == correction.STOP_REPEATED_WINNER
    _assert_one_failed_attempt(repeated)
    assert repeated.work.candidate_races == 2
    assert repeated.work.revisions == 1
    assert repeated.work.replan_updates == 0
    assert repeated.work.source_materializations == 1
    assert repeated.work.repository_tests == 1

    # Forged consequence receipts and post-test source drift both roll back
    # exactly and stop before any consequence-driven replan.
    forged_spec = _write_repository(
        tmp_path / "forged_consequence",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        visible_expected=visible_expected,
    )
    forged = correction.prepare_autonomous_correction_v1(
        frontier,
        forged_spec,
    ).repair(forge_consequence_attempt=1)
    assert forged.status == correction.STOP_FORGED_CONSEQUENCE
    assert len(forged.attempts) == 0
    assert forged.work.repository_consequence_roots == 1
    assert forged.work.consequence_policy_races == 0
    assert forged.work.reverts == 1
    assert forged.work.rollback_verifications == 1
    assert (forged_spec.repo_root / forged_spec.source_relative_path).read_bytes() == (root_source)

    drift_spec = _write_repository(
        tmp_path / "source_drift",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        visible_expected=visible_expected,
        drift_source=True,
    )
    drifted = correction.prepare_autonomous_correction_v1(
        frontier,
        drift_spec,
    ).repair()
    assert drifted.status == correction.STOP_SOURCE_DRIFT
    assert len(drifted.attempts) == 0
    assert drifted.work.repository_tests == 1
    assert drifted.work.repository_consequence_roots == 0
    assert drifted.work.consequence_policy_races == 0
    assert drifted.work.reverts == 1
    assert drifted.work.rollback_verifications == 1
    assert (drift_spec.repo_root / drift_spec.source_relative_path).read_bytes() == (root_source)

    # Reversing/remapping the public record slots produces the same cell-owned
    # candidate trajectory and the same final program.
    remap_spec = _write_repository(
        tmp_path / "remapped",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        visible_expected=visible_expected,
    )
    remapped = correction.prepare_autonomous_correction_v1(
        frontier,
        remap_spec,
        candidate_records=tuple(reversed(supported)),
    ).repair()
    assert remapped.status == correction.COMMITTED
    assert [item.candidate_sha256 for item in remapped.attempts] == [
        item.candidate_sha256 for item in result.attempts
    ]
    assert [item.program_sha256 for item in remapped.attempts] == [
        item.program_sha256 for item in result.attempts
    ]
    assert [item.candidate_address for item in remapped.attempts] == [
        item.candidate_address for item in result.attempts
    ]
    assert remapped.source_after_sha256 == result.source_after_sha256

    # If both real tests fail, the bounded controller rolls back the second
    # patch and stops after exactly two attempts; it never cycles.
    both_fail_spec = _write_repository(
        tmp_path / "both_fail",
        root_source=root_source,
        root_program_sha256=root_program.sha256,
        visible_expected=visible_expected,
        force_failure=True,
    )
    both_failed = correction.prepare_autonomous_correction_v1(
        frontier,
        both_fail_spec,
    ).repair()
    assert both_failed.status == correction.STOP_ATTEMPT_BUDGET
    assert len(both_failed.attempts) == 2
    assert all(item.repository_test_exit_code != 0 for item in both_failed.attempts)
    assert both_failed.work.candidate_races == 2
    assert both_failed.work.source_materializations == 2
    assert both_failed.work.patch_writes == 2
    assert both_failed.work.repository_tests == 2
    assert both_failed.work.consequence_policy_races == 2
    assert both_failed.work.revisions == 2
    assert both_failed.work.reverts == 2
    assert both_failed.work.commits == 0
    assert (both_fail_spec.repo_root / both_fail_spec.source_relative_path).read_bytes() == (
        root_source
    )

    implementation = inspect.getsource(correction)
    candidate_race = inspect.getsource(correction._CorrectionCellsV1.race)
    assert "_terminals" not in implementation
    assert "next(" not in implementation
    assert "terminal:commit" not in implementation
    assert "filter_digits" not in implementation
    assert "0123456789" not in implementation
    assert "[::-1]" not in implementation
    for forbidden in (
        "np.max",
        "np.argmax",
        "np.flatnonzero",
        "sorted(",
        "select_exact_semantic_vector",
        "desired_outputs",
    ):
        assert forbidden not in candidate_race
    assert candidate_race.count("self._selector.select(currents, stop_cell=STOP_CELL)") == 1
    assert implementation.count("frontier.materialize_selected(candidate)") == 1
    assert implementation.count("subprocess.run(") == 1
    assert implementation.count("_read_desired(") == 2
