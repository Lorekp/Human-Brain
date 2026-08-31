from __future__ import annotations

import inspect
import json
import sys
import time
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.r3_pilot import causal_microrepository_repair_loop_v2 as string_repair
from experiments.r3_pilot import causal_predicate_gated_fold_v1 as gated
from experiments.r3_pilot import causal_recurrent_fragment_relation_policy_v2 as recurrent
from experiments.r3_pilot import causal_reusable_program_fragment_library_v1 as fragments
from experiments.r3_pilot import causal_task_observation_candidate_plan_bank_v1 as bank
from experiments.r3_pilot import common_continual_repository_action_memory_v1 as memory
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


def _frontier(
    policy: string_repair.FrozenStringRelationPolicyV2,
    signature: TypedSignature,
    visible_cases: tuple[str, ...],
    visible_expected: tuple[str, ...],
) -> tuple[
    string_repair.StringRelationFrontierV2,
    tuple[string_repair.StringRelationCandidateRecordV2, ...],
]:
    frontier = policy.build(signature, tuple((value,) for value in visible_cases))
    assert frontier.status == string_repair.READY
    supported = tuple(
        item for item in frontier.candidate_records if item.receipt_cell_support > 0
    )
    assert len(supported) == 2
    assert {item.receipt_cell_support for item in supported} == {1, 2}
    assert tuple(
        "".join(character for character in value if character.isdigit())
        for value in visible_cases
    ) == visible_expected
    assert all(value == value[::-1] for value in visible_expected)
    return frontier, supported


def _stage_repository(root: Path, source_name: str, source: bytes) -> None:
    root.mkdir()
    (root / source_name).write_bytes(source)


def _control_spec(
    root: Path,
    source_name: str,
    source: bytes,
    root_program_sha256: str,
    *,
    root_source_sha256: str | None = None,
) -> correction.AutonomousCorrectionSpecV1:
    return correction.AutonomousCorrectionSpecV1(
        repo_root=root,
        source_relative_path=source_name,
        expected_manifest_relative_path="not-created.json",
        test_command=(sys.executable, "-B", "not-created.py"),
        root_program_sha256=root_program_sha256,
        root_source_sha256=(
            sha256(source).hexdigest()
            if root_source_sha256 is None
            else root_source_sha256
        ),
    )


def _install_hidden_fixture(
    root: Path,
    *,
    source_name: str,
    source: bytes,
    root_program_sha256: str,
    visible_expected: tuple[str, ...],
    repository_input: str,
    repository_expected: str,
    manifest_name: str,
    test_name: str,
) -> correction.AutonomousCorrectionSpecV1:
    case_name = f"{test_name}.case.json"
    (root / manifest_name).write_text(
        json.dumps({"expected": visible_expected}, separators=(",", ":")),
        encoding="utf-8",
    )
    (root / case_name).write_text(
        json.dumps(
            {"payload": repository_input, "expected": repository_expected},
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
        "actual = namespace['clean_text'](record['payload'])\n"
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
        tuple(sha256(array.tobytes(order="C")).hexdigest() for _, array in value._arrays()),
    )


def _candidate_evidence(
    recall: memory.RepositoryActionRecallV1,
) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        sorted(
            (
                item.candidate_action_sha256,
                item.excitatory_ticks,
                item.inhibitory_ticks,
            )
            for item in recall.candidates
        )
    )


def _assert_preaction_stop(result: correction.AutonomousCorrectionResultV1) -> None:
    result.authenticate()
    assert result.status in {
        correction.STOP_ATTESTATION,
        correction.STOP_LEARNED_RECALL,
    }
    assert result.repository_actions == ()
    assert result.work.desired_manifest_reads == 0
    assert result.work.candidate_races == 0
    assert result.work.source_materializations == 0
    assert result.work.patch_writes == 0
    assert result.work.repository_tests == 0


def test_replay_free_continual_repository_action_memory_curriculum(
    tmp_path: Path,
) -> None:
    started = time.perf_counter()
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
    string_policy_fit_count = 0
    string_policy_fit_count += 1
    policy = string_repair.fit_string_relation_policy_v2(base, solved)
    assert string_policy_fit_count == 1

    root_program = solved.identity_program
    compiled_root = typed_program.compile(root_program).source.encode("utf-8")
    source_harbor = (
        compiled_root.rstrip()
        + b"\n\n# Incoming records preserve the Harbor archive convention.\n"
    )
    source_cedar = (
        compiled_root.rstrip()
        + b"\n\n# Incoming records preserve the Cedar ledger convention.\n"
    )
    assert len(source_harbor) <= 256
    assert len(source_cedar) <= 256
    assert b"alpha" not in source_harbor.lower() + source_cedar.lower()
    assert b"beta" not in source_harbor.lower() + source_cedar.lower()

    visible_cohorts = (
        (
            ("a1b2c1", "x9y8z9", "zero00", "no-digits"),
            ("121", "989", "00", ""),
        ),
        (
            ("m3n4m3", "p5q6p5", "seven77", "blank"),
            ("343", "565", "77", ""),
        ),
        (
            ("r1s0r1", "t2u3t2", "pair44", "void"),
            ("101", "232", "44", ""),
        ),
        (
            ("v6w7v6", "x8y9x8", "double55", "none"),
            ("676", "898", "55", ""),
        ),
    )
    assert len({value for cases, _ in visible_cohorts for value in cases}) == 16
    frontiers = tuple(
        _frontier(policy, signature, cases, expected)
        for cases, expected in visible_cohorts
    )

    repository_layouts = (
        (tmp_path / "coast", "solution.py", source_harbor),
        (tmp_path / "grove", "worker.py", source_cedar),
        (tmp_path / "relocated", "repaired_module.py", source_harbor),
        (tmp_path / "ledger", "handler.py", source_cedar),
    )
    for root, source_name, source in repository_layouts:
        _stage_repository(root, source_name, source)
    projected_harbor = memory.project_attested_repository_source_v1(source_harbor)
    projected_cedar = memory.project_attested_repository_source_v1(source_cedar)
    assert projected_harbor.sha256 != projected_cedar.sha256

    total_materializations = 0
    total_repository_tests = 0

    # Episode 1: the desired-free frontier exists before the hidden fixture.
    frontier1, supported1 = frontiers[0]
    spec1 = _install_hidden_fixture(
        repository_layouts[0][0],
        source_name=repository_layouts[0][1],
        source=source_harbor,
        root_program_sha256=root_program.sha256,
        visible_expected=visible_cohorts[0][1],
        repository_input="A1b2!",
        repository_expected="12",
        manifest_name="visible_expected.json",
        test_name="repo_test.py",
    )
    before = frontier1.source_materialization_count
    result1 = correction.prepare_autonomous_correction_v1(
        policy,
        frontier1,
        spec1,
    ).repair()
    result1.authenticate()
    assert result1.status == correction.COMMITTED
    assert [item.receipt_cell_support for item in result1.attempts] == [2, 1]
    assert [item.repository_test_exit_code for item in result1.attempts] == [3, 0]
    assert [item.rolled_back for item in result1.attempts] == [True, False]
    assert len(result1.repository_actions) == 2
    total_materializations += frontier1.source_materialization_count - before
    total_repository_tests += result1.work.repository_tests
    memory1 = memory.start_continual_repository_action_memory_v1(
        result1.repository_actions
    )
    memory1_snapshot = _memory_snapshot(memory1)

    # Cohort order is authenticated, while the original committed result remains usable.
    with pytest.raises(ValueError):
        memory.start_continual_repository_action_memory_v1(
            tuple(reversed(result1.repository_actions))
        )
    assert _memory_snapshot(memory1) == memory1_snapshot

    # Episode 2: freeze the first parent before installing the disjoint fixture.
    frontier2, supported2 = frontiers[1]
    assert memory1.checkpoint_sha256 == memory1_snapshot[0]
    spec2 = _install_hidden_fixture(
        repository_layouts[1][0],
        source_name=repository_layouts[1][1],
        source=source_cedar,
        root_program_sha256=root_program.sha256,
        visible_expected=visible_cohorts[1][1],
        repository_input="C3d4!",
        repository_expected="43",
        manifest_name="answers.json",
        test_name="check_repo.py",
    )
    before = frontier2.source_materialization_count
    result2 = correction.prepare_autonomous_correction_v1(
        policy,
        frontier2,
        spec2,
    ).repair()
    result2.authenticate()
    assert result2.status == correction.COMMITTED
    assert len(result2.attempts) == 1
    assert result2.attempts[0].receipt_cell_support == 2
    assert result2.attempts[0].repository_test_exit_code == 0
    assert (
        result2.repository_actions[0].candidate_action_sha256
        != result1.repository_actions[-1].candidate_action_sha256
    )
    total_materializations += frontier2.source_materialization_count - before
    total_repository_tests += result2.work.repository_tests
    memory2 = memory.advance_continual_repository_action_memory_v1(
        memory1,
        result2.repository_actions,
    )
    assert _memory_snapshot(memory1) == memory1_snapshot
    memory2_snapshot = _memory_snapshot(memory2)

    # Context swaps use the same action bank and candidate support/order but swap recall.
    harbor_recall = memory2.recall_attested_source(source_harbor, frontier2)
    cedar_recall = memory2.recall_attested_source(source_cedar, frontier2)
    assert harbor_recall.status == memory.RECALLED
    assert cedar_recall.status == memory.RECALLED
    assert (
        harbor_recall.selected_candidate_action_sha256
        == result1.repository_actions[-1].candidate_action_sha256
    )
    assert (
        cedar_recall.selected_candidate_action_sha256
        == result2.repository_actions[0].candidate_action_sha256
    )
    assert (
        harbor_recall.selected_candidate_action_sha256
        != cedar_recall.selected_candidate_action_sha256
    )

    # Reversing the authenticated fresh candidate presentation changes no choice.
    remapped_recall = memory2.recall_attested_source(
        source_harbor,
        frontier2,
        candidate_records=tuple(reversed(supported2)),
    )
    assert remapped_recall.status == memory.RECALLED
    assert (
        remapped_recall.selected_candidate_action_sha256
        == harbor_recall.selected_candidate_action_sha256
    )

    # Episode 3: relocated/renamed repository, exact recurring source, one action.
    frontier3, supported3 = frontiers[2]
    spec3 = _install_hidden_fixture(
        repository_layouts[2][0],
        source_name=repository_layouts[2][1],
        source=source_harbor,
        root_program_sha256=root_program.sha256,
        visible_expected=visible_cohorts[2][1],
        repository_input="Q5r6!",
        repository_expected="56",
        manifest_name="fresh_visible.json",
        test_name="verify_patch.py",
    )
    before = frontier3.source_materialization_count
    result3 = correction.prepare_autonomous_correction_v1(
        policy,
        frontier3,
        spec3,
        action_memory=memory2,
    ).repair()
    result3.authenticate()
    assert result3.status == correction.COMMITTED
    assert len(result3.attempts) == 1
    assert result3.work.source_materializations == 1
    assert result3.work.repository_tests == 1
    assert (
        result3.repository_actions[0].candidate_action_sha256
        == result1.repository_actions[-1].candidate_action_sha256
    )
    assert result3.repository_actions[0].source_context_sha256 == projected_harbor.sha256
    assert (
        result3.repository_actions[0].source_context_sha256
        == result1.repository_actions[0].source_context_sha256
    )
    total_materializations += frontier3.source_materialization_count - before
    total_repository_tests += result3.work.repository_tests

    plasticity_off = memory.advance_continual_repository_action_memory_v1(
        memory2,
        result3.repository_actions,
        plasticity_gate=False,
    )
    assert plasticity_off is memory2
    assert _memory_snapshot(memory2) == memory2_snapshot

    forged_lesson = result3.repository_actions[0]._replace(
        consequence_sha256="0" * 64
    )
    with pytest.raises(ValueError):
        memory.advance_continual_repository_action_memory_v1(
            memory2,
            (forged_lesson,),
        )
    assert _memory_snapshot(memory2) == memory2_snapshot

    memory3 = memory.advance_continual_repository_action_memory_v1(
        memory2,
        result3.repository_actions,
    )
    assert _memory_snapshot(memory2) == memory2_snapshot
    memory3_snapshot = _memory_snapshot(memory3)
    alpha_before_beta_revision = memory3.recall_attested_source(
        source_harbor,
        frontier3,
    )
    assert alpha_before_beta_revision.status == memory.RECALLED

    # Episode 4: the old Cedar choice now fails, rolls back, and the alternate wins.
    frontier4, supported4 = frontiers[3]
    cedar_before_revision = memory3.recall_attested_source(source_cedar, frontier4)
    assert cedar_before_revision.status == memory.RECALLED
    assert (
        cedar_before_revision.selected_candidate_action_sha256
        == result2.repository_actions[0].candidate_action_sha256
    )
    spec4 = _install_hidden_fixture(
        repository_layouts[3][0],
        source_name=repository_layouts[3][1],
        source=source_cedar,
        root_program_sha256=root_program.sha256,
        visible_expected=visible_cohorts[3][1],
        repository_input="T7u8!",
        repository_expected="78",
        manifest_name="current_expected.json",
        test_name="exercise.py",
    )
    before = frontier4.source_materialization_count
    result4 = correction.prepare_autonomous_correction_v1(
        policy,
        frontier4,
        spec4,
        action_memory=memory3,
    ).repair()
    result4.authenticate()
    assert result4.status == correction.COMMITTED
    assert [item.repository_test_exit_code for item in result4.attempts] == [3, 0]
    assert [item.rolled_back for item in result4.attempts] == [True, False]
    assert (
        result4.repository_actions[0].candidate_action_sha256
        == result2.repository_actions[0].candidate_action_sha256
    )
    assert (
        result4.repository_actions[1].candidate_action_sha256
        == result1.repository_actions[-1].candidate_action_sha256
    )
    assert result4.repository_actions[0].controller_lesson_body_sha256s == tuple(
        item.lesson_body_sha256 for item in result4.repository_actions
    )
    total_materializations += frontier4.source_materialization_count - before
    total_repository_tests += result4.work.repository_tests

    contradiction = memory.advance_continual_repository_action_memory_v1(
        memory3,
        (result4.repository_actions[0],),
    )
    assert _memory_snapshot(memory3) == memory3_snapshot
    collided = contradiction.recall_attested_source(source_cedar, frontier4)
    assert collided.status in {
        memory.STOP_CANDIDATE_AMBIGUOUS,
        memory.STOP_CANDIDATE_UNKNOWN,
    }
    assert collided.selected_candidate_action_sha256 is None
    contradiction_snapshot = _memory_snapshot(contradiction)

    memory4 = memory.advance_continual_repository_action_memory_v1(
        contradiction,
        (result4.repository_actions[1],),
    )
    assert _memory_snapshot(contradiction) == contradiction_snapshot
    final_cedar = memory4.recall_attested_source(
        source_cedar,
        frontier4,
        candidate_records=tuple(reversed(supported4)),
    )
    assert final_cedar.status == memory.RECALLED
    assert (
        final_cedar.selected_candidate_action_sha256
        == result4.repository_actions[1].candidate_action_sha256
    )
    alpha_after_beta_revision = memory4.recall_attested_source(
        source_harbor,
        frontier3,
        candidate_records=tuple(reversed(supported3)),
    )
    assert alpha_after_beta_revision.status == memory.RECALLED
    assert (
        alpha_after_beta_revision.selected_candidate_action_sha256
        == alpha_before_beta_revision.selected_candidate_action_sha256
    )
    assert _candidate_evidence(alpha_after_beta_revision) == _candidate_evidence(
        alpha_before_beta_revision
    )

    # One-byte unknown visible source and all recall-cell ablations stop pre-action.
    source_unknown = source_harbor.replace(b"Harbor", b"Harbou")
    assert len(source_unknown) == len(source_harbor)
    assert sum(a != b for a, b in zip(source_unknown, source_harbor, strict=True)) == 1
    unknown_root = tmp_path / "unknown"
    _stage_repository(unknown_root, "module.py", source_unknown)
    unknown_result = correction.prepare_autonomous_correction_v1(
        policy,
        frontier3,
        _control_spec(
            unknown_root,
            "module.py",
            source_unknown,
            root_program.sha256,
        ),
        action_memory=memory4,
    ).repair()
    _assert_preaction_stop(unknown_result)

    gate_root = tmp_path / "gates"
    _stage_repository(gate_root, "renamed.py", source_harbor)
    gate_spec = _control_spec(
        gate_root,
        "renamed.py",
        source_harbor,
        root_program.sha256,
    )
    for gate in (
        {"memory_recall_gate": False},
        {"memory_context_gate": False},
        {"memory_candidate_gate": False},
    ):
        gated_result = correction.prepare_autonomous_correction_v1(
            policy,
            frontier3,
            gate_spec,
            action_memory=memory4,
            **gate,
        ).repair()
        _assert_preaction_stop(gated_result)

    forged_root = tmp_path / "forged-root"
    _stage_repository(forged_root, "solution.py", source_harbor)
    forged_root_result = correction.prepare_autonomous_correction_v1(
        policy,
        frontier3,
        _control_spec(
            forged_root,
            "solution.py",
            source_harbor,
            root_program.sha256,
            root_source_sha256="0" * 64,
        ),
        action_memory=memory4,
    ).repair()
    _assert_preaction_stop(forged_root_result)

    final_summary = memory4.training_summary
    assert total_materializations == 6
    assert total_repository_tests == 6
    assert sum(
        item.work.unselected_source_materializations
        for item in (result1, result2, result3, result4)
    ) == 0
    assert sum(
        item.work.unselected_repository_tests
        for item in (result1, result2, result3, result4)
    ) == 0
    assert final_summary.context_count == 2
    assert final_summary.provenance_count == 6
    assert final_summary.parent_lesson_replay_count == 0
    assert final_summary.retained_parent_count == 0
    assert final_summary.raw_lesson_retention_count == 0
    assert final_summary.parent_mutation_count == 0
    assert final_summary.added_cell_count <= memory.MAX_ADDED_CELL_COUNT
    assert final_summary.trainable_state_bytes <= memory.MAX_TRAINABLE_STATE_BYTES
    assert final_summary.checkpoint_array_bytes <= memory.MAX_CHECKPOINT_ARRAY_BYTES
    assert final_summary.learning_cell_steps <= memory.MAX_CONTROLLER_CELL_STEPS
    memory4.authenticate()

    lower_source = inspect.getsource(memory)
    correction_source = inspect.getsource(correction)
    assert "common_microrepository_autonomous_correction_v1" not in lower_source
    assert "expected_manifest" not in lower_source
    assert "test_command" not in lower_source
    assert "_seal_physical_repository_action_lessons_v1" not in memory.__all__
    assert "learned_evidence" not in inspect.signature(
        correction.prepare_autonomous_correction_v1
    ).parameters
    assert correction_source.count("fit_string_relation_policy_v2(") == 0

    elapsed = time.perf_counter() - started
    assert elapsed < 240.0
    print(
        json.dumps(
            {
                "elapsed_seconds": round(elapsed, 6),
                "policy_checkpoint_sha256": policy.checkpoint_sha256,
                "result_sha256s": [
                    result.sha256 for result in (result1, result2, result3, result4)
                ],
                "memory_checkpoint_sha256s": [
                    value.checkpoint_sha256
                    for value in (memory1, memory2, memory3, contradiction, memory4)
                ],
                "repository_tests": total_repository_tests,
                "source_materializations": total_materializations,
                "memory_cell_steps": final_summary.learning_cell_steps,
                "checkpoint_array_bytes": final_summary.checkpoint_array_bytes,
                "trainable_state_bytes": final_summary.trainable_state_bytes,
                "claim_boundary": memory.CLAIM_BOUNDARY,
            },
            sort_keys=True,
        )
    )
