from __future__ import annotations

import inspect
import json
import sys
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import pytest

from experiments.r3_pilot import causal_microrepository_repair_loop_v2 as string_repair
from experiments.r3_pilot import causal_predicate_gated_fold_v1 as gated
from experiments.r3_pilot import causal_recurrent_fragment_relation_policy_v2 as recurrent
from experiments.r3_pilot import causal_reusable_program_fragment_library_v1 as fragments
from experiments.r3_pilot import causal_task_observation_candidate_plan_bank_v1 as plan_bank
from experiments.r3_pilot import common_continual_repository_action_memory_v1 as memory
from experiments.r3_pilot import common_factorized_repository_action_memory_v1 as factor_memory
from experiments.r3_pilot import common_microrepository_autonomous_correction_v1 as correction
from experiments.r3_pilot import common_repository_action_bank_v1 as action_bank
from experiments.r3_pilot import common_repository_source_views_v1 as source_views
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

    list_context = source_views.project_attested_repository_source_v1(list_source)
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

    string_context = source_views.project_attested_repository_source_v1(string_source)
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


def test_consequence_learned_lexical_factors_transfer_and_abstain(
    tmp_path: Path,
) -> None:
    recurrent_policy = _fit_recurrent_policy_once()
    signature = _string_signature()
    string_training = string_repair.solve_string_relation_training_v2(
        recurrent_policy,
        signature,
        (
            ("A1e2Z9",),
            ("u3B4i8",),
            ("O5x6a0",),
            ("e7Y8U9",),
        ),
    )
    policy = string_repair.fit_string_relation_policy_v2(
        recurrent_policy,
        string_training,
    )
    root_program = string_training.identity_program

    sources = (
        b"def f0(items):\n    return items[::-1]\n\n"
        b"def clean_text(payload):\n    return payload\n",
        b"def f1(values):\n\n    return values[ :: -1 ]\n\n"
        b"def clean_text(arg):\n    return arg\n",
        b"def f2(items):\n    return items[:]\n\n"
        b"def clean_text(payload):\n    return payload\n",
        b"def f3(values):\n\n    return values[ : ]\n\n"
        b"def clean_text(arg):\n    return arg\n",
        b"# zephyr quartz amber migration 3141592653589793\n"
        b"# deliberately byte-new transfer observation\n"
        b"def f4(seq):\n    return seq [::-1]\n\n"
        b"def clean_text(text):\n    return text\n",
        b"# cobalt juniper tidal migration 2718281828459045\n"
        b"# independently byte-new transfer observation\n"
        b"def f5(seq):\n    return seq [ :]\n\n"
        b"def clean_text(text):\n    return text\n",
    )
    fake_source = (
        b'def f6(seq):\n    x = "seq[::-1]"\n    return seq\n\n'
        b"# seq[:]\ndef clean_text(text):\n    return text\n"
    )
    neither_source = (
        b"def f7(seq):\n    return tuple(seq)\n\n"
        b"def clean_text(text):\n    return text\n"
    )
    both_source = (
        b"def f8(seq):\n    x = seq[::-1]\n    y = seq[:]\n    return x\n\n"
        b"def clean_text(text):\n    return text\n"
    )

    visible_case = (
        ("a1b2c1", "x9y8z9", "zero00", "no-digits"),
        ("121", "989", "00", ""),
    )
    frozen_frontier = _build_string_frontier(
        policy,
        signature,
        visible_case[0],
        visible_case[1],
    )
    # One neutral action bank is frozen before every fresh source observation.
    frontiers = (frozen_frontier,) * len(sources)
    visible_cases = (visible_case,) * len(sources)
    banks = tuple(frontier.repository_action_bank for frontier in frontiers)
    assert all(bank.action_sha256s == banks[0].action_sha256s for bank in banks[1:])
    assert all(_stable_program_map(bank) == _stable_program_map(banks[0]) for bank in banks[1:])

    layouts = tuple(
        (tmp_path / name, source_name, source)
        for name, source_name, source in zip(
            ("amber", "birch", "cobalt", "dune", "ember", "fjord"),
            ("a.py", "b.py", "c.py", "d.py", "e.py", "f.py"),
            sources,
            strict=True,
        )
    )
    for root, source_name, source in layouts:
        _stage_repository(root, source_name, source)
    observations = tuple(
        source_views.project_repository_source_views_v1(source) for source in sources
    )
    with pytest.raises(source_views.RepositorySourceCapacityError):
        source_views.project_repository_source_views_v1(b"#" * 257)

    hidden = (
        ("A1b2!", "21", [2]),
        ("C3d4?", "43", [2]),
        ("E5f6.", "56", [2, 1]),
        ("G7h8,", "78", [2, 1]),
    )
    results: list[correction.AutonomousCorrectionResultV1] = []
    experiences: list[factor_memory.FactorizedRepositoryActionExperienceV1] = []
    composite: factor_memory.FrozenFactorizedRepositoryActionMemoryV1 | None = None
    for index, (hidden_input, hidden_expected, expected_supports) in enumerate(hidden):
        root, source_name, source = layouts[index]
        spec = _install_hidden_fixture(
            root,
            source_name=source_name,
            source=source,
            root_program_sha256=root_program.sha256,
            visible_expected=visible_cases[index][1],
            entry_point="clean_text",
            hidden_input=hidden_input,
            hidden_expected=hidden_expected,
            manifest_name=f"visible-{index}.json",
            test_name=f"check-{index}.py",
        )
        result = correction.prepare_autonomous_correction_v1(
            frontiers[index],
            spec,
        ).repair()
        result.authenticate()
        assert result.status == correction.COMMITTED
        assert [item.receipt_cell_support for item in result.attempts] == expected_supports
        experience = factor_memory.make_factorized_repository_action_experience_v1(
            observations[index],
            result.repository_actions,
        )
        composite = (
            factor_memory.start_factorized_repository_action_memory_v1(experience)
            if composite is None
            else factor_memory.advance_factorized_repository_action_memory_v1(
                composite,
                experience,
            )
        )
        results.append(result)
        experiences.append(experience)

    assert composite is not None
    composite.authenticate()
    summary = composite.training_summary
    assert summary.policy_count == 1
    assert summary.source_count == 4
    assert summary.factor_count == 16
    assert summary.admitted_factor_count == 7
    assert summary.raw_experience_retention_count == 0
    assert summary.parent_mutation_count == 0
    assert summary.trainable_state_bytes == 96
    assert summary.checkpoint_array_bytes == 3_530
    assert summary.factor_learning_cell_steps == 151_200
    assert summary.learning_cell_steps == 169_200
    assert len(composite.source_provenance) == 4

    factor_sets = tuple(
        {factor.semantic_sha256 for factor in observation.lexical_view.factors}
        for observation in observations[:4]
    )
    reverse_only = (factor_sets[0] & factor_sets[1]) - factor_sets[2] - factor_sets[3]
    copy_only = (factor_sets[2] & factor_sets[3]) - factor_sets[0] - factor_sets[1]
    common = set.intersection(*factor_sets)
    admissions = {item.semantic_sha256: item for item in composite.factor_admissions}
    support_by_action = {
        banks[0].action_sha256_for(record): record.receipt_cell_support
        for record in banks[0].records
    }
    assert len(reverse_only) == 5
    assert len(copy_only) == 2
    assert all(admissions[root].distinct_source_count >= 2 for root in reverse_only | copy_only)
    assert all(
        support_by_action[admissions[root].selected_candidate_action_sha256] == 2
        for root in reverse_only
    )
    assert all(
        support_by_action[admissions[root].selected_candidate_action_sha256] == 1
        for root in copy_only
    )
    assert all(admissions[root].selected_candidate_action_sha256 is None for root in common)

    frozen_root = composite.checkpoint_sha256
    with pytest.raises(ValueError, match="replay"):
        factor_memory.advance_factorized_repository_action_memory_v1(
            composite,
            experiences[0],
        )
    composite.authenticate()
    assert composite.checkpoint_sha256 == frozen_root

    exact_recall = composite.recall_attested_source(
        sources[0],
        observations[0].exact_context,
        banks[0],
    )
    exact_recall.authenticate()
    assert exact_recall.status == memory.RECALLED
    assert exact_recall.tier == factor_memory.EXACT_EPISODIC
    assert exact_recall.work.lexical_projection_cell_steps == 0
    assert exact_recall.work.factor_match_cell_steps == 0
    assert exact_recall.work.factor_action_race_cell_steps == 0

    oversized_source = b"#" * 257
    with pytest.raises(source_views.RepositorySourceCapacityError):
        composite.recall_attested_source(
            oversized_source,
            observations[0].exact_context,
            banks[0],
        )
    oversized_root = tmp_path / "oversized"
    _stage_repository(oversized_root, "oversized.py", oversized_source)
    oversized_result = correction.prepare_autonomous_correction_v1(
        frontiers[0],
        _control_spec(
            oversized_root,
            "oversized.py",
            oversized_source,
            root_program.sha256,
        ),
        action_memory=composite,
    ).repair()
    _assert_preaction_stop(oversized_result, correction.STOP_LEARNED_RECALL)

    reverse_context = observations[4].exact_context
    copy_context = observations[5].exact_context
    gate_stop = composite.recall_attested_source(
        sources[4],
        reverse_context,
        banks[4],
        factor_gate=False,
    )
    gate_stop.authenticate()
    assert gate_stop.status == factor_memory.STOP_FACTOR_GATE
    assert gate_stop.tier == factor_memory.LEXICAL_FACTOR
    assert gate_stop.work.lexical_projection_cell_steps == 0
    assert gate_stop.work.factor_match_cell_steps == 0
    assert gate_stop.work.factor_action_race_cell_steps == 0
    controller_gate_stop = correction.prepare_autonomous_correction_v1(
        frontiers[4],
        _control_spec(
            layouts[4][0],
            layouts[4][1],
            sources[4],
            root_program.sha256,
        ),
        action_memory=composite,
        memory_factor_gate=False,
    ).repair()
    _assert_preaction_stop(controller_gate_stop, correction.STOP_LEARNED_RECALL)

    control_results = tuple(
        composite.recall_attested_source(
            source,
            source_views.project_attested_repository_source_v1(source),
            banks[4],
        )
        for source in (fake_source, neither_source, both_source)
    )
    for result in control_results:
        result.authenticate()
    assert tuple(result.status for result in control_results) == (
        factor_memory.STOP_FACTOR_UNKNOWN,
        factor_memory.STOP_FACTOR_UNKNOWN,
        factor_memory.STOP_FACTOR_CONFLICT,
    )
    assert control_results[0].work.factor_action_race_cell_steps == 0
    assert control_results[1].work.factor_action_race_cell_steps == 0
    assert control_results[2].work.factor_action_race_cell_steps > 0
    assert control_results[2].work.factor_consensus_cell_steps == 0

    reverse_recall = composite.recall_attested_source(
        sources[4],
        reverse_context,
        banks[4],
    )
    copy_recall = composite.recall_attested_source(
        sources[5],
        copy_context,
        banks[5],
    )
    reordered_copy_recall = composite.recall_attested_source(
        sources[5],
        copy_context,
        banks[5],
        candidate_records=tuple(reversed(banks[5].records)),
    )
    assert reverse_recall.status == memory.RECALLED
    assert copy_recall.status == memory.RECALLED
    assert reverse_recall.tier == factor_memory.LEXICAL_FACTOR
    assert copy_recall.tier == factor_memory.LEXICAL_FACTOR
    assert reordered_copy_recall == copy_recall
    assert support_by_action[reverse_recall.selected_candidate_action_sha256] == 2
    assert support_by_action[copy_recall.selected_candidate_action_sha256] == 1

    transfer_specs = (
        _install_hidden_fixture(
            layouts[4][0],
            source_name=layouts[4][1],
            source=sources[4],
            root_program_sha256=root_program.sha256,
            visible_expected=visible_cases[4][1],
            entry_point="clean_text",
            hidden_input="M2n5!",
            hidden_expected="52",
            manifest_name="fresh-reverse.json",
            test_name="fresh-reverse.py",
        ),
        _install_hidden_fixture(
            layouts[5][0],
            source_name=layouts[5][1],
            source=sources[5],
            root_program_sha256=root_program.sha256,
            visible_expected=visible_cases[5][1],
            entry_point="clean_text",
            hidden_input="P3q7?",
            hidden_expected="37",
            manifest_name="fresh-copy.json",
            test_name="fresh-copy.py",
        ),
    )
    transfer_results = (
        correction.prepare_autonomous_correction_v1(
            frontiers[4],
            transfer_specs[0],
            action_memory=composite,
        ).repair(),
        correction.prepare_autonomous_correction_v1(
            frontiers[5],
            transfer_specs[1],
            candidate_records=tuple(reversed(banks[5].records)),
            action_memory=composite,
        ).repair(),
    )
    for result in transfer_results:
        result.authenticate()
        assert result.status == correction.COMMITTED
        assert len(result.attempts) == 1
        assert result.work.unselected_source_materializations == 0
        assert result.work.unselected_repository_tests == 0
    assert [item.receipt_cell_support for item in transfer_results[0].attempts] == [2]
    assert [item.receipt_cell_support for item in transfer_results[1].attempts] == [1]
    composite.authenticate()
    assert composite.checkpoint_sha256 == frozen_root
