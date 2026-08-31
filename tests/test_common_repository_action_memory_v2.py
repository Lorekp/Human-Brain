from __future__ import annotations

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
from experiments.r3_pilot import common_microrepository_autonomous_correction_v1 as correction
from experiments.r3_pilot import common_repository_action_memory_v2 as memory
from experiments.r3_pilot import common_repository_source_views_v1 as source_views
from experiments.r3_pilot import common_rooted_structured_event_graph_v1 as structured_graph
from experiments.r3_pilot import common_structured_role_path_projection_v1 as role_projection
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import python_entry_return_provenance_v1 as python_provenance
from experiments.r3_pilot import typed_semantic_packed_absolute_map_fold_v1 as absolute
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature


def _predicate_checkpoint() -> gated.FrozenPredicateGateV1:
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
            value, declared_type=output_cells.OutputType.INT
        )
        for value in values
    )


def _fit_recurrent_policy_once(
) -> recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2:
    signature = absolute.list_int_to_int_signature()
    publication = plan_bank.CausalTaskObservationCandidatePlanBankV1.build(
        _predicate_checkpoint(), signature, (([64],), ([-2],), ([4],), ([5],))
    )
    library = fragments.fit_reusable_program_fragment_library_v1(
        publication,
        (
            _encoded_ints((64, 2, 4, 5)),
            _encoded_ints((0, -2, 4, 0)),
        ),
    )
    cases = (
        ([-8, 7, -1],),
        ([6, 5, 4, -3],),
        ([-2, 3, 9],),
        ([12, -11, 0, 5],),
        ([-9, 2, 6],),
        ([5, -4, 3],),
        ([8, 7],),
        ([-1, 0, 12],),
    )
    frontier = library.build(signature, cases)
    return recurrent.fit_causal_recurrent_fragment_relation_policy_v2(
        library,
        (
            recurrent.FragmentRelationLesson(
                frontier,
                frontier.select(_encoded_ints((-1, 1, 1, 0, 2, -1, 0, -1))),
            ),
            recurrent.FragmentRelationLesson(
                frontier,
                frontier.select(_encoded_ints((9, 7, 5, 11, 11, 7, 0, 1))),
            ),
        ),
    )


def _string_signature() -> TypedSignature:
    string = ProgramType.String()
    return TypedSignature(
        entry_point="clean_text",
        parameters=(ProgramParameter("payload", string),),
        result=string,
    )


def _string_frontier() -> tuple[object, object]:
    recurrent_policy = _fit_recurrent_policy_once()
    signature = _string_signature()
    training = string_repair.solve_string_relation_training_v2(
        recurrent_policy,
        signature,
        (("A1e2Z9",), ("u3B4i8",), ("O5x6a0",), ("e7Y8U9",)),
    )
    policy = string_repair.fit_string_relation_policy_v2(
        recurrent_policy, training
    )
    inputs = ("a1b2c1", "x9y8z9", "zero00", "no-digits")
    expected = ("121", "989", "00", "")
    frontier = policy.build(signature, tuple((value,) for value in inputs))
    encoded = frontier.encode_visible_expected(list(expected))
    assert all(record.physical_outputs == encoded for record in frontier.repository_action_bank.records)
    return frontier, training.identity_program


def _observation(source: bytes) -> memory.RepositoryActionObservationV2:
    provenance = python_provenance.observe_python_entry_return_provenance_v1(
        source, entry_point="clean_text"
    )
    provenance.authenticate()
    receipt = structured_graph.make_structured_event_graph_producer_receipt_v1(
        source_sha256=provenance.source_sha256,
        graph_sha256=provenance.graph.sha256,
        producer_schema_sha256=provenance.schema_sha256,
        producer_attestation_sha256=provenance.sha256,
    )
    roles = role_projection.project_structured_role_paths_v1(provenance.graph)
    views = source_views.project_repository_source_views_v1(source)
    return memory.make_repository_action_observation_v2(
        views,
        roles,
        producer_receipt=receipt,
    )


def _committed_lessons(
    root: Path,
    source: bytes,
    *,
    frontier: object,
    root_program: object,
    hidden_expected: str,
) -> tuple[object, ...]:
    root.mkdir()
    source_name = "solution.py"
    manifest_name = "visible.json"
    case_name = "hidden.case.json"
    test_name = "check_hidden.py"
    (root / source_name).write_bytes(source)
    (root / manifest_name).write_text(
        json.dumps({"expected": ["121", "989", "00", ""]}),
        encoding="utf-8",
    )
    (root / case_name).write_text(
        json.dumps({"input": "A1b2!", "expected": hidden_expected}),
        encoding="utf-8",
    )
    (root / test_name).write_text(
        "import json\n"
        "from pathlib import Path\n"
        "root=Path(__file__).parent\n"
        f"ns={{}}; p=root/{source_name!r}\n"
        "exec(compile(p.read_text(encoding='utf-8'),str(p),'exec'),ns)\n"
        f"r=json.loads((root/{case_name!r}).read_text(encoding='utf-8'))\n"
        "raise SystemExit(0 if ns['clean_text'](r['input'])==r['expected'] else 3)\n",
        encoding="utf-8",
    )
    spec = correction.AutonomousCorrectionSpecV1(
        repo_root=root,
        source_relative_path=source_name,
        expected_manifest_relative_path=manifest_name,
        test_command=(sys.executable, "-B", test_name),
        root_program_sha256=root_program.sha256,
        root_source_sha256=sha256(source).hexdigest(),
    )
    result = correction.prepare_autonomous_correction_v1(frontier, spec).repair()
    result.authenticate()
    assert result.status == correction.COMMITTED
    return result.repository_actions


def test_counterbalanced_causal_role_transfer_and_controls(tmp_path: Path) -> None:
    frontier, root_program = _string_frontier()
    bank = frontier.repository_action_bank
    training_sources = (
        b"def clean_text(alpha):\n    r0=alpha[::-1]\n    i0=alpha[:]\n    r1=alpha[::-1]\n    i1=alpha[:]\n    return r0\n",
        b"def clean_text(beta):\n    j1=beta[:]\n    q1=beta[::-1]\n    j0=beta[:]\n    q0=beta[::-1]\n    return q0\n",
        b"def clean_text(gamma):\n    r0=gamma[::-1]\n    i0=gamma[:]\n    r1=gamma[::-1]\n    i1=gamma[:]\n    return i0\n",
        b"def clean_text(delta):\n    j1=delta[:]\n    q1=delta[::-1]\n    j0=delta[:]\n    q0=delta[::-1]\n    return j0\n",
    )
    hidden_expected = ("21", "21", "12", "12")
    observations = tuple(_observation(source) for source in training_sources)
    with pytest.raises(ValueError, match="observation authentication"):
        memory.make_repository_action_observation_v2(
            observations[0].source_views,
            observations[0].structural_roles,
            producer_receipt=observations[1].producer_receipt,
        )
    state: memory.FrozenRepositoryActionMemoryV2 | None = None
    experience_roots: list[str] = []
    for index, (observation, source, expected) in enumerate(
        zip(observations, training_sources, hidden_expected, strict=True)
    ):
        lessons = _committed_lessons(
            tmp_path / f"train-{index}",
            source,
            frontier=frontier,
            root_program=root_program,
            hidden_expected=expected,
        )
        experience = memory.make_repository_action_experience_v2(
            observation, lessons
        )
        state = (
            memory.start_repository_action_memory_v2(experience)
            if state is None
            else memory.advance_repository_action_memory_v2(state, experience)
        )
        experience_roots.append(experience.sha256)

    assert state is not None
    state.authenticate()
    summary = state.training_summary
    assert summary.source_count == 4
    assert summary.structural_role_factor_count == 8
    assert summary.admitted_structural_role_factor_count == 4
    assert summary.trainable_state_bytes <= memory.MAX_TRAINABLE_STATE_BYTES
    assert summary.checkpoint_array_bytes <= memory.MAX_CHECKPOINT_ARRAY_BYTES
    assert summary.raw_experience_retention_count == 0
    assert summary.parent_mutation_count == 0
    assert len(state.source_provenance) == 4
    assert tuple(item.experience_sha256 for item in state.source_provenance) == tuple(
        experience_roots
    )
    assert all(
        admission.selected_candidate_action_sha256 is None
        for admission in state.lexical_admissions
    )

    role_admissions = state.structural_role_admissions
    admitted = tuple(
        admission
        for admission in role_admissions
        if admission.selected_candidate_action_sha256 is not None
    )
    decoys = tuple(
        admission
        for admission in role_admissions
        if admission.selected_candidate_action_sha256 is None
    )
    assert len(admitted) == 4
    assert len(decoys) == 4
    assert {admission.distinct_source_count for admission in admitted} == {2}
    assert {admission.distinct_source_count for admission in decoys} == {4}
    assert len({item.selected_candidate_action_sha256 for item in admitted}) == 2

    frozen_root = state.checkpoint_sha256
    exact = state.recall_attested_observation(observations[0], bank)
    exact.authenticate()
    assert exact.status == memory.RECALLED
    assert exact.tier == memory.EXACT_EPISODIC
    assert exact.work.role_applicability_race_cell_steps == 0
    assert exact.work.role_factor_match_cell_steps == 0
    assert exact.work.lexical_factor_match_cell_steps == 0

    reverse_transfer = _observation(
        b"def clean_text(eta):\n    a=eta[:]\n    b=eta[::-1]\n    c=eta[::-1]\n    d=eta[:]\n    return c\n"
    )
    identity_transfer = _observation(
        b"def clean_text(theta):\n    a=theta[::-1]\n    b=theta[:]\n    c=theta[:]\n    d=theta[::-1]\n    return c\n"
    )
    reverse = state.recall_attested_observation(reverse_transfer, bank)
    identity = state.recall_attested_observation(identity_transfer, bank)
    reverse.authenticate()
    identity.authenticate()
    assert (reverse.status, identity.status) == (memory.RECALLED, memory.RECALLED)
    assert (reverse.tier, identity.tier) == (
        memory.STRUCTURAL_ROLE,
        memory.STRUCTURAL_ROLE,
    )
    forged_role_with_exact_work = reverse._replace(work=exact.work, sha256="")
    forged_role_with_exact_work = forged_role_with_exact_work._replace(
        sha256=memory._digest(
            memory._RECALL_NAMESPACE,
            memory._recall_payload(forged_role_with_exact_work._asdict()),
        )
    )
    with pytest.raises(ValueError, match="recall authentication"):
        forged_role_with_exact_work.authenticate()
    assert reverse.selected_candidate_action_sha256 != identity.selected_candidate_action_sha256
    assert reverse.work.role_factor_match_cell_steps > 0
    assert reverse.work.lexical_factor_match_cell_steps == 0
    assert reverse.work.total_cell_steps <= memory.MAX_RECALL_CELL_STEPS
    assert reverse.work.total_reads <= memory.MAX_RECALL_ROOT_READS

    reversed_candidates = tuple(reversed(bank.records))
    order_control = state.recall_attested_observation(
        reverse_transfer, bank, candidate_records=reversed_candidates
    )
    order_control.authenticate()
    assert order_control.status == memory.RECALLED
    assert order_control.selected_candidate_action_sha256 == reverse.selected_candidate_action_sha256
    assert order_control.candidates == reversed_candidates

    role_gate = state.recall_attested_observation(
        reverse_transfer, bank, structural_role_gate=False
    )
    assert role_gate.status == memory.STOP_STRUCTURAL_ROLE_GATE
    assert role_gate.work.role_applicability_race_cell_steps == 0
    exact_gate = state.recall_attested_observation(
        observations[0], bank, exact_candidate_gate=False
    )
    assert exact_gate.status == memory.STOP_EXACT_CANDIDATE_GATE
    assert exact_gate.work.role_applicability_race_cell_steps == 0

    unknown = _observation(
        b"def clean_text(iota):\n    bad=iota[1:]\n    r=iota[::-1]\n    i=iota[:]\n    d=iota[::-1]\n    return bad\n"
    )
    unknown_recall = state.recall_attested_observation(unknown, bank)
    unknown_recall.authenticate()
    assert unknown_recall.status == memory.STOP_LEXICAL_FACTOR_UNKNOWN
    assert unknown_recall.tier == memory.LEXICAL_FACTOR
    assert unknown_recall.work.role_applicability_race_cell_steps > 0
    assert unknown_recall.work.role_factor_match_cell_steps == 0
    lexical_gate = state.recall_attested_observation(
        unknown, bank, lexical_factor_gate=False
    )
    assert lexical_gate.status == memory.STOP_LEXICAL_FACTOR_GATE

    conflict = _observation(
        b"def clean_text(kappa):\n    out=kappa[::-1] if len(kappa)%2 else kappa[:]\n    r=kappa[::-1]\n    i=kappa[:]\n    return out\n"
    )
    conflict_recall = state.recall_attested_observation(conflict, bank)
    conflict_recall.authenticate()
    assert conflict_recall.status == memory.STOP_STRUCTURAL_ROLE_CONFLICT
    assert conflict_recall.tier == memory.STRUCTURAL_ROLE
    assert conflict_recall.work.role_factor_match_cell_steps > 0
    assert conflict_recall.work.lexical_factor_match_cell_steps == 0

    state.authenticate()
    assert state.checkpoint_sha256 == frozen_root
