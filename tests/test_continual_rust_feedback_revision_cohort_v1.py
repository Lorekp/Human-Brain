from __future__ import annotations

from dataclasses import fields
from hashlib import sha256

import pytest
from experiments.r3_pilot import common_continual_documentation_action_memory_v1 as continual
from experiments.r3_pilot import common_promoted_rust_action_vault_v1 as promoted_vault
from experiments.r3_pilot import common_rust_action_intervention_v1 as intervention
from experiments.r3_pilot import continual_rust_feedback_revision_cohort_v1 as cohort
from humanbrain import rust_selected_action_executor

# Public development expectations, not a secret or protected oracle.
PUBLIC_ORACLE_OUTPUTS = (
    ((2, 2, 7, 9), (-4, 0, 1, 3)),
    ((1, 8, -4), (5, -2, 0, 5)),
)


def test_public_cohort_revises_after_real_failure_and_retains_unrelated_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed_execution_roots: list[str] = []
    oracle_execution_counts: list[int] = []
    execute_training_selection = intervention.PreparedRustActionInterventionV1.execute_selected
    execute_recalled_selection = (
        rust_selected_action_executor.PreparedRustSelectedActionExecutorV2.execute_selected_once
    )

    def observe_training_selection(self, selection, batch):
        execution = execute_training_selection(self, selection, batch)
        completed_execution_roots.append(execution.sha256)
        return execution

    monkeypatch.setattr(
        intervention.PreparedRustActionInterventionV1,
        "execute_selected",
        observe_training_selection,
    )

    def observe_recalled_selection(self, selection):
        execution = execute_recalled_selection(self, selection)
        completed_execution_roots.append(execution.sha256)
        return execution

    monkeypatch.setattr(
        rust_selected_action_executor.PreparedRustSelectedActionExecutorV2,
        "execute_selected_once",
        observe_recalled_selection,
    )

    def public_oracle() -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
        # The public callback must not be invoked until all nine executions finish.
        assert len(completed_execution_roots) == 9
        oracle_execution_counts.append(len(completed_execution_roots))
        return PUBLIC_ORACLE_OUTPUTS

    spec = cohort.DEFAULT_PUBLIC_SPEC
    spec.authenticate()
    result = cohort.run_cohort_v1(spec=spec, oracle_callback=public_oracle)
    result.authenticate()

    assert result.status == "PASS_CONTINUAL_RUST_FEEDBACK_REVISION_COHORT_V1"
    assert result.spec is spec
    assert result.claim_boundary == cohort.CLAIM_BOUNDARY
    assert oracle_execution_counts == [9]

    attempts = result.attempts
    assert tuple(attempt.ordinal for attempt in attempts) == tuple(range(1, 10))
    assert tuple(attempt.case_label for attempt in attempts) == (
        "A1",
        "A2",
        "B1",
        "B2",
        "A3",
        "ADAPT_FAIL",
        "ADAPT_PASS",
        "R_PRIME",
        "U_PRIME",
    )
    assert tuple(attempt.phase for attempt in attempts) == (
        "TRAIN",
        "TRAIN",
        "TRAIN",
        "TRAIN",
        "TRAIN",
        "ADAPTATION_FAIL",
        "ADAPTATION_PASS",
        "EVALUATION",
        "EVALUATION",
    )

    candidate_a, candidate_b = result.candidate_sha256s
    expected_candidates = (
        candidate_a,
        candidate_a,
        candidate_b,
        candidate_b,
        candidate_a,
        candidate_a,
        candidate_b,
        candidate_b,
        candidate_a,
    )
    assert tuple(attempt.selected_candidate_sha256 for attempt in attempts) == expected_candidates

    expected_candidate_objects = (
        cohort.ACTION_A_CANDIDATE,
        cohort.ACTION_A_CANDIDATE,
        cohort.ACTION_B_CANDIDATE,
        cohort.ACTION_B_CANDIDATE,
        cohort.ACTION_A_CANDIDATE,
        cohort.ACTION_A_CANDIDATE,
        cohort.ACTION_B_CANDIDATE,
        cohort.ACTION_B_CANDIDATE,
        cohort.ACTION_A_CANDIDATE,
    )
    material_labels = (
        "A1",
        "A2",
        "B1",
        "B2",
        "A3",
        "ADAPT",
        "ADAPT",
        "R_PRIME",
        "U_PRIME",
    )
    expected_materials = tuple(cohort._task_material(spec, label) for label in material_labels)
    expected_batches = (
        *spec.training_batches,
        spec.adaptation_batch,
        spec.adaptation_batch,
        *spec.evaluation_batches,
    )
    adaptation_expected_sha256 = cohort.feedback_revision_output_sha256_v1(
        spec.adaptation_expected_outputs
    )
    expected_output_sha256s = (
        *tuple(
            cohort.feedback_revision_output_sha256_v1(outputs)
            for outputs in spec.training_expected_outputs
        ),
        adaptation_expected_sha256,
        adaptation_expected_sha256,
        *spec.evaluation_expected_output_sha256s,
    )
    assert tuple(
        (
            attempt.task_sha256,
            attempt.task_instance_sha256,
            attempt.task_source_sha256,
            attempt.expected_output_sha256,
        )
        for attempt in attempts
    ) == tuple(
        (
            material.task.sha256,
            material.task_instance_sha256,
            material.task_source_sha256,
            expected_output_sha256,
        )
        for material, expected_output_sha256 in zip(
            expected_materials,
            expected_output_sha256s,
            strict=True,
        )
    )

    expected_lesson_outputs = (
        *spec.training_expected_outputs,
        spec.adaptation_expected_outputs,
        spec.adaptation_expected_outputs,
    )
    assert tuple(
        (
            lesson.task,
            lesson.task_instance_sha256,
            lesson.task_source_sha256,
            lesson.evidence,
            lesson.expected_outputs,
        )
        for lesson in result.lessons
    ) == tuple(
        (
            material.task,
            material.task_instance_sha256,
            material.task_source_sha256,
            material.evidence,
            expected_outputs,
        )
        for material, expected_outputs in zip(
            expected_materials[:7],
            expected_lesson_outputs,
            strict=True,
        )
    )

    assert len(result.intervention_selections) == 5
    assert len(result.promoted_vault_receipts) == 4
    assert len(result.promoted_vault_authorities) == 4
    assert len(result.continual_recall_selections) == 4
    assert len(result.executions) == 9
    assert len(result.lessons) == 7

    intervention_authority = result.intervention_authority
    intervention_catalogue = result.intervention_catalogue
    assert intervention_authority is not None
    assert intervention_catalogue is not None
    intervention_authority.authenticate_structure()
    intervention_catalogue.authenticate_structure()
    canonical_candidate_sources = tuple(
        sorted(cohort.CANDIDATE_SOURCES, key=lambda pair: pair[0].sha256)
    )
    intervention_bindings = tuple(
        intervention._make_binding(candidate, source)
        for candidate, source in canonical_candidate_sources
    )
    assert intervention_catalogue.candidate_sha256s == tuple(
        candidate.sha256 for candidate, _source in canonical_candidate_sources
    )
    assert intervention_catalogue.binding_sha256s == tuple(
        binding.sha256 for binding in intervention_bindings
    )
    assert intervention_catalogue.rust_source_sha256s == tuple(
        sha256(source.encode("ascii", errors="strict")).hexdigest()
        for _candidate, source in canonical_candidate_sources
    )
    assert intervention_catalogue.action_count == len(canonical_candidate_sources)
    assert intervention_authority.catalogue_sha256 == intervention_catalogue.sha256

    expected_intervention_sha256s = tuple(
        cohort._digest(
            cohort._INTERVENTION_NAMESPACE,
            {
                "spec_sha256": spec.sha256,
                "case_label": attempt.case_label,
                "task_sha256": material.task.sha256,
                "candidate_sha256": candidate.sha256,
                "batch": batch,
                "expected_output_sha256": (
                    cohort.feedback_revision_output_sha256_v1(expected_outputs)
                ),
            },
        )
        for attempt, material, candidate, batch, expected_outputs in zip(
            attempts[:5],
            expected_materials[:5],
            expected_candidate_objects[:5],
            spec.training_batches,
            spec.training_expected_outputs,
            strict=True,
        )
    )
    assert (
        tuple(selection.intervention_sha256 for selection in result.intervention_selections)
        == expected_intervention_sha256s
    )
    intervention_bindings_by_candidate = {
        binding.candidate.sha256: binding for binding in intervention_bindings
    }
    expected_intervention_release_sha256s = tuple(
        intervention._make_release(
            selection,
            intervention_bindings_by_candidate[selection.candidate_sha256],
        ).sha256
        for selection in result.intervention_selections
    )
    for selection, execution, candidate in zip(
        result.intervention_selections,
        result.executions[:5],
        expected_candidate_objects[:5],
        strict=True,
    ):
        assert selection.authority_sha256 == intervention_authority.sha256
        assert selection.catalogue_sha256 == intervention_catalogue.sha256
        assert selection.candidate_sha256 == candidate.sha256
        assert execution.selected_source.authority_sha256 == intervention_authority.sha256
        assert execution.selected_source.selection_sha256 == selection.sha256
        assert execution.selected_source.action_handle == candidate.candidate_id
    assert (
        tuple(execution.selected_source.release_sha256 for execution in result.executions[:5])
        == expected_intervention_release_sha256s
    )

    for selection in result.intervention_selections:
        selection.authenticate_structure()
    for receipt in result.promoted_vault_receipts:
        receipt.authenticate_structure()
    for authority in result.promoted_vault_authorities:
        authority.authenticate_structure()
    for recall in result.continual_recall_selections:
        recall.authenticate()
    for execution in result.executions:
        execution.authenticate_structure()
    for lesson in result.lessons:
        lesson.authenticate()

    expected_batch_commitments = tuple(
        cohort._executor_batch_commitments(batch) for batch in expected_batches
    )
    assert (
        tuple(
            (
                execution.preparation.batch_sha256,
                execution.preparation.stdin_sha256,
                execution.preparation.case_count,
            )
            for execution in result.executions
        )
        == expected_batch_commitments
    )

    selection_roots = tuple(
        selection.sha256 for selection in result.intervention_selections
    ) + tuple(recall.sha256 for recall in result.continual_recall_selections)
    assert len(set(selection_roots)) == 9
    assert selection_roots == tuple(attempt.selection_sha256 for attempt in attempts)
    assert (
        tuple(selection.candidate_sha256 for selection in result.intervention_selections)
        == expected_candidates[:5]
    )
    assert (
        tuple(recall.selected_candidate_sha256 for recall in result.continual_recall_selections)
        == expected_candidates[5:]
    )
    assert tuple(execution.sha256 for execution in result.executions) == tuple(
        completed_execution_roots
    )
    assert tuple(attempt.execution_sha256 for attempt in attempts) == tuple(
        completed_execution_roots
    )
    assert tuple(lesson.sha256 for lesson in result.lessons) == tuple(
        attempt.lesson_sha256 for attempt in attempts[:7]
    )

    expected_vault_checkpoints = (
        result.trained_checkpoint_sha256,
        result.failed_checkpoint_sha256,
        result.final_checkpoint_sha256,
        result.final_checkpoint_sha256,
    )
    lesson_provenance = tuple(
        continual._provenance_from_lesson(lesson) for lesson in result.lessons
    )
    expected_vault_provenance = (
        lesson_provenance[:5],
        lesson_provenance[:6],
        lesson_provenance[:7],
        lesson_provenance[:7],
    )
    source_by_candidate_sha256 = {
        candidate.sha256: source for candidate, source in cohort.CANDIDATE_SOURCES
    }
    vault_bindings = tuple(
        promoted_vault._make_binding(
            candidate,
            source_by_candidate_sha256[candidate.sha256],
        )
        for candidate in (cohort.ACTION_A_CANDIDATE, cohort.ACTION_B_CANDIDATE)
    )
    vault_bindings_by_candidate = {binding.candidate.sha256: binding for binding in vault_bindings}
    expected_vault_release_sha256s: list[str] = []
    for offset, (receipt, authority, recall, execution) in enumerate(
        zip(
            result.promoted_vault_receipts,
            result.promoted_vault_authorities,
            result.continual_recall_selections,
            result.executions[5:],
            strict=True,
        )
    ):
        assert receipt.memory_checkpoint_sha256 == expected_vault_checkpoints[offset]
        assert receipt.memory_checkpoint_sha256 == recall.checkpoint_sha256
        assert receipt.memory_checkpoint_sha256 == attempts[offset + 5].checkpoint_before_sha256
        assert receipt.candidate_sha256s == (candidate_a, candidate_b)
        assert receipt.promoted_candidate_sha256s == (candidate_a, candidate_b)
        assert receipt.binding_sha256s == tuple(binding.sha256 for binding in vault_bindings)
        assert receipt.rust_source_sha256s == tuple(
            binding.rust_source_sha256 for binding in vault_bindings
        )
        provenance_prefix = expected_vault_provenance[offset]
        assert receipt.provenance_sha256s == tuple(row.sha256 for row in provenance_prefix)
        assert authority.vault_receipt_sha256 == receipt.sha256
        assert execution.selected_source.authority_sha256 == authority.sha256
        assert execution.selected_source.selection_sha256 == recall.sha256
        selected_candidate = expected_candidate_objects[offset + 5]
        selected_binding = vault_bindings_by_candidate[selected_candidate.sha256]
        promotion_provenance_sha256s = tuple(
            row.sha256
            for row in provenance_prefix
            if row.candidate_sha256 == selected_candidate.sha256 and row.passed_cell == 1
        )
        expected_vault_release_sha256s.append(
            promoted_vault._make_release(
                authority_sha256=authority.sha256,
                selection=recall,
                binding=selected_binding,
                promotion_provenance_sha256s=promotion_provenance_sha256s,
            ).sha256
        )
    assert len({authority.sha256 for authority in result.promoted_vault_authorities}) == 4
    assert tuple(execution.selected_source.release_sha256 for execution in result.executions) == (
        *expected_intervention_release_sha256s,
        *expected_vault_release_sha256s,
    )

    failed, revised, fresh_revisable, fresh_unrelated = attempts[5:]
    failed_execution, revised_execution = result.executions[5:7]
    failed_lesson, revised_lesson = result.lessons[5:7]
    assert failed.task_sha256 == revised.task_sha256
    assert failed.task_instance_sha256 == revised.task_instance_sha256
    assert failed.task_source_sha256 == revised.task_source_sha256
    assert failed_lesson.task == revised_lesson.task
    assert failed_lesson.task_instance_sha256 == revised_lesson.task_instance_sha256
    assert failed_lesson.task_source_sha256 == revised_lesson.task_source_sha256
    assert failed_lesson.evidence == revised_lesson.evidence
    assert failed_execution.preparation.batch_sha256 == revised_execution.preparation.batch_sha256
    assert failed_execution.preparation.stdin_sha256 == revised_execution.preparation.stdin_sha256
    assert len({failed.selection_sha256, revised.selection_sha256}) == 2
    assert len({failed.execution_sha256, revised.execution_sha256}) == 2
    assert len({failed.public_feedback_sha256, revised.public_feedback_sha256}) == 2
    assert len({failed.lesson_sha256, revised.lesson_sha256}) == 2
    assert failed.candidate_scores == (
        (0, candidate_a, 1, 3, 0, 3),
        (1, candidate_b, 1, 2, 0, 2),
    )
    assert revised.candidate_scores == (
        (0, candidate_a, 1, 3, 2, 1),
        (1, candidate_b, 1, 2, 0, 2),
    )
    control = result.failure_update_off_recall
    assert control is not None
    control.authenticate()
    assert control.selected_candidate_sha256 == candidate_a
    assert control.checkpoint_sha256 == result.trained_checkpoint_sha256
    assert control.task_sha256 == failed.task_sha256
    assert control.candidate_scores == failed.candidate_scores
    assert control.sha256 == result.continual_recall_selections[0].sha256
    assert failed.checkpoint_before_sha256 == result.trained_checkpoint_sha256
    assert failed.checkpoint_after_sha256 == result.failed_checkpoint_sha256
    assert revised.checkpoint_before_sha256 == result.failed_checkpoint_sha256
    assert failed.selected_candidate_sha256 == candidate_a
    assert revised.selected_candidate_sha256 == candidate_b
    assert fresh_revisable.candidate_scores == (
        (0, candidate_a, 1, 3, 2, 1),
        (1, candidate_b, 1, 4, 0, 4),
    )
    assert fresh_unrelated.candidate_scores == (
        (0, candidate_a, 1, 1, 0, 1),
        (1, candidate_b, 1, 0, 0, 0),
    )
    assert (failed.passed, revised.passed) == (0, 1)
    assert failed.expected_output_sha256 != failed.observed_output_sha256
    assert revised.expected_output_sha256 == revised.observed_output_sha256
    assert tuple(attempt.passed for attempt in attempts[7:]) == (1, 1)
    assert all(
        attempt.expected_output_sha256 == attempt.observed_output_sha256
        for attempt in (*attempts[:5], revised, *attempts[7:])
    )

    assert tuple(attempt.memory_updated for attempt in attempts) == (
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
    )
    lineage = result.checkpoint_lineage_sha256s
    assert len(lineage) == 7
    assert len(set(lineage)) == 7
    assert result.initial_checkpoint_sha256 == lineage[0]
    assert result.trained_checkpoint_sha256 == lineage[4]
    assert result.failed_checkpoint_sha256 == lineage[5]
    assert result.final_checkpoint_sha256 == lineage[6]
    updated_attempts = attempts[:7]
    assert tuple(attempt.checkpoint_after_sha256 for attempt in updated_attempts) == lineage
    assert (
        tuple(attempt.checkpoint_before_sha256 for attempt in updated_attempts[1:]) == lineage[:-1]
    )
    assert all(
        attempt.parent_checkpoint_after_sha256 == attempt.checkpoint_before_sha256
        for attempt in attempts
    )
    assert all(
        attempt.checkpoint_before_sha256
        == attempt.checkpoint_after_sha256
        == result.final_checkpoint_sha256
        for attempt in attempts[7:]
    )

    work = result.work
    assert work.source_release_count == 9
    assert work.compile_count == 9
    assert work.run_count == 9
    assert work.unselected_execution_count == 0
    assert work.memory_update_count == 7
    assert work.evaluation_memory_update_count == 0
    assert work.intervention_issue_count == 5
    assert work.vault_recall_count == 4
    assert work.oracle_callback_count == 1
    assert work.oracle_comparison_count == 2
    assert work.parent_mutation_count == 0
    assert work.parent_lesson_replay_count == 0
    assert work.source_release_count == len(result.executions)
    assert work.compile_count == sum(
        execution.compile_receipt.compile_invocations for execution in result.executions
    )
    assert work.run_count == sum(
        execution.run_receipt.run_invocations for execution in result.executions
    )
    assert work.unselected_execution_count == sum(
        execution.selected_source.selection_sha256 not in set(selection_roots)
        for execution in result.executions
    )
    assert work.memory_update_count == sum(attempt.memory_updated for attempt in result.attempts)
    assert work.evaluation_memory_update_count == sum(
        attempt.memory_updated for attempt in result.evaluation_attempts
    )
    assert work.oracle_callback_count == int(result.oracle_observation_sha256 is not None)
    assert work.parent_mutation_count == sum(
        attempt.parent_checkpoint_after_sha256 != attempt.checkpoint_before_sha256
        for attempt in result.attempts
    )
    assert work.parent_lesson_replay_count == len(result.lessons) - len(
        {lesson.sha256 for lesson in result.lessons}
    )
    assert work.intervention_issue_count == len(result.intervention_selections)
    assert work.vault_recall_count == len(result.continual_recall_selections)
    assert work.oracle_comparison_count == len(result.evaluation_match_cells)
    assert result.evaluation_match_cells == (1, 1)


@pytest.mark.parametrize(
    ("disabled_gate", "expected_status"),
    (
        (
            "cell_gate",
            "STOP_CONTINUAL_RUST_FEEDBACK_REVISION_COHORT_CELL_OFF",
        ),
        (
            "documentation_gate",
            "STOP_CONTINUAL_RUST_FEEDBACK_REVISION_COHORT_DOCUMENTATION_OFF",
        ),
    ),
)
def test_recall_gates_stop_before_any_execution_or_oracle(
    disabled_gate: str,
    expected_status: str,
) -> None:
    def forbidden_oracle() -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
        pytest.fail("a stopped cohort must not open the oracle")

    gates = {"cell_gate": True, "documentation_gate": True}
    gates[disabled_gate] = False
    result = cohort.run_cohort_v1(
        spec=cohort.DEFAULT_PUBLIC_SPEC,
        oracle_callback=forbidden_oracle,
        **gates,
    )
    result.authenticate()

    assert result.status == expected_status
    assert result.spec is cohort.DEFAULT_PUBLIC_SPEC
    assert result.attempts == ()
    assert result.intervention_authority is None
    assert result.intervention_catalogue is None
    assert result.intervention_selections == ()
    assert result.promoted_vault_receipts == ()
    assert result.promoted_vault_authorities == ()
    assert result.continual_recall_selections == ()
    assert result.executions == ()
    assert result.lessons == ()
    assert result.failure_update_off_recall is None
    assert result.evaluation_match_cells == ()
    assert result.checkpoint_lineage_sha256s == ()
    assert result.initial_checkpoint_sha256 is None
    assert result.trained_checkpoint_sha256 is None
    assert result.failed_checkpoint_sha256 is None
    assert result.final_checkpoint_sha256 is None
    assert result.oracle_observation_sha256 is None
    assert all(
        getattr(result.work, field.name) == 0
        for field in fields(result.work)
        if field.name.endswith("_count")
    )
