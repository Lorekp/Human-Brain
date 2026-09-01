from __future__ import annotations

import os
import shutil
from hashlib import sha256

import pytest
from experiments.r3_pilot import common_continual_token_edit_schema_v1 as continual
from experiments.r3_pilot import common_rust_action_intervention_v1 as intervention
from experiments.r3_pilot import continual_rust_token_edit_schema_cohort_v1 as cohort

pytestmark = pytest.mark.skipif(
    os.name != "nt" or shutil.which("rustc") is None or shutil.which("rustfmt") is None,
    reason="the trusted C1 cohort requires Windows, rustc, and rustfmt",
)


def test_trusted_public_c1_cohort_learns_then_only_proposes_fresh_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed_source_roots: list[str] = []
    execute_selected = intervention.PreparedRustActionInterventionV1.execute_selected

    def observe_selected(self, selection, batch):
        execution = execute_selected(self, selection, batch)
        executed_source_roots.append(execution.selected_source.source_sha256)
        return execution

    monkeypatch.setattr(
        intervention.PreparedRustActionInterventionV1,
        "execute_selected",
        observe_selected,
    )

    spec = cohort.DEFAULT_PUBLIC_SPEC
    spec.authenticate()
    result = cohort.run_cohort_v1(spec)
    result.authenticate()

    assert result.status == cohort.STATUS_C1_STRUCTURAL_PROPOSAL
    assert result.claim_boundary == cohort.CLAIM_BOUNDARY
    assert result.spec is spec
    assert tuple(item.label for item in spec.episodes) == ("A", "B")
    assert tuple(item.batch for item in spec.episodes) == (
        cohort.A_BATCH,
        cohort.B_BATCH,
    )
    assert tuple(item.expected_outputs for item in spec.episodes) == (
        cohort.A_EXPECTED_OUTPUTS,
        cohort.B_EXPECTED_OUTPUTS,
    )
    assert spec.learner_config_sha256 == continual.CONFIG_SHA256
    assert spec.candidate_sha256s == tuple(
        candidate.sha256 for candidate, _source in cohort.CANDIDATE_SOURCES
    )
    assert spec.training_source_sha256s == tuple(
        sha256(source.encode("ascii")).hexdigest()
        for _candidate, source in cohort.CANDIDATE_SOURCES
    )
    assert (
        spec.expected_fresh_edited_source_sha256
        == sha256(cohort.EXPECTED_FRESH_EDITED_SOURCE.encode("ascii")).hexdigest()
    )
    material_roots = tuple(
        root
        for item in spec.episodes
        for root in (
            item.repository_sha256,
            item.task_sha256,
            item.task_instance_sha256,
            item.task_source_sha256,
            item.test_source_sha256,
        )
    )
    assert len(set(material_roots)) == 10

    expected_candidates = (
        cohort.A_FAILED_CANDIDATE,
        cohort.A_SUCCESSFUL_CANDIDATE,
        cohort.B_FAILED_CANDIDATE,
        cohort.B_SUCCESSFUL_CANDIDATE,
    )
    expected_sources = (
        cohort.A_FAILED_SOURCE,
        cohort.A_SUCCESSFUL_SOURCE,
        cohort.B_FAILED_SOURCE,
        cohort.B_SUCCESSFUL_SOURCE,
    )
    expected_phases = (
        cohort.FAILED,
        cohort.SUCCESSFUL,
        cohort.FAILED,
        cohort.SUCCESSFUL,
    )
    assert tuple(item.candidate_id for item in expected_candidates) == (
        "slot_q7",
        "slot_z4",
        "slot_m2",
        "slot_v9",
    )
    assert result.candidate_sha256s == tuple(item.sha256 for item in expected_candidates)
    assert result.intervention_catalogue == intervention._make_catalogue(
        cohort._canonical_bindings()
    )
    assert result.intervention_authority.catalogue_sha256 == (result.intervention_catalogue.sha256)

    assert len(result.intervention_selections) == 4
    assert len(result.executions) == 4
    assert len(result.consequences) == 4
    assert len(result.evidence) == 4
    bindings = {binding.candidate.sha256: binding for binding in cohort._canonical_bindings()}
    for index, (
        selection,
        execution,
        consequence,
        evidence,
        candidate,
        source,
        phase,
    ) in enumerate(
        zip(
            result.intervention_selections,
            result.executions,
            result.consequences,
            result.evidence,
            expected_candidates,
            expected_sources,
            expected_phases,
            strict=True,
        )
    ):
        material = spec.episodes[index // 2]
        binding = bindings[candidate.sha256]
        expected_release = intervention._make_release(selection, binding)
        batch_root, stdin_root, case_count = cohort._executor_batch_commitments(material.batch)
        frame = continual.project_token_source_v1(source)
        expected_consequence = continual.make_token_edit_consequence_v1(
            source_sha256=frame.source_sha256,
            batch_sha256=batch_root,
            test_manifest_sha256=material.test_source_sha256,
            execution_sha256=execution.sha256,
            observed_outputs=execution.outputs,
            expected_outputs=material.expected_outputs,
            compile_cell=1,
            run_cell=1,
        )

        assert selection.authority_sha256 == result.intervention_authority.sha256
        assert selection.catalogue_sha256 == result.intervention_catalogue.sha256
        assert selection.candidate_sha256 == candidate.sha256
        assert execution.selected_source.selection_sha256 == selection.sha256
        assert execution.selected_source.release_sha256 == expected_release.sha256
        assert execution.selected_source.action_handle == candidate.candidate_id
        assert execution.selected_source.source == source
        assert execution.selected_source.source_sha256 == sha256(source.encode("ascii")).hexdigest()
        assert execution.preparation.expected_authority_sha256 == (
            result.intervention_authority.sha256
        )
        assert execution.preparation.batch_sha256 == batch_root
        assert execution.preparation.stdin_sha256 == stdin_root
        assert execution.preparation.case_count == case_count
        assert execution.run_receipt.batch_sha256 == batch_root
        assert execution.run_receipt.stdin_sha256 == stdin_root
        assert execution.compile_receipt.compile_invocations == 1
        assert execution.run_receipt.run_invocations == 1
        assert execution.outputs == (
            material.batch if phase == cohort.FAILED else material.expected_outputs
        )
        assert consequence == expected_consequence
        assert consequence.public_test_cell == int(phase == cohort.SUCCESSFUL)
        assert evidence.ordinal == index + 1
        assert evidence.phase == phase
        assert evidence.candidate_sha256 == candidate.sha256
        assert evidence.selection_sha256 == selection.sha256
        assert evidence.release_sha256 == expected_release.sha256
        assert evidence.execution_sha256 == execution.sha256
        assert evidence.consequence_sha256 == consequence.sha256
        assert evidence.batch_sha256 == batch_root
        assert evidence.stdin_sha256 == stdin_root
        assert evidence.test_source_sha256 == material.test_source_sha256

    assert tuple(executed_source_roots) == tuple(
        sha256(source.encode("ascii")).hexdigest() for source in expected_sources
    )
    assert len(set(executed_source_roots)) == 4
    assert sha256(cohort.FRESH_STRUCTURAL_SOURCE.encode("ascii")).hexdigest() not in (
        executed_source_roots
    )
    assert sha256(cohort.EXPECTED_FRESH_EDITED_SOURCE.encode("ascii")).hexdigest() not in (
        executed_source_roots
    )

    assert len(result.episodes) == 2
    assert tuple(item.failed_consequence.public_test_cell for item in result.episodes) == (
        0,
        0,
    )
    assert tuple(item.successful_consequence.public_test_cell for item in result.episodes) == (
        1,
        1,
    )
    assert tuple(item.failed_source.source for item in result.episodes) == (
        cohort.A_FAILED_SOURCE,
        cohort.B_FAILED_SOURCE,
    )
    assert tuple(item.successful_source.source for item in result.episodes) == (
        cohort.A_SUCCESSFUL_SOURCE,
        cohort.B_SUCCESSFUL_SOURCE,
    )
    assert len(set(result.checkpoint_lineage_sha256s)) == 3
    assert len(set(result.checkpoint_array_lineage_sha256s)) == 3

    assert result.one_source_proposal.status == (continual.STOP_INSUFFICIENT_INDEPENDENT_SOURCES)
    assert result.one_source_proposal.edited_source is None
    assert result.one_source_proposal.schema_support_counts[3] == 1
    assert result.one_source_proposal.schema_raw_scores[3] == 2
    assert result.one_source_proposal.race_simultaneous_spike_candidates == 2
    assert result.promoted_schema_indices == (3,)
    assert result.promoted_schema_sha256s == (continual.PARAMETERIZED_SCHEMAS[3].sha256,)
    assert result.final_support_counts[3] == 2
    assert tuple(
        index
        for index, support in enumerate(result.final_support_counts)
        if support >= continual.MIN_INDEPENDENT_EPISODES
    ) == (3,)
    assert result.fresh_source.source == cohort.FRESH_STRUCTURAL_SOURCE
    assert result.fresh_proposal.status == continual.READY
    assert result.fresh_proposal.selected_schema_index == 3
    assert result.fresh_proposal.applicable_schema_indices == (0, 1, 2, 3)
    assert result.fresh_proposal.anchor_token_index == 51
    assert result.fresh_proposal.target_token_index == 52
    assert result.fresh_proposal.donor_token_index == 54
    assert result.fresh_source.tokens[result.fresh_proposal.donor_token_index] == "boundary"
    training_tokens = {
        token
        for episode in result.episodes
        for frame in (episode.failed_source, episode.successful_source)
        for token in frame.tokens
    }
    assert "boundary" not in training_tokens
    assert result.fresh_proposal.edited_source == cohort.EXPECTED_FRESH_EDITED_SOURCE
    assert result.work.record(include_sha256=False) == {
        "intervention_issue_count": 4,
        "spent_selection_count": 4,
        "source_release_count": 4,
        "execution_attempt_count": 4,
        "successful_execution_count": 4,
        "compile_count": 4,
        "run_count": 4,
        "failed_consequence_count": 2,
        "successful_consequence_count": 2,
        "schema_advance_count": 2,
        "parent_immutability_check_count": 2,
        "proposed_source_execution_count": 0,
    }
