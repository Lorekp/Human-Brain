from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest
from experiments.r3_pilot import common_continual_token_edit_consequence_policy_v1 as policy
from experiments.r3_pilot import common_continual_token_edit_schema_v1 as token_edits

SEED_FAILED = "alpha > alpha omega"
SEED_SUCCESSFUL = "alpha > omega omega"
ADAPTATION_SOURCE = "delta >= delta limit"
ADAPTATION_EDIT = "delta >= limit limit"
EVALUATION_SOURCE = "signal < signal threshold"
EVALUATION_EDIT = "signal < threshold threshold"

EXPECTED_OUTPUTS = ((1, 1, 1),)
FAILED_OUTPUTS = ((1, 0, 1),)
EXPECTED_SCHEMA_INDEX = 3
COLLIDING_SCHEMA_INDEX = 1


def _root(label: str) -> str:
    return sha256(f"token-edit-consequence-policy-test:{label}".encode("ascii")).hexdigest()


def _consequence(
    *,
    source_sha256: str,
    label: str,
    passed: bool,
    context_label: str | None = None,
    batch_sha256: str | None = None,
    test_manifest_sha256: str | None = None,
    execution_sha256: str | None = None,
) -> token_edits.TokenEditConsequenceV1:
    context = label if context_label is None else context_label
    return token_edits.make_token_edit_consequence_v1(
        source_sha256=source_sha256,
        batch_sha256=_root(f"batch:{context}") if batch_sha256 is None else batch_sha256,
        test_manifest_sha256=(
            _root(f"manifest:{context}") if test_manifest_sha256 is None else test_manifest_sha256
        ),
        execution_sha256=(
            _root(f"execution:{label}") if execution_sha256 is None else execution_sha256
        ),
        observed_outputs=EXPECTED_OUTPUTS if passed else FAILED_OUTPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
    )


def _seed_episode() -> token_edits.FailedSuccessfulTokenEditEpisodeV1:
    failed = token_edits.project_token_source_v1(SEED_FAILED)
    successful = token_edits.project_token_source_v1(SEED_SUCCESSFUL)
    return token_edits.seal_failed_successful_token_edit_episode_v1(
        repository_sha256=_root("seed:repository"),
        task_sha256=_root("seed:task"),
        task_instance_sha256=_root("seed:task-instance"),
        task_source_sha256=_root("seed:task-source"),
        test_source_sha256=_root("seed:test-source"),
        failed_source=failed,
        failed_consequence=_consequence(
            source_sha256=failed.source_sha256,
            label="seed:failed",
            passed=False,
            context_label="seed",
        ),
        successful_source=successful,
        successful_consequence=_consequence(
            source_sha256=successful.source_sha256,
            label="seed:successful",
            passed=True,
            context_label="seed",
        ),
    )


def _base_checkpoint() -> token_edits.FrozenContinualTokenEditSchemaV1:
    return token_edits.advance_continual_token_edit_schema_v1(
        token_edits.start_continual_token_edit_schema_v1(),
        _seed_episode(),
    )


def _started_policy() -> policy.FrozenContinualTokenEditConsequencePolicyV1:
    return policy.start_continual_token_edit_consequence_policy_v1(_base_checkpoint())


def _exploration_proposal(
    checkpoint: policy.FrozenContinualTokenEditConsequencePolicyV1,
) -> policy.TokenEditConsequencePolicyProposalV1:
    return checkpoint.propose(
        token_edits.project_token_source_v1(ADAPTATION_SOURCE),
        exploration_gate=True,
    )


def _sealed_outcome(
    checkpoint: policy.FrozenContinualTokenEditConsequencePolicyV1,
    *,
    passed: bool,
    label: str,
) -> policy.TokenEditConsequencePolicyOutcomeV1:
    proposal = _exploration_proposal(checkpoint)
    assert proposal.edited_source_sha256 is not None
    return policy.seal_token_edit_consequence_policy_outcome_v1(
        proposal=proposal,
        consequence=_consequence(
            source_sha256=proposal.edited_source_sha256,
            label=label,
            passed=passed,
        ),
    )


def _scores_by_action(
    proposal: policy.TokenEditConsequencePolicyProposalV1,
) -> dict[str, int]:
    return dict(
        zip(
            proposal.activation.active_action_sha256s,
            proposal.activation.base_scores,
            strict=True,
        )
    )


def _effective_by_action(
    proposal: policy.TokenEditConsequencePolicyProposalV1,
) -> dict[str, int]:
    return dict(
        zip(
            proposal.activation.active_action_sha256s,
            proposal.decision.effective_scores,
            strict=True,
        )
    )


def test_typed_attested_pass_promotes_schema_for_a_fresh_normal_query() -> None:
    base = _base_checkpoint()
    parent = policy.start_continual_token_edit_consequence_policy_v1(base)
    adaptation = token_edits.project_token_source_v1(ADAPTATION_SOURCE)
    evaluation = token_edits.project_token_source_v1(EVALUATION_SOURCE)
    schema_root = token_edits.PARAMETERIZED_SCHEMAS[EXPECTED_SCHEMA_INDEX].sha256
    collision_root = token_edits.PARAMETERIZED_SCHEMAS[COLLIDING_SCHEMA_INDEX].sha256

    assert len(parent.catalogue) == len(token_edits.PARAMETERIZED_SCHEMAS) == 24
    assert tuple(item.action_sha256 for item in parent.catalogue) == (
        parent.outcome_memory.action_sha256s
    )
    assert {item.action_sha256: item.schema_index for item in parent.catalogue} == {
        schema.sha256: schema.schema_index for schema in token_edits.PARAMETERIZED_SCHEMAS
    }
    pre_update = parent.propose(adaptation)
    assert pre_update.status == policy.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
    assert pre_update.candidate_view.applicable_schema_indices == (1, 3)
    assert _scores_by_action(pre_update) == {collision_root: 0, schema_root: 2}

    exploration = parent.propose(adaptation, exploration_gate=True)
    assert exploration.status == policy.EXPLORATION_READY
    assert exploration.selected_schema_index == EXPECTED_SCHEMA_INDEX
    assert exploration.selected_schema_sha256 == schema_root
    assert exploration.edited_source == ADAPTATION_EDIT
    assert exploration.decision.selected_action_sha256 == schema_root
    assert exploration.decision.selected_action_index is not None
    assert (
        parent.outcome_memory.action_sha256s[exploration.decision.selected_action_index]
        == schema_root
    )

    assert exploration.edited_source_sha256 is not None
    typed_outcome = policy.seal_token_edit_consequence_policy_outcome_v1(
        proposal=exploration,
        consequence=_consequence(
            source_sha256=exploration.edited_source_sha256,
            label="adaptation:pass",
            passed=True,
        ),
    )
    assert typed_outcome.passed_cell == 1
    parent_root = parent.checkpoint_sha256
    base_root = base.checkpoint_sha256
    base_array_root = base.checkpoint_array_sha256
    base_provenance = base.provenance
    successor = policy.advance_continual_token_edit_consequence_policy_v1(
        parent,
        typed_outcome,
    )

    assert successor is not parent
    assert successor.base_checkpoint is base
    assert parent.checkpoint_sha256 == parent_root
    assert base.checkpoint_sha256 == base_root
    assert base.checkpoint_array_sha256 == base_array_root
    assert base.provenance == base_provenance
    action_row = successor.outcome_memory.action_sha256s.index(schema_root)
    assert successor.outcome_memory.excitatory_ticks[action_row] == 1
    assert successor.outcome_memory.inhibitory_ticks[action_row] == 0

    assert parent.propose(evaluation).status == policy.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
    fresh = successor.propose(evaluation)
    assert fresh.status == policy.READY
    assert fresh.candidate_view.applicable_schema_indices == (1, 3)
    assert fresh.selected_schema_index == EXPECTED_SCHEMA_INDEX
    assert fresh.edited_source == EVALUATION_EDIT
    assert _effective_by_action(fresh)[schema_root] == 4
    assert (
        successor.propose(evaluation, outcome_gate=False).status
        == policy.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
    )


def test_typed_attested_failure_inhibits_the_selected_schema() -> None:
    parent = _started_policy()
    typed_outcome = _sealed_outcome(
        parent,
        passed=False,
        label="adaptation:fail",
    )
    assert typed_outcome.passed_cell == 0
    successor = policy.advance_continual_token_edit_consequence_policy_v1(
        parent,
        typed_outcome,
    )
    schema_root = token_edits.PARAMETERIZED_SCHEMAS[EXPECTED_SCHEMA_INDEX].sha256
    action_row = successor.outcome_memory.action_sha256s.index(schema_root)
    assert successor.outcome_memory.excitatory_ticks[action_row] == 0
    assert successor.outcome_memory.inhibitory_ticks[action_row] == 1

    fresh = successor.propose(
        token_edits.project_token_source_v1(EVALUATION_SOURCE),
        exploration_gate=True,
    )
    assert fresh.status == policy.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
    assert _effective_by_action(fresh)[schema_root] == 0
    assert fresh.edited_source is None


def test_gates_stop_or_return_the_exact_immutable_parent() -> None:
    parent = _started_policy()
    source = token_edits.project_token_source_v1(ADAPTATION_SOURCE)
    cell_off = parent.propose(source, cell_gate=False, exploration_gate=True)
    schema_off = parent.propose(source, schema_gate=False, exploration_gate=True)
    assert cell_off.status == policy.STOP_CELL_OFF
    assert cell_off.candidate_view.status == token_edits.STOP_CELL_OFF
    assert schema_off.status == policy.STOP_SCHEMA_OFF
    assert schema_off.decision.status == policy.STOP_NO_CANDIDATE
    assert schema_off.candidate_view.status == token_edits.STOP_SCHEMA_OFF
    assert cell_off.edited_source is schema_off.edited_source is None
    with pytest.raises(policy.ContinualTokenEditConsequencePolicyError):
        policy.seal_token_edit_consequence_policy_outcome_v1(
            proposal=schema_off,
            consequence=_consequence(
                source_sha256=source.source_sha256,
                label="schema-off",
                passed=False,
            ),
        )

    typed_outcome = _sealed_outcome(
        parent,
        passed=True,
        label="gates:pass",
    )
    parent_root = parent.checkpoint_sha256
    base_root = parent.base_checkpoint.checkpoint_sha256
    base_array_root = parent.base_checkpoint.checkpoint_array_sha256
    assert (
        policy.advance_continual_token_edit_consequence_policy_v1(
            parent,
            typed_outcome,
            plasticity_gate=False,
        )
        is parent
    )
    assert (
        policy.advance_continual_token_edit_consequence_policy_v1(
            parent,
            typed_outcome,
            consequence_gate=False,
        )
        is parent
    )
    assert parent.checkpoint_sha256 == parent_root
    assert parent.base_checkpoint.checkpoint_sha256 == base_root
    assert parent.base_checkpoint.checkpoint_array_sha256 == base_array_root


def test_consequence_lineage_splice_and_stale_parent_are_rejected() -> None:
    parent = _started_policy()
    proposal = _exploration_proposal(parent)
    assert proposal.edited_source_sha256 is not None
    wrong_source_consequence = _consequence(
        source_sha256=proposal.candidate_view.source_sha256,
        label="wrong-source",
        passed=True,
    )
    with pytest.raises(policy.ContinualTokenEditConsequencePolicyError):
        policy.seal_token_edit_consequence_policy_outcome_v1(
            proposal=proposal,
            consequence=wrong_source_consequence,
        )

    forged = replace(proposal, selected_schema_index=COLLIDING_SCHEMA_INDEX)
    with pytest.raises(policy.ContinualTokenEditConsequencePolicyError):
        forged.authenticate()

    typed_outcome = policy.seal_token_edit_consequence_policy_outcome_v1(
        proposal=proposal,
        consequence=_consequence(
            source_sha256=proposal.edited_source_sha256,
            label="lineage:pass",
            passed=True,
        ),
    )
    successor = policy.advance_continual_token_edit_consequence_policy_v1(
        parent,
        typed_outcome,
    )
    with pytest.raises(policy.ContinualTokenEditConsequencePolicyError):
        policy.advance_continual_token_edit_consequence_policy_v1(
            successor,
            typed_outcome,
        )

    fresh = successor.propose(token_edits.project_token_source_v1(EVALUATION_SOURCE))
    assert fresh.edited_source_sha256 is not None
    fresh_outcome = policy.seal_token_edit_consequence_policy_outcome_v1(
        proposal=fresh,
        consequence=_consequence(
            source_sha256=fresh.edited_source_sha256,
            label="lineage:fresh",
            passed=True,
        ),
    )
    spliced = replace(typed_outcome, proposal=fresh_outcome.proposal)
    with pytest.raises(policy.ContinualTokenEditConsequencePolicyError):
        spliced.authenticate()


def test_typed_consequence_custody_rejects_aliases_and_retains_auxiliary_roots() -> None:
    parent = _started_policy()
    proposal = _exploration_proposal(parent)
    assert proposal.edited_source_sha256 is not None

    checkpoint_alias = _consequence(
        source_sha256=proposal.edited_source_sha256,
        label="custody:checkpoint-alias",
        passed=True,
        execution_sha256=parent.checkpoint_sha256,
    )
    with pytest.raises(policy.ContinualTokenEditConsequencePolicyError, match="custody"):
        policy.seal_token_edit_consequence_policy_outcome_v1(
            proposal=proposal,
            consequence=checkpoint_alias,
        )

    repeated_role = _root("custody:repeated-role")
    within_consequence_alias = _consequence(
        source_sha256=proposal.edited_source_sha256,
        label="custody:within-consequence-alias",
        passed=True,
        batch_sha256=repeated_role,
        test_manifest_sha256=repeated_role,
    )
    with pytest.raises(policy.ContinualTokenEditConsequencePolicyError, match="event roles"):
        policy.seal_token_edit_consequence_policy_outcome_v1(
            proposal=proposal,
            consequence=within_consequence_alias,
        )

    authentic_consequence = _consequence(
        source_sha256=proposal.edited_source_sha256,
        label="custody:authentic",
        passed=True,
    )
    typed = policy.seal_token_edit_consequence_policy_outcome_v1(
        proposal=proposal,
        consequence=authentic_consequence,
    )
    expected_auxiliary = tuple(
        sorted(
            (
                authentic_consequence.batch_sha256,
                authentic_consequence.test_manifest_sha256,
                authentic_consequence.feedback_sha256,
            )
        )
    )
    assert typed.selected_outcome.auxiliary_event_sha256s == expected_auxiliary
    successor = policy.advance_continual_token_edit_consequence_policy_v1(parent, typed)
    assert successor.outcome_memory.provenance[0].auxiliary_event_sha256s == expected_auxiliary

    fresh = successor.propose(token_edits.project_token_source_v1(EVALUATION_SOURCE))
    assert fresh.edited_source_sha256 is not None
    historical_alias = _consequence(
        source_sha256=fresh.edited_source_sha256,
        label="custody:historical-alias",
        passed=True,
        batch_sha256=authentic_consequence.feedback_sha256,
    )
    with pytest.raises(policy.ContinualTokenEditConsequencePolicyError, match="historical"):
        policy.seal_token_edit_consequence_policy_outcome_v1(
            proposal=fresh,
            consequence=historical_alias,
        )


def test_generic_stop_materializes_nothing_and_exploration_only_its_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _started_policy()
    source = token_edits.project_token_source_v1(ADAPTATION_SOURCE)
    calls: list[int] = []
    original = policy._apply_schema_v1

    def counted_apply(
        schema: token_edits.ParameterizedTokenEditSchemaV1,
        frame: token_edits.TokenSourceFrameV1,
    ) -> object:
        calls.append(schema.schema_index)
        return original(schema, frame)

    monkeypatch.setattr(policy, "_apply_schema_v1", counted_apply)
    stopped = parent.propose(source)
    assert stopped.status == policy.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
    assert stopped.edited_source is None
    assert not hasattr(stopped, "base_proposal")
    assert stopped.candidate_view.status == policy.CANDIDATES_READY
    assert stopped.candidate_view.applicable_schema_indices == (1, 3)
    assert stopped.candidate_view.record()["v1_race_run"] is False
    assert stopped.candidate_view.record()["candidate_bindings_exposed"] is False
    assert stopped.candidate_view.record()["edited_source_exposed"] is False
    assert calls == []

    selected = parent.propose(source, exploration_gate=True)
    assert selected.status == policy.EXPLORATION_READY
    assert selected.selected_schema_index == EXPECTED_SCHEMA_INDEX
    assert selected.edited_source == ADAPTATION_EDIT
    assert calls == [EXPECTED_SCHEMA_INDEX]
