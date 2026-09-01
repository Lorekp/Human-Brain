from __future__ import annotations

from dataclasses import fields, replace
from hashlib import sha256

import numpy as np
import pytest
from experiments.r3_pilot import common_continual_token_edit_schema_v1 as token_edit

EPISODE_A_FAILED = """\
pub fn solve(readings: &[i64]) -> Vec<i64> {
    let ceiling = 7_i64;
    readings
        .iter()
        .copied()
        .map(|reading| if reading > reading { ceiling } else { reading })
        .collect()
}
"""
EPISODE_A_SUCCESSFUL = """\
pub fn solve(readings: &[i64]) -> Vec<i64> {
    let ceiling = 7_i64;
    readings
        .iter()
        .copied()
        .map(|reading| if reading > ceiling { ceiling } else { reading })
        .collect()
}
"""

EPISODE_B_FAILED = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let floor = -3_i64;
    samples
        .iter()
        .copied()
        .map(|temperature| match temperature < temperature {
            true => floor,
            false => temperature,
        })
        .collect()
}
"""
EPISODE_B_SUCCESSFUL = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let floor = -3_i64;
    samples
        .iter()
        .copied()
        .map(|temperature| match temperature < floor {
            true => floor,
            false => temperature,
        })
        .collect()
}
"""

DEPENDENT_FAILED = """\
pub fn solve(series: &[i64]) -> Vec<i64> {
    let boundary = 17_i64;
    series
        .iter()
        .copied()
        .map(|signal| if signal > signal { boundary } else { signal })
        .collect()
}
"""
DEPENDENT_SUCCESSFUL = """\
pub fn solve(series: &[i64]) -> Vec<i64> {
    let boundary = 17_i64;
    series
        .iter()
        .copied()
        .map(|signal| if signal > boundary { boundary } else { signal })
        .collect()
}
"""

FRESH_PROBE = """\
pub fn solve(measures: &[i64]) -> Vec<i64> {
    let boundary = 11_i64;
    let mut output = Vec::new();
    for measure in measures.iter().copied() {
        output.push(if measure >= measure { boundary } else { measure });
    }
    output
}
"""
EXPECTED_FRESH_EDIT = """\
pub fn solve(measures: &[i64]) -> Vec<i64> {
    let boundary = 11_i64;
    let mut output = Vec::new();
    for measure in measures.iter().copied() {
        output.push(if measure >= boundary { boundary } else { measure });
    }
    output
}
"""

EXPECTED_SCHEMA_INDEX = 3
EXPECTED_OUTPUTS = ((2, 3, 5),)
FAILED_OUTPUTS = ((2, 3, 4),)


def _root(label: str) -> str:
    return sha256(f"token-edit-test:{label}".encode("ascii")).hexdigest()


def _episode(
    *,
    label: str,
    failed_source: str,
    successful_source: str,
    repository_label: str | None = None,
    task_label: str | None = None,
    task_instance_label: str | None = None,
    task_source_label: str | None = None,
    test_source_label: str | None = None,
) -> token_edit.FailedSuccessfulTokenEditEpisodeV1:
    failed_frame = token_edit.project_token_source_v1(failed_source)
    successful_frame = token_edit.project_token_source_v1(successful_source)
    batch_sha256 = _root(f"batch:{label}")
    manifest_sha256 = _root(f"manifest:{label}")
    failed_consequence = token_edit.make_token_edit_consequence_v1(
        source_sha256=failed_frame.source_sha256,
        batch_sha256=batch_sha256,
        test_manifest_sha256=manifest_sha256,
        execution_sha256=_root(f"execution:{label}:failed"),
        observed_outputs=FAILED_OUTPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
    )
    successful_consequence = token_edit.make_token_edit_consequence_v1(
        source_sha256=successful_frame.source_sha256,
        batch_sha256=batch_sha256,
        test_manifest_sha256=manifest_sha256,
        execution_sha256=_root(f"execution:{label}:successful"),
        observed_outputs=EXPECTED_OUTPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
    )
    return token_edit.seal_failed_successful_token_edit_episode_v1(
        repository_sha256=_root(repository_label or f"repository:{label}"),
        task_sha256=_root(task_label or f"task:{label}"),
        task_instance_sha256=_root(task_instance_label or f"task-instance:{label}"),
        task_source_sha256=_root(task_source_label or f"task-source:{label}"),
        test_source_sha256=_root(test_source_label or f"test-source:{label}"),
        failed_source=failed_frame,
        failed_consequence=failed_consequence,
        successful_source=successful_frame,
        successful_consequence=successful_consequence,
    )


def _episode_a() -> token_edit.FailedSuccessfulTokenEditEpisodeV1:
    return _episode(
        label="alpha",
        failed_source=EPISODE_A_FAILED,
        successful_source=EPISODE_A_SUCCESSFUL,
    )


def _episode_b() -> token_edit.FailedSuccessfulTokenEditEpisodeV1:
    return _episode(
        label="beta",
        failed_source=EPISODE_B_FAILED,
        successful_source=EPISODE_B_SUCCESSFUL,
    )


def _one_source_checkpoint() -> token_edit.FrozenContinualTokenEditSchemaV1:
    return token_edit.advance_continual_token_edit_schema_v1(
        token_edit.start_continual_token_edit_schema_v1(),
        _episode_a(),
    )


def _ready_checkpoint() -> token_edit.FrozenContinualTokenEditSchemaV1:
    return token_edit.advance_continual_token_edit_schema_v1(
        _one_source_checkpoint(),
        _episode_b(),
    )


def _assert_stopped_without_edit(proposal: token_edit.TokenEditProposalV1) -> None:
    assert proposal.selected_schema_index is None
    assert proposal.selected_schema_cell_id is None
    assert proposal.selected_schema_sha256 is None
    assert proposal.anchor_token_index is None
    assert proposal.target_token_index is None
    assert proposal.donor_token_index is None
    assert proposal.anchor_occurrence_sha256 is None
    assert proposal.target_occurrence_sha256 is None
    assert proposal.donor_occurrence_sha256 is None
    assert proposal.edited_source_sha256 is None
    assert proposal.edited_source is None
    proposal.authenticate()


def test_empty_and_one_source_checkpoint_stop_without_mutating_parent() -> None:
    empty = token_edit.start_continual_token_edit_schema_v1()
    empty_root = empty.checkpoint_sha256
    empty_array_root = empty.checkpoint_array_sha256
    empty_summary = empty.training_summary
    probe = token_edit.project_token_source_v1(FRESH_PROBE)

    assert empty_summary.episode_count == 0
    assert empty_summary.independent_episode_count == 0
    assert empty_summary.promoted_schema_indices == ()
    assert empty.support_counts == tuple(0 for _ in token_edit.PARAMETERIZED_SCHEMAS)
    empty_stop = empty.propose(probe)
    assert empty_stop.status == token_edit.STOP_INSUFFICIENT_INDEPENDENT_SOURCES
    _assert_stopped_without_edit(empty_stop)

    one_source = token_edit.advance_continual_token_edit_schema_v1(empty, _episode_a())

    empty.authenticate()
    assert empty.checkpoint_sha256 == empty_root
    assert empty.checkpoint_array_sha256 == empty_array_root
    assert empty.provenance == ()
    assert one_source.provenance[0].matched_schema_indices == (EXPECTED_SCHEMA_INDEX,)
    assert one_source.provenance[0].independent_cell == 1
    assert one_source.support_counts[EXPECTED_SCHEMA_INDEX] == 1
    assert one_source.training_summary.promoted_schema_indices == ()
    one_source_stop = one_source.propose(probe)
    assert one_source_stop.status == token_edit.STOP_INSUFFICIENT_INDEPENDENT_SOURCES
    assert one_source_stop.schema_raw_scores[EXPECTED_SCHEMA_INDEX] == 2
    assert one_source_stop.race_simultaneous_spike_candidates == 2
    _assert_stopped_without_edit(one_source_stop)

    tonic_off = one_source.propose(probe, tonic_gate=False)
    assert tonic_off.status == token_edit.CONTROL_TONIC_OFF_READY
    assert tonic_off.selected_schema_index == EXPECTED_SCHEMA_INDEX
    assert tonic_off.race_simultaneous_spike_candidates == 1
    tonic_off.authenticate()


def test_two_independent_episodes_propose_fresh_structural_copy() -> None:
    one_source = _one_source_checkpoint()
    parent_root = one_source.checkpoint_sha256
    parent_array_root = one_source.checkpoint_array_sha256
    parent_support = one_source.support_counts
    parent_provenance = one_source.provenance

    ready = token_edit.advance_continual_token_edit_schema_v1(one_source, _episode_b())
    ready.authenticate()
    one_source.authenticate()
    assert one_source.checkpoint_sha256 == parent_root
    assert one_source.checkpoint_array_sha256 == parent_array_root
    assert one_source.support_counts == parent_support
    assert one_source.provenance == parent_provenance
    assert ready.support_counts[EXPECTED_SCHEMA_INDEX] == 2
    assert ready.training_summary.independent_episode_count == 2
    assert ready.training_summary.promoted_schema_indices == (EXPECTED_SCHEMA_INDEX,)

    probe = token_edit.project_token_source_v1(FRESH_PROBE)
    checkpoint_root = ready.checkpoint_sha256
    checkpoint_array_root = ready.checkpoint_array_sha256
    proposal = ready.propose(probe)

    assert proposal.status == token_edit.READY
    assert proposal.selected_schema_index == EXPECTED_SCHEMA_INDEX
    assert proposal.input_source_sha256 == probe.source_sha256
    assert proposal.edited_source == EXPECTED_FRESH_EDIT
    assert proposal.edited_source not in {
        EPISODE_A_FAILED,
        EPISODE_A_SUCCESSFUL,
        EPISODE_B_FAILED,
        EPISODE_B_SUCCESSFUL,
    }
    assert proposal.target_token_index is not None
    assert proposal.donor_token_index is not None
    assert proposal.anchor_token_index is not None
    assert probe.tokens[proposal.anchor_token_index] == ">="
    assert probe.tokens[proposal.target_token_index] == "measure"
    assert probe.tokens[proposal.donor_token_index] == "boundary"
    assert (
        len(
            {
                proposal.anchor_occurrence_sha256,
                proposal.target_occurrence_sha256,
                proposal.donor_occurrence_sha256,
            }
        )
        == 3
    )
    training_source_roots = {
        source_root
        for row in ready.provenance
        for source_root in (row.failed_source_sha256, row.successful_source_sha256)
    }
    assert proposal.input_source_sha256 not in training_source_roots
    assert proposal.edited_source_sha256 not in training_source_roots
    assert "ceiling" not in proposal.edited_source
    assert "floor" not in proposal.edited_source
    proposal.authenticate()
    ready.authenticate()
    assert ready.checkpoint_sha256 == checkpoint_root
    assert ready.checkpoint_array_sha256 == checkpoint_array_root


def test_replay_is_rejected_and_dependent_source_does_not_reach_quorum() -> None:
    first_episode = _episode_a()
    one_source = token_edit.advance_continual_token_edit_schema_v1(
        token_edit.start_continual_token_edit_schema_v1(),
        first_episode,
    )
    with pytest.raises(
        token_edit.ContinualTokenEditSchemaError,
        match="episode or receipt replay",
    ):
        token_edit.advance_continual_token_edit_schema_v1(one_source, first_episode)

    dependent = _episode(
        label="delta",
        failed_source=DEPENDENT_FAILED,
        successful_source=DEPENDENT_SUCCESSFUL,
        task_source_label="task-source:alpha",
    )
    dependent_checkpoint = token_edit.advance_continual_token_edit_schema_v1(
        one_source,
        dependent,
    )

    assert dependent_checkpoint.training_summary.episode_count == 2
    assert dependent_checkpoint.training_summary.independent_episode_count == 1
    assert dependent_checkpoint.training_summary.dependent_episode_count == 1
    assert dependent_checkpoint.provenance[-1].independent_cell == 0
    assert dependent_checkpoint.support_counts == one_source.support_counts
    assert dependent_checkpoint.training_summary.promoted_schema_indices == ()
    proposal = dependent_checkpoint.propose(token_edit.project_token_source_v1(FRESH_PROBE))
    assert proposal.status == token_edit.STOP_INSUFFICIENT_INDEPENDENT_SOURCES
    _assert_stopped_without_edit(proposal)


def test_plasticity_and_consequence_gates_return_the_exact_parent() -> None:
    one_source = _one_source_checkpoint()
    second_episode = _episode_b()
    parent_root = one_source.checkpoint_sha256
    parent_array_root = one_source.checkpoint_array_sha256
    parent_provenance = one_source.provenance

    assert (
        token_edit.advance_continual_token_edit_schema_v1(
            one_source,
            second_episode,
            plasticity_gate=False,
        )
        is one_source
    )
    assert (
        token_edit.advance_continual_token_edit_schema_v1(
            one_source,
            second_episode,
            consequence_gate=False,
        )
        is one_source
    )
    one_source.authenticate()
    assert one_source.checkpoint_sha256 == parent_root
    assert one_source.checkpoint_array_sha256 == parent_array_root
    assert one_source.provenance == parent_provenance
    assert one_source.training_summary.promoted_schema_indices == ()


def test_cell_and_schema_gates_hide_a_ready_edit() -> None:
    ready = _ready_checkpoint()
    probe = token_edit.project_token_source_v1(FRESH_PROBE)
    checkpoint_root = ready.checkpoint_sha256
    checkpoint_array_root = ready.checkpoint_array_sha256

    cell_off = ready.propose(probe, cell_gate=False)
    assert cell_off.status == token_edit.STOP_CELL_OFF
    assert cell_off.race_physical_cell_steps == 0
    _assert_stopped_without_edit(cell_off)

    schema_off = ready.propose(probe, schema_gate=False)
    assert schema_off.status == token_edit.STOP_SCHEMA_OFF
    assert schema_off.race_physical_cell_steps == 0
    _assert_stopped_without_edit(schema_off)

    ready.authenticate()
    assert ready.checkpoint_sha256 == checkpoint_root
    assert ready.checkpoint_array_sha256 == checkpoint_array_root


def test_proposal_and_checkpoint_authentication_rederive_their_causal_context() -> None:
    ready = _ready_checkpoint()
    proposal = ready.propose(token_edit.project_token_source_v1(FRESH_PROBE))
    forged_partial = replace(
        proposal,
        edited_source="forged",
        edited_source_sha256=sha256(b"forged").hexdigest(),
        sha256="",
    )
    forged_proposal = replace(
        forged_partial,
        sha256=token_edit._digest(
            token_edit._PROPOSAL_NAMESPACE,
            forged_partial.record(include_sha256=False),
        ),
    )
    with pytest.raises(
        token_edit.ContinualTokenEditSchemaError,
        match="does not rederive from its exact context",
    ):
        forged_proposal.authenticate()

    first, second = ready.provenance
    forged_row_partial = replace(
        second,
        repository_sha256=first.repository_sha256,
        task_source_sha256=first.task_source_sha256,
        sha256="",
    )
    forged_row = replace(
        forged_row_partial,
        sha256=token_edit._digest(
            token_edit._PROVENANCE_NAMESPACE,
            forged_row_partial.record(include_sha256=False),
        ),
    )
    forged_provenance = (first, forged_row)
    summary_record = {
        field.name: getattr(ready.training_summary, field.name)
        for field in fields(ready.training_summary)
        if field.name != "checkpoint_sha256"
    }
    forged_summary = replace(
        ready.training_summary,
        checkpoint_sha256=token_edit._checkpoint_sha256(
            ready._support,
            forged_provenance,
            summary_record,
        ),
    )
    forged_checkpoint = token_edit.FrozenContinualTokenEditSchemaV1(
        support=ready._support,
        provenance=forged_provenance,
        summary=forged_summary,
    )
    with pytest.raises(
        token_edit.ContinualTokenEditSchemaError,
        match="historical custody replay",
    ):
        forged_checkpoint.authenticate()

    def rewrite_match(row, *, parent_root: str):
        partial = replace(
            row,
            parent_checkpoint_sha256=parent_root,
            matched_schema_indices=(0,),
            sha256="",
        )
        return replace(
            partial,
            sha256=token_edit._digest(
                token_edit._PROVENANCE_NAMESPACE,
                partial.record(include_sha256=False),
            ),
        )

    empty = token_edit.start_continual_token_edit_schema_v1()
    forged_first = rewrite_match(first, parent_root=empty.checkpoint_sha256)
    support_after_first = np.zeros(
        (len(token_edit.PARAMETERIZED_SCHEMAS),),
        dtype=token_edit.TICK_DTYPE,
    )
    support_after_first[0] = 1
    after_first_summary = token_edit._training_summary_record(
        support=support_after_first,
        provenance=(forged_first,),
        parent_checkpoint_sha256=empty.checkpoint_sha256,
        mode="REPLAY_FREE_CONTINUAL_TOKEN_EDIT_ADVANCE_V1",
    )
    forged_after_first_root = token_edit._checkpoint_sha256(
        support_after_first,
        (forged_first,),
        after_first_summary,
    )
    forged_second = rewrite_match(second, parent_root=forged_after_first_root)
    forged_match_provenance = (forged_first, forged_second)
    forged_match_support = np.zeros(
        (len(token_edit.PARAMETERIZED_SCHEMAS),),
        dtype=token_edit.TICK_DTYPE,
    )
    forged_match_support[0] = 2
    forged_match_summary_record = token_edit._training_summary_record(
        support=forged_match_support,
        provenance=forged_match_provenance,
        parent_checkpoint_sha256=forged_after_first_root,
        mode="REPLAY_FREE_CONTINUAL_TOKEN_EDIT_ADVANCE_V1",
    )
    forged_match_root = token_edit._checkpoint_sha256(
        forged_match_support,
        forged_match_provenance,
        forged_match_summary_record,
    )
    forged_match_checkpoint = token_edit.FrozenContinualTokenEditSchemaV1(
        support=forged_match_support,
        provenance=forged_match_provenance,
        summary=token_edit.TokenEditSchemaTrainingSummaryV1(
            **forged_match_summary_record,
            checkpoint_sha256=forged_match_root,
        ),
    )
    forged_match_checkpoint.authenticate()
    with pytest.raises(
        token_edit.ContinualTokenEditSchemaError,
        match="checkpoint authentication failed",
    ):
        forged_match_checkpoint.authenticate(expected_checkpoint_sha256=ready.checkpoint_sha256)
