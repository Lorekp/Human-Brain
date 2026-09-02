from __future__ import annotations

from hashlib import sha256
from inspect import signature

from humanbrain.knowledge_conditioned_occurrence_substitution import (
    FrozenKnowledgeConditionedOccurrenceSubstitution,
    KnowledgeSubstitutionAblation,
    KnowledgeSubstitutionExecutionAuthority,
    KnowledgeSubstitutionSuccess,
    grow,
)
from humanbrain.multiscale_opaque_event_tape import OpaqueWorkspaceFile
from humanbrain.neuronal_relational_reader import RelationalReadingSuccess
from humanbrain.opaque_document_event_tape import OpaqueDocument
from humanbrain.persistent_neuronal_document_learner import advance, consolidate, start
from humanbrain.unified_opaque_observation_tape import UnifiedOpaqueObservationTape


def _document(run: str, ordinal: int, content: bytes) -> OpaqueDocument:
    return OpaqueDocument(
        document_id=f"document/{run}/{ordinal}".encode(),
        provenance_id=f"provenance/{run}/{ordinal}".encode(),
        content=content,
    )


def _lesson(run: str, subject: bytes, answer: bytes) -> RelationalReadingSuccess:
    return RelationalReadingSuccess(
        query=b"Route " + subject + b" through lumet.",
        documents=(
            _document(run, 0, subject + b" pavo " + answer + b"."),
            _document(run, 1, b"qira."),
        ),
        observed_answer=answer,
        provenance_id=f"lesson/{run}".encode(),
    )


def _observation(
    run: str,
    subject: bytes,
    files: tuple[OpaqueWorkspaceFile, ...],
) -> UnifiedOpaqueObservationTape:
    return UnifiedOpaqueObservationTape.observe(
        b"Route " + subject + b" through lumet.",
        (_document(f"carrier/{run}", 0, b"qira."),),
        files,
        (f"event/{run}".encode(),),
    )


def test_frozen_fact_to_occurrence_substitution_transfer_and_stops() -> None:
    learner = advance(start(), _lesson("schema-a", b"axor", b"iona")).successor
    learner = advance(learner, _lesson("schema-b", b"bevik", b"julo")).successor
    retained = consolidate(
        learner,
        (
            _document(
                "retained",
                0,
                (
                    b"kelo pavo vesa. rilo pavo luma. "
                    b"moro pavo zani. qelo pavo raku."
                ),
            ),
        ),
    ).successor
    assert retained.training_summary.consolidated_fact_count == 4

    before_a = (
        OpaqueWorkspaceFile("alpha.unit", b"kelo state dull\n"),
        OpaqueWorkspaceFile("decoy-a.unit", b"other inert spare\n"),
    )
    after_a = (
        OpaqueWorkspaceFile("alpha.unit", b"kelo state vesa\n"),
        before_a[1],
    )
    before_b = (
        OpaqueWorkspaceFile("beta.unit", b"rilo phase pale\n"),
        OpaqueWorkspaceFile("decoy-b.unit", b"foreign quiet token\n"),
    )
    after_b = (
        OpaqueWorkspaceFile("beta.unit", b"rilo phase luma\n"),
        before_b[1],
    )
    train_a = _observation("train-a", b"kelo", before_a)
    train_b = _observation("train-b", b"rilo", before_b)
    expected = {train_a.stable_hash: after_a, train_b.stable_hash: after_b}

    def evaluator(
        *,
        observation: UnifiedOpaqueObservationTape,
        after_workspace: tuple[OpaqueWorkspaceFile, ...],
    ) -> bool:
        return expected.get(observation.stable_hash) == after_workspace

    authority = KnowledgeSubstitutionExecutionAuthority(
        evaluator,
        sha256(b"frozen fact-to-occurrence fixture evaluator/v1").hexdigest(),
    )

    def build_bridge(producer) -> FrozenKnowledgeConditionedOccurrenceSubstitution:
        rows = []
        for ordinal, (observation, after) in enumerate(
            ((train_a, after_a), (train_b, after_b))
        ):
            receipt = authority.execute_and_issue(
                producer_checkpoint_sha256=producer.checkpoint_sha256,
                observation=observation,
                after_workspace=after,
            )
            rows.append(
                KnowledgeSubstitutionSuccess.seal(
                    observation=observation,
                    after_workspace=after,
                    execution_receipt=receipt,
                    provenance_id=f"bridge-training/{ordinal}".encode(),
                )
            )
        return grow(producer, rows)

    bridge = build_bridge(retained)
    assert bridge.training_summary.demonstration_count == 2
    assert bridge.training_summary.retained_full_transition_count == 0
    assert bridge.training_summary.binder_wildcard_hops >= 1
    assert tuple(signature(bridge.respond).parameters) == ("observation", "ablation")

    fresh_files = (
        OpaqueWorkspaceFile("fresh-a.unit", b"moro mode gray\n"),
        OpaqueWorkspaceFile("fresh-b.unit", b"qelo shift white\n"),
        OpaqueWorkspaceFile("fresh-decoy.unit", b"unseen inert spare\n"),
    )
    moro_observation = _observation("fresh-moro", b"moro", fresh_files)
    qelo_observation = _observation("fresh-qelo", b"qelo", fresh_files)
    root = bridge.checkpoint_sha256
    moro = bridge.respond(moro_observation)
    qelo = bridge.respond(qelo_observation)
    assert moro.decision == qelo.decision == "EDIT"
    assert moro.workspace == (
        OpaqueWorkspaceFile("fresh-a.unit", b"moro mode zani\n"),
        fresh_files[1],
        fresh_files[2],
    )
    assert qelo.workspace == (
        fresh_files[0],
        OpaqueWorkspaceFile("fresh-b.unit", b"qelo shift raku\n"),
        fresh_files[2],
    )
    assert all(b"zani" not in item.content for item in fresh_files)
    assert all(b"raku" not in item.content for item in fresh_files)
    assert b"zani" not in moro_observation.reconstruct_request()
    assert b"raku" not in qelo_observation.reconstruct_request()
    assert all(b"zani" not in item.content for item in moro_observation.reconstruct_documents())
    assert all(b"raku" not in item.content for item in qelo_observation.reconstruct_documents())
    assert moro.broadcast_sha256 != qelo.broadcast_sha256
    assert moro.emission_events != qelo.emission_events
    assert bridge.checkpoint_sha256 == root
    for decision in (moro, qelo):
        assert decision.physical_cell_steps > 0
        final = decision.structural_races[-1]
        assert final.population == "action/edit_or_stop"
        assert final.candidate_cell_ids == (0,)
        assert final.tonic_stop_cell_id == 1
        assert not final.stopped

    duplicate_anchor = _observation(
        "duplicate",
        b"moro",
        (
            OpaqueWorkspaceFile("duplicate-a.unit", b"moro mode gray\n"),
            OpaqueWorkspaceFile("duplicate-b.unit", b"moro shift white\n"),
        ),
    )
    duplicate = bridge.respond(duplicate_anchor)
    missing = bridge.respond(_observation("missing", b"nexa", fresh_files))
    assert duplicate.decision == missing.decision == "STOP"
    assert duplicate.stop_stage == "anchor_cleanup"
    assert missing.stop_stage == "broadcast"

    conflict_source = _document("conflict", 0, b"moro pavo ulon.")
    conflicted = consolidate(retained, (conflict_source,)).successor
    conflict = build_bridge(conflicted).respond(moro_observation)
    assert conflict.decision == "STOP"
    assert conflict.stop_stage == "broadcast"
    assert conflict.stop_reason == "broadcast:retrieval:collision"
    for stopped in (duplicate, missing, conflict):
        final = stopped.structural_races[-1]
        assert final.population == "action/edit_or_stop"
        assert final.candidate_cell_ids == (0,)
        assert final.tonic_stop_cell_id == 1
        assert final.stopped

    for ablation in (
        KnowledgeSubstitutionAblation.BRIDGE_OFF,
        KnowledgeSubstitutionAblation.UNBIND_OFF,
        KnowledgeSubstitutionAblation.ANCHOR_CLEANUP_OFF,
        KnowledgeSubstitutionAblation.PAYLOAD_CLEANUP_OFF,
        KnowledgeSubstitutionAblation.PROPAGATION_OFF,
        KnowledgeSubstitutionAblation.BOUNDARY_OFF,
        KnowledgeSubstitutionAblation.PROGRAM_OFF,
        KnowledgeSubstitutionAblation.FINAL_EDIT_OFF,
    ):
        stopped = bridge.respond(moro_observation, ablation=ablation)
        assert stopped.decision == "STOP"
        assert stopped.workspace is None
        final = stopped.structural_races[-1]
        assert final.population == "action/edit_or_stop"
        assert final.candidate_cell_ids == (0,)
        assert final.tonic_stop_cell_id == 1
        assert final.stopped
    assert bridge.checkpoint_sha256 == root
