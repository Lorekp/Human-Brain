from __future__ import annotations

from hashlib import sha256
from inspect import signature

import pytest
from experiments.r4_integrated.neutral_document_python_session import run
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


def _neutral_action_seed():
    from humanbrain.neutral_document_action import start as start_neutral

    learner = start_neutral()
    batches = (
        (b"moru uses orli", b"orli serves moru"),
        (b"sena uses ulon", b"ulon serves sena"),
        (b"kelo uses vesa", b"rilo uses luma"),
    )
    for index, batch in enumerate(batches):
        learner = learner.consolidate(
            tuple(
                _document(f"neutral-{index}", ordinal, content)
                for ordinal, content in enumerate(batch)
            )
        )
    return learner


def _neutral_action_success(learner, key: bytes, value: bytes):
    path = key.decode() + ".py"
    decoy = OpaqueWorkspaceFile("decoy.py", b"foreign = 'spare'\n")
    files = (decoy, OpaqueWorkspaceFile(path, key + b" = 'old" + key + b"'\n"))
    after = (decoy, OpaqueWorkspaceFile(path, key + b" = '" + value + b"'\n"))
    observation = UnifiedOpaqueObservationTape.observe(b"serves " + key, (), files)
    # This is a receipt-custody unit fixture. The development runner separately
    # exercises actual Python subprocess execution and behavioral scoring.
    authority = KnowledgeSubstitutionExecutionAuthority(
        lambda **kwargs: kwargs["after_workspace"] == after,
        sha256(b"unit-receipt-acceptance").hexdigest(),
    )
    receipt = authority.execute_and_issue(
        producer_checkpoint_sha256=learner.checkpoint_sha256,
        observation=observation,
        after_workspace=after,
    )
    return KnowledgeSubstitutionSuccess.seal(
        observation=observation,
        after_workspace=after,
        execution_receipt=receipt,
        provenance_id=b"unit-demonstration/" + key,
    )


def test_neutral_action_rejects_success_from_before_document_update() -> None:
    learner = _neutral_action_seed()
    success = _neutral_action_success(learner, b"kelo", b"vesa")
    newer = learner.consolidate((_document("later", 0, b"qelo uses raku"),))
    newer_root = newer.checkpoint_sha256
    with pytest.raises(ValueError, match="another parent checkpoint"):
        newer.learn_action(success)
    assert newer.checkpoint_sha256 == newer_root
    assert newer.demonstration_count == 0
    learned = learner.learn_action(success)
    assert learned.parent_checkpoint_sha256 == learner.checkpoint_sha256
    assert learned.demonstration_count == 1


def test_neutral_action_preserves_multi_anchor_ambiguity_after_transfer() -> None:
    learner = _neutral_action_seed()
    for key, value in ((b"kelo", b"vesa"), (b"rilo", b"luma")):
        learner = learner.learn_action(_neutral_action_success(learner, key, value))
    learner = learner.consolidate(
        (
            _document("fresh-a", 0, b"moro uses zani"),
            _document("fresh-b", 0, b"qelo uses zani"),
        )
    )
    assert learner.retained_demonstration_event_bytes == 0
    files = (
        OpaqueWorkspaceFile("fresh.py", b"moro = 'gray'\n"),
        OpaqueWorkspaceFile("other.py", b"qelo = 'white'\n"),
    )
    checkpoint = learner.checkpoint_sha256
    transfer = learner.respond(b"serves moro", files=files)
    assert transfer.status == "EDIT"
    assert transfer.after_workspace == (
        OpaqueWorkspaceFile("fresh.py", b"moro = 'zani'\n"),
        files[1],
    )
    collision = learner.respond(b"serves moro. serves qelo.", files=files)
    assert collision.producer_decision.emitted == b"zani"
    assert collision.producer_decision.support_query_lexemes == (b"moro", b"qelo")
    assert collision.status == "STOP"
    assert collision.stop_reason == "action:ambiguous_anchor"
    assert collision.after_workspace is None
    assert collision.motor_races[-1].stopped
    assert learner.checkpoint_sha256 == checkpoint


def test_neutral_document_python_session_executes_fresh_transfer(tmp_path) -> None:
    report = run(tmp_path / "session")
    assert report["narrow_integration_passed"]
    transfer = [row for row in report["attempts"] if row["stage"] == "transfer"]
    assert [row["passed"] for row in transfer] == [True, False, True, False]
    assert [row["execution"]["stdout"].strip() for row in transfer] == [
        "zani", "gray", "raku", "white",
    ]
    assert all(not row["before_execution"]["passed"] for row in transfer)
    assert all(row["execution"]["decoys_unchanged"] for row in transfer)
    assert all(row["execution"]["materialized_matches_action"] for row in transfer)
    assert all(not row["source_documents_present"] for row in transfer)
    variation = next(row for row in report["attempts"] if row["stage"] == "layout-variation")
    assert not variation["passed"]  # Keep the unlearned layout visible in development.
    assert variation["status"] == "STOP"


def test_contextual_motor_uses_all_request_seeds_and_stops_on_competing_targets() -> None:
    from experiments.r4_integrated.contextual_occurrence_motor import ContextualOccurrenceMotor
    from experiments.r4_integrated.neutral_document_python_session import Case

    motor = ContextualOccurrenceMotor()
    first = Case("motor-a", "axel", "fresh", "motor_alpha", "cold")
    checkpoint = motor.checkpoint_sha256
    bad_after = (*first.files(corrected=True)[:-1], OpaqueWorkspaceFile("wrong.py", b"changed"))
    with pytest.raises(ValueError, match="preserve all workspace paths"):
        motor.learn(b"Configure axel", first.files(), bad_after)
    assert motor.checkpoint_sha256 == checkpoint
    for case in (first, Case("motor-b", "bora", "ready", "motor_beta", "warm")):
        motor.learn(f"Configure {case.key}".encode(), case.files(), case.files(corrected=True))
    assert motor.demonstration_count == 2
    assert motor.retained_demonstration_event_bytes == 0
    checkpoint = motor.checkpoint_sha256
    case = Case("fresh", "cera", "bright", "fresh_options", "dark")
    request = b"Configure cera for the service in this workspace"
    decision = motor.apply(request, case.files(), b"bright")
    assert decision.edited
    assert decision.files == case.files(corrected=True)
    assert decision.request_seed_count == len(set(request.split()))
    assert len(decision.binder_decisions) == decision.request_seed_count
    assert decision.physical_cell_steps > 0 and decision.evaluated_cells > 0
    multiword = motor.apply(request, case.files(), b"bright blue")
    assert multiword.edited
    assert next(item.content for item in multiword.files if item.path == "fresh_options.py") == (
        b"cera = 'bright blue'\n"
    )
    # Both natural-language entities support the same learned action pattern.
    # Neither request order nor file order may break the physical collision.
    files = tuple(
        sorted(
            (*case.files(), OpaqueWorkspaceFile("another.py", b"daro = 'dim'\n")),
            key=lambda item: item.path,
        )
    )
    collision = motor.apply(b"Configure cera and daro", files, b"bright")
    assert not collision.edited
    assert collision.files == files
    assert collision.supported_role_count == 2
    assert collision.selection_race.stop_reason == "collision"
    assert motor.checkpoint_sha256 == checkpoint
