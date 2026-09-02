from __future__ import annotations

from humanbrain.neuronal_relational_reader import RelationalReadingSuccess
from humanbrain.opaque_document_event_tape import (
    OpaqueDocument,
    OpaqueDocumentEventTape,
)
from humanbrain.persistent_neuronal_document_learner import (
    PersistentDocumentAblation,
    advance,
    start,
)


def _document(run: str, ordinal: int, content: str) -> OpaqueDocument:
    return OpaqueDocument(
        document_id=f"document/{run}/{ordinal}".encode(),
        provenance_id=f"document-provenance/{run}/{ordinal}".encode(),
        content=content.encode(),
    )


def _case(
    run: str,
    surface: str,
    subject: str,
    answer: str,
    *,
    late: bool,
) -> tuple[bytes, tuple[OpaqueDocument, ...]]:
    if surface == "forward":
        query = f"Route {subject} through lumet."
        fact = f"{subject} pavo {answer}."
    else:
        query = f"Trace kori for {subject}."
        fact = f"{answer} sulex {subject}."
    if late:
        documents = (
            _document(run, 0, "qira."),
            _document(run, 1, f"nori. {fact}"),
        )
    else:
        documents = (
            _document(run, 0, fact),
            _document(run, 1, "qira."),
        )
    return query.encode(), documents


def _lesson(
    run: str,
    surface: str,
    subject: str,
    answer: str,
    *,
    late: bool,
) -> RelationalReadingSuccess:
    query, documents = _case(run, surface, subject, answer, late=late)
    return RelationalReadingSuccess(
        query=query,
        documents=documents,
        observed_answer=answer.encode(),
        provenance_id=f"lesson-provenance/{run}".encode(),
    )


def test_replay_free_exact_surface_growth_and_physical_one_hop_read() -> None:
    empty = start()
    assert empty.checkpoint_sha256 == start().checkpoint_sha256
    assert empty.training_summary.accepted_episode_count == 0
    assert empty.training_summary.retained_raw_training_row_count == 0
    assert empty.training_summary.retained_raw_document_count == 0
    assert empty.training_summary.retained_surface_literal_byte_count == 0
    assert empty.training_summary.retained_canonical_state_byte_count > 0
    assert empty.training_summary.retained_pointer_count == 0
    assert empty.training_summary.retained_active_cell_count == 0
    assert empty.training_summary.cumulative_binding_update_count == 0

    forward_one = _lesson(
        "forward-one",
        "forward",
        "axor",
        "iona",
        late=False,
    )
    first = advance(empty, forward_one)
    pending = first.successor
    assert first.status == "PENDING"
    assert first.updated
    assert first.schema_support_before == 0
    assert first.schema_support_after == 1
    assert first.delta_chunks_added == 1
    assert first.parent_episode_replay_count == 0
    assert first.parent_mutation_count == 0
    assert first.retained_raw_training_row_count == 0
    assert first.learning_cell_steps > 0
    assert first.learning_active_synapses > 0
    assert pending.training_summary.pending_schema_count == 1
    assert pending.training_summary.promoted_schema_count == 0
    assert pending.training_summary.retained_raw_document_count == 0
    assert pending.training_summary.retained_surface_literal_byte_count > 0
    assert (
        pending.training_summary.retained_canonical_state_byte_count
        > empty.training_summary.retained_canonical_state_byte_count
    )
    assert pending.training_summary.retained_pointer_count == 3
    assert pending.training_summary.retained_active_cell_count > 0
    assert pending.training_summary.cumulative_binding_update_count == 2
    assert pending.training_summary.cumulative_learning_active_synapses > 0

    fresh_forward_query, fresh_forward_documents = _case(
        "fresh-forward",
        "forward",
        "moru",
        "orli",
        late=True,
    )
    early = pending.respond(fresh_forward_query, fresh_forward_documents)
    assert early.decision == "STOP"
    assert early.stop_stage == "schema"
    assert early.answer is None
    assert early.physical_cell_steps > 0
    assert any(race.stopped for race in early.races)

    changed_row = _lesson(
        "changed-duplicate-row",
        "forward",
        "zarek",
        "ulon",
        late=True,
    )
    duplicate_provenance = RelationalReadingSuccess(
        query=changed_row.query,
        documents=changed_row.documents,
        observed_answer=changed_row.observed_answer,
        provenance_id=forward_one.provenance_id,
    )
    replay = advance(pending, duplicate_provenance)
    assert replay.status == "NO_UPDATE"
    assert not replay.updated
    assert replay.successor is pending
    assert replay.successor_checkpoint_sha256 == pending.checkpoint_sha256
    assert replay.delta_chunks_added == 0
    assert replay.learning_cell_steps == 0
    assert replay.learning_active_synapses == 0
    assert replay.parent_mutation_count == 0

    reused_document_source = RelationalReadingSuccess(
        query=b"Route zarek through lumet.",
        documents=(
            OpaqueDocument(
                document_id=forward_one.documents[0].document_id,
                provenance_id=b"document-provenance/reused-source/new-version",
                content=b"zarek pavo ulon.",
            ),
            _document("reused-source", 1, "qira."),
        ),
        observed_answer=b"ulon",
        provenance_id=b"lesson-provenance/reused-source",
    )
    reused_source = advance(pending, reused_document_source)
    assert reused_source.status == "NO_UPDATE"
    assert not reused_source.updated
    assert reused_source.successor is pending
    assert reused_source.schema_support_before == 1
    assert reused_source.schema_support_after == 1
    assert reused_source.delta_chunks_added == 0
    assert reused_source.parent_episode_replay_count == 0
    assert reused_source.parent_mutation_count == 0
    assert reused_source.learning_cell_steps > 0
    assert reused_source.learning_active_synapses > 0
    assert pending.schema_support_ticks == (1,)

    pending_root = pending.checkpoint_sha256
    pending_chunks = pending.evidence_chunks
    forward_two = _lesson(
        "forward-two",
        "forward",
        "beviklong",
        "julo",
        late=True,
    )
    second = advance(pending, forward_two)
    promoted = second.successor
    assert second.status == "PROMOTED"
    assert second.promoted_schema
    assert second.parent_mutation_count == 0
    assert second.schema_support_before == 1
    assert second.schema_support_after == 2
    assert promoted.training_summary.promoted_schema_count == 1
    assert promoted.training_summary.pending_schema_count == 0
    assert promoted.schema_support_ticks == (2,)
    assert pending.checkpoint_sha256 == pending_root
    assert pending.evidence_chunks == pending_chunks
    assert promoted.evidence_chunks[0] is pending.evidence_chunks[0]

    learned = promoted.respond(fresh_forward_query, fresh_forward_documents)
    assert learned.decision == "EMIT"
    assert learned.answer == b"orli"
    assert learned.completed_hops == 1
    assert learned.stop_reason is None
    assert learned.schema_sha256 in promoted.promoted_schema_sha256s
    assert learned.physical_cell_steps == sum(race.physical_cell_steps for race in learned.races)
    tape = OpaqueDocumentEventTape.observe(
        fresh_forward_query,
        fresh_forward_documents,
    )
    expected = next(
        occurrence
        for document in tape.document_token_occurrences
        for occurrence in document
        if occurrence.identity_event == b"orli"
    )
    assert learned.answer_occurrence_event == expected.occurrence_event
    assert learned.answer_document_id == expected.document_id
    assert learned.answer_provenance_id == expected.provenance_id
    populations = {race.population for race in learned.races}
    assert any("query_frame/literal" in population for population in populations)
    assert any("document_frame/literal" in population for population in populations)
    assert any("subject_equality/event_equality" in population for population in populations)
    assert "persistent/unique_answer_selection" in populations
    assert any(
        "motor_" in population and "byte_equality" in population for population in populations
    )
    assert "persistent/live_occurrence_motor" in populations

    reverse_one = _lesson(
        "reverse-one",
        "reverse",
        "cuniq",
        "kemi",
        late=False,
    )
    third = advance(promoted, reverse_one)
    interleaved = third.successor
    assert third.status == "PENDING"
    assert interleaved.training_summary.promoted_schema_count == 1
    assert interleaved.training_summary.pending_schema_count == 1
    assert (
        interleaved.respond(
            fresh_forward_query,
            fresh_forward_documents,
        ).answer
        == b"orli"
    )

    fresh_reverse_query, fresh_reverse_documents = _case(
        "fresh-reverse",
        "reverse",
        "prax",
        "ravi",
        late=True,
    )
    reverse_early = interleaved.respond(
        fresh_reverse_query,
        fresh_reverse_documents,
    )
    assert reverse_early.decision == "STOP"
    assert reverse_early.stop_stage == "query_frame"

    reverse_two = _lesson(
        "reverse-two",
        "reverse",
        "dralextended",
        "lora",
        late=True,
    )
    fourth = advance(interleaved, reverse_two)
    complete = fourth.successor
    assert fourth.status == "PROMOTED"
    assert fourth.parent_mutation_count == 0
    assert complete.training_summary.accepted_episode_count == 4
    assert complete.training_summary.delta_chunk_count == 4
    assert complete.training_summary.schema_count == 2
    assert complete.training_summary.promoted_schema_count == 2
    assert complete.training_summary.pending_schema_count == 0
    assert complete.training_summary.parent_episode_replay_count == 0
    assert complete.training_summary.retained_raw_training_row_count == 0
    assert complete.training_summary.retained_raw_document_count == 0
    assert complete.training_summary.retained_canonical_state_byte_count > 0
    assert complete.training_summary.retained_pointer_count == 10
    assert complete.training_summary.retained_active_cell_count > 0
    assert complete.training_summary.cumulative_binding_update_count == 8
    assert complete.training_summary.cumulative_learning_cell_steps > 0
    assert complete.training_summary.cumulative_learning_active_synapses > 0
    assert complete.evidence_chunks[0] is pending.evidence_chunks[0]
    assert all(
        not hasattr(chunk, field)
        for chunk in complete.evidence_chunks
        for field in ("query", "documents", "observed_answer", "provenance_id")
    )
    assert (
        complete.respond(
            fresh_forward_query,
            fresh_forward_documents,
        ).answer
        == b"orli"
    )
    reverse = complete.respond(fresh_reverse_query, fresh_reverse_documents)
    assert reverse.decision == "EMIT"
    assert reverse.answer == b"ravi"

    assert pending.checkpoint_sha256 == pending_root
    assert pending.training_summary.promoted_schema_count == 0
    assert (
        pending.respond(
            fresh_forward_query,
            fresh_forward_documents,
        ).decision
        == "STOP"
    )

    ambiguous_query = b"Route rima through lumet."
    ambiguous_documents = (
        _document("ambiguous", 0, "rima pavo tuli."),
        _document("ambiguous", 1, "rima pavo vexa."),
    )
    ambiguous = complete.respond(ambiguous_query, ambiguous_documents)
    assert ambiguous.decision == "STOP"
    assert ambiguous.stop_stage == "answer_selection"
    assert ambiguous.stop_reason == "answer_selection:collision"
    assert any(
        race.population == "persistent/unique_answer_selection"
        and race.stopped
        and race.stop_reason == "collision"
        for race in ambiguous.races
    )

    missing_documents = (
        _document("missing", 0, "rima tarsa tuli."),
        _document("missing", 1, "qira."),
    )
    missing = complete.respond(ambiguous_query, missing_documents)
    assert missing.decision == "STOP"
    assert missing.stop_stage == "answer_selection"
    assert missing.stop_reason == "answer_selection:tonic_stop"

    expected_stages = {
        PersistentDocumentAblation.QUERY_FRAME_OFF: "query_frame",
        PersistentDocumentAblation.DOCUMENT_FRAME_OFF: "answer_selection",
        PersistentDocumentAblation.SUBJECT_JOIN_OFF: "answer_selection",
        PersistentDocumentAblation.ANSWER_SELECTION_OFF: "answer_selection",
        PersistentDocumentAblation.MOTOR_OFF: "motor",
    }
    for ablation, expected_stage in expected_stages.items():
        ablated = complete.respond(
            fresh_forward_query,
            fresh_forward_documents,
            ablation,
        )
        assert ablated.decision == "STOP"
        assert ablated.answer is None
        assert ablated.stop_stage == expected_stage
        assert ablated.physical_cell_steps > 0
        assert any(race.stopped for race in ablated.races)
