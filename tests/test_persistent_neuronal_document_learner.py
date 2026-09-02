from __future__ import annotations

from hashlib import sha256

from humanbrain.neuronal_relational_reader import RelationalReadingSuccess
from humanbrain.opaque_document_event_tape import (
    OpaqueDocument,
    OpaqueDocumentEventTape,
)
from humanbrain.persistent_neuronal_document_learner import (
    PersistentDocumentAblation,
    advance,
    consolidate,
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
    assert empty.training_summary.association_count == 0
    assert empty.training_summary.retained_association_pointer_count == 0
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
    assert learned.physical_cell_steps == (
        sum(race.physical_cell_steps for race in learned.races)
        + sum(race.physical_cell_steps for race in learned.relation_binding_races)
        + sum(race.physical_cell_steps for race in learned.relation_cleanup_races)
    )
    assert learned.relation_binding_races
    assert learned.relation_cleanup_races
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
    assert complete.training_summary.retained_pointer_count == 12
    assert complete.training_summary.retained_association_pointer_count == 2
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
        PersistentDocumentAblation.RELATION_BINDING_OFF: "answer_selection",
        PersistentDocumentAblation.RELATION_CLEANUP_OFF: "answer_selection",
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


def test_full_cobinding_learns_only_the_two_ordered_cross_surface_links() -> None:
    def lesson(run: str, query: bytes, fact: bytes, answer: bytes):
        return RelationalReadingSuccess(
            query=query,
            documents=(
                OpaqueDocument(
                    document_id=f"bridge-document/{run}/0".encode(),
                    provenance_id=f"bridge-document-provenance/{run}/0".encode(),
                    content=fact,
                ),
                _document(f"bridge-{run}", 1, "qira."),
            ),
            observed_answer=answer,
            provenance_id=f"bridge-lesson-provenance/{run}".encode(),
        )

    surface_a = lesson(
        "surface-a",
        b"Route axor through lumet.",
        b"axor pavo iona.",
        b"iona",
    )
    first = advance(start(), surface_a)
    before_bridge = first.successor
    assert first.status == "PENDING"

    prebridge_query = b"Route moru through lumet."
    prebridge_documents = (
        _document("prebridge", 0, "qira."),
        _document("prebridge", 1, "orli sulex moru."),
    )
    assert before_bridge.respond(prebridge_query, prebridge_documents).decision == "STOP"

    for decoy in (
        lesson(
            "same-subject-only",
            b"Chart axor beside naver.",
            b"axor tivo zelu.",
            b"zelu",
        ),
        lesson(
            "same-answer-only",
            b"Mark qem beneath sorin.",
            b"qem vatu iona.",
            b"iona",
        ),
        lesson(
            "reversed-binding",
            b"Follow iona around pira.",
            b"iona daxu axor.",
            b"axor",
        ),
    ):
        transition = advance(before_bridge, decoy)
        assert transition.status == "PENDING"
        assert not transition.association_updated
        before_bridge = transition.successor

    assert before_bridge.training_summary.association_count == 0
    surface_b = lesson(
        "surface-b",
        b"Trace kori for axor.",
        b"iona sulex axor.",
        b"iona",
    )
    bridge = advance(before_bridge, surface_b)
    associated = bridge.successor
    assert bridge.status == "ASSOCIATED"
    assert bridge.association_updated
    assert bridge.association_link_count == 2
    assert len(bridge.association_binding_races) == 2
    assert all(not race.stopped for race in bridge.association_binding_races)
    assert len(bridge.association_cleanup_races) == 1
    assert not bridge.association_cleanup_races[0].stopped
    assert associated.training_summary.association_count == 2
    assert associated.training_summary.learned_equivalence_count == 1
    assert associated.training_summary.associated_surface_count == 2
    assert associated.training_summary.promoted_schema_count == 0
    assert associated.training_summary.retained_association_pointer_count == 2
    assert associated.training_summary.retained_raw_training_row_count == 0
    assert associated.training_summary.retained_raw_document_count == 0

    cross_documents = (
        _document("cross-a-to-b", 0, "moru tivo vexa."),
        _document("cross-a-to-b", 1, "nori. orli sulex moru."),
    )
    cross = associated.respond(prebridge_query, cross_documents)
    assert cross.decision == "EMIT"
    assert cross.answer == b"orli"
    assert cross.relation_binding_races
    assert cross.relation_cleanup_races
    assert any(not race.stopped for race in cross.relation_cleanup_races)
    assert any("associated_path_or" in race.population for race in cross.races)

    reverse_query = b"Trace kori for prax."
    reverse_documents = (
        _document("cross-b-to-a", 0, "prax pavo ravi."),
        _document("cross-b-to-a", 1, "prax tivo noxa."),
    )
    assert associated.respond(reverse_query, reverse_documents).answer == b"ravi"
    same_surface_documents = (_document("same-a", 0, "moru pavo orli."),)
    assert associated.respond(prebridge_query, same_surface_documents).decision == "STOP"
    assert before_bridge.respond(prebridge_query, cross_documents).decision == "STOP"

    ambiguous_documents = (
        _document("bridge-ambiguous", 0, "tuli sulex moru."),
        _document("bridge-ambiguous", 1, "vexa sulex moru."),
    )
    ambiguous = associated.respond(prebridge_query, ambiguous_documents)
    assert ambiguous.decision == "STOP"
    assert ambiguous.stop_reason == "answer_selection:collision"

    for ablation in (
        PersistentDocumentAblation.RELATION_BINDING_OFF,
        PersistentDocumentAblation.RELATION_CLEANUP_OFF,
        PersistentDocumentAblation.ANSWER_SELECTION_OFF,
    ):
        stopped = associated.respond(prebridge_query, cross_documents, ablation)
        assert stopped.decision == "STOP"
        assert stopped.stop_stage == "answer_selection"
    binding_off = associated.respond(
        prebridge_query,
        cross_documents,
        PersistentDocumentAblation.RELATION_BINDING_OFF,
    )
    assert any(race.stopped for race in binding_off.relation_binding_races)
    cleanup_off = associated.respond(
        prebridge_query,
        cross_documents,
        PersistentDocumentAblation.RELATION_CLEANUP_OFF,
    )
    assert any(race.stopped for race in cleanup_off.relation_cleanup_races)


def test_neutral_document_consolidation_and_delayed_neuronal_recall() -> None:
    first = advance(
        start(),
        _lesson("memory-frame-one", "forward", "axor", "iona", late=False),
    )
    trained = advance(
        first.successor,
        _lesson("memory-frame-two", "forward", "beviklong", "julo", late=True),
    ).successor
    assert trained.training_summary.promoted_schema_count == 1
    assert trained.training_summary.consolidated_fact_count == 0

    source_one = OpaqueDocument(
        document_id=b"neutral/document/one",
        provenance_id=b"neutral/provenance/one",
        content=(
            b"moru pavo orli. prax pavo ravi. "
            b"moru tivo vexa. qem tivo orli."
        ),
    )
    source_two = OpaqueDocument(
        document_id=b"neutral/document/two",
        provenance_id=b"neutral/provenance/two",
        content=b"moru pavo orli.",
    )
    neutral_documents = (source_one, source_two)
    trained_root = trained.checkpoint_sha256

    consolidation_off = consolidate(
        trained,
        neutral_documents,
        PersistentDocumentAblation.CONSOLIDATION_OFF,
    )
    assert consolidation_off.status == "STOP"
    assert not consolidation_off.updated
    assert consolidation_off.successor is trained
    assert consolidation_off.successor_checkpoint_sha256 == trained_root
    assert consolidation_off.stop_stage == "consolidation"
    assert consolidation_off.stop_reason == "consolidation:tonic_stop"
    assert consolidation_off.consolidation_cell_steps > 0
    assert not consolidation_off.binding_races
    assert any(
        race.population == "persistent/consolidation/write_control"
        for race in consolidation_off.races
    )

    stored_transition = consolidate(trained, neutral_documents)
    stored = stored_transition.successor
    assert stored_transition.status == "STORED"
    assert stored_transition.updated
    assert stored_transition.facts_added == 2
    assert len(stored_transition.fact_sha256s) == 2
    assert set(stored_transition.fact_sha256s) == set(stored.consolidated_fact_sha256s)
    assert stored_transition.parent_episode_replay_count == 0
    assert stored_transition.parent_mutation_count == 0
    assert stored_transition.retained_raw_document_count == 0
    assert stored_transition.consolidation_cell_steps > 0
    assert stored_transition.consolidation_active_synapses > 0
    assert stored_transition.binding_races
    assert trained.checkpoint_sha256 == trained_root
    assert trained.training_summary.consolidated_fact_count == 0
    assert stored.training_summary.consolidated_fact_count == 2
    assert stored.training_summary.neutral_episode_count == 1
    assert stored.training_summary.retained_fact_lexical_byte_count == 8
    assert stored.training_summary.retained_fact_provenance_digest_byte_count == 192
    assert stored.training_summary.retained_raw_training_row_count == 0
    assert stored.training_summary.retained_raw_document_count == 0
    assert stored.training_summary.cumulative_consolidation_cell_steps > 0
    two_source_fact = next(fact for fact in stored._facts if len(fact.sources) == 2)
    assert len(two_source_fact.binding_trace_sha256s) == 4
    assert all(len(source.binding_trace_sha256s) == 2 for source in two_source_fact.sources)
    assert (
        stored.training_summary.cumulative_binding_update_count
        - trained.training_summary.cumulative_binding_update_count
        == 12
    )
    assert {lexeme.identity_event for lexeme in stored._lexemes} == {b"orli", b"ravi"}
    assert all(
        not hasattr(fact, field)
        for fact in stored._facts
        for field in ("subject", "answer", "query", "documents", "content")
    )
    expected_provenance = {
        sha256(source_one.provenance_id).hexdigest(),
        sha256(source_two.provenance_id).hexdigest(),
    }
    assert set(stored.consolidated_provenance_sha256s) == expected_provenance

    recall_documents = (_document("memory-recall", 0, "qira."),)
    assert trained.respond(
        b"Route moru through lumet.",
        recall_documents,
    ).decision == "STOP"
    stored_root = stored.checkpoint_sha256
    moru = stored.respond(b"Route moru through lumet.", recall_documents)
    assert moru.decision == "EMIT"
    assert moru.answer == b"orli"
    assert moru.answer_source == "memory"
    assert moru.answer_occurrence_event is None
    assert moru.answer_document_id is None
    assert moru.answer_provenance_id is None
    assert moru.memory_fact_sha256 in stored.consolidated_fact_sha256s
    assert set(moru.answer_provenance_sha256s) == expected_provenance
    assert moru.physical_cell_steps > 0
    populations = {race.population for race in moru.races}
    assert "persistent/retrieval/unique_fact_selection" in populations
    assert any("key_equality/event_equality" in population for population in populations)
    assert "persistent/retrieval/lexical_motor" in populations
    assert moru.relation_binding_races
    assert moru.relation_cleanup_races
    assert stored.checkpoint_sha256 == stored_root

    prax = stored.respond(b"Route prax through lumet.", recall_documents)
    assert prax.decision == "EMIT"
    assert prax.answer == b"ravi"
    qem = stored.respond(b"Route qem through lumet.", recall_documents)
    assert qem.decision == "STOP"
    assert qem.stop_stage == "retrieval"

    retrieval_off = stored.respond(
        b"Route moru through lumet.",
        recall_documents,
        PersistentDocumentAblation.RETRIEVAL_OFF,
    )
    assert retrieval_off.decision == "STOP"
    assert retrieval_off.stop_stage == "retrieval"
    assert retrieval_off.physical_cell_steps > 0
    assert any(race.stopped for race in retrieval_off.relation_cleanup_races)
    motor_off = stored.respond(
        b"Route moru through lumet.",
        recall_documents,
        PersistentDocumentAblation.MOTOR_OFF,
    )
    assert motor_off.decision == "STOP"
    assert motor_off.stop_stage == "motor"

    duplicate = consolidate(stored, neutral_documents)
    assert duplicate.status == "NO_UPDATE"
    assert not duplicate.updated
    assert duplicate.successor is stored
    assert duplicate.facts_added == 0

    conflict_source = OpaqueDocument(
        document_id=b"neutral/document/conflict",
        provenance_id=b"neutral/provenance/conflict",
        content=b"moru pavo ulon.",
    )
    conflict_transition = consolidate(stored, (conflict_source,))
    conflicted = conflict_transition.successor
    assert conflict_transition.status == "STORED"
    assert conflict_transition.facts_added == 1
    conflict = conflicted.respond(b"Route moru through lumet.", recall_documents)
    assert conflict.decision == "STOP"
    assert conflict.stop_stage == "retrieval"
    assert conflict.stop_reason == "retrieval:collision"
    assert any(
        race.population == "persistent/retrieval/unique_fact_selection"
        and race.stopped
        and race.stop_reason == "collision"
        for race in conflict.races
    )
    assert stored.respond(b"Route moru through lumet.", recall_documents).answer == b"orli"
