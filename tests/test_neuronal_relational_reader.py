from __future__ import annotations

import inspect
from dataclasses import fields

import pytest
from humanbrain.neuronal_relational_reader import (
    NeuronalRelationalReader,
    RelationalReaderAblation,
    RelationalReadingSuccess,
)
from humanbrain.opaque_document_event_tape import OpaqueDocument, OpaqueDocumentEventTape


def _document(run: str, ordinal: int, content: str) -> OpaqueDocument:
    return OpaqueDocument(
        document_id=f"document-{run}-{ordinal}".encode(),
        provenance_id=f"document-provenance-{run}-{ordinal}".encode(),
        content=content.encode(),
    )


def _success(
    run: str,
    subject: str,
    intermediate: str,
    answer: str,
    *,
    reversed_arguments: bool,
) -> RelationalReadingSuccess:
    if reversed_arguments:
        first = f"{intermediate} sulex {subject}."
        second = f"{answer} ovar {intermediate}."
    else:
        first = f"{subject} pavo {intermediate}."
        second = f"{intermediate} nir {answer}."
    return RelationalReadingSuccess(
        query=f"Route {subject} through lumet then kori.".encode(),
        documents=(
            _document(run, 0, first),
            _document(run, 1, second),
        ),
        observed_answer=answer.encode(),
        provenance_id=f"success-provenance-{run}".encode(),
    )


def _fresh_documents(
    run: str,
    subject: str,
    intermediate: str,
    answer: str,
    *,
    first_reversed: bool,
    second_reversed: bool,
) -> tuple[OpaqueDocument, ...]:
    first = (
        f"{intermediate} sulex {subject}." if first_reversed else f"{subject} pavo {intermediate}."
    )
    second = (
        f"{answer} ovar {intermediate}." if second_reversed else f"{intermediate} nir {answer}."
    )
    # The second hop is presented before the first, both live facts move to a
    # later sentence, and learned-looking as well as subject-bearing distractors
    # remain fully eligible to the circuit.
    return (
        _document(run, 0, f"qira tevon wexa. {second}"),
        _document(run, 1, f"{subject} zint yalo. fexa nori peld."),
        _document(run, 2, "tari pavo ulen. sova nir deri."),
        _document(run, 3, f"vaku jera timo. {first}"),
    )


def test_learns_generic_surfaces_and_runs_a_physical_two_hop_read() -> None:
    training = (
        _success("a", "axor", "enza", "iona", reversed_arguments=False),
        _success("b", "bevik", "firo", "julo", reversed_arguments=False),
        _success("c", "cuniq", "gavi", "kemi", reversed_arguments=True),
        _success("d", "dral", "hexa", "lora", reversed_arguments=True),
    )
    reader = NeuronalRelationalReader.grow(training)

    assert tuple(field.name for field in fields(RelationalReadingSuccess)) == (
        "query",
        "documents",
        "observed_answer",
        "provenance_id",
    )
    assert tuple(inspect.signature(reader.respond).parameters) == (
        "query",
        "documents",
        "ablation",
    )
    assert reader.training_summary.success_count == 4
    assert reader.training_summary.hop_count == 2
    assert reader.training_summary.surface_frame_count == 4
    assert reader.training_summary.forward_surface_count == 2
    assert reader.training_summary.reverse_surface_count == 2
    assert reader.training_summary.minimum_surface_support == 2
    assert reader.training_summary.retained_concrete_endpoint_count == 0

    concrete_fillers = {
        value.encode()
        for value in (
            "axor",
            "enza",
            "iona",
            "bevik",
            "firo",
            "julo",
            "cuniq",
            "gavi",
            "kemi",
            "dral",
            "hexa",
            "lora",
        )
    }
    assert concrete_fillers.isdisjoint(reader.learned_surface_events)
    assert not hasattr(reader, "_successes")

    query = b"Route moru through lumet then kori."
    documents = _fresh_documents(
        "fresh-a",
        "moru",
        "nexa",
        "orli",
        first_reversed=True,
        second_reversed=False,
    )
    decision = reader.respond(query, documents)
    assert decision.decision == "EMIT"
    assert decision.answer == b"orli"
    assert decision.completed_hops == 2
    assert decision.physical_cell_steps > 0
    assert any(race.population == "query_to_document_equality_join" for race in decision.races)
    assert any(race.population == "recurrent_document_equality_join" for race in decision.races)
    assert decision.races[-1].population == "live_answer_motor"
    populations = tuple(race.population for race in decision.races)
    assert any(
        "/byte_equality_" in population and population.endswith("/coincidence_cells")
        for population in populations
    )
    assert any("/feature_conjunction_" in population for population in populations)
    assert any("/frame_or_" in population for population in populations)
    assert any(
        population.startswith("motor_candidate_") and "/live_byte_equality/" in population
        for population in populations
    )
    live = OpaqueDocumentEventTape.observe(query, documents)
    assert any(
        occurrence.identity_event == decision.answer
        and occurrence.occurrence_event == decision.answer_occurrence_event
        for occurrence in live.token_occurrences
    )

    changed_query = reader.respond(b"Route moru through lumeu then kori.", documents)
    assert changed_query.decision == "STOP"
    assert changed_query.stop_stage == "query_frame"
    assert any(
        race.population.startswith("query_frame/literal_3/byte_equality_4/")
        for race in changed_query.races
    )

    # A second fresh read recombines the other pair of learned orientations and
    # places the answer at the beginning rather than the end of its sentence.
    second_query = b"Route prax through lumet then kori."
    second_documents = _fresh_documents(
        "fresh-b",
        "prax",
        "quor",
        "ravi",
        first_reversed=False,
        second_reversed=True,
    )
    second_decision = reader.respond(second_query, second_documents)
    assert second_decision.decision == "EMIT"
    assert second_decision.answer == b"ravi"

    ambiguous_query = b"Route rima through lumet then kori."
    ambiguous_documents = (
        _document("ambiguous", 0, "rima pavo sato."),
        _document("ambiguous", 1, "sato nir tuli."),
        _document("ambiguous", 2, "vexa ovar sato."),
    )
    ambiguous = reader.respond(ambiguous_query, ambiguous_documents)
    assert ambiguous.decision == "STOP"
    assert ambiguous.stop_stage == "recurrent_join"
    assert ambiguous.stop_reason == "recurrent_join:collision"
    assert ambiguous.answer is None

    missing_query = b"Route wilo through lumet then kori."
    missing_documents = (
        _document("missing", 0, "wilo pavo xeno."),
        _document("missing", 1, "xeno tarsa yuni."),
    )
    missing = reader.respond(missing_query, missing_documents)
    assert missing.decision == "STOP"
    assert missing.stop_stage == "recurrent_join"
    assert missing.answer is None
    assert any(
        race.population.startswith("hop_2_edge_") and "/literal_1/byte_equality_" in race.population
        for race in missing.races
    )
    assert any("/frame_or_" in race.population for race in missing.races)

    expected_stages = {
        RelationalReaderAblation.EQUALITY_OFF: "first_join",
        RelationalReaderAblation.FRAME_BINDING_OFF: "query_frame",
        RelationalReaderAblation.RECURRENCE_OFF: "recurrent_join",
        RelationalReaderAblation.MOTOR_OFF: "motor",
    }
    for ablation, stage in expected_stages.items():
        stopped = reader.respond(query, documents, ablation)
        assert stopped.decision == "STOP"
        assert stopped.stop_stage == stage
        assert stopped.answer is None
        assert stopped.races[-1].stopped

    normal_equality_control = next(
        race for race in decision.races if race.population == "equality_control"
    )
    equality_off = reader.respond(query, documents, RelationalReaderAblation.EQUALITY_OFF)
    disabled_equality_control = next(
        race for race in equality_off.races if race.population == "equality_control"
    )
    assert (
        normal_equality_control.cell_scores_sha256 != disabled_equality_control.cell_scores_sha256
    )

    aliasing_training = (
        _success("alias-a", "Route", "enza", "iona", reversed_arguments=False),
        training[1],
        training[2],
        training[3],
    )
    with pytest.raises(ValueError, match="aliases a concrete training endpoint"):
        NeuronalRelationalReader.grow(aliasing_training)
