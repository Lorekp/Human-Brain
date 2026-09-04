from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from humanbrain.contextual_evidence_memory import ContextualEvidenceMemory
from humanbrain.frozen_context_encoder import FrozenContextEncoder
from humanbrain.opaque_document_event_tape import OpaqueDocument


@pytest.fixture(scope="module")
def encoder() -> FrozenContextEncoder:
    pytest.importorskip("onnxruntime", reason="optional contextual runtime is not installed")
    pytest.importorskip("tokenizers", reason="optional contextual tokenizer is not installed")
    model_dir = Path(__file__).resolve().parents[1] / ".models/all-MiniLM-L6-v2"
    if not (model_dir / "manifest.json").exists():
        pytest.skip("pinned contextual model is not downloaded; tests never download it")
    return FrozenContextEncoder(model_dir)


def document(identity: bytes = b"public-test") -> OpaqueDocument:
    return OpaqueDocument(
        identity,
        b"synthetic-unit-test",
        "Café receives 17 messages. Hazel receives 23 messages.".encode(),
    )


def test_real_encoder_preserves_utf8_pooling_and_rejects_truncation(encoder) -> None:
    raw = "Café naïve data.".encode()
    encoded = encoder.encode(raw)
    assert encoded.input_sha256 == sha256(raw).hexdigest()
    assert encoded.input_byte_count == len(raw)
    for start, end in encoded.byte_spans:
        assert raw[start:end].decode("utf-8")
    pooled = encoded.pool_spans(((0, len("Café".encode())),))
    assert pooled.shape == (1, 384)
    assert np.isfinite(pooled).all()
    for array in (encoded.token_features, encoded.sentence_vector, pooled):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    np.testing.assert_array_equal(encoded.token_features, encoder.encode(raw).token_features)
    with pytest.raises(ValueError, match="splits a contextual subtoken"):
        encoded.pool_spans(((1, len("Café".encode())),))
    with pytest.raises(ValueError, match="declared limit"):
        encoder.encode(b"word " * 300)


def test_ingestion_control_storage_and_inference_are_identical_and_read_only(encoder) -> None:
    update = ContextualEvidenceMemory(encoder)
    frozen = ContextualEvidenceMemory(encoder, learning_enabled=False)
    source = document()
    for memory in (update, frozen):
        memory.ingest((source,))
    assert update.checkpoint_sha256 == frozen.checkpoint_sha256
    assert update.storage_summary == frozen.storage_summary
    assert update.storage_summary["raw_source_bytes"] == len(source.content)
    assert update.storage_summary["compiled_race_array_bytes"] == 2 * 600 * 8
    # Generic candidates retain names and multiword spans, not just likely values.
    query = b"How many messages are received?"
    for value in ("Café".encode(), "Café receives 17".encode(), b"23"):
        assert update.decision_for_span(query, value).emitted == value
    before = update.checkpoint_sha256
    arrays = update.checkpoint_arrays()
    for policy in (update.respond, update.external_only, update.conventional):
        decision = policy(query)
        assert decision.checkpoint_sha256 == before
        assert update.checkpoint_sha256 == before
    for name, expected in arrays.items():
        np.testing.assert_array_equal(update.checkpoint_arrays()[name], expected)


def test_yoked_feedback_preserves_frozen_synapses_and_exact_eligibility(encoder) -> None:
    update = ContextualEvidenceMemory(encoder)
    frozen = ContextualEvidenceMemory(encoder, learning_enabled=False)
    for memory in (update, frozen):
        memory.ingest((document(),))
    initial = frozen.synapse_sha256
    query = b"How many messages does Cafe receive?"
    for reward in (1, -1):
        # Synthetic teaching exercises custody, without making a capability claim.
        decision = update.decision_for_span(query, b"17")
        replayed = frozen.replay_decision(decision)
        assert replayed.candidate_index == decision.candidate_index
        assert replayed.emitted == decision.emitted
        assert replayed.eligibility == decision.eligibility
        assert replayed.storage_sha256 == decision.storage_sha256
        assert replayed.checkpoint_sha256 == frozen.checkpoint_sha256
        assert replayed.mode == "yoked_teacher_executed_span"
        previous = update.synapse_sha256
        assert update.observe_feedback(decision, reward)["synaptic_update"]
        assert update.synapse_sha256 != previous
        assert not frozen.observe_feedback(replayed, reward)["synaptic_update"]
        assert frozen.synapse_sha256 == initial
        with pytest.raises(ValueError, match="stale parent"):
            update.observe_feedback(decision, reward)
        with pytest.raises(ValueError, match="stale parent"):
            frozen.observe_feedback(replayed, reward)
    assert update.feedback_count == frozen.feedback_count == 2


def test_replay_rejects_changed_store_tampering_and_stop(encoder) -> None:
    update = ContextualEvidenceMemory(encoder)
    frozen = ContextualEvidenceMemory(encoder, learning_enabled=False)
    update.ingest((document(),))
    frozen.ingest((document(),))
    query = b"How many messages does Cafe receive?"
    decision = update.decision_for_span(query, b"17")
    with pytest.raises(ValueError, match="custody changed"):
        frozen.replay_decision(replace(decision, emitted=b"23"))
    with pytest.raises(ValueError, match="STOP has no selected candidate"):
        frozen.replay_decision(update.respond(query))
    changed = ContextualEvidenceMemory(encoder, learning_enabled=False)
    changed.ingest((document(b"different-source-identity"),))
    with pytest.raises(ValueError, match="different retained source store"):
        changed.replay_decision(decision)
    with pytest.raises(ValueError, match="frozen synapses"):
        update.replay_decision(decision)


def test_teacher_span_never_chooses_a_duplicate_by_entity_identity(encoder) -> None:
    memory = ContextualEvidenceMemory(encoder)
    memory.ingest(
        (
            OpaqueDocument(
                b"duplicates",
                b"public-test",
                b"Juniper receives 17 messages. Hazel receives 17 messages.",
            ),
        )
    )
    before = memory.checkpoint_sha256
    for query in (b"How many messages does Juniper receive?", b"How many does Hazel receive?"):
        with pytest.raises(ValueError, match="exactly one retained occurrence"):
            memory.decision_for_span(query, b"17")
    assert memory.checkpoint_sha256 == before


def test_exported_synapses_are_independent_and_match_their_checksum(encoder) -> None:
    memory = ContextualEvidenceMemory(encoder)
    memory.ingest((document(),))
    memory.observe_feedback(memory.decision_for_span(b"How many messages?", b"17"), 1)
    arrays = memory.checkpoint_arrays()
    digest = sha256(
        arrays["readout_synapses_pa"].tobytes() + arrays["pacemaker_synapses_pa"].tobytes(),
    ).hexdigest()
    assert digest == memory.synapse_sha256
    assert sum(a.nbytes for a in arrays.values()) == memory.storage_summary["synapse_array_bytes"]
    for array in arrays.values():
        array.fill(0)
    assert memory.synapse_sha256 == digest


def test_pairwise_update_improves_gap_without_changing_bias_or_pure_query_weights(encoder) -> None:
    memory = ContextualEvidenceMemory(encoder)
    memory.ingest((document(),))
    query = b"How many messages does Cafe receive?"
    # A preceding absolute update leaves a nonzero persistent bias: pairwise
    # learning must neither use nor overwrite that state.
    memory.observe_feedback(memory.decision_for_span(query, b"23"), 1)
    rejected = memory.decision_for_span(query, b"23")
    accepted = memory.decision_for_span(query, b"17")
    delta = np.asarray(accepted.eligibility) - np.asarray(rejected.eligibility)
    np.testing.assert_array_equal(delta[:384], np.zeros(384))
    before = memory.checkpoint_arrays()
    assert np.any(before["pacemaker_synapses_pa"] != 0)
    receipt = memory.observe_comparison(rejected, accepted)
    after = memory.checkpoint_arrays()
    assert receipt["native_update_applied"] and receipt["synaptic_update"]
    assert receipt["gap_after"] > receipt["gap_before"]
    np.testing.assert_array_equal(after["pacemaker_synapses_pa"], before["pacemaker_synapses_pa"])
    np.testing.assert_array_equal(
        after["readout_synapses_pa"][:, :384], before["readout_synapses_pa"][:, :384]
    )
    assert memory.feedback_count == 3 and memory.update_count == 2
    with pytest.raises(ValueError, match="stale parent"):
        memory.observe_comparison(rejected, accepted)


def test_pairwise_replay_freezes_weights_and_counts_two_matched_events(encoder) -> None:
    update = ContextualEvidenceMemory(encoder)
    frozen = ContextualEvidenceMemory(encoder, learning_enabled=False)
    for memory in (update, frozen):
        memory.ingest((document(),))
    query = b"How many messages does Cafe receive?"
    rejected = update.decision_for_span(query, b"23")
    accepted = update.decision_for_span(query, b"17")
    replay_rejected = frozen.replay_decision(rejected)
    replay_accepted = frozen.replay_decision(accepted)
    frozen_weights = frozen.synapse_sha256
    first = update.observe_comparison(rejected, accepted)
    second = frozen.observe_comparison(replay_rejected, replay_accepted)
    assert first["eligibility_delta_sha256"] == second["eligibility_delta_sha256"]
    assert first["native_update_applied"] and not second["native_update_applied"]
    assert not second["synaptic_update"] and frozen.synapse_sha256 == frozen_weights
    assert update.feedback_count == frozen.feedback_count == 2
    assert update.update_count == 1 and frozen.update_count == 0
    assert second["before"] != second["after"]
    with pytest.raises(ValueError, match="stale parent"):
        frozen.observe_comparison(replay_rejected, replay_accepted)


def test_pairwise_rejects_mismatched_query_same_occurrence_and_stop_atomically(encoder) -> None:
    memory = ContextualEvidenceMemory(encoder)
    memory.ingest((document(),))
    query = b"How many messages does Cafe receive?"
    rejected = memory.decision_for_span(query, b"23")
    accepted = memory.decision_for_span(b"How many messages does Hazel receive?", b"17")
    before = memory.checkpoint_sha256
    with pytest.raises(ValueError, match="same query"):
        memory.observe_comparison(rejected, accepted)
    with pytest.raises(ValueError, match="different candidate occurrences"):
        memory.observe_comparison(rejected, rejected)
    with pytest.raises(ValueError, match="STOP has no selected candidate"):
        memory.observe_comparison(memory.respond(query), rejected)
    with pytest.raises(ValueError, match="custody changed"):
        memory.observe_comparison(rejected, replace(rejected, emitted=b"changed"))
    assert memory.checkpoint_sha256 == before
    assert memory.feedback_count == memory.update_count == 0
