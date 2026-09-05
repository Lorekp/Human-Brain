from __future__ import annotations

import json
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from humanbrain.contextual_evidence_memory import (
    ContextualEvidenceMemory,
    _candidate_match_features,
    _position_kernels,
)
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
    for policy in (
        update.respond,
        update.external_only,
        update.conventional,
        update.raw_score_reference,
    ):
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


@pytest.mark.parametrize("interaction_gain", (1.0, 16.0))
@pytest.mark.parametrize("candidate_interactions", (False, True))
def test_revisit_preserves_occurrences_after_updates_and_appended_duplicates(
    encoder, interaction_gain, candidate_interactions
) -> None:
    update = ContextualEvidenceMemory(
        encoder, interaction_gain=interaction_gain, candidate_interactions=candidate_interactions
    )
    frozen = ContextualEvidenceMemory(
        encoder,
        learning_enabled=False,
        interaction_gain=interaction_gain,
        candidate_interactions=candidate_interactions,
    )
    for memory in (update, frozen):
        memory.ingest((document(),))
    query = b"How many messages does Cafe receive?"
    rejected = update.decision_for_span(query, b"23")
    accepted = update.decision_for_span(query, b"17")
    original_sha = (rejected.sha256, accepted.sha256)
    initial_frozen_synapses = frozen.synapse_sha256
    frozen_pair = tuple(frozen.replay_decision(row) for row in (rejected, accepted))
    first = update.observe_comparison(rejected, accepted)
    frozen.observe_comparison(*frozen_pair)
    assert first["synaptic_update"]
    if candidate_interactions:
        assert np.any(update.checkpoint_arrays()["readout_synapses_pa"][:, 1537:] != 0)

    appended = OpaqueDocument(
        b"new-source", b"public-test", b"Birch receives 23 messages and sends 31."
    )
    for memory in (update, frozen):
        memory.ingest((appended,))
    with pytest.raises(ValueError, match="exactly one retained occurrence"):
        update.decision_for_span(query, b"23")
    # Exercise another real update after append, before revisiting either side.
    intervening = update.decision_for_span(b"How many does Birch send?", b"31")
    yoked = frozen.replay_decision(intervening)
    assert update.observe_feedback(intervening, 1)["synaptic_update"]
    frozen.observe_feedback(yoked, 1)

    revisited = []
    for memory in (update, frozen):
        before = memory.checkpoint_sha256
        summary = memory.storage_summary
        pair = tuple(memory.revisit_decision(row) for row in (rejected, accepted))
        for old, current in zip((rejected, accepted), pair, strict=True):
            assert current.candidate_index == old.candidate_index
            assert current.emitted == old.emitted
            assert current.query == old.query and current.eligibility == old.eligibility
            assert current.checkpoint_sha256 == before != old.checkpoint_sha256
            assert current.storage_sha256 != old.storage_sha256
            assert current.mode == "revisited_teacher_executed_span"
            assert current.physical_cell_steps == 0 and current.spike_step is None
        assert memory.checkpoint_sha256 == before and memory.storage_summary == summary
        assert summary["storage_ancestry_root_count"] == 3
        assert summary["storage_ancestry_payload_bytes"] == 3 * (64 + 8)
        revisited.append(pair)
    assert (rejected.sha256, accepted.sha256) == original_sha
    with pytest.raises(ValueError, match="different retained source store"):
        frozen.replay_decision(rejected)
    with pytest.raises(ValueError, match="frozen synapses"):
        update.replay_decision(rejected)

    updated_receipt = update.observe_comparison(*revisited[0])
    frozen_receipt = frozen.observe_comparison(*revisited[1])
    assert updated_receipt["eligibility_delta_sha256"] == first["eligibility_delta_sha256"]
    assert frozen_receipt["eligibility_delta_sha256"] == first["eligibility_delta_sha256"]
    assert updated_receipt["native_update_applied"] == (updated_receipt["gap_before"] < 1)
    assert not frozen_receipt["synaptic_update"]
    assert frozen.synapse_sha256 == initial_frozen_synapses
    assert update.feedback_count == frozen.feedback_count == 5
    assert update.update_count == 2 + int(updated_receipt["native_update_applied"])
    assert frozen.update_count == 0


def test_revisit_rejects_forged_stop_and_unrelated_source_identity(encoder) -> None:
    memory = ContextualEvidenceMemory(encoder)
    memory.ingest((document(),))
    query = b"How many messages does Cafe receive?"
    selected = memory.decision_for_span(query, b"17")
    before = memory.checkpoint_sha256
    with pytest.raises(ValueError, match="custody changed"):
        memory.revisit_decision(replace(selected, emitted=b"23"))
    with pytest.raises(ValueError, match="STOP has no selected candidate"):
        memory.revisit_decision(memory.respond(query))
    for learning_enabled in (True, False):
        other = ContextualEvidenceMemory(encoder, learning_enabled=learning_enabled)
        other.ingest((document(b"different-identity-same-content"),))
        same_text = other.decision_for_span(query, b"17")
        assert same_text.eligibility == selected.eligibility
        assert same_text.candidate_index == selected.candidate_index
        with pytest.raises(ValueError, match="different retained source store"):
            other.revisit_decision(selected)
        assert other.feedback_count == other.update_count == 0
    assert memory.checkpoint_sha256 == before


def test_revisit_validates_original_count_and_retained_eligibility_even_with_new_sha(
    encoder,
) -> None:
    memory = ContextualEvidenceMemory(encoder)
    memory.ingest((document(),))
    query = b"How many messages does Cafe receive?"
    selected = memory.decision_for_span(query, b"17")
    original_count = memory.storage_summary["candidate_count"]
    memory.ingest((document(b"later-identical-content"),))
    rows, _ = memory._features(query)
    duplicate_index = selected.candidate_index + original_count
    assert memory._candidates[duplicate_index].emitted == selected.emitted
    assert tuple(rows[duplicate_index]) == selected.eligibility

    for changed in (
        replace(selected, candidate_index=duplicate_index),
        replace(selected, candidate_index=-1),
        replace(selected, candidate_index=1.5),
        replace(selected, emitted=b"23"),
        replace(selected, eligibility=(0.0,) * len(selected.eligibility)),
    ):
        # A self-consistent checksum alone is not an occurrence validation.
        record = asdict(changed)
        record.pop("sha256")
        record["query"] = changed.query.hex()
        record["emitted"] = changed.emitted.hex()
        digest = sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()
        before = memory.checkpoint_sha256
        with pytest.raises(ValueError, match="eligibility does not match"):
            memory.revisit_decision(replace(changed, sha256=digest))
        assert memory.checkpoint_sha256 == before
    # Retained source context must still produce the original exact eligibility.
    candidate = memory._candidates[selected.candidate_index]
    memory._contexts[candidate.window_index] = np.zeros(384)
    with pytest.raises(ValueError, match="eligibility does not match"):
        memory.revisit_decision(selected)


@pytest.mark.parametrize("gain", (0.0, -1.0, float("nan"), float("inf"), -float("inf")))
def test_interaction_gain_rejects_nonpositive_and_nonfinite_values(encoder, gain) -> None:
    with pytest.raises(ValueError, match="interaction_gain must be finite and positive"):
        ContextualEvidenceMemory(encoder, interaction_gain=gain)


def test_interaction_gain_scales_only_product_blocks_and_binds_identity(encoder) -> None:
    default = ContextualEvidenceMemory(encoder)
    baseline = ContextualEvidenceMemory(encoder, interaction_gain=1.0)
    amplified = ContextualEvidenceMemory(encoder, interaction_gain=16.0)
    frozen = ContextualEvidenceMemory(encoder, interaction_gain=16.0, learning_enabled=False)
    assert default.checkpoint_sha256 == baseline.checkpoint_sha256
    assert amplified.checkpoint_sha256 == frozen.checkpoint_sha256
    assert baseline.checkpoint_sha256 != amplified.checkpoint_sha256
    for memory in (default, baseline, amplified, frozen):
        memory.ingest((document(),))
        assert memory.storage_summary["interaction_gain"] == memory.interaction_gain
        assert memory.representation_policy["interaction_gain"] == memory.interaction_gain
    query = b"Does Cafe receive 17 messages?"
    default_rows, default_similarity = default._features(query)
    baseline_rows, baseline_similarity = baseline._features(query)
    amplified_rows, amplified_similarity = amplified._features(query)
    np.testing.assert_array_equal(default_rows, baseline_rows)
    np.testing.assert_array_equal(default_similarity, baseline_similarity)
    np.testing.assert_array_equal(baseline_similarity, amplified_similarity)
    np.testing.assert_array_equal(
        baseline_rows[:, 768:1152], baseline_rows[:, :384] * baseline_rows[:, 384:768]
    )
    np.testing.assert_array_equal(amplified_rows[:, :768], baseline_rows[:, :768])
    np.testing.assert_array_equal(amplified_rows[:, 768:1536], 16 * baseline_rows[:, 768:1536])
    np.testing.assert_array_equal(amplified_rows[:, 1536:], baseline_rows[:, 1536:])
    for policy in (amplified.respond, amplified.conventional, amplified.external_only):
        decision = policy(query)
        if not decision.stopped:
            assert decision.eligibility == tuple(amplified_rows[decision.candidate_index])
    teacher = amplified.decision_for_span(query, b"17")
    assert teacher.eligibility == tuple(amplified_rows[teacher.candidate_index])
    assert frozen.replay_decision(teacher).eligibility == teacher.eligibility
    with pytest.raises(AttributeError):
        amplified.interaction_gain = 1.0


def test_pairwise_native_update_uses_scaled_coordinates_and_rejects_other_gain(encoder) -> None:
    baseline = ContextualEvidenceMemory(encoder)
    amplified = ContextualEvidenceMemory(encoder, interaction_gain=16.0)
    frozen_baseline = ContextualEvidenceMemory(encoder, learning_enabled=False)
    frozen_amplified = ContextualEvidenceMemory(
        encoder, interaction_gain=16.0, learning_enabled=False
    )
    sources = (document(), OpaqueDocument(b"birch", b"public", b"Birch sends 31 reports."))
    for memory in (baseline, amplified, frozen_baseline, frozen_amplified):
        memory.ingest(sources)
    query = b"How many messages does Cafe receive?"
    old = baseline.decision_for_span(query, b"17")
    rejected = amplified.decision_for_span(query, b"31")
    accepted = amplified.decision_for_span(query, b"17")
    assert old.candidate_index == accepted.candidate_index and old.emitted == accepted.emitted
    assert old.storage_sha256 != accepted.storage_sha256
    for incompatible, decision in ((frozen_amplified, old), (frozen_baseline, accepted)):
        with pytest.raises(ValueError, match="different retained source store"):
            incompatible.replay_decision(decision)
        with pytest.raises(ValueError, match="different retained source store"):
            incompatible.revisit_decision(decision)
    delta = np.asarray(accepted.eligibility) - np.asarray(rejected.eligibility)
    assert np.linalg.norm(delta[768:1152]) > 0
    assert np.linalg.norm(delta[1152:1536]) > 0
    frozen_pair = tuple(frozen_amplified.replay_decision(row) for row in (rejected, accepted))
    receipt = amplified.observe_comparison(rejected, accepted)
    frozen_receipt = frozen_amplified.observe_comparison(*frozen_pair)
    step = amplified.learning_rate / np.sqrt(1.0 + delta @ delta)
    weights = amplified.checkpoint_arrays()
    np.testing.assert_allclose(weights["readout_synapses_pa"][0], step * delta, atol=1e-14)
    np.testing.assert_allclose(weights["readout_synapses_pa"][1], -step * delta, atol=1e-14)
    np.testing.assert_array_equal(weights["pacemaker_synapses_pa"], np.zeros(2))
    assert receipt["gap_after"] == pytest.approx(2 * step * (delta @ delta))
    assert receipt["eligibility_delta_sha256"] == frozen_receipt["eligibility_delta_sha256"]
    assert receipt["synaptic_update"] and not frozen_receipt["synaptic_update"]


def test_raw_score_reference_preserves_distinct_saturated_drives_without_mutation(encoder) -> None:
    memory = ContextualEvidenceMemory(encoder, interaction_gain=16.0)
    query = b"How many messages does Cafe receive?"
    assert memory.raw_score_reference(query).stopped
    memory.ingest((document(),))
    assert memory.raw_score_reference(query).stopped
    selected = memory.decision_for_span(query, b"17")
    # A large span drive saturates tanh, but retains a unique raw score maximum.
    memory._weights[0, 384:768] = 1_000 * memory._vectors[selected.candidate_index]
    before = memory.checkpoint_sha256
    raw = memory.raw_score_reference(query)
    assert raw.candidate_index == selected.candidate_index
    assert raw.emitted == b"17" and raw.mode == "raw_learned_score"
    assert raw.eligibility == selected.eligibility
    assert raw.physical_cell_steps == 0 and raw.spike_step is None
    assert memory.conventional(query).stopped
    assert memory.checkpoint_sha256 == before
    memory._weights.fill(0)
    memory._bias[0] = 1_000
    assert memory.raw_score_reference(query).stopped  # Equal raw scores stay ambiguous.


def test_raw_score_reference_requires_strict_win_over_tonic_stop(encoder) -> None:
    memory = ContextualEvidenceMemory(encoder)
    memory.ingest((OpaqueDocument(b"single", b"public", b"ready"),))
    threshold = np.arctanh((600.1 - 600.0) / 200.0)
    for bias in (0.0, threshold):
        memory._bias[0] = bias
        assert memory.raw_score_reference(b"Configure").stopped
    memory._bias[0] = np.nextafter(threshold, np.inf)
    assert memory.raw_score_reference(b"Configure").emitted == b"ready"


@pytest.mark.parametrize("candidate_interactions", (False, True))
def test_below_stop_candidate_exploration_preserves_custody_and_corrective_feedback(
    encoder, candidate_interactions
) -> None:
    update = ContextualEvidenceMemory(encoder, candidate_interactions=candidate_interactions)
    frozen = ContextualEvidenceMemory(
        encoder, candidate_interactions=candidate_interactions, learning_enabled=False
    )
    query = "How many messages does Café receive?".encode()
    for memory in (update, frozen):
        memory.ingest((document(),))
        selected = memory.decision_for_span(query, b"23")
        # A known, unique wrong leader at a negative raw drive: ordinary
        # thresholded policies must still abstain. Both mates have the same seed.
        memory._weights[0, 384:768] = 0.001 * memory._vectors[selected.candidate_index]
        memory._bias[0] = -5.0
    assert update.checkpoint_sha256 == frozen.checkpoint_sha256
    before, storage = update.checkpoint_sha256, update.storage_summary
    frozen_synapses = frozen.synapse_sha256
    rows, _ = update._features(query)
    assert np.max(update._raw_drives(rows)) < 0
    for policy in (
        update.raw_score_reference,
        update.conventional,
        update.respond,
        update.respond_interpolated,
    ):
        assert policy(query).stopped
    rejected = update.raw_candidate_proposal(query)
    assert rejected.emitted == b"23" and rejected.candidate_index == selected.candidate_index
    assert rejected.mode == "raw_candidate_exploration"
    assert rejected.eligibility == tuple(rows[rejected.candidate_index])
    assert rejected.checkpoint_sha256 == before
    assert rejected.physical_cell_steps == 0 and rejected.spike_step is None
    assert rejected.readout_policy is None and rejected.estimated_crossing_time_ms is None
    assert update.checkpoint_sha256 == before and update.storage_summary == storage
    # Execution authentication belongs to the runner. This unit test checks
    # that the proposal supplies ordinary same-query occurrence custody for credit.
    accepted = update.decision_for_span(query, b"17")
    with pytest.raises(ValueError, match="custody changed"):
        update.observe_comparison(replace(rejected, emitted=b"17"), accepted)
    frozen_pair = tuple(frozen.replay_decision(d) for d in (rejected, accepted))
    assert frozen_pair[0].mode == "yoked_raw_candidate_exploration"
    first = update.observe_comparison(rejected, accepted)
    second = frozen.observe_comparison(*frozen_pair)
    assert first["gap_before"] < 0 and first["gap_after"] > first["gap_before"]
    assert first["native_update_applied"] and first["synaptic_update"]
    assert not second["native_update_applied"] and not second["synaptic_update"]
    assert first["eligibility_delta_sha256"] == second["eligibility_delta_sha256"]
    assert frozen.synapse_sha256 == frozen_synapses
    assert update.feedback_count == frozen.feedback_count == 2
    assert update.update_count == 1 and frozen.update_count == 0
    with pytest.raises(ValueError, match="stale parent"):
        update.observe_comparison(rejected, accepted)


def test_candidate_exploration_rejects_empty_and_equal_leaders(encoder) -> None:
    memory = ContextualEvidenceMemory(encoder)
    query = b"Configure the setting"
    for populated in (False, True):
        if populated:
            memory.ingest((document(),))
            memory._bias[0] = -5  # Tonic exclusion must not break candidate ties.
        before = memory.checkpoint_sha256
        decision = memory.raw_candidate_proposal(query)
        assert decision.stopped and decision.mode == "raw_candidate_exploration"
        assert decision.stop_reason == "empty_or_tied_candidates"
        assert decision.eligibility == () and decision.emitted is None
        assert memory.checkpoint_sha256 == before


def test_relative_word_geometry_has_declared_direction_scale_and_interval_mean() -> None:
    kernels = _position_kernels(np.asarray([-2, 0, 2]))
    np.testing.assert_allclose(kernels[:, 1], 1)
    np.testing.assert_allclose(kernels[:3, 0], np.exp(-2 / np.asarray([2, 8, 32])))
    np.testing.assert_array_equal(kernels[:3, 2], 0)
    np.testing.assert_array_equal(kernels[3:, 0], 0)
    np.testing.assert_allclose(kernels[3:, 2], kernels[:3, 0])

    # Two distinct query anchors at adjacent source positions. Exact and
    # semantic channels coincide here, making every hand-computed value explicit.
    spans = np.asarray([[0, 0], [1, 1], [0, 1]])
    features = _candidate_match_features(np.eye(2), np.eye(2), spans)
    joint = features[:, :72].reshape(3, 6, 6, 2)
    direct = features[:, 72:].reshape(3, 6, 2)
    decay = np.exp(-0.5)
    np.testing.assert_allclose(joint[0, 0, 0], 0)
    np.testing.assert_allclose(joint[1, 0, 0], decay / 2)
    np.testing.assert_allclose(joint[0, 3, 3], decay / 2)
    np.testing.assert_allclose(direct[1, 0], (1 + decay) / 2)
    np.testing.assert_allclose(direct[0, 0], 0.5)
    np.testing.assert_allclose(direct[2], 1)  # Both words are inside the interval.
    np.testing.assert_allclose(
        _candidate_match_features(np.eye(2), np.eye(2), spans, collapsed=True), 1
    )
    single = _candidate_match_features(np.asarray([[1.0, 0]]), np.asarray([[0.8, 0.2]]), spans)
    np.testing.assert_array_equal(single[:, :72], 0)  # No other query occurrence.
    np.testing.assert_array_equal(
        _candidate_match_features(np.empty((0, 2)), np.empty((0, 2)), spans), 0
    )


def test_candidate_interactions_preserve_default_features_and_account_retained_state(
    encoder,
) -> None:
    default = ContextualEvidenceMemory(encoder, interaction_gain=16)
    disabled = ContextualEvidenceMemory(encoder, interaction_gain=16, candidate_interactions=False)
    enabled = ContextualEvidenceMemory(encoder, interaction_gain=16, candidate_interactions=True)
    assert default.checkpoint_sha256 == disabled.checkpoint_sha256
    assert "candidate_interactions" not in default.representation_policy
    assert enabled.checkpoint_sha256 != default.checkpoint_sha256
    for memory in (default, disabled, enabled):
        memory.ingest((document(),))
    query = "How many messages does Café receive?".encode()
    original, similarity = default._features(query)
    explicit, _ = disabled._features(query)
    expanded, expanded_similarity = enabled._features(query)
    assert original.shape[1] == 1537 and expanded.shape[1] == 1621
    np.testing.assert_array_equal(explicit, original)
    np.testing.assert_array_equal(expanded[:, :1537], original)
    np.testing.assert_array_equal(expanded_similarity, similarity)
    assert np.any(expanded[:, 1537:] > 0)
    assert [c.emitted for c in enabled._candidates] == [c.emitted for c in default._candidates]
    assert (
        enabled.storage_summary["context_vector_bytes"]
        == default.storage_summary["context_vector_bytes"]
    )
    windows = enabled._interaction_windows
    assert len(windows) == 1
    for window in windows:
        assert not window.word_candidate_indices.flags.writeable
        assert not window.candidate_word_spans.flags.writeable
        for index, word in zip(window.word_candidate_indices, window.words, strict=True):
            assert enabled._candidates[index].emitted.decode().casefold() == word
    summary = enabled.storage_summary
    new_payload = sum(w.payload_bytes for w in windows)
    assert summary["candidate_interaction_payload_bytes"] == new_payload
    assert summary["candidate_interaction_array_bytes"] == sum(
        w.word_candidate_indices.nbytes + w.candidate_word_spans.nbytes for w in windows
    )
    # New retained state is the explicit window payload and three arrays whose
    # feature width grows: GO/NO_GO weights and one eligibility vector.
    assert (
        summary["retained_array_and_byte_payload_bytes"]
        - default.storage_summary["retained_array_and_byte_payload_bytes"]
        == new_payload + 3 * 84 * 8
    )
    np.testing.assert_array_equal(enabled._features(b"?!")[0][:, 1537:], 0)
    with pytest.raises(AttributeError):
        enabled.candidate_interactions = False


def test_exact_interactions_bind_occurrences_under_consistent_renaming(encoder) -> None:
    original = ContextualEvidenceMemory(encoder, candidate_interactions=True)
    renamed = ContextualEvidenceMemory(encoder, candidate_interactions=True)
    original.ingest((document(),))
    renamed.ingest(
        (
            OpaqueDocument(
                b"renamed", b"public", b"Birch receives 17 messages. Juniper receives 23 messages."
            ),
        )
    )
    before = original._features("Café messages".encode())[0][:, 1537:]
    after = renamed._features(b"Birch messages")[0][:, 1537:]
    # Only exact channels are rename invariant; pretrained semantic vectors need
    # not be. Identical occurrence geometry preserves all 42 exact features.
    np.testing.assert_array_equal(before[:, ::2], after[:, ::2])
    other = renamed._features(b"Juniper messages")[0][:, 1537:]
    indices = [
        next(i for i, c in enumerate(renamed._candidates) if c.emitted == value)
        for value in (b"17", b"23")
    ]
    assert not np.array_equal(after[indices[0], ::2], other[indices[0], ::2])
    assert not np.array_equal(after[indices[0], ::2], after[indices[1], ::2])


def test_candidate_interaction_ablations_are_read_only_and_never_receive_feedback(encoder) -> None:
    memory = ContextualEvidenceMemory(encoder, candidate_interactions=True)
    frozen = ContextualEvidenceMemory(encoder, candidate_interactions=True, learning_enabled=False)
    for model in (memory, frozen):
        model.ingest((document(),))
    query = "Café messages".encode()
    selected = memory.decision_for_span(query, b"17")
    memory._weights[0, 384:768] = 1_000 * memory._vectors[selected.candidate_index]
    before = memory.checkpoint_sha256
    ordinary, _ = memory._features(query)
    for ablation in ("disabled", "position_collapsed"):
        rows, _ = memory._features(query, interaction_ablation=ablation)
        np.testing.assert_array_equal(rows[:, :1537], ordinary[:, :1537])
        if ablation == "disabled":
            np.testing.assert_array_equal(rows[:, 1537:], 0)
        else:
            np.testing.assert_array_equal(rows[:, 1537:], np.tile(rows[0, 1537:], (len(rows), 1)))
        decision = memory.raw_score_reference(query, interaction_ablation=ablation)
        assert decision.emitted == b"17"
        assert decision.mode == "raw_learned_score_ablation_" + ablation
        assert decision.eligibility == tuple(rows[decision.candidate_index])
        assert decision.physical_cell_steps == 0
        with pytest.raises(ValueError, match="read-only"):
            memory.observe_feedback(decision, 1)
        with pytest.raises(ValueError, match="read-only"):
            memory.revisit_decision(decision)
        with pytest.raises(ValueError, match="read-only"):
            frozen.replay_decision(decision)
    assert memory.checkpoint_sha256 == before
    assert memory.feedback_count == frozen.feedback_count == 0
    with pytest.raises(ValueError, match="unknown interaction ablation"):
        memory.raw_score_reference(query, interaction_ablation="arbitrary")
    baseline = ContextualEvidenceMemory(encoder)
    with pytest.raises(ValueError, match="requires candidate_interactions"):
        baseline.raw_score_reference(query, interaction_ablation="disabled")


@pytest.mark.parametrize("binding", (False, True))
def test_interpolated_readout_recovers_same_checkpoint_choice_and_preserves_ties(
    encoder, binding
) -> None:
    memory = ContextualEvidenceMemory(
        encoder, interaction_gain=16.0, candidate_interactions=binding
    )
    memory.ingest(
        (
            OpaqueDocument(b"timing-alpha", b"public", b"alpha"),
            OpaqueDocument(b"timing-beta", b"public", b"beta"),
        )
    )
    query = b"Select beta"
    rows, _ = memory._features(query)
    alpha = memory.decision_for_span(query, b"alpha").candidate_index
    beta = memory.decision_for_span(query, b"beta").candidate_index
    # Set a controlled synaptic state, not a learning result: two distinct
    # candidate currents cross in one coarse interval, with beta strictly first.
    drives = np.zeros(len(rows))
    drives[[alpha, beta]] = np.arctanh((np.asarray((799.0, 799.1)) - 600.0) / 200.0)
    memory._weights[0] = np.linalg.lstsq(rows, drives, rcond=None)[0]
    np.testing.assert_allclose(rows @ memory._weights[0], drives, atol=1e-12)
    before = memory.checkpoint_sha256
    discrete = memory.respond(query)
    refined = memory.respond_interpolated(query)
    assert discrete.stopped and "simultaneous first spikes" in discrete.stop_reason
    assert refined.candidate_index == memory.conventional(query).candidate_index == beta
    assert refined.emitted == b"beta" and refined.eligibility == tuple(rows[beta])
    assert refined.checkpoint_sha256 == discrete.checkpoint_sha256 == before
    assert memory.checkpoint_sha256 == before
    assert refined.readout_policy == memory.readout_policy["comparison"]
    assert discrete.readout_policy == memory.readout_policy["default"]
    assert refined.coarse_spike_candidates == 2 and refined.simultaneous_spike_candidates == 1
    assert refined.spike_step * 0.05 < refined.estimated_crossing_time_ms < (
        refined.spike_step + 1
    ) * 0.05
    with pytest.raises(ValueError, match="custody changed"):
        memory.observe_feedback(replace(refined, estimated_crossing_time_ms=0.0), 1)
    assert memory.feedback_count == memory.update_count == 0
    memory._weights.fill(0)
    memory._bias[0] = 3.0
    tied = memory.respond_interpolated(query)
    assert tied.stopped and "indistinguishable" in tied.stop_reason
