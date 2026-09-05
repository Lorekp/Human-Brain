from __future__ import annotations

import json
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from humanbrain.contextual_evidence_memory import (
    ContextualEvidenceMemory,
    _affine_currents,
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


def test_affine_current_map_preserves_stop_geometry_without_overflow() -> None:
    threshold = np.arctanh((600.1 - 600.0) / 200.0)
    raw = np.asarray([-2.0, 0.0, threshold, 3.0, 3.0])
    actual = _affine_currents(raw)
    np.testing.assert_allclose(actual[:-1], 600.1 + 200 * ((raw - threshold) / (3 - threshold)))
    assert actual[-1] == actual[2] == 600.1
    assert actual[0] < actual[1] < actual[2] < actual[3] == actual[4]
    permutation = np.asarray([4, 2, 0, 3, 1])
    np.testing.assert_array_equal(_affine_currents(raw[permutation])[:-1], actual[permutation])
    np.testing.assert_array_equal(_affine_currents(np.empty(0)), [600.1])
    with np.errstate(over="raise", invalid="raise"):
        extreme = _affine_currents(np.asarray([-np.finfo(float).max, np.finfo(float).max]))
    np.testing.assert_allclose(extreme, [400.1, 800.1, 600.1])
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="finite raw drives"):
            _affine_currents(np.asarray([value]))


@pytest.mark.parametrize("binding", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_affine_neuronal_readout_recovers_saturated_choice_without_identity_changes(
    encoder, binding, reverse
) -> None:
    memory = ContextualEvidenceMemory(encoder, candidate_interactions=binding)
    documents = (
        OpaqueDocument(b"affine-alpha", b"public", b"alpha"),
        OpaqueDocument(b"affine-beta", b"public", b"beta"),
    )
    memory.ingest(documents[::-1] if reverse else documents)
    query = b"Select beta"
    rows, _ = memory._features(query)
    # Controlled readout state, not a capability result: both raw drives are
    # positive and tanh-saturated, while beta has a distinct larger score.
    drives = np.asarray([48.0 if c.emitted == b"beta" else 32.0 for c in memory._candidates])
    memory._weights[0] = np.linalg.lstsq(rows, drives, rcond=None)[0]
    np.testing.assert_allclose(memory._raw_drives(rows), drives, atol=1e-12)
    before, storage = memory.checkpoint_sha256, memory.storage_summary
    old_policies = memory.readout_policy
    old = tuple(policy(query) for policy in (
        memory.respond, memory.respond_interpolated, memory.conventional, memory.raw_score_reference
    ))
    assert all(decision.stopped for decision in old[:3])
    assert old[3].emitted == b"beta"
    selected = memory.respond_affine_interpolated(query)
    assert selected.emitted == b"beta" and selected.candidate_index == old[3].candidate_index
    assert selected.mode == "neuronal_affine_interpolated"
    assert selected.eligibility == tuple(rows[selected.candidate_index])
    assert selected.physical_cell_steps > 0 and selected.spike_step is not None
    assert np.isfinite(selected.estimated_crossing_time_ms)
    assert selected.readout_policy.startswith("affine_stop_centered_v1;")
    assert "I=600.1+200*(d/R);STOP=600.1" in selected.readout_policy
    assert "linear_glif_endpoint_interpolation_v1" in selected.readout_policy
    assert selected.readout_policy not in old_policies.values()
    assert memory.checkpoint_sha256 == before == selected.checkpoint_sha256
    assert memory.storage_summary == storage and memory.readout_policy == old_policies
    assert old == tuple(policy(query) for policy in (
        memory.respond, memory.respond_interpolated, memory.conventional, memory.raw_score_reference
    ))
    with pytest.raises(ValueError, match="custody changed"):
        memory.observe_feedback(replace(selected, readout_policy=old_policies["comparison"]), 1)
    assert memory.checkpoint_sha256 == before


def test_affine_neuronal_empty_frozen_true_ties_and_floating_aliases_stop(encoder) -> None:
    memory = ContextualEvidenceMemory(encoder, learning_enabled=False)
    query = b"Configure"
    assert memory.respond_affine_interpolated(query).stopped
    memory.ingest((OpaqueDocument(b"single-affine", b"public", b"ready"),))
    assert memory.respond_affine_interpolated(query).stopped
    threshold = np.arctanh((600.1 - 600.0) / 200.0)
    for raw in (threshold, np.nextafter(threshold, np.inf)):
        memory._bias[0] = raw
        before = memory.checkpoint_sha256
        # The adjacent raw value is strictly above STOP, but current addition
        # rounds it onto STOP. Such aliases cannot justify inventing a winner.
        assert _affine_currents(np.asarray([raw]))[0] == 600.1
        decision = memory.respond_affine_interpolated(query)
        assert decision.stopped and "indistinguishable" in decision.stop_reason
        assert memory.checkpoint_sha256 == before
    assert memory.raw_score_reference(query).emitted == b"ready"
    memory._bias[0] = np.finfo(float).max
    assert memory.respond_affine_interpolated(query).emitted == b"ready"
    memory.ingest((OpaqueDocument(b"second-affine", b"public", b"steady"),))
    assert memory.respond_affine_interpolated(query).stopped  # Equal positive candidate leaders.
    for raw in (float("nan"), float("inf"), -float("inf")):
        memory._bias[0] = raw
        before = memory.checkpoint_sha256
        with pytest.raises(ValueError, match="finite raw drives"):
            memory.respond_affine_interpolated(query)
        assert memory.checkpoint_sha256 == before


@pytest.mark.parametrize("learning_rate", (0.001, 1.0))
@pytest.mark.parametrize("starting_gap", (-2.0, 0.0, 0.75))
def test_capped_comparison_native_step_matches_margin_arithmetic_and_frozen_exposure(
    encoder, learning_rate, starting_gap
) -> None:
    update = ContextualEvidenceMemory(
        encoder, learning_rate=learning_rate, candidate_interactions=True,
        comparison_step_policy="unit_margin_capped",
    )
    frozen = ContextualEvidenceMemory(
        encoder, learning_rate=learning_rate, candidate_interactions=True,
        comparison_step_policy="unit_margin_capped", learning_enabled=False,
    )
    for memory in (update, frozen):
        memory.ingest((document(),))
        memory._bias[0] = 0.25  # Persistent bias must not enter the pair calculation.
    assert update.checkpoint_sha256 == frozen.checkpoint_sha256
    query = "How many messages does Café receive?".encode()
    rejected = update.decision_for_span(query, b"23")
    accepted = update.decision_for_span(query, b"17")
    delta = np.asarray(accepted.eligibility) - np.asarray(rejected.eligibility)
    squared_norm = float(delta @ delta)
    initial_weights = starting_gap * delta / squared_norm
    for memory in (update, frozen):
        memory._weights[0] = initial_weights
    rejected = update.decision_for_span(query, b"23")
    accepted = update.decision_for_span(query, b"17")
    measured_gap = float(initial_weights @ delta)
    expected_rate = min(
        learning_rate, (1 - measured_gap) * np.sqrt(1 + squared_norm) / (2 * squared_norm)
    )
    expected_step = expected_rate / np.sqrt(1 + squared_norm)
    frozen_pair = tuple(frozen.replay_decision(d) for d in (rejected, accepted))
    frozen_sha = frozen.synapse_sha256
    receipt = update.observe_comparison(rejected, accepted)
    frozen_receipt = frozen.observe_comparison(*frozen_pair)
    json.dumps(receipt, allow_nan=False)
    json.dumps(frozen_receipt, allow_nan=False)
    arrays = update.checkpoint_arrays()
    np.testing.assert_allclose(
        arrays["readout_synapses_pa"][0], initial_weights + expected_step * delta, atol=1e-14
    )
    np.testing.assert_allclose(arrays["readout_synapses_pa"][1], -expected_step * delta, atol=1e-14)
    np.testing.assert_array_equal(arrays["pacemaker_synapses_pa"], [0.25, 0])
    np.testing.assert_array_equal(arrays["readout_synapses_pa"][:, :384], 0)
    assert receipt["eligibility_delta_norm_squared"] == squared_norm
    assert receipt["proposed_learning_rate"] == receipt["effective_learning_rate"] == expected_rate
    assert receipt["cap_active"] == (expected_rate < learning_rate)
    assert receipt["gap_before"] == pytest.approx(starting_gap, abs=1e-12)
    assert receipt["gap_after"] == pytest.approx(
        starting_gap + 2 * expected_step * squared_norm, abs=1e-12
    )
    assert receipt["gap_after"] <= 1 + 1e-12
    assert receipt["native_update_applied"] and receipt["synaptic_update"]
    assert receipt["comparison_step_policy"] == "unit_margin_capped"
    assert frozen_receipt["proposed_learning_rate"] == expected_rate
    assert frozen_receipt["effective_learning_rate"] == 0
    assert not frozen_receipt["native_update_applied"] and not frozen_receipt["synaptic_update"]
    assert frozen.synapse_sha256 == frozen_sha
    assert receipt["eligibility_delta_sha256"] == frozen_receipt["eligibility_delta_sha256"]
    assert update.feedback_count == frozen.feedback_count == 2
    assert update.update_count == 1 and frozen.update_count == 0
    with pytest.raises(ValueError, match="stale parent"):
        update.observe_comparison(rejected, accepted)


def test_comparison_policy_preserves_default_identity_and_absolute_feedback(encoder) -> None:
    default = ContextualEvidenceMemory(encoder)
    fixed = ContextualEvidenceMemory(encoder, comparison_step_policy="fixed")
    capped = ContextualEvidenceMemory(encoder, comparison_step_policy="unit_margin_capped")
    capped_frozen = ContextualEvidenceMemory(
        encoder, comparison_step_policy="unit_margin_capped", learning_enabled=False
    )
    query = b"How many messages?"
    for memory in (default, fixed, capped, capped_frozen):
        memory.ingest((document(),))
    assert default.checkpoint_sha256 == fixed.checkpoint_sha256 != capped.checkpoint_sha256
    assert capped.checkpoint_sha256 == capped_frozen.checkpoint_sha256
    # The fixed checkpoint keeps the exact pre-policy identity dictionary.
    old_state = {
        "encoder": encoder.sha256, "representation": default.representation_policy,
        "readouts": default.readout_policy, "storage": default._storage_root,
        "synapses": default.synapse_sha256, "feedback": default._feedback_root,
    }
    expected = sha256(json.dumps(old_state, sort_keys=True).encode()).hexdigest()
    assert default.checkpoint_sha256 == expected
    assert default.decision_for_span(query, b"17") == fixed.decision_for_span(query, b"17")
    np.testing.assert_array_equal(default._features(query)[0], capped._features(query)[0])
    assert default.storage_summary == capped.storage_summary
    with pytest.raises(ValueError, match="different retained source store"):
        capped_frozen.replay_decision(default.decision_for_span(query, b"17"))
    for memory in (default, fixed, capped):
        memory.observe_feedback(memory.decision_for_span(query, b"17"), 1)
    assert default.synapse_sha256 == fixed.synapse_sha256 == capped.synapse_sha256
    assert default.update_count == fixed.update_count == capped.update_count == 1
    with pytest.raises(AttributeError):
        capped.comparison_step_policy = "fixed"
    with pytest.raises(ValueError, match="unknown comparison_step_policy"):
        ContextualEvidenceMemory(encoder, comparison_step_policy="other")


def test_capped_comparison_satisfied_and_zero_delta_skip_without_mutation(encoder) -> None:
    memory = ContextualEvidenceMemory(encoder, comparison_step_policy="unit_margin_capped")
    memory.ingest((document(),))
    query = b"How many messages?"
    a = memory.decision_for_span(query, b"17")
    b = memory.decision_for_span(query, b"23")
    delta = np.asarray(a.eligibility) - np.asarray(b.eligibility)
    memory._weights[0] = 2 * delta / (delta @ delta)
    pair = [memory.decision_for_span(query, value) for value in (b"23", b"17")]
    before = memory.synapse_sha256
    receipt = memory.observe_comparison(*pair)
    assert receipt["gap_before"] == pytest.approx(2)
    assert receipt["step_skipped_reason"] == "satisfied_margin"
    assert receipt["effective_learning_rate"] == receipt["proposed_learning_rate"] == 0
    assert not receipt["native_update_applied"] and memory.synapse_sha256 == before
    assert memory.feedback_count == 2 and memory.update_count == 0

    # Different source occurrences may have identical features. Their zero
    # contrast cannot be learned; it must not produce a scratch-bias-only update.
    memory = ContextualEvidenceMemory(encoder, comparison_step_policy="unit_margin_capped")
    memory.ingest((
        OpaqueDocument(b"a", b"public", b"ready"), OpaqueDocument(b"b", b"public", b"ready")
    ))
    rows, _ = memory._features(query)
    pair = [memory._decision(query, rows, i, "teacher_executed_span") for i in (0, 1)]
    before = memory.synapse_sha256
    receipt = memory.observe_comparison(*pair)
    assert receipt["step_skipped_reason"] == "zero_delta"
    assert receipt["eligibility_delta_norm_squared"] == 0
    assert receipt["effective_learning_rate"] == receipt["proposed_learning_rate"] == 0
    assert not receipt["native_update_applied"] and not receipt["synaptic_update"]
    assert memory.synapse_sha256 == before and memory.update_count == 0
    assert memory.feedback_count == 2


class _SyntheticSpanEncoding:
    """Orthogonal word vectors for native plasticity tests; no pretrained model."""

    def __init__(self, raw: bytes):
        self.raw = raw
        self.sentence_vector = self.pool_spans(((0, len(raw)),))[0]

    def pool_spans(self, spans):
        vectors = np.zeros((len(spans), 384))
        vocabulary = {b"alpha": 0, b"beta": 1, b"gamma": 2, b"request": 3, b"another": 4}
        for row, (start, end) in zip(vectors, spans, strict=True):
            for word in self.raw[start:end].split():
                index = vocabulary.get(word, 5 + int.from_bytes(sha256(word).digest()[:2]) % 379)
                row[index] += 1.0
        return vectors


class _SyntheticProtectionEncoder:
    sha256 = sha256(b"synthetic-orthogonal-word-encoder-v1").hexdigest()

    def encode(self, raw):
        return _SyntheticSpanEncoding(raw)


def _protection_memory(**options):
    memory = ContextualEvidenceMemory(
        _SyntheticProtectionEncoder(),
        comparison_protection_policy="orthogonal_certified",
        **options,
    )
    memory.ingest((OpaqueDocument(b"synthetic", b"public-test", b"alpha beta gamma"),))
    return memory


def _protection_pair(memory, rejected=b"alpha", accepted=b"beta", query=b"request"):
    return tuple(memory.decision_for_span(query, word) for word in (rejected, accepted))


def test_protected_comparisons_preserve_raw_margins_and_original_normalizer() -> None:
    memory = _protection_memory()
    original_pair = _protection_pair(memory)
    first = memory.observe_comparison(*original_pair)
    d1 = np.asarray(original_pair[1].eligibility) - np.asarray(original_pair[0].eligibility)
    assert first["protection"]["consolidated"]
    assert first["protection"]["basis_after_rank"] == 1
    first_margin = first["gap_after"]
    pair = _protection_pair(memory, b"beta", b"gamma")
    d2 = np.asarray(pair[1].eligibility) - np.asarray(pair[0].eligibility)
    p = d2 - d1 * ((d1 @ d2) / (d1 @ d1))
    weights_before = memory._weights.copy()
    receipt = memory.observe_comparison(*pair)
    expected = p / np.sqrt(1 + d2 @ d2)
    np.testing.assert_allclose(memory._weights - weights_before, [expected, -expected], atol=1e-14)
    assert receipt["gap_after"] == pytest.approx(
        receipt["gap_before"] + 2 * (d2 @ p) / np.sqrt(1 + d2 @ d2), abs=1e-14
    )
    assert receipt["protection"]["original_dot_projected"] == pytest.approx(d2 @ p)
    assert receipt["protection"]["projected_delta_norm"] == pytest.approx(np.linalg.norm(p))
    assert receipt["eligibility_delta_sha256"] == sha256(d2.tobytes()).hexdigest()
    assert receipt["gap_after"] < 1  # No consolidation before its own unit certification.
    assert not receipt["protection"]["consolidated"]
    assert memory.protection_summary["protected_count"] == 1
    assert float((memory._weights[0] - memory._weights[1]) @ d1) == pytest.approx(first_margin)
    # The second authentic offer reaches the margin and alone can add direction two.
    next_receipt = memory.observe_comparison(*_protection_pair(memory, b"beta", b"gamma"))
    assert next_receipt["gap_after"] >= 1
    assert next_receipt["protection"]["basis_after_rank"] == 2
    assert next_receipt["protection"]["protected_margin_violations"] == 0
    np.testing.assert_allclose(
        memory._protection_basis @ memory._protection_basis.T, np.eye(2), atol=1e-14
    )
    np.testing.assert_array_equal(memory._weights[:, :384], 0)
    np.testing.assert_array_equal(memory._bias, 0)
    np.testing.assert_array_equal(memory._eligibility, 0)
    np.testing.assert_array_equal(memory._projected_eligibility, 0)


def test_protected_revisit_deduplicates_after_append_with_zero_frozen_growth() -> None:
    update, frozen = _protection_memory(), _protection_memory(learning_enabled=False)
    assert update.checkpoint_sha256 == frozen.checkpoint_sha256
    pair = _protection_pair(update)
    frozen_pair = tuple(frozen.replay_decision(value) for value in pair)
    update_receipt = update.observe_comparison(*pair)
    frozen_receipt = frozen.observe_comparison(*frozen_pair)
    assert update_receipt["eligibility_delta_sha256"] == frozen_receipt["eligibility_delta_sha256"]
    assert not frozen_receipt["protection"]["consolidated"]
    assert frozen.protection_summary["learned_payload_bytes"] == 0
    assert frozen.protection_summary["basis_rank"] == 0
    before = update.protection_summary
    arrays = update.checkpoint_arrays()
    records = json.loads(arrays["protection_certification"].tobytes())
    assert len(records) == 1 and records[0]["certification_margin"] == update_receipt["gap_after"]
    learned_bytes = sum(arrays[name].nbytes for name in (
        "protection_basis", "protected_contrast_deltas", "protection_certification"
    ))
    assert before["learned_payload_bytes"] == learned_bytes
    added = OpaqueDocument(b"appended", b"public-test", b"alpha beta gamma")
    for memory in (update, frozen):
        memory.ingest((added,))
    assert update.protection_summary == before
    rebound = tuple(update.revisit_decision(value) for value in pair)
    for old, new in zip(pair, rebound, strict=True):
        assert (old.candidate_index, old.eligibility) == (new.candidate_index, new.eligibility)
    frozen_pair = tuple(frozen.replay_decision(value) for value in rebound)
    repeat = update.observe_comparison(*rebound)
    frozen.observe_comparison(*frozen_pair)
    assert repeat["protection"]["contrast_id"] == update_receipt["protection"]["contrast_id"]
    assert repeat["protection"]["already_certified"]
    assert repeat["step_skipped_reason"] == "satisfied_margin"
    assert update.protection_summary == before
    assert frozen.protection_summary["learned_payload_bytes"] == 0
    assert frozen.update_count == 0 and frozen.feedback_count == update.feedback_count == 4
    parent = update.checkpoint_sha256
    for select in (
        update.respond_affine_interpolated, update.raw_score_reference, update.external_only
    ):
        select(b"request")
    assert update.checkpoint_sha256 == parent and update.protection_summary == before
    for array in arrays.values():
        array.fill(0)
    assert update.protection_summary == before  # Exported arrays own their memory.


def test_protected_dependent_certification_and_conflicting_projection_stall_are_bounded() -> None:
    memory = _protection_memory()
    memory.observe_comparison(*_protection_pair(memory))
    # A new query has the same contrast, already fitted: record its first
    # certification despite the native gate no-op, without adding a basis row.
    receipt = memory.observe_comparison(*_protection_pair(memory, query=b"another"))
    assert receipt["gap_before"] >= 1 and not receipt["native_update_applied"]
    assert receipt["protection"]["consolidated"]
    assert not receipt["protection"]["independent_direction_added"]
    assert memory.protection_summary["basis_rank"] == 1
    assert memory.protection_summary["protected_count"] == 2
    before = memory.protection_summary
    weights = memory.synapse_sha256
    receipt = memory.observe_comparison(*_protection_pair(memory, b"beta", b"alpha"))
    assert receipt["gap_before"] < 0
    assert receipt["step_skipped_reason"] == "projection_exhausted"
    assert receipt["protection"]["projection_stalled"]
    assert not receipt["native_update_applied"] and not receipt["protection"]["consolidated"]
    assert memory.synapse_sha256 == weights and memory.protection_summary == before
    assert memory.feedback_count == 6 and memory.update_count == 1


def test_protected_custody_policy_and_capacity_fail_explicitly() -> None:
    memory = _protection_memory(protection_capacity=1)
    pair = _protection_pair(memory)
    parent = memory.checkpoint_sha256
    with pytest.raises(ValueError, match="pairwise comparison"):
        memory.observe_feedback(pair[1], 1)
    with pytest.raises(ValueError, match="custody changed"):
        memory.observe_comparison(replace(pair[0], emitted=b"gamma"), pair[1])
    assert memory.checkpoint_sha256 == parent and memory.feedback_count == 0
    memory.observe_comparison(*pair)
    with pytest.raises(ValueError, match="stale parent"):
        memory.observe_comparison(*pair)
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        memory.observe_comparison(*_protection_pair(memory, query=b"another"))
    assert memory.protection_summary["protected_count"] == 1
    assert memory.feedback_count == 2
    different = _protection_memory(protection_capacity=2, learning_enabled=False)
    with pytest.raises(ValueError, match="different retained source store"):
        different.replay_decision(pair[0])
    for options in (
        {"comparison_protection_policy": "invalid"},
        {"protection_capacity": True},
        {"protection_capacity": 0},
        {"comparison_protection_policy": "orthogonal_certified",
         "comparison_step_policy": "unit_margin_capped"},
    ):
        with pytest.raises(ValueError):
            ContextualEvidenceMemory(_SyntheticProtectionEncoder(), **options)


def test_protected_distinct_ingested_occurrences_do_not_share_certification() -> None:
    memory = _protection_memory()
    first = memory.observe_comparison(*_protection_pair(memory))
    # Repeating even the same document identity creates separate retained
    # occurrences. Equal bytes and deltas alone must not merge their custody.
    memory.ingest((OpaqueDocument(b"synthetic", b"public-test", b"alpha beta gamma"),))
    rows, _ = memory._features(b"request")
    indices = [next(
        i for i, c in enumerate(memory._candidates)
        if c.document_index == 1 and c.emitted == word
    ) for word in (b"alpha", b"beta")]
    pair = [memory._decision(b"request", rows, i, "teacher_executed_span") for i in indices]
    second = memory.observe_comparison(*pair)
    assert second["eligibility_delta_sha256"] == first["eligibility_delta_sha256"]
    assert second["protection"]["contrast_id"] != first["protection"]["contrast_id"]
    assert second["protection"]["consolidated"]
    assert memory.protection_summary["protected_count"] == 2
    assert memory.protection_summary["basis_rank"] == 1


def test_protected_validation_rejects_native_raw_contrast_erasure(monkeypatch) -> None:
    from humanbrain.native import NativeThreeFactorLearner

    memory = _protection_memory()
    memory.observe_comparison(*_protection_pair(memory))
    before_count = memory.feedback_count
    real_update = NativeThreeFactorLearner.update_projected

    def corrupt_update(learner, *args, **kwargs):
        result = real_update(learner, *args, **kwargs)
        memory._weights[0] += memory._protected_deltas[0]
        return result

    monkeypatch.setattr(NativeThreeFactorLearner, "update_projected", corrupt_update)
    with pytest.raises(RuntimeError, match="margin preservation failed"):
        memory.observe_comparison(*_protection_pair(memory, b"beta", b"gamma"))
    assert memory.feedback_count == before_count  # A failed check is not a passed receipt.
    np.testing.assert_array_equal(memory._eligibility, 0)
    np.testing.assert_array_equal(memory._projected_eligibility, 0)


def _covariance_memory(**options):
    memory = ContextualEvidenceMemory(
        _SyntheticProtectionEncoder(), comparison_step_policy="arow", **options
    )
    memory.ingest((OpaqueDocument(b"synthetic", b"public-test", b"alpha beta gamma"),))
    return memory


def test_arow_core_covariance_custody_frozen_and_explicit_export() -> None:
    update = _covariance_memory(candidate_interactions=True)
    frozen = _covariance_memory(candidate_interactions=True, learning_enabled=False)
    initial = update.covariance_summary
    assert update.checkpoint_sha256 == frozen.checkpoint_sha256
    assert initial == frozen.covariance_summary
    assert initial["allocated_array_bytes"] == 1621**2 * 8
    assert initial["learned_payload_bytes"] == 0
    assert "covariance" not in update.checkpoint_arrays()
    assert not update._covariance.flags.writeable
    original = _protection_pair(update)
    frozen_pair = tuple(frozen.replay_decision(d) for d in original)
    d = np.array(original[1].eligibility) - np.array(original[0].eligibility)
    beta = 1 / (d @ d + 1)
    receipt = update.observe_comparison(*original)
    frozen_receipt = frozen.observe_comparison(*frozen_pair)
    np.testing.assert_allclose(update._weights, [0.5 * beta * d, -0.5 * beta * d], atol=1e-14)
    np.testing.assert_allclose(update._covariance, np.eye(len(d)) - beta * np.outer(d, d))
    assert receipt["gap_after"] == pytest.approx((d @ d) / (1 + d @ d))
    assert receipt["covariance"]["covariance_eligibility_sha256"] == sha256(d.tobytes()).hexdigest()
    assert receipt["covariance"]["covariance_update"]
    assert update.covariance_summary["update_count"] == 1
    assert update.covariance_summary["learned_payload_bytes"] == 1621**2 * 8
    assert frozen.covariance_summary == initial
    assert not frozen_receipt["native_update_applied"]
    assert not frozen_receipt["covariance"]["direction_computed"]
    assert frozen_receipt["covariance"]["directional_variance"] is None
    assert frozen.update_count == 0 and frozen.feedback_count == update.feedback_count == 2
    for memory in (update, frozen):
        np.testing.assert_array_equal(memory._bias, 0)
        np.testing.assert_array_equal(memory._eligibility, 0)
        np.testing.assert_array_equal(memory._covariance_eligibility, 0)
        exported = memory.checkpoint_arrays(include_learning_state=True)["covariance"]
        assert (
            sha256(exported.tobytes()).hexdigest()
            == memory.covariance_summary["covariance_sha256"]
        )
        exported.fill(0)
        assert np.any(memory._covariance)  # Caller cannot change retained/cached state.
        before = memory.checkpoint_sha256
        for select in (
            memory.respond_affine_interpolated, memory.raw_score_reference, memory.external_only
        ):
            select(b"request")
        assert memory.checkpoint_sha256 == before
        assert not memory._covariance.flags.writeable
        plain = ContextualEvidenceMemory(_SyntheticProtectionEncoder(), candidate_interactions=True)
        plain.ingest(tuple(memory._documents))
        additional = (
            memory.storage_summary["retained_array_and_byte_payload_bytes"]
            - plain.storage_summary["retained_array_and_byte_payload_bytes"]
        )
        assert additional == initial["allocated_array_bytes"] + initial["scratch_array_bytes"]
    with pytest.raises(ValueError, match="stale parent"):
        update.observe_comparison(*original)
    frozen_fixed = ContextualEvidenceMemory(_SyntheticProtectionEncoder(), learning_enabled=False,
                                           candidate_interactions=True)
    frozen_fixed.ingest(tuple(update._documents))
    with pytest.raises(ValueError, match="different retained source store"):
        frozen_fixed.replay_decision(original[0])


def test_arow_core_revisit_has_history_dependent_direction_without_source_mutation() -> None:
    memory = _covariance_memory()
    original = _protection_pair(memory)
    first = memory.observe_comparison(*original)
    pair = _protection_pair(memory, b"beta", b"gamma")
    d = np.asarray(pair[1].eligibility) - np.asarray(pair[0].eligibility)
    before_s = memory.checkpoint_arrays(include_learning_state=True)["covariance"]
    before_w = memory._weights.copy()
    p = before_s @ d
    margin = float((before_w[0] - before_w[1]) @ d)
    beta = 1 / (d @ p + 1)
    alpha = (1 - margin) * beta
    receipt = memory.observe_comparison(*pair)
    np.testing.assert_allclose(memory._weights - before_w, [alpha * p / 2, -alpha * p / 2])
    np.testing.assert_allclose(memory._covariance, before_s - beta * np.outer(p, p), atol=1e-14)
    assert not np.allclose(p, d)
    assert receipt["covariance"]["before"] == first["covariance"]["after"]
    state = memory.covariance_summary
    memory.ingest((OpaqueDocument(b"later", b"public", b"another request"),))
    assert memory.covariance_summary == state
    rebound = tuple(memory.revisit_decision(d) for d in original)
    assert [d.eligibility for d in rebound] == [d.eligibility for d in original]
    assert memory.observe_comparison(*rebound)["covariance"]["before"] == state


def test_arow_core_satisfied_zero_and_policy_rejections() -> None:
    memory = _covariance_memory()
    pair = _protection_pair(memory)
    d = np.asarray(pair[1].eligibility) - np.asarray(pair[0].eligibility)
    memory._weights[0] = 2 * d / (d @ d)
    state = memory.covariance_summary
    receipt = memory.observe_comparison(*_protection_pair(memory))
    assert receipt["step_skipped_reason"] == "satisfied_margin"
    assert not receipt["native_update_applied"]
    assert memory.covariance_summary == state
    with pytest.raises(ValueError, match="pairwise comparison"):
        memory.observe_feedback(_protection_pair(memory)[1], 1)
    for options in (
        {"learning_rate": 2.0}, {"comparison_protection_policy": "orthogonal_certified"}
    ):
        with pytest.raises(ValueError):
            _covariance_memory(**options)
    memory = _covariance_memory()
    memory.ingest((OpaqueDocument(b"duplicate", b"public", b"alpha beta gamma"),))
    rows, _ = memory._features(b"request")
    indices = [i for i, c in enumerate(memory._candidates) if c.emitted == b"alpha"]
    pair = [memory._decision(b"request", rows, i, "teacher_executed_span") for i in indices]
    before = memory.covariance_summary
    receipt = memory.observe_comparison(*pair)
    assert receipt["step_skipped_reason"] == "zero_delta"
    assert memory.covariance_summary == before


def _projected_bilinear_memory(**options):
    memory = ContextualEvidenceMemory(
        _SyntheticProtectionEncoder(), candidate_interactions=True,
        interaction_gain=16.0, projected_bilinear=True, comparison_step_policy="arow", **options
    )
    memory.ingest((OpaqueDocument(b"projection-test", b"synthetic", b"alpha beta gamma"),))
    return memory


def test_projected_bilinear_recipe_formula_and_independent_block_ablations() -> None:
    memory = _projected_bilinear_memory()
    baseline = ContextualEvidenceMemory(
        _SyntheticProtectionEncoder(), candidate_interactions=True,
        interaction_gain=16.0, comparison_step_policy="arow"
    )
    baseline.ingest(tuple(memory._documents))
    random = np.random.default_rng(0)
    for matrix in (memory._projected_query_matrix, memory._projected_span_matrix):
        q, r = np.linalg.qr(random.standard_normal((384, 32), dtype=np.float64), mode="reduced")
        expected = q * np.where(np.diag(r) < 0, -1.0, 1.0)
        np.testing.assert_array_equal(matrix, expected)
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(32), atol=1e-14)
        with pytest.raises(ValueError):
            matrix.setflags(write=True)
    assert memory.projection_summary["query_matrix_sha256"] != memory.projection_summary[
        "span_matrix_sha256"
    ]
    full, external = memory._features(b"request")
    old, old_external = baseline._features(b"request")
    np.testing.assert_array_equal(full[:, :1621], old)
    np.testing.assert_array_equal(external, old_external)
    for i, span in enumerate(memory._vectors):
        q = full[i, :384]
        expected = 16 * np.outer(q @ memory._projected_query_matrix,
                                 span @ memory._projected_span_matrix).ravel()
        np.testing.assert_array_equal(full[i, 1621:], expected)
    disabled, _ = memory._features(b"request", projected_bilinear_ablation="disabled")
    diagonal, _ = memory._features(b"request", projected_bilinear_ablation="diagonal_only")
    binding_disabled, _ = memory._features(b"request", interaction_ablation="disabled")
    np.testing.assert_array_equal(disabled[:, :1621], full[:, :1621])
    np.testing.assert_array_equal(disabled[:, 1621:], 0)
    np.testing.assert_array_equal(diagonal[:, :1621], full[:, :1621])
    expected_diagonal = full[:, 1621:].reshape(-1, 32, 32) * np.eye(32)
    np.testing.assert_array_equal(diagonal[:, 1621:], expected_diagonal.reshape(-1, 1024))
    np.testing.assert_array_equal(binding_disabled[:, 1621:], full[:, 1621:])
    assert np.any(full[:, 1621:] != diagonal[:, 1621:])
    assert baseline.projection_summary == {"policy": "none"}
    assert "projected_bilinear" not in baseline.representation_policy
    assert "projected_bilinear" not in baseline.storage_summary


def test_projected_bilinear_learning_and_frozen_state_account_for_full_dimension() -> None:
    memory = _projected_bilinear_memory()
    frozen = _projected_bilinear_memory(learning_enabled=False)
    assert memory.checkpoint_sha256 == frozen.checkpoint_sha256
    projection, initial_covariance = memory.projection_summary, memory.covariance_summary
    assert projection["fixed_array_bytes"] == 196608
    assert projection["feature_dimension"] == 2645
    assert initial_covariance["allocated_array_bytes"] == 55968200
    assert initial_covariance["scratch_array_bytes"] == 2645 * 8
    pair = _protection_pair(memory)
    frozen_pair = tuple(frozen.replay_decision(row) for row in pair)
    delta = np.array(pair[1].eligibility) - np.array(pair[0].eligibility)
    beta = 1 / (1 + delta @ delta)
    receipt = memory.observe_comparison(*pair)
    control = frozen.observe_comparison(*frozen_pair)
    np.testing.assert_allclose(memory._weights, [beta * delta / 2, -beta * delta / 2], atol=1e-14)
    assert np.any(memory._weights[:, 1621:])
    assert receipt["covariance"]["matrix_vector_entries"] == 2645**2
    assert control["covariance"]["matrix_vector_entries"] == 0
    assert frozen.covariance_summary == initial_covariance
    assert memory.projection_summary == frozen.projection_summary == projection
    assert memory.feedback_count == frozen.feedback_count == 2
    assert frozen.update_count == 0
    np.testing.assert_array_equal(frozen._weights, 0)
    for model in (memory, frozen):
        np.testing.assert_array_equal(model._bias, 0)
        np.testing.assert_array_equal(model._weights[:, :384], 0)
        assert "projected_query_matrix" not in model.checkpoint_arrays()
        arrays = model.checkpoint_arrays(include_learning_state=True)
        assert set(arrays) == {
            "readout_synapses_pa", "pacemaker_synapses_pa", "covariance",
            "projected_query_matrix", "projected_span_matrix"
        }
        assert arrays["readout_synapses_pa"].shape == (2, 2645)
        for kind in ("query", "span"):
            array = arrays[f"projected_{kind}_matrix"]
            assert sha256(array.tobytes()).hexdigest() == projection[f"{kind}_matrix_sha256"]
            array.fill(0)
        assert model.projection_summary == projection
        storage = model.storage_summary
        source_bytes = sum(storage[key] for key in (
            "raw_source_bytes", "candidate_lexical_bytes", "context_vector_bytes",
            "identity_bytes", "storage_ancestry_payload_bytes",
            "candidate_interaction_payload_bytes"
        ))
        assert storage["payload_bytes"] == source_bytes + 55968200 + 196608
        assert storage["retained_array_and_byte_payload_bytes"] == (
            storage["payload_bytes"] + storage["synapse_array_bytes"] + 16 + 2 * 2645 * 8
            + storage["compiled_race_array_bytes"]
        )


def test_projected_bilinear_replay_preserves_expanded_eligibility_after_ingestion() -> None:
    memory = _projected_bilinear_memory()
    original = _protection_pair(memory)
    memory.observe_comparison(*original)
    projection, covariance = memory.projection_summary, memory.covariance_summary
    # Enough added rows to exercise a different candidate population, without
    # a model download or a project TRAIN/fresh fixture.
    memory.ingest((OpaqueDocument(b"later", b"synthetic", b"another gamma " * 40),))
    rebound = tuple(memory.revisit_decision(row) for row in original)
    assert [row.eligibility for row in rebound] == [row.eligibility for row in original]
    assert memory.projection_summary == projection and memory.covariance_summary == covariance
    parent = memory.checkpoint_sha256
    for ablation in ("disabled", "diagonal_only"):
        decision = memory.raw_score_reference(b"request", projected_bilinear_ablation=ablation)
        assert f"_projected_bilinear_ablation_{ablation}" in decision.mode
        with pytest.raises(ValueError, match="read-only"):
            memory.observe_comparison(decision, rebound[1])
        with pytest.raises(ValueError, match="read-only"):
            memory.revisit_decision(decision)
    assert memory.checkpoint_sha256 == parent and memory.projection_summary == projection
    assert memory.covariance_summary == covariance
    receipt = memory.observe_comparison(*rebound)
    assert receipt["eligibility_delta_sha256"] == sha256(
        (np.asarray(original[1].eligibility) - original[0].eligibility).tobytes()
    ).hexdigest()
    assert memory.projection_summary == projection


def test_projected_bilinear_policy_rejects_invalid_or_cross_feature_custody() -> None:
    encoder = _SyntheticProtectionEncoder()
    for options in ({"projected_bilinear": 1}, {"projected_bilinear": True}):
        with pytest.raises(ValueError, match="projected"):
            ContextualEvidenceMemory(encoder, **options)
    plain = ContextualEvidenceMemory(
        encoder, candidate_interactions=True, interaction_gain=16.0,
        comparison_step_policy="arow", learning_enabled=False
    )
    projected = _projected_bilinear_memory()
    plain.ingest(tuple(projected._documents))
    decision = _protection_pair(projected)[0]
    with pytest.raises(ValueError, match="different retained source store"):
        plain.replay_decision(decision)
    with pytest.raises(ValueError, match="requires projected_bilinear"):
        plain.raw_score_reference(b"request", projected_bilinear_ablation="disabled")
    with pytest.raises(ValueError, match="unknown projected"):
        projected.raw_score_reference(b"request", projected_bilinear_ablation="other")
    changed = replace(decision, eligibility=(*decision.eligibility[:-1], 99.0))
    before = projected.checkpoint_sha256
    with pytest.raises(ValueError, match="custody changed"):
        projected.observe_comparison(changed, _protection_pair(projected)[1])
    assert projected.checkpoint_sha256 == before and projected.feedback_count == 0


def _full_choice_memory(*, duplicate=False, **options):
    from humanbrain.full_choice_evidence_memory import FullChoiceEvidenceMemory

    memory = FullChoiceEvidenceMemory(_SyntheticProtectionEncoder(), **options)
    docs = [OpaqueDocument(b"synthetic-one", b"synthetic-test", b"alpha beta gamma")]
    if duplicate:
        docs.append(OpaqueDocument(b"synthetic-two", b"synthetic-test", b"alpha delta"))
    memory.ingest(tuple(docs))
    memory.seal_training_pool()
    return memory


def _full_choice_lesson(memory, query=b"alpha", answer=b"alpha"):
    teacher = {
        "passed": True, "test_scope": "synthetic numerical test, not an execution claim",
        "teacher_kind": "authorized_exact_answer" if answer is not None else
        "authorized_absent_evidence_stop",
        "answer_hex": None if answer is None else answer.hex(),
        "workspace_unchanged": answer is None,
    }
    return memory.teacher_lesson(
        memory.raw_score_reference(query), answer, teacher_id="synthetic-test-teacher",
        attempt_receipt={"passed": False, "test_scope": "synthetic receipt"},
        teacher_receipt=teacher,
    )


@pytest.mark.parametrize("gold", [(0,), (0, 2), (4,)])
def test_full_choice_marginal_gradient_matches_finite_differences(gold) -> None:
    from humanbrain.full_choice_evidence_memory import full_choice_loss_gradient

    features = np.array([[1., 0., 2.], [0., 2., 1.], [1., -1., 0.], [-2., 0., 1.]])
    theta = np.array([0.2, -0.1, 0.3, -0.4])
    loss, gradient, logits = full_choice_loss_gradient(features, theta[:-1], theta[-1], gold)
    expected = np.log(np.exp(logits).sum()) - np.log(np.exp(logits[list(gold)]).sum())
    assert loss == pytest.approx(expected, abs=1e-14)
    numerical = np.empty(4)
    for i in range(4):
        positive, negative = theta.copy(), theta.copy()
        positive[i] += 1e-6
        negative[i] -= 1e-6
        numerical[i] = (
            full_choice_loss_gradient(features, positive[:-1], positive[-1], gold)[0]
            - full_choice_loss_gradient(features, negative[:-1], negative[-1], gold)[0]
        ) / 2e-6
    np.testing.assert_allclose(gradient, numerical, atol=1e-9, rtol=1e-9)
    # An unexecuted high-scoring competitor still enters the normalization.
    more = np.vstack((features, [8., -8., 8.]))
    extended_gold = (5,) if gold == (4,) else gold
    assert full_choice_loss_gradient(more, theta[:-1], theta[-1], extended_gold)[0] > loss


def test_full_choice_duplicate_credit_frozen_replay_and_exact_sgd() -> None:
    update = _full_choice_memory(duplicate=True)
    frozen = _full_choice_memory(duplicate=True, learning_enabled=False)
    assert update.checkpoint_sha256 == frozen.checkpoint_sha256
    lesson = _full_choice_lesson(update)
    json.dumps(lesson.record(), allow_nan=False)
    initial = frozen.checkpoint_arrays()
    receipt = update.observe_teacher(lesson)
    control = frozen.observe_teacher(lesson)
    assert len(receipt["gold_indices"]) == 2
    assert receipt["gradient_sha256"] == control["gradient_sha256"]
    assert receipt["features_sha256"] == control["features_sha256"]
    gradient = np.asarray(receipt["gradient"]) * receipt["clip_scale"]
    eta = update.learning_rate
    arrays = update.checkpoint_arrays()
    np.testing.assert_array_equal(arrays["readout_synapses_pa"],
                                  [-0.5 * eta * gradient[:-1], 0.5 * eta * gradient[:-1]])
    assert arrays["stop_score"][0] == -eta * gradient[-1]
    assert not arrays["pacemaker_synapses_pa"].any()
    assert receipt["sgd_update_applied"] and not receipt["native_update_applied"]
    assert control["effective_learning_rate"] == 0 and not control["sgd_update_applied"]
    for key in initial:
        np.testing.assert_array_equal(frozen.checkpoint_arrays()[key], initial[key])
    replay = update.observe_teacher(lesson)
    assert replay["lesson_sha256"] == receipt["lesson_sha256"]
    assert replay["before"] == receipt["after"]
    assert replay["gradient_sha256"] != receipt["gradient_sha256"]
    assert replay["new_execution_credit"] is False
    assert update.storage_summary["native_learning_update_count"] == 0


def test_full_choice_custody_stop_authorization_growth_and_legacy_rejection() -> None:
    memory = _full_choice_memory()
    lesson = _full_choice_lesson(memory, b"another", None)
    before = memory.checkpoint_sha256
    for changed in (replace(lesson, answer=b"beta"), replace(lesson, teacher_id="forged")):
        with pytest.raises(ValueError, match="custody"):
            memory.observe_teacher(changed)
    assert memory.checkpoint_sha256 == before
    bad_teacher = json.loads(lesson.teacher_receipt_json)
    bad_teacher["workspace_unchanged"] = False
    with pytest.raises(ValueError, match="explicitly authorize"):
        memory.teacher_lesson(
            memory.raw_score_reference(b"another"), None, teacher_id="test",
            attempt_receipt={"passed": True}, teacher_receipt=bad_teacher,
        )
    with pytest.raises(ValueError, match="no occurrence"):
        _full_choice_lesson(memory, answer=b"not present")
    receipt = memory.observe_teacher(lesson)
    assert receipt["gold_indices"] == [memory.storage_summary["candidate_count"]]
    assert memory.raw_score_reference(b"another").stopped
    with pytest.raises(ValueError, match="current original action checkpoint"):
        memory.teacher_lesson(
            lesson.original_decision, None, teacher_id="test",
            attempt_receipt={"passed": False},
            teacher_receipt=json.loads(lesson.teacher_receipt_json),
        )
    for method in (memory.observe_comparison, memory.observe_feedback, memory.revisit_decision):
        with pytest.raises(ValueError, match="authenticated"):
            method(lesson.original_decision)
    weights = memory.synapse_sha256
    memory.ingest((OpaqueDocument(b"growth", b"synthetic-test", b"epsilon zeta"),))
    assert memory.synapse_sha256 == weights
    with pytest.raises(ValueError, match="unchanged sealed"):
        memory.observe_teacher(lesson)
    with pytest.raises(ValueError, match="cannot be replaced"):
        memory.seal_training_pool()
    memory.respond(b"another")  # Source growth remains valid read-only inference.
    assert memory.synapse_sha256 == weights


def test_full_choice_fixed_synthetic_policy_acquires_supported_and_stop_queries() -> None:
    memory = _full_choice_memory()
    frozen = _full_choice_memory(learning_enabled=False)
    cases = [(word + b" request" + str(i).encode(), word)
             for i in range(16) for word in (b"alpha", b"beta", b"gamma")]
    cases += [(b"another request" + str(i).encode(), None) for i in range(16)]
    previous = None
    for _ in range(4):
        for query, answer in cases:
            lesson = _full_choice_lesson(memory, query, answer)
            memory.observe_teacher(lesson)
            frozen.observe_teacher(lesson)
            if previous is not None:
                memory.observe_teacher(previous)
                frozen.observe_teacher(previous)
            previous = lesson
    assert memory.feedback_count == frozen.feedback_count == 511
    selections = [memory.raw_score_reference(query) for query, _ in cases]
    assert sum(
        d.emitted == answer for d, (_, answer) in zip(selections[:48], cases[:48], strict=True)
    ) >= 44
    assert sum(d.stopped for d in selections[48:]) >= 15
    parent = memory.checkpoint_sha256
    for decision, (query, _) in zip(selections, cases, strict=True):
        assert memory.respond_affine_interpolated(query).candidate_index == decision.candidate_index
    assert memory.checkpoint_sha256 == parent
    assert frozen.update_count == 0 and not frozen.checkpoint_arrays()["stop_score"].any()


def test_full_choice_compact_audit_export_preserves_input_custody() -> None:
    memory = _full_choice_memory()
    queries = (b"alpha", b"another request")
    parent = memory.checkpoint_sha256
    basis = memory.export_audit_basis(queries)
    arrays, meta = basis["arrays"], basis["metadata"]
    assert len(arrays["candidate_vectors"]) == memory.storage_summary["candidate_count"]
    assert arrays["query_sentence_vectors"].shape == (2, 384)
    assert arrays["query_word_vectors"].shape == (3, 384)
    assert [bytes.fromhex(q["query_hex"]) for q in meta["queries"]] == list(queries)
    assert meta["source_pool"] == memory.source_pool
    assert meta["readout_policy"] == memory.readout_policy
    for array in arrays.values():
        assert array.ndim <= 2 and array.dtype != object
    json.dumps(meta, allow_nan=False)
    arrays["candidate_vectors"][:] = 999
    assert memory.checkpoint_sha256 == parent
    fresh = memory.export_audit_basis(queries)
    assert not np.any(fresh["arrays"]["candidate_vectors"] == 999)


def test_full_choice_empty_stop_noop_and_nonfinite_rejection() -> None:
    from humanbrain.full_choice_evidence_memory import (
        FullChoiceEvidenceMemory,
        full_choice_loss_gradient,
    )

    memory = FullChoiceEvidenceMemory(_SyntheticProtectionEncoder())
    memory.seal_training_pool()
    initial = memory.synapse_sha256
    receipt = memory.observe_teacher(_full_choice_lesson(memory, b"another", None))
    assert receipt["loss_before"] == receipt["loss_after"] == 0
    assert receipt["gradient_norm"] == 0 and receipt["effective_learning_rate"] == 0
    assert memory.synapse_sha256 == initial and memory.update_count == 0
    assert memory.respond(b"another").stopped and memory.raw_score_reference(b"another").stopped
    with pytest.raises(ValueError, match="finite"):
        full_choice_loss_gradient(np.array([[np.nan]]), np.zeros(1), 0., (0,))
    with pytest.raises(ValueError, match="nonempty unique subset"):
        full_choice_loss_gradient(np.ones((1, 1)), np.zeros(1), 0., ())
