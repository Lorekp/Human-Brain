from __future__ import annotations

import pytest
from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_annotation_capsule_v1 as annotation,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_source_event_batch_selector_v1 as batch,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_source_event_producer_v1 as producer,
)
from experiments.r3_pilot import (
    public_train_list_int_to_int_prompt_input_capsule_v1 as inputs,
)


def test_batch_selector_issues_exact_banked_certificates_and_zero_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = inputs.load_input_capsule()
    before_e = producer.STATIC_SELECTOR._excitatory.tobytes()
    before_h = producer.STATIC_SELECTOR._inhibitory.tobytes()

    total_candidates = 0
    total_selected = 0
    first_records = None
    for audit in annotation.CAPSULE.fit_identity_audits:
        records = state.lookup(audit.input_root_sha256).neural_records()
        if first_records is None:
            first_records = records
        scalar_certificate = producer.produce_source_event_bundle(*records)
        scalar_events = producer.require_source_event_bundle(scalar_certificate)
        vector = batch.select_source_events_batch(*records)
        batch_certificate = batch.produce_source_event_bundle_batch(*records)
        batch_events = producer.require_source_event_bundle(batch_certificate)

        assert vector.candidates == producer._candidate_events(
            *producer._normalize_prompt_records(*records)[:2]
        )
        assert vector.selected == scalar_events
        assert batch_events == scalar_events
        assert batch_certificate.sha256 == scalar_certificate.sha256
        assert batch_certificate.bundle_sha256 == scalar_certificate.bundle_sha256
        assert producer._certificate_payload(batch_certificate._receipt) == (
            producer._certificate_payload(scalar_certificate._receipt)
        )
        assert vector.candidate_event_count == scalar_certificate.candidate_event_count
        assert vector.selected_event_count == len(scalar_events)
        assert vector.fitted_glif_cell_steps == 554 * producer._OUTPUT_STEPS
        assert vector.compiled_kernel_calls == 1
        total_candidates += vector.candidate_event_count
        total_selected += vector.selected_event_count

    assert total_candidates == 4_986
    assert total_selected == 77
    assert first_records is not None
    reversed_records = (
        tuple(reversed(first_records[0])),
        tuple(reversed(first_records[1])),
        first_records[2],
    )
    assert batch.select_source_events_batch(*reversed_records).selected == (
        batch.select_source_events_batch(*first_records).selected
    )
    zero = batch.select_source_events_batch(*first_records, cell_gain=0.0)
    assert zero.candidate_event_count == 554
    assert zero.selected == ()
    assert zero.compiled_kernel_calls == 1

    def forbid_occurrence_inference(
        events: tuple[producer.PhysicalSourceEvent, ...],
    ) -> object:
        raise AssertionError(f"zero gain reached occurrence inference: {events!r}")

    monkeypatch.setattr(
        producer.occurrence_static,
        "infer_occurrence_factors",
        forbid_occurrence_inference,
    )
    zero_certificate = batch.produce_source_event_bundle_batch(
        *first_records,
        cell_gain=0.0,
    )
    zero_receipt = producer._authenticate_certificate(zero_certificate)
    assert zero_certificate.status == producer.STOP_ZERO
    assert zero_certificate.candidate_event_count == 554
    assert zero_certificate.selected_event_count == 0
    assert zero_receipt.events == ()
    assert zero_receipt.downstream_certificate is None
    assert zero_receipt.downstream_view_sha256 is None
    assert producer.STATIC_SELECTOR._excitatory.tobytes() == before_e
    assert producer.STATIC_SELECTOR._inhibitory.tobytes() == before_h
