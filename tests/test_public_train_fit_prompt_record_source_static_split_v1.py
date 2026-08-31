from __future__ import annotations

import pytest
from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_static_checkpoint_capsule_v1 as occurrence_static,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_source_event_producer_v1 as producer,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_source_static_split_v1 as split,
)


def test_split_source_selection_retains_nonshape_and_static_is_exactly_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One self-posed physical assembly from the frozen selector vocabulary has
    # exactly one learned local event.  It is not a public prompt/cohort row.
    assembly = (25, 35, 81, 104, 120, 171, 199, 313, 337, 375, 474, 499)
    spans = (
        {
            "start_token": 0,
            "stop_token": 1,
            "first_spike_step": 0,
            "assembly_cells": assembly,
            "spike_cells": assembly,
        },
    )
    signature = dict(producer._EXPECTED_SIGNATURE)
    original_infer = occurrence_static.infer_occurrence_factors

    def forbid_static(events: object) -> object:
        raise AssertionError(f"source selection reached static inference: {events!r}")

    monkeypatch.setattr(occurrence_static, "infer_occurrence_factors", forbid_static)
    zero = split.produce_source_selection_batch(
        spans,
        (),
        signature,
        cell_gain=0.0,
    )
    selected = split.produce_source_selection_batch(spans, (), signature)

    assert zero.status == split.SOURCE_ZERO
    assert zero.candidate_event_count == zero.candidate_raw_event_count == 8
    assert zero.candidate_literal_event_count == zero.selected_event_count == 0
    assert zero.selected_raw_event_count == zero.selected_literal_event_count == 0
    assert zero.event_root_sha256
    assert zero.work_root_sha256
    assert zero.selector_fitted_glif_step_count == 8 * producer._OUTPUT_STEPS
    assert zero.selector_compiled_kernel_call_count == 1
    assert zero.static_inference_call_count == 0

    assert selected.status == split.SOURCE_SELECTED
    assert selected.candidate_event_count == selected.candidate_raw_event_count == 8
    assert selected.candidate_literal_event_count == 0
    assert selected.selected_raw_event_count == selected.total_event_count == 1
    assert selected.selected_literal_event_count == 0
    assert selected.total_event_count not in (8, 9)
    assert selected.candidate_event_root_sha256 == zero.candidate_event_root_sha256
    assert selected.selected_event_root_sha256 != zero.selected_event_root_sha256
    assert selected.static_inference_call_count == 0
    selected_root_before = selected.sha256
    work_root_before = selected.work_root_sha256

    observed_gains: list[float] = []
    original_decode = occurrence_static._decode

    def observe_decode(query: object, *, cell_gain: float = 1.0) -> object:
        observed_gains.append(cell_gain)
        return original_decode(query, cell_gain=cell_gain)

    issued_static: list[occurrence_static.OccurrenceFactorStaticCertificate] = []

    def capture_static(
        events: tuple[producer.PhysicalSourceEvent, ...],
    ) -> occurrence_static.OccurrenceFactorStaticCertificate:
        certificate = original_infer(events)
        issued_static.append(certificate)
        return certificate

    monkeypatch.setattr(occurrence_static, "_decode", observe_decode)
    monkeypatch.setattr(occurrence_static, "infer_occurrence_factors", capture_static)
    static_off = split.infer_static_from_source(selected, cell_gain=0.0)
    off_gains = tuple(observed_gains)
    observed_gains.clear()
    static_on = split.infer_static_from_source(selected, cell_gain=1.0)

    assert static_off is issued_static[0]
    assert static_on is issued_static[1]
    assert static_off.status == occurrence_static.STOP_ZERO
    assert static_on.status == occurrence_static.STOP_ZERO
    assert off_gains and set(off_gains) == {0.0}
    assert observed_gains and set(observed_gains) == {1.0}
    occurrence_static._authenticate_certificate(static_off)
    occurrence_static._authenticate_certificate(static_on)
    assert selected.sha256 == selected_root_before
    assert selected.work_root_sha256 == work_root_before

    forged_source = split.SourceSelectionCertificate(selected._receipt)
    with pytest.raises(TypeError, match="issued exact object"):
        split.infer_static_from_source(forged_source)

    monkeypatch.setattr(
        occurrence_static,
        "infer_occurrence_factors",
        lambda events: object(),
    )
    with pytest.raises(TypeError, match="issued exact object"):
        split.infer_static_from_source(selected)
