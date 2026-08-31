from __future__ import annotations

import pytest
from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_static_checkpoint_capsule_v1 as occurrence_static,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v3_bound_execution as engine,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_source_event_producer_v1 as producer,
)


def test_v3_source_static_core_reuses_one_nonshape_certificate_and_retains_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This is the banked split-seam self-posed assembly, not a V3/public row.
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
    selector_calls: list[float] = []
    static_calls: list[tuple[object, float]] = []
    original_selector = engine.source_static.produce_source_selection_batch
    original_static = engine.source_static.infer_static_from_source

    def observe_selector(
        span_records: object,
        literal_records: object,
        signature_record: object,
        *,
        cell_gain: float = 1.0,
    ) -> object:
        selector_calls.append(cell_gain)
        return original_selector(
            span_records,
            literal_records,
            signature_record,
            cell_gain=cell_gain,
        )

    def observe_static(certificate: object, cell_gain: float = 1.0) -> object:
        static_calls.append((certificate, cell_gain))
        return original_static(certificate, cell_gain)

    monkeypatch.setattr(
        engine.source_static, "produce_source_selection_batch", observe_selector
    )
    monkeypatch.setattr(
        engine.source_static, "infer_static_from_source", observe_static
    )
    stage = engine._run_source_static_stage(
        spans,
        (),
        signature,
        expected_candidate_count=8,
    )

    assert selector_calls == [0.0, 1.0]
    assert [gain for _certificate, gain in static_calls] == [0.0, 1.0]
    assert static_calls[0][0] is static_calls[1][0]
    assert static_calls[0][0] is stage.source_gain1_certificate
    assert stage.source_gain0_certificate.status == engine.source_static.SOURCE_ZERO
    assert stage.source_gain1_certificate.status == engine.source_static.SOURCE_SELECTED
    assert stage.selected_event_count == 1
    assert stage.selected_event_count not in (8, 9)
    assert stage.static_off_certificate.status == occurrence_static.STOP_ZERO
    assert stage.static_on_certificate.status == occurrence_static.STOP_ZERO
    assert stage.static_off_query_count == stage.static_on_query_count == 1
    assert stage.source_query_count == 16
    assert stage.source_fitted_glif_step_count == 16 * producer._OUTPUT_STEPS
    assert stage.static_query_count == 2
    assert stage.static_fitted_glif_step_count == 2 * occurrence_static.OUTPUT_STEPS
