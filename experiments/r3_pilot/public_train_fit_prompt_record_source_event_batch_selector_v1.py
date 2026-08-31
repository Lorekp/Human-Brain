"""Vectorized physical candidate selection for the frozen FIT prompt producer.

This is a speed-only successor primitive.  It preserves the producer's exact
normalization, candidate construction, fitted GLIF response, and event order,
but evaluates the independent output cells in one compiled vector call.  It
can also pass that exact selection through the frozen producer's occurrence
inference and private certificate authority without changing the certificate
interface consumed downstream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
from experiments.r3_pilot import (
    public_train_fit_prompt_record_source_event_producer_v1 as producer,
)


@dataclass(frozen=True, slots=True)
class BatchSourceEventSelection:
    """One exact batch evaluation of the frozen producer candidate surface."""

    candidates: tuple[producer.PhysicalSourceEvent, ...]
    selected: tuple[producer.PhysicalSourceEvent, ...]
    candidate_event_count: int
    selected_event_count: int
    fitted_glif_cell_steps: int
    compiled_kernel_calls: int


def _validated_gain(cell_gain: float) -> float:
    if isinstance(cell_gain, bool) or not isinstance(cell_gain, (int, float)):
        raise TypeError("cell_gain must be a finite nonnegative number")
    gain = float(cell_gain)
    if not isfinite(gain) or gain < 0.0:
        raise ValueError("cell_gain must be a finite nonnegative number")
    return gain


def _select_normalized_batch(
    assemblies: tuple[tuple[int, ...], ...],
    literal_cells: tuple[tuple[int, ...], ...],
    *,
    gain: float,
) -> BatchSourceEventSelection:
    candidates = producer._candidate_events(assemblies, literal_cells)
    currents = np.zeros(len(candidates), dtype=np.float64)
    for offset, event in enumerate(candidates):
        feature_index = producer._FEATURE_INDEX.get(producer._event_feature(event))
        if feature_index is None:
            continue
        score = int(producer.STATIC_SELECTOR._excitatory[0, feature_index]) - int(
            producer.STATIC_SELECTOR._inhibitory[0, feature_index]
        )
        if score >= 1:
            currents[offset] = producer._OUTPUT_CURRENT_PA * gain

    spike_mask = producer._SPIKE_KERNEL.accumulated_spike_mask(currents)
    selected = tuple(
        event for event, spiked in zip(candidates, spike_mask, strict=True) if spiked
    )
    return BatchSourceEventSelection(
        candidates=candidates,
        selected=selected,
        candidate_event_count=len(candidates),
        selected_event_count=len(selected),
        fitted_glif_cell_steps=len(candidates) * producer._OUTPUT_STEPS,
        compiled_kernel_calls=1,
    )


def select_source_events_batch(
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
    *,
    cell_gain: float = 1.0,
) -> BatchSourceEventSelection:
    """Evaluate the exact neutral candidate surface in one fitted-GLIF batch."""

    gain = _validated_gain(cell_gain)
    producer.STATIC_SELECTOR.authenticate()
    assemblies, literal_cells = producer._normalize_prompt_records(
        span_records,
        literal_records,
        signature_role_record,
    )
    result = _select_normalized_batch(assemblies, literal_cells, gain=gain)
    producer.STATIC_SELECTOR.authenticate()
    return result


def produce_source_event_bundle_batch(
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
    *,
    cell_gain: float = 1.0,
) -> producer.PromptRecordSourceEventCertificate:
    """Issue the producer's exact certificate from one vectorized selection."""

    gain = _validated_gain(cell_gain)
    producer.STATIC_SELECTOR.authenticate()
    producer.occurrence_static.STATIC_CHECKPOINT.authenticate()
    assemblies, literal_cells = producer._normalize_prompt_records(
        span_records,
        literal_records,
        signature_role_record,
    )
    request_sha = producer._digest(
        producer._REQUEST_NAMESPACE,
        {
            "raw_assemblies": assemblies,
            "literal_event_cells": literal_cells,
            "signature_role_record": producer._plain(signature_role_record),
        },
    )
    selection = _select_normalized_batch(assemblies, literal_cells, gain=gain)
    events = selection.selected
    if not events:
        return producer._issue_certificate(
            status=producer.STOP_ZERO,
            stop_reason="no learned physical source event was selected",
            request_sha256=request_sha,
            candidate_event_count=selection.candidate_event_count,
            events=(),
            downstream_certificate=None,
            downstream_view_sha256=None,
        )

    raw_count = sum(type(event) is producer.RawPhysicalSourceEvent for event in events)
    literal_count = sum(
        type(event) is producer.LiteralPhysicalSourceEvent for event in events
    )
    if raw_count not in (6, 7) or literal_count != 2 or len(events) not in (8, 9):
        return producer._issue_certificate(
            status=producer.STOP_UNSUPPORTED,
            stop_reason=(
                "selected source events are not one retained 8/9-node FIT shape"
            ),
            request_sha256=request_sha,
            candidate_event_count=selection.candidate_event_count,
            events=events,
            downstream_certificate=None,
            downstream_view_sha256=None,
        )

    downstream = producer.occurrence_static.infer_occurrence_factors(events)
    if downstream.status != producer.occurrence_static.READY:
        return producer._issue_certificate(
            status=producer.STOP_UNSUPPORTED,
            stop_reason=(
                "selected source events have no authenticated static "
                "occurrence-factor view"
            ),
            request_sha256=request_sha,
            candidate_event_count=selection.candidate_event_count,
            events=events,
            downstream_certificate=downstream,
            downstream_view_sha256=None,
        )
    view = producer.occurrence_static.require_occurrence_factors(downstream)
    producer.STATIC_SELECTOR.authenticate()
    return producer._issue_certificate(
        status=producer.READY,
        stop_reason=None,
        request_sha256=request_sha,
        candidate_event_count=selection.candidate_event_count,
        events=events,
        downstream_certificate=downstream,
        downstream_view_sha256=view.sha256,
    )


CLAIM_BOUNDARY = (
    "same-FIT exact scalar-certificate parity and selector-stage vectorization "
    "only; no V2 runner, unseen-prompt transfer, end-to-end speed, protected "
    "evaluation, or coding capability evidence"
)


__all__ = [
    "CLAIM_BOUNDARY",
    "BatchSourceEventSelection",
    "produce_source_event_bundle_batch",
    "select_source_events_batch",
]
