"""Vectorized physical candidate selection for the frozen FIT prompt producer.

This is a speed-only successor primitive.  It preserves the producer's exact
normalization, candidate construction, fitted GLIF response, and event order,
but evaluates the independent output cells in one compiled vector call.
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


def select_source_events_batch(
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
    *,
    cell_gain: float = 1.0,
) -> BatchSourceEventSelection:
    """Evaluate the exact neutral candidate surface in one fitted-GLIF batch."""

    if isinstance(cell_gain, bool) or not isinstance(cell_gain, (int, float)):
        raise TypeError("cell_gain must be a finite nonnegative number")
    gain = float(cell_gain)
    if not isfinite(gain) or gain < 0.0:
        raise ValueError("cell_gain must be a finite nonnegative number")

    producer.STATIC_SELECTOR.authenticate()
    assemblies, literal_cells = producer._normalize_prompt_records(
        span_records,
        literal_records,
        signature_role_record,
    )
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
    producer.STATIC_SELECTOR.authenticate()
    return BatchSourceEventSelection(
        candidates=candidates,
        selected=selected,
        candidate_event_count=len(candidates),
        selected_event_count=len(selected),
        fitted_glif_cell_steps=len(candidates) * producer._OUTPUT_STEPS,
        compiled_kernel_calls=1,
    )


__all__ = ["BatchSourceEventSelection", "select_source_events_batch"]
