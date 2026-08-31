"""Exact split between physical source selection and static inference.

The source operation normalizes prompt records, evaluates the banked vector
selector once, and issues an opaque certificate for every resulting event set.
It deliberately makes no occurrence-static call and imposes no retained-shape
admission rule.  The static operation accepts only that exact issued object,
reuses its selected neutral events, and returns the banked static certificate
without requiring a READY result.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import NamedTuple

from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_static_checkpoint_capsule_v1 as occurrence_static,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_source_event_batch_selector_v1 as batch_selector,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_source_event_producer_v1 as producer,
)

SCHEMA = "PUBLIC_TRAIN_FIT_PROMPT_RECORD_SOURCE_STATIC_SPLIT_V1"
SOURCE_SELECTED = f"{SCHEMA}_SOURCE_SELECTED"
SOURCE_ZERO = f"{SCHEMA}_SOURCE_ZERO"

BATCH_SELECTOR_BANK_COMMIT = "a3a42912812c51965fcfa8be7706eb002a4b6872"
PROMPT_PRODUCER_BANK_COMMIT = "1232d2ddc14501557ac8d85e8c0846a7c4a7cee2"
STATIC_BANK_COMMIT = "d2376c7dc75314ae285ffb98edeae3e704b63d49"
BATCH_SELECTOR_SOURCE_SHA256 = (
    "54ff8bcc8ef4df63104e54df8113d39473bb00eacec409c0bbf8065c526e0074"
)
PROMPT_PRODUCER_SOURCE_SHA256 = (
    "5e195bccf8c8ebbe59bfc6d89f6f27b9fc15ee6fe8a6a62cbe96f57650d04784"
)
STATIC_SOURCE_SHA256 = (
    "0af68609e46afd65e8dc50c656f7ad22919a08329045d3068f8a457813870c89"
)

NEW_CELL_COUNT = 0
REUSED_SELECTOR_OUTPUT_CELL_COUNT = 1
STATIC_INFERENCE_CALLS_DURING_SOURCE_SELECTION = 0
PUBLIC_COHORT_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
PANEL_SPEND_COUNT = 0

REPO_ROOT = Path(__file__).resolve().parents[2]
_NAMESPACE = b"public-train-fit-prompt-record-source-static-split-v1\0"
_DEPENDENCY_SHA256S = {
    "experiments/r3_pilot/public_train_fit_prompt_record_source_event_batch_selector_v1.py": (
        BATCH_SELECTOR_SOURCE_SHA256
    ),
    "experiments/r3_pilot/public_train_fit_prompt_record_source_event_producer_v1.py": (
        PROMPT_PRODUCER_SOURCE_SHA256
    ),
    "experiments/r3_pilot/public_train_fit_occurrence_factor_static_checkpoint_capsule_v1.py": (
        STATIC_SOURCE_SHA256
    ),
}


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(label: str, value: object) -> str:
    result = sha256()
    result.update(_NAMESPACE)
    result.update(label.encode("ascii"))
    result.update(b"\0")
    result.update(_canonical_json(value))
    return result.hexdigest()


DEPENDENCY_ROOT_SHA256 = _digest("dependencies", _DEPENDENCY_SHA256S)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _authenticate_dependencies() -> None:
    for relative, expected in _DEPENDENCY_SHA256S.items():
        if _file_sha256(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"banked split-source dependency changed: {relative}")
    producer.STATIC_SELECTOR.authenticate()
    if (
        batch_selector.producer is not producer
        or producer.occurrence_static is not occurrence_static
        or producer._OUTPUT_STEPS != occurrence_static.OUTPUT_STEPS
        or batch_selector.producer.STATIC_SELECTOR.checkpoint_sha256
        != producer.CHECKPOINT_SHA256
    ):
        raise RuntimeError("split source/static dependency authority changed")


PhysicalSourceEvent = producer.PhysicalSourceEvent


def _event_records(
    events: tuple[PhysicalSourceEvent, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(event.neural_record() for event in events)


def _event_root(label: str, events: tuple[PhysicalSourceEvent, ...]) -> str:
    return _digest(label, _event_records(events))


class _SourceSelectionReceipt(NamedTuple):
    authority: object
    status: str
    stop_reason: str | None
    cell_gain: float
    normalized_assemblies: tuple[tuple[int, ...], ...]
    normalized_literal_cells: tuple[tuple[int, ...], ...]
    request_sha256: str
    candidates: tuple[PhysicalSourceEvent, ...]
    selected_events: tuple[PhysicalSourceEvent, ...]
    candidate_raw_event_count: int
    candidate_literal_event_count: int
    selected_raw_event_count: int
    selected_literal_event_count: int
    candidate_event_root_sha256: str
    selected_event_root_sha256: str
    selector_fitted_glif_step_count: int
    selector_compiled_kernel_call_count: int
    static_inference_call_count: int
    physical_work_receipt_sha256: str
    sha256: str


_SOURCE_CERTIFICATE_AUTHORITY = object()
_ISSUED_SOURCE_CERTIFICATES: dict[int, SourceSelectionCertificate] = {}


class SourceSelectionCertificate:
    """Opaque exact-object authority over one normalized vector selection."""

    __slots__ = ("_receipt",)

    def __init__(self, receipt: _SourceSelectionReceipt) -> None:
        if receipt.authority is not _SOURCE_CERTIFICATE_AUTHORITY:
            raise TypeError("source-selection certificate requires private authority")
        self._receipt = receipt

    @property
    def status(self) -> str:
        return self._receipt.status

    @property
    def stop_reason(self) -> str | None:
        return self._receipt.stop_reason

    @property
    def cell_gain(self) -> float:
        return self._receipt.cell_gain

    @property
    def request_sha256(self) -> str:
        return self._receipt.request_sha256

    @property
    def candidate_event_count(self) -> int:
        return len(self._receipt.candidates)

    @property
    def candidate_raw_event_count(self) -> int:
        return self._receipt.candidate_raw_event_count

    @property
    def candidate_literal_event_count(self) -> int:
        return self._receipt.candidate_literal_event_count

    @property
    def selected_raw_event_count(self) -> int:
        return self._receipt.selected_raw_event_count

    @property
    def selected_literal_event_count(self) -> int:
        return self._receipt.selected_literal_event_count

    @property
    def selected_event_count(self) -> int:
        return len(self._receipt.selected_events)

    @property
    def total_event_count(self) -> int:
        return len(self._receipt.selected_events)

    @property
    def candidate_event_root_sha256(self) -> str:
        return self._receipt.candidate_event_root_sha256

    @property
    def selected_event_root_sha256(self) -> str:
        return self._receipt.selected_event_root_sha256

    @property
    def event_root_sha256(self) -> str:
        return self._receipt.selected_event_root_sha256

    @property
    def selector_fitted_glif_step_count(self) -> int:
        return self._receipt.selector_fitted_glif_step_count

    @property
    def selector_compiled_kernel_call_count(self) -> int:
        return self._receipt.selector_compiled_kernel_call_count

    @property
    def static_inference_call_count(self) -> int:
        return self._receipt.static_inference_call_count

    @property
    def physical_work_receipt_sha256(self) -> str:
        return self._receipt.physical_work_receipt_sha256

    @property
    def work_root_sha256(self) -> str:
        return self._receipt.physical_work_receipt_sha256

    @property
    def selector_checkpoint_sha256(self) -> str:
        return producer.CHECKPOINT_SHA256

    @property
    def sha256(self) -> str:
        return self._receipt.sha256


def _work_payload(receipt: _SourceSelectionReceipt) -> dict[str, object]:
    return {
        "request_sha256": receipt.request_sha256,
        "candidate_event_root_sha256": receipt.candidate_event_root_sha256,
        "selected_event_root_sha256": receipt.selected_event_root_sha256,
        "selector_candidate_query_count": len(receipt.candidates),
        "selector_output_cell_count": REUSED_SELECTOR_OUTPUT_CELL_COUNT,
        "selector_fitted_glif_step_count": receipt.selector_fitted_glif_step_count,
        "selector_compiled_kernel_call_count": (
            receipt.selector_compiled_kernel_call_count
        ),
        "static_inference_call_count": receipt.static_inference_call_count,
    }


def _certificate_payload(receipt: _SourceSelectionReceipt) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "status": receipt.status,
        "stop_reason": receipt.stop_reason,
        "cell_gain": receipt.cell_gain,
        "request_sha256": receipt.request_sha256,
        "candidate_event_count": len(receipt.candidates),
        "candidate_raw_event_count": receipt.candidate_raw_event_count,
        "candidate_literal_event_count": receipt.candidate_literal_event_count,
        "selected_event_count": len(receipt.selected_events),
        "selected_raw_event_count": receipt.selected_raw_event_count,
        "selected_literal_event_count": receipt.selected_literal_event_count,
        "candidate_event_root_sha256": receipt.candidate_event_root_sha256,
        "selected_event_root_sha256": receipt.selected_event_root_sha256,
        "selector_checkpoint_sha256": producer.CHECKPOINT_SHA256,
        "dependency_root_sha256": DEPENDENCY_ROOT_SHA256,
        "physical_work_receipt_sha256": receipt.physical_work_receipt_sha256,
    }


def _issue_source_selection(
    *,
    cell_gain: float,
    normalized_assemblies: tuple[tuple[int, ...], ...],
    normalized_literal_cells: tuple[tuple[int, ...], ...],
    selection: batch_selector.BatchSourceEventSelection,
) -> SourceSelectionCertificate:
    candidates = selection.candidates
    selected = selection.selected
    request_sha = _digest(
        "normalized-request",
        {
            "raw_assemblies": normalized_assemblies,
            "literal_event_cells": normalized_literal_cells,
            "signature_role_record": producer._EXPECTED_SIGNATURE,
        },
    )
    candidate_root = _event_root("normalized-candidates", candidates)
    selected_root = _event_root("selected-events", selected)
    partial = _SourceSelectionReceipt(
        _SOURCE_CERTIFICATE_AUTHORITY,
        SOURCE_SELECTED if selected else SOURCE_ZERO,
        None if selected else "no learned physical source event was selected",
        cell_gain,
        normalized_assemblies,
        normalized_literal_cells,
        request_sha,
        candidates,
        selected,
        sum(type(event) is producer.RawPhysicalSourceEvent for event in candidates),
        sum(
            type(event) is producer.LiteralPhysicalSourceEvent
            for event in candidates
        ),
        sum(type(event) is producer.RawPhysicalSourceEvent for event in selected),
        sum(
            type(event) is producer.LiteralPhysicalSourceEvent for event in selected
        ),
        candidate_root,
        selected_root,
        selection.fitted_glif_cell_steps,
        selection.compiled_kernel_calls,
        STATIC_INFERENCE_CALLS_DURING_SOURCE_SELECTION,
        "",
        "",
    )
    with_work = partial._replace(
        physical_work_receipt_sha256=_digest("source-work", _work_payload(partial))
    )
    receipt = with_work._replace(
        sha256=_digest("source-certificate", _certificate_payload(with_work))
    )
    certificate = SourceSelectionCertificate(receipt)
    _ISSUED_SOURCE_CERTIFICATES[id(certificate)] = certificate
    return certificate


def _authenticate_source_selection(
    certificate: SourceSelectionCertificate,
) -> _SourceSelectionReceipt:
    if (
        type(certificate) is not SourceSelectionCertificate
        or _ISSUED_SOURCE_CERTIFICATES.get(id(certificate)) is not certificate
        or certificate._receipt.authority is not _SOURCE_CERTIFICATE_AUTHORITY
    ):
        raise TypeError("source-selection certificate is not an issued exact object")
    _authenticate_dependencies()
    receipt = certificate._receipt
    candidates = producer._candidate_events(
        receipt.normalized_assemblies,
        receipt.normalized_literal_cells,
    )
    candidate_records = _event_records(candidates)
    selected_records = _event_records(receipt.selected_events)
    selected_record_set = {_canonical_json(record) for record in selected_records}
    exact_selected = tuple(
        event
        for event, record in zip(candidates, candidate_records, strict=True)
        if _canonical_json(record) in selected_record_set
    )
    expected_status = SOURCE_SELECTED if receipt.selected_events else SOURCE_ZERO
    expected_reason = (
        None
        if receipt.selected_events
        else "no learned physical source event was selected"
    )
    if (
        receipt.candidates != candidates
        or receipt.selected_events != exact_selected
        or len(selected_records) != len(selected_record_set)
        or receipt.status != expected_status
        or receipt.stop_reason != expected_reason
        or receipt.request_sha256
        != _digest(
            "normalized-request",
            {
                "raw_assemblies": receipt.normalized_assemblies,
                "literal_event_cells": receipt.normalized_literal_cells,
                "signature_role_record": producer._EXPECTED_SIGNATURE,
            },
        )
        or receipt.candidate_raw_event_count
        != sum(type(event) is producer.RawPhysicalSourceEvent for event in candidates)
        or receipt.candidate_literal_event_count
        != sum(
            type(event) is producer.LiteralPhysicalSourceEvent
            for event in candidates
        )
        or receipt.selected_raw_event_count
        != sum(
            type(event) is producer.RawPhysicalSourceEvent
            for event in receipt.selected_events
        )
        or receipt.selected_literal_event_count
        != sum(
            type(event) is producer.LiteralPhysicalSourceEvent
            for event in receipt.selected_events
        )
        or receipt.candidate_event_root_sha256
        != _event_root("normalized-candidates", candidates)
        or receipt.selected_event_root_sha256
        != _event_root("selected-events", receipt.selected_events)
        or receipt.selector_fitted_glif_step_count
        != len(candidates) * producer._OUTPUT_STEPS
        or receipt.selector_compiled_kernel_call_count != 1
        or receipt.static_inference_call_count != 0
        or receipt.physical_work_receipt_sha256
        != _digest("source-work", _work_payload(receipt))
        or receipt.sha256
        != _digest("source-certificate", _certificate_payload(receipt))
    ):
        raise RuntimeError("source-selection certificate receipt changed")
    return receipt


def produce_source_selection_batch(
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
    *,
    cell_gain: float = 1.0,
) -> SourceSelectionCertificate:
    """Issue one exact source-only certificate from normalized prompt records."""

    _authenticate_dependencies()
    gain = batch_selector._validated_gain(cell_gain)
    normalized = producer._normalize_prompt_records(
        span_records,
        literal_records,
        signature_role_record,
    )
    selection = batch_selector.select_source_events_batch(
        span_records,
        literal_records,
        signature_role_record,
        cell_gain=gain,
    )
    expected_candidates = producer._candidate_events(*normalized)
    if (
        selection.candidates != expected_candidates
        or selection.candidate_event_count != len(expected_candidates)
        or selection.selected_event_count != len(selection.selected)
        or selection.fitted_glif_cell_steps
        != len(expected_candidates) * producer._OUTPUT_STEPS
        or selection.compiled_kernel_calls != 1
    ):
        raise RuntimeError("vector source selection changed normalized candidates")
    certificate = _issue_source_selection(
        cell_gain=gain,
        normalized_assemblies=normalized[0],
        normalized_literal_cells=normalized[1],
        selection=selection,
    )
    _authenticate_source_selection(certificate)
    return certificate


_STATIC_GAIN_LOCK = Lock()


def infer_static_from_source(
    certificate: SourceSelectionCertificate,
    cell_gain: float = 1.0,
) -> occurrence_static.OccurrenceFactorStaticCertificate:
    """Run authentic static inference on the exact certified selected events."""

    receipt = _authenticate_source_selection(certificate)
    gain = batch_selector._validated_gain(cell_gain)
    with _STATIC_GAIN_LOCK:
        original_decode = occurrence_static._decode

        def decode_at_exact_gain(
            query: object,
            *,
            cell_gain: float = 1.0,
        ) -> object:
            del cell_gain
            return original_decode(query, cell_gain=gain)

        occurrence_static._decode = decode_at_exact_gain
        try:
            static_certificate = occurrence_static.infer_occurrence_factors(
                receipt.selected_events
            )
        finally:
            occurrence_static._decode = original_decode

    static_receipt = occurrence_static._authenticate_certificate(static_certificate)
    occurrence_static.STATIC_CHECKPOINT.authenticate()
    ordered = occurrence_static._canonical_source_bundle(receipt.selected_events)
    expected_request = occurrence_static._digest(
        "inference-request",
        tuple(event.neural_record() for event in ordered),
    )
    if static_receipt.request_sha256 != expected_request:
        raise RuntimeError("static certificate did not bind the selected source events")
    return static_certificate


CLAIM_BOUNDARY = (
    "self-posed physical prompt-record source selection and exact source/static "
    "causal separation only; selected events remain finite-checkpoint membership "
    "replay, static READY is not assumed, and no public cohort, V3 execution, "
    "unseen-prompt transfer, semantic inference, protected evaluation, or coding "
    "capability is claimed"
)


__all__ = [
    "BATCH_SELECTOR_BANK_COMMIT",
    "BATCH_SELECTOR_SOURCE_SHA256",
    "CLAIM_BOUNDARY",
    "DEPENDENCY_ROOT_SHA256",
    "NEW_CELL_COUNT",
    "PROMPT_PRODUCER_BANK_COMMIT",
    "PROMPT_PRODUCER_SOURCE_SHA256",
    "SCHEMA",
    "SOURCE_SELECTED",
    "SOURCE_ZERO",
    "STATIC_BANK_COMMIT",
    "STATIC_SOURCE_SHA256",
    "SourceSelectionCertificate",
    "infer_static_from_source",
    "produce_source_selection_batch",
]
