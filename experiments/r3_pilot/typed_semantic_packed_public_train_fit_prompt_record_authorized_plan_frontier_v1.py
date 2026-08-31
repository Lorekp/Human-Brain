"""Actual prompt records to a private desired-free packed-plan frontier.

The build seam accepts only physical prompt records and ordinary typed-frontier
inputs.  It authenticates the banked prompt-event producer result, authenticates
the resulting static occurrence-factor certificate, and only then delegates to
the banked private relay-to-plan converter.  Desired outputs enter solely through
``select``.

This is a same-FIT integration frontier.  The learned prompt assemblies are
finite-cohort cues; unseen-prompt induction and canonical panel selection remain
outside this module's claim.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from typing import NamedTuple

from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_static_checkpoint_capsule_v1 as occurrence_static,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_source_event_producer_v1 as prompt_producer,
)
from experiments.r3_pilot import (
    typed_semantic_packed_public_train_fit_occurrence_factor_authorized_plan_slice_v1 as packed_converter,  # noqa: E501
)
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "PACKED_PUBLIC_TRAIN_FIT_PROMPT_RECORD_AUTHORIZED_PLAN_FRONTIER_V1"
READY = packed_converter.READY
SELECTED = packed_converter.SELECTED
STOP_SOURCE_EVENT_CERTIFICATE = (
    "STOP_PACKED_PUBLIC_TRAIN_FIT_PROMPT_RECORD_AUTHORIZED_PLAN_FRONTIER_V1_"
    "SOURCE_EVENT_CERTIFICATE"
)
STOP_OCCURRENCE_CERTIFICATE = (
    "STOP_PACKED_PUBLIC_TRAIN_FIT_PROMPT_RECORD_AUTHORIZED_PLAN_FRONTIER_V1_"
    "OCCURRENCE_CERTIFICATE"
)

DESIRED_BUILD_INPUT_COUNT = 0
CANONICAL_PANEL_SPEND_COUNT = 0
PROTECTED_INPUT_COUNT = 0

_NAMESPACE = b"packed-public-train-fit-prompt-record-plan-frontier-v1\0"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(label: str, value: object) -> str:
    result = sha256()
    result.update(_NAMESPACE)
    result.update(label.encode("utf-8"))
    result.update(b"\0")
    result.update(_canonical_json(value))
    return result.hexdigest()


class PromptRecordAuthorizedBuildState(NamedTuple):
    status: str
    stop_reason: str | None
    source_event_certificate_sha256: str | None
    source_event_status: str | None
    source_candidate_event_count: int
    source_event_count: int
    occurrence_certificate_sha256: str | None
    occurrence_status: str | None
    occurrence_view_sha256: str | None
    occurrence_count: int
    producer_build_attempts: int
    occurrence_inference_attempts: int
    plan_build_attempts: int
    delegated_build_state_sha256: str | None
    desired_reads_before_select: int
    proof_witnesses_before_select: int
    sha256: str


class PromptRecordAuthorizedSelection(NamedTuple):
    status: str
    build_state_sha256: str
    delegated_selection_sha256: str | None
    selected_eclass_id: str | None
    exact_proof_count: int
    witnesses: tuple[object, ...]
    selected_proof_forest_sha256: str | None
    work: object | None
    sha256: str


def _state_payload(state: PromptRecordAuthorizedBuildState) -> dict[str, object]:
    return {
        field: getattr(state, field)
        for field in state._fields
        if field != "sha256"
    }


def _new_state(**values: object) -> PromptRecordAuthorizedBuildState:
    partial = PromptRecordAuthorizedBuildState(**values, sha256="")  # type: ignore[arg-type]
    return partial._replace(sha256=_digest("build-state", _state_payload(partial)))


def _source_sha(certificate: object | None) -> str | None:
    if type(certificate) is prompt_producer.PromptRecordSourceEventCertificate:
        return certificate.sha256
    return None


def _source_status(certificate: object | None) -> str | None:
    if type(certificate) is prompt_producer.PromptRecordSourceEventCertificate:
        return certificate.status
    return None


def _source_candidate_count(certificate: object | None) -> int:
    if type(certificate) is prompt_producer.PromptRecordSourceEventCertificate:
        return certificate.candidate_event_count
    return 0


def _occurrence_sha(certificate: object | None) -> str | None:
    if type(certificate) is occurrence_static.OccurrenceFactorStaticCertificate:
        return certificate.sha256
    return None


def _occurrence_status(certificate: object | None) -> str | None:
    if type(certificate) is occurrence_static.OccurrenceFactorStaticCertificate:
        return certificate.status
    return None


_ISSUED_FRONTIERS: dict[int, object] = {}


class PackedPublicTrainFitPromptRecordAuthorizedPlanFrontierV1:
    """Deep actual-record build -> desired-only select module."""

    __slots__ = (
        "_events",
        "_frontier",
        "_occurrence_certificate",
        "_source_certificate",
        "_state",
    )

    def __init__(self) -> None:
        raise TypeError(
            "use PackedPublicTrainFitPromptRecordAuthorizedPlanFrontierV1.build"
        )

    @classmethod
    def _issue(
        cls,
        state: PromptRecordAuthorizedBuildState,
        *,
        source_certificate: object | None,
        events: tuple[object, ...],
        occurrence_certificate: object | None,
        frontier: packed_converter.PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1
        | None,
    ) -> PackedPublicTrainFitPromptRecordAuthorizedPlanFrontierV1:
        instance = object.__new__(cls)
        instance._source_certificate = source_certificate
        instance._events = events
        instance._occurrence_certificate = occurrence_certificate
        instance._frontier = frontier
        instance._state = state
        _ISSUED_FRONTIERS[id(instance)] = instance
        return instance

    @classmethod
    def _stopped(
        cls,
        status: str,
        reason: str,
        *,
        source_certificate: object | None = None,
        events: tuple[object, ...] = (),
        occurrence_certificate: object | None = None,
        producer_build_attempts: int = 1,
        occurrence_inference_attempts: int = 0,
    ) -> PackedPublicTrainFitPromptRecordAuthorizedPlanFrontierV1:
        state = _new_state(
            status=status,
            stop_reason=reason,
            source_event_certificate_sha256=_source_sha(source_certificate),
            source_event_status=_source_status(source_certificate),
            source_candidate_event_count=_source_candidate_count(source_certificate),
            source_event_count=len(events),
            occurrence_certificate_sha256=_occurrence_sha(occurrence_certificate),
            occurrence_status=_occurrence_status(occurrence_certificate),
            occurrence_view_sha256=None,
            occurrence_count=0,
            producer_build_attempts=producer_build_attempts,
            occurrence_inference_attempts=occurrence_inference_attempts,
            plan_build_attempts=0,
            delegated_build_state_sha256=None,
            desired_reads_before_select=0,
            proof_witnesses_before_select=0,
        )
        return cls._issue(
            state,
            source_certificate=source_certificate,
            events=events,
            occurrence_certificate=occurrence_certificate,
            frontier=None,
        )

    @classmethod
    def build(
        cls,
        span_records: Sequence[Mapping[str, object]],
        literal_records: Sequence[Mapping[str, object]],
        signature_role_record: Mapping[str, object],
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_cost: int,
    ) -> PackedPublicTrainFitPromptRecordAuthorizedPlanFrontierV1:
        """Build without accepting or inspecting desired outputs."""

        try:
            source_certificate = prompt_producer.produce_source_event_bundle(
                span_records,
                literal_records,
                signature_role_record,
            )
        except (TypeError, ValueError) as error:
            return cls._stopped(
                STOP_SOURCE_EVENT_CERTIFICATE,
                str(error),
            )

        try:
            events = prompt_producer.require_source_event_bundle(source_certificate)
        except TypeError as error:
            return cls._stopped(
                STOP_SOURCE_EVENT_CERTIFICATE,
                str(error),
                source_certificate=source_certificate,
            )
        except RuntimeError as error:
            if (
                type(source_certificate)
                is prompt_producer.PromptRecordSourceEventCertificate
                and source_certificate.status != prompt_producer.READY
            ):
                return cls._stopped(
                    STOP_SOURCE_EVENT_CERTIFICATE,
                    str(error),
                    source_certificate=source_certificate,
                )
            raise

        occurrence_certificate = occurrence_static.infer_occurrence_factors(events)
        try:
            occurrence_view = occurrence_static.require_occurrence_factors(
                occurrence_certificate
            )
        except TypeError as error:
            return cls._stopped(
                STOP_OCCURRENCE_CERTIFICATE,
                str(error),
                source_certificate=source_certificate,
                events=events,
                occurrence_inference_attempts=1,
            )
        except RuntimeError as error:
            if (
                type(occurrence_certificate)
                is occurrence_static.OccurrenceFactorStaticCertificate
                and occurrence_certificate.status != occurrence_static.READY
            ):
                return cls._stopped(
                    STOP_OCCURRENCE_CERTIFICATE,
                    str(error),
                    source_certificate=source_certificate,
                    events=events,
                    occurrence_certificate=occurrence_certificate,
                    occurrence_inference_attempts=1,
                )
            raise

        frontier = (
            packed_converter.PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1.build(
                signature,
                cases,
                literals,
                max_cost,
                occurrence_certificate,
            )
        )
        if (
            frontier.state.desired_reads_before_select != 0
            or frontier.state.proof_witnesses_before_select != 0
        ):
            raise RuntimeError("delegated build consumed desired or proof evidence")
        state = _new_state(
            status=frontier.status,
            stop_reason=frontier.state.stop_reason,
            source_event_certificate_sha256=source_certificate.sha256,
            source_event_status=source_certificate.status,
            source_candidate_event_count=source_certificate.candidate_event_count,
            source_event_count=len(events),
            occurrence_certificate_sha256=occurrence_certificate.sha256,
            occurrence_status=occurrence_certificate.status,
            occurrence_view_sha256=occurrence_view.sha256,
            occurrence_count=len(occurrence_view.occurrence_carriers),
            producer_build_attempts=1,
            occurrence_inference_attempts=1,
            plan_build_attempts=frontier.state.plan_build_attempts,
            delegated_build_state_sha256=frontier.sha256,
            desired_reads_before_select=frontier.state.desired_reads_before_select,
            proof_witnesses_before_select=(
                frontier.state.proof_witnesses_before_select
            ),
        )
        return cls._issue(
            state,
            source_certificate=source_certificate,
            events=events,
            occurrence_certificate=occurrence_certificate,
            frontier=frontier,
        )

    @property
    def state(self) -> PromptRecordAuthorizedBuildState:
        return self._state

    @property
    def status(self) -> str:
        return self._state.status

    @property
    def sha256(self) -> str:
        return self._state.sha256

    @property
    def plan_slice_receipt(self) -> object | None:
        return (
            None if self._frontier is None else self._frontier.plan_slice_receipt
        )

    @property
    def work(self) -> object | None:
        return None if self._frontier is None else self._frontier.work

    def _authenticate(self) -> None:
        if _ISSUED_FRONTIERS.get(id(self)) is not self:
            raise TypeError("prompt-record plan frontier is not an issued exact object")
        if self._state.sha256 != _digest(
            "build-state", _state_payload(self._state)
        ):
            raise RuntimeError("prompt-record plan frontier state changed")
        if self._events and (
            type(self._source_certificate)
            is not prompt_producer.PromptRecordSourceEventCertificate
            or prompt_producer.require_source_event_bundle(self._source_certificate)
            != self._events
        ):
            raise RuntimeError("prompt-record source-event binding changed")
        if self._occurrence_certificate is not None:
            if (
                type(self._occurrence_certificate)
                is not occurrence_static.OccurrenceFactorStaticCertificate
            ):
                raise RuntimeError("static occurrence certificate binding changed")
            view = occurrence_static.require_occurrence_factors(
                self._occurrence_certificate
            )
            if view.sha256 != self._state.occurrence_view_sha256:
                raise RuntimeError("static occurrence view binding changed")
        if (
            self._frontier is not None
            and self._frontier.sha256 != self._state.delegated_build_state_sha256
        ):
            raise RuntimeError("delegated packed-plan frontier binding changed")

    def _selection(
        self,
        status: str,
        delegated: packed_converter.PublicTrainFitOccurrenceAuthorizedSelection
        | None = None,
    ) -> PromptRecordAuthorizedSelection:
        witnesses = () if delegated is None else delegated.witnesses
        work = None if delegated is None else delegated.work
        payload = {
            "status": status,
            "build_state_sha256": self._state.sha256,
            "delegated_selection_sha256": (
                None if delegated is None else delegated.sha256
            ),
            "selected_eclass_id": (
                None if delegated is None else delegated.selected_eclass_id
            ),
            "exact_proof_count": 0 if delegated is None else delegated.exact_proof_count,
            "witness_sha256s": tuple(
                getattr(witness, "sha256", None) for witness in witnesses
            ),
            "selected_proof_forest_sha256": (
                None if delegated is None else delegated.selected_proof_forest_sha256
            ),
            "work_sha256": None if work is None else getattr(work, "sha256", None),
        }
        return PromptRecordAuthorizedSelection(
            status,
            self._state.sha256,
            None if delegated is None else delegated.sha256,
            None if delegated is None else delegated.selected_eclass_id,
            0 if delegated is None else delegated.exact_proof_count,
            witnesses,
            None if delegated is None else delegated.selected_proof_forest_sha256,
            work,
            _digest("selection", payload),
        )

    def select(self, desired: Sequence[object]) -> PromptRecordAuthorizedSelection:
        """Consume desired outputs only after the desired-free build is frozen."""

        self._authenticate()
        if self._frontier is None:
            return self._selection(self._state.status)
        delegated = self._frontier.select(desired)
        self._authenticate()
        return self._selection(delegated.status, delegated)


CLAIM_BOUNDARY = (
    "public-TRAIN same-FIT actual prompt-record to learned neutral source events, "
    "target-free static occurrence factors, and private packed-plan frontier only; "
    "learned assemblies remain finite-cohort cues, while unseen prompt transfer, "
    "canonical panel selection, protected evaluation, coding benchmarks, and broad "
    "capability remain unproven"
)


__all__ = [
    "CANONICAL_PANEL_SPEND_COUNT",
    "CLAIM_BOUNDARY",
    "DESIRED_BUILD_INPUT_COUNT",
    "PROTECTED_INPUT_COUNT",
    "READY",
    "SCHEMA",
    "SELECTED",
    "STOP_OCCURRENCE_CERTIFICATE",
    "STOP_SOURCE_EVENT_CERTIFICATE",
    "PackedPublicTrainFitPromptRecordAuthorizedPlanFrontierV1",
    "PromptRecordAuthorizedBuildState",
    "PromptRecordAuthorizedSelection",
]
