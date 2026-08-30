"""Fail-closed prompt authority for the canonical TRAIN joint successor.

The static prompt V4 capsule can already issue teacher-free actual structural
winners. The static factor V4 capsule can infer factors from a
``RawSpanObservation``. Those facts do not yet compose: prompt V4 publishes no
learned direction/cue-slot/selected-slot relay that could choose the factor
carrier assembly for the same prompt. Accepting a caller-supplied factor
observation here would let the caller choose the semantic factors.

This module therefore binds the actual prompt certificate and its raw physical
span/literal records into a private receipt, then stops before factor inference.
The next mechanism must learn the missing selector-role relay and derive the
factor observation inside this authority seam. Only after that may the receipt
reach the separately frozen prompt-authorized pre-materialization plan slice.
No packed frontier, desired output, proof forest, or witness is touched here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import NamedTuple, NoReturn

from experiments.r3_pilot import (
    public_train_factor_workspace_v4_static_checkpoint_capsule_v1 as factor_checkpoint,
)
from experiments.r3_pilot import (
    public_train_prompt_selection_v4_static_checkpoint_capsule_v1 as prompt_checkpoint,
)

SCHEMA = "canonical-train-joint-prompt-authority-v1"
PROMPT_AUTHORITY_READY = "CANONICAL_TRAIN_PROMPT_AUTHORITY_V1_READY"
STOP_PROMPT_AUTHORITY = "STOP_CANONICAL_TRAIN_PROMPT_AUTHORITY_V1"
STOP_JOINT_FACTOR_SELECTOR_BINDING = (
    "STOP_CANONICAL_TRAIN_JOINT_FACTOR_SELECTOR_BINDING_V1"
)
MISSING_SELECTOR_RELAY_REASON = (
    "actual prompt winners lack a learned factor direction/cue-slot/selected-slot relay"
)
MISSING_SELECTOR_RELAY = "MISSING_LEARNED_DIRECTION_CUE_SELECTED_SLOT_RELAY"
UNRESOLVED_FACTOR_CARRIER = "UNRESOLVED_WITHOUT_AUTHORIZED_SELECTOR"

_NAMESPACE = b"canonical-train-joint-prompt-authority-v1\0"


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
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


@dataclass(frozen=True, slots=True)
class PromptSpanAuthorityRecord:
    start_token: int
    stop_token: int
    first_spike_step: int
    assembly_cells: tuple[int, ...]
    spike_cells: tuple[int, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "start_token": self.start_token,
            "stop_token": self.stop_token,
            "first_spike_step": self.first_spike_step,
            "assembly_cells": self.assembly_cells,
            "spike_cells": self.spike_cells,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class PromptLiteralAuthorityRecord:
    canonical_token_index: int
    typed_output_active_cells: tuple[int, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "canonical_token_index": self.canonical_token_index,
            "typed_output_active_cells": self.typed_output_active_cells,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class CanonicalTrainJointAuthorityState:
    prompt_status: str
    status: str
    stop_reason: str | None
    prompt_checkpoint_sha256: str
    prompt_input_sha256: str
    prompt_actual_winner_sha256: str
    prompt_certificate_sha256: str
    prompt_occurrence_sha256s: tuple[str, ...]
    span_record_sha256s: tuple[str, ...]
    literal_record_sha256s: tuple[str, ...]
    signature_role_sha256: str
    emitted_relay_count: int
    edge_binding_count: int
    factor_checkpoint_sha256: str
    factor_artifact_sha256: str
    factor_selector_relay_status: str
    factor_carrier_assembly_status: str
    factor_inference_attempts: int
    sha256: str


class _PromptAuthorityReceipt(NamedTuple):
    issuer: object
    certificate: prompt_checkpoint.PromptStructuralCertificate
    span_records: tuple[PromptSpanAuthorityRecord, ...]
    literal_records: tuple[PromptLiteralAuthorityRecord, ...]
    signature_role_record: object
    state: CanonicalTrainJointAuthorityState
    sha256: str


def _span_record(row: Mapping[str, object]) -> PromptSpanAuthorityRecord:
    required = {
        "start_token",
        "stop_token",
        "first_spike_step",
        "assembly_cells",
        "spike_cells",
    }
    if not isinstance(row, Mapping) or set(row) != required:
        raise ValueError("physical span record schema changed")
    payload = {
        "start_token": int(row["start_token"]),
        "stop_token": int(row["stop_token"]),
        "first_spike_step": int(row["first_spike_step"]),
        "assembly_cells": tuple(int(cell) for cell in row["assembly_cells"]),
        "spike_cells": tuple(int(cell) for cell in row["spike_cells"]),
    }
    return PromptSpanAuthorityRecord(
        **payload,
        sha256=_digest("prompt-span-record", payload),
    )


def _literal_record(row: Mapping[str, object]) -> PromptLiteralAuthorityRecord:
    required = {"canonical_token_index", "typed_output_active_cells"}
    if not isinstance(row, Mapping) or set(row) != required:
        raise ValueError("typed literal record schema changed")
    payload = {
        "canonical_token_index": int(row["canonical_token_index"]),
        "typed_output_active_cells": tuple(
            int(cell) for cell in row["typed_output_active_cells"]
        ),
    }
    return PromptLiteralAuthorityRecord(
        **payload,
        sha256=_digest("prompt-literal-record", payload),
    )


def _prompt_certificate_payload(
    certificate: prompt_checkpoint.PromptStructuralCertificate,
) -> dict[str, object]:
    return {
        "status": certificate.status,
        "checkpoint_sha256": certificate.checkpoint_sha256,
        "input_sha256": certificate.input_sha256,
        "occurrences": tuple(
            {
                "sequence_index": occurrence.sequence_index,
                "state_sha256": occurrence.state_sha256,
                "channel_decision_sha256": occurrence.channel_winner.decision_sha256,
                "branch_decision_sha256s": tuple(
                    winner.decision_sha256 for winner in occurrence.branch_winners
                ),
                "emitted_bindings": occurrence.emitted_bindings,
                "occurrence_binding_sha256": (
                    None
                    if occurrence.occurrence_binding is None
                    else occurrence.occurrence_binding.sha256
                ),
                "sha256": occurrence.sha256,
            }
            for occurrence in certificate.occurrences
        ),
        "terminal_decision_sha256": (
            None
            if certificate.terminal_winner is None
            else certificate.terminal_winner.decision_sha256
        ),
        "actual_winner_sha256": certificate.actual_winner_sha256,
        "certificate_sha256": certificate.sha256,
    }


def _state_payload(state: CanonicalTrainJointAuthorityState) -> dict[str, object]:
    return {
        field: getattr(state, field)
        for field in state.__dataclass_fields__
        if field != "sha256"
    }


def _receipt_payload(receipt: _PromptAuthorityReceipt) -> dict[str, object]:
    return {
        "certificate": _prompt_certificate_payload(receipt.certificate),
        "span_records": tuple(row.record() for row in receipt.span_records),
        "literal_records": tuple(row.record() for row in receipt.literal_records),
        "signature_role_record": receipt.signature_role_record,
        "state_sha256": receipt.state.sha256,
        "schema": SCHEMA,
    }


def _make_receipt_authority() -> tuple[
    Callable[..., _PromptAuthorityReceipt],
    Callable[[_PromptAuthorityReceipt], bool],
]:
    issuer = object()
    issued: dict[int, _PromptAuthorityReceipt] = {}

    def issue(
        *,
        certificate: prompt_checkpoint.PromptStructuralCertificate,
        span_records: tuple[PromptSpanAuthorityRecord, ...],
        literal_records: tuple[PromptLiteralAuthorityRecord, ...],
        signature_role_record: object,
        state: CanonicalTrainJointAuthorityState,
    ) -> _PromptAuthorityReceipt:
        partial = _PromptAuthorityReceipt(
            issuer,
            certificate,
            span_records,
            literal_records,
            signature_role_record,
            state,
            "",
        )
        receipt = partial._replace(
            sha256=_digest("prompt-authority-receipt", _receipt_payload(partial))
        )
        issued[id(receipt)] = receipt
        return receipt

    def authenticate(receipt: _PromptAuthorityReceipt) -> bool:
        return bool(
            type(receipt) is _PromptAuthorityReceipt
            and issued.get(id(receipt)) is receipt
            and receipt.issuer is issuer
            and receipt.state.sha256
            == _digest("joint-authority-state", _state_payload(receipt.state))
            and receipt.sha256
            == _digest("prompt-authority-receipt", _receipt_payload(receipt))
        )

    return issue, authenticate


_issue_prompt_authority, _authenticate_prompt_authority = _make_receipt_authority()


class CanonicalTrainJointPromptAuthorityHandoff:
    """Opaque same-input prompt capability for the missing factor-binding relay."""

    __slots__ = ("_receipt",)

    def __init__(self, receipt: _PromptAuthorityReceipt) -> None:
        if not _authenticate_prompt_authority(receipt):
            raise TypeError("prompt authority requires a private inference receipt")
        self._receipt = receipt

    @property
    def state(self) -> CanonicalTrainJointAuthorityState:
        return self._receipt.state

    @property
    def status(self) -> str:
        return self._receipt.state.status

    @property
    def sha256(self) -> str:
        return self._receipt.sha256


def _prompt_stop_reason(
    certificate: prompt_checkpoint.PromptStructuralCertificate,
) -> str | None:
    if certificate.checkpoint_sha256 != prompt_checkpoint.BANKED_CHECKPOINT_SHA256:
        raise RuntimeError("prompt authority received a different checkpoint")
    terminal = certificate.terminal_winner
    if certificate.status != prompt_checkpoint.STOP:
        return f"physical prompt inference terminated as {certificate.status}"
    if terminal is None or terminal.decision != prompt_checkpoint.STOP:
        return "physical prompt inference lacks its terminal STOP winner"
    if not certificate.occurrences:
        return "physical prompt inference emitted no structural occurrence"
    for sequence_index, occurrence in enumerate(certificate.occurrences):
        if occurrence.sequence_index != sequence_index:
            return "physical prompt occurrence sequence is not contiguous"
        channel = occurrence.channel_winner
        if channel.decision != "RELAY" or channel.relay_id is None:
            return "physical prompt channel did not publish one relay winner"
        expected_bindings = tuple(
            (winner.branch_id, winner.relay_id)
            for winner in occurrence.branch_winners
        )
        if occurrence.emitted_bindings != expected_bindings:
            return "physical prompt branch winners lost their executed bindings"
        if any(
            winner.decision not in {"RELAY", prompt_checkpoint.STOP}
            for winner in occurrence.branch_winners
        ):
            return "physical prompt branch ended in a non-authoritative race state"
    return None


def _authority_state(
    certificate: prompt_checkpoint.PromptStructuralCertificate,
    spans: tuple[PromptSpanAuthorityRecord, ...],
    literals: tuple[PromptLiteralAuthorityRecord, ...],
    signature_role_record: object,
    prompt_stop_reason: str | None,
) -> CanonicalTrainJointAuthorityState:
    prompt_ready = prompt_stop_reason is None
    partial = CanonicalTrainJointAuthorityState(
        prompt_status=(PROMPT_AUTHORITY_READY if prompt_ready else STOP_PROMPT_AUTHORITY),
        status=(
            STOP_JOINT_FACTOR_SELECTOR_BINDING
            if prompt_ready
            else STOP_PROMPT_AUTHORITY
        ),
        stop_reason=(MISSING_SELECTOR_RELAY_REASON if prompt_ready else prompt_stop_reason),
        prompt_checkpoint_sha256=certificate.checkpoint_sha256,
        prompt_input_sha256=certificate.input_sha256,
        prompt_actual_winner_sha256=certificate.actual_winner_sha256,
        prompt_certificate_sha256=certificate.sha256,
        prompt_occurrence_sha256s=tuple(row.sha256 for row in certificate.occurrences),
        span_record_sha256s=tuple(row.sha256 for row in spans),
        literal_record_sha256s=tuple(row.sha256 for row in literals),
        signature_role_sha256=_digest("signature-role-record", signature_role_record),
        emitted_relay_count=sum(
            int(occurrence.channel_winner.relay_id is not None)
            + sum(relay_id is not None for _, relay_id in occurrence.emitted_bindings)
            for occurrence in certificate.occurrences
        ),
        edge_binding_count=sum(
            row.occurrence_binding is not None for row in certificate.occurrences
        ),
        factor_checkpoint_sha256=factor_checkpoint.STATIC_CHECKPOINT.checkpoint_sha256,
        factor_artifact_sha256=factor_checkpoint.STATIC_CHECKPOINT.artifact_sha256,
        factor_selector_relay_status=MISSING_SELECTOR_RELAY,
        factor_carrier_assembly_status=UNRESOLVED_FACTOR_CARRIER,
        factor_inference_attempts=0,
        sha256="",
    )
    return CanonicalTrainJointAuthorityState(
        **_state_payload(partial),
        sha256=_digest("joint-authority-state", _state_payload(partial)),
    )


def infer_prompt_authority(
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
) -> CanonicalTrainJointPromptAuthorityHandoff:
    """Issue prompt authority, then STOP before an unbound factor observation."""

    spans = tuple(_span_record(row) for row in span_records)
    literals = tuple(_literal_record(row) for row in literal_records)
    signature_role = _plain(signature_role_record)
    certificate = prompt_checkpoint.infer_prompt(
        tuple(row.record(include_sha256=False) for row in spans),
        tuple(row.record(include_sha256=False) for row in literals),
        signature_role,
    )
    state = _authority_state(
        certificate,
        spans,
        literals,
        signature_role,
        _prompt_stop_reason(certificate),
    )
    return CanonicalTrainJointPromptAuthorityHandoff(
        _issue_prompt_authority(
            certificate=certificate,
            span_records=spans,
            literal_records=literals,
            signature_role_record=signature_role,
            state=state,
        )
    )


def _require_receipt(
    handoff: CanonicalTrainJointPromptAuthorityHandoff,
) -> _PromptAuthorityReceipt:
    if type(handoff) is not CanonicalTrainJointPromptAuthorityHandoff:
        raise TypeError("authority input must be a prompt authority handoff")
    receipt = handoff._receipt
    if not _authenticate_prompt_authority(receipt):
        raise RuntimeError("prompt authority receipt authentication failed")
    return receipt


def require_joint_authority(
    handoff: CanonicalTrainJointPromptAuthorityHandoff,
) -> NoReturn:
    """Fail closed until factor selector roles are learned from this prompt."""

    receipt = _require_receipt(handoff)
    if receipt.state.prompt_status != PROMPT_AUTHORITY_READY:
        raise RuntimeError(receipt.state.stop_reason or "prompt authority stopped")
    raise RuntimeError(MISSING_SELECTOR_RELAY_REASON)


__all__ = [
    "MISSING_SELECTOR_RELAY",
    "MISSING_SELECTOR_RELAY_REASON",
    "PROMPT_AUTHORITY_READY",
    "SCHEMA",
    "STOP_JOINT_FACTOR_SELECTOR_BINDING",
    "STOP_PROMPT_AUTHORITY",
    "UNRESOLVED_FACTOR_CARRIER",
    "CanonicalTrainJointAuthorityState",
    "CanonicalTrainJointPromptAuthorityHandoff",
    "PromptLiteralAuthorityRecord",
    "PromptSpanAuthorityRecord",
    "infer_prompt_authority",
    "require_joint_authority",
]
