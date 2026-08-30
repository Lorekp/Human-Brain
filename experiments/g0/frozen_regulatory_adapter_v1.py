"""Parameter-free raw-event adapter for the G0 regulatory domain.

This module intentionally has no generator, evaluator, graph, or learner
dependency.  It serializes only current opaque entity-token/binary-state pairs,
raw clamp actions, time, observed next states, boundaries, and raw query slots.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, fields
from hashlib import sha256
from typing import TypeAlias

ADAPTER_SCOPE = "FROZEN_G0_REGULATORY_ADAPTER_V1"
ADAPTER_STATUS = "SOURCE_FROZEN_NO_CANDIDATE_OR_COHORT"
EXPECTED_G0_ADAPTER_CONTRACT_SHA256 = (
    "29a6dcec69d89c74c373e7f4bf8709e5e57d7cc06d3aa15f8ad3a2691db1fd99"
)

ENTITY_COUNT = 8
ACTION_NOOP = 0
ACTION_CLAMP = 1
ANSWER_SLOT_BINARY = 0

_MAGIC = b"G0EV1\0"
_OBSERVATION = 1
_ACTION = 2
_OUTCOME = 3
_BOUNDARY = 4
_QUERY = 5

_FORBIDDEN_FIELDS = frozenset(
    {
        "adjacency",
        "edge_signs",
        "update_equations",
        "ancestors",
        "reachability",
        "topological_order",
        "causal_effects",
        "counterfactual_twins",
        "graph_identity",
        "motif_identity",
        "gold_answers",
    }
)


def _validate_token(value: int, label: str) -> None:
    if type(value) is not int or value < 0 or value > 0xFFFF_FFFF:
        raise ValueError(f"{label} must be one raw uint32 token")


def _validate_time(value: int) -> None:
    if type(value) is not int or value < 0 or value > 0xFFFF_FFFF:
        raise ValueError("time must be one raw uint32 value")


def _validate_entity_states(rows: tuple[tuple[int, int], ...]) -> None:
    if len(rows) != ENTITY_COUNT:
        raise ValueError("regulatory observation requires exactly eight entity states")
    tokens: list[int] = []
    for token, value in rows:
        _validate_token(token, "entity token")
        if value not in (0, 1):
            raise ValueError("entity state must be binary")
        tokens.append(token)
    if len(set(tokens)) != ENTITY_COUNT:
        raise ValueError("entity tokens must be distinct")


@dataclass(frozen=True, slots=True)
class ObservationEvent:
    entity_states: tuple[tuple[int, int], ...]
    time: int

    def __post_init__(self) -> None:
        _validate_entity_states(self.entity_states)
        _validate_time(self.time)


@dataclass(frozen=True, slots=True)
class ActionEvent:
    action_id: int
    entity_token: int | None = None
    value: int | None = None

    def __post_init__(self) -> None:
        if self.action_id == ACTION_NOOP:
            if self.entity_token is not None or self.value is not None:
                raise ValueError("raw no-op action cannot carry operands")
            return
        if self.action_id != ACTION_CLAMP:
            raise ValueError("unknown raw regulatory action id")
        if self.entity_token is None:
            raise ValueError("raw clamp action requires an entity token")
        _validate_token(self.entity_token, "action entity token")
        if self.value not in (0, 1):
            raise ValueError("raw clamp action requires a binary value")


@dataclass(frozen=True, slots=True)
class OutcomeEvent:
    entity_states: tuple[tuple[int, int], ...]
    time: int

    def __post_init__(self) -> None:
        _validate_entity_states(self.entity_states)
        _validate_time(self.time)


@dataclass(frozen=True, slots=True)
class BoundaryEvent:
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ValueError("boundary flags must be exact booleans")


@dataclass(frozen=True, slots=True)
class QueryEvent:
    target_token: int
    answer_slot_token: int

    def __post_init__(self) -> None:
        _validate_token(self.target_token, "query target token")
        _validate_token(self.answer_slot_token, "answer slot token")


RegulatoryEvent: TypeAlias = (  # noqa: UP040 -- runtime is Python 3.11.
    ObservationEvent | ActionEvent | OutcomeEvent | BoundaryEvent | QueryEvent
)


def _state_payload(rows: tuple[tuple[int, int], ...], time: int) -> bytes:
    ordered = tuple(sorted(rows))
    payload = bytearray(struct.pack("<IBB", time, len(ordered), 5))
    for token, value in ordered:
        payload.extend(struct.pack("<IB", token, value))
    return bytes(payload)


def serialize_event(event: RegulatoryEvent) -> bytes:
    """Serialize one permitted event with no fitted state or semantic lookup."""

    if type(event) is ObservationEvent:
        tag = _OBSERVATION
        payload = _state_payload(event.entity_states, event.time)
    elif type(event) is ActionEvent:
        tag = _ACTION
        if event.action_id == ACTION_NOOP:
            payload = struct.pack("<BB", ACTION_NOOP, 0)
        else:
            if event.entity_token is None or event.value is None:
                raise RuntimeError("validated clamp action lost its operands")
            payload = struct.pack("<BBIB", ACTION_CLAMP, 1, event.entity_token, event.value)
    elif type(event) is OutcomeEvent:
        tag = _OUTCOME
        payload = _state_payload(event.entity_states, event.time)
    elif type(event) is BoundaryEvent:
        tag = _BOUNDARY
        payload = struct.pack("<BB", int(event.terminated), int(event.truncated))
    elif type(event) is QueryEvent:
        tag = _QUERY
        payload = struct.pack("<II", event.target_token, event.answer_slot_token)
    else:
        raise TypeError("unsupported regulatory event type")
    return _MAGIC + struct.pack("<BI", tag, len(payload)) + payload


def _decode_state_payload(payload: bytes) -> tuple[tuple[tuple[int, int], ...], int]:
    if len(payload) < 6:
        raise ValueError("truncated regulatory state payload")
    time, count, width = struct.unpack_from("<IBB", payload)
    if count != ENTITY_COUNT or width != 5 or len(payload) != 6 + count * width:
        raise ValueError("regulatory state shape changed")
    rows = tuple(struct.unpack_from("<IB", payload, 6 + offset * width) for offset in range(count))
    frozen = tuple((int(token), int(value)) for token, value in rows)
    return frozen, int(time)


def deserialize_event(frame: bytes) -> RegulatoryEvent:
    """Byte-exact inverse for permitted fields only."""

    header_size = len(_MAGIC) + 5
    if len(frame) < header_size or frame[: len(_MAGIC)] != _MAGIC:
        raise ValueError("invalid regulatory event frame")
    tag, size = struct.unpack_from("<BI", frame, len(_MAGIC))
    payload = frame[header_size:]
    if len(payload) != size:
        raise ValueError("regulatory event payload length changed")
    if tag == _OBSERVATION:
        rows, time = _decode_state_payload(payload)
        return ObservationEvent(rows, time)
    if tag == _ACTION:
        if payload == struct.pack("<BB", ACTION_NOOP, 0):
            return ActionEvent(ACTION_NOOP)
        if len(payload) != 7:
            raise ValueError("invalid raw clamp action payload")
        action_id, present, token, value = struct.unpack("<BBIB", payload)
        if present != 1:
            raise ValueError("raw clamp operand marker changed")
        return ActionEvent(int(action_id), int(token), int(value))
    if tag == _OUTCOME:
        rows, time = _decode_state_payload(payload)
        return OutcomeEvent(rows, time)
    if tag == _BOUNDARY:
        if len(payload) != 2:
            raise ValueError("invalid boundary payload")
        terminated, truncated = struct.unpack("<BB", payload)
        if terminated not in (0, 1) or truncated not in (0, 1):
            raise ValueError("boundary payload is not binary")
        return BoundaryEvent(bool(terminated), bool(truncated))
    if tag == _QUERY:
        if len(payload) != 8:
            raise ValueError("invalid query payload")
        target, slot = struct.unpack("<II", payload)
        return QueryEvent(int(target), int(slot))
    raise ValueError("unknown regulatory event tag")


def serialize_transcript(events: tuple[RegulatoryEvent, ...]) -> bytes:
    transcript = bytearray(b"G0TR1\0" + struct.pack("<I", len(events)))
    for event in events:
        frame = serialize_event(event)
        transcript.extend(struct.pack("<I", len(frame)))
        transcript.extend(frame)
    return bytes(transcript)


def transcript_sha256(events: tuple[RegulatoryEvent, ...]) -> str:
    return sha256(serialize_transcript(events)).hexdigest()


def negative_field_audit() -> dict[str, object]:
    """Prove that the public event schema contains no forbidden gold field."""

    event_types = (ObservationEvent, ActionEvent, OutcomeEvent, BoundaryEvent, QueryEvent)
    present = frozenset(field.name for event_type in event_types for field in fields(event_type))
    overlap = tuple(sorted(present & _FORBIDDEN_FIELDS))
    return {
        "event_field_names": tuple(sorted(present)),
        "forbidden_field_names": tuple(sorted(_FORBIDDEN_FIELDS)),
        "forbidden_overlap": overlap,
        "clean": not overlap,
    }


ADAPTER_RECORD = {
    "scope": ADAPTER_SCOPE,
    "status": ADAPTER_STATUS,
    "g0_adapter_contract_sha256": EXPECTED_G0_ADAPTER_CONTRACT_SHA256,
    "event_magic_hex": _MAGIC.hex(),
    "event_tags": (_OBSERVATION, _ACTION, _OUTCOME, _BOUNDARY, _QUERY),
    "observation_tensor_shape": (ENTITY_COUNT, 5),
    "actions": (ACTION_NOOP, ACTION_CLAMP),
    "answer_slots": (ANSWER_SLOT_BINARY,),
    "deterministic": True,
    "fitted_parameter_count": 0,
    "generator_dependency_count": 0,
    "evaluator_dependency_count": 0,
    "learner_dependency_count": 0,
    "forbidden_field_audit": negative_field_audit(),
}
ADAPTER_SHA256 = sha256(
    b"frozen-g0-regulatory-adapter-v1\0" + repr(ADAPTER_RECORD).encode("utf-8")
).hexdigest()


__all__ = [
    "ACTION_CLAMP",
    "ACTION_NOOP",
    "ADAPTER_RECORD",
    "ADAPTER_SCOPE",
    "ADAPTER_SHA256",
    "ADAPTER_STATUS",
    "ANSWER_SLOT_BINARY",
    "EXPECTED_G0_ADAPTER_CONTRACT_SHA256",
    "ActionEvent",
    "BoundaryEvent",
    "ObservationEvent",
    "OutcomeEvent",
    "QueryEvent",
    "RegulatoryEvent",
    "deserialize_event",
    "negative_field_audit",
    "serialize_event",
    "serialize_transcript",
    "transcript_sha256",
]
