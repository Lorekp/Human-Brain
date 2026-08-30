"""Candidate-visible spatial serialization for the frozen G0 gate.

The adapter is deliberately a dependency leaf.  It accepts only the standard
partial image, direction, raw action, reward, and episode-boundary fields.  It
does not import a generator, environment, evaluator, protocol, or codebook.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256

ADAPTER_SCOPE = "G0_SPATIAL_CANDIDATE_ADAPTER_V1"
IMAGE_SHAPE = (7, 7, 3)
IMAGE_BYTE_COUNT = 7 * 7 * 3
DIRECTION_IDS = (0, 1, 2, 3)
RAW_ACTION_IDS = (0, 1, 2, 3, 4, 5, 6)

CANDIDATE_FIELDS = (
    "image",
    "direction",
    "action_id",
    "reward",
    "terminated",
    "truncated",
)
SPATIAL_FORBIDDEN_FIELDS = (
    "mission",
    "full-grid",
    "absolute-agent-position",
    "visibility-mask",
    "shortest-path",
    "object-tracking-identity",
    "solver-state",
    "codebook-inverse",
    "layout-or-family-identity",
    "oracle-actions",
)

_MAGIC = b"G0-SPATIAL-CAND1"
_HEADER = struct.Struct(">16sI")
_STEP_TAIL = struct.Struct(">BBdBB")
_STEP_BYTE_COUNT = IMAGE_BYTE_COUNT + _STEP_TAIL.size
_AUDIT_NAMESPACE = b"g0-spatial-candidate-field-audit-v1\0"


class CandidateSurfaceError(ValueError):
    """A candidate record did not have the exact public surface."""


class CandidateTranscriptError(ValueError):
    """A candidate transcript was not the frozen deterministic byte format."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _normalise_field_name(name: str) -> str:
    return "-".join(name.strip().lower().replace("_", "-").split())


@dataclass(frozen=True, slots=True)
class CandidateFieldAudit:
    permitted_fields: tuple[str, ...]
    checked_forbidden_fields: tuple[str, ...]
    detected_forbidden_fields: tuple[str, ...]
    unknown_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    input_field_count: int
    sha256: str


def audit_candidate_record(record: Mapping[str, object]) -> CandidateFieldAudit:
    """Reject before reading values unless the record has exactly six fields."""

    keys = tuple(record)
    if any(type(key) is not str for key in keys):
        raise CandidateSurfaceError("STOP_G0_SPATIAL_CANDIDATE_SURFACE")

    raw_key_set = set(keys)
    normalised_keys = {_normalise_field_name(key) for key in keys}
    forbidden = tuple(
        field for field in SPATIAL_FORBIDDEN_FIELDS if field in normalised_keys
    )
    unknown = tuple(sorted(raw_key_set.difference(CANDIDATE_FIELDS)))
    missing = tuple(field for field in CANDIDATE_FIELDS if field not in raw_key_set)
    if forbidden or unknown or missing or len(keys) != len(CANDIDATE_FIELDS):
        raise CandidateSurfaceError("STOP_G0_SPATIAL_CANDIDATE_SURFACE")

    audit_record = {
        "scope": ADAPTER_SCOPE,
        "permitted_fields": CANDIDATE_FIELDS,
        "checked_forbidden_fields": SPATIAL_FORBIDDEN_FIELDS,
        "detected_forbidden_fields": (),
        "unknown_fields": (),
        "missing_fields": (),
        "input_field_count": len(keys),
    }
    digest = sha256(_AUDIT_NAMESPACE + _canonical_json(audit_record)).hexdigest()
    return CandidateFieldAudit(
        permitted_fields=CANDIDATE_FIELDS,
        checked_forbidden_fields=SPATIAL_FORBIDDEN_FIELDS,
        detected_forbidden_fields=(),
        unknown_fields=(),
        missing_fields=(),
        input_field_count=len(keys),
        sha256=digest,
    )


@dataclass(frozen=True, slots=True)
class SpatialStep:
    """One opaque candidate-visible pre-action observation and raw outcome."""

    image: bytes
    direction: int
    action_id: int
    reward: float
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if type(self.image) is not bytes or len(self.image) != IMAGE_BYTE_COUNT:
            raise CandidateSurfaceError("STOP_G0_SPATIAL_IMAGE_SHAPE")
        if type(self.direction) is not int or self.direction not in DIRECTION_IDS:
            raise CandidateSurfaceError("STOP_G0_SPATIAL_DIRECTION")
        if type(self.action_id) is not int or self.action_id not in RAW_ACTION_IDS:
            raise CandidateSurfaceError("STOP_G0_SPATIAL_RAW_ACTION")
        if type(self.reward) is not float or not math.isfinite(self.reward):
            raise CandidateSurfaceError("STOP_G0_SPATIAL_REWARD")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise CandidateSurfaceError("STOP_G0_SPATIAL_BOUNDARY")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> SpatialStep:
        audit_candidate_record(record)
        reward = record["reward"]
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise CandidateSurfaceError("STOP_G0_SPATIAL_REWARD")
        return cls(
            image=record["image"],  # type: ignore[arg-type]
            direction=record["direction"],  # type: ignore[arg-type]
            action_id=record["action_id"],  # type: ignore[arg-type]
            reward=float(reward),
            terminated=record["terminated"],  # type: ignore[arg-type]
            truncated=record["truncated"],  # type: ignore[arg-type]
        )

    def record(self) -> dict[str, object]:
        return {
            "image": self.image,
            "direction": self.direction,
            "action_id": self.action_id,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


def encode_transcript(steps: Iterable[SpatialStep]) -> bytes:
    """Serialize steps with a fixed endian, shape, and record width."""

    rows = tuple(steps)
    if len(rows) > 0xFFFF_FFFF:
        raise CandidateTranscriptError("STOP_G0_SPATIAL_TRANSCRIPT_SIZE")
    payload = bytearray(_HEADER.pack(_MAGIC, len(rows)))
    for row in rows:
        if type(row) is not SpatialStep:
            raise CandidateTranscriptError("STOP_G0_SPATIAL_TRANSCRIPT_ROW")
        payload.extend(row.image)
        payload.extend(
            _STEP_TAIL.pack(
                row.direction,
                row.action_id,
                row.reward,
                int(row.terminated),
                int(row.truncated),
            )
        )
    return bytes(payload)


def decode_transcript(payload: bytes) -> tuple[SpatialStep, ...]:
    """Decode without interpreting the image codebook or action semantics."""

    if type(payload) is not bytes or len(payload) < _HEADER.size:
        raise CandidateTranscriptError("STOP_G0_SPATIAL_TRANSCRIPT_FORMAT")
    magic, row_count = _HEADER.unpack_from(payload)
    if magic != _MAGIC:
        raise CandidateTranscriptError("STOP_G0_SPATIAL_TRANSCRIPT_FORMAT")
    expected_size = _HEADER.size + row_count * _STEP_BYTE_COUNT
    if len(payload) != expected_size:
        raise CandidateTranscriptError("STOP_G0_SPATIAL_TRANSCRIPT_FORMAT")

    rows: list[SpatialStep] = []
    offset = _HEADER.size
    for _ in range(row_count):
        image = payload[offset : offset + IMAGE_BYTE_COUNT]
        offset += IMAGE_BYTE_COUNT
        direction, action_id, reward, terminated, truncated = _STEP_TAIL.unpack_from(
            payload, offset
        )
        offset += _STEP_TAIL.size
        if terminated not in (0, 1) or truncated not in (0, 1):
            raise CandidateTranscriptError("STOP_G0_SPATIAL_TRANSCRIPT_FORMAT")
        rows.append(
            SpatialStep(
                image=image,
                direction=direction,
                action_id=action_id,
                reward=reward,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )
    result = tuple(rows)
    if encode_transcript(result) != payload:
        raise CandidateTranscriptError("STOP_G0_SPATIAL_TRANSCRIPT_ROUND_TRIP")
    return result


def transcript_from_records(records: Iterable[Mapping[str, object]]) -> bytes:
    return encode_transcript(SpatialStep.from_record(record) for record in records)


def transcript_sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise CandidateTranscriptError("STOP_G0_SPATIAL_TRANSCRIPT_FORMAT")
    return sha256(payload).hexdigest()


__all__ = [
    "ADAPTER_SCOPE",
    "CANDIDATE_FIELDS",
    "DIRECTION_IDS",
    "IMAGE_BYTE_COUNT",
    "IMAGE_SHAPE",
    "RAW_ACTION_IDS",
    "SPATIAL_FORBIDDEN_FIELDS",
    "CandidateFieldAudit",
    "CandidateSurfaceError",
    "CandidateTranscriptError",
    "SpatialStep",
    "audit_candidate_record",
    "decode_transcript",
    "encode_transcript",
    "transcript_from_records",
    "transcript_sha256",
]
