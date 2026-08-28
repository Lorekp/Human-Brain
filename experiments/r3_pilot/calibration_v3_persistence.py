"""Fail-closed persistence primitives for the frozen R3 calibration-v3 plan.

This module owns only artifact framing and local persistence policy.  It does
not import the native runtime, open corpus data, execute a fixture, or decide a
scientific result.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

MAX_ENVELOPE_BYTES = 1 << 20
RESUME_ARTIFACT_ID = "r3-f0-resume-manifest-v1"
RESULT_ARTIFACT_ID = "r3-calibration-v3-result-envelope-v1"
TIMING_CONTRACT_ID = "r3-cumulative-active-wall-v2"
ACTIVE_WALL_CAP_NS = 54_000_000_000_000

_OUTER_KEYS = frozenset({"artifact_id", "payload", "payload_sha256", "schema_version"})
_RESUME_PAYLOAD_KEYS = frozenset(
    {
        "arm_slug",
        "artifact_path",
        "checkpoint",
        "checkpoint_publication",
        "circuit_contract",
        "claim_boundary",
        "construction_seed",
        "cursor",
        "phase",
        "previous_state",
        "protocol_sha256",
        "reference_trajectory_id",
        "screen_id",
        "source_binding",
        "source_bundle_identity_sha256",
        "stream_prefixes",
        "thresholds",
        "timing",
        "train_input",
    }
)
_RESULT_PAYLOAD_KEYS = frozenset(
    {
        "artifact_path",
        "claim_boundary",
        "construction_seeds",
        "phase",
        "protocol_sha256",
        "result",
        "screen_id",
        "source_binding",
        "source_bundle_identity_sha256",
        "thresholds",
    }
)
_THRESHOLD_KEYS = frozenset({"theta_active", "theta_match"})
_CURSOR_KEYS = frozenset(
    {
        "after_transition_event",
        "book_name",
        "book_ordinal_zero_based",
        "event_count",
        "future_book_resets",
        "last_token_index_zero_based",
        "learning_step",
        "next_token_index_zero_based",
        "remaining_transition_events",
        "reset_before_next_token",
    }
)
_CURSOR_UINT_KEYS = _CURSOR_KEYS - {"book_name", "reset_before_next_token"}
_TIMING_KEYS = frozenset({"prior_cumulative_ns", "session_active_ns", "cumulative_active_ns"})
_CLAIM_BOUNDARY_KEYS = frozenset(
    {"capability_evidence", "held_out_data_accessed", "transfer_evidence"}
)
_R3_F0_RESULT_KEYS = frozenset(
    {
        "arm_slug",
        "comparisons",
        "kind",
        "reference",
        "resumes",
        "structural_metrics",
        "train_input",
        "verdict",
    }
)
_ELECTRICAL_RESULT_KEYS = frozenset(
    {
        "boundaries",
        "fixture",
        "monotonicity",
        "proximal_channel",
        "seed",
        "verdict",
        "winner_relay",
    }
)
_Q4_PASS_RESULT_KEYS = frozenset(
    {
        "circuit_identity_sha256",
        "cohort_count",
        "construction_seed",
        "next_reserve_cell",
        "segment_count",
        "status",
    }
)
_Q4_STOP_RESULT_KEYS = frozenset({"construction_seed", "reason", "status"})
_Q255_PASS_RESULT_KEYS = frozenset(
    {
        "circuit_identity_sha256",
        "learned_recurrences",
        "maximum_winner_overlap",
        "next_reserve_cell",
        "novel_probes",
        "segment_count",
        "status",
        "unique_source_contexts",
        "unique_target_winner_tuples",
    }
)
_Q255_STOP_RESULT_KEYS = frozenset({"reason", "status"})
_DEVELOPMENT_SELECTION_RESULT_KEYS = frozenset({"selected_arm_slug", "verdicts"})
_FINAL_VERDICT_RESULT_KEYS = frozenset({"selected_arm_slug", "verdict", "first_failed_gate"})
_FINAL_PASS_VERDICT = "PASS_R3_CALIBRATION_V3"
_FINAL_REJECT_VERDICT = "REJECT_R3_CALIBRATION_V3"
_FINAL_PRE_SELECTION_FAILURE_GATES = (
    "electrical_v3",
    "compressed_reserve_q4",
    "full_cohort_q255",
    "r3_f0_development",
    "development_selection",
)
_FINAL_POST_SELECTION_FAILURE_GATES = (
    "independent_confirmation",
    "deterministic_confirmation",
)
_Q255_REGIMES = (
    "production-overlap-ordered-pairs",
    "target-disjoint-isolation-control",
)


class PersistenceStop(RuntimeError):
    """A fail-closed artifact framing or filesystem STOP."""


class EnvelopeStop(PersistenceStop):
    """An envelope is not the exact canonical schema."""


class ArtifactPathStop(PersistenceStop):
    """An artifact path is outside or unsafe within the frozen namespace."""


@contextmanager
def _read_only_descriptor(destination: Path) -> Iterator[int]:
    """Open one read descriptor and normalize close without masking a primary STOP."""

    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(destination, flags)
    except OSError as exc:
        raise ArtifactPathStop("artifact final cannot be opened safely") from exc
    try:
        yield descriptor
    except BaseException as primary:
        close_failure: BaseException | None = None
        try:
            os.close(descriptor)
        except BaseException as exc:
            close_failure = exc
        if isinstance(primary, OSError):
            stopped = ArtifactPathStop("artifact descriptor operation failed")
            if close_failure is not None:
                stopped.add_note(
                    f"artifact descriptor close also failed: {type(close_failure).__name__}"
                )
            raise stopped from primary
        if close_failure is not None:
            primary.add_note(
                f"artifact descriptor close also failed: {type(close_failure).__name__}"
            )
        raise
    else:
        try:
            os.close(descriptor)
        except OSError as exc:
            raise ArtifactPathStop("artifact descriptor close failed") from exc


class ArtifactAuthorityStop(PersistenceStop):
    """The protocol or caller did not supply authority for an official codec."""


class ArtifactPairState(StrEnum):
    """Pair publication state; COMMITTED does not validate checkpoint semantics."""

    ABSENT = "ABSENT"
    CHECKPOINT_ONLY = "CHECKPOINT_ONLY"
    MANIFEST_ONLY = "MANIFEST_ONLY"
    COMMITTED = "COMMITTED"


class ArtifactPairStop(PersistenceStop):
    """A resume pair is incomplete and requires an operator STOP."""

    def __init__(self, state: ArtifactPairState) -> None:
        self.state = state
        super().__init__(f"resume artifact pair is {state.value}")


class TimingState(StrEnum):
    WITHIN_CAP = "WITHIN_CAP"
    OVER_CAP = "OVER_CAP"
    TIMING_INDETERMINATE = "TIMING_INDETERMINATE"


class ArtifactDestinationKind(StrEnum):
    BINARY_CHECKPOINT = "BINARY_CHECKPOINT"
    DEVELOPMENT_SELECTION = "DEVELOPMENT_SELECTION"
    ELECTRICAL_RESULT = "ELECTRICAL_RESULT"
    FINAL_VERDICT = "FINAL_VERDICT"
    Q4_RESULT = "Q4_RESULT"
    Q255_RESULT = "Q255_RESULT"
    RESUME_MANIFEST = "RESUME_MANIFEST"
    R3_RESULT = "R3_RESULT"
    UNFROZEN_RESULT = "UNFROZEN_RESULT"


@dataclass(frozen=True, slots=True)
class DestinationSpec:
    """Protocol-derived semantics for one exact frozen artifact destination."""

    relative_path: str
    kind: ArtifactDestinationKind
    screen_id: str | None = None
    phase: str | None = None
    arm_slug: str | None = None
    thresholds: tuple[int, int] | None = None
    construction_seed: int | None = None
    construction_seeds: tuple[int, ...] | None = None
    regime: str | None = None
    candidate_thresholds: tuple[tuple[str, tuple[int, int]], ...] | None = None
    resume_cursor: tuple[tuple[str, object], ...] | None = None


@dataclass(frozen=True, slots=True)
class PreparedResumePair:
    """One absent, protocol-derived checkpoint/manifest destination pair."""

    checkpoint_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class CheckpointFinalPin:
    """Opaque identity of one verified immutable checkpoint final."""

    checkpoint_path: Path
    _checkpoint_relative_path: str
    _filesystem_identity: tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class ManifestFinalPin:
    """Exact bounded manifest bytes and filesystem identity read before import."""

    manifest_path: Path
    _manifest_relative_path: str
    _filesystem_identity: tuple[int, int, int, int, int]
    _manifest_sha256: str


@dataclass(frozen=True, slots=True)
class ManifestCommittedResumePair:
    """One canonical manifest followed by its exact plain checkpoint final."""

    checkpoint_path: Path
    manifest: ArtifactEnvelope
    manifest_bytes: bytes
    manifest_pin: ManifestFinalPin


@dataclass(frozen=True, slots=True)
class CumulativeActiveWallTiming:
    """One exact clean session receipt, or an irreversible unclean-session state."""

    prior_cumulative_ns: int
    session_active_ns: int | None
    cumulative_active_ns: int | None

    contract_id: ClassVar[str] = TIMING_CONTRACT_ID
    cap_ns: ClassVar[int] = ACTIVE_WALL_CAP_NS

    def __post_init__(self) -> None:
        if type(self.prior_cumulative_ns) is not int or self.prior_cumulative_ns < 0:
            raise ValueError("prior_cumulative_ns must be a non-negative exact int")
        if self.session_active_ns is None or self.cumulative_active_ns is None:
            if self.session_active_ns is not None or self.cumulative_active_ns is not None:
                raise ValueError("indeterminate timing must omit session and cumulative together")
            return
        if type(self.session_active_ns) is not int or self.session_active_ns < 0:
            raise ValueError("session_active_ns must be a non-negative exact int")
        if type(self.cumulative_active_ns) is not int or self.cumulative_active_ns < 0:
            raise ValueError("cumulative_active_ns must be a non-negative exact int")
        if self.prior_cumulative_ns + self.session_active_ns != self.cumulative_active_ns:
            raise ValueError("timing receipt violates prior + session == cumulative")

    @classmethod
    def complete_fresh_session(cls, session_active_ns: int) -> CumulativeActiveWallTiming:
        if type(session_active_ns) is not int or session_active_ns < 0:
            raise ValueError("session_active_ns must be a non-negative exact int")
        return cls(
            prior_cumulative_ns=0,
            session_active_ns=session_active_ns,
            cumulative_active_ns=session_active_ns,
        )

    @classmethod
    def complete_resumed_session(
        cls,
        prior_cumulative_ns: int,
        session_active_ns: int,
    ) -> CumulativeActiveWallTiming:
        if type(prior_cumulative_ns) is not int or prior_cumulative_ns < 0:
            raise ValueError("prior_cumulative_ns must be a non-negative exact int")
        if type(session_active_ns) is not int or session_active_ns < 0:
            raise ValueError("session_active_ns must be a non-negative exact int")
        return cls(
            prior_cumulative_ns=prior_cumulative_ns,
            session_active_ns=session_active_ns,
            cumulative_active_ns=prior_cumulative_ns + session_active_ns,
        )

    @property
    def state(self) -> TimingState:
        if self.cumulative_active_ns is None:
            return TimingState.TIMING_INDETERMINATE
        if self.cumulative_active_ns <= self.cap_ns:
            return TimingState.WITHIN_CAP
        return TimingState.OVER_CAP

    def plus_completed_session(self, active_ns: int) -> CumulativeActiveWallTiming:
        if type(active_ns) is not int or active_ns < 0:
            raise ValueError("active_ns must be a non-negative exact int")
        if self.cumulative_active_ns is None:
            return self
        return CumulativeActiveWallTiming.complete_resumed_session(
            self.cumulative_active_ns,
            active_ns,
        )

    def with_unclean_interruption(self) -> CumulativeActiveWallTiming:
        prior = (
            self.prior_cumulative_ns
            if self.cumulative_active_ns is None
            else self.cumulative_active_ns
        )
        return CumulativeActiveWallTiming(
            prior_cumulative_ns=prior,
            session_active_ns=None,
            cumulative_active_ns=None,
        )


@dataclass(frozen=True, slots=True)
class ConfirmationTimingComparison:
    """Keep fresh/resumed receipts distinct and expose only their conservative max."""

    fresh: CumulativeActiveWallTiming
    resumed: CumulativeActiveWallTiming

    def __post_init__(self) -> None:
        if type(self.fresh) is not CumulativeActiveWallTiming:
            raise TypeError("fresh must be an exact CumulativeActiveWallTiming")
        if type(self.resumed) is not CumulativeActiveWallTiming:
            raise TypeError("resumed must be an exact CumulativeActiveWallTiming")
        if self.fresh.prior_cumulative_ns != 0:
            raise ValueError("fresh timing must start with prior_cumulative_ns zero")

    @property
    def conservative_cumulative_active_ns(self) -> int | None:
        fresh = self.fresh.cumulative_active_ns
        resumed = self.resumed.cumulative_active_ns
        if fresh is None or resumed is None:
            return None
        return max(fresh, resumed)

    @property
    def state(self) -> TimingState:
        cumulative = self.conservative_cumulative_active_ns
        if cumulative is None:
            return TimingState.TIMING_INDETERMINATE
        if cumulative <= ACTIVE_WALL_CAP_NS:
            return TimingState.WITHIN_CAP
        return TimingState.OVER_CAP


@dataclass(frozen=True, slots=True)
class ArtifactEnvelope:
    artifact_id: str
    payload: dict[str, Any]
    payload_sha256: str
    schema_version: int


def _exact_uint(label: str, value: object, *, bits: int) -> int:
    if type(value) is not int or not 0 <= value < (1 << bits):
        raise ValueError(f"{label} must be an exact u{bits}")
    return value


def _exact_uint_tuple(
    label: str,
    values: object,
    *,
    length: int,
    bits: int,
) -> tuple[int, ...]:
    if type(values) is not tuple or len(values) != length:
        raise ValueError(f"{label} must contain exactly {length} entries")
    return tuple(
        _exact_uint(f"{label}[{index}]", value, bits=bits) for index, value in enumerate(values)
    )


def _u16le(value: int) -> bytes:
    return value.to_bytes(2, "little", signed=False)


def _u32le(value: int) -> bytes:
    return value.to_bytes(4, "little", signed=False)


def _u64le(value: int) -> bytes:
    return value.to_bytes(8, "little", signed=False)


@dataclass(frozen=True, slots=True)
class TemporalSpikeFrame:
    cell_id: int
    tick: int

    def __post_init__(self) -> None:
        _exact_uint("cell_id", self.cell_id, bits=32)
        _exact_uint("tick", self.tick, bits=16)

    def canonical_bytes(self) -> bytes:
        return _u32le(self.cell_id) + _u16le(self.tick)


@dataclass(frozen=True, slots=True)
class WinnerChainFrame:
    token_id: int
    event_count: int
    winner_context: tuple[int, ...]

    def __post_init__(self) -> None:
        _exact_uint("token_id", self.token_id, bits=32)
        _exact_uint("event_count", self.event_count, bits=64)
        winners = _exact_uint_tuple("winner_context", self.winner_context, length=11, bits=32)
        object.__setattr__(self, "winner_context", winners)

    def canonical_row(self) -> bytes:
        row = bytearray(_u32le(self.token_id))
        row.extend(_u64le(self.event_count))
        for winner in self.winner_context:
            row.extend(_u32le(winner))
        if len(row) != 56:  # pragma: no cover - protected by fixed widths above.
            raise AssertionError("winner chain row lost its frozen 56-byte layout")
        return bytes(row)


def _exact_spike_tuple(label: str, values: object) -> tuple[TemporalSpikeFrame, ...]:
    if type(values) is not tuple or len(values) != 11:
        raise ValueError(f"{label} must contain exactly 11 spikes")
    if any(type(value) is not TemporalSpikeFrame for value in values):
        raise TypeError(f"{label} entries must be TemporalSpikeFrame values")
    return values


@dataclass(frozen=True, slots=True)
class TemporalStepFrame:
    """The exact persisted view of one successful native temporal token call."""

    token_id: int
    punished_segments: int
    event_count: int
    candidate_segments: int
    matching_segments: int
    active_segments: int
    postings_visited: int
    predicted: bool
    predicted_columns: int
    allocated_segments: int
    inhibition_onset_ticks: tuple[int, ...]
    column_simulated_ticks: tuple[int, ...]
    winner_spikes: tuple[TemporalSpikeFrame, ...]
    basket_spikes: tuple[TemporalSpikeFrame, ...]
    proximal_spikes: tuple[TemporalSpikeFrame, ...]
    proximal_relay_spikes: tuple[TemporalSpikeFrame, ...]
    winner_relay_spikes: tuple[TemporalSpikeFrame, ...]
    winner_context: tuple[int, ...]
    proximal_context: tuple[int, ...]
    context: tuple[int, ...]

    def __post_init__(self) -> None:
        _exact_uint("token_id", self.token_id, bits=32)
        _exact_uint("punished_segments", self.punished_segments, bits=32)
        _exact_uint("event_count", self.event_count, bits=64)
        candidate_segments = _exact_uint("candidate_segments", self.candidate_segments, bits=64)
        matching_segments = _exact_uint("matching_segments", self.matching_segments, bits=64)
        active_segments = _exact_uint("active_segments", self.active_segments, bits=64)
        postings_visited = _exact_uint("postings_visited", self.postings_visited, bits=64)
        if matching_segments > candidate_segments:
            raise ValueError("matching_segments exceeds candidate_segments")
        if active_segments > matching_segments:
            raise ValueError("active_segments exceeds matching_segments")
        if candidate_segments > postings_visited:
            raise ValueError("candidate_segments exceeds postings_visited")
        if type(self.predicted) is not bool:
            raise TypeError("predicted must be an exact bool")
        if _exact_uint("predicted_columns", self.predicted_columns, bits=8) > 11:
            raise ValueError("predicted_columns exceeds the frozen active width")
        if _exact_uint("allocated_segments", self.allocated_segments, bits=8) > 11:
            raise ValueError("allocated_segments exceeds the frozen active width")
        object.__setattr__(
            self,
            "inhibition_onset_ticks",
            _exact_uint_tuple(
                "inhibition_onset_ticks",
                self.inhibition_onset_ticks,
                length=11,
                bits=16,
            ),
        )
        object.__setattr__(
            self,
            "column_simulated_ticks",
            _exact_uint_tuple(
                "column_simulated_ticks",
                self.column_simulated_ticks,
                length=11,
                bits=16,
            ),
        )
        for field_name in (
            "winner_spikes",
            "basket_spikes",
            "proximal_spikes",
            "proximal_relay_spikes",
            "winner_relay_spikes",
        ):
            object.__setattr__(
                self,
                field_name,
                _exact_spike_tuple(field_name, getattr(self, field_name)),
            )
        for field_name in ("winner_context", "proximal_context", "context"):
            object.__setattr__(
                self,
                field_name,
                _exact_uint_tuple(
                    field_name,
                    getattr(self, field_name),
                    length=22,
                    bits=32,
                ),
            )

    def canonical_row(self) -> bytes:
        row = bytearray()
        row.extend(_u32le(self.token_id))
        row.extend(_u32le(self.punished_segments))
        row.extend(_u64le(self.event_count))
        row.extend(_u64le(self.candidate_segments))
        row.extend(_u64le(self.matching_segments))
        row.extend(_u64le(self.active_segments))
        row.extend(_u64le(self.postings_visited))
        row.append(int(self.predicted))
        row.append(self.predicted_columns)
        row.append(self.allocated_segments)
        for tick in self.inhibition_onset_ticks:
            row.extend(_u16le(tick))
        for tick in self.column_simulated_ticks:
            row.extend(_u16le(tick))
        for spikes in (
            self.winner_spikes,
            self.basket_spikes,
            self.proximal_spikes,
            self.proximal_relay_spikes,
            self.winner_relay_spikes,
        ):
            for spike in spikes:
                row.extend(spike.canonical_bytes())
        for context in (self.winner_context, self.proximal_context, self.context):
            for cell_id in context:
                row.extend(_u32le(cell_id))
        if len(row) != 689:  # pragma: no cover - protected by fixed widths above.
            raise AssertionError("step chain row lost its frozen 689-byte layout")
        return bytes(row)


class HashChainKind(StrEnum):
    WINNER = "WINNER"
    STEP = "STEP"

    @property
    def domain(self) -> bytes:
        if self is HashChainKind.WINNER:
            return b"humanbrain-r3-f0-winner-chain-v1\0"
        return b"humanbrain-r3-f0-step-chain-v1\0"


@dataclass(frozen=True, slots=True)
class ResumableHashChain:
    """A canonical prefix digest restored only from a committed manifest."""

    kind: HashChainKind
    count: int
    digest: bytes

    def __post_init__(self) -> None:
        if type(self.kind) is not HashChainKind:
            raise TypeError("kind must be an exact HashChainKind")
        _exact_uint("count", self.count, bits=64)
        if type(self.digest) is not bytes or len(self.digest) != 32:
            raise ValueError("digest must be exactly 32 bytes")

    @classmethod
    def start(cls, kind: HashChainKind) -> ResumableHashChain:
        if type(kind) is not HashChainKind:
            raise TypeError("kind must be an exact HashChainKind")
        return cls(kind=kind, count=0, digest=hashlib.sha256(kind.domain).digest())

    @classmethod
    def restore(
        cls,
        *,
        kind: HashChainKind,
        count: int,
        digest_hex: str,
    ) -> ResumableHashChain:
        if (
            type(digest_hex) is not str
            or len(digest_hex) != 64
            or any(character not in "0123456789abcdef" for character in digest_hex)
        ):
            raise ValueError("digest_hex must be one lowercase SHA-256 identity")
        return cls(kind=kind, count=count, digest=bytes.fromhex(digest_hex))

    @property
    def hexdigest(self) -> str:
        return self.digest.hex()

    def _append(self, row: bytes) -> ResumableHashChain:
        if self.count == (1 << 64) - 1:
            raise ValueError("hash-chain count cannot exceed u64")
        next_count = self.count + 1
        digest = hashlib.sha256(self.kind.domain + self.digest + _u64le(next_count) + row).digest()
        return ResumableHashChain(kind=self.kind, count=next_count, digest=digest)

    def append_winner(self, frame: WinnerChainFrame) -> ResumableHashChain:
        if self.kind is not HashChainKind.WINNER:
            raise ValueError("winner frames require the winner hash-chain domain")
        if type(frame) is not WinnerChainFrame:
            raise TypeError("frame must be an exact WinnerChainFrame")
        return self._append(frame.canonical_row())

    def append_step(self, frame: TemporalStepFrame) -> ResumableHashChain:
        if self.kind is not HashChainKind.STEP:
            raise ValueError("step frames require the step hash-chain domain")
        if type(frame) is not TemporalStepFrame:
            raise TypeError("frame must be an exact TemporalStepFrame")
        return self._append(frame.canonical_row())


_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _is_reparse(value: os.stat_result) -> bool:
    return bool(int(getattr(value, "st_file_attributes", 0)) & _FILE_ATTRIBUTE_REPARSE_POINT)


def _is_plain_directory(value: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode) and not _is_reparse(value)
    )


def _is_plain_regular_file(value: os.stat_result) -> bool:
    return (
        stat.S_ISREG(value.st_mode) and not stat.S_ISLNK(value.st_mode) and not _is_reparse(value)
    )


def _validate_relative_path(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise ArtifactPathStop(f"{label} must be a string")
    parts = value.split("/")
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ArtifactPathStop(f"{label} is not a canonical repository-relative path")
    return value


def _collect_artifact_paths(value: object, *, artifact_root: str) -> tuple[str, ...]:
    paths: list[str] = []

    def collect(current: object) -> None:
        if type(current) is str:
            if current.startswith(f"{artifact_root}/") and current.endswith((".json", ".r3ntcp1")):
                paths.append(_validate_relative_path(current, label="artifact path"))
            return
        if type(current) is dict:
            for child in current.values():
                collect(child)
            return
        if type(current) is list:
            for child in current:
                collect(child)

    collect(value)
    if not paths:
        raise ArtifactPathStop("validated protocol supplies no artifact paths")
    if len(paths) != len(set(paths)):
        raise ArtifactPathStop("validated protocol supplies duplicate artifact paths")
    return tuple(paths)


def _artifact_parent_directories(
    artifact_paths: tuple[str, ...],
    *,
    artifact_root: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Derive direct parents and their complete repository-relative closure."""

    direct: set[str] = set()
    closure: set[str] = set()
    root_depth = len(artifact_root.split("/"))
    for artifact_path in artifact_paths:
        parts = artifact_path.split("/")[:-1]
        direct.add("/".join(parts))
        for length in range(root_depth, len(parts) + 1):
            closure.add("/".join(parts[:length]))

    def order(value: str) -> tuple[int, str]:
        return value.count("/"), value

    return tuple(sorted(direct, key=order)), tuple(sorted(closure, key=order))


def _protocol_object(value: object, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ArtifactAuthorityStop(f"validated protocol {label} must be an exact object")
    return value


def _protocol_int_list(value: object, *, label: str) -> tuple[int, ...]:
    if type(value) is not list or not value:
        raise ArtifactAuthorityStop(f"validated protocol {label} must be a non-empty list")
    integers: list[int] = []
    for item in value:
        if type(item) is not int or item < 0:
            raise ArtifactAuthorityStop(f"validated protocol {label} has a non-integer entry")
        integers.append(item)
    return tuple(integers)


def _threshold_arm_map(validated_protocol: Mapping[str, object]) -> dict[str, tuple[int, int]]:
    selection = _protocol_object(
        validated_protocol.get("threshold_selection"),
        label="threshold_selection",
    )
    candidates = selection.get("candidate_arms")
    if type(candidates) is not list or not candidates:
        raise ArtifactAuthorityStop("validated protocol has no candidate threshold arms")
    arms: dict[str, tuple[int, int]] = {}
    for candidate in candidates:
        if (
            type(candidate) is not list
            or len(candidate) != 2
            or any(type(value) is not int or value < 0 for value in candidate)
        ):
            raise ArtifactAuthorityStop("candidate threshold arms must be exact integer pairs")
        theta_match, theta_active = candidate
        slug = f"theta-m-{theta_match}_theta-a-{theta_active}"
        if slug in arms:
            raise ArtifactAuthorityStop("candidate threshold arms contain a duplicate")
        arms[slug] = (theta_match, theta_active)
    return arms


def _fixture_phase_seeds(
    validated_protocol: Mapping[str, object],
    phase: str,
) -> tuple[int, ...]:
    selection = _protocol_object(
        validated_protocol.get("threshold_selection"),
        label="threshold_selection",
    )
    return _protocol_int_list(
        selection.get(f"{phase}_seeds"),
        label=f"threshold_selection.{phase}_seeds",
    )


def _final_verdict_contract(
    validated_protocol: Mapping[str, object],
    *,
    construction_seeds: tuple[int, ...],
) -> dict[str, Any]:
    result_contract = _protocol_object(
        validated_protocol.get("result_contract"),
        label="result_contract",
    )
    observed = _protocol_object(
        result_contract.get("final_verdict"),
        label="result_contract.final_verdict",
    )
    expected = {
        "construction_seeds": list(construction_seeds),
        "phase": "final",
        "result_keys": [
            "selected_arm_slug",
            "verdict",
            "first_failed_gate",
        ],
        "screen_id": "threshold_selection",
        "verdicts": {
            _FINAL_PASS_VERDICT: {
                "first_failed_gate": None,
                "selected_arm_slug": "candidate_arm_required",
                "thresholds": "selected_arm_required",
            },
            _FINAL_REJECT_VERDICT: {
                "post_selection": {
                    "first_failed_gates": list(_FINAL_POST_SELECTION_FAILURE_GATES),
                    "selected_arm_slug": "candidate_arm_required",
                    "thresholds": "selected_arm_required",
                },
                "pre_selection": {
                    "first_failed_gates": list(_FINAL_PRE_SELECTION_FAILURE_GATES),
                    "selected_arm_slug": None,
                    "thresholds": None,
                },
            },
        },
    }
    if observed != expected:
        raise ArtifactAuthorityStop("final-verdict contract differs from the frozen policy")
    return observed


def _protocol_path_list(value: object, *, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise ArtifactAuthorityStop(f"validated protocol {label} must be a non-empty list")
    return tuple(_validate_relative_path(path, label=label) for path in value)


def _r3_screen_contract(validated_protocol: Mapping[str, object]) -> dict[str, Any]:
    screens = _protocol_object(validated_protocol.get("screens"), label="screens")
    return _protocol_object(screens.get("r3_f0_train"), label="screens.r3_f0_train")


def _r3_phase_seeds(screen: dict[str, Any], phase: str) -> tuple[int, ...]:
    scope = _protocol_object(
        screen.get("construction_seed_scope"),
        label="screens.r3_f0_train.construction_seed_scope",
    )
    return _protocol_int_list(
        scope.get(phase),
        label=f"screens.r3_f0_train.construction_seed_scope.{phase}",
    )


def _resume_milestones(screen: dict[str, Any]) -> dict[int, tuple[tuple[str, object], ...]]:
    determinism = _protocol_object(
        screen.get("determinism"),
        label="screens.r3_f0_train.determinism",
    )
    rows = determinism.get("resume_milestones")
    if type(rows) is not list or not rows:
        raise ArtifactAuthorityStop("validated protocol has no R3 resume milestones")
    milestones: dict[int, tuple[tuple[str, object], ...]] = {}
    for row_value in rows:
        row = _protocol_object(row_value, label="R3 resume milestone")
        if frozenset(row) != _CURSOR_KEYS:
            raise ArtifactAuthorityStop("R3 resume milestone fields differ from cursor schema")
        event = row["after_transition_event"]
        if type(event) is not int or event < 0:
            raise ArtifactAuthorityStop("R3 resume milestone event must be an exact int")
        if type(row["book_name"]) is not str or not row["book_name"]:
            raise ArtifactAuthorityStop("R3 resume milestone book_name must be non-empty")
        if type(row["reset_before_next_token"]) is not bool:
            raise ArtifactAuthorityStop("R3 resume reset flag must be an exact bool")
        if any(type(row[key]) is not int or row[key] < 0 for key in _CURSOR_UINT_KEYS):
            raise ArtifactAuthorityStop("R3 resume milestone counters must be exact ints")
        if event in milestones:
            raise ArtifactAuthorityStop("R3 resume milestones contain a duplicate event")
        milestones[event] = tuple((key, row[key]) for key in sorted(row))
    return milestones


def _seed_from_filename(filename: str, *, prefix: str) -> int:
    suffix = ".json"
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        raise ArtifactPathStop("R3 result path has a noncanonical seed filename")
    digits = filename[len(prefix) : -len(suffix)]
    if len(digits) != 10 or not digits.isascii() or not digits.isdigit():
        raise ArtifactPathStop("R3 result seed filename is not ten decimal digits")
    return int(digits)


def _parse_r3_result_path(
    path: str,
    *,
    artifact_root: str,
    phase: str,
    declared_arm_slug: str | None,
) -> tuple[str, int]:
    prefix = f"{artifact_root}/"
    parts = path.removeprefix(prefix).split("/")
    if (
        not path.startswith(prefix)
        or len(parts) != 4
        or parts[0] != "r3-f0"
        or parts[1] != phase.replace("_", "-")
    ):
        raise ArtifactPathStop("R3 result path does not match its protocol phase")
    arm_slug = parts[2]
    if declared_arm_slug is not None and declared_arm_slug != arm_slug:
        raise ArtifactPathStop("R3 result path does not match its protocol arm key")
    seed = _seed_from_filename(parts[3], prefix="construction-seed-")
    return arm_slug, seed


def _parse_r3_resume_path(
    path: str,
    *,
    artifact_root: str,
) -> tuple[str, int, ArtifactDestinationKind]:
    prefix = f"{artifact_root}/"
    parts = path.removeprefix(prefix).split("/")
    if (
        not path.startswith(prefix)
        or len(parts) != 5
        or parts[0:2] != ["r3-f0", "deterministic-confirmation"]
        or parts[3] != "resume"
    ):
        raise ArtifactPathStop("R3 resume destination has a noncanonical path")
    filename = parts[4]
    if filename.endswith(".resume.json"):
        kind = ArtifactDestinationKind.RESUME_MANIFEST
        digits = filename.removeprefix("transition-").removesuffix(".resume.json")
    elif filename.endswith(".r3ntcp1"):
        kind = ArtifactDestinationKind.BINARY_CHECKPOINT
        digits = filename.removeprefix("transition-").removesuffix(".r3ntcp1")
    else:
        raise ArtifactPathStop("R3 resume destination has an unknown file type")
    if (
        not filename.startswith("transition-")
        or len(digits) != 10
        or not digits.isascii()
        or not digits.isdigit()
    ):
        raise ArtifactPathStop("R3 resume event is not ten decimal digits")
    return parts[2], int(digits), kind


def _derive_destination_specs(
    validated_protocol: Mapping[str, object],
    *,
    contract: dict[str, Any],
    artifact_root: str,
    artifact_paths: tuple[str, ...],
) -> dict[str, DestinationSpec]:
    arms = _threshold_arm_map(validated_protocol)
    screen = _r3_screen_contract(validated_protocol)
    milestones = _resume_milestones(screen)
    r3_contract = _protocol_object(contract.get("r3_f0_train"), label="R3 artifact contract")
    destinations: dict[str, DestinationSpec] = {}

    def add(spec: DestinationSpec) -> None:
        if spec.relative_path in destinations:
            raise ArtifactPathStop("artifact destination has multiple protocol descriptors")
        destinations[spec.relative_path] = spec

    fixture_seeds = {
        phase: _fixture_phase_seeds(validated_protocol, phase)
        for phase in ("development", "independent_confirmation")
    }
    final_construction_seeds = tuple(
        dict.fromkeys(fixture_seeds["development"] + fixture_seeds["independent_confirmation"])
    )
    final_contract = _final_verdict_contract(
        validated_protocol,
        construction_seeds=final_construction_seeds,
    )
    electrical_contract = _protocol_object(
        contract.get("electrical_v3"),
        label="electrical_v3 artifact contract",
    )
    q4_contract = _protocol_object(
        contract.get("compressed_reserve_q4"),
        label="compressed_reserve_q4 artifact contract",
    )
    q255_contract = _protocol_object(
        contract.get("full_cohort_q255"),
        label="full_cohort_q255 artifact contract",
    )
    for phase, seeds in fixture_seeds.items():
        phase_path = phase.replace("_", "-")
        electrical_paths = _protocol_path_list(
            electrical_contract.get(phase),
            label=f"electrical_v3.{phase}",
        )
        if len(electrical_paths) != len(seeds):
            raise ArtifactAuthorityStop("electrical result paths do not cover the phase seeds")
        for seed, path in zip(seeds, electrical_paths, strict=True):
            expected = f"{artifact_root}/electrical-v3/{phase_path}/seed-{seed:010d}.json"
            if path != expected:
                raise ArtifactPathStop("electrical result path differs from its frozen coordinate")
            add(
                DestinationSpec(
                    relative_path=path,
                    kind=ArtifactDestinationKind.ELECTRICAL_RESULT,
                    screen_id="electrical_v3",
                    phase=phase,
                    construction_seeds=(seed,),
                )
            )

        q4_phase = _protocol_object(
            q4_contract.get(phase),
            label=f"compressed_reserve_q4.{phase}",
        )
        if frozenset(q4_phase) != frozenset(arms):
            raise ArtifactAuthorityStop("q4 result grid does not cover the candidate arms")
        for arm_slug, thresholds in arms.items():
            paths = _protocol_path_list(
                q4_phase[arm_slug],
                label=f"compressed_reserve_q4.{phase}.{arm_slug}",
            )
            if len(paths) != len(seeds):
                raise ArtifactAuthorityStop("q4 result paths do not cover the phase seeds")
            for seed, path in zip(seeds, paths, strict=True):
                expected = f"{artifact_root}/q4/{phase_path}/{arm_slug}/seed-{seed:010d}.json"
                if path != expected:
                    raise ArtifactPathStop("q4 result path differs from its frozen coordinate")
                add(
                    DestinationSpec(
                        relative_path=path,
                        kind=ArtifactDestinationKind.Q4_RESULT,
                        screen_id="compressed_reserve_q4",
                        phase=phase,
                        arm_slug=arm_slug,
                        thresholds=thresholds,
                        construction_seeds=(seed,),
                    )
                )

        q255_phase = _protocol_object(
            q255_contract.get(phase),
            label=f"full_cohort_q255.{phase}",
        )
        if frozenset(q255_phase) != frozenset(_Q255_REGIMES):
            raise ArtifactAuthorityStop("q255 result grid does not cover both frozen regimes")
        for regime in _Q255_REGIMES:
            regime_grid = _protocol_object(
                q255_phase[regime],
                label=f"full_cohort_q255.{phase}.{regime}",
            )
            if frozenset(regime_grid) != frozenset(arms):
                raise ArtifactAuthorityStop("q255 result grid does not cover the candidate arms")
            for arm_slug, thresholds in arms.items():
                paths = _protocol_path_list(
                    regime_grid[arm_slug],
                    label=f"full_cohort_q255.{phase}.{regime}.{arm_slug}",
                )
                if len(paths) != len(seeds):
                    raise ArtifactAuthorityStop("q255 result paths do not cover the phase seeds")
                for seed, path in zip(seeds, paths, strict=True):
                    expected = (
                        f"{artifact_root}/q255/{phase_path}/{regime}/{arm_slug}/"
                        f"seed-{seed:010d}.json"
                    )
                    if path != expected:
                        raise ArtifactPathStop(
                            "q255 result path differs from its frozen coordinate"
                        )
                    add(
                        DestinationSpec(
                            relative_path=path,
                            kind=ArtifactDestinationKind.Q255_RESULT,
                            screen_id="full_cohort_q255",
                            phase=phase,
                            arm_slug=arm_slug,
                            thresholds=thresholds,
                            construction_seeds=(seed,),
                            regime=regime,
                        )
                    )

    selection_contract = _protocol_object(
        contract.get("selection"),
        label="selection artifact contract",
    )
    development_selection_path = _validate_relative_path(
        selection_contract.get("development"),
        label="development selection path",
    )
    if development_selection_path != (f"{artifact_root}/selection/development-selection.json"):
        raise ArtifactPathStop("development selection path is not canonical")
    final_selection_path = _validate_relative_path(
        selection_contract.get("final"),
        label="final selection path",
    )
    if final_selection_path != f"{artifact_root}/selection/final-verdict.json":
        raise ArtifactPathStop("final selection path is not canonical")
    add(
        DestinationSpec(
            relative_path=development_selection_path,
            kind=ArtifactDestinationKind.DEVELOPMENT_SELECTION,
            screen_id="threshold_selection",
            phase="development",
            construction_seeds=fixture_seeds["development"],
            candidate_thresholds=tuple(arms.items()),
        )
    )
    add(
        DestinationSpec(
            relative_path=final_selection_path,
            kind=ArtifactDestinationKind.FINAL_VERDICT,
            screen_id=final_contract["screen_id"],
            phase=final_contract["phase"],
            construction_seeds=final_construction_seeds,
            candidate_thresholds=tuple(arms.items()),
        )
    )

    for phase in ("development", "deterministic_confirmation"):
        phase_contract = r3_contract.get(phase)
        if type(phase_contract) is not dict or not phase_contract:
            raise ArtifactAuthorityStop(
                f"R3 {phase} artifact contract must be a non-empty arm-to-path object"
            )
        declared_paths = tuple(phase_contract.items())
        allowed_seeds = _r3_phase_seeds(screen, phase)
        for declared_slug, path_value in declared_paths:
            path = _validate_relative_path(path_value, label="R3 result path")
            arm_slug, seed = _parse_r3_result_path(
                path,
                artifact_root=artifact_root,
                phase=phase,
                declared_arm_slug=declared_slug,
            )
            if arm_slug not in arms:
                raise ArtifactAuthorityStop("R3 result arm is not a candidate threshold arm")
            if seed not in allowed_seeds:
                raise ArtifactAuthorityStop("R3 result seed is outside its protocol phase")
            add(
                DestinationSpec(
                    relative_path=path,
                    kind=ArtifactDestinationKind.R3_RESULT,
                    screen_id="r3_f0_train",
                    phase=phase,
                    arm_slug=arm_slug,
                    thresholds=arms[arm_slug],
                    construction_seeds=(seed,),
                )
            )

    resume_paths = r3_contract.get("deterministic_confirmation_resumes")
    if type(resume_paths) is not list or not resume_paths:
        raise ArtifactAuthorityStop("R3 artifact contract has no resume destinations")
    resume_seeds = _r3_phase_seeds(screen, "deterministic_confirmation")
    if len(resume_seeds) != 1:
        raise ArtifactAuthorityStop("R3 resume authority requires exactly one construction seed")
    for path_value in resume_paths:
        path = _validate_relative_path(path_value, label="R3 resume path")
        arm_slug, event, kind = _parse_r3_resume_path(path, artifact_root=artifact_root)
        if arm_slug not in arms:
            raise ArtifactAuthorityStop("R3 resume arm is not a candidate threshold arm")
        if event not in milestones:
            raise ArtifactAuthorityStop("R3 resume path is not a frozen milestone")
        add(
            DestinationSpec(
                relative_path=path,
                kind=kind,
                screen_id="r3_f0_train",
                phase="deterministic_confirmation",
                arm_slug=arm_slug,
                thresholds=arms[arm_slug],
                construction_seed=resume_seeds[0],
                resume_cursor=milestones[event],
            )
        )

    for path in artifact_paths:
        if path in destinations:
            continue
        if path.endswith((".r3ntcp1", ".resume.json")):
            raise ArtifactAuthorityStop("binary/resume destination lacks an R3 descriptor")
        destinations[path] = DestinationSpec(
            relative_path=path,
            kind=ArtifactDestinationKind.UNFROZEN_RESULT,
        )
    return destinations


def _canonical_json_bytes(value: object) -> bytes:
    try:
        _validate_json_tree(value)
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except EnvelopeStop:
        raise
    except (RecursionError, TypeError, ValueError) as exc:
        raise EnvelopeStop("value is not canonical JSON data") from exc


def _validate_json_tree(value: object) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        raise EnvelopeStop("floating-point JSON values are forbidden")
    if type(value) is list:
        for child in value:
            _validate_json_tree(child)
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise EnvelopeStop("JSON object keys must be strings")
            _validate_json_tree(child)
        return
    raise EnvelopeStop("value is not canonical JSON data")


def _payload_keys(artifact_id: str) -> frozenset[str]:
    if artifact_id == RESUME_ARTIFACT_ID:
        return _RESUME_PAYLOAD_KEYS
    if artifact_id == RESULT_ARTIFACT_ID:
        return _RESULT_PAYLOAD_KEYS
    raise EnvelopeStop("artifact_id is not a frozen calibration-v3 envelope ID")


def _require_exact_keys(
    value: dict[Any, Any],
    expected_keys: frozenset[str],
    *,
    label: str,
) -> None:
    if any(type(key) is not str for key in value):
        raise EnvelopeStop(f"{label} keys must be exact strings")
    observed = frozenset(value)
    if observed != expected_keys:
        missing = sorted(expected_keys - observed)
        unknown = sorted(observed - expected_keys)
        raise EnvelopeStop(f"{label} fields differ: missing={missing}, unknown={unknown}")


def _require_exact_object(
    value: object,
    expected_keys: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise EnvelopeStop(f"{label} must be a JSON object")
    _require_exact_keys(value, expected_keys, label=label)
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise EnvelopeStop(f"{label} must be a non-negative exact int")
    return value


def _require_nonempty_string(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise EnvelopeStop(f"{label} must be a non-empty string")
    return value


def _require_lower_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EnvelopeStop(f"{label} must be a lowercase SHA-256 identity")
    return value


def _require_object(value: object, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise EnvelopeStop(f"{label} must be a JSON object")
    return value


def _validate_thresholds(value: object) -> None:
    thresholds = _require_exact_object(value, _THRESHOLD_KEYS, label="thresholds")
    _require_nonnegative_int(thresholds["theta_match"], label="theta_match")
    _require_nonnegative_int(thresholds["theta_active"], label="theta_active")


def _validate_claim_boundary(value: object) -> None:
    claim = _require_exact_object(value, _CLAIM_BOUNDARY_KEYS, label="claim_boundary")
    if any(claim[key] is not False for key in _CLAIM_BOUNDARY_KEYS):
        raise EnvelopeStop("claim_boundary must contain the three exact false claims")


def _validate_common_payload(payload: dict[str, Any]) -> None:
    _require_lower_sha256(payload["protocol_sha256"], label="protocol_sha256")
    _require_lower_sha256(
        payload["source_bundle_identity_sha256"],
        label="source_bundle_identity_sha256",
    )
    _require_nonempty_string(payload["screen_id"], label="screen_id")
    _require_nonempty_string(payload["phase"], label="phase")
    _validate_claim_boundary(payload["claim_boundary"])
    _require_object(payload["source_binding"], label="source_binding")


def _bind_screen_phase(payload: dict[str, Any], destination: DestinationSpec) -> None:
    if destination.screen_id is None or payload["screen_id"] != destination.screen_id:
        raise EnvelopeStop("screen_id does not match the frozen artifact destination")
    if destination.phase is None or payload["phase"] != destination.phase:
        raise EnvelopeStop("phase does not match the frozen artifact destination")


def _bind_thresholds(payload: dict[str, Any], destination: DestinationSpec) -> None:
    _validate_thresholds(payload["thresholds"])
    if destination.thresholds is None:
        raise EnvelopeStop("artifact destination has no frozen threshold semantics")
    theta_match, theta_active = destination.thresholds
    if payload["thresholds"] != {
        "theta_active": theta_active,
        "theta_match": theta_match,
    }:
        raise EnvelopeStop("thresholds do not match the frozen artifact destination")


def _bind_common_destination(payload: dict[str, Any], destination: DestinationSpec) -> None:
    _bind_screen_phase(payload, destination)
    _bind_thresholds(payload, destination)


def _validate_resume_payload(
    payload: dict[str, Any],
    *,
    destination: DestinationSpec,
) -> None:
    _validate_common_payload(payload)
    if destination.kind is not ArtifactDestinationKind.RESUME_MANIFEST:
        raise EnvelopeStop("resume payload requires a frozen resume-manifest destination")
    _bind_common_destination(payload, destination)
    if (
        destination.construction_seed is None
        or _require_nonnegative_int(payload["construction_seed"], label="construction_seed")
        != destination.construction_seed
    ):
        raise EnvelopeStop("construction_seed does not match the frozen resume destination")
    _require_nonempty_string(payload["arm_slug"], label="arm_slug")
    if destination.arm_slug is None or payload["arm_slug"] != destination.arm_slug:
        raise EnvelopeStop("arm_slug does not match the frozen resume destination")
    _require_nonempty_string(payload["reference_trajectory_id"], label="reference_trajectory_id")
    for field_name in (
        "checkpoint",
        "checkpoint_publication",
        "circuit_contract",
        "previous_state",
        "stream_prefixes",
        "train_input",
    ):
        _require_object(payload[field_name], label=field_name)
    cursor = _require_exact_object(payload["cursor"], _CURSOR_KEYS, label="cursor")
    if type(cursor["book_name"]) is not str or not cursor["book_name"]:
        raise EnvelopeStop("cursor book_name must be a non-empty string")
    if type(cursor["reset_before_next_token"]) is not bool:
        raise EnvelopeStop("cursor reset_before_next_token must be an exact bool")
    for key in _CURSOR_UINT_KEYS:
        _require_nonnegative_int(cursor[key], label=f"cursor {key}")
    if destination.resume_cursor is None or cursor != dict(destination.resume_cursor):
        raise EnvelopeStop("cursor does not match the frozen resume milestone")
    timing = _require_exact_object(payload["timing"], _TIMING_KEYS, label="timing")
    prior = _require_nonnegative_int(
        timing["prior_cumulative_ns"], label="timing prior_cumulative_ns"
    )
    session = _require_nonnegative_int(
        timing["session_active_ns"], label="timing session_active_ns"
    )
    cumulative = _require_nonnegative_int(
        timing["cumulative_active_ns"], label="timing cumulative_active_ns"
    )
    try:
        CumulativeActiveWallTiming(
            prior_cumulative_ns=prior,
            session_active_ns=session,
            cumulative_active_ns=cumulative,
        )
    except ValueError as exc:
        raise EnvelopeStop("timing violates the cumulative active-wall contract") from exc


def _bind_result_construction_seeds(
    payload: dict[str, Any],
    destination: DestinationSpec,
) -> None:
    seeds = payload["construction_seeds"]
    if type(seeds) is not list or not seeds:
        raise EnvelopeStop("construction_seeds must be a non-empty JSON array")
    for index, seed in enumerate(seeds):
        _require_nonnegative_int(seed, label=f"construction_seeds[{index}]")
    if destination.construction_seeds is None or tuple(seeds) != destination.construction_seeds:
        raise EnvelopeStop("construction_seeds do not match the frozen result destination")


def _validate_electrical_result(
    payload: dict[str, Any],
    destination: DestinationSpec,
) -> None:
    _require_exact_object(payload["thresholds"], frozenset(), label="electrical thresholds")
    result = _require_exact_object(
        payload["result"],
        _ELECTRICAL_RESULT_KEYS,
        label="electrical_v3 result",
    )
    if destination.construction_seeds is None or len(destination.construction_seeds) != 1:
        raise EnvelopeStop("electrical destination has no exact construction seed")
    if (
        _require_nonnegative_int(result["seed"], label="electrical_v3 seed")
        != destination.construction_seeds[0]
    ):
        raise EnvelopeStop("electrical_v3 seed does not match its frozen destination")
    verdict = _require_nonempty_string(result["verdict"], label="electrical_v3 verdict")
    if verdict not in {
        "PASS_RESERVE_VALID_ELECTRICAL_V3",
        "REJECT_RESERVE_VALID_ELECTRICAL_V3",
    }:
        raise EnvelopeStop("electrical_v3 verdict is not a frozen fixture verdict")
    for field_name in (
        "boundaries",
        "fixture",
        "monotonicity",
        "proximal_channel",
        "winner_relay",
    ):
        _require_object(result[field_name], label=f"electrical_v3 {field_name}")


def _validate_q4_result(payload: dict[str, Any], destination: DestinationSpec) -> None:
    _bind_thresholds(payload, destination)
    result = _require_object(payload["result"], label="compressed_reserve_q4 result")
    status = _require_nonempty_string(
        result.get("status"),
        label="compressed_reserve_q4 status",
    )
    expected_keys = _Q4_PASS_RESULT_KEYS if status == "PASS" else _Q4_STOP_RESULT_KEYS
    if status not in {"PASS", "STOP"}:
        raise EnvelopeStop("compressed_reserve_q4 status is not PASS or STOP")
    _require_exact_keys(result, expected_keys, label="compressed_reserve_q4 result")
    if destination.construction_seeds is None or len(destination.construction_seeds) != 1:
        raise EnvelopeStop("q4 destination has no exact construction seed")
    if (
        _require_nonnegative_int(
            result["construction_seed"],
            label="compressed_reserve_q4 construction_seed",
        )
        != destination.construction_seeds[0]
    ):
        raise EnvelopeStop("q4 construction_seed does not match its frozen destination")
    if status == "STOP":
        _require_nonempty_string(result["reason"], label="compressed_reserve_q4 reason")
        return
    _require_lower_sha256(
        result["circuit_identity_sha256"],
        label="compressed_reserve_q4 circuit_identity_sha256",
    )
    for field_name in ("cohort_count", "next_reserve_cell", "segment_count"):
        _require_nonnegative_int(result[field_name], label=f"compressed_reserve_q4 {field_name}")


def _validate_q255_result(payload: dict[str, Any], destination: DestinationSpec) -> None:
    _bind_thresholds(payload, destination)
    if destination.regime not in _Q255_REGIMES:
        raise EnvelopeStop("q255 destination has no frozen regime")
    result = _require_object(payload["result"], label="full_cohort_q255 result")
    status = _require_nonempty_string(result.get("status"), label="full_cohort_q255 status")
    expected_keys = _Q255_PASS_RESULT_KEYS if status == "PASS" else _Q255_STOP_RESULT_KEYS
    if status not in {"PASS", "STOP"}:
        raise EnvelopeStop("full_cohort_q255 status is not PASS or STOP")
    _require_exact_keys(result, expected_keys, label="full_cohort_q255 result")
    if status == "STOP":
        _require_nonempty_string(result["reason"], label="full_cohort_q255 reason")
        return
    _require_lower_sha256(
        result["circuit_identity_sha256"],
        label="full_cohort_q255 circuit_identity_sha256",
    )
    for field_name in (
        "learned_recurrences",
        "maximum_winner_overlap",
        "next_reserve_cell",
        "novel_probes",
        "segment_count",
        "unique_source_contexts",
        "unique_target_winner_tuples",
    ):
        _require_nonnegative_int(result[field_name], label=f"full_cohort_q255 {field_name}")


def _validate_development_selection(
    payload: dict[str, Any],
    destination: DestinationSpec,
) -> None:
    result = _require_exact_object(
        payload["result"],
        _DEVELOPMENT_SELECTION_RESULT_KEYS,
        label="development selection result",
    )
    candidates = dict(destination.candidate_thresholds or ())
    if not candidates:
        raise EnvelopeStop("development selection destination has no candidate arms")
    verdicts = _require_object(result["verdicts"], label="development selection verdicts")
    _require_exact_keys(
        verdicts,
        frozenset(candidates),
        label="development selection verdicts",
    )
    if any(type(verdicts[arm_slug]) is not bool for arm_slug in candidates):
        raise EnvelopeStop("development selection verdicts must be exact booleans")
    selected = _require_nonempty_string(
        result["selected_arm_slug"],
        label="development selected_arm_slug",
    )
    first_passing = next(
        (arm_slug for arm_slug in candidates if verdicts[arm_slug]),
        None,
    )
    if first_passing is None:
        raise EnvelopeStop("development selection cannot publish without a passing arm")
    if selected != first_passing:
        raise EnvelopeStop("development selection is not the first passing candidate arm")
    _validate_thresholds(payload["thresholds"])
    theta_match, theta_active = candidates[selected]
    if payload["thresholds"] != {
        "theta_active": theta_active,
        "theta_match": theta_match,
    }:
        raise EnvelopeStop("development selection thresholds differ from the selected arm")


def _bind_final_selected_arm(
    payload: dict[str, Any],
    result: dict[str, Any],
    candidates: dict[str, tuple[int, int]],
) -> None:
    selected = _require_nonempty_string(
        result["selected_arm_slug"],
        label="final selected_arm_slug",
    )
    if selected not in candidates:
        raise EnvelopeStop("final selected_arm_slug is not a candidate arm")
    _validate_thresholds(payload["thresholds"])
    theta_match, theta_active = candidates[selected]
    if payload["thresholds"] != {
        "theta_active": theta_active,
        "theta_match": theta_match,
    }:
        raise EnvelopeStop("final thresholds differ from the selected arm")


def _validate_final_verdict(
    payload: dict[str, Any],
    destination: DestinationSpec,
) -> None:
    result = _require_exact_object(
        payload["result"],
        _FINAL_VERDICT_RESULT_KEYS,
        label="final verdict result",
    )
    candidates = dict(destination.candidate_thresholds or ())
    if not candidates:
        raise EnvelopeStop("final-verdict destination has no candidate arms")
    verdict = _require_nonempty_string(result["verdict"], label="final verdict")
    if verdict not in {_FINAL_PASS_VERDICT, _FINAL_REJECT_VERDICT}:
        raise EnvelopeStop("final verdict is not a frozen calibration-v3 verdict")

    first_failed_gate = result["first_failed_gate"]
    if verdict == _FINAL_PASS_VERDICT:
        if first_failed_gate is not None:
            raise EnvelopeStop("final PASS requires a null first_failed_gate")
        _bind_final_selected_arm(payload, result, candidates)
        return

    if type(first_failed_gate) is not str or first_failed_gate not in {
        *_FINAL_PRE_SELECTION_FAILURE_GATES,
        *_FINAL_POST_SELECTION_FAILURE_GATES,
    }:
        raise EnvelopeStop("final REJECT has no frozen first_failed_gate")
    if first_failed_gate in _FINAL_PRE_SELECTION_FAILURE_GATES:
        if result["selected_arm_slug"] is not None or payload["thresholds"] is not None:
            raise EnvelopeStop("pre-selection final REJECT requires null arm and thresholds")
        return
    _bind_final_selected_arm(payload, result, candidates)


def _validate_r3_result(payload: dict[str, Any], destination: DestinationSpec) -> None:
    _bind_thresholds(payload, destination)
    _require_object(payload["result"], label="result")
    result = _require_exact_object(
        payload["result"], _R3_F0_RESULT_KEYS, label="r3_f0_train result"
    )
    if result["kind"] != "r3_f0_train":
        raise EnvelopeStop("r3_f0_train result kind is not exact")
    _require_nonempty_string(result["arm_slug"], label="r3_f0_train arm_slug")
    if destination.arm_slug is None or result["arm_slug"] != destination.arm_slug:
        raise EnvelopeStop("R3 arm_slug does not match the frozen result destination")
    _require_nonempty_string(result["verdict"], label="r3_f0_train verdict")
    for field_name in (
        "comparisons",
        "reference",
        "structural_metrics",
        "train_input",
    ):
        _require_object(result[field_name], label=f"r3_f0_train {field_name}")
    if type(result["resumes"]) is not list:
        raise EnvelopeStop("r3_f0_train resumes must be a JSON array")


def _validate_result_payload(
    payload: dict[str, Any],
    *,
    destination: DestinationSpec,
) -> None:
    _validate_common_payload(payload)
    if destination.kind not in {
        ArtifactDestinationKind.DEVELOPMENT_SELECTION,
        ArtifactDestinationKind.ELECTRICAL_RESULT,
        ArtifactDestinationKind.FINAL_VERDICT,
        ArtifactDestinationKind.Q4_RESULT,
        ArtifactDestinationKind.Q255_RESULT,
        ArtifactDestinationKind.R3_RESULT,
    }:
        raise EnvelopeStop("result payload requires a frozen result destination")
    _bind_screen_phase(payload, destination)
    _bind_result_construction_seeds(payload, destination)
    if destination.kind is ArtifactDestinationKind.ELECTRICAL_RESULT:
        _validate_electrical_result(payload, destination)
    elif destination.kind is ArtifactDestinationKind.Q4_RESULT:
        _validate_q4_result(payload, destination)
    elif destination.kind is ArtifactDestinationKind.Q255_RESULT:
        _validate_q255_result(payload, destination)
    elif destination.kind is ArtifactDestinationKind.DEVELOPMENT_SELECTION:
        _validate_development_selection(payload, destination)
    elif destination.kind is ArtifactDestinationKind.FINAL_VERDICT:
        _validate_final_verdict(payload, destination)
    else:
        _validate_r3_result(payload, destination)


def _validate_payload(
    artifact_id: str,
    payload: object,
    *,
    destination: DestinationSpec,
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise EnvelopeStop("payload must be a JSON object")
    expected = _payload_keys(artifact_id)
    _require_exact_keys(payload, expected, label="payload")
    validated = dict(payload)
    if artifact_id == RESUME_ARTIFACT_ID:
        _validate_resume_payload(validated, destination=destination)
    else:
        _validate_result_payload(validated, destination=destination)
    return validated


def _decode_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise EnvelopeStop(f"duplicate JSON object key: {key}")
        value[key] = child
    return value


def _reject_float(_value: str) -> object:
    raise EnvelopeStop("floating-point JSON values are forbidden")


def _reject_constant(_value: str) -> object:
    raise EnvelopeStop("non-finite JSON constants are forbidden")


def _decode_canonical_envelope(
    raw: bytes,
    *,
    destination: DestinationSpec,
) -> ArtifactEnvelope:
    if type(raw) is not bytes:
        raise EnvelopeStop("envelope input must be exact bytes")
    if len(raw) > MAX_ENVELOPE_BYTES:
        raise EnvelopeStop("envelope exceeds the one-mebibyte bound")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_decode_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except EnvelopeStop:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise EnvelopeStop("envelope is not strict UTF-8 JSON") from exc
    if type(value) is not dict or frozenset(value) != _OUTER_KEYS:
        raise EnvelopeStop("outer envelope fields differ from the frozen schema")
    artifact_id = value["artifact_id"]
    if type(artifact_id) is not str:
        raise EnvelopeStop("artifact_id must be a string")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise EnvelopeStop("schema_version must be integer 1")
    payload = _validate_payload(
        artifact_id,
        value["payload"],
        destination=destination,
    )
    payload_sha256 = value["payload_sha256"]
    if (
        type(payload_sha256) is not str
        or len(payload_sha256) != 64
        or any(character not in "0123456789abcdef" for character in payload_sha256)
    ):
        raise EnvelopeStop("payload_sha256 must be a lowercase SHA-256 identity")
    observed_payload_sha256 = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if payload_sha256 != observed_payload_sha256:
        raise EnvelopeStop("payload_sha256 does not bind the canonical payload")
    if _canonical_json_bytes(value) != raw:
        raise EnvelopeStop("envelope bytes are not the exact canonical encoding")
    return ArtifactEnvelope(
        artifact_id=artifact_id,
        payload=payload,
        payload_sha256=payload_sha256,
        schema_version=1,
    )


class CalibrationV3ArtifactStore:
    """One deep module for the protocol-supplied immutable artifact namespace."""

    def __init__(
        self,
        *,
        repository_root: Path,
        artifact_root: str,
        destinations: Mapping[str, DestinationSpec],
        protocol_sha256: str,
        source_binding_bytes: bytes,
        claim_boundary_bytes: bytes,
        source_binding_runnable: bool,
        expected_source_bundle_identity_sha256: str,
        durable_parent_directories: tuple[str, ...],
    ) -> None:
        self._repository_root = repository_root
        self._artifact_root = artifact_root
        self._destinations = dict(destinations)
        self._artifact_paths = frozenset(destinations)
        self._protocol_sha256 = protocol_sha256
        self._source_binding_bytes = source_binding_bytes
        self._claim_boundary_bytes = claim_boundary_bytes
        self._source_binding_runnable = source_binding_runnable
        self._expected_source_bundle_identity_sha256 = expected_source_bundle_identity_sha256
        self._durable_parent_directories = durable_parent_directories

    @classmethod
    def open(
        cls,
        *,
        repository_root: Path,
        validated_protocol: dict[str, object],
        expected_source_bundle_identity_sha256: str | None = None,
    ) -> CalibrationV3ArtifactStore:
        if expected_source_bundle_identity_sha256 is None:
            raise ArtifactAuthorityStop(
                "official store requires an external source-bundle identity"
            )
        if (
            type(expected_source_bundle_identity_sha256) is not str
            or len(expected_source_bundle_identity_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_source_bundle_identity_sha256
            )
        ):
            raise ArtifactAuthorityStop(
                "external source-bundle identity must be one lowercase SHA-256"
            )
        if type(validated_protocol) is not dict:
            raise ArtifactAuthorityStop("validated protocol must be an exact object")
        contract = validated_protocol.get("artifact_contract")
        if type(contract) is not dict:
            raise ArtifactPathStop("validated protocol has no artifact contract")
        artifact_root = _validate_relative_path(contract.get("root"), label="artifact root")
        paths = _collect_artifact_paths(contract, artifact_root=artifact_root)
        direct_parents, parent_closure = _artifact_parent_directories(
            paths,
            artifact_root=artifact_root,
        )
        inventory = _protocol_object(
            contract.get("path_inventory"),
            label="artifact_contract.path_inventory",
        )
        if (
            type(inventory.get("direct_parent_directories")) is not int
            or inventory["direct_parent_directories"] != len(direct_parents)
            or type(inventory.get("full_parent_directory_closure")) is not int
            or inventory["full_parent_directory_closure"] != len(parent_closure)
        ):
            raise ArtifactAuthorityStop(
                "artifact parent-directory inventory differs from frozen paths"
            )
        destinations = _derive_destination_specs(
            validated_protocol,
            contract=contract,
            artifact_root=artifact_root,
            artifact_paths=paths,
        )
        source_binding = _protocol_object(
            validated_protocol.get("source_binding"),
            label="source_binding",
        )
        if type(source_binding.get("runnable")) is not bool:
            raise ArtifactAuthorityStop("protocol source_binding.runnable must be exact bool")
        result_contract = _protocol_object(
            validated_protocol.get("result_contract"),
            label="result_contract",
        )
        if result_contract.get("artifact_creation_requires_runnable_source_binding") is not True:
            raise ArtifactAuthorityStop("protocol does not require runnable artifact creation")
        claim_boundary = validated_protocol.get("claim_boundary")
        try:
            _validate_claim_boundary(claim_boundary)
            protocol_bytes = _canonical_json_bytes(validated_protocol)
            source_binding_bytes = _canonical_json_bytes(source_binding)
            claim_boundary_bytes = _canonical_json_bytes(claim_boundary)
        except EnvelopeStop as exc:
            raise ArtifactAuthorityStop("validated protocol authority is not canonical") from exc
        root = Path(repository_root).absolute()
        try:
            root_stat = os.lstat(root)
        except OSError as exc:
            raise ArtifactPathStop("repository root is unavailable") from exc
        if not _is_plain_directory(root_stat):
            raise ArtifactPathStop("repository root must be a plain directory")
        return cls(
            repository_root=root,
            artifact_root=artifact_root,
            destinations=destinations,
            protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
            source_binding_bytes=source_binding_bytes,
            claim_boundary_bytes=claim_boundary_bytes,
            source_binding_runnable=source_binding["runnable"],
            expected_source_bundle_identity_sha256=(expected_source_bundle_identity_sha256),
            durable_parent_directories=parent_closure,
        )

    def durable_directory_requests(self) -> tuple[tuple[Path, str], ...]:
        """Return the exact root-relative directory closure in parent-first order."""

        return tuple(
            (self._repository_root, relative_directory)
            for relative_directory in self._durable_parent_directories
        )

    def _require_member(self, relative_path: object) -> str:
        path = _validate_relative_path(relative_path, label="artifact path")
        if path not in self._artifact_paths:
            raise ArtifactPathStop("artifact path is not in the frozen protocol namespace")
        return path

    def _destination(self, relative_path: object) -> DestinationSpec:
        member = self._require_member(relative_path)
        return self._destinations[member]

    def _authorize_json_destination(
        self,
        destination: DestinationSpec,
        *,
        artifact_id: str | None,
        encode: bool,
    ) -> None:
        if destination.kind is ArtifactDestinationKind.BINARY_CHECKPOINT:
            raise ArtifactPathStop("binary checkpoint destinations have no JSON codec")
        if destination.kind is ArtifactDestinationKind.UNFROZEN_RESULT:
            raise ArtifactAuthorityStop(
                "artifact destination has no frozen official result descriptor"
            )
        expected_id = (
            RESUME_ARTIFACT_ID
            if destination.kind is ArtifactDestinationKind.RESUME_MANIFEST
            else RESULT_ARTIFACT_ID
        )
        if artifact_id is not None and artifact_id != expected_id:
            raise ArtifactPathStop("artifact ID does not match the destination type")
        if encode and not self._source_binding_runnable:
            raise ArtifactAuthorityStop(
                "official artifact encoding requires runnable protocol source binding"
            )

    def _bind_payload_authorities(self, payload: dict[str, Any]) -> None:
        if payload["protocol_sha256"] != self._protocol_sha256:
            raise EnvelopeStop("protocol_sha256 does not bind the validated protocol")
        if _canonical_json_bytes(payload["source_binding"]) != self._source_binding_bytes:
            raise EnvelopeStop("source_binding does not bind the validated protocol")
        if _canonical_json_bytes(payload["claim_boundary"]) != self._claim_boundary_bytes:
            raise EnvelopeStop("claim_boundary does not bind the validated protocol")
        if payload["source_bundle_identity_sha256"] != self._expected_source_bundle_identity_sha256:
            raise EnvelopeStop("source_bundle_identity_sha256 does not bind the external authority")

    def _absolute_member(self, member: str) -> Path:
        return self._repository_root.joinpath(*member.split("/"))

    def _require_parent_chain(self, member: str) -> bool:
        current = self._repository_root
        for part in member.split("/")[:-1]:
            current = current / part
            try:
                observed = os.lstat(current)
            except FileNotFoundError:
                return False
            except OSError as exc:
                raise ArtifactPathStop(f"artifact parent is unavailable: {current}") from exc
            if not _is_plain_directory(observed):
                raise ArtifactPathStop(f"artifact parent is not a plain directory: {current}")
        return True

    def require_r3_result_destination(self, relative_path: str) -> DestinationSpec:
        """Return the exact frozen R3 result descriptor without creating files."""

        destination = self._destination(relative_path)
        if (
            destination.kind is not ArtifactDestinationKind.R3_RESULT
            or destination.screen_id != "r3_f0_train"
            or destination.phase not in {"development", "deterministic_confirmation"}
            or destination.arm_slug is None
            or destination.thresholds is None
            or destination.construction_seeds is None
            or len(destination.construction_seeds) != 1
        ):
            raise ArtifactPathStop("destination is not one exact frozen R3 result")
        return destination

    def require_result_destination(self, relative_path: str) -> DestinationSpec:
        """Return one exact official JSON-result descriptor without creating files."""

        destination = self._destination(relative_path)
        if destination.kind not in {
            ArtifactDestinationKind.DEVELOPMENT_SELECTION,
            ArtifactDestinationKind.ELECTRICAL_RESULT,
            ArtifactDestinationKind.FINAL_VERDICT,
            ArtifactDestinationKind.Q4_RESULT,
            ArtifactDestinationKind.Q255_RESULT,
            ArtifactDestinationKind.R3_RESULT,
        }:
            raise ArtifactPathStop("destination is not an official calibration-v3 result")
        return destination

    def prepare_absent_result(self, relative_path: str) -> Path:
        """Require one prebuilt result parent and prove its exact final absent."""

        if not self._source_binding_runnable:
            raise ArtifactAuthorityStop(
                "result preparation requires runnable protocol source binding"
            )
        destination = self.require_result_destination(relative_path)
        if not self._require_parent_chain(destination.relative_path):
            raise ArtifactPathStop("result parent namespace was not prepared before publication")
        if self._plain_file_presence(destination.relative_path):
            raise ArtifactPathStop("result final already exists")
        result_path = self._absolute_member(destination.relative_path)
        if self._plain_file_presence(destination.relative_path):
            raise ArtifactPathStop("result final already exists")
        return result_path

    def _resume_pair_destinations(
        self,
        manifest_relative_path: str,
    ) -> tuple[DestinationSpec, DestinationSpec]:
        manifest = self._destination(manifest_relative_path)
        if (
            manifest.kind is not ArtifactDestinationKind.RESUME_MANIFEST
            or not manifest.relative_path.endswith(".resume.json")
        ):
            raise ArtifactPathStop("resume transaction requires a manifest destination")
        checkpoint_path = manifest.relative_path.removesuffix(".resume.json") + ".r3ntcp1"
        checkpoint = self._destination(checkpoint_path)
        if (
            checkpoint.kind is not ArtifactDestinationKind.BINARY_CHECKPOINT
            or checkpoint.relative_path.removesuffix(".r3ntcp1")
            != manifest.relative_path.removesuffix(".resume.json")
            or checkpoint.screen_id != manifest.screen_id
            or checkpoint.phase != manifest.phase
            or checkpoint.arm_slug != manifest.arm_slug
            or checkpoint.thresholds != manifest.thresholds
            or checkpoint.construction_seed != manifest.construction_seed
            or checkpoint.resume_cursor != manifest.resume_cursor
        ):
            raise ArtifactPathStop("resume pair descriptors do not share exact semantics")
        return checkpoint, manifest

    def require_resume_manifest_destination(
        self,
        manifest_relative_path: str,
    ) -> DestinationSpec:
        """Return one exact resume descriptor without opening or creating files."""

        _checkpoint, manifest = self._resume_pair_destinations(manifest_relative_path)
        return manifest

    def require_resume_manifest_for(
        self,
        *,
        arm_slug: str,
        after_transition_event: int,
    ) -> DestinationSpec:
        """Return the unique frozen manifest for one arm and milestone event."""

        if type(arm_slug) is not str or not arm_slug:
            raise ArtifactPathStop("resume arm slug must be one exact non-empty string")
        if type(after_transition_event) is not int or after_transition_event < 0:
            raise ArtifactPathStop("resume milestone event must be one non-negative exact int")
        matches = tuple(
            destination
            for destination in self._destinations.values()
            if destination.kind is ArtifactDestinationKind.RESUME_MANIFEST
            and destination.arm_slug == arm_slug
            and destination.resume_cursor is not None
            and dict(destination.resume_cursor).get("after_transition_event")
            == after_transition_event
        )
        if len(matches) != 1:
            raise ArtifactPathStop("resume arm and milestone do not select one frozen manifest")
        return matches[0]

    def prepare_absent_resume_pair(
        self,
        manifest_relative_path: str,
    ) -> PreparedResumePair:
        """Require a prebuilt parent namespace and prove both exact finals absent."""

        if not self._source_binding_runnable:
            raise ArtifactAuthorityStop(
                "resume pair preparation requires runnable protocol source binding"
            )
        checkpoint, manifest = self._resume_pair_destinations(manifest_relative_path)

        if not self._require_parent_chain(
            checkpoint.relative_path
        ) or not self._require_parent_chain(manifest.relative_path):
            raise ArtifactPathStop(
                "resume pair parent namespace was not prepared before active time"
            )

        checkpoint_exists = self._plain_file_presence(checkpoint.relative_path)
        manifest_exists = self._plain_file_presence(manifest.relative_path)
        if checkpoint_exists or manifest_exists:
            if checkpoint_exists and manifest_exists:
                state = ArtifactPairState.COMMITTED
            elif checkpoint_exists:
                state = ArtifactPairState.CHECKPOINT_ONLY
            else:
                state = ArtifactPairState.MANIFEST_ONLY
            raise ArtifactPairStop(state)

        checkpoint_path = self._absolute_member(checkpoint.relative_path)
        manifest_path = self._absolute_member(manifest.relative_path)
        checkpoint_exists = self._plain_file_presence(checkpoint.relative_path)
        manifest_exists = self._plain_file_presence(manifest.relative_path)
        if checkpoint_exists or manifest_exists:
            if checkpoint_exists and manifest_exists:
                state = ArtifactPairState.COMMITTED
            elif checkpoint_exists:
                state = ArtifactPairState.CHECKPOINT_ONLY
            else:
                state = ArtifactPairState.MANIFEST_ONLY
            raise ArtifactPairStop(state)
        return PreparedResumePair(
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
        )

    def open_committed_resume_pair(
        self,
        manifest_relative_path: str,
    ) -> ManifestCommittedResumePair:
        """Canonical-read the manifest first, then require its paired checkpoint."""

        checkpoint, manifest = self._resume_pair_destinations(manifest_relative_path)
        raw, manifest_identity = self._read_bounded_regular_file_with_identity(
            manifest.relative_path
        )
        envelope = self.decode(manifest.relative_path, raw)
        if not self._plain_file_presence(checkpoint.relative_path):
            raise ArtifactPairStop(ArtifactPairState.MANIFEST_ONLY)
        return ManifestCommittedResumePair(
            checkpoint_path=self._absolute_member(checkpoint.relative_path),
            manifest=envelope,
            manifest_bytes=raw,
            manifest_pin=ManifestFinalPin(
                manifest_path=self._absolute_member(manifest.relative_path),
                _manifest_relative_path=manifest.relative_path,
                _filesystem_identity=manifest_identity,
                _manifest_sha256=hashlib.sha256(raw).hexdigest(),
            ),
        )

    def confirm_committed_resume_pair(
        self,
        manifest_relative_path: str,
        expected_manifest_bytes: bytes,
    ) -> ManifestCommittedResumePair:
        """Reopen exact manifest bytes first, then re-require the checkpoint final."""

        if type(expected_manifest_bytes) is not bytes:
            raise EnvelopeStop("expected immutable bytes must be exact bytes")
        checkpoint, manifest = self._resume_pair_destinations(manifest_relative_path)
        observed, manifest_identity = self._read_bounded_regular_file_with_identity(
            manifest.relative_path
        )
        if observed != expected_manifest_bytes:
            raise EnvelopeStop("published resume manifest differs from expected bytes")
        envelope = self.decode(manifest.relative_path, observed)
        if not self._plain_file_presence(checkpoint.relative_path):
            raise ArtifactPairStop(ArtifactPairState.MANIFEST_ONLY)
        return ManifestCommittedResumePair(
            checkpoint_path=self._absolute_member(checkpoint.relative_path),
            manifest=envelope,
            manifest_bytes=observed,
            manifest_pin=ManifestFinalPin(
                manifest_path=self._absolute_member(manifest.relative_path),
                _manifest_relative_path=manifest.relative_path,
                _filesystem_identity=manifest_identity,
                _manifest_sha256=hashlib.sha256(observed).hexdigest(),
            ),
        )

    def pin_checkpoint_final(self, manifest_relative_path: str) -> CheckpointFinalPin:
        """Pin the exact paired checkpoint before native verification and commit."""

        checkpoint, _manifest = self._resume_pair_destinations(manifest_relative_path)
        identity = self._stable_plain_file_identity(checkpoint.relative_path)
        return CheckpointFinalPin(
            checkpoint_path=self._absolute_member(checkpoint.relative_path),
            _checkpoint_relative_path=checkpoint.relative_path,
            _filesystem_identity=identity,
        )

    def confirm_checkpoint_final(self, pin: CheckpointFinalPin) -> Path:
        """Require that a pinned checkpoint final still has its exact identity."""

        if type(pin) is not CheckpointFinalPin:
            raise TypeError("checkpoint final pin must be an exact CheckpointFinalPin")
        destination = self._destination(pin._checkpoint_relative_path)
        expected_path = self._absolute_member(destination.relative_path)
        if (
            destination.kind is not ArtifactDestinationKind.BINARY_CHECKPOINT
            or pin.checkpoint_path != expected_path
        ):
            raise ArtifactPathStop("checkpoint final pin is outside this frozen store")
        observed = self._stable_plain_file_identity(destination.relative_path)
        if observed != pin._filesystem_identity:
            raise ArtifactPathStop("checkpoint final identity changed after native verification")
        return expected_path

    def confirm_manifest_final(self, pin: ManifestFinalPin) -> Path:
        """Require that the exact bounded manifest read before import is unchanged."""

        if type(pin) is not ManifestFinalPin:
            raise TypeError("manifest final pin must be an exact ManifestFinalPin")
        destination = self._destination(pin._manifest_relative_path)
        expected_path = self._absolute_member(destination.relative_path)
        if (
            destination.kind is not ArtifactDestinationKind.RESUME_MANIFEST
            or pin.manifest_path != expected_path
        ):
            raise ArtifactPathStop("manifest final pin is outside this frozen store")
        observed, identity = self._read_bounded_regular_file_with_identity(
            destination.relative_path
        )
        if (
            identity != pin._filesystem_identity
            or hashlib.sha256(observed).hexdigest() != pin._manifest_sha256
        ):
            raise ArtifactPathStop("manifest final identity changed during native import")
        return expected_path

    def confirm_exact(self, relative_path: str, expected_bytes: bytes) -> ArtifactEnvelope:
        """Bounded-reopen exact JSON bytes without reinterpreting publisher state."""

        if type(expected_bytes) is not bytes:
            raise EnvelopeStop("expected immutable bytes must be exact bytes")
        destination = self._destination(relative_path)
        self._authorize_json_destination(destination, artifact_id=None, encode=False)
        observed = self._read_bounded_regular_file(destination.relative_path)
        if observed != expected_bytes:
            raise EnvelopeStop("published artifact bytes differ from the expected envelope")
        return self.decode(destination.relative_path, observed)

    def _plain_file_presence(self, member: str) -> bool:
        if not self._require_parent_chain(member):
            return False
        destination = self._absolute_member(member)
        try:
            observed = os.lstat(destination)
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise ArtifactPathStop(f"artifact final is unavailable: {destination}") from exc
        if not _is_plain_regular_file(observed):
            raise ArtifactPathStop(f"artifact final is not a plain regular file: {destination}")
        return True

    @staticmethod
    def _filesystem_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )

    def _stable_plain_file_identity(self, member: str) -> tuple[int, int, int, int, int]:
        if not self._require_parent_chain(member):
            raise ArtifactPathStop("artifact parent does not exist")
        destination = self._absolute_member(member)
        try:
            before = os.lstat(destination)
        except OSError as exc:
            raise ArtifactPathStop("artifact final is unavailable") from exc
        if not _is_plain_regular_file(before):
            raise ArtifactPathStop("artifact final is not a plain regular file")

        with _read_only_descriptor(destination) as descriptor:
            opened = os.fstat(descriptor)
            if not _is_plain_regular_file(opened):
                raise ArtifactPathStop("opened artifact is not a plain regular file")
            opened_identity = self._filesystem_identity(opened)
            if opened_identity != self._filesystem_identity(before):
                raise ArtifactPathStop("artifact final changed during safe open")
            try:
                after = os.lstat(destination)
            except OSError as exc:
                raise ArtifactPathStop("artifact final changed during identity pin") from exc
            if (
                not _is_plain_regular_file(after)
                or self._filesystem_identity(after) != opened_identity
            ):
                raise ArtifactPathStop("artifact final changed during identity pin")
            return opened_identity

    def _read_bounded_regular_file_with_identity(
        self,
        member: str,
    ) -> tuple[bytes, tuple[int, int, int, int, int]]:
        if not self._require_parent_chain(member):
            raise ArtifactPathStop("artifact parent does not exist")
        destination = self._absolute_member(member)
        try:
            before = os.lstat(destination)
        except OSError as exc:
            raise ArtifactPathStop("artifact final is unavailable") from exc
        if not _is_plain_regular_file(before):
            raise ArtifactPathStop("artifact final is not a plain regular file")
        if before.st_size > MAX_ENVELOPE_BYTES:
            raise ArtifactPathStop("artifact final exceeds the one-mebibyte bound")

        with _read_only_descriptor(destination) as descriptor:
            opened = os.fstat(descriptor)
            if not _is_plain_regular_file(opened):
                raise ArtifactPathStop("opened artifact is not a plain regular file")
            opened_identity = self._filesystem_identity(opened)
            if opened_identity != self._filesystem_identity(before):
                raise ArtifactPathStop("artifact final changed during safe open")
            chunks = bytearray()
            while True:
                chunk = os.read(descriptor, min(65_536, MAX_ENVELOPE_BYTES + 1 - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
                if len(chunks) > MAX_ENVELOPE_BYTES:
                    raise ArtifactPathStop("artifact final exceeds the one-mebibyte bound")
            after = os.fstat(descriptor)
            after_identity = self._filesystem_identity(after)
            if after_identity != opened_identity or len(chunks) != after.st_size:
                raise ArtifactPathStop("artifact final changed during bounded read")
            try:
                named_after = os.lstat(destination)
            except OSError as exc:
                raise ArtifactPathStop("artifact final changed during bounded read") from exc
            if (
                not _is_plain_regular_file(named_after)
                or self._filesystem_identity(named_after) != after_identity
            ):
                raise ArtifactPathStop("artifact final changed during bounded read")
            return bytes(chunks), after_identity

    def _read_bounded_regular_file(self, member: str) -> bytes:
        observed, _identity = self._read_bounded_regular_file_with_identity(member)
        return observed

    def read(self, relative_path: str) -> ArtifactEnvelope:
        destination = self._destination(relative_path)
        self._authorize_json_destination(destination, artifact_id=None, encode=False)
        member = destination.relative_path
        return self.decode(member, self._read_bounded_regular_file(member))

    def _encode(self, artifact_id: str, payload: object) -> bytes:
        if type(payload) is not dict:
            raise EnvelopeStop("payload must be a JSON object")
        expected_keys = _payload_keys(artifact_id)
        _require_exact_keys(payload, expected_keys, label="payload")
        destination = self._destination(payload["artifact_path"])
        self._authorize_json_destination(
            destination,
            artifact_id=artifact_id,
            encode=True,
        )
        validated_payload = _validate_payload(
            artifact_id,
            payload,
            destination=destination,
        )
        self._bind_payload_authorities(validated_payload)
        payload_bytes = _canonical_json_bytes(validated_payload)
        value = {
            "artifact_id": artifact_id,
            "payload": validated_payload,
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "schema_version": 1,
        }
        raw = _canonical_json_bytes(value)
        if len(raw) > MAX_ENVELOPE_BYTES:
            raise EnvelopeStop("envelope exceeds the one-mebibyte bound")
        return raw

    def encode_resume(self, payload: dict[str, object]) -> bytes:
        return self._encode(RESUME_ARTIFACT_ID, payload)

    def encode_result(self, payload: dict[str, object]) -> bytes:
        return self._encode(RESULT_ARTIFACT_ID, payload)

    def decode(self, relative_path: str, raw: bytes) -> ArtifactEnvelope:
        destination = self._destination(relative_path)
        self._authorize_json_destination(destination, artifact_id=None, encode=False)
        member = destination.relative_path
        envelope = _decode_canonical_envelope(raw, destination=destination)
        if envelope.payload["artifact_path"] != member:
            raise EnvelopeStop("payload artifact_path does not bind the requested member")
        expected_id = (
            RESUME_ARTIFACT_ID
            if destination.kind is ArtifactDestinationKind.RESUME_MANIFEST
            else RESULT_ARTIFACT_ID
        )
        if envelope.artifact_id != expected_id:
            raise EnvelopeStop("artifact ID does not match the frozen destination type")
        self._bind_payload_authorities(envelope.payload)
        return envelope
