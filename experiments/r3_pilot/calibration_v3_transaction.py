"""Fail-closed checkpoint/resume transactions for calibration-v3.

Importing this module is transport-only.  Native execution is loaded lazily,
after a nominal runnable authority and its future circuit-identity map pass all
pure checks.  The official runner remains disabled, so this module cannot run a
fixture or publish an outcome under the current schema-four protocol.

The same-process trust boundary is explicit: the future official executor must
retain the coordinator privately and give scientific control flow only a
``CalibrationV3Session`` or ``CommittedResumeCheckpoint``.  Reflective access
to underscored fields, direct native-runtime imports, monkeypatching, and
``object.__setattr__`` are equivalent to modifying trusted harness source and
are outside that boundary.  Within the supported public seam, every
caller-held receipt is untrusted and the coordinator retains all authoritative
state.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import struct
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

if __package__:
    from . import calibration_v3_persistence as persistence
    from .calibration_v3_authority import FrozenPlanAuthority
else:  # pragma: no cover - script-directory import support.
    import calibration_v3_persistence as persistence
    from calibration_v3_authority import FrozenPlanAuthority


_IDENTITY_FIELDS = (
    "format_version",
    "header_bytes",
    "checkpoint_bytes",
    "body_bytes",
    "protocol_identity_sha256",
    "source_bundle_identity_sha256",
    "circuit_identity_sha256",
    "body_sha256",
    "semantic_index_sha256",
    "segment_count",
    "cohort_count",
    "next_reserve_cell",
    "event_count",
    "learning_step",
    "prior_context_length",
    "winner_context_length",
    "poison_code",
    "logical_state_sha256",
)
_IDENTITY_SHA_FIELDS = frozenset(
    {
        "protocol_identity_sha256",
        "source_bundle_identity_sha256",
        "circuit_identity_sha256",
        "body_sha256",
        "semantic_index_sha256",
        "logical_state_sha256",
    }
)
_STEP_FIELDS = frozenset(
    {
        "token_id",
        "punished_segments",
        "event_count",
        "predicted",
        "predicted_columns",
        "allocated_segments",
        "inhibition_onset_ticks",
        "column_simulated_ticks",
        "winner_spikes",
        "basket_spikes",
        "proximal_spikes",
        "proximal_relay_spikes",
        "winner_relay_spikes",
        "winner_context",
        "proximal_context",
        "context",
    }
)
_SPIKE_FIELDS = (
    "winner_spikes",
    "basket_spikes",
    "proximal_spikes",
    "proximal_relay_spikes",
    "winner_relay_spikes",
)
_PUBLICATION_FIELDS = frozenset(
    {
        "outcome",
        "temp_cleanup",
        "primary_error",
        "cleanup_errors",
        "owned_temp_pid",
        "owned_temp_nonce",
        "checkpoint_bytes",
        "encoded_bytes",
    }
)
_ERROR_FIELDS = frozenset({"stage", "domain", "code"})
_STREAM_FIELDS = frozenset({"winner", "step"})
_CHAIN_FIELDS = frozenset({"count", "sha256"})
_MAX_AUTHORITY_BYTES = 1 << 20
_MAX_CLEANUP_ERRORS = 4
_MAX_MANIFEST_BYTES = 1 << 20
_CHECKPOINT_FORMAT_VERSION = 1
_CHECKPOINT_HEADER_BYTES = 256
_CHECKPOINT_CONTEXT_CELLS = 22
_CHECKPOINT_PAIRS_PER_SEGMENT = 10
_CHECKPOINT_LOGICAL_STATE_DOMAIN = b"humanbrain-r3-checkpoint-logical-state-v1\x00"
_UINT8_MAX = (1 << 8) - 1
_UINT16_MAX = (1 << 16) - 1
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_MAX_STOP_MESSAGE_CHARACTERS = 512


class TransactionTimingState(StrEnum):
    """Authoritative timing status carried by every transaction STOP."""

    NOT_STARTED = "NOT_STARTED"
    WITHIN_CAP = "WITHIN_CAP"
    OVER_CAP = "OVER_CAP"
    TIMING_INDETERMINATE = "TIMING_INDETERMINATE"


@dataclass(frozen=True, slots=True)
class CalibrationV3CleanupFailure:
    """Bounded operational identity for one failed native close attempt."""

    stage: str
    error_type: str


@dataclass(frozen=True, slots=True)
class _TransactionStopReceipt:
    timing_state: TransactionTimingState
    timing: persistence.CumulativeActiveWallTiming | None
    checkpoint_publication: object | None
    manifest_publication: object | None
    directory_preparation: object | None
    manifest_reconciliation: object | None
    cleanup_failures: tuple[CalibrationV3CleanupFailure, ...]


class CalibrationV3TransactionStop(RuntimeError):
    """A fail-closed transaction, timing, authority, or persistence STOP."""

    __slots__ = ("_receipt",)

    _PROTECTED_RECEIPT_FIELDS = frozenset(
        {
            "_checkpoint_publication",
            "_cleanup_failures",
            "_directory_preparation",
            "_manifest_publication",
            "_manifest_reconciliation",
            "_receipt",
            "_timing",
            "_timing_state",
        }
    )

    def __init__(
        self,
        message: str,
        *,
        timing_state: TransactionTimingState = TransactionTimingState.NOT_STARTED,
        timing: persistence.CumulativeActiveWallTiming | None = None,
        checkpoint_publication: object | None = None,
        manifest_publication: object | None = None,
        directory_preparation: object | None = None,
        manifest_reconciliation: object | None = None,
        cleanup_failures: tuple[CalibrationV3CleanupFailure, ...] = (),
    ) -> None:
        if type(message) is not str:
            raise TypeError("transaction STOP message must be exact text")
        if type(timing_state) is not TransactionTimingState:
            raise TypeError("timing_state must be an exact TransactionTimingState")
        if timing is not None and type(timing) is not persistence.CumulativeActiveWallTiming:
            raise TypeError("timing must be an exact CumulativeActiveWallTiming or None")
        if type(cleanup_failures) is not tuple or any(
            type(failure) is not CalibrationV3CleanupFailure for failure in cleanup_failures
        ):
            raise TypeError("cleanup_failures must contain exact cleanup failure records")
        self._receipt = _TransactionStopReceipt(
            timing_state=timing_state,
            timing=timing,
            checkpoint_publication=checkpoint_publication,
            manifest_publication=manifest_publication,
            directory_preparation=directory_preparation,
            manifest_reconciliation=manifest_reconciliation,
            cleanup_failures=cleanup_failures,
        )
        super().__init__(message)

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._PROTECTED_RECEIPT_FIELDS and hasattr(self, "_receipt"):
            raise AttributeError("transaction STOP receipt is immutable")
        super().__setattr__(name, value)

    @property
    def timing_state(self) -> TransactionTimingState:
        return self._receipt.timing_state

    @property
    def timing(self) -> persistence.CumulativeActiveWallTiming | None:
        return self._receipt.timing

    @property
    def checkpoint_publication(self) -> object | None:
        return self._receipt.checkpoint_publication

    @property
    def manifest_publication(self) -> object | None:
        return self._receipt.manifest_publication

    @property
    def directory_preparation(self) -> object | None:
        return self._receipt.directory_preparation

    @property
    def manifest_reconciliation(self) -> object | None:
        return self._receipt.manifest_reconciliation

    @property
    def cleanup_failures(self) -> tuple[CalibrationV3CleanupFailure, ...]:
        return self._receipt.cleanup_failures


class _CoordinatorState(StrEnum):
    READY = "READY"
    STARTING = "STARTING"
    ACTIVE = "ACTIVE"
    COMMITTING = "COMMITTING"
    RECORDING = "RECORDING"
    SEALED = "SEALED"
    TERMINAL_STOP = "TERMINAL_STOP"
    TERMINAL_OVER_CAP = "TERMINAL_OVER_CAP"
    CLOSED = "CLOSED"


@dataclass(frozen=True, slots=True)
class _RunCoordinates:
    screen_id: str
    phase: str
    arm_slug: str
    thresholds: tuple[int, int]
    construction_seed: int

    @classmethod
    def from_destination(cls, destination: persistence.DestinationSpec) -> _RunCoordinates:
        if (
            destination.screen_id is None
            or destination.phase is None
            or destination.arm_slug is None
            or destination.thresholds is None
        ):
            raise CalibrationV3TransactionStop("destination run coordinates are incomplete")
        if destination.construction_seed is not None and destination.construction_seeds is None:
            seed = destination.construction_seed
        elif (
            destination.construction_seed is None
            and destination.construction_seeds is not None
            and len(destination.construction_seeds) == 1
        ):
            seed = destination.construction_seeds[0]
        else:
            raise CalibrationV3TransactionStop("destination lacks one exact construction seed")
        return cls(
            screen_id=destination.screen_id,
            phase=destination.phase,
            arm_slug=destination.arm_slug,
            thresholds=destination.thresholds,
            construction_seed=seed,
        )


def _is_lower_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_json_tree(value: object, *, depth: int = 0) -> None:
    if depth > 128:
        raise CalibrationV3TransactionStop("authority protocol JSON is too deeply nested")
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is list:
        for child in value:
            _validate_json_tree(child, depth=depth + 1)
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise CalibrationV3TransactionStop("authority protocol keys must be exact strings")
        for child in value.values():
            _validate_json_tree(child, depth=depth + 1)
        return
    raise CalibrationV3TransactionStop("authority protocol is not exact JSON data")


def _canonical_json_bytes(value: object) -> bytes:
    _validate_json_tree(value)
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise CalibrationV3TransactionStop("authority protocol is not canonical JSON") from exc
    if len(raw) > _MAX_AUTHORITY_BYTES:
        raise CalibrationV3TransactionStop("authority protocol exceeds the one-mebibyte bound")
    return raw


def _json_snapshot(value: object) -> Any:
    try:
        return json.loads(_canonical_json_bytes(value))
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise CalibrationV3TransactionStop("authority protocol cannot be snapshotted") from exc


def _exact_object(value: object, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise CalibrationV3TransactionStop(f"{label} must be an exact object")
    return value


def _exact_keys(value: object, expected: frozenset[str], *, label: str) -> dict[str, Any]:
    exact = _exact_object(value, label=label)
    if frozenset(exact) != expected:
        raise CalibrationV3TransactionStop(f"{label} fields differ from the transaction schema")
    return exact


def _exact_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise CalibrationV3TransactionStop(f"{label} must be a non-negative exact int")
    return value


def _exact_bounded_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise CalibrationV3TransactionStop(
            f"{label} must be an exact int in [{minimum}, {maximum}]"
        )
    return value


def _native_runtime() -> Any:
    """Load native transport only after transaction authority validation."""

    try:
        return importlib.import_module("humanbrain.native_temporal_runtime")
    except Exception as exc:
        raise CalibrationV3TransactionStop("native temporal runtime is unavailable") from exc


def _safe_exception_message(primary: BaseException) -> str:
    """Extract bounded plain text without invoking caller-controlled ``__str__``."""

    try:
        arguments = BaseException.__getattribute__(primary, "args")
    except BaseException:
        arguments = ()
    if type(arguments) is tuple and arguments and type(arguments[0]) is str:
        message = arguments[0]
        if message:
            return message[:_MAX_STOP_MESSAGE_CHARACTERS]
    try:
        type_name = type.__getattribute__(type(primary), "__name__")
    except BaseException:
        type_name = "BaseException"
    if type(type_name) is not str or not type_name:
        type_name = "BaseException"
    return f"transaction operation failed: {type_name[:128]}"


@dataclass(frozen=True, slots=True)
class FrozenResumeProgress:
    """Exact resumable scientific prefix; no caller-authored dictionaries."""

    last_step: persistence.TemporalStepFrame
    winner_chain: persistence.ResumableHashChain
    step_chain: persistence.ResumableHashChain

    def __post_init__(self) -> None:
        if type(self.last_step) is not persistence.TemporalStepFrame:
            raise TypeError("last_step must be an exact TemporalStepFrame")
        if type(self.winner_chain) is not persistence.ResumableHashChain:
            raise TypeError("winner_chain must be an exact ResumableHashChain")
        if type(self.step_chain) is not persistence.ResumableHashChain:
            raise TypeError("step_chain must be an exact ResumableHashChain")
        if self.winner_chain.kind is not persistence.HashChainKind.WINNER:
            raise ValueError("winner_chain must use the WINNER domain")
        if self.step_chain.kind is not persistence.HashChainKind.STEP:
            raise ValueError("step_chain must use the STEP domain")


class CalibrationV3Session:
    """Public scientific-control capability for one active interval only."""

    __slots__ = ("_owner", "_token")

    def __init__(
        self,
        *,
        owner: CalibrationV3Transaction,
        token: object,
    ) -> None:
        self._owner = owner
        self._token = token

    @property
    def progress(self) -> FrozenResumeProgress | None:
        return self._owner._active_progress(self)

    @property
    def manifest_reconciliation(self) -> object | None:
        return self._owner._active_manifest_reconciliation(self)

    def advance_block(
        self,
        token_ids: object,
        *,
        mode: object,
        reset_before: bool = False,
    ) -> tuple[persistence.TemporalStepFrame, ...]:
        """Advance only through the transaction-owned native handle."""

        return self._owner._advance_active(
            self,
            token_ids,
            mode=mode,
            reset_before=reset_before,
        )

    def check_cap(self) -> persistence.TimingState:
        """Perform the sole permitted within-book cumulative-cap check."""

        return self._owner._check_active_session_cap(self)

    def close(self) -> None:
        self._owner._close_active_session(self)


class CommittedResumeCheckpoint:
    """Opaque single-use capability after a manifest-last checkpoint commit.

    Receipt fields are display-only.  The coordinator retains the authoritative
    timing and publication records and accepts only the exact issued object.
    """

    __slots__ = (
        "_checkpoint_publication",
        "_manifest_publication",
        "_owner",
        "_timing",
        "_token",
    )

    def __init__(
        self,
        owner: CalibrationV3Transaction,
        token: object,
        timing: persistence.CumulativeActiveWallTiming,
        checkpoint_publication: object,
        manifest_publication: object,
    ) -> None:
        self._owner = owner
        self._token = token
        self._timing = timing
        self._checkpoint_publication = checkpoint_publication
        self._manifest_publication = manifest_publication

    def __setattr__(self, name: str, value: object) -> None:
        if name in self.__slots__ and hasattr(self, name):
            raise AttributeError("committed checkpoint receipt is immutable")
        super().__setattr__(name, value)

    @property
    def timing(self) -> persistence.CumulativeActiveWallTiming:
        return self._timing

    @property
    def checkpoint_publication(self) -> object:
        return self._checkpoint_publication

    @property
    def manifest_publication(self) -> object:
        return self._manifest_publication

    def close(self) -> None:
        self._owner._close_committed(self)


def _serialize_identity(identity: object, native: Any) -> dict[str, object]:
    if type(identity) is not native.TemporalCheckpointIdentity:
        raise CalibrationV3TransactionStop("checkpoint identity has the wrong nominal type")
    if identity.format_version != _CHECKPOINT_FORMAT_VERSION:
        raise CalibrationV3TransactionStop("checkpoint identity format version is unsupported")
    if identity.header_bytes != _CHECKPOINT_HEADER_BYTES:
        raise CalibrationV3TransactionStop("checkpoint identity header length is invalid")
    checkpoint_bytes = _exact_bounded_int(
        identity.checkpoint_bytes,
        label="checkpoint identity checkpoint_bytes",
        minimum=_CHECKPOINT_HEADER_BYTES,
        maximum=min(_UINT64_MAX, sys.maxsize),
    )
    body_bytes = _exact_bounded_int(
        identity.body_bytes,
        label="checkpoint identity body_bytes",
        maximum=_UINT64_MAX,
    )
    if body_bytes != checkpoint_bytes - _CHECKPOINT_HEADER_BYTES:
        raise CalibrationV3TransactionStop("checkpoint identity framing lengths differ")
    segment_count = _exact_bounded_int(
        identity.segment_count,
        label="checkpoint identity segment_count",
        maximum=_UINT32_MAX,
    )
    cohort_count = _exact_bounded_int(
        identity.cohort_count,
        label="checkpoint identity cohort_count",
        maximum=_UINT32_MAX,
    )
    next_reserve_cell = _exact_bounded_int(
        identity.next_reserve_cell,
        label="checkpoint identity next_reserve_cell",
        maximum=_UINT32_MAX,
    )
    event_count = _exact_bounded_int(
        identity.event_count,
        label="checkpoint identity event_count",
        maximum=_UINT64_MAX,
    )
    learning_step = _exact_bounded_int(
        identity.learning_step,
        label="checkpoint identity learning_step",
        maximum=_UINT64_MAX,
    )
    prior_context_length = _exact_bounded_int(
        identity.prior_context_length,
        label="checkpoint identity prior_context_length",
        maximum=min(_UINT8_MAX, _CHECKPOINT_CONTEXT_CELLS),
    )
    winner_context_length = _exact_bounded_int(
        identity.winner_context_length,
        label="checkpoint identity winner_context_length",
        maximum=min(_UINT8_MAX, _CHECKPOINT_CONTEXT_CELLS),
    )
    poison_code = _exact_bounded_int(
        identity.poison_code,
        label="checkpoint identity poison_code",
        maximum=_UINT16_MAX,
    )
    if poison_code != 0:
        raise CalibrationV3TransactionStop("checkpoint identity is poisoned")
    expected_body_bytes = (
        cohort_count * 4
        + prior_context_length * 4
        + winner_context_length * 4
        + segment_count
        * (4 + _CHECKPOINT_PAIRS_PER_SEGMENT * 4 + _CHECKPOINT_PAIRS_PER_SEGMENT * 2)
    )
    if body_bytes != expected_body_bytes:
        raise CalibrationV3TransactionStop(
            "checkpoint identity scalar counts disagree with the body length"
        )
    for field_name in _IDENTITY_SHA_FIELDS:
        if not _is_lower_sha256(getattr(identity, field_name)):
            raise CalibrationV3TransactionStop(
                f"checkpoint identity {field_name} is not a lowercase SHA-256"
            )
    expected_logical_state = hashlib.sha256(
        _CHECKPOINT_LOGICAL_STATE_DOMAIN
        + bytes.fromhex(identity.body_sha256)
        + struct.pack(
            "<IIIQQBBH",
            segment_count,
            cohort_count,
            next_reserve_cell,
            event_count,
            learning_step,
            prior_context_length,
            winner_context_length,
            poison_code,
        )
    ).hexdigest()
    if identity.logical_state_sha256 != expected_logical_state:
        raise CalibrationV3TransactionStop("checkpoint identity logical-state digest is invalid")
    return {field_name: getattr(identity, field_name) for field_name in _IDENTITY_FIELDS}


def _serialize_error(error: object, native: Any) -> dict[str, object]:
    if type(error) is not native.TemporalCheckpointPublicationError:
        raise CalibrationV3TransactionStop("checkpoint publication error type is invalid")
    if type(error.stage) is not native.TemporalCheckpointErrorStage:
        raise CalibrationV3TransactionStop("checkpoint publication error stage is invalid")
    if type(error.domain) is not native.TemporalCheckpointErrorDomain:
        raise CalibrationV3TransactionStop("checkpoint publication error domain is invalid")
    if (
        error.stage is native.TemporalCheckpointErrorStage.NONE
        or error.domain is native.TemporalCheckpointErrorDomain.NONE
    ):
        raise CalibrationV3TransactionStop("checkpoint publication active error is zero")
    return {
        "stage": error.stage.name,
        "domain": error.domain.name,
        "code": _exact_bounded_int(
            error.code,
            label="checkpoint publication error code",
            maximum=_UINT32_MAX,
        ),
    }


def _serialize_checkpoint_publication(
    publication: object,
    native: Any,
    *,
    expected_checkpoint_bytes: int,
) -> dict[str, object]:
    if type(publication) is not native.TemporalCheckpointPublication:
        raise CalibrationV3TransactionStop("checkpoint publication report type is invalid")
    if type(publication.outcome) is not native.TemporalCheckpointPublicationOutcome:
        raise CalibrationV3TransactionStop("checkpoint publication outcome is invalid")
    if type(publication.temp_cleanup) is not native.TemporalCheckpointCleanup:
        raise CalibrationV3TransactionStop("checkpoint publication cleanup is invalid")
    if (
        type(publication.cleanup_errors) is not tuple
        or len(publication.cleanup_errors) > _MAX_CLEANUP_ERRORS
    ):
        raise CalibrationV3TransactionStop(
            "checkpoint cleanup errors violate the native ABI",
            checkpoint_publication=publication,
        )
    primary = (
        None
        if publication.primary_error is None
        else _serialize_error(publication.primary_error, native)
    )
    cleanup_errors = [_serialize_error(error, native) for error in publication.cleanup_errors]
    owned_temp_pid = _exact_bounded_int(
        publication.owned_temp_pid,
        label="checkpoint publication owned_temp_pid",
        maximum=_UINT32_MAX,
    )
    owned_temp_nonce = _exact_bounded_int(
        publication.owned_temp_nonce,
        label="checkpoint publication owned_temp_nonce",
        maximum=_UINT64_MAX,
    )
    checkpoint_bytes = _exact_bounded_int(
        publication.checkpoint_bytes,
        label="checkpoint publication checkpoint_bytes",
        minimum=_CHECKPOINT_HEADER_BYTES,
        maximum=min(_UINT64_MAX, sys.maxsize),
    )
    encoded_bytes = _exact_bounded_int(
        publication.encoded_bytes,
        label="checkpoint publication encoded_bytes",
        maximum=_UINT64_MAX,
    )
    if encoded_bytes > checkpoint_bytes:
        raise CalibrationV3TransactionStop(
            "checkpoint publication encoded length exceeds checkpoint length",
            checkpoint_publication=publication,
        )
    cleanup = publication.temp_cleanup
    outcome = publication.outcome
    if cleanup is native.TemporalCheckpointCleanup.NOT_APPLICABLE:
        if (
            outcome is not native.TemporalCheckpointPublicationOutcome.NOT_PUBLISHED
            or cleanup_errors
            or owned_temp_pid != 0
            or owned_temp_nonce != 0
            or encoded_bytes != 0
        ):
            raise CalibrationV3TransactionStop(
                "checkpoint N/A cleanup report is inconsistent",
                checkpoint_publication=publication,
            )
    elif cleanup is native.TemporalCheckpointCleanup.CLEAN:
        if cleanup_errors or owned_temp_pid == 0:
            raise CalibrationV3TransactionStop(
                "clean checkpoint publication report is inconsistent",
                checkpoint_publication=publication,
            )
    elif not cleanup_errors or owned_temp_pid == 0:
        raise CalibrationV3TransactionStop(
            "orphaned checkpoint publication report is inconsistent",
            checkpoint_publication=publication,
        )
    if outcome in {
        native.TemporalCheckpointPublicationOutcome.DURABLE_PUBLISHED,
        native.TemporalCheckpointPublicationOutcome.COLLISION,
    }:
        if primary is not None or encoded_bytes != checkpoint_bytes:
            raise CalibrationV3TransactionStop(
                "successful checkpoint publication report is inconsistent",
                checkpoint_publication=publication,
            )
        if cleanup is native.TemporalCheckpointCleanup.NOT_APPLICABLE:
            raise CalibrationV3TransactionStop(
                "successful checkpoint publication lacks a temporary artifact",
                checkpoint_publication=publication,
            )
    elif outcome is native.TemporalCheckpointPublicationOutcome.NOT_PUBLISHED:
        if publication.primary_error is None:
            raise CalibrationV3TransactionStop(
                "not-published checkpoint report lacks a primary error",
                checkpoint_publication=publication,
            )
        allowed_stages = (
            {
                native.TemporalCheckpointErrorStage.PARENT_OPEN,
                native.TemporalCheckpointErrorStage.PARENT_IDENTITY,
                native.TemporalCheckpointErrorStage.FILESYSTEM_POLICY,
                native.TemporalCheckpointErrorStage.TEMP_CREATE,
                native.TemporalCheckpointErrorStage.PARENT_SYNC,
            }
            if cleanup is native.TemporalCheckpointCleanup.NOT_APPLICABLE
            else {
                native.TemporalCheckpointErrorStage.PARENT_IDENTITY,
                native.TemporalCheckpointErrorStage.ENCODE,
                native.TemporalCheckpointErrorStage.FILE_SYNC,
                native.TemporalCheckpointErrorStage.FILE_CLOSE,
                native.TemporalCheckpointErrorStage.TEMP_IDENTITY,
                native.TemporalCheckpointErrorStage.EXCLUSIVE_PUBLISH,
            }
        )
        if publication.primary_error.stage not in allowed_stages:
            raise CalibrationV3TransactionStop(
                "not-published checkpoint stage is inconsistent with cleanup",
                checkpoint_publication=publication,
            )
    elif (
        publication.primary_error is None
        or encoded_bytes != checkpoint_bytes
        or cleanup is native.TemporalCheckpointCleanup.NOT_APPLICABLE
        or publication.primary_error.stage
        not in {
            native.TemporalCheckpointErrorStage.EXCLUSIVE_PUBLISH,
            native.TemporalCheckpointErrorStage.PARENT_SYNC,
        }
    ):
        raise CalibrationV3TransactionStop(
            "unconfirmed checkpoint publication report is inconsistent",
            checkpoint_publication=publication,
        )
    if outcome is not native.TemporalCheckpointPublicationOutcome.DURABLE_PUBLISHED:
        raise CalibrationV3TransactionStop(
            "checkpoint was not durably published",
            checkpoint_publication=publication,
        )
    if checkpoint_bytes != expected_checkpoint_bytes:
        raise CalibrationV3TransactionStop(
            "checkpoint publication byte counts differ from checkpoint identity",
            checkpoint_publication=publication,
        )
    if cleanup is native.TemporalCheckpointCleanup.ORPHANED and (
        len(publication.cleanup_errors) != 1
        or publication.cleanup_errors[0].stage
        not in {
            native.TemporalCheckpointErrorStage.TEMP_IDENTITY,
            native.TemporalCheckpointErrorStage.TEMP_CLEANUP,
        }
    ):
        raise CalibrationV3TransactionStop(
            "durable checkpoint cleanup errors have an unreachable stage",
            checkpoint_publication=publication,
        )
    return {
        "outcome": outcome.name,
        "temp_cleanup": cleanup.name,
        "primary_error": primary,
        "cleanup_errors": cleanup_errors,
        "owned_temp_pid": owned_temp_pid,
        "owned_temp_nonce": owned_temp_nonce,
        "checkpoint_bytes": checkpoint_bytes,
        "encoded_bytes": encoded_bytes,
    }


def _validate_manifest_error(error: object, native: Any) -> None:
    if type(error) is not native.TemporalPublicationError:
        raise CalibrationV3TransactionStop("manifest publication error type is invalid")
    if type(error.stage) is not native.TemporalPublicationErrorStage:
        raise CalibrationV3TransactionStop("manifest publication error stage is invalid")
    if type(error.domain) is not native.TemporalPublicationErrorDomain:
        raise CalibrationV3TransactionStop("manifest publication error domain is invalid")
    if (
        error.stage is native.TemporalPublicationErrorStage.NONE
        or error.domain is native.TemporalPublicationErrorDomain.NONE
        or error.domain is native.TemporalPublicationErrorDomain.CHECKPOINT_ENCODER_RESERVED
    ):
        raise CalibrationV3TransactionStop("manifest publication active error is invalid")
    _exact_bounded_int(
        error.code,
        label="manifest publication error code",
        maximum=_UINT32_MAX,
    )


def _validate_manifest_publication(
    publication: object,
    expected_bytes: bytes,
    native: Any,
) -> None:
    if type(publication) is not native.TemporalImmutableBytesPublication:
        raise CalibrationV3TransactionStop(
            "manifest publication report type is invalid",
            manifest_publication=publication,
        )
    if type(publication.outcome) is not native.TemporalPublicationOutcome:
        raise CalibrationV3TransactionStop(
            "manifest publication outcome is invalid",
            manifest_publication=publication,
        )
    if type(publication.temp_cleanup) is not native.TemporalPublicationCleanup:
        raise CalibrationV3TransactionStop(
            "manifest publication cleanup is invalid",
            manifest_publication=publication,
        )
    if (
        type(publication.cleanup_errors) is not tuple
        or len(publication.cleanup_errors) > _MAX_CLEANUP_ERRORS
    ):
        raise CalibrationV3TransactionStop(
            "manifest cleanup errors violate the native ABI",
            manifest_publication=publication,
        )
    if publication.primary_error is not None:
        _validate_manifest_error(publication.primary_error, native)
    for error in publication.cleanup_errors:
        _validate_manifest_error(error, native)

    expected_length = len(expected_bytes)
    if not 1 <= expected_length <= _MAX_MANIFEST_BYTES:
        raise CalibrationV3TransactionStop("encoded resume manifest length is out of range")
    owned_temp_pid = _exact_bounded_int(
        publication.owned_temp_pid,
        label="manifest publication owned_temp_pid",
        maximum=_UINT32_MAX,
    )
    owned_temp_nonce = _exact_bounded_int(
        publication.owned_temp_nonce,
        label="manifest publication owned_temp_nonce",
        maximum=_UINT64_MAX,
    )
    payload_bytes = _exact_bounded_int(
        publication.payload_bytes,
        label="manifest publication payload_bytes",
        minimum=1,
        maximum=_MAX_MANIFEST_BYTES,
    )
    written_bytes = _exact_bounded_int(
        publication.written_bytes,
        label="manifest publication written_bytes",
        maximum=_MAX_MANIFEST_BYTES,
    )
    if payload_bytes != expected_length or written_bytes > payload_bytes:
        raise CalibrationV3TransactionStop(
            "manifest publication byte counts are inconsistent",
            manifest_publication=publication,
        )

    cleanup = publication.temp_cleanup
    outcome = publication.outcome
    cleanup_errors = publication.cleanup_errors
    if cleanup is native.TemporalPublicationCleanup.NOT_APPLICABLE:
        if (
            outcome is not native.TemporalPublicationOutcome.NOT_PUBLISHED
            or cleanup_errors
            or owned_temp_pid != 0
            or owned_temp_nonce != 0
            or written_bytes != 0
        ):
            raise CalibrationV3TransactionStop(
                "manifest N/A cleanup report is inconsistent",
                manifest_publication=publication,
            )
    elif cleanup is native.TemporalPublicationCleanup.CLEAN:
        if cleanup_errors or owned_temp_pid == 0:
            raise CalibrationV3TransactionStop(
                "clean manifest publication report is inconsistent",
                manifest_publication=publication,
            )
    elif not cleanup_errors or owned_temp_pid == 0:
        raise CalibrationV3TransactionStop(
            "orphaned manifest publication report is inconsistent",
            manifest_publication=publication,
        )

    if outcome in {
        native.TemporalPublicationOutcome.DURABLE_PUBLISHED,
        native.TemporalPublicationOutcome.COLLISION,
    }:
        if publication.primary_error is not None or written_bytes != payload_bytes:
            raise CalibrationV3TransactionStop(
                "successful manifest publication report is inconsistent",
                manifest_publication=publication,
            )
        if cleanup is native.TemporalPublicationCleanup.NOT_APPLICABLE:
            raise CalibrationV3TransactionStop(
                "successful manifest publication lacks a temporary artifact",
                manifest_publication=publication,
            )
    elif outcome is native.TemporalPublicationOutcome.NOT_PUBLISHED:
        if publication.primary_error is None:
            raise CalibrationV3TransactionStop(
                "not-published manifest report lacks a primary error",
                manifest_publication=publication,
            )
        allowed_stages = (
            {
                native.TemporalPublicationErrorStage.PARENT_OPEN,
                native.TemporalPublicationErrorStage.PARENT_IDENTITY,
                native.TemporalPublicationErrorStage.FILESYSTEM_POLICY,
                native.TemporalPublicationErrorStage.TEMP_CREATE,
                native.TemporalPublicationErrorStage.PARENT_SYNC,
            }
            if cleanup is native.TemporalPublicationCleanup.NOT_APPLICABLE
            else {
                native.TemporalPublicationErrorStage.PARENT_IDENTITY,
                native.TemporalPublicationErrorStage.WRITE,
                native.TemporalPublicationErrorStage.FILE_SYNC,
                native.TemporalPublicationErrorStage.FILE_CLOSE,
                native.TemporalPublicationErrorStage.TEMP_IDENTITY,
                native.TemporalPublicationErrorStage.EXCLUSIVE_PUBLISH,
            }
        )
        if publication.primary_error.stage not in allowed_stages:
            raise CalibrationV3TransactionStop(
                "not-published manifest stage is inconsistent with cleanup",
                manifest_publication=publication,
            )
    elif (
        publication.primary_error is None
        or written_bytes != payload_bytes
        or cleanup is native.TemporalPublicationCleanup.NOT_APPLICABLE
        or publication.primary_error.stage
        not in {
            native.TemporalPublicationErrorStage.EXCLUSIVE_PUBLISH,
            native.TemporalPublicationErrorStage.PARENT_SYNC,
        }
    ):
        raise CalibrationV3TransactionStop(
            "unconfirmed manifest publication report is inconsistent",
            manifest_publication=publication,
        )

    if outcome is not native.TemporalPublicationOutcome.DURABLE_PUBLISHED:
        raise CalibrationV3TransactionStop(
            "resume manifest was not durably published",
            manifest_publication=publication,
        )
    if cleanup is native.TemporalPublicationCleanup.ORPHANED and (
        len(cleanup_errors) != 1
        or cleanup_errors[0].stage
        not in {
            native.TemporalPublicationErrorStage.TEMP_IDENTITY,
            native.TemporalPublicationErrorStage.TEMP_CLEANUP,
        }
    ):
        raise CalibrationV3TransactionStop(
            "durable manifest cleanup errors have an unreachable stage",
            manifest_publication=publication,
        )


def _require_durable_directory_preparation(
    preparation: object,
    native: Any,
) -> None:
    """Accept only one exact, clean, fully durable native directory receipt."""

    if type(preparation) is not native.TemporalDurableDirectoryPreparation:
        raise CalibrationV3TransactionStop(
            "directory preparation receipt has the wrong nominal type",
            directory_preparation=preparation,
        )
    required = _exact_nonnegative_int(
        preparation.required_component_count,
        label="directory preparation required_component_count",
    )
    confirmed = _exact_nonnegative_int(
        preparation.confirmed_component_count,
        label="directory preparation confirmed_component_count",
    )
    created = _exact_nonnegative_int(
        preparation.created_component_count,
        label="directory preparation created_component_count",
    )
    if (
        type(preparation.outcome) is not native.TemporalDurabilityOutcome
        or preparation.outcome is not native.TemporalDurabilityOutcome.DURABLE
        or preparation.primary_error is not None
        or preparation.cleanup_errors != ()
        or type(preparation.durably_prepared) is not bool
        or preparation.durably_prepared is not True
        or required == 0
        or confirmed != required
        or created > required
        or preparation.failed_component_index is not None
    ):
        raise CalibrationV3TransactionStop(
            "frozen artifact directory was not durably prepared",
            directory_preparation=preparation,
        )


def _require_durable_manifest_reconciliation(
    reconciliation: object,
    expected_bytes: bytes,
    native: Any,
) -> None:
    """Accept only exact bytes durably re-established in their existing final."""

    if type(reconciliation) is not native.TemporalImmutableBytesReconciliation:
        raise CalibrationV3TransactionStop(
            "manifest reconciliation receipt has the wrong nominal type",
            manifest_reconciliation=reconciliation,
        )
    expected_length = len(expected_bytes)
    if (
        type(reconciliation.outcome) is not native.TemporalDurabilityOutcome
        or reconciliation.outcome is not native.TemporalDurabilityOutcome.DURABLE
        or reconciliation.primary_error is not None
        or reconciliation.cleanup_errors != ()
        or type(reconciliation.durably_reconciled) is not bool
        or reconciliation.durably_reconciled is not True
        or reconciliation.expected_bytes != expected_length
        or reconciliation.verified_bytes != expected_length
    ):
        raise CalibrationV3TransactionStop(
            "resume manifest durability could not be re-established",
            manifest_reconciliation=reconciliation,
        )


def _serialize_step(step: persistence.TemporalStepFrame) -> dict[str, object]:
    if type(step) is not persistence.TemporalStepFrame:
        raise CalibrationV3TransactionStop("previous state is not a TemporalStepFrame")

    def spikes(values: tuple[persistence.TemporalSpikeFrame, ...]) -> list[dict[str, int]]:
        return [{"cell_id": spike.cell_id, "tick": spike.tick} for spike in values]

    return {
        "token_id": step.token_id,
        "punished_segments": step.punished_segments,
        "event_count": step.event_count,
        "predicted": step.predicted,
        "predicted_columns": step.predicted_columns,
        "allocated_segments": step.allocated_segments,
        "inhibition_onset_ticks": list(step.inhibition_onset_ticks),
        "column_simulated_ticks": list(step.column_simulated_ticks),
        "winner_spikes": spikes(step.winner_spikes),
        "basket_spikes": spikes(step.basket_spikes),
        "proximal_spikes": spikes(step.proximal_spikes),
        "proximal_relay_spikes": spikes(step.proximal_relay_spikes),
        "winner_relay_spikes": spikes(step.winner_relay_spikes),
        "winner_context": list(step.winner_context),
        "proximal_context": list(step.proximal_context),
        "context": list(step.context),
    }


def _freeze_native_step(step: object, native: Any) -> persistence.TemporalStepFrame:
    """Copy one exact native result into the canonical persisted frame type."""

    if type(step) is not native.TemporalCircuitStep:
        raise CalibrationV3TransactionStop("native step has the wrong nominal type")

    def spikes(label: str, values: object) -> tuple[persistence.TemporalSpikeFrame, ...]:
        if type(values) is not tuple or any(
            type(value) is not native.TemporalSpike for value in values
        ):
            raise CalibrationV3TransactionStop(f"native step {label} has the wrong nominal type")
        try:
            return tuple(
                persistence.TemporalSpikeFrame(cell_id=value.cell_id, tick=value.tick)
                for value in values
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise CalibrationV3TransactionStop(
                f"native step {label} violates the persisted frame schema"
            ) from exc

    try:
        return persistence.TemporalStepFrame(
            token_id=step.token_id,
            punished_segments=step.punished_segments,
            event_count=step.event_count,
            predicted=step.predicted,
            predicted_columns=step.predicted_columns,
            allocated_segments=step.allocated_segments,
            inhibition_onset_ticks=step.inhibition_onset_ticks,
            column_simulated_ticks=step.column_simulated_ticks,
            winner_spikes=spikes("winner_spikes", step.winner_spikes),
            basket_spikes=spikes("basket_spikes", step.basket_spikes),
            proximal_spikes=spikes("proximal_spikes", step.proximal_spikes),
            proximal_relay_spikes=spikes("proximal_relay_spikes", step.proximal_relay_spikes),
            winner_relay_spikes=spikes("winner_relay_spikes", step.winner_relay_spikes),
            winner_context=step.winner_context,
            proximal_context=step.proximal_context,
            context=step.context,
        )
    except CalibrationV3TransactionStop:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise CalibrationV3TransactionStop(
            "native step violates the persisted frame schema"
        ) from exc


def _serialize_progress(progress: FrozenResumeProgress) -> dict[str, object]:
    return {
        "winner": {
            "count": progress.winner_chain.count,
            "sha256": progress.winner_chain.hexdigest,
        },
        "step": {
            "count": progress.step_chain.count,
            "sha256": progress.step_chain.hexdigest,
        },
    }


def _parse_identity(value: object, native: Any) -> object:
    payload = _exact_keys(value, frozenset(_IDENTITY_FIELDS), label="checkpoint")
    values: dict[str, object] = {}
    for field_name in _IDENTITY_FIELDS:
        field_value = payload[field_name]
        if field_name in _IDENTITY_SHA_FIELDS:
            if not _is_lower_sha256(field_value):
                raise CalibrationV3TransactionStop(
                    f"checkpoint {field_name} is not a lowercase SHA-256"
                )
        else:
            _exact_nonnegative_int(field_value, label=f"checkpoint {field_name}")
        values[field_name] = field_value
    if values["poison_code"] != 0:
        raise CalibrationV3TransactionStop("checkpoint manifest records a poisoned circuit")
    try:
        identity = native.TemporalCheckpointIdentity(**values)
    except (TypeError, ValueError) as exc:
        raise CalibrationV3TransactionStop("checkpoint identity cannot be reconstructed") from exc
    _serialize_identity(identity, native)
    return identity


def _parse_error(value: object, native: Any) -> object:
    payload = _exact_keys(value, _ERROR_FIELDS, label="checkpoint publication error")
    if type(payload["stage"]) is not str or type(payload["domain"]) is not str:
        raise CalibrationV3TransactionStop("checkpoint publication error enums must be names")
    try:
        stage = native.TemporalCheckpointErrorStage[payload["stage"]]
        domain = native.TemporalCheckpointErrorDomain[payload["domain"]]
    except (KeyError, TypeError) as exc:
        raise CalibrationV3TransactionStop("checkpoint publication error enum is unknown") from exc
    code = _exact_bounded_int(
        payload["code"],
        label="checkpoint publication error code",
        maximum=_UINT32_MAX,
    )
    return native.TemporalCheckpointPublicationError(stage=stage, domain=domain, code=code)


def _parse_checkpoint_publication(
    value: object,
    native: Any,
    *,
    expected_checkpoint_bytes: int,
) -> object:
    payload = _exact_keys(value, _PUBLICATION_FIELDS, label="checkpoint_publication")
    if type(payload["outcome"]) is not str or type(payload["temp_cleanup"]) is not str:
        raise CalibrationV3TransactionStop("checkpoint publication enums must be names")
    try:
        outcome = native.TemporalCheckpointPublicationOutcome[payload["outcome"]]
        cleanup = native.TemporalCheckpointCleanup[payload["temp_cleanup"]]
    except (KeyError, TypeError) as exc:
        raise CalibrationV3TransactionStop("checkpoint publication enum is unknown") from exc
    primary_value = payload["primary_error"]
    primary = None if primary_value is None else _parse_error(primary_value, native)
    cleanup_values = payload["cleanup_errors"]
    if type(cleanup_values) is not list:
        raise CalibrationV3TransactionStop("checkpoint cleanup_errors must be an array")
    cleanup_errors = tuple(_parse_error(error, native) for error in cleanup_values)
    publication = native.TemporalCheckpointPublication(
        outcome=outcome,
        temp_cleanup=cleanup,
        primary_error=primary,
        cleanup_errors=cleanup_errors,
        owned_temp_pid=_exact_nonnegative_int(
            payload["owned_temp_pid"], label="checkpoint publication owned_temp_pid"
        ),
        owned_temp_nonce=_exact_nonnegative_int(
            payload["owned_temp_nonce"], label="checkpoint publication owned_temp_nonce"
        ),
        checkpoint_bytes=_exact_nonnegative_int(
            payload["checkpoint_bytes"], label="checkpoint publication checkpoint_bytes"
        ),
        encoded_bytes=_exact_nonnegative_int(
            payload["encoded_bytes"], label="checkpoint publication encoded_bytes"
        ),
    )
    _serialize_checkpoint_publication(
        publication,
        native,
        expected_checkpoint_bytes=expected_checkpoint_bytes,
    )
    return publication


def _parse_spikes(value: object, *, label: str) -> tuple[persistence.TemporalSpikeFrame, ...]:
    if type(value) is not list:
        raise CalibrationV3TransactionStop(f"{label} must be an array")
    spikes: list[persistence.TemporalSpikeFrame] = []
    for raw in value:
        row = _exact_keys(raw, frozenset({"cell_id", "tick"}), label=label)
        try:
            spikes.append(
                persistence.TemporalSpikeFrame(
                    cell_id=_exact_nonnegative_int(row["cell_id"], label=f"{label} cell_id"),
                    tick=_exact_nonnegative_int(row["tick"], label=f"{label} tick"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise CalibrationV3TransactionStop(f"{label} contains an invalid spike") from exc
    return tuple(spikes)


def _exact_int_tuple(value: object, *, label: str) -> tuple[int, ...]:
    if type(value) is not list:
        raise CalibrationV3TransactionStop(f"{label} must be an array")
    return tuple(_exact_nonnegative_int(item, label=label) for item in value)


def _parse_step(value: object) -> persistence.TemporalStepFrame:
    payload = _exact_keys(value, _STEP_FIELDS, label="previous_state")
    if type(payload["predicted"]) is not bool:
        raise CalibrationV3TransactionStop("previous_state predicted must be an exact bool")
    kwargs: dict[str, object] = {
        "token_id": _exact_nonnegative_int(payload["token_id"], label="previous token_id"),
        "punished_segments": _exact_nonnegative_int(
            payload["punished_segments"], label="previous punished_segments"
        ),
        "event_count": _exact_nonnegative_int(payload["event_count"], label="previous event_count"),
        "predicted": payload["predicted"],
        "predicted_columns": _exact_nonnegative_int(
            payload["predicted_columns"], label="previous predicted_columns"
        ),
        "allocated_segments": _exact_nonnegative_int(
            payload["allocated_segments"], label="previous allocated_segments"
        ),
        "inhibition_onset_ticks": _exact_int_tuple(
            payload["inhibition_onset_ticks"], label="previous inhibition_onset_ticks"
        ),
        "column_simulated_ticks": _exact_int_tuple(
            payload["column_simulated_ticks"], label="previous column_simulated_ticks"
        ),
        "winner_context": _exact_int_tuple(
            payload["winner_context"], label="previous winner_context"
        ),
        "proximal_context": _exact_int_tuple(
            payload["proximal_context"], label="previous proximal_context"
        ),
        "context": _exact_int_tuple(payload["context"], label="previous context"),
    }
    for field_name in _SPIKE_FIELDS:
        kwargs[field_name] = _parse_spikes(payload[field_name], label=f"previous {field_name}")
    try:
        return persistence.TemporalStepFrame(**kwargs)
    except (TypeError, ValueError) as exc:
        raise CalibrationV3TransactionStop("previous_state violates the step schema") from exc


def _parse_progress(value: object, previous_state: object) -> FrozenResumeProgress:
    streams = _exact_keys(value, _STREAM_FIELDS, label="stream_prefixes")

    def chain(name: str, kind: persistence.HashChainKind) -> persistence.ResumableHashChain:
        row = _exact_keys(streams[name], _CHAIN_FIELDS, label=f"stream_prefixes {name}")
        count = _exact_nonnegative_int(row["count"], label=f"stream_prefixes {name} count")
        digest = row["sha256"]
        if not _is_lower_sha256(digest):
            raise CalibrationV3TransactionStop(f"stream_prefixes {name} sha256 is not lowercase")
        try:
            return persistence.ResumableHashChain.restore(
                kind=kind,
                count=count,
                digest_hex=digest,
            )
        except (TypeError, ValueError) as exc:
            raise CalibrationV3TransactionStop(
                f"stream_prefixes {name} cannot be restored"
            ) from exc

    return FrozenResumeProgress(
        last_step=_parse_step(previous_state),
        winner_chain=chain("winner", persistence.HashChainKind.WINNER),
        step_chain=chain("step", persistence.HashChainKind.STEP),
    )


class CalibrationV3Transaction:
    """Trusted executor-owned coordinator; never pass it to scientific callbacks."""

    __slots__ = (
        "_active_token",
        "_circuit",
        "_circuit_contract",
        "_claim_boundary",
        "_clock",
        "_definition",
        "_definition_scalars",
        "_expected_circuit_identities",
        "_expected_circuit_identity",
        "_expected_threshold_count",
        "_expected_token_count",
        "_last_clock_ns",
        "_manifest_reconciliation",
        "_native",
        "_next_resume_index",
        "_operation_lock",
        "_prior_cumulative_ns",
        "_progress",
        "_protocol_sha256",
        "_resume_events",
        "_run_coordinates",
        "_sealed_checkpoint_publication",
        "_sealed_commit",
        "_sealed_manifest_publication",
        "_sealed_timing",
        "_sealed_token",
        "_source_binding",
        "_source_bundle_sha256",
        "_start_ns",
        "_state",
        "_store",
        "_terminal_receipt",
        "_train_input",
    )

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("use CalibrationV3Transaction.from_authority()")

    @classmethod
    def from_authority(
        cls,
        authority: FrozenPlanAuthority,
    ) -> CalibrationV3Transaction:
        """Consume one authority with the non-injectable production clock."""

        return cls._from_authority(authority, clock=time.perf_counter_ns)

    @classmethod
    def _from_authority_for_test(
        cls,
        authority: FrozenPlanAuthority,
        *,
        clock: Callable[[], int],
    ) -> CalibrationV3Transaction:
        """Trusted test-only constructor; never part of the official executor seam."""

        return cls._from_authority(authority, clock=clock)

    @classmethod
    def _from_authority(
        cls,
        authority: FrozenPlanAuthority,
        *,
        clock: Callable[[], int],
    ) -> CalibrationV3Transaction:
        """Validate and snapshot one nominal authority without touching native code."""

        if type(authority) is not FrozenPlanAuthority:
            raise CalibrationV3TransactionStop("transaction requires exact frozen-plan authority")
        if not isinstance(authority.repository_root, Path):
            raise CalibrationV3TransactionStop("authority repository root must be a pathlib.Path")
        authority_protocol = authority.protocol
        if type(authority_protocol) is not dict:
            raise CalibrationV3TransactionStop("authority protocol must be an exact object")
        protocol_bytes = _canonical_json_bytes(authority_protocol)
        observed_protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
        if (
            not _is_lower_sha256(authority.protocol_sha256)
            or authority.protocol_sha256 != observed_protocol_sha256
        ):
            raise CalibrationV3TransactionStop("authority protocol digest is incoherent")
        if not _is_lower_sha256(authority.source_bundle_identity_sha256):
            raise CalibrationV3TransactionStop("authority source-bundle digest is invalid")
        try:
            protocol = json.loads(protocol_bytes)
        except (UnicodeDecodeError, ValueError) as exc:  # pragma: no cover - encoded above.
            raise CalibrationV3TransactionStop(
                "authority protocol snapshot cannot be decoded"
            ) from exc
        binding = _exact_object(protocol.get("source_binding"), label="source_binding")
        blockers = binding.get("blockers")
        if binding.get("runnable") is not True:
            raise CalibrationV3TransactionStop("transaction authority is not runnable")
        if type(blockers) is not list or blockers:
            raise CalibrationV3TransactionStop("runnable transaction authority has blockers")
        if not callable(clock):
            raise CalibrationV3TransactionStop("transaction clock must be callable")

        identities = cls._validate_circuit_identity_authority(protocol)
        resume_events = cls._validate_resume_event_authority(protocol)
        (
            circuit_contract,
            train_input,
            definition_scalars,
            expected_token_count,
            expected_threshold_count,
        ) = cls._validate_execution_authority(protocol)
        try:
            store = persistence.CalibrationV3ArtifactStore.open(
                repository_root=authority.repository_root,
                validated_protocol=protocol,
                expected_source_bundle_identity_sha256=(authority.source_bundle_identity_sha256),
            )
        except persistence.PersistenceStop as exc:
            raise CalibrationV3TransactionStop(
                f"transaction artifact store authority failed: {exc}"
            ) from exc

        self = object.__new__(cls)
        self._protocol_sha256 = observed_protocol_sha256
        self._source_bundle_sha256 = authority.source_bundle_identity_sha256
        self._source_binding = _json_snapshot(binding)
        self._claim_boundary = _json_snapshot(protocol.get("claim_boundary"))
        self._circuit_contract = circuit_contract
        self._train_input = train_input
        self._definition_scalars = definition_scalars
        self._expected_token_count = expected_token_count
        self._expected_threshold_count = expected_threshold_count
        self._expected_circuit_identities = identities
        self._resume_events = resume_events
        self._clock = clock
        self._store = store
        self._state = _CoordinatorState.READY
        self._operation_lock = threading.RLock()
        self._native = None
        self._circuit = None
        self._definition = None
        self._expected_circuit_identity = None
        self._run_coordinates = None
        self._next_resume_index = None
        self._progress = None
        self._start_ns = None
        self._last_clock_ns = None
        self._manifest_reconciliation = None
        self._prior_cumulative_ns = None
        self._active_token = None
        self._sealed_checkpoint_publication = None
        self._sealed_commit = None
        self._sealed_manifest_publication = None
        self._sealed_timing = None
        self._sealed_token = None
        self._terminal_receipt = None
        return self

    @staticmethod
    def _validate_execution_authority(
        protocol: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], tuple[tuple[str, int], ...], int, int]:
        circuit = _exact_keys(
            protocol.get("circuit_contract"),
            frozenset(
                {
                    "active_columns",
                    "base_cells_per_column",
                    "cohort_cells",
                    "column_count",
                    "contacts_per_segment",
                    "context_physical_width",
                    "max_pyramidal_cells",
                    "novel_race_cap",
                    "projection_seed",
                    "source_relay_pairs_per_segment",
                }
            ),
            label="circuit_contract",
        )
        for name, value in circuit.items():
            _exact_nonnegative_int(value, label=f"circuit_contract.{name}")
        capacity = _exact_object(protocol.get("capacity"), label="capacity")
        structure = _exact_object(capacity.get("structure"), label="capacity.structure")
        segment_count = _exact_nonnegative_int(
            structure.get("segment_count"),
            label="capacity.structure.segment_count",
        )
        max_segments = _exact_nonnegative_int(
            structure.get("max_segments_per_cell"),
            label="capacity.structure.max_segments_per_cell",
        )
        configured_cells = _exact_nonnegative_int(
            structure.get("configured_physical_cells"),
            label="capacity.structure.configured_physical_cells",
        )
        screens = _exact_object(protocol.get("screens"), label="screens")
        r3_screen = _exact_object(screens.get("r3_f0_train"), label="screens.r3_f0_train")
        train_input = _exact_object(r3_screen.get("input"), label="screens.r3_f0_train.input")
        full_screen = _exact_object(
            screens.get("full_cohort_q255"),
            label="screens.full_cohort_q255",
        )
        codebook = _exact_object(
            full_screen.get("production_codebook"),
            label="screens.full_cohort_q255.production_codebook",
        )
        token_count = _exact_nonnegative_int(
            codebook.get("token_count"),
            label="production_codebook.token_count",
        )
        definition_scalars = tuple(
            sorted(
                {
                    "base_cells_per_column": circuit["base_cells_per_column"],
                    "column_count": circuit["column_count"],
                    "max_pyramidal_cells": circuit["max_pyramidal_cells"],
                    "max_segments_per_cell": max_segments,
                    "novel_race_cap": circuit["novel_race_cap"],
                    "segment_capacity": segment_count,
                }.items()
            )
        )
        return (
            _json_snapshot(circuit),
            _json_snapshot(train_input),
            definition_scalars,
            token_count,
            configured_cells,
        )

    @staticmethod
    def _validate_circuit_identity_authority(
        protocol: dict[str, Any],
    ) -> dict[tuple[str, int], str]:
        selection = _exact_object(protocol.get("threshold_selection"), label="threshold_selection")
        candidates = selection.get("candidate_arms")
        if type(candidates) is not list or not candidates:
            raise CalibrationV3TransactionStop("circuit identity authority has no arms")
        arms: set[str] = set()
        for candidate in candidates:
            if (
                type(candidate) is not list
                or len(candidate) != 2
                or any(type(value) is not int or value < 0 for value in candidate)
            ):
                raise CalibrationV3TransactionStop("circuit identity arm is invalid")
            arms.add(f"theta-m-{candidate[0]}_theta-a-{candidate[1]}")
        if len(arms) != len(candidates):
            raise CalibrationV3TransactionStop("circuit identity arms contain duplicates")

        screens = _exact_object(protocol.get("screens"), label="screens")
        screen = _exact_object(screens.get("r3_f0_train"), label="screens.r3_f0_train")
        scope = _exact_object(
            screen.get("construction_seed_scope"),
            label="r3_f0 construction_seed_scope",
        )
        development = scope.get("development")
        confirmation = scope.get("deterministic_confirmation")
        if development != [17] or confirmation != [17]:
            raise CalibrationV3TransactionStop(
                "circuit identity authority requires exact construction seed 17"
            )
        rows = _exact_object(
            screen.get("native_circuit_identity_sha256_by_arm_and_seed"),
            label="native circuit identity authority",
        )
        if set(rows) != arms:
            raise CalibrationV3TransactionStop(
                "native circuit identity authority arm set is incomplete"
            )
        seed_key = "construction-seed-0000000017"
        identities: dict[tuple[str, int], str] = {}
        for arm in sorted(arms):
            seed_rows = _exact_object(rows[arm], label=f"native circuit identity {arm}")
            if set(seed_rows) != {seed_key}:
                raise CalibrationV3TransactionStop(
                    "native circuit identity authority seed set is incomplete"
                )
            digest = seed_rows[seed_key]
            if not _is_lower_sha256(digest):
                raise CalibrationV3TransactionStop(
                    "native circuit identity digest is not lowercase SHA-256"
                )
            identities[(arm, 17)] = digest
        return identities

    @staticmethod
    def _validate_resume_event_authority(protocol: dict[str, Any]) -> tuple[int, ...]:
        screens = _exact_object(protocol.get("screens"), label="screens")
        screen = _exact_object(screens.get("r3_f0_train"), label="screens.r3_f0_train")
        determinism = _exact_object(
            screen.get("determinism"),
            label="screens.r3_f0_train.determinism",
        )
        rows = determinism.get("resume_milestones")
        if type(rows) is not list or not rows:
            raise CalibrationV3TransactionStop("resume milestone authority is empty")
        events: list[int] = []
        for raw_row in rows:
            row = _exact_object(raw_row, label="resume milestone")
            event = _exact_nonnegative_int(
                row.get("after_transition_event"),
                label="resume milestone after_transition_event",
            )
            if events and event <= events[-1]:
                raise CalibrationV3TransactionStop(
                    "resume milestone authority is not strictly ordered"
                )
            events.append(event)
        return tuple(events)

    def _clock_value(self) -> int:
        try:
            value = self._clock()
        except Exception as exc:
            raise CalibrationV3TransactionStop("active-wall clock failed") from exc
        if type(value) is not int or value < 0:
            raise CalibrationV3TransactionStop(
                "active-wall clock must return a non-negative exact int"
            )
        return value

    def _definition_authority(
        self,
        definition: object,
        destination: persistence.DestinationSpec,
        native: Any,
    ) -> str:
        if type(definition) is not native.NativeTemporalCircuitDefinition:
            raise CalibrationV3TransactionStop("native circuit definition type is invalid")
        if destination.arm_slug is None or destination.thresholds is None:
            raise CalibrationV3TransactionStop("destination has no circuit arm authority")
        seed = destination.construction_seed
        if seed is None:
            seeds = destination.construction_seeds
            if seeds is None or len(seeds) != 1:
                raise CalibrationV3TransactionStop("destination has no exact construction seed")
            seed = seeds[0]
        expected_identity = self._expected_circuit_identities.get((destination.arm_slug, seed))
        if expected_identity is None:
            raise CalibrationV3TransactionStop("destination has no native circuit identity")
        config = getattr(definition, "config", None)
        theta_match, theta_active = destination.thresholds
        expected_scalars = dict(self._definition_scalars)
        expected_scalars.update(
            {
                "construction_seed": seed,
                "theta_match": theta_match,
                "theta_active": theta_active,
            }
        )
        for field_name, expected in expected_scalars.items():
            if getattr(config, field_name, None) != expected:
                raise CalibrationV3TransactionStop(
                    f"native circuit definition {field_name} lacks frozen authority"
                )
        try:
            token_count = len(definition.token_codes)
            threshold_count = len(definition.threshold_offsets_mv)
        except (AttributeError, TypeError) as exc:
            raise CalibrationV3TransactionStop(
                "native circuit definition tables are unavailable"
            ) from exc
        if token_count != self._expected_token_count:
            raise CalibrationV3TransactionStop("native circuit token codebook count drifted")
        if threshold_count != self._expected_threshold_count:
            raise CalibrationV3TransactionStop("native circuit threshold table count drifted")
        return expected_identity

    def _validate_identity_authority(
        self,
        identity: object,
        expected_circuit_identity: str,
        native: Any,
    ) -> dict[str, object]:
        payload = _serialize_identity(identity, native)
        if identity.protocol_identity_sha256 != self._protocol_sha256:
            raise CalibrationV3TransactionStop("checkpoint protocol identity drifted")
        if identity.source_bundle_identity_sha256 != self._source_bundle_sha256:
            raise CalibrationV3TransactionStop("checkpoint source-bundle identity drifted")
        if identity.circuit_identity_sha256 != expected_circuit_identity:
            raise CalibrationV3TransactionStop("checkpoint circuit identity drifted")
        return payload

    def _prepare_durable_namespace(self, native: Any) -> None:
        """Durably prepare the complete frozen output namespace before timing."""

        try:
            for repository_root, relative_directory in self._store.durable_directory_requests():
                preparation = native.prepare_durable_directory(
                    repository_root,
                    Path(relative_directory),
                )
                _require_durable_directory_preparation(preparation, native)
        except CalibrationV3TransactionStop:
            raise
        except Exception as exc:
            raise CalibrationV3TransactionStop(_safe_exception_message(exc)) from exc

    @staticmethod
    def _close_handle_once(
        handle: object | None,
        *,
        stage: str,
    ) -> tuple[CalibrationV3CleanupFailure, ...]:
        if handle is None:
            return ()
        try:
            handle.close()
        except BaseException as exc:
            return (
                CalibrationV3CleanupFailure(
                    stage=stage,
                    error_type=type(exc).__name__,
                ),
            )
        return ()

    def _unavailable_stop(self, message: str) -> CalibrationV3TransactionStop:
        receipt = self._terminal_receipt
        if receipt is None:
            return CalibrationV3TransactionStop(message)
        return CalibrationV3TransactionStop(
            message,
            timing_state=receipt.timing_state,
            timing=receipt.timing,
            checkpoint_publication=receipt.checkpoint_publication,
            manifest_publication=receipt.manifest_publication,
            directory_preparation=receipt.directory_preparation,
            manifest_reconciliation=receipt.manifest_reconciliation,
            cleanup_failures=receipt.cleanup_failures,
        )

    def _require_ready(self) -> None:
        if self._state is not _CoordinatorState.READY:
            raise self._unavailable_stop("transaction coordinator is no longer ready")

    def _require_active_session(self, session: object) -> CalibrationV3Session:
        if (
            type(session) is not CalibrationV3Session
            or getattr(session, "_owner", None) is not self
        ):
            raise CalibrationV3TransactionStop("session is bound to another transaction")
        if (
            self._state is not _CoordinatorState.ACTIVE
            or self._active_token is None
            or getattr(session, "_token", None) is not self._active_token
            or self._circuit is None
        ):
            raise self._unavailable_stop("session was consumed or is not active")
        return session

    def _require_sealed_commit(
        self,
        commit: object,
    ) -> CommittedResumeCheckpoint:
        if (
            type(commit) is not CommittedResumeCheckpoint
            or getattr(commit, "_owner", None) is not self
        ):
            raise CalibrationV3TransactionStop("commit is bound to another transaction")
        if (
            self._state is not _CoordinatorState.SEALED
            or commit is not self._sealed_commit
            or self._sealed_token is None
            or getattr(commit, "_token", None) is not self._sealed_token
            or type(self._sealed_timing) is not persistence.CumulativeActiveWallTiming
            or self._sealed_checkpoint_publication is None
            or self._sealed_manifest_publication is None
            or self._circuit is None
        ):
            raise self._unavailable_stop("commit was consumed or is unavailable")
        return commit

    def _current_timing(self) -> persistence.CumulativeActiveWallTiming:
        if (
            self._start_ns is None
            or self._last_clock_ns is None
            or self._prior_cumulative_ns is None
        ):
            raise CalibrationV3TransactionStop("active timing state is incomplete")
        now = self._clock_value()
        if now < self._last_clock_ns:
            raise CalibrationV3TransactionStop("active-wall clock moved backward")
        self._last_clock_ns = now
        return persistence.CumulativeActiveWallTiming.complete_resumed_session(
            self._prior_cumulative_ns,
            now - self._start_ns,
        )

    def _terminalize(
        self,
        primary: BaseException,
        *,
        timing_state: TransactionTimingState,
        timing: persistence.CumulativeActiveWallTiming | None = None,
        checkpoint_publication: object | None = None,
        manifest_publication: object | None = None,
        directory_preparation: object | None = None,
        manifest_reconciliation: object | None = None,
        cleanup_failures: tuple[CalibrationV3CleanupFailure, ...] = (),
    ) -> CalibrationV3TransactionStop:
        circuit = self._circuit
        self._circuit = None
        self._active_token = None
        self._sealed_checkpoint_publication = None
        self._sealed_commit = None
        self._sealed_manifest_publication = None
        self._sealed_timing = None
        self._sealed_token = None
        cleanup = cleanup_failures + self._close_handle_once(
            circuit,
            stage="original_circuit_close",
        )
        message = _safe_exception_message(primary)
        if manifest_reconciliation is None:
            manifest_reconciliation = self._manifest_reconciliation
        if timing is not None and timing.state is persistence.TimingState.OVER_CAP:
            timing_state = TransactionTimingState.OVER_CAP
        self._state = (
            _CoordinatorState.TERMINAL_OVER_CAP
            if timing_state is TransactionTimingState.OVER_CAP
            else _CoordinatorState.TERMINAL_STOP
        )
        receipt = _TransactionStopReceipt(
            timing_state=timing_state,
            timing=timing,
            checkpoint_publication=checkpoint_publication,
            manifest_publication=manifest_publication,
            directory_preparation=directory_preparation,
            manifest_reconciliation=manifest_reconciliation,
            cleanup_failures=cleanup,
        )
        self._terminal_receipt = receipt
        stopped = CalibrationV3TransactionStop(
            message,
            timing_state=timing_state,
            timing=timing,
            checkpoint_publication=checkpoint_publication,
            manifest_publication=manifest_publication,
            directory_preparation=directory_preparation,
            manifest_reconciliation=manifest_reconciliation,
            cleanup_failures=cleanup,
        )
        return stopped

    def _active_progress(self, session: CalibrationV3Session) -> FrozenResumeProgress | None:
        with self._operation_lock:
            self._require_active_session(session)
            return self._progress

    def _active_manifest_reconciliation(
        self,
        session: CalibrationV3Session,
    ) -> object | None:
        with self._operation_lock:
            self._require_active_session(session)
            return self._manifest_reconciliation

    def _advance_active(
        self,
        session: CalibrationV3Session,
        token_ids: object,
        *,
        mode: object,
        reset_before: bool,
    ) -> tuple[persistence.TemporalStepFrame, ...]:
        with self._operation_lock:
            self._require_active_session(session)
            try:
                native_steps = self._circuit.advance_block(
                    token_ids,
                    mode=mode,
                    reset_before=reset_before,
                )
                if type(native_steps) is not tuple:
                    raise CalibrationV3TransactionStop(
                        "native advance result has the wrong nominal type"
                    )
                native = self._native
                if native is None:
                    raise CalibrationV3TransactionStop(
                        "active branch has no cached native authority"
                    )
                frozen_steps = tuple(_freeze_native_step(step, native) for step in native_steps)
                if not frozen_steps:
                    return ()

                progress = self._progress
                if progress is None:
                    winner_chain = persistence.ResumableHashChain.start(
                        persistence.HashChainKind.WINNER
                    )
                    step_chain = persistence.ResumableHashChain.start(
                        persistence.HashChainKind.STEP
                    )
                elif type(progress) is FrozenResumeProgress:
                    winner_chain = progress.winner_chain
                    step_chain = progress.step_chain
                else:  # pragma: no cover - coordinator assigns exact records only.
                    raise CalibrationV3TransactionStop("active progress has the wrong nominal type")
                for index, frame in enumerate(frozen_steps):
                    if frame.event_count != step_chain.count + 1:
                        raise CalibrationV3TransactionStop(
                            "native step event count is not the next stream position"
                        )
                    step_chain = step_chain.append_step(frame)
                    if not (reset_before and index == 0):
                        winner_chain = winner_chain.append_winner(
                            persistence.WinnerChainFrame(
                                token_id=frame.token_id,
                                event_count=frame.event_count,
                                winner_context=frame.winner_context[:11],
                            )
                        )
                self._progress = FrozenResumeProgress(
                    last_step=frozen_steps[-1],
                    winner_chain=winner_chain,
                    step_chain=step_chain,
                )
                return frozen_steps
            except BaseException as exc:
                if (
                    self._state
                    in {
                        _CoordinatorState.TERMINAL_STOP,
                        _CoordinatorState.TERMINAL_OVER_CAP,
                    }
                    and self._terminal_receipt is not None
                ):
                    if isinstance(exc, Exception):
                        raise self._unavailable_stop(
                            "transaction already reached a terminal state"
                        ) from exc
                    raise
                stopped = self._terminalize(
                    exc,
                    timing_state=TransactionTimingState.TIMING_INDETERMINATE,
                )
                if isinstance(exc, Exception):
                    raise stopped from exc
                raise

    def _check_active_session_cap(
        self,
        session: CalibrationV3Session,
    ) -> persistence.TimingState:
        with self._operation_lock:
            self._require_active_session(session)
            try:
                timing = self._current_timing()
            except BaseException as exc:
                stopped = self._terminalize(
                    exc,
                    timing_state=TransactionTimingState.TIMING_INDETERMINATE,
                )
                if isinstance(exc, Exception):
                    raise stopped from exc
                raise
            if timing.state is persistence.TimingState.OVER_CAP:
                stopped = CalibrationV3TransactionStop("cumulative active-wall cap was crossed")
                raise self._terminalize(
                    stopped,
                    timing_state=TransactionTimingState.OVER_CAP,
                    timing=timing,
                ) from stopped
            return timing.state

    def _close_active_session(self, session: CalibrationV3Session) -> None:
        with self._operation_lock:
            self._require_active_session(session)
            stopped = CalibrationV3TransactionStop(
                "active session closed before a committed boundary"
            )
            raise self._terminalize(
                stopped,
                timing_state=TransactionTimingState.TIMING_INDETERMINATE,
            ) from stopped

    def start_fresh(
        self,
        result_path: str,
        definition: object,
    ) -> CalibrationV3Session:
        """Start the coordinator's sole fresh branch interval."""

        with self._operation_lock:
            self._require_ready()
            try:
                destination = self._store.require_r3_result_destination(result_path)
            except persistence.PersistenceStop as exc:
                raise CalibrationV3TransactionStop(
                    f"fresh result destination is invalid: {exc}"
                ) from exc
            native = _native_runtime()
            expected_identity = self._definition_authority(definition, destination, native)
            run = _RunCoordinates.from_destination(destination)
            self._prepare_durable_namespace(native)
            start_ns = self._clock_value()
            self._native = native
            self._definition = definition
            self._expected_circuit_identity = expected_identity
            self._run_coordinates = run
            self._next_resume_index = 0
            self._progress = None
            self._manifest_reconciliation = None
            self._start_ns = start_ns
            self._last_clock_ns = start_ns
            self._prior_cumulative_ns = 0
            self._state = _CoordinatorState.STARTING
            try:
                self._circuit = native.NativeTemporalCircuit.fresh(definition)
                identity = self._circuit.checkpoint_identity()
                self._validate_identity_authority(identity, expected_identity, native)
                self._active_token = object()
                self._state = _CoordinatorState.ACTIVE
                session = CalibrationV3Session(owner=self, token=self._active_token)
                timing = self._current_timing()
            except BaseException as exc:
                stopped = self._terminalize(
                    exc,
                    timing_state=TransactionTimingState.TIMING_INDETERMINATE,
                )
                if isinstance(exc, Exception):
                    raise stopped from exc
                raise
            if timing.state is persistence.TimingState.OVER_CAP:
                stopped = CalibrationV3TransactionStop(
                    "cumulative active-wall cap was crossed during fresh construction"
                )
                raise self._terminalize(
                    stopped,
                    timing_state=TransactionTimingState.OVER_CAP,
                    timing=timing,
                ) from stopped
            return session

    @staticmethod
    def _validate_progress(
        progress: FrozenResumeProgress,
        destination: persistence.DestinationSpec,
        identity: object | None = None,
    ) -> dict[str, object]:
        if type(progress) is not FrozenResumeProgress:
            raise CalibrationV3TransactionStop("resume progress has the wrong nominal type")
        if destination.resume_cursor is None:
            raise CalibrationV3TransactionStop("resume destination has no frozen cursor")
        cursor = dict(destination.resume_cursor)
        if progress.winner_chain.count != cursor["after_transition_event"]:
            raise CalibrationV3TransactionStop("winner prefix count differs from the cursor")
        if progress.step_chain.count != cursor["event_count"]:
            raise CalibrationV3TransactionStop("step prefix count differs from the cursor")
        if progress.last_step.event_count != cursor["event_count"]:
            raise CalibrationV3TransactionStop("previous step differs from the cursor")
        if identity is not None and (
            identity.event_count != cursor["event_count"]
            or identity.learning_step != cursor["learning_step"]
            or identity.event_count != progress.step_chain.count
            or identity.prior_context_length != len(progress.last_step.context)
            or identity.winner_context_length != len(progress.last_step.winner_context)
        ):
            raise CalibrationV3TransactionStop(
                "checkpoint identity and resumable progress lack parity"
            )
        return cursor

    def _next_resume_destination(self) -> persistence.DestinationSpec:
        run = self._run_coordinates
        next_index = self._next_resume_index
        if type(run) is not _RunCoordinates or type(next_index) is not int:
            raise CalibrationV3TransactionStop("active branch coordinates are incomplete")
        if run.screen_id != "r3_f0_train" or run.phase != "deterministic_confirmation":
            raise CalibrationV3TransactionStop(
                "active branch has no deterministic-confirmation resume schedule"
            )
        if not 0 <= next_index < len(self._resume_events):
            raise CalibrationV3TransactionStop("active branch has no next frozen milestone")
        destination = self._store.require_resume_manifest_for(
            arm_slug=run.arm_slug,
            after_transition_event=self._resume_events[next_index],
        )
        if _RunCoordinates.from_destination(destination) != run:
            raise CalibrationV3TransactionStop(
                "next resume destination differs from active run coordinates"
            )
        return destination

    def _manifest_payload(
        self,
        *,
        destination: persistence.DestinationSpec,
        identity_payload: dict[str, object],
        publication_payload: dict[str, object],
        progress: FrozenResumeProgress,
        cursor: dict[str, object],
        timing: persistence.CumulativeActiveWallTiming,
    ) -> dict[str, object]:
        if (
            destination.arm_slug is None
            or destination.phase is None
            or destination.screen_id is None
            or destination.thresholds is None
            or destination.construction_seed is None
            or timing.session_active_ns is None
            or timing.cumulative_active_ns is None
        ):
            raise CalibrationV3TransactionStop("resume manifest inputs are incomplete")
        theta_match, theta_active = destination.thresholds
        reference = (
            f"r3-f0/deterministic-confirmation/{destination.arm_slug}/"
            f"construction-seed-{destination.construction_seed:010d}/fresh"
        )
        return {
            "arm_slug": destination.arm_slug,
            "artifact_path": destination.relative_path,
            "checkpoint": identity_payload,
            "checkpoint_publication": publication_payload,
            "circuit_contract": _json_snapshot(self._circuit_contract),
            "claim_boundary": _json_snapshot(self._claim_boundary),
            "construction_seed": destination.construction_seed,
            "cursor": cursor,
            "phase": destination.phase,
            "previous_state": _serialize_step(progress.last_step),
            "protocol_sha256": self._protocol_sha256,
            "reference_trajectory_id": reference,
            "screen_id": destination.screen_id,
            "source_binding": _json_snapshot(self._source_binding),
            "source_bundle_identity_sha256": self._source_bundle_sha256,
            "stream_prefixes": _serialize_progress(progress),
            "thresholds": {
                "theta_active": theta_active,
                "theta_match": theta_match,
            },
            "timing": {
                "prior_cumulative_ns": timing.prior_cumulative_ns,
                "session_active_ns": timing.session_active_ns,
                "cumulative_active_ns": timing.cumulative_active_ns,
            },
            "train_input": _json_snapshot(self._train_input),
        }

    def _require_durable_manifest_publication(
        self,
        publication: object,
        expected_bytes: bytes,
        native: Any,
    ) -> None:
        _validate_manifest_publication(publication, expected_bytes, native)

    def commit_resume_checkpoint(
        self,
        session: CalibrationV3Session,
    ) -> CommittedResumeCheckpoint:
        """Commit the coordinator-owned branch at its exact next milestone."""

        with self._operation_lock:
            self._require_active_session(session)
            publication: object | None = None
            manifest_publication: object | None = None
            verifier: object | None = None
            timing: persistence.CumulativeActiveWallTiming | None = None
            commit: CommittedResumeCheckpoint | None = None
            boundary_confirmed = False
            cleanup_failures: tuple[CalibrationV3CleanupFailure, ...] = ()
            try:
                progress = self._progress
                if type(progress) is not FrozenResumeProgress:
                    raise CalibrationV3TransactionStop(
                        "active branch has no transaction-owned resumable progress"
                    )
                destination = self._next_resume_destination()
                cursor = self._validate_progress(progress, destination)
                timing = self._current_timing()
                if timing.state is persistence.TimingState.OVER_CAP:
                    stopped = CalibrationV3TransactionStop(
                        "cumulative active-wall cap was crossed before checkpoint publication"
                    )
                    raise self._terminalize(
                        stopped,
                        timing_state=TransactionTimingState.OVER_CAP,
                        timing=timing,
                    ) from stopped

                self._state = _CoordinatorState.COMMITTING
                self._active_token = None
                native = self._native
                if native is None:
                    raise CalibrationV3TransactionStop(
                        "active branch has no cached native authority"
                    )
                prepared = self._store.prepare_absent_resume_pair(destination.relative_path)
                timing = self._current_timing()
                if timing.state is persistence.TimingState.OVER_CAP:
                    stopped = CalibrationV3TransactionStop(
                        "cumulative active-wall cap was crossed before checkpoint publication"
                    )
                    raise self._terminalize(
                        stopped,
                        timing_state=TransactionTimingState.OVER_CAP,
                        timing=timing,
                    ) from stopped

                pre_identity = self._circuit.checkpoint_identity()
                identity_payload = self._validate_identity_authority(
                    pre_identity,
                    self._expected_circuit_identity,
                    native,
                )
                self._validate_progress(progress, destination, pre_identity)
                timing = self._current_timing()
                if timing.state is persistence.TimingState.OVER_CAP:
                    stopped = CalibrationV3TransactionStop(
                        "cumulative active-wall cap was crossed before checkpoint publication"
                    )
                    raise self._terminalize(
                        stopped,
                        timing_state=TransactionTimingState.OVER_CAP,
                        timing=timing,
                    ) from stopped

                publication = self._circuit.publish_checkpoint(prepared.checkpoint_path)
                # Publication is non-cancellable: finish verification and record any overshoot.
                timing = self._current_timing()
                publication_payload = _serialize_checkpoint_publication(
                    publication,
                    native,
                    expected_checkpoint_bytes=pre_identity.checkpoint_bytes,
                )

                post_identity = self._circuit.checkpoint_identity()
                self._validate_identity_authority(
                    post_identity,
                    self._expected_circuit_identity,
                    native,
                )
                if post_identity != pre_identity:
                    raise CalibrationV3TransactionStop("post-publication identity lacks parity")

                checkpoint_pin = self._store.pin_checkpoint_final(destination.relative_path)
                verifier = native.NativeTemporalCircuit.from_checkpoint_path(
                    self._definition,
                    prepared.checkpoint_path,
                )
                imported_identity = verifier.checkpoint_identity()
                self._validate_identity_authority(
                    imported_identity,
                    self._expected_circuit_identity,
                    native,
                )
                if imported_identity != pre_identity:
                    raise CalibrationV3TransactionStop(
                        "fresh-import checkpoint identity lacks parity"
                    )
                self._validate_progress(progress, destination, imported_identity)
                verifier_cleanup = self._close_handle_once(
                    verifier,
                    stage="verifier_close",
                )
                verifier = None
                cleanup_failures += verifier_cleanup
                if verifier_cleanup:
                    raise CalibrationV3TransactionStop(
                        "checkpoint verifier close failed",
                    )
                self._store.confirm_checkpoint_final(checkpoint_pin)

                timing = self._current_timing()
                self._state = _CoordinatorState.RECORDING
                payload = self._manifest_payload(
                    destination=destination,
                    identity_payload=identity_payload,
                    publication_payload=publication_payload,
                    progress=progress,
                    cursor=cursor,
                    timing=timing,
                )
                manifest_bytes = self._store.encode_resume(payload)
                manifest_publication = native.publish_immutable_bytes(
                    prepared.manifest_path,
                    manifest_bytes,
                )
                self._require_durable_manifest_publication(
                    manifest_publication,
                    manifest_bytes,
                    native,
                )
                self._store.confirm_committed_resume_pair(
                    destination.relative_path,
                    manifest_bytes,
                )
                self._store.confirm_checkpoint_final(checkpoint_pin)
                boundary_confirmed = True

                if timing is None:  # pragma: no cover - recording assigns it.
                    raise CalibrationV3TransactionStop("checkpoint timing receipt was lost")
                if timing.state is persistence.TimingState.OVER_CAP:
                    stopped = CalibrationV3TransactionStop(
                        "cumulative active-wall cap was crossed during checkpoint publication"
                    )
                    raise self._terminalize(
                        stopped,
                        timing_state=TransactionTimingState.OVER_CAP,
                        timing=timing,
                        checkpoint_publication=publication,
                        manifest_publication=manifest_publication,
                    ) from stopped
                if self._next_resume_index is None:  # pragma: no cover - checked above.
                    raise CalibrationV3TransactionStop("next resume index was lost")
                if publication is None or manifest_publication is None:
                    raise CalibrationV3TransactionStop("publication receipts were lost")

                sealed_token = object()
                commit = CommittedResumeCheckpoint(
                    self,
                    sealed_token,
                    timing,
                    publication,
                    manifest_publication,
                )
                self._progress = progress
                self._next_resume_index += 1
                self._sealed_checkpoint_publication = publication
                self._sealed_commit = commit
                self._sealed_manifest_publication = manifest_publication
                self._sealed_timing = timing
                self._sealed_token = sealed_token
                self._state = _CoordinatorState.SEALED
            except BaseException as exc:
                if self._state in {
                    _CoordinatorState.TERMINAL_STOP,
                    _CoordinatorState.TERMINAL_OVER_CAP,
                } and isinstance(exc, CalibrationV3TransactionStop):
                    raise
                cleanup_failures += self._close_handle_once(
                    verifier,
                    stage="verifier_close",
                )
                stopped = self._terminalize(
                    exc,
                    timing_state=(
                        TransactionTimingState.WITHIN_CAP
                        if boundary_confirmed
                        else TransactionTimingState.TIMING_INDETERMINATE
                    ),
                    timing=timing,
                    checkpoint_publication=publication,
                    manifest_publication=manifest_publication,
                    cleanup_failures=cleanup_failures,
                )
                if isinstance(exc, Exception):
                    raise stopped from exc
                raise
            if commit is None:  # pragma: no cover - success constructs it above.
                raise CalibrationV3TransactionStop("sealed commit capability was lost")
            return commit

    def continue_after(
        self,
        commit: CommittedResumeCheckpoint,
    ) -> CalibrationV3Session:
        """Consume one clean commit and start the next active interval."""

        with self._operation_lock:
            self._require_sealed_commit(commit)
            timing = self._sealed_timing
            if type(timing) is not persistence.CumulativeActiveWallTiming:
                raise CalibrationV3TransactionStop("sealed timing type is invalid")
            if timing.state is not persistence.TimingState.WITHIN_CAP:
                raise CalibrationV3TransactionStop(
                    "cumulative active-wall cap forbids continuation",
                    timing_state=TransactionTimingState.OVER_CAP,
                    timing=timing,
                )
            if timing.cumulative_active_ns is None:  # pragma: no cover - state checked.
                raise CalibrationV3TransactionStop("sealed timing is indeterminate")
            active_token = object()
            session = CalibrationV3Session(owner=self, token=active_token)
            start_ns = self._clock_value()
            if self._last_clock_ns is None or start_ns < self._last_clock_ns:
                raise CalibrationV3TransactionStop("active-wall clock moved backward")
            self._start_ns = start_ns
            self._last_clock_ns = start_ns
            self._prior_cumulative_ns = timing.cumulative_active_ns
            self._active_token = active_token
            self._sealed_checkpoint_publication = None
            self._sealed_commit = None
            self._sealed_manifest_publication = None
            self._sealed_timing = None
            self._sealed_token = None
            self._state = _CoordinatorState.ACTIVE
            return session

    def _close_committed(self, commit: CommittedResumeCheckpoint) -> None:
        with self._operation_lock:
            self._require_sealed_commit(commit)
            timing = self._sealed_timing
            checkpoint_publication = self._sealed_checkpoint_publication
            manifest_publication = self._sealed_manifest_publication
            circuit = self._circuit
            self._circuit = None
            self._sealed_checkpoint_publication = None
            self._sealed_commit = None
            self._sealed_manifest_publication = None
            self._sealed_timing = None
            self._sealed_token = None
            self._state = _CoordinatorState.CLOSED
            cleanup = self._close_handle_once(
                circuit,
                stage="committed_circuit_close",
            )
            if cleanup:
                stopped = CalibrationV3TransactionStop(
                    "committed circuit close failed",
                    timing_state=TransactionTimingState.WITHIN_CAP,
                    timing=timing,
                    checkpoint_publication=checkpoint_publication,
                    manifest_publication=manifest_publication,
                    cleanup_failures=cleanup,
                    manifest_reconciliation=self._manifest_reconciliation,
                )
                self._terminal_receipt = stopped._receipt
                raise stopped

    def _validate_resume_payload(
        self,
        payload: dict[str, Any],
        destination: persistence.DestinationSpec,
        expected_circuit_identity: str,
        native: Any,
    ) -> tuple[object, FrozenResumeProgress, persistence.CumulativeActiveWallTiming]:
        if (
            payload["circuit_contract"] != self._circuit_contract
            or payload["train_input"] != self._train_input
        ):
            raise CalibrationV3TransactionStop("resume manifest contract binding drifted")
        if destination.resume_cursor is None or payload["cursor"] != dict(
            destination.resume_cursor
        ):
            raise CalibrationV3TransactionStop("resume manifest cursor drifted")
        if destination.arm_slug is None or destination.construction_seed is None:
            raise CalibrationV3TransactionStop("resume destination identity is incomplete")
        expected_reference = (
            f"r3-f0/deterministic-confirmation/{destination.arm_slug}/"
            f"construction-seed-{destination.construction_seed:010d}/fresh"
        )
        if payload["reference_trajectory_id"] != expected_reference:
            raise CalibrationV3TransactionStop("resume reference trajectory drifted")
        identity = _parse_identity(payload["checkpoint"], native)
        self._validate_identity_authority(identity, expected_circuit_identity, native)
        _parse_checkpoint_publication(
            payload["checkpoint_publication"],
            native,
            expected_checkpoint_bytes=identity.checkpoint_bytes,
        )
        progress = _parse_progress(payload["stream_prefixes"], payload["previous_state"])
        self._validate_progress(progress, destination, identity)
        timing_value = _exact_keys(
            payload["timing"],
            frozenset({"prior_cumulative_ns", "session_active_ns", "cumulative_active_ns"}),
            label="resume timing",
        )
        try:
            timing = persistence.CumulativeActiveWallTiming(
                prior_cumulative_ns=_exact_nonnegative_int(
                    timing_value["prior_cumulative_ns"],
                    label="resume prior_cumulative_ns",
                ),
                session_active_ns=_exact_nonnegative_int(
                    timing_value["session_active_ns"],
                    label="resume session_active_ns",
                ),
                cumulative_active_ns=_exact_nonnegative_int(
                    timing_value["cumulative_active_ns"],
                    label="resume cumulative_active_ns",
                ),
            )
        except ValueError as exc:
            raise CalibrationV3TransactionStop("resume timing arithmetic is invalid") from exc
        return identity, progress, timing

    def open_resume(
        self,
        manifest_path: str,
        definition: object,
    ) -> CalibrationV3Session:
        """Start the coordinator's sole resumed branch at one exact manifest."""

        with self._operation_lock:
            self._require_ready()
            try:
                destination = self._store.require_resume_manifest_destination(manifest_path)
            except persistence.PersistenceStop as exc:
                raise CalibrationV3TransactionStop(
                    f"resume manifest destination is invalid: {exc}"
                ) from exc
            native = _native_runtime()
            expected_identity = self._definition_authority(definition, destination, native)
            run = _RunCoordinates.from_destination(destination)
            if destination.resume_cursor is None:
                raise CalibrationV3TransactionStop("resume destination has no frozen cursor")
            current_event = dict(destination.resume_cursor)["after_transition_event"]
            try:
                next_index = self._resume_events.index(current_event) + 1
            except ValueError as exc:  # pragma: no cover - store authority excludes this.
                raise CalibrationV3TransactionStop(
                    "resume destination lacks milestone authority"
                ) from exc
            self._prepare_durable_namespace(native)
            start_ns = self._clock_value()
            self._native = native
            self._definition = definition
            self._expected_circuit_identity = expected_identity
            self._run_coordinates = run
            self._next_resume_index = next_index
            self._progress = None
            self._manifest_reconciliation = None
            self._start_ns = start_ns
            self._last_clock_ns = start_ns
            self._prior_cumulative_ns = None
            self._state = _CoordinatorState.STARTING
            timing: persistence.CumulativeActiveWallTiming | None = None
            reconciliation: object | None = None
            try:
                pair = self._store.open_committed_resume_pair(manifest_path)
                recorded_identity, progress, prior_timing = self._validate_resume_payload(
                    pair.manifest.payload,
                    destination,
                    expected_identity,
                    native,
                )
                timing = prior_timing
                if prior_timing.state is not persistence.TimingState.WITHIN_CAP:
                    raise CalibrationV3TransactionStop(
                        "resume prior cumulative time already exceeds the cap"
                    )
                reconciliation = native.reconcile_existing_immutable_bytes(
                    pair.manifest_pin.manifest_path,
                    pair.manifest_bytes,
                )
                _require_durable_manifest_reconciliation(
                    reconciliation,
                    pair.manifest_bytes,
                    native,
                )
                self._manifest_reconciliation = reconciliation
                pair = self._store.confirm_committed_resume_pair(
                    manifest_path,
                    pair.manifest_bytes,
                )
                if prior_timing.cumulative_active_ns is None:  # pragma: no cover - validated.
                    raise CalibrationV3TransactionStop("resume prior timing is indeterminate")
                self._prior_cumulative_ns = prior_timing.cumulative_active_ns
                checkpoint_pin = self._store.pin_checkpoint_final(manifest_path)
                self._circuit = native.NativeTemporalCircuit.from_checkpoint_path(
                    definition,
                    checkpoint_pin.checkpoint_path,
                )
                imported_identity = self._circuit.checkpoint_identity()
                self._validate_identity_authority(
                    imported_identity,
                    expected_identity,
                    native,
                )
                if imported_identity != recorded_identity:
                    raise CalibrationV3TransactionStop(
                        "resumed checkpoint identity lacks manifest parity"
                    )
                self._validate_progress(progress, destination, imported_identity)
                self._store.confirm_checkpoint_final(checkpoint_pin)
                self._store.confirm_manifest_final(pair.manifest_pin)
                self._progress = progress
                self._active_token = object()
                self._state = _CoordinatorState.ACTIVE
                session = CalibrationV3Session(owner=self, token=self._active_token)
                timing = self._current_timing()
            except BaseException as exc:
                timing_state = (
                    TransactionTimingState.OVER_CAP
                    if timing is not None and timing.state is persistence.TimingState.OVER_CAP
                    else TransactionTimingState.TIMING_INDETERMINATE
                )
                stopped = self._terminalize(
                    exc,
                    timing_state=timing_state,
                    timing=timing,
                    manifest_reconciliation=reconciliation,
                )
                if isinstance(exc, Exception):
                    raise stopped from exc
                raise
            if timing.state is persistence.TimingState.OVER_CAP:
                stopped = CalibrationV3TransactionStop(
                    "cumulative active-wall cap was crossed during resume import"
                )
                raise self._terminalize(
                    stopped,
                    timing_state=TransactionTimingState.OVER_CAP,
                    timing=timing,
                ) from stopped
            return session


__all__ = [
    "CalibrationV3CleanupFailure",
    "CalibrationV3Session",
    "CalibrationV3Transaction",
    "CalibrationV3TransactionStop",
    "CommittedResumeCheckpoint",
    "FrozenResumeProgress",
    "TransactionTimingState",
]
