"""Authority-gated Python capability for the native temporal circuit runtime.

Importing this module defines transport records only.  It neither loads the
native library nor starts an experiment; a validated live authority is required
by :class:`NativeTemporalCircuit` before any native entry point is reached.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import math
import mmap
import os
import platform
import stat
import struct
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import ModuleType
from typing import BinaryIO, Final

from humanbrain.native import (
    NativeBuildError as _NativeBuildError,
    NativeLibraryIdentity as _NativeLibraryIdentity,
    _load_library,
    _source_digest as _current_native_source_digest,
    loaded_library_identity as _loaded_library_identity,
    procedural_runtime_source_bundle_sha256 as _procedural_runtime_source_bundle_sha256,
    procedural_runtime_source_manifest as _procedural_runtime_source_manifest,
)

_UINT8_MAX: Final = (1 << 8) - 1
_UINT16_MAX: Final = (1 << 16) - 1
_UINT32_MAX: Final = (1 << 32) - 1
_ACTIVE_COLUMNS: Final = 11
_CONTEXT_CELLS: Final = 22
_PAIRS_PER_SEGMENT: Final = 10
_MAX_PERMANENCE_TICKS: Final = 500
_CHECKPOINT_V1_MAGIC: Final = b"R3NTCP1\x00"
_CHECKPOINT_V1_FORMAT_VERSION: Final = 1
_CHECKPOINT_V1_HEADER_BYTES: Final = 256
_CHECKPOINT_V1_BASIC_HEADER_BYTES: Final = 24
_CHECKPOINT_LOGICAL_STATE_DOMAIN: Final = b"humanbrain-r3-checkpoint-logical-state-v1\x00"
_MAX_IMMUTABLE_PAYLOAD_BYTES: Final = 1 << 20
_GOVERNING_PROTOCOL_MODULE: Final = "experiments.r3_pilot.calibration_v3_protocol"
_GATE3_AUTHORITY_MODULE: Final = "experiments.r3_pilot.calibration_v3_gate3_authority"


class _NativeAuthority(ctypes.Structure):
    _fields_ = [
        ("protocol_identity_sha256", ctypes.c_uint8 * 32),
        ("source_bundle_identity_sha256", ctypes.c_uint8 * 32),
    ]


class _NativeCheckpointIdentity(ctypes.Structure):
    _fields_ = [("header", ctypes.c_uint8 * _CHECKPOINT_V1_HEADER_BYTES)]


class _NativeConfig(ctypes.Structure):
    _fields_ = [
        ("column_count", ctypes.c_uint32),
        ("max_pyramidal_cells", ctypes.c_uint32),
        ("segment_capacity", ctypes.c_uint32),
        ("construction_seed", ctypes.c_uint32),
        ("connected_permanence_ticks", ctypes.c_uint16),
        ("initial_permanence_ticks", ctypes.c_uint16),
        ("permanence_increment_ticks", ctypes.c_uint16),
        ("negative_decrement_ticks", ctypes.c_uint16),
        ("base_cells_per_column", ctypes.c_uint8),
        ("novel_race_cap", ctypes.c_uint8),
        ("prediction_quorum", ctypes.c_uint8),
        ("max_segments_per_cell", ctypes.c_uint8),
        ("theta_match", ctypes.c_uint8),
        ("theta_active", ctypes.c_uint8),
        ("context_policy", ctypes.c_uint8),
        ("plasticity_policy", ctypes.c_uint8),
    ]


class _NativeTokenCode(ctypes.Structure):
    _fields_ = [
        ("token_id", ctypes.c_uint32),
        ("columns", ctypes.c_uint32 * _ACTIVE_COLUMNS),
    ]


class _NativeDefinition(ctypes.Structure):
    _fields_ = [
        ("config", _NativeConfig),
        ("token_codes", ctypes.POINTER(_NativeTokenCode)),
        ("token_code_count", ctypes.c_uint64),
        ("threshold_offsets_mv", ctypes.POINTER(ctypes.c_double)),
        ("threshold_offset_count", ctypes.c_uint64),
        ("authority", _NativeAuthority),
    ]


class _NativeDefinitionIdentityRequest(ctypes.Structure):
    _fields_ = [
        ("config", _NativeConfig),
        ("token_codes", ctypes.POINTER(_NativeTokenCode)),
        ("token_code_count", ctypes.c_uint64),
        ("threshold_offsets_mv", ctypes.POINTER(ctypes.c_double)),
        ("threshold_offset_count", ctypes.c_uint64),
    ]


class _NativeDefinitionIdentity(ctypes.Structure):
    _fields_ = [("circuit_identity_sha256", ctypes.c_uint8 * 32)]


class _NativeImportRequest(ctypes.Structure):
    _fields_ = [
        ("definition", _NativeDefinition),
        ("checkpoint_bytes", ctypes.POINTER(ctypes.c_uint8)),
        ("checkpoint_len", ctypes.c_uint64),
    ]


class _NativeCheckpointPathRequest(ctypes.Structure):
    _fields_ = [
        ("parent_utf8", ctypes.POINTER(ctypes.c_uint8)),
        ("parent_len", ctypes.c_uint64),
        ("final_basename_utf8", ctypes.POINTER(ctypes.c_uint8)),
        ("final_basename_len", ctypes.c_uint64),
        ("reserved_zero", ctypes.c_uint8 * 8),
    ]


class _NativeImmutableBytesPathRequest(ctypes.Structure):
    _fields_ = [
        ("parent_utf8", ctypes.POINTER(ctypes.c_uint8)),
        ("parent_len", ctypes.c_uint64),
        ("final_basename_utf8", ctypes.POINTER(ctypes.c_uint8)),
        ("final_basename_len", ctypes.c_uint64),
        ("payload", ctypes.POINTER(ctypes.c_uint8)),
        ("payload_len", ctypes.c_uint64),
        ("reserved_zero", ctypes.c_uint8 * 8),
    ]


class _NativeDurableDirectoryPathRequest(ctypes.Structure):
    _fields_ = [
        ("root_utf8", ctypes.POINTER(ctypes.c_uint8)),
        ("root_len", ctypes.c_uint64),
        ("relative_directory_utf8", ctypes.POINTER(ctypes.c_uint8)),
        ("relative_directory_len", ctypes.c_uint64),
        ("reserved_zero", ctypes.c_uint8 * 8),
    ]


class _NativeCheckpointPublicationError(ctypes.Structure):
    _fields_ = [
        ("stage", ctypes.c_uint16),
        ("domain", ctypes.c_uint16),
        ("code", ctypes.c_uint32),
    ]


class _NativeCheckpointPublicationReport(ctypes.Structure):
    _fields_ = [
        ("outcome", ctypes.c_uint16),
        ("temp_cleanup", ctypes.c_uint8),
        ("cleanup_error_count", ctypes.c_uint8),
        ("primary_error", _NativeCheckpointPublicationError),
        ("cleanup_errors", _NativeCheckpointPublicationError * 4),
        ("owned_temp_pid", ctypes.c_uint32),
        ("owned_temp_nonce", ctypes.c_uint64),
        ("checkpoint_bytes", ctypes.c_uint64),
        ("encoded_bytes", ctypes.c_uint64),
        ("reserved_zero", ctypes.c_uint8 * 8),
    ]


class _NativeImmutableBytesReconciliationReport(ctypes.Structure):
    _fields_ = [
        ("outcome", ctypes.c_uint16),
        ("cleanup_error_count", ctypes.c_uint8),
        ("reserved_zero_u8", ctypes.c_uint8),
        ("primary_error", _NativeCheckpointPublicationError),
        ("cleanup_errors", _NativeCheckpointPublicationError * 2),
        ("reserved_zero_u32", ctypes.c_uint32),
        ("required_bytes", ctypes.c_uint64),
        ("verified_bytes", ctypes.c_uint64),
        ("reserved_zero", ctypes.c_uint8 * 16),
    ]


class _NativeDurableDirectoryPreparationReport(ctypes.Structure):
    _fields_ = [
        ("outcome", ctypes.c_uint16),
        ("cleanup_error_count", ctypes.c_uint8),
        ("reserved_zero_u8", ctypes.c_uint8),
        ("primary_error", _NativeCheckpointPublicationError),
        ("cleanup_errors", _NativeCheckpointPublicationError * 2),
        ("required_component_count", ctypes.c_uint32),
        ("confirmed_component_count", ctypes.c_uint32),
        ("created_component_count", ctypes.c_uint32),
        ("failed_component_index", ctypes.c_uint32),
        ("reserved_zero_u32", ctypes.c_uint32),
        ("reserved_zero", ctypes.c_uint8 * 16),
    ]


class _PyBuffer(ctypes.Structure):
    """CPython 3.11+ stable-ABI ``Py_buffer`` transport layout."""

    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.c_void_p),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_void_p),
        ("shape", ctypes.c_void_p),
        ("strides", ctypes.c_void_p),
        ("suboffsets", ctypes.c_void_p),
        ("internal", ctypes.c_void_p),
    ]


class _NativeSpike(ctypes.Structure):
    _fields_ = [
        ("cell_id", ctypes.c_uint32),
        ("tick", ctypes.c_uint16),
        ("reserved_zero", ctypes.c_uint16),
    ]


class _NativeStep(ctypes.Structure):
    _fields_ = [
        ("token_id", ctypes.c_uint32),
        ("punished_segments", ctypes.c_uint32),
        ("event_count", ctypes.c_uint64),
        ("predicted", ctypes.c_uint8),
        ("predicted_columns", ctypes.c_uint8),
        ("allocated_segments", ctypes.c_uint8),
        ("reserved_zero", ctypes.c_uint8),
        ("inhibition_onset_ticks", ctypes.c_uint16 * _ACTIVE_COLUMNS),
        ("column_simulated_ticks", ctypes.c_uint16 * _ACTIVE_COLUMNS),
        ("winner_spikes", _NativeSpike * _ACTIVE_COLUMNS),
        ("basket_spikes", _NativeSpike * _ACTIVE_COLUMNS),
        ("proximal_spikes", _NativeSpike * _ACTIVE_COLUMNS),
        ("proximal_relay_spikes", _NativeSpike * _ACTIVE_COLUMNS),
        ("winner_relay_spikes", _NativeSpike * _ACTIVE_COLUMNS),
        ("winner_context", ctypes.c_uint32 * _CONTEXT_CELLS),
        ("proximal_context", ctypes.c_uint32 * _CONTEXT_CELLS),
        ("context", ctypes.c_uint32 * _CONTEXT_CELLS),
    ]


class _NativeAdvanceRequest(ctypes.Structure):
    _fields_ = [
        ("token_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("output", ctypes.POINTER(_NativeStep)),
        ("token_count", ctypes.c_uint64),
        ("output_capacity", ctypes.c_uint64),
        ("mode", ctypes.c_uint8),
        ("reset_before", ctypes.c_uint8),
        ("reserved_zero", ctypes.c_uint8 * 6),
    ]


class _NativeReport(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("message_len", ctypes.c_uint32),
        ("completed_prefix", ctypes.c_uint64),
        ("required", ctypes.c_uint64),
        ("observed", ctypes.c_uint64),
        ("location", ctypes.c_uint64),
        ("poison_code", ctypes.c_uint16),
        ("stage_code", ctypes.c_uint16),
        ("reserved_zero", ctypes.c_uint32),
        ("message_utf8", ctypes.c_uint8 * 464),
    ]


def _require_frozen_layout(
    record: type[ctypes.Structure],
    size: int,
    alignment: int,
    offsets: tuple[int, ...],
) -> None:
    observed_offsets = tuple(getattr(record, name).offset for name, _ctype in record._fields_)
    if (
        ctypes.sizeof(record) != size
        or ctypes.alignment(record) != alignment
        or observed_offsets != offsets
    ):
        raise RuntimeError(f"unsupported native temporal ABI layout for {record.__name__}")


def _assert_frozen_abi() -> None:
    machine = platform.machine().strip().lower().replace("-", "_")
    if (
        ctypes.sizeof(ctypes.c_void_p) != 8
        or sys.byteorder != "little"
        or machine not in {"amd64", "x86_64"}
    ):
        raise RuntimeError("native temporal runtime requires little-endian x64 Python")
    _require_frozen_layout(_NativeAuthority, 64, 1, (0, 32))
    _require_frozen_layout(_NativeCheckpointIdentity, 256, 1, (0,))
    _require_frozen_layout(
        _NativeConfig,
        32,
        4,
        (0, 4, 8, 12, 16, 18, 20, 22, 24, 25, 26, 27, 28, 29, 30, 31),
    )
    _require_frozen_layout(_NativeTokenCode, 48, 4, (0, 4))
    _require_frozen_layout(_NativeDefinition, 128, 8, (0, 32, 40, 48, 56, 64))
    _require_frozen_layout(_NativeDefinitionIdentityRequest, 64, 8, (0, 32, 40, 48, 56))
    _require_frozen_layout(_NativeDefinitionIdentity, 32, 1, (0,))
    _require_frozen_layout(_NativeImportRequest, 144, 8, (0, 128, 136))
    _require_frozen_layout(_NativeCheckpointPathRequest, 40, 8, (0, 8, 16, 24, 32))
    _require_frozen_layout(
        _NativeImmutableBytesPathRequest,
        56,
        8,
        (0, 8, 16, 24, 32, 40, 48),
    )
    _require_frozen_layout(
        _NativeDurableDirectoryPathRequest,
        40,
        8,
        (0, 8, 16, 24, 32),
    )
    _require_frozen_layout(_NativeCheckpointPublicationError, 8, 4, (0, 2, 4))
    _require_frozen_layout(
        _NativeCheckpointPublicationReport,
        80,
        8,
        (0, 2, 3, 4, 12, 44, 48, 56, 64, 72),
    )
    _require_frozen_layout(
        _NativeImmutableBytesReconciliationReport,
        64,
        8,
        (0, 2, 3, 4, 12, 28, 32, 40, 48),
    )
    _require_frozen_layout(
        _NativeDurableDirectoryPreparationReport,
        64,
        4,
        (0, 2, 3, 4, 12, 28, 32, 36, 40, 44, 48),
    )
    _require_frozen_layout(
        _PyBuffer,
        80,
        8,
        (0, 8, 16, 24, 32, 36, 40, 48, 56, 64, 72),
    )
    _require_frozen_layout(_NativeSpike, 8, 4, (0, 4, 6))
    _require_frozen_layout(
        _NativeStep,
        768,
        8,
        (0, 4, 8, 16, 17, 18, 19, 20, 42, 64, 152, 240, 328, 416, 504, 592, 680),
    )
    _require_frozen_layout(_NativeAdvanceRequest, 40, 8, (0, 8, 16, 24, 32, 33, 34))
    _require_frozen_layout(
        _NativeReport,
        512,
        8,
        (0, 4, 8, 16, 24, 32, 40, 42, 44, 48),
    )


_assert_frozen_abi()


def _python_buffer_api() -> object:
    return ctypes.pythonapi


def _readonly_buffer_api(
    purpose: str = "checkpoint path import",
) -> tuple[object, object]:
    if platform.python_implementation() != "CPython":
        raise RuntimeError(f"{purpose} requires the CPython stable buffer ABI")
    if tuple(sys.version_info[:2]) < (3, 11):
        raise RuntimeError(f"{purpose} requires CPython 3.11 or newer")
    api = _python_buffer_api()
    try:
        get_buffer = getattr(api, "PyObject_GetBuffer")
        release_buffer = getattr(api, "PyBuffer_Release")
    except AttributeError as exc:
        raise RuntimeError(f"{purpose} requires PyObject_GetBuffer and PyBuffer_Release") from exc
    get_buffer.argtypes = [
        ctypes.py_object,
        ctypes.POINTER(_PyBuffer),
        ctypes.c_int,
    ]
    get_buffer.restype = ctypes.c_int
    release_buffer.argtypes = [ctypes.POINTER(_PyBuffer)]
    release_buffer.restype = None
    return get_buffer, release_buffer


class TemporalContextPolicy(IntEnum):
    """Frozen source family used to form distal segment contexts."""

    BOUND_WINNERS = 1
    GENERIC_AFFERENTS = 2


class TemporalPlasticityPolicy(IntEnum):
    """Frozen learned-state mutation policy."""

    GROWTH_OFF = 0
    PLASTIC = 1


class TemporalStepMode(IntEnum):
    """Only token IDs plus this explicit mutation choice reach native code."""

    READ_ONLY = 0
    LEARN = 1


class NativeTemporalRuntimeStatus(IntEnum):
    """Frozen status codes returned by every reporting runtime export."""

    OK = 0
    INVALID_ARGUMENT = 1
    INVALID_DEFINITION = 2
    CHECKPOINT_REJECTED = 3
    ALLOCATION_FAILED = 4
    OUTPUT_CAPACITY = 5
    UNKNOWN_TOKEN = 6
    POISONED = 7
    SCIENTIFIC_STOP = 8
    CAPACITY_STOP = 9
    INTERNAL_INVARIANT = 10


class TemporalCheckpointPublicationOutcome(IntEnum):
    """Frozen completion-plane outcome for exclusive checkpoint publication."""

    DURABLE_PUBLISHED = 1
    COLLISION = 2
    NOT_PUBLISHED = 3
    PUBLISHED_BUT_NOT_CONFIRMED_DURABLE = 4


class TemporalCheckpointCleanup(IntEnum):
    """Frozen disposition of the publisher-owned temporary artifact."""

    NOT_APPLICABLE = 0
    CLEAN = 1
    ORPHANED = 2


class TemporalCheckpointErrorDomain(IntEnum):
    """Frozen origin of one checkpoint-publication error code."""

    NONE = 0
    POSIX = 1
    WIN32 = 2
    POLICY = 3
    ENCODER = 4
    IO = 5


class TemporalCheckpointErrorStage(IntEnum):
    """Frozen ordered stage at which checkpoint publication encountered an error."""

    NONE = 0
    PARENT_OPEN = 1
    PARENT_IDENTITY = 2
    FILESYSTEM_POLICY = 3
    TEMP_CREATE = 4
    ENCODE = 5
    FILE_SYNC = 6
    FILE_CLOSE = 7
    TEMP_IDENTITY = 8
    EXCLUSIVE_PUBLISH = 9
    TEMP_CLEANUP = 10
    PARENT_SYNC = 11


class TemporalPublicationOutcome(IntEnum):
    """Completion-plane outcome for exclusive immutable-byte publication."""

    DURABLE_PUBLISHED = 1
    COLLISION = 2
    NOT_PUBLISHED = 3
    PUBLISHED_BUT_NOT_CONFIRMED_DURABLE = 4


class TemporalPublicationCleanup(IntEnum):
    """Disposition of the publisher-owned temporary artifact."""

    NOT_APPLICABLE = 0
    CLEAN = 1
    ORPHANED = 2


class TemporalPublicationErrorDomain(IntEnum):
    """Origin of one immutable-byte publication error code."""

    NONE = 0
    POSIX = 1
    WIN32 = 2
    POLICY = 3
    CHECKPOINT_ENCODER_RESERVED = 4
    IO = 5


class TemporalPublicationErrorStage(IntEnum):
    """Ordered stage at which immutable-byte publication encountered an error."""

    NONE = 0
    PARENT_OPEN = 1
    PARENT_IDENTITY = 2
    FILESYSTEM_POLICY = 3
    TEMP_CREATE = 4
    WRITE = 5
    FILE_SYNC = 6
    FILE_CLOSE = 7
    TEMP_IDENTITY = 8
    EXCLUSIVE_PUBLISH = 9
    TEMP_CLEANUP = 10
    PARENT_SYNC = 11


class TemporalDurabilityOutcome(IntEnum):
    """Whether native storage durability was established by this call."""

    DURABLE = 1
    NOT_DURABLE = 2


class TemporalDurabilityErrorDomain(IntEnum):
    """Origin of one immutable-path durability error."""

    NONE = 0
    POSIX = 1
    WIN32 = 2
    POLICY = 3
    CHECKPOINT_ENCODER_RESERVED = 4
    IO = 5


class TemporalDurabilityErrorStage(IntEnum):
    """Frozen stage at which durability establishment failed."""

    NONE = 0
    PARENT_OPEN = 1
    PARENT_IDENTITY = 2
    FILESYSTEM_POLICY = 3
    FINAL_OPEN = 4
    FINAL_IDENTITY = 5
    FINAL_READ = 6
    PAYLOAD_MISMATCH = 7
    FILE_SYNC = 8
    DIRECTORY_CREATE = 9
    DIRECTORY_OPEN = 10
    DIRECTORY_IDENTITY = 11
    DIRECTORY_SYNC = 12
    FILE_CLOSE = 13
    PATH_IDENTITY = 14


_PRE_TEMP_PUBLICATION_ERROR_STAGES: Final = frozenset(
    {
        TemporalCheckpointErrorStage.PARENT_OPEN,
        TemporalCheckpointErrorStage.PARENT_IDENTITY,
        TemporalCheckpointErrorStage.FILESYSTEM_POLICY,
        TemporalCheckpointErrorStage.TEMP_CREATE,
        TemporalCheckpointErrorStage.PARENT_SYNC,
    }
)
_TEMP_BACKED_NOT_PUBLISHED_ERROR_STAGES: Final = frozenset(
    {
        TemporalCheckpointErrorStage.PARENT_IDENTITY,
        TemporalCheckpointErrorStage.ENCODE,
        TemporalCheckpointErrorStage.FILE_SYNC,
        TemporalCheckpointErrorStage.FILE_CLOSE,
        TemporalCheckpointErrorStage.TEMP_IDENTITY,
        TemporalCheckpointErrorStage.EXCLUSIVE_PUBLISH,
    }
)
_UNCONFIRMED_PUBLICATION_ERROR_STAGES: Final = frozenset(
    {
        TemporalCheckpointErrorStage.EXCLUSIVE_PUBLISH,
        TemporalCheckpointErrorStage.PARENT_SYNC,
    }
)


@dataclass(frozen=True, slots=True)
class TemporalCheckpointIdentity:
    """Canonical checkpoint-v1 identity and state fields for one live circuit."""

    format_version: int
    header_bytes: int
    checkpoint_bytes: int
    body_bytes: int
    protocol_identity_sha256: str
    source_bundle_identity_sha256: str
    circuit_identity_sha256: str
    body_sha256: str
    semantic_index_sha256: str
    segment_count: int
    cohort_count: int
    next_reserve_cell: int
    event_count: int
    learning_step: int
    prior_context_length: int
    winner_context_length: int
    poison_code: int
    logical_state_sha256: str


@dataclass(frozen=True, slots=True)
class TemporalCheckpointPublicationError:
    """One native publication error; tuple order is the native cleanup order."""

    stage: TemporalCheckpointErrorStage
    domain: TemporalCheckpointErrorDomain
    code: int


@dataclass(frozen=True, slots=True)
class TemporalCheckpointPublication:
    """Immutable completion report; only durable publication is resumable."""

    outcome: TemporalCheckpointPublicationOutcome
    temp_cleanup: TemporalCheckpointCleanup
    primary_error: TemporalCheckpointPublicationError | None
    cleanup_errors: tuple[TemporalCheckpointPublicationError, ...]
    owned_temp_pid: int
    owned_temp_nonce: int
    checkpoint_bytes: int
    encoded_bytes: int

    @property
    def semantically_resumable(self) -> bool:
        return self.outcome is TemporalCheckpointPublicationOutcome.DURABLE_PUBLISHED


@dataclass(frozen=True, slots=True)
class TemporalPublicationError:
    """One native immutable-byte publication error in native cleanup order."""

    stage: TemporalPublicationErrorStage
    domain: TemporalPublicationErrorDomain
    code: int


@dataclass(frozen=True, slots=True)
class TemporalImmutableBytesPublication:
    """Immutable result of one no-overwrite publication attempt."""

    outcome: TemporalPublicationOutcome
    temp_cleanup: TemporalPublicationCleanup
    primary_error: TemporalPublicationError | None
    cleanup_errors: tuple[TemporalPublicationError, ...]
    owned_temp_pid: int
    owned_temp_nonce: int
    payload_bytes: int
    written_bytes: int

    @property
    def durably_published(self) -> bool:
        return self.outcome is TemporalPublicationOutcome.DURABLE_PUBLISHED


@dataclass(frozen=True, slots=True)
class TemporalDurabilityError:
    """One native durability error in preserved cleanup order."""

    stage: TemporalDurabilityErrorStage
    domain: TemporalDurabilityErrorDomain
    code: int


@dataclass(frozen=True, slots=True)
class TemporalImmutableBytesReconciliation:
    """Receipt for flushing an existing exact immutable byte artifact."""

    outcome: TemporalDurabilityOutcome
    primary_error: TemporalDurabilityError | None
    cleanup_errors: tuple[TemporalDurabilityError, ...]
    expected_bytes: int
    verified_bytes: int

    @property
    def durably_reconciled(self) -> bool:
        return self.outcome is TemporalDurabilityOutcome.DURABLE


@dataclass(frozen=True, slots=True)
class TemporalDurableDirectoryPreparation:
    """Receipt for one recursively flushed relative directory closure."""

    outcome: TemporalDurabilityOutcome
    primary_error: TemporalDurabilityError | None
    cleanup_errors: tuple[TemporalDurabilityError, ...]
    required_component_count: int
    confirmed_component_count: int
    created_component_count: int
    failed_component_index: int | None

    @property
    def durably_prepared(self) -> bool:
        return self.outcome is TemporalDurabilityOutcome.DURABLE


def _exact_int(label: str, value: object, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class NativeTemporalCircuitConfig:
    """Exact checkpoint-v1 circuit configuration; every policy is explicit."""

    column_count: int
    max_pyramidal_cells: int
    segment_capacity: int
    construction_seed: int
    connected_permanence_ticks: int
    initial_permanence_ticks: int
    permanence_increment_ticks: int
    negative_decrement_ticks: int
    base_cells_per_column: int
    novel_race_cap: int
    prediction_quorum: int
    max_segments_per_cell: int
    theta_match: int
    theta_active: int
    context_policy: TemporalContextPolicy
    plasticity_policy: TemporalPlasticityPolicy

    def __post_init__(self) -> None:
        _exact_int("column_count", self.column_count, _ACTIVE_COLUMNS, _UINT32_MAX)
        _exact_int("max_pyramidal_cells", self.max_pyramidal_cells, 1, _UINT32_MAX)
        _exact_int(
            "segment_capacity",
            self.segment_capacity,
            1,
            _UINT32_MAX // _PAIRS_PER_SEGMENT,
        )
        _exact_int("construction_seed", self.construction_seed, 0, _UINT32_MAX)
        for label, value in (
            ("connected_permanence_ticks", self.connected_permanence_ticks),
            ("initial_permanence_ticks", self.initial_permanence_ticks),
            ("permanence_increment_ticks", self.permanence_increment_ticks),
            ("negative_decrement_ticks", self.negative_decrement_ticks),
        ):
            _exact_int(label, value, 1, _MAX_PERMANENCE_TICKS)
        _exact_int("base_cells_per_column", self.base_cells_per_column, 1, _UINT8_MAX)
        _exact_int("novel_race_cap", self.novel_race_cap, 1, 8)
        _exact_int("prediction_quorum", self.prediction_quorum, 1, _ACTIVE_COLUMNS)
        _exact_int("max_segments_per_cell", self.max_segments_per_cell, 1, _UINT8_MAX)
        _exact_int("theta_match", self.theta_match, 1, 2 * _PAIRS_PER_SEGMENT)
        _exact_int("theta_active", self.theta_active, 1, 2 * _PAIRS_PER_SEGMENT)
        if self.theta_match > self.theta_active:
            raise ValueError("theta_match cannot exceed theta_active")
        if self.base_cells_per_column > self.novel_race_cap:
            raise ValueError("base_cells_per_column cannot exceed novel_race_cap")
        if type(self.context_policy) is not TemporalContextPolicy:
            raise TypeError("context_policy must be a TemporalContextPolicy")
        if type(self.plasticity_policy) is not TemporalPlasticityPolicy:
            raise TypeError("plasticity_policy must be a TemporalPlasticityPolicy")
        if (
            self.context_policy is TemporalContextPolicy.GENERIC_AFFERENTS
            and self.base_cells_per_column < 2
        ):
            raise ValueError("generic-afferent policy requires at least two base cells")
        base_population = self.column_count * self.base_cells_per_column
        if base_population > self.max_pyramidal_cells:
            raise ValueError("base population exceeds max_pyramidal_cells")
        if _physical_cell_count(self) > _UINT32_MAX:
            raise ValueError("configured physical cell offsets overflow uint32")


_TokenCodeInput = Mapping[int, Sequence[int]] | Sequence[tuple[int, Sequence[int]]]


def _physical_cell_count(config: NativeTemporalCircuitConfig) -> int:
    return 2 * config.max_pyramidal_cells + 3 * config.column_count


@dataclass(frozen=True, slots=True, init=False)
class NativeTemporalCircuitDefinition:
    """Owned canonical codebook and bit-preserved finite f64 threshold table."""

    config: NativeTemporalCircuitConfig
    token_codes: tuple[tuple[int, tuple[int, ...]], ...]
    threshold_offsets_mv: tuple[float, ...]

    def __init__(
        self,
        config: NativeTemporalCircuitConfig,
        token_codes: _TokenCodeInput,
        threshold_offsets_mv: Sequence[float],
    ) -> None:
        if type(config) is not NativeTemporalCircuitConfig:
            raise TypeError("config must be a NativeTemporalCircuitConfig")
        raw_items = token_codes.items() if isinstance(token_codes, Mapping) else token_codes
        normalized: list[tuple[int, tuple[int, ...]]] = []
        seen_tokens: set[int] = set()
        seen_columns: dict[tuple[int, ...], int] = {}
        for index, raw_entry in enumerate(raw_items):
            try:
                raw_token, raw_columns = raw_entry
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"token_codes entry {index} must be a (token_id, columns) pair"
                ) from exc
            token = _exact_int("token ID", raw_token, 0, _UINT32_MAX - 1)
            if token in seen_tokens:
                raise ValueError(f"duplicate token ID {token}")
            seen_tokens.add(token)
            try:
                columns = tuple(
                    sorted(
                        _exact_int("column ID", column, 0, config.column_count - 1)
                        for column in raw_columns
                    )
                )
            except TypeError as exc:
                raise TypeError(f"token {token} columns must be a sequence") from exc
            if len(columns) != _ACTIVE_COLUMNS or len(set(columns)) != _ACTIVE_COLUMNS:
                raise ValueError(
                    f"token {token} must activate exactly {_ACTIVE_COLUMNS} unique columns"
                )
            duplicate = seen_columns.get(columns)
            if duplicate is not None:
                raise ValueError(f"tokens {duplicate} and {token} must not share one column code")
            seen_columns[columns] = token
            normalized.append((token, columns))
        if not normalized:
            raise ValueError("token_codes must not be empty")
        normalized.sort(key=lambda entry: entry[0])

        thresholds = tuple(threshold_offsets_mv)
        physical_cell_count = _physical_cell_count(config)
        if len(thresholds) != physical_cell_count:
            raise ValueError(
                f"threshold_offsets_mv must contain exactly {physical_cell_count} entries"
            )
        if any(type(value) is not float or not math.isfinite(value) for value in thresholds):
            raise ValueError("threshold_offsets_mv entries must be finite exact f64 values")

        object.__setattr__(self, "config", config)
        object.__setattr__(self, "token_codes", tuple(normalized))
        object.__setattr__(self, "threshold_offsets_mv", thresholds)

    def circuit_identity_sha256(self) -> str:
        """Return the native Route-0 identity without opening a runtime."""

        library = _temporal_library()
        _attested_current_library_identity()
        request, token_array, threshold_array = _native_definition_identity_request(self)
        output = _NativeDefinitionIdentity()
        report = _NativeReport()
        returned = library.humanbrain_temporal_circuit_definition_identity_v1(
            ctypes.byref(request),
            ctypes.byref(output),
            ctypes.byref(report),
        )
        # Keep borrowed ctypes arrays alive until the native call has returned.
        del token_array, threshold_array
        decoded = _decode_report(returned, report, maximum_completed_prefix=0)
        identity = bytes(output.circuit_identity_sha256)
        if decoded.status is not NativeTemporalRuntimeStatus.OK:
            if any(identity):
                raise _invariant_error(
                    "failed circuit definition identity call returned nonzero output"
                )
            _raise_report(decoded)
        if decoded.required != 32 or decoded.observed != 32 or decoded.location != 0:
            raise _invariant_error(
                "successful circuit definition identity has invalid diagnostics",
                required=32,
                observed=decoded.observed,
                location=decoded.location,
            )
        return identity.hex()


@dataclass(frozen=True, slots=True)
class TemporalSpike:
    cell_id: int
    tick: int


@dataclass(frozen=True, slots=True)
class TemporalCircuitStep:
    token_id: int
    punished_segments: int
    event_count: int
    predicted: bool
    predicted_columns: int
    allocated_segments: int
    inhibition_onset_ticks: tuple[int, ...]
    column_simulated_ticks: tuple[int, ...]
    winner_spikes: tuple[TemporalSpike, ...]
    basket_spikes: tuple[TemporalSpike, ...]
    proximal_spikes: tuple[TemporalSpike, ...]
    proximal_relay_spikes: tuple[TemporalSpike, ...]
    winner_relay_spikes: tuple[TemporalSpike, ...]
    winner_context: tuple[int, ...]
    proximal_context: tuple[int, ...]
    context: tuple[int, ...]


class NativeTemporalRuntimeError(RuntimeError):
    """One native failure, including any prefix already committed by a block."""

    def __init__(
        self,
        status: NativeTemporalRuntimeStatus,
        message: str,
        *,
        completed_prefix: int = 0,
        required: int = 0,
        observed: int = 0,
        location: int = 0,
        poison_code: int = 0,
        stage_code: int = 0,
        completed_steps: tuple[TemporalCircuitStep, ...] = (),
    ) -> None:
        if type(status) is not NativeTemporalRuntimeStatus:
            raise TypeError("status must be a NativeTemporalRuntimeStatus")
        if type(message) is not str:
            raise TypeError("message must be text")
        if not 0 <= len(completed_steps) <= completed_prefix:
            raise ValueError("retained steps cannot exceed the native committed prefix")
        self.status = status
        self.message = message
        self.completed_prefix = completed_prefix
        self.required = required
        self.observed = observed
        self.location = location
        self.poison_code = poison_code
        self.stage_code = stage_code
        self.completed_steps = completed_steps
        detail = f"{status.name.lower()}: {message}"
        if completed_prefix:
            detail += f" (committed prefix: {completed_prefix})"
        if len(completed_steps) != completed_prefix:
            detail += f" (safely decoded steps: {len(completed_steps)})"
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class _FreshAuthority:
    protocol_identity_sha256: bytes
    source_bundle_identity_sha256: bytes

    def __post_init__(self) -> None:
        if (
            type(self.protocol_identity_sha256) is not bytes
            or len(self.protocol_identity_sha256) != 32
            or type(self.source_bundle_identity_sha256) is not bytes
            or len(self.source_bundle_identity_sha256) != 32
        ):
            raise ValueError("native temporal authority digests must contain exactly 32 bytes")

    def native(self) -> _NativeAuthority:
        return _NativeAuthority(
            (ctypes.c_uint8 * 32)(*self.protocol_identity_sha256),
            (ctypes.c_uint8 * 32)(*self.source_bundle_identity_sha256),
        )


def _sha256_bytes(label: str, digest: object) -> bytes:
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError(f"{label} did not return a canonical SHA-256")
    return bytes.fromhex(digest)


def _governing_protocol() -> ModuleType:
    try:
        return importlib.import_module(_GOVERNING_PROTOCOL_MODULE)
    except ImportError as exc:
        raise RuntimeError("governing calibration-v3 protocol is unavailable") from exc


def _fresh_authority() -> _FreshAuthority:
    governing = _governing_protocol()
    protocol = governing.build_protocol()
    governing.validate_protocol(protocol, require_runnable=True)
    binding = protocol["source_binding"]
    governing.validate_committed_source_blobs(governing.REPO_ROOT, binding)
    governing.validate_source_bytes(governing.REPO_ROOT, binding)
    protocol_digest = _sha256_bytes("protocol identity", governing.protocol_sha256())
    try:
        gate3 = importlib.import_module(_GATE3_AUTHORITY_MODULE)
    except ImportError as exc:
        raise RuntimeError("governing calibration-v3 Gate3 authority is unavailable") from exc
    source_authority = gate3.authorize_gate3_source(
        governing.REPO_ROOT,
        binding,
        live_manifest=_procedural_runtime_source_manifest,
        live_bundle_sha256=_procedural_runtime_source_bundle_sha256,
    )
    source_digest = _sha256_bytes(
        "procedural runtime source identity",
        source_authority.source_bundle_sha256,
    )
    return _FreshAuthority(protocol_digest, source_digest)


_BINDINGS_LOCK = threading.Lock()
_BOUND_LIBRARY: object | None = None


def _bind_temporal_symbols(library: object) -> None:
    try:
        definition_identity = library.humanbrain_temporal_circuit_definition_identity_v1
        create = library.humanbrain_temporal_circuit_runtime_create_v1
        import_checkpoint = library.humanbrain_temporal_circuit_runtime_import_checkpoint_v1
        advance = library.humanbrain_temporal_circuit_runtime_advance_block_v1
        checkpoint_size = library.humanbrain_temporal_circuit_runtime_checkpoint_size_v1
        checkpoint_identity = library.humanbrain_temporal_circuit_runtime_checkpoint_identity_v1
        export_checkpoint = library.humanbrain_temporal_circuit_runtime_export_checkpoint_v1
        export_checkpoint_path = (
            library.humanbrain_temporal_circuit_runtime_export_checkpoint_path_v1
        )
        publish_immutable_bytes_path = library.humanbrain_temporal_publish_immutable_bytes_path_v1
        reconcile_existing_immutable_bytes_path = (
            library.humanbrain_temporal_reconcile_existing_immutable_bytes_path_v1
        )
        prepare_durable_directory_path = (
            library.humanbrain_temporal_prepare_durable_directory_path_v1
        )
        destroy = library.humanbrain_temporal_circuit_runtime_destroy
    except AttributeError as exc:
        raise _NativeBuildError("native temporal runtime ABI symbols are unavailable") from exc

    handle_output = ctypes.POINTER(ctypes.c_void_p)
    report_output = ctypes.POINTER(_NativeReport)
    definition_identity.argtypes = [
        ctypes.POINTER(_NativeDefinitionIdentityRequest),
        ctypes.POINTER(_NativeDefinitionIdentity),
        report_output,
    ]
    create.argtypes = [ctypes.POINTER(_NativeDefinition), handle_output, report_output]
    import_checkpoint.argtypes = [
        ctypes.POINTER(_NativeImportRequest),
        handle_output,
        report_output,
    ]
    advance.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NativeAdvanceRequest),
        report_output,
    ]
    checkpoint_size.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NativeAuthority),
        report_output,
    ]
    checkpoint_identity.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NativeAuthority),
        ctypes.POINTER(_NativeCheckpointIdentity),
        report_output,
    ]
    export_checkpoint.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NativeAuthority),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        report_output,
    ]
    export_checkpoint_path.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NativeAuthority),
        ctypes.POINTER(_NativeCheckpointPathRequest),
        ctypes.POINTER(_NativeCheckpointPublicationReport),
        report_output,
    ]
    publish_immutable_bytes_path.argtypes = [
        ctypes.POINTER(_NativeImmutableBytesPathRequest),
        ctypes.POINTER(_NativeCheckpointPublicationReport),
        report_output,
    ]
    reconcile_existing_immutable_bytes_path.argtypes = [
        ctypes.POINTER(_NativeImmutableBytesPathRequest),
        ctypes.POINTER(_NativeImmutableBytesReconciliationReport),
        report_output,
    ]
    prepare_durable_directory_path.argtypes = [
        ctypes.POINTER(_NativeDurableDirectoryPathRequest),
        ctypes.POINTER(_NativeDurableDirectoryPreparationReport),
        report_output,
    ]
    destroy.argtypes = [ctypes.c_void_p]
    for function in (
        definition_identity,
        create,
        import_checkpoint,
        advance,
        checkpoint_size,
        checkpoint_identity,
        export_checkpoint,
        export_checkpoint_path,
        publish_immutable_bytes_path,
        reconcile_existing_immutable_bytes_path,
        prepare_durable_directory_path,
    ):
        function.restype = ctypes.c_uint32
    destroy.restype = None


def _temporal_library() -> ctypes.CDLL:
    global _BOUND_LIBRARY
    library = _load_library()
    with _BINDINGS_LOCK:
        if _BOUND_LIBRARY is not library:
            _bind_temporal_symbols(library)
            _BOUND_LIBRARY = library
    return library


def _attested_current_library_identity() -> _NativeLibraryIdentity:
    identity = _loaded_library_identity()
    if type(identity) is not _NativeLibraryIdentity:
        raise _invariant_error("loaded library identity has an unexpected type")
    current_source_digest = _current_native_source_digest()
    if identity.source_digest != current_source_digest:
        raise _NativeBuildError(
            "loaded native library does not match the current native build inputs"
        )
    return identity


@dataclass(frozen=True, slots=True)
class _DecodedReport:
    status: NativeTemporalRuntimeStatus
    message: str
    completed_prefix: int
    required: int
    observed: int
    location: int
    poison_code: int
    stage_code: int


def _invariant_error(
    message: str,
    *,
    completed_prefix: int | None = None,
    completed_steps: tuple[TemporalCircuitStep, ...] = (),
    required: int = 0,
    observed: int = 0,
    location: int = 0,
) -> NativeTemporalRuntimeError:
    return NativeTemporalRuntimeError(
        NativeTemporalRuntimeStatus.INTERNAL_INVARIANT,
        f"native ABI invariant failed: {message}",
        completed_prefix=(len(completed_steps) if completed_prefix is None else completed_prefix),
        required=required,
        observed=observed,
        location=location,
        completed_steps=completed_steps,
    )


def _decode_report(
    returned_status: object,
    report: _NativeReport,
    *,
    maximum_completed_prefix: int,
) -> _DecodedReport:
    raw_completed_prefix = int(report.completed_prefix)
    preserved_completed_prefix = (
        raw_completed_prefix if raw_completed_prefix <= maximum_completed_prefix else 0
    )

    def invariant(
        message: str,
        *,
        required: int = 0,
        observed: int = 0,
        location: int = 0,
    ) -> NativeTemporalRuntimeError:
        return _invariant_error(
            message,
            completed_prefix=preserved_completed_prefix,
            required=required,
            observed=observed,
            location=location,
        )

    try:
        returned = int(returned_status)
    except (TypeError, ValueError, OverflowError) as exc:
        raise invariant("native return status is not an integer") from exc
    if returned not in NativeTemporalRuntimeStatus._value2member_map_:
        raise invariant("native return status is outside the frozen range", observed=returned)
    if int(report.status) not in NativeTemporalRuntimeStatus._value2member_map_:
        raise invariant("report status is outside the frozen range", observed=int(report.status))
    if returned != int(report.status):
        raise invariant(
            "return and report statuses disagree", required=returned, observed=int(report.status)
        )
    if report.reserved_zero != 0:
        raise invariant("report reserved field is nonzero")
    if report.message_len > 464:
        raise invariant("report message length exceeds its fixed buffer")
    message_bytes = bytes(report.message_utf8)
    if any(message_bytes[report.message_len :]):
        raise invariant("report message tail is nonzero")
    try:
        message = message_bytes[: report.message_len].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise invariant("report message is not valid UTF-8") from exc
    if report.poison_code > 15:
        raise invariant("report poison code is outside the frozen range")
    if report.stage_code > 8:
        raise invariant("report stage code is outside the frozen range")
    if report.completed_prefix > maximum_completed_prefix:
        raise invariant(
            "report completed prefix exceeds the supplied block",
            required=maximum_completed_prefix,
            observed=int(report.completed_prefix),
        )
    status = NativeTemporalRuntimeStatus(returned)
    if status is NativeTemporalRuntimeStatus.OK:
        if message or report.poison_code != 0 or report.stage_code != 0:
            raise invariant("successful report carries failure diagnostics")
    elif not message:
        raise invariant("failed report has no diagnostic message")
    return _DecodedReport(
        status=status,
        message=message,
        completed_prefix=int(report.completed_prefix),
        required=int(report.required),
        observed=int(report.observed),
        location=int(report.location),
        poison_code=int(report.poison_code),
        stage_code=int(report.stage_code),
    )


def _raise_report(
    decoded: _DecodedReport,
    completed_steps: tuple[TemporalCircuitStep, ...] = (),
) -> None:
    raise NativeTemporalRuntimeError(
        decoded.status,
        decoded.message,
        completed_prefix=decoded.completed_prefix,
        required=decoded.required,
        observed=decoded.observed,
        location=decoded.location,
        poison_code=decoded.poison_code,
        stage_code=decoded.stage_code,
        completed_steps=completed_steps,
    )


def _decode_checkpoint_identity(
    returned_status: object,
    native: _NativeCheckpointIdentity,
    report: _NativeReport,
    authority: _FreshAuthority,
) -> TemporalCheckpointIdentity:
    decoded = _decode_report(returned_status, report, maximum_completed_prefix=0)
    header = bytes(native.header)
    if decoded.status is not NativeTemporalRuntimeStatus.OK:
        if any(header):
            raise _invariant_error("failed checkpoint identity call returned nonzero output")
        if (
            decoded.required
            or decoded.observed
            or decoded.location
            or decoded.poison_code
            or decoded.stage_code
        ):
            raise _invariant_error("failed checkpoint identity report carries scalar diagnostics")
        _raise_report(decoded)
    if (
        decoded.required != _CHECKPOINT_V1_HEADER_BYTES
        or decoded.observed != _CHECKPOINT_V1_HEADER_BYTES
        or decoded.location != 0
    ):
        raise _invariant_error(
            "checkpoint identity report does not describe one exact header",
            required=_CHECKPOINT_V1_HEADER_BYTES,
            observed=decoded.observed,
        )
    if header[:8] != _CHECKPOINT_V1_MAGIC:
        raise _invariant_error("checkpoint identity magic does not match")
    format_version = int.from_bytes(header[8:10], "little")
    if format_version != _CHECKPOINT_V1_FORMAT_VERSION:
        raise _invariant_error(
            "checkpoint identity format version is unsupported",
            observed=format_version,
        )
    header_bytes = int.from_bytes(header[10:12], "little")
    if header_bytes != _CHECKPOINT_V1_HEADER_BYTES:
        raise _invariant_error(
            "checkpoint identity header length does not match",
            observed=header_bytes,
        )
    if any(header[12:16]):
        raise _invariant_error("checkpoint identity reserved bytes are nonzero")

    checkpoint_bytes = int.from_bytes(header[16:24], "little")
    body_bytes = int.from_bytes(header[24:32], "little")
    if (
        checkpoint_bytes < _CHECKPOINT_V1_HEADER_BYTES
        or body_bytes != checkpoint_bytes - _CHECKPOINT_V1_HEADER_BYTES
    ):
        raise _invariant_error(
            "checkpoint identity framing lengths are inconsistent",
            required=checkpoint_bytes - _CHECKPOINT_V1_HEADER_BYTES,
            observed=body_bytes,
        )
    if header[32:64] != authority.protocol_identity_sha256:
        raise _invariant_error("checkpoint identity protocol authority does not match")
    if header[64:96] != authority.source_bundle_identity_sha256:
        raise _invariant_error("checkpoint identity source authority does not match")

    (
        segment_count,
        cohort_count,
        next_reserve_cell,
        event_count,
        learning_step,
        prior_context_length,
        winner_context_length,
        poison_code,
    ) = struct.unpack("<IIIQQBBH", header[192:224])
    if prior_context_length > _CONTEXT_CELLS:
        raise _invariant_error(
            "checkpoint identity prior-context length exceeds the frozen capacity"
        )
    if winner_context_length > _CONTEXT_CELLS:
        raise _invariant_error(
            "checkpoint identity winner-context length exceeds the frozen capacity"
        )
    if poison_code > 15:
        raise _invariant_error("checkpoint identity poison code is outside the frozen range")
    expected_body_bytes = (
        cohort_count * 4
        + prior_context_length * 4
        + winner_context_length * 4
        + segment_count * (4 + _PAIRS_PER_SEGMENT * 4 + _PAIRS_PER_SEGMENT * 2)
    )
    if body_bytes != expected_body_bytes:
        raise _invariant_error(
            "checkpoint identity scalar counts disagree with the body length",
            required=expected_body_bytes,
            observed=body_bytes,
        )
    logical_state = hashlib.sha256(
        _CHECKPOINT_LOGICAL_STATE_DOMAIN + header[128:160] + header[192:224]
    ).digest()
    if header[224:256] != logical_state:
        raise _invariant_error("checkpoint identity logical-state digest does not match")
    return TemporalCheckpointIdentity(
        format_version=format_version,
        header_bytes=header_bytes,
        checkpoint_bytes=checkpoint_bytes,
        body_bytes=body_bytes,
        protocol_identity_sha256=header[32:64].hex(),
        source_bundle_identity_sha256=header[64:96].hex(),
        circuit_identity_sha256=header[96:128].hex(),
        body_sha256=header[128:160].hex(),
        semantic_index_sha256=header[160:192].hex(),
        segment_count=segment_count,
        cohort_count=cohort_count,
        next_reserve_cell=next_reserve_cell,
        event_count=event_count,
        learning_step=learning_step,
        prior_context_length=prior_context_length,
        winner_context_length=winner_context_length,
        poison_code=poison_code,
        logical_state_sha256=header[224:256].hex(),
    )


def _decode_publication_error(
    native: _NativeCheckpointPublicationError,
    *,
    label: str,
) -> TemporalCheckpointPublicationError | None:
    raw_stage = int(native.stage)
    raw_domain = int(native.domain)
    raw_code = int(native.code)
    if raw_stage not in TemporalCheckpointErrorStage._value2member_map_:
        raise _invariant_error(f"{label} stage is outside the frozen range")
    if raw_domain not in TemporalCheckpointErrorDomain._value2member_map_:
        raise _invariant_error(f"{label} domain is outside the frozen range")
    stage = TemporalCheckpointErrorStage(raw_stage)
    domain = TemporalCheckpointErrorDomain(raw_domain)
    if stage is TemporalCheckpointErrorStage.NONE:
        if domain is not TemporalCheckpointErrorDomain.NONE or raw_code != 0:
            raise _invariant_error(f"{label} has a malformed zero error")
        return None
    if domain is TemporalCheckpointErrorDomain.NONE:
        raise _invariant_error(f"{label} has an active stage without an error domain")
    return TemporalCheckpointPublicationError(stage, domain, raw_code)


def _decode_publication(
    returned_status: object,
    native: _NativeCheckpointPublicationReport,
    runtime_report: _NativeReport,
) -> TemporalCheckpointPublication:
    decoded_runtime = _decode_report(returned_status, runtime_report, maximum_completed_prefix=0)
    if decoded_runtime.status is not NativeTemporalRuntimeStatus.OK:
        if bytes(native) != bytes(ctypes.sizeof(type(native))):
            raise _invariant_error(
                "failed checkpoint-path export changed its zero publication report"
            )
        _raise_report(decoded_runtime)

    if any(native.reserved_zero):
        raise _invariant_error("checkpoint publication reserved bytes are nonzero")
    raw_outcome = int(native.outcome)
    if raw_outcome not in TemporalCheckpointPublicationOutcome._value2member_map_:
        raise _invariant_error("checkpoint publication outcome is outside the frozen range")
    raw_cleanup = int(native.temp_cleanup)
    if raw_cleanup not in TemporalCheckpointCleanup._value2member_map_:
        raise _invariant_error("checkpoint publication cleanup tag is outside the frozen range")
    cleanup_error_count = int(native.cleanup_error_count)
    if cleanup_error_count > len(native.cleanup_errors):
        raise _invariant_error("checkpoint publication cleanup error count exceeds capacity")

    primary_error = _decode_publication_error(
        native.primary_error, label="checkpoint publication primary error"
    )
    decoded_cleanup_slots = tuple(
        _decode_publication_error(slot, label=f"checkpoint publication cleanup error {index}")
        for index, slot in enumerate(native.cleanup_errors)
    )
    if any(error is None for error in decoded_cleanup_slots[:cleanup_error_count]):
        raise _invariant_error("checkpoint publication active cleanup error is zero")
    if any(error is not None for error in decoded_cleanup_slots[cleanup_error_count:]):
        raise _invariant_error("checkpoint publication unused cleanup error slot is nonzero")
    cleanup_errors = tuple(
        error for error in decoded_cleanup_slots[:cleanup_error_count] if error is not None
    )

    outcome = TemporalCheckpointPublicationOutcome(raw_outcome)
    cleanup = TemporalCheckpointCleanup(raw_cleanup)
    owned_temp_pid = int(native.owned_temp_pid)
    owned_temp_nonce = int(native.owned_temp_nonce)
    checkpoint_bytes = int(native.checkpoint_bytes)
    encoded_bytes = int(native.encoded_bytes)

    if decoded_runtime.location != 0:
        raise _invariant_error("checkpoint publication runtime report has a location")
    if decoded_runtime.required != checkpoint_bytes or decoded_runtime.observed != encoded_bytes:
        raise _invariant_error(
            "checkpoint publication and runtime report lengths disagree",
            required=checkpoint_bytes,
            observed=decoded_runtime.observed,
        )
    if checkpoint_bytes < _CHECKPOINT_V1_HEADER_BYTES or checkpoint_bytes > sys.maxsize:
        raise _invariant_error(
            "checkpoint publication length is outside the canonical host range",
            observed=checkpoint_bytes,
        )
    if encoded_bytes > checkpoint_bytes:
        raise _invariant_error(
            "checkpoint publication encoded length exceeds its checkpoint length",
            required=checkpoint_bytes,
            observed=encoded_bytes,
        )

    if cleanup is TemporalCheckpointCleanup.NOT_APPLICABLE:
        if (
            outcome is not TemporalCheckpointPublicationOutcome.NOT_PUBLISHED
            or cleanup_errors
            or owned_temp_pid != 0
            or owned_temp_nonce != 0
            or encoded_bytes != 0
        ):
            raise _invariant_error("checkpoint publication N/A cleanup has a temporary artifact")
    elif cleanup is TemporalCheckpointCleanup.CLEAN:
        if cleanup_errors or owned_temp_pid == 0:
            raise _invariant_error("checkpoint publication clean disposition is inconsistent")
    elif not cleanup_errors or owned_temp_pid == 0:
        raise _invariant_error("checkpoint publication orphaned disposition is inconsistent")

    if outcome in (
        TemporalCheckpointPublicationOutcome.DURABLE_PUBLISHED,
        TemporalCheckpointPublicationOutcome.COLLISION,
    ):
        if primary_error is not None or encoded_bytes != checkpoint_bytes:
            raise _invariant_error("successful checkpoint publication outcome is inconsistent")
        if cleanup is TemporalCheckpointCleanup.NOT_APPLICABLE:
            raise _invariant_error(
                "successful checkpoint publication did not create a temporary artifact"
            )
    elif outcome is TemporalCheckpointPublicationOutcome.NOT_PUBLISHED:
        if primary_error is None:
            raise _invariant_error("not-published checkpoint outcome has no primary error")
        allowed_stages = (
            _PRE_TEMP_PUBLICATION_ERROR_STAGES
            if cleanup is TemporalCheckpointCleanup.NOT_APPLICABLE
            else _TEMP_BACKED_NOT_PUBLISHED_ERROR_STAGES
        )
        if primary_error.stage not in allowed_stages:
            raise _invariant_error(
                "not-published checkpoint primary stage is inconsistent with cleanup"
            )
    else:
        if (
            primary_error is None
            or encoded_bytes != checkpoint_bytes
            or cleanup is TemporalCheckpointCleanup.NOT_APPLICABLE
        ):
            raise _invariant_error("unconfirmed-durability checkpoint outcome is inconsistent")
        if primary_error.stage not in _UNCONFIRMED_PUBLICATION_ERROR_STAGES:
            raise _invariant_error(
                "unconfirmed-durability checkpoint primary stage is inconsistent"
            )

    return TemporalCheckpointPublication(
        outcome=outcome,
        temp_cleanup=cleanup,
        primary_error=primary_error,
        cleanup_errors=cleanup_errors,
        owned_temp_pid=owned_temp_pid,
        owned_temp_nonce=owned_temp_nonce,
        checkpoint_bytes=checkpoint_bytes,
        encoded_bytes=encoded_bytes,
    )


def _decode_immutable_publication_error(
    native: _NativeCheckpointPublicationError,
    *,
    label: str,
) -> TemporalPublicationError | None:
    raw_stage = int(native.stage)
    raw_domain = int(native.domain)
    raw_code = int(native.code)
    if raw_stage not in TemporalPublicationErrorStage._value2member_map_:
        raise _invariant_error(f"{label} stage is outside the frozen range")
    if raw_domain not in TemporalPublicationErrorDomain._value2member_map_:
        raise _invariant_error(f"{label} domain is outside the frozen range")
    stage = TemporalPublicationErrorStage(raw_stage)
    domain = TemporalPublicationErrorDomain(raw_domain)
    if domain is TemporalPublicationErrorDomain.CHECKPOINT_ENCODER_RESERVED:
        raise _invariant_error(f"{label} uses the checkpoint-only encoder error domain")
    if stage is TemporalPublicationErrorStage.NONE:
        if domain is not TemporalPublicationErrorDomain.NONE or raw_code != 0:
            raise _invariant_error(f"{label} has a malformed zero error")
        return None
    if domain is TemporalPublicationErrorDomain.NONE:
        raise _invariant_error(f"{label} has an active stage without an error domain")
    return TemporalPublicationError(stage, domain, raw_code)


def _decode_immutable_bytes_publication(
    returned_status: object,
    native: _NativeCheckpointPublicationReport,
    runtime_report: _NativeReport,
    *,
    expected_payload_bytes: int,
) -> TemporalImmutableBytesPublication:
    decoded_runtime = _decode_report(returned_status, runtime_report, maximum_completed_prefix=0)
    if decoded_runtime.status is not NativeTemporalRuntimeStatus.OK:
        if bytes(native) != bytes(ctypes.sizeof(type(native))):
            raise _invariant_error(
                "failed immutable-byte publication changed its zero publication report"
            )
        _raise_report(decoded_runtime)

    if any(native.reserved_zero):
        raise _invariant_error("immutable-byte publication reserved bytes are nonzero")
    raw_outcome = int(native.outcome)
    if raw_outcome not in TemporalPublicationOutcome._value2member_map_:
        raise _invariant_error("immutable-byte publication outcome is outside the frozen range")
    raw_cleanup = int(native.temp_cleanup)
    if raw_cleanup not in TemporalPublicationCleanup._value2member_map_:
        raise _invariant_error("immutable-byte publication cleanup tag is outside the frozen range")
    cleanup_error_count = int(native.cleanup_error_count)
    if cleanup_error_count > len(native.cleanup_errors):
        raise _invariant_error("immutable-byte publication cleanup error count exceeds capacity")

    primary_error = _decode_immutable_publication_error(
        native.primary_error, label="immutable-byte publication primary error"
    )
    decoded_cleanup_slots = tuple(
        _decode_immutable_publication_error(
            slot, label=f"immutable-byte publication cleanup error {index}"
        )
        for index, slot in enumerate(native.cleanup_errors)
    )
    if any(error is None for error in decoded_cleanup_slots[:cleanup_error_count]):
        raise _invariant_error("immutable-byte publication active cleanup error is zero")
    if any(error is not None for error in decoded_cleanup_slots[cleanup_error_count:]):
        raise _invariant_error("immutable-byte publication unused cleanup error slot is nonzero")
    cleanup_errors = tuple(
        error for error in decoded_cleanup_slots[:cleanup_error_count] if error is not None
    )

    outcome = TemporalPublicationOutcome(raw_outcome)
    cleanup = TemporalPublicationCleanup(raw_cleanup)
    owned_temp_pid = int(native.owned_temp_pid)
    owned_temp_nonce = int(native.owned_temp_nonce)
    payload_bytes = int(native.checkpoint_bytes)
    written_bytes = int(native.encoded_bytes)

    if not 1 <= expected_payload_bytes <= _MAX_IMMUTABLE_PAYLOAD_BYTES:
        raise _invariant_error(
            "immutable-byte caller length is outside the bounded range",
            observed=expected_payload_bytes,
        )
    if payload_bytes != expected_payload_bytes:
        raise _invariant_error(
            "immutable-byte publication length differs from the caller payload",
            required=expected_payload_bytes,
            observed=payload_bytes,
        )
    if decoded_runtime.location != 0:
        raise _invariant_error("immutable-byte publication runtime report has a location")
    if decoded_runtime.required != payload_bytes or decoded_runtime.observed != written_bytes:
        raise _invariant_error(
            "immutable-byte publication and runtime report lengths disagree",
            required=payload_bytes,
            observed=decoded_runtime.observed,
        )
    if not 1 <= payload_bytes <= _MAX_IMMUTABLE_PAYLOAD_BYTES:
        raise _invariant_error(
            "immutable-byte publication length is outside the bounded range",
            observed=payload_bytes,
        )
    if written_bytes > payload_bytes:
        raise _invariant_error(
            "immutable-byte publication written length exceeds payload length",
            required=payload_bytes,
            observed=written_bytes,
        )

    if cleanup is TemporalPublicationCleanup.NOT_APPLICABLE:
        if (
            outcome is not TemporalPublicationOutcome.NOT_PUBLISHED
            or cleanup_errors
            or owned_temp_pid != 0
            or owned_temp_nonce != 0
            or written_bytes != 0
        ):
            raise _invariant_error(
                "immutable-byte publication N/A cleanup has a temporary artifact"
            )
    elif cleanup is TemporalPublicationCleanup.CLEAN:
        if cleanup_errors or owned_temp_pid == 0:
            raise _invariant_error("immutable-byte publication clean disposition is inconsistent")
    elif not cleanup_errors or owned_temp_pid == 0:
        raise _invariant_error("immutable-byte publication orphaned disposition is inconsistent")

    if outcome in (
        TemporalPublicationOutcome.DURABLE_PUBLISHED,
        TemporalPublicationOutcome.COLLISION,
    ):
        if primary_error is not None or written_bytes != payload_bytes:
            raise _invariant_error("successful immutable-byte publication outcome is inconsistent")
        if cleanup is TemporalPublicationCleanup.NOT_APPLICABLE:
            raise _invariant_error(
                "successful immutable-byte publication did not create a temporary artifact"
            )
    elif outcome is TemporalPublicationOutcome.NOT_PUBLISHED:
        if primary_error is None:
            raise _invariant_error("not-published immutable-byte outcome has no primary error")
        allowed_stages = (
            frozenset(
                {
                    TemporalPublicationErrorStage.PARENT_OPEN,
                    TemporalPublicationErrorStage.PARENT_IDENTITY,
                    TemporalPublicationErrorStage.FILESYSTEM_POLICY,
                    TemporalPublicationErrorStage.TEMP_CREATE,
                    TemporalPublicationErrorStage.PARENT_SYNC,
                }
            )
            if cleanup is TemporalPublicationCleanup.NOT_APPLICABLE
            else frozenset(
                {
                    TemporalPublicationErrorStage.PARENT_IDENTITY,
                    TemporalPublicationErrorStage.WRITE,
                    TemporalPublicationErrorStage.FILE_SYNC,
                    TemporalPublicationErrorStage.FILE_CLOSE,
                    TemporalPublicationErrorStage.TEMP_IDENTITY,
                    TemporalPublicationErrorStage.EXCLUSIVE_PUBLISH,
                }
            )
        )
        if primary_error.stage not in allowed_stages:
            raise _invariant_error(
                "not-published immutable-byte primary stage is inconsistent with cleanup"
            )
    elif (
        primary_error is None
        or written_bytes != payload_bytes
        or cleanup is TemporalPublicationCleanup.NOT_APPLICABLE
    ):
        raise _invariant_error("unconfirmed-durability immutable-byte outcome is inconsistent")
    elif primary_error.stage not in frozenset(
        {
            TemporalPublicationErrorStage.EXCLUSIVE_PUBLISH,
            TemporalPublicationErrorStage.PARENT_SYNC,
        }
    ):
        raise _invariant_error(
            "unconfirmed-durability immutable-byte primary stage is inconsistent"
        )

    return TemporalImmutableBytesPublication(
        outcome=outcome,
        temp_cleanup=cleanup,
        primary_error=primary_error,
        cleanup_errors=cleanup_errors,
        owned_temp_pid=owned_temp_pid,
        owned_temp_nonce=owned_temp_nonce,
        payload_bytes=payload_bytes,
        written_bytes=written_bytes,
    )


def _decode_durability_error(
    native: _NativeCheckpointPublicationError,
    *,
    label: str,
) -> TemporalDurabilityError | None:
    raw_stage = int(native.stage)
    raw_domain = int(native.domain)
    raw_code = int(native.code)
    if raw_stage not in TemporalDurabilityErrorStage._value2member_map_:
        raise _invariant_error(f"{label} stage is outside the frozen range")
    if raw_domain not in TemporalDurabilityErrorDomain._value2member_map_:
        raise _invariant_error(f"{label} domain is outside the frozen range")
    stage = TemporalDurabilityErrorStage(raw_stage)
    domain = TemporalDurabilityErrorDomain(raw_domain)
    if domain is TemporalDurabilityErrorDomain.CHECKPOINT_ENCODER_RESERVED:
        raise _invariant_error(f"{label} uses the checkpoint-only encoder domain")
    if stage is TemporalDurabilityErrorStage.NONE:
        if domain is not TemporalDurabilityErrorDomain.NONE or raw_code != 0:
            raise _invariant_error(f"{label} has a malformed zero error")
        return None
    if domain is TemporalDurabilityErrorDomain.NONE:
        raise _invariant_error(f"{label} has an active stage without an error domain")
    return TemporalDurabilityError(stage, domain, raw_code)


def _decode_durability_errors(
    primary_native: _NativeCheckpointPublicationError,
    cleanup_native: Sequence[_NativeCheckpointPublicationError],
    cleanup_error_count: int,
    *,
    label: str,
) -> tuple[TemporalDurabilityError | None, tuple[TemporalDurabilityError, ...]]:
    if cleanup_error_count > len(cleanup_native):
        raise _invariant_error(f"{label} cleanup error count exceeds capacity")
    primary = _decode_durability_error(primary_native, label=f"{label} primary error")
    slots = tuple(
        _decode_durability_error(error, label=f"{label} cleanup error {index}")
        for index, error in enumerate(cleanup_native)
    )
    if any(error is None for error in slots[:cleanup_error_count]):
        raise _invariant_error(f"{label} active cleanup error is zero")
    if any(error is not None for error in slots[cleanup_error_count:]):
        raise _invariant_error(f"{label} unused cleanup error slot is nonzero")
    return primary, tuple(error for error in slots[:cleanup_error_count] if error is not None)


def _decode_immutable_bytes_reconciliation(
    returned_status: object,
    native: _NativeImmutableBytesReconciliationReport,
    runtime_report: _NativeReport,
    *,
    expected_bytes: int,
) -> TemporalImmutableBytesReconciliation:
    decoded_runtime = _decode_report(returned_status, runtime_report, maximum_completed_prefix=0)
    if decoded_runtime.status is not NativeTemporalRuntimeStatus.OK:
        if bytes(native) != bytes(ctypes.sizeof(type(native))):
            raise _invariant_error("failed immutable-byte reconciliation changed its zero receipt")
        _raise_report(decoded_runtime)
    if native.reserved_zero_u8 or native.reserved_zero_u32 or any(native.reserved_zero):
        raise _invariant_error("immutable-byte reconciliation reserved fields are nonzero")
    if int(native.outcome) not in TemporalDurabilityOutcome._value2member_map_:
        raise _invariant_error("immutable-byte reconciliation outcome is outside the frozen range")
    primary, cleanup = _decode_durability_errors(
        native.primary_error,
        native.cleanup_errors,
        int(native.cleanup_error_count),
        label="immutable-byte reconciliation",
    )
    required = int(native.required_bytes)
    verified = int(native.verified_bytes)
    if required != expected_bytes or verified > required:
        raise _invariant_error(
            "immutable-byte reconciliation lengths are inconsistent",
            required=expected_bytes,
            observed=verified,
        )
    if (
        decoded_runtime.required != required
        or decoded_runtime.observed != verified
        or decoded_runtime.location != 0
    ):
        raise _invariant_error("immutable-byte reconciliation runtime lengths disagree")
    outcome = TemporalDurabilityOutcome(int(native.outcome))
    if outcome is TemporalDurabilityOutcome.DURABLE:
        if primary is not None or cleanup or verified != required:
            raise _invariant_error("durable immutable-byte reconciliation is inconsistent")
    elif primary is None:
        raise _invariant_error("failed immutable-byte reconciliation has no primary error")
    return TemporalImmutableBytesReconciliation(
        outcome=outcome,
        primary_error=primary,
        cleanup_errors=cleanup,
        expected_bytes=required,
        verified_bytes=verified,
    )


def _decode_durable_directory_preparation(
    returned_status: object,
    native: _NativeDurableDirectoryPreparationReport,
    runtime_report: _NativeReport,
    *,
    expected_component_count: int,
) -> TemporalDurableDirectoryPreparation:
    decoded_runtime = _decode_report(returned_status, runtime_report, maximum_completed_prefix=0)
    if decoded_runtime.status is not NativeTemporalRuntimeStatus.OK:
        if bytes(native) != bytes(ctypes.sizeof(type(native))):
            raise _invariant_error("failed durable-directory preparation changed its zero receipt")
        _raise_report(decoded_runtime)
    if native.reserved_zero_u8 or native.reserved_zero_u32 or any(native.reserved_zero):
        raise _invariant_error("durable-directory preparation reserved fields are nonzero")
    if int(native.outcome) not in TemporalDurabilityOutcome._value2member_map_:
        raise _invariant_error("durable-directory outcome is outside the frozen range")
    primary, cleanup = _decode_durability_errors(
        native.primary_error,
        native.cleanup_errors,
        int(native.cleanup_error_count),
        label="durable-directory preparation",
    )
    required = int(native.required_component_count)
    confirmed = int(native.confirmed_component_count)
    created = int(native.created_component_count)
    failed_raw = int(native.failed_component_index)
    if (
        required != expected_component_count
        or confirmed > required
        or created > required
        or decoded_runtime.required != required
        or decoded_runtime.observed != confirmed
        or decoded_runtime.location != 0
    ):
        raise _invariant_error("durable-directory component counts are inconsistent")
    outcome = TemporalDurabilityOutcome(int(native.outcome))
    if outcome is TemporalDurabilityOutcome.DURABLE:
        if primary is not None or cleanup or confirmed != required or failed_raw != _UINT32_MAX:
            raise _invariant_error("durable-directory success receipt is inconsistent")
        failed_component_index = None
    else:
        if primary is None or (failed_raw != _UINT32_MAX and failed_raw >= required):
            raise _invariant_error("durable-directory failure receipt is inconsistent")
        failed_component_index = None if failed_raw == _UINT32_MAX else failed_raw
    return TemporalDurableDirectoryPreparation(
        outcome=outcome,
        primary_error=primary,
        cleanup_errors=cleanup,
        required_component_count=required,
        confirmed_component_count=confirmed,
        created_component_count=created,
        failed_component_index=failed_component_index,
    )


def _checkpoint_rejected(message: str, *, observed: int = 0) -> NativeTemporalRuntimeError:
    return NativeTemporalRuntimeError(
        NativeTemporalRuntimeStatus.CHECKPOINT_REJECTED,
        message,
        observed=observed,
    )


def _lexical_checkpoint_path_parts(destination: Path) -> tuple[bytes, bytes]:
    if not isinstance(destination, Path):
        raise TypeError("checkpoint path must be a pathlib.Path")
    try:
        return (
            str(destination.parent).encode("utf-8", errors="strict"),
            destination.name.encode("utf-8", errors="strict"),
        )
    except UnicodeEncodeError as exc:
        raise ValueError("checkpoint path must be valid UTF-8") from exc


def _lexical_publication_path_parts(destination: Path) -> tuple[bytes, bytes]:
    if not isinstance(destination, Path):
        raise TypeError("publication path must be a pathlib.Path")
    try:
        return (
            str(destination.parent).encode("utf-8", errors="strict"),
            destination.name.encode("utf-8", errors="strict"),
        )
    except UnicodeEncodeError as exc:
        raise ValueError("publication path must be valid UTF-8") from exc


def publish_immutable_bytes(destination: Path, payload: bytes) -> TemporalImmutableBytesPublication:
    """Attempt one authority-gated, exclusive publication of opaque bytes.

    The publisher does not interpret JSON or establish cross-file semantics.
    Callers using a manifest as a commit record must durably publish every
    referenced data object first and publish that manifest last.
    """

    if type(payload) is not bytes:
        raise TypeError("immutable publication payload must be exact bytes")
    if not 1 <= len(payload) <= _MAX_IMMUTABLE_PAYLOAD_BYTES:
        raise ValueError(
            f"immutable publication payload length must be in [1, {_MAX_IMMUTABLE_PAYLOAD_BYTES}]"
        )
    if not isinstance(destination, Path):
        raise TypeError("publication path must be a pathlib.Path")
    authority_before_load = _fresh_authority()
    library = _temporal_library()
    authority_after_load = _fresh_authority()
    if authority_after_load != authority_before_load:
        raise NativeTemporalRuntimeError(
            NativeTemporalRuntimeStatus.INVALID_DEFINITION,
            "fresh authority changed while preparing immutable-byte publication",
        )
    _attested_current_library_identity()
    parent_utf8, final_basename_utf8 = _lexical_publication_path_parts(destination)
    try:
        parent_array = (ctypes.c_uint8 * len(parent_utf8)).from_buffer_copy(parent_utf8)
        basename_array = (ctypes.c_uint8 * len(final_basename_utf8)).from_buffer_copy(
            final_basename_utf8
        )
    except (MemoryError, OverflowError) as exc:
        raise NativeTemporalRuntimeError(
            NativeTemporalRuntimeStatus.ALLOCATION_FAILED,
            "immutable-byte publication transport allocation failed",
            required=len(parent_utf8) + len(final_basename_utf8),
        ) from exc
    get_buffer, release_buffer = _readonly_buffer_api("immutable-byte publication")
    buffer_view = _PyBuffer()
    buffer_acquired = False
    try:
        try:
            acquisition_status = int(get_buffer(payload, ctypes.byref(buffer_view), 0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise _invariant_error(
                "PyObject_GetBuffer returned an invalid immutable-byte status"
            ) from exc
        if acquisition_status != 0:
            raise _invariant_error(
                "PyObject_GetBuffer failed for the immutable-byte payload",
                observed=acquisition_status,
            )
        buffer_acquired = True
        _validate_readonly_simple_buffer(
            buffer_view,
            len(payload),
            subject="immutable-byte payload",
            expected_length="exact bytes length",
        )
        payload_pointer = ctypes.cast(
            ctypes.c_void_p(buffer_view.buf),
            ctypes.POINTER(ctypes.c_uint8),
        )
        request = _NativeImmutableBytesPathRequest(
            ctypes.cast(parent_array, ctypes.POINTER(ctypes.c_uint8)),
            len(parent_utf8),
            ctypes.cast(basename_array, ctypes.POINTER(ctypes.c_uint8)),
            len(final_basename_utf8),
            payload_pointer,
            len(payload),
            (ctypes.c_uint8 * 8)(),
        )
        publication_report = _NativeCheckpointPublicationReport()
        runtime_report = _NativeReport()
        returned = library.humanbrain_temporal_publish_immutable_bytes_path_v1(
            ctypes.byref(request),
            ctypes.byref(publication_report),
            ctypes.byref(runtime_report),
        )
    finally:
        if buffer_acquired:
            release_buffer(ctypes.byref(buffer_view))
    return _decode_immutable_bytes_publication(
        returned,
        publication_report,
        runtime_report,
        expected_payload_bytes=len(payload),
    )


def reconcile_existing_immutable_bytes(
    destination: Path,
    expected_payload: bytes,
) -> TemporalImmutableBytesReconciliation:
    """Verify and flush one existing exact immutable artifact without mutation."""

    if type(expected_payload) is not bytes:
        raise TypeError("reconciliation payload must be exact bytes")
    if not 1 <= len(expected_payload) <= _MAX_IMMUTABLE_PAYLOAD_BYTES:
        raise ValueError(
            f"reconciliation payload length must be in [1, {_MAX_IMMUTABLE_PAYLOAD_BYTES}]"
        )
    if not isinstance(destination, Path):
        raise TypeError("reconciliation path must be a pathlib.Path")
    authority_before_load = _fresh_authority()
    library = _temporal_library()
    authority_after_load = _fresh_authority()
    if authority_after_load != authority_before_load:
        raise NativeTemporalRuntimeError(
            NativeTemporalRuntimeStatus.INVALID_DEFINITION,
            "fresh authority changed while preparing immutable-byte reconciliation",
        )
    _attested_current_library_identity()
    parent_utf8, final_basename_utf8 = _lexical_publication_path_parts(destination)
    try:
        parent_array = (ctypes.c_uint8 * len(parent_utf8)).from_buffer_copy(parent_utf8)
        basename_array = (ctypes.c_uint8 * len(final_basename_utf8)).from_buffer_copy(
            final_basename_utf8
        )
        payload_array = (ctypes.c_uint8 * len(expected_payload)).from_buffer_copy(expected_payload)
    except (MemoryError, OverflowError) as exc:
        raise NativeTemporalRuntimeError(
            NativeTemporalRuntimeStatus.ALLOCATION_FAILED,
            "immutable-byte reconciliation transport allocation failed",
            required=(len(parent_utf8) + len(final_basename_utf8) + len(expected_payload)),
        ) from exc
    request = _NativeImmutableBytesPathRequest(
        ctypes.cast(parent_array, ctypes.POINTER(ctypes.c_uint8)),
        len(parent_utf8),
        ctypes.cast(basename_array, ctypes.POINTER(ctypes.c_uint8)),
        len(final_basename_utf8),
        ctypes.cast(payload_array, ctypes.POINTER(ctypes.c_uint8)),
        len(expected_payload),
        (ctypes.c_uint8 * 8)(),
    )
    durability_report = _NativeImmutableBytesReconciliationReport()
    runtime_report = _NativeReport()
    returned = library.humanbrain_temporal_reconcile_existing_immutable_bytes_path_v1(
        ctypes.byref(request),
        ctypes.byref(durability_report),
        ctypes.byref(runtime_report),
    )
    return _decode_immutable_bytes_reconciliation(
        returned,
        durability_report,
        runtime_report,
        expected_bytes=len(expected_payload),
    )


def _lexical_durable_directory_parts(
    root: Path,
    relative_directory: Path,
) -> tuple[bytes, bytes, int]:
    if not isinstance(root, Path) or not isinstance(relative_directory, Path):
        raise TypeError("durable directory paths must be pathlib.Path values")
    if not root.is_absolute():
        raise ValueError("durable directory root must be absolute")
    if relative_directory.is_absolute():
        raise ValueError("durable directory child must be relative")
    components = relative_directory.parts
    if not components or any(
        component in {"", ".", ".."} or "/" in component or "\\" in component
        for component in components
    ):
        raise ValueError("durable directory child must contain plain relative components")
    try:
        return (
            str(root).encode("utf-8", errors="strict"),
            "/".join(components).encode("utf-8", errors="strict"),
            len(components),
        )
    except UnicodeEncodeError as exc:
        raise ValueError("durable directory paths must be valid UTF-8") from exc


def prepare_durable_directory(
    root: Path,
    relative_directory: Path,
) -> TemporalDurableDirectoryPreparation:
    """Create or reconcile and flush every component below an existing root."""

    root_utf8, relative_utf8, component_count = _lexical_durable_directory_parts(
        root, relative_directory
    )
    authority_before_load = _fresh_authority()
    library = _temporal_library()
    authority_after_load = _fresh_authority()
    if authority_after_load != authority_before_load:
        raise NativeTemporalRuntimeError(
            NativeTemporalRuntimeStatus.INVALID_DEFINITION,
            "fresh authority changed while preparing durable directory",
        )
    _attested_current_library_identity()
    try:
        root_array = (ctypes.c_uint8 * len(root_utf8)).from_buffer_copy(root_utf8)
        relative_array = (ctypes.c_uint8 * len(relative_utf8)).from_buffer_copy(relative_utf8)
    except (MemoryError, OverflowError) as exc:
        raise NativeTemporalRuntimeError(
            NativeTemporalRuntimeStatus.ALLOCATION_FAILED,
            "durable-directory transport allocation failed",
            required=len(root_utf8) + len(relative_utf8),
        ) from exc
    request = _NativeDurableDirectoryPathRequest(
        ctypes.cast(root_array, ctypes.POINTER(ctypes.c_uint8)),
        len(root_utf8),
        ctypes.cast(relative_array, ctypes.POINTER(ctypes.c_uint8)),
        len(relative_utf8),
        (ctypes.c_uint8 * 8)(),
    )
    durability_report = _NativeDurableDirectoryPreparationReport()
    runtime_report = _NativeReport()
    returned = library.humanbrain_temporal_prepare_durable_directory_path_v1(
        ctypes.byref(request),
        ctypes.byref(durability_report),
        ctypes.byref(runtime_report),
    )
    return _decode_durable_directory_preparation(
        returned,
        durability_report,
        runtime_report,
        expected_component_count=component_count,
    )


def _checkpoint_file_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (int(metadata.st_dev), int(metadata.st_ino), int(metadata.st_size))


def _checkpoint_path_is_reparse(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _validate_checkpoint_regular_file(
    metadata: os.stat_result,
    *,
    descriptor: bool,
) -> tuple[int, int, int]:
    subject = "checkpoint descriptor" if descriptor else "checkpoint path"
    if stat.S_ISLNK(metadata.st_mode) or _checkpoint_path_is_reparse(metadata):
        raise _checkpoint_rejected(f"{subject} is a symlink or reparse point")
    if not stat.S_ISREG(metadata.st_mode):
        raise _checkpoint_rejected(f"{subject} is not a regular file")
    identity = _checkpoint_file_identity(metadata)
    if identity[2] < _CHECKPOINT_V1_HEADER_BYTES or identity[2] > sys.maxsize:
        raise _checkpoint_rejected(
            f"{subject} size is outside the checkpoint-v1 host range",
            observed=identity[2],
        )
    return identity


def _validate_checkpoint_v1_basic_header(header: bytes, file_size: int) -> None:
    if len(header) != _CHECKPOINT_V1_BASIC_HEADER_BYTES:
        raise _checkpoint_rejected("checkpoint-v1 basic header is truncated")
    if header[:8] != _CHECKPOINT_V1_MAGIC:
        raise _checkpoint_rejected("checkpoint-v1 magic does not match")
    version = int.from_bytes(header[8:10], "little")
    if version != _CHECKPOINT_V1_FORMAT_VERSION:
        raise _checkpoint_rejected("checkpoint-v1 format version is unsupported", observed=version)
    header_bytes = int.from_bytes(header[10:12], "little")
    if header_bytes != _CHECKPOINT_V1_HEADER_BYTES:
        raise _checkpoint_rejected(
            "checkpoint-v1 header length does not match", observed=header_bytes
        )
    if any(header[12:16]):
        raise _checkpoint_rejected("checkpoint-v1 reserved basic-header bytes are nonzero")
    declared_total = int.from_bytes(header[16:24], "little")
    if declared_total != file_size:
        raise _checkpoint_rejected(
            "checkpoint-v1 declared total does not equal the file size",
            observed=declared_total,
        )


def _validate_readonly_simple_buffer(
    view: _PyBuffer,
    expected_len: int,
    *,
    subject: str = "checkpoint mapping",
    expected_length: str = "descriptor size",
) -> None:
    if not view.obj or not view.buf:
        raise _invariant_error(f"{subject} returned a null buffer owner or address")
    if int(view.len) != expected_len:
        raise _invariant_error(
            f"{subject} buffer length differs from the {expected_length}",
            required=expected_len,
            observed=int(view.len),
        )
    if int(view.readonly) != 1:
        raise _invariant_error(f"{subject} buffer is not read-only")
    if int(view.itemsize) != 1 or int(view.ndim) != 1:
        raise _invariant_error(f"{subject} buffer has non-simple scalar geometry")
    if any(
        (
            view.format,
            view.shape,
            view.strides,
            view.suboffsets,
            view.internal,
        )
    ):
        raise _invariant_error(f"{subject} buffer has non-simple pointer geometry")


def _convert_spikes(values: Sequence[_NativeSpike]) -> tuple[TemporalSpike, ...]:
    converted: list[TemporalSpike] = []
    for spike in values:
        if spike.reserved_zero != 0:
            raise _invariant_error("spike reserved field is nonzero")
        converted.append(TemporalSpike(int(spike.cell_id), int(spike.tick)))
    return tuple(converted)


def _convert_step(step: _NativeStep) -> TemporalCircuitStep:
    if step.reserved_zero != 0:
        raise _invariant_error("step reserved field is nonzero")
    if step.predicted not in (0, 1):
        raise _invariant_error("step predicted flag is not boolean")
    if step.predicted_columns > _ACTIVE_COLUMNS:
        raise _invariant_error("step predicted-column count exceeds the active width")
    if step.allocated_segments > _ACTIVE_COLUMNS:
        raise _invariant_error("step allocation count exceeds the active width")
    return TemporalCircuitStep(
        token_id=int(step.token_id),
        punished_segments=int(step.punished_segments),
        event_count=int(step.event_count),
        predicted=bool(step.predicted),
        predicted_columns=int(step.predicted_columns),
        allocated_segments=int(step.allocated_segments),
        inhibition_onset_ticks=tuple(int(value) for value in step.inhibition_onset_ticks),
        column_simulated_ticks=tuple(int(value) for value in step.column_simulated_ticks),
        winner_spikes=_convert_spikes(step.winner_spikes),
        basket_spikes=_convert_spikes(step.basket_spikes),
        proximal_spikes=_convert_spikes(step.proximal_spikes),
        proximal_relay_spikes=_convert_spikes(step.proximal_relay_spikes),
        winner_relay_spikes=_convert_spikes(step.winner_relay_spikes),
        winner_context=tuple(int(value) for value in step.winner_context),
        proximal_context=tuple(int(value) for value in step.proximal_context),
        context=tuple(int(value) for value in step.context),
    )


def _native_definition_inputs(
    definition: NativeTemporalCircuitDefinition,
) -> tuple[_NativeConfig, object, object]:
    config = definition.config
    native_config = _NativeConfig(
        config.column_count,
        config.max_pyramidal_cells,
        config.segment_capacity,
        config.construction_seed,
        config.connected_permanence_ticks,
        config.initial_permanence_ticks,
        config.permanence_increment_ticks,
        config.negative_decrement_ticks,
        config.base_cells_per_column,
        config.novel_race_cap,
        config.prediction_quorum,
        config.max_segments_per_cell,
        config.theta_match,
        config.theta_active,
        int(config.context_policy),
        int(config.plasticity_policy),
    )
    token_array = (_NativeTokenCode * len(definition.token_codes))()
    for index, (token_id, columns) in enumerate(definition.token_codes):
        token_array[index].token_id = token_id
        token_array[index].columns[:] = columns
    threshold_array = (ctypes.c_double * len(definition.threshold_offsets_mv))(
        *definition.threshold_offsets_mv
    )
    return native_config, token_array, threshold_array


def _native_definition(
    definition: NativeTemporalCircuitDefinition,
    authority: _FreshAuthority,
) -> tuple[_NativeDefinition, object, object]:
    native_config, token_array, threshold_array = _native_definition_inputs(definition)
    native = _NativeDefinition(
        native_config,
        ctypes.cast(token_array, ctypes.POINTER(_NativeTokenCode)),
        len(token_array),
        ctypes.cast(threshold_array, ctypes.POINTER(ctypes.c_double)),
        len(threshold_array),
        authority.native(),
    )
    return native, token_array, threshold_array


def _native_definition_identity_request(
    definition: NativeTemporalCircuitDefinition,
) -> tuple[_NativeDefinitionIdentityRequest, object, object]:
    native_config, token_array, threshold_array = _native_definition_inputs(definition)
    request = _NativeDefinitionIdentityRequest(
        native_config,
        ctypes.cast(token_array, ctypes.POINTER(_NativeTokenCode)),
        len(token_array),
        ctypes.cast(threshold_array, ctypes.POINTER(ctypes.c_double)),
        len(threshold_array),
    )
    return request, token_array, threshold_array


class NativeTemporalCircuit:
    """Owned live native circuit; construct only through an authority gate."""

    __slots__ = (
        "_operation_lock",
        "_handle",
        "_library",
        "_creation_authority",
        "_library_identity",
        "_definition",
    )

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError(
            "use NativeTemporalCircuit.fresh(), from_checkpoint(), or from_checkpoint_path()"
        )

    @classmethod
    def fresh(cls, definition: NativeTemporalCircuitDefinition) -> NativeTemporalCircuit:
        return cls._open(definition, checkpoint=None)

    @classmethod
    def from_checkpoint(
        cls,
        definition: NativeTemporalCircuitDefinition,
        checkpoint: bytes,
    ) -> NativeTemporalCircuit:
        if type(checkpoint) is not bytes:
            raise TypeError("checkpoint must be immutable bytes")
        return cls._open(definition, checkpoint=checkpoint)

    @classmethod
    def from_checkpoint_path(
        cls,
        definition: NativeTemporalCircuitDefinition,
        path: Path,
    ) -> NativeTemporalCircuit:
        """Import one controlled immutable local checkpoint without copying it.

        This bounded-memory path is only for regular files produced by the
        exclusive checkpoint publisher and held immutable with no concurrent
        writer.  It is not an adversarial or general persistence boundary.
        The supplied path is used lexically and a final-component symlink or
        reparse point is rejected.
        """

        if not isinstance(path, Path):
            raise TypeError("checkpoint path must be a pathlib.Path")
        self, library, native_definition, definition_keepalive = cls._prepare_open(definition)
        get_buffer, release_buffer = _readonly_buffer_api()
        handle_out = ctypes.c_void_p()
        report = _NativeReport()
        descriptor: int | None = None
        file_object: BinaryIO | None = None
        mapping: mmap.mmap | None = None
        buffer_view = _PyBuffer()
        buffer_acquired = False
        checkpoint_pointer: object | None = None
        request: _NativeImportRequest | None = None
        validated_handle: int | None = None
        try:
            path_before = os.lstat(path)
            path_identity = _validate_checkpoint_regular_file(path_before, descriptor=False)
            open_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
            if hasattr(os, "O_NOFOLLOW"):
                open_flags |= os.O_NOFOLLOW
            descriptor = os.open(path, open_flags)
            try:
                file_object = os.fdopen(descriptor, "rb", buffering=0)
            except BaseException:
                os.close(descriptor)
                descriptor = None
                raise
            descriptor = None
            descriptor_before = os.fstat(file_object.fileno())
            descriptor_identity = _validate_checkpoint_regular_file(
                descriptor_before, descriptor=True
            )
            if descriptor_identity != path_identity:
                raise _checkpoint_rejected(
                    "checkpoint path identity changed while opening the file"
                )
            basic_header = file_object.read(_CHECKPOINT_V1_BASIC_HEADER_BYTES)
            _validate_checkpoint_v1_basic_header(basic_header, descriptor_identity[2])
            try:
                mapping = mmap.mmap(
                    file_object.fileno(),
                    descriptor_identity[2],
                    access=mmap.ACCESS_READ,
                )
            except (MemoryError, OverflowError) as exc:
                raise NativeTemporalRuntimeError(
                    NativeTemporalRuntimeStatus.ALLOCATION_FAILED,
                    "checkpoint read-only mapping allocation failed",
                    required=descriptor_identity[2],
                ) from exc
            if len(mapping) != descriptor_identity[2]:
                raise _checkpoint_rejected(
                    "checkpoint mapping length differs from the descriptor size"
                )
            try:
                acquisition_status = int(get_buffer(mapping, ctypes.byref(buffer_view), 0))
            except (TypeError, ValueError, OverflowError) as exc:
                raise _invariant_error("PyObject_GetBuffer returned a non-integer status") from exc
            if acquisition_status != 0:
                raise _invariant_error(
                    "PyObject_GetBuffer failed for a read-only checkpoint mapping",
                    observed=acquisition_status,
                )
            buffer_acquired = True
            _validate_readonly_simple_buffer(buffer_view, descriptor_identity[2])
            checkpoint_pointer = ctypes.cast(
                ctypes.c_void_p(buffer_view.buf),
                ctypes.POINTER(ctypes.c_uint8),
            )
            request = _NativeImportRequest(
                native_definition,
                checkpoint_pointer,
                descriptor_identity[2],
            )
            returned = library.humanbrain_temporal_circuit_runtime_import_checkpoint_v1(
                ctypes.byref(request),
                ctypes.byref(handle_out),
                ctypes.byref(report),
            )
            descriptor_after = os.fstat(file_object.fileno())
            path_after = os.lstat(path)
            if (
                _validate_checkpoint_regular_file(descriptor_after, descriptor=True)
                != descriptor_identity
                or _validate_checkpoint_regular_file(path_after, descriptor=False) != path_identity
            ):
                raise _checkpoint_rejected(
                    "checkpoint descriptor identity or size changed during import"
                )
            validated_handle = self._validated_open_handle(returned, report, handle_out)
            del definition_keepalive
        except BaseException:
            if handle_out.value:
                library.humanbrain_temporal_circuit_runtime_destroy(handle_out.value)
            raise
        finally:
            active_exception = sys.exc_info()[0] is not None
            request = None
            checkpoint_pointer = None
            cleanup_error: BaseException | None = None
            if buffer_acquired:
                buffer_acquired = False
                try:
                    release_buffer(ctypes.byref(buffer_view))
                except BaseException as exc:
                    cleanup_error = exc
            if mapping is not None:
                try:
                    mapping.close()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            if file_object is not None:
                try:
                    file_object.close()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            elif descriptor is not None:
                try:
                    os.close(descriptor)
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            if cleanup_error is not None and not active_exception:
                if handle_out.value:
                    library.humanbrain_temporal_circuit_runtime_destroy(handle_out.value)
                    handle_out.value = None
                raise cleanup_error
        if validated_handle is None:
            raise _invariant_error("checkpoint-path import did not validate a handle")
        self._handle = validated_handle
        return self

    @classmethod
    def _prepare_open(
        cls,
        definition: NativeTemporalCircuitDefinition,
    ) -> tuple[
        NativeTemporalCircuit,
        ctypes.CDLL,
        _NativeDefinition,
        tuple[object, object],
    ]:
        if type(definition) is not NativeTemporalCircuitDefinition:
            raise TypeError("definition must be a NativeTemporalCircuitDefinition")
        self = cls.__new__(cls)
        self._operation_lock = threading.RLock()
        self._handle = None
        self._library = None
        self._creation_authority = None
        self._library_identity = None
        self._definition = definition
        authority = _fresh_authority()
        library = _temporal_library()
        confirmed_authority = _fresh_authority()
        if confirmed_authority != authority:
            raise NativeTemporalRuntimeError(
                NativeTemporalRuntimeStatus.INVALID_DEFINITION,
                "validated authority changed while loading native code",
            )
        authority = confirmed_authority
        library_identity = _attested_current_library_identity()
        self._library = library
        self._creation_authority = authority
        self._library_identity = library_identity
        native_definition, token_array, threshold_array = _native_definition(definition, authority)
        return self, library, native_definition, (token_array, threshold_array)

    @staticmethod
    def _validated_open_handle(
        returned: object,
        report: _NativeReport,
        handle_out: ctypes.c_void_p,
    ) -> int:
        decoded = _decode_report(returned, report, maximum_completed_prefix=0)
        if decoded.status is not NativeTemporalRuntimeStatus.OK:
            if handle_out.value:
                raise _invariant_error("failed creation published a runtime handle")
            _raise_report(decoded)
        if decoded.required or decoded.observed or decoded.location:
            raise _invariant_error("successful creation carries unexpected scalar diagnostics")
        if not handle_out.value:
            raise _invariant_error("successful creation did not publish a runtime handle")
        return int(handle_out.value)

    @classmethod
    def _open(
        cls,
        definition: NativeTemporalCircuitDefinition,
        *,
        checkpoint: bytes | None,
    ) -> NativeTemporalCircuit:
        self, library, native_definition, definition_keepalive = cls._prepare_open(definition)
        # These owning arrays stay in local variables until the native create
        # or import returns; Rust copies them before publishing the handle.
        report = _NativeReport()
        handle_out = ctypes.c_void_p()
        try:
            if checkpoint is None:
                returned = library.humanbrain_temporal_circuit_runtime_create_v1(
                    ctypes.byref(native_definition),
                    ctypes.byref(handle_out),
                    ctypes.byref(report),
                )
            else:
                checkpoint_array = (ctypes.c_uint8 * len(checkpoint)).from_buffer_copy(checkpoint)
                checkpoint_pointer = ctypes.cast(checkpoint_array, ctypes.POINTER(ctypes.c_uint8))
                request = _NativeImportRequest(
                    native_definition,
                    checkpoint_pointer,
                    len(checkpoint),
                )
                returned = library.humanbrain_temporal_circuit_runtime_import_checkpoint_v1(
                    ctypes.byref(request),
                    ctypes.byref(handle_out),
                    ctypes.byref(report),
                )
            del definition_keepalive
            validated_handle = self._validated_open_handle(returned, report, handle_out)
        except BaseException:
            if handle_out.value:
                library.humanbrain_temporal_circuit_runtime_destroy(handle_out.value)
            raise
        self._handle = validated_handle
        return self

    @property
    def definition(self) -> NativeTemporalCircuitDefinition:
        return self._definition

    def __enter__(self) -> NativeTemporalCircuit:
        with self._operation_lock:
            self._require_open()
            return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __copy__(self) -> None:
        raise TypeError("NativeTemporalCircuit cannot be copied")

    def __deepcopy__(self, memo: object) -> None:
        del memo
        raise TypeError("NativeTemporalCircuit cannot be copied")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_open(self) -> int:
        if self._handle is None:
            raise RuntimeError("native temporal circuit is closed")
        return self._handle

    def close(self) -> None:
        operation_lock = getattr(self, "_operation_lock", None)
        if operation_lock is None:
            return
        with operation_lock:
            handle = getattr(self, "_handle", None)
            if handle is not None:
                self._handle = None
                library = self._library
                if library is not None:
                    library.humanbrain_temporal_circuit_runtime_destroy(handle)

    def advance_block(
        self,
        token_ids: Sequence[int],
        *,
        mode: TemporalStepMode,
        reset_before: bool = False,
    ) -> tuple[TemporalCircuitStep, ...]:
        if type(mode) is not TemporalStepMode:
            raise TypeError("mode must be a TemporalStepMode")
        if type(reset_before) is not bool:
            raise TypeError("reset_before must be an exact bool")
        if isinstance(token_ids, (str, bytes, bytearray)):
            raise TypeError("token_ids must be a sequence of exact integers")
        try:
            normalized = tuple(
                _exact_int("token ID", token_id, 0, _UINT32_MAX - 1) for token_id in token_ids
            )
        except TypeError:
            raise
        if reset_before and not normalized:
            raise ValueError("reset_before requires at least one token")
        token_array = (ctypes.c_uint32 * len(normalized))(*normalized)
        output_array = (_NativeStep * len(normalized))()
        token_pointer = ctypes.cast(token_array, ctypes.POINTER(ctypes.c_uint32))
        output_pointer = ctypes.cast(output_array, ctypes.POINTER(_NativeStep))
        request = _NativeAdvanceRequest(
            token_pointer,
            output_pointer,
            len(normalized),
            len(normalized),
            int(mode),
            int(reset_before),
            (ctypes.c_uint8 * 6)(),
        )
        report = _NativeReport()
        with self._operation_lock:
            handle = self._require_open()
            returned = self._library.humanbrain_temporal_circuit_runtime_advance_block_v1(
                handle, ctypes.byref(request), ctypes.byref(report)
            )
            try:
                decoded = _decode_report(returned, report, maximum_completed_prefix=len(normalized))
            except NativeTemporalRuntimeError as exc:
                recovered: list[TemporalCircuitStep] = []
                for index in range(exc.completed_prefix):
                    try:
                        step = _convert_step(output_array[index])
                    except NativeTemporalRuntimeError:
                        break
                    if step.token_id != normalized[index]:
                        break
                    recovered.append(step)
                raise NativeTemporalRuntimeError(
                    exc.status,
                    exc.message,
                    completed_prefix=exc.completed_prefix,
                    required=exc.required,
                    observed=exc.observed,
                    location=exc.location,
                    poison_code=exc.poison_code,
                    stage_code=exc.stage_code,
                    completed_steps=tuple(recovered),
                ) from exc
            converted: list[TemporalCircuitStep] = []
            for index in range(decoded.completed_prefix):
                try:
                    step = _convert_step(output_array[index])
                except NativeTemporalRuntimeError as exc:
                    detail = exc.message.removeprefix("native ABI invariant failed: ")
                    raise _invariant_error(
                        detail,
                        completed_prefix=decoded.completed_prefix,
                        completed_steps=tuple(converted),
                        location=index,
                    ) from exc
                if step.token_id != normalized[index]:
                    raise _invariant_error(
                        "step token ID differs from its input",
                        completed_prefix=decoded.completed_prefix,
                        completed_steps=tuple(converted),
                        required=normalized[index],
                        observed=step.token_id,
                        location=index,
                    )
                converted.append(step)
            completed_steps = tuple(converted)
            if decoded.status is not NativeTemporalRuntimeStatus.OK:
                _raise_report(decoded, completed_steps)
            if decoded.completed_prefix != len(normalized):
                raise _invariant_error(
                    "successful block did not publish every requested step",
                    completed_steps=completed_steps,
                    required=len(normalized),
                    observed=decoded.completed_prefix,
                )
            if decoded.required or decoded.observed or decoded.location:
                raise _invariant_error(
                    "successful block carries unexpected scalar diagnostics",
                    completed_steps=completed_steps,
                )
            return completed_steps

    def publish_checkpoint(self, destination: Path) -> TemporalCheckpointPublication:
        """Attempt one exclusive lexical-path publication with no wrapper retry."""

        with self._operation_lock:
            handle = self._require_open()
            fresh_authority = _fresh_authority()
            if fresh_authority != self._creation_authority:
                raise NativeTemporalRuntimeError(
                    NativeTemporalRuntimeStatus.INVALID_DEFINITION,
                    "fresh authority does not match circuit creation authority",
                )
            current_identity = _attested_current_library_identity()
            if current_identity != self._library_identity:
                raise _invariant_error("loaded native library identity changed")
            parent_utf8, final_basename_utf8 = _lexical_checkpoint_path_parts(destination)
            try:
                parent_array = (ctypes.c_uint8 * len(parent_utf8)).from_buffer_copy(parent_utf8)
                basename_array = (ctypes.c_uint8 * len(final_basename_utf8)).from_buffer_copy(
                    final_basename_utf8
                )
            except (MemoryError, OverflowError) as exc:
                raise NativeTemporalRuntimeError(
                    NativeTemporalRuntimeStatus.ALLOCATION_FAILED,
                    "checkpoint path transport allocation failed",
                    required=len(parent_utf8) + len(final_basename_utf8),
                ) from exc
            request = _NativeCheckpointPathRequest(
                ctypes.cast(parent_array, ctypes.POINTER(ctypes.c_uint8)),
                len(parent_utf8),
                ctypes.cast(basename_array, ctypes.POINTER(ctypes.c_uint8)),
                len(final_basename_utf8),
                (ctypes.c_uint8 * 8)(),
            )
            native_authority = fresh_authority.native()
            publication_report = _NativeCheckpointPublicationReport()
            runtime_report = _NativeReport()
            returned = self._library.humanbrain_temporal_circuit_runtime_export_checkpoint_path_v1(
                handle,
                ctypes.byref(native_authority),
                ctypes.byref(request),
                ctypes.byref(publication_report),
                ctypes.byref(runtime_report),
            )
            return _decode_publication(returned, publication_report, runtime_report)

    def checkpoint(self) -> bytes:
        with self._operation_lock:
            handle = self._require_open()
            fresh_authority = _fresh_authority()
            if fresh_authority != self._creation_authority:
                raise NativeTemporalRuntimeError(
                    NativeTemporalRuntimeStatus.INVALID_DEFINITION,
                    "fresh authority does not match circuit creation authority",
                )
            current_identity = _attested_current_library_identity()
            if current_identity != self._library_identity:
                raise _invariant_error("loaded native library identity changed")
            native_authority = fresh_authority.native()
            size_report = _NativeReport()
            size_status = self._library.humanbrain_temporal_circuit_runtime_checkpoint_size_v1(
                handle, ctypes.byref(native_authority), ctypes.byref(size_report)
            )
            decoded_size = _decode_report(size_status, size_report, maximum_completed_prefix=0)
            if decoded_size.status is not NativeTemporalRuntimeStatus.OK:
                _raise_report(decoded_size)
            if decoded_size.observed or decoded_size.location:
                raise _invariant_error(
                    "checkpoint-size report carries unexpected scalar diagnostics"
                )
            required = decoded_size.required
            if required < 256 or required > sys.maxsize:
                raise _invariant_error(
                    "checkpoint size is outside the canonical host range",
                    observed=required,
                )
            try:
                output = (ctypes.c_uint8 * required)()
            except (MemoryError, OverflowError) as exc:
                raise NativeTemporalRuntimeError(
                    NativeTemporalRuntimeStatus.ALLOCATION_FAILED,
                    "checkpoint output allocation failed",
                    required=required,
                ) from exc
            export_report = _NativeReport()
            export_status = self._library.humanbrain_temporal_circuit_runtime_export_checkpoint_v1(
                handle,
                ctypes.byref(native_authority),
                output,
                required,
                ctypes.byref(export_report),
            )
            decoded_export = _decode_report(
                export_status, export_report, maximum_completed_prefix=0
            )
            if decoded_export.status is not NativeTemporalRuntimeStatus.OK:
                _raise_report(decoded_export)
            if (
                decoded_export.required != required
                or decoded_export.observed != required
                or decoded_export.location != 0
            ):
                raise _invariant_error(
                    "exact checkpoint export length changed after size preflight",
                    required=required,
                    observed=decoded_export.observed,
                )
            return bytes(output)

    def checkpoint_identity(self) -> TemporalCheckpointIdentity:
        with self._operation_lock:
            handle = self._require_open()
            fresh_authority = _fresh_authority()
            if fresh_authority != self._creation_authority:
                raise NativeTemporalRuntimeError(
                    NativeTemporalRuntimeStatus.INVALID_DEFINITION,
                    "fresh authority does not match circuit creation authority",
                )
            current_identity = _attested_current_library_identity()
            if current_identity != self._library_identity:
                raise _invariant_error("loaded native library identity changed")
            native_authority = fresh_authority.native()
            native_identity = _NativeCheckpointIdentity()
            report = _NativeReport()
            returned = self._library.humanbrain_temporal_circuit_runtime_checkpoint_identity_v1(
                handle,
                ctypes.byref(native_authority),
                ctypes.byref(native_identity),
                ctypes.byref(report),
            )
            return _decode_checkpoint_identity(returned, native_identity, report, fresh_authority)


__all__ = [
    "NativeTemporalCircuitConfig",
    "NativeTemporalCircuitDefinition",
    "NativeTemporalCircuit",
    "NativeTemporalRuntimeError",
    "NativeTemporalRuntimeStatus",
    "TemporalCheckpointCleanup",
    "TemporalCheckpointErrorDomain",
    "TemporalCheckpointErrorStage",
    "TemporalCheckpointIdentity",
    "TemporalCheckpointPublication",
    "TemporalCheckpointPublicationError",
    "TemporalCheckpointPublicationOutcome",
    "TemporalImmutableBytesPublication",
    "TemporalImmutableBytesReconciliation",
    "TemporalDurableDirectoryPreparation",
    "TemporalDurabilityError",
    "TemporalDurabilityErrorDomain",
    "TemporalDurabilityErrorStage",
    "TemporalDurabilityOutcome",
    "TemporalPublicationCleanup",
    "TemporalPublicationError",
    "TemporalPublicationErrorDomain",
    "TemporalPublicationErrorStage",
    "TemporalPublicationOutcome",
    "TemporalCircuitStep",
    "TemporalContextPolicy",
    "TemporalPlasticityPolicy",
    "TemporalSpike",
    "TemporalStepMode",
    "publish_immutable_bytes",
    "prepare_durable_directory",
    "reconcile_existing_immutable_bytes",
]
