"""Authority-gated Python capability for the native temporal circuit runtime.

Importing this module defines transport records only.  It neither loads the
native library nor starts an experiment; a validated live authority is required
by :class:`NativeTemporalCircuit` before any native entry point is reached.
"""

from __future__ import annotations

import ctypes
import importlib
import math
import platform
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from types import ModuleType
from typing import Final

from humanbrain.native import (
    NativeBuildError as _NativeBuildError,
    NativeLibraryIdentity as _NativeLibraryIdentity,
    _load_library,
    _source_digest as _current_native_source_digest,
    loaded_library_identity as _loaded_library_identity,
    procedural_runtime_source_bundle_sha256 as _procedural_runtime_source_bundle_sha256,
)


_UINT8_MAX: Final = (1 << 8) - 1
_UINT16_MAX: Final = (1 << 16) - 1
_UINT32_MAX: Final = (1 << 32) - 1
_ACTIVE_COLUMNS: Final = 11
_CONTEXT_CELLS: Final = 22
_PAIRS_PER_SEGMENT: Final = 10
_MAX_PERMANENCE_TICKS: Final = 500
_GOVERNING_PROTOCOL_MODULE: Final = (
    "experiments.r3_pilot.calibration_v3_protocol"
)


class _NativeAuthority(ctypes.Structure):
    _fields_ = [
        ("protocol_identity_sha256", ctypes.c_uint8 * 32),
        ("source_bundle_identity_sha256", ctypes.c_uint8 * 32),
    ]


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


class _NativeImportRequest(ctypes.Structure):
    _fields_ = [
        ("definition", _NativeDefinition),
        ("checkpoint_bytes", ctypes.POINTER(ctypes.c_uint8)),
        ("checkpoint_len", ctypes.c_uint64),
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
    observed_offsets = tuple(
        getattr(record, name).offset for name, _ctype in record._fields_
    )
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
    _require_frozen_layout(
        _NativeConfig,
        32,
        4,
        (0, 4, 8, 12, 16, 18, 20, 22, 24, 25, 26, 27, 28, 29, 30, 31),
    )
    _require_frozen_layout(_NativeTokenCode, 48, 4, (0, 4))
    _require_frozen_layout(_NativeDefinition, 128, 8, (0, 32, 40, 48, 56, 64))
    _require_frozen_layout(_NativeImportRequest, 144, 8, (0, 128, 136))
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
        _exact_int(
            "max_pyramidal_cells", self.max_pyramidal_cells, 1, _UINT32_MAX
        )
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
        _exact_int(
            "base_cells_per_column", self.base_cells_per_column, 1, _UINT8_MAX
        )
        _exact_int("novel_race_cap", self.novel_race_cap, 1, 8)
        _exact_int("prediction_quorum", self.prediction_quorum, 1, _ACTIVE_COLUMNS)
        _exact_int(
            "max_segments_per_cell", self.max_segments_per_cell, 1, _UINT8_MAX
        )
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


_TokenCodeInput = (
    Mapping[int, Sequence[int]] | Sequence[tuple[int, Sequence[int]]]
)


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
                raise ValueError(
                    f"tokens {duplicate} and {token} must not share one column code"
                )
            seen_columns[columns] = token
            normalized.append((token, columns))
        if not normalized:
            raise ValueError("token_codes must not be empty")
        normalized.sort(key=lambda entry: entry[0])

        thresholds = tuple(threshold_offsets_mv)
        physical_cell_count = _physical_cell_count(config)
        if len(thresholds) != physical_cell_count:
            raise ValueError(
                "threshold_offsets_mv must contain exactly "
                f"{physical_cell_count} entries"
            )
        if any(type(value) is not float or not math.isfinite(value) for value in thresholds):
            raise ValueError("threshold_offsets_mv entries must be finite exact f64 values")

        object.__setattr__(self, "config", config)
        object.__setattr__(self, "token_codes", tuple(normalized))
        object.__setattr__(self, "threshold_offsets_mv", thresholds)


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
            raise ValueError(
                "retained steps cannot exceed the native committed prefix"
            )
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
        raise RuntimeError(
            "governing calibration-v3 protocol is unavailable"
        ) from exc


def _fresh_authority() -> _FreshAuthority:
    governing = _governing_protocol()
    protocol = governing.build_protocol()
    governing.validate_protocol(protocol, require_runnable=True)
    binding = protocol["source_binding"]
    governing.validate_committed_source_blobs(governing.REPO_ROOT, binding)
    governing.validate_source_bytes(governing.REPO_ROOT, binding)
    protocol_digest = _sha256_bytes(
        "protocol identity", governing.protocol_sha256()
    )
    source_digest = _sha256_bytes(
        "procedural runtime source identity",
        _procedural_runtime_source_bundle_sha256(governing.REPO_ROOT),
    )
    return _FreshAuthority(protocol_digest, source_digest)


_BINDINGS_LOCK = threading.Lock()
_BOUND_LIBRARY: object | None = None


def _bind_temporal_symbols(library: object) -> None:
    try:
        create = library.humanbrain_temporal_circuit_runtime_create_v1
        import_checkpoint = (
            library.humanbrain_temporal_circuit_runtime_import_checkpoint_v1
        )
        advance = library.humanbrain_temporal_circuit_runtime_advance_block_v1
        checkpoint_size = (
            library.humanbrain_temporal_circuit_runtime_checkpoint_size_v1
        )
        export_checkpoint = (
            library.humanbrain_temporal_circuit_runtime_export_checkpoint_v1
        )
        destroy = library.humanbrain_temporal_circuit_runtime_destroy
    except AttributeError as exc:
        raise _NativeBuildError(
            "native temporal runtime ABI symbols are unavailable"
        ) from exc

    handle_output = ctypes.POINTER(ctypes.c_void_p)
    report_output = ctypes.POINTER(_NativeReport)
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
    export_checkpoint.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NativeAuthority),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        report_output,
    ]
    destroy.argtypes = [ctypes.c_void_p]
    for function in (create, import_checkpoint, advance, checkpoint_size, export_checkpoint):
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
        completed_prefix=(
            len(completed_steps)
            if completed_prefix is None
            else completed_prefix
        ),
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
        raw_completed_prefix
        if raw_completed_prefix <= maximum_completed_prefix
        else 0
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
        raise invariant(
            "native return status is outside the frozen range", observed=returned
        )
    if int(report.status) not in NativeTemporalRuntimeStatus._value2member_map_:
        raise invariant(
            "report status is outside the frozen range", observed=int(report.status)
        )
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


def _native_definition(
    definition: NativeTemporalCircuitDefinition,
    authority: _FreshAuthority,
) -> tuple[_NativeDefinition, object, object]:
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
    native = _NativeDefinition(
        native_config,
        ctypes.cast(token_array, ctypes.POINTER(_NativeTokenCode)),
        len(token_array),
        ctypes.cast(threshold_array, ctypes.POINTER(ctypes.c_double)),
        len(threshold_array),
        authority.native(),
    )
    return native, token_array, threshold_array


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
        raise TypeError("use NativeTemporalCircuit.fresh() or from_checkpoint()")

    @classmethod
    def fresh(
        cls, definition: NativeTemporalCircuitDefinition
    ) -> NativeTemporalCircuit:
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
    def _open(
        cls,
        definition: NativeTemporalCircuitDefinition,
        *,
        checkpoint: bytes | None,
    ) -> NativeTemporalCircuit:
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
        native_definition, token_array, threshold_array = _native_definition(
            definition, authority
        )
        # These owning arrays stay in local variables until the native create
        # or import returns; Rust copies them before publishing the handle.
        _definition_keepalive = (token_array, threshold_array)
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
                checkpoint_array = (ctypes.c_uint8 * len(checkpoint)).from_buffer_copy(
                    checkpoint
                )
                checkpoint_pointer = ctypes.cast(
                    checkpoint_array, ctypes.POINTER(ctypes.c_uint8)
                )
                request = _NativeImportRequest(
                    native_definition,
                    checkpoint_pointer,
                    len(checkpoint),
                )
                returned = (
                    library.humanbrain_temporal_circuit_runtime_import_checkpoint_v1(
                        ctypes.byref(request),
                        ctypes.byref(handle_out),
                        ctypes.byref(report),
                    )
                )
            del _definition_keepalive
        except BaseException:
            if handle_out.value:
                library.humanbrain_temporal_circuit_runtime_destroy(handle_out.value)
            raise
        try:
            decoded = _decode_report(returned, report, maximum_completed_prefix=0)
            if decoded.status is not NativeTemporalRuntimeStatus.OK:
                if handle_out.value:
                    raise _invariant_error("failed creation published a runtime handle")
                _raise_report(decoded)
            if decoded.required or decoded.observed or decoded.location:
                raise _invariant_error(
                    "successful creation carries unexpected scalar diagnostics"
                )
            if not handle_out.value:
                raise _invariant_error("successful creation did not publish a runtime handle")
        except BaseException:
            if handle_out.value:
                library.humanbrain_temporal_circuit_runtime_destroy(handle_out.value)
            raise
        self._handle = int(handle_out.value)
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
                _exact_int("token ID", token_id, 0, _UINT32_MAX - 1)
                for token_id in token_ids
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
                decoded = _decode_report(
                    returned, report, maximum_completed_prefix=len(normalized)
                )
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
                    detail = exc.message.removeprefix(
                        "native ABI invariant failed: "
                    )
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
            size_status = (
                self._library.humanbrain_temporal_circuit_runtime_checkpoint_size_v1(
                    handle, ctypes.byref(native_authority), ctypes.byref(size_report)
                )
            )
            decoded_size = _decode_report(
                size_status, size_report, maximum_completed_prefix=0
            )
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
            export_status = (
                self._library.humanbrain_temporal_circuit_runtime_export_checkpoint_v1(
                    handle,
                    ctypes.byref(native_authority),
                    output,
                    required,
                    ctypes.byref(export_report),
                )
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


__all__ = [
    "NativeTemporalCircuitConfig",
    "NativeTemporalCircuitDefinition",
    "NativeTemporalCircuit",
    "NativeTemporalRuntimeError",
    "NativeTemporalRuntimeStatus",
    "TemporalCircuitStep",
    "TemporalContextPolicy",
    "TemporalPlasticityPolicy",
    "TemporalSpike",
    "TemporalStepMode",
]
