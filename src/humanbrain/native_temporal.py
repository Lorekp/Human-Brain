"""Opaque native fixed-pair temporal store and recognition index."""

from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass
from typing import Final, Sequence

from humanbrain.native import _load_library

_UINT32_MAX: Final = (1 << 32) - 1
_PAIR_COUNT: Final = 10
_MAX_PERMANENCE_TICKS: Final = 500
_STATUS_NAMES: Final = {
    0: "ok",
    1: "invalid native input or state",
    2: "segment capacity exhausted",
    3: "posting-block capacity exhausted",
    4: "owner segment capacity exhausted",
    5: "temporal store is poisoned after an earlier failure",
    6: "native bounded allocation failed",
    7: "diagnostic destination capacity is insufficient",
}


def _exact_bounded_int(owner: str, value: object, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{owner} must be an exact integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{owner} must be in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class NativeTemporalStoreConfig:
    """Frozen store bounds over one contiguous physical source-GID domain."""

    source_count: int
    owner_count: int
    segment_capacity: int
    theta_match: int
    theta_active: int
    source_gid_base: int = 0
    connected_permanence_ticks: int = 250

    def __post_init__(self) -> None:
        _exact_bounded_int(
            "source_count", self.source_count, _PAIR_COUNT, _UINT32_MAX
        )
        _exact_bounded_int("owner_count", self.owner_count, 1, _UINT32_MAX)
        _exact_bounded_int(
            "segment_capacity",
            self.segment_capacity,
            1,
            _UINT32_MAX // _PAIR_COUNT,
        )
        _exact_bounded_int(
            "source_gid_base", self.source_gid_base, 0, _UINT32_MAX
        )
        if self.source_gid_base + self.source_count > _UINT32_MAX + 1:
            raise ValueError("source_gid_base domain overflows uint32")
        _exact_bounded_int(
            "connected_permanence_ticks",
            self.connected_permanence_ticks,
            1,
            _MAX_PERMANENCE_TICKS,
        )
        _exact_bounded_int("theta_match", self.theta_match, 1, 20)
        _exact_bounded_int("theta_active", self.theta_active, 1, 20)
        if self.theta_match > self.theta_active:
            raise ValueError("theta_match cannot exceed theta_active")


@dataclass(frozen=True, slots=True)
class TemporalOwnerDiagnostics:
    owner: int
    best_matching_segment: int | None
    matching_potential_contacts: int
    matching_connected_contacts: int
    best_active_segment: int | None
    active_potential_contacts: int
    active_connected_contacts: int


@dataclass(frozen=True, slots=True)
class TemporalPrepareDiagnostics:
    candidate_segments: int
    matching_segments: int
    active_segments: int
    max_potential_contacts: int
    max_connected_contacts: int
    postings_visited: int
    owners: tuple[TemporalOwnerDiagnostics, ...]


class NativeTemporalStoreError(RuntimeError):
    def __init__(self, status: int) -> None:
        self.status = int(status)
        super().__init__(_STATUS_NAMES.get(self.status, f"native status {self.status}"))


class _NativeTemporalConfig(ctypes.Structure):
    _fields_ = [
        ("source_count", ctypes.c_size_t),
        ("owner_count", ctypes.c_size_t),
        ("segment_capacity", ctypes.c_size_t),
        ("source_gid_base", ctypes.c_uint32),
        ("connected_permanence_ticks", ctypes.c_uint16),
        ("theta_match", ctypes.c_uint8),
        ("theta_active", ctypes.c_uint8),
        ("padding_zero", ctypes.c_uint32),
    ]


class _NativeOwnerDiagnostic(ctypes.Structure):
    _fields_ = [
        ("owner", ctypes.c_uint32),
        ("best_matching_segment", ctypes.c_uint32),
        ("best_active_segment", ctypes.c_uint32),
        ("matching_potential_contacts", ctypes.c_uint8),
        ("matching_connected_contacts", ctypes.c_uint8),
        ("active_potential_contacts", ctypes.c_uint8),
        ("active_connected_contacts", ctypes.c_uint8),
    ]


class _NativePrepareSummary(ctypes.Structure):
    _fields_ = [
        ("candidate_segments", ctypes.c_size_t),
        ("matching_segments", ctypes.c_size_t),
        ("active_segments", ctypes.c_size_t),
        ("owner_result_count", ctypes.c_size_t),
        ("postings_visited", ctypes.c_uint64),
        ("max_potential_contacts", ctypes.c_uint8),
        ("max_connected_contacts", ctypes.c_uint8),
        ("padding_zero", ctypes.c_uint8 * 6),
    ]


_BINDINGS_READY = False


def _temporal_library() -> ctypes.CDLL:
    global _BINDINGS_READY
    library = _load_library()
    if _BINDINGS_READY:
        return library
    void_pointer = ctypes.c_void_p
    u32_pointer = ctypes.POINTER(ctypes.c_uint32)
    library.humanbrain_temporal_store_create.argtypes = [
        ctypes.POINTER(_NativeTemporalConfig)
    ]
    library.humanbrain_temporal_store_create.restype = void_pointer
    library.humanbrain_temporal_store_destroy.argtypes = [void_pointer]
    library.humanbrain_temporal_store_destroy.restype = None
    library.humanbrain_temporal_store_allocate.argtypes = [
        void_pointer,
        ctypes.c_uint32,
        u32_pointer,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
        u32_pointer,
    ]
    library.humanbrain_temporal_store_allocate.restype = ctypes.c_uint32
    library.humanbrain_temporal_store_prepare.argtypes = [
        void_pointer,
        u32_pointer,
        ctypes.c_size_t,
        ctypes.POINTER(_NativeOwnerDiagnostic),
        ctypes.c_size_t,
        ctypes.POINTER(_NativePrepareSummary),
    ]
    library.humanbrain_temporal_store_prepare.restype = ctypes.c_uint32
    _BINDINGS_READY = True
    return library


def _optional_u32_array(
    values: tuple[int, ...],
) -> ctypes.Array[ctypes.c_uint32] | None:
    if not values:
        return None
    return (ctypes.c_uint32 * len(values))(*values)


class NativeTemporalStore:
    """Bounded native segment store at the Python/Rust adapter seam."""

    def __init__(self, config: NativeTemporalStoreConfig) -> None:
        if not isinstance(config, NativeTemporalStoreConfig):
            raise TypeError("config must be a NativeTemporalStoreConfig")
        self._operation_lock = threading.RLock()
        self._handle: int | None = None
        self.config = config
        self._library = _temporal_library()
        self._native_config = _NativeTemporalConfig(
            config.source_count,
            config.owner_count,
            config.segment_capacity,
            config.source_gid_base,
            config.connected_permanence_ticks,
            config.theta_match,
            config.theta_active,
            0,
        )
        self._handle = self._library.humanbrain_temporal_store_create(
            ctypes.byref(self._native_config)
        )
        if not self._handle:
            raise NativeTemporalStoreError(6)
        result_capacity = min(config.owner_count, config.segment_capacity)
        self._owner_results = (_NativeOwnerDiagnostic * result_capacity)()

    def __enter__(self) -> NativeTemporalStore:
        with self._operation_lock:
            self._require_open()
            return self

    def __copy__(self) -> None:
        raise TypeError("NativeTemporalStore cannot be copied")

    def __deepcopy__(self, memo: object) -> None:
        del memo
        raise TypeError("NativeTemporalStore cannot be copied")

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_open(self) -> None:
        if not self._handle:
            raise RuntimeError("temporal store is closed")

    def close(self) -> None:
        with self._operation_lock:
            handle = self._handle
            if handle:
                self._handle = None
                self._library.humanbrain_temporal_store_destroy(handle)

    def allocate_segment(
        self,
        owner: int,
        sources: Sequence[int],
        permanence_ticks: Sequence[int],
    ) -> int:
        """Append one complete ten-pair segment and return its stable ID."""

        with self._operation_lock:
            self._require_open()
            owner_value = _exact_bounded_int(
                "owner", owner, 0, self.config.owner_count - 1
            )
            source_values = tuple(sources)
            if len(source_values) != _PAIR_COUNT:
                raise ValueError("sources must contain exactly 10 physical source GIDs")
            source_maximum = self.config.source_gid_base + self.config.source_count - 1
            normalized = tuple(
                _exact_bounded_int(
                    "sources entry",
                    source,
                    self.config.source_gid_base,
                    source_maximum,
                )
                for source in source_values
            )
            if any(left >= right for left, right in zip(normalized, normalized[1:])):
                raise ValueError("segment sources must be strictly ascending")
            tick_values = tuple(permanence_ticks)
            if len(tick_values) != _PAIR_COUNT:
                raise ValueError(
                    "permanence_ticks must contain exactly 10 tick values"
                )
            normalized_ticks = tuple(
                _exact_bounded_int(
                    "permanence_ticks entry", ticks, 1, _MAX_PERMANENCE_TICKS
                )
                for ticks in tick_values
            )
            native_sources = _optional_u32_array(normalized)
            native_ticks = (ctypes.c_uint16 * _PAIR_COUNT)(*normalized_ticks)
            segment_id = ctypes.c_uint32(_UINT32_MAX)
            status = int(
                self._library.humanbrain_temporal_store_allocate(
                    self._handle,
                    owner_value,
                    native_sources,
                    native_ticks,
                    _PAIR_COUNT,
                    ctypes.byref(segment_id),
                )
            )
            if status != 0:
                raise NativeTemporalStoreError(status)
            return int(segment_id.value)

    def prepare(self, context_sources: Sequence[int]) -> TemporalPrepareDiagnostics:
        """Recognize physical source GIDs without mutating learned state."""

        with self._operation_lock:
            self._require_open()
            source_values = tuple(context_sources)
            source_maximum = self.config.source_gid_base + self.config.source_count - 1
            normalized = tuple(
                _exact_bounded_int(
                    "context_sources entry",
                    source,
                    self.config.source_gid_base,
                    source_maximum,
                )
                for source in source_values
            )
            if len(set(normalized)) != len(normalized):
                raise ValueError("context_sources must be unique")
            native_sources = _optional_u32_array(normalized)
            summary = _NativePrepareSummary()
            status = int(
                self._library.humanbrain_temporal_store_prepare(
                    self._handle,
                    native_sources,
                    len(normalized),
                    self._owner_results,
                    len(self._owner_results),
                    ctypes.byref(summary),
                )
            )
            if status != 0:
                raise NativeTemporalStoreError(status)
            owners = tuple(
                TemporalOwnerDiagnostics(
                    owner=int(record.owner),
                    best_matching_segment=(
                        None
                        if record.best_matching_segment == _UINT32_MAX
                        else int(record.best_matching_segment)
                    ),
                    matching_potential_contacts=int(
                        record.matching_potential_contacts
                    ),
                    matching_connected_contacts=int(
                        record.matching_connected_contacts
                    ),
                    best_active_segment=(
                        None
                        if record.best_active_segment == _UINT32_MAX
                        else int(record.best_active_segment)
                    ),
                    active_potential_contacts=int(record.active_potential_contacts),
                    active_connected_contacts=int(record.active_connected_contacts),
                )
                for record in self._owner_results[: summary.owner_result_count]
            )
            return TemporalPrepareDiagnostics(
                candidate_segments=int(summary.candidate_segments),
                matching_segments=int(summary.matching_segments),
                active_segments=int(summary.active_segments),
                max_potential_contacts=int(summary.max_potential_contacts),
                max_connected_contacts=int(summary.max_connected_contacts),
                postings_visited=int(summary.postings_visited),
                owners=owners,
            )


__all__ = [
    "NativeTemporalStore",
    "NativeTemporalStoreConfig",
    "NativeTemporalStoreError",
    "TemporalOwnerDiagnostics",
    "TemporalPrepareDiagnostics",
]
