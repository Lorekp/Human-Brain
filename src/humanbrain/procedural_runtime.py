"""Native source-driven simulation over a sparse procedural working set.

The runtime's configured address space is metadata. Rust owns every biological
step, resident-state transition, event delivery, and procedural edge expansion;
Python validates and marshals one batch per :meth:`advance` call. Event batches
may be built as :class:`ProceduralInputEvent` records or as bulk NumPy arrays
with dtype :data:`PROCEDURAL_EVENT_DTYPE` (see :func:`procedural_input_events`)
-- the array form is validated vectorized and handed to the FFI zero-copy, so
host marshaling cost is independent of the event count at first order.

The eight mutable per-cell fields are the pinned GLIF5 voltage, two after-spike
currents, spike and voltage threshold components, refractory counter, recurrent
synaptic current, and eligibility trace. A cell is evictable only when all eight
are canonical positive zero (or integer zero) after a real fixed-step update.
Nonzero exponential tails therefore remain resident and updated until they
underflow or cancel exactly. Resident-capacity overflow is deterministic and
fail-stop; this slice makes no bounded-steady-state or 100-billion-updated-cell
claim.
"""

from __future__ import annotations

import ctypes
import math
import sys
import threading
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

from humanbrain.cells import AllenGlif5Parameters, FittedGlif5Constants
from humanbrain.native import _F64_POINTER, _GlifConfig, _load_library
from humanbrain.procedural_topology import (
    ProceduralTopology,
    TOPOLOGY_PROTOCOL,
    TopologyManifest,
)
from humanbrain.scale_accounting import (
    CellCounts,
    EdgeCounts,
    EventCounts,
    MeasurementInterval,
    MemoryCounts,
    ScaleAccounting,
    ScaleClaim,
    ShardMemory,
    WorkCounts,
)

_UINT64_MAX: Final = (1 << 64) - 1
_MAX_RESIDENT_CAPACITY: Final = sys.maxsize // 128
_MAX_EVENT_CAPACITY: Final = sys.maxsize // 64
_MAX_SPIKE_CAPACITY: Final = sys.maxsize // 16
_MAX_DISTINCT_CAPACITY: Final = sys.maxsize // 16
_MAX_FANOUT_CAPACITY: Final = sys.maxsize // 32
_DENSE_REFERENCE_MAX_CELLS: Final = 65_536
_SUPPORTED_TOPOLOGY: Final = "uniform_no_self"

_STATUS_NAMES: Final = {
    0: "ok",
    1: "invalid native input or state",
    2: "resident capacity exhausted",
    3: "event capacity exhausted",
    4: "duplicate canonical event key",
    5: "runtime is poisoned after an earlier failure",
    6: "uint64 biological-step counter overflow",
    7: "snapshot destination capacity is insufficient",
    8: "native bounded allocation failed",
    9: "interval spike capacity exhausted",
    10: "interval distinct-cell capacity exhausted",
}

_STEPPING_MODE_IDS = {
    "full_touch": 0,
    "event_proportional": 1,
}


def _exact_uint64(owner: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{owner} must be an exact integer")
    if not 0 <= value <= _UINT64_MAX:
        raise ValueError(f"{owner} must fit uint64")
    return value


def _positive_finite(owner: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{owner} must be a finite positive number")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{owner} must be a finite positive number")
    return scalar


def _finite_canonical(owner: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{owner} must be a finite number")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{owner} must be a finite number")
    return 0.0 if scalar == 0.0 else scalar


@dataclass(frozen=True, slots=True)
class ProceduralRuntimeConfig:
    """Bounded execution policy for one counter64-v2 address manifest."""

    manifest: TopologyManifest
    resident_capacity: int
    event_capacity: int
    spike_capacity: int
    interval_distinct_capacity: int
    recurrent_weight_pa: float
    recurrent_delay_steps: int
    recurrent_tau_ms: float = 8.0
    eligibility_tau_ms: float = 80.0
    stepping_mode: str = "full_touch"

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, TopologyManifest):
            raise TypeError("manifest must be a TopologyManifest")
        if self.manifest.protocol != TOPOLOGY_PROTOCOL:
            raise ValueError("runtime requires the counter64-v2 topology protocol")
        if self.manifest.topology != _SUPPORTED_TOPOLOGY:
            raise ValueError(
                f"first procedural runtime slice requires {_SUPPORTED_TOPOLOGY!r}"
            )
        if self.manifest.cell_count > _UINT64_MAX:
            raise ValueError("runtime cell_count must fit uint64")
        if self.manifest.fanout > _MAX_FANOUT_CAPACITY:
            raise ValueError(
                f"runtime fanout must be at most {_MAX_FANOUT_CAPACITY}"
            )
        for name, value, maximum in (
            ("resident_capacity", self.resident_capacity, _MAX_RESIDENT_CAPACITY),
            ("event_capacity", self.event_capacity, _MAX_EVENT_CAPACITY),
            ("spike_capacity", self.spike_capacity, _MAX_SPIKE_CAPACITY),
            (
                "interval_distinct_capacity",
                self.interval_distinct_capacity,
                _MAX_DISTINCT_CAPACITY,
            ),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be in [1, {maximum}]")
        if self.resident_capacity > self.manifest.cell_count:
            raise ValueError("resident_capacity cannot exceed configured cells")
        if self.interval_distinct_capacity < self.resident_capacity:
            raise ValueError(
                "interval_distinct_capacity cannot be below resident_capacity"
            )
        _positive_finite("recurrent_weight_pa", self.recurrent_weight_pa)
        if self.stepping_mode not in _STEPPING_MODE_IDS:
            raise ValueError(
                "stepping_mode must be 'full_touch' or 'event_proportional'"
            )
        delay = _exact_uint64("recurrent_delay_steps", self.recurrent_delay_steps)
        if delay == 0:
            raise ValueError("recurrent_delay_steps must be at least one")
        _positive_finite("recurrent_tau_ms", self.recurrent_tau_ms)
        _positive_finite("eligibility_tau_ms", self.eligibility_tau_ms)
        dt_ms = AllenGlif5Parameters().dt_ms
        for name, tau_ms in (
            ("recurrent_tau_ms", self.recurrent_tau_ms),
            ("eligibility_tau_ms", self.eligibility_tau_ms),
        ):
            decay = math.exp(-dt_ms / float(tau_ms))
            if not 0.0 < decay < 1.0:
                raise ValueError(
                    f"{name} must produce a finite decay strictly inside (0, 1)"
                )


@dataclass(frozen=True, slots=True)
class ProceduralInputEvent:
    """One current delivery with an address-space source as provenance.

    ``source_gid`` is always a configured cell identity in this first slice;
    there is no sentinel value. For caller-injected events, ``edge_slot`` is a
    bounded provenance/order value and does not assert that the target is that
    source's procedural base-edge target. Native spike-generated events do use
    their real base-edge slot. Neither provenance field materializes the source.
    """

    due_step: int
    target_gid: int
    source_gid: int
    edge_slot: int
    current_pa: float

    def __post_init__(self) -> None:
        for name in ("due_step", "target_gid", "source_gid", "edge_slot"):
            _exact_uint64(f"event.{name}", getattr(self, name))
        object.__setattr__(
            self,
            "current_pa",
            _finite_canonical("event.current_pa", self.current_pa),
        )


@dataclass(frozen=True, slots=True)
class ProceduralResidentState:
    gid: int
    voltage_relative_mv: float
    after_spike_currents_pa: tuple[float, float]
    threshold_spike_mv: float
    threshold_voltage_mv: float
    refractory_steps: int
    synaptic_current_pa: float
    eligibility: float


@dataclass(frozen=True, slots=True)
class ProceduralSpike:
    step: int
    gid: int


@dataclass(frozen=True, slots=True)
class ProceduralSnapshot:
    current_step: int
    residents: tuple[ProceduralResidentState, ...]
    pending_events: tuple[ProceduralInputEvent, ...]
    spikes: tuple[ProceduralSpike, ...]
    poisoned: bool
    overflow_reason: str | None
    native_payload_bytes: int


@dataclass(frozen=True, slots=True)
class ProceduralAdvance:
    spikes: tuple[ProceduralSpike, ...]
    accounting: ScaleAccounting
    retained_nonzero_cells: int
    resident_capacity: int
    event_capacity: int
    spike_capacity: int
    interval_distinct_capacity: int
    capacity_overflowed: bool
    configured_ids_scanned: int


class ProceduralRuntimeError(RuntimeError):
    def __init__(self, status: int) -> None:
        self.status = int(status)
        super().__init__(_STATUS_NAMES.get(self.status, f"native status {self.status}"))


class _NativeProceduralConfig(ctypes.Structure):
    _fields_ = [
        ("configured_cells", ctypes.c_uint64),
        ("fanout", ctypes.c_uint64),
        ("seed", ctypes.c_uint64),
        ("resident_capacity", ctypes.c_size_t),
        ("event_capacity", ctypes.c_size_t),
        ("spike_capacity", ctypes.c_size_t),
        ("interval_distinct_capacity", ctypes.c_size_t),
        ("recurrent_weight_pa", ctypes.c_double),
        ("recurrent_delay_steps", ctypes.c_uint64),
        ("recurrent_decay", ctypes.c_double),
        ("eligibility_decay", ctypes.c_double),
        ("stepping_mode", ctypes.c_uint32),
    ]


class _NativeEvent(ctypes.Structure):
    _fields_ = [
        ("due_step", ctypes.c_uint64),
        ("target_gid", ctypes.c_uint64),
        ("source_gid", ctypes.c_uint64),
        ("edge_slot", ctypes.c_uint64),
        ("current_pa", ctypes.c_double),
    ]


class _NativeResident(ctypes.Structure):
    _fields_ = [
        ("gid", ctypes.c_uint64),
        ("voltage_relative_mv", ctypes.c_double),
        ("after_spike_current_0_pa", ctypes.c_double),
        ("after_spike_current_1_pa", ctypes.c_double),
        ("threshold_spike_mv", ctypes.c_double),
        ("threshold_voltage_mv", ctypes.c_double),
        ("refractory_steps", ctypes.c_int32),
        ("synaptic_current_pa", ctypes.c_double),
        ("eligibility", ctypes.c_double),
    ]


class _NativeSpike(ctypes.Structure):
    _fields_ = [("step", ctypes.c_uint64), ("gid", ctypes.c_uint64)]


class _NativeDormant(ctypes.Structure):
    _fields_ = [
        ("gid", ctypes.c_uint64),
        ("eligibility_raw", ctypes.c_double),
        ("elig_step", ctypes.c_uint64),
    ]


#: NumPy structured dtype laid out byte-for-byte like :class:`_NativeEvent`
#: (four little-endian ``uint64`` keys followed by one ``float64`` current;
#: packed, no padding, itemsize 40). A contiguous 1-D array of this dtype can
#: be handed to :meth:`NativeProceduralRuntime.advance` directly -- the FFI
#: receives a zero-copy pointer into the array's own buffer, bypassing
#: per-event Python objects entirely.
PROCEDURAL_EVENT_DTYPE: Final = np.dtype(
    [
        ("due_step", "<u8"),
        ("target_gid", "<u8"),
        ("source_gid", "<u8"),
        ("edge_slot", "<u8"),
        ("current_pa", "<f8"),
    ]
)


def _assert_event_layout() -> None:
    """Fail at import if the NumPy view and the C struct ever drift apart."""
    if PROCEDURAL_EVENT_DTYPE.itemsize != ctypes.sizeof(_NativeEvent):
        raise RuntimeError(
            "PROCEDURAL_EVENT_DTYPE itemsize "
            f"{PROCEDURAL_EVENT_DTYPE.itemsize} differs from _NativeEvent "
            f"size {ctypes.sizeof(_NativeEvent)}"
        )
    for name, _field in _NativeEvent._fields_:
        c_offset = getattr(_NativeEvent, name).offset
        np_offset = PROCEDURAL_EVENT_DTYPE.fields[name][1]
        if c_offset != np_offset:
            raise RuntimeError(
                f"event field {name!r} offset mismatch: "
                f"_NativeEvent={c_offset} dtype={np_offset}"
            )


_assert_event_layout()



class _NativeCounters(ctypes.Structure):
    _fields_ = [
        ("current_step", ctypes.c_uint64),
        ("resident_peak", ctypes.c_size_t),
        ("resident_distinct", ctypes.c_size_t),
        ("updated_distinct", ctypes.c_size_t),
        ("spiking_distinct", ctypes.c_size_t),
        ("cell_step_evaluations", ctypes.c_uint64),
        ("spike_events", ctypes.c_uint64),
        ("generated_edges", ctypes.c_uint64),
        ("delivered_events", ctypes.c_uint64),
        ("configured_ids_scanned", ctypes.c_uint64),
        ("queue_peak", ctypes.c_size_t),
        ("native_payload_bytes", ctypes.c_size_t),
        ("retained_nonzero_cells", ctypes.c_size_t),
        ("overflow_code", ctypes.c_uint32),
        # Explicit tail padding mirroring the Rust `padding_zero` field: the
        # native snapshot copies the whole struct byte-for-byte, so declaring
        # it here documents that the full FFI image is defined (and keeps raw
        # cross-run comparisons of these bytes deterministic).
        ("padding_zero", ctypes.c_uint32),
    ]


_BINDINGS_READY = False


def _procedural_library() -> ctypes.CDLL:
    global _BINDINGS_READY
    library = _load_library()
    if _BINDINGS_READY:
        return library
    void_pointer = ctypes.c_void_p
    event_pointer = ctypes.POINTER(_NativeEvent)
    resident_pointer = ctypes.POINTER(_NativeResident)
    spike_pointer = ctypes.POINTER(_NativeSpike)
    size_pointer = ctypes.POINTER(ctypes.c_size_t)

    library.humanbrain_procedural_runtime_create.argtypes = [
        ctypes.POINTER(_GlifConfig),
        ctypes.POINTER(_NativeProceduralConfig),
        _F64_POINTER,
        _F64_POINTER,
        _F64_POINTER,
    ]
    library.humanbrain_procedural_runtime_create.restype = void_pointer
    library.humanbrain_procedural_runtime_destroy.argtypes = [void_pointer]
    library.humanbrain_procedural_runtime_destroy.restype = None
    library.humanbrain_procedural_runtime_advance.argtypes = [
        void_pointer,
        ctypes.c_uint64,
        event_pointer,
        ctypes.c_size_t,
    ]
    library.humanbrain_procedural_runtime_advance.restype = ctypes.c_uint32
    library.humanbrain_procedural_runtime_counts.argtypes = [
        void_pointer,
        size_pointer,
        size_pointer,
        size_pointer,
    ]
    library.humanbrain_procedural_runtime_counts.restype = ctypes.c_uint32
    library.humanbrain_procedural_runtime_snapshot.argtypes = [
        void_pointer,
        resident_pointer,
        ctypes.c_size_t,
        event_pointer,
        ctypes.c_size_t,
        spike_pointer,
        ctypes.c_size_t,
        ctypes.POINTER(_NativeCounters),
    ]
    library.humanbrain_procedural_runtime_snapshot.restype = ctypes.c_uint32
    library.humanbrain_procedural_runtime_dormant_snapshot.argtypes = [
        void_pointer,
        ctypes.POINTER(_NativeDormant),
        ctypes.c_size_t,
        size_pointer,
    ]
    library.humanbrain_procedural_runtime_dormant_snapshot.restype = (
        ctypes.c_uint32
    )
    library.humanbrain_procedural_runtime_read_cells.argtypes = [
        void_pointer,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        resident_pointer,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    library.humanbrain_procedural_runtime_read_cells.restype = ctypes.c_uint32

    library.humanbrain_dense_reference_create.argtypes = [
        ctypes.POINTER(_GlifConfig),
        ctypes.POINTER(_NativeProceduralConfig),
        _F64_POINTER,
        _F64_POINTER,
        _F64_POINTER,
        ctypes.POINTER(ctypes.c_uint64),
        _F64_POINTER,
    ]
    library.humanbrain_dense_reference_create.restype = void_pointer
    library.humanbrain_dense_reference_destroy.argtypes = [void_pointer]
    library.humanbrain_dense_reference_destroy.restype = None
    library.humanbrain_dense_reference_advance.argtypes = [
        void_pointer,
        ctypes.c_uint64,
        event_pointer,
        ctypes.c_size_t,
    ]
    library.humanbrain_dense_reference_advance.restype = ctypes.c_uint32
    library.humanbrain_dense_reference_counts.argtypes = [
        void_pointer,
        size_pointer,
        size_pointer,
        size_pointer,
    ]
    library.humanbrain_dense_reference_counts.restype = ctypes.c_uint32
    library.humanbrain_dense_reference_snapshot.argtypes = [
        void_pointer,
        resident_pointer,
        ctypes.c_size_t,
        event_pointer,
        ctypes.c_size_t,
        spike_pointer,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.humanbrain_dense_reference_snapshot.restype = ctypes.c_uint32
    _BINDINGS_READY = True
    return library


def _native_parameters(
) -> tuple[_GlifConfig, np.ndarray, np.ndarray, np.ndarray]:
    parameters = AllenGlif5Parameters()
    constants = FittedGlif5Constants.from_parameters(parameters)
    glif = _GlifConfig(
        1,
        len(constants.after_spike_decay),
        parameters.spike_cut_steps,
        parameters.leak_ns,
        constants.voltage_step_scale,
        constants.threshold_beta_scale,
        constants.threshold_exp_decay,
        constants.threshold_voltage_factor,
        constants.threshold_beta_factor,
        constants.spike_threshold_decay,
        parameters.threshold_inf_relative_mv,
        parameters.voltage_reset_scale,
        parameters.voltage_reset_offset_mv,
        constants.spike_cut_threshold_decay,
        parameters.threshold_spike_increment_mv,
    )
    return (
        glif,
        constants.after_spike_decay,
        constants.after_spike_cut_decay,
        constants.after_spike_amplitudes_pa,
    )


def _native_runtime_config(config: ProceduralRuntimeConfig) -> _NativeProceduralConfig:
    parameters = AllenGlif5Parameters()
    return _NativeProceduralConfig(
        config.manifest.cell_count,
        config.manifest.fanout,
        config.manifest.seed,
        config.resident_capacity,
        config.event_capacity,
        config.spike_capacity,
        config.interval_distinct_capacity,
        config.recurrent_weight_pa,
        config.recurrent_delay_steps,
        np.exp(-parameters.dt_ms / config.recurrent_tau_ms),
        np.exp(-parameters.dt_ms / config.eligibility_tau_ms),
        _STEPPING_MODE_IDS[config.stepping_mode],
    )


def _event_batch(
    config: ProceduralRuntimeConfig,
    current_step: int,
    events: Sequence[ProceduralInputEvent],
) -> ctypes.Array[_NativeEvent] | None:
    if len(events) > config.event_capacity:
        raise ValueError("event batch exceeds event_capacity before native dispatch")
    native: list[_NativeEvent] = []
    seen: set[tuple[int, int, int, int]] = set()
    for event in events:
        if not isinstance(event, ProceduralInputEvent):
            raise TypeError("events must contain ProceduralInputEvent records")
        if event.due_step < current_step:
            raise ValueError("event due_step cannot precede the runtime step")
        if event.target_gid >= config.manifest.cell_count:
            raise ValueError("event target_gid is outside the configured address space")
        if event.source_gid >= config.manifest.cell_count:
            raise ValueError("event source_gid is outside the configured address space")
        if event.edge_slot >= config.manifest.fanout:
            raise ValueError("event edge_slot is outside the configured source row")
        key = (event.due_step, event.target_gid, event.source_gid, event.edge_slot)
        if key in seen:
            raise ValueError("events contain a duplicate canonical ordering key")
        seen.add(key)
        native.append(
            _NativeEvent(
                event.due_step,
                event.target_gid,
                event.source_gid,
                event.edge_slot,
                event.current_pa,
            )
        )
    if not native:
        return None
    return (_NativeEvent * len(native))(*native)


def procedural_input_events(
    due_steps: object,
    target_gids: object,
    source_gids: object,
    edge_slots: object,
    current_pa: object,
) -> np.ndarray:
    """Build a bulk :data:`PROCEDURAL_EVENT_DTYPE` batch without per-event
    Python objects.

    Each argument is either an integer/float scalar (broadcast across the
    batch) or a sequence/array of exact integers (``uint64`` columns) /
    finite floats (``current_pa``). Record-level semantics match
    :class:`ProceduralInputEvent.__post_init__`: integer columns are rejected
    when any entry is negative or exceeds ``uint64``, and ``-0.0`` currents
    canonicalize to ``+0.0``. Batch-level rules (due-step freshness, address
    space, fanout bound, duplicate canonical keys, capacity) are enforced by
    :meth:`NativeProceduralRuntime.advance` exactly as for record lists.

    Returns a 1-D C-contiguous structured array whose buffer can be passed
    straight into the C-ABI advance call.
    """
    columns = []
    for name, value in (
        ("due_step", due_steps),
        ("target_gid", target_gids),
        ("source_gid", source_gids),
        ("edge_slot", edge_slots),
    ):
        if isinstance(value, np.ndarray):
            values = value
            if values.dtype.kind not in ("i", "u"):
                raise TypeError(f"events.{name} must be integer-typed")
        else:
            if isinstance(value, (list, tuple)):
                values = np.asarray(value)
                if values.dtype.kind == "O":
                    # Ints beyond int64 range arrive as object arrays; keep
                    # the exact-integer contract before converting.
                    if not all(type(entry) is int for entry in values):
                        raise TypeError(
                            f"events.{name} entries must be exact integers"
                        )
                elif values.dtype.kind not in ("i", "u"):
                    raise TypeError(
                        f"events.{name} entries must be exact integers"
                    )
            else:
                if isinstance(value, np.generic):
                    if value.dtype.kind not in ("i", "u"):
                        raise TypeError(f"events.{name} must be an exact integer")
                elif type(value) is not int:
                    raise TypeError(f"events.{name} must be an exact integer")
                values = np.asarray([value])
        if values.size:
            if values.dtype.kind == "O":
                if min(values) < 0 or max(values) > _UINT64_MAX:
                    raise ValueError(f"event.{name} must fit uint64")
            elif int(values.min()) < 0 or int(values.max()) > _UINT64_MAX:
                raise ValueError(f"event.{name} must fit uint64")
        columns.append(values.astype(np.uint64, copy=False))
    currents = np.asarray(current_pa, dtype=np.float64)
    if currents.size and not bool(np.all(np.isfinite(currents))):
        raise ValueError("event.current_pa must be a finite number")
    currents = currents.astype(np.float64, copy=True)
    currents[currents == 0.0] = 0.0  # canonicalize -0.0 like the record path
    shape = np.broadcast_shapes(*(c.shape for c in columns), currents.shape)
    events = np.empty(int(np.prod(shape)) if shape else 1, dtype=PROCEDURAL_EVENT_DTYPE)
    flat_columns = [np.ravel(np.broadcast_to(c, shape)) for c in columns]
    flat_currents = np.ravel(np.broadcast_to(currents, shape))
    events["due_step"] = flat_columns[0]
    events["target_gid"] = flat_columns[1]
    events["source_gid"] = flat_columns[2]
    events["edge_slot"] = flat_columns[3]
    events["current_pa"] = flat_currents
    return events


def _event_batch_array(
    config: ProceduralRuntimeConfig,
    current_step: int,
    events: np.ndarray,
) -> tuple[np.ndarray, ctypes.POINTER[_NativeEvent] | None, int]:
    """Validate one bulk :data:`PROCEDURAL_EVENT_DTYPE` batch vectorized.

    Applies exactly the rules of :func:`_event_batch` -- capacity, stale
    due steps, configured address space, fanout bound, finite currents,
    duplicate canonical keys -- using array reductions instead of a Python
    loop. Returns ``(buffer_owner, pointer, count)``: the caller must keep
    ``buffer_owner`` referenced until the FFI call has returned. The pointer
    aliases the caller's buffer (zero-copy) unless normalization
    (non-contiguity, misalignment, ``-0.0`` canonicalization) forces a copy.
    """
    if not isinstance(events, np.ndarray) or events.dtype != PROCEDURAL_EVENT_DTYPE:
        raise TypeError(
            "bulk events must be a NumPy array with dtype "
            "PROCEDURAL_EVENT_DTYPE"
        )
    flat = np.ascontiguousarray(events.reshape(-1))
    count = int(flat.shape[0])
    if count > config.event_capacity:
        raise ValueError("event batch exceeds event_capacity before native dispatch")
    if count == 0:
        return flat, None, 0
    due_steps = flat["due_step"]
    targets = flat["target_gid"]
    sources = flat["source_gid"]
    slots = flat["edge_slot"]
    if int(due_steps.min()) < current_step:
        raise ValueError("event due_step cannot precede the runtime step")
    cell_bound = config.manifest.cell_count
    if int(targets.max()) >= cell_bound:
        raise ValueError("event target_gid is outside the configured address space")
    if int(sources.max()) >= cell_bound:
        raise ValueError("event source_gid is outside the configured address space")
    fanout = config.manifest.fanout
    if int(slots.max()) >= fanout:
        raise ValueError("event edge_slot is outside the configured source row")
    currents = flat["current_pa"]
    if not bool(np.all(np.isfinite(currents))):
        raise ValueError("event.current_pa must be a finite number")
    zero_mask = currents == 0.0
    if bool(np.any(zero_mask)):
        # Canonicalize -0.0 to +0.0 exactly like ProceduralInputEvent does;
        # this is the one case that gives up the zero-copy alias.
        normalized = np.array(flat, copy=True)
        normalized["current_pa"][zero_mask] = 0.0
        flat = normalized
    order = np.lexsort((slots, sources, targets, due_steps))
    if count > 1:
        adjacent_equal = np.ones(count - 1, dtype=bool)
        for column in (due_steps, targets, sources, slots):
            ordered = column[order]
            adjacent_equal &= ordered[1:] == ordered[:-1]
        if bool(np.any(adjacent_equal)):
            raise ValueError("events contain a duplicate canonical ordering key")
    address = flat.ctypes.data
    alignment = ctypes.alignment(ctypes.c_double)
    if address % alignment:
        flat = np.array(flat, copy=True)
        address = flat.ctypes.data
    if address % alignment:  # pragma: no cover - NumPy allocations are aligned
        raise RuntimeError("normalized event buffer is not pointer-aligned")
    return flat, ctypes.cast(address, ctypes.POINTER(_NativeEvent)), count


def _resident(record: _NativeResident) -> ProceduralResidentState:
    return ProceduralResidentState(
        gid=int(record.gid),
        voltage_relative_mv=float(record.voltage_relative_mv),
        after_spike_currents_pa=(
            float(record.after_spike_current_0_pa),
            float(record.after_spike_current_1_pa),
        ),
        threshold_spike_mv=float(record.threshold_spike_mv),
        threshold_voltage_mv=float(record.threshold_voltage_mv),
        refractory_steps=int(record.refractory_steps),
        synaptic_current_pa=float(record.synaptic_current_pa),
        eligibility=float(record.eligibility),
    )


def _event(record: _NativeEvent) -> ProceduralInputEvent:
    return ProceduralInputEvent(
        due_step=int(record.due_step),
        target_gid=int(record.target_gid),
        source_gid=int(record.source_gid),
        edge_slot=int(record.edge_slot),
        current_pa=float(record.current_pa),
    )


def _spike(record: _NativeSpike) -> ProceduralSpike:
    return ProceduralSpike(step=int(record.step), gid=int(record.gid))


class NativeProceduralRuntime:
    """Opaque native sparse runtime at the module's production seam."""

    def __init__(self, config: ProceduralRuntimeConfig) -> None:
        if not isinstance(config, ProceduralRuntimeConfig):
            raise TypeError("config must be a ProceduralRuntimeConfig")
        self._operation_lock = threading.RLock()
        self._handle: int | None = None
        self.config = config
        self._library = _procedural_library()
        (
            self._glif,
            self._after_decay,
            self._after_cut_decay,
            self._after_amplitudes,
        ) = _native_parameters()
        self._native_config = _native_runtime_config(config)
        self._handle = self._library.humanbrain_procedural_runtime_create(
            ctypes.byref(self._glif),
            ctypes.byref(self._native_config),
            self._after_decay.ctypes.data_as(_F64_POINTER),
            self._after_cut_decay.ctypes.data_as(_F64_POINTER),
            self._after_amplitudes.ctypes.data_as(_F64_POINTER),
        )
        if not self._handle:
            raise ProceduralRuntimeError(1)
        self._current_step = 0

    def __enter__(self) -> NativeProceduralRuntime:
        with self._operation_lock:
            self._require_open()
            return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_open(self) -> None:
        if not self._handle:
            raise RuntimeError("procedural runtime is closed")

    def close(self) -> None:
        with self._operation_lock:
            handle = self._handle
            if handle:
                self._handle = None
                self._library.humanbrain_procedural_runtime_destroy(handle)

    def advance(
        self,
        step_count: int,
        events: Sequence[ProceduralInputEvent] | np.ndarray = (),
    ) -> ProceduralAdvance:
        """Advance the runtime, delivering caller-supplied external events.

        ``events`` is either a sequence of :class:`ProceduralInputEvent`
        records or a contiguous 1-D NumPy array with dtype
        :data:`PROCEDURAL_EVENT_DTYPE`. The array form marshals in bulk --
        vectorized validation and one zero-copy buffer pointer -- and applies
        the identical rule set (capacity, freshness, address space, fanout
        bound, finite currents, duplicate canonical keys) as the record form.
        """
        with self._operation_lock:
            return self._advance_locked(step_count, events)

    def _advance_locked(
        self,
        step_count: int,
        events: Sequence[ProceduralInputEvent] | np.ndarray,
    ) -> ProceduralAdvance:
        self._require_open()
        steps = _exact_uint64("step_count", step_count)
        if steps == 0:
            raise ValueError("step_count must be at least one")
        if self._current_step + steps > _UINT64_MAX:
            raise ValueError("advance would overflow the uint64 biological-step counter")
        if isinstance(events, np.ndarray):
            bulk_buffer, native_events, event_count = _event_batch_array(
                self.config, self._current_step, events
            )
        else:
            bulk_buffer = None
            native_events = _event_batch(self.config, self._current_step, events)
            event_count = 0 if native_events is None else len(native_events)
        status = int(
            self._library.humanbrain_procedural_runtime_advance(
                self._handle,
                steps,
                native_events,
                event_count,
            )
        )
        if status != 0:
            raise ProceduralRuntimeError(status)
        start_step = self._current_step
        self._current_step += steps
        snapshot, counters = self._read_snapshot(include_state=False)
        accounting = ScaleAccounting(
            claim=ScaleClaim(
                label="procedural-runtime-correctness",
                text=(
                    "Native source-driven fitted-cell execution; this record "
                    "contains no speed or capability claim."
                ),
            ),
            interval=MeasurementInterval(
                name="native-procedural-advance",
                start_step=start_step,
                step_count=steps,
                biological_step_seconds=AllenGlif5Parameters().dt_ms / 1_000.0,
                wall_seconds=0.0,
            ),
            cells=CellCounts(
                configured=self.config.manifest.cell_count,
                addressable=self.config.manifest.cell_count,
                resident_peak=int(counters.resident_peak),
                resident_distinct=int(counters.resident_distinct),
                updated_distinct=int(counters.updated_distinct),
                spiking_distinct=int(counters.spiking_distinct),
            ),
            work=WorkCounts(
                cell_step_evaluations=int(counters.cell_step_evaluations),
                spike_events=int(counters.spike_events),
                analytic_catchups=0,
                analytic_step_equivalents=0,
            ),
            edges=EdgeCounts(
                configured=(
                    self.config.manifest.cell_count * self.config.manifest.fanout
                ),
                resident=0,
                mutable=0,
                generated=int(counters.generated_edges),
            ),
            events=EventCounts(
                delivered=int(counters.delivered_events),
                queue_peak=int(counters.queue_peak),
                sent_bytes_by_shard_pair=(),
            ),
            memory=MemoryCounts(
                bytes_by_shard=(
                    ShardMemory(
                        shard="cpu-native-procedural",
                        host_bytes=int(counters.native_payload_bytes),
                        device_bytes=0,
                    ),
                )
            ),
        )
        return ProceduralAdvance(
            spikes=snapshot.spikes,
            accounting=accounting,
            retained_nonzero_cells=int(counters.retained_nonzero_cells),
            resident_capacity=self.config.resident_capacity,
            event_capacity=self.config.event_capacity,
            spike_capacity=self.config.spike_capacity,
            interval_distinct_capacity=self.config.interval_distinct_capacity,
            capacity_overflowed=False,
            configured_ids_scanned=int(counters.configured_ids_scanned),
        )

    def _read_snapshot(
        self, *, include_state: bool
    ) -> tuple[ProceduralSnapshot, _NativeCounters]:
        resident_count = ctypes.c_size_t()
        pending_count = ctypes.c_size_t()
        spike_count = ctypes.c_size_t()
        status = int(
            self._library.humanbrain_procedural_runtime_counts(
                self._handle,
                ctypes.byref(resident_count),
                ctypes.byref(pending_count),
                ctypes.byref(spike_count),
            )
        )
        if status != 0:
            raise ProceduralRuntimeError(status)
        resident_records = (
            (_NativeResident * resident_count.value)()
            if include_state and resident_count.value
            else None
        )
        pending_records = (
            (_NativeEvent * pending_count.value)()
            if include_state and pending_count.value
            else None
        )
        spike_records = (
            (_NativeSpike * spike_count.value)() if spike_count.value else None
        )
        counters = _NativeCounters()
        status = int(
            self._library.humanbrain_procedural_runtime_snapshot(
                self._handle,
                resident_records,
                0 if resident_records is None else len(resident_records),
                pending_records,
                0 if pending_records is None else len(pending_records),
                spike_records,
                0 if spike_records is None else len(spike_records),
                ctypes.byref(counters),
            )
        )
        if status != 0:
            raise ProceduralRuntimeError(status)
        overflow = int(counters.overflow_code)
        snapshot = ProceduralSnapshot(
            current_step=int(counters.current_step),
            residents=(
                tuple(_resident(record) for record in resident_records)
                if resident_records is not None
                else ()
            ),
            pending_events=(
                tuple(_event(record) for record in pending_records)
                if pending_records is not None
                else ()
            ),
            spikes=(
                tuple(_spike(record) for record in spike_records)
                if spike_records is not None
                else ()
            ),
            poisoned=overflow != 0,
            overflow_reason=_STATUS_NAMES.get(overflow) if overflow else None,
            native_payload_bytes=int(counters.native_payload_bytes),
        )
        return snapshot, counters

    def snapshot(self) -> ProceduralSnapshot:
        with self._operation_lock:
            self._require_open()
            snapshot, _ = self._read_snapshot(include_state=True)
            return snapshot

    def read_cells(
        self, gids: Sequence[int]
    ) -> tuple[ProceduralResidentState | None, ...]:
        """Targeted read of active resident state, in request order.

        ``None`` marks a requested cell that is not actively resident (not
        in ``residents``/``staged`` at this instant). Dormant cells
        (event-proportional mode) read as absent; use ``dormant_cells()``
        for those. Duplicate gids get independent repeated entries and
        results align positionally with the request. Out-of-range gids are
        rejected here before any FFI call as a courtesy; natively they are
        simply absent. Caller obligation: re-read after every
        :meth:`advance` -- eligibility mirrors cached across steps silently
        stale the learning inputs they feed.
        """
        with self._operation_lock:
            self._require_open()
            count = len(gids)
            if count == 0:
                return ()
            cell_bound = self.config.manifest.cell_count
            values = []
            append_value = values.append
            for gid in gids:
                value = _exact_uint64("gids entry", gid)
                if value >= cell_bound:
                    raise ValueError(
                        f"gids entry {value} is outside the configured "
                        "address space"
                    )
                append_value(value)
            native_gids = (ctypes.c_uint64 * count)(*values)
            records = (_NativeResident * count)()
            found = (ctypes.c_uint8 * count)()
            status = self._library.humanbrain_procedural_runtime_read_cells(
                self._handle,
                native_gids,
                count,
                records,
                found,
            )
            if status != 0:
                raise ProceduralRuntimeError(status)
            return tuple(
                _resident(records[index]) if found[index] else None
                for index in range(count)
            )

    def dormant_cells(self) -> tuple[tuple[int, float, int], ...]:
        """Raw dormant-store contents as ``(gid, eligibility_raw, elig_step)``.

        Event-proportional mode only; ``full_touch`` runtimes always report an
        empty tuple. Values are the lazily stored pre-catch-up numbers: to
        obtain the equivalent full-touch eligibility at the current step,
        replay ``current_step - elig_step - 1`` eligibility-decay multiplies
        (one canonical multiply each). Ascending GID order.
        """
        with self._operation_lock:
            self._require_open()
            count = ctypes.c_size_t()
            status = int(
                self._library.humanbrain_procedural_runtime_dormant_snapshot(
                    self._handle,
                    None,
                    0,
                    ctypes.byref(count),
                )
            )
            if status != 0:
                raise ProceduralRuntimeError(status)
            if not count.value:
                return ()
            records = (_NativeDormant * count.value)()
            status = int(
                self._library.humanbrain_procedural_runtime_dormant_snapshot(
                    self._handle,
                    records,
                    len(records),
                    ctypes.byref(count),
                )
            )
            if status != 0:
                raise ProceduralRuntimeError(status)
            return tuple(
                (int(record.gid), float(record.eligibility_raw), int(record.elig_step))
                for record in records
            )

    @property
    def dormant_count(self) -> int:
        with self._operation_lock:
            self._require_open()
            count = ctypes.c_size_t()
            status = int(
                self._library.humanbrain_procedural_runtime_dormant_snapshot(
                    self._handle,
                    None,
                    0,
                    ctypes.byref(count),
                )
            )
            if status != 0:
                raise ProceduralRuntimeError(status)
            return int(count.value)

    @property
    def current_step(self) -> int:
        return self._current_step


class NativeDenseProceduralReference:
    """Small dense/native oracle with caller-materialized uint64 edge rows."""

    def __init__(
        self,
        config: ProceduralRuntimeConfig,
        explicit_targets: np.ndarray,
        explicit_weights_pa: np.ndarray,
    ) -> None:
        if not isinstance(config, ProceduralRuntimeConfig):
            raise TypeError("config must be a ProceduralRuntimeConfig")
        self._operation_lock = threading.RLock()
        self._handle: int | None = None
        cells = config.manifest.cell_count
        if cells > _DENSE_REFERENCE_MAX_CELLS:
            raise ValueError(
                f"dense reference is limited to {_DENSE_REFERENCE_MAX_CELLS} cells"
            )
        shape = (cells, config.manifest.fanout)
        if (
            not isinstance(explicit_targets, np.ndarray)
            or explicit_targets.dtype != np.uint64
            or explicit_targets.shape != shape
            or not explicit_targets.flags.c_contiguous
        ):
            raise ValueError("explicit_targets must be a contiguous uint64 edge matrix")
        if (
            not isinstance(explicit_weights_pa, np.ndarray)
            or explicit_weights_pa.dtype != np.float64
            or explicit_weights_pa.shape != shape
            or not explicit_weights_pa.flags.c_contiguous
            or not np.all(np.isfinite(explicit_weights_pa))
        ):
            raise ValueError("explicit_weights_pa must be a finite contiguous float64 matrix")
        if np.any(explicit_targets >= cells):
            raise ValueError("explicit target is outside the configured address space")
        self.config = config
        self._library = _procedural_library()
        (
            self._glif,
            self._after_decay,
            self._after_cut_decay,
            self._after_amplitudes,
        ) = _native_parameters()
        self._native_config = _native_runtime_config(config)
        self._targets = explicit_targets
        self._weights = explicit_weights_pa
        self._handle = self._library.humanbrain_dense_reference_create(
            ctypes.byref(self._glif),
            ctypes.byref(self._native_config),
            self._after_decay.ctypes.data_as(_F64_POINTER),
            self._after_cut_decay.ctypes.data_as(_F64_POINTER),
            self._after_amplitudes.ctypes.data_as(_F64_POINTER),
            self._targets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            self._weights.ctypes.data_as(_F64_POINTER),
        )
        if not self._handle:
            raise ProceduralRuntimeError(1)
        self._current_step = 0

    @classmethod
    def materialized(cls, config: ProceduralRuntimeConfig) -> NativeDenseProceduralReference:
        """Materialize independent Python-oracle rows for a small dense check."""
        oracle = ProceduralTopology(config.manifest)
        cells = config.manifest.cell_count
        fanout = config.manifest.fanout
        targets = np.empty((cells, fanout), dtype=np.uint64)
        weights = np.empty((cells, fanout), dtype=np.float64)
        for source in range(cells):
            targets[source] = tuple(
                edge.target_gid for edge in oracle.base_edges(source)
            )
            sign = (
                -1.0
                if oracle.describe_cell(source).outgoing_connectivity_role
                == "inhibitory"
                else 1.0
            )
            weights[source].fill(sign * config.recurrent_weight_pa)
        return cls(config, targets, weights)

    def __enter__(self) -> NativeDenseProceduralReference:
        with self._operation_lock:
            self._require_open()
            return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_open(self) -> None:
        if not self._handle:
            raise RuntimeError("dense procedural reference is closed")

    def close(self) -> None:
        with self._operation_lock:
            handle = self._handle
            if handle:
                self._handle = None
                self._library.humanbrain_dense_reference_destroy(handle)

    def advance(
        self,
        step_count: int,
        events: Sequence[ProceduralInputEvent] | np.ndarray = (),
    ) -> tuple[ProceduralSpike, ...]:
        with self._operation_lock:
            return self._advance_locked(step_count, events)

    def _advance_locked(
        self,
        step_count: int,
        events: Sequence[ProceduralInputEvent] | np.ndarray,
    ) -> tuple[ProceduralSpike, ...]:
        self._require_open()
        steps = _exact_uint64("step_count", step_count)
        if steps == 0:
            raise ValueError("step_count must be at least one")
        if self._current_step + steps > _UINT64_MAX:
            raise ValueError("advance would overflow the uint64 biological-step counter")
        if isinstance(events, np.ndarray):
            bulk_buffer, native_events, event_count = _event_batch_array(
                self.config, self._current_step, events
            )
        else:
            bulk_buffer = None
            native_events = _event_batch(self.config, self._current_step, events)
            event_count = 0 if native_events is None else len(native_events)
        status = int(
            self._library.humanbrain_dense_reference_advance(
                self._handle,
                steps,
                native_events,
                event_count,
            )
        )
        if status != 0:
            raise ProceduralRuntimeError(status)
        self._current_step += steps
        return self.snapshot().spikes

    def snapshot(self) -> ProceduralSnapshot:
        with self._operation_lock:
            self._require_open()
            cell_count = ctypes.c_size_t()
            pending_count = ctypes.c_size_t()
            spike_count = ctypes.c_size_t()
            status = int(
                self._library.humanbrain_dense_reference_counts(
                    self._handle,
                    ctypes.byref(cell_count),
                    ctypes.byref(pending_count),
                    ctypes.byref(spike_count),
                )
            )
            if status != 0:
                raise ProceduralRuntimeError(status)
            cells = (_NativeResident * cell_count.value)()
            pending = (
                (_NativeEvent * pending_count.value)() if pending_count.value else None
            )
            spikes = (_NativeSpike * spike_count.value)() if spike_count.value else None
            current_step = ctypes.c_uint64()
            poison = ctypes.c_uint32()
            status = int(
                self._library.humanbrain_dense_reference_snapshot(
                    self._handle,
                    cells,
                    len(cells),
                    pending,
                    0 if pending is None else len(pending),
                    spikes,
                    0 if spikes is None else len(spikes),
                    ctypes.byref(current_step),
                    ctypes.byref(poison),
                )
            )
            if status != 0:
                raise ProceduralRuntimeError(status)
            overflow = int(poison.value)
            return ProceduralSnapshot(
                current_step=int(current_step.value),
                residents=tuple(_resident(record) for record in cells),
                pending_events=(
                    tuple(_event(record) for record in pending)
                    if pending is not None
                    else ()
                ),
                spikes=(
                    tuple(_spike(record) for record in spikes)
                    if spikes is not None
                    else ()
                ),
                poisoned=overflow != 0,
                overflow_reason=_STATUS_NAMES.get(overflow) if overflow else None,
                native_payload_bytes=0,
            )


__all__ = [
    "NativeDenseProceduralReference",
    "NativeProceduralRuntime",
    "PROCEDURAL_EVENT_DTYPE",
    "ProceduralAdvance",
    "ProceduralInputEvent",
    "ProceduralResidentState",
    "ProceduralRuntimeConfig",
    "ProceduralRuntimeError",
    "ProceduralSnapshot",
    "ProceduralSpike",
    "procedural_input_events",
]
