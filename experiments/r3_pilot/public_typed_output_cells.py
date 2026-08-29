"""Factorized physical output cells for typed execution observations.

This substrate represents execution outputs only.  It deliberately contains no
AST node, edit-site, production, or parent/child structural state.  Scalar
values use the established thermometer/opponent exact-cell pattern; lists use
one length bank and four independent item banks rather than a cell per whole
list.  A materialized target projection drives one fitted GLIF coincidence
cell only when every active typed factor matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation


class OutputType(Enum):
    INT = "Int"
    BOOL = "Bool"
    LIST_INT = "List[Int]"


TYPE_LABELS = tuple(output_type.value for output_type in OutputType)
TYPE_CELL_COUNT = len(TYPE_LABELS)

INT_MIN = -20
INT_MAX = 20
INT_KNOWN_CELL_COUNT = INT_MAX - INT_MIN + 1
INT_UNKNOWN_CELL = INT_KNOWN_CELL_COUNT
INT_CELL_COUNT = INT_KNOWN_CELL_COUNT + 1
INT_THRESHOLDS = tuple(range(INT_MIN + 1, INT_MAX + 1))

BOOL_FALSE_CELL = 0
BOOL_TRUE_CELL = 1
BOOL_UNKNOWN_CELL = 2
BOOL_CELL_COUNT = 3

LIST_CAPACITY = 4
LIST_LENGTH_UNKNOWN_CELL = LIST_CAPACITY + 1
LIST_LENGTH_CELL_COUNT = LIST_CAPACITY + 2

TYPE_OFFSET = 0
INT_OFFSET = TYPE_OFFSET + TYPE_CELL_COUNT
BOOL_OFFSET = INT_OFFSET + INT_CELL_COUNT
LIST_LENGTH_OFFSET = BOOL_OFFSET + BOOL_CELL_COUNT
LIST_ITEM_OFFSET = LIST_LENGTH_OFFSET + LIST_LENGTH_CELL_COUNT
TOTAL_CELL_COUNT = LIST_ITEM_OFFSET + LIST_CAPACITY * INT_CELL_COUNT

TYPE_SLICE = slice(TYPE_OFFSET, INT_OFFSET)
INT_SLICE = slice(INT_OFFSET, BOOL_OFFSET)
BOOL_SLICE = slice(BOOL_OFFSET, LIST_LENGTH_OFFSET)
LIST_LENGTH_SLICE = slice(LIST_LENGTH_OFFSET, LIST_ITEM_OFFSET)
LIST_ITEMS_SLICE = slice(LIST_ITEM_OFFSET, TOTAL_CELL_COUNT)

FACTOR_LABELS = (
    "type",
    "int",
    "bool",
    "list-length",
    *(f"list-item-{position}" for position in range(LIST_CAPACITY)),
)
FACTOR_COUNT = len(FACTOR_LABELS)

_INPUT_CURRENT_PA = 5_000.0
_INPUT_STEPS = 30
_PROJECTION_SCALE_PA = 10_000.0
_PROJECTION_LIMIT_PA = 5_000.0
_COINCIDENCE_SCALE_PA = 10_000.0
_COINCIDENCE_LIMIT_PA = 5_000.0


def _freeze(array: np.ndarray, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    frozen = np.ascontiguousarray(array, dtype=dtype)
    frozen.setflags(write=False)
    return frozen


def _drive_spike_mask(currents_pa: np.ndarray, *, steps: int = _INPUT_STEPS) -> np.ndarray:
    currents = np.ascontiguousarray(currents_pa, dtype=np.float64)
    population = AdaptiveGlifPopulation(len(currents))
    seen = np.zeros(population.cell_count, dtype=bool)
    for _ in range(steps):
        fired = np.flatnonzero(population.step(currents))
        if fired.size:
            seen[fired] = True
    return _freeze(seen.astype(np.float64))


def _physical_one_hot(cell_count: int, active_cell: int) -> np.ndarray:
    if not 0 <= active_cell < cell_count:
        raise ValueError("active cell lies outside its physical bank")
    currents = np.zeros(cell_count, dtype=np.float64)
    currents[active_cell] = _INPUT_CURRENT_PA
    spikes = _drive_spike_mask(currents)
    if np.count_nonzero(spikes) != 1 or not spikes[active_cell]:
        raise RuntimeError("categorical output bank did not produce one physical cell")
    return spikes


def _int_projection() -> tuple[np.ndarray, np.ndarray]:
    weights = np.zeros(
        (INT_KNOWN_CELL_COUNT, len(INT_THRESHOLDS)), dtype=np.float64
    )
    bias = np.zeros(INT_KNOWN_CELL_COUNT, dtype=np.float64)
    weights[0, 0] = -1.0
    bias[0] = 1.0
    for row in range(1, INT_KNOWN_CELL_COUNT - 1):
        weights[row, row - 1] = 1.0
        weights[row, row] = -1.0
    weights[-1, -1] = 1.0
    return _freeze(weights), _freeze(bias)


def _encode_int_bank(value: object) -> tuple[np.ndarray, bool]:
    # Exact type checks are intentional: Python bool is an int subclass.
    if type(value) is not int or not INT_MIN <= value <= INT_MAX:
        return _physical_one_hot(INT_CELL_COUNT, INT_UNKNOWN_CELL), False

    thermometer_currents = np.asarray(
        [
            _INPUT_CURRENT_PA if value >= threshold else 0.0
            for threshold in INT_THRESHOLDS
        ],
        dtype=np.float64,
    )
    thermometer = _drive_spike_mask(thermometer_currents)
    weights, bias = _int_projection()
    net = weights @ thermometer + bias
    exact_currents = np.ascontiguousarray(
        np.clip(
            net * _PROJECTION_SCALE_PA,
            -_PROJECTION_LIMIT_PA,
            _PROJECTION_LIMIT_PA,
        )
    )
    known_spikes = _drive_spike_mask(exact_currents)
    spikes = _freeze(np.concatenate((known_spikes, np.zeros(1, dtype=np.float64))))
    if np.count_nonzero(spikes) != 1:
        raise RuntimeError("integer opponent projection did not produce one exact cell")
    return spikes, True


def _coerce_output_type(value: OutputType | str) -> OutputType:
    if isinstance(value, OutputType):
        return value
    if type(value) is str:
        try:
            return OutputType(value)
        except ValueError as error:
            raise ValueError(f"unsupported declared output type: {value!r}") from error
    raise TypeError("declared_type must be an OutputType or its exact label")


def _inferred_output_type(value: object) -> OutputType | None:
    # bool must be normalized before int because bool subclasses int in Python.
    if type(value) is bool:
        return OutputType.BOOL
    if type(value) is int:
        return OutputType.INT
    if type(value) is list:
        return OutputType.LIST_INT
    return None


@dataclass(frozen=True, slots=True)
class TypedOutputEncoding:
    """One factorized physical output observation."""

    output_type: OutputType | None
    cells: np.ndarray
    active_factor_mask: np.ndarray
    supported: bool
    sha256: str

    @property
    def type_cells(self) -> np.ndarray:
        return self.cells[TYPE_SLICE]

    @property
    def int_cells(self) -> np.ndarray:
        return self.cells[INT_SLICE]

    @property
    def bool_cells(self) -> np.ndarray:
        return self.cells[BOOL_SLICE]

    @property
    def list_length_cells(self) -> np.ndarray:
        return self.cells[LIST_LENGTH_SLICE]

    @property
    def list_item_cells(self) -> np.ndarray:
        return self.cells[LIST_ITEMS_SLICE].reshape(LIST_CAPACITY, INT_CELL_COUNT)

    @property
    def unknown_active(self) -> bool:
        return bool(
            self.int_cells[INT_UNKNOWN_CELL]
            or self.bool_cells[BOOL_UNKNOWN_CELL]
            or self.list_length_cells[LIST_LENGTH_UNKNOWN_CELL]
            or np.any(self.list_item_cells[:, INT_UNKNOWN_CELL])
        )

    @property
    def active_factors(self) -> tuple[str, ...]:
        return tuple(
            label
            for label, active in zip(
                FACTOR_LABELS, self.active_factor_mask, strict=True
            )
            if active
        )


class TypedOutputCellBank:
    """Encode Int, Bool, and bounded List[Int] outputs into fixed cell banks."""

    @staticmethod
    def projection_sha256() -> str:
        weights, bias = _int_projection()
        digest = sha256()
        digest.update(b"public-typed-output-cells-v1\0")
        digest.update("\n".join(TYPE_LABELS).encode("ascii"))
        digest.update("\n".join(FACTOR_LABELS).encode("ascii"))
        digest.update(
            np.asarray(
                (
                    INT_MIN,
                    INT_MAX,
                    INT_UNKNOWN_CELL,
                    BOOL_UNKNOWN_CELL,
                    LIST_CAPACITY,
                    LIST_LENGTH_UNKNOWN_CELL,
                    TOTAL_CELL_COUNT,
                ),
                dtype=np.int64,
            ).tobytes()
        )
        digest.update(weights.tobytes())
        digest.update(bias.tobytes())
        digest.update(
            np.asarray(
                (
                    _INPUT_CURRENT_PA,
                    _PROJECTION_SCALE_PA,
                    _PROJECTION_LIMIT_PA,
                    _COINCIDENCE_SCALE_PA,
                    _COINCIDENCE_LIMIT_PA,
                ),
                dtype=np.float64,
            ).tobytes()
        )
        return digest.hexdigest()

    @classmethod
    def encode(
        cls,
        value: object,
        *,
        declared_type: OutputType | str | None = None,
    ) -> TypedOutputEncoding:
        output_type = (
            _inferred_output_type(value)
            if declared_type is None
            else _coerce_output_type(declared_type)
        )
        cells = np.zeros(TOTAL_CELL_COUNT, dtype=np.float64)
        active = np.zeros(FACTOR_COUNT, dtype=bool)

        if output_type is None:
            # An unsupported untyped value has no valid type tag.  Every value
            # family exposes its physical UNKNOWN cell, so it cannot silently
            # alias any supported output or match an UNKNOWN target.
            cells[INT_OFFSET + INT_UNKNOWN_CELL] = _physical_one_hot(
                INT_CELL_COUNT, INT_UNKNOWN_CELL
            )[INT_UNKNOWN_CELL]
            cells[BOOL_OFFSET + BOOL_UNKNOWN_CELL] = _physical_one_hot(
                BOOL_CELL_COUNT, BOOL_UNKNOWN_CELL
            )[BOOL_UNKNOWN_CELL]
            cells[LIST_LENGTH_OFFSET + LIST_LENGTH_UNKNOWN_CELL] = _physical_one_hot(
                LIST_LENGTH_CELL_COUNT, LIST_LENGTH_UNKNOWN_CELL
            )[LIST_LENGTH_UNKNOWN_CELL]
            for position in range(LIST_CAPACITY):
                offset = LIST_ITEM_OFFSET + position * INT_CELL_COUNT
                cells[offset + INT_UNKNOWN_CELL] = _physical_one_hot(
                    INT_CELL_COUNT, INT_UNKNOWN_CELL
                )[INT_UNKNOWN_CELL]
            active[1:] = True
            return cls._finish(None, cells, active, supported=False)

        type_index = tuple(OutputType).index(output_type)
        cells[TYPE_SLICE] = _physical_one_hot(TYPE_CELL_COUNT, type_index)
        active[0] = True

        if output_type is OutputType.INT:
            int_cells, supported = _encode_int_bank(value)
            cells[INT_SLICE] = int_cells
            active[1] = True
            return cls._finish(output_type, cells, active, supported=supported)

        if output_type is OutputType.BOOL:
            if type(value) is bool:
                bool_cell = BOOL_TRUE_CELL if value else BOOL_FALSE_CELL
                supported = True
            else:
                bool_cell = BOOL_UNKNOWN_CELL
                supported = False
            cells[BOOL_SLICE] = _physical_one_hot(BOOL_CELL_COUNT, bool_cell)
            active[2] = True
            return cls._finish(output_type, cells, active, supported=supported)

        if type(value) is not list or len(value) > LIST_CAPACITY:
            cells[LIST_LENGTH_SLICE] = _physical_one_hot(
                LIST_LENGTH_CELL_COUNT, LIST_LENGTH_UNKNOWN_CELL
            )
            active[3] = True
            return cls._finish(output_type, cells, active, supported=False)

        cells[LIST_LENGTH_SLICE] = _physical_one_hot(
            LIST_LENGTH_CELL_COUNT, len(value)
        )
        active[3] = True
        supported = True
        for position, item in enumerate(value):
            item_cells, item_supported = _encode_int_bank(item)
            offset = LIST_ITEM_OFFSET + position * INT_CELL_COUNT
            cells[offset : offset + INT_CELL_COUNT] = item_cells
            active[4 + position] = True
            supported = supported and item_supported
        return cls._finish(output_type, cells, active, supported=supported)

    @classmethod
    def _finish(
        cls,
        output_type: OutputType | None,
        cells: np.ndarray,
        active: np.ndarray,
        *,
        supported: bool,
    ) -> TypedOutputEncoding:
        frozen_cells = _freeze(cells)
        frozen_active = _freeze(active, dtype=np.bool_)
        digest = sha256()
        digest.update(cls.projection_sha256().encode("ascii"))
        digest.update(
            b"UNKNOWN-TYPE" if output_type is None else output_type.value.encode("ascii")
        )
        digest.update(frozen_cells.tobytes())
        digest.update(frozen_active.tobytes())
        digest.update(b"1" if supported else b"0")
        return TypedOutputEncoding(
            output_type=output_type,
            cells=frozen_cells,
            active_factor_mask=frozen_active,
            supported=supported,
            sha256=digest.hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class TypedCoincidenceResult:
    spiked: bool
    spike_mask: np.ndarray
    current_pa: float
    matched_factor_cells: int
    active_target_factors: int
    detector_sha256: str
    predicted_sha256: str


@dataclass(frozen=True, slots=True)
class FixedTypedCoincidenceDetector:
    """Materialized exact-target synapses into one fitted coincidence cell."""

    weights: np.ndarray
    bias: float
    active_target_factors: int
    enabled: bool
    target_sha256: str
    sha256: str

    @classmethod
    def for_target(
        cls,
        target: object | TypedOutputEncoding,
        *,
        declared_type: OutputType | str | None = None,
    ) -> FixedTypedCoincidenceDetector:
        if isinstance(target, TypedOutputEncoding):
            if declared_type is not None:
                raise ValueError("declared_type cannot accompany an encoded target")
            encoded = target
        else:
            encoded = TypedOutputCellBank.encode(target, declared_type=declared_type)

        weights = np.zeros(TOTAL_CELL_COUNT, dtype=np.float64)
        active_count = int(np.count_nonzero(encoded.active_factor_mask))
        enabled = encoded.supported
        if enabled:
            selected = np.flatnonzero(encoded.cells)
            if len(selected) != active_count:
                raise RuntimeError("each active target factor must expose one exact cell")
            weights[selected] = 1.0
            bias = -(active_count - 0.5)
        else:
            # UNKNOWN is never a target category.  A disabled target still runs
            # through the physical detector, but its fixed current is inhibitory.
            bias = -1.0

        frozen_weights = _freeze(weights)
        digest = sha256()
        digest.update(TypedOutputCellBank.projection_sha256().encode("ascii"))
        digest.update(encoded.sha256.encode("ascii"))
        digest.update(frozen_weights.tobytes())
        digest.update(np.asarray((bias,), dtype=np.float64).tobytes())
        digest.update(b"1" if enabled else b"0")
        return cls(
            weights=frozen_weights,
            bias=bias,
            active_target_factors=active_count,
            enabled=enabled,
            target_sha256=encoded.sha256,
            sha256=digest.hexdigest(),
        )

    def detect(
        self,
        predicted: object | TypedOutputEncoding,
        *,
        declared_type: OutputType | str | None = None,
    ) -> TypedCoincidenceResult:
        if isinstance(predicted, TypedOutputEncoding):
            if declared_type is not None:
                raise ValueError("declared_type cannot accompany an encoded prediction")
            encoded = predicted
        else:
            encoded = TypedOutputCellBank.encode(
                predicted, declared_type=declared_type
            )

        matched = int(round(float(self.weights @ encoded.cells)))
        net = float(self.weights @ encoded.cells + self.bias)
        current = float(
            np.clip(
                net * _COINCIDENCE_SCALE_PA,
                -_COINCIDENCE_LIMIT_PA,
                _COINCIDENCE_LIMIT_PA,
            )
        )
        spike_mask = _drive_spike_mask(np.asarray((current,), dtype=np.float64))
        if np.count_nonzero(spike_mask) > 1:
            raise RuntimeError("typed exact-coincidence detector produced multiple cells")
        return TypedCoincidenceResult(
            spiked=bool(spike_mask[0]),
            spike_mask=spike_mask,
            current_pa=current,
            matched_factor_cells=matched,
            active_target_factors=self.active_target_factors,
            detector_sha256=self.sha256,
            predicted_sha256=encoded.sha256,
        )


def exact_typed_output_coincidence(
    predicted: object,
    desired: object,
    *,
    declared_type: OutputType | str | None = None,
) -> TypedCoincidenceResult:
    """Build a fixed desired-output detector and race one prediction through it."""

    detector = FixedTypedCoincidenceDetector.for_target(
        desired, declared_type=declared_type
    )
    return detector.detect(predicted, declared_type=declared_type)


__all__ = [
    "BOOL_CELL_COUNT",
    "BOOL_UNKNOWN_CELL",
    "FACTOR_LABELS",
    "FixedTypedCoincidenceDetector",
    "INT_CELL_COUNT",
    "INT_MAX",
    "INT_MIN",
    "INT_UNKNOWN_CELL",
    "LIST_CAPACITY",
    "LIST_LENGTH_CELL_COUNT",
    "LIST_LENGTH_UNKNOWN_CELL",
    "OutputType",
    "TOTAL_CELL_COUNT",
    "TYPE_CELL_COUNT",
    "TypedCoincidenceResult",
    "TypedOutputCellBank",
    "TypedOutputEncoding",
    "exact_typed_output_coincidence",
]
