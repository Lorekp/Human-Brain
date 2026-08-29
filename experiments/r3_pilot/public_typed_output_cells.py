"""Compact factorized physical output cells for typed execution observations.

The physical projection remains the same 3,284 globally addressed cells, but a
retained observation stores only one local winner per factor plus the exact
virtual-dense scientific digest. Whole-projection arrays exist solely behind
the explicitly named audit materializer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from hashlib import sha256
from struct import Struct

import numpy as np

from humanbrain.resting_glif_race import CompiledRestingGlifRace


class OutputType(Enum):
    INT = "Int"
    BOOL = "Bool"
    LIST_INT = "List[Int]"
    CHAR = "Char"
    STRING = "String"


TYPE_LABELS = tuple(output_type.value for output_type in OutputType)
TYPE_CELL_COUNT = len(TYPE_LABELS)

INT_MIN = -64
INT_MAX = 64
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

PRINTABLE_ASCII_MIN = 0x20
PRINTABLE_ASCII_MAX = 0x7E
PRINTABLE_ASCII = tuple(
    chr(code) for code in range(PRINTABLE_ASCII_MIN, PRINTABLE_ASCII_MAX + 1)
)
CHAR_UNKNOWN_CELL = len(PRINTABLE_ASCII)
CHAR_CELL_COUNT = CHAR_UNKNOWN_CELL + 1

STRING_CAPACITY = 26
STRING_LENGTH_UNKNOWN_CELL = STRING_CAPACITY + 1
STRING_LENGTH_CELL_COUNT = STRING_CAPACITY + 2

TYPE_OFFSET = 0
INT_OFFSET = TYPE_OFFSET + TYPE_CELL_COUNT
BOOL_OFFSET = INT_OFFSET + INT_CELL_COUNT
LIST_LENGTH_OFFSET = BOOL_OFFSET + BOOL_CELL_COUNT
LIST_ITEM_OFFSET = LIST_LENGTH_OFFSET + LIST_LENGTH_CELL_COUNT
CHAR_OFFSET = LIST_ITEM_OFFSET + LIST_CAPACITY * INT_CELL_COUNT
STRING_LENGTH_OFFSET = CHAR_OFFSET + CHAR_CELL_COUNT
STRING_ITEM_OFFSET = STRING_LENGTH_OFFSET + STRING_LENGTH_CELL_COUNT
TOTAL_CELL_COUNT = STRING_ITEM_OFFSET + STRING_CAPACITY * CHAR_CELL_COUNT

TYPE_FACTOR_SLOT = 0
INT_FACTOR_SLOT = 1
BOOL_FACTOR_SLOT = 2
LIST_LENGTH_FACTOR_SLOT = 3
LIST_ITEM_FACTOR_START = 4
CHAR_FACTOR_SLOT = LIST_ITEM_FACTOR_START + LIST_CAPACITY
STRING_LENGTH_FACTOR_SLOT = CHAR_FACTOR_SLOT + 1
STRING_ITEM_FACTOR_START = STRING_LENGTH_FACTOR_SLOT + 1

FACTOR_LABELS = (
    "type",
    "int",
    "bool",
    "list-length",
    *(f"list-item-{position}" for position in range(LIST_CAPACITY)),
    "char",
    "string-length",
    *(f"string-char-{position}" for position in range(STRING_CAPACITY)),
)
FACTOR_COUNT = len(FACTOR_LABELS)

_FACTOR_OFFSETS = (
    TYPE_OFFSET,
    INT_OFFSET,
    BOOL_OFFSET,
    LIST_LENGTH_OFFSET,
    *(LIST_ITEM_OFFSET + position * INT_CELL_COUNT for position in range(LIST_CAPACITY)),
    CHAR_OFFSET,
    STRING_LENGTH_OFFSET,
    *(
        STRING_ITEM_OFFSET + position * CHAR_CELL_COUNT
        for position in range(STRING_CAPACITY)
    ),
)
_FACTOR_WIDTHS = (
    TYPE_CELL_COUNT,
    INT_CELL_COUNT,
    BOOL_CELL_COUNT,
    LIST_LENGTH_CELL_COUNT,
    *(INT_CELL_COUNT for _ in range(LIST_CAPACITY)),
    CHAR_CELL_COUNT,
    STRING_LENGTH_CELL_COUNT,
    *(CHAR_CELL_COUNT for _ in range(STRING_CAPACITY)),
)
_FACTOR_UNKNOWN_WINNERS: tuple[int | None, ...] = (
    None,
    INT_UNKNOWN_CELL,
    BOOL_UNKNOWN_CELL,
    LIST_LENGTH_UNKNOWN_CELL,
    *(INT_UNKNOWN_CELL for _ in range(LIST_CAPACITY)),
    CHAR_UNKNOWN_CELL,
    STRING_LENGTH_UNKNOWN_CELL,
    *(CHAR_UNKNOWN_CELL for _ in range(STRING_CAPACITY)),
)

_INPUT_CURRENT_PA = 5_000.0
_INPUT_STEPS = 30
_PROJECTION_SCALE_PA = 10_000.0
_PROJECTION_LIMIT_PA = 5_000.0
_COINCIDENCE_SCALE_PA = 10_000.0
_COINCIDENCE_LIMIT_PA = 5_000.0
_PHYSICAL_ONE_HOT_CACHE_SIZE = (
    TYPE_CELL_COUNT
    + 1  # The Int bank uses this path only for UNKNOWN.
    + BOOL_CELL_COUNT
    + LIST_LENGTH_CELL_COUNT
    + CHAR_CELL_COUNT
    + STRING_LENGTH_CELL_COUNT
)
_OUTPUT_SPIKE_KERNEL = CompiledRestingGlifRace.compile(step_count=_INPUT_STEPS)

_INACTIVE_FACTOR_WINNER = 0xFFFF
_FACTOR_WINNERS = Struct(f"<{FACTOR_COUNT}H")
_ZERO_HASH_CHUNK = bytes(4_096)
_FLOAT64_ONE_BYTES = np.asarray((1.0,), dtype=np.float64).tobytes()

if not (
    len(_FACTOR_OFFSETS)
    == len(_FACTOR_WIDTHS)
    == len(_FACTOR_UNKNOWN_WINNERS)
    == FACTOR_COUNT
):
    raise RuntimeError("typed output factor geometry is inconsistent")
if any(width > _INACTIVE_FACTOR_WINNER for width in _FACTOR_WIDTHS):
    raise RuntimeError("typed output factor bank width exceeds uint16 capacity")
if _FACTOR_WINNERS.size != 72:
    raise RuntimeError("typed output compact winner payload must occupy 72 bytes")


def _freeze(array: np.ndarray, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    frozen = np.ascontiguousarray(array, dtype=dtype)
    frozen.setflags(write=False)
    return frozen


def _drive_spike_mask(currents_pa: np.ndarray, *, steps: int = _INPUT_STEPS) -> np.ndarray:
    currents = np.ascontiguousarray(currents_pa, dtype=np.float64)
    kernel = (
        _OUTPUT_SPIKE_KERNEL
        if steps == _INPUT_STEPS
        else CompiledRestingGlifRace.compile(step_count=steps)
    )
    return _freeze(kernel.accumulated_spike_mask(currents).astype(np.float64))


@lru_cache(maxsize=_PHYSICAL_ONE_HOT_CACHE_SIZE)
def _physical_one_hot(cell_count: int, active_cell: int) -> np.ndarray:
    if not 0 <= active_cell < cell_count:
        raise ValueError("active cell lies outside its physical bank")
    currents = np.zeros(cell_count, dtype=np.float64)
    currents[active_cell] = _INPUT_CURRENT_PA
    spikes = _drive_spike_mask(currents)
    if np.count_nonzero(spikes) != 1 or not spikes[active_cell]:
        raise RuntimeError("categorical output bank did not produce one physical cell")
    return spikes


def _categorical_winner(cell_count: int, active_cell: int) -> int:
    _physical_one_hot(cell_count, active_cell)
    return active_cell


@lru_cache(maxsize=1)
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


def _encode_int_winner(value: object) -> tuple[int, bool]:
    # Exact type checks are intentional: Python bool is an int subclass.
    if type(value) is not int or not INT_MIN <= value <= INT_MAX:
        return _categorical_winner(INT_CELL_COUNT, INT_UNKNOWN_CELL), False

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
    selected = np.flatnonzero(known_spikes)
    if len(selected) != 1:
        raise RuntimeError("integer opponent projection did not produce one exact cell")
    return int(selected[0]), True


def _encode_char_winner(value: object) -> tuple[int, bool]:
    if (
        type(value) is not str
        or len(value) != 1
        or not PRINTABLE_ASCII_MIN <= ord(value) <= PRINTABLE_ASCII_MAX
    ):
        return _categorical_winner(CHAR_CELL_COUNT, CHAR_UNKNOWN_CELL), False
    winner = ord(value) - PRINTABLE_ASCII_MIN
    return _categorical_winner(CHAR_CELL_COUNT, winner), True


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
    if type(value) is bool:
        return OutputType.BOOL
    if type(value) is int:
        return OutputType.INT
    if type(value) is list:
        return OutputType.LIST_INT
    if type(value) is str:
        # Python has no distinct Char runtime type; Char is declaration-only.
        return OutputType.STRING
    return None


def _normalize_factor_winners(
    winners: Sequence[int | None],
) -> tuple[int | None, ...]:
    if isinstance(winners, (str, bytes)):
        raise TypeError("factor winners must be a sequence of int or None")
    if len(winners) != FACTOR_COUNT:
        raise ValueError(f"factor winners must contain exactly {FACTOR_COUNT} entries")
    normalized: list[int | None] = []
    for slot, (winner, width) in enumerate(
        zip(winners, _FACTOR_WIDTHS, strict=True)
    ):
        if winner is None:
            normalized.append(None)
            continue
        if type(winner) is not int:
            raise TypeError(f"factor winner {slot} must be an int or None")
        if not 0 <= winner < width:
            raise ValueError(f"factor winner {slot} lies outside its physical bank")
        normalized.append(winner)
    return tuple(normalized)


def _unpack_factor_winners(payload: bytes) -> tuple[int | None, ...]:
    if type(payload) is not bytes or len(payload) != _FACTOR_WINNERS.size:
        raise ValueError("factor winner payload must be exactly 72 immutable bytes")
    return tuple(
        None if winner == _INACTIVE_FACTOR_WINNER else winner
        for winner in _FACTOR_WINNERS.unpack(payload)
    )


_EXPECTED_TYPE_UNSET = object()


def _analyze_canonical_winners(
    winners: Sequence[int | None],
    *,
    expected_output_type: OutputType | None | object = _EXPECTED_TYPE_UNSET,
) -> tuple[OutputType | None, bool, bool]:
    normalized = _normalize_factor_winners(winners)
    type_winner = normalized[TYPE_FACTOR_SLOT]
    output_type = None if type_winner is None else tuple(OutputType)[type_winner]
    if (
        expected_output_type is not _EXPECTED_TYPE_UNSET
        and output_type is not expected_output_type
    ):
        raise ValueError("output type does not match its physical type winner")

    active_slots = {slot for slot, winner in enumerate(normalized) if winner is not None}
    if output_type is None:
        if active_slots != set(range(1, FACTOR_COUNT)) or any(
            normalized[slot] != _FACTOR_UNKNOWN_WINNERS[slot]
            for slot in range(1, FACTOR_COUNT)
        ):
            raise ValueError("untyped UNKNOWN factor shape is not canonical")
    elif output_type is OutputType.INT:
        if active_slots != {TYPE_FACTOR_SLOT, INT_FACTOR_SLOT}:
            raise ValueError("Int factor shape is not canonical")
    elif output_type is OutputType.BOOL:
        if active_slots != {TYPE_FACTOR_SLOT, BOOL_FACTOR_SLOT}:
            raise ValueError("Bool factor shape is not canonical")
    elif output_type is OutputType.CHAR:
        if active_slots != {TYPE_FACTOR_SLOT, CHAR_FACTOR_SLOT}:
            raise ValueError("Char factor shape is not canonical")
    elif output_type is OutputType.LIST_INT:
        length = normalized[LIST_LENGTH_FACTOR_SLOT]
        if length is None:
            raise ValueError("List[Int] length factor is missing")
        required = {TYPE_FACTOR_SLOT, LIST_LENGTH_FACTOR_SLOT}
        if length != LIST_LENGTH_UNKNOWN_CELL:
            required.update(range(LIST_ITEM_FACTOR_START, LIST_ITEM_FACTOR_START + length))
        if active_slots != required:
            raise ValueError("List[Int] factor shape is not canonical")
    elif output_type is OutputType.STRING:
        length = normalized[STRING_LENGTH_FACTOR_SLOT]
        if length is None:
            raise ValueError("String length factor is missing")
        required = {TYPE_FACTOR_SLOT, STRING_LENGTH_FACTOR_SLOT}
        if length != STRING_LENGTH_UNKNOWN_CELL:
            required.update(
                range(STRING_ITEM_FACTOR_START, STRING_ITEM_FACTOR_START + length)
            )
        if active_slots != required:
            raise ValueError("String factor shape is not canonical")
    else:  # pragma: no cover - exhaustive over the current enum.
        raise ValueError(f"unsupported output type: {output_type!r}")

    unknown_active = any(
        winner is not None
        and unknown is not None
        and winner == unknown
        for winner, unknown in zip(
            normalized, _FACTOR_UNKNOWN_WINNERS, strict=True
        )
    )
    supported = output_type is not None and not unknown_active
    return output_type, supported, unknown_active


def _active_global_indices_from_winners(
    winners: Sequence[int | None],
) -> tuple[int, ...]:
    return tuple(
        offset + winner
        for offset, winner in zip(_FACTOR_OFFSETS, winners, strict=True)
        if winner is not None
    )


def _update_zero_float64_cells(digest: object, cell_count: int) -> None:
    byte_count = cell_count * np.dtype(np.float64).itemsize
    full_chunks, remainder = divmod(byte_count, len(_ZERO_HASH_CHUNK))
    for _ in range(full_chunks):
        digest.update(_ZERO_HASH_CHUNK)  # type: ignore[attr-defined]
    if remainder:
        digest.update(memoryview(_ZERO_HASH_CHUNK)[:remainder])  # type: ignore[attr-defined]


def _update_virtual_dense_projection(
    digest: object,
    winners: Sequence[int | None],
    *,
    enabled: bool = True,
) -> None:
    if not enabled:
        _update_zero_float64_cells(digest, TOTAL_CELL_COUNT)
        return
    next_cell = 0
    for cell in _active_global_indices_from_winners(winners):
        _update_zero_float64_cells(digest, cell - next_cell)
        digest.update(_FLOAT64_ONE_BYTES)  # type: ignore[attr-defined]
        next_cell = cell + 1
    _update_zero_float64_cells(digest, TOTAL_CELL_COUNT - next_cell)


@lru_cache(maxsize=1)
def _projection_sha256() -> str:
    weights, bias = _int_projection()
    digest = sha256()
    digest.update(b"public-typed-output-cells-v2\0")
    digest.update("\n".join(TYPE_LABELS).encode("ascii"))
    digest.update("\n".join(FACTOR_LABELS).encode("ascii"))
    digest.update(bytes(range(PRINTABLE_ASCII_MIN, PRINTABLE_ASCII_MAX + 1)))
    digest.update(
        np.asarray(
            (
                TYPE_OFFSET,
                TYPE_CELL_COUNT,
                INT_OFFSET,
                INT_MIN,
                INT_MAX,
                INT_UNKNOWN_CELL,
                INT_CELL_COUNT,
                BOOL_OFFSET,
                BOOL_FALSE_CELL,
                BOOL_TRUE_CELL,
                BOOL_UNKNOWN_CELL,
                BOOL_CELL_COUNT,
                LIST_LENGTH_OFFSET,
                LIST_CAPACITY,
                LIST_LENGTH_UNKNOWN_CELL,
                LIST_LENGTH_CELL_COUNT,
                LIST_ITEM_OFFSET,
                CHAR_OFFSET,
                CHAR_UNKNOWN_CELL,
                CHAR_CELL_COUNT,
                STRING_LENGTH_OFFSET,
                STRING_CAPACITY,
                STRING_LENGTH_UNKNOWN_CELL,
                STRING_LENGTH_CELL_COUNT,
                STRING_ITEM_OFFSET,
                TOTAL_CELL_COUNT,
                FACTOR_COUNT,
                _INPUT_STEPS,
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


def _encoding_physical_digest(
    output_type: OutputType | None,
    winners: Sequence[int | None],
    *,
    supported: bool,
) -> bytes:
    digest = sha256()
    digest.update(_projection_sha256().encode("ascii"))
    digest.update(
        b"UNKNOWN-TYPE" if output_type is None else output_type.value.encode("ascii")
    )
    _update_virtual_dense_projection(digest, winners)
    digest.update(bytes(winner is not None for winner in winners))
    digest.update(b"1" if supported else b"0")
    return digest.digest()


@dataclass(frozen=True, slots=True, init=False)
class TypedOutputEncoding:
    """Immutable compact observation owned by :class:`TypedOutputCellBank`."""

    factor_winners_le: bytes
    physical_digest: bytes

    def __new__(cls) -> TypedOutputEncoding:
        raise TypeError("TypedOutputEncoding construction is bank-owned")

    @property
    def output_type(self) -> OutputType | None:
        output_type, _, _ = _analyze_canonical_winners(
            _unpack_factor_winners(self.factor_winners_le)
        )
        return output_type

    @property
    def supported(self) -> bool:
        _, supported, _ = _analyze_canonical_winners(
            _unpack_factor_winners(self.factor_winners_le)
        )
        return supported

    @property
    def unknown_active(self) -> bool:
        _, _, unknown_active = _analyze_canonical_winners(
            _unpack_factor_winners(self.factor_winners_le)
        )
        return unknown_active

    @property
    def active_factors(self) -> tuple[str, ...]:
        winners = _unpack_factor_winners(self.factor_winners_le)
        _analyze_canonical_winners(winners)
        return tuple(
            label
            for label, winner in zip(FACTOR_LABELS, winners, strict=True)
            if winner is not None
        )

    @property
    def sha256(self) -> str:
        return self.physical_digest.hex()

    def factor_winner(self, factor_slot: int) -> int | None:
        if type(factor_slot) is not int:
            raise TypeError("factor_slot must be an int")
        if not 0 <= factor_slot < FACTOR_COUNT:
            raise IndexError("factor_slot lies outside the typed output projection")
        return _unpack_factor_winners(self.factor_winners_le)[factor_slot]

    def active_global_cell_indices(self) -> tuple[int, ...]:
        winners = _unpack_factor_winners(self.factor_winners_le)
        _analyze_canonical_winners(winners)
        return _active_global_indices_from_winners(winners)


def _new_encoding(
    winners: Sequence[int | None],
    *,
    expected_output_type: OutputType | None,
) -> TypedOutputEncoding:
    normalized = _normalize_factor_winners(winners)
    output_type, supported, _ = _analyze_canonical_winners(
        normalized, expected_output_type=expected_output_type
    )
    payload = _FACTOR_WINNERS.pack(
        *(
            _INACTIVE_FACTOR_WINNER if winner is None else winner
            for winner in normalized
        )
    )
    physical_digest = _encoding_physical_digest(
        output_type, normalized, supported=supported
    )
    encoded = object.__new__(TypedOutputEncoding)
    object.__setattr__(encoded, "factor_winners_le", payload)
    object.__setattr__(encoded, "physical_digest", physical_digest)
    return encoded


@dataclass(frozen=True, slots=True)
class DenseTypedOutputAudit:
    """Explicit immutable whole-projection materialization for parity audits."""

    cells: np.ndarray
    active_factor_mask: np.ndarray


class TypedOutputCellBank:
    """Encode, validate, and decode bounded values in fixed physical banks."""

    @staticmethod
    def projection_sha256() -> str:
        return _projection_sha256()

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
        winners: list[int | None] = [None] * FACTOR_COUNT

        if output_type is None:
            for slot in range(1, FACTOR_COUNT):
                unknown = _FACTOR_UNKNOWN_WINNERS[slot]
                if unknown is None:  # pragma: no cover - geometry invariant.
                    raise RuntimeError("non-type factor has no UNKNOWN winner")
                winners[slot] = _categorical_winner(_FACTOR_WIDTHS[slot], unknown)
            return _new_encoding(winners, expected_output_type=None)

        type_index = tuple(OutputType).index(output_type)
        winners[TYPE_FACTOR_SLOT] = _categorical_winner(TYPE_CELL_COUNT, type_index)

        if output_type is OutputType.INT:
            winner, _ = _encode_int_winner(value)
            winners[INT_FACTOR_SLOT] = winner
            return _new_encoding(winners, expected_output_type=output_type)

        if output_type is OutputType.BOOL:
            if type(value) is bool:
                bool_cell = BOOL_TRUE_CELL if value else BOOL_FALSE_CELL
            else:
                bool_cell = BOOL_UNKNOWN_CELL
            winners[BOOL_FACTOR_SLOT] = _categorical_winner(
                BOOL_CELL_COUNT, bool_cell
            )
            return _new_encoding(winners, expected_output_type=output_type)

        if output_type is OutputType.CHAR:
            winner, _ = _encode_char_winner(value)
            winners[CHAR_FACTOR_SLOT] = winner
            return _new_encoding(winners, expected_output_type=output_type)

        if output_type is OutputType.STRING:
            if type(value) is not str or len(value) > STRING_CAPACITY:
                winners[STRING_LENGTH_FACTOR_SLOT] = _categorical_winner(
                    STRING_LENGTH_CELL_COUNT, STRING_LENGTH_UNKNOWN_CELL
                )
                return _new_encoding(winners, expected_output_type=output_type)

            winners[STRING_LENGTH_FACTOR_SLOT] = _categorical_winner(
                STRING_LENGTH_CELL_COUNT, len(value)
            )
            for position, item in enumerate(value):
                winner, _ = _encode_char_winner(item)
                winners[STRING_ITEM_FACTOR_START + position] = winner
            return _new_encoding(winners, expected_output_type=output_type)

        if type(value) is not list or len(value) > LIST_CAPACITY:
            winners[LIST_LENGTH_FACTOR_SLOT] = _categorical_winner(
                LIST_LENGTH_CELL_COUNT, LIST_LENGTH_UNKNOWN_CELL
            )
            return _new_encoding(winners, expected_output_type=output_type)

        winners[LIST_LENGTH_FACTOR_SLOT] = _categorical_winner(
            LIST_LENGTH_CELL_COUNT, len(value)
        )
        for position, item in enumerate(value):
            winner, _ = _encode_int_winner(item)
            winners[LIST_ITEM_FACTOR_START + position] = winner
        return _new_encoding(winners, expected_output_type=output_type)

    @classmethod
    def validate(cls, encoded: TypedOutputEncoding) -> TypedOutputEncoding:
        if type(encoded) is not TypedOutputEncoding:
            raise TypeError("encoded must be a TypedOutputEncoding")
        if type(encoded.factor_winners_le) is not bytes:
            raise ValueError("factor winner payload must be immutable bytes")
        if len(encoded.factor_winners_le) != _FACTOR_WINNERS.size:
            raise ValueError("factor winner payload must be exactly 72 bytes")
        if type(encoded.physical_digest) is not bytes:
            raise ValueError("physical digest must be immutable bytes")
        if len(encoded.physical_digest) != sha256().digest_size:
            raise ValueError("physical digest must be exactly 32 bytes")

        winners = _unpack_factor_winners(encoded.factor_winners_le)
        output_type, supported, _ = _analyze_canonical_winners(winners)
        expected_digest = _encoding_physical_digest(
            output_type, winners, supported=supported
        )
        if encoded.physical_digest != expected_digest:
            raise ValueError("physical digest does not match the canonical encoding")
        return encoded

    @classmethod
    def decode(cls, encoded: TypedOutputEncoding) -> object:
        """Recover a value only from a validated canonical encoding."""

        return cls._decode_trusted(cls.validate(encoded))

    @staticmethod
    def _decode_trusted(encoded: TypedOutputEncoding) -> object:
        winners = _unpack_factor_winners(encoded.factor_winners_le)
        output_type, supported, _ = _analyze_canonical_winners(winners)
        if not supported or output_type is None:
            raise ValueError("UNKNOWN output encodings cannot be decoded")

        if output_type is OutputType.INT:
            winner = winners[INT_FACTOR_SLOT]
            assert winner is not None
            return INT_MIN + winner
        if output_type is OutputType.BOOL:
            winner = winners[BOOL_FACTOR_SLOT]
            assert winner is not None
            return winner == BOOL_TRUE_CELL
        if output_type is OutputType.LIST_INT:
            length = winners[LIST_LENGTH_FACTOR_SLOT]
            assert length is not None
            return [
                INT_MIN + int(winners[LIST_ITEM_FACTOR_START + position])
                for position in range(length)
            ]
        if output_type is OutputType.CHAR:
            winner = winners[CHAR_FACTOR_SLOT]
            assert winner is not None
            return PRINTABLE_ASCII[winner]
        if output_type is OutputType.STRING:
            length = winners[STRING_LENGTH_FACTOR_SLOT]
            assert length is not None
            return "".join(
                PRINTABLE_ASCII[int(winners[STRING_ITEM_FACTOR_START + position])]
                for position in range(length)
            )
        raise ValueError(f"unsupported output type: {output_type!r}")

    @classmethod
    def _from_physical_winners(
        cls,
        output_type: OutputType | None,
        winners: Sequence[int | None],
    ) -> TypedOutputEncoding:
        if output_type is not None and type(output_type) is not OutputType:
            raise TypeError("output_type must be an OutputType or None")
        return _new_encoding(winners, expected_output_type=output_type)

    @staticmethod
    def _drive_factor(
        encoded: TypedOutputEncoding,
        factor_slot: int,
        current_pa: float,
    ) -> np.ndarray:
        winner = encoded.factor_winner(factor_slot)
        currents = np.zeros(_FACTOR_WIDTHS[factor_slot], dtype=np.float64)
        if winner is not None:
            currents[winner] = current_pa
        return np.ascontiguousarray(currents)

    @classmethod
    def materialize_for_audit(
        cls, encoded: TypedOutputEncoding
    ) -> DenseTypedOutputAudit:
        validated = cls.validate(encoded)
        winners = _unpack_factor_winners(validated.factor_winners_le)
        cells = np.zeros(TOTAL_CELL_COUNT, dtype=np.float64)
        active_indices = _active_global_indices_from_winners(winners)
        if active_indices:
            cells[np.asarray(active_indices, dtype=np.intp)] = 1.0
        active = np.asarray(
            tuple(winner is not None for winner in winners), dtype=np.bool_
        )
        return DenseTypedOutputAudit(
            cells=_freeze(cells),
            active_factor_mask=_freeze(active, dtype=np.bool_),
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


def _detector_bias(target: TypedOutputEncoding) -> float:
    if not target.supported:
        return -1.0
    return -(len(target.active_factors) - 0.5)


def _detector_physical_digest(target: TypedOutputEncoding) -> bytes:
    winners = _unpack_factor_winners(target.factor_winners_le)
    enabled = target.supported
    bias = _detector_bias(target)
    digest = sha256()
    digest.update(_projection_sha256().encode("ascii"))
    digest.update(target.sha256.encode("ascii"))
    _update_virtual_dense_projection(digest, winners, enabled=enabled)
    digest.update(np.asarray((bias,), dtype=np.float64).tobytes())
    digest.update(b"1" if enabled else b"0")
    return digest.digest()


@dataclass(frozen=True, slots=True, init=False)
class FixedTypedCoincidenceDetector:
    """Canonical exact-target synapses represented by one compact target."""

    target: TypedOutputEncoding
    physical_digest: bytes

    def __new__(cls) -> FixedTypedCoincidenceDetector:
        raise TypeError("FixedTypedCoincidenceDetector construction is bank-owned")

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
            encoded = TypedOutputCellBank.validate(target)
        else:
            encoded = TypedOutputCellBank.encode(target, declared_type=declared_type)

        detector = object.__new__(cls)
        object.__setattr__(detector, "target", encoded)
        object.__setattr__(
            detector, "physical_digest", _detector_physical_digest(encoded)
        )
        return detector

    @property
    def bias(self) -> float:
        return _detector_bias(self.target)

    @property
    def active_target_factors(self) -> int:
        return len(self.target.active_factors)

    @property
    def enabled(self) -> bool:
        return self.target.supported

    @property
    def target_sha256(self) -> str:
        return self.target.sha256

    @property
    def sha256(self) -> str:
        return self.physical_digest.hex()

    def _validate(self) -> None:
        if type(self.target) is not TypedOutputEncoding:
            raise TypeError("detector target must be a TypedOutputEncoding")
        TypedOutputCellBank.validate(self.target)
        if type(self.physical_digest) is not bytes:
            raise ValueError("detector physical digest must be immutable bytes")
        if len(self.physical_digest) != sha256().digest_size:
            raise ValueError("detector physical digest must be exactly 32 bytes")
        if self.physical_digest != _detector_physical_digest(self.target):
            raise ValueError("detector physical digest is not canonical")

    def detect(
        self,
        predicted: object | TypedOutputEncoding,
        *,
        declared_type: OutputType | str | None = None,
    ) -> TypedCoincidenceResult:
        return self.detect_many((predicted,), declared_type=declared_type)[0]

    def detect_many(
        self,
        predicted: Sequence[object | TypedOutputEncoding],
        *,
        declared_type: OutputType | str | None = None,
    ) -> tuple[TypedCoincidenceResult, ...]:
        """Validate external rows, then drive one fitted-GLIF population."""

        self._validate()
        if isinstance(predicted, (str, bytes)) or not isinstance(predicted, Sequence):
            raise TypeError("predicted must be a sequence of typed outputs")
        if len(predicted) == 0:
            return ()

        encoded_rows: list[TypedOutputEncoding] = []
        for item in predicted:
            if isinstance(item, TypedOutputEncoding):
                if declared_type is not None:
                    raise ValueError(
                        "declared_type cannot accompany an encoded prediction"
                    )
                encoded_rows.append(TypedOutputCellBank.validate(item))
            else:
                encoded_rows.append(
                    TypedOutputCellBank.encode(item, declared_type=declared_type)
                )
        return self._detect_many_trusted(tuple(encoded_rows))

    def _detect_many_trusted(
        self,
        predicted: Sequence[TypedOutputEncoding],
    ) -> tuple[TypedCoincidenceResult, ...]:
        """Drive already-validated compact rows without repeated digest work."""

        if isinstance(predicted, (str, bytes)) or not isinstance(predicted, Sequence):
            raise TypeError("predicted must be a sequence of typed encodings")
        if len(predicted) == 0:
            return ()
        if not all(type(item) is TypedOutputEncoding for item in predicted):
            raise TypeError("trusted predictions must be TypedOutputEncoding values")

        row_count = len(predicted)
        if self.enabled:
            target_words = np.frombuffer(self.target.factor_winners_le, dtype="<u2")
            active_slots = target_words != _INACTIVE_FACTOR_WINNER
            packed_rows = b"".join(item.factor_winners_le for item in predicted)
            row_words = np.frombuffer(packed_rows, dtype="<u2").reshape(
                row_count, FACTOR_COUNT
            )
            matched = np.count_nonzero(
                row_words[:, active_slots] == target_words[active_slots], axis=1
            ).astype(np.float64, copy=False)
        else:
            matched = np.zeros(row_count, dtype=np.float64)

        currents = np.ascontiguousarray(
            np.clip(
                (matched + self.bias) * _COINCIDENCE_SCALE_PA,
                -_COINCIDENCE_LIMIT_PA,
                _COINCIDENCE_LIMIT_PA,
            )
        )
        batch_spike_mask = _drive_spike_mask(currents)

        results: list[TypedCoincidenceResult] = []
        detector_sha256 = self.sha256
        active_target_factors = self.active_target_factors
        for index, encoded in enumerate(predicted):
            spike_mask = _freeze(
                np.asarray((batch_spike_mask[index],), dtype=np.float64)
            )
            results.append(
                TypedCoincidenceResult(
                    spiked=bool(spike_mask[0]),
                    spike_mask=spike_mask,
                    current_pa=float(currents[index]),
                    matched_factor_cells=int(matched[index]),
                    active_target_factors=active_target_factors,
                    detector_sha256=detector_sha256,
                    predicted_sha256=encoded.sha256,
                )
            )
        return tuple(results)


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
    "BOOL_FACTOR_SLOT",
    "BOOL_UNKNOWN_CELL",
    "CHAR_CELL_COUNT",
    "CHAR_FACTOR_SLOT",
    "CHAR_UNKNOWN_CELL",
    "DenseTypedOutputAudit",
    "FACTOR_COUNT",
    "FACTOR_LABELS",
    "FixedTypedCoincidenceDetector",
    "INT_CELL_COUNT",
    "INT_FACTOR_SLOT",
    "INT_MAX",
    "INT_MIN",
    "INT_UNKNOWN_CELL",
    "LIST_CAPACITY",
    "LIST_ITEM_FACTOR_START",
    "LIST_LENGTH_CELL_COUNT",
    "LIST_LENGTH_FACTOR_SLOT",
    "LIST_LENGTH_UNKNOWN_CELL",
    "OutputType",
    "PRINTABLE_ASCII",
    "PRINTABLE_ASCII_MAX",
    "PRINTABLE_ASCII_MIN",
    "STRING_CAPACITY",
    "STRING_ITEM_FACTOR_START",
    "STRING_LENGTH_CELL_COUNT",
    "STRING_LENGTH_FACTOR_SLOT",
    "STRING_LENGTH_UNKNOWN_CELL",
    "TOTAL_CELL_COUNT",
    "TYPE_CELL_COUNT",
    "TYPE_FACTOR_SLOT",
    "TypedCoincidenceResult",
    "TypedOutputCellBank",
    "TypedOutputEncoding",
    "exact_typed_output_coincidence",
]
