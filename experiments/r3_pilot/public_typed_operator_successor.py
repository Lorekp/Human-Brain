"""Target-free physical successor for factorized typed operator effects.

The model learns primitive local value transitions, not programs or targets.
Integer operands first meet in a physical coincidence population; fixed
operator-efference banks then project that pair into Int or Bool output races.
Unary absolute and identity/copy paths use the same fitted resting-GLIF race
boundary.  Candidate programs are never constructed, compiled, or executed.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256

import numpy as np

from experiments.r3_pilot import public_typed_output_cells as output_cells
from humanbrain.resting_glif_race import CompiledRestingGlifRace
from humanbrain.typed_program import (
    ASCII_CLASS_LABELS,
    FactorAction,
    _ASCII_CLASS_CHARACTERS,
)


_CONTACT_CURRENT_PA = 5_000.0
_PAIR_BIAS_PA = -5_000.0
_UNKNOWN_TONIC_PA = 1_000.0
_RACE_STEPS = 600
_EVENT_BATCH_MAX_ROWS = 256

_INT_KNOWN = output_cells.INT_CELL_COUNT - 1
_PAIR_KNOWN = _INT_KNOWN * _INT_KNOWN
_PAIR_UNKNOWN = _PAIR_KNOWN
_PAIR_COUNT = _PAIR_KNOWN + 1

_ASCII_CLASS_UNKNOWN_CELL = len(ASCII_CLASS_LABELS)
_ASCII_CLASS_CELL_COUNT = _ASCII_CLASS_UNKNOWN_CELL + 1
_CLASS_CHAR_KNOWN = len(ASCII_CLASS_LABELS) * len(output_cells.PRINTABLE_ASCII)
_CLASS_CHAR_UNKNOWN = _CLASS_CHAR_KNOWN
_CLASS_CHAR_COUNT = _CLASS_CHAR_KNOWN + 1

# The bridge's observed effect domain is much smaller than the 16,641-entry
# physical operand plane.  Keep range expansion from silently growing memory.
_EFFECT_MEMO_MAX_ENTRIES = 4_096

_RACE = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
_UNKNOWN_VALUE = object()


def _freeze(array: np.ndarray) -> np.ndarray:
    frozen = np.ascontiguousarray(array, dtype=np.float64)
    frozen.setflags(write=False)
    return frozen


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = sha256()
    digest.update(b"public-typed-operator-successor-v1\0")
    for array in arrays:
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    digest.update(
        np.asarray(
            (_ASCII_CLASS_CELL_COUNT, _CLASS_CHAR_COUNT),
            dtype="<i8",
        ).tobytes()
    )
    for label in ASCII_CLASS_LABELS:
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(_ASCII_CLASS_CHARACTERS[label].encode("ascii"))
        digest.update(b"\0")
    digest.update(output_cells.TypedOutputCellBank.projection_sha256().encode("ascii"))
    digest.update(
        np.asarray(
            (_CONTACT_CURRENT_PA, _PAIR_BIAS_PA, _UNKNOWN_TONIC_PA, _RACE_STEPS),
            dtype=np.float64,
        ).tobytes()
    )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class OperatorRace:
    bank: str
    candidate_count: int
    winner_cell: int
    spike_step: int
    simultaneous_spike_candidates: int
    currents_sha256: str


@dataclass(frozen=True, slots=True)
class _AsciiClassEncoding:
    """One compiler-internal physical ASCII-class bank state."""

    label: str | None
    cells: np.ndarray
    supported: bool
    sha256: str


@lru_cache(maxsize=_ASCII_CLASS_CELL_COUNT)
def _ascii_class_from_winner(cell: int) -> _AsciiClassEncoding:
    if not 0 <= cell < _ASCII_CLASS_CELL_COUNT:
        raise ValueError("ASCII class winner lies outside its physical bank")
    cells = output_cells._physical_one_hot(_ASCII_CLASS_CELL_COUNT, cell)
    label = None if cell == _ASCII_CLASS_UNKNOWN_CELL else ASCII_CLASS_LABELS[cell]
    digest = sha256()
    digest.update(b"public-typed-ascii-class-v1\0")
    digest.update(cells.tobytes())
    return _AsciiClassEncoding(
        label=label,
        cells=cells,
        supported=label is not None,
        sha256=digest.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class OperatorEffectMemoInfo:
    """Auditable snapshot of the runtime-only normalized-effect memo."""

    max_entries: int
    entries: int
    hits: int
    misses: int
    evictions: int


_OperatorEffectKey = tuple[str, ...]
_OperatorEffect = tuple[
    output_cells.TypedOutputEncoding,
    tuple[OperatorRace, ...],
]


@dataclass(slots=True)
class _OperatorEffectMemo:
    """Bounded least-recently-used memo; never part of model identity."""

    max_entries: int = _EFFECT_MEMO_MAX_ENTRIES
    _entries: OrderedDict[_OperatorEffectKey, _OperatorEffect] = field(
        default_factory=OrderedDict
    )
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    def __post_init__(self) -> None:
        if type(self.max_entries) is not int or self.max_entries <= 0:
            raise ValueError("max_entries must be a positive exact integer")

    def resolve(
        self,
        key: _OperatorEffectKey,
        factory: Callable[[], _OperatorEffect],
    ) -> _OperatorEffect:
        effect = self._entries.pop(key, None)
        if effect is not None:
            self.hits += 1
            self._entries[key] = effect
            return effect
        self.misses += 1
        effect = factory()
        if len(self._entries) >= self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1
        self._entries[key] = effect
        return effect

    def miss_mask(
        self,
        keys: Sequence[_OperatorEffectKey],
    ) -> tuple[bool, ...]:
        """Plan sequential LRU misses without mutating the live memo."""

        simulated: OrderedDict[_OperatorEffectKey, None] = OrderedDict.fromkeys(
            self._entries
        )
        misses: list[bool] = []
        for key in keys:
            if key in simulated:
                simulated.move_to_end(key)
                misses.append(False)
                continue
            misses.append(True)
            if len(simulated) >= self.max_entries:
                simulated.popitem(last=False)
            simulated[key] = None
        return tuple(misses)

    def clear(self) -> None:
        self._entries.clear()
        self.hits = self.misses = self.evictions = 0

    def info(self) -> OperatorEffectMemoInfo:
        return OperatorEffectMemoInfo(
            max_entries=self.max_entries,
            entries=len(self._entries),
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
        )


def _physical_races(
    currents_pa: np.ndarray,
    *,
    bank: str,
) -> tuple[OperatorRace, ...]:
    """Race bounded independent rows while retaining each full-current digest."""

    currents = np.ascontiguousarray(currents_pa, dtype=np.float64)
    if currents.ndim != 2 or not 1 <= currents.shape[0] <= _EVENT_BATCH_MAX_ROWS:
        raise ValueError(
            "physical race batch must contain between 1 and "
            f"{_EVENT_BATCH_MAX_ROWS} rows"
        )
    results = _RACE.race_many(currents)
    return tuple(
        OperatorRace(
            bank=bank,
            candidate_count=currents.shape[1],
            winner_cell=result.winner_cell,
            spike_step=result.spike_step,
            simultaneous_spike_candidates=result.simultaneous_spike_candidates,
            currents_sha256=sha256(row.tobytes()).hexdigest(),
        )
        for row, result in zip(currents, results, strict=True)
    )


def _physical_race(currents_pa: np.ndarray, *, bank: str) -> OperatorRace:
    currents = np.ascontiguousarray(currents_pa, dtype=np.float64)
    if currents.ndim != 1:
        raise ValueError("physical race currents must be one-dimensional")
    return _physical_races(currents[None, :], bank=bank)[0]


def _single_event(activity: np.ndarray) -> tuple[int, float] | None:
    """Return one active presynaptic event without decoding its semantic value."""

    active = np.flatnonzero(activity)
    if active.size == 0:
        return None
    if active.size != 1:
        raise RuntimeError("typed output factor contains multiple active cells")
    cell = int(active[0])
    return cell, float(activity[cell])


@lru_cache(maxsize=len(output_cells.OutputType) + 1)
def _unknown(output_type: output_cells.OutputType | None) -> output_cells.TypedOutputEncoding:
    """Materialize each immutable physical UNKNOWN encoding once."""

    if output_type is None:
        return output_cells.TypedOutputCellBank.encode(_UNKNOWN_VALUE)
    return output_cells.TypedOutputCellBank.encode(
        _UNKNOWN_VALUE,
        declared_type=output_type,
    )


@lru_cache(maxsize=output_cells.INT_CELL_COUNT)
def _int_from_winner(cell: int) -> output_cells.TypedOutputEncoding:
    """Decode an Int winner through one immutable physical cell pattern."""

    if cell == output_cells.INT_UNKNOWN_CELL:
        return _unknown(output_cells.OutputType.INT)
    return output_cells.TypedOutputCellBank.encode(
        output_cells.INT_MIN + cell,
        declared_type=output_cells.OutputType.INT,
    )


@lru_cache(maxsize=output_cells.BOOL_CELL_COUNT)
def _bool_from_winner(cell: int) -> output_cells.TypedOutputEncoding:
    """Decode a Bool winner through one immutable physical cell pattern."""

    if cell == output_cells.BOOL_UNKNOWN_CELL:
        return _unknown(output_cells.OutputType.BOOL)
    return output_cells.TypedOutputCellBank.encode(
        cell == output_cells.BOOL_TRUE_CELL,
        declared_type=output_cells.OutputType.BOOL,
    )


@lru_cache(maxsize=output_cells.CHAR_CELL_COUNT)
def _char_from_winner(cell: int) -> output_cells.TypedOutputEncoding:
    """Decode a Char winner through one immutable physical cell pattern."""

    if cell == output_cells.CHAR_UNKNOWN_CELL:
        return _unknown(output_cells.OutputType.CHAR)
    return output_cells.TypedOutputCellBank.encode(
        chr(output_cells.PRINTABLE_ASCII_MIN + cell),
        declared_type=output_cells.OutputType.CHAR,
    )


def _string_from_winners(
    length: int,
    character_cells: tuple[int, ...],
) -> output_cells.TypedOutputEncoding:
    """Rebuild one String solely from its physical factor-race winners."""

    if length == output_cells.STRING_LENGTH_UNKNOWN_CELL:
        return _unknown(output_cells.OutputType.STRING)
    if length != len(character_cells) or length > output_cells.STRING_CAPACITY:
        raise RuntimeError("physical String winners do not match their length factor")
    cells = np.zeros(output_cells.TOTAL_CELL_COUNT, dtype=np.float64)
    active = np.zeros(len(output_cells.FACTOR_LABELS), dtype=bool)
    type_cell = tuple(output_cells.OutputType).index(output_cells.OutputType.STRING)
    cells[output_cells.TYPE_SLICE] = output_cells._physical_one_hot(
        output_cells.TYPE_CELL_COUNT,
        type_cell,
    )
    active[output_cells.FACTOR_LABELS.index("type")] = True
    cells[output_cells.STRING_LENGTH_SLICE] = output_cells._physical_one_hot(
        output_cells.STRING_LENGTH_CELL_COUNT,
        length,
    )
    active[output_cells.FACTOR_LABELS.index("string-length")] = True
    supported = True
    for position, cell in enumerate(character_cells):
        offset = output_cells.STRING_ITEM_OFFSET + position * output_cells.CHAR_CELL_COUNT
        cells[offset : offset + output_cells.CHAR_CELL_COUNT] = (
            output_cells._physical_one_hot(output_cells.CHAR_CELL_COUNT, cell)
        )
        active[
            output_cells.FACTOR_LABELS.index(f"string-char-{position}")
        ] = True
        supported = supported and cell != output_cells.CHAR_UNKNOWN_CELL
    return output_cells.TypedOutputCellBank._finish(
        output_cells.OutputType.STRING,
        cells,
        active,
        supported=supported,
    )


@lru_cache(maxsize=output_cells.INT_CELL_COUNT)
def _literal_int(value: int) -> output_cells.TypedOutputEncoding:
    """Reuse the immutable physical encoding of a target-free literal."""

    return output_cells.TypedOutputCellBank.encode(
        value,
        declared_type=output_cells.OutputType.INT,
    )


def _unknown_effect(
    output_type: output_cells.OutputType | None,
    bank: str,
) -> _OperatorEffect:
    return (
        _unknown(output_type),
        (
            _physical_race(
                np.asarray((_UNKNOWN_TONIC_PA,), dtype=np.float64),
                bank=bank,
            ),
        ),
    )


def _decode_copy(
    encoded: output_cells.TypedOutputEncoding,
) -> tuple[output_cells.TypedOutputEncoding, tuple[OperatorRace, ...]]:
    """Copy an encoding through fixed one-to-one physical factor races."""

    if encoded.output_type is None:
        race = _physical_race(
            np.asarray((_UNKNOWN_TONIC_PA,), dtype=np.float64),
            bank="copy-unknown-type",
        )
        return _unknown(None), (race,)

    type_currents = np.ascontiguousarray(
        encoded.type_cells * _CONTACT_CURRENT_PA
    )
    type_race = _physical_race(type_currents, bank="copy-type")
    races: list[OperatorRace] = [type_race]

    if encoded.output_type is output_cells.OutputType.INT:
        race = _physical_race(
            encoded.int_cells * _CONTACT_CURRENT_PA,
            bank="copy-int",
        )
        races.append(race)
        return _int_from_winner(race.winner_cell), tuple(races)

    if encoded.output_type is output_cells.OutputType.BOOL:
        race = _physical_race(
            encoded.bool_cells * _CONTACT_CURRENT_PA,
            bank="copy-bool",
        )
        races.append(race)
        return _bool_from_winner(race.winner_cell), tuple(races)

    if encoded.output_type is output_cells.OutputType.CHAR:
        race = _physical_race(
            encoded.char_cells * _CONTACT_CURRENT_PA,
            bank="copy-char",
        )
        races.append(race)
        return _char_from_winner(race.winner_cell), tuple(races)

    if encoded.output_type is output_cells.OutputType.STRING:
        length_race = _physical_race(
            encoded.string_length_cells * _CONTACT_CURRENT_PA,
            bank="copy-string-length",
        )
        races.append(length_race)
        if length_race.winner_cell == output_cells.STRING_LENGTH_UNKNOWN_CELL:
            return _unknown(output_cells.OutputType.STRING), tuple(races)
        character_cells: list[int] = []
        for position in range(length_race.winner_cell):
            character_race = _physical_race(
                encoded.string_item_cells[position] * _CONTACT_CURRENT_PA,
                bank=f"copy-string-char-{position}",
            )
            races.append(character_race)
            character_cells.append(character_race.winner_cell)
        return (
            _string_from_winners(
                length_race.winner_cell,
                tuple(character_cells),
            ),
            tuple(races),
        )

    length_race = _physical_race(
        encoded.list_length_cells * _CONTACT_CURRENT_PA,
        bank="copy-list-length",
    )
    races.append(length_race)
    if length_race.winner_cell == output_cells.LIST_LENGTH_UNKNOWN_CELL:
        return _unknown(output_cells.OutputType.LIST_INT), tuple(races)
    length = length_race.winner_cell
    values: list[int] = []
    for position in range(length):
        item_race = _physical_race(
            encoded.list_item_cells[position] * _CONTACT_CURRENT_PA,
            bank=f"copy-list-item-{position}",
        )
        races.append(item_race)
        if item_race.winner_cell == output_cells.INT_UNKNOWN_CELL:
            return _unknown(output_cells.OutputType.LIST_INT), tuple(races)
        values.append(output_cells.INT_MIN + item_race.winner_cell)
    return (
        output_cells.TypedOutputCellBank.encode(
            values,
            declared_type=output_cells.OutputType.LIST_INT,
        ),
        tuple(races),
    )


@dataclass(frozen=True, slots=True)
class TypedOperatorPrediction:
    output: output_cells.TypedOutputEncoding
    races: tuple[OperatorRace, ...]
    supported: bool
    action: FactorAction
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenTypedOperatorSuccessor:
    unary_absolute_weights_pa: np.ndarray
    unary_bias_pa: np.ndarray
    pair_left_contacts_pa: np.ndarray
    pair_right_contacts_pa: np.ndarray
    pair_bias_pa: np.ndarray
    add_weights_pa: np.ndarray
    subtract_weights_pa: np.ndarray
    less_than_weights_pa: np.ndarray
    int_output_bias_pa: np.ndarray
    bool_output_bias_pa: np.ndarray
    class_pair_class_contacts_pa: np.ndarray
    class_pair_char_contacts_pa: np.ndarray
    class_pair_bias_pa: np.ndarray
    contains_weights_pa: np.ndarray
    checkpoint_sha256: str
    _effect_memo: _OperatorEffectMemo = field(
        default_factory=_OperatorEffectMemo,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def live_checkpoint_sha256(self) -> str:
        return _array_sha256(
            self.unary_absolute_weights_pa,
            self.unary_bias_pa,
            self.pair_left_contacts_pa,
            self.pair_right_contacts_pa,
            self.pair_bias_pa,
            self.add_weights_pa,
            self.subtract_weights_pa,
            self.less_than_weights_pa,
            self.int_output_bias_pa,
            self.bool_output_bias_pa,
            self.class_pair_class_contacts_pa,
            self.class_pair_char_contacts_pa,
            self.class_pair_bias_pa,
            self.contains_weights_pa,
        )

    def effect_memo_info(self) -> OperatorEffectMemoInfo:
        """Return counters for the runtime-only normalized-effect memo."""

        return self._effect_memo.info()

    def clear_effect_memo(self) -> None:
        """Drop runtime reuse without changing the fitted checkpoint."""

        self._effect_memo.clear()

    def _ascii_class(
        self,
        label: str,
        *,
        enabled: bool,
    ) -> tuple[_AsciiClassEncoding, tuple[OperatorRace, ...]]:
        if label not in ASCII_CLASS_LABELS:
            raise ValueError("unknown frozen ASCII class")
        cell = (
            ASCII_CLASS_LABELS.index(label)
            if enabled
            else _ASCII_CLASS_UNKNOWN_CELL
        )
        currents = (
            output_cells._physical_one_hot(_ASCII_CLASS_CELL_COUNT, cell)
            * _CONTACT_CURRENT_PA
        )
        race = _physical_race(currents, bank="ascii-class")
        return _ascii_class_from_winner(race.winner_cell), (race,)

    def _contains(
        self,
        ascii_class: _AsciiClassEncoding,
        character: output_cells.TypedOutputEncoding,
    ) -> tuple[output_cells.TypedOutputEncoding, tuple[OperatorRace, ...]]:
        pair_currents = self.class_pair_bias_pa.copy()
        class_event = _single_event(ascii_class.cells)
        if class_event is not None:
            cell, amplitude = class_event
            pair_currents += self.class_pair_class_contacts_pa[:, cell] * amplitude
        char_event = _single_event(character.char_cells)
        if char_event is not None:
            cell, amplitude = char_event
            pair_currents += self.class_pair_char_contacts_pa[:, cell] * amplitude
        pair_race = _physical_race(
            np.ascontiguousarray(pair_currents),
            bank="operator-ascii-class-char-pair",
        )
        output_currents = self.bool_output_bias_pa.copy()
        output_currents += self.contains_weights_pa[:, pair_race.winner_cell]
        output_race = _physical_race(
            np.ascontiguousarray(output_currents),
            bank="operator-contains-output",
        )
        return _bool_from_winner(output_race.winner_cell), (pair_race, output_race)

    def _string_positions(
        self,
        value: output_cells.TypedOutputEncoding,
        positions: tuple[int, ...],
        *,
        bank: str,
    ) -> tuple[output_cells.TypedOutputEncoding, tuple[OperatorRace, ...]]:
        if value.output_type is not output_cells.OutputType.STRING:
            return _unknown_effect(output_cells.OutputType.STRING, f"{bank}-unknown")
        type_race = _physical_race(
            value.type_cells * _CONTACT_CURRENT_PA,
            bank="copy-type",
        )
        source_length_race = _physical_race(
            value.string_length_cells * _CONTACT_CURRENT_PA,
            bank="copy-string-length",
        )
        races: list[OperatorRace] = [type_race, source_length_race]
        source_length = source_length_race.winner_cell
        if (
            source_length == output_cells.STRING_LENGTH_UNKNOWN_CELL
            or any(not 0 <= position < source_length for position in positions)
            or len(positions) > output_cells.STRING_CAPACITY
        ):
            return _unknown(output_cells.OutputType.STRING), tuple(races)
        output_length_race = _physical_race(
            output_cells._physical_one_hot(
                output_cells.STRING_LENGTH_CELL_COUNT,
                len(positions),
            )
            * _CONTACT_CURRENT_PA,
            bank=f"{bank}-string-length",
        )
        races.append(output_length_race)
        character_cells: list[int] = []
        for output_position, source_position in enumerate(positions):
            race = _physical_race(
                value.string_item_cells[source_position] * _CONTACT_CURRENT_PA,
                bank=f"{bank}-string-char-{output_position}",
            )
            races.append(race)
            character_cells.append(race.winner_cell)
        return (
            _string_from_winners(len(positions), tuple(character_cells)),
            tuple(races),
        )

    @staticmethod
    def _source(
        label: str,
        selected: output_cells.TypedOutputEncoding,
        scopes: tuple[output_cells.TypedOutputEncoding, ...],
        literal: int | None,
    ) -> output_cells.TypedOutputEncoding:
        if label == "SELECTED":
            return selected
        if label in {"SCOPE0", "BOUND0"}:
            return scopes[0] if scopes else _unknown(None)
        if label == "SCOPE1":
            return scopes[1] if len(scopes) > 1 else _unknown(None)
        if label == "LITERAL":
            if type(literal) is not int:
                return _unknown(output_cells.OutputType.INT)
            return _literal_int(literal)
        return _unknown(None)

    def _unary(
        self,
        operand: output_cells.TypedOutputEncoding,
    ) -> tuple[output_cells.TypedOutputEncoding, tuple[OperatorRace, ...]]:
        currents = self.unary_bias_pa.copy()
        event = _single_event(operand.int_cells)
        if event is not None:
            cell, amplitude = event
            currents += self.unary_absolute_weights_pa[:, cell] * amplitude
        currents = np.ascontiguousarray(currents)
        race = _physical_race(currents, bank="operator-absolute-output")
        return _int_from_winner(race.winner_cell), (race,)

    def _binary_many(
        self,
        operator: str,
        left_rows: Sequence[output_cells.TypedOutputEncoding],
        right_rows: Sequence[output_cells.TypedOutputEncoding],
    ) -> tuple[_OperatorEffect, ...]:
        """Project one-hot spike events and run bounded physical race batches."""

        left = tuple(left_rows)
        right = tuple(right_rows)
        if not left or len(left) != len(right):
            raise ValueError("binary event rows must be nonempty and aligned")
        if operator not in {"add", "subtract", "less-than"}:
            raise ValueError("binary event batch has an unsupported operator")

        effects: list[_OperatorEffect] = []
        for start in range(0, len(left), _EVENT_BATCH_MAX_ROWS):
            left_chunk = left[start : start + _EVENT_BATCH_MAX_ROWS]
            right_chunk = right[start : start + _EVENT_BATCH_MAX_ROWS]
            row_count = len(left_chunk)
            pair_currents = np.broadcast_to(
                self.pair_bias_pa,
                (row_count, _PAIR_COUNT),
            ).copy()
            for row, encoded in enumerate(left_chunk):
                event = _single_event(encoded.int_cells)
                if event is not None:
                    cell, amplitude = event
                    pair_currents[row] += (
                        self.pair_left_contacts_pa[:, cell] * amplitude
                    )
            for row, encoded in enumerate(right_chunk):
                event = _single_event(encoded.int_cells)
                if event is not None:
                    cell, amplitude = event
                    pair_currents[row] += (
                        self.pair_right_contacts_pa[:, cell] * amplitude
                    )
            pair_currents = np.ascontiguousarray(pair_currents)
            pair_races = _physical_races(
                pair_currents,
                bank="operator-int-pair",
            )
            pair_winners = np.fromiter(
                (race.winner_cell for race in pair_races),
                dtype=np.intp,
                count=row_count,
            )

            if operator == "less-than":
                output_bias = self.bool_output_bias_pa
                output_weights = self.less_than_weights_pa
                output_bank = "operator-less-than-output"
                decode = _bool_from_winner
            else:
                output_bias = self.int_output_bias_pa
                output_weights = (
                    self.add_weights_pa
                    if operator == "add"
                    else self.subtract_weights_pa
                )
                output_bank = f"operator-{operator}-output"
                decode = _int_from_winner

            output_currents = np.broadcast_to(
                output_bias,
                (row_count, output_bias.size),
            ).copy()
            for row, cell in enumerate(pair_winners):
                output_currents[row] += output_weights[:, int(cell)]
            output_currents = np.ascontiguousarray(output_currents)
            output_races = _physical_races(
                output_currents,
                bank=output_bank,
            )
            effects.extend(
                (
                    decode(output_race.winner_cell),
                    (pair_race, output_race),
                )
                for pair_race, output_race in zip(
                    pair_races,
                    output_races,
                    strict=True,
                )
            )
        return tuple(effects)

    def _binary(
        self,
        operator: str,
        left: output_cells.TypedOutputEncoding,
        right: output_cells.TypedOutputEncoding,
    ) -> tuple[output_cells.TypedOutputEncoding, tuple[OperatorRace, ...]]:
        return self._binary_many(operator, (left,), (right,))[0]

    def _extremum(
        self,
        operator: str,
        left: output_cells.TypedOutputEncoding,
        right: output_cells.TypedOutputEncoding,
    ) -> tuple[output_cells.TypedOutputEncoding, tuple[OperatorRace, ...]]:
        """Select one operand through less-than cells, then physically copy it."""

        comparison, comparison_races = self._binary("less-than", left, right)
        event = _single_event(comparison.bool_cells)
        if event is None or event[0] == output_cells.BOOL_UNKNOWN_CELL:
            selected = _unknown(output_cells.OutputType.INT)
        else:
            less_than = event[0] == output_cells.BOOL_TRUE_CELL
            take_left = less_than if operator == "minimum" else not less_than
            selected = left if take_left else right
        copied, copy_races = _decode_copy(selected)
        return copied, (*comparison_races, *copy_races)

    def _predict_effects_many(
        self,
        selected: tuple[output_cells.TypedOutputEncoding, ...],
        action: FactorAction,
        scopes: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
    ) -> tuple[_OperatorEffect, ...]:
        if not selected:
            return ()
        if action.schema != "APPLY2" or action.operator not in {
            "add",
            "subtract",
            "less-than",
        }:
            return tuple(
                self._predict_effect(value, action, row)
                for value, row in zip(selected, scopes, strict=True)
            )

        left_rows = tuple(
            self._source(
                action.left_source,
                value,
                row,
                action.literal,
            )
            for value, row in zip(selected, scopes, strict=True)
        )
        right_rows = tuple(
            self._source(
                action.right_source,
                value,
                row,
                action.literal,
            )
            for value, row in zip(selected, scopes, strict=True)
        )
        keys = tuple(
            (
                self.checkpoint_sha256,
                "binary",
                action.operator,
                left.sha256,
                right.sha256,
            )
            for left, right in zip(left_rows, right_rows, strict=True)
        )
        miss_mask = self._effect_memo.miss_mask(keys)
        miss_indices = tuple(
            index for index, missed in enumerate(miss_mask) if missed
        )
        if miss_indices:
            missed_effects = self._binary_many(
                action.operator,
                tuple(left_rows[index] for index in miss_indices),
                tuple(right_rows[index] for index in miss_indices),
            )
            effects_by_index = dict(
                zip(miss_indices, missed_effects, strict=True)
            )
        else:
            effects_by_index = {}
        return tuple(
            self._effect_memo.resolve(
                key,
                lambda index=index: effects_by_index[index],
            )
            for index, key in enumerate(keys)
        )

    def predict(
        self,
        selected: output_cells.TypedOutputEncoding,
        action: FactorAction,
        *,
        scopes: Sequence[output_cells.TypedOutputEncoding] = (),
    ) -> TypedOperatorPrediction:
        """Predict one local edit effect without constructing a candidate AST."""

        if not isinstance(selected, output_cells.TypedOutputEncoding):
            raise TypeError("selected must be a physical typed-output encoding")
        if type(action) is not FactorAction:
            raise TypeError("action must be a FactorAction")
        scope_tuple = tuple(scopes)
        if not all(
            isinstance(scope, output_cells.TypedOutputEncoding)
            for scope in scope_tuple
        ):
            raise TypeError("scopes must contain physical typed-output encodings")
        before = self.live_checkpoint_sha256
        if before != self.checkpoint_sha256:
            raise RuntimeError("typed operator successor changed before prediction")

        output, races = self._predict_effect(selected, action, scope_tuple)

        after = self.live_checkpoint_sha256
        if after != before:
            raise RuntimeError("frozen typed operator successor changed in inference")
        return TypedOperatorPrediction(
            output=output,
            races=races,
            supported=output.supported,
            action=action,
            checkpoint_before_sha256=before,
            checkpoint_after_sha256=after,
        )

    def _predict_effect(
        self,
        selected: output_cells.TypedOutputEncoding,
        action: FactorAction,
        scopes: tuple[output_cells.TypedOutputEncoding, ...],
    ) -> tuple[output_cells.TypedOutputEncoding, tuple[OperatorRace, ...]]:
        """Run one effect after the caller authenticates the frozen arrays."""

        if action.schema == "DONE" and (
            action.path == ()
            and action.operator == "NONE"
            and action.left_source == "NONE"
            and action.right_source == "NONE"
            and action.literal is None
        ):
            return self._effect_memo.resolve(
                (self.checkpoint_sha256, "copy", selected.sha256),
                lambda: _decode_copy(selected),
            )
        elif (
            action.schema == "APPLY1"
            and action.operator == "absolute"
            and action.right_source == "NONE"
        ):
            operand = self._source(
                action.left_source, selected, scopes, action.literal
            )
            return self._effect_memo.resolve(
                (self.checkpoint_sha256, "unary-absolute", operand.sha256),
                lambda: self._unary(operand),
            )
        elif action.schema == "APPLY2" and action.operator in {
            "add",
            "subtract",
            "less-than",
        }:
            left = self._source(
                action.left_source, selected, scopes, action.literal
            )
            right = self._source(
                action.right_source, selected, scopes, action.literal
            )
            return self._effect_memo.resolve(
                (
                    self.checkpoint_sha256,
                    "binary",
                    action.operator,
                    left.sha256,
                    right.sha256,
                ),
                lambda: self._binary(action.operator, left, right),
            )
        elif action.schema == "APPLY2" and action.operator in {
            "minimum",
            "maximum",
        }:
            left = self._source(
                action.left_source, selected, scopes, action.literal
            )
            right = self._source(
                action.right_source, selected, scopes, action.literal
            )
            return self._effect_memo.resolve(
                (
                    self.checkpoint_sha256,
                    "extremum",
                    action.operator,
                    left.sha256,
                    right.sha256,
                ),
                lambda: self._extremum(action.operator, left, right),
            )
        elif (
            action.schema == "MAP_BIND"
            and action.operator == "map"
            and action.left_source == "BOUND0"
            and action.right_source in {"SELECTED", "SCOPE0", "SCOPE1"}
            and action.literal is None
        ):
            collection = self._source(
                action.right_source, selected, scopes, action.literal
            )
            if collection.output_type is output_cells.OutputType.LIST_INT:
                return self._effect_memo.resolve(
                    (
                        self.checkpoint_sha256,
                        "map-copy",
                        collection.sha256,
                    ),
                    lambda: _decode_copy(collection),
                )
            return self._effect_memo.resolve(
                (self.checkpoint_sha256, "map-unknown"),
                lambda: _unknown_effect(
                    output_cells.OutputType.LIST_INT,
                    "operator-map-unknown",
                ),
            )
        return self._effect_memo.resolve(
            (self.checkpoint_sha256, "action-unknown"),
            lambda: _unknown_effect(None, "operator-action-unknown"),
        )

    def predict_many(
        self,
        selected: Sequence[output_cells.TypedOutputEncoding],
        action: FactorAction,
        *,
        scope_rows: Sequence[Sequence[output_cells.TypedOutputEncoding]] | None = None,
    ) -> tuple[TypedOperatorPrediction, ...]:
        return self.predict_actions_many(
            selected,
            (action,),
            scope_rows=scope_rows,
        )[0]

    def predict_actions_many(
        self,
        selected: Sequence[output_cells.TypedOutputEncoding],
        actions: Sequence[FactorAction],
        *,
        scope_rows: Sequence[Sequence[output_cells.TypedOutputEncoding]] | None = None,
    ) -> tuple[tuple[TypedOperatorPrediction, ...], ...]:
        """Predict a fixed action bank under one checkpoint authentication."""

        selected_rows = tuple(selected)
        action_rows = tuple(actions)
        if scope_rows is None:
            scopes = ((),) * len(selected_rows)
        else:
            scopes = tuple(tuple(row) for row in scope_rows)
            if len(scopes) != len(selected_rows):
                raise ValueError("scope_rows must align with selected observations")
        if not all(
            isinstance(value, output_cells.TypedOutputEncoding)
            for value in selected_rows
        ) or not all(
            all(isinstance(scope, output_cells.TypedOutputEncoding) for scope in row)
            for row in scopes
        ):
            raise TypeError("batched rows must contain physical typed-output encodings")
        if not all(type(action) is FactorAction for action in action_rows):
            raise TypeError("actions must contain FactorAction values")
        before = self.live_checkpoint_sha256
        if before != self.checkpoint_sha256:
            raise RuntimeError("typed operator successor changed before batch")
        effects = tuple(
            self._predict_effects_many(selected_rows, action, scopes)
            for action in action_rows
        )
        after = self.live_checkpoint_sha256
        if after != before:
            raise RuntimeError("frozen typed operator successor changed in batch")
        return tuple(
            tuple(
                TypedOperatorPrediction(
                    output=output,
                    races=races,
                    supported=output.supported,
                    action=action,
                    checkpoint_before_sha256=before,
                    checkpoint_after_sha256=after,
                )
                for output, races in action_effects
            )
            for action, action_effects in zip(action_rows, effects, strict=True)
        )


@dataclass(frozen=True, slots=True)
class TypedOperatorTrainingSummary:
    checkpoint_sha256: str
    primitive_observations: int
    unary_observations: int
    binary_observations_per_operator: int
    pair_cells: int
    learned_projection_synapses: int
    fixed_pair_contacts: int
    claim_boundary: str


def train_public_typed_operator_successor(
) -> tuple[FrozenTypedOperatorSuccessor, TypedOperatorTrainingSummary]:
    """Grow readonly primitive transition projections without task targets."""

    unary = np.zeros(
        (output_cells.INT_CELL_COUNT, output_cells.INT_CELL_COUNT),
        dtype=np.float64,
    )
    unary_bias = np.zeros(output_cells.INT_CELL_COUNT, dtype=np.float64)
    unary_bias[output_cells.INT_UNKNOWN_CELL] = _UNKNOWN_TONIC_PA
    for input_cell in range(_INT_KNOWN):
        value = output_cells.INT_MIN + input_cell
        output_cell = abs(value) - output_cells.INT_MIN
        unary[output_cell, input_cell] = _CONTACT_CURRENT_PA
    unary[output_cells.INT_UNKNOWN_CELL, output_cells.INT_UNKNOWN_CELL] = (
        _CONTACT_CURRENT_PA
    )

    left_contacts = np.zeros((_PAIR_COUNT, output_cells.INT_CELL_COUNT))
    right_contacts = np.zeros_like(left_contacts)
    pair_bias = np.full(_PAIR_COUNT, _PAIR_BIAS_PA, dtype=np.float64)
    pair_bias[_PAIR_UNKNOWN] = _UNKNOWN_TONIC_PA
    for left_cell in range(_INT_KNOWN):
        for right_cell in range(_INT_KNOWN):
            pair_cell = left_cell * _INT_KNOWN + right_cell
            left_contacts[pair_cell, left_cell] = _CONTACT_CURRENT_PA
            right_contacts[pair_cell, right_cell] = _CONTACT_CURRENT_PA

    add = np.zeros((output_cells.INT_CELL_COUNT, _PAIR_COUNT), dtype=np.float64)
    subtract = np.zeros_like(add)
    less_than = np.zeros((output_cells.BOOL_CELL_COUNT, _PAIR_COUNT), dtype=np.float64)
    int_bias = np.zeros(output_cells.INT_CELL_COUNT, dtype=np.float64)
    int_bias[output_cells.INT_UNKNOWN_CELL] = _UNKNOWN_TONIC_PA
    bool_bias = np.zeros(output_cells.BOOL_CELL_COUNT, dtype=np.float64)
    bool_bias[output_cells.BOOL_UNKNOWN_CELL] = _UNKNOWN_TONIC_PA

    for left_cell in range(_INT_KNOWN):
        left = output_cells.INT_MIN + left_cell
        for right_cell in range(_INT_KNOWN):
            right = output_cells.INT_MIN + right_cell
            pair_cell = left_cell * _INT_KNOWN + right_cell
            added = left + right
            subtracted = left - right
            add_cell = (
                added - output_cells.INT_MIN
                if output_cells.INT_MIN <= added <= output_cells.INT_MAX
                else output_cells.INT_UNKNOWN_CELL
            )
            subtract_cell = (
                subtracted - output_cells.INT_MIN
                if output_cells.INT_MIN <= subtracted <= output_cells.INT_MAX
                else output_cells.INT_UNKNOWN_CELL
            )
            add[add_cell, pair_cell] = _CONTACT_CURRENT_PA
            subtract[subtract_cell, pair_cell] = _CONTACT_CURRENT_PA
            less_than[
                output_cells.BOOL_TRUE_CELL
                if left < right
                else output_cells.BOOL_FALSE_CELL,
                pair_cell,
            ] = _CONTACT_CURRENT_PA

    add[output_cells.INT_UNKNOWN_CELL, _PAIR_UNKNOWN] = _CONTACT_CURRENT_PA
    subtract[output_cells.INT_UNKNOWN_CELL, _PAIR_UNKNOWN] = _CONTACT_CURRENT_PA
    less_than[output_cells.BOOL_UNKNOWN_CELL, _PAIR_UNKNOWN] = _CONTACT_CURRENT_PA

    class_contacts = np.zeros(
        (_CLASS_CHAR_COUNT, _ASCII_CLASS_CELL_COUNT),
        dtype=np.float64,
    )
    char_contacts = np.zeros(
        (_CLASS_CHAR_COUNT, output_cells.CHAR_CELL_COUNT),
        dtype=np.float64,
    )
    class_pair_bias = np.full(
        _CLASS_CHAR_COUNT,
        _PAIR_BIAS_PA,
        dtype=np.float64,
    )
    class_pair_bias[_CLASS_CHAR_UNKNOWN] = _UNKNOWN_TONIC_PA
    contains = np.zeros(
        (output_cells.BOOL_CELL_COUNT, _CLASS_CHAR_COUNT),
        dtype=np.float64,
    )
    for class_cell, label in enumerate(ASCII_CLASS_LABELS):
        members = frozenset(_ASCII_CLASS_CHARACTERS[label])
        for char_cell, character in enumerate(output_cells.PRINTABLE_ASCII):
            pair_cell = class_cell * len(output_cells.PRINTABLE_ASCII) + char_cell
            class_contacts[pair_cell, class_cell] = _CONTACT_CURRENT_PA
            char_contacts[pair_cell, char_cell] = _CONTACT_CURRENT_PA
            contains[
                output_cells.BOOL_TRUE_CELL
                if character in members
                else output_cells.BOOL_FALSE_CELL,
                pair_cell,
            ] = _CONTACT_CURRENT_PA
    contains[output_cells.BOOL_UNKNOWN_CELL, _CLASS_CHAR_UNKNOWN] = (
        _CONTACT_CURRENT_PA
    )

    arrays = tuple(
        _freeze(array)
        for array in (
            unary,
            unary_bias,
            left_contacts,
            right_contacts,
            pair_bias,
            add,
            subtract,
            less_than,
            int_bias,
            bool_bias,
            class_contacts,
            char_contacts,
            class_pair_bias,
            contains,
        )
    )
    checkpoint = _array_sha256(*arrays)
    frozen = FrozenTypedOperatorSuccessor(*arrays, checkpoint_sha256=checkpoint)
    learned = int(
        np.count_nonzero(unary)
        + np.count_nonzero(add)
        + np.count_nonzero(subtract)
        + np.count_nonzero(less_than)
        + np.count_nonzero(contains)
    )
    fixed_contacts = int(
        np.count_nonzero(left_contacts)
        + np.count_nonzero(right_contacts)
        + np.count_nonzero(class_contacts)
        + np.count_nonzero(char_contacts)
    )
    return frozen, TypedOperatorTrainingSummary(
        checkpoint_sha256=checkpoint,
        primitive_observations=(
            _INT_KNOWN + 3 * _PAIR_KNOWN + _CLASS_CHAR_KNOWN
        ),
        unary_observations=_INT_KNOWN,
        binary_observations_per_operator=_PAIR_KNOWN,
        pair_cells=_PAIR_COUNT,
        learned_projection_synapses=learned,
        fixed_pair_contacts=fixed_contacts,
        claim_boundary=(
            "Target-free local Int/Bool and ASCII-class/Char primitive effects, "
            "plus factorized List[Int]/String copy only; no candidate AST, "
            "composite target, planner, transfer, "
            "Simple-SWE, M0, held-out, or C1 evidence."
        ),
    )


__all__ = [
    "FrozenTypedOperatorSuccessor",
    "OperatorEffectMemoInfo",
    "OperatorRace",
    "TypedOperatorPrediction",
    "TypedOperatorTrainingSummary",
    "train_public_typed_operator_successor",
]
