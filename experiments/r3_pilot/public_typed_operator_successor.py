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
from humanbrain.typed_program import FactorAction


_CONTACT_CURRENT_PA = 5_000.0
_PAIR_BIAS_PA = -5_000.0
_UNKNOWN_TONIC_PA = 1_000.0
_RACE_STEPS = 600

_INT_KNOWN = output_cells.INT_CELL_COUNT - 1
_PAIR_KNOWN = _INT_KNOWN * _INT_KNOWN
_PAIR_UNKNOWN = _PAIR_KNOWN
_PAIR_COUNT = _PAIR_KNOWN + 1

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
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
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


def _physical_race(currents_pa: np.ndarray, *, bank: str) -> OperatorRace:
    currents = np.ascontiguousarray(currents_pa, dtype=np.float64)
    result = _RACE.race_many(currents[None, :])[0]
    return OperatorRace(
        bank=bank,
        candidate_count=len(currents),
        winner_cell=result.winner_cell,
        spike_step=result.spike_step,
        simultaneous_spike_candidates=result.simultaneous_spike_candidates,
        currents_sha256=sha256(currents.tobytes()).hexdigest(),
    )


@lru_cache(maxsize=4)
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
        )

    def effect_memo_info(self) -> OperatorEffectMemoInfo:
        """Return counters for the runtime-only normalized-effect memo."""

        return self._effect_memo.info()

    def clear_effect_memo(self) -> None:
        """Drop runtime reuse without changing the fitted checkpoint."""

        self._effect_memo.clear()

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
        currents = np.ascontiguousarray(
            self.unary_bias_pa + self.unary_absolute_weights_pa @ operand.int_cells
        )
        race = _physical_race(currents, bank="operator-absolute-output")
        return _int_from_winner(race.winner_cell), (race,)

    def _binary(
        self,
        operator: str,
        left: output_cells.TypedOutputEncoding,
        right: output_cells.TypedOutputEncoding,
    ) -> tuple[output_cells.TypedOutputEncoding, tuple[OperatorRace, ...]]:
        pair_currents = np.ascontiguousarray(
            self.pair_bias_pa
            + self.pair_left_contacts_pa @ left.int_cells
            + self.pair_right_contacts_pa @ right.int_cells
        )
        pair_race = _physical_race(pair_currents, bank="operator-int-pair")
        pair_spike = np.zeros(_PAIR_COUNT, dtype=np.float64)
        pair_spike[pair_race.winner_cell] = 1.0

        if operator == "less-than":
            currents = np.ascontiguousarray(
                self.bool_output_bias_pa + self.less_than_weights_pa @ pair_spike
            )
            output_race = _physical_race(currents, bank="operator-less-than-output")
            return _bool_from_winner(output_race.winner_cell), (
                pair_race,
                output_race,
            )

        weights = (
            self.add_weights_pa if operator == "add" else self.subtract_weights_pa
        )
        currents = np.ascontiguousarray(
            self.int_output_bias_pa + weights @ pair_spike
        )
        output_race = _physical_race(currents, bank=f"operator-{operator}-output")
        return _int_from_winner(output_race.winner_cell), (pair_race, output_race)

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
            tuple(
                self._predict_effect(value, action, row)
                for value, row in zip(selected_rows, scopes, strict=True)
            )
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
        )
    )
    checkpoint = _array_sha256(*arrays)
    frozen = FrozenTypedOperatorSuccessor(*arrays, checkpoint_sha256=checkpoint)
    learned = int(
        np.count_nonzero(unary)
        + np.count_nonzero(add)
        + np.count_nonzero(subtract)
        + np.count_nonzero(less_than)
    )
    fixed_contacts = int(np.count_nonzero(left_contacts) + np.count_nonzero(right_contacts))
    return frozen, TypedOperatorTrainingSummary(
        checkpoint_sha256=checkpoint,
        primitive_observations=_INT_KNOWN + 3 * _PAIR_KNOWN,
        unary_observations=_INT_KNOWN,
        binary_observations_per_operator=_PAIR_KNOWN,
        pair_cells=_PAIR_COUNT,
        learned_projection_synapses=learned,
        fixed_pair_contacts=fixed_contacts,
        claim_boundary=(
            "Target-free local Int/Bool primitive effects and List[Int] identity "
            "copy only; no candidate AST, composite target, planner, transfer, "
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
