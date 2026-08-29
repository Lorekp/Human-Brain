"""Target-free bounded List[Int] behavior for a two-factor map sequence.

The first factor action may create an identity ``map`` binder.  The second may
edit only that newly created element body with the existing scalar APPLY1 or
APPLY2 operators.  List elements fan out through the frozen scalar operator
successor; fixed physical races then reassemble the list type, length, and item
banks.  This module does not enumerate or actuate actions and never constructs
a hypothetical program.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from experiments.r3_pilot import public_typed_operator_successor as scalar
from experiments.r3_pilot import public_typed_output_cells as output_cells
from humanbrain.resting_glif_race import CompiledRestingGlifRace
from humanbrain.typed_program import FactorAction


SUPPORTED = "SUPPORTED"
UNKNOWN = "UNKNOWN"

_CONTACT_CURRENT_PA = 5_000.0
_UNKNOWN_TONIC_PA = 1_000.0
_RACE_STEPS = 600
_RACE = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
_UNKNOWN_VALUE = object()
_LIST_TYPE_CELL = tuple(output_cells.OutputType).index(
    output_cells.OutputType.LIST_INT
)
_ELEMENT_SOURCES = frozenset({"SELECTED", "BOUND0", "SCOPE0"})
_BODY_SOURCES = frozenset((*_ELEMENT_SOURCES, "LITERAL"))
_COLLECTION_SOURCES = frozenset({"SELECTED", "SCOPE0", "SCOPE1", "BOUND0"})


@dataclass(frozen=True, slots=True)
class ListFactorRace:
    bank: str
    candidate_count: int
    winner_cell: int
    unknown_cell: int
    spike_step: int
    simultaneous_spike_candidates: int
    currents_sha256: str

    @property
    def unknown_won(self) -> bool:
        return self.winner_cell == self.unknown_cell


def _factor_race(
    factor_cells: np.ndarray,
    *,
    unknown_cell: int,
    bank: str,
) -> ListFactorRace:
    cells = np.ascontiguousarray(factor_cells, dtype=np.float64)
    if cells.ndim != 1 or not 0 <= unknown_cell < len(cells):
        raise ValueError("physical factor race has an invalid UNKNOWN cell")
    active = np.flatnonzero(cells > 0.0)
    currents = np.zeros(len(cells), dtype=np.float64)
    if active.size == 1:
        currents[int(active[0])] = _CONTACT_CURRENT_PA
    else:
        currents[unknown_cell] = _UNKNOWN_TONIC_PA
    result = _RACE.race_many(currents[None, :])[0]
    return ListFactorRace(
        bank=bank,
        candidate_count=len(currents),
        winner_cell=result.winner_cell,
        unknown_cell=unknown_cell,
        spike_step=result.spike_step,
        simultaneous_spike_candidates=result.simultaneous_spike_candidates,
        currents_sha256=sha256(currents.tobytes()).hexdigest(),
    )


def _type_race(
    encoding: output_cells.TypedOutputEncoding,
    *,
    bank: str,
) -> ListFactorRace:
    return _factor_race(
        np.concatenate((encoding.type_cells, np.zeros(1, dtype=np.float64))),
        unknown_cell=output_cells.TYPE_CELL_COUNT,
        bank=bank,
    )


def _length_race(
    encoding: output_cells.TypedOutputEncoding,
    *,
    bank: str,
) -> ListFactorRace:
    return _factor_race(
        encoding.list_length_cells,
        unknown_cell=output_cells.LIST_LENGTH_UNKNOWN_CELL,
        bank=bank,
    )


def _item_race(
    int_cells: np.ndarray,
    *,
    bank: str,
) -> ListFactorRace:
    return _factor_race(
        int_cells,
        unknown_cell=output_cells.INT_UNKNOWN_CELL,
        bank=bank,
    )


def _sequence_is_supported(first: FactorAction, second: FactorAction) -> bool:
    if (
        first.schema != "MAP_BIND"
        or first.operator != "map"
        or first.left_source != "BOUND0"
        or first.right_source not in _COLLECTION_SOURCES
        or first.literal is not None
        or second.path != (*first.path, 0, 0)
    ):
        return False
    if second.schema == "APPLY1":
        return (
            second.operator == "absolute"
            and second.left_source in _ELEMENT_SOURCES
            and second.right_source == "NONE"
            and second.literal is None
        )
    if second.schema != "APPLY2" or second.operator not in {"add", "subtract"}:
        return False
    sources = (second.left_source, second.right_source)
    uses_literal = "LITERAL" in sources
    return (
        all(source in _BODY_SOURCES for source in sources)
        and any(source in _ELEMENT_SOURCES for source in sources)
        and uses_literal == (second.literal is not None)
    )


def _sequence_race(supported: bool) -> ListFactorRace:
    cells = np.zeros(2, dtype=np.float64)
    cells[0 if supported else 1] = 1.0
    return _factor_race(cells, unknown_cell=1, bank="list-map-sequence")


def _list_unknown() -> output_cells.TypedOutputEncoding:
    return output_cells.TypedOutputCellBank.encode(
        _UNKNOWN_VALUE,
        declared_type=output_cells.OutputType.LIST_INT,
    )


def _int_encoding(winner_cell: int) -> output_cells.TypedOutputEncoding:
    value: object = (
        _UNKNOWN_VALUE
        if winner_cell == output_cells.INT_UNKNOWN_CELL
        else output_cells.INT_MIN + winner_cell
    )
    return output_cells.TypedOutputCellBank.encode(
        value,
        declared_type=output_cells.OutputType.INT,
    )


@dataclass(frozen=True, slots=True)
class ListMapCasePrediction:
    output: output_cells.TypedOutputEncoding
    map_prediction: scalar.TypedOperatorPrediction | None
    element_predictions: tuple[scalar.TypedOperatorPrediction, ...]
    fanout_races: tuple[ListFactorRace, ...]
    reassembly_races: tuple[ListFactorRace, ...]
    supported: bool

    @property
    def races(self) -> tuple[scalar.OperatorRace | ListFactorRace, ...]:
        map_races = () if self.map_prediction is None else self.map_prediction.races
        element_races = tuple(
            race
            for prediction in self.element_predictions
            for race in prediction.races
        )
        return (*map_races, *self.fanout_races, *element_races, *self.reassembly_races)


@dataclass(frozen=True, slots=True)
class ListMapBehaviorPrediction:
    status: str
    first_action: FactorAction
    second_action: FactorAction
    sequence_race: ListFactorRace
    cases: tuple[ListMapCasePrediction, ...]
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str

    @property
    def outputs(self) -> tuple[output_cells.TypedOutputEncoding, ...]:
        return tuple(case.output for case in self.cases)

    @property
    def supported(self) -> bool:
        return self.status == SUPPORTED

    @property
    def races(self) -> tuple[scalar.OperatorRace | ListFactorRace, ...]:
        return (
            self.sequence_race,
            *(race for case in self.cases for race in case.races),
        )


def _checkpoint_sha256(operator_checkpoint_sha256: str) -> str:
    digest = sha256()
    digest.update(b"public-typed-list-map-behavior-v1\0")
    digest.update(operator_checkpoint_sha256.encode("ascii"))
    digest.update(output_cells.TypedOutputCellBank.projection_sha256().encode("ascii"))
    digest.update(
        np.asarray(
            (
                output_cells.LIST_CAPACITY,
                output_cells.INT_MIN,
                output_cells.INT_MAX,
                _CONTACT_CURRENT_PA,
                _UNKNOWN_TONIC_PA,
                _RACE_STEPS,
            ),
            dtype=np.float64,
        ).tobytes()
    )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenListMapBehavior:
    """Predict MAP_BIND then element-body behavior through physical cells."""

    operator_successor: scalar.FrozenTypedOperatorSuccessor
    checkpoint_sha256: str

    @classmethod
    def build(
        cls,
        operator_successor: scalar.FrozenTypedOperatorSuccessor,
    ) -> FrozenListMapBehavior:
        if not isinstance(operator_successor, scalar.FrozenTypedOperatorSuccessor):
            raise TypeError("list-map behavior needs a frozen operator successor")
        if operator_successor.live_checkpoint_sha256 != (
            operator_successor.checkpoint_sha256
        ):
            raise RuntimeError("operator successor changed before list-map freeze")
        return cls(
            operator_successor=operator_successor,
            checkpoint_sha256=_checkpoint_sha256(
                operator_successor.checkpoint_sha256
            ),
        )

    @property
    def live_checkpoint_sha256(self) -> str:
        return _checkpoint_sha256(self.operator_successor.live_checkpoint_sha256)

    def predict(
        self,
        selected: Sequence[output_cells.TypedOutputEncoding],
        first_action: FactorAction,
        second_action: FactorAction,
        *,
        scope_rows: Sequence[Sequence[output_cells.TypedOutputEncoding]] | None = None,
    ) -> ListMapBehaviorPrediction:
        """Predict one two-action List[Int] behavior without candidate programs."""

        selected_rows = tuple(selected)
        if not selected_rows or not all(
            isinstance(value, output_cells.TypedOutputEncoding)
            for value in selected_rows
        ):
            raise TypeError("selected must contain physical typed-output encodings")
        if type(first_action) is not FactorAction or type(second_action) is not FactorAction:
            raise TypeError("list-map behavior requires two FactorAction values")
        if scope_rows is None:
            scopes = ((),) * len(selected_rows)
        else:
            scopes = tuple(tuple(row) for row in scope_rows)
            if len(scopes) != len(selected_rows):
                raise ValueError("scope_rows must align with selected observations")
            if not all(
                all(
                    isinstance(value, output_cells.TypedOutputEncoding)
                    for value in row
                )
                for row in scopes
            ):
                raise TypeError("scope_rows must contain physical encodings")

        before = self.live_checkpoint_sha256
        if before != self.checkpoint_sha256:
            raise RuntimeError("list-map behavior checkpoint changed before prediction")
        supported_sequence = _sequence_is_supported(first_action, second_action)
        sequence_race = _sequence_race(supported_sequence)
        if not supported_sequence:
            unknown_cases = tuple(
                ListMapCasePrediction(
                    output=_list_unknown(),
                    map_prediction=None,
                    element_predictions=(),
                    fanout_races=(),
                    reassembly_races=(
                        _factor_race(
                            np.zeros(output_cells.LIST_LENGTH_CELL_COUNT),
                            unknown_cell=output_cells.LIST_LENGTH_UNKNOWN_CELL,
                            bank=f"list-map-unsupported-{index}",
                        ),
                    ),
                    supported=False,
                )
                for index in range(len(selected_rows))
            )
            after = self.live_checkpoint_sha256
            if after != before:
                raise RuntimeError("frozen list-map behavior changed in inference")
            return ListMapBehaviorPrediction(
                status=UNKNOWN,
                first_action=first_action,
                second_action=second_action,
                sequence_race=sequence_race,
                cases=unknown_cases,
                checkpoint_before_sha256=before,
                checkpoint_after_sha256=after,
            )

        map_predictions = self.operator_successor.predict_many(
            selected_rows,
            first_action,
            scope_rows=scopes,
        )
        cases = tuple(
            self._predict_case(
                map_prediction,
                second_action,
                outer_scopes,
                case_index,
            )
            for case_index, (map_prediction, outer_scopes) in enumerate(
                zip(map_predictions, scopes, strict=True)
            )
        )
        after = self.live_checkpoint_sha256
        if after != before:
            raise RuntimeError("frozen list-map behavior changed in inference")
        all_supported = all(case.supported for case in cases)
        return ListMapBehaviorPrediction(
            status=SUPPORTED if all_supported else UNKNOWN,
            first_action=first_action,
            second_action=second_action,
            sequence_race=sequence_race,
            cases=cases,
            checkpoint_before_sha256=before,
            checkpoint_after_sha256=after,
        )

    def _predict_case(
        self,
        map_prediction: scalar.TypedOperatorPrediction,
        second_action: FactorAction,
        outer_scopes: tuple[output_cells.TypedOutputEncoding, ...],
        case_index: int,
    ) -> ListMapCasePrediction:
        collection = map_prediction.output
        type_race = _type_race(collection, bank=f"list-map-type-{case_index}")
        length_race = _length_race(
            collection,
            bank=f"list-map-length-{case_index}",
        )
        fanout: list[ListFactorRace] = [type_race, length_race]
        reassembly: list[ListFactorRace] = [
            _type_race(collection, bank=f"list-map-reassemble-type-{case_index}"),
            _length_race(
                collection,
                bank=f"list-map-reassemble-length-{case_index}",
            ),
        ]
        if (
            type_race.winner_cell != _LIST_TYPE_CELL
            or length_race.unknown_won
            or length_race.winner_cell > output_cells.LIST_CAPACITY
        ):
            return ListMapCasePrediction(
                output=_list_unknown(),
                map_prediction=map_prediction,
                element_predictions=(),
                fanout_races=tuple(fanout),
                reassembly_races=tuple(reassembly),
                supported=False,
            )

        raw_items: list[object] = []
        element_predictions: list[scalar.TypedOperatorPrediction] = []
        for position in range(length_race.winner_cell):
            item_fanout = _item_race(
                collection.list_item_cells[position],
                bank=f"list-map-fanout-{case_index}-{position}",
            )
            fanout.append(item_fanout)
            element = _int_encoding(item_fanout.winner_cell)
            element_prediction = self.operator_successor.predict(
                element,
                second_action,
                scopes=(element, *outer_scopes),
            )
            element_predictions.append(element_prediction)
            item_reassembly = _item_race(
                element_prediction.output.int_cells,
                bank=f"list-map-reassemble-item-{case_index}-{position}",
            )
            reassembly.append(item_reassembly)
            raw_items.append(
                _UNKNOWN_VALUE
                if item_reassembly.unknown_won
                else output_cells.INT_MIN + item_reassembly.winner_cell
            )

        output = output_cells.TypedOutputCellBank.encode(
            raw_items,
            declared_type=output_cells.OutputType.LIST_INT,
        )
        races = (*map_prediction.races, *fanout, *reassembly)
        race_unique = all(
            race.simultaneous_spike_candidates == 1 for race in races
        ) and all(
            race.simultaneous_spike_candidates == 1
            for prediction in element_predictions
            for race in prediction.races
        )
        return ListMapCasePrediction(
            output=output,
            map_prediction=map_prediction,
            element_predictions=tuple(element_predictions),
            fanout_races=tuple(fanout),
            reassembly_races=tuple(reassembly),
            supported=(
                map_prediction.supported
                and output.supported
                and all(item.supported for item in element_predictions)
                and race_unique
            ),
        )


def build_public_list_map_behavior() -> FrozenListMapBehavior:
    operator_successor, _ = scalar.train_public_typed_operator_successor()
    return FrozenListMapBehavior.build(operator_successor)


__all__ = [
    "FrozenListMapBehavior",
    "ListFactorRace",
    "ListMapBehaviorPrediction",
    "ListMapCasePrediction",
    "SUPPORTED",
    "UNKNOWN",
    "build_public_list_map_behavior",
]
