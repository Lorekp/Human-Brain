"""Physical demultiplexing from one complete FactorAction winner.

The upstream planner supplies a stable candidate bank and a one-hot physical
winner.  Fixed synapses fan that cell into six independent fitted-GLIF races:
site, schema, operator, left source, right source, and literal.  This module has
no candidate scoring, successor model, program construction, compilation, or
execution.  The sole actuator entry point forwards the parity-checked selected
action to the compiler-owned typed edit seam.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from humanbrain import typed_program as tp
from humanbrain.cells import AdaptiveGlifPopulation


FACTOR_NAMES = (
    "site",
    "schema",
    "operator",
    "left-source",
    "right-source",
    "literal",
)
SCHEMA_LABELS = ("DONE", "APPLY1", "APPLY2", "MAP_BIND")
OPERATOR_LABELS = ("NONE", "absolute", "add", "subtract", "less-than", "map")
SOURCE_LABELS = ("NONE", "SELECTED", "SCOPE0", "SCOPE1", "BOUND0", "LITERAL")

_MOTOR_CURRENT_PA = 5_000.0
_RACE_STEPS = 30

FactorValue = tuple[int, ...] | str | int | None


class FactorMotorRaceError(RuntimeError):
    pass


class FactorMotorParityError(RuntimeError):
    pass


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _freeze(array: np.ndarray) -> np.ndarray:
    frozen = np.ascontiguousarray(array, dtype=np.float64)
    frozen.setflags(write=False)
    return frozen


def _action_record(action: tp.FactorAction) -> dict[str, object]:
    return {
        "path": action.path,
        "schema": action.schema,
        "operator": action.operator,
        "left_source": action.left_source,
        "right_source": action.right_source,
        "literal": action.literal,
    }


def _factor_value(action: tp.FactorAction, factor_name: str) -> FactorValue:
    if factor_name == "site":
        return action.path
    if factor_name == "schema":
        return action.schema
    if factor_name == "operator":
        return action.operator
    if factor_name == "left-source":
        return action.left_source
    if factor_name == "right-source":
        return action.right_source
    if factor_name == "literal":
        return action.literal
    raise AssertionError("unknown factor-motor axis")


@dataclass(frozen=True, slots=True)
class FixedOneHotFactorProjection:
    name: str
    labels: tuple[FactorValue, ...]
    weights_pa: np.ndarray
    sha256: str

    def currents(self, selected_candidate_cells: np.ndarray) -> np.ndarray:
        if selected_candidate_cells.shape != (self.weights_pa.shape[1],):
            raise ValueError("selected candidate cells do not match the fixed projection")
        return np.ascontiguousarray(self.weights_pa @ selected_candidate_cells)


def _build_factor_projection(
    factor_name: str,
    labels: tuple[FactorValue, ...],
    candidates: tuple[tp.FactorAction, ...],
    candidate_bank_sha256: str,
) -> FixedOneHotFactorProjection:
    weights = np.zeros((len(labels), len(candidates)), dtype=np.float64)
    label_index = {label: index for index, label in enumerate(labels)}
    if len(label_index) != len(labels):
        raise ValueError(f"{factor_name} factor labels must be unique")
    for candidate_cell, action in enumerate(candidates):
        weights[label_index[_factor_value(action, factor_name)], candidate_cell] = (
            _MOTOR_CURRENT_PA
        )
    frozen = _freeze(weights)
    digest = sha256()
    digest.update(candidate_bank_sha256.encode("ascii"))
    digest.update(factor_name.encode("ascii"))
    digest.update(_canonical_json(labels))
    digest.update(frozen.tobytes())
    return FixedOneHotFactorProjection(
        name=factor_name,
        labels=labels,
        weights_pa=frozen,
        sha256=digest.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class FactorRaceAudit:
    factor_name: str
    candidates: tuple[FactorValue, ...]
    currents_pa: tuple[float, ...]
    winner_cell: int
    winner: FactorValue
    spike_step: int
    simultaneous_spike_candidates: int
    projection_sha256: str


def _race_factor(
    projection: FixedOneHotFactorProjection,
    selected_candidate_cells: np.ndarray,
) -> FactorRaceAudit:
    currents = projection.currents(selected_candidate_cells)
    population = AdaptiveGlifPopulation(len(projection.labels))
    for step in range(_RACE_STEPS):
        fired = np.flatnonzero(population.step(currents))
        if not fired.size:
            continue
        if fired.size != 1:
            raise FactorMotorRaceError(
                f"{projection.name} motor produced {fired.size} simultaneous first spikes"
            )
        winner_cell = int(fired[0])
        return FactorRaceAudit(
            factor_name=projection.name,
            candidates=projection.labels,
            currents_pa=tuple(float(current) for current in currents),
            winner_cell=winner_cell,
            winner=projection.labels[winner_cell],
            spike_step=step,
            simultaneous_spike_candidates=1,
            projection_sha256=projection.sha256,
        )
    raise FactorMotorRaceError(
        f"{projection.name} motor produced no spike within its fixed bound"
    )


def _resolved_record(races: tuple[FactorRaceAudit, ...]) -> dict[str, object]:
    return {race.factor_name: race.winner for race in races}


def _decision_sha256(
    *,
    candidate_bank_sha256: str,
    projection_sha256: str,
    selected_candidate_cells_sha256: str,
    selected_candidate_cell: int,
    source_action: tp.FactorAction,
    races: tuple[FactorRaceAudit, ...],
) -> str:
    record = {
        "candidate_bank_sha256": candidate_bank_sha256,
        "projection_sha256": projection_sha256,
        "selected_candidate_cells_sha256": selected_candidate_cells_sha256,
        "selected_candidate_cell": selected_candidate_cell,
        "source_action": _action_record(source_action),
        "races": tuple(
            {
                "factor_name": race.factor_name,
                "winner_cell": race.winner_cell,
                "winner": race.winner,
                "spike_step": race.spike_step,
                "simultaneous_spike_candidates": race.simultaneous_spike_candidates,
                "projection_sha256": race.projection_sha256,
            }
            for race in races
        ),
    }
    return sha256(_canonical_json(record)).hexdigest()


@dataclass(frozen=True, slots=True)
class FactorMotorDecision:
    selected_candidate_cell: int
    source_action: tp.FactorAction
    selected_site: tuple[int, ...]
    selected_schema: str
    selected_operator: str
    selected_left_source: str
    selected_right_source: str
    selected_literal: int | None
    races: tuple[FactorRaceAudit, ...]
    parity_checked: bool
    candidate_bank_sha256: str
    projection_sha256: str
    selected_candidate_cells_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class FixedFactorMotorProjection:
    candidate_bank: tuple[tp.FactorAction, ...]
    factors: tuple[FixedOneHotFactorProjection, ...]
    candidate_bank_sha256: str
    sha256: str

    @classmethod
    def build(
        cls,
        candidate_bank: Sequence[tp.FactorAction],
    ) -> FixedFactorMotorProjection:
        candidates = tuple(candidate_bank)
        if not candidates or not all(type(action) is tp.FactorAction for action in candidates):
            raise TypeError("factor motor needs a non-empty FactorAction candidate bank")
        if len(set(candidates)) != len(candidates):
            raise ValueError("factor motor candidate bank must be stable and unique")
        done = tuple(action for action in candidates if action.schema == "DONE")
        if len(done) != 1 or _action_record(done[0]) != {
            "path": (),
            "schema": "DONE",
            "operator": "NONE",
            "left_source": "NONE",
            "right_source": "NONE",
            "literal": None,
        }:
            raise ValueError("complete factor motor bank needs one global DONE action")

        candidate_bank_sha = sha256(
            _canonical_json(tuple(_action_record(action) for action in candidates))
        ).hexdigest()
        site_labels: tuple[FactorValue, ...] = tuple(
            sorted({action.path for action in candidates})
        )
        literal_labels: tuple[FactorValue, ...] = (
            None,
            *tuple(
                sorted(
                    {
                        action.literal
                        for action in candidates
                        if action.literal is not None
                    }
                )
            ),
        )
        label_banks: tuple[tuple[FactorValue, ...], ...] = (
            site_labels,
            SCHEMA_LABELS,
            OPERATOR_LABELS,
            SOURCE_LABELS,
            SOURCE_LABELS,
            literal_labels,
        )
        factors = tuple(
            _build_factor_projection(
                factor_name,
                labels,
                candidates,
                candidate_bank_sha,
            )
            for factor_name, labels in zip(FACTOR_NAMES, label_banks, strict=True)
        )
        digest = sha256()
        digest.update(candidate_bank_sha.encode("ascii"))
        for factor in factors:
            digest.update(factor.sha256.encode("ascii"))
        return cls(
            candidate_bank=candidates,
            factors=factors,
            candidate_bank_sha256=candidate_bank_sha,
            sha256=digest.hexdigest(),
        )

    def factor(self, name: str) -> FixedOneHotFactorProjection:
        try:
            return next(factor for factor in self.factors if factor.name == name)
        except StopIteration as error:
            raise KeyError(name) from error

    def resolve(
        self,
        selected_candidate_cells: Sequence[float] | np.ndarray,
    ) -> FactorMotorDecision:
        selected = np.ascontiguousarray(selected_candidate_cells, dtype=np.float64)
        if selected.shape != (len(self.candidate_bank),):
            raise ValueError("physical candidate winner has the wrong bank width")
        if not np.all((selected == 0.0) | (selected == 1.0)):
            raise ValueError("physical candidate winner must be a binary spike mask")
        active = np.flatnonzero(selected)
        if active.size != 1:
            raise FactorMotorRaceError(
                "factor motor requires exactly one physically selected candidate cell"
            )
        selected_cell = int(active[0])
        source_action = self.candidate_bank[selected_cell]
        races = tuple(_race_factor(factor, selected) for factor in self.factors)
        if tuple(race.factor_name for race in races) != FACTOR_NAMES:
            raise FactorMotorParityError("factor motor lost its fixed six-race order")
        winners = tuple(race.winner for race in races)
        expected = (
            source_action.path,
            source_action.schema,
            source_action.operator,
            source_action.left_source,
            source_action.right_source,
            source_action.literal,
        )
        if winners != expected:
            raise FactorMotorParityError(
                "physical factor winners do not match the source FactorAction"
            )
        selected_sha = sha256(selected.tobytes()).hexdigest()
        decision_sha = _decision_sha256(
            candidate_bank_sha256=self.candidate_bank_sha256,
            projection_sha256=self.sha256,
            selected_candidate_cells_sha256=selected_sha,
            selected_candidate_cell=selected_cell,
            source_action=source_action,
            races=races,
        )
        return FactorMotorDecision(
            selected_candidate_cell=selected_cell,
            source_action=source_action,
            selected_site=source_action.path,
            selected_schema=source_action.schema,
            selected_operator=source_action.operator,
            selected_left_source=source_action.left_source,
            selected_right_source=source_action.right_source,
            selected_literal=source_action.literal,
            races=races,
            parity_checked=True,
            candidate_bank_sha256=self.candidate_bank_sha256,
            projection_sha256=self.sha256,
            selected_candidate_cells_sha256=selected_sha,
            sha256=decision_sha,
        )


def _validate_actuator_decision(decision: FactorMotorDecision) -> None:
    if type(decision) is not FactorMotorDecision or not decision.parity_checked:
        raise TypeError("factor actuator requires a parity-checked motor decision")
    if len(decision.races) != len(FACTOR_NAMES) or tuple(
        race.factor_name for race in decision.races
    ) != FACTOR_NAMES:
        raise FactorMotorParityError("factor actuator requires all six motor races")
    if any(
        race.simultaneous_spike_candidates != 1
        or race.candidates[race.winner_cell] != race.winner
        for race in decision.races
    ):
        raise FactorMotorRaceError("factor actuator rejected a non-unique motor audit")
    winners = tuple(race.winner for race in decision.races)
    source = decision.source_action
    if winners != (
        source.path,
        source.schema,
        source.operator,
        source.left_source,
        source.right_source,
        source.literal,
    ):
        raise FactorMotorParityError("factor actuator rejected a parity mismatch")
    expected_sha = _decision_sha256(
        candidate_bank_sha256=decision.candidate_bank_sha256,
        projection_sha256=decision.projection_sha256,
        selected_candidate_cells_sha256=decision.selected_candidate_cells_sha256,
        selected_candidate_cell=decision.selected_candidate_cell,
        source_action=source,
        races=decision.races,
    )
    if expected_sha != decision.sha256:
        raise FactorMotorParityError("factor actuator rejected a changed decision audit")


def actuate_selected_factor_action(
    program: tp.Program,
    decision: FactorMotorDecision,
) -> tp.Program:
    """Apply only a complete, unique, parity-checked six-factor decision."""

    _validate_actuator_decision(decision)
    return tp.apply_factor_action(program, decision.source_action)


__all__ = [
    "FACTOR_NAMES",
    "FactorMotorDecision",
    "FactorMotorParityError",
    "FactorMotorRaceError",
    "FactorRaceAudit",
    "FixedFactorMotorProjection",
    "FixedOneHotFactorProjection",
    "actuate_selected_factor_action",
]
