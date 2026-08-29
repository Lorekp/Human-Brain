"""Integrated physical H=1 factor planner for one typed root edit.

The incumbent alone is compiled and observed.  Every compiler-valid factor
action receives a target-free operator-successor prediction and remains in the
fixed value race; non-root actions receive physical typed UNKNOWN at this
root-only boundary.  The expensive structural successor runs exactly once,
after the unique value winner has passed through all six motor races.  Only
that structurally supported motor decision may reach the sole actuator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from experiments.r3_pilot import public_typed_factor_motor as factor_motor
from experiments.r3_pilot import public_typed_factor_value_race as value_race
from experiments.r3_pilot import public_typed_operator_successor as operator
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_program_observation as root_observation
from experiments.r3_pilot import public_typed_structure_successor as structure
from humanbrain import typed_program as tp
from humanbrain.resting_glif_race import RestingGlifRaceFailure
from humanbrain.typed_program_types import ProgramType


PUBLIC_TWO_INT_CASES = ((7, 2), (-3, 4), (5, 5))
PUBLIC_BOOL_CASES = ((-1,), (2,), (4,))
PUBLIC_LITERAL_BANK = tuple(range(-4, 5))

STATUS_SELECTED = "SELECTED"
STATUS_STOP_ALL_UNKNOWN = "STOP_ALL_UNKNOWN"
STATUS_STOP_VALUE_TIE = "STOP_VALUE_TIE"
STATUS_STOP_STRUCTURE_UNKNOWN = "STOP_STRUCTURE_UNKNOWN"

_UNKNOWN_VALUE = object()


def _declared_output_type(program_type: ProgramType) -> output_cells.OutputType:
    if program_type.constructor == "Int" and not program_type.arguments:
        return output_cells.OutputType.INT
    if program_type.constructor == "Bool" and not program_type.arguments:
        return output_cells.OutputType.BOOL
    if (
        program_type.constructor == "List"
        and len(program_type.arguments) == 1
        and program_type.arguments[0].constructor == "Int"
        and not program_type.arguments[0].arguments
    ):
        return output_cells.OutputType.LIST_INT
    raise ValueError("root factor planner supports Int, Bool, or List[Int]")


def _encode_value(
    value: object,
    program_type: ProgramType,
) -> output_cells.TypedOutputEncoding:
    declared = _declared_output_type(program_type)
    materialized = (
        list(value)
        if declared is output_cells.OutputType.LIST_INT
        and isinstance(value, tuple)
        else value
    )
    return output_cells.TypedOutputCellBank.encode(
        materialized,
        declared_type=declared,
    )


def _scope_rows(
    program: tp.Program,
    observation: root_observation.CurrentProgramRootObservation,
) -> tuple[tuple[output_cells.TypedOutputEncoding, ...], ...]:
    parameters = tuple(reversed(program.signature.parameters))
    rows: list[tuple[output_cells.TypedOutputEncoding, ...]] = []
    for case in observation.cases:
        values = tuple(reversed(case.arguments))
        rows.append(
            tuple(
                _encode_value(value, parameter.type)
                for parameter, value in zip(parameters, values, strict=True)
            )
        )
    return tuple(rows)


def _typed_unknown(
    declared_type: output_cells.OutputType,
) -> output_cells.TypedOutputEncoding:
    return output_cells.TypedOutputCellBank.encode(
        _UNKNOWN_VALUE,
        declared_type=declared_type,
    )


@dataclass(frozen=True, slots=True)
class CandidateValuePrediction:
    action: tp.FactorAction
    operator_predictions: tuple[operator.TypedOperatorPrediction, ...]
    scored_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    root_site_supported: bool


@dataclass(frozen=True, slots=True)
class TypedFactorH1RootDecision:
    status: str
    stop_reason: str | None
    incumbent_sha256: str
    candidate_bank: tp.FactorCandidateBank
    current_observation: root_observation.CurrentProgramRootObservation
    current_structure: structure.StructureObservation
    desired: tuple[output_cells.TypedOutputEncoding, ...]
    candidate_predictions: tuple[CandidateValuePrediction, ...]
    value_projection: value_race.FixedTypedFactorValueProjection
    value_decision: value_race.TypedFactorValueDecision | None
    diagnostic_currents_pa: tuple[float, ...]
    motor_projection: factor_motor.FixedFactorMotorProjection | None
    motor_decision: factor_motor.FactorMotorDecision | None
    selected_structure: structure.StructurePrediction | None
    selected_program: tp.Program | None
    final_observation: root_observation.CurrentProgramRootObservation | None
    final_structure: structure.StructureObservation | None
    structure_prediction_calls: int
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str


def _planner_checkpoint_sha256(
    operator_sha256: str,
    structure_sha256: str,
) -> str:
    digest = sha256()
    digest.update(b"public-typed-factor-h1-root-v1\0")
    digest.update(operator_sha256.encode("ascii"))
    digest.update(structure_sha256.encode("ascii"))
    digest.update(np.asarray(PUBLIC_LITERAL_BANK, dtype=np.int64).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenTypedFactorH1Root:
    operator_successor: operator.FrozenTypedOperatorSuccessor
    structure_successor: structure.FrozenStructureSuccessor
    checkpoint_sha256: str

    @property
    def live_checkpoint_sha256(self) -> str:
        return _planner_checkpoint_sha256(
            self.operator_successor.live_checkpoint_sha256,
            self.structure_successor.live_checkpoint_sha256,
        )

    def _stop(
        self,
        *,
        status: str,
        reason: str,
        incumbent: tp.Program,
        candidate_bank: tp.FactorCandidateBank,
        current: root_observation.CurrentProgramRootObservation,
        current_structure: structure.StructureObservation,
        desired: tuple[output_cells.TypedOutputEncoding, ...],
        candidates: tuple[CandidateValuePrediction, ...],
        projection: value_race.FixedTypedFactorValueProjection,
        value_decision: value_race.TypedFactorValueDecision | None,
        currents: tuple[float, ...],
        motor_projection: factor_motor.FixedFactorMotorProjection | None = None,
        motor_decision: factor_motor.FactorMotorDecision | None = None,
        selected_structure: structure.StructurePrediction | None = None,
        structure_prediction_calls: int = 0,
        before_sha256: str,
    ) -> TypedFactorH1RootDecision:
        after = self.live_checkpoint_sha256
        if after != before_sha256 or after != self.checkpoint_sha256:
            raise RuntimeError("frozen root-factor planner changed during STOP")
        return TypedFactorH1RootDecision(
            status=status,
            stop_reason=reason,
            incumbent_sha256=incumbent.sha256,
            candidate_bank=candidate_bank,
            current_observation=current,
            current_structure=current_structure,
            desired=desired,
            candidate_predictions=candidates,
            value_projection=projection,
            value_decision=value_decision,
            diagnostic_currents_pa=currents,
            motor_projection=motor_projection,
            motor_decision=motor_decision,
            selected_structure=selected_structure,
            selected_program=None,
            final_observation=None,
            final_structure=None,
            structure_prediction_calls=structure_prediction_calls,
            checkpoint_before_sha256=before_sha256,
            checkpoint_after_sha256=after,
        )

    def plan(
        self,
        incumbent: tp.Program,
        cases: Sequence[tuple[object, ...]],
        desired_outputs: Sequence[object],
    ) -> TypedFactorH1RootDecision:
        """Physically select and actuate at most one real root edit."""

        if not isinstance(incumbent, tp.Program):
            raise TypeError("root factor planning requires an incumbent Program")
        before = self.live_checkpoint_sha256
        if before != self.checkpoint_sha256:
            raise RuntimeError("root-factor checkpoint changed before planning")

        candidate_bank = tp.factor_action_candidates(
            incumbent,
            literals=PUBLIC_LITERAL_BANK,
        )
        current = root_observation.observe_current_program_root(incumbent, cases)
        if not current.all_supported:
            raise ValueError("incumbent root observation must be physically supported")
        current_structure = structure.observe_program_structure(incumbent)
        selected_rows = tuple(case.output_encoding for case in current.cases)
        scopes = _scope_rows(incumbent, current)
        supplied_desired = tuple(desired_outputs)
        if len(supplied_desired) != len(selected_rows):
            raise ValueError("desired outputs must align with the public case grid")
        desired = tuple(
            _encode_value(value, incumbent.signature.result)
            for value in supplied_desired
        )
        if not all(item.supported for item in desired):
            raise ValueError("desired outputs must inhabit physical typed cells")

        declared_root = _declared_output_type(incumbent.signature.result)
        unknown_root = _typed_unknown(declared_root)
        candidate_predictions: list[CandidateValuePrediction] = []
        scored_rows: list[tuple[output_cells.TypedOutputEncoding, ...]] = []
        for action in candidate_bank.actions:
            effects = self.operator_successor.predict_many(
                selected_rows,
                action,
                scope_rows=scopes,
            )
            root_supported = action.path == ()
            scored = (
                tuple(effect.output for effect in effects)
                if root_supported
                else (unknown_root,) * len(effects)
            )
            candidate_predictions.append(
                CandidateValuePrediction(
                    action=action,
                    operator_predictions=effects,
                    scored_outputs=scored,
                    root_site_supported=root_supported,
                )
            )
            scored_rows.append(scored)
        frozen_candidates = tuple(candidate_predictions)
        prediction_rows = tuple(scored_rows)
        projection = value_race.FixedTypedFactorValueProjection.build(
            candidate_bank.actions,
            desired,
        )
        try:
            value_decision = projection.select_unique(prediction_rows)
        except RestingGlifRaceFailure as error:
            features = projection.features(prediction_rows)
            currents = tuple(float(value) for value in projection.tier_currents(features))
            return self._stop(
                status=STATUS_STOP_VALUE_TIE,
                reason=str(error),
                incumbent=incumbent,
                candidate_bank=candidate_bank,
                current=current,
                current_structure=current_structure,
                desired=desired,
                candidates=frozen_candidates,
                projection=projection,
                value_decision=None,
                currents=currents,
                before_sha256=before,
            )
        if value_decision.selected_candidate_cell is None:
            return self._stop(
                status=STATUS_STOP_ALL_UNKNOWN,
                reason="fixed value race selected its physical STOP cell",
                incumbent=incumbent,
                candidate_bank=candidate_bank,
                current=current,
                current_structure=current_structure,
                desired=desired,
                candidates=frozen_candidates,
                projection=projection,
                value_decision=value_decision,
                currents=value_decision.currents_pa,
                before_sha256=before,
            )

        motor_projection = factor_motor.FixedFactorMotorProjection.build(
            candidate_bank.actions
        )
        selected_spikes = np.zeros(len(candidate_bank.actions), dtype=np.float64)
        selected_spikes[value_decision.selected_candidate_cell] = 1.0
        motor_decision = motor_projection.resolve(selected_spikes)
        if motor_decision.source_action != value_decision.selected_action:
            raise RuntimeError("value winner and six-factor motor lost parity")

        selected_structure = self.structure_successor.predict(
            current_structure,
            motor_decision.source_action,
        )
        if not selected_structure.supported:
            return self._stop(
                status=STATUS_STOP_STRUCTURE_UNKNOWN,
                reason="selected action produced physical structural UNKNOWN",
                incumbent=incumbent,
                candidate_bank=candidate_bank,
                current=current,
                current_structure=current_structure,
                desired=desired,
                candidates=frozen_candidates,
                projection=projection,
                value_decision=value_decision,
                currents=value_decision.currents_pa,
                motor_projection=motor_projection,
                motor_decision=motor_decision,
                selected_structure=selected_structure,
                structure_prediction_calls=1,
                before_sha256=before,
            )

        selected_program = factor_motor.actuate_selected_factor_action(
            incumbent,
            motor_decision,
        )
        final_observation = root_observation.observe_current_program_root(
            selected_program,
            cases,
        )
        final_structure = structure.observe_program_structure(selected_program)
        after = self.live_checkpoint_sha256
        if after != before or after != self.checkpoint_sha256:
            raise RuntimeError("frozen root-factor planner changed during planning")
        return TypedFactorH1RootDecision(
            status=STATUS_SELECTED,
            stop_reason=None,
            incumbent_sha256=incumbent.sha256,
            candidate_bank=candidate_bank,
            current_observation=current,
            current_structure=current_structure,
            desired=desired,
            candidate_predictions=frozen_candidates,
            value_projection=projection,
            value_decision=value_decision,
            diagnostic_currents_pa=value_decision.currents_pa,
            motor_projection=motor_projection,
            motor_decision=motor_decision,
            selected_structure=selected_structure,
            selected_program=selected_program,
            final_observation=final_observation,
            final_structure=final_structure,
            structure_prediction_calls=1,
            checkpoint_before_sha256=before,
            checkpoint_after_sha256=after,
        )


def build_public_typed_factor_h1_root() -> FrozenTypedFactorH1Root:
    """Build the two target-free frozen successor layers for visible planning."""

    operator_successor, _ = operator.train_public_typed_operator_successor()
    structure_successor, _ = structure.train_public_structure_successor()
    checkpoint = _planner_checkpoint_sha256(
        operator_successor.checkpoint_sha256,
        structure_successor.checkpoint_sha256,
    )
    return FrozenTypedFactorH1Root(
        operator_successor=operator_successor,
        structure_successor=structure_successor,
        checkpoint_sha256=checkpoint,
    )


__all__ = [
    "CandidateValuePrediction",
    "FrozenTypedFactorH1Root",
    "PUBLIC_BOOL_CASES",
    "PUBLIC_LITERAL_BANK",
    "PUBLIC_TWO_INT_CASES",
    "STATUS_SELECTED",
    "STATUS_STOP_ALL_UNKNOWN",
    "STATUS_STOP_STRUCTURE_UNKNOWN",
    "STATUS_STOP_VALUE_TIE",
    "TypedFactorH1RootDecision",
    "build_public_typed_factor_h1_root",
]
