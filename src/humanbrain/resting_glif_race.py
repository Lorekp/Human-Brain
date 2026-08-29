"""Compiled first-spike races for independent resting GLIF5 populations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation, AllenGlif5Parameters


@dataclass(frozen=True, slots=True)
class RestingGlifRaceResult:
    """Observable first-spike result for one row of candidate currents."""

    winner_cell: int
    spike_step: int
    simultaneous_spike_candidates: int


class RestingGlifRaceFailure(RuntimeError):
    """A compiled race row had no spike or no unique first spiker."""


class CompiledRestingGlifRace:
    """Batch constant-current races that each start from fitted GLIF5 rest."""

    def __init__(
        self,
        *,
        parameters: AllenGlif5Parameters,
        unit_voltage: np.ndarray,
        unit_voltage_threshold: np.ndarray,
    ) -> None:
        self.parameters = parameters
        self._unit_voltage = np.ascontiguousarray(unit_voltage, dtype=np.float64)
        self._unit_voltage_threshold = np.ascontiguousarray(
            unit_voltage_threshold,
            dtype=np.float64,
        )
        if (
            self._unit_voltage.ndim != 1
            or self._unit_voltage.shape != self._unit_voltage_threshold.shape
            or self._unit_voltage.size < 1
        ):
            raise ValueError("compiled GLIF5 responses must be paired nonempty vectors")
        spike_gain = self._unit_voltage - self._unit_voltage_threshold
        if np.any(spike_gain <= 0.0) or np.any(np.diff(spike_gain) < 0.0):
            raise ValueError("compiled resting GLIF5 spike gain must be monotonic")
        self._unit_voltage.setflags(write=False)
        self._unit_voltage_threshold.setflags(write=False)

    @classmethod
    def compile(
        cls,
        *,
        step_count: int,
        parameters: AllenGlif5Parameters | None = None,
    ) -> CompiledRestingGlifRace:
        """Compile the constant-current response shared by every batch row."""

        if type(step_count) is not int or step_count < 1:
            raise ValueError("step_count must be a positive integer")
        if parameters is not None and not isinstance(parameters, AllenGlif5Parameters):
            raise TypeError("parameters must be AllenGlif5Parameters")
        population = AdaptiveGlifPopulation(1, parameters)
        unit_voltage, unit_voltage_threshold = population.compile_resting_response(
            step_count
        )
        return cls(
            parameters=population.parameters,
            unit_voltage=unit_voltage,
            unit_voltage_threshold=unit_voltage_threshold,
        )

    def race_many(
        self,
        currents_pa: Sequence[Sequence[float]] | np.ndarray,
        *,
        require_unique: bool = True,
    ) -> tuple[RestingGlifRaceResult, ...]:
        """Race one current row per independent population without sharing state."""

        if type(require_unique) is not bool:
            raise TypeError("require_unique must be a bool")
        currents = np.asarray(currents_pa, dtype=np.float64)
        if currents.ndim != 2 or currents.shape[0] < 1 or currents.shape[1] < 1:
            raise ValueError("currents_pa must be a nonempty rows-by-candidates matrix")
        if not np.all(np.isfinite(currents)):
            raise ValueError("currents_pa must contain only finite values")
        currents = np.ascontiguousarray(currents)

        winning_current = np.max(currents, axis=1)
        crossing = self._unit_voltage[:, None] * winning_current[None, :] > (
            self.parameters.threshold_inf_relative_mv
            + self._unit_voltage_threshold[:, None] * winning_current[None, :]
        )
        step_count = int(self._unit_voltage.size)
        first_steps = np.full(currents.shape[0], step_count, dtype=np.intp)
        for row in range(currents.shape[0]):
            row_crossings = np.flatnonzero(crossing[:, row])
            if row_crossings.size:
                first_steps[row] = int(row_crossings[0])

        safe_steps = np.minimum(first_steps, step_count - 1)
        fired = self._unit_voltage[safe_steps, None] * currents > (
            self.parameters.threshold_inf_relative_mv
            + self._unit_voltage_threshold[safe_steps, None] * currents
        )
        simultaneous = np.count_nonzero(fired, axis=1)
        candidate_ids = np.arange(currents.shape[1], dtype=np.intp)[None, :]
        winners = np.min(
            np.where(fired, candidate_ids, currents.shape[1]),
            axis=1,
        )

        results: list[RestingGlifRaceResult] = []
        for row in range(currents.shape[0]):
            if first_steps[row] == step_count:
                raise RestingGlifRaceFailure(
                    f"race row {row} produced no spike within its fixed bound"
                )
            count = int(simultaneous[row])
            if count < 1:
                raise RuntimeError("compiled first-spike crossing lost its firing cell")
            if require_unique and count != 1:
                raise RestingGlifRaceFailure(
                    f"race row {row} produced {count} simultaneous first spikes"
                )
            results.append(
                RestingGlifRaceResult(
                    winner_cell=int(winners[row]),
                    spike_step=int(first_steps[row]),
                    simultaneous_spike_candidates=count,
                )
            )
        return tuple(results)


__all__ = [
    "CompiledRestingGlifRace",
    "RestingGlifRaceFailure",
    "RestingGlifRaceResult",
]
