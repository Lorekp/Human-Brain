"""Tie-inclusive physical beam selection through resting GLIF5 spike tiers.

Every round races the complete candidate population from rest.  The entire
earliest simultaneous-spike ensemble is retained and a materialized fixed
self-contact projection inhibits every member of that ensemble.  Selection
therefore advances tier-by-tier without a host or cell-index tie break.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from numbers import Real

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation, AllenGlif5Parameters
from humanbrain.resting_glif_race import (
    RestingGlifRaceFailure,
    RestingGlifRaceResult,
)


RETAINED = "RETAINED"
STOP_LEADING_TIER = "STOP_LEADING_TIER"


def _finite_current(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real current")
    current = float(value)
    if not np.isfinite(current):
        raise ValueError(f"{name} must be finite")
    return current


def _projection_sha256(weights_pa: np.ndarray, inhibition_current_pa: float) -> str:
    digest = sha256()
    digest.update(b"resting-glif-tier-self-inhibition-v1")
    digest.update(np.asarray(weights_pa.shape, dtype="<i8").tobytes())
    digest.update(np.asarray((inhibition_current_pa,), dtype="<f8").tobytes())
    digest.update(np.asarray(weights_pa, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FixedTierSelfInhibition:
    """Fixed one-contact-per-cell projection for a simultaneous spike tier."""

    candidate_count: int
    inhibition_current_pa: float
    weights_pa: np.ndarray
    sha256: str

    @classmethod
    def materialize(
        cls,
        *,
        candidate_count: int,
        inhibition_current_pa: Real,
    ) -> FixedTierSelfInhibition:
        if type(candidate_count) is not int or candidate_count < 1:
            raise ValueError("candidate_count must be a positive integer")
        inhibition = _finite_current(
            inhibition_current_pa,
            "inhibition_current_pa",
        )
        if inhibition <= 0.0:
            raise ValueError("inhibition_current_pa must be positive")
        weights = np.ascontiguousarray(
            np.full(candidate_count, -inhibition, dtype=np.float64)
        )
        digest = _projection_sha256(weights, inhibition)
        weights.setflags(write=False)
        return cls(
            candidate_count=candidate_count,
            inhibition_current_pa=inhibition,
            weights_pa=weights,
            sha256=digest,
        )

    @property
    def live_sha256(self) -> str:
        return _projection_sha256(self.weights_pa, self.inhibition_current_pa)

    def project(self, tier_spikes: Sequence[float] | np.ndarray) -> np.ndarray:
        spikes = np.asarray(tier_spikes, dtype=np.float64)
        if spikes.shape != (self.candidate_count,):
            raise ValueError("tier trigger must have one spike slot per candidate")
        if not np.all(np.isfinite(spikes)):
            raise ValueError("tier trigger must contain only finite values")
        if not np.all(np.logical_or(spikes == 0.0, spikes == 1.0)):
            raise ValueError("tier trigger must be binary")
        if np.count_nonzero(spikes) < 1:
            raise ValueError("tier trigger must contain at least one spike")
        return np.ascontiguousarray(self.weights_pa * spikes)


@dataclass(frozen=True, slots=True)
class RestingGlifTierBeamRound:
    """One full-population race and inhibition of its complete first tier."""

    tier_index: int
    race: RestingGlifRaceResult
    tier_cells: tuple[int, ...]
    leading_stop: bool
    currents_before_pa: tuple[float, ...]
    currents_after_pa: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RestingGlifTierBeamResult:
    """Tie-inclusive retained cells and the complete physical tier ledger."""

    status: str
    retained_cells: tuple[int, ...]
    retained_candidate_cells: tuple[int, ...]
    rounds: tuple[RestingGlifTierBeamRound, ...]
    stop_cell: int | None
    final_currents_pa: tuple[float, ...]
    inhibition_projection_before_sha256: str
    inhibition_projection_after_sha256: str


class CompiledRestingGlifTierBeam:
    """Reusable bounded-current selector that retains complete spike tiers."""

    def __init__(
        self,
        *,
        parameters: AllenGlif5Parameters,
        unit_voltage: np.ndarray,
        unit_voltage_threshold: np.ndarray,
        projection: FixedTierSelfInhibition,
        beam_width: int,
        min_current_pa: float,
        max_current_pa: float,
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
        self._projection = projection
        self.beam_width = beam_width
        self.candidate_count = projection.candidate_count
        self.min_current_pa = min_current_pa
        self.max_current_pa = max_current_pa

    @classmethod
    def compile(
        cls,
        *,
        candidate_count: int,
        step_count: int,
        min_current_pa: Real,
        max_current_pa: Real,
        inhibition_current_pa: Real,
        beam_width: int = 4,
        parameters: AllenGlif5Parameters | None = None,
    ) -> CompiledRestingGlifTierBeam:
        """Compile the unit response and validate the current envelope."""

        if type(candidate_count) is not int or candidate_count < 1:
            raise ValueError("candidate_count must be a positive integer")
        if type(beam_width) is not int or not 1 <= beam_width <= candidate_count:
            raise ValueError("beam_width must be in [1, candidate_count]")
        if type(step_count) is not int or step_count < 1:
            raise ValueError("step_count must be a positive integer")
        if parameters is not None and not isinstance(parameters, AllenGlif5Parameters):
            raise TypeError("parameters must be AllenGlif5Parameters")
        lower = _finite_current(min_current_pa, "min_current_pa")
        upper = _finite_current(max_current_pa, "max_current_pa")
        if lower > upper:
            raise ValueError("min_current_pa must not exceed max_current_pa")
        inhibition = _finite_current(
            inhibition_current_pa,
            "inhibition_current_pa",
        )
        if inhibition <= upper - lower:
            raise ValueError(
                "inhibition_current_pa must exceed the declared current span"
            )
        population = AdaptiveGlifPopulation(1, parameters)
        unit_voltage, unit_voltage_threshold = population.compile_resting_response(
            step_count
        )
        return cls(
            parameters=population.parameters,
            unit_voltage=unit_voltage,
            unit_voltage_threshold=unit_voltage_threshold,
            projection=FixedTierSelfInhibition.materialize(
                candidate_count=candidate_count,
                inhibition_current_pa=inhibition,
            ),
            beam_width=beam_width,
            min_current_pa=lower,
            max_current_pa=upper,
        )

    @property
    def inhibition_projection_sha256(self) -> str:
        return self._projection.sha256

    @property
    def live_inhibition_projection_sha256(self) -> str:
        return self._projection.live_sha256

    def _first_tier(
        self,
        currents_pa: np.ndarray,
    ) -> tuple[RestingGlifRaceResult, tuple[int, ...]]:
        for step in range(self._unit_voltage.size):
            fired = np.flatnonzero(
                self._unit_voltage[step] * currents_pa
                > self.parameters.threshold_inf_relative_mv
                + self._unit_voltage_threshold[step] * currents_pa
            )
            if fired.size:
                cells = tuple(int(cell) for cell in fired)
                return (
                    RestingGlifRaceResult(
                        winner_cell=cells[0],
                        spike_step=step,
                        simultaneous_spike_candidates=len(cells),
                    ),
                    cells,
                )
        raise RestingGlifRaceFailure(
            "tier race produced no spike within its fixed bound"
        )

    def select(
        self,
        currents_pa: Sequence[float] | np.ndarray,
        *,
        stop_cell: int | None = None,
    ) -> RestingGlifTierBeamResult:
        """Retain complete tiers until B real candidates have physically fired."""

        currents = np.asarray(currents_pa, dtype=np.float64)
        if currents.shape != (self.candidate_count,):
            raise ValueError("currents_pa must match the compiled candidate count")
        if not np.all(np.isfinite(currents)):
            raise ValueError("currents_pa must contain only finite values")
        if np.any(currents < self.min_current_pa) or np.any(
            currents > self.max_current_pa
        ):
            raise ValueError("currents_pa left the compiled current envelope")
        if stop_cell is not None:
            if type(stop_cell) is not int or not 0 <= stop_cell < self.candidate_count:
                raise ValueError("stop_cell must identify one compiled cell")
            if self.beam_width > self.candidate_count - 1:
                raise ValueError("beam_width exceeds the non-STOP candidate count")

        working = np.ascontiguousarray(currents.copy())
        projection_before = self.live_inhibition_projection_sha256
        if projection_before != self.inhibition_projection_sha256:
            raise RuntimeError("tier self-inhibition projection changed before selection")

        retained_cells: list[int] = []
        retained_candidates: list[int] = []
        rounds: list[RestingGlifTierBeamRound] = []
        inhibited = np.zeros(self.candidate_count, dtype=np.bool_)
        while len(retained_candidates) < self.beam_width:
            before = tuple(float(current) for current in working)
            race, tier_cells = self._first_tier(working)
            tier_index = len(rounds)
            tier_ids = np.asarray(tier_cells, dtype=np.intp)
            if np.any(inhibited[tier_ids]):
                raise RuntimeError("a physically inhibited tier cell fired twice")
            leading_stop = (
                tier_index == 0
                and stop_cell is not None
                and stop_cell in tier_cells
            )
            trigger = np.zeros(self.candidate_count, dtype=np.float64)
            trigger[tier_ids] = 1.0
            working += self._projection.project(trigger)
            inhibited[tier_ids] = True
            retained_cells.extend(tier_cells)
            if not leading_stop:
                retained_candidates.extend(
                    cell for cell in tier_cells if cell != stop_cell
                )
            rounds.append(
                RestingGlifTierBeamRound(
                    tier_index=tier_index,
                    race=race,
                    tier_cells=tier_cells,
                    leading_stop=leading_stop,
                    currents_before_pa=before,
                    currents_after_pa=tuple(float(current) for current in working),
                )
            )
            if leading_stop:
                break

        projection_after = self.live_inhibition_projection_sha256
        if projection_after != projection_before:
            raise RuntimeError("tier self-inhibition projection changed during selection")
        status = STOP_LEADING_TIER if rounds[-1].leading_stop else RETAINED
        return RestingGlifTierBeamResult(
            status=status,
            retained_cells=tuple(retained_cells),
            retained_candidate_cells=tuple(retained_candidates),
            rounds=tuple(rounds),
            stop_cell=stop_cell,
            final_currents_pa=tuple(float(current) for current in working),
            inhibition_projection_before_sha256=projection_before,
            inhibition_projection_after_sha256=projection_after,
        )


__all__ = [
    "CompiledRestingGlifTierBeam",
    "FixedTierSelfInhibition",
    "RETAINED",
    "RestingGlifRaceFailure",
    "RestingGlifTierBeamResult",
    "RestingGlifTierBeamRound",
    "STOP_LEADING_TIER",
]
