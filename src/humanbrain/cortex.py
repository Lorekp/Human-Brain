"""Sparse dendritic memory and event-driven human cortical cell simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import log1p

import numpy as np

from humanbrain.cells import AllenGlif3Parameters
from humanbrain.race_bias import PredictiveBiasConfig, apply_predictive_bias


@dataclass(frozen=True)
class HumanCellParameters:
    """Language-cortex parameters around a pinned Allen human GLIF3 cell.

    Passive membrane, threshold, reset, and after-spike parameters are fitted
    data from Allen human specimen 488392806. Dendritic association currents
    and inhibitory competition are explicit language-circuit parameters.
    """

    race_ms: float = 30.0
    cell: AllenGlif3Parameters = field(default_factory=AllenGlif3Parameters)
    tau_nmda_ms: float = 35.0
    tau_gabaa_ms: float = 8.0
    branch_current_pa: float = 95.0
    dendritic_probability_gain: float = 16.0
    basket_inhibition_pa: float = 240.0

    def __post_init__(self) -> None:
        positive = (
            self.race_ms,
            self.tau_nmda_ms,
            self.tau_gabaa_ms,
            self.branch_current_pa,
            self.dendritic_probability_gain,
            self.basket_inhibition_pa,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("cell time constants and drives must be positive")


@dataclass(frozen=True)
class CellRace:
    winner: int
    spike_time_ms: float
    peak_voltage_mv: float
    active_cells: int
    simulated_steps: int


class DendriticCortex:
    """Token cells with structurally grown context branches.

    A branch is keyed by the phase-ordered presynaptic token cells that contact
    it and terminates on one postsynaptic token cell. Repetition strengthens
    the branch. During recall, matching branches create NMDA-like drive; the
    output is the first pyramidal cell to spike under basket-cell inhibition.
    """

    def __init__(
        self,
        cell_count: int,
        max_order: int = 8,
        parameters: HumanCellParameters | None = None,
        seed: int = 7,
        predictive_bias: PredictiveBiasConfig | None = None,
    ) -> None:
        if cell_count < 2:
            raise ValueError("a cortex needs at least two token cells")
        if not 1 <= max_order <= 512:
            raise ValueError("max_order must be between 1 and 512")
        self.cell_count = int(cell_count)
        self.max_order = int(max_order)
        self.parameters = parameters or HumanCellParameters()
        # Probe P-B (predictive-state race bias): None / mode "off" leaves every
        # code path below untouched — flag-off is byte-identical to the
        # incumbent race (tests/test_race_bias.py gate 1).
        self.predictive_bias = predictive_bias
        self._branches: dict[tuple[int, ...], dict[int, float]] = {}
        self._unigram = np.zeros(self.cell_count, dtype=np.float64)
        rng = np.random.default_rng(seed)
        self._threshold_offset = rng.normal(0.0, 0.025, self.cell_count)
        self._grown_transitions = 0
        cell = self.parameters.cell
        self._nmda_decay = np.exp(-cell.dt_ms / self.parameters.tau_nmda_ms)
        self._voltage_step_scale = cell.dt_ms / cell.capacitance_pf
        self._race_steps = max(1, round(self.parameters.race_ms / cell.dt_ms))

        # The fitted GLIF3 membrane and decaying NMDA current are linear before
        # the first spike. Compile their exact forward-Euler unit response once
        # so every race need only scale it for the active synaptic currents.
        def compile_response(decay: float) -> np.ndarray:
            response = np.empty(self._race_steps, dtype=np.float64)
            voltage = 0.0
            current = 1.0
            for step in range(self._race_steps):
                voltage += self._voltage_step_scale * (current - cell.leak_ns * voltage)
                response[step] = voltage
                current *= decay
            return response

        self._unit_nmda_response = compile_response(self._nmda_decay)
        self._unit_nmda_peak_step = int(np.argmax(self._unit_nmda_response))

    @property
    def branch_count(self) -> int:
        return sum(len(targets) for targets in self._branches.values())

    @property
    def grown_transitions(self) -> int:
        return self._grown_transitions

    def grow_sequence(self, token_ids: Sequence[int], strength: float = 1.0) -> None:
        """Grow local dendritic associations from a replayed token sequence."""
        if strength <= 0.0:
            raise ValueError("growth strength must be positive")
        ids = [int(token_id) for token_id in token_ids]
        if any(token_id < 0 or token_id >= self.cell_count for token_id in ids):
            raise ValueError("token id outside cortical sheet")
        for target_index, target in enumerate(ids):
            self._unigram[target] += strength
            available = min(self.max_order, target_index)
            for order in range(1, available + 1):
                context = tuple(ids[target_index - order : target_index])
                targets = self._branches.setdefault(context, {})
                targets[target] = targets.get(target, 0.0) + strength
                self._grown_transitions += 1

    def _dendritic_drive(self, context: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        probability_by_cell: dict[int, float] = {}
        uniform_escape = 1.0
        available = min(self.max_order, len(context))
        # Branches converge from basal (short) to apical (specific) dendrites.
        # Each level reserves escape current for its lower-order input in
        # proportion to the number of distinct continuations it has grown.
        # This is Witten-Bell interpolation implemented as synaptic current,
        # not an external token selector.
        for order in range(1, available + 1):
            branch = self._branches.get(tuple(context[-order:]))
            if not branch:
                continue
            total_strength = sum(branch.values())
            distinct_targets = len(branch)
            denominator = total_strength + distinct_targets
            lower_weight = distinct_targets / denominator
            targets = probability_by_cell.keys() | branch.keys()
            probability_by_cell = {
                target: (
                    branch.get(target, 0.0)
                    + distinct_targets
                    * probability_by_cell.get(target, uniform_escape / self.cell_count)
                )
                / denominator
                for target in targets
            }
            uniform_escape *= lower_weight
        if not probability_by_cell:
            drive_by_cell: dict[int, float] = {}
            observed = np.flatnonzero(self._unigram > 0.0)
            if observed.size == 0:
                raise RuntimeError("cortex has not grown any sequence")
            for target in observed:
                drive_by_cell[int(target)] = 0.22 * log1p(float(self._unigram[target]))
        else:
            drive_by_cell = {
                target: self.parameters.dendritic_probability_gain * probability
                + 0.025 * log1p(float(self._unigram[target]))
                for target, probability in probability_by_cell.items()
            }
        active = np.fromiter(sorted(drive_by_cell), dtype=np.int32)
        drives = np.fromiter((drive_by_cell[int(i)] for i in active), dtype=np.float64)
        return active, drives

    def race(self, context: Sequence[int]) -> CellRace:
        """Simulate active pyramidal cells and return the first spike."""
        active, association_drive = self._dendritic_drive(context)
        bias = self.predictive_bias
        if bias is not None and bias.mode != "off":
            # Probe P-B: sustained somatic bias on theta-matched cells,
            # consumed downstream by this same first-spike race unchanged.
            association_drive = apply_predictive_bias(
                self, active, association_drive, context, bias
            )
        p = self.parameters
        cell = p.cell
        size = len(active)
        threshold = cell.threshold_relative_mv + self._threshold_offset[active]
        peak = cell.rest_mv
        phases = (
            association_drive,
            # A weak learned branch can remain subthreshold. A diffuse
            # neuromodulatory/arousal drive starts a second biological race;
            # it does not choose a token or inspect a non-cell score.
            association_drive + 6.0,
        )
        for phase_index, phase_drive in enumerate(phases):
            # Allen represents voltage relative to its fitted leak reversal.
            i_nmda = phase_drive * p.branch_current_pa
            # The fitted membrane is linear before its first spike. Search the
            # compiled response's monotonic rising limb for each cell's first
            # crossing without materializing a steps-by-cells matrix.
            rising_response = self._unit_nmda_response[: self._unit_nmda_peak_step + 1]
            first_steps = np.searchsorted(
                rising_response,
                threshold / i_nmda,
                side="left",
            )
            valid = first_steps < rising_response.size
            if not valid.any():
                peak = max(
                    peak,
                    cell.rest_mv + float(rising_response[-1] * np.max(i_nmda)),
                )
                continue
            step = int(np.min(first_steps[valid]))
            peak = max(
                peak,
                cell.rest_mv + float(self._unit_nmda_response[step] * np.max(i_nmda)),
            )
            indices = np.flatnonzero(first_steps == step)
            # Within a sub-millisecond integration bin, the largest threshold
            # crossing is the earlier continuous-time spike. Its fitted reset
            # and basket inhibition suppress every later competitor.
            local = int(
                indices[
                    np.argmax(self._unit_nmda_response[step] * i_nmda[indices] - threshold[indices])
                ]
            )
            elapsed_steps = phase_index * self._race_steps + step + 1
            return CellRace(
                winner=int(active[local]),
                spike_time_ms=elapsed_steps * cell.dt_ms,
                peak_voltage_mv=peak,
                active_cells=size,
                simulated_steps=elapsed_steps,
            )
        raise RuntimeError("active human cells did not spike under arousal")

    def matching_segments(self, context: Sequence[int]) -> list[tuple[int, int, float]]:
        """Raw segments matching the recent winner history, deterministically.

        Returns ``(order, target_cell, strength)`` for every grown branch whose
        presynaptic key equals the equally long tail of ``context``, ascending
        by order then target. Theta gating (connected-contact count) is a
        caller concern; see ``humanbrain.race_bias``.
        """
        ctx = tuple(int(token) for token in context)
        available = min(self.max_order, len(ctx))
        segments: list[tuple[int, int, float]] = []
        for order in range(1, available + 1):
            branch = self._branches.get(ctx[len(ctx) - order :])
            if not branch:
                continue
            for target in sorted(branch):
                segments.append((order, int(target), float(branch[target])))
        return segments

    def adjust_branch_strength(self, context, target: int, delta: float) -> bool:
        """Adjust one segment's strength, flooring at zero (segment removed).

        Used by the predictedSegmentDecrement analog in
        ``humanbrain.race_bias.grow_with_segment_decrement``. Returns whether
        the segment existed.
        """
        key = tuple(int(token) for token in context)
        targets = self._branches.get(key)
        if targets is None:
            return False
        tid = int(target)
        if tid not in targets:
            return False
        value = float(targets[tid]) + float(delta)
        if value <= 0.0:
            del targets[tid]
        else:
            targets[tid] = value
        return True

    def iter_branches(self) -> Iterable[tuple[tuple[int, ...], int, float]]:
        """Yield learned biological state for artifact serialization."""
        for context, targets in self._branches.items():
            for target, strength in targets.items():
                yield context, target, strength
