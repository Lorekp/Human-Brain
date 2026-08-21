"""Sparse recurrent token circuit made only from cells and synapses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation, AllenGlif5Parameters
from humanbrain.topology import RECURRENT_TOPOLOGIES, build_recurrent_targets


@dataclass(frozen=True)
class RecurrentCircuitConfig:
    reservoir_cells: int = 256
    active_reservoir_cells: int | None = None
    input_fanout: int = 24
    recurrent_fanout: int = 8
    token_interval_ms: float = 5.0
    input_pulse_ms: float = 1.5
    input_current_pa: float = 5_000.0
    recurrent_current_pa: float = 650.0
    recurrent_tau_ms: float = 8.0
    eligibility_tau_ms: float = 80.0
    output_base_current_pa: float = 450.0
    output_race_ms: float = 30.0
    readout_ridge: float = 1.0
    output_synapse_gain_pa: float = 1_600.0
    inhibitory_fraction: float = 0.2
    random_seed: int = 17
    recurrent_topology: str = "random"

    def __post_init__(self) -> None:
        if self.reservoir_cells < 8:
            raise ValueError("reservoir needs at least eight cells")
        active_cells = self.active_reservoir_cells or self.reservoir_cells
        if not 8 <= active_cells <= self.reservoir_cells:
            raise ValueError("active reservoir cells must be between eight and total cells")
        if not 1 <= self.input_fanout <= active_cells:
            raise ValueError("input fanout must fit the active reservoir")
        if not 1 <= self.recurrent_fanout <= active_cells:
            raise ValueError("recurrent fanout must fit the active reservoir")
        positive = (
            self.token_interval_ms,
            self.input_pulse_ms,
            self.input_current_pa,
            self.recurrent_current_pa,
            self.recurrent_tau_ms,
            self.eligibility_tau_ms,
            self.output_base_current_pa,
            self.output_race_ms,
            self.readout_ridge,
            self.output_synapse_gain_pa,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("circuit time, current, and plasticity values must be positive")
        if self.input_pulse_ms > self.token_interval_ms:
            raise ValueError("input pulse cannot exceed the token interval")
        if not 0.0 <= self.inhibitory_fraction < 1.0:
            raise ValueError("inhibitory fraction must be in [0, 1)")
        if self.recurrent_topology not in RECURRENT_TOPOLOGIES:
            raise ValueError("unknown recurrent topology")
        if self.recurrent_topology != "random" and self.recurrent_fanout >= active_cells:
            raise ValueError("non-random topology must exclude self-connections")


@dataclass(frozen=True)
class RecurrentStep:
    token_id: int
    reservoir_spikes: int
    simulated_steps: int
    biological_time_ms: float


@dataclass(frozen=True)
class RecurrentRace:
    winner: int
    spike_time_ms: float
    peak_voltage_mv: float
    active_cells: int
    simulated_steps: int


class RecurrentCellCircuit:
    """A liquid-state circuit with human-fitted adaptive cells.

    Token IDs only choose fixed afferent synapses. Context persists in fitted
    membrane voltage, after-spike currents, adaptive thresholds, recurrent
    synaptic current, and an eligibility trace driven by presynaptic spikes.
    The only learned values are synaptic strengths into output cells. Output is
    selected by the first adaptive human-fitted cell to cross threshold.
    """

    def __init__(
        self,
        token_cells: int,
        config: RecurrentCircuitConfig | None = None,
        cell_parameters: AllenGlif5Parameters | None = None,
    ) -> None:
        if token_cells < 2:
            raise ValueError("a circuit needs at least two token cells")
        self.token_cells = int(token_cells)
        self.config = config or RecurrentCircuitConfig()
        self.active_reservoir_cells = (
            self.config.active_reservoir_cells or self.config.reservoir_cells
        )
        self.cell_parameters = cell_parameters or AllenGlif5Parameters()
        self._reservoir = AdaptiveGlifPopulation(
            self.config.reservoir_cells, self.cell_parameters
        )
        self._synaptic_current_pa = np.zeros(self.config.reservoir_cells, dtype=np.float64)
        self._eligibility = np.zeros(self.config.reservoir_cells, dtype=np.float64)
        self._readout_synapses_pa = np.zeros(
            (self.token_cells, self.config.reservoir_cells), dtype=np.float64
        )
        self._pacemaker_synapses_pa = np.zeros(self.token_cells, dtype=np.float64)
        feature_count = self.config.reservoir_cells + 1
        self._normal_matrix = np.zeros((feature_count, feature_count), dtype=np.float64)
        self._target_moments = np.zeros((self.token_cells, feature_count), dtype=np.float64)
        self._growth_examples = 0
        self._observed_tokens = 0

        rng = np.random.default_rng(self.config.random_seed)
        self._afferent_targets = np.empty(
            (self.token_cells, self.config.input_fanout), dtype=np.int32
        )
        for token in range(self.token_cells):
            self._afferent_targets[token] = rng.choice(
                self.active_reservoir_cells,
                self.config.input_fanout,
                replace=False,
            )
        self._recurrent_targets = np.zeros(
            (self.config.reservoir_cells, self.config.recurrent_fanout), dtype=np.int32
        )
        self._recurrent_targets[: self.active_reservoir_cells] = build_recurrent_targets(
            self.active_reservoir_cells,
            self.config.recurrent_fanout,
            rng,
            self.config.recurrent_topology,
        )
        signs = np.zeros(self.config.reservoir_cells, dtype=np.float64)
        signs[: self.active_reservoir_cells] = 1.0
        inhibitory_count = round(
            self.config.inhibitory_fraction * self.active_reservoir_cells
        )
        if inhibitory_count:
            inhibitory = rng.choice(
                self.active_reservoir_cells, inhibitory_count, replace=False
            )
            signs[inhibitory] = -1.0
        self._recurrent_weights_pa = np.repeat(
            (signs * self.config.recurrent_current_pa)[:, None],
            self.config.recurrent_fanout,
            axis=1,
        )
        # Tiny fixed heterogeneity avoids artificial simultaneous spikes while
        # remaining several orders below learned and afferent currents.
        self._output_heterogeneity_pa = rng.normal(0.0, 0.01, self.token_cells)

    @property
    def learned_synapse_count(self) -> int:
        return int(np.count_nonzero(self._readout_synapses_pa))

    @property
    def observed_tokens(self) -> int:
        return self._observed_tokens

    @property
    def eligibility(self) -> np.ndarray:
        return self._eligibility.copy()

    def reset_sequence(self) -> None:
        self._reservoir.reset()
        self._synaptic_current_pa.fill(0.0)
        self._eligibility.fill(0.0)
        self._observed_tokens = 0

    def observe(self, token_id: int) -> RecurrentStep:
        """Apply one token as afferent current and advance the cell circuit."""
        token = int(token_id)
        if not 0 <= token < self.token_cells:
            raise ValueError("token id outside output-cell population")
        p = self.cell_parameters
        cfg = self.config
        steps = max(1, round(cfg.token_interval_ms / p.dt_ms))
        pulse_steps = max(1, round(cfg.input_pulse_ms / p.dt_ms))
        recurrent_decay = np.exp(-p.dt_ms / cfg.recurrent_tau_ms)
        eligibility_decay = np.exp(-p.dt_ms / cfg.eligibility_tau_ms)
        afferents = self._afferent_targets[token]
        spike_total = 0

        for step in range(steps):
            current = self._synaptic_current_pa.copy()
            if step < pulse_steps:
                current[afferents] += cfg.input_current_pa
            spikes = self._reservoir.step(current)
            self._synaptic_current_pa *= recurrent_decay
            self._eligibility *= eligibility_decay
            if np.any(spikes):
                presynaptic = np.flatnonzero(spikes)
                spike_total += int(presynaptic.size)
                self._eligibility[presynaptic] += 1.0
                targets = self._recurrent_targets[presynaptic].reshape(-1)
                weights = self._recurrent_weights_pa[presynaptic].reshape(-1)
                np.add.at(self._synaptic_current_pa, targets, weights)

        self._observed_tokens += 1
        return RecurrentStep(
            token_id=token,
            reservoir_spikes=spike_total,
            simulated_steps=steps,
            biological_time_ms=steps * p.dt_ms,
        )

    def strengthen(self, target_token_id: int) -> None:
        """Accumulate spike-driven sufficient statistics for consolidation."""
        target = int(target_token_id)
        if not 0 <= target < self.token_cells:
            raise ValueError("target token outside output-cell population")
        if self._observed_tokens < 1:
            raise RuntimeError("observe a token before strengthening output synapses")
        feature = np.empty(self.config.reservoir_cells + 1, dtype=np.float64)
        feature[:-1] = self._eligibility
        feature[-1] = 1.0
        self._normal_matrix += np.outer(feature, feature)
        self._target_moments[target] += feature
        self._growth_examples += 1

    def _consolidate_readout(self) -> None:
        """Fit output-synapse strengths from accumulated cell traces."""
        if self._growth_examples < 1:
            raise RuntimeError("cannot consolidate without growth examples")
        regularized = self._normal_matrix.copy()
        diagonal = np.diag_indices_from(regularized)
        regularized[diagonal] += self.config.readout_ridge
        # The pacemaker cell is a tonic biological bias and needs much less
        # regularization than the contextual reservoir synapses.
        regularized[-1, -1] -= self.config.readout_ridge * 0.99
        coefficients = np.linalg.solve(regularized, self._target_moments.T).T
        gain = self.config.output_synapse_gain_pa
        self._readout_synapses_pa[:] = coefficients[:, :-1] * gain
        self._pacemaker_synapses_pa[:] = coefficients[:, -1] * gain

    def grow_sequence(self, token_ids: Sequence[int], epochs: int = 1) -> None:
        """Replay a sequence and grow synapses from cell eligibility traces."""
        ids = [int(token_id) for token_id in token_ids]
        if len(ids) < 2:
            raise ValueError("growth sequence needs at least two tokens")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        for _ in range(epochs):
            self.reset_sequence()
            for index in range(len(ids) - 1):
                self.observe(ids[index])
                self.strengthen(ids[index + 1])
        self._consolidate_readout()

    def output_current_pa(self) -> np.ndarray:
        """Read the current delivered by all learned output synapses."""
        return (
            self.config.output_base_current_pa
            + self._pacemaker_synapses_pa
            + self._output_heterogeneity_pa
            + self._readout_synapses_pa @ self._eligibility
        )

    def race(self) -> RecurrentRace:
        """Race output cells under synaptic current; the first spike is the token."""
        if self._observed_tokens < 1:
            raise RuntimeError("observe context before racing output cells")
        current = self.output_current_pa()
        output = AdaptiveGlifPopulation(self.token_cells, self.cell_parameters)
        p = self.cell_parameters
        steps = max(1, round(self.config.output_race_ms / p.dt_ms))
        peak = p.rest_mv
        for step in range(steps):
            fired = output.step(current)
            peak = max(
                peak,
                p.rest_mv + float(np.max(output.state.voltage_relative_mv)),
            )
            if np.any(fired):
                # Fixed heterogeneity makes an exact simultaneous crossing
                # vanishingly unlikely. Array order is only the final physical
                # tie-break between spikes in the same 0.05-ms integration bin.
                winner = int(np.flatnonzero(fired)[0])
                return RecurrentRace(
                    winner=winner,
                    spike_time_ms=(step + 1) * p.dt_ms,
                    peak_voltage_mv=peak,
                    active_cells=self.token_cells,
                    simulated_steps=step + 1,
                )
        raise RuntimeError("output human cells did not spike during the race")

    def predict_next(self, context: Sequence[int]) -> RecurrentRace:
        """Replay a context through cell state and race output token cells."""
        if not context:
            raise ValueError("prediction context cannot be empty")
        self.reset_sequence()
        for token_id in context:
            self.observe(int(token_id))
        return self.race()
