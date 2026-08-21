"""Sparse recurrent token circuit made only from cells and synapses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation, AllenGlif5Parameters
from humanbrain.native import NativeThreeFactorLearner, recurrent_kernel
from humanbrain.topology import RECURRENT_TOPOLOGIES, build_recurrent_targets

READOUT_LEARNING_RULES = frozenset({"ridge", "three_factor"})


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
    readout_learning_rule: str = "ridge"
    local_learning_rate_pa: float = 64.0
    local_competitive_margin_pa: float = 64.0
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
            self.local_learning_rate_pa,
            self.local_competitive_margin_pa,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("circuit time, current, and plasticity values must be positive")
        if self.input_pulse_ms > self.token_interval_ms:
            raise ValueError("input pulse cannot exceed the token interval")
        if not 0.0 <= self.inhibitory_fraction < 1.0:
            raise ValueError("inhibitory fraction must be in [0, 1)")
        if self.recurrent_topology not in RECURRENT_TOPOLOGIES:
            raise ValueError("unknown recurrent topology")
        if self.readout_learning_rule not in READOUT_LEARNING_RULES:
            raise ValueError("unknown readout learning rule")
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
    The default circuit learns synaptic strengths into output cells. The
    explicit developmental controller can additionally grow, score, prune, and
    homeostatically scale the existing sparse recurrent synapses during
    training. Output is selected by the first adaptive human-fitted cell to
    cross threshold.
    """

    _STATISTICS_BATCH_SIZE = 128

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
        self._reservoir = AdaptiveGlifPopulation(self.config.reservoir_cells, self.cell_parameters)
        self._synaptic_current_pa = np.zeros(self.config.reservoir_cells, dtype=np.float64)
        self._eligibility = np.zeros(self.config.reservoir_cells, dtype=np.float64)
        self._readout_synapses_pa = np.zeros(
            (self.token_cells, self.config.reservoir_cells), dtype=np.float64
        )
        self._pacemaker_synapses_pa = np.zeros(self.token_cells, dtype=np.float64)
        feature_count = self.config.reservoir_cells + 1
        self._normal_matrix: np.ndarray | None = None
        self._target_moments: np.ndarray | None = None
        self._feature_batch: np.ndarray | None = None
        self._growth_feature_count = feature_count
        self._pending_growth_examples = 0
        self._growth_examples = 0
        self._local_updates = 0
        self._observed_tokens = 0
        self._learning_frozen = False

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
        inhibitory_count = round(self.config.inhibitory_fraction * self.active_reservoir_cells)
        if inhibitory_count:
            inhibitory = rng.choice(self.active_reservoir_cells, inhibitory_count, replace=False)
            signs[inhibitory] = -1.0
        self._recurrent_weights_pa = np.repeat(
            (signs * self.config.recurrent_current_pa)[:, None],
            self.config.recurrent_fanout,
            axis=1,
        )
        # Tiny fixed heterogeneity avoids artificial simultaneous spikes while
        # remaining several orders below learned and afferent currents.
        self._output_heterogeneity_pa = rng.normal(0.0, 0.01, self.token_cells)
        self._token_steps = max(
            1, round(self.config.token_interval_ms / self.cell_parameters.dt_ms)
        )
        self._pulse_steps = max(1, round(self.config.input_pulse_ms / self.cell_parameters.dt_ms))
        self._recurrent_decay = np.exp(-self.cell_parameters.dt_ms / self.config.recurrent_tau_ms)
        self._eligibility_decay = np.exp(
            -self.cell_parameters.dt_ms / self.config.eligibility_tau_ms
        )
        self._compile_output_race()
        self._bind_native_kernel()

    def _bind_native_kernel(self) -> None:
        """Bind native execution to the circuit's current checkpoint arrays."""
        self._native_kernel = recurrent_kernel(
            self._reservoir,
            token_steps=self._token_steps,
            pulse_steps=self._pulse_steps,
            input_fanout=self.config.input_fanout,
            recurrent_fanout=self.config.recurrent_fanout,
            input_current_pa=self.config.input_current_pa,
            recurrent_decay=self._recurrent_decay,
            eligibility_decay=self._eligibility_decay,
            afferent_targets=self._afferent_targets,
            recurrent_targets=self._recurrent_targets,
            recurrent_weights_pa=self._recurrent_weights_pa,
            synaptic_current_pa=self._synaptic_current_pa,
            eligibility=self._eligibility,
        )
        self._native_local_learner = (
            NativeThreeFactorLearner(
                eligibility=self._eligibility,
                output_heterogeneity_pa=self._output_heterogeneity_pa,
                readout_synapses_pa=self._readout_synapses_pa,
                pacemaker_synapses_pa=self._pacemaker_synapses_pa,
            )
            if self.config.readout_learning_rule == "three_factor"
            else None
        )

    def _compile_output_race(self) -> None:
        """Compile the exact resting GLIF5 response before its first spike."""
        p = self.cell_parameters
        steps = max(1, round(self.config.output_race_ms / p.dt_ms))
        (
            self._output_unit_voltage,
            self._output_unit_voltage_threshold,
        ) = self._reservoir.compile_resting_response(steps)

    @property
    def learned_synapse_count(self) -> int:
        return int(np.count_nonzero(self._readout_synapses_pa))

    @property
    def learning_frozen(self) -> bool:
        return self._learning_frozen

    @property
    def local_update_count(self) -> int:
        return self._local_updates

    @property
    def observed_tokens(self) -> int:
        return self._observed_tokens

    @property
    def simulation_backend(self) -> str:
        return self._native_kernel.backend

    @property
    def simulation_device(self) -> str:
        return self._native_kernel.device_name

    @property
    def eligibility(self) -> np.ndarray:
        return self._eligibility.copy()

    def synchronize_structure_from_host(self) -> None:
        """Publish in-place afferent, recurrent-topology, and weight mutations."""
        self._native_kernel.synchronize_structure_from_host()

    def reset_sequence(self) -> None:
        self._reservoir.reset()
        self._synaptic_current_pa.fill(0.0)
        self._eligibility.fill(0.0)
        self._native_kernel.synchronize_from_host()
        self._observed_tokens = 0

    def observe(self, token_id: int) -> RecurrentStep:
        """Apply one token as afferent current and advance the cell circuit."""
        token = int(token_id)
        if not 0 <= token < self.token_cells:
            raise ValueError("token id outside output-cell population")
        spike_total = self._native_kernel.observe(token)

        self._observed_tokens += 1
        return RecurrentStep(
            token_id=token,
            reservoir_spikes=spike_total,
            simulated_steps=self._token_steps,
            biological_time_ms=self._token_steps * self.cell_parameters.dt_ms,
        )

    def strengthen(self, target_token_id: int) -> None:
        """Apply the configured readout plasticity rule after one observation."""
        target = int(target_token_id)
        if not 0 <= target < self.token_cells:
            raise ValueError("target token outside output-cell population")
        if self._observed_tokens < 1:
            raise RuntimeError("observe a token before strengthening output synapses")
        if self._learning_frozen:
            raise RuntimeError("readout learning is frozen")
        self._growth_examples += 1
        if self.config.readout_learning_rule == "three_factor":
            assert self._native_local_learner is not None
            winner = self._native_local_learner.update(
                target,
                self.config.local_learning_rate_pa,
                self.config.local_competitive_margin_pa,
            )
            self._local_updates += int(winner is not None)
            return

        self._ensure_growth_storage()
        assert self._feature_batch is not None
        assert self._target_moments is not None
        feature = self._feature_batch[self._pending_growth_examples]
        feature[:-1] = self._eligibility
        feature[-1] = 1.0
        self._target_moments[target] += feature
        self._pending_growth_examples += 1
        if self._pending_growth_examples == self._STATISTICS_BATCH_SIZE:
            self._flush_growth_statistics()

    def _ensure_growth_storage(self) -> None:
        """Allocate dense ridge statistics only when the circuit learns."""
        if self._normal_matrix is None:
            self._normal_matrix = np.zeros(
                (self._growth_feature_count, self._growth_feature_count),
                dtype=np.float64,
            )
        if self._target_moments is None:
            self._target_moments = np.zeros(
                (self.token_cells, self._growth_feature_count),
                dtype=np.float64,
            )
        if self._feature_batch is None:
            self._feature_batch = np.empty(
                (self._STATISTICS_BATCH_SIZE, self._growth_feature_count),
                dtype=np.float64,
            )

    def _flush_growth_statistics(self) -> None:
        """Accumulate a rank-k batch of cell traces into readout statistics."""
        if self._pending_growth_examples < 1:
            return
        assert self._feature_batch is not None
        assert self._normal_matrix is not None
        features = self._feature_batch[: self._pending_growth_examples]
        self._normal_matrix += features.T @ features
        self._pending_growth_examples = 0

    def _consolidate_readout(self) -> None:
        """Fit output-synapse strengths from accumulated cell traces."""
        if self.config.readout_learning_rule != "ridge":
            return
        if self._growth_examples < 1:
            raise RuntimeError("cannot consolidate without growth examples")
        self._flush_growth_statistics()
        assert self._normal_matrix is not None
        assert self._target_moments is not None
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

    def freeze_learning(self) -> None:
        """Finalize any deferred rule state and disable further plasticity."""
        if self._learning_frozen:
            return
        if self._growth_examples < 1:
            raise RuntimeError("cannot freeze readout before growth")
        self._consolidate_readout()
        self._learning_frozen = True

    def grow_sequence(self, token_ids: Sequence[int], epochs: int = 1) -> None:
        """Replay a sequence and grow synapses from cell eligibility traces."""
        if self._learning_frozen:
            raise RuntimeError("readout learning is frozen")
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
        if self.config.readout_learning_rule == "ridge":
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
        p = self.cell_parameters
        winning_current = current[int(np.argmax(current))]
        winner_voltage = self._output_unit_voltage * winning_current
        winner_threshold = self._output_unit_voltage_threshold * winning_current
        crossing_steps = np.flatnonzero(
            winner_voltage > (p.threshold_inf_relative_mv + winner_threshold)
        )
        if crossing_steps.size:
            step = int(crossing_steps[0])
            voltage_at_crossing = self._output_unit_voltage[step] * current
            fired = voltage_at_crossing > (
                p.threshold_inf_relative_mv + self._output_unit_voltage_threshold[step] * current
            )
            # Apply the fitted reset on the firing integration bin before
            # reporting its peak, exactly as stepwise simulation does.
            voltage_at_crossing[fired] = (
                p.voltage_reset_scale * voltage_at_crossing[fired] + p.voltage_reset_offset_mv
            )
            peak_relative = float(np.max(voltage_at_crossing))
            if step:
                peak_relative = max(
                    peak_relative,
                    float(winner_voltage[step - 1]),
                )
            winner = int(np.flatnonzero(fired)[0])
            return RecurrentRace(
                winner=winner,
                spike_time_ms=(step + 1) * p.dt_ms,
                peak_voltage_mv=max(p.rest_mv, p.rest_mv + peak_relative),
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
