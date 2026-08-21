"""Human-data-fitted Allen GLIF cell equations used by the language cortex."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ALLEN_MODEL_ID = 503612028
ALLEN_SPECIMEN_ID = 488392806
ALLEN_CONFIG_SHA256 = "57e20326d27146c3afbd412ca3ba0bcaee67a22f27729292b36666b9ff36c889"
ALLEN_GLIF5_MODEL_ID = 523000114
ALLEN_GLIF5_SPECIMEN_ID = 508378649
ALLEN_GLIF5_CONFIG_SHA256 = "69fe726023e3289ac85857ea9125c55fee1618891e7b65403784c1b961bad149"


@dataclass(frozen=True)
class AllenGlif3Parameters:
    """Pinned Allen human L2 GLIF3 parameters in mV/ms/pA/pF/nS.

    The source JSON uses volts, seconds, amps, farads, and ohms. This port
    retains the AllenSDK model's forward-Euler voltage equation, constant
    fitted threshold, exponential after-spike currents, zero-voltage reset,
    and spike-cut refractory interval.
    """

    model_id: int = ALLEN_MODEL_ID
    specimen_id: int = ALLEN_SPECIMEN_ID
    rest_mv: float = -69.04794057210288
    dt_ms: float = 0.05
    capacitance_pf: float = 164.528173438895
    input_resistance_mohm: float = 118.06332822001779
    threshold_relative_mv: float = 25.631077893053025
    spike_cut_steps: int = 77
    after_spike_amplitudes_pa: tuple[float, float] = (
        -135.48819176942798,
        755.8435771154525,
    )
    after_spike_taus_ms: tuple[float, float] = (
        33.33333333333334,
        3.3333333333333335,
    )

    def __post_init__(self) -> None:
        if self.model_id != ALLEN_MODEL_ID or self.specimen_id != ALLEN_SPECIMEN_ID:
            raise ValueError("unrecognized Allen human-cell provenance")
        if self.dt_ms <= 0.0 or self.capacitance_pf <= 0.0 or self.input_resistance_mohm <= 0.0:
            raise ValueError("GLIF time and passive parameters must be positive")
        if self.spike_cut_steps < 0:
            raise ValueError("spike cut cannot be negative")
        if len(self.after_spike_amplitudes_pa) != len(self.after_spike_taus_ms):
            raise ValueError("after-spike amplitudes and time constants must align")

    @property
    def leak_ns(self) -> float:
        return 1_000.0 / self.input_resistance_mohm

    @property
    def membrane_tau_ms(self) -> float:
        return self.input_resistance_mohm * self.capacitance_pf / 1_000.0

    @property
    def threshold_absolute_mv(self) -> float:
        return self.rest_mv + self.threshold_relative_mv

    @property
    def spike_cut_ms(self) -> float:
        return self.spike_cut_steps * self.dt_ms

    @classmethod
    def from_allen_config(cls, path: str | Path) -> AllenGlif3Parameters:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        expected_methods = {
            "voltage_dynamics_method": "linear_forward_euler",
            "threshold_dynamics_method": "inf",
            "voltage_reset_method": "zero",
            "AScurrent_dynamics_method": "exp",
            "AScurrent_reset_method": "sum",
        }
        for field, name in expected_methods.items():
            if data[field]["name"] != name:
                raise ValueError(f"unsupported Allen rule {field}={data[field]['name']}")
        coeffs = data["coeffs"]
        rest_mv = float(data["El_reference"]) * 1_000.0
        return cls(
            rest_mv=rest_mv,
            dt_ms=float(data["dt"]) * 1_000.0,
            capacitance_pf=float(data["C"]) * float(coeffs["C"]) * 1e12,
            input_resistance_mohm=(float(data["R_input"]) / float(coeffs["G"]) / 1e6),
            threshold_relative_mv=(float(data["th_inf"]) * float(coeffs["th_inf"]) * 1_000.0),
            spike_cut_steps=int(data["spike_cut_length"]),
            after_spike_amplitudes_pa=tuple(
                float(value) * float(scale) * 1e12
                for value, scale in zip(data["asc_amp_array"], coeffs["asc_amp_array"], strict=True)
            ),
            after_spike_taus_ms=tuple(float(value) * 1_000.0 for value in data["asc_tau_array"]),
        )


@dataclass(frozen=True)
class AllenGlif5Parameters:
    """Pinned adaptive GLIF5 fit from one human MTG pyramidal cell.

    Values use mV, ms, pA, pF, and nS. Unlike the fast GLIF3 race cell,
    this cell retains fitted reset rules, after-spike currents, and both the
    spike- and voltage-dependent threshold components needed for recurrent
    state. The equations follow the AllenSDK ``three_components_exact`` model.
    """

    model_id: int = ALLEN_GLIF5_MODEL_ID
    specimen_id: int = ALLEN_GLIF5_SPECIMEN_ID
    rest_mv: float = -70.5407829284668
    dt_ms: float = 0.05
    capacitance_pf: float = 263.16486906079916
    input_resistance_mohm: float = 298.63483359781054
    threshold_inf_relative_mv: float = 23.806910928976624
    spike_cut_steps: int = 113
    after_spike_amplitudes_pa: tuple[float, float] = (
        -41.71849286427266,
        -63.68622662400284,
    )
    after_spike_taus_ms: tuple[float, float] = (100.0, 33.33333333333334)
    voltage_reset_scale: float = 1.669558016548051
    voltage_reset_offset_mv: float = -39.60700777695478
    threshold_spike_increment_mv: float = 2.5455720316895298
    threshold_spike_decay_per_s: float = 92.14596727562302
    threshold_voltage_coupling_per_s: float = 6.503496294985007
    threshold_voltage_decay_per_s: float = 19.773561643976418

    def __post_init__(self) -> None:
        if self.model_id != ALLEN_GLIF5_MODEL_ID or self.specimen_id != ALLEN_GLIF5_SPECIMEN_ID:
            raise ValueError("unrecognized adaptive Allen human-cell provenance")
        positive = (
            self.dt_ms,
            self.capacitance_pf,
            self.input_resistance_mohm,
            self.threshold_inf_relative_mv,
            self.threshold_spike_decay_per_s,
            self.threshold_voltage_decay_per_s,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("adaptive GLIF passive and threshold parameters must be positive")
        if self.spike_cut_steps < 0:
            raise ValueError("spike cut cannot be negative")
        if len(self.after_spike_amplitudes_pa) != len(self.after_spike_taus_ms):
            raise ValueError("after-spike amplitudes and time constants must align")

    @property
    def leak_ns(self) -> float:
        return 1_000.0 / self.input_resistance_mohm

    @property
    def membrane_tau_ms(self) -> float:
        return self.input_resistance_mohm * self.capacitance_pf / 1_000.0

    @property
    def spike_cut_ms(self) -> float:
        return self.spike_cut_steps * self.dt_ms

    @classmethod
    def from_allen_config(cls, path: str | Path) -> AllenGlif5Parameters:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        expected_methods = {
            "voltage_dynamics_method": "linear_forward_euler",
            "threshold_dynamics_method": "three_components_exact",
            "threshold_reset_method": "three_components",
            "voltage_reset_method": "v_before",
            "AScurrent_dynamics_method": "exp",
            "AScurrent_reset_method": "sum",
        }
        for field, name in expected_methods.items():
            if data[field]["name"] != name:
                raise ValueError(f"unsupported Allen rule {field}={data[field]['name']}")
        coeffs = data["coeffs"]
        threshold_dynamics = data["threshold_dynamics_method"]["params"]
        threshold_reset = data["threshold_reset_method"]["params"]
        voltage_reset = data["voltage_reset_method"]["params"]
        return cls(
            rest_mv=float(data["El_reference"]) * 1_000.0,
            dt_ms=float(data["dt"]) * 1_000.0,
            capacitance_pf=float(data["C"]) * float(coeffs["C"]) * 1e12,
            input_resistance_mohm=float(data["R_input"]) / float(coeffs["G"]) / 1e6,
            threshold_inf_relative_mv=(float(data["th_inf"]) * float(coeffs["th_inf"]) * 1_000.0),
            spike_cut_steps=int(data["spike_cut_length"]),
            after_spike_amplitudes_pa=tuple(
                float(value) * float(scale) * 1e12
                for value, scale in zip(data["asc_amp_array"], coeffs["asc_amp_array"], strict=True)
            ),
            after_spike_taus_ms=tuple(float(value) * 1_000.0 for value in data["asc_tau_array"]),
            voltage_reset_scale=float(voltage_reset["a"]),
            voltage_reset_offset_mv=float(voltage_reset["b"]) * 1_000.0,
            threshold_spike_increment_mv=float(threshold_reset["a_spike"]) * 1_000.0,
            threshold_spike_decay_per_s=float(threshold_reset["b_spike"]),
            threshold_voltage_coupling_per_s=(
                float(threshold_dynamics["a_voltage"]) * float(coeffs["a"])
            ),
            threshold_voltage_decay_per_s=(
                float(threshold_dynamics["b_voltage"]) * float(coeffs["b"])
            ),
        )


@dataclass
class AdaptiveGlifState:
    """All persistent state of a population of adaptive human-fitted cells."""

    voltage_relative_mv: np.ndarray
    after_spike_currents_pa: np.ndarray
    threshold_spike_mv: np.ndarray
    threshold_voltage_mv: np.ndarray
    refractory_steps: np.ndarray

    @property
    def cell_count(self) -> int:
        return int(self.voltage_relative_mv.size)


class AdaptiveGlifPopulation:
    """Vectorized stateful simulator for homogeneous Allen GLIF5 cells."""

    def __init__(self, cell_count: int, parameters: AllenGlif5Parameters | None = None) -> None:
        if cell_count < 1:
            raise ValueError("a population needs at least one cell")
        self._cell_count = int(cell_count)
        self.parameters = parameters or AllenGlif5Parameters()
        self.state = AdaptiveGlifState(
            voltage_relative_mv=np.zeros(cell_count, dtype=np.float64),
            after_spike_currents_pa=np.zeros(
                (cell_count, len(self.parameters.after_spike_taus_ms)), dtype=np.float64
            ),
            threshold_spike_mv=np.zeros(cell_count, dtype=np.float64),
            threshold_voltage_mv=np.zeros(cell_count, dtype=np.float64),
            refractory_steps=np.zeros(cell_count, dtype=np.int32),
        )
        # These factors depend only on the fitted cell parameters.  Keeping
        # them on the population avoids recomputing exponentials at every
        # 0.05-ms integration step without changing the GLIF5 equations.
        dt_s = self.parameters.dt_ms / 1_000.0
        leak_per_s = self.parameters.leak_ns / self.parameters.capacitance_pf * 1_000.0
        a_voltage = self.parameters.threshold_voltage_coupling_per_s
        b_voltage = self.parameters.threshold_voltage_decay_per_s
        self._voltage_step_scale = self.parameters.dt_ms / self.parameters.capacitance_pf
        self._threshold_phi = a_voltage / (b_voltage - leak_per_s)
        self._threshold_beta_scale = 1.0 / self.parameters.leak_ns
        self._threshold_exp_leak = np.exp(-leak_per_s * dt_s)
        self._threshold_exp_decay = np.exp(-b_voltage * dt_s)
        self._threshold_voltage_ratio = a_voltage / b_voltage
        # Algebraic coefficients for the Allen three-component threshold
        # equation. This is the expanded form of the reference expression and
        # avoids recomputing its shared ``phi * (voltage - beta)`` term.
        self._threshold_voltage_factor = self._threshold_phi * (
            self._threshold_exp_leak - self._threshold_exp_decay
        )
        self._threshold_beta_factor = (
            -self._threshold_voltage_factor
            + self._threshold_voltage_ratio * (1.0 - self._threshold_exp_decay)
        )
        self._spike_threshold_decay = np.exp(-self.parameters.threshold_spike_decay_per_s * dt_s)
        after_spike_taus_ms = np.asarray(self.parameters.after_spike_taus_ms, dtype=np.float64)
        self._after_spike_decay = np.exp(-self.parameters.dt_ms / after_spike_taus_ms)
        self._after_spike_cut_decay = np.exp(-self.parameters.spike_cut_ms / after_spike_taus_ms)
        self._after_spike_amplitudes_pa = np.asarray(
            self.parameters.after_spike_amplitudes_pa, dtype=np.float64
        )
        self._spike_cut_threshold_decay = np.exp(
            -self.parameters.threshold_spike_decay_per_s * (self.parameters.spike_cut_ms / 1_000.0)
        )
        # Integration workspaces are owned by the population. Reusing them is
        # important at the fitted 0.05-ms timestep, where allocation overhead
        # otherwise rivals the cell equations for every biological step.
        self._work_after_sum = np.empty(cell_count, dtype=np.float64)
        self._work_total_current = np.empty(cell_count, dtype=np.float64)
        self._work_next_voltage = np.empty(cell_count, dtype=np.float64)
        self._work_next_spike_threshold = np.empty(cell_count, dtype=np.float64)
        self._work_next_after = np.empty_like(self.state.after_spike_currents_pa)

    @property
    def cell_count(self) -> int:
        return self._cell_count

    @property
    def threshold_relative_mv(self) -> np.ndarray:
        p = self.parameters
        return (
            p.threshold_inf_relative_mv
            + self.state.threshold_spike_mv
            + self.state.threshold_voltage_mv
        )

    def reset(self) -> None:
        self.state.voltage_relative_mv.fill(0.0)
        self.state.after_spike_currents_pa.fill(0.0)
        self.state.threshold_spike_mv.fill(0.0)
        self.state.threshold_voltage_mv.fill(0.0)
        self.state.refractory_steps.fill(0)

    def compile_resting_response(
        self,
        step_count: int,
        current_pa: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compile subthreshold voltage and threshold under constant current.

        A resting homogeneous population is linear until its first spike. The
        returned unit response lets output races scale the fitted cell dynamics
        across cells without repeatedly integrating identical resting state.
        """
        if step_count < 1:
            raise ValueError("response needs at least one integration step")
        p = self.parameters
        voltage = 0.0
        voltage_threshold = 0.0
        response_voltage = np.empty(step_count, dtype=np.float64)
        response_threshold = np.empty(step_count, dtype=np.float64)
        total_current = float(current_pa)
        beta = total_current * self._threshold_beta_scale
        for step in range(step_count):
            previous_voltage = voltage
            voltage = previous_voltage + self._voltage_step_scale * (
                total_current - p.leak_ns * previous_voltage
            )
            voltage_threshold = (
                self._threshold_phi * (previous_voltage - beta) * self._threshold_exp_leak
                + self._threshold_exp_decay
                * (
                    voltage_threshold
                    - self._threshold_phi * (previous_voltage - beta)
                    - self._threshold_voltage_ratio * beta
                )
                + self._threshold_voltage_ratio * beta
            )
            response_voltage[step] = voltage
            response_threshold[step] = voltage_threshold
        return response_voltage, response_threshold

    def step(self, injected_current_pa: float | Sequence[float] | np.ndarray) -> np.ndarray:
        """Advance one fitted 0.05-ms step and return the cells that spike."""
        current = np.asarray(injected_current_pa, dtype=np.float64)
        if current.ndim == 0:
            current = np.full(self.cell_count, float(current), dtype=np.float64)
        if current.shape != (self.cell_count,):
            raise ValueError("injected current must be scalar or one value per cell")

        p = self.parameters
        state = self.state
        active = state.refractory_steps == 0
        np.subtract(
            state.refractory_steps,
            1,
            out=state.refractory_steps,
            where=~active,
        )
        if not active.any():
            return active

        previous_voltage = state.voltage_relative_mv
        previous_after = state.after_spike_currents_pa
        previous_after.sum(axis=1, out=self._work_after_sum)
        np.add(current, self._work_after_sum, out=self._work_total_current)
        total_current = self._work_total_current

        # Allen evaluates voltage and threshold from the same t0 state, then
        # advances the after-spike currents for t1.
        np.multiply(p.leak_ns, previous_voltage, out=self._work_next_voltage)
        np.subtract(total_current, self._work_next_voltage, out=self._work_next_voltage)
        np.multiply(
            self._voltage_step_scale,
            self._work_next_voltage,
            out=self._work_next_voltage,
        )
        np.add(previous_voltage, self._work_next_voltage, out=self._work_next_voltage)
        next_voltage = self._work_next_voltage

        beta = total_current * self._threshold_beta_scale
        previous_voltage_threshold = state.threshold_voltage_mv
        next_voltage_threshold = (
            self._threshold_exp_decay * previous_voltage_threshold
            + self._threshold_voltage_factor * previous_voltage
            + self._threshold_beta_factor * beta
        )
        np.multiply(
            state.threshold_spike_mv,
            self._spike_threshold_decay,
            out=self._work_next_spike_threshold,
        )
        next_spike_threshold = self._work_next_spike_threshold
        np.multiply(
            previous_after,
            self._after_spike_decay,
            out=self._work_next_after,
        )
        next_after = self._work_next_after

        np.copyto(state.voltage_relative_mv, next_voltage, where=active)
        np.copyto(state.threshold_voltage_mv, next_voltage_threshold, where=active)
        np.copyto(state.threshold_spike_mv, next_spike_threshold, where=active)
        np.copyto(
            state.after_spike_currents_pa,
            next_after,
            where=active[:, None],
        )

        crossed = active & (
            next_voltage
            > (p.threshold_inf_relative_mv + next_voltage_threshold + next_spike_threshold)
        )
        if not crossed.any():
            return crossed

        spike_rows = np.flatnonzero(crossed)
        state.voltage_relative_mv[spike_rows] = (
            p.voltage_reset_scale * next_voltage[spike_rows] + p.voltage_reset_offset_mv
        )
        state.after_spike_currents_pa[spike_rows] = (
            self._after_spike_amplitudes_pa + next_after[spike_rows] * self._after_spike_cut_decay
        )
        state.threshold_spike_mv[spike_rows] = (
            next_spike_threshold[spike_rows] * self._spike_cut_threshold_decay
            + p.threshold_spike_increment_mv
        )
        # Allen holds the voltage-dependent threshold component during a cut.
        state.refractory_steps[spike_rows] = p.spike_cut_steps
        return crossed


@dataclass(frozen=True)
class GlifTrace:
    time_ms: np.ndarray
    voltage_mv: np.ndarray
    spike_times_ms: tuple[float, ...]


@dataclass(frozen=True)
class AdaptiveGlifTrace:
    time_ms: np.ndarray
    voltage_mv: np.ndarray
    threshold_mv: np.ndarray
    spike_times_ms: tuple[float, ...]


def simulate_glif3_current(
    parameters: AllenGlif3Parameters,
    current_pa: float | Sequence[float],
    duration_ms: float | None = None,
) -> GlifTrace:
    """Run the pinned Allen GLIF3 equations for a current-clamp trace."""
    if np.isscalar(current_pa):
        if duration_ms is None or duration_ms <= 0.0:
            raise ValueError("positive duration_ms is required for scalar current")
        step_count = max(1, round(duration_ms / parameters.dt_ms))
        stimulus = np.full(step_count, float(current_pa), dtype=np.float64)
    else:
        stimulus = np.asarray(current_pa, dtype=np.float64)
        if stimulus.ndim != 1 or stimulus.size == 0:
            raise ValueError("current trace must be a non-empty vector")
        step_count = int(stimulus.size)

    voltage = np.empty(step_count, dtype=np.float64)
    relative_v = 0.0
    after = np.zeros(len(parameters.after_spike_taus_ms), dtype=np.float64)
    amps = np.asarray(parameters.after_spike_amplitudes_pa, dtype=np.float64)
    taus = np.asarray(parameters.after_spike_taus_ms, dtype=np.float64)
    asc_decay = np.exp(-parameters.dt_ms / taus)
    cut_decay = np.exp(-parameters.spike_cut_ms / taus)
    refractory = 0
    spikes: list[float] = []

    for step, injected in enumerate(stimulus):
        if refractory > 0:
            voltage[step] = np.nan
            refractory -= 1
            continue
        after *= asc_decay
        relative_v += (
            parameters.dt_ms
            / parameters.capacitance_pf
            * (float(injected) + float(after.sum()) - parameters.leak_ns * relative_v)
        )
        if relative_v > parameters.threshold_relative_mv:
            spikes.append((step + 1) * parameters.dt_ms)
            voltage[step] = np.nan
            relative_v = 0.0
            after = amps + after * cut_decay
            refractory = parameters.spike_cut_steps
        else:
            voltage[step] = parameters.rest_mv + relative_v
    time = np.arange(step_count, dtype=np.float64) * parameters.dt_ms
    return GlifTrace(time_ms=time, voltage_mv=voltage, spike_times_ms=tuple(spikes))


def simulate_glif5_current(
    parameters: AllenGlif5Parameters,
    current_pa: float | Sequence[float],
    duration_ms: float | None = None,
) -> AdaptiveGlifTrace:
    """Run a current clamp through the pinned adaptive human GLIF5 cell."""
    if np.isscalar(current_pa):
        if duration_ms is None or duration_ms <= 0.0:
            raise ValueError("positive duration_ms is required for scalar current")
        step_count = max(1, round(duration_ms / parameters.dt_ms))
        stimulus = np.full(step_count, float(current_pa), dtype=np.float64)
    else:
        stimulus = np.asarray(current_pa, dtype=np.float64)
        if stimulus.ndim != 1 or stimulus.size == 0:
            raise ValueError("current trace must be a non-empty vector")
        step_count = int(stimulus.size)

    population = AdaptiveGlifPopulation(1, parameters)
    voltage = np.full(step_count, np.nan, dtype=np.float64)
    threshold = np.full(step_count, np.nan, dtype=np.float64)
    spikes: list[float] = []
    for step, injected in enumerate(stimulus):
        was_refractory = bool(population.state.refractory_steps[0])
        fired = population.step(float(injected))[0]
        if fired:
            spikes.append(step * parameters.dt_ms)
        elif not was_refractory:
            voltage[step] = parameters.rest_mv + population.state.voltage_relative_mv[0]
            threshold[step] = parameters.rest_mv + population.threshold_relative_mv[0]
    time = np.arange(step_count, dtype=np.float64) * parameters.dt_ms
    return AdaptiveGlifTrace(
        time_ms=time,
        voltage_mv=voltage,
        threshold_mv=threshold,
        spike_times_ms=tuple(spikes),
    )
