"""Human-data-fitted Allen GLIF cell equations used by the language cortex."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanbrain.human_cell_identity import (
    HUMAN_GLIF3_IDENTITY,
    HUMAN_GLIF5_IDENTITY,
)
from humanbrain.native import NativeGlifKernel


@dataclass(frozen=True)
class AllenGlif3Parameters:
    """Pinned Allen human L2 GLIF3 parameters in mV/ms/pA/pF/nS.

    The source JSON uses volts, seconds, amps, farads, and ohms. This port
    retains the AllenSDK model's forward-Euler voltage equation, constant
    fitted threshold, exponential after-spike currents, zero-voltage reset,
    and spike-cut refractory interval.
    """

    model_id: int = HUMAN_GLIF3_IDENTITY.model_id
    specimen_id: int = HUMAN_GLIF3_IDENTITY.specimen_id
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
        if (
            self.model_id != HUMAN_GLIF3_IDENTITY.model_id
            or self.specimen_id != HUMAN_GLIF3_IDENTITY.specimen_id
        ):
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

    model_id: int = HUMAN_GLIF5_IDENTITY.model_id
    specimen_id: int = HUMAN_GLIF5_IDENTITY.specimen_id
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
        if (
            self.model_id != HUMAN_GLIF5_IDENTITY.model_id
            or self.specimen_id != HUMAN_GLIF5_IDENTITY.specimen_id
        ):
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


@dataclass(frozen=True, slots=True)
class FittedGlif5Constants:
    """One immutable translation of fitted GLIF5 parameters into step constants."""

    voltage_step_scale: float
    threshold_phi: float
    threshold_beta_scale: float
    threshold_exp_leak: float
    threshold_exp_decay: float
    threshold_voltage_ratio: float
    threshold_voltage_factor: float
    threshold_beta_factor: float
    spike_threshold_decay: float
    after_spike_decay: np.ndarray
    after_spike_cut_decay: np.ndarray
    after_spike_amplitudes_pa: np.ndarray
    spike_cut_threshold_decay: float

    @classmethod
    def from_parameters(
        cls, parameters: AllenGlif5Parameters
    ) -> FittedGlif5Constants:
        dt_s = parameters.dt_ms / 1_000.0
        leak_per_s = parameters.leak_ns / parameters.capacitance_pf * 1_000.0
        a_voltage = parameters.threshold_voltage_coupling_per_s
        b_voltage = parameters.threshold_voltage_decay_per_s
        threshold_phi = a_voltage / (b_voltage - leak_per_s)
        threshold_exp_leak = np.exp(-leak_per_s * dt_s)
        threshold_exp_decay = np.exp(-b_voltage * dt_s)
        threshold_voltage_ratio = a_voltage / b_voltage
        threshold_voltage_factor = threshold_phi * (
            threshold_exp_leak - threshold_exp_decay
        )
        threshold_beta_factor = (
            -threshold_voltage_factor
            + threshold_voltage_ratio * (1.0 - threshold_exp_decay)
        )
        after_taus = np.asarray(parameters.after_spike_taus_ms, dtype=np.float64)
        after_spike_decay = np.ascontiguousarray(
            np.exp(-parameters.dt_ms / after_taus), dtype=np.float64
        )
        after_spike_cut_decay = np.ascontiguousarray(
            np.exp(-parameters.spike_cut_ms / after_taus), dtype=np.float64
        )
        after_spike_amplitudes_pa = np.ascontiguousarray(
            parameters.after_spike_amplitudes_pa, dtype=np.float64
        )
        for values in (
            after_spike_decay,
            after_spike_cut_decay,
            after_spike_amplitudes_pa,
        ):
            values.setflags(write=False)
        return cls(
            voltage_step_scale=parameters.dt_ms / parameters.capacitance_pf,
            threshold_phi=threshold_phi,
            threshold_beta_scale=1.0 / parameters.leak_ns,
            threshold_exp_leak=threshold_exp_leak,
            threshold_exp_decay=threshold_exp_decay,
            threshold_voltage_ratio=threshold_voltage_ratio,
            threshold_voltage_factor=threshold_voltage_factor,
            threshold_beta_factor=threshold_beta_factor,
            spike_threshold_decay=np.exp(
                -parameters.threshold_spike_decay_per_s * dt_s
            ),
            after_spike_decay=after_spike_decay,
            after_spike_cut_decay=after_spike_cut_decay,
            after_spike_amplitudes_pa=after_spike_amplitudes_pa,
            spike_cut_threshold_decay=np.exp(
                -parameters.threshold_spike_decay_per_s
                * (parameters.spike_cut_ms / 1_000.0)
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
    """Native stateful simulator for homogeneous Allen GLIF5 cells."""

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
        self.constants = FittedGlif5Constants.from_parameters(self.parameters)
        # Integration workspaces are owned by the population. Reusing them is
        # important at the fitted 0.05-ms timestep, where allocation overhead
        # otherwise rivals the cell equations for every biological step.
        self._work_after_sum = np.empty(cell_count, dtype=np.float64)
        self._work_total_current = np.empty(cell_count, dtype=np.float64)
        self._work_next_voltage = np.empty(cell_count, dtype=np.float64)
        self._work_next_spike_threshold = np.empty(cell_count, dtype=np.float64)
        self._work_next_after = np.empty_like(self.state.after_spike_currents_pa)
        self._native_kernel = NativeGlifKernel(self)

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
        constants = self.constants
        beta = total_current * constants.threshold_beta_scale
        for step in range(step_count):
            previous_voltage = voltage
            voltage = previous_voltage + constants.voltage_step_scale * (
                total_current - p.leak_ns * previous_voltage
            )
            voltage_threshold = (
                constants.threshold_phi
                * (previous_voltage - beta)
                * constants.threshold_exp_leak
                + constants.threshold_exp_decay
                * (
                    voltage_threshold
                    - constants.threshold_phi * (previous_voltage - beta)
                    - constants.threshold_voltage_ratio * beta
                )
                + constants.threshold_voltage_ratio * beta
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
        return self._native_kernel.step(np.ascontiguousarray(current))


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


class TwoCompartmentGLIF5(AdaptiveGlifPopulation):
    """APICAL-AND two-compartment GLIF5 (preregistered; CPU path only).

    Frozen by ``docs/research/apical-and-preregistration.md`` @ 05fb3ef
    (committed before this class existed). The soma is a bit-exact NumPy
    replication of the native ``glif_step_collect`` equations over the fitted
    Allen constants; the new apical compartment is:

      (a) leaky integrator   ``v_a += (dt/tau_a) * (u_a - v_a)`` per step;
      (b) plateau: ``v_a > theta_p`` with ``gate_remaining == 0`` opens a
          somatic-threshold gate for ``round(gate_duration_ms/dt)`` steps
          (re-opening on the same test while the depolarization persists);
      (c) gated crossing test: ``next_v > total_threshold * (1 - g_gate)``
          wherever ``gate_remaining > 0`` — multiplicative lowering only;
      (d) ``gate_remaining[gate_remaining > 0] -= 1``.

    With zero apical drive every somatic array update is BIT-IDENTICAL to
    :class:`AdaptiveGlifPopulation` (flag-off parity tests). An optional
    leading arm axis lets one instance simulate the whole preregistered
    18-arm grid at once; per-arm results equal independent single-arm
    instances bitwise (tested).
    """

    def __init__(
        self,
        cell_count: int,
        parameters: AllenGlif5Parameters | None = None,
        *,
        tau_a_ms: float | Sequence[float] = 100.0,
        theta_p: float | Sequence[float] = 0.5,
        g_gate: float | Sequence[float] = 0.6,
        gate_duration_ms: float = 100.0,
        arm_count: int | None = None,
    ) -> None:
        grids = (tau_a_ms, theta_p, g_gate)
        lengths = {len(value) for value in grids if isinstance(value, (list, tuple))}
        if arm_count is None:
            if len(lengths) > 1:
                raise ValueError("compartment grids must have equal length")
            arm_count = lengths.pop() if lengths else 1
        if not isinstance(arm_count, int) or arm_count < 1:
            raise ValueError("arm_count must be a positive integer")

        def _vector(name: str, value: float | Sequence[float]) -> np.ndarray:
            array = np.asarray(value, dtype=np.float64)
            if array.ndim == 0:
                array = np.full(arm_count, float(array), dtype=np.float64)
            if array.shape != (arm_count,):
                raise ValueError(f"{name} must be scalar or length-{arm_count}")
            return np.ascontiguousarray(array)

        tau_vector = _vector("tau_a_ms", tau_a_ms)
        theta_vector = _vector("theta_p", theta_p)
        gate_vector = _vector("g_gate", g_gate)
        if not np.all(np.isfinite(tau_vector)) or np.any(tau_vector <= 0.0):
            raise ValueError("tau_a_ms must be finite and positive")
        if not np.all(np.isfinite(theta_vector)) or np.any(theta_vector <= 0.0):
            raise ValueError("theta_p must be finite and positive")
        if not np.all(np.isfinite(gate_vector)) or np.any(gate_vector < 0.0):
            raise ValueError("g_gate must be finite and nonnegative")
        if np.any(gate_vector >= 1.0):
            raise ValueError("g_gate must be below 1.0")
        if not (math.isfinite(gate_duration_ms) and gate_duration_ms > 0.0):
            raise ValueError("gate_duration_ms must be finite and positive")
        super().__init__(cell_count, parameters)
        dt = self.parameters.dt_ms
        self.arm_count = int(arm_count)
        # Re-allocate the inherited state with the leading arm axis. The
        # native kernel never binds these arrays (this subclass is CPU-path
        # only and driven through step()), so the shape change is safe.
        channels = len(self.parameters.after_spike_taus_ms)
        shape = (self.arm_count, cell_count)
        self.state = AdaptiveGlifState(
            voltage_relative_mv=np.zeros(shape, dtype=np.float64),
            after_spike_currents_pa=np.zeros(
                (self.arm_count, cell_count, channels), dtype=np.float64
            ),
            threshold_spike_mv=np.zeros(shape, dtype=np.float64),
            threshold_voltage_mv=np.zeros(shape, dtype=np.float64),
            refractory_steps=np.zeros(shape, dtype=np.int32),
        )
        self.tau_a_ms = tau_vector
        self.theta_p = theta_vector
        self.g_gate = gate_vector
        self.gate_duration_ms = float(gate_duration_ms)
        self.gate_steps = max(1, round(gate_duration_ms / dt))
        # Per-arm constants precomputed once so every consumer sees the same bits.
        self._apical_scale = np.ascontiguousarray((dt / tau_vector)[:, None])
        self._gate_drop = np.ascontiguousarray(1.0 - gate_vector)
        self.apical_voltage_mv = np.zeros((self.arm_count, cell_count), dtype=np.float64)
        self.gate_remaining = np.zeros((self.arm_count, cell_count), dtype=np.int32)
        self._apical_drive = np.zeros((self.arm_count, cell_count), dtype=np.float64)
        self._spike_mask = np.zeros((self.arm_count, cell_count), dtype=np.bool_)

    @classmethod
    def from_grid(
        cls,
        cell_count: int,
        *,
        tau_a_ms: Sequence[float],
        theta_p: Sequence[float],
        g_gate: Sequence[float],
        gate_duration_ms: float = 100.0,
        parameters: AllenGlif5Parameters | None = None,
    ) -> "TwoCompartmentGLIF5":
        """One batched population over the CARTESIAN-PRODUCT arm grid.

        Arm a of the output corresponds to
        ``itertools.product(tau_a_ms, theta_p, g_gate)`` position a.
        """
        import itertools

        combos = list(itertools.product(tau_a_ms, theta_p, g_gate))
        if not combos:
            raise ValueError("arm grid needs at least one combination")
        return cls(
            cell_count,
            parameters,
            tau_a_ms=[c[0] for c in combos],
            theta_p=[c[1] for c in combos],
            g_gate=[c[2] for c in combos],
            gate_duration_ms=gate_duration_ms,
            arm_count=len(combos),
        )

    def set_apical_drive(self, drive: np.ndarray | None) -> None:
        """Publish the per-token apical drive ``u_a`` (copied, shape (A,c))."""
        if drive is None:
            self._apical_drive.fill(0.0)
            return
        vector = np.asarray(drive, dtype=np.float64)
        if vector.ndim == 1 and self.arm_count == 1:
            vector = vector[None, :]
        if vector.shape != (self.arm_count, self.cell_count):
            raise ValueError("apical drive must have shape (arm_count, cell_count)")
        np.copyto(self._apical_drive, np.ascontiguousarray(vector))

    @property
    def cell_count(self) -> int:
        # Base property derives from voltage.size, which now includes the arm axis.
        return int(self._cell_count)

    def gate_multiplier(self) -> np.ndarray:
        """The multiplicative soma-threshold factor per cell right now."""
        out = np.ones((self.arm_count, self.cell_count), dtype=np.float64)
        open_rows = np.nonzero(self.gate_remaining > 0)
        out[open_rows] = self._gate_drop[open_rows[0]]
        return out

    def reset(self) -> None:
        super().reset()
        self.apical_voltage_mv.fill(0.0)
        self.gate_remaining.fill(0.0)
        self._apical_drive.fill(0.0)

    def _prepare_current(self, injected_current_pa) -> np.ndarray:
        current = np.asarray(injected_current_pa, dtype=np.float64)
        if current.ndim == 0:
            current = np.full(
                (self.arm_count, self.cell_count), float(current), dtype=np.float64
            )
        elif current.ndim == 1 and self.arm_count == 1:
            current = current[None, :]
        if current.shape != (self.arm_count, self.cell_count):
            raise ValueError(
                "injected current must be scalar or (arm_count, cell_count)"
            )
        return current

    def step(self, injected_current_pa) -> np.ndarray:
        """Advance every arm by one fitted 0.05 ms step; returns spike mask."""
        current = self._prepare_current(injected_current_pa)
        state = self.state
        constants = self.constants
        parameters = self.parameters

        # (a) apical Euler integration (frozen literal update order)
        delta = self._apical_drive - self.apical_voltage_mv
        delta *= self._apical_scale
        self.apical_voltage_mv += delta

        # (b) plateau opening on the frozen test
        opening = (self.apical_voltage_mv > self.theta_p[:, None]) & (
            self.gate_remaining == 0
        )
        self.gate_remaining[opening] = self.gate_steps

        # (c) soma: bit-exact replication of glif_step_collect for the
        # non-refractory cells; refractory cells only decrement.
        refractory = state.refractory_steps
        resting = refractory == 0
        blocked = ~resting
        if blocked.any():
            refractory[blocked] -= 1
        indices = np.nonzero(resting)
        if indices[0].size:
            rows, cols = indices
            previous_voltage = state.voltage_relative_mv[indices]
            after = state.after_spike_currents_pa[indices]
            after_sum = after[:, 0] + after[:, 1]
            total_current = current[indices] + after_sum
            leak = parameters.leak_ns * previous_voltage
            voltage_delta = constants.voltage_step_scale * (total_current - leak)
            next_voltage = previous_voltage + voltage_delta

            beta = total_current * constants.threshold_beta_scale
            voltage_term = constants.threshold_voltage_factor * previous_voltage
            beta_term = constants.threshold_beta_factor * beta
            decayed_threshold = constants.threshold_exp_decay * (
                state.threshold_voltage_mv[indices]
            )
            next_voltage_threshold = decayed_threshold + voltage_term + beta_term
            next_spike_threshold = (
                state.threshold_spike_mv[indices] * constants.spike_threshold_decay
            )

            total_threshold = (
                parameters.threshold_inf_relative_mv
                + next_voltage_threshold
                + next_spike_threshold
            )
            multiplier = np.ones(total_threshold.shape, dtype=np.float64)
            open_here = self.gate_remaining[indices] > 0
            if open_here.any():
                multiplier[open_here] = self._gate_drop[rows[open_here]]
            crossed = next_voltage > total_threshold * multiplier

            state.voltage_relative_mv[indices] = next_voltage
            state.threshold_voltage_mv[indices] = next_voltage_threshold
            state.threshold_spike_mv[indices] = next_spike_threshold
            state.after_spike_currents_pa[indices] = (
                after * constants.after_spike_decay[None, :]
            )
            mask = self._spike_mask
            mask.fill(False)
            mask[indices] = crossed
            crossed_indices = (rows[crossed], cols[crossed])
            if crossed_indices[0].size:
                state.voltage_relative_mv[crossed_indices] = (
                    parameters.voltage_reset_scale * next_voltage[crossed]
                    + parameters.voltage_reset_offset_mv
                )
                # Native order: the decay above is committed first, and the
                # spike reset reads the ALREADY-DECAYED after-current.
                after_crossed = state.after_spike_currents_pa[crossed_indices]
                state.after_spike_currents_pa[crossed_indices] = (
                    constants.after_spike_amplitudes_pa[None, :]
                    + after_crossed * constants.after_spike_cut_decay[None, :]
                )
                state.threshold_spike_mv[crossed_indices] = (
                    next_spike_threshold[crossed]
                    * constants.spike_cut_threshold_decay
                    + parameters.threshold_spike_increment_mv
                )
                refractory[crossed_indices] = parameters.spike_cut_steps
        else:
            self._spike_mask.fill(False)

        # (d) gate clock decrement
        self.gate_remaining -= self.gate_remaining > 0
        return self._spike_mask


def build_arm_grid() -> dict[str, dict[str, tuple]]:
    """The frozen 18-arm grid from the preregistration (§2)."""
    taus = (50.0, 100.0, 200.0)
    thetas = (0.25, 0.50, 0.75)
    gates = (0.30, 0.60)
    arms = {}
    for tau in taus:
        for theta in thetas:
            for gate in gates:
                name = f"A{tau:g}_{theta:g}_{gate:g}"
                arms[name] = {
                    "tau_a_ms": tau,
                    "theta_p": theta,
                    "g_gate": gate,
                }
    return arms
