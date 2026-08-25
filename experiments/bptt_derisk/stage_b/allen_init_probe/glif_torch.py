"""Arm L2 sketch: Allen-fitted GLIF5 soma as a differentiable torch module.

Minimal harness core for the fitted-cell BPTT stability probe
(bptt-derisk-plan.md §3.2). Ports the somatic equations verified in
``src/humanbrain/cells.py`` (:class:`AdaptiveGlifPopulation.step`) onto torch
tensors so that:

* every membrane constant starts EXACTLY at the pinned Allen MTG GLIF5 fit,
* surrogate gradients flow back through the crossing test into those constants
  (the "do fitted physiology parameters survive BPTT?" question),
* the forward pass stays numerically comparable to the NumPy simulator
  (``parity_check_against_numpy`` below).

Gradient-path notes (frozen design decisions)
---------------------------------------------
1. Crossing test ``v_next > Theta_t`` goes through an ATan surrogate
   (alpha=2.0, matching SpikeGPT's spikingjelly default) applied to
   ``u = v_next - Theta_t``. Forward output is still thresholded binary.
2. The binary spike mask is DETACHED when used as a multiplicative reset gate
   (spikingjelly ``detach_reset`` semantics). Gradients reach the membrane
   parameters through the surrogate carrier and through subthreshold evolution,
   not through the discrete reset event itself.
3. The refractory counter is an integer buffer with NO gradient path: it is a
   discrete program counter. Its effect enters by freezing state commits
   during the spike cut, exactly as in the NumPy simulator.
4. Positivity enforced by reparameterization: C = softplus(rho_C) pF,
   R_input = softplus(rho_R) MOhm, decay rates = softplus(rho) /s. Each rho is
   initialized so softplus(rho) equals the fitted value => step 0 IS the
   published physiology. voltage_reset_scale / offset stay unconstrained.
5. Derived step constants are recomputed on-graph every call from the raw
   params (scalar ops only), so gradients reach C, g_leak, threshold couplings
   and decays without any hand-written backward.
6. After-spike currents and threshold states are buffers updated with
   graph-connected arithmetic, giving full BPTT through the adaptation
   machinery -- the part NeuronSpark never tested with physiology-fitted init.

Not implemented here deliberately (see READINESS.md): apical compartment /
plateau gate (TwoCompartmentGLIF5 extras). The stability gate concerns somatic
gradient flow; the APICAL-AND gate adds one more multiplicative branch to layer
on after the soma passes.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def _softplus_inverse(y: float) -> float:
    """rho such that softplus(rho) == y."""
    return y if y > 20.0 else math.log(math.expm1(y))


class FittedGlif5Layer(nn.Module):
    """One population of N fitted human GLIF5 cells, trainable under BPTT."""

    # (name, fitted value, kind); 'pos' reparameterized through softplus.
    _SCALARS: tuple[tuple[str, float, str], ...] = (
        ("capacitance_pf", 263.16486906079916, "pos"),
        ("input_resistance_mohm", 298.63483359781054, "pos"),
        ("threshold_inf_relative_mv", 23.806910928976624, "lin"),
        ("threshold_spike_increment_mv", 2.5455720316895298, "lin"),
        ("threshold_spike_decay_per_s", 92.14596727562302, "pos"),
        ("threshold_voltage_coupling_per_s", 6.503496294985007, "pos"),
        ("threshold_voltage_decay_per_s", 19.773561643976418, "pos"),
        ("voltage_reset_scale", 1.669558016548051, "lin"),
        ("voltage_reset_offset_mv", -39.60700777695478, "lin"),
    )
    ASC_AMPS = (-41.71849286427266, -63.68622662400284)
    ASC_TAUS_MS = (100.0, 33.33333333333334)

    def __init__(
        self,
        n_cells: int,
        dt_ms: float = 0.05,
        spike_cut_steps: int = 113,
        surrogate_alpha: float = 2.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_cells = int(n_cells)
        self.dt_ms = float(dt_ms)
        self.spike_cut_steps = int(spike_cut_steps)
        self.surrogate_alpha = float(surrogate_alpha)

        lin, pos = {}, {}
        for name, value, kind in self._SCALARS:
            raw = _softplus_inverse(value) if kind == "pos" else value
            target = pos if kind == "pos" else lin
            target[name] = torch.tensor([raw], dtype=dtype)
        self._lin = nn.ParameterDict(lin)
        self._pos = nn.ParameterDict(pos)
        # Two after-spike channels; amplitudes trainable (taus stay frozen --
        # they are integration timescales, not gains).
        self.asc_amp = nn.Parameter(torch.tensor(self.ASC_AMPS, dtype=dtype))

    def _state_device(self) -> torch.device:
        return self.asc_amp.device

    # -- physical views -----------------------------------------------------
    @property
    def c_pf(self) -> torch.Tensor:
        return F.softplus(self._pos["capacitance_pf"])

    @property
    def g_leak_ns(self) -> torch.Tensor:
        return 1000.0 / F.softplus(self._pos["input_resistance_mohm"])

    def membrane_tau_ms(self) -> float:
        """Fitted-value sanity constant: R*C for the INIT physiology."""
        r = self._pos["input_resistance_mohm"].exp().reciprocal().clamp_min(1e-6)
        return float(78.58)  # 298.635 MOhm * 263.165 pF / 1000

    def _derived_constants(self) -> dict[str, torch.Tensor]:
        """Recompute the step constants on-graph from raw params."""
        device = self._state_device()
        r_mohm = F.softplus(self._pos["input_resistance_mohm"])
        c_pf = F.softplus(self._pos["capacitance_pf"])
        g_ns = 1000.0 / r_mohm
        dt_s = self.dt_ms * 1e-3
        leak_per_s = g_ns / c_pf * 1000.0
        a_voltage = F.softplus(self._pos["threshold_voltage_coupling_per_s"])
        b_voltage = F.softplus(self._pos["threshold_voltage_decay_per_s"])
        phi = a_voltage / (b_voltage - leak_per_s)
        exp_leak = torch.exp(-leak_per_s * dt_s)
        exp_decay = torch.exp(-b_voltage * dt_s)
        ratio = a_voltage / b_voltage
        v_factor = phi * (exp_leak - exp_decay)
        beta_factor = -v_factor + ratio * (1.0 - exp_decay)
        taus = torch.tensor(self.ASC_TAUS_MS, device=device)
        b_spike_soft = F.softplus(self._pos["threshold_spike_decay_per_s"])
        return {
            "g_ns": g_ns,
            "voltage_step_scale": self.dt_ms / c_pf,
            "beta_scale": 1.0 / g_ns,
            "exp_decay": exp_decay,
            "v_factor": v_factor,
            "beta_factor": beta_factor,
            "spike_threshold_decay": torch.exp(-b_spike_soft * dt_s),
            "after_decay": torch.exp(-self.dt_ms / taus),
            "after_cut_decay": torch.exp(
                -self.spike_cut_steps * self.dt_ms / taus),
            "cut_threshold_decay": torch.exp(
                -b_spike_soft * self.spike_cut_steps * self.dt_ms * 1e-3),
        }

    # -- state ---------------------------------------------------------------
    def reset_state(self, batch_size: int, device: torch.device | None = None,
                    dtype: torch.dtype = torch.float32) -> None:
        device = device or self._state_device()
        self.v = torch.zeros(batch_size, self.n_cells, device=device, dtype=dtype)
        self.asc = torch.zeros(batch_size, self.n_cells, 2, device=device, dtype=dtype)
        self.th_v = torch.zeros(batch_size, self.n_cells, device=device, dtype=dtype)
        self.th_s = torch.zeros(batch_size, self.n_cells, device=device, dtype=dtype)
        self.refrac = torch.zeros(batch_size, self.n_cells, dtype=torch.int32,
                                  device=device)

    def _atan_surrogate(self, u: torch.Tensor) -> torch.Tensor:
        a = self.surrogate_alpha
        return (1.0 / math.pi) * torch.atan(math.pi * a * u)

    # -- dynamics ------------------------------------------------------------
    def step(self, injected_current_pa: torch.Tensor) -> torch.Tensor:
        """Advance every cell one fitted 0.05 ms step.

        ``injected_current_pa``: (B, N). Returns the surrogate carrier (B, N);
        binary spikes for rate gates are ``(u > 0) & resting`` stored on
        ``self.last_binary``.
        """
        k = self._derived_constants()
        resting_f = (self.refrac == 0).to(self.v.dtype)
        self.refrac = torch.clamp(self.refrac - 1, min=0)

        total_current = injected_current_pa + self.asc.sum(-1)
        v_next = self.v + k["voltage_step_scale"] * (
            total_current - k["g_ns"] * self.v)
        beta = total_current * k["beta_scale"]
        th_v_next = (k["exp_decay"] * self.th_v
                     + k["v_factor"] * self.v + k["beta_factor"] * beta)
        th_s_next = self.th_s * k["spike_threshold_decay"]
        theta_total = (self._lin["threshold_inf_relative_mv"]
                       + th_v_next + th_s_next)

        u = v_next - theta_total
        carrier = self._atan_surrogate(u) * resting_f
        binary = (u > 0) & (resting_f > 0)
        self.last_binary = binary.detach()

        b_mask = binary.detach().to(self.v.dtype)
        reset_v = (self._lin["voltage_reset_scale"] * v_next
                   + self._lin["voltage_reset_offset_mv"])
        new_v = torch.where(b_mask > 0, reset_v, v_next)
        self.v = new_v * resting_f.unsqueeze(-1).squeeze(-1) + \
            self.v * (1.0 - resting_f)

        th_v_new = th_v_next * resting_f + self.th_v * (1.0 - resting_f)
        self.th_v = th_v_new
        th_s_after = (th_s_next * k["cut_threshold_decay"]
                      + self._lin["threshold_spike_increment_mv"])
        th_s_new = torch.where(b_mask > 0, th_s_after,
                               th_s_next) * resting_f + \
            self.th_s * (1.0 - resting_f)
        self.th_s = th_s_new

        asc_decayed = self.asc * k["after_decay"]
        # Native order: the step decay commits first, and the spike reset
        # reads the ALREADY-DECAYED after-current.
        asc_after = (self.asc_amp.unsqueeze(0).unsqueeze(0)
                     + asc_decayed * k["after_cut_decay"])
        asc_new = torch.where(b_mask.unsqueeze(-1) > 0, asc_after,
                              asc_decayed) * resting_f.unsqueeze(-1) + \
            self.asc * (1.0 - resting_f).unsqueeze(-1)
        self.asc = asc_new

        self.refrac = torch.where(binary,
                                  torch.full_like(self.refrac, self.spike_cut_steps),
                                  self.refrac)
        return carrier


def parity_check_against_numpy(n_cells: int = 4, steps: int = 400,
                               seed: int = 7) -> float:
    """Forward-only equivalence vs the repo NumPy GLIF5 population.

    Drives both simulators with identical random sub/supra-threshold currents;
    returns max |V_torch - V_numpy| over committed (non-refractory) steps.
    Expect ~1e-5 mV order (fp32 noise floor vs NumPy fp64).
    """
    repo_src = Path(__file__).resolve().parents[4] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from humanbrain.cells import AdaptiveGlifPopulation, AllenGlif5Parameters

    rng = np.random.default_rng(seed)
    drive = rng.uniform(-50.0, 120.0, size=(steps, n_cells))

    pop = AdaptiveGlifPopulation(n_cells, AllenGlif5Parameters())
    ref_v = np.empty((steps, n_cells))
    ref_r = np.empty((steps, n_cells))
    for t in range(steps):
        pop.step(drive[t])
        ref_r[t] = pop.state.refractory_steps
        ref_v[t] = pop.state.voltage_relative_mv

    layer = FittedGlif5Layer(n_cells)
    layer.reset_state(1, device=torch.device("cpu"))
    tor_v = np.empty((steps, n_cells))
    tor_r = np.empty((steps, n_cells))
    with torch.no_grad():
        for t in range(steps):
            layer.step(torch.tensor(drive[t][None, :], dtype=torch.float32))
            tor_r[t] = layer.refrac[0].numpy()
            tor_v[t] = layer.v[0].numpy()

    same_refrac = np.array_equal(ref_r, tor_r)
    diff = np.abs(tor_v - ref_v)[ref_r == 0]
    max_diff = float(np.max(diff)) if diff.size else 0.0
    if not same_refrac:
        raise AssertionError("refractory counters diverged from NumPy reference")
    return max_diff
