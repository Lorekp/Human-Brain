"""Stage-0 runner: the local apical-LMS update riding observe().

Subclasses the stock APICAL-AND token harness
(experiments/apical_probes/two_comp_harness.py::ApicalTokenRunner) so the
native observe() semantics -- leading afferent pulse over ``pulse_steps``
substeps, per-token spike-count reset, eligibility/synaptic decays, recurrent
delivery order -- are inherited verbatim. On top of that:

  Per supervised position t (teacher-forced replay):
    1. drive  = drive_scale * u_A                 (pre-update hosted state)
    2. super().observe(token_byte, apical_u=drive); log apical voltage v_A
    3. reveal y_t; e_t = y_t - <u_A, W_A x_t>     (next-token residual)
    4. u_A <- alpha*u_A + (gamma*e_t)*(W_A x_t)   (local three-factor rule)
  At decision points (flag): step 4 is skipped entirely -- no e_t is formed
  from the hidden label, u_A and the reference state stay frozen.

Parity contract: with ``lms_enabled=False`` the runner performs EXACTLY the
parent's arithmetic -- set_apical_drive is never called with nonzero content,
no RNG draws are added, and no simulator array is touched outside the parent
path -- so runs are bit-identical to the stock harness (tested).

u_A lives OUTSIDE the population, updated only by the local rule from the
revealed stream; somatic spikes cannot reach it (tested bitwise).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from humanbrain.cells import TwoCompartmentGLIF5  # noqa: E402

_HARNESS_PATH = REPO / "experiments" / "apical_probes" / "two_comp_harness.py"
_SPEC = importlib.util.spec_from_file_location("two_comp_harness_stage0", _HARNESS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(_SPEC)
sys.modules["two_comp_harness_stage0"] = HARNESS
_SPEC.loader.exec_module(HARNESS)

from experiments.apical_lms_stage0 import garg_task, lms  # noqa: E402


@dataclass(frozen=True)
class EpisodeRun:
    """Logged trajectories for one episode (context AND query positions)."""

    v_traj: np.ndarray       # (T, cells) apical voltage AFTER each token window
    u_traj: np.ndarray       # (T, d_apical) hosted state BEFORE each token
    ref_traj: np.ndarray     # (T, d_task) reference online-LMS state BEFORE each token
    pred: np.ndarray         # (T,) hosted readout BEFORE each token (NaN never; queries keep their readout)
    spike_totals: np.ndarray # (T,) int64 somatic spikes during each token window


class Stage0Runner(HARNESS.ApicalTokenRunner):
    """Garg-task replay through TwoCompartmentGLIF5 with the local LMS ride-along."""

    def __init__(
        self,
        population: TwoCompartmentGLIF5,
        *,
        wiring: dict[str, np.ndarray],
        constants,
        task_dim: int,
        wa: np.ndarray,
        lms_enabled: bool,
        pop_params: dict,
        **parent_kwargs,
    ) -> None:
        super().__init__(population, wiring=wiring, **parent_kwargs)
        self.constants = constants
        self.task_dim = int(task_dim)
        self.wa = wa
        self.lms_enabled = bool(lms_enabled)
        self._pop_params = dict(pop_params)
        self._wiring_source = {key: value.copy() for key, value in wiring.items()}
        self.u_A = np.zeros(constants.d_apical, dtype=np.float64)
        self.ref_w = np.zeros(self.task_dim, dtype=np.float64)

    # ------------------------------------------------------------ construction

    @classmethod
    def build(
        cls,
        constants,
        *,
        task_dim: int,
        lms_enabled: bool,
        seed: int | None = None,
        wiring_seed: int | None = None,
        input_current_pa: float | None = None,
    ) -> "Stage0Runner":
        """Deterministically construct population + synthetic wiring + projections."""
        seed = constants.base_seed if seed is None else int(seed)
        wiring_seed = constants.wiring_seed if wiring_seed is None else int(wiring_seed)
        current = (
            constants.input_current_pa if input_current_pa is None else float(input_current_pa)
        )

        population = TwoCompartmentGLIF5(
            constants.cell_count,
            tau_a_ms=constants.tau_a_ms,
            theta_p=constants.theta_p,
            g_gate=constants.g_gate,
            gate_duration_ms=constants.gate_duration_ms,
        )

        rng = np.random.default_rng(wiring_seed)
        afferent_targets = np.empty((constants.vocab, constants.aff_fanout), dtype=np.int32)
        for token in range(constants.vocab):
            afferent_targets[token] = rng.choice(
                constants.cell_count, constants.aff_fanout, replace=False
            )
        recurrent_targets = np.empty((constants.cell_count, constants.rec_fanout), dtype=np.int32)
        for cell in range(constants.cell_count):
            recurrent_targets[cell] = rng.choice(
                constants.cell_count, constants.rec_fanout, replace=False
            )
        recurrent_weights_pa = rng.normal(
            constants.rec_weight_mean_pa, constants.rec_weight_std_pa,
            size=(constants.cell_count, constants.rec_fanout),
        )

        dt = constants.dt_ms
        elig_taus = np.geomspace(
            constants.elig_tau_ms_start, constants.elig_tau_ms_stop, constants.elig_banks
        )
        wiring = {
            "afferent_targets": afferent_targets,
            "recurrent_targets": recurrent_targets,
            "recurrent_weights_pa": recurrent_weights_pa,
        }
        wa = garg_task.build_wa(task_dim, constants.d_apical, seed=constants.wa_seed)

        return cls(
            population,
            wiring=wiring,
            constants=constants,
            task_dim=task_dim,
            wa=wa,
            lms_enabled=lms_enabled,
            pop_params=dict(
                cell_count=constants.cell_count,
                tau_a_ms=constants.tau_a_ms,
                theta_p=constants.theta_p,
                g_gate=constants.g_gate,
                gate_duration_ms=constants.gate_duration_ms,
            ),
            token_steps=max(1, round(constants.token_interval_ms / dt)),
            pulse_steps=max(1, round(constants.input_pulse_ms / dt)),
            input_current_pa=current,
            recurrent_decay=float(np.exp(-dt / constants.recurrent_tau_ms)),
            elig_decays=np.ascontiguousarray(np.exp(-(dt / elig_taus))),
        )

    # -------------------------------------------------------------- parity aids

    def harness_kwargs(self) -> dict:
        """Constructor kwargs reproducing this runner under the stock harness."""
        return {
            "wiring": {key: value.copy() for key, value in self._wiring_source.items()},
            "token_steps": self.token_steps,
            "pulse_steps": self.pulse_steps,
            "input_current_pa": self.input_current_pa,
            "recurrent_decay": self.recurrent_decay,
            "elig_decays": np.array(self._elig_decays.reshape(-1)),
        }

    def population_params(self) -> dict:
        return dict(self._pop_params)

    # -------------------------------------------------------------------- API

    def reset_state(self) -> None:
        """Full reset INCLUDING the hosted estimator (u_A starts at zero)."""
        self.reset()
        self.u_A.fill(0.0)
        self.ref_w.fill(0.0)

    def observe(self, token_id: int, apical_u: np.ndarray | None = None) -> None:
        """One teacher-forced token; the LMS-off path is the parent, verbatim."""
        if not self.lms_enabled:
            super().observe(token_id)  # apical_u=None -> parent leaves zeros
            return
        if apical_u is None:
            apical_u = self.constants.drive_scale * self.u_A
        super().observe(token_id, apical_u=apical_u)

    def run_episode(self, episode: garg_task.Episode) -> EpisodeRun:
        """Replay one episode; returns all logged trajectories."""
        self.reset_state()
        t_total = episode.n_positions
        v_traj = np.empty((t_total, self.population.cell_count), dtype=np.float64)
        u_traj = np.empty((t_total, self.constants.d_apical), dtype=np.float64)
        ref_traj = np.empty((t_total, self.task_dim), dtype=np.float64)
        pred = np.empty(t_total, dtype=np.float64)
        spikes = np.empty(t_total, dtype=np.int64)

        alpha = self.constants.alpha
        gamma = self.constants.gamma

        for t in range(t_total):
            # states as they enter position t (pre-update logging alignment)
            u_traj[t] = self.u_A
            ref_traj[t] = self.ref_w

            x_t = episode.xs[t]
            z_t = self.wa @ x_t if self.lms_enabled else None
            pred[t] = lms.predict_scalar(self.u_A, z_t) if z_t is not None else float("nan")

            self.observe(int(episode.token_bytes[t]))
            v_traj[t] = self.population.apical_voltage_mv[0]
            spikes[t] = int(self.token_spike_counts.sum())

            if episode.flags[t]:
                continue  # decision point: error gated OFF, both states frozen

            # reveal y_t -> two INDEPENDENT recursions on the same revealed stream:
            # the hosted apical state, and the classical reference online-LMS.
            if self.lms_enabled:
                e_hosted = float(episode.ys[t]) - pred[t]
                self.u_A = lms.apical_lms_update(self.u_A, z_t, e_hosted, alpha, gamma)
            e_ref = float(episode.ys[t]) - lms.predict_scalar(self.ref_w, x_t)
            self.ref_w = alpha * self.ref_w + (gamma * e_ref) * x_t

        return EpisodeRun(
            v_traj=v_traj,
            u_traj=u_traj,
            ref_traj=ref_traj,
            pred=pred,
            spike_totals=spikes,
        )
