"""FROZEN Stage-0 constants -- transcribed before implementation landed.

Provenance (both committed at e16d22c, i.e. before this file existed):

* docs/research/path-decision-synthesis.md, section
  "Probe P-A -- B' = APICAL-LMS Stage-0":
    - local apical update  du_A = alpha*u_A + gamma*e_t*x_t  riding observe();
      e_t = next-token prediction residual, gated OFF at decision points;
      u_A never reset by somatic spikes;
    - hand-set alpha ~= 0.99-class and gamma inside the stability bound
      gamma < 2/(d_eff + 2);
    - declare d_apical <= 384 jointly with gamma at freeze
      (width cliff: d_apical >= 512 diverges [DD-DCL SS1.3]);
    - gate / falsifier: linear-probe R^2 of the apical trajectory vs a
      reference online-LMS trajectory >= 0.85 (benchmark band 0.85-0.93,
      DD-DCL SS1.3). Below threshold -> B' dies at Stage 0.

* docs/research/deep-dive-dendricl-ctp.md SS1.1-1.3:
    - learned alpha -> 0.994-0.997; learned gamma ~= 0.2x the theoretical
      optimum gamma* = 1/(d+2), always inside gamma < 2/(d+2);
    - u_A initialized to zero, NOT reset by somatic spikes;
    - d_model = d_apical = 384 units -> cell_count = d_apical here;
    - Garg-2022 ICL benchmark shape: d_task = 10 primary / 20 secondary,
      context length k = 2d supervised steps, query positions flagged.

Any change to these values is a new preregistration, not an edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def gamma_star(d_eff: int) -> float:
    """Theoretical optimal leaky-LMS step size 1/(d_eff + 2) [DD-DCL SS1.2]."""
    return 1.0 / (float(d_eff) + 2.0)


def stability_bound(d_eff: int) -> float:
    """LMS stability bound gamma < 2/(d_eff + 2) [synthesis doc, DD-DCL SS4.1]."""
    return 2.0 / (float(d_eff) + 2.0)


D_APICAL_WIDTH_CLIFF = 384  # trains; >= 512 diverges (DD-DCL SS1.3)


@dataclass(frozen=True)
class FrozenConstants:
    """One immutable translation of the Stage-0 preregistration."""

    # ---- estimator constants (the freeze) ---------------------------------
    alpha: float = 0.99                    # "alpha ~= 0.99-class" (spec); DD-DCL learned 0.994-0.997
    d_apical: int = 384                    # declared jointly with gamma at freeze (width cliff <= 384)
    gamma: float = field(default=-1.0)     # -1 => derive as GAMMA_FRACTION * gamma*(d_eff)
    gamma_fraction: float = 0.2            # learned gamma ~= 0.2 x gamma* across all 18 configs
    r2_gate: float = 0.85                  # PRE-SET falsifier threshold (band 0.85-0.93)

    # ---- Garg-2022 task shape (DD-DCL SS1.1/SS1.3) -------------------------
    task_dim_default: int = 10             # Garg d=10 primary (d=20 secondary leg)
    context_mult: int = 2                  # k = 2d supervised steps
    query_len: int = 8                     # decision-point tail; error gated OFF there
    noise_sigma: float = 1.0               # y = w.x + eps, eps ~ N(0, sigma^2) (Prop-1 form)
    w_std: float = 1.0                     # w ~ N(0, w_std^2 I_d) (Garg prior)

    # ---- substrate constants (existing fitted GLIF5 class defaults) --------
    cell_count: int = 384                  # one unit per apical dimension (DD-DCL SS1.1)
    tau_a_ms: float = 100.0                # Ca2+ plateau timescale class (paper motivation)
    theta_p: float = 0.5                   # frozen APICAL-AND plateau default; untuned here
    g_gate: float = 0.6                    # idem
    gate_duration_ms: float = 100.0        # idem
    drive_scale: float = 1.0               # u_A delivered to the membrane unscaled

    # ---- token cadence (matches experiments/apical_probes stack) -----------
    dt_ms: float = 0.05
    token_interval_ms: float = 5.0         # -> token_steps = 100 substeps
    input_pulse_ms: float = 1.5            # -> pulse_steps = 30 substeps
    input_current_pa: float = 5_000.0      # RecurrentCircuitConfig default
    recurrent_tau_ms: float = 8.0          # -> recurrent_decay = exp(-dt/tau)
    elig_tau_ms_start: float = 80.0        # ELIGIBILITY_TAUS_MS = geomspace(80, 4000, 8)
    elig_tau_ms_stop: float = 4000.0
    elig_banks: int = 8
    vocab: int = 256                       # one byte per token (label channel)
    aff_fanout: int = 24                   # RecurrentCircuitConfig.input_fanout
    rec_fanout: int = 8                    # RecurrentCircuitConfig.recurrent_fanout
    rec_weight_mean_pa: float = 650.0      # RecurrentCircuitConfig.recurrent_current_pa
    rec_weight_std_pa: float = 65.0

    # ---- seeds --------------------------------------------------------------
    base_seed: int = 20260825
    wa_seed: int = 20260825                # W_A projection seed (frozen jointly)
    wiring_seed: int = 17                  # RecurrentCircuitConfig.random_seed default

    # ---- probe scorer -------------------------------------------------------
    ridge_lambda: float = 1e-3             # frozen ridge penalty (standardized features)

    # ---- label-byte channel -------------------------------------------------
    label_byte_scale: float = 16.0         # tanh squash scale for y -> byte
    byte_neutral: int = 128                # stream head / missing previous label
    byte_query: int = 129                  # decision-point sentinel byte

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly inside (0, 1)")
        if self.d_apical > D_APICAL_WIDTH_CLIFF:
            raise ValueError(
                f"width cliff violated: d_apical={self.d_apical} "
                f"exceeds the frozen ceiling 384 (DD-DCL SS1.3)"
            )
        if self.d_apical < 1:
            raise ValueError("d_apical must be positive")
        gamma = self.gamma
        if gamma == -1.0:
            # hand-set at gamma_fraction x gamma*, INSIDE the stability bound
            object.__setattr__(
                self, "gamma", self.gamma_fraction * gamma_star(self.d_apical)
            )
            gamma = self.gamma
        bound = stability_bound(self.d_apical)
        if not 0.0 < gamma < bound:
            raise ValueError(
                f"LMS stability bound violated: gamma={gamma!r} must satisfy "
                f"0 < gamma < 2/(d_eff+2) = {bound!r} at d_eff={self.d_apical}"
            )
        if self.cell_count < self.d_apical:
            raise ValueError("cell_count must host every apical dimension")
        if self.r2_gate <= 0.0 or self.r2_gate >= 1.0:
            raise ValueError("r2_gate must lie strictly inside (0, 1)")


FROZEN = FrozenConstants()
