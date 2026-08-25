"""APICAL-LMS Stage-0 tests (P-A execution-order step 2).

Preregistration: docs/research/path-decision-synthesis.md @ e16d22c,
section "Probe P-A -- B' = APICAL-LMS Stage-0"; benchmark-shape constants
transcribed from docs/research/deep-dive-dendricl-ctp.md SS1.1-1.3 BEFORE
the implementation landed (RED-first discipline, same pattern as
tests/test_two_compartment_glif5.py).

Covered here:
  * flag-off parity: the LMS-disabled Stage0Runner is BIT-IDENTICAL to the
    stock ApicalTokenRunner over the same wiring and token stream;
  * determinism: same seed -> bitwise-equal trajectories;
  * update-rule identity vs hand-computed traces (canonical op order);
  * hosted-lift identity u_A(t) == W_A @ w_ref(t);
  * error gating OFF at decision/query positions (u_A frozen there);
  * u_A never reset by somatic spikes (quiet vs spiking soma bitwise);
  * LMS stability-bound rejection gamma < 2/(d_eff + 2);
  * Garg task generator contract; probe scorer sanity; cohort-leg guard.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _entry in (str(REPO), str(REPO / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from experiments.apical_lms_stage0 import freeze, garg_task, lms  # noqa: E402
from experiments.apical_lms_stage0 import runner, scoring  # noqa: E402

FROZEN = freeze.FROZEN


def _load_harness():
    """Load the stock APICAL-AND token harness (parity baseline)."""
    import importlib.util

    path = REPO / "experiments" / "apical_probes" / "two_comp_harness.py"
    spec = importlib.util.spec_from_file_location("two_comp_harness_lms_parity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["two_comp_harness_lms_parity"] = module
    spec.loader.exec_module(module)
    return module


HARNESS = _load_harness()


def _digest(arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        h.update(b"|")
    return h.hexdigest()


def _snapshot(population) -> str:
    """Full bitwise snapshot of every persistent simulator array."""
    state = population.state
    return _digest(
        [
            state.voltage_relative_mv,
            state.after_spike_currents_pa,
            state.threshold_spike_mv,
            state.threshold_voltage_mv,
            state.refractory_steps.astype(np.int64),
            population.apical_voltage_mv,
            population.gate_remaining.astype(np.int64),
            population._apical_drive,
        ]
    )


def _runner_snapshot(r) -> str:
    return _snapshot(r.population) + "#" + _digest([r.synaptic_current_pa]) + _digest(
        [r.eligibility]
    ) + _digest([r.token_spike_counts.astype(np.int64)])


def _make_runner(*, lms_enabled: bool, input_current_pa: float | None = None, seed: int = 1234):
    kwargs = {}
    if input_current_pa is not None:
        kwargs["input_current_pa"] = input_current_pa
    return runner.Stage0Runner.build(
        FROZEN, task_dim=10, lms_enabled=lms_enabled, seed=seed, **kwargs
    )


def _spec() -> garg_task.TaskSpec:
    return garg_task.TaskSpec(
        d_task=10,
        context_mult=FROZEN.context_mult,
        query_len=4,
        noise_sigma=FROZEN.noise_sigma,
        w_std=FROZEN.w_std,
    )


def _make_episode(seed: int = 7):
    rng = np.random.default_rng(seed)
    return garg_task.generate_episode(rng, _spec())


# ------------------------------------------------------------------- freeze


def test_frozen_constants_match_synthesis_spec():
    """Gate + constants frozen before implementation (synthesis doc @ e16d22c)."""
    assert FROZEN.alpha == 0.99
    assert FROZEN.d_apical == 384
    assert FROZEN.cell_count == 384  # one unit per apical dimension (DD-DCL SS1.1)
    assert FROZEN.r2_gate == 0.85  # benchmark band 0.85-0.93 (DD-DCL SS1.3)
    assert FROZEN.task_dim_default == 10
    # gamma hand-set at ~0.2x the theoretical optimum (DD-DCL SS1.2), strictly
    # inside the frozen stability bound gamma < 2/(d_eff + 2).
    assert FROZEN.gamma == pytest.approx(0.2 * freeze.gamma_star(FROZEN.d_apical))
    assert 0.0 < FROZEN.gamma < freeze.stability_bound(FROZEN.d_apical)


def test_stability_bound_rejection():
    bound = freeze.stability_bound(FROZEN.d_apical)
    with pytest.raises(ValueError, match="stability"):
        freeze.FrozenConstants(gamma=bound)  # strict inequality: equality rejects
    with pytest.raises(ValueError, match="stability"):
        freeze.FrozenConstants(gamma=bound * 1.001)
    with pytest.raises(ValueError, match="stability"):
        freeze.FrozenConstants(gamma=1e-2)
    with pytest.raises(ValueError, match="stability"):
        freeze.FrozenConstants(gamma=0.0)
    with pytest.raises(ValueError, match="stability"):
        freeze.FrozenConstants(gamma=-1e-6)
    # inside-bound acceptance and width-cliff rejection (d_apical <= 384)
    freeze.FrozenConstants(gamma=FROZEN.gamma)
    with pytest.raises(ValueError, match="384"):
        freeze.FrozenConstants(d_apical=512, gamma=0.2 * freeze.gamma_star(512))


# -------------------------------------------------------------- flag-off parity


def test_flagoff_parity_bit_identical_direct_observe_loop():
    """THE red-first gate: LMS off => bit-identical to the stock harness."""
    r = _make_runner(lms_enabled=False)
    toks = np.random.default_rng(99).integers(0, FROZEN.vocab, size=12)

    for tok in toks:
        r.observe(int(tok))
    got = _runner_snapshot(r)

    pop2 = type(r.population)(**r.population_params())
    plain = HARNESS.ApicalTokenRunner(pop2, **r.harness_kwargs())
    for tok in toks:
        plain.observe(int(tok))
    want = _runner_snapshot(plain)

    assert got == want


def test_flagoff_parity_bit_identical_over_full_episode():
    """Same gate driven through run_episode (token bytes + pulse schedule)."""
    r = _make_runner(lms_enabled=False)
    episode = _make_episode()
    r.run_episode(episode)
    got = _runner_snapshot(r)

    pop2 = type(r.population)(**r.population_params())
    plain = HARNESS.ApicalTokenRunner(pop2, **r.harness_kwargs())
    for byte in episode.token_bytes:
        plain.observe(int(byte))
    want = _runner_snapshot(plain)

    assert got == want


# ------------------------------------------------------------------ determinism


def test_determinism_same_seed_bitwise_identical():
    episodes_a = [_make_episode(s) for s in (7, 8, 9)]
    episodes_b = [_make_episode(s) for s in (7, 8, 9)]

    ra = _make_runner(lms_enabled=True, seed=1234)
    rb = _make_runner(lms_enabled=True, seed=1234)
    digests = []
    for run in (ra, rb):
        pieces = []
        for ep in episodes_a if run is ra else episodes_b:
            out = run.run_episode(ep)
            pieces.extend(
                [
                    out.v_traj,
                    out.u_traj,
                    out.ref_traj,
                    out.pred,
                    out.spike_totals.astype(np.float64),
                ]
            )
        digests.append(_digest(pieces))
    assert digests[0] == digests[1]


# --------------------------------------------- update-rule identity (hand traces)


def test_update_rule_matches_hand_computed_trace_bitwise():
    """Canonical grouping u += (gamma*e)*z after alpha*u, np.dot readout."""
    rng = np.random.default_rng(5)
    u0 = rng.normal(size=FROZEN.d_apical) * 0.1
    z = rng.normal(size=FROZEN.d_apical)
    e = 0.737

    # hand-computed, literal op order
    expected = FROZEN.alpha * u0 + (FROZEN.gamma * e) * z
    got = lms.apical_lms_update(u0, z, e, FROZEN.alpha, FROZEN.gamma)
    assert got.tobytes() == expected.tobytes()

    # chained multi-step hand trace
    u_hand = u0.copy()
    u_mod = u0.copy()
    for k in range(25):
        ek = 0.5 * np.sin(k) + 0.1
        zk = rng.normal(size=FROZEN.d_apical) * (1.0 + 0.01 * k)
        u_hand = FROZEN.alpha * u_hand + (FROZEN.gamma * ek) * zk
        u_mod = lms.apical_lms_update(u_mod, zk, ek, FROZEN.alpha, FROZEN.gamma)
    assert u_mod.tobytes() == u_hand.tobytes()

    assert lms.predict_scalar(u0, z) == float(np.dot(u0, z))


def test_reference_recursion_matches_hand_computed_trace_bitwise():
    """Logged trajectories equal literal re-derivations (reference AND hosted)."""
    episode = _make_episode()
    r = _make_runner(lms_enabled=True, seed=42)
    out = r.run_episode(episode)

    alpha, gamma = FROZEN.alpha, FROZEN.gamma

    # --- reference classical online-LMS, literal re-derivation
    spec = _spec()
    w = np.zeros(spec.d_task, dtype=np.float64)
    ref_hand = []
    for t in range(episode.flags.size):
        ref_hand.append(w.copy())
        if episode.flags[t]:
            continue  # error gated OFF at decision points
        e = episode.ys[t] - float(np.dot(w, episode.xs[t]))
        w = alpha * w + (gamma * e) * episode.xs[t]
    assert out.ref_traj.tobytes() == np.asarray(ref_hand).tobytes()

    # --- hosted readout through the W_A basis, literal re-derivation
    u = np.zeros(FROZEN.d_apical, dtype=np.float64)
    pred_hand = []
    for t in range(episode.flags.size):
        z = r.wa @ episode.xs[t]
        pred_hand.append(float(np.dot(u, z)))  # readout happens at EVERY position
        if episode.flags[t]:
            continue  # update gated OFF
        e = episode.ys[t] - float(np.dot(u, z))
        u = alpha * u + (gamma * e) * z
    assert out.pred.tobytes() == np.asarray(pred_hand).tobytes()


def test_hosted_state_is_exact_lift_of_reference():
    """u_A(t) == W_A @ w_ref(t) throughout an episode (orthonormal W_A)."""
    episode = _make_episode()
    r = _make_runner(lms_enabled=True, seed=42)
    out = r.run_episode(episode)

    lifted = out.ref_traj @ r.wa.T  # (T, d_apical)
    assert float(np.abs(out.u_traj - lifted).max()) <= 1e-9


# ------------------------------------------------------- gating at decision points


def test_error_gated_off_at_decision_points():
    """Query positions freeze u_A and the reference state (no update at all)."""
    episode = _make_episode()
    r = _make_runner(lms_enabled=True, seed=42)
    out = r.run_episode(episode)

    q0 = int(np.flatnonzero(episode.flags)[0])
    # the state entering the FIRST decision point survives to the end unchanged:
    # no update fires at any flagged position (error gated OFF)
    assert out.u_traj[q0].tobytes() == out.u_traj[-1].tobytes()
    assert out.ref_traj[q0].tobytes() == out.ref_traj[-1].tobytes()
    # ...while the supervised phase strictly evolved the state
    assert not np.array_equal(out.u_traj[0], out.u_traj[q0])


def test_u_not_reset_by_somatic_spikes():
    """Quiet vs heavily spiking soma: identical apical trajectories, bitwise."""
    episode = _make_episode()
    spiky = _make_runner(lms_enabled=True, seed=77, input_current_pa=FROZEN.input_current_pa)
    quiet = _make_runner(lms_enabled=True, seed=77, input_current_pa=30.0)

    out_s = spiky.run_episode(episode)
    out_q = quiet.run_episode(episode)

    assert out_s.spike_totals.sum() > 0, "spiky arm must actually spike (guard vacuity)"
    assert out_q.spike_totals.sum() == 0, "quiet arm must stay subthreshold"
    assert out_s.v_traj.tobytes() == out_q.v_traj.tobytes()
    assert out_s.u_traj.tobytes() == out_q.u_traj.tobytes()
    assert out_s.ref_traj.tobytes() == out_q.ref_traj.tobytes()


# ------------------------------------------------------------ Garg task generator


def test_garg_generator_contract_and_determinism():
    spec = _spec()
    ep_a = _make_episode(11)
    ep_b = _make_episode(11)
    t_total = spec.context_mult * spec.d_task + spec.query_len

    assert ep_a.xs.shape == (t_total, spec.d_task)
    assert ep_a.ys.shape == (t_total,)
    assert ep_a.flags.shape == (t_total,)
    assert ep_a.token_bytes.shape == (t_total,)
    assert ep_a.w.shape == (spec.d_task,)
    assert ep_a.flags.dtype == np.bool_
    assert ep_a.token_bytes.dtype == np.uint8
    assert not ep_a.flags[:-spec.query_len].any()
    assert ep_a.flags[-spec.query_len :].all()
    assert np.isfinite(ep_a.ys[~ep_a.flags]).all()
    assert np.isnan(ep_a.ys[ep_a.flags]).all()
    # labels reconstruct as y = w.x + eps (within tolerance for stored eps)
    recon = ep_a.xs[~ep_a.flags] @ ep_a.w
    assert np.abs(recon - ep_a.ys[~ep_a.flags]).max() <= spec.noise_sigma * 6.0
    # determinism
    assert ep_a.xs.tobytes() == ep_b.xs.tobytes()
    assert ep_a.ys.tobytes() == ep_b.ys.tobytes()
    assert ep_a.w.tobytes() == ep_b.w.tobytes()
    assert ep_a.token_bytes.tobytes() == ep_b.token_bytes.tobytes()

    wa = garg_task.build_wa(spec.d_task, FROZEN.d_apical, seed=FROZEN.wa_seed)
    assert wa.shape == (FROZEN.d_apical, spec.d_task)
    assert np.allclose(wa.T @ wa, np.eye(spec.d_task), atol=1e-10)


# ------------------------------------------------------------------ probe scorer


def test_probe_recovers_reference_from_hosted_state():
    """Scorer sanity WITHOUT the membrane: u_A itself probed vs reference."""
    episodes = [_make_episode(s) for s in (21, 22, 23, 24, 25, 26)]
    r = _make_runner(lms_enabled=True, seed=55)
    feats, targets = [], []
    for ep in episodes:
        out = r.run_episode(ep)
        feats.append(out.u_traj)
        targets.append(out.ref_traj)
    result = scoring.probe_and_score(
        np.concatenate(feats), np.concatenate(targets),
        n_test_rows=2 * episodes[0].flags.size, ridge_lambda=FROZEN.ridge_lambda,
    )
    assert result.mean_r2 > 0.99  # exact linear lift must be recovered ~perfectly
    assert result.per_dim_r2.size == _spec().d_task


def test_smoke_verdict_end_to_end_ready_to_fire():
    """Tiny fixture leg completes and yields a serializable verdict."""
    episodes = [_make_episode(s) for s in range(31, 36)]
    r = _make_runner(lms_enabled=True, seed=56)
    verdict = scoring.evaluate(episodes, r, n_test_episodes=2)
    assert isinstance(verdict.passed, bool)
    assert verdict.gate == FROZEN.r2_gate
    assert np.isfinite(verdict.mean_r2)
    payload = scoring.verdict_to_dict(verdict)
    json_text = __import__("json").dumps(payload)
    assert str(FROZEN.r2_gate) in json_text


# ------------------------------------------------------------- corpus-leg guard


def test_corpus_cohort_guard_blocks_without_retro_probe_license():
    import experiments.apical_lms_stage0.run_stage0 as entry

    with pytest.raises(SystemExit):
        entry.enforce_cohort_guard(cohort=True, license_token=None)
    entry.enforce_cohort_guard(cohort=False, license_token=None)  # no raise
    entry.enforce_cohort_guard(cohort=True, license_token="retro-probe:PASS")  # no raise
