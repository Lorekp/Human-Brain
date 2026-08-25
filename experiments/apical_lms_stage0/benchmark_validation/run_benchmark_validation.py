#!/usr/bin/env python3
"""Benchmark validation: Stage-0 harness vs DendriCL's published Garg-task numbers.

NON-DECISIVE calibration legs (fixture-class, no --cohort, no license): these
runs measure how our fitted-GLIF5 Stage-0 harness compares to the DendriCL
paper's reported curve at their PUBLISHED operating points
(docs/research/deep-dive-dendricl-ctp.md SS1.3), and map the apical width
response empirically. The corpus cohort leg and its 0.85 gate verdict remain
owned by the licensed leg (run_stage0.py --cohort); nothing here consumes that
decision. freeze.py is NOT edited: every non-frozen operating point here is a
locally-declared variant of the frozen dataclass.

Gamma arms (why three):
  frozen      alpha=0.99, gamma = 0.2 * gamma*(d_apical)      [the freeze]
  gstar_a99   alpha=0.99, gamma = 0.2 * gamma*(d_task)       [DendriCL
              transcription: their Prop-1 bound gamma < 2/d and optimum
              gamma* = 1/(d+2) are stated w.r.t. the TASK dimension d of x_t
              (DD-DCL SS1.1/1.2), not the 384-unit embedding width; with the
              orthonormal-lift W_A the update lives entirely on a d_task-dim
              subspace, so the task-dim reading is the mechanistically matched
              one. Recorded empirically; the freeze stays untouched.]
  gstar_a996  alpha=0.996, gamma = 0.2 * gamma*(d_task)      [adds their
              learned-alpha class 0.994-0.997 -> closest published pair]

Three R^2 layers are recorded at every operating point:

  r2_ref      task-level R^2 of the abstract reference online-LMS recursion,
              read out directly -- the idealized-linear-recurrence ceiling with
              the arm's constants;
  r2_decoded  task-level R^2 when the estimator is DECODED FROM THE MEMBRANE:
              ridge probe fit on TRAIN episodes' context rows (v_A -> w_ref),
              applied to TEST episodes' query-position voltages. The honest
              fitted-GLIF5 number (physiology included);
  probe_r2    held-out trajectory fidelity R^2 (v_A rows -> reference states;
              split BY EPISODE), the Stage-0 gate metric, comparable to
              DendriCL's published apical-probe column
              (0.873 @d10 / 0.931 @d20 / 0.97 @d30).

Task-level R^2 convention: pooled squared error of predictions against the
NOISELESS target w.x_t at held-out query positions (labels there are
unrevealed by construction; Garg-2022 normalized-error convention).

Width sweep: d_apical = cell_count = width in {10, 32, 128, 384} ("one unit
per apical dimension", freeze.py) under two gamma arms:
  joint   gamma derived from d_apical (the preregistered joint declaration);
  gstar   gamma pinned at 0.2 * gamma*(d_task=10) -- isolates the pure width
          response at a fixed, in-subspace-stable step size.
Mechanical adaptation: fanouts are min(frozen, cell_count) because a fanout
cannot exceed the population size. Everything else frozen-derived.

Usage: python experiments/apical_lms_stage0/benchmark_validation/run_benchmark_validation.py
       [--only NAME ...] [--force]   # resumable: existing result JSONs skipped

Deterministic: fixed seeds; every leg starts from a fresh runner + generator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

STAGE0 = Path(__file__).resolve().parents[1]
REPO = STAGE0.resolve().parents[1]
for _entry in (str(REPO), str(REPO / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from experiments.apical_lms_stage0 import freeze, garg_task, runner, scoring  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Published DendriCL comparators (VERBATIM from deep-dive-dendricl-ctp.md SS1.3)
# ---------------------------------------------------------------------------
DENDRICL_PUBLISHED = {
    "task": {10: 0.807, 20: 0.820, 25: 0.809, 30: 0.807, 40: 0.787, 50: 0.649},
    "probe": {10: 0.873, 20: 0.931, 30: 0.97},
    "seed_sigma_max": 0.036,
    "continuous_ablation_no_lif": 0.833,
    "pure_lif": {10: 0.801, 20: 0.086},
    "active_dendrites": {10: 0.668, 20: 0.061},
}

# Leg plans ------------------------------------------------------------------
VALIDATION_POINTS = [
    # (name, d_task, n_train, n_test)
    ("val_d10_primary", 10, 160, 40),
    ("val_d20_secondary", 20, 160, 40),
    ("super_d25", 25, 160, 40),
    ("super_d30", 30, 160, 40),
    ("super_d40", 40, 120, 40),
    ("super_d50", 50, 120, 40),
]
GAMMA_ARMS_VALIDATION = ("frozen", "gstar_a99", "gstar_a996")
# Seed stability at the primary anchor (episode-draw seeds; W_A/wiring frozen).
SEED_LEGS = [
    ("seeds_d10", 10, 160, 40, [11, 22, 33], "gstar_a996"),
    ("seeds_d30", 30, 160, 40, [11, 22], "gstar_a996"),
]
WIDTH_POINTS = [10, 32, 128, 384]
GAMMA_ARMS_WIDTH = ("joint", "gstar_a99")


def make_constants(d_task: int, gamma_mode: str, width: int | None = None):
    """Frozen constants, optionally re-declared at a swept width / gamma arm.

    Modes:
      frozen      alpha=.99, gamma derived from d_apical (default 384);
      joint       width-sweep twin of ``frozen``: gamma derived from the
                  SWEPT d_apical (preregistered joint declaration);
      gstar_a99   alpha=.99,  gamma = 0.2*gamma*(d_task);
      gstar_a996  alpha=.996, gamma = 0.2*gamma*(d_task).
    """
    if gamma_mode in ("frozen", "joint"):
        kwargs: dict = {}
        if width is not None:
            kwargs.update(
                d_apical=width,
                cell_count=width,
                aff_fanout=min(freeze.FROZEN.aff_fanout, width),
                rec_fanout=min(freeze.FROZEN.rec_fanout, width),
            )
        return freeze.FrozenConstants(**kwargs)
    if gamma_mode not in ("gstar_a99", "gstar_a996"):
        raise ValueError(f"unknown gamma_mode {gamma_mode!r}")
    # DendriCL transcription: gamma at the TASK dimension. The dataclass
    # validates gamma against the d_apical-derived bound, so declare the arm
    # explicitly here (freeze.py itself uses the same object.__setattr__ trick
    # for gamma derivation; the frozen file is not edited).
    alpha = 0.996 if gamma_mode == "gstar_a996" else 0.99
    if width is not None:
        c = freeze.FrozenConstants(alpha=alpha, **dict(
            d_apical=width,
            cell_count=width,
            aff_fanout=min(freeze.FROZEN.aff_fanout, width),
            rec_fanout=min(freeze.FROZEN.rec_fanout, width),
        ))
    else:
        c = freeze.FrozenConstants(alpha=alpha, gamma=-1.0)
    g = c.gamma_fraction * freeze.gamma_star(d_task)
    task_bound = freeze.stability_bound(d_task)
    if not 0.0 < g < task_bound:
        raise ValueError(f"gstar arm violates task-dim bound: {g} !< {task_bound}")
    object.__setattr__(c, "gamma", g)
    return c


def run_leg(name: str, *, d_task: int, n_train: int, n_test: int, seed: int,
            gamma_mode: str = "frozen", width: int | None = None) -> dict:
    """One operating point: generate episodes, replay, score all three layers."""
    t0 = time.perf_counter()
    c = make_constants(d_task, gamma_mode, width)
    spec = garg_task.TaskSpec(
        d_task=d_task,
        context_mult=c.context_mult,
        query_len=c.query_len,
        noise_sigma=c.noise_sigma,
        w_std=c.w_std,
    )
    rng = np.random.default_rng(seed)
    episodes = [
        garg_task.generate_episode(rng, spec, label_byte_scale=c.label_byte_scale)
        for _ in range(n_train + n_test)
    ]
    train_eps, test_eps = episodes[:n_train], episodes[n_train:]

    sim = runner.Stage0Runner.build(c, task_dim=d_task, lms_enabled=True, seed=seed)

    # ---- one replay pass over ALL episodes, keeping what each layer needs ----
    feats, targets = [], []          # probe layer: v_A rows -> reference states
    feats_q = []                     # decoded-task layer: v_A rows AT query positions
    ref_q, xs_q, wx_q = [], [], []   # task layer, reference recursion
    lift_err = 0.0                   # exact-lift identity check
    spikes_per_token = []
    for ep in episodes:
        out = sim.run_episode(ep)
        keep = ~ep.flags             # context positions only (probe convention)
        qmask = ep.flags
        feats.append(out.v_traj[keep])
        targets.append(out.ref_traj[keep])
        feats_q.append(out.v_traj[qmask])
        ref_q.append(out.pred[qmask])
        xs_q.append(ep.xs[qmask])
        wx_q.append(ep.xs[qmask] @ ep.w)
        spikes_per_token.append(float(out.spike_totals.mean()))
        # exact-lift sanity: hosted prediction must equal <ref_w, x_t>
        dot_ref = np.einsum("td,td->t", out.ref_traj, ep.xs)
        lift_err = max(lift_err, float(np.max(np.abs(out.pred - dot_ref))))

    features = np.concatenate(feats)
    tgt = np.concatenate(targets)

    # ---- probe layer (gate metric): split LAST n_test episodes' rows ----------
    rows_per_ep = int((~episodes[0].flags).sum())
    probe = scoring.probe_and_score(
        features, tgt,
        n_test_rows=n_test * rows_per_ep,
        ridge_lambda=c.ridge_lambda,
    )

    # ---- decoded-through-membrane task layer ---------------------------------
    # Probe fit on TRAIN episodes' context rows only; applied to TEST episodes'
    # query-position voltages (state is frozen across the query tail).
    split = features.shape[0] - n_test * rows_per_ep
    model = scoring.ridge_fit(features[:split], tgt[:split], c.ridge_lambda)

    dec_rows = []
    for ep, v_q in zip(test_eps, feats_q[-n_test:]):
        w_hat = scoring.ridge_predict(model, v_q)      # (n_queries, d_task)
        q_idx = np.nonzero(ep.flags)[0]
        dec_rows.append(np.einsum("rd,rd->r", w_hat, ep.xs[q_idx]))

    wx_test = np.concatenate(wx_q[-n_test:])
    ref_test = np.concatenate(ref_q[-n_test:])

    wall = time.perf_counter() - t0
    payload = {
        "leg": name,
        "kind": "width" if width is not None else "validation",
        "gamma_mode": gamma_mode,
        "seed": seed,
        "d_task": d_task,
        "context_len": spec.context_len,
        "query_len": spec.query_len,
        "noise_sigma": c.noise_sigma,
        "w_std": c.w_std,
        "alpha": c.alpha,
        "gamma_fraction": c.gamma_fraction,
        "gamma": c.gamma,
        "stability_bound_at_d_apical": freeze.stability_bound(c.d_apical),
        "gamma_star_task_dim": freeze.gamma_star(d_task),
        "stability_bound_at_task_dim": freeze.stability_bound(d_task),
        "d_apical": c.d_apical,
        "cell_count": c.cell_count,
        "aff_fanout": c.aff_fanout,
        "rec_fanout": c.rec_fanout,
        "n_train_episodes": n_train,
        "n_test_episodes": n_test,
        "ridge_lambda": c.ridge_lambda,
        "probe_mean_r2": probe.mean_r2,
        "probe_median_r2": probe.median_r2,
        "probe_min_r2": probe.min_r2,
        "probe_per_dim_r2": [float(v) for v in probe.per_dim_r2],
        "gate": c.r2_gate,
        "gate_pass_nondecisive": bool(probe.mean_r2 >= c.r2_gate),
        "task_ref_r2_vs_wx": float(_r2_pooled(wx_test, ref_test)),
        "task_decoded_r2_vs_wx": float(_r2_pooled(wx_test, np.concatenate(dec_rows))),
        "n_query_positions_test": int(sum(int(e.flags.sum()) for e in test_eps)),
        "exact_lift_max_abs_err": lift_err,
        "mean_spikes_per_token": float(np.mean(spikes_per_token)),
        "wall_s": round(wall, 2),
        "dendricl_published_task": DENDRICL_PUBLISHED["task"].get(d_task),
        "dendricl_published_probe": DENDRICL_PUBLISHED["probe"].get(d_task),
    }
    return payload


def _r2_pooled(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to named legs (substring match)")
    parser.add_argument("--force", action="store_true",
                        help="re-run legs whose result JSON already exists")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, dict]] = []
    for name, d_task, n_train, n_test in VALIDATION_POINTS:
        for arm in GAMMA_ARMS_VALIDATION:
            jobs.append((f"{name}_{arm}",
                         dict(d_task=d_task, n_train=n_train, n_test=n_test,
                              seed=freeze.FROZEN.base_seed, gamma_mode=arm)))
    for name, d_task, n_train, n_test, seeds, arm in SEED_LEGS:
        for s in seeds:
            jobs.append((f"{name}_s{s}", dict(d_task=d_task, n_train=n_train,
                                              n_test=n_test, seed=s, gamma_mode=arm)))
    for width in WIDTH_POINTS:
        for arm in GAMMA_ARMS_WIDTH:
            jobs.append((f"width{width}_d10_{arm}",
                         dict(d_task=10, n_train=160, n_test=40,
                              seed=freeze.FROZEN.base_seed, gamma_mode=arm,
                              width=width)))

    ran, skipped = [], []
    for name, kwargs in jobs:
        if args.only and not any(pat in name for pat in args.only):
            continue
        out_path = RESULTS_DIR / f"{name}.json"
        if out_path.exists() and not args.force:
            skipped.append(name)
            print(f"[skip] {name} (result exists)", flush=True)
            continue
        print(f"[run ] {name}: {kwargs}", flush=True)
        try:
            payload = run_leg(name, **kwargs)
        except Exception as exc:  # noqa: BLE001 - record and continue other legs
            print(f"[FAIL] {name}: {exc!r}", flush=True)
            out_path.write_text(json.dumps({"leg": name, "error": repr(exc)},
                                           indent=2), encoding="utf-8")
            raise
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ran.append(name)
        print(
            f"[done] {name}: probe={payload['probe_mean_r2']:.4f} "
            f"ref={payload['task_ref_r2_vs_wx']:.4f} "
            f"decoded={payload['task_decoded_r2_vs_wx']:.4f} "
            f"({payload['wall_s']:.0f}s)",
            flush=True,
        )

    print(f"\nlegs run: {len(ran)}, skipped: {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
