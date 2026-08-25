#!/usr/bin/env python3
"""B' Stage-0 CORPUS LEG — decisive dual-layer execution (license-gated).

Fires the licensed corpus cohort leg and reads the AMENDED two-layer gate
(benchmark-validation amendment, commit 14cc57b: probe-R^2 alone cannot
certify hosting — near-static recursion kept probe-R^2 at 0.83+):

  layer (a)  probe-R^2 (mean held-out ridge fidelity v_A -> reference LMS
             trajectory, split BY EPISODE) >= 0.85  [the frozen Stage-0 gate];
  layer (b)  decoded-through-membrane task R^2 >= abstract-recursion task R^2
             at every matched operating point [DendriCL published-band
             relationship; both layers reported verbatim per episode batch].

Plus the confirmatory d_eff sweep {10, 32, 128, 384} under FROZEN constants
(joint gamma declaration: gamma derived from the swept d_apical) to confirm
the no-width-cliff finding at cohort scale. Sweep metrics are DIAGNOSTIC:
they never gate (layer (a) gates only at the preregistered configuration;
layer (b) is evaluated at every reported point).

DOCUMENTED INVOCATION (fixed here BEFORE firing; sizing authority delegated
at licensing time, README "Cohort sizing to be set at licensing time"):
    python experiments/apical_lms_stage0/run_stage0.py --cohort \
        --license-token retro-probe:PASS \
        --train-episodes 640 --test-episodes 160
Rationale for 640/160: (i) at or above the sanctioned 160/40 minimum;
(ii) same 80/20 train/test convention as the validated benchmark protocol;
(iii) removes probe starvation (README: fixture-scale R^2 rises with probe
training rows) as a confounder so the one-shot verdict reflects the membrane,
not the probe budget. NO frozen constant, gate, shape, or seed policy changes.
One shot, stop-on-failure: this size is not revisited based on its outcome.

Exit codes: 0 = PASS (both layers) -> B' proceeds to Stage 1;
            1 = valid terminal FAIL (below-gate; scientific result — artifacts
                preserved, B' dies at Stage 0);
            2 = infrastructure failure (no verdict consumed).

Deterministic end to end: fixed seeds; episodes independent (full reset each);
results assemble in submission order.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

STAGE0 = Path(__file__).resolve().parents[1]
REPO = STAGE0.resolve().parents[1]
for _entry in (str(REPO), str(REPO / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from experiments.apical_lms_stage0 import freeze  # noqa: E402
from experiments.apical_lms_stage0.benchmark_validation import (  # noqa: E402
    run_benchmark_validation as bmv,
)

# ---- the licensing facts ----------------------------------------------------
LICENSE_TOKEN = "retro-probe:PASS"          # verdict commit d999ed3
LICENSE_COMMIT = "d999ed3"
HARNESS_COMMIT = "23bb93b"
AMENDMENT_COMMIT = "14cc57b"

# ---- documented invocation (frozen before firing) ----------------------------
N_TRAIN = 640
N_TEST = 160
SEED = freeze.FROZEN.base_seed              # 20260825
PRIMARY_D_TASK = 10                         # Garg primary anchor
SWEEP_WIDTHS = (10, 32, 128, 384)           # d_apical = cell_count, joint gamma

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"


def main() -> int:
    t_start = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--license-token", type=str, required=True)
    args = parser.parse_args()
    if args.license_token != LICENSE_TOKEN:
        print(f"REFUSED: license token {args.license_token!r} != "
              f"{LICENSE_TOKEN!r} ({LICENSE_COMMIT})", flush=True)
        return 2

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                          capture_output=True, text=True)
    provenance = {
        "branch": "codex/path2-pilot",
        "head": head.stdout.strip(),
        "harness_commit": HARNESS_COMMIT,
        "license_commit": LICENSE_COMMIT,
        "amendment_commit": AMENDMENT_COMMIT,
        "license_token": LICENSE_TOKEN,
        "n_train_episodes": N_TRAIN,
        "n_test_episodes": N_TEST,
        "seed": SEED,
        "base_config": {
            "alpha": freeze.FROZEN.alpha,
            "d_apical": freeze.FROZEN.d_apical,
            "cell_count": freeze.FROZEN.cell_count,
            "gamma_frozen_at_d_apical": freeze.FROZEN.gamma,
            "gamma_fraction": freeze.FROZEN.gamma_fraction,
            "gamma_star_task_dim": freeze.gamma_star(PRIMARY_D_TASK),
            "stability_bound_d_apical": freeze.stability_bound(freeze.FROZEN.d_apical),
            "stability_bound_task_dim": freeze.stability_bound(PRIMARY_D_TASK),
            "r2_gate": freeze.FROZEN.r2_gate,
        },
    }
    print("provenance:", json.dumps(provenance, indent=2), flush=True)

    # ------------------------------------------------------------------
    # STEP 1 — fire the licensed corpus leg (the mandated CLI, verbatim).
    # Layer (a) artifact of record; stop-on-failure exit code preserved.
    # ------------------------------------------------------------------
    stage0_out = RESULTS_DIR / "cohort_primary_licensed_invocation.json"
    cmd = [
        sys.executable, str(STAGE0 / "run_stage0.py"),
        "--cohort",
        "--license-token", LICENSE_TOKEN,
        "--train-episodes", str(N_TRAIN),
        "--test-episodes", str(N_TEST),
        "--out", str(stage0_out),
    ]
    print("\n[step 1] firing licensed corpus leg:", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    (RESULTS_DIR / "cohort_primary_licensed_invocation.stdout.log").write_text(
        proc.stdout, encoding="utf-8")
    if proc.returncode not in (0, 1):
        (RESULTS_DIR / "cohort_primary_licensed_invocation.stderr.log").write_text(
            proc.stderr, encoding="utf-8")
        print(f"[step 1] INFRA FAILURE rc={proc.returncode}\n{proc.stderr[-2000:]}",
              flush=True)
        return 2
    stage0_payload = json.loads(stage0_out.read_text(encoding="utf-8"))
    print(f"[step 1] done in {time.perf_counter()-t0:.0f}s rc={proc.returncode} "
          f"probe_r2={stage0_payload['mean_r2']:.6f} "
          f"gate={stage0_payload['gate']} passed={stage0_payload['passed']}",
          flush=True)

    # ------------------------------------------------------------------
    # STEP 2 — dual-layer read of the IDENTICAL primary point (three-layer
    # machinery reused verbatim from the validated calibration runner).
    # Doubles as a determinism cross-check against step 1.
    # ------------------------------------------------------------------
    print("\n[step 2] dual-layer primary recompute (probe + ref + decoded)", flush=True)
    t0 = time.perf_counter()
    primary = bmv.run_leg(
        f"corpus_primary_d{PRIMARY_D_TASK}_frozen",
        d_task=PRIMARY_D_TASK, n_train=N_TRAIN, n_test=N_TEST,
        seed=SEED, gamma_mode="frozen",
    )
    primary["leg"] = "corpus_primary_d10_frozen"
    primary["kind"] = "corpus-primary"
    delta = abs(primary["probe_mean_r2"] - stage0_payload["mean_r2"])
    primary["crosscheck_abs_delta_probe_r2_vs_stage0_cli"] = delta
    primary["crosscheck_match"] = bool(delta <= 1e-9)
    (RESULTS_DIR / "cohort_primary_dual_layer.json").write_text(
        json.dumps(primary, indent=2), encoding="utf-8")
    print(f"[step 2] done in {time.perf_counter()-t0:.0f}s "
          f"probe={primary['probe_mean_r2']:.6f} ref={primary['task_ref_r2_vs_wx']:.6f} "
          f"decoded={primary['task_decoded_r2_vs_wx']:.6f} "
          f"lift_err={primary['exact_lift_max_abs_err']:.2e} crossmatch={delta:.2e}",
          flush=True)

    # ------------------------------------------------------------------
    # STEP 3 — d_eff sweep under frozen constants (joint gamma reading).
    # Width 384 needs no separate leg: make_constants("frozen") IS the
    # primary configuration (validated bit-identity precedent, T3).
    # ------------------------------------------------------------------
    sweep_rows = [{
        "leg": primary["leg"],
        "d_apical": primary["d_apical"],
        "cell_count": primary["cell_count"],
        "gamma": primary["gamma"],
        "stability_bound_at_d_apical": primary["stability_bound_at_d_apical"],
        "mean_spikes_per_token": primary["mean_spikes_per_token"],
        "probe_mean_r2": primary["probe_mean_r2"],
        "ref_task_r2": primary["task_ref_r2_vs_wx"],
        "decoded_task_r2": primary["task_decoded_r2_vs_wx"],
        "decoded_ge_ref": bool(primary["task_decoded_r2_vs_wx"] >= primary["task_ref_r2_vs_wx"]),
        "source": "corpus_primary_dual_layer.json",
    }]
    for width in SWEEP_WIDTHS:
        if width == freeze.FROZEN.d_apical:
            continue  # identical config to the primary leg by construction
        name = f"corpus_width{width}_d{PRIMARY_D_TASK}_frozenjoint"
        print(f"\n[step 3] width leg d_apical={width}", flush=True)
        t0 = time.perf_counter()
        row = bmv.run_leg(name, d_task=PRIMARY_D_TASK, n_train=N_TRAIN,
                          n_test=N_TEST, seed=SEED, gamma_mode="frozen",
                          width=width)
        row["kind"] = "corpus-width-sweep"
        (RESULTS_DIR / f"{name}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8")
        sweep_rows.append({
            "leg": name,
            "d_apical": row["d_apical"],
            "cell_count": row["cell_count"],
            "gamma": row["gamma"],
            "stability_bound_at_d_apical": row["stability_bound_at_d_apical"],
            "mean_spikes_per_token": row["mean_spikes_per_token"],
            "probe_mean_r2": row["probe_mean_r2"],
            "ref_task_r2": row["task_ref_r2_vs_wx"],
            "decoded_task_r2": row["task_decoded_r2_vs_wx"],
            "decoded_ge_ref": bool(row["task_decoded_r2_vs_wx"] >= row["task_ref_r2_vs_wx"]),
            "source": f"{name}.json",
        })
        print(f"[step 3] done in {time.perf_counter()-t0:.0f}s "
              f"probe={row['probe_mean_r2']:.6f} ref={row['task_ref_r2_vs_wx']:.6f} "
              f"decoded={row['task_decoded_r2_vs_wx']:.6f}", flush=True)
    sweep_rows.sort(key=lambda r: r["d_apical"])

    # ------------------------------------------------------------------
    # STEP 4 — assemble the amended two-layer verdict.
    # ------------------------------------------------------------------
    gate = freeze.FROZEN.r2_gate
    layer_a = {
        "metric": "mean held-out probe R^2 (v_A -> reference online-LMS trajectory)",
        "value": stage0_payload["mean_r2"],
        "threshold": gate,
        "pass": bool(stage0_payload["mean_r2"] >= gate),
        "artifact": "cohort_primary_licensed_invocation.json",
        "stage0_cli_exit_code": proc.returncode,
        "median_r2": stage0_payload.get("median_r2"),
    }
    layer_b_points = [
        {
            "point": f"d_task={PRIMARY_D_TASK}, d_apical={r['d_apical']}",
            "decoded_task_r2": r["decoded_task_r2"],
            "abstract_ref_task_r2": r["ref_task_r2"],
            "margin_decoded_minus_ref": r["decoded_task_r2"] - r["ref_task_r2"],
            "hold": bool(r["decoded_task_r2"] >= r["ref_task_r2"]),
        }
        for r in sweep_rows
    ]
    layer_b = {
        "relationship": "decoded-through-membrane task R^2 >= abstract-recursion task R^2 "
                        "at every matched operating point (DendriCL published-band relationship)",
        "points": layer_b_points,
        "pass": all(p["hold"] for p in layer_b_points),
    }
    # Non-gating confirmation diagnostics (no-cliff map under frozen constants).
    probes = [r["probe_mean_r2"] for r in sweep_rows]
    diagnostics = {
        "question": "does the no-width-cliff finding hold at cohort scale under frozen constants?",
        "sweep": sweep_rows,
        "probe_r2_range_across_widths": [min(probes), max(probes)],
        "probe_r2_flat_within": float(max(probes) - min(probes)),
        "gamma_tracks_shrinkage": [
            {"d_apical": r["d_apical"], "gamma": r["gamma"],
             "ref_task_r2": r["ref_task_r2"], "decoded_task_r2": r["decoded_task_r2"]}
            for r in sweep_rows
        ],
        "note": "diagnostics never gate (skill discipline: only frozen gates decide)",
    }

    passed = bool(layer_a["pass"] and layer_b["pass"])
    verdict = {
        "verdict": "PASS" if passed else "FAIL",
        "semantics": (
            "B' proceeds to Stage 1 (decision-point decoding on corpus replay)"
            if passed else
            "fitted GLIF5 apical membrane cannot host the recursion; B' dies at Stage 0"
        ),
        "gate": {
            "amendment": "dual-layer (14cc57b): probe-R^2 alone cannot certify hosting",
            "layer_a_probe": layer_a,
            "layer_b_hosting": layer_b,
            "conclusion": passed,
        },
        "determinism_crosscheck": {
            "abs_delta_probe_r2_stage0cli_vs_threelayer": delta,
            "match": bool(delta <= 1e-9),
            "exact_lift_max_abs_err": primary["exact_lift_max_abs_err"],
        },
        "diagnostics_non_gating": diagnostics,
        "provenance": provenance,
        "wall_s_total": round(time.perf_counter() - t_start, 1),
    }
    out = HERE / "corpus_leg_verdict.json"
    out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print("\n================ CORPUS LEG VERDICT ================", flush=True)
    print(f"layer (a) probe-R^2 {layer_a['value']:.6f} >= {gate} : "
          f"{'PASS' if layer_a['pass'] else 'FAIL'}", flush=True)
    for p in layer_b_points:
        print(f"layer (b) {p['point']}: decoded {p['decoded_task_r2']:.6f} vs ref "
              f"{p['abstract_ref_task_r2']:.6f} (margin {p['margin_decoded_minus_ref']:+.6f}) : "
              f"{'HOLD' if p['hold'] else 'VIOLATED'}", flush=True)
    print(f"VERDICT: {verdict['verdict']} -> {verdict['semantics']}", flush=True)
    print(f"total wall {verdict['wall_s_total']:.0f}s; verdict JSON: {out}", flush=True)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
