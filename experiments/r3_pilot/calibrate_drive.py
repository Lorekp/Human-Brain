"""C1 pre-launch drive calibration: measure the drive-vs-N scaling law (Blocker 1).

Sanctioned PRE-LAUNCH measurement for the C1 readiness blocker
(`experiments/r3_pilot/C1_READINESS.md`, commit 10453e0): the FROZEN default
lane constants were reported silent at N = 131,072 (spike-rate pre-flight
FAIL, mean winner-set size 0) while the same driver passes at the N = 96
fixture (mean |W| ~= 53.8, density 0.00555). This script measures, on one
200-token TRAIN slice (the exact G-S0 prefix):

  1. mean winner-set size + spike density for every (N, g) on the grid
     N in {96-ref, 1024, 8192, 65536, 131072} x g in {1, 50, 150, 300},
     where the drive gain g multiplies the frozen afferent current
     (RecurrentCircuitConfig.input_current_pa = 5000 pA * g); every other
     constant stays at its frozen default;
  2. an explicit backend cross-check at N = 131,072 (auto/cpu/cuda) because
     the auto policy switches kernels at 100,000 cells
     (`humanbrain.cuda._CUDA_MINIMUM_CELLS`) -- the fixture evidence and the
     scale failure were produced by DIFFERENT kernels;
  3. a confirmation leg at the recommended setting over a >= 4,096-step
     prefix (the prereg section 6 gate horizon used by `spike_preflight_gate`).

Instrument parity: the recorder IS `r3_pilot_glue.SinkSpikeRecorder`, so
`mean_activity_density` / `spike_rate_health` are computed exactly as the C1
gate computes them. Winner sets W_{t-1} come from the observation-sink spike
stream (`circuit._token_spike_counts > 0`) exactly as in
`r3_pilot_glue.run_arm`; the tracker is intentionally NOT attached (this
measures lane signal, not growth mechanics).

Output: `calib_curve.json` beside this script (rewritten after every probe,
so an interrupted sweep keeps all completed rows; supplementary invocations
append to the same manifest and never rerun completed probes).

Prereg discipline: this freezes BEFORE C1. Nothing here tunes anything
post-launch; the resulting choice is recorded as a Change Note in
`docs/research/r3-drive-calibration.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

os.environ.setdefault("HUMANBRAIN_NATIVE_CACHE", str(REPO_ROOT / ".native-cache"))

from humanbrain.cuda import should_use_cuda  # noqa: E402
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig  # noqa: E402
from humanbrain.r3_pilot_glue import (  # noqa: E402
    CONCEPT_ID_CEILING,
    CONSTRUCTION_SEED,
    SinkSpikeRecorder,
)

# Frozen defaults under calibration (RecurrentCircuitConfig defaults).
BASE_DRIVE_PA = 5_000.0
INPUT_FANOUT = 24
TOKEN_CELLS = int(CONCEPT_ID_CEILING)  # concept band ceiling, ids [1024, 1536)

N_TOKENS = 200                     # G-S0 slice length
CONFIRM_STEPS_MIN = 4_096          # prereg section 6 gate horizon
WINNER_BAND = (20.0, 60.0)         # "G-S0-like" winner-set band (fixture 53.8)
DENSITY_FLOOR = 0.001              # prereg section 6 population-rate floor

CACHE_DIR = Path(__file__).resolve().parent / "c1_dry_run" / "cache"
GS0_SLICE_SHA256 = (  # gs0_health_run.json slice.slice_sha256 (fixture anchor)
    "de5d527b12968762d6bd614c3a77c705e053e272361a72d20f3c0a21b272ef3c"
)


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_slice() -> tuple[list[int], dict]:
    """First 200 TRAIN concept ids == the exact G-S0 fixture prefix.

    Loads the cached full-side TRAIN stream produced by run_c1_cohort.py's
    build_token_caches (sha-pinned), asserts corpus + npy integrity, slices
    the canonical 200-token prefix and asserts its sha equals the G-S0 slice
    sha so every probe replays the fixture's own token sequence.
    """
    streams = json.loads((CACHE_DIR / "streams.json").read_text(encoding="utf-8"))
    rec = streams["sides"]["train"]
    arr = np.load(CACHE_DIR / "train_tokens.npy")
    if hashlib.sha256(arr.tobytes()).hexdigest() != rec["npy_sha256"]:
        raise SystemExit("STOP: cached TRAIN stream sha mismatch (cache drift)")
    tokens = [int(t) for t in arr[:N_TOKENS]]
    sha = hashlib.sha256(",".join(str(t) for t in tokens).encode("utf-8")).hexdigest()
    info = {
        "n_tokens": len(tokens),
        "distinct_ids": len(set(tokens)),
        "max_token_id": max(tokens),
        "slice_sha256": sha,
        "matches_gs0_fixture_slice": sha == GS0_SLICE_SHA256,
        "train_stream_sha256": rec.get("stream_sha256"),
        "trimmed_corpus_sha256": streams.get("trimmed_sha256"),
        "cache_dir": str(CACHE_DIR),
    }
    if not info["matches_gs0_fixture_slice"]:
        raise SystemExit(f"STOP: 200-token prefix sha {sha} != G-S0 fixture slice")
    return tokens, info


def probe(tokens: list[int], *, lane_cells: int, gain: float, device: str) -> dict:
    """Replay the slice through ONE lane at (N, g, device); return metrics."""
    os.environ["HUMANBRAIN_DEVICE"] = device
    # The policy env var is read at circuit construction time (inside
    # recurrent_kernel -> should_use_cuda), so setting it here selects the
    # backend for THIS probe only.
    config = RecurrentCircuitConfig(
        reservoir_cells=lane_cells,
        input_current_pa=BASE_DRIVE_PA * gain,  # the ONLY knob under calibration
    )
    circuit = RecurrentCellCircuit(TOKEN_CELLS, config=config)
    recorder = SinkSpikeRecorder()
    circuit.observation_sink = recorder

    backend_cuda = bool(should_use_cuda(lane_cells))
    t0 = time.time()
    total_spikes_reported = 0
    for tok in tokens:
        step = circuit.observe(tok)
        total_spikes_reported += int(step.reservoir_spikes)
    elapsed = time.time() - t0

    sizes = [len(w) for w in recorder.winner_sets]
    union = set().union(*recorder.winner_sets) if recorder.winner_sets else set()
    health = recorder.spike_rate_health(DENSITY_FLOOR)
    sink_total = int(sum(recorder.spike_totals))
    row = {
        "lane_cells": lane_cells,
        "gain": gain,
        "device_policy": device,
        "backend": "cuda" if backend_cuda else "cpu-native",
        "input_current_pa": BASE_DRIVE_PA * gain,
        "mean_winner_set_size": float(np.mean(sizes)),
        "median_winner_set_size": float(np.median(sizes)),
        "max_winner_set_size": int(max(sizes)),
        "empty_winner_fraction": float(sum(1 for s in sizes if s == 0) / len(sizes)),
        "distinct_winners": len(union),
        "mean_activity_density": health["mean_activity_density"],
        "density_floor_passed": bool(health["passed"]),
        "sink_total_lane_spikes": sink_total,
        "observe_return_total_spikes": total_spikes_reported,
        # Artifact detector: observe() reports spikes while the mirrored
        # count bank reads empty => per-cell counts were never copied back.
        "telemetry_consistent": (sink_total > 0) == (sum(sizes) > 0),
        "wall_clock_seconds": round(elapsed, 3),
        "steps_per_second": round(len(tokens) / max(elapsed, 1e-9), 1),
    }
    del circuit, recorder
    return row


def confirm(tokens: list[int], *, lane_cells: int, gain: float, device: str,
            min_steps: int) -> dict:
    """Gate-horizon confirmation: >= min_steps steps at (N*, g*)."""
    os.environ["HUMANBRAIN_DEVICE"] = device
    config = RecurrentCircuitConfig(
        reservoir_cells=lane_cells, input_current_pa=BASE_DRIVE_PA * gain,
    )
    circuit = RecurrentCellCircuit(TOKEN_CELLS, config=config)
    recorder = SinkSpikeRecorder()
    circuit.observation_sink = recorder
    passes = -(-min_steps // len(tokens))  # ceil
    stream = tokens * passes
    t0 = time.time()
    for tok in stream:
        circuit.observe(tok)
    elapsed = time.time() - t0
    sizes = [len(w) for w in recorder.winner_sets]
    health = recorder.spike_rate_health(DENSITY_FLOOR)
    row = {
        "lane_cells": lane_cells,
        "gain": gain,
        "device_policy": device,
        "backend": "cuda" if should_use_cuda(lane_cells) else "cpu-native",
        "steps": len(stream),
        "passes": passes,
        "mean_winner_set_size": float(np.mean(sizes)),
        "empty_winner_fraction": float(sum(1 for s in sizes if s == 0) / len(sizes)),
        "mean_activity_density": health["mean_activity_density"],
        "density_floor_passed": bool(health["passed"]),
        "prereg_gate_would_pass": bool(health["passed"] and np.mean(sizes) > 0),
        "first_4096_mean_winner_set_size": float(np.mean(sizes[:min_steps])),
        "wall_clock_seconds": round(elapsed, 1),
    }
    del circuit, recorder
    return row


def single_cell_fi(gains: Sequence[float]) -> dict:
    """Offline F-I evidence for the mechanism section (no circuit needed).

    Integrates ONE pinned GLIF5 cell (AllenGlif5Parameters) with the exact
    kernel arithmetic: voltage forward-Euler, three-component threshold,
    after-spike currents, spike-cut refractory. Stimulus = the frozen
    afferent pulse (5000*g pA for pulse_steps sub-steps, then quiet until
    token_steps), from reset state. Also binary-searches the sustained
    rheobase. Establishes: one spike max per driven cell per token
    (refractory > token window), firing threshold gain, latency-vs-g.
    """
    from humanbrain.cells import AllenGlif5Parameters, FittedGlif5Constants

    p = AllenGlif5Parameters()
    c = FittedGlif5Constants.from_parameters(p)
    token_steps = max(1, round(5.0 / p.dt_ms))
    pulse_steps = max(1, round(1.5 / p.dt_ms))

    def run(I_pa: float, steps: int) -> tuple[int, float]:
        v = 0.0
        after = np.zeros(len(p.after_spike_taus_ms))
        th_spike = 0.0
        th_voltage = 0.0
        refractory = 0
        spikes = 0
        first_ms = None
        for step in range(steps):
            if refractory > 0:
                refractory -= 1
                continue
            total = (I_pa if step < pulse_steps else 0.0) + float(after.sum())
            leak = p.leak_ns * v
            v += c.voltage_step_scale * (total - leak)
            beta = total * c.threshold_beta_scale
            th_voltage = (c.threshold_exp_decay * th_voltage
                          + c.threshold_voltage_factor * v
                          + c.threshold_beta_factor * beta)
            th_spike *= c.spike_threshold_decay
            if v > p.threshold_inf_relative_mv + th_voltage + th_spike:
                v = p.voltage_reset_scale * v + p.voltage_reset_offset_mv
                after = c.after_spike_amplitudes_pa + after * c.after_spike_cut_decay
                th_spike = (th_spike * c.spike_cut_threshold_decay
                            + p.threshold_spike_increment_mv)
                refractory = p.spike_cut_steps
                spikes += 1
                if first_ms is None:
                    first_ms = (step + 1) * p.dt_ms
        return spikes, float("nan") if first_ms is None else first_ms

    def rheobase() -> float:
        lo, hi = 0.0, 500.0

        def fires(I: float) -> bool:
            v = 0.0
            after = np.zeros(len(p.after_spike_taus_ms))
            th_spike = 0.0
            th_voltage = 0.0
            refractory = 0
            for _ in range(60000):  # 3 s simulated sustained drive
                if refractory > 0:
                    refractory -= 1
                    continue
                total = I + float(after.sum())
                v += c.voltage_step_scale * (total - p.leak_ns * v)
                beta = total * c.threshold_beta_scale
                th_voltage = (c.threshold_exp_decay * th_voltage
                              + c.threshold_voltage_factor * v
                              + c.threshold_beta_factor * beta)
                th_spike *= c.spike_threshold_decay
                if v > p.threshold_inf_relative_mv + th_voltage + th_spike:
                    return True
            return False

        while hi - lo > 0.5:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if fires(mid) else (lo, mid)
        return hi

    rows = {}
    for g in gains:
        spikes, first = run(BASE_DRIVE_PA * g, token_steps)
        rows[str(g)] = {"spikes_per_token": spikes, "first_spike_ms": first}
    return {
        "cell": "AllenGlif5Parameters (pinned human MTG GLIF5)",
        "token_steps": token_steps,
        "pulse_steps": pulse_steps,
        "spike_cut_steps": p.spike_cut_steps,
        "pulse_response_by_gain": rows,
        "sustained_rheobase_pa": round(rheobase(), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid-n", type=int, nargs="+",
                    default=[1024, 8192, 65536, 131072])
    ap.add_argument("--grid-g", type=float, nargs="+",
                    default=[1.0, 50.0, 150.0, 300.0])
    ap.add_argument("--grid-device", choices=("auto", "cpu", "cuda"), default="auto",
                    help="device policy for grid rows (cross-checks always "
                         "force explicit devices)")
    ap.add_argument("--fixture-ref", action="store_true", default=True)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "calib_curve.json")
    ap.add_argument("--skip-confirm", action="store_true",
                    help="skip the >= 4096-step confirmation leg")
    ap.add_argument("--single-cell-fi", action="store_true",
                    help="only run the offline single-cell F-I probe and "
                         "merge it into the manifest as single_cell_fi")
    args = ap.parse_args()

    print("[1/4] loading the G-S0-pinned 200-token TRAIN slice...", flush=True)
    tokens, slice_info = load_slice()
    print(f"      {slice_info['n_tokens']} tokens, {slice_info['distinct_ids']} distinct, "
          f"max id {slice_info['max_token_id']}; "
          f"G-S0 slice match: {slice_info['matches_gs0_fixture_slice']}", flush=True)

    manifest: dict = {
        "experiment": "r3-c1-drive-calibration",
        "dispatch": "C1_READINESS.md Blocker 1 (pre-launch, sanctioned)",
        "git_rev": git_rev(),
        "construction_seed": CONSTRUCTION_SEED,
        "base_drive_pa": BASE_DRIVE_PA,
        "input_fanout_frozen": INPUT_FANOUT,
        "winner_band": list(WINNER_BAND),
        "density_floor": DENSITY_FLOOR,
        "slice": slice_info,
        "rows": [],
        "confirmations": [],
    }
    # Append mode: reuse an existing manifest so supplementary sweeps extend
    # the SAME artifact; completed (N, g, device) probes are never rerun.
    # Derived keys are recomputed; everything else (slice pinning,
    # single_cell_fi, ...) carries over untouched.
    _DERIVED = ("rows", "confirmations", "minimal_g_by_N", "recommendation",
                "telemetry_inconsistent_rows_used_as_evidence_only", "git_rev")
    if args.out.exists():
        try:
            prior = json.loads(args.out.read_text(encoding="utf-8"))
            if prior.get("experiment") == manifest["experiment"]:
                for key, value in prior.items():
                    if key not in manifest and key not in _DERIVED:
                        manifest[key] = value
                manifest["rows"] = prior.get("rows", [])
                manifest["confirmations"] = prior.get("confirmations", [])
                print(f"      appending to existing manifest "
                      f"({len(manifest['rows'])} rows already present)", flush=True)
        except (json.JSONDecodeError, OSError):
            pass

    if args.single_cell_fi:
        print("[fi] integrating one pinned GLIF5 cell across gains...", flush=True)
        fi = single_cell_fi([1.0, 2.0, 5.0, 10.0, 50.0, 150.0, 300.0]
                            + [round(g * 0.1, 1) for g in range(2, 8)])
        manifest["single_cell_fi"] = fi
        args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                            encoding="utf-8")
        print(f"      rheobase={fi['sustained_rheobase_pa']} pA; "
              f"spikes/token by gain: "
              f"{ {g: r['spikes_per_token'] for g, r in fi['pulse_response_by_gain'].items()} }",
              flush=True)
        print(f"manifest: {args.out}", flush=True)
        return 0

    def emit(row: dict) -> None:
        manifest["rows"].append(row)
        args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        g, n = row["gain"], row["lane_cells"]
        print(
            f"      N={n:>6} g={g:>5.0f} [{row['backend']:>10}] "
            f"|W|={row['mean_winner_set_size']:8.2f} "
            f"(empty {row['empty_winner_fraction']:.2f}, distinct {row['distinct_winners']}) "
            f"density={row['mean_activity_density']:.2e} "
            f"sink_spikes={row['sink_total_lane_spikes']} "
            f"consistent={row['telemetry_consistent']} "
            f"({row['steps_per_second']:.0f} steps/s)",
            flush=True,
        )

    plans: list[tuple[int, float, str]] = []
    done = {
        (r.get("lane_cells"), r.get("gain"), r.get("device_policy"))
        for r in manifest["rows"] if "error" not in r
    }
    if args.fixture_ref and not any(
        (n, g, d) in done for n, g, d in [(96, 1.0, "auto")]
    ) and args.grid_device == "auto":
        plans.append((96, 1.0, "auto"))          # fixture anchor (CPU path)
    for n in args.grid_n:
        for g in args.grid_g:
            plans.append((n, g, args.grid_device))
    # Backend cross-check ONLY where the auto policy flips kernels (>=100k):
    # isolates kernel-switch effects from genuine population physics.
    for n in args.grid_n:
        if n >= 100_000:
            plans.append((n, 1.0, "cpu"))
            plans.append((n, 1.0, "cuda"))
    plans = [p for p in plans if p not in done]
    plans = sorted(set(plans), key=lambda t: (t[0], t[1], t[2]))

    print(f"[2/4] sweeping {len(plans)} (N, g, device) probes x {len(tokens)} tokens...",
          flush=True)
    for i, (n, g, dev) in enumerate(plans, start=1):
        try:
            emit(probe(tokens, lane_cells=n, gain=g, device=dev))
        except Exception as exc:  # keep the sweep alive; record the failure
            manifest["rows"].append({
                "lane_cells": n, "gain": g, "device_policy": dev,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"      N={n} g={g} [{dev}] ERROR {type(exc).__name__}: {exc}", flush=True)

    # ---- selection -----------------------------------------------------
    print("[3/4] selecting minimal g(N) and the C1 recommendation...", flush=True)
    # Only telemetry-consistent rows measure physics: a row whose sink read
    # zeros while observe() reported spikes is a mirroring failure at that
    # (N, device), not lane behavior, and must not drive the choice.
    consistent = [r for r in manifest["rows"]
                  if "error" not in r and r.get("telemetry_consistent")]
    dropped = [r for r in manifest["rows"]
               if "error" not in r and not r.get("telemetry_consistent")]
    manifest["telemetry_inconsistent_rows_used_as_evidence_only"] = [
        {"lane_cells": r["lane_cells"], "gain": r["gain"],
         "device_policy": r["device_policy"], "backend": r["backend"],
         "sink_total_lane_spikes": r["sink_total_lane_spikes"],
         "observe_return_total_spikes": r["observe_return_total_spikes"]}
        for r in dropped
    ]
    minimal_g: dict[str, object] = {}
    for n in sorted({r["lane_cells"] for r in consistent}):
        candidates = sorted((r for r in consistent if r["lane_cells"] == n),
                            key=lambda r: r["gain"])
        in_band = [r for r in candidates
                   if WINNER_BAND[0] <= r["mean_winner_set_size"] <= WINNER_BAND[1]]
        gate_ok = [r for r in candidates
                   if r["density_floor_passed"] and r["mean_winner_set_size"] > 0]
        minimal_g[str(n)] = {
            "minimal_g_winner_band": in_band[0]["gain"] if in_band else None,
            "minimal_g_gate_passing": gate_ok[0]["gain"] if gate_ok else None,
            "per_g_winner_sizes": {str(r["gain"]): r["mean_winner_set_size"]
                                   for r in candidates},
            "per_g_density": {str(r["gain"]): r["mean_activity_density"]
                              for r in candidates},
        }
        print(f"      N={n:>6}: minimal g for |W| in {WINNER_BAND} -> "
              f"{minimal_g[str(n)]['minimal_g_winner_band']}; "
              f"section-6 gate-passing g -> "
              f"{minimal_g[str(n)]['minimal_g_gate_passing']}", flush=True)
    manifest["minimal_g_by_N"] = minimal_g

    # Recommendation: smallest telemetry-consistent g at the C1 lane size
    # that passes BOTH prereg-section-6 criteria (density floor + nonzero
    # winner sets); fall back to band membership if none passes the floor.
    c1_rows = sorted((r for r in consistent
                      if r["lane_cells"] == max(args.grid_n)),
                     key=lambda r: r["gain"])
    rec = next((r for r in c1_rows
                if r["density_floor_passed"] and r["mean_winner_set_size"] > 0), None)
    basis = "smallest g at N=C1_LANE_CELLS passing BOTH section-6 criteria"
    if rec is None:
        rec = next((r for r in c1_rows
                    if WINNER_BAND[0] <= r["mean_winner_set_size"] <= WINNER_BAND[1]),
                   None)
        basis = ("smallest g at N=C1_LANE_CELLS with mean |W| in the G-S0 band "
                 "(density floor NOT met anywhere on the grid)")
    manifest["recommendation"] = (
        {
            "rule": basis,
            "lane_cells": rec["lane_cells"],
            "device_policy_used": rec["device_policy"],
            "backend": rec["backend"],
            "gain": rec["gain"],
            "input_current_pa": rec["input_current_pa"],
            "mean_winner_set_size": rec["mean_winner_set_size"],
            "mean_activity_density": rec["mean_activity_density"],
            "density_floor_passed_at_scale": rec["density_floor_passed"],
        } if rec else {"rule": "no g on the grid restored the band at C1 lane size"}
    )

    if not args.skip_confirm and isinstance(manifest["recommendation"].get("gain"), (int, float)):
        g_star = float(manifest["recommendation"]["gain"])
        already = any(c.get("gain") == g_star for c in manifest["confirmations"])
        if not already:
            # Confirm on the SAME policy the recommendation was measured on:
            # under an artifact-blind policy the gate reads zeros regardless
            # of lane physics (see telemetry_inconsistent_rows_... evidence).
            dev_star = str(manifest["recommendation"].get("device_policy_used", "auto"))
            print(f"[4/4] confirming at N={max(args.grid_n)}, g={g_star:.0f}, "
                  f"device={dev_star} over >= {CONFIRM_STEPS_MIN} steps...", flush=True)
            conf = confirm(tokens, lane_cells=max(args.grid_n), gain=g_star,
                           device=dev_star, min_steps=CONFIRM_STEPS_MIN)
            manifest["confirmations"].append(conf)
            print(f"      gate-would-pass={conf['prereg_gate_would_pass']} "
                  f"|W|={conf['mean_winner_set_size']:.2f} "
                  f"density={conf['mean_activity_density']:.2e}", flush=True)

    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"manifest: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
