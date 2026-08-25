#!/usr/bin/env python3
"""APICAL-AND offline probes (preregistration §4-5, commit 05fb3ef).

Consumes the cohort artifacts and evaluates the frozen gates per arm:

- G-A1 conjunction index (from conjunction_results_cohort.json, already run);
- G-A2 separation: R0-G1 definition on VAL decision-point trace-bank vectors
  (f32, bias excluded): pooled MEDIAN 1 - cosine over unordered pairs within
  the same QA family stratum; PASS iff > 0.10;
- G-A3 decode z >= 2 vs unigram: span accuracy via the r0-c machinery VERBATIM
  (logistic + 2-layer MLP, TRAIN-answer-vocab head, ranked-consumption,
  639-position denominator); PASS iff best decoder >= 0.061851478566709396
  (= unigram 29/639 + 2 sigma). Controls recomputed identically.

Decode arms run as isolated subprocesses with per-arm JSON checkpoints
(probes_decode_worker.py) and bounded retry rounds, so an externally killed
worker loses at most one arm instead of wedging a result queue.

Emits probes_results.json with every number verbatim plus the mechanical
per-arm and program verdicts (including the pre-declared falsifier clause).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

R0C_PATH = REPO / "experiments" / "r0c_probes" / "probes.py"
RSPEC = importlib.util.spec_from_file_location("r0c_probes", R0C_PATH)
assert RSPEC is not None and RSPEC.loader is not None
R0C = importlib.util.module_from_spec(RSPEC)
sys.modules["r0c_probes"] = R0C
RSPEC.loader.exec_module(R0C)

SCHEMA_VERSION = "humanbrain/apical-verdict/v1"
GA2_THRESHOLD = 0.10                 # median 1-cosine must exceed this
GA3_THRESHOLD = 0.061851478566709396  # unigram 29/639 + 2 sigma (frozen)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def separation_1minus_cos(matrix: np.ndarray, families: list[str]) -> dict:
    """Pooled median 1-cosine over unordered pairs within a family stratum."""
    norms = np.linalg.norm(matrix, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    unit = matrix / safe[:, None]
    cos = unit @ unit.T
    one_minus = 1.0 - cos
    values: list[float] = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if families[i] == families[j]:
                values.append(float(one_minus[i, j]))
    return {
        "n_pairs": len(values),
        "median_1minus_cos": float(np.median(values)) if values else None,
    }


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", default=str(
        Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        / "Temp" / "hb_bible_freeze"))
    parser.add_argument("--workers", type=int,
                        default=max(2, min(6, (os.cpu_count() or 4) - 4)))
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()

    conj_path = OUT_DIR / "conjunction_results_cohort.json"
    npz_path = OUT_DIR / "logged_states_apical_cohort.npz"
    manifest_path = OUT_DIR / "logged_states_apical_cohort_manifest.json"

    conjunction = json.loads(conj_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    z = np.load(npz_path)

    arm_names = manifest["arm_names"]
    rows = manifest["rows"]
    val_meta = [r for r in rows if r["split"] == "val"]
    val_families = [r["family"] or "unknown" for r in val_meta]

    log(f"evaluating {len(arm_names)} arms: G-A2 separation + G-A3 decode")

    # ---------------- G-A2 separation (cheap, inline) ------------------------
    ga2 = {}
    for arm_index, arm in enumerate(arm_names):
        traces_val = z["val_boundary_traces"][:, arm_index, :].astype(np.float64)
        ga2[arm] = separation_1minus_cos(traces_val, val_families)
    log("G-A2 done for all arms")

    # ---------------- G-A3 decode (isolated subprocess per arm) --------------
    cache_dir = OUT_DIR / "decode_cache"
    worker_script = OUT_DIR / "probes_decode_worker.py"
    pending = [
        (i, arm) for i, arm in enumerate(arm_names)
        if not (cache_dir / f"arm_{arm}.json").exists()
    ]
    log(f"G-A3 decode: {len(arm_names) - len(pending)} cached, "
        f"{len(pending)} to run on {args.workers} subprocess slots")

    round_no = 0
    while pending and round_no < args.rounds:
        round_no += 1
        log(f"G-A3 decode round {round_no}: {len(pending)} arms")
        # simple process-slot scheduler; each arm is one isolated subprocess
        running: dict[subprocess.Popen, tuple[int, str]] = {}
        queue = list(pending)
        finished: list[tuple[int, str]] = []
        while queue or running:
            while queue and len(running) < args.workers:
                i, arm = queue.pop(0)
                proc = subprocess.Popen(
                    [sys.executable, str(worker_script),
                     "--arm-index", str(i), "--npz", str(npz_path),
                     "--freeze-dir", args.freeze_dir,
                     "--out", str(cache_dir / f"arm_{arm}.json")],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                running[proc] = (i, arm)
            time.sleep(5)
            for proc in list(running):
                code = proc.poll()
                if code is None:
                    continue
                i, arm = running.pop(proc)
                output = proc.stdout.read() if proc.stdout else ""
                if code == 0 and (cache_dir / f"arm_{arm}.json").exists():
                    finished.append((i, arm))
                    log(f"G-A3 {arm} OK ({len(finished)}/{len(arm_names)} total)")
                else:
                    tail = "\n".join(output.splitlines()[-3:])
                    log(f"G-A3 {arm} FAILED rc={code}: {tail}")
        pending = [
            (i, arm) for i, arm in pending
            if (i, arm) not in finished and not (cache_dir / f"arm_{arm}.json").exists()
        ]

    ga3: dict[str, dict] = {}
    missing = []
    for i, arm in enumerate(arm_names):
        path = cache_dir / f"arm_{arm}.json"
        if path.exists():
            ga3[arm] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(arm)
    if missing:
        log(f"WARNING: decode incomplete for {missing} after {args.rounds} rounds")

    # ---------------- verdict assembly ---------------------------------------
    reference_arm = next(a for a in arm_names if a in ga3)
    majority_prior = (
        ga3[reference_arm]["controls"]["majority_prior_within_answer_vocab"]["accuracy"]
    )
    verdicts = {}
    falsified_arms = []
    passing_arms = []
    for arm in arm_names:
        g1 = conjunction["summary_per_arm"][arm]["GA1_pass"]
        med = ga2[arm]["median_1minus_cos"]
        g2 = bool(med is not None and med > GA2_THRESHOLD)
        if arm in ga3:
            best_acc = max(
                max(ga3[arm][fs]["logistic"]["accuracy"],
                    ga3[arm][fs]["mlp"]["accuracy"])
                for fs in ("raw_voltage_spikes", "traces_bank")
            )
            g3 = bool(best_acc >= GA3_THRESHOLD)
            g3_note = "evaluated"
        else:
            best_acc = None
            g3 = False
            g3_note = "not evaluated (decode incomplete)"
        arm_pass = bool(g1 and g2 and g3)
        verdicts[arm] = {
            "GA1_conjunction_pass": g1,
            "GA1_median_ci": conjunction["summary_per_arm"][arm]["median_ci"],
            "GA2_separation_median_1minus_cos": med,
            "GA2_pass": g2,
            "GA3_best_span_accuracy": best_acc,
            "GA3_pass": g3,
            "GA3_note": g3_note,
            "ARM_PASS": arm_pass,
        }
        if g1 and best_acc is not None and best_acc < majority_prior:
            falsified_arms.append({
                "arm": arm,
                "GA1_median_ci": conjunction["summary_per_arm"][arm]["median_ci"],
                "best_decode": best_acc,
                "majority_prior": majority_prior,
            })
        if arm_pass:
            passing_arms.append(arm)

    program_verdict = "PASS" if passing_arms else "NEGATIVE-FALSIFIED"
    results = {
        "schema_version": SCHEMA_VERSION,
        "gates": {
            "GA2_threshold_median_1minus_cos": GA2_THRESHOLD,
            "GA3_threshold_span_accuracy": GA3_THRESHOLD,
        },
        "majority_prior_within_answer_vocab": majority_prior,
        "comparators": {
            "unigram_smoothed": R0C.UNIGRAM_SPAN_ACC,
            "trigram_smoothed": R0C.TRIGRAM_SPAN_ACC,
        },
        "GA2_per_arm": ga2,
        "GA3_per_arm": ga3,
        "GA3_unevaluated_arms": missing,
        "verdicts_per_arm": verdicts,
        "passing_arms": passing_arms,
        "falsifier_triggered_arms": falsified_arms,
        "PROGRAM_VERDICT": program_verdict,
    }
    (OUT_DIR / "probes_results.json").write_text(
        json.dumps(results, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=True, indent=1),
        encoding="utf-8",
    )
    log(f"PROGRAM VERDICT: {program_verdict}")
    for arm, v in verdicts.items():
        acc = v["GA3_best_span_accuracy"]
        log(f"  {arm}: G1={v['GA1_conjunction_pass']} (CI={v['GA1_median_ci']}) "
            f"G2={v['GA2_pass']} ({v['GA2_separation_median_1minus_cos']:.4f}) "
            f"G3={acc if acc is not None else 'n/e'}")


if __name__ == "__main__":
    main()
