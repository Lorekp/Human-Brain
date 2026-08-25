#!/usr/bin/env python3
"""T4-subset: r0-c G-A3 span protocol with apical features, 4 corner arms.

Full 18-arm sweep exceeded budget (~7 min PER softmax fit at 2048x673).
This subset covers the grid corners + the committed reference arm, with
per-arm JSON checkpoints:
  A50_0.75_0.6   (best GA1 conjunction arm)
  A100_0.25_0.6  (committed-reference arm)
  A200_0.25_0.3  (slow-apical, low-gate corner)
  A200_0.75_0.6  (slow-apical, high-gate corner)
Feature sets: apical_only, voltage_plus_apical; CONTROL_SEED=11 shuffled
controls identical to probes_decode_worker.py; plus harness-fidelity check
(reproduces committed GA3 number for A100_0.25_0.6/raw_voltage_spikes).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

from retro_probe import DEFAULT_NPZ, load_r0c, run_t4_arm

OUT = HERE / "t4_subset_results.json"
ARMS = ["A50_0.75_0.6", "A100_0.25_0.6", "A200_0.25_0.3", "A200_0.75_0.6"]
FREEZE_DIR = (Path(os.environ.get("LOCALAPPDATA",
                                  str(Path.home() / "AppData" / "Local")))
              / "Temp" / "hb_bible_freeze")


def log(m):
    print(f"[{time.strftime('%H:%M:%Z', time.gmtime())}] {m}", flush=True)


def main():
    man = json.loads((HERE.parent / "logged_states_apical_cohort_manifest.json")
                     .read_text(encoding="utf-8"))
    name_to_idx = {a: i for i, a in enumerate(man["arm_names"])}
    z = np.load(DEFAULT_NPZ)
    r0c = load_r0c()

    results = {}
    if OUT.exists():
        results = json.loads(OUT.read_text(encoding="utf-8"))

    for arm in ARMS:
        if arm in results.get("arms", {}):
            log(f"{arm}: cached")
            continue
        t0 = time.perf_counter()
        res = run_t4_arm(r0c, z, name_to_idx[arm], str(FREEZE_DIR),
                         ("apical_only", "voltage_plus_apical"))
        results.setdefault("arms", {})[arm] = res
        json.dump(results, open(OUT, "w"), indent=1)
        log(f"{arm} done in {time.perf_counter()-t0:.0f}s "
            f"apical_only={res['apical_only']['logistic_span_accuracy']:.6f} "
            f"(shuffle {res['apical_only']['shuffled_label_control_span_accuracy']:.6f})")

    # harness fidelity vs committed G-A3
    if "harness_fidelity" not in results:
        fid = run_t4_arm(r0c, z, name_to_idx["A100_0.25_0.6"], str(FREEZE_DIR),
                         ("raw_voltage_spikes",))
        committed = json.loads(
            (HERE.parent / "probes_results.json").read_text(encoding="utf-8")
        )["GA3_per_arm"]["A100_0.25_0.6"]["raw_voltage_spikes"]["logistic"]
        results["harness_fidelity"] = {
            "note": ("re-run of a committed G-A3 cell through this script; "
                     "must equal committed number exactly"),
            "reproduced_span_accuracy":
                fid["raw_voltage_spikes"]["logistic_span_accuracy"],
            "committed_span_accuracy": committed["accuracy"],
            "exact_match": bool(
                fid["raw_voltage_spikes"]["logistic_span_accuracy"]
                == committed["accuracy"]),
        }
        json.dump(results, open(OUT, "w"), indent=1)
        log(f"fidelity exact_match={results['harness_fidelity']['exact_match']}")

    # summary across subset
    for fs in ("apical_only", "voltage_plus_apical"):
        accs = [(v[fs]["logistic_span_accuracy"], v[fs][
                    "shuffled_label_control_span_accuracy"], a)
                for a, v in results["arms"].items()]
        accs.sort(reverse=True)
        log(f"{fs}: best {accs[0][2]} acc={accs[0][0]:.6f} "
            f"(its shuffle={accs[0][1]:.6f}); range "
            f"[{min(a[0] for a in accs):.6f},{max(a[0] for a in accs):.6f}]; "
            f"shuffle-control range [{min(a[1] for a in accs):.6f},"
            f"{max(a[1] for a in accs):.6f}]")
    log("done")


if __name__ == "__main__":
    main()
