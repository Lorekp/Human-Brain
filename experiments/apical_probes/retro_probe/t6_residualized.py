#!/usr/bin/env python3
"""T6: prompt-length-residualized family probe (apical feature set).

Residualize every cell's voltage/apical feature against prompt_bytes
(linear coefficient fit on TRAIN only, applied to VAL), then rerun the
identical strict-split logistic probe (C=0.01, 100 perms).
Compares against the prompt_bytes_only baseline accuracy (0.5833):
survival well above that baseline implies distributed decision-point
structure beyond gross input echo.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
NPZ = (r"C:/Users/laure/Desktop/DeepSim/.worktrees/"
       r"human-brain-a013-cognition-design/experiments/apical_probes/"
       r"logged_states_apical_cohort.npz")
MANIFEST = HERE.parent / "logged_states_apical_cohort_manifest.json"
OUT = HERE / "t6_residualized_probe.json"

FIXED_C = 0.01
N_PERMS = 100
MASTER_SEED = 7
FAMILIES = ["who_said", "address_lookup", "verse_continuation"]


def log(m):
    print(f"[{time.strftime('%H:%M:%Z', time.gmtime())}] {m}", flush=True)


def wilson(k, n, zc=1.959963984540054):
    p = k / n
    den = 1 + zc * zc / n
    center = (p + zc * zc / (2 * n)) / den
    half = zc * np.sqrt(p * (1 - p) / n + zc * zc / (4 * n * n)) / den
    return center - half, center + half


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = man["rows"]
    tr_rows = [r for r in rows if r["split"] == "train"]
    va_rows = [r for r in rows if r["split"] == "val"]
    y_tr = np.array([FAMILIES.index(r["family"]) for r in tr_rows])
    y_va = np.array([FAMILIES.index(r["family"]) for r in va_rows])
    pb_tr = np.array([r["prompt_bytes"] for r in tr_rows], dtype=np.float64)
    pb_va = np.array([r["prompt_bytes"] for r in va_rows], dtype=np.float64)

    z = np.load(NPZ)
    arms = man["arm_names"]
    results = {"schema_version": "humanbrain/retro-probe-t6/v1",
               "fixed_C": FIXED_C, "n_perms": N_PERMS,
               "prompt_bytes_only_baseline_accuracy": 0.5833333333333334,
               "arms": {}}

    for ai in range(18):
        arm_out = {}
        for key, arr in (("apical", "apical"), ("voltage", "voltage")):
            Xtr = z[f"train_boundary_{arr}"][:, ai, :].astype(np.float64)
            Xva = z[f"val_boundary_{arr}"][:, ai, :].astype(np.float64)
            # residualize against prompt length (train-fit coefficients)
            A_tr = np.stack([pb_tr, np.ones_like(pb_tr)], axis=1)
            A_va = np.stack([pb_va, np.ones_like(pb_va)], axis=1)
            beta, *_ = np.linalg.lstsq(A_tr, Xtr, rcond=None)
            Xtr_r = Xtr - A_tr @ beta
            Xva_r = Xva - A_va @ beta
            mu, sd = Xtr_r.mean(0), Xtr_r.std(0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            Xtr_s, Xva_s = (Xtr_r - mu) / sd, (Xva_r - mu) / sd
            rng = np.random.default_rng(np.random.SeedSequence(
                MASTER_SEED + 4000).spawn(36)[ai])
            m = LogisticRegression(C=FIXED_C, max_iter=1000).fit(Xtr_s, y_tr)
            pred = m.predict(Xva_s)
            obs = float((pred == y_va).mean())
            lo, hi = wilson(int((pred == y_va).sum()), len(y_va))
            nulls = np.empty(N_PERMS)
            for i in range(N_PERMS):
                mm = LogisticRegression(C=FIXED_C, max_iter=1000)
                mm.fit(Xtr_s, rng.permutation(y_tr))
                nulls[i] = (mm.predict(Xva_s) == y_va).mean()
            arm_out[key] = {
                "val_accuracy": obs, "wilson95": [lo, hi],
                "shuffle_mean": float(nulls.mean()),
                "shuffle_q95": float(np.quantile(nulls, 0.95)),
                "effect_abs": float(obs - nulls.mean()),
                "empirical_p_one_sided": float(
                    (1 + int((nulls >= obs).sum())) / (1 + len(nulls)))}
        results["arms"][arms[ai]] = arm_out
        json.dump(results, open(OUT, "w"), indent=1)
        a_ = arm_out["apical"]
        log(f"{arms[ai]}: apical-resid acc={a_['val_accuracy']:.4f} "
            f"(p={a_['empirical_p_one_sided']:.3f}) "
            f"voltage-resid acc={arm_out['voltage']['val_accuracy']:.4f}")

    accs = [c["val_accuracy"] for a in results["arms"].values()
            for c in a.values()]
    log(f"T6 done: residualized acc range [{min(accs):.4f},{max(accs):.4f}] "
        f"(baseline prompt-bytes-only 0.5833)")


if __name__ == "__main__":
    main()
