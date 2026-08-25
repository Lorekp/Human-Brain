#!/usr/bin/env python3
"""T5: summary-statistic control probes for the retro-probe.

Question: is the strong T1 family decoding (0.85-0.93) carried by
DISTRIBUTED structure in the continuous states, or by trivial global
activity / prompt-length proxies?

Variants (same strict split, fixed C=0.01, 200 perms, identical pipeline):
  sum7      : [mean(v), std(v), max|v|, frac_active(v),
               mean(u_A), std(u_A), total_somatic_spikes]
  sum7_pb   : sum7 + prompt_bytes
  pb_only   : [prompt_bytes]
If sum7/pb_only already reach T1-level accuracy, T1 evidence is confounded
by gross scale; if they sit near chance while the full vector decodes,
the structure is genuinely distributed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
NPZ = (r"C:/Users/laure/Desktop/DeepSim/.worktrees/"
       r"human-brain-a013-cognition-design/experiments/apical_probes/"
       r"logged_states_apical_cohort.npz")
MANIFEST = HERE.parent / "logged_states_apical_cohort_manifest.json"
OUT = HERE / "t5_summary_controls.json"

FIXED_C = 0.01
N_PERMS = 200
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


def band(obs, nulls):
    return {
        "observed": float(obs), "shuffle_mean": float(nulls.mean()),
        "shuffle_sd": float(nulls.std(ddof=1)),
        "shuffle_min": float(nulls.min()), "shuffle_max": float(nulls.max()),
        "shuffle_q95": float(np.quantile(nulls, 0.95)),
        "effect_abs": float(obs - nulls.mean()),
        "empirical_p_one_sided": float(
            (1 + int((nulls >= obs).sum())) / (1 + len(nulls))),
    }


def main():
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = man["rows"]
    tr_rows = [r for r in rows if r["split"] == "train"]
    va_rows = [r for r in rows if r["split"] == "val"]
    y_tr = np.array([FAMILIES.index(r["family"]) for r in tr_rows])
    y_va = np.array([FAMILIES.index(r["family"]) for r in va_rows])
    pb_tr = np.array([r["prompt_bytes"] for r in tr_rows], dtype=np.float64)
    pb_va = np.array([r["prompt_bytes"] for r in va_rows], dtype=np.float64)

    # family-wise prompt-length stats (confound audit)
    audit = {}
    for f_i, f in enumerate(FAMILIES):
        m_tr = pb_tr[y_tr == f_i]
        m_va = pb_va[y_va == f_i]
        audit[f] = {"train_prompt_bytes_mean": float(m_tr.mean()),
                    "train_prompt_bytes_sd": float(m_tr.std()),
                    "val_prompt_bytes_mean": float(m_va.mean())}
    log("prompt_bytes by family: " + json.dumps(audit))

    z = np.load(NPZ)
    arms = man["arm_names"]

    def summaries(split):
        v = z[f"{split}_boundary_voltage"].astype(np.float64)   # n x 18 x C
        a = z[f"{split}_boundary_apical"].astype(np.float64)
        sp = z[f"{split}_total_somatic_spikes"].astype(np.float64)
        feats = np.stack([
            v.mean(axis=2), v.std(axis=2), np.abs(v).max(axis=2),
            (np.abs(v) > 1e-9).mean(axis=2),
            a.mean(axis=2), a.std(axis=2),
            np.broadcast_to(sp[:, None], v.shape[:2]),
        ], axis=2)  # n x 18 x 7
        return feats

    S_tr = summaries("train")
    S_va = summaries("val")

    def run_variant(name, build):
        out = {}
        for ai in range(18):
            Xtr, Xva = build(S_tr[:, ai, :], S_va[:, ai, :], ai)
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            Xtr_s, Xva_s = (Xtr - mu) / sd, (Xva - mu) / sd
            rng = np.random.default_rng(np.random.SeedSequence(
                MASTER_SEED + 3000).spawn(54)[ai])
            m = LogisticRegression(C=FIXED_C, max_iter=1000).fit(Xtr_s, y_tr)
            pred = m.predict(Xva_s)
            obs = float((pred == y_va).mean())
            bacc = float(balanced_accuracy_score(y_va, pred))
            lo, hi = wilson(int((pred == y_va).sum()), len(y_va))
            nulls = np.empty(N_PERMS)
            for i in range(N_PERMS):
                mm = LogisticRegression(C=FIXED_C, max_iter=1000)
                mm.fit(Xtr_s, rng.permutation(y_tr))
                nulls[i] = (mm.predict(Xva_s) == y_va).mean()
            out[arms[ai]] = {
                "val_accuracy": obs, "wilson95": [lo, hi],
                "val_balanced_accuracy": bacc,
                "shuffle_accuracy": band(obs, nulls)}
            log(f"{name} {arms[ai]}: acc={obs:.4f} "
                f"(band mean={nulls.mean():.4f}) "
                f"p={out[arms[ai]]['shuffle_accuracy']['empirical_p_one_sided']:.3f}")
        return out

    results = {
        "schema_version": "humanbrain/retro-probe-t5/v1",
        "fixed_C": FIXED_C, "n_perms": N_PERMS,
        "prompt_bytes_audit_by_family": audit,
        "variants": {}}

    def b_sum7(tr7, va7, ai):
        return tr7, va7

    def b_sum7_pb(tr7, va7, ai):
        return (np.concatenate([tr7, pb_tr[:, None]], axis=1),
                np.concatenate([va7, pb_va[:, None]], axis=1))

    def b_pb_only(tr7, va7, ai):
        return pb_tr[:, None], pb_va[:, None]

    for name, fn in (("sum7", b_sum7), ("sum7_plus_prompt_bytes", b_sum7_pb),
                     ("prompt_bytes_only", b_pb_only)):
        log(f"T5 variant {name}")
        results["variants"][name] = run_variant(name, fn)
        json.dump(results, open(OUT, "w"), indent=1)

    # headline comparison
    log("SUMMARY (best arm per variant, family3):")
    for name, per_arm in results["variants"].items():
        accs = [(cfg["val_accuracy"], arm) for arm, cfg in per_arm.items()]
        accs.sort(reverse=True)
        ps = [cfg["shuffle_accuracy"]["empirical_p_one_sided"]
              for cfg in per_arm.values()]
        log(f"  {name}: acc range [{min(a for a,_ in accs):.4f},"
            f"{max(a for a,_ in accs):.4f}] best={accs[0][1]} "
            f"min_p={min(ps):.3f}")
    log("done -> " + str(OUT))


if __name__ == "__main__":
    main()
