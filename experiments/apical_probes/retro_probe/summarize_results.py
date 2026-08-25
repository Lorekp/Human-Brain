#!/usr/bin/env python3
"""Summarize retro_probe_results.json into FINDINGS.md tables + verdict."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "retro_probe_results.json").read_text(encoding="utf-8"))

ARMS = 18


def bonf(p):
    return min(1.0, p * ARMS)


def fmt_band(b):
    return (f"obs={b['observed']:.4f} band=[{b['shuffle_min']:.4f},"
            f"{b['shuffle_max']:.4f}] mean={b['shuffle_mean']:.4f}"
            f" sd={b['shuffle_sd']:.4f} q95={b['shuffle_q95']:.4f}"
            f" p={b['empirical_p_one_sided']:.3f} p_adj={bonf(b['empirical_p_one_sided']):.3f}")


print("=" * 78)
print("T1 family3 strict split (primary) - accuracy")
print("=" * 78)
t1 = R.get("T1_family3_strict_split", {})
for fs in ("apical", "concat", "voltage"):
    if fs not in t1:
        continue
    print(f"\n-- feature set: {fs}")
    rows = []
    for arm, cfg in t1[fs].items():
        b = cfg["shuffle_accuracy"]
        rows.append((b["empirical_p_one_sided"], arm, cfg, b))
    rows.sort()
    for p, arm, cfg, b in rows:
        lo, hi = cfg["wilson95"]
        print(f"{arm:14s} acc={cfg['val_accuracy']:.4f} "
              f"wilson95=[{lo:.4f},{hi:.4f}] maj={cfg['majority_class_freq_val']:.4f} "
              f"bandmean={b['shuffle_mean']:.4f} q95={b['shuffle_q95']:.4f} "
              f"eff={b['effect_abs']:+.4f} d={b['effect_cohens_d']} "
              f"p={p:.3f} p_adj={bonf(p):.3f}")
    best_p, best_arm, _, _ = rows[0]
    print(f"   -> min p across arms: {best_p:.4f} ({best_arm}), "
          f"Bonferroni x18: {bonf(best_p):.4f}")

if "sensitivity_reference_arm_A100_0.25_0.6" in R:
    print("\n" + "=" * 78)
    print("Sensitivity grid (reference arm A100_0.25_0.6, true labels)")
    print("=" * 78)
    for fs, grid in R["sensitivity_reference_arm_A100_0.25_0.6"]["grid"].items():
        line = ", ".join(f"C={g['C']}: val={g['val_accuracy']:.4f} "
                         f"(cv={g['train_cv_accuracy_mean']:.4f})"
                         for g in grid)
        print(f"-- {fs}: {line}")

print("\n" + "=" * 78)
print("T2 speaker probes pooled-CV")
print("=" * 78)
t2 = R.get("T2_speaker_pooledCV", {})
for tn, by_fs in t2.items():
    for fs, per_arm in by_fs.items():
        rows = []
        for arm, cfg in per_arm.items():
            b = cfg["shuffle_accuracy"]
            rows.append((b["empirical_p_one_sided"], arm, cfg, b))
        rows.sort()
        p0, arm0, cfg0, b0 = rows[0]
        print(f"\n-- {tn} / {fs}: best arm {arm0}")
        lo, hi = cfg0["wilson95"]
        acc_key = "cv_accuracy" if "cv_accuracy" in cfg0 else "val_accuracy"
        maj_key = ("majority_class_freq" if "majority_class_freq" in cfg0
                   else "majority_class_freq_val")
        bacc_key = ("cv_balanced_accuracy" if "cv_balanced_accuracy" in cfg0
                    else "val_balanced_accuracy")
        print(f"   acc={cfg0[acc_key]:.4f} wilson95=[{lo:.4f},{hi:.4f}] "
              f"maj={cfg0[maj_key]:.4f} "
              f"bacc={cfg0[bacc_key]:.4f} "
              f"bandmean={b0['shuffle_mean']:.4f} q95={b0['shuffle_q95']:.4f} "
              f"p={p0:.3f} p_adj={bonf(p0):.3f}")
        # range across arms
        accs = [r[2][acc_key] for r in rows]
        ps = [r[0] for r in rows]
        print(f"   all-arms acc range: [{min(accs):.4f},{max(accs):.4f}]; "
              f"min p={min(ps):.3f}")

print("\n" + "=" * 78)
print("T3 chapter_distance ridge strict split")
print("=" * 78)
t3 = R.get("T3_chapter_distance_ridge_strict_split", {})
for fs, per_arm in t3.items():
    rows = []
    for arm, cfg in per_arm.items():
        b = cfg["shuffle_R2"]
        rows.append((b["empirical_p_one_sided"], arm, cfg, b))
    rows.sort()
    p0, arm0, cfg0, b0 = rows[0]
    print(f"\n-- {fs}: best arm {arm0}")
    print(f"   R2={cfg0['val_R2']:.4f} pearson_r={cfg0['val_pearson_r']:.4f} "
          f"alpha={cfg0['alpha_selected']:.4g} "
          f"bandmean={b0['shuffle_mean']:.4f} q95={b0['shuffle_q95']:.4f} "
          f"p={p0:.3f} p_adj={bonf(p0):.3f}")
    r2s = [r[2]["val_R2"] for r in rows]
    print(f"   all-arms R2 range: [{min(r2s):.4f},{max(r2s):.4f}]")

print("\n" + "=" * 78)
print("T4 r0-c span protocol with apical features")
print("=" * 78)
fid = R.get("T4_harness_fidelity_check")
if fid:
    print(f"harness fidelity: reproduced={fid['reproduced_span_accuracy']!r} "
          f"committed={fid['committed_span_accuracy']!r} "
          f"exact_match={fid['exact_match']}")
t4 = R.get("T4_span_protocol_with_apical_features", {})
for fs in ("apical_only", "voltage_plus_apical"):
    rows = []
    for arm, cfg in t4.items():
        if fs not in cfg:
            continue
        rows.append((cfg[fs]["logistic_span_accuracy"], arm,
                     cfg[fs]["shuffled_label_control_span_accuracy"]))
    rows.sort(reverse=True)
    if not rows:
        continue
    accs = [r[0] for r in rows]
    shufs = [r[2] for r in rows]
    print(f"\n-- {fs}: best arm {rows[0][1]} acc={rows[0][0]:.17g} "
          f"(its shuffle control={rows[0][2]:.17g})")
    print(f"   acc range across arms: [{min(accs):.6f},{max(accs):.6f}]; "
          f"shuffle-control range: [{min(shufs):.6f},{max(shufs):.6f}]")

print("\ndone.")
