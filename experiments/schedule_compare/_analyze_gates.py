"""Gate analysis for the a013 schedule-compare experiment (read-only over metrics JSONs).

Preregistered accounting (docs/research/a013-schedule-experiment-spec.md):
- Primary: train_cases_passed fraction delta (spaced - massed), Wilson 95% CI
  excluding zero -> PASS.
- Secondary: interleaved - blocked calibration-cases delta > 0, CI excluding zero,
  where "blocked" maps onto the arms table as {massed, spaced} pooled (the arms
  table defines no separate 'blocked' arm; both are non-interleaved structures).
- Exploratory (non-gating): margin distribution shift (median; p10 not persisted
  by schema v1 -- noted, not imputed).

CI method: Wilson score intervals per proportion (z = 1.959964), combined into a
Newcombe hybrid-score ("square-and-add") interval for the difference of two
independent pooled binomials. Gate passes iff the interval lies strictly above 0.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "schedule_compare"
SCHEDES = ("massed", "spaced", "interleaved")
NS = (128, 512)
SEEDS = (7, 8, 9)
Z = 1.959963984540054


def load_runs() -> dict[tuple[str, int, int], dict]:
    runs: dict[tuple[str, int, int], dict] = {}
    for sched in SCHEDES:
        for n in NS:
            for seed in SEEDS:
                path = OUT / f"{sched}_N{n}_s{seed}.json"
                rec = json.loads(path.read_text(encoding="utf-8"))
                assert rec["schedule"] == sched and rec["N"] == n and rec["seed"] == seed
                assert rec["readout_rule"] == "ridge"
                assert rec["data_scale"] == 42
                assert rec["supervision"] == "all_transition"
                runs[(sched, n, seed)] = rec
    return runs


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def newcombe_diff(k1: int, n1: int, k0: int, n0: int) -> tuple[float, float, float]:
    """Newcombe method-10 CI for p1 - p0 from independent binomials."""
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1)
    l0, u0 = wilson(k0, n0)
    d = p1 - p0
    lower = d - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    upper = d + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return d, lower, upper


def fmt_ci(d: float, lo: float, hi: float) -> str:
    return f"delta={d:+.4f}  Newcombe95%=[{lo:+.4f}, {hi:+.4f}]"


def main() -> None:
    runs = load_runs()
    print("== per-arm table ==")
    rows = []
    for sched in SCHEDES:
        for n in NS:
            for seed in SEEDS:
                r = runs[(sched, n, seed)]
                rows.append(
                    f"{sched:>11} N={n:<4} s={seed}  "
                    f"train {r['train_cases_passed']:>3}/{r['train_cases_total']} "
                    f"({r['train_cases_passed'] / r['train_cases_total']:.4f})  "
                    f"replay {r['train_replay_tasks']:>2}/42  "
                    f"calib {r['calib_cases_passed']:>2}/{r['calib_cases_total']} "
                    f"({r['calib_cases_passed'] / r['calib_cases_total']:.4f})  "
                    f"margin_med {r['margin_median_pa']:+.1f} pA"
                )
    print("\n".join(rows))

    curricula = {r["curriculum_sha256"] for r in runs.values()}
    print(f"\ncurriculum_sha256 distinct: {len(curricula)} (expect 3, one per seed)")

    def pool(scheds: list[str], field_pass: str, n_filter: int | None = None):
        k = sum(runs[(s, n, sd)][field_pass] for s in scheds for n in NS for sd in SEEDS
                if n_filter is None or n == n_filter)
        tot = sum(runs[(s, n, sd)][field_pass.replace("passed", "total")]
                  for s in scheds for n in NS for sd in SEEDS
                  if n_filter is None or n == n_filter)
        return k, tot

    def paired_deltas(s1: str, s0: str, field_pass: str, n_filter: int | None = None):
        out = []
        for n in NS:
            if n_filter is not None and n != n_filter:
                continue
            for sd in SEEDS:
                r1, r0 = runs[(s1, n, sd)], runs[(s0, n, sd)]
                t = r1[field_pass.replace("passed", "total")]
                out.append((r1[field_pass] - r0[field_pass]) / t)
        return out

    print("\n== PRIMARY GATE: spaced vs massed, TRAIN cases ==")
    primary_pass = True
    for label, nf in (("N=128", 128), ("N=512", 512), ("pooled N", None)):
        ks, ns_ = pool(["spaced"], "train_cases_passed", nf)
        km, nm = pool(["massed"], "train_cases_passed", nf)
        d, lo, hi = newcombe_diff(ks, ns_, km, nm)
        ok = lo > 0
        if nf is None:
            primary_pass = ok
        print(f"  {label}: spaced {ks}/{ns_} ({ks/ns_:.4f}) vs massed {km}/{nm} ({km/nm:.4f})  "
              f"{fmt_ci(d, lo, hi)}  CI>0: {ok}")
    pd = paired_deltas("spaced", "massed", "train_cases_passed")
    print(f"  per-(N,seed) deltas (6 pairs): {[f'{x:+.4f}' for x in pd]}  positive: {sum(x > 0 for x in pd)}/6")
    verdict_primary = "PASS" if primary_pass else "FAIL"
    print(f"  PRIMARY VERDICT (pooled-N Newcombe CI excludes zero on the preregistered side): {verdict_primary}")

    print("\n== SECONDARY GATE: interleaved vs blocked(=massed+spaced pooled), CALIB cases ==")
    secondary_pass = True
    for label, inter, blocked in (("N=128", ["interleaved"], ["massed", "spaced"]),
                                  ("N=512", ["interleaved"], ["massed", "spaced"]),
                                  ("pooled N", ["interleaved"], ["massed", "spaced"])):
        nf = None if label == "pooled N" else int(label.split("=")[1])
        ki, ni = pool(inter, "calib_cases_passed", nf)
        kb, nb = pool(blocked, "calib_cases_passed", nf)
        d, lo, hi = newcombe_diff(ki, ni, kb, nb)
        ok = lo > 0
        if nf is None:
            secondary_pass = ok
        print(f"  {label}: interleaved {ki}/{ni} ({ki/ni:.4f}) vs blocked {kb}/{nb} ({kb/nb:.4f})  "
              f"{fmt_ci(d, lo, hi)}  CI>0: {ok}")
    pd_cal = paired_deltas("interleaved", "massed", "calib_cases_passed") + \
             paired_deltas("interleaved", "spaced", "calib_cases_passed")
    print(f"  per-(N,seed) deltas vs each blocked arm (12 pairs): "
          f"{[f'{x:+.4f}' for x in pd_cal]}  positive: {sum(x > 0 for x in pd_cal)}/12")
    verdict_secondary = "PASS" if secondary_pass else "FAIL"
    print(f"  SECONDARY VERDICT: {verdict_secondary}")

    print("\n== EXPLORATORY (non-gating): margin medians (pA) ==")
    for sched in SCHEDES:
        meds = [runs[(sched, n, sd)]["margin_median_pa"] for n in NS for sd in SEEDS]
        print(f"  {sched:>11}: median-of-medians {sorted(meds)[len(meds)//2]:+.1f}, range [{min(meds):+.1f}, {max(meds):+.1f}]")

    print("\n== replication cross-check vs readout_compare ridge arms (same config: massed/all_transition/scale42/N{128,512}) ==")
    for n in NS:
        for sd in SEEDS:
            pilot = json.loads((ROOT / "experiments" / "readout_compare" / f"ridge_N{n}_s{sd}.json").read_text(encoding="utf-8"))
            mine = runs[("massed", n, sd)]
            keys = [k for k in mine if k not in ("experiment_id", "arm", "wall_s", "git_head")]
            diff = [k for k in keys if mine[k] != pilot.get(k)]
            print(f"  massed_N{n}_s{sd}: {'IDENTICAL' if not diff else 'DIFFERS: ' + str(diff)}")


if __name__ == "__main__":
    sys.exit(main())
