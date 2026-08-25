"""Gate analysis for the replay-compare dev-gate (read-only over metrics JSONs).

Preregistered accounting (experiments/replay_compare/PLAN.md, frozen pre-run):
- PRIMARY: calib_cases_passed fraction delta (errtop21 - rand21) > 0, pooled
  over all 9 (N, seed) cells, Newcombe method-10 hybrid-score 95% CI excluding
  zero on the positive side => PASS. Per-N breakdowns and per-cell paired
  deltas are REPORTED, not gating.
- SECONDARY: no catastrophic forgetting - errtop21 full-train (126-case)
  fraction drop vs the same-cell all42 control < 5 pp at EVERY cell AND pooled
  => PASS. rand21 drops reported alongside (diagnostic).
- Exploratory, non-gating: replay counts, margins, selection composition,
  overlap, and a replication cross-check of all42_N512_s* against
  schedule_compare/massed_N512_s* (identical effective configuration).

CI method: Wilson score intervals per proportion (z = 1.959963984540054),
combined into Newcombe method-10 intervals for differences of pooled
independent binomials (same implementation as schedule_compare/_analyze_gates.py).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "replay_compare"
ARMS = ("all42", "errtop21", "rand21")
NS = (512, 1024)
SEEDS = (7, 8, 9)
Z = 1.959963984540054


def load_runs() -> dict[tuple[str, int, int], dict]:
    runs: dict[tuple[str, int, int], dict] = {}
    for arm in ARMS:
        for n in NS:
            for seed in SEEDS:
                path = OUT / f"{arm}_N{n}_s{seed}.json"
                rec = json.loads(path.read_text(encoding="utf-8"))
                expected_arm = f"{arm}_N{n}_s{seed}"
                assert rec["arm"] == expected_arm and rec["N"] == n and rec["seed"] == seed, path
                assert rec["readout_rule"] == "ridge" and rec["schedule"] == "massed"
                assert rec["supervision"] == "all_transition"
                assert rec["data_scale"] == 42
                runs[(arm, n, seed)] = rec
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


def pool(runs, arm: str, field_pass: str, n_filter: int | None = None):
    field_total = field_pass.replace("passed", "total")
    k = sum(
        r[field_pass]
        for (a, n, s), r in runs.items()
        if a == arm and (n_filter is None or n == n_filter)
    )
    tot = sum(
        r[field_total]
        for (a, n, s), r in runs.items()
        if a == arm and (n_filter is None or n == n_filter)
    )
    return k, tot


def main() -> int:
    runs = load_runs()

    print("== pairing invariants ==")
    for n in NS:
        for seed in SEEDS:
            digests = {
                runs[(a, n, seed)]["curriculum_sha256"] for a in ARMS
            }
            assert len(digests) == 1, f"cell N{n} s{seed}: curriculum mismatch"
    print("  one curriculum_sha256 per cell across arms: OK (9 cells)")

    print("\n== per-arm table (train eval always full-42/126; calib /54) ==")
    for n in NS:
        for seed in SEEDS:
            for arm in ARMS:
                r = runs[(arm, n, seed)]
                ext = r["replay_extension_v1"]
                print(
                    f"{arm:>8} N={n:<4} s={seed}  "
                    f"k={ext['k_replayed']:>2}  "
                    f"train {r['train_cases_passed']:>3}/{r['train_cases_total']} "
                    f"({r['train_cases_passed'] / r['train_cases_total']:.4f})  "
                    f"label-replay {r['train_replay_tasks']:>2}/42  "
                    f"calib {r['calib_cases_passed']:>2}/{r['calib_cases_total']} "
                    f"({r['calib_cases_passed'] / r['calib_cases_total']:.4f})  "
                    f"margin_med {r['margin_median_pa']:+.1f} pA"
                )

    print("\n== PRIMARY GATE: errtop21 vs rand21 on CALIB cases (pooled) ==")
    primary_pass = False
    for label, nf in (("N=512", 512), ("N=1024", 1024), ("pooled", None)):
        ke, ne = pool(runs, "errtop21", "calib_cases_passed", nf)
        kr, nr = pool(runs, "rand21", "calib_cases_passed", nf)
        d, lo, hi = newcombe_diff(ke, ne, kr, nr)
        ok = lo > 0
        if nf is None:
            primary_pass = ok
        print(
            f"  {label}: errtop21 {ke}/{ne} ({ke / ne:.4f}) vs rand21 {kr}/{nr} "
            f"({kr / nr:.4f})  {fmt_ci(d, lo, hi)}  CI>0: {ok}"
        )
    paired = []
    for n in NS:
        for seed in SEEDS:
            re_, rr = runs[("errtop21", n, seed)], runs[("rand21", n, seed)]
            t = re_["calib_cases_total"]
            paired.append((re_["calib_cases_passed"] - rr["calib_cases_passed"]) / t)
    print(
        f"  per-cell paired deltas (9): {[f'{x:+.4f}' for x in paired]}  "
        f"positive: {sum(x > 0 for x in paired)}/9"
    )
    verdict_primary = "PASS" if primary_pass else "FAIL"
    print(f"  PRIMARY VERDICT (pooled Newcombe CI excludes zero): {verdict_primary}")

    print("\n== SECONDARY GATE: forgetting guard (< 5 pp full-train drop vs all42) ==")

    def drops(arm: str, nf: int | None) -> list[float]:
        out = []
        for (a, n, s), r in sorted(runs.items()):
            if a != arm or (nf is not None and n != nf):
                continue
            control = runs[("all42", n, s)]
            out.append(
                control["train_cases_passed"] / control["train_cases_total"]
                - r["train_cases_passed"] / r["train_cases_total"]
            )
        return out

    secondary_pass = True
    for arm in ("errtop21", "rand21"):
        tag = "gated" if arm == "errtop21" else "diagnostic"
        for label, nf in (("N=512", 512), ("N=1024", 1024), ("pooled", None)):
            ka, na = pool(runs, arm, "train_cases_passed", nf)
            kc, nc = pool(runs, "all42", "train_cases_passed", nf)
            pooled_drop = kc / nc - ka / na
            per_cell = drops(arm, nf)
            worst = max(per_cell)
            print(
                f"  {arm:>8} [{tag}] {label}: control {kc}/{nc} ({kc / nc:.4f}) vs "
                f"{ka}/{na} ({ka / na:.4f})  drop={pooled_drop * 100:+.2f} pp  "
                f"worst-cell={worst * 100:+.2f} pp"
            )
            if arm == "errtop21":
                if pooled_drop >= 0.05 or worst >= 0.05:
                    secondary_pass = False
    verdict_secondary = "PASS" if secondary_pass else "FAIL"
    print(f"  SECONDARY VERDICT (errtop21 drop < 5pp everywhere): {verdict_secondary}")

    print("\n== EXPLORATORY (non-gating): selection composition & overlap ==")
    for n in NS:
        for seed in SEEDS:
            ctrl = runs[("all42", n, seed)]["replay_extension_v1"]["per_task_errors"]
            hist = {e: sum(1 for r in ctrl if r["error_count"] == e) for e in (3, 2, 1, 0)}
            top = set(runs[("errtop21", n, seed)]["replay_extension_v1"]["selected_train_indices"])
            rnd = set(runs[("rand21", n, seed)]["replay_extension_v1"]["selected_train_indices"])
            errors_of_top = [
                row["error_count"] for row in ctrl if row["train_index"] in top
            ]
            zero_in_top = sum(1 for e in errors_of_top if e == 0)
            print(
                f"  N={n:<4} s={seed}: error hist {hist}; top21 min-error "
                f"{min(errors_of_top)} (zero-error tasks pulled in by tie-break: "
                f"{zero_in_top}); |top∩rand|={len(top & rnd)}"
            )

    print("\n== replication cross-check: all42_N512_s* vs schedule_compare/massed_N512_s* ==")
    for seed in SEEDS:
        mine = runs[("all42", 512, seed)]
        pilot_path = ROOT / "experiments" / "schedule_compare" / f"massed_N512_s{seed}.json"
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        keys = [k for k in mine if k not in ("experiment_id", "arm", "wall_s",
                                             "git_head", "replay_extension_v1")]
        int_diffs = [k for k in keys if isinstance(mine[k], int) and mine[k] != pilot.get(k)]
        float_diffs = [
            k for k in keys
            if isinstance(mine[k], float)
            and abs(mine[k] - pilot.get(k, float("nan"))) > 1e-6
        ]
        status = "REPLICATES" if not int_diffs and not float_diffs else f"DIFFERS int={int_diffs} float={float_diffs}"
        print(f"  s={seed}: {status}")

    print(f"\nFINAL: primary={verdict_primary}  secondary={verdict_secondary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
