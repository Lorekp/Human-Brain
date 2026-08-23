"""Analysis for the codebook_compare decisive representation experiment.

Reads the per-arm metrics JSONs written by benchmarks/ladder_experiments.py
into experiments/codebook_compare/, verifies cross-arm curriculum identity,
builds the capacity and transfer tables, computes Newcombe hybrid-score
(Wilson-based) CIs on calibration deltas, and emits verdict inputs for
FINDINGS.md.

Design split:
- CONFIRMATORY (preregistered): 3 codebooks x N {128,512} x seeds {7,8,9} = 18.
- EXTENSION (tasked capacity point): same arms at N=1024 = 9 more (27 total).

Usage: python _analyze.py [--dir experiments/codebook_compare]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

Z = 1.959963984540054
ARMS = ("explicit", "random", "similarity")
NS_PRIMARY = (128, 512)
NS_EXTENSION = (1024,)
ALL_NS = NS_PRIMARY + NS_EXTENSION
SEEDS = (7, 8, 9)
CALIB_CASES = 54  # fixed 18-task calibration cohort, 3 cases each


def wilson(x: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_diff(
    x1: int, n1: int, x2: int, n2: int, z: float = Z
) -> tuple[float, float, float] | None:
    """Newcombe hybrid-score CI for p1 - p2 (independent proportions).

    Method 10 of Newcombe (1998): square-and-add from the two Wilson intervals.
    Returns None when either denominator is zero.
    """
    if n1 == 0 or n2 == 0:
        return None
    l1, u1 = wilson(x1, n1, z)
    l2, u2 = wilson(x2, n2, z)
    p1, p2 = x1 / n1, x2 / n2
    d = p1 - p2
    lower = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (d, lower, upper)


def load_records(directory: Path) -> list[dict]:
    records = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_path"] = path.name
        records.append(record)
    return records


def codebook_of(record: dict) -> str:
    for key in ("codebook", "codebook_kind", "input_codebook"):
        if key in record:
            return str(record[key])
    if "config_dump" in record:
        for key in ("codebook", "codebook_kind", "input_codebook"):
            if key in record["config_dump"]:
                return str(record["config_dump"][key])
    arm = str(record.get("arm", ""))
    for candidate in ARMS:
        if arm.startswith(candidate):
            return candidate
    raise KeyError(f"cannot determine codebook for {record['_path']}")


def frac(num: int, den: int) -> float:
    return num / den if den else 0.0


def msd(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1 or statistics.pstdev(values) == 0.0:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="experiments/codebook_compare")
    arguments = parser.parse_args()
    directory = Path(arguments.dir)
    records = load_records(directory)

    expected = {(a, n, s) for a in ARMS for n in ALL_NS for s in SEEDS}
    seen: dict[tuple[str, int, int], dict] = {}
    curricula: dict[tuple[int, int], set[str]] = {}
    codebook_hashes: dict[tuple[str, int], set[str]] = {}
    for record in records:
        arm = codebook_of(record)
        n_cells = int(record["N"])
        seed = int(record["seed"])
        seen[(arm, n_cells, seed)] = record
        curricula.setdefault((seed, n_cells), set()).add(str(record["curriculum_sha256"]))
        dump = record.get("config_dump", {})
        cb_hash = dump.get("codebook_sha256") or dump.get("codebook_provenance_sha256")
        if cb_hash:
            codebook_hashes.setdefault((arm, n_cells), set()).add(str(cb_hash))

    missing_primary = {
        (a, n, s)
        for a in ARMS
        for n in NS_PRIMARY
        for s in SEEDS
    } - set(seen)
    missing_ext = {
        (a, n, s) for a in ARMS for n in NS_EXTENSION for s in SEEDS
    } - set(seen)
    extra = set(seen) - expected
    print(
        f"# completeness: {len(seen)}/{len(expected)} records; "
        f"missing PRIMARY={sorted(missing_primary)}; missing EXTENSION={sorted(missing_ext)}; "
        f"extra={sorted(extra)}"
    )
    mixed = {k: v for k, v in curricula.items() if len(v) > 1}
    print(
        "# curriculum identity across arms within seed/N: "
        + ("OK (identical)" if not mixed else f"MIXED: {mixed!r}")
    )
    unstable_cb = {k: v for k, v in codebook_hashes.items() if len(v) > 1}
    if codebook_hashes:
        print(
            "# codebook provenance hash stability per (arm,N) across seeds: "
            + ("stable" if not unstable_cb else f"UNSTABLE: {unstable_cb!r}")
        )

    print("\n## capacity table (TRAIN)")
    print("| codebook | N | train_cases frac (/126) | train_replay frac (/42) | margin_median_pa |")
    print("| --- | --- | --- | --- | --- |")
    for arm in ARMS:
        for n_cells in ALL_NS:
            subset = [seen[(arm, n_cells, s)] for s in SEEDS if (arm, n_cells, s) in seen]
            case_f = [frac(r["train_cases_passed"], r["train_cases_total"]) for r in subset]
            rep_f = [
                frac(r["train_replay_tasks"], r["config_dump"]["train_task_count"])
                for r in subset
            ]
            marg = [r["margin_median_pa"] for r in subset if r.get("margin_count")]
            print(f"| {arm} | {n_cells} | {msd(case_f)} | {msd(rep_f)} | {msd(marg)} |")

    print("\n## transfer table (CALIBRATION)")
    print("| codebook | N | calib_cases frac (/54) | per-seed passes (/54) |")
    print("| --- | --- | --- | --- |")
    calib_counts: dict[tuple[str, int], list[int]] = {}
    for arm in ARMS:
        for n_cells in ALL_NS:
            subset = [seen[(arm, n_cells, s)] for s in SEEDS if (arm, n_cells, s) in seen]
            fracs = [frac(r["calib_cases_passed"], r["calib_cases_total"]) for r in subset]
            raw = [str(r["calib_cases_passed"]) for r in subset]
            calib_counts[(arm, n_cells)] = [int(r["calib_cases_passed"]) for r in subset]
            print(f"| {arm} | {n_cells} | {msd(fracs)} | {'/'.join(raw)} |")

    def pooled_counts(arm: str, ns_list: tuple[int, ...]) -> tuple[int, int]:
        xs = sum((calib_counts.get((arm, n), []) for n in ns_list), [])
        return sum(xs), CALIB_CASES * len(xs)

    print("\n## delta CIs on calibration-cases fraction (Newcombe hybrid-score, confirmatory cells)")
    print("| comparison | scope | delta | 95% CI | excludes zero |")
    print("| --- | --- | --- | --- | --- |")
    verdict_inputs: dict[str, tuple[int, int, float, float, float]] = {}
    for left, right in (("similarity", "random"), ("random", "explicit")):
        for label, ns_scope in (("N=128", (128,)), ("N=512", (512,)), ("pooled {128,512}", NS_PRIMARY)):
            x1, n1 = pooled_counts(left, ns_scope)
            x2, n2 = pooled_counts(right, ns_scope)
            result = newcombe_diff(x1, n1, x2, n2)
            if result is None:
                print(f"| {left} − {right} | {label} | INCOMPLETE | — | — |")
                continue
            d, lo, hi = result
            excl = lo > 0 or hi < 0
            print(f"| {left} − {right} | {label} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {'YES' if excl else 'no'} |")
            if label.startswith("pooled"):
                verdict_inputs[f"{left}−{right}"] = (x1, n1, d, lo, hi)

    print("\n## extension cells (N=1024, exploratory)")
    for left, right in (("similarity", "random"), ("random", "explicit")):
        x1, n1 = pooled_counts(left, NS_EXTENSION)
        x2, n2 = pooled_counts(right, NS_EXTENSION)
        result = newcombe_diff(x1, n1, x2, n2)
        if result is None:
            print(f"- {left} − {right}: INCOMPLETE")
            continue
        d, lo, hi = result
        excl = lo > 0 or hi < 0
        print(
            f"- {left} − {right} @ N=1024 pooled: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] "
            f"({'excludes' if excl else 'includes'} zero)"
        )

    print("\n## per-seed paired deltas (robustness; seeds ordered 7,8,9)")
    for left, right in (("similarity", "random"), ("random", "explicit")):
        for n_cells in ALL_NS:
            cl = calib_counts.get((left, n_cells), [])
            cr = calib_counts.get((right, n_cells), [])
            if not cl or not cr:
                continue
            deltas = [cl[i] - cr[i] for i in range(min(len(cl), len(cr)))]
            signs = ",".join("+" if d > 0 else ("-" if d < 0 else "0") for d in deltas)
            print(f"- {left} − {right} @ N={n_cells}: {deltas} (signs {signs})")

    print("\n## H-code verdict inputs (pooled confirmatory cells)")
    for tag, (x1, n1, d, lo, hi) in verdict_inputs.items():
        direction = ">" if lo > 0 else ("<" if hi < 0 else "≈")
        print(f"- {tag}: {x1}/{n1}, delta {d:+.4f}, CI [{lo:+.4f}, {hi:+.4f}] → {direction} 0")

    wall = [r.get("wall_s") for r in records if isinstance(r.get("wall_s"), (int, float))]
    heads = sorted({str(r.get("git_head")) for r in records})
    print(f"\n# git_head(s) of records: {heads}")
    print(f"# total wall_s across {len(records)} records (shared box, informational only): {sum(wall):.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
