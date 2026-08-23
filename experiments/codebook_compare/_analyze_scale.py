"""Scale analysis for the codebook_compare N-scaling extension.

Reads the same per-arm metrics JSONs as ``_analyze.py`` and adds the
scale-up questions the 27-run matrix could not answer:

- does the similarity capacity advantage at N=512 (0.735 vs 0.587 vs
  explicit) persist, grow, or vanish at N >= 4096?
- does ANY arm's calibration (transfer) fraction move above the ~0.15
  plateau as N grows two-plus orders of magnitude beyond the primary rung?

Reuses the statistics helpers from ``_analyze.py`` unchanged (Wilson /
Newcombe hybrid-score). Usage::

    python _analyze_scale.py [--dir experiments/codebook_compare]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "codebook_analyze_base", HERE / "_analyze.py"
)
assert _SPEC is not None and _SPEC.loader is not None
BASE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("codebook_analyze_base", BASE)
_SPEC.loader.exec_module(BASE)

ARMS = BASE.ARMS
SEEDS = BASE.SEEDS
CALIB_CASES = BASE.CALIB_CASES
NS_BASELINE = (128, 512, 1024)
NS_SCALE = (4096, 16384, 32768)
ALL_NS = NS_BASELINE + NS_SCALE


def frac(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def spread(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="experiments/codebook_compare")
    arguments = parser.parse_args(argv)
    directory = Path(arguments.dir)
    records = BASE.load_records(directory)

    train_cases: dict[tuple[str, int], list[int]] = {}
    train_totals: dict[tuple[str, int], list[int]] = {}
    train_replay: dict[tuple[str, int], list[int]] = {}
    train_tasks: dict[tuple[str, int], list[int]] = {}
    margins: dict[tuple[str, int], list[float]] = {}
    calib: dict[tuple[str, int], list[int]] = {}
    calib_totals: dict[tuple[str, int], list[int]] = {}
    backends: dict[tuple[str, int], set[str]] = {}
    train_task_totals: dict[tuple[str, int], list[int]] = {}

    for record in records:
        arm = str(record.get("arm"))
        n_cells = int(record["N"])
        codebook = BASE.codebook_of(record)
        key = (codebook, n_cells)
        train_cases.setdefault(key, []).append(int(record["train_cases_passed"]))
        train_totals.setdefault(key, []).append(int(record["train_cases_total"]))
        train_replay.setdefault(key, []).append(int(record["train_replay_tasks"]))
        margins.setdefault(key, []).append(float(record["margin_median_pa"]))
        calib.setdefault(key, []).append(int(record["calib_cases_passed"]))
        calib_totals.setdefault(key, []).append(int(record["calib_cases_total"]))
        backend = record.get("config_dump", {}).get("simulation_backend")
        backends.setdefault(key, set()).add(str(backend))
        count = int(record.get("config_dump", {}).get("train_task_count") or 0)
        train_task_totals.setdefault(key, []).append(count)

    present_scale = [n for n in NS_SCALE if any((a, n) in calib for a in ARMS)]
    print(f"# records loaded: {len(records)} from {directory}")
    print(f"# scale Ns present: {present_scale or 'none yet'}")
    for (codebook, n_cells), values in sorted(backends.items()):
        print(f"# backend {codebook} @ N={n_cells}: {sorted(values)}")

    print("\n## capacity table by N (TRAIN, mean ± spread over seeds)")
    header = "| codebook | N | train_cases frac | train_replay frac | margin_median_pa |"
    print(header)
    print("| --- | --- | --- | --- | --- |")
    for n_cells in ALL_NS:
        for codebook in ARMS:
            key = (codebook, n_cells)
            if key not in train_cases:
                continue
            case_fracs = [
                frac(c, t) for c, t in zip(train_cases[key], train_totals[key])
            ]
            replay_fracs = [
                frac(r, t) for r, t in zip(train_replay[key], train_task_totals[key])
            ]
            print(
                f"| {codebook} | {n_cells} "
                f"| {mean(case_fracs):.4f} ± {spread(case_fracs):.4f} "
                f"| {mean(replay_fracs):.4f} ± {spread(replay_fracs):.4f} "
                f"| {mean(margins[key]):.4f} ± {spread(margins[key]):.4f} |"
            )

    print("\n## transfer table by N (CALIBRATION)")
    print("| codebook | N | calib_cases frac | per-seed passes |")
    print("| --- | --- | --- | --- |")
    for n_cells in ALL_NS:
        for codebook in ARMS:
            key = (codebook, n_cells)
            if key not in calib:
                continue
            fracs = [
                frac(c, t) for c, t in zip(calib[key], calib_totals[key])
            ]
            passes = "/".join(str(value) for value in sorted(calib[key]))
            print(
                f"| {codebook} | {n_cells} "
                f"| {mean(fracs):.4f} ± {spread(fracs):.4f} | {passes} |"
            )

    print("\n## plateau readout (calibration fraction by N, per arm)")
    for codebook in ARMS:
        points = []
        for n_cells in ALL_NS:
            key = (codebook, n_cells)
            if key in calib:
                fracs = [
                    frac(c, t) for c, t in zip(calib[key], calib_totals[key])
                ]
                points.append((n_cells, mean(fracs)))
        rendered = ", ".join(f"N={n}:{value:.3f}" for n, value in points)
        print(f"- {codebook}: {rendered}")

    print("\n## delta CIs on calibration fraction (Newcombe hybrid-score)")
    for n_cells in ALL_NS:
        for left, right in (("similarity", "random"), ("random", "explicit")):
            lk, rk = (left, n_cells), (right, n_cells)
            if lk not in calib or rk not in calib:
                continue
            x1, n1 = sum(calib[lk]), sum(calib_totals[lk])
            x2, n2 = sum(calib[rk]), sum(calib_totals[rk])
            result = BASE.newcombe_diff(x1, n1, x2, n2)
            if result is None:
                continue
            d, lo, hi = result
            excl = lo > 0 or hi < 0
            print(
                f"- {left} − {right} @ N={n_cells}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] "
                f"({'excludes' if excl else 'includes'} zero)"
            )

    print("\n## capacity advantage deltas (train_cases frac, similarity − random)")
    for n_cells in ALL_NS:
        sk, rk = ("similarity", n_cells), ("random", n_cells)
        if sk not in train_cases or rk not in train_cases:
            continue
        s_fracs = [
            frac(c, t) for c, t in zip(train_cases[sk], train_totals[sk])
        ]
        r_fracs = [
            frac(c, t) for c, t in zip(train_cases[rk], train_totals[rk])
        ]
        d = mean(s_fracs) - mean(r_fracs)
        print(f"- N={n_cells}: similarity−random capacity delta = {d:+.4f}")

    print("\n## state separation by N (sealed diagnostic, ei_sweep analog)")
    print("| codebook | N | elig 1-cos pooled | syncurr 1-cos pooled | pairs |")
    print("| --- | --- | --- | --- | --- |")
    for n_cells in ALL_NS:
        for codebook in ARMS:
            key = (codebook, n_cells)
            if key not in calib:
                continue
            values = []
            for path in sorted(directory.glob("*.json")):
                if path.name.startswith("_"):
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                if BASE.codebook_of(record) != codebook or int(record["N"]) != n_cells:
                    continue
                block = record.get("diagnostics", {}).get("state_separation")
                if not block:
                    continue
                eligibility = block["eligibility"]["median_1minuscos_pooled"]
                synaptic_current = block["synaptic_current"]["median_1minuscos_pooled"]
                pairs = block["eligibility"]["pairs_Int"] + block["eligibility"]["pairs_List_Int"]
                values.append((eligibility, synaptic_current, pairs))
            if not values:
                continue
            eligibility_values = [value[0] for value in values if value[0] is not None]
            synaptic_values = [value[1] for value in values if value[1] is not None]
            pairs_seen = sorted({value[2] for value in values})
            mean_elig = mean(eligibility_values) if eligibility_values else float("nan")
            mean_syn = mean(synaptic_values) if synaptic_values else float("nan")
            print(
                f"| {codebook} | {n_cells} | {mean_elig:.6f} ± {spread(eligibility_values) if eligibility_values else 0:.6f} "
                f"| {mean_syn:.6f} ± {spread(synaptic_values) if synaptic_values else 0:.6f} "
                f"| {pairs_seen} |"
            )

    wall = [r.get("wall_s") for r in records if isinstance(r.get("wall_s"), (int, float))]
    heads = sorted({str(r.get("git_head")) for r in records})
    print(f"\n# git_head(s) of records: {heads}")
    print(
        f"# total wall_s across {len(records)} records "
        f"(shared box, informational only): {sum(wall):.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
