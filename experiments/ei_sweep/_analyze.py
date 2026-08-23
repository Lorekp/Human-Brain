"""Aggregate ei_sweep records and evaluate the PREREGISTERED gates G1/G2.

Gate definitions are quoted verbatim from PREREGISTRATION.md (frozen before
execution); this script ONLY computes them from the metrics JSONs.
Writes _analysis_output.txt and prints to stdout.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FRACTIONS = (0.0, 0.1, 0.2, 0.35)
SEEDS = (7, 8, 9)


def load() -> dict[float, dict[int, dict[str, object]]]:
    table: dict[float, dict[int, dict[str, object]]] = defaultdict(dict)
    for path in sorted(HERE.glob("f*_N*_s*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        fraction = float(record["config_dump"]["inhibitory_fraction"])
        assert fraction in FRACTIONS, fraction
        assert int(record["N"]) == 1024
        table[fraction][int(record["seed"])] = record
    return table


def fmt_mean_sd(values: list[float]) -> str:
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"


def main() -> int:
    table = load()
    missing = [
        (f, s)
        for f in FRACTIONS
        for s in SEEDS
        if s not in table.get(f, {})
    ]
    assert not missing, f"incomplete matrix: {missing}"

    lines: list[str] = []
    lines.append("== Per-arm aggregates (mean ± sd over seeds 7/8/9) ==")
    lines.append(
        "fraction | calib_cases(/54) | train_cases(/126) | train_replay(/42) "
        "| margin_median_pa | margin_min_pa | wall_s"
    )
    for fraction in FRACTIONS:
        records = [table[fraction][s] for s in SEEDS]
        col = lambda key: [float(r[key]) for r in records]  # noqa: E731
        lines.append(
            f"{fraction:.3f} | {fmt_mean_sd(col('calib_cases_passed'))} | "
            f"{fmt_mean_sd(col('train_cases_passed'))} | "
            f"{fmt_mean_sd(col('train_replay_tasks'))} | "
            f"{fmt_mean_sd(col('margin_median_pa'))} | "
            f"{fmt_mean_sd(col('margin_min_pa'))} | "
            f"{fmt_mean_sd(col('wall_s'))}"
        )

    lines.append("")
    lines.append("== Per-seed raw (calib/54 · margin_median_pa · train/126) ==")
    for fraction in FRACTIONS:
        cells = [
            (
                f"s{s}: {table[fraction][s]['calib_cases_passed']}/54 · "
                f"{table[fraction][s]['margin_median_pa']:.1f} · "
                f"{table[fraction][s]['train_cases_passed']}/126"
            )
            for s in SEEDS
        ]
        lines.append(f"f={fraction:.3f}: " + " | ".join(cells))

    # ------------------------------------------------------------------
    # Preregistered gates (PREREGISTRATION.md, frozen before execution)
    # ------------------------------------------------------------------
    base = 0.0
    paired: dict[float, dict[str, list[int]]] = {}
    for fraction in FRACTIONS[1:]:
        paired[fraction] = {
            metric: [
                int(
                    table[fraction][s][key if metric == "calib" else "margin_median_pa"]
                    > table[base][s][key if metric == "calib" else "margin_median_pa"]
                )
                for s in SEEDS
            ]
            for metric, key in (("calib", "calib_cases_passed"), ("margin", ""))
        }

    lines.append("")
    lines.append("== Paired per-seed wins vs f=0.0 (need >=2/3 BOTH metrics) ==")
    g1_candidates: list[float] = []
    for fraction in FRACTIONS[1:]:
        calib_wins = sum(paired[fraction]["calib"])
        margin_wins = sum(paired[fraction]["margin"])
        ok = calib_wins >= 2 and margin_wins >= 2
        if fraction in (0.1, 0.2) and ok:
            g1_candidates.append(fraction)
        lines.append(
            f"f={fraction:.3f}: calib wins {calib_wins}/3 "
            f"(per-seed {paired[fraction]['calib']}), "
            f"margin_median wins {margin_wins}/3 "
            f"(per-seed {paired[fraction]['margin']}) -> G1-candidate: {ok}"
        )

    g1 = bool(g1_candidates)

    def mean_of(fraction: float, key: str) -> float:
        return float(np.mean([float(table[fraction][s][key]) for s in SEEDS]))

    g2_parts: list[str] = []
    g2 = False
    if g1:
        f_star = max(
            g1_candidates,
            key=lambda f: (
                mean_of(f, "calib_cases_passed"),
                mean_of(f, "margin_median_pa"),
            ),
        )
        m_035 = mean_of(0.35, "margin_median_pa")
        m_star = mean_of(f_star, "margin_median_pa")
        c_035 = mean_of(0.35, "calib_cases_passed")
        c_star = mean_of(f_star, "calib_cases_passed")
        part_a = m_035 < m_star
        part_b = c_035 <= c_star
        g2 = part_a and part_b
        g2_parts = [
            f"f* = {f_star:.3f}",
            f"margin_median(0.35)={m_035:.4f} < margin_median(f*)={m_star:.4f}: {part_a}",
            f"calib(0.35)={c_035:.4f} <= calib(f*)={c_star:.4f}: {part_b}",
        ]
    else:
        g2_parts.append("G2 not evaluated: G1 failed")

    lines.append("")
    lines.append("== VERDICTS (preregistered) ==")
    lines.append(f"G1 (intermediate beats 0.0 on calib AND margin): {g1}")
    for part in g2_parts:
        lines.append(f"G2: {part}")
    verdict = "CONFIRMED" if (g1 and g2) else ("PARTIAL" if g1 else "FALSIFIED")
    lines.append(f"P-EI: {verdict}")

    text = "\n".join(lines) + "\n"
    (HERE / "_analysis_output.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
