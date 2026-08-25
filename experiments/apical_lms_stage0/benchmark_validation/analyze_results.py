#!/usr/bin/env python3
"""Assemble comparison tables from benchmark_validation/results/*.json.

Prints markdown tables used verbatim in FINDINGS.md:
  T1  validation at DendriCL's published operating points x gamma arms;
  T2  seed stability vs their reported sigma <= 0.036;
  T3  width sweep (the empirical width-cliff map).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
ARMS = ("frozen", "gstar_a99", "gstar_a996")
ARM_LABEL = {"frozen": "frozen (a=.99, g=.2/(d_ap+2))",
             "gstar_a99": "gstar (a=.99, g=.2/(d_task+2))",
             "gstar_a996": "gstar+a996 (a=.996, g=.2/(d_task+2))"}


def load(name: str) -> dict | None:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def f4(x: float) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "  --  "


def table1() -> str:
    points = [("val_d10_primary", 10, 0.807, 0.873),
              ("val_d20_secondary", 20, 0.820, 0.931),
              ("super_d25", 25, 0.809, None),
              ("super_d30", 30, 0.807, 0.97),
              ("super_d40", 40, 0.787, None),
              ("super_d50", 50, 0.649, None)]
    out = []
    out.append("| d_task | DendriCL published task R² | DendriCL published probe R² | "
               "arm | our probe R² (gate metric) | our abstract-ref task R² | "
               "our decoded-through-membrane task R² | Δ decoded vs published task |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for base, d, pub_task, pub_probe in points:
        for arm in ARMS:
            r = load(f"{base}_{arm}")
            if r is None or "error" in r:
                out.append(f"| {d} | {pub_task} | {pub_probe} | {arm} | MISSING | | | |")
                continue
            delta = r["task_decoded_r2_vs_wx"] - pub_task
            out.append(
                f"| {d} | {pub_task:.3f} | {pub_probe if pub_probe is not None else '--'} "
                f"| {arm} | {f4(r['probe_mean_r2'])} | {f4(r['task_ref_r2_vs_wx'])} "
                f"| {f4(r['task_decoded_r2_vs_wx'])} | {delta:+.4f} |"
            )
    return "\n".join(out)


def table2() -> str:
    out = ["| leg (arm=gstar_a996) | seeds | mean probe R² | σ probe R² | "
           "mean decoded task R² | σ decoded task R² | DendriCL seed σ |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, d in (("seeds_d10", 10), ("seeds_d30", 30)):
        anchor = ("val_d10_primary_gstar_a996" if d == 10
                  else "super_d30_gstar_a996")
        base = load(anchor)
        rows = [load(f"{name}_s{s}") for s in ((11, 22, 33) if d == 10 else (11, 22))]
        rows = [r for r in rows if r and "error" not in r]
        if base:
            rows.append(base)
        if not rows:
            continue
        pr = [r["probe_mean_r2"] for r in rows]
        tr = [r["task_decoded_r2_vs_wx"] for r in rows]
        import statistics as st
        out.append(f"| {name} (incl. base-seed anchor) | {len(rows)} | {f4(st.mean(pr))} "
                   f"| {f4(st.stdev(pr)) if len(rows) > 1 else 'n/a'} "
                   f"| {f4(st.mean(tr))} | {f4(st.stdev(tr)) if len(rows) > 1 else 'n/a'} "
                   f"| ≤ 0.036 |")
    return "\n".join(out)


def table3() -> str:
    out = ["| d_apical = cell_count | γ | bound 2/(d_ap+2) | arm | mean spikes/token | "
           "probe R² | abstract-ref task R² | decoded task R² |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for width in (10, 32, 128, 384):
        for arm in ("joint", "gstar_a99"):
            r = load(f"width{width}_d10_{arm}")
            if r is None or "error" in r:
                out.append(f"| {width} | | | {arm} | MISSING | | | |")
                continue
            out.append(
                f"| {width} | {r['gamma']:.3e} | {r['stability_bound_at_d_apical']:.3e} "
                f"| {arm} | {r['mean_spikes_per_token']:.1f} | {f4(r['probe_mean_r2'])} "
                f"| {f4(r['task_ref_r2_vs_wx'])} | {f4(r['task_decoded_r2_vs_wx'])} |"
            )
    return "\n".join(out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "t1"):
        print("### T1 — Validation at DendriCL's published operating points\n")
        print(table1())
    if which in ("all", "t2"):
        print("\n### T2 — Seed stability\n")
        print(table2())
    if which in ("all", "t3"):
        print("\n### T3 — Width sweep\n")
        print(table3())
