"""Render FINDINGS.md table blocks from a completed artifact JSON.

Usage: python fill_findings.py <artifact.json> [--post]
Prints markdown: baseline-style matrix table (+ mirror-read line); with
--post also a seam-detection provenance line.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    data = json.loads(Path(sys.argv[1]).read_text())
    post = "--post" in sys.argv
    rows = data["rows"]
    by_key = {(r["mode"], r["activity_pct_of_configured"]): r for r in rows}
    pcts = sorted({r["activity_pct_of_configured"] for r in rows})

    print("| Activity | Fixture | EP ms/tok | EP tok/s | FT ms/tok | FT tok/s | EP/FT | EP cell-steps/tok | FT cell-steps/tok |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    ratios = data.get("ep_over_ft_ratio_by_activity", {})
    for pct in pcts:
        ep = by_key.get(("event_proportional", pct))
        ft = by_key.get(("full_touch", pct))

        def cell(r, key):
            if r is None:
                return "-"
            return r["stream"].get(key, "-")

        fixture = (
            ep or ft
        )["capacity_plan"]["fixture_residents"]
        ev = (ep or ft)["events_per_token"]
        print(
            f"| {pct}% ({ev // 1000}k ev/tok) | {fixture:,} "
            f"| {cell(ep, 'median_token_ms')} | {cell(ep, 'tokens_per_second_end_to_end')} "
            f"| {cell(ft, 'median_token_ms')} | {cell(ft, 'tokens_per_second_end_to_end')} "
            f"| {ratios.get(f'{pct}pct', '-')} "
            f"| {cell(ep, 'cell_steps_per_token_median_call'):,} "
            f"| {cell(ft, 'cell_steps_per_token_median_call'):,} |"
        )

    mirrors = []
    for pct in pcts:
        ep = by_key.get(("event_proportional", pct))
        ft = by_key.get(("full_touch", pct))
        for r in (ep, ft):
            m = r.get("mirror_read", {}) if r else {}
            if not m:
                continue
            tag = f"{r['mode'][:2]}{pct}"
            if m.get("path") == "read_cells_optimized":
                mirrors.append(
                    f"{tag}: {m['median_read_us']}us/token "
                    f"({m['overhead_pct_of_token']}% of token)"
                )
    print()
    print("Mirror-read (K=16, read_cells optimized path): " + "; ".join(mirrors))

    if post:
        det = data["seam_detection"]
        prov = det.get("provenance", {})
        print()
        print(
            f"Run provenance: commit `{data['commit'][:9]}`, "
            f"seam_level `{det['seam_level']}` "
            f"(scan-arm median {det['indexed_staged']['scan_heavy_median_ms']}ms vs "
            f"{det['indexed_staged']['linear_floor_ms']}ms floor; source digest "
            f"`{str(prov.get('source_digest'))[:12]}`, procedural_rs_dirty="
            f"{prov.get('procedural_rs_dirty')})."
        )


if __name__ == "__main__":
    main()
