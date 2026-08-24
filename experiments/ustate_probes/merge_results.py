#!/usr/bin/env python3
"""Merge per-arm view probe results into the canonical record (verbatim).

Reads experiments/*/view_*/probes_results.json produced by the UNCHANGED
r0c probes.py invoked once per arm-view, verifies controls are identical
across views, and writes:

- experiments/ustate_probes/probes_results_merged.json  (all six arms)
- prints nothing else; run verdict.py afterwards for the gate evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACT_DIRS = (
    REPO / "experiments" / "ustate_probes",
    REPO / "experiments" / "ustate_probes_concept",
)


def main() -> None:
    merged = None
    sources: dict[str, str] = {}
    for base in ARTIFACT_DIRS:
        for view in sorted(base.glob("view_*")):
            res_path = view / "probes_results.json"
            if not res_path.exists():
                continue
            res = json.loads(res_path.read_text(encoding="utf-8"))
            if merged is None:
                merged = res
                merged["P1"]["arms"] = {}
                merged["P2"]["arms"] = {}
                merged["P3"] = {}
            else:
                assert res["P1"]["controls"] == merged["P1"]["controls"], (
                    f"controls diverge at {view}"
                )
                assert (
                    res["comparators"] == merged["comparators"]
                ), f"comparators diverge at {view}"
            arm_keys_p1 = list(res["P1"]["arms"])
            assert len(arm_keys_p1) == 1, arm_keys_p1
            key = arm_keys_p1[0]
            merged["P1"]["arms"][key] = res["P1"]["arms"][key]
            p2_key = list(res["P2"]["arms"])
            assert len(p2_key) == 1 and p2_key[0] == key
            merged["P2"]["arms"][key] = res["P2"]["arms"][key]
            p3_key = list(res["P3"])
            assert len(p3_key) == 1 and p3_key[0] == key
            merged["P3"][key] = res["P3"][key]
            sources[key] = str(view.relative_to(REPO))

    assert merged is not None, "no per-arm results found"
    merged["merge_note"] = {
        "execution": (
            "UNCHANGED probes.py invoked once per bit-identical per-arm view "
            "artifact (make_arm_views.py) for wall-clock feasibility on the "
            "throttled CI host; results merged verbatim; controls verified "
            "identical across views"
        ),
        "sources": sources,
    }
    out = REPO / "experiments" / "ustate_probes" / "probes_results_merged.json"
    out.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} with arms: {sorted(merged['P1']['arms'])}")


if __name__ == "__main__":
    main()
