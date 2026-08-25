#!/usr/bin/env python3
"""Dense-resample driver: re-log ONE U-STATE arm at stride 1 (every token).

Archive-saturation follow-up to the windowed retro-probe
(experiments/windowed_probes/FINDINGS.md, "Archive saturation caveat"):
the week-1 archives logged stride-16 snapshots, so items averaged ~8.8
eligible history snapshots and k=64 windows were only 8-14% filled. This
driver re-materializes ONE arm (default lam097_concept) over the SAME
freeze corpus / seed / decision rule with ``--stride 1`` -- the densest
possible observation-sink sampling -- by reusing the UNCHANGED week-1
harness (experiments/ustate_probes/log_states_union.py): same replay,
same boundary capture, same manifest schema, only the stride differs.

The npz + manifest land in an out-dir (local-only binaries; commit the
manifest as the hash anchor). The curriculum digest is stride-independent
and must equal the committed concept digest f0bc99f6..., which the
downstream windowed scorer verifies before scoring.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "ustate_probes"))

import log_states_union as lsu  # noqa: E402  (week-1 harness, unchanged)


def arm_to_lambda(arm: str) -> tuple[float, str]:
    """Map 'lam097_concept' -> (0.97, 'concept') over the frozen sweep."""
    for lam in lsu.LAMBDA_SWEEP:
        base = f"lam{int(round(lam * 100)):03d}"
        if arm == f"{base}_concept":
            return float(lam), "concept"
        if arm == base:
            return float(lam), "byte"
    raise SystemExit(
        f"unknown arm {arm!r}; frozen sweep offers: "
        + ", ".join(
            f"lam{int(round(l * 100)):03d}[_concept]" for l in lsu.LAMBDA_SWEEP
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", default="lam097_concept")
    p.add_argument("--stride", type=int, default=1,
                   help="snapshot every Nth token (1 = every token)")
    p.add_argument("--out-dir", required=True,
                   help="where the dense npz+manifest are written (local-only)")
    p.add_argument("--freeze-dir", default=str(lsu.DEFAULT_FREEZE_DIR))
    args = p.parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    lam, encoding = arm_to_lambda(args.arm)
    if encoding != "concept":
        # byte arms additionally run gate0/curriculum anchors that expect the
        # week-1 byte out-dir layout; this follow-up is concept-scoped.
        raise SystemExit("dense resample supports concept arms only")

    # Reuse the UNCHANGED harness end-to-end with the sweep narrowed to the
    # one requested arm. Everything else (seed 7, n=2048/w=32 projection,
    # tokenizer fit, boundary capture, manifest writing) is bit-identical
    # to the week-1 code path.
    lsu.LAMBDA_SWEEP = [lam]
    sys.argv = [
        "log_states_union.py",
        "--freeze-dir", args.freeze_dir,
        "--out-dir", args.out_dir,
        "--seed", str(lsu.DEFAULT_SEED),
        "--stride", str(args.stride),
        "--encoding", encoding,
    ]
    lsu.main()


if __name__ == "__main__":
    main()
