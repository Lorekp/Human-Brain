#!/usr/bin/env python3
"""R2 Bible-QA pilot post-verdict analyzer (FINDINGS-style).

Reads the pilot record written by experiments/bible_qa_r2_pilot/run_pilot.py
plus the preregistered baselines in experiments/qa_baselines_bible_v1/ and
emits:

- a machine-readable analysis record (Wilson CIs, Newcombe 95% intervals on
  pilot-vs-baseline span-acc deltas, degeneracy-guard verdicts, per-family
  breakdown, mechanical rule readout);
- FINDINGS-style markdown ready to paste into the verdict document.

All logic lives in ``humanbrain.qa_analysis`` (importable, red-tested); this
CLI is a thin wrapper. Before the real pilot finishes you can exercise the
whole pipeline deterministically:

    python tools/analyze_bible_qa.py --synthesize-fixture fixture_record.json
    python tools/analyze_bible_qa.py --record fixture_record.json --stdout

stdlib-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from humanbrain.qa_analysis import (  # noqa: E402
    analyze_pilot_record,
    canonical_json_bytes,
    load_baseline_records,
    render_findings_markdown,
    synthesize_fixture_record,
)

DEFAULT_RECORD = ROOT / "experiments" / "bible_qa_r2_pilot" / "pilot_record.json"
DEFAULT_BASELINES = ROOT / "experiments" / "qa_baselines_bible_v1"


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--record",
        type=Path,
        default=DEFAULT_RECORD,
        help=f"pilot record JSON (default: {DEFAULT_RECORD})",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=DEFAULT_BASELINES,
        help=f"directory of baseline records (default: {DEFAULT_BASELINES})",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="analysis JSON output path (default: alongside the record)",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="FINDINGS markdown output path (default: alongside the record)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print the markdown to stdout as well",
    )
    parser.add_argument(
        "--synthesize-fixture",
        type=Path,
        default=None,
        metavar="PATH",
        help="write a deterministic synthetic pilot record to PATH and exit",
    )
    parser.add_argument("--fixture-seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.synthesize_fixture is not None:
        fixture = synthesize_fixture_record(args.fixture_seed)
        write_atomic(args.synthesize_fixture, json.dumps(fixture, indent=1, sort_keys=True).encode("utf-8") + b"\n")
        print(f"synthetic pilot record (seed={args.fixture_seed}) -> {args.synthesize_fixture}")
        return 0

    if not args.record.exists():
        parser.error(
            f"pilot record not found: {args.record}\n"
            "The record is written by experiments/bible_qa_r2_pilot/run_pilot.py "
            "when the pilot finishes. Until then, smoke-test the pipeline with:\n"
            "  python tools/analyze_bible_qa.py --synthesize-fixture fixture_record.json"
        )
    record = json.loads(args.record.read_text(encoding="utf-8"))
    baselines = load_baseline_records(args.baselines_dir)
    analysis = analyze_pilot_record(record, baselines)
    # Baseline degeneracy guards ride along for the guard section of the report.
    from humanbrain.qa_analysis import guard_verdict

    analysis["baseline_guards"] = [
        guard_verdict(baseline["metrics"]["degeneracy"], arm)
        for arm, baseline in sorted(baselines.items())
    ]

    markdown = render_findings_markdown(analysis)

    out_json = args.out_json or args.record.with_name("r2_pilot_analysis.json")
    out_md = args.out_md or args.record.with_name("FINDINGS-pilot-analysis.md")
    write_atomic(out_json, json.dumps(analysis, indent=1, sort_keys=True).encode("utf-8") + b"\n")
    write_atomic(out_md, markdown.encode("utf-8"))
    if args.stdout:
        print(markdown, end="")
    print(f"analysis -> {out_json}")
    print(f"findings -> {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
