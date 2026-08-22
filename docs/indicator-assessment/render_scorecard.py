#!/usr/bin/env python3
"""Render a filled indicator scorecard JSON as a fixed-width text table.

Usage:
    python render_scorecard.py SCORECARD.json [--delta BASE_SCORECARD.json]

Input format: {"<INDICATOR-ID>": {"level": "...", "one_line_evidence": "..."}}
Top-level keys beginning with "_" are treated as metadata and skipped.

Levels must be exactly one of ABSENT / PARTIAL / PRESENT / UNCERTAIN.
Invalid levels abort with exit code 2. Unknown indicator ids render last
with a warning. With --delta, two extra columns show the base level and a
categorical delta (raised/lowered/unchanged/new/removed/incomparable).
This tool renders bookkeeping tables only; it computes no scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VALID_LEVELS = ("ABSENT", "PARTIAL", "PRESENT", "UNCERTAIN")
INDICATOR_ORDER = (
    "RPT-1", "RPT-2",
    "GWT-1", "GWT-2", "GWT-3", "GWT-4",
    "HOT-1", "HOT-2", "HOT-3", "HOT-4",
    "AST-1",
    "PP-1",
    "AE-1", "AE-2",
)
_LEVEL_RANK = {"ABSENT": 0, "PARTIAL": 1, "PRESENT": 2}
EVIDENCE_WIDTH = 56


def load_scorecard(path: str) -> dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"error: cannot read scorecard {path}: {exc}")
    if not isinstance(data, dict):
        sys.exit(f"error: {path} must be a JSON object")
    cleaned: dict[str, dict] = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if not isinstance(value, dict) or "level" not in value:
            sys.exit(f"error: {path}[{key!r}] needs at least a 'level' field")
        level = value.get("level")
        if level not in VALID_LEVELS:
            sys.exit(
                f"error: {path}[{key!r}] level {level!r} not in "
                f"{list(VALID_LEVELS)}"
            )
        evidence = value.get("one_line_evidence", "")
        if not isinstance(evidence, str):
            sys.exit(f"error: {path}[{key!r}] 'one_line_evidence' must be text")
        cleaned[key] = {"level": level, "one_line_evidence": evidence}
    return cleaned


def ordered_ids(cards: dict[str, dict]) -> list[str]:
    known = [i for i in INDICATOR_ORDER if i in cards]
    unknown = sorted(i for i in cards if i not in INDICATOR_ORDER)
    return known + unknown


def delta_text(base_level: str | None, head_level: str | None) -> str:
    if head_level is None:
        return "removed"
    if base_level is None:
        return "new"
    if base_level == "UNCERTAIN" or head_level == "UNCERTAIN":
        return "incomparable"
    rank_change = _LEVEL_RANK[head_level] - _LEVEL_RANK[base_level]
    if rank_change > 0:
        return "raised"
    if rank_change < 0:
        return "lowered"
    return "unchanged"


def clip(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3].rstrip() + "..."


def render(head: dict[str, dict], base: dict[str, dict] | None) -> str:
    if base is None:
        header = f"{'INDICATOR':<8}  {'LEVEL':<9}  EVIDENCE"
        rows = [
            f"{i:<8}  {head[i]['level']:<9}  "
            f"{clip(head[i]['one_line_evidence'], EVIDENCE_WIDTH)}"
            for i in ordered_ids(head)
        ]
    else:
        header = (
            f"{'INDICATOR':<8}  {'BASE':<9}  {'HEAD':<9}  "
            f"{'DELTA':<12}  EVIDENCE"
        )
        rows = []
        for ind in ordered_ids(head):
            b_level = base.get(ind, {}).get("level")
            h_level = head[ind]["level"]
            rows.append(
                f"{ind:<8}  {(b_level or '-'):<9}  {h_level:<9}  "
                f"{delta_text(b_level, h_level):<12}  "
                f"{clip(head[ind]['one_line_evidence'], EVIDENCE_WIDTH)}"
            )
        for ind in sorted(i for i in base if i not in head):
            rows.append(
                f"{ind:<8}  {base[ind]['level']:<9}  {'-':<9}  "
                f"{'removed':<12}  (absent from head scorecard)"
            )
    counts = {lvl: 0 for lvl in VALID_LEVELS}
    for entry in head.values():
        counts[entry["level"]] += 1
    tally = ", ".join(f"{lvl}={n}" for lvl, n in counts.items())
    footer = (
        f"Levels: {tally}. Counts are bookkeeping, not scores; no numeric "
        "or aggregate consciousness measure exists."
    )
    rule = "-" * max(len(header), *(len(r) for r in rows)) if rows else ""
    return "\n".join([header, rule, *rows, rule, footer])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scorecard", help="filled scorecard JSON (head)")
    parser.add_argument(
        "--delta",
        metavar="BASE_JSON",
        help="baseline scorecard JSON; adds BASE/HEAD/DELTA columns",
    )
    args = parser.parse_args(argv)
    head = load_scorecard(args.scorecard)
    base = load_scorecard(args.delta) if args.delta else None
    for ind in head:
        if ind not in INDICATOR_ORDER:
            print(f"warning: unknown indicator id {ind!r}", file=sys.stderr)
    print(render(head, base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
