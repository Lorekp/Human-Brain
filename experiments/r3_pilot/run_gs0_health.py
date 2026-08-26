"""G-S0 fixture wiring health check: A1/A2 arms at N=96 on a ~200-token TRAIN slice.

Dispatch (r3-segment-growth-preregistration.md G-S0 + r3-tokenizer-freeze.md §4,
re-dispatch after provider interruption): build the replay driver supplying
``W_{t-1}`` winner sets per token from the observation-sink spike stream of the
GLIF3 lane at the N=96 fixture population, using the frozen concept mapping
(``initial_pool = id mod 32``, ``slot = id // 32``), then run arms
A1 growth-on / A2 growth-off over a ~200-token TRAIN slice.

G-S0 pass here = wiring health only: clean run, predicted-path fraction
computable, burst telemetry present, bit-identical rerun, A2 structural-zero
tripwire, spike-rate pre-flight.  All gates (G-S1..S3) need the C1 cohort.

Slice construction mirrors experiments/concept_sdr_composition/run_composition.py:
frozen FREEZE-CANDIDATE-v3-BIBLE.json TRAIN split (corpus sha asserted), base
BytePhraseTokenizer(max_phrases=0), concepts mined freq>=50 / MI>=5bits /
cap 512 into band [1024,1536); verses in canon order until >= 200 concept-id
tokens accumulate, truncated to exactly N_TOKENS.  Deterministic prefix.

Outputs gs0_health_run.json beside this script and prints a summary table.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from humanbrain.concept_tokens import ConceptConfig, ConceptTokenizer, mine_concepts_detailed
from humanbrain.r3_pilot_glue import (
    CONSTRUCTION_SEED,
    gs0_health_checks,
    run_arm,
)
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

FREEZE_DIR = Path(os.environ.get("TEMP", os.environ.get("LOCALAPPDATA", ""))) / "hb_bible_freeze"
N_TOKENS = 200
LANE_CELLS = 96  # N=96 fixture leg (repo fixture-leg convention)
PASSES = 2

VERSE_RE = re.compile(r"^(\d+):(\d+)\s")


def load_bible_train() -> tuple[list[tuple[str, str]], dict]:
    """(book short_name, book text) for every TRAIN book, canonical order."""
    freeze = json.loads((FREEZE_DIR / "FREEZE-CANDIDATE-v3-BIBLE.json").read_text(encoding="utf-8"))
    spans = json.loads((FREEZE_DIR / "evidence" / "pg10_book_spans_v3.json").read_text(encoding="utf-8"))
    trimmed = (FREEZE_DIR / "evidence_pg10.trimmed.bin").read_bytes()

    digest = hashlib.sha256(trimmed).hexdigest()
    expected = freeze["corpus"]["sha256_trimmed"]
    if digest != expected:
        raise SystemExit(f"STOP: corpus drift sha256 {digest} != frozen {expected}")
    books = []
    for book in spans["books"]:
        if book["role"] != "train":
            continue
        start, end = book["span_offsets"]
        books.append((book["short_name"], trimmed[start:end].decode("utf-8")))
    provenance = {
        "candidate_id": freeze["candidate_id"],
        "trimmed_sha256": digest,
        "sha256_match": True,
        "train_books": len(books),
    }
    return books, provenance


def segment_verses(book_text: str) -> list[str]:
    paragraphs = re.split(r"\r?\n\s*\r?\n", book_text)
    return [p.strip() for p in paragraphs if VERSE_RE.match(p.strip())]


def build_slice(n_tokens: int) -> tuple[list[int], dict]:
    """First n_tokens concept ids over TRAIN verses in canon order."""
    t0 = time.time()
    books, prov = load_bible_train()
    train_texts = [t for _, t in books]
    base = BytePhraseTokenizer.fit(train_texts, TokenizerConfig(max_phrases=0, min_frequency=3))
    assert base.vocab_size == 256, f"base vocab {base.vocab_size} != 256"
    candidates = mine_concepts_detailed(train_texts, base, ConceptConfig())
    stack = ConceptTokenizer(base, tuple(c.payload for c in candidates))

    tokens: list[int] = []
    verses_used = 0
    for name, text in books:
        for verse in segment_verses(text):
            tokens.extend(stack.encode(verse))
            verses_used += 1
            if len(tokens) >= n_tokens:
                break
        if len(tokens) >= n_tokens:
            break
    tokens = tokens[:n_tokens]
    distinct = sorted(set(tokens))
    info = {
        **prov,
        "n_tokens": len(tokens),
        "n_verses_used": verses_used,
        "books_touched": verses_used and name or "",
        "distinct_ids": len(distinct),
        "concept_band_ids": sum(1 for t in distinct if t >= 1024),
        "max_token_id": max(distinct),
        "slice_sha256": hashlib.sha256(
            ",".join(str(t) for t in tokens).encode("utf-8")
        ).hexdigest(),
        "mining_seconds": round(time.time() - t0, 1),
        "tokenizer": "BytePhraseTokenizer(max_phrases=0,min_frequency=3) + ConceptTokenizer(freq>=50,MI>=5,cap512)",
    }
    return tokens, info


def arm_summary(run) -> dict:
    return {
        "arm": run.arm,
        "steps": len(run.rows),
        "f_pred": run.predicted_path_fraction,
        "passes": run.passes,
        "burst_fraction_by_pass": run.burst_fraction_by_pass(),
        "total_segments": run.total_segments,
        "total_contacts": run.total_contacts,
        "elections": run.elections,
        "allocated_segments": sum(1 for r in run.rows if r.allocated_segment),
        "punished_events": sum(len(r.punished_segments) for r in run.rows),
        "removed_contacts": sum(r.removed_contacts for r in run.rows),
        "permanences_sha256": run.permanences_sha256,
        "telemetry_sha256": run.telemetry_sha256,
    }


def main() -> int:
    print(f"[1/3] building {N_TOKENS}-token TRAIN slice (frozen corpus + concept stack)...", flush=True)
    tokens, slice_info = build_slice(N_TOKENS)
    print(
        f"      {slice_info['n_verses_used']} verses -> {len(tokens)} tokens, "
        f"{slice_info['distinct_ids']} distinct ids "
        f"({slice_info['concept_band_ids']} concept-band), max id {slice_info['max_token_id']}",
        flush=True,
    )
    print(f"      slice sha256 {slice_info['slice_sha256'][:16]}... corpus sha OK", flush=True)

    print(f"[2/3] replaying arms at N={LANE_CELLS}, seed={CONSTRUCTION_SEED}, {PASSES} passes...", flush=True)
    t0 = time.time()
    a1, recorder_a1, diag_a1 = run_arm(tokens, arm="A1", lane_cells=LANE_CELLS, passes=PASSES)
    a1b, _, _ = run_arm(tokens, arm="A1", lane_cells=LANE_CELLS, passes=PASSES)  # determinism rerun
    a2, recorder_a2, diag_a2 = run_arm(tokens, arm="A2", lane_cells=LANE_CELLS, passes=PASSES)
    elapsed = time.time() - t0
    print(f"      replay wall-clock {elapsed:.1f}s", flush=True)

    print("[3/3] G-S0 wiring-health battery...", flush=True)
    checks = gs0_health_checks(a1, a2, a1b, recorder_a1)

    manifest = {
        "experiment": "r3-gs0-fixture-wiring-health",
        "dispatch": "G-S0 fixture wiring re-dispatch; tokenizer freeze 966d6e6; P-B fd2c1c8",
        "prereg": "docs/research/r3-segment-growth-preregistration.md §6 G-S0",
        "freeze": "docs/research/r3-tokenizer-freeze.md §4",
        "slice": slice_info,
        "lane_cells": LANE_CELLS,
        "passes": PASSES,
        "construction_seed": CONSTRUCTION_SEED,
        "winner_set_source": "observation-sink spike stream (circuit._token_spike_counts > 0)",
        "arms": {
            "A1": {**arm_summary(a1), "diagnostics": diag_a1},
            "A1_rerun": arm_summary(a1b),
            "A2": {**arm_summary(a2), "diagnostics": diag_a2},
        },
        "health_checks": checks,
        "wall_clock_seconds": round(elapsed, 1),
    }
    out_path = Path(__file__).resolve().parent / "gs0_health_run.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print(f"A1 f_pred          : {a1.predicted_path_fraction:.4f} "
          f"(segments={a1.total_segments}, contacts={a1.total_contacts})")
    print(f"A1 pass f_pred     : {[round(p['predicted_fraction'], 4) for p in a1.passes]}")
    print(f"A1 burst frac/pass : {[round(p['burst_fraction'], 4) for p in a1.passes]}")
    print(f"A2 f_pred          : {a2.predicted_path_fraction:.4f} (structural zero tripwire)")
    print(f"bit-identical rerun: {checks['bit_identical_reruns']}")
    print(f"spike pre-flight   : {checks['spike_rate_preflight']} "
          f"(mean density {recorder_a1.mean_activity_density():.4f})")
    print(f"G-S0 wiring health : {'PASS' if checks['all_passed'] else 'FAIL'}")
    print(f"manifest           : {out_path}")
    return 0 if checks["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
