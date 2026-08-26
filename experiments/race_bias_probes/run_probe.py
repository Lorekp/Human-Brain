#!/usr/bin/env python3
"""Probe P-B runner: predictive-state depolarizing bias into the first-spike race.

Preregistration: experiments/race_bias_probes/preregistration.json (frozen
before implementation). Two legs, stop-on-failure between them:

  --fixture   N=96 pyramidal-cell sheet. Infrastructure leg ONLY: pipeline-level
              flag-off bit-parity (incumbent constructor path vs flag-off probe
              path: race records AND grown branch-state hashes), double-run
              determinism, all five arms execute with finite telemetry,
              coverage report. Never scored for the verdict.

  --cohort    THE single decisive pass at N=1024: full TRAIN-stream growth,
              all VAL items (639 answer-word positions, denominator enforced)
              and all TRAIN items (TRAIN-self early-warning split). Arms:
              off / proportional / shuffled / decrement_plus (+ diagnostic
              decrement_minus). Mechanical verdict per the frozen gates.

Scoring is race-resolved only: the race winner IS the prediction; there is no
offline head anywhere. Span protocol mirrors tools/qa_baselines.answer_span_eval
verbatim (wordization, teacher-forced context, per-position match counting).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from humanbrain.cortex import CellRace, DendriticCortex  # noqa: E402
from humanbrain.race_bias import (  # noqa: E402
    BIAS_GAIN_PA_DEFAULT,
    CONTACT_CAP,
    DECREMENT_FRACTION_DEFAULT,
    GROWTH_INCREMENT,
    THETA_CONNECTED_DEFAULT,
    PredictiveBiasConfig,
    connected_contact_count,
    grow_with_segment_decrement,
)

PREREG_PATH = Path(__file__).resolve().parent / "preregistration.json"
FREEZE_DIR = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    / "Temp"
    / "hb_bible_freeze"
    / "evidence"
)
WORD_RE = re.compile(r"[a-z0-9']+")
UNK_TOKEN = "<unk>"

FIXTURE_CELLS = 96
COHORT_CELLS = 1024
FIXTURE_GROWTH_TOKENS = 60_000
FIXTURE_VAL_ITEMS = 12
FIXTURE_TRAIN_ITEMS = 12

# Frozen comparator anchors (experiments/qa_baselines_bible_v1, verbatim).
UNIGRAM_SPAN_ACC = 0.04538341158059468  # 29/639
MAJORITY_PRIOR_SPAN_ACC = 0.0406885758998435  # 26/639
TRIGRAM_INFORMATIONAL = 0.162754303599374  # 104/639
EXPECTED_VAL_POSITIONS = 639


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def load_preregistration() -> dict:
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    constants = prereg["constants"]
    expected = {
        "theta_connected": THETA_CONNECTED_DEFAULT,
        "contact_cap": CONTACT_CAP,
        "growth_increment": GROWTH_INCREMENT,
        "bias_gain_pa_per_matched_segment": BIAS_GAIN_PA_DEFAULT,
        "predicted_segment_decrement_fraction": DECREMENT_FRACTION_DEFAULT,
    }
    for key, value in expected.items():
        if float(constants[key]) != float(value):
            raise SystemExit(
                f"ABORT: preregistration constant {key}={constants[key]} does not "
                f"match implemented {value} — freeze drift before any spend"
            )
    return prereg


def load_freeze() -> dict:
    train_man = json.loads((FREEZE_DIR / "qa_manifest_train.json").read_text("utf-8"))
    val_man = json.loads((FREEZE_DIR / "qa_manifest_val.json").read_text("utf-8"))
    spans = json.loads((FREEZE_DIR / "pg10_book_spans_v3.json").read_text("utf-8"))
    trimmed = (FREEZE_DIR / "pg10.trimmed.bin").read_bytes()
    chunks = []
    for book in spans["books"]:
        if book["role"] == "train":
            lo, hi = book["span_offsets"]
            chunks.append(trimmed[lo:hi])
    train_stream_bytes = b"\n".join(chunks)
    digest = hashlib.sha256(train_stream_bytes).hexdigest()
    log(f"train stream: {len(train_stream_bytes):,} bytes sha256 {digest[:16]}…")
    return {
        "train_items": train_man["items"],
        "val_items": val_man["items"],
        "train_stream_words": words(train_stream_bytes.decode("utf-8")),
        "train_stream_sha256": digest,
    }


class Sheet:
    """Deterministic word-type -> pyramidal-cell assignment (prereg N semantics)."""

    def __init__(self, train_stream_words: list[str], cell_budget: int):
        counts = Counter(train_stream_words)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self.cell_budget = int(cell_budget)
        self.word_to_cell: dict[str, int] = {}
        for index, (word, _count) in enumerate(ranked[: cell_budget - 1]):
            self.word_to_cell[word] = index
        self.unk_cell = cell_budget - 1
        self.cell_to_word = {cell: w for w, cell in self.word_to_cell.items()}
        self.vocab_types = len(ranked)

    def encode(self, stream_words: list[str]) -> list[int]:
        unk = self.unk_cell
        table = self.word_to_cell
        return [table.get(word, unk) for word in stream_words]

    def decode(self, cell: int) -> str:
        return self.cell_to_word.get(int(cell), UNK_TOKEN)


# ------------------------------------------------------------------- scoring


def span_eval(
    cortex: DendriticCortex,
    sheet: Sheet,
    items: list[dict],
    *,
    expected_positions: int | None,
    label: str,
) -> dict:
    """Race-resolved teacher-forced span eval (answer_span_eval semantics)."""
    predictions = 0
    correct = 0
    spike_times: list[float] = []
    winners_counter: Counter[str] = Counter()
    unreachable_targets = 0
    started = time.perf_counter()
    for item in items:
        prompt_w = words(item["serialized_prompt_answer"].split("\nA:")[0])
        answer_w = words(item["answer"])
        context = sheet.encode(prompt_w)
        for target in answer_w:
            race = cortex.race(context)
            pred_word = sheet.decode(race.winner)
            correct += int(pred_word == target)
            predictions += 1
            spike_times.append(float(race.spike_time_ms))
            winners_counter[pred_word] += 1
            if pred_word == UNK_TOKEN:
                unreachable_targets += 1
            context.append(sheet.encode([target])[0])
    elapsed = time.perf_counter() - started
    if expected_positions is not None and predictions != expected_positions:
        raise SystemExit(
            f"ABORT: {label} produced {predictions} answer-word positions, "
            f"expected {expected_positions} — denominator mismatch"
        )
    spikes = np.asarray(spike_times, dtype=np.float64)
    return {
        "label": label,
        "tasks": len(items),
        "predictions": predictions,
        "correct": int(correct),
        "span_acc": float(correct / predictions) if predictions else 0.0,
        "distinct_predicted_answers": len(winners_counter),
        "modal_share": (
            max(winners_counter.values()) / predictions if predictions else 0.0
        ),
        "unk_predictions": int(unreachable_targets),
        "mean_winning_spike_ms": float(spikes.mean()) if spikes.size else 0.0,
        "median_winning_spike_ms": (
            float(np.median(spikes)) if spikes.size else 0.0
        ),
        "eval_seconds": round(elapsed, 3),
    }


def mcnemar_paired(arm_flags: list[int], control_flags: list[int]) -> dict:
    """Paired normal-approximation McNemar on position-correct flags."""
    both_right = sum(1 for a, b in zip(arm_flags, control_flags) if a and b)
    arm_only = sum(1 for a, b in zip(arm_flags, control_flags) if a and not b)
    control_only = sum(1 for a, b in zip(arm_flags, control_flags) if b and not a)
    discordant = arm_only + control_only
    if discordant == 0:
        return {
            "arm_only_correct": arm_only,
            "control_only_correct": control_only,
            "discordant": 0,
            "z": 0.0,
        }
    z = (arm_only - control_only) / (discordant ** 0.5)
    return {
        "arm_only_correct": arm_only,
        "control_only_correct": control_only,
        "discordant": discordant,
        "z": float(z),
    }


def position_flags(
    cortex: DendriticCortex, sheet: Sheet, items: list[dict]
) -> list[int]:
    """Per-position correctness flags aligned across arms (same eval order)."""
    flags: list[int] = []
    for item in items:
        prompt_w = words(item["serialized_prompt_answer"].split("\nA:")[0])
        answer_w = words(item["answer"])
        context = sheet.encode(prompt_w)
        for target in answer_w:
            race = cortex.race(context)
            flags.append(int(sheet.decode(race.winner) == target))
            context.append(sheet.encode([target])[0])
    return flags


# ------------------------------------------------------------- construction


def build_arm_cortex(
    sheet: Sheet,
    train_stream_ids: list[int],
    mode: str,
    n_cells: int,
    *,
    seed: int = 7,
    decrement_fraction: float | None = None,
    sign: int = 1,
) -> tuple[DendriticCortex, dict]:
    """Grow one arm's cortex from scratch; returns (cortex, growth_telemetry)."""
    if mode == "off":
        cortex = DendriticCortex(cell_count=n_cells, max_order=8, seed=seed)
        cortex.grow_sequence(train_stream_ids)
        return cortex, {"mode": "off"}
    if mode in ("proportional", "shuffled"):
        cortex = DendriticCortex(
            cell_count=n_cells,
            max_order=8,
            seed=seed,
            predictive_bias=PredictiveBiasConfig(mode=mode),
        )
        cortex.grow_sequence(train_stream_ids)
        return cortex, {"mode": mode}
    if mode == "decrement_minus" or mode == "decrement_plus":
        cortex = DendriticCortex(
            cell_count=n_cells,
            max_order=8,
            seed=seed,
            predictive_bias=PredictiveBiasConfig(mode="proportional"),
        )
        events = grow_with_segment_decrement(
            cortex,
            train_stream_ids,
            decrement_fraction=(
                DECREMENT_FRACTION_DEFAULT if decrement_fraction is None else decrement_fraction
            ),
            sign=(-1 if mode == "decrement_minus" else 1),
        )
        return cortex, {"mode": mode, "events": events}
    raise ValueError(f"unknown arm mode {mode!r}")


ARMS = ("off", "proportional", "shuffled", "decrement_plus", "decrement_minus")


def run_arms(n_cells: int, items_val: list[dict], items_train: list[dict],
             train_stream_ids: list[int], sheet: Sheet,
             *, expected_val_positions: int | None) -> dict:
    results: dict[str, dict] = {}
    for mode in ARMS:
        log(f"arm {mode}: growing cortex …")
        started = time.perf_counter()
        cortex, growth = build_arm_cortex(
            sheet, train_stream_ids, mode, n_cells
        )
        growth_seconds = round(time.perf_counter() - started, 1)
        log(f"arm {mode}: growth done in {growth_seconds}s; evaluating …")
        val_row = span_eval(
            cortex, sheet, items_val,
            expected_positions=expected_val_positions, label="VAL",
        )
        train_row = span_eval(
            cortex, sheet, items_train, expected_positions=None, label="TRAIN-self",
        )
        flags = position_flags(cortex, sheet, items_val)
        results[mode] = {
            "growth": growth,
            "growth_seconds": growth_seconds,
            "branch_count": cortex.branch_count,
            "VAL": val_row,
            "TRAIN_self": train_row,
            "_val_position_flags": flags,
        }
        log(
            f"arm {mode}: VAL span_acc={val_row['span_acc']:.6f} "
            f"({val_row['correct']}/{val_row['predictions']})  "
            f"TRAIN-self={train_row['span_acc']:.6f} "
            f"({train_row['correct']}/{train_row['predictions']})  "
            f"spike_med={val_row['median_winning_spike_ms']:.3f}ms"
        )
    return results


# --------------------------------------------------------------------- legs


def parity_gate(n_cells: int, stream_ids: list[int], probe_contexts: list[list[int]]) -> dict:
    """Incumbent constructor path vs flag-off probe path must be byte-identical."""
    incumbent = DendriticCortex(cell_count=n_cells, max_order=8, seed=7)
    flagged = DendriticCortex(
        cell_count=n_cells,
        max_order=8,
        seed=7,
        predictive_bias=PredictiveBiasConfig(mode="off"),
    )

    def state_hash(cortex: DendriticCortex) -> str:
        rows = [
            [json.dumps(list(ctx)), int(tgt), repr(float(strength))]
            for ctx, tgt, strength in cortex.iter_branches()
        ]
        rows.sort()
        return hashlib.sha256(canonical_json(rows)).hexdigest()

    for cortex in (incumbent, flagged):
        cortex.grow_sequence(stream_ids)
    records_incumbent, records_flagged = [], []
    for context in probe_contexts:
        for sink, cortex in ((records_incumbent, incumbent), (records_flagged, flagged)):
            race = cortex.race(context)
            sink.append(
                [
                    int(race.winner),
                    repr(float(race.spike_time_ms)),
                    repr(float(race.peak_voltage_mv)),
                    int(race.active_cells),
                    int(race.simulated_steps),
                ]
            )
    identical_records = records_incumbent == records_flagged
    identical_state = state_hash(incumbent) == state_hash(flagged)
    return {
        "n_cells": n_cells,
        "probe_contexts": len(probe_contexts),
        "race_records_identical": bool(identical_records),
        "state_hash_identical": bool(identical_state),
        "incumbent_state_sha256": state_hash(incumbent),
        "pass": bool(identical_records and identical_state),
    }


def run_fixture(prereg: dict) -> dict:
    log(f"FIXTURE LEG N={FIXTURE_CELLS} (infrastructure only)")
    freeze = load_freeze()
    stream_words = freeze["train_stream_words"][:FIXTURE_GROWTH_TOKENS]
    sheet = Sheet(freeze["train_stream_words"], FIXTURE_CELLS)
    stream_ids = sheet.encode(stream_words)
    val_items = freeze["val_items"][:FIXTURE_VAL_ITEMS]
    train_items = freeze["train_items"][:FIXTURE_TRAIN_ITEMS]

    probe_contexts = [
        stream_ids[:k] for k in range(1, min(len(stream_ids), 400) + 1, 17)
    ]
    parity = parity_gate(FIXTURE_CELLS, stream_ids, probe_contexts)
    if not parity["pass"]:
        return {"leg": "fixture_n96", "parity": parity, "pass": False}
    log("parity gate PASS (flag-off byte-identical to incumbent race)")

    # Determinism: identical proportional-arm construction twice.
    cortex_a, _ = build_arm_cortex(sheet, stream_ids, "proportional", FIXTURE_CELLS)
    cortex_b, _ = build_arm_cortex(sheet, stream_ids, "proportional", FIXTURE_CELLS)
    det_contexts = [stream_ids[:k] for k in range(1, 300, 11)]
    det_a = [[int(cortex_a.race(c).winner)] for c in det_contexts]
    det_b = [[int(cortex_b.race(c).winner)] for c in det_contexts]
    deterministic = det_a == det_b
    log(f"determinism double-run: {'PASS' if deterministic else 'FAIL'}")

    results = run_arms(FIXTURE_CELLS, val_items, train_items, stream_ids, sheet,
                       expected_val_positions=None)
    finite = all(
        np.isfinite(row["VAL"]["mean_winning_spike_ms"])
        and row["VAL"]["predictions"] > 0
        for row in results.values()
    )
    coverage = {
        "sheet_cells": FIXTURE_CELLS,
        "word_types_in_train_stream": sheet.vocab_types,
        "cells_with_words": len(sheet.word_to_cell),
        "val_target_positions": sum(
            len(words(i["answer"])) for i in val_items
        ),
    }
    passed = bool(deterministic and finite)
    output = {
        "leg": "fixture_n96",
        "prereg_sha256": hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest(),
        "train_stream_sha256": freeze["train_stream_sha256"],
        "parity": parity,
        "deterministic": bool(deterministic),
        "arms_finite": bool(finite),
        "coverage": coverage,
        "arms": {m: {k: v for k, v in row.items() if not k.startswith("_")}
                 for m, row in results.items()},
        "pass": passed,
    }
    log(f"FIXTURE VERDICT: {'PASS' if passed else 'FAIL'}")
    return output


def run_cohort(prereg: dict) -> dict:
    log(f"COHORT LEG N={COHORT_CELLS} (the single decisive pass)")
    freeze = load_freeze()
    sheet = Sheet(freeze["train_stream_words"], COHORT_CELLS)
    train_stream_ids = sheet.encode(freeze["train_stream_words"])
    val_items = freeze["val_items"]
    train_items = freeze["train_items"]

    val_targets = [t for item in val_items for t in words(item["answer"])]
    reachable = sum(1 for t in val_targets if t in sheet.word_to_cell)
    prompt_tokens = [
        w
        for item in val_items
        for w in words(item["serialized_prompt_answer"].split("\nA:")[0])
    ]
    prompt_reachable = sum(1 for w in prompt_tokens if w in sheet.word_to_cell)
    coverage = {
        "sheet_cells": COHORT_CELLS,
        "word_types_in_train_stream": sheet.vocab_types,
        "cells_with_words": len(sheet.word_to_cell),
        "val_answer_positions": len(val_targets),
        "val_answer_positions_with_dedicated_cell": reachable,
        "val_target_coverage": reachable / len(val_targets),
        "val_prompt_tokens": len(prompt_tokens),
        "val_prompt_tokens_not_unk": prompt_reachable,
        "val_prompt_coverage": prompt_reachable / len(prompt_tokens),
    }
    log(json.dumps(coverage))

    results = run_arms(COHORT_CELLS, val_items, train_items, train_stream_ids, sheet,
                       expected_val_positions=EXPECTED_VAL_POSITIONS)

    primary = ["proportional", "decrement_plus"]
    best_mode = max(primary, key=lambda m: results[m]["VAL"]["span_acc"])
    best = results[best_mode]["VAL"]
    control = results["shuffled"]["VAL"]
    mcnemar = mcnemar_paired(
        results[best_mode]["_val_position_flags"],
        results["shuffled"]["_val_position_flags"],
    )
    above_control = best["span_acc"] > control["span_acc"]
    above_unigram = best["span_acc"] > UNIGRAM_SPAN_ACC
    above_majority = best["span_acc"] > MAJORITY_PRIOR_SPAN_ACC
    train_leak_ratio = (
        best["span_acc"] / results[best_mode]["TRAIN_self"]["span_acc"]
        if results[best_mode]["TRAIN_self"]["span_acc"] > 0
        else None
    )
    verdict_pass = bool(above_control and above_unigram and above_majority)
    verdict = {
        "best_primary_arm": best_mode,
        "gates": {
            "above_shuffled_control": bool(above_control),
            "above_unigram_4.538": bool(above_unigram),
            "above_majority_prior_4.069": bool(above_majority),
        },
        "mcnemar_vs_shuffled": mcnemar,
        "comparators": {
            "unigram": UNIGRAM_SPAN_ACC,
            "majority_prior": MAJORITY_PRIOR_SPAN_ACC,
            "trigram_informational": TRIGRAM_INFORMATIONAL,
        },
        "train_self_to_val_ratio": train_leak_ratio,
        "pass": verdict_pass,
    }
    output = {
        "leg": "cohort_n1024",
        "prereg_sha256": hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest(),
        "train_stream_sha256": freeze["train_stream_sha256"],
        "coverage": coverage,
        "arms": {
            m: {k: v for k, v in row.items() if not k.startswith("_")}
            for m, row in results.items()
        },
        "verdict": verdict,
    }
    log(
        f"COHORT VERDICT: {'PASS' if verdict_pass else 'FAIL'} "
        f"(best primary {best_mode}: VAL {best['span_acc']:.6f} vs shuffled "
        f"{control['span_acc']:.6f}, unigram {UNIGRAM_SPAN_ACC:.6f})"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fixture", action="store_true", help="N=96 infrastructure leg")
    group.add_argument("--cohort", action="store_true", help="N=1024 decisive pass")
    args = parser.parse_args()

    prereg = load_preregistration()
    started = time.perf_counter()
    output = run_fixture(prereg) if args.fixture else run_cohort(prereg)
    output["wall_seconds"] = round(time.perf_counter() - started, 1)
    output["environment"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }

    suffix = "fixture" if args.fixture else "cohort"
    out_path = Path(__file__).resolve().parent / f"{suffix}_results.json"
    out_path.write_bytes(canonical_json(output) + b"\n")
    log(f"wrote {out_path}")
    if not output.get("pass", False) and args.fixture:
        raise SystemExit(2)  # stop-on-failure: abort before any cohort spend
    if args.cohort and not output["verdict"]["pass"]:
        raise SystemExit(1)  # honest terminal negative — cohort NOT retried


if __name__ == "__main__":
    main()
