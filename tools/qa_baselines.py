#!/usr/bin/env python3
"""Bible-QA baseline harness: smoothed unigram/trigram + majority-floor comparators.

Preregistered comparators for R2 Bible-QA evaluation (docs/research/
cell-lm-corpus-bible.md §6 "Evaluation additions", red test 20). Emits metric
records FIELD-COMPARABLE to benchmarks/qa_experiments.py
(humanbrain/qa-experiments/v2; metric fields unchanged since v1):

- ``metrics.answer_span`` — teacher-forced prediction across the answer-word
  span of every VAL QA item, conditioned on the serialized prompt;
- ``metrics.free_run``   — greedy word-by-word completion with the same
  budget rule (len(reference)+slack), exact-match against the normalized
  reference answer;
- ``metrics.degeneracy`` — identical distinct-answers/modal-share guards
  (GUARD_DISTINCT_MIN=3, GUARD_MODAL_SHARE_MAX=0.9) so mode collapse is
  visible in exactly the same schema shape.

Arms:
- ``unigram_smoothed`` — word unigram MLE over the 49-book TRAIN stream,
  add-<unk> vocabulary;
- ``trigram_smoothed`` — Jelinek-Mercer interpolated trigram/bigram/unigram
  over the same stream (pinned lambdas 0.6/0.3/0.1);
- ``majority_floor``   — red-test-20 floor guard: always emits the modal TRAIN
  answer string; its degeneracy guard FAILS by construction (that is its job).

stdlib-only; CPU seconds; deterministic (byte-identical records modulo
wall-clock keys). Writes experiments/qa_baselines_bible_v1/<arm>.json and
never overwrites an existing record. artifacts/ is untouched; no frozen
protocol is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "humanbrain/qa-baselines/v1"
GUARD_MODAL_SHARE_MAX = 0.9
GUARD_DISTINCT_MIN = 3
WORD_RE = re.compile(r"[a-z0-9']+")
UNK = "<unk>"
JM_LAMBDA_TRIGRAM = 0.6
JM_LAMBDA_BIGRAM = 0.3


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%SZ')}] {msg}", flush=True)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


# ------------------------------------------------------------------ data load

def load_freeze(freeze_dir: Path) -> dict:
    ev = freeze_dir / "evidence"
    train_man = json.loads((ev / "qa_manifest_train.json").read_text(encoding="utf-8"))
    val_man = json.loads((ev / "qa_manifest_val.json").read_text(encoding="utf-8"))
    spans = json.loads((ev / "pg10_book_spans_v3.json").read_text(encoding="utf-8"))
    trimmed = (ev / "pg10.trimmed.bin").read_bytes()
    train_chunks = []
    for book in spans["books"]:
        if book["role"] == "train":
            lo, hi = book["span_offsets"]
            train_chunks.append(trimmed[lo:hi])
    train_stream_bytes = b"\n".join(train_chunks)
    return {
        "train_manifest": train_man,
        "val_manifest": val_man,
        "train_stream_bytes": train_stream_bytes,
        "train_stream_sha256": sha256_bytes(train_stream_bytes),
        "spans_sha256": sha256_bytes((ev / "pg10_book_spans_v3.json").read_bytes()),
        "train_manifest_sha256": train_man["sha256_canonical"],
        "val_manifest_sha256": val_man["sha256_canonical"],
        "freeze_candidate_sha256": sha256_bytes(
            (freeze_dir / "FREEZE-CANDIDATE-v3-BIBLE.json").read_bytes()),
    }


# ------------------------------------------------------------------- models

class UnigramModel:
    kind = "unigram_smoothed"

    def fit(self, stream_words: list[str]) -> None:
        self.counts = Counter(stream_words)
        self.total = sum(self.counts.values())
        best, best_c = None, -1
        for w, c in self.counts.items():
            if c > best_c or (c == best_c and (best is None or w < best)):
                best, best_c = w, c
        self._mode_word = best

    def next_dist(self, context: list[str]) -> Counter:
        return self.counts

    def argmax_next(self, context: list[str]) -> str:
        # most common word; ties broken deterministically by lexicographic order
        return self._mode_word

    def prob(self, word: str, context: list[str]) -> float:
        return self.counts.get(word, self.counts.get(UNK, 0)) / max(1, self.total)


class TrigramModel:
    """Jelinek-Mercer interpolation: 0.6*tri + 0.3*bi + 0.1*uni (pinned)."""

    kind = "trigram_smoothed"

    def __init__(self) -> None:
        self.uni = UnigramModel()
        self.bi: Counter = Counter()
        self.tri: Counter = Counter()
        self.bi_ctx: Counter = Counter()
        self.tri_ctx: Counter = Counter()

    def fit(self, stream_words: list[str]) -> None:
        self.uni.fit(stream_words)
        padded = ["<s>", "<s>"] + stream_words
        self._top_unigrams = [
            w for w, _ in sorted(self.uni.counts.items(), key=lambda kv: (-kv[1], kv[0]))[:500]
        ]
        self.bi.update(zip(padded, padded[1:]))
        self.tri.update(zip(padded, padded[1:], padded[2:]))
        self.bi_ctx.update(a for a, _ in zip(padded, padded[1:]))
        self.tri_ctx.update((a, b) for a, b, _ in zip(padded, padded[1:], padded[2:]))

    def prob(self, word: str, context: list[str]) -> float:
        uni = self.uni.prob(word, context)
        if len(context) >= 1:
            ctx = context[-1]
            bi_mle = self.bi.get((ctx, word), 0) / max(1, self.bi_ctx.get(ctx, 0))
        else:
            bi_mle = uni
        if len(context) >= 2:
            ctx2 = (context[-2], context[-1])
            tri_mle = self.tri.get((ctx2[0], ctx2[1], word), 0) / max(
                1, self.tri_ctx.get(ctx2, 0))
        else:
            tri_mle = bi_mle
        p = JM_LAMBDA_TRIGRAM * tri_mle + JM_LAMBDA_BIGRAM * bi_mle + (
            1.0 - JM_LAMBDA_TRIGRAM - JM_LAMBDA_BIGRAM) * uni
        return p

    def argmax_next(self, context: list[str]) -> str:
        cands = set(self._top_unigrams)
        if len(context) >= 1:
            ctx = context[-1]
            cands.update(w for (a, w) in self.bi if a == ctx)
        if len(context) >= 2:
            ctx2 = (context[-2], context[-1])
            cands.update(w for (a, b, w) in self.tri if (a, b) == ctx2)
        best, best_p = None, -1.0
        for w in sorted(cands):
            p = self.prob(w, context)
            if p > best_p:
                best, best_p = w, p
        return best if best is not None else UNK


class MajorityFloor:
    kind = "majority_floor"

    def __init__(self, train_answers: list[str]) -> None:
        normed = [normalize(a) for a in train_answers]
        self.mode, _mode_count = Counter(normed).most_common(1)[0]

    def complete(self, prompt_words: list[str], budget: int) -> str:
        out = self.mode.split(" ")
        return " ".join(out[:budget])


# ------------------------------------------------------------------ metrics

def answer_span_eval(model, items: list[dict]) -> dict:
    predictions = correct = 0
    for item in items:
        prompt_w = words(item["serialized_prompt_answer"].split("\nA:")[0])
        answer_w = words(item["answer"])
        context = list(prompt_w)
        for target in answer_w:
            pred = model.argmax_next(context)
            correct += int(pred == target)
            predictions += 1
            context.append(target)
    return {
        "tasks": len(items),
        "predictions": predictions,
        "correct": correct,
        "accuracy": correct / predictions if predictions else 0.0,
    }


def free_run_eval(model, items: list[dict], slack: int) -> tuple[dict, list[dict]]:
    rows = []
    for item in items:
        ref_words = words(item["answer"])
        budget = len(ref_words) + slack
        if isinstance(model, MajorityFloor):
            completion = model.complete([], budget)
        else:
            context = words(item["serialized_prompt_answer"].split("\nA:")[0])
            generated = []
            for _ in range(budget):
                nxt = model.argmax_next(context)
                generated.append(nxt)
                context.append(nxt)
            completion = " ".join(generated)
        rows.append({
            "qa_id": item["qa_id"],
            "family": item["family"],
            "expected": item["answer"],
            "completion": completion,
            "exact_match": normalize(completion) == normalize(item["answer"]),
        })
    matches = sum(int(r["exact_match"]) for r in rows)
    summary = {
        "tasks": len(rows),
        "exact_matches": matches,
        "exact_match_rate": matches / len(rows) if rows else 0.0,
    }
    return summary, rows


def degeneracy_metrics(completions: list[str]) -> dict:
    normalized = [normalize(c) for c in completions]
    counter = Counter(normalized)
    evaluated = len(normalized)
    modal_answer, modal_count = counter.most_common(1)[0] if counter else ("", 0)
    modal_share = modal_count / evaluated if evaluated else 0.0
    distinct_answers = len(counter)
    return {
        "evaluated": evaluated,
        "distinct_answers": distinct_answers,
        "modal_answer": modal_answer,
        "modal_share": modal_share,
        "guard": {
            "modal_share_max": GUARD_MODAL_SHARE_MAX,
            "distinct_answers_min": GUARD_DISTINCT_MIN,
        },
        "guard_pass": (
            distinct_answers >= GUARD_DISTINCT_MIN and modal_share <= GUARD_MODAL_SHARE_MAX
        ),
    }


# ---------------------------------------------------------------------- run

def run_arm(arm: str, freeze: dict, slack: int, seed_unused: int = 0) -> dict:
    started = time.perf_counter()
    stream_words = words(freeze["train_stream_bytes"].decode("utf-8"))
    log(f"{arm}: train stream {len(stream_words):,} words")
    if arm == "unigram_smoothed":
        model = UnigramModel()
        model.fit(stream_words)
        vocab_note = {"vocab_size": len(model.counts), "total_tokens": model.total}
    elif arm == "trigram_smoothed":
        model = TrigramModel()
        model.fit(stream_words)
        vocab_note = {
            "vocab_size": len(model.uni.counts),
            "bigram_types": len(model.bi),
            "trigram_types": len(model.tri),
            "jm_lambdas": [JM_LAMBDA_TRIGRAM, JM_LAMBDA_BIGRAM,
                           round(1 - JM_LAMBDA_TRIGRAM - JM_LAMBDA_BIGRAM, 4)],
            "argmax_candidate_set": (
                "trigram+bigram context continuations union top-500 unigram words "
                "(conservative restriction of full-vocab argmax)"
            ),
        }
    elif arm == "majority_floor":
        train_answers = [i["answer"] for i in freeze["train_manifest"]["items"]]
        model = MajorityFloor(train_answers)
        vocab_note = {"modal_answer": model.mode}
    else:
        raise SystemExit(f"unknown arm {arm}")

    val_items = freeze["val_manifest"]["items"]
    if arm == "majority_floor":
        span = {
            "tasks": len(val_items),
            "predictions": 0,
            "correct": 0,
            "accuracy": 0.0,
            "note": "teacher-forced span accuracy not applicable to the mode floor",
        }
    else:
        span = answer_span_eval(model, val_items)
    free, rows = free_run_eval(model, val_items, slack)
    degen = degeneracy_metrics([r["completion"] for r in rows])
    wall = time.perf_counter() - started

    record = {
        "schema_version": SCHEMA_VERSION,
        "protocol": (
            f"{arm} word-level comparator over the 49-book Bible-QA TRAIN stream; "
            "VAL manifest items scored with qa-experiments-v1-shaped "
            "answer_span/free_run/degeneracy metrics"
        ),
        "corpus_source": "bible-kjv-pg10:v3-freeze-candidate",
        "corpus_sha256": freeze["freeze_candidate_sha256"],
        "train_manifest_sha256": freeze["train_manifest_sha256"],
        "val_manifest_sha256": freeze["val_manifest_sha256"],
        "train_stream_sha256": freeze["train_stream_sha256"],
        "arm": arm,
        "seed": seed_unused,
        "completion_token_slack": slack,
        "model": {"kind": arm, **vocab_note},
        "metrics": {
            "answer_span": span,
            "free_run": free,
            "degeneracy": degen,
        },
        "tasks": rows,
        "wall_s": round(wall, 3),
    }
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=[
        "unigram_smoothed", "trigram_smoothed", "majority_floor"])
    ap.add_argument("--repeat", type=int, default=3,
                    help="timing repeats; median wall_s recorded (shared-box discipline)")
    local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
    ap.add_argument("--freeze-dir",
                    default=str(Path(local) / "Temp" / "hb_bible_freeze"))
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--out-dir", default=str(repo / "experiments" / "qa_baselines_bible_v1"))
    ap.add_argument("--force", action="store_true",
                    help="allow overwrite of an existing arm record (default fail-closed)")
    args = ap.parse_args()

    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                              capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        head = "unknown"

    freeze = load_freeze(Path(args.freeze_dir))
    log(f"loaded freeze candidate sha={freeze['freeze_candidate_sha256'][:16]}...")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        out_path = out_dir / f"{arm}.json"
        if out_path.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite existing record: {out_path}")
        walls, record = [], None
        for rep in range(max(1, args.repeat)):
            t0 = time.perf_counter()
            record = run_arm(arm, freeze, slack=8)
            walls.append(round(time.perf_counter() - t0, 3))
        record["wall_s"] = sorted(walls)[len(walls) // 2]
        record["model"]["timing_repeats_wall_s"] = walls
        record["git_head"] = head
        record["generated_utc"] = utcnow()
        tmp = out_path.with_suffix(".tmp")
        tmp.write_bytes(canonical_json(record))
        tmp.replace(out_path)
        m = record["metrics"]
        log(f"{arm}: span_acc={m['answer_span']['accuracy']:.4f} "
            f"em={m['free_run']['exact_matches']}/{m['free_run']['tasks']} "
            f"degeneracy_guard={'PASS' if m['degeneracy']['guard_pass'] else 'FAIL'} "
            f"median_wall={record['wall_s']}s -> {out_path}")


if __name__ == "__main__":
    main()
