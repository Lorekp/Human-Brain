"""QA-v1 experiment gate: verse-context question answering on the cell LM.

Grows one ``RecurrentCellLanguageModel`` on TRAIN ``qa_prompt + ' ' + answer``
texts, then evaluates held-out CALIBRATION tasks three ways:

- teacher-forced accuracy counted ONLY across the answer token span;
- free-run ``complete()`` exact match against the reference answer;
- first-class degeneracy metrics (distinct-answer count, modal-answer share
  with explicit guard thresholds) so mode collapse is visible in the schema.

The typed-hole seam is untouched: everything flows through the existing
language interface, whose vocabulary/output-cell alignment invariant is
re-checked defensively here (see ``humanbrain.language`` and
``humanbrain.recurrent_language`` constructors).

Records are byte-deterministic apart from ``wall_s``: ``--check-determinism``
runs twice and compares canonical JSON with wall-clock keys stripped.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
import json
from time import perf_counter

from humanbrain.qa_corpus import (
    CORPUS_SOURCE,
    QATask,
    build_curriculum,
    corpus_digest,
    growth_text,
    prompt_for_task,
)
from humanbrain.recurrent import RecurrentCircuitConfig
from humanbrain.recurrent_language import (
    RecurrentCellLanguageModel,
    RecurrentModelConfig,
)

SCHEMA_VERSION = "humanbrain/qa-experiments/v1"
GUARD_MODAL_SHARE_MAX = 0.9
GUARD_DISTINCT_MIN = 3


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 17
    max_train_tasks: int | None = None
    growth_epochs: int = 2
    max_phrases: int = 512
    min_phrase_frequency: int = 2
    circuit: RecurrentCircuitConfig = field(
        default_factory=lambda: RecurrentCircuitConfig(reservoir_cells=96)
    )
    eval_limit: int | None = None
    completion_token_slack: int = 8


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _strip_wall(value: object) -> object:
    """Recursively drop wall-clock keys so records compare byte-identically."""
    if isinstance(value, dict):
        return {
            key: _strip_wall(item)
            for key, item in value.items()
            if key not in ("wall_s", "wall_seconds")
        }
    if isinstance(value, list):
        return [_strip_wall(item) for item in value]
    return value


def _span_evaluation(model_like: object, prompt: str, answer: str) -> dict[str, object]:
    """Teacher-force ``prompt + ' ' + answer``, counting only the answer span."""
    tokenizer = model_like._tokenizer  # noqa: SLF001 - same-repo benchmark seam
    circuit = model_like._circuit  # noqa: SLF001
    full_ids = tokenizer.encode(prompt + " " + answer)
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(" " + answer)
    span_start = len(full_ids) - len(answer_ids)
    if tuple(full_ids[: len(prompt_ids)]) != tuple(prompt_ids):
        raise ValueError("prompt tokens must be a prefix of the evaluation encoding")
    if tuple(full_ids[span_start:]) != tuple(answer_ids):
        raise ValueError("answer tokens must be a suffix of the evaluation encoding")
    circuit.reset_sequence()
    circuit.observe(full_ids[0])
    correct = 0
    predictions = 0
    races = 0
    for target_index in range(1, len(full_ids)):
        race = circuit.race()
        races += 1
        if target_index >= span_start:
            predictions += 1
            correct += int(race.winner == full_ids[target_index])
        circuit.observe(full_ids[target_index])
    return {
        "predictions": predictions,
        "correct": correct,
        "accuracy": correct / predictions if predictions else 0.0,
        "races": races,
    }


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def _free_run(model: RecurrentCellLanguageModel, task: QATask, *, slack: int) -> dict[str, object]:
    tokenizer = model._tokenizer  # noqa: SLF001
    budget = len(tokenizer.encode(" " + task.answer)) + slack
    completion = model.complete(prompt_for_task(task), max_new_tokens=budget)
    return {
        "completion": completion,
        "normalized_completion": _normalize(completion),
        "expected": task.answer,
        "exact_match": _normalize(completion) == _normalize(task.answer),
    }


def _degeneracy_metrics(completions: list[str]) -> dict[str, object]:
    """Mode-collapse visibility: distinct answers plus modal-answer share."""
    normalized = [_normalize(text) for text in completions]
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
            distinct_answers >= GUARD_DISTINCT_MIN
            and modal_share <= GUARD_MODAL_SHARE_MAX
        ),
    }


def run(config: ExperimentConfig | None = None) -> dict[str, object]:
    """Grow on TRAIN prompts, evaluate CALIBRATION, return a schema record."""
    cfg = config or ExperimentConfig()
    curriculum = build_curriculum(cfg.seed)
    train_tasks: tuple[QATask, ...] = curriculum["train_tasks"]
    eval_tasks: tuple[QATask, ...] = curriculum["calibration_tasks"]
    if cfg.max_train_tasks is not None:
        train_tasks = train_tasks[: cfg.max_train_tasks]
    if cfg.eval_limit is not None:
        eval_tasks = eval_tasks[: cfg.eval_limit]

    growth_texts = [growth_text(task) for task in train_tasks]
    started = perf_counter()
    model = RecurrentCellLanguageModel.grow(
        growth_texts,
        RecurrentModelConfig(
            growth_epochs=cfg.growth_epochs,
            max_phrases=cfg.max_phrases,
            min_phrase_frequency=cfg.min_phrase_frequency,
            circuit=cfg.circuit,
        ),
    )
    vocabulary_alignment_pass = model.vocabulary_size == model._circuit.token_cells  # noqa: SLF001
    if not vocabulary_alignment_pass:
        raise RuntimeError(
            "vocabulary and output-cell population diverged during growth"
        )

    span_rows = [
        _span_evaluation(model, prompt_for_task(task), task.answer)
        for task in eval_tasks
    ]
    free_rows = [_free_run(model, task, slack=cfg.completion_token_slack) for task in eval_tasks]
    wall_s = perf_counter() - started

    span_predictions = sum(int(row["predictions"]) for row in span_rows)
    span_correct = sum(int(row["correct"]) for row in span_rows)
    free_matches = sum(int(row["exact_match"]) for row in free_rows)
    degeneracy = _degeneracy_metrics([str(row["normalized_completion"]) for row in free_rows])

    task_details = [
        {
            "qa_id": task.qa_id,
            "cohort": task.cohort,
            "ordinal": task.ordinal,
            "references": list(task.references),
            "question": task.question,
            "expected": row["expected"],
            "completion": row["completion"],
            "exact_match": row["exact_match"],
            "span_predictions": span_row["predictions"],
            "span_correct": span_row["correct"],
        }
        for task, span_row, row in zip(eval_tasks, span_rows, free_rows)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": (
            "grow RecurrentCellLanguageModel on TRAIN qa_prompt+' '+answer texts; "
            "held-out CALIBRATION teacher-forced answer-span accuracy, free-run "
            "complete() exact match, and first-class degeneracy metrics"
        ),
        "corpus_source": CORPUS_SOURCE,
        "corpus_sha256": corpus_digest(),
        "curriculum_sha256": curriculum["sha256"],
        "seed": cfg.seed,
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "model": {
            "vocabulary_size": model.vocabulary_size,
            "token_cells": model._circuit.token_cells,  # noqa: SLF001
            "vocabulary_alignment_pass": vocabulary_alignment_pass,
            "learned_synapses": model.learned_synapses,
            "reservoir_cells": model.reservoir_cells,
            "active_reservoir_cells": model.active_reservoir_cells,
        },
        "training_digest": model.training_digest,
        "metrics": {
            "answer_span": {
                "tasks": len(span_rows),
                "predictions": span_predictions,
                "correct": span_correct,
                "accuracy": span_correct / span_predictions if span_predictions else 0.0,
            },
            "free_run": {
                "tasks": len(free_rows),
                "exact_matches": free_matches,
                "exact_match_rate": free_matches / len(free_rows) if free_rows else 0.0,
            },
            "degeneracy": degeneracy,
        },
        "tasks": task_details,
        "wall_s": wall_s,
    }


def check_determinism(config: ExperimentConfig | None = None) -> dict[str, object]:
    """Run twice; records must be byte-identical once wall-clock keys are stripped."""
    cfg = config or ExperimentConfig()
    record_a = run(cfg)
    record_b = run(cfg)
    digest_a = sha256(_canonical_json(_strip_wall(record_a))).hexdigest()
    digest_b = sha256(_canonical_json(_strip_wall(record_b))).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": cfg.seed,
        "determinism_pass": digest_a == digest_b,
        "record_sha256_a": digest_a,
        "record_sha256_b": digest_b,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--max-train-tasks", type=int, default=None)
    parser.add_argument("--growth-epochs", type=int, default=ExperimentConfig.growth_epochs)
    parser.add_argument("--max-phrases", type=int, default=ExperimentConfig.max_phrases)
    parser.add_argument(
        "--min-phrase-frequency", type=int, default=ExperimentConfig.min_phrase_frequency
    )
    parser.add_argument("--reservoir-cells", type=int, default=96)
    parser.add_argument("--input-fanout", type=int, default=24)
    parser.add_argument("--recurrent-fanout", type=int, default=8)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--check-determinism", action="store_true")
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="exit non-zero when the degeneracy guard fails",
    )
    args = parser.parse_args()
    config = ExperimentConfig(
        seed=args.seed,
        max_train_tasks=args.max_train_tasks,
        growth_epochs=args.growth_epochs,
        max_phrases=args.max_phrases,
        min_phrase_frequency=args.min_phrase_frequency,
        circuit=RecurrentCircuitConfig(
            reservoir_cells=args.reservoir_cells,
            input_fanout=args.input_fanout,
            recurrent_fanout=args.recurrent_fanout,
        ),
        eval_limit=args.eval_limit,
    )
    if args.check_determinism:
        result: dict[str, object] = check_determinism(config)
    else:
        result = run(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_pass and not result["metrics"]["degeneracy"]["guard_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
