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

Execution backend seam (mirrors ladder_experiments.py): ``--device cuda``
binds ``HUMANBRAIN_DEVICE=cuda`` before any circuit exists and routes the
grow/read path through the CUDA full-touch kernel; ``--cells N`` sizes the
reservoir (historical default 96). At large N the dense-ridge normal matrix
cannot exist, so ``--readout-rule auto`` upgrades to ``sparse_three_factor``
once its projection breaches a 2 GiB host budget — the same rule/active-cell
contract lm_million_smoke.py drives at N=1e6.

Growth-tokenizer seam (``--tokenizer {byte,bpe4096,concept}``): ``byte``
keeps the historical in-grow BytePhraseTokenizer fit; ``bpe4096`` constructs
the plain BPE baseline (humanbrain.bpe_tokenizer); ``concept`` mines
recurring multi-token concepts from the TRAIN growth texts per
``humanbrain.concept_tokens`` and wraps them above the same base fit in a
reserved band ([1024, 1024+512)) — concepts spanning the ``answer:`` +
``' '`` prompt/answer junction are dropped so the answer-span prefix/suffix
seam below stays intact. Every arm flows through one language interface whose
vocabulary/output-cell alignment invariant is re-checked defensively here
(see ``humanbrain.language`` and ``humanbrain.recurrent_language``
constructors). Concept-arm mining gates default to the a013 compression-
study protocol (freq >= 50, MI >= 5 bits); lower them for fixture-scale
corpora via ``--concept-min-frequency`` / ``--concept-min-mi-bits``.

Records are byte-deterministic apart from ``wall_s``: ``--check-determinism``
runs twice and compares canonical JSON with wall-clock keys stripped.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from dataclasses import replace
from time import perf_counter

from humanbrain.bpe_tokenizer import BpeTokenizer
from humanbrain.concept_tokens import (
    CONCEPT_ID_BASE,
    ConceptConfig,
    ConceptTokenizer,
    mine_concepts_detailed,
)
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
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

SCHEMA_VERSION = "humanbrain/qa-experiments/v2"
GUARD_MODAL_SHARE_MAX = 0.9
GUARD_DISTINCT_MIN = 3

# Mirror of ladder_experiments' single-process host budget: the dense ridge
# readout allocates an F x F float64 normal matrix (F = active cells + 1), so
# ``auto`` switches to sparse_three_factor once that projection breaches it.
_MAX_RIDGE_HOST_BYTES = 2 * 1024**3
_DEVICES = ("cpu", "cuda")
_READOUT_RULES = ("ridge", "three_factor", "sparse_three_factor")
# Growth-tokenizer arms (see _build_growth_tokenizer): "byte" keeps the
# historical in-grow BytePhraseTokenizer fit, "bpe4096" constructs the plain
# BPE baseline promoted to humanbrain.bpe_tokenizer, and "concept" mines
# reserved-band concepts from the TRAIN growth texts above a base fit that
# mirrors grow()'s own configuration.
_TOKENIZERS = ("byte", "bpe4096", "concept")
_BPE4096_VOCAB_SIZE = 4096


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
    # Execution backend seam (fd84598 lineage, mirrors ladder_experiments):
    # "auto" honors HUMANBRAIN_DEVICE; explicit cpu/cuda wins over it. The
    # environment is bound inside run() BEFORE any circuit is constructed.
    device: str = "auto"
    # "auto" keeps the historical dense-ridge rule while its statistics fit a
    # single-process host budget and upgrades to sparse_three_factor beyond.
    readout_rule: str = "auto"
    # Growth-tokenizer arm; see _TOKENIZERS/_build_growth_tokenizer. The
    # concept-arm mining gates default to the a013 compression-study protocol
    # (freq >= 50, MI >= 5 bits) — sized for full-book corpora, so fixture-
    # scale runs lower them via CLI to let concepts actually flow.
    tokenizer: str = "byte"
    concept_min_frequency: int = 50
    concept_min_mutual_information_bits: float = 5.0


def _resolve_device(requested: str) -> str:
    """Effective device: explicit request wins, else HUMANBRAIN_DEVICE."""
    if requested in _DEVICES:
        return requested
    if requested != "auto":
        raise ValueError(f"device must be one of ('auto', 'cpu', 'cuda'), got {requested!r}")
    environment = os.environ.get("HUMANBRAIN_DEVICE", "cpu").strip().lower()
    return environment if environment in _DEVICES else "cpu"


def _resolve_tokenizer(requested: str) -> str:
    """Effective growth-tokenizer arm."""
    if requested not in _TOKENIZERS:
        raise ValueError(f"tokenizer must be one of {_TOKENIZERS}, got {requested!r}")
    return requested


def _build_growth_tokenizer(
    kind: str,
    texts: list[str],
    *,
    max_phrases: int,
    min_phrase_frequency: int,
    concept_min_frequency: int,
    concept_min_mutual_information_bits: float,
) -> tuple[object | None, dict[str, object]]:
    """Pre-fitted growth tokenizer for the requested arm plus its provenance.

    Returns ``(None, {"kind": "byte"})`` for the byte arm so ``grow()`` keeps
    its internal fit untouched. The concept arm fits the base BytePhrase
    tokenizer with EXACTLY grow()'s configuration (max_phrases /
    min_phrase_frequency), mines recurring multi-token concepts from the same
    TRAIN texts, then wraps both in one ConceptTokenizer whose encode/decode
    interface matches the base. Concepts containing ``b": "`` are dropped:
    qa_corpus pins every prompt to end ``answer:``, so a concept spanning that
    colon-space junction could fuse prompt tail into the answer span and
    break the prefix/suffix seam enforced by _span_evaluation.

    The vocabulary/output-cell alignment invariant (recurrent_language.py
    constructor) holds for every arm because the circuit is sized from the
    returned tokenizer's own vocab_size.
    """
    meta: dict[str, object] = {"kind": kind}
    if kind == "byte":
        return None, meta
    if kind == "concept":
        base = BytePhraseTokenizer.fit(
            texts,
            TokenizerConfig(max_phrases=max_phrases, min_frequency=min_phrase_frequency),
        )
        candidates = [
            candidate
            for candidate in mine_concepts_detailed(
                texts,
                base,
                ConceptConfig(
                    min_frequency=concept_min_frequency,
                    min_mutual_information_bits=concept_min_mutual_information_bits,
                ),
            )
            if b": " not in candidate.payload
        ]
        meta.update(
            {
                "base_vocab_size": base.vocab_size,
                "concept_count": len(candidates),
                "concept_id_base": CONCEPT_ID_BASE,
                "min_frequency": concept_min_frequency,
                "min_mutual_information_bits": concept_min_mutual_information_bits,
            }
        )
        return ConceptTokenizer(base, tuple(c.payload for c in candidates)), meta
    if kind == "bpe4096":
        tokenizer = BpeTokenizer.fit(texts, vocab_size=_BPE4096_VOCAB_SIZE)
        meta.update(
            {
                "target_vocab_size": _BPE4096_VOCAB_SIZE,
                "vocab_size": tokenizer.vocab_size,
            }
        )
        return tokenizer, meta
    raise ValueError(f"tokenizer must be one of {_TOKENIZERS}, got {kind!r}")


def _resolve_readout_rule(reservoir_cells: int, requested: str) -> str:
    """Effective readout rule; ``auto`` guards the dense ridge projection."""
    if requested == "auto":
        features = reservoir_cells + 1
        projected_bytes = features * features * 8
        return "ridge" if projected_bytes <= _MAX_RIDGE_HOST_BYTES else "sparse_three_factor"
    if requested not in _READOUT_RULES:
        raise ValueError(f"readout rule must be auto or one of {_READOUT_RULES}, got {requested!r}")
    return requested


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
    device = _resolve_device(cfg.device)
    readout_rule = _resolve_readout_rule(cfg.circuit.reservoir_cells, cfg.readout_rule)
    tokenizer_kind = _resolve_tokenizer(cfg.tokenizer)
    # Bind the effective device BEFORE any circuit exists (recurrent_kernel
    # reads HUMANBRAIN_DEVICE at construction), mirroring ladder_experiments.
    os.environ["HUMANBRAIN_DEVICE"] = device
    effective_circuit = replace(cfg.circuit, readout_learning_rule=readout_rule)
    curriculum = build_curriculum(cfg.seed)
    train_tasks: tuple[QATask, ...] = curriculum["train_tasks"]
    eval_tasks: tuple[QATask, ...] = curriculum["calibration_tasks"]
    if cfg.max_train_tasks is not None:
        train_tasks = train_tasks[: cfg.max_train_tasks]
    if cfg.eval_limit is not None:
        eval_tasks = eval_tasks[: cfg.eval_limit]

    growth_texts = [growth_text(task) for task in train_tasks]
    tokenizer_override, tokenizer_meta = _build_growth_tokenizer(
        tokenizer_kind,
        growth_texts,
        max_phrases=cfg.max_phrases,
        min_phrase_frequency=cfg.min_phrase_frequency,
        concept_min_frequency=cfg.concept_min_frequency,
        concept_min_mutual_information_bits=cfg.concept_min_mutual_information_bits,
    )
    started = perf_counter()
    model = RecurrentCellLanguageModel.grow(
        growth_texts,
        RecurrentModelConfig(
            growth_epochs=cfg.growth_epochs,
            max_phrases=cfg.max_phrases,
            min_phrase_frequency=cfg.min_phrase_frequency,
            circuit=effective_circuit,
            tokenizer=tokenizer_override,
        ),
    )
    grow_wall_s = perf_counter() - started
    kernel_backend = model._circuit.simulation_backend  # noqa: SLF001 - benchmark seam
    if device == "cuda" and kernel_backend != "cuda":
        raise RuntimeError(
            f"--device cuda requested but the circuit backend is {kernel_backend!r}"
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
    eval_wall_s = perf_counter() - started - grow_wall_s
    wall_s = perf_counter() - started

    growth_token_steps = sum(
        len(model._tokenizer.encode(text)) for text in growth_texts  # noqa: SLF001
    ) * cfg.growth_epochs
    # Circuit operations: each span task teacher-forces len(full_ids)-1 races;
    # each free-run task observes len(prompt) then races answer+slack steps.
    eval_token_steps = sum(int(row["races"]) for row in span_rows) + sum(
        len(model._tokenizer.encode(prompt_for_task(task)))  # noqa: SLF001
        + len(model._tokenizer.encode(" " + task.answer))  # noqa: SLF001
        + cfg.completion_token_slack
        for task in eval_tasks
    )

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
            "--tokenizer selects the growth tokenizer arm (byte/bpe4096/concept); "
            "held-out CALIBRATION teacher-forced answer-span accuracy, free-run "
            "complete() exact match, and first-class degeneracy metrics"
        ),
        "corpus_source": CORPUS_SOURCE,
        "corpus_sha256": corpus_digest(),
        "curriculum_sha256": curriculum["sha256"],
        "seed": cfg.seed,
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "execution": {
            "requested_device": cfg.device,
            "device": device,
            "kernel_backend": kernel_backend,
            "simulation_device": model._circuit.simulation_device,  # noqa: SLF001
            "readout_learning_rule": effective_circuit.readout_learning_rule,
            "reservoir_cells": effective_circuit.reservoir_cells,
            "active_reservoir_cells": model.active_reservoir_cells,
        },
        "model": {
            "tokenizer": tokenizer_meta,
            "vocabulary_size": model.vocabulary_size,
            "token_cells": model._circuit.token_cells,  # noqa: SLF001
            "vocabulary_alignment_pass": vocabulary_alignment_pass,
            "learned_synapses": model.learned_synapses,
            "reservoir_cells": model.reservoir_cells,
            "active_reservoir_cells": model.active_reservoir_cells,
        },
        "training_digest": model.training_digest,
        "throughput": {
            "growth": {"token_steps": growth_token_steps, "wall_s": grow_wall_s},
            "eval": {"token_steps": eval_token_steps, "wall_s": eval_wall_s},
        },
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
    parser.add_argument(
        "--cells",
        "--reservoir-cells",
        dest="reservoir_cells",
        type=int,
        default=96,
        help=(
            "reservoir cell count N; the historical default is 96 (pinned-CPU "
            "python circuit); large N (>=128k) requires --device cuda and the "
            "sparse_three_factor readout (--readout-rule auto selects it)"
        ),
    )
    parser.add_argument("--input-fanout", type=int, default=24)
    parser.add_argument("--recurrent-fanout", type=int, default=8)
    parser.add_argument(
        "--readout-rule",
        choices=("auto", "ridge", "three_factor", "sparse_three_factor"),
        default="auto",
        help=(
            "readout learning rule: auto keeps dense ridge while its F x F "
            "normal matrix fits a 2 GiB host budget and upgrades to "
            "sparse_three_factor beyond (required at N >= ~16k)"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "execution backend: auto honors HUMANBRAIN_DEVICE (default cpu); "
            "cuda selects the CUDA full-touch kernel path for large-N circuits"
        ),
    )
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument(
        "--tokenizer",
        choices=_TOKENIZERS,
        default="byte",
        help=(
            "growth tokenizer: byte = BytePhraseTokenizer fit on TRAIN "
            "(historical default); bpe4096 = plain BPE baseline; concept = "
            "BytePhrase + reserved-band [1024,1536) concepts mined from the "
            "TRAIN texts"
        ),
    )
    parser.add_argument(
        "--concept-min-frequency",
        type=int,
        default=ExperimentConfig.concept_min_frequency,
        help=(
            "concept arm mining gate: n-gram occurrence floor in the TRAIN "
            "texts (study protocol 50; lower it for fixture-scale corpora)"
        ),
    )
    parser.add_argument(
        "--concept-min-mi-bits",
        dest="concept_min_mutual_information_bits",
        type=float,
        default=ExperimentConfig.concept_min_mutual_information_bits,
        help="concept arm mining gate: minimum mutual information in bits",
    )
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
        device=args.device,
        readout_rule=args.readout_rule,
        tokenizer=args.tokenizer,
        concept_min_frequency=args.concept_min_frequency,
        concept_min_mutual_information_bits=args.concept_min_mutual_information_bits,
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
