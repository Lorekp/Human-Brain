"""Red tests for the QA-v1 experiment gate (``benchmarks/qa_experiments``).

Covers the delegated red-test list for the a013 cognition-design QA track:

- vocabulary alignment invariant enforced where the plumbing map pins it
  (``language.py`` cell-model constructor and ``recurrent_language.py``
  recurrent-model constructor);
- teacher-forced accuracy counts ONLY the answer span;
- degeneracy guard thresholds are first-class fields in every result record;
- two identical ``run()`` invocations produce byte-identical records minus
  wall-clock keys.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from humanbrain.cortex import DendriticCortex, HumanCellParameters
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig
from humanbrain.recurrent_language import RecurrentCellLanguageModel
from humanbrain.language import CellLanguageModel
from humanbrain.tokenizer import BytePhraseTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "qa_experiments.py"
SPEC = importlib.util.spec_from_file_location("qa_experiments", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
QA = importlib.util.module_from_spec(SPEC)
# Register before exec: Python 3.14 dataclasses resolve annotations through
# sys.modules, which importlib file-location modules do not populate.
sys.modules.setdefault("qa_experiments", QA)
SPEC.loader.exec_module(QA)


def _tiny_config(seed: int = 3) -> "QA.ExperimentConfig":
    return QA.ExperimentConfig(
        seed=seed,
        max_train_tasks=6,
        growth_epochs=1,
        max_phrases=128,
        min_phrase_frequency=2,
        circuit=RecurrentCircuitConfig(
            reservoir_cells=48,
            input_fanout=12,
            recurrent_fanout=4,
            output_race_ms=20.0,
        ),
        eval_limit=2,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Vocabulary alignment invariant (pinned enforcement sites)
# ---------------------------------------------------------------------------


def test_recurrent_model_rejects_vocab_circuit_mismatch() -> None:
    tokenizer = BytePhraseTokenizer(phrases=[b"ab"])
    assert tokenizer.vocab_size == 257
    circuit = RecurrentCellCircuit(
        token_cells=256,
        config=RecurrentCircuitConfig(
            reservoir_cells=32, input_fanout=8, recurrent_fanout=4
        ),
    )
    assert circuit.token_cells == 256
    with pytest.raises(ValueError):
        RecurrentCellLanguageModel(tokenizer, circuit, training_digest="0" * 64)


def test_cell_model_rejects_tokenizer_cortex_mismatch() -> None:
    tokenizer = BytePhraseTokenizer()
    assert tokenizer.vocab_size == 256
    cortex = DendriticCortex(
        cell_count=300,
        max_order=2,
        parameters=HumanCellParameters(),
        seed=7,
    )
    assert cortex.cell_count == 300
    with pytest.raises(ValueError):
        CellLanguageModel(tokenizer, cortex, training_digest="0" * 64)


def test_run_record_reports_aligned_vocabulary() -> None:
    record = QA.run(_tiny_config())
    model_stats = record["model"]
    assert model_stats["vocabulary_size"] == model_stats["token_cells"]
    assert model_stats["vocabulary_alignment_pass"] is True


# ---------------------------------------------------------------------------
# Teacher-forced answer-span-only counting
# ---------------------------------------------------------------------------


class _ScriptedCircuit:
    """Duck-typed stand-in for RecurrentCellCircuit with a race script."""

    def __init__(self, winners: list[int]) -> None:
        self._winners = list(winners)
        self.race_calls = 0

    def reset_sequence(self) -> None:  # pragma: no cover - trivial
        pass

    def observe(self, token_id: int) -> None:  # pragma: no cover - trivial
        self.last_observed = token_id

    def race(self):
        winner = self._winners[self.race_calls]
        self.race_calls += 1
        return SimpleNamespace(
            winner=winner, spike_time_ms=1.0, peak_voltage_mv=1.0, active_cells=1
        )


PROMPT = "John 11:35 Jesus wept.\nQuestion: Who wept?\nanswer:"
ANSWER = "Jesus"


def _model_like(winners: list[int]) -> SimpleNamespace:
    return SimpleNamespace(_circuit=_ScriptedCircuit(winners), _tokenizer=BytePhraseTokenizer())


def _full_ids() -> tuple[list[int], int]:
    tokenizer = BytePhraseTokenizer()
    full = tokenizer.encode(f"{PROMPT} {ANSWER}")
    span_start = len(full) - len(tokenizer.encode(f" {ANSWER}"))
    return full, span_start


def _script(full: list[int], correct_targets: set[int]) -> list[int]:
    """Winners indexed by race call: race r predicts target r+1."""
    return [
        full[r + 1] if (r + 1) in correct_targets else (full[r + 1] + 97) % 256
        for r in range(len(full) - 1)
    ]


def test_span_accuracy_counts_only_answer_positions_when_correct() -> None:
    full, span_start = _full_ids()
    # Wrong everywhere before the span, correct across the whole span.
    winners = _script(full, set(range(span_start, len(full))))
    result = QA._span_evaluation(_model_like(winners), PROMPT, ANSWER)
    assert result["predictions"] == len(full) - span_start
    assert result["correct"] == result["predictions"]
    assert result["accuracy"] == 1.0
    # Teacher forcing walks the entire sequence, but only span races count.
    assert result["races"] == len(full) - 1


def test_span_accuracy_counts_only_answer_positions_when_wrong() -> None:
    full, span_start = _full_ids()
    # Correct before the span, wrong across the whole span.
    winners = _script(full, set(range(0, span_start)))
    result = QA._span_evaluation(_model_like(full), PROMPT, ANSWER)
    assert result["predictions"] == len(full) - span_start
    assert result["correct"] == 0
    assert result["accuracy"] == 0.0


def test_span_evaluation_never_counts_context_predictions_in_accuracy() -> None:
    full, span_start = _full_ids()
    assert span_start >= 2
    # Exactly one correct race, targeting the last CONTEXT position just
    # before the span: it must not be counted.
    winners = _script(full, {span_start - 1})
    result = QA._span_evaluation(_model_like(full), PROMPT, ANSWER)
    assert result["correct"] == 0
    assert result["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Degeneracy metrics are first-class schema fields
# ---------------------------------------------------------------------------


def test_degeneracy_metrics_and_guard_thresholds_are_reported() -> None:
    collapsed = QA._degeneracy_metrics(["alpha", "alpha", "alpha"])
    assert collapsed["evaluated"] == 3
    assert collapsed["distinct_answers"] == 1
    assert collapsed["modal_share"] == 1.0
    guard = collapsed["guard"]
    assert guard["modal_share_max"] == QA.GUARD_MODAL_SHARE_MAX
    assert guard["distinct_answers_min"] == QA.GUARD_DISTINCT_MIN
    assert collapsed["guard_pass"] is False

    varied = QA._degeneracy_metrics(["alpha", "beta", "gamma"])
    assert varied["distinct_answers"] == 3
    assert varied["modal_share"] == pytest.approx(1 / 3)
    assert varied["guard_pass"] is True


def test_run_record_carries_degeneracy_block() -> None:
    record = QA.run(_tiny_config())
    degeneracy = record["metrics"]["degeneracy"]
    assert degeneracy["evaluated"] == record["metrics"]["free_run"]["tasks"]
    assert 0.0 <= degeneracy["modal_share"] <= 1.0
    assert degeneracy["guard"]["modal_share_max"] == QA.GUARD_MODAL_SHARE_MAX
    assert isinstance(degeneracy["guard_pass"], bool)


# ---------------------------------------------------------------------------
# Determinism: byte-identical records minus wall clock
# ---------------------------------------------------------------------------


def _strip_wall(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _strip_wall(item)
            for key, item in value.items()
            if key not in ("wall_s", "wall_seconds")
        }
    if isinstance(value, list):
        return [_strip_wall(item) for item in value]
    return value


def test_two_runs_are_byte_identical_minus_wall_clock() -> None:
    record_a = QA.run(_tiny_config())
    record_b = QA.run(_tiny_config())

    # Wall clock exists and is stripped recursively.
    assert "wall_s" in record_a
    stripped_a = _strip_wall(record_a)
    stripped_b = _strip_wall(record_b)

    digest_a = sha256(_canonical(stripped_a)).hexdigest()
    digest_b = sha256(_canonical(stripped_b)).hexdigest()
    assert digest_a == digest_b
    assert record_a["training_digest"] == record_b["training_digest"]

    # The public checker agrees and reports both digests.
    verdict = QA.check_determinism(_tiny_config())
    assert verdict["determinism_pass"] is True
    assert verdict["record_sha256_a"] == digest_a
    assert verdict["record_sha256_b"] == digest_b


def test_curriculum_digest_flows_into_run_record() -> None:
    from humanbrain import qa_corpus

    config = _tiny_config(seed=11)
    record = QA.run(config)
    assert record["curriculum_sha256"] == qa_corpus.build_curriculum(11)["sha256"]
    assert record["corpus_sha256"] == qa_corpus.corpus_digest()
    assert record["schema_version"] == "humanbrain/qa-experiments/v1"
