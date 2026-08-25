"""Red/green tests for the frozen Python-QA corpus bridge.

Covers ``humanbrain.python_qa_adapter`` (the ``--corpus python_qa`` handle
for ``benchmarks/qa_experiments``): pinned-digest integrity against BOTH the
manifest and the locked n-gram baseline records, full-cohort curriculum
semantics (75 TRAIN / 25 VAL), prompt-framing parity with the Bible arm,
guard rails against silent slice re-keying, and one small end-to-end
harness run through the python_qa arm.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from humanbrain import python_qa_adapter as pqa
from humanbrain import qa_corpus
from humanbrain.recurrent import RecurrentCircuitConfig

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "qa_experiments.py"


def _load_qa_experiments():
    spec = importlib.util.spec_from_file_location("qa_experiments_mod", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("qa_experiments_mod", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def artifact() -> dict:
    return pqa.load_frozen_corpus()


def test_frozen_corpus_matches_every_pin(artifact: dict) -> None:
    """Artifact digest must equal manifest pin AND the locked M1 comparators."""
    manifest = json.loads(
        (ROOT / "experiments" / "python_qa_corpus" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert str(manifest["corpus_sha256"]) == pqa.PINNED_CORPUS_SHA256
    assert str(manifest["artifact_sha256"]) == hashlib.sha256(
        (ROOT / "experiments" / "python_qa_corpus" / "python_qa_corpus_v1.json").read_bytes()
    ).hexdigest()
    # The n-gram baseline records were locked BEFORE any cell run; every cell
    # record must carry THEIR corpus digest to stay field-comparable.
    for arm in ("unigram_smoothed", "trigram_smoothed", "majority_floor"):
        record = json.loads(
            (ROOT / "experiments" / "python_qa_baselines_v1" / f"{arm}.json").read_text(
                encoding="utf-8"
            )
        )
        assert record["corpus_sha256"] == pqa.PINNED_CORPUS_SHA256, arm
    assert pqa.corpus_digest() == pqa.PINNED_CORPUS_SHA256


def test_curriculum_serves_full_frozen_cohorts(artifact: dict) -> None:
    curriculum = pqa.build_curriculum(17)
    train_tasks = curriculum["train_tasks"]
    val_tasks = curriculum["calibration_tasks"]
    assert len(train_tasks) == len(artifact["entries"]["TRAIN"]) == 75  # type: ignore[index]
    assert len(val_tasks) == len(artifact["entries"]["VAL"]) == 25  # type: ignore[index]
    assert {task.cohort for task in train_tasks} == {"TRAIN"}
    assert {task.cohort for task in val_tasks} == {"VAL"}
    # Artifact identity: same (question, answer) pairs, no live-mining drift.
    artifact_train_pairs = {
        (str(record["question"]), str(record["answer"]))
        for record in artifact["entries"]["TRAIN"]  # type: ignore[index]
    }
    assert {(task.question, task.answer) for task in train_tasks} == artifact_train_pairs
    # Dense ordinals across the union; opaque ids unique.
    ordinals = sorted(task.ordinal for task in (*train_tasks, *val_tasks))
    assert ordinals == list(range(len(ordinals)))
    tasks = (*train_tasks, *val_tasks)
    assert len({task.qa_id for task in tasks}) == len(tasks)


def test_seed_keys_domains_and_digest_not_cohorts() -> None:
    a = pqa.build_curriculum(17)
    b = pqa.build_curriculum(18)
    assert a["sha256"] != b["sha256"]
    assert a["identifier_domain"] != b["identifier_domain"]
    assert a["order_domain"] != b["order_domain"]
    # Determinism: same seed -> byte-stable curriculum digest.
    assert pqa.build_curriculum(17)["sha256"] == a["sha256"]
    # Same underlying split regardless of seed.
    assert len(a["train_tasks"]) == len(b["train_tasks"])
    assert [t.qa_id for t in a["train_tasks"]] == [t.qa_id for t in b["train_tasks"]]  # type: ignore[attr-defined]


def test_prompt_framing_is_byte_identical_to_bible_arm() -> None:
    task = pqa.corpus_entries()["TRAIN"][0]
    expected = qa_corpus.qa_prompt(task.context, task.question)
    assert pqa.prompt_for_task(task) == expected
    assert pqa.prompt_for_task(task).endswith("\nanswer:")
    assert f"Question: {task.question}" in pqa.prompt_for_task(task)
    assert pqa.growth_text(task) == expected + " " + task.answer


def test_guards_against_partial_slices_and_bad_seeds() -> None:
    with pytest.raises(ValueError):
        pqa.build_curriculum(7, calibration_task_count=6)
    with pytest.raises(ValueError):
        pqa.build_curriculum("17")  # type: ignore[arg-type]


def test_harness_python_qa_arm_end_to_end(tmp_path: None = None) -> None:
    qa_experiments = _load_qa_experiments()
    config = qa_experiments.ExperimentConfig(
        seed=17,
        corpus="python_qa",
        growth_epochs=1,
        circuit=RecurrentCircuitConfig(reservoir_cells=48),
        eval_limit=4,
    )
    record = qa_experiments.run(config)
    assert record["corpus_arm"] == "python_qa"
    assert record["corpus_source"] == pqa.CORPUS_SOURCE
    assert record["corpus_sha256"] == pqa.PINNED_CORPUS_SHA256
    assert record["curriculum_sha256"] == pqa.build_curriculum(17)["sha256"]
    assert record["train_task_count"] == 75
    assert record["eval_task_count"] == 4
    metrics = record["metrics"]
    assert set(metrics) == {"answer_span", "free_run", "degeneracy"}
    assert set(metrics["answer_span"]) == {"tasks", "predictions", "correct", "accuracy"}  # type: ignore[arg-type]
    assert set(metrics["free_run"]) == {"tasks", "exact_matches", "exact_match_rate"}  # type: ignore[arg-type]
    assert metrics["degeneracy"]["evaluated"] == 4  # type: ignore[index]
    assert record["execution"]["kernel_backend"] == "cpu"
    assert record["model"]["vocabulary_alignment_pass"] is True


def test_harness_rejects_unknown_corpus_arm() -> None:
    qa_experiments = _load_qa_experiments()
    with pytest.raises(ValueError):
        qa_experiments._resolve_corpus("swe")  # noqa: SLF001 - unit under test
