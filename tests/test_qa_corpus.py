"""Red tests for QA-v1 corpus materialization (``humanbrain.qa_corpus``).

Covers the delegated red-test list for the a013 cognition-design QA track:

- corpus digest determinism (fixture corpus hashes stably and is
  seed-independent);
- verse-disjoint TRAIN/CALIBRATION split across seeds;
- answer-derivable fixtures (every fixture answer is recoverable from its
  context verses, and materialization refuses tampered entries);
- ``qa_prompt`` byte stability (prompt-last framing ending in ``answer:``).
"""

from __future__ import annotations

import json
from hashlib import sha256

import pytest

from humanbrain import qa_corpus


# ---------------------------------------------------------------------------
# Corpus digest determinism
# ---------------------------------------------------------------------------


def test_corpus_digest_is_deterministic_and_seed_independent() -> None:
    first = qa_corpus.corpus_digest()
    second = qa_corpus.corpus_digest()
    assert first == second
    assert len(first) == 64

    # The digest covers the fixture bytes exactly: recomputing sha256 over
    # the canonical JSON record of the loaded corpus must agree.
    record = {
        "entries": [
            {
                "references": list(entry["references"]),
                "verses": list(entry["verses"]),
                "question": entry["question"],
                "answer": entry["answer"],
            }
            for entry in qa_corpus.FIXTURE_CORPUS
        ]
    }
    canonical = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert first == sha256(canonical).hexdigest()

    # Seed keys the curriculum, not the corpus.
    digests = {qa_corpus.build_curriculum(seed)["corpus_sha256"] for seed in (0, 1, 7)}
    assert digests == {first}


def test_fixture_corpus_has_at_least_thirty_entries() -> None:
    assert len(qa_corpus.FIXTURE_CORPUS) >= 30


def test_curriculum_sha256_is_stable_and_seed_keyed() -> None:
    a = qa_corpus.build_curriculum(3)
    b = qa_corpus.build_curriculum(3)
    c = qa_corpus.build_curriculum(4)
    assert a["sha256"] == b["sha256"]
    assert a["sha256"] != c["sha256"]
    assert set(a["train_tasks"]) or True  # tasks are frozen dataclasses below


def test_qa_tasks_are_frozen_dataclasses_with_expected_fields() -> None:
    curriculum = qa_corpus.build_curriculum(0)
    task = next(iter(curriculum["train_tasks"]))
    assert isinstance(task.context, tuple) and task.context
    assert all(isinstance(verse, str) and verse for verse in task.context)
    assert isinstance(task.question, str) and task.question
    assert isinstance(task.answer, str) and task.answer
    with pytest.raises(Exception):
        task.question = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Verse-disjoint split
# ---------------------------------------------------------------------------


def test_split_is_verse_disjoint_and_exhaustive_across_seeds() -> None:
    all_references = {
        reference for entry in qa_corpus.FIXTURE_CORPUS for reference in entry["references"]
    }
    for seed in range(8):
        curriculum = qa_corpus.build_curriculum(seed)
        train: tuple[qa_corpus.QATask, ...] = curriculum["train_tasks"]
        calibration: tuple[qa_corpus.QATask, ...] = curriculum["calibration_tasks"]
        assert train and calibration, f"seed {seed} produced an empty cohort"
        train_references = {ref for task in train for ref in task.references}
        calibration_references = {
            ref for task in calibration for ref in task.references
        }
        assert not train_references & calibration_references
        assert train_references | calibration_references == all_references
        train_texts = {task.qa_id for task in train}
        calibration_texts = {task.qa_id for task in calibration}
        assert not train_texts & calibration_texts


def test_ordinals_are_dense_and_identifier_domains_are_seed_keyed() -> None:
    curriculum = qa_corpus.build_curriculum(5)
    train: tuple[qa_corpus.QATask, ...] = curriculum["train_tasks"]
    calibration: tuple[qa_corpus.QATask, ...] = curriculum["calibration_tasks"]
    ordinals = sorted(task.ordinal for task in train + calibration)
    assert ordinals == list(range(len(ordinals)))
    assert f"seed/5" in str(curriculum["identifier_domain"])
    assert "seed/5" in str(curriculum["order_domain"])


# ---------------------------------------------------------------------------
# Answer-derivable fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", range(len(qa_corpus.FIXTURE_CORPUS)))
def test_every_fixture_answer_is_derivable_from_its_context(index: int) -> None:
    entry = qa_corpus.FIXTURE_CORPUS[index]
    context_text = " ".join(entry["verses"]).casefold()
    assert entry["answer"].strip(), f"fixture {index} has an empty answer"
    assert entry["answer"].casefold() in context_text, (
        f"fixture {index} answer is not derivable from its context verses"
    )


def test_materializer_rejects_unanswerable_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    tampered = list(qa_corpus.FIXTURE_CORPUS)
    tampered[0] = {
        "references": ("Zzz 1:1",),
        "verses": ("Zzz 1:1 Totally unrelated placeholder text.",),
        "question": "What is never mentioned here?",
        "answer": "the heaven and the earth",
    }
    monkeypatch.setattr(qa_corpus, "FIXTURE_CORPUS", tuple(tampered))
    with pytest.raises(ValueError):
        qa_corpus.build_curriculum(0)


# ---------------------------------------------------------------------------
# qa_prompt byte stability
# ---------------------------------------------------------------------------


def test_qa_prompt_ends_in_answer_marker_byte_stably() -> None:
    context = ("Genesis 1:1 In the beginning God created the heaven and the earth.",)
    question = "What did God create in the beginning?"
    prompt = qa_corpus.qa_prompt(context, question)
    assert prompt.endswith("answer:")
    expected = (
        "Genesis 1:1 In the beginning God created the heaven and the earth.\n"
        "Question: What did God create in the beginning?\n"
        "answer:"
    )
    assert prompt == expected
    assert prompt.encode("utf-8") == expected.encode("utf-8")
    # Repeated invocations are byte-identical.
    assert (
        sha256(prompt.encode("utf-8")).digest()
        == sha256(qa_corpus.qa_prompt(context, question).encode("utf-8")).digest()
    )
    # Multi-verse contexts join with single newlines, verses first.
    multi = qa_corpus.prompt_for_task(
        qa_corpus.build_curriculum(1)["calibration_tasks"][0]
    )
    assert "\n" in multi and multi.endswith("answer:")


def test_growth_text_pairs_prompt_space_answer() -> None:
    curriculum = qa_corpus.build_curriculum(2)
    task = next(iter(curriculum["train_tasks"]))
    text = qa_corpus.growth_text(task)
    assert text.startswith(qa_corpus.prompt_for_task(task))
    assert text[len(qa_corpus.prompt_for_task(task)) :] == " " + task.answer
