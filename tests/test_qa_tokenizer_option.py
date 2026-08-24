"""Red tests for the ``--tokenizer`` option in the QA-v1 experiment gate.

Extends ``tests/test_qa_gate.py`` to the concept-token track (a013): the QA
path must accept ``--tokenizer {byte,bpe4096,concept}`` where ``concept``
mines recurring multi-token concepts from the TRAIN growth texts per
``humanbrain.concept_tokens`` and wraps them above BytePhraseTokenizer in the
same encode/decode interface.

Pinned here:

- concept mining is deterministic for a given corpus and configuration;
- encode/decode round-trips losslessly on growth AND evaluation texts;
- concept ids stay inside the reserved band ``[1024, 1024+MAX_CONCEPTS)``
  and never collide with the base bands;
- the vocabulary/output-cell alignment invariant holds through ``run()``
  for every tokenizer arm, with tokenizer provenance in the record;
- the answer-span seam precondition (prompt ids are a prefix of the full
  encoding, answer ids a suffix) survives concept fusion — no mined
  concept may span the ``answer:`` + ``' '`` prompt/answer junction;
- two concept-arm runs produce byte-identical records minus wall clock.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from humanbrain.bpe_tokenizer import BpeTokenizer
from humanbrain.concept_tokens import (
    CONCEPT_ID_BASE,
    ConceptConfig,
    ConceptTokenizer,
    MAX_CONCEPTS,
    mine_concepts,
)
from humanbrain.qa_corpus import build_curriculum, growth_text, prompt_for_task
from humanbrain.recurrent import RecurrentCircuitConfig
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "qa_experiments.py"
SPEC = importlib.util.spec_from_file_location("qa_experiments", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
QA = importlib.util.module_from_spec(SPEC)
# Register before exec: Python 3.14 dataclasses resolve annotations through
# sys.modules, which importlib file-location modules do not populate.
sys.modules.setdefault("qa_experiments", QA)
SPEC.loader.exec_module(QA)

SEED = 3
TRAIN_TASKS = 8


def _curriculum(seed: int = SEED) -> dict[str, object]:
    return build_curriculum(seed)


def _growth_texts(seed: int = SEED, count: int = TRAIN_TASKS) -> list[str]:
    tasks = _curriculum(seed)["train_tasks"][:count]  # type: ignore[index]
    return [growth_text(task) for task in tasks]


def _eval_texts(seed: int = SEED, limit: int = 2) -> list[tuple[str, str]]:
    tasks = _curriculum(seed)["calibration_tasks"][:limit]  # type: ignore[index]
    return [(prompt_for_task(task), task.answer) for task in tasks]


def _circuit_config() -> RecurrentCircuitConfig:
    return RecurrentCircuitConfig(
        reservoir_cells=48,
        input_fanout=12,
        recurrent_fanout=4,
        output_race_ms=20.0,
    )


def _tiny_config(**overrides: object) -> "QA.ExperimentConfig":
    params: dict[str, object] = {
        "seed": SEED,
        "max_train_tasks": TRAIN_TASKS,
        "growth_epochs": 1,
        "max_phrases": 128,
        "min_phrase_frequency": 2,
        "circuit": _circuit_config(),
        "eval_limit": 2,
    }
    params.update(overrides)
    return QA.ExperimentConfig(**params)  # type: ignore[arg-type]


def _concept_config(seed: int = SEED) -> "QA.ExperimentConfig":
    """Fixture-scale mining gates: the study defaults (freq>=50, MI>=5 bits)
    are tuned for full-book corpora and legitimately mine zero concepts from
    the 24-text fixture; lowered gates make concept tokens actually flow."""
    return _tiny_config(
        seed=seed,
        tokenizer="concept",
        concept_min_frequency=2,
        concept_min_mutual_information_bits=2.0,
    )


# ---------------------------------------------------------------------------
# Deterministic mining + lossless round-trip on QA texts
# ---------------------------------------------------------------------------


def test_concept_mining_is_deterministic_and_roundtrips_qa_texts() -> None:
    texts = _growth_texts()
    gates = ConceptConfig(min_frequency=2, min_mutual_information_bits=2.0)
    base = BytePhraseTokenizer.fit(texts, TokenizerConfig(max_phrases=128, min_frequency=2))

    concepts_a = mine_concepts(texts, base, gates)
    concepts_b = mine_concepts(texts, base, gates)
    assert concepts_a == concepts_b, "mining must be deterministic for fixed corpus+config"
    assert len(concepts_a) >= 1, "lowered gates must surface concepts at fixture scale"

    tokenizer = ConceptTokenizer(base, concepts_a)
    assert len(tokenizer.concepts) <= MAX_CONCEPTS
    for text in texts + [full for full, _ in _eval_texts()]:
        ids = tokenizer.encode(text)
        assert tokenizer.encode(text) == ids, "encoding must be repeatable"
        assert tokenizer.decode_bytes(ids) == text.encode("utf-8"), "round-trip must be lossless"


def test_junction_safe_construction_never_spans_answer_colon_space() -> None:
    """No mined concept may contain ``b\": \"`` — the QA prompt framing pins
    prompts to end ``answer:``, and a concept spanning that junction would
    fuse prompt tail into the answer span and break the prefix/suffix seam."""
    texts = _growth_texts()
    base = BytePhraseTokenizer.fit(texts, TokenizerConfig(max_phrases=128, min_frequency=2))
    gates = ConceptConfig(min_frequency=2, min_mutual_information_bits=2.0)
    built = QA._build_growth_tokenizer(  # noqa: SLF001 - benchmark seam under test
        "concept",
        texts,
        max_phrases=128,
        min_phrase_frequency=2,
        concept_min_frequency=2,
        concept_min_mutual_information_bits=2.0,
    )
    tokenizer, meta = built
    assert isinstance(tokenizer, ConceptTokenizer)
    assert meta["kind"] == "concept"
    for payload in tokenizer.concepts:
        assert b": " not in payload, f"junction-spanning concept mined: {payload!r}"

    # The span-seam precondition holds for every train and eval text pair.
    eval_pairs = _eval_texts()
    for text in texts:
        colon_index = text.rfind("answer:")
        assert colon_index >= 0
        prompt, answer = text[:colon_index + len("answer:")], " " + text[colon_index + len("answer:") + 1:]
        full = tokenizer.encode(prompt + " " + answer)
        prompt_ids = tokenizer.encode(prompt)
        answer_ids = tokenizer.encode(answer)
        assert tuple(full[: len(prompt_ids)]) == tuple(prompt_ids)
        assert tuple(full[len(full) - len(answer_ids) :]) == tuple(answer_ids)
    for prompt, answer in eval_pairs:
        full = tokenizer.encode(prompt + " " + answer)
        prompt_ids = tokenizer.encode(prompt)
        answer_ids = tokenizer.encode(" " + answer)
        assert tuple(full[: len(prompt_ids)]) == tuple(prompt_ids)
        assert tuple(full[len(full) - len(answer_ids) :]) == tuple(answer_ids)


# ---------------------------------------------------------------------------
# Reserved-band layout: no collisions with the base bands
# ---------------------------------------------------------------------------


def test_concept_ids_stay_in_reserved_band_without_collision() -> None:
    texts = _growth_texts()
    base = BytePhraseTokenizer.fit(texts, TokenizerConfig(max_phrases=128, min_frequency=2))
    gates = ConceptConfig(min_frequency=2, min_mutual_information_bits=2.0)
    tokenizer = ConceptTokenizer(base, mine_concepts(texts, base, gates))

    assert base.vocab_size <= CONCEPT_ID_BASE, "constructor precondition"
    assert tokenizer.vocab_size == max(
        base.vocab_size, CONCEPT_ID_BASE + len(tokenizer.concepts)
    )

    observed: set[int] = set()
    for text in texts + [full for full, _ in _eval_texts()]:
        observed.update(tokenizer.encode(text))
    base_band = {token_id for token_id in observed if token_id < CONCEPT_ID_BASE}
    concept_band = {token_id for token_id in observed if token_id >= CONCEPT_ID_BASE}
    assert base_band and concept_band, "fixture run must exercise both bands"
    assert max(base_band) < base.vocab_size, "base ids escape the base vocabulary"
    assert min(concept_band) >= CONCEPT_ID_BASE, "concept ids below the reserved band"
    assert max(concept_band) < CONCEPT_ID_BASE + MAX_CONCEPTS, "concept ids overflow the band"


def test_concept_vocab_size_covers_the_reserved_band_end() -> None:
    """Regression: vocab_size must bound the highest emittable id.

    Concept ids live at [1024, 1024+n), so ``base + n`` understates the id
    space and any consumer sizing dense resources from vocab_size (circuit
    output cells) would reject valid concept ids."""
    texts = _growth_texts()
    base = BytePhraseTokenizer.fit(texts, TokenizerConfig(max_phrases=128, min_frequency=2))
    gates = ConceptConfig(min_frequency=2, min_mutual_information_bits=2.0)

    empty = ConceptTokenizer(base, ())
    assert empty.vocab_size == base.vocab_size

    full = ConceptTokenizer(base, mine_concepts(texts, base, gates))
    assert len(full.concepts) >= 1
    assert full.vocab_size == CONCEPT_ID_BASE + len(full.concepts)
    assert all(token_id < full.vocab_size for text in texts for token_id in full.encode(text))


# ---------------------------------------------------------------------------
# run() alignment invariant + provenance for every tokenizer arm
# ---------------------------------------------------------------------------


def test_run_concept_arm_aligns_vocab_and_records_provenance() -> None:
    record = QA.run(_concept_config())
    model = record["model"]
    assert model["vocabulary_alignment_pass"] is True
    assert model["vocabulary_size"] == model["token_cells"]

    provenance = model["tokenizer"]
    assert provenance["kind"] == "concept"
    assert provenance["concept_count"] >= 1
    assert provenance["concept_id_base"] == CONCEPT_ID_BASE
    assert provenance["min_frequency"] == 2
    assert provenance["min_mutual_information_bits"] == 2.0
    # vocab_size is the reserved-band end: ids jump across [768, 1024)
    # headroom, so the circuit must be sized past CONCEPT_ID_BASE.
    assert model["vocabulary_size"] == (
        provenance["concept_id_base"] + provenance["concept_count"]
    )

    metrics = record["metrics"]
    assert {"answer_span", "free_run", "degeneracy"} <= set(metrics)


def test_default_byte_arm_records_byte_kind_and_stays_aligned() -> None:
    record = QA.run(_tiny_config())
    model = record["model"]
    assert model["tokenizer"]["kind"] == "byte"
    assert model["vocabulary_alignment_pass"] is True
    assert model["vocabulary_size"] == model["token_cells"]


def test_bpe4096_arm_aligns_vocab_and_records_kind() -> None:
    record = QA.run(_tiny_config(tokenizer="bpe4096"))
    model = record["model"]
    assert model["tokenizer"]["kind"] == "bpe4096"
    assert model["vocabulary_alignment_pass"] is True
    assert model["vocabulary_size"] == model["token_cells"]


def test_unknown_tokenizer_kind_is_rejected() -> None:
    with pytest.raises(ValueError):
        QA.run(_tiny_config(tokenizer="gpt2"))


def test_span_seam_precondition_holds_for_all_three_arms() -> None:
    """The _span_evaluation guards demand encode(prefix)/encode(suffix)
    consistency; pin it for byte, concept, and bpe4096 alike, each built
    through the production _build_growth_tokenizer path."""
    texts = _growth_texts()

    def arms() -> dict[str, object]:
        byte_tokenizer = BytePhraseTokenizer.fit(
            texts, TokenizerConfig(max_phrases=128, min_frequency=2)
        )
        concept_tokenizer, _ = QA._build_growth_tokenizer(  # noqa: SLF001 - seam under test
            "concept",
            texts,
            max_phrases=128,
            min_phrase_frequency=2,
            concept_min_frequency=2,
            concept_min_mutual_information_bits=2.0,
        )
        bpe_tokenizer, _ = QA._build_growth_tokenizer(  # noqa: SLF001 - seam under test
            "bpe4096",
            texts,
            max_phrases=128,
            min_phrase_frequency=2,
            concept_min_frequency=50,
            concept_min_mutual_information_bits=5.0,
        )
        return {
            "byte": byte_tokenizer,
            "concept": concept_tokenizer,
            "bpe4096": bpe_tokenizer,
        }

    pairs = [
        (prompt, " " + answer)
        for prompt, answer in _eval_texts()
    ] + [
        # Split growth texts at the real framing boundary ("answer:"), never
        # at the last space: several fixture answers contain spaces, and a
        # mid-answer split would manufacture fake fusion opportunities that
        # the QA seam never exercises.
        (text[: text.rfind("answer:") + len("answer:")], text[text.rfind("answer:") + len("answer:"):])
        for text in texts
    ]
    for name, tokenizer in arms().items():
        for prompt, answer in pairs:
            full = tokenizer.encode(prompt + answer)
            prompt_ids = tokenizer.encode(prompt)
            answer_ids = tokenizer.encode(answer)
            assert tuple(full[: len(prompt_ids)]) == tuple(prompt_ids), (
                f"{name}: prompt ids not a prefix of full encoding"
            )
            assert tuple(full[len(full) - len(answer_ids) :]) == tuple(answer_ids), (
                f"{name}: answer ids not a suffix of full encoding"
            )


def test_unfiltered_concept_mining_would_break_the_span_seam() -> None:
    """Negative control: raw mined concepts include junction-spanners such as
    b\"answer: the LORD\"; without the b\": \" filter the prefix/suffix seam
    fails on every fixture pair. This pins WHY _build_growth_tokenizer
    filters, so the guard is not 'simplified' away later."""
    texts = _growth_texts()
    base = BytePhraseTokenizer.fit(texts, TokenizerConfig(max_phrases=128, min_frequency=2))
    raw_concepts = mine_concepts(
        texts, base, ConceptConfig(min_frequency=2, min_mutual_information_bits=2.0)
    )
    assert any(b": " in payload for payload in raw_concepts), (
        "fixture corpus must keep mining junction-spanning candidates for "
        "this control to stay meaningful"
    )
    raw_tokenizer = ConceptTokenizer(base, raw_concepts)
    cut = len("answer:")
    violated = 0
    for text in texts:
        prompt, answer = text[: text.rfind("answer:") + cut], text[text.rfind("answer:") + cut :]
        full = raw_tokenizer.encode(prompt + answer)
        prompt_ids = raw_tokenizer.encode(prompt)
        if tuple(full[: len(prompt_ids)]) != tuple(prompt_ids):
            violated += 1
    assert violated >= 1, "unfiltered concepts must break the seam somewhere"


# ---------------------------------------------------------------------------
# Concept-arm determinism end to end
# ---------------------------------------------------------------------------


def test_concept_arm_check_determinism_passes() -> None:
    verdict = QA.check_determinism(_concept_config(seed=SEED))
    assert verdict["determinism_pass"] is True
    assert verdict["record_sha256_a"] == verdict["record_sha256_b"]
