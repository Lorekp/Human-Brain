from humanbrain.concept_tokens import (
    CONCEPT_ID_BASE,
    ConceptConfig,
    ConceptTokenizer,
    MAX_CONCEPTS,
    mine_concepts,
)
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig


def _repetitive_corpus() -> list[str]:
    verse = "and the LORD said unto Moses, saying\n"
    code = "    def __init__(self):\n        return self._value\n"
    return [verse * 40 + code * 40] * 3


def test_concept_layer_is_lossless_for_unicode_and_unseen_text() -> None:
    tokenizer = ConceptTokenizer.fit(
        _repetitive_corpus(),
        base_config=TokenizerConfig(max_phrases=64, min_frequency=2),
        config=ConceptConfig(min_frequency=20, min_mutual_information_bits=1.0),
    )
    text = "and the LORD said unto Moses — Δ unseen tail"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_concept_ids_never_collide_with_base_bands() -> None:
    tokenizer = ConceptTokenizer.fit(
        _repetitive_corpus(),
        base_config=TokenizerConfig(max_phrases=64, min_frequency=2),
        config=ConceptConfig(min_frequency=20, min_mutual_information_bits=1.0),
    )
    base = tokenizer.base_tokenizer
    assert base.vocab_size <= 256 + 64  # byte band [0,256) + phrase band
    assert CONCEPT_ID_BASE == 1024 and MAX_CONCEPTS == 512
    concept_band = range(CONCEPT_ID_BASE, CONCEPT_ID_BASE + len(tokenizer.concepts))
    assert all(token_id >= base.vocab_size for token_id in concept_band)
    # Every concept id maps back to its exact payload bytes.
    for offset, payload in enumerate(tokenizer.concepts):
        assert tokenizer.token_bytes(CONCEPT_ID_BASE + offset) == payload


def test_constructor_rejects_base_tokenizer_reaching_concept_band() -> None:
    oversized = BytePhraseTokenizer(
        (f"phrase-piece-{index:04d}".encode() for index in range(1100))
    )
    assert oversized.vocab_size > CONCEPT_ID_BASE
    try:
        ConceptTokenizer(oversized, (b"cell membrane",))
    except ValueError as error:
        assert "reserved concept band" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for oversized base vocabulary")


def test_mining_is_deterministic_across_runs() -> None:
    corpus = _repetitive_corpus()
    base = BytePhraseTokenizer.fit(corpus, TokenizerConfig(max_phrases=64, min_frequency=2))
    first = mine_concepts(corpus, base)
    second = mine_concepts(corpus, base)
    assert first == second
    assert len(first) > 0
    # NOTE: rank keys use overlap-aware token-position counts, so naive
    # non-overlapping bytes.count values need not be monotone across the list;
    # determinism of the exact ordered tuple is the contract under test.
    assert all(isinstance(payload, bytes) and payload for payload in first)


def test_concepts_compress_repeated_text_below_base_layer() -> None:
    corpus = _repetitive_corpus()
    base = BytePhraseTokenizer.fit(corpus, TokenizerConfig(max_phrases=64, min_frequency=2))
    tokenizer = ConceptTokenizer(base, mine_concepts(corpus, base))
    probe = "".join(corpus)
    base_only = len(base.encode(probe))
    with_concepts = len(tokenizer.encode(probe))
    assert with_concepts < base_only


def test_min_frequency_gate_suppresses_sparse_corpora() -> None:
    concepts = mine_concepts(
        ["a tiny corpus"],
        BytePhraseTokenizer.fit(["a tiny corpus"]),
        ConceptConfig(min_frequency=50, max_tokens=4),
    )
    assert concepts == ()


def test_decode_ignores_out_of_range_ids_like_the_base_layer() -> None:
    tokenizer = ConceptTokenizer.fit(
        _repetitive_corpus(),
        base_config=TokenizerConfig(max_phrases=64, min_frequency=2),
        config=ConceptConfig(min_frequency=20, min_mutual_information_bits=1.0),
    )
    assert tokenizer.decode_bytes([10**9]) == b""


def test_concept_encoding_is_deterministic_per_text() -> None:
    tokenizer = ConceptTokenizer.fit(
        _repetitive_corpus(),
        base_config=TokenizerConfig(max_phrases=64, min_frequency=2),
        config=ConceptConfig(min_frequency=20, min_mutual_information_bits=1.0),
    )
    text = _repetitive_corpus()[0][:200]
    assert tokenizer.encode(text) == tokenizer.encode(text)
