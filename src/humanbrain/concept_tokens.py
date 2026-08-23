"""Concept-chunk tokenizer: recurring multi-token concepts above BytePhraseTokenizer.

Instead of stopping at surface forms (bytes + short phrase pieces), this layer
mines RECURRING MULTI-TOKEN CONCEPTS from a fitted corpus and maps each to a
single id in a reserved band, so a phrase such as ``the LORD said`` or
``def __init__(self):`` becomes one atomic unit.

ID layout (cited from ``tokenizer.BytePhraseTokenizer``)::

    _PIECE candidates -> self._id_to_bytes = [bytes([i]) for i in range(256)] + unique
    phrase ids enumerate(unique, start=256), capped by TokenizerConfig.max_phrases = 512

Therefore the base vocabulary occupies ids ``[0, 256 + max_phrases)`` — ``[0, 768)``
under the default configuration.  Concept ids live in the reserved band
``[CONCEPT_ID_BASE, CONCEPT_ID_BASE + MAX_CONCEPTS)`` = ``[1024, 1536)``; ids
``[768, 1024)`` stay unused as headroom.  The constructor refuses any base
tokenizer whose vocabulary reaches into the concept band.

The miner (``mine_concepts``) is exact for its stated gates and stdlib-only:

* n-gram frequency via level-wise Apriori generation: an n-gram is only counted
  where its (n-1)-token prefix is itself frequent, which preserves every
  n-gram meeting ``min_frequency`` while bounding work;
* mutual-information gate ``I(ngram) = log2( p(ngram) / prod p(token_i) )``
  measured in bits over base-token unigram statistics;
* deduplicated, ranked by (-count, -mutual information, bytes) so fitting is
  deterministic for a given corpus and configuration.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

#: First id of the reserved concept band (see module docstring for the layout).
CONCEPT_ID_BASE = 1024
#: Maximum number of concepts; the reserved band is [1024, 1024 + MAX_CONCEPTS).
MAX_CONCEPTS = 512


@dataclass(frozen=True)
class ConceptConfig:
    """Mining gates. Defaults follow the a013 brief: freq >= 50, MI filter."""

    #: n-gram must occur at least this many times in the mining corpus.
    min_frequency: int = 50
    #: Minimum pointwise mutual information of the n-gram, in bits.
    min_mutual_information_bits: float = 5.0
    #: Maximum number of concepts retained after ranking.
    max_concepts: int = MAX_CONCEPTS
    #: A concept spans at least this many base tokens (multi-token by definition).
    min_tokens: int = 2
    #: Longest concept, in base tokens.
    max_tokens: int = 12


@dataclass(frozen=True)
class ConceptCandidate:
    """A mined concept with its mining statistics."""

    payload: bytes
    count: int
    mutual_information_bits: float


def mine_concepts_detailed(
    texts: Iterable[str],
    base: BytePhraseTokenizer,
    config: ConceptConfig | None = None,
) -> tuple[ConceptCandidate, ...]:
    """Mine recurring multi-token concepts from ``texts`` above ``base``.

    Returns candidates in deterministic rank order (descending count,
    descending mutual information, ascending bytes).
    """
    cfg = config or ConceptConfig()
    streams = [base.encode(text) for text in texts]
    total_tokens = sum(len(stream) for stream in streams)
    if total_tokens == 0 or cfg.min_tokens < 2 or cfg.max_tokens < cfg.min_tokens:
        return ()

    unigram: Counter[int] = Counter()
    for stream in streams:
        unigram.update(stream)
    log_total = math.log2(total_tokens)

    def mutual_information_bits(ngram: Sequence[int], count: int) -> float:
        # log2( (count/N) / prod(count_i/N) ) = log2(count) + (n-1)*log2(N) - sum(log2(count_i))
        score = math.log2(count) + (len(ngram) - 1) * log_total
        for token_id in ngram:
            score -= math.log2(unigram[token_id])
        return score

    # pool: n-gram -> (count, mi bits); frequent_prev: frequency-passing (n-1)-grams.
    # The Apriori chain always starts at bigrams so prefixes are available even
    # when cfg.min_tokens > 2; only levels >= min_tokens feed the pool.
    pool: dict[tuple[int, ...], tuple[int, float]] = {}
    frequent_prev: dict[tuple[int, ...], int] | None = None
    for level in range(2, cfg.max_tokens + 1):
        counts: Counter[tuple[int, ...]] = Counter()
        if level == 2:
            for stream in streams:
                for i in range(len(stream) - 1):
                    counts[(stream[i], stream[i + 1])] += 1
        else:
            prefix_len = level - 1
            if not frequent_prev:
                break
            for stream in streams:
                for i in range(len(stream) - prefix_len):
                    if tuple(stream[i : i + prefix_len]) in frequent_prev:
                        counts[tuple(stream[i : i + level])] += 1
        frequent = {
            ngram: count for ngram, count in counts.items() if count >= cfg.min_frequency
        }
        if level >= cfg.min_tokens:
            for ngram, count in frequent.items():
                score = mutual_information_bits(ngram, count)
                if score >= cfg.min_mutual_information_bits:
                    pool[ngram] = (count, score)
        frequent_prev = frequent

    ranked: list[tuple[int, float, bytes, tuple[int, ...]]] = []
    seen_bytes: set[bytes] = set()
    for ngram, (count, score) in pool.items():
        payload = base.decode_bytes(ngram)
        if payload in seen_bytes:
            continue
        seen_bytes.add(payload)
        ranked.append((count, score, payload, ngram))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return tuple(
        ConceptCandidate(payload=payload, count=count, mutual_information_bits=score)
        for count, score, payload, _ in ranked[: cfg.max_concepts]
    )


def mine_concepts(
    texts: Iterable[str],
    base: BytePhraseTokenizer,
    config: ConceptConfig | None = None,
) -> tuple[bytes, ...]:
    """Mine recurring multi-token concepts; see :func:`mine_concepts_detailed`."""
    return tuple(
        candidate.payload for candidate in mine_concepts_detailed(texts, base, config)
    )


class ConceptTokenizer:
    """Lossless tokenizer whose ids are base ids plus reserved-band concept ids.

    Wraps a fitted :class:`BytePhraseTokenizer`; concept ids occupy
    ``[CONCEPT_ID_BASE, CONCEPT_ID_BASE + len(concepts))`` and never collide
    with the base bands ``[0, 256)`` (bytes) and ``[256, 256 + max_phrases)``
    (phrase pieces) because the constructor enforces
    ``base.vocab_size <= CONCEPT_ID_BASE``.
    """

    def __init__(self, base: BytePhraseTokenizer, concepts: Iterable[bytes] = ()) -> None:
        if base.vocab_size > CONCEPT_ID_BASE:
            raise ValueError(
                "base tokenizer vocabulary reaches the reserved concept band: "
                f"vocab_size={base.vocab_size} must be <= CONCEPT_ID_BASE={CONCEPT_ID_BASE}"
            )
        unique: list[bytes] = []
        seen: set[bytes] = set()
        for piece in concepts:
            payload = bytes(piece)
            if not payload or payload in seen:
                continue
            seen.add(payload)
            unique.append(payload)
        if len(unique) > MAX_CONCEPTS:
            raise ValueError(
                f"{len(unique)} concepts exceeds the reserved band capacity {MAX_CONCEPTS}"
            )
        self._base = base
        self._concepts = tuple(unique)
        self._concept_token_ids: list[tuple[int, ...]] = []
        for payload in self._concepts:
            ids = tuple(base.encode(payload.decode("utf-8")))
            if len(ids) < 2:
                raise ValueError(
                    "concepts must span multiple base tokens; "
                    f"{payload!r} encodes to {len(ids)} token(s)"
                )
            self._concept_token_ids.append(ids)

        self._trie: list[dict[int, int]] = [{}]
        self._terminal: list[int] = [-1]
        for index, ids in enumerate(self._concept_token_ids):
            node = 0
            for token_id in ids:
                child = self._trie[node].get(token_id)
                if child is None:
                    child = len(self._trie)
                    self._trie[node][token_id] = child
                    self._trie.append({})
                    self._terminal.append(-1)
                node = child
            self._terminal[node] = index

    @classmethod
    def fit(
        cls,
        texts: Iterable[str],
        base: BytePhraseTokenizer | None = None,
        config: ConceptConfig | None = None,
        base_config: TokenizerConfig | None = None,
    ) -> "ConceptTokenizer":
        """Fit the base tokenizer (if none supplied) and mine concepts."""
        materialized = list(texts)
        fitted_base = base if base is not None else BytePhraseTokenizer.fit(
            materialized, base_config
        )
        concepts = mine_concepts(materialized, fitted_base, config)
        return cls(fitted_base, concepts)

    @property
    def base_tokenizer(self) -> BytePhraseTokenizer:
        return self._base

    @property
    def concepts(self) -> tuple[bytes, ...]:
        """Concept payloads, indexed by ``id - CONCEPT_ID_BASE``."""
        return self._concepts

    @property
    def concept_token_sequences(self) -> tuple[tuple[int, ...], ...]:
        """Canonical base-token id sequence of each concept."""
        return tuple(self._concept_token_ids)

    @property
    def vocab_size(self) -> int:
        return self._base.vocab_size + len(self._concepts)

    def encode(self, text: str) -> list[int]:
        base_ids = self._base.encode(text)
        ids: list[int] = []
        cursor = 0
        size = len(base_ids)
        while cursor < size:
            node = 0
            scan = cursor
            matched_index = -1
            matched_end = cursor
            while scan < size:
                child = self._trie[node].get(base_ids[scan])
                if child is None:
                    break
                node = child
                scan += 1
                terminal = self._terminal[node]
                if terminal >= 0:
                    matched_index = terminal
                    matched_end = scan
            if matched_index >= 0:
                ids.append(CONCEPT_ID_BASE + matched_index)
                cursor = matched_end
            else:
                ids.append(base_ids[cursor])
                cursor += 1
        return ids

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        expanded: list[int] = []
        concept_count = len(self._concepts)
        for token_id in ids:
            offset = token_id - CONCEPT_ID_BASE
            if 0 <= offset < concept_count:
                expanded.extend(self._concept_token_ids[offset])
            else:
                expanded.append(token_id)
        return self._base.decode_bytes(expanded)

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        offset = token_id - CONCEPT_ID_BASE
        if 0 <= offset < len(self._concepts):
            return self._concepts[offset]
        return self._base.token_bytes(token_id)
