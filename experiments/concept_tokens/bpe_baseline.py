"""Plain byte-level BPE baseline (stdlib-only) for the concept-token study.

GPT-2-style: pretokenize into the same piece shapes BytePhraseTokenizer uses,
train byte-pair merges over weighted unique pieces, encode by applying merges
lowest-rank-first. Vocab layout matches BytePhraseTokenizer conventions:
ids 0..255 are raw UTF-8 bytes, ids 256.. carry merged symbols.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

_PIECE = re.compile(r"[A-Za-z_][A-Za-z_0-9]{1,47}|[0-9]{2,24}|[ \t]{2,16}|\r?\n|[^\w\s]", re.UNICODE)


def _chunks(text: str) -> Iterable[str]:
    cursor = 0
    for match in _PIECE.finditer(text):
        if match.start() > cursor:
            # Unmatched characters (e.g. lone ASCII spaces) fall back to bytes.
            for char in text[cursor : match.start()]:
                yield char
        yield match.group(0)
        cursor = match.end()
    for char in text[cursor:]:
        yield char


class BpeTokenizer:
    def __init__(self, merges: list[tuple[int, int]]) -> None:
        self._merges = merges
        self._rank = {pair: rank for rank, pair in enumerate(merges)}
        self._id_to_bytes: list[bytes] = [bytes([i]) for i in range(256)]
        self._pair_to_id: dict[tuple[int, int], int] = {}
        for left, right in merges:
            new_id = len(self._id_to_bytes)
            self._pair_to_id[(left, right)] = new_id
            self._id_to_bytes.append(self._id_to_bytes[left] + self._id_to_bytes[right])
        self._chunk_cache: dict[tuple[int, ...], list[int]] = {}

    @classmethod
    def fit(cls, texts: Iterable[str], vocab_size: int = 4096) -> "BpeTokenizer":
        chunk_counts: Counter[str] = Counter()
        for text in texts:
            chunk_counts.update(_chunks(text))
        symbols: dict[str, tuple[int, ...]] = {
            chunk: tuple(chunk.encode("utf-8")) for chunk in chunk_counts
        }
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_to_chunks: dict[tuple[int, int], set[str]] = {}

        def tally(chunk: str, weight: int) -> None:
            seq = symbols[chunk]
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += weight
                pair_to_chunks.setdefault(pair, set()).add(chunk)

        for chunk, weight in chunk_counts.items():
            tally(chunk, weight)

        merges: list[tuple[int, int]] = []
        target_merges = max(0, vocab_size - 256)
        while len(merges) < target_merges and pair_counts:
            best_pair = max(
                (pair for pair, count in pair_counts.items() if count >= 2),
                key=lambda pair: (pair_counts[pair], -pair[0], -pair[1]),
                default=None,
            )
            if best_pair is None:
                break
            merges.append(best_pair)
            new_id = 256 + len(merges) - 1
            affected = pair_to_chunks.get(best_pair, set())
            recompute: set[str] = set()
            for chunk in affected:
                seq = symbols[chunk]
                recompute.add(chunk)
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_counts[pair] -= chunk_counts[chunk]
                merged: list[int] = []
                i = 0
                while i < len(seq):
                    if (
                        i + 1 < len(seq)
                        and seq[i] == best_pair[0]
                        and seq[i + 1] == best_pair[1]
                    ):
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(seq[i])
                        i += 1
                symbols[chunk] = tuple(merged)
            for chunk in recompute:
                tally(chunk, chunk_counts[chunk])
            pair_counts = Counter({pair: c for pair, c in pair_counts.items() if c >= 2})
            pair_to_chunks.pop(best_pair, None)
        return cls(merges)

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_bytes)

    @property
    def num_merges(self) -> int:
        return len(self._merges)

    def _encode_chunk(self, seq: tuple[int, ...]) -> list[int]:
        cached = self._chunk_cache.get(seq)
        if cached is not None:
            return cached
        work = list(seq)
        while len(work) >= 2:
            best_rank = None
            best_pos = -1
            for i in range(len(work) - 1):
                rank = self._rank.get((work[i], work[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pos = i
            if best_rank is None:
                break
            work[best_pos : best_pos + 2] = [
                self._pair_to_id[(work[best_pos], work[best_pos + 1])]
            ]
        self._chunk_cache[seq] = work
        return work

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _chunks(text):
            ids.extend(self._encode_chunk(tuple(chunk.encode("utf-8"))))
        return ids

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(
            self._id_to_bytes[token_id]
            for token_id in ids
            if 0 <= token_id < len(self._id_to_bytes)
        )

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")
