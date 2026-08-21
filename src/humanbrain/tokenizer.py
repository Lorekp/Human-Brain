"""A compact byte tokenizer with corpus-grown phrase tokens."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable


_PIECE = re.compile(r"[A-Za-z_][A-Za-z_0-9]{1,47}|[0-9]{2,24}|[ \t]{2,16}|\r?\n|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class TokenizerConfig:
    max_phrases: int = 512
    min_frequency: int = 2


class BytePhraseTokenizer:
    """Lossless UTF-8 bytes plus frequent, greedily matched phrase tokens.

    IDs 0..255 are bytes. Phrase IDs are stable for a fitted corpus because
    candidates are ordered by frequency, byte length, then byte value.
    """

    def __init__(self, phrases: Iterable[bytes] = ()) -> None:
        unique = sorted(
            {bytes(piece) for piece in phrases if len(piece) > 1},
            key=lambda piece: (-len(piece), piece),
        )
        self._id_to_bytes = [bytes([i]) for i in range(256)] + unique
        self._trie: list[dict[int, int]] = [{}]
        self._terminal: list[int] = [-1]
        for token_id, piece in enumerate(unique, start=256):
            node = 0
            for value in piece:
                child = self._trie[node].get(value)
                if child is None:
                    child = len(self._trie)
                    self._trie[node][value] = child
                    self._trie.append({})
                    self._terminal.append(-1)
                node = child
            self._terminal[node] = token_id

    @classmethod
    def fit(
        cls,
        texts: Iterable[str],
        config: TokenizerConfig | None = None,
    ) -> "BytePhraseTokenizer":
        cfg = config or TokenizerConfig()
        counts: Counter[bytes] = Counter()
        for text in texts:
            for match in _PIECE.finditer(text):
                piece = match.group(0).encode("utf-8")
                if 1 < len(piece) <= 48:
                    counts[piece] += 1
        candidates = [
            (count, piece)
            for piece, count in counts.items()
            if count >= cfg.min_frequency
        ]
        candidates.sort(key=lambda item: (-item[0], -len(item[1]), item[1]))
        return cls(piece for _, piece in candidates[: cfg.max_phrases])

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_bytes)

    @property
    def phrases(self) -> tuple[bytes, ...]:
        return tuple(self._id_to_bytes[256:])

    def encode(self, text: str) -> list[int]:
        raw = text.encode("utf-8")
        ids: list[int] = []
        cursor = 0
        while cursor < len(raw):
            node = 0
            scan = cursor
            matched_id = -1
            matched_end = cursor
            while scan < len(raw):
                child = self._trie[node].get(raw[scan])
                if child is None:
                    break
                node = child
                scan += 1
                terminal = self._terminal[node]
                if terminal >= 0:
                    matched_id = terminal
                    matched_end = scan
            if matched_id >= 0:
                ids.append(matched_id)
                cursor = matched_end
            else:
                ids.append(raw[cursor])
                cursor += 1
        return ids

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        pieces: list[bytes] = []
        size = len(self._id_to_bytes)
        for token_id in ids:
            if 0 <= token_id < size:
                pieces.append(self._id_to_bytes[token_id])
        return b"".join(pieces)

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        if 0 <= token_id < len(self._id_to_bytes):
            return self._id_to_bytes[token_id]
        raise IndexError(f"token id {token_id} outside vocabulary")

