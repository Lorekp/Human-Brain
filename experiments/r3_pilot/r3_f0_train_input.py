"""Exact TRAIN-only input authority for the R3-F0 structural fixture.

Importing this module performs no filesystem or corpus I/O.  The public
capability returned by :func:`load_verified_train_input` exposes immutable,
bounded per-book token chunks and no side, path, or range selection surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterator, Protocol


ARTIFACT_PATH = Path(__file__).with_name("r3_f0_train_input_v1.json")
_ARTIFACT_MAX_BYTES = 32 * 1024
_ARTIFACT_ID = "r3-f0-train-input-v1"
_CANDIDATE_ID = "cell-lm-v1-freeze-candidate-bible-kjv-pg10-v3"
_TRAIN_INDICES = (*range(27), *range(44, 66))
_HELD_OUT_INDICES = tuple(range(27, 44))
_TRAIN_STREAM_SHA256 = (
    "f5d2ef9a096e1bb184af925d0523e3a3b80c5bdf7e821b9fca9f65c66059e7b5"
)
_TOKENIZER_SHA256 = (
    "130f42fc92e5c1f42c53d8904aeefbc4db21c189dc5b9fae98157dc06c9218a2"
)
_BOOK_LENGTHS_SHA256 = (
    "f6afe1b50deb0c94d48d2c56688fcd52eacc1c573438134d585fe00ddd7ec5f1"
)
_FIRST_200_SHA256 = (
    "de5d527b12968762d6bd614c3a77c705e053e272361a72d20f3c0a21b272ef3c"
)


class TrainInputError(RuntimeError):
    """Fail-closed TRAIN input authority or corpus mismatch."""


@dataclass(frozen=True, slots=True)
class TrainBookAuthority:
    ordinal: int
    corpus_index: int
    short_name: str
    byte_start: int
    byte_end: int
    span_bytes: int
    span_sha256: str
    token_count: int
    token_sha256_le_i32: str


@dataclass(frozen=True, slots=True)
class TrainInputAuthority:
    artifact_sha256: str
    candidate_id: str
    freeze_manifest_path: str
    freeze_manifest_max_bytes: int
    freeze_manifest_sha256: str
    span_manifest_path: str
    span_manifest_max_bytes: int
    span_manifest_sha256: str
    corpus_path: str
    corpus_size_bytes: int
    historical_full_corpus_sha256: str
    held_out_interval: tuple[int, int]
    preamble_interval: tuple[int, int]
    concept_count: int
    tokenizer_sha256: str
    tokenizer_tag: str
    verse_paragraph_split_regex: str
    verse_prefix_regex: str
    book_token_lengths_sha256: str
    first_200_comma_joined_sha256: str
    token_count: int
    train_stream_sha256: str
    books: tuple[TrainBookAuthority, ...]


@dataclass(frozen=True, slots=True)
class EncodedTrainBook:
    """One immutable, independently authenticated TRAIN token chunk."""

    ordinal: int
    corpus_index: int
    short_name: str
    token_count: int
    token_bytes_le_i32: bytes
    token_sha256_le_i32: str


@dataclass(frozen=True, slots=True)
class VerifiedTrainInput:
    """Authenticated TRAIN-only input with no corpus/path selection surface."""

    train_input_authority_sha256: str
    candidate_id: str
    freeze_manifest_sha256: str
    span_manifest_sha256: str
    train_corpus_bytes_live_verified: int
    historical_full_corpus_sha256_not_live_verified: str
    token_count: int
    train_stream_sha256: str
    tokenizer_sha256: str
    book_token_lengths_sha256: str
    first_200_comma_joined_sha256: str
    _books: tuple[EncodedTrainBook, ...]

    def iter_books(self) -> Iterator[EncodedTrainBook]:
        """Yield canonical, bounded per-book chunks without further I/O."""

        return iter(self._books)


class _PositionedCorpus(Protocol):
    size_bytes: int

    def __enter__(self) -> "_PositionedCorpus": ...

    def __exit__(self, *args: object) -> None: ...

    def read_exact_at(self, offset: int, length: int) -> bytes: ...


class _TrainInputSource(Protocol):
    def read_bounded(self, relative_path: str, maximum: int) -> bytes: ...

    def open_corpus(
        self,
        relative_path: str,
        expected_size: int,
        allowed_ranges: tuple[tuple[int, int], ...],
    ) -> _PositionedCorpus: ...


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainInputError(message)


def _expect_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label} keys drifted")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bounded_bytes(path: Path, maximum: int, label: str) -> bytes:
    try:
        with path.open("rb", buffering=0) as handle:
            raw = handle.read(maximum + 1)
    except OSError as exc:
        raise TrainInputError(f"cannot read {label}: {exc}") from exc
    _require(len(raw) <= maximum, f"{label} exceeds {maximum} bytes")
    return raw


def _path_is_reparse(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(
        reparse_flag
        and getattr(metadata, "st_file_attributes", 0) & reparse_flag
    )


def _parse_authority(value: object, *, artifact_sha256: str) -> TrainInputAuthority:
    _require(_is_sha256(artifact_sha256), "authority artifact SHA-256 is invalid")
    _require(isinstance(value, dict), "authority root must be an object")
    root = value
    _expect_keys(
        root,
        {
            "artifact_id",
            "books",
            "candidate_id",
            "claim_boundary",
            "files",
            "schema_version",
            "split",
            "stream",
            "tokenizer",
        },
        "authority",
    )
    _require(root["artifact_id"] == _ARTIFACT_ID, "authority id drifted")
    _require(root["schema_version"] == 1, "authority schema drifted")
    _require(root["candidate_id"] == _CANDIDATE_ID, "candidate id drifted")
    _require(
        root["claim_boundary"]
        == {
            "capability_evidence": False,
            "held_out_bytes_live_verified": False,
            "train_input_authority_only": True,
        },
        "claim boundary drifted",
    )

    files = root["files"]
    _require(isinstance(files, dict), "files must be an object")
    _expect_keys(files, {"freeze_manifest", "span_manifest", "trimmed_corpus"}, "files")
    freeze_file = files["freeze_manifest"]
    span_file = files["span_manifest"]
    corpus_file = files["trimmed_corpus"]
    _require(isinstance(freeze_file, dict), "freeze manifest binding must be an object")
    _require(isinstance(span_file, dict), "span manifest binding must be an object")
    _require(isinstance(corpus_file, dict), "corpus binding must be an object")
    _require(
        freeze_file
        == {
            "max_bytes": 8192,
            "relative_path": "FREEZE-CANDIDATE-v3-BIBLE.json",
            "sha256": "361c83907d9b353c8128a55ea2b4ca49a01c2aa8c8d1f6224db4661c4c9e166c",
        },
        "freeze manifest binding drifted",
    )
    _require(
        span_file
        == {
            "max_bytes": 32768,
            "relative_path": "evidence/pg10_book_spans_v3.json",
            "sha256": "5ccee9ef81e2bde85715fd4c985864a7be896ed66545370626f81349f6f994eb",
        },
        "span manifest binding drifted",
    )
    _require(
        corpus_file
        == {
            "historical_full_sha256_not_live_verified": (
                "d585e5644505b34ca516446f9b4fa2e95ee8ffa860d27858a7ca6450ee9ad3aa"
            ),
            "relative_path": "evidence/pg10.trimmed.bin",
            "size_bytes": 4436168,
        },
        "corpus binding drifted",
    )

    split = root["split"]
    _require(isinstance(split, dict), "split must be an object")
    _require(
        split
        == {
            "held_out_book_count": 17,
            "held_out_corpus_indices": list(_HELD_OUT_INDICES),
            "held_out_interval": [3247429, 4024635],
            "preamble_interval": [0, 2185],
            "train_book_count": 49,
            "train_corpus_indices": list(_TRAIN_INDICES),
            "train_span_bytes": 3656777,
        },
        "split authority drifted",
    )

    tokenizer = root["tokenizer"]
    _require(isinstance(tokenizer, dict), "tokenizer must be an object")
    _require(
        tokenizer
        == {
            "base": {"max_phrases": 0, "min_frequency": 3, "vocab_size": 256},
            "concept": {
                "concept_count": 512,
                "max_concepts": 512,
                "max_tokens": 12,
                "min_frequency": 50,
                "min_mutual_information_bits": 5.0,
                "min_tokens": 2,
            },
            "sha256": _TOKENIZER_SHA256,
            "tag": (
                "BytePhraseTokenizer(max_phrases=0,min_frequency=3)"
                "+ConceptTokenizer(freq>=50,MI>=5,cap512)"
            ),
            "verse_paragraph_split_regex": r"\r?\n\s*\r?\n",
            "verse_prefix_regex": r"^(\d+):(\d+)\s",
        },
        "tokenizer authority drifted",
    )

    stream = root["stream"]
    _require(isinstance(stream, dict), "stream must be an object")
    _require(
        stream
        == {
            "book_token_lengths_sha256": _BOOK_LENGTHS_SHA256,
            "encoding": (
                "canonical-book-order concatenation of little-endian signed int32 token ids"
            ),
            "first_200_comma_joined_sha256": _FIRST_200_SHA256,
            "sha256": _TRAIN_STREAM_SHA256,
            "token_count": 2_117_412,
        },
        "stream authority drifted",
    )

    raw_books = root["books"]
    _require(isinstance(raw_books, list), "books must be an array")
    _require(len(raw_books) == 49, "authority must contain 49 TRAIN books")
    books: list[TrainBookAuthority] = []
    previous_end = 0
    for ordinal, raw_book in enumerate(raw_books):
        _require(isinstance(raw_book, dict), f"book {ordinal} must be an object")
        _expect_keys(
            raw_book,
            {
                "byte_end",
                "byte_start",
                "corpus_index",
                "ordinal",
                "short_name",
                "span_bytes",
                "span_sha256",
                "token_count",
                "token_sha256_le_i32",
            },
            f"book {ordinal}",
        )
        _require(raw_book["ordinal"] == ordinal, f"book {ordinal} ordinal drifted")
        _require(
            raw_book["corpus_index"] == _TRAIN_INDICES[ordinal],
            f"book {ordinal} corpus index drifted",
        )
        start = raw_book["byte_start"]
        end = raw_book["byte_end"]
        length = raw_book["span_bytes"]
        _require(type(start) is int and type(end) is int, f"book {ordinal} span must be integer")
        _require(type(length) is int and length > 0, f"book {ordinal} length invalid")
        _require(end - start == length, f"book {ordinal} span length drifted")
        _require(start >= previous_end, f"book {ordinal} span order drifted")
        _require(isinstance(raw_book["short_name"], str), f"book {ordinal} name invalid")
        _require(_is_sha256(raw_book["span_sha256"]), f"book {ordinal} span hash invalid")
        _require(
            type(raw_book["token_count"]) is int and raw_book["token_count"] > 0,
            f"book {ordinal} token count invalid",
        )
        _require(
            _is_sha256(raw_book["token_sha256_le_i32"]),
            f"book {ordinal} token hash invalid",
        )
        books.append(
            TrainBookAuthority(
                ordinal=ordinal,
                corpus_index=raw_book["corpus_index"],
                short_name=raw_book["short_name"],
                byte_start=start,
                byte_end=end,
                span_bytes=length,
                span_sha256=raw_book["span_sha256"],
                token_count=raw_book["token_count"],
                token_sha256_le_i32=raw_book["token_sha256_le_i32"],
            )
        )
        previous_end = end
    _require(sum(book.span_bytes for book in books) == 3_656_777, "TRAIN bytes drifted")
    _require(sum(book.token_count for book in books) == 2_117_412, "TRAIN tokens drifted")

    return TrainInputAuthority(
        artifact_sha256=artifact_sha256,
        candidate_id=root["candidate_id"],
        freeze_manifest_path=freeze_file["relative_path"],
        freeze_manifest_max_bytes=freeze_file["max_bytes"],
        freeze_manifest_sha256=freeze_file["sha256"],
        span_manifest_path=span_file["relative_path"],
        span_manifest_max_bytes=span_file["max_bytes"],
        span_manifest_sha256=span_file["sha256"],
        corpus_path=corpus_file["relative_path"],
        corpus_size_bytes=corpus_file["size_bytes"],
        historical_full_corpus_sha256=corpus_file[
            "historical_full_sha256_not_live_verified"
        ],
        held_out_interval=tuple(split["held_out_interval"]),
        preamble_interval=tuple(split["preamble_interval"]),
        concept_count=tokenizer["concept"]["concept_count"],
        tokenizer_sha256=tokenizer["sha256"],
        tokenizer_tag=tokenizer["tag"],
        verse_paragraph_split_regex=tokenizer["verse_paragraph_split_regex"],
        verse_prefix_regex=tokenizer["verse_prefix_regex"],
        book_token_lengths_sha256=stream["book_token_lengths_sha256"],
        first_200_comma_joined_sha256=stream["first_200_comma_joined_sha256"],
        token_count=stream["token_count"],
        train_stream_sha256=stream["sha256"],
        books=tuple(books),
    )


def load_train_input_authority() -> TrainInputAuthority:
    """Fast-check and return the committed TRAIN authority without corpus I/O."""

    raw = _bounded_bytes(ARTIFACT_PATH, _ARTIFACT_MAX_BYTES, "TRAIN authority")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainInputError(f"invalid TRAIN authority JSON: {exc}") from exc
    return _parse_authority(value, artifact_sha256=hashlib.sha256(raw).hexdigest())


class _UnbufferedPositionedCorpus:
    """Private exact-range reader; it cannot service an unregistered range."""

    def __init__(
        self,
        path: Path,
        expected_size: int,
        allowed_ranges: tuple[tuple[int, int], ...],
    ) -> None:
        self._path = path
        self._expected_size = expected_size
        self._allowed_ranges = frozenset(allowed_ranges)
        self._descriptor: int | None = None
        self.size_bytes = expected_size

    def __enter__(self) -> "_UnbufferedPositionedCorpus":
        _require(self._descriptor is None, "TRAIN corpus is already open")
        try:
            before = self._path.lstat()
        except OSError as exc:
            raise TrainInputError(f"cannot stat TRAIN corpus: {exc}") from exc
        _require(
            not stat.S_ISLNK(before.st_mode) and not _path_is_reparse(before),
            "TRAIN corpus must not be a symlink or reparse point",
        )
        _require(
            stat.S_ISREG(before.st_mode),
            "TRAIN corpus path must be a regular file",
        )
        _require(
            before.st_size == self._expected_size,
            f"TRAIN corpus path size {before.st_size} != {self._expected_size}",
        )
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor: int | None = None
        try:
            descriptor = os.open(self._path, flags)
            opened = os.fstat(descriptor)
        except OSError as exc:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as close_exc:
                    raise TrainInputError(
                        f"cannot open TRAIN corpus: {exc}; descriptor cleanup failed: {close_exc}"
                    ) from close_exc
            raise TrainInputError(f"cannot open TRAIN corpus: {exc}") from exc
        try:
            _require(
                not stat.S_ISLNK(opened.st_mode) and not _path_is_reparse(opened),
                "TRAIN corpus descriptor must not be a symlink or reparse point",
            )
            _require(stat.S_ISREG(opened.st_mode), "TRAIN corpus must be a regular file")
            _require(
                opened.st_size == self._expected_size,
                f"TRAIN corpus size {opened.st_size} != {self._expected_size}",
            )
            _require(
                (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                "TRAIN corpus identity changed while opening",
            )
        except TrainInputError:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        self.size_bytes = opened.st_size
        return self

    def __exit__(self, *_args: object) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is not None:
            os.close(descriptor)

    def read_exact_at(self, offset: int, length: int) -> bytes:
        end = offset + length
        _require(
            (offset, end) in self._allowed_ranges,
            f"unauthorized TRAIN corpus range [{offset}, {end})",
        )
        descriptor = self._descriptor
        _require(descriptor is not None, "TRAIN corpus is not open")
        try:
            os.lseek(descriptor, offset, os.SEEK_SET)
            chunks: list[bytes] = []
            remaining = length
            while remaining:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    raise TrainInputError(
                        f"short TRAIN corpus read at {offset}: {length - remaining}/{length}"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
        except OSError as exc:
            raise TrainInputError(f"cannot read TRAIN corpus range: {exc}") from exc
        return b"".join(chunks)


class _FilesystemTrainInputSource:
    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, relative_path: str) -> Path:
        _require("\\" not in relative_path, "authority paths must use canonical slashes")
        parts = relative_path.split("/")
        _require(
            bool(parts) and all(part not in {"", ".", ".."} for part in parts),
            "authority path is not canonical",
        )
        return self._root.joinpath(*parts)

    def read_bounded(self, relative_path: str, maximum: int) -> bytes:
        return _bounded_bytes(self._path(relative_path), maximum, relative_path)

    def open_corpus(
        self,
        relative_path: str,
        expected_size: int,
        allowed_ranges: tuple[tuple[int, int], ...],
    ) -> _UnbufferedPositionedCorpus:
        return _UnbufferedPositionedCorpus(
            self._path(relative_path), expected_size, allowed_ranges
        )


def _default_source() -> _FilesystemTrainInputSource:
    temporary_root = os.environ.get("TEMP")
    _require(bool(temporary_root), "TEMP is required to resolve the frozen Bible root")
    return _FilesystemTrainInputSource(Path(temporary_root) / "hb_bible_freeze")


def _decode_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainInputError(f"invalid {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _metadata_train_rows(
    authority: TrainInputAuthority,
    freeze: dict[str, Any],
    spans: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    _require(freeze.get("candidate_id") == authority.candidate_id, "candidate id drifted")
    corpus = freeze.get("corpus")
    evidence = freeze.get("evidence_files")
    split = freeze.get("split")
    _require(isinstance(corpus, dict), "freeze corpus schema drifted")
    _require(isinstance(evidence, dict), "freeze evidence schema drifted")
    _require(isinstance(split, dict), "freeze split schema drifted")
    _require(
        corpus.get("sha256_trimmed") == authority.historical_full_corpus_sha256,
        "historical full-corpus identity drifted",
    )
    _require(
        corpus.get("size_trimmed_bytes") == authority.corpus_size_bytes,
        "freeze corpus size drifted",
    )
    _require(evidence.get("book_spans") == authority.span_manifest_path, "span path drifted")
    _require(evidence.get("trimmed") == authority.corpus_path, "corpus path drifted")

    raw_books = spans.get("books")
    _require(isinstance(raw_books, list), "span books must be an array")
    train_rows: list[dict[str, Any]] = []
    held_rows: list[dict[str, Any]] = []
    previous_end = 0
    for position, row in enumerate(raw_books):
        _require(isinstance(row, dict), f"span row {position} must be an object")
        role = row.get("role")
        _require(role in {"train", "val"}, f"span row {position} role invalid")
        offsets = row.get("span_offsets")
        _require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(type(value) is int for value in offsets),
            f"span row {position} offsets invalid",
        )
        start, end = offsets
        _require(
            0 <= start < end <= authority.corpus_size_bytes,
            f"span row {position} out of bounds",
        )
        _require(start >= previous_end, f"span row {position} order drifted")
        _require(row.get("bytes") == end - start, f"span row {position} length drifted")
        previous_end = end
        if role == "train":
            train_rows.append(row)
        else:
            held_rows.append(row)

    # This is the critical ordering point: the complete role partition and its
    # authority comparison finish before the caller can open the corpus.
    _require(len(train_rows) == len(authority.books), "TRAIN book count drifted")
    _require(
        len(held_rows) == split.get("val_book_count"),
        "HELD-OUT book count drifted",
    )
    _require(
        [row.get("index") for row in held_rows] == split.get("val_book_indices"),
        "HELD-OUT index roster drifted",
    )
    if held_rows:
        _require(
            (
                held_rows[0]["span_offsets"][0],
                held_rows[-1]["span_offsets"][1],
            )
            == authority.held_out_interval,
            "HELD-OUT interval drifted",
        )
    for book, row in zip(authority.books, train_rows, strict=True):
        _require(row.get("index") == book.corpus_index, f"{book.short_name} index drifted")
        _require(row.get("short_name") == book.short_name, f"{book.short_name} name drifted")
        _require(
            tuple(row["span_offsets"]) == (book.byte_start, book.byte_end),
            f"{book.short_name} span drifted",
        )
        _require(row.get("bytes") == book.span_bytes, f"{book.short_name} bytes drifted")
    return tuple(train_rows)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _verify_train_input(
    authority: TrainInputAuthority,
    source: _TrainInputSource,
) -> VerifiedTrainInput:
    freeze_raw = source.read_bounded(
        authority.freeze_manifest_path, authority.freeze_manifest_max_bytes
    )
    _require(
        hashlib.sha256(freeze_raw).hexdigest() == authority.freeze_manifest_sha256,
        "freeze manifest SHA-256 drifted",
    )
    span_raw = source.read_bounded(
        authority.span_manifest_path, authority.span_manifest_max_bytes
    )
    _require(
        hashlib.sha256(span_raw).hexdigest() == authority.span_manifest_sha256,
        "span manifest SHA-256 drifted",
    )
    freeze = _decode_json(freeze_raw, "freeze manifest JSON")
    spans = _decode_json(span_raw, "span manifest JSON")
    _metadata_train_rows(authority, freeze, spans)

    allowed_ranges = tuple(
        (book.byte_start, book.byte_end) for book in authority.books
    )
    decoded_books: list[tuple[TrainBookAuthority, str]] = []
    with source.open_corpus(
        authority.corpus_path, authority.corpus_size_bytes, allowed_ranges
    ) as corpus:
        _require(
            corpus.size_bytes == authority.corpus_size_bytes,
            "TRAIN corpus size drifted",
        )
        for book in authority.books:
            raw = corpus.read_exact_at(book.byte_start, book.span_bytes)
            _require(len(raw) == book.span_bytes, f"{book.short_name} short span read")
            _require(
                hashlib.sha256(raw).hexdigest() == book.span_sha256,
                f"{book.short_name} span SHA-256 drifted",
            )
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise TrainInputError(f"{book.short_name} is not strict UTF-8") from exc
            decoded_books.append((book, text))

    # Import only after every authorized TRAIN byte has passed its raw-span
    # check.  No experiment runner or held-out loader participates.
    import numpy as np

    from humanbrain.concept_tokens import (
        ConceptConfig,
        ConceptTokenizer,
        mine_concepts_detailed,
    )
    from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

    texts = [text for _, text in decoded_books]
    base = BytePhraseTokenizer.fit(
        texts, TokenizerConfig(max_phrases=0, min_frequency=3)
    )
    _require(base.vocab_size == 256, "base tokenizer vocabulary drifted")
    candidates = mine_concepts_detailed(
        texts,
        base,
        ConceptConfig(
            min_frequency=50,
            min_mutual_information_bits=5.0,
            max_concepts=512,
            min_tokens=2,
            max_tokens=12,
        ),
    )
    _require(
        len(candidates) == authority.concept_count,
        "TRAIN concept count drifted",
    )
    stack = ConceptTokenizer(base, tuple(candidate.payload for candidate in candidates))
    tokenizer_payload = {
        "concepts_hex": [payload.hex() for payload in stack.concepts],
        "tag": authority.tokenizer_tag,
    }
    _require(
        hashlib.sha256(_canonical_json_bytes(tokenizer_payload)).hexdigest()
        == authority.tokenizer_sha256,
        "TRAIN tokenizer SHA-256 drifted",
    )

    paragraph_re = re.compile(authority.verse_paragraph_split_regex)
    verse_re = re.compile(authority.verse_prefix_regex)
    stream_hasher = hashlib.sha256()
    book_lengths: list[tuple[str, int]] = []
    first_ids: list[int] = []
    encoded_books: list[EncodedTrainBook] = []
    for book, text in decoded_books:
        ids: list[int] = []
        paragraphs = paragraph_re.split(text)
        for paragraph in paragraphs:
            verse = paragraph.strip()
            if verse_re.match(verse):
                ids.extend(int(token) for token in stack.encode(verse))
        token_bytes = np.asarray(ids, dtype="<i4").tobytes()
        token_sha = hashlib.sha256(token_bytes).hexdigest()
        _require(len(ids) == book.token_count, f"{book.short_name} token count drifted")
        _require(
            token_sha == book.token_sha256_le_i32,
            f"{book.short_name} token SHA-256 drifted",
        )
        stream_hasher.update(token_bytes)
        book_lengths.append((book.short_name, len(ids)))
        if len(first_ids) < 200:
            first_ids.extend(ids[: 200 - len(first_ids)])
        encoded_books.append(
            EncodedTrainBook(
                ordinal=book.ordinal,
                corpus_index=book.corpus_index,
                short_name=book.short_name,
                token_count=len(ids),
                token_bytes_le_i32=token_bytes,
                token_sha256_le_i32=token_sha,
            )
        )

    token_count = sum(book.token_count for book in encoded_books)
    _require(token_count == authority.token_count, "TRAIN token count drifted")
    stream_sha = stream_hasher.hexdigest()
    _require(stream_sha == authority.train_stream_sha256, "TRAIN stream SHA-256 drifted")
    book_lengths_sha = hashlib.sha256(_canonical_json_bytes(book_lengths)).hexdigest()
    _require(
        book_lengths_sha == authority.book_token_lengths_sha256,
        "TRAIN book-token-length SHA-256 drifted",
    )
    first_200_sha = hashlib.sha256(
        ",".join(map(str, first_ids)).encode("utf-8")
    ).hexdigest()
    _require(
        first_200_sha == authority.first_200_comma_joined_sha256,
        "TRAIN first-200 SHA-256 drifted",
    )
    return VerifiedTrainInput(
        train_input_authority_sha256=authority.artifact_sha256,
        candidate_id=authority.candidate_id,
        freeze_manifest_sha256=authority.freeze_manifest_sha256,
        span_manifest_sha256=authority.span_manifest_sha256,
        train_corpus_bytes_live_verified=sum(
            book.span_bytes for book in authority.books
        ),
        historical_full_corpus_sha256_not_live_verified=(
            authority.historical_full_corpus_sha256
        ),
        token_count=token_count,
        train_stream_sha256=stream_sha,
        tokenizer_sha256=authority.tokenizer_sha256,
        book_token_lengths_sha256=book_lengths_sha,
        first_200_comma_joined_sha256=first_200_sha,
        _books=tuple(encoded_books),
    )


def load_verified_train_input() -> VerifiedTrainInput:
    """Authenticate and materialize only the frozen canonical TRAIN stream."""

    authority = load_train_input_authority()
    source = _default_source()
    return _verify_train_input(authority, source)


__all__ = [
    "TrainBookAuthority",
    "TrainInputAuthority",
    "TrainInputError",
    "EncodedTrainBook",
    "VerifiedTrainInput",
    "load_train_input_authority",
    "load_verified_train_input",
]
