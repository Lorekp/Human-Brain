from __future__ import annotations

import builtins
from dataclasses import replace
import hashlib
import inspect
import json
import io
import mmap
import os
import stat
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.r3_pilot import r3_f0_train_input as train_input
from experiments.r3_pilot.r3_f0_train_input import (
    TrainBookAuthority,
    TrainInputAuthority,
    TrainInputError,
    load_train_input_authority,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "experiments" / "r3_pilot" / "r3_f0_train_input_v1.json"


def test_train_input_module_import_performs_no_data_io() -> None:
    script = r"""
from pathlib import Path
import builtins
import mmap
import os

original_open = Path.open
original_builtin_open = builtins.open
original_os_open = os.open
original_mmap = mmap.mmap

def is_data_path(path):
    try:
        return Path(path).suffix in {'.json', '.bin'}
    except TypeError:
        return False

def guarded_open(path, *args, **kwargs):
    if is_data_path(path):
        raise AssertionError(f'import touched data: {path}')
    return original_open(path, *args, **kwargs)

def guarded_builtin_open(path, *args, **kwargs):
    if is_data_path(path):
        raise AssertionError(f'import touched data: {path}')
    return original_builtin_open(path, *args, **kwargs)

def guarded_os_open(path, *args, **kwargs):
    if is_data_path(path):
        raise AssertionError(f'import touched data: {path}')
    return original_os_open(path, *args, **kwargs)

def guarded_mmap(*args, **kwargs):
    raise AssertionError('import created an mmap')

Path.open = guarded_open
builtins.open = guarded_builtin_open
os.open = guarded_os_open
mmap.mmap = guarded_mmap
import experiments.r3_pilot.r3_f0_train_input
print('IMPORT_OK')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "IMPORT_OK"


def test_committed_train_input_authority_freezes_exact_production_identity() -> None:
    raw = ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "29304cf71cce83ab34ee8253117043aec8413806ba0047ff850bb3c93a31d966"
    )
    artifact = json.loads(raw.decode("utf-8"))

    assert artifact["artifact_id"] == "r3-f0-train-input-v1"
    assert artifact["schema_version"] == 1
    assert artifact["candidate_id"] == (
        "cell-lm-v1-freeze-candidate-bible-kjv-pg10-v3"
    )
    assert artifact["files"] == {
        "freeze_manifest": {
            "max_bytes": 8192,
            "relative_path": "FREEZE-CANDIDATE-v3-BIBLE.json",
            "sha256": (
                "361c83907d9b353c8128a55ea2b4ca49a01c2aa8c8d1f6224db4661c4c9e166c"
            ),
        },
        "span_manifest": {
            "max_bytes": 32768,
            "relative_path": "evidence/pg10_book_spans_v3.json",
            "sha256": (
                "5ccee9ef81e2bde85715fd4c985864a7be896ed66545370626f81349f6f994eb"
            ),
        },
        "trimmed_corpus": {
            "historical_full_sha256_not_live_verified": (
                "d585e5644505b34ca516446f9b4fa2e95ee8ffa860d27858a7ca6450ee9ad3aa"
            ),
            "relative_path": "evidence/pg10.trimmed.bin",
            "size_bytes": 4436168,
        },
    }
    assert artifact["split"]["train_book_count"] == 49
    assert artifact["split"]["held_out_book_count"] == 17
    assert artifact["split"]["train_span_bytes"] == 3656777
    assert artifact["split"]["held_out_interval"] == [3247429, 4024635]
    assert artifact["stream"] == {
        "book_token_lengths_sha256": (
            "f6afe1b50deb0c94d48d2c56688fcd52eacc1c573438134d585fe00ddd7ec5f1"
        ),
        "encoding": "canonical-book-order concatenation of little-endian signed int32 token ids",
        "first_200_comma_joined_sha256": (
            "de5d527b12968762d6bd614c3a77c705e053e272361a72d20f3c0a21b272ef3c"
        ),
        "sha256": (
            "f5d2ef9a096e1bb184af925d0523e3a3b80c5bdf7e821b9fca9f65c66059e7b5"
        ),
        "token_count": 2117412,
    }
    assert len(artifact["books"]) == 49
    assert [row["corpus_index"] for row in artifact["books"]] == [
        *range(27),
        *range(44, 66),
    ]
    assert all(len(row["span_sha256"]) == 64 for row in artifact["books"])
    assert all(len(row["token_sha256_le_i32"]) == 64 for row in artifact["books"])


def test_fast_authority_check_reads_only_the_bounded_committed_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = ARTIFACT.read_bytes()
    read_sizes: list[int] = []
    original_open = Path.open

    class RecordingBytes(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return super().read(size)

    def guarded_open(path: Path, *args: object, **kwargs: object) -> object:
        if path == ARTIFACT:
            mode = args[0] if args else kwargs.pop("mode", None)
            assert mode == "rb"
            assert kwargs.get("buffering") == 0
            return RecordingBytes(raw)
        if path.suffix in {".json", ".bin"}:
            raise AssertionError(f"fast check touched external data: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    authority = load_train_input_authority()

    assert authority.artifact_sha256 == hashlib.sha256(raw).hexdigest()
    assert authority.token_count == 2_117_412
    assert authority.train_stream_sha256 == (
        "f5d2ef9a096e1bb184af925d0523e3a3b80c5bdf7e821b9fca9f65c66059e7b5"
    )
    assert len(authority.books) == 49
    assert read_sizes == [32 * 1024 + 1]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _synthetic_source_fixture() -> tuple[
    TrainInputAuthority, "_MemorySource", tuple[bytes, bytes], tuple[tuple[int, int], ...]
]:
    train_texts = (b"1:1 Alpha\n\n1:2 Beta", b"1:1 Gamma")
    held_text = b"1:1 SECRET-HELD-OUT"
    preamble = b"HDR"
    first_start = len(preamble)
    first_end = first_start + len(train_texts[0])
    held_start = first_end
    held_end = held_start + len(held_text)
    second_start = held_end
    second_end = second_start + len(train_texts[1])
    binary = preamble + train_texts[0] + held_text + train_texts[1]

    span_value = {
        "books": [
            {
                "bytes": len(train_texts[0]),
                "index": 0,
                "role": "train",
                "short_name": "Train One",
                "span_offsets": [first_start, first_end],
            },
            {
                "bytes": len(held_text),
                "index": 1,
                "role": "val",
                "short_name": "Held One",
                "span_offsets": [held_start, held_end],
            },
            {
                "bytes": len(train_texts[1]),
                "index": 2,
                "role": "train",
                "short_name": "Train Two",
                "span_offsets": [second_start, second_end],
            },
        ]
    }
    freeze_value = {
        "candidate_id": "synthetic-train-only",
        "corpus": {
            "sha256_trimmed": hashlib.sha256(binary).hexdigest(),
            "size_trimmed_bytes": len(binary),
        },
        "evidence_files": {
            "book_spans": "evidence/pg10_book_spans_v3.json",
            "trimmed": "evidence/pg10.trimmed.bin",
        },
        "split": {
            "train_book_count": 2,
            "val_book_count": 1,
            "val_book_indices": [1],
            "val_books_short": ["Held One"],
        },
    }
    freeze_raw = json.dumps(freeze_value, sort_keys=True).encode("utf-8")
    span_raw = json.dumps(span_value, sort_keys=True).encode("utf-8")

    # With fewer than 50 recurrences the frozen concept miner selects no
    # concepts, so the exact token stream is the UTF-8 bytes of each stripped
    # verse, with paragraph separators removed at verse boundaries.
    token_ids = (
        tuple(b"1:1 Alpha1:2 Beta"),
        tuple(b"1:1 Gamma"),
    )
    token_payloads = tuple(
        b"".join(struct.pack("<i", token_id) for token_id in ids)
        for ids in token_ids
    )
    tag = (
        "BytePhraseTokenizer(max_phrases=0,min_frequency=3)"
        "+ConceptTokenizer(freq>=50,MI>=5,cap512)"
    )
    tokenizer_sha = hashlib.sha256(
        _canonical_json_bytes({"concepts_hex": [], "tag": tag})
    ).hexdigest()
    book_lengths = [("Train One", len(token_ids[0])), ("Train Two", len(token_ids[1]))]
    first_ids = (*token_ids[0], *token_ids[1])[:200]
    stream_payload = b"".join(token_payloads)
    authorities = (
        TrainBookAuthority(
            ordinal=0,
            corpus_index=0,
            short_name="Train One",
            byte_start=first_start,
            byte_end=first_end,
            span_bytes=len(train_texts[0]),
            span_sha256=hashlib.sha256(train_texts[0]).hexdigest(),
            token_count=len(token_ids[0]),
            token_sha256_le_i32=hashlib.sha256(token_payloads[0]).hexdigest(),
        ),
        TrainBookAuthority(
            ordinal=1,
            corpus_index=2,
            short_name="Train Two",
            byte_start=second_start,
            byte_end=second_end,
            span_bytes=len(train_texts[1]),
            span_sha256=hashlib.sha256(train_texts[1]).hexdigest(),
            token_count=len(token_ids[1]),
            token_sha256_le_i32=hashlib.sha256(token_payloads[1]).hexdigest(),
        ),
    )
    authority = TrainInputAuthority(
        artifact_sha256="0" * 64,
        candidate_id="synthetic-train-only",
        freeze_manifest_path="FREEZE-CANDIDATE-v3-BIBLE.json",
        freeze_manifest_max_bytes=8192,
        freeze_manifest_sha256=hashlib.sha256(freeze_raw).hexdigest(),
        span_manifest_path="evidence/pg10_book_spans_v3.json",
        span_manifest_max_bytes=32768,
        span_manifest_sha256=hashlib.sha256(span_raw).hexdigest(),
        corpus_path="evidence/pg10.trimmed.bin",
        corpus_size_bytes=len(binary),
        historical_full_corpus_sha256=hashlib.sha256(binary).hexdigest(),
        held_out_interval=(held_start, held_end),
        preamble_interval=(0, first_start),
        concept_count=0,
        tokenizer_sha256=tokenizer_sha,
        tokenizer_tag=tag,
        verse_paragraph_split_regex=r"\r?\n\s*\r?\n",
        verse_prefix_regex=r"^(\d+):(\d+)\s",
        book_token_lengths_sha256=hashlib.sha256(
            _canonical_json_bytes(book_lengths)
        ).hexdigest(),
        first_200_comma_joined_sha256=hashlib.sha256(
            ",".join(map(str, first_ids)).encode("utf-8")
        ).hexdigest(),
        token_count=sum(map(len, token_ids)),
        train_stream_sha256=hashlib.sha256(stream_payload).hexdigest(),
        books=authorities,
    )
    source = _MemorySource(
        metadata={
            authority.freeze_manifest_path: freeze_raw,
            authority.span_manifest_path: span_raw,
        },
        binary=binary,
        held_out_interval=authority.held_out_interval,
    )
    allowed = tuple((book.byte_start, book.byte_end) for book in authorities)
    return authority, source, token_payloads, allowed


class _MemoryReader:
    def __init__(
        self,
        source: "_MemorySource",
        allowed_ranges: tuple[tuple[int, int], ...],
    ) -> None:
        self._source = source
        self._allowed_ranges = allowed_ranges

    def __enter__(self) -> "_MemoryReader":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    @property
    def size_bytes(self) -> int:
        return len(self._source.binary)

    def read_exact_at(self, offset: int, length: int) -> bytes:
        end = offset + length
        held_start, held_end = self._source.held_out_interval
        if offset < held_end and end > held_start:
            raise AssertionError("HELD-OUT range was read")
        if (offset, end) not in self._allowed_ranges:
            raise AssertionError(f"unauthorized range {(offset, end)}")
        self._source.events.append(("range", offset, end))
        return self._source.binary[offset:end]


class _MemorySource:
    def __init__(
        self,
        *,
        metadata: dict[str, bytes],
        binary: bytes,
        held_out_interval: tuple[int, int],
    ) -> None:
        self.metadata = metadata
        self.binary = binary
        self.held_out_interval = held_out_interval
        self.events: list[tuple[Any, ...]] = []

    def read_bounded(self, relative_path: str, maximum: int) -> bytes:
        self.events.append(("metadata", relative_path))
        raw = self.metadata[relative_path]
        if len(raw) > maximum:
            raise AssertionError("metadata bound exceeded")
        return raw

    def open_corpus(
        self,
        relative_path: str,
        expected_size: int,
        allowed_ranges: tuple[tuple[int, int], ...],
    ) -> _MemoryReader:
        assert relative_path == "evidence/pg10.trimmed.bin"
        assert expected_size == len(self.binary)
        self.events.append(("open", relative_path, allowed_ranges))
        return _MemoryReader(self, allowed_ranges)


def _install_synthetic_input(
    monkeypatch: pytest.MonkeyPatch,
    authority: TrainInputAuthority,
    source: _MemorySource,
) -> None:
    monkeypatch.setattr(train_input, "load_train_input_authority", lambda: authority)
    monkeypatch.setattr(train_input, "_default_source", lambda: source)


def test_verified_input_filters_metadata_before_open_and_reads_only_train_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, source, expected_payloads, allowed = _synthetic_source_fixture()
    _install_synthetic_input(monkeypatch, authority, source)

    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.endswith("run_c1_cohort") or name.endswith("run_gs0_health"):
            raise AssertionError(f"old runner imported: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(Path, "read_bytes", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("whole-file Path.read_bytes is forbidden")
    ))
    monkeypatch.setattr(
        hashlib,
        "file_digest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("whole-file hashlib.file_digest is forbidden")
        ),
    )

    verified = train_input.load_verified_train_input()
    chunks = tuple(verified.iter_books())

    assert verified.token_count == authority.token_count
    assert verified.train_input_authority_sha256 == authority.artifact_sha256
    assert verified.freeze_manifest_sha256 == authority.freeze_manifest_sha256
    assert verified.span_manifest_sha256 == authority.span_manifest_sha256
    assert verified.train_corpus_bytes_live_verified == sum(
        book.span_bytes for book in authority.books
    )
    assert (
        verified.historical_full_corpus_sha256_not_live_verified
        == authority.historical_full_corpus_sha256
    )
    assert verified.train_stream_sha256 == authority.train_stream_sha256
    assert tuple(chunk.token_bytes_le_i32 for chunk in chunks) == expected_payloads
    assert tuple((chunk.ordinal, chunk.corpus_index) for chunk in chunks) == ((0, 0), (1, 2))
    assert source.events[:3] == [
        ("metadata", authority.freeze_manifest_path),
        ("metadata", authority.span_manifest_path),
        ("open", authority.corpus_path, allowed),
    ]
    assert [event for event in source.events if event[0] == "range"] == [
        ("range", *allowed[0]),
        ("range", *allowed[1]),
    ]


def test_held_out_only_byte_changes_are_unobservable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, source_a, _expected_payloads, _allowed = _synthetic_source_fixture()
    held_start, held_end = authority.held_out_interval
    changed = bytearray(source_a.binary)
    changed[held_start:held_end] = b"X" * (held_end - held_start)
    source_b = _MemorySource(
        metadata=dict(source_a.metadata),
        binary=bytes(changed),
        held_out_interval=authority.held_out_interval,
    )

    _install_synthetic_input(monkeypatch, authority, source_a)
    first = train_input.load_verified_train_input()
    monkeypatch.setattr(train_input, "_default_source", lambda: source_b)
    second = train_input.load_verified_train_input()

    assert tuple(first.iter_books()) == tuple(second.iter_books())
    assert all(
        event[0] != "range"
        or (event[1], event[2]) != authority.held_out_interval
        for event in source_a.events
    )
    assert all(
        event[0] != "range"
        or (event[1], event[2]) != authority.held_out_interval
        for event in source_b.events
    )


def test_preamble_only_byte_changes_are_unobservable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, source_a, _expected_payloads, _allowed = _synthetic_source_fixture()
    preamble_start, preamble_end = authority.preamble_interval
    changed = bytearray(source_a.binary)
    changed[preamble_start:preamble_end] = b"X" * (preamble_end - preamble_start)
    source_b = _MemorySource(
        metadata=dict(source_a.metadata),
        binary=bytes(changed),
        held_out_interval=authority.held_out_interval,
    )

    _install_synthetic_input(monkeypatch, authority, source_a)
    first = train_input.load_verified_train_input()
    monkeypatch.setattr(train_input, "_default_source", lambda: source_b)
    second = train_input.load_verified_train_input()

    assert tuple(first.iter_books()) == tuple(second.iter_books())
    assert all(
        event[0] != "range"
        or (event[1], event[2]) != authority.preamble_interval
        for event in (*source_a.events, *source_b.events)
    )


def test_authority_range_drift_stops_before_corpus_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, source, _expected_payloads, _allowed = _synthetic_source_fixture()
    held_start, held_end = authority.held_out_interval
    drifted_first = replace(
        authority.books[0],
        byte_start=held_start,
        byte_end=held_end,
        span_bytes=held_end - held_start,
    )
    drifted = replace(
        authority,
        books=(drifted_first, *authority.books[1:]),
    )
    _install_synthetic_input(monkeypatch, drifted, source)

    with pytest.raises(TrainInputError, match="Train One span drifted"):
        train_input.load_verified_train_input()

    assert source.events == [
        ("metadata", authority.freeze_manifest_path),
        ("metadata", authority.span_manifest_path),
    ]


def test_train_span_corruption_stops_before_tokenizer_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, source, _expected_payloads, _allowed = _synthetic_source_fixture()
    changed = bytearray(source.binary)
    changed[authority.books[0].byte_start] ^= 1
    source.binary = bytes(changed)
    _install_synthetic_input(monkeypatch, authority, source)

    with pytest.raises(TrainInputError, match="Train One span SHA-256 drifted"):
        train_input.load_verified_train_input()

    assert [event for event in source.events if event[0] == "range"] == [
        ("range", authority.books[0].byte_start, authority.books[0].byte_end)
    ]


def test_role_drift_is_rejected_before_the_corpus_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, source, _expected_payloads, _allowed = _synthetic_source_fixture()
    span_value = json.loads(source.metadata[authority.span_manifest_path])
    span_value["books"][1]["role"] = "train"
    span_raw = json.dumps(span_value, sort_keys=True).encode("utf-8")
    source.metadata[authority.span_manifest_path] = span_raw
    authority = replace(
        authority,
        span_manifest_sha256=hashlib.sha256(span_raw).hexdigest(),
    )
    _install_synthetic_input(monkeypatch, authority, source)

    with pytest.raises(TrainInputError, match="TRAIN book count drifted"):
        train_input.load_verified_train_input()

    assert source.events == [
        ("metadata", authority.freeze_manifest_path),
        ("metadata", authority.span_manifest_path),
    ]


def test_filesystem_source_uses_only_unbuffered_positioned_train_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authority, source, expected_payloads, allowed = _synthetic_source_fixture()
    freeze_root = tmp_path / "hb_bible_freeze"
    corpus_path = freeze_root / "evidence" / "pg10.trimmed.bin"
    corpus_path.parent.mkdir(parents=True)
    (freeze_root / authority.freeze_manifest_path).write_bytes(
        source.metadata[authority.freeze_manifest_path]
    )
    (freeze_root / authority.span_manifest_path).write_bytes(
        source.metadata[authority.span_manifest_path]
    )
    corpus_path.write_bytes(source.binary)

    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setattr(train_input, "load_train_input_authority", lambda: authority)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("whole-file Path.read_bytes is forbidden")
        ),
    )
    monkeypatch.setattr(
        hashlib,
        "file_digest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("whole-file hashlib.file_digest is forbidden")
        ),
    )

    original_open = os.open
    original_lseek = os.lseek
    original_read = os.read
    corpus_descriptors: set[int] = set()
    operations: list[tuple[object, ...]] = []

    def recording_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        descriptor = original_open(path, flags, *args, **kwargs)
        if Path(path) == corpus_path:
            corpus_descriptors.add(descriptor)
            operations.append(("open", Path(path), flags))
        return descriptor

    def recording_lseek(descriptor: int, offset: int, whence: int) -> int:
        if descriptor in corpus_descriptors:
            operations.append(("seek", offset, whence))
        return original_lseek(descriptor, offset, whence)

    def recording_read(descriptor: int, length: int) -> bytes:
        if descriptor in corpus_descriptors:
            assert length != authority.corpus_size_bytes
            operations.append(("read", length))
        return original_read(descriptor, length)

    monkeypatch.setattr(train_input.os, "open", recording_open)
    monkeypatch.setattr(train_input.os, "lseek", recording_lseek)
    monkeypatch.setattr(train_input.os, "read", recording_read)

    verified = train_input.load_verified_train_input()

    assert tuple(book.token_bytes_le_i32 for book in verified.iter_books()) == (
        expected_payloads
    )
    assert operations[0][:2] == ("open", corpus_path)
    assert operations[1:] == [
        ("seek", allowed[0][0], os.SEEK_SET),
        ("read", allowed[0][1] - allowed[0][0]),
        ("seek", allowed[1][0], os.SEEK_SET),
        ("read", allowed[1][1] - allowed[1][0]),
    ]


def _fake_file_stat(
    *,
    size: int = 10,
    device: int = 1,
    inode: int = 2,
    mode: int = stat.S_IFREG,
    reparse: bool = False,
) -> SimpleNamespace:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return SimpleNamespace(
        st_dev=device,
        st_file_attributes=reparse_flag if reparse else 0,
        st_ino=inode,
        st_mode=mode,
        st_size=size,
    )


def test_filesystem_reader_closes_descriptor_when_fstat_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = train_input._UnbufferedPositionedCorpus(
        Path("corpus.bin"), 10, ((0, 10),)
    )
    closed: list[int] = []
    monkeypatch.setattr(Path, "lstat", lambda _path: _fake_file_stat())
    monkeypatch.setattr(train_input.os, "open", lambda *_args: 71)
    monkeypatch.setattr(
        train_input.os,
        "fstat",
        lambda _descriptor: (_ for _ in ()).throw(OSError("fstat failed")),
    )
    monkeypatch.setattr(train_input.os, "close", closed.append)

    with pytest.raises(TrainInputError, match="cannot open TRAIN corpus"):
        reader.__enter__()

    assert closed == [71]


@pytest.mark.parametrize(
    ("opened", "message"),
    [
        (_fake_file_stat(mode=stat.S_IFDIR), "regular file"),
        (_fake_file_stat(size=11), "size 11 != 10"),
        (_fake_file_stat(inode=3), "identity changed"),
    ],
)
def test_filesystem_reader_closes_descriptor_on_validation_rejection(
    monkeypatch: pytest.MonkeyPatch,
    opened: SimpleNamespace,
    message: str,
) -> None:
    reader = train_input._UnbufferedPositionedCorpus(
        Path("corpus.bin"), 10, ((0, 10),)
    )
    closed: list[int] = []
    monkeypatch.setattr(Path, "lstat", lambda _path: _fake_file_stat())
    monkeypatch.setattr(train_input.os, "open", lambda *_args: 72)
    monkeypatch.setattr(train_input.os, "fstat", lambda _descriptor: opened)
    monkeypatch.setattr(train_input.os, "close", closed.append)

    with pytest.raises(TrainInputError, match=message):
        reader.__enter__()

    assert closed == [72]


def test_filesystem_reader_rejects_path_and_descriptor_reparse_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = train_input._UnbufferedPositionedCorpus(
        Path("corpus.bin"), 10, ((0, 10),)
    )
    opens: list[object] = []
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda _path: _fake_file_stat(reparse=True),
    )
    monkeypatch.setattr(train_input.os, "open", lambda *args: opens.append(args))

    with pytest.raises(TrainInputError, match="symlink or reparse point"):
        reader.__enter__()

    assert opens == []

    closed: list[int] = []
    monkeypatch.setattr(Path, "lstat", lambda _path: _fake_file_stat())
    monkeypatch.setattr(train_input.os, "open", lambda *_args: 73)
    monkeypatch.setattr(
        train_input.os,
        "fstat",
        lambda _descriptor: _fake_file_stat(reparse=True),
    )
    monkeypatch.setattr(train_input.os, "close", closed.append)

    with pytest.raises(TrainInputError, match="descriptor.*reparse point"):
        reader.__enter__()

    assert closed == [73]


def test_filesystem_reader_closes_after_positioned_read_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "corpus.bin"
    corpus_path.write_bytes(b"0123456789")
    original_close = os.close
    closed: list[int] = []

    def recording_close(descriptor: int) -> None:
        closed.append(descriptor)
        original_close(descriptor)

    monkeypatch.setattr(
        train_input.os,
        "read",
        lambda *_args: (_ for _ in ()).throw(OSError("read failed")),
    )
    monkeypatch.setattr(train_input.os, "close", recording_close)
    reader = train_input._UnbufferedPositionedCorpus(
        corpus_path, 10, ((0, 10),)
    )

    with pytest.raises(TrainInputError, match="cannot read TRAIN corpus range"):
        with reader as opened:
            opened.read_exact_at(0, 10)

    assert len(closed) == 1


def test_default_source_has_no_environment_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEMP", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", "ignored")

    with pytest.raises(TrainInputError, match="TEMP is required"):
        train_input._default_source()


def test_public_train_input_surface_has_no_path_or_range_selection() -> None:
    assert tuple(inspect.signature(train_input.load_verified_train_input).parameters) == ()
    assert tuple(inspect.signature(train_input.VerifiedTrainInput.iter_books).parameters) == (
        "self",
    )
    assert "_UnbufferedPositionedCorpus" not in train_input.__all__
    assert "_FilesystemTrainInputSource" not in train_input.__all__
