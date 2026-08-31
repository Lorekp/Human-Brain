#!/usr/bin/env python3
"""Verify the frozen Rust Book source lock and build a corpus receipt offline.

The command performs no download and writes no source text. It authenticates an
ignored local archive plus an exact ten-file raw-Markdown extraction, then uses
the generic provenance-bound knowledge-corpus adapter.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanbrain.knowledge_corpus import (  # noqa: E402
    BenchmarkDenyManifest,
    KnowledgeCorpus,
    KnowledgeCorpusCaps,
    KnowledgeCorpusManifest,
    KnowledgeFileSpec,
    KnowledgeSourceSpec,
    build_local_knowledge_corpus,
)


LOCK_SCHEMA = "humanbrain.rust-book-knowledge-lock.v1"
DEFAULT_LOCK_PATH = (
    ROOT / "data" / "knowledge" / "rust-book-nostarch-third-printing.lock.json"
)
DEFAULT_CACHE_DIR = ROOT / ".knowledge-cache"
DEFAULT_RECEIPT_PATH = (
    DEFAULT_CACHE_DIR
    / "rust-book-nostarch-third-printing-knowledge-corpus-receipt.json"
)

ARCHIVE_URL = (
    "https://github.com/rust-lang/book/archive/refs/tags/"
    "nostarch-third-printing.tar.gz"
)
ARCHIVE_PATH = "rust-book-nostarch-third-printing.tar.gz"
ARCHIVE_ROOT = "book-nostarch-third-printing"
EXTRACTED_ROOT = "rust-book-nostarch-third-printing/book-nostarch-third-printing"
ARCHIVE_SHA256 = "74fb097c9fb2b8c4cccabfe0e6eb060a78add3037f051932d60e347e68286dca"
ARCHIVE_BYTES = 5_975_646
UPSTREAM_TAG = "nostarch-third-printing"
UPSTREAM_COMMIT = "844ddcd4d6d7726143ed8165950e395ccfd1f155"
SOURCE_ID = "rust-book-nostarch-third-printing"
LICENSE_EXPRESSION = "MIT OR Apache-2.0"
LICENSE_SELECTION = "MIT"
INCLUDED_FILES = 10
INCLUDED_BYTES = 185_367

FILE_METADATA: dict[str, dict[str, object]] = {
    "src/ch04-01-what-is-ownership.md": {
        "bytes": 25_424,
        "chapter": "ownership",
        "sha256": "7c6b95dfc1a5e87f39a40bcda2ab794ebfcd239326d13ea5c3bd432691ff381a",
        "symbol": "rust.memory.ownership",
    },
    "src/ch04-02-references-and-borrowing.md": {
        "bytes": 10_608,
        "chapter": "borrowing",
        "sha256": "7d983eec6235630df6e85c7a5b4cfbdfc7c380c780f60774df65ba2fa1d05ca4",
        "symbol": "rust.memory.borrowing",
    },
    "src/ch04-03-slices.md": {
        "bytes": 13_237,
        "chapter": "slices",
        "sha256": "fb0ac90f3652f4096624bc008f2a5ade603ed1d7af078281cec7a88da66e82bb",
        "symbol": "rust.memory.slices",
    },
    "src/ch09-02-recoverable-errors-with-result.md": {
        "bytes": 26_432,
        "chapter": "recoverable-errors",
        "sha256": "dab3e88d324cdde74cf96356c58a423747a5735ac6848d8d4592dc7c13637c62",
        "symbol": "rust.errors.result",
    },
    "src/ch10-02-traits.md": {
        "bytes": 18_682,
        "chapter": "traits",
        "sha256": "705ae11758c687175e47104a302a9b5f694d04d683bffc28dcc61ffe9a0dc80e",
        "symbol": "rust.abstraction.traits",
    },
    "src/ch10-03-lifetime-syntax.md": {
        "bytes": 30_935,
        "chapter": "lifetimes",
        "sha256": "8660fe043e6ca70645f6734f357dc61224ecd4e3454ebde7b696f89d6a0d6e91",
        "symbol": "rust.memory.lifetimes",
    },
    "src/ch11-01-writing-tests.md": {
        "bytes": 25_511,
        "chapter": "testing",
        "sha256": "03cec10df505d25d37717447f906d392506c18f4af1ba7ef9cc32f05cfc15359",
        "symbol": "rust.testing.writing_tests",
    },
    "src/ch13-02-iterators.md": {
        "bytes": 10_152,
        "chapter": "iterators",
        "sha256": "5f2a112bb6bbda36bc2ae2bf29196b433dd0fbcf649fcb0dd3f9448094592fd1",
        "symbol": "rust.iteration.iterators",
    },
    "src/ch16-02-message-passing.md": {
        "bytes": 11_867,
        "chapter": "message-passing",
        "sha256": "883ed745d87151fa820cd337420a24f59cf8391d4ce87d2ee22af3bd278e05d1",
        "symbol": "rust.concurrent.message_passing",
    },
    "src/ch16-03-shared-state.md": {
        "bytes": 12_519,
        "chapter": "shared-state",
        "sha256": "9aa8adf23ccf6ceaf7586b9aac35c30769bd13ebfbb8950a185d9b136104f10b",
        "symbol": "rust.concurrent.shared_state",
    },
}

LICENSES = [
    {
        "bytes": 10_847,
        "path": "LICENSE-APACHE",
        "sha256": "0f2763086f981043fb18879abfa15e75ecfc188219ef9eba4483f45ce7438e1c",
        "spdx": "Apache-2.0",
        "url": (
            "https://github.com/rust-lang/book/blob/"
            f"{UPSTREAM_COMMIT}/LICENSE-APACHE"
        ),
    },
    {
        "bytes": 1_071,
        "path": "LICENSE-MIT",
        "sha256": "0621878e61f0d0fda054bcbe02df75192c28bde1ecc8289cbd86aeba2dd72720",
        "spdx": "MIT",
        "url": (
            "https://github.com/rust-lang/book/blob/"
            f"{UPSTREAM_COMMIT}/LICENSE-MIT"
        ),
    },
]

BENCHMARK_POLICY = {
    "forbidden_path_tokens": ["deepswe", "swe-bench", "swe_bench", "swebench"],
    "statement": (
        "Official Rust Book raw Markdown only; Hugging Face datasets, benchmark "
        "tasks, gold patches, and solution corpora are absent."
    ),
}
BENCHMARK_DENY = {
    "domains": ["huggingface.co", "swebench.com"],
    "paths": ["deepswe.json", "swe-bench.json", "swe_bench.json"],
    "schema": "humanbrain.benchmark-deny-manifest.v1",
    "sha256": [],
}
RAW_MARKDOWN_POLICY = {
    "content_type": "markdown",
    "resolve_mdbook_includes": False,
    "resolve_rustdoc_includes": False,
    "statement": "Ingest authenticated chapter bytes exactly as raw Markdown.",
}
CAPS = {
    "max_bytes_per_source": 190_000,
    "max_tokenish_per_source": 50_000,
    "max_total_bytes": 190_000,
    "max_total_tokenish": 50_000,
}


class RustBookLockError(RuntimeError):
    """The committed lock or ignored local cache drifted."""


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def canonical_lock_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode(
        "utf-8"
    ) + b"\n"


def digest(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _expected_source() -> dict[str, object]:
    return {
        "archive_bytes": ARCHIVE_BYTES,
        "archive_path": ARCHIVE_PATH,
        "archive_root": ARCHIVE_ROOT,
        "archive_sha256": ARCHIVE_SHA256,
        "attribution": "The Rust Project Developers, The Rust Programming Language",
        "canonical_url": ARCHIVE_URL,
        "extracted_root": EXTRACTED_ROOT,
        "license_evidence_url": (
            "https://github.com/rust-lang/book/tree/" f"{UPSTREAM_COMMIT}"
        ),
        "license_expression_spdx": LICENSE_EXPRESSION,
        "license_spdx": LICENSE_SELECTION,
        "licenses": LICENSES,
        "provenance_confidence": 1.0,
        "release_tag": UPSTREAM_TAG,
        "source_id": SOURCE_ID,
        "upstream_revision": UPSTREAM_COMMIT,
    }


def _expected_files() -> list[dict[str, object]]:
    return [
        {
            "bytes": metadata["bytes"],
            "chapter": metadata["chapter"],
            "content_type": "markdown",
            "path": path,
            "sha256": metadata["sha256"],
            "split": "acquire",
            "symbol": metadata["symbol"],
        }
        for path, metadata in FILE_METADATA.items()
    ]


def _validate_lock(payload: dict[str, Any]) -> None:
    if payload.get("schema") != LOCK_SCHEMA:
        raise RustBookLockError("lock schema differs")
    if payload.get("source") != _expected_source():
        raise RustBookLockError("official source provenance pin differs")
    if payload.get("benchmark_policy") != BENCHMARK_POLICY:
        raise RustBookLockError("benchmark policy differs")
    if payload.get("benchmark_deny_manifest") != BENCHMARK_DENY:
        raise RustBookLockError("benchmark deny manifest differs")
    if payload.get("raw_markdown_policy") != RAW_MARKDOWN_POLICY:
        raise RustBookLockError("raw Markdown policy differs")
    if payload.get("caps") != CAPS:
        raise RustBookLockError("corpus caps differ")

    files = payload.get("files")
    expected_files = _expected_files()
    if files != expected_files:
        raise RustBookLockError("exact ten-file allowlist differs")
    if payload.get("file_inventory_sha256") != digest(files):
        raise RustBookLockError("file inventory digest differs")
    expected_totals = {
        "included": {"bytes": INCLUDED_BYTES, "files": INCLUDED_FILES},
        "splits": {
            "acquire": {
                "bytes": INCLUDED_BYTES,
                "files": INCLUDED_FILES,
            }
        },
    }
    if payload.get("totals") != expected_totals:
        raise RustBookLockError("lock totals differ")


def load_lock(path: str | Path = DEFAULT_LOCK_PATH) -> tuple[dict[str, Any], bytes]:
    lock_path = Path(path)
    raw = lock_path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RustBookLockError("lock is not JSON") from error
    if not isinstance(payload, dict):
        raise RustBookLockError("lock must be a JSON object")
    if raw != canonical_lock_bytes(payload):
        raise RustBookLockError("lock bytes are not canonical")
    _validate_lock(payload)
    return payload, raw


def corpus_inputs_from_lock(
    payload: dict[str, Any],
) -> tuple[KnowledgeCorpusManifest, BenchmarkDenyManifest, KnowledgeCorpusCaps]:
    source = payload["source"]
    files = tuple(
        KnowledgeFileSpec(
            path=row["path"],
            expected_raw_sha256=row["sha256"],
            split=row["split"],
            content_type=row["content_type"],
            chapter=row["chapter"],
            symbol=row["symbol"],
        )
        for row in payload["files"]
    )
    manifest = KnowledgeCorpusManifest(
        sources=(
            KnowledgeSourceSpec(
                source_id=source["source_id"],
                canonical_url=source["canonical_url"],
                upstream_revision=source["upstream_revision"],
                license_spdx=source["license_spdx"],
                license_evidence_url=source["license_evidence_url"],
                attribution=source["attribution"],
                provenance_confidence=source["provenance_confidence"],
                files=files,
            ),
        )
    )
    deny = payload["benchmark_deny_manifest"]
    deny_manifest = BenchmarkDenyManifest(
        paths=tuple(deny["paths"]),
        domains=tuple(deny["domains"]),
        sha256=tuple(deny["sha256"]),
        schema=deny["schema"],
    )
    return manifest, deny_manifest, KnowledgeCorpusCaps(**payload["caps"])


def verify_local_cache(payload: dict[str, Any], cache_dir: str | Path) -> Path:
    cache = Path(cache_dir).resolve()
    source = payload["source"]
    archive = (cache / source["archive_path"]).resolve()
    if not archive.is_file() or archive.stat().st_size != source["archive_bytes"]:
        raise RustBookLockError("local archive is absent or has the wrong size")
    if file_sha256(archive) != source["archive_sha256"]:
        raise RustBookLockError("local archive SHA-256 differs")

    extracted_root = (cache / Path(source["extracted_root"])).resolve()
    if not extracted_root.is_dir():
        raise RustBookLockError("local extraction root is absent")
    all_nodes = tuple(extracted_root.rglob("*"))
    if any(node.is_symlink() for node in all_nodes):
        raise RustBookLockError("local extraction contains a symlink")
    actual_files = {
        path.relative_to(extracted_root).as_posix(): path
        for path in all_nodes
        if path.is_file()
    }
    locked_rows = {row["path"]: row for row in payload["files"]}
    if set(actual_files) != set(locked_rows):
        missing = sorted(set(locked_rows) - set(actual_files))
        extra = sorted(set(actual_files) - set(locked_rows))
        raise RustBookLockError(
            f"local extraction inventory differs; missing={missing[:3]}, extra={extra[:3]}"
        )
    for relative_path, row in locked_rows.items():
        local = actual_files[relative_path]
        if local.stat().st_size != row["bytes"] or file_sha256(local) != row["sha256"]:
            raise RustBookLockError(f"local extracted file drifted: {relative_path}")
    return extracted_root


def build_from_cache(
    *,
    lock_path: str | Path = DEFAULT_LOCK_PATH,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> tuple[KnowledgeCorpus, dict[str, object]]:
    payload, lock_bytes = load_lock(lock_path)
    extracted_root = verify_local_cache(payload, cache_dir)
    manifest, deny_manifest, caps = corpus_inputs_from_lock(payload)
    corpus = build_local_knowledge_corpus(
        manifest,
        {payload["source"]["source_id"]: extracted_root},
        deny_manifest,
        caps,
    )
    receipt_bytes = corpus.canonical_manifest_bytes()
    chunk_kinds = Counter(chunk.kind for chunk in corpus.chunks)
    chunk_splits = Counter(chunk.split for chunk in corpus.chunks)
    summary: dict[str, object] = {
        "adapter_version_sha256": corpus.receipt.adapter_version_sha256,
        "archive_sha256": payload["source"]["archive_sha256"],
        "benchmark_deny_manifest_sha256": corpus.receipt.benchmark_deny_manifest_sha256,
        "chunk_count": corpus.receipt.chunk_count,
        "chunk_kinds": dict(sorted(chunk_kinds.items())),
        "chunk_splits": dict(sorted(chunk_splits.items())),
        "exact_dedup_cluster_count": corpus.receipt.exact_dedup_cluster_count,
        "file_inventory_sha256": payload["file_inventory_sha256"],
        "included_bytes": corpus.receipt.raw_bytes,
        "included_files": INCLUDED_FILES,
        "input_manifest_sha256": corpus.receipt.input_manifest_sha256,
        "license_expression_spdx": payload["source"]["license_expression_spdx"],
        "license_spdx": payload["source"]["license_spdx"],
        "lock_sha256": sha256(lock_bytes).hexdigest(),
        "receipt_sha256": sha256(receipt_bytes).hexdigest(),
        "records_sha256": corpus.receipt.records_sha256,
        "symbol_count": len({chunk.symbol for chunk in corpus.chunks}),
        "tokenish": corpus.receipt.tokenish,
    }
    return corpus, summary


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--receipt-out", type=Path, default=DEFAULT_RECEIPT_PATH)
    args = parser.parse_args(argv)
    try:
        corpus, summary = build_from_cache(
            lock_path=args.lock, cache_dir=args.cache_dir
        )
    except (OSError, ValueError, RustBookLockError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    write_atomic(args.receipt_out, corpus.canonical_manifest_bytes())
    summary["receipt_path"] = str(args.receipt_out.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
