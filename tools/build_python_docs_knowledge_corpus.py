#!/usr/bin/env python3
"""Verify the frozen Python 3.14.7 docs lock and build a corpus receipt.

The command performs no download and writes no documentation text.  It verifies
the ignored local archive and extraction against a committed explicit inventory,
then invokes the generic local knowledge-corpus adapter and writes its receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
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


LOCK_SCHEMA = "humanbrain.python-docs-knowledge-lock.v1"
DEFAULT_LOCK_PATH = ROOT / "data" / "knowledge" / "python-3.14.7-docs.lock.json"
DEFAULT_CACHE_DIR = ROOT / ".knowledge-cache"
DEFAULT_RECEIPT_PATH = (
    DEFAULT_CACHE_DIR / "python-3.14.7-knowledge-corpus-receipt.json"
)

ARCHIVE_URL = "https://docs.python.org/3/archives/python-3.14-docs-text.zip"
ARCHIVE_SHA256 = "de728eb5b1de8d44ab04f234caeef0bc237bebcb8a6af8a219f35f9d9e35c0c0"
ARCHIVE_BYTES = 4_617_392
UPSTREAM_REVISION = "3.14.7"
ARCHIVE_FILES = 536
EXTRACTED_BYTES = 14_662_987

FAMILY_SPLITS = {
    "acquire": (
        "deprecations",
        "distributing",
        "faq",
        "glossary",
        "installing",
        "library",
    ),
    "validation": ("c-api", "extending", "tutorial"),
    "holdout": ("howto", "reference", "using"),
    "excluded": ("meta", "whatsnew"),
}
SPLIT_BYTES = {
    "acquire": 7_210_313,
    "validation": 1_562_249,
    "holdout": 1_529_560,
    "excluded": 4_360_865,
}
SPLIT_FILES = {
    "acquire": 354,
    "validation": 101,
    "holdout": 49,
    "excluded": 32,
}
INCLUDED_BYTES = 10_302_122
INCLUDED_FILES = 504

_SHA256 = re.compile(r"[0-9a-f]{64}")


class PythonDocsLockError(RuntimeError):
    """The committed lock or ignored local cache drifted."""


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def canonical_lock_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def digest(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _expected_totals(files: list[dict[str, Any]]) -> dict[str, object]:
    split_files = Counter(row["split"] for row in files)
    split_bytes = Counter()
    for row in files:
        split_bytes[row["split"]] += row["bytes"]
    included = [row for row in files if row["split"] != "excluded"]
    included_bytes = sum(row["bytes"] for row in included)
    return {
        "archive": {"bytes": sum(row["bytes"] for row in files), "files": len(files)},
        "excluded": {
            "bytes": split_bytes["excluded"],
            "files": split_files["excluded"],
        },
        "included": {"bytes": included_bytes, "files": len(included)},
        "splits": {
            split: {
                "byte_percent_x100": round(split_bytes[split] * 10_000 / included_bytes),
                "bytes": split_bytes[split],
                "files": split_files[split],
            }
            for split in ("acquire", "validation", "holdout")
        },
    }


def _validate_lock(payload: dict[str, Any]) -> None:
    if payload.get("schema") != LOCK_SCHEMA:
        raise PythonDocsLockError("lock schema differs")
    source = payload.get("source")
    if not isinstance(source, dict):
        raise PythonDocsLockError("lock source must be an object")
    expected_source = {
        "archive_bytes": ARCHIVE_BYTES,
        "archive_path": "python-3.14-docs-text.zip",
        "archive_sha256": ARCHIVE_SHA256,
        "attribution": "Python Software Foundation, Python 3.14.7 documentation",
        "canonical_url": ARCHIVE_URL,
        "extracted_root": "python-3.14-docs-text/python-3.14-docs-text",
        "license_evidence_url": "https://docs.python.org/3.14/license.html",
        "license_spdx": "PSF-2.0",
        "provenance_confidence": 1.0,
        "source_id": "python-3.14.7-docs",
        "upstream_revision": UPSTREAM_REVISION,
    }
    if source != expected_source:
        raise PythonDocsLockError("official source provenance pin differs")
    expected_families = {key: list(value) for key, value in FAMILY_SPLITS.items()}
    if payload.get("family_splits") != expected_families:
        raise PythonDocsLockError("whole-family split assignment differs")

    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise PythonDocsLockError("lock file inventory is absent")
    paths = [row.get("path") for row in files if isinstance(row, dict)]
    if len(paths) != len(files) or paths != sorted(paths) or len(paths) != len(set(paths)):
        raise PythonDocsLockError("file inventory paths must be unique and sorted")

    family_to_split = {
        family: split for split, families in FAMILY_SPLITS.items() for family in families
    }
    forbidden_tokens = payload.get("benchmark_policy", {}).get(
        "forbidden_path_tokens"
    )
    if not isinstance(forbidden_tokens, list) or not forbidden_tokens:
        raise PythonDocsLockError("benchmark path exclusions are absent")
    for row in files:
        if set(row) not in (
            {"bytes", "family", "path", "sha256", "split"},
            {"bytes", "exclude_reason", "family", "path", "sha256", "split"},
        ):
            raise PythonDocsLockError("file inventory row shape differs")
        path = row["path"]
        family = row["family"]
        split = row["split"]
        if not isinstance(path, str) or "\\" in path or path.startswith("/"):
            raise PythonDocsLockError("file inventory path is not canonical")
        if any(token.casefold() in path.casefold() for token in forbidden_tokens):
            raise PythonDocsLockError("benchmark path entered the docs inventory")
        expected_family = (
            path.split("/", 1)[0]
            if "/" in path
            else ("glossary" if path == "glossary.txt" else "meta")
        )
        if family != expected_family:
            raise PythonDocsLockError("file crossed its top-level documentation family")
        if family_to_split.get(family) != split:
            raise PythonDocsLockError("file crossed its frozen family split")
        if split == "excluded":
            expected_reason = "whatsnew-family" if family == "whatsnew" else "legal-meta"
            if row.get("exclude_reason") != expected_reason:
                raise PythonDocsLockError("excluded file reason differs")
        elif "exclude_reason" in row:
            raise PythonDocsLockError("included file has an exclusion reason")
        if (
            not isinstance(row["bytes"], int)
            or isinstance(row["bytes"], bool)
            or row["bytes"] <= 0
            or not isinstance(row["sha256"], str)
            or not _SHA256.fullmatch(row["sha256"])
        ):
            raise PythonDocsLockError("file size or SHA-256 is invalid")

    if payload.get("file_inventory_sha256") != digest(files):
        raise PythonDocsLockError("file inventory digest differs")
    included = [row for row in files if row["split"] != "excluded"]
    if payload.get("included_inventory_sha256") != digest(included):
        raise PythonDocsLockError("included inventory digest differs")
    if payload.get("family_split_sha256") != digest(payload["family_splits"]):
        raise PythonDocsLockError("family split digest differs")
    if payload.get("totals") != _expected_totals(files):
        raise PythonDocsLockError("lock totals differ from explicit inventory")
    if (
        len(files) != ARCHIVE_FILES
        or sum(row["bytes"] for row in files) != EXTRACTED_BYTES
        or len(included) != INCLUDED_FILES
        or sum(row["bytes"] for row in included) != INCLUDED_BYTES
    ):
        raise PythonDocsLockError("official extraction totals differ")
    for split in SPLIT_FILES:
        rows = [row for row in files if row["split"] == split]
        if len(rows) != SPLIT_FILES[split] or sum(row["bytes"] for row in rows) != SPLIT_BYTES[split]:
            raise PythonDocsLockError(f"frozen {split} totals differ")


def load_lock(path: str | Path = DEFAULT_LOCK_PATH) -> tuple[dict[str, Any], bytes]:
    lock_path = Path(path)
    raw = lock_path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise PythonDocsLockError("lock is not JSON") from error
    if not isinstance(payload, dict):
        raise PythonDocsLockError("lock must be a JSON object")
    if raw != canonical_lock_bytes(payload):
        raise PythonDocsLockError("lock bytes are not canonical")
    _validate_lock(payload)
    return payload, raw


def corpus_inputs_from_lock(
    payload: dict[str, Any],
) -> tuple[KnowledgeCorpusManifest, BenchmarkDenyManifest, KnowledgeCorpusCaps]:
    source = payload["source"]
    file_specs = tuple(
        KnowledgeFileSpec(
            path=row["path"],
            expected_raw_sha256=row["sha256"],
            split=row["split"],
            content_type="plain-text",
            chapter=row["family"],
            symbol=None,
        )
        for row in payload["files"]
        if row["split"] != "excluded"
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
                files=file_specs,
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
    caps = KnowledgeCorpusCaps(**payload["caps"])
    return manifest, deny_manifest, caps


def verify_local_cache(payload: dict[str, Any], cache_dir: str | Path) -> Path:
    cache = Path(cache_dir).resolve()
    source = payload["source"]
    archive = (cache / source["archive_path"]).resolve()
    if not archive.is_file() or archive.stat().st_size != source["archive_bytes"]:
        raise PythonDocsLockError("local archive is absent or has the wrong size")
    if file_sha256(archive) != source["archive_sha256"]:
        raise PythonDocsLockError("local archive SHA-256 differs")

    extracted_root = (cache / Path(source["extracted_root"])).resolve()
    if not extracted_root.is_dir():
        raise PythonDocsLockError("local extraction root is absent")
    all_nodes = tuple(extracted_root.rglob("*"))
    if any(node.is_symlink() for node in all_nodes):
        raise PythonDocsLockError("local extraction contains a symlink")
    actual_files = {
        path.relative_to(extracted_root).as_posix(): path
        for path in all_nodes
        if path.is_file()
    }
    locked_rows = {row["path"]: row for row in payload["files"]}
    if set(actual_files) != set(locked_rows):
        missing = sorted(set(locked_rows) - set(actual_files))
        extra = sorted(set(actual_files) - set(locked_rows))
        raise PythonDocsLockError(
            f"local extraction inventory differs; missing={missing[:3]}, extra={extra[:3]}"
        )
    for relative_path in sorted(locked_rows):
        row = locked_rows[relative_path]
        local = actual_files[relative_path]
        if local.stat().st_size != row["bytes"] or file_sha256(local) != row["sha256"]:
            raise PythonDocsLockError(f"local extracted file drifted: {relative_path}")
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
    chunk_splits = Counter(chunk.split for chunk in corpus.chunks)
    summary: dict[str, object] = {
        "adapter_version_sha256": corpus.receipt.adapter_version_sha256,
        "archive_sha256": payload["source"]["archive_sha256"],
        "benchmark_deny_manifest_sha256": corpus.receipt.benchmark_deny_manifest_sha256,
        "chunk_count": corpus.receipt.chunk_count,
        "chunk_splits": dict(sorted(chunk_splits.items())),
        "exact_dedup_cluster_count": corpus.receipt.exact_dedup_cluster_count,
        "file_inventory_sha256": payload["file_inventory_sha256"],
        "included_bytes": corpus.receipt.raw_bytes,
        "included_files": INCLUDED_FILES,
        "input_manifest_sha256": corpus.receipt.input_manifest_sha256,
        "lock_sha256": sha256(lock_bytes).hexdigest(),
        "receipt_sha256": sha256(receipt_bytes).hexdigest(),
        "records_sha256": corpus.receipt.records_sha256,
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
    except (OSError, ValueError, PythonDocsLockError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    write_atomic(args.receipt_out, corpus.canonical_manifest_bytes())
    summary["receipt_path"] = str(args.receipt_out.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
