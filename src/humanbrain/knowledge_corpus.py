"""Deterministic, provenance-bound local knowledge-corpus ingestion.

This module is an environment adapter.  It does not download data, retrieve
memories, infer metadata, or teach the cell system.  Version 1 only groups
normalization-exact duplicate chunks; near-duplicate detection is deliberately
outside its claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import unicodedata
from typing import Mapping
from urllib.parse import urlsplit


MANIFEST_SCHEMA = "humanbrain.knowledge-corpus-manifest.v1"
DENY_MANIFEST_SCHEMA = "humanbrain.benchmark-deny-manifest.v1"
OUTPUT_SCHEMA = "humanbrain.knowledge-corpus.v1"
CHUNK_SCHEMA = "humanbrain.knowledge-chunk.v1"
ADAPTER_VERSION = "humanbrain.local-knowledge-corpus.v1"

PERMISSIVE_LICENSE_SPDX = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC-BY-4.0",
        "CC0-1.0",
        "ISC",
        "MIT",
        "PSF-2.0",
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?")
_SPLIT = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})([^\r\n]*)(?:\r\n|\r|\n)?$")
_MOVING_REVISIONS = frozenset(
    {"head", "main", "master", "latest", "stable", "tip", "trunk"}
)
_CONTENT_TYPES = frozenset({"markdown", "plain-text", "source-code"})


class KnowledgeCorpusError(ValueError):
    """Base error for rejected corpus inputs."""


class KnowledgeCorpusPolicyError(KnowledgeCorpusError):
    """The explicit manifest violates ingestion policy."""


class KnowledgeCorpusIntegrityError(KnowledgeCorpusError):
    """Local bytes do not match their declared provenance."""


class KnowledgeCorpusCapacityError(KnowledgeCorpusError):
    """A hard byte or token-ish bound was exceeded."""


@dataclass(frozen=True, slots=True)
class KnowledgeFileSpec:
    """One explicitly allowlisted upstream file."""

    path: str
    expected_raw_sha256: str
    split: str
    content_type: str
    chapter: str | None = None
    symbol: str | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeSourceSpec:
    """Source-level legal and upstream provenance for allowlisted files."""

    source_id: str
    canonical_url: str
    upstream_revision: str
    license_spdx: str
    license_evidence_url: str
    attribution: str
    provenance_confidence: float
    files: tuple[KnowledgeFileSpec, ...]


@dataclass(frozen=True, slots=True)
class KnowledgeCorpusManifest:
    """Explicit source allowlist; filesystem locations are supplied separately."""

    sources: tuple[KnowledgeSourceSpec, ...]
    schema: str = MANIFEST_SCHEMA
    adapter_version: str = ADAPTER_VERSION


@dataclass(frozen=True, slots=True)
class BenchmarkDenyManifest:
    """Known benchmark identities that the acquisition lane must not ingest."""

    paths: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    sha256: tuple[str, ...] = ()
    schema: str = DENY_MANIFEST_SCHEMA


@dataclass(frozen=True, slots=True)
class KnowledgeCorpusCaps:
    """Hard bounds over bytes read and a stable UTF-8-bytes/4 token estimate."""

    max_bytes_per_source: int
    max_total_bytes: int
    max_tokenish_per_source: int
    max_total_tokenish: int


@dataclass(frozen=True, slots=True)
class KnowledgeFileReceipt:
    source_id: str
    path: str
    split: str
    chapter: str | None
    symbol: str | None
    raw_sha256: str
    normalized_sha256: str
    raw_bytes: int
    normalized_bytes: int
    tokenish: int
    chunk_count: int


@dataclass(frozen=True, slots=True)
class KnowledgeSourceReceipt:
    source_id: str
    canonical_url: str
    upstream_revision: str
    license_spdx: str
    license_evidence_url: str
    attribution_sha256: str
    provenance_confidence: float
    raw_bytes: int
    tokenish: int
    files: tuple[KnowledgeFileReceipt, ...]


@dataclass(frozen=True, slots=True)
class KnowledgeChunkRecord:
    schema: str
    chunk_id: str
    source_id: str
    canonical_url: str
    upstream_revision: str
    path: str
    chapter: str | None
    symbol: str | None
    split: str
    kind: str
    language: str | None
    ordinal: int
    text: str
    raw_sha256: str
    normalized_sha256: str
    raw_bytes: int
    normalized_bytes: int
    tokenish: int
    source_file_raw_sha256: str
    source_file_normalized_sha256: str
    license_spdx: str
    license_evidence_url: str
    attribution_sha256: str
    provenance_confidence: float
    input_manifest_sha256: str
    adapter_version_sha256: str
    benchmark_deny_manifest_sha256: str
    exact_dedup_cluster_sha256: str
    exact_dedup_representative_chunk_id: str
    exact_dedup_cluster_size: int


@dataclass(frozen=True, slots=True)
class KnowledgeCorpusReceipt:
    schema: str
    adapter_version: str
    adapter_version_sha256: str
    input_manifest_sha256: str
    benchmark_deny_manifest_sha256: str
    sources: tuple[KnowledgeSourceReceipt, ...]
    raw_bytes: int
    tokenish: int
    chunk_count: int
    exact_dedup_cluster_count: int
    records_sha256: str


@dataclass(frozen=True, slots=True)
class KnowledgeCorpus:
    """Immutable output at the adapter seam, ready for a later cell ingester."""

    receipt: KnowledgeCorpusReceipt
    chunks: tuple[KnowledgeChunkRecord, ...]

    def canonical_manifest_bytes(self) -> bytes:
        return _canonical_json_bytes(
            {"record_type": "manifest", "receipt": asdict(self.receipt)}
        )

    def canonical_jsonl_bytes(self) -> bytes:
        rows = [
            {"record_type": "manifest", "receipt": asdict(self.receipt)},
            *(
                {"record_type": "chunk", "chunk": asdict(chunk)}
                for chunk in self.chunks
            ),
        ]
        return b"".join(_canonical_json_bytes(row) for row in rows)


@dataclass(frozen=True, slots=True)
class _ChunkDraft:
    chunk_id: str
    source: KnowledgeSourceSpec
    file: KnowledgeFileSpec
    ordinal: int
    kind: str
    language: str | None
    text: str
    raw_sha256: str
    normalized_sha256: str
    raw_bytes: int
    normalized_bytes: int
    tokenish: int
    source_file_raw_sha256: str
    source_file_normalized_sha256: str
    attribution_sha256: str
    cluster_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _digest(value: object) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _tokenish(text: str) -> int:
    size = len(text.encode("utf-8"))
    return (size + 3) // 4


def _validate_url(value: str, field: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise KnowledgeCorpusPolicyError(f"{field} must be a canonical HTTP(S) URL")
    return parsed.hostname.casefold().rstrip(".")


def _validate_relpath(value: str, field: str) -> str:
    if "\\" in value:
        raise KnowledgeCorpusPolicyError(f"{field} must use POSIX separators")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise KnowledgeCorpusPolicyError(f"{field} must be a canonical relative path")
    canonical = path.as_posix()
    if canonical != value:
        raise KnowledgeCorpusPolicyError(f"{field} must be canonical: {canonical}")
    return canonical.casefold()


def _validate_sha256(value: str, field: str) -> None:
    if not _SHA256.fullmatch(value):
        raise KnowledgeCorpusPolicyError(f"{field} must be lowercase SHA-256")


def _validate_optional_locator(value: str | None, field: str) -> None:
    if value is not None and (not value.strip() or any(ord(char) < 32 for char in value)):
        raise KnowledgeCorpusPolicyError(f"{field} must be a non-empty printable string")


def _validate_caps(caps: KnowledgeCorpusCaps) -> None:
    for field, value in asdict(caps).items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise KnowledgeCorpusPolicyError(f"{field} must be a positive integer")


def _validate_deny_manifest(deny: BenchmarkDenyManifest) -> tuple[set[str], set[str], set[str]]:
    if deny.schema != DENY_MANIFEST_SCHEMA:
        raise KnowledgeCorpusPolicyError("benchmark deny manifest schema differs")
    paths = {_validate_relpath(path, "benchmark deny path") for path in deny.paths}
    domains: set[str] = set()
    for domain in deny.domains:
        normalized = domain.casefold().rstrip(".")
        if (
            not normalized
            or "/" in normalized
            or ":" in normalized
            or normalized.startswith(".")
            or normalized.endswith(".")
        ):
            raise KnowledgeCorpusPolicyError("benchmark deny domain is invalid")
        domains.add(normalized)
    hashes = set(deny.sha256)
    for value in hashes:
        _validate_sha256(value, "benchmark deny hash")
    if len(paths) != len(deny.paths) or len(domains) != len(deny.domains) or len(hashes) != len(deny.sha256):
        raise KnowledgeCorpusPolicyError("benchmark deny manifest contains duplicates")
    return paths, domains, hashes


def _domain_is_denied(domain: str, denied: set[str]) -> bool:
    return any(domain == item or domain.endswith("." + item) for item in denied)


def _validate_manifest(
    manifest: KnowledgeCorpusManifest,
    denied_paths: set[str],
    denied_domains: set[str],
    denied_hashes: set[str],
) -> None:
    if manifest.schema != MANIFEST_SCHEMA or manifest.adapter_version != ADAPTER_VERSION:
        raise KnowledgeCorpusPolicyError("knowledge manifest schema or adapter version differs")
    if not manifest.sources:
        raise KnowledgeCorpusPolicyError("knowledge manifest must allowlist at least one source")

    source_ids: set[str] = set()
    for source in manifest.sources:
        if not _IDENTIFIER.fullmatch(source.source_id) or source.source_id in source_ids:
            raise KnowledgeCorpusPolicyError("source_id must be unique lowercase identifier")
        source_ids.add(source.source_id)
        domain = _validate_url(source.canonical_url, "canonical_url")
        _validate_url(source.license_evidence_url, "license_evidence_url")
        if _domain_is_denied(domain, denied_domains):
            raise KnowledgeCorpusPolicyError(
                f"source domain is benchmark-denied: {domain}"
            )
        if (
            not source.upstream_revision.strip()
            or source.upstream_revision.casefold() in _MOVING_REVISIONS
        ):
            raise KnowledgeCorpusPolicyError("upstream_revision must pin a release or commit")
        if source.license_spdx not in PERMISSIVE_LICENSE_SPDX:
            raise KnowledgeCorpusPolicyError(
                f"license is not in the permissive lane: {source.license_spdx!r}"
            )
        if not source.attribution.strip():
            raise KnowledgeCorpusPolicyError("attribution must be explicit")
        if (
            isinstance(source.provenance_confidence, bool)
            or not isinstance(source.provenance_confidence, (float, int))
            or not 0.0 <= source.provenance_confidence <= 1.0
        ):
            raise KnowledgeCorpusPolicyError("provenance_confidence must be within [0, 1]")
        if not source.files:
            raise KnowledgeCorpusPolicyError("each source must allowlist at least one file")

        seen_paths: set[str] = set()
        for file in source.files:
            path = _validate_relpath(file.path, "source path")
            if path in seen_paths:
                raise KnowledgeCorpusPolicyError("source file paths must be unique")
            seen_paths.add(path)
            if path in denied_paths:
                raise KnowledgeCorpusPolicyError(
                    f"source path is benchmark-denied: {file.path}"
                )
            _validate_sha256(file.expected_raw_sha256, "expected_raw_sha256")
            if file.expected_raw_sha256 in denied_hashes:
                raise KnowledgeCorpusPolicyError(
                    f"source hash is benchmark-denied: {source.source_id}/{file.path}"
                )
            if not _SPLIT.fullmatch(file.split):
                raise KnowledgeCorpusPolicyError("split must be a lowercase identifier")
            if file.content_type not in _CONTENT_TYPES:
                raise KnowledgeCorpusPolicyError(
                    f"unsupported content_type: {file.content_type!r}"
                )
            _validate_optional_locator(file.chapter, "chapter")
            _validate_optional_locator(file.symbol, "symbol")


def _split_paragraphs(raw_text: str) -> list[tuple[str, str, str | None]]:
    chunks: list[tuple[str, str, str | None]] = []
    paragraph: list[str] = []
    for line in raw_text.splitlines(keepends=True):
        if line.strip():
            paragraph.append(line)
        elif paragraph:
            chunks.append(("paragraph", "".join(paragraph), None))
            paragraph = []
    if paragraph:
        chunks.append(("paragraph", "".join(paragraph), None))
    return chunks


def _split_markdown(raw_text: str) -> list[tuple[str, str, str | None]]:
    chunks: list[tuple[str, str, str | None]] = []
    paragraph: list[str] = []
    code: list[str] | None = None
    fence_character = ""
    fence_length = 0
    language: str | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            chunks.append(("paragraph", "".join(paragraph), None))
            paragraph = []

    for line in raw_text.splitlines(keepends=True):
        fence = _FENCE.match(line)
        if code is None:
            if fence is not None:
                flush_paragraph()
                marker = fence.group(1)
                fence_character = marker[0]
                fence_length = len(marker)
                info = fence.group(2).strip()
                language = info.split(maxsplit=1)[0] if info else None
                code = []
            elif line.strip():
                paragraph.append(line)
            else:
                flush_paragraph()
        else:
            if (
                fence is not None
                and fence.group(1)[0] == fence_character
                and len(fence.group(1)) >= fence_length
                and not fence.group(2).strip()
            ):
                chunks.append(("code-block", "".join(code), language))
                code = None
                language = None
            else:
                code.append(line)

    if code is not None:
        raise KnowledgeCorpusIntegrityError("markdown contains an unclosed code fence")
    flush_paragraph()
    return chunks


def _split_chunks(
    raw_text: str, content_type: str
) -> list[tuple[str, str, str | None]]:
    if content_type == "markdown":
        chunks = _split_markdown(raw_text)
    elif content_type == "plain-text":
        chunks = _split_paragraphs(raw_text)
    else:
        chunks = [("code-block", raw_text, None)]
    return [chunk for chunk in chunks if _normalize_text(chunk[1])]


def _resolve_local_file(
    supplied: str | Path, file: KnowledgeFileSpec, file_count: int
) -> Path:
    anchor = Path(supplied).resolve()
    if anchor.is_file():
        if file_count != 1:
            raise KnowledgeCorpusIntegrityError(
                "a file anchor can satisfy only a one-file source"
            )
        return anchor
    if not anchor.is_dir():
        raise KnowledgeCorpusIntegrityError(f"local source does not exist: {anchor}")
    candidate = (anchor / Path(*PurePosixPath(file.path).parts)).resolve()
    try:
        candidate.relative_to(anchor)
    except ValueError as error:
        raise KnowledgeCorpusIntegrityError("source path escaped its local root") from error
    if not candidate.is_file():
        raise KnowledgeCorpusIntegrityError(f"allowlisted source file is absent: {file.path}")
    return candidate


def _enforce_caps(
    caps: KnowledgeCorpusCaps,
    source_id: str,
    source_bytes: int,
    total_bytes: int,
    source_tokenish: int,
    total_tokenish: int,
) -> None:
    if source_bytes > caps.max_bytes_per_source:
        raise KnowledgeCorpusCapacityError(f"source byte cap exceeded: {source_id}")
    if total_bytes > caps.max_total_bytes:
        raise KnowledgeCorpusCapacityError("total byte cap exceeded")
    if source_tokenish > caps.max_tokenish_per_source:
        raise KnowledgeCorpusCapacityError(f"source token-ish cap exceeded: {source_id}")
    if total_tokenish > caps.max_total_tokenish:
        raise KnowledgeCorpusCapacityError("total token-ish cap exceeded")


def build_local_knowledge_corpus(
    manifest: KnowledgeCorpusManifest,
    local_sources: Mapping[str, str | Path],
    benchmark_deny_manifest: BenchmarkDenyManifest,
    caps: KnowledgeCorpusCaps,
) -> KnowledgeCorpus:
    """Validate and chunk allowlisted local bytes without external lookups.

    ``local_sources`` maps each manifest ``source_id`` to either its local root
    directory or, for a one-file source, directly to that file.
    """

    _validate_caps(caps)
    denied_paths, denied_domains, denied_hashes = _validate_deny_manifest(
        benchmark_deny_manifest
    )
    _validate_manifest(manifest, denied_paths, denied_domains, denied_hashes)

    expected_keys = {source.source_id for source in manifest.sources}
    if set(local_sources) != expected_keys:
        missing = sorted(expected_keys - set(local_sources))
        extra = sorted(set(local_sources) - expected_keys)
        raise KnowledgeCorpusIntegrityError(
            f"local source keys differ; missing={missing}, extra={extra}"
        )

    input_manifest_sha256 = _digest(asdict(manifest))
    adapter_version_sha256 = _digest({"adapter_version": ADAPTER_VERSION})
    deny_manifest_sha256 = _digest(asdict(benchmark_deny_manifest))
    source_receipts: list[KnowledgeSourceReceipt] = []
    drafts: list[_ChunkDraft] = []
    total_bytes = 0
    total_tokenish = 0

    for source in manifest.sources:
        source_bytes = 0
        source_tokenish = 0
        file_receipts: list[KnowledgeFileReceipt] = []
        attribution_sha256 = sha256(
            _normalize_text(source.attribution).encode("utf-8")
        ).hexdigest()

        for file in source.files:
            path = _resolve_local_file(
                local_sources[source.source_id], file, len(source.files)
            )
            advertised_size = path.stat().st_size
            _enforce_caps(
                caps,
                source.source_id,
                source_bytes + advertised_size,
                total_bytes + advertised_size,
                source_tokenish,
                total_tokenish,
            )
            raw = path.read_bytes()
            raw_sha256 = sha256(raw).hexdigest()
            if raw_sha256 != file.expected_raw_sha256:
                raise KnowledgeCorpusIntegrityError(
                    f"source bytes differ from allowlist: {source.source_id}/{file.path}"
                )
            if raw_sha256 in denied_hashes:
                raise KnowledgeCorpusPolicyError(
                    f"source hash is benchmark-denied: {source.source_id}/{file.path}"
                )
            try:
                raw_text = raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise KnowledgeCorpusIntegrityError(
                    f"source is not UTF-8: {source.source_id}/{file.path}"
                ) from error
            normalized_file = _normalize_text(raw_text)
            normalized_file_bytes = normalized_file.encode("utf-8")
            normalized_file_sha256 = sha256(normalized_file_bytes).hexdigest()
            if normalized_file_sha256 in denied_hashes:
                raise KnowledgeCorpusPolicyError(
                    f"normalized source hash is benchmark-denied: {source.source_id}/{file.path}"
                )
            file_tokenish = _tokenish(normalized_file)
            source_bytes += len(raw)
            total_bytes += len(raw)
            source_tokenish += file_tokenish
            total_tokenish += file_tokenish
            _enforce_caps(
                caps,
                source.source_id,
                source_bytes,
                total_bytes,
                source_tokenish,
                total_tokenish,
            )

            chunk_start = len(drafts)
            for ordinal, (kind, raw_chunk, language) in enumerate(
                _split_chunks(raw_text, file.content_type)
            ):
                normalized = _normalize_text(raw_chunk)
                raw_chunk_bytes = raw_chunk.encode("utf-8")
                normalized_bytes = normalized.encode("utf-8")
                chunk_raw_sha256 = sha256(raw_chunk_bytes).hexdigest()
                chunk_normalized_sha256 = sha256(normalized_bytes).hexdigest()
                if (
                    chunk_raw_sha256 in denied_hashes
                    or chunk_normalized_sha256 in denied_hashes
                ):
                    raise KnowledgeCorpusPolicyError(
                        f"chunk hash is benchmark-denied: {source.source_id}/{file.path}#{ordinal}"
                    )
                cluster_sha256 = _digest(
                    {
                        "kind": kind,
                        "language": language,
                        "normalized_sha256": chunk_normalized_sha256,
                    }
                )
                chunk_id = _digest(
                    {
                        "adapter_version_sha256": adapter_version_sha256,
                        "benchmark_deny_manifest_sha256": deny_manifest_sha256,
                        "input_manifest_sha256": input_manifest_sha256,
                        "source_id": source.source_id,
                        "path": file.path,
                        "ordinal": ordinal,
                        "kind": kind,
                        "language": language,
                        "raw_sha256": chunk_raw_sha256,
                        "normalized_sha256": chunk_normalized_sha256,
                    }
                )
                drafts.append(
                    _ChunkDraft(
                        chunk_id=chunk_id,
                        source=source,
                        file=file,
                        ordinal=ordinal,
                        kind=kind,
                        language=language,
                        text=normalized,
                        raw_sha256=chunk_raw_sha256,
                        normalized_sha256=chunk_normalized_sha256,
                        raw_bytes=len(raw_chunk_bytes),
                        normalized_bytes=len(normalized_bytes),
                        tokenish=_tokenish(normalized),
                        source_file_raw_sha256=raw_sha256,
                        source_file_normalized_sha256=normalized_file_sha256,
                        attribution_sha256=attribution_sha256,
                        cluster_sha256=cluster_sha256,
                    )
                )

            file_receipts.append(
                KnowledgeFileReceipt(
                    source_id=source.source_id,
                    path=file.path,
                    split=file.split,
                    chapter=file.chapter,
                    symbol=file.symbol,
                    raw_sha256=raw_sha256,
                    normalized_sha256=normalized_file_sha256,
                    raw_bytes=len(raw),
                    normalized_bytes=len(normalized_file_bytes),
                    tokenish=file_tokenish,
                    chunk_count=len(drafts) - chunk_start,
                )
            )

        source_receipts.append(
            KnowledgeSourceReceipt(
                source_id=source.source_id,
                canonical_url=source.canonical_url,
                upstream_revision=source.upstream_revision,
                license_spdx=source.license_spdx,
                license_evidence_url=source.license_evidence_url,
                attribution_sha256=attribution_sha256,
                provenance_confidence=float(source.provenance_confidence),
                raw_bytes=source_bytes,
                tokenish=source_tokenish,
                files=tuple(file_receipts),
            )
        )

    cluster_counts: dict[str, int] = {}
    cluster_representatives: dict[str, str] = {}
    for draft in drafts:
        cluster_counts[draft.cluster_sha256] = cluster_counts.get(draft.cluster_sha256, 0) + 1
        cluster_representatives.setdefault(draft.cluster_sha256, draft.chunk_id)

    chunks = tuple(
        KnowledgeChunkRecord(
            schema=CHUNK_SCHEMA,
            chunk_id=draft.chunk_id,
            source_id=draft.source.source_id,
            canonical_url=draft.source.canonical_url,
            upstream_revision=draft.source.upstream_revision,
            path=draft.file.path,
            chapter=draft.file.chapter,
            symbol=draft.file.symbol,
            split=draft.file.split,
            kind=draft.kind,
            language=draft.language,
            ordinal=draft.ordinal,
            text=draft.text,
            raw_sha256=draft.raw_sha256,
            normalized_sha256=draft.normalized_sha256,
            raw_bytes=draft.raw_bytes,
            normalized_bytes=draft.normalized_bytes,
            tokenish=draft.tokenish,
            source_file_raw_sha256=draft.source_file_raw_sha256,
            source_file_normalized_sha256=draft.source_file_normalized_sha256,
            license_spdx=draft.source.license_spdx,
            license_evidence_url=draft.source.license_evidence_url,
            attribution_sha256=draft.attribution_sha256,
            provenance_confidence=float(draft.source.provenance_confidence),
            input_manifest_sha256=input_manifest_sha256,
            adapter_version_sha256=adapter_version_sha256,
            benchmark_deny_manifest_sha256=deny_manifest_sha256,
            exact_dedup_cluster_sha256=draft.cluster_sha256,
            exact_dedup_representative_chunk_id=cluster_representatives[draft.cluster_sha256],
            exact_dedup_cluster_size=cluster_counts[draft.cluster_sha256],
        )
        for draft in drafts
    )
    records_sha256 = _digest([asdict(chunk) for chunk in chunks])
    receipt = KnowledgeCorpusReceipt(
        schema=OUTPUT_SCHEMA,
        adapter_version=ADAPTER_VERSION,
        adapter_version_sha256=adapter_version_sha256,
        input_manifest_sha256=input_manifest_sha256,
        benchmark_deny_manifest_sha256=deny_manifest_sha256,
        sources=tuple(source_receipts),
        raw_bytes=total_bytes,
        tokenish=total_tokenish,
        chunk_count=len(chunks),
        exact_dedup_cluster_count=len(cluster_counts),
        records_sha256=records_sha256,
    )
    return KnowledgeCorpus(receipt=receipt, chunks=chunks)


__all__ = [
    "ADAPTER_VERSION",
    "BenchmarkDenyManifest",
    "KnowledgeChunkRecord",
    "KnowledgeCorpus",
    "KnowledgeCorpusCapacityError",
    "KnowledgeCorpusCaps",
    "KnowledgeCorpusError",
    "KnowledgeCorpusIntegrityError",
    "KnowledgeCorpusManifest",
    "KnowledgeCorpusPolicyError",
    "KnowledgeFileSpec",
    "KnowledgeSourceSpec",
    "PERMISSIVE_LICENSE_SPDX",
    "build_local_knowledge_corpus",
]
