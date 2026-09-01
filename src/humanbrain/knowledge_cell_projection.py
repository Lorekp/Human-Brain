"""Neutral, provenance-bound projection from knowledge records to cell events.

The adapter admits only caller-named chunk ids from one caller-named split.  It
does not see a task, action, candidate, expected answer, or evaluation result.
Raw text is reduced to deterministic lexical factors and first-line signature
symbols; the resulting bounded graph and producer receipt remain rooted in the
authenticated corpus-record digest.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256

from experiments.r3_pilot import common_rooted_structured_event_graph_v1 as structured_graph

from humanbrain.knowledge_consolidation import lexical_factors_from_text_v1
from humanbrain.knowledge_corpus import (
    KnowledgeChunkRecord,
    KnowledgeCorpus,
    KnowledgeCorpusError,
)

SCHEMA = "humanbrain.knowledge-cell-projection.v1"
MAX_ADMITTED_GROUPS = 8
MAX_ADMITTED_CHUNKS = 16
MAX_GROUP_MEMBERS = 4
MAX_QUERY_FACTORS = 24
MAX_SOURCE_FACTORS_PER_GROUP = 128
MAX_SYMBOLS_PER_CHUNK = 2
MAX_ADMITTED_UTF8_BYTES = 16 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SIGNATURE = re.compile(
    r"^\s*(?P<symbol>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\s*\("
)
_SYMBOL = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
)

_CHUNK_NAMESPACE = b"knowledge-cell-projected-chunk-v1"
_GROUP_ID_NAMESPACE = b"knowledge-cell-projected-group-id-v1"
_PROJECTION_NAMESPACE = b"knowledge-cell-projection-v1"
_SCHEMA_NAMESPACE = b"knowledge-cell-projection-schema-v1"
_ATTESTATION_NAMESPACE = b"knowledge-cell-projection-attestation-v1"
_EVENT_PAYLOAD_NAMESPACE = b"knowledge-cell-projection-event-v1"
_ROOT_PAYLOAD_NAMESPACE = b"knowledge-cell-projection-root-v1"
_BOUNDARY_PAYLOAD_NAMESPACE = b"knowledge-cell-projection-boundary-v1"


class KnowledgeCellProjectionError(ValueError):
    """An admitted knowledge projection failed integrity or capacity checks."""


def _canonical_json(value: object, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + suffix
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _payload16(namespace: bytes, value: object) -> bytes:
    return sha256(namespace + b"\0" + _canonical_json(value)).digest()[:16]


def document_symbols_from_text_v1(text: str) -> tuple[str, ...]:
    """Extract a qualified symbol only from the first non-empty signature line."""

    if type(text) is not str:
        raise TypeError("knowledge text must be one exact string")
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    match = _SIGNATURE.match(first_line)
    return () if match is None else (match.group("symbol").casefold(),)


@dataclass(frozen=True, slots=True)
class ProjectedKnowledgeChunkV1:
    chunk_id: str
    member_chunk_ids: tuple[str, ...]
    record_sha256: str
    source_id: str
    path: str
    split: str
    ordinal_start: int
    ordinal_end: int
    content_factors: tuple[str, ...]
    document_symbols: tuple[str, ...]
    declared_symbols: tuple[str, ...]
    provenance_roots: tuple[str, ...]
    sha256: str

    def authenticate(self) -> None:
        if (
            not _valid_sha256(self.chunk_id)
            or type(self.member_chunk_ids) is not tuple
            or not 1 <= len(self.member_chunk_ids) <= MAX_GROUP_MEMBERS
            or any(not _valid_sha256(value) for value in self.member_chunk_ids)
            or len(set(self.member_chunk_ids)) != len(self.member_chunk_ids)
            or not _valid_sha256(self.record_sha256)
            or type(self.source_id) is not str
            or not self.source_id
            or type(self.path) is not str
            or not self.path
            or type(self.split) is not str
            or not self.split
            or type(self.ordinal_start) is not int
            or type(self.ordinal_end) is not int
            or self.ordinal_end - self.ordinal_start + 1 != len(self.member_chunk_ids)
            or type(self.content_factors) is not tuple
            or self.content_factors != tuple(sorted(set(self.content_factors)))
            or not self.content_factors
            or len(self.content_factors) > MAX_SOURCE_FACTORS_PER_GROUP
            or type(self.document_symbols) is not tuple
            or self.document_symbols
            != tuple(sorted(set(self.document_symbols)))
            or len(self.document_symbols) > MAX_SYMBOLS_PER_CHUNK
            or any(_SYMBOL.fullmatch(value) is None for value in self.document_symbols)
            or type(self.declared_symbols) is not tuple
            or self.declared_symbols != tuple(sorted(set(self.declared_symbols)))
            or any(_SYMBOL.fullmatch(value) is None for value in self.declared_symbols)
            or type(self.provenance_roots) is not tuple
            or self.provenance_roots != tuple(sorted(set(self.provenance_roots)))
            or any(not _valid_sha256(value) for value in self.provenance_roots)
        ):
            raise KnowledgeCellProjectionError(
                "projected knowledge chunk authentication failed"
            )
        payload = {
            "chunk_id": self.chunk_id,
            "member_chunk_ids": self.member_chunk_ids,
            "record_sha256": self.record_sha256,
            "source_id": self.source_id,
            "path": self.path,
            "split": self.split,
            "ordinal_start": self.ordinal_start,
            "ordinal_end": self.ordinal_end,
            "content_factors": self.content_factors,
            "document_symbols": self.document_symbols,
            "declared_symbols": self.declared_symbols,
            "provenance_roots": self.provenance_roots,
        }
        if self.sha256 != _digest(_CHUNK_NAMESPACE, payload):
            raise KnowledgeCellProjectionError(
                "projected knowledge chunk root changed"
            )


def _projected_chunk(
    records: tuple[KnowledgeChunkRecord, ...],
) -> ProjectedKnowledgeChunkV1:
    if not 1 <= len(records) <= MAX_GROUP_MEMBERS:
        raise KnowledgeCellProjectionError(
            f"chunk group must contain 1..{MAX_GROUP_MEMBERS} records"
        )
    first = records[0]
    if any(
        record.source_id != first.source_id
        or record.path != first.path
        or record.split != first.split
        for record in records
    ):
        raise KnowledgeCellProjectionError(
            "chunk group crosses source, path, or split"
        )
    ordinals = tuple(record.ordinal for record in records)
    if ordinals != tuple(range(ordinals[0], ordinals[0] + len(records))):
        raise KnowledgeCellProjectionError(
            "chunk group must be ordered and ordinal-contiguous"
        )
    member_chunk_ids = tuple(record.chunk_id for record in records)
    group_id = _digest(_GROUP_ID_NAMESPACE, member_chunk_ids)
    record_sha256 = sha256(
        _canonical_json([asdict(record) for record in records], newline=True)
    ).hexdigest()
    content_factors = tuple(
        sorted(
            {
                factor
                for record in records
                for factor in lexical_factors_from_text_v1(record.text)
            }
        )
    )
    if len(content_factors) > MAX_SOURCE_FACTORS_PER_GROUP:
        raise KnowledgeCellProjectionError(
            "source chunk group exceeds "
            f"{MAX_SOURCE_FACTORS_PER_GROUP} lexical factors"
        )
    document_symbols = tuple(
        sorted(
            {
                symbol
                for record in records
                for symbol in document_symbols_from_text_v1(record.text)
            }
        )
    )
    declared_symbols = tuple(
        sorted(
            {
                record.symbol.casefold()
                for record in records
                if record.symbol is not None
            }
        )
    )
    if any(_SYMBOL.fullmatch(value) is None for value in declared_symbols):
        raise KnowledgeCellProjectionError("declared corpus symbol is malformed")
    provenance_roots = tuple(
        sorted(
            {
                record_sha256,
                *(
                    root
                    for record in records
                    for root in (
                        record.raw_sha256,
                        record.normalized_sha256,
                        record.source_file_raw_sha256,
                        record.source_file_normalized_sha256,
                        record.attribution_sha256,
                        record.input_manifest_sha256,
                        record.adapter_version_sha256,
                        record.benchmark_deny_manifest_sha256,
                        record.exact_dedup_cluster_sha256,
                    )
                ),
            }
        )
    )
    payload = {
        "chunk_id": group_id,
        "member_chunk_ids": member_chunk_ids,
        "record_sha256": record_sha256,
        "source_id": first.source_id,
        "path": first.path,
        "split": first.split,
        "ordinal_start": ordinals[0],
        "ordinal_end": ordinals[-1],
        "content_factors": content_factors,
        "document_symbols": document_symbols,
        "declared_symbols": declared_symbols,
        "provenance_roots": provenance_roots,
    }
    result = ProjectedKnowledgeChunkV1(
        chunk_id=group_id,
        member_chunk_ids=member_chunk_ids,
        record_sha256=record_sha256,
        source_id=first.source_id,
        path=first.path,
        split=first.split,
        ordinal_start=ordinals[0],
        ordinal_end=ordinals[-1],
        content_factors=content_factors,
        document_symbols=document_symbols,
        declared_symbols=declared_symbols,
        provenance_roots=provenance_roots,
        sha256=_digest(_CHUNK_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def _build_graph(
    records_sha256: str,
    split: str,
    admitted_chunk_groups: tuple[tuple[str, ...], ...],
    admitted_chunk_ids: tuple[str, ...],
    chunks: tuple[ProjectedKnowledgeChunkV1, ...],
) -> structured_graph.RootedStructuredEventGraphV1:
    event_payloads = tuple(
        _payload16(_EVENT_PAYLOAD_NAMESPACE, chunk.sha256) for chunk in chunks
    )
    carrier_payloads = (
        _payload16(_ROOT_PAYLOAD_NAMESPACE, records_sha256),
        _payload16(
            _BOUNDARY_PAYLOAD_NAMESPACE,
            {
                "split": split,
                "admitted_chunk_groups": admitted_chunk_groups,
                "admitted_chunk_ids": admitted_chunk_ids,
            },
        ),
    )
    incidences = (
        (0, 1, 0),
        *(
            (1, len(carrier_payloads) + ordinal, ordinal)
            for ordinal in range(len(chunks))
        ),
    )
    return structured_graph.project_rooted_structured_event_graph_v1(
        event_payloads,
        carrier_payloads=carrier_payloads,
        root_carrier_indices=(0,),
        boundary_carrier_indices=(1,),
        incidences=incidences,
    )


def _producer_schema_sha256() -> str:
    return _digest(
        _SCHEMA_NAMESPACE,
        {
            "schema": SCHEMA,
            "max_admitted_groups": MAX_ADMITTED_GROUPS,
            "max_admitted_chunks": MAX_ADMITTED_CHUNKS,
            "max_group_members": MAX_GROUP_MEMBERS,
            "max_query_factors": MAX_QUERY_FACTORS,
            "max_source_factors_per_group": MAX_SOURCE_FACTORS_PER_GROUP,
            "max_symbols_per_chunk": MAX_SYMBOLS_PER_CHUNK,
            "max_admitted_utf8_bytes": MAX_ADMITTED_UTF8_BYTES,
            "signature_rule": "first-nonempty-qualified-call-line",
            "lexical_rule": "all-distinct-ascii-identifiers-plus-dot-underscore-parts",
        },
    )


def _producer_attestation_sha256(
    *,
    records_sha256: str,
    split: str,
    admitted_chunk_groups: tuple[tuple[str, ...], ...],
    admitted_chunk_ids: tuple[str, ...],
    chunks: tuple[ProjectedKnowledgeChunkV1, ...],
    graph_sha256: str,
) -> str:
    return _digest(
        _ATTESTATION_NAMESPACE,
        {
            "records_sha256": records_sha256,
            "split": split,
            "admitted_chunk_groups": admitted_chunk_groups,
            "admitted_chunk_ids": admitted_chunk_ids,
            "chunks": tuple(chunk.sha256 for chunk in chunks),
            "graph_sha256": graph_sha256,
        },
    )


def _projection_sha256(
    *,
    records_sha256: str,
    split: str,
    admitted_chunk_groups: tuple[tuple[str, ...], ...],
    admitted_chunk_ids: tuple[str, ...],
    chunks: tuple[ProjectedKnowledgeChunkV1, ...],
    graph_sha256: str,
    producer_receipt_sha256: str,
) -> str:
    return _digest(
        _PROJECTION_NAMESPACE,
        {
            "records_sha256": records_sha256,
            "split": split,
            "admitted_chunk_groups": admitted_chunk_groups,
            "admitted_chunk_ids": admitted_chunk_ids,
            "chunks": tuple(chunk.sha256 for chunk in chunks),
            "graph_sha256": graph_sha256,
            "producer_receipt_sha256": producer_receipt_sha256,
        },
    )


@dataclass(frozen=True, slots=True)
class KnowledgeCellProjectionV1:
    records_sha256: str
    split: str
    admitted_chunk_groups: tuple[tuple[str, ...], ...]
    admitted_chunk_ids: tuple[str, ...]
    chunks: tuple[ProjectedKnowledgeChunkV1, ...]
    graph: structured_graph.RootedStructuredEventGraphV1
    producer_receipt: structured_graph.StructuredEventGraphProducerReceiptV1
    sha256: str

    def authenticate(self) -> None:
        if (
            not _valid_sha256(self.records_sha256)
            or type(self.split) is not str
            or not self.split
            or type(self.admitted_chunk_ids) is not tuple
            or not 1 <= len(self.admitted_chunk_ids) <= MAX_ADMITTED_CHUNKS
            or len(set(self.admitted_chunk_ids)) != len(self.admitted_chunk_ids)
            or type(self.admitted_chunk_groups) is not tuple
            or not 1 <= len(self.admitted_chunk_groups) <= MAX_ADMITTED_GROUPS
            or tuple(
                value for group in self.admitted_chunk_groups for value in group
            )
            != self.admitted_chunk_ids
            or type(self.chunks) is not tuple
            or tuple(chunk.member_chunk_ids for chunk in self.chunks)
            != self.admitted_chunk_groups
        ):
            raise KnowledgeCellProjectionError("knowledge projection shape changed")
        for chunk in self.chunks:
            if type(chunk) is not ProjectedKnowledgeChunkV1:
                raise TypeError("exact projected knowledge chunks required")
            chunk.authenticate()
            if chunk.split != self.split:
                raise KnowledgeCellProjectionError("admitted split changed")
        self.graph.authenticate()
        expected_graph = _build_graph(
            self.records_sha256,
            self.split,
            self.admitted_chunk_groups,
            self.admitted_chunk_ids,
            self.chunks,
        )
        if self.graph != expected_graph:
            raise KnowledgeCellProjectionError("knowledge projection graph changed")
        self.producer_receipt.authenticate()
        attestation = _producer_attestation_sha256(
            records_sha256=self.records_sha256,
            split=self.split,
            admitted_chunk_groups=self.admitted_chunk_groups,
            admitted_chunk_ids=self.admitted_chunk_ids,
            chunks=self.chunks,
            graph_sha256=self.graph.sha256,
        )
        if (
            self.producer_receipt.source_sha256 != self.records_sha256
            or self.producer_receipt.graph_sha256 != self.graph.sha256
            or self.producer_receipt.producer_schema_sha256
            != _producer_schema_sha256()
            or self.producer_receipt.producer_attestation_sha256 != attestation
            or self.sha256
            != _projection_sha256(
                records_sha256=self.records_sha256,
                split=self.split,
                admitted_chunk_groups=self.admitted_chunk_groups,
                admitted_chunk_ids=self.admitted_chunk_ids,
                chunks=self.chunks,
                graph_sha256=self.graph.sha256,
                producer_receipt_sha256=self.producer_receipt.sha256,
            )
        ):
            raise KnowledgeCellProjectionError(
                "knowledge projection producer receipt authentication failed"
            )


def _authenticate_corpus(corpus: KnowledgeCorpus) -> None:
    try:
        corpus.authenticate()
    except KnowledgeCorpusError as error:
        raise KnowledgeCellProjectionError(
            "knowledge corpus authentication failed"
        ) from error


def project_knowledge_corpus_to_cells_v1(
    corpus: KnowledgeCorpus,
    *,
    admitted_chunk_groups: tuple[tuple[str, ...], ...],
    split: str,
) -> KnowledgeCellProjectionV1:
    """Project an explicit immutable corpus subset without task information."""

    _authenticate_corpus(corpus)
    if type(admitted_chunk_groups) is not tuple or any(
        type(group) is not tuple for group in admitted_chunk_groups
    ):
        raise TypeError("admitted chunk groups must be exact nested tuples")
    if not 1 <= len(admitted_chunk_groups) <= MAX_ADMITTED_GROUPS:
        raise KnowledgeCellProjectionError(
            f"admission must contain 1..{MAX_ADMITTED_GROUPS} groups"
        )
    supplied_ids = tuple(
        value for group in admitted_chunk_groups for value in group
    )
    if any(
        not 1 <= len(group) <= MAX_GROUP_MEMBERS
        for group in admitted_chunk_groups
    ):
        raise KnowledgeCellProjectionError(
            f"each group must contain 1..{MAX_GROUP_MEMBERS} chunks"
        )
    if any(not _valid_sha256(value) for value in supplied_ids):
        raise KnowledgeCellProjectionError("admitted chunk id is malformed")
    if len(set(supplied_ids)) != len(supplied_ids):
        raise KnowledgeCellProjectionError("admitted chunk groups overlap")
    if not 1 <= len(supplied_ids) <= MAX_ADMITTED_CHUNKS:
        raise KnowledgeCellProjectionError(
            f"admission must contain 1..{MAX_ADMITTED_CHUNKS} chunks"
        )
    if type(split) is not str or not split:
        raise TypeError("admitted split must be one non-empty exact string")
    by_id = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    missing = tuple(value for value in supplied_ids if value not in by_id)
    if missing:
        raise KnowledgeCellProjectionError(f"unknown admitted chunk ids: {missing}")
    selected_groups = tuple(
        tuple(by_id[value] for value in group) for group in admitted_chunk_groups
    )
    selected = tuple(record for group in selected_groups for record in group)
    if any(record.split != split for record in selected):
        raise KnowledgeCellProjectionError("admitted chunk is outside requested split")
    for group in selected_groups:
        _projected_chunk(group)
    canonical_group_pairs = tuple(
        sorted(
            zip(admitted_chunk_groups, selected_groups, strict=True),
            key=lambda pair: (
                pair[1][0].source_id,
                pair[1][0].path,
                pair[1][0].ordinal,
                pair[0],
            ),
        )
    )
    canonical_groups = tuple(pair[0] for pair in canonical_group_pairs)
    canonical_selected_groups = tuple(pair[1] for pair in canonical_group_pairs)
    canonical_ids = tuple(value for group in canonical_groups for value in group)
    total_bytes = sum(len(chunk.text.encode("utf-8")) for chunk in selected)
    if total_bytes > MAX_ADMITTED_UTF8_BYTES:
        raise KnowledgeCellProjectionError(
            f"admission exceeds {MAX_ADMITTED_UTF8_BYTES} UTF-8 bytes"
        )
    chunks = tuple(_projected_chunk(group) for group in canonical_selected_groups)
    graph = _build_graph(
        corpus.receipt.records_sha256,
        split,
        canonical_groups,
        canonical_ids,
        chunks,
    )
    attestation = _producer_attestation_sha256(
        records_sha256=corpus.receipt.records_sha256,
        split=split,
        admitted_chunk_groups=canonical_groups,
        admitted_chunk_ids=canonical_ids,
        chunks=chunks,
        graph_sha256=graph.sha256,
    )
    receipt = structured_graph.make_structured_event_graph_producer_receipt_v1(
        source_sha256=corpus.receipt.records_sha256,
        graph_sha256=graph.sha256,
        producer_schema_sha256=_producer_schema_sha256(),
        producer_attestation_sha256=attestation,
    )
    result = KnowledgeCellProjectionV1(
        records_sha256=corpus.receipt.records_sha256,
        split=split,
        admitted_chunk_groups=canonical_groups,
        admitted_chunk_ids=canonical_ids,
        chunks=chunks,
        graph=graph,
        producer_receipt=receipt,
        sha256=_projection_sha256(
            records_sha256=corpus.receipt.records_sha256,
            split=split,
            admitted_chunk_groups=canonical_groups,
            admitted_chunk_ids=canonical_ids,
            chunks=chunks,
            graph_sha256=graph.sha256,
            producer_receipt_sha256=receipt.sha256,
        ),
    )
    result.authenticate()
    return result


__all__ = [
    "MAX_ADMITTED_CHUNKS",
    "MAX_ADMITTED_GROUPS",
    "MAX_QUERY_FACTORS",
    "MAX_SOURCE_FACTORS_PER_GROUP",
    "SCHEMA",
    "KnowledgeCellProjectionError",
    "KnowledgeCellProjectionV1",
    "ProjectedKnowledgeChunkV1",
    "document_symbols_from_text_v1",
    "lexical_factors_from_text_v1",
    "project_knowledge_corpus_to_cells_v1",
]
