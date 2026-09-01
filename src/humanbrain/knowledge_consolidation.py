"""Deterministic provenance-bound lexical consolidation for knowledge corpora.

This is a host-side compression baseline, not evidence that simulated cells
learned abstractions.  It discovers recurring lexical factors from authenticated
record text, retains exact provenance, and exposes one task-independent seam.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from hashlib import sha256

from humanbrain.knowledge_corpus import (
    KnowledgeChunkRecord,
    KnowledgeCorpus,
    KnowledgeCorpusError,
)

SCHEMA = "humanbrain.knowledge-consolidation.v1"
LEXICAL_FACTOR_SCHEMA = "humanbrain.knowledge-lexical-factors.v1"
CLAIM_BOUNDARY = (
    "Deterministic provenance-bound host lexical compression baseline; "
    "not cell-learned abstraction."
)

MIN_FACTOR_SUPPORT = 2
MAX_SELECTED_FACTORS_PER_CARD = 128
MAX_SELECTED_CHUNKS = 4_096
MAX_GROUPS = 512
MAX_FACTOR_OBSERVATIONS = 1_000_000
MAX_FACTOR_MEMBER_REFERENCES = 1_000_000
MAX_CANONICAL_OUTPUT_BYTES = 32 * 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}")
_LEXICAL_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")

_MEMBER_NAMESPACE = b"knowledge-consolidation-member-v1"
_OBSERVATION_NAMESPACE = b"knowledge-consolidation-factor-observation-v1"
_FILE_NAMESPACE = b"knowledge-consolidation-file-v1"
_SCOPE_NAMESPACE = b"knowledge-consolidation-scope-v1"
_SOURCE_NAMESPACE = b"knowledge-consolidation-source-v1"
_SPLIT_NAMESPACE = b"knowledge-consolidation-split-v1"
_FACTOR_NAMESPACE = b"knowledge-consolidation-factor-v1"
_CARD_NAMESPACE = b"knowledge-consolidation-card-v1"
_CORPUS_NAMESPACE = b"knowledge-consolidation-corpus-v1"
_PRODUCER_NAMESPACE = b"knowledge-consolidation-producer-v1"
_POLICY_NAMESPACE = b"knowledge-consolidation-policy-v1"


class KnowledgeConsolidationError(ValueError):
    """The corpus or consolidation receipt failed integrity or capacity checks."""


def _canonical_json(value: object, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + suffix
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _record_digest(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def lexical_factors_from_text_v1(text: str) -> tuple[str, ...]:
    """Return every distinct ASCII identifier factor in canonical order."""

    if type(text) is not str:
        raise TypeError("knowledge text must be one exact string")
    factors: set[str] = set()
    for match in _LEXICAL_TOKEN.finditer(text):
        token = match.group(0).casefold()
        factors.add(f"tok:{token}")
        for part in re.split(r"[._]+", token):
            if part:
                factors.add(f"tok:{part}")
    return tuple(sorted(factors))


_POLICY = {
    "claim_boundary": CLAIM_BOUNDARY,
    "factor_cap": MAX_SELECTED_FACTORS_PER_CARD,
    "factor_source": "KnowledgeChunkRecord.text only",
    "grouping": [
        "source_id",
        "split",
        "chapter when present; otherwise discriminator-tagged path",
    ],
    "lexical_factor_schema": LEXICAL_FACTOR_SCHEMA,
    "lexical_pattern": _LEXICAL_TOKEN.pattern,
    "max_canonical_output_bytes": MAX_CANONICAL_OUTPUT_BYTES,
    "max_factor_member_references": MAX_FACTOR_MEMBER_REFERENCES,
    "max_factor_observations": MAX_FACTOR_OBSERVATIONS,
    "max_groups": MAX_GROUPS,
    "max_selected_chunks": MAX_SELECTED_CHUNKS,
    "minimum_support": MIN_FACTOR_SUPPORT,
    "ranking": ["-score", "-support", "scope_frequency", "factor"],
    "score": "support * (scope_count + 1 - scope_frequency)",
    "selected_factor_storage": "lexicographic after ranking cap",
    "support_unit": "distinct exact_dedup_cluster_sha256 within one scope",
}
POLICY_SHA256 = _digest(_POLICY_NAMESPACE, _POLICY)


@dataclass(frozen=True, slots=True)
class ConsolidatedKnowledgeMemberV1:
    chunk_id: str
    source_id: str
    path: str
    chapter: str | None
    split: str
    scope_kind: str
    scope_value: str
    ordinal: int
    exact_dedup_cluster_sha256: str
    declared_symbol: str | None
    record_sha256: str
    lexical_factors: tuple[str, ...]
    lexical_factors_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ConsolidatedKnowledgeFileV1:
    source_id: str
    path: str
    chapter: str | None
    split: str
    member_chunk_ids: tuple[str, ...]
    member_sha256s: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class ConsolidatedKnowledgeScopeV1:
    source_id: str
    split: str
    scope_kind: str
    scope_value: str
    file_sha256s: tuple[str, ...]
    member_chunk_ids: tuple[str, ...]
    member_sha256s: tuple[str, ...]
    declared_symbols: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class ConsolidatedKnowledgeSourceV1:
    source_id: str
    split: str
    canonical_url: str
    upstream_revision: str
    license_spdx: str
    license_evidence_url: str
    attribution_sha256: str
    provenance_confidence: float
    input_manifest_sha256: str
    adapter_version_sha256: str
    benchmark_deny_manifest_sha256: str
    file_sha256s: tuple[str, ...]
    scope_sha256s: tuple[str, ...]
    member_sha256s: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class ConsolidatedKnowledgeFactorV1:
    factor: str
    support: int
    scope_frequency: int
    score: int
    supporting_exact_dedup_cluster_sha256s: tuple[str, ...]
    supporting_member_chunk_ids: tuple[str, ...]
    supporting_member_sha256s: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class KnowledgeConceptCardV1:
    source_id: str
    split: str
    scope_kind: str
    scope_value: str
    policy_sha256: str
    source_sha256: str
    split_sha256: str
    scope_sha256: str
    file_sha256s: tuple[str, ...]
    member_chunk_ids: tuple[str, ...]
    member_sha256s: tuple[str, ...]
    declared_symbols: tuple[str, ...]
    selected_factors: tuple[str, ...]
    factors: tuple[ConsolidatedKnowledgeFactorV1, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class KnowledgeConsolidationV1:
    schema: str
    claim_boundary: str
    split: str
    input_records_sha256: str
    policy_sha256: str
    members: tuple[ConsolidatedKnowledgeMemberV1, ...]
    files: tuple[ConsolidatedKnowledgeFileV1, ...]
    groups: tuple[ConsolidatedKnowledgeScopeV1, ...]
    sources: tuple[ConsolidatedKnowledgeSourceV1, ...]
    split_sha256: str
    group_count: int
    scope_count: int
    card_count: int
    cards: tuple[KnowledgeConceptCardV1, ...]
    corpus_sha256: str
    producer_sha256: str

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json(asdict(self), newline=True)

    def authenticate_structure(self) -> None:
        """Verify canonical structure and digests without claiming source binding."""

        _authenticate_consolidation(self)

    def authenticate(self, corpus: KnowledgeCorpus) -> None:
        """Verify canonical structure and exact derivation from ``corpus``."""

        _authenticate_consolidation(self)
        _authenticate_consolidation_against_corpus(self, corpus)


def _scope_key(record: KnowledgeChunkRecord) -> tuple[str, str, str, str]:
    if record.chapter is not None:
        return (record.source_id, record.split, "chapter", record.chapter)
    return (record.source_id, record.split, "path", record.path)


def _member_payload(member: ConsolidatedKnowledgeMemberV1) -> dict[str, object]:
    payload = asdict(member)
    payload.pop("sha256")
    return payload


def _file_payload(file: ConsolidatedKnowledgeFileV1) -> dict[str, object]:
    payload = asdict(file)
    payload.pop("sha256")
    return payload


def _scope_payload(scope: ConsolidatedKnowledgeScopeV1) -> dict[str, object]:
    payload = asdict(scope)
    payload.pop("sha256")
    return payload


def _source_payload(source: ConsolidatedKnowledgeSourceV1) -> dict[str, object]:
    payload = asdict(source)
    payload.pop("sha256")
    return payload


def _factor_payload(factor: ConsolidatedKnowledgeFactorV1) -> dict[str, object]:
    payload = asdict(factor)
    payload.pop("sha256")
    return payload


def _card_payload(card: KnowledgeConceptCardV1) -> dict[str, object]:
    payload = asdict(card)
    payload.pop("sha256")
    return payload


def _split_payload(
    split: str,
    members: tuple[ConsolidatedKnowledgeMemberV1, ...],
    files: tuple[ConsolidatedKnowledgeFileV1, ...],
    groups: tuple[ConsolidatedKnowledgeScopeV1, ...],
    sources: tuple[ConsolidatedKnowledgeSourceV1, ...],
) -> dict[str, object]:
    return {
        "split": split,
        "member_sha256s": [member.sha256 for member in members],
        "file_sha256s": [file.sha256 for file in files],
        "group_sha256s": [group.sha256 for group in groups],
        "source_sha256s": [source.sha256 for source in sources],
    }


def _corpus_payload(result: KnowledgeConsolidationV1) -> dict[str, object]:
    return {
        "schema": result.schema,
        "claim_boundary": result.claim_boundary,
        "split": result.split,
        "input_records_sha256": result.input_records_sha256,
        "policy_sha256": result.policy_sha256,
        "split_sha256": result.split_sha256,
        "member_sha256s": [member.sha256 for member in result.members],
        "file_sha256s": [file.sha256 for file in result.files],
        "group_sha256s": [group.sha256 for group in result.groups],
        "source_sha256s": [source.sha256 for source in result.sources],
        "card_sha256s": [card.sha256 for card in result.cards],
        "group_count": result.group_count,
        "scope_count": result.scope_count,
        "card_count": result.card_count,
    }


def _producer_payload(result: KnowledgeConsolidationV1) -> dict[str, object]:
    return {
        "schema": result.schema,
        "claim_boundary": result.claim_boundary,
        "input_records_sha256": result.input_records_sha256,
        "policy_sha256": result.policy_sha256,
        "split_sha256": result.split_sha256,
        "corpus_sha256": result.corpus_sha256,
        "card_sha256s": [card.sha256 for card in result.cards],
    }


def _authenticate_corpus(corpus: KnowledgeCorpus) -> None:
    try:
        corpus.authenticate()
    except KnowledgeCorpusError as error:
        raise KnowledgeConsolidationError(
            "knowledge corpus authentication failed"
        ) from error


def _observation_payload(record_sha256: str, factors: tuple[str, ...]) -> dict[str, object]:
    return {
        "policy_sha256": POLICY_SHA256,
        "record_sha256": record_sha256,
        "lexical_factors": factors,
    }


def _make_member(
    record: KnowledgeChunkRecord, factors: tuple[str, ...]
) -> ConsolidatedKnowledgeMemberV1:
    scope = _scope_key(record)
    record_sha256 = _record_digest(asdict(record))
    draft = ConsolidatedKnowledgeMemberV1(
        chunk_id=record.chunk_id,
        source_id=record.source_id,
        path=record.path,
        chapter=record.chapter,
        split=record.split,
        scope_kind=scope[2],
        scope_value=scope[3],
        ordinal=record.ordinal,
        exact_dedup_cluster_sha256=record.exact_dedup_cluster_sha256,
        declared_symbol=record.symbol,
        record_sha256=record_sha256,
        lexical_factors=factors,
        lexical_factors_sha256=_digest(
            _OBSERVATION_NAMESPACE, _observation_payload(record_sha256, factors)
        ),
        sha256="",
    )
    return ConsolidatedKnowledgeMemberV1(
        **{**_member_payload(draft), "sha256": _digest(_MEMBER_NAMESPACE, _member_payload(draft))}
    )


def _metadata_tuple(record: KnowledgeChunkRecord) -> tuple[object, ...]:
    return (
        record.canonical_url,
        record.upstream_revision,
        record.license_spdx,
        record.license_evidence_url,
        record.attribution_sha256,
        float(record.provenance_confidence),
        record.input_manifest_sha256,
        record.adapter_version_sha256,
        record.benchmark_deny_manifest_sha256,
    )


def _factor_row(
    *,
    factor: str,
    support: int,
    scope_frequency: int,
    score: int,
    cluster_sha256s: tuple[str, ...],
    matching: tuple[ConsolidatedKnowledgeMemberV1, ...],
) -> ConsolidatedKnowledgeFactorV1:
    draft = ConsolidatedKnowledgeFactorV1(
        factor=factor,
        support=support,
        scope_frequency=scope_frequency,
        score=score,
        supporting_exact_dedup_cluster_sha256s=cluster_sha256s,
        supporting_member_chunk_ids=tuple(member.chunk_id for member in matching),
        supporting_member_sha256s=tuple(member.sha256 for member in matching),
        sha256="",
    )
    payload = _factor_payload(draft)
    return ConsolidatedKnowledgeFactorV1(
        **{**payload, "sha256": _digest(_FACTOR_NAMESPACE, payload)}
    )


def _authenticate_consolidation(result: KnowledgeConsolidationV1) -> None:
    if type(result) is not KnowledgeConsolidationV1:
        raise TypeError("exact immutable KnowledgeConsolidationV1 required")
    if (
        result.schema != SCHEMA
        or result.claim_boundary != CLAIM_BOUNDARY
        or result.policy_sha256 != POLICY_SHA256
    ):
        raise KnowledgeConsolidationError("consolidation policy identity differs")
    if type(result.split) is not str or not result.split:
        raise KnowledgeConsolidationError("consolidation split is invalid")
    tuple_fields = (result.members, result.files, result.groups, result.sources, result.cards)
    if any(type(values) is not tuple for values in tuple_fields):
        raise TypeError("consolidation collections must be exact tuples")
    if any(type(value) is not ConsolidatedKnowledgeMemberV1 for value in result.members):
        raise TypeError("consolidation members differ")
    if any(type(value) is not ConsolidatedKnowledgeFileV1 for value in result.files):
        raise TypeError("consolidation files differ")
    if any(type(value) is not ConsolidatedKnowledgeScopeV1 for value in result.groups):
        raise TypeError("consolidation groups differ")
    if any(type(value) is not ConsolidatedKnowledgeSourceV1 for value in result.sources):
        raise TypeError("consolidation sources differ")
    if any(type(value) is not KnowledgeConceptCardV1 for value in result.cards):
        raise TypeError("consolidation cards differ")
    if not all(
        (result.members, result.files, result.groups, result.sources, result.cards)
    ):
        raise KnowledgeConsolidationError("consolidation output is empty")
    if not _valid_sha256(result.input_records_sha256):
        raise KnowledgeConsolidationError("input records root is malformed")

    member_keys = tuple(
        (
            member.source_id,
            member.scope_kind,
            member.scope_value,
            member.path,
            member.ordinal,
            member.chunk_id,
        )
        for member in result.members
    )
    if member_keys != tuple(sorted(member_keys)):
        raise KnowledgeConsolidationError("consolidation member order differs")

    for member in result.members:
        if (
            member.split != result.split
            or member.scope_kind not in {"chapter", "path"}
            or not _valid_sha256(member.chunk_id)
            or not _valid_sha256(member.record_sha256)
            or not _valid_sha256(member.exact_dedup_cluster_sha256)
            or type(member.lexical_factors) is not tuple
            or member.lexical_factors != tuple(sorted(set(member.lexical_factors)))
            or member.lexical_factors_sha256
            != _digest(
                _OBSERVATION_NAMESPACE,
                _observation_payload(member.record_sha256, member.lexical_factors),
            )
            or member.sha256 != _digest(_MEMBER_NAMESPACE, _member_payload(member))
        ):
            raise KnowledgeConsolidationError("member receipt authentication failed")
    if len({member.chunk_id for member in result.members}) != len(result.members):
        raise KnowledgeConsolidationError("member chunk ids are not unique")
    members_by_id = {member.chunk_id: member for member in result.members}

    expected_files: dict[tuple[str, str, str], tuple[ConsolidatedKnowledgeMemberV1, ...]] = {}
    for key in sorted({(m.source_id, m.path, m.split) for m in result.members}):
        expected_files[key] = tuple(
            member
            for member in result.members
            if (member.source_id, member.path, member.split) == key
        )
    if len(result.files) != len(expected_files):
        raise KnowledgeConsolidationError("file receipt coverage differs")
    file_keys = tuple((file.source_id, file.path, file.split) for file in result.files)
    if file_keys != tuple(sorted(set(file_keys))):
        raise KnowledgeConsolidationError("file receipt order differs")
    files_by_root: dict[str, ConsolidatedKnowledgeFileV1] = {}
    for file in result.files:
        members = expected_files.get((file.source_id, file.path, file.split))
        if (
            members is None
            or file.member_chunk_ids != tuple(member.chunk_id for member in members)
            or file.member_sha256s != tuple(member.sha256 for member in members)
            or file.chapter != members[0].chapter
            or file.sha256 != _digest(_FILE_NAMESPACE, _file_payload(file))
        ):
            raise KnowledgeConsolidationError("file receipt authentication failed")
        files_by_root[file.sha256] = file
    if len(files_by_root) != len(result.files):
        raise KnowledgeConsolidationError("file receipt roots are not unique")

    expected_groups: dict[tuple[str, str, str, str], tuple[ConsolidatedKnowledgeMemberV1, ...]] = {}
    for key in sorted(
        {(m.source_id, m.split, m.scope_kind, m.scope_value) for m in result.members}
    ):
        expected_groups[key] = tuple(
            member
            for member in result.members
            if (member.source_id, member.split, member.scope_kind, member.scope_value) == key
        )
    if len(result.groups) != len(expected_groups):
        raise KnowledgeConsolidationError("scope receipt coverage differs")
    group_keys = tuple(
        (group.source_id, group.split, group.scope_kind, group.scope_value)
        for group in result.groups
    )
    if group_keys != tuple(sorted(set(group_keys))):
        raise KnowledgeConsolidationError("scope receipt order differs")
    groups_by_key: dict[tuple[str, str, str, str], ConsolidatedKnowledgeScopeV1] = {}
    for group in result.groups:
        key = (group.source_id, group.split, group.scope_kind, group.scope_value)
        members = expected_groups.get(key)
        expected_file_roots = tuple(
            file.sha256
            for file in result.files
            if file.source_id == group.source_id
            and any(member.path == file.path for member in members or ())
        )
        symbols = tuple(
            sorted({m.declared_symbol for m in members or () if m.declared_symbol is not None})
        )
        if (
            members is None
            or group.file_sha256s != expected_file_roots
            or group.member_chunk_ids != tuple(member.chunk_id for member in members)
            or group.member_sha256s != tuple(member.sha256 for member in members)
            or group.declared_symbols != symbols
            or group.sha256 != _digest(_SCOPE_NAMESPACE, _scope_payload(group))
        ):
            raise KnowledgeConsolidationError("scope receipt authentication failed")
        groups_by_key[key] = group

    source_ids = tuple(source.source_id for source in result.sources)
    if source_ids != tuple(sorted(set(source_ids))):
        raise KnowledgeConsolidationError("source receipt order differs")
    sources_by_id: dict[str, ConsolidatedKnowledgeSourceV1] = {}
    for source in result.sources:
        source_members = tuple(m for m in result.members if m.source_id == source.source_id)
        expected_file_roots = tuple(
            f.sha256 for f in result.files if f.source_id == source.source_id
        )
        expected_group_roots = tuple(
            g.sha256 for g in result.groups if g.source_id == source.source_id
        )
        if (
            not source_members
            or source.split != result.split
            or source.file_sha256s != expected_file_roots
            or source.scope_sha256s != expected_group_roots
            or source.member_sha256s != tuple(m.sha256 for m in source_members)
            or source.sha256 != _digest(_SOURCE_NAMESPACE, _source_payload(source))
        ):
            raise KnowledgeConsolidationError("source receipt authentication failed")
        sources_by_id[source.source_id] = source
    if set(sources_by_id) != {member.source_id for member in result.members}:
        raise KnowledgeConsolidationError("source receipt coverage differs")

    expected_split = _digest(
        _SPLIT_NAMESPACE,
        _split_payload(result.split, result.members, result.files, result.groups, result.sources),
    )
    if result.split_sha256 != expected_split:
        raise KnowledgeConsolidationError("split receipt authentication failed")

    seen_card_keys: set[tuple[str, str, str, str]] = set()
    expected_factor_clusters: dict[tuple[str, str, str, str], dict[str, set[str]]] = {}
    expected_factor_members: dict[
        tuple[str, str, str, str],
        dict[str, list[ConsolidatedKnowledgeMemberV1]],
    ] = {}
    for key, scope_members in expected_groups.items():
        clusters: dict[str, set[str]] = defaultdict(set)
        matching: dict[str, list[ConsolidatedKnowledgeMemberV1]] = defaultdict(list)
        for member in scope_members:
            for factor in member.lexical_factors:
                clusters[factor].add(member.exact_dedup_cluster_sha256)
                matching[factor].append(member)
        expected_factor_clusters[key] = clusters
        expected_factor_members[key] = matching
    expected_frequency: dict[str, int] = defaultdict(int)
    for clusters in expected_factor_clusters.values():
        for factor in clusters:
            expected_frequency[factor] += 1
    expected_active_keys = tuple(
        key
        for key in expected_groups
        if any(
            len(cluster_roots) >= MIN_FACTOR_SUPPORT
            for cluster_roots in expected_factor_clusters[key].values()
        )
    )
    for card in result.cards:
        key = (card.source_id, card.split, card.scope_kind, card.scope_value)
        group = groups_by_key.get(key)
        source = sources_by_id.get(card.source_id)
        if group is None or source is None or key in seen_card_keys:
            raise KnowledgeConsolidationError("card scope/source binding differs")
        seen_card_keys.add(key)
        if type(card.factors) is not tuple or any(
            type(factor) is not ConsolidatedKnowledgeFactorV1 for factor in card.factors
        ):
            raise TypeError("card factors differ")
        for factor in card.factors:
            if (
                factor.support < MIN_FACTOR_SUPPORT
                or factor.scope_frequency < 1
                or factor.score
                != factor.support * (result.scope_count + 1 - factor.scope_frequency)
                or factor.support != len(factor.supporting_exact_dedup_cluster_sha256s)
                or len(factor.supporting_member_chunk_ids) != len(factor.supporting_member_sha256s)
                or any(
                    chunk_id not in set(group.member_chunk_ids)
                    for chunk_id in factor.supporting_member_chunk_ids
                )
                or tuple(
                    members_by_id[value].sha256 for value in factor.supporting_member_chunk_ids
                )
                != factor.supporting_member_sha256s
                or factor.sha256 != _digest(_FACTOR_NAMESPACE, _factor_payload(factor))
            ):
                raise KnowledgeConsolidationError("factor receipt authentication failed")
        factor_names = tuple(factor.factor for factor in card.factors)
        expected_ranked = sorted(
            (
                (
                    factor,
                    len(cluster_roots),
                    expected_frequency[factor],
                    len(cluster_roots) * (result.scope_count + 1 - expected_frequency[factor]),
                )
                for factor, cluster_roots in expected_factor_clusters[key].items()
                if len(cluster_roots) >= MIN_FACTOR_SUPPORT
            ),
            key=lambda row: (-row[3], -row[1], row[2], row[0]),
        )[:MAX_SELECTED_FACTORS_PER_CARD]
        expected_names = tuple(sorted(row[0] for row in expected_ranked))
        expected_stats = {row[0]: row[1:] for row in expected_ranked}
        expected_factors = tuple(
            _factor_row(
                factor=factor,
                support=expected_stats[factor][0],
                scope_frequency=expected_stats[factor][1],
                score=expected_stats[factor][2],
                cluster_sha256s=tuple(sorted(expected_factor_clusters[key][factor])),
                matching=tuple(expected_factor_members[key][factor]),
            )
            for factor in expected_names
        )
        if (
            not factor_names
            or factor_names != tuple(sorted(factor_names))
            or card.selected_factors != factor_names
            or card.policy_sha256 != result.policy_sha256
            or card.source_sha256 != source.sha256
            or card.split_sha256 != result.split_sha256
            or card.scope_sha256 != group.sha256
            or card.file_sha256s != group.file_sha256s
            or card.member_chunk_ids != group.member_chunk_ids
            or card.member_sha256s != group.member_sha256s
            or card.declared_symbols != group.declared_symbols
            or card.factors != expected_factors
            or card.selected_factors != expected_names
            or card.sha256 != _digest(_CARD_NAMESPACE, _card_payload(card))
        ):
            raise KnowledgeConsolidationError("concept card authentication failed")

    if (
        result.group_count != len(result.groups)
        or result.scope_count != len(result.groups)
        or result.card_count != len(result.cards)
        or result.group_count > MAX_GROUPS
        or len(result.members) > MAX_SELECTED_CHUNKS
        or tuple(
            (card.source_id, card.split, card.scope_kind, card.scope_value) for card in result.cards
        )
        != expected_active_keys
    ):
        raise KnowledgeConsolidationError("consolidation counts differ")
    if result.corpus_sha256 != _digest(_CORPUS_NAMESPACE, _corpus_payload(result)):
        raise KnowledgeConsolidationError("consolidated corpus root differs")
    if result.producer_sha256 != _digest(_PRODUCER_NAMESPACE, _producer_payload(result)):
        raise KnowledgeConsolidationError("consolidation producer root differs")
    if len(result.canonical_json_bytes()) > MAX_CANONICAL_OUTPUT_BYTES:
        raise KnowledgeConsolidationError("consolidation output exceeds hard byte cap")


def _records_for_split(
    corpus: KnowledgeCorpus, split: str
) -> tuple[KnowledgeChunkRecord, ...]:
    return tuple(
        sorted(
            (record for record in corpus.chunks if record.split == split),
            key=lambda record: (
                record.source_id,
                *_scope_key(record)[2:],
                record.path,
                record.ordinal,
                record.chunk_id,
            ),
        )
    )


def _authenticate_consolidation_against_corpus(
    result: KnowledgeConsolidationV1,
    corpus: KnowledgeCorpus,
) -> None:
    _authenticate_corpus(corpus)
    if result.input_records_sha256 != corpus.receipt.records_sha256:
        raise KnowledgeConsolidationError("consolidation input corpus differs")

    records = _records_for_split(corpus, result.split)
    expected_members = tuple(
        _make_member(record, lexical_factors_from_text_v1(record.text))
        for record in records
    )
    if not records or result.members != expected_members:
        raise KnowledgeConsolidationError("consolidation corpus members differ")

    for source in result.sources:
        source_records = tuple(
            record for record in records if record.source_id == source.source_id
        )
        if not source_records:
            raise KnowledgeConsolidationError("consolidation source provenance differs")
        expected_metadata = _metadata_tuple(source_records[0])
        source_metadata = (
            source.canonical_url,
            source.upstream_revision,
            source.license_spdx,
            source.license_evidence_url,
            source.attribution_sha256,
            source.provenance_confidence,
            source.input_manifest_sha256,
            source.adapter_version_sha256,
            source.benchmark_deny_manifest_sha256,
        )
        if source_metadata != expected_metadata:
            raise KnowledgeConsolidationError("consolidation source provenance differs")


def consolidate_knowledge_corpus_v1(
    corpus: KnowledgeCorpus,
    *,
    split: str,
) -> KnowledgeConsolidationV1:
    """Compress recurring text factors from every record in one authenticated split."""

    _authenticate_corpus(corpus)
    if type(split) is not str or not split:
        raise TypeError("split must be one non-empty exact string")
    records = _records_for_split(corpus, split)
    if not records:
        raise KnowledgeConsolidationError("requested split has no knowledge records")
    if len(records) > MAX_SELECTED_CHUNKS:
        raise KnowledgeConsolidationError("selected split exceeds chunk capacity")
    scope_keys = tuple(sorted({_scope_key(record) for record in records}))
    if len(scope_keys) > MAX_GROUPS:
        raise KnowledgeConsolidationError("selected split exceeds scope capacity")

    factors_by_chunk_id: dict[str, tuple[str, ...]] = {}
    observations = 0
    for record in records:
        factors = lexical_factors_from_text_v1(record.text)
        observations += len(factors)
        if observations > MAX_FACTOR_OBSERVATIONS:
            raise KnowledgeConsolidationError("lexical observations exceed hard capacity")
        factors_by_chunk_id[record.chunk_id] = factors
    members = tuple(
        _make_member(record, factors_by_chunk_id[record.chunk_id]) for record in records
    )
    member_by_id = {member.chunk_id: member for member in members}

    files_list: list[ConsolidatedKnowledgeFileV1] = []
    for source_id, path in sorted({(record.source_id, record.path) for record in records}):
        selected = tuple(
            member for member in members if member.source_id == source_id and member.path == path
        )
        chapters = {member.chapter for member in selected}
        if len(chapters) != 1:
            raise KnowledgeConsolidationError("one source path has inconsistent chapters")
        draft = ConsolidatedKnowledgeFileV1(
            source_id=source_id,
            path=path,
            chapter=selected[0].chapter,
            split=split,
            member_chunk_ids=tuple(member.chunk_id for member in selected),
            member_sha256s=tuple(member.sha256 for member in selected),
            sha256="",
        )
        payload = _file_payload(draft)
        files_list.append(
            ConsolidatedKnowledgeFileV1(**{**payload, "sha256": _digest(_FILE_NAMESPACE, payload)})
        )
    files = tuple(files_list)

    groups_list: list[ConsolidatedKnowledgeScopeV1] = []
    for source_id, scope_split, scope_kind, scope_value in scope_keys:
        selected = tuple(
            member
            for member in members
            if (member.source_id, member.split, member.scope_kind, member.scope_value)
            == (source_id, scope_split, scope_kind, scope_value)
        )
        paths = {member.path for member in selected}
        file_roots = tuple(
            file.sha256 for file in files if file.source_id == source_id and file.path in paths
        )
        symbols = tuple(
            sorted(
                {
                    member.declared_symbol
                    for member in selected
                    if member.declared_symbol is not None
                }
            )
        )
        draft = ConsolidatedKnowledgeScopeV1(
            source_id=source_id,
            split=scope_split,
            scope_kind=scope_kind,
            scope_value=scope_value,
            file_sha256s=file_roots,
            member_chunk_ids=tuple(member.chunk_id for member in selected),
            member_sha256s=tuple(member.sha256 for member in selected),
            declared_symbols=symbols,
            sha256="",
        )
        payload = _scope_payload(draft)
        groups_list.append(
            ConsolidatedKnowledgeScopeV1(
                **{**payload, "sha256": _digest(_SCOPE_NAMESPACE, payload)}
            )
        )
    groups = tuple(groups_list)

    sources_list: list[ConsolidatedKnowledgeSourceV1] = []
    for source_id in sorted({record.source_id for record in records}):
        source_records = tuple(record for record in records if record.source_id == source_id)
        metadata = {_metadata_tuple(record) for record in source_records}
        if len(metadata) != 1:
            raise KnowledgeConsolidationError("selected source metadata is inconsistent")
        values = next(iter(metadata))
        draft = ConsolidatedKnowledgeSourceV1(
            source_id=source_id,
            split=split,
            canonical_url=str(values[0]),
            upstream_revision=str(values[1]),
            license_spdx=str(values[2]),
            license_evidence_url=str(values[3]),
            attribution_sha256=str(values[4]),
            provenance_confidence=float(values[5]),
            input_manifest_sha256=str(values[6]),
            adapter_version_sha256=str(values[7]),
            benchmark_deny_manifest_sha256=str(values[8]),
            file_sha256s=tuple(file.sha256 for file in files if file.source_id == source_id),
            scope_sha256s=tuple(group.sha256 for group in groups if group.source_id == source_id),
            member_sha256s=tuple(
                member.sha256 for member in members if member.source_id == source_id
            ),
            sha256="",
        )
        payload = _source_payload(draft)
        sources_list.append(
            ConsolidatedKnowledgeSourceV1(
                **{**payload, "sha256": _digest(_SOURCE_NAMESPACE, payload)}
            )
        )
    sources = tuple(sources_list)
    split_sha256 = _digest(_SPLIT_NAMESPACE, _split_payload(split, members, files, groups, sources))

    scope_factor_clusters: dict[tuple[str, str, str, str], dict[str, set[str]]] = {}
    scope_factor_members: dict[
        tuple[str, str, str, str], dict[str, list[ConsolidatedKnowledgeMemberV1]]
    ] = {}
    member_references = 0
    for key in scope_keys:
        factor_clusters: dict[str, set[str]] = defaultdict(set)
        factor_members: dict[str, list[ConsolidatedKnowledgeMemberV1]] = defaultdict(list)
        for record in records:
            if _scope_key(record) != key:
                continue
            member = member_by_id[record.chunk_id]
            factors = factors_by_chunk_id[record.chunk_id]
            for factor in factors:
                factor_clusters[factor].add(record.exact_dedup_cluster_sha256)
                factor_members[factor].append(member)
        scope_factor_clusters[key] = factor_clusters
        scope_factor_members[key] = factor_members

    recurring = {
        key: {
            factor: clusters
            for factor, clusters in factor_clusters.items()
            if len(clusters) >= MIN_FACTOR_SUPPORT
        }
        for key, factor_clusters in scope_factor_clusters.items()
    }
    active_keys = tuple(key for key in scope_keys if recurring[key])
    scope_count = len(scope_keys)
    scope_frequency: dict[str, int] = defaultdict(int)
    for key in scope_keys:
        for factor in scope_factor_clusters[key]:
            scope_frequency[factor] += 1

    group_by_key = {
        (group.source_id, group.split, group.scope_kind, group.scope_value): group
        for group in groups
    }
    source_by_id = {source.source_id: source for source in sources}
    cards_list: list[KnowledgeConceptCardV1] = []
    for key in active_keys:
        ranked = sorted(
            (
                (
                    factor,
                    len(clusters),
                    scope_frequency[factor],
                    len(clusters) * (scope_count + 1 - scope_frequency[factor]),
                )
                for factor, clusters in recurring[key].items()
            ),
            key=lambda row: (-row[3], -row[1], row[2], row[0]),
        )[:MAX_SELECTED_FACTORS_PER_CARD]
        selected_names = tuple(sorted(row[0] for row in ranked))
        stats = {row[0]: row[1:] for row in ranked}
        factors_list: list[ConsolidatedKnowledgeFactorV1] = []
        for factor in selected_names:
            support, frequency, score = stats[factor]
            matching = tuple(scope_factor_members[key][factor])
            member_references += len(matching)
            if member_references > MAX_FACTOR_MEMBER_REFERENCES:
                raise KnowledgeConsolidationError("factor provenance exceeds hard capacity")
            factors_list.append(
                _factor_row(
                    factor=factor,
                    support=support,
                    scope_frequency=frequency,
                    score=score,
                    cluster_sha256s=tuple(sorted(recurring[key][factor])),
                    matching=matching,
                )
            )
        group = group_by_key[key]
        source = source_by_id[key[0]]
        draft = KnowledgeConceptCardV1(
            source_id=key[0],
            split=key[1],
            scope_kind=key[2],
            scope_value=key[3],
            policy_sha256=POLICY_SHA256,
            source_sha256=source.sha256,
            split_sha256=split_sha256,
            scope_sha256=group.sha256,
            file_sha256s=group.file_sha256s,
            member_chunk_ids=group.member_chunk_ids,
            member_sha256s=group.member_sha256s,
            declared_symbols=group.declared_symbols,
            selected_factors=selected_names,
            factors=tuple(factors_list),
            sha256="",
        )
        payload = _card_payload(draft)
        cards_list.append(replace(draft, sha256=_digest(_CARD_NAMESPACE, payload)))
    cards = tuple(cards_list)

    draft_result = KnowledgeConsolidationV1(
        schema=SCHEMA,
        claim_boundary=CLAIM_BOUNDARY,
        split=split,
        input_records_sha256=corpus.receipt.records_sha256,
        policy_sha256=POLICY_SHA256,
        members=members,
        files=files,
        groups=groups,
        sources=sources,
        split_sha256=split_sha256,
        group_count=len(groups),
        scope_count=scope_count,
        card_count=len(cards),
        cards=cards,
        corpus_sha256="",
        producer_sha256="",
    )
    corpus_sha256 = _digest(_CORPUS_NAMESPACE, _corpus_payload(draft_result))
    with_corpus = replace(draft_result, corpus_sha256=corpus_sha256)
    producer_sha256 = _digest(_PRODUCER_NAMESPACE, _producer_payload(with_corpus))
    result = replace(with_corpus, producer_sha256=producer_sha256)
    result.authenticate(corpus)
    return result


__all__ = [
    "CLAIM_BOUNDARY",
    "LEXICAL_FACTOR_SCHEMA",
    "MAX_SELECTED_FACTORS_PER_CARD",
    "MIN_FACTOR_SUPPORT",
    "POLICY_SHA256",
    "SCHEMA",
    "ConsolidatedKnowledgeFactorV1",
    "ConsolidatedKnowledgeFileV1",
    "ConsolidatedKnowledgeMemberV1",
    "ConsolidatedKnowledgeScopeV1",
    "ConsolidatedKnowledgeSourceV1",
    "KnowledgeConceptCardV1",
    "KnowledgeConsolidationError",
    "KnowledgeConsolidationV1",
    "consolidate_knowledge_corpus_v1",
    "lexical_factors_from_text_v1",
]
