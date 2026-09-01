from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from hashlib import sha256

from experiments.r3_pilot.common_continual_token_edit_schema_v1 import (
    TokenSourceFrameV1,
    project_token_source_v1,
)

SCHEMA = "common-preceding-let-declaration-occurrences-v1"
CLAIM_BOUNDARY = (
    "a bounded environment adapter losslessly tokenizes one supplied ASCII source and "
    "enumerates its complete exact-two set of simple preceding let-declaration identifier "
    "occurrences relative to one supplied primary identifier span; tokenization, declaration "
    "grammar, the exact-two policy, primary span, and farther/nearer relation ontology are "
    "supplied and are not learned; not target correctness, semantic localization, diagnosis, "
    "source synthesis, debugging, transfer, SWE performance, or intelligence"
)

FARTHER_PRECEDING_DECLARATION = "FARTHER_PRECEDING_DECLARATION"
NEARER_PRECEDING_DECLARATION = "NEARER_PRECEDING_DECLARATION"
PRECEDING_DECLARATION_RELATIONS = (
    FARTHER_PRECEDING_DECLARATION,
    NEARER_PRECEDING_DECLARATION,
)

_RELATION_COMPONENT_NAMESPACE = b"common-preceding-declaration-relation-component-v1"
_OCCURRENCE_NAMESPACE = b"common-preceding-let-declaration-occurrence-v1"
_ROSTER_NAMESPACE = b"common-preceding-let-declaration-roster-v1"
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class PrecedingLetDeclarationOccurrenceError(ValueError):
    pass


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256_HEX.fullmatch(value) is not None


def _ascii_source_bytes(source: str) -> bytes:
    if type(source) is not str:
        raise TypeError("preceding-declaration source must be one exact string")
    try:
        return source.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise PrecedingLetDeclarationOccurrenceError(
            "preceding-declaration source must be bounded ASCII"
        ) from exc


def preceding_declaration_relation_component_sha256_v1(relation: str) -> str:
    """Root one canonical relation without source, span, identifier, or outcome context."""

    if type(relation) is not str or relation not in PRECEDING_DECLARATION_RELATIONS:
        raise PrecedingLetDeclarationOccurrenceError(
            "preceding-declaration relation is not canonical"
        )
    return _digest(_RELATION_COMPONENT_NAMESPACE, {"relation": relation})


def _token_byte_spans(
    frame: TokenSourceFrameV1,
) -> tuple[tuple[int, int], ...]:
    if type(frame) is not TokenSourceFrameV1:
        raise TypeError("preceding-declaration token frame must be exact")
    frame.authenticate()
    source_bytes = _ascii_source_bytes(frame.source)
    if len(frame.gaps) != len(frame.tokens) + 1:
        raise PrecedingLetDeclarationOccurrenceError(
            "preceding-declaration token/gap cardinality failed"
        )

    cursor = 0
    spans: list[tuple[int, int]] = []
    for gap, token in zip(frame.gaps[:-1], frame.tokens, strict=True):
        gap_bytes = gap.encode("ascii", errors="strict")
        token_bytes = token.encode("ascii", errors="strict")
        cursor += len(gap_bytes)
        token_start = cursor
        token_end = token_start + len(token_bytes)
        if source_bytes[token_start:token_end] != token_bytes:
            raise PrecedingLetDeclarationOccurrenceError(
                "preceding-declaration token byte span lost source custody"
            )
        spans.append((token_start, token_end))
        cursor = token_end
    cursor += len(frame.gaps[-1].encode("ascii", errors="strict"))
    if cursor != len(source_bytes):
        raise PrecedingLetDeclarationOccurrenceError(
            "preceding-declaration token gaps do not cover the source"
        )
    return tuple(spans)


def _line_span(source: bytes, *, byte_start: int, byte_end: int) -> tuple[int, int]:
    line_start = max(
        source.rfind(b"\n", 0, byte_start),
        source.rfind(b"\r", 0, byte_start),
    ) + 1
    line_ends = tuple(
        position
        for position in (
            source.find(b"\n", byte_end),
            source.find(b"\r", byte_end),
        )
        if position >= 0
    )
    line_end = min(line_ends) if line_ends else len(source)
    if not line_start <= byte_start < byte_end <= line_end:
        raise PrecedingLetDeclarationOccurrenceError(
            "preceding-declaration identifier is not contained by one line"
        )
    return line_start, line_end


@dataclass(frozen=True, slots=True)
class PrecedingLetDeclarationOccurrenceV1:
    source_sha256: str
    source_size_bytes: int
    token_frame_sha256: str
    primary_span_sha256: str
    identifier_token_index: int
    identifier_bytes: bytes
    identifier_sha256: str
    identifier_byte_start: int
    identifier_byte_end: int
    distance_to_primary_bytes: int
    declaration_line_bytes: bytes
    declaration_line_byte_start: int
    declaration_line_byte_end: int
    declaration_line_sha256: str
    relation: str
    relation_component_sha256: str
    sha256: str

    @property
    def identifier(self) -> str:
        return self.identifier_bytes.decode("ascii")

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": SCHEMA,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "token_frame_sha256": self.token_frame_sha256,
            "primary_span_sha256": self.primary_span_sha256,
            "identifier_token_index": self.identifier_token_index,
            "identifier_hex": self.identifier_bytes.hex(),
            "identifier_sha256": self.identifier_sha256,
            "identifier_byte_start": self.identifier_byte_start,
            "identifier_byte_end": self.identifier_byte_end,
            "distance_to_primary_bytes": self.distance_to_primary_bytes,
            "declaration_line_hex": self.declaration_line_bytes.hex(),
            "declaration_line_byte_start": self.declaration_line_byte_start,
            "declaration_line_byte_end": self.declaration_line_byte_end,
            "declaration_line_sha256": self.declaration_line_sha256,
            "relation": self.relation,
            "relation_component_sha256": self.relation_component_sha256,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(
        self,
        *,
        source: str,
        token_frame: TokenSourceFrameV1,
        primary_byte_start: int,
        primary_byte_end: int,
    ) -> None:
        source_bytes = _ascii_source_bytes(source)
        if type(token_frame) is not TokenSourceFrameV1:
            raise TypeError("preceding-declaration token frame must be exact")
        token_frame.authenticate()
        token_spans = _token_byte_spans(token_frame)
        if (
            type(primary_byte_start) is not int
            or type(primary_byte_end) is not int
            or not 0 <= primary_byte_start < primary_byte_end <= len(source_bytes)
        ):
            raise PrecedingLetDeclarationOccurrenceError(
                "preceding-declaration primary span is invalid"
            )
        if (
            type(self.source_size_bytes) is not int
            or type(self.identifier_token_index) is not int
            or type(self.identifier_bytes) is not bytes
            or type(self.identifier_byte_start) is not int
            or type(self.identifier_byte_end) is not int
            or type(self.distance_to_primary_bytes) is not int
            or type(self.declaration_line_bytes) is not bytes
            or type(self.declaration_line_byte_start) is not int
            or type(self.declaration_line_byte_end) is not int
            or not all(
                _is_sha256(root)
                for root in (
                    self.source_sha256,
                    self.token_frame_sha256,
                    self.primary_span_sha256,
                    self.identifier_sha256,
                    self.declaration_line_sha256,
                    self.relation_component_sha256,
                    self.sha256,
                )
            )
            or not 0 <= self.identifier_token_index < len(token_spans)
        ):
            raise PrecedingLetDeclarationOccurrenceError(
                "preceding-declaration occurrence fields are not exact"
            )

        identifier_span = token_spans[self.identifier_token_index]
        line_span = _line_span(
            source_bytes,
            byte_start=self.identifier_byte_start,
            byte_end=self.identifier_byte_end,
        )
        if (
            token_frame.source != source
            or self.source_sha256 != sha256(source_bytes).hexdigest()
            or self.source_size_bytes != len(source_bytes)
            or self.token_frame_sha256 != token_frame.sha256
            or self.primary_span_sha256
            != sha256(source_bytes[primary_byte_start:primary_byte_end]).hexdigest()
            or token_frame.kinds[self.identifier_token_index] != "IDENTIFIER"
            or identifier_span != (self.identifier_byte_start, self.identifier_byte_end)
            or self.identifier_bytes
            != source_bytes[self.identifier_byte_start : self.identifier_byte_end]
            or self.identifier_sha256 != sha256(self.identifier_bytes).hexdigest()
            or self.identifier_byte_end > primary_byte_start
            or self.distance_to_primary_bytes
            != primary_byte_start - self.identifier_byte_end
            or line_span
            != (self.declaration_line_byte_start, self.declaration_line_byte_end)
            or self.declaration_line_bytes
            != source_bytes[
                self.declaration_line_byte_start : self.declaration_line_byte_end
            ]
            or self.declaration_line_sha256
            != sha256(self.declaration_line_bytes).hexdigest()
            or self.relation not in PRECEDING_DECLARATION_RELATIONS
            or self.relation_component_sha256
            != preceding_declaration_relation_component_sha256_v1(self.relation)
            or self.sha256 != _digest(_OCCURRENCE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise PrecedingLetDeclarationOccurrenceError(
                "preceding-declaration occurrence authentication failed"
            )


@dataclass(frozen=True, slots=True)
class PrecedingLetDeclarationRosterV1:
    source: str
    source_sha256: str
    source_size_bytes: int
    token_frame: TokenSourceFrameV1
    primary_byte_start: int
    primary_byte_end: int
    primary_span_bytes: bytes
    primary_span_sha256: str
    occurrences: tuple[PrecedingLetDeclarationOccurrenceV1, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": SCHEMA,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "token_frame_sha256": self.token_frame.sha256,
            "primary_byte_start": self.primary_byte_start,
            "primary_byte_end": self.primary_byte_end,
            "primary_span_hex": self.primary_span_bytes.hex(),
            "primary_span_sha256": self.primary_span_sha256,
            "occurrences": tuple(occurrence.record() for occurrence in self.occurrences),
            "complete_eligible_occurrence_count": len(self.occurrences),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        projected = _project_preceding_let_declaration_occurrences_v1(
            self.source,
            primary_byte_start=self.primary_byte_start,
            primary_byte_end=self.primary_byte_end,
            authenticate=False,
        )
        if self != projected:
            raise PrecedingLetDeclarationOccurrenceError(
                "preceding-declaration roster authentication failed"
            )
        for occurrence in self.occurrences:
            occurrence.authenticate(
                source=self.source,
                token_frame=self.token_frame,
                primary_byte_start=self.primary_byte_start,
                primary_byte_end=self.primary_byte_end,
            )


@dataclass(frozen=True, slots=True)
class _UnrelatedOccurrence:
    identifier_token_index: int
    identifier_bytes: bytes
    identifier_byte_start: int
    identifier_byte_end: int
    declaration_line_bytes: bytes
    declaration_line_byte_start: int
    declaration_line_byte_end: int


def _preceding_simple_let_declarations(
    *,
    source: bytes,
    frame: TokenSourceFrameV1,
    token_spans: tuple[tuple[int, int], ...],
    primary_byte_start: int,
    primary_identifier_bytes: bytes,
) -> tuple[_UnrelatedOccurrence, ...]:
    eligible: list[_UnrelatedOccurrence] = []
    for let_index, (token, kind, let_span) in enumerate(
        zip(frame.tokens, frame.kinds, token_spans, strict=True)
    ):
        if token != "let" or kind != "KEYWORD" or let_span[1] > primary_byte_start:
            continue
        if let_index + 3 >= len(frame.tokens):
            raise PrecedingLetDeclarationOccurrenceError(
                "preceding let declaration is malformed or unterminated"
            )

        identifier_index = let_index + 1
        equal_index = let_index + 2
        identifier_span = token_spans[identifier_index]
        line_start, line_end = _line_span(
            source,
            byte_start=identifier_span[0],
            byte_end=identifier_span[1],
        )
        if (
            frame.kinds[identifier_index] != "IDENTIFIER"
            or frame.tokens[equal_index] != "="
            or frame.kinds[equal_index] != "ASSIGNMENT"
            or not line_start <= let_span[0] < token_spans[equal_index][1] <= line_end
            or source[line_start : let_span[0]].strip()
        ):
            raise PrecedingLetDeclarationOccurrenceError(
                "preceding let declaration is not a simple standalone binding"
            )

        semicolon_index: int | None = None
        for candidate_index in range(let_index + 3, len(frame.tokens)):
            candidate_start, candidate_end = token_spans[candidate_index]
            if candidate_start >= line_end:
                break
            if frame.tokens[candidate_index] == ";":
                semicolon_index = candidate_index
                break
            if candidate_end > line_end:
                break
        if semicolon_index is None or semicolon_index == let_index + 3:
            raise PrecedingLetDeclarationOccurrenceError(
                "preceding let declaration is malformed or unterminated"
            )
        semicolon_end = token_spans[semicolon_index][1]
        if source[semicolon_end:line_end].strip():
            raise PrecedingLetDeclarationOccurrenceError(
                "preceding let declaration line contains trailing syntax"
            )

        identifier_bytes = source[identifier_span[0] : identifier_span[1]]
        if identifier_bytes == primary_identifier_bytes:
            continue
        if line_end >= len(source) or source[line_end] not in (0x0A, 0x0D):
            raise PrecedingLetDeclarationOccurrenceError(
                "preceding let declaration candidate line is not newline-terminated"
            )
        eligible.append(
            _UnrelatedOccurrence(
                identifier_token_index=identifier_index,
                identifier_bytes=identifier_bytes,
                identifier_byte_start=identifier_span[0],
                identifier_byte_end=identifier_span[1],
                declaration_line_bytes=source[line_start:line_end],
                declaration_line_byte_start=line_start,
                declaration_line_byte_end=line_end,
            )
        )
    return tuple(sorted(eligible, key=lambda item: item.identifier_byte_start))


def _seal_occurrence(
    *,
    source: bytes,
    frame: TokenSourceFrameV1,
    primary_byte_start: int,
    primary_span_sha256: str,
    candidate: _UnrelatedOccurrence,
    relation: str,
) -> PrecedingLetDeclarationOccurrenceV1:
    partial = PrecedingLetDeclarationOccurrenceV1(
        source_sha256=sha256(source).hexdigest(),
        source_size_bytes=len(source),
        token_frame_sha256=frame.sha256,
        primary_span_sha256=primary_span_sha256,
        identifier_token_index=candidate.identifier_token_index,
        identifier_bytes=candidate.identifier_bytes,
        identifier_sha256=sha256(candidate.identifier_bytes).hexdigest(),
        identifier_byte_start=candidate.identifier_byte_start,
        identifier_byte_end=candidate.identifier_byte_end,
        distance_to_primary_bytes=primary_byte_start - candidate.identifier_byte_end,
        declaration_line_bytes=candidate.declaration_line_bytes,
        declaration_line_byte_start=candidate.declaration_line_byte_start,
        declaration_line_byte_end=candidate.declaration_line_byte_end,
        declaration_line_sha256=sha256(candidate.declaration_line_bytes).hexdigest(),
        relation=relation,
        relation_component_sha256=preceding_declaration_relation_component_sha256_v1(
            relation
        ),
        sha256="",
    )
    return replace(
        partial,
        sha256=_digest(_OCCURRENCE_NAMESPACE, partial.record(include_sha256=False)),
    )


def _project_preceding_let_declaration_occurrences_v1(
    source: str,
    *,
    primary_byte_start: int,
    primary_byte_end: int,
    authenticate: bool,
) -> PrecedingLetDeclarationRosterV1:
    source_bytes = _ascii_source_bytes(source)
    if (
        type(primary_byte_start) is not int
        or type(primary_byte_end) is not int
        or not 0 <= primary_byte_start < primary_byte_end <= len(source_bytes)
    ):
        raise PrecedingLetDeclarationOccurrenceError(
            "preceding-declaration primary span is invalid"
        )

    try:
        frame = project_token_source_v1(source)
    except (TypeError, ValueError) as exc:
        raise PrecedingLetDeclarationOccurrenceError(
            "preceding-declaration token projection failed"
        ) from exc
    frame.authenticate()
    token_spans = _token_byte_spans(frame)
    primary_indexes = tuple(
        index
        for index, (kind, span) in enumerate(zip(frame.kinds, token_spans, strict=True))
        if kind == "IDENTIFIER" and span == (primary_byte_start, primary_byte_end)
    )
    if len(primary_indexes) != 1:
        raise PrecedingLetDeclarationOccurrenceError(
            "preceding-declaration primary span must be exactly one identifier token"
        )
    primary_span_bytes = source_bytes[primary_byte_start:primary_byte_end]
    primary_span_sha256 = sha256(primary_span_bytes).hexdigest()

    candidates = _preceding_simple_let_declarations(
        source=source_bytes,
        frame=frame,
        token_spans=token_spans,
        primary_byte_start=primary_byte_start,
        primary_identifier_bytes=primary_span_bytes,
    )
    if len(candidates) != len(PRECEDING_DECLARATION_RELATIONS):
        raise PrecedingLetDeclarationOccurrenceError(
            "preceding-declaration complete eligible roster must contain exactly two occurrences"
        )
    occurrences = tuple(
        _seal_occurrence(
            source=source_bytes,
            frame=frame,
            primary_byte_start=primary_byte_start,
            primary_span_sha256=primary_span_sha256,
            candidate=candidate,
            relation=relation,
        )
        for candidate, relation in zip(
            candidates,
            PRECEDING_DECLARATION_RELATIONS,
            strict=True,
        )
    )
    partial = PrecedingLetDeclarationRosterV1(
        source=source,
        source_sha256=sha256(source_bytes).hexdigest(),
        source_size_bytes=len(source_bytes),
        token_frame=frame,
        primary_byte_start=primary_byte_start,
        primary_byte_end=primary_byte_end,
        primary_span_bytes=primary_span_bytes,
        primary_span_sha256=primary_span_sha256,
        occurrences=occurrences,
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_ROSTER_NAMESPACE, partial.record(include_sha256=False)),
    )
    if authenticate:
        result.authenticate()
    return result


def project_preceding_let_declaration_occurrences_v1(
    source: str,
    *,
    primary_byte_start: int,
    primary_byte_end: int,
) -> PrecedingLetDeclarationRosterV1:
    """Project the complete exact-two preceding declaration occurrence roster."""

    return _project_preceding_let_declaration_occurrences_v1(
        source,
        primary_byte_start=primary_byte_start,
        primary_byte_end=primary_byte_end,
        authenticate=True,
    )
