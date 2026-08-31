"""Opaque exact and Python-lexical views over attested repository source bytes.

This module is an environment adapter, not a source-code reasoner.  The exact
view preserves the original repository-context projection byte for byte.  The
lexical view removes trivia and lexeme contents, projects bounded adjacent
token trigrams through the generic common-event cells, and retains only
authenticated roots, traces, counts, and physical work.
"""

from __future__ import annotations

import io
import json
import struct
import token
import tokenize
from hashlib import sha256
from typing import NamedTuple

from experiments.g0 import common_event_cells_v1 as common_events

SCHEMA = "COMMON_REPOSITORY_SOURCE_VIEWS_V1"

TRACE_LANES = 12
MAX_SOURCE_BYTES = 256
MAX_SEMANTIC_TOKENS = 64
MAX_SOURCE_FACTORS = 24
LEXICAL_FACTOR_VERSION = 1
LEXICAL_FACTOR_EVENT_BYTES = 8

_SOURCE_CONTEXT_NAMESPACE = b"common-repository-source-context-v1"
_LEXICAL_FACTOR_NAMESPACE = b"common-repository-lexical-factor-v1"
_LEXICAL_VIEW_NAMESPACE = b"common-repository-python-lexical-view-v1"
_SOURCE_VIEWS_NAMESPACE = b"common-repository-source-views-v1"

_SOURCE_CELLS = common_events.CommonEventCells()
_LEXICAL_CELLS = common_events.CommonEventCells()

_NAME_TOKEN_ID = 1
_NUMBER_TOKEN_ID = 2
_STRING_TOKEN_ID = 3
_OPERATOR_TOKEN_ID_BASE = 0x0100

# Frozen for schema v1.  Deriving the positions from the running interpreter
# would let a future Python release silently renumber existing observations.
_OPERATOR_LEXEMES = (
    "!=",
    "%",
    "%=",
    "&",
    "&=",
    "(",
    ")",
    "*",
    "**",
    "**=",
    "*=",
    "+",
    "+=",
    ",",
    "-",
    "-=",
    "->",
    ".",
    "...",
    "/",
    "//",
    "//=",
    "/=",
    ":",
    ":=",
    ";",
    "<",
    "<<",
    "<<=",
    "<=",
    "=",
    "==",
    ">",
    ">=",
    ">>",
    ">>=",
    "@",
    "@=",
    "[",
    "]",
    "^",
    "^=",
    "{",
    "|",
    "|=",
    "}",
    "~",
)
_OPERATOR_TOKEN_IDS = {
    lexeme: _OPERATOR_TOKEN_ID_BASE + index
    for index, lexeme in enumerate(_OPERATOR_LEXEMES)
}
_DROPPED_TOKEN_TYPES = frozenset(
    {
        tokenize.ENCODING,
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
    }
)


class RepositorySourceViewError(ValueError):
    """Base error for a source that cannot form a bounded v1 view."""


class RepositorySourceCapacityError(RepositorySourceViewError):
    """The complete source observation exceeds a frozen schema capacity."""


class RepositorySourceTokenError(RepositorySourceViewError):
    """The stdlib tokenizer could not produce one complete lexical stream."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def _valid_sha256(value: object) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


class RepositorySourceContextV1(NamedTuple):
    source_sha256: str
    frame_sha256: str
    trace_cells: tuple[int, ...]
    projection_cell_steps: int
    sha256: str

    def authenticate(self) -> None:
        if (
            not _valid_sha256(self.source_sha256)
            or not _valid_sha256(self.frame_sha256)
            or type(self.trace_cells) is not tuple
            or len(self.trace_cells) != TRACE_LANES
            or any(type(cell) is not int for cell in self.trace_cells)
            or type(self.projection_cell_steps) is not int
            or self.projection_cell_steps < 0
        ):
            raise ValueError("repository source context shape changed")
        expected = _digest(
            _SOURCE_CONTEXT_NAMESPACE,
            {
                "source": self.source_sha256,
                "frame": self.frame_sha256,
                "trace": self.trace_cells,
                "projection_cell_steps": self.projection_cell_steps,
            },
        )
        if self.sha256 != expected:
            raise ValueError("repository source context authentication failed")


class RepositoryLexicalFactorV1(NamedTuple):
    semantic_sha256: str
    frame_sha256: str
    trace_cells: tuple[int, ...]
    projection_cell_steps: int
    sha256: str

    def authenticate(self) -> None:
        payload = {
            "semantic": self.semantic_sha256,
            "frame": self.frame_sha256,
            "trace": self.trace_cells,
            "projection_cell_steps": self.projection_cell_steps,
        }
        if (
            not _valid_sha256(self.semantic_sha256)
            or not _valid_sha256(self.frame_sha256)
            or type(self.trace_cells) is not tuple
            or len(self.trace_cells) != TRACE_LANES
            or any(type(cell) is not int for cell in self.trace_cells)
            or type(self.projection_cell_steps) is not int
            or self.projection_cell_steps < 0
            or self.sha256 != _digest(_LEXICAL_FACTOR_NAMESPACE, payload)
        ):
            raise ValueError("repository lexical factor authentication failed")


class PythonRepositoryLexicalViewV1(NamedTuple):
    source_sha256: str
    token_count: int
    factor_count: int
    factors: tuple[RepositoryLexicalFactorV1, ...]
    projection_cell_steps: int
    sha256: str

    def authenticate(self) -> None:
        if type(self.factors) is not tuple:
            raise TypeError("repository lexical factors must be one exact tuple")
        for factor in self.factors:
            if type(factor) is not RepositoryLexicalFactorV1:
                raise TypeError("exact repository lexical factors required")
            factor.authenticate()
        factor_roots = tuple(factor.sha256 for factor in self.factors)
        payload = {
            "source": self.source_sha256,
            "token_count": self.token_count,
            "factor_count": self.factor_count,
            "factors": factor_roots,
            "projection_cell_steps": self.projection_cell_steps,
        }
        if (
            not _valid_sha256(self.source_sha256)
            or type(self.token_count) is not int
            or not 0 <= self.token_count <= MAX_SEMANTIC_TOKENS
            or type(self.factor_count) is not int
            or self.factor_count != len(self.factors)
            or self.factor_count > MAX_SOURCE_FACTORS
            or len(set(factor_roots)) != self.factor_count
            or len({factor.semantic_sha256 for factor in self.factors})
            != self.factor_count
            or type(self.projection_cell_steps) is not int
            or self.projection_cell_steps
            != sum(factor.projection_cell_steps for factor in self.factors)
            or self.sha256 != _digest(_LEXICAL_VIEW_NAMESPACE, payload)
        ):
            raise ValueError("repository lexical view authentication failed")


class RepositorySourceViewsV1(NamedTuple):
    exact_context: RepositorySourceContextV1
    lexical_view: PythonRepositoryLexicalViewV1
    sha256: str

    def authenticate(self) -> None:
        if type(self.exact_context) is not RepositorySourceContextV1:
            raise TypeError("exact repository source context required")
        if type(self.lexical_view) is not PythonRepositoryLexicalViewV1:
            raise TypeError("exact Python repository lexical view required")
        self.exact_context.authenticate()
        self.lexical_view.authenticate()
        payload = {
            "source": self.exact_context.source_sha256,
            "exact": self.exact_context.sha256,
            "lexical": self.lexical_view.sha256,
        }
        if (
            self.exact_context.source_sha256 != self.lexical_view.source_sha256
            or self.sha256 != _digest(_SOURCE_VIEWS_NAMESPACE, payload)
        ):
            raise ValueError("repository source views authentication failed")


def project_attested_repository_source_v1(source: bytes) -> RepositorySourceContextV1:
    """Project one already-attested visible source; retain no raw bytes."""

    if type(source) is not bytes:
        raise TypeError("repository source must be exact bytes")
    frame = _SOURCE_CELLS.project(source)
    source_root = _bytes_sha256(source)
    payload = {
        "source": source_root,
        "frame": frame.sha256,
        "trace": frame.trace_cells,
        "projection_cell_steps": frame.work.total,
    }
    result = RepositorySourceContextV1(
        source_sha256=source_root,
        frame_sha256=frame.sha256,
        trace_cells=frame.trace_cells,
        projection_cell_steps=frame.work.total,
        sha256=_digest(_SOURCE_CONTEXT_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def _normalized_python_token_ids(source: bytes) -> tuple[int, ...]:
    if len(source) > MAX_SOURCE_BYTES:
        raise RepositorySourceCapacityError(
            f"repository source exceeds {MAX_SOURCE_BYTES} bytes"
        )
    normalized: list[int] = []
    try:
        stream = tokenize.tokenize(io.BytesIO(source).readline)
        for item in stream:
            if item.type in _DROPPED_TOKEN_TYPES:
                continue
            if item.type == tokenize.ERRORTOKEN:
                raise RepositorySourceTokenError("repository source contains ERRORTOKEN")
            if item.type == token.NAME:
                normalized_id = _NAME_TOKEN_ID
            elif item.type == token.NUMBER:
                normalized_id = _NUMBER_TOKEN_ID
            elif item.type == token.STRING:
                normalized_id = _STRING_TOKEN_ID
            elif item.type == token.OP:
                normalized_id = _OPERATOR_TOKEN_IDS.get(item.string)
                if normalized_id is None:
                    raise RepositorySourceTokenError(
                        "repository source contains an operator outside schema v1"
                    )
            else:
                raise RepositorySourceTokenError(
                    "repository source contains an unsupported token kind"
                )
            normalized.append(normalized_id)
            if len(normalized) > MAX_SEMANTIC_TOKENS:
                raise RepositorySourceCapacityError(
                    f"repository source exceeds {MAX_SEMANTIC_TOKENS} semantic tokens"
                )
    except RepositorySourceViewError:
        raise
    except (IndentationError, SyntaxError, UnicodeDecodeError, tokenize.TokenError) as exc:
        raise RepositorySourceTokenError("repository source tokenization failed") from exc
    return tuple(normalized)


def _lexical_factor_events(token_ids: tuple[int, ...]) -> tuple[bytes, ...]:
    events = {
        struct.pack(
            "<HHHH",
            LEXICAL_FACTOR_VERSION,
            token_ids[index],
            token_ids[index + 1],
            token_ids[index + 2],
        )
        for index in range(max(0, len(token_ids) - 2))
    }
    ordered = tuple(sorted(events))
    if len(ordered) > MAX_SOURCE_FACTORS:
        raise RepositorySourceCapacityError(
            f"repository source exceeds {MAX_SOURCE_FACTORS} lexical factors"
        )
    if any(len(event) != LEXICAL_FACTOR_EVENT_BYTES for event in ordered):
        raise RuntimeError("repository lexical factor encoding changed")
    return ordered


def project_python_repository_lexical_view_v1(
    source: bytes,
) -> PythonRepositoryLexicalViewV1:
    """Project normalized adjacent Python-token trigrams; retain no lexemes."""

    if type(source) is not bytes:
        raise TypeError("repository source must be exact bytes")
    token_ids = _normalized_python_token_ids(source)
    factors: list[RepositoryLexicalFactorV1] = []
    for event in _lexical_factor_events(token_ids):
        frame = _LEXICAL_CELLS.project(event)
        semantic_root = sha256(b"repository-lexical-trigram-v1\0" + event).hexdigest()
        payload = {
            "semantic": semantic_root,
            "frame": frame.sha256,
            "trace": frame.trace_cells,
            "projection_cell_steps": frame.work.total,
        }
        factor = RepositoryLexicalFactorV1(
            semantic_sha256=semantic_root,
            frame_sha256=frame.sha256,
            trace_cells=frame.trace_cells,
            projection_cell_steps=frame.work.total,
            sha256=_digest(_LEXICAL_FACTOR_NAMESPACE, payload),
        )
        factor.authenticate()
        factors.append(factor)
    factor_tuple = tuple(factors)
    projection_steps = sum(factor.projection_cell_steps for factor in factor_tuple)
    payload = {
        "source": _bytes_sha256(source),
        "token_count": len(token_ids),
        "factor_count": len(factor_tuple),
        "factors": tuple(factor.sha256 for factor in factor_tuple),
        "projection_cell_steps": projection_steps,
    }
    result = PythonRepositoryLexicalViewV1(
        source_sha256=_bytes_sha256(source),
        token_count=len(token_ids),
        factor_count=len(factor_tuple),
        factors=factor_tuple,
        projection_cell_steps=projection_steps,
        sha256=_digest(_LEXICAL_VIEW_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def project_repository_source_views_v1(source: bytes) -> RepositorySourceViewsV1:
    """Build the exact and lexical views from one authenticated byte string."""

    if type(source) is not bytes:
        raise TypeError("repository source must be exact bytes")
    if len(source) > MAX_SOURCE_BYTES:
        raise RepositorySourceCapacityError(
            f"repository source exceeds {MAX_SOURCE_BYTES} bytes"
        )
    exact = project_attested_repository_source_v1(source)
    lexical = project_python_repository_lexical_view_v1(source)
    payload = {
        "source": exact.source_sha256,
        "exact": exact.sha256,
        "lexical": lexical.sha256,
    }
    result = RepositorySourceViewsV1(
        exact_context=exact,
        lexical_view=lexical,
        sha256=_digest(_SOURCE_VIEWS_NAMESPACE, payload),
    )
    result.authenticate()
    return result


__all__ = [
    "LEXICAL_FACTOR_EVENT_BYTES",
    "LEXICAL_FACTOR_VERSION",
    "MAX_SEMANTIC_TOKENS",
    "MAX_SOURCE_BYTES",
    "MAX_SOURCE_FACTORS",
    "SCHEMA",
    "TRACE_LANES",
    "PythonRepositoryLexicalViewV1",
    "RepositoryLexicalFactorV1",
    "RepositorySourceCapacityError",
    "RepositorySourceContextV1",
    "RepositorySourceTokenError",
    "RepositorySourceViewError",
    "RepositorySourceViewsV1",
    "project_attested_repository_source_v1",
    "project_python_repository_lexical_view_v1",
    "project_repository_source_views_v1",
]
