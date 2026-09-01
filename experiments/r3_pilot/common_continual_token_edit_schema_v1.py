from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256

import numpy as np
from experiments.g0 import common_event_cells_v1 as common_events

SCHEMA = "common-continual-token-edit-schema-v1"
CLAIM_BOUNDARY = (
    "bounded replay-free selection and consolidation of one relative single-token "
    "copy-replacement schema from a fixed lexical anchor/direction hypothesis bank and "
    "independent failed-to-successful public episodes; the Rust-like ASCII lexer and "
    "hypothesis ontology are supplied, not learned; not semantic diagnosis, general patch "
    "synthesis, transfer, SWE performance, or intelligence"
)

MAX_SOURCE_BYTES = 4_096
MAX_SOURCE_TOKENS = 256
MAX_TOKEN_BYTES = 128
MAX_OUTPUT_CASES = 64
MAX_OUTPUT_VALUES = 256
MAX_EPISODES = 32
MIN_INDEPENDENT_EPISODES = 2
TICK_CAP = 4
TICK_DTYPE = np.uint8

SCHEMA_CELL_OFFSET = 4_400_000
SCHEMA_TONIC_CELL_ID = SCHEMA_CELL_OFFSET + 255
SCHEMA_TONIC_SCORE = 2

READY = "READY"
CONTROL_TONIC_OFF_READY = "CONTROL_TONIC_OFF_READY"
STOP_CELL_OFF = "STOP_CELL_OFF"
STOP_SCHEMA_OFF = "STOP_SCHEMA_OFF"
STOP_NO_SCHEMA_MATCH = "STOP_NO_SCHEMA_MATCH"
STOP_INSUFFICIENT_INDEPENDENT_SOURCES = "STOP_INSUFFICIENT_INDEPENDENT_SOURCES"
STOP_PHYSICAL_RACE = "STOP_PHYSICAL_RACE"
STOP_SEEN_SOURCE = "STOP_SEEN_SOURCE"
STOP_ARTIFACT_REPLAY = "STOP_ARTIFACT_REPLAY"

_CONFIG_NAMESPACE = b"common-continual-token-edit-schema-v1-config"
_SOURCE_FRAME_NAMESPACE = b"common-token-source-frame-v1"
_OUTPUT_NAMESPACE = b"common-token-edit-output-v1"
_CONSEQUENCE_NAMESPACE = b"common-token-edit-consequence-v1"
_FEEDBACK_NAMESPACE = b"common-token-edit-feedback-v1"
_MATERIALIZATION_NAMESPACE = b"common-token-edit-materialization-v1"
_EPISODE_NAMESPACE = b"common-token-edit-episode-v1"
_SCHEMA_NAMESPACE = b"common-token-edit-schema-v1"
_OCCURRENCE_NAMESPACE = b"common-token-edit-occurrence-v1"
_PROVENANCE_NAMESPACE = b"common-token-edit-provenance-v1"
_CHECKPOINT_NAMESPACE = b"common-continual-token-edit-checkpoint-v1"
_PROPOSAL_NAMESPACE = b"common-token-edit-proposal-v1"

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")

_RUST_KEYWORDS = frozenset(
    {
        "as",
        "async",
        "await",
        "break",
        "const",
        "continue",
        "crate",
        "dyn",
        "else",
        "enum",
        "extern",
        "false",
        "fn",
        "for",
        "if",
        "impl",
        "in",
        "let",
        "loop",
        "match",
        "mod",
        "move",
        "mut",
        "pub",
        "ref",
        "return",
        "self",
        "Self",
        "static",
        "struct",
        "super",
        "trait",
        "true",
        "type",
        "unsafe",
        "use",
        "where",
        "while",
    }
)

_COMPARISON_TOKENS = frozenset(("<", ">", "<=", ">=", "==", "!="))
_ASSIGNMENT_TOKENS = frozenset(("=", "+=", "-=", "*=", "/=", "%="))
_ARITHMETIC_TOKENS = frozenset(("+", "-", "*", "/", "%"))
_OPEN_DELIMITERS = frozenset(("(", "[", "{"))
_CLOSE_DELIMITERS = frozenset((")", "]", "}"))
_FILLER_KINDS = frozenset(("IDENTIFIER", "NUMBER", "STRING", "CHAR"))

_TOKEN_PATTERN = re.compile(
    r"(?P<LINE_COMMENT>//[^\r\n]*)"
    r"|(?P<BLOCK_COMMENT>/\*.*?\*/)"
    r'|(?P<STRING>"(?:\\.|[^"\\])*")'
    r"|(?P<CHAR>'(?:\\.|[^'\\])*')"
    r"|(?P<IDENTIFIER>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<NUMBER>\d[A-Za-z0-9_]*(?:\.\d[A-Za-z0-9_]*)?)"
    r"|(?P<MULTI_OPERATOR><<=|>>=|::|->|=>|==|!=|<=|>=|&&|\|\||\+=|-=|\*=|/=|%=|\.\.=|\.\.)"
    r"|(?P<SINGLE>\S)",
    re.DOTALL,
)


class ContinualTokenEditSchemaError(ValueError):
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


def _valid_sha256(value: object) -> bool:
    return type(value) is str and _SHA256_HEX.fullmatch(value) is not None


def _validate_roots(*values: str) -> None:
    if any(not _valid_sha256(value) for value in values):
        raise ContinualTokenEditSchemaError("token-edit custody root is invalid")


def _array_sha256(value: np.ndarray) -> str:
    return _digest(
        b"common-token-edit-array-v1",
        {
            "dtype": value.dtype.str,
            "shape": value.shape,
            "bytes": sha256(value.tobytes(order="C")).hexdigest(),
        },
    )


def _freeze_array(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value).copy()
    result.setflags(write=False)
    return result


def _token_kind(group: str, text: str) -> str:
    if group == "IDENTIFIER":
        if text == "if":
            return "KEYWORD_IF"
        if text == "match":
            return "KEYWORD_MATCH"
        return "KEYWORD" if text in _RUST_KEYWORDS else "IDENTIFIER"
    if group in ("NUMBER", "STRING", "CHAR"):
        return group
    if text in _COMPARISON_TOKENS:
        return "COMPARISON"
    if text in _ASSIGNMENT_TOKENS:
        return "ASSIGNMENT"
    if text in _ARITHMETIC_TOKENS:
        return "ARITHMETIC"
    if text in _OPEN_DELIMITERS:
        return "OPEN_DELIMITER"
    if text in _CLOSE_DELIMITERS:
        return "CLOSE_DELIMITER"
    return "OPERATOR"


def _scan_source(source: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if type(source) is not str:
        raise TypeError("token source must be one exact string")
    try:
        encoded = source.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise ContinualTokenEditSchemaError("token source must be bounded ASCII") from exc
    if not encoded or len(encoded) > MAX_SOURCE_BYTES:
        raise ContinualTokenEditSchemaError("token source byte capacity failed")

    tokens: list[str] = []
    kinds: list[str] = []
    gaps: list[str] = []
    previous_token_end = 0
    for match in _TOKEN_PATTERN.finditer(source):
        group = match.lastgroup
        if group in ("LINE_COMMENT", "BLOCK_COMMENT"):
            continue
        if group is None:
            raise ContinualTokenEditSchemaError("token source scan lost its token kind")
        text = match.group(0)
        gaps.append(source[previous_token_end : match.start()])
        tokens.append(text)
        kinds.append(_token_kind(group, text))
        previous_token_end = match.end()
        if len(tokens) > MAX_SOURCE_TOKENS:
            raise ContinualTokenEditSchemaError("token source token capacity failed")
        if len(text.encode("ascii")) > MAX_TOKEN_BYTES:
            raise ContinualTokenEditSchemaError("token source contains an oversized token")
    gaps.append(source[previous_token_end:])
    if not tokens or len(gaps) != len(tokens) + 1:
        raise ContinualTokenEditSchemaError("token source produced no bounded tokens")
    reconstructed = "".join(
        (*(gaps[index] + token for index, token in enumerate(tokens)), gaps[-1])
    )
    if reconstructed != source:
        raise ContinualTokenEditSchemaError("token source projection is not lossless")
    return tuple(tokens), tuple(kinds), tuple(gaps)


def _token_event(kind: str, token: str) -> common_events.CommonEventFrame:
    return common_events.CommonEventCells().project(
        kind.encode("ascii") + b"\0" + token.encode("ascii")
    )


@dataclass(frozen=True, slots=True)
class TokenSourceFrameV1:
    source: str
    source_sha256: str
    tokens: tuple[str, ...]
    kinds: tuple[str, ...]
    gaps: tuple[str, ...]
    token_trace_cells: tuple[tuple[int, ...], ...]
    token_event_sha256s: tuple[str, ...]
    projection_cell_steps: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "source_sha256": self.source_sha256,
            "token_count": len(self.tokens),
            "token_sha256s": tuple(
                sha256(token.encode("ascii")).hexdigest() for token in self.tokens
            ),
            "kinds": self.kinds,
            "gap_sha256s": tuple(sha256(gap.encode("ascii")).hexdigest() for gap in self.gaps),
            "token_trace_cells": self.token_trace_cells,
            "token_event_sha256s": self.token_event_sha256s,
            "projection_cell_steps": self.projection_cell_steps,
            "raw_source_retained_by_checkpoint": False,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        projected = _project_token_source(self.source, authenticate=False)
        if (
            self.source_sha256 != projected.source_sha256
            or self.tokens != projected.tokens
            or self.kinds != projected.kinds
            or self.gaps != projected.gaps
            or self.token_trace_cells != projected.token_trace_cells
            or self.token_event_sha256s != projected.token_event_sha256s
            or self.projection_cell_steps != projected.projection_cell_steps
            or self.sha256 != projected.sha256
        ):
            raise ContinualTokenEditSchemaError("token source frame authentication failed")


def _project_token_source(source: str, *, authenticate: bool) -> TokenSourceFrameV1:
    tokens, kinds, gaps = _scan_source(source)
    events = tuple(_token_event(kind, token) for kind, token in zip(kinds, tokens, strict=True))
    record = {
        "source_sha256": sha256(source.encode("ascii")).hexdigest(),
        "token_count": len(tokens),
        "token_sha256s": tuple(sha256(token.encode("ascii")).hexdigest() for token in tokens),
        "kinds": kinds,
        "gap_sha256s": tuple(sha256(gap.encode("ascii")).hexdigest() for gap in gaps),
        "token_trace_cells": tuple(event.trace_cells for event in events),
        "token_event_sha256s": tuple(event.sha256 for event in events),
        "projection_cell_steps": sum(event.work.total for event in events),
        "raw_source_retained_by_checkpoint": False,
    }
    result = TokenSourceFrameV1(
        source=source,
        source_sha256=record["source_sha256"],
        tokens=tokens,
        kinds=kinds,
        gaps=gaps,
        token_trace_cells=record["token_trace_cells"],
        token_event_sha256s=record["token_event_sha256s"],
        projection_cell_steps=record["projection_cell_steps"],
        sha256=_digest(_SOURCE_FRAME_NAMESPACE, record),
    )
    if authenticate:
        result.authenticate()
    return result


def project_token_source_v1(source: str) -> TokenSourceFrameV1:
    return _project_token_source(source, authenticate=True)


def _validate_outputs(
    outputs: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    if (
        type(outputs) is not tuple
        or not outputs
        or len(outputs) > MAX_OUTPUT_CASES
        or any(
            type(row) is not tuple
            or len(row) > MAX_OUTPUT_VALUES
            or any(type(value) is not int for value in row)
            for row in outputs
        )
    ):
        raise ContinualTokenEditSchemaError("token-edit outputs are outside the bounded protocol")
    return outputs


def _output_sha256(outputs: tuple[tuple[int, ...], ...]) -> str:
    return _digest(_OUTPUT_NAMESPACE, _validate_outputs(outputs))


@dataclass(frozen=True, slots=True)
class TokenEditConsequenceV1:
    source_sha256: str
    batch_sha256: str
    test_manifest_sha256: str
    execution_sha256: str
    compile_cell: int
    run_cell: int
    observed_outputs: tuple[tuple[int, ...], ...]
    expected_outputs: tuple[tuple[int, ...], ...]
    public_test_cell: int
    feedback_sha256: str
    sha256: str

    @property
    def observed_output_sha256(self) -> str:
        return _output_sha256(self.observed_outputs)

    @property
    def expected_output_sha256(self) -> str:
        return _output_sha256(self.expected_outputs)

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "source_sha256": self.source_sha256,
            "batch_sha256": self.batch_sha256,
            "test_manifest_sha256": self.test_manifest_sha256,
            "execution_sha256": self.execution_sha256,
            "compile_cell": self.compile_cell,
            "run_cell": self.run_cell,
            "observed_outputs": self.observed_outputs,
            "expected_outputs": self.expected_outputs,
            "public_test_cell": self.public_test_cell,
            "feedback_sha256": self.feedback_sha256,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        observed = _validate_outputs(self.observed_outputs)
        expected = _validate_outputs(self.expected_outputs)
        _validate_roots(
            self.source_sha256,
            self.batch_sha256,
            self.test_manifest_sha256,
            self.execution_sha256,
            self.feedback_sha256,
        )
        feedback = _digest(
            _FEEDBACK_NAMESPACE,
            {
                "execution_sha256": self.execution_sha256,
                "source_sha256": self.source_sha256,
                "batch_sha256": self.batch_sha256,
                "test_manifest_sha256": self.test_manifest_sha256,
                "compile_cell": self.compile_cell,
                "run_cell": self.run_cell,
                "observed_output_sha256": _output_sha256(observed),
                "expected_output_sha256": _output_sha256(expected),
                "public_test_cell": self.public_test_cell,
            },
        )
        if (
            type(self.compile_cell) is not int
            or self.compile_cell not in (0, 1)
            or type(self.run_cell) is not int
            or self.run_cell not in (0, 1)
            or type(self.public_test_cell) is not int
            or self.public_test_cell not in (0, 1)
            or self.public_test_cell != int(observed == expected)
            or self.feedback_sha256 != feedback
            or self.sha256 != _digest(_CONSEQUENCE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualTokenEditSchemaError("token-edit consequence authentication failed")


def make_token_edit_consequence_v1(
    *,
    source_sha256: str,
    batch_sha256: str,
    test_manifest_sha256: str,
    execution_sha256: str,
    observed_outputs: tuple[tuple[int, ...], ...],
    expected_outputs: tuple[tuple[int, ...], ...],
    compile_cell: int = 1,
    run_cell: int = 1,
) -> TokenEditConsequenceV1:
    observed = _validate_outputs(observed_outputs)
    expected = _validate_outputs(expected_outputs)
    public_test_cell = int(observed == expected)
    feedback_sha256 = _digest(
        _FEEDBACK_NAMESPACE,
        {
            "execution_sha256": execution_sha256,
            "source_sha256": source_sha256,
            "batch_sha256": batch_sha256,
            "test_manifest_sha256": test_manifest_sha256,
            "compile_cell": compile_cell,
            "run_cell": run_cell,
            "observed_output_sha256": _output_sha256(observed),
            "expected_output_sha256": _output_sha256(expected),
            "public_test_cell": public_test_cell,
        },
    )
    record = {
        "source_sha256": source_sha256,
        "batch_sha256": batch_sha256,
        "test_manifest_sha256": test_manifest_sha256,
        "execution_sha256": execution_sha256,
        "compile_cell": compile_cell,
        "run_cell": run_cell,
        "observed_outputs": observed,
        "expected_outputs": expected,
        "public_test_cell": public_test_cell,
        "feedback_sha256": feedback_sha256,
    }
    result = TokenEditConsequenceV1(
        **record,
        sha256=_digest(_CONSEQUENCE_NAMESPACE, record),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class FailedSuccessfulTokenEditEpisodeV1:
    repository_sha256: str
    task_sha256: str
    task_instance_sha256: str
    task_source_sha256: str
    test_source_sha256: str
    failed_source: TokenSourceFrameV1
    failed_consequence: TokenEditConsequenceV1
    successful_source: TokenSourceFrameV1
    successful_consequence: TokenEditConsequenceV1
    materialization_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "repository_sha256": self.repository_sha256,
            "task_sha256": self.task_sha256,
            "task_instance_sha256": self.task_instance_sha256,
            "task_source_sha256": self.task_source_sha256,
            "test_source_sha256": self.test_source_sha256,
            "failed_source_sha256": self.failed_source.sha256,
            "failed_consequence_sha256": self.failed_consequence.sha256,
            "successful_source_sha256": self.successful_source.sha256,
            "successful_consequence_sha256": self.successful_consequence.sha256,
            "materialization_sha256": self.materialization_sha256,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if (
            type(self.failed_source) is not TokenSourceFrameV1
            or type(self.successful_source) is not TokenSourceFrameV1
        ):
            raise TypeError("token-edit episode sources must be exact physical frames")
        if (
            type(self.failed_consequence) is not TokenEditConsequenceV1
            or type(self.successful_consequence) is not TokenEditConsequenceV1
        ):
            raise TypeError("token-edit episode consequences must be exact receipts")
        self.failed_source.authenticate()
        self.successful_source.authenticate()
        self.failed_consequence.authenticate()
        self.successful_consequence.authenticate()
        _validate_roots(
            self.repository_sha256,
            self.task_sha256,
            self.task_instance_sha256,
            self.task_source_sha256,
            self.test_source_sha256,
            self.materialization_sha256,
        )
        mismatches = (
            tuple(
                index
                for index, (before, after) in enumerate(
                    zip(
                        self.failed_source.tokens,
                        self.successful_source.tokens,
                        strict=True,
                    )
                )
                if before != after
            )
            if len(self.failed_source.tokens) == len(self.successful_source.tokens)
            else ()
        )
        changed_index = mismatches[0] if len(mismatches) == 1 else None
        copy_bound = (
            changed_index is not None
            and self.failed_source.kinds[changed_index] == "IDENTIFIER"
            and self.successful_source.kinds[changed_index] == "IDENTIFIER"
            and self.successful_source.tokens[changed_index]
            in (
                *self.failed_source.tokens[:changed_index],
                *self.failed_source.tokens[changed_index + 1 :],
            )
        )
        materialization = _digest(
            _MATERIALIZATION_NAMESPACE,
            {
                "repository_sha256": self.repository_sha256,
                "failed_source_sha256": self.failed_source.sha256,
                "successful_source_sha256": self.successful_source.sha256,
                "changed_token_count": len(mismatches),
            },
        )
        distinct_custody_roots = (
            self.repository_sha256,
            self.task_sha256,
            self.task_instance_sha256,
            self.task_source_sha256,
            self.test_source_sha256,
            self.failed_source.source_sha256,
            self.successful_source.source_sha256,
            self.failed_consequence.execution_sha256,
            self.successful_consequence.execution_sha256,
            self.failed_consequence.feedback_sha256,
            self.successful_consequence.feedback_sha256,
            self.materialization_sha256,
        )
        if (
            self.failed_consequence.source_sha256 != self.failed_source.source_sha256
            or self.successful_consequence.source_sha256 != self.successful_source.source_sha256
            or self.failed_consequence.batch_sha256 != self.successful_consequence.batch_sha256
            or self.failed_consequence.test_manifest_sha256
            != self.successful_consequence.test_manifest_sha256
            or self.failed_consequence.expected_outputs
            != self.successful_consequence.expected_outputs
            or self.failed_consequence.compile_cell != 1
            or self.failed_consequence.run_cell != 1
            or self.failed_consequence.public_test_cell != 0
            or self.successful_consequence.compile_cell != 1
            or self.successful_consequence.run_cell != 1
            or self.successful_consequence.public_test_cell != 1
            or self.failed_consequence.execution_sha256
            == self.successful_consequence.execution_sha256
            or self.failed_consequence.feedback_sha256
            == self.successful_consequence.feedback_sha256
            or len(set(distinct_custody_roots)) != len(distinct_custody_roots)
            or self.failed_source.gaps != self.successful_source.gaps
            or self.failed_source.kinds != self.successful_source.kinds
            or not copy_bound
            or self.materialization_sha256 != materialization
            or self.sha256 != _digest(_EPISODE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualTokenEditSchemaError(
                "failed-to-successful token-edit episode authentication failed"
            )


def seal_failed_successful_token_edit_episode_v1(
    *,
    repository_sha256: str,
    task_sha256: str,
    task_instance_sha256: str,
    task_source_sha256: str,
    test_source_sha256: str,
    failed_source: TokenSourceFrameV1,
    failed_consequence: TokenEditConsequenceV1,
    successful_source: TokenSourceFrameV1,
    successful_consequence: TokenEditConsequenceV1,
) -> FailedSuccessfulTokenEditEpisodeV1:
    materialization_sha256 = _digest(
        _MATERIALIZATION_NAMESPACE,
        {
            "repository_sha256": repository_sha256,
            "failed_source_sha256": failed_source.sha256,
            "successful_source_sha256": successful_source.sha256,
            "changed_token_count": sum(
                before != after
                for before, after in zip(
                    failed_source.tokens,
                    successful_source.tokens,
                    strict=True,
                )
            )
            if len(failed_source.tokens) == len(successful_source.tokens)
            else -1,
        },
    )
    record = {
        "repository_sha256": repository_sha256,
        "task_sha256": task_sha256,
        "task_instance_sha256": task_instance_sha256,
        "task_source_sha256": task_source_sha256,
        "test_source_sha256": test_source_sha256,
        "failed_source_sha256": failed_source.sha256,
        "failed_consequence_sha256": failed_consequence.sha256,
        "successful_source_sha256": successful_source.sha256,
        "successful_consequence_sha256": successful_consequence.sha256,
        "materialization_sha256": materialization_sha256,
    }
    result = FailedSuccessfulTokenEditEpisodeV1(
        repository_sha256=repository_sha256,
        task_sha256=task_sha256,
        task_instance_sha256=task_instance_sha256,
        task_source_sha256=task_source_sha256,
        test_source_sha256=test_source_sha256,
        failed_source=failed_source,
        failed_consequence=failed_consequence,
        successful_source=successful_source,
        successful_consequence=successful_consequence,
        materialization_sha256=materialization_sha256,
        sha256=_digest(_EPISODE_NAMESPACE, record),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class ParameterizedTokenEditSchemaV1:
    schema_index: int
    anchor_kind: str
    target_direction: int
    donor_direction: int
    schema_cell_id: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_index": self.schema_index,
            "operation": "COPY_REPLACE_ONE_TOKEN",
            "anchor_kind": self.anchor_kind,
            "target": "NEAREST_FILLER",
            "target_direction": self.target_direction,
            "guard": "MIRROR_FILLER_EQUALITY",
            "donor": "NEAREST_DISTINCT_SAME_KIND_FILLER",
            "donor_direction": self.donor_direction,
            "schema_cell_id": self.schema_cell_id,
            "concrete_identifier_or_literal_retained": False,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if (
            type(self.schema_index) is not int
            or self.schema_index < 0
            or self.anchor_kind not in _ANCHOR_KINDS
            or self.target_direction not in (-1, 1)
            or self.donor_direction not in (-1, 1)
            or self.schema_cell_id != SCHEMA_CELL_OFFSET + self.schema_index
            or self.sha256 != _digest(_SCHEMA_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualTokenEditSchemaError("token-edit schema authentication failed")


_ANCHOR_KINDS = (
    "COMPARISON",
    "ASSIGNMENT",
    "ARITHMETIC",
    "OPEN_DELIMITER",
    "KEYWORD_IF",
    "KEYWORD_MATCH",
)


def _build_schemas() -> tuple[ParameterizedTokenEditSchemaV1, ...]:
    rows: list[ParameterizedTokenEditSchemaV1] = []
    for anchor_kind in _ANCHOR_KINDS:
        for target_direction in (-1, 1):
            for donor_direction in (-1, 1):
                index = len(rows)
                partial = ParameterizedTokenEditSchemaV1(
                    schema_index=index,
                    anchor_kind=anchor_kind,
                    target_direction=target_direction,
                    donor_direction=donor_direction,
                    schema_cell_id=SCHEMA_CELL_OFFSET + index,
                    sha256="",
                )
                rows.append(
                    replace(
                        partial,
                        sha256=_digest(
                            _SCHEMA_NAMESPACE,
                            partial.record(include_sha256=False),
                        ),
                    )
                )
    return tuple(rows)


PARAMETERIZED_SCHEMAS = _build_schemas()
for _schema in PARAMETERIZED_SCHEMAS:
    _schema.authenticate()
if len(PARAMETERIZED_SCHEMAS) >= 255:
    raise RuntimeError("token-edit schema bank exceeds physical output capacity")

CONFIG_SHA256 = _digest(
    _CONFIG_NAMESPACE,
    {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "capacities": {
            "max_source_bytes": MAX_SOURCE_BYTES,
            "max_source_tokens": MAX_SOURCE_TOKENS,
            "max_token_bytes": MAX_TOKEN_BYTES,
            "max_output_cases": MAX_OUTPUT_CASES,
            "max_output_values": MAX_OUTPUT_VALUES,
            "max_episodes": MAX_EPISODES,
            "tick_cap": TICK_CAP,
            "tick_dtype": np.dtype(TICK_DTYPE).str,
        },
        "lexical_projection": {
            "token_pattern": _TOKEN_PATTERN.pattern,
            "rust_keywords": tuple(sorted(_RUST_KEYWORDS)),
            "comparison_tokens": tuple(sorted(_COMPARISON_TOKENS)),
            "assignment_tokens": tuple(sorted(_ASSIGNMENT_TOKENS)),
            "arithmetic_tokens": tuple(sorted(_ARITHMETIC_TOKENS)),
            "open_delimiters": tuple(sorted(_OPEN_DELIMITERS)),
            "close_delimiters": tuple(sorted(_CLOSE_DELIMITERS)),
            "filler_kinds": tuple(sorted(_FILLER_KINDS)),
        },
        "minimum_independent_episodes": MIN_INDEPENDENT_EPISODES,
        "schemas": tuple(row.record() for row in PARAMETERIZED_SCHEMAS),
        "tonic_cell_id": SCHEMA_TONIC_CELL_ID,
        "tonic_score": SCHEMA_TONIC_SCORE,
    },
)


@dataclass(frozen=True, slots=True)
class _AppliedSchemaV1:
    schema_index: int
    anchor_token_index: int
    target_token_index: int
    donor_token_index: int
    edited_source: TokenSourceFrameV1


def _nearest_filler(
    frame: TokenSourceFrameV1,
    origin: int,
    direction: int,
    *,
    required_kind: str | None = None,
    distinct_from: str | None = None,
) -> int | None:
    index = origin + direction
    while 0 <= index < len(frame.tokens):
        kind = frame.kinds[index]
        token = frame.tokens[index]
        if (
            kind in _FILLER_KINDS
            and (required_kind is None or kind == required_kind)
            and (distinct_from is None or token != distinct_from)
        ):
            return index
        index += direction
    return None


def _replace_token(frame: TokenSourceFrameV1, index: int, token: str) -> TokenSourceFrameV1:
    tokens = list(frame.tokens)
    tokens[index] = token
    source = "".join(
        (*(frame.gaps[position] + value for position, value in enumerate(tokens)), frame.gaps[-1])
    )
    return project_token_source_v1(source)


def _apply_schema(
    schema: ParameterizedTokenEditSchemaV1,
    frame: TokenSourceFrameV1,
) -> _AppliedSchemaV1 | None:
    bindings: set[tuple[int, int, int]] = set()
    for anchor, kind in enumerate(frame.kinds):
        if kind != schema.anchor_kind:
            continue
        target = _nearest_filler(frame, anchor, schema.target_direction)
        if target is None:
            continue
        target_kind = frame.kinds[target]
        target_token = frame.tokens[target]
        mirror = _nearest_filler(
            frame,
            anchor,
            -schema.target_direction,
            required_kind=target_kind,
        )
        if mirror is None or frame.tokens[mirror] != target_token:
            continue
        donor = _nearest_filler(
            frame,
            target,
            schema.donor_direction,
            required_kind=target_kind,
            distinct_from=target_token,
        )
        if donor is not None:
            bindings.add((anchor, target, donor))
    if len(bindings) != 1:
        return None
    anchor, target, donor = next(iter(bindings))
    edited = _replace_token(frame, target, frame.tokens[donor])
    return _AppliedSchemaV1(
        schema_index=schema.schema_index,
        anchor_token_index=anchor,
        target_token_index=target,
        donor_token_index=donor,
        edited_source=edited,
    )


def _occurrence_sha256(source_sha256: str, token_index: int) -> str:
    return _digest(
        _OCCURRENCE_NAMESPACE,
        {
            "source_sha256": source_sha256,
            "token_index": token_index,
        },
    )


def _projected_source_equal(
    left: TokenSourceFrameV1,
    right: TokenSourceFrameV1,
) -> tuple[bool, int]:
    if len(left.token_trace_cells) != len(right.token_trace_cells):
        return False, 0
    work = len(left.token_trace_cells) * common_events.TRACE_LANES
    equal = all(
        left_cells == right_cells
        for left_cells, right_cells in zip(
            left.token_trace_cells,
            right.token_trace_cells,
            strict=True,
        )
    )
    if equal and left.source != right.source:
        raise ContinualTokenEditSchemaError(
            "projected source traces collided during token-edit learning"
        )
    return equal, work


def _run_schema_race(
    raw_scores: tuple[int, ...],
    *,
    tonic_gate: bool,
) -> tuple[object, int]:
    currents = np.asarray(
        [
            [
                *(common_events._CURRENT_TABLE[score] for score in raw_scores),
                common_events._CURRENT_TABLE[SCHEMA_TONIC_SCORE if tonic_gate else 0],
            ]
        ],
        dtype=np.float64,
    )
    return (
        common_events._OUTPUT_RACE.race_many(currents, require_unique=False)[0],
        currents.shape[1] * common_events.RACE_STEPS,
    )


def _matched_schemas(
    episode: FailedSuccessfulTokenEditEpisodeV1,
) -> tuple[tuple[int, ...], int, int, int]:
    matches: list[int] = []
    actuations = 0
    trace_comparisons = 0
    projection_steps = 0
    for schema in PARAMETERIZED_SCHEMAS:
        applied = _apply_schema(schema, episode.failed_source)
        if applied is None:
            continue
        actuations += 1
        projection_steps += applied.edited_source.projection_cell_steps
        equal, work = _projected_source_equal(
            applied.edited_source,
            episode.successful_source,
        )
        trace_comparisons += work
        if equal:
            matches.append(schema.schema_index)
    return tuple(matches), actuations, trace_comparisons, projection_steps


@dataclass(frozen=True, slots=True)
class TokenEditSchemaProvenanceV1:
    episode_sha256: str
    parent_checkpoint_sha256: str
    repository_sha256: str
    task_sha256: str
    task_instance_sha256: str
    task_source_sha256: str
    test_source_sha256: str
    failed_source_sha256: str
    successful_source_sha256: str
    failed_execution_sha256: str
    successful_execution_sha256: str
    failed_feedback_sha256: str
    successful_feedback_sha256: str
    materialization_sha256: str
    matched_schema_indices: tuple[int, ...]
    independent_cell: int
    candidate_actuation_count: int
    token_trace_cell_comparison_count: int
    candidate_projection_cell_steps: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        _validate_roots(
            self.episode_sha256,
            self.parent_checkpoint_sha256,
            self.repository_sha256,
            self.task_sha256,
            self.task_instance_sha256,
            self.task_source_sha256,
            self.test_source_sha256,
            self.failed_source_sha256,
            self.successful_source_sha256,
            self.failed_execution_sha256,
            self.successful_execution_sha256,
            self.failed_feedback_sha256,
            self.successful_feedback_sha256,
            self.materialization_sha256,
        )
        if (
            self.matched_schema_indices != tuple(sorted(set(self.matched_schema_indices)))
            or any(
                type(index) is not int or not 0 <= index < len(PARAMETERIZED_SCHEMAS)
                for index in self.matched_schema_indices
            )
            or type(self.independent_cell) is not int
            or self.independent_cell not in (0, 1)
            or any(
                type(value) is not int or value < 0
                for value in (
                    self.candidate_actuation_count,
                    self.token_trace_cell_comparison_count,
                    self.candidate_projection_cell_steps,
                )
            )
            or self.sha256 != _digest(_PROVENANCE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualTokenEditSchemaError("token-edit provenance authentication failed")


@dataclass(frozen=True, slots=True)
class TokenEditSchemaTrainingSummaryV1:
    mode: str
    parent_checkpoint_sha256: str | None
    episode_count: int
    independent_episode_count: int
    dependent_episode_count: int
    schema_support_update_count: int
    candidate_actuation_count: int
    token_trace_cell_comparison_count: int
    candidate_projection_cell_steps: int
    support_counts: tuple[int, ...]
    promoted_schema_indices: tuple[int, ...]
    checkpoint_array_bytes: int
    checkpoint_sha256: str


def _training_summary_record(
    *,
    support: np.ndarray,
    provenance: tuple[TokenEditSchemaProvenanceV1, ...],
    parent_checkpoint_sha256: str | None,
    mode: str,
) -> dict[str, object]:
    independent = sum(row.independent_cell for row in provenance)
    support_updates = sum(
        len(row.matched_schema_indices) * row.independent_cell for row in provenance
    )
    return {
        "mode": mode,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "episode_count": len(provenance),
        "independent_episode_count": independent,
        "dependent_episode_count": len(provenance) - independent,
        "schema_support_update_count": support_updates,
        "candidate_actuation_count": sum(row.candidate_actuation_count for row in provenance),
        "token_trace_cell_comparison_count": sum(
            row.token_trace_cell_comparison_count for row in provenance
        ),
        "candidate_projection_cell_steps": sum(
            row.candidate_projection_cell_steps for row in provenance
        ),
        "support_counts": tuple(int(value) for value in support),
        "promoted_schema_indices": tuple(
            index for index, value in enumerate(support) if int(value) >= MIN_INDEPENDENT_EPISODES
        ),
        "checkpoint_array_bytes": support.nbytes,
    }


def _checkpoint_sha256(
    support: np.ndarray,
    provenance: tuple[TokenEditSchemaProvenanceV1, ...],
    summary_record: dict[str, object],
) -> str:
    return _digest(
        _CHECKPOINT_NAMESPACE,
        {
            "config_sha256": CONFIG_SHA256,
            "support_array_sha256": _array_sha256(support),
            "provenance": tuple(row.record() for row in provenance),
            "summary": summary_record,
        },
    )


class FrozenContinualTokenEditSchemaV1:
    __slots__ = ("_provenance", "_summary", "_support")

    def __init__(
        self,
        *,
        support: np.ndarray,
        provenance: tuple[TokenEditSchemaProvenanceV1, ...],
        summary: TokenEditSchemaTrainingSummaryV1,
    ) -> None:
        self._support = _freeze_array(support)
        self._provenance = provenance
        self._summary = summary

    @property
    def checkpoint_sha256(self) -> str:
        return self._summary.checkpoint_sha256

    @property
    def checkpoint_array_sha256(self) -> str:
        return _array_sha256(self._support)

    @property
    def training_summary(self) -> TokenEditSchemaTrainingSummaryV1:
        return self._summary

    @property
    def provenance(self) -> tuple[TokenEditSchemaProvenanceV1, ...]:
        return self._provenance

    @property
    def support_counts(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._support)

    def authenticate(self, *, expected_checkpoint_sha256: str | None = None) -> None:
        if expected_checkpoint_sha256 is not None and not _valid_sha256(expected_checkpoint_sha256):
            raise TypeError("expected token-edit checkpoint root must be exact SHA-256")
        if (
            type(self._support) is not np.ndarray
            or self._support.dtype != TICK_DTYPE
            or self._support.shape != (len(PARAMETERIZED_SCHEMAS),)
            or self._support.flags.writeable
            or type(self._provenance) is not tuple
            or any(type(row) is not TokenEditSchemaProvenanceV1 for row in self._provenance)
            or len(self._provenance) > MAX_EPISODES
            or type(self._summary) is not TokenEditSchemaTrainingSummaryV1
        ):
            raise ContinualTokenEditSchemaError("token-edit checkpoint structure failed")
        seen_context_roots: set[str] = set()
        seen_replay_roots: set[str] = set()
        seen_custody_roots: set[str] = set()
        expected = np.zeros((len(PARAMETERIZED_SCHEMAS),), dtype=TICK_DTYPE)
        prefix_provenance: tuple[TokenEditSchemaProvenanceV1, ...] = ()
        expected_summary_record = _training_summary_record(
            support=expected,
            provenance=prefix_provenance,
            parent_checkpoint_sha256=None,
            mode="EMPTY_CONTINUAL_TOKEN_EDIT_SCHEMA_V1",
        )
        prefix_root = _checkpoint_sha256(
            expected,
            prefix_provenance,
            expected_summary_record,
        )
        for row in self._provenance:
            row.authenticate()
            context_roots = {
                row.repository_sha256,
                row.task_sha256,
                row.task_instance_sha256,
                row.task_source_sha256,
                row.test_source_sha256,
                row.failed_source_sha256,
                row.successful_source_sha256,
            }
            replay_roots = {
                row.episode_sha256,
                row.failed_execution_sha256,
                row.successful_execution_sha256,
                row.failed_feedback_sha256,
                row.successful_feedback_sha256,
                row.materialization_sha256,
            }
            if (
                len(context_roots) != 7
                or len(replay_roots) != 6
                or not context_roots.isdisjoint(replay_roots)
                or not replay_roots.isdisjoint(seen_custody_roots)
                or not context_roots.isdisjoint(seen_replay_roots)
                or row.independent_cell != int(context_roots.isdisjoint(seen_context_roots))
                or row.parent_checkpoint_sha256 != prefix_root
            ):
                raise ContinualTokenEditSchemaError(
                    "token-edit checkpoint historical custody replay failed"
                )
            seen_context_roots.update(context_roots)
            seen_replay_roots.update(replay_roots)
            seen_custody_roots.update(context_roots)
            seen_custody_roots.update(replay_roots)
            if row.independent_cell == 1:
                for index in row.matched_schema_indices:
                    expected[index] = np.uint8(min(TICK_CAP, int(expected[index]) + 1))
            parent_root = prefix_root
            prefix_provenance = (*prefix_provenance, row)
            expected_summary_record = _training_summary_record(
                support=expected,
                provenance=prefix_provenance,
                parent_checkpoint_sha256=parent_root,
                mode="REPLAY_FREE_CONTINUAL_TOKEN_EDIT_ADVANCE_V1",
            )
            prefix_root = _checkpoint_sha256(
                expected,
                prefix_provenance,
                expected_summary_record,
            )
        summary_record = {
            field.name: getattr(self._summary, field.name)
            for field in dataclasses.fields(self._summary)
            if field.name != "checkpoint_sha256"
        }
        if (
            not np.array_equal(self._support, expected)
            or summary_record != expected_summary_record
            or self._summary.checkpoint_sha256 != prefix_root
            or (
                expected_checkpoint_sha256 is not None and prefix_root != expected_checkpoint_sha256
            )
        ):
            raise ContinualTokenEditSchemaError("token-edit checkpoint authentication failed")

    def propose(
        self,
        source: TokenSourceFrameV1,
        *,
        cell_gate: bool = True,
        schema_gate: bool = True,
        tonic_gate: bool = True,
    ) -> TokenEditProposalV1:
        self.authenticate()
        if type(source) is not TokenSourceFrameV1:
            raise TypeError("token-edit proposal source must be one exact frame")
        if (
            type(cell_gate) is not bool
            or type(schema_gate) is not bool
            or type(tonic_gate) is not bool
        ):
            raise TypeError("token-edit proposal gates must be exact booleans")
        source.authenticate()
        result = _derive_token_edit_proposal(
            self,
            source,
            cell_gate=cell_gate,
            schema_gate=schema_gate,
            tonic_gate=tonic_gate,
        )
        result.authenticate()
        return result


@dataclass(frozen=True, slots=True)
class TokenEditProposalV1:
    checkpoint: FrozenContinualTokenEditSchemaV1
    input_source: TokenSourceFrameV1
    status: str
    selected_schema_index: int | None
    selected_schema_cell_id: int | None
    selected_schema_sha256: str | None
    anchor_token_index: int | None
    target_token_index: int | None
    donor_token_index: int | None
    anchor_occurrence_sha256: str | None
    target_occurrence_sha256: str | None
    donor_occurrence_sha256: str | None
    input_source_sha256: str
    edited_source_sha256: str | None
    edited_source: str | None
    applicable_schema_indices: tuple[int, ...]
    schema_support_counts: tuple[int, ...]
    schema_raw_scores: tuple[int, ...]
    race_winner_cell: int | None
    race_spike_step: int | None
    race_simultaneous_spike_candidates: int
    race_physical_cell_steps: int
    cell_gate: bool
    schema_gate: bool
    tonic_gate: bool
    checkpoint_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name
            not in (
                "checkpoint",
                "edited_source",
                "input_source",
                "sha256",
            )
        }
        result["edited_source_sha256"] = self.edited_source_sha256
        result["raw_edited_source_retained_by_checkpoint"] = False
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.checkpoint) is not FrozenContinualTokenEditSchemaV1:
            raise TypeError("token-edit proposal must retain its exact frozen checkpoint")
        if type(self.input_source) is not TokenSourceFrameV1:
            raise TypeError("token-edit proposal must retain its exact source frame")
        self.checkpoint.authenticate()
        self.input_source.authenticate()
        early_stop = self.status in {STOP_CELL_OFF, STOP_SCHEMA_OFF, STOP_SEEN_SOURCE}
        if (
            self.status
            not in {
                CONTROL_TONIC_OFF_READY,
                READY,
                STOP_CELL_OFF,
                STOP_SCHEMA_OFF,
                STOP_NO_SCHEMA_MATCH,
                STOP_INSUFFICIENT_INDEPENDENT_SOURCES,
                STOP_PHYSICAL_RACE,
                STOP_SEEN_SOURCE,
                STOP_ARTIFACT_REPLAY,
            }
            or self.input_source_sha256 != self.input_source.source_sha256
            or self.checkpoint_sha256 != self.checkpoint.checkpoint_sha256
            or self.schema_support_counts != self.checkpoint.support_counts
            or not _valid_sha256(self.input_source_sha256)
            or not _valid_sha256(self.checkpoint_sha256)
            or type(self.cell_gate) is not bool
            or type(self.schema_gate) is not bool
            or type(self.tonic_gate) is not bool
            or type(self.applicable_schema_indices) is not tuple
            or self.applicable_schema_indices != tuple(sorted(set(self.applicable_schema_indices)))
            or any(
                type(index) is not int or not 0 <= index < len(PARAMETERIZED_SCHEMAS)
                for index in self.applicable_schema_indices
            )
            or type(self.schema_support_counts) is not tuple
            or len(self.schema_support_counts) != len(PARAMETERIZED_SCHEMAS)
            or any(
                type(value) is not int or not 0 <= value <= TICK_CAP
                for value in self.schema_support_counts
            )
            or type(self.schema_raw_scores) is not tuple
            or len(self.schema_raw_scores) != len(PARAMETERIZED_SCHEMAS)
            or self.schema_raw_scores
            != tuple(
                min(common_events.MAX_OUTPUT_SCORE, support * 2)
                if index in self.applicable_schema_indices
                else 0
                for index, support in enumerate(self.schema_support_counts)
            )
            or (
                self.race_winner_cell is not None
                and (
                    type(self.race_winner_cell) is not int
                    or not 0 <= self.race_winner_cell <= len(PARAMETERIZED_SCHEMAS)
                )
            )
            or (
                self.race_spike_step is not None
                and (type(self.race_spike_step) is not int or self.race_spike_step < 0)
            )
            or type(self.race_simultaneous_spike_candidates) is not int
            or self.race_simultaneous_spike_candidates < 0
            or type(self.race_physical_cell_steps) is not int
            or self.race_physical_cell_steps < 0
            or (
                self.status == STOP_CELL_OFF
                and (self.cell_gate or self.race_physical_cell_steps != 0)
            )
            or (
                self.status == STOP_SCHEMA_OFF
                and (not self.cell_gate or self.schema_gate or self.race_physical_cell_steps != 0)
            )
            or (
                self.status == STOP_SEEN_SOURCE
                and (
                    not self.cell_gate or not self.schema_gate or self.race_physical_cell_steps != 0
                )
            )
            or (
                early_stop
                and (
                    self.applicable_schema_indices
                    or any(self.schema_raw_scores)
                    or self.race_winner_cell is not None
                    or self.race_spike_step is not None
                    or self.race_simultaneous_spike_candidates != 0
                )
            )
            or self.sha256 != _digest(_PROPOSAL_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualTokenEditSchemaError("token-edit proposal authentication failed")
        if not early_stop:
            race, race_steps = _run_schema_race(
                self.schema_raw_scores,
                tonic_gate=self.tonic_gate,
            )
            if (
                self.race_winner_cell != getattr(race, "winner_cell", None)
                or self.race_spike_step != getattr(race, "spike_step", None)
                or self.race_simultaneous_spike_candidates
                != getattr(race, "simultaneous_spike_candidates", 0)
                or self.race_physical_cell_steps != race_steps
            ):
                raise ContinualTokenEditSchemaError(
                    "token-edit proposal physical race authentication failed"
                )
            maximum_support = max(
                (self.schema_support_counts[index] for index in self.applicable_schema_indices),
                default=0,
            )
            if (
                (self.status == STOP_NO_SCHEMA_MATCH and self.applicable_schema_indices)
                or (
                    self.status == STOP_INSUFFICIENT_INDEPENDENT_SOURCES
                    and (
                        not self.applicable_schema_indices
                        or not self.tonic_gate
                        or maximum_support >= MIN_INDEPENDENT_EPISODES
                    )
                )
                or (
                    self.status == STOP_PHYSICAL_RACE
                    and (
                        not self.applicable_schema_indices
                        or self.race_simultaneous_spike_candidates == 1
                        or (self.tonic_gate and maximum_support < MIN_INDEPENDENT_EPISODES)
                    )
                )
                or (
                    self.status in {CONTROL_TONIC_OFF_READY, READY, STOP_ARTIFACT_REPLAY}
                    and (
                        not self.applicable_schema_indices
                        or self.race_winner_cell == len(PARAMETERIZED_SCHEMAS)
                        or self.race_simultaneous_spike_candidates != 1
                    )
                )
            ):
                raise ContinualTokenEditSchemaError(
                    "token-edit proposal status is inconsistent with its physical race"
                )
        ready_status = self.status in {CONTROL_TONIC_OFF_READY, READY}
        if ready_status:
            if (
                type(self.selected_schema_index) is not int
                or not 0 <= self.selected_schema_index < len(PARAMETERIZED_SCHEMAS)
                or self.selected_schema_cell_id != SCHEMA_CELL_OFFSET + self.selected_schema_index
                or self.selected_schema_sha256
                != PARAMETERIZED_SCHEMAS[self.selected_schema_index].sha256
                or type(self.anchor_token_index) is not int
                or self.anchor_token_index < 0
                or type(self.target_token_index) is not int
                or self.target_token_index < 0
                or type(self.donor_token_index) is not int
                or self.donor_token_index < 0
                or len(
                    {
                        self.anchor_token_index,
                        self.target_token_index,
                        self.donor_token_index,
                    }
                )
                != 3
                or self.selected_schema_index not in self.applicable_schema_indices
                or self.race_winner_cell != self.selected_schema_index
                or self.race_simultaneous_spike_candidates != 1
                or not self.cell_gate
                or not self.schema_gate
                or (self.status == READY and not self.tonic_gate)
                or (self.status == CONTROL_TONIC_OFF_READY and self.tonic_gate)
                or self.anchor_occurrence_sha256
                != _occurrence_sha256(
                    self.input_source_sha256,
                    self.anchor_token_index,
                )
                or self.target_occurrence_sha256
                != _occurrence_sha256(
                    self.input_source_sha256,
                    self.target_token_index,
                )
                or self.donor_occurrence_sha256
                != _occurrence_sha256(
                    self.input_source_sha256,
                    self.donor_token_index,
                )
                or self.edited_source is None
                or self.edited_source_sha256
                != sha256(self.edited_source.encode("ascii")).hexdigest()
            ):
                raise ContinualTokenEditSchemaError("READY token-edit proposal is incomplete")
        elif any(
            value is not None
            for value in (
                self.selected_schema_index,
                self.selected_schema_cell_id,
                self.selected_schema_sha256,
                self.anchor_token_index,
                self.target_token_index,
                self.donor_token_index,
                self.anchor_occurrence_sha256,
                self.target_occurrence_sha256,
                self.donor_occurrence_sha256,
                self.edited_source_sha256,
                self.edited_source,
            )
        ):
            raise ContinualTokenEditSchemaError("STOP token-edit proposal exposed an edit")
        canonical = _derive_token_edit_proposal(
            self.checkpoint,
            self.input_source,
            cell_gate=self.cell_gate,
            schema_gate=self.schema_gate,
            tonic_gate=self.tonic_gate,
        )
        if self.record() != canonical.record() or self.edited_source != canonical.edited_source:
            raise ContinualTokenEditSchemaError(
                "token-edit proposal does not rederive from its exact context"
            )


def _proposal(
    *,
    checkpoint: FrozenContinualTokenEditSchemaV1,
    source: TokenSourceFrameV1,
    status: str,
    winner: int | None,
    application: _AppliedSchemaV1 | None,
    applicable: tuple[bool, ...],
    raw_scores: tuple[int, ...],
    race: object,
    race_steps: int,
    cell_gate: bool,
    schema_gate: bool,
    tonic_gate: bool,
) -> TokenEditProposalV1:
    schema = None if winner is None else PARAMETERIZED_SCHEMAS[winner]
    edited = None if application is None else application.edited_source
    partial = TokenEditProposalV1(
        checkpoint=checkpoint,
        input_source=source,
        status=status,
        selected_schema_index=winner,
        selected_schema_cell_id=None if schema is None else schema.schema_cell_id,
        selected_schema_sha256=None if schema is None else schema.sha256,
        anchor_token_index=None if application is None else application.anchor_token_index,
        target_token_index=None if application is None else application.target_token_index,
        donor_token_index=None if application is None else application.donor_token_index,
        anchor_occurrence_sha256=(
            None
            if application is None
            else _occurrence_sha256(source.source_sha256, application.anchor_token_index)
        ),
        target_occurrence_sha256=(
            None
            if application is None
            else _occurrence_sha256(source.source_sha256, application.target_token_index)
        ),
        donor_occurrence_sha256=(
            None
            if application is None
            else _occurrence_sha256(source.source_sha256, application.donor_token_index)
        ),
        input_source_sha256=source.source_sha256,
        edited_source_sha256=None if edited is None else edited.source_sha256,
        edited_source=None if edited is None else edited.source,
        applicable_schema_indices=tuple(index for index, active in enumerate(applicable) if active),
        schema_support_counts=checkpoint.support_counts,
        schema_raw_scores=raw_scores,
        race_winner_cell=getattr(race, "winner_cell", None),
        race_spike_step=getattr(race, "spike_step", None),
        race_simultaneous_spike_candidates=getattr(race, "simultaneous_spike_candidates", 0),
        race_physical_cell_steps=race_steps,
        cell_gate=cell_gate,
        schema_gate=schema_gate,
        tonic_gate=tonic_gate,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_PROPOSAL_NAMESPACE, partial.record(include_sha256=False)),
    )
    return result


def _stop_proposal(
    checkpoint: FrozenContinualTokenEditSchemaV1,
    source: TokenSourceFrameV1,
    status: str,
    cell_gate: bool,
    schema_gate: bool,
    tonic_gate: bool,
) -> TokenEditProposalV1:
    class _StoppedRace:
        winner_cell = None
        spike_step = None
        simultaneous_spike_candidates = 0

    return _proposal(
        checkpoint=checkpoint,
        source=source,
        status=status,
        winner=None,
        application=None,
        applicable=tuple(False for _ in PARAMETERIZED_SCHEMAS),
        raw_scores=tuple(0 for _ in PARAMETERIZED_SCHEMAS),
        race=_StoppedRace(),
        race_steps=0,
        cell_gate=cell_gate,
        schema_gate=schema_gate,
        tonic_gate=tonic_gate,
    )


def _derive_token_edit_proposal(
    checkpoint: FrozenContinualTokenEditSchemaV1,
    source: TokenSourceFrameV1,
    *,
    cell_gate: bool,
    schema_gate: bool,
    tonic_gate: bool,
) -> TokenEditProposalV1:
    seen_sources = {
        value
        for row in checkpoint.provenance
        for value in (row.failed_source_sha256, row.successful_source_sha256)
    }
    if not cell_gate:
        return _stop_proposal(
            checkpoint,
            source,
            STOP_CELL_OFF,
            cell_gate,
            schema_gate,
            tonic_gate,
        )
    if not schema_gate:
        return _stop_proposal(
            checkpoint,
            source,
            STOP_SCHEMA_OFF,
            cell_gate,
            schema_gate,
            tonic_gate,
        )
    if source.source_sha256 in seen_sources:
        return _stop_proposal(
            checkpoint,
            source,
            STOP_SEEN_SOURCE,
            cell_gate,
            schema_gate,
            tonic_gate,
        )

    applications = tuple(_apply_schema(schema, source) for schema in PARAMETERIZED_SCHEMAS)
    applicable = tuple(application is not None for application in applications)
    raw_scores = tuple(
        min(common_events.MAX_OUTPUT_SCORE, checkpoint.support_counts[index] * 2)
        if applicable[index]
        else 0
        for index in range(len(PARAMETERIZED_SCHEMAS))
    )
    race, race_steps = _run_schema_race(raw_scores, tonic_gate=tonic_gate)
    maximum_support = max(
        (checkpoint.support_counts[index] for index, active in enumerate(applicable) if active),
        default=0,
    )
    winner: int | None = None
    if not any(applicable):
        status = STOP_NO_SCHEMA_MATCH
    elif race.simultaneous_spike_candidates != 1:
        status = (
            STOP_INSUFFICIENT_INDEPENDENT_SOURCES
            if tonic_gate and maximum_support < MIN_INDEPENDENT_EPISODES
            else STOP_PHYSICAL_RACE
        )
    elif race.winner_cell == len(PARAMETERIZED_SCHEMAS):
        status = STOP_INSUFFICIENT_INDEPENDENT_SOURCES
    else:
        status = READY if tonic_gate else CONTROL_TONIC_OFF_READY
        winner = race.winner_cell
    application = None if winner is None else applications[winner]
    if winner is not None and application is None:
        raise RuntimeError("physical token-edit winner had no source binding")
    if application is not None and application.edited_source.source_sha256 in seen_sources:
        status = STOP_ARTIFACT_REPLAY
        winner = None
        application = None
    return _proposal(
        checkpoint=checkpoint,
        source=source,
        status=status,
        winner=winner,
        application=application,
        applicable=applicable,
        raw_scores=raw_scores,
        race=race,
        race_steps=race_steps,
        cell_gate=cell_gate,
        schema_gate=schema_gate,
        tonic_gate=tonic_gate,
    )


def _freeze_checkpoint(
    *,
    support: np.ndarray,
    provenance: tuple[TokenEditSchemaProvenanceV1, ...],
    parent_checkpoint_sha256: str | None,
    mode: str,
) -> FrozenContinualTokenEditSchemaV1:
    summary_record = _training_summary_record(
        support=support,
        provenance=provenance,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        mode=mode,
    )
    checkpoint = _checkpoint_sha256(support, provenance, summary_record)
    summary = TokenEditSchemaTrainingSummaryV1(
        **summary_record,
        checkpoint_sha256=checkpoint,
    )
    result = FrozenContinualTokenEditSchemaV1(
        support=support,
        provenance=provenance,
        summary=summary,
    )
    result.authenticate()
    return result


def start_continual_token_edit_schema_v1() -> FrozenContinualTokenEditSchemaV1:
    return _freeze_checkpoint(
        support=np.zeros((len(PARAMETERIZED_SCHEMAS),), dtype=TICK_DTYPE),
        provenance=(),
        parent_checkpoint_sha256=None,
        mode="EMPTY_CONTINUAL_TOKEN_EDIT_SCHEMA_V1",
    )


def _episode_is_independent(
    episode: FailedSuccessfulTokenEditEpisodeV1,
    provenance: Sequence[TokenEditSchemaProvenanceV1],
) -> bool:
    episode_context_roots = {
        episode.repository_sha256,
        episode.task_sha256,
        episode.task_instance_sha256,
        episode.task_source_sha256,
        episode.test_source_sha256,
        episode.failed_source.source_sha256,
        episode.successful_source.source_sha256,
    }
    return all(
        episode_context_roots.isdisjoint(
            {
                row.repository_sha256,
                row.task_sha256,
                row.task_instance_sha256,
                row.task_source_sha256,
                row.test_source_sha256,
                row.failed_source_sha256,
                row.successful_source_sha256,
            }
        )
        for row in provenance
    )


def _reject_replay(
    episode: FailedSuccessfulTokenEditEpisodeV1,
    provenance: Sequence[TokenEditSchemaProvenanceV1],
) -> None:
    roots = {
        episode.sha256,
        episode.failed_consequence.execution_sha256,
        episode.successful_consequence.execution_sha256,
        episode.failed_consequence.feedback_sha256,
        episode.successful_consequence.feedback_sha256,
        episode.materialization_sha256,
    }
    for row in provenance:
        previous = {
            row.episode_sha256,
            row.failed_execution_sha256,
            row.successful_execution_sha256,
            row.failed_feedback_sha256,
            row.successful_feedback_sha256,
            row.materialization_sha256,
        }
        if roots.intersection(previous):
            raise ContinualTokenEditSchemaError("token-edit episode or receipt replay is forbidden")


def advance_continual_token_edit_schema_v1(
    parent: FrozenContinualTokenEditSchemaV1,
    episode: FailedSuccessfulTokenEditEpisodeV1,
    *,
    plasticity_gate: bool = True,
    consequence_gate: bool = True,
) -> FrozenContinualTokenEditSchemaV1:
    if type(parent) is not FrozenContinualTokenEditSchemaV1:
        raise TypeError("token-edit advance requires one exact frozen parent")
    if type(episode) is not FailedSuccessfulTokenEditEpisodeV1:
        raise TypeError("token-edit advance requires one exact completed episode")
    if type(plasticity_gate) is not bool or type(consequence_gate) is not bool:
        raise TypeError("token-edit advance gates must be exact booleans")
    parent.authenticate()
    episode.authenticate()
    _reject_replay(episode, parent.provenance)
    parent_root = parent.checkpoint_sha256
    parent_array_root = parent.checkpoint_array_sha256
    if not plasticity_gate or not consequence_gate:
        return parent
    if len(parent.provenance) == MAX_EPISODES:
        raise ContinualTokenEditSchemaError("token-edit provenance capacity exhausted")

    matched, actuations, trace_comparisons, projection_steps = _matched_schemas(episode)
    independent = int(_episode_is_independent(episode, parent.provenance))
    partial = TokenEditSchemaProvenanceV1(
        episode_sha256=episode.sha256,
        parent_checkpoint_sha256=parent_root,
        repository_sha256=episode.repository_sha256,
        task_sha256=episode.task_sha256,
        task_instance_sha256=episode.task_instance_sha256,
        task_source_sha256=episode.task_source_sha256,
        test_source_sha256=episode.test_source_sha256,
        failed_source_sha256=episode.failed_source.source_sha256,
        successful_source_sha256=episode.successful_source.source_sha256,
        failed_execution_sha256=episode.failed_consequence.execution_sha256,
        successful_execution_sha256=episode.successful_consequence.execution_sha256,
        failed_feedback_sha256=episode.failed_consequence.feedback_sha256,
        successful_feedback_sha256=episode.successful_consequence.feedback_sha256,
        materialization_sha256=episode.materialization_sha256,
        matched_schema_indices=matched,
        independent_cell=independent,
        candidate_actuation_count=actuations,
        token_trace_cell_comparison_count=trace_comparisons,
        candidate_projection_cell_steps=projection_steps,
        sha256="",
    )
    provenance_row = replace(
        partial,
        sha256=_digest(_PROVENANCE_NAMESPACE, partial.record(include_sha256=False)),
    )
    provenance_row.authenticate()
    support = np.array(parent._support, copy=True, order="C")
    if independent:
        for index in matched:
            support[index] = np.uint8(min(TICK_CAP, int(support[index]) + 1))
    successor = _freeze_checkpoint(
        support=support,
        provenance=(*parent.provenance, provenance_row),
        parent_checkpoint_sha256=parent_root,
        mode="REPLAY_FREE_CONTINUAL_TOKEN_EDIT_ADVANCE_V1",
    )
    parent.authenticate()
    if (
        parent.checkpoint_sha256 != parent_root
        or parent.checkpoint_array_sha256 != parent_array_root
    ):
        raise RuntimeError("token-edit advance mutated its parent")
    return successor


__all__ = [
    "CLAIM_BOUNDARY",
    "CONFIG_SHA256",
    "CONTROL_TONIC_OFF_READY",
    "MIN_INDEPENDENT_EPISODES",
    "PARAMETERIZED_SCHEMAS",
    "READY",
    "SCHEMA",
    "STOP_ARTIFACT_REPLAY",
    "STOP_CELL_OFF",
    "STOP_INSUFFICIENT_INDEPENDENT_SOURCES",
    "STOP_NO_SCHEMA_MATCH",
    "STOP_PHYSICAL_RACE",
    "STOP_SCHEMA_OFF",
    "STOP_SEEN_SOURCE",
    "ContinualTokenEditSchemaError",
    "FailedSuccessfulTokenEditEpisodeV1",
    "FrozenContinualTokenEditSchemaV1",
    "ParameterizedTokenEditSchemaV1",
    "TokenEditConsequenceV1",
    "TokenEditProposalV1",
    "TokenEditSchemaProvenanceV1",
    "TokenEditSchemaTrainingSummaryV1",
    "TokenSourceFrameV1",
    "advance_continual_token_edit_schema_v1",
    "make_token_edit_consequence_v1",
    "project_token_source_v1",
    "seal_failed_successful_token_edit_episode_v1",
    "start_continual_token_edit_schema_v1",
]
