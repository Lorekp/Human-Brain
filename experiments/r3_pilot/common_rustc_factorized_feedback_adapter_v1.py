"""Project caller-attested rustc JSON diagnostics into two opaque cue axes.

Rustc error codes form the first axis.  Primary spans into the exact selected
ASCII source are normalized into first-half/second-half cues on the second axis.
Messages, rendered text, labels, explanations, and suggested replacements are
retained only through the raw stderr hash and never enter cognition.

The relative-half projection is explicit project scaffolding, not a rustc or
language-server convention and not learned localization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from experiments.r3_pilot.common_factorized_feedback_hypothesis_memory_v1 import (
    FactorizedFeedbackObservationV1,
    seal_factorized_feedback_observation_v1,
)

SCHEMA = "COMMON_RUSTC_FACTORIZED_FEEDBACK_ADAPTER_V1"
CLAIM_BOUNDARY = (
    "caller-attested-rustc-json-error-code-and-hand-built-relative-source-half-"
    "projection-into-two-opaque-factor-cue-axes-not-message-or-suggestion-use-"
    "learned-localization-diagnosis-repair-selection-source-synthesis-swe-or-intelligence"
)

COMPILED = "COMPILED"
COMPILE_FAILED = "COMPILE_FAILED"
NO_CODE = "NO_CODE"
FIRST_HALF = "FIRST_HALF"
SECOND_HALF = "SECOND_HALF"
UNLOCATED = "UNLOCATED"

MAX_SOURCE_BYTES = 65_536
MAX_STDERR_BYTES = 1_048_576
MAX_ERROR_DIAGNOSTICS = 64
MAX_PRIMARY_SPANS = 64

_SHA256_HEX = frozenset("0123456789abcdef")
_CODE_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
_EMPTY_SHA256 = sha256(b"").hexdigest()

_SPAN_NAMESPACE = b"common-rustc-factorized-feedback-primary-span-v1"
_ERROR_NAMESPACE = b"common-rustc-factorized-feedback-error-v1"
_ATTEMPT_NAMESPACE = b"common-rustc-factorized-feedback-attempt-v1"
_ARGUMENT_POLICY_NAMESPACE = b"common-rustc-factorized-feedback-argument-policy-v1"
_CLASS_CUE_NAMESPACE = b"common-rustc-factorized-feedback-class-cue-v1"
_REGION_CUE_NAMESPACE = b"common-rustc-factorized-feedback-region-cue-v1"
_OBSERVATION_CONTEXT_NAMESPACE = b"common-rustc-factorized-feedback-observation-context-v1"


class RustcFactorizedFeedbackAdapterError(ValueError):
    """A structured compiler attempt or its projection failed authentication."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _valid_code(value: object) -> bool:
    return (
        type(value) is str
        and 1 <= len(value) <= 64
        and all(character in _CODE_CHARACTERS for character in value)
    )


def _argument_policy_sha256(
    source_file_name: str,
    arguments: tuple[str, ...],
) -> str:
    normalized = tuple(
        "<SELECTED_SOURCE>" if value == source_file_name else value for value in arguments
    )
    return _digest(_ARGUMENT_POLICY_NAMESPACE, {"arguments": normalized})


@dataclass(frozen=True, slots=True)
class RustcPrimarySpanV1:
    file_name: str
    byte_start: int
    byte_end: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "file_name": self.file_name,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if (
            type(self.file_name) is not str
            or not self.file_name
            or type(self.byte_start) is not int
            or type(self.byte_end) is not int
            or not 0 <= self.byte_start <= self.byte_end
            or self.sha256 != _digest(_SPAN_NAMESPACE, self.record(include_sha256=False))
        ):
            raise RustcFactorizedFeedbackAdapterError("rustc primary span authentication failed")


def _seal_span(*, file_name: str, byte_start: int, byte_end: int) -> RustcPrimarySpanV1:
    payload = {
        "file_name": file_name,
        "byte_start": byte_start,
        "byte_end": byte_end,
    }
    result = RustcPrimarySpanV1(
        **payload,
        sha256=_digest(_SPAN_NAMESPACE, payload),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class RustcErrorDiagnosticV1:
    level: str
    code: str
    primary_spans: tuple[RustcPrimarySpanV1, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "level": self.level,
            "code": self.code,
            "primary_spans": tuple(value.record() for value in self.primary_spans),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.primary_spans) is not tuple:
            raise TypeError("rustc diagnostic spans must be one exact tuple")
        for span in self.primary_spans:
            if type(span) is not RustcPrimarySpanV1:
                raise TypeError("rustc diagnostic spans must be exact")
            span.authenticate()
        ordered = tuple(
            sorted(
                self.primary_spans,
                key=lambda value: (value.file_name, value.byte_start, value.byte_end),
            )
        )
        if (
            self.level != "error"
            or not _valid_code(self.code)
            or len(self.primary_spans) > MAX_PRIMARY_SPANS
            or len({value.sha256 for value in self.primary_spans}) != len(self.primary_spans)
            or ordered != self.primary_spans
            or self.sha256 != _digest(_ERROR_NAMESPACE, self.record(include_sha256=False))
        ):
            raise RustcFactorizedFeedbackAdapterError(
                "rustc error diagnostic authentication failed"
            )


def _seal_error(
    *,
    code: str,
    primary_spans: tuple[RustcPrimarySpanV1, ...],
) -> RustcErrorDiagnosticV1:
    ordered = tuple(
        sorted(
            {value.sha256: value for value in primary_spans}.values(),
            key=lambda value: (value.file_name, value.byte_start, value.byte_end),
        )
    )
    payload = {
        "level": "error",
        "code": code,
        "primary_spans": ordered,
    }
    result = RustcErrorDiagnosticV1(
        **payload,
        sha256=_digest(
            _ERROR_NAMESPACE,
            {
                **payload,
                "primary_spans": tuple(value.record() for value in ordered),
            },
        ),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class RustcDiagnosticCompileAttemptV1:
    status: str
    rustc_identity_sha256: str
    compiler_policy_sha256: str
    source_file_name: str
    source_sha256: str
    source_size_bytes: int
    arguments: tuple[str, ...]
    returncode: int
    stdout_sha256: str
    stderr_sha256: str
    compile_invocations: int
    error_diagnostics: tuple[RustcErrorDiagnosticV1, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "status": self.status,
            "rustc_identity_sha256": self.rustc_identity_sha256,
            "compiler_policy_sha256": self.compiler_policy_sha256,
            "source_file_name": self.source_file_name,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "arguments": self.arguments,
            "returncode": self.returncode,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "compile_invocations": self.compile_invocations,
            "error_diagnostics": tuple(value.record() for value in self.error_diagnostics),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.error_diagnostics) is not tuple:
            raise TypeError("rustc attempt diagnostics must be one exact tuple")
        for diagnostic in self.error_diagnostics:
            if type(diagnostic) is not RustcErrorDiagnosticV1:
                raise TypeError("rustc attempt diagnostics must be exact")
            diagnostic.authenticate()
        ordered = tuple(sorted(self.error_diagnostics, key=lambda value: value.sha256))
        if (
            self.status not in (COMPILED, COMPILE_FAILED)
            or not _valid_sha256(self.rustc_identity_sha256)
            or type(self.source_file_name) is not str
            or not self.source_file_name
            or "/" in self.source_file_name
            or "\\" in self.source_file_name
            or not _valid_sha256(self.source_sha256)
            or type(self.source_size_bytes) is not int
            or not 1 <= self.source_size_bytes <= MAX_SOURCE_BYTES
            or type(self.arguments) is not tuple
            or any(type(value) is not str or not value for value in self.arguments)
            or self.arguments.count(self.source_file_name) != 1
            or self.arguments.count("--error-format=json") != 1
            or self.compiler_policy_sha256
            != _argument_policy_sha256(self.source_file_name, self.arguments)
            or type(self.returncode) is not int
            or self.returncode < 0
            or self.stdout_sha256 != _EMPTY_SHA256
            or not _valid_sha256(self.stderr_sha256)
            or type(self.compile_invocations) is not int
            or self.compile_invocations != 1
            or len(self.error_diagnostics) > MAX_ERROR_DIAGNOSTICS
            or ordered != self.error_diagnostics
            or len({value.sha256 for value in ordered}) != len(ordered)
            or any(
                span.file_name != self.source_file_name or span.byte_end > self.source_size_bytes
                for diagnostic in ordered
                for span in diagnostic.primary_spans
            )
            or (self.status == COMPILED and (self.returncode != 0 or ordered))
            or (self.status == COMPILE_FAILED and (self.returncode == 0 or not ordered))
            or self.sha256 != _digest(_ATTEMPT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise RustcFactorizedFeedbackAdapterError(
                "rustc diagnostic compile attempt authentication failed"
            )


def _parse_error_diagnostics(
    *,
    stderr: bytes,
    source_file_name: str,
    source_size_bytes: int,
) -> tuple[RustcErrorDiagnosticV1, ...]:
    rows: list[RustcErrorDiagnosticV1] = []
    try:
        lines = stderr.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RustcFactorizedFeedbackAdapterError("rustc JSON stderr is not UTF-8") from exc
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RustcFactorizedFeedbackAdapterError(
                "rustc JSON stderr contains a non-JSON line"
            ) from exc
        if type(record) is not dict:
            raise RustcFactorizedFeedbackAdapterError("rustc JSON message must be one object")
        if record.get("$message_type") != "diagnostic" or record.get("level") != "error":
            continue
        code_record = record.get("code")
        if code_record is None:
            code = NO_CODE
        elif type(code_record) is dict and _valid_code(code_record.get("code")):
            code = code_record["code"]
        else:
            raise RustcFactorizedFeedbackAdapterError("rustc diagnostic code is malformed")
        spans_record = record.get("spans")
        if type(spans_record) is not list:
            raise RustcFactorizedFeedbackAdapterError("rustc diagnostic spans are malformed")
        spans: list[RustcPrimarySpanV1] = []
        for span_record in spans_record:
            if type(span_record) is not dict:
                raise RustcFactorizedFeedbackAdapterError("rustc diagnostic span is malformed")
            if span_record.get("is_primary") is not True:
                continue
            file_name = span_record.get("file_name")
            byte_start = span_record.get("byte_start")
            byte_end = span_record.get("byte_end")
            if (
                type(file_name) is not str
                or type(byte_start) is not int
                or type(byte_end) is not int
                or byte_start < 0
                or byte_end < byte_start
            ):
                raise RustcFactorizedFeedbackAdapterError(
                    "rustc primary span coordinates are malformed"
                )
            if file_name != source_file_name:
                continue
            if byte_end > source_size_bytes:
                raise RustcFactorizedFeedbackAdapterError(
                    "rustc primary span exceeds selected source"
                )
            spans.append(
                _seal_span(
                    file_name=file_name,
                    byte_start=byte_start,
                    byte_end=byte_end,
                )
            )
        rows.append(_seal_error(code=code, primary_spans=tuple(spans)))
    unique = {value.sha256: value for value in rows}
    result = tuple(sorted(unique.values(), key=lambda value: value.sha256))
    substantive = tuple(value for value in result if value.code != NO_CODE or value.primary_spans)
    if substantive:
        # Rustc appends a structurally empty uncoded abort summary to ordinary
        # failures. Retain an empty NO_CODE only when it is the sole evidence.
        result = substantive
    if len(result) > MAX_ERROR_DIAGNOSTICS:
        raise RustcFactorizedFeedbackAdapterError("rustc error diagnostic capacity exceeded")
    return result


def seal_rustc_diagnostic_compile_attempt_v1(
    *,
    rustc_identity_sha256: str,
    source_file_name: str,
    source: bytes,
    arguments: tuple[str, ...],
    returncode: int,
    stdout: bytes,
    stderr: bytes,
    compile_invocations: int = 1,
) -> RustcDiagnosticCompileAttemptV1:
    if type(source) is not bytes or type(stdout) is not bytes or type(stderr) is not bytes:
        raise TypeError("rustc attempt source and streams must be exact bytes")
    if type(arguments) is not tuple:
        raise TypeError("rustc attempt arguments must be one exact tuple")
    if not 1 <= len(source) <= MAX_SOURCE_BYTES or len(stderr) > MAX_STDERR_BYTES:
        raise RustcFactorizedFeedbackAdapterError(
            "rustc attempt source or stderr capacity exceeded"
        )
    try:
        source.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RustcFactorizedFeedbackAdapterError(
            "relative-half projection requires exact ASCII source"
        ) from exc
    if stdout:
        raise RustcFactorizedFeedbackAdapterError(
            "rustc diagnostic compile unexpectedly wrote stdout"
        )
    errors = _parse_error_diagnostics(
        stderr=stderr,
        source_file_name=source_file_name,
        source_size_bytes=len(source),
    )
    status = COMPILED if returncode == 0 else COMPILE_FAILED
    payload = {
        "status": status,
        "rustc_identity_sha256": rustc_identity_sha256,
        "compiler_policy_sha256": _argument_policy_sha256(
            source_file_name,
            arguments,
        ),
        "source_file_name": source_file_name,
        "source_sha256": sha256(source).hexdigest(),
        "source_size_bytes": len(source),
        "arguments": arguments,
        "returncode": returncode,
        "stdout_sha256": sha256(stdout).hexdigest(),
        "stderr_sha256": sha256(stderr).hexdigest(),
        "compile_invocations": compile_invocations,
        "error_diagnostics": errors,
    }
    result = RustcDiagnosticCompileAttemptV1(
        **payload,
        sha256=_digest(
            _ATTEMPT_NAMESPACE,
            {
                "schema": SCHEMA,
                "claim_boundary": CLAIM_BOUNDARY,
                **payload,
                "error_diagnostics": tuple(value.record() for value in errors),
            },
        ),
    )
    result.authenticate()
    return result


def _class_cue_sha256(attempt: RustcDiagnosticCompileAttemptV1, code: str) -> str:
    return _digest(
        _CLASS_CUE_NAMESPACE,
        {
            "stage": "compile",
            "level": "error",
            "code": code,
            "rustc_identity_sha256": attempt.rustc_identity_sha256,
            "compiler_policy_sha256": attempt.compiler_policy_sha256,
        },
    )


def _region_cue_sha256(attempt: RustcDiagnosticCompileAttemptV1, region: str) -> str:
    return _digest(
        _REGION_CUE_NAMESPACE,
        {
            "stage": "compile",
            "projection": "relative-source-half-v1",
            "region": region,
            "rustc_identity_sha256": attempt.rustc_identity_sha256,
            "compiler_policy_sha256": attempt.compiler_policy_sha256,
        },
    )


def project_rust_factorized_feedback_v1(
    attempt: RustcDiagnosticCompileAttemptV1,
    *,
    selection_context_sha256: str,
    active_hypothesis_sha256s: tuple[str, ...],
) -> FactorizedFeedbackObservationV1:
    if type(attempt) is not RustcDiagnosticCompileAttemptV1:
        raise TypeError("feedback projection requires one exact rustc attempt")
    attempt.authenticate()
    if attempt.status != COMPILE_FAILED:
        raise RustcFactorizedFeedbackAdapterError(
            "only a structured compile failure produces repair cues"
        )
    if not _valid_sha256(selection_context_sha256):
        raise RustcFactorizedFeedbackAdapterError("selection context root is invalid")
    class_cues = tuple(
        sorted(
            {
                _class_cue_sha256(attempt, diagnostic.code)
                for diagnostic in attempt.error_diagnostics
            }
        )
    )
    regions: set[str] = set()
    for diagnostic in attempt.error_diagnostics:
        for span in diagnostic.primary_spans:
            midpoint = (span.byte_start + span.byte_end) // 2
            regions.add(FIRST_HALF if 2 * midpoint < attempt.source_size_bytes else SECOND_HALF)
    if not regions:
        regions.add(UNLOCATED)
    region_cues = tuple(sorted(_region_cue_sha256(attempt, region) for region in regions))
    rooted_context = _digest(
        _OBSERVATION_CONTEXT_NAMESPACE,
        {
            "selection_context_sha256": selection_context_sha256,
            "attempt_sha256": attempt.sha256,
        },
    )
    return seal_factorized_feedback_observation_v1(
        selection_context_sha256=rooted_context,
        axis_cue_sha256s=(class_cues, region_cues),
        active_hypothesis_sha256s=active_hypothesis_sha256s,
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "COMPILED",
    "COMPILE_FAILED",
    "FIRST_HALF",
    "MAX_ERROR_DIAGNOSTICS",
    "MAX_PRIMARY_SPANS",
    "MAX_SOURCE_BYTES",
    "MAX_STDERR_BYTES",
    "NO_CODE",
    "SCHEMA",
    "SECOND_HALF",
    "UNLOCATED",
    "RustcDiagnosticCompileAttemptV1",
    "RustcErrorDiagnosticV1",
    "RustcFactorizedFeedbackAdapterError",
    "RustcPrimarySpanV1",
    "project_rust_factorized_feedback_v1",
    "seal_rustc_diagnostic_compile_attempt_v1",
]
