from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from experiments.r3_pilot import common_ascii_word_unigram_cues_v1 as ascii_cues
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as ordered
from experiments.r3_pilot import common_rustc_lexical_unordered_cue_adapter_v1 as c9
from experiments.r3_pilot import common_unordered_cue_factor_memory_v1 as unordered

SCHEMA = "COMMON_RUSTC_MESSAGE_LINE_UNORDERED_CUE_ADAPTER_V1"
CLAIM_BOUNDARY = (
    "authenticated-rustc-parent-message-and-complete-containing-source-line-"
    "lowercase-ascii-alphanumeric-word-unigrams-as-one-role-neutral-unordered-"
    "cue-set-not-error-code-region-role-provider-target-label-output-edit-"
    "candidate-learned-tokenization-localization-diagnosis-repair-semantics-"
    "source-synthesis-swe-or-intelligence"
)

MAX_TOTAL_CUES = 16

_SHA256_HEX = frozenset("0123456789abcdef")
_OBSERVATION_CONTEXT_NAMESPACE = b"common-rustc-message-line-observation-context-v1"
_ENVELOPE_NAMESPACE = b"common-rustc-message-line-unordered-envelope-v1"


class RustcMessageLineUnorderedCueAdapterError(ValueError):
    """Exact rustc message-and-line cue provenance failed authentication."""


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


def _rooted_observation_context(
    attempt: ordered.RustcDiagnosticCompileAttemptV1,
    selection_context_sha256: str,
) -> str:
    if not _valid_sha256(selection_context_sha256):
        raise RustcMessageLineUnorderedCueAdapterError("selection context root is invalid")
    return _digest(
        _OBSERVATION_CONTEXT_NAMESPACE,
        {
            "selection_context_sha256": selection_context_sha256,
            "rustc_identity_sha256": attempt.rustc_identity_sha256,
            "compiler_policy_sha256": attempt.compiler_policy_sha256,
        },
    )


def _sole_primary_span(
    attempt: ordered.RustcDiagnosticCompileAttemptV1,
) -> ordered.RustcPrimarySpanV1:
    if len(attempt.error_diagnostics) != 1:
        raise RustcMessageLineUnorderedCueAdapterError(
            "message-line projection requires exactly one retained error diagnostic"
        )
    spans = attempt.error_diagnostics[0].primary_spans
    if len(spans) != 1:
        raise RustcMessageLineUnorderedCueAdapterError(
            "message-line projection requires exactly one primary span"
        )
    return spans[0]


def _source_bytes(
    attempt: ordered.RustcDiagnosticCompileAttemptV1,
    source: str,
) -> bytes:
    if type(source) is not str:
        raise TypeError("message-line projection source must be one exact string")
    try:
        result = source.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise RustcMessageLineUnorderedCueAdapterError(
            "message-line projection source must be ASCII"
        ) from exc
    if (
        len(result) != attempt.source_size_bytes
        or sha256(result).hexdigest() != attempt.source_sha256
    ):
        raise RustcMessageLineUnorderedCueAdapterError(
            "source does not match the authenticated rustc attempt"
        )
    return result


def _derive_projection(
    attempt: ordered.RustcDiagnosticCompileAttemptV1,
    *,
    stderr: bytes,
    source: str,
    selection_context_sha256: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    ascii_cues.AsciiWordUnigramCueEnvelopeV1,
    unordered.UnorderedCueFactorObservationV1,
]:
    try:
        c9._authenticate_attempt_stderr(attempt, stderr)
    except c9.RustcLexicalUnorderedCueAdapterError as exc:
        raise RustcMessageLineUnorderedCueAdapterError(
            "rustc attempt and stderr authentication failed"
        ) from exc

    primary_span = _sole_primary_span(attempt)
    source_bytes = _source_bytes(attempt, source)
    try:
        sealed_span = ascii_cues.seal_ascii_byte_span_v1(
            source_bytes,
            byte_start=primary_span.byte_start,
            byte_end=primary_span.byte_end,
        )
        source_line_envelope = ascii_cues.project_ascii_word_unigram_cues_v1(
            source_bytes,
            span=sealed_span,
        )
    except ascii_cues.AsciiWordUnigramCueError as exc:
        raise RustcMessageLineUnorderedCueAdapterError(
            "authenticated primary span could not project complete source-line cues"
        ) from exc

    try:
        message_tokens = c9._parent_message_tokens(stderr=stderr, attempt=attempt)
    except c9.RustcLexicalUnorderedCueAdapterError as exc:
        raise RustcMessageLineUnorderedCueAdapterError(
            "authenticated parent message token projection failed"
        ) from exc
    message_cue_sha256s = tuple(
        ascii_cues.ascii_word_unigram_cue_sha256_v1(token) for token in message_tokens
    )
    source_line_tokens = source_line_envelope.tokens
    source_line_cue_sha256s = source_line_envelope.cue_sha256s

    distinct_tokens = set((*message_tokens, *source_line_tokens))
    merged_cues = tuple(sorted({*message_cue_sha256s, *source_line_cue_sha256s}))
    if len(merged_cues) != len(distinct_tokens):
        raise RustcMessageLineUnorderedCueAdapterError(
            "distinct message-line tokens did not retain distinct generic cue roots"
        )
    if len(merged_cues) > MAX_TOTAL_CUES:
        raise RustcMessageLineUnorderedCueAdapterError("merged message-line cue budget exceeded")
    observation = unordered.seal_unordered_cue_factor_observation_v1(
        selection_context_sha256=_rooted_observation_context(
            attempt,
            selection_context_sha256,
        ),
        cue_sha256s=merged_cues,
    )
    return (
        message_tokens,
        message_cue_sha256s,
        source_line_tokens,
        source_line_cue_sha256s,
        source_line_envelope,
        observation,
    )


@dataclass(frozen=True, slots=True)
class RustcMessageLineUnorderedCueEnvelopeV1:
    """Frozen provenance rederiving rustc message + source line -> one cue set."""

    selection_context_sha256: str
    attempt: ordered.RustcDiagnosticCompileAttemptV1
    stderr: bytes
    source: str
    message_tokens: tuple[str, ...]
    message_cue_sha256s: tuple[str, ...]
    source_line_tokens: tuple[str, ...]
    source_line_cue_sha256s: tuple[str, ...]
    source_line_envelope: ascii_cues.AsciiWordUnigramCueEnvelopeV1
    observation: unordered.UnorderedCueFactorObservationV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "selection_context_sha256": self.selection_context_sha256,
            "attempt_sha256": self.attempt.sha256,
            "stderr_sha256": sha256(self.stderr).hexdigest(),
            "source": self.source,
            "message_tokens": self.message_tokens,
            "message_cue_sha256s": self.message_cue_sha256s,
            "source_line_tokens": self.source_line_tokens,
            "source_line_cue_sha256s": self.source_line_cue_sha256s,
            "source_line_envelope": self.source_line_envelope.record(),
            "observation": self.observation.record(),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.attempt) is not ordered.RustcDiagnosticCompileAttemptV1:
            raise TypeError("message-line envelope rustc attempt must be exact")
        if type(self.stderr) is not bytes:
            raise TypeError("message-line envelope stderr must be exact bytes")
        if type(self.source) is not str:
            raise TypeError("message-line envelope source must be one exact string")
        if any(
            type(value) is not tuple
            for value in (
                self.message_tokens,
                self.message_cue_sha256s,
                self.source_line_tokens,
                self.source_line_cue_sha256s,
            )
        ):
            raise TypeError("message-line token and cue inventories must be exact tuples")
        if type(self.source_line_envelope) is not ascii_cues.AsciiWordUnigramCueEnvelopeV1:
            raise TypeError("message-line source-line envelope must be exact")
        if type(self.observation) is not unordered.UnorderedCueFactorObservationV1:
            raise TypeError("message-line unordered observation must be exact")
        if not _valid_sha256(self.sha256):
            raise RustcMessageLineUnorderedCueAdapterError("message-line envelope root is invalid")

        (
            message_tokens,
            message_cue_sha256s,
            source_line_tokens,
            source_line_cue_sha256s,
            source_line_envelope,
            observation,
        ) = _derive_projection(
            self.attempt,
            stderr=self.stderr,
            source=self.source,
            selection_context_sha256=self.selection_context_sha256,
        )
        if (
            self.message_tokens != message_tokens
            or self.message_cue_sha256s != message_cue_sha256s
            or self.source_line_tokens != source_line_tokens
            or self.source_line_cue_sha256s != source_line_cue_sha256s
            or self.source_line_envelope != source_line_envelope
            or self.observation != observation
            or self.sha256 != _digest(_ENVELOPE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise RustcMessageLineUnorderedCueAdapterError(
                "rustc message-line unordered cue envelope authentication failed"
            )


def project_rustc_message_line_unordered_cue_observation_v1(
    attempt: ordered.RustcDiagnosticCompileAttemptV1,
    *,
    stderr: bytes,
    source: str,
    selection_context_sha256: str,
) -> RustcMessageLineUnorderedCueEnvelopeV1:
    """Project authenticated parent-message and containing-line word cues."""

    (
        message_tokens,
        message_cue_sha256s,
        source_line_tokens,
        source_line_cue_sha256s,
        source_line_envelope,
        observation,
    ) = _derive_projection(
        attempt,
        stderr=stderr,
        source=source,
        selection_context_sha256=selection_context_sha256,
    )
    partial = RustcMessageLineUnorderedCueEnvelopeV1(
        selection_context_sha256=selection_context_sha256,
        attempt=attempt,
        stderr=stderr,
        source=source,
        message_tokens=message_tokens,
        message_cue_sha256s=message_cue_sha256s,
        source_line_tokens=source_line_tokens,
        source_line_cue_sha256s=source_line_cue_sha256s,
        source_line_envelope=source_line_envelope,
        observation=observation,
        sha256="",
    )
    result = RustcMessageLineUnorderedCueEnvelopeV1(
        selection_context_sha256=partial.selection_context_sha256,
        attempt=partial.attempt,
        stderr=partial.stderr,
        source=partial.source,
        message_tokens=partial.message_tokens,
        message_cue_sha256s=partial.message_cue_sha256s,
        source_line_tokens=partial.source_line_tokens,
        source_line_cue_sha256s=partial.source_line_cue_sha256s,
        source_line_envelope=partial.source_line_envelope,
        observation=partial.observation,
        sha256=_digest(_ENVELOPE_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


__all__ = [
    "CLAIM_BOUNDARY",
    "MAX_TOTAL_CUES",
    "SCHEMA",
    "RustcMessageLineUnorderedCueAdapterError",
    "RustcMessageLineUnorderedCueEnvelopeV1",
    "project_rustc_message_line_unordered_cue_observation_v1",
]
