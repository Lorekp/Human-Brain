"""Project parent rustc error messages plus the established region cue.

The exact caller-attested JSON stderr remains environment provenance.  Only
lowercase ASCII-alphanumeric word unigrams from top-level parent ``message``
fields enter cognition.  Rustc codes, rendered diagnostics, child messages,
and suggested replacements do not.  Backticks and underscores are ordinary
delimiters; their contents receive no identifier-specific treatment.

The relative-source-half cue is reused unchanged from the established ordered
adapter.  This is fixed lexical/region extraction, not learned tokenization,
localization, diagnosis, or repair semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as ordered
from experiments.r3_pilot.common_unordered_cue_factor_memory_v1 import (
    UnorderedCueFactorObservationV1,
    seal_unordered_cue_factor_observation_v1,
)

SCHEMA = "COMMON_RUSTC_LEXICAL_UNORDERED_CUE_ADAPTER_V1"
CLAIM_BOUNDARY = (
    "caller-attested-rustc-parent-error-message-ascii-word-unigrams-and-established-"
    "relative-source-half-cue-not-error-code-rendered-child-suggestion-replacement-"
    "use-learned-tokenization-localization-diagnosis-repair-semantics-source-"
    "synthesis-swe-or-intelligence"
)

MAX_LEXICAL_FEATURES = 15
MAX_TOTAL_CUES = 16
MAX_PARENT_MESSAGE_UTF8_BYTES = 8_192
MAX_LEXICAL_TOKEN_CHARACTERS = 64

_LEXICAL_CUE_NAMESPACE = b"common-rustc-lexical-unordered-cue-v1"
_OBSERVATION_CONTEXT_NAMESPACE = b"common-rustc-lexical-observation-context-v1"
_ENVELOPE_NAMESPACE = b"common-rustc-lexical-unordered-envelope-v1"

# The ordered projector requires a non-empty hypothesis bank even though its
# region projection does not depend on bank membership.  Authentication uses
# this fixed environment-only root and discards the ordered observation.
_REGION_REAUTH_HYPOTHESIS_SHA256 = sha256(
    b"common-rustc-lexical-region-reauth-hypothesis-v1"
).hexdigest()


class RustcLexicalUnorderedCueAdapterError(ValueError):
    """Exact rustc lexical projection provenance failed authentication."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _flush_ascii_word(current: list[str], tokens: set[str]) -> None:
    if not current:
        return
    token = "".join(current).lower()
    current.clear()
    if len(token) > MAX_LEXICAL_TOKEN_CHARACTERS:
        raise RustcLexicalUnorderedCueAdapterError(
            "parent diagnostic lexical token capacity exceeded"
        )
    tokens.add(token)


def _ascii_word_unigrams(
    message: str,
) -> tuple[str, ...]:
    """Fixed lowercase ASCII-alphanumeric unigram tokenizer."""

    if type(message) is not str or not message:
        raise RustcLexicalUnorderedCueAdapterError(
            "parent error diagnostic message must be one non-empty string"
        )
    if len(message.encode("utf-8")) > MAX_PARENT_MESSAGE_UTF8_BYTES:
        raise RustcLexicalUnorderedCueAdapterError(
            "parent error diagnostic message capacity exceeded"
        )

    tokens: set[str] = set()
    current: list[str] = []
    for character in message:
        if "A" <= character <= "Z" or "a" <= character <= "z" or "0" <= character <= "9":
            current.append(character)
        else:
            _flush_ascii_word(current, tokens)
    _flush_ascii_word(current, tokens)
    return tuple(sorted(tokens))


def _parent_message_tokens(
    *,
    stderr: bytes,
    attempt: ordered.RustcDiagnosticCompileAttemptV1,
) -> tuple[str, ...]:
    try:
        lines = stderr.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RustcLexicalUnorderedCueAdapterError("rustc JSON stderr is not UTF-8") from exc

    retained_roots = {value.sha256 for value in attempt.error_diagnostics}
    represented_roots: set[str] = set()
    tokens: set[str] = set()
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RustcLexicalUnorderedCueAdapterError(
                "rustc JSON stderr contains a non-JSON line"
            ) from exc
        if type(record) is not dict:
            raise RustcLexicalUnorderedCueAdapterError("rustc JSON message must be one object")
        if record.get("$message_type") != "diagnostic" or record.get("level") != "error":
            continue

        parsed_row = ordered._parse_error_diagnostics(
            stderr=line.encode("utf-8"),
            source_file_name=attempt.source_file_name,
            source_size_bytes=attempt.source_size_bytes,
        )
        matching = {value.sha256 for value in parsed_row}.intersection(retained_roots)
        if not matching:
            # Excludes rustc's uncoded empty abort summary when the established
            # parser excluded it from the authenticated attempt.
            continue
        represented_roots.update(matching)
        tokens.update(_ascii_word_unigrams(record.get("message")))

    result = tuple(sorted(tokens))
    if represented_roots != retained_roots or not result:
        raise RustcLexicalUnorderedCueAdapterError(
            "authenticated parent diagnostics yielded incomplete lexical evidence"
        )
    if len(result) > MAX_LEXICAL_FEATURES:
        raise RustcLexicalUnorderedCueAdapterError(
            "parent diagnostic lexical feature budget exceeded"
        )
    return result


def _lexical_cue_sha256s(
    attempt: ordered.RustcDiagnosticCompileAttemptV1,
    tokens: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            _digest(
                _LEXICAL_CUE_NAMESPACE,
                {
                    "stage": "compile",
                    "projection": "parent-error-message-ascii-word-unigram-v1",
                    "token": token,
                    "rustc_identity_sha256": attempt.rustc_identity_sha256,
                    "compiler_policy_sha256": attempt.compiler_policy_sha256,
                },
            )
            for token in tokens
        )
    )


def _rooted_observation_context(
    attempt: ordered.RustcDiagnosticCompileAttemptV1,
    selection_context_sha256: str,
) -> str:
    return _digest(
        _OBSERVATION_CONTEXT_NAMESPACE,
        {
            "selection_context_sha256": selection_context_sha256,
            "rustc_identity_sha256": attempt.rustc_identity_sha256,
            "compiler_policy_sha256": attempt.compiler_policy_sha256,
        },
    )


def _authenticate_attempt_stderr(
    attempt: ordered.RustcDiagnosticCompileAttemptV1,
    stderr: bytes,
) -> None:
    if type(attempt) is not ordered.RustcDiagnosticCompileAttemptV1:
        raise TypeError("lexical projection requires one exact rustc attempt")
    if type(stderr) is not bytes:
        raise TypeError("lexical projection requires exact raw stderr bytes")
    attempt.authenticate()
    if attempt.status != ordered.COMPILE_FAILED:
        raise RustcLexicalUnorderedCueAdapterError(
            "only a structured compile failure produces lexical repair cues"
        )
    if (
        len(stderr) > ordered.MAX_STDERR_BYTES
        or sha256(stderr).hexdigest() != attempt.stderr_sha256
    ):
        raise RustcLexicalUnorderedCueAdapterError(
            "raw rustc stderr does not match the authenticated attempt"
        )
    parsed = ordered._parse_error_diagnostics(
        stderr=stderr,
        source_file_name=attempt.source_file_name,
        source_size_bytes=attempt.source_size_bytes,
    )
    if parsed != attempt.error_diagnostics:
        raise RustcLexicalUnorderedCueAdapterError(
            "raw rustc stderr diagnostics differ from the authenticated attempt"
        )


def _ordered_region_cue_sha256s(
    attempt: ordered.RustcDiagnosticCompileAttemptV1,
    *,
    selection_context_sha256: str,
    active_hypothesis_sha256s: tuple[str, ...],
) -> tuple[str, ...]:
    projected = ordered.project_rust_factorized_feedback_v1(
        attempt,
        selection_context_sha256=selection_context_sha256,
        active_hypothesis_sha256s=active_hypothesis_sha256s,
    )
    return projected.axis_cue_sha256s[1]


@dataclass(frozen=True, slots=True)
class RustcLexicalUnorderedCueEnvelopeV1:
    """Frozen provenance that can rederive stderr -> tokens -> observation."""

    selection_context_sha256: str
    attempt: ordered.RustcDiagnosticCompileAttemptV1
    stderr: bytes
    lexical_tokens: tuple[str, ...]
    lexical_cue_sha256s: tuple[str, ...]
    region_cue_sha256s: tuple[str, ...]
    observation: UnorderedCueFactorObservationV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "selection_context_sha256": self.selection_context_sha256,
            "attempt_sha256": self.attempt.sha256,
            "stderr_sha256": sha256(self.stderr).hexdigest(),
            "lexical_tokens": self.lexical_tokens,
            "lexical_cue_sha256s": self.lexical_cue_sha256s,
            "region_cue_sha256s": self.region_cue_sha256s,
            "observation": self.observation.record(),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        _authenticate_attempt_stderr(self.attempt, self.stderr)
        if type(self.lexical_tokens) is not tuple:
            raise TypeError("lexical tokens must be one exact tuple")
        if type(self.lexical_cue_sha256s) is not tuple:
            raise TypeError("lexical cue roots must be one exact tuple")
        if type(self.region_cue_sha256s) is not tuple:
            raise TypeError("region cue roots must be one exact tuple")
        if type(self.observation) is not UnorderedCueFactorObservationV1:
            raise TypeError("lexical envelope observation must be exact")
        self.observation.authenticate()

        tokens = _parent_message_tokens(stderr=self.stderr, attempt=self.attempt)
        lexical = _lexical_cue_sha256s(self.attempt, tokens)
        regions = _ordered_region_cue_sha256s(
            self.attempt,
            selection_context_sha256=self.selection_context_sha256,
            active_hypothesis_sha256s=(_REGION_REAUTH_HYPOTHESIS_SHA256,),
        )
        code_cues = {
            ordered._class_cue_sha256(self.attempt, diagnostic.code)
            for diagnostic in self.attempt.error_diagnostics
        }
        combined = tuple(sorted((*lexical, *regions)))
        expected_observation = seal_unordered_cue_factor_observation_v1(
            selection_context_sha256=_rooted_observation_context(
                self.attempt,
                self.selection_context_sha256,
            ),
            cue_sha256s=combined,
        )
        if (
            not ordered._valid_sha256(self.selection_context_sha256)
            or tokens != self.lexical_tokens
            or lexical != self.lexical_cue_sha256s
            or regions != self.region_cue_sha256s
            or len(set(combined)) != len(combined)
            or len(combined) > MAX_TOTAL_CUES
            or set(combined).intersection(code_cues)
            or expected_observation != self.observation
            or self.sha256 != _digest(_ENVELOPE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise RustcLexicalUnorderedCueAdapterError(
                "rustc lexical unordered cue envelope authentication failed"
            )


def project_rustc_lexical_unordered_cue_observation_v1(
    attempt: ordered.RustcDiagnosticCompileAttemptV1,
    *,
    stderr: bytes,
    selection_context_sha256: str,
    active_hypothesis_sha256s: tuple[str, ...],
) -> RustcLexicalUnorderedCueEnvelopeV1:
    """Authenticate stderr and expose only lexical plus established region cues."""

    _authenticate_attempt_stderr(attempt, stderr)
    regions = _ordered_region_cue_sha256s(
        attempt,
        selection_context_sha256=selection_context_sha256,
        active_hypothesis_sha256s=active_hypothesis_sha256s,
    )
    tokens = _parent_message_tokens(stderr=stderr, attempt=attempt)
    lexical = _lexical_cue_sha256s(attempt, tokens)
    combined = tuple(sorted((*lexical, *regions)))
    if len(set(combined)) != len(combined) or len(combined) > MAX_TOTAL_CUES:
        raise RustcLexicalUnorderedCueAdapterError(
            "lexical and region cue roots are not disjoint or exceed total capacity"
        )
    observation = seal_unordered_cue_factor_observation_v1(
        selection_context_sha256=_rooted_observation_context(
            attempt,
            selection_context_sha256,
        ),
        cue_sha256s=combined,
    )
    payload = {
        "selection_context_sha256": selection_context_sha256,
        "attempt": attempt,
        "stderr": stderr,
        "lexical_tokens": tokens,
        "lexical_cue_sha256s": lexical,
        "region_cue_sha256s": regions,
        "observation": observation,
    }
    record = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "selection_context_sha256": selection_context_sha256,
        "attempt_sha256": attempt.sha256,
        "stderr_sha256": sha256(stderr).hexdigest(),
        "lexical_tokens": tokens,
        "lexical_cue_sha256s": lexical,
        "region_cue_sha256s": regions,
        "observation": observation.record(),
    }
    result = RustcLexicalUnorderedCueEnvelopeV1(
        **payload,
        sha256=_digest(_ENVELOPE_NAMESPACE, record),
    )
    result.authenticate()
    return result


__all__ = [
    "CLAIM_BOUNDARY",
    "MAX_LEXICAL_FEATURES",
    "MAX_LEXICAL_TOKEN_CHARACTERS",
    "MAX_PARENT_MESSAGE_UTF8_BYTES",
    "MAX_TOTAL_CUES",
    "SCHEMA",
    "RustcLexicalUnorderedCueAdapterError",
    "RustcLexicalUnorderedCueEnvelopeV1",
    "project_rustc_lexical_unordered_cue_observation_v1",
]
