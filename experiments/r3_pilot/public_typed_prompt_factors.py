"""Generic physical literal factors extracted from public prompt text.

Input preprocessing recognizes only boundary-delimited signed numeric lexemes.
It has no task-family vocabulary or structural edit knowledge.  A unique
in-range integer becomes an exact typed-output goal; every ambiguous state is
represented by the physical Int UNKNOWN cell.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from experiments.r3_pilot import public_typed_output_cells as output_cells
from humanbrain.typed_program import FactorAction


STATUS_SUPPORTED = "SUPPORTED"
STATUS_NO_LITERAL = "NO_LITERAL"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_OUT_OF_RANGE = "OUT_OF_RANGE"

_IDENTIFIER_CHARACTERS = "A-Za-z0-9_"
_NUMERIC_LEXEME = re.compile(
    rf"(?<![{_IDENTIFIER_CHARACTERS}])"
    r"(?P<value>[+-]?(?:(?:\d+\.\d+|\.\d+|\d+)(?:[eE][+-]?\d+)?))"
    rf"(?![{_IDENTIFIER_CHARACTERS}])"
)
_TOKENIZER_SHA256 = sha256(
    b"public-typed-prompt-integer-lexemes-v1\0"
    + _NUMERIC_LEXEME.pattern.encode("ascii")
).hexdigest()
_UNKNOWN_VALUE = object()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _integer_lexemes(prompt: str) -> tuple[str, ...]:
    integer_lexemes: list[str] = []
    for match in _NUMERIC_LEXEME.finditer(prompt):
        value = match.group("value")
        if not any(character in value for character in ".eE"):
            integer_lexemes.append(value)
    return tuple(integer_lexemes)


def _distinct_integer_values(
    lexemes: tuple[str, ...],
) -> tuple[tuple[int, ...], bool]:
    values: list[int] = []
    unrepresentable = False
    for lexeme in lexemes:
        try:
            value = int(lexeme)
        except ValueError:
            unrepresentable = True
            continue
        if value not in values:
            values.append(value)
    return tuple(values), unrepresentable


@dataclass(frozen=True, slots=True)
class PromptFactorObservation:
    prompt_sha256: str
    tokenizer_sha256: str
    integer_lexemes: tuple[str, ...]
    distinct_literals: tuple[int, ...]
    literal_encodings: tuple[output_cells.TypedOutputEncoding, ...]
    status: str
    literal: int | None
    literal_goal: output_cells.TypedOutputEncoding
    output_projection_sha256: str
    sha256: str

    @property
    def supported(self) -> bool:
        return self.status == STATUS_SUPPORTED


def observe_prompt_factors(prompt: str) -> PromptFactorObservation:
    """Extract one generic signed-integer goal from a public description."""

    if type(prompt) is not str or not prompt:
        raise ValueError("prompt-factor observation requires non-empty text")
    lexemes = _integer_lexemes(prompt)
    distinct_literals, unrepresentable = _distinct_integer_values(lexemes)
    in_range = tuple(
        value
        for value in distinct_literals
        if output_cells.INT_MIN <= value <= output_cells.INT_MAX
    )
    literal_encodings = tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.INT,
        )
        for value in in_range
    )
    out_of_range = unrepresentable or len(in_range) != len(distinct_literals)

    if out_of_range:
        status = STATUS_OUT_OF_RANGE
        literal = None
    elif not distinct_literals:
        status = STATUS_NO_LITERAL
        literal = None
    elif len(distinct_literals) == 1:
        status = STATUS_SUPPORTED
        literal = distinct_literals[0]
    else:
        status = STATUS_AMBIGUOUS
        literal = None

    literal_goal = (
        literal_encodings[0]
        if status == STATUS_SUPPORTED
        else output_cells.TypedOutputCellBank.encode(
            _UNKNOWN_VALUE,
            declared_type=output_cells.OutputType.INT,
        )
    )
    prompt_sha = sha256(prompt.encode("utf-8")).hexdigest()
    projection_sha = output_cells.TypedOutputCellBank.projection_sha256()
    record = {
        "prompt_sha256": prompt_sha,
        "tokenizer_sha256": _TOKENIZER_SHA256,
        "integer_lexemes": lexemes,
        "distinct_literals": distinct_literals,
        "literal_encoding_sha256": tuple(
            encoding.sha256 for encoding in literal_encodings
        ),
        "status": status,
        "literal": literal,
        "literal_goal_sha256": literal_goal.sha256,
        "output_projection_sha256": projection_sha,
    }
    return PromptFactorObservation(
        prompt_sha256=prompt_sha,
        tokenizer_sha256=_TOKENIZER_SHA256,
        integer_lexemes=lexemes,
        distinct_literals=distinct_literals,
        literal_encodings=literal_encodings,
        status=status,
        literal=literal,
        literal_goal=literal_goal,
        output_projection_sha256=projection_sha,
        sha256=sha256(_canonical_json(record)).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class PromptLiteralFactorRace:
    candidate_literal: int | None
    candidate_encoding_sha256: str
    spiked: bool
    spike_mask: np.ndarray
    current_pa: float
    matched_factor_cells: int
    active_goal_factors: int
    projection_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class FixedPromptLiteralProjection:
    goal_encoding_sha256: str
    detector: output_cells.FixedTypedCoincidenceDetector
    sha256: str

    @classmethod
    def build(
        cls,
        observation: PromptFactorObservation,
    ) -> FixedPromptLiteralProjection:
        if type(observation) is not PromptFactorObservation:
            raise TypeError("literal projection requires a PromptFactorObservation")
        detector = output_cells.FixedTypedCoincidenceDetector.for_target(
            observation.literal_goal
        )
        digest = sha256()
        digest.update(b"public-typed-prompt-literal-projection-v1\0")
        digest.update(observation.literal_goal.sha256.encode("ascii"))
        digest.update(detector.sha256.encode("ascii"))
        return cls(
            goal_encoding_sha256=observation.literal_goal.sha256,
            detector=detector,
            sha256=digest.hexdigest(),
        )

    def race(self, candidate: FactorAction) -> PromptLiteralFactorRace:
        """Run one candidate literal factor through the fixed physical detector."""

        if type(candidate) is not FactorAction:
            raise TypeError("literal-factor race requires a FactorAction")
        candidate_encoding = output_cells.TypedOutputCellBank.encode(
            candidate.literal if candidate.literal is not None else _UNKNOWN_VALUE,
            declared_type=output_cells.OutputType.INT,
        )
        coincidence = self.detector.detect(candidate_encoding)
        record = {
            "candidate_literal": candidate.literal,
            "candidate_encoding_sha256": candidate_encoding.sha256,
            "spiked": coincidence.spiked,
            "current_pa": coincidence.current_pa,
            "matched_factor_cells": coincidence.matched_factor_cells,
            "active_goal_factors": coincidence.active_target_factors,
            "projection_sha256": self.sha256,
        }
        return PromptLiteralFactorRace(
            candidate_literal=candidate.literal,
            candidate_encoding_sha256=candidate_encoding.sha256,
            spiked=coincidence.spiked,
            spike_mask=coincidence.spike_mask,
            current_pa=coincidence.current_pa,
            matched_factor_cells=coincidence.matched_factor_cells,
            active_goal_factors=coincidence.active_target_factors,
            projection_sha256=self.sha256,
            sha256=sha256(_canonical_json(record)).hexdigest(),
        )


__all__ = [
    "FixedPromptLiteralProjection",
    "PromptFactorObservation",
    "PromptLiteralFactorRace",
    "STATUS_AMBIGUOUS",
    "STATUS_NO_LITERAL",
    "STATUS_OUT_OF_RANGE",
    "STATUS_SUPPORTED",
    "observe_prompt_factors",
]
