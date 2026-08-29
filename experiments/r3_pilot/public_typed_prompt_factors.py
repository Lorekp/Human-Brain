"""Generic physical action factors extracted from public prompt text.

Input preprocessing recognizes only boundary-delimited signed numeric lexemes.
It has no task-family vocabulary or structural edit knowledge.  A unique
in-range integer becomes an exact typed-output goal; every ambiguous state is
represented by the physical Int UNKNOWN cell.

The operand channel is similarly narrow: declared parameter names/order plus
binary operator words become one operator/source/orientation pattern.  A fixed
synaptic projection fans that physical pattern cell into every matching
``FactorAction`` cell.  Site and schema are intentionally outside this layer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from experiments.r3_pilot import public_typed_output_cells as output_cells
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.typed_program import FactorAction
from humanbrain.typed_program_types import TypedSignature


STATUS_SUPPORTED = "SUPPORTED"
STATUS_NO_LITERAL = "NO_LITERAL"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_OUT_OF_RANGE = "OUT_OF_RANGE"

BINDING_STATUS_SUPPORTED = "SUPPORTED"
BINDING_STATUS_UNKNOWN = "UNKNOWN"
BINDING_STATUS_AMBIGUOUS = "AMBIGUOUS"

PROJECTION_STATUS_PROJECTED = "PROJECTED"
PROJECTION_STATUS_STOP_UNKNOWN = "STOP_UNKNOWN"
PROJECTION_STATUS_STOP_AMBIGUOUS = "STOP_AMBIGUOUS"
PROJECTION_STATUS_STOP_UNSUPPORTED = "STOP_UNSUPPORTED"

PHYSICAL_STOP = "STOP"

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

_OPERAND_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[+-]?\d+|<|-")
_INFIX_HYPHEN = re.compile(r"(?<=[A-Za-z0-9_])-(?=[A-Za-z0-9_])")
_BINARY_OPERATORS = frozenset({"subtract", "less-than"})
_PROMPT_SOURCES = frozenset({"SCOPE0", "SCOPE1", "LITERAL"})
_GENERIC_SINGLE_ARGUMENT = frozenset(
    {"argument", "datum", "input", "integer", "number", "value"}
)
_OPERAND_FILLERS = frozenset(
    {
        "a",
        "an",
        "and",
        "compute",
        "for",
        "given",
        "integer",
        "integers",
        "is",
        "of",
        "return",
        "result",
        "supplied",
        "than",
        "the",
        "whether",
    }
)
_PROMPT_CONTACT_CURRENT_PA = 5_000.0
_PROMPT_RACE_STEPS = 30
_UNRESOLVED_OPERAND = object()


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


@dataclass(frozen=True, slots=True)
class PromptOperandPattern:
    """One binary operator with de-Bruijn operand sources in prompt order."""

    operator: str
    left_source: str
    right_source: str
    literal: int | None

    def __post_init__(self) -> None:
        if self.operator not in _BINARY_OPERATORS:
            raise ValueError("prompt operand pattern has an unsupported operator")
        if (
            self.left_source not in _PROMPT_SOURCES
            or self.right_source not in _PROMPT_SOURCES
        ):
            raise ValueError("prompt operand pattern has an unsupported source")
        literal_count = int(self.left_source == "LITERAL") + int(
            self.right_source == "LITERAL"
        )
        if literal_count > 1:
            raise ValueError("one factor action cannot bind two prompt literals")
        if (literal_count == 1) != (self.literal is not None):
            raise ValueError("prompt literal must agree with its operand source")
        if self.literal is not None and type(self.literal) is not int:
            raise TypeError("prompt operand literal must be an integer")


@dataclass(frozen=True, slots=True)
class _PromptOperand:
    source: str
    literal: int | None = None


def _pattern_record(pattern: PromptOperandPattern) -> dict[str, object]:
    return {
        "operator": pattern.operator,
        "left_source": pattern.left_source,
        "right_source": pattern.right_source,
        "literal": pattern.literal,
    }


def _prompt_operand_tokens(prompt: str) -> tuple[str, ...]:
    normalized = re.sub(r"\bless\s*-\s*than\b", "less than", prompt.casefold())
    normalized = _INFIX_HYPHEN.sub(" - ", normalized)
    return tuple(match.group(0) for match in _OPERAND_TOKEN.finditer(normalized))


def _declared_sources(
    signature: TypedSignature,
) -> tuple[dict[str, str | None], dict[str, str | None], str | None]:
    declared: dict[str, str | None] = {}
    for index, parameter in enumerate(signature.parameters):
        name = parameter.name.casefold()
        if name in declared:
            return {}, {}, "parameter names collide after prompt case-folding"
        depth = len(signature.parameters) - index - 1
        declared[name] = f"SCOPE{depth}" if depth < 2 else None

    ordinal: dict[str, str | None] = {}
    for index, name in enumerate(("first", "second")):
        if index >= len(signature.parameters):
            break
        depth = len(signature.parameters) - index - 1
        ordinal[name] = f"SCOPE{depth}" if depth < 2 else None
    return declared, ordinal, None


def _operand_token(
    token: str,
    *,
    declared: dict[str, str | None],
    ordinal: dict[str, str | None],
    parameter_count: int,
) -> _PromptOperand | object | None:
    if token in declared:
        source = declared[token]
        return (
            _UNRESOLVED_OPERAND
            if source is None
            else _PromptOperand(source)
        )
    if token in ordinal:
        source = ordinal[token]
        return (
            _UNRESOLVED_OPERAND
            if source is None
            else _PromptOperand(source)
        )
    if re.fullmatch(r"[+-]?\d+", token):
        return _PromptOperand("LITERAL", int(token))
    if parameter_count == 1 and token in _GENERIC_SINGLE_ARGUMENT:
        return _PromptOperand("SCOPE0")
    if token in _OPERAND_FILLERS:
        return None
    return _UNRESOLVED_OPERAND


def _nearest_operand(
    tokens: tuple[str, ...],
    start: int,
    step: int,
    *,
    stop: int | None,
    declared: dict[str, str | None],
    ordinal: dict[str, str | None],
    parameter_count: int,
) -> _PromptOperand | object | None:
    cursor = start
    inspected = 0
    while 0 <= cursor < len(tokens) and inspected < 6:
        if stop is not None and cursor == stop:
            break
        operand = _operand_token(
            tokens[cursor],
            declared=declared,
            ordinal=ordinal,
            parameter_count=parameter_count,
        )
        if operand is not None:
            return operand
        cursor += step
        inspected += 1
    return None


def _operand_pattern(
    operator: str,
    left: _PromptOperand | object | None,
    right: _PromptOperand | object | None,
) -> PromptOperandPattern | None:
    if not isinstance(left, _PromptOperand) or not isinstance(right, _PromptOperand):
        return None
    literals = tuple(
        operand.literal
        for operand in (left, right)
        if operand.source == "LITERAL"
    )
    if len(literals) > 1:
        return None
    return PromptOperandPattern(
        operator=operator,
        left_source=left.source,
        right_source=right.source,
        literal=literals[0] if literals else None,
    )


@dataclass(frozen=True, slots=True)
class PromptBindingObservation:
    prompt_sha256: str
    signature_sha256: str
    tokens: tuple[str, ...]
    recognized_expression_count: int
    patterns: tuple[PromptOperandPattern, ...]
    status: str
    pattern: PromptOperandPattern | None
    reason: str
    sha256: str

    @property
    def supported(self) -> bool:
        return self.status == BINDING_STATUS_SUPPORTED


def observe_prompt_bindings(
    prompt: str,
    signature: TypedSignature,
) -> PromptBindingObservation:
    """Observe one generic binary operand relation without task outputs."""

    if type(prompt) is not str or not prompt:
        raise ValueError("prompt binding observation requires non-empty text")
    if not isinstance(signature, TypedSignature):
        raise TypeError("prompt binding observation requires a TypedSignature")

    tokens = _prompt_operand_tokens(prompt)
    declared, ordinal, declaration_error = _declared_sources(signature)
    patterns: list[PromptOperandPattern] = []
    recognized = 0
    failures = int(declaration_error is not None)
    parameter_count = len(signature.parameters)

    def record(
        operator: str,
        left: _PromptOperand | object | None,
        right: _PromptOperand | object | None,
    ) -> None:
        nonlocal recognized, failures
        recognized += 1
        pattern = _operand_pattern(operator, left, right)
        if pattern is None:
            failures += 1
        elif pattern not in patterns:
            patterns.append(pattern)

    if declaration_error is None:
        for index, token in enumerate(tokens):
            common = {
                "declared": declared,
                "ordinal": ordinal,
                "parameter_count": parameter_count,
            }
            if token in {"minus", "-"}:
                record(
                    "subtract",
                    _nearest_operand(tokens, index - 1, -1, stop=None, **common),
                    _nearest_operand(tokens, index + 1, 1, stop=None, **common),
                )
            elif (
                token == "less"
                and index + 1 < len(tokens)
                and tokens[index + 1] == "than"
            ):
                record(
                    "less-than",
                    _nearest_operand(tokens, index - 1, -1, stop=None, **common),
                    _nearest_operand(tokens, index + 2, 1, stop=None, **common),
                )
            elif token == "<":
                record(
                    "less-than",
                    _nearest_operand(tokens, index - 1, -1, stop=None, **common),
                    _nearest_operand(tokens, index + 1, 1, stop=None, **common),
                )
            elif token == "subtract":
                from_index = next(
                    (
                        candidate
                        for candidate in range(index + 1, min(index + 7, len(tokens)))
                        if tokens[candidate] == "from"
                    ),
                    None,
                )
                if from_index is not None:
                    subtrahend = _nearest_operand(
                        tokens,
                        index + 1,
                        1,
                        stop=from_index,
                        **common,
                    )
                    minuend = _nearest_operand(
                        tokens,
                        from_index + 1,
                        1,
                        stop=None,
                        **common,
                    )
                    record("subtract", minuend, subtrahend)
                else:
                    right = _nearest_operand(
                        tokens,
                        index + 1,
                        1,
                        stop=None,
                        **common,
                    )
                    left = (
                        _PromptOperand("SCOPE0")
                        if parameter_count == 1
                        else _UNRESOLVED_OPERAND
                    )
                    record("subtract", left, right)

    if len(patterns) == 1 and failures == 0:
        status = BINDING_STATUS_SUPPORTED
        pattern = patterns[0]
        reason = "one declared binary operand relation"
    elif len(patterns) > 1 or (patterns and failures):
        status = BINDING_STATUS_AMBIGUOUS
        pattern = None
        reason = "prompt contains conflicting or partially unresolved relations"
    else:
        status = BINDING_STATUS_UNKNOWN
        pattern = None
        reason = (
            declaration_error
            or "prompt has no complete supported operand relation"
        )

    signature_record = signature.to_record()
    signature_sha = sha256(_canonical_json(signature_record)).hexdigest()
    record_value = {
        "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
        "signature_sha256": signature_sha,
        "tokens": tokens,
        "recognized_expression_count": recognized,
        "patterns": tuple(_pattern_record(item) for item in patterns),
        "status": status,
        "pattern": None if pattern is None else _pattern_record(pattern),
        "reason": reason,
    }
    return PromptBindingObservation(
        prompt_sha256=record_value["prompt_sha256"],
        signature_sha256=signature_sha,
        tokens=tokens,
        recognized_expression_count=recognized,
        patterns=tuple(patterns),
        status=status,
        pattern=pattern,
        reason=reason,
        sha256=sha256(_canonical_json(record_value)).hexdigest(),
    )


def _action_operand_pattern(action: FactorAction) -> PromptOperandPattern | None:
    if (
        action.schema != "APPLY2"
        or action.operator not in _BINARY_OPERATORS
        or action.left_source not in _PROMPT_SOURCES
        or action.right_source not in _PROMPT_SOURCES
    ):
        return None
    literal_count = int(action.left_source == "LITERAL") + int(
        action.right_source == "LITERAL"
    )
    if literal_count > 1 or (literal_count == 1) != (action.literal is not None):
        return None
    return PromptOperandPattern(
        action.operator,
        action.left_source,
        action.right_source,
        action.literal,
    )


def _factor_action_record(action: FactorAction) -> dict[str, object]:
    return {
        "path": action.path,
        "schema": action.schema,
        "operator": action.operator,
        "left_source": action.left_source,
        "right_source": action.right_source,
        "literal": action.literal,
    }


def _freeze_projection(array: np.ndarray) -> np.ndarray:
    frozen = np.ascontiguousarray(array, dtype=np.float64)
    frozen.setflags(write=False)
    return frozen


def _binding_projection_sha256(
    candidates: tuple[FactorAction, ...],
    patterns: tuple[PromptOperandPattern, ...],
    race_weights_pa: np.ndarray,
    action_weights_pa: np.ndarray,
) -> str:
    digest = sha256()
    digest.update(b"public-typed-prompt-binding-projection-v1\0")
    digest.update(
        _canonical_json(tuple(_factor_action_record(item) for item in candidates))
    )
    digest.update(_canonical_json(tuple(_pattern_record(item) for item in patterns)))
    digest.update(race_weights_pa.tobytes())
    digest.update(action_weights_pa.tobytes())
    digest.update(str(_PROMPT_RACE_STEPS).encode("ascii"))
    return digest.hexdigest()


def _physical_first_spike(currents_pa: np.ndarray) -> tuple[int, int, int]:
    population = AdaptiveGlifPopulation(currents_pa.size)
    for step in range(_PROMPT_RACE_STEPS):
        fired = np.flatnonzero(population.step(currents_pa))
        if fired.size:
            if fired.size != 1:
                raise RuntimeError(
                    "prompt binding race produced simultaneous first spikes"
                )
            return int(fired[0]), step, int(fired.size)
    raise RuntimeError("prompt binding race produced no spike within its fixed bound")


@dataclass(frozen=True, slots=True)
class PromptBindingProjectionDecision:
    status: str
    observation: PromptBindingObservation
    physical_winner: PromptOperandPattern | str
    winner_cell: int
    stop_cell: int
    spike_step: int
    simultaneous_spike_candidates: int
    input_cells: tuple[float, ...]
    race_currents_pa: tuple[float, ...]
    action_currents_pa: tuple[float, ...]
    matching_action_cells: tuple[int, ...]
    projection_sha256: str
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str


@dataclass(frozen=True, slots=True)
class FixedPromptBindingProjection:
    """Readonly prompt-pattern race and pattern-to-action synapses."""

    candidate_bank: tuple[FactorAction, ...]
    patterns: tuple[PromptOperandPattern, ...]
    race_weights_pa: np.ndarray
    action_weights_pa: np.ndarray
    sha256: str

    @classmethod
    def build(
        cls,
        candidate_bank: tuple[FactorAction, ...] | list[FactorAction],
    ) -> FixedPromptBindingProjection:
        candidates = tuple(candidate_bank)
        if not candidates or not all(
            type(action) is FactorAction for action in candidates
        ):
            raise TypeError("prompt binding projection needs FactorAction candidates")
        if len(set(candidates)) != len(candidates):
            raise ValueError("prompt binding candidate bank must be unique")

        pattern_list: list[PromptOperandPattern] = []
        action_patterns: list[PromptOperandPattern | None] = []
        for action in candidates:
            pattern = _action_operand_pattern(action)
            action_patterns.append(pattern)
            if pattern is not None and pattern not in pattern_list:
                pattern_list.append(pattern)
        patterns = tuple(pattern_list)
        if not patterns:
            raise ValueError("candidate bank has no prompt-observable binary action")

        stop_cell = len(patterns)
        race_weights = np.zeros((stop_cell + 1, stop_cell + 1), dtype=np.float64)
        for pattern_cell in range(stop_cell):
            race_weights[pattern_cell, pattern_cell] = _PROMPT_CONTACT_CURRENT_PA
        race_weights[stop_cell, stop_cell] = _PROMPT_CONTACT_CURRENT_PA

        action_weights = np.zeros((len(candidates), len(patterns)), dtype=np.float64)
        for action_cell, pattern in enumerate(action_patterns):
            if pattern is not None:
                action_weights[action_cell, patterns.index(pattern)] = (
                    _PROMPT_CONTACT_CURRENT_PA
                )

        frozen_race = _freeze_projection(race_weights)
        frozen_actions = _freeze_projection(action_weights)
        digest = _binding_projection_sha256(
            candidates,
            patterns,
            frozen_race,
            frozen_actions,
        )
        return cls(
            candidate_bank=candidates,
            patterns=patterns,
            race_weights_pa=frozen_race,
            action_weights_pa=frozen_actions,
            sha256=digest,
        )

    @property
    def live_checkpoint_sha256(self) -> str:
        return _binding_projection_sha256(
            self.candidate_bank,
            self.patterns,
            self.race_weights_pa,
            self.action_weights_pa,
        )

    def project(
        self,
        prompt: str,
        signature: TypedSignature,
    ) -> PromptBindingProjectionDecision:
        """Race prompt evidence, then fan out only its winning factor cell."""

        before = self.live_checkpoint_sha256
        if before != self.sha256:
            raise RuntimeError("prompt binding projection changed before inference")
        observation = observe_prompt_bindings(prompt, signature)
        pattern_cell: int | None = None
        if observation.supported and observation.pattern in self.patterns:
            pattern_cell = self.patterns.index(observation.pattern)

        stop_cell = len(self.patterns)
        input_cells = np.zeros(stop_cell + 1, dtype=np.float64)
        input_cells[stop_cell if pattern_cell is None else pattern_cell] = 1.0
        race_currents = np.ascontiguousarray(self.race_weights_pa @ input_cells)
        winner_cell, spike_step, simultaneous = _physical_first_spike(race_currents)

        pattern_spikes = np.zeros(len(self.patterns), dtype=np.float64)
        if pattern_cell is None:
            if winner_cell != stop_cell:
                raise RuntimeError("physical prompt UNKNOWN did not reach STOP")
            physical_winner: PromptOperandPattern | str = PHYSICAL_STOP
            if observation.status == BINDING_STATUS_AMBIGUOUS:
                status = PROJECTION_STATUS_STOP_AMBIGUOUS
            elif observation.status == BINDING_STATUS_UNKNOWN:
                status = PROJECTION_STATUS_STOP_UNKNOWN
            else:
                status = PROJECTION_STATUS_STOP_UNSUPPORTED
        else:
            if winner_cell != pattern_cell:
                raise RuntimeError("prompt pattern race lost its exact physical cell")
            pattern_spikes[pattern_cell] = 1.0
            physical_winner = self.patterns[pattern_cell]
            status = PROJECTION_STATUS_PROJECTED

        action_currents = np.ascontiguousarray(
            self.action_weights_pa @ pattern_spikes
        )
        matching = tuple(int(cell) for cell in np.flatnonzero(action_currents))
        after = self.live_checkpoint_sha256
        if after != before or after != self.sha256:
            raise RuntimeError("prompt binding projection changed during inference")
        return PromptBindingProjectionDecision(
            status=status,
            observation=observation,
            physical_winner=physical_winner,
            winner_cell=winner_cell,
            stop_cell=stop_cell,
            spike_step=spike_step,
            simultaneous_spike_candidates=simultaneous,
            input_cells=tuple(float(value) for value in input_cells),
            race_currents_pa=tuple(float(value) for value in race_currents),
            action_currents_pa=tuple(float(value) for value in action_currents),
            matching_action_cells=matching,
            projection_sha256=self.sha256,
            checkpoint_before_sha256=before,
            checkpoint_after_sha256=after,
        )


__all__ = [
    "BINDING_STATUS_AMBIGUOUS",
    "BINDING_STATUS_SUPPORTED",
    "BINDING_STATUS_UNKNOWN",
    "FixedPromptBindingProjection",
    "FixedPromptLiteralProjection",
    "PHYSICAL_STOP",
    "PROJECTION_STATUS_PROJECTED",
    "PROJECTION_STATUS_STOP_AMBIGUOUS",
    "PROJECTION_STATUS_STOP_UNKNOWN",
    "PROJECTION_STATUS_STOP_UNSUPPORTED",
    "PromptBindingObservation",
    "PromptBindingProjectionDecision",
    "PromptFactorObservation",
    "PromptLiteralFactorRace",
    "PromptOperandPattern",
    "STATUS_AMBIGUOUS",
    "STATUS_NO_LITERAL",
    "STATUS_OUT_OF_RANGE",
    "STATUS_SUPPORTED",
    "observe_prompt_bindings",
    "observe_prompt_factors",
]
