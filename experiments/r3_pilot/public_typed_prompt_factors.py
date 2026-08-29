"""Generic physical action factors extracted from public prompt text.

Input preprocessing recognizes only boundary-delimited signed numeric lexemes.
It has no task-family vocabulary or structural edit knowledge.  A unique
in-range integer becomes an exact typed-output goal; every ambiguous state is
represented by the physical Int UNKNOWN cell.

The flat operand channel remains the H=1 interface: declared parameter
names/order plus binary operator words become one operator/source/orientation
pattern, then fixed synapses fan that physical cell into matching
``FactorAction`` cells.  The H=2 interface adds conservative compositions: a
root binary relation may wrap either operand in ``absolute``, or a typed List
prompt may bind a map element before applying a binary body relation.  Both
project only to ordered sequences whose compiler-observed selected sources
agree. Unsupported or conflicting structure reaches a physical STOP cell;
prompt evidence never ranks the broader candidate bank.
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

_OPERAND_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[+-]?\d+|<|-|\+")
_INFIX_HYPHEN = re.compile(r"(?<=[A-Za-z0-9_])-(?=[A-Za-z0-9_])")
_BINARY_OPERATORS = frozenset({"add", "subtract", "less-than"})
_UNARY_OPERATORS = frozenset({"absolute"})
_ABSOLUTE_TOKENS = frozenset({"abs", "absolute"})
_ABSOLUTE_FILLERS = frozenset({"of", "the", "value"})
_PROMPT_SOURCES = frozenset({"SCOPE0", "SCOPE1", "BOUND0", "LITERAL"})
_NUMBER_WORD_LITERALS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_MAP_QUANTIFIERS = frozenset({"each", "every"})
_GENERIC_SINGLE_ARGUMENT = frozenset(
    {"argument", "datum", "element", "input", "integer", "item", "number", "value"}
)
_OPERAND_FILLERS = frozenset(
    {
        "a",
        "an",
        "and",
        "compute",
        "each",
        "every",
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
    single_argument_source: str = "SCOPE0",
) -> _PromptOperand | object | None:
    if token in declared:
        source = (
            single_argument_source
            if parameter_count == 1 and single_argument_source == "BOUND0"
            else declared[token]
        )
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
    if token in _NUMBER_WORD_LITERALS:
        return _PromptOperand("LITERAL", _NUMBER_WORD_LITERALS[token])
    if parameter_count == 1 and token in _GENERIC_SINGLE_ARGUMENT:
        return _PromptOperand(single_argument_source)
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


@dataclass(frozen=True, slots=True)
class PromptComposedOperand:
    """One declared source with a generic outer-to-inner unary chain."""

    source: str
    literal: int | None
    unary_operators: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.source not in _PROMPT_SOURCES:
            raise ValueError("composed prompt operand has an unsupported source")
        if (self.source == "LITERAL") != (self.literal is not None):
            raise ValueError("composed prompt literal must agree with its source")
        if self.literal is not None and type(self.literal) is not int:
            raise TypeError("composed prompt literal must be an integer")
        if not all(operator in _UNARY_OPERATORS for operator in self.unary_operators):
            raise ValueError("composed prompt operand has an unsupported unary operator")


@dataclass(frozen=True, slots=True)
class PromptCompositionPattern:
    """One binary relation, optionally inside a map-bound body."""

    operator: str
    left: PromptComposedOperand
    right: PromptComposedOperand
    container_schema: str | None = None

    def __post_init__(self) -> None:
        if self.operator not in _BINARY_OPERATORS:
            raise ValueError("prompt composition has an unsupported binary operator")
        if not isinstance(self.left, PromptComposedOperand) or not isinstance(
            self.right,
            PromptComposedOperand,
        ):
            raise TypeError("prompt composition requires two composed operands")
        literal_count = int(self.left.source == "LITERAL") + int(
            self.right.source == "LITERAL"
        )
        if literal_count > 1:
            raise ValueError("one prompt composition cannot bind two literals")
        if self.container_schema not in {None, "MAP_BIND"}:
            raise ValueError("prompt composition has an unsupported container")


@dataclass(frozen=True, slots=True)
class PromptSequenceEvidence:
    """Two ordered actions plus selected-source identities at their sites."""

    first_action: FactorAction
    first_selected_source: str | None
    second_action: FactorAction
    second_selected_source: str | None

    def __post_init__(self) -> None:
        if type(self.first_action) is not FactorAction or type(
            self.second_action
        ) is not FactorAction:
            raise TypeError("prompt sequence evidence requires FactorAction values")
        valid_selected = {None, "SCOPE0", "SCOPE1", "BOUND0"}
        if (
            self.first_selected_source not in valid_selected
            or self.second_selected_source not in valid_selected
        ):
            raise ValueError(
                "selected prompt sources must be SCOPE0/SCOPE1/BOUND0 or absent"
            )


def _composed_operand_record(
    operand: PromptComposedOperand,
) -> dict[str, object]:
    return {
        "source": operand.source,
        "literal": operand.literal,
        "unary_operators": operand.unary_operators,
    }


def _composition_record(
    pattern: PromptCompositionPattern,
) -> dict[str, object]:
    return {
        "operator": pattern.operator,
        "left": _composed_operand_record(pattern.left),
        "right": _composed_operand_record(pattern.right),
        "container_schema": pattern.container_schema,
    }


def _sequence_evidence_record(
    evidence: PromptSequenceEvidence,
) -> dict[str, object]:
    return {
        "first_action": _factor_action_record(evidence.first_action),
        "first_selected_source": evidence.first_selected_source,
        "second_action": _factor_action_record(evidence.second_action),
        "second_selected_source": evidence.second_selected_source,
    }


def _absolute_prefix_before(
    tokens: tuple[str, ...],
    operand_index: int,
    *,
    stop: int | None,
) -> bool:
    cursor = operand_index - 1
    inspected = 0
    while cursor >= 0 and inspected < 6:
        if stop is not None and cursor == stop:
            break
        token = tokens[cursor]
        if token in _ABSOLUTE_FILLERS:
            cursor -= 1
            inspected += 1
            continue
        return token in _ABSOLUTE_TOKENS
    return False


def _nearest_composed_operand(
    tokens: tuple[str, ...],
    start: int,
    step: int,
    *,
    stop: int | None,
    declared: dict[str, str | None],
    ordinal: dict[str, str | None],
    parameter_count: int,
    single_argument_source: str,
) -> PromptComposedOperand | object | None:
    cursor = start
    inspected = 0
    unary: list[str] = []
    while 0 <= cursor < len(tokens) and inspected < 10:
        if stop is not None and cursor == stop:
            break
        token = tokens[cursor]
        if token in _ABSOLUTE_TOKENS:
            unary.append("absolute")
            cursor += step
            inspected += 1
            continue
        if unary and token in _ABSOLUTE_FILLERS:
            cursor += step
            inspected += 1
            continue
        operand = _operand_token(
            token,
            declared=declared,
            ordinal=ordinal,
            parameter_count=parameter_count,
            single_argument_source=single_argument_source,
        )
        if isinstance(operand, _PromptOperand):
            if step < 0 and not unary and _absolute_prefix_before(
                tokens,
                cursor,
                stop=stop,
            ):
                unary.append("absolute")
            return PromptComposedOperand(
                source=operand.source,
                literal=operand.literal,
                unary_operators=tuple(unary),
            )
        if operand is _UNRESOLVED_OPERAND:
            return _UNRESOLVED_OPERAND
        cursor += step
        inspected += 1
    return None


def _composition_pattern(
    operator: str,
    left: PromptComposedOperand | object | None,
    right: PromptComposedOperand | object | None,
    *,
    container_schema: str | None,
) -> PromptCompositionPattern | None:
    if not isinstance(left, PromptComposedOperand) or not isinstance(
        right,
        PromptComposedOperand,
    ):
        return None
    literal_count = int(left.source == "LITERAL") + int(
        right.source == "LITERAL"
    )
    if literal_count > 1:
        return None
    return PromptCompositionPattern(
        operator=operator,
        left=left,
        right=right,
        container_schema=container_schema,
    )


def _prompt_container_schema(
    tokens: tuple[str, ...],
    signature: TypedSignature,
) -> str | None:
    if not _MAP_QUANTIFIERS.intersection(tokens) or len(signature.parameters) != 1:
        return None
    parameter_type = signature.parameters[0].type
    if parameter_type.constructor == "List" and signature.result == parameter_type:
        return "MAP_BIND"
    return None


@dataclass(frozen=True, slots=True)
class PromptCompositionObservation:
    prompt_sha256: str
    signature_sha256: str
    tokens: tuple[str, ...]
    recognized_expression_count: int
    patterns: tuple[PromptCompositionPattern, ...]
    status: str
    pattern: PromptCompositionPattern | None
    reason: str
    sha256: str

    @property
    def supported(self) -> bool:
        return self.status == BINDING_STATUS_SUPPORTED


def observe_prompt_compositions(
    prompt: str,
    signature: TypedSignature,
) -> PromptCompositionObservation:
    """Observe one root binary relation and its nested unary operand structure."""

    if type(prompt) is not str or not prompt:
        raise ValueError("prompt composition observation requires non-empty text")
    if not isinstance(signature, TypedSignature):
        raise TypeError("prompt composition observation requires a TypedSignature")

    tokens = _prompt_operand_tokens(prompt)
    declared, ordinal, declaration_error = _declared_sources(signature)
    patterns: list[PromptCompositionPattern] = []
    recognized = 0
    failures = int(declaration_error is not None)
    parameter_count = len(signature.parameters)
    container_schema = _prompt_container_schema(tokens, signature)

    def record(
        operator: str,
        left: PromptComposedOperand | object | None,
        right: PromptComposedOperand | object | None,
    ) -> None:
        nonlocal recognized, failures
        recognized += 1
        pattern = _composition_pattern(
            operator,
            left,
            right,
            container_schema=container_schema,
        )
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
                "single_argument_source": (
                    "BOUND0" if container_schema == "MAP_BIND" else "SCOPE0"
                ),
            }
            if token in {"plus", "+"}:
                record(
                    "add",
                    _nearest_composed_operand(
                        tokens,
                        index - 1,
                        -1,
                        stop=None,
                        **common,
                    ),
                    _nearest_composed_operand(
                        tokens,
                        index + 1,
                        1,
                        stop=None,
                        **common,
                    ),
                )
            elif token in {"minus", "-"}:
                record(
                    "subtract",
                    _nearest_composed_operand(
                        tokens,
                        index - 1,
                        -1,
                        stop=None,
                        **common,
                    ),
                    _nearest_composed_operand(
                        tokens,
                        index + 1,
                        1,
                        stop=None,
                        **common,
                    ),
                )
            elif (
                token == "less"
                and index + 1 < len(tokens)
                and tokens[index + 1] == "than"
            ):
                record(
                    "less-than",
                    _nearest_composed_operand(
                        tokens,
                        index - 1,
                        -1,
                        stop=None,
                        **common,
                    ),
                    _nearest_composed_operand(
                        tokens,
                        index + 2,
                        1,
                        stop=None,
                        **common,
                    ),
                )
            elif token == "<":
                record(
                    "less-than",
                    _nearest_composed_operand(
                        tokens,
                        index - 1,
                        -1,
                        stop=None,
                        **common,
                    ),
                    _nearest_composed_operand(
                        tokens,
                        index + 1,
                        1,
                        stop=None,
                        **common,
                    ),
                )
            elif token in {"add", "added"}:
                to_index = next(
                    (
                        candidate
                        for candidate in range(index + 1, min(index + 10, len(tokens)))
                        if tokens[candidate] == "to"
                    ),
                    None,
                )
                if to_index is not None:
                    addend = _nearest_composed_operand(
                        tokens,
                        index + 1 if token == "add" else index - 1,
                        1 if token == "add" else -1,
                        stop=to_index if token == "add" else None,
                        **common,
                    )
                    base = _nearest_composed_operand(
                        tokens,
                        to_index + 1,
                        1,
                        stop=None,
                        **common,
                    )
                    record("add", base, addend)
                else:
                    right = _nearest_composed_operand(
                        tokens,
                        index + 1,
                        1,
                        stop=None,
                        **common,
                    )
                    left: PromptComposedOperand | object = PromptComposedOperand(
                        "BOUND0" if container_schema == "MAP_BIND" else "SCOPE0",
                        None,
                        (),
                    )
                    record("add", left, right)
            elif token in {"subtract", "subtracted"}:
                from_index = next(
                    (
                        candidate
                        for candidate in range(index + 1, min(index + 10, len(tokens)))
                        if tokens[candidate] == "from"
                    ),
                    None,
                )
                if from_index is not None:
                    subtrahend = _nearest_composed_operand(
                        tokens,
                        index + 1 if token == "subtract" else index - 1,
                        1 if token == "subtract" else -1,
                        stop=from_index if token == "subtract" else None,
                        **common,
                    )
                    minuend = _nearest_composed_operand(
                        tokens,
                        from_index + 1,
                        1,
                        stop=None,
                        **common,
                    )
                    record("subtract", minuend, subtrahend)
                else:
                    right = _nearest_composed_operand(
                        tokens,
                        index + 1,
                        1,
                        stop=None,
                        **common,
                    )
                    left: PromptComposedOperand | object = (
                        PromptComposedOperand(
                            (
                                "BOUND0"
                                if container_schema == "MAP_BIND"
                                else "SCOPE0"
                            ),
                            None,
                            (),
                        )
                        if parameter_count == 1
                        else _UNRESOLVED_OPERAND
                    )
                    record("subtract", left, right)

    if len(patterns) == 1 and failures == 0:
        status = BINDING_STATUS_SUPPORTED
        pattern = patterns[0]
        reason = "one declared compositional operand relation"
    elif len(patterns) > 1 or (patterns and failures):
        status = BINDING_STATUS_AMBIGUOUS
        pattern = None
        reason = "prompt contains conflicting or partially unresolved relations"
    else:
        status = BINDING_STATUS_UNKNOWN
        pattern = None
        reason = (
            declaration_error
            or "prompt has no complete supported compositional operand relation"
        )

    signature_sha = sha256(_canonical_json(signature.to_record())).hexdigest()
    record_value = {
        "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
        "signature_sha256": signature_sha,
        "tokens": tokens,
        "recognized_expression_count": recognized,
        "patterns": tuple(_composition_record(item) for item in patterns),
        "status": status,
        "pattern": None if pattern is None else _composition_record(pattern),
        "reason": reason,
    }
    return PromptCompositionObservation(
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


def _evidence_operand(
    action: FactorAction,
    source: str,
    selected_source: str | None,
) -> PromptComposedOperand | None:
    normalized = selected_source if source == "SELECTED" else source
    if normalized not in _PROMPT_SOURCES:
        return None
    literal = action.literal if normalized == "LITERAL" else None
    if (normalized == "LITERAL") != (literal is not None):
        return None
    return PromptComposedOperand(normalized, literal, ())


def _sequence_composition_pattern(
    evidence: PromptSequenceEvidence,
) -> PromptCompositionPattern | None:
    first = evidence.first_action
    second = evidence.second_action
    if (
        first.path == ()
        and first.schema == "MAP_BIND"
        and first.operator == "map"
        and first.left_source == "BOUND0"
        and second.path == (0, 0)
        and second.schema == "APPLY2"
        and second.operator in _BINARY_OPERATORS
    ):
        collection = _evidence_operand(
            first,
            first.right_source,
            evidence.first_selected_source,
        )
        left = _evidence_operand(
            second,
            second.left_source,
            evidence.second_selected_source,
        )
        right = _evidence_operand(
            second,
            second.right_source,
            evidence.second_selected_source,
        )
        if collection is None or collection.source not in {"SCOPE0", "SCOPE1"}:
            return None
        if left is None or right is None:
            return None
        return PromptCompositionPattern(
            second.operator,
            left,
            right,
            container_schema="MAP_BIND",
        )
    if (
        first.path == (0, 0)
        and evidence.first_selected_source == "BOUND0"
        and first.schema == "APPLY2"
        and first.operator in _BINARY_OPERATORS
        and second.schema == "DONE"
    ):
        left = _evidence_operand(
            first,
            first.left_source,
            evidence.first_selected_source,
        )
        right = _evidence_operand(
            first,
            first.right_source,
            evidence.first_selected_source,
        )
        if left is None or right is None:
            return None
        return PromptCompositionPattern(
            first.operator,
            left,
            right,
            container_schema="MAP_BIND",
        )
    if (
        first.path != ()
        or first.schema != "APPLY2"
        or first.operator not in _BINARY_OPERATORS
    ):
        return None
    left = _evidence_operand(
        first,
        first.left_source,
        evidence.first_selected_source,
    )
    right = _evidence_operand(
        first,
        first.right_source,
        evidence.first_selected_source,
    )
    if left is None or right is None:
        return None
    if second.schema == "DONE":
        return PromptCompositionPattern(first.operator, left, right)
    if (
        second.schema != "APPLY1"
        or second.operator != "absolute"
        or second.right_source != "NONE"
        or second.path not in {(0,), (1,)}
    ):
        return None
    child_index = second.path[0]
    nested = _evidence_operand(
        second,
        second.left_source,
        evidence.second_selected_source,
    )
    operands = [left, right]
    if nested is None or nested != operands[child_index]:
        return None
    operands[child_index] = PromptComposedOperand(
        nested.source,
        nested.literal,
        ("absolute", *nested.unary_operators),
    )
    return PromptCompositionPattern(first.operator, operands[0], operands[1])


def _sequence_projection_sha256(
    candidates: tuple[PromptSequenceEvidence, ...],
    patterns: tuple[PromptCompositionPattern, ...],
    race_weights_pa: np.ndarray,
    sequence_weights_pa: np.ndarray,
) -> str:
    digest = sha256()
    digest.update(b"public-typed-prompt-sequence-projection-v2\0")
    digest.update(
        _canonical_json(tuple(_sequence_evidence_record(item) for item in candidates))
    )
    digest.update(
        _canonical_json(tuple(_composition_record(item) for item in patterns))
    )
    digest.update(race_weights_pa.tobytes())
    digest.update(sequence_weights_pa.tobytes())
    digest.update(str(_PROMPT_RACE_STEPS).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PromptSequenceProjectionDecision:
    status: str
    observation: PromptCompositionObservation
    physical_winner: PromptCompositionPattern | str
    winner_cell: int
    stop_cell: int
    spike_step: int
    simultaneous_spike_candidates: int
    input_cells: tuple[float, ...]
    race_currents_pa: tuple[float, ...]
    sequence_currents_pa: tuple[float, ...]
    matching_sequence_cells: tuple[int, ...]
    projection_sha256: str
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str


@dataclass(frozen=True, slots=True)
class FixedPromptSequenceProjection:
    """Readonly prompt-composition race and exact pattern-to-sequence synapses."""

    candidate_bank: tuple[PromptSequenceEvidence, ...]
    patterns: tuple[PromptCompositionPattern, ...]
    race_weights_pa: np.ndarray
    sequence_weights_pa: np.ndarray
    sha256: str

    @classmethod
    def build(
        cls,
        candidate_bank: tuple[PromptSequenceEvidence, ...]
        | list[PromptSequenceEvidence],
    ) -> FixedPromptSequenceProjection:
        candidates = tuple(candidate_bank)
        if not candidates or not all(
            type(item) is PromptSequenceEvidence for item in candidates
        ):
            raise TypeError(
                "prompt sequence projection needs PromptSequenceEvidence candidates"
            )
        if len(set(candidates)) != len(candidates):
            raise ValueError("prompt sequence candidate bank must be unique")

        pattern_list: list[PromptCompositionPattern] = []
        candidate_patterns: list[PromptCompositionPattern | None] = []
        for evidence in candidates:
            pattern = _sequence_composition_pattern(evidence)
            candidate_patterns.append(pattern)
            if pattern is not None and pattern not in pattern_list:
                pattern_list.append(pattern)
        patterns = tuple(pattern_list)
        stop_cell = len(patterns)
        race_weights = np.zeros((stop_cell + 1, stop_cell + 1), dtype=np.float64)
        for pattern_cell in range(stop_cell):
            race_weights[pattern_cell, pattern_cell] = _PROMPT_CONTACT_CURRENT_PA
        race_weights[stop_cell, stop_cell] = _PROMPT_CONTACT_CURRENT_PA

        sequence_weights = np.zeros(
            (len(candidates), len(patterns)),
            dtype=np.float64,
        )
        for sequence_cell, pattern in enumerate(candidate_patterns):
            if pattern is not None:
                sequence_weights[sequence_cell, patterns.index(pattern)] = (
                    _PROMPT_CONTACT_CURRENT_PA
                )
        frozen_race = _freeze_projection(race_weights)
        frozen_sequences = _freeze_projection(sequence_weights)
        digest = _sequence_projection_sha256(
            candidates,
            patterns,
            frozen_race,
            frozen_sequences,
        )
        return cls(
            candidate_bank=candidates,
            patterns=patterns,
            race_weights_pa=frozen_race,
            sequence_weights_pa=frozen_sequences,
            sha256=digest,
        )

    @property
    def live_checkpoint_sha256(self) -> str:
        return _sequence_projection_sha256(
            self.candidate_bank,
            self.patterns,
            self.race_weights_pa,
            self.sequence_weights_pa,
        )

    def project(
        self,
        prompt: str,
        signature: TypedSignature,
    ) -> PromptSequenceProjectionDecision:
        """Race one prompt composition, then project its exact sequence cells."""

        before = self.live_checkpoint_sha256
        if before != self.sha256:
            raise RuntimeError("prompt sequence projection changed before inference")
        observation = observe_prompt_compositions(prompt, signature)
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
                raise RuntimeError("physical prompt composition UNKNOWN did not stop")
            physical_winner: PromptCompositionPattern | str = PHYSICAL_STOP
            if observation.status == BINDING_STATUS_AMBIGUOUS:
                status = PROJECTION_STATUS_STOP_AMBIGUOUS
            elif observation.status == BINDING_STATUS_UNKNOWN:
                status = PROJECTION_STATUS_STOP_UNKNOWN
            else:
                status = PROJECTION_STATUS_STOP_UNSUPPORTED
        else:
            if winner_cell != pattern_cell:
                raise RuntimeError("prompt composition race lost its exact cell")
            pattern_spikes[pattern_cell] = 1.0
            physical_winner = self.patterns[pattern_cell]
            status = PROJECTION_STATUS_PROJECTED

        sequence_currents = np.ascontiguousarray(
            self.sequence_weights_pa @ pattern_spikes
        )
        matching = tuple(int(cell) for cell in np.flatnonzero(sequence_currents))
        after = self.live_checkpoint_sha256
        if after != before or after != self.sha256:
            raise RuntimeError("prompt sequence projection changed during inference")
        return PromptSequenceProjectionDecision(
            status=status,
            observation=observation,
            physical_winner=physical_winner,
            winner_cell=winner_cell,
            stop_cell=stop_cell,
            spike_step=spike_step,
            simultaneous_spike_candidates=simultaneous,
            input_cells=tuple(float(value) for value in input_cells),
            race_currents_pa=tuple(float(value) for value in race_currents),
            sequence_currents_pa=tuple(float(value) for value in sequence_currents),
            matching_sequence_cells=matching,
            projection_sha256=self.sha256,
            checkpoint_before_sha256=before,
            checkpoint_after_sha256=after,
        )


__all__ = [
    "BINDING_STATUS_AMBIGUOUS",
    "BINDING_STATUS_SUPPORTED",
    "BINDING_STATUS_UNKNOWN",
    "PHYSICAL_STOP",
    "PROJECTION_STATUS_PROJECTED",
    "PROJECTION_STATUS_STOP_AMBIGUOUS",
    "PROJECTION_STATUS_STOP_UNKNOWN",
    "PROJECTION_STATUS_STOP_UNSUPPORTED",
    "STATUS_AMBIGUOUS",
    "STATUS_NO_LITERAL",
    "STATUS_OUT_OF_RANGE",
    "STATUS_SUPPORTED",
    "FixedPromptBindingProjection",
    "FixedPromptLiteralProjection",
    "FixedPromptSequenceProjection",
    "PromptBindingObservation",
    "PromptBindingProjectionDecision",
    "PromptComposedOperand",
    "PromptCompositionObservation",
    "PromptCompositionPattern",
    "PromptFactorObservation",
    "PromptLiteralFactorRace",
    "PromptOperandPattern",
    "PromptSequenceEvidence",
    "PromptSequenceProjectionDecision",
    "observe_prompt_bindings",
    "observe_prompt_compositions",
    "observe_prompt_factors",
]
