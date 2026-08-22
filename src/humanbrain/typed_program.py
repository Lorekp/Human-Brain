"""Layer-1 typed program values and the deep grow/complete/compile seam.

Layer 0 remains the explicitly named scalar control in :mod:`humanbrain.program`.
This module does not alias or fall back to that control.
"""

from __future__ import annotations

import ast
import base64
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256

import numpy as np

from humanbrain.cells import ALLEN_GLIF5_CONFIG_SHA256, AllenGlif5Parameters
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig, RecurrentRace
from humanbrain.recurrent_state import RecurrentCellSynapseState
from humanbrain.typed_program_types import (
    ProgramParameter,
    ProgramType,
    TypedSignature,
    _record_sequence,
    _strict_record,
)


class ProgramFormatError(ValueError):
    """A token stream is not one complete, canonical, well-typed program."""


class PersistentStateChangedError(RuntimeError):
    """Completion changed state that a frozen checkpoint owns."""


@dataclass(frozen=True, slots=True)
class Example:
    """One public growth example, with no task/family/ordinal identity."""

    prompt: str
    signature: TypedSignature
    teacher_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("program growth prompt must be a non-empty string")
        if not isinstance(self.signature, TypedSignature):
            raise TypeError("program growth signature must be a TypedSignature")
        if not isinstance(self.teacher_tokens, tuple) or not self.teacher_tokens:
            raise ValueError("program growth teacher tokens must be a non-empty tuple")
        if not all(isinstance(token, str) and token for token in self.teacher_tokens):
            raise TypeError("program growth teacher tokens must be non-empty strings")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _program_sha256(tokens: tuple[str, ...], signature: TypedSignature) -> str:
    return sha256(
        _canonical_json({"signature": signature.to_record(), "tokens": list(tokens)})
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Program:
    """An immutable typed program; grammar and AST details remain private."""

    tokens: tuple[str, ...]
    signature: TypedSignature
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, tuple) or not self.tokens:
            raise ValueError("program tokens must be a non-empty tuple")
        if not all(isinstance(token, str) and token for token in self.tokens):
            raise TypeError("program tokens must be non-empty strings")
        if not isinstance(self.signature, TypedSignature):
            raise TypeError("program signature must be a TypedSignature")
        object.__setattr__(self, "sha256", _program_sha256(self.tokens, self.signature))


@dataclass(frozen=True, slots=True)
class CompiledPython:
    """Structurally emitted, unannotated Python source."""

    source: str
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("compiled Python source must be non-empty")
        object.__setattr__(self, "sha256", sha256(self.source.encode("utf-8")).hexdigest())


@dataclass(frozen=True, slots=True)
class ProgramRaceAudit:
    accepted_token: str
    winner_cell_id: int
    spike_time_ms: float
    simulated_steps: int
    prefix_sha256: str
    allowed_cell_ids: tuple[int, ...]
    excluded_cell_ids: tuple[int, ...]
    output_current_sha256: str


@dataclass(frozen=True, slots=True)
class Completion:
    program: Program
    races: tuple[ProgramRaceAudit, ...]
    prompt_sha256: str
    model_input_sha256: str
    checkpoint_sha256: str
    checkpoint_cell_synapse_before: str
    checkpoint_cell_synapse_after: str

    @property
    def race_count(self) -> int:
        return len(self.races)


@dataclass(frozen=True, slots=True)
class _FrozenArray:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes


@dataclass(frozen=True, slots=True)
class _FrozenCircuitState:
    metadata_json: bytes
    arrays: tuple[_FrozenArray, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class _FrozenCheckpointPayload:
    circuit: _FrozenCircuitState
    training_digest: str
    training_input_sha256: tuple[str, ...]
    protocol_json: bytes


@dataclass(frozen=True, slots=True, init=False)
class FrozenCheckpoint:
    """Immutable Layer-1 state restored into a new circuit for each completion."""

    sha256: str
    training_digest: str
    training_input_sha256: tuple[str, ...]
    cell_synapse_sha256: str
    protocol_sha256: str
    _bytes: bytes = field(repr=False, compare=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("FrozenCheckpoint must be constructed with from_bytes")

    def to_bytes(self) -> bytes:
        self._validated_payload()
        return self._bytes

    def _validated_payload(self) -> _FrozenCheckpointPayload:
        if type(self._bytes) is not bytes:
            raise ValueError("frozen checkpoint payload is not raw bytes")
        payload = _decode_checkpoint_bytes(self._bytes)
        if _checkpoint_bytes(payload) != self._bytes:
            raise ValueError("frozen checkpoint is not canonically encoded")
        expected = (
            sha256(self._bytes).hexdigest(),
            payload.training_digest,
            payload.training_input_sha256,
            payload.circuit.sha256,
            sha256(payload.protocol_json).hexdigest(),
        )
        actual = (
            self.sha256,
            self.training_digest,
            self.training_input_sha256,
            self.cell_synapse_sha256,
            self.protocol_sha256,
        )
        if actual != expected:
            raise ValueError("frozen checkpoint derived fields do not match its bytes")
        return payload

    @classmethod
    def from_bytes(cls, value: bytes) -> FrozenCheckpoint:
        if type(value) is not bytes:
            raise TypeError("frozen checkpoint payload must be bytes")
        payload = _decode_checkpoint_bytes(value)
        if _checkpoint_bytes(payload) != value:
            raise ValueError("frozen checkpoint is not canonically encoded")
        checkpoint = object.__new__(cls)
        object.__setattr__(checkpoint, "sha256", sha256(value).hexdigest())
        object.__setattr__(checkpoint, "training_digest", payload.training_digest)
        object.__setattr__(
            checkpoint,
            "training_input_sha256",
            payload.training_input_sha256,
        )
        object.__setattr__(checkpoint, "cell_synapse_sha256", payload.circuit.sha256)
        object.__setattr__(
            checkpoint,
            "protocol_sha256",
            sha256(payload.protocol_json).hexdigest(),
        )
        object.__setattr__(checkpoint, "_bytes", value)
        return checkpoint


@dataclass(frozen=True, slots=True)
class _TypeVariable:
    identity: str | int


@dataclass(frozen=True, slots=True)
class _ConstructedType:
    constructor: str
    arguments: tuple[_TypeTerm, ...] = ()


_TypeTerm = _TypeVariable | _ConstructedType


def _type_term(value: ProgramType) -> _ConstructedType:
    return _ConstructedType(
        value.constructor,
        tuple(_type_term(argument) for argument in value.arguments),
    )


def _list_type(element: _TypeTerm) -> _ConstructedType:
    return _ConstructedType("List", (element,))


def _function_type(parameters: tuple[_TypeTerm, ...], result: _TypeTerm) -> _ConstructedType:
    return _ConstructedType("Function", (*parameters, result))


@dataclass(frozen=True, slots=True)
class _PrimitiveSpec:
    token: str
    role: str
    result: _TypeTerm | None = None
    arguments: tuple[_TypeTerm, ...] = ()
    lowering: str = ""
    value: int | str | None = None


_INT = _type_term(ProgramType.Int())
_A = _TypeVariable("a")
_B = _TypeVariable("b")
_LITERAL_DIGITS = 4

# This is the sole language inventory. Its order fixes output-cell identity;
# its type/role fields drive typed-hole masking and parsing; and its lowering
# field drives structural Python emission.
_PRIMITIVE_SPECS = (
    _PrimitiveSpec("prompt-signature-delimiter", "input-delimiter"),
    _PrimitiveSpec("input-delimiter", "input-delimiter"),
    _PrimitiveSpec("add", "operation", _INT, (_INT, _INT), "add"),
    _PrimitiveSpec(
        "map",
        "operation",
        _list_type(_B),
        (_function_type((_A,), _B), _list_type(_A)),
        "map",
    ),
    _PrimitiveSpec("bind", "binder", lowering="binder"),
    _PrimitiveSpec("variable-0", "variable", lowering="variable", value=0),
    _PrimitiveSpec("integer", "literal", _INT, lowering="integer"),
    _PrimitiveSpec("+", "sign", value="+"),
    _PrimitiveSpec("-", "sign", value="-"),
    *tuple(_PrimitiveSpec(str(value), "digit", value=value) for value in range(10)),
    _PrimitiveSpec("end", "literal-end"),
)
_SPEC_BY_TOKEN = {spec.token: spec for spec in _PRIMITIVE_SPECS}
if len(_SPEC_BY_TOKEN) != len(_PRIMITIVE_SPECS):
    raise RuntimeError("primitive tokens must be unique")
_TOKEN_TO_ID = {spec.token: 256 + index for index, spec in enumerate(_PRIMITIVE_SPECS)}
_ID_TO_SPEC = {_TOKEN_TO_ID[spec.token]: spec for spec in _PRIMITIVE_SPECS}
_PROMPT_SIGNATURE_DELIMITER_ID = _TOKEN_TO_ID["prompt-signature-delimiter"]
_INPUT_DELIMITER_ID = _TOKEN_TO_ID["input-delimiter"]
_TOKEN_CELL_COUNT = 256 + len(_PRIMITIVE_SPECS)

_CIRCUIT_CONFIG = RecurrentCircuitConfig(
    reservoir_cells=128,
    input_fanout=16,
    recurrent_fanout=8,
    random_seed=17,
)
_GROWTH_EPOCHS = 3
_MAX_TRAINING_EXAMPLES = 256
_MAX_CHECKPOINT_BYTES = 2_097_152
_MAX_PROGRAM_TOKENS = 128
_MAX_AST_NODES = 64
_MAX_AST_DEPTH = 16
_PROTOCOL_VERSION = "humanbrain-layer1-typed-program-v2"
_COMPILER_ID = "humanbrain-structural-python-ast-v1"
_ALLEN_GLIF5_PARAMETERS = AllenGlif5Parameters()

_FROZEN_CIRCUIT_METADATA_KEYS = frozenset(
    {
        "token_cells",
        "allen_glif5_config_sha256",
        "cell_parameters",
        "circuit_config",
        "growth_examples",
        "local_updates",
        "runtime_active_reservoir_cells",
        "developmental_state",
    }
)
_EXPECTED_CELL_PARAMETERS = json.loads(
    _canonical_json(asdict(_ALLEN_GLIF5_PARAMETERS)).decode("utf-8")
)
_EXPECTED_CIRCUIT_CONFIG = json.loads(_canonical_json(asdict(_CIRCUIT_CONFIG)).decode("utf-8"))


def _array_layout(
    dtype: object,
    shape: tuple[int, ...],
) -> tuple[str, tuple[int, ...], int]:
    trusted_dtype = np.dtype(dtype)
    item_count = 1
    for size in shape:
        item_count *= size
    return trusted_dtype.str, shape, item_count * trusted_dtype.itemsize


_EXPECTED_FROZEN_ARRAY_LAYOUTS = {
    "afferent_targets": _array_layout(
        np.int32,
        (_TOKEN_CELL_COUNT, _CIRCUIT_CONFIG.input_fanout),
    ),
    "recurrent_targets": _array_layout(
        np.int32,
        (_CIRCUIT_CONFIG.reservoir_cells, _CIRCUIT_CONFIG.recurrent_fanout),
    ),
    "recurrent_weights_pa": _array_layout(
        np.float64,
        (_CIRCUIT_CONFIG.reservoir_cells, _CIRCUIT_CONFIG.recurrent_fanout),
    ),
    "output_heterogeneity_pa": _array_layout(
        np.float64,
        (_TOKEN_CELL_COUNT,),
    ),
    "pacemaker_synapses_pa": _array_layout(
        np.float64,
        (_TOKEN_CELL_COUNT,),
    ),
    "readout_synapses_pa": _array_layout(
        np.float64,
        (_TOKEN_CELL_COUNT, _CIRCUIT_CONFIG.reservoir_cells),
    ),
    "normal_matrix": _array_layout(
        np.float64,
        (
            _CIRCUIT_CONFIG.reservoir_cells + 1,
            _CIRCUIT_CONFIG.reservoir_cells + 1,
        ),
    ),
    "target_moments": _array_layout(
        np.float64,
        (_TOKEN_CELL_COUNT, _CIRCUIT_CONFIG.reservoir_cells + 1),
    ),
}
_EXPECTED_FROZEN_ARRAY_NAMES = tuple(sorted(_EXPECTED_FROZEN_ARRAY_LAYOUTS))


def _type_term_record(term: _TypeTerm) -> object:
    if isinstance(term, _TypeVariable):
        return {"variable": term.identity}
    return {
        "constructor": term.constructor,
        "arguments": [_type_term_record(argument) for argument in term.arguments],
    }


def _primitive_table_record() -> list[dict[str, object]]:
    return [
        {
            "token": spec.token,
            "role": spec.role,
            "result": None if spec.result is None else _type_term_record(spec.result),
            "arguments": [_type_term_record(argument) for argument in spec.arguments],
            "lowering": spec.lowering,
            "value": spec.value,
        }
        for spec in _PRIMITIVE_SPECS
    ]


_PRIMITIVE_TABLE_SHA256 = sha256(_canonical_json(_primitive_table_record())).hexdigest()
_CIRCUIT_CONFIG_SHA256 = sha256(_canonical_json(asdict(_CIRCUIT_CONFIG))).hexdigest()


@dataclass(frozen=True, slots=True)
class Layer1Protocol:
    """Fixed, auditable commitments for the Layer-1 language and fitted circuit."""

    protocol_version: str = _PROTOCOL_VERSION
    compiler_id: str = _COMPILER_ID
    primitive_table_sha256: str = _PRIMITIVE_TABLE_SHA256
    circuit_config_sha256: str = _CIRCUIT_CONFIG_SHA256
    growth_epochs: int = _GROWTH_EPOCHS
    allen_glif5_config_sha256: str = ALLEN_GLIF5_CONFIG_SHA256
    allen_glif5_parameters: AllenGlif5Parameters = _ALLEN_GLIF5_PARAMETERS
    max_training_examples: int = _MAX_TRAINING_EXAMPLES
    max_checkpoint_bytes: int = _MAX_CHECKPOINT_BYTES
    literal_digit_bound: int = _LITERAL_DIGITS
    max_ast_nodes: int = _MAX_AST_NODES
    max_ast_depth: int = _MAX_AST_DEPTH
    max_output_tokens: int = _MAX_PROGRAM_TOKENS

    def __post_init__(self) -> None:
        expected = (
            _PROTOCOL_VERSION,
            _COMPILER_ID,
            _PRIMITIVE_TABLE_SHA256,
            _CIRCUIT_CONFIG_SHA256,
            _GROWTH_EPOCHS,
            ALLEN_GLIF5_CONFIG_SHA256,
            _ALLEN_GLIF5_PARAMETERS,
            _MAX_TRAINING_EXAMPLES,
            _MAX_CHECKPOINT_BYTES,
            _LITERAL_DIGITS,
            _MAX_AST_NODES,
            _MAX_AST_DEPTH,
            _MAX_PROGRAM_TOKENS,
        )
        actual = (
            self.protocol_version,
            self.compiler_id,
            self.primitive_table_sha256,
            self.circuit_config_sha256,
            self.growth_epochs,
            self.allen_glif5_config_sha256,
            self.allen_glif5_parameters,
            self.max_training_examples,
            self.max_checkpoint_bytes,
            self.literal_digit_bound,
            self.max_ast_nodes,
            self.max_ast_depth,
            self.max_output_tokens,
        )
        if actual != expected:
            raise ValueError("Layer1Protocol commitments are fixed for this implementation")


def _protocol_json(protocol: Layer1Protocol) -> bytes:
    return _canonical_json(
        {
            "protocol_version": protocol.protocol_version,
            "compiler_id": protocol.compiler_id,
            "primitive_table": _primitive_table_record(),
            "primitive_table_sha256": protocol.primitive_table_sha256,
            "circuit_config": asdict(_CIRCUIT_CONFIG),
            "circuit_config_sha256": protocol.circuit_config_sha256,
            "growth_epochs": protocol.growth_epochs,
            "allen_glif5_config_sha256": protocol.allen_glif5_config_sha256,
            "allen_glif5_parameters": asdict(protocol.allen_glif5_parameters),
            "max_training_examples": protocol.max_training_examples,
            "max_checkpoint_bytes": protocol.max_checkpoint_bytes,
            "literal_digit_bound": protocol.literal_digit_bound,
            "max_ast_nodes": protocol.max_ast_nodes,
            "max_ast_depth": protocol.max_ast_depth,
            "max_output_tokens": protocol.max_output_tokens,
        }
    )


def _instantiate_term(
    term: _TypeTerm,
    variables: dict[str, _TypeVariable],
    next_variable: int,
) -> tuple[_TypeTerm, int]:
    if isinstance(term, _TypeVariable):
        if not isinstance(term.identity, str):
            return term, next_variable
        variable = variables.get(term.identity)
        if variable is None:
            variable = _TypeVariable(next_variable)
            variables[term.identity] = variable
            next_variable += 1
        return variable, next_variable
    arguments: list[_TypeTerm] = []
    for argument in term.arguments:
        instantiated, next_variable = _instantiate_term(argument, variables, next_variable)
        arguments.append(instantiated)
    return _ConstructedType(term.constructor, tuple(arguments)), next_variable


def _instantiate_spec(
    spec: _PrimitiveSpec, next_variable: int
) -> tuple[_TypeTerm, tuple[_TypeTerm, ...], int]:
    assert spec.result is not None
    variables: dict[str, _TypeVariable] = {}
    result, next_variable = _instantiate_term(spec.result, variables, next_variable)
    arguments: list[_TypeTerm] = []
    for argument in spec.arguments:
        instantiated, next_variable = _instantiate_term(argument, variables, next_variable)
        arguments.append(instantiated)
    return result, tuple(arguments), next_variable


def _resolve(term: _TypeTerm, substitutions: Mapping[int, _TypeTerm]) -> _TypeTerm:
    if isinstance(term, _TypeVariable):
        if isinstance(term.identity, str):
            return term
        replacement = substitutions.get(term.identity)
        return term if replacement is None else _resolve(replacement, substitutions)
    return _ConstructedType(
        term.constructor,
        tuple(_resolve(argument, substitutions) for argument in term.arguments),
    )


def _occurs(
    variable: _TypeVariable,
    term: _TypeTerm,
    substitutions: Mapping[int, _TypeTerm],
) -> bool:
    resolved = _resolve(term, substitutions)
    if isinstance(resolved, _TypeVariable):
        return resolved == variable
    return any(_occurs(variable, argument, substitutions) for argument in resolved.arguments)


def _unify(
    left: _TypeTerm,
    right: _TypeTerm,
    substitutions: dict[int, _TypeTerm],
) -> bool:
    left = _resolve(left, substitutions)
    right = _resolve(right, substitutions)
    if left == right:
        return True
    if isinstance(left, _TypeVariable) and isinstance(left.identity, int):
        if _occurs(left, right, substitutions):
            return False
        substitutions[left.identity] = right
        return True
    if isinstance(right, _TypeVariable) and isinstance(right.identity, int):
        if _occurs(right, left, substitutions):
            return False
        substitutions[right.identity] = left
        return True
    if not isinstance(left, _ConstructedType) or not isinstance(right, _ConstructedType):
        return False
    if left.constructor != right.constructor or len(left.arguments) != len(right.arguments):
        return False
    return all(
        _unify(left_argument, right_argument, substitutions)
        for left_argument, right_argument in zip(left.arguments, right.arguments, strict=True)
    )


def _function_parts(
    term: _TypeTerm,
) -> tuple[tuple[_TypeTerm, ...], _TypeTerm] | None:
    if (
        not isinstance(term, _ConstructedType)
        or term.constructor != "Function"
        or len(term.arguments) < 1
    ):
        return None
    return term.arguments[:-1], term.arguments[-1]


@dataclass(frozen=True, slots=True)
class _Hole:
    expected: _TypeTerm
    environment: tuple[_TypeTerm, ...]


@dataclass(slots=True)
class _LiteralState:
    sign: str | None = None
    digits: list[int] = field(default_factory=list)


class _TypedHoleState:
    def __init__(self, signature: TypedSignature) -> None:
        environment = tuple(
            _type_term(parameter.type) for parameter in reversed(signature.parameters)
        )
        self._holes = [_Hole(_type_term(signature.result), environment)]
        self._literal: _LiteralState | None = None
        self._substitutions: dict[int, _TypeTerm] = {}
        self._next_variable = 0

    @property
    def complete(self) -> bool:
        return not self._holes and self._literal is None

    def allowed(self) -> tuple[_PrimitiveSpec, ...]:
        if self.complete:
            return ()
        if self._literal is not None:
            literal = self._literal
            if literal.sign is None:
                return tuple(spec for spec in _PRIMITIVE_SPECS if spec.role == "sign")
            if not literal.digits:
                return tuple(
                    spec
                    for spec in _PRIMITIVE_SPECS
                    if spec.role == "digit" and not (literal.sign == "-" and spec.value == 0)
                )
            allowed_roles = {"literal-end"}
            if len(literal.digits) < _LITERAL_DIGITS and literal.digits[0] != 0:
                allowed_roles.add("digit")
            return tuple(spec for spec in _PRIMITIVE_SPECS if spec.role in allowed_roles)

        hole = self._holes[-1]
        allowed: list[_PrimitiveSpec] = []
        for spec in _PRIMITIVE_SPECS:
            trial = dict(self._substitutions)
            if spec.role in {"operation", "literal"}:
                assert spec.result is not None
                result, _, _ = _instantiate_spec(spec, self._next_variable)
                if _unify(result, hole.expected, trial):
                    allowed.append(spec)
            elif spec.role == "variable":
                index = int(spec.value)
                if index < len(hole.environment) and _unify(
                    hole.environment[index], hole.expected, trial
                ):
                    allowed.append(spec)
            elif (
                spec.role == "binder"
                and _function_parts(_resolve(hole.expected, trial)) is not None
            ):
                allowed.append(spec)
        return tuple(allowed)

    def consume(self, spec: _PrimitiveSpec) -> None:
        if spec not in self.allowed():
            raise ProgramFormatError(f"token {spec.token!r} is invalid at this typed hole")
        if self._literal is not None:
            if spec.role == "sign":
                self._literal.sign = str(spec.value)
            elif spec.role == "digit":
                self._literal.digits.append(int(spec.value))
            elif spec.role == "literal-end":
                self._literal = None
            return

        hole = self._holes.pop()
        if spec.role == "operation":
            assert spec.result is not None
            result, arguments, next_variable = _instantiate_spec(spec, self._next_variable)
            if not _unify(result, hole.expected, self._substitutions):
                raise ProgramFormatError("primitive does not inhabit the typed hole")
            self._next_variable = next_variable
            for argument in reversed(arguments):
                self._holes.append(_Hole(argument, hole.environment))
        elif spec.role == "variable":
            index = int(spec.value)
            if not _unify(hole.environment[index], hole.expected, self._substitutions):
                raise ProgramFormatError("variable does not inhabit the typed hole")
        elif spec.role == "literal":
            assert spec.result is not None
            result, _, next_variable = _instantiate_spec(spec, self._next_variable)
            if not _unify(result, hole.expected, self._substitutions):
                raise ProgramFormatError("literal does not inhabit the typed hole")
            self._next_variable = next_variable
            self._literal = _LiteralState()
        elif spec.role == "binder":
            parts = _function_parts(_resolve(hole.expected, self._substitutions))
            if parts is None:
                raise ProgramFormatError("binder requires an internal function hole")
            parameters, result = parts
            environment = (*reversed(parameters), *hole.environment)
            self._holes.append(_Hole(result, environment))


@dataclass(frozen=True, slots=True)
class _Expression:
    spec: _PrimitiveSpec
    children: tuple[_Expression, ...] = ()
    variable_index: int | None = None
    integer_value: int | None = None


def _parse_expression(
    tokens: tuple[str, ...],
    index: int,
) -> tuple[_Expression, int]:
    if index >= len(tokens):
        raise ProgramFormatError("program expression is incomplete")
    try:
        spec = _SPEC_BY_TOKEN[tokens[index]]
    except KeyError as error:
        raise ProgramFormatError(f"unknown program token {tokens[index]!r}") from error

    if spec.role == "variable":
        return _Expression(spec, variable_index=int(spec.value)), index + 1
    if spec.role == "literal":
        cursor = index + 1
        if cursor >= len(tokens) or _SPEC_BY_TOKEN.get(tokens[cursor]) is None:
            raise ProgramFormatError("integer literal is missing its sign")
        sign_spec = _SPEC_BY_TOKEN[tokens[cursor]]
        if sign_spec.role != "sign":
            raise ProgramFormatError("integer literal is missing its sign")
        cursor += 1
        digits: list[int] = []
        while cursor < len(tokens):
            part = _SPEC_BY_TOKEN.get(tokens[cursor])
            if part is None:
                raise ProgramFormatError("unknown integer literal token")
            if part.role == "literal-end":
                break
            if part.role != "digit":
                raise ProgramFormatError("invalid integer literal token")
            digits.append(int(part.value))
            cursor += 1
        if cursor >= len(tokens) or not digits:
            raise ProgramFormatError("integer literal is incomplete")
        magnitude = int("".join(str(digit) for digit in digits))
        integer_value = -magnitude if sign_spec.value == "-" else magnitude
        return _Expression(spec, integer_value=integer_value), cursor + 1
    if spec.role == "binder":
        body, cursor = _parse_expression(tokens, index + 1)
        return _Expression(spec, children=(body,)), cursor
    if spec.role != "operation":
        raise ProgramFormatError("structural token cannot fill an expression hole")
    cursor = index + 1
    children: list[_Expression] = []
    for _ in spec.arguments:
        child, cursor = _parse_expression(tokens, cursor)
        children.append(child)
    return _Expression(spec, children=tuple(children)), cursor


def _expression_metrics(expression: _Expression) -> tuple[int, int]:
    if not expression.children:
        return 1, 1
    child_metrics = tuple(_expression_metrics(child) for child in expression.children)
    return (
        1 + sum(nodes for nodes, _ in child_metrics),
        1 + max(depth for _, depth in child_metrics),
    )


def _validate_program(tokens: tuple[str, ...], signature: TypedSignature) -> _Expression:
    if not tokens or len(tokens) > _MAX_PROGRAM_TOKENS:
        raise ProgramFormatError("program token count is outside the Layer-1 bound")
    state = _TypedHoleState(signature)
    for token in tokens:
        try:
            spec = _SPEC_BY_TOKEN[token]
        except KeyError as error:
            raise ProgramFormatError(f"unknown program token {token!r}") from error
        state.consume(spec)
    if not state.complete:
        raise ProgramFormatError("program is incomplete")
    expression, cursor = _parse_expression(tokens, 0)
    if cursor != len(tokens):
        raise ProgramFormatError("program has trailing tokens")
    nodes, depth = _expression_metrics(expression)
    if nodes > _MAX_AST_NODES:
        raise ProgramFormatError("program exceeds the Layer-1 AST node bound")
    if depth > _MAX_AST_DEPTH:
        raise ProgramFormatError("program exceeds the Layer-1 AST depth bound")
    return expression


def _program_token_ids(tokens: Sequence[str]) -> tuple[int, ...]:
    try:
        return tuple(_TOKEN_TO_ID[token] for token in tokens)
    except KeyError as error:
        raise ProgramFormatError(f"unknown program token {error.args[0]!r}") from error


def _signature_bytes(signature: TypedSignature) -> bytes:
    return _canonical_json(signature.to_record())


def _model_input(prompt: str, signature: TypedSignature) -> tuple[int, ...]:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("completion prompt must be a non-empty string")
    return (
        *prompt.encode("utf-8"),
        _PROMPT_SIGNATURE_DELIMITER_ID,
        *_signature_bytes(signature),
        _INPUT_DELIMITER_ID,
    )


def _token_stream_bytes(token_ids: Sequence[int]) -> bytes:
    return b"".join(int(token_id).to_bytes(2, "big") for token_id in token_ids)


def _freeze_circuit(circuit: RecurrentCellCircuit) -> _FrozenCircuitState:
    state = circuit.capture_cell_synapse_state()
    arrays = tuple(
        _FrozenArray(
            name=name,
            dtype=np.ascontiguousarray(state.arrays[name]).dtype.str,
            shape=tuple(int(size) for size in state.arrays[name].shape),
            data=np.ascontiguousarray(state.arrays[name]).tobytes(),
        )
        for name in sorted(state.arrays)
    )
    return _FrozenCircuitState(
        metadata_json=_canonical_json(state.metadata),
        arrays=arrays,
        sha256=state.sha256,
    )


def _fixed_layout_circuit_sha256(frozen: _FrozenCircuitState) -> str:
    digest = sha256(frozen.metadata_json)
    for item in frozen.arrays:
        digest.update(item.name.encode("ascii"))
        digest.update(item.dtype.encode("ascii"))
        digest.update(json.dumps(item.shape).encode("ascii"))
        digest.update(item.data)
    return digest.hexdigest()


def _restore_circuit(frozen: _FrozenCircuitState) -> RecurrentCellCircuit:
    metadata = json.loads(frozen.metadata_json.decode("utf-8"))
    arrays = {
        item.name: np.frombuffer(item.data, dtype=np.dtype(item.dtype)).reshape(item.shape)
        for item in frozen.arrays
    }
    state = RecurrentCellSynapseState(metadata=metadata, arrays=arrays)
    return RecurrentCellCircuit.restore_cell_synapse_state(state)


def _checkpoint_bytes(payload: _FrozenCheckpointPayload) -> bytes:
    return _canonical_json(
        {
            "format": _PROTOCOL_VERSION,
            "protocol": json.loads(payload.protocol_json.decode("utf-8")),
            "training_digest": payload.training_digest,
            "training_input_sha256": list(payload.training_input_sha256),
            "circuit": {
                "metadata": json.loads(payload.circuit.metadata_json.decode("utf-8")),
                "arrays": [
                    {
                        "name": item.name,
                        "dtype": item.dtype,
                        "shape": list(item.shape),
                        "data": base64.b64encode(item.data).decode("ascii"),
                    }
                    for item in payload.circuit.arrays
                ],
                "sha256": payload.circuit.sha256,
            },
        }
    )


def _digest_text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_frozen_circuit_metadata(value: object) -> Mapping[str, object]:
    metadata = _strict_record(
        value,
        _FROZEN_CIRCUIT_METADATA_KEYS,
        "frozen circuit metadata",
    )
    fixed_values_match = (
        type(metadata["token_cells"]) is int
        and metadata["token_cells"] == _TOKEN_CELL_COUNT
        and metadata["allen_glif5_config_sha256"] == ALLEN_GLIF5_CONFIG_SHA256
        and _canonical_json(metadata["cell_parameters"])
        == _canonical_json(_EXPECTED_CELL_PARAMETERS)
        and _canonical_json(metadata["circuit_config"])
        == _canonical_json(_EXPECTED_CIRCUIT_CONFIG)
        and type(metadata["growth_examples"]) is int
        and metadata["growth_examples"] > 0
        and type(metadata["local_updates"]) is int
        and metadata["local_updates"] == 0
        and type(metadata["runtime_active_reservoir_cells"]) is int
        and metadata["runtime_active_reservoir_cells"] == _CIRCUIT_CONFIG.reservoir_cells
        and metadata["developmental_state"] is False
    )
    if not fixed_values_match:
        raise ValueError("frozen circuit metadata does not match the protocol")
    return metadata


def _validate_frozen_array_headers(
    value: object,
) -> tuple[Mapping[str, object], ...]:
    records = tuple(
        _strict_record(
            item,
            frozenset({"name", "dtype", "shape", "data"}),
            "frozen array record",
        )
        for item in _record_sequence(value, "frozen arrays")
    )
    names = tuple(record["name"] for record in records)
    if names != _EXPECTED_FROZEN_ARRAY_NAMES:
        raise ValueError("frozen array names do not match the protocol")

    for record, name in zip(records, names, strict=True):
        expected_dtype, expected_shape, expected_byte_length = _EXPECTED_FROZEN_ARRAY_LAYOUTS[name]
        if record["dtype"] != expected_dtype:
            raise ValueError(f"frozen array {name} dtype does not match the protocol")
        shape = _record_sequence(record["shape"], "frozen array shape")
        if not all(type(size) is int for size in shape) or tuple(shape) != expected_shape:
            raise ValueError(f"frozen array {name} shape does not match the protocol")
        data = record["data"]
        expected_base64_length = 4 * ((expected_byte_length + 2) // 3)
        if not isinstance(data, str):
            raise ValueError("frozen array data must be base64 text")
        if len(data) != expected_base64_length:
            raise ValueError(f"frozen array {name} byte length does not match the protocol")
    return records


def _decode_checkpoint_bytes(value: bytes) -> _FrozenCheckpointPayload:
    if len(value) > _MAX_CHECKPOINT_BYTES:
        raise ValueError(
            f"frozen checkpoint exceeds the {_MAX_CHECKPOINT_BYTES}-byte protocol limit"
        )
    try:
        decoded = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("frozen checkpoint is not canonical JSON") from error
    record = _strict_record(
        decoded,
        frozenset(
            {
                "format",
                "protocol",
                "training_digest",
                "training_input_sha256",
                "circuit",
            }
        ),
        "frozen checkpoint record",
    )
    if record["format"] != _PROTOCOL_VERSION:
        raise ValueError("unsupported frozen checkpoint format")
    protocol_json = _canonical_json(record["protocol"])
    if protocol_json != _protocol_json(Layer1Protocol()):
        raise ValueError("frozen checkpoint protocol commitments do not match")
    training_digest = _digest_text(record["training_digest"], "training digest")
    input_hash_values = _record_sequence(
        record["training_input_sha256"],
        "training input digests",
    )
    input_hash_count = len(input_hash_values)
    if input_hash_count == 0:
        raise ValueError("frozen checkpoint needs at least one training input digest")
    if input_hash_count > _MAX_TRAINING_EXAMPLES:
        raise ValueError(
            f"frozen checkpoint permits at most {_MAX_TRAINING_EXAMPLES} "
            "training input digests"
        )
    input_hashes = tuple(
        _digest_text(item, "training input digest")
        for item in input_hash_values
    )
    circuit_record = _strict_record(
        record["circuit"],
        frozenset({"metadata", "arrays", "sha256"}),
        "frozen circuit record",
    )
    metadata = _validate_frozen_circuit_metadata(circuit_record["metadata"])
    array_records = _validate_frozen_array_headers(circuit_record["arrays"])
    arrays: list[_FrozenArray] = []
    for array_record in array_records:
        name = array_record["name"]
        dtype = array_record["dtype"]
        data = array_record["data"]
        shape_values = tuple(array_record["shape"])
        try:
            raw = base64.b64decode(data.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as error:
            raise ValueError("frozen array data is not canonical base64") from error
        expected_byte_length = _EXPECTED_FROZEN_ARRAY_LAYOUTS[name][2]
        if len(raw) != expected_byte_length:
            raise ValueError(f"frozen array {name} byte length does not match the protocol")
        arrays.append(
            _FrozenArray(
                name=name,
                dtype=dtype,
                shape=tuple(shape_values),
                data=raw,
            )
        )
    circuit = _FrozenCircuitState(
        metadata_json=_canonical_json(metadata),
        arrays=tuple(arrays),
        sha256=_digest_text(circuit_record["sha256"], "cell/synapse digest"),
    )
    # Fixed layout lets us validate raw array bytes without a NumPy view or
    # recurrent-circuit construction. Completion performs the sole restore.
    if _fixed_layout_circuit_sha256(circuit) != circuit.sha256:
        raise ValueError("frozen checkpoint cell/synapse state digest mismatch")
    return _FrozenCheckpointPayload(
        circuit=circuit,
        training_digest=training_digest,
        training_input_sha256=input_hashes,
        protocol_json=protocol_json,
    )


def grow(protocol: Layer1Protocol, examples: Sequence[Example]) -> FrozenCheckpoint:
    """Fit one Layer-1 recurrent cell circuit and freeze its persistent state."""
    if type(protocol) is not Layer1Protocol:
        raise TypeError("grow requires the fixed Layer1Protocol")
    if isinstance(examples, (str, bytes, bytearray)) or not isinstance(examples, Sequence):
        raise TypeError("Layer-1 growth examples must be an ordered sequence")
    example_count = len(examples)
    if example_count > protocol.max_training_examples:
        raise ValueError(
            f"Layer-1 growth accepts at most {protocol.max_training_examples} examples"
        )
    corpus = tuple(examples)
    if not corpus:
        raise ValueError("at least one public Layer-1 growth example is required")
    if len(corpus) != example_count:
        raise ValueError("Layer-1 growth examples changed while being materialized")
    if not all(isinstance(example, Example) for example in corpus):
        raise TypeError("Layer-1 growth examples must be Example values")

    circuit = RecurrentCellCircuit(token_cells=_TOKEN_CELL_COUNT, config=_CIRCUIT_CONFIG)
    digest = sha256()
    input_hashes: list[str] = []
    for example in corpus:
        _validate_program(example.teacher_tokens, example.signature)
        input_ids = _model_input(example.prompt, example.signature)
        input_bytes = _token_stream_bytes(input_ids)
        input_hashes.append(sha256(input_bytes).hexdigest())
        teacher_ids = _program_token_ids(example.teacher_tokens)
        circuit.grow_sequence((*input_ids, *teacher_ids), epochs=protocol.growth_epochs)
        digest.update(input_bytes)
        digest.update(_token_stream_bytes(teacher_ids))
    circuit.freeze_learning()
    payload = _FrozenCheckpointPayload(
        circuit=_freeze_circuit(circuit),
        training_digest=digest.hexdigest(),
        training_input_sha256=tuple(input_hashes),
        protocol_json=_protocol_json(protocol),
    )
    return FrozenCheckpoint.from_bytes(_checkpoint_bytes(payload))


def complete(
    checkpoint: FrozenCheckpoint,
    prompt: str,
    typed_signature: TypedSignature,
) -> Completion:
    """Restore, observe exactly prompt+signature+delimiter, and emit one program."""
    if type(checkpoint) is not FrozenCheckpoint:
        raise TypeError("complete requires an exact FrozenCheckpoint")
    if not isinstance(typed_signature, TypedSignature):
        raise TypeError("complete requires a TypedSignature")
    payload = checkpoint._validated_payload()
    checkpoint_bytes = checkpoint._bytes
    checkpoint_identity = (
        checkpoint.sha256,
        checkpoint.training_digest,
        checkpoint.training_input_sha256,
        checkpoint.cell_synapse_sha256,
        checkpoint.protocol_sha256,
        checkpoint_bytes,
    )
    circuit = _restore_circuit(payload.circuit)
    before = circuit.cell_synapse_state_sha256
    if before != checkpoint.cell_synapse_sha256:
        raise ValueError("restored checkpoint cell/synapse digest mismatch")
    input_ids = _model_input(prompt, typed_signature)
    circuit.reset_sequence()
    for token_id in input_ids:
        circuit.observe(token_id)

    state = _TypedHoleState(typed_signature)
    tokens: list[str] = []
    audits: list[ProgramRaceAudit] = []
    while not state.complete and len(tokens) < _MAX_PROGRAM_TOKENS:
        allowed_specs = state.allowed()
        if not allowed_specs:
            raise ProgramFormatError("typed hole has no inhabiting primitive")
        allowed_cell_ids = tuple(_TOKEN_TO_ID[spec.token] for spec in allowed_specs)
        prefix_sha256 = sha256(_canonical_json(tokens)).hexdigest()
        race: RecurrentRace = circuit.race(allowed_token_ids=allowed_cell_ids)
        accepted = _ID_TO_SPEC[race.winner]
        state.consume(accepted)
        tokens.append(accepted.token)
        audits.append(
            ProgramRaceAudit(
                accepted_token=accepted.token,
                winner_cell_id=race.winner,
                spike_time_ms=race.spike_time_ms,
                simulated_steps=race.simulated_steps,
                prefix_sha256=prefix_sha256,
                allowed_cell_ids=allowed_cell_ids,
                excluded_cell_ids=race.excluded_token_ids,
                output_current_sha256=race.output_current_sha256,
            )
        )
        circuit.observe(race.winner)
    if not state.complete:
        raise RuntimeError("cell program did not terminate within the Layer-1 bound")

    emitted = tuple(tokens)
    _validate_program(emitted, typed_signature)
    after = circuit.cell_synapse_state_sha256
    if after != before:
        raise PersistentStateChangedError("completion changed frozen cell/synapse state")
    if (
        checkpoint.sha256,
        checkpoint.training_digest,
        checkpoint.training_input_sha256,
        checkpoint.cell_synapse_sha256,
        checkpoint.protocol_sha256,
        checkpoint._bytes,
    ) != checkpoint_identity:
        raise PersistentStateChangedError("completion changed supplied checkpoint bytes")
    return Completion(
        program=Program(emitted, typed_signature),
        races=tuple(audits),
        prompt_sha256=sha256(prompt.encode("utf-8")).hexdigest(),
        model_input_sha256=sha256(_token_stream_bytes(input_ids)).hexdigest(),
        checkpoint_sha256=checkpoint.sha256,
        checkpoint_cell_synapse_before=before,
        checkpoint_cell_synapse_after=after,
    )


@dataclass(slots=True)
class _LoweringContext:
    next_binder: int = 0

    def name(self, stem: str) -> str:
        name = f"{stem}_{self.next_binder}"
        self.next_binder += 1
        return name


def _lower_expression(
    expression: _Expression,
    environment: tuple[str, ...],
    context: _LoweringContext,
) -> ast.expr:
    lowering = expression.spec.lowering
    if lowering == "variable":
        assert expression.variable_index is not None
        if expression.variable_index >= len(environment):
            raise ProgramFormatError("de-Bruijn variable is outside its compiler scope")
        return ast.Name(id=environment[expression.variable_index], ctx=ast.Load())
    if lowering == "integer":
        assert expression.integer_value is not None
        return ast.Constant(value=expression.integer_value)
    if lowering == "add":
        return ast.BinOp(
            left=_lower_expression(expression.children[0], environment, context),
            op=ast.Add(),
            right=_lower_expression(expression.children[1], environment, context),
        )
    if lowering == "map":
        binder, collection = expression.children
        if binder.spec.role != "binder" or len(binder.children) != 1:
            raise ProgramFormatError("map requires one explicit binder")
        name = context.name("item")
        return ast.ListComp(
            elt=_lower_expression(binder.children[0], (name, *environment), context),
            generators=[
                ast.comprehension(
                    target=ast.Name(id=name, ctx=ast.Store()),
                    iter=_lower_expression(collection, environment, context),
                    ifs=[],
                    is_async=0,
                )
            ],
        )
    raise ProgramFormatError("primitive has no structural Python lowering")


def compile(program: Program) -> CompiledPython:
    """Compile only a validated Program/signature into safe structural Python."""
    if not isinstance(program, Program):
        raise TypeError("compile requires a Program")
    if _program_sha256(program.tokens, program.signature) != program.sha256:
        raise ValueError("program digest mismatch")
    expression = _validate_program(program.tokens, program.signature)
    environment = tuple(parameter.name for parameter in reversed(program.signature.parameters))
    module = ast.Module(
        body=[
            ast.FunctionDef(
                name=program.signature.entry_point,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[
                        ast.arg(arg=parameter.name) for parameter in program.signature.parameters
                    ],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=[
                    ast.Return(
                        value=_lower_expression(
                            expression,
                            environment,
                            _LoweringContext(),
                        )
                    )
                ],
                decorator_list=[],
                returns=None,
                type_comment=None,
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    return CompiledPython(ast.unparse(module) + "\n")


__all__ = [
    "CompiledPython",
    "Completion",
    "Example",
    "FrozenCheckpoint",
    "Layer1Protocol",
    "PersistentStateChangedError",
    "Program",
    "ProgramFormatError",
    "ProgramParameter",
    "ProgramRaceAudit",
    "ProgramType",
    "TypedSignature",
    "compile",
    "complete",
    "grow",
]
