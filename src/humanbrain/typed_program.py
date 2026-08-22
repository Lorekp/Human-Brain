"""Layer-1 typed program values and the deep grow/complete/compile seam.

Layer 0 remains the explicitly named scalar control in :mod:`humanbrain.program`.
This module does not alias or fall back to that control.
"""

from __future__ import annotations

import ast
import json
import keyword
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import ClassVar

import numpy as np

from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig, RecurrentRace
from humanbrain.recurrent_state import RecurrentCellSynapseState


def _strict_record(value: object, keys: frozenset[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _record_sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an ordered sequence")
    return value


def _python_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(f"{label} must be a non-keyword Python identifier")
    return value


@dataclass(frozen=True, slots=True)
class ProgramType:
    """One immutable public type in the Layer-1 program language."""

    constructor: str
    arguments: tuple[ProgramType, ...] = ()

    _ARITY: ClassVar[dict[str, int]] = {
        "Int": 0,
        "Bool": 0,
        "Char": 0,
        "String": 0,
        "List": 1,
        "Pair": 2,
    }

    def __post_init__(self) -> None:
        if not isinstance(self.constructor, str) or self.constructor not in self._ARITY:
            raise ValueError("unknown program type constructor")
        if not isinstance(self.arguments, tuple) or not all(
            isinstance(argument, ProgramType) for argument in self.arguments
        ):
            raise TypeError("program type arguments must be a tuple of ProgramType values")
        if len(self.arguments) != self._ARITY[self.constructor]:
            raise ValueError(
                f"{self.constructor} expects {self._ARITY[self.constructor]} type arguments"
            )

    @classmethod
    def Int(cls) -> ProgramType:
        return cls("Int")

    @classmethod
    def Bool(cls) -> ProgramType:
        return cls("Bool")

    @classmethod
    def Char(cls) -> ProgramType:
        return cls("Char")

    @classmethod
    def String(cls) -> ProgramType:
        return cls("String")

    @classmethod
    def List(cls, element: ProgramType) -> ProgramType:
        return cls("List", (element,))

    @classmethod
    def Pair(cls, first: ProgramType, second: ProgramType) -> ProgramType:
        return cls("Pair", (first, second))

    def to_record(self) -> dict[str, object]:
        return {
            "constructor": self.constructor,
            "arguments": [argument.to_record() for argument in self.arguments],
        }

    @classmethod
    def from_record(cls, value: object) -> ProgramType:
        record = _strict_record(
            value,
            frozenset({"constructor", "arguments"}),
            "program type record",
        )
        constructor = record["constructor"]
        if not isinstance(constructor, str):
            raise ValueError("program type constructor must be a string")
        arguments = tuple(
            cls.from_record(argument)
            for argument in _record_sequence(record["arguments"], "program type arguments")
        )
        return cls(constructor=constructor, arguments=arguments)


@dataclass(frozen=True, slots=True)
class ProgramParameter:
    """One ordered, named parameter at the Layer-1 seam."""

    name: str
    type: ProgramType

    def __post_init__(self) -> None:
        _python_identifier(self.name, "program parameter name")
        if not isinstance(self.type, ProgramType):
            raise TypeError("program parameter type must be a ProgramType")

    def to_record(self) -> dict[str, object]:
        return {"name": self.name, "type": self.type.to_record()}

    @classmethod
    def from_record(cls, value: object) -> ProgramParameter:
        record = _strict_record(
            value,
            frozenset({"name", "type"}),
            "program parameter record",
        )
        return cls(
            name=_python_identifier(record["name"], "program parameter name"),
            type=ProgramType.from_record(record["type"]),
        )


@dataclass(frozen=True, slots=True)
class TypedSignature:
    """The complete ordered callable type supplied to growth and completion."""

    entry_point: str
    parameters: tuple[ProgramParameter, ...]
    result: ProgramType

    def __post_init__(self) -> None:
        _python_identifier(self.entry_point, "program entry point")
        if not isinstance(self.parameters, tuple) or not all(
            isinstance(parameter, ProgramParameter) for parameter in self.parameters
        ):
            raise TypeError("signature parameters must be a tuple of ProgramParameter values")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(names) != len(set(names)):
            raise ValueError("signature parameter names must be unique")
        if not isinstance(self.result, ProgramType):
            raise TypeError("signature result must be a ProgramType")

    def to_record(self) -> dict[str, object]:
        return {
            "entry_point": self.entry_point,
            "parameters": [parameter.to_record() for parameter in self.parameters],
            "result": self.result.to_record(),
        }

    @classmethod
    def from_record(cls, value: object) -> TypedSignature:
        record = _strict_record(
            value,
            frozenset({"entry_point", "parameters", "result"}),
            "typed signature record",
        )
        return cls(
            entry_point=_python_identifier(record["entry_point"], "program entry point"),
            parameters=tuple(
                ProgramParameter.from_record(parameter)
                for parameter in _record_sequence(
                    record["parameters"], "typed signature parameters"
                )
            ),
            result=ProgramType.from_record(record["result"]),
        )


class ProgramFormatError(ValueError):
    """A token stream is not one complete, canonical, well-typed program."""


class PersistentStateChangedError(RuntimeError):
    """Completion changed state that a frozen checkpoint owns."""


@dataclass(frozen=True, slots=True)
class Layer1Protocol:
    """The fixed Layer-1 protocol marker; it has no model-tuning surface."""


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


@dataclass(frozen=True, slots=True)
class FrozenCheckpoint:
    """Immutable Layer-1 state restored into a new circuit for each completion."""

    sha256: str
    training_digest: str
    training_input_sha256: tuple[str, ...]
    cell_synapse_sha256: str
    _payload: _FrozenCheckpointPayload = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _PrimitiveSpec:
    token: str
    role: str
    result: ProgramType | None = None
    arguments: tuple[ProgramType, ...] = ()
    lowering: str = ""
    value: int | str | None = None


_INT = ProgramType.Int()
_LITERAL_DIGITS = 4

# This is the sole language inventory. Its order fixes output-cell identity;
# its type/role fields drive typed-hole masking and parsing; and its lowering
# field drives structural Python emission.
_PRIMITIVE_SPECS = (
    _PrimitiveSpec("input-delimiter", "input-delimiter"),
    _PrimitiveSpec("add", "operation", _INT, (_INT, _INT), "add"),
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
_TOKEN_TO_ID = {
    spec.token: 256 + index for index, spec in enumerate(_PRIMITIVE_SPECS)
}
_ID_TO_SPEC = {_TOKEN_TO_ID[spec.token]: spec for spec in _PRIMITIVE_SPECS}
_INPUT_DELIMITER_ID = _TOKEN_TO_ID["input-delimiter"]
_TOKEN_CELL_COUNT = 256 + len(_PRIMITIVE_SPECS)

_CIRCUIT_CONFIG = RecurrentCircuitConfig(
    reservoir_cells=128,
    input_fanout=16,
    recurrent_fanout=8,
    random_seed=17,
)
_GROWTH_EPOCHS = 3
_MAX_PROGRAM_TOKENS = 128


@dataclass(frozen=True, slots=True)
class _Hole:
    expected: ProgramType
    environment: tuple[ProgramType, ...]


@dataclass(slots=True)
class _LiteralState:
    sign: str | None = None
    digits: list[int] = field(default_factory=list)


class _TypedHoleState:
    def __init__(self, signature: TypedSignature) -> None:
        environment = tuple(parameter.type for parameter in reversed(signature.parameters))
        self._holes = [_Hole(signature.result, environment)]
        self._literal: _LiteralState | None = None

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
            if spec.role in {"operation", "literal"} and spec.result == hole.expected:
                allowed.append(spec)
            elif spec.role == "variable":
                index = int(spec.value)
                if index < len(hole.environment) and hole.environment[index] == hole.expected:
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
            for argument in reversed(spec.arguments):
                self._holes.append(_Hole(argument, hole.environment))
        elif spec.role == "literal":
            self._literal = _LiteralState()


@dataclass(frozen=True, slots=True)
class _Expression:
    spec: _PrimitiveSpec
    children: tuple[_Expression, ...] = ()
    variable_index: int | None = None
    integer_value: int | None = None


def _parse_expression(
    tokens: tuple[str, ...],
    index: int,
    expected: ProgramType,
    environment: tuple[ProgramParameter, ...],
) -> tuple[_Expression, int]:
    if index >= len(tokens):
        raise ProgramFormatError("program expression is incomplete")
    try:
        spec = _SPEC_BY_TOKEN[tokens[index]]
    except KeyError as error:
        raise ProgramFormatError(f"unknown program token {tokens[index]!r}") from error

    if spec.role == "variable":
        variable_index = int(spec.value)
        if variable_index >= len(environment) or environment[variable_index].type != expected:
            raise ProgramFormatError("variable does not inhabit the current typed hole")
        return _Expression(spec, variable_index=variable_index), index + 1
    if spec.result != expected:
        raise ProgramFormatError("primitive does not inhabit the current typed hole")
    if spec.role == "literal":
        cursor = index + 1
        if cursor >= len(tokens) or _SPEC_BY_TOKEN.get(tokens[cursor], None) is None:
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
    if spec.role != "operation":
        raise ProgramFormatError("structural token cannot fill an expression hole")
    cursor = index + 1
    children: list[_Expression] = []
    for argument in spec.arguments:
        child, cursor = _parse_expression(tokens, cursor, argument, environment)
        children.append(child)
    return _Expression(spec, children=tuple(children)), cursor


def _validate_program(
    tokens: tuple[str, ...], signature: TypedSignature
) -> _Expression:
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
    environment = tuple(reversed(signature.parameters))
    expression, cursor = _parse_expression(tokens, 0, signature.result, environment)
    if cursor != len(tokens):
        raise ProgramFormatError("program has trailing tokens")
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


def _restore_circuit(frozen: _FrozenCircuitState) -> RecurrentCellCircuit:
    metadata = json.loads(frozen.metadata_json.decode("utf-8"))
    arrays = {
        item.name: np.frombuffer(item.data, dtype=np.dtype(item.dtype))
        .copy()
        .reshape(item.shape)
        for item in frozen.arrays
    }
    state = RecurrentCellSynapseState(metadata=metadata, arrays=arrays)
    if state.sha256 != frozen.sha256:
        raise ValueError("frozen checkpoint cell/synapse state digest mismatch")
    return RecurrentCellCircuit.restore_cell_synapse_state(state)


def _checkpoint_sha256(payload: _FrozenCheckpointPayload) -> str:
    return sha256(
        _canonical_json(
            {
                "format": "humanbrain-layer1-typed-program-v1",
                "primitive_token_order": [spec.token for spec in _PRIMITIVE_SPECS],
                "cell_synapse_sha256": payload.circuit.sha256,
                "training_digest": payload.training_digest,
                "training_input_sha256": list(payload.training_input_sha256),
            }
        )
    ).hexdigest()


def grow(protocol: Layer1Protocol, examples: Sequence[Example]) -> FrozenCheckpoint:
    """Fit one Layer-1 recurrent cell circuit and freeze its persistent state."""
    if type(protocol) is not Layer1Protocol:
        raise TypeError("grow requires the fixed Layer1Protocol")
    corpus = tuple(examples)
    if not corpus:
        raise ValueError("at least one public Layer-1 growth example is required")
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
        circuit.grow_sequence((*input_ids, *teacher_ids), epochs=_GROWTH_EPOCHS)
        digest.update(input_bytes)
        digest.update(_token_stream_bytes(teacher_ids))
    circuit.freeze_learning()
    payload = _FrozenCheckpointPayload(
        circuit=_freeze_circuit(circuit),
        training_digest=digest.hexdigest(),
        training_input_sha256=tuple(input_hashes),
    )
    checkpoint_digest = _checkpoint_sha256(payload)
    return FrozenCheckpoint(
        sha256=checkpoint_digest,
        training_digest=payload.training_digest,
        training_input_sha256=payload.training_input_sha256,
        cell_synapse_sha256=payload.circuit.sha256,
        _payload=payload,
    )


def complete(
    checkpoint: FrozenCheckpoint,
    prompt: str,
    typed_signature: TypedSignature,
) -> Completion:
    """Restore, observe exactly prompt+signature+delimiter, and emit one program."""
    if not isinstance(checkpoint, FrozenCheckpoint):
        raise TypeError("complete requires a FrozenCheckpoint")
    if not isinstance(typed_signature, TypedSignature):
        raise TypeError("complete requires a TypedSignature")
    if _checkpoint_sha256(checkpoint._payload) != checkpoint.sha256:
        raise ValueError("frozen checkpoint digest mismatch")

    circuit = _restore_circuit(checkpoint._payload.circuit)
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
    if _checkpoint_sha256(checkpoint._payload) != checkpoint.sha256:
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


def _lower_expression(
    expression: _Expression,
    environment: tuple[ProgramParameter, ...],
) -> ast.expr:
    lowering = expression.spec.lowering
    if lowering == "variable":
        assert expression.variable_index is not None
        return ast.Name(id=environment[expression.variable_index].name, ctx=ast.Load())
    if lowering == "integer":
        assert expression.integer_value is not None
        return ast.Constant(value=expression.integer_value)
    if lowering == "add":
        return ast.BinOp(
            left=_lower_expression(expression.children[0], environment),
            op=ast.Add(),
            right=_lower_expression(expression.children[1], environment),
        )
    raise ProgramFormatError("primitive has no structural Python lowering")


def compile(program: Program) -> CompiledPython:
    """Compile only a validated Program/signature into safe structural Python."""
    if not isinstance(program, Program):
        raise TypeError("compile requires a Program")
    if _program_sha256(program.tokens, program.signature) != program.sha256:
        raise ValueError("program digest mismatch")
    expression = _validate_program(program.tokens, program.signature)
    environment = tuple(reversed(program.signature.parameters))
    module = ast.Module(
        body=[
            ast.FunctionDef(
                name=program.signature.entry_point,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=parameter.name) for parameter in program.signature.parameters],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=[ast.Return(value=_lower_expression(expression, environment))],
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
