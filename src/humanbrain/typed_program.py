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
from itertools import islice

import numpy as np

from humanbrain.cells import AllenGlif5Parameters
from humanbrain.human_cell_identity import HUMAN_GLIF5_IDENTITY
from humanbrain.recurrent import (
    RecurrentCellCircuit,
    RecurrentCircuitConfig,
    RecurrentRace,
    _BASE_STATE_ARRAYS as _CIRCUIT_BASE_STATE_ARRAYS,
    _READOUT_STATE_LAYOUTS as _CIRCUIT_READOUT_STATE_LAYOUTS,
)
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
class ProgramEditSite:
    """One stable typed expression path and its complete prefix-token subtree."""

    path: tuple[int, ...]
    tokens: tuple[str, ...]
    result_type_label: str
    head_production: str
    scope_binding_type_labels: tuple[str, str]
    is_binder: bool


@dataclass(frozen=True, slots=True)
class FactorActionSite:
    """Compiler-owned metadata sufficient to enumerate generic edit factors."""

    path: tuple[int, ...]
    result_type_label: str
    head_production: str
    scope_binding_type_labels: tuple[str, str]
    is_binder: bool
    selected_scope_index: int | None


@dataclass(frozen=True, slots=True)
class FactorAction:
    """One generic, factorized typed edit selected by physical motor races."""

    path: tuple[int, ...]
    schema: str
    operator: str
    left_source: str
    right_source: str
    literal: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.path, tuple) or not all(
            type(part) is int and part >= 0 for part in self.path
        ):
            raise TypeError("factor-action path must contain non-negative integers")
        if self.schema not in {"DONE", "APPLY1", "APPLY2", "MAP_BIND"}:
            raise ValueError("unknown factor-action schema")
        if self.operator not in {
            "NONE",
            "absolute",
            "add",
            "subtract",
            "less-than",
            "map",
            "minimum",
            "maximum",
        }:
            raise ValueError("unknown factor-action operator")
        sources = {"NONE", "SELECTED", "SCOPE0", "SCOPE1", "BOUND0", "LITERAL"}
        if self.left_source not in sources or self.right_source not in sources:
            raise ValueError("unknown factor-action operand source")
        if self.literal is not None and (
            type(self.literal) is not int
            or abs(self.literal) >= 10**_LITERAL_DIGITS
        ):
            raise ValueError("factor-action literal exceeds the typed grammar bound")


def _factor_action_record(action: FactorAction) -> dict[str, object]:
    return {
        "path": action.path,
        "schema": action.schema,
        "operator": action.operator,
        "left_source": action.left_source,
        "right_source": action.right_source,
        "literal": action.literal,
    }


_GLOBAL_DONE_FACTOR_ACTION = FactorAction(
    (),
    "DONE",
    "NONE",
    "NONE",
    "NONE",
    None,
)


@dataclass(frozen=True, slots=True)
class FactorCandidateBank:
    """One deterministic typed FactorAction bank for an incumbent program."""

    program_sha256: str
    literals: tuple[int, ...]
    actions: tuple[FactorAction, ...]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.program_sha256, str)
            or len(self.program_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.program_sha256
            )
        ):
            raise ValueError("factor candidate bank needs a lowercase program SHA-256")
        if (
            not isinstance(self.literals, tuple)
            or not self.literals
            or not all(type(value) is int for value in self.literals)
        ):
            raise TypeError("factor candidate literals must be a non-empty integer tuple")
        if len(set(self.literals)) != len(self.literals):
            raise ValueError("factor candidate literals must be unique")
        if self.literals != tuple(sorted(self.literals)):
            raise ValueError("factor candidate literals must be in canonical order")
        if any(abs(value) >= 10**_LITERAL_DIGITS for value in self.literals):
            raise ValueError("factor candidate literal exceeds the typed grammar bound")
        if not isinstance(self.actions, tuple) or not all(
            type(action) is FactorAction for action in self.actions
        ):
            raise TypeError("factor candidate actions must be a tuple of FactorAction values")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("factor candidate actions must be unique")
        if (
            not self.actions
            or self.actions[0] != _GLOBAL_DONE_FACTOR_ACTION
            or sum(action == _GLOBAL_DONE_FACTOR_ACTION for action in self.actions) != 1
        ):
            raise ValueError("factor candidate bank needs one leading global DONE action")
        object.__setattr__(
            self,
            "sha256",
            sha256(
                _canonical_json(
                    tuple(_factor_action_record(action) for action in self.actions)
                )
            ).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class CompiledEditSite:
    """Trusted Python for one value-producing site of an incumbent program."""

    path: tuple[int, ...]
    result_type_label: str
    scope_type_labels: tuple[str, ...]
    source: str
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.path, tuple) or not all(
            type(part) is int and part >= 0 for part in self.path
        ):
            raise TypeError("compiled edit-site path must contain non-negative integers")
        if not isinstance(self.result_type_label, str) or not self.result_type_label:
            raise ValueError("compiled edit-site result type must be non-empty")
        if not isinstance(self.scope_type_labels, tuple) or not all(
            isinstance(label, str) and label for label in self.scope_type_labels
        ):
            raise TypeError("compiled edit-site scope types must be non-empty strings")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("compiled edit-site source must be non-empty")
        object.__setattr__(self, "sha256", sha256(self.source.encode("utf-8")).hexdigest())


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
    allowed_output_currents_pa: tuple[float, ...]
    excluded_cell_ids: tuple[int, ...]
    output_current_sha256: str
    eligibility_bytes: bytes = field(repr=False)


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
    training_sequence_token_counts: tuple[int, ...]
    training_teacher_token_counts: tuple[int, ...]
    protocol_json: bytes


@dataclass(frozen=True, slots=True, init=False)
class FrozenCheckpoint:
    """Immutable Layer-1 state restored into a new circuit for each completion."""

    lane: str
    sha256: str
    training_digest: str
    training_input_sha256: tuple[str, ...]
    training_sequence_token_counts: tuple[int, ...]
    training_teacher_token_counts: tuple[int, ...]
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
            json.loads(payload.protocol_json.decode("utf-8"))["lane"],
            sha256(self._bytes).hexdigest(),
            payload.training_digest,
            payload.training_input_sha256,
            payload.training_sequence_token_counts,
            payload.training_teacher_token_counts,
            payload.circuit.sha256,
            sha256(payload.protocol_json).hexdigest(),
        )
        actual = (
            self.lane,
            self.sha256,
            self.training_digest,
            self.training_input_sha256,
            self.training_sequence_token_counts,
            self.training_teacher_token_counts,
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
        object.__setattr__(
            checkpoint,
            "lane",
            json.loads(payload.protocol_json.decode("utf-8"))["lane"],
        )
        object.__setattr__(checkpoint, "sha256", sha256(value).hexdigest())
        object.__setattr__(checkpoint, "training_digest", payload.training_digest)
        object.__setattr__(
            checkpoint,
            "training_input_sha256",
            payload.training_input_sha256,
        )
        object.__setattr__(
            checkpoint,
            "training_sequence_token_counts",
            payload.training_sequence_token_counts,
        )
        object.__setattr__(
            checkpoint,
            "training_teacher_token_counts",
            payload.training_teacher_token_counts,
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
_BOOL = _type_term(ProgramType.Bool())
_A = _TypeVariable("a")
_B = _TypeVariable("b")
_LITERAL_DIGITS = 4

# This is the sole language inventory. Its order fixes output-cell identity;
# its type/role fields drive typed-hole masking and parsing; and its lowering
# field drives structural Python emission.
_PRIMITIVE_SPECS = (
    _PrimitiveSpec("context-delimiter", "input-delimiter"),
    _PrimitiveSpec("input-delimiter", "input-delimiter"),
    _PrimitiveSpec("add", "operation", _INT, (_INT, _INT), "add"),
    _PrimitiveSpec("subtract", "operation", _INT, (_INT, _INT), "subtract"),
    _PrimitiveSpec("minimum", "operation", _INT, (_INT, _INT), "minimum"),
    _PrimitiveSpec("maximum", "operation", _INT, (_INT, _INT), "maximum"),
    _PrimitiveSpec("absolute", "operation", _INT, (_INT,), "absolute"),
    _PrimitiveSpec("less-than", "operation", _BOOL, (_INT, _INT), "less-than"),
    _PrimitiveSpec("if", "operation", _A, (_BOOL, _A, _A), "if"),
    _PrimitiveSpec(
        "map",
        "operation",
        _list_type(_B),
        (_function_type((_A,), _B), _list_type(_A)),
        "map",
    ),
    _PrimitiveSpec("sum", "operation", _INT, (_list_type(_INT),), "sum"),
    _PrimitiveSpec(
        "count",
        "operation",
        _INT,
        (_function_type((_INT,), _BOOL), _list_type(_INT)),
        "count",
    ),
    _PrimitiveSpec("bind", "binder", lowering="binder"),
    _PrimitiveSpec("variable-0", "variable", lowering="variable", value=0),
    _PrimitiveSpec("variable-1", "variable", lowering="variable", value=1),
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
_CONTEXT_DELIMITER_ID = _TOKEN_TO_ID["context-delimiter"]
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
# v5 checkpoint diet (documented cap change): the frozen Layer-1 artifact
# serializes fitted readout WEIGHTS plus topology/config state only — never
# the ridge sufficient statistics (`normal_matrix` is F^2 with
# F = K*R + 1 = 1025 under the multi-trace feature space, i.e. ~8.4 MB of
# raw f64 before base64, versus ~2.2 MB for the (277, 1024) weight matrix).
# Statistics are runtime-transient per protocol ("stats never migrate",
# docs/research/r0-multitrace-protocol.md §1.5 disposition); a restored
# circuit that ever resumes training refits them from scratch — retraining
# is the sanctioned path. 4 MiB covers the dieted payload (~3.1 MB canonical
# JSON: weights dominate) with headroom; 2 MiB cannot, even dieted.
_MAX_CHECKPOINT_BYTES = 4_194_304
_MAX_PROGRAM_TOKENS = 128
_MAX_AST_NODES = 64
_MAX_AST_DEPTH = 16
_PROTOCOL_VERSION = "humanbrain-layer1-typed-program-v5"
_A0_FRAMING_ORDER = "prompt/context-delimiter/canonical-signature/input-delimiter"
_A1_FRAMING_ORDER = "canonical-signature/context-delimiter/prompt/input-delimiter"
_ALL_TRANSITION_SUPERVISION = "all-next-token-targets"
_PROGRAM_ONLY_SUPERVISION = "program-token-targets-only"
_LANE_COMMITMENTS = {
    "A0": (_A0_FRAMING_ORDER, _ALL_TRANSITION_SUPERVISION),
    "A1": (_A1_FRAMING_ORDER, _ALL_TRANSITION_SUPERVISION),
    "A2": (_A1_FRAMING_ORDER, _PROGRAM_ONLY_SUPERVISION),
}
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
        (
            _TOKEN_CELL_COUNT,
            # v5 diet: the fitted weight matrix spans the full multi-trace
            # feature space (K*R sources per token cell). The F^2 ridge
            # statistics are deliberately ABSENT from this table: they are
            # runtime-only state and never migrate through a checkpoint.
            len(_CIRCUIT_CONFIG.eligibility_taus_ms)
            * _CIRCUIT_CONFIG.reservoir_cells,
        ),
    ),
}
_FROZEN_RUNTIME_ONLY_ARRAYS = frozenset({"normal_matrix", "target_moments"})
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

    lane: str = "A2"
    protocol_version: str = _PROTOCOL_VERSION
    compiler_id: str = _COMPILER_ID
    primitive_table_sha256: str = _PRIMITIVE_TABLE_SHA256
    circuit_config_sha256: str = _CIRCUIT_CONFIG_SHA256
    growth_epochs: int = _GROWTH_EPOCHS
    framing_order: str = _A1_FRAMING_ORDER
    supervision_scope: str = _PROGRAM_ONLY_SUPERVISION
    allen_glif5_config_sha256: str = HUMAN_GLIF5_IDENTITY.config_sha256
    allen_glif5_parameters: AllenGlif5Parameters = _ALLEN_GLIF5_PARAMETERS
    max_training_examples: int = _MAX_TRAINING_EXAMPLES
    max_checkpoint_bytes: int = _MAX_CHECKPOINT_BYTES
    literal_digit_bound: int = _LITERAL_DIGITS
    max_ast_nodes: int = _MAX_AST_NODES
    max_ast_depth: int = _MAX_AST_DEPTH
    max_output_tokens: int = _MAX_PROGRAM_TOKENS

    @classmethod
    def for_lane(cls, lane: str) -> Layer1Protocol:
        """Construct one of the two diagnostic lanes or sole eligible A2 lane."""
        if lane not in _LANE_COMMITMENTS:
            raise ValueError("Layer-1 lane must be exactly A0, A1, or A2")
        framing_order, supervision_scope = _LANE_COMMITMENTS[lane]
        return cls(
            lane=lane,
            framing_order=framing_order,
            supervision_scope=supervision_scope,
        )

    def __post_init__(self) -> None:
        if self.lane not in _LANE_COMMITMENTS:
            raise ValueError("Layer-1 lane must be exactly A0, A1, or A2")
        expected_framing, expected_supervision = _LANE_COMMITMENTS[self.lane]
        expected = (
            self.lane,
            _PROTOCOL_VERSION,
            _COMPILER_ID,
            _PRIMITIVE_TABLE_SHA256,
            _CIRCUIT_CONFIG_SHA256,
            _GROWTH_EPOCHS,
            expected_framing,
            expected_supervision,
            HUMAN_GLIF5_IDENTITY.config_sha256,
            _ALLEN_GLIF5_PARAMETERS,
            _MAX_TRAINING_EXAMPLES,
            _MAX_CHECKPOINT_BYTES,
            _LITERAL_DIGITS,
            _MAX_AST_NODES,
            _MAX_AST_DEPTH,
            _MAX_PROGRAM_TOKENS,
        )
        actual = (
            self.lane,
            self.protocol_version,
            self.compiler_id,
            self.primitive_table_sha256,
            self.circuit_config_sha256,
            self.growth_epochs,
            self.framing_order,
            self.supervision_scope,
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
            "lane": protocol.lane,
            "protocol_version": protocol.protocol_version,
            "compiler_id": protocol.compiler_id,
            "primitive_table": _primitive_table_record(),
            "primitive_table_sha256": protocol.primitive_table_sha256,
            "circuit_config": asdict(_CIRCUIT_CONFIG),
            "circuit_config_sha256": protocol.circuit_config_sha256,
            "growth_epochs": protocol.growth_epochs,
            "framing_order": protocol.framing_order,
            "supervision_scope": protocol.supervision_scope,
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
    token_start: int
    token_stop: int
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
        return (
            _Expression(
                spec,
                token_start=index,
                token_stop=index + 1,
                variable_index=int(spec.value),
            ),
            index + 1,
        )
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
        return (
            _Expression(
                spec,
                token_start=index,
                token_stop=cursor + 1,
                integer_value=integer_value,
            ),
            cursor + 1,
        )
    if spec.role == "binder":
        body, cursor = _parse_expression(tokens, index + 1)
        return (
            _Expression(
                spec,
                token_start=index,
                token_stop=cursor,
                children=(body,),
            ),
            cursor,
        )
    if spec.role != "operation":
        raise ProgramFormatError("structural token cannot fill an expression hole")
    cursor = index + 1
    children: list[_Expression] = []
    for _ in spec.arguments:
        child, cursor = _parse_expression(tokens, cursor)
        children.append(child)
    return (
        _Expression(
            spec,
            token_start=index,
            token_stop=cursor,
            children=tuple(children),
        ),
        cursor,
    )


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


def _type_term_label(
    term: _TypeTerm,
    substitutions: Mapping[int, _TypeTerm],
) -> str:
    resolved = _resolve(term, substitutions)
    if isinstance(resolved, _TypeVariable):
        return "unresolved"
    if resolved.constructor == "Function":
        return "internal-function"
    if not resolved.arguments:
        return resolved.constructor
    arguments = ",".join(
        _type_term_label(argument, substitutions) for argument in resolved.arguments
    )
    return f"{resolved.constructor}[{arguments}]"


def _typed_edit_site_records(
    root: _Expression,
    signature: TypedSignature,
) -> tuple[
    tuple[
        tuple[
            _Expression,
            tuple[int, ...],
            _TypeTerm,
            tuple[_TypeTerm, ...],
        ],
        ...,
    ],
    dict[int, _TypeTerm],
]:
    """Replay the existing type machinery while retaining each node's local scope."""

    substitutions: dict[int, _TypeTerm] = {}
    next_variable = 0
    records: list[
        tuple[
            _Expression,
            tuple[int, ...],
            _TypeTerm,
            tuple[_TypeTerm, ...],
        ]
    ] = []

    def visit(
        expression: _Expression,
        path: tuple[int, ...],
        expected: _TypeTerm,
        environment: tuple[_TypeTerm, ...],
    ) -> None:
        nonlocal next_variable
        spec = expression.spec

        if spec.role == "operation":
            assert spec.result is not None
            result, arguments, following_variable = _instantiate_spec(
                spec,
                next_variable,
            )
            if not _unify(result, expected, substitutions):
                raise ProgramFormatError("operation does not inhabit its typed edit site")
            next_variable = following_variable
            if len(arguments) != len(expression.children):
                raise ProgramFormatError("operation child count differs from its grammar")
            records.append((expression, path, expected, environment))
            for child_index, (child, argument) in enumerate(
                zip(expression.children, arguments, strict=True)
            ):
                visit(child, (*path, child_index), argument, environment)
            return

        if spec.role == "variable":
            index = int(spec.value)
            if index >= len(environment) or not _unify(
                environment[index],
                expected,
                substitutions,
            ):
                raise ProgramFormatError("variable does not inhabit its typed edit site")
            records.append((expression, path, expected, environment))
            return

        if spec.role == "literal":
            assert spec.result is not None
            result, _, following_variable = _instantiate_spec(spec, next_variable)
            if not _unify(result, expected, substitutions):
                raise ProgramFormatError("literal does not inhabit its typed edit site")
            next_variable = following_variable
            records.append((expression, path, expected, environment))
            return

        if spec.role == "binder":
            parts = _function_parts(_resolve(expected, substitutions))
            if parts is None or len(expression.children) != 1:
                raise ProgramFormatError("binder does not inhabit an internal function site")
            parameters, result = parts
            records.append((expression, path, expected, environment))
            binder_environment = (*reversed(parameters), *environment)
            visit(expression.children[0], (*path, 0), result, binder_environment)
            return

        raise ProgramFormatError("structural token cannot describe an edit site")

    environment = tuple(
        _type_term(parameter.type) for parameter in reversed(signature.parameters)
    )
    visit(root, (), _type_term(signature.result), environment)
    return tuple(records), substitutions


def program_edit_sites(program: Program) -> tuple[ProgramEditSite, ...]:
    """Return every expression subtree with stable local type/scope metadata."""

    if not isinstance(program, Program):
        raise TypeError("program must be a Program")
    root = _validate_program(program.tokens, program.signature)
    records, substitutions = _typed_edit_site_records(root, program.signature)

    sites: list[ProgramEditSite] = []
    for expression, path, expected, environment in records:
        scope_labels = tuple(
            _type_term_label(environment[index], substitutions)
            if index < len(environment)
            else "unbound"
            for index in range(2)
        )
        sites.append(
            ProgramEditSite(
                path=path,
                tokens=program.tokens[expression.token_start : expression.token_stop],
                result_type_label=_type_term_label(expected, substitutions),
                head_production=expression.spec.token,
                scope_binding_type_labels=(scope_labels[0], scope_labels[1]),
                is_binder=expression.spec.role == "binder",
            )
        )
    return tuple(sites)


def replace_subtree(
    program: Program,
    path: tuple[int, ...],
    replacement_tokens: tuple[str, ...],
) -> Program:
    """Replace one expression subtree and validate the complete typed result."""

    if not isinstance(program, Program):
        raise TypeError("program must be a Program")
    if not isinstance(path, tuple) or not all(
        type(part) is int and part >= 0 for part in path
    ):
        raise TypeError("edit path must be a tuple of non-negative integers")
    if not isinstance(replacement_tokens, tuple) or not replacement_tokens:
        raise ValueError("replacement tokens must be a non-empty tuple")
    if not all(isinstance(token, str) and token for token in replacement_tokens):
        raise TypeError("replacement tokens must be non-empty strings")
    _, replacement_stop = _parse_expression(replacement_tokens, 0)
    if replacement_stop != len(replacement_tokens):
        raise ProgramFormatError("replacement must contain exactly one expression")

    root = _validate_program(program.tokens, program.signature)
    selected = root
    for part in path:
        if part >= len(selected.children):
            raise ValueError(f"program has no edit site at path {path!r}")
        selected = selected.children[part]
    tokens = (
        *program.tokens[: selected.token_start],
        *replacement_tokens,
        *program.tokens[selected.token_stop :],
    )
    edited = Program(tokens, program.signature)
    compile(edited)
    return edited


def _factor_literal_tokens(value: int) -> tuple[str, ...]:
    sign = "-" if value < 0 else "+"
    return ("integer", sign, *tuple(str(abs(value))), "end")


def _inside_binder_scope(
    sites: Sequence[ProgramEditSite | FactorActionSite],
    path: tuple[int, ...],
) -> bool:
    return any(
        site.is_binder
        and len(path) > len(site.path)
        and path[: len(site.path)] == site.path
        for site in sites
    )


def _factor_literal_bank(literals: Sequence[int]) -> tuple[int, ...]:
    try:
        supplied = tuple(literals)
    except TypeError as error:
        raise TypeError(
            "factor candidate literals must be an iterable of integers"
        ) from error
    if not supplied:
        raise ValueError("factor candidate literal bank must be non-empty")
    if not all(type(value) is int for value in supplied):
        raise TypeError("factor candidate literals must be exact Python integers")
    if len(set(supplied)) != len(supplied):
        raise ValueError("factor candidate literal bank must not contain duplicates")
    if any(abs(value) >= 10**_LITERAL_DIGITS for value in supplied):
        raise ValueError("factor candidate literal exceeds the typed grammar bound")
    return tuple(sorted(supplied))


def _factor_dynamic_sources(
    site: FactorActionSite,
    sites: Sequence[FactorActionSite],
    required_type_label: str,
) -> tuple[str, ...]:
    selected_available = site.result_type_label == required_type_label
    inside_binder = _inside_binder_scope(sites, site.path)

    sources: list[str] = []
    if selected_available:
        sources.append("SELECTED")
    for index, type_label in enumerate(site.scope_binding_type_labels):
        if type_label != required_type_label:
            continue
        if selected_available and site.selected_scope_index == index:
            continue
        if inside_binder and index == 0:
            continue
        sources.append(f"SCOPE{index}")
    if (
        inside_binder
        and site.scope_binding_type_labels[0] == required_type_label
        and not (selected_available and site.selected_scope_index == 0)
    ):
        sources.append("BOUND0")
    return tuple(sources)


def _ordered_factor_operands(
    dynamic_sources: tuple[str, ...],
    literals: tuple[int, ...],
) -> tuple[tuple[str, str, int | None], ...]:
    operands: list[tuple[str, str, int | None]] = []
    for left_source in dynamic_sources:
        for right_source in dynamic_sources:
            operands.append((left_source, right_source, None))
    for dynamic_source in dynamic_sources:
        for literal in literals:
            operands.append((dynamic_source, "LITERAL", literal))
            operands.append(("LITERAL", dynamic_source, literal))
    return tuple(operands)


def factor_action_sites(program: Program) -> tuple[FactorActionSite, ...]:
    """Project incumbent edit sites into candidate-enumeration metadata."""

    if not isinstance(program, Program):
        raise TypeError("factor-action site projection requires a Program")
    projected: list[FactorActionSite] = []
    for site in program_edit_sites(program):
        selected_scope_index = (
            int(site.tokens[0][-1])
            if len(site.tokens) == 1
            and site.tokens[0] in {"variable-0", "variable-1"}
            else None
        )
        projected.append(
            FactorActionSite(
                path=site.path,
                result_type_label=site.result_type_label,
                head_production=site.head_production,
                scope_binding_type_labels=site.scope_binding_type_labels,
                is_binder=site.is_binder,
                selected_scope_index=selected_scope_index,
            )
        )
    return tuple(projected)


def factor_actions_for_sites(
    sites: Sequence[FactorActionSite],
    *,
    literals: Sequence[int],
) -> tuple[FactorAction, ...]:
    """Enumerate actions from compiler-owned typed site metadata.

    This is the common legality seam for real incumbent sites and learned-world-
    model predictions.  It never materializes, compiles, or executes a candidate
    program; callers that predict hypothetical structure must supply the same
    :class:`FactorActionSite` records.  This keeps hypothetical candidate
    enumeration independent of subtree tokens or candidate AST construction.
    """

    try:
        site_bank = tuple(sites)
    except TypeError as error:
        raise TypeError(
            "factor sites must be an iterable of FactorActionSite values"
        ) from error
    if not all(type(site) is FactorActionSite for site in site_bank):
        raise TypeError("factor sites must contain FactorActionSite values")
    if len({site.path for site in site_bank}) != len(site_bank):
        raise ValueError("factor sites must have unique paths")
    literal_bank = _factor_literal_bank(literals)
    actions: list[FactorAction] = []
    seen: set[FactorAction] = set()

    def append(action: FactorAction) -> None:
        if action not in seen:
            seen.add(action)
            actions.append(action)

    append(_GLOBAL_DONE_FACTOR_ACTION)
    for site in site_bank:
        if site.result_type_label == "Int":
            dynamic_sources = _factor_dynamic_sources(site, site_bank, "Int")
            for source in dynamic_sources:
                append(
                    FactorAction(
                        site.path,
                        "APPLY1",
                        "absolute",
                        source,
                        "NONE",
                        None,
                    )
                )
            operands = _ordered_factor_operands(dynamic_sources, literal_bank)
            for operator in ("add", "subtract"):
                for left_source, right_source, literal in operands:
                    append(
                        FactorAction(
                            site.path,
                            "APPLY2",
                            operator,
                            left_source,
                            right_source,
                            literal,
                        )
                    )
        elif site.result_type_label == "Bool":
            dynamic_sources = _factor_dynamic_sources(site, site_bank, "Int")
            for left_source, right_source, literal in _ordered_factor_operands(
                dynamic_sources,
                literal_bank,
            ):
                append(
                    FactorAction(
                        site.path,
                        "APPLY2",
                        "less-than",
                        left_source,
                        right_source,
                        literal,
                    )
                )
        elif site.result_type_label == "List[Int]":
            for collection_source in _factor_dynamic_sources(
                site,
                site_bank,
                "List[Int]",
            ):
                append(
                    FactorAction(
                        site.path,
                        "MAP_BIND",
                        "map",
                        "BOUND0",
                        collection_source,
                        None,
                    )
                )

    return tuple(actions)


def factor_action_candidates(
    program: Program,
    *,
    literals: Sequence[int],
) -> FactorCandidateBank:
    """Enumerate the complete typed factor bank without materializing edits."""

    if not isinstance(program, Program):
        raise TypeError("factor candidate enumeration requires a Program")
    if _program_sha256(program.tokens, program.signature) != program.sha256:
        raise ValueError("program digest mismatch")
    literal_bank = _factor_literal_bank(literals)
    actions = factor_actions_for_sites(
        factor_action_sites(program),
        literals=literal_bank,
    )
    return FactorCandidateBank(
        program_sha256=program.sha256,
        literals=literal_bank,
        actions=actions,
    )


def _factor_source_tokens(
    action: FactorAction,
    source: str,
    selected: ProgramEditSite,
    sites: Sequence[ProgramEditSite],
    *,
    constructed_bound: bool = False,
) -> tuple[str, ...]:
    if source == "SELECTED":
        return selected.tokens
    if source in {"SCOPE0", "SCOPE1"}:
        index = int(source[-1])
        if selected.scope_binding_type_labels[index] == "unbound":
            raise ValueError(f"{source} is not bound at edit site {selected.path!r}")
        return (f"variable-{index}",)
    if source == "BOUND0":
        if not constructed_bound and not _inside_binder_scope(sites, selected.path):
            raise ValueError("BOUND0 is available only inside a binder body")
        return ("variable-0",)
    if source == "LITERAL":
        if action.literal is None:
            raise ValueError("LITERAL source requires one factor-action literal")
        return _factor_literal_tokens(action.literal)
    if source == "NONE":
        raise ValueError("NONE cannot materialize an operand")
    raise AssertionError("FactorAction rejected an unknown operand source")


def apply_factor_action(program: Program, action: FactorAction) -> Program:
    """Materialize and apply one selected generic action through the typed seam."""

    if not isinstance(program, Program):
        raise TypeError("factor-action actuator requires a Program")
    if type(action) is not FactorAction:
        raise TypeError("factor-action actuator requires a FactorAction")
    sites = program_edit_sites(program)

    if action.schema == "DONE":
        if (
            action.path != ()
            or action.operator != "NONE"
            or action.left_source != "NONE"
            or action.right_source != "NONE"
            or action.literal is not None
        ):
            raise ValueError("DONE must be the global factor-action identity")
        return program

    selected = next((site for site in sites if site.path == action.path), None)
    if selected is None:
        raise ValueError(f"program has no edit site at path {action.path!r}")
    uses_literal = action.left_source == "LITERAL" or action.right_source == "LITERAL"
    if uses_literal != (action.literal is not None):
        raise ValueError("factor-action literal must match its operand sources")

    if action.schema == "APPLY1":
        if action.operator != "absolute" or action.right_source != "NONE":
            raise ValueError("APPLY1 supports only unary absolute")
        operand = _factor_source_tokens(action, action.left_source, selected, sites)
        replacement = ("absolute", *operand)
    elif action.schema == "APPLY2":
        if action.operator not in {
            "add",
            "subtract",
            "less-than",
            "minimum",
            "maximum",
        }:
            raise ValueError(
                "APPLY2 supports add, subtract, less-than, minimum, or maximum"
            )
        if action.left_source == "NONE" or action.right_source == "NONE":
            raise ValueError("APPLY2 requires two operand sources")
        left = _factor_source_tokens(action, action.left_source, selected, sites)
        right = _factor_source_tokens(action, action.right_source, selected, sites)
        replacement = (action.operator, *left, *right)
    elif action.schema == "MAP_BIND":
        if (
            action.operator != "map"
            or action.left_source != "BOUND0"
            or action.right_source not in {"SELECTED", "SCOPE0", "SCOPE1", "BOUND0"}
            or action.literal is not None
        ):
            raise ValueError(
                "MAP_BIND requires BOUND0 as its body and one outer collection source"
            )
        body = _factor_source_tokens(
            action,
            action.left_source,
            selected,
            sites,
            constructed_bound=True,
        )
        collection = _factor_source_tokens(
            action,
            action.right_source,
            selected,
            sites,
        )
        replacement = ("map", "bind", *body, *collection)
    else:
        raise AssertionError("FactorAction rejected an unknown schema")

    return replace_subtree(program, action.path, replacement)


def _program_token_ids(tokens: Sequence[str]) -> tuple[int, ...]:
    try:
        return tuple(_TOKEN_TO_ID[token] for token in tokens)
    except KeyError as error:
        raise ProgramFormatError(f"unknown program token {error.args[0]!r}") from error


def _signature_bytes(signature: TypedSignature) -> bytes:
    return _canonical_json(signature.to_record())


def _model_input(
    prompt: str,
    signature: TypedSignature,
    *,
    framing_order: str = _A1_FRAMING_ORDER,
) -> tuple[int, ...]:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("completion prompt must be a non-empty string")
    prompt_ids = tuple(prompt.encode("utf-8"))
    signature_ids = _signature_bytes(signature)
    if framing_order == _A0_FRAMING_ORDER:
        return (
            *prompt_ids,
            _CONTEXT_DELIMITER_ID,
            *signature_ids,
            _INPUT_DELIMITER_ID,
        )
    if framing_order == _A1_FRAMING_ORDER:
        return (
            *signature_ids,
            _CONTEXT_DELIMITER_ID,
            *prompt_ids,
            _INPUT_DELIMITER_ID,
        )
    raise ValueError("unsupported Layer-1 framing order")


def _token_stream_bytes(token_ids: Sequence[int]) -> bytes:
    return b"".join(int(token_id).to_bytes(2, "big") for token_id in token_ids)


def _freeze_circuit(circuit: RecurrentCellCircuit) -> _FrozenCircuitState:
    # v5 dieted capture: ridge sufficient statistics are runtime-only and the
    # metadata reduces to the canonical circuit schema, so the digest is taken
    # over exactly the persisted domain (decode/complete recompute it with the
    # same dieted capture).
    state = circuit.capture_cell_synapse_state(include_ridge_statistics=False)
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


def _persisted_circuit_bindings(
    readout_learning_rule: str,
) -> tuple[tuple[str, str, object], ...]:
    """Checkpoint array bindings that survive the v5 diet, in restore order.

    Derived from the circuit's own state tables minus the runtime-only
    statistics arrays, so the frozen artifact can never silently drift from
    what ``capture_cell_synapse_state`` produces.
    """
    layout = _CIRCUIT_READOUT_STATE_LAYOUTS[readout_learning_rule]
    return tuple(
        (attribute, key, dtype)
        for attribute, key, dtype in (
            *_CIRCUIT_BASE_STATE_ARRAYS,
            *layout.direct_arrays,
        )
        if key not in _FROZEN_RUNTIME_ONLY_ARRAYS
    )


def _restore_circuit(frozen: _FrozenCircuitState) -> RecurrentCellCircuit:
    metadata = json.loads(frozen.metadata_json.decode("utf-8"))
    arrays = {
        item.name: np.frombuffer(item.data, dtype=np.dtype(item.dtype)).reshape(item.shape)
        for item in frozen.arrays
    }
    # Dieted restore (v5): metadata and every array header/digest were already
    # strictly validated against the protocol during decode. Ridge statistics
    # are intentionally absent — a restored circuit keeps them as None
    # (runtime-transient); resumed training would refit them from scratch,
    # the sanctioned "stats never migrate" disposition (§1.5).
    config = RecurrentCircuitConfig(**metadata["circuit_config"])
    circuit = RecurrentCellCircuit(
        token_cells=int(metadata["token_cells"]),
        config=config,
        cell_parameters=AllenGlif5Parameters(**metadata["cell_parameters"]),
    )
    for attribute, key, dtype in _persisted_circuit_bindings(
        config.readout_learning_rule
    ):
        existing = getattr(circuit, attribute)
        value = np.array(arrays[key], dtype=dtype, copy=True, order="C")
        if value.shape != existing.shape:
            raise ValueError(
                f"checkpoint array {key} has shape {value.shape}, "
                f"expected {existing.shape}"
            )
        setattr(circuit, attribute, value)
    circuit.active_reservoir_cells = int(metadata["runtime_active_reservoir_cells"])
    circuit._bind_native_kernel()
    circuit._growth_examples = int(metadata["growth_examples"])
    circuit._local_updates = int(metadata["local_updates"])
    circuit._learning_frozen = True
    return circuit


def _checkpoint_bytes(payload: _FrozenCheckpointPayload) -> bytes:
    return _canonical_json(
        {
            "format": _PROTOCOL_VERSION,
            "protocol": json.loads(payload.protocol_json.decode("utf-8")),
            "training_digest": payload.training_digest,
            "training_input_sha256": list(payload.training_input_sha256),
            "training_sequence_token_counts": list(payload.training_sequence_token_counts),
            "training_teacher_token_counts": list(payload.training_teacher_token_counts),
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
        and metadata["allen_glif5_config_sha256"] == HUMAN_GLIF5_IDENTITY.config_sha256
        and _canonical_json(metadata["cell_parameters"])
        == _canonical_json(_EXPECTED_CELL_PARAMETERS)
        and _canonical_json(metadata["circuit_config"]) == _canonical_json(_EXPECTED_CIRCUIT_CONFIG)
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
                "training_sequence_token_counts",
                "training_teacher_token_counts",
                "circuit",
            }
        ),
        "frozen checkpoint record",
    )
    if record["format"] != _PROTOCOL_VERSION:
        raise ValueError("unsupported frozen checkpoint format")
    protocol_record = record["protocol"]
    if not isinstance(protocol_record, Mapping):
        raise ValueError("frozen checkpoint protocol must be a record")
    lane = protocol_record.get("lane")
    if not isinstance(lane, str):
        raise ValueError("frozen checkpoint protocol lane must be text")
    try:
        protocol = Layer1Protocol.for_lane(lane)
    except ValueError as error:
        raise ValueError("frozen checkpoint protocol lane is unsupported") from error
    protocol_json = _canonical_json(protocol_record)
    if protocol_json != _protocol_json(protocol):
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
            f"frozen checkpoint permits at most {_MAX_TRAINING_EXAMPLES} training input digests"
        )
    input_hashes = tuple(_digest_text(item, "training input digest") for item in input_hash_values)
    sequence_token_count_values = _record_sequence(
        record["training_sequence_token_counts"],
        "training sequence token counts",
    )
    sequence_count = len(sequence_token_count_values)
    if sequence_count == 0:
        raise ValueError("frozen checkpoint needs at least one training sequence token count")
    if sequence_count > _MAX_TRAINING_EXAMPLES:
        raise ValueError(
            f"frozen checkpoint permits at most {_MAX_TRAINING_EXAMPLES} "
            "training sequence token counts"
        )
    if not all(type(item) is int for item in sequence_token_count_values):
        raise ValueError("training sequence token counts must be integers")
    sequence_token_counts = tuple(int(item) for item in sequence_token_count_values)
    if any(count < 2 for count in sequence_token_counts):
        raise ValueError("each training sequence must contain at least two tokens")
    if sequence_count != input_hash_count:
        raise ValueError("frozen checkpoint needs one count per training input digest")
    teacher_token_count_values = _record_sequence(
        record["training_teacher_token_counts"],
        "training teacher token counts",
    )
    teacher_count = len(teacher_token_count_values)
    if teacher_count == 0:
        raise ValueError("frozen checkpoint needs at least one training teacher token count")
    if teacher_count > _MAX_TRAINING_EXAMPLES:
        raise ValueError(
            f"frozen checkpoint permits at most {_MAX_TRAINING_EXAMPLES} "
            "training teacher token counts"
        )
    if not all(type(item) is int for item in teacher_token_count_values):
        raise ValueError("training teacher token counts must be integers")
    teacher_token_counts = tuple(int(item) for item in teacher_token_count_values)
    if any(count < 1 for count in teacher_token_counts):
        raise ValueError("each training teacher must contain at least one token")
    if teacher_count != input_hash_count:
        raise ValueError("frozen checkpoint needs one teacher count per training input digest")
    if any(
        teacher_count >= sequence_count
        for teacher_count, sequence_count in zip(
            teacher_token_counts, sequence_token_counts, strict=True
        )
    ):
        raise ValueError("training teacher count must be smaller than sequence count")
    circuit_record = _strict_record(
        record["circuit"],
        frozenset({"metadata", "arrays", "sha256"}),
        "frozen circuit record",
    )
    metadata = _validate_frozen_circuit_metadata(circuit_record["metadata"])
    # A2 supervises only teacher-forced program-token decisions.
    # A0 and A1 supervise every next-token transition in the framed sequence.
    if protocol.supervision_scope == _PROGRAM_ONLY_SUPERVISION:
        expected_growth_examples = _GROWTH_EPOCHS * sum(teacher_token_counts)
    else:
        expected_growth_examples = _GROWTH_EPOCHS * sum(
            count - 1 for count in sequence_token_counts
        )
    if metadata["growth_examples"] != expected_growth_examples:
        raise ValueError(
            "frozen checkpoint growth count differs from the lane's supervised "
            "transition count"
        )
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
        training_sequence_token_counts=sequence_token_counts,
        training_teacher_token_counts=teacher_token_counts,
        protocol_json=protocol_json,
    )


def grow(protocol: Layer1Protocol, examples: Sequence[Example]) -> FrozenCheckpoint:
    """Fit and freeze one Layer-1 circuit under its lane supervision contract."""
    if type(protocol) is not Layer1Protocol:
        raise TypeError("grow requires the fixed Layer1Protocol")
    if isinstance(examples, (str, bytes, bytearray)) or not isinstance(examples, Sequence):
        raise TypeError("Layer-1 growth examples must be an ordered sequence")
    corpus = tuple(islice(examples, protocol.max_training_examples + 1))
    if len(corpus) > protocol.max_training_examples:
        raise ValueError(
            f"Layer-1 growth accepts at most {protocol.max_training_examples} examples"
        )
    if not corpus:
        raise ValueError("at least one public Layer-1 growth example is required")
    if not all(isinstance(example, Example) for example in corpus):
        raise TypeError("Layer-1 growth examples must be Example values")

    circuit = RecurrentCellCircuit(token_cells=_TOKEN_CELL_COUNT, config=_CIRCUIT_CONFIG)
    digest = sha256()
    input_hashes: list[str] = []
    sequence_token_counts: list[int] = []
    teacher_token_counts: list[int] = []
    prepared_sequences: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for example in corpus:
        _validate_program(example.teacher_tokens, example.signature)
        input_ids = _model_input(
            example.prompt,
            example.signature,
            framing_order=protocol.framing_order,
        )
        input_bytes = _token_stream_bytes(input_ids)
        input_hashes.append(sha256(input_bytes).hexdigest())
        teacher_ids = _program_token_ids(example.teacher_tokens)
        training_ids = (*input_ids, *teacher_ids)
        sequence_token_counts.append(len(training_ids))
        teacher_token_counts.append(len(teacher_ids))
        prepared_sequences.append((input_ids, teacher_ids))
        digest.update(input_bytes)
        digest.update(_token_stream_bytes(teacher_ids))

    # Each epoch is one complete, ordered corpus pass. Every lane observes the
    # same input and emitted-prefix states. A0/A1 strengthen every transition;
    # A2 strengthens only teacher-forced program-token decisions.
    for _ in range(protocol.growth_epochs):
        for input_ids, teacher_ids in prepared_sequences:
            training_ids = (*input_ids, *teacher_ids)
            circuit.reset_sequence()
            for index, observed_token_id in enumerate(training_ids[:-1]):
                circuit.observe(observed_token_id)
                if protocol.supervision_scope == _ALL_TRANSITION_SUPERVISION or index + 1 >= len(
                    input_ids
                ):
                    circuit.strengthen(training_ids[index + 1])
    circuit.freeze_learning()
    payload = _FrozenCheckpointPayload(
        circuit=_freeze_circuit(circuit),
        training_digest=digest.hexdigest(),
        training_input_sha256=tuple(input_hashes),
        training_sequence_token_counts=tuple(sequence_token_counts),
        training_teacher_token_counts=tuple(teacher_token_counts),
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
        checkpoint.lane,
        checkpoint.sha256,
        checkpoint.training_digest,
        checkpoint.training_input_sha256,
        checkpoint.training_sequence_token_counts,
        checkpoint.training_teacher_token_counts,
        checkpoint.cell_synapse_sha256,
        checkpoint.protocol_sha256,
        checkpoint_bytes,
    )
    circuit = _restore_circuit(payload.circuit)
    before = circuit.capture_cell_synapse_state(
        include_ridge_statistics=False
    ).sha256
    if before != checkpoint.cell_synapse_sha256:
        raise ValueError("restored checkpoint cell/synapse digest mismatch")
    protocol_record = json.loads(payload.protocol_json.decode("utf-8"))
    input_ids = _model_input(
        prompt,
        typed_signature,
        framing_order=str(protocol_record["framing_order"]),
    )
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
        eligibility_bytes = circuit.eligibility.astype("<f8", copy=False).tobytes(order="C")
        output_current_pa = circuit.output_current_pa()
        allowed_output_currents_pa = tuple(
            float(output_current_pa[cell_id]) for cell_id in allowed_cell_ids
        )
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
                allowed_output_currents_pa=allowed_output_currents_pa,
                excluded_cell_ids=race.excluded_token_ids,
                output_current_sha256=race.output_current_sha256,
                eligibility_bytes=eligibility_bytes,
            )
        )
        circuit.observe(race.winner)
    if not state.complete:
        raise RuntimeError("cell program did not terminate within the Layer-1 bound")

    emitted = tuple(tokens)
    _validate_program(emitted, typed_signature)
    after = circuit.capture_cell_synapse_state(
        include_ridge_statistics=False
    ).sha256
    if after != before:
        raise PersistentStateChangedError("completion changed frozen cell/synapse state")
    if (
        checkpoint.lane,
        checkpoint.sha256,
        checkpoint.training_digest,
        checkpoint.training_input_sha256,
        checkpoint.training_sequence_token_counts,
        checkpoint.training_teacher_token_counts,
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
    if lowering == "subtract":
        return ast.BinOp(
            left=_lower_expression(expression.children[0], environment, context),
            op=ast.Sub(),
            right=_lower_expression(expression.children[1], environment, context),
        )
    if lowering == "absolute":
        return ast.Call(
            func=ast.Name(id="abs", ctx=ast.Load()),
            args=[_lower_expression(expression.children[0], environment, context)],
            keywords=[],
        )
    if lowering in {"minimum", "maximum"}:
        return ast.Call(
            func=ast.Name(
                id="min" if lowering == "minimum" else "max",
                ctx=ast.Load(),
            ),
            args=[
                _lower_expression(expression.children[0], environment, context),
                _lower_expression(expression.children[1], environment, context),
            ],
            keywords=[],
        )
    if lowering == "less-than":
        return ast.Compare(
            left=_lower_expression(expression.children[0], environment, context),
            ops=[ast.Lt()],
            comparators=[
                _lower_expression(expression.children[1], environment, context)
            ],
        )
    if lowering == "if":
        return ast.IfExp(
            test=_lower_expression(expression.children[0], environment, context),
            body=_lower_expression(expression.children[1], environment, context),
            orelse=_lower_expression(expression.children[2], environment, context),
        )
    if lowering == "sum":
        return ast.Call(
            func=ast.Name(id="sum", ctx=ast.Load()),
            args=[_lower_expression(expression.children[0], environment, context)],
            keywords=[],
        )
    if lowering == "count":
        binder, collection = expression.children
        if binder.spec.role != "binder" or len(binder.children) != 1:
            raise ProgramFormatError("count requires one explicit predicate binder")
        name = context.name("item")
        return ast.Call(
            func=ast.Name(id="sum", ctx=ast.Load()),
            args=[
                ast.GeneratorExp(
                    elt=ast.Constant(value=1),
                    generators=[
                        ast.comprehension(
                            target=ast.Name(id=name, ctx=ast.Store()),
                            iter=_lower_expression(collection, environment, context),
                            ifs=[
                                _lower_expression(
                                    binder.children[0],
                                    (name, *environment),
                                    context,
                                )
                            ],
                            is_async=0,
                        )
                    ],
                )
            ],
            keywords=[],
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


def compile_edit_site(program: Program, path: tuple[int, ...]) -> CompiledEditSite:
    """Compile one incumbent value site against its generic local scope."""

    if not isinstance(program, Program):
        raise TypeError("compile_edit_site requires a Program")
    if not isinstance(path, tuple) or not all(
        type(part) is int and part >= 0 for part in path
    ):
        raise TypeError("edit-site path must contain non-negative integers")
    if _program_sha256(program.tokens, program.signature) != program.sha256:
        raise ValueError("program digest mismatch")

    root = _validate_program(program.tokens, program.signature)
    records, substitutions = _typed_edit_site_records(root, program.signature)
    selected = next((record for record in records if record[1] == path), None)
    if selected is None:
        raise ValueError(f"program has no edit site at path {path!r}")

    expression, selected_path, expected, environment = selected
    result_type_label = _type_term_label(expected, substitutions)
    if expression.spec.role == "binder" or result_type_label == "internal-function":
        raise ProgramFormatError("binder/internal-function edit sites are non-value probes")

    scope_type_labels = tuple(
        _type_term_label(binding, substitutions) for binding in environment
    )
    scope_names = tuple(f"scope_{index}" for index in range(len(environment)))
    module = ast.Module(
        body=[
            ast.FunctionDef(
                name="edit_site",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=name) for name in scope_names],
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
                            scope_names,
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
    return CompiledEditSite(
        path=selected_path,
        result_type_label=result_type_label,
        scope_type_labels=scope_type_labels,
        source=ast.unparse(module) + "\n",
    )


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
    "CompiledEditSite",
    "CompiledPython",
    "Completion",
    "Example",
    "FactorAction",
    "FactorActionSite",
    "FactorCandidateBank",
    "FrozenCheckpoint",
    "Layer1Protocol",
    "PersistentStateChangedError",
    "Program",
    "ProgramEditSite",
    "ProgramFormatError",
    "ProgramParameter",
    "ProgramRaceAudit",
    "ProgramType",
    "TypedSignature",
    "apply_factor_action",
    "compile",
    "compile_edit_site",
    "complete",
    "factor_action_candidates",
    "factor_action_sites",
    "factor_actions_for_sites",
    "grow",
    "program_edit_sites",
    "replace_subtree",
]
