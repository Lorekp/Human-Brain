"""Prepared one-shot visible Simple-SWE probe for the native typed workspace.

One retained native temporal circuit learns scalar addition and list mapping by
absolute value through the typed workspace, then receives one teacher-free
list-map-add prompt. Importing or compiling this source does not spend the
pre-registered visible development task.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, REPO_ROOT, Path(__file__).parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import public_simple_swe_native_typed_sketch_probe as base
from humanbrain.cortex import seeded_threshold_offsets_mv
from humanbrain.native import loaded_library_identity
from humanbrain.native_temporal_runtime import (
    COPY_END_TOKEN_ID,
    SLOT_COUNT,
    NativeTemporalCircuit,
    NativeTemporalCircuitConfig,
    NativeTemporalCircuitDefinition,
    NativeTemporalRuntimeError,
    TemporalContextPolicy,
    TemporalGenerationPath,
    TemporalPlasticityPolicy,
    TemporalStepMode,
    TypedValue,
    TypedWorkspaceVocabulary,
    copy_slot_token_id,
    slot_end_token_id,
    slot_start_token_id,
)
from humanbrain.sdr_union import token_sdr_indices
from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.sandbox import (
    SandboxConfigurationError,
    execute_completion,
    load_worker_snapshot,
    python_runtime_identity,
    resolve_python_executable,
)
from humanbrain.simple_swe.tasks import GeneratedTask, generate_visible_partition
from humanbrain.typed_program import Program, ProgramType, compile as compile_typed_program


OUTPUT_PATH = Path(__file__).with_name(
    "public_simple_swe_native_typed_workspace_probe.json"
)
PROTOCOL_PATH = REPO_ROOT / "benchmarks" / "simple_swe_protocol_v1.json"
SCALAR_FAMILY = "t0_scalar_offset"
ABS_FAMILY = "t1_list_map_abs"
ADD_FAMILY = "t1_list_map_add"
TRAIN_KEYS = ((SCALAR_FAMILY, 0), (ABS_FAMILY, 0))
DEVELOPMENT_KEY = (ADD_FAMILY, 2)
DEVELOPMENT_EXPECTED_SOURCE = (
    "def solve_c01dcc26b43f0be3(values):\n"
    "    return [item_0 + -1 for item_0 in values]\n"
)

EXPECTED_SOURCE_RECORDS: dict[tuple[str, int], dict[str, str | None]] = {
    (SCALAR_FAMILY, 0): {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/000",
        "entry_point": "solve_d4c2e76cb6f6e7b8",
        "parameter": "n",
        "literal": "5",
        "task_sha256": "76a3a308651bb98a67801dbcbf7d2750a6de6f30748d6f500a30121dc950ea21",
    },
    (ABS_FAMILY, 0): {
        "task_id": "simple-swe-v1/train/tier-1/t1_list_map_abs/000",
        "entry_point": "solve_738987e19fc67a16",
        "parameter": "values",
        "literal": None,
        "task_sha256": "8f31b752aa2801a4a2f61ddd9d9e5a9de52d39b4e55fe960ae87d797d9073824",
    },
    (ADD_FAMILY, 2): {
        "task_id": "simple-swe-v1/train/tier-1/t1_list_map_add/002",
        "entry_point": "solve_c01dcc26b43f0be3",
        "parameter": "values",
        "literal": "-1",
        "task_sha256": "18a2f8e715e755bd3dd9209a4a777c8283388c4e63a92f7c51f278a4ae5e96a9",
    },
}

# Native token identities remain arbitrary.  The table is only a lossless
# adapter between the physical circuit and the existing typed_program grammar.
_PRIMITIVE_ORDER = (
    "add",
    "variable-0",
    "integer",
    "+",
    "-",
    "end",
    "map",
    "bind",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "absolute",
)
PRIMITIVE_TOKEN_IDS = {
    primitive: 1024 + ordinal for ordinal, primitive in enumerate(_PRIMITIVE_ORDER)
}
PRIMITIVE_BY_TOKEN_ID = {
    token_id: primitive for primitive, token_id in PRIMITIVE_TOKEN_IDS.items()
}

PLAN_COMMIT_TOKEN_ID = 2048
PLAN_COMPONENT_COUNT = 2
TYPED_CUE_NAMES = (
    "expression-int-head",
    "expression-int-child-0",
    "expression-int-child-1",
    "expression-list-int-head",
    "expression-list-int-child-0",
    "expression-list-int-child-1",
    "expression-int-to-int-head",
    "expression-int-to-int-child-0",
    "expression-int-to-int-child-1",
    "literal-sign",
    "literal-first-digit",
    "literal-first-nonzero-digit",
    "literal-digit-or-end",
    "literal-end",
    "eos",
    "operation-component-slot-absent",
    "operation-component-slot-ready",
)
TYPED_CUE_TOKEN_IDS = {
    name: PLAN_COMMIT_TOKEN_ID + ordinal + 1
    for ordinal, name in enumerate(TYPED_CUE_NAMES)
}
TYPED_SURFACE_ORDER = (
    "add",
    "absolute",
    "map",
    "bind",
    "variable-0",
    "integer",
    "+",
    "-",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "end",
)
TYPED_SURFACE_TOKEN_IDS = tuple(
    PRIMITIVE_TOKEN_IDS[primitive] for primitive in TYPED_SURFACE_ORDER
) + (base.EOS_TOKEN_ID,)
TYPED_PRIVATE_TOKEN_IDS = (
    PLAN_COMMIT_TOKEN_ID,
    *TYPED_CUE_TOKEN_IDS.values(),
)
_RESERVED_CODE_COUNT = len(TYPED_PRIVATE_TOKEN_IDS)
_RESERVED_COLUMN_COUNT = _RESERVED_CODE_COUNT * base.ACTIVE_COLUMNS
TYPED_WORKSPACE_VOCABULARY = TypedWorkspaceVocabulary(
    plan_commit_token_id=PLAN_COMMIT_TOKEN_ID,
    cue_token_ids=tuple(TYPED_CUE_TOKEN_IDS.values()),
    surface_token_ids=TYPED_SURFACE_TOKEN_IDS,
)

# Reuse the already-attested native definition/generation helpers with this
# language inventory.  No runtime or learning behavior is replaced here.
base.PRIMITIVE_TOKEN_IDS = PRIMITIVE_TOKEN_IDS
base.PRIMITIVE_BY_TOKEN_ID = PRIMITIVE_BY_TOKEN_ID


def _typed_value(program_type: object) -> TypedValue:
    if type(program_type) is not ProgramType:
        raise TypeError("typed workspace requires an exact ProgramType")
    if program_type == ProgramType.Int():
        return TypedValue.INT
    if program_type == ProgramType.List(ProgramType.Int()):
        return TypedValue.LIST_INT
    raise RuntimeError("typed workspace supports only Int and List[Int] public values")


@dataclass(frozen=True, slots=True)
class _WorkspaceEpisode:
    prompt_steps: tuple[object, ...]
    workspace_steps: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _WorkspaceTrace:
    generations: tuple[object, ...]


class _TypedWorkspaceApi:
    """The sole adapter to the native atomic typed-workspace API."""

    @staticmethod
    def train_task(
        circuit: NativeTemporalCircuit,
        *,
        prompt_tokens: tuple[int, ...],
        parameter_type: object,
        result_type: object,
        teacher_tokens: tuple[int, ...],
    ) -> _WorkspaceEpisode:
        prompt_steps = circuit.advance_block(
            prompt_tokens,
            mode=TemporalStepMode.LEARN,
            reset_before=True,
        )
        begin_step = circuit.begin_typed_session(
            TYPED_WORKSPACE_VOCABULARY,
            parameter_type=_typed_value(parameter_type),
            result_type=_typed_value(result_type),
            mode=TemporalStepMode.LEARN,
        )
        teacher_steps = tuple(
            circuit.teach_typed_token(token_id) for token_id in teacher_tokens
        )
        return _WorkspaceEpisode(
            prompt_steps=prompt_steps,
            workspace_steps=(begin_step, *teacher_steps),
        )

    @staticmethod
    def begin_development(
        circuit: NativeTemporalCircuit,
        *,
        prompt_tokens: tuple[int, ...],
        parameter_type: object,
        result_type: object,
    ) -> _WorkspaceEpisode:
        prompt_steps = circuit.advance_block(
            prompt_tokens,
            mode=TemporalStepMode.READ_ONLY,
            reset_before=True,
        )
        begin_step = circuit.begin_typed_session(
            TYPED_WORKSPACE_VOCABULARY,
            parameter_type=_typed_value(parameter_type),
            result_type=_typed_value(result_type),
            mode=TemporalStepMode.READ_ONLY,
        )
        return _WorkspaceEpisode(
            prompt_steps=prompt_steps,
            workspace_steps=(begin_step,),
        )

    @staticmethod
    def generate(circuit: NativeTemporalCircuit) -> _WorkspaceTrace:
        return _WorkspaceTrace(generations=circuit.generate_typed())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--python-executable", type=Path, required=True)
    return parser.parse_args()


def _select_and_confirm_tasks() -> tuple[
    object, tuple[GeneratedTask, ...], GeneratedTask
]:
    protocol = load_protocol(PROTOCOL_PATH)
    visible_train = generate_visible_partition(protocol, "train")
    selected: dict[tuple[str, int], GeneratedTask] = {}
    for key in (*TRAIN_KEYS, DEVELOPMENT_KEY):
        family, ordinal = key
        task = next(
            item
            for item in visible_train
            if item.public.family == family and item.public.ordinal == ordinal
        )
        expected = EXPECTED_SOURCE_RECORDS[key]
        parameters = task.public.signature.parameters
        observed = {
            "task_id": task.public.task_id,
            "entry_point": task.public.signature.entry_point,
            "parameter": parameters[0].name if len(parameters) == 1 else None,
            "task_sha256": task.public.task_sha256,
        }
        expected_source = {
            name: expected[name]
            for name in ("task_id", "entry_point", "parameter", "task_sha256")
        }
        if observed != expected_source:
            raise RuntimeError(f"visible task source record changed for {key!r}: {observed!r}")
        literal = expected["literal"]
        if family == SCALAR_FAMILY:
            phrase = f"fixed offset {literal}."
        elif family == ADD_FAMILY:
            phrase = f"with {literal} added to every integer"
        else:
            phrase = "absolute value of each integer"
        if phrase not in task.public.prompt:
            raise RuntimeError(f"visible task wording changed for {key!r}")
        if len(task.private.regression_cases) != 4 or len(task.private.property_cases) != 32:
            raise RuntimeError(f"visible sandbox case counts changed for {key!r}")
        selected[key] = task
    return protocol, tuple(selected[key] for key in TRAIN_KEYS), selected[DEVELOPMENT_KEY]


def _task_fields(task: GeneratedTask) -> tuple[bytes, bytes, bytes]:
    expected = EXPECTED_SOURCE_RECORDS[(task.public.family, task.public.ordinal)]
    literal = expected["literal"]
    return (
        task.public.signature.entry_point.encode("ascii"),
        task.public.signature.parameters[0].name.encode("ascii"),
        b"" if literal is None else str(literal).encode("ascii"),
    )


def _canonical_literal_primitives(literal: bytes) -> tuple[str, ...]:
    text = literal.decode("ascii")
    signed = text if text.startswith("-") else f"+{text}"
    if signed in {"+0", "-0"} or not signed[1:].isdigit():
        raise RuntimeError(f"non-canonical visible literal {text!r}")
    return tuple(signed)


def _prompt_spans(
    task: GeneratedTask,
) -> tuple[bytes, tuple[tuple[int, int, int, tuple[int, ...]], ...]]:
    prompt = task.public.prompt.encode("ascii")
    entry_point, parameter, literal = _task_fields(task)
    signature = b"def " + entry_point + b"(" + parameter + b"):"
    signature_start = prompt.index(signature)
    entry_start = signature_start + len(b"def ")
    parameter_start = entry_start + len(entry_point) + 1
    spans: list[tuple[int, int, int, tuple[int, ...]]] = [
        (entry_start, entry_start + len(entry_point), 0, tuple(entry_point)),
        (parameter_start, parameter_start + len(parameter), 1, tuple(parameter)),
    ]
    if literal:
        anchor = b"fixed offset " if task.public.family == SCALAR_FAMILY else b"with "
        literal_start = prompt.index(anchor, parameter_start) + len(anchor)
        if prompt[literal_start : literal_start + len(literal)] != literal:
            raise RuntimeError("visible literal span no longer matches")
        canonical_ids = tuple(
            PRIMITIVE_TOKEN_IDS[token]
            for token in _canonical_literal_primitives(literal)
        )
        spans.append((literal_start, literal_start + len(literal), 2, canonical_ids))
    return prompt, tuple(spans)


def _detokenize_prompt(tokens: tuple[int, ...]) -> bytes:
    starts = {slot_start_token_id(slot): slot for slot in range(SLOT_COUNT)}
    ends = {slot_end_token_id(slot): slot for slot in range(SLOT_COUNT)}
    output = bytearray()
    active_slot: int | None = None
    literal_parts: list[str] = []
    for token_id in tokens:
        if token_id in starts:
            if active_slot is not None:
                raise RuntimeError("nested prompt slots are forbidden")
            active_slot = starts[token_id]
            literal_parts = []
            continue
        if token_id in ends:
            if active_slot != ends[token_id]:
                raise RuntimeError("mismatched prompt slot boundary")
            if active_slot == 2:
                signed = "".join(literal_parts)
                raw = signed[1:] if signed.startswith("+") else signed
                output.extend(raw.encode("ascii"))
            active_slot = None
            continue
        if active_slot == 2:
            try:
                literal_parts.append(PRIMITIVE_BY_TOKEN_ID[token_id])
            except KeyError as error:
                raise RuntimeError("non-primitive token in literal slot") from error
        else:
            if not 0 <= token_id < 128:
                raise RuntimeError("non-ASCII token outside canonical literal slot")
            output.append(token_id)
    if active_slot is not None:
        raise RuntimeError("unterminated prompt slot")
    return bytes(output)


def _primitive_id(token: str) -> int:
    return PRIMITIVE_TOKEN_IDS[token]


def _internal_target(family: str) -> tuple[int, ...]:
    if family == SCALAR_FAMILY:
        primitives: tuple[str | None, ...] = (
            "add", "variable-0", "integer", None, "end"
        )
    elif family == ABS_FAMILY:
        primitives = ("map", "bind", "absolute", "variable-0", "variable-0")
    elif family == ADD_FAMILY:
        primitives = (
            "map", "bind", "add", "variable-0", "integer", None, "end", "variable-0"
        )
    else:
        raise RuntimeError("unsupported typed-workspace family")
    return (
        *(
            copy_slot_token_id(2) if primitive is None else _primitive_id(primitive)
            for primitive in primitives
        ),
        base.EOS_TOKEN_ID,
    )


def _expected_program_tokens(task: GeneratedTask) -> tuple[str, ...]:
    literal = _task_fields(task)[2]
    if task.public.family == SCALAR_FAMILY:
        return (
            "add", "variable-0", "integer", *_canonical_literal_primitives(literal), "end"
        )
    if task.public.family == ABS_FAMILY:
        return ("map", "bind", "absolute", "variable-0", "variable-0")
    if task.public.family == ADD_FAMILY:
        return (
            "map", "bind", "add", "variable-0", "integer",
            *_canonical_literal_primitives(literal), "end", "variable-0",
        )
    raise RuntimeError("unsupported typed-workspace family")


def _expected_compiled_source(task: GeneratedTask) -> str:
    entry_point, parameter, literal = (
        value.decode("ascii") for value in _task_fields(task)
    )
    if task.public.family == SCALAR_FAMILY:
        return f"def {entry_point}({parameter}):\n    return {parameter} + {literal}\n"
    if task.public.family == ABS_FAMILY:
        return (
            f"def {entry_point}({parameter}):\n"
            f"    return [abs(item_0) for item_0 in {parameter}]\n"
        )
    if task.public.family == ADD_FAMILY:
        source = (
            f"def {entry_point}({parameter}):\n"
            f"    return [item_0 + {literal} for item_0 in {parameter}]\n"
        )
        if (task.public.family, task.public.ordinal) == DEVELOPMENT_KEY:
            if source != DEVELOPMENT_EXPECTED_SOURCE:
                raise RuntimeError("pre-registered development source changed")
            return DEVELOPMENT_EXPECTED_SOURCE
        return source
    raise RuntimeError("unsupported typed-workspace family")


# Patch only the data adapters used by the shared physical harness helpers.
base._task_fields = _task_fields
base._canonical_literal_primitives = _canonical_literal_primitives
base._prompt_spans = _prompt_spans
base._detokenize_prompt = _detokenize_prompt
base._internal_target = _internal_target
base._expected_program_tokens = _expected_program_tokens
base._expected_compiled_source = _expected_compiled_source


def _definition(
    train_tasks: tuple[GeneratedTask, ...],
    development_task: GeneratedTask,
) -> NativeTemporalCircuitDefinition:
    """Build one codebook with workspace controls isolated by minicolumn."""

    train_prompts = tuple(base._tokenized_prompt(task) for task in train_tasks)
    development_prompt = base._tokenized_prompt(development_task)
    train_targets = tuple(
        _internal_target(task.public.family) for task in train_tasks
    )
    control_tokens = {
        *(slot_start_token_id(slot) for slot in range(SLOT_COUNT)),
        *(slot_end_token_id(slot) for slot in range(SLOT_COUNT)),
        *(copy_slot_token_id(slot) for slot in range(SLOT_COUNT)),
        COPY_END_TOKEN_ID,
    }
    used_token_ids = tuple(
        sorted(
            {
                *(token for prompt in train_prompts for token in prompt),
                *development_prompt,
                *PRIMITIVE_BY_TOKEN_ID,
                *control_tokens,
                *TYPED_PRIVATE_TOKEN_IDS,
                base.EOS_TOKEN_ID,
            }
        )
    )

    # Structural cues are read-only. Only prompt, two plan-component commits,
    # and teacher arrivals can allocate, so this is the tight capacity bound.
    train_learning_arrivals = tuple(
        len(prompt) + PLAN_COMPONENT_COUNT + len(target)
        for prompt, target in zip(train_prompts, train_targets, strict=True)
    )
    segment_capacity = (
        sum(arrivals - 1 for arrivals in train_learning_arrivals)
        * base.ACTIVE_COLUMNS
    )
    base_population = base.COLUMN_COUNT * base.BASE_CELLS_PER_COLUMN
    config = NativeTemporalCircuitConfig(
        column_count=base.COLUMN_COUNT,
        max_pyramidal_cells=base_population + segment_capacity,
        segment_capacity=segment_capacity,
        construction_seed=base.CONSTRUCTION_SEED,
        connected_permanence_ticks=250,
        initial_permanence_ticks=255,
        permanence_increment_ticks=50,
        negative_decrement_ticks=2,
        base_cells_per_column=base.BASE_CELLS_PER_COLUMN,
        novel_race_cap=base.BASE_CELLS_PER_COLUMN,
        prediction_quorum=base.ACTIVE_COLUMNS,
        max_segments_per_cell=64,
        theta_match=16,
        theta_active=18,
        context_policy=TemporalContextPolicy.BOUND_WINNERS,
        plasticity_policy=TemporalPlasticityPolicy.PLASTIC,
    )

    remaining_column_count = base.COLUMN_COUNT - _RESERVED_COLUMN_COUNT
    if remaining_column_count < base.ACTIVE_COLUMNS:
        raise RuntimeError("typed workspace reserved too many minicolumns")

    def token_code(token_id: int) -> tuple[int, ...]:
        if token_id in TYPED_PRIVATE_TOKEN_IDS:
            private_ordinal = TYPED_PRIVATE_TOKEN_IDS.index(token_id)
            start = base.ACTIVE_COLUMNS * private_ordinal
            return tuple(range(start, start + base.ACTIVE_COLUMNS))
        return tuple(
            _RESERVED_COLUMN_COUNT + int(column)
            for column in token_sdr_indices(
                token_id,
                n_bits=remaining_column_count,
                w_ones=base.ACTIVE_COLUMNS,
                seed=base.CODEBOOK_SEED,
            )
        )

    token_codes = {token_id: token_code(token_id) for token_id in used_token_ids}
    reserved_codes = {
        token_id: token_codes[token_id]
        for token_id in TYPED_PRIVATE_TOKEN_IDS
    }
    if len({column for code in reserved_codes.values() for column in code}) != (
        _RESERVED_COLUMN_COUNT
    ):
        raise RuntimeError("typed workspace reserved code ranges overlap")
    if any(
        column < _RESERVED_COLUMN_COUNT
        for token_id, code in token_codes.items()
        if token_id not in reserved_codes
        for column in code
    ):
        raise RuntimeError("prompt/surface code entered the workspace reserve")

    physical_cell_count = 2 * config.max_pyramidal_cells + 3 * config.column_count
    return NativeTemporalCircuitDefinition(
        config,
        token_codes,
        seeded_threshold_offsets_mv(
            physical_cell_count,
            config.construction_seed,
        ).tolist(),
    )


def _generation_failure(error: NativeTemporalRuntimeError, position: int) -> dict[str, object]:
    return {
        "kind": "model",
        "gate": "GENERATION_STOP",
        "detail": {
            "generation_position": position,
            "status": error.status.name,
            "message": error.message,
            "required": error.required,
            "observed": error.observed,
            "location": error.location,
            "poison_code": error.poison_code,
            "stage_code": error.stage_code,
        },
    }


def _copy_transaction(
    generations: list[dict[str, object]], expected_tokens: tuple[str, ...]
) -> tuple[dict[str, object] | None, bool]:
    """Score slot-two copy while accounting for one typed cue per hole."""

    transactions: list[dict[str, object]] = []
    active: dict[str, object] | None = None
    exact = True
    previous_event_count: int | None = None
    for generation in generations:
        path = str(generation["path"])
        event_count = int(generation["temporal_event_count"])
        if path == TemporalGenerationPath.ENTER_COPY.name:
            exact &= active is None and previous_event_count is not None
            enter_delta = (
                None
                if previous_event_count is None
                else event_count - previous_event_count
            )
            active = {
                "slot_ordinal": generation["slot_ordinal"],
                "control_token_id": generation["control_token_id"],
                "enter_event_count": event_count,
                "enter_event_count_delta": enter_delta,
                "enter_advanced_temporal_after_typed_cue": (
                    enter_delta is not None and enter_delta > 0
                ),
                "copied_places": [generation["copy_place_ordinal"]],
                "copied_primitives": [generation["surface_primitive"]],
                "copied_event_counts": [event_count],
                "copied_physical_election_counts": [
                    generation["physical_election_count"]
                ],
                "copied_candidate_counts": [generation["candidate_tokens_physical"]],
                "exit_event_count": None,
                "exit_event_count_delta": None,
                "exit_surface_primitive": None,
            }
        elif path == TemporalGenerationPath.COPY.name:
            exact &= active is not None
            if active is not None:
                exact &= generation["slot_ordinal"] == active["slot_ordinal"]
                active["copied_places"].append(generation["copy_place_ordinal"])
                active["copied_primitives"].append(generation["surface_primitive"])
                active["copied_event_counts"].append(event_count)
                active["copied_physical_election_counts"].append(
                    generation["physical_election_count"]
                )
                active["copied_candidate_counts"].append(
                    generation["candidate_tokens_physical"]
                )
        elif path == TemporalGenerationPath.EXIT_COPY.name:
            exact &= active is not None
            if active is not None:
                exact &= generation["slot_ordinal"] == active["slot_ordinal"]
                enter_count = int(active["enter_event_count"])
                exit_delta = event_count - enter_count
                active["exit_event_count"] = event_count
                active["exit_event_count_delta"] = exit_delta
                active["exit_surface_primitive"] = generation["surface_primitive"]
                active["temporal_event_count_held_across_copied_tokens"] = all(
                    int(value) == enter_count
                    for value in active["copied_event_counts"]
                )
                active["exit_resumed_temporal_after_typed_cue"] = exit_delta > 0
                transactions.append(active)
                active = None
        previous_event_count = event_count

    exact &= active is None and len(transactions) == 1
    transaction = transactions[0] if len(transactions) == 1 else None
    if transaction is not None:
        exact &= transaction["slot_ordinal"] == 2
        exact &= transaction["control_token_id"] == copy_slot_token_id(2)
        exact &= transaction["copied_places"] == list(range(len(expected_tokens)))
        exact &= transaction["copied_primitives"] == list(expected_tokens)
        exact &= transaction["exit_surface_primitive"] == "end"
        exact &= bool(transaction["enter_advanced_temporal_after_typed_cue"])
        exact &= bool(transaction["temporal_event_count_held_across_copied_tokens"])
        exact &= bool(transaction["exit_resumed_temporal_after_typed_cue"])
        exact &= all(
            int(value) >= 1
            for value in transaction["copied_physical_election_counts"]
        )
        exact &= all(
            int(value) >= 1 for value in transaction["copied_candidate_counts"]
        )
    return transaction, exact


def _development_once(
    *,
    circuit: NativeTemporalCircuit,
    task: GeneratedTask,
    protocol: object,
    python_executable: Path,
    runtime_identity: dict[str, str],
    worker_source: bytes,
    worker_sha256: str,
) -> tuple[dict[str, object], dict[str, object] | None]:
    signature = task.public.signature
    episode = _TypedWorkspaceApi.begin_development(
        circuit,
        prompt_tokens=base._tokenized_prompt(task),
        parameter_type=signature.parameters[0].type,
        result_type=signature.result,
    )
    expected_tokens = _expected_program_tokens(task)
    expected_surface_ids = tuple(
        PRIMITIVE_TOKEN_IDS[token] for token in expected_tokens
    ) + (base.EOS_TOKEN_ID,)
    generations: list[dict[str, object]] = []
    emitted_tokens: list[str] = []
    first_failure: dict[str, object] | None = None
    terminated_by_eos = False

    # Native generation is one no-argument atomic run. The expected stream is
    # host-side scoring data and never feeds back into that immutable trace.
    workspace_trace: _WorkspaceTrace | None = None
    try:
        workspace_trace = _TypedWorkspaceApi.generate(circuit)
    except NativeTemporalRuntimeError as error:
        first_failure = _generation_failure(error, 0)
    else:
        for position, generation in enumerate(workspace_trace.generations):
            generation_record = base._generation_record(position, generation)
            generations.append(generation_record)

        observed_surface_ids = tuple(
            int(generation.surface_token_id)
            for generation in workspace_trace.generations
        )
        for position in range(max(len(expected_surface_ids), len(observed_surface_ids))):
            expected_token_id = (
                expected_surface_ids[position]
                if position < len(expected_surface_ids)
                else None
            )
            observed_token_id = (
                observed_surface_ids[position]
                if position < len(observed_surface_ids)
                else None
            )
            if observed_token_id != expected_token_id:
                first_failure = {
                    "kind": "model",
                    "gate": "EXACT_TYPED_TOKEN_STREAM",
                    "detail": {
                        "first_divergence_position": position,
                        "expected_surface_token_id": expected_token_id,
                        "expected_surface_primitive": (
                            "<END_OF_TRACE>"
                            if expected_token_id is None
                            else "<EOS>"
                            if expected_token_id == base.EOS_TOKEN_ID
                            else PRIMITIVE_BY_TOKEN_ID[expected_token_id]
                        ),
                        "observed_surface_token_id": observed_token_id,
                        "observed_surface_primitive": (
                            "<MISSING>"
                            if observed_token_id is None
                            else "<EOS>"
                            if observed_token_id == base.EOS_TOKEN_ID
                            else PRIMITIVE_BY_TOKEN_ID.get(observed_token_id)
                        ),
                    },
                }
                break
        terminated_by_eos = bool(
            observed_surface_ids
            and observed_surface_ids[-1] == base.EOS_TOKEN_ID
            and generations[-1]["path"] == TemporalGenerationPath.TEMPORAL.name
        )
        for token_id in observed_surface_ids:
            if token_id == base.EOS_TOKEN_ID:
                break
            primitive = PRIMITIVE_BY_TOKEN_ID.get(token_id)
            if primitive is None:
                break
            emitted_tokens.append(primitive)

    literal_tokens = _canonical_literal_primitives(_task_fields(task)[2])
    transaction, copy_trace_exact = _copy_transaction(generations, literal_tokens)
    fitted_cell_races_exact = all(
        int(generation["physical_election_count"]) >= 1
        and int(generation["candidate_tokens_physical"]) >= 1
        and int(generation["winner_input_columns"]) == base.ACTIVE_COLUMNS
        for generation in generations
    )
    if first_failure is None and not terminated_by_eos:
        first_failure = {"kind": "model", "gate": "ORDINARY_EOS", "detail": None}
    elif first_failure is None and not copy_trace_exact:
        first_failure = {
            "kind": "model", "gate": "SLOT_2_PHYSICAL_COPY", "detail": transaction
        }
    elif first_failure is None and not fitted_cell_races_exact:
        first_failure = {"kind": "model", "gate": "FITTED_CELL_RACES", "detail": None}

    emitted = tuple(emitted_tokens)
    program_record: dict[str, object] = {
        "attempted": False,
        "tokens": list(emitted),
        "expected_tokens": list(expected_tokens),
        "exact_tokens": emitted == expected_tokens,
        "program_sha256": None,
        "compiled_source": None,
        "expected_compiled_source": _expected_compiled_source(task),
        "exact_compiled_source": False,
        "compiler_received_prompt": False,
        "compiler_received_tests": False,
    }
    compiled_source: str | None = None
    if first_failure is None:
        try:
            program = Program(tokens=emitted, signature=task.public.signature)
            compiled_source = compile_typed_program(program).source
        except (TypeError, ValueError) as error:
            first_failure = {
                "kind": "model",
                "gate": "EXISTING_TYPED_PROGRAM_COMPILE",
                "detail": f"{type(error).__name__}: {error}",
            }
        else:
            program_record.update(
                {
                    "attempted": True,
                    "program_sha256": program.sha256,
                    "compiled_source": compiled_source,
                    "exact_compiled_source": (
                        compiled_source == program_record["expected_compiled_source"]
                    ),
                }
            )
            if not program_record["exact_compiled_source"]:
                first_failure = {
                    "kind": "model", "gate": "EXACT_SIGNATURE_PYTHON", "detail": None
                }

    sandbox_record: dict[str, object] = {
        "attempted": False,
        "outcome": None,
        "passed": False,
        "sandboxed": False,
        "detail": "not run because an earlier gate stopped",
        "wall_seconds": 0.0,
    }
    if first_failure is None and compiled_source is not None:
        try:
            sandbox_result = execute_completion(
                compiled_source,
                task.public,
                task.private,
                protocol.execution,
                python_executable=python_executable,
                python_executable_sha256=runtime_identity["executable_sha256"],
                worker_source=worker_source,
                worker_sha256=worker_sha256,
            )
        except SandboxConfigurationError as error:
            first_failure = {
                "kind": "harness",
                "gate": "PUBLIC_SIMPLE_SWE_SANDBOX_CONFIGURATION",
                "detail": str(error),
            }
            sandbox_record["detail"] = str(error)
        else:
            sandbox_record = {
                "attempted": True,
                **asdict(sandbox_result),
                "regression_case_count": len(task.private.regression_cases),
                "property_case_count": len(task.private.property_cases),
                "worker_sha256": worker_sha256,
                "python_runtime": runtime_identity,
            }
            if not sandbox_result.passed:
                first_failure = {
                    "kind": "model",
                    "gate": "PUBLIC_SIMPLE_SWE_SANDBOX",
                    "detail": sandbox_result.outcome,
                }

    return (
        {
            "task": base._task_record(task),
            "attempt_number": 1,
            "prompt_steps": len(episode.prompt_steps),
            "workspace_steps_before_generation": len(episode.workspace_steps),
            "workspace_step_token_ids": [
                step.token_id for step in episode.workspace_steps
            ],
            "workspace_step_event_counts": [
                step.event_count for step in episode.workspace_steps
            ],
            "generation_call_count": 1,
            "atomic_no_argument_generation": True,
            "composed_plan_query_exposed_by_transport": False,
            "teacher_tokens_after_prompt": 0,
            "target_entered_runtime": False,
            "expected_stream_used_only_for_host_scoring": True,
            "completed_generations": len(generations),
            "terminated_by_ordinary_eos": terminated_by_eos,
            "program": program_record,
            "slot_2_copy_transaction": transaction,
            "slot_2_copy_trace_exact": copy_trace_exact,
            "fitted_cell_races_exact": fitted_cell_races_exact,
            "sandbox": sandbox_record,
            "first_failure": first_failure,
            "generations": generations,
        },
        first_failure,
    )


def _compiler_check(task: GeneratedTask) -> dict[str, object]:
    tokens = _expected_program_tokens(task)
    source = compile_typed_program(Program(tokens, task.public.signature)).source
    expected = _expected_compiled_source(task)
    if source != expected:
        raise RuntimeError(f"typed compiler source mismatch for {task.public.task_id}")
    return {
        "task_id": task.public.task_id,
        "tokens": list(tokens),
        "compiled_source": source,
        "exact": True,
    }


def main() -> int:
    args = _parse_args()
    started_ns = time.perf_counter_ns()
    protocol, train_tasks, development_task = _select_and_confirm_tasks()
    compiler_checks = [_compiler_check(task) for task in train_tasks]
    definition = _definition(train_tasks, development_task)

    executable = resolve_python_executable(args.python_executable)
    runtime_identity = python_runtime_identity(executable, protocol.raw.get("language"))
    if runtime_identity["python_version"] != "3.11.15":
        raise RuntimeError("typed-workspace probe requires explicit repo Python 3.11.15")
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = hashlib.sha256(worker_path.read_bytes()).hexdigest()
    worker_source = load_worker_snapshot(worker_sha256)

    train_records: list[dict[str, object]] = []
    with base._public_development_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            for task in train_tasks:
                signature = task.public.signature
                episode = _TypedWorkspaceApi.train_task(
                    circuit,
                    prompt_tokens=base._tokenized_prompt(task),
                    parameter_type=signature.parameters[0].type,
                    result_type=signature.result,
                    teacher_tokens=_internal_target(task.public.family),
                )
                steps = (*episode.prompt_steps, *episode.workspace_steps)
                train_records.append(
                    {
                        **base._task_record(task),
                        "steps": len(steps),
                        "prompt_steps": len(episode.prompt_steps),
                        "workspace_steps": len(episode.workspace_steps),
                        "allocated_segments": sum(step.allocated_segments for step in steps),
                        "workspace_step_token_ids": [
                            step.token_id for step in episode.workspace_steps
                        ],
                        "workspace_step_event_counts": [
                            step.event_count for step in episode.workspace_steps
                        ],
                        "internal_target": [
                            "COPY_SLOT_2" if token == copy_slot_token_id(2)
                            else "<EOS>" if token == base.EOS_TOKEN_ID
                            else PRIMITIVE_BY_TOKEN_ID[token]
                            for token in _internal_target(task.public.family)
                        ],
                    }
                )
            trained_identity = circuit.checkpoint_identity()
            development_record, first_failure = _development_once(
                circuit=circuit,
                task=development_task,
                protocol=protocol,
                python_executable=executable,
                runtime_identity=runtime_identity,
                worker_source=worker_source,
                worker_sha256=worker_sha256,
            )
        native_identity = loaded_library_identity()

    passed = first_failure is None
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-native-typed-workspace-composition",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": (
            "PASS_NATIVE_TYPED_WORKSPACE_COMPOSITION"
            if passed else "STOP_NATIVE_TYPED_WORKSPACE_COMPOSITION"
        ),
        "capability_pass": passed,
        "first_failure": first_failure,
        "claim_boundary": (
            "One retained native temporal circuit sees one visible scalar-offset task "
            "and one visible list-map-absolute task, then attempts visible list-map-add "
            "ordinal 002 exactly once. A pass is direct visible-development evidence that "
            "native result-type and slot-state cues acquired independent retained scaffold "
            "and operation components, then structurally projected their splice at typed "
            "holes so the "
            "circuit recombined an outer map/binder subtree with an inner add/literal "
            "subtree and physically copied the unseen -1 surface literal. It is not M0, "
            "family-disjoint generalization, eval evidence, broad coding competence, or C1."
        ),
        "authority": {
            "source_partition_generated": "train",
            "eval_partition_generated": False,
            "train_keys": [list(key) for key in TRAIN_KEYS],
            "development_key": list(DEVELOPMENT_KEY),
            "development_attempts_completed": 1,
            "model_retries": 0,
            "harness_retries": 0,
            "scoring_reports_first_model_divergence": True,
            "compile_and_sandbox_skipped_after_first_divergence": True,
            "atomic_generation_received_python_feedback": False,
            "development_target_used_as_runtime_input": False,
            "teacher_tokens_after_development_prompt": 0,
            "prompt_family_ids_injected": False,
            "prompt_solution_labels_injected": False,
            "prompt_bracket_scope": "signature-entry/signature-parameter/literal-when-present",
            "python_candidate_enumeration": False,
            "python_argmax": False,
            "learned_readout_parameters": 0,
            "typed_compiler_received_prompt": False,
            "typed_compiler_received_tests": False,
            "synthetic_runtime_authority": True,
        },
        "protocol": {
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.sha256,
            "language": protocol.raw.get("language"),
        },
        "typed_subtrees": {
            "scalar_train": "add variable-0 integer COPY[2] end",
            "map_absolute_train": "map bind absolute variable-0 variable-0",
            "map_add_development_expected": (
                "map bind add variable-0 integer COPY[2] end variable-0"
            ),
            "development_stream_equals_a_train_stream": False,
            "absolute_primitive_type": "Int -> Int",
            "compiler_checks": compiler_checks,
        },
        "slot_bank": {
            "slot_count": SLOT_COUNT,
            "copied_slot_ordinal": 2,
            "max_surface_tokens_per_slot": 22,
            "copy_control_token_id": copy_slot_token_id(2),
            "copy_end_token_id": COPY_END_TOKEN_ID,
        },
        "physical_definition": {
            "circuit_identity_sha256": definition.circuit_identity_sha256(),
            "column_count": definition.config.column_count,
            "active_columns_per_token": base.ACTIVE_COLUMNS,
            "used_token_count": len(definition.token_codes),
            "physical_cell_count": len(definition.threshold_offsets_mv),
            "segment_capacity": definition.config.segment_capacity,
            "theta_match": definition.config.theta_match,
            "theta_active": definition.config.theta_active,
            "plan_commit_token_id": PLAN_COMMIT_TOKEN_ID,
            "plan_component_count": PLAN_COMPONENT_COUNT,
            "typed_cue_token_ids": TYPED_CUE_TOKEN_IDS,
            "typed_private_token_ids": list(TYPED_PRIVATE_TOKEN_IDS),
            "typed_surface_token_ids": list(TYPED_SURFACE_TOKEN_IDS),
            "workspace_reserved_column_count": _RESERVED_COLUMN_COUNT,
            "prompt_surface_column_count": (
                base.COLUMN_COUNT - _RESERVED_COLUMN_COUNT
            ),
            "workspace_codes_disjoint_from_prompt_and_surface": True,
        },
        "train": {
            "tasks": train_records,
            "reset_before_each": True,
            "learned_state_retained_between_tasks": True,
            "live_segments_after_train": trained_identity.segment_count,
            "learning_steps_after_train": trained_identity.learning_step,
        },
        "visible_development_teacher_free": development_record,
        "public_simple_swe_runtime": {
            "explicit_repo_venv": str(executable),
            "python_runtime": runtime_identity,
            "worker_sha256": worker_sha256,
            "regression_cases": 4,
            "property_cases": 32,
        },
        "runtime": {
            "elapsed_wall_ns": time.perf_counter_ns() - started_ns,
            "python": sys.version,
            "platform": platform.platform(),
            "native_source_digest": native_identity.source_digest,
            "native_binary_sha256": native_identity.binary_sha256,
            "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "outcome": payload["outcome"],
                "first_failure": first_failure,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
