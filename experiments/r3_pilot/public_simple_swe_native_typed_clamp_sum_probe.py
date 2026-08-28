"""Prepared one-shot visible Simple-SWE clamp-then-sum follow-on.

One retained native temporal circuit learns visible t2_clamp_then_sum/000,
then receives visible t2_clamp_then_sum/001 exactly once without a teacher.
Both tasks come from the public TRAIN partition. Importing or compiling this
source spends nothing; execution is refused until the source is committed and
its output artifact does not exist.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
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
    "public_simple_swe_native_typed_clamp_sum_probe.json"
)
PROTOCOL_PATH = REPO_ROOT / "benchmarks" / "simple_swe_protocol_v1.json"
FAMILY = "t2_clamp_then_sum"
TRAIN_KEY = (FAMILY, 0)
FOLLOW_ON_KEY = (FAMILY, 1)
EXPECTED_TASK_IDS = {
    TRAIN_KEY: "simple-swe-v1/train/tier-2/t2_clamp_then_sum/000",
    FOLLOW_ON_KEY: "simple-swe-v1/train/tier-2/t2_clamp_then_sum/001",
}
EXPECTED_BOUNDS = {
    TRAIN_KEY: ("-7", "7"),
    FOLLOW_ON_KEY: ("-10", "10"),
}

# Preserve the established primitive token order and append Sum once.
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
    "less-than",
    "if",
    "sum",
)
PRIMITIVE_TOKEN_IDS = {
    primitive: 1024 + ordinal for ordinal, primitive in enumerate(_PRIMITIVE_ORDER)
}
PRIMITIVE_BY_TOKEN_ID = {
    token_id: primitive for primitive, token_id in PRIMITIVE_TOKEN_IDS.items()
}

PLAN_COMMIT_TOKEN_ID = 2048
# Parameter signature, result signature, neutral operation, lower slot, upper slot.
PLAN_COMPONENT_COUNT = 5
PLAN_STORE_ARRIVALS_PER_ELIGIBLE_TEACHER = 2
EXPECTED_TEACHER_CALL_COUNT = 24
# teach_typed_token dual-writes from the current cue before advancing state:
# Sum at 0 and the nested clamp at 3..21 have expression/literal cues. Map,
# Bind, the final ListInt variable, and EOS at 1, 2, 22, and 23 do not.
PLAN_STORE_TEACHER_POSITIONS = (0, *range(3, 22))
TYPED_CUE_NAMES = (
    "expression-int-head",
    "expression-int-child-0",
    "expression-int-child-1",
    "expression-int-child-2",
    "expression-bool-head",
    "expression-bool-child-0",
    "expression-bool-child-1",
    "expression-bool-child-2",
    "expression-list-int-head",
    "expression-list-int-child-0",
    "expression-list-int-child-1",
    "expression-list-int-child-2",
    "expression-int-to-int-head",
    "expression-int-to-int-child-0",
    "expression-int-to-int-child-1",
    "expression-int-to-int-child-2",
    "literal-sign",
    "literal-first-digit",
    "literal-first-nonzero-digit",
    "literal-digit-or-end",
    "literal-end",
    "eos",
    "operation-component-neutral",
    "operation-component-lower-bound-slot-2",
    "operation-component-upper-bound-slot-3",
    "operation-component-any-bound",
)
TYPED_CUE_TOKEN_IDS = {
    name: PLAN_COMMIT_TOKEN_ID + ordinal + 1
    for ordinal, name in enumerate(TYPED_CUE_NAMES)
}
ADDRESS_TOKEN_BASE = PLAN_COMMIT_TOKEN_ID + 1 + len(TYPED_CUE_NAMES)


def _address_offset(path: tuple[int, ...]) -> int:
    level_start = 0
    level_width = 1
    rank = 0
    for ordinal in path:
        if ordinal not in {0, 1, 2}:
            raise RuntimeError("typed address contains an unsupported child ordinal")
        level_start += level_width
        level_width *= 3
        rank = rank * 3 + ordinal
    return level_start + rank


# sum(map(bind(clamp), input)): every structural hole that can be queried.
ADDRESS_PATHS = (
    (),
    (0,),
    (0, 0),
    (0, 1),
    (0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 1),
    (0, 0, 0, 2),
    (0, 0, 0, 0, 0),
    (0, 0, 0, 0, 1),
    (0, 0, 0, 2, 0),
    (0, 0, 0, 2, 1),
    (0, 0, 0, 2, 2),
    (0, 0, 0, 2, 0, 0),
    (0, 0, 0, 2, 0, 1),
)
ADDRESS_OFFSETS = tuple(sorted({_address_offset(path) for path in ADDRESS_PATHS}))
if ADDRESS_OFFSETS != (0, 1, 4, 5, 13, 40, 41, 42, 121, 122, 127, 128, 129, 382, 383):
    raise RuntimeError("typed clamp-sum structural address set changed")
ADDRESS_TOKEN_IDS = tuple(ADDRESS_TOKEN_BASE + offset for offset in ADDRESS_OFFSETS)

_TYPED_SURFACE_PREFIX = (
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
TYPED_SURFACE_TOKEN_IDS = (
    *(PRIMITIVE_TOKEN_IDS[primitive] for primitive in _TYPED_SURFACE_PREFIX),
    base.EOS_TOKEN_ID,
    PRIMITIVE_TOKEN_IDS["less-than"],
    PRIMITIVE_TOKEN_IDS["if"],
    PRIMITIVE_TOKEN_IDS["sum"],
)
if TYPED_CUE_NAMES[22:] != (
    "operation-component-neutral",
    "operation-component-lower-bound-slot-2",
    "operation-component-upper-bound-slot-3",
    "operation-component-any-bound",
):
    raise RuntimeError("typed clamp-sum operation cue ordinals changed")
if (
    len(TYPED_SURFACE_TOKEN_IDS) != 23
    or TYPED_SURFACE_TOKEN_IDS[22] != PRIMITIVE_TOKEN_IDS["sum"]
):
    raise RuntimeError("typed clamp-sum requires 23 surfaces with Sum at ordinal 22")
TYPED_PRIVATE_TOKEN_IDS = (
    PLAN_COMMIT_TOKEN_ID,
    *TYPED_CUE_TOKEN_IDS.values(),
    *ADDRESS_TOKEN_IDS,
)
_RESERVED_COLUMN_COUNT = len(TYPED_PRIVATE_TOKEN_IDS) * base.ACTIVE_COLUMNS

# The shared trace formatter is identity-neutral; install this probe's surface map.
base.PRIMITIVE_TOKEN_IDS = PRIMITIVE_TOKEN_IDS
base.PRIMITIVE_BY_TOKEN_ID = PRIMITIVE_BY_TOKEN_ID


def _workspace_vocabulary() -> TypedWorkspaceVocabulary:
    return TypedWorkspaceVocabulary(
        plan_commit_token_id=PLAN_COMMIT_TOKEN_ID,
        address_token_base=ADDRESS_TOKEN_BASE,
        cue_token_ids=tuple(TYPED_CUE_TOKEN_IDS.values()),
        surface_token_ids=TYPED_SURFACE_TOKEN_IDS,
    )


def _typed_value(program_type: object) -> TypedValue:
    if type(program_type) is not ProgramType:
        raise TypeError("typed workspace requires an exact ProgramType")
    if program_type == ProgramType.Int():
        return TypedValue.INT
    if program_type == ProgramType.List(ProgramType.Int()):
        return TypedValue.LIST_INT
    raise RuntimeError("typed clamp-sum workspace received an unsupported public value")


@dataclass(frozen=True, slots=True)
class _TaskFields:
    entry_point: bytes
    parameter: bytes
    lower: bytes
    upper: bytes


@dataclass(frozen=True, slots=True)
class _WorkspaceEpisode:
    prompt_steps: tuple[object, ...]
    workspace_steps: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _FollowOnRuntimeInput:
    """The complete and only Python value allowed into follow-on inference."""

    prompt_tokens: tuple[int, ...]
    parameter_type: object
    result_type: object


class _TypedWorkspaceApi:
    """Sole adapter to the native atomic typed-workspace interface."""

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
            _workspace_vocabulary(),
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
    def begin_follow_on(
        circuit: NativeTemporalCircuit,
        runtime_input: _FollowOnRuntimeInput,
    ) -> _WorkspaceEpisode:
        prompt_steps = circuit.advance_block(
            runtime_input.prompt_tokens,
            mode=TemporalStepMode.READ_ONLY,
            reset_before=True,
        )
        begin_step = circuit.begin_typed_session(
            _workspace_vocabulary(),
            parameter_type=_typed_value(runtime_input.parameter_type),
            result_type=_typed_value(runtime_input.result_type),
            mode=TemporalStepMode.READ_ONLY,
        )
        return _WorkspaceEpisode(
            prompt_steps=prompt_steps,
            workspace_steps=(begin_step,),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--python-executable", type=Path, required=True)
    return parser.parse_args()


def _assert_source_committed_and_output_absent(output: Path) -> None:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing one-shot artifact: {output}")
    source = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", source],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise RuntimeError("probe source must be committed and unmodified before its one shot")
    if SLOT_COUNT != 4:
        raise RuntimeError("typed clamp-sum probe requires the four-slot transport")


def _select_tasks() -> tuple[object, GeneratedTask, GeneratedTask]:
    protocol = load_protocol(PROTOCOL_PATH)
    visible_train = generate_visible_partition(protocol, "train")
    selected: dict[tuple[str, int], GeneratedTask] = {}
    for key in (TRAIN_KEY, FOLLOW_ON_KEY):
        family, ordinal = key
        task = next(
            item
            for item in visible_train
            if item.public.family == family and item.public.ordinal == ordinal
        )
        if task.public.task_id != EXPECTED_TASK_IDS[key]:
            raise RuntimeError(f"visible task identity changed for {key!r}")
        _task_fields(task.public)
        selected[key] = task
    return protocol, selected[TRAIN_KEY], selected[FOLLOW_ON_KEY]


def _task_fields(public_task: object) -> _TaskFields:
    if type(public_task).__name__ != "PublicTask":
        raise TypeError("typed clamp-sum task parsing requires the exact public task view")
    key = (public_task.family, public_task.ordinal)
    if key not in EXPECTED_BOUNDS:
        raise RuntimeError(f"unsupported typed clamp-sum key {key!r}")
    signature = public_task.signature
    if (
        len(signature.parameters) != 1
        or signature.parameters[0].type != ProgramType.List(ProgramType.Int())
        or signature.result != ProgramType.Int()
    ):
        raise RuntimeError("typed clamp-sum public signature changed")
    entry = signature.entry_point.encode("ascii")
    parameter = signature.parameters[0].name.encode("ascii")
    lower_text, upper_text = EXPECTED_BOUNDS[key]
    prompt = public_task.prompt.encode("ascii")
    signature_bytes = b"def " + entry + b"(" + parameter + b"):"
    prompt.index(signature_bytes)
    bound_anchor = b"Limit every integer to "
    lower_start = prompt.index(bound_anchor) + len(bound_anchor)
    lower = lower_text.encode("ascii")
    upper = upper_text.encode("ascii")
    if prompt[lower_start : lower_start + len(lower)] != lower:
        raise RuntimeError("visible lower bound changed")
    through = b" through "
    upper_start = lower_start + len(lower)
    if prompt[upper_start : upper_start + len(through)] != through:
        raise RuntimeError("visible bound separator changed")
    upper_start += len(through)
    if prompt[upper_start : upper_start + len(upper)] != upper:
        raise RuntimeError("visible upper bound changed")
    return _TaskFields(entry, parameter, lower, upper)


def _canonical_literal_primitives(literal: bytes) -> tuple[str, ...]:
    text = literal.decode("ascii")
    signed = text if text.startswith("-") else f"+{text}"
    if signed == "-0" or not signed[1:].isdigit():
        raise RuntimeError(f"non-canonical visible literal {text!r}")
    return tuple(signed)


def _prompt_spans(
    public_task: object,
) -> tuple[bytes, tuple[tuple[int, int, int, tuple[int, ...]], ...]]:
    fields = _task_fields(public_task)
    prompt = public_task.prompt.encode("ascii")
    signature = b"def " + fields.entry_point + b"(" + fields.parameter + b"):"
    signature_start = prompt.index(signature)
    entry_start = signature_start + len(b"def ")
    parameter_start = entry_start + len(fields.entry_point) + 1
    bound_anchor = b"Limit every integer to "
    lower_start = prompt.index(bound_anchor, parameter_start) + len(bound_anchor)
    upper_start = lower_start + len(fields.lower) + len(b" through ")
    spans = (
        (
            entry_start,
            entry_start + len(fields.entry_point),
            0,
            tuple(fields.entry_point),
        ),
        (
            parameter_start,
            parameter_start + len(fields.parameter),
            1,
            tuple(fields.parameter),
        ),
        (
            lower_start,
            lower_start + len(fields.lower),
            2,
            tuple(
                PRIMITIVE_TOKEN_IDS[token]
                for token in _canonical_literal_primitives(fields.lower)
            ),
        ),
        (
            upper_start,
            upper_start + len(fields.upper),
            3,
            tuple(
                PRIMITIVE_TOKEN_IDS[token]
                for token in _canonical_literal_primitives(fields.upper)
            ),
        ),
    )
    return prompt, spans


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
            if active_slot in {2, 3}:
                signed = "".join(literal_parts)
                raw = signed[1:] if signed.startswith("+") else signed
                output.extend(raw.encode("ascii"))
            active_slot = None
            continue
        if active_slot in {2, 3}:
            primitive = PRIMITIVE_BY_TOKEN_ID.get(token_id)
            if primitive is None:
                raise RuntimeError("non-primitive token in a bound slot")
            literal_parts.append(primitive)
        else:
            if not 0 <= token_id < 128:
                raise RuntimeError("non-ASCII token outside a canonical bound slot")
            output.append(token_id)
    if active_slot is not None:
        raise RuntimeError("unterminated prompt slot")
    return bytes(output)


def _tokenized_public_prompt(public_task: object) -> tuple[int, ...]:
    prompt, spans = _prompt_spans(public_task)
    tokens: list[int] = []
    cursor = 0
    for start, end, slot, replacement in spans:
        if start < cursor or end <= start:
            raise RuntimeError("public prompt spans overlap")
        tokens.extend(prompt[cursor:start])
        tokens.append(slot_start_token_id(slot))
        tokens.extend(replacement)
        tokens.append(slot_end_token_id(slot))
        cursor = end
    tokens.extend(prompt[cursor:])
    result = tuple(tokens)
    if _detokenize_prompt(result) != prompt:
        raise RuntimeError("typed prompt tokenizer is not lossless")
    return result


def _follow_on_runtime_input(public_task: object) -> _FollowOnRuntimeInput:
    if (public_task.family, public_task.ordinal) != FOLLOW_ON_KEY:
        raise RuntimeError("follow-on runtime key changed")
    signature = public_task.signature
    return _FollowOnRuntimeInput(
        prompt_tokens=_tokenized_public_prompt(public_task),
        parameter_type=signature.parameters[0].type,
        result_type=signature.result,
    )


def _internal_teacher_tokens() -> tuple[int, ...]:
    primitives: tuple[str | tuple[str, int], ...] = (
        "sum",
        "map",
        "bind",
        "if",
        "less-than",
        "variable-0",
        "integer",
        ("copy", 2),
        "end",
        "integer",
        ("copy", 2),
        "end",
        "if",
        "less-than",
        "integer",
        ("copy", 3),
        "end",
        "variable-0",
        "integer",
        ("copy", 3),
        "end",
        "variable-0",
        "variable-0",
    )
    return (
        *(
            copy_slot_token_id(item[1])
            if isinstance(item, tuple)
            else PRIMITIVE_TOKEN_IDS[item]
            for item in primitives
        ),
        base.EOS_TOKEN_ID,
    )


def _learning_arrival_accounting(
    train_prompt: tuple[int, ...],
    teacher: tuple[int, ...],
) -> dict[str, int]:
    if len(teacher) != EXPECTED_TEACHER_CALL_COUNT:
        raise RuntimeError("typed clamp-sum fixed teacher stream length changed")
    expected_nested_clamp = (
        PRIMITIVE_TOKEN_IDS["if"],
        PRIMITIVE_TOKEN_IDS["less-than"],
        PRIMITIVE_TOKEN_IDS["variable-0"],
        PRIMITIVE_TOKEN_IDS["integer"],
        copy_slot_token_id(2),
        PRIMITIVE_TOKEN_IDS["end"],
        PRIMITIVE_TOKEN_IDS["integer"],
        copy_slot_token_id(2),
        PRIMITIVE_TOKEN_IDS["end"],
        PRIMITIVE_TOKEN_IDS["if"],
        PRIMITIVE_TOKEN_IDS["less-than"],
        PRIMITIVE_TOKEN_IDS["integer"],
        copy_slot_token_id(3),
        PRIMITIVE_TOKEN_IDS["end"],
        PRIMITIVE_TOKEN_IDS["variable-0"],
        PRIMITIVE_TOKEN_IDS["integer"],
        copy_slot_token_id(3),
        PRIMITIVE_TOKEN_IDS["end"],
        PRIMITIVE_TOKEN_IDS["variable-0"],
    )
    if (
        teacher[0] != PRIMITIVE_TOKEN_IDS["sum"]
        or teacher[1] != PRIMITIVE_TOKEN_IDS["map"]
        or teacher[2] != PRIMITIVE_TOKEN_IDS["bind"]
        or teacher[3:22] != expected_nested_clamp
        or teacher[22] != PRIMITIVE_TOKEN_IDS["variable-0"]
        or teacher[23] != base.EOS_TOKEN_ID
    ):
        raise RuntimeError("typed clamp-sum fixed teacher stream boundaries changed")
    plan_store_teacher_count = len(PLAN_STORE_TEACHER_POSITIONS)
    if plan_store_teacher_count != 20:
        raise RuntimeError("typed clamp-sum cue-position plan-store count changed")
    plan_store_arrivals = (
        plan_store_teacher_count * PLAN_STORE_ARRIVALS_PER_ELIGIBLE_TEACHER
    )
    total_learning_arrivals = (
        len(train_prompt)
        + PLAN_COMPONENT_COUNT
        + len(teacher)
        + plan_store_arrivals
    )
    segment_capacity = (total_learning_arrivals - 1) * base.ACTIVE_COLUMNS
    base_population = base.COLUMN_COUNT * base.BASE_CELLS_PER_COLUMN
    return {
        "train_prompt_arrivals": len(train_prompt),
        "retained_plan_component_arrivals": PLAN_COMPONENT_COUNT,
        "typed_state_advance_arrivals": len(teacher),
        "plan_store_teacher_count": plan_store_teacher_count,
        "plan_store_arrivals_per_eligible_teacher": (
            PLAN_STORE_ARRIVALS_PER_ELIGIBLE_TEACHER
        ),
        "exact_and_backoff_plan_store_arrivals": plan_store_arrivals,
        "total_learning_arrivals": total_learning_arrivals,
        "segments_per_noninitial_arrival": base.ACTIVE_COLUMNS,
        "segment_capacity": segment_capacity,
        "base_pyramidal_cells": base_population,
        "reserve_pyramidal_cells": segment_capacity,
        "max_pyramidal_cells": base_population + segment_capacity,
    }


def _expected_program_tokens(public_task: object) -> tuple[str, ...]:
    fields = _task_fields(public_task)
    lower = _canonical_literal_primitives(fields.lower)
    upper = _canonical_literal_primitives(fields.upper)
    return (
        "sum",
        "map",
        "bind",
        "if",
        "less-than",
        "variable-0",
        "integer",
        *lower,
        "end",
        "integer",
        *lower,
        "end",
        "if",
        "less-than",
        "integer",
        *upper,
        "end",
        "variable-0",
        "integer",
        *upper,
        "end",
        "variable-0",
        "variable-0",
    )


def _expected_compiled_source(public_task: object) -> str:
    fields = _task_fields(public_task)
    entry = fields.entry_point.decode("ascii")
    parameter = fields.parameter.decode("ascii")
    lower = fields.lower.decode("ascii")
    upper = fields.upper.decode("ascii")
    return (
        f"def {entry}({parameter}):\n"
        f"    return sum([{lower} if item_0 < {lower} else "
        f"{upper} if {upper} < item_0 else item_0 for item_0 in {parameter}])\n"
    )


def _public_task_record(public_task: object) -> dict[str, object]:
    fields = _task_fields(public_task)
    prompt_tokens = _tokenized_public_prompt(public_task)
    return {
        "task_id": public_task.task_id,
        "partition": public_task.partition,
        "family": public_task.family,
        "ordinal": public_task.ordinal,
        "entry_point": fields.entry_point.decode("ascii"),
        "parameter": fields.parameter.decode("ascii"),
        "lower": fields.lower.decode("ascii"),
        "upper": fields.upper.decode("ascii"),
        "task_sha256": public_task.task_sha256,
        "hidden_test_sha256": public_task.hidden_test_sha256,
        "public_prompt_sha256": hashlib.sha256(
            public_task.prompt.encode("utf-8")
        ).hexdigest(),
        "public_prompt_lossless_after_tokenization": (
            _detokenize_prompt(prompt_tokens) == public_task.prompt.encode("ascii")
        ),
        "prompt_token_count": len(prompt_tokens),
    }


def _definition(
    train_task: GeneratedTask,
    follow_on_input: _FollowOnRuntimeInput,
) -> tuple[NativeTemporalCircuitDefinition, dict[str, int]]:
    train_prompt = _tokenized_public_prompt(train_task.public)
    teacher = _internal_teacher_tokens()
    control_tokens = {
        *(slot_start_token_id(slot) for slot in range(SLOT_COUNT)),
        *(slot_end_token_id(slot) for slot in range(SLOT_COUNT)),
        *(copy_slot_token_id(slot) for slot in range(SLOT_COUNT)),
        COPY_END_TOKEN_ID,
    }
    used_token_ids = tuple(
        sorted(
            {
                *train_prompt,
                *follow_on_input.prompt_tokens,
                *PRIMITIVE_BY_TOKEN_ID,
                *control_tokens,
                *TYPED_PRIVATE_TOKEN_IDS,
                base.EOS_TOKEN_ID,
            }
        )
    )

    capacity_accounting = _learning_arrival_accounting(train_prompt, teacher)
    segment_capacity = capacity_accounting["segment_capacity"]
    config = NativeTemporalCircuitConfig(
        column_count=base.COLUMN_COUNT,
        max_pyramidal_cells=capacity_accounting["max_pyramidal_cells"],
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
        raise RuntimeError("typed clamp-sum private codes leave too few surface columns")

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
    private_columns = {
        column
        for token_id in TYPED_PRIVATE_TOKEN_IDS
        for column in token_codes[token_id]
    }
    if len(private_columns) != _RESERVED_COLUMN_COUNT:
        raise RuntimeError("typed clamp-sum private code ranges overlap")
    if any(
        column < _RESERVED_COLUMN_COUNT
        for token_id, code in token_codes.items()
        if token_id not in TYPED_PRIVATE_TOKEN_IDS
        for column in code
    ):
        raise RuntimeError("prompt/surface code entered the typed private reserve")

    physical_cell_count = 2 * config.max_pyramidal_cells + 3 * config.column_count
    return (
        NativeTemporalCircuitDefinition(
            config,
            token_codes,
            seeded_threshold_offsets_mv(
                physical_cell_count,
                config.construction_seed,
            ).tolist(),
        ),
        capacity_accounting,
    )


def _generation_failure(error: NativeTemporalRuntimeError) -> dict[str, object]:
    return {
        "kind": "model",
        "gate": "GENERATION_STOP",
        "detail": {
            "status": error.status.name,
            "message": error.message,
            "required": error.required,
            "observed": error.observed,
            "location": error.location,
            "poison_code": error.poison_code,
            "stage_code": error.stage_code,
        },
    }


def _expected_copy_schedule(
    public_task: object,
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    fields = _task_fields(public_task)
    lower = _canonical_literal_primitives(fields.lower)
    upper = _canonical_literal_primitives(fields.upper)
    return ((2, lower), (2, lower), (3, upper), (3, upper))


def _copy_transactions(
    generations: list[dict[str, object]],
    expected: tuple[tuple[int, tuple[str, ...]], ...],
) -> tuple[list[dict[str, object]], bool]:
    transactions: list[dict[str, object]] = []
    active: dict[str, object] | None = None
    exact = True
    for generation in generations:
        path = str(generation["path"])
        if path == TemporalGenerationPath.ENTER_COPY.name:
            exact &= active is None
            active = {
                "slot_ordinal": generation["slot_ordinal"],
                "control_token_id": generation["control_token_id"],
                "copied_places": [generation["copy_place_ordinal"]],
                "copied_primitives": [generation["surface_primitive"]],
                "physical_election_counts": [generation["physical_election_count"]],
            }
        elif path == TemporalGenerationPath.COPY.name:
            exact &= active is not None
            if active is not None:
                exact &= generation["slot_ordinal"] == active["slot_ordinal"]
                active["copied_places"].append(generation["copy_place_ordinal"])
                active["copied_primitives"].append(generation["surface_primitive"])
                active["physical_election_counts"].append(
                    generation["physical_election_count"]
                )
        elif path == TemporalGenerationPath.EXIT_COPY.name:
            exact &= active is not None
            if active is not None:
                exact &= generation["slot_ordinal"] == active["slot_ordinal"]
                active["exit_surface_primitive"] = generation["surface_primitive"]
                transactions.append(active)
                active = None

    exact &= active is None and len(transactions) == len(expected)
    for transaction, (slot, primitives) in zip(transactions, expected, strict=False):
        exact &= transaction["slot_ordinal"] == slot
        exact &= transaction["control_token_id"] == copy_slot_token_id(slot)
        exact &= transaction["copied_places"] == list(range(len(primitives)))
        exact &= transaction["copied_primitives"] == list(primitives)
        exact &= transaction["exit_surface_primitive"] == "end"
        exact &= all(int(value) >= 1 for value in transaction["physical_election_counts"])
    return transactions, exact


def _score_follow_on(
    *,
    episode: _WorkspaceEpisode,
    raw_generations: tuple[object, ...] | None,
    native_failure: dict[str, object] | None,
    task: GeneratedTask,
    protocol: object,
    python_executable: Path,
    runtime_identity: dict[str, str],
    worker_source: bytes,
    worker_sha256: str,
) -> tuple[dict[str, object], dict[str, object] | None]:
    # This is the first point where the follow-on target or private cases may
    # enter Python, and the native circuit is already closed.
    expected_tokens = _expected_program_tokens(task.public)
    expected_surface_ids = tuple(
        PRIMITIVE_TOKEN_IDS[token] for token in expected_tokens
    ) + (base.EOS_TOKEN_ID,)
    generations = [
        base._generation_record(position, generation)
        for position, generation in enumerate(raw_generations or ())
    ]
    observed_surface_ids = tuple(
        int(generation.surface_token_id) for generation in (raw_generations or ())
    )
    first_failure = native_failure
    if first_failure is None and observed_surface_ids != expected_surface_ids:
        first_failure = {
            "kind": "model",
            "gate": "EXACT_TYPED_TOKEN_STREAM",
            "detail": {
                "expected": list(expected_surface_ids),
                "observed": list(observed_surface_ids),
            },
        }
    terminated_by_eos = bool(
        observed_surface_ids
        and observed_surface_ids[-1] == base.EOS_TOKEN_ID
        and generations[-1]["path"] == TemporalGenerationPath.TEMPORAL.name
    )
    if first_failure is None and not terminated_by_eos:
        first_failure = {"kind": "model", "gate": "ORDINARY_EOS", "detail": None}

    transactions, copy_trace_exact = _copy_transactions(
        generations,
        _expected_copy_schedule(task.public),
    )
    if first_failure is None and not copy_trace_exact:
        first_failure = {
            "kind": "model",
            "gate": "BOUND_LITERAL_PHYSICAL_COPY",
            "detail": transactions,
        }
    fitted_cell_races_exact = all(
        int(generation["physical_election_count"]) >= 1
        and int(generation["candidate_tokens_physical"]) >= 1
        and int(generation["winner_input_columns"]) == base.ACTIVE_COLUMNS
        for generation in generations
    )
    if first_failure is None and not fitted_cell_races_exact:
        first_failure = {"kind": "model", "gate": "FITTED_CELL_RACES", "detail": None}

    emitted = tuple(
        PRIMITIVE_BY_TOKEN_ID[token_id]
        for token_id in observed_surface_ids
        if token_id != base.EOS_TOKEN_ID and token_id in PRIMITIVE_BY_TOKEN_ID
    )
    compiled_source: str | None = None
    program_record: dict[str, object] = {
        "tokens": list(emitted),
        "expected_tokens": list(expected_tokens),
        "exact_tokens": emitted == expected_tokens,
        "compiled_source": None,
        "expected_compiled_source": _expected_compiled_source(task.public),
        "exact_compiled_source": False,
        "compiler_received_prompt": False,
        "compiler_received_tests": False,
    }
    if first_failure is None:
        try:
            program = Program(emitted, task.public.signature)
            compiled_source = compile_typed_program(program).source
        except (TypeError, ValueError) as error:
            first_failure = {
                "kind": "model",
                "gate": "EXISTING_TYPED_PROGRAM_COMPILE",
                "detail": f"{type(error).__name__}: {error}",
            }
        else:
            exact_source = compiled_source == program_record["expected_compiled_source"]
            program_record.update(
                {
                    "program_sha256": program.sha256,
                    "compiled_source": compiled_source,
                    "exact_compiled_source": exact_source,
                }
            )
            if not exact_source:
                first_failure = {
                    "kind": "model",
                    "gate": "EXACT_SIGNATURE_PYTHON",
                    "detail": None,
                }

    sandbox_record: dict[str, object] = {
        "attempted": False,
        "passed": False,
        "detail": "not run because an earlier gate stopped",
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
            "task": _public_task_record(task.public),
            "attempt_number": 1,
            "prompt_steps": len(episode.prompt_steps),
            "workspace_steps_before_generation": len(episode.workspace_steps),
            "generation_call_count": 1,
            "atomic_no_argument_generation": True,
            "teacher_tokens_after_prompt": 0,
            "target_entered_runtime": False,
            "private_cases_entered_runtime": False,
            "expected_stream_used_only_after_native_close": True,
            "terminated_by_ordinary_eos": terminated_by_eos,
            "program": program_record,
            "copy_transactions": transactions,
            "copy_slot_schedule": [2, 2, 3, 3],
            "copy_trace_exact": copy_trace_exact,
            "fitted_cell_races_exact": fitted_cell_races_exact,
            "sandbox": sandbox_record,
            "first_failure": first_failure,
            "generations": generations,
        },
        first_failure,
    )


def _teacher_label(token_id: int) -> str:
    for slot in (2, 3):
        if token_id == copy_slot_token_id(slot):
            return f"COPY_SLOT_{slot}"
    if token_id == base.EOS_TOKEN_ID:
        return "<EOS>"
    return PRIMITIVE_BY_TOKEN_ID[token_id]


def main() -> int:
    args = _parse_args()
    _assert_source_committed_and_output_absent(args.output)
    started_ns = time.perf_counter_ns()
    protocol, train_task, follow_on_task = _select_tasks()
    train_public = train_task.public
    follow_on_input = _follow_on_runtime_input(follow_on_task.public)
    teacher = _internal_teacher_tokens()
    definition, capacity_accounting = _definition(train_task, follow_on_input)

    executable = resolve_python_executable(args.python_executable)
    runtime_identity = python_runtime_identity(executable, protocol.raw.get("language"))
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = hashlib.sha256(worker_path.read_bytes()).hexdigest()
    worker_source = load_worker_snapshot(worker_sha256)

    with base._public_development_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            signature = train_public.signature
            train_episode = _TypedWorkspaceApi.train_task(
                circuit,
                prompt_tokens=_tokenized_public_prompt(train_public),
                parameter_type=signature.parameters[0].type,
                result_type=signature.result,
                teacher_tokens=teacher,
            )
            trained_identity = circuit.checkpoint_identity()
            follow_on_episode = _TypedWorkspaceApi.begin_follow_on(circuit, follow_on_input)
            try:
                raw_generations = circuit.generate_typed()
                native_failure = None
            except NativeTemporalRuntimeError as error:
                raw_generations = None
                native_failure = _generation_failure(error)
        native_identity = loaded_library_identity()

    # Target construction, every compiler check, and private sandbox cases begin
    # only after atomic generation and after the native circuit has closed.
    train_compiler_source = compile_typed_program(
        Program(_expected_program_tokens(train_public), train_public.signature)
    ).source
    if train_compiler_source != _expected_compiled_source(train_public):
        raise RuntimeError("typed compiler cannot express the TRAIN clamp-sum target")
    follow_on_record, first_failure = _score_follow_on(
        episode=follow_on_episode,
        raw_generations=raw_generations,
        native_failure=native_failure,
        task=follow_on_task,
        protocol=protocol,
        python_executable=executable,
        runtime_identity=runtime_identity,
        worker_source=worker_source,
        worker_sha256=worker_sha256,
    )

    passed = first_failure is None
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-native-typed-clamp-sum",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": (
            "PASS_NATIVE_TYPED_CLAMP_SUM"
            if passed
            else "STOP_NATIVE_TYPED_CLAMP_SUM"
        ),
        "capability_pass": passed,
        "first_failure": first_failure,
        "claim_boundary": (
            "One retained native temporal circuit learns visible TRAIN "
            "t2_clamp_then_sum/000 and attempts visible TRAIN /001 exactly once. "
            "A pass is same-family public follow-on evidence for generic "
            "sum(map(bind(nested clamp), input)) with unseen physical bounds. "
            "It is not family-disjoint transfer, a DEV/eval result, broad coding "
            "competence, M0, or C1."
        ),
        "authority": {
            "source_partition_generated": "train",
            "eval_partition_generated": False,
            "train_key": list(TRAIN_KEY),
            "follow_on_key": list(FOLLOW_ON_KEY),
            "follow_on_attempts_completed": 1,
            "model_retries": 0,
            "harness_retries": 0,
            "follow_on_target_used_as_runtime_input": False,
            "follow_on_private_cases_used_as_runtime_input": False,
            "teacher_tokens_after_follow_on_prompt": 0,
            "follow_on_runtime_fields": [
                "lossless_prompt_tokens",
                "public_parameter_type",
                "public_result_type",
            ],
            "python_candidate_enumeration": False,
            "python_argmax": False,
        },
        "protocol": {
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.sha256,
            "language": protocol.raw.get("language"),
        },
        "typed_programs": {
            "train_teacher": [_teacher_label(token_id) for token_id in teacher],
            "train_compiled_source": train_compiler_source,
            "follow_on_expected": " ".join(
                _expected_program_tokens(follow_on_task.public)
            ),
            "follow_on_stream_equals_train_stream": False,
        },
        "factorized_plan": {
            "component_count_with_two_bounds": PLAN_COMPONENT_COUNT,
            "parameter_signature_winners": 3,
            "result_signature_winners": 2,
            "lower_bound_winners": 3,
            "upper_bound_winners": 3,
            "cue_22": "neutral/absent-bound",
            "cue_23": "physical lower-bound slot 2 ready",
            "cue_24": "physical upper-bound slot 3 ready",
            "cue_25": "any-bound operation backoff",
            "typed_cue_count": len(TYPED_CUE_NAMES),
            "typed_surface_count": len(TYPED_SURFACE_TOKEN_IDS),
            "surface_22": "sum",
            "plan_store_teacher_positions_zero_based": list(
                PLAN_STORE_TEACHER_POSITIONS
            ),
        },
        "slot_bank": {
            "slot_count": SLOT_COUNT,
            "entry_slot_ordinal": 0,
            "parameter_slot_ordinal": 1,
            "lower_bound_slot_ordinal": 2,
            "upper_bound_slot_ordinal": 3,
            "expected_copy_slot_schedule": [2, 2, 3, 3],
            "copy_control_token_ids": [copy_slot_token_id(2), copy_slot_token_id(3)],
            "copy_end_token_id": COPY_END_TOKEN_ID,
        },
        "physical_definition": {
            "circuit_identity_sha256": definition.circuit_identity_sha256(),
            "column_count": definition.config.column_count,
            "active_columns_per_token": base.ACTIVE_COLUMNS,
            "used_token_count": len(definition.token_codes),
            "physical_cell_count": len(definition.threshold_offsets_mv),
            "segment_capacity": definition.config.segment_capacity,
            "plan_commit_token_id": PLAN_COMMIT_TOKEN_ID,
            "typed_cue_token_ids": TYPED_CUE_TOKEN_IDS,
            "typed_surface_token_ids": list(TYPED_SURFACE_TOKEN_IDS),
            "sum_surface_ordinal": len(TYPED_SURFACE_TOKEN_IDS) - 1,
            "address_token_base": ADDRESS_TOKEN_BASE,
            "address_offsets": list(ADDRESS_OFFSETS),
            "typed_private_token_ids": list(TYPED_PRIVATE_TOKEN_IDS),
            "workspace_reserved_column_count": _RESERVED_COLUMN_COUNT,
            "capacity_accounting": capacity_accounting,
        },
        "train": {
            "task": _public_task_record(train_public),
            "prompt_steps": len(train_episode.prompt_steps),
            "workspace_steps": len(train_episode.workspace_steps),
            "reset_before": True,
            "live_segments_after_train": trained_identity.segment_count,
            "learning_steps_after_train": trained_identity.learning_step,
        },
        "visible_train_follow_on_teacher_free": follow_on_record,
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
