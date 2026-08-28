"""Prepared one-shot visible Simple-SWE nested-clamp probe.

One retained native temporal circuit learns the visible scalar-nonnegative/000
and scalar-clamp/000 tasks, then receives scalar-clamp/007 exactly once without
a teacher.  Importing or compiling this source does not spend that development
task.  The executable path refuses to run until this source is committed and
the requested output artifact is absent.
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
    "public_simple_swe_native_typed_clamp_probe.json"
)
PROTOCOL_PATH = REPO_ROOT / "benchmarks" / "simple_swe_protocol_v1.json"
NONNEGATIVE_FAMILY = "t0_scalar_nonnegative"
CLAMP_FAMILY = "t0_scalar_clamp"
TRAIN_KEYS = ((NONNEGATIVE_FAMILY, 0), (CLAMP_FAMILY, 0))
DEVELOPMENT_KEY = (CLAMP_FAMILY, 7)
DEVELOPMENT_EXPECTED_SOURCE = (
    "def solve_617891da99e65692(n):\n"
    "    return -5 if n < -5 else 5 if 5 < n else n\n"
)

EXPECTED_SOURCE_RECORDS: dict[tuple[str, int], dict[str, object]] = {
    (NONNEGATIVE_FAMILY, 0): {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_nonnegative/000",
        "entry_point": "solve_8ed5dc71c909936d",
        "parameter": "n",
        "lower": None,
        "upper": None,
        "task_sha256": "92af3d9c9830101cf5a78e1336b068e6a2dedf3f63d6ce3ef817e690b2074b5e",
        "prompt": (
            "Return exactly one complete Python function. Use only Python built-ins; "
            "do not emit Markdown, prose, tests, imports, or input/output examples.\n"
            "Implement this exact task:\n"
            "def solve_8ed5dc71c909936d(n):\n"
            "    \"\"\"Return n when it is nonnegative, otherwise return zero.\"\"\""
        ),
    },
    (CLAMP_FAMILY, 0): {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_clamp/000",
        "entry_point": "solve_86f0ac8ef86cc2ae",
        "parameter": "n",
        "lower": "-2",
        "upper": "2",
        "task_sha256": "c4ae41ed141f99dcb34c2c10279ccd55089b2f12ac0d3beca3068d2f6d8bf983",
        "prompt": (
            "Return exactly one complete Python function. Use only Python built-ins; "
            "do not emit Markdown, prose, tests, imports, or input/output examples.\n"
            "Implement this exact task:\n"
            "def solve_86f0ac8ef86cc2ae(n):\n"
            "    \"\"\"Return n limited to the inclusive range -2 through 2.\"\"\""
        ),
    },
    (CLAMP_FAMILY, 7): {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_clamp/007",
        "entry_point": "solve_617891da99e65692",
        "parameter": "n",
        "lower": "-5",
        "upper": "5",
        "task_sha256": "4d3a2c4da9c5f0adf9b4542dbbec11b2bb28940a4bd08ff6af2c1ed3502fa9e3",
        "prompt": (
            "Return exactly one complete Python function. Use only Python built-ins; "
            "do not emit Markdown, prose, tests, imports, or input/output examples.\n"
            "Implement this exact task:\n"
            "def solve_617891da99e65692(n):\n"
            "    \"\"\"Return n limited to the inclusive range -5 through 5.\"\"\""
        ),
    },
}

# Preserve all existing physical primitive identities and append only the two
# new typed surfaces.
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
    "operation-component-bound-slot-count-0",
    "operation-component-bound-slot-count-1",
    "operation-component-bound-slot-count-2",
)
TYPED_CUE_TOKEN_IDS = {
    name: PLAN_COMMIT_TOKEN_ID + ordinal + 1
    for ordinal, name in enumerate(TYPED_CUE_NAMES)
}
ADDRESS_TOKEN_BASE = PLAN_COMMIT_TOKEN_ID + 1 + len(TYPED_CUE_NAMES)
ADDRESS_OFFSETS = (0, 1, 2, 3, 4, 5, 10, 11, 12, 31, 32)
ADDRESS_TOKEN_IDS = tuple(ADDRESS_TOKEN_BASE + offset for offset in ADDRESS_OFFSETS)

# Native indices 0..19 remain the passed workspace mapping. Less-than and if
# are appended at 20 and 21 after the ordinary EOS surface.
_LEGACY_TYPED_SURFACE_ORDER = (
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
    *(PRIMITIVE_TOKEN_IDS[primitive] for primitive in _LEGACY_TYPED_SURFACE_ORDER),
    base.EOS_TOKEN_ID,
    PRIMITIVE_TOKEN_IDS["less-than"],
    PRIMITIVE_TOKEN_IDS["if"],
)
TYPED_PRIVATE_TOKEN_IDS = (
    PLAN_COMMIT_TOKEN_ID,
    *TYPED_CUE_TOKEN_IDS.values(),
    *ADDRESS_TOKEN_IDS,
)
_RESERVED_CODE_COUNT = len(TYPED_PRIVATE_TOKEN_IDS)
_RESERVED_COLUMN_COUNT = _RESERVED_CODE_COUNT * base.ACTIVE_COLUMNS

# The shared formatter is neutral, but it must decode this probe's added
# physical primitive identities.
base.PRIMITIVE_TOKEN_IDS = PRIMITIVE_TOKEN_IDS
base.PRIMITIVE_BY_TOKEN_ID = PRIMITIVE_BY_TOKEN_ID


def _workspace_vocabulary() -> TypedWorkspaceVocabulary:
    """Construct only after the 25/22/address transport has landed."""

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
    if program_type == ProgramType.Bool():
        return TypedValue.BOOL
    if program_type == ProgramType.List(ProgramType.Int()):
        return TypedValue.LIST_INT
    raise RuntimeError("typed clamp workspace received an unsupported public value")


@dataclass(frozen=True, slots=True)
class _TaskFields:
    entry_point: bytes
    parameter: bytes
    lower: bytes | None
    upper: bytes | None


@dataclass(frozen=True, slots=True)
class _WorkspaceEpisode:
    prompt_steps: tuple[object, ...]
    workspace_steps: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _WorkspaceTrace:
    generations: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _DevelopmentRuntimeInput:
    """The complete and only Python value allowed into DEV native inference."""

    prompt_tokens: tuple[int, ...]
    parameter_type: object
    result_type: object


class _TypedWorkspaceApi:
    """Sole adapter to the native atomic typed-workspace API."""

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
    def begin_development(
        circuit: NativeTemporalCircuit,
        runtime_input: _DevelopmentRuntimeInput,
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

    @staticmethod
    def generate(circuit: NativeTemporalCircuit) -> _WorkspaceTrace:
        return _WorkspaceTrace(generations=circuit.generate_typed())


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
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            source,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise RuntimeError("probe source must be committed and unmodified before its one shot")
    if SLOT_COUNT != 4:
        raise RuntimeError("typed clamp probe requires the landed four-slot transport")


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
            "prompt": task.public.prompt,
        }
        expected_public = {
            name: expected[name]
            for name in ("task_id", "entry_point", "parameter", "task_sha256", "prompt")
        }
        if observed != expected_public:
            raise RuntimeError(f"visible task source record changed for {key!r}: {observed!r}")
        if len(task.private.regression_cases) != 4 or len(task.private.property_cases) != 32:
            raise RuntimeError(f"visible sandbox case counts changed for {key!r}")
        selected[key] = task
    return protocol, tuple(selected[key] for key in TRAIN_KEYS), selected[DEVELOPMENT_KEY]


def _task_fields(task: GeneratedTask) -> _TaskFields:
    expected = EXPECTED_SOURCE_RECORDS[(task.public.family, task.public.ordinal)]
    lower = expected["lower"]
    upper = expected["upper"]
    return _TaskFields(
        entry_point=task.public.signature.entry_point.encode("ascii"),
        parameter=task.public.signature.parameters[0].name.encode("ascii"),
        lower=None if lower is None else str(lower).encode("ascii"),
        upper=None if upper is None else str(upper).encode("ascii"),
    )


def _canonical_literal_primitives(literal: bytes) -> tuple[str, ...]:
    text = literal.decode("ascii")
    signed = text if text.startswith("-") else f"+{text}"
    if signed == "-0" or not signed[1:].isdigit():
        raise RuntimeError(f"non-canonical visible literal {text!r}")
    return tuple(signed)


def _prompt_spans(
    task: GeneratedTask,
) -> tuple[bytes, tuple[tuple[int, int, int, tuple[int, ...]], ...]]:
    prompt = task.public.prompt.encode("ascii")
    fields = _task_fields(task)
    signature = b"def " + fields.entry_point + b"(" + fields.parameter + b"):"
    signature_start = prompt.index(signature)
    entry_start = signature_start + len(b"def ")
    parameter_start = entry_start + len(fields.entry_point) + 1
    spans: list[tuple[int, int, int, tuple[int, ...]]] = [
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
    ]
    if task.public.family == CLAMP_FAMILY:
        if fields.lower is None or fields.upper is None:
            raise RuntimeError("clamp task source record has no bounds")
        anchor = b"inclusive range "
        lower_start = prompt.index(anchor, parameter_start) + len(anchor)
        lower_end = lower_start + len(fields.lower)
        through = b" through "
        if prompt[lower_start:lower_end] != fields.lower:
            raise RuntimeError("visible lower-bound span no longer matches")
        if prompt[lower_end : lower_end + len(through)] != through:
            raise RuntimeError("visible clamp separator no longer matches")
        upper_start = lower_end + len(through)
        upper_end = upper_start + len(fields.upper)
        if prompt[upper_start:upper_end] != fields.upper:
            raise RuntimeError("visible upper-bound span no longer matches")
        spans.extend(
            (
                (
                    lower_start,
                    lower_end,
                    2,
                    tuple(
                        PRIMITIVE_TOKEN_IDS[token]
                        for token in _canonical_literal_primitives(fields.lower)
                    ),
                ),
                (
                    upper_start,
                    upper_end,
                    3,
                    tuple(
                        PRIMITIVE_TOKEN_IDS[token]
                        for token in _canonical_literal_primitives(fields.upper)
                    ),
                ),
            )
        )
    elif task.public.family != NONNEGATIVE_FAMILY:
        raise RuntimeError("unsupported typed-clamp family")
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
            if active_slot in {2, 3}:
                signed = "".join(literal_parts)
                raw = signed[1:] if signed.startswith("+") else signed
                output.extend(raw.encode("ascii"))
            active_slot = None
            continue
        if active_slot in {2, 3}:
            try:
                literal_parts.append(PRIMITIVE_BY_TOKEN_ID[token_id])
            except KeyError as error:
                raise RuntimeError("non-primitive token in bound slot") from error
        else:
            if not 0 <= token_id < 128:
                raise RuntimeError("non-ASCII token outside canonical bound slot")
            output.append(token_id)
    if active_slot is not None:
        raise RuntimeError("unterminated prompt slot")
    return bytes(output)


def _tokenized_prompt(task: GeneratedTask) -> tuple[int, ...]:
    prompt, spans = _prompt_spans(task)
    tokens: list[int] = []
    cursor = 0
    for start, end, slot, replacement in spans:
        if start < cursor or end <= start:
            raise RuntimeError("task prompt spans overlap")
        tokens.extend(prompt[cursor:start])
        tokens.append(slot_start_token_id(slot))
        tokens.extend(replacement)
        tokens.append(slot_end_token_id(slot))
        cursor = end
    tokens.extend(prompt[cursor:])
    result = tuple(tokens)
    if _detokenize_prompt(result) != prompt:
        raise RuntimeError("typed prompt tokenizer is not a lossless public-wording view")
    return result


def _primitive_id(token: str) -> int:
    return PRIMITIVE_TOKEN_IDS[token]


def _internal_target(family: str) -> tuple[int, ...]:
    if family == NONNEGATIVE_FAMILY:
        primitives: tuple[str | tuple[str, int], ...] = (
            "if",
            "less-than",
            "variable-0",
            "integer",
            "+",
            "0",
            "end",
            "integer",
            "+",
            "0",
            "end",
            "variable-0",
        )
    elif family == CLAMP_FAMILY:
        primitives = (
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
        )
    else:
        raise RuntimeError("unsupported typed-clamp family")
    return (
        *(
            copy_slot_token_id(item[1])
            if isinstance(item, tuple)
            else _primitive_id(item)
            for item in primitives
        ),
        base.EOS_TOKEN_ID,
    )


def _expected_program_tokens(task: GeneratedTask) -> tuple[str, ...]:
    fields = _task_fields(task)
    if task.public.family == NONNEGATIVE_FAMILY:
        return (
            "if",
            "less-than",
            "variable-0",
            "integer",
            "+",
            "0",
            "end",
            "integer",
            "+",
            "0",
            "end",
            "variable-0",
        )
    if task.public.family == CLAMP_FAMILY:
        if fields.lower is None or fields.upper is None:
            raise RuntimeError("clamp task has no expected bounds")
        lower = _canonical_literal_primitives(fields.lower)
        upper = _canonical_literal_primitives(fields.upper)
        return (
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
        )
    raise RuntimeError("unsupported typed-clamp family")


def _expected_compiled_source(task: GeneratedTask) -> str:
    fields = _task_fields(task)
    entry = fields.entry_point.decode("ascii")
    parameter = fields.parameter.decode("ascii")
    if task.public.family == NONNEGATIVE_FAMILY:
        return f"def {entry}({parameter}):\n    return 0 if {parameter} < 0 else {parameter}\n"
    if task.public.family == CLAMP_FAMILY:
        if fields.lower is None or fields.upper is None:
            raise RuntimeError("clamp task has no expected bounds")
        lower = fields.lower.decode("ascii")
        upper = fields.upper.decode("ascii")
        source = (
            f"def {entry}({parameter}):\n"
            f"    return {lower} if {parameter} < {lower} else "
            f"{upper} if {upper} < {parameter} else {parameter}\n"
        )
        if (task.public.family, task.public.ordinal) == DEVELOPMENT_KEY:
            if source != DEVELOPMENT_EXPECTED_SOURCE:
                raise RuntimeError("pre-registered development source changed")
            return DEVELOPMENT_EXPECTED_SOURCE
        return source
    raise RuntimeError("unsupported typed-clamp family")


def _task_record(task: GeneratedTask) -> dict[str, object]:
    fields = _task_fields(task)
    prompt_tokens = _tokenized_prompt(task)
    return {
        "task_id": task.public.task_id,
        "partition": task.public.partition,
        "family": task.public.family,
        "ordinal": task.public.ordinal,
        "entry_point": fields.entry_point.decode("ascii"),
        "parameter": fields.parameter.decode("ascii"),
        "lower": None if fields.lower is None else fields.lower.decode("ascii"),
        "upper": None if fields.upper is None else fields.upper.decode("ascii"),
        "task_sha256": task.public.task_sha256,
        "hidden_test_sha256": task.public.hidden_test_sha256,
        "public_prompt_sha256": hashlib.sha256(
            task.public.prompt.encode("utf-8")
        ).hexdigest(),
        "public_prompt_lossless_after_tokenization": (
            _detokenize_prompt(prompt_tokens) == task.public.prompt.encode("ascii")
        ),
        "prompt_token_count": len(prompt_tokens),
        "regression_case_count": len(task.private.regression_cases),
        "property_case_count": len(task.private.property_cases),
    }


def _definition(
    train_tasks: tuple[GeneratedTask, ...],
    development_task: GeneratedTask,
) -> NativeTemporalCircuitDefinition:
    """Build one codebook with private plan, cue, and address assemblies."""

    train_prompts = tuple(_tokenized_prompt(task) for task in train_tasks)
    development_prompt = _tokenized_prompt(development_task)
    train_targets = tuple(_internal_target(task.public.family) for task in train_tasks)
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

    # Prompt, two plan commits, and teacher arrivals are the only learning
    # arrivals. Address cues and copied surfaces are read-only physical work.
    train_learning_arrivals = tuple(
        len(prompt) + PLAN_COMPONENT_COUNT + len(target)
        for prompt, target in zip(train_prompts, train_targets, strict=True)
    )
    segment_capacity = (
        sum(arrivals - 1 for arrivals in train_learning_arrivals)
        * base.ACTIVE_COLUMNS
    )
    if tuple(map(len, train_prompts)) != (270, 273):
        raise RuntimeError("frozen typed-clamp prompt token counts changed")
    if tuple(map(len, train_targets)) != (13, 20):
        raise RuntimeError("frozen typed-clamp teacher counts changed")
    if train_learning_arrivals != (285, 295) or segment_capacity != 6_358:
        raise RuntimeError("frozen typed-clamp segment-capacity accounting changed")
    if _RESERVED_CODE_COUNT != 37 or _RESERVED_COLUMN_COUNT != 407:
        raise RuntimeError("frozen typed-clamp private-code accounting changed")

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
    if remaining_column_count != 105:
        raise RuntimeError("typed-clamp private reserve no longer leaves 105 columns")

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
        token_id: token_codes[token_id] for token_id in TYPED_PRIVATE_TOKEN_IDS
    }
    if len({column for code in reserved_codes.values() for column in code}) != 407:
        raise RuntimeError("typed-clamp private code ranges overlap")
    if any(
        column < _RESERVED_COLUMN_COUNT
        for token_id, code in token_codes.items()
        if token_id not in reserved_codes
        for column in code
    ):
        raise RuntimeError("prompt/surface code entered the typed-clamp reserve")

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


def _expected_copy_schedule(task: GeneratedTask) -> tuple[tuple[int, tuple[str, ...]], ...]:
    fields = _task_fields(task)
    if fields.lower is None or fields.upper is None:
        return ()
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
    previous_event_count: int | None = None
    for generation in generations:
        path = str(generation["path"])
        event_count = int(generation["temporal_event_count"])
        if path == TemporalGenerationPath.ENTER_COPY.name:
            exact &= active is None and previous_event_count is not None
            enter_delta = (
                None if previous_event_count is None else event_count - previous_event_count
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
                    int(value) == enter_count for value in active["copied_event_counts"]
                )
                active["exit_resumed_temporal_after_typed_cue"] = exit_delta > 0
                transactions.append(active)
                active = None
        previous_event_count = event_count

    exact &= active is None and len(transactions) == len(expected)
    for transaction, (slot, primitives) in zip(transactions, expected, strict=False):
        exact &= transaction["slot_ordinal"] == slot
        exact &= transaction["control_token_id"] == copy_slot_token_id(slot)
        exact &= transaction["copied_places"] == list(range(len(primitives)))
        exact &= transaction["copied_primitives"] == list(primitives)
        exact &= transaction["exit_surface_primitive"] == "end"
        exact &= bool(transaction["enter_advanced_temporal_after_typed_cue"])
        exact &= bool(transaction["temporal_event_count_held_across_copied_tokens"])
        exact &= bool(transaction["exit_resumed_temporal_after_typed_cue"])
        exact &= all(
            int(value) >= 1
            for value in transaction["copied_physical_election_counts"]
        )
        exact &= all(int(value) >= 1 for value in transaction["copied_candidate_counts"])
    return transactions, exact


def _development_runtime_input(public_task: object) -> _DevelopmentRuntimeInput:
    signature = public_task.signature
    return _DevelopmentRuntimeInput(
        prompt_tokens=_tokenized_prompt_from_public(public_task),
        parameter_type=signature.parameters[0].type,
        result_type=signature.result,
    )


def _tokenized_prompt_from_public(public_task: object) -> tuple[int, ...]:
    """Reject any accidental private wrapper at the DEV runtime boundary."""

    if type(public_task).__name__ != "PublicTask":
        raise TypeError("development runtime input requires the exact public task view")
    key = (public_task.family, public_task.ordinal)
    if key != DEVELOPMENT_KEY:
        raise RuntimeError("development runtime input key changed")
    expected = EXPECTED_SOURCE_RECORDS[key]
    entry = str(expected["entry_point"]).encode("ascii")
    parameter = str(expected["parameter"]).encode("ascii")
    lower = str(expected["lower"]).encode("ascii")
    upper = str(expected["upper"]).encode("ascii")
    prompt = public_task.prompt.encode("ascii")
    signature = b"def " + entry + b"(" + parameter + b"):"
    signature_start = prompt.index(signature)
    entry_start = signature_start + 4
    parameter_start = entry_start + len(entry) + 1
    anchor = b"inclusive range "
    lower_start = prompt.index(anchor, parameter_start) + len(anchor)
    upper_start = lower_start + len(lower) + len(b" through ")
    spans = (
        (entry_start, entry_start + len(entry), 0, tuple(entry)),
        (parameter_start, parameter_start + len(parameter), 1, tuple(parameter)),
        (
            lower_start,
            lower_start + len(lower),
            2,
            tuple(PRIMITIVE_TOKEN_IDS[token] for token in _canonical_literal_primitives(lower)),
        ),
        (
            upper_start,
            upper_start + len(upper),
            3,
            tuple(PRIMITIVE_TOKEN_IDS[token] for token in _canonical_literal_primitives(upper)),
        ),
    )
    tokens: list[int] = []
    cursor = 0
    for start, end, slot, replacement in spans:
        tokens.extend(prompt[cursor:start])
        tokens.append(slot_start_token_id(slot))
        tokens.extend(replacement)
        tokens.append(slot_end_token_id(slot))
        cursor = end
    tokens.extend(prompt[cursor:])
    result = tuple(tokens)
    if _detokenize_prompt(result) != prompt:
        raise RuntimeError("public-only DEV tokenizer is not lossless")
    return result


def _generate_development(
    circuit: NativeTemporalCircuit,
    runtime_input: _DevelopmentRuntimeInput,
) -> tuple[_WorkspaceEpisode, _WorkspaceTrace | None, dict[str, object] | None]:
    episode = _TypedWorkspaceApi.begin_development(circuit, runtime_input)
    try:
        trace = _TypedWorkspaceApi.generate(circuit)
    except NativeTemporalRuntimeError as error:
        return episode, None, _generation_failure(error, 0)
    return episode, trace, None


def _score_development(
    *,
    episode: _WorkspaceEpisode,
    workspace_trace: _WorkspaceTrace | None,
    native_failure: dict[str, object] | None,
    task: GeneratedTask,
    protocol: object,
    python_executable: Path,
    runtime_identity: dict[str, str],
    worker_source: bytes,
    worker_sha256: str,
) -> tuple[dict[str, object], dict[str, object] | None]:
    expected_tokens = _expected_program_tokens(task)
    expected_surface_ids = tuple(
        PRIMITIVE_TOKEN_IDS[token] for token in expected_tokens
    ) + (base.EOS_TOKEN_ID,)
    generations: list[dict[str, object]] = []
    emitted_tokens: list[str] = []
    first_failure = native_failure
    terminated_by_eos = False

    if workspace_trace is not None:
        for position, generation in enumerate(workspace_trace.generations):
            generations.append(base._generation_record(position, generation))
        observed_surface_ids = tuple(
            int(generation.surface_token_id) for generation in workspace_trace.generations
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

    transactions, copy_trace_exact = _copy_transactions(
        generations, _expected_copy_schedule(task)
    )
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
            "kind": "model",
            "gate": "BOUND_LITERAL_PHYSICAL_COPY",
            "detail": transactions,
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
                    "kind": "model",
                    "gate": "EXACT_SIGNATURE_PYTHON",
                    "detail": None,
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
            "task": _task_record(task),
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
            "private_cases_entered_runtime": False,
            "expected_stream_used_only_for_host_scoring": True,
            "completed_generations": len(generations),
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


def _teacher_label(token: int) -> str:
    for slot in (2, 3):
        if token == copy_slot_token_id(slot):
            return f"COPY_SLOT_{slot}"
    if token == base.EOS_TOKEN_ID:
        return "<EOS>"
    return PRIMITIVE_BY_TOKEN_ID[token]


def main() -> int:
    args = _parse_args()
    _assert_source_committed_and_output_absent(args.output)
    started_ns = time.perf_counter_ns()
    protocol, train_tasks, development_task = _select_and_confirm_tasks()
    compiler_checks = [_compiler_check(task) for task in train_tasks]
    definition = _definition(train_tasks, development_task)

    executable = resolve_python_executable(args.python_executable)
    runtime_identity = python_runtime_identity(executable, protocol.raw.get("language"))
    if runtime_identity["python_version"] != "3.11.15":
        raise RuntimeError("typed-clamp probe requires explicit repo Python 3.11.15")
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = hashlib.sha256(worker_path.read_bytes()).hexdigest()
    worker_source = load_worker_snapshot(worker_sha256)

    # This immutable value is the whole DEV-to-native authority surface.
    development_runtime_input = _development_runtime_input(development_task.public)
    train_records: list[dict[str, object]] = []
    with base._public_development_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            for task in train_tasks:
                signature = task.public.signature
                teacher = _internal_target(task.public.family)
                episode = _TypedWorkspaceApi.train_task(
                    circuit,
                    prompt_tokens=_tokenized_prompt(task),
                    parameter_type=signature.parameters[0].type,
                    result_type=signature.result,
                    teacher_tokens=teacher,
                )
                steps = (*episode.prompt_steps, *episode.workspace_steps)
                train_records.append(
                    {
                        **_task_record(task),
                        "steps": len(steps),
                        "prompt_steps": len(episode.prompt_steps),
                        "workspace_steps": len(episode.workspace_steps),
                        "allocated_segments": sum(
                            step.allocated_segments for step in steps
                        ),
                        "workspace_step_token_ids": [
                            step.token_id for step in episode.workspace_steps
                        ],
                        "workspace_step_event_counts": [
                            step.event_count for step in episode.workspace_steps
                        ],
                        "internal_target": [_teacher_label(token) for token in teacher],
                    }
                )
            trained_identity = circuit.checkpoint_identity()
            development_episode, development_trace, native_failure = (
                _generate_development(circuit, development_runtime_input)
            )
        native_identity = loaded_library_identity()

    # Expected tokens and private cases enter only after atomic generation and
    # after the native circuit has closed.
    development_record, first_failure = _score_development(
        episode=development_episode,
        workspace_trace=development_trace,
        native_failure=native_failure,
        task=development_task,
        protocol=protocol,
        python_executable=executable,
        runtime_identity=runtime_identity,
        worker_source=worker_source,
        worker_sha256=worker_sha256,
    )

    passed = first_failure is None
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-native-typed-clamp",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": (
            "PASS_NATIVE_TYPED_CLAMP" if passed else "STOP_NATIVE_TYPED_CLAMP"
        ),
        "capability_pass": passed,
        "first_failure": first_failure,
        "claim_boundary": (
            "One retained native temporal circuit sees visible scalar-nonnegative/000 "
            "and scalar-clamp/000, then attempts visible scalar-clamp/007 exactly once. "
            "A pass shows teacher-free same-family nested typed-program replay with two "
            "unseen physical bound values copied through generic structural addresses. "
            "It is not family-disjoint transfer, M0, eval evidence, broad coding "
            "competence, or C1."
        ),
        "authority": {
            "source_partition_generated": "train",
            "eval_partition_generated": False,
            "train_keys": [list(key) for key in TRAIN_KEYS],
            "development_key": list(DEVELOPMENT_KEY),
            "development_attempts_completed": 1,
            "model_retries": 0,
            "harness_retries": 0,
            "atomic_generation_received_python_feedback": False,
            "development_target_used_as_runtime_input": False,
            "development_private_cases_used_as_runtime_input": False,
            "teacher_tokens_after_development_prompt": 0,
            "development_runtime_fields": [
                "lossless_prompt_tokens",
                "public_parameter_type",
                "public_result_type",
            ],
            "prompt_family_ids_injected": False,
            "prompt_solution_labels_injected": False,
            "prompt_bracket_scope": (
                "signature-entry/signature-parameter/lower-and-upper-bounds-when-present"
            ),
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
        "typed_programs": {
            "nonnegative_train": (
                "if less-than variable-0 integer + 0 end integer + 0 end variable-0"
            ),
            "clamp_train": (
                "if less-than variable-0 integer COPY[2] end integer COPY[2] end "
                "if less-than integer COPY[3] end variable-0 integer COPY[3] end "
                "variable-0"
            ),
            "clamp_development_expected": " ".join(
                _expected_program_tokens(development_task)
            ),
            "development_stream_equals_a_train_stream": False,
            "compiler_checks": compiler_checks,
        },
        "slot_bank": {
            "slot_count": SLOT_COUNT,
            "entry_slot_ordinal": 0,
            "parameter_slot_ordinal": 1,
            "lower_bound_slot_ordinal": 2,
            "upper_bound_slot_ordinal": 3,
            "typed_copy_bound_slots_only": True,
            "expected_copy_slot_schedule": [2, 2, 3, 3],
            "max_surface_tokens_per_slot": 22,
            "copy_control_token_ids": [copy_slot_token_id(2), copy_slot_token_id(3)],
            "copy_end_token_id": COPY_END_TOKEN_ID,
            "nonnegative_zero_word_left_as_lossless_ascii": True,
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
            "typed_surface_token_ids": list(TYPED_SURFACE_TOKEN_IDS),
            "address_token_base": ADDRESS_TOKEN_BASE,
            "address_offsets": list(ADDRESS_OFFSETS),
            "address_token_ids": list(ADDRESS_TOKEN_IDS),
            "typed_private_token_ids": list(TYPED_PRIVATE_TOKEN_IDS),
            "workspace_reserved_column_count": _RESERVED_COLUMN_COUNT,
            "prompt_surface_column_count": base.COLUMN_COUNT - _RESERVED_COLUMN_COUNT,
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
