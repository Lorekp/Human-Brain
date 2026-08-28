"""Prepared one-shot self-posed Simple-SWE shift-then-sum probe.

One retained native temporal circuit learns exactly untouched public TRAIN
t1_list_map_add/004 and t2_clamp_then_sum/002. It then receives the frozen
self-posed t2_shift_then_sum/000 task exactly once without a teacher. The
target family is absent from teaching and its +6 literal is absent from both
donors. Importing or compiling this source spends nothing; execution is
refused until the source is committed and its output artifact does not exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

# The probe is directly executable, so its local source roots are installed
# before repository imports below.
# ruff: noqa: E402


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
from humanbrain.simple_swe.tasks import (
    GeneratedTask,
    HiddenCase,
    PrivateTask,
    PublicTask,
    canonical_sha256,
    generate_visible_partition,
)
from humanbrain.typed_program import Program, ProgramParameter, ProgramType, TypedSignature
from humanbrain.typed_program import compile as compile_typed_program

OUTPUT_PATH = Path(__file__).with_name(
    "public_simple_swe_native_typed_shift_sum_probe.json"
)
PROTOCOL_PATH = REPO_ROOT / "benchmarks" / "simple_swe_protocol_v1.json"
LIST_ADD_KEY = ("t1_list_map_add", 4)
CLAMP_SUM_KEY = ("t2_clamp_then_sum", 2)
TRAIN_KEYS = (LIST_ADD_KEY, CLAMP_SUM_KEY)
TARGET_KEY = ("t2_shift_then_sum", 0)
EXPECTED_PROTOCOL_ID = "human-cell-simple-swe-v1"
EXPECTED_PROTOCOL_SHA256 = (
    "f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2"
)
EXPECTED_SOURCE_RECORDS: dict[tuple[str, int], dict[str, str | None]] = {
    LIST_ADD_KEY: {
        "task_id": "simple-swe-v1/train/tier-1/t1_list_map_add/004",
        "entry_point": "solve_fa9aceee8d7adbba",
        "parameter": "values",
        "literal": "5",
        "lower": None,
        "upper": None,
        "task_sha256": "fc7b98d7831a68957ad65a7b908426c2a773f61e779bddaded8158f3a5321219",
        "hidden_test_sha256": "5e94bb98cb96f1046c2910af74747a6458da1eb7434f3109e34ba25c9a92f7d0",
        "public_prompt_sha256": "7a09c8ebba45e7b3dd885fd97e046def672c49883a2dfc2bf54346a616f55c67",
    },
    CLAMP_SUM_KEY: {
        "task_id": "simple-swe-v1/train/tier-2/t2_clamp_then_sum/002",
        "entry_point": "solve_7994ea445e2aa4d7",
        "parameter": "values",
        "literal": None,
        "lower": "-4",
        "upper": "4",
        "task_sha256": "90c8dbb8627b6167c441440812327140f512061278fae5e386c901b6c75fd004",
        "hidden_test_sha256": "2075d15caca60a7470d2e6a3b52f8705b0ad15c5020786241c81a070e75f8866",
        "public_prompt_sha256": "e6b84dfaef34d704cf74e9649e8192379fa04745d4020f5202a381d6cf1730d4",
    },
    TARGET_KEY: {
        "task_id": "self-posed-simple-swe-v1/train/tier-2/t2_shift_then_sum/000",
        "entry_point": "solve",
        "parameter": "values",
        "literal": "6",
        "lower": None,
        "upper": None,
        "task_sha256": "1ea4917cff833393152730f9bc447b36c6f5097b6375afcf9bbcde5594446d2b",
        "hidden_test_sha256": "720402b4bfa1672ee3da8c5883fe33e2691fb74103caf4c3ffebd078f7acfa2a",
        "public_prompt_sha256": "933b5e9bdc7b15442f6e34264657ce2cf5586fbe117b22cb73cc7cef857dcbe5",
    },
}
EXPECTED_PROMPT_TOKEN_COUNTS = {
    LIST_ADD_KEY: 281,
    CLAMP_SUM_KEY: 284,
    TARGET_KEY: 262,
}
EXPECTED_TARGET_TOKENS = (
    "sum",
    "map",
    "bind",
    "add",
    "variable-0",
    "integer",
    "+",
    "6",
    "end",
    "variable-0",
)
EXPECTED_TARGET_SOURCE = (
    "def solve(values):\n"
    "    return sum([item_0 + 6 for item_0 in values])\n"
)

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
# Physical begin-session Learn arrivals by exact public teacher signature.
PLAN_COMPONENT_ARRIVALS = {LIST_ADD_KEY: 5, CLAMP_SUM_KEY: 6}
PHYSICAL_LEARN_STEPS_PER_ELIGIBLE_TEACHER = 2
EXPECTED_TEACHER_CALL_COUNTS = {LIST_ADD_KEY: 9, CLAMP_SUM_KEY: 24}
# Every non-EOS teacher call is eligible after scaffold re-rooting.  Each such
# call performs exactly two physical Learn steps total (transfer, then exact),
# and advances TypedHoleState in memory only once.
EXPECTED_ELIGIBLE_TEACHER_CALL_COUNTS = {LIST_ADD_KEY: 8, CLAMP_SUM_KEY: 23}
EXPECTED_PHYSICAL_TEACHER_ARRIVALS = {LIST_ADD_KEY: 17, CLAMP_SUM_KEY: 47}
EXPECTED_EPISODE_LEARNING_ARRIVALS = {LIST_ADD_KEY: 303, CLAMP_SUM_KEY: 337}
EXPECTED_TOTAL_LEARNING_ARRIVALS = 640
EXPECTED_NONINITIAL_LEARNING_ARRIVALS = 638
EXPECTED_SEGMENT_CAPACITY = 7018
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


# Union of scalar Add, Sum/Map/Bind/clamp, and root Map/Bind/Add structural
# holes.  No family, primitive, or target label enters this generic address set.
ADDRESS_PATHS = (
    (),
    (0,),
    (1,),
    (0, 0),
    (0, 1),
    (0, 0, 0),
    (0, 0, 1),
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
if ADDRESS_OFFSETS != (
    0,
    1,
    2,
    4,
    5,
    13,
    14,
    40,
    41,
    42,
    121,
    122,
    127,
    128,
    129,
    382,
    383,
):
    raise RuntimeError("typed scaffold re-root structural address set changed")
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
    raise RuntimeError("typed scaffold re-root operation cue ordinals changed")
if (
    len(TYPED_SURFACE_TOKEN_IDS) != 23
    or TYPED_SURFACE_TOKEN_IDS[22] != PRIMITIVE_TOKEN_IDS["sum"]
):
    raise RuntimeError("typed scaffold re-root requires 23 surfaces with Sum at ordinal 22")
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
    raise RuntimeError("typed scaffold re-root workspace received an unsupported public value")


@dataclass(frozen=True, slots=True)
class _TaskFields:
    entry_point: bytes
    parameter: bytes
    literal: bytes | None
    lower: bytes | None
    upper: bytes | None


@dataclass(frozen=True, slots=True)
class _WorkspaceEpisode:
    prompt_steps: tuple[object, ...]
    workspace_steps: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _TargetRuntimeInput:
    """The complete and only Python value allowed into target inference."""

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
    def begin_target(
        circuit: NativeTemporalCircuit,
        runtime_input: _TargetRuntimeInput,
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
        raise RuntimeError("typed scaffold re-root probe requires the four-slot transport")


def _build_target_task() -> GeneratedTask:
    signature = TypedSignature(
        entry_point="solve",
        parameters=(
            ProgramParameter(
                name="values",
                type=ProgramType.List(ProgramType.Int()),
            ),
        ),
        result=ProgramType.Int(),
    )
    description = "Add 6 to every integer in values, then return their sum."
    prompt = (
        "Return exactly one complete Python function. Use only Python built-ins; "
        "do not emit Markdown, prose, tests, imports, or input/output examples.\n"
        "Implement this exact task:\n"
        "def solve(values):\n"
        f'    """{description}"""'
    )

    def expected(values: list[int]) -> int:
        return sum(value + 6 for value in values)

    regressions = tuple(
        HiddenCase(
            args=(values,),
            expected=expected(values),
            distribution="fixed-regression",
        )
        for values in ([], [0], [-6, 1], [-3, 4, 10])
    )
    rng = random.Random(0x5A17000)
    properties: list[HiddenCase] = []
    for _ in range(32):
        values = [rng.randint(-80, 80) for _ in range(rng.randint(0, 18))]
        properties.append(
            HiddenCase(
                args=(values,),
                expected=expected(values),
                distribution="seeded-random-small",
            )
        )

    task_id = str(EXPECTED_SOURCE_RECORDS[TARGET_KEY]["task_id"])
    private_record = {
        "task_id": task_id,
        "signature": signature.to_record(),
        "regression_cases": [asdict(case) for case in regressions],
        "property_cases": [asdict(case) for case in properties],
    }
    hidden_test_sha256 = canonical_sha256(private_record)
    public_record = {
        "task_id": task_id,
        "partition": "train",
        "tier": 2,
        "family": TARGET_KEY[0],
        "ordinal": TARGET_KEY[1],
        "signature": signature.to_record(),
        "prompt": prompt,
        "hidden_test_sha256": hidden_test_sha256,
    }
    task_sha256 = canonical_sha256(public_record)
    return GeneratedTask(
        public=PublicTask(
            task_id=task_id,
            partition="train",
            tier=2,
            family=TARGET_KEY[0],
            ordinal=TARGET_KEY[1],
            signature=signature,
            prompt=prompt,
            hidden_test_sha256=hidden_test_sha256,
            task_sha256=task_sha256,
        ),
        private=PrivateTask(
            task_id=task_id,
            public_task_sha256=task_sha256,
            regression_cases=regressions,
            property_cases=tuple(properties),
            test_sha256=hidden_test_sha256,
        ),
    )


def _select_tasks() -> tuple[object, tuple[GeneratedTask, ...], GeneratedTask]:
    protocol = load_protocol(PROTOCOL_PATH)
    if (
        protocol.protocol_id != EXPECTED_PROTOCOL_ID
        or protocol.sha256 != EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError("public Simple-SWE protocol identity changed")
    visible_train = generate_visible_partition(protocol, "train")
    selected: dict[tuple[str, int], GeneratedTask] = {}
    for key in TRAIN_KEYS:
        family, ordinal = key
        try:
            task = next(
                item
                for item in visible_train
                if item.public.family == family and item.public.ordinal == ordinal
            )
        except StopIteration as error:
            raise RuntimeError(f"visible TRAIN task disappeared for {key!r}") from error
        _task_fields(task.public)
        if len(task.private.regression_cases) != 4 or len(task.private.property_cases) != 32:
            raise RuntimeError(f"visible sandbox case counts changed for {key!r}")
        selected[key] = task
    target = _build_target_task()
    _task_fields(target.public)
    if len(target.private.regression_cases) != 4 or len(target.private.property_cases) != 32:
        raise RuntimeError("self-posed target sandbox case counts changed")
    return protocol, tuple(selected[key] for key in TRAIN_KEYS), target


def _task_fields(public_task: object) -> _TaskFields:
    if type(public_task).__name__ != "PublicTask":
        raise TypeError("typed shift-sum parsing requires the exact public task view")
    key = (public_task.family, public_task.ordinal)
    if key not in EXPECTED_SOURCE_RECORDS:
        raise RuntimeError(f"unsupported typed shift-sum key {key!r}")
    expected = EXPECTED_SOURCE_RECORDS[key]
    signature = public_task.signature
    if len(signature.parameters) != 1:
        raise RuntimeError("visible public signature no longer has one parameter")
    expected_parameter_type = ProgramType.List(ProgramType.Int())
    expected_result_type = (
        ProgramType.List(ProgramType.Int()) if key == LIST_ADD_KEY else ProgramType.Int()
    )
    if (
        signature.parameters[0].type != expected_parameter_type
        or signature.result != expected_result_type
    ):
        raise RuntimeError(f"visible public signature changed for {key!r}")
    observed = {
        "task_id": public_task.task_id,
        "entry_point": signature.entry_point,
        "parameter": signature.parameters[0].name,
        "task_sha256": public_task.task_sha256,
        "hidden_test_sha256": public_task.hidden_test_sha256,
        "public_prompt_sha256": hashlib.sha256(
            public_task.prompt.encode("utf-8")
        ).hexdigest(),
    }
    expected_provenance = {
        name: expected[name]
        for name in (
            "task_id",
            "entry_point",
            "parameter",
            "task_sha256",
            "hidden_test_sha256",
            "public_prompt_sha256",
        )
    }
    if observed != expected_provenance:
        raise RuntimeError(f"visible task provenance changed for {key!r}: {observed!r}")
    entry = str(expected["entry_point"]).encode("ascii")
    parameter = str(expected["parameter"]).encode("ascii")
    literal = (
        None
        if expected["literal"] is None
        else str(expected["literal"]).encode("ascii")
    )
    lower = (
        None if expected["lower"] is None else str(expected["lower"]).encode("ascii")
    )
    upper = (
        None if expected["upper"] is None else str(expected["upper"]).encode("ascii")
    )
    prompt = public_task.prompt.encode("ascii")
    if public_task.partition != "train":
        raise RuntimeError(f"task left the public TRAIN partition for {key!r}")
    signature_bytes = b"def " + entry + b"(" + parameter + b"):"
    prompt.index(signature_bytes)
    if key == LIST_ADD_KEY:
        if literal is None:
            raise RuntimeError("list-add source record lost its literal")
        phrase = b"with " + literal + b" added to every integer"
    elif key == CLAMP_SUM_KEY:
        if lower is None or upper is None:
            raise RuntimeError("clamp-sum source record lost a bound")
        phrase = b"Limit every integer to " + lower + b" through " + upper
    else:
        if literal is None:
            raise RuntimeError("target source record lost its literal")
        phrase = b"Add " + literal + b" to every integer"
    if phrase not in prompt:
        raise RuntimeError(f"visible task wording changed for {key!r}")
    return _TaskFields(entry, parameter, literal, lower, upper)


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
    key = (public_task.family, public_task.ordinal)
    prompt = public_task.prompt.encode("ascii")
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
    if fields.literal is not None:
        anchor = b"with " if key == LIST_ADD_KEY else b"Add "
        literal_start = prompt.index(anchor, parameter_start) + len(anchor)
        if prompt[literal_start : literal_start + len(fields.literal)] != fields.literal:
            raise RuntimeError("visible literal span no longer matches")
        spans.append(
            (
                literal_start,
                literal_start + len(fields.literal),
                2,
                tuple(
                    PRIMITIVE_TOKEN_IDS[token]
                    for token in _canonical_literal_primitives(fields.literal)
                ),
            )
        )
    else:
        if fields.lower is None or fields.upper is None:
            raise RuntimeError("clamp-sum donor lost a bound")
        bound_anchor = b"Limit every integer to "
        lower_start = prompt.index(bound_anchor, parameter_start) + len(bound_anchor)
        upper_start = lower_start + len(fields.lower) + len(b" through ")
        spans.extend(
            (
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
        )
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
    key = (public_task.family, public_task.ordinal)
    if len(result) != EXPECTED_PROMPT_TOKEN_COUNTS[key]:
        raise RuntimeError(f"public prompt token count changed for {key!r}")
    return result


def _target_runtime_input(public_task: object) -> _TargetRuntimeInput:
    if (public_task.family, public_task.ordinal) != TARGET_KEY:
        raise RuntimeError("teacher-free target runtime key changed")
    signature = public_task.signature
    return _TargetRuntimeInput(
        prompt_tokens=_tokenized_public_prompt(public_task),
        parameter_type=signature.parameters[0].type,
        result_type=signature.result,
    )


def _internal_teacher_tokens(key: tuple[str, int]) -> tuple[int, ...]:
    if key == LIST_ADD_KEY:
        primitives: tuple[str | tuple[str, int], ...] = (
            "map",
            "bind",
            "add",
            "variable-0",
            "integer",
            ("copy", 2),
            "end",
            "variable-0",
        )
    elif key == CLAMP_SUM_KEY:
        primitives = (
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
    else:
        raise RuntimeError(f"unsupported teacher key {key!r}")
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
    train_prompts: dict[tuple[str, int], tuple[int, ...]],
    teachers: dict[tuple[str, int], tuple[int, ...]],
) -> dict[str, object]:
    if set(train_prompts) != set(TRAIN_KEYS) or set(teachers) != set(TRAIN_KEYS):
        raise RuntimeError("capacity accounting must contain exactly both teachers")
    observed_prompt_counts = {key: len(train_prompts[key]) for key in TRAIN_KEYS}
    expected_prompt_counts = {
        key: EXPECTED_PROMPT_TOKEN_COUNTS[key] for key in TRAIN_KEYS
    }
    if observed_prompt_counts != expected_prompt_counts:
        raise RuntimeError(
            f"teacher prompt token counts changed: {observed_prompt_counts!r}"
        )
    list_add_teacher = teachers[LIST_ADD_KEY]
    clamp_teacher = teachers[CLAMP_SUM_KEY]
    if len(list_add_teacher) != EXPECTED_TEACHER_CALL_COUNTS[LIST_ADD_KEY]:
        raise RuntimeError("list-add fixed teacher stream length changed")
    if list_add_teacher != (
        PRIMITIVE_TOKEN_IDS["map"],
        PRIMITIVE_TOKEN_IDS["bind"],
        PRIMITIVE_TOKEN_IDS["add"],
        PRIMITIVE_TOKEN_IDS["variable-0"],
        PRIMITIVE_TOKEN_IDS["integer"],
        copy_slot_token_id(2),
        PRIMITIVE_TOKEN_IDS["end"],
        PRIMITIVE_TOKEN_IDS["variable-0"],
        base.EOS_TOKEN_ID,
    ):
        raise RuntimeError("list-add fixed teacher stream changed")
    if len(clamp_teacher) != EXPECTED_TEACHER_CALL_COUNTS[CLAMP_SUM_KEY]:
        raise RuntimeError("clamp-sum fixed teacher stream length changed")
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
        clamp_teacher[0] != PRIMITIVE_TOKEN_IDS["sum"]
        or clamp_teacher[1] != PRIMITIVE_TOKEN_IDS["map"]
        or clamp_teacher[2] != PRIMITIVE_TOKEN_IDS["bind"]
        or clamp_teacher[3:22] != expected_nested_clamp
        or clamp_teacher[22] != PRIMITIVE_TOKEN_IDS["variable-0"]
        or clamp_teacher[23] != base.EOS_TOKEN_ID
    ):
        raise RuntimeError("clamp-sum fixed teacher stream boundaries changed")

    episode_records: dict[str, dict[str, int]] = {}
    total_learning_arrivals = 0
    noninitial_learning_arrivals = 0
    for key in TRAIN_KEYS:
        teacher_call_count = len(teachers[key])
        eligible_teacher_call_count = teacher_call_count - 1
        if eligible_teacher_call_count != EXPECTED_ELIGIBLE_TEACHER_CALL_COUNTS[key]:
            raise RuntimeError(f"eligible teacher count changed for {key!r}")
        physical_teacher_arrivals = (
            eligible_teacher_call_count * PHYSICAL_LEARN_STEPS_PER_ELIGIBLE_TEACHER
            + 1  # singleton EOS exact write
        )
        if physical_teacher_arrivals != EXPECTED_PHYSICAL_TEACHER_ARRIVALS[key]:
            raise RuntimeError(f"physical teacher arrivals changed for {key!r}")
        episode_learning_arrivals = (
            len(train_prompts[key])
            + PLAN_COMPONENT_ARRIVALS[key]
            + physical_teacher_arrivals
        )
        if episode_learning_arrivals != EXPECTED_EPISODE_LEARNING_ARRIVALS[key]:
            raise RuntimeError(f"learning arrivals changed for {key!r}")
        label = f"{key[0]}/{key[1]:03d}"
        episode_records[label] = {
            "prompt_arrivals": len(train_prompts[key]),
            "plan_component_arrivals": PLAN_COMPONENT_ARRIVALS[key],
            "teacher_api_calls": teacher_call_count,
            "eligible_teacher_api_calls": eligible_teacher_call_count,
            "physical_learn_steps_per_eligible_teacher_call": (
                PHYSICAL_LEARN_STEPS_PER_ELIGIBLE_TEACHER
            ),
            "singleton_eos_arrivals": 1,
            "physical_teacher_arrivals": physical_teacher_arrivals,
            "total_learning_arrivals": episode_learning_arrivals,
            "noninitial_learning_arrivals": episode_learning_arrivals - 1,
        }
        total_learning_arrivals += episode_learning_arrivals
        # reset_before=True makes the first prompt arrival of each episode the
        # only arrival that cannot allocate a predictive segment.
        noninitial_learning_arrivals += episode_learning_arrivals - 1

    if total_learning_arrivals != EXPECTED_TOTAL_LEARNING_ARRIVALS:
        raise RuntimeError("total physical Learn arrival count changed")
    if noninitial_learning_arrivals != EXPECTED_NONINITIAL_LEARNING_ARRIVALS:
        raise RuntimeError("noninitial physical Learn arrival count changed")
    segment_capacity = noninitial_learning_arrivals * base.ACTIVE_COLUMNS
    if segment_capacity != EXPECTED_SEGMENT_CAPACITY:
        raise RuntimeError("exact worst-case segment capacity changed")
    base_population = base.COLUMN_COUNT * base.BASE_CELLS_PER_COLUMN
    return {
        "episodes": episode_records,
        "eligible_teacher_call_physical_learn_steps": (
            PHYSICAL_LEARN_STEPS_PER_ELIGIBLE_TEACHER
        ),
        "total_learning_arrivals": total_learning_arrivals,
        "reset_before_episode_count": len(TRAIN_KEYS),
        "noninitial_learning_arrivals": noninitial_learning_arrivals,
        "segments_per_noninitial_arrival": base.ACTIVE_COLUMNS,
        "segment_capacity": segment_capacity,
        "base_pyramidal_cells": base_population,
        "reserve_pyramidal_cells": segment_capacity,
        "max_pyramidal_cells": base_population + segment_capacity,
    }


def _expected_program_tokens(public_task: object) -> tuple[str, ...]:
    fields = _task_fields(public_task)
    key = (public_task.family, public_task.ordinal)
    if key == LIST_ADD_KEY:
        if fields.literal is None:
            raise RuntimeError("list-add donor lost its literal")
        return (
            "map",
            "bind",
            "add",
            "variable-0",
            "integer",
            *_canonical_literal_primitives(fields.literal),
            "end",
            "variable-0",
        )
    if key == CLAMP_SUM_KEY:
        if fields.lower is None or fields.upper is None:
            raise RuntimeError("clamp-sum teacher lost a bound")
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
    if key == TARGET_KEY:
        if fields.literal != b"6":
            raise RuntimeError("teacher-free target literal changed")
        return EXPECTED_TARGET_TOKENS
    raise RuntimeError(f"unsupported expected program key {key!r}")


def _expected_compiled_source(public_task: object) -> str:
    fields = _task_fields(public_task)
    key = (public_task.family, public_task.ordinal)
    entry = fields.entry_point.decode("ascii")
    parameter = fields.parameter.decode("ascii")
    if key == LIST_ADD_KEY:
        if fields.literal is None:
            raise RuntimeError("list-add donor lost its literal")
        literal = fields.literal.decode("ascii")
        return (
            f"def {entry}({parameter}):\n"
            f"    return [item_0 + {literal} for item_0 in {parameter}]\n"
        )
    if key == CLAMP_SUM_KEY:
        if fields.lower is None or fields.upper is None:
            raise RuntimeError("clamp-sum teacher lost a bound")
        lower = fields.lower.decode("ascii")
        upper = fields.upper.decode("ascii")
        return (
            f"def {entry}({parameter}):\n"
            f"    return sum([{lower} if item_0 < {lower} else "
            f"{upper} if {upper} < item_0 else item_0 for item_0 in {parameter}])\n"
        )
    if key == TARGET_KEY:
        if (
            f"def {entry}({parameter}):\n"
            f"    return sum([item_0 + 6 for item_0 in {parameter}])\n"
        ) != EXPECTED_TARGET_SOURCE:
            raise RuntimeError("frozen target source is inconsistent with provenance")
        return EXPECTED_TARGET_SOURCE
    raise RuntimeError(f"unsupported expected source key {key!r}")


def _public_task_record(public_task: object) -> dict[str, object]:
    fields = _task_fields(public_task)
    key = (public_task.family, public_task.ordinal)
    prompt_tokens = _tokenized_public_prompt(public_task)
    if len(prompt_tokens) != EXPECTED_PROMPT_TOKEN_COUNTS[key]:
        raise RuntimeError(f"public prompt token count changed for {key!r}")
    return {
        "task_id": public_task.task_id,
        "partition": public_task.partition,
        "family": public_task.family,
        "ordinal": public_task.ordinal,
        "entry_point": fields.entry_point.decode("ascii"),
        "parameter": fields.parameter.decode("ascii"),
        "literal": (
            None if fields.literal is None else fields.literal.decode("ascii")
        ),
        "lower": None if fields.lower is None else fields.lower.decode("ascii"),
        "upper": None if fields.upper is None else fields.upper.decode("ascii"),
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
    train_tasks: tuple[GeneratedTask, ...],
    target_input: _TargetRuntimeInput,
) -> tuple[NativeTemporalCircuitDefinition, dict[str, object]]:
    if tuple((task.public.family, task.public.ordinal) for task in train_tasks) != TRAIN_KEYS:
        raise RuntimeError("definition must receive exactly the frozen teacher order")
    train_prompts = {
        key: _tokenized_public_prompt(task.public)
        for key, task in zip(TRAIN_KEYS, train_tasks, strict=True)
    }
    teachers = {key: _internal_teacher_tokens(key) for key in TRAIN_KEYS}
    control_tokens = {
        *(slot_start_token_id(slot) for slot in range(SLOT_COUNT)),
        *(slot_end_token_id(slot) for slot in range(SLOT_COUNT)),
        *(copy_slot_token_id(slot) for slot in range(SLOT_COUNT)),
        COPY_END_TOKEN_ID,
    }
    used_token_ids = tuple(
        sorted(
            {
                *(token for prompt in train_prompts.values() for token in prompt),
                *target_input.prompt_tokens,
                *PRIMITIVE_BY_TOKEN_ID,
                *control_tokens,
                *TYPED_PRIVATE_TOKEN_IDS,
                base.EOS_TOKEN_ID,
            }
        )
    )

    capacity_accounting = _learning_arrival_accounting(train_prompts, teachers)
    segment_capacity = int(capacity_accounting["segment_capacity"])
    config = NativeTemporalCircuitConfig(
        column_count=base.COLUMN_COUNT,
        max_pyramidal_cells=int(capacity_accounting["max_pyramidal_cells"]),
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
        raise RuntimeError("typed scaffold re-root private codes leave too few surface columns")

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
        raise RuntimeError("typed scaffold re-root private code ranges overlap")
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
    if fields.literal is not None:
        return ((2, _canonical_literal_primitives(fields.literal)),)
    if fields.lower is None or fields.upper is None:
        raise RuntimeError("copy schedule requires a literal or both bounds")
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


def _score_target(
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
    # This is the first point where the target solution or private cases may
    # enter Python, and the native circuit is already closed.
    expected_tokens = _expected_program_tokens(task.public)
    expected_surface_ids = (
        *(PRIMITIVE_TOKEN_IDS[token] for token in expected_tokens),
        base.EOS_TOKEN_ID,
    )
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
    fitted_cell_races_exact = bool(generations) and all(
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
            "target_expected_solution_entered_native_runtime": False,
            "private_cases_entered_native_runtime": False,
            "expected_stream_used_only_after_native_close": True,
            "terminated_by_ordinary_eos": terminated_by_eos,
            "program": program_record,
            "copy_transactions": transactions,
            "copy_slot_schedule": [
                slot for slot, _ in _expected_copy_schedule(task.public)
            ],
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
    protocol, train_tasks, target_task = _select_tasks()
    if {task.public.family for task in train_tasks} & {target_task.public.family}:
        raise RuntimeError("teacher-free target family must be absent from teaching")
    target_input = _target_runtime_input(target_task.public)
    teachers = {key: _internal_teacher_tokens(key) for key in TRAIN_KEYS}
    definition, capacity_accounting = _definition(train_tasks, target_input)

    executable = resolve_python_executable(args.python_executable)
    runtime_identity = python_runtime_identity(executable, protocol.raw.get("language"))
    if runtime_identity["python_version"] != "3.11.15":
        raise RuntimeError("scaffold re-root probe requires explicit repo Python 3.11.15")
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = hashlib.sha256(worker_path.read_bytes()).hexdigest()
    worker_source = load_worker_snapshot(worker_sha256)

    trained_episodes: list[tuple[GeneratedTask, _WorkspaceEpisode, object]] = []
    with base._public_development_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            for key, task in zip(TRAIN_KEYS, train_tasks, strict=True):
                signature = task.public.signature
                episode = _TypedWorkspaceApi.train_task(
                    circuit,
                    prompt_tokens=_tokenized_public_prompt(task.public),
                    parameter_type=signature.parameters[0].type,
                    result_type=signature.result,
                    teacher_tokens=teachers[key],
                )
                trained_episodes.append((task, episode, circuit.checkpoint_identity()))
            trained_identity = circuit.checkpoint_identity()
            target_episode = _TypedWorkspaceApi.begin_target(circuit, target_input)
            try:
                raw_generations = circuit.generate_typed()
                native_failure = None
            except NativeTemporalRuntimeError as error:
                raw_generations = None
                native_failure = _generation_failure(error)
        native_identity = loaded_library_identity()

    # Expected target access, every compiler check, and private sandbox cases
    # begin only after atomic generation and after the native circuit has closed.
    compiler_records: list[dict[str, object]] = []
    for task in train_tasks:
        tokens = _expected_program_tokens(task.public)
        compiled_source = compile_typed_program(
            Program(tokens, task.public.signature)
        ).source
        expected_source = _expected_compiled_source(task.public)
        if compiled_source != expected_source:
            raise RuntimeError(
                f"typed compiler cannot express teacher {task.public.task_id}"
            )
        compiler_records.append(
            {
                "task_id": task.public.task_id,
                "tokens": list(tokens),
                "compiled_source": compiled_source,
                "exact": True,
            }
        )
    target_record, first_failure = _score_target(
        episode=target_episode,
        raw_generations=raw_generations,
        native_failure=native_failure,
        task=target_task,
        protocol=protocol,
        python_executable=executable,
        runtime_identity=runtime_identity,
        worker_source=worker_source,
        worker_sha256=worker_sha256,
    )

    passed = first_failure is None
    train_records = [
        {
            "task": _public_task_record(task.public),
            "teacher": [
                _teacher_label(token_id)
                for token_id in teachers[(task.public.family, task.public.ordinal)]
            ],
            "prompt_steps": len(episode.prompt_steps),
            "workspace_api_calls": len(episode.workspace_steps),
            "reset_before": True,
            "live_segments_after_task": identity.segment_count,
            "learning_steps_after_task": identity.learning_step,
        }
        for task, episode, identity in trained_episodes
    ]
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-native-typed-shift-sum",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": (
            "PASS_NATIVE_TYPED_SHIFT_SUM"
            if passed
            else "STOP_NATIVE_TYPED_SHIFT_SUM"
        ),
        "capability_pass": passed,
        "first_failure": first_failure,
        "claim_boundary": (
            "One retained native temporal circuit learns untouched public TRAIN "
            "t1_list_map_add/004 and t2_clamp_then_sum/002, then attempts frozen "
            "self-posed t2_shift_then_sum/000 exactly once without a teacher. A "
            "pass shows Sum/Map/Bind and Add/literal donor subtrees re-root and "
            "compose with unseen +6. It is not a held-out benchmark, broad coding "
            "competence, M0, or C1."
        ),
        "authority": {
            "source_partition_generated": "train",
            "target_source": "frozen_self_posed",
            "eval_partition_generated": False,
            "train_keys": [list(key) for key in TRAIN_KEYS],
            "teacher_free_target_key": list(TARGET_KEY),
            "teacher_free_target_attempts_completed": 1,
            "model_retries": 0,
            "harness_retries": 0,
            "target_family_absent_from_teaching": True,
            "target_expected_solution_used_as_runtime_input": False,
            "target_private_cases_used_as_runtime_input": False,
            "teacher_tokens_after_target_prompt": 0,
            "target_runtime_fields": [
                "lossless_prompt_tokens",
                "public_parameter_type",
                "public_result_type",
            ],
            "atomic_no_argument_generation_call_count": 1,
            "caller_supplied_candidate_tokens": False,
            "caller_supplied_address_tokens": False,
            "caller_supplied_target_tokens": False,
            "python_candidate_enumeration": False,
            "python_argmax": False,
        },
        "protocol": {
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.sha256,
            "language": protocol.raw.get("language"),
        },
        "typed_programs": {
            "teacher_compiler_checks": compiler_records,
            "target_expected_tokens": list(EXPECTED_TARGET_TOKENS),
            "target_expected_source": EXPECTED_TARGET_SOURCE,
            "target_stream_differs_from_every_teacher_stream": all(
                _expected_program_tokens(task.public) != EXPECTED_TARGET_TOKENS
                for task in train_tasks
            ),
        },
        "factorized_plan": {
            "list_add_plan_component_arrivals": PLAN_COMPONENT_ARRIVALS[LIST_ADD_KEY],
            "two_bound_plan_component_arrivals": PLAN_COMPONENT_ARRIVALS[CLAMP_SUM_KEY],
            "parameter_signature_winners": 3,
            "result_signature_winners": 2,
            "cue_22": "neutral/absent-bound",
            "cue_23": "physical lower-bound slot 2 ready",
            "cue_24": "physical upper-bound slot 3 ready",
            "cue_25": "any-bound operation backoff",
            "typed_cue_count": len(TYPED_CUE_NAMES),
            "typed_surface_count": len(TYPED_SURFACE_TOKEN_IDS),
            "surface_22": "sum",
            "physical_learn_steps_per_eligible_teacher_call": 2,
            "typed_state_advances_per_teacher_call": 1,
        },
        "scaffold_reroot": {
            "selection_order": "exact then generic scaffold backoff",
            "generic_backoff_context": "structural address with one leading ordinal removed",
            "root_address_rebases_to_itself": True,
            "typed_hole_state_mutated_by_rebase": False,
            "family_primitive_or_target_cue": False,
            "target_add_teacher_occurrence_was_root_nested": True,
        },
        "slot_bank": {
            "slot_count": SLOT_COUNT,
            "entry_slot_ordinal": 0,
            "parameter_slot_ordinal": 1,
            "literal_or_lower_bound_slot_ordinal": 2,
            "upper_bound_slot_ordinal": 3,
            "teacher_copy_slot_schedules": {
                "t1_list_map_add/004": [2],
                "t2_clamp_then_sum/002": [2, 2, 3, 3],
            },
            "target_expected_copy_slot_schedule": [2],
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
            "tasks": train_records,
            "reset_before_each": True,
            "learned_state_retained_between_tasks": True,
            "live_segments_after_train": trained_identity.segment_count,
            "learning_steps_after_train": trained_identity.learning_step,
        },
        "self_posed_target_teacher_free": target_record,
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
