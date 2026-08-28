"""Prepared one-shot visible-TRAIN cross-donor clamp-then-sum probe.

One retained native temporal circuit learns exactly visible TRAIN
``t0_scalar_clamp/001`` and ``t2_map_then_count/002``.  It then receives
visible TRAIN ``t2_clamp_then_sum/003`` exactly once without a teacher.  The
probe asks whether the root Sum/Map/Bind scaffold and the descendant scalar
clamp subtree can be recombined.  Importing or compiling this source spends
nothing; execution is refused until this source is committed and the output
artifact is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path

# This probe is directly executable.  Install its repository roots before the
# imports below; the reusable typed workspace lives in the prior committed
# shift-sum probe and no production Python API is extended here.
# ruff: noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PROBE_DIR = Path(__file__).resolve().parent
for import_root in (SRC_ROOT, REPO_ROOT, PROBE_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import public_simple_swe_native_typed_shift_sum_probe as harness
from humanbrain.native import loaded_library_identity
from humanbrain.native_temporal_runtime import (
    COPY_END_TOKEN_ID,
    SLOT_COUNT,
    NativeTemporalCircuit,
    NativeTemporalRuntimeError,
    TemporalStepMode,
    copy_slot_token_id,
    slot_end_token_id,
    slot_start_token_id,
)
from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.sandbox import (
    execute_completion,
    load_worker_snapshot,
    python_runtime_identity,
    resolve_python_executable,
)
from humanbrain.simple_swe.tasks import GeneratedTask, generate_visible_partition
from humanbrain.typed_program import Program, ProgramType
from humanbrain.typed_program import compile as compile_typed_program

base = harness.base
PRIMITIVE_TOKEN_IDS = harness.PRIMITIVE_TOKEN_IDS
PRIMITIVE_BY_TOKEN_ID = harness.PRIMITIVE_BY_TOKEN_ID
PLAN_COMMIT_TOKEN_ID = harness.PLAN_COMMIT_TOKEN_ID
TYPED_CUE_NAMES = harness.TYPED_CUE_NAMES
TYPED_CUE_TOKEN_IDS = harness.TYPED_CUE_TOKEN_IDS
ADDRESS_TOKEN_BASE = harness.ADDRESS_TOKEN_BASE
TYPED_SURFACE_TOKEN_IDS = harness.TYPED_SURFACE_TOKEN_IDS
# Exact union of scalar-clamp, map-count, and clamp-sum absolute/rebased holes.
# The three programs require 24 address codes, so this probe uses 768 columns;
# 512 cannot hold 1 plan + 26 cue + 24 address disjoint 11-column assemblies.
COLUMN_COUNT = 768
ADDRESS_OFFSETS = (
    0,
    1,
    2,
    3,
    4,
    5,
    10,
    11,
    12,
    13,
    31,
    32,
    40,
    41,
    42,
    121,
    122,
    127,
    128,
    129,
    367,
    368,
    382,
    383,
)
ADDRESS_TOKEN_IDS = tuple(ADDRESS_TOKEN_BASE + offset for offset in ADDRESS_OFFSETS)
TYPED_PRIVATE_TOKEN_IDS = (
    PLAN_COMMIT_TOKEN_ID,
    *TYPED_CUE_TOKEN_IDS.values(),
    *ADDRESS_TOKEN_IDS,
)
_RESERVED_COLUMN_COUNT = len(TYPED_PRIVATE_TOKEN_IDS) * base.ACTIVE_COLUMNS
if _RESERVED_COLUMN_COUNT != 561 or COLUMN_COUNT - _RESERVED_COLUMN_COUNT != 207:
    raise RuntimeError("cross-donor private-code reserve changed")
_WorkspaceEpisode = harness._WorkspaceEpisode
_TargetRuntimeInput = harness._TargetRuntimeInput
_TypedWorkspaceApi = harness._TypedWorkspaceApi
_typed_value = harness._typed_value
_workspace_vocabulary = harness._workspace_vocabulary

OUTPUT_PATH = Path(__file__).with_suffix(".json")
SPEND_MARKER_PATH = Path(__file__).with_suffix(".spent.json")
PROTOCOL_PATH = REPO_ROOT / "benchmarks" / "simple_swe_protocol_v1.json"
HELPER_PATH = PROBE_DIR / "public_simple_swe_native_typed_shift_sum_probe.py"
EXPECTED_HELPER_SHA256 = (
    "206309e97caa37ec5bdcf15936bf02b1e78684074f0f601f3997cb5e99941b61"
)
NATIVE_MECHANISM_PATHS = (
    REPO_ROOT / "src" / "humanbrain" / "_native_src" / "src" / "temporal_runtime.rs",
    REPO_ROOT
    / "src"
    / "humanbrain"
    / "_native_src"
    / "src"
    / "temporal_runtime"
    / "checkpoint.rs",
    REPO_ROOT
    / "src"
    / "humanbrain"
    / "_native_src"
    / "src"
    / "temporal_runtime"
    / "typed_workspace.rs",
    REPO_ROOT
    / "src"
    / "humanbrain"
    / "_native_src"
    / "src"
    / "temporal_runtime"
    / "typed_hole_state.rs",
)
EXPECTED_NATIVE_MECHANISM_SHA256 = {
    "src/humanbrain/_native_src/src/temporal_runtime.rs": (
        "752dc9d752ef06cde8f3de30247c7842e54ac22b5892331566af4a92f1b04972"
    ),
    "src/humanbrain/_native_src/src/temporal_runtime/checkpoint.rs": (
        "bdb1ab444a826e706b9c8284816a6e54847b0bb6c156723135324fffb377faba"
    ),
    "src/humanbrain/_native_src/src/temporal_runtime/typed_hole_state.rs": (
        "e61a9e8101b5447ef245eda161e99a4e3631f1582b86f37213a36983d9774827"
    ),
    "src/humanbrain/_native_src/src/temporal_runtime/typed_workspace.rs": (
        "4d3ebde3d598e9b1f102c23169ae5d595666e9aef8737cd54822c566d363db1c"
    ),
}

SCALAR_CLAMP_KEY = ("t0_scalar_clamp", 1)
MAP_COUNT_KEY = ("t2_map_then_count", 2)
TRAIN_KEYS = (SCALAR_CLAMP_KEY, MAP_COUNT_KEY)
TARGET_KEY = ("t2_clamp_then_sum", 3)
EXPECTED_PROTOCOL_ID = "human-cell-simple-swe-v1"
EXPECTED_PROTOCOL_SHA256 = (
    "f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2"
)

EXPECTED_SOURCE_RECORDS: dict[tuple[str, int], dict[str, str | None]] = {
    SCALAR_CLAMP_KEY: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_clamp/001",
        "entry_point": "solve_170d49a7273d556b",
        "parameter": "n",
        "lower": "-8",
        "upper": "8",
        "delta": None,
        "threshold": None,
        "task_sha256": "a16137e73b71c16ffce2f5f4c1455e8e0601da3db7f0909577d5664ae63906cc",
        "hidden_test_sha256": "f59fde4993af7d14d68dd29d1db112ee76ce30893ece6a80fac7326d52520b07",
        "public_prompt_sha256": "9a56cd6184e4f047134ee3e9db0e912c4bc4caceaad42951681788de1443865d",
    },
    MAP_COUNT_KEY: {
        "task_id": "simple-swe-v1/train/tier-2/t2_map_then_count/002",
        "entry_point": "solve_da11bb06f61153ef",
        "parameter": "values",
        "lower": None,
        "upper": None,
        "delta": "-2",
        "threshold": "0",
        "task_sha256": "01cd6a2a8a9ec79d0c08bd0185e4490d6de062adfa793b4f89e652051bbe1cf0",
        "hidden_test_sha256": "7f4f22d6525e5fdf084713e51924dfd80446cfb217c3cd2b9fa00dc8d174d6e2",
        "public_prompt_sha256": "67b8362c12490beeb45f99f9d839ed3695552c191b5143a6a3d9f9013dce5f16",
    },
    TARGET_KEY: {
        "task_id": "simple-swe-v1/train/tier-2/t2_clamp_then_sum/003",
        "entry_point": "solve_11200de88aa84919",
        "parameter": "values",
        "lower": "-3",
        "upper": "3",
        "delta": None,
        "threshold": None,
        "task_sha256": "45c180458098a57635740f724c11a034dee634e85fa04fdbbf65b92f20537175",
        "hidden_test_sha256": "252b293e8dc41ba08135feedf0524015e316b45587820c8c3a1be01e17c13cb2",
        "public_prompt_sha256": "49a2bd955d232771590573bcfaeff3922c5d2ce41019abaa4f8e5719db8fdcf0",
    },
}

EXPECTED_PROMPT_TOKEN_COUNTS = {
    SCALAR_CLAMP_KEY: 273,
    MAP_COUNT_KEY: 290,
    TARGET_KEY: 284,
}
EXPECTED_TARGET_TOKENS = (
    "sum",
    "map",
    "bind",
    "if",
    "less-than",
    "variable-0",
    "integer",
    "-",
    "3",
    "end",
    "integer",
    "-",
    "3",
    "end",
    "if",
    "less-than",
    "integer",
    "+",
    "3",
    "end",
    "variable-0",
    "integer",
    "+",
    "3",
    "end",
    "variable-0",
    "variable-0",
)
EXPECTED_TARGET_SOURCE = (
    "def solve_11200de88aa84919(values):\n"
    "    return sum([-3 if item_0 < -3 else 3 if 3 < item_0 else "
    "item_0 for item_0 in values])\n"
)

# One exact and one generic transfer learn step per non-EOS teacher call.
PLAN_COMPONENT_ARRIVALS = {SCALAR_CLAMP_KEY: 6, MAP_COUNT_KEY: 6}
PHYSICAL_LEARN_STEPS_PER_ELIGIBLE_TEACHER = 2
EXPECTED_TEACHER_CALL_COUNTS = {SCALAR_CLAMP_KEY: 20, MAP_COUNT_KEY: 23}
EXPECTED_PHYSICAL_TEACHER_ARRIVALS = {SCALAR_CLAMP_KEY: 39, MAP_COUNT_KEY: 45}
EXPECTED_EPISODE_LEARNING_ARRIVALS = {SCALAR_CLAMP_KEY: 318, MAP_COUNT_KEY: 341}
EXPECTED_BASE_LEARNING_ARRIVALS = 659
EXPECTED_POINTER_CUE_LEARNING_ARRIVALS = 10
EXPECTED_TOTAL_LEARNING_ARRIVALS = 669
EXPECTED_BASE_NONINITIAL_LEARNING_ARRIVALS = 657
EXPECTED_NONINITIAL_LEARNING_ARRIVALS = 667
EXPECTED_BASE_SEGMENT_ROWS = 7_227

# Pointer integration is downstream of root election.  Its only capacity
# increment is at most 11 cue rows + 11 recurrence rows per call.  Exact replay
# allocates zero; the executed artifact derives actual recruitment from native
# segment deltas rather than assuming every contextual cue is novel.
ROLE_BEARING_CONSTRUCTORS = (
    "add",
    "absolute",
    "less-than",
    "if",
    "map",
    "sum",
    "bind",
)
MAX_POINTER_ROWS_PER_CUE_CALL = 22
EXPECTED_POINTER_CUE_SEGMENT_ROWS_UPPER_BOUND = 110
EXPECTED_POINTER_RECURRENCE_ROWS_UPPER_BOUND = 110
EXPECTED_POINTER_SEGMENT_ROWS_UPPER_BOUND = 220
EXPECTED_ORDINARY_SEGMENT_ROWS_UPPER_BOUND = 7_337
EXPECTED_SEGMENT_CAPACITY = 7_447


@dataclass(frozen=True, slots=True)
class _PointerOccurrence:
    teacher_position: int
    surface: str
    cue: str
    absolute_address: tuple[int, ...]
    relative_address: tuple[int, ...]
    parent_occurrence: int | None


# Exact constructor walk before each teacher token advances TypedHoleState.
# ``teacher_position`` identifies the opaque surface-winner context produced by
# the corresponding exact temporal transition; cue, parent pointer, and
# parent-relative address complete the native pointer-cue identity.
POINTER_OCCURRENCES_BY_DONOR = {
    SCALAR_CLAMP_KEY: (
        _PointerOccurrence(0, "if", "expression-int-head", (), (), None),
        _PointerOccurrence(1, "less-than", "expression-bool-child-0", (0,), (0,), 0),
        _PointerOccurrence(9, "if", "expression-int-child-2", (2,), (2,), 0),
        _PointerOccurrence(10, "less-than", "expression-bool-child-0", (2, 0), (0,), 2),
    ),
    MAP_COUNT_KEY: (
        _PointerOccurrence(0, "sum", "expression-int-head", (), (), None),
        _PointerOccurrence(1, "map", "expression-list-int-child-0", (0,), (0,), 0),
        _PointerOccurrence(2, "bind", "expression-int-to-int-child-0", (0, 0), (0,), 1),
        _PointerOccurrence(3, "if", "expression-int-head", (0, 0, 0), (), None),
        _PointerOccurrence(4, "less-than", "expression-bool-child-0", (0, 0, 0, 0), (0,), 3),
        _PointerOccurrence(
            8,
            "add",
            "expression-int-child-1",
            (0, 0, 0, 0, 1),
            (1,),
            4,
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class _TaskFields:
    entry_point: bytes
    parameter: bytes
    lower: bytes | None
    upper: bytes | None
    delta: bytes | None
    threshold: bytes | None


@dataclass(frozen=True, slots=True)
class _PointerAllocationObservation:
    teacher_position: int
    surface: str
    learning_step_delta: int
    segment_row_delta: int
    returned_exact_rows: int
    inferred_transfer_rows: int
    actual_pointer_rows: int


def _infer_pointer_rows_from_teacher_delta(
    *,
    learning_step_delta: int,
    segment_row_delta: int,
    returned_exact_rows: int,
) -> tuple[int, int]:
    """Split one constructor call's hidden transfer and pointer allocations.

    A role-opening teacher call performs exactly three Learn races: generic
    transfer, exact surface, and pointer cue.  The public return exposes the
    exact surface allocation.  The hidden transfer projection may allocate any
    0..11 destination rows.  A novel pointer cue atomically adds eleven cue
    rows plus eleven recurrence rows; exact pointer replay adds neither.  Thus
    hidden totals 0..11 and 22..33 are disjoint, separating recruitment without
    looking up native owners in Python.
    """

    if learning_step_delta != 3:
        raise RuntimeError("role-opening teacher call did not perform three Learn races")
    if not 0 <= returned_exact_rows <= base.ACTIVE_COLUMNS:
        raise RuntimeError("exact surface allocation is outside its row bound")
    hidden_rows = segment_row_delta - returned_exact_rows
    if 0 <= hidden_rows <= base.ACTIVE_COLUMNS:
        return hidden_rows, 0
    if not (
        MAX_POINTER_ROWS_PER_CUE_CALL
        <= hidden_rows
        <= MAX_POINTER_ROWS_PER_CUE_CALL + base.ACTIVE_COLUMNS
    ):
        raise RuntimeError("constructor segment delta cannot be split deterministically")
    return hidden_rows - MAX_POINTER_ROWS_PER_CUE_CALL, MAX_POINTER_ROWS_PER_CUE_CALL


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", type=Path, required=True)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _assert_one_shot_paths_absent() -> None:
    for kind, path in (
        ("spend marker", SPEND_MARKER_PATH),
        ("one-shot artifact", OUTPUT_PATH),
    ):
        if path.exists():
            raise RuntimeError(f"refusing spent target; {kind} exists: {path}")


def _create_spend_marker() -> dict[str, object]:
    """Durably and exclusively spend the target immediately before generation."""

    _assert_one_shot_paths_absent()
    marker_payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-native-typed-clamp-sum-cross-donor",
        "state": "TARGET_ATTEMPT_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target_key": list(TARGET_KEY),
        "target_task_id": EXPECTED_SOURCE_RECORDS[TARGET_KEY]["task_id"],
        "canonical_artifact_path": _display_path(OUTPUT_PATH),
        "probe_source_sha256": _sha256_file(Path(__file__)),
    }
    serialized = json.dumps(
        marker_payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    try:
        with SPEND_MARKER_PATH.open("x", encoding="utf-8", newline="\n") as marker:
            marker.write(serialized)
            marker.flush()
            os.fsync(marker.fileno())
    except FileExistsError as error:
        raise RuntimeError(
            f"refusing spent target; spend marker exists: {SPEND_MARKER_PATH}"
        ) from error
    # Fail closed if a legacy writer produced an artifact concurrently.  The
    # durable marker is deliberately retained even though generation stays shut.
    if OUTPUT_PATH.exists():
        raise RuntimeError(
            "one-shot artifact appeared while creating the spend marker; "
            "marker retained and target not attempted"
        )
    return {
        "path": _display_path(SPEND_MARKER_PATH),
        "sha256": _sha256_file(SPEND_MARKER_PATH),
        "state": marker_payload["state"],
        "exclusive_create": True,
        "flushed_and_fsynced_before_generate": True,
    }


def _assert_source_committed_and_output_absent() -> None:
    _assert_one_shot_paths_absent()
    if _sha256_file(HELPER_PATH) != EXPECTED_HELPER_SHA256:
        raise RuntimeError("committed typed-workspace helper identity changed")
    expected_native_paths = {
        path.relative_to(REPO_ROOT).as_posix() for path in NATIVE_MECHANISM_PATHS
    }
    if set(EXPECTED_NATIVE_MECHANISM_SHA256) != expected_native_paths:
        raise RuntimeError("native mechanism source pins are incomplete")
    for path in NATIVE_MECHANISM_PATHS:
        source = path.relative_to(REPO_ROOT).as_posix()
        if _sha256_file(path) != EXPECTED_NATIVE_MECHANISM_SHA256[source]:
            raise RuntimeError(f"native mechanism source identity changed: {source}")
    for path in (
        Path(__file__).resolve(),
        HELPER_PATH.resolve(),
        *NATIVE_MECHANISM_PATHS,
    ):
        source = path.relative_to(REPO_ROOT).as_posix()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", source],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout:
            raise RuntimeError(
                "probe, helper, and native mechanism sources must be committed "
                f"and unmodified: {source}"
            )
    if SLOT_COUNT != 4:
        raise RuntimeError("cross-donor clamp-sum probe requires four transport slots")


def _select_tasks() -> tuple[object, tuple[GeneratedTask, ...], GeneratedTask]:
    protocol = load_protocol(PROTOCOL_PATH)
    if (
        protocol.protocol_id != EXPECTED_PROTOCOL_ID
        or protocol.sha256 != EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError("public Simple-SWE protocol identity changed")
    visible_train = generate_visible_partition(protocol, "train")
    selected: dict[tuple[str, int], GeneratedTask] = {}
    for key in (*TRAIN_KEYS, TARGET_KEY):
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
    return protocol, tuple(selected[key] for key in TRAIN_KEYS), selected[TARGET_KEY]


def _task_fields(public_task: object) -> _TaskFields:
    if type(public_task).__name__ != "PublicTask":
        raise TypeError("cross-donor parsing requires the exact public task view")
    key = (public_task.family, public_task.ordinal)
    if key not in EXPECTED_SOURCE_RECORDS:
        raise RuntimeError(f"unsupported cross-donor task key {key!r}")
    expected = EXPECTED_SOURCE_RECORDS[key]
    signature = public_task.signature
    expected_parameter_type = (
        ProgramType.Int()
        if key == SCALAR_CLAMP_KEY
        else ProgramType.List(ProgramType.Int())
    )
    if (
        len(signature.parameters) != 1
        or signature.parameters[0].type != expected_parameter_type
        or signature.result != ProgramType.Int()
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
    if public_task.partition != "train":
        raise RuntimeError(f"task left the visible TRAIN partition for {key!r}")

    def optional_bytes(name: str) -> bytes | None:
        value = expected[name]
        return None if value is None else str(value).encode("ascii")

    fields = _TaskFields(
        entry_point=str(expected["entry_point"]).encode("ascii"),
        parameter=str(expected["parameter"]).encode("ascii"),
        lower=optional_bytes("lower"),
        upper=optional_bytes("upper"),
        delta=optional_bytes("delta"),
        threshold=optional_bytes("threshold"),
    )
    prompt = public_task.prompt.encode("ascii")
    signature_bytes = b"def " + fields.entry_point + b"(" + fields.parameter + b"):"
    prompt.index(signature_bytes)
    if key == SCALAR_CLAMP_KEY:
        phrase = b"inclusive range -8 through 8"
    elif key == MAP_COUNT_KEY:
        phrase = b"Add -2 to each integer and count results strictly greater than 0"
    else:
        phrase = b"Limit every integer to -3 through 3, then return their sum"
    if phrase not in prompt:
        raise RuntimeError(f"visible task wording changed for {key!r}")
    return fields


def _canonical_literal_primitives(literal: bytes) -> tuple[str, ...]:
    return harness._canonical_literal_primitives(literal)


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
    if key == MAP_COUNT_KEY:
        if fields.delta is None or fields.threshold is None:
            raise RuntimeError("map-count task lost delta or threshold")
        delta_start = prompt.index(b"Add ", parameter_start) + len(b"Add ")
        threshold_start = prompt.index(b"strictly greater than ", delta_start) + len(
            b"strictly greater than "
        )
        literal_spans = (
            (delta_start, fields.delta, 2),
            (threshold_start, fields.threshold, 3),
        )
    else:
        if fields.lower is None or fields.upper is None:
            raise RuntimeError("clamp task lost lower or upper bound")
        anchor = b"inclusive range " if key == SCALAR_CLAMP_KEY else b"Limit every integer to "
        lower_start = prompt.index(anchor, parameter_start) + len(anchor)
        upper_start = lower_start + len(fields.lower) + len(b" through ")
        literal_spans = (
            (lower_start, fields.lower, 2),
            (upper_start, fields.upper, 3),
        )
    for start, literal, slot in literal_spans:
        if prompt[start : start + len(literal)] != literal:
            raise RuntimeError(f"visible slot {slot} span no longer matches")
        spans.append(
            (
                start,
                start + len(literal),
                slot,
                tuple(
                    PRIMITIVE_TOKEN_IDS[token]
                    for token in _canonical_literal_primitives(literal)
                ),
            )
        )
    spans.sort(key=lambda span: span[0])
    return prompt, tuple(spans)


def _detokenize_prompt(tokens: tuple[int, ...]) -> bytes:
    return harness._detokenize_prompt(tokens)


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
        raise RuntimeError("cross-donor prompt tokenizer is not lossless")
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


def _encode_teacher(
    primitives: tuple[str | tuple[str, int], ...],
) -> tuple[int, ...]:
    return (
        *(
            copy_slot_token_id(item[1])
            if isinstance(item, tuple)
            else PRIMITIVE_TOKEN_IDS[item]
            for item in primitives
        ),
        base.EOS_TOKEN_ID,
    )


def _internal_teacher_tokens(key: tuple[str, int]) -> tuple[int, ...]:
    if key == SCALAR_CLAMP_KEY:
        primitives: tuple[str | tuple[str, int], ...] = (
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
    elif key == MAP_COUNT_KEY:
        primitives = (
            "sum",
            "map",
            "bind",
            "if",
            "less-than",
            "integer",
            ("copy", 3),
            "end",
            "add",
            "variable-0",
            "integer",
            ("copy", 2),
            "end",
            "integer",
            "+",
            "1",
            "end",
            "integer",
            "+",
            "0",
            "end",
            "variable-0",
        )
    else:
        raise RuntimeError(f"unsupported teacher key {key!r}")
    return _encode_teacher(primitives)


def _learning_arrival_accounting(
    train_prompts: dict[tuple[str, int], tuple[int, ...]],
    teachers: dict[tuple[str, int], tuple[int, ...]],
) -> dict[str, object]:
    if set(train_prompts) != set(TRAIN_KEYS) or set(teachers) != set(TRAIN_KEYS):
        raise RuntimeError("capacity accounting must contain exactly both donors")
    episodes: dict[str, object] = {}
    base_learning_arrivals = 0
    all_occurrence_labels: list[str] = []
    for key in TRAIN_KEYS:
        prompt_count = len(train_prompts[key])
        teacher_count = len(teachers[key])
        if prompt_count != EXPECTED_PROMPT_TOKEN_COUNTS[key]:
            raise RuntimeError(f"teacher prompt token count changed for {key!r}")
        if teacher_count != EXPECTED_TEACHER_CALL_COUNTS[key]:
            raise RuntimeError(f"fixed teacher stream length changed for {key!r}")
        eligible_teacher_calls = teacher_count - 1
        physical_teacher_arrivals = (
            eligible_teacher_calls * PHYSICAL_LEARN_STEPS_PER_ELIGIBLE_TEACHER
            + 1
        )
        if physical_teacher_arrivals != EXPECTED_PHYSICAL_TEACHER_ARRIVALS[key]:
            raise RuntimeError(f"physical teacher arrivals changed for {key!r}")
        base_episode_arrivals = (
            prompt_count + PLAN_COMPONENT_ARRIVALS[key] + physical_teacher_arrivals
        )
        if base_episode_arrivals != EXPECTED_EPISODE_LEARNING_ARRIVALS[key]:
            raise RuntimeError(f"episode learning arrivals changed for {key!r}")
        occurrence_records: list[dict[str, object]] = []
        for occurrence_index, occurrence in enumerate(POINTER_OCCURRENCES_BY_DONOR[key]):
            observed_surface = PRIMITIVE_BY_TOKEN_ID[
                teachers[key][occurrence.teacher_position]
            ]
            if observed_surface != occurrence.surface:
                raise RuntimeError(
                    f"constructor occurrence moved for {key!r} at "
                    f"teacher position {occurrence.teacher_position}"
                )
            if occurrence.surface not in ROLE_BEARING_CONSTRUCTORS:
                raise RuntimeError(
                    f"non-role-bearing pointer surface {occurrence.surface!r}"
                )
            if occurrence.cue not in TYPED_CUE_NAMES:
                raise RuntimeError(f"unknown pointer semantic cue {occurrence.cue!r}")
            if occurrence.parent_occurrence is not None and not (
                0 <= occurrence.parent_occurrence < occurrence_index
            ):
                raise RuntimeError("pointer parent must precede its descendant")
            absolute_offset = harness._address_offset(occurrence.absolute_address)
            relative_offset = harness._address_offset(occurrence.relative_address)
            if absolute_offset not in ADDRESS_OFFSETS or relative_offset not in ADDRESS_OFFSETS:
                raise RuntimeError("constructor pointer address is absent from the codebook")
            # This is a frozen occurrence label for capacity provenance.  It
            # does not claim that the native winner tuple is physically unique.
            surface_feedback_source = f"{key[0]}/{key[1]:03d}@{occurrence.teacher_position}"
            parent_source = (
                None
                if occurrence.parent_occurrence is None
                else f"{key[0]}/{key[1]:03d}#{occurrence.parent_occurrence}"
            )
            all_occurrence_labels.append(surface_feedback_source)
            occurrence_records.append(
                {
                    "occurrence": occurrence_index,
                    "teacher_position": occurrence.teacher_position,
                    "surface": occurrence.surface,
                    "surface_feedback_source": surface_feedback_source,
                    "semantic_cue": occurrence.cue,
                    "absolute_address": list(occurrence.absolute_address),
                    "absolute_address_offset": absolute_offset,
                    "relative_address": list(occurrence.relative_address),
                    "relative_address_offset": relative_offset,
                    "parent_occurrence": occurrence.parent_occurrence,
                    "parent_pointer_source": parent_source,
                    "maximum_persistent_rows": MAX_POINTER_ROWS_PER_CUE_CALL,
                }
            )
        pointer_call_count = len(occurrence_records)
        episodes[f"{key[0]}/{key[1]:03d}"] = {
            "prompt_arrivals": prompt_count,
            "plan_component_arrivals": PLAN_COMPONENT_ARRIVALS[key],
            "teacher_api_calls": teacher_count,
            "eligible_teacher_api_calls": eligible_teacher_calls,
            "physical_learn_steps_per_eligible_teacher_call": (
                PHYSICAL_LEARN_STEPS_PER_ELIGIBLE_TEACHER
            ),
            "singleton_eos_arrivals": 1,
            "physical_teacher_arrivals": physical_teacher_arrivals,
            "base_learning_arrivals": base_episode_arrivals,
            "constructor_pointer_cue_arrivals": pointer_call_count,
            "total_learning_arrivals": base_episode_arrivals + pointer_call_count,
            "constructor_pointer_occurrences": occurrence_records,
            "constructor_pointer_row_capacity_upper_bound": (
                pointer_call_count * MAX_POINTER_ROWS_PER_CUE_CALL
            ),
        }
        base_learning_arrivals += base_episode_arrivals

    if base_learning_arrivals != EXPECTED_BASE_LEARNING_ARRIVALS:
        raise RuntimeError("base physical Learn arrivals changed")
    pointer_cue_arrivals = len(all_occurrence_labels)
    if pointer_cue_arrivals != EXPECTED_POINTER_CUE_LEARNING_ARRIVALS:
        raise RuntimeError("constructor-pointer occurrence count changed")
    if len(set(all_occurrence_labels)) != pointer_cue_arrivals:
        raise RuntimeError("constructor-pointer occurrence labels unexpectedly alias")
    total_learning_arrivals = base_learning_arrivals + pointer_cue_arrivals
    if total_learning_arrivals != EXPECTED_TOTAL_LEARNING_ARRIVALS:
        raise RuntimeError("total physical Learn arrivals changed")
    # Each reset begins with no prior winner context, so the first prompt
    # arrival of each donor cannot allocate a temporal row.
    noninitial_learning_arrivals = total_learning_arrivals - len(TRAIN_KEYS)
    if noninitial_learning_arrivals != EXPECTED_NONINITIAL_LEARNING_ARRIVALS:
        raise RuntimeError("noninitial Learn arrival accounting changed")
    base_noninitial = base_learning_arrivals - len(TRAIN_KEYS)
    if base_noninitial != EXPECTED_BASE_NONINITIAL_LEARNING_ARRIVALS:
        raise RuntimeError("base noninitial Learn arrival accounting changed")
    base_segment_rows = base_noninitial * base.ACTIVE_COLUMNS
    if base_segment_rows != EXPECTED_BASE_SEGMENT_ROWS:
        raise RuntimeError("base temporal row budget changed")
    ordinary_segment_rows_upper_bound = (
        noninitial_learning_arrivals * base.ACTIVE_COLUMNS
    )
    if (
        ordinary_segment_rows_upper_bound
        != EXPECTED_ORDINARY_SEGMENT_ROWS_UPPER_BOUND
    ):
        raise RuntimeError("ordinary Learn row budget changed")
    pointer_cue_rows_upper_bound = pointer_cue_arrivals * base.ACTIVE_COLUMNS
    pointer_recurrence_rows_upper_bound = pointer_cue_arrivals * base.ACTIVE_COLUMNS
    if (
        pointer_cue_rows_upper_bound
        != EXPECTED_POINTER_CUE_SEGMENT_ROWS_UPPER_BOUND
    ):
        raise RuntimeError("constructor-pointer cue row budget changed")
    if (
        pointer_recurrence_rows_upper_bound
        != EXPECTED_POINTER_RECURRENCE_ROWS_UPPER_BOUND
    ):
        raise RuntimeError("constructor-pointer recurrence row budget changed")
    pointer_segment_rows_upper_bound = (
        pointer_cue_rows_upper_bound + pointer_recurrence_rows_upper_bound
    )
    if (
        pointer_segment_rows_upper_bound
        != EXPECTED_POINTER_SEGMENT_ROWS_UPPER_BOUND
    ):
        raise RuntimeError("constructor-pointer row budget changed")
    segment_capacity = (
        ordinary_segment_rows_upper_bound + pointer_recurrence_rows_upper_bound
    )
    if segment_capacity != EXPECTED_SEGMENT_CAPACITY:
        raise RuntimeError("cross-donor segment capacity changed")
    base_population = COLUMN_COUNT * base.BASE_CELLS_PER_COLUMN
    return {
        "episodes": episodes,
        "base_learning_arrivals": base_learning_arrivals,
        "constructor_pointer_cue_arrivals": pointer_cue_arrivals,
        "total_learning_arrivals": total_learning_arrivals,
        "resets": len(TRAIN_KEYS),
        "noninitial_learning_arrivals": noninitial_learning_arrivals,
        "segments_per_noninitial_arrival": base.ACTIVE_COLUMNS,
        "base_temporal_segment_rows_upper_bound": base_segment_rows,
        "ordinary_segment_rows_upper_bound_including_pointer_cues": (
            ordinary_segment_rows_upper_bound
        ),
        "constructor_pointer": {
            "role_bearing_constructors": list(ROLE_BEARING_CONSTRUCTORS),
            "identity": (
                "opaque surface winners + semantic cue/role + parent-relative "
                "address + parent pointer when active"
            ),
            "exact_role_opening_call_count": pointer_cue_arrivals,
            "rows_per_cue_call_upper_bound": MAX_POINTER_ROWS_PER_CUE_CALL,
            "cue_rows_per_novel_call": base.ACTIVE_COLUMNS,
            "recurrence_rows_per_novel_call": base.ACTIVE_COLUMNS,
            "exact_replay_rows": 0,
            "cue_rows_upper_bound": pointer_cue_rows_upper_bound,
            "recurrence_rows_upper_bound": pointer_recurrence_rows_upper_bound,
            "total_rows_upper_bound": pointer_segment_rows_upper_bound,
            "frozen_occurrence_labels_are_unique": True,
            "physical_pointer_identity_uniqueness_claimed": False,
            "capacity_assumes_every_call_recruits": True,
            "root_election_is_upstream": True,
        },
        "segment_capacity": segment_capacity,
        "base_pyramidal_cells": base_population,
        "reserve_pyramidal_cells": segment_capacity,
        "max_pyramidal_cells": base_population + segment_capacity,
    }


def _expected_program_tokens(public_task: object) -> tuple[str, ...]:
    fields = _task_fields(public_task)
    key = (public_task.family, public_task.ordinal)
    if key == SCALAR_CLAMP_KEY:
        if fields.lower is None or fields.upper is None:
            raise RuntimeError("scalar clamp lost its bounds")
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
    if key == MAP_COUNT_KEY:
        if fields.delta is None or fields.threshold is None:
            raise RuntimeError("map-count donor lost delta or threshold")
        delta = _canonical_literal_primitives(fields.delta)
        threshold = _canonical_literal_primitives(fields.threshold)
        return (
            "sum",
            "map",
            "bind",
            "if",
            "less-than",
            "integer",
            *threshold,
            "end",
            "add",
            "variable-0",
            "integer",
            *delta,
            "end",
            "integer",
            "+",
            "1",
            "end",
            "integer",
            "+",
            "0",
            "end",
            "variable-0",
        )
    if key == TARGET_KEY:
        if fields.lower is None or fields.upper is None:
            raise RuntimeError("clamp-sum target lost its bounds")
        lower = _canonical_literal_primitives(fields.lower)
        upper = _canonical_literal_primitives(fields.upper)
        result = (
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
        if result != EXPECTED_TARGET_TOKENS:
            raise RuntimeError("frozen target token stream changed")
        return result
    raise RuntimeError(f"unsupported expected-program key {key!r}")


def _expected_compiled_source(public_task: object) -> str:
    fields = _task_fields(public_task)
    key = (public_task.family, public_task.ordinal)
    entry = fields.entry_point.decode("ascii")
    parameter = fields.parameter.decode("ascii")
    if key in {SCALAR_CLAMP_KEY, TARGET_KEY}:
        if fields.lower is None or fields.upper is None:
            raise RuntimeError("clamp source requires both bounds")
        lower = fields.lower.decode("ascii")
        upper = fields.upper.decode("ascii")
        expression = (
            f"{lower} if item_0 < {lower} else {upper} if {upper} < item_0 "
            "else item_0"
            if key == TARGET_KEY
            else f"{lower} if {parameter} < {lower} else {upper} if {upper} < "
            f"{parameter} else {parameter}"
        )
        if key == TARGET_KEY:
            source = (
                f"def {entry}({parameter}):\n"
                f"    return sum([{expression} for item_0 in {parameter}])\n"
            )
            if source != EXPECTED_TARGET_SOURCE:
                raise RuntimeError("frozen target source changed")
            return source
        return f"def {entry}({parameter}):\n    return {expression}\n"
    if key == MAP_COUNT_KEY:
        if fields.delta is None or fields.threshold is None:
            raise RuntimeError("map-count source requires delta and threshold")
        delta = fields.delta.decode("ascii")
        threshold = fields.threshold.decode("ascii")
        return (
            f"def {entry}({parameter}):\n"
            f"    return sum([1 if {threshold} < item_0 + {delta} else 0 "
            f"for item_0 in {parameter}])\n"
        )
    raise RuntimeError(f"unsupported expected-source key {key!r}")


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
        "lower": None if fields.lower is None else fields.lower.decode("ascii"),
        "upper": None if fields.upper is None else fields.upper.decode("ascii"),
        "delta": None if fields.delta is None else fields.delta.decode("ascii"),
        "threshold": (
            None if fields.threshold is None else fields.threshold.decode("ascii")
        ),
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


@contextmanager
def _configured_harness() -> object:
    """Temporarily bind task-specific pure helpers into the committed harness."""

    overrides = {
        "TRAIN_KEYS": TRAIN_KEYS,
        "EXPECTED_PROMPT_TOKEN_COUNTS": EXPECTED_PROMPT_TOKEN_COUNTS,
        "ADDRESS_OFFSETS": ADDRESS_OFFSETS,
        "ADDRESS_TOKEN_IDS": ADDRESS_TOKEN_IDS,
        "TYPED_PRIVATE_TOKEN_IDS": TYPED_PRIVATE_TOKEN_IDS,
        "_RESERVED_COLUMN_COUNT": _RESERVED_COLUMN_COUNT,
        "_tokenized_public_prompt": _tokenized_public_prompt,
        "_internal_teacher_tokens": _internal_teacher_tokens,
        "_learning_arrival_accounting": _learning_arrival_accounting,
        "_expected_program_tokens": _expected_program_tokens,
        "_expected_compiled_source": _expected_compiled_source,
        "_public_task_record": _public_task_record,
        "_expected_copy_schedule": _expected_copy_schedule,
        "compile_typed_program": compile_typed_program,
        "execute_completion": execute_completion,
    }
    previous = {name: getattr(harness, name) for name in overrides}
    previous_column_count = base.COLUMN_COUNT
    try:
        base.COLUMN_COUNT = COLUMN_COUNT
        for name, value in overrides.items():
            setattr(harness, name, value)
        yield
    finally:
        base.COLUMN_COUNT = previous_column_count
        for name, value in previous.items():
            setattr(harness, name, value)


def _definition(
    train_tasks: tuple[GeneratedTask, ...],
    target_input: _TargetRuntimeInput,
) -> tuple[object, dict[str, object]]:
    with _configured_harness():
        return harness._definition(train_tasks, target_input)


def _expected_copy_schedule(
    public_task: object,
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    fields = _task_fields(public_task)
    key = (public_task.family, public_task.ordinal)
    if key == MAP_COUNT_KEY:
        if fields.delta is None or fields.threshold is None:
            raise RuntimeError("map-count copy schedule lost its literals")
        return (
            (3, _canonical_literal_primitives(fields.threshold)),
            (2, _canonical_literal_primitives(fields.delta)),
        )
    if fields.lower is None or fields.upper is None:
        raise RuntimeError("clamp copy schedule lost its bounds")
    lower = _canonical_literal_primitives(fields.lower)
    upper = _canonical_literal_primitives(fields.upper)
    return ((2, lower), (2, lower), (3, upper), (3, upper))


def _failure_localization(
    *,
    gate: str,
    generation_position: int | None,
    expected_tokens: tuple[str, ...],
) -> dict[str, object]:
    if generation_position is None:
        scope = "post-generation"
        expected_surface = None
    else:
        scope = "root" if generation_position == 0 else "descendant"
        expected_surface = (
            expected_tokens[generation_position]
            if generation_position < len(expected_tokens)
            else "<EOS>"
            if generation_position == len(expected_tokens)
            else None
        )
    return {
        "gate": gate,
        "scope": scope,
        "generation_position": generation_position,
        "expected_surface": expected_surface,
        "root_expected_surface": expected_tokens[0],
        "root_election_is_upstream_of_constructor_pointer": True,
    }


def _generation_failure(
    error: NativeTemporalRuntimeError,
    expected_tokens: tuple[str, ...],
) -> dict[str, object]:
    position = int(error.completed_prefix)
    return {
        "kind": "model",
        "gate": "GENERATION_STOP",
        "localization": _failure_localization(
            gate="GENERATION_STOP",
            generation_position=position,
            expected_tokens=expected_tokens,
        ),
        "detail": {
            "status": error.status.name,
            "message": error.message,
            "completed_surface_prefix": position,
            "required": error.required,
            "observed": error.observed,
            "location": error.location,
            "poison_code": error.poison_code,
            "stage_code": error.stage_code,
        },
    }


def _first_stream_mismatch(
    observed: tuple[int, ...],
    expected: tuple[int, ...],
) -> int | None:
    sentinel = object()
    for position, (observed_id, expected_id) in enumerate(
        zip_longest(observed, expected, fillvalue=sentinel)
    ):
        if observed_id != expected_id:
            return position
    return None


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
    # Reuse the committed atomic scoring/sandbox harness.  Only the task parser,
    # expected program, and copy schedule are rebound for this frozen cohort.
    with _configured_harness():
        record, first_failure = harness._score_target(
            episode=episode,
            raw_generations=raw_generations,
            native_failure=native_failure,
            task=task,
            protocol=protocol,
            python_executable=python_executable,
            runtime_identity=runtime_identity,
            worker_source=worker_source,
            worker_sha256=worker_sha256,
        )
    if first_failure is not None and "localization" not in first_failure:
        gate = str(first_failure["gate"])
        position: int | None = None
        if gate == "EXACT_TYPED_TOKEN_STREAM":
            observed = tuple(
                int(generation["surface_token_id"])
                for generation in record["generations"]
            )
            expected = (
                *(PRIMITIVE_TOKEN_IDS[token] for token in EXPECTED_TARGET_TOKENS),
                base.EOS_TOKEN_ID,
            )
            position = _first_stream_mismatch(observed, expected)
        elif gate == "ORDINARY_EOS":
            position = len(record["generations"])
        first_failure["localization"] = _failure_localization(
            gate=gate,
            generation_position=position,
            expected_tokens=EXPECTED_TARGET_TOKENS,
        )
    record["first_failure_localization"] = (
        None if first_failure is None else first_failure["localization"]
    )
    return record, first_failure


def _teacher_label(token_id: int) -> str:
    for slot in (2, 3):
        if token_id == copy_slot_token_id(slot):
            return f"COPY_SLOT_{slot}"
    if token_id == base.EOS_TOKEN_ID:
        return "<EOS>"
    return PRIMITIVE_BY_TOKEN_ID[token_id]


def _pointer_observation_record(
    observation: _PointerAllocationObservation,
) -> dict[str, int | str | bool]:
    return {
        "teacher_position": observation.teacher_position,
        "surface": observation.surface,
        "learning_step_delta": observation.learning_step_delta,
        "segment_row_delta": observation.segment_row_delta,
        "returned_exact_rows": observation.returned_exact_rows,
        "inferred_transfer_rows": observation.inferred_transfer_rows,
        "actual_pointer_rows": observation.actual_pointer_rows,
        "recruited": observation.actual_pointer_rows
        == MAX_POINTER_ROWS_PER_CUE_CALL,
    }


def _train_task_with_pointer_accounting(
    circuit: NativeTemporalCircuit,
    *,
    key: tuple[str, int],
    prompt_tokens: tuple[int, ...],
    parameter_type: object,
    result_type: object,
    teacher_tokens: tuple[int, ...],
) -> tuple[_WorkspaceEpisode, tuple[_PointerAllocationObservation, ...]]:
    """Use the established typed adapter while measuring pointer row deltas."""

    occurrences = {
        occurrence.teacher_position: occurrence
        for occurrence in POINTER_OCCURRENCES_BY_DONOR[key]
    }
    if len(occurrences) != len(POINTER_OCCURRENCES_BY_DONOR[key]):
        raise RuntimeError("constructor pointer positions are not injective")
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
    teacher_steps: list[object] = []
    observations: list[_PointerAllocationObservation] = []
    for position, token_id in enumerate(teacher_tokens):
        occurrence = occurrences.get(position)
        before = None if occurrence is None else circuit.checkpoint_identity()
        step = circuit.teach_typed_token(token_id)
        teacher_steps.append(step)
        if occurrence is None:
            continue
        after = circuit.checkpoint_identity()
        assert before is not None
        learning_step_delta = after.learning_step - before.learning_step
        segment_row_delta = after.segment_count - before.segment_count
        transfer_rows, pointer_rows = _infer_pointer_rows_from_teacher_delta(
            learning_step_delta=learning_step_delta,
            segment_row_delta=segment_row_delta,
            returned_exact_rows=step.allocated_segments,
        )
        observations.append(
            _PointerAllocationObservation(
                teacher_position=position,
                surface=occurrence.surface,
                learning_step_delta=learning_step_delta,
                segment_row_delta=segment_row_delta,
                returned_exact_rows=step.allocated_segments,
                inferred_transfer_rows=transfer_rows,
                actual_pointer_rows=pointer_rows,
            )
        )
    if len(observations) != len(occurrences):
        raise RuntimeError("not every frozen constructor pointer call was observed")
    return (
        _WorkspaceEpisode(
            prompt_steps=prompt_steps,
            workspace_steps=(begin_step, *teacher_steps),
        ),
        tuple(observations),
    )


def main() -> int:
    args = _parse_args()
    _assert_source_committed_and_output_absent()
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
        raise RuntimeError("cross-donor probe requires explicit repo Python 3.11.15")
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = hashlib.sha256(worker_path.read_bytes()).hexdigest()
    worker_source = load_worker_snapshot(worker_sha256)

    trained_episodes: list[
        tuple[
            GeneratedTask,
            _WorkspaceEpisode,
            object,
            tuple[_PointerAllocationObservation, ...],
        ]
    ] = []
    native_error: NativeTemporalRuntimeError | None = None
    with base._public_development_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            for key, task in zip(TRAIN_KEYS, train_tasks, strict=True):
                signature = task.public.signature
                episode, pointer_observations = _train_task_with_pointer_accounting(
                    circuit,
                    key=key,
                    prompt_tokens=_tokenized_public_prompt(task.public),
                    parameter_type=signature.parameters[0].type,
                    result_type=signature.result,
                    teacher_tokens=teachers[key],
                )
                trained_episodes.append(
                    (
                        task,
                        episode,
                        circuit.checkpoint_identity(),
                        pointer_observations,
                    )
                )
            trained_identity = circuit.checkpoint_identity()
            observed_pointer_cue_arrivals = (
                trained_identity.learning_step - EXPECTED_BASE_LEARNING_ARRIVALS
            )
            if (
                trained_identity.learning_step != EXPECTED_TOTAL_LEARNING_ARRIVALS
                or observed_pointer_cue_arrivals
                != EXPECTED_POINTER_CUE_LEARNING_ARRIVALS
            ):
                raise RuntimeError(
                    "native constructor-pointer arrival accounting changed"
                )
            all_pointer_observations = tuple(
                observation
                for _, _, _, observations in trained_episodes
                for observation in observations
            )
            actual_pointer_rows = sum(
                observation.actual_pointer_rows
                for observation in all_pointer_observations
            )
            if (
                len(all_pointer_observations)
                != EXPECTED_POINTER_CUE_LEARNING_ARRIVALS
                or actual_pointer_rows > EXPECTED_POINTER_SEGMENT_ROWS_UPPER_BOUND
                or actual_pointer_rows % MAX_POINTER_ROWS_PER_CUE_CALL
            ):
                raise RuntimeError("native constructor-pointer allocation accounting changed")
            actual_recruited_pointer_count = (
                actual_pointer_rows // MAX_POINTER_ROWS_PER_CUE_CALL
            )
            target_episode = _TypedWorkspaceApi.begin_target(circuit, target_input)
            try:
                spend_marker_record = _create_spend_marker()
                raw_generations = circuit.generate_typed()
            except NativeTemporalRuntimeError as error:
                raw_generations = None
                native_error = error
        native_identity = loaded_library_identity()

    if (
        not SPEND_MARKER_PATH.exists()
        or _sha256_file(SPEND_MARKER_PATH) != spend_marker_record["sha256"]
    ):
        raise RuntimeError("durable target spend marker changed after generation")

    # Target expected tokens, compiler checks, and private cases enter only
    # after the atomic native attempt and after the circuit has closed.
    native_failure = (
        None
        if native_error is None
        else _generation_failure(native_error, EXPECTED_TARGET_TOKENS)
    )
    compiler_records: list[dict[str, object]] = []
    for task in train_tasks:
        tokens = _expected_program_tokens(task.public)
        compiled_source = compile_typed_program(
            Program(tokens, task.public.signature)
        ).source
        expected_source = _expected_compiled_source(task.public)
        if compiled_source != expected_source:
            raise RuntimeError(f"typed compiler cannot express donor {task.public.task_id}")
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
            "copy_slot_schedule": [
                slot for slot, _ in _expected_copy_schedule(task.public)
            ],
            "prompt_steps": len(episode.prompt_steps),
            "workspace_api_calls": len(episode.workspace_steps),
            "reset_before": True,
            "live_segments_after_task": identity.segment_count,
            "learning_steps_after_task": identity.learning_step,
            "constructor_pointer_allocations": [
                _pointer_observation_record(observation)
                for observation in pointer_observations
            ],
        }
        for task, episode, identity, pointer_observations in trained_episodes
    ]
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-native-typed-clamp-sum-cross-donor",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": (
            "PASS_NATIVE_TYPED_CLAMP_SUM_CROSS_DONOR"
            if passed
            else "STOP_NATIVE_TYPED_CLAMP_SUM_CROSS_DONOR"
        ),
        "capability_pass": passed,
        "first_failure": first_failure,
        "claim_boundary": (
            "One retained native temporal circuit learns visible TRAIN "
            "t0_scalar_clamp/001 and t2_map_then_count/002, then attempts visible "
            "TRAIN t2_clamp_then_sum/003 exactly once without a teacher. A pass "
            "shows root Sum/Map/Bind and descendant scalar-clamp structure "
            "recombine with unseen -3/+3 bounds. All tasks were present in the "
            "historical prompt-only public baseline, but this target is fresh for "
            "the native typed lane. This is not DEV, eval, held-out, M0, C1, or "
            "broad coding evidence."
        ),
        "authority": {
            "source_partition_generated": "train",
            "eval_partition_generated": False,
            "dev_or_held_out_opened": False,
            "train_keys": [list(key) for key in TRAIN_KEYS],
            "teacher_free_target_key": list(TARGET_KEY),
            "teacher_free_target_attempts_completed": 1,
            "model_retries": 0,
            "harness_retries": 0,
            "artifact_overwrite_permitted": False,
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
        "target_spend_marker": {
            **spend_marker_record,
            "exists_after_attempt": True,
            "retained_on_model_failure": True,
            "artifact_may_coexist_after_attempt": True,
        },
        "protocol": {
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.sha256,
            "language": protocol.raw.get("language"),
        },
        "typed_programs": {
            "donor_compiler_checks": compiler_records,
            "target_expected_tokens": list(EXPECTED_TARGET_TOKENS),
            "target_expected_source": EXPECTED_TARGET_SOURCE,
            "target_stream_differs_from_every_donor_stream": all(
                _expected_program_tokens(task.public) != EXPECTED_TARGET_TOKENS
                for task in train_tasks
            ),
        },
        "localization_contract": {
            "root_surface": "sum",
            "root_election_is_upstream_of_constructor_pointer": True,
            "native_completed_prefix_zero": "root",
            "native_completed_prefix_positive": "descendant",
            "successful_stream_mismatch_uses_first_surface_position": True,
        },
        "constructor_pointer": {
            **capacity_accounting["constructor_pointer"],
            "observed_pointer_cue_learning_arrivals_after_train": (
                observed_pointer_cue_arrivals
            ),
            "actual_recruited_pointer_count": actual_recruited_pointer_count,
            "actual_exact_replay_pointer_call_count": (
                EXPECTED_POINTER_CUE_LEARNING_ARRIVALS
                - actual_recruited_pointer_count
            ),
            "actual_cue_rows": actual_pointer_rows // 2,
            "actual_recurrence_rows": actual_pointer_rows // 2,
            "actual_total_rows": actual_pointer_rows,
            "allocation_measurement": (
                "post-call checkpoint segment delta minus returned exact-surface "
                "rows; hidden transfer allocation is 0..11 rows and novel "
                "pointer recruitment adds exactly 11 cue + 11 recurrence rows"
            ),
            "python_owner_lookup": False,
            "post_donor_checkpoint_segment_count": trained_identity.segment_count,
            "training_learning_step_asserted": trained_identity.learning_step,
        },
        "slot_bank": {
            "slot_count": SLOT_COUNT,
            "entry_slot_ordinal": 0,
            "parameter_slot_ordinal": 1,
            "literal_or_lower_bound_or_delta_slot_ordinal": 2,
            "upper_bound_or_threshold_slot_ordinal": 3,
            "donor_copy_slot_schedules": {
                "t0_scalar_clamp/001": [2, 2, 3, 3],
                "t2_map_then_count/002": [3, 2],
            },
            "target_expected_copy_slot_schedule": [2, 2, 3, 3],
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
        "visible_train_target_teacher_free": target_record,
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
            "probe_source_sha256": _sha256_file(Path(__file__)),
            "helper_source_sha256": _sha256_file(HELPER_PATH),
            "native_mechanism_source_sha256": {
                path.relative_to(REPO_ROOT).as_posix(): _sha256_file(path)
                for path in NATIVE_MECHANISM_PATHS
            },
        },
    }
    with OUTPUT_PATH.open("x", encoding="utf-8", newline="\n") as output_file:
        output_file.write(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(
        json.dumps(
            {
                "outcome": payload["outcome"],
                "first_failure": first_failure,
                "output": str(OUTPUT_PATH),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
