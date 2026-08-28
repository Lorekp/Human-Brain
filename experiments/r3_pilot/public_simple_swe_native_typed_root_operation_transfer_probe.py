"""Prepared one-shot visible-TRAIN cross-family root-operation-transfer probe.

One retained native temporal circuit learns visible TRAIN
``t2_clamp_then_sum/004``.  It then receives visible TRAIN
``t0_scalar_clamp/004`` exactly once without a teacher.  The target asks the
root ``Expression(Int)`` cue to retrieve the donor's nested clamp ``If`` via
direction-isolated ``HeadToRootTransfer``.  Importing or compiling spends
nothing.  The committed native mechanism must be source-pinned before the
one-shot target can execute.
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

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PROBE_DIR = Path(__file__).resolve().parent
for import_root in (SRC_ROOT, REPO_ROOT, PROBE_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import public_simple_swe_native_typed_clamp_sum_cross_donor_probe as harness
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
_BASE_LEARNING_ARRIVAL_ACCOUNTING = harness._learning_arrival_accounting
_BASE_INFER_POINTER_ROWS = harness._infer_pointer_rows_from_teacher_delta
COLUMN_COUNT = 768

# Frozen safe superset of the donor and target absolute/rebased holes.
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
    raise RuntimeError("root-operation-transfer private-code reserve changed")

OUTPUT_PATH = Path(__file__).with_suffix(".json")
SPEND_MARKER_PATH = Path(__file__).with_suffix(".spent.json")
PROTOCOL_PATH = REPO_ROOT / "benchmarks" / "simple_swe_protocol_v1.json"
HELPER_PATH = PROBE_DIR / "public_simple_swe_native_typed_clamp_sum_cross_donor_probe.py"
EXPECTED_HELPER_SHA256 = (
    "3cf58f9eb6dcb6e0ce089ea58fe1ca6ba778546b546f12babf1539bdebc6decb"
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
        "012d4f79505e3e0e381ebf9397db0c74d695120e7dfe8d511a338f8b896b9dc4"
    ),
    "src/humanbrain/_native_src/src/temporal_runtime/checkpoint.rs": (
        "bdb1ab444a826e706b9c8284816a6e54847b0bb6c156723135324fffb377faba"
    ),
    "src/humanbrain/_native_src/src/temporal_runtime/typed_workspace.rs": (
        "ebcdfb848b33b29e45709042216a44efbec535bbebe16513ee933722bf7e3874"
    ),
    "src/humanbrain/_native_src/src/temporal_runtime/typed_hole_state.rs": (
        "e61a9e8101b5447ef245eda161e99a4e3631f1582b86f37213a36983d9774827"
    ),
}

CLAMP_SUM_KEY = ("t2_clamp_then_sum", 4)
TRAIN_KEYS = (CLAMP_SUM_KEY,)
TARGET_KEY = ("t0_scalar_clamp", 4)
KNOWN_USED_T2_CLAMP_THEN_SUM_KEYS = frozenset(
    {
        ("t2_clamp_then_sum", 0),
        ("t2_clamp_then_sum", 1),
        ("t2_clamp_then_sum", 2),
        ("t2_clamp_then_sum", 3),
        ("t2_clamp_then_sum", 6),
    }
)
KNOWN_SPENT_VISIBLE_TRAIN_TARGET_KEYS = frozenset(
    {
        ("t0_scalar_offset", 3),
        ("t0_scalar_clamp", 3),
        ("t0_scalar_clamp", 7),
        ("t1_list_map_add", 2),
        ("t1_list_map_add", 3),
        ("t1_list_map_add", 7),
        ("t2_clamp_then_sum", 1),
        ("t2_clamp_then_sum", 3),
        ("t2_map_then_count", 1),
    }
)
if CLAMP_SUM_KEY in KNOWN_USED_T2_CLAMP_THEN_SUM_KEYS:
    raise RuntimeError("root-operation-transfer donor is already used")
if TARGET_KEY in KNOWN_SPENT_VISIBLE_TRAIN_TARGET_KEYS:
    raise RuntimeError("root-operation-transfer target is already spent")
EXPECTED_PROTOCOL_ID = "human-cell-simple-swe-v1"
EXPECTED_PROTOCOL_SHA256 = (
    "f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2"
)

EXPECTED_SOURCE_RECORDS: dict[tuple[str, int], dict[str, str | None]] = {
    CLAMP_SUM_KEY: {
        "task_id": "simple-swe-v1/train/tier-2/t2_clamp_then_sum/004",
        "entry_point": "solve_3a8f3044aebcda43",
        "parameter": "values",
        "lower": "-7",
        "upper": "7",
        "delta": None,
        "threshold": None,
        "task_sha256": "0fda43c0174b1889fce3f4905874baccb7f7093918a615fb491827e4f1f97f36",
        "hidden_test_sha256": "e6ac0ce5c7533c4bcd2442730cf353a5477c6f64b86339efae79c49b744dfd15",
        "public_prompt_sha256": "d6e38c184e14be4db1c06798aff933b513d4547578ead335dc538b6014aa672f",
    },
    TARGET_KEY: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_clamp/004",
        "entry_point": "solve_34d8ee337c1af11d",
        "parameter": "n",
        "lower": "-2",
        "upper": "2",
        "delta": None,
        "threshold": None,
        "task_sha256": "56269c68b1146a8e33cc025be078fc63cfa13b29c0e72ba40962366f3c12946b",
        "hidden_test_sha256": "74af584d91f30a0257b055f17c5bfc4d9d32f6872687d5c7f57271ff33b5a9f2",
        "public_prompt_sha256": "735b211249e0d4324140cb6ffd53580494513262dffe6688ecc3b10508414b38",
    },
}
EXPECTED_PROMPT_TOKEN_COUNTS = {
    CLAMP_SUM_KEY: 284,
    TARGET_KEY: 273,
}
EXPECTED_TARGET_TOKENS = (
    "if",
    "less-than",
    "variable-0",
    "integer",
    "-",
    "2",
    "end",
    "integer",
    "-",
    "2",
    "end",
    "if",
    "less-than",
    "integer",
    "+",
    "2",
    "end",
    "variable-0",
    "integer",
    "+",
    "2",
    "end",
    "variable-0",
)
EXPECTED_TARGET_SOURCE = (
    "def solve_34d8ee337c1af11d(n):\n"
    "    return -2 if n < -2 else 2 if 2 < n else n\n"
)

# Frozen root-operation-transfer accounting.
PLAN_COMPONENT_ARRIVALS = {CLAMP_SUM_KEY: 6}
PHYSICAL_LEARN_STEPS_PER_ELIGIBLE_TEACHER = 2
EXPECTED_TEACHER_CALL_COUNTS = {CLAMP_SUM_KEY: 24}
EXPECTED_PHYSICAL_TEACHER_ARRIVALS = {CLAMP_SUM_KEY: 48}
EXPECTED_EPISODE_LEARNING_ARRIVALS = {CLAMP_SUM_KEY: 338}
EXPECTED_BASE_LEARNING_ARRIVALS = 338
EXPECTED_POINTER_CUE_LEARNING_ARRIVALS = 11
EXPECTED_TOTAL_LEARNING_ARRIVALS = 349
EXPECTED_BASE_NONINITIAL_LEARNING_ARRIVALS = 337
EXPECTED_NONINITIAL_LEARNING_ARRIVALS = 348
EXPECTED_BASE_SEGMENT_ROWS = 3_707
ROLE_BEARING_CONSTRUCTORS = (*harness.ROLE_BEARING_CONSTRUCTORS, "integer")
MAX_POINTER_ROWS_PER_CUE_CALL = 22
HEAD_TO_ROOT_TRANSFER_TEACHER_POSITION = 3
HEAD_TO_ROOT_TRANSFER_LEARNING_ARRIVALS = 1
HEAD_TO_ROOT_TRANSFER_SEGMENT_ROWS = 11
EXPECTED_POINTER_CUE_SEGMENT_ROWS_UPPER_BOUND = 121
EXPECTED_POINTER_RECURRENCE_ROWS_UPPER_BOUND = 121
EXPECTED_POINTER_SEGMENT_ROWS_UPPER_BOUND = 242
EXPECTED_ORDINARY_SEGMENT_ROWS_UPPER_BOUND = 3_828
EXPECTED_SEGMENT_CAPACITY = 3_949

_PointerOccurrence = harness._PointerOccurrence
POINTER_OCCURRENCES_BY_DONOR = {
    CLAMP_SUM_KEY: (
        _PointerOccurrence(0, "sum", "expression-int-head", (), (), None),
        _PointerOccurrence(1, "map", "expression-list-int-child-0", (0,), (0,), 0),
        _PointerOccurrence(2, "bind", "expression-int-to-int-child-0", (0, 0), (0,), 1),
        _PointerOccurrence(3, "if", "expression-int-head", (0, 0, 0), (), None),
        _PointerOccurrence(4, "less-than", "expression-bool-child-0", (0, 0, 0, 0), (0,), 3),
        _PointerOccurrence(6, "integer", "expression-int-child-1", (0, 0, 0, 0, 1), (1,), 4),
        _PointerOccurrence(9, "integer", "expression-int-child-1", (0, 0, 0, 1), (1,), 3),
        _PointerOccurrence(12, "if", "expression-int-child-2", (0, 0, 0, 2), (2,), 3),
        _PointerOccurrence(13, "less-than", "expression-bool-child-0", (0, 0, 0, 2, 0), (0,), 7),
        _PointerOccurrence(14, "integer", "expression-int-child-0", (0, 0, 0, 2, 0, 0), (0,), 8),
        _PointerOccurrence(18, "integer", "expression-int-child-1", (0, 0, 0, 2, 1), (1,), 7),
    ),
}

_WorkspaceEpisode = harness._WorkspaceEpisode
_TargetRuntimeInput = harness._TargetRuntimeInput
_TypedWorkspaceApi = harness._TypedWorkspaceApi
_typed_value = harness._typed_value
_workspace_vocabulary = harness._workspace_vocabulary


@dataclass(frozen=True, slots=True)
class _TaskFields:
    entry_point: bytes
    parameter: bytes
    lower: bytes | None
    upper: bytes | None
    delta: bytes | None
    threshold: bytes | None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", type=Path, required=True)
    return parser.parse_args()


def _assert_one_shot_paths_absent() -> None:
    for kind, path in (("spend marker", SPEND_MARKER_PATH), ("one-shot artifact", OUTPUT_PATH)):
        if path.exists():
            raise RuntimeError(f"refusing spent target; {kind} exists: {path}")


def _create_spend_marker() -> dict[str, object]:
    """Durably and exclusively spend the target immediately before generation."""

    _assert_one_shot_paths_absent()
    marker_payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-native-typed-root-operation-transfer",
        "state": "TARGET_ATTEMPT_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target_key": list(TARGET_KEY),
        "target_task_id": EXPECTED_SOURCE_RECORDS[TARGET_KEY]["task_id"],
        "canonical_artifact_path": _display_path(OUTPUT_PATH),
        "probe_source_sha256": _sha256_file(Path(__file__)),
    }
    serialized = json.dumps(marker_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with SPEND_MARKER_PATH.open("x", encoding="utf-8", newline="\n") as marker:
            marker.write(serialized)
            marker.flush()
            os.fsync(marker.fileno())
    except FileExistsError as error:
        raise RuntimeError(f"refusing spent target; spend marker exists: {SPEND_MARKER_PATH}") from error
    if OUTPUT_PATH.exists():
        raise RuntimeError("one-shot artifact appeared while creating spend marker; marker retained")
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
        raise RuntimeError("committed root-operation-transfer helper identity changed")
    expected_native_paths = {
        path.relative_to(REPO_ROOT).as_posix() for path in NATIVE_MECHANISM_PATHS
    }
    if set(EXPECTED_NATIVE_MECHANISM_SHA256) != expected_native_paths:
        raise RuntimeError("native root-operation-transfer mechanism source pins are incomplete")
    for path in NATIVE_MECHANISM_PATHS:
        source = path.relative_to(REPO_ROOT).as_posix()
        if _sha256_file(path) != EXPECTED_NATIVE_MECHANISM_SHA256[source]:
            raise RuntimeError(f"native mechanism source identity changed: {source}")
    for path in (Path(__file__).resolve(), HELPER_PATH.resolve(), *NATIVE_MECHANISM_PATHS):
        source = path.relative_to(REPO_ROOT).as_posix()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", source],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout:
            raise RuntimeError(f"probe, helper, and native sources must be committed: {source}")
    if SLOT_COUNT != 4:
        raise RuntimeError("root-operation-transfer probe requires four transport slots")


def _select_tasks() -> tuple[object, tuple[GeneratedTask, ...], GeneratedTask]:
    protocol = load_protocol(PROTOCOL_PATH)
    if protocol.protocol_id != EXPECTED_PROTOCOL_ID or protocol.sha256 != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("public Simple-SWE protocol identity changed")
    visible_train = generate_visible_partition(protocol, "train")
    selected: dict[tuple[str, int], GeneratedTask] = {}
    for key in (*TRAIN_KEYS, TARGET_KEY):
        family, ordinal = key
        try:
            task = next(
                item for item in visible_train
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
        raise TypeError("root-operation-transfer parsing requires the exact public task view")
    key = (public_task.family, public_task.ordinal)
    if key not in EXPECTED_SOURCE_RECORDS:
        raise RuntimeError(f"unsupported root-operation-transfer task key {key!r}")
    expected = EXPECTED_SOURCE_RECORDS[key]
    signature = public_task.signature
    expected_parameter_type = (
        ProgramType.List(ProgramType.Int())
        if key == CLAMP_SUM_KEY
        else ProgramType.Int()
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
        "public_prompt_sha256": hashlib.sha256(public_task.prompt.encode("utf-8")).hexdigest(),
    }
    expected_provenance = {
        name: expected[name]
        for name in ("task_id", "entry_point", "parameter", "task_sha256", "hidden_test_sha256", "public_prompt_sha256")
    }
    if observed != expected_provenance or public_task.partition != "train":
        raise RuntimeError(f"visible task provenance changed for {key!r}: {observed!r}")

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
    phrases = {
        CLAMP_SUM_KEY: b"Limit every integer to -7 through 7, then return their sum",
        TARGET_KEY: b"inclusive range -2 through 2",
    }
    if phrases[key] not in prompt:
        raise RuntimeError(f"visible task wording changed for {key!r}")
    return fields


def _canonical_literal_primitives(literal: bytes) -> tuple[str, ...]:
    return harness._canonical_literal_primitives(literal)


def _prompt_spans(public_task: object) -> tuple[bytes, tuple[tuple[int, int, int, tuple[int, ...]], ...]]:
    fields = _task_fields(public_task)
    key = (public_task.family, public_task.ordinal)
    prompt = public_task.prompt.encode("ascii")
    signature = b"def " + fields.entry_point + b"(" + fields.parameter + b"):"
    signature_start = prompt.index(signature)
    entry_start = signature_start + len(b"def ")
    parameter_start = entry_start + len(fields.entry_point) + 1
    spans: list[tuple[int, int, int, tuple[int, ...]]] = [
        (entry_start, entry_start + len(fields.entry_point), 0, tuple(fields.entry_point)),
        (parameter_start, parameter_start + len(fields.parameter), 1, tuple(fields.parameter)),
    ]
    if key == CLAMP_SUM_KEY:
        assert fields.lower is not None and fields.upper is not None
        lower_start = prompt.index(b"Limit every integer to ", parameter_start) + len(b"Limit every integer to ")
        upper_start = lower_start + len(fields.lower) + len(b" through ")
        literal_spans = ((lower_start, fields.lower, 2), (upper_start, fields.upper, 3))
    else:
        assert fields.lower is not None and fields.upper is not None
        lower_start = prompt.index(b"inclusive range ", parameter_start) + len(b"inclusive range ")
        upper_start = lower_start + len(fields.lower) + len(b" through ")
        literal_spans = ((lower_start, fields.lower, 2), (upper_start, fields.upper, 3))
    for start, literal, slot in literal_spans:
        if prompt[start : start + len(literal)] != literal:
            raise RuntimeError(f"visible slot {slot} span no longer matches")
        spans.append(
            (start, start + len(literal), slot, tuple(PRIMITIVE_TOKEN_IDS[token] for token in _canonical_literal_primitives(literal)))
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
    key = (public_task.family, public_task.ordinal)
    if _detokenize_prompt(result) != prompt or len(result) != EXPECTED_PROMPT_TOKEN_COUNTS[key]:
        raise RuntimeError(f"lossless public prompt tokenization changed for {key!r}")
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


def _encode_teacher(primitives: tuple[str | tuple[str, int], ...]) -> tuple[int, ...]:
    return (
        *(copy_slot_token_id(item[1]) if isinstance(item, tuple) else PRIMITIVE_TOKEN_IDS[item] for item in primitives),
        base.EOS_TOKEN_ID,
    )


def _internal_teacher_tokens(key: tuple[str, int]) -> tuple[int, ...]:
    if key == CLAMP_SUM_KEY:
        return _encode_teacher((
            "sum", "map", "bind", "if", "less-than", "variable-0", "integer", ("copy", 2), "end",
            "integer", ("copy", 2), "end", "if", "less-than", "integer", ("copy", 3), "end",
            "variable-0", "integer", ("copy", 3), "end", "variable-0", "variable-0",
        ))
    raise RuntimeError(f"unsupported donor teacher key {key!r}")


def _expected_program_tokens(public_task: object) -> tuple[str, ...]:
    fields = _task_fields(public_task)
    key = (public_task.family, public_task.ordinal)
    if key == CLAMP_SUM_KEY:
        assert fields.lower is not None and fields.upper is not None
        lower = _canonical_literal_primitives(fields.lower)
        upper = _canonical_literal_primitives(fields.upper)
        return (
            "sum", "map", "bind", "if", "less-than", "variable-0", "integer", *lower, "end",
            "integer", *lower, "end", "if", "less-than", "integer", *upper, "end", "variable-0",
            "integer", *upper, "end", "variable-0", "variable-0",
        )
    if key == TARGET_KEY:
        assert fields.lower is not None and fields.upper is not None
        lower = _canonical_literal_primitives(fields.lower)
        upper = _canonical_literal_primitives(fields.upper)
        result = (
            "if", "less-than", "variable-0", "integer", *lower, "end",
            "integer", *lower, "end", "if", "less-than", "integer", *upper,
            "end", "variable-0", "integer", *upper, "end", "variable-0",
        )
        if result != EXPECTED_TARGET_TOKENS:
            raise RuntimeError("frozen scalar target token stream changed")
        return result
    raise RuntimeError(f"unsupported expected-program key {key!r}")


def _expected_compiled_source(public_task: object) -> str:
    fields = _task_fields(public_task)
    key = (public_task.family, public_task.ordinal)
    entry = fields.entry_point.decode("ascii")
    parameter = fields.parameter.decode("ascii")
    if key == CLAMP_SUM_KEY:
        assert fields.lower is not None and fields.upper is not None
        lower = fields.lower.decode("ascii")
        upper = fields.upper.decode("ascii")
        return (
            f"def {entry}({parameter}):\n"
            f"    return sum([{lower} if item_0 < {lower} else {upper} if {upper} < item_0 else item_0 for item_0 in {parameter}])\n"
        )
    if key == TARGET_KEY:
        assert fields.lower is not None and fields.upper is not None
        lower = fields.lower.decode("ascii")
        upper = fields.upper.decode("ascii")
        source = (
            f"def {entry}({parameter}):\n"
            f"    return {lower} if {parameter} < {lower} else {upper} if {upper} < {parameter} else {parameter}\n"
        )
        if source != EXPECTED_TARGET_SOURCE:
            raise RuntimeError("frozen scalar target source changed")
        return source
    raise RuntimeError(f"unsupported expected-source key {key!r}")


def _expected_copy_schedule(public_task: object) -> tuple[tuple[int, tuple[str, ...]], ...]:
    fields = _task_fields(public_task)
    key = (public_task.family, public_task.ordinal)
    assert fields.lower is not None and fields.upper is not None
    lower = _canonical_literal_primitives(fields.lower)
    upper = _canonical_literal_primitives(fields.upper)
    return ((2, lower), (2, lower), (3, upper), (3, upper))


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
        "threshold": None if fields.threshold is None else fields.threshold.decode("ascii"),
        "task_sha256": public_task.task_sha256,
        "hidden_test_sha256": public_task.hidden_test_sha256,
        "public_prompt_sha256": hashlib.sha256(public_task.prompt.encode("utf-8")).hexdigest(),
        "public_prompt_lossless_after_tokenization": _detokenize_prompt(prompt_tokens) == public_task.prompt.encode("ascii"),
        "prompt_token_count": len(prompt_tokens),
    }


@contextmanager
def _without_head_to_root_capacity_delta() -> object:
    """Expose the inherited accounting contract while deriving its one-row delta."""

    overrides = {
        "EXPECTED_PHYSICAL_TEACHER_ARRIVALS": {CLAMP_SUM_KEY: 47},
        "EXPECTED_EPISODE_LEARNING_ARRIVALS": {CLAMP_SUM_KEY: 337},
        "EXPECTED_BASE_LEARNING_ARRIVALS": 337,
        "EXPECTED_TOTAL_LEARNING_ARRIVALS": 348,
        "EXPECTED_BASE_NONINITIAL_LEARNING_ARRIVALS": 336,
        "EXPECTED_NONINITIAL_LEARNING_ARRIVALS": 347,
        "EXPECTED_BASE_SEGMENT_ROWS": 3_696,
        "EXPECTED_ORDINARY_SEGMENT_ROWS_UPPER_BOUND": 3_817,
        "EXPECTED_SEGMENT_CAPACITY": 3_938,
    }
    previous = {name: getattr(harness, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(harness, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(harness, name, value)


def _configured_learning_arrival_accounting(
    prompts: dict[tuple[str, int], tuple[int, ...]],
    teachers: dict[tuple[str, int], tuple[int, ...]],
) -> dict[str, object]:
    with _without_head_to_root_capacity_delta():
        accounting = _BASE_LEARNING_ARRIVAL_ACCOUNTING(prompts, teachers)

    episode_key = f"{CLAMP_SUM_KEY[0]}/{CLAMP_SUM_KEY[1]:03d}"
    episode = accounting["episodes"][episode_key]
    for field in (
        "physical_teacher_arrivals",
        "base_learning_arrivals",
        "total_learning_arrivals",
    ):
        episode[field] += HEAD_TO_ROOT_TRANSFER_LEARNING_ARRIVALS
    episode["head_to_root_transfer_learning_arrivals"] = (
        HEAD_TO_ROOT_TRANSFER_LEARNING_ARRIVALS
    )
    episode["head_to_root_transfer_segment_rows"] = (
        HEAD_TO_ROOT_TRANSFER_SEGMENT_ROWS
    )

    for field in (
        "base_learning_arrivals",
        "total_learning_arrivals",
        "noninitial_learning_arrivals",
    ):
        accounting[field] += HEAD_TO_ROOT_TRANSFER_LEARNING_ARRIVALS
    for field in (
        "base_temporal_segment_rows_upper_bound",
        "ordinary_segment_rows_upper_bound_including_pointer_cues",
        "segment_capacity",
        "reserve_pyramidal_cells",
        "max_pyramidal_cells",
    ):
        accounting[field] += HEAD_TO_ROOT_TRANSFER_SEGMENT_ROWS
    accounting["head_to_root_transfer"] = {
        "projection": "HeadToRootTransfer",
        "teacher_position": HEAD_TO_ROOT_TRANSFER_TEACHER_POSITION,
        "learning_arrivals": HEAD_TO_ROOT_TRANSFER_LEARNING_ARRIVALS,
        "ordinary_segment_rows_upper_bound": HEAD_TO_ROOT_TRANSFER_SEGMENT_ROWS,
        "direction_isolated": True,
    }
    return accounting


@contextmanager
def _configured_harness() -> object:
    overrides = {
        "TRAIN_KEYS": TRAIN_KEYS,
        "EXPECTED_PROMPT_TOKEN_COUNTS": EXPECTED_PROMPT_TOKEN_COUNTS,
        "ADDRESS_OFFSETS": ADDRESS_OFFSETS,
        "ADDRESS_TOKEN_IDS": ADDRESS_TOKEN_IDS,
        "TYPED_PRIVATE_TOKEN_IDS": TYPED_PRIVATE_TOKEN_IDS,
        "_RESERVED_COLUMN_COUNT": _RESERVED_COLUMN_COUNT,
        "PLAN_COMPONENT_ARRIVALS": PLAN_COMPONENT_ARRIVALS,
        "PHYSICAL_LEARN_STEPS_PER_ELIGIBLE_TEACHER": PHYSICAL_LEARN_STEPS_PER_ELIGIBLE_TEACHER,
        "EXPECTED_TEACHER_CALL_COUNTS": EXPECTED_TEACHER_CALL_COUNTS,
        "EXPECTED_PHYSICAL_TEACHER_ARRIVALS": EXPECTED_PHYSICAL_TEACHER_ARRIVALS,
        "EXPECTED_EPISODE_LEARNING_ARRIVALS": EXPECTED_EPISODE_LEARNING_ARRIVALS,
        "EXPECTED_BASE_LEARNING_ARRIVALS": EXPECTED_BASE_LEARNING_ARRIVALS,
        "EXPECTED_POINTER_CUE_LEARNING_ARRIVALS": EXPECTED_POINTER_CUE_LEARNING_ARRIVALS,
        "EXPECTED_TOTAL_LEARNING_ARRIVALS": EXPECTED_TOTAL_LEARNING_ARRIVALS,
        "EXPECTED_BASE_NONINITIAL_LEARNING_ARRIVALS": EXPECTED_BASE_NONINITIAL_LEARNING_ARRIVALS,
        "EXPECTED_NONINITIAL_LEARNING_ARRIVALS": EXPECTED_NONINITIAL_LEARNING_ARRIVALS,
        "EXPECTED_BASE_SEGMENT_ROWS": EXPECTED_BASE_SEGMENT_ROWS,
        "EXPECTED_ORDINARY_SEGMENT_ROWS_UPPER_BOUND": EXPECTED_ORDINARY_SEGMENT_ROWS_UPPER_BOUND,
        "EXPECTED_POINTER_CUE_SEGMENT_ROWS_UPPER_BOUND": EXPECTED_POINTER_CUE_SEGMENT_ROWS_UPPER_BOUND,
        "EXPECTED_POINTER_RECURRENCE_ROWS_UPPER_BOUND": EXPECTED_POINTER_RECURRENCE_ROWS_UPPER_BOUND,
        "EXPECTED_POINTER_SEGMENT_ROWS_UPPER_BOUND": EXPECTED_POINTER_SEGMENT_ROWS_UPPER_BOUND,
        "EXPECTED_SEGMENT_CAPACITY": EXPECTED_SEGMENT_CAPACITY,
        "ROLE_BEARING_CONSTRUCTORS": ROLE_BEARING_CONSTRUCTORS,
        "POINTER_OCCURRENCES_BY_DONOR": POINTER_OCCURRENCES_BY_DONOR,
        "_tokenized_public_prompt": _tokenized_public_prompt,
        "_internal_teacher_tokens": _internal_teacher_tokens,
        "_learning_arrival_accounting": _configured_learning_arrival_accounting,
        "_infer_pointer_rows_from_teacher_delta": _infer_pointer_rows_from_teacher_delta,
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


def _learning_arrival_accounting(prompts: dict[tuple[str, int], tuple[int, ...]], teachers: dict[tuple[str, int], tuple[int, ...]]) -> dict[str, object]:
    with _configured_harness():
        return _configured_learning_arrival_accounting(prompts, teachers)


def _definition(train_tasks: tuple[GeneratedTask, ...], target_input: _TargetRuntimeInput) -> tuple[object, dict[str, object]]:
    with _configured_harness():
        return harness._definition(train_tasks, target_input)


def _infer_pointer_rows_from_teacher_delta(*, learning_step_delta: int, segment_row_delta: int, returned_exact_rows: int) -> tuple[int, int]:
    if learning_step_delta == 4:
        learning_step_delta -= HEAD_TO_ROOT_TRANSFER_LEARNING_ARRIVALS
        segment_row_delta -= HEAD_TO_ROOT_TRANSFER_SEGMENT_ROWS
    return _BASE_INFER_POINTER_ROWS(
        learning_step_delta=learning_step_delta,
        segment_row_delta=segment_row_delta,
        returned_exact_rows=returned_exact_rows,
    )


def _train_task_with_head_to_root_accounting(
    circuit: NativeTemporalCircuit,
    *,
    key: tuple[str, int],
    prompt_tokens: tuple[int, ...],
    parameter_type: object,
    result_type: object,
    teacher_tokens: tuple[int, ...],
) -> tuple[_WorkspaceEpisode, tuple[object, ...]]:
    with _configured_harness():
        episode, observations = harness._train_task_with_pointer_accounting(
            circuit,
            key=key,
            prompt_tokens=prompt_tokens,
            parameter_type=parameter_type,
            result_type=result_type,
            teacher_tokens=teacher_tokens,
        )
    observed_deltas = {
        observation.teacher_position: observation.learning_step_delta
        for observation in observations
    }
    expected_deltas = {
        occurrence.teacher_position: (
            4
            if occurrence.teacher_position == HEAD_TO_ROOT_TRANSFER_TEACHER_POSITION
            else 3
        )
        for occurrence in POINTER_OCCURRENCES_BY_DONOR[key]
    }
    if observed_deltas != expected_deltas:
        raise RuntimeError(
            "HeadToRootTransfer must add exactly one Learn race at donor position 3"
        )
    return episode, observations


def _failure_localization(*, gate: str, generation_position: int | None, expected_tokens: tuple[str, ...]) -> dict[str, object]:
    if generation_position is None:
        scope, expected_surface = "post-generation", None
    else:
        scope = "root" if generation_position == 0 else "descendant"
        expected_surface = expected_tokens[generation_position] if generation_position < len(expected_tokens) else "<EOS>"
    return {
        "gate": gate,
        "scope": scope,
        "generation_position": generation_position,
        "expected_surface": expected_surface,
        "root_expected_surface": expected_tokens[0],
        "root_operation_transfer_is_probe_focus": True,
    }


def _generation_failure(error: NativeTemporalRuntimeError, expected_tokens: tuple[str, ...]) -> dict[str, object]:
    position = int(error.completed_prefix)
    return {
        "kind": "model",
        "gate": "GENERATION_STOP",
        "localization": _failure_localization(gate="GENERATION_STOP", generation_position=position, expected_tokens=expected_tokens),
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


def _score_target(**kwargs: object) -> tuple[dict[str, object], dict[str, object] | None]:
    with _configured_harness():
        record, first_failure = harness._score_target(**kwargs)
    if first_failure is not None and "localization" not in first_failure:
        gate = str(first_failure["gate"])
        position: int | None = None
        if gate == "EXACT_TYPED_TOKEN_STREAM":
            observed = tuple(int(generation["surface_token_id"]) for generation in record["generations"])
            expected = (*(PRIMITIVE_TOKEN_IDS[token] for token in EXPECTED_TARGET_TOKENS), base.EOS_TOKEN_ID)
            sentinel = object()
            position = next((i for i, pair in enumerate(zip_longest(observed, expected, fillvalue=sentinel)) if pair[0] != pair[1]), None)
        elif gate == "ORDINARY_EOS":
            position = len(record["generations"])
        first_failure["localization"] = _failure_localization(gate=gate, generation_position=position, expected_tokens=EXPECTED_TARGET_TOKENS)
    record["first_failure_localization"] = None if first_failure is None else first_failure["localization"]
    return record, first_failure


def _teacher_label(token_id: int) -> str:
    for slot in (2, 3):
        if token_id == copy_slot_token_id(slot):
            return f"COPY_SLOT_{slot}"
    if token_id == base.EOS_TOKEN_ID:
        return "<EOS>"
    return PRIMITIVE_BY_TOKEN_ID[token_id]


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
        raise RuntimeError("root-operation-transfer probe requires repo Python 3.11.15")
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = hashlib.sha256(worker_path.read_bytes()).hexdigest()
    worker_source = load_worker_snapshot(worker_sha256)

    trained: list[tuple[GeneratedTask, object, object, tuple[object, ...]]] = []
    native_error: NativeTemporalRuntimeError | None = None
    with base._public_development_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            for key, task in zip(TRAIN_KEYS, train_tasks, strict=True):
                signature = task.public.signature
                episode, observations = _train_task_with_head_to_root_accounting(
                    circuit,
                    key=key,
                    prompt_tokens=_tokenized_public_prompt(task.public),
                    parameter_type=signature.parameters[0].type,
                    result_type=signature.result,
                    teacher_tokens=teachers[key],
                )
                trained.append((task, episode, circuit.checkpoint_identity(), observations))
            trained_identity = circuit.checkpoint_identity()
            if trained_identity.learning_step != EXPECTED_TOTAL_LEARNING_ARRIVALS:
                raise RuntimeError("native learning-arrival accounting changed")
            target_episode = _TypedWorkspaceApi.begin_target(circuit, target_input)
            try:
                spend_marker_record = _create_spend_marker()
                raw_generations = circuit.generate_typed()
            except NativeTemporalRuntimeError as error:
                raw_generations = None
                native_error = error
        native_identity = loaded_library_identity()

    native_failure = None if native_error is None else _generation_failure(native_error, EXPECTED_TARGET_TOKENS)
    compiler_records: list[dict[str, object]] = []
    for task in train_tasks:
        tokens = _expected_program_tokens(task.public)
        compiled_source = compile_typed_program(Program(tokens, task.public.signature)).source
        expected_source = _expected_compiled_source(task.public)
        if compiled_source != expected_source:
            raise RuntimeError(f"typed compiler cannot express donor {task.public.task_id}")
        compiler_records.append({"task_id": task.public.task_id, "tokens": list(tokens), "compiled_source": compiled_source, "exact": True})
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
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-native-typed-root-operation-transfer",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": "PASS_NATIVE_TYPED_ROOT_OPERATION_TRANSFER" if passed else "STOP_NATIVE_TYPED_ROOT_OPERATION_TRANSFER",
        "capability_pass": passed,
        "first_failure": first_failure,
        "claim_boundary": (
            "One retained native circuit learns visible TRAIN clamp-sum/004, "
            "then attempts visible TRAIN scalar-clamp/004 once without a "
            "teacher. This fresh, family-disjoint target tests only whether "
            "root Expression(Int) can retrieve the donor's nested clamp If "
            "through direction-isolated HeadToRootTransfer. PASS would not "
            "establish descendant "
            "assembly, literal relocation, broad transfer, or coding ability. "
            "It is not DEV, eval, held-out, M0, or C1 evidence."
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
            "atomic_no_argument_generation_call_count": 1,
            "caller_supplied_target_tokens": False,
        },
        "target_spend_marker": {**spend_marker_record, "exists_after_attempt": True, "retained_on_model_failure": True},
        "protocol": {"protocol_id": protocol.protocol_id, "protocol_sha256": protocol.sha256},
        "typed_programs": {"donor_compiler_checks": compiler_records, "target_expected_tokens": list(EXPECTED_TARGET_TOKENS), "target_expected_source": EXPECTED_TARGET_SOURCE},
        "root_operation_transfer_contract": {
            "mechanism_required_before_run": True,
            "donor_source": "nested If at t2_clamp_then_sum/004 teacher position 3",
            "donor_cue": "expression-int-head",
            "target_cue": "root expression-int-head",
            "required_projection": "HeadToRootTransfer",
            "required_first_surface": "if",
            "descendant_assembly_claimed": False,
        },
        "physical_definition": {"circuit_identity_sha256": definition.circuit_identity_sha256(), "segment_capacity": definition.config.segment_capacity, "capacity_accounting": capacity_accounting},
        "train": {
            "tasks": [
                {
                    "task": _public_task_record(task.public),
                    "teacher": [_teacher_label(token) for token in teachers[(task.public.family, task.public.ordinal)]],
                    "live_segments_after_task": identity.segment_count,
                    "learning_steps_after_task": identity.learning_step,
                }
                for task, _episode, identity, _observations in trained
            ],
            "live_segments_after_train": trained_identity.segment_count,
            "learning_steps_after_train": trained_identity.learning_step,
        },
        "visible_train_target_teacher_free": target_record,
        "runtime": {
            "elapsed_wall_ns": time.perf_counter_ns() - started_ns,
            "python": sys.version,
            "platform": platform.platform(),
            "native_source_digest": native_identity.source_digest,
            "native_binary_sha256": native_identity.binary_sha256,
            "probe_source_sha256": _sha256_file(Path(__file__)),
            "helper_source_sha256": _sha256_file(HELPER_PATH),
            "native_mechanism_source_sha256": {path.relative_to(REPO_ROOT).as_posix(): _sha256_file(path) for path in NATIVE_MECHANISM_PATHS},
        },
    }
    with OUTPUT_PATH.open("x", encoding="utf-8", newline="\n") as output_file:
        output_file.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"outcome": payload["outcome"], "first_failure": first_failure, "output": str(OUTPUT_PATH)}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
