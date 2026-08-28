"""Prepared one-shot visible-TRAIN cross-family literal-leaf-pointer probe.

One retained native temporal circuit learns visible TRAIN
``t2_clamp_then_sum/006``.  It then receives visible TRAIN
``t0_scalar_clamp/003`` exactly once without a teacher.  The target requires
the learned clamp subtree—including ``Integer -> sign`` and lower/upper COPY
choice—to relocate from inside ``Sum/Map/Bind`` to scalar root.  Importing or
compiling spends nothing.  The committed native mechanism is source-pinned
before the one-shot target can execute.
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
    raise RuntimeError("literal-leaf-pointer private-code reserve changed")

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
        "cb149996120261158f0cb2252e970b78d146be4c95782a5599c60ad88ce43be3"
    ),
    "src/humanbrain/_native_src/src/temporal_runtime/checkpoint.rs": (
        "bdb1ab444a826e706b9c8284816a6e54847b0bb6c156723135324fffb377faba"
    ),
    "src/humanbrain/_native_src/src/temporal_runtime/typed_workspace.rs": (
        "66fea07398382565d3850e5b02ddb564f064dd61eaffa1d9b912766b1879727c"
    ),
    "src/humanbrain/_native_src/src/temporal_runtime/typed_hole_state.rs": (
        "e61a9e8101b5447ef245eda161e99a4e3631f1582b86f37213a36983d9774827"
    ),
}

CLAMP_SUM_KEY = ("t2_clamp_then_sum", 6)
TRAIN_KEYS = (CLAMP_SUM_KEY,)
TARGET_KEY = ("t0_scalar_clamp", 3)
KNOWN_SPENT_VISIBLE_TRAIN_TARGET_KEYS = frozenset(
    {
        ("t0_scalar_offset", 3),
        ("t0_scalar_clamp", 7),
        ("t1_list_map_add", 2),
        ("t1_list_map_add", 3),
        ("t1_list_map_add", 7),
        ("t2_clamp_then_sum", 1),
        ("t2_clamp_then_sum", 3),
        ("t2_map_then_count", 1),
    }
)
if TARGET_KEY in KNOWN_SPENT_VISIBLE_TRAIN_TARGET_KEYS:
    raise RuntimeError("literal-leaf target is already spent")
EXPECTED_PROTOCOL_ID = "human-cell-simple-swe-v1"
EXPECTED_PROTOCOL_SHA256 = (
    "f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2"
)

EXPECTED_SOURCE_RECORDS: dict[tuple[str, int], dict[str, str | None]] = {
    CLAMP_SUM_KEY: {
        "task_id": "simple-swe-v1/train/tier-2/t2_clamp_then_sum/006",
        "entry_point": "solve_eb0c75562d9b056e",
        "parameter": "values",
        "lower": "-8",
        "upper": "8",
        "delta": None,
        "threshold": None,
        "task_sha256": "a8fe9ddcb86b802e43ac75a119b47b24f5f3068b10184241e40af59fa0f80615",
        "hidden_test_sha256": "387ba427c83783d53d5506fb235dc48eb9cd688ef2fb01fb494623453cd076d4",
        "public_prompt_sha256": "df65c83afc1168f6c01d1ec357de18f1ef6b5a1ea79c50441273b42c86352232",
    },
    TARGET_KEY: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_clamp/003",
        "entry_point": "solve_ab40788bba21c775",
        "parameter": "n",
        "lower": "-11",
        "upper": "11",
        "delta": None,
        "threshold": None,
        "task_sha256": "2507554dfcf51d21f4fdad81f437736d62e3b98953f0a7d37ec9aa673d5ff8a1",
        "hidden_test_sha256": "e65b577016c3e6dad1c9355e719a72bded2c5b85024e34bdd80c05eec0cbd758",
        "public_prompt_sha256": "da78ecb832d861b2c2eb45f45241a44af0326b99a2e38b2808859f404e2e86dd",
    },
}
EXPECTED_PROMPT_TOKEN_COUNTS = {
    CLAMP_SUM_KEY: 284,
    TARGET_KEY: 275,
}
EXPECTED_TARGET_TOKENS = (
    "if",
    "less-than",
    "variable-0",
    "integer",
    "-",
    "1",
    "1",
    "end",
    "integer",
    "-",
    "1",
    "1",
    "end",
    "if",
    "less-than",
    "integer",
    "+",
    "1",
    "1",
    "end",
    "variable-0",
    "integer",
    "+",
    "1",
    "1",
    "end",
    "variable-0",
)
EXPECTED_TARGET_SOURCE = (
    "def solve_ab40788bba21c775(n):\n"
    "    return -11 if n < -11 else 11 if 11 < n else n\n"
)

# Frozen literal-leaf-pointer accounting.
PLAN_COMPONENT_ARRIVALS = {CLAMP_SUM_KEY: 6}
PHYSICAL_LEARN_STEPS_PER_ELIGIBLE_TEACHER = 2
EXPECTED_TEACHER_CALL_COUNTS = {CLAMP_SUM_KEY: 24}
EXPECTED_PHYSICAL_TEACHER_ARRIVALS = {CLAMP_SUM_KEY: 47}
EXPECTED_EPISODE_LEARNING_ARRIVALS = {CLAMP_SUM_KEY: 337}
EXPECTED_BASE_LEARNING_ARRIVALS = 337
EXPECTED_POINTER_CUE_LEARNING_ARRIVALS = 11
EXPECTED_TOTAL_LEARNING_ARRIVALS = 348
EXPECTED_BASE_NONINITIAL_LEARNING_ARRIVALS = 336
EXPECTED_NONINITIAL_LEARNING_ARRIVALS = 347
EXPECTED_BASE_SEGMENT_ROWS = 3_696
ROLE_BEARING_CONSTRUCTORS = (*harness.ROLE_BEARING_CONSTRUCTORS, "integer")
MAX_POINTER_ROWS_PER_CUE_CALL = 22
EXPECTED_POINTER_CUE_SEGMENT_ROWS_UPPER_BOUND = 121
EXPECTED_POINTER_RECURRENCE_ROWS_UPPER_BOUND = 121
EXPECTED_POINTER_SEGMENT_ROWS_UPPER_BOUND = 242
EXPECTED_ORDINARY_SEGMENT_ROWS_UPPER_BOUND = 3_817
EXPECTED_SEGMENT_CAPACITY = 3_938

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
        "experiment": "r3-public-simple-swe-native-typed-literal-leaf-pointer",
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
        raise RuntimeError("committed literal-leaf helper identity changed")
    expected_native_paths = {
        path.relative_to(REPO_ROOT).as_posix() for path in NATIVE_MECHANISM_PATHS
    }
    if set(EXPECTED_NATIVE_MECHANISM_SHA256) != expected_native_paths:
        raise RuntimeError("native literal-leaf mechanism source pins are incomplete")
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
        raise RuntimeError("literal-leaf-pointer probe requires four transport slots")


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
        raise TypeError("literal-leaf parsing requires the exact public task view")
    key = (public_task.family, public_task.ordinal)
    if key not in EXPECTED_SOURCE_RECORDS:
        raise RuntimeError(f"unsupported literal-leaf task key {key!r}")
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
        CLAMP_SUM_KEY: b"Limit every integer to -8 through 8, then return their sum",
        TARGET_KEY: b"inclusive range -11 through 11",
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
        return harness._learning_arrival_accounting(prompts, teachers)


def _definition(train_tasks: tuple[GeneratedTask, ...], target_input: _TargetRuntimeInput) -> tuple[object, dict[str, object]]:
    with _configured_harness():
        return harness._definition(train_tasks, target_input)


def _infer_pointer_rows_from_teacher_delta(*, learning_step_delta: int, segment_row_delta: int, returned_exact_rows: int) -> tuple[int, int]:
    return harness._infer_pointer_rows_from_teacher_delta(
        learning_step_delta=learning_step_delta,
        segment_row_delta=segment_row_delta,
        returned_exact_rows=returned_exact_rows,
    )


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
        "root_election_is_upstream_of_constructor_pointer": True,
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
        raise RuntimeError("literal-leaf-pointer probe requires repo Python 3.11.15")
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = hashlib.sha256(worker_path.read_bytes()).hexdigest()
    worker_source = load_worker_snapshot(worker_sha256)

    trained: list[tuple[GeneratedTask, object, object, tuple[object, ...]]] = []
    native_error: NativeTemporalRuntimeError | None = None
    with base._public_development_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            for key, task in zip(TRAIN_KEYS, train_tasks, strict=True):
                signature = task.public.signature
                with _configured_harness():
                    episode, observations = harness._train_task_with_pointer_accounting(
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
        "experiment": "r3-public-simple-swe-native-typed-literal-leaf-pointer",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": "PASS_NATIVE_TYPED_LITERAL_LEAF_POINTER" if passed else "STOP_NATIVE_TYPED_LITERAL_LEAF_POINTER",
        "capability_pass": passed,
        "first_failure": first_failure,
        "claim_boundary": (
            "One retained native circuit learns visible TRAIN clamp-sum/006, "
            "then attempts visible TRAIN scalar-clamp/003 once without a "
            "teacher. This fresh target family tests relocation of the learned "
            "clamp subtree from inside Sum/Map/Bind to scalar root, including "
            "Integer-to-sign and bound-slot COPY edges. It is not DEV, eval, "
            "held-out, M0, C1, or broad coding evidence."
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
        "literal_leaf_pointer_contract": {
            "mechanism_required_before_run": True,
            "target_required_edges": [
                "Integer->- at scalar-root lower-bound roles",
                "Integer->+ at scalar-root upper-bound roles",
                "lower/upper bound COPY choice after subtree relocation",
            ],
            "parent_and_relative_hole_conditioned": True,
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
