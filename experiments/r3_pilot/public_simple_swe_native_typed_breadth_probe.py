"""Prepared capped visible-TRAIN breadth probe for the frozen typed mechanism.

The probe trains one native circuit on two public TRAIN donors, exports one
in-memory checkpoint, and restores that exact checkpoint into an independent
circuit for each ordered teacher-free target.  Every target is native-typed
fresh (it was never a teacher-free target of this mechanism), but all visible
TRAIN tasks appeared in the historical prompt-only baseline.  Importing this
module spends nothing; the executable path is deliberately one-shot.
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
from itertools import chain
from pathlib import Path

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PROBE_DIR = Path(__file__).resolve().parent
for import_root in (SRC_ROOT, REPO_ROOT, PROBE_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import public_simple_swe_native_typed_clamp_sum_cross_donor_probe as shared
import public_simple_swe_native_typed_root_operation_transfer_probe as single
import public_simple_swe_native_typed_shift_sum_probe as atomic
from humanbrain.native import loaded_library_identity
from humanbrain.native_temporal_runtime import (
    SLOT_COUNT,
    NativeTemporalCircuit,
    NativeTemporalRuntimeError,
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

base = shared.base
PRIMITIVE_TOKEN_IDS = shared.PRIMITIVE_TOKEN_IDS
PRIMITIVE_BY_TOKEN_ID = shared.PRIMITIVE_BY_TOKEN_ID
PLAN_COMMIT_TOKEN_ID = shared.PLAN_COMMIT_TOKEN_ID
TYPED_CUE_NAMES = shared.TYPED_CUE_NAMES
TYPED_CUE_TOKEN_IDS = shared.TYPED_CUE_TOKEN_IDS
ADDRESS_TOKEN_BASE = shared.ADDRESS_TOKEN_BASE
TYPED_SURFACE_TOKEN_IDS = shared.TYPED_SURFACE_TOKEN_IDS
_TargetRuntimeInput = shared._TargetRuntimeInput
_WorkspaceEpisode = shared._WorkspaceEpisode
_TypedWorkspaceApi = shared._TypedWorkspaceApi
_PointerOccurrence = shared._PointerOccurrence
_BASE_INFER_POINTER_ROWS = shared._infer_pointer_rows_from_teacher_delta

COLUMN_COUNT = 768
# The inherited cross-donor bank omitted the list-map-add Integer hole at path
# (0, 0, 1), offset 14. Replace unused map-count offset 367 with it while
# retaining offset 368 as inert reserve padding; this preserves the audited
# 51-private-code / 768-column geometry and its disjoint surface codebook.
ADDRESS_OFFSETS = tuple(
    sorted({*single.ADDRESS_OFFSETS, 14} - {367})
)
if ADDRESS_OFFSETS != (
    0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 31, 32, 40, 41, 42,
    121, 122, 127, 128, 129, 368, 382, 383,
):
    raise RuntimeError("breadth-probe structural address set changed")
ADDRESS_TOKEN_IDS = tuple(ADDRESS_TOKEN_BASE + offset for offset in ADDRESS_OFFSETS)
TYPED_PRIVATE_TOKEN_IDS = (
    PLAN_COMMIT_TOKEN_ID,
    *TYPED_CUE_TOKEN_IDS.values(),
    *ADDRESS_TOKEN_IDS,
)
_RESERVED_COLUMN_COUNT = len(TYPED_PRIVATE_TOKEN_IDS) * base.ACTIVE_COLUMNS
if _RESERVED_COLUMN_COUNT != 561 or COLUMN_COUNT - _RESERVED_COLUMN_COUNT != 207:
    raise RuntimeError("breadth-probe private-code reserve changed")

OUTPUT_PATH = Path(__file__).with_suffix(".json")
PROTOCOL_PATH = REPO_ROOT / "benchmarks" / "simple_swe_protocol_v1.json"
EXPECTED_PROTOCOL_ID = "human-cell-simple-swe-v1"
EXPECTED_PROTOCOL_SHA256 = (
    "f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2"
)

LIST_ADD_DONOR = ("t1_list_map_add", 6)
CLAMP_SUM_DONOR = ("t2_clamp_then_sum", 8)
TRAIN_KEYS = (LIST_ADD_DONOR, CLAMP_SUM_DONOR)
TARGET_KEYS = (
    ("t1_list_map_add", 5),
    ("t0_scalar_offset", 9),
    ("t0_scalar_clamp", 9),
    ("t2_clamp_then_sum", 7),
)
KNOWN_NATIVE_TYPED_SPENT_TARGET_KEYS = frozenset(
    {
        ("t0_scalar_offset", 3),
        ("t0_scalar_clamp", 3),
        ("t0_scalar_clamp", 4),
        ("t0_scalar_clamp", 7),
        ("t1_list_map_add", 2),
        ("t1_list_map_add", 3),
        ("t1_list_map_add", 7),
        ("t2_clamp_then_sum", 1),
        ("t2_clamp_then_sum", 3),
        ("t2_map_then_count", 1),
        ("t2_shift_then_sum", 0),
    }
)
if set(TRAIN_KEYS) & set(TARGET_KEYS):
    raise RuntimeError("breadth donors and targets overlap")
if set(TARGET_KEYS) & KNOWN_NATIVE_TYPED_SPENT_TARGET_KEYS:
    raise RuntimeError("breadth target is already spent by a native typed probe")


def _marker_name(key: tuple[str, int]) -> str:
    return f"{Path(__file__).stem}.{key[0]}-{key[1]:03d}.spent.json"


SPEND_MARKER_PATHS = {key: PROBE_DIR / _marker_name(key) for key in TARGET_KEYS}

SOURCE_RECORDS: dict[tuple[str, int], dict[str, str | None]] = {
    LIST_ADD_DONOR: {
        "task_id": "simple-swe-v1/train/tier-1/t1_list_map_add/006",
        "entry_point": "solve_3234660fc6ed5836",
        "parameter": "values",
        "literal": "5",
        "lower": None,
        "upper": None,
        "task_sha256": "d84cb2c97461ae225283e486a3a213542fbd0f905cbce107fc1bc56fbafa1007",
        "hidden_test_sha256": "beef3d4baad1151774b7e65eac59fecd549309c3b727114674738f0854aaa9d2",
        "public_prompt_sha256": "7349ab488cd6638aae3e0cebdf0b75d7e93e82b5a7e177a4603727ab28af9a9c",
    },
    CLAMP_SUM_DONOR: {
        "task_id": "simple-swe-v1/train/tier-2/t2_clamp_then_sum/008",
        "entry_point": "solve_406c4e0bb2d37975",
        "parameter": "values",
        "literal": None,
        "lower": "-5",
        "upper": "5",
        "task_sha256": "f8334301d3354c523ee2daa84680831ee7e17da039c10867f1a0a0a1c2a7170d",
        "hidden_test_sha256": "d2a40bad64f365ff203e170384035f832aa304d0133d3c6250e877757cbd1a28",
        "public_prompt_sha256": "7d53d6d482a411b3f1f6971026e6c2a3146708d2df0b8c8486c0ae08ed717210",
    },
    TARGET_KEYS[0]: {
        "task_id": "simple-swe-v1/train/tier-1/t1_list_map_add/005",
        "entry_point": "solve_187d335b1d4ad45a",
        "parameter": "values",
        "literal": "1",
        "lower": None,
        "upper": None,
        "task_sha256": "6e7e11480f0edb7036062f0ef1e7aa60cf47968847a6dd0ab20ed90b748a647d",
        "hidden_test_sha256": "6beb7bafe5ce7eaa89bd4547ba5add88d71b21be0d9e6d1670d39d02a96ac45e",
        "public_prompt_sha256": "8f580169ee90871ddb161e74b0fee53de9999111f31214c484d0817b3901f29f",
    },
    TARGET_KEYS[1]: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/009",
        "entry_point": "solve_ac2b11ade24b1a08",
        "parameter": "n",
        "literal": "9",
        "lower": None,
        "upper": None,
        "task_sha256": "7035bee949ec9b32fb2dfa9f50c2fb23582f6ae29c0fb81e287fa05ef0ce6bae",
        "hidden_test_sha256": "fffe29d3b8ee44c11abb845c796752ede5a29024328981a5efb4e45f0667dbef",
        "public_prompt_sha256": "41ed6338aaa7e8a72cf521ebd851a5b69374acc283e66c1ffb5e31c446e7cc97",
    },
    TARGET_KEYS[2]: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_clamp/009",
        "entry_point": "solve_0fec62f3497dd728",
        "parameter": "n",
        "literal": None,
        "lower": "-12",
        "upper": "12",
        "task_sha256": "60cd26abbb3c4f2042ebcdfc4104e2a4e7b7bff1ab7a48f019320cdaf643afb9",
        "hidden_test_sha256": "e390574ac50fb176fe93bfada0ae4b19604739c068133a7d3dd4c7782e18eff7",
        "public_prompt_sha256": "47f609b89602e55336191ac5aadc59d704177ffdd8d610c9aa537aa79b1933d6",
    },
    TARGET_KEYS[3]: {
        "task_id": "simple-swe-v1/train/tier-2/t2_clamp_then_sum/007",
        "entry_point": "solve_0060e779b610d153",
        "parameter": "values",
        "literal": None,
        "lower": "-2",
        "upper": "2",
        "task_sha256": "945ca9b17161a1afe4cc230a1a946a1223c83ef7e1da9a834a4e5e183d94d7e8",
        "hidden_test_sha256": "d0af7db72c45d4dfa42fc8025c0025ee9ecded601b9e1b2b9344af364772fbc5",
        "public_prompt_sha256": "844cd003e5c617202b0aabc7b486c801b5033315cc3d66f0a856ec7c29421dcf",
    },
}

EXPECTED_PROMPT_TOKEN_COUNTS = {
    LIST_ADD_DONOR: 281,
    CLAMP_SUM_DONOR: 284,
    TARGET_KEYS[0]: 281,
    TARGET_KEYS[1]: 251,
    TARGET_KEYS[2]: 275,
    TARGET_KEYS[3]: 284,
}

EXPECTED_TARGET_TOKENS = {
    TARGET_KEYS[0]: (
        "map", "bind", "add", "variable-0", "integer", "+", "1", "end", "variable-0",
    ),
    TARGET_KEYS[1]: ("add", "variable-0", "integer", "+", "9", "end"),
    TARGET_KEYS[2]: (
        "if", "less-than", "variable-0", "integer", "-", "1", "2", "end",
        "integer", "-", "1", "2", "end", "if", "less-than", "integer", "+", "1", "2", "end",
        "variable-0", "integer", "+", "1", "2", "end", "variable-0",
    ),
    TARGET_KEYS[3]: (
        "sum", "map", "bind", "if", "less-than", "variable-0", "integer", "-", "2", "end",
        "integer", "-", "2", "end", "if", "less-than", "integer", "+", "2", "end",
        "variable-0", "integer", "+", "2", "end", "variable-0", "variable-0",
    ),
}

PLAN_COMPONENT_ARRIVALS = {LIST_ADD_DONOR: 5, CLAMP_SUM_DONOR: 6}
EXPECTED_TEACHER_CALL_COUNTS = {LIST_ADD_DONOR: 9, CLAMP_SUM_DONOR: 24}
EXPECTED_PHYSICAL_TEACHER_ARRIVALS = {LIST_ADD_DONOR: 18, CLAMP_SUM_DONOR: 48}
EXPECTED_EPISODE_LEARNING_ARRIVALS = {LIST_ADD_DONOR: 304, CLAMP_SUM_DONOR: 338}
EXPECTED_BASE_LEARNING_ARRIVALS = 642
EXPECTED_POINTER_CUE_LEARNING_ARRIVALS = 15
EXPECTED_TOTAL_LEARNING_ARRIVALS = 657
EXPECTED_BASE_NONINITIAL_LEARNING_ARRIVALS = 640
EXPECTED_NONINITIAL_LEARNING_ARRIVALS = 655
EXPECTED_BASE_SEGMENT_ROWS = 7_040
EXPECTED_ORDINARY_SEGMENT_ROWS_UPPER_BOUND = 7_205
EXPECTED_POINTER_CUE_SEGMENT_ROWS_UPPER_BOUND = 165
EXPECTED_POINTER_RECURRENCE_ROWS_UPPER_BOUND = 165
EXPECTED_POINTER_SEGMENT_ROWS_UPPER_BOUND = 330
EXPECTED_SEGMENT_CAPACITY = 7_370
PHYSICAL_LEARN_STEPS_PER_ELIGIBLE_TEACHER = 2
HEAD_TO_ROOT_TRANSFER_SEGMENT_ROWS = base.ACTIVE_COLUMNS
MAX_POINTER_ROWS_PER_CUE_CALL = 2 * base.ACTIVE_COLUMNS
ROLE_BEARING_CONSTRUCTORS = (*shared.ROLE_BEARING_CONSTRUCTORS, "integer")
HEAD_TO_ROOT_TRANSFER_POSITIONS = {LIST_ADD_DONOR: frozenset({2}), CLAMP_SUM_DONOR: frozenset({3})}

POINTER_OCCURRENCES_BY_DONOR = {
    LIST_ADD_DONOR: (
        _PointerOccurrence(0, "map", "expression-list-int-head", (), (), None),
        _PointerOccurrence(1, "bind", "expression-int-to-int-child-0", (0,), (0,), 0),
        _PointerOccurrence(2, "add", "expression-int-head", (0, 0), (), None),
        _PointerOccurrence(4, "integer", "expression-int-child-1", (0, 0, 1), (1,), 2),
    ),
    CLAMP_SUM_DONOR: single.POINTER_OCCURRENCES_BY_DONOR[single.CLAMP_SUM_KEY],
}

PINNED_SOURCE_SHA256 = {
    "experiments/r3_pilot/public_simple_swe_native_typed_sketch_probe.py": "95fb51bded52d246b5f91f0bacaa2b112d22b9a91d945fd89d12edc29f2a7457",
    "experiments/r3_pilot/public_simple_swe_native_typed_root_operation_transfer_probe.py": "59fe2acde729ac8f097d297b27ec118d0ddf7f7a5d4668f9cf169fd823f4c452",
    "experiments/r3_pilot/public_simple_swe_native_typed_clamp_sum_cross_donor_probe.py": "3cf58f9eb6dcb6e0ce089ea58fe1ca6ba778546b546f12babf1539bdebc6decb",
    "experiments/r3_pilot/public_simple_swe_native_typed_shift_sum_probe.py": "206309e97caa37ec5bdcf15936bf02b1e78684074f0f601f3997cb5e99941b61",
    "src/humanbrain/native_temporal_runtime.py": "77b41b014d35648adbe989fe3c31059856f60410758dddb3d9794e54a0ed227a",
    "src/humanbrain/_native_src/src/temporal_runtime.rs": "012d4f79505e3e0e381ebf9397db0c74d695120e7dfe8d511a338f8b896b9dc4",
    "src/humanbrain/_native_src/src/temporal_runtime/checkpoint.rs": "bdb1ab444a826e706b9c8284816a6e54847b0bb6c156723135324fffb377faba",
    "src/humanbrain/_native_src/src/temporal_runtime/typed_workspace.rs": "ebcdfb848b33b29e45709042216a44efbec535bbebe16513ee933722bf7e3874",
    "src/humanbrain/_native_src/src/temporal_runtime/typed_hole_state.rs": "e61a9e8101b5447ef245eda161e99a4e3631f1582b86f37213a36983d9774827",
}


@dataclass(frozen=True, slots=True)
class _TaskFields:
    entry_point: bytes
    parameter: bytes
    literal: bytes | None
    lower: bytes | None
    upper: bytes | None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", type=Path, required=True)
    return parser.parse_args()


def _assert_one_shot_paths_absent() -> None:
    paths = (("artifact", OUTPUT_PATH), *(('spend marker', path) for path in SPEND_MARKER_PATHS.values()))
    for kind, path in paths:
        if path.exists():
            raise RuntimeError(f"refusing spent breadth cohort; {kind} exists: {path}")


def _assert_sources_committed_and_unspent() -> None:
    _assert_one_shot_paths_absent()
    for source, expected in PINNED_SOURCE_SHA256.items():
        path = REPO_ROOT / source
        if _sha256_file(path) != expected:
            raise RuntimeError(f"frozen breadth dependency changed: {source}")
    for path in (Path(__file__).resolve(), *(REPO_ROOT / source for source in PINNED_SOURCE_SHA256)):
        source = path.relative_to(REPO_ROOT).as_posix()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", source],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout:
            raise RuntimeError(f"breadth probe and pinned dependencies must be committed: {source}")
    if SLOT_COUNT != 4:
        raise RuntimeError("breadth probe requires four typed transport slots")


def _create_spend_marker(
    key: tuple[str, int], *, attempt_index: int, checkpoint_sha256: str
) -> dict[str, object]:
    if key not in SPEND_MARKER_PATHS:
        raise RuntimeError(f"unknown breadth target marker key {key!r}")
    if OUTPUT_PATH.exists():
        raise RuntimeError("breadth artifact appeared before target spend")
    path = SPEND_MARKER_PATHS[key]
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-native-typed-breadth",
        "state": "TARGET_ATTEMPT_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "attempt_index": attempt_index,
        "target_key": list(key),
        "target_task_id": SOURCE_RECORDS[key]["task_id"],
        "trained_checkpoint_sha256": checkpoint_sha256,
        "canonical_artifact_path": _display_path(OUTPUT_PATH),
        "probe_source_sha256": _sha256_file(Path(__file__)),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as marker:
            marker.write(serialized)
            marker.flush()
            os.fsync(marker.fileno())
    except FileExistsError as error:
        raise RuntimeError(f"refusing spent target; spend marker exists: {path}") from error
    if OUTPUT_PATH.exists():
        raise RuntimeError("breadth artifact appeared during target spend; marker retained")
    return {
        "path": _display_path(path),
        "sha256": _sha256_file(path),
        "state": payload["state"],
        "exclusive_create": True,
        "flushed_and_fsynced_before_generate": True,
    }


def _select_tasks() -> tuple[object, tuple[GeneratedTask, ...], tuple[GeneratedTask, ...]]:
    protocol = load_protocol(PROTOCOL_PATH)
    if protocol.protocol_id != EXPECTED_PROTOCOL_ID or protocol.sha256 != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("public Simple-SWE protocol identity changed")
    visible_train = generate_visible_partition(protocol, "train")
    selected: dict[tuple[str, int], GeneratedTask] = {}
    for key in (*TRAIN_KEYS, *TARGET_KEYS):
        try:
            task = next(
                item for item in visible_train
                if (item.public.family, item.public.ordinal) == key
            )
        except StopIteration as error:
            raise RuntimeError(f"visible TRAIN task disappeared for {key!r}") from error
        _task_fields(task.public)
        if len(task.private.regression_cases) != 4 or len(task.private.property_cases) != 32:
            raise RuntimeError(f"visible sandbox case counts changed for {key!r}")
        selected[key] = task
    return (
        protocol,
        tuple(selected[key] for key in TRAIN_KEYS),
        tuple(selected[key] for key in TARGET_KEYS),
    )


def _task_fields(public_task: object) -> _TaskFields:
    if type(public_task).__name__ != "PublicTask":
        raise TypeError("breadth parsing requires the exact public task view")
    key = (public_task.family, public_task.ordinal)
    if key not in SOURCE_RECORDS:
        raise RuntimeError(f"unsupported breadth task key {key!r}")
    expected = SOURCE_RECORDS[key]
    signature = public_task.signature
    expected_parameter_type = (
        ProgramType.List(ProgramType.Int())
        if key[0] in {"t1_list_map_add", "t2_clamp_then_sum"}
        else ProgramType.Int()
    )
    expected_result_type = (
        ProgramType.List(ProgramType.Int())
        if key[0] == "t1_list_map_add"
        else ProgramType.Int()
    )
    if (
        len(signature.parameters) != 1
        or signature.parameters[0].type != expected_parameter_type
        or signature.result != expected_result_type
    ):
        raise RuntimeError(f"visible signature changed for {key!r}")
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
        for name in (
            "task_id", "entry_point", "parameter", "task_sha256",
            "hidden_test_sha256", "public_prompt_sha256",
        )
    }
    if public_task.partition != "train" or observed != expected_provenance:
        raise RuntimeError(f"visible TRAIN provenance changed for {key!r}: {observed!r}")

    def optional_bytes(name: str) -> bytes | None:
        value = expected[name]
        return None if value is None else str(value).encode("ascii")

    fields = _TaskFields(
        str(expected["entry_point"]).encode("ascii"),
        str(expected["parameter"]).encode("ascii"),
        optional_bytes("literal"),
        optional_bytes("lower"),
        optional_bytes("upper"),
    )
    prompt = public_task.prompt.encode("ascii")
    signature_bytes = b"def " + fields.entry_point + b"(" + fields.parameter + b"):"
    if signature_bytes not in prompt:
        raise RuntimeError(f"visible callable wording changed for {key!r}")
    if key[0] == "t1_list_map_add":
        phrase = b"with " + fields.literal + b" added to every integer"
    elif key[0] == "t0_scalar_offset":
        phrase = b"fixed offset " + fields.literal
    elif key[0] == "t0_scalar_clamp":
        phrase = b"inclusive range " + fields.lower + b" through " + fields.upper
    else:
        phrase = b"Limit every integer to " + fields.lower + b" through " + fields.upper
    if phrase not in prompt:
        raise RuntimeError(f"visible task wording changed for {key!r}")
    return fields


def _canonical_literal_primitives(literal: bytes) -> tuple[str, ...]:
    text = literal.decode("ascii")
    signed = text if text.startswith("-") else f"+{text}"
    if signed == "-0" or not signed[1:].isdigit():
        raise RuntimeError(f"non-canonical visible literal {text!r}")
    return tuple(signed)


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
    if fields.literal is not None:
        anchor = b"with " if key[0] == "t1_list_map_add" else b"fixed offset "
        start = prompt.index(anchor, parameter_start) + len(anchor)
        spans.append(
            (start, start + len(fields.literal), 2,
             tuple(PRIMITIVE_TOKEN_IDS[token] for token in _canonical_literal_primitives(fields.literal)))
        )
    else:
        if fields.lower is None or fields.upper is None:
            raise RuntimeError("bounded task lost one literal")
        anchor = b"inclusive range " if key[0] == "t0_scalar_clamp" else b"Limit every integer to "
        lower_start = prompt.index(anchor, parameter_start) + len(anchor)
        upper_start = lower_start + len(fields.lower) + len(b" through ")
        spans.extend(
            (
                (lower_start, lower_start + len(fields.lower), 2,
                 tuple(PRIMITIVE_TOKEN_IDS[token] for token in _canonical_literal_primitives(fields.lower))),
                (upper_start, upper_start + len(fields.upper), 3,
                 tuple(PRIMITIVE_TOKEN_IDS[token] for token in _canonical_literal_primitives(fields.upper))),
            )
        )
    return prompt, tuple(spans)


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
    if atomic._detokenize_prompt(result) != prompt:
        raise RuntimeError("breadth prompt tokenizer is not lossless")
    key = (public_task.family, public_task.ordinal)
    if len(result) != EXPECTED_PROMPT_TOKEN_COUNTS[key]:
        raise RuntimeError(f"public prompt token count changed for {key!r}")
    return result


def _target_runtime_input(public_task: object) -> _TargetRuntimeInput:
    key = (public_task.family, public_task.ordinal)
    if key not in TARGET_KEYS:
        raise RuntimeError("target runtime input received a donor")
    signature = public_task.signature
    return _TargetRuntimeInput(
        prompt_tokens=_tokenized_public_prompt(public_task),
        parameter_type=signature.parameters[0].type,
        result_type=signature.result,
    )


def _encode_teacher(primitives: tuple[str | tuple[str, int], ...]) -> tuple[int, ...]:
    return (
        *(copy_slot_token_id(item[1]) if isinstance(item, tuple) else PRIMITIVE_TOKEN_IDS[item]
          for item in primitives),
        base.EOS_TOKEN_ID,
    )


def _internal_teacher_tokens(key: tuple[str, int]) -> tuple[int, ...]:
    if key == LIST_ADD_DONOR:
        primitives: tuple[str | tuple[str, int], ...] = (
            "map", "bind", "add", "variable-0", "integer", ("copy", 2), "end", "variable-0",
        )
    elif key == CLAMP_SUM_DONOR:
        primitives = (
            "sum", "map", "bind", "if", "less-than", "variable-0", "integer", ("copy", 2), "end",
            "integer", ("copy", 2), "end", "if", "less-than", "integer", ("copy", 3), "end",
            "variable-0", "integer", ("copy", 3), "end", "variable-0", "variable-0",
        )
    else:
        raise RuntimeError(f"unsupported donor key {key!r}")
    return _encode_teacher(primitives)


def _expected_program_tokens(public_task: object) -> tuple[str, ...]:
    fields = _task_fields(public_task)
    key = (public_task.family, public_task.ordinal)
    if key[0] == "t1_list_map_add":
        assert fields.literal is not None
        result = (
            "map", "bind", "add", "variable-0", "integer",
            *_canonical_literal_primitives(fields.literal), "end", "variable-0",
        )
    elif key[0] == "t0_scalar_offset":
        assert fields.literal is not None
        result = (
            "add", "variable-0", "integer", *_canonical_literal_primitives(fields.literal), "end",
        )
    else:
        assert fields.lower is not None and fields.upper is not None
        lower = _canonical_literal_primitives(fields.lower)
        upper = _canonical_literal_primitives(fields.upper)
        clamp = (
            "if", "less-than", "variable-0", "integer", *lower, "end",
            "integer", *lower, "end", "if", "less-than", "integer", *upper, "end",
            "variable-0", "integer", *upper, "end", "variable-0",
        )
        result = ("sum", "map", "bind", *clamp, "variable-0") if key[0] == "t2_clamp_then_sum" else clamp
    if key in EXPECTED_TARGET_TOKENS and result != EXPECTED_TARGET_TOKENS[key]:
        raise RuntimeError(f"frozen target token stream changed for {key!r}")
    return result


def _expected_compiled_source(public_task: object) -> str:
    fields = _task_fields(public_task)
    key = (public_task.family, public_task.ordinal)
    entry = fields.entry_point.decode("ascii")
    parameter = fields.parameter.decode("ascii")
    if key[0] == "t1_list_map_add":
        assert fields.literal is not None
        body = f"[item_0 + {fields.literal.decode('ascii')} for item_0 in {parameter}]"
    elif key[0] == "t0_scalar_offset":
        assert fields.literal is not None
        body = f"{parameter} + {fields.literal.decode('ascii')}"
    else:
        assert fields.lower is not None and fields.upper is not None
        lower, upper = fields.lower.decode("ascii"), fields.upper.decode("ascii")
        clamp = f"{lower} if {parameter} < {lower} else {upper} if {upper} < {parameter} else {parameter}"
        if key[0] == "t2_clamp_then_sum":
            clamp = f"{lower} if item_0 < {lower} else {upper} if {upper} < item_0 else item_0"
            body = f"sum([{clamp} for item_0 in {parameter}])"
        else:
            body = clamp
    return f"def {entry}({parameter}):\n    return {body}\n"


def _expected_copy_schedule(public_task: object) -> tuple[tuple[int, tuple[str, ...]], ...]:
    fields = _task_fields(public_task)
    if fields.literal is not None:
        return ((2, _canonical_literal_primitives(fields.literal)),)
    assert fields.lower is not None and fields.upper is not None
    lower = _canonical_literal_primitives(fields.lower)
    upper = _canonical_literal_primitives(fields.upper)
    return ((2, lower), (2, lower), (3, upper), (3, upper))


def _public_task_record(public_task: object) -> dict[str, object]:
    fields = _task_fields(public_task)
    key = (public_task.family, public_task.ordinal)
    prompt_tokens = _tokenized_public_prompt(public_task)
    return {
        "task_id": public_task.task_id,
        "partition": public_task.partition,
        "family": public_task.family,
        "ordinal": public_task.ordinal,
        "entry_point": fields.entry_point.decode("ascii"),
        "parameter": fields.parameter.decode("ascii"),
        "literal": None if fields.literal is None else fields.literal.decode("ascii"),
        "lower": None if fields.lower is None else fields.lower.decode("ascii"),
        "upper": None if fields.upper is None else fields.upper.decode("ascii"),
        "task_sha256": public_task.task_sha256,
        "hidden_test_sha256": public_task.hidden_test_sha256,
        "public_prompt_sha256": hashlib.sha256(public_task.prompt.encode("utf-8")).hexdigest(),
        "public_prompt_lossless_after_tokenization": atomic._detokenize_prompt(prompt_tokens) == public_task.prompt.encode("ascii"),
        "prompt_token_count": len(prompt_tokens),
    }


def _infer_pointer_rows_from_teacher_delta(
    *, learning_step_delta: int, segment_row_delta: int, returned_exact_rows: int
) -> tuple[int, int]:
    if learning_step_delta == 4:
        learning_step_delta -= 1
        segment_row_delta -= HEAD_TO_ROOT_TRANSFER_SEGMENT_ROWS
    return _BASE_INFER_POINTER_ROWS(
        learning_step_delta=learning_step_delta,
        segment_row_delta=segment_row_delta,
        returned_exact_rows=returned_exact_rows,
    )


def _learning_arrival_accounting(
    train_prompts: dict[tuple[str, int], tuple[int, ...]],
    teachers: dict[tuple[str, int], tuple[int, ...]],
) -> dict[str, object]:
    if tuple(train_prompts) != TRAIN_KEYS or tuple(teachers) != TRAIN_KEYS:
        raise RuntimeError("breadth capacity accounting requires the frozen donor order")
    episodes: dict[str, object] = {}
    base_arrivals = 0
    pointer_arrivals = 0
    for key in TRAIN_KEYS:
        prompt_count = len(train_prompts[key])
        teacher_count = len(teachers[key])
        if prompt_count != EXPECTED_PROMPT_TOKEN_COUNTS[key] or teacher_count != EXPECTED_TEACHER_CALL_COUNTS[key]:
            raise RuntimeError(f"breadth donor stream changed for {key!r}")
        eligible = teacher_count - 1
        head_transfer_count = len(HEAD_TO_ROOT_TRANSFER_POSITIONS[key])
        physical_teacher = eligible * PHYSICAL_LEARN_STEPS_PER_ELIGIBLE_TEACHER + 1 + head_transfer_count
        episode_arrivals = prompt_count + PLAN_COMPONENT_ARRIVALS[key] + physical_teacher
        pointer_count = len(POINTER_OCCURRENCES_BY_DONOR[key])
        if physical_teacher != EXPECTED_PHYSICAL_TEACHER_ARRIVALS[key] or episode_arrivals != EXPECTED_EPISODE_LEARNING_ARRIVALS[key]:
            raise RuntimeError(f"breadth donor physical accounting changed for {key!r}")
        episodes[f"{key[0]}/{key[1]:03d}"] = {
            "prompt_arrivals": prompt_count,
            "plan_component_arrivals": PLAN_COMPONENT_ARRIVALS[key],
            "teacher_api_calls": teacher_count,
            "eligible_teacher_api_calls": eligible,
            "head_to_root_transfer_arrivals": head_transfer_count,
            "physical_teacher_arrivals": physical_teacher,
            "base_learning_arrivals": episode_arrivals,
            "constructor_pointer_cue_arrivals": pointer_count,
            "total_learning_arrivals": episode_arrivals + pointer_count,
        }
        base_arrivals += episode_arrivals
        pointer_arrivals += pointer_count
    total = base_arrivals + pointer_arrivals
    noninitial = total - len(TRAIN_KEYS)
    ordinary_rows = noninitial * base.ACTIVE_COLUMNS
    recurrence_rows = pointer_arrivals * base.ACTIVE_COLUMNS
    segment_capacity = ordinary_rows + recurrence_rows
    observed = (
        base_arrivals, pointer_arrivals, total, base_arrivals - len(TRAIN_KEYS),
        noninitial, (base_arrivals - len(TRAIN_KEYS)) * base.ACTIVE_COLUMNS,
        ordinary_rows, recurrence_rows, segment_capacity,
    )
    expected = (
        EXPECTED_BASE_LEARNING_ARRIVALS, EXPECTED_POINTER_CUE_LEARNING_ARRIVALS,
        EXPECTED_TOTAL_LEARNING_ARRIVALS, EXPECTED_BASE_NONINITIAL_LEARNING_ARRIVALS,
        EXPECTED_NONINITIAL_LEARNING_ARRIVALS, EXPECTED_BASE_SEGMENT_ROWS,
        EXPECTED_ORDINARY_SEGMENT_ROWS_UPPER_BOUND,
        EXPECTED_POINTER_RECURRENCE_ROWS_UPPER_BOUND, EXPECTED_SEGMENT_CAPACITY,
    )
    if observed != expected:
        raise RuntimeError(f"breadth aggregate capacity accounting changed: {observed!r}")
    base_population = COLUMN_COUNT * base.BASE_CELLS_PER_COLUMN
    return {
        "episodes": episodes,
        "base_learning_arrivals": base_arrivals,
        "constructor_pointer_cue_arrivals": pointer_arrivals,
        "total_learning_arrivals": total,
        "resets": len(TRAIN_KEYS),
        "noninitial_learning_arrivals": noninitial,
        "base_temporal_segment_rows_upper_bound": EXPECTED_BASE_SEGMENT_ROWS,
        "ordinary_segment_rows_upper_bound_including_pointer_cues": ordinary_rows,
        "constructor_pointer": {
            "exact_role_opening_call_count": pointer_arrivals,
            "cue_rows_upper_bound": EXPECTED_POINTER_CUE_SEGMENT_ROWS_UPPER_BOUND,
            "recurrence_rows_upper_bound": recurrence_rows,
            "total_rows_upper_bound": EXPECTED_POINTER_SEGMENT_ROWS_UPPER_BOUND,
            "capacity_assumes_every_call_recruits": True,
        },
        "segment_capacity": segment_capacity,
        "base_pyramidal_cells": base_population,
        "reserve_pyramidal_cells": segment_capacity,
        "max_pyramidal_cells": base_population + segment_capacity,
    }


@contextmanager
def _configured_shared() -> object:
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
        "_learning_arrival_accounting": _learning_arrival_accounting,
        "_infer_pointer_rows_from_teacher_delta": _infer_pointer_rows_from_teacher_delta,
        "_expected_program_tokens": _expected_program_tokens,
        "_expected_compiled_source": _expected_compiled_source,
        "_public_task_record": _public_task_record,
        "_expected_copy_schedule": _expected_copy_schedule,
        "compile_typed_program": compile_typed_program,
        "execute_completion": execute_completion,
    }
    previous = {name: getattr(shared, name) for name in overrides}
    previous_column_count = base.COLUMN_COUNT
    try:
        base.COLUMN_COUNT = COLUMN_COUNT
        for name, value in overrides.items():
            setattr(shared, name, value)
        yield
    finally:
        base.COLUMN_COUNT = previous_column_count
        for name, value in previous.items():
            setattr(shared, name, value)


@contextmanager
def _configured_atomic_scoring() -> object:
    overrides = {
        "_expected_program_tokens": _expected_program_tokens,
        "_expected_compiled_source": _expected_compiled_source,
        "_public_task_record": _public_task_record,
        "_expected_copy_schedule": _expected_copy_schedule,
        "compile_typed_program": compile_typed_program,
        "execute_completion": execute_completion,
    }
    previous = {name: getattr(atomic, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(atomic, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(atomic, name, value)


def _definition(
    train_tasks: tuple[GeneratedTask, ...], target_inputs: tuple[_TargetRuntimeInput, ...]
) -> tuple[object, dict[str, object], tuple[int, ...]]:
    target_prompt_union = tuple(sorted(set(chain.from_iterable(item.prompt_tokens for item in target_inputs))))
    union_input = _TargetRuntimeInput(
        prompt_tokens=target_prompt_union,
        parameter_type=ProgramType.Int(),
        result_type=ProgramType.Int(),
    )
    with _configured_shared():
        definition, accounting = shared._definition(train_tasks, union_input)
    defined_token_ids = {token_id for token_id, _columns in definition.token_codes}
    for target_input in target_inputs:
        if not set(target_input.prompt_tokens) <= defined_token_ids:
            raise RuntimeError("definition omitted a target prompt token")
    return definition, accounting, target_prompt_union


def _generation_failure(error: NativeTemporalRuntimeError, expected_tokens: tuple[str, ...]) -> dict[str, object]:
    position = int(error.completed_prefix)
    return {
        "kind": "model",
        "gate": "GENERATION_STOP",
        "detail": {
            "status": error.status.name,
            "message": error.message,
            "completed_surface_prefix": position,
            "retained_completed_generations": len(error.completed_generations),
            "expected_surface": expected_tokens[position] if position < len(expected_tokens) else "<EOS>",
            "required": error.required,
            "observed": error.observed,
            "location": error.location,
            "poison_code": error.poison_code,
            "stage_code": error.stage_code,
        },
    }


def _completed_generations(error: NativeTemporalRuntimeError) -> tuple[object, ...]:
    generations = tuple(error.completed_generations)
    if len(generations) != error.completed_prefix:
        raise RuntimeError(
            "Python typed transport did not retain the complete committed generation prefix"
        )
    return generations


def _classify_structural_record(
    record: dict[str, object], expected_tokens: tuple[str, ...]
) -> dict[str, bool]:
    generations = record["generations"]
    observed = tuple(int(item["surface_token_id"]) for item in generations)
    expected_ast = tuple(PRIMITIVE_TOKEN_IDS[token] for token in expected_tokens)
    surface_exact = observed in (expected_ast, (*expected_ast, base.EOS_TOKEN_ID))
    copy_control_exact = bool(record["copy_trace_exact"])
    fitted_cell_races_exact = bool(record["fitted_cell_races_exact"])
    structural_exact = surface_exact and copy_control_exact and fitted_cell_races_exact
    ordinary_eos = bool(record["terminated_by_ordinary_eos"])
    end_to_end_pass = record["first_failure"] is None
    return {
        "surface_ids_exact_through_complete_ast": surface_exact,
        "physical_copy_control_exact": copy_control_exact,
        "fitted_cell_races_exact": fitted_cell_races_exact,
        "structural_exact": structural_exact,
        "ordinary_eos": ordinary_eos,
        "end_to_end_pass": end_to_end_pass,
        "ordinary_eos_full_pass": structural_exact and ordinary_eos and end_to_end_pass,
        "pre_completion_structural_failure": not structural_exact,
    }


def _stop_decision(records: list[dict[str, object]]) -> str:
    exact = sum(bool(record["classification"]["structural_exact"]) for record in records)
    failures = sum(bool(record["classification"]["pre_completion_structural_failure"]) for record in records)
    eos_passes = sum(bool(record["classification"]["ordinary_eos_full_pass"]) for record in records)
    if exact >= 3 and eos_passes >= 1:
        return "LICENSE_CLOSED_LOOP_EDIT"
    if exact >= 3:
        return "STOP_THREE_STRUCTURAL_EXACT_WITHOUT_EOS_LICENSE"
    if failures >= 2:
        return "STOP_TWO_STRUCTURAL_FAILURES"
    if len(records) >= len(TARGET_KEYS):
        return "STOP_ABSOLUTE_FOUR_WITHOUT_LICENSE"
    return "CONTINUE"


def _teacher_label(token_id: int) -> str:
    for slot in (2, 3):
        if token_id == copy_slot_token_id(slot):
            return f"COPY_SLOT_{slot}"
    if token_id == base.EOS_TOKEN_ID:
        return "<EOS>"
    return PRIMITIVE_BY_TOKEN_ID[token_id]


def main() -> int:
    args = _parse_args()
    _assert_sources_committed_and_unspent()
    started_ns = time.perf_counter_ns()
    protocol, train_tasks, target_tasks = _select_tasks()
    target_inputs = tuple(_target_runtime_input(task.public) for task in target_tasks)
    definition, capacity_accounting, target_prompt_union = _definition(train_tasks, target_inputs)
    teachers = {key: _internal_teacher_tokens(key) for key in TRAIN_KEYS}

    executable = resolve_python_executable(args.python_executable)
    runtime_identity = python_runtime_identity(executable, protocol.raw.get("language"))
    if runtime_identity["python_version"] != "3.11.15":
        raise RuntimeError("breadth probe requires explicit repo Python 3.11.15")
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = _sha256_file(worker_path)
    worker_source = load_worker_snapshot(worker_sha256)

    donor_compiler_checks = []
    for task in train_tasks:
        tokens = _expected_program_tokens(task.public)
        source = compile_typed_program(Program(tokens, task.public.signature)).source
        if source != _expected_compiled_source(task.public):
            raise RuntimeError(f"typed compiler cannot express donor {task.public.task_id}")
        donor_compiler_checks.append({"task_id": task.public.task_id, "exact": True})

    trained: list[dict[str, object]] = []
    target_records: list[dict[str, object]] = []
    with base._public_development_authority():
        with _configured_shared():
            with NativeTemporalCircuit.fresh(definition) as circuit:
                for key, task in zip(TRAIN_KEYS, train_tasks, strict=True):
                    signature = task.public.signature
                    episode, observations = shared._train_task_with_pointer_accounting(
                        circuit,
                        key=key,
                        prompt_tokens=_tokenized_public_prompt(task.public),
                        parameter_type=signature.parameters[0].type,
                        result_type=signature.result,
                        teacher_tokens=teachers[key],
                    )
                    identity = circuit.checkpoint_identity()
                    trained.append(
                        {
                            "task": _public_task_record(task.public),
                            "teacher": [_teacher_label(token) for token in teachers[key]],
                            "prompt_steps": len(episode.prompt_steps),
                            "workspace_steps": len(episode.workspace_steps),
                            "learning_steps_after_task": identity.learning_step,
                            "live_segments_after_task": identity.segment_count,
                            "pointer_observations": [
                                {
                                    "teacher_position": item.teacher_position,
                                    "surface": item.surface,
                                    "learning_step_delta": item.learning_step_delta,
                                    "segment_row_delta": item.segment_row_delta,
                                    "actual_pointer_rows": item.actual_pointer_rows,
                                }
                                for item in observations
                            ],
                        }
                    )
                trained_identity = circuit.checkpoint_identity()
                if trained_identity.learning_step != EXPECTED_TOTAL_LEARNING_ARRIVALS:
                    raise RuntimeError("frozen breadth Learn accounting changed")
                checkpoint = circuit.checkpoint()
        checkpoint_sha256 = hashlib.sha256(checkpoint).hexdigest()

        for attempt_index, (key, task, target_input) in enumerate(
            zip(TARGET_KEYS, target_tasks, target_inputs, strict=True), start=1
        ):
            native_error: NativeTemporalRuntimeError | None = None
            with NativeTemporalCircuit.from_checkpoint(definition, checkpoint) as restored:
                if restored.checkpoint_identity() != trained_identity:
                    raise RuntimeError("restored target circuit differs from the frozen trained checkpoint")
                target_episode = _TypedWorkspaceApi.begin_target(restored, target_input)
                try:
                    spend_marker = _create_spend_marker(
                        key,
                        attempt_index=attempt_index,
                        checkpoint_sha256=checkpoint_sha256,
                    )
                    raw_generations = restored.generate_typed()
                except NativeTemporalRuntimeError as error:
                    raw_generations = _completed_generations(error)
                    native_error = error

            native_failure = (
                None
                if native_error is None
                else _generation_failure(native_error, EXPECTED_TARGET_TOKENS[key])
            )
            with _configured_atomic_scoring():
                record, first_failure = atomic._score_target(
                    episode=target_episode,
                    raw_generations=raw_generations,
                    native_failure=native_failure,
                    task=task,
                    protocol=protocol,
                    python_executable=executable,
                    runtime_identity=runtime_identity,
                    worker_source=worker_source,
                    worker_sha256=worker_sha256,
                )
            record["classification"] = _classify_structural_record(
                record, EXPECTED_TARGET_TOKENS[key]
            )
            record["target_spend_marker"] = {
                **spend_marker,
                "exists_after_attempt": True,
                "retained_on_model_failure": True,
            }
            record["first_failure"] = first_failure
            target_records.append(record)
            if _stop_decision(target_records) != "CONTINUE":
                break
        native_identity = loaded_library_identity()

    stop_decision = _stop_decision(target_records)
    licensed = stop_decision == "LICENSE_CLOSED_LOOP_EDIT"

    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-native-typed-breadth",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": stop_decision,
        "closed_loop_edit_licensed": licensed,
        "claim_boundary": (
            "Capped reuse check on visible TRAIN only. Targets are native-typed-fresh, not globally untouched: "
            "the historical prompt-only baseline covered all 90 TRAIN tasks. This is not DEV, eval, held-out, M0, or C1 evidence."
        ),
        "authority": {
            "source_partition_generated": "train",
            "dev_or_held_out_opened": False,
            "train_keys": [list(key) for key in TRAIN_KEYS],
            "ordered_teacher_free_target_keys": [list(key) for key in TARGET_KEYS],
            "attempted_target_keys": [
                [record["task"]["family"], record["task"]["ordinal"]]
                for record in target_records
            ],
            "model_retries": 0,
            "harness_retries": 0,
            "checkpoint_trained_once": True,
            "independent_restore_per_attempt": True,
            "target_expected_solutions_used_as_runtime_input": False,
            "target_private_cases_used_as_runtime_input": False,
            "atomic_no_argument_generation_call_count": len(target_records),
            "stop_rule": {
                "license_requires_structural_exact_targets": 3,
                "license_requires_ordinary_eos_full_passes": 1,
                "stop_after_pre_completion_structural_failures": 2,
                "absolute_attempt_cap": 4,
            },
        },
        "protocol": {"protocol_id": protocol.protocol_id, "protocol_sha256": protocol.sha256},
        "physical_definition": {
            "circuit_identity_sha256": definition.circuit_identity_sha256(),
            "segment_capacity": definition.config.segment_capacity,
            "capacity_accounting": capacity_accounting,
            "target_prompt_codebook_union_size": len(target_prompt_union),
            "all_target_prompts_in_codebook": True,
        },
        "train": {
            "tasks": trained,
            "donor_compiler_checks": donor_compiler_checks,
            "trained_checkpoint_sha256": checkpoint_sha256,
            "trained_checkpoint_bytes": len(checkpoint),
            "learning_steps_after_train": trained_identity.learning_step,
            "live_segments_after_train": trained_identity.segment_count,
        },
        "visible_train_targets_teacher_free": target_records,
        "runtime": {
            "elapsed_wall_ns": time.perf_counter_ns() - started_ns,
            "python": sys.version,
            "platform": platform.platform(),
            "native_source_digest": native_identity.source_digest,
            "native_binary_sha256": native_identity.binary_sha256,
            "probe_source_sha256": _sha256_file(Path(__file__)),
            "pinned_source_sha256": {
                source: _sha256_file(REPO_ROOT / source) for source in PINNED_SOURCE_SHA256
            },
        },
    }
    with OUTPUT_PATH.open("x", encoding="utf-8", newline="\n") as output_file:
        output_file.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"outcome": stop_decision, "output": str(OUTPUT_PATH)}, sort_keys=True))
    return 0 if licensed else 2


if __name__ == "__main__":
    raise SystemExit(main())
