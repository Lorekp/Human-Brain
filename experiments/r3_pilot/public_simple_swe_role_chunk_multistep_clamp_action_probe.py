"""One-shot public-TRAIN multi-step clamp action-routing probe.

A donor-fitted RoleChunk encoder normalizes scalar-clamp /002, /007, and
/006 to one 88-event concept stream.  A fresh native circuit learns two
fixed visible-checker trajectories: identity -> lower clamp -> full clamp,
and lower clamp -> full clamp.  Two exact target checkpoint clones then
start from different external sources while receiving the same /006 prompt;
their genuine coarse checker observations must drive seven physical action
elections through the corresponding multi-step repair paths.

The /007 donor was previously a native typed-clamp target, while /006 was
disclosed to the historical prompt-only baseline.  This experiment therefore
makes no freshness, semantic-error, synthesis, or transfer claim.  Importing
or compiling this module does not spend the target.
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
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PROBE_DIR = Path(__file__).resolve().parent
for import_root in (SRC_ROOT, REPO_ROOT, PROBE_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import public_simple_swe_role_chunk_feedback_action_probe as feedback_probe
import public_simple_swe_role_chunk_layer0_action_probe as action_probe
import public_simple_swe_role_chunk_route_probe as route_probe
from humanbrain.cortex import seeded_threshold_offsets_mv
from humanbrain.native import loaded_library_identity
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitConfig,
    NativeTemporalCircuitDefinition,
    TemporalContextPolicy,
    TemporalPlasticityPolicy,
    TemporalStepMode,
)
from humanbrain.program import Layer0ProgramCompiler, Layer0ProgramGrammar, Layer0ProgramSignature
from humanbrain.role_chunk_episode import (
    RoleChunkEpisode,
    RoleChunkEpisodeEncoder,
    RolePayloadKind,
)
from humanbrain.sdr_union import token_sdr_indices
from humanbrain.simple_swe.sandbox import (
    load_worker_snapshot,
    python_runtime_identity,
    resolve_python_executable,
)
from humanbrain.simple_swe.tasks import GeneratedTask, PublicTask, generate_visible_partition


OUTPUT_PATH = Path(__file__).with_suffix(".json")
SPEND_MARKER_PATH = Path(__file__).with_name(
    f"{Path(__file__).stem}.t0_scalar_clamp-006.spent.json"
)
SIGNED_SOURCE_PATH = Path(__file__).with_name(
    "public_simple_swe_role_chunk_signed_residual_action_probe.py"
)
SIGNED_ARTIFACT_PATH = Path(__file__).with_name(
    "public_simple_swe_role_chunk_signed_residual_action_probe.json"
)
PRIOR_CLAMP_SOURCE_PATH = Path(__file__).with_name(
    "public_simple_swe_native_typed_clamp_probe.py"
)
PRIOR_CLAMP_ARTIFACT_PATH = Path(__file__).with_name(
    "public_simple_swe_native_typed_clamp_probe.json"
)
HISTORICAL_BASELINE_PATH = REPO_ROOT / "artifacts" / "simple_swe_public_baseline_v1.json"

FAMILY = "t0_scalar_clamp"
IDENTITY_DONOR_ORDINAL = 2
LOWER_DONOR_ORDINAL = 7
TARGET_ORDINAL = 6
TASK_RECORDS = {
    2: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_clamp/002",
        "task_sha256": "eda0e95f4496763b7d95044d03867c9f8aff3ef1dcf0c4e894f81e882dbb3a11",
        "hidden_test_sha256": "416aeb4ac865d03c160baec0db1ed366552c1f67a64d0569cf2decbff1a27fa8",
        "prompt_sha256": "c33674a7f9d4453d544272a7559c491f381d66e63df368a3827cd34b716d4324",
    },
    7: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_clamp/007",
        "task_sha256": "4d3a2c4da9c5f0adf9b4542dbbec11b2bb28940a4bd08ff6af2c1ed3502fa9e3",
        "hidden_test_sha256": "4ece18626b16fa0e4bc1f051ab1e926925c0ac6de70d9c8a298dc0c6d882493a",
        "prompt_sha256": "003e96a2bc2caf5ec84ef7ad4c49a1c93bb046ea7e8157aacc7e25aaaffd879a",
    },
    6: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_clamp/006",
        "task_sha256": "4a9736fd2a13a240afb08bd69ebe28904f76fdc6f7a73a10246fa92e5a3319ba",
        "hidden_test_sha256": "421dcf63d757dbaa5ca3c1ce35e160a52ab5fe69c97ef877283f5786a0d5fd99",
        "prompt_sha256": "42e0663491a773d96b06c82eac324222ab6f3446c6f9bc07368dfa1f1b2467aa",
    },
}
TARGET_TASK_ID = str(TASK_RECORDS[TARGET_ORDINAL]["task_id"])
LOWER_BOUND = -5
UPPER_BOUND = 5
STRUCTURAL_SHA256 = "d5ff4e912248df14be67fb9d4ae4935f382fc69f387ab6ac1c312fc3dd87775b"
CONCEPT_STREAM_SHA256 = "487b0f798ef2fcda52ffd0f03fd516c89f44fa3338074ec7d86c800e59b05962"
CONCEPT_STREAM_EVENTS = 88

TEST_TOKEN_ID = 1540
DONE_TOKEN_ID = 1541
FIX_LOWER_TOKEN_ID = 1544
FIX_FULL_TOKEN_ID = 1545
OBS_PASS_TOKEN_ID = 1601
OBS_LOW_TOKEN_ID = 1604
OBS_HIGH_TOKEN_ID = 1605
ACTION_NAME_BY_TOKEN = {
    TEST_TOKEN_ID: "TEST",
    DONE_TOKEN_ID: "DONE",
    FIX_LOWER_TOKEN_ID: "FIX_LOWER",
    FIX_FULL_TOKEN_ID: "FIX_FULL",
}
IDENTITY_LAYER0_TOKENS = ("ARG0", "END")
LOWER_LAYER0_TOKENS = (
    "MAX",
    "ARG0",
    "SUB",
    "ZERO",
    "ADD",
    "TWO",
    "ADD",
    "TWO",
    "ONE",
    "END",
)
FULL_LAYER0_TOKENS = (
    "MIN",
    "MAX",
    "ARG0",
    "SUB",
    "ZERO",
    "ADD",
    "TWO",
    "ADD",
    "TWO",
    "ONE",
    "ADD",
    "TWO",
    "ADD",
    "TWO",
    "ONE",
    "END",
)
VISIBLE_CASES = ((-6, -5), (6, 5), (0, 0))

COLUMN_COUNT = 4_096
ACTIVE_COLUMNS = 11
CODEBOOK_SEED = 0
CONSTRUCTION_SEED = 17
BASE_CELLS_PER_COLUMN = 8
THETA_MATCH = 16
THETA_ACTIVE = 18
EXPECTED_CAPACITY = {
    "training_sequence_count": 2,
    "noninitial_arrivals": 186,
    "segment_capacity": 2_046,
    "max_pyramidal_cells": 34_814,
}

EXPECTED_SIGNED_SOURCE_SHA256 = (
    "c1115e11d6fd336d259f071bf7640bd96764657415493ebd1fbe6028a61c7a4e"
)
EXPECTED_SIGNED_ARTIFACT_SHA256 = (
    "cce99ffdc9f6d310cf8c4cd3e1e10f83825853990f0d1bc149bd67fe77c3a7b1"
)
EXPECTED_PRIOR_CLAMP_SOURCE_SHA256 = (
    "6a60ad9a6fbddb3244cb7c6090d3ac874f5a6626d67dda89355664ffc23790a4"
)
EXPECTED_PRIOR_CLAMP_ARTIFACT_SHA256 = (
    "8400a11df244d2a573e32b491480b14e1dc7f6c061b6cb4ab00f7f11af09d839"
)
EXPECTED_HISTORICAL_BASELINE_SHA256 = (
    "524c9df0971b1684e22398866fef0108f48fd40ab3fe8caeb221196c1e753f8b"
)

PINNED_SOURCE_SHA256 = {
    "experiments/r3_pilot/public_simple_swe_role_chunk_signed_residual_action_probe.py": EXPECTED_SIGNED_SOURCE_SHA256,
    "experiments/r3_pilot/public_simple_swe_role_chunk_feedback_action_probe.py": "67017f8902620cd24c7e3a0f1cea320f3e2ff39bb53875873b03c7fe2e9e5a8b",
    "experiments/r3_pilot/public_simple_swe_role_chunk_layer0_action_probe.py": "8c2c86c1cdc6026639cdbce0ba88ac5145101c2c2dced8c3a56155f72326f3d1",
    "experiments/r3_pilot/public_simple_swe_role_chunk_route_probe.py": "5b4c80b9fe3a8d780b5a286fb33ad69885316e8b09a2a2723f7b2a46db1c1001",
    "experiments/r3_pilot/public_simple_swe_native_typed_clamp_probe.py": EXPECTED_PRIOR_CLAMP_SOURCE_SHA256,
    "src/humanbrain/native.py": "8963db700914828a0d5212419431ff50d50089c750be259c23fea69087748a6f",
    "src/humanbrain/native_temporal_runtime.py": "77b41b014d35648adbe989fe3c31059856f60410758dddb3d9794e54a0ed227a",
    "src/humanbrain/program.py": "ca5c46aa04688954099d029cc94829f4b8ca1e0fddc651d4aff2bdbe5fb05ddd",
    "src/humanbrain/role_chunk_episode.py": "8dc8a9102569f6bdc3135479bd38f8074993efc2a93234202fcf4e96d65940f5",
    "src/humanbrain/concept_tokens.py": "9070deaf7d9a8a54add24e3a213d8b6aa7bbb24b3d35ad66a7b68b3d1148f43e",
    "src/humanbrain/tokenizer.py": "08ef114a758006868944ad3091ca66619d3c082086ea2d44d93211d82a9791ce",
    "src/humanbrain/cortex.py": "7a7704a14324203d2cce427a11fbd58ffbbc345185ba34633449f59cf294624d",
    "src/humanbrain/sdr_union.py": "1de7f6edd914584c4c72d7e42426254590049cfb69cbbfa730afa3179afa63f9",
    "src/humanbrain/simple_swe/protocol.py": "22993997004a156b7d94b8d79198deae3e2f71f04d1aebc9740b342c0aa94849",
    "src/humanbrain/simple_swe/tasks.py": "d247bb1a81dca47846a787292a8a3f29c01938771070517b6fa35c9d8f7eed23",
    "src/humanbrain/simple_swe/sandbox.py": "2da8083bb48638bb0119b7e6a552a2134b915e07f2eecd1d69fd2dbf7d365b35",
    "src/humanbrain/simple_swe/_worker.py": "258aaf45decb8ed6635a85b92ee970591932b2cf53b691149a0d66777601c870",
    "src/humanbrain/_native_src/src/temporal_runtime.rs": "012d4f79505e3e0e381ebf9397db0c74d695120e7dfe8d511a338f8b896b9dc4",
    "src/humanbrain/_native_src/src/temporal_runtime/checkpoint.rs": "bdb1ab444a826e706b9c8284816a6e54847b0bb6c156723135324fffb377faba",
    "benchmarks/simple_swe_protocol_v1.json": "f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", type=Path, default=None)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _display_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _nested_task_records(value: object, task_id: str) -> tuple[dict[str, object], ...]:
    found: list[dict[str, object]] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if item.get("task_id") == task_id:
                found.append(item)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return tuple(found)


def _assert_anchors() -> dict[str, object]:
    if _sha256_file(SIGNED_SOURCE_PATH) != EXPECTED_SIGNED_SOURCE_SHA256:
        raise RuntimeError("signed-residual source anchor changed")
    if _sha256_file(SIGNED_ARTIFACT_PATH) != EXPECTED_SIGNED_ARTIFACT_SHA256:
        raise RuntimeError("signed-residual result anchor changed")
    signed = json.loads(SIGNED_ARTIFACT_PATH.read_text(encoding="utf-8"))
    if (
        signed.get("outcome") != "PASS_FIXED_SIGNED_RESIDUAL_ACTION_ROUTING"
        or not isinstance(signed.get("gates"), dict)
        or not all(signed["gates"].values())
        or signed.get("runtime", {}).get("probe_source_sha256")
        != EXPECTED_SIGNED_SOURCE_SHA256
    ):
        raise RuntimeError("signed-residual anchor is not an exact PASS")

    if _sha256_file(PRIOR_CLAMP_SOURCE_PATH) != EXPECTED_PRIOR_CLAMP_SOURCE_SHA256:
        raise RuntimeError("prior native clamp source anchor changed")
    if _sha256_file(PRIOR_CLAMP_ARTIFACT_PATH) != EXPECTED_PRIOR_CLAMP_ARTIFACT_SHA256:
        raise RuntimeError("prior native clamp result anchor changed")
    prior = json.loads(PRIOR_CLAMP_ARTIFACT_PATH.read_text(encoding="utf-8"))
    prior_007 = _nested_task_records(
        prior, str(TASK_RECORDS[LOWER_DONOR_ORDINAL]["task_id"])
    )
    if (
        prior.get("outcome") != "PASS_NATIVE_TYPED_CLAMP"
        or prior.get("capability_pass") is not True
        or not prior_007
    ):
        raise RuntimeError("/007 prior target reuse is no longer disclosed")

    if _sha256_file(HISTORICAL_BASELINE_PATH) != EXPECTED_HISTORICAL_BASELINE_SHA256:
        raise RuntimeError("historical prompt baseline changed")
    baseline = json.loads(HISTORICAL_BASELINE_PATH.read_text(encoding="utf-8"))
    disclosures = _nested_task_records(baseline, TARGET_TASK_ID)
    if not disclosures or any(
        record.get("prompt_sha256") != TASK_RECORDS[TARGET_ORDINAL]["prompt_sha256"]
        for record in disclosures
    ):
        raise RuntimeError("historical baseline no longer discloses the target prompt")
    return {
        "signed_residual_outcome": signed["outcome"],
        "signed_residual_source_sha256": EXPECTED_SIGNED_SOURCE_SHA256,
        "signed_residual_artifact_sha256": EXPECTED_SIGNED_ARTIFACT_SHA256,
        "donor_007_previously_used_as_native_typed_target": True,
        "prior_native_clamp_artifact_sha256": EXPECTED_PRIOR_CLAMP_ARTIFACT_SHA256,
        "target_historical_prompt_disclosure_count": len(disclosures),
        "freshness_claimed": False,
    }


def _assert_frozen_sources() -> None:
    for source, expected in PINNED_SOURCE_SHA256.items():
        if _sha256_file(REPO_ROOT / source) != expected:
            raise RuntimeError(f"frozen multistep-clamp dependency changed: {source}")


def _prior_public_artifact_inventory() -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    for path in sorted(PROBE_DIR.glob("public_simple_swe*.json")):
        if path.resolve() == OUTPUT_PATH.resolve():
            continue
        text = path.read_text(encoding="utf-8")
        if TARGET_TASK_ID in text:
            raise RuntimeError(f"target already appears in prior experiment artifact: {path}")
        inventory.append({"path": _display_path(path), "sha256": _sha256_file(path)})
    if not inventory:
        raise RuntimeError("prior public Simple-SWE artifact inventory is empty")
    return inventory


def _assert_committed_and_unspent() -> None:
    if OUTPUT_PATH.exists() or SPEND_MARKER_PATH.exists():
        raise RuntimeError("refusing spent multistep-clamp action probe")
    _assert_anchors()
    _assert_frozen_sources()
    _prior_public_artifact_inventory()
    paths = (
        Path(__file__).resolve(),
        SIGNED_ARTIFACT_PATH,
        PRIOR_CLAMP_ARTIFACT_PATH,
        HISTORICAL_BASELINE_PATH,
        *(REPO_ROOT / source for source in PINNED_SOURCE_SHA256),
    )
    for path in paths:
        relative = path.resolve().relative_to(REPO_ROOT).as_posix()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", relative],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout:
            raise RuntimeError(f"multistep-clamp source must be committed: {relative}")


def _select_tasks() -> tuple[
    object,
    tuple[GeneratedTask, ...],
    tuple[GeneratedTask, GeneratedTask],
    GeneratedTask,
]:
    protocol, fit_donors, _parent_probes = route_probe._select_tasks()
    visible = generate_visible_partition(protocol, "train")
    indexed = {(task.public.family, task.public.ordinal): task for task in visible}
    training = (
        indexed[(FAMILY, IDENTITY_DONOR_ORDINAL)],
        indexed[(FAMILY, LOWER_DONOR_ORDINAL)],
    )
    target = indexed[(FAMILY, TARGET_ORDINAL)]
    for task in (*training, target):
        expected = TASK_RECORDS[task.public.ordinal]
        if (
            task.public.partition != "train"
            or task.public.task_id != expected["task_id"]
            or task.public.task_sha256 != expected["task_sha256"]
            or task.public.hidden_test_sha256 != expected["hidden_test_sha256"]
            or hashlib.sha256(task.public.prompt.encode("utf-8")).hexdigest()
            != expected["prompt_sha256"]
        ):
            raise RuntimeError(f"public scalar-clamp /{task.public.ordinal:03d} changed")
    return protocol, fit_donors, training, target


def _layer0_signature(episode: RoleChunkEpisode, public: PublicTask) -> Layer0ProgramSignature:
    if episode.task_id != public.task_id or episode.restore_prompt() != public.prompt:
        raise ValueError("role episode does not match public scalar-clamp task")
    if public.partition != "train" or len(public.signature.parameters) != 1:
        raise ValueError("multistep clamp action requires one public TRAIN parameter")
    parameter = public.signature.parameters[0]
    if parameter.type.constructor != "Int" or public.signature.result.constructor != "Int":
        raise ValueError("multistep clamp action requires an Int -> Int signature")
    entries = episode.payloads_for(RolePayloadKind.ENTRY_POINT)
    parameters = episode.payloads_for(RolePayloadKind.PARAMETER)
    integers = episode.payloads_for(RolePayloadKind.INTEGER_LITERAL)
    floats = episode.payloads_for(RolePayloadKind.FLOAT_LITERAL)
    if len(entries) != 1 or len(parameters) != 1 or len(integers) != 2 or floats:
        raise ValueError("scalar-clamp role sidecar shape changed")
    if entries[0].value != public.signature.entry_point or parameters[0].value != parameter.name:
        raise ValueError("role sidecar disagrees with public signature")
    return Layer0ProgramSignature(entries[0].value, parameters[0].value)


def _public_bounds(episode: RoleChunkEpisode, public: PublicTask) -> tuple[int, int]:
    _layer0_signature(episode, public)
    payloads = episode.payloads_for(RolePayloadKind.INTEGER_LITERAL)
    try:
        bounds = tuple(int(payload.value) for payload in payloads)
    except ValueError as error:
        raise RuntimeError("public clamp bound is not an integer") from error
    if any(str(value) != payload.value for value, payload in zip(bounds, payloads, strict=True)):
        raise RuntimeError("public clamp bound is not canonical")
    if bounds != (LOWER_BOUND, UPPER_BOUND):
        raise RuntimeError("public clamp bounds differ from frozen visible task")
    return bounds


def _assert_episode_frozen(task: GeneratedTask, episode: RoleChunkEpisode) -> None:
    _public_bounds(episode, task.public)
    if (
        len(episode.token_ids) != CONCEPT_STREAM_EVENTS
        or episode.structural_sha256 != STRUCTURAL_SHA256
        or _sha256_json(tuple(episode.token_ids)) != CONCEPT_STREAM_SHA256
    ):
        raise RuntimeError("scalar-clamp task differs from frozen 88-event concept stream")


def _compile_source(
    tokens: tuple[str, ...], episode: RoleChunkEpisode, public: PublicTask
) -> str:
    signature = _layer0_signature(episode, public)
    if tokens == IDENTITY_LAYER0_TOKENS:
        max_nodes = 1
    elif tokens == LOWER_LAYER0_TOKENS:
        max_nodes = 9
    elif tokens == FULL_LAYER0_TOKENS:
        max_nodes = 15
    else:
        raise ValueError("Layer-0 source is outside the fixed clamp actuators")
    return Layer0ProgramCompiler(Layer0ProgramGrammar(max_nodes=max_nodes)).compile(
        tokens, signature
    )


def _visible_checker(source: str, entry_point: str) -> dict[str, object]:
    namespace: dict[str, object] = {"__builtins__": {"max": max, "min": min}}
    records: list[dict[str, object]] = []
    try:
        exec(compile(source, "<multistep-clamp-visible-checker>", "exec"), namespace)
        function = namespace.get(entry_point)
        if not callable(function):
            raise TypeError("compiled source did not define the public entry point")
        for index, (input_value, expected) in enumerate(VISIBLE_CASES):
            observed = function(input_value)
            if type(observed) is not int:
                raise TypeError("visible checker observed a non-exact integer")
            exact = observed == expected
            records.append(
                {
                    "index": index,
                    "input": input_value,
                    "expected": expected,
                    "observed": observed,
                    "passed": exact,
                }
            )
            if not exact:
                break
    except Exception as error:
        return {
            "frozen_cases": [list(case) for case in VISIBLE_CASES],
            "executed_cases": records,
            "classification": None,
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }

    low = [
        {"index": 0, "input": -6, "expected": -5, "observed": -6, "passed": False}
    ]
    high = [
        {"index": 0, "input": -6, "expected": -5, "observed": -5, "passed": True},
        {"index": 1, "input": 6, "expected": 5, "observed": 6, "passed": False},
    ]
    passed = [
        {
            "index": index,
            "input": input_value,
            "expected": expected,
            "observed": expected,
            "passed": True,
        }
        for index, (input_value, expected) in enumerate(VISIBLE_CASES)
    ]
    classification = "LOW" if records == low else "HIGH" if records == high else "PASS" if records == passed else None
    error = None if classification is not None else "unexpected exact visible-checker trajectory"
    return {
        "frozen_cases": [list(case) for case in VISIBLE_CASES],
        "executed_cases": records,
        "classification": classification,
        "passed": classification == "PASS",
        "error": error,
    }


def _observation_token(checker: dict[str, object]) -> int:
    if checker.get("error") is not None:
        raise RuntimeError("malformed or unexpected clamp checker result cannot become feedback")
    if checker.get("frozen_cases") != [list(case) for case in VISIBLE_CASES]:
        raise RuntimeError("visible clamp checker cases changed")
    classification = checker.get("classification")
    expected_pass = classification == "PASS"
    if checker.get("passed") is not expected_pass:
        raise RuntimeError("visible clamp checker pass flag is inconsistent")
    mapping = {
        "LOW": OBS_LOW_TOKEN_ID,
        "HIGH": OBS_HIGH_TOKEN_ID,
        "PASS": OBS_PASS_TOKEN_ID,
    }
    token = mapping.get(classification)
    if token is None:
        raise RuntimeError("unsupported clamp observation cannot become feedback")
    return token


def _training_sequences(
    training: tuple[GeneratedTask, GeneratedTask],
    episodes: dict[int, RoleChunkEpisode],
) -> tuple[tuple[int, ...], tuple[int, ...], list[dict[str, object]]]:
    identity_task, lower_task = training
    identity_episode = episodes[IDENTITY_DONOR_ORDINAL]
    lower_episode = episodes[LOWER_DONOR_ORDINAL]

    identity_source = _compile_source(
        IDENTITY_LAYER0_TOKENS, identity_episode, identity_task.public
    )
    identity_lower_source = _compile_source(
        LOWER_LAYER0_TOKENS, identity_episode, identity_task.public
    )
    identity_full_source = _compile_source(
        FULL_LAYER0_TOKENS, identity_episode, identity_task.public
    )
    identity_checkers = (
        _visible_checker(identity_source, identity_task.public.signature.entry_point),
        _visible_checker(identity_lower_source, identity_task.public.signature.entry_point),
        _visible_checker(identity_full_source, identity_task.public.signature.entry_point),
    )
    identity_observations = tuple(_observation_token(record) for record in identity_checkers)

    lower_source = _compile_source(LOWER_LAYER0_TOKENS, lower_episode, lower_task.public)
    lower_full_source = _compile_source(FULL_LAYER0_TOKENS, lower_episode, lower_task.public)
    lower_checkers = (
        _visible_checker(lower_source, lower_task.public.signature.entry_point),
        _visible_checker(lower_full_source, lower_task.public.signature.entry_point),
    )
    lower_observations = tuple(_observation_token(record) for record in lower_checkers)

    if identity_observations != (OBS_LOW_TOKEN_ID, OBS_HIGH_TOKEN_ID, OBS_PASS_TOKEN_ID):
        raise RuntimeError("identity donor observations differ from frozen trajectory")
    if lower_observations != (OBS_HIGH_TOKEN_ID, OBS_PASS_TOKEN_ID):
        raise RuntimeError("lower donor observations differ from frozen trajectory")

    identity_sequence = (
        *identity_episode.token_ids,
        TEST_TOKEN_ID,
        OBS_LOW_TOKEN_ID,
        FIX_LOWER_TOKEN_ID,
        OBS_HIGH_TOKEN_ID,
        FIX_FULL_TOKEN_ID,
        OBS_PASS_TOKEN_ID,
        DONE_TOKEN_ID,
    )
    lower_sequence = (
        *lower_episode.token_ids,
        TEST_TOKEN_ID,
        OBS_HIGH_TOKEN_ID,
        FIX_FULL_TOKEN_ID,
        OBS_PASS_TOKEN_ID,
        DONE_TOKEN_ID,
    )
    derivations = [
        {
            "task_id": identity_task.public.task_id,
            "initial_source_kind": "identity",
            "visible_checkers": list(identity_checkers),
            "derived_observation_token_ids": list(identity_observations),
            "teacher_action_token_ids": [
                TEST_TOKEN_ID,
                FIX_LOWER_TOKEN_ID,
                FIX_FULL_TOKEN_ID,
                DONE_TOKEN_ID,
            ],
            "all_observations_derived_before_learn": True,
        },
        {
            "task_id": lower_task.public.task_id,
            "initial_source_kind": "lower_only",
            "visible_checkers": list(lower_checkers),
            "derived_observation_token_ids": list(lower_observations),
            "teacher_action_token_ids": [TEST_TOKEN_ID, FIX_FULL_TOKEN_ID, DONE_TOKEN_ID],
            "all_observations_derived_before_learn": True,
        },
    ]
    return identity_sequence, lower_sequence, derivations


def _definition(
    sequences: tuple[tuple[int, ...], tuple[int, ...]], common_stream: tuple[int, ...]
) -> tuple[NativeTemporalCircuitDefinition, dict[str, int]]:
    noninitial_arrivals = sum(len(sequence) - 1 for sequence in sequences)
    segment_capacity = ACTIVE_COLUMNS * noninitial_arrivals
    config = NativeTemporalCircuitConfig(
        column_count=COLUMN_COUNT,
        max_pyramidal_cells=COLUMN_COUNT * BASE_CELLS_PER_COLUMN + segment_capacity,
        segment_capacity=segment_capacity,
        construction_seed=CONSTRUCTION_SEED,
        connected_permanence_ticks=250,
        initial_permanence_ticks=255,
        permanence_increment_ticks=50,
        negative_decrement_ticks=2,
        base_cells_per_column=BASE_CELLS_PER_COLUMN,
        novel_race_cap=BASE_CELLS_PER_COLUMN,
        prediction_quorum=ACTIVE_COLUMNS,
        max_segments_per_cell=64,
        theta_match=THETA_MATCH,
        theta_active=THETA_ACTIVE,
        context_policy=TemporalContextPolicy.BOUND_WINNERS,
        plasticity_policy=TemporalPlasticityPolicy.PLASTIC,
    )
    used = tuple(sorted({*(token for sequence in sequences for token in sequence), *common_stream}))
    token_codes = {
        token: tuple(
            int(column)
            for column in token_sdr_indices(
                token, n_bits=COLUMN_COUNT, w_ones=ACTIVE_COLUMNS, seed=CODEBOOK_SEED
            )
        )
        for token in used
    }
    if any(len(columns) != ACTIVE_COLUMNS for columns in token_codes.values()):
        raise RuntimeError("action/observation codes are not equal width")
    physical_cells = 2 * config.max_pyramidal_cells + 3 * config.column_count
    definition = NativeTemporalCircuitDefinition(
        config,
        token_codes,
        seeded_threshold_offsets_mv(physical_cells, config.construction_seed).tolist(),
    )
    capacity = {
        "training_sequence_count": len(sequences),
        "noninitial_arrivals": noninitial_arrivals,
        "segment_capacity": segment_capacity,
        "max_pyramidal_cells": config.max_pyramidal_cells,
    }
    if capacity != EXPECTED_CAPACITY:
        raise RuntimeError("multistep-clamp capacity differs from frozen arithmetic")
    return definition, capacity


def _prepare_training() -> tuple[
    object,
    tuple[GeneratedTask, GeneratedTask],
    GeneratedTask,
    RoleChunkEpisodeEncoder,
    dict[int, RoleChunkEpisode],
    NativeTemporalCircuitDefinition,
    bytes,
    object,
    dict[str, object],
    dict[str, int],
]:
    protocol, fit_donors, training, target = _select_tasks()
    encoder = RoleChunkEpisodeEncoder.fit_public_train(
        tuple(task.public for task in fit_donors),
        concept_config=route_probe.DEFAULT_ROLE_CHUNK_CONCEPT_CONFIG,
        base_config=route_probe.BASE_TOKENIZER_CONFIG,
    )
    if len(fit_donors) != 8 or not encoder.tokenizer.concepts:
        raise RuntimeError("existing eight-donor RoleChunk fit changed")
    episodes = {task.public.ordinal: encoder.encode(task.public) for task in training}
    for task in training:
        _assert_episode_frozen(task, episodes[task.public.ordinal])
    if len({tuple(episode.token_ids) for episode in episodes.values()}) != 1:
        raise RuntimeError("clamp donor concept streams are not bit exact")

    identity_sequence, lower_sequence, derivations = _training_sequences(training, episodes)
    sequences = (identity_sequence, lower_sequence)
    common_stream = tuple(episodes[IDENTITY_DONOR_ORDINAL].token_ids)
    definition, capacity = _definition(sequences, common_stream)
    train_episodes: list[dict[str, object]] = []
    with route_probe._public_train_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            for sequence, derivation in zip(sequences, derivations, strict=True):
                steps = circuit.advance_block(
                    sequence, mode=TemporalStepMode.LEARN, reset_before=True
                )
                train_episodes.append(
                    {
                        **derivation,
                        "sequence_token_ids": list(sequence),
                        **route_probe._step_summary(steps),
                    }
                )
            identity = circuit.checkpoint_identity()
            checkpoint = circuit.checkpoint()
    return (
        protocol,
        training,
        target,
        encoder,
        episodes,
        definition,
        checkpoint,
        identity,
        {
            "episodes": train_episodes,
            "checkpoint_sha256": _sha256_bytes(checkpoint),
            "checkpoint_identity": asdict(identity),
        },
        capacity,
    )


def _prepare_probe() -> tuple[
    object,
    tuple[GeneratedTask, GeneratedTask],
    GeneratedTask,
    dict[int, RoleChunkEpisode],
    NativeTemporalCircuitDefinition,
    bytes,
    object,
    dict[str, object],
    dict[str, int],
]:
    (
        protocol,
        training,
        target,
        encoder,
        episodes,
        definition,
        checkpoint,
        identity,
        train_record,
        capacity,
    ) = _prepare_training()
    target_episode = encoder.encode(target.public)
    _assert_episode_frozen(target, target_episode)
    if tuple(target_episode.token_ids) != tuple(episodes[IDENTITY_DONOR_ORDINAL].token_ids):
        raise RuntimeError("target does not share the donor clamp concept stream")
    return (
        protocol,
        training,
        target,
        {**episodes, TARGET_ORDINAL: target_episode},
        definition,
        checkpoint,
        identity,
        train_record,
        capacity,
    )


def _observed_surface_token(raw: dict[str, object]) -> int | None:
    generation = raw.get("generation")
    return None if not isinstance(generation, dict) else int(generation["surface_token_id"])


def _dispatch_observed_action(
    token: int | None,
    source: str,
    episode: RoleChunkEpisode,
    public: PublicTask,
) -> tuple[str, dict[str, object]]:
    action = ACTION_NAME_BY_TOKEN.get(token)
    if action is None:
        return source, {
            "observed_token_id": token,
            "dispatched": False,
            "action": None,
            "source_changed": False,
            "source_sha256_after": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        }
    updated = source
    if action == "FIX_LOWER":
        updated = _compile_source(LOWER_LAYER0_TOKENS, episode, public)
    elif action == "FIX_FULL":
        updated = _compile_source(FULL_LAYER0_TOKENS, episode, public)
    return updated, {
        "observed_token_id": token,
        "dispatched": True,
        "action": action,
        "source_changed": updated != source,
        "source_sha256_after": hashlib.sha256(updated.encode("utf-8")).hexdigest(),
    }


def _create_spend_marker(target: GeneratedTask, checkpoint_sha256: str) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-multistep-clamp-action",
        "state": "TARGET_ATTEMPT_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target_task_id": target.public.task_id,
        "checkpoint_sha256": checkpoint_sha256,
        "actual_generation_calls_planned": 4,
        "control_generation_calls_planned": 3,
        "atomic_no_argument_generation_calls_planned": 7,
        "canonical_target_sandbox_calls_planned": 1,
        "model_retries": 0,
        "harness_retries": 0,
        "probe_source_sha256": _sha256_file(Path(__file__)),
        "canonical_artifact_path": _display_path(OUTPUT_PATH),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with SPEND_MARKER_PATH.open("x", encoding="utf-8", newline="\n") as marker:
            marker.write(serialized)
            marker.flush()
            os.fsync(marker.fileno())
    except FileExistsError as error:
        raise RuntimeError("refusing already spent multistep-clamp target") from error
    return {
        "path": _display_path(SPEND_MARKER_PATH),
        "sha256": _sha256_file(SPEND_MARKER_PATH),
        "exclusive_create": True,
        "flushed_and_fsynced_before_all_seven_generate_calls": True,
    }


def _feedback_matches(record: dict[str, object], token: int) -> bool:
    return bool(
        record.get("advanced") is True
        and record.get("observation_token_id") == token
        and record.get("native_stop") is None
    )


def _score_after_all_calls(
    *,
    actual_calls: tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]],
    control_calls: tuple[dict[str, object], dict[str, object], dict[str, object]],
    actual_observations: tuple[int, int, int],
    control_observations: tuple[int, int],
    actual_feedback: tuple[dict[str, object], dict[str, object], dict[str, object]],
    control_feedback: tuple[dict[str, object], dict[str, object]],
    dispatches: tuple[dict[str, object], ...],
    actual_sources: tuple[str, ...],
    control_sources: tuple[str, ...],
    identity_source: str,
    lower_source: str,
    full_source: str,
    actual_checkers: tuple[dict[str, object], dict[str, object], dict[str, object]],
    control_checkers: tuple[dict[str, object], dict[str, object]],
    sandbox: dict[str, object],
) -> tuple[dict[str, object], dict[str, bool]]:
    expected_actual = (TEST_TOKEN_ID, FIX_LOWER_TOKEN_ID, FIX_FULL_TOKEN_ID, DONE_TOKEN_ID)
    expected_control = (TEST_TOKEN_ID, FIX_FULL_TOKEN_ID, DONE_TOKEN_ID)
    scored = {
        **{
            f"actual_{index + 1}": route_probe._score_generation(raw, expected)
            for index, (raw, expected) in enumerate(zip(actual_calls, expected_actual, strict=True))
        },
        **{
            f"control_{index + 1}": route_probe._score_generation(raw, expected)
            for index, (raw, expected) in enumerate(zip(control_calls, expected_control, strict=True))
        },
    }
    expected_dispatches = (
        (TEST_TOKEN_ID, "TEST", False),
        (TEST_TOKEN_ID, "TEST", False),
        (FIX_LOWER_TOKEN_ID, "FIX_LOWER", True),
        (FIX_FULL_TOKEN_ID, "FIX_FULL", True),
        (FIX_FULL_TOKEN_ID, "FIX_FULL", True),
        (DONE_TOKEN_ID, "DONE", False),
        (DONE_TOKEN_ID, "DONE", False),
    )
    gates = {
        "first_clone_generations_bit_exact": actual_calls[0] == control_calls[0],
        "physical_exact_four_and_three_routes": all(
            record.get("correct") is True for record in scored.values()
        ),
        "exact_coarse_observation_routes": bool(
            actual_observations == (OBS_LOW_TOKEN_ID, OBS_HIGH_TOKEN_ID, OBS_PASS_TOKEN_ID)
            and control_observations == (OBS_HIGH_TOKEN_ID, OBS_PASS_TOKEN_ID)
            and all(
                _feedback_matches(record, token)
                for record, token in zip(actual_feedback, actual_observations, strict=True)
            )
            and all(
                _feedback_matches(record, token)
                for record, token in zip(control_feedback, control_observations, strict=True)
            )
        ),
        "exact_observed_route_dispatch": bool(
            len(dispatches) == len(expected_dispatches)
            and all(
                dispatch.get("dispatched") is True
                and dispatch.get("observed_token_id") == token
                and dispatch.get("action") == action
                and dispatch.get("source_changed") is changed
                for dispatch, (token, action, changed) in zip(
                    dispatches, expected_dispatches, strict=True
                )
            )
        ),
        "exact_identity_lower_full_mutations": bool(
            actual_sources
            == (identity_source, identity_source, lower_source, full_source, full_source)
            and control_sources == (lower_source, lower_source, full_source, full_source)
            and sum(dispatch.get("source_changed") is True for dispatch in dispatches) == 3
        ),
        "actual_low_then_still_high_then_pass": bool(
            tuple(record.get("classification") for record in actual_checkers)
            == ("LOW", "HIGH", "PASS")
            and actual_checkers[0].get("passed") is False
            and actual_checkers[1].get("passed") is False
            and actual_checkers[2].get("passed") is True
        ),
        "control_high_then_pass": bool(
            tuple(record.get("classification") for record in control_checkers)
            == ("HIGH", "PASS")
            and control_checkers[0].get("passed") is False
            and control_checkers[1].get("passed") is True
        ),
        "done_actions_do_not_mutate": bool(
            dispatches[5].get("action") == "DONE"
            and dispatches[5].get("source_changed") is False
            and dispatches[6].get("action") == "DONE"
            and dispatches[6].get("source_changed") is False
        ),
        "canonical_target_train_sandbox_all_36": bool(
            sandbox.get("passed") is True
            and sandbox.get("sandboxed") is True
            and sandbox.get("outcome") == "passed"
            and int(sandbox.get("regression_case_count", 0))
            + int(sandbox.get("property_case_count", 0))
            == 36
        ),
    }
    return scored, gates


def main() -> int:
    args = _parse_args()
    _assert_committed_and_unspent()
    prior_inventory = _prior_public_artifact_inventory()
    anchors = _assert_anchors()
    started_ns = time.perf_counter_ns()
    (
        protocol,
        training,
        target,
        episodes,
        definition,
        checkpoint,
        checkpoint_identity,
        train_record,
        capacity,
    ) = _prepare_probe()
    del training

    executable = resolve_python_executable(args.python_executable)
    runtime_identity = python_runtime_identity(executable, protocol.raw.get("language"))
    if runtime_identity.get("python_version") != "3.11.15":
        raise RuntimeError("multistep-clamp probe requires repo Python 3.11.15")
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = _sha256_file(worker_path)
    worker_source = load_worker_snapshot(worker_sha256)

    target_episode = episodes[TARGET_ORDINAL]
    signature = _layer0_signature(target_episode, target.public)
    _public_bounds(target_episode, target.public)
    identity_source = _compile_source(IDENTITY_LAYER0_TOKENS, target_episode, target.public)
    lower_source = _compile_source(LOWER_LAYER0_TOKENS, target_episode, target.public)
    full_source = _compile_source(FULL_LAYER0_TOKENS, target_episode, target.public)
    actual_source = identity_source
    control_source = lower_source
    actual_sources = [actual_source]
    control_sources = [control_source]

    with route_probe._public_train_authority():
        with ExitStack() as stack:
            actual = stack.enter_context(
                NativeTemporalCircuit.from_checkpoint(definition, checkpoint)
            )
            control = stack.enter_context(
                NativeTemporalCircuit.from_checkpoint(definition, checkpoint)
            )
            if (
                actual.checkpoint_identity() != checkpoint_identity
                or control.checkpoint_identity() != checkpoint_identity
            ):
                raise RuntimeError("multistep-clamp checkpoint clone differs")
            actual_prompt = actual.advance_block(
                tuple(target_episode.token_ids),
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            control_prompt = control.advance_block(
                tuple(target_episode.token_ids),
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            if route_probe._step_summary(actual_prompt) != route_probe._step_summary(control_prompt):
                raise RuntimeError("checkpoint clones received different target prompt traces")

            marker = _create_spend_marker(target, _sha256_bytes(checkpoint))
            actual_first = route_probe._generate_once(actual)
            control_first = route_probe._generate_once(control)
            actual_source, actual_first_dispatch = _dispatch_observed_action(
                _observed_surface_token(actual_first), actual_source, target_episode, target.public
            )
            control_source, control_first_dispatch = _dispatch_observed_action(
                _observed_surface_token(control_first), control_source, target_episode, target.public
            )
            actual_sources.append(actual_source)
            control_sources.append(control_source)
            actual_first_checker = _visible_checker(actual_source, signature.entry_point)
            control_first_checker = _visible_checker(control_source, signature.entry_point)
            actual_first_observation = _observation_token(actual_first_checker)
            control_first_observation = _observation_token(control_first_checker)
            actual_first_feedback = feedback_probe._advance_feedback(
                actual, actual_first_observation
            )
            control_first_feedback = feedback_probe._advance_feedback(
                control, control_first_observation
            )

            actual_second = route_probe._generate_once(actual)
            control_second = route_probe._generate_once(control)
            actual_source, actual_second_dispatch = _dispatch_observed_action(
                _observed_surface_token(actual_second), actual_source, target_episode, target.public
            )
            control_source, control_second_dispatch = _dispatch_observed_action(
                _observed_surface_token(control_second), control_source, target_episode, target.public
            )
            actual_sources.append(actual_source)
            control_sources.append(control_source)
            actual_second_checker = _visible_checker(actual_source, signature.entry_point)
            control_second_checker = _visible_checker(control_source, signature.entry_point)
            actual_second_observation = _observation_token(actual_second_checker)
            control_second_observation = _observation_token(control_second_checker)
            actual_second_feedback = feedback_probe._advance_feedback(
                actual, actual_second_observation
            )
            control_second_feedback = feedback_probe._advance_feedback(
                control, control_second_observation
            )

            actual_third = route_probe._generate_once(actual)
            control_third = route_probe._generate_once(control)
            actual_source, actual_third_dispatch = _dispatch_observed_action(
                _observed_surface_token(actual_third), actual_source, target_episode, target.public
            )
            control_source, control_third_dispatch = _dispatch_observed_action(
                _observed_surface_token(control_third), control_source, target_episode, target.public
            )
            actual_sources.append(actual_source)
            control_sources.append(control_source)
            actual_third_checker = _visible_checker(actual_source, signature.entry_point)
            actual_third_observation = _observation_token(actual_third_checker)
            actual_third_feedback = feedback_probe._advance_feedback(
                actual, actual_third_observation
            )

            actual_fourth = route_probe._generate_once(actual)
            actual_source, actual_fourth_dispatch = _dispatch_observed_action(
                _observed_surface_token(actual_fourth), actual_source, target_episode, target.public
            )
            actual_sources.append(actual_source)

    sandbox = action_probe._sandbox_record(
        actual_source,
        target,
        protocol,
        python_executable=executable,
        runtime_identity=runtime_identity,
        worker_source=worker_source,
        worker_sha256=worker_sha256,
    )

    actual_calls = (actual_first, actual_second, actual_third, actual_fourth)
    control_calls = (control_first, control_second, control_third)
    actual_observations = (
        actual_first_observation,
        actual_second_observation,
        actual_third_observation,
    )
    control_observations = (control_first_observation, control_second_observation)
    actual_feedback = (
        actual_first_feedback,
        actual_second_feedback,
        actual_third_feedback,
    )
    control_feedback = (control_first_feedback, control_second_feedback)
    dispatches = (
        actual_first_dispatch,
        control_first_dispatch,
        actual_second_dispatch,
        control_second_dispatch,
        actual_third_dispatch,
        control_third_dispatch,
        actual_fourth_dispatch,
    )
    actual_checkers = (
        actual_first_checker,
        actual_second_checker,
        actual_third_checker,
    )
    control_checkers = (control_first_checker, control_second_checker)
    scored_calls, gates = _score_after_all_calls(
        actual_calls=actual_calls,
        control_calls=control_calls,
        actual_observations=actual_observations,
        control_observations=control_observations,
        actual_feedback=actual_feedback,
        control_feedback=control_feedback,
        dispatches=dispatches,
        actual_sources=tuple(actual_sources),
        control_sources=tuple(control_sources),
        identity_source=identity_source,
        lower_source=lower_source,
        full_source=full_source,
        actual_checkers=actual_checkers,
        control_checkers=control_checkers,
        sandbox=sandbox,
    )
    passed = all(gates.values())
    outcome = (
        "PASS_FIXED_MULTISTEP_CLAMP_ACTION_ROUTING"
        if passed
        else "STOP_FIXED_MULTISTEP_CLAMP_ACTION_ROUTING"
    )
    native_identity = loaded_library_identity()
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-multistep-clamp-action",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": outcome,
        "claim_boundary": (
            "Visible TRAIN only: one repeated normalized scalar-clamp template routes fixed LOW/HIGH/PASS observations through fixed lower/full Layer-0 actuators over multiple steps. "
            "This is fixed multi-step actuator routing, not semantic error interpretation, self-correction, program synthesis, transfer, DEV/eval, held-out, M0, or C1 capability."
        ),
        "authority": {
            "partition": "train",
            "target_task_id": target.public.task_id,
            "donor_007_previously_used_as_native_typed_target": True,
            "target_historical_prompt_baseline_reuse": True,
            "freshness_claimed": False,
            "checkpoint_clones": 2,
            "identical_prompt_presentations": 2,
            "actual_atomic_no_argument_generation_calls": 4,
            "control_atomic_no_argument_generation_calls": 3,
            "total_atomic_no_argument_generation_calls": 7,
            "canonical_target_sandbox_calls": 1,
            "model_retries": 0,
            "harness_retries": 0,
            "hidden_cases_or_verdicts_entered_cell_runtime": False,
            "expected_action_labels_scored_only_after_all_seven_calls": True,
            "prior_public_artifact_inventory": prior_inventory,
        },
        "anchors": anchors,
        "tasks": {str(ordinal): record for ordinal, record in TASK_RECORDS.items()},
        "representation": {
            "fit_donor_count": 8,
            "all_three_streams_bit_exact": True,
            "concept_stream_events": CONCEPT_STREAM_EVENTS,
            "structural_sha256": STRUCTURAL_SHA256,
            "concept_stream_sha256": CONCEPT_STREAM_SHA256,
        },
        "training": train_record,
        "physical_definition": {
            "circuit_identity_sha256": definition.circuit_identity_sha256(),
            "column_count": COLUMN_COUNT,
            "active_columns": ACTIVE_COLUMNS,
            "theta_match": THETA_MATCH,
            "theta_active": THETA_ACTIVE,
            "context_policy": TemporalContextPolicy.BOUND_WINNERS.name,
            "constants_status": "inherited fixture constants; not calibration-v3 selection",
            "equal_width_action_and_observation_codes": True,
            **capacity,
        },
        "spend_marker": marker,
        "prompt": route_probe._step_summary(actual_prompt),
        "actual_identity_start": {
            "source_trace": actual_sources,
            "raw_generations": list(actual_calls),
            "visible_checkers": list(actual_checkers),
            "observation_token_ids": list(actual_observations),
            "feedback_steps": list(actual_feedback),
            "dispatches": [
                actual_first_dispatch,
                actual_second_dispatch,
                actual_third_dispatch,
                actual_fourth_dispatch,
            ],
            "final_source": actual_source,
        },
        "lower_start_causal_control": {
            "source_trace": control_sources,
            "raw_generations": list(control_calls),
            "visible_checkers": list(control_checkers),
            "observation_token_ids": list(control_observations),
            "feedback_steps": list(control_feedback),
            "dispatches": [
                control_first_dispatch,
                control_second_dispatch,
                control_third_dispatch,
            ],
            "final_source": control_source,
        },
        "scored_after_all_seven_calls": scored_calls,
        "sandbox": sandbox,
        "gates": gates,
        "runtime": {
            "elapsed_wall_ns": time.perf_counter_ns() - started_ns,
            "python": sys.version,
            "platform": platform.platform(),
            "native_source_digest": native_identity.source_digest,
            "native_binary_sha256": native_identity.binary_sha256,
            "probe_source_sha256": _sha256_file(Path(__file__)),
            "pinned_source_sha256": {
                source: _sha256_file(REPO_ROOT / source)
                for source in PINNED_SOURCE_SHA256
            },
        },
    }
    with OUTPUT_PATH.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"outcome": outcome, "gates": gates}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
