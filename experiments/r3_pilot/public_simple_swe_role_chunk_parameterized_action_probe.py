"""R3-PAC0: one-shot public-TRAIN role-bound parameterized construction probe.

Two fresh fitted-cell circuits are composed without sharing learned state.  A
RoleChunk sign router sees only the sign of a residual derived by running an
identity source and physically elects BUILD_ADD or BUILD_SUB.  The elected
token is then forwarded unchanged as a temporal input to a fresh role/place
builder.  That builder must physically elect the operator, copy the public
entrypoint and magnitude from ordinal slots, emit every remaining source byte,
and terminate through ordinary EOS.

Scalar-offset /004 appeared in the historical prompt-only syntax-error
baseline under an older task hash, so this probe makes no freshness claim.  It
does assert that /004 has no prior routed-action artifact or spend marker.
Importing or compiling this module does not spend the target.
"""

from __future__ import annotations

import argparse
import ast
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
import public_simple_swe_role_chunk_signed_residual_action_probe as signed_probe
import public_train_dev_role_place_scalar_routing_probe as role_place_probe
from humanbrain.cortex import seeded_threshold_offsets_mv
from humanbrain.native import loaded_library_identity
from humanbrain.native_temporal_runtime import (
    COPY_END_TOKEN_ID,
    SLOT_COUNT,
    NativeTemporalCircuit,
    NativeTemporalCircuitConfig,
    NativeTemporalCircuitDefinition,
    TemporalContextPolicy,
    TemporalGenerationPath,
    TemporalPlasticityPolicy,
    TemporalStepMode,
    copy_slot_token_id,
    slot_end_token_id,
    slot_start_token_id,
)
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
from humanbrain.simple_swe.tasks import GeneratedTask, generate_visible_partition


OUTPUT_PATH = Path(__file__).with_suffix(".json")
SPEND_MARKER_PATH = Path(__file__).with_name(
    f"{Path(__file__).stem}.t0_scalar_offset-004.spent.json"
)
SIGNED_SOURCE_PATH = Path(__file__).with_name(
    "public_simple_swe_role_chunk_signed_residual_action_probe.py"
)
SIGNED_ARTIFACT_PATH = Path(__file__).with_name(
    "public_simple_swe_role_chunk_signed_residual_action_probe.json"
)
ROLE_PLACE_SOURCE_PATH = Path(__file__).with_name(
    "public_train_dev_role_place_scalar_routing_probe.py"
)
ROLE_PLACE_ARTIFACT_PATH = Path(__file__).with_name(
    "public_train_dev_role_place_scalar_routing_probe.json"
)
ROUTE_ARTIFACT_PATH = Path(__file__).with_name(
    "public_simple_swe_role_chunk_route_probe.json"
)
HISTORICAL_BASELINE_PATH = REPO_ROOT / "artifacts" / "simple_swe_public_baseline_v1.json"

FAMILY = "t0_scalar_offset"
ADD_DONOR_ORDINAL = 6
SUB_DONOR_ORDINAL = 5
TARGET_ORDINAL = 4
PAYLOAD_CONTROL_MAGNITUDE = 8
TASK_RECORDS = {
    6: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/006",
        "task_sha256": "77d64375abe077ae005e519d6f7b19a3f122edd4766813f2f2bbc0ccc36af68f",
        "hidden_test_sha256": "3d24c71a350a25aa0adb5e0ee56b888539ce9cd49a0925c4f6d1739d62af6f13",
        "prompt_sha256": "f05f67071561279ca9ff9b89f07878f0acd1ea071027fcedffbefce4e27d2855",
        "entry_point": "solve_2c9984a52db14adc",
        "parameter": "n",
        "offset": 2,
    },
    5: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/005",
        "task_sha256": "0e6aab1811d3f356ea461681b16fd9fa3cea6ed017635a36c6a5bfa174e8329b",
        "hidden_test_sha256": "e2bcf8f5603cba4d9803cd620366598e61db15c7ad7504750dd03b51bc211229",
        "prompt_sha256": "0943ca4f7a05635155329a2fda88affdb791a1239107031e9b09247f499d7bf9",
        "entry_point": "solve_6acb574ebf911e92",
        "parameter": "n",
        "offset": -8,
    },
    4: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/004",
        "task_sha256": "699c033a254eae423ac718e2d2182225373eb5700e32dc4714f73b61a65b1ddc",
        "hidden_test_sha256": "ac8d729519a713aaffed81be6a7c687a6fa4da764a31ed91f14581e579bbdaed",
        "prompt_sha256": "7882a35a22cc6b538172be93988c86d040f3ed8ba41557e76b9990ad0506925e",
        "entry_point": "solve_5899fcfd152e762e",
        "parameter": "n",
        "offset": -9,
    },
}
TARGET_TASK_ID = str(TASK_RECORDS[TARGET_ORDINAL]["task_id"])
HISTORICAL_TARGET_TASK_SHA256 = (
    "bb2fa5fd7c1047b744e59dc6b444ae08567fd24a9a6d4e9873696072806d1261"
)
STRUCTURAL_SHA256 = "dd70041ef789a3ed38cf4cf23196068b3f7e2cc45fb5f1f7de562c7d8d778a46"
CONCEPT_STREAM_SHA256 = "fab3180295cc0ced8715005d36474a8221e1b7c706c84ba2e8824176890294db"
CONCEPT_STREAM_EVENTS = 56

TEST_TOKEN_ID = 1540
BUILD_ADD_TOKEN_ID = 1546
BUILD_SUB_TOKEN_ID = 1547
OBS_POS_TOKEN_ID = 1606
OBS_NEG_TOKEN_ID = 1607
BUILD_NAME_BY_TOKEN = {
    BUILD_ADD_TOKEN_ID: "BUILD_ADD",
    BUILD_SUB_TOKEN_ID: "BUILD_SUB",
}

ROUTER_COLUMN_COUNT = 4_096
BUILDER_COLUMN_COUNT = 512
ACTIVE_COLUMNS = 11
CODEBOOK_SEED = 0
BASE_CELLS_PER_COLUMN = 8
CONSTRUCTION_SEED = 17
THETA_MATCH = 16
THETA_ACTIVE = 18
EXPECTED_ROUTER_CAPACITY = {
    "training_sequence_lengths": [59, 59],
    "noninitial_arrivals": 116,
    "segment_capacity": 1_276,
    "max_pyramidal_cells": 34_044,
}
EXPECTED_BUILDER_CAPACITY = {
    "training_sequence_lengths": [89, 89],
    "noninitial_arrivals": 176,
    "segment_capacity": 1_936,
    "max_pyramidal_cells": 6_032,
}

EOS_TOKEN_ID = role_place_probe.EOS_TOKEN_ID
BUILDER_PROMPT_PREFIX = b"Function "
BUILDER_PARAMETER_ROLE = b" parameter "
BUILDER_VALUE_ROLE = b" value "
BUILDER_ROUTE_SEPARATOR = b" "
BUILDER_PROMPT_SUFFIX = b".\n"
SOURCE_PREFIX = b"def "
SOURCE_BEFORE_OPERATOR = b"(n):\n    return n "
SOURCE_BEFORE_MAGNITUDE = b" "
SOURCE_SUFFIX = b"\n"

EXPECTED_SIGNED_SOURCE_SHA256 = (
    "c1115e11d6fd336d259f071bf7640bd96764657415493ebd1fbe6028a61c7a4e"
)
EXPECTED_SIGNED_ARTIFACT_SHA256 = (
    "cce99ffdc9f6d310cf8c4cd3e1e10f83825853990f0d1bc149bd67fe77c3a7b1"
)
EXPECTED_ROLE_PLACE_SOURCE_SHA256 = (
    "b3b58662a2e578a780af7a35483484d1bae69afcc54e7e34e4383fad6f5f8016"
)
EXPECTED_ROLE_PLACE_ARTIFACT_SHA256 = (
    "3afac696077d7524444b9128c52e1c0655468955697626a0451eaaf539f23ec4"
)
EXPECTED_ROUTE_ARTIFACT_SHA256 = (
    "9ee0df214df3525a2e2898129b03295d9ad20706e26a7c52039db40c7a62884b"
)
EXPECTED_HISTORICAL_BASELINE_SHA256 = (
    "524c9df0971b1684e22398866fef0108f48fd40ab3fe8caeb221196c1e753f8b"
)

PINNED_SOURCE_SHA256 = {
    "experiments/r3_pilot/public_simple_swe_role_chunk_signed_residual_action_probe.py": EXPECTED_SIGNED_SOURCE_SHA256,
    "experiments/r3_pilot/public_simple_swe_role_chunk_feedback_action_probe.py": "67017f8902620cd24c7e3a0f1cea320f3e2ff39bb53875873b03c7fe2e9e5a8b",
    "experiments/r3_pilot/public_simple_swe_role_chunk_layer0_action_probe.py": "8c2c86c1cdc6026639cdbce0ba88ac5145101c2c2dced8c3a56155f72326f3d1",
    "experiments/r3_pilot/public_simple_swe_role_chunk_route_probe.py": "5b4c80b9fe3a8d780b5a286fb33ad69885316e8b09a2a2723f7b2a46db1c1001",
    "experiments/r3_pilot/public_train_dev_role_place_scalar_routing_probe.py": EXPECTED_ROLE_PLACE_SOURCE_SHA256,
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
    "src/humanbrain/_native_src/src/temporal_runtime/role_place_copy.rs": "3d454eaf4a98ed6fe8073fe6008a85d55bb948fd814515cc619d908214f189c4",
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

    if _sha256_file(ROLE_PLACE_SOURCE_PATH) != EXPECTED_ROLE_PLACE_SOURCE_SHA256:
        raise RuntimeError("role/place source anchor changed")
    if _sha256_file(ROLE_PLACE_ARTIFACT_PATH) != EXPECTED_ROLE_PLACE_ARTIFACT_SHA256:
        raise RuntimeError("role/place result anchor changed")
    role_place = json.loads(ROLE_PLACE_ARTIFACT_PATH.read_text(encoding="utf-8"))
    if (
        role_place.get("outcome") != "PASS_TWO_ROUTE_VARIABLE_LITERAL_SCALAR_SEMANTICS"
        or role_place.get("capability_pass") is not True
        or not all(
            record.get("post_generation_score", {}).get("copy_transactions_exact") is True
            for record in role_place.get("dev_teacher_free", [])
        )
    ):
        raise RuntimeError("role/place anchor is not an exact physical COPY PASS")

    if _sha256_file(ROUTE_ARTIFACT_PATH) != EXPECTED_ROUTE_ARTIFACT_SHA256:
        raise RuntimeError("role-chunk route artifact changed")
    route = json.loads(ROUTE_ARTIFACT_PATH.read_text(encoding="utf-8"))
    donor_reuse = {
        ordinal: bool(_nested_task_records(route, str(TASK_RECORDS[ordinal]["task_id"])))
        for ordinal in (ADD_DONOR_ORDINAL, SUB_DONOR_ORDINAL)
    }
    if donor_reuse != {ADD_DONOR_ORDINAL: True, SUB_DONOR_ORDINAL: True}:
        raise RuntimeError("/005 and /006 prior route-donor reuse is no longer disclosed")

    if _sha256_file(HISTORICAL_BASELINE_PATH) != EXPECTED_HISTORICAL_BASELINE_SHA256:
        raise RuntimeError("historical prompt baseline changed")
    baseline = json.loads(HISTORICAL_BASELINE_PATH.read_text(encoding="utf-8"))
    disclosures = _nested_task_records(baseline, TARGET_TASK_ID)
    if (
        not disclosures
        or any(
            record.get("prompt_sha256") != TASK_RECORDS[TARGET_ORDINAL]["prompt_sha256"]
            or record.get("task_sha256") != HISTORICAL_TARGET_TASK_SHA256
            for record in disclosures
        )
        or HISTORICAL_TARGET_TASK_SHA256 == TASK_RECORDS[TARGET_ORDINAL]["task_sha256"]
    ):
        raise RuntimeError("historical /004 prompt disclosure or task-content drift changed")
    return {
        "signed_residual_outcome": signed["outcome"],
        "signed_residual_artifact_sha256": EXPECTED_SIGNED_ARTIFACT_SHA256,
        "role_place_outcome": role_place["outcome"],
        "role_place_artifact_sha256": EXPECTED_ROLE_PLACE_ARTIFACT_SHA256,
        "route_artifact_sha256": EXPECTED_ROUTE_ARTIFACT_SHA256,
        "donors_005_and_006_previously_used_in_role_chunk_routing": True,
        "target_historical_prompt_disclosure_count": len(disclosures),
        "historical_target_task_sha256": HISTORICAL_TARGET_TASK_SHA256,
        "current_target_task_sha256": TASK_RECORDS[TARGET_ORDINAL]["task_sha256"],
        "historical_task_content_drift": True,
        "freshness_claimed": False,
    }


def _assert_frozen_sources() -> None:
    for source, expected in PINNED_SOURCE_SHA256.items():
        if _sha256_file(REPO_ROOT / source) != expected:
            raise RuntimeError(f"frozen R3-PAC0 dependency changed: {source}")


def _prior_routed_target_inventory() -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    for path in sorted(PROBE_DIR.glob("*.json")):
        if path.resolve() == OUTPUT_PATH.resolve():
            continue
        text = path.read_text(encoding="utf-8")
        if TARGET_TASK_ID in text:
            raise RuntimeError(f"target already appears in prior routed artifact/marker: {path}")
        inventory.append({"path": _display_path(path), "sha256": _sha256_file(path)})
    if not inventory:
        raise RuntimeError("prior R3 artifact inventory is empty")
    return inventory


def _assert_committed_and_unspent() -> None:
    if OUTPUT_PATH.exists() or SPEND_MARKER_PATH.exists():
        raise RuntimeError("refusing spent R3-PAC0 target")
    _assert_anchors()
    _assert_frozen_sources()
    _prior_routed_target_inventory()
    paths = (
        Path(__file__).resolve(),
        SIGNED_ARTIFACT_PATH,
        ROLE_PLACE_ARTIFACT_PATH,
        ROUTE_ARTIFACT_PATH,
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
            raise RuntimeError(f"R3-PAC0 source must be committed: {relative}")


def _select_tasks() -> tuple[
    object,
    tuple[GeneratedTask, ...],
    tuple[GeneratedTask, GeneratedTask],
    GeneratedTask,
]:
    protocol, fit_donors, _route_probes = route_probe._select_tasks()
    visible = generate_visible_partition(protocol, "train")
    indexed = {(task.public.family, task.public.ordinal): task for task in visible}
    training = (
        indexed[(FAMILY, ADD_DONOR_ORDINAL)],
        indexed[(FAMILY, SUB_DONOR_ORDINAL)],
    )
    target = indexed[(FAMILY, TARGET_ORDINAL)]
    for task in (*training, target):
        expected = TASK_RECORDS[task.public.ordinal]
        parameter = task.public.signature.parameters[0]
        if (
            task.public.partition != "train"
            or task.public.task_id != expected["task_id"]
            or task.public.task_sha256 != expected["task_sha256"]
            or task.public.hidden_test_sha256 != expected["hidden_test_sha256"]
            or hashlib.sha256(task.public.prompt.encode("utf-8")).hexdigest()
            != expected["prompt_sha256"]
            or task.public.signature.entry_point != expected["entry_point"]
            or parameter.name != expected["parameter"]
            or parameter.type.constructor != "Int"
            or task.public.signature.result.constructor != "Int"
        ):
            raise RuntimeError(f"public scalar-offset /{task.public.ordinal:03d} changed")
    return protocol, fit_donors, training, target


def _fit_role_chunk_encoder(fit_donors: tuple[GeneratedTask, ...]) -> RoleChunkEpisodeEncoder:
    encoder = RoleChunkEpisodeEncoder.fit_public_train(
        tuple(task.public for task in fit_donors),
        concept_config=route_probe.DEFAULT_ROLE_CHUNK_CONCEPT_CONFIG,
        base_config=route_probe.BASE_TOKENIZER_CONFIG,
    )
    if len(fit_donors) != 8 or not encoder.tokenizer.concepts:
        raise RuntimeError("existing eight-donor RoleChunk fit changed")
    return encoder


def _offset(episode: RoleChunkEpisode, task: GeneratedTask) -> int:
    signed_probe._layer0_signature(episode, task.public)
    payloads = episode.payloads_for(RolePayloadKind.INTEGER_LITERAL)
    if len(payloads) != 1:
        raise RuntimeError("scalar-offset sidecar no longer has exactly one integer")
    try:
        value = int(payloads[0].value)
    except ValueError as error:
        raise RuntimeError("scalar-offset payload is not an integer") from error
    if str(value) != payloads[0].value or value != TASK_RECORDS[task.public.ordinal]["offset"]:
        raise RuntimeError("scalar-offset payload differs from frozen public identity")
    return value


def _assert_concept_episode(episode: RoleChunkEpisode, task: GeneratedTask) -> None:
    _offset(episode, task)
    if (
        len(episode.token_ids) != CONCEPT_STREAM_EVENTS
        or episode.structural_sha256 != STRUCTURAL_SHA256
        or _sha256_json(tuple(episode.token_ids)) != CONCEPT_STREAM_SHA256
    ):
        raise RuntimeError("scalar-offset task differs from frozen concept stream")


def _identity_source(episode: RoleChunkEpisode, task: GeneratedTask) -> str:
    return signed_probe._compile_source(
        signed_probe.IDENTITY_LAYER0_TOKENS, episode, task.public
    )


def _residual_checker(
    episode: RoleChunkEpisode, task: GeneratedTask
) -> dict[str, object]:
    offset = _offset(episode, task)
    signature = signed_probe._layer0_signature(episode, task.public)
    return signed_probe._visible_checker(
        _identity_source(episode, task), signature.entry_point, offset
    )


def _sign_observation(checker: dict[str, object]) -> int:
    if checker.get("error") is not None or checker.get("inputs") != list(signed_probe.VISIBLE_INPUTS):
        raise RuntimeError("malformed residual checker cannot become sign feedback")
    residual = checker.get("residual")
    if (
        not isinstance(residual, list)
        or len(residual) != len(signed_probe.VISIBLE_INPUTS)
        or any(type(value) is not int for value in residual)
        or len(set(residual)) != 1
        or residual[0] == 0
    ):
        raise RuntimeError("mixed, zero, or non-integer residual cannot become sign feedback")
    return OBS_POS_TOKEN_ID if residual[0] > 0 else OBS_NEG_TOKEN_ID


def _router_sequences(
    training: tuple[GeneratedTask, GeneratedTask],
    episodes: dict[int, RoleChunkEpisode],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], list[dict[str, object]]]:
    sequences: list[tuple[int, ...]] = []
    records: list[dict[str, object]] = []
    for task in training:
        episode = episodes[task.public.ordinal]
        checker = _residual_checker(episode, task)
        observation = _sign_observation(checker)
        build = BUILD_ADD_TOKEN_ID if observation == OBS_POS_TOKEN_ID else BUILD_SUB_TOKEN_ID
        sequence = (*episode.token_ids, TEST_TOKEN_ID, observation, build)
        sequences.append(sequence)
        records.append(
            {
                "task_id": task.public.task_id,
                "public_offset": _offset(episode, task),
                "identity_source": _identity_source(episode, task),
                "visible_residual_checker": checker,
                "derived_sign_token_id": observation,
                "teacher_build_token_id": build,
                "residual_derived_before_learn": True,
                "only_sign_token_entered_cells_after_test": True,
                "sequence_token_ids": list(sequence),
            }
        )
    if (
        records[0]["derived_sign_token_id"] != OBS_POS_TOKEN_ID
        or records[1]["derived_sign_token_id"] != OBS_NEG_TOKEN_ID
    ):
        raise RuntimeError("donor sign observations differ from frozen trajectories")
    return (sequences[0], sequences[1]), records


def _router_definition(
    sequences: tuple[tuple[int, ...], tuple[int, ...]]
) -> tuple[NativeTemporalCircuitDefinition, dict[str, object]]:
    lengths = [len(sequence) for sequence in sequences]
    noninitial_arrivals = sum(length - 1 for length in lengths)
    segment_capacity = ACTIVE_COLUMNS * noninitial_arrivals
    config = NativeTemporalCircuitConfig(
        column_count=ROUTER_COLUMN_COUNT,
        max_pyramidal_cells=ROUTER_COLUMN_COUNT * BASE_CELLS_PER_COLUMN
        + segment_capacity,
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
    used = tuple(sorted({token for sequence in sequences for token in sequence}))
    token_codes = {
        token: tuple(
            int(column)
            for column in token_sdr_indices(
                token,
                n_bits=ROUTER_COLUMN_COUNT,
                w_ones=ACTIVE_COLUMNS,
                seed=CODEBOOK_SEED,
            )
        )
        for token in used
    }
    physical_cells = 2 * config.max_pyramidal_cells + 3 * config.column_count
    definition = NativeTemporalCircuitDefinition(
        config,
        token_codes,
        seeded_threshold_offsets_mv(physical_cells, config.construction_seed).tolist(),
    )
    capacity: dict[str, object] = {
        "training_sequence_lengths": lengths,
        "noninitial_arrivals": noninitial_arrivals,
        "segment_capacity": segment_capacity,
        "max_pyramidal_cells": config.max_pyramidal_cells,
    }
    if capacity != EXPECTED_ROUTER_CAPACITY:
        raise RuntimeError("router capacity differs from exact sequence arithmetic")
    return definition, capacity


def _builder_payload(
    task: GeneratedTask,
    episode: RoleChunkEpisode,
    *,
    magnitude_override: int | None = None,
) -> dict[str, object]:
    signature = signed_probe._layer0_signature(episode, task.public)
    offset = _offset(episode, task)
    magnitude = abs(offset) if magnitude_override is None else magnitude_override
    if magnitude <= 0 or str(magnitude).startswith("-"):
        raise RuntimeError("builder magnitude must be a positive decimal")
    identifier = signature.entry_point.encode("ascii")
    parameter = signature.parameter_name.encode("ascii")
    magnitude_bytes = str(magnitude).encode("ascii")
    if parameter != b"n" or not magnitude_bytes.isdigit():
        raise RuntimeError("R3-PAC0 builder is frozen to scalar parameter n and decimal magnitude")
    return {
        "task_id": task.public.task_id,
        "identifier": identifier,
        "parameter": parameter,
        "magnitude": magnitude_bytes,
        "public_offset": offset,
        "magnitude_override": magnitude_override,
    }


def _require_build_token(build_token: int) -> int:
    if build_token not in BUILD_NAME_BY_TOKEN:
        raise RuntimeError("observed route is not a physical BUILD token")
    return build_token


def _builder_prompt(payload: dict[str, object], build_token: int) -> tuple[int, ...]:
    build_token = _require_build_token(build_token)
    identifier = bytes(payload["identifier"])
    parameter = bytes(payload["parameter"])
    magnitude = bytes(payload["magnitude"])
    return (
        *BUILDER_PROMPT_PREFIX,
        slot_start_token_id(0),
        *identifier,
        slot_end_token_id(0),
        *BUILDER_PARAMETER_ROLE,
        slot_start_token_id(1),
        *parameter,
        slot_end_token_id(1),
        *BUILDER_VALUE_ROLE,
        slot_start_token_id(2),
        *magnitude,
        slot_end_token_id(2),
        *BUILDER_ROUTE_SEPARATOR,
        build_token,
        *BUILDER_PROMPT_SUFFIX,
    )


def _builder_internal_target(
    payload: dict[str, object], build_token: int
) -> tuple[int, ...]:
    build_token = _require_build_token(build_token)
    operator = ord("+") if build_token == BUILD_ADD_TOKEN_ID else ord("-")
    parameter = bytes(payload["parameter"])
    if parameter != b"n":
        raise RuntimeError("R3-PAC0 source skeleton is frozen to n")
    return (
        *SOURCE_PREFIX,
        copy_slot_token_id(0),
        *SOURCE_BEFORE_OPERATOR,
        operator,
        *SOURCE_BEFORE_MAGNITUDE,
        copy_slot_token_id(2),
        *SOURCE_SUFFIX,
        EOS_TOKEN_ID,
    )


def _builder_train_episode(
    payload: dict[str, object], build_token: int
) -> tuple[int, ...]:
    return (*_builder_prompt(payload, build_token), *_builder_internal_target(payload, build_token))


def _builder_definition(
    sequences: tuple[tuple[int, ...], tuple[int, ...]]
) -> tuple[NativeTemporalCircuitDefinition, dict[str, object]]:
    lengths = [len(sequence) for sequence in sequences]
    noninitial_arrivals = sum(length - 1 for length in lengths)
    segment_capacity = ACTIVE_COLUMNS * noninitial_arrivals
    config = NativeTemporalCircuitConfig(
        column_count=BUILDER_COLUMN_COUNT,
        max_pyramidal_cells=BUILDER_COLUMN_COUNT * BASE_CELLS_PER_COLUMN
        + segment_capacity,
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
    role_tokens = {
        *(slot_start_token_id(slot) for slot in range(SLOT_COUNT)),
        *(slot_end_token_id(slot) for slot in range(SLOT_COUNT)),
        *(copy_slot_token_id(slot) for slot in range(SLOT_COUNT)),
        COPY_END_TOKEN_ID,
    }
    used = tuple(
        sorted(
            {
                *range(128),
                EOS_TOKEN_ID,
                BUILD_ADD_TOKEN_ID,
                BUILD_SUB_TOKEN_ID,
                *role_tokens,
                *(token for sequence in sequences for token in sequence),
            }
        )
    )
    token_codes = {
        token: tuple(
            int(column)
            for column in token_sdr_indices(
                token,
                n_bits=BUILDER_COLUMN_COUNT,
                w_ones=ACTIVE_COLUMNS,
                seed=CODEBOOK_SEED,
            )
        )
        for token in used
    }
    physical_cells = 2 * config.max_pyramidal_cells + 3 * config.column_count
    definition = NativeTemporalCircuitDefinition(
        config,
        token_codes,
        seeded_threshold_offsets_mv(physical_cells, config.construction_seed).tolist(),
    )
    capacity: dict[str, object] = {
        "training_sequence_lengths": lengths,
        "noninitial_arrivals": noninitial_arrivals,
        "segment_capacity": segment_capacity,
        "max_pyramidal_cells": config.max_pyramidal_cells,
    }
    if capacity != EXPECTED_BUILDER_CAPACITY:
        raise RuntimeError("builder capacity differs from exact sequence arithmetic")
    return definition, capacity


def _train_circuit(
    definition: NativeTemporalCircuitDefinition,
    sequences: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[bytes, object, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    with route_probe._public_train_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            for sequence in sequences:
                steps = circuit.advance_block(
                    sequence, mode=TemporalStepMode.LEARN, reset_before=True
                )
                records.append(
                    {
                        "sequence_token_ids": list(sequence),
                        **route_probe._step_summary(steps),
                    }
                )
            identity = circuit.checkpoint_identity()
            checkpoint = circuit.checkpoint()
    return checkpoint, identity, records


def _prepare_training() -> dict[str, object]:
    protocol, fit_donors, training, target = _select_tasks()
    encoder = _fit_role_chunk_encoder(fit_donors)
    episodes = {task.public.ordinal: encoder.encode(task.public) for task in training}
    for task in training:
        _assert_concept_episode(episodes[task.public.ordinal], task)
    if len({tuple(episode.token_ids) for episode in episodes.values()}) != 1:
        raise RuntimeError("router donor concept streams are not bit exact")

    router_sequences, router_derivations = _router_sequences(training, episodes)
    router_definition, router_capacity = _router_definition(router_sequences)
    router_checkpoint, router_identity, router_steps = _train_circuit(
        router_definition, router_sequences
    )
    router_train = []
    for derivation, steps in zip(router_derivations, router_steps, strict=True):
        router_train.append({**derivation, **steps})

    add_payload = _builder_payload(training[0], episodes[ADD_DONOR_ORDINAL])
    sub_payload = _builder_payload(training[1], episodes[SUB_DONOR_ORDINAL])
    builder_sequences = (
        _builder_train_episode(add_payload, BUILD_ADD_TOKEN_ID),
        _builder_train_episode(sub_payload, BUILD_SUB_TOKEN_ID),
    )
    builder_definition, builder_capacity = _builder_definition(builder_sequences)
    builder_checkpoint, builder_identity, builder_steps = _train_circuit(
        builder_definition, builder_sequences
    )
    builder_train = []
    for task, payload, route, sequence, steps in zip(
        training,
        (add_payload, sub_payload),
        (BUILD_ADD_TOKEN_ID, BUILD_SUB_TOKEN_ID),
        builder_sequences,
        builder_steps,
        strict=True,
    ):
        builder_train.append(
            {
                "task_id": task.public.task_id,
                "entrypoint_slot": bytes(payload["identifier"]).decode("ascii"),
                "parameter_slot": bytes(payload["parameter"]).decode("ascii"),
                "magnitude_slot": bytes(payload["magnitude"]).decode("ascii"),
                "teacher_build_token_id": route,
                "temporal_target_contains_literal_magnitude": False,
                **steps,
            }
        )
    return {
        "protocol": protocol,
        "training": training,
        "target": target,
        "encoder": encoder,
        "episodes": episodes,
        "router_definition": router_definition,
        "router_checkpoint": router_checkpoint,
        "router_checkpoint_identity": router_identity,
        "router_train": router_train,
        "router_capacity": router_capacity,
        "builder_definition": builder_definition,
        "builder_checkpoint": builder_checkpoint,
        "builder_checkpoint_identity": builder_identity,
        "builder_train": builder_train,
        "builder_capacity": builder_capacity,
    }


def _prepare_probe() -> dict[str, object]:
    prepared = _prepare_training()
    target = prepared["target"]
    encoder = prepared["encoder"]
    target_episode = encoder.encode(target.public)
    _assert_concept_episode(target_episode, target)
    donor_episode = prepared["episodes"][ADD_DONOR_ORDINAL]
    if tuple(target_episode.token_ids) != tuple(donor_episode.token_ids):
        raise RuntimeError("target does not share the donor concept stream")
    return {**prepared, "target_episode": target_episode}


def _observed_surface_token(raw: dict[str, object]) -> int | None:
    generation = raw.get("generation")
    return None if not isinstance(generation, dict) else int(generation["surface_token_id"])


def _create_spend_marker(
    target: GeneratedTask, router_checkpoint_sha256: str, builder_checkpoint_sha256: str
) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-parameterized-action",
        "state": "TARGET_ATTEMPT_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target_task_id": target.public.task_id,
        "router_checkpoint_sha256": router_checkpoint_sha256,
        "builder_checkpoint_sha256": builder_checkpoint_sha256,
        "router_target_no_argument_generation_calls_planned": 4,
        "builder_target_runs_planned": 3,
        "builder_target_no_argument_generation_calls_maximum": 3
        * role_place_probe.MAX_GENERATIONS,
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
        raise RuntimeError("refusing already spent R3-PAC0 target") from error
    return {
        "path": _display_path(SPEND_MARKER_PATH),
        "sha256": _sha256_file(SPEND_MARKER_PATH),
        "exclusive_create": True,
        "flushed_and_fsynced_before_all_target_router_and_builder_generate_calls": True,
    }


def _expected_source_bytes(payload: dict[str, object], build_token: int) -> bytes:
    build_token = _require_build_token(build_token)
    operator = b"+" if build_token == BUILD_ADD_TOKEN_ID else b"-"
    return (
        SOURCE_PREFIX
        + bytes(payload["identifier"])
        + SOURCE_BEFORE_OPERATOR
        + operator
        + SOURCE_BEFORE_MAGNITUDE
        + bytes(payload["magnitude"])
        + SOURCE_SUFFIX
    )


def _analyze_builder_run(
    payload: dict[str, object], build_token: int, run: dict[str, object]
) -> dict[str, object]:
    emitted = bytes(run["emitted"])
    expected = _expected_source_bytes(payload, build_token)
    source: str | None = None
    compile_valid = False
    function_name: str | None = None
    semantic_exact = False
    semantic_cases: list[dict[str, object]] = []
    namespace: dict[str, object] = {}
    try:
        source = emitted.decode("ascii", errors="strict")
        module = ast.parse(source, filename="<r3-pac0-builder>", mode="exec")
        code = compile(module, "<r3-pac0-builder>", "exec")
        compile_valid = True
        if len(module.body) == 1 and isinstance(module.body[0], ast.FunctionDef):
            function_name = module.body[0].name
        exec(code, {"__builtins__": {}}, namespace)
    except (UnicodeDecodeError, SyntaxError, ValueError, TypeError, NameError):
        pass

    identifier = bytes(payload["identifier"]).decode("ascii")
    magnitude = int(bytes(payload["magnitude"]).decode("ascii"))
    signed_offset = magnitude if build_token == BUILD_ADD_TOKEN_ID else -magnitude
    function = namespace.get(identifier)
    if callable(function):
        semantic_exact = True
        for value in signed_probe.VISIBLE_INPUTS:
            expected_value = value + signed_offset
            try:
                observed = function(value)
            except Exception as error:
                observed = None
                error_name: str | None = type(error).__name__
                semantic_exact = False
            else:
                error_name = None
                semantic_exact &= type(observed) is int and observed == expected_value
            semantic_cases.append(
                {
                    "input": value,
                    "expected": expected_value,
                    "observed": observed,
                    "error": error_name,
                }
            )

    transactions, transaction_structure = role_place_probe._copy_transactions(
        run["generations"]
    )
    expected_places = [
        list(range(len(bytes(payload["identifier"])))),
        list(range(len(bytes(payload["magnitude"])))),
    ]
    copy_exact = bool(
        transaction_structure
        and [record["copied_places"] for record in transactions] == expected_places
    )
    operator = ord("+") if build_token == BUILD_ADD_TOKEN_ID else ord("-")
    operator_records = [
        generation
        for generation in run["generations"]
        if generation["path"] == TemporalGenerationPath.TEMPORAL.name
        and generation["surface_token_id"] == operator
    ]
    operator_physical = bool(
        len(operator_records) == 1
        and operator_records[0]["physical_election_count"] == 1
        and operator_records[0]["winner_input_columns"] == ACTIVE_COLUMNS
    )
    final_eos = bool(
        run["generations"]
        and run["generations"][-1]["surface_token_id"] == EOS_TOKEN_ID
        and run["generations"][-1]["path"] == TemporalGenerationPath.TEMPORAL.name
    )
    passed = bool(
        run["stop"] is None
        and run["terminated_by_eos"] is True
        and final_eos
        and emitted == expected
        and compile_valid
        and function_name == identifier
        and copy_exact
        and operator_physical
        and semantic_exact
    )
    return {
        "passed": passed,
        "source": source,
        "emitted_sha256": hashlib.sha256(emitted).hexdigest(),
        "expected_sha256": hashlib.sha256(expected).hexdigest(),
        "exact_expected_source": emitted == expected,
        "compile_valid": compile_valid,
        "function_name": function_name,
        "signed_offset": signed_offset,
        "copy_transactions_exact": copy_exact,
        "copy_transactions": transactions,
        "operator_elected_physically": operator_physical,
        "ordinary_eos": final_eos,
        "semantic_exact": semantic_exact,
        "semantic_cases": semantic_cases,
    }


def _serializable_builder_run(
    run: dict[str, object], analysis: dict[str, object]
) -> dict[str, object]:
    emitted = bytes(run["emitted"])
    return {
        "emitted_code_text": emitted.decode("ascii", errors="replace"),
        "emitted_code_hex": emitted.hex(),
        "emitted_code_sha256": hashlib.sha256(emitted).hexdigest(),
        "generations": run["generations"],
        "terminated_by_eos": run["terminated_by_eos"],
        "stop": run["stop"],
        "post_generation_analysis": analysis,
    }


def _feedback_matches(record: dict[str, object], token: int) -> bool:
    return bool(
        record.get("advanced") is True
        and record.get("observation_token_id") == token
        and record.get("native_stop") is None
    )


def _score_after_all_target_calls(
    *,
    actual_first: dict[str, object],
    intervention_first: dict[str, object],
    actual_second: dict[str, object],
    intervention_second: dict[str, object],
    actual_observation: int,
    intervention_observation: int,
    actual_feedback: dict[str, object],
    intervention_feedback: dict[str, object],
    actual_build: int | None,
    intervention_build: int | None,
    builder_prompts: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    actual_analysis: dict[str, object],
    operation_analysis: dict[str, object],
    payload_analysis: dict[str, object],
    actual_checker: dict[str, object],
    operation_checker: dict[str, object],
    payload_checker: dict[str, object],
    sandbox: dict[str, object],
) -> tuple[dict[str, object], dict[str, bool]]:
    router_scored = {
        "actual_first": route_probe._score_generation(actual_first, TEST_TOKEN_ID),
        "intervention_first": route_probe._score_generation(
            intervention_first, TEST_TOKEN_ID
        ),
        "actual_second": route_probe._score_generation(
            actual_second, BUILD_SUB_TOKEN_ID
        ),
        "intervention_second": route_probe._score_generation(
            intervention_second, BUILD_ADD_TOKEN_ID
        ),
    }
    actual_prompt, operation_prompt, payload_prompt = builder_prompts
    build_tokens = {BUILD_ADD_TOKEN_ID, BUILD_SUB_TOKEN_ID}
    actual_without_route = tuple(token for token in actual_prompt if token not in build_tokens)
    operation_without_route = tuple(
        token for token in operation_prompt if token not in build_tokens
    )
    payload_differences = [
        (actual_token, payload_token)
        for actual_token, payload_token in zip(actual_prompt, payload_prompt, strict=True)
        if actual_token != payload_token
    ]
    gates = {
        "router_first_generations_bit_exact_test": bool(
            actual_first == intervention_first
            and router_scored["actual_first"]["correct"]
            and router_scored["intervention_first"]["correct"]
        ),
        "real_negative_and_injected_positive_are_only_feedback_tokens": bool(
            actual_observation == OBS_NEG_TOKEN_ID
            and intervention_observation == OBS_POS_TOKEN_ID
            and _feedback_matches(actual_feedback, OBS_NEG_TOKEN_ID)
            and _feedback_matches(intervention_feedback, OBS_POS_TOKEN_ID)
        ),
        "physical_sub_add_build_elections": bool(
            router_scored["actual_second"]["correct"]
            and router_scored["intervention_second"]["correct"]
            and actual_build == BUILD_SUB_TOKEN_ID
            and intervention_build == BUILD_ADD_TOKEN_ID
        ),
        "observed_build_tokens_forwarded_as_temporal_inputs": bool(
            actual_prompt.count(BUILD_SUB_TOKEN_ID) == 1
            and operation_prompt.count(BUILD_ADD_TOKEN_ID) == 1
            and payload_prompt.count(BUILD_SUB_TOKEN_ID) == 1
        ),
        "operation_and_payload_interventions_change_one_factor": bool(
            actual_without_route == operation_without_route
            and payload_differences == [(ord("9"), ord("8"))]
        ),
        "actual_role_bound_n_minus_9": bool(
            actual_analysis.get("passed") is True
            and actual_analysis.get("signed_offset") == -9
        ),
        "operation_flip_retains_9_and_emits_plus_9": bool(
            operation_analysis.get("passed") is True
            and operation_analysis.get("signed_offset") == 9
        ),
        "payload_intervention_retains_sub_and_emits_minus_8": bool(
            payload_analysis.get("passed") is True
            and payload_analysis.get("signed_offset") == -8
        ),
        "actual_visible_pass_controls_fail_for_intended_reasons": bool(
            actual_checker.get("passed") is True
            and actual_checker.get("residual") == [0, 0, 0]
            and operation_checker.get("passed") is False
            and operation_checker.get("residual") == [-18, -18, -18]
            and payload_checker.get("passed") is False
            and payload_checker.get("residual") == [-1, -1, -1]
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
    return router_scored, gates


def main() -> int:
    args = _parse_args()
    _assert_committed_and_unspent()
    prior_inventory = _prior_routed_target_inventory()
    anchors = _assert_anchors()
    started_ns = time.perf_counter_ns()
    prepared = _prepare_probe()
    protocol = prepared["protocol"]
    target = prepared["target"]
    target_episode = prepared["target_episode"]
    target_offset = _offset(target_episode, target)
    if target_offset != -9:
        raise RuntimeError("R3-PAC0 target offset changed")

    executable = resolve_python_executable(args.python_executable)
    runtime_identity = python_runtime_identity(executable, protocol.raw.get("language"))
    if runtime_identity.get("python_version") != "3.11.15":
        raise RuntimeError("R3-PAC0 requires repo Python 3.11.15")
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = _sha256_file(worker_path)
    worker_source = load_worker_snapshot(worker_sha256)

    router_definition = prepared["router_definition"]
    router_checkpoint = prepared["router_checkpoint"]
    router_identity = prepared["router_checkpoint_identity"]
    builder_definition = prepared["builder_definition"]
    builder_checkpoint = prepared["builder_checkpoint"]
    builder_identity = prepared["builder_checkpoint_identity"]

    with route_probe._public_train_authority():
        with ExitStack() as stack:
            actual_router = stack.enter_context(
                NativeTemporalCircuit.from_checkpoint(router_definition, router_checkpoint)
            )
            intervention_router = stack.enter_context(
                NativeTemporalCircuit.from_checkpoint(router_definition, router_checkpoint)
            )
            if (
                actual_router.checkpoint_identity() != router_identity
                or intervention_router.checkpoint_identity() != router_identity
            ):
                raise RuntimeError("router checkpoint clones differ")
            actual_prompt_steps = actual_router.advance_block(
                tuple(target_episode.token_ids),
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            intervention_prompt_steps = intervention_router.advance_block(
                tuple(target_episode.token_ids),
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            if route_probe._step_summary(actual_prompt_steps) != route_probe._step_summary(
                intervention_prompt_steps
            ):
                raise RuntimeError("router clones received different target prompt traces")

            marker = _create_spend_marker(
                target,
                _sha256_bytes(router_checkpoint),
                _sha256_bytes(builder_checkpoint),
            )
            actual_first = route_probe._generate_once(actual_router)
            intervention_first = route_probe._generate_once(intervention_router)

            actual_residual_checker = _residual_checker(target_episode, target)
            actual_observation = _sign_observation(actual_residual_checker)
            intervention_observation = OBS_POS_TOKEN_ID
            actual_feedback = feedback_probe._advance_feedback(
                actual_router, actual_observation
            )
            intervention_feedback = feedback_probe._advance_feedback(
                intervention_router, intervention_observation
            )
            actual_second = route_probe._generate_once(actual_router)
            intervention_second = route_probe._generate_once(intervention_router)

        actual_build = _observed_surface_token(actual_second)
        intervention_build = _observed_surface_token(intervention_second)
        actual_build_for_prompt = _require_build_token(actual_build)
        intervention_build_for_prompt = _require_build_token(intervention_build)

        actual_payload = _builder_payload(target, target_episode)
        operation_payload = _builder_payload(target, target_episode)
        payload_intervention = _builder_payload(
            target, target_episode, magnitude_override=PAYLOAD_CONTROL_MAGNITUDE
        )
        actual_builder_prompt = _builder_prompt(actual_payload, actual_build_for_prompt)
        operation_builder_prompt = _builder_prompt(
            operation_payload, intervention_build_for_prompt
        )
        payload_builder_prompt = _builder_prompt(
            payload_intervention, actual_build_for_prompt
        )

        with ExitStack() as stack:
            actual_builder = stack.enter_context(
                NativeTemporalCircuit.from_checkpoint(builder_definition, builder_checkpoint)
            )
            operation_builder = stack.enter_context(
                NativeTemporalCircuit.from_checkpoint(builder_definition, builder_checkpoint)
            )
            payload_builder = stack.enter_context(
                NativeTemporalCircuit.from_checkpoint(builder_definition, builder_checkpoint)
            )
            if any(
                circuit.checkpoint_identity() != builder_identity
                for circuit in (actual_builder, operation_builder, payload_builder)
            ):
                raise RuntimeError("builder checkpoint clones differ")
            actual_builder_prompt_steps = actual_builder.advance_block(
                actual_builder_prompt,
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            operation_builder_prompt_steps = operation_builder.advance_block(
                operation_builder_prompt,
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            payload_builder_prompt_steps = payload_builder.advance_block(
                payload_builder_prompt,
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            actual_builder_run = role_place_probe._generate_teacher_free(actual_builder)
            operation_builder_run = role_place_probe._generate_teacher_free(operation_builder)
            payload_builder_run = role_place_probe._generate_teacher_free(payload_builder)

    actual_analysis = _analyze_builder_run(
        actual_payload, actual_build_for_prompt, actual_builder_run
    )
    operation_analysis = _analyze_builder_run(
        operation_payload, intervention_build_for_prompt, operation_builder_run
    )
    payload_analysis = _analyze_builder_run(
        payload_intervention, actual_build_for_prompt, payload_builder_run
    )
    actual_source = actual_analysis.get("source")
    operation_source = operation_analysis.get("source")
    payload_source = payload_analysis.get("source")
    if not all(isinstance(source, str) for source in (actual_source, operation_source, payload_source)):
        raise RuntimeError("builder output did not decode to source")

    signature = signed_probe._layer0_signature(target_episode, target.public)
    actual_checker = signed_probe._visible_checker(
        actual_source, signature.entry_point, target_offset
    )
    operation_checker = signed_probe._visible_checker(
        operation_source, signature.entry_point, target_offset
    )
    payload_checker = signed_probe._visible_checker(
        payload_source, signature.entry_point, target_offset
    )
    sandbox = action_probe._sandbox_record(
        actual_source,
        target,
        protocol,
        python_executable=executable,
        runtime_identity=runtime_identity,
        worker_source=worker_source,
        worker_sha256=worker_sha256,
    )

    router_scored, gates = _score_after_all_target_calls(
        actual_first=actual_first,
        intervention_first=intervention_first,
        actual_second=actual_second,
        intervention_second=intervention_second,
        actual_observation=actual_observation,
        intervention_observation=intervention_observation,
        actual_feedback=actual_feedback,
        intervention_feedback=intervention_feedback,
        actual_build=actual_build,
        intervention_build=intervention_build,
        builder_prompts=(
            actual_builder_prompt,
            operation_builder_prompt,
            payload_builder_prompt,
        ),
        actual_analysis=actual_analysis,
        operation_analysis=operation_analysis,
        payload_analysis=payload_analysis,
        actual_checker=actual_checker,
        operation_checker=operation_checker,
        payload_checker=payload_checker,
        sandbox=sandbox,
    )
    passed = all(gates.values())
    outcome = "PASS_R3_PAC0_ROLE_BOUND_CONSTRUCTION" if passed else "STOP_R3_PAC0_ROLE_BOUND_CONSTRUCTION"
    native_identity = loaded_library_identity()
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-parameterized-action",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": outcome,
        "claim_boundary": (
            "Public TRAIN only: an execution-derived residual sign is reduced to one cell input; a fitted-cell router elects a BUILD token; and a separate fresh fitted-cell role/place builder uses that physical token plus public sidecar slots to emit one exact scalar source. This is narrow role-bound parameterized construction, not numeric inference, arbitrary AST construction, semantic diagnosis, transfer, DEV/eval, held-out, M0, or C1 capability."
        ),
        "limitations": {
            "target_magnitude_source": "absolute public RoleChunk integer sidecar payload; not execution residual",
            "host_factorization": (
                "Python obtains residual sign from real identity execution and absolute magnitude from the public sidecar; only the coarse sign token enters the router, while the magnitude enters builder slot 2."
            ),
            "fixed_source_skeleton": "def <copied-entrypoint>(n): return n <cell-elected-operator> <copied-magnitude>",
            "fixed_parameter": "n",
            "numeral_9_claimed_novel_corpus_wide": False,
            "donor_task_reuse": "public TRAIN scalar-offset /005 and /006 were prior role-chunk routing donors; both circuits here are fresh",
            "fresh_target_claimed": False,
        },
        "authority": {
            "partition": "train",
            "target_task_id": target.public.task_id,
            "historical_prompt_baseline_exposure": True,
            "historical_task_content_drift": True,
            "freshness_claimed": False,
            "donors_005_and_006_previously_used_in_role_chunk_routing": True,
            "no_prior_routed_action_target_or_spend_marker": True,
            "router_target_checkpoint_clones": 2,
            "builder_target_checkpoint_clones": 3,
            "router_target_no_argument_generation_calls": 4,
            "builder_target_no_argument_generation_calls": sum(
                len(run["generations"])
                for run in (
                    actual_builder_run,
                    operation_builder_run,
                    payload_builder_run,
                )
            ),
            "canonical_target_sandbox_calls": 1,
            "model_retries": 0,
            "harness_retries": 0,
            "hidden_cases_or_verdicts_entered_cell_runtime": False,
            "expected_routes_and_sources_scored_only_after_all_target_generate_calls": True,
            "python_bridge": "forward observed BUILD token unchanged into builder temporal prompt",
            "python_selected_target_operator_or_source": False,
            "prior_routed_artifact_inventory": prior_inventory,
        },
        "anchors": anchors,
        "tasks": {str(ordinal): record for ordinal, record in TASK_RECORDS.items()},
        "representation": {
            "fit_donor_count": 8,
            "all_three_concept_streams_bit_exact": True,
            "concept_stream_events": CONCEPT_STREAM_EVENTS,
            "structural_sha256": STRUCTURAL_SHA256,
            "concept_stream_sha256": CONCEPT_STREAM_SHA256,
        },
        "training": {
            "router": prepared["router_train"],
            "builder": prepared["builder_train"],
            "router_checkpoint_sha256": _sha256_bytes(router_checkpoint),
            "builder_checkpoint_sha256": _sha256_bytes(builder_checkpoint),
        },
        "physical_definitions": {
            "router": {
                "circuit_identity_sha256": router_definition.circuit_identity_sha256(),
                "column_count": ROUTER_COLUMN_COUNT,
                "active_columns": ACTIVE_COLUMNS,
                "theta_match": THETA_MATCH,
                "theta_active": THETA_ACTIVE,
                "context_policy": TemporalContextPolicy.BOUND_WINNERS.name,
                **prepared["router_capacity"],
            },
            "builder": {
                "circuit_identity_sha256": builder_definition.circuit_identity_sha256(),
                "column_count": BUILDER_COLUMN_COUNT,
                "active_columns": ACTIVE_COLUMNS,
                "theta_match": THETA_MATCH,
                "theta_active": THETA_ACTIVE,
                "context_policy": TemporalContextPolicy.BOUND_WINNERS.name,
                **prepared["builder_capacity"],
            },
            "constants_status": "inherited fixture constants; no calibration claim",
        },
        "spend_marker": marker,
        "router_target": {
            "prompt": route_probe._step_summary(actual_prompt_steps),
            "actual_real_residual_checker": actual_residual_checker,
            "actual_observation_token_id": actual_observation,
            "operation_intervention_observation_token_id": intervention_observation,
            "actual_feedback": actual_feedback,
            "intervention_feedback": intervention_feedback,
            "actual_first_generation": actual_first,
            "intervention_first_generation": intervention_first,
            "actual_second_generation": actual_second,
            "intervention_second_generation": intervention_second,
            "actual_observed_build_token_id": actual_build,
            "intervention_observed_build_token_id": intervention_build,
            "scored_after_all_target_calls": router_scored,
        },
        "builder_target": {
            "actual_prompt_steps": route_probe._step_summary(actual_builder_prompt_steps),
            "operation_prompt_steps": route_probe._step_summary(
                operation_builder_prompt_steps
            ),
            "payload_prompt_steps": route_probe._step_summary(payload_builder_prompt_steps),
            "actual": _serializable_builder_run(actual_builder_run, actual_analysis),
            "operation_intervention": _serializable_builder_run(
                operation_builder_run, operation_analysis
            ),
            "payload_intervention": _serializable_builder_run(
                payload_builder_run, payload_analysis
            ),
        },
        "visible_target_checker": {
            "actual": actual_checker,
            "operation_intervention": operation_checker,
            "payload_intervention": payload_checker,
        },
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
