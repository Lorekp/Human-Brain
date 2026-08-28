"""Contingent public-TRAIN R3-PAC2 scalar/list source-shape construction.

Four visible TRAIN donors teach a prompt-plus-serialized-observation shape
route and a role/place source builder.  The eventual sole target is the
explicitly unused ``t1_list_map_add/009`` follow-on.  This module is
intentionally fail-closed until the upstream R3-PAC1 artifact records a pass;
importing it and preparing donor material never encodes or executes the target.

Claim boundary: a two-shape, role-bound direct-source builder over public TRAIN
templates.  The target prompt also identifies its family and all MAP donors
have positive residuals, so this does not isolate observation-only shape
causality or cross-shape sign causality.  It is not arbitrary AST synthesis,
family-disjoint transfer, DEV/eval, a protected milestone, M0, or C1 evidence.
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PROBE_DIR = Path(__file__).resolve().parent
for import_root in (SRC_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

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
    copy_slot_token_id,
    slot_end_token_id,
    slot_start_token_id,
)
from humanbrain.role_chunk_episode import RoleChunkEpisode, RoleChunkEpisodeEncoder, RolePayloadKind
from humanbrain.sdr_union import token_sdr_indices
from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.sandbox import load_worker_snapshot, python_runtime_identity, resolve_python_executable
from humanbrain.simple_swe.tasks import GeneratedTask, PublicTask, generate_visible_partition

from experiments.r3_pilot import public_simple_swe_role_chunk_feedback_action_probe as feedback_probe
from experiments.r3_pilot import public_simple_swe_role_chunk_feedback_parameterized_action_probe as pac1_probe
from experiments.r3_pilot import public_simple_swe_role_chunk_layer0_action_probe as action_probe
from experiments.r3_pilot import public_simple_swe_role_chunk_parameterized_action_probe as pac0_probe
from experiments.r3_pilot import public_simple_swe_role_chunk_route_probe as route_probe
from experiments.r3_pilot import public_train_dev_role_place_scalar_routing_probe as role_place_probe


OUTPUT_PATH = Path(__file__).with_suffix(".json")
SPEND_MARKER_PATH = Path(__file__).with_name(f"{Path(__file__).stem}.t1_list_map_add-009.spent.json")
PAC1_SOURCE_PATH = Path(__file__).with_name(
    "public_simple_swe_role_chunk_feedback_parameterized_action_probe.py"
)
PAC1_ARTIFACT_PATH = PAC1_SOURCE_PATH.with_suffix(".json")
HISTORICAL_BASELINE_PATH = REPO_ROOT / "artifacts" / "simple_swe_public_baseline_v1.json"
EXPECTED_HISTORICAL_BASELINE_SHA256 = pac0_probe.EXPECTED_HISTORICAL_BASELINE_SHA256
HISTORICAL_TARGET_TASK_SHA256 = "e3453e80ed535e5551d8d030c71a9dab20a2c1f00fb468494e335234a58d8be2"

SCALAR_FAMILY = "t0_scalar_offset"
MAP_FAMILY = "t1_list_map_add"
DONOR_ORDINALS = (5, 6)
TARGET_ORDINAL = 9
TARGET_TASK_ID = "simple-swe-v1/train/tier-1/t1_list_map_add/009"

TASK_RECORDS: dict[tuple[str, int], dict[str, object]] = {
    (SCALAR_FAMILY, 5): {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/005",
        "task_sha256": "0e6aab1811d3f356ea461681b16fd9fa3cea6ed017635a36c6a5bfa174e8329b",
        "hidden_test_sha256": "e2bcf8f5603cba4d9803cd620366598e61db15c7ad7504750dd03b51bc211229",
        "prompt_sha256": "0943ca4f7a05635155329a2fda88affdb791a1239107031e9b09247f499d7bf9",
        "entry_point": "solve_6acb574ebf911e92",
        "parameter": "n",
        "offset": -8,
    },
    (SCALAR_FAMILY, 6): {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/006",
        "task_sha256": "77d64375abe077ae005e519d6f7b19a3f122edd4766813f2f2bbc0ccc36af68f",
        "hidden_test_sha256": "3d24c71a350a25aa0adb5e0ee56b888539ce9cd49a0925c4f6d1739d62af6f13",
        "prompt_sha256": "f05f67071561279ca9ff9b89f07878f0acd1ea071027fcedffbefce4e27d2855",
        "entry_point": "solve_2c9984a52db14adc",
        "parameter": "n",
        "offset": 2,
    },
    (MAP_FAMILY, 5): {
        "task_id": "simple-swe-v1/train/tier-1/t1_list_map_add/005",
        "task_sha256": "6e7e11480f0edb7036062f0ef1e7aa60cf47968847a6dd0ab20ed90b748a647d",
        "hidden_test_sha256": "6beb7bafe5ce7eaa89bd4547ba5add88d71b21be0d9e6d1670d39d02a96ac45e",
        "prompt_sha256": "8f580169ee90871ddb161e74b0fee53de9999111f31214c484d0817b3901f29f",
        "entry_point": "solve_187d335b1d4ad45a",
        "parameter": "values",
        "offset": 1,
    },
    (MAP_FAMILY, 6): {
        "task_id": "simple-swe-v1/train/tier-1/t1_list_map_add/006",
        "task_sha256": "d84cb2c97461ae225283e486a3a213542fbd0f905cbce107fc1bc56fbafa1007",
        "hidden_test_sha256": "beef3d4baad1151774b7e65eac59fecd549309c3b727114674738f0854aaa9d2",
        "prompt_sha256": "7349ab488cd6638aae3e0cebdf0b75d7e93e82b5a7e177a4603727ab28af9a9c",
        "entry_point": "solve_3234660fc6ed5836",
        "parameter": "values",
        "offset": 5,
    },
    (MAP_FAMILY, TARGET_ORDINAL): {
        "task_id": TARGET_TASK_ID,
        "task_sha256": "65a1d6b824fbb2694512b1a56468b70de8dc1788aec0d4b896f255e2091b38a7",
        "hidden_test_sha256": "0211fe2294dbb6e67743e639a2d055e1f85376d14490bad53d2bbcf49b27711b",
        "prompt_sha256": "7c65cac042a0319df38b7e94fcc56d284338d0dcfdb5a508acfb2ec6c90f19fa",
        "entry_point": "solve_26ac026192446751",
        "parameter": "values",
        "offset": 3,
    },
}

TEST_TOKEN_ID = pac0_probe.TEST_TOKEN_ID
BUILD_ADD_TOKEN_ID = pac0_probe.BUILD_ADD_TOKEN_ID
BUILD_SUB_TOKEN_ID = pac0_probe.BUILD_SUB_TOKEN_ID
OBS_SCALAR_TOKEN_ID = 1608
OBS_LIST_TOKEN_ID = 1609
SHAPE_SCALAR_TOKEN_ID = route_probe.ROUTE_BY_FAMILY[SCALAR_FAMILY]
SHAPE_MAP_TOKEN_ID = route_probe.ROUTE_BY_FAMILY[MAP_FAMILY]
SHAPE_NAME_BY_TOKEN = {
    SHAPE_SCALAR_TOKEN_ID: "SCALAR",
    SHAPE_MAP_TOKEN_ID: "MAP",
}

ROUTER_COLUMN_COUNT = pac0_probe.ROUTER_COLUMN_COUNT
BUILDER_COLUMN_COUNT = pac0_probe.BUILDER_COLUMN_COUNT
ACTIVE_COLUMNS = pac0_probe.ACTIVE_COLUMNS
CODEBOOK_SEED = pac0_probe.CODEBOOK_SEED
BASE_CELLS_PER_COLUMN = pac0_probe.BASE_CELLS_PER_COLUMN
CONSTRUCTION_SEED = pac0_probe.CONSTRUCTION_SEED
THETA_MATCH = pac0_probe.THETA_MATCH
THETA_ACTIVE = pac0_probe.THETA_ACTIVE
EOS_TOKEN_ID = role_place_probe.EOS_TOKEN_ID
MAX_GENERATIONS = 128

SCALAR_STRUCTURAL_SHA256 = pac0_probe.STRUCTURAL_SHA256
SCALAR_CONCEPT_STREAM_SHA256 = pac0_probe.CONCEPT_STREAM_SHA256
SCALAR_CONCEPT_EVENTS = pac0_probe.CONCEPT_STREAM_EVENTS
MAP_STRUCTURAL_SHA256 = "c403a69416f3fc24c0655b56df7912952989d07c4f9f21e944aad023c419f769"
MAP_CONCEPT_STREAM_SHA256 = "aa48efdef4588d61145f49a39960df219377d1c2ada4110c49db4ff0dfcedf54"
MAP_CONCEPT_EVENTS = 66

EXPECTED_SHAPE_ROUTER_CAPACITY = {
    "training_sequence_lengths": [59, 59, 69, 69],
    "noninitial_arrivals": 252,
    "segment_capacity": 2_772,
    "max_pyramidal_cells": 35_540,
    "threshold_offset_cells": 83_368,
}
EXPECTED_BUILDER_CAPACITY = {
    "training_sequence_lengths": [60, 60, 88, 88],
    "noninitial_arrivals": 292,
    "segment_capacity": 3_212,
    "max_pyramidal_cells": 7_308,
    "threshold_offset_cells": 16_152,
}

LIST_VISIBLE_INPUTS: tuple[tuple[int, ...], ...] = ((0,), (-1, 1), (4, 4, -2))
SOURCE_PREFIX = b"def "
SOURCE_OPEN_PARAMETER = b"("
SOURCE_AFTER_PARAMETER = b"):\n    return "
SCALAR_SPACE = b" "
MAP_PREFIX = b"[item_0 "
MAP_BEFORE_ITERABLE = b" for item_0 in "
MAP_SUFFIX = b"]\n"
SOURCE_SUFFIX = b"\n"

EXPECTED_PAC1_SOURCE_SHA256 = "09c92d3a717fb548a7992af5ce21da309425741f3dbc2857b52ea56caee579ea"
EXPECTED_PAC1_ARTIFACT_SHA256 = "916765e4bf667f4f654fbfde24492c8f17944891aa2efc9b0e815feb04e677f6"

CLAIM_BOUNDARY = (
    "Public TRAIN prompt-plus-serialized-observation two-shape construction only; "
    "observation-only shape causality and cross-shape sign causality are not isolated; "
    "not arbitrary AST synthesis, transfer, DEV/eval, M0, or C1."
)
LIMITATIONS = {
    "host_factorization": (
        "Python reduces real visible execution into container-shape, sign, and canonical "
        "magnitude-byte observations before cells receive them."
    ),
    "shape_attribution": (
        "The normalized target prompt independently identifies MAP; OBS_LIST is "
        "checker-derived and physically advanced, but OBS_LIST-only causal shape "
        "selection is not shown."
    ),
    "operator_attribution": (
        "Both MAP donors and the MAP target have positive residuals; exact physical plus "
        "election is shown, but cross-shape OBS_POS-to-ADD causality is not isolated."
    ),
    "fixed_source_skeletons": [
        "def COPY0(COPY1): return COPY1 +/- COPY2",
        "def COPY0(COPY1): return [item_0 +/- COPY2 for item_0 in COPY1]",
    ],
    "fixed_map_binder": "item_0",
    "donor_reuse": (
        "scalar-offset and list-map-add /005 and /006 are reused public TRAIN donors; "
        "all PAC2 circuits are fresh"
    ),
    "historical_prompt_exposure": True,
    "freshness_claimed": False,
    "family_disjoint_transfer_claimed": False,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", type=Path, default=None)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _task_matches(task: GeneratedTask, expected: dict[str, object]) -> bool:
    public = task.public
    parameter = public.signature.parameters[0]
    return bool(
        public.partition == "train"
        and public.task_id == expected["task_id"]
        and public.task_sha256 == expected["task_sha256"]
        and public.hidden_test_sha256 == expected["hidden_test_sha256"]
        and hashlib.sha256(public.prompt.encode()).hexdigest() == expected["prompt_sha256"]
        and public.signature.entry_point == expected["entry_point"]
        and parameter.name == expected["parameter"]
    )


def _select_training() -> tuple[object, tuple[GeneratedTask, ...], tuple[GeneratedTask, ...]]:
    protocol, fit_donors, _spent_route_probes = route_probe._select_tasks()
    visible = generate_visible_partition(protocol, "train")
    indexed = {(task.public.family, task.public.ordinal): task for task in visible}
    training = tuple(
        indexed[(family, ordinal)]
        for family in (SCALAR_FAMILY, MAP_FAMILY)
        for ordinal in DONOR_ORDINALS
    )
    for task in training:
        expected = TASK_RECORDS[(task.public.family, task.public.ordinal)]
        if not _task_matches(task, expected):
            raise RuntimeError(f"public donor changed: {task.public.task_id}")
        parameter = task.public.signature.parameters[0]
        if task.public.family == SCALAR_FAMILY:
            valid_type = parameter.type.constructor == "Int" and task.public.signature.result.constructor == "Int"
        else:
            valid_type = (
                parameter.type.constructor == "List"
                and parameter.type.arguments[0].constructor == "Int"
                and task.public.signature.result.constructor == "List"
                and task.public.signature.result.arguments[0].constructor == "Int"
            )
        if not valid_type:
            raise RuntimeError("donor callable type changed")
    return protocol, fit_donors, training


def _select_target(protocol: object) -> GeneratedTask:
    visible = generate_visible_partition(protocol, "train")
    target = next(
        task
        for task in visible
        if task.public.family == MAP_FAMILY and task.public.ordinal == TARGET_ORDINAL
    )
    if not _task_matches(target, TASK_RECORDS[(MAP_FAMILY, TARGET_ORDINAL)]):
        raise RuntimeError("public R3-PAC2 target changed")
    return target


def _assert_episode(episode: RoleChunkEpisode, task: GeneratedTask) -> None:
    if episode.task_id != task.public.task_id or episode.restore_prompt() != task.public.prompt:
        raise RuntimeError("role episode does not restore the public prompt")
    if task.public.family == SCALAR_FAMILY:
        expected = (SCALAR_CONCEPT_EVENTS, SCALAR_STRUCTURAL_SHA256, SCALAR_CONCEPT_STREAM_SHA256)
    elif task.public.family == MAP_FAMILY:
        expected = (MAP_CONCEPT_EVENTS, MAP_STRUCTURAL_SHA256, MAP_CONCEPT_STREAM_SHA256)
    else:
        raise RuntimeError("unsupported PAC2 family")
    if (
        len(episode.token_ids) != expected[0]
        or episode.structural_sha256 != expected[1]
        or _sha256_json(tuple(episode.token_ids)) != expected[2]
    ):
        raise RuntimeError("donor differs from its frozen normalized template")


def _public_offset(episode: RoleChunkEpisode, task: GeneratedTask) -> int:
    if episode.task_id != task.public.task_id or episode.restore_prompt() != task.public.prompt:
        raise RuntimeError("environment episode/task mismatch")
    payloads = episode.payloads_for(RolePayloadKind.INTEGER_LITERAL)
    if len(payloads) != 1:
        raise RuntimeError("offset task requires one public integer literal")
    try:
        offset = int(payloads[0].value)
    except ValueError as error:
        raise RuntimeError("public offset is not an integer") from error
    expected = TASK_RECORDS[(task.public.family, task.public.ordinal)]["offset"]
    if str(offset) != payloads[0].value or offset != expected:
        raise RuntimeError("public offset changed")
    return offset


def _list_identity_source(public: PublicTask) -> str:
    parameter = public.signature.parameters[0]
    if parameter.name != "values":
        raise RuntimeError("list identity is frozen to parameter values")
    return f"def {public.signature.entry_point}(values):\n    return values\n"


def _visible_list_checker(source: str, entrypoint: str, offset: int) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    namespace: dict[str, object] = {}
    error_name: str | None = None
    try:
        code = compile(ast.parse(source, filename="<r3-pac2-visible>", mode="exec"), "<r3-pac2-visible>", "exec")
        exec(code, {"__builtins__": {}}, namespace)
        function = namespace[entrypoint]
    except (SyntaxError, ValueError, TypeError, NameError, KeyError) as error:
        function = None
        error_name = type(error).__name__
    passed = callable(function)
    residual: list[int] = []
    for raw in LIST_VISIBLE_INPUTS:
        values = list(raw)
        wanted = [value + offset for value in values]
        try:
            observed = function(list(values)) if callable(function) else None
        except Exception as error:  # noqa: BLE001 - diagnostic taxonomy is recorded
            observed = None
            case_error: str | None = type(error).__name__
        else:
            case_error = None
        exact = type(observed) is list and observed == wanted
        passed &= exact
        if type(observed) is list and len(observed) == len(wanted) and all(type(v) is int for v in observed):
            residual.extend(want - got for want, got in zip(wanted, observed, strict=True))
        cases.append({"input": values, "expected": wanted, "observed": observed, "error": case_error, "passed": exact})
    return {
        "source": source,
        "entrypoint": entrypoint,
        "offset": offset,
        "cases": cases,
        "residual": residual,
        "error": error_name,
        "passed": bool(passed),
        "container_shape": "list_int_to_list_int",
    }


def _run_visible_list_identity_environment(episode: RoleChunkEpisode, task: GeneratedTask) -> dict[str, object]:
    return _visible_list_checker(
        _list_identity_source(task.public),
        task.public.signature.entry_point,
        _public_offset(episode, task),
    )


def _run_visible_scalar_identity_environment(
    episode: RoleChunkEpisode, task: GeneratedTask
) -> dict[str, object]:
    source = pac1_probe._identity_source(task.public)
    return pac1_probe.signed_probe._visible_checker(
        source,
        task.public.signature.entry_point,
        _public_offset(episode, task),
    )


def _uniform_list_signed_residual(checker: dict[str, object]) -> int:
    residual = checker.get("residual")
    cases = checker.get("cases")
    if (
        checker.get("container_shape") != "list_int_to_list_int"
        or checker.get("error") is not None
        or checker.get("passed") is not False
        or not isinstance(cases, list)
        or len(cases) != len(LIST_VISIBLE_INPUTS)
        or not isinstance(residual, list)
        or len(residual) != sum(len(values) for values in LIST_VISIBLE_INPUTS)
        or any(type(value) is not int for value in residual)
        or len(set(residual)) != 1
        or residual[0] == 0
    ):
        raise RuntimeError("mixed, zero, non-integer, or malformed list residual")
    return int(residual[0])


def _list_sign_observation(checker: dict[str, object]) -> int:
    return pac0_probe.OBS_POS_TOKEN_ID if _uniform_list_signed_residual(checker) > 0 else pac0_probe.OBS_NEG_TOKEN_ID


def _list_feedback_slot_block(checker: dict[str, object]) -> tuple[int, ...]:
    magnitude = str(abs(_uniform_list_signed_residual(checker))).encode("ascii")
    if not magnitude.isdigit() or magnitude.startswith(b"0"):
        raise RuntimeError("list feedback magnitude is not canonical positive decimal")
    return (slot_start_token_id(2), *magnitude, slot_end_token_id(2))


def _parameter_slot_block(parameter: str) -> tuple[int, ...]:
    value = parameter.encode("ascii")
    if value not in {b"n", b"values"}:
        raise RuntimeError("PAC2 parameter slot is outside the two frozen shapes")
    return (slot_start_token_id(1), *value, slot_end_token_id(1))


def _shape_observation_from_checker(checker: dict[str, object]) -> int:
    if checker.get("container_shape") == "list_int_to_list_int":
        _uniform_list_signed_residual(checker)
        return OBS_LIST_TOKEN_ID
    pac1_probe._uniform_signed_residual(checker)
    return OBS_SCALAR_TOKEN_ID


def _shape_tokens_from_checker(checker: dict[str, object]) -> tuple[int, int]:
    observation = _shape_observation_from_checker(checker)
    shape = (
        SHAPE_MAP_TOKEN_ID
        if observation == OBS_LIST_TOKEN_ID
        else SHAPE_SCALAR_TOKEN_ID
    )
    return observation, shape


def _feedback_for_donor(episode: RoleChunkEpisode, task: GeneratedTask) -> dict[str, object]:
    if task.public.family == SCALAR_FAMILY:
        return _run_visible_scalar_identity_environment(episode, task)
    return _run_visible_list_identity_environment(episode, task)


def _feedback_slot_from_checker(checker: dict[str, object]) -> tuple[int, ...]:
    if _shape_observation_from_checker(checker) == OBS_LIST_TOKEN_ID:
        return _list_feedback_slot_block(checker)
    return pac1_probe._feedback_slot_block(checker)


def _sign_observation_from_checker(checker: dict[str, object]) -> int:
    if _shape_observation_from_checker(checker) == OBS_LIST_TOKEN_ID:
        return _list_sign_observation(checker)
    return pac1_probe._sign_observation(checker)


def _teacher_build_label_for_checker(checker: dict[str, object]) -> int:
    """Return a donor teacher label; target builder input never uses this mapping."""
    observation = _sign_observation_from_checker(checker)
    return BUILD_ADD_TOKEN_ID if observation == pac0_probe.OBS_POS_TOKEN_ID else BUILD_SUB_TOKEN_ID


def _shape_router_sequences(
    training: tuple[GeneratedTask, ...],
    episodes: dict[tuple[str, int], RoleChunkEpisode],
    feedback: dict[tuple[str, int], dict[str, object]],
) -> tuple[tuple[tuple[int, ...], ...], list[dict[str, object]]]:
    sequences: list[tuple[int, ...]] = []
    records: list[dict[str, object]] = []
    for task in training:
        key = (task.public.family, task.public.ordinal)
        observation, shape = _shape_tokens_from_checker(feedback[key])
        sequence = (*episodes[key].token_ids, TEST_TOKEN_ID, observation, shape)
        sequences.append(sequence)
        records.append({
            "task_id": task.public.task_id,
            "shape_observation_token_id": observation,
            "teacher_shape_token_id": shape,
            "sequence_token_ids": list(sequence),
        })
    return tuple(sequences), records


def _definition(
    sequences: tuple[tuple[int, ...], ...], *, column_count: int, expected: dict[str, object]
) -> tuple[NativeTemporalCircuitDefinition, dict[str, object]]:
    lengths = [len(sequence) for sequence in sequences]
    noninitial_arrivals = sum(length - 1 for length in lengths)
    segment_capacity = ACTIVE_COLUMNS * noninitial_arrivals
    config = NativeTemporalCircuitConfig(
        column_count=column_count,
        max_pyramidal_cells=column_count * BASE_CELLS_PER_COLUMN + segment_capacity,
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
    used = tuple(sorted({*range(128), EOS_TOKEN_ID, *role_tokens, *(token for sequence in sequences for token in sequence)}))
    token_codes = {
        token: tuple(int(column) for column in token_sdr_indices(token, n_bits=column_count, w_ones=ACTIVE_COLUMNS, seed=CODEBOOK_SEED))
        for token in used
    }
    threshold_offset_cells = 2 * config.max_pyramidal_cells + 3 * config.column_count
    definition = NativeTemporalCircuitDefinition(
        config,
        token_codes,
        seeded_threshold_offsets_mv(threshold_offset_cells, config.construction_seed).tolist(),
    )
    capacity = {
        "training_sequence_lengths": lengths,
        "noninitial_arrivals": noninitial_arrivals,
        "segment_capacity": segment_capacity,
        "max_pyramidal_cells": config.max_pyramidal_cells,
        "threshold_offset_cells": threshold_offset_cells,
    }
    if capacity != expected:
        raise RuntimeError("PAC2 capacity differs from exact sequence arithmetic")
    return definition, capacity


def _builder_teacher_target(shape_token: int, build_token: int) -> tuple[int, ...]:
    if shape_token not in SHAPE_NAME_BY_TOKEN:
        raise RuntimeError("builder shape is not physical")
    build_token = pac1_probe._require_build_token(build_token)
    operator = ord("+") if build_token == BUILD_ADD_TOKEN_ID else ord("-")
    common = (
        *SOURCE_PREFIX,
        copy_slot_token_id(0),
        *SOURCE_OPEN_PARAMETER,
        copy_slot_token_id(1),
        *SOURCE_AFTER_PARAMETER,
    )
    if shape_token == SHAPE_SCALAR_TOKEN_ID:
        return (
            *common,
            copy_slot_token_id(1),
            *SCALAR_SPACE,
            operator,
            *SCALAR_SPACE,
            copy_slot_token_id(2),
            *SOURCE_SUFFIX,
            EOS_TOKEN_ID,
        )
    return (
        *common,
        *MAP_PREFIX,
        operator,
        *SCALAR_SPACE,
        copy_slot_token_id(2),
        *MAP_BEFORE_ITERABLE,
        copy_slot_token_id(1),
        *MAP_SUFFIX,
        EOS_TOKEN_ID,
    )


def _builder_train_episode(task: GeneratedTask, checker: dict[str, object]) -> tuple[int, ...]:
    _shape_observation, shape = _shape_tokens_from_checker(checker)
    sign_observation = _sign_observation_from_checker(checker)
    build = _teacher_build_label_for_checker(checker)
    parameter = task.public.signature.parameters[0].name
    return (
        *pac1_probe._entrypoint_slot_block(task.public.signature.entry_point),
        *_parameter_slot_block(parameter),
        *_feedback_slot_from_checker(checker),
        shape,
        sign_observation,
        *_builder_teacher_target(shape, build),
    )


def _prepare_donor_material() -> dict[str, object]:
    protocol, fit_donors, training = _select_training()
    encoder: RoleChunkEpisodeEncoder = pac0_probe._fit_role_chunk_encoder(fit_donors)
    episodes = {
        (task.public.family, task.public.ordinal): encoder.encode(task.public)
        for task in training
    }
    for task in training:
        _assert_episode(episodes[(task.public.family, task.public.ordinal)], task)
    feedback = {
        (task.public.family, task.public.ordinal): _feedback_for_donor(
            episodes[(task.public.family, task.public.ordinal)], task
        )
        for task in training
    }
    router_sequences, router_records = _shape_router_sequences(
        training, episodes, feedback
    )
    router_definition, router_capacity = _definition(
        router_sequences,
        column_count=ROUTER_COLUMN_COUNT,
        expected=EXPECTED_SHAPE_ROUTER_CAPACITY,
    )
    builder_sequences = tuple(
        _builder_train_episode(task, feedback[(task.public.family, task.public.ordinal)])
        for task in training
    )
    builder_definition, builder_capacity = _definition(
        builder_sequences,
        column_count=BUILDER_COLUMN_COUNT,
        expected=EXPECTED_BUILDER_CAPACITY,
    )
    return {
        "protocol": protocol,
        "training": training,
        "encoder": encoder,
        "episodes": episodes,
        "feedback": feedback,
        "router_sequences": router_sequences,
        "router_records": router_records,
        "router_definition": router_definition,
        "router_capacity": router_capacity,
        "builder_sequences": builder_sequences,
        "builder_definition": builder_definition,
        "builder_capacity": builder_capacity,
    }


def _train_material(prepared: dict[str, object]) -> dict[str, object]:
    router_checkpoint, router_identity, router_steps = pac0_probe._train_circuit(
        prepared["router_definition"], prepared["router_sequences"]
    )
    builder_checkpoint, builder_identity, builder_steps = pac0_probe._train_circuit(
        prepared["builder_definition"], prepared["builder_sequences"]
    )
    return {
        **prepared,
        "router_checkpoint": router_checkpoint,
        "router_checkpoint_identity": router_identity,
        "router_steps": router_steps,
        "builder_checkpoint": builder_checkpoint,
        "builder_checkpoint_identity": builder_identity,
        "builder_steps": builder_steps,
    }


def _assert_upstream_pass() -> dict[str, object]:
    if not EXPECTED_PAC1_SOURCE_SHA256 or not EXPECTED_PAC1_ARTIFACT_SHA256:
        raise RuntimeError("R3-PAC2 remains held until committed passing PAC1 pins are frozen")
    if _sha256_file(PAC1_SOURCE_PATH) != EXPECTED_PAC1_SOURCE_SHA256:
        raise RuntimeError("PAC1 source differs from the frozen upstream pin")
    pac1_probe._assert_frozen_sources()
    if not PAC1_ARTIFACT_PATH.is_file() or _sha256_file(PAC1_ARTIFACT_PATH) != EXPECTED_PAC1_ARTIFACT_SHA256:
        raise RuntimeError("PAC1 passing artifact is absent or changed")
    payload = json.loads(PAC1_ARTIFACT_PATH.read_text(encoding="utf-8"))
    gates = payload.get("gates")
    if not str(payload.get("outcome", "")).startswith("PASS_R3_PAC1") or not isinstance(gates, dict) or not all(gates.values()):
        raise RuntimeError("PAC1 did not record a complete pass")
    return {"source_sha256": EXPECTED_PAC1_SOURCE_SHA256, "artifact_sha256": EXPECTED_PAC1_ARTIFACT_SHA256, "outcome": payload["outcome"]}


def _assert_source_committed_clean() -> None:
    relative = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    committed = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{relative}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", relative],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if committed.returncode != 0 or status.stdout.strip():
        raise RuntimeError("commit the frozen clean R3-PAC2 source before target spend")


def _assert_history_anchor() -> dict[str, object]:
    if _sha256_file(HISTORICAL_BASELINE_PATH) != EXPECTED_HISTORICAL_BASELINE_SHA256:
        raise RuntimeError("historical public prompt baseline changed")
    payload = json.loads(HISTORICAL_BASELINE_PATH.read_text(encoding="utf-8"))
    records = pac0_probe._nested_task_records(payload, TARGET_TASK_ID)
    if (
        len(records) != 2
        or {record.get("task_sha256") for record in records}
        != {HISTORICAL_TARGET_TASK_SHA256}
        or {record.get("prompt_sha256") for record in records}
        != {TASK_RECORDS[(MAP_FAMILY, TARGET_ORDINAL)]["prompt_sha256"]}
        or any(record.get("passed") is not False for record in records)
    ):
        raise RuntimeError("historical PAC2 target disclosure changed")
    return {
        "historical_prompt_baseline_sha256": EXPECTED_HISTORICAL_BASELINE_SHA256,
        "historical_target_occurrences": 2,
        "historical_target_task_sha256": HISTORICAL_TARGET_TASK_SHA256,
        "current_target_task_sha256": TASK_RECORDS[(MAP_FAMILY, TARGET_ORDINAL)]["task_sha256"],
        "historical_task_content_drift": True,
    }


def _contains_task_id(value: object, task_id: str) -> bool:
    if isinstance(value, str):
        return value == task_id
    if isinstance(value, dict):
        return value.get("target_task_id") == task_id or value.get("task_id") == task_id or any(
            _contains_task_id(item, task_id) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_task_id(item, task_id) for item in value)
    return False


def _assert_target_unspent() -> list[str]:
    if OUTPUT_PATH.exists() or SPEND_MARKER_PATH.exists():
        raise RuntimeError("R3-PAC2 target is already spent")
    prior: list[str] = []
    for path in sorted(PROBE_DIR.glob("*.json")):
        if path in {OUTPUT_PATH, HISTORICAL_BASELINE_PATH}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if _contains_task_id(payload, TARGET_TASK_ID):
            prior.append(path.name)
    if prior:
        raise RuntimeError(f"R3-PAC2 target appears in prior experiment artifacts: {prior}")
    return prior


def _expected_source_bytes(
    entrypoint: str,
    parameter: str,
    magnitude: bytes,
    shape_token: int,
    build_token: int,
) -> bytes:
    if shape_token not in SHAPE_NAME_BY_TOKEN:
        raise RuntimeError("unknown source shape")
    operator = b"+" if pac1_probe._require_build_token(build_token) == BUILD_ADD_TOKEN_ID else b"-"
    common = SOURCE_PREFIX + entrypoint.encode("ascii") + SOURCE_OPEN_PARAMETER + parameter.encode("ascii") + SOURCE_AFTER_PARAMETER
    if shape_token == SHAPE_SCALAR_TOKEN_ID:
        return common + parameter.encode("ascii") + b" " + operator + b" " + magnitude + SOURCE_SUFFIX
    return common + MAP_PREFIX + operator + b" " + magnitude + MAP_BEFORE_ITERABLE + parameter.encode("ascii") + MAP_SUFFIX


def _assert_generation_budget() -> dict[str, int]:
    # One decimal byte is the longest magnitude in the frozen PAC2 cohort.
    longest = _expected_source_bytes(
        str(TASK_RECORDS[(MAP_FAMILY, TARGET_ORDINAL)]["entry_point"]),
        str(TASK_RECORDS[(MAP_FAMILY, TARGET_ORDINAL)]["parameter"]),
        b"9",
        SHAPE_MAP_TOKEN_ID,
        BUILD_ADD_TOKEN_ID,
    )
    required_calls = len(longest) + 1  # ordinary EOS
    if len(longest) != 81 or required_calls > MAX_GENERATIONS:
        raise RuntimeError("PAC2 source no longer fits the frozen direct-generation budget")
    return {
        "longest_expected_source_bytes": len(longest),
        "required_calls_including_eos": required_calls,
        "generation_budget": MAX_GENERATIONS,
    }


def _copy_transactions(
    generations: Iterable[dict[str, object]], expected_slots: list[int]
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
            active = {
                "slot": generation["slot_ordinal"],
                "control_token_id": generation["control_token_id"],
                "enter_event_count": event_count,
                "copied_places": [generation["copy_place_ordinal"]],
                "copied_event_counts": [event_count],
                "control_advanced_temporal_once": previous_event_count is not None and event_count == previous_event_count + 1,
            }
        elif path == TemporalGenerationPath.COPY.name:
            exact &= active is not None
            if active is not None:
                exact &= generation["slot_ordinal"] == active["slot"]
                active["copied_places"].append(generation["copy_place_ordinal"])
                active["copied_event_counts"].append(event_count)
        elif path == TemporalGenerationPath.EXIT_COPY.name:
            exact &= active is not None
            if active is not None:
                exact &= generation["slot_ordinal"] == active["slot"]
                enter_count = int(active["enter_event_count"])
                active["temporal_event_count_held_across_copied_bytes"] = all(
                    int(value) == enter_count for value in active["copied_event_counts"]
                )
                active["exit_resumed_temporal_once"] = event_count == enter_count + 1
                transactions.append(active)
                active = None
        previous_event_count = event_count
    exact &= active is None
    exact &= [record["slot"] for record in transactions] == expected_slots
    exact &= [record["control_token_id"] for record in transactions] == [copy_slot_token_id(slot) for slot in expected_slots]
    exact &= all(
        bool(record["control_advanced_temporal_once"])
        and bool(record["temporal_event_count_held_across_copied_bytes"])
        and bool(record["exit_resumed_temporal_once"])
        for record in transactions
    )
    return transactions, bool(exact)


def _analyze_builder_run(
    *,
    entrypoint: str,
    parameter: str,
    magnitude: bytes,
    shape_token: int,
    build_token: int,
    run: dict[str, object],
) -> dict[str, object]:
    emitted = bytes(run["emitted"])
    expected = _expected_source_bytes(entrypoint, parameter, magnitude, shape_token, build_token)
    source: str | None = None
    compile_valid = False
    namespace: dict[str, object] = {}
    try:
        source = emitted.decode("ascii", errors="strict")
        code = compile(ast.parse(source, filename="<r3-pac2-builder>", mode="exec"), "<r3-pac2-builder>", "exec")
        exec(code, {"__builtins__": {}}, namespace)
        compile_valid = True
    except (UnicodeDecodeError, SyntaxError, ValueError, TypeError, NameError):
        pass
    operator = ord("+") if build_token == BUILD_ADD_TOKEN_ID else ord("-")
    operator_records = [
        generation
        for generation in run["generations"]
        if generation["path"] == TemporalGenerationPath.TEMPORAL.name and generation["surface_token_id"] == operator
    ]
    operator_physical = bool(len(operator_records) == 1 and operator_records[0]["physical_election_count"] == 1 and operator_records[0]["winner_input_columns"] == ACTIVE_COLUMNS)
    expected_slots = [0, 1, 1, 2] if shape_token == SHAPE_SCALAR_TOKEN_ID else [0, 1, 2, 1]
    transactions, copy_exact = _copy_transactions(run["generations"], expected_slots)
    expected_place_lengths = [len(entrypoint), len(parameter), len(parameter), len(magnitude)] if shape_token == SHAPE_SCALAR_TOKEN_ID else [len(entrypoint), len(parameter), len(magnitude), len(parameter)]
    copy_exact &= [record["copied_places"] for record in transactions] == [list(range(length)) for length in expected_place_lengths]
    final_eos = bool(run["generations"] and run["generations"][-1]["surface_token_id"] == EOS_TOKEN_ID and run["generations"][-1]["path"] == TemporalGenerationPath.TEMPORAL.name)
    signed_offset = int(magnitude.decode("ascii")) * (1 if build_token == BUILD_ADD_TOKEN_ID else -1)
    function = namespace.get(entrypoint)
    constructed_semantic_exact = callable(function)
    if callable(function):
        domain: Iterable[object] = (-3, 0, 4) if shape_token == SHAPE_SCALAR_TOKEN_ID else (list(values) for values in LIST_VISIBLE_INPUTS)
        for value in domain:
            wanted = value + signed_offset if type(value) is int else [item + signed_offset for item in value]
            try:
                observed = function(value)
            except Exception:  # noqa: BLE001 - summarized below
                constructed_semantic_exact = False
                break
            constructed_semantic_exact &= type(observed) is type(wanted) and observed == wanted
    passed = bool(
        run["stop"] is None
        and run["terminated_by_eos"] is True
        and final_eos
        and emitted == expected
        and compile_valid
        and copy_exact
        and operator_physical
        and constructed_semantic_exact
    )
    return {
        "passed": passed,
        "source": source,
        "exact_expected_source": emitted == expected,
        "compile_valid": compile_valid,
        "shape": SHAPE_NAME_BY_TOKEN[shape_token],
        "signed_offset": signed_offset,
        "copy_slots": [record["slot"] for record in transactions],
        "copy_transactions_exact": bool(copy_exact),
        "copy_transactions": transactions,
        "operator_elected_physically": operator_physical,
        "ordinary_eos": final_eos,
        "constructed_semantic_exact": bool(constructed_semantic_exact),
        "emitted_sha256": hashlib.sha256(emitted).hexdigest(),
        "expected_sha256": hashlib.sha256(expected).hexdigest(),
    }


def _builder_prompt(
    entrypoint: str,
    parameter: str,
    feedback_slot: tuple[int, ...],
    shape_token: int,
    sign_observation_token: int,
) -> tuple[int, ...]:
    if shape_token not in SHAPE_NAME_BY_TOKEN:
        raise RuntimeError("builder prompt shape is not physical")
    if sign_observation_token not in {
        pac0_probe.OBS_POS_TOKEN_ID,
        pac0_probe.OBS_NEG_TOKEN_ID,
    }:
        raise RuntimeError("builder prompt sign cue is not raw checker feedback")
    return (
        *pac1_probe._entrypoint_slot_block(entrypoint),
        *_parameter_slot_block(parameter),
        *feedback_slot,
        shape_token,
        sign_observation_token,
    )


def _observed_surface_token(raw: dict[str, object]) -> int | None:
    generation = raw.get("generation")
    return None if not isinstance(generation, dict) else int(generation["surface_token_id"])


def _generate_teacher_free(circuit: NativeTemporalCircuit) -> dict[str, object]:
    """Generate direct source with the PAC2 map-shape budget, preserving PAC0 semantics."""
    emitted = bytearray()
    generations: list[dict[str, object]] = []
    stop: dict[str, object] | None = None
    terminated_by_eos = False
    for position in range(MAX_GENERATIONS):
        try:
            generation = circuit.generate()
        except NativeTemporalRuntimeError as error:
            stop = role_place_probe._native_stop(error, position=position)
            break
        generations.append(role_place_probe._generation_record(position, generation))
        if generation.surface_token_id == EOS_TOKEN_ID:
            terminated_by_eos = True
            break
        if not 0 <= generation.surface_token_id < 128:
            stop = {
                "kind": "NON_ASCII_SURFACE_TOKEN",
                "generation_position": position,
                "observed": int(generation.surface_token_id),
            }
            break
        emitted.append(generation.surface_token_id)
    if not terminated_by_eos and stop is None:
        stop = {
            "kind": "GENERATION_BUDGET_EXHAUSTED",
            "generation_budget": MAX_GENERATIONS,
        }
    return {
        "emitted": bytes(emitted),
        "generations": generations,
        "terminated_by_eos": terminated_by_eos,
        "stop": stop,
    }


def _create_spend_marker(target: GeneratedTask, router_sha256: str, builder_sha256: str) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-shape-action",
        "state": "TARGET_ATTEMPT_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target_task_id": target.public.task_id,
        "router_checkpoint_sha256": router_sha256,
        "builder_checkpoint_sha256": builder_sha256,
        "router_target_no_argument_generation_calls_planned": 2,
        "builder_target_runs_planned": 3,
        "builder_target_no_argument_generation_calls_maximum": 3 * MAX_GENERATIONS,
        "canonical_target_sandbox_calls_planned": 1,
        "model_retries": 0,
        "harness_retries": 0,
        "probe_source_sha256": _sha256_file(Path(__file__)),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with SPEND_MARKER_PATH.open("x", encoding="utf-8", newline="\n") as marker:
            marker.write(serialized)
            marker.flush()
            os.fsync(marker.fileno())
    except FileExistsError as error:
        raise RuntimeError("refusing already spent R3-PAC2 target") from error
    return {"path": SPEND_MARKER_PATH.name, "sha256": _sha256_file(SPEND_MARKER_PATH), "exclusive_create": True}


def _router_prefix_gates(
    *,
    actual_first: dict[str, object],
    identity_checker: dict[str, object],
    actual_feedback: dict[str, object],
    actual_second: dict[str, object],
) -> dict[str, bool]:
    derived_shape_observation = _shape_observation_from_checker(identity_checker)
    return {
        "router_physically_selected_test": bool(
            route_probe._score_generation(actual_first, TEST_TOKEN_ID)["correct"]
        ),
        "list_feedback_advanced_once": bool(
            derived_shape_observation == OBS_LIST_TOKEN_ID
            and actual_feedback.get("advanced") is True
            and actual_feedback.get("observation_token_id") == derived_shape_observation
            and actual_feedback.get("native_stop") is None
        ),
        "target_prompt_plus_list_observation_selected_map_shape": bool(
            route_probe._score_generation(actual_second, SHAPE_MAP_TOKEN_ID)["correct"]
        ),
    }


def _model_stop_gates(
    *,
    actual_first: dict[str, object],
    identity_checker: dict[str, object],
    actual_feedback: dict[str, object],
    actual_second: dict[str, object],
) -> dict[str, bool]:
    return {
        **_router_prefix_gates(
            actual_first=actual_first,
            identity_checker=identity_checker,
            actual_feedback=actual_feedback,
            actual_second=actual_second,
        ),
        "actual_map_builder_exact": False,
        "builder_shape_intervention_scalar_exact": False,
        "payload_control_map_builder_exact": False,
        "actual_visible_checker_passed": False,
        "both_single_factor_controls_failed": False,
        "canonical_target_train_sandbox_all_36": False,
    }


def _sandbox_not_attempted(reason: str) -> dict[str, object]:
    return {
        "passed": False,
        "sandboxed": False,
        "outcome": "not_attempted",
        "reason": reason,
        "regression_case_count": 0,
        "property_case_count": 0,
    }


def _runtime_record(started_ns: int) -> dict[str, object]:
    native_identity = loaded_library_identity()
    return {
        "elapsed_wall_ns": time.perf_counter_ns() - started_ns,
        "python": sys.version,
        "platform": platform.platform(),
        "native_source_digest": native_identity.source_digest,
        "native_binary_sha256": native_identity.binary_sha256,
        "probe_source_sha256": _sha256_file(Path(__file__)),
    }


def _write_output(payload: dict[str, object]) -> None:
    with OUTPUT_PATH.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _terminal_router_stop_payload(
    *,
    upstream: dict[str, object],
    history_anchor: dict[str, object],
    generation_budget: dict[str, int],
    prior_inventory: list[str],
    prepared: dict[str, object],
    target: GeneratedTask,
    marker: dict[str, object],
    actual_prompt: object,
    actual_first: dict[str, object],
    identity_checker: dict[str, object],
    actual_feedback: dict[str, object],
    actual_second: dict[str, object],
    actual_shape: int | None,
    started_ns: int,
) -> dict[str, object]:
    reason = "shape router emitted no recognized physical shape token"
    gates = _model_stop_gates(
        actual_first=actual_first,
        identity_checker=identity_checker,
        actual_feedback=actual_feedback,
        actual_second=actual_second,
    )
    return {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-shape-action",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": "STOP_R3_PAC2_TWO_SHAPE_LIST_MAP_CONSTRUCTION",
        "terminal_model_failure": reason,
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": LIMITATIONS,
        "upstream_pac1": upstream,
        "history_anchor": history_anchor,
        "generation_budget": generation_budget,
        "authority": {
            "partition": "train",
            "target_task_id": target.public.task_id,
            "historical_prompt_baseline_exposure": True,
            "fresh_routed_action_target": True,
            "prior_target_artifact_inventory": prior_inventory,
            "model_retries": 0,
            "harness_retries": 0,
            "expected_routes_and_sources_scored_after_all_completed_target_calls": True,
            "builder_sign_input": "not reached",
            "python_selected_target_operator_or_source": False,
            "hidden_results_entered_cell_runtime": False,
        },
        "training": {
            "task_ids": [task.public.task_id for task in prepared["training"]],
            "router_capacity": prepared["router_capacity"],
            "builder_capacity": prepared["builder_capacity"],
            "router_checkpoint_sha256": hashlib.sha256(prepared["router_checkpoint"]).hexdigest(),
            "builder_checkpoint_sha256": hashlib.sha256(prepared["builder_checkpoint"]).hexdigest(),
        },
        "spend_marker": marker,
        "router_target": {
            "actual_prompt": route_probe._step_summary(actual_prompt),
            "actual_first": actual_first,
            "identity_checker": identity_checker,
            "actual_feedback": actual_feedback,
            "actual_second": actual_second,
            "actual_shape_token_id": actual_shape,
        },
        "builder_target": {
            "not_attempted_reason": reason,
            "prompt_steps": [],
            "analyses": [],
        },
        "visible_target_checkers": [],
        "sandbox": _sandbox_not_attempted(reason),
        "gates": gates,
        "runtime": _runtime_record(started_ns),
    }


def _score_posthoc(
    *,
    actual_first: dict[str, object],
    identity_checker: dict[str, object],
    actual_feedback: dict[str, object],
    actual_second: dict[str, object],
    actual_analysis: dict[str, object],
    shape_analysis: dict[str, object],
    payload_analysis: dict[str, object],
    actual_checker: dict[str, object],
    shape_checker: dict[str, object],
    payload_checker: dict[str, object],
    sandbox: dict[str, object],
) -> dict[str, bool]:
    return {
        **_router_prefix_gates(
            actual_first=actual_first,
            identity_checker=identity_checker,
            actual_feedback=actual_feedback,
            actual_second=actual_second,
        ),
        "actual_map_builder_exact": actual_analysis.get("passed") is True and actual_analysis.get("copy_slots") == [0, 1, 2, 1],
        "builder_shape_intervention_scalar_exact": shape_analysis.get("passed") is True and shape_analysis.get("copy_slots") == [0, 1, 1, 2],
        "payload_control_map_builder_exact": payload_analysis.get("passed") is True and payload_analysis.get("copy_slots") == [0, 1, 2, 1],
        "actual_visible_checker_passed": actual_checker.get("passed") is True,
        "both_single_factor_controls_failed": shape_checker.get("passed") is False and payload_checker.get("passed") is False,
        "canonical_target_train_sandbox_all_36": bool(
            sandbox.get("passed") is True
            and sandbox.get("sandboxed") is True
            and sandbox.get("outcome") == "passed"
            and int(sandbox.get("regression_case_count", 0)) + int(sandbox.get("property_case_count", 0)) == 36
        ),
    }


def main() -> int:
    # The upstream gate is deliberately first: while PAC1 is uncommitted or has
    # no passing artifact, PAC2 cannot even prepare/encode its target.
    upstream = _assert_upstream_pass()
    _assert_source_committed_clean()
    history_anchor = _assert_history_anchor()
    prior_inventory = _assert_target_unspent()
    generation_budget = _assert_generation_budget()
    args = _parse_args()
    started_ns = time.perf_counter_ns()
    prepared = _train_material(_prepare_donor_material())
    protocol = prepared["protocol"]
    target = _select_target(protocol)
    target_episode = prepared["encoder"].encode(target.public)
    _assert_episode(target_episode, target)

    executable = resolve_python_executable(args.python_executable)
    runtime_identity = python_runtime_identity(executable, protocol.raw.get("language"))
    if runtime_identity.get("python_version") != "3.11.15":
        raise RuntimeError("R3-PAC2 requires repo Python 3.11.15")
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = _sha256_file(worker_path)
    worker_source = load_worker_snapshot(worker_sha256)

    router_definition = prepared["router_definition"]
    builder_definition = prepared["builder_definition"]
    router_checkpoint = prepared["router_checkpoint"]
    builder_checkpoint = prepared["builder_checkpoint"]
    with route_probe._public_train_authority():
        with ExitStack() as stack:
            actual_router = stack.enter_context(NativeTemporalCircuit.from_checkpoint(router_definition, router_checkpoint))
            actual_prompt = actual_router.advance_block(tuple(target_episode.token_ids), mode=TemporalStepMode.READ_ONLY, reset_before=True)
            marker = _create_spend_marker(target, hashlib.sha256(router_checkpoint).hexdigest(), hashlib.sha256(builder_checkpoint).hexdigest())
            actual_first = route_probe._generate_once(actual_router)
            identity_checker = _run_visible_list_identity_environment(target_episode, target)
            shape_observation = _shape_observation_from_checker(identity_checker)
            actual_feedback = feedback_probe._advance_feedback(actual_router, shape_observation)
            actual_second = route_probe._generate_once(actual_router)

        actual_shape = _observed_surface_token(actual_second)
        if actual_shape not in SHAPE_NAME_BY_TOKEN:
            payload = _terminal_router_stop_payload(
                upstream=upstream,
                history_anchor=history_anchor,
                generation_budget=generation_budget,
                prior_inventory=prior_inventory,
                prepared=prepared,
                target=target,
                marker=marker,
                actual_prompt=actual_prompt,
                actual_first=actual_first,
                identity_checker=identity_checker,
                actual_feedback=actual_feedback,
                actual_second=actual_second,
                actual_shape=actual_shape,
                started_ns=started_ns,
            )
            _write_output(payload)
            print(json.dumps({"outcome": payload["outcome"], "gates": payload["gates"]}, sort_keys=True))
            return 2
        shape_control_shape = SHAPE_SCALAR_TOKEN_ID
        sign_observation = _sign_observation_from_checker(identity_checker)
        target_feedback = _list_feedback_slot_block(identity_checker)
        payload_donor = prepared["feedback"][(MAP_FAMILY, 5)]
        payload_feedback = _list_feedback_slot_block(payload_donor)
        target_magnitude = bytes(target_feedback[1:-1])
        payload_magnitude = bytes(payload_feedback[1:-1])
        entrypoint = target.public.signature.entry_point
        parameter = target.public.signature.parameters[0].name
        prompts = (
            _builder_prompt(entrypoint, parameter, target_feedback, actual_shape, sign_observation),
            _builder_prompt(entrypoint, parameter, target_feedback, shape_control_shape, sign_observation),
            _builder_prompt(entrypoint, parameter, payload_feedback, actual_shape, sign_observation),
        )
        with ExitStack() as stack:
            builders = [stack.enter_context(NativeTemporalCircuit.from_checkpoint(builder_definition, builder_checkpoint)) for _ in prompts]
            prompt_steps = [builder.advance_block(prompt, mode=TemporalStepMode.READ_ONLY, reset_before=True) for builder, prompt in zip(builders, prompts, strict=True)]
            runs = [_generate_teacher_free(builder) for builder in builders]

    # The mapping below is posthoc scoring only.  The builder received the raw
    # checker-derived OBS_POS/OBS_NEG token, never a host-selected BUILD token.
    expected_build = (
        BUILD_ADD_TOKEN_ID
        if sign_observation == pac0_probe.OBS_POS_TOKEN_ID
        else BUILD_SUB_TOKEN_ID
    )
    analyses = (
        _analyze_builder_run(entrypoint=entrypoint, parameter=parameter, magnitude=target_magnitude, shape_token=actual_shape, build_token=expected_build, run=runs[0]),
        _analyze_builder_run(entrypoint=entrypoint, parameter=parameter, magnitude=target_magnitude, shape_token=shape_control_shape, build_token=expected_build, run=runs[1]),
        _analyze_builder_run(entrypoint=entrypoint, parameter=parameter, magnitude=payload_magnitude, shape_token=actual_shape, build_token=expected_build, run=runs[2]),
    )
    raw_sources = tuple(analysis.get("source") for analysis in analyses)
    sources = tuple(source if isinstance(source, str) else "" for source in raw_sources)
    checkers = (
        _visible_list_checker(sources[0], entrypoint, 3),
        _visible_list_checker(sources[1], entrypoint, 3),
        _visible_list_checker(sources[2], entrypoint, 3),
    )
    if analyses[0].get("passed") is True and isinstance(raw_sources[0], str):
        sandbox = action_probe._sandbox_record(
            sources[0], target, protocol,
            python_executable=executable,
            runtime_identity=runtime_identity,
            worker_source=worker_source,
            worker_sha256=worker_sha256,
        )
    else:
        sandbox = _sandbox_not_attempted(
            "actual builder source was incomplete, invalid, or physically inexact"
        )
    gates = _score_posthoc(
        actual_first=actual_first,
        identity_checker=identity_checker,
        actual_feedback=actual_feedback,
        actual_second=actual_second,
        actual_analysis=analyses[0],
        shape_analysis=analyses[1],
        payload_analysis=analyses[2],
        actual_checker=checkers[0],
        shape_checker=checkers[1],
        payload_checker=checkers[2],
        sandbox=sandbox,
    )
    passed = all(gates.values())
    outcome = (
        "PASS_R3_PAC2_TWO_SHAPE_LIST_MAP_CONSTRUCTION"
        if passed
        else "STOP_R3_PAC2_TWO_SHAPE_LIST_MAP_CONSTRUCTION"
    )
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-shape-action",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": outcome,
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": LIMITATIONS,
        "upstream_pac1": upstream,
        "history_anchor": history_anchor,
        "generation_budget": generation_budget,
        "authority": {
            "partition": "train",
            "target_task_id": target.public.task_id,
            "historical_prompt_baseline_exposure": True,
            "fresh_routed_action_target": True,
            "prior_target_artifact_inventory": prior_inventory,
            "model_retries": 0,
            "harness_retries": 0,
            "expected_routes_and_sources_scored_after_all_target_calls": True,
            "builder_sign_input": "raw checker-derived OBS_POS/OBS_NEG token",
            "python_selected_target_operator_or_source": False,
            "hidden_results_entered_cell_runtime": False,
        },
        "training": {
            "task_ids": [task.public.task_id for task in prepared["training"]],
            "router_capacity": prepared["router_capacity"],
            "builder_capacity": prepared["builder_capacity"],
            "router_checkpoint_sha256": hashlib.sha256(router_checkpoint).hexdigest(),
            "builder_checkpoint_sha256": hashlib.sha256(builder_checkpoint).hexdigest(),
        },
        "spend_marker": marker,
        "router_target": {
            "actual_prompt": route_probe._step_summary(actual_prompt),
            "actual_first": actual_first,
            "identity_checker": identity_checker,
            "actual_feedback": actual_feedback,
            "actual_second": actual_second,
            "shape_control": "builder input intervention SHAPE_MAP -> SHAPE_SCALAR; router cross-context path was not used",
        },
        "builder_target": {
            "prompt_steps": [route_probe._step_summary(steps) for steps in prompt_steps],
            "analyses": analyses,
        },
        "visible_target_checkers": checkers,
        "sandbox": sandbox,
        "gates": gates,
        "runtime": _runtime_record(started_ns),
    }
    _write_output(payload)
    print(json.dumps({"outcome": outcome, "gates": gates}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
