"""One-shot public-TRAIN signed-residual multi-action routing probe.

A donor-fitted role-chunk encoder normalizes scalar-offset /000 (+5), /008
(-5), and /007 (-5) to one concept stream.  A fresh native circuit learns two
literal trajectories whose observation token is derived by executing a frozen
visible checker against an identity Layer-0 source before LEARN.  Two target
checkpoint clones then differ only in one signed-residual feedback token and
must route to fixed ADD5 versus SUB5 actuators.

The /008 donor was previously a route-probe target and /007 was disclosed to
the historical prompt-only baseline.  This experiment makes no freshness,
literal-induction, synthesis, semantic, or transfer claim.  Importing or
compiling this module does not spend the target.
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
from humanbrain.role_chunk_episode import RoleChunkEpisode, RolePayloadKind
from humanbrain.sdr_union import token_sdr_indices
from humanbrain.simple_swe.sandbox import (
    load_worker_snapshot,
    python_runtime_identity,
    resolve_python_executable,
)
from humanbrain.simple_swe.tasks import GeneratedTask, PublicTask, generate_visible_partition


OUTPUT_PATH = Path(__file__).with_suffix(".json")
SPEND_MARKER_PATH = Path(__file__).with_name(
    f"{Path(__file__).stem}.t0_scalar_offset-007.spent.json"
)
FEEDBACK_SOURCE_PATH = Path(__file__).with_name(
    "public_simple_swe_role_chunk_feedback_action_probe.py"
)
FEEDBACK_ARTIFACT_PATH = Path(__file__).with_name(
    "public_simple_swe_role_chunk_feedback_action_probe.json"
)
ROUTE_ARTIFACT_PATH = Path(__file__).with_name(
    "public_simple_swe_role_chunk_route_probe.json"
)
HISTORICAL_BASELINE_PATH = REPO_ROOT / "artifacts" / "simple_swe_public_baseline_v1.json"

FAMILY = "t0_scalar_offset"
PLUS_DONOR_ORDINAL = 0
MINUS_DONOR_ORDINAL = 8
TARGET_ORDINAL = 7
TASK_RECORDS = {
    0: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/000",
        "task_sha256": "76a3a308651bb98a67801dbcbf7d2750a6de6f30748d6f500a30121dc950ea21",
        "hidden_test_sha256": "4f11a30a2c3035855d7661e089d0c534ffd836ae9b93673dd6f97d5168425190",
        "prompt_sha256": "cc721bf6d74e3b2fe9dc695e5deafbd73c520f8ac9a2bd0445fb3551bcdf15e0",
        "literal": 5,
    },
    8: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/008",
        "task_sha256": "49af88aa70cc249e28b5671067a4b48efc80930de4f67346daf6fcd9a6b027f9",
        "hidden_test_sha256": "28b10380a415d28d1dfbad1335be3b908c28fdcf12ee044384771eeedbaf51d7",
        "prompt_sha256": "1eb80f660b2ad781f266daaef2c525a13f2ac70aebad6e6364b6ff0f833e8efc",
        "literal": -5,
    },
    7: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/007",
        "task_sha256": "0853e92baae3c3e3d42c49006f61c216216010d846cb45b785e9710ceb22cf9f",
        "hidden_test_sha256": "0a035ae80022528eb35f1ed23ec5ac8099cf384d8dfc89dc3907dc0bfbf940a2",
        "prompt_sha256": "3f0876349788bc1b2f94600c3e402ab021799eddde680d92e2634fba4597e795",
        "literal": -5,
    },
}
TARGET_TASK_ID = str(TASK_RECORDS[TARGET_ORDINAL]["task_id"])
STRUCTURAL_SHA256 = "dd70041ef789a3ed38cf4cf23196068b3f7e2cc45fb5f1f7de562c7d8d778a46"
CONCEPT_STREAM_SHA256 = "fab3180295cc0ced8715005d36474a8221e1b7c706c84ba2e8824176890294db"
CONCEPT_STREAM_EVENTS = 56

TEST_TOKEN_ID = 1540
FIX_ADD5_TOKEN_ID = 1542
FIX_SUB5_TOKEN_ID = 1543
OBS_PLUS5_TOKEN_ID = 1602
OBS_MINUS5_TOKEN_ID = 1603
ACTION_NAME_BY_TOKEN = {
    TEST_TOKEN_ID: "TEST",
    FIX_ADD5_TOKEN_ID: "FIX_ADD5",
    FIX_SUB5_TOKEN_ID: "FIX_SUB5",
}
IDENTITY_LAYER0_TOKENS = ("ARG0", "END")
ADD5_LAYER0_TOKENS = ("ADD", "ARG0", "ADD", "TWO", "ADD", "TWO", "ONE", "END")
SUB5_LAYER0_TOKENS = ("SUB", "ARG0", "ADD", "TWO", "ADD", "TWO", "ONE", "END")
VISIBLE_INPUTS = (-4, 0, 7)

COLUMN_COUNT = 4_096
ACTIVE_COLUMNS = 11
CODEBOOK_SEED = 0
CONSTRUCTION_SEED = 17
BASE_CELLS_PER_COLUMN = 8
THETA_MATCH = 16
THETA_ACTIVE = 18
EXPECTED_CAPACITY = {
    "training_sequence_count": 2,
    "noninitial_arrivals": 116,
    "segment_capacity": 1_276,
    "max_pyramidal_cells": 34_044,
}

EXPECTED_FEEDBACK_SOURCE_SHA256 = (
    "67017f8902620cd24c7e3a0f1cea320f3e2ff39bb53875873b03c7fe2e9e5a8b"
)
EXPECTED_FEEDBACK_ARTIFACT_SHA256 = (
    "086683d594e1dfdc2e2d8d393a2334b0bb9a93208dbc2431ff4f38c2f4c0e34a"
)
EXPECTED_ROUTE_ARTIFACT_SHA256 = (
    "9ee0df214df3525a2e2898129b03295d9ad20706e26a7c52039db40c7a62884b"
)
EXPECTED_HISTORICAL_BASELINE_SHA256 = (
    "524c9df0971b1684e22398866fef0108f48fd40ab3fe8caeb221196c1e753f8b"
)

PINNED_SOURCE_SHA256 = {
    "experiments/r3_pilot/public_simple_swe_role_chunk_feedback_action_probe.py": EXPECTED_FEEDBACK_SOURCE_SHA256,
    "experiments/r3_pilot/public_simple_swe_role_chunk_layer0_action_probe.py": "8c2c86c1cdc6026639cdbce0ba88ac5145101c2c2dced8c3a56155f72326f3d1",
    "experiments/r3_pilot/public_simple_swe_role_chunk_route_probe.py": "5b4c80b9fe3a8d780b5a286fb33ad69885316e8b09a2a2723f7b2a46db1c1001",
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
    if _sha256_file(FEEDBACK_SOURCE_PATH) != EXPECTED_FEEDBACK_SOURCE_SHA256:
        raise RuntimeError("feedback-action source anchor changed")
    if _sha256_file(FEEDBACK_ARTIFACT_PATH) != EXPECTED_FEEDBACK_ARTIFACT_SHA256:
        raise RuntimeError("feedback-action result anchor changed")
    feedback = json.loads(FEEDBACK_ARTIFACT_PATH.read_text(encoding="utf-8"))
    if (
        feedback.get("outcome") != "PASS_REPETITION_CONDITIONED_BINARY_FEEDBACK_ACTION"
        or not isinstance(feedback.get("gates"), dict)
        or not all(feedback["gates"].values())
        or feedback.get("runtime", {}).get("probe_source_sha256")
        != EXPECTED_FEEDBACK_SOURCE_SHA256
    ):
        raise RuntimeError("feedback-action anchor is not an exact PASS")

    if _sha256_file(ROUTE_ARTIFACT_PATH) != EXPECTED_ROUTE_ARTIFACT_SHA256:
        raise RuntimeError("route artifact disclosure anchor changed")
    route = json.loads(ROUTE_ARTIFACT_PATH.read_text(encoding="utf-8"))
    prior_008 = _nested_task_records(
        route, str(TASK_RECORDS[MINUS_DONOR_ORDINAL]["task_id"])
    )
    if not prior_008:
        raise RuntimeError("/008 prior route-target use is no longer disclosed")

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
        "feedback_action_outcome": feedback["outcome"],
        "feedback_action_source_sha256": EXPECTED_FEEDBACK_SOURCE_SHA256,
        "feedback_action_artifact_sha256": EXPECTED_FEEDBACK_ARTIFACT_SHA256,
        "minus_donor_008_previously_used_as_route_target": True,
        "target_historical_prompt_disclosure_count": len(disclosures),
        "freshness_claimed": False,
    }


def _assert_frozen_sources() -> None:
    for source, expected in PINNED_SOURCE_SHA256.items():
        if _sha256_file(REPO_ROOT / source) != expected:
            raise RuntimeError(f"frozen signed-residual dependency changed: {source}")


def _durable_marker_inventory() -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    for path in sorted(PROBE_DIR.glob("*.spent.json")):
        text = path.read_text(encoding="utf-8")
        if TARGET_TASK_ID in text:
            raise RuntimeError(f"target already appears in durable marker: {path}")
        inventory.append({"path": _display_path(path), "sha256": _sha256_file(path)})
    return inventory


def _assert_committed_and_unspent() -> None:
    if OUTPUT_PATH.exists() or SPEND_MARKER_PATH.exists():
        raise RuntimeError("refusing spent signed-residual action probe")
    _assert_anchors()
    _assert_frozen_sources()
    paths = (
        Path(__file__).resolve(),
        FEEDBACK_ARTIFACT_PATH,
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
            raise RuntimeError(f"signed-residual source must be committed: {relative}")


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
        indexed[(FAMILY, PLUS_DONOR_ORDINAL)],
        indexed[(FAMILY, MINUS_DONOR_ORDINAL)],
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
            raise RuntimeError(f"public scalar-offset /{task.public.ordinal:03d} changed")
    return protocol, fit_donors, training, target


def _layer0_signature(episode: RoleChunkEpisode, public: PublicTask) -> Layer0ProgramSignature:
    if episode.task_id != public.task_id or episode.restore_prompt() != public.prompt:
        raise ValueError("role episode does not match public scalar-offset task")
    if public.partition != "train" or len(public.signature.parameters) != 1:
        raise ValueError("signed-residual action requires one public TRAIN parameter")
    parameter = public.signature.parameters[0]
    if parameter.type.constructor != "Int" or public.signature.result.constructor != "Int":
        raise ValueError("signed-residual action requires an Int -> Int signature")
    entries = episode.payloads_for(RolePayloadKind.ENTRY_POINT)
    parameters = episode.payloads_for(RolePayloadKind.PARAMETER)
    integers = episode.payloads_for(RolePayloadKind.INTEGER_LITERAL)
    floats = episode.payloads_for(RolePayloadKind.FLOAT_LITERAL)
    if len(entries) != 1 or len(parameters) != 1 or len(integers) != 1 or floats:
        raise ValueError("scalar-offset role sidecar shape changed")
    if entries[0].value != public.signature.entry_point or parameters[0].value != parameter.name:
        raise ValueError("role sidecar disagrees with public signature")
    return Layer0ProgramSignature(entries[0].value, parameters[0].value)


def _public_offset(episode: RoleChunkEpisode, public: PublicTask) -> int:
    _layer0_signature(episode, public)
    payloads = episode.payloads_for(RolePayloadKind.INTEGER_LITERAL)
    try:
        offset = int(payloads[0].value)
    except ValueError as error:
        raise RuntimeError("public offset payload is not an integer") from error
    if str(offset) != payloads[0].value:
        raise RuntimeError("public offset payload is not canonical")
    expected = TASK_RECORDS[public.ordinal]["literal"]
    if offset != expected:
        raise RuntimeError("public offset differs from frozen visible task")
    return offset


def _compile_source(
    tokens: tuple[str, ...], episode: RoleChunkEpisode, public: PublicTask
) -> str:
    signature = _layer0_signature(episode, public)
    max_nodes = 1 if tokens == IDENTITY_LAYER0_TOKENS else 7
    if tokens not in (IDENTITY_LAYER0_TOKENS, ADD5_LAYER0_TOKENS, SUB5_LAYER0_TOKENS):
        raise ValueError("Layer-0 source is outside the fixed signed-residual actuators")
    return Layer0ProgramCompiler(Layer0ProgramGrammar(max_nodes=max_nodes)).compile(
        tokens, signature
    )


def _visible_checker(
    source: str, entry_point: str, expected_offset: int
) -> dict[str, object]:
    namespace: dict[str, object] = {"__builtins__": {}}
    try:
        exec(compile(source, "<signed-residual-visible-checker>", "exec"), namespace)
        function = namespace.get(entry_point)
        if not callable(function):
            raise TypeError("compiled source did not define the public entry point")
        observed = tuple(function(value) for value in VISIBLE_INPUTS)
    except Exception as error:
        return {
            "inputs": list(VISIBLE_INPUTS),
            "expected_offset": expected_offset,
            "expected": None,
            "observed": None,
            "residual": None,
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }
    expected = tuple(value + expected_offset for value in VISIBLE_INPUTS)
    if any(type(value) is not int for value in observed):
        return {
            "inputs": list(VISIBLE_INPUTS),
            "expected_offset": expected_offset,
            "expected": list(expected),
            "observed": list(observed),
            "residual": None,
            "passed": False,
            "error": "visible checker observed a non-exact integer",
        }
    residual = tuple(wanted - actual for wanted, actual in zip(expected, observed, strict=True))
    return {
        "inputs": list(VISIBLE_INPUTS),
        "expected_offset": expected_offset,
        "expected": list(expected),
        "observed": list(observed),
        "residual": list(residual),
        "passed": observed == expected,
        "error": None,
    }


def _observation_token(checker: dict[str, object]) -> int:
    if checker.get("error") is not None:
        raise RuntimeError("malformed visible checker result cannot become cell feedback")
    if checker.get("inputs") != list(VISIBLE_INPUTS):
        raise RuntimeError("visible checker inputs changed")
    expected = checker.get("expected")
    observed = checker.get("observed")
    residual = checker.get("residual")
    if not all(
        isinstance(values, list)
        and len(values) == len(VISIBLE_INPUTS)
        and all(type(value) is int for value in values)
        for values in (expected, observed, residual)
    ):
        raise RuntimeError("visible checker did not return exact integer vectors")
    recomputed = [wanted - actual for wanted, actual in zip(expected, observed, strict=True)]
    if residual != recomputed:
        raise RuntimeError("visible checker residual is inconsistent")
    if residual == [5, 5, 5]:
        return OBS_PLUS5_TOKEN_ID
    if residual == [-5, -5, -5]:
        return OBS_MINUS5_TOKEN_ID
    raise RuntimeError("mixed or unsupported signed residual cannot become feedback")


def _training_sequences(
    training: tuple[GeneratedTask, GeneratedTask],
    episodes: dict[int, RoleChunkEpisode],
) -> tuple[tuple[int, ...], tuple[int, ...], list[dict[str, object]]]:
    sequences: list[tuple[int, ...]] = []
    derivations: list[dict[str, object]] = []
    for task in training:
        episode = episodes[task.public.ordinal]
        identity = _compile_source(IDENTITY_LAYER0_TOKENS, episode, task.public)
        offset = _public_offset(episode, task.public)
        checker = _visible_checker(identity, task.public.signature.entry_point, offset)
        observation = _observation_token(checker)
        action = FIX_ADD5_TOKEN_ID if observation == OBS_PLUS5_TOKEN_ID else FIX_SUB5_TOKEN_ID
        sequences.append((*episode.token_ids, TEST_TOKEN_ID, observation, action))
        derivations.append(
            {
                "task_id": task.public.task_id,
                "public_offset_from_sidecar": offset,
                "identity_source": identity,
                "visible_checker": checker,
                "derived_observation_token_id": observation,
                "teacher_action_token_id": action,
                "observation_derived_before_learn": True,
            }
        )
    if (
        derivations[0]["derived_observation_token_id"] != OBS_PLUS5_TOKEN_ID
        or derivations[1]["derived_observation_token_id"] != OBS_MINUS5_TOKEN_ID
    ):
        raise RuntimeError("donor signed observations differ from frozen trajectories")
    return sequences[0], sequences[1], derivations


def _definition(
    sequences: tuple[tuple[int, ...], tuple[int, ...]], target_stream: tuple[int, ...]
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
    used = tuple(sorted({*(token for sequence in sequences for token in sequence), *target_stream}))
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
        raise RuntimeError("signed-residual capacity differs from frozen arithmetic")
    return definition, capacity


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
    protocol, fit_donors, training, target = _select_tasks()
    encoder, _concept_streams, _base_streams, _records = (
        route_probe._prepare_representations(fit_donors, (*training, target))
    )
    episodes = {
        task.public.ordinal: encoder.encode(task.public) for task in (*training, target)
    }
    if len(fit_donors) != 8 or _public_offset(
        episodes[TARGET_ORDINAL], target.public
    ) != -5:
        raise RuntimeError("target offset or donor-fitted encoder cohort changed")
    streams = {tuple(episode.token_ids) for episode in episodes.values()}
    if (
        len(streams) != 1
        or any(len(stream) != CONCEPT_STREAM_EVENTS for stream in streams)
        or any(episode.structural_sha256 != STRUCTURAL_SHA256 for episode in episodes.values())
        or any(_sha256_json(tuple(episode.token_ids)) != CONCEPT_STREAM_SHA256 for episode in episodes.values())
    ):
        raise RuntimeError("scalar-offset tasks do not share the frozen concept stream")

    plus_sequence, minus_sequence, derivations = _training_sequences(training, episodes)
    sequences = (plus_sequence, minus_sequence)
    definition, capacity = _definition(sequences, tuple(episodes[TARGET_ORDINAL].token_ids))
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
        }
    updated = source
    if action == "FIX_ADD5":
        updated = _compile_source(ADD5_LAYER0_TOKENS, episode, public)
    elif action == "FIX_SUB5":
        updated = _compile_source(SUB5_LAYER0_TOKENS, episode, public)
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
        "experiment": "r3-public-simple-swe-role-chunk-signed-residual-action",
        "state": "TARGET_ATTEMPT_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target_task_id": target.public.task_id,
        "checkpoint_sha256": checkpoint_sha256,
        "atomic_no_argument_generation_calls_planned": 4,
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
        raise RuntimeError("refusing already spent signed-residual target") from error
    return {
        "path": _display_path(SPEND_MARKER_PATH),
        "sha256": _sha256_file(SPEND_MARKER_PATH),
        "exclusive_create": True,
        "flushed_and_fsynced_before_all_four_generate_calls": True,
    }


def _score_after_all_calls(
    *,
    actual_first: dict[str, object],
    control_first: dict[str, object],
    actual_second: dict[str, object],
    control_second: dict[str, object],
    actual_observation: int,
    control_observation: int,
    actual_feedback: dict[str, object],
    control_feedback: dict[str, object],
    dispatches: tuple[dict[str, object], ...],
    actual_final_checker: dict[str, object],
    control_final_checker: dict[str, object],
    sandbox: dict[str, object],
) -> tuple[dict[str, object], dict[str, bool]]:
    scored = {
        "actual_first": route_probe._score_generation(actual_first, TEST_TOKEN_ID),
        "control_first": route_probe._score_generation(control_first, TEST_TOKEN_ID),
        "actual_second": route_probe._score_generation(actual_second, FIX_SUB5_TOKEN_ID),
        "control_second": route_probe._score_generation(control_second, FIX_ADD5_TOKEN_ID),
    }
    expected_dispatches = (
        (TEST_TOKEN_ID, "TEST", False),
        (TEST_TOKEN_ID, "TEST", False),
        (FIX_SUB5_TOKEN_ID, "FIX_SUB5", True),
        (FIX_ADD5_TOKEN_ID, "FIX_ADD5", True),
    )
    gates = {
        "first_clone_generations_bit_exact": actual_first == control_first,
        "physical_test_test_first": bool(
            scored["actual_first"]["correct"] and scored["control_first"]["correct"]
        ),
        "observation_only_divergent_cell_input": bool(
            actual_observation == OBS_MINUS5_TOKEN_ID
            and control_observation == OBS_PLUS5_TOKEN_ID
            and actual_feedback.get("advanced") is True
            and control_feedback.get("advanced") is True
            and actual_feedback.get("observation_token_id") == actual_observation
            and control_feedback.get("observation_token_id") == control_observation
        ),
        "physical_sub5_add5_second": bool(
            scored["actual_second"]["correct"] and scored["control_second"]["correct"]
        ),
        "exact_observed_route_dispatch": all(
            dispatch.get("dispatched") is True
            and dispatch.get("observed_token_id") == token
            and dispatch.get("action") == action
            and dispatch.get("source_changed") is changed
            for dispatch, (token, action, changed) in zip(
                dispatches, expected_dispatches, strict=True
            )
        ),
        "actual_and_control_changed_once": bool(
            sum(dispatch.get("source_changed") is True for dispatch in dispatches[:2]) == 0
            and dispatches[2].get("source_changed") is True
            and dispatches[3].get("source_changed") is True
        ),
        "visible_actual_and_mirror_control_pass": bool(
            actual_final_checker.get("passed") is True
            and control_final_checker.get("passed") is True
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
    marker_inventory = _durable_marker_inventory()
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
        raise RuntimeError("signed-residual probe requires repo Python 3.11.15")
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = _sha256_file(worker_path)
    worker_source = load_worker_snapshot(worker_sha256)

    target_episode = episodes[TARGET_ORDINAL]
    signature = _layer0_signature(target_episode, target.public)
    target_offset = _public_offset(target_episode, target.public)
    mirrored_offset = -target_offset
    identity_source = _compile_source(IDENTITY_LAYER0_TOKENS, target_episode, target.public)
    actual_source = identity_source
    control_source = identity_source

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
                raise RuntimeError("signed-residual checkpoint clone differs")
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
            actual_checker = _visible_checker(
                actual_source, signature.entry_point, target_offset
            )
            control_checker = _visible_checker(
                control_source, signature.entry_point, mirrored_offset
            )
            actual_observation = _observation_token(actual_checker)
            control_observation = _observation_token(control_checker)
            actual_feedback = feedback_probe._advance_feedback(actual, actual_observation)
            control_feedback = feedback_probe._advance_feedback(control, control_observation)

            actual_second = route_probe._generate_once(actual)
            control_second = route_probe._generate_once(control)

    actual_source, actual_second_dispatch = _dispatch_observed_action(
        _observed_surface_token(actual_second), actual_source, target_episode, target.public
    )
    control_source, control_second_dispatch = _dispatch_observed_action(
        _observed_surface_token(control_second), control_source, target_episode, target.public
    )
    actual_final_checker = _visible_checker(
        actual_source, signature.entry_point, target_offset
    )
    control_final_checker = _visible_checker(
        control_source, signature.entry_point, mirrored_offset
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

    dispatches = (
        actual_first_dispatch,
        control_first_dispatch,
        actual_second_dispatch,
        control_second_dispatch,
    )
    scored_calls, gates = _score_after_all_calls(
        actual_first=actual_first,
        control_first=control_first,
        actual_second=actual_second,
        control_second=control_second,
        actual_observation=actual_observation,
        control_observation=control_observation,
        actual_feedback=actual_feedback,
        control_feedback=control_feedback,
        dispatches=dispatches,
        actual_final_checker=actual_final_checker,
        control_final_checker=control_final_checker,
        sandbox=sandbox,
    )
    passed = all(gates.values())
    outcome = (
        "PASS_FIXED_SIGNED_RESIDUAL_ACTION_ROUTING"
        if passed
        else "STOP_FIXED_SIGNED_RESIDUAL_ACTION_ROUTING"
    )
    native_identity = loaded_library_identity()
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-signed-residual-action",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": outcome,
        "claim_boundary": (
            "Visible TRAIN only: one shared normalized scalar-offset template routes exact +/-5 residual tokens to two fixed Layer-0 actuators. "
            "This is fixed two-way actuator routing, not literal induction, program synthesis, semantic interpretation, self-correction, transfer, DEV/eval, held-out, M0, or C1 capability."
        ),
        "authority": {
            "partition": "train",
            "target_task_id": target.public.task_id,
            "minus_donor_008_previously_used_as_route_target": True,
            "target_historical_prompt_baseline_reuse": True,
            "freshness_claimed": False,
            "checkpoint_clones": 2,
            "identical_prompt_presentations": 2,
            "atomic_no_argument_generation_calls": 4,
            "canonical_target_sandbox_calls": 1,
            "model_retries": 0,
            "harness_retries": 0,
            "hidden_cases_or_verdicts_entered_cell_runtime": False,
            "expected_action_labels_scored_only_after_all_four_calls": True,
            "preexisting_durable_marker_inventory": marker_inventory,
        },
        "anchors": anchors,
        "tasks": {
            str(ordinal): record for ordinal, record in TASK_RECORDS.items()
        },
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
        "actual": {
            "expected_offset_from_public_sidecar": target_offset,
            "initial_source": identity_source,
            "first_raw_generation": actual_first,
            "first_dispatch": actual_first_dispatch,
            "visible_checker_before_feedback": actual_checker,
            "observation_token_id": actual_observation,
            "feedback_step": actual_feedback,
            "second_raw_generation": actual_second,
            "second_dispatch": actual_second_dispatch,
            "final_visible_checker": actual_final_checker,
            "final_source": actual_source,
        },
        "mirrored_causal_control": {
            "mirrored_expected_offset": mirrored_offset,
            "initial_source": identity_source,
            "first_raw_generation": control_first,
            "first_dispatch": control_first_dispatch,
            "visible_checker_before_feedback": control_checker,
            "observation_token_id": control_observation,
            "feedback_step": control_feedback,
            "second_raw_generation": control_second,
            "second_dispatch": control_second_dispatch,
            "final_visible_checker": control_final_checker,
            "final_source": control_source,
        },
        "scored_after_all_four_calls": scored_calls,
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
