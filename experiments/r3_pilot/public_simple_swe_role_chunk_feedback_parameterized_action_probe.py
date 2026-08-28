"""R3-PAC1: execution-feedback parameterized construction probe.

Two fresh fitted-cell circuits are composed without shared learned state.  A
RoleChunk router receives only the sign of a residual from a real public-TRAIN
identity execution and physically elects BUILD_ADD or BUILD_SUB.  Before that
execution, three exact builder-checkpoint clones capture the same public target
entrypoint in role slot 0.  Afterwards they capture either the target checker's
absolute residual bytes or a genuine donor checker's bytes in slot 2, receive
the observed BUILD token, and generate a complete scalar function directly.

Scalar-offset /002 appeared in two historical prompt-only baseline arms under
an older task hash.  No global freshness claim is made.  The target has no
prior native/routed-action artifact or spend marker.  Importing or compiling
this module does not spend it.
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

import public_simple_swe_role_chunk_parameterized_action_probe as pac0_probe
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
from humanbrain.program import (
    Layer0ProgramCompiler,
    Layer0ProgramGrammar,
    Layer0ProgramSignature,
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
from humanbrain.simple_swe.tasks import GeneratedTask, PublicTask, generate_visible_partition


route_probe = pac0_probe.route_probe
feedback_probe = pac0_probe.feedback_probe
action_probe = pac0_probe.action_probe
signed_probe = pac0_probe.signed_probe
role_place_probe = pac0_probe.role_place_probe

OUTPUT_PATH = Path(__file__).with_suffix(".json")
SPEND_MARKER_PATH = Path(__file__).with_name(
    f"{Path(__file__).stem}.t0_scalar_offset-002.spent.json"
)
PAC0_SOURCE_PATH = Path(__file__).with_name(
    "public_simple_swe_role_chunk_parameterized_action_probe.py"
)
PAC0_ARTIFACT_PATH = Path(__file__).with_name(
    "public_simple_swe_role_chunk_parameterized_action_probe.json"
)
HISTORICAL_BASELINE_PATH = REPO_ROOT / "artifacts" / "simple_swe_public_baseline_v1.json"

FAMILY = "t0_scalar_offset"
ADD_DONOR_ORDINAL = 0
SUB_DONOR_ORDINAL = 1
TARGET_ORDINAL = 2
PAYLOAD_CONTROL_DONOR_ORDINAL = SUB_DONOR_ORDINAL
TASK_RECORDS = {
    0: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/000",
        "task_sha256": "76a3a308651bb98a67801dbcbf7d2750a6de6f30748d6f500a30121dc950ea21",
        "hidden_test_sha256": "4f11a30a2c3035855d7661e089d0c534ffd836ae9b93673dd6f97d5168425190",
        "prompt_sha256": "cc721bf6d74e3b2fe9dc695e5deafbd73c520f8ac9a2bd0445fb3551bcdf15e0",
        "entry_point": "solve_d4c2e76cb6f6e7b8",
        "parameter": "n",
        "offset": 5,
    },
    1: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/001",
        "task_sha256": "65087aed8fc475d533af01823cb5bcd2d5ca6b7c66537a5c9eec9f1f973b0587",
        "hidden_test_sha256": "3fe0b3ec9080014ce2fd4cabd17223f42946c8cbc0d97e4386faefb04b42a9d7",
        "prompt_sha256": "0dd9a25ff8c5da412840edb227dd6a1afa3a3d0263568135da04442eb9de35c2",
        "entry_point": "solve_aeb58999626eec7c",
        "parameter": "n",
        "offset": -7,
    },
    2: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/002",
        "task_sha256": "7953171b75182cc2b620f08155bc9a15fe7c2cb572704c4fe07b53bd509c9978",
        "hidden_test_sha256": "2044af34672857f860dd092a3258261c558350d6a1cfdd1e85731f7d68a4645e",
        "prompt_sha256": "bdf5e4f62fd5ad5a3f2aaf5696399ba90ea12fa89568d914f7645fa2f12fd4f0",
        "entry_point": "solve_f6e1ac74b5400d74",
        "parameter": "n",
        "offset": -2,
    },
}
TARGET_TASK_ID = str(TASK_RECORDS[TARGET_ORDINAL]["task_id"])
HISTORICAL_TARGET_TASK_SHA256 = (
    "2df1112850c993edfbf3894d7ef5a700985bbd36157e5cbeee354776bdfec302"
)

STRUCTURAL_SHA256 = pac0_probe.STRUCTURAL_SHA256
CONCEPT_STREAM_SHA256 = pac0_probe.CONCEPT_STREAM_SHA256
CONCEPT_STREAM_EVENTS = pac0_probe.CONCEPT_STREAM_EVENTS
TEST_TOKEN_ID = pac0_probe.TEST_TOKEN_ID
BUILD_ADD_TOKEN_ID = pac0_probe.BUILD_ADD_TOKEN_ID
BUILD_SUB_TOKEN_ID = pac0_probe.BUILD_SUB_TOKEN_ID
OBS_POS_TOKEN_ID = pac0_probe.OBS_POS_TOKEN_ID
OBS_NEG_TOKEN_ID = pac0_probe.OBS_NEG_TOKEN_ID
BUILD_NAME_BY_TOKEN = pac0_probe.BUILD_NAME_BY_TOKEN

ROUTER_COLUMN_COUNT = pac0_probe.ROUTER_COLUMN_COUNT
BUILDER_COLUMN_COUNT = pac0_probe.BUILDER_COLUMN_COUNT
ACTIVE_COLUMNS = pac0_probe.ACTIVE_COLUMNS
CODEBOOK_SEED = pac0_probe.CODEBOOK_SEED
BASE_CELLS_PER_COLUMN = pac0_probe.BASE_CELLS_PER_COLUMN
CONSTRUCTION_SEED = pac0_probe.CONSTRUCTION_SEED
THETA_MATCH = pac0_probe.THETA_MATCH
THETA_ACTIVE = pac0_probe.THETA_ACTIVE
EXPECTED_ROUTER_CAPACITY = pac0_probe.EXPECTED_ROUTER_CAPACITY
EXPECTED_BUILDER_CAPACITY = {
    "training_sequence_lengths": [56, 56],
    "noninitial_arrivals": 110,
    "segment_capacity": 1_210,
    "max_pyramidal_cells": 5_306,
    "threshold_offset_cells": 12_148,
}

EOS_TOKEN_ID = role_place_probe.EOS_TOKEN_ID
SOURCE_PREFIX = b"def "
SOURCE_BEFORE_OPERATOR = b"(n):\n    return n "
SOURCE_BEFORE_MAGNITUDE = b" "
SOURCE_SUFFIX = b"\n"
EXPECTED_SOURCE_SHA256 = {
    "actual": "4e5ae71905f25d84cff5ff3932958af9d5add775e79e1bdae8b4501b92de700b",
    "sign_intervention": "1b114698a12473eede038496bf4d9f1fc45960a75c55b51c429c7769e3e92eba",
    "payload_intervention": "04b29bc752f6b4c5f5604e1210b630399607d819f0e812c675d26d248011b00e",
}

EXPECTED_PAC0_SOURCE_SHA256 = (
    "3b91785bc9dff5993071ae20bf9bde8988749b035d920e6b2ec2d7ce057aeae7"
)
EXPECTED_PAC0_ARTIFACT_SHA256 = (
    "08869a08ab1b9c7385d19cdaa66fd7d3e814b064cf3eca3efb19908f47ec3b0f"
)
EXPECTED_HISTORICAL_BASELINE_SHA256 = pac0_probe.EXPECTED_HISTORICAL_BASELINE_SHA256
PINNED_SOURCE_SHA256 = {
    "experiments/r3_pilot/public_simple_swe_role_chunk_parameterized_action_probe.py": EXPECTED_PAC0_SOURCE_SHA256,
    **pac0_probe.PINNED_SOURCE_SHA256,
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
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _display_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _assert_anchors() -> dict[str, object]:
    if _sha256_file(PAC0_SOURCE_PATH) != EXPECTED_PAC0_SOURCE_SHA256:
        raise RuntimeError("R3-PAC0 source anchor changed")
    if _sha256_file(PAC0_ARTIFACT_PATH) != EXPECTED_PAC0_ARTIFACT_SHA256:
        raise RuntimeError("R3-PAC0 artifact anchor changed")
    pac0 = json.loads(PAC0_ARTIFACT_PATH.read_text(encoding="utf-8"))
    if pac0.get("outcome") != "PASS_R3_PAC0_ROLE_BOUND_CONSTRUCTION":
        raise RuntimeError("R3-PAC0 did not pass")
    pac0_anchors = pac0_probe._assert_anchors()
    if _sha256_file(HISTORICAL_BASELINE_PATH) != EXPECTED_HISTORICAL_BASELINE_SHA256:
        raise RuntimeError("historical baseline changed")
    baseline = json.loads(HISTORICAL_BASELINE_PATH.read_text(encoding="utf-8"))
    disclosures = pac0_probe._nested_task_records(baseline, TARGET_TASK_ID)
    if (
        len(disclosures) != 2
        or any(record.get("task_sha256") != HISTORICAL_TARGET_TASK_SHA256 for record in disclosures)
        or any(record.get("prompt_sha256") != TASK_RECORDS[TARGET_ORDINAL]["prompt_sha256"] for record in disclosures)
        or HISTORICAL_TARGET_TASK_SHA256 == TASK_RECORDS[TARGET_ORDINAL]["task_sha256"]
    ):
        raise RuntimeError("historical /002 prompt-only disclosure changed")
    return {
        "pac0_outcome": pac0["outcome"],
        "pac0_source_sha256": EXPECTED_PAC0_SOURCE_SHA256,
        "pac0_artifact_sha256": EXPECTED_PAC0_ARTIFACT_SHA256,
        "pac0_transitive_anchors": pac0_anchors,
        "historical_baseline_sha256": EXPECTED_HISTORICAL_BASELINE_SHA256,
        "historical_target_occurrences": len(disclosures),
        "historical_target_task_sha256": HISTORICAL_TARGET_TASK_SHA256,
        "current_target_task_sha256": TASK_RECORDS[TARGET_ORDINAL]["task_sha256"],
        "historical_prompt_sha256_unchanged": True,
        "historical_task_content_drift": True,
    }


def _assert_frozen_sources() -> None:
    for relative, expected in PINNED_SOURCE_SHA256.items():
        if _sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"pinned source changed: {relative}")


def _prior_routed_target_inventory() -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    excluded = {Path(__file__).resolve(), OUTPUT_PATH.resolve(), SPEND_MARKER_PATH.resolve()}
    for path in sorted(PROBE_DIR.glob("public_simple_swe_*.py")) + sorted(
        PROBE_DIR.glob("public_simple_swe_*.json")
    ):
        if path.resolve() in excluded:
            continue
        text = path.read_text(encoding="utf-8")
        if TARGET_TASK_ID in text:
            inventory.append({"path": _display_path(path), "sha256": _sha256_file(path)})
    return inventory


def _assert_committed_and_unspent() -> None:
    _assert_frozen_sources()
    if OUTPUT_PATH.exists() or SPEND_MARKER_PATH.exists():
        raise RuntimeError("refusing already spent R3-PAC1 target")
    if _prior_routed_target_inventory():
        raise RuntimeError("R3-PAC1 target already appears in a routed artifact")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", str(Path(__file__).resolve())],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("commit the frozen R3-PAC1 source before spending target")


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
            or hashlib.sha256(task.public.prompt.encode()).hexdigest() != expected["prompt_sha256"]
            or task.public.signature.entry_point != expected["entry_point"]
            or parameter.name != expected["parameter"]
            or parameter.type.constructor != "Int"
            or task.public.signature.result.constructor != "Int"
        ):
            raise RuntimeError(f"public scalar-offset /{task.public.ordinal:03d} changed")
    return protocol, fit_donors, training, target


def _fit_role_chunk_encoder(fit_donors: tuple[GeneratedTask, ...]) -> RoleChunkEpisodeEncoder:
    return pac0_probe._fit_role_chunk_encoder(fit_donors)


def _environment_expected_offset(episode: RoleChunkEpisode, public: PublicTask) -> int:
    """Read the public task literal only inside the visible environment boundary."""
    if episode.task_id != public.task_id or episode.restore_prompt() != public.prompt:
        raise RuntimeError("role episode does not match public environment task")
    payloads = episode.payloads_for(RolePayloadKind.INTEGER_LITERAL)
    if len(payloads) != 1:
        raise RuntimeError("public environment requires one integer literal")
    try:
        offset = int(payloads[0].value)
    except ValueError as error:
        raise RuntimeError("public environment offset is not an integer") from error
    expected = TASK_RECORDS[public.ordinal]["offset"]
    if str(offset) != payloads[0].value or offset != expected:
        raise RuntimeError("public environment offset changed")
    return offset


def _assert_concept_episode(episode: RoleChunkEpisode, task: GeneratedTask) -> None:
    if (
        episode.task_id != task.public.task_id
        or episode.restore_prompt() != task.public.prompt
        or len(episode.token_ids) != CONCEPT_STREAM_EVENTS
        or episode.structural_sha256 != STRUCTURAL_SHA256
        or _sha256_json(tuple(episode.token_ids)) != CONCEPT_STREAM_SHA256
    ):
        raise RuntimeError("scalar-offset task differs from frozen concept stream")


def _identity_source(public: PublicTask) -> str:
    if public.partition != "train" or len(public.signature.parameters) != 1:
        raise RuntimeError("identity diagnostic requires one public TRAIN parameter")
    parameter = public.signature.parameters[0]
    if (
        parameter.name != "n"
        or parameter.type.constructor != "Int"
        or public.signature.result.constructor != "Int"
    ):
        raise RuntimeError("identity diagnostic is frozen to scalar Int parameter n")
    return Layer0ProgramCompiler(Layer0ProgramGrammar(max_nodes=1)).compile(
        signed_probe.IDENTITY_LAYER0_TOKENS,
        Layer0ProgramSignature(public.signature.entry_point, parameter.name),
    )


def _run_visible_identity_environment(
    episode: RoleChunkEpisode, task: GeneratedTask
) -> dict[str, object]:
    expected_offset = _environment_expected_offset(episode, task.public)
    return signed_probe._visible_checker(
        _identity_source(task.public), task.public.signature.entry_point, expected_offset
    )


def _uniform_signed_residual(checker: dict[str, object]) -> int:
    if (
        checker.get("error") is not None
        or checker.get("inputs") != list(signed_probe.VISIBLE_INPUTS)
        or checker.get("passed") is not False
    ):
        raise RuntimeError("malformed or passing identity result cannot become feedback")
    expected = checker.get("expected")
    observed = checker.get("observed")
    residual = checker.get("residual")
    if (
        not isinstance(expected, list)
        or not isinstance(observed, list)
        or not isinstance(residual, list)
        or len(expected) != len(signed_probe.VISIBLE_INPUTS)
        or len(observed) != len(signed_probe.VISIBLE_INPUTS)
        or len(residual) != len(signed_probe.VISIBLE_INPUTS)
        or any(type(value) is not int for value in (*expected, *observed, *residual))
        or any(wanted - actual != delta for wanted, actual, delta in zip(expected, observed, residual, strict=True))
        or len(set(residual)) != 1
        or residual[0] == 0
    ):
        raise RuntimeError("mixed, zero, non-integer, or inconsistent residual cannot become feedback")
    return int(residual[0])


def _sign_observation(checker: dict[str, object]) -> int:
    return OBS_POS_TOKEN_ID if _uniform_signed_residual(checker) > 0 else OBS_NEG_TOKEN_ID


def _feedback_magnitude_bytes(checker: dict[str, object]) -> bytes:
    """Serialize only a completed real checker record into canonical decimal bytes."""
    residual = _uniform_signed_residual(checker)
    magnitude = str(abs(residual)).encode("ascii")
    if not magnitude.isdigit() or magnitude.startswith(b"0"):
        raise RuntimeError("feedback magnitude is not canonical positive decimal bytes")
    return magnitude


def _entrypoint_slot_block(entrypoint: str) -> tuple[int, ...]:
    identifier = entrypoint.encode("ascii")
    if len(identifier) != 22 or not identifier.startswith(b"solve_"):
        raise RuntimeError("R3-PAC1 entrypoint slot shape changed")
    return (slot_start_token_id(0), *identifier, slot_end_token_id(0))


def _feedback_slot_block(checker: dict[str, object]) -> tuple[int, ...]:
    magnitude = _feedback_magnitude_bytes(checker)
    return (slot_start_token_id(2), *magnitude, slot_end_token_id(2))


def _require_build_token(build_token: int | None) -> int:
    if build_token not in BUILD_NAME_BY_TOKEN:
        raise RuntimeError("observed route is not a physical BUILD token")
    return int(build_token)


def _builder_teacher_target(build_token: int) -> tuple[int, ...]:
    build_token = _require_build_token(build_token)
    operator = ord("+") if build_token == BUILD_ADD_TOKEN_ID else ord("-")
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
    entrypoint: str, checker: dict[str, object], build_token: int
) -> tuple[int, ...]:
    return (
        *_entrypoint_slot_block(entrypoint),
        *_feedback_slot_block(checker),
        _require_build_token(build_token),
        *_builder_teacher_target(build_token),
    )


def _router_sequences(
    training: tuple[GeneratedTask, GeneratedTask],
    episodes: dict[int, RoleChunkEpisode],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], list[dict[str, object]]]:
    sequences: list[tuple[int, ...]] = []
    records: list[dict[str, object]] = []
    for task in training:
        episode = episodes[task.public.ordinal]
        checker = _run_visible_identity_environment(episode, task)
        observation = _sign_observation(checker)
        build = BUILD_ADD_TOKEN_ID if observation == OBS_POS_TOKEN_ID else BUILD_SUB_TOKEN_ID
        sequence = (*episode.token_ids, TEST_TOKEN_ID, observation, build)
        sequences.append(sequence)
        records.append(
            {
                "task_id": task.public.task_id,
                "identity_source": _identity_source(task.public),
                "visible_identity_checker": checker,
                "uniform_signed_residual": _uniform_signed_residual(checker),
                "feedback_magnitude_hex": _feedback_magnitude_bytes(checker).hex(),
                "derived_sign_token_id": observation,
                "teacher_build_token_id": build,
                "residual_derived_before_learn": True,
                "sequence_token_ids": list(sequence),
            }
        )
    if (
        records[0]["uniform_signed_residual"] != 5
        or records[1]["uniform_signed_residual"] != -7
        or records[0]["derived_sign_token_id"] != OBS_POS_TOKEN_ID
        or records[1]["derived_sign_token_id"] != OBS_NEG_TOKEN_ID
    ):
        raise RuntimeError("donor feedback differs from frozen +5/-7 trajectories")
    return (sequences[0], sequences[1]), records


def _builder_definition(
    sequences: tuple[tuple[int, ...], tuple[int, ...]]
) -> tuple[NativeTemporalCircuitDefinition, dict[str, object]]:
    lengths = [len(sequence) for sequence in sequences]
    noninitial_arrivals = sum(length - 1 for length in lengths)
    segment_capacity = ACTIVE_COLUMNS * noninitial_arrivals
    config = NativeTemporalCircuitConfig(
        column_count=BUILDER_COLUMN_COUNT,
        max_pyramidal_cells=BUILDER_COLUMN_COUNT * BASE_CELLS_PER_COLUMN + segment_capacity,
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
    threshold_offset_cells = 2 * config.max_pyramidal_cells + 3 * config.column_count
    definition = NativeTemporalCircuitDefinition(
        config,
        token_codes,
        seeded_threshold_offsets_mv(threshold_offset_cells, config.construction_seed).tolist(),
    )
    capacity: dict[str, object] = {
        "training_sequence_lengths": lengths,
        "noninitial_arrivals": noninitial_arrivals,
        "segment_capacity": segment_capacity,
        "max_pyramidal_cells": config.max_pyramidal_cells,
        "threshold_offset_cells": threshold_offset_cells,
    }
    if capacity != EXPECTED_BUILDER_CAPACITY:
        raise RuntimeError("builder capacity differs from exact sequence arithmetic")
    return definition, capacity


def _prepare_training() -> dict[str, object]:
    protocol, fit_donors, training, target = _select_tasks()
    encoder = _fit_role_chunk_encoder(fit_donors)
    episodes = {task.public.ordinal: encoder.encode(task.public) for task in training}
    for task in training:
        _assert_concept_episode(episodes[task.public.ordinal], task)
    if len({tuple(episode.token_ids) for episode in episodes.values()}) != 1:
        raise RuntimeError("router donor concept streams are not bit exact")

    router_sequences, derivations = _router_sequences(training, episodes)
    router_definition, router_capacity = pac0_probe._router_definition(router_sequences)
    router_checkpoint, router_identity, router_steps = pac0_probe._train_circuit(
        router_definition, router_sequences
    )
    router_train = [
        {**derivation, **steps}
        for derivation, steps in zip(derivations, router_steps, strict=True)
    ]

    feedback_by_ordinal = {
        task.public.ordinal: derivation["visible_identity_checker"]
        for task, derivation in zip(training, derivations, strict=True)
    }
    builder_sequences = tuple(
        _builder_train_episode(
            task.public.signature.entry_point,
            feedback_by_ordinal[task.public.ordinal],
            BUILD_ADD_TOKEN_ID if task.public.ordinal == ADD_DONOR_ORDINAL else BUILD_SUB_TOKEN_ID,
        )
        for task in training
    )
    builder_definition, builder_capacity = _builder_definition(builder_sequences)
    builder_checkpoint, builder_identity, builder_steps = pac0_probe._train_circuit(
        builder_definition, builder_sequences
    )
    builder_train = []
    for task, sequence, steps in zip(training, builder_sequences, builder_steps, strict=True):
        checker = feedback_by_ordinal[task.public.ordinal]
        builder_train.append(
            {
                "task_id": task.public.task_id,
                "entrypoint_slot": task.public.signature.entry_point,
                "feedback_magnitude_slot": _feedback_magnitude_bytes(checker).decode("ascii"),
                "feedback_lineage": "real_visible_identity_checker_uniform_residual",
                "temporal_target_contains_literal_magnitude": False,
                "sequence_token_ids": list(sequence),
                **steps,
            }
        )
    return {
        "protocol": protocol,
        "training": training,
        "target": target,
        "encoder": encoder,
        "episodes": episodes,
        "feedback_by_ordinal": feedback_by_ordinal,
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
    target_episode = prepared["encoder"].encode(target.public)
    _assert_concept_episode(target_episode, target)
    donor_episode = prepared["episodes"][ADD_DONOR_ORDINAL]
    if tuple(target_episode.token_ids) != tuple(donor_episode.token_ids):
        raise RuntimeError("target does not share the donor concept stream")
    return {**prepared, "target_episode": target_episode}


def _observed_surface_token(raw: dict[str, object]) -> int | None:
    generation = raw.get("generation")
    return None if not isinstance(generation, dict) else int(generation["surface_token_id"])


def _expected_source_bytes(
    entrypoint: str, checker: dict[str, object], build_token: int
) -> bytes:
    operator = b"+" if _require_build_token(build_token) == BUILD_ADD_TOKEN_ID else b"-"
    return (
        SOURCE_PREFIX
        + entrypoint.encode("ascii")
        + SOURCE_BEFORE_OPERATOR
        + operator
        + SOURCE_BEFORE_MAGNITUDE
        + _feedback_magnitude_bytes(checker)
        + SOURCE_SUFFIX
    )


def _analyze_builder_run(
    entrypoint: str,
    checker: dict[str, object],
    build_token: int,
    run: dict[str, object],
) -> dict[str, object]:
    emitted = bytes(run["emitted"])
    expected = _expected_source_bytes(entrypoint, checker, build_token)
    source: str | None = None
    compile_valid = False
    function_name: str | None = None
    semantic_exact = False
    namespace: dict[str, object] = {}
    try:
        source = emitted.decode("ascii", errors="strict")
        module = ast.parse(source, filename="<r3-pac1-builder>", mode="exec")
        code = compile(module, "<r3-pac1-builder>", "exec")
        compile_valid = True
        if len(module.body) == 1 and isinstance(module.body[0], ast.FunctionDef):
            function_name = module.body[0].name
        exec(code, {"__builtins__": {}}, namespace)
    except (UnicodeDecodeError, SyntaxError, ValueError, TypeError, NameError):
        pass
    magnitude = int(_feedback_magnitude_bytes(checker).decode("ascii"))
    signed_offset = magnitude if build_token == BUILD_ADD_TOKEN_ID else -magnitude
    function = namespace.get(entrypoint)
    semantic_cases: list[dict[str, object]] = []
    if callable(function):
        semantic_exact = True
        for value in signed_probe.VISIBLE_INPUTS:
            wanted = value + signed_offset
            try:
                observed = function(value)
            except Exception as error:
                observed = None
                error_name: str | None = type(error).__name__
                semantic_exact = False
            else:
                error_name = None
                semantic_exact &= type(observed) is int and observed == wanted
            semantic_cases.append(
                {"input": value, "expected": wanted, "observed": observed, "error": error_name}
            )
    transactions, transaction_structure = role_place_probe._copy_transactions(run["generations"])
    copy_exact = bool(
        transaction_structure
        and [record["copied_places"] for record in transactions]
        == [list(range(len(entrypoint))), list(range(len(_feedback_magnitude_bytes(checker))))]
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
        and function_name == entrypoint
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
        "feedback_magnitude": magnitude,
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


def _create_spend_marker(
    target: GeneratedTask, router_checkpoint_sha256: str, builder_checkpoint_sha256: str
) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-feedback-parameterized-action",
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
        raise RuntimeError("refusing already spent R3-PAC1 target") from error
    return {
        "path": _display_path(SPEND_MARKER_PATH),
        "sha256": _sha256_file(SPEND_MARKER_PATH),
        "exclusive_create": True,
        "flushed_and_fsynced_before_all_target_no_argument_generation_calls": True,
    }


def _score_after_all_target_calls(
    *,
    actual_first: dict[str, object],
    sign_first: dict[str, object],
    actual_second: dict[str, object],
    sign_second: dict[str, object],
    actual_observation: int,
    sign_observation: int,
    actual_feedback: dict[str, object],
    sign_feedback: dict[str, object],
    actual_build: int,
    sign_build: int,
    pre_blocks: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    observation_blocks: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    build_blocks: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    actual_checker: dict[str, object],
    payload_donor_checker: dict[str, object],
    analyses: tuple[dict[str, object], dict[str, object], dict[str, object]],
    visible_checks: dict[str, dict[str, object]],
    sandbox: dict[str, object],
) -> tuple[dict[str, object], dict[str, bool]]:
    router_scored = {
        "actual_first": route_probe._score_generation(actual_first, TEST_TOKEN_ID),
        "sign_first": route_probe._score_generation(sign_first, TEST_TOKEN_ID),
        "actual_second": route_probe._score_generation(actual_second, BUILD_SUB_TOKEN_ID),
        "sign_second": route_probe._score_generation(sign_second, BUILD_ADD_TOKEN_ID),
    }
    actual_pre, sign_pre, payload_pre = pre_blocks
    actual_observation_block, sign_observation_block, payload_observation_block = observation_blocks
    actual_build_block, sign_build_block, payload_build_block = build_blocks
    actual_analysis, sign_analysis, payload_analysis = analyses
    actual_magnitude = _feedback_magnitude_bytes(actual_checker)
    donor_magnitude = _feedback_magnitude_bytes(payload_donor_checker)
    gates = {
        "router_first_generations_bit_exact_test": bool(
            actual_first == sign_first
            and router_scored["actual_first"]["correct"]
            and router_scored["sign_first"]["correct"]
        ),
        "real_negative_and_injected_positive_only_router_divergence": bool(
            actual_observation == OBS_NEG_TOKEN_ID
            and sign_observation == OBS_POS_TOKEN_ID
            and _feedback_matches(actual_feedback, OBS_NEG_TOKEN_ID)
            and _feedback_matches(sign_feedback, OBS_POS_TOKEN_ID)
        ),
        "physical_sub_add_build_elections": bool(
            router_scored["actual_second"]["correct"]
            and router_scored["sign_second"]["correct"]
            and actual_build == BUILD_SUB_TOKEN_ID
            and sign_build == BUILD_ADD_TOKEN_ID
        ),
        "identical_entrypoint_captured_before_execution": bool(
            actual_pre == sign_pre == payload_pre
            and actual_pre == _entrypoint_slot_block(TASK_RECORDS[TARGET_ORDINAL]["entry_point"])
        ),
        "target_checker_bytes_shared_by_actual_and_sign_control": bool(
            actual_magnitude == b"2"
            and actual_observation_block == sign_observation_block
            and actual_observation_block == (
                slot_start_token_id(2), ord("2"), slot_end_token_id(2)
            )
        ),
        "payload_control_uses_real_donor_checker_bytes_only": bool(
            donor_magnitude == b"7"
            and payload_observation_block
            == (slot_start_token_id(2), ord("7"), slot_end_token_id(2))
            and payload_observation_block != actual_observation_block
        ),
        "observed_build_tokens_forwarded_without_host_operator_choice": bool(
            actual_build_block == payload_build_block == (BUILD_SUB_TOKEN_ID,)
            and sign_build_block == (BUILD_ADD_TOKEN_ID,)
        ),
        "actual_direct_n_minus_2": bool(
            actual_analysis.get("passed") is True
            and actual_analysis.get("signed_offset") == -2
            and actual_analysis.get("emitted_sha256") == EXPECTED_SOURCE_SHA256["actual"]
        ),
        "sign_flip_retains_feedback_2_and_emits_plus_2": bool(
            sign_analysis.get("passed") is True
            and sign_analysis.get("signed_offset") == 2
            and sign_analysis.get("feedback_magnitude") == 2
            and sign_analysis.get("emitted_sha256") == EXPECTED_SOURCE_SHA256["sign_intervention"]
        ),
        "payload_flip_retains_sub_and_copies_donor_7": bool(
            payload_analysis.get("passed") is True
            and payload_analysis.get("signed_offset") == -7
            and payload_analysis.get("feedback_magnitude") == 7
            and payload_analysis.get("emitted_sha256") == EXPECTED_SOURCE_SHA256["payload_intervention"]
        ),
        "all_operators_temporal_physical_and_magnitudes_native_copy": all(
            analysis.get("operator_elected_physically") is True
            and analysis.get("copy_transactions_exact") is True
            for analysis in analyses
        ),
        "actual_visible_pass_controls_fail_for_isolated_reasons": bool(
            visible_checks["actual_on_target"].get("passed") is True
            and visible_checks["sign_on_target"].get("passed") is False
            and visible_checks["sign_on_mirror_plus_2"].get("passed") is True
            and visible_checks["payload_on_target"].get("passed") is False
            and visible_checks["payload_on_minus_7"].get("passed") is True
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
    target_entrypoint = target.public.signature.entry_point

    executable = resolve_python_executable(args.python_executable)
    runtime_identity = python_runtime_identity(executable, protocol.raw.get("language"))
    if runtime_identity.get("python_version") != "3.11.15":
        raise RuntimeError("R3-PAC1 requires repo Python 3.11.15")
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = _sha256_file(worker_path)
    worker_source = load_worker_snapshot(worker_sha256)

    router_definition = prepared["router_definition"]
    router_checkpoint = prepared["router_checkpoint"]
    router_identity = prepared["router_checkpoint_identity"]
    builder_definition = prepared["builder_definition"]
    builder_checkpoint = prepared["builder_checkpoint"]
    builder_identity = prepared["builder_checkpoint_identity"]
    target_pre_block = _entrypoint_slot_block(target_entrypoint)

    with route_probe._public_train_authority():
        with ExitStack() as stack:
            actual_router = stack.enter_context(
                NativeTemporalCircuit.from_checkpoint(router_definition, router_checkpoint)
            )
            sign_router = stack.enter_context(
                NativeTemporalCircuit.from_checkpoint(router_definition, router_checkpoint)
            )
            builders = tuple(
                stack.enter_context(
                    NativeTemporalCircuit.from_checkpoint(builder_definition, builder_checkpoint)
                )
                for _ in range(3)
            )
            if (
                actual_router.checkpoint_identity() != router_identity
                or sign_router.checkpoint_identity() != router_identity
                or any(builder.checkpoint_identity() != builder_identity for builder in builders)
            ):
                raise RuntimeError("target checkpoint clones differ")

            actual_router_prompt_steps = actual_router.advance_block(
                tuple(target_episode.token_ids),
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            sign_router_prompt_steps = sign_router.advance_block(
                tuple(target_episode.token_ids),
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            if route_probe._step_summary(actual_router_prompt_steps) != route_probe._step_summary(
                sign_router_prompt_steps
            ):
                raise RuntimeError("router clones received different target prompt traces")

            pre_steps = tuple(
                builder.advance_block(
                    target_pre_block,
                    mode=TemporalStepMode.READ_ONLY,
                    reset_before=True,
                )
                for builder in builders
            )
            if len({json.dumps(route_probe._step_summary(steps), sort_keys=True) for steps in pre_steps}) != 1:
                raise RuntimeError("builder clones differed before identity execution")

            marker = _create_spend_marker(
                target,
                _sha256_bytes(router_checkpoint),
                _sha256_bytes(builder_checkpoint),
            )
            actual_first = route_probe._generate_once(actual_router)
            sign_first = route_probe._generate_once(sign_router)

            # This is the only target sidecar-literal boundary before final scoring.
            target_identity_checker = _run_visible_identity_environment(target_episode, target)
            actual_observation = _sign_observation(target_identity_checker)
            sign_observation = OBS_POS_TOKEN_ID
            actual_feedback = feedback_probe._advance_feedback(actual_router, actual_observation)
            sign_feedback = feedback_probe._advance_feedback(sign_router, sign_observation)
            actual_second = route_probe._generate_once(actual_router)
            sign_second = route_probe._generate_once(sign_router)
            actual_build = _require_build_token(_observed_surface_token(actual_second))
            sign_build = _require_build_token(_observed_surface_token(sign_second))

            payload_donor_checker = prepared["feedback_by_ordinal"][PAYLOAD_CONTROL_DONOR_ORDINAL]
            observation_blocks = (
                _feedback_slot_block(target_identity_checker),
                _feedback_slot_block(target_identity_checker),
                _feedback_slot_block(payload_donor_checker),
            )
            build_blocks = (
                (actual_build,),
                (sign_build,),
                (actual_build,),
            )
            observation_steps = tuple(
                builder.advance_block(
                    block,
                    mode=TemporalStepMode.READ_ONLY,
                    reset_before=False,
                )
                for builder, block in zip(builders, observation_blocks, strict=True)
            )
            build_steps = tuple(
                builder.advance_block(
                    block,
                    mode=TemporalStepMode.READ_ONLY,
                    reset_before=False,
                )
                for builder, block in zip(builders, build_blocks, strict=True)
            )
            builder_runs = tuple(
                role_place_probe._generate_teacher_free(builder) for builder in builders
            )

    analyses = (
        _analyze_builder_run(
            target_entrypoint, target_identity_checker, actual_build, builder_runs[0]
        ),
        _analyze_builder_run(
            target_entrypoint, target_identity_checker, sign_build, builder_runs[1]
        ),
        _analyze_builder_run(
            target_entrypoint, payload_donor_checker, actual_build, builder_runs[2]
        ),
    )
    sources = tuple(analysis.get("source") for analysis in analyses)
    if not all(isinstance(source, str) for source in sources):
        raise RuntimeError("builder output did not decode to source")
    actual_offset = _uniform_signed_residual(target_identity_checker)
    mirror_offset = int(_feedback_magnitude_bytes(target_identity_checker).decode("ascii"))
    payload_offset = _uniform_signed_residual(payload_donor_checker)
    visible_checks = {
        "actual_on_target": signed_probe._visible_checker(
            sources[0], target_entrypoint, actual_offset
        ),
        "sign_on_target": signed_probe._visible_checker(
            sources[1], target_entrypoint, actual_offset
        ),
        "sign_on_mirror_plus_2": signed_probe._visible_checker(
            sources[1], target_entrypoint, mirror_offset
        ),
        "payload_on_target": signed_probe._visible_checker(
            sources[2], target_entrypoint, actual_offset
        ),
        "payload_on_minus_7": signed_probe._visible_checker(
            sources[2], target_entrypoint, payload_offset
        ),
    }
    sandbox = action_probe._sandbox_record(
        sources[0],
        target,
        protocol,
        python_executable=executable,
        runtime_identity=runtime_identity,
        worker_source=worker_source,
        worker_sha256=worker_sha256,
    )

    router_scored, gates = _score_after_all_target_calls(
        actual_first=actual_first,
        sign_first=sign_first,
        actual_second=actual_second,
        sign_second=sign_second,
        actual_observation=actual_observation,
        sign_observation=sign_observation,
        actual_feedback=actual_feedback,
        sign_feedback=sign_feedback,
        actual_build=actual_build,
        sign_build=sign_build,
        pre_blocks=(target_pre_block, target_pre_block, target_pre_block),
        observation_blocks=observation_blocks,
        build_blocks=build_blocks,
        actual_checker=target_identity_checker,
        payload_donor_checker=payload_donor_checker,
        analyses=analyses,
        visible_checks=visible_checks,
        sandbox=sandbox,
    )
    passed = all(gates.values())
    outcome = (
        "PASS_R3_PAC1_EXECUTION_FEEDBACK_CONSTRUCTION"
        if passed
        else "STOP_R3_PAC1_EXECUTION_FEEDBACK_CONSTRUCTION"
    )
    native_identity = loaded_library_identity()
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-feedback-parameterized-action",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": outcome,
        "claim_boundary": (
            "Public TRAIN only: a real identity execution yields a signed uniform residual; only its sign enters a fitted-cell router, while its absolute decimal bytes enter a native role/place observation slot after execution. The physical BUILD election and physical slot COPY then emit one exact fixed-skeleton scalar source. This is narrow execution-feedback parameterized construction, not arbitrary numeric inference, AST synthesis, semantic diagnosis, transfer, DEV/eval, held-out, M0, or C1 capability."
        ),
        "limitations": {
            "fixed_source_skeleton": "def <copied-entrypoint>(n): return n <cell-elected-operator> <copied-feedback-magnitude>",
            "fixed_parameter": "n",
            "host_environment_formatting": "Python validates a real visible checker, computes abs(uniform residual), and serializes canonical decimal bytes; it does not select target operator/source or copy a sidecar magnitude.",
            "target_sidecar_magnitude_fed_to_builder": False,
            "target_sidecar_offset_scope": "visible public environment checker and final posthoc scoring only",
            "donor_reuse": "public TRAIN scalar-offset /000 and /001 are reused as donors in fresh circuits",
            "numeral_2_claimed_novel_corpus_wide": False,
            "fresh_target_claimed": False,
        },
        "authority": {
            "partition": "train",
            "target_task_id": target.public.task_id,
            "historical_prompt_baseline_exposure": True,
            "historical_task_content_drift": True,
            "freshness_claimed": False,
            "no_prior_native_or_routed_target_artifact_or_spend_marker": True,
            "router_target_checkpoint_clones": 2,
            "builder_target_checkpoint_clones": 3,
            "router_target_no_argument_generation_calls": 4,
            "builder_target_no_argument_generation_calls": sum(
                len(run["generations"]) for run in builder_runs
            ),
            "canonical_target_sandbox_calls": 1,
            "model_retries": 0,
            "harness_retries": 0,
            "hidden_cases_or_verdicts_entered_cell_runtime": False,
            "expected_routes_and_sources_scored_only_after_all_target_generate_calls": True,
            "python_bridge": "forward observed BUILD token unchanged and checker-derived magnitude slot bytes",
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
        "timeline": {
            "builder_slot0_captured_before_target_identity_execution": True,
            "target_identity_checker": target_identity_checker,
            "builder_slot2_appended_after_execution_without_reset": True,
            "physical_build_appended_after_slot2_without_reset": True,
        },
        "router_target": {
            "actual_prompt": route_probe._step_summary(actual_router_prompt_steps),
            "sign_prompt": route_probe._step_summary(sign_router_prompt_steps),
            "actual_observation_token_id": actual_observation,
            "sign_intervention_observation_token_id": sign_observation,
            "actual_feedback": actual_feedback,
            "sign_feedback": sign_feedback,
            "actual_first_generation": actual_first,
            "sign_first_generation": sign_first,
            "actual_second_generation": actual_second,
            "sign_second_generation": sign_second,
            "actual_observed_build_token_id": actual_build,
            "sign_observed_build_token_id": sign_build,
            "scored_after_all_target_calls": router_scored,
        },
        "builder_target": {
            "pre_execution_steps": [route_probe._step_summary(steps) for steps in pre_steps],
            "post_execution_observation_steps": [
                route_probe._step_summary(steps) for steps in observation_steps
            ],
            "physical_build_input_steps": [
                route_probe._step_summary(steps) for steps in build_steps
            ],
            "actual": _serializable_builder_run(builder_runs[0], analyses[0]),
            "sign_intervention": _serializable_builder_run(builder_runs[1], analyses[1]),
            "payload_intervention": _serializable_builder_run(builder_runs[2], analyses[2]),
            "payload_intervention_donor_checker": payload_donor_checker,
        },
        "visible_posthoc_checks": visible_checks,
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
                source: _sha256_file(REPO_ROOT / source) for source in PINNED_SOURCE_SHA256
            },
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "outcome": outcome,
                "artifact": _display_path(OUTPUT_PATH),
                "gates": gates,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
