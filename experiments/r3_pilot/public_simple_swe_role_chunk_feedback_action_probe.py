"""One-shot visible-TRAIN binary-feedback action-routing probe.

Two scalar-nonnegative normalized donor episodes teach a fresh native circuit
``TEST -> OBS_FAIL -> FIX`` and ``TEST -> OBS_PASS -> DONE`` continuations.
Checkpoint clones then receive the same /007 prompt.  Their emitted actions
alone mutate two supplied Layer-0 environments, one public diagnostic bit is
fed back, and both circuits generate once more.  Hidden TRAIN cases are used
exactly once, after all four native calls, only to score the actual environment.

The /007 prompt was already disclosed to the historical prompt-only baseline;
this experiment therefore makes no freshness claim.  Importing or compiling
this module does not spend the target.
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

import public_simple_swe_role_chunk_layer0_action_probe as action_probe
import public_simple_swe_role_chunk_route_probe as route_probe
from humanbrain.cortex import seeded_threshold_offsets_mv
from humanbrain.native import loaded_library_identity
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitConfig,
    NativeTemporalCircuitDefinition,
    NativeTemporalRuntimeError,
    TemporalContextPolicy,
    TemporalPlasticityPolicy,
    TemporalStepMode,
)
from humanbrain.program import Layer0ProgramCompiler, Layer0ProgramGrammar
from humanbrain.role_chunk_episode import RoleChunkEpisode
from humanbrain.sdr_union import token_sdr_indices
from humanbrain.simple_swe.sandbox import (
    load_worker_snapshot,
    python_runtime_identity,
    resolve_python_executable,
)
from humanbrain.simple_swe.tasks import GeneratedTask, generate_visible_partition


OUTPUT_PATH = Path(__file__).with_suffix(".json")
SPEND_MARKER_PATH = Path(__file__).with_name(
    f"{Path(__file__).stem}.t0_scalar_nonnegative-007.spent.json"
)
PREVIOUS_ACTION_SOURCE_PATH = Path(__file__).with_name(
    "public_simple_swe_role_chunk_layer0_action_probe.py"
)
PREVIOUS_ACTION_ARTIFACT_PATH = Path(__file__).with_name(
    "public_simple_swe_role_chunk_layer0_action_probe.json"
)
HISTORICAL_BASELINE_PATH = REPO_ROOT / "artifacts" / "simple_swe_public_baseline_v1.json"

TARGET_FAMILY = "t0_scalar_nonnegative"
TARGET_ORDINAL = 7
TARGET_TASK_ID = "simple-swe-v1/train/tier-0/t0_scalar_nonnegative/007"
TARGET_TASK_SHA256 = "395b07bb304adbe334b8c4be16179715028c62b544e3225b144c7f355373d060"
TARGET_HIDDEN_TEST_SHA256 = (
    "a080bcdf2a33fb51357a204e73311c0b505d3200b570552a95df76093f855f3d"
)
TARGET_PROMPT_SHA256 = (
    "be6031ff7d52cd42bb249698061260f342277868f4d5ede0cc8d8da9947b69b3"
)
TARGET_STRUCTURAL_SHA256 = (
    "736854ba550d057d152c9701e65fe4a6d4528d4d0581db6454333e487bd3f899"
)
TARGET_CONCEPT_STREAM_SHA256 = (
    "c4fc0c6d9eb257e2655453fb9c644ce7d82465bfe54a512c028fcb5dba3c42c7"
)

TEST_TOKEN_ID = 1540
DONE_TOKEN_ID = 1541
FIX_TOKEN_ID = 1536
OBS_FAIL_TOKEN_ID = 1600
OBS_PASS_TOKEN_ID = 1601
ACTION_NAME_BY_TOKEN = {
    TEST_TOKEN_ID: "TEST",
    DONE_TOKEN_ID: "DONE",
    FIX_TOKEN_ID: "FIX",
}
FIX_LAYER0_TOKENS = ("MAX", "ZERO", "ARG0", "END")
IDENTITY_LAYER0_TOKENS = ("ARG0", "END")
PUBLIC_DIAGNOSTIC_INPUT = -7
PUBLIC_DIAGNOSTIC_EXPECTED = 0

COLUMN_COUNT = 4_096
ACTIVE_COLUMNS = 11
CODEBOOK_SEED = 0
CONSTRUCTION_SEED = 17
BASE_CELLS_PER_COLUMN = 8
THETA_MATCH = 16
THETA_ACTIVE = 18

EXPECTED_PREVIOUS_ACTION_SOURCE_SHA256 = (
    "8c2c86c1cdc6026639cdbce0ba88ac5145101c2c2dced8c3a56155f72326f3d1"
)
EXPECTED_PREVIOUS_ACTION_ARTIFACT_SHA256 = (
    "6f2cf848dd7d7eadc156dd078e554c20e815e0871f8e74d9a016c6a37c0f8302"
)
EXPECTED_HISTORICAL_BASELINE_SHA256 = (
    "524c9df0971b1684e22398866fef0108f48fd40ab3fe8caeb221196c1e753f8b"
)

PINNED_SOURCE_SHA256 = {
    "experiments/r3_pilot/public_simple_swe_role_chunk_layer0_action_probe.py": EXPECTED_PREVIOUS_ACTION_SOURCE_SHA256,
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


def _nested_task_records(value: object) -> tuple[dict[str, object], ...]:
    found: list[dict[str, object]] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if item.get("task_id") == TARGET_TASK_ID:
                found.append(item)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return tuple(found)


def _assert_anchors() -> dict[str, object]:
    if _sha256_file(PREVIOUS_ACTION_SOURCE_PATH) != EXPECTED_PREVIOUS_ACTION_SOURCE_SHA256:
        raise RuntimeError("previous Layer-0 action source changed")
    if (
        _sha256_file(PREVIOUS_ACTION_ARTIFACT_PATH)
        != EXPECTED_PREVIOUS_ACTION_ARTIFACT_SHA256
    ):
        raise RuntimeError("previous Layer-0 action result changed")
    previous = json.loads(PREVIOUS_ACTION_ARTIFACT_PATH.read_text(encoding="utf-8"))
    if (
        previous.get("outcome") != "PASS_ROLE_ROUTE_LAYER0_ACTION"
        or not isinstance(previous.get("gates"), dict)
        or not all(previous["gates"].values())
        or previous.get("runtime", {}).get("probe_source_sha256")
        != EXPECTED_PREVIOUS_ACTION_SOURCE_SHA256
    ):
        raise RuntimeError("previous Layer-0 action anchor is not an exact PASS")

    if _sha256_file(HISTORICAL_BASELINE_PATH) != EXPECTED_HISTORICAL_BASELINE_SHA256:
        raise RuntimeError("historical prompt-baseline artifact changed")
    baseline = json.loads(HISTORICAL_BASELINE_PATH.read_text(encoding="utf-8"))
    disclosures = _nested_task_records(baseline)
    if not disclosures or any(
        record.get("prompt_sha256") != TARGET_PROMPT_SHA256 for record in disclosures
    ):
        raise RuntimeError("historical baseline disclosure no longer matches /007 prompt")
    return {
        "previous_action_outcome": previous["outcome"],
        "previous_action_artifact_sha256": EXPECTED_PREVIOUS_ACTION_ARTIFACT_SHA256,
        "previous_action_source_sha256": EXPECTED_PREVIOUS_ACTION_SOURCE_SHA256,
        "historical_baseline_sha256": EXPECTED_HISTORICAL_BASELINE_SHA256,
        "historical_prompt_disclosure_count": len(disclosures),
        "freshness_claimed": False,
    }


def _assert_frozen_sources() -> None:
    for source, expected in PINNED_SOURCE_SHA256.items():
        if _sha256_file(REPO_ROOT / source) != expected:
            raise RuntimeError(f"frozen feedback-action dependency changed: {source}")


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
        raise RuntimeError("refusing spent role-chunk feedback-action probe")
    _assert_anchors()
    _assert_frozen_sources()
    paths = (
        Path(__file__).resolve(),
        PREVIOUS_ACTION_ARTIFACT_PATH,
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
            raise RuntimeError(f"feedback-action source must be committed: {relative}")


def _select_target(protocol: object) -> GeneratedTask:
    matches = tuple(
        task
        for task in generate_visible_partition(protocol, "train")
        if task.public.family == TARGET_FAMILY and task.public.ordinal == TARGET_ORDINAL
    )
    if len(matches) != 1:
        raise RuntimeError("feedback target selection is not unique")
    target = matches[0]
    public = target.public
    if (
        public.task_id != TARGET_TASK_ID
        or public.partition != "train"
        or public.task_sha256 != TARGET_TASK_SHA256
        or public.hidden_test_sha256 != TARGET_HIDDEN_TEST_SHA256
        or hashlib.sha256(public.prompt.encode("utf-8")).hexdigest()
        != TARGET_PROMPT_SHA256
    ):
        raise RuntimeError("public /007 target identity changed")
    return target


def _training_sequences(
    donors: tuple[GeneratedTask, ...],
    concept_streams: dict[tuple[str, int], tuple[int, ...]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    indexed = {(task.public.family, task.public.ordinal): task for task in donors}
    keys = ((TARGET_FAMILY, 5), (TARGET_FAMILY, 6))
    if set(indexed) & set(keys) != set(keys):
        raise RuntimeError("scalar normalized donors are missing")
    fail_sequence = (
        *concept_streams[keys[0]],
        TEST_TOKEN_ID,
        OBS_FAIL_TOKEN_ID,
        FIX_TOKEN_ID,
    )
    pass_sequence = (
        *concept_streams[keys[1]],
        TEST_TOKEN_ID,
        OBS_PASS_TOKEN_ID,
        DONE_TOKEN_ID,
    )
    return fail_sequence, pass_sequence


def _definition(
    sequences: tuple[tuple[int, ...], tuple[int, ...]],
    target_stream: tuple[int, ...],
) -> tuple[NativeTemporalCircuitDefinition, dict[str, int]]:
    noninitial_arrivals = sum(len(sequence) - 1 for sequence in sequences)
    segment_capacity = ACTIVE_COLUMNS * noninitial_arrivals
    base_population = COLUMN_COUNT * BASE_CELLS_PER_COLUMN
    config = NativeTemporalCircuitConfig(
        column_count=COLUMN_COUNT,
        max_pyramidal_cells=base_population + segment_capacity,
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
    used_token_ids = tuple(
        sorted(
            {
                *(token for sequence in sequences for token in sequence),
                *target_stream,
                TEST_TOKEN_ID,
                DONE_TOKEN_ID,
                FIX_TOKEN_ID,
                OBS_FAIL_TOKEN_ID,
                OBS_PASS_TOKEN_ID,
            }
        )
    )
    token_codes = {
        token: tuple(
            int(column)
            for column in token_sdr_indices(
                token,
                n_bits=COLUMN_COUNT,
                w_ones=ACTIVE_COLUMNS,
                seed=CODEBOOK_SEED,
            )
        )
        for token in used_token_ids
    }
    if any(len(columns) != ACTIVE_COLUMNS for columns in token_codes.values()):
        raise RuntimeError("feedback/action token codes are not equal width")
    physical_cell_count = 2 * config.max_pyramidal_cells + 3 * config.column_count
    definition = NativeTemporalCircuitDefinition(
        config,
        token_codes,
        seeded_threshold_offsets_mv(
            physical_cell_count, config.construction_seed
        ).tolist(),
    )
    return definition, {
        "training_sequence_count": len(sequences),
        "noninitial_arrivals": noninitial_arrivals,
        "segment_capacity": segment_capacity,
        "max_pyramidal_cells": config.max_pyramidal_cells,
    }


def _prepare_probe() -> tuple[
    object,
    GeneratedTask,
    RoleChunkEpisode,
    NativeTemporalCircuitDefinition,
    bytes,
    object,
    dict[str, object],
    dict[str, int],
]:
    protocol, donors, _parent_probe_records = route_probe._select_tasks()
    target = _select_target(protocol)
    encoder, concept_streams, _base_streams, _records = (
        route_probe._prepare_representations(donors, (target,))
    )
    episode = encoder.encode(target.public)
    if (
        episode.structural_sha256 != TARGET_STRUCTURAL_SHA256
        or _sha256_json(tuple(episode.token_ids)) != TARGET_CONCEPT_STREAM_SHA256
        or tuple(episode.token_ids)
        != concept_streams[(TARGET_FAMILY, route_probe.DONOR_ORDINALS[0])]
    ):
        raise RuntimeError("/007 role-chunk representation differs from scalar donors")

    sequences = _training_sequences(donors, concept_streams)
    definition, capacity = _definition(sequences, tuple(episode.token_ids))
    train_episodes: list[dict[str, object]] = []
    with route_probe._public_train_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            for ordinal, sequence in zip((5, 6), sequences, strict=True):
                steps = circuit.advance_block(
                    sequence, mode=TemporalStepMode.LEARN, reset_before=True
                )
                train_episodes.append(
                    {
                        "task_id": f"simple-swe-v1/train/tier-0/{TARGET_FAMILY}/{ordinal:03d}",
                        "sequence_token_ids": list(sequence),
                        **route_probe._step_summary(steps),
                    }
                )
            checkpoint_identity = circuit.checkpoint_identity()
            checkpoint = circuit.checkpoint()
    return (
        protocol,
        target,
        episode,
        definition,
        checkpoint,
        checkpoint_identity,
        {
            "episodes": train_episodes,
            "checkpoint_sha256": _sha256_bytes(checkpoint),
            "checkpoint_identity": asdict(checkpoint_identity),
        },
        capacity,
    )


def _compile_environment_source(
    tokens: tuple[str, ...], episode: RoleChunkEpisode, target: GeneratedTask
) -> str:
    signature = action_probe._layer0_signature(episode, target.public)
    max_nodes = 3 if tokens == FIX_LAYER0_TOKENS else 1
    return Layer0ProgramCompiler(Layer0ProgramGrammar(max_nodes=max_nodes)).compile(
        tokens, signature
    )


def _observed_surface_token(raw: dict[str, object]) -> int | None:
    generation = raw.get("generation")
    if not isinstance(generation, dict):
        return None
    return int(generation["surface_token_id"])


def _dispatch_observed_action(
    observed_token: int | None,
    source: str,
    episode: RoleChunkEpisode,
    target: GeneratedTask,
) -> tuple[str, dict[str, object]]:
    action_name = ACTION_NAME_BY_TOKEN.get(observed_token)
    if action_name is None:
        return source, {
            "observed_token_id": observed_token,
            "dispatched": False,
            "action": None,
            "source_changed": False,
            "source_sha256_after": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        }
    updated = source
    if action_name == "FIX":
        compiled = action_probe._compile_action(
            FIX_LAYER0_TOKENS, episode, target.public
        )
        updated = str(compiled["source"])
    return updated, {
        "observed_token_id": observed_token,
        "dispatched": True,
        "action": action_name,
        "source_changed": updated != source,
        "source_sha256_after": hashlib.sha256(updated.encode("utf-8")).hexdigest(),
    }


def _run_public_diagnostic(source: str, entry_point: str) -> dict[str, object]:
    namespace: dict[str, object] = {"__builtins__": {"max": max}}
    try:
        exec(compile(source, "<public-feedback-diagnostic>", "exec"), namespace)
        function = namespace[entry_point]
        observed = function(PUBLIC_DIAGNOSTIC_INPUT)  # type: ignore[operator]
    except Exception as error:
        return {
            "input": PUBLIC_DIAGNOSTIC_INPUT,
            "expected": PUBLIC_DIAGNOSTIC_EXPECTED,
            "observed": None,
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "input": PUBLIC_DIAGNOSTIC_INPUT,
        "expected": PUBLIC_DIAGNOSTIC_EXPECTED,
        "observed": observed,
        "passed": observed == PUBLIC_DIAGNOSTIC_EXPECTED,
        "error": None,
    }


def _observation_token(diagnostic: dict[str, object]) -> int:
    if diagnostic.get("error") is not None:
        raise RuntimeError("public diagnostic error cannot become feedback")
    if (
        diagnostic.get("input") != PUBLIC_DIAGNOSTIC_INPUT
        or diagnostic.get("expected") != PUBLIC_DIAGNOSTIC_EXPECTED
        or type(diagnostic.get("observed")) is not int
    ):
        raise RuntimeError("public diagnostic did not produce a genuine scalar result")
    equality = diagnostic["observed"] == PUBLIC_DIAGNOSTIC_EXPECTED
    if diagnostic.get("passed") is not equality:
        raise RuntimeError("public diagnostic equality flag is inconsistent")
    return OBS_PASS_TOKEN_ID if equality else OBS_FAIL_TOKEN_ID


def _advance_feedback(
    circuit: NativeTemporalCircuit, observation_token: int
) -> dict[str, object]:
    try:
        steps = circuit.advance_block(
            (observation_token,),
            mode=TemporalStepMode.READ_ONLY,
            reset_before=False,
        )
    except NativeTemporalRuntimeError as error:
        return {
            "observation_token_id": observation_token,
            "advanced": False,
            "native_stop": {
                "status": error.status.name,
                "message": error.message,
                "required": error.required,
                "observed": error.observed,
                "location": error.location,
                "poison_code": error.poison_code,
                "stage_code": error.stage_code,
            },
        }
    return {
        "observation_token_id": observation_token,
        "advanced": True,
        "native_stop": None,
        **route_probe._step_summary(steps),
    }


def _create_spend_marker(target: GeneratedTask, checkpoint_sha256: str) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-feedback-action",
        "state": "TARGET_ATTEMPT_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target_task_id": target.public.task_id,
        "family": target.public.family,
        "ordinal": target.public.ordinal,
        "checkpoint_sha256": checkpoint_sha256,
        "actual_and_pass_control_checkpoint_clones": 2,
        "atomic_no_argument_generation_calls_planned": 4,
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
        raise RuntimeError("refusing already spent feedback-action target") from error
    return {
        "path": _display_path(SPEND_MARKER_PATH),
        "sha256": _sha256_file(SPEND_MARKER_PATH),
        "exclusive_create": True,
        "flushed_and_fsynced_before_all_four_generate_calls": True,
    }


def _physical_score(raw: dict[str, object], expected: int) -> dict[str, object]:
    return route_probe._score_generation(raw, expected)


def _score_after_all_calls(
    *,
    actual_first: dict[str, object],
    control_first: dict[str, object],
    actual_second: dict[str, object],
    control_second: dict[str, object],
    actual_first_dispatch: dict[str, object],
    control_first_dispatch: dict[str, object],
    actual_second_dispatch: dict[str, object],
    control_second_dispatch: dict[str, object],
    actual_diagnostic: dict[str, object],
    control_diagnostic: dict[str, object],
    sandbox: dict[str, object],
) -> tuple[dict[str, object], dict[str, bool]]:
    scored = {
        "actual_first": _physical_score(actual_first, TEST_TOKEN_ID),
        "control_first": _physical_score(control_first, TEST_TOKEN_ID),
        "actual_second": _physical_score(actual_second, FIX_TOKEN_ID),
        "control_second": _physical_score(control_second, DONE_TOKEN_ID),
    }
    dispatches = (
        actual_first_dispatch,
        control_first_dispatch,
        actual_second_dispatch,
        control_second_dispatch,
    )
    expected_dispatches = (
        (TEST_TOKEN_ID, "TEST"),
        (TEST_TOKEN_ID, "TEST"),
        (FIX_TOKEN_ID, "FIX"),
        (DONE_TOKEN_ID, "DONE"),
    )
    gates = {
        "first_clone_generations_bit_exact": actual_first == control_first,
        "physical_test_test_first": bool(
            scored["actual_first"]["correct"]
            and scored["control_first"]["correct"]
        ),
        "public_fail_pass_feedback": bool(
            actual_diagnostic.get("passed") is False
            and control_diagnostic.get("passed") is True
        ),
        "physical_fix_done_second": bool(
            scored["actual_second"]["correct"]
            and scored["control_second"]["correct"]
        ),
        "exact_observed_route_dispatch": all(
            dispatch.get("dispatched") is True
            and dispatch.get("observed_token_id") == token
            and dispatch.get("action") == action
            for dispatch, (token, action) in zip(
                dispatches, expected_dispatches, strict=True
            )
        ),
        "actual_changed_once_control_unchanged": bool(
            sum(
                dispatch.get("source_changed") is True
                for dispatch in (actual_first_dispatch, actual_second_dispatch)
            )
            == 1
            and all(
                dispatch.get("source_changed") is False
                for dispatch in (control_first_dispatch, control_second_dispatch)
            )
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
        target,
        episode,
        definition,
        checkpoint,
        checkpoint_identity,
        train_record,
        capacity,
    ) = _prepare_probe()

    executable = resolve_python_executable(args.python_executable)
    runtime_identity = python_runtime_identity(executable, protocol.raw.get("language"))
    if runtime_identity.get("python_version") != "3.11.15":
        raise RuntimeError("feedback-action probe requires repo Python 3.11.15")
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = _sha256_file(worker_path)
    worker_source = load_worker_snapshot(worker_sha256)

    actual_source = _compile_environment_source(
        IDENTITY_LAYER0_TOKENS, episode, target
    )
    control_source = _compile_environment_source(FIX_LAYER0_TOKENS, episode, target)
    signature = action_probe._layer0_signature(episode, target.public)

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
                raise RuntimeError("feedback-action checkpoint clone differs")
            actual_prompt = actual.advance_block(
                tuple(episode.token_ids),
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            control_prompt = control.advance_block(
                tuple(episode.token_ids),
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            if route_probe._step_summary(actual_prompt) != route_probe._step_summary(
                control_prompt
            ):
                raise RuntimeError("checkpoint clones received different prompt traces")

            marker = _create_spend_marker(target, _sha256_bytes(checkpoint))
            actual_first = route_probe._generate_once(actual)
            control_first = route_probe._generate_once(control)

            actual_source, actual_first_dispatch = _dispatch_observed_action(
                _observed_surface_token(actual_first), actual_source, episode, target
            )
            control_source, control_first_dispatch = _dispatch_observed_action(
                _observed_surface_token(control_first), control_source, episode, target
            )
            actual_diagnostic = _run_public_diagnostic(
                actual_source, signature.entry_point
            )
            control_diagnostic = _run_public_diagnostic(
                control_source, signature.entry_point
            )
            actual_observation = _observation_token(actual_diagnostic)
            control_observation = _observation_token(control_diagnostic)
            actual_feedback = _advance_feedback(actual, actual_observation)
            control_feedback = _advance_feedback(control, control_observation)

            actual_second = route_probe._generate_once(actual)
            control_second = route_probe._generate_once(control)

    actual_source, actual_second_dispatch = _dispatch_observed_action(
        _observed_surface_token(actual_second), actual_source, episode, target
    )
    control_source, control_second_dispatch = _dispatch_observed_action(
        _observed_surface_token(control_second), control_source, episode, target
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

    scored_calls, gates = _score_after_all_calls(
        actual_first=actual_first,
        control_first=control_first,
        actual_second=actual_second,
        control_second=control_second,
        actual_first_dispatch=actual_first_dispatch,
        control_first_dispatch=control_first_dispatch,
        actual_second_dispatch=actual_second_dispatch,
        control_second_dispatch=control_second_dispatch,
        actual_diagnostic=actual_diagnostic,
        control_diagnostic=control_diagnostic,
        sandbox=sandbox,
    )
    passed = all(gates.values())
    outcome = (
        "PASS_REPETITION_CONDITIONED_BINARY_FEEDBACK_ACTION"
        if passed
        else "STOP_REPETITION_CONDITIONED_BINARY_FEEDBACK_ACTION"
    )
    native_identity = loaded_library_identity()
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-feedback-action",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": outcome,
        "claim_boundary": (
            "One visible-TRAIN normalized-template replay routes one frozen public diagnostic bit to a fixed Layer-0 actuator. "
            "This is repetition-conditioned binary feedback routing, not semantic interpretation, self-correction, transfer, code generation, DEV/eval, held-out, M0, or C1 capability."
        ),
        "authority": {
            "partition": "train",
            "target_task_id": target.public.task_id,
            "historical_prompt_baseline_reuse": True,
            "freshness_claimed": False,
            "identical_prompt_presentations": 2,
            "checkpoint_clones": 2,
            "atomic_no_argument_generation_calls": 4,
            "canonical_target_sandbox_calls": 1,
            "model_retries": 0,
            "harness_retries": 0,
            "expected_action_labels_used_as_runtime_input": False,
            "hidden_cases_or_verdicts_entered_cell_runtime": False,
            "hidden_cases_used_once_only_after_all_generation_calls": True,
            "preexisting_durable_marker_inventory": marker_inventory,
        },
        "anchors": anchors,
        "target": {
            "task_id": target.public.task_id,
            "task_sha256": target.public.task_sha256,
            "hidden_test_sha256": target.public.hidden_test_sha256,
            "prompt_sha256": TARGET_PROMPT_SHA256,
            "structural_sha256": episode.structural_sha256,
            "concept_stream_sha256": _sha256_json(tuple(episode.token_ids)),
            "concept_prompt_events": len(episode.token_ids),
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
            "initial_source_kind": "provided_identity",
            "first_raw_generation": actual_first,
            "first_dispatch": actual_first_dispatch,
            "public_diagnostic": actual_diagnostic,
            "observation_token_id": actual_observation,
            "feedback_step": actual_feedback,
            "second_raw_generation": actual_second,
            "second_dispatch": actual_second_dispatch,
            "final_source": actual_source,
            "final_source_sha256": hashlib.sha256(
                actual_source.encode("utf-8")
            ).hexdigest(),
        },
        "pass_control": {
            "initial_source_kind": "provided_correct_max",
            "first_raw_generation": control_first,
            "first_dispatch": control_first_dispatch,
            "public_diagnostic": control_diagnostic,
            "observation_token_id": control_observation,
            "feedback_step": control_feedback,
            "second_raw_generation": control_second,
            "second_dispatch": control_second_dispatch,
            "final_source": control_source,
            "final_source_sha256": hashlib.sha256(
                control_source.encode("utf-8")
            ).hexdigest(),
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
