"""One-shot visible-TRAIN R3-SR0 normalized-template route probe.

Eight donor prompts (two repeated payload-normalized templates per family)
teach four private route tokens.  Exactly one /008 prompt per family is then
presented to four matched native arms: concept/base tokenization crossed with
true/conflicting labels.  The probe claims only repetition-conditioned
normalized-template routing, never structural generalization or code solving.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from humanbrain.concept_tokens import CONCEPT_ID_BASE, MAX_CONCEPTS
from humanbrain.cortex import seeded_threshold_offsets_mv
from humanbrain.native import loaded_library_identity
import humanbrain.native_temporal_runtime as runtime_adapter
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitConfig,
    NativeTemporalCircuitDefinition,
    NativeTemporalRuntimeError,
    TemporalContextPolicy,
    TemporalGenerationPath,
    TemporalPlasticityPolicy,
    TemporalStepMode,
)
from humanbrain.role_chunk_episode import (
    DEFAULT_ROLE_CHUNK_CONCEPT_CONFIG,
    RoleChunkEpisodeEncoder,
)
from humanbrain.sdr_union import token_sdr_indices
from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.tasks import GeneratedTask, PublicTask, generate_visible_partition
from humanbrain.tokenizer import TokenizerConfig


OUTPUT_PATH = Path(__file__).with_suffix(".json")
PROTOCOL_PATH = REPO_ROOT / "benchmarks" / "simple_swe_protocol_v1.json"
EXPECTED_PROTOCOL_ID = "human-cell-simple-swe-v1"
EXPECTED_PROTOCOL_SHA256 = (
    "f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2"
)

FAMILY_ORDER = (
    "t0_scalar_nonnegative",
    "t0_scalar_offset",
    "t1_list_map_abs",
    "t1_list_map_add",
)
DONOR_ORDINALS = (5, 6)
PROBE_ORDINAL = 8
UNUSED_FOLLOW_ON_ORDINAL = 9
ARM_ORDER = (
    "concept_true",
    "concept_conflict",
    "base_true",
    "base_conflict",
)
ROUTE_TOKEN_BASE = CONCEPT_ID_BASE + MAX_CONCEPTS
ROUTE_TOKEN_IDS = tuple(ROUTE_TOKEN_BASE + index for index in range(len(FAMILY_ORDER)))
ROUTE_BY_FAMILY = dict(zip(FAMILY_ORDER, ROUTE_TOKEN_IDS, strict=True))

COLUMN_COUNT = 4_096
ACTIVE_COLUMNS = 11
CODEBOOK_SEED = 0
CONSTRUCTION_SEED = 17
BASE_CELLS_PER_COLUMN = 8
THETA_MATCH = 16
THETA_ACTIVE = 18
BASE_TOKENIZER_CONFIG = TokenizerConfig(max_phrases=512, min_frequency=2)

SPEND_MARKER_PATHS = {
    family: Path(__file__).with_name(
        f"{Path(__file__).stem}.{family}-{PROBE_ORDINAL:03d}.spent.json"
    )
    for family in FAMILY_ORDER
}

PINNED_SOURCE_SHA256 = {
    "src/humanbrain/role_chunk_episode.py": "8dc8a9102569f6bdc3135479bd38f8074993efc2a93234202fcf4e96d65940f5",
    "src/humanbrain/concept_tokens.py": "9070deaf7d9a8a54add24e3a213d8b6aa7bbb24b3d35ad66a7b68b3d1148f43e",
    "src/humanbrain/tokenizer.py": "08ef114a758006868944ad3091ca66619d3c082086ea2d44d93211d82a9791ce",
    "src/humanbrain/cortex.py": "7a7704a14324203d2cce427a11fbd58ffbbc345185ba34633449f59cf294624d",
    "src/humanbrain/sdr_union.py": "1de7f6edd914584c4c72d7e42426254590049cfb69cbbfa730afa3179afa63f9",
    "src/humanbrain/simple_swe/tasks.py": "d247bb1a81dca47846a787292a8a3f29c01938771070517b6fa35c9d8f7eed23",
    "src/humanbrain/native_temporal_runtime.py": "77b41b014d35648adbe989fe3c31059856f60410758dddb3d9794e54a0ed227a",
    "src/humanbrain/_native_src/src/temporal_runtime.rs": "012d4f79505e3e0e381ebf9397db0c74d695120e7dfe8d511a338f8b896b9dc4",
    "src/humanbrain/_native_src/src/temporal_runtime/checkpoint.rs": "bdb1ab444a826e706b9c8284816a6e54847b0bb6c156723135324fffb377faba",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _display_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


@contextlib.contextmanager
def _public_train_authority():
    """Bypass only the intentionally non-runnable calibration source hold."""

    original = runtime_adapter._fresh_authority
    authority = runtime_adapter._FreshAuthority(b"\x91" * 32, b"\xe3" * 32)
    runtime_adapter._fresh_authority = lambda: authority
    try:
        yield
    finally:
        runtime_adapter._fresh_authority = original


def _assert_one_shot_paths_absent() -> None:
    for kind, path in (
        ("artifact", OUTPUT_PATH),
        *(("spend marker", path) for path in SPEND_MARKER_PATHS.values()),
    ):
        if path.exists():
            raise RuntimeError(f"refusing spent R3-SR0 probe; {kind} exists: {path}")


def _assert_sources_committed_and_unspent() -> None:
    _assert_one_shot_paths_absent()
    for source, expected in PINNED_SOURCE_SHA256.items():
        if _sha256_file(REPO_ROOT / source) != expected:
            raise RuntimeError(f"frozen R3-SR0 dependency changed: {source}")
    paths = (Path(__file__).resolve(), *(REPO_ROOT / source for source in PINNED_SOURCE_SHA256))
    for path in paths:
        source = path.relative_to(REPO_ROOT).as_posix()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", source],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout:
            raise RuntimeError(f"R3-SR0 source must be committed before spend: {source}")


def _select_tasks() -> tuple[object, tuple[GeneratedTask, ...], tuple[GeneratedTask, ...]]:
    protocol = load_protocol(PROTOCOL_PATH)
    if protocol.protocol_id != EXPECTED_PROTOCOL_ID or protocol.sha256 != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("public Simple-SWE protocol identity changed")
    visible = generate_visible_partition(protocol, "train")
    indexed = {(task.public.family, task.public.ordinal): task for task in visible}
    donors = tuple(
        indexed[(family, ordinal)]
        for ordinal in DONOR_ORDINALS
        for family in FAMILY_ORDER
    )
    probes = tuple(indexed[(family, PROBE_ORDINAL)] for family in FAMILY_ORDER)
    if any(task.public.partition != "train" for task in (*donors, *probes)):
        raise RuntimeError("R3-SR0 selected a non-TRAIN task")
    return protocol, donors, probes


def _task_key(task: PublicTask) -> tuple[str, int]:
    return task.family, task.ordinal


def _prepare_representations(
    donors: tuple[GeneratedTask, ...], probes: tuple[GeneratedTask, ...]
) -> tuple[
    RoleChunkEpisodeEncoder,
    dict[tuple[str, int], tuple[int, ...]],
    dict[tuple[str, int], tuple[int, ...]],
    list[dict[str, object]],
]:
    donor_public = tuple(task.public for task in donors)
    encoder = RoleChunkEpisodeEncoder.fit_public_train(
        donor_public,
        concept_config=DEFAULT_ROLE_CHUNK_CONCEPT_CONFIG,
        base_config=BASE_TOKENIZER_CONFIG,
    )
    if not encoder.tokenizer.concepts:
        raise RuntimeError("R3-SR0 requires a nonempty donor-only concept inventory")

    all_public = (*donor_public, *(task.public for task in probes))
    concept_streams: dict[tuple[str, int], tuple[int, ...]] = {}
    base_streams: dict[tuple[str, int], tuple[int, ...]] = {}
    task_records: list[dict[str, object]] = []
    structural_by_family: dict[str, set[str]] = {family: set() for family in FAMILY_ORDER}
    concept_by_family: dict[str, set[tuple[int, ...]]] = {family: set() for family in FAMILY_ORDER}
    base_by_family: dict[str, set[tuple[int, ...]]] = {family: set() for family in FAMILY_ORDER}

    for task in all_public:
        episode = encoder.encode(task)
        key = _task_key(task)
        concept = tuple(episode.token_ids)
        base = tuple(encoder.tokenizer.base_tokenizer.encode(episode.structural_text))
        concept_ids_used = sum(CONCEPT_ID_BASE <= token < ROUTE_TOKEN_BASE for token in concept)
        if concept_ids_used == 0:
            raise RuntimeError(f"R3-SR0 task emitted no concept IDs: {task.task_id}")
        if set((*concept, *base)) & set(ROUTE_TOKEN_IDS):
            raise RuntimeError("prompt representation entered the private route-token band")
        if episode.restore_prompt() != task.prompt:
            raise RuntimeError("role-chunk sidecar failed exact public-prompt restoration")
        concept_streams[key] = concept
        base_streams[key] = base
        structural_by_family[task.family].add(episode.structural_sha256)
        concept_by_family[task.family].add(concept)
        base_by_family[task.family].add(base)
        task_records.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "ordinal": task.ordinal,
                "task_sha256": task.task_sha256,
                "hidden_test_sha256": task.hidden_test_sha256,
                "public_prompt_sha256": hashlib.sha256(task.prompt.encode("utf-8")).hexdigest(),
                "structural_sha256": episode.structural_sha256,
                "payload_kind_counts": dict(sorted(Counter(p.kind.value for p in episode.payloads).items())),
                "prompt_restored_exact": True,
                "base_events": len(base),
                "concept_events": len(concept),
                "concept_id_events": concept_ids_used,
                "base_stream_sha256": _sha256_json(base),
                "concept_stream_sha256": _sha256_json(concept),
            }
        )

    if any(len(structural_by_family[family]) != 1 for family in FAMILY_ORDER):
        raise RuntimeError("ordinals 005/006/008 did not normalize to one template per family")
    if any(len(concept_by_family[family]) != 1 for family in FAMILY_ORDER):
        raise RuntimeError("concept streams differ inside a normalized family template")
    if any(len(base_by_family[family]) != 1 for family in FAMILY_ORDER):
        raise RuntimeError("base streams differ inside a normalized family template")
    if len({next(iter(structural_by_family[family])) for family in FAMILY_ORDER}) != 4:
        raise RuntimeError("R3-SR0 family templates collapsed together")
    return encoder, concept_streams, base_streams, task_records


def _conflict_route(family: str, ordinal: int) -> int:
    index = FAMILY_ORDER.index(family)
    shift = 1 if ordinal == DONOR_ORDINALS[0] else 2
    return ROUTE_TOKEN_IDS[(index + shift) % len(ROUTE_TOKEN_IDS)]


def _arm_route(arm: str, task: PublicTask) -> int:
    if arm.endswith("_true"):
        return ROUTE_BY_FAMILY[task.family]
    return _conflict_route(task.family, task.ordinal)


def _arm_stream(
    arm: str,
    key: tuple[str, int],
    concept_streams: dict[tuple[str, int], tuple[int, ...]],
    base_streams: dict[tuple[str, int], tuple[int, ...]],
) -> tuple[int, ...]:
    return concept_streams[key] if arm.startswith("concept_") else base_streams[key]


def _definition(
    donors: tuple[GeneratedTask, ...],
    probes: tuple[GeneratedTask, ...],
    concept_streams: dict[tuple[str, int], tuple[int, ...]],
    base_streams: dict[tuple[str, int], tuple[int, ...]],
) -> tuple[NativeTemporalCircuitDefinition, dict[str, int]]:
    donor_keys = tuple(_task_key(task.public) for task in donors)
    all_keys = (*donor_keys, *(_task_key(task.public) for task in probes))
    concept_noninitial = sum(len(concept_streams[key]) for key in donor_keys)
    base_noninitial = sum(len(base_streams[key]) for key in donor_keys)
    segment_capacity = ACTIVE_COLUMNS * max(concept_noninitial, base_noninitial)
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
                *(token for key in all_keys for token in concept_streams[key]),
                *(token for key in all_keys for token in base_streams[key]),
                *ROUTE_TOKEN_IDS,
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
    physical_cell_count = 2 * config.max_pyramidal_cells + 3 * config.column_count
    definition = NativeTemporalCircuitDefinition(
        config,
        token_codes,
        seeded_threshold_offsets_mv(physical_cell_count, config.construction_seed).tolist(),
    )
    return definition, {
        "concept_noninitial_arrivals": concept_noninitial,
        "base_noninitial_arrivals": base_noninitial,
        "segment_capacity": segment_capacity,
        "max_pyramidal_cells": config.max_pyramidal_cells,
    }


def _step_summary(steps: tuple[object, ...]) -> dict[str, int]:
    fields = (
        "allocated_segments",
        "punished_segments",
        "candidate_segments",
        "matching_segments",
        "active_segments",
        "postings_visited",
    )
    return {"steps": len(steps), **{field: sum(int(getattr(step, field)) for step in steps) for field in fields}}


def _train_arm(
    arm: str,
    definition: NativeTemporalCircuitDefinition,
    donors: tuple[GeneratedTask, ...],
    concept_streams: dict[tuple[str, int], tuple[int, ...]],
    base_streams: dict[tuple[str, int], tuple[int, ...]],
) -> tuple[bytes, object, dict[str, object]]:
    episodes: list[dict[str, object]] = []
    with NativeTemporalCircuit.fresh(definition) as circuit:
        for task in donors:
            key = _task_key(task.public)
            prompt = _arm_stream(arm, key, concept_streams, base_streams)
            label = _arm_route(arm, task.public)
            steps = circuit.advance_block((*prompt, label), mode=TemporalStepMode.LEARN, reset_before=True)
            episodes.append(
                {
                    "task_id": task.public.task_id,
                    "route_token_id": label,
                    **_step_summary(steps),
                }
            )
        identity = circuit.checkpoint_identity()
        checkpoint = circuit.checkpoint()
    return checkpoint, identity, {
        "episodes": episodes,
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "checkpoint_identity": asdict(identity),
    }


def _create_spend_marker(
    task: PublicTask,
    *,
    attempt_index: int,
    checkpoint_sha256_by_arm: dict[str, str],
) -> dict[str, object]:
    path = SPEND_MARKER_PATHS[task.family]
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-route",
        "state": "TARGET_ATTEMPT_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "attempt_index": attempt_index,
        "target_task_id": task.task_id,
        "family": task.family,
        "ordinal": task.ordinal,
        "arms": list(ARM_ORDER),
        "atomic_generation_calls_planned": len(ARM_ORDER),
        "checkpoint_sha256_by_arm": checkpoint_sha256_by_arm,
        "probe_source_sha256": _sha256_file(Path(__file__)),
        "canonical_artifact_path": _display_path(OUTPUT_PATH),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as marker:
            marker.write(serialized)
            marker.flush()
            os.fsync(marker.fileno())
    except FileExistsError as error:
        raise RuntimeError(f"refusing spent R3-SR0 target: {path}") from error
    return {
        "path": _display_path(path),
        "sha256": _sha256_file(path),
        "exclusive_create": True,
        "flushed_and_fsynced_before_four_generate_calls": True,
    }


def _generate_once(circuit: NativeTemporalCircuit) -> dict[str, object]:
    try:
        generation = circuit.generate()
    except NativeTemporalRuntimeError as error:
        return {
            "generation": None,
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
        "generation": {
            "surface_token_id": int(generation.surface_token_id),
            "path": generation.path.name,
            "activated_columns": int(generation.activated_columns),
            "winner_input_columns": int(generation.winner_input_columns),
            "candidate_tokens": int(generation.candidate_tokens),
            "physical_election_count": int(generation.physical_election_count),
            "temporal_event_count": int(generation.temporal_event_count),
        },
        "native_stop": None,
    }


def _score_generation(raw: dict[str, object], expected_route: int) -> dict[str, object]:
    generation = raw["generation"]
    physical = bool(
        generation is not None
        and generation["path"] == TemporalGenerationPath.TEMPORAL.name
        and generation["physical_election_count"] == 1
        and generation["activated_columns"] == ACTIVE_COLUMNS
        and generation["winner_input_columns"] == ACTIVE_COLUMNS
        and generation["candidate_tokens"] >= 1
    )
    observed = None if generation is None else int(generation["surface_token_id"])
    return {
        **raw,
        "expected_route_token_id_used_only_after_all_generation_calls": expected_route,
        "physical_route_election_valid": physical,
        "correct": physical and observed == expected_route,
    }


def _gate(correct: dict[str, int], prompt_events: dict[str, int]) -> dict[str, object]:
    concept_margin = correct["concept_true"] - correct["concept_conflict"]
    base_margin = correct["base_true"] - correct["base_conflict"]
    concept_licensed = correct["concept_true"] >= 3 and concept_margin >= 2
    base_licensed = correct["base_true"] >= 3 and base_margin >= 2
    selected: str | None = None
    if concept_licensed and not base_licensed:
        selected = "concept"
    elif base_licensed and not concept_licensed:
        selected = "base"
    elif concept_licensed and base_licensed:
        if correct["concept_true"] > correct["base_true"]:
            selected = "concept"
        elif correct["base_true"] > correct["concept_true"]:
            selected = "base"
        elif prompt_events["concept_true"] < prompt_events["base_true"]:
            selected = "concept"
        else:
            selected = "base"
    if correct["concept_true"] > correct["base_true"]:
        attribution = "CONCEPT_ACCURACY_ADVANTAGE"
    elif correct["concept_true"] < correct["base_true"]:
        attribution = "CONCEPT_ACCURACY_REGRESSION"
    else:
        attribution = "NO_CONCEPT_ACCURACY_ADVANTAGE"
    return {
        "correct_by_arm": correct,
        "matched_control_margins": {"concept": concept_margin, "base": base_margin},
        "licensed_by_representation": {"concept": concept_licensed, "base": base_licensed},
        "selected_representation": selected,
        "concept_attribution": attribution,
    }


def main() -> int:
    _assert_sources_committed_and_unspent()
    started_ns = time.perf_counter_ns()
    protocol, donors, probes = _select_tasks()
    encoder, concept_streams, base_streams, task_records = _prepare_representations(donors, probes)
    definition, capacity = _definition(donors, probes, concept_streams, base_streams)

    conflict_labels = [
        _conflict_route(task.public.family, task.public.ordinal) for task in donors
    ]
    if Counter(conflict_labels) != Counter({route: 2 for route in ROUTE_TOKEN_IDS}):
        raise RuntimeError("conflicting donor labels are not globally balanced")
    if any(label == ROUTE_BY_FAMILY[task.public.family] for label, task in zip(conflict_labels, donors, strict=True)):
        raise RuntimeError("conflicting donor labels retained a canonical family label")

    checkpoints: dict[str, bytes] = {}
    identities: dict[str, object] = {}
    train_records: dict[str, dict[str, object]] = {}
    target_records: list[dict[str, object]] = []
    with _public_train_authority():
        for arm in ARM_ORDER:
            checkpoint, identity, record = _train_arm(
                arm, definition, donors, concept_streams, base_streams
            )
            checkpoints[arm] = checkpoint
            identities[arm] = identity
            train_records[arm] = record

        checkpoint_hashes = {
            arm: train_records[arm]["checkpoint_sha256"] for arm in ARM_ORDER
        }
        for attempt_index, task in enumerate(probes, start=1):
            key = _task_key(task.public)
            prompt_summaries: dict[str, dict[str, int]] = {}
            raw_by_arm: dict[str, dict[str, object]] = {}
            with ExitStack() as stack:
                restored: dict[str, NativeTemporalCircuit] = {}
                for arm in ARM_ORDER:
                    circuit = stack.enter_context(
                        NativeTemporalCircuit.from_checkpoint(definition, checkpoints[arm])
                    )
                    if circuit.checkpoint_identity() != identities[arm]:
                        raise RuntimeError(f"restored R3-SR0 checkpoint differs for {arm}")
                    prompt = _arm_stream(arm, key, concept_streams, base_streams)
                    steps = circuit.advance_block(
                        prompt, mode=TemporalStepMode.READ_ONLY, reset_before=True
                    )
                    prompt_summaries[arm] = _step_summary(steps)
                    restored[arm] = circuit
                marker = _create_spend_marker(
                    task.public,
                    attempt_index=attempt_index,
                    checkpoint_sha256_by_arm=checkpoint_hashes,
                )
                for arm in ARM_ORDER:
                    raw_by_arm[arm] = _generate_once(restored[arm])

            expected_route = ROUTE_BY_FAMILY[task.public.family]
            scored = {
                arm: {
                    "prompt": prompt_summaries[arm],
                    **_score_generation(raw_by_arm[arm], expected_route),
                }
                for arm in ARM_ORDER
            }
            target_records.append(
                {
                    "task_id": task.public.task_id,
                    "family": task.public.family,
                    "ordinal": task.public.ordinal,
                    "target_prompt_entered_native_runtime": True,
                    "target_expected_route_label_entered_native_runtime": False,
                    "teacher_tokens_after_prompt": 0,
                    "spend_marker": marker,
                    "arms": scored,
                }
            )
        native_identity = loaded_library_identity()

    correct = {
        arm: sum(bool(record["arms"][arm]["correct"]) for record in target_records)
        for arm in ARM_ORDER
    }
    prompt_events = {
        arm: sum(int(record["arms"][arm]["prompt"]["steps"]) for record in target_records)
        for arm in ARM_ORDER
    }
    gate = _gate(correct, prompt_events)
    selected = gate["selected_representation"]
    if selected == "concept":
        outcome = "LICENSE_CONCEPT_REPETITION_CONDITIONED_ROUTE"
    elif selected == "base":
        outcome = "LICENSE_BASE_REPETITION_CONDITIONED_ROUTE"
    else:
        outcome = "STOP_REPETITION_CONDITIONED_ROUTE"

    tokenizer = encoder.tokenizer
    base_phrases = [phrase.hex() for phrase in tokenizer.base_tokenizer.phrases]
    concepts = [concept.hex() for concept in tokenizer.concepts]
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-route",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": outcome,
        "action_route_licensed": selected is not None,
        "claim_boundary": (
            "Visible TRAIN only: two repeated payload-normalized donor templates per family and one /008 template replay. "
            "This measures repetition-conditioned normalized-template route election, not paraphrase transfer, structural generalization, code generation, DEV/eval, held-out, M0, or C1 capability."
        ),
        "authority": {
            "partition": "train",
            "donor_order": [[task.public.family, task.public.ordinal] for task in donors],
            "probe_order": [[task.public.family, task.public.ordinal] for task in probes],
            "unused_follow_on_ordinal": UNUSED_FOLLOW_ON_ORDINAL,
            "arms": list(ARM_ORDER),
            "atomic_no_argument_generation_call_count": len(probes) * len(ARM_ORDER),
            "generation_calls_per_target": len(ARM_ORDER),
            "model_retries": 0,
            "harness_retries": 0,
            "target_labels_used_as_runtime_input": False,
            "private_cases_used_as_runtime_input": False,
        },
        "protocol": {"protocol_id": protocol.protocol_id, "protocol_sha256": protocol.sha256},
        "tasks": task_records,
        "tokenizer": {
            "base_config": asdict(BASE_TOKENIZER_CONFIG),
            "concept_config": asdict(DEFAULT_ROLE_CHUNK_CONCEPT_CONFIG),
            "base_vocab_size": tokenizer.base_tokenizer.vocab_size,
            "base_phrase_count": len(base_phrases),
            "base_phrase_sha256": _sha256_json(base_phrases),
            "concept_count": len(concepts),
            "concept_sha256": _sha256_json(concepts),
            "all_streams_used_concept_ids": True,
        },
        "routes": {
            "private_route_token_base": ROUTE_TOKEN_BASE,
            "true_by_family": ROUTE_BY_FAMILY,
            "conflict_rule": "/005 -> (family+1)%4; /006 -> (family+2)%4",
            "conflict_global_counts": dict(sorted(Counter(conflict_labels).items())),
        },
        "physical_definition": {
            "circuit_identity_sha256": definition.circuit_identity_sha256(),
            "column_count": COLUMN_COUNT,
            "active_columns": ACTIVE_COLUMNS,
            "theta_match": THETA_MATCH,
            "theta_active": THETA_ACTIVE,
            "constants_status": "inherited fixture constants; not calibration-v3 selection",
            "token_code_count": len(definition.token_codes),
            **capacity,
        },
        "train": train_records,
        "probes": target_records,
        "gate": {**gate, "prompt_events_by_arm": prompt_events},
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
    with OUTPUT_PATH.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"outcome": outcome, "selected": selected, "correct": correct}, sort_keys=True))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
