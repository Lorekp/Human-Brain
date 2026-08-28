"""Tiny PUBLIC/TRAIN-only code recall through the native physical R3 circuit.

This is deliberately a capability smoke, not an evaluation.  It replays one
public teacher from ``typed_program_public_train_protocol_v1.json`` and asks the
learned physical segment state to discriminate its entry-point token from the
other two TRAIN entry points.  The entry point is present verbatim in the cue,
so this measures long-range physical copy/recognition rather than semantic
solving.  It is candidate-scored recognition, not autonomous generation: the
current runtime recognizes an arriving token but does not yet run an output-
token race.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from collections.abc import Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from humanbrain.cortex import seeded_threshold_offsets_mv
from humanbrain.native import loaded_library_identity
import humanbrain.native_temporal_runtime as runtime_adapter
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitConfig,
    NativeTemporalCircuitDefinition,
    TemporalCircuitStep,
    TemporalContextPolicy,
    TemporalPlasticityPolicy,
    TemporalStepMode,
)
from humanbrain.sdr_union import token_sdr_indices
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig
from humanbrain.typed_program import Program, compile as compile_program
from humanbrain.typed_program_types import TypedSignature


PROTOCOL_PATH = REPO_ROOT / "benchmarks" / "typed_program_public_train_protocol_v1.json"
OUTPUT_PATH = Path(__file__).with_name("public_train_code_recall_smoke.json")
CONSTRUCTION_SEED = 17
CODEBOOK_SEED = 0
COLUMN_COUNT = 512
ACTIVE_COLUMNS = 11
BASE_CELLS_PER_COLUMN = 8
TRAIN_EPOCHS = 2


@dataclass(frozen=True, slots=True)
class PublicEpisode:
    task: dict[str, object]
    cue: str
    target_source: str
    cue_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    sequence: tuple[int, ...]


@contextlib.contextmanager
def _synthetic_public_smoke_authority() -> Iterator[None]:
    """Bypass only the protocol source-binding hold; never open C1 material."""

    original = runtime_adapter._fresh_authority
    authority = runtime_adapter._FreshAuthority(b"\x63" * 32, b"\x91" * 32)
    runtime_adapter._fresh_authority = lambda: authority
    try:
        yield
    finally:
        runtime_adapter._fresh_authority = original


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def _canonical_cue(task: dict[str, object]) -> str:
    signature = task["signature"]
    if not isinstance(signature, dict):
        raise TypeError("public TRAIN signature must be an object")
    entry_point = signature["entry_point"]
    return (
        f"TASK {task['task_id']}\n"
        f"PROMPT {task['prompt']}\n"
        f"ENTRY {entry_point}\n"
        "PYTHON\n"
    )


def _load_public_teachers() -> tuple[bytes, dict[str, object], list[tuple[dict[str, object], str, str]]]:
    raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema") != "humanbrain-public-typed-program-train-protocol-v1":
        raise RuntimeError("unexpected public TRAIN protocol schema")
    scope = protocol.get("scope")
    if not isinstance(scope, dict) or scope.get("operation") != "public-train-only":
        raise RuntimeError("protocol is not explicitly PUBLIC/TRAIN-only")
    train = protocol.get("train")
    if not isinstance(train, dict) or train.get("teacher_programs_public") is not True:
        raise RuntimeError("TRAIN teachers are not declared public")
    tasks = train.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 3:
        raise RuntimeError("expected the three frozen public TRAIN tasks")

    teachers: list[tuple[dict[str, object], str, str]] = []
    for raw_task in tasks:
        if not isinstance(raw_task, dict):
            raise TypeError("public TRAIN task must be an object")
        raw_tokens = raw_task.get("teacher_tokens")
        if not isinstance(raw_tokens, list) or not all(
            isinstance(token, str) and token for token in raw_tokens
        ):
            raise TypeError("public TRAIN teacher tokens are malformed")
        signature = TypedSignature.from_record(raw_task["signature"])
        target = compile_program(Program(tuple(raw_tokens), signature)).source
        teachers.append((raw_task, _canonical_cue(raw_task), target))
    return raw, protocol, teachers


def _build_episodes(
    teachers: Sequence[tuple[dict[str, object], str, str]],
) -> tuple[BytePhraseTokenizer, int, tuple[PublicEpisode, ...]]:
    tokenizer = BytePhraseTokenizer.fit(
        (text for _task, cue, target in teachers for text in (cue, target)),
        TokenizerConfig(max_phrases=256, min_frequency=1),
    )
    eos_id = tokenizer.vocab_size
    episodes: list[PublicEpisode] = []
    for task, cue, target in teachers:
        cue_ids = tuple(tokenizer.encode(cue))
        target_ids = tuple(tokenizer.encode(target))
        episodes.append(
            PublicEpisode(
                task=task,
                cue=cue,
                target_source=target,
                cue_ids=cue_ids,
                target_ids=target_ids,
                sequence=(*cue_ids, *target_ids, eos_id),
            )
        )
    return tokenizer, eos_id, tuple(episodes)


def _build_definition(
    episodes: Sequence[PublicEpisode], eos_id: int
) -> NativeTemporalCircuitDefinition:
    used_token_ids = sorted(
        {token_id for episode in episodes for token_id in episode.sequence} | {eos_id}
    )
    transition_count = sum(len(episode.sequence) - 1 for episode in episodes)
    worst_case_segments = transition_count * ACTIVE_COLUMNS * TRAIN_EPOCHS
    base_population = COLUMN_COUNT * BASE_CELLS_PER_COLUMN
    config = NativeTemporalCircuitConfig(
        column_count=COLUMN_COUNT,
        max_pyramidal_cells=base_population + worst_case_segments,
        segment_capacity=worst_case_segments,
        construction_seed=CONSTRUCTION_SEED,
        connected_permanence_ticks=250,
        initial_permanence_ticks=255,
        permanence_increment_ticks=50,
        negative_decrement_ticks=2,
        base_cells_per_column=BASE_CELLS_PER_COLUMN,
        novel_race_cap=8,
        prediction_quorum=ACTIVE_COLUMNS,
        max_segments_per_cell=64,
        theta_match=10,
        theta_active=13,
        context_policy=TemporalContextPolicy.BOUND_WINNERS,
        plasticity_policy=TemporalPlasticityPolicy.PLASTIC,
    )
    token_codes = {
        token_id: tuple(
            sorted(
                map(
                    int,
                    token_sdr_indices(
                        token_id,
                        seed=CODEBOOK_SEED,
                        n_bits=COLUMN_COUNT,
                        w_ones=ACTIVE_COLUMNS,
                    ),
                )
            )
        )
        for token_id in used_token_ids
    }
    physical_cell_count = 2 * config.max_pyramidal_cells + 3 * config.column_count
    return NativeTemporalCircuitDefinition(
        config=config,
        token_codes=token_codes,
        threshold_offsets_mv=seeded_threshold_offsets_mv(
            physical_cell_count, CONSTRUCTION_SEED
        ).tolist(),
    )


def _score(step: TemporalCircuitStep) -> tuple[int, int, int, int]:
    """Non-learned readout of physical native prediction evidence."""

    return (
        int(step.predicted),
        step.predicted_columns,
        step.active_segments,
        step.matching_segments,
    )


def _token_text(tokenizer: BytePhraseTokenizer, eos_id: int, token_id: int) -> str:
    if token_id == eos_id:
        return "<EOS>"
    return tokenizer.token_bytes(token_id).decode("utf-8", errors="backslashreplace")


def _probe_entry_point(
    definition: NativeTemporalCircuitDefinition,
    trained_checkpoint: bytes,
    episode: PublicEpisode,
    tokenizer: BytePhraseTokenizer,
    candidates: Sequence[int],
) -> dict[str, object]:
    signature = episode.task["signature"]
    if not isinstance(signature, dict):
        raise TypeError("task signature must be an object")
    entry_point = str(signature["entry_point"])
    name_ids = tokenizer.encode(entry_point)
    if len(name_ids) != 1:
        raise RuntimeError("public entry point must encode as one candidate token")
    expected_token = name_ids[0]
    marker = episode.target_source.index(entry_point)
    decision_prefix = tuple(tokenizer.encode(episode.target_source[:marker]))

    with NativeTemporalCircuit.from_checkpoint(definition, trained_checkpoint) as live:
        context_steps = live.advance_block(
            (*episode.cue_ids, *decision_prefix),
            mode=TemporalStepMode.READ_ONLY,
            reset_before=True,
        )
        decision_checkpoint = live.checkpoint()

    probes: list[tuple[tuple[int, int, int, int], int]] = []
    for candidate in candidates:
        with NativeTemporalCircuit.from_checkpoint(definition, decision_checkpoint) as fork:
            step = fork.advance_block([candidate], mode=TemporalStepMode.READ_ONLY)[0]
        probes.append((_score(step), candidate))
    probes.sort(key=lambda item: item[0], reverse=True)
    best_score = probes[0][0]
    tied = [probe for probe in probes if probe[0] == best_score]
    chosen = tied[0][1] if len(tied) == 1 and best_score != (0, 0, 0, 0) else None

    return {
        "task_id": episode.task["task_id"],
        "cue": episode.cue,
        "presented_code_prefix": episode.target_source[:marker],
        "expected_entry_point": entry_point,
        "expected_token_id": expected_token,
        "selected_entry_point": (
            _token_text(tokenizer, -1, chosen) if chosen is not None else None
        ),
        "selected_token_id": chosen,
        "pass": chosen == expected_token,
        "unique_physical_maximum": len(tied) == 1,
        "best_physical_score": list(best_score),
        "predicted_candidate_count": sum(score[0] for score, _token in probes),
        "context_predicted_steps": sum(int(step.predicted) for step in context_steps),
        "context_step_count": len(context_steps),
        "candidate_scores": [
            {
                "entry_point": _token_text(tokenizer, -1, candidate),
                "token_id": candidate,
                "physical_score": list(score),
                "expected": candidate == expected_token,
            }
            for score, candidate in probes
        ],
    }


def main() -> int:
    args = _parse_args()
    started_ns = time.perf_counter_ns()
    protocol_bytes, protocol, teachers = _load_public_teachers()
    tokenizer, eos_id, episodes = _build_episodes(teachers)
    definition = _build_definition(episodes, eos_id)
    entry_point_candidates = tuple(
        sorted(
            tokenizer.encode(str(episode.task["signature"]["entry_point"]))[0]  # type: ignore[index]
            for episode in episodes
        )
    )
    if len(set(entry_point_candidates)) != len(episodes):
        raise RuntimeError("public entry-point candidates must be distinct")

    training: list[dict[str, object]] = []
    replay_episodes = episodes[:1]
    with _synthetic_public_smoke_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            for epoch in range(TRAIN_EPOCHS):
                for episode in replay_episodes:
                    steps = circuit.advance_block(
                        episode.sequence,
                        mode=TemporalStepMode.LEARN,
                        reset_before=True,
                    )
                    training.append(
                        {
                            "epoch": epoch,
                            "task_id": episode.task["task_id"],
                            "steps": len(steps),
                            "predicted_steps": sum(int(step.predicted) for step in steps),
                            "allocated_segments": sum(step.allocated_segments for step in steps),
                            "punished_segments": sum(step.punished_segments for step in steps),
                        }
                    )
            trained_checkpoint = circuit.checkpoint()
            identity = circuit.checkpoint_identity()

        decisions = [
            _probe_entry_point(
                definition,
                trained_checkpoint,
                episode,
                tokenizer,
                entry_point_candidates,
            )
            for episode in replay_episodes
        ]
        native_identity = loaded_library_identity()

    decision_passes = sum(int(decision["pass"]) for decision in decisions)
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-train-code-recall-smoke",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": (
            "PASS_PUBLIC_TRAIN_CANDIDATE_RECALL"
            if decision_passes == len(decisions)
            else "FAIL_PUBLIC_TRAIN_CANDIDATE_RECALL"
        ),
        "claim_boundary": (
            "One public TRAIN teacher only. After replay, its public cue plus the presented "
            "code prefix 'def ' is restored once per candidate and the physical runtime "
            "scores its entry point against two untrained public entry-point negatives. "
            "The tokenizer and closed candidates are derived from all three public TRAIN "
            "definitions. The entry point is present verbatim in the cue, so this is "
            "closed-set long-range copy/recognition and candidate-scored memorization with "
            "no learned readout; it is not autonomous "
            "token election, code generation, semantic solving, development generalization, "
            "Simple-SWE, held-out, C1, or benchmark evidence."
        ),
        "authority": {
            "protocol_path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "partition": "train",
            "teacher_programs_public": True,
            "development_tasks_used": False,
            "development_completion_opened": False,
            "held_out_material_read": False,
            "synthetic_runtime_authority": True,
        },
        "tokenization": {
            "tokenizer": "humanbrain.tokenizer.BytePhraseTokenizer",
            "vocab_size_without_eos": tokenizer.vocab_size,
            "phrase_count": len(tokenizer.phrases),
            "eos_token_id": eos_id,
            "used_definition_token_count": len(definition.token_codes),
            "closed_entry_point_candidate_count": len(entry_point_candidates),
        },
        "physical_definition": {
            "circuit_identity_sha256": definition.circuit_identity_sha256(),
            "construction_seed": CONSTRUCTION_SEED,
            "codebook_seed": CODEBOOK_SEED,
            "column_count": definition.config.column_count,
            "active_columns_per_token": ACTIVE_COLUMNS,
            "max_pyramidal_cells": definition.config.max_pyramidal_cells,
            "physical_cell_count": len(definition.threshold_offsets_mv),
            "segment_capacity": definition.config.segment_capacity,
            "theta_match": definition.config.theta_match,
            "theta_active": definition.config.theta_active,
            "prediction_quorum": definition.config.prediction_quorum,
            "context_policy": definition.config.context_policy.name,
            "plasticity_policy": definition.config.plasticity_policy.name,
            "cell_thresholds": "seeded fitted threshold offsets",
        },
        "replay": {
            "epochs": TRAIN_EPOCHS,
            "teacher_task_count": len(replay_episodes),
            "teacher_sequence_tokens_per_epoch": sum(
                len(episode.sequence) for episode in replay_episodes
            ),
            "training_records": training,
            "checkpoint_bytes": len(trained_checkpoint),
            "live_segments": identity.segment_count,
            "learning_steps": identity.learning_step,
        },
        "candidate_recall": {
            "policy": "unique max of [predicted, predicted_columns, active_segments, matching_segments]",
            "learned_readout_parameters": 0,
            "checkpoint_per_candidate": True,
            "autonomous_token_election": False,
            "pass_count": decision_passes,
            "task_count": len(decisions),
            "decisions": decisions,
        },
        "runtime": {
            "elapsed_wall_ns": time.perf_counter_ns() - started_ns,
            "python": sys.version,
            "platform": platform.platform(),
            "native_source_digest": native_identity.source_digest,
            "native_binary_sha256": native_identity.binary_sha256,
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "outcome": payload["outcome"],
        "candidate_recall_passes": decision_passes,
        "task_count": len(decisions),
        "live_segments": identity.segment_count,
        "checkpoint_bytes": len(trained_checkpoint),
        "elapsed_wall_ns": payload["runtime"]["elapsed_wall_ns"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if payload["outcome"] == "PASS_PUBLIC_TRAIN_CANDIDATE_RECALL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
