"""One public synthetic TRAIN/DEV probe for weak physical output routing.

TRAIN contains the literal ``inc/+1`` and ``dec/-1`` prompt-to-function
episodes. DEV presents the unseen ``bump/+1`` prompt once. The only scored
generation is one native output election followed by READ_ONLY feedback of the
emitted token. No DEV target token or candidate set reaches the runtime.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Iterator
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import sys
import time


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
    NativeTemporalRuntimeError,
    TemporalContextPolicy,
    TemporalOutputRoute,
    TemporalPlasticityPolicy,
    TemporalStepMode,
)


OUTPUT_PATH = Path(__file__).with_name("public_train_dev_weak_output_election_probe.json")
TRAIN_EXAMPLES = (
    {
        "example_id": "train-inc-add-one",
        "prompt": "Function inc adds one.\n",
        "code": "def inc(x):\n    return x + 1\n",
    },
    {
        "example_id": "train-dec-subtract-one",
        "prompt": "Function dec subtracts one.\n",
        "code": "def dec(x):\n    return x - 1\n",
    },
)
DEV_EXAMPLE = {
    "example_id": "dev-bump-add-one",
    "prompt": "Function bump adds one.\n",
    "scoring_first_output_byte": "d",
}
EOS_TOKEN = 256
ASCII_TOKEN_LIMIT = 128
EXPECTED_TOKEN = ord("d")
ACTIVE_COLUMNS = 11
BASE_CELLS_PER_COLUMN = 8
TRAIN_REPLAYS = 2


@contextlib.contextmanager
def _synthetic_public_probe_authority() -> Iterator[None]:
    """Bypass only the intentionally non-runnable source-binding hold."""

    original = runtime_adapter._fresh_authority
    authority = runtime_adapter._FreshAuthority(b"\x97" * 32, b"\xc2" * 32)
    runtime_adapter._fresh_authority = lambda: authority
    try:
        yield
    finally:
        runtime_adapter._fresh_authority = original


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def _episode(example: dict[str, str]) -> tuple[int, ...]:
    return (*example["prompt"].encode("ascii"), *example["code"].encode("ascii"), EOS_TOKEN)


def _definition() -> NativeTemporalCircuitDefinition:
    train_episodes = tuple(_episode(example) for example in TRAIN_EXAMPLES)
    token_ids = (*range(ASCII_TOKEN_LIMIT), EOS_TOKEN)
    column_count = len(token_ids) * ACTIVE_COLUMNS
    base_population = column_count * BASE_CELLS_PER_COLUMN
    maximum_new_segments = sum(len(episode) - 1 for episode in train_episodes) * ACTIVE_COLUMNS
    config = NativeTemporalCircuitConfig(
        column_count=column_count,
        max_pyramidal_cells=base_population + maximum_new_segments,
        segment_capacity=maximum_new_segments,
        construction_seed=17,
        connected_permanence_ticks=250,
        initial_permanence_ticks=255,
        permanence_increment_ticks=50,
        negative_decrement_ticks=2,
        base_cells_per_column=BASE_CELLS_PER_COLUMN,
        novel_race_cap=BASE_CELLS_PER_COLUMN,
        prediction_quorum=ACTIVE_COLUMNS,
        max_segments_per_cell=64,
        theta_match=16,
        theta_active=18,
        context_policy=TemporalContextPolicy.BOUND_WINNERS,
        plasticity_policy=TemporalPlasticityPolicy.PLASTIC,
    )
    token_codes = {
        token_id: tuple(
            range(code_index * ACTIVE_COLUMNS, (code_index + 1) * ACTIVE_COLUMNS)
        )
        for code_index, token_id in enumerate(token_ids)
    }
    physical_cell_count = 2 * config.max_pyramidal_cells + 3 * config.column_count
    return NativeTemporalCircuitDefinition(
        config,
        token_codes,
        seeded_threshold_offsets_mv(physical_cell_count, config.construction_seed).tolist(),
    )


def _stop_record(error: NativeTemporalRuntimeError, *, phase: str) -> dict[str, object]:
    return {
        "phase": phase,
        "status": error.status.name,
        "message": error.message,
        "completed_prefix": error.completed_prefix,
        "required": error.required,
        "observed": error.observed,
        "location": error.location,
        "poison_code": error.poison_code,
        "stage_code": error.stage_code,
    }


def main() -> int:
    args = _parse_args()
    started_ns = time.perf_counter_ns()
    definition = _definition()
    train_episodes = tuple(_episode(example) for example in TRAIN_EXAMPLES)

    election = None
    feedback = None
    stop: dict[str, object] | None = None
    state_unchanged_by_election: bool | None = None
    with _synthetic_public_probe_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            training = []
            for replay in range(TRAIN_REPLAYS):
                episode_records = []
                for example, episode in zip(TRAIN_EXAMPLES, train_episodes, strict=True):
                    steps = circuit.advance_block(
                        episode,
                        mode=TemporalStepMode.LEARN,
                        reset_before=True,
                    )
                    episode_records.append(
                        {
                            "example_id": example["example_id"],
                            "steps": len(steps),
                            "allocated_segments": sum(
                                step.allocated_segments for step in steps
                            ),
                            "matching_segments": sum(step.matching_segments for step in steps),
                            "active_segments": sum(step.active_segments for step in steps),
                            "predicted_steps": sum(int(step.predicted) for step in steps),
                        }
                    )
                training.append({"replay": replay, "episodes": episode_records})
            trained_identity = circuit.checkpoint_identity()

            prompt_steps = circuit.advance_block(
                tuple(DEV_EXAMPLE["prompt"].encode("ascii")),
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            before_election = circuit.checkpoint_identity()
            try:
                election = circuit.elect_output()
                after_election = circuit.checkpoint_identity()
                state_unchanged_by_election = after_election == before_election
                feedback = circuit.advance_block(
                    [election.token_id],
                    mode=TemporalStepMode.READ_ONLY,
                )[0]
            except NativeTemporalRuntimeError as error:
                stop = _stop_record(
                    error,
                    phase="elect_output" if election is None else "read_only_feedback",
                )
        native_identity = loaded_library_identity()

    expected_train_transitions = sum(len(episode) - 1 for episode in train_episodes)
    replay_two = training[1]["episodes"]
    training_stable = (
        sum(int(record["allocated_segments"]) for record in replay_two) == 0
        and sum(int(record["predicted_steps"]) for record in replay_two)
        == expected_train_transitions
    )
    passed = (
        training_stable
        and election is not None
        and election.token_id == EXPECTED_TOKEN
        and election.route is TemporalOutputRoute.WEAK_MATCHING
        and state_unchanged_by_election is True
        and feedback is not None
        and not feedback.predicted
    )
    if passed:
        outcome = "PASS_WEAK_MATCHING_EMITS_D_WITH_UNPREDICTED_FEEDBACK"
    elif stop is not None:
        outcome = "STOP_WEAK_MATCHING_OUTPUT_ELECTION"
    elif election is not None and election.route is not TemporalOutputRoute.WEAK_MATCHING:
        outcome = "FAIL_DEV_ELECTION_USED_NON_WEAK_ROUTE"
    elif election is not None and election.token_id != EXPECTED_TOKEN:
        outcome = "FAIL_WEAK_MATCHING_EMITTED_WRONG_TOKEN"
    elif feedback is not None and feedback.predicted:
        outcome = "FAIL_READ_ONLY_FEEDBACK_WAS_PREDICTED"
    elif state_unchanged_by_election is False:
        outcome = "FAIL_OUTPUT_ELECTION_MUTATED_RUNTIME_STATE"
    else:
        outcome = "INVALID_UNSTABLE_TRAIN_REPLAY"

    payload = {
        "schema_version": 1,
        "experiment": "r3-public-train-dev-weak-output-election-probe",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": outcome,
        "capability_pass": passed,
        "claim_boundary": (
            "Two literal public synthetic TRAIN prompt-to-function episodes establish physical "
            "temporal segments. One unseen public synthetic DEV prompt is presented once, then "
            "exactly one native physical output election is scored. PASS means only that the "
            "potential-only WEAK_MATCHING route emitted the shared first code byte 'd' and that "
            "READ_ONLY feedback of that byte remained unpredicted. The DEV scoring byte never "
            "enters inference; no candidate enumeration, Python argmax, learned head, retry, "
            "adaptation, held-out material, C1 material, coding claim, or SWE claim is involved."
        ),
        "authority": {
            "partitions": ["public-synthetic-train", "public-synthetic-dev"],
            "dev_attempts": 1,
            "dev_scoring_token_used_as_runtime_input": False,
            "held_out_material_read": False,
            "c1_material_read": False,
            "synthetic_runtime_authority": True,
        },
        "frozen_examples": {
            "train": list(TRAIN_EXAMPLES),
            "dev": DEV_EXAMPLE,
        },
        "physical_definition": {
            "circuit_identity_sha256": definition.circuit_identity_sha256(),
            "column_count": definition.config.column_count,
            "max_pyramidal_cells": definition.config.max_pyramidal_cells,
            "segment_capacity": definition.config.segment_capacity,
            "theta_match": definition.config.theta_match,
            "theta_active": definition.config.theta_active,
            "context_policy": definition.config.context_policy.name,
            "plasticity_policy": definition.config.plasticity_policy.name,
        },
        "training": {
            "replays": TRAIN_REPLAYS,
            "expected_transitions_per_replay": expected_train_transitions,
            "stable_second_replay": training_stable,
            "records": training,
            "live_segments": trained_identity.segment_count,
            "learning_steps": trained_identity.learning_step,
        },
        "dev_once": {
            "prompt_token_count": len(prompt_steps),
            "teacher_tokens_after_prompt": 0,
            "election_attempts": 1,
            "completed_elections": 1 if election is not None else 0,
            "stop": stop,
            "expected_scoring_token_id": EXPECTED_TOKEN,
            "expected_scoring_byte_hex": f"{EXPECTED_TOKEN:02x}",
            "emitted_token_id": None if election is None else election.token_id,
            "emitted_byte_hex": (
                None
                if election is None or not 0 <= election.token_id < ASCII_TOKEN_LIMIT
                else f"{election.token_id:02x}"
            ),
            "route": None if election is None else election.route.name,
            "activated_columns": None if election is None else election.activated_columns,
            "candidate_tokens_physical": (
                None if election is None else election.candidate_tokens
            ),
            "winner_input_columns": (
                None if election is None else election.winner_input_columns
            ),
            "state_unchanged_by_election": state_unchanged_by_election,
            "feedback_mode": "READ_ONLY" if feedback is not None else None,
            "feedback_predicted": None if feedback is None else feedback.predicted,
            "feedback_predicted_columns": (
                None if feedback is None else feedback.predicted_columns
            ),
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
    print(
        json.dumps(
            {
                "outcome": outcome,
                "emitted_token_id": payload["dev_once"]["emitted_token_id"],
                "route": payload["dev_once"]["route"],
                "feedback_predicted": payload["dev_once"]["feedback_predicted"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
