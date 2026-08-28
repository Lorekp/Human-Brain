"""PUBLIC/TRAIN-only free-running physical chain: 0 -> 1 -> 2 -> 3.

After two physical TRAIN replays, cue token 0 is presented exactly once.  Each
subsequent token is elected by the native output-cell race and that emitted
token is committed through the generator's atomic READ_ONLY feedback.
An untrained static distractor shares one column with token 1, making the first
election a two-candidate physical race without Python candidate enumeration.
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

from humanbrain.native import loaded_library_identity
import humanbrain.native_temporal_runtime as runtime_adapter
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitConfig,
    NativeTemporalCircuitDefinition,
    TemporalContextPolicy,
    TemporalPlasticityPolicy,
    TemporalStepMode,
)


OUTPUT_PATH = Path(__file__).with_name("public_train_free_running_chain_smoke.json")
TRAIN_SEQUENCE = (0, 1, 2, 3)
EXPECTED_EMISSIONS = TRAIN_SEQUENCE[1:]
DISTRACTOR_TOKEN = 99
ACTIVE_COLUMNS = 11
COLUMN_COUNT = 54
BASE_CELLS_PER_COLUMN = 2
MAX_PYRAMIDAL_CELLS = COLUMN_COUNT * BASE_CELLS_PER_COLUMN
TRAIN_REPLAYS = 2


@contextlib.contextmanager
def _synthetic_public_train_authority() -> Iterator[None]:
    """Bypass only the intentionally non-runnable source-binding hold."""

    original = runtime_adapter._fresh_authority
    authority = runtime_adapter._FreshAuthority(b"\x71" * 32, b"\xb6" * 32)
    runtime_adapter._fresh_authority = lambda: authority
    try:
        yield
    finally:
        runtime_adapter._fresh_authority = original


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def _definition() -> NativeTemporalCircuitDefinition:
    config = NativeTemporalCircuitConfig(
        column_count=COLUMN_COUNT,
        max_pyramidal_cells=MAX_PYRAMIDAL_CELLS,
        segment_capacity=(len(TRAIN_SEQUENCE) - 1) * ACTIVE_COLUMNS,
        construction_seed=17,
        connected_permanence_ticks=250,
        initial_permanence_ticks=255,
        permanence_increment_ticks=17,
        negative_decrement_ticks=10,
        base_cells_per_column=BASE_CELLS_PER_COLUMN,
        novel_race_cap=BASE_CELLS_PER_COLUMN,
        prediction_quorum=ACTIVE_COLUMNS,
        max_segments_per_cell=4,
        theta_match=18,
        theta_active=20,
        context_policy=TemporalContextPolicy.BOUND_WINNERS,
        plasticity_policy=TemporalPlasticityPolicy.PLASTIC,
    )
    token_codes = {
        0: tuple(range(0, 11)),
        1: tuple(range(11, 22)),
        2: tuple(range(22, 33)),
        3: tuple(range(33, 44)),
        DISTRACTOR_TOKEN: (11, *range(44, 54)),
    }
    physical_cell_count = 2 * MAX_PYRAMIDAL_CELLS + 3 * COLUMN_COUNT
    return NativeTemporalCircuitDefinition(
        config,
        token_codes,
        (0.0,) * physical_cell_count,
    )


def main() -> int:
    args = _parse_args()
    started_ns = time.perf_counter_ns()
    definition = _definition()

    with _synthetic_public_train_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            training = []
            for replay in range(TRAIN_REPLAYS):
                steps = circuit.advance_block(
                    TRAIN_SEQUENCE,
                    mode=TemporalStepMode.LEARN,
                    reset_before=True,
                )
                training.append(
                    {
                        "replay": replay,
                        "allocated_segments": sum(step.allocated_segments for step in steps),
                        "matching_segments": sum(step.matching_segments for step in steps),
                        "active_segments": sum(step.active_segments for step in steps),
                        "predicted_steps": sum(int(step.predicted) for step in steps),
                    }
                )

            cue_step = circuit.advance_block(
                [TRAIN_SEQUENCE[0]],
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )[0]
            free_run = []
            for position, expected_token in enumerate(EXPECTED_EMISSIONS, start=1):
                generation = circuit.generate()
                free_run.append(
                    {
                        "position": position,
                        "expected_token_id": expected_token,
                        "emitted_token_id": generation.surface_token_id,
                        "correct": generation.surface_token_id == expected_token,
                        "output_cell_id": generation.surface_output_cell_id,
                        "generation_index": generation.generation_index,
                        "generation_path": generation.path.name,
                        "temporal_event_count": generation.temporal_event_count,
                        "activated_columns": generation.activated_columns,
                        "candidate_tokens_physical": generation.candidate_tokens,
                        "winner_input_columns": generation.winner_input_columns,
                        "winner_tick": generation.surface_winner_tick,
                        "basket_tick": generation.surface_basket_tick,
                        "inhibition_onset_tick": generation.surface_inhibition_onset_tick,
                        "simulated_ticks": generation.total_simulated_ticks,
                        "feedback_mode": "ATOMIC_GENERATE_READ_ONLY",
                        "feedback_arriving_token_id": generation.surface_token_id,
                        "feedback_predicted": generation.feedback_predicted,
                        "feedback_predicted_columns": generation.feedback_predicted_columns,
                    }
                )
            identity = circuit.checkpoint_identity()
        native_identity = loaded_library_identity()

    emitted = tuple(record["emitted_token_id"] for record in free_run)
    candidate_counts = tuple(record["candidate_tokens_physical"] for record in free_run)
    passed = (
        training[0]["allocated_segments"] == 3 * ACTIVE_COLUMNS
        and training[1]["matching_segments"] == 3 * ACTIVE_COLUMNS
        and identity.segment_count == 3 * ACTIVE_COLUMNS
        and emitted == EXPECTED_EMISSIONS
        and all(record["feedback_predicted"] for record in free_run)
        and max(candidate_counts) > 1
    )
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-train-free-running-chain-smoke",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": (
            "PASS_PUBLIC_TRAIN_FREE_RUNNING_PHYSICAL_CHAIN"
            if passed
            else "FAIL_PUBLIC_TRAIN_FREE_RUNNING_PHYSICAL_CHAIN"
        ),
        "claim_boundary": (
            "One memorized four-token public synthetic TRAIN chain. Cue token 0 arrives once; "
            "tokens 1, 2, and 3 are then emitted and fed back by atomic native generate calls. "
            "The first generation includes an untrained static "
            "one-column-overlap distractor, so it is a two-candidate physical race. There is no "
            "teacher token after the cue, Python candidate enumeration, argmax, or learned head. "
            "This is not open-vocabulary generation, semantic reasoning, generalization, coding "
            "ability, held-out evidence, C1, or benchmark evidence."
        ),
        "authority": {
            "partition": "public-synthetic-train",
            "literal_train_sequence": list(TRAIN_SEQUENCE),
            "untrained_static_distractor_token": DISTRACTOR_TOKEN,
            "development_material_used": False,
            "held_out_material_read": False,
            "c1_material_read": False,
            "synthetic_runtime_authority": True,
        },
        "physical_definition": {
            "circuit_identity_sha256": definition.circuit_identity_sha256(),
            "column_count": COLUMN_COUNT,
            "active_columns_per_token": ACTIVE_COLUMNS,
            "max_pyramidal_cells": MAX_PYRAMIDAL_CELLS,
            "segment_capacity": definition.config.segment_capacity,
            "connected_permanence_ticks": definition.config.connected_permanence_ticks,
            "initial_permanence_ticks": definition.config.initial_permanence_ticks,
            "theta_match": definition.config.theta_match,
            "theta_active": definition.config.theta_active,
            "maximum_code_overlap": 1,
            "context_policy": definition.config.context_policy.name,
            "plasticity_policy": definition.config.plasticity_policy.name,
        },
        "training": {
            "replays": TRAIN_REPLAYS,
            "records": training,
            "live_segments": identity.segment_count,
            "learning_steps": identity.learning_step,
        },
        "free_run": {
            "cue_arrivals": [cue_step.token_id],
            "teacher_tokens_after_cue": 0,
            "emitted_sequence": list(emitted),
            "expected_sequence": list(EXPECTED_EMISSIONS),
            "election_candidate_counts": list(candidate_counts),
            "contains_multi_candidate_election": max(candidate_counts) > 1,
            "python_candidate_enumeration": False,
            "python_argmax": False,
            "learned_readout_parameters": 0,
            "steps": free_run,
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
                "outcome": payload["outcome"],
                "emitted_sequence": list(emitted),
                "election_candidate_counts": list(candidate_counts),
                "live_segments": identity.segment_count,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
