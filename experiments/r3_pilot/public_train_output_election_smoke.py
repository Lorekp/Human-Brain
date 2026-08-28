"""PUBLIC/TRAIN-only 0 -> 1 learning and native physical token emission.

Both literal tokens are public synthetic TRAIN material.  The circuit learns
the transition twice, is then shown cue token 0 alone, and must emit token 1
through the native output-cell race.  No candidate token or score reaches the
election call.
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


OUTPUT_PATH = Path(__file__).with_name("public_train_output_election_smoke.json")
CUE_TOKEN = 0
TARGET_TOKEN = 1
ACTIVE_COLUMNS = 11
COLUMN_COUNT = 22
BASE_CELLS_PER_COLUMN = 2
MAX_PYRAMIDAL_CELLS = COLUMN_COUNT * BASE_CELLS_PER_COLUMN
TRAIN_REPLAYS = 2


@contextlib.contextmanager
def _synthetic_public_train_authority() -> Iterator[None]:
    """Bypass only the intentionally non-runnable source-binding hold."""

    original = runtime_adapter._fresh_authority
    authority = runtime_adapter._FreshAuthority(b"\x6f" * 32, b"\xa4" * 32)
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
        segment_capacity=ACTIVE_COLUMNS,
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
        CUE_TOKEN: tuple(range(ACTIVE_COLUMNS)),
        TARGET_TOKEN: tuple(range(ACTIVE_COLUMNS, COLUMN_COUNT)),
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
                target_step = circuit.advance_block(
                    [CUE_TOKEN, TARGET_TOKEN],
                    mode=TemporalStepMode.LEARN,
                    reset_before=True,
                )[1]
                training.append(
                    {
                        "replay": replay,
                        "allocated_segments": target_step.allocated_segments,
                        "matching_segments": target_step.matching_segments,
                        "active_segments": target_step.active_segments,
                        "predicted_columns": target_step.predicted_columns,
                    }
                )

            cue_step = circuit.advance_block(
                [CUE_TOKEN],
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )[0]
            election = circuit.elect_output()
            identity = circuit.checkpoint_identity()
        native_identity = loaded_library_identity()

    passed = (
        training[0]["allocated_segments"] == ACTIVE_COLUMNS
        and training[1]["matching_segments"] == ACTIVE_COLUMNS
        and identity.segment_count == ACTIVE_COLUMNS
        and election.token_id == TARGET_TOKEN
        and election.activated_columns == ACTIVE_COLUMNS
        and election.candidate_tokens == 1
        and election.winner_input_columns == ACTIVE_COLUMNS
    )
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-train-output-election-smoke",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": (
            "PASS_PUBLIC_TRAIN_PHYSICAL_OUTPUT_ELECTION"
            if passed
            else "FAIL_PUBLIC_TRAIN_PHYSICAL_OUTPUT_ELECTION"
        ),
        "claim_boundary": (
            "A two-token public synthetic TRAIN transition only. After two physical learning "
            "replays, cue token 0 arrives alone and the native coupled output-cell/basket-cell "
            "race emits token 1. The election accepts no arriving candidate, score, current, "
            "winner, or Python argmax. This is a one-transition memorization smoke, not semantic "
            "generation, generalization, coding ability, held-out evidence, C1, or benchmark evidence."
        ),
        "authority": {
            "partition": "public-synthetic-train",
            "literal_train_sequence": [CUE_TOKEN, TARGET_TOKEN],
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
            "segment_capacity": ACTIVE_COLUMNS,
            "connected_permanence_ticks": definition.config.connected_permanence_ticks,
            "initial_permanence_ticks": definition.config.initial_permanence_ticks,
            "theta_match": definition.config.theta_match,
            "theta_active": definition.config.theta_active,
            "context_policy": definition.config.context_policy.name,
            "plasticity_policy": definition.config.plasticity_policy.name,
        },
        "training": {
            "replays": TRAIN_REPLAYS,
            "records": training,
            "live_segments": identity.segment_count,
            "learning_steps": identity.learning_step,
        },
        "cue": {
            "arriving_token_id": cue_step.token_id,
            "mode": "READ_ONLY",
            "reset_before": True,
        },
        "election": {
            "emitted_token_id": election.token_id,
            "output_cell_id": election.output_cell_id,
            "activated_columns": election.activated_columns,
            "candidate_tokens_physical": election.candidate_tokens,
            "winner_input_columns": election.winner_input_columns,
            "winner_tick": election.winner_tick,
            "basket_tick": election.basket_tick,
            "inhibition_onset_tick": election.inhibition_onset_tick,
            "simulated_ticks": election.simulated_ticks,
            "arriving_candidate_token_supplied": False,
            "python_candidate_enumeration": False,
            "python_argmax": False,
            "learned_readout_parameters": 0,
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
                "emitted_token_id": election.token_id,
                "activated_columns": election.activated_columns,
                "candidate_tokens_physical": election.candidate_tokens,
                "live_segments": identity.segment_count,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
