"""PUBLIC/TRAIN-only teacher-free emission of one memorized Python function.

TRAIN replays literal UTF-8 byte IDs for a public prompt followed by a short
function and EOS.  Inference presents the prompt once, then supplies no teacher
token: every code byte and EOS is selected by the native physical output race
and fed back READ_ONLY to form the next context.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
from collections.abc import Iterator
from datetime import UTC, datetime
import hashlib
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
    TemporalContextPolicy,
    TemporalPlasticityPolicy,
    TemporalStepMode,
)


OUTPUT_PATH = Path(__file__).with_name("public_train_teacher_free_code_emission_smoke.json")
PROMPT_TEXT = "Emit the memorized increment function.\n"
CODE_TEXT = "def inc(x):\n    return x + 1\n"
PROMPT_BYTES = PROMPT_TEXT.encode("utf-8")
CODE_BYTES = CODE_TEXT.encode("utf-8")
EOS_TOKEN = 256
ACTIVE_COLUMNS = 11
BASE_CELLS_PER_COLUMN = 8
TRAIN_REPLAYS = 2


@contextlib.contextmanager
def _synthetic_public_train_authority() -> Iterator[None]:
    """Bypass only the intentionally non-runnable source-binding hold."""

    original = runtime_adapter._fresh_authority
    authority = runtime_adapter._FreshAuthority(b"\x79" * 32, b"\xc2" * 32)
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
    train_tokens = (*PROMPT_BYTES, *CODE_BYTES, EOS_TOKEN)
    used_token_ids = tuple(sorted(set(train_tokens)))
    column_count = len(used_token_ids) * ACTIVE_COLUMNS
    base_population = column_count * BASE_CELLS_PER_COLUMN
    maximum_new_segments = (len(train_tokens) - 1) * ACTIVE_COLUMNS
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
        for code_index, token_id in enumerate(used_token_ids)
    }
    physical_cell_count = 2 * config.max_pyramidal_cells + 3 * config.column_count
    return NativeTemporalCircuitDefinition(
        config,
        token_codes,
        seeded_threshold_offsets_mv(physical_cell_count, config.construction_seed).tolist(),
    )


def main() -> int:
    args = _parse_args()
    started_ns = time.perf_counter_ns()
    definition = _definition()
    train_tokens = (*PROMPT_BYTES, *CODE_BYTES, EOS_TOKEN)

    with _synthetic_public_train_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            training = []
            for replay in range(TRAIN_REPLAYS):
                steps = circuit.advance_block(
                    train_tokens,
                    mode=TemporalStepMode.LEARN,
                    reset_before=True,
                )
                training.append(
                    {
                        "replay": replay,
                        "steps": len(steps),
                        "allocated_segments": sum(step.allocated_segments for step in steps),
                        "matching_segments": sum(step.matching_segments for step in steps),
                        "active_segments": sum(step.active_segments for step in steps),
                        "predicted_steps": sum(int(step.predicted) for step in steps),
                        "punished_segments": sum(step.punished_segments for step in steps),
                    }
                )

            prompt_steps = circuit.advance_block(
                tuple(PROMPT_BYTES),
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            emitted_bytes = bytearray()
            free_run = []
            terminated_by_eos = False
            maximum_elections = len(CODE_BYTES) + 8
            for position in range(maximum_elections):
                election = circuit.elect_output()
                feedback = circuit.advance_block(
                    [election.token_id],
                    mode=TemporalStepMode.READ_ONLY,
                )[0]
                is_eos = election.token_id == EOS_TOKEN
                if not is_eos:
                    if not 0 <= election.token_id <= 255:
                        raise RuntimeError("physical election emitted a non-byte token before EOS")
                    emitted_bytes.append(election.token_id)
                free_run.append(
                    {
                        "position": position,
                        "emitted_token_id": election.token_id,
                        "emitted_byte_hex": (
                            None if is_eos else f"{election.token_id:02x}"
                        ),
                        "eos": is_eos,
                        "output_cell_id": election.output_cell_id,
                        "activated_columns": election.activated_columns,
                        "candidate_tokens_physical": election.candidate_tokens,
                        "winner_input_columns": election.winner_input_columns,
                        "winner_tick": election.winner_tick,
                        "basket_tick": election.basket_tick,
                        "feedback_mode": "READ_ONLY",
                        "feedback_predicted": feedback.predicted,
                        "feedback_predicted_columns": feedback.predicted_columns,
                    }
                )
                if is_eos:
                    terminated_by_eos = True
                    break
            identity = circuit.checkpoint_identity()
        native_identity = loaded_library_identity()

    emitted = bytes(emitted_bytes)
    emitted_text: str | None
    utf8_valid = False
    ast_valid = False
    compile_valid = False
    try:
        emitted_text = emitted.decode("utf-8", errors="strict")
        utf8_valid = True
        parsed = ast.parse(emitted_text, filename="<physical-emission>", mode="exec")
        ast_valid = True
        compile(parsed, "<physical-emission>", "exec")
        compile_valid = True
    except (UnicodeDecodeError, SyntaxError, ValueError, TypeError):
        emitted_text = None

    expected_transition_count = len(train_tokens) - 1
    candidate_counts = tuple(
        int(record["candidate_tokens_physical"]) for record in free_run
    )
    passed = (
        training[1]["allocated_segments"] == 0
        and training[1]["predicted_steps"] == expected_transition_count
        and terminated_by_eos
        and emitted == CODE_BYTES
        and utf8_valid
        and ast_valid
        and compile_valid
        and all(bool(record["feedback_predicted"]) for record in free_run)
        and candidate_counts
        and set(candidate_counts) == {1}
    )
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-train-teacher-free-code-emission-smoke",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": (
            "PASS_PUBLIC_TRAIN_TEACHER_FREE_CODE_EMISSION"
            if passed
            else "FAIL_PUBLIC_TRAIN_TEACHER_FREE_CODE_EMISSION"
        ),
        "claim_boundary": (
            "One literal public synthetic TRAIN prompt-to-function sequence is memorized. At "
            "inference the prompt arrives once, then every function byte and EOS is emitted by "
            "the native physical output race and fed back READ_ONLY. Static disjoint byte codes "
            "make this a one-physical-candidate-per-step emission path. There is no teacher token "
            "after the prompt, Python candidate enumeration, argmax, or learned head. Exact output "
            "is parsed and compiled but not executed. This is rote code emission, not task solving, "
            "semantic reasoning, generalization, coding ability, SWE, held-out, C1, or benchmark evidence."
        ),
        "authority": {
            "partition": "public-synthetic-train",
            "prompt_text": PROMPT_TEXT,
            "teacher_code_text": CODE_TEXT,
            "development_material_used": False,
            "held_out_material_read": False,
            "c1_material_read": False,
            "synthetic_runtime_authority": True,
        },
        "tokenization": {
            "scheme": "literal UTF-8 byte IDs plus EOS=256",
            "prompt_token_count": len(PROMPT_BYTES),
            "code_token_count": len(CODE_BYTES),
            "eos_token_id": EOS_TOKEN,
            "used_token_count": len(definition.token_codes),
            "static_codebook": "disjoint 11-column code per used byte/EOS",
        },
        "physical_definition": {
            "circuit_identity_sha256": definition.circuit_identity_sha256(),
            "column_count": definition.config.column_count,
            "active_columns_per_token": ACTIVE_COLUMNS,
            "max_pyramidal_cells": definition.config.max_pyramidal_cells,
            "physical_cell_count": len(definition.threshold_offsets_mv),
            "segment_capacity": definition.config.segment_capacity,
            "connected_permanence_ticks": definition.config.connected_permanence_ticks,
            "initial_permanence_ticks": definition.config.initial_permanence_ticks,
            "theta_match": definition.config.theta_match,
            "theta_active": definition.config.theta_active,
            "context_policy": definition.config.context_policy.name,
            "plasticity_policy": definition.config.plasticity_policy.name,
            "cell_thresholds": "seeded fitted threshold offsets",
        },
        "training": {
            "replays": TRAIN_REPLAYS,
            "sequence_tokens_including_eos": len(train_tokens),
            "records": training,
            "live_segments": identity.segment_count,
            "learning_steps": identity.learning_step,
        },
        "inference": {
            "prompt_arrivals": len(prompt_steps),
            "teacher_tokens_after_prompt": 0,
            "election_count_including_eos": len(free_run),
            "terminated_by_eos": terminated_by_eos,
            "emitted_code_text": emitted_text,
            "emitted_code_hex": emitted.hex(),
            "emitted_code_sha256": hashlib.sha256(emitted).hexdigest(),
            "expected_code_sha256": hashlib.sha256(CODE_BYTES).hexdigest(),
            "exact_teacher_bytes": emitted == CODE_BYTES,
            "utf8_valid": utf8_valid,
            "ast_parse_valid": ast_valid,
            "compile_valid": compile_valid,
            "python_candidate_enumeration": False,
            "python_argmax": False,
            "learned_readout_parameters": 0,
            "candidate_counts": list(candidate_counts),
            "candidate_boundary": "one physical candidate per step from disjoint byte codes",
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
                "emitted_code_sha256": payload["inference"]["emitted_code_sha256"],
                "emitted_code_bytes": len(emitted),
                "elections_including_eos": len(free_run),
                "live_segments": identity.segment_count,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
