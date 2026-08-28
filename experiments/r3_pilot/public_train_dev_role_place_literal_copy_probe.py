"""PUBLIC synthetic TRAIN/DEV probe for two physical role/place copy slots.

TRAIN teaches one normalized ``inc``/``1`` add template whose output recalls
ordinal slots 0 and 2. DEV supplies only a normalized prompt with
unseen ``boost`` and ``7`` values. Atomic native ``generate()`` must surface the
entire function and EOS without a Python-selected token.
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
from humanbrain.sdr_union import token_sdr_indices


OUTPUT_PATH = Path(__file__).with_name("public_train_dev_role_place_literal_copy_probe.json")
TRAIN_IDENTIFIER = b"inc"
TRAIN_LITERAL = b"1"
DEV_IDENTIFIER = b"boost"
DEV_LITERAL = b"7"
PROMPT_PREFIX = b"Function "
PROMPT_MIDDLE = b" adds "
PROMPT_SUFFIX = b".\n"
TARGET_PREFIX = b"def "
TARGET_MIDDLE = b"(x):\n    return x + "
TARGET_SUFFIX = b"\n"
EXPECTED_CODE = b"def boost(x):\n    return x + 7\n"
EOS_TOKEN_ID = 256
COLUMN_COUNT = 512
ACTIVE_COLUMNS = 11
CODEBOOK_SEED = 0
BASE_CELLS_PER_COLUMN = 8
CONSTRUCTION_SEED = 17
MAX_GENERATIONS = 64


@contextlib.contextmanager
def _synthetic_public_probe_authority() -> Iterator[None]:
    """Bypass only the intentionally non-runnable source-binding hold."""

    original = runtime_adapter._fresh_authority
    authority = runtime_adapter._FreshAuthority(b"\x91" * 32, b"\xe3" * 32)
    runtime_adapter._fresh_authority = lambda: authority
    try:
        yield
    finally:
        runtime_adapter._fresh_authority = original


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def _normalized_prompt(identifier: bytes, literal: bytes) -> tuple[int, ...]:
    return (
        *PROMPT_PREFIX,
        slot_start_token_id(0),
        *identifier,
        slot_end_token_id(0),
        *PROMPT_MIDDLE,
        slot_start_token_id(2),
        *literal,
        slot_end_token_id(2),
        *PROMPT_SUFFIX,
    )


def _train_episode() -> tuple[int, ...]:
    return (
        *_normalized_prompt(TRAIN_IDENTIFIER, TRAIN_LITERAL),
        *TARGET_PREFIX,
        copy_slot_token_id(0),
        *TARGET_MIDDLE,
        copy_slot_token_id(2),
        *TARGET_SUFFIX,
        EOS_TOKEN_ID,
    )


def _definition() -> NativeTemporalCircuitDefinition:
    train = _train_episode()
    dev = _normalized_prompt(DEV_IDENTIFIER, DEV_LITERAL)
    role_tokens = {
        *(slot_start_token_id(slot) for slot in range(SLOT_COUNT)),
        *(slot_end_token_id(slot) for slot in range(SLOT_COUNT)),
        *(copy_slot_token_id(slot) for slot in range(SLOT_COUNT)),
        COPY_END_TOKEN_ID,
    }
    used_token_ids = tuple(sorted({*train, *dev, *role_tokens}))
    segment_capacity = (len(train) - 1) * ACTIVE_COLUMNS
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
        theta_match=16,
        theta_active=18,
        context_policy=TemporalContextPolicy.BOUND_WINNERS,
        plasticity_policy=TemporalPlasticityPolicy.PLASTIC,
    )
    token_codes = {
        token_id: tuple(
            int(column)
            for column in token_sdr_indices(
                token_id,
                n_bits=COLUMN_COUNT,
                w_ones=ACTIVE_COLUMNS,
                seed=CODEBOOK_SEED,
            )
        )
        for token_id in used_token_ids
    }
    physical_cell_count = 2 * config.max_pyramidal_cells + 3 * config.column_count
    return NativeTemporalCircuitDefinition(
        config,
        token_codes,
        seeded_threshold_offsets_mv(
            physical_cell_count,
            config.construction_seed,
        ).tolist(),
    )


def _generation_record(position: int, generation: object) -> dict[str, object]:
    token_id = int(generation.surface_token_id)
    return {
        "position": position,
        "generation_index": int(generation.generation_index),
        "path": generation.path.name,
        "slot_ordinal": generation.slot_ordinal,
        "copy_place_ordinal": generation.copy_place_ordinal,
        "surface_token_id": token_id,
        "surface_byte_hex": None if token_id == EOS_TOKEN_ID else f"{token_id:02x}",
        "temporal_event_count": int(generation.temporal_event_count),
        "control_token_id": generation.control_token_id,
        "physical_election_count": int(generation.physical_election_count),
        "candidate_tokens_physical": int(generation.candidate_tokens),
        "winner_input_columns": int(generation.winner_input_columns),
        "feedback_predicted": bool(generation.feedback_predicted),
    }


def _compile_analysis(source: bytes) -> dict[str, object]:
    result: dict[str, object] = {
        "ascii_valid": False,
        "ast_parse_valid": False,
        "compile_valid": False,
        "function_name": None,
    }
    try:
        text = source.decode("ascii", errors="strict")
        result["ascii_valid"] = True
        module = ast.parse(text, filename="<role-place-literal-copy-dev>", mode="exec")
        result["ast_parse_valid"] = True
        compile(module, "<role-place-literal-copy-dev>", "exec")
        result["compile_valid"] = True
    except (UnicodeDecodeError, SyntaxError, ValueError, TypeError):
        return result
    if len(module.body) == 1 and isinstance(module.body[0], ast.FunctionDef):
        result["function_name"] = module.body[0].name
    return result


def _analyze_copy_transactions(
    generations: list[dict[str, object]],
) -> tuple[list[dict[str, object]], bool]:
    transactions: list[dict[str, object]] = []
    active: dict[str, object] | None = None
    exact = True
    previous_event_count: int | None = None
    for record in generations:
        event_count = int(record["temporal_event_count"])
        path = str(record["path"])
        if path == TemporalGenerationPath.ENTER_COPY.name:
            exact &= active is None and previous_event_count is not None
            active = {
                "slot": record["slot_ordinal"],
                "control_token_id": record["control_token_id"],
                "enter_event_count": event_count,
                "control_advanced_temporal_once": (
                    previous_event_count is not None
                    and event_count == previous_event_count + 1
                ),
                "copied_places": [record["copy_place_ordinal"]],
                "copied_event_counts": [event_count],
                "exit_event_count": None,
                "exit_surface_token_id": None,
            }
        elif path == TemporalGenerationPath.COPY.name:
            exact &= active is not None
            if active is not None:
                exact &= record["slot_ordinal"] == active["slot"]
                active["copied_places"].append(record["copy_place_ordinal"])
                active["copied_event_counts"].append(event_count)
        elif path == TemporalGenerationPath.EXIT_COPY.name:
            exact &= active is not None
            if active is not None:
                exact &= record["slot_ordinal"] == active["slot"]
                active["exit_event_count"] = event_count
                active["exit_surface_token_id"] = record["surface_token_id"]
                entry_count = int(active["enter_event_count"])
                active["temporal_event_count_held_across_copied_bytes"] = all(
                    int(value) == entry_count for value in active["copied_event_counts"]
                )
                active["exit_resumed_temporal_once"] = event_count == entry_count + 1
                transactions.append(active)
                active = None
        previous_event_count = event_count
    exact &= active is None
    exact &= [record["slot"] for record in transactions] == [
        0,
        2,
    ]
    exact &= [record["control_token_id"] for record in transactions] == [
        copy_slot_token_id(0),
        copy_slot_token_id(2),
    ]
    exact &= all(
        bool(record["control_advanced_temporal_once"])
        and bool(record["temporal_event_count_held_across_copied_bytes"])
        and bool(record["exit_resumed_temporal_once"])
        for record in transactions
    )
    exact &= [record["copied_places"] for record in transactions] == [
        [0, 1, 2, 3, 4],
        [0],
    ]
    return transactions, exact


def main() -> int:
    args = _parse_args()
    started_ns = time.perf_counter_ns()
    definition = _definition()
    train = _train_episode()
    dev_prompt = _normalized_prompt(DEV_IDENTIFIER, DEV_LITERAL)

    with _synthetic_public_probe_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            train_steps = circuit.advance_block(
                train,
                mode=TemporalStepMode.LEARN,
                reset_before=True,
            )
            trained_identity = circuit.checkpoint_identity()
            dev_steps = circuit.advance_block(
                dev_prompt,
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            emitted_bytes = bytearray()
            generations: list[dict[str, object]] = []
            terminated_by_eos = False
            stop: dict[str, object] | None = None
            for position in range(MAX_GENERATIONS):
                try:
                    generation = circuit.generate()
                except NativeTemporalRuntimeError as error:
                    stop = {
                        "generation_position": position,
                        "status": error.status.name,
                        "message": error.message,
                        "required": error.required,
                        "observed": error.observed,
                        "location": error.location,
                        "poison_code": error.poison_code,
                        "stage_code": error.stage_code,
                    }
                    break
                generations.append(_generation_record(position, generation))
                if generation.surface_token_id == EOS_TOKEN_ID:
                    terminated_by_eos = True
                    break
                if not 0 <= generation.surface_token_id < 128:
                    stop = {
                        "generation_position": position,
                        "status": "NON_ASCII_SURFACE_TOKEN",
                        "observed": int(generation.surface_token_id),
                    }
                    break
                emitted_bytes.append(generation.surface_token_id)
        native_identity = loaded_library_identity()

    # Frozen DEV target appears only after the sole teacher-free generation.
    emitted = bytes(emitted_bytes)
    compile_analysis = _compile_analysis(emitted)
    transactions, copy_transactions_exact = _analyze_copy_transactions(generations)
    exact_output = emitted == EXPECTED_CODE
    final_eos_exact = bool(
        generations
        and generations[-1]["surface_token_id"] == EOS_TOKEN_ID
        and generations[-1]["path"] == TemporalGenerationPath.TEMPORAL.name
    )
    passed = bool(
        stop is None
        and terminated_by_eos
        and final_eos_exact
        and exact_output
        and compile_analysis["compile_valid"]
        and compile_analysis["function_name"] == "boost"
        and copy_transactions_exact
    )

    payload = {
        "schema_version": 1,
        "experiment": "r3-public-train-dev-role-place-literal-copy-probe",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": (
            "PASS_PHYSICAL_MULTI_SLOT_COPY_AND_ROUTE_RESUME"
            if passed
            else "FAIL_PHYSICAL_MULTI_SLOT_COPY_AND_ROUTE_RESUME"
        ),
        "capability_pass": passed,
        "claim_boundary": (
            "One public synthetic TRAIN inc/1 add sequence teaches a literal temporal "
            "template with internal ordinal slot-0 and slot-2 recall controls. One public synthetic "
            "DEV prompt captures unseen ASCII identifier boost and unseen ASCII literal 7 "
            "inside deterministic physical role markers and supplies no target. Atomic native "
            "generate() and fitted cell races select every surface token and EOS over a seeded "
            "overlapping 11-of-512 codebook. This establishes exact two-slot transient copy, "
            "one temporal advance per COPY control, temporal hold across copied bytes, route "
            "resumption, and compiling Python for this one micro-probe. It is not broad "
            "systematic generalization, semantic coding competence, SWE, protected data, C1, "
            "or benchmark evidence."
        ),
        "authority": {
            "partitions": ["public-synthetic-train", "public-synthetic-dev"],
            "train_examples": 1,
            "dev_attempts": 1,
            "dev_target_used_as_runtime_input": False,
            "teacher_tokens_after_dev_prompt": 0,
            "python_candidate_enumeration": False,
            "python_argmax": False,
            "learned_readout_parameters": 0,
            "protected_or_c1_material_read": False,
            "synthetic_runtime_authority": True,
        },
        "tokenization": {
            "scheme": "ASCII byte IDs, EOS=256, seven reserved native role/control IDs",
            "slot_0_start_token_id": slot_start_token_id(0),
            "slot_0_end_token_id": slot_end_token_id(0),
            "slot_2_start_token_id": slot_start_token_id(2),
            "slot_2_end_token_id": slot_end_token_id(2),
            "copy_slot_0_token_id": copy_slot_token_id(0),
            "copy_slot_2_token_id": copy_slot_token_id(2),
            "copy_end_token_id": COPY_END_TOKEN_ID,
            "static_codebook": "seeded overlapping 11-of-512 token_sdr_indices",
            "codebook_seed": CODEBOOK_SEED,
        },
        "physical_definition": {
            "circuit_identity_sha256": definition.circuit_identity_sha256(),
            "column_count": definition.config.column_count,
            "active_columns_per_token": ACTIVE_COLUMNS,
            "used_token_count": len(definition.token_codes),
            "physical_cell_count": len(definition.threshold_offsets_mv),
            "segment_capacity": definition.config.segment_capacity,
            "connected_permanence_ticks": definition.config.connected_permanence_ticks,
            "initial_permanence_ticks": definition.config.initial_permanence_ticks,
            "theta_match": definition.config.theta_match,
            "theta_active": definition.config.theta_active,
            "cell_thresholds": "seeded fitted threshold offsets",
        },
        "train": {
            "identifier": TRAIN_IDENTIFIER.decode("ascii"),
            "literal": TRAIN_LITERAL.decode("ascii"),
            "steps": len(train_steps),
            "internal_target_shape": (
                "def <COPY[0]>(x): return x + <COPY[2]>; EOS"
            ),
            "literal_identifier_or_value_in_target": False,
            "allocated_segments": sum(step.allocated_segments for step in train_steps),
            "live_segments": trained_identity.segment_count,
            "learning_steps": trained_identity.learning_step,
        },
        "dev_teacher_free": {
            "identifier": DEV_IDENTIFIER.decode("ascii"),
            "literal": DEV_LITERAL.decode("ascii"),
            "prompt_steps": len(dev_steps),
            "completed_generations": len(generations),
            "terminated_by_eos": terminated_by_eos,
            "final_eos_exact": final_eos_exact,
            "stop": stop,
            "emitted_code_text": (
                emitted.decode("ascii") if compile_analysis["ascii_valid"] else None
            ),
            "emitted_code_hex": emitted.hex(),
            "emitted_code_sha256": hashlib.sha256(emitted).hexdigest(),
            "expected_code_sha256": hashlib.sha256(EXPECTED_CODE).hexdigest(),
            "exact_expected_bytes": exact_output,
            "compile_analysis": compile_analysis,
            "copy_transactions_exact": copy_transactions_exact,
            "copy_transactions": transactions,
            "generations": generations,
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
                "completed_generations": len(generations),
                "copy_slots": [record["slot"] for record in transactions],
                "copy_transactions_exact": copy_transactions_exact,
                "terminated_by_eos": terminated_by_eos,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
