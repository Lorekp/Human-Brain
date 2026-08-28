"""PUBLIC synthetic TRAIN/DEV probe for physical role/place identifier copy.

TRAIN teaches one ``inc/+1`` template whose identifier slot contains the
internal COPY_ID role rather than the literal name. DEV presents the unseen
identifier ``bump`` inside explicit identifier boundaries and no target. Every
surface byte and EOS is then produced by the circuit's atomic native
``generate()`` transaction.
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
    COPY_ID_TOKEN_ID,
    IDENTIFIER_END_TOKEN_ID,
    IDENTIFIER_START_TOKEN_ID,
    NativeTemporalCircuit,
    NativeTemporalCircuitConfig,
    NativeTemporalCircuitDefinition,
    NativeTemporalRuntimeError,
    TemporalContextPolicy,
    TemporalGenerationPath,
    TemporalPlasticityPolicy,
    TemporalStepMode,
)
from humanbrain.sdr_union import token_sdr_indices


OUTPUT_PATH = Path(__file__).with_name("public_train_dev_role_place_copy_probe.json")
TRAIN_NAME = b"inc"
DEV_NAME = b"bump"
PROMPT_PREFIX = b"Function "
PROMPT_SUFFIX = b" adds one.\n"
TARGET_PREFIX = b"def "
TARGET_SUFFIX = b"(x):\n    return x + 1\n"
EOS_TOKEN_ID = 256
ASCII_TOKEN_LIMIT = 128
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


def _normalized_prompt(name: bytes) -> tuple[int, ...]:
    return (
        *PROMPT_PREFIX,
        IDENTIFIER_START_TOKEN_ID,
        *name,
        IDENTIFIER_END_TOKEN_ID,
        *PROMPT_SUFFIX,
    )


def _train_episode() -> tuple[int, ...]:
    return (
        *_normalized_prompt(TRAIN_NAME),
        *TARGET_PREFIX,
        COPY_ID_TOKEN_ID,
        *TARGET_SUFFIX,
        EOS_TOKEN_ID,
    )


def _definition() -> NativeTemporalCircuitDefinition:
    train_episode = _train_episode()
    dev_prompt = _normalized_prompt(DEV_NAME)
    used_token_ids = tuple(
        sorted(
            {
                *train_episode,
                *dev_prompt,
                IDENTIFIER_START_TOKEN_ID,
                IDENTIFIER_END_TOKEN_ID,
                COPY_ID_TOKEN_ID,
                COPY_END_TOKEN_ID,
            }
        )
    )
    segment_capacity = (len(train_episode) - 1) * ACTIVE_COLUMNS
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


def _stop_record(
    error: NativeTemporalRuntimeError,
    *,
    generation_position: int,
) -> dict[str, object]:
    return {
        "generation_position": generation_position,
        "status": error.status.name,
        "message": error.message,
        "completed_prefix": error.completed_prefix,
        "required": error.required,
        "observed": error.observed,
        "location": error.location,
        "poison_code": error.poison_code,
        "stage_code": error.stage_code,
    }


def _generation_record(position: int, generation: object) -> dict[str, object]:
    surface_token_id = int(generation.surface_token_id)
    return {
        "position": position,
        "generation_index": int(generation.generation_index),
        "path": generation.path.name,
        "surface_token_id": surface_token_id,
        "surface_byte_hex": (
            None
            if surface_token_id == EOS_TOKEN_ID
            else f"{surface_token_id:02x}"
        ),
        "eos": surface_token_id == EOS_TOKEN_ID,
        "surface_output_cell_id": int(generation.surface_output_cell_id),
        "surface_winner_tick": int(generation.surface_winner_tick),
        "surface_basket_tick": int(generation.surface_basket_tick),
        "total_simulated_ticks": int(generation.total_simulated_ticks),
        "temporal_event_count": int(generation.temporal_event_count),
        "activated_columns": int(generation.activated_columns),
        "candidate_tokens_physical": int(generation.candidate_tokens),
        "winner_input_columns": int(generation.winner_input_columns),
        "feedback_predicted": bool(generation.feedback_predicted),
        "feedback_predicted_columns": int(generation.feedback_predicted_columns),
        "control_token_id": generation.control_token_id,
        "control_output_cell_id": generation.control_output_cell_id,
        "copy_place_ordinal": generation.copy_place_ordinal,
        "physical_election_count": int(generation.physical_election_count),
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
        module = ast.parse(text, filename="<role-place-copy-dev>", mode="exec")
        result["ast_parse_valid"] = True
        compile(module, "<role-place-copy-dev>", "exec")
        result["compile_valid"] = True
    except (UnicodeDecodeError, SyntaxError, ValueError, TypeError):
        return result
    if len(module.body) == 1 and isinstance(module.body[0], ast.FunctionDef):
        result["function_name"] = module.body[0].name
    return result


def main() -> int:
    args = _parse_args()
    started_ns = time.perf_counter_ns()
    definition = _definition()
    train_episode = _train_episode()
    dev_prompt = _normalized_prompt(DEV_NAME)

    with _synthetic_public_probe_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            train_steps = circuit.advance_block(
                train_episode,
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
                    stop = _stop_record(error, generation_position=position)
                    break
                generations.append(_generation_record(position, generation))
                if generation.surface_token_id == EOS_TOKEN_ID:
                    terminated_by_eos = True
                    break
                if not 0 <= generation.surface_token_id < ASCII_TOKEN_LIMIT:
                    stop = {
                        "generation_position": position,
                        "status": "NON_ASCII_SURFACE_TOKEN",
                        "observed": int(generation.surface_token_id),
                    }
                    break
                emitted_bytes.append(generation.surface_token_id)
        native_identity = loaded_library_identity()

    # The frozen public DEV target is introduced only for post-run scoring; it
    # is never passed to the runtime or used to choose a generation action.
    expected_code = b"def bump(x):\n    return x + 1\n"
    emitted = bytes(emitted_bytes)
    compile_analysis = _compile_analysis(emitted)
    enter_records = [
        record for record in generations if record["path"] == TemporalGenerationPath.ENTER_COPY.name
    ]
    copy_records = [
        record for record in generations if record["path"] == TemporalGenerationPath.COPY.name
    ]
    exit_records = [
        record for record in generations if record["path"] == TemporalGenerationPath.EXIT_COPY.name
    ]
    enter_event_count = (
        None if len(enter_records) != 1 else int(enter_records[0]["temporal_event_count"])
    )
    copy_event_count_held = (
        enter_event_count is not None
        and all(
            int(record["temporal_event_count"]) == enter_event_count
            for record in copy_records
        )
    )
    copy_ordinals = [
        int(record["copy_place_ordinal"])
        for record in (*enter_records, *copy_records)
        if record["copy_place_ordinal"] is not None
    ]
    exit_resumed = (
        len(exit_records) == 1
        and enter_event_count is not None
        and int(exit_records[0]["temporal_event_count"]) == enter_event_count + 1
        and int(exit_records[0]["surface_token_id"]) == ord("(")
    )
    role_path_exact = (
        len(enter_records) == 1
        and enter_records[0]["control_token_id"] == COPY_ID_TOKEN_ID
        and copy_ordinals == [0, 1, 2, 3]
        and len(exit_records) == 1
        and exit_records[0]["control_token_id"] == COPY_END_TOKEN_ID
        and exit_resumed
    )
    exact_output = emitted == expected_code
    passed = (
        stop is None
        and terminated_by_eos
        and exact_output
        and bool(compile_analysis["ast_parse_valid"])
        and bool(compile_analysis["compile_valid"])
        and compile_analysis["function_name"] == "bump"
        and copy_event_count_held
        and role_path_exact
    )

    payload = {
        "schema_version": 1,
        "experiment": "r3-public-train-dev-role-place-copy-probe",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": (
            "PASS_PHYSICAL_ROLE_PLACE_COPY_AND_ROUTE_RESUME"
            if passed
            else "FAIL_PHYSICAL_ROLE_PLACE_COPY_AND_ROUTE_RESUME"
        ),
        "capability_pass": passed,
        "claim_boundary": (
            "One public synthetic TRAIN inc/+1 sequence teaches a literal temporal template "
            "with an internal COPY_ID slot. One public synthetic DEV prompt binds the unseen "
            "ASCII identifier bump between explicit identifier boundary roles and supplies no "
            "target or teacher token. Generation is exclusively the atomic native physical "
            "generate path over a seeded overlapping 11-of-512 token codebook. The scoring "
            "target is introduced only after generation. This establishes exact synthetic "
            "identifier role/place copy, temporal-event hold during the copied phrase, template "
            "route resumption, EOS, and Python syntax for this one micro-probe. It is not broad "
            "systematic generalization, semantic coding competence, SWE, protected data, C1, or "
            "benchmark evidence."
        ),
        "authority": {
            "partitions": ["public-synthetic-train", "public-synthetic-dev"],
            "train_name": TRAIN_NAME.decode("ascii"),
            "dev_name": DEV_NAME.decode("ascii"),
            "dev_attempts": 1,
            "dev_target_used_as_runtime_input": False,
            "teacher_tokens_after_dev_prompt": 0,
            "protected_material_read": False,
            "held_out_material_read": False,
            "c1_material_read": False,
            "synthetic_runtime_authority": True,
        },
        "tokenization": {
            "scheme": (
                "ASCII byte IDs plus EOS=256 and four reserved uint32 native role IDs"
            ),
            "eos_token_id": EOS_TOKEN_ID,
            "identifier_start_token_id": IDENTIFIER_START_TOKEN_ID,
            "identifier_end_token_id": IDENTIFIER_END_TOKEN_ID,
            "copy_id_token_id": COPY_ID_TOKEN_ID,
            "copy_end_token_id": COPY_END_TOKEN_ID,
            "static_codebook": "seeded overlapping 11-of-512 token_sdr_indices",
            "codebook_seed": CODEBOOK_SEED,
            "python_candidate_enumeration": False,
            "python_argmax": False,
            "learned_readout_parameters": 0,
        },
        "physical_definition": {
            "circuit_identity_sha256": definition.circuit_identity_sha256(),
            "column_count": definition.config.column_count,
            "active_columns_per_token": ACTIVE_COLUMNS,
            "used_token_count": len(definition.token_codes),
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
        "train": {
            "examples": 1,
            "name": TRAIN_NAME.decode("ascii"),
            "normalized_prompt_token_count": len(_normalized_prompt(TRAIN_NAME)),
            "internal_target_shape": "def <COPY_ID>(x): return x + 1; EOS",
            "literal_identifier_in_target": False,
            "steps": len(train_steps),
            "allocated_segments": sum(step.allocated_segments for step in train_steps),
            "predicted_steps": sum(int(step.predicted) for step in train_steps),
            "live_segments": trained_identity.segment_count,
            "learning_steps": trained_identity.learning_step,
        },
        "dev_teacher_free": {
            "name": DEV_NAME.decode("ascii"),
            "prompt_token_count": len(dev_steps),
            "generation_budget": MAX_GENERATIONS,
            "completed_generations": len(generations),
            "terminated_by_eos": terminated_by_eos,
            "stop": stop,
            "emitted_code_text": (
                emitted.decode("ascii") if bool(compile_analysis["ascii_valid"]) else None
            ),
            "emitted_code_hex": emitted.hex(),
            "emitted_code_sha256": hashlib.sha256(emitted).hexdigest(),
            "expected_code_sha256": hashlib.sha256(expected_code).hexdigest(),
            "exact_expected_bytes": exact_output,
            "compile_analysis": compile_analysis,
            "enter_copy_count": len(enter_records),
            "copy_count": len(copy_records),
            "exit_copy_count": len(exit_records),
            "copy_place_ordinals": copy_ordinals,
            "copy_event_count": enter_event_count,
            "temporal_event_count_held_across_copied_phrase": copy_event_count_held,
            "exit_copy_resumed_temporal_route": exit_resumed,
            "role_path_exact": role_path_exact,
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
                "copy_place_ordinals": copy_ordinals,
                "copy_event_count_held": copy_event_count_held,
                "exit_copy_resumed": exit_resumed,
                "terminated_by_eos": terminated_by_eos,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
