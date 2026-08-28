"""PUBLIC synthetic single-shot probe for operator routing plus identifier copy.

TRAIN teaches reset-separated ``inc/+1`` and ``dec/-1`` templates.  Target
entry points use ordinal slot-0 recall, never literal names.  Teacher-free
DEV then presents unseen ``bump/+1`` and ``drop/-1`` prompts once each, with a
transient reset between them, and uses only atomic native ``generate()``.
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


OUTPUT_PATH = Path(__file__).with_name(
    "public_train_dev_role_place_operator_routing_probe.json"
)
TRAIN_CASES = (
    {"name": b"inc", "route": b"adds one", "operator": b"+"},
    {"name": b"dec", "route": b"subtracts one", "operator": b"-"},
)
DEV_CASES = (
    {"name": b"bump", "route": b"adds one", "operator": b"+"},
    {"name": b"drop", "route": b"subtracts one", "operator": b"-"},
)
PROMPT_PREFIX = b"Function "
PROMPT_TERMINATOR = b".\n"
TARGET_PREFIX = b"def "
TARGET_BEFORE_OPERATOR = b"(x):\n    return x "
TARGET_AFTER_OPERATOR = b" 1\n"
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


def _normalized_prompt(name: bytes, route: bytes) -> tuple[int, ...]:
    return (
        *PROMPT_PREFIX,
        slot_start_token_id(0),
        *name,
        slot_end_token_id(0),
        ord(" "),
        *route,
        *PROMPT_TERMINATOR,
    )


def _internal_target(operator: bytes) -> tuple[int, ...]:
    return (
        *TARGET_PREFIX,
        copy_slot_token_id(0),
        *TARGET_BEFORE_OPERATOR,
        operator[0],
        *TARGET_AFTER_OPERATOR,
        EOS_TOKEN_ID,
    )


def _train_episode(case: dict[str, bytes]) -> tuple[int, ...]:
    return (
        *_normalized_prompt(case["name"], case["route"]),
        *_internal_target(case["operator"]),
    )


def _expected_code(case: dict[str, bytes]) -> bytes:
    return (
        TARGET_PREFIX
        + case["name"]
        + TARGET_BEFORE_OPERATOR
        + case["operator"]
        + TARGET_AFTER_OPERATOR
    )


def _definition() -> NativeTemporalCircuitDefinition:
    train_episodes = tuple(_train_episode(case) for case in TRAIN_CASES)
    dev_prompts = tuple(
        _normalized_prompt(case["name"], case["route"]) for case in DEV_CASES
    )
    used_token_ids = tuple(
        sorted(
            {
                *(token for episode in train_episodes for token in episode),
                *(token for prompt in dev_prompts for token in prompt),
                *(slot_start_token_id(slot) for slot in range(SLOT_COUNT)),
                *(slot_end_token_id(slot) for slot in range(SLOT_COUNT)),
                *(copy_slot_token_id(slot) for slot in range(SLOT_COUNT)),
                COPY_END_TOKEN_ID,
            }
        )
    )
    segment_capacity = sum(len(episode) - 1 for episode in train_episodes) * ACTIVE_COLUMNS
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
            None if surface_token_id == EOS_TOKEN_ID else f"{surface_token_id:02x}"
        ),
        "eos": surface_token_id == EOS_TOKEN_ID,
        "temporal_event_count": int(generation.temporal_event_count),
        "candidate_tokens_physical": int(generation.candidate_tokens),
        "winner_input_columns": int(generation.winner_input_columns),
        "feedback_predicted": bool(generation.feedback_predicted),
        "feedback_predicted_columns": int(generation.feedback_predicted_columns),
        "control_token_id": generation.control_token_id,
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
        module = ast.parse(text, filename="<role-place-operator-dev>", mode="exec")
        result["ast_parse_valid"] = True
        compile(module, "<role-place-operator-dev>", "exec")
        result["compile_valid"] = True
    except (UnicodeDecodeError, SyntaxError, ValueError, TypeError):
        return result
    if len(module.body) == 1 and isinstance(module.body[0], ast.FunctionDef):
        result["function_name"] = module.body[0].name
    return result


def _score_dev(
    case: dict[str, bytes],
    emitted: bytes,
    generations: list[dict[str, object]],
    *,
    terminated_by_eos: bool,
    stop: dict[str, object] | None,
) -> tuple[bool, dict[str, object]]:
    expected = _expected_code(case)
    compile_analysis = _compile_analysis(emitted)
    enter_records = [
        record
        for record in generations
        if record["path"] == TemporalGenerationPath.ENTER_COPY.name
    ]
    copy_records = [
        record
        for record in generations
        if record["path"] == TemporalGenerationPath.COPY.name
    ]
    exit_records = [
        record
        for record in generations
        if record["path"] == TemporalGenerationPath.EXIT_COPY.name
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
    role_path_exact = (
        len(enter_records) == 1
        and enter_records[0]["control_token_id"] == copy_slot_token_id(0)
        and copy_ordinals == list(range(len(case["name"])))
        and len(exit_records) == 1
        and exit_records[0]["control_token_id"] == COPY_END_TOKEN_ID
        and enter_event_count is not None
        and int(exit_records[0]["temporal_event_count"]) == enter_event_count + 1
        and int(exit_records[0]["surface_token_id"]) == ord("(")
    )
    operator_position = emitted.find(case["operator"])
    operator_record = (
        None
        if operator_position < 0 or operator_position >= len(generations)
        else generations[operator_position]
    )
    operator_route_exact = (
        operator_record is not None
        and operator_record["path"] == TemporalGenerationPath.TEMPORAL.name
        and int(operator_record["surface_token_id"]) == case["operator"][0]
    )
    exact_output = emitted == expected
    passed = (
        stop is None
        and terminated_by_eos
        and exact_output
        and bool(compile_analysis["ast_parse_valid"])
        and bool(compile_analysis["compile_valid"])
        and compile_analysis["function_name"] == case["name"].decode("ascii")
        and copy_event_count_held
        and role_path_exact
        and operator_route_exact
    )
    return passed, {
        "name": case["name"].decode("ascii"),
        "learned_prompt_route": case["route"].decode("ascii"),
        "expected_operator": case["operator"].decode("ascii"),
        "attempts": 1,
        "target_used_as_runtime_input": False,
        "teacher_tokens_after_prompt": 0,
        "terminated_by_eos": terminated_by_eos,
        "stop": stop,
        "completed_generations": len(generations),
        "emitted_code_text": (
            emitted.decode("ascii") if bool(compile_analysis["ascii_valid"]) else None
        ),
        "emitted_code_hex": emitted.hex(),
        "emitted_code_sha256": hashlib.sha256(emitted).hexdigest(),
        "expected_code_sha256": hashlib.sha256(expected).hexdigest(),
        "exact_expected_bytes": exact_output,
        "compile_analysis": compile_analysis,
        "copy_place_ordinals": copy_ordinals,
        "temporal_event_count_held_across_copied_phrase": copy_event_count_held,
        "role_path_exact": role_path_exact,
        "operator_generation_position": operator_position,
        "operator_generation": operator_record,
        "operator_route_exact": operator_route_exact,
        "passed": passed,
        "generations": generations,
    }


def main() -> int:
    args = _parse_args()
    started_ns = time.perf_counter_ns()
    definition = _definition()
    train_episodes = tuple(_train_episode(case) for case in TRAIN_CASES)

    with _synthetic_public_probe_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            train_steps = []
            for episode in train_episodes:
                train_steps.append(
                    circuit.advance_block(
                        episode,
                        mode=TemporalStepMode.LEARN,
                        reset_before=True,
                    )
                )
            trained_identity = circuit.checkpoint_identity()

            dev_results: list[dict[str, object]] = []
            dev_passes: list[bool] = []
            for case in DEV_CASES:
                prompt_steps = circuit.advance_block(
                    _normalized_prompt(case["name"], case["route"]),
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
                dev_passed, dev_record = _score_dev(
                    case,
                    bytes(emitted_bytes),
                    generations,
                    terminated_by_eos=terminated_by_eos,
                    stop=stop,
                )
                dev_record["prompt_steps"] = len(prompt_steps)
                dev_results.append(dev_record)
                dev_passes.append(dev_passed)

            final_identity = circuit.checkpoint_identity()
        native_identity = loaded_library_identity()

    learned_state_held = (
        final_identity.segment_count == trained_identity.segment_count
        and final_identity.cohort_count == trained_identity.cohort_count
        and final_identity.next_reserve_cell == trained_identity.next_reserve_cell
        and final_identity.learning_step == trained_identity.learning_step
    )
    operators_discriminated = (
        len(dev_results) == 2
        and dev_results[0]["expected_operator"] == "+"
        and dev_results[1]["expected_operator"] == "-"
        and dev_results[0]["operator_route_exact"] is True
        and dev_results[1]["operator_route_exact"] is True
    )
    passed = all(dev_passes) and learned_state_held and operators_discriminated
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-train-dev-role-place-operator-routing-probe",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": (
            "PASS_OPERATOR_ROUTING_PLUS_NOVEL_NAME_COPY"
            if passed
            else "STOP_OPERATOR_ROUTING_PLUS_NOVEL_NAME_COPY"
        ),
        "capability_pass": passed,
        "claim_boundary": (
            "Two public synthetic TRAIN episodes teach inc/+1 and dec/-1 templates with "
            "explicit slot-0 boundaries and internal ordinal slot-0 target recalls. Teacher-free "
            "public synthetic DEV presents unseen bump/+1 and drop/-1 prompts exactly once "
            "each, resetting transient state while retaining learned state. All surface tokens "
            "and EOS come from unchanged atomic native generate(). This tests only two-way "
            "learned prompt-route operator discrimination combined with novel-name role/place "
            "copy; it is not broad systematic generalization, semantic coding competence, SWE, "
            "protected data, C1, or benchmark evidence."
        ),
        "authority": {
            "partitions": ["public-synthetic-train", "public-synthetic-dev"],
            "train_names": [case["name"].decode("ascii") for case in TRAIN_CASES],
            "dev_names": [case["name"].decode("ascii") for case in DEV_CASES],
            "dev_attempts_each": 1,
            "dev_targets_used_as_runtime_input": False,
            "teacher_tokens_after_dev_prompts": 0,
            "transient_reset_between_dev_prompts": True,
            "learned_state_retained_between_dev_prompts": True,
            "protected_material_read": False,
            "held_out_material_read": False,
            "c1_material_read": False,
            "synthetic_runtime_authority": True,
        },
        "tokenization": {
            "scheme": "ASCII byte IDs plus EOS=256 and four reserved uint32 native role IDs",
            "eos_token_id": EOS_TOKEN_ID,
            "slot_0_start_token_id": slot_start_token_id(0),
            "slot_0_end_token_id": slot_end_token_id(0),
            "copy_slot_0_token_id": copy_slot_token_id(0),
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
            "theta_match": definition.config.theta_match,
            "theta_active": definition.config.theta_active,
            "context_policy": definition.config.context_policy.name,
            "plasticity_policy": definition.config.plasticity_policy.name,
        },
        "train": {
            "examples": 2,
            "templates": [
                {
                    "name": case["name"].decode("ascii"),
                    "prompt_route": case["route"].decode("ascii"),
                    "operator": case["operator"].decode("ascii"),
                    "internal_target_name": "COPY[0]",
                    "literal_identifier_in_target": False,
                    "steps": len(steps),
                    "allocated_segments": sum(step.allocated_segments for step in steps),
                    "predicted_steps": sum(int(step.predicted) for step in steps),
                }
                for case, steps in zip(TRAIN_CASES, train_steps, strict=True)
            ],
            "live_segments": trained_identity.segment_count,
            "learning_steps": trained_identity.learning_step,
        },
        "dev_teacher_free": dev_results,
        "post_dev_invariants": {
            "learned_state_held": learned_state_held,
            "operators_discriminated": operators_discriminated,
            "trained_segment_count": trained_identity.segment_count,
            "final_segment_count": final_identity.segment_count,
            "trained_learning_step": trained_identity.learning_step,
            "final_learning_step": final_identity.learning_step,
            "trained_next_reserve_cell": trained_identity.next_reserve_cell,
            "final_next_reserve_cell": final_identity.next_reserve_cell,
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
                "dev": [
                    {
                        "name": result["name"],
                        "operator": result["expected_operator"],
                        "exact": result["exact_expected_bytes"],
                        "eos": result["terminated_by_eos"],
                    }
                    for result in dev_results
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
