"""PUBLIC two-route semantic probe over atomic physical role/place generation.

Two reset-separated TRAIN episodes teach retained add and subtract routes with
COPY_ID and COPY_LITERAL output controls. Each DEV prompt is presented once;
generation stops on the first failure and never receives a target or oracle.
Only after generation do exact-code, compilation, and scalar semantics checks
run over the surfaced public function.
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
    COPY_LITERAL_TOKEN_ID,
    IDENTIFIER_END_TOKEN_ID,
    IDENTIFIER_START_TOKEN_ID,
    LITERAL_END_TOKEN_ID,
    LITERAL_START_TOKEN_ID,
    NativeTemporalCircuit,
    NativeTemporalCircuitConfig,
    NativeTemporalCircuitDefinition,
    NativeTemporalRuntimeError,
    TemporalContextPolicy,
    TemporalCopySlot,
    TemporalGenerationPath,
    TemporalPlasticityPolicy,
    TemporalStepMode,
)
from humanbrain.sdr_union import token_sdr_indices


OUTPUT_PATH = Path(__file__).with_name(
    "public_train_dev_role_place_scalar_routing_probe.json"
)
TRAIN_CASES = (
    {"identifier": b"inc", "literal": b"1", "route": b"adds", "operator": b"+"},
    {
        "identifier": b"dec",
        "literal": b"2",
        "route": b"subtracts",
        "operator": b"-",
    },
)
# Prompt inputs only. Exact target bytes and semantic oracles are constructed
# after each teacher-free generation has completed.
DEV_PROMPTS = (
    {"identifier": b"boost", "literal": b"37", "route": b"adds"},
    {"identifier": b"drop", "literal": b"14", "route": b"subtracts"},
)
PUBLIC_SCALAR_INPUTS = (-5, 0, 9)
PROMPT_PREFIX = b"Function "
PROMPT_VALUE_ROLE = b" value "
PROMPT_ROUTE_SEPARATOR = b" "
PROMPT_SUFFIX = b".\n"
TARGET_PREFIX = b"def "
TARGET_BEFORE_OPERATOR = b"(x):\n    return x "
TARGET_BEFORE_LITERAL = b" "
TARGET_SUFFIX = b"\n"
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


def _normalized_prompt(case: dict[str, bytes]) -> tuple[int, ...]:
    # The route phrase follows LITERAL_END so context normalization erases the
    # novel scalar surface while preserving the learned add/subtract branch.
    return (
        *PROMPT_PREFIX,
        IDENTIFIER_START_TOKEN_ID,
        *case["identifier"],
        IDENTIFIER_END_TOKEN_ID,
        *PROMPT_VALUE_ROLE,
        LITERAL_START_TOKEN_ID,
        *case["literal"],
        LITERAL_END_TOKEN_ID,
        *PROMPT_ROUTE_SEPARATOR,
        *case["route"],
        *PROMPT_SUFFIX,
    )


def _internal_target(case: dict[str, bytes]) -> tuple[int, ...]:
    return (
        *TARGET_PREFIX,
        COPY_ID_TOKEN_ID,
        *TARGET_BEFORE_OPERATOR,
        case["operator"][0],
        *TARGET_BEFORE_LITERAL,
        COPY_LITERAL_TOKEN_ID,
        *TARGET_SUFFIX,
        EOS_TOKEN_ID,
    )


def _train_episode(case: dict[str, bytes]) -> tuple[int, ...]:
    return (*_normalized_prompt(case), *_internal_target(case))


def _definition() -> NativeTemporalCircuitDefinition:
    train_episodes = tuple(_train_episode(case) for case in TRAIN_CASES)
    dev_prompts = tuple(_normalized_prompt(case) for case in DEV_PROMPTS)
    role_tokens = {
        IDENTIFIER_START_TOKEN_ID,
        IDENTIFIER_END_TOKEN_ID,
        LITERAL_START_TOKEN_ID,
        LITERAL_END_TOKEN_ID,
        COPY_ID_TOKEN_ID,
        COPY_LITERAL_TOKEN_ID,
        COPY_END_TOKEN_ID,
    }
    used_token_ids = tuple(
        sorted(
            {
                *(token for episode in train_episodes for token in episode),
                *(token for prompt in dev_prompts for token in prompt),
                *role_tokens,
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


def _native_stop(error: NativeTemporalRuntimeError, *, position: int) -> dict[str, object]:
    return {
        "kind": "NATIVE_STOP",
        "generation_position": position,
        "status": error.status.name,
        "message": error.message,
        "required": error.required,
        "observed": error.observed,
        "location": error.location,
        "poison_code": error.poison_code,
        "stage_code": error.stage_code,
    }


def _generation_record(position: int, generation: object) -> dict[str, object]:
    token_id = int(generation.surface_token_id)
    return {
        "position": position,
        "generation_index": int(generation.generation_index),
        "path": generation.path.name,
        "copy_slot": None if generation.copy_slot is None else generation.copy_slot.name,
        "copy_place_ordinal": generation.copy_place_ordinal,
        "surface_token_id": token_id,
        "surface_byte_hex": None if token_id == EOS_TOKEN_ID else f"{token_id:02x}",
        "temporal_event_count": int(generation.temporal_event_count),
        "control_token_id": generation.control_token_id,
        "physical_election_count": int(generation.physical_election_count),
        "candidate_tokens_physical": int(generation.candidate_tokens),
        "winner_input_columns": int(generation.winner_input_columns),
    }


def _generate_teacher_free(circuit: NativeTemporalCircuit) -> dict[str, object]:
    emitted = bytearray()
    generations: list[dict[str, object]] = []
    stop: dict[str, object] | None = None
    terminated_by_eos = False
    for position in range(MAX_GENERATIONS):
        try:
            generation = circuit.generate()
        except NativeTemporalRuntimeError as error:
            stop = _native_stop(error, position=position)
            break
        generations.append(_generation_record(position, generation))
        if generation.surface_token_id == EOS_TOKEN_ID:
            terminated_by_eos = True
            break
        if not 0 <= generation.surface_token_id < 128:
            stop = {
                "kind": "NON_ASCII_SURFACE_TOKEN",
                "generation_position": position,
                "observed": int(generation.surface_token_id),
            }
            break
        emitted.append(generation.surface_token_id)
    if not terminated_by_eos and stop is None:
        stop = {
            "kind": "GENERATION_BUDGET_EXHAUSTED",
            "generation_budget": MAX_GENERATIONS,
        }
    return {
        "emitted": bytes(emitted),
        "generations": generations,
        "terminated_by_eos": terminated_by_eos,
        "stop": stop,
    }


def _copy_transactions(
    generations: list[dict[str, object]],
) -> tuple[list[dict[str, object]], bool]:
    transactions: list[dict[str, object]] = []
    active: dict[str, object] | None = None
    exact = True
    previous_event_count: int | None = None
    for generation in generations:
        path = str(generation["path"])
        event_count = int(generation["temporal_event_count"])
        if path == TemporalGenerationPath.ENTER_COPY.name:
            exact &= active is None and previous_event_count is not None
            active = {
                "slot": generation["copy_slot"],
                "control_token_id": generation["control_token_id"],
                "enter_event_count": event_count,
                "control_advanced_temporal_once": (
                    previous_event_count is not None
                    and event_count == previous_event_count + 1
                ),
                "copied_places": [generation["copy_place_ordinal"]],
                "copied_event_counts": [event_count],
                "exit_event_count": None,
                "exit_surface_token_id": None,
            }
        elif path == TemporalGenerationPath.COPY.name:
            exact &= active is not None
            if active is not None:
                exact &= generation["copy_slot"] == active["slot"]
                active["copied_places"].append(generation["copy_place_ordinal"])
                active["copied_event_counts"].append(event_count)
        elif path == TemporalGenerationPath.EXIT_COPY.name:
            exact &= active is not None
            if active is not None:
                exact &= generation["copy_slot"] == active["slot"]
                active["exit_event_count"] = event_count
                active["exit_surface_token_id"] = generation["surface_token_id"]
                enter_count = int(active["enter_event_count"])
                active["temporal_event_count_held_across_copied_bytes"] = all(
                    int(value) == enter_count for value in active["copied_event_counts"]
                )
                active["exit_resumed_temporal_once"] = event_count == enter_count + 1
                transactions.append(active)
                active = None
        previous_event_count = event_count
    exact &= active is None
    exact &= [record["slot"] for record in transactions] == [
        TemporalCopySlot.IDENTIFIER.name,
        TemporalCopySlot.LITERAL.name,
    ]
    exact &= [record["control_token_id"] for record in transactions] == [
        COPY_ID_TOKEN_ID,
        COPY_LITERAL_TOKEN_ID,
    ]
    exact &= all(
        bool(record["control_advanced_temporal_once"])
        and bool(record["temporal_event_count_held_across_copied_bytes"])
        and bool(record["exit_resumed_temporal_once"])
        for record in transactions
    )
    return transactions, exact


def _score_after_generation(
    prompt: dict[str, bytes],
    run: dict[str, object],
) -> dict[str, object]:
    # Exact output and semantic oracles are intentionally created only here,
    # after this DEV generation has irreversibly completed.
    operator = b"+" if prompt["route"] == b"adds" else b"-"
    expected = (
        TARGET_PREFIX
        + prompt["identifier"]
        + TARGET_BEFORE_OPERATOR
        + operator
        + TARGET_BEFORE_LITERAL
        + prompt["literal"]
        + TARGET_SUFFIX
    )
    emitted = run["emitted"]
    compile_analysis: dict[str, object] = {
        "ascii_valid": False,
        "ast_parse_valid": False,
        "compile_valid": False,
        "function_name": None,
    }
    namespace: dict[str, object] = {}
    try:
        source = emitted.decode("ascii", errors="strict")
        compile_analysis["ascii_valid"] = True
        module = ast.parse(source, filename="<role-place-scalar-dev>", mode="exec")
        compile_analysis["ast_parse_valid"] = True
        code = compile(module, "<role-place-scalar-dev>", "exec")
        compile_analysis["compile_valid"] = True
        if len(module.body) == 1 and isinstance(module.body[0], ast.FunctionDef):
            compile_analysis["function_name"] = module.body[0].name
        exec(code, {"__builtins__": {}}, namespace)
    except (UnicodeDecodeError, SyntaxError, ValueError, TypeError, NameError):
        pass

    name = prompt["identifier"].decode("ascii")
    literal = int(prompt["literal"].decode("ascii"))
    function = namespace.get(name)
    semantic_cases: list[dict[str, object]] = []
    semantic_exact = callable(function)
    if callable(function):
        for scalar_input in PUBLIC_SCALAR_INPUTS:
            expected_value = (
                scalar_input + literal if operator == b"+" else scalar_input - literal
            )
            try:
                observed_value = function(scalar_input)
            except Exception as error:  # Public generated function; record exact failure.
                observed_value = None
                semantic_exact = False
                error_name: str | None = type(error).__name__
            else:
                semantic_exact &= type(observed_value) is int and observed_value == expected_value
                error_name = None
            semantic_cases.append(
                {
                    "input": scalar_input,
                    "expected": expected_value,
                    "observed": observed_value,
                    "error": error_name,
                    "exact": observed_value == expected_value and error_name is None,
                }
            )

    transactions, transaction_structure = _copy_transactions(run["generations"])
    expected_places = [
        list(range(len(prompt["identifier"]))),
        list(range(len(prompt["literal"]))),
    ]
    copy_exact = bool(
        transaction_structure
        and [record["copied_places"] for record in transactions] == expected_places
    )
    exact_bytes = emitted == expected
    final_eos_exact = bool(
        run["generations"]
        and run["generations"][-1]["surface_token_id"] == EOS_TOKEN_ID
        and run["generations"][-1]["path"] == TemporalGenerationPath.TEMPORAL.name
    )
    passed = bool(
        run["stop"] is None
        and run["terminated_by_eos"]
        and final_eos_exact
        and exact_bytes
        and compile_analysis["compile_valid"]
        and compile_analysis["function_name"] == name
        and copy_exact
        and semantic_exact
    )
    return {
        "passed": passed,
        "operator_oracle_created_post_generation": operator.decode("ascii"),
        "expected_code_sha256": hashlib.sha256(expected).hexdigest(),
        "exact_expected_bytes": exact_bytes,
        "final_eos_exact": final_eos_exact,
        "compile_analysis": compile_analysis,
        "copy_transactions_exact": copy_exact,
        "copy_transactions": transactions,
        "semantic_inputs": list(PUBLIC_SCALAR_INPUTS),
        "semantic_cases": semantic_cases,
        "semantic_exact": semantic_exact,
    }


def _first_failed_gate(run: dict[str, object], score: dict[str, object]) -> str:
    if run["stop"] is not None:
        return "GENERATION_STOP"
    if not run["terminated_by_eos"]:
        return "EOS"
    if not score["exact_expected_bytes"]:
        return "EXACT_CODE"
    if not score["compile_analysis"]["compile_valid"]:
        return "PYTHON_COMPILE"
    if not score["copy_transactions_exact"]:
        return "ROLE_PLACE_COPY_TRACE"
    if not score["semantic_exact"]:
        return "SCALAR_SEMANTICS"
    return "UNKNOWN"


def main() -> int:
    args = _parse_args()
    started_ns = time.perf_counter_ns()
    definition = _definition()
    train_episodes = tuple(_train_episode(case) for case in TRAIN_CASES)
    dev_records: list[dict[str, object]] = []
    first_failure: dict[str, object] | None = None
    train_records: list[dict[str, object]] = []

    with _synthetic_public_probe_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            for index, (case, episode) in enumerate(zip(TRAIN_CASES, train_episodes)):
                steps = circuit.advance_block(
                    episode,
                    mode=TemporalStepMode.LEARN,
                    reset_before=True,
                )
                train_records.append(
                    {
                        "index": index,
                        "identifier": case["identifier"].decode("ascii"),
                        "literal": case["literal"].decode("ascii"),
                        "route": case["route"].decode("ascii"),
                        "operator": case["operator"].decode("ascii"),
                        "steps": len(steps),
                        "allocated_segments": sum(step.allocated_segments for step in steps),
                    }
                )
            trained_identity = circuit.checkpoint_identity()

            for index, prompt in enumerate(DEV_PROMPTS):
                prompt_steps = circuit.advance_block(
                    _normalized_prompt(prompt),
                    mode=TemporalStepMode.READ_ONLY,
                    reset_before=True,
                )
                run = _generate_teacher_free(circuit)
                score = _score_after_generation(prompt, run)
                emitted = run.pop("emitted")
                record = {
                    "index": index,
                    "identifier": prompt["identifier"].decode("ascii"),
                    "literal": prompt["literal"].decode("ascii"),
                    "route": prompt["route"].decode("ascii"),
                    "prompt_steps": len(prompt_steps),
                    "target_or_oracle_entered_runtime": False,
                    "teacher_tokens_after_prompt": 0,
                    "emitted_code_text": emitted.decode("ascii", errors="replace"),
                    "emitted_code_hex": emitted.hex(),
                    "emitted_code_sha256": hashlib.sha256(emitted).hexdigest(),
                    **run,
                    "post_generation_score": score,
                }
                dev_records.append(record)
                if not score["passed"]:
                    first_failure = {
                        "dev_index": index,
                        "identifier": record["identifier"],
                        "gate": _first_failed_gate(run, score),
                        "native_or_surface_stop": run["stop"],
                    }
                    break
        native_identity = loaded_library_identity()

    passed = first_failure is None and len(dev_records) == len(DEV_PROMPTS)
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-role-place-scalar-routing-probe",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": (
            "PASS_TWO_ROUTE_VARIABLE_LITERAL_SCALAR_SEMANTICS"
            if passed
            else "STOP_TWO_ROUTE_VARIABLE_LITERAL_SCALAR_SEMANTICS"
        ),
        "capability_pass": passed,
        "first_failure": first_failure,
        "claim_boundary": (
            "Two public synthetic reset-separated TRAIN episodes teach retained adds/1 and "
            "subtracts/2 templates with internal COPY_ID and COPY_LITERAL controls. Two public "
            "synthetic DEV prompts are attempted once in order and stop on first failure: unseen "
            "boost/adds/37 then drop/subtracts/14. Native atomic generate() and fitted cell races "
            "select all surface bytes and ordinary EOS; post-generation checks require exact "
            "compiling code, two physical copy slots including multi-digit literals, learned + "
            "versus - routing, and exact execution on three fixed public scalar inputs. This is "
            "a narrow two-route synthetic semantic probe, not broad compositional generalization, "
            "semantic coding competence, SWE, protected data, C1, or benchmark evidence."
        ),
        "authority": {
            "partitions": ["public-synthetic-train", "public-synthetic-dev"],
            "train_examples": len(TRAIN_CASES),
            "planned_dev_examples": len(DEV_PROMPTS),
            "attempted_dev_examples": len(dev_records),
            "dev_attempts_per_prompt": 1,
            "stop_on_first_failure": True,
            "dev_target_used_as_runtime_input": False,
            "teacher_tokens_after_dev_prompt": 0,
            "python_candidate_enumeration": False,
            "python_argmax": False,
            "learned_readout_parameters": 0,
            "protected_or_c1_material_read": False,
            "synthetic_runtime_authority": True,
        },
        "normalization": {
            "prompt_shape": (
                "Function <IDENTIFIER> value <LITERAL> <ROUTE>. Route follows LITERAL_END so "
                "the normalized temporal context retains learned adds/subtracts discrimination."
            ),
            "identifier_start_token_id": IDENTIFIER_START_TOKEN_ID,
            "identifier_end_token_id": IDENTIFIER_END_TOKEN_ID,
            "literal_start_token_id": LITERAL_START_TOKEN_ID,
            "literal_end_token_id": LITERAL_END_TOKEN_ID,
            "copy_id_token_id": COPY_ID_TOKEN_ID,
            "copy_literal_token_id": COPY_LITERAL_TOKEN_ID,
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
        },
        "train": {
            "episodes": train_records,
            "reset_before_each": True,
            "learned_state_retained_between_episodes": True,
            "internal_target_shape": (
                "def <COPY_ID>(x): return x <LEARNED_OPERATOR> <COPY_LITERAL>; EOS"
            ),
            "literal_identifier_or_value_in_target": False,
            "live_segments_after_train": trained_identity.segment_count,
            "learning_steps_after_train": trained_identity.learning_step,
        },
        "dev_teacher_free": dev_records,
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
                "attempted_dev_examples": len(dev_records),
                "first_failure": first_failure,
                "semantic_passes": [
                    record["post_generation_score"]["semantic_exact"]
                    for record in dev_records
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
