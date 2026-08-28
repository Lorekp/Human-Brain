"""One-shot visible Simple-SWE scalar-offset probe for the ordinal slot bank.

Two visible train tasks teach one normalized template. A third visible train
task is used exactly once as same-family development: only its public prompt
fields reach the circuit, then atomic native ``generate()`` emits the function.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Iterator
from dataclasses import asdict
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
from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.sandbox import (
    SandboxConfigurationError,
    execute_completion,
    load_worker_snapshot,
    python_runtime_identity,
    resolve_python_executable,
)
from humanbrain.simple_swe.tasks import GeneratedTask, generate_visible_partition


OUTPUT_PATH = Path(__file__).with_name(
    "public_simple_swe_slot_bank_scalar_offset_probe.json"
)
PROTOCOL_PATH = REPO_ROOT / "benchmarks" / "simple_swe_protocol_v1.json"
FAMILY = "t0_scalar_offset"
TRAIN_ORDINALS = (0, 1)
DEVELOPMENT_ORDINAL = 3
EXPECTED_SOURCE_RECORDS = {
    0: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/000",
        "entry_point": "solve_d4c2e76cb6f6e7b8",
        "parameter": "n",
        "literal": "5",
        "task_sha256": "76a3a308651bb98a67801dbcbf7d2750a6de6f30748d6f500a30121dc950ea21",
    },
    1: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/001",
        "entry_point": "solve_aeb58999626eec7c",
        "parameter": "n",
        "literal": "-7",
        "task_sha256": "65087aed8fc475d533af01823cb5bcd2d5ca6b7c66537a5c9eec9f1f973b0587",
    },
    3: {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/003",
        "entry_point": "solve_c5fdae9089d40e29",
        "parameter": "n",
        "literal": "7",
        "task_sha256": "b6890a1705d8fef59a981af7858538023f876c65412d57af2eacdf21f645dc15",
    },
}
PROMPT_PREFIX = b"Function "
PROMPT_PARAMETER_SEPARATOR = b"("
PROMPT_OFFSET_ROLE = b") offset "
PROMPT_SUFFIX = b".\n"
TARGET_PREFIX = b"def "
TARGET_AFTER_ENTRY = b"("
TARGET_AFTER_PARAMETER = b"):\n    return "
TARGET_BEFORE_LITERAL = b" + "
TARGET_SUFFIX = b"\n"
EOS_TOKEN_ID = 256
COLUMN_COUNT = 512
ACTIVE_COLUMNS = 11
CODEBOOK_SEED = 0
BASE_CELLS_PER_COLUMN = 8
CONSTRUCTION_SEED = 17
MAX_GENERATIONS = 96


@contextlib.contextmanager
def _public_development_authority() -> Iterator[None]:
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
    parser.add_argument("--python-executable", type=Path)
    return parser.parse_args()


def _select_and_confirm_tasks() -> tuple[object, tuple[GeneratedTask, ...], GeneratedTask]:
    protocol = load_protocol(PROTOCOL_PATH)
    visible_train = generate_visible_partition(protocol, "train")
    selected: dict[int, GeneratedTask] = {}
    for ordinal in (*TRAIN_ORDINALS, DEVELOPMENT_ORDINAL):
        task = next(
            item
            for item in visible_train
            if item.public.family == FAMILY and item.public.ordinal == ordinal
        )
        expected = EXPECTED_SOURCE_RECORDS[ordinal]
        parameters = task.public.signature.parameters
        observed = {
            "task_id": task.public.task_id,
            "entry_point": task.public.signature.entry_point,
            "parameter": parameters[0].name if len(parameters) == 1 else None,
            "literal": expected["literal"],
            "task_sha256": task.public.task_sha256,
        }
        if observed != expected:
            raise RuntimeError(
                f"visible scalar-offset source record {ordinal} changed: {observed!r}"
            )
        if f"fixed offset {expected['literal']}." not in task.public.prompt:
            raise RuntimeError(f"visible scalar-offset literal {ordinal} differs from source")
        if len(task.private.regression_cases) != 4 or len(task.private.property_cases) != 32:
            raise RuntimeError("visible scalar-offset sandbox case counts changed")
        selected[ordinal] = task
    return (
        protocol,
        tuple(selected[ordinal] for ordinal in TRAIN_ORDINALS),
        selected[DEVELOPMENT_ORDINAL],
    )


def _task_fields(task: GeneratedTask) -> tuple[bytes, bytes, bytes]:
    expected = EXPECTED_SOURCE_RECORDS[task.public.ordinal]
    return (
        task.public.signature.entry_point.encode("ascii"),
        task.public.signature.parameters[0].name.encode("ascii"),
        str(expected["literal"]).encode("ascii"),
    )


def _normalized_prompt(task: GeneratedTask) -> tuple[int, ...]:
    entry_point, parameter, literal = _task_fields(task)
    return (
        *PROMPT_PREFIX,
        slot_start_token_id(0),
        *entry_point,
        slot_end_token_id(0),
        *PROMPT_PARAMETER_SEPARATOR,
        slot_start_token_id(1),
        *parameter,
        slot_end_token_id(1),
        *PROMPT_OFFSET_ROLE,
        slot_start_token_id(2),
        *literal,
        slot_end_token_id(2),
        *PROMPT_SUFFIX,
    )


def _internal_target() -> tuple[int, ...]:
    return (
        *TARGET_PREFIX,
        copy_slot_token_id(0),
        *TARGET_AFTER_ENTRY,
        copy_slot_token_id(1),
        *TARGET_AFTER_PARAMETER,
        copy_slot_token_id(1),
        *TARGET_BEFORE_LITERAL,
        copy_slot_token_id(2),
        *TARGET_SUFFIX,
        EOS_TOKEN_ID,
    )


def _train_episode(task: GeneratedTask) -> tuple[int, ...]:
    return (*_normalized_prompt(task), *_internal_target())


def _definition(
    train_tasks: tuple[GeneratedTask, ...], development_task: GeneratedTask
) -> NativeTemporalCircuitDefinition:
    train_episodes = tuple(_train_episode(task) for task in train_tasks)
    development_prompt = _normalized_prompt(development_task)
    control_tokens = {
        *(slot_start_token_id(slot) for slot in range(SLOT_COUNT)),
        *(slot_end_token_id(slot) for slot in range(SLOT_COUNT)),
        *(copy_slot_token_id(slot) for slot in range(SLOT_COUNT)),
        COPY_END_TOKEN_ID,
    }
    used_token_ids = tuple(
        sorted(
            {
                *(token for episode in train_episodes for token in episode),
                *development_prompt,
                *control_tokens,
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
    }


def _generate_once(circuit: NativeTemporalCircuit) -> dict[str, object]:
    emitted = bytearray()
    generations: list[dict[str, object]] = []
    stop: dict[str, object] | None = None
    terminated_by_eos = False
    for position in range(MAX_GENERATIONS):
        try:
            generation = circuit.generate()
        except NativeTemporalRuntimeError as error:
            stop = {
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
                "slot_ordinal": generation["slot_ordinal"],
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
                exact &= generation["slot_ordinal"] == active["slot_ordinal"]
                active["copied_places"].append(generation["copy_place_ordinal"])
                active["copied_event_counts"].append(event_count)
        elif path == TemporalGenerationPath.EXIT_COPY.name:
            exact &= active is not None
            if active is not None:
                exact &= generation["slot_ordinal"] == active["slot_ordinal"]
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
    expected_slots = [0, 1, 1, 2]
    expected_controls = [copy_slot_token_id(slot) for slot in expected_slots]
    exact &= [record["slot_ordinal"] for record in transactions] == expected_slots
    exact &= [record["control_token_id"] for record in transactions] == expected_controls
    exact &= [record["copied_places"] for record in transactions] == [
        list(range(22)),
        [0],
        [0],
        [0],
    ]
    exact &= all(
        bool(record["control_advanced_temporal_once"])
        and bool(record["temporal_event_count_held_across_copied_bytes"])
        and bool(record["exit_resumed_temporal_once"])
        for record in transactions
    )
    return transactions, exact


def _expected_source_after_generation(task: GeneratedTask) -> bytes:
    entry_point, parameter, literal = _task_fields(task)
    return (
        TARGET_PREFIX
        + entry_point
        + TARGET_AFTER_ENTRY
        + parameter
        + TARGET_AFTER_PARAMETER
        + parameter
        + TARGET_BEFORE_LITERAL
        + literal
        + TARGET_SUFFIX
    )


def _task_record(task: GeneratedTask) -> dict[str, object]:
    entry_point, parameter, literal = _task_fields(task)
    return {
        "task_id": task.public.task_id,
        "partition": task.public.partition,
        "family": task.public.family,
        "ordinal": task.public.ordinal,
        "entry_point": entry_point.decode("ascii"),
        "parameter": parameter.decode("ascii"),
        "literal": literal.decode("ascii"),
        "task_sha256": task.public.task_sha256,
        "hidden_test_sha256": task.public.hidden_test_sha256,
        "regression_case_count": len(task.private.regression_cases),
        "property_case_count": len(task.private.property_cases),
    }


def main() -> int:
    args = _parse_args()
    started_ns = time.perf_counter_ns()
    protocol, train_tasks, development_task = _select_and_confirm_tasks()
    definition = _definition(train_tasks, development_task)
    train_records: list[dict[str, object]] = []

    with _public_development_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            for task in train_tasks:
                episode = _train_episode(task)
                steps = circuit.advance_block(
                    episode,
                    mode=TemporalStepMode.LEARN,
                    reset_before=True,
                )
                train_records.append(
                    {
                        **_task_record(task),
                        "steps": len(steps),
                        "allocated_segments": sum(step.allocated_segments for step in steps),
                    }
                )
            trained_identity = circuit.checkpoint_identity()
            prompt_steps = circuit.advance_block(
                _normalized_prompt(development_task),
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            run = _generate_once(circuit)
        native_identity = loaded_library_identity()

    emitted = run.pop("emitted")
    expected = _expected_source_after_generation(development_task)
    exact_output = emitted == expected
    compile_valid = False
    compile_error: str | None = None
    if run["stop"] is None and run["terminated_by_eos"] and exact_output:
        try:
            compile(emitted.decode("ascii"), "<visible-slot-bank-development>", "exec")
            compile_valid = True
        except (UnicodeDecodeError, SyntaxError, ValueError, TypeError) as error:
            compile_error = type(error).__name__

    transactions, copy_trace_exact = _copy_transactions(run["generations"])
    first_failure: dict[str, object] | None = None
    if run["stop"] is not None:
        first_failure = {"gate": "GENERATION_STOP", "detail": run["stop"]}
    elif not run["terminated_by_eos"]:
        first_failure = {"gate": "ORDINARY_EOS", "detail": None}
    elif not exact_output:
        first_failure = {"gate": "EXACT_SOURCE_BYTES", "detail": None}
    elif not compile_valid:
        first_failure = {"gate": "PYTHON_COMPILE", "detail": compile_error}
    elif not copy_trace_exact:
        first_failure = {"gate": "ORDINAL_SLOT_TRACE", "detail": None}

    sandbox_record: dict[str, object] = {
        "attempted": False,
        "outcome": None,
        "passed": False,
        "sandboxed": False,
        "detail": "not run because an earlier gate stopped",
        "wall_seconds": 0.0,
    }
    if first_failure is None:
        try:
            executable = resolve_python_executable(args.python_executable)
            runtime_identity = python_runtime_identity(
                executable, protocol.raw.get("language")
            )
            worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
            worker_sha256 = hashlib.sha256(worker_path.read_bytes()).hexdigest()
            worker_source = load_worker_snapshot(worker_sha256)
            sandbox_result = execute_completion(
                emitted.decode("ascii"),
                development_task.public,
                development_task.private,
                protocol.execution,
                python_executable=executable,
                python_executable_sha256=runtime_identity["executable_sha256"],
                worker_source=worker_source,
                worker_sha256=worker_sha256,
            )
            sandbox_record = {
                "attempted": True,
                **asdict(sandbox_result),
                "regression_case_count": len(development_task.private.regression_cases),
                "property_case_count": len(development_task.private.property_cases),
                "worker_sha256": worker_sha256,
                "python_runtime": runtime_identity,
            }
        except SandboxConfigurationError as error:
            sandbox_record = {
                **sandbox_record,
                "detail": str(error),
                "configuration_stop": type(error).__name__,
            }
            first_failure = {
                "gate": "PUBLIC_SIMPLE_SWE_SANDBOX_CONFIGURATION",
                "detail": str(error),
            }
        else:
            if not sandbox_result.passed:
                first_failure = {
                    "gate": "PUBLIC_SIMPLE_SWE_SANDBOX",
                    "detail": sandbox_result.outcome,
                }

    passed = first_failure is None
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-ordinal-slot-bank-scalar-offset",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": (
            "PASS_VISIBLE_SAME_FAMILY_SLOT_BANK_SCALAR_OFFSET"
            if passed
            else "STOP_VISIBLE_SAME_FAMILY_SLOT_BANK_SCALAR_OFFSET"
        ),
        "capability_pass": passed,
        "first_failure": first_failure,
        "claim_boundary": (
            "Two public visible Simple-SWE train scalar-offset tasks (ordinals 000 and 001) "
            "teach one normalized physical template with a syntax-neutral three-slot bank. "
            "Visible train ordinal 003 is used exactly once as a same-family development "
            "probe: only its public entry point, unary parameter, and literal are captured; "
            "its target never enters the runtime. Atomic no-argument native generate() and "
            "fitted cell races surface the exact function and ordinary EOS, including repeated "
            "slot-1 recall. Post-generation scoring uses the existing public Simple-SWE "
            "preflight/sandbox with four regression and 32 property cases. This is visible "
            "same-family development evidence, not a family-disjoint split, M0, eval, broad "
            "coding generalization, protected-data evidence, or C1."
        ),
        "authority": {
            "source_partition_generated": "train",
            "eval_partition_generated": False,
            "train_ordinals": list(TRAIN_ORDINALS),
            "development_ordinal": DEVELOPMENT_ORDINAL,
            "development_attempts": 1,
            "development_target_used_as_runtime_input": False,
            "teacher_tokens_after_development_prompt": 0,
            "python_candidate_enumeration": False,
            "python_argmax": False,
            "learned_readout_parameters": 0,
            "synthetic_runtime_authority": True,
        },
        "protocol": {
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.sha256,
            "language": protocol.raw.get("language"),
        },
        "slot_bank": {
            "slot_count": SLOT_COUNT,
            "max_surface_tokens_per_slot": 22,
            "roles": {"0": "entry point", "1": "unary parameter", "2": "literal"},
            "copy_controls": [copy_slot_token_id(slot) for slot in range(SLOT_COUNT)],
            "copy_end_token_id": COPY_END_TOKEN_ID,
            "internal_target_shape": (
                "def COPY[0](COPY[1]): return COPY[1] + COPY[2]; EOS"
            ),
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
            "tasks": train_records,
            "reset_before_each": True,
            "learned_state_retained_between_tasks": True,
            "live_segments_after_train": trained_identity.segment_count,
            "learning_steps_after_train": trained_identity.learning_step,
        },
        "visible_development_teacher_free": {
            "task": _task_record(development_task),
            "prompt_steps": len(prompt_steps),
            "generation_budget": MAX_GENERATIONS,
            "completed_generations": len(run["generations"]),
            "terminated_by_ordinary_eos": run["terminated_by_eos"],
            "stop": run["stop"],
            "emitted_source": emitted.decode("ascii", errors="replace"),
            "emitted_source_hex": emitted.hex(),
            "emitted_source_sha256": hashlib.sha256(emitted).hexdigest(),
            "expected_source_sha256": hashlib.sha256(expected).hexdigest(),
            "exact_expected_bytes": exact_output,
            "python_compile_valid": compile_valid,
            "copy_entry_ordinals": [record["slot_ordinal"] for record in transactions],
            "repeated_slot_1_recall": [record["slot_ordinal"] for record in transactions].count(1)
            == 2,
            "temporal_event_count_held_across_all_copied_bytes": all(
                bool(record["temporal_event_count_held_across_copied_bytes"])
                for record in transactions
            ),
            "copy_trace_exact": copy_trace_exact,
            "copy_transactions": transactions,
            "generations": run["generations"],
        },
        "public_simple_swe_sandbox": sandbox_record,
        "runtime": {
            "elapsed_wall_ns": time.perf_counter_ns() - started_ns,
            "python": sys.version,
            "platform": platform.platform(),
            "native_source_digest": native_identity.source_digest,
            "native_binary_sha256": native_identity.binary_sha256,
            "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
                "first_failure": first_failure,
                "copy_entry_ordinals": payload["visible_development_teacher_free"][
                    "copy_entry_ordinals"
                ],
                "sandbox_outcome": sandbox_record["outcome"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
