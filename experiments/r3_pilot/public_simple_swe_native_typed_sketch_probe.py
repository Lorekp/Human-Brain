"""One-shot native typed-sketch probe on two visible Simple-SWE families.

One retained native temporal circuit learns a scalar-add sketch and a list-map
sketch.  Public task wording selects the sketch; ordinal slot 2 physically
copies the canonical signed literal tokens into the existing Layer-1 typed
program compiler.
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
from humanbrain.typed_program import Program, compile as compile_typed_program


OUTPUT_PATH = Path(__file__).with_name(
    "public_simple_swe_native_typed_sketch_probe.json"
)
PROTOCOL_PATH = REPO_ROOT / "benchmarks" / "simple_swe_protocol_v1.json"
SCALAR_FAMILY = "t0_scalar_offset"
LIST_FAMILY = "t1_list_map_add"
TRAIN_KEYS = ((SCALAR_FAMILY, 0), (LIST_FAMILY, 0))
DEVELOPMENT_KEYS = ((SCALAR_FAMILY, 3), (LIST_FAMILY, 7))
EXPECTED_SOURCE_RECORDS = {
    (SCALAR_FAMILY, 0): {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/000",
        "entry_point": "solve_d4c2e76cb6f6e7b8",
        "parameter": "n",
        "literal": "5",
        "task_sha256": "76a3a308651bb98a67801dbcbf7d2750a6de6f30748d6f500a30121dc950ea21",
    },
    (LIST_FAMILY, 0): {
        "task_id": "simple-swe-v1/train/tier-1/t1_list_map_add/000",
        "entry_point": "solve_e6ed84e84252b575",
        "parameter": "values",
        "literal": "5",
        "task_sha256": "d2700b281f9839c0f07abd0717e23ea14bfa49a48f9214667609f5d333caf698",
    },
    (SCALAR_FAMILY, 3): {
        "task_id": "simple-swe-v1/train/tier-0/t0_scalar_offset/003",
        "entry_point": "solve_c5fdae9089d40e29",
        "parameter": "n",
        "literal": "7",
        "task_sha256": "b6890a1705d8fef59a981af7858538023f876c65412d57af2eacdf21f645dc15",
    },
    (LIST_FAMILY, 7): {
        "task_id": "simple-swe-v1/train/tier-1/t1_list_map_add/007",
        "entry_point": "solve_f1450339c6dbe783",
        "parameter": "values",
        "literal": "3",
        "task_sha256": "b4463745a7934e52239a21db0b9ebcac343dccd87032515f44fa1195f9872e85",
    },
}

# Native token IDs are intentionally arbitrary.  Their only public meaning is
# this lossless adapter to the already-established Layer-1 primitive strings.
_PRIMITIVE_ORDER = (
    "add",
    "variable-0",
    "integer",
    "+",
    "-",
    "end",
    "map",
    "bind",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)
PRIMITIVE_TOKEN_IDS = {
    primitive: 1024 + ordinal for ordinal, primitive in enumerate(_PRIMITIVE_ORDER)
}
PRIMITIVE_BY_TOKEN_ID = {
    token_id: primitive for primitive, token_id in PRIMITIVE_TOKEN_IDS.items()
}

EOS_TOKEN_ID = 256
COLUMN_COUNT = 512
ACTIVE_COLUMNS = 11
CODEBOOK_SEED = 0
BASE_CELLS_PER_COLUMN = 8
CONSTRUCTION_SEED = 17
MAX_GENERATIONS = 24


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
    parser.add_argument("--python-executable", type=Path, required=True)
    return parser.parse_args()


def _select_and_confirm_tasks() -> tuple[
    object, tuple[GeneratedTask, ...], tuple[GeneratedTask, ...]
]:
    protocol = load_protocol(PROTOCOL_PATH)
    visible_train = generate_visible_partition(protocol, "train")
    selected: dict[tuple[str, int], GeneratedTask] = {}
    for key in (*TRAIN_KEYS, *DEVELOPMENT_KEYS):
        family, ordinal = key
        task = next(
            item
            for item in visible_train
            if item.public.family == family and item.public.ordinal == ordinal
        )
        expected = EXPECTED_SOURCE_RECORDS[key]
        parameters = task.public.signature.parameters
        observed = {
            "task_id": task.public.task_id,
            "entry_point": task.public.signature.entry_point,
            "parameter": parameters[0].name if len(parameters) == 1 else None,
            "literal": expected["literal"],
            "task_sha256": task.public.task_sha256,
        }
        if observed != expected:
            raise RuntimeError(f"visible task source record changed for {key!r}: {observed!r}")
        literal = str(expected["literal"])
        literal_phrase = (
            f"fixed offset {literal}."
            if family == SCALAR_FAMILY
            else f"with {literal} added to every integer"
        )
        if literal_phrase not in task.public.prompt:
            raise RuntimeError(f"visible literal wording changed for {key!r}")
        if len(task.private.regression_cases) != 4 or len(task.private.property_cases) != 32:
            raise RuntimeError(f"visible sandbox case counts changed for {key!r}")
        selected[key] = task
    return (
        protocol,
        tuple(selected[key] for key in TRAIN_KEYS),
        tuple(selected[key] for key in DEVELOPMENT_KEYS),
    )


def _task_fields(task: GeneratedTask) -> tuple[bytes, bytes, bytes]:
    expected = EXPECTED_SOURCE_RECORDS[(task.public.family, task.public.ordinal)]
    return (
        task.public.signature.entry_point.encode("ascii"),
        task.public.signature.parameters[0].name.encode("ascii"),
        str(expected["literal"]).encode("ascii"),
    )


def _canonical_literal_primitives(literal: bytes) -> tuple[str, ...]:
    text = literal.decode("ascii")
    signed = text if text.startswith("-") else f"+{text}"
    if signed in {"+0", "-0"} or not signed[1:].isdigit():
        raise RuntimeError(f"non-canonical visible literal {text!r}")
    return tuple(signed)


def _prompt_spans(task: GeneratedTask) -> tuple[
    bytes, tuple[tuple[int, int, int, tuple[int, ...]], ...]
]:
    prompt = task.public.prompt.encode("ascii")
    entry_point, parameter, literal = _task_fields(task)
    signature = b"def " + entry_point + b"(" + parameter + b"):"
    signature_start = prompt.index(signature)
    entry_start = signature_start + len(b"def ")
    parameter_start = entry_start + len(entry_point) + 1
    if task.public.family == SCALAR_FAMILY:
        literal_anchor = b"fixed offset "
    elif task.public.family == LIST_FAMILY:
        literal_anchor = b"with "
    else:
        raise RuntimeError("unsupported typed-sketch family")
    literal_start = prompt.index(literal_anchor, parameter_start) + len(literal_anchor)
    if prompt[literal_start : literal_start + len(literal)] != literal:
        raise RuntimeError("visible literal span no longer matches")
    canonical_literal_ids = tuple(
        PRIMITIVE_TOKEN_IDS[token]
        for token in _canonical_literal_primitives(literal)
    )
    spans = (
        (entry_start, entry_start + len(entry_point), 0, tuple(entry_point)),
        (parameter_start, parameter_start + len(parameter), 1, tuple(parameter)),
        (literal_start, literal_start + len(literal), 2, canonical_literal_ids),
    )
    return prompt, spans


def _tokenized_prompt(task: GeneratedTask) -> tuple[int, ...]:
    prompt, spans = _prompt_spans(task)
    tokens: list[int] = []
    cursor = 0
    for start, end, slot, replacement in spans:
        if start < cursor or end <= start:
            raise RuntimeError("task prompt spans overlap")
        tokens.extend(prompt[cursor:start])
        tokens.append(slot_start_token_id(slot))
        tokens.extend(replacement)
        tokens.append(slot_end_token_id(slot))
        cursor = end
    tokens.extend(prompt[cursor:])
    result = tuple(tokens)
    if _detokenize_prompt(result) != prompt:
        raise RuntimeError("typed prompt tokenizer is not a lossless public-wording view")
    return result


def _detokenize_prompt(tokens: tuple[int, ...]) -> bytes:
    starts = {slot_start_token_id(slot): slot for slot in range(SLOT_COUNT)}
    ends = {slot_end_token_id(slot): slot for slot in range(SLOT_COUNT)}
    output = bytearray()
    active_slot: int | None = None
    literal_parts: list[str] = []
    for token_id in tokens:
        if token_id in starts:
            if active_slot is not None:
                raise RuntimeError("nested prompt slots are forbidden")
            active_slot = starts[token_id]
            literal_parts = []
            continue
        if token_id in ends:
            if active_slot != ends[token_id]:
                raise RuntimeError("mismatched prompt slot boundary")
            if active_slot == 2:
                signed = "".join(literal_parts)
                raw = signed[1:] if signed.startswith("+") else signed
                output.extend(raw.encode("ascii"))
            active_slot = None
            continue
        if active_slot == 2:
            try:
                literal_parts.append(PRIMITIVE_BY_TOKEN_ID[token_id])
            except KeyError as error:
                raise RuntimeError("non-primitive token in literal slot") from error
        else:
            if not 0 <= token_id < 128:
                raise RuntimeError("non-ASCII token outside canonical literal slot")
            output.append(token_id)
    if active_slot is not None:
        raise RuntimeError("unterminated prompt slot")
    return bytes(output)


def _primitive_id(token: str) -> int:
    return PRIMITIVE_TOKEN_IDS[token]


def _internal_target(family: str) -> tuple[int, ...]:
    if family == SCALAR_FAMILY:
        primitives: tuple[str | None, ...] = (
            "add",
            "variable-0",
            "integer",
            None,
            "end",
        )
    elif family == LIST_FAMILY:
        primitives = (
            "map",
            "bind",
            "add",
            "variable-0",
            "integer",
            None,
            "end",
            "variable-0",
        )
    else:
        raise RuntimeError("unsupported typed-sketch family")
    return (
        *(
            copy_slot_token_id(2) if primitive is None else _primitive_id(primitive)
            for primitive in primitives
        ),
        EOS_TOKEN_ID,
    )


def _expected_program_tokens(task: GeneratedTask) -> tuple[str, ...]:
    literal = _task_fields(task)[2]
    signed = _canonical_literal_primitives(literal)
    if task.public.family == SCALAR_FAMILY:
        return ("add", "variable-0", "integer", *signed, "end")
    if task.public.family == LIST_FAMILY:
        return (
            "map",
            "bind",
            "add",
            "variable-0",
            "integer",
            *signed,
            "end",
            "variable-0",
        )
    raise RuntimeError("unsupported typed-sketch family")


def _train_episode(task: GeneratedTask) -> tuple[int, ...]:
    return (*_tokenized_prompt(task), *_internal_target(task.public.family))


def _definition(
    train_tasks: tuple[GeneratedTask, ...],
    development_tasks: tuple[GeneratedTask, ...],
) -> NativeTemporalCircuitDefinition:
    train_episodes = tuple(_train_episode(task) for task in train_tasks)
    development_prompts = tuple(_tokenized_prompt(task) for task in development_tasks)
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
                *(token for prompt in development_prompts for token in prompt),
                *PRIMITIVE_BY_TOKEN_ID,
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
        "surface_primitive": (
            "<EOS>" if token_id == EOS_TOKEN_ID else PRIMITIVE_BY_TOKEN_ID.get(token_id)
        ),
        "temporal_event_count": int(generation.temporal_event_count),
        "control_token_id": generation.control_token_id,
        "physical_election_count": int(generation.physical_election_count),
        "candidate_tokens_physical": int(generation.candidate_tokens),
        "winner_input_columns": int(generation.winner_input_columns),
    }


def _generate_once(circuit: NativeTemporalCircuit) -> dict[str, object]:
    emitted: list[str] = []
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
        record = _generation_record(position, generation)
        generations.append(record)
        token_id = int(generation.surface_token_id)
        if token_id == EOS_TOKEN_ID:
            terminated_by_eos = True
            break
        primitive = PRIMITIVE_BY_TOKEN_ID.get(token_id)
        if primitive is None:
            stop = {
                "kind": "UNKNOWN_TYPED_SURFACE_TOKEN",
                "generation_position": position,
                "observed": token_id,
            }
            break
        emitted.append(primitive)
    if not terminated_by_eos and stop is None:
        stop = {
            "kind": "GENERATION_BUDGET_EXHAUSTED",
            "generation_budget": MAX_GENERATIONS,
        }
    return {
        "emitted_tokens": tuple(emitted),
        "generations": generations,
        "terminated_by_eos": terminated_by_eos,
        "stop": stop,
    }


def _copy_transaction(
    generations: list[dict[str, object]], expected_tokens: tuple[str, ...]
) -> tuple[dict[str, object] | None, bool]:
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
                "copied_primitives": [generation["surface_primitive"]],
                "copied_event_counts": [event_count],
                "copied_physical_election_counts": [
                    generation["physical_election_count"]
                ],
                "copied_candidate_counts": [generation["candidate_tokens_physical"]],
                "exit_event_count": None,
                "exit_surface_primitive": None,
            }
        elif path == TemporalGenerationPath.COPY.name:
            exact &= active is not None
            if active is not None:
                exact &= generation["slot_ordinal"] == active["slot_ordinal"]
                active["copied_places"].append(generation["copy_place_ordinal"])
                active["copied_primitives"].append(generation["surface_primitive"])
                active["copied_event_counts"].append(event_count)
                active["copied_physical_election_counts"].append(
                    generation["physical_election_count"]
                )
                active["copied_candidate_counts"].append(
                    generation["candidate_tokens_physical"]
                )
        elif path == TemporalGenerationPath.EXIT_COPY.name:
            exact &= active is not None
            if active is not None:
                exact &= generation["slot_ordinal"] == active["slot_ordinal"]
                active["exit_event_count"] = event_count
                active["exit_surface_primitive"] = generation["surface_primitive"]
                enter_count = int(active["enter_event_count"])
                active["temporal_event_count_held_across_copied_tokens"] = all(
                    int(value) == enter_count for value in active["copied_event_counts"]
                )
                active["exit_resumed_temporal_once"] = event_count == enter_count + 1
                transactions.append(active)
                active = None
        previous_event_count = event_count
    exact &= active is None and len(transactions) == 1
    transaction = transactions[0] if len(transactions) == 1 else None
    if transaction is not None:
        exact &= transaction["slot_ordinal"] == 2
        exact &= transaction["control_token_id"] == copy_slot_token_id(2)
        exact &= transaction["copied_places"] == list(range(len(expected_tokens)))
        exact &= transaction["copied_primitives"] == list(expected_tokens)
        exact &= transaction["exit_surface_primitive"] == "end"
        exact &= bool(transaction["control_advanced_temporal_once"])
        exact &= bool(transaction["temporal_event_count_held_across_copied_tokens"])
        exact &= bool(transaction["exit_resumed_temporal_once"])
        exact &= all(
            int(value) >= 1
            for value in transaction["copied_physical_election_counts"]
        )
        exact &= all(int(value) >= 1 for value in transaction["copied_candidate_counts"])
    return transaction, exact


def _expected_compiled_source(task: GeneratedTask) -> str:
    entry_point, parameter, literal = (
        value.decode("ascii") for value in _task_fields(task)
    )
    if task.public.family == SCALAR_FAMILY:
        return f"def {entry_point}({parameter}):\n    return {parameter} + {literal}\n"
    if task.public.family == LIST_FAMILY:
        return (
            f"def {entry_point}({parameter}):\n"
            f"    return [item_0 + {literal} for item_0 in {parameter}]\n"
        )
    raise RuntimeError("unsupported typed-sketch family")


def _task_record(task: GeneratedTask) -> dict[str, object]:
    entry_point, parameter, literal = _task_fields(task)
    prompt_tokens = _tokenized_prompt(task)
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
        "public_prompt_sha256": hashlib.sha256(
            task.public.prompt.encode("utf-8")
        ).hexdigest(),
        "public_prompt_lossless_after_tokenization": (
            _detokenize_prompt(prompt_tokens) == task.public.prompt.encode("ascii")
        ),
        "regression_case_count": len(task.private.regression_cases),
        "property_case_count": len(task.private.property_cases),
    }


def _development_once(
    *,
    circuit: NativeTemporalCircuit,
    task: GeneratedTask,
    protocol: object,
    python_executable: Path,
    runtime_identity: dict[str, str],
    worker_source: bytes,
    worker_sha256: str,
) -> tuple[dict[str, object], dict[str, object] | None]:
    prompt_tokens = _tokenized_prompt(task)
    prompt_steps = circuit.advance_block(
        prompt_tokens,
        mode=TemporalStepMode.READ_ONLY,
        reset_before=True,
    )
    run = _generate_once(circuit)
    emitted_tokens = run.pop("emitted_tokens")
    expected_tokens = _expected_program_tokens(task)
    literal_tokens = _canonical_literal_primitives(_task_fields(task)[2])
    transaction, copy_trace_exact = _copy_transaction(
        run["generations"], literal_tokens
    )
    fitted_cell_races_exact = all(
        int(generation["physical_election_count"]) >= 1
        and int(generation["candidate_tokens_physical"]) >= 1
        and int(generation["winner_input_columns"]) == ACTIVE_COLUMNS
        for generation in run["generations"]
        if generation["surface_token_id"] != EOS_TOKEN_ID
    )

    first_failure: dict[str, object] | None = None
    if run["stop"] is not None:
        first_failure = {"kind": "model", "gate": "GENERATION_STOP", "detail": run["stop"]}
    elif not run["terminated_by_eos"]:
        first_failure = {"kind": "model", "gate": "ORDINARY_EOS", "detail": None}
    elif emitted_tokens != expected_tokens:
        first_failure = {
            "kind": "model",
            "gate": "EXACT_TYPED_TOKEN_STREAM",
            "detail": {
                "expected": list(expected_tokens),
                "observed": list(emitted_tokens),
            },
        }
    elif not copy_trace_exact:
        first_failure = {"kind": "model", "gate": "SLOT_2_PHYSICAL_COPY", "detail": transaction}
    elif not fitted_cell_races_exact:
        first_failure = {"kind": "model", "gate": "FITTED_CELL_RACES", "detail": None}

    program_record: dict[str, object] = {
        "attempted": False,
        "tokens": list(emitted_tokens),
        "expected_tokens": list(expected_tokens),
        "exact_tokens": emitted_tokens == expected_tokens,
        "program_sha256": None,
        "compiled_source": None,
        "expected_compiled_source": _expected_compiled_source(task),
        "exact_compiled_source": False,
        "compiler_received_prompt": False,
        "compiler_received_tests": False,
    }
    compiled_source: str | None = None
    if first_failure is None:
        try:
            program = Program(tokens=emitted_tokens, signature=task.public.signature)
            compiled_source = compile_typed_program(program).source
        except (TypeError, ValueError) as error:
            first_failure = {
                "kind": "model",
                "gate": "EXISTING_TYPED_PROGRAM_COMPILE",
                "detail": f"{type(error).__name__}: {error}",
            }
        else:
            program_record.update(
                {
                    "attempted": True,
                    "program_sha256": program.sha256,
                    "compiled_source": compiled_source,
                    "exact_compiled_source": (
                        compiled_source == program_record["expected_compiled_source"]
                    ),
                }
            )
            if not program_record["exact_compiled_source"]:
                first_failure = {
                    "kind": "model",
                    "gate": "EXACT_SIGNATURE_PYTHON",
                    "detail": None,
                }

    sandbox_record: dict[str, object] = {
        "attempted": False,
        "outcome": None,
        "passed": False,
        "sandboxed": False,
        "detail": "not run because an earlier gate stopped",
        "wall_seconds": 0.0,
    }
    if first_failure is None and compiled_source is not None:
        try:
            sandbox_result = execute_completion(
                compiled_source,
                task.public,
                task.private,
                protocol.execution,
                python_executable=python_executable,
                python_executable_sha256=runtime_identity["executable_sha256"],
                worker_source=worker_source,
                worker_sha256=worker_sha256,
            )
        except SandboxConfigurationError as error:
            first_failure = {
                "kind": "harness",
                "gate": "PUBLIC_SIMPLE_SWE_SANDBOX_CONFIGURATION",
                "detail": str(error),
            }
            sandbox_record["detail"] = str(error)
        else:
            sandbox_record = {
                "attempted": True,
                **asdict(sandbox_result),
                "regression_case_count": len(task.private.regression_cases),
                "property_case_count": len(task.private.property_cases),
                "worker_sha256": worker_sha256,
                "python_runtime": runtime_identity,
            }
            if not sandbox_result.passed:
                first_failure = {
                    "kind": "model",
                    "gate": "PUBLIC_SIMPLE_SWE_SANDBOX",
                    "detail": sandbox_result.outcome,
                }

    record = {
        "task": _task_record(task),
        "attempt_number": 1,
        "prompt_steps": len(prompt_steps),
        "teacher_tokens_after_prompt": 0,
        "target_entered_runtime": False,
        "generation_budget": MAX_GENERATIONS,
        "completed_generations": len(run["generations"]),
        "terminated_by_ordinary_eos": run["terminated_by_eos"],
        "stop": run["stop"],
        "program": program_record,
        "slot_2_copy_transaction": transaction,
        "slot_2_copy_trace_exact": copy_trace_exact,
        "fitted_cell_races_exact": fitted_cell_races_exact,
        "sandbox": sandbox_record,
        "first_failure": first_failure,
        "generations": run["generations"],
    }
    return record, first_failure


def main() -> int:
    args = _parse_args()
    started_ns = time.perf_counter_ns()
    protocol, train_tasks, development_tasks = _select_and_confirm_tasks()
    definition = _definition(train_tasks, development_tasks)

    executable = resolve_python_executable(args.python_executable)
    runtime_identity = python_runtime_identity(executable, protocol.raw.get("language"))
    if runtime_identity["python_version"] != "3.11.15":
        raise RuntimeError(
            "native typed-sketch probe requires the explicit repo Python 3.11.15"
        )
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = hashlib.sha256(worker_path.read_bytes()).hexdigest()
    worker_source = load_worker_snapshot(worker_sha256)

    train_records: list[dict[str, object]] = []
    development_records: list[dict[str, object]] = []
    first_failure: dict[str, object] | None = None
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
                        "internal_target": [
                            "COPY_SLOT_2" if token == copy_slot_token_id(2)
                            else "<EOS>" if token == EOS_TOKEN_ID
                            else PRIMITIVE_BY_TOKEN_ID[token]
                            for token in _internal_target(task.public.family)
                        ],
                    }
                )
            trained_identity = circuit.checkpoint_identity()
            for task in development_tasks:
                record, failure = _development_once(
                    circuit=circuit,
                    task=task,
                    protocol=protocol,
                    python_executable=executable,
                    runtime_identity=runtime_identity,
                    worker_source=worker_source,
                    worker_sha256=worker_sha256,
                )
                development_records.append(record)
                if failure is not None:
                    first_failure = {
                        **failure,
                        "task_id": task.public.task_id,
                    }
                    break
        native_identity = loaded_library_identity()

    passed = first_failure is None and len(development_records) == len(DEVELOPMENT_KEYS)
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-native-typed-sketch-two-route",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": (
            "PASS_NATIVE_TYPED_SKETCH_TWO_ROUTE"
            if passed
            else "STOP_NATIVE_TYPED_SKETCH_TWO_ROUTE"
        ),
        "capability_pass": passed,
        "first_failure": first_failure,
        "claim_boundary": (
            "One public visible train task from scalar-offset ordinal 000 and one from "
            "list-map-add ordinal 000 teach two existing Layer-1 typed sketches in one "
            "retained native temporal circuit. Scalar ordinal 003 (+7) is attempted once, "
            "then list-map ordinal 007 (+3) once only if scalar passes. Actual public wording "
            "is losslessly tokenized; no family ID or solution label enters either prompt. "
            "Ordinal slot 2 physically copies canonical signed primitive tokens, all semantic "
            "tokens come from fitted-cell races, and the existing typed_program compiler sees "
            "only the generated Program plus the real TypedSignature before the public 4+32 "
            "sandbox. This is visible two-family development evidence, not family-disjoint "
            "generalization, M0, eval, broad coding competence, protected-data evidence, or C1."
        ),
        "authority": {
            "source_partition_generated": "train",
            "eval_partition_generated": False,
            "train_keys": [list(key) for key in TRAIN_KEYS],
            "development_keys": [list(key) for key in DEVELOPMENT_KEYS],
            "development_attempt_order": [
                "t0_scalar_offset/003",
                "t1_list_map_add/007",
            ],
            "development_attempts_completed": len(development_records),
            "model_retries": 0,
            "harness_retries": 0,
            "development_targets_used_as_runtime_input": False,
            "teacher_tokens_after_development_prompt": 0,
            "prompt_family_ids_injected": False,
            "prompt_solution_labels_injected": False,
            "python_candidate_enumeration": False,
            "python_argmax": False,
            "learned_readout_parameters": 0,
            "typed_compiler_received_prompt": False,
            "typed_compiler_received_tests": False,
            "synthetic_runtime_authority": True,
        },
        "protocol": {
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.sha256,
            "language": protocol.raw.get("language"),
        },
        "typed_primitive_adapter": {
            "native_token_ids_arbitrary": True,
            "primitive_by_native_token_id": {
                str(token_id): primitive
                for token_id, primitive in sorted(PRIMITIVE_BY_TOKEN_ID.items())
            },
            "scalar_sketch": "add variable-0 integer COPY[2] end",
            "list_sketch": (
                "map bind add variable-0 integer COPY[2] end variable-0"
            ),
            "compiler_module": "humanbrain.typed_program",
            "new_dsl_or_compiler": False,
        },
        "slot_bank": {
            "slot_count": SLOT_COUNT,
            "copied_slot_ordinal": 2,
            "max_surface_tokens_per_slot": 22,
            "prompt_roles": {
                "0": "public signature entry point",
                "1": "public signature unary parameter",
                "2": "canonical signed literal primitives",
            },
            "copy_control_token_id": copy_slot_token_id(2),
            "copy_end_token_id": COPY_END_TOKEN_ID,
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
            "exact_source_reservation_runtime_base": "d455613",
        },
        "train": {
            "tasks": train_records,
            "reset_before_each": True,
            "learned_state_retained_between_tasks": True,
            "live_segments_after_train": trained_identity.segment_count,
            "learning_steps_after_train": trained_identity.learning_step,
        },
        "visible_development_teacher_free": development_records,
        "public_simple_swe_runtime": {
            "explicit_repo_venv": str(executable),
            "python_runtime": runtime_identity,
            "worker_sha256": worker_sha256,
            "regression_cases_per_task": 4,
            "property_cases_per_task": 32,
        },
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
                "development_attempts_completed": len(development_records),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
