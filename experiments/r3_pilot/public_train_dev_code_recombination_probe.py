"""One frozen PUBLIC synthetic TRAIN/DEV code recombination probe.

TRAIN contains ``inc/+1`` and ``dec/-1`` prompt-to-function episodes. DEV asks
for the unseen identifier ``bump`` recombined with the seen ``+1`` body. The
DEV target is scoring-only: inference presents the DEV prompt once, then every
output token must come from native physical election and emitted-token feedback.
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
    NativeTemporalRuntimeError,
    TemporalContextPolicy,
    TemporalPlasticityPolicy,
    TemporalStepMode,
)


OUTPUT_PATH = Path(__file__).with_name("public_train_dev_code_recombination_probe.json")
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
    "expected_code": "def bump(x):\n    return x + 1\n",
}
EOS_TOKEN = 256
ASCII_TOKEN_LIMIT = 128
ACTIVE_COLUMNS = 11
BASE_CELLS_PER_COLUMN = 8
TRAIN_REPLAYS = 2


@contextlib.contextmanager
def _synthetic_public_probe_authority() -> Iterator[None]:
    """Bypass only the intentionally non-runnable source-binding hold."""

    original = runtime_adapter._fresh_authority
    authority = runtime_adapter._FreshAuthority(b"\x83" * 32, b"\xd5" * 32)
    runtime_adapter._fresh_authority = lambda: authority
    try:
        yield
    finally:
        runtime_adapter._fresh_authority = original


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def _episode(example: dict[str, str], code_key: str) -> tuple[int, ...]:
    return (
        *example["prompt"].encode("ascii"),
        *example[code_key].encode("ascii"),
        EOS_TOKEN,
    )


def _definition() -> NativeTemporalCircuitDefinition:
    train_episodes = tuple(_episode(example, "code") for example in TRAIN_EXAMPLES)
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


def _stop_record(
    error: NativeTemporalRuntimeError,
    *,
    phase: str,
    emission_position: int,
) -> dict[str, object]:
    return {
        "phase": phase,
        "emission_position": emission_position,
        "status": error.status.name,
        "message": error.message,
        "completed_prefix": error.completed_prefix,
        "required": error.required,
        "observed": error.observed,
        "location": error.location,
        "poison_code": error.poison_code,
        "stage_code": error.stage_code,
    }


def _analyze_code(source: str | None) -> dict[str, object]:
    analysis: dict[str, object] = {
        "ast_parse_valid": False,
        "compile_valid": False,
        "function_name": None,
        "operator": None,
        "constant": None,
    }
    if source is None:
        return analysis
    try:
        module = ast.parse(source, filename="<physical-dev-emission>", mode="exec")
        analysis["ast_parse_valid"] = True
        compile(module, "<physical-dev-emission>", "exec")
        analysis["compile_valid"] = True
    except (SyntaxError, ValueError, TypeError):
        return analysis
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        return analysis
    function = module.body[0]
    analysis["function_name"] = function.name
    if len(function.body) != 1 or not isinstance(function.body[0], ast.Return):
        return analysis
    value = function.body[0].value
    if not isinstance(value, ast.BinOp):
        return analysis
    analysis["operator"] = type(value.op).__name__
    if isinstance(value.right, ast.Constant):
        analysis["constant"] = value.right.value
    return analysis


def main() -> int:
    args = _parse_args()
    started_ns = time.perf_counter_ns()
    definition = _definition()
    train_episodes = tuple(_episode(example, "code") for example in TRAIN_EXAMPLES)
    expected_code = DEV_EXAMPLE["expected_code"].encode("ascii")
    dev_prompt = tuple(DEV_EXAMPLE["prompt"].encode("ascii"))

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
                            "punished_segments": sum(step.punished_segments for step in steps),
                        }
                    )
                training.append({"replay": replay, "episodes": episode_records})
            trained_identity = circuit.checkpoint_identity()

            dev_attempts = 1
            prompt_steps = circuit.advance_block(
                dev_prompt,
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            emitted_bytes = bytearray()
            free_run = []
            stop: dict[str, object] | None = None
            terminated_by_eos = False
            maximum_elections = len(expected_code) + 8
            for position in range(maximum_elections):
                try:
                    election = circuit.elect_output()
                except NativeTemporalRuntimeError as error:
                    stop = _stop_record(
                        error,
                        phase="elect_output",
                        emission_position=position,
                    )
                    break
                is_eos = election.token_id == EOS_TOKEN
                if not is_eos:
                    if not 0 <= election.token_id < ASCII_TOKEN_LIMIT:
                        stop = {
                            "phase": "decode",
                            "emission_position": position,
                            "status": "NON_ASCII_TOKEN",
                            "observed": election.token_id,
                        }
                        break
                    emitted_bytes.append(election.token_id)
                try:
                    feedback = circuit.advance_block(
                        [election.token_id],
                        mode=TemporalStepMode.READ_ONLY,
                    )[0]
                except NativeTemporalRuntimeError as error:
                    stop = _stop_record(
                        error,
                        phase="emitted_token_feedback",
                        emission_position=position,
                    )
                    break
                free_run.append(
                    {
                        "position": position,
                        "emitted_token_id": election.token_id,
                        "emitted_byte_hex": (
                            None if is_eos else f"{election.token_id:02x}"
                        ),
                        "eos": is_eos,
                        "activated_columns": election.activated_columns,
                        "candidate_tokens_physical": election.candidate_tokens,
                        "winner_input_columns": election.winner_input_columns,
                        "feedback_predicted": feedback.predicted,
                        "feedback_predicted_columns": feedback.predicted_columns,
                    }
                )
                if is_eos:
                    terminated_by_eos = True
                    break
        native_identity = loaded_library_identity()

    emitted = bytes(emitted_bytes)
    emitted_text: str | None = None
    utf8_valid = False
    try:
        emitted_text = emitted.decode("ascii", errors="strict")
        utf8_valid = True
    except UnicodeDecodeError:
        pass
    code_analysis = _analyze_code(emitted_text if emitted else None)
    exact_recombination = terminated_by_eos and emitted == expected_code
    train_inc = TRAIN_EXAMPLES[0]["code"].encode("ascii")
    train_dec = TRAIN_EXAMPLES[1]["code"].encode("ascii")

    expected_train_transitions = sum(len(episode) - 1 for episode in train_episodes)
    replay_two = training[1]["episodes"]
    replay_two_allocations = sum(
        int(record["allocated_segments"]) for record in replay_two
    )
    replay_two_predictions = sum(int(record["predicted_steps"]) for record in replay_two)
    training_stable = (
        replay_two_allocations == 0
        and replay_two_predictions == expected_train_transitions
    )
    if not training_stable:
        outcome = "INVALID_UNSTABLE_TRAIN_REPLAY"
        localization = "TRAIN replay was not fully recalled, so DEV composition is confounded."
    elif exact_recombination and bool(code_analysis["compile_valid"]):
        outcome = "PASS_NOVEL_NAME_PLUS_OPERATOR_RECOMBINATION"
        localization = "The physical circuit recombined the unseen identifier with the seen +1 body."
    elif stop is not None and not free_run and stop.get("phase") == "elect_output":
        outcome = "STOP_NOVEL_DEV_PROMPT_OUTPUT_ROUTING"
        localization = (
            "The novel DEV prompt produced no physically electable output token; prompt-to-output "
            "routing fails before identifier copy or operator routing can be tested."
        )
    elif terminated_by_eos and emitted == train_inc:
        outcome = "FAIL_RETRIEVED_TRAIN_INC_TEMPLATE"
        localization = (
            "The seen add-one suffix routed to the memorized inc template; unseen identifier "
            "binding/copy is the smallest missing composition primitive."
        )
    elif terminated_by_eos and emitted == train_dec:
        outcome = "FAIL_RETRIEVED_TRAIN_DEC_TEMPLATE"
        localization = (
            "The DEV prompt routed to the memorized dec template; both operator selection and "
            "unseen identifier binding remain unresolved."
        )
    elif bool(code_analysis["compile_valid"]):
        outcome = "FAIL_WRONG_BUT_VALID_DEV_FUNCTION"
        localization = (
            "Physical emission formed valid Python but missed the exact name/operator binding; "
            "the decoded AST fields localize which binding failed."
        )
    elif stop is not None:
        outcome = "STOP_DURING_DEV_FREE_RUN"
        localization = (
            "Physical sequence routing began but stopped before EOS; the recorded phase and "
            "emission position localize the first missing transition."
        )
    else:
        outcome = "FAIL_INVALID_OR_UNTERMINATED_DEV_EMISSION"
        localization = (
            "Physical emission did not form a complete Python function; sequence routing or EOS "
            "termination failed before composition could be credited."
        )

    payload = {
        "schema_version": 1,
        "experiment": "r3-public-train-dev-code-recombination-probe",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": outcome,
        "capability_pass": exact_recombination and bool(code_analysis["compile_valid"]),
        "probe_completed": True,
        "localization": localization,
        "claim_boundary": (
            "Two literal public synthetic TRAIN prompt-to-function pairs and one frozen public "
            "synthetic DEV recombination are used. DEV is attempted exactly once. Only the DEV "
            "prompt reaches the runtime; the scoring target never does. Any DEV output must come "
            "from native physical election plus emitted-token READ_ONLY feedback, with no Python "
            "candidate enumeration, argmax, learned head, retry, or adaptation. This diagnoses "
            "prompt routing, identifier copy/binding, operator routing, and sequence termination; "
            "it is not coding competence, generalization beyond this micro-probe, SWE, protected "
            "data, C1, or benchmark evidence."
        ),
        "frozen_examples": {
            "train": list(TRAIN_EXAMPLES),
            "dev": DEV_EXAMPLE,
            "dev_target_used_as_runtime_input": False,
        },
        "authority": {
            "partitions": ["public-synthetic-train", "public-synthetic-dev"],
            "dev_attempts": dev_attempts,
            "protected_material_read": False,
            "held_out_material_read": False,
            "c1_material_read": False,
            "synthetic_runtime_authority": True,
        },
        "tokenization": {
            "scheme": "literal ASCII byte IDs 0..127 plus EOS=256",
            "eos_token_id": EOS_TOKEN,
            "static_codebook": "disjoint 11-column code per ASCII byte/EOS",
            "python_candidate_enumeration": False,
            "python_argmax": False,
            "learned_readout_parameters": 0,
        },
        "physical_definition": {
            "circuit_identity_sha256": definition.circuit_identity_sha256(),
            "column_count": definition.config.column_count,
            "max_pyramidal_cells": definition.config.max_pyramidal_cells,
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
            "expected_transitions_per_replay": expected_train_transitions,
            "stable_second_replay": training_stable,
            "records": training,
            "live_segments": trained_identity.segment_count,
            "learning_steps": trained_identity.learning_step,
        },
        "dev_once": {
            "prompt_token_count": len(prompt_steps),
            "teacher_tokens_after_prompt": 0,
            "maximum_election_budget": maximum_elections,
            "completed_elections": len(free_run),
            "terminated_by_eos": terminated_by_eos,
            "stop": stop,
            "emitted_code_text": emitted_text,
            "emitted_code_hex": emitted.hex(),
            "emitted_code_sha256": hashlib.sha256(emitted).hexdigest(),
            "expected_code_sha256": hashlib.sha256(expected_code).hexdigest(),
            "exact_expected_bytes": emitted == expected_code,
            "ascii_valid": utf8_valid,
            "code_analysis": code_analysis,
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
                "outcome": outcome,
                "capability_pass": payload["capability_pass"],
                "completed_elections": len(free_run),
                "stop": stop,
                "emitted_code_text": emitted_text,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
