"""Fresh source-local alpha-binding gate over the public coding prior.

The gate asks whether one cell model can learn generic tool syntax while
selecting source-local lexical positions from an inventory mechanically derived
from the exact starter source.  It uses only the eight frozen public-prior cases
for growth and mechanically alpha-renamed public cases for inference.  Public
expected sources are scoring-only custody.  No source is executed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated.public_coding_prior_v1 import (
    PUBLIC_CODING_PRIOR_V1_SHA256,
    SOURCE_CORPUS_SHA256,
    SOURCE_FREEZE_COMMIT,
    SOURCE_MANIFEST_SHA256,
    public_coding_prior_v1,
)
from experiments.r4_integrated.source_local_alpha_binding_tasks_v1 import (
    ALPHA_BINDING_STREAM_SHA256,
    CASE_COUNT,
    AlphaBindingCaseV1,
    source_local_alpha_binding_cases_v1,
)
from humanbrain import CellLanguageModel, ModelConfig
from humanbrain import persistent_coding as coding
from humanbrain.pointer_binding import (
    PointerBoundCellModel,
    PointerTrainingExample,
    SymbolInventory,
    identifier_inventory,
    materialize,
    pointerize,
)

SCHEMA = "source-local-alpha-binding-gate-v1"
RESERVATION_SCHEMA = "source-local-alpha-binding-gate-reservation-v1"
WRITE_TOOL = "write_source"
MAX_NEW_TOKENS = 512
TRAINING_REPETITIONS = 2

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "benchmarks/r4_source_local_alpha_binding_gate_v1.json"
POINTER_SOURCE_PATH = REPO_ROOT / "src/humanbrain/pointer_binding.py"
TASK_SOURCE_PATH = Path(__file__).with_name("source_local_alpha_binding_tasks_v1.py")
PRIOR_SOURCE_PATH = Path(__file__).with_name("public_coding_prior_v1.py")
PERSISTENT_CODING_SOURCE_PATH = REPO_ROOT / "src/humanbrain/persistent_coding.py"

MODEL_CONFIG = ModelConfig(
    max_order=256,
    max_phrases=256,
    min_phrase_frequency=2,
)

_POINTER = re.compile(r"<@[0-9]+>")
_WRITE_AFFORDANCE = coding.ToolAffordance.seal(
    WRITE_TOOL,
    {
        "additionalProperties": False,
        "properties": {"source": {"type": "string"}},
        "required": ["source"],
        "type": "object",
    },
)


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _frame(starter_source: str, public_prompt: str) -> coding.ObservationFrame:
    """Seal the exact target-free persistent-coding observation contract."""

    return coding.ObservationFrame.seal(
        (
            coding.ObservationBlock.seal("current-source", starter_source),
            coding.ObservationBlock.seal("public-task", public_prompt),
        ),
        (_WRITE_AFFORDANCE,),
    )


def _tool_output(source: str) -> str:
    payload = {"arguments": {"source": source}, "name": WRITE_TOOL}
    return f"<tool>{coding._json(payload)}</tool>"


def _training_corpora() -> tuple[
    tuple[PointerTrainingExample, ...],
    tuple[str, ...],
]:
    prior = public_coding_prior_v1()
    if len(prior) != CASE_COUNT:
        raise ValueError("pointer gate requires every frozen public-prior case")
    pointer_examples: list[PointerTrainingExample] = []
    raw_trajectories: list[str] = []
    for example in prior:
        frame = _frame(example.starter_source, example.public_prompt)
        prompt = coding._prompt(frame)
        output = _tool_output(example.successful_source)
        inventory = identifier_inventory(example.starter_source)
        for _ in range(TRAINING_REPETITIONS):
            pointer_examples.append(PointerTrainingExample(prompt, output, inventory))
            raw_trajectories.append(f"{prompt}\n{output}")
    expected_count = CASE_COUNT * TRAINING_REPETITIONS
    if len(pointer_examples) != expected_count or len(raw_trajectories) != expected_count:
        raise RuntimeError("pointer and raw growth corpora are not matched")
    return tuple(pointer_examples), tuple(raw_trajectories)


def _grow_models() -> tuple[PointerBoundCellModel, CellLanguageModel]:
    pointer_examples, raw_trajectories = _training_corpora()
    return (
        PointerBoundCellModel.grow(pointer_examples, MODEL_CONFIG),
        CellLanguageModel.grow(raw_trajectories, MODEL_CONFIG),
    )


def _source_from_decision(decision: coding.Decision) -> str | None:
    if not isinstance(decision, coding.ToolAction) or decision.tool_name != WRITE_TOOL:
        return None
    try:
        arguments = json.loads(decision.arguments_json)
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(arguments, dict)
        or set(arguments) != {"source"}
        or not isinstance(arguments["source"], str)
    ):
        return None
    return arguments["source"]


def _ast_signature(source: str | None) -> dict[str, object] | None:
    if source is None:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(tree.body) != 1 or len(functions) != 1:
        return None
    function = functions[0]
    arguments = function.args
    return {
        "entrypoint": function.name,
        "positional_only": tuple(arg.arg for arg in arguments.posonlyargs),
        "parameters": tuple(arg.arg for arg in arguments.args),
        "keyword_only": tuple(arg.arg for arg in arguments.kwonlyargs),
        "vararg": None if arguments.vararg is None else arguments.vararg.arg,
        "kwarg": None if arguments.kwarg is None else arguments.kwarg.arg,
    }


def _decision_record(decision: coding.Decision) -> dict[str, object]:
    if isinstance(decision, coding.ToolAction):
        return {
            "kind": "tool",
            "sha256": decision.sha256,
            "frame_sha256": decision.frame_sha256,
            "model_digest": decision.model_digest,
            "tool_name": decision.tool_name,
            "arguments_json": decision.arguments_json,
        }
    return {
        "kind": "stop",
        "sha256": decision.sha256,
        "frame_sha256": decision.frame_sha256,
        "model_digest": decision.model_digest,
        "reason": decision.reason,
    }


def _resolve_pointer_output(
    *,
    frame: coding.ObservationFrame,
    model_digest: str,
    raw_output: str,
    inventory: SymbolInventory,
) -> tuple[str | None, coding.Decision, dict[str, str] | None]:
    try:
        resolved = materialize(raw_output, inventory)
    except (TypeError, ValueError) as error:
        stopped = coding._decode(
            frame,
            model_digest,
            "<stop>pointer_binding_failed_closed</stop>",
        )
        return None, stopped, {"type": type(error).__name__, "message": str(error)}
    return resolved, coding._decode(frame, model_digest, resolved), None


def _run_case(
    case: AlphaBindingCaseV1,
    pointer_model: PointerBoundCellModel,
    raw_model: CellLanguageModel,
) -> dict[str, object]:
    case.authenticate()
    frame = _frame(case.starter_source, case.prompt)
    prompt = coding._prompt(frame)
    inventory = identifier_inventory(case.starter_source)
    expected_signature = _ast_signature(case.expected_source)
    if expected_signature is None:
        raise ValueError("scoring-only expected source has no canonical AST signature")

    raw_pointer_output = pointer_model.raw_complete(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        stop_text="</tool>",
    )
    materialized_output, decision, materialization_error = _resolve_pointer_output(
        frame=frame,
        model_digest=pointer_model.training_digest,
        raw_output=raw_pointer_output,
        inventory=inventory,
    )
    materialized_source = _source_from_decision(decision)
    signature = _ast_signature(materialized_source)
    positive_exact = (
        isinstance(decision, coding.ToolAction)
        and materialized_source == case.expected_source
        and signature == expected_signature
    )

    binding_off_raw_output = raw_model.complete(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        stop_text="</tool>",
    )
    binding_off_decision = coding._decode(
        frame,
        raw_model.training_digest,
        binding_off_raw_output,
    )
    binding_off_source = _source_from_decision(binding_off_decision)
    binding_off_signature = _ast_signature(binding_off_source)
    binding_off_exact = (
        isinstance(binding_off_decision, coding.ToolAction)
        and binding_off_source == case.expected_source
        and binding_off_signature == expected_signature
    )

    if not inventory.symbols:
        raise ValueError("starter-source lexical inventory is empty")
    rotated_symbols = (*inventory.symbols[1:], inventory.symbols[0])
    rotated_inventory = SymbolInventory(rotated_symbols)
    rotated_raw_output = pointer_model.raw_complete(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        stop_text="</tool>",
    )
    rotated_output, rotated_decision, rotated_error = _resolve_pointer_output(
        frame=frame,
        model_digest=pointer_model.training_digest,
        raw_output=rotated_raw_output,
        inventory=rotated_inventory,
    )
    rotated_source = _source_from_decision(rotated_decision)
    rotated_signature = _ast_signature(rotated_source)
    rotated_expected_source = materialize(
        pointerize(case.expected_source, inventory),
        rotated_inventory,
    )
    rotation_followed = (
        raw_pointer_output == rotated_raw_output
        and isinstance(rotated_decision, coding.ToolAction)
        and rotated_source == rotated_expected_source
        and rotated_source != case.expected_source
    )

    return {
        "case_id": case.case_id,
        "frame_sha256": frame.sha256,
        "inventory": {
            "derivation": (
                "first_unique_non_keyword_ascii_identifiers_from_starter_source"
            ),
            "lexical_position_count": len(inventory.symbols),
            "sha256": inventory.sha256,
        },
        "raw_pointer_output": raw_pointer_output,
        "raw_pointer_output_contains_pointer": bool(_POINTER.search(raw_pointer_output)),
        "materialized_output": materialized_output,
        "materialization_error": materialization_error,
        "materialized_decision": _decision_record(decision),
        "materialized_tool_action_source": materialized_source,
        "ast_signature": signature,
        "scoring_only_expected_ast_signature": expected_signature,
        "scoring_only_expected_source": case.expected_source,
        "positive_exact": positive_exact,
        "binding_off": {
            "raw_output": binding_off_raw_output,
            "decision": _decision_record(binding_off_decision),
            "tool_action_source": binding_off_source,
            "ast_signature": binding_off_signature,
            "exact_binding_retained": binding_off_exact,
        },
        "rotated_inventory_control": {
            "inventory": {
                "derivation": "one_position_rotation_of_starter_source_inventory",
                "lexical_position_count": len(rotated_inventory.symbols),
                "sha256": rotated_inventory.sha256,
            },
            "raw_pointer_output": rotated_raw_output,
            "raw_matches_positive": rotated_raw_output == raw_pointer_output,
            "materialized_output": rotated_output,
            "materialization_error": rotated_error,
            "decision": _decision_record(rotated_decision),
            "tool_action_source": rotated_source,
            "ast_signature": rotated_signature,
            "exact_rotated_source": rotated_expected_source,
            "follows_rotation_and_loses_original": rotation_followed,
        },
    }


def _empty_inventory_control(
    case: AlphaBindingCaseV1,
    pointer_model: PointerBoundCellModel,
) -> dict[str, object]:
    case.authenticate()
    frame = _frame(case.starter_source, case.prompt)
    prompt = coding._prompt(frame)
    inventory = SymbolInventory(())
    raw_output = pointer_model.raw_complete(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        stop_text="</tool>",
    )
    resolved, decision, error = _resolve_pointer_output(
        frame=frame,
        model_digest=pointer_model.training_digest,
        raw_output=raw_output,
        inventory=inventory,
    )
    stopped = (
        resolved is None
        and error is not None
        and isinstance(decision, coding.StopDecision)
        and decision.reason == "pointer_binding_failed_closed"
    )
    return {
        "case_id": case.case_id,
        "inventory": {
            "derivation": "deliberately_empty_control",
            "lexical_position_count": 0,
            "sha256": inventory.sha256,
        },
        "raw_pointer_output": raw_output,
        "raw_pointer_output_contains_pointer": bool(_POINTER.search(raw_output)),
        "materialized_output": resolved,
        "materialization_error": error,
        "decision": _decision_record(decision),
        "failed_closed_with_stop": stopped,
    }


def _reservation_record(
    pointer_model: PointerBoundCellModel,
    raw_model: CellLanguageModel,
) -> dict[str, object]:
    return {
        "schema": RESERVATION_SCHEMA,
        "status": "RUNNING",
        "public_coding_prior_sha256": PUBLIC_CODING_PRIOR_V1_SHA256,
        "alpha_binding_stream_sha256": ALPHA_BINDING_STREAM_SHA256,
        "case_count": CASE_COUNT,
        "training_example_count": CASE_COUNT * TRAINING_REPETITIONS,
        "model_config": {
            "max_order": MODEL_CONFIG.max_order,
            "max_phrases": MODEL_CONFIG.max_phrases,
            "min_phrase_frequency": MODEL_CONFIG.min_phrase_frequency,
        },
        "pointer_model_digest": pointer_model.training_digest,
        "raw_control_model_digest": raw_model.training_digest,
        "source_custody": {
            "gate_sha256": _sha256_file(Path(__file__)),
            "pointer_binding_sha256": _sha256_file(POINTER_SOURCE_PATH),
            "alpha_tasks_sha256": _sha256_file(TASK_SOURCE_PATH),
            "public_prior_module_sha256": _sha256_file(PRIOR_SOURCE_PATH),
            "persistent_coding_sha256": _sha256_file(PERSISTENT_CODING_SOURCE_PATH),
            "prior_corpus_sha256": SOURCE_CORPUS_SHA256,
            "prior_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "prior_freeze_commit": SOURCE_FREEZE_COMMIT,
        },
    }


def run_gate(
    pointer_model: PointerBoundCellModel,
    raw_model: CellLanguageModel,
) -> dict[str, object]:
    cases = source_local_alpha_binding_cases_v1()
    if len(cases) != CASE_COUNT:
        raise ValueError("fresh alpha-binding stream count changed")

    case_records = tuple(
        _run_case(case, pointer_model, raw_model) for case in cases
    )
    empty_inventory = _empty_inventory_control(cases[0], pointer_model)

    positive_exact_count = sum(bool(record["positive_exact"]) for record in case_records)
    pointer_output_count = sum(
        bool(record["raw_pointer_output_contains_pointer"]) for record in case_records
    )
    binding_off_exact_count = sum(
        bool(record["binding_off"]["exact_binding_retained"])
        for record in case_records
    )
    rotation_follow_count = sum(
        bool(
            record["rotated_inventory_control"][
                "follows_rotation_and_loses_original"
            ]
        )
        for record in case_records
    )
    passed = (
        positive_exact_count == CASE_COUNT
        and pointer_output_count == CASE_COUNT
        and binding_off_exact_count == 0
        and rotation_follow_count == CASE_COUNT
        and bool(empty_inventory["failed_closed_with_stop"])
    )
    return {
        "schema": SCHEMA,
        "status": (
            "PASS_SOURCE_LOCAL_ALPHA_BINDING_GATE_V1"
            if passed
            else "FAIL_SOURCE_LOCAL_ALPHA_BINDING_GATE_V1"
        ),
        "claim_boundary": (
            "neuron-selected lexical source positions over mechanically derived "
            "starter-source inventories; public expected sources are scoring-only; "
            "no execution, target R4 stream, private data, target teacher, "
            "protected evidence, or integrated-capability claim"
        ),
        "public_coding_prior_sha256": PUBLIC_CODING_PRIOR_V1_SHA256,
        "alpha_binding_stream_sha256": ALPHA_BINDING_STREAM_SHA256,
        "case_count": CASE_COUNT,
        "training_example_count": CASE_COUNT * TRAINING_REPETITIONS,
        "training_uses_all_public_prior_cases": True,
        "pointer_model_digest": pointer_model.training_digest,
        "raw_control_model_digest": raw_model.training_digest,
        "positive_exact_count": positive_exact_count,
        "raw_pointer_output_count": pointer_output_count,
        "binding_off_exact_count": binding_off_exact_count,
        "rotation_follow_count": rotation_follow_count,
        "empty_inventory_failed_closed": empty_inventory[
            "failed_closed_with_stop"
        ],
        "sandbox_executions": 0,
        "target_r4_stream_inputs": 0,
        "private_inputs": 0,
        "target_teacher_inputs": 0,
        "cases": case_records,
        "empty_inventory_control": empty_inventory,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    pointer_model, raw_model = _grow_models()
    reservation = _reservation_record(pointer_model, raw_model)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(reservation, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
    try:
        result = run_gate(pointer_model, raw_model)
    except BaseException as error:
        failed = dict(reservation)
        failed["status"] = "FAILED"
        failed["error_type"] = type(error).__name__
        failed["error"] = str(error)
        args.output.write_text(
            json.dumps(failed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    completed_reservation = dict(reservation)
    completed_reservation["status"] = "COMPLETED"
    result["reservation"] = completed_reservation
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
