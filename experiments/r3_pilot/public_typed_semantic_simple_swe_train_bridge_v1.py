"""Broad public-TRAIN Simple-SWE gate for bottom-up typed semantic synthesis.

The shared frozen synthesizer receives only a typed signature and the four
visible regression input/output rows.  Prompt text, task ids, family labels,
and the 32 property rows never enter model selection.  A selected completion
is scored once by the canonical Simple-SWE sandbox over the regressions and
property cases.  This is public development evidence, not M0, DEV/eval,
protected, held-out, Path-2, or C1 evidence.
"""

# ruff: noqa: E402 -- direct-script execution installs repo import roots first.

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_synthesizer as semantic
from humanbrain.simple_swe.protocol import ProtocolSpec, load_protocol
from humanbrain.simple_swe.sandbox import (
    execute_completion,
    load_worker_snapshot,
    python_runtime_identity,
    resolve_python_executable,
)
from humanbrain.simple_swe.tasks import GeneratedTask, generate_visible_partition
from humanbrain.typed_program_types import ProgramType, TypedSignature

SCHEMA = "public-typed-semantic-simple-swe-train-bridge-v1"
PASS_VERDICT = "PASS_PUBLIC_TYPED_SEMANTIC_SIMPLE_SWE_TRAIN_BRIDGE_V1"
STOP_VERDICT = "STOP_PUBLIC_TYPED_SEMANTIC_SIMPLE_SWE_TRAIN_BRIDGE_V1"

PUBLIC_FAMILIES = (
    "t0_scalar_offset",
    "t1_list_map_add",
    "t1_list_map_abs",
)
FAMILY_QUOTA = 10
FAMILY_PASS_THRESHOLD = 8
FAMILY_FAILURE_STOP = FAMILY_QUOTA - FAMILY_PASS_THRESHOLD + 1
VISIBLE_REGRESSION_CASES = 4
UNEXPOSED_PROPERTY_CASES = 32
PUBLIC_LITERAL_BANK = tuple(range(-9, 10))
MAX_AST_COST = 6

_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)
_PROTOCOL_PATH = REPO_ROOT / "benchmarks/simple_swe_protocol_v1.json"
_ARTIFACT_PATH = Path(__file__).with_suffix(".json")
_PRIOR_BRIDGE_ARTIFACT_PATH = Path(__file__).with_name(
    "public_factorized_simple_swe_train_bridge_v1.json"
)
_REQUIRED_PRIOR_BRIDGE_ARTIFACT_SHA256 = (
    "f748e50cc98ddc987bcdbbc5804b4203e04b0c0376509d66f374f822781fae32"
)
_MECHANISM_SOURCE_PATHS = (
    "experiments/r3_pilot/public_typed_semantic_simple_swe_train_bridge_v1.py",
    "experiments/r3_pilot/public_typed_semantic_synthesizer.py",
    "experiments/r3_pilot/public_typed_semantic_vector_race.py",
    "experiments/r3_pilot/public_typed_operator_successor.py",
    "experiments/r3_pilot/public_typed_output_cells.py",
    "src/humanbrain/typed_semantic_synthesis.py",
    "src/humanbrain/typed_program.py",
    "src/humanbrain/typed_program_types.py",
    "src/humanbrain/resting_glif_race.py",
    "src/humanbrain/resting_glif_tier_beam.py",
)


@dataclass(frozen=True, slots=True)
class SandboxRuntime:
    executable: Path
    identity: dict[str, str]
    worker_source: bytes
    worker_sha256: str


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _snapshot(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot(item) for item in value)
    return value


def _visible_problem(
    task: GeneratedTask,
) -> tuple[TypedSignature, tuple[tuple[object, ...], ...], tuple[object, ...]]:
    regressions = task.private.regression_cases
    if len(regressions) != VISIBLE_REGRESSION_CASES:
        raise RuntimeError("Simple-SWE regression-case count changed")
    arguments: list[tuple[object, ...]] = []
    desired: list[object] = []
    for case in regressions:
        row = _snapshot(case.args)
        if not isinstance(row, tuple):
            raise RuntimeError("Simple-SWE case arguments are not a tuple")
        arguments.append(row)
        desired.append(_snapshot(case.expected))
    return task.public.signature, tuple(arguments), tuple(desired)


def _encode_visible(
    value: object,
    declared_type: ProgramType,
) -> output_cells.TypedOutputEncoding:
    if declared_type == _INT:
        output_type = output_cells.OutputType.INT
    elif declared_type == _LIST_INT:
        output_type = output_cells.OutputType.LIST_INT
        if isinstance(value, tuple):
            value = list(value)
    else:
        raise RuntimeError("semantic bridge supports only Int and List[Int]")
    return output_cells.TypedOutputCellBank.encode(
        value,
        declared_type=output_type,
    )


def require_public_visible_support(
    tasks: Sequence[GeneratedTask],
) -> dict[str, object]:
    """Fail before model setup if any visible semantic row is unrepresentable."""

    selected = tuple(tasks)
    failures: list[dict[str, object]] = []
    encoded_values = 0
    for task in selected:
        signature, arguments, desired = _visible_problem(task)
        for case_index, (row, expected) in enumerate(
            zip(arguments, desired, strict=True)
        ):
            if len(row) != len(signature.parameters):
                failures.append(
                    {
                        "task_id": task.public.task_id,
                        "case_index": case_index,
                        "role": "arguments",
                        "status": "ARITY_MISMATCH",
                    }
                )
                continue
            for parameter, value in zip(signature.parameters, row, strict=True):
                encoded_values += 1
                if not _encode_visible(value, parameter.type).supported:
                    failures.append(
                        {
                            "task_id": task.public.task_id,
                            "case_index": case_index,
                            "role": parameter.name,
                            "status": "OUTPUT_UNKNOWN",
                        }
                    )
            encoded_values += 1
            if not _encode_visible(expected, signature.result).supported:
                failures.append(
                    {
                        "task_id": task.public.task_id,
                        "case_index": case_index,
                        "role": "desired",
                        "status": "OUTPUT_UNKNOWN",
                    }
                )
    if failures:
        raise RuntimeError(
            "public TRAIN visible values exceed the physical output bank: "
            + json.dumps(failures, sort_keys=True, allow_nan=False)
        )
    return {
        "tasks": len(selected),
        "cases": sum(len(_visible_problem(task)[1]) for task in selected),
        "encoded_values": encoded_values,
        "int_min": output_cells.INT_MIN,
        "int_max": output_cells.INT_MAX,
        "list_capacity": output_cells.LIST_CAPACITY,
    }


def select_public_train_tasks(
    protocol: ProtocolSpec,
) -> tuple[GeneratedTask, ...]:
    """Select all 30 declared tasks in family-balanced fixed order."""

    if protocol.path != _PROTOCOL_PATH.resolve():
        raise RuntimeError("bridge requires the canonical Simple-SWE v1 protocol")
    if protocol.regression_cases_per_task != VISIBLE_REGRESSION_CASES:
        raise RuntimeError("canonical regression-case count changed")
    if protocol.property_cases_per_task != UNEXPOSED_PROPERTY_CASES:
        raise RuntimeError("canonical property-case count changed")
    train = protocol.partitions["train"]
    if any(
        train.family_quotas.get(family) != FAMILY_QUOTA
        for family in PUBLIC_FAMILIES
    ):
        raise RuntimeError("selected public TRAIN family quota changed")

    generated = generate_visible_partition(protocol, "train")
    indexed = {
        (task.public.family, task.public.ordinal): task
        for task in generated
        if task.public.family in PUBLIC_FAMILIES
    }
    expected_keys = {
        (family, ordinal)
        for family in PUBLIC_FAMILIES
        for ordinal in range(FAMILY_QUOTA)
    }
    if set(indexed) != expected_keys:
        raise RuntimeError("public TRAIN selection omitted or duplicated an ordinal")
    selected = tuple(
        indexed[(family, ordinal)]
        for ordinal in range(FAMILY_QUOTA)
        for family in PUBLIC_FAMILIES
    )
    if len({task.public.task_id for task in selected}) != len(selected):
        raise RuntimeError("public TRAIN bridge contains duplicate task ids")
    for task in selected:
        if task.public.partition != "train":
            raise RuntimeError("bridge may use public TRAIN tasks only")
        if len(task.private.regression_cases) != VISIBLE_REGRESSION_CASES:
            raise RuntimeError("selected task regression-case count changed")
        if len(task.private.property_cases) != UNEXPOSED_PROPERTY_CASES:
            raise RuntimeError("selected task property-case count changed")
    return selected


def _require_prior_bridge_boundary(
    path: Path = _PRIOR_BRIDGE_ARTIFACT_PATH,
) -> dict[str, object]:
    """Authenticate the exact broad STOP that licensed this replacement."""

    try:
        artifact_bytes = path.read_bytes()
        payload = json.loads(artifact_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "exact prior public-TRAIN bridge boundary is absent; gate stays unrun"
        ) from error
    observed_sha256 = sha256(artifact_bytes).hexdigest()
    if (
        observed_sha256 != _REQUIRED_PRIOR_BRIDGE_ARTIFACT_SHA256
        or not isinstance(payload, dict)
        or payload.get("schema") != "public-factorized-simple-swe-train-bridge-v1"
        or payload.get("verdict")
        != "STOP_PUBLIC_FACTORIZED_SIMPLE_SWE_TRAIN_BRIDGE_V1"
        or payload.get("declared_tasks") != 30
        or payload.get("attempted_tasks") != 7
        or payload.get("passed_tasks") != 2
        or payload.get("stop_family") != "t0_scalar_offset"
        or payload.get("stop_reason") != "FAMILY_FAILURE_LIMIT"
    ):
        raise RuntimeError("prior public-TRAIN bridge is not the exact broad STOP")
    return {
        "schema": payload["schema"],
        "verdict": payload["verdict"],
        "artifact_sha256": observed_sha256,
        "declared_tasks": 30,
        "attempted_tasks": 7,
        "passed_tasks": 2,
        "stop_family": "t0_scalar_offset",
    }


def _mechanism_source_freeze() -> dict[str, object]:
    files: dict[str, str] = {}
    for relative in _MECHANISM_SOURCE_PATHS:
        path = REPO_ROOT / relative
        try:
            files[relative] = sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise RuntimeError(f"required semantic source is absent: {relative}") from error
    return {
        "files": files,
        "bundle_sha256": sha256(_canonical_json(files)).hexdigest(),
    }


def _prepare_sandbox(
    protocol: ProtocolSpec,
    python_executable: Path | None,
) -> SandboxRuntime:
    executable = resolve_python_executable(python_executable)
    language = protocol.raw.get("language")
    identity = python_runtime_identity(executable, language)
    worker_path = SRC_ROOT / "humanbrain/simple_swe/_worker.py"
    worker_sha256 = sha256(worker_path.read_bytes()).hexdigest()
    return SandboxRuntime(
        executable=executable,
        identity=identity,
        worker_source=load_worker_snapshot(worker_sha256),
        worker_sha256=worker_sha256,
    )


def _tier_record(
    attempt: semantic.SemanticSynthesisTierAttempt,
) -> dict[str, object]:
    race = attempt.race
    race_record: dict[str, object] | None = None
    if race is not None:
        race_record = {
            "status": race.status,
            "candidate_keys": race.candidate_keys,
            "candidate_vector_sha256s": race.candidate_vector_sha256s,
            "desired_vector_sha256": race.desired_vector_sha256,
            "detector_sha256s": race.detector_sha256s,
            "projection_sha256": race.projection_sha256,
            "currents_sha256": race.currents_sha256,
            "tier_sha256": race.tier_sha256,
            "exact_winner_keys": race.exact_winner_keys,
            "winner_key": race.winner_key,
            "sha256": race.sha256,
        }
    return {
        "ast_cost": attempt.ast_cost,
        "evaluated_candidates": attempt.evaluated_candidates,
        "representative_candidate_ids": attempt.representative_candidate_ids,
        "race": race_record,
    }


def _task_record(
    model: semantic.FrozenTypedSemanticSynthesizer,
    task: GeneratedTask,
    protocol: ProtocolSpec,
    runtime: SandboxRuntime,
) -> dict[str, object]:
    started = perf_counter()
    signature, arguments, desired = _visible_problem(task)
    visible_problem_sha256 = sha256(
        _canonical_json(
            {
                "signature": signature.to_record(),
                "arguments": arguments,
                "desired_outputs": desired,
            }
        )
    ).hexdigest()
    initial_checkpoint = model.live_checkpoint_sha256
    if initial_checkpoint != model.checkpoint_sha256:
        raise RuntimeError("shared semantic synthesizer is not frozen")

    # This is the complete model-selection seam.  No task metadata or prompt is
    # supplied to the synthesizer.
    decision = model.synthesize(signature, arguments, desired)
    if (
        model.live_checkpoint_sha256 != initial_checkpoint
        or decision.checkpoint_before_sha256 != initial_checkpoint
        or decision.checkpoint_after_sha256 != initial_checkpoint
    ):
        raise RuntimeError("shared semantic synthesizer changed across a TRAIN task")

    sandbox_record: dict[str, object] | None = None
    passed = False
    program_sha256: str | None = None
    source_sha256: str | None = None
    if decision.status == semantic.STATUS_DONE:
        if decision.program is None or decision.source is None:
            raise RuntimeError("DONE semantic decision omitted its compiled program")
        program_sha256 = decision.program.sha256
        source_sha256 = sha256(decision.source.encode("utf-8")).hexdigest()
        sandbox_result = execute_completion(
            decision.source,
            task.public,
            task.private,
            protocol.execution,
            python_executable=runtime.executable,
            python_executable_sha256=runtime.identity["executable_sha256"],
            worker_source=runtime.worker_source,
            worker_sha256=runtime.worker_sha256,
        )
        sandbox_record = asdict(sandbox_result)
        passed = sandbox_result.passed
    elif decision.status != semantic.STATUS_STOP:
        raise RuntimeError("semantic synthesizer returned an unknown terminal status")

    return {
        "task_id": task.public.task_id,
        "family": task.public.family,
        "ordinal": task.public.ordinal,
        "task_sha256": task.public.task_sha256,
        "signature": signature.to_record(),
        "visible_problem_sha256": visible_problem_sha256,
        "semantic_status": decision.status,
        "stop_reason": decision.stop_reason,
        "candidate_count": decision.candidate_count,
        "observed_semantic_classes": decision.observed_semantic_classes,
        "tier_attempts": tuple(_tier_record(item) for item in decision.tier_attempts),
        "selected_candidate_id": decision.selected_candidate_id,
        "terminal_program_sha256": program_sha256,
        "terminal_source_sha256": source_sha256,
        "operator_predictions": decision.operator_predictions,
        "operator_races": decision.operator_races,
        "operator_races_sha256": decision.operator_races_sha256,
        "checkpoint_immutable": True,
        "visible_regression_cases": VISIBLE_REGRESSION_CASES,
        "property_cases_exposed_to_synthesizer": 0,
        "property_cases_scored_in_sandbox": UNEXPOSED_PROPERTY_CASES,
        "sandbox": sandbox_record,
        "passed": passed,
        "wall_time_seconds": perf_counter() - started,
    }


def run_public_train_cohort(
    model: semantic.FrozenTypedSemanticSynthesizer,
    training_summary: object,
    runtime: SandboxRuntime,
    protocol: ProtocolSpec,
    tasks: Sequence[GeneratedTask],
    prerequisite: dict[str, object],
    source_freeze: dict[str, object],
) -> dict[str, object]:
    """Run the fixed cohort with one synthesizer and a decisive family gate."""

    selected = tuple(tasks)
    canonical = select_public_train_tasks(protocol)
    if tuple(task.public.task_id for task in selected) != tuple(
        task.public.task_id for task in canonical
    ):
        raise RuntimeError("runner task order differs from the complete public cohort")
    if (
        prerequisite.get("verdict")
        != "STOP_PUBLIC_FACTORIZED_SIMPLE_SWE_TRAIN_BRIDGE_V1"
        or prerequisite.get("artifact_sha256")
        != _REQUIRED_PRIOR_BRIDGE_ARTIFACT_SHA256
    ):
        raise RuntimeError("exact prior broad STOP boundary is required")
    if source_freeze != _mechanism_source_freeze():
        raise RuntimeError("semantic mechanism source changed before the cohort")
    if model.live_checkpoint_sha256 != model.checkpoint_sha256:
        raise RuntimeError("shared semantic synthesizer is not frozen before the cohort")

    started = perf_counter()
    results: list[dict[str, object]] = []
    attempted = Counter[str]()
    passed = Counter[str]()
    stop_family: str | None = None
    for task in selected:
        result = _task_record(model, task, protocol, runtime)
        results.append(result)
        family = task.public.family
        attempted[family] += 1
        passed[family] += int(bool(result["passed"]))
        if attempted[family] - passed[family] >= FAMILY_FAILURE_STOP:
            stop_family = family
            break

    after_sources = _mechanism_source_freeze()
    if after_sources != source_freeze:
        raise RuntimeError("semantic mechanism source changed during the cohort")
    checkpoint_immutable = model.live_checkpoint_sha256 == model.checkpoint_sha256
    summaries = {
        family: {
            "passed": passed[family],
            "failed": attempted[family] - passed[family],
            "attempted": attempted[family],
            "declared": FAMILY_QUOTA,
            "gate": FAMILY_PASS_THRESHOLD,
        }
        for family in PUBLIC_FAMILIES
    }
    cohort_passed = (
        len(results) == len(selected)
        and all(passed[family] >= FAMILY_PASS_THRESHOLD for family in PUBLIC_FAMILIES)
        and checkpoint_immutable
    )
    return {
        "schema": SCHEMA,
        "verdict": PASS_VERDICT if cohort_passed else STOP_VERDICT,
        "stop_reason": None if cohort_passed else "FAMILY_FAILURE_LIMIT",
        "stop_family": stop_family,
        "declared_tasks": len(selected),
        "attempted_tasks": len(results),
        "passed_tasks": sum(passed.values()),
        "task_order": tuple(task.public.task_id for task in selected),
        "attempted_task_order": tuple(result["task_id"] for result in results),
        "family_results": summaries,
        "results": tuple(results),
        "protocol": {
            "protocol_id": protocol.protocol_id,
            "sha256": protocol.sha256,
            "partition": "train",
            "families": PUBLIC_FAMILIES,
        },
        "visibility_contract": {
            "typed_signature_visible": True,
            "regression_arguments_visible_per_task": VISIBLE_REGRESSION_CASES,
            "regression_outputs_visible_per_task": VISIBLE_REGRESSION_CASES,
            "prompt_visible_to_synthesizer": False,
            "task_or_family_identity_visible_to_synthesizer": False,
            "property_cases_visible_per_task": 0,
            "property_cases_sandboxed_per_task": UNEXPOSED_PROPERTY_CASES,
        },
        "synthesizer": {
            "builds": 1,
            "literal_bank": PUBLIC_LITERAL_BANK,
            "max_ast_cost": MAX_AST_COST,
            "checkpoint_sha256": model.checkpoint_sha256,
            "operator_checkpoint_sha256": model.operator.checkpoint_sha256,
            "checkpoint_immutable": checkpoint_immutable,
            "training_summary": asdict(training_summary),
            "source_freeze": source_freeze,
        },
        "prerequisite": prerequisite,
        "sandbox": {
            "python": runtime.identity,
            "worker_sha256": runtime.worker_sha256,
            "limits": asdict(protocol.execution),
        },
        "wall_time_seconds": perf_counter() - started,
        "claim_boundary": (
            "All 30 tasks from three canonical public TRAIN families; typed "
            "signature plus four regression input/output rows are visible to "
            "the shared frozen bottom-up synthesizer, while prompt/task/family "
            "identity and 32 property cases are excluded from model selection. "
            "This is not M0, DEV/eval, protected, held-out, Path-2, or C1 evidence."
        ),
    }


def write_public_artifact(payload: dict[str, object], path: Path) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(_json_bytes(payload))
    except FileExistsError as error:
        raise RuntimeError("typed semantic TRAIN bridge artifact already exists") from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=_PROTOCOL_PATH)
    parser.add_argument("--json", type=Path, default=_ARTIFACT_PATH)
    parser.add_argument("--python-executable", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.json.exists():
        raise RuntimeError("typed semantic TRAIN bridge artifact already exists")

    protocol = load_protocol(args.protocol)
    tasks = select_public_train_tasks(protocol)
    prerequisite = _require_prior_bridge_boundary(
        _PRIOR_BRIDGE_ARTIFACT_PATH
    )
    require_public_visible_support(tasks)
    source_freeze = _mechanism_source_freeze()

    # Boundary, visibility, and source checks precede sandbox/model setup.
    runtime = _prepare_sandbox(protocol, args.python_executable)
    model, training_summary = semantic.build_public_typed_semantic_synthesizer(
        literals=PUBLIC_LITERAL_BANK,
        max_ast_cost=MAX_AST_COST,
    )
    payload = run_public_train_cohort(
        model,
        training_summary,
        runtime,
        protocol,
        tasks,
        prerequisite,
        source_freeze,
    )
    write_public_artifact(payload, args.json)
    print(json.dumps({"artifact": str(args.json), "verdict": payload["verdict"]}))
    return 0 if payload["verdict"] == PASS_VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAMILY_FAILURE_STOP",
    "FAMILY_PASS_THRESHOLD",
    "FAMILY_QUOTA",
    "MAX_AST_COST",
    "PASS_VERDICT",
    "PUBLIC_FAMILIES",
    "PUBLIC_LITERAL_BANK",
    "SCHEMA",
    "STOP_VERDICT",
    "UNEXPOSED_PROPERTY_CASES",
    "VISIBLE_REGRESSION_CASES",
    "require_public_visible_support",
    "run_public_train_cohort",
    "select_public_train_tasks",
    "write_public_artifact",
]
