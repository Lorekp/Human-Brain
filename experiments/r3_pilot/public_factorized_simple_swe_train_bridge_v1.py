"""Public factorized source-editor bridge into canonical Simple-SWE TRAIN.

This development cohort uses every ordinal from three supported public TRAIN
families.  The frozen factorized planner sees exactly the four fixed regression
input/output pairs for each task.  The 32 property cases remain outside the
planner and are scored only by the existing Simple-SWE sandbox together with
the regressions.

This is deliberately not the canonical prompt-only completion interface or an
M0 result: visible outputs are an adapter input.  Import and protocol checks do
not build a cell model, execute a task, or write an artifact.  The exact banked
two-task recovery STOP localizes the remaining defect after a real MAP_BIND
actuation; that immutable boundary plus its generic source repair are required
before sandbox or model initialization.  The result artifact is exclusive.
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

from experiments.r3_pilot import (
    public_factorized_source_edit_recovery_v1 as recovery_v1,
)
from experiments.r3_pilot import public_factorized_source_editor as source_editor
from experiments.r3_pilot import public_typed_factor_h1_root as factor_h1
from experiments.r3_pilot import public_typed_factor_horizon2_int as h2
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_program_observation as root_observation
from humanbrain import typed_program as tp
from humanbrain.simple_swe.protocol import ProtocolSpec, load_protocol
from humanbrain.simple_swe.sandbox import (
    load_worker_snapshot,
    python_runtime_identity,
    resolve_python_executable,
)
from humanbrain.simple_swe.tasks import GeneratedTask, generate_visible_partition

SCHEMA = "public-factorized-simple-swe-train-bridge-v1"
PASS_VERDICT = "PASS_PUBLIC_FACTORIZED_SIMPLE_SWE_TRAIN_BRIDGE_V1"
STOP_VERDICT = "STOP_PUBLIC_FACTORIZED_SIMPLE_SWE_TRAIN_BRIDGE_V1"

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
MAX_EDITS = 2
PUBLIC_LITERAL_BANK = tuple(range(-9, 10))

_PROTOCOL_PATH = REPO_ROOT / "benchmarks/simple_swe_protocol_v1.json"
_ARTIFACT_PATH = Path(__file__).with_suffix(".json")
_RECOVERY_ARTIFACT_PATH = Path(recovery_v1.__file__).with_suffix(".json")
_REQUIRED_RECOVERY_ARTIFACT_SHA256 = (
    "79ff62f36835eb252c9dacffd5fd2df9da600503724b367c67cb3a61a9239cd4"
)
_REQUIRED_REPAIRED_MODULE_SHA256 = {
    "public_typed_prompt_factors.py": (
        "a03c061c1c6db9bfad49a3a5eca849692545b59f8654b3349d0d189ad07d4da6"
    ),
}


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


def _snapshot(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot(item) for item in value)
    return value


def _visible_cases(task: GeneratedTask) -> tuple[source_editor.VisibleCase, ...]:
    """Expose regression cases only, normalizing sequence values for cells."""

    regressions = task.private.regression_cases
    if len(regressions) != VISIBLE_REGRESSION_CASES:
        raise RuntimeError("Simple-SWE regression-case count changed")
    cases: list[source_editor.VisibleCase] = []
    for hidden_case in regressions:
        arguments = _snapshot(hidden_case.args)
        if not isinstance(arguments, tuple):
            raise RuntimeError("Simple-SWE case arguments are not a tuple")
        cases.append(
            source_editor.VisibleCase(
                arguments=arguments,
                expected=_snapshot(hidden_case.expected),
            )
        )
    return tuple(cases)


def require_public_visible_support(
    tasks: Sequence[GeneratedTask],
) -> dict[str, object]:
    """Reject unsupported visible incumbents or goals before expensive setup."""

    selected = tuple(tasks)
    failures: list[dict[str, object]] = []
    for task in selected:
        visible = _visible_cases(task)
        incumbent = tp.Program(("variable-0",), task.public.signature)
        observed = root_observation.observe_current_program_root(
            incumbent,
            tuple(case.arguments for case in visible),
        )
        for case, current in zip(visible, observed.cases, strict=True):
            common = {
                "task_id": task.public.task_id,
                "family": task.public.family,
                "ordinal": task.public.ordinal,
                "case_index": current.case_index,
                "arguments": current.arguments,
                "declared_result": task.public.signature.result.to_record(),
            }
            if current.status != root_observation.STATUS_OK:
                failures.append(
                    {
                        **common,
                        "role": "incumbent",
                        "value": current.raw_output,
                        "status": current.status,
                    }
                )
            desired = factor_h1._encode_value(
                case.expected,
                task.public.signature.result,
            )
            if not desired.supported:
                failures.append(
                    {
                        **common,
                        "role": "desired",
                        "value": case.expected,
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
        "cases": sum(len(_visible_cases(task)) for task in selected),
        "int_min": output_cells.INT_MIN,
        "int_max": output_cells.INT_MAX,
        "list_capacity": output_cells.LIST_CAPACITY,
    }


def select_public_train_tasks(
    protocol: ProtocolSpec,
) -> tuple[GeneratedTask, ...]:
    """Select all 30 declared tasks in a family-balanced fixed order."""

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

    # Round-robin order prevents one family from monopolizing an early STOP.
    selected = tuple(
        indexed[(family, ordinal)]
        for ordinal in range(FAMILY_QUOTA)
        for family in PUBLIC_FAMILIES
    )
    if len(selected) != len(PUBLIC_FAMILIES) * FAMILY_QUOTA:
        raise RuntimeError("public TRAIN bridge task count changed")
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


def _require_recovery_boundary(
    path: Path = _RECOVERY_ARTIFACT_PATH,
) -> dict[str, object]:
    """Authenticate the exact immutable STOP that licenses the broad cohort."""

    try:
        artifact_bytes = path.read_bytes()
        payload = json.loads(artifact_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "exact source-edit recovery boundary is absent; bridge stays unrun"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError("source-edit recovery artifact is not a JSON object")
    results = payload.get("results")
    scalar = results[0] if isinstance(results, list) and len(results) == 2 else None
    mapped = results[1] if isinstance(results, list) and len(results) == 2 else None
    mapped_control = (
        mapped.get("max_edits_one_control") if isinstance(mapped, dict) else None
    )
    mapped_edit = (
        mapped_control.get("edit") if isinstance(mapped_control, dict) else None
    )
    mapped_actions = (
        mapped_edit.get("applied_actions") if isinstance(mapped_edit, dict) else None
    )
    first_action = (
        mapped_actions[0]
        if isinstance(mapped_actions, list) and len(mapped_actions) == 1
        else None
    )
    if (
        sha256(artifact_bytes).hexdigest() != _REQUIRED_RECOVERY_ARTIFACT_SHA256
        or payload.get("schema") != recovery_v1.SCHEMA
        or payload.get("verdict") != recovery_v1.STOP_VERDICT
        or payload.get("passed_tasks") != 1
        or payload.get("attempted_tasks") != 2
        or payload.get("total_tasks") != 2
        or payload.get("checkpoint_immutable") is not True
        or not isinstance(results, list)
        or len(results) != 2
        or not isinstance(scalar, dict)
        or scalar.get("name") != "four-minus-abs-datum"
        or scalar.get("passed") is not True
        or scalar.get("checkpoint_immutable") is not True
        or not isinstance(mapped, dict)
        or mapped.get("name") != "list-map-add-four"
        or mapped.get("passed") is not False
        or mapped.get("checkpoint_immutable") is not True
        or not isinstance(mapped_edit, dict)
        or mapped_edit.get("status") != source_editor.STATUS_PLANNER_STOP
        or mapped_edit.get("checkpoint_immutable") is not True
        or first_action
        != {
            "left_source": "BOUND0",
            "literal": None,
            "operator": "map",
            "path": [],
            "right_source": "SELECTED",
            "schema": "MAP_BIND",
        }
    ):
        raise RuntimeError(
            "source-edit recovery is not the exact MAP_BIND STOP boundary"
        )
    return {
        "schema": payload["schema"],
        "verdict": payload["verdict"],
        "artifact_sha256": _REQUIRED_RECOVERY_ARTIFACT_SHA256,
        "passed_tasks": 1,
        "failed_task": "list-map-add-four",
        "first_applied_action": first_action,
        "model_checkpoint_sha256": payload.get("model_checkpoint_sha256"),
        "checkpoint_immutable": True,
    }


def _require_repaired_mechanism() -> dict[str, object]:
    """Pin the exact prompt-projection module carrying the generic repair."""

    observed: dict[str, str] = {}
    for name, expected_sha256 in _REQUIRED_REPAIRED_MODULE_SHA256.items():
        path = Path(__file__).with_name(name)
        try:
            module_sha256 = sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise RuntimeError(f"required repaired module is absent: {name}") from error
        if module_sha256 != expected_sha256:
            raise RuntimeError(f"required repaired module changed: {name}")
        observed[name] = module_sha256
    return {
        "verified": True,
        "module_sha256": observed,
        "repair": "materialized MAP_BIND body-to-DONE prompt projection",
    }


def _require_bridge_prerequisite() -> dict[str, object]:
    return {
        "recovery_boundary": _require_recovery_boundary(
            _RECOVERY_ARTIFACT_PATH
        ),
        "repaired_mechanism": _require_repaired_mechanism(),
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


def _task_record(
    editor: source_editor.FactorizedSourceEditor,
    task: GeneratedTask,
    protocol: ProtocolSpec,
    runtime: SandboxRuntime,
) -> dict[str, object]:
    started = perf_counter()
    initial_checkpoint = editor.planner.live_checkpoint_sha256
    incumbent = tp.Program(("variable-0",), task.public.signature)
    result = editor.edit(
        incumbent,
        task.public.prompt,
        _visible_cases(task),
        max_edits=MAX_EDITS,
    )
    if editor.planner.live_checkpoint_sha256 != initial_checkpoint:
        raise RuntimeError("shared planner checkpoint changed across a TRAIN task")

    sandbox_record: dict[str, object] | None = None
    passed = False
    if result.status == source_editor.STATUS_DONE:
        sandbox_result = source_editor.execute_completed_source(
            result,
            task,
            protocol.execution,
            python_executable=runtime.executable,
            python_executable_sha256=runtime.identity["executable_sha256"],
            worker_source=runtime.worker_source,
            worker_sha256=runtime.worker_sha256,
        )
        sandbox_record = asdict(sandbox_result)
        passed = sandbox_result.passed

    return {
        "task_id": task.public.task_id,
        "family": task.public.family,
        "ordinal": task.public.ordinal,
        "task_sha256": task.public.task_sha256,
        "status": result.status,
        "stop_reason": result.stop_reason,
        "edit_count": result.edit_count,
        "applied_actions": tuple(asdict(action) for action in result.applied_actions),
        "terminal_program_sha256": result.program.sha256,
        "terminal_source_sha256": sha256(result.source.encode("utf-8")).hexdigest(),
        "checkpoint_immutable": result.checkpoint_audit.immutable,
        "visible_regression_cases": VISIBLE_REGRESSION_CASES,
        "property_cases_exposed_to_planner": 0,
        "property_cases_scored_in_sandbox": UNEXPOSED_PROPERTY_CASES,
        "sandbox": sandbox_record,
        "passed": passed,
        "wall_time_seconds": perf_counter() - started,
    }


def run_public_train_cohort(
    model: h2.FrozenFactorizedHorizon2Int,
    runtime: SandboxRuntime,
    protocol: ProtocolSpec,
    tasks: Sequence[GeneratedTask],
    prerequisite: dict[str, object],
) -> dict[str, object]:
    """Run the fixed cohort with one planner and a decisive family gate."""

    selected = tuple(tasks)
    canonical = select_public_train_tasks(protocol)
    if tuple(task.public.task_id for task in selected) != tuple(
        task.public.task_id for task in canonical
    ):
        raise RuntimeError("runner task order differs from the complete public cohort")
    recovery_boundary = prerequisite.get("recovery_boundary")
    repaired_mechanism = prerequisite.get("repaired_mechanism")
    if (
        not isinstance(recovery_boundary, dict)
        or recovery_boundary.get("verdict") != recovery_v1.STOP_VERDICT
        or recovery_boundary.get("artifact_sha256")
        != _REQUIRED_RECOVERY_ARTIFACT_SHA256
        or not isinstance(repaired_mechanism, dict)
        or repaired_mechanism.get("verified") is not True
    ):
        raise RuntimeError("exact recovery boundary and source repair are required")
    if model.live_checkpoint_sha256 != model.checkpoint_sha256:
        raise RuntimeError("shared planner is not frozen before the cohort")

    started = perf_counter()
    editor = source_editor.FactorizedSourceEditor(model)
    results: list[dict[str, object]] = []
    attempted = Counter[str]()
    passed = Counter[str]()
    stop_family: str | None = None

    for task in selected:
        result = _task_record(editor, task, protocol, runtime)
        results.append(result)
        family = task.public.family
        attempted[family] += 1
        passed[family] += int(bool(result["passed"]))
        if attempted[family] - passed[family] >= FAMILY_FAILURE_STOP:
            stop_family = family
            break

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
        and model.live_checkpoint_sha256 == model.checkpoint_sha256
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
            "regression_outputs_visible_per_task": VISIBLE_REGRESSION_CASES,
            "property_cases_visible_per_task": 0,
            "property_cases_sandboxed_per_task": UNEXPOSED_PROPERTY_CASES,
        },
        "planner": {
            "builds": 1,
            "literal_bank": PUBLIC_LITERAL_BANK,
            "max_edits": MAX_EDITS,
            "checkpoint_sha256": model.checkpoint_sha256,
            "checkpoint_immutable": model.live_checkpoint_sha256
            == model.checkpoint_sha256,
        },
        "prerequisite": prerequisite,
        "sandbox": {
            "python": runtime.identity,
            "worker_sha256": runtime.worker_sha256,
            "limits": asdict(protocol.execution),
        },
        "wall_time_seconds": perf_counter() - started,
        "claim_boundary": (
            "All 30 tasks from three canonical public TRAIN families; four "
            "regression outputs are visible to the editor and 32 property "
            "cases are sandbox-only. This is not prompt-only completion, M0, "
            "DEV/eval, protected, held-out, Path-2, or C1 evidence."
        ),
    }


def write_public_artifact(payload: dict[str, object], path: Path) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(_json_bytes(payload))
    except FileExistsError as error:
        raise RuntimeError("public TRAIN bridge artifact already exists") from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=_PROTOCOL_PATH)
    parser.add_argument("--json", type=Path, default=_ARTIFACT_PATH)
    parser.add_argument("--python-executable", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.json.exists():
        raise RuntimeError("public TRAIN bridge artifact already exists")

    protocol = load_protocol(args.protocol)
    tasks = select_public_train_tasks(protocol)
    prerequisite = _require_bridge_prerequisite()
    require_public_visible_support(tasks)

    # The exact STOP boundary, repair, and support preflight precede setup.
    runtime = _prepare_sandbox(protocol, args.python_executable)
    model = h2.build_public_factorized_horizon2_int(literals=PUBLIC_LITERAL_BANK)
    payload = run_public_train_cohort(
        model,
        runtime,
        protocol,
        tasks,
        prerequisite,
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
    "MAX_EDITS",
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
