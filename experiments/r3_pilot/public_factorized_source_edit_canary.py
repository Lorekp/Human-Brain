"""One-shot four-task public source-editing canary.

Each task starts from the typed identity program.  The frozen factorized H=2
planner sees only the source-declared visible cases, applies one real edit,
reobserves, and replans.  Disjoint source-declared score cases reach only the
existing Simple-SWE sandbox.  This is public pre-M0 development, not a
canonical Simple-SWE, held-out, C1, or capability run.

The exact banked created-child V3 PASS artifact licenses this public increment.
Importing this module spends nothing; the exclusive result artifact permits one
run only.
"""

# ruff: noqa: E402 -- direct-script execution installs repo import roots first.

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.r3_pilot import public_factorized_source_editor as source_editor
from experiments.r3_pilot import public_typed_factor_horizon2_int as h2
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_program_observation as observation
from humanbrain import typed_program as tp
from humanbrain.simple_swe import make_task
from humanbrain.simple_swe.protocol import ExecutionLimits
from humanbrain.simple_swe.sandbox import (
    ExecutionResult,
    execute_completion,
    load_worker_snapshot,
    python_runtime_identity,
    resolve_python_executable,
)
from humanbrain.simple_swe.tasks import GeneratedTask
from humanbrain.typed_program_types import (
    ProgramParameter,
    ProgramType,
    TypedSignature,
)

SCHEMA = "public-factorized-source-edit-canary-v1"
PASS_VERDICT = "PASS_PUBLIC_FACTORIZED_SOURCE_EDIT_CANARY_V1"
STOP_VERDICT = "STOP_PUBLIC_FACTORIZED_SOURCE_EDIT_CANARY_V1"
PUBLIC_LITERAL_BANK = tuple(range(-4, 5))
MAX_EDITS = 2
_ARTIFACT_PATH = Path(__file__).with_suffix(".json")
_CREATED_CHILD_V3_PATH = Path(__file__).with_name(
    "public_typed_factor_horizon2_int_created_child_gate_v3.json"
)
_REQUIRED_CREATED_CHILD_VERDICT = "PASS_TYPED_FACTOR_HORIZON2_INT_CREATED_CHILD_V3"
_REQUIRED_CREATED_CHILD_ARTIFACT_SHA256 = (
    "432872d76a9f628305f2a921843586957f587d033112bde03d28f6a442b10104"
)

PUBLIC_EXECUTION_LIMITS = ExecutionLimits(
    wall_seconds=2.0,
    cpu_seconds=2,
    memory_mib=128,
    processes=1,
    output_bytes=4_096,
    max_completion_bytes=8_192,
    max_ast_nodes=128,
    python_hash_seed=17,
)


def _int_signature(entry_point: str) -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(entry_point, (ProgramParameter("value", integer),), integer)


def _list_signature(entry_point: str) -> TypedSignature:
    integer = ProgramType.Int()
    list_int = ProgramType.List(integer)
    return TypedSignature(
        entry_point,
        (ProgramParameter("values", list_int),),
        list_int,
    )


def _absolute(path: tuple[int, ...]) -> tp.FactorAction:
    return tp.FactorAction(path, "APPLY1", "absolute", "SELECTED", "NONE", None)


def _subtract(
    path: tuple[int, ...], left: str, right: str, literal: int
) -> tp.FactorAction:
    return tp.FactorAction(path, "APPLY2", "subtract", left, right, literal)


_MAP = tp.FactorAction((), "MAP_BIND", "map", "BOUND0", "SELECTED", None)


def _score_task(
    *,
    family: str,
    signature: TypedSignature,
    description: str,
    oracle,
    arguments: tuple[tuple[object, ...], ...],
) -> GeneratedTask:
    return make_task(
        partition="public-canary",
        family=family,
        ordinal=0,
        signature=signature,
        description=description,
        oracle=oracle,
        regression_inputs=arguments[:4],
        property_inputs=tuple(
            (case, "public-disjoint") for case in arguments[4:]
        ),
    )


@dataclass(frozen=True, slots=True)
class PublicSourceEditCanaryTask:
    name: str
    prompt: str
    signature: TypedSignature
    visible_cases: tuple[source_editor.VisibleCase, ...]
    score_task: GeneratedTask
    expected_actions: tuple[tp.FactorAction, tp.FactorAction]
    expected_tokens: tuple[str, ...]
    expected_source: str
    reverse_equivalent_actions: tuple[tp.FactorAction, tp.FactorAction] | None = None

    @property
    def identity(self) -> tp.Program:
        return tp.Program(("variable-0",), self.signature)


_SCALAR_SCORE_ARGS = ((0,), (-6,), (2,), (13,), (-21,), (-3,), (7,), (34,))
_LIST_SCORE_ARGS = (
    ([0],),
    ([1],),
    ([-7, 2],),
    ([6, -1, -9],),
    ([3, 3],),
    ([-12, 4, 0],),
    ([1, -5],),
    ([8],),
)

_SIG_ABS_MINUS_THREE = _int_signature("solve_abs_minus_three")
_SIG_THREE_MINUS_ABS = _int_signature("solve_three_minus_abs")
_SIG_MAP_ABSOLUTE = _list_signature("solve_map_absolute")
_SIG_MAP_SUBTRACT_TWO = _list_signature("solve_map_subtract_two")


PUBLIC_TASKS = (
    PublicSourceEditCanaryTask(
        "abs-value-minus-three",
        "Return abs(the input) - 3.",
        _SIG_ABS_MINUS_THREE,
        tuple(
            source_editor.VisibleCase((value,), abs(value) - 3)
            for value in (-8, -1, 4, 9)
        ),
        _score_task(
            family="t0_factorized_abs_minus_three",
            signature=_SIG_ABS_MINUS_THREE,
            description="Return abs(value) minus three.",
            oracle=lambda value: abs(value) - 3,
            arguments=_SCALAR_SCORE_ARGS,
        ),
        (_subtract((), "SELECTED", "LITERAL", 3), _absolute((0,))),
        ("subtract", "absolute", "variable-0", "integer", "+", "3", "end"),
        "def solve_abs_minus_three(value):\n    return abs(value) - 3\n",
        (_absolute(()), _subtract((), "SELECTED", "LITERAL", 3)),
    ),
    PublicSourceEditCanaryTask(
        "three-minus-abs-value",
        "Return 3 - abs(the input).",
        _SIG_THREE_MINUS_ABS,
        tuple(
            source_editor.VisibleCase((value,), 3 - abs(value))
            for value in (-8, -1, 4, 9)
        ),
        _score_task(
            family="t0_factorized_three_minus_abs",
            signature=_SIG_THREE_MINUS_ABS,
            description="Return three minus abs(value).",
            oracle=lambda value: 3 - abs(value),
            arguments=_SCALAR_SCORE_ARGS,
        ),
        (_subtract((), "LITERAL", "SELECTED", 3), _absolute((1,))),
        ("subtract", "integer", "+", "3", "end", "absolute", "variable-0"),
        "def solve_three_minus_abs(value):\n    return 3 - abs(value)\n",
        (_absolute(()), _subtract((), "LITERAL", "SELECTED", 3)),
    ),
    PublicSourceEditCanaryTask(
        "list-map-absolute",
        "Return a new list containing the absolute value of every value.",
        _SIG_MAP_ABSOLUTE,
        (
            source_editor.VisibleCase(((-4, 0, 7),), (4, 0, 7)),
            source_editor.VisibleCase(((),), ()),
            source_editor.VisibleCase(((-2, 5),), (2, 5)),
        ),
        _score_task(
            family="t1_factorized_map_absolute",
            signature=_SIG_MAP_ABSOLUTE,
            description="Return abs(value) for every value.",
            oracle=lambda values: [abs(value) for value in values],
            arguments=_LIST_SCORE_ARGS,
        ),
        (_MAP, _absolute((0, 0))),
        ("map", "bind", "absolute", "variable-0", "variable-0"),
        "def solve_map_absolute(values):\n    return [abs(item_0) for item_0 in values]\n",
    ),
    PublicSourceEditCanaryTask(
        "list-map-subtract-two",
        "Return a new list with two subtracted from every value.",
        _SIG_MAP_SUBTRACT_TWO,
        (
            source_editor.VisibleCase(((-3, 0, 4),), (-5, -2, 2)),
            source_editor.VisibleCase(((),), ()),
            source_editor.VisibleCase(((2, -1),), (0, -3)),
        ),
        _score_task(
            family="t1_factorized_map_subtract_two",
            signature=_SIG_MAP_SUBTRACT_TWO,
            description="Subtract two from every value.",
            oracle=lambda values: [value - 2 for value in values],
            arguments=_LIST_SCORE_ARGS,
        ),
        (_MAP, _subtract((0, 0), "SELECTED", "LITERAL", 2)),
        (
            "map",
            "bind",
            "subtract",
            "variable-0",
            "integer",
            "+",
            "2",
            "end",
            "variable-0",
        ),
        "def solve_map_subtract_two(values):\n"
        "    return [item_0 - 2 for item_0 in values]\n",
    ),
)


@dataclass(frozen=True, slots=True)
class SandboxRuntime:
    executable: Path
    identity: dict[str, str]
    worker_source: bytes
    worker_sha256: str


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _action_record(action: tp.FactorAction) -> dict[str, object]:
    return asdict(action)


def _sequence_record(sequence: h2.FactorH2Sequence | None) -> object:
    if sequence is None:
        return None
    return {
        "first": _action_record(sequence.first),
        "second": _action_record(sequence.second),
    }


def _snapshot(value: object) -> object:
    return tuple(_snapshot(item) for item in value) if isinstance(value, (list, tuple)) else value


def _apply_actions(
    incumbent: tp.Program, actions: tuple[tp.FactorAction, ...]
) -> tp.Program:
    program = incumbent
    for action in actions:
        if action not in tp.factor_action_candidates(
            program, literals=PUBLIC_LITERAL_BANK
        ).actions:
            raise RuntimeError(f"declared action is not compiler-admissible: {action}")
        program = tp.apply_factor_action(program, action)
    return program


def _visible_outputs(
    program: tp.Program, cases: tuple[source_editor.VisibleCase, ...]
) -> tuple[object, ...]:
    observed = observation.observe_current_program_root(
        program, tuple(case.arguments for case in cases)
    )
    if not observed.all_supported:
        raise RuntimeError("declared program produced an unsupported visible output")
    return tuple(case.raw_output for case in observed.cases)


def _assert_public_protocol() -> dict[str, bool]:
    """Bind each prompt/trajectory to exact compiler output and visible behavior."""

    if tuple(task.name for task in PUBLIC_TASKS) != (
        "abs-value-minus-three",
        "three-minus-abs-value",
        "list-map-absolute",
        "list-map-subtract-two",
    ):
        raise RuntimeError("public canary task order changed")
    aliases: dict[str, bool] = {}
    for task in PUBLIC_TASKS:
        visible = {json.dumps(case.arguments, sort_keys=True) for case in task.visible_cases}
        scoring = {
            json.dumps(case.args, sort_keys=True)
            for case in (
                *task.score_task.private.regression_cases,
                *task.score_task.private.property_cases,
            )
        }
        if not visible.isdisjoint(scoring):
            raise RuntimeError(f"visible/scoring cases overlap for {task.name}")

        expected = _apply_actions(task.identity, task.expected_actions)
        desired = tuple(_snapshot(case.expected) for case in task.visible_cases)
        if (
            expected.tokens != task.expected_tokens
            or tp.compile(expected).source != task.expected_source
            or _visible_outputs(expected, task.visible_cases) != desired
        ):
            raise RuntimeError(f"declared trajectory disagrees with task {task.name}")

        alias = False
        if task.reverse_equivalent_actions is not None:
            reverse = _apply_actions(task.identity, task.reverse_equivalent_actions)
            alias = reverse == expected
            if not alias:
                raise RuntimeError(f"declared order alias disappeared for {task.name}")
        aliases[task.name] = alias
    return aliases


def _assert_created_child_prerequisite() -> dict[str, object]:
    artifact_bytes = _CREATED_CHILD_V3_PATH.read_bytes()
    artifact_sha256 = sha256(artifact_bytes).hexdigest()
    payload = json.loads(artifact_bytes)
    if (
        artifact_sha256 != _REQUIRED_CREATED_CHILD_ARTIFACT_SHA256
        or payload.get("schema") != "public-typed-factor-horizon2-int-created-child-v3"
        or payload.get("verdict") != _REQUIRED_CREATED_CHILD_VERDICT
        or payload.get("passed_tasks") != 2
    ):
        raise RuntimeError(
            "exact created-child V3 prerequisite did not PASS; source-edit canary stays unspent"
        )
    return {
        "verdict": payload["verdict"],
        "artifact_sha256": artifact_sha256,
        "checkpoint_sha256": payload.get("model_checkpoint_sha256"),
    }


def _prepare_sandbox(python_executable: Path | None) -> SandboxRuntime:
    executable = resolve_python_executable(python_executable)
    identity = python_runtime_identity(executable, "python-3.11")
    worker_path = SRC_ROOT / "humanbrain/simple_swe/_worker.py"
    worker_sha = sha256(worker_path.read_bytes()).hexdigest()
    return SandboxRuntime(
        executable,
        identity,
        load_worker_snapshot(worker_sha),
        worker_sha,
    )


def _execute(source: str, task: GeneratedTask, runtime: SandboxRuntime) -> ExecutionResult:
    return execute_completion(
        source,
        task.public,
        task.private,
        PUBLIC_EXECUTION_LIMITS,
        python_executable=runtime.executable,
        python_executable_sha256=runtime.identity["executable_sha256"],
        worker_source=runtime.worker_source,
        worker_sha256=runtime.worker_sha256,
    )


def _h1_exact_actions(
    decision: h2.FactorizedH2Decision,
) -> tuple[tp.FactorAction, ...]:
    detectors = tuple(
        output_cells.FixedTypedCoincidenceDetector.for_target(target)
        for target in decision.desired
    )
    return tuple(
        action
        for action, prediction in zip(
            decision.candidate_actions, decision.first_predictions, strict=True
        )
        if all(
            detector.detect(predicted).spiked
            for detector, predicted in zip(
                detectors, prediction.root_outputs, strict=True
            )
        )
    )


def _edit_record(result: source_editor.FactorizedSourceEditResult) -> dict[str, object]:
    return {
        "status": result.status,
        "stop_reason": result.stop_reason,
        "edit_count": result.edit_count,
        "tokens": result.program.tokens,
        "source": result.source,
        "selected_sequences": tuple(
            _sequence_record(decision.selected_sequence)
            for decision in result.decisions
        ),
        "checkpoint_immutable": result.checkpoint_audit.immutable,
    }


def _selected_actions(
    result: source_editor.FactorizedSourceEditResult,
) -> tuple[tp.FactorAction, ...]:
    return tuple(
        decision.selected_sequence.first
        for decision in result.decisions
        if decision.status == h2.STATUS_SELECTED
        and decision.selected_sequence is not None
    )


def _allowed_trajectories(
    task: PublicSourceEditCanaryTask,
) -> tuple[tuple[tp.FactorAction, tp.FactorAction], ...]:
    if task.reverse_equivalent_actions is None:
        return (task.expected_actions,)
    return (task.expected_actions, task.reverse_equivalent_actions)


def run_public_task(
    model: h2.FrozenFactorizedHorizon2Int,
    task: PublicSourceEditCanaryTask,
    runtime: SandboxRuntime,
) -> dict[str, object]:
    editor = source_editor.FactorizedSourceEditor(model)
    checkpoint = model.checkpoint_sha256
    incumbent_score = _execute(tp.compile(task.identity).source, task.score_task, runtime)

    bounded = editor.edit(
        task.identity, task.prompt, task.visible_cases, max_edits=1
    )
    h1_exact = _h1_exact_actions(bounded.decisions[0])
    bounded_score = _execute(bounded.source, task.score_task, runtime)

    error: str | None = None
    final: source_editor.FactorizedSourceEditResult | None = None
    final_score: ExecutionResult | None = None
    if incumbent_score.passed:
        error = "identity incumbent passed"
    elif h1_exact:
        error = "a physical H=1 all-visible-case solution exists"
    elif bounded.status != source_editor.STATUS_MAX_EDITS:
        error = f"max_edits=1 control stopped: {bounded.stop_reason}"
    elif not any(
        _selected_actions(bounded) == trajectory[:1]
        for trajectory in _allowed_trajectories(task)
    ):
        error = "max_edits=1 control selected no valid first edit"
    elif bounded_score.passed:
        error = "max_edits=1 control passed"
    else:
        final = editor.edit(
            task.identity, task.prompt, task.visible_cases, max_edits=MAX_EDITS
        )
        if final.status != source_editor.STATUS_DONE:
            error = f"two-edit editor stopped: {final.stop_reason}"
        elif _selected_actions(final) not in _allowed_trajectories(task):
            error = "editor selected no mechanically equivalent trajectory"
        elif final.program.tokens != task.expected_tokens or final.source != task.expected_source:
            error = "terminal typed program differs from the declared target"
        else:
            final_score = source_editor.execute_completed_source(
                final,
                task.score_task,
                PUBLIC_EXECUTION_LIMITS,
                python_executable=runtime.executable,
                python_executable_sha256=runtime.identity["executable_sha256"],
                worker_source=runtime.worker_source,
                worker_sha256=runtime.worker_sha256,
            )
            if not final_score.passed:
                error = f"terminal source failed sandbox: {final_score.detail}"

    immutable = (
        model.live_checkpoint_sha256 == checkpoint
        and bounded.checkpoint_audit.immutable
        and (final is None or final.checkpoint_audit.immutable)
    )
    if error is None and not immutable:
        error = "checkpoint changed"
    return {
        "name": task.name,
        "passed": error is None,
        "error": error,
        "construction_order_alias": task.reverse_equivalent_actions is not None,
        "allowed_trajectories": tuple(
            tuple(_action_record(action) for action in trajectory)
            for trajectory in _allowed_trajectories(task)
        ),
        "h1_exact_actions": tuple(_action_record(action) for action in h1_exact),
        "incumbent_control": asdict(incumbent_score),
        "max_edits_one_control": {
            "edit": _edit_record(bounded),
            "sandbox": asdict(bounded_score),
        },
        "final_edit": None if final is None else _edit_record(final),
        "final_sandbox": None if final_score is None else asdict(final_score),
        "checkpoint_immutable": immutable,
    }


def run_public_matrix(
    model: h2.FrozenFactorizedHorizon2Int,
    runtime: SandboxRuntime,
    prerequisite: dict[str, object],
) -> dict[str, object]:
    started = perf_counter()
    results: list[dict[str, object]] = []
    for task in PUBLIC_TASKS:
        result = run_public_task(model, task, runtime)
        results.append(result)
        if not result["passed"]:
            break
    passed = sum(bool(result["passed"]) for result in results)
    return {
        "schema": SCHEMA,
        "verdict": PASS_VERDICT if passed == 4 else STOP_VERDICT,
        "passed_tasks": passed,
        "attempted_tasks": len(results),
        "total_tasks": 4,
        "task_order": tuple(task.name for task in PUBLIC_TASKS),
        "results": tuple(results),
        "prerequisite": prerequisite,
        "model_checkpoint_sha256": model.checkpoint_sha256,
        "checkpoint_immutable": model.live_checkpoint_sha256 == model.checkpoint_sha256,
        "sandbox": {
            "python": runtime.identity,
            "worker_sha256": runtime.worker_sha256,
            "limits": asdict(PUBLIC_EXECUTION_LIMITS),
        },
        "wall_time_seconds": perf_counter() - started,
        "claim_boundary": (
            "Four source-declared public pre-M0 tasks only; no canonical "
            "Simple-SWE partition, M0, DEV/eval, held-out, protected data, or C1."
        ),
    }


def write_public_artifact(payload: dict[str, object], path: Path) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(_json_bytes(payload))
    except FileExistsError as error:
        raise RuntimeError("public source-edit canary artifact already exists") from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=_ARTIFACT_PATH)
    parser.add_argument("--python-executable", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.json.exists():
        raise RuntimeError("public source-edit canary artifact already exists")
    _assert_public_protocol()
    prerequisite = _assert_created_child_prerequisite()
    runtime = _prepare_sandbox(args.python_executable)
    model = h2.build_public_factorized_horizon2_int(literals=PUBLIC_LITERAL_BANK)
    payload = run_public_matrix(model, runtime, prerequisite)
    write_public_artifact(payload, args.json)
    print(json.dumps({"artifact": str(args.json), "verdict": payload["verdict"]}))
    return 0 if payload["verdict"] == PASS_VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MAX_EDITS",
    "PASS_VERDICT",
    "PUBLIC_EXECUTION_LIMITS",
    "PUBLIC_LITERAL_BANK",
    "PUBLIC_TASKS",
    "SCHEMA",
    "STOP_VERDICT",
    "PublicSourceEditCanaryTask",
    "main",
    "run_public_matrix",
    "run_public_task",
]
