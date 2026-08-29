"""Prepared two-task public recovery cohort for factorized source editing.

The cohort is fresh: it does not replay either spent canary.  Both tasks start
from typed identity, expose only source-declared public diagnostics, and score
disjoint source-declared cases through the existing sandbox.  Importing this
module and running its static protocol checks do not build the cell model or
write an artifact.
"""

# ruff: noqa: E402 -- direct-script execution installs repo import roots first.

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.r3_pilot import public_factorized_source_edit_canary as canary_v1
from experiments.r3_pilot import public_factorized_source_edit_canary_v2 as canary_v2
from experiments.r3_pilot import public_factorized_source_editor as source_editor
from experiments.r3_pilot import public_typed_factor_horizon2_int as h2
from experiments.r3_pilot import public_typed_prompt_factors as prompt_factors
from humanbrain import typed_program as tp

SCHEMA = "public-factorized-source-edit-recovery-v1"
PASS_VERDICT = "PASS_PUBLIC_FACTORIZED_SOURCE_EDIT_RECOVERY_V1"
STOP_VERDICT = "STOP_PUBLIC_FACTORIZED_SOURCE_EDIT_RECOVERY_V1"
PUBLIC_LITERAL_BANK = canary_v1.PUBLIC_LITERAL_BANK
PUBLIC_EXECUTION_LIMITS = canary_v1.PUBLIC_EXECUTION_LIMITS
MAX_EDITS = canary_v1.MAX_EDITS
_ARTIFACT_PATH = Path(__file__).with_suffix(".json")
_V2_ARTIFACT_PATH = Path(__file__).with_name(
    "public_factorized_source_edit_canary_v2.json"
)
_V2_ARTIFACT_SHA256 = (
    "a87e0509e101ba5cfd8d5610f5b7e66b7fb87855995829837f89eb95af2a9026"
)

_MAP = tp.FactorAction((), "MAP_BIND", "map", "BOUND0", "SELECTED", None)


def _signature(name: str, entry_point: str, *, is_list: bool) -> object:
    integer = canary_v1.ProgramType.Int()
    value_type = canary_v1.ProgramType.List(integer) if is_list else integer
    return canary_v1.TypedSignature(
        entry_point,
        (canary_v1.ProgramParameter(name, value_type),),
        value_type,
    )


_SCALAR_SIGNATURE = _signature("datum", "solve_four_minus_abs", is_list=False)
_LIST_SIGNATURE = _signature("items", "solve_map_add_four", is_list=True)
_SCALAR_SCORE_ARGS = ((0,), (-6,), (1,), (12,), (-21,), (-3,), (7,), (34,))
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

_SCALAR_ACTIONS = (
    tp.FactorAction((), "APPLY2", "subtract", "LITERAL", "SELECTED", 4),
    tp.FactorAction((1,), "APPLY1", "absolute", "SELECTED", "NONE", None),
)
_SCALAR_REVERSE = (
    tp.FactorAction((), "APPLY1", "absolute", "SELECTED", "NONE", None),
    tp.FactorAction((), "APPLY2", "subtract", "LITERAL", "SELECTED", 4),
)
_LIST_ACTIONS = (
    _MAP,
    tp.FactorAction(
        (0, 0), "APPLY2", "add", "SELECTED", "LITERAL", 4
    ),
)

PUBLIC_TASKS = (
    canary_v1.PublicSourceEditCanaryTask(
        "four-minus-abs-datum",
        "Return 4 minus the absolute value of datum.",
        _SCALAR_SIGNATURE,
        tuple(
            source_editor.VisibleCase((value,), 4 - abs(value))
            for value in (-11, -2, 5, 13)
        ),
        canary_v1._score_task(
            family="t0_factorized_four_minus_abs_recovery",
            signature=_SCALAR_SIGNATURE,
            description="Return four minus abs(datum).",
            oracle=lambda datum: 4 - abs(datum),
            arguments=_SCALAR_SCORE_ARGS,
        ),
        _SCALAR_ACTIONS,
        ("subtract", "integer", "+", "4", "end", "absolute", "variable-0"),
        "def solve_four_minus_abs(datum):\n    return 4 - abs(datum)\n",
        _SCALAR_REVERSE,
    ),
    canary_v1.PublicSourceEditCanaryTask(
        "list-map-add-four",
        "Return a new list containing every item plus 4.",
        _LIST_SIGNATURE,
        (
            source_editor.VisibleCase(((-5, 0, 8),), (-1, 4, 12)),
            source_editor.VisibleCase(((),), ()),
            source_editor.VisibleCase(((3, -2),), (7, 2)),
        ),
        canary_v1._score_task(
            family="t1_factorized_map_add_four_recovery",
            signature=_LIST_SIGNATURE,
            description="Add four to every item.",
            oracle=lambda items: [item + 4 for item in items],
            arguments=_LIST_SCORE_ARGS,
        ),
        _LIST_ACTIONS,
        (
            "map",
            "bind",
            "add",
            "variable-0",
            "integer",
            "+",
            "4",
            "end",
            "variable-0",
        ),
        "def solve_map_add_four(items):\n"
        "    return [item_0 + 4 for item_0 in items]\n",
    ),
)


def _assert_recovery_protocol() -> None:
    if tuple(task.name for task in PUBLIC_TASKS) != (
        "four-minus-abs-datum",
        "list-map-add-four",
    ):
        raise RuntimeError("public recovery task order changed")

    for task in PUBLIC_TASKS:
        expected = canary_v1._apply_actions(task.identity, task.expected_actions)
        desired = tuple(
            canary_v1._snapshot(case.expected) for case in task.visible_cases
        )
        if (
            expected.tokens != task.expected_tokens
            or tp.compile(expected).source != task.expected_source
            or canary_v1._visible_outputs(expected, task.visible_cases) != desired
        ):
            raise RuntimeError(f"recovery declaration disagrees with {task.name}")
        visible = {
            json.dumps(case.arguments, sort_keys=True)
            for case in task.visible_cases
        }
        scoring = {
            json.dumps(case.args, sort_keys=True)
            for case in (
                *task.score_task.private.regression_cases,
                *task.score_task.private.property_cases,
            )
        }
        if not visible.isdisjoint(scoring):
            raise RuntimeError(f"visible/scoring cases overlap for {task.name}")

        exact_h1: list[tp.FactorAction] = []
        for action in tp.factor_action_candidates(
            task.identity, literals=PUBLIC_LITERAL_BANK
        ).actions:
            program = (
                task.identity
                if action.schema == "DONE"
                else tp.apply_factor_action(task.identity, action)
            )
            try:
                outputs = canary_v1._visible_outputs(program, task.visible_cases)
            except RuntimeError:
                continue
            if outputs == desired:
                exact_h1.append(action)
        if exact_h1:
            raise RuntimeError(f"recovery task has an H1 solution: {task.name}")

    scalar_prompt = prompt_factors.observe_prompt_compositions(
        PUBLIC_TASKS[0].prompt, PUBLIC_TASKS[0].signature
    )
    list_prompt = prompt_factors.observe_prompt_compositions(
        PUBLIC_TASKS[1].prompt, PUBLIC_TASKS[1].signature
    )
    if (
        not scalar_prompt.supported
        or scalar_prompt.pattern is None
        or scalar_prompt.pattern.container_schema is not None
        or not list_prompt.supported
        or list_prompt.pattern is None
        or list_prompt.pattern.container_schema != "MAP_BIND"
    ):
        raise RuntimeError("recovery prompts are outside the generic composition seam")


def _assert_v2_boundary() -> dict[str, object]:
    artifact_bytes = _V2_ARTIFACT_PATH.read_bytes()
    artifact_sha256 = sha256(artifact_bytes).hexdigest()
    payload = json.loads(artifact_bytes)
    if (
        artifact_sha256 != _V2_ARTIFACT_SHA256
        or payload.get("verdict") != canary_v2.STOP_VERDICT
        or payload.get("passed_tasks") != 3
        or payload.get("attempted_tasks") != 4
    ):
        raise RuntimeError("exact public source-edit V2 boundary is absent")
    return {
        "verdict": payload["verdict"],
        "artifact_sha256": artifact_sha256,
        "passed_tasks": payload["passed_tasks"],
        "failed_task": payload["results"][-1]["name"],
    }


def run_recovery_matrix(
    model: h2.FrozenFactorizedHorizon2Int,
    runtime: canary_v1.SandboxRuntime,
    prerequisite: dict[str, object],
) -> dict[str, object]:
    started = perf_counter()
    results: list[dict[str, object]] = []
    for task in PUBLIC_TASKS:
        result = canary_v2.run_public_task(model, task, runtime)
        results.append(result)
        if not result["passed"]:
            break
    passed = sum(bool(result["passed"]) for result in results)
    return {
        "schema": SCHEMA,
        "verdict": PASS_VERDICT if passed == 2 else STOP_VERDICT,
        "passed_tasks": passed,
        "attempted_tasks": len(results),
        "total_tasks": 2,
        "task_order": tuple(task.name for task in PUBLIC_TASKS),
        "results": tuple(results),
        "prerequisite": prerequisite,
        "model_checkpoint_sha256": model.checkpoint_sha256,
        "checkpoint_immutable": model.live_checkpoint_sha256
        == model.checkpoint_sha256,
        "sandbox": {
            "python": runtime.identity,
            "worker_sha256": runtime.worker_sha256,
            "limits": asdict(PUBLIC_EXECUTION_LIMITS),
        },
        "wall_time_seconds": perf_counter() - started,
        "claim_boundary": (
            "Two fresh source-declared public pre-M0 recovery tasks only; no "
            "V1/V2 replay, canonical Simple-SWE, M0, DEV/eval, held-out, "
            "protected data, or C1."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=_ARTIFACT_PATH)
    parser.add_argument("--python-executable", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.json.exists():
        raise RuntimeError("public source-edit recovery artifact already exists")
    _assert_recovery_protocol()
    prerequisite = _assert_v2_boundary()
    runtime = canary_v1._prepare_sandbox(args.python_executable)
    model = h2.build_public_factorized_horizon2_int(literals=PUBLIC_LITERAL_BANK)
    payload = run_recovery_matrix(model, runtime, prerequisite)
    canary_v1.write_public_artifact(payload, args.json)
    print(json.dumps({"artifact": str(args.json), "verdict": payload["verdict"]}))
    return 0 if payload["verdict"] == PASS_VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())
