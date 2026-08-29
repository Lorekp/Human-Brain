"""One-shot V2 of the four-task public source-editing canary.

V2 keeps V1's exact public tasks, cases, literal bank, and created-child V3
prerequisite.  Its sole change is target-independent edit accounting: an action
counts as applied only when its checkpoint-audit step records a real actuation.
"""

# ruff: noqa: E402 -- direct-script execution installs repo import roots first.

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.r3_pilot import public_factorized_source_edit_canary as v1
from experiments.r3_pilot import public_factorized_source_editor as source_editor
from experiments.r3_pilot import public_typed_factor_horizon2_int as h2

SCHEMA = "public-factorized-source-edit-canary-v2"
PASS_VERDICT = "PASS_PUBLIC_FACTORIZED_SOURCE_EDIT_CANARY_V2"
STOP_VERDICT = "STOP_PUBLIC_FACTORIZED_SOURCE_EDIT_CANARY_V2"
PUBLIC_LITERAL_BANK = v1.PUBLIC_LITERAL_BANK
PUBLIC_EXECUTION_LIMITS = v1.PUBLIC_EXECUTION_LIMITS
PUBLIC_TASKS = v1.PUBLIC_TASKS
MAX_EDITS = v1.MAX_EDITS
PublicSourceEditCanaryTask = v1.PublicSourceEditCanaryTask
SandboxRuntime = v1.SandboxRuntime
_ARTIFACT_PATH = Path(__file__).with_suffix(".json")


def _applied_actions(
    result: source_editor.FactorizedSourceEditResult,
) -> tuple[object, ...]:
    """Return first actions whose decisions reached a real actuator call."""

    applied_steps = tuple(
        step
        for step in result.checkpoint_audit.steps
        if step.live_after_actuation_sha256 is not None
    )
    if len(applied_steps) != result.edit_count or result.edit_count != len(
        result.actuations
    ):
        raise RuntimeError("source-editor actuation ledger is inconsistent")

    actions = []
    for step in applied_steps:
        if not 0 <= step.decision_index < len(result.decisions):
            raise RuntimeError("actuation step points outside the decision ledger")
        decision = result.decisions[step.decision_index]
        if decision.status != h2.STATUS_SELECTED or decision.selected_sequence is None:
            raise RuntimeError("actuation step does not point to a selected decision")
        actions.append(decision.selected_sequence.first)
    return tuple(actions)


def _edit_record(
    result: source_editor.FactorizedSourceEditResult,
) -> dict[str, object]:
    return {
        **v1._edit_record(result),
        "applied_actions": tuple(
            v1._action_record(action) for action in _applied_actions(result)
        ),
    }


def run_public_task(
    model: h2.FrozenFactorizedHorizon2Int,
    task: PublicSourceEditCanaryTask,
    runtime: SandboxRuntime,
) -> dict[str, object]:
    editor = source_editor.FactorizedSourceEditor(model)
    checkpoint = model.checkpoint_sha256
    incumbent_score = v1._execute(
        v1.tp.compile(task.identity).source, task.score_task, runtime
    )

    bounded = editor.edit(
        task.identity, task.prompt, task.visible_cases, max_edits=1
    )
    bounded_applied = _applied_actions(bounded)
    h1_exact = v1._h1_exact_actions(bounded.decisions[0])
    bounded_score = v1._execute(bounded.source, task.score_task, runtime)

    error: str | None = None
    final: source_editor.FactorizedSourceEditResult | None = None
    final_score = None
    if incumbent_score.passed:
        error = "identity incumbent passed"
    elif h1_exact:
        error = "a physical H=1 all-visible-case solution exists"
    elif bounded.status != source_editor.STATUS_MAX_EDITS:
        error = f"max_edits=1 control stopped: {bounded.stop_reason}"
    elif not any(
        bounded_applied == trajectory[:1]
        for trajectory in v1._allowed_trajectories(task)
    ):
        error = "max_edits=1 control applied no valid first edit"
    elif bounded_score.passed:
        error = "max_edits=1 control passed"
    else:
        final = editor.edit(
            task.identity, task.prompt, task.visible_cases, max_edits=MAX_EDITS
        )
        final_applied = _applied_actions(final)
        if final.status != source_editor.STATUS_DONE:
            error = f"two-edit editor stopped: {final.stop_reason}"
        elif final_applied not in v1._allowed_trajectories(task):
            error = "editor applied no mechanically equivalent trajectory"
        elif (
            final.program.tokens != task.expected_tokens
            or final.source != task.expected_source
        ):
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
            tuple(v1._action_record(action) for action in trajectory)
            for trajectory in v1._allowed_trajectories(task)
        ),
        "h1_exact_actions": tuple(v1._action_record(action) for action in h1_exact),
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=_ARTIFACT_PATH)
    parser.add_argument("--python-executable", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.json.exists():
        raise RuntimeError("public source-edit canary V2 artifact already exists")
    v1._assert_public_protocol()
    prerequisite = v1._assert_created_child_prerequisite()
    runtime = v1._prepare_sandbox(args.python_executable)
    model = h2.build_public_factorized_horizon2_int(literals=PUBLIC_LITERAL_BANK)
    payload = run_public_matrix(model, runtime, prerequisite)
    v1.write_public_artifact(payload, args.json)
    print(json.dumps({"artifact": str(args.json), "verdict": payload["verdict"]}))
    return 0 if payload["verdict"] == PASS_VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())
