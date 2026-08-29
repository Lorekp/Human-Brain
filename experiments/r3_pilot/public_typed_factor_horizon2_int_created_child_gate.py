"""Public development gate for factorized H=2 Int created-child planning.

The two fixtures are intentionally value-disambiguated.  Each first edit
creates the child site used by the second edit; the planner must predict that
site before actuation, execute only the selected first edit, reobserve, and
replan.  This runner is public-development evidence only and never opens a
protected evaluation surface.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.r3_pilot import public_typed_factor_horizon2_int as h2
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_program_observation as root_observation
from humanbrain import typed_program as tp
from humanbrain.typed_program_types import (
    ProgramParameter,
    ProgramType,
    TypedSignature,
)

SCHEMA = "public-typed-factor-horizon2-int-created-child-v1"
PASS_VERDICT = "PASS_TYPED_FACTOR_HORIZON2_INT_CREATED_CHILD_V1"
STOP_VERDICT = "STOP_TYPED_FACTOR_HORIZON2_INT_CREATED_CHILD_V1"
PUBLIC_LITERAL_BANK = tuple(range(-4, 5))
_ARTIFACT_PATH = Path(__file__).with_suffix(".json")
_DONE = tp.FactorAction((), "DONE", "NONE", "NONE", "NONE", None)


def _two_int_signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "solve",
        (
            ProgramParameter("alpha", integer),
            ProgramParameter("beta", integer),
        ),
        integer,
    )


@dataclass(frozen=True, slots=True)
class PublicCreatedChildTask:
    name: str
    prompt: str
    signature: TypedSignature
    cases: tuple[tuple[int, int], ...]
    incumbent_tokens: tuple[str, ...]
    desired_outputs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PublicCreatedChildTarget:
    name: str
    created_path: tuple[int, ...]
    expected_first: tp.FactorAction
    expected_second: tp.FactorAction
    expected_tokens: tuple[str, ...]


PUBLIC_TASKS = (
    PublicCreatedChildTask(
        name="left-created-child",
        prompt="Subtract beta from the absolute value of alpha.",
        signature=_two_int_signature(),
        cases=((-7, 2), (7, 2), (-7, -3), (7, -3)),
        incumbent_tokens=("variable-1",),
        desired_outputs=(5, 5, 10, 10),
    ),
    PublicCreatedChildTask(
        name="right-created-child",
        prompt="Subtract the absolute value of beta from alpha.",
        signature=_two_int_signature(),
        cases=((2, -7), (2, 7), (-3, -7), (-3, 7)),
        incumbent_tokens=("variable-0",),
        desired_outputs=(-5, -5, -10, -10),
    ),
)

PUBLIC_TARGETS = (
    PublicCreatedChildTarget(
        name="left-created-child",
        created_path=(0,),
        expected_first=tp.FactorAction(
            (), "APPLY2", "subtract", "SELECTED", "SCOPE0", None
        ),
        expected_second=tp.FactorAction(
            (0,), "APPLY1", "absolute", "SELECTED", "NONE", None
        ),
        expected_tokens=(
            "subtract",
            "absolute",
            "variable-1",
            "variable-0",
        ),
    ),
    PublicCreatedChildTarget(
        name="right-created-child",
        created_path=(1,),
        expected_first=tp.FactorAction(
            (), "APPLY2", "subtract", "SCOPE1", "SELECTED", None
        ),
        expected_second=tp.FactorAction(
            (1,), "APPLY1", "absolute", "SELECTED", "NONE", None
        ),
        expected_tokens=(
            "subtract",
            "variable-1",
            "absolute",
            "variable-0",
        ),
    ),
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _lf_sha256(path: Path) -> str:
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _action_record(action: tp.FactorAction) -> dict[str, object]:
    return asdict(action)


def _sequence_record(
    sequence: h2.FactorH2Sequence | None,
) -> dict[str, object] | None:
    if sequence is None:
        return None
    return {
        "first": _action_record(sequence.first),
        "second": _action_record(sequence.second),
    }


def _decision_record(decision: h2.FactorizedH2Decision) -> dict[str, object]:
    tiers = decision.first_tiers
    motor_unique = decision.motor_decision is not None and all(
        race.simultaneous_spike_candidates == 1
        for race in decision.motor_decision.races
    )
    return {
        "status": decision.status,
        "stop_reason": decision.stop_reason,
        "candidate_count": len(decision.candidate_actions),
        "first_admissible_count": (
            0 if tiers is None else len(tiers.admissible_candidate_cells)
        ),
        "first_excluded_count": (
            0 if tiers is None else len(tiers.excluded_candidate_cells)
        ),
        "first_tier_count": 0 if tiers is None else len(tiers.rounds),
        "expanded_first_count": decision.unique_first_expansions,
        "unique_sequence_predictions": decision.unique_sequence_predictions,
        "cumulative_sequence_evaluations": (
            decision.cumulative_sequence_evaluations
        ),
        "terminal_attempt_sequence_counts": tuple(
            attempt.sequence_count for attempt in decision.terminal_attempts
        ),
        "selected_sequence": _sequence_record(decision.selected_sequence),
        "motor_races_unique": motor_unique,
        "checkpoint_immutable": (
            decision.checkpoint_before_sha256
            == decision.checkpoint_after_sha256
        ),
    }


def _actuation_record(actuation: h2.FactorH2Actuation) -> dict[str, object]:
    return {
        "actuator_calls": actuation.actuator_calls,
        "tokens": actuation.program.tokens,
        "structure_parity": actuation.structure_parity,
        "action_site_parity": actuation.action_site_parity,
        "root_output_parity": actuation.root_output_parity,
    }


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
            decision.candidate_actions,
            decision.first_predictions,
            strict=True,
        )
        if all(
            detector.detect(predicted).spiked
            for detector, predicted in zip(
                detectors,
                prediction.root_outputs,
                strict=True,
            )
        )
    )


@dataclass(frozen=True, slots=True)
class PublicCreatedChildTaskResult:
    name: str
    passed: bool
    error: str | None
    initial_paths: tuple[tuple[int, ...], ...]
    created_path: tuple[int, ...]
    created_path_absent_initially: bool
    created_path_present_hypothetically: bool
    hypothetical_second_action_admissible: bool
    h1_exact_actions: tuple[dict[str, object], ...]
    no_h1_all_case_solution: bool
    expected_trajectory: tuple[dict[str, object], ...]
    actual_trajectory: tuple[dict[str, object], ...]
    first_decision: dict[str, object]
    first_actuation: dict[str, object] | None
    second_decision: dict[str, object] | None
    second_actuation: dict[str, object] | None
    done_decision: dict[str, object] | None
    final_tokens: tuple[str, ...]
    expected_tokens: tuple[str, ...]
    final_outputs: tuple[object, ...]
    desired_outputs: tuple[int, ...]
    checkpoint_immutable: bool


def run_public_task(
    model: h2.FrozenFactorizedHorizon2Int,
    task: PublicCreatedChildTask,
) -> PublicCreatedChildTaskResult:
    """Run inference before consulting the separate expected action target."""

    incumbent = tp.Program(task.incumbent_tokens, task.signature)
    initial = model.observe(incumbent, task.cases)
    checkpoint_before = model.live_checkpoint_sha256
    first = model.plan(initial, task.prompt, task.desired_outputs)

    # The exact action/token target is intentionally unavailable to inference.
    target = next(item for item in PUBLIC_TARGETS if item.name == task.name)
    initial_paths = tuple(site.path for site in initial.action_sites)
    h1_actions = _h1_exact_actions(first)
    created_hypothetically = (
        first.selected_first_transition is not None
        and target.created_path
        in tuple(
            site.path
            for site in first.selected_first_transition.predicted.action_sites
        )
    )
    second_admissible = (
        first.selected_first_transition is not None
        and target.expected_second
        in model.world.candidate_actions(
            first.selected_first_transition.predicted
        )
    )
    expected_trajectory = (
        _action_record(target.expected_first),
        _action_record(target.expected_second),
        _action_record(_DONE),
    )
    actual: list[dict[str, object]] = []
    if first.selected_sequence is not None:
        actual.append(_action_record(first.selected_sequence.first))
    first_actuation: h2.FactorH2Actuation | None = None
    second: h2.FactorizedH2Decision | None = None
    second_actuation: h2.FactorH2Actuation | None = None
    done: h2.FactorizedH2Decision | None = None
    final_program = incumbent
    final_outputs: tuple[object, ...] = ()

    def finish(error: str | None) -> PublicCreatedChildTaskResult:
        decisions = tuple(item for item in (first, second, done) if item is not None)
        immutable = (
            checkpoint_before
            == model.checkpoint_sha256
            == model.live_checkpoint_sha256
            and all(
                item.checkpoint_before_sha256
                == item.checkpoint_after_sha256
                == model.checkpoint_sha256
                for item in decisions
            )
        )
        return PublicCreatedChildTaskResult(
            name=task.name,
            passed=error is None,
            error=error,
            initial_paths=initial_paths,
            created_path=target.created_path,
            created_path_absent_initially=target.created_path not in initial_paths,
            created_path_present_hypothetically=created_hypothetically,
            hypothetical_second_action_admissible=second_admissible,
            h1_exact_actions=tuple(_action_record(action) for action in h1_actions),
            no_h1_all_case_solution=not h1_actions,
            expected_trajectory=expected_trajectory,
            actual_trajectory=tuple(actual),
            first_decision=_decision_record(first),
            first_actuation=(
                None if first_actuation is None else _actuation_record(first_actuation)
            ),
            second_decision=None if second is None else _decision_record(second),
            second_actuation=(
                None
                if second_actuation is None
                else _actuation_record(second_actuation)
            ),
            done_decision=None if done is None else _decision_record(done),
            final_tokens=final_program.tokens,
            expected_tokens=target.expected_tokens,
            final_outputs=final_outputs,
            desired_outputs=task.desired_outputs,
            checkpoint_immutable=immutable,
        )

    if h1_actions:
        return finish("a physical H=1 all-case solution exists")
    if first.status != h2.STATUS_SELECTED:
        return finish(f"first H=2 plan stopped: {first.stop_reason}")
    if first.selected_sequence != h2.FactorH2Sequence(
        target.expected_first,
        target.expected_second,
    ):
        return finish("first H=2 plan selected the wrong two-ply sequence")
    if target.created_path in initial_paths:
        return finish("declared created-child path already exists in the incumbent")
    if not created_hypothetically or not second_admissible:
        return finish("selected learned transition did not expose the child edit")

    first_actuation = model.actuate_and_reobserve(incumbent, first)
    final_program = first_actuation.program
    if (
        first_actuation.actuator_calls != 1
        or not first_actuation.structure_parity
        or not first_actuation.action_site_parity
        or not first_actuation.root_output_parity
    ):
        return finish("first actuation lost predicted/reobserved parity")

    second = model.plan(
        first_actuation.observation,
        task.prompt,
        task.desired_outputs,
    )
    if second.selected_sequence is not None:
        actual.append(_action_record(second.selected_sequence.first))
    if second.status != h2.STATUS_SELECTED:
        return finish(f"second H=2 plan stopped: {second.stop_reason}")
    if second.selected_sequence != h2.FactorH2Sequence(
        target.expected_second,
        _DONE,
    ):
        return finish("replanning did not select the child edit followed by DONE")

    second_actuation = model.actuate_and_reobserve(first_actuation.program, second)
    final_program = second_actuation.program
    if (
        second_actuation.actuator_calls != 1
        or not second_actuation.structure_parity
        or not second_actuation.action_site_parity
        or not second_actuation.root_output_parity
    ):
        return finish("second actuation lost predicted/reobserved parity")

    observed_final = root_observation.observe_current_program_root(
        final_program,
        task.cases,
    )
    final_outputs = tuple(case.raw_output for case in observed_final.cases)
    done = model.plan(
        second_actuation.observation,
        task.prompt,
        task.desired_outputs,
    )
    if done.selected_sequence is not None:
        actual.append(_action_record(done.selected_sequence.first))
    if done.status != h2.STATUS_DONE or done.selected_sequence != h2.FactorH2Sequence(
        _DONE,
        _DONE,
    ):
        return finish("terminal replan did not select DONE")
    done_actuation = model.actuate_and_reobserve(final_program, done)
    if done_actuation.actuator_calls != 0 or done_actuation.program != final_program:
        return finish("DONE was not terminal")
    if final_program.tokens != target.expected_tokens:
        return finish("final program tokens differ from the declared target")
    if final_outputs != task.desired_outputs:
        return finish("final real outputs differ from the declared target")
    if tuple(actual) != expected_trajectory:
        return finish("actual edit trajectory differs from the declared target")
    result = finish(None)
    if not result.checkpoint_immutable:
        return finish("frozen checkpoint changed during the task")
    return result


def _exception_result(
    task: PublicCreatedChildTask,
    error: Exception,
    model: h2.FrozenFactorizedHorizon2Int,
) -> PublicCreatedChildTaskResult:
    target = next(item for item in PUBLIC_TARGETS if item.name == task.name)
    return PublicCreatedChildTaskResult(
        name=task.name,
        passed=False,
        error=f"{type(error).__name__}: {error}",
        initial_paths=(),
        created_path=target.created_path,
        created_path_absent_initially=False,
        created_path_present_hypothetically=False,
        hypothetical_second_action_admissible=False,
        h1_exact_actions=(),
        no_h1_all_case_solution=False,
        expected_trajectory=(
            _action_record(target.expected_first),
            _action_record(target.expected_second),
            _action_record(_DONE),
        ),
        actual_trajectory=(),
        first_decision={},
        first_actuation=None,
        second_decision=None,
        second_actuation=None,
        done_decision=None,
        final_tokens=task.incumbent_tokens,
        expected_tokens=target.expected_tokens,
        final_outputs=(),
        desired_outputs=task.desired_outputs,
        checkpoint_immutable=(
            model.live_checkpoint_sha256 == model.checkpoint_sha256
        ),
    )


@dataclass(frozen=True, slots=True)
class PublicCreatedChildMatrixResult:
    schema: str
    verdict: str
    passed_tasks: int
    attempted_tasks: int
    total_tasks: int
    wall_time_seconds: float
    literal_bank: tuple[int, ...]
    model_checkpoint_sha256: str
    world_checkpoint_sha256: str
    runner_source_lf_sha256: str
    tasks: tuple[PublicCreatedChildTaskResult, ...]
    retry_policy: str
    claim_boundary: str


def run_public_matrix() -> PublicCreatedChildMatrixResult:
    """Run the two public tasks once, stopping at the first mechanism failure."""

    started = perf_counter()
    model = h2.build_public_factorized_horizon2_int(literals=PUBLIC_LITERAL_BANK)
    results: list[PublicCreatedChildTaskResult] = []
    for task in PUBLIC_TASKS:
        try:
            result = run_public_task(model, task)
        except Exception as error:
            result = _exception_result(task, error, model)
        results.append(result)
        if not result.passed:
            break
    passed = sum(item.passed for item in results)
    verdict = (
        PASS_VERDICT
        if passed == len(PUBLIC_TASKS) and len(results) == len(PUBLIC_TASKS)
        else STOP_VERDICT
    )
    return PublicCreatedChildMatrixResult(
        schema=SCHEMA,
        verdict=verdict,
        passed_tasks=passed,
        attempted_tasks=len(results),
        total_tasks=len(PUBLIC_TASKS),
        wall_time_seconds=perf_counter() - started,
        literal_bank=PUBLIC_LITERAL_BANK,
        model_checkpoint_sha256=model.checkpoint_sha256,
        world_checkpoint_sha256=model.world.checkpoint_sha256,
        runner_source_lf_sha256=_lf_sha256(Path(__file__)),
        tasks=tuple(results),
        retry_policy=(
            "Public development fixtures are rerunnable only as a newly named "
            "run; this artifact is exclusive and receives no prompt, case, "
            "current, literal-bank, candidate-order, seed, or mechanism tuning."
        ),
        claim_boundary=(
            "Two visible public Int tasks with genuine learned-cell H=2 "
            "lookahead over arbitrary compiler-representable hypothetical "
            "second sites, one real first actuation followed by reobservation "
            "and replanning, and terminal DONE. Not Bool/List, Simple-SWE, "
            "M0, DEV/eval/held-out, protected data, C1, or broad coding evidence."
        ),
    )


def _artifact_bytes(result: PublicCreatedChildMatrixResult) -> bytes:
    return (
        json.dumps(asdict(result), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_public_artifact(
    result: PublicCreatedChildMatrixResult,
    path: Path = _ARTIFACT_PATH,
) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(_artifact_bytes(result))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise FileExistsError(
            f"public created-child H=2 artifact already exists: {path}"
        ) from error


def _validate_predeclared_boundary() -> None:
    if PUBLIC_LITERAL_BANK != tuple(range(-4, 5)):
        raise RuntimeError("public created-child literal bank changed")
    if len(PUBLIC_TASKS) != len(PUBLIC_TARGETS) or len(PUBLIC_TASKS) != 2:
        raise RuntimeError("public created-child gate must contain exactly two tasks")
    task_names = tuple(task.name for task in PUBLIC_TASKS)
    target_names = tuple(target.name for target in PUBLIC_TARGETS)
    if task_names != target_names or len(set(task_names)) != 2:
        raise RuntimeError("public created-child task/target order changed")


def main() -> int:
    if _ARTIFACT_PATH.exists():
        prior = json.loads(_ARTIFACT_PATH.read_bytes())
        print(
            json.dumps(
                {
                    "artifact": str(_ARTIFACT_PATH),
                    "verdict": prior.get("verdict"),
                    "guard": "existing artifact; public matrix not rerun",
                },
                sort_keys=True,
            )
        )
        return 0
    _validate_predeclared_boundary()
    result = run_public_matrix()
    write_public_artifact(result, _ARTIFACT_PATH)
    print(
        json.dumps(
            {
                "artifact": str(_ARTIFACT_PATH),
                "verdict": result.verdict,
                "passed_tasks": result.passed_tasks,
                "attempted_tasks": result.attempted_tasks,
                "total_tasks": result.total_tasks,
                "wall_time_seconds": result.wall_time_seconds,
            },
            sort_keys=True,
        )
    )
    return 0 if result.verdict == PASS_VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PASS_VERDICT",
    "PUBLIC_LITERAL_BANK",
    "PUBLIC_TARGETS",
    "PUBLIC_TASKS",
    "SCHEMA",
    "STOP_VERDICT",
    "PublicCreatedChildMatrixResult",
    "PublicCreatedChildTarget",
    "PublicCreatedChildTask",
    "PublicCreatedChildTaskResult",
    "run_public_matrix",
    "run_public_task",
    "write_public_artifact",
]
