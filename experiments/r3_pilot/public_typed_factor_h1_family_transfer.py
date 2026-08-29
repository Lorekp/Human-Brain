"""One-shot public family-transfer gate for the frozen hierarchical H=1 planner.

This module adds no inference mechanism.  It authenticates the previously
banked hierarchical planner evidence, freezes four new public fixtures, and
calls :meth:`FrozenHierarchicalPromptH1.plan` once per fixture.  Exact program
targets and expected physical tiers are consulted only after each inference
call returns.
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
for search_path in (REPO_ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.r3_pilot import (  # noqa: E402
    public_typed_factor_h1_prompt_hierarchical as hierarchical,
)
from humanbrain import typed_program as tp  # noqa: E402
from humanbrain.typed_program_types import (  # noqa: E402
    ProgramParameter,
    ProgramType,
    TypedSignature,
)


SCHEMA = "public-typed-factor-h1-family-transfer-v1"
PASS_VERDICT = "PASS_TYPED_FACTOR_H1_FAMILY_TRANSFER_V1"
STOP_VERDICT = "STOP_TYPED_FACTOR_H1_FAMILY_TRANSFER_V1"

PRIOR_ARTIFACT_LF_SHA256 = (
    "4ee553a90a9c54baaa92d65d731f7b8daec853dad0cad212bbd45c597c8d719a"
)
PRIOR_SPEND_MARKER_LF_SHA256 = (
    "8c2ba4791f4ad9dcb7feb05a0773a5ce9d63215529cb3bae58b23e13d91969ad"
)
PRIOR_PLANNER_SOURCE_LF_SHA256 = (
    "591c1759ea81e9774ae96c80bf5bac587422320fe3d9a103e7631f615d163ed8"
)
PRIOR_MODEL_CHECKPOINT_SHA256 = (
    "de8733825b824d734a22c110627f131aa60fd390e81c01aff1b374b3c1af7176"
)

PUBLIC_LITERAL_BANK = tuple(range(-4, 5))
PUBLIC_TWO_INT_CASES = ((9, -4), (-6, 3), (2, 2), (0, -1))
PUBLIC_BOOL_CASES = ((-5,), (-2,), (0,), (4,))

_ARTIFACT_PATH = Path(__file__).with_suffix(".json")
_SPEND_MARKER_PATH = Path(__file__).with_suffix(".spend.json")
_PRIOR_ARTIFACT_PATH = hierarchical._ARTIFACT_PATH
_PRIOR_SPEND_MARKER_PATH = hierarchical._SPEND_MARKER_PATH
_PRIOR_PLANNER_SOURCE_PATH = Path(hierarchical.__file__)


def _int_signature_two() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "solve",
        (
            ProgramParameter("alpha", integer),
            ProgramParameter("beta", integer),
        ),
        integer,
    )


def _bool_signature_one() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "solve",
        (ProgramParameter("datum", integer),),
        ProgramType.Bool(),
    )


@dataclass(frozen=True, slots=True)
class PublicFamilyTransferTask:
    name: str
    prompt: str
    signature: TypedSignature
    cases: tuple[tuple[object, ...], ...]
    incumbent_tokens: tuple[str, ...]
    desired_outputs: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class PublicFamilyTransferTarget:
    name: str
    expected_tokens: tuple[str, ...]
    expected_leading_actions: tuple[tp.FactorAction, ...]
    expected_selected_action: tp.FactorAction


def _apply2(
    operator: str,
    left_source: str,
    right_source: str,
    literal: int | None = None,
) -> tp.FactorAction:
    return tp.FactorAction(
        path=(),
        schema="APPLY2",
        operator=operator,
        left_source=left_source,
        right_source=right_source,
        literal=literal,
    )


PUBLIC_TASKS = (
    PublicFamilyTransferTask(
        name="two-int-alpha-minus-beta",
        prompt="Subtract beta from alpha.",
        signature=_int_signature_two(),
        cases=PUBLIC_TWO_INT_CASES,
        incumbent_tokens=("add", "variable-0", "integer", "+", "0", "end"),
        desired_outputs=(13, -9, 0, 1),
    ),
    PublicFamilyTransferTask(
        name="two-int-beta-minus-alpha",
        prompt="Subtract alpha from beta.",
        signature=_int_signature_two(),
        cases=PUBLIC_TWO_INT_CASES,
        incumbent_tokens=("add", "variable-0", "integer", "+", "0", "end"),
        desired_outputs=(-13, 9, 0, -1),
    ),
    PublicFamilyTransferTask(
        name="bool-datum-less-than-neg2",
        prompt="Return whether datum < -2.",
        signature=_bool_signature_one(),
        cases=PUBLIC_BOOL_CASES,
        incumbent_tokens=(
            "less-than",
            "absolute",
            "variable-0",
            "integer",
            "+",
            "4",
            "end",
        ),
        desired_outputs=(True, False, False, False),
    ),
    PublicFamilyTransferTask(
        name="bool-neg2-less-than-datum",
        prompt="Return whether -2 < datum.",
        signature=_bool_signature_one(),
        cases=PUBLIC_BOOL_CASES,
        incumbent_tokens=(
            "less-than",
            "absolute",
            "variable-0",
            "integer",
            "+",
            "4",
            "end",
        ),
        desired_outputs=(False, False, True, True),
    ),
)

PUBLIC_TARGETS = (
    PublicFamilyTransferTarget(
        name="two-int-alpha-minus-beta",
        expected_tokens=("subtract", "variable-1", "variable-0"),
        expected_leading_actions=(
            _apply2("subtract", "SCOPE1", "SELECTED"),
            _apply2("subtract", "SCOPE1", "SCOPE0"),
        ),
        expected_selected_action=_apply2("subtract", "SCOPE1", "SCOPE0"),
    ),
    PublicFamilyTransferTarget(
        name="two-int-beta-minus-alpha",
        expected_tokens=("subtract", "variable-0", "variable-1"),
        expected_leading_actions=(
            _apply2("subtract", "SELECTED", "SCOPE1"),
            _apply2("subtract", "SCOPE0", "SCOPE1"),
        ),
        expected_selected_action=_apply2("subtract", "SCOPE0", "SCOPE1"),
    ),
    PublicFamilyTransferTarget(
        name="bool-datum-less-than-neg2",
        expected_tokens=(
            "less-than",
            "variable-0",
            "integer",
            "-",
            "2",
            "end",
        ),
        expected_leading_actions=tuple(
            _apply2("less-than", "SCOPE0", "LITERAL", literal)
            for literal in (-4, -3, -2)
        ),
        expected_selected_action=_apply2(
            "less-than", "SCOPE0", "LITERAL", -2
        ),
    ),
    PublicFamilyTransferTarget(
        name="bool-neg2-less-than-datum",
        expected_tokens=(
            "less-than",
            "integer",
            "-",
            "2",
            "end",
            "variable-0",
        ),
        expected_leading_actions=tuple(
            _apply2("less-than", "LITERAL", "SCOPE0", literal)
            for literal in (-2, -1)
        ),
        expected_selected_action=_apply2(
            "less-than", "LITERAL", "SCOPE0", -2
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


def _lf_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def _lf_sha256(path: Path) -> str:
    return sha256(_lf_bytes(path)).hexdigest()


def _action_record(action: tp.FactorAction) -> dict[str, object]:
    return {
        "path": action.path,
        "schema": action.schema,
        "operator": action.operator,
        "left_source": action.left_source,
        "right_source": action.right_source,
        "literal": action.literal,
    }


@dataclass(frozen=True, slots=True)
class PriorPassEvidence:
    schema: str
    verdict: str
    artifact_lf_sha256: str
    spend_marker_lf_sha256: str
    planner_source_lf_sha256: str
    model_checkpoint_sha256: str
    commitment_sha256: str


def authenticate_prior_pass() -> PriorPassEvidence:
    """Authenticate the exact committed hierarchy and its sole PASS run."""

    artifact_sha = _lf_sha256(_PRIOR_ARTIFACT_PATH)
    marker_sha = _lf_sha256(_PRIOR_SPEND_MARKER_PATH)
    source_sha = _lf_sha256(_PRIOR_PLANNER_SOURCE_PATH)
    if artifact_sha != PRIOR_ARTIFACT_LF_SHA256:
        raise RuntimeError("prior hierarchical PASS artifact SHA-256 changed")
    if marker_sha != PRIOR_SPEND_MARKER_LF_SHA256:
        raise RuntimeError("prior hierarchical spend marker SHA-256 changed")
    if source_sha != PRIOR_PLANNER_SOURCE_LF_SHA256:
        raise RuntimeError("prior hierarchical planner source SHA-256 changed")

    payload = json.loads(_PRIOR_ARTIFACT_PATH.read_text(encoding="utf-8"))
    marker = json.loads(_PRIOR_SPEND_MARKER_PATH.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != hierarchical.SCHEMA
        or payload.get("verdict") != hierarchical.PASS_VERDICT
        or payload.get("passed_tasks") != payload.get("total_tasks")
        or payload.get("total_tasks") != 4
        or payload.get("model_checkpoint_sha256")
        != PRIOR_MODEL_CHECKPOINT_SHA256
        or marker.get("commitment_sha256") != payload.get("commitment_sha256")
    ):
        raise RuntimeError("prior hierarchical evidence is not the pinned 4/4 PASS")
    return PriorPassEvidence(
        schema=payload["schema"],
        verdict=payload["verdict"],
        artifact_lf_sha256=artifact_sha,
        spend_marker_lf_sha256=marker_sha,
        planner_source_lf_sha256=source_sha,
        model_checkpoint_sha256=payload["model_checkpoint_sha256"],
        commitment_sha256=payload["commitment_sha256"],
    )


@dataclass(frozen=True, slots=True)
class PublicFamilyTransferTaskResult:
    name: str
    prompt: str
    passed: bool
    status: str
    error: str | None
    candidate_count: int
    root_successor_action_count: int
    nonroot_unknown_count: int
    leading_value_tier_cells: tuple[int, ...]
    leading_value_tier_actions: tuple[dict[str, object], ...]
    expected_leading_value_tier_actions: tuple[dict[str, object], ...]
    leading_value_tier_current_pa: tuple[float, ...]
    value_stop_in_leading_tier: bool
    prompt_status: str | None
    prompt_pattern: dict[str, object] | None
    prompt_matching_tier_cells: tuple[int, ...]
    selected_candidate_cell: int | None
    selected_action: dict[str, object] | None
    expected_selected_action: dict[str, object]
    final_tokens: tuple[str, ...]
    expected_tokens: tuple[str, ...]
    final_outputs: tuple[object, ...]
    desired_outputs: tuple[object, ...]
    structure_prediction_calls: int
    actuator_calls: int
    operator_races_unique: bool
    prompt_race_unique: bool
    motor_races_unique: bool
    structure_races_unique: bool
    checkpoint_immutable: bool


def _prompt_pattern_record(
    decision: hierarchical.HierarchicalPromptH1Decision,
) -> dict[str, object] | None:
    if decision.prompt_decision is None:
        return None
    pattern = decision.prompt_decision.observation.pattern
    if pattern is None:
        return None
    return {
        "operator": pattern.operator,
        "left_source": pattern.left_source,
        "right_source": pattern.right_source,
        "literal": pattern.literal,
    }


def run_public_task(
    model: hierarchical.FrozenHierarchicalPromptH1,
    task: PublicFamilyTransferTask,
) -> PublicFamilyTransferTaskResult:
    """Run frozen inference first, then consult the separate exact target."""

    incumbent = tp.Program(task.incumbent_tokens, task.signature)
    decision = model.plan(
        incumbent,
        task.prompt,
        task.cases,
        task.desired_outputs,
    )

    target = next(item for item in PUBLIC_TARGETS if item.name == task.name)
    final_program = decision.selected_program or incumbent
    final_observation = decision.final_observation or decision.current_observation
    final_outputs = tuple(case.raw_output for case in final_observation.cases)
    checkpoint_immutable = (
        decision.checkpoint_before_sha256
        == decision.checkpoint_after_sha256
        == model.checkpoint_sha256
        == model.live_checkpoint_sha256
    )
    operator_races_unique = all(
        race.simultaneous_spike_candidates == 1
        for candidate in decision.candidate_predictions
        for prediction in candidate.operator_predictions
        for race in prediction.races
    )
    motor_races_unique = (
        decision.motor_decision is not None
        and all(
            race.simultaneous_spike_candidates == 1
            for race in decision.motor_decision.races
        )
    )
    structure_races = (
        ()
        if decision.structure_prediction is None
        else (
            decision.structure_prediction.transition_race,
            *tuple(item.race for item in decision.structure_prediction.rule_races),
            *tuple(item.race for item in decision.structure_prediction.field_races),
        )
    )
    structure_races_unique = bool(structure_races) and all(
        race.simultaneous_spike_candidates == 1 for race in structure_races
    )
    prompt_race_unique = (
        decision.prompt_action_race is not None
        and decision.prompt_action_race.simultaneous_spike_candidates == 1
    )
    value_stop_in_leading_tier = (
        decision.value_tier is not None
        and bool(decision.value_tier.rounds)
        and decision.value_tier.stop_cell in decision.value_tier.rounds[0].tier_cells
    )
    passed = (
        decision.status == hierarchical.STATUS_SELECTED
        and decision.leading_actions == target.expected_leading_actions
        and decision.selected_action == target.expected_selected_action
        and final_program.tokens == target.expected_tokens
        and final_outputs == task.desired_outputs
        and not value_stop_in_leading_tier
        and decision.prompt_decision is not None
        and decision.prompt_decision.status == "PROJECTED"
        and len(decision.prompt_decision.matching_action_cells) == 1
        and decision.structure_prediction_calls == 1
        and decision.actuator_calls == 1
        and operator_races_unique
        and prompt_race_unique
        and motor_races_unique
        and structure_races_unique
        and checkpoint_immutable
    )
    leading_currents = tuple(
        decision.value_currents_pa[cell]
        for cell in decision.leading_candidate_cells
    )
    return PublicFamilyTransferTaskResult(
        name=task.name,
        prompt=task.prompt,
        passed=passed,
        status=decision.status,
        error=None if passed else (decision.stop_reason or "post-inference mismatch"),
        candidate_count=len(decision.candidate_bank.actions),
        root_successor_action_count=decision.root_successor_action_count,
        nonroot_unknown_count=decision.nonroot_unknown_count,
        leading_value_tier_cells=decision.leading_candidate_cells,
        leading_value_tier_actions=tuple(
            _action_record(action) for action in decision.leading_actions
        ),
        expected_leading_value_tier_actions=tuple(
            _action_record(action) for action in target.expected_leading_actions
        ),
        leading_value_tier_current_pa=leading_currents,
        value_stop_in_leading_tier=value_stop_in_leading_tier,
        prompt_status=(
            None
            if decision.prompt_decision is None
            else decision.prompt_decision.status
        ),
        prompt_pattern=_prompt_pattern_record(decision),
        prompt_matching_tier_cells=(
            ()
            if decision.prompt_decision is None
            else decision.prompt_decision.matching_action_cells
        ),
        selected_candidate_cell=decision.selected_candidate_cell,
        selected_action=(
            None
            if decision.selected_action is None
            else _action_record(decision.selected_action)
        ),
        expected_selected_action=_action_record(target.expected_selected_action),
        final_tokens=final_program.tokens,
        expected_tokens=target.expected_tokens,
        final_outputs=final_outputs,
        desired_outputs=task.desired_outputs,
        structure_prediction_calls=decision.structure_prediction_calls,
        actuator_calls=decision.actuator_calls,
        operator_races_unique=operator_races_unique,
        prompt_race_unique=prompt_race_unique,
        motor_races_unique=motor_races_unique,
        structure_races_unique=structure_races_unique,
        checkpoint_immutable=checkpoint_immutable,
    )


@dataclass(frozen=True, slots=True)
class PublicFamilyTransferMatrixResult:
    schema: str
    verdict: str
    passed_tasks: int
    total_tasks: int
    wall_time_seconds: float
    literal_bank: tuple[int, ...]
    prior_pass: PriorPassEvidence
    model_checkpoint_sha256: str
    tasks: tuple[PublicFamilyTransferTaskResult, ...]
    retry_policy: str
    claim_boundary: str


def run_public_matrix(
    prior_pass: PriorPassEvidence,
) -> PublicFamilyTransferMatrixResult:
    """Spend the four frozen transfer fixtures once as one cohort."""

    started = perf_counter()
    model = hierarchical.build_public_hierarchical_prompt_h1()
    if model.checkpoint_sha256 != prior_pass.model_checkpoint_sha256:
        raise RuntimeError("live frozen model differs from authenticated prior PASS")
    results = tuple(run_public_task(model, task) for task in PUBLIC_TASKS)
    passed = sum(task.passed for task in results)
    return PublicFamilyTransferMatrixResult(
        schema=SCHEMA,
        verdict=PASS_VERDICT if passed == len(results) else STOP_VERDICT,
        passed_tasks=passed,
        total_tasks=len(results),
        wall_time_seconds=perf_counter() - started,
        literal_bank=PUBLIC_LITERAL_BANK,
        prior_pass=prior_pass,
        model_checkpoint_sha256=model.checkpoint_sha256,
        tasks=results,
        retry_policy=(
            "One four-task visible public-development cohort; no retry or "
            "task, prompt, case, incumbent, literal-bank, tier, current, "
            "mechanism, or ordering tuning."
        ),
        claim_boundary=(
            "Visible public Int/Bool H=1 family-factor transfer across renamed "
            "bindings, paraphrased operator syntax, signed literal -2, changed "
            "cases, and unseen incumbent structures. Reuses the frozen "
            "target-free operator/structure successors and physical "
            "value/prompt/motor hierarchy. Not List, H>1, Simple-SWE, M0, "
            "DEV/eval/held-out, protected, C1, or coding capability."
        ),
    )


def _task_record(task: PublicFamilyTransferTask) -> dict[str, object]:
    return {
        "name": task.name,
        "prompt": task.prompt,
        "signature": task.signature.to_record(),
        "cases": task.cases,
        "incumbent_tokens": task.incumbent_tokens,
        "desired_outputs": task.desired_outputs,
    }


def _target_record(target: PublicFamilyTransferTarget) -> dict[str, object]:
    return {
        "name": target.name,
        "expected_tokens": target.expected_tokens,
        "expected_leading_actions": tuple(
            _action_record(action) for action in target.expected_leading_actions
        ),
        "expected_selected_action": _action_record(target.expected_selected_action),
    }


def _fresh_commitment(prior_pass: PriorPassEvidence) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "literal_bank": PUBLIC_LITERAL_BANK,
        "tasks": tuple(_task_record(task) for task in PUBLIC_TASKS),
        "validation_targets": tuple(_target_record(target) for target in PUBLIC_TARGETS),
        "prior_pass": asdict(prior_pass),
        "runner_source_lf_sha256": _lf_sha256(Path(__file__)),
    }


def _acquire_spend_marker(commitment: dict[str, object]) -> bool:
    marker = {
        "schema": f"{SCHEMA}-spend",
        "commitment": commitment,
        "commitment_sha256": sha256(_canonical_json(commitment)).hexdigest(),
        "disposition": (
            "Retain after PASS, STOP, exception, interruption, or crash; this "
            "four-task public family-transfer cohort must never rerun."
        ),
    }
    try:
        with _SPEND_MARKER_PATH.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(marker, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return False
    return True


def _validate_predeclared_boundary() -> None:
    if PUBLIC_LITERAL_BANK != tuple(range(-4, 5)):
        raise RuntimeError("family-transfer literal bank changed")
    if PUBLIC_LITERAL_BANK != hierarchical.PUBLIC_LITERAL_BANK:
        raise RuntimeError("family-transfer literal bank differs from frozen planner")
    if len(PUBLIC_TASKS) != len(PUBLIC_TARGETS) or len(PUBLIC_TASKS) != 4:
        raise RuntimeError("family-transfer boundary must contain four tasks")
    task_names = tuple(task.name for task in PUBLIC_TASKS)
    target_names = tuple(target.name for target in PUBLIC_TARGETS)
    if len(set(task_names)) != 4 or task_names != target_names:
        raise RuntimeError("family-transfer tasks and targets differ")


def main() -> int:
    if _ARTIFACT_PATH.exists():
        prior = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "artifact": str(_ARTIFACT_PATH),
                    "verdict": prior.get("verdict"),
                    "guard": "existing artifact; family-transfer cohort not rerun",
                },
                sort_keys=True,
            )
        )
        return 0
    if _SPEND_MARKER_PATH.exists():
        print(
            json.dumps(
                {
                    "artifact": str(_ARTIFACT_PATH),
                    "spend_marker": str(_SPEND_MARKER_PATH),
                    "guard": "family-transfer cohort already spent; no rerun",
                },
                sort_keys=True,
            )
        )
        return 2

    _validate_predeclared_boundary()
    prior_pass = authenticate_prior_pass()
    commitment = _fresh_commitment(prior_pass)
    if not _acquire_spend_marker(commitment):
        raise RuntimeError("family-transfer spend marker already exists")

    try:
        result = run_public_matrix(prior_pass)
        payload: dict[str, object] = asdict(result)
    except Exception as error:
        payload = {
            "schema": SCHEMA,
            "verdict": STOP_VERDICT,
            "passed_tasks": 0,
            "total_tasks": len(PUBLIC_TASKS),
            "wall_time_seconds": None,
            "literal_bank": PUBLIC_LITERAL_BANK,
            "prior_pass": asdict(prior_pass),
            "model_checkpoint_sha256": PRIOR_MODEL_CHECKPOINT_SHA256,
            "tasks": (),
            "error": f"{type(error).__name__}: {error}",
            "retry_policy": "Spent after exception; no retry or tuning.",
            "claim_boundary": "No transfer claim after physical or runtime STOP.",
        }
    payload["commitment_sha256"] = sha256(_canonical_json(commitment)).hexdigest()
    _ARTIFACT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(_ARTIFACT_PATH),
                "verdict": payload["verdict"],
                "passed_tasks": payload["passed_tasks"],
                "total_tasks": payload["total_tasks"],
                "commitment_sha256": payload["commitment_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["verdict"] == PASS_VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PASS_VERDICT",
    "PRIOR_ARTIFACT_LF_SHA256",
    "PRIOR_MODEL_CHECKPOINT_SHA256",
    "PRIOR_PLANNER_SOURCE_LF_SHA256",
    "PRIOR_SPEND_MARKER_LF_SHA256",
    "PUBLIC_BOOL_CASES",
    "PUBLIC_LITERAL_BANK",
    "PUBLIC_TARGETS",
    "PUBLIC_TASKS",
    "PUBLIC_TWO_INT_CASES",
    "PriorPassEvidence",
    "PublicFamilyTransferMatrixResult",
    "PublicFamilyTransferTarget",
    "PublicFamilyTransferTask",
    "PublicFamilyTransferTaskResult",
    "SCHEMA",
    "STOP_VERDICT",
    "authenticate_prior_pass",
    "run_public_matrix",
    "run_public_task",
]
