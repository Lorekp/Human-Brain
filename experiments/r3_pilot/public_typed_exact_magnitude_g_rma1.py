"""One-shot G-RMA1 runner over the frozen exact-magnitude public actor.

Importing this module is inert.  ``run_g_rma1`` first reconstructs the public
actor and verifies every banked commitment.  Only after that pre-target barrier
does it create the fresh four-case grid and ``abs(x) + 3`` target, then perform
exactly three frozen cell selections and typed applications.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for search_path in (REPO_ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.r3_pilot import (  # noqa: E402
    public_typed_exact_magnitude_reward_actor as exact,
)
from experiments.r3_pilot import (  # noqa: E402
    public_typed_reward_modulated_edit_actor as banked,
)

_EXPECTED_FROZEN_SHA256 = (
    "fbf4141c90a11b40dfdb3cba507c6885c44b525f3113e4439ca347df37836dc6"
)
_EXPECTED_PROJECTION_SHA256 = (
    "dbddb64f0809ec0111ed68a5d02c2abadc0f3fce4c332779ad876397abe741a5"
)
_EXPECTED_ATOMIC_ARTIFACT_SHA256 = (
    "f86f364e4f6110cb2aa42eb8e16515ba3d258686fa078cff05af0f93ac1f933e"
)
_ARTIFACT_PATH = Path(__file__).with_name(
    "public_typed_exact_magnitude_g_rma1_probe.json"
)
_ACTION_BY_LABEL = {action.label: action for action in banked.ACTIONS}


@dataclass(frozen=True, slots=True)
class FrozenCommitments:
    seed: int
    feature_cells: int
    readout_fan_in: int
    action_cells: int
    output_cells_with_sink: int
    curriculum_epochs: int
    motor_trials: int
    learning_rate_pa: float
    maximum_weight_pa: float
    output_base_current_pa: float
    output_race_steps: int
    projection_current_scale_pa: float
    projection_current_limit_pa: float
    projection_steps: int
    projection_sha256: str
    frozen_checkpoint_sha256: str
    atomic_artifact_sha256: str
    atomic_verdict: str
    pacemaker_all_zero: bool


@dataclass(frozen=True, slots=True)
class CompositionStep:
    step: int
    before_outputs: tuple[int, ...]
    selected_action: str
    program_tokens: tuple[str, ...]
    after_outputs: tuple[int, ...]
    selection: banked.ActionSelection


@dataclass(frozen=True, slots=True)
class GRMA1Result:
    schema: str
    commitments: FrozenCommitments
    arguments: tuple[int, ...]
    desired_outputs: tuple[int, ...]
    expected_actions: tuple[str, ...]
    expected_final_tokens: tuple[str, ...]
    steps: tuple[CompositionStep, ...]
    selection_attempts: int
    applied_actions: int
    final_outputs: tuple[int, ...]
    final_tokens: tuple[str, ...]
    exact_action_sequence: bool
    exact_final_outputs: bool
    exact_final_tokens: bool
    unique_races: bool
    frozen_checkpoint_immutable: bool
    failure_stage: str | None
    error: str | None
    verdict: str
    retry_policy: str
    claim_boundary: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"G-RMA1 pre-target commitment failed: {message}")


def _verify_atomic_readiness() -> tuple[str, str]:
    _require(exact._ARTIFACT_PATH.is_file(), "exact-stage artifact is missing")
    artifact_bytes = exact._ARTIFACT_PATH.read_bytes()
    artifact_sha = sha256(artifact_bytes).hexdigest()
    _require(
        artifact_sha == _EXPECTED_ATOMIC_ARTIFACT_SHA256,
        "exact-stage artifact hash changed",
    )
    artifact = json.loads(artifact_bytes.decode("utf-8"))
    _require(
        artifact["verdict"] == "PASS_EXACT_MAGNITUDE_ATOMIC_REPLAY",
        "exact-stage artifact is not PASS",
    )
    _require(
        artifact["projection_sha256"] == _EXPECTED_PROJECTION_SHA256,
        "artifact projection hash changed",
    )
    _require(
        artifact["training"]["frozen_checkpoint_sha256"]
        == _EXPECTED_FROZEN_SHA256,
        "artifact frozen checkpoint changed",
    )
    _require(
        artifact["replay"]["passed_goals"]
        == artifact["replay"]["total_goals"]
        == 10
        and all(
            arm["passed"] and not arm["tied"]
            for arm in artifact["replay"]["arms"]
        ),
        "exact-stage atomic replay is not 10/10 unique",
    )
    return artifact_sha, artifact["verdict"]


def _verify_frozen_commitments(
    frozen: exact.FrozenExactMagnitudeRewardActor,
    training: banked.CurriculumSummary,
    atomic_artifact_sha256: str,
    atomic_verdict: str,
) -> FrozenCommitments:
    """Fail closed before any fresh target or argument grid is constructed."""
    projection_sha = exact.ExactMagnitudeTransitionCellEncoder().projection_sha256
    expected_actions = (
        "DONE",
        "WRAP_ABS",
        "WRAP_ADD[1]",
        "WRAP_ADD[2]",
        "WRAP_ADD[3]",
        "WRAP_ADD[4]",
        "WRAP_SUB[1]",
        "WRAP_SUB[2]",
        "WRAP_SUB[3]",
        "WRAP_SUB[4]",
    )
    _require(banked.SEED == frozen.seed == 17, "seed is not 17")
    _require(banked._FEATURE_CELLS == exact._FEATURE_CELLS == 260, "width is not 260")
    _require(banked._READOUT_FAN_IN == 65, "readout fan-in is not 65")
    _require(
        tuple(action.label for action in banked.ACTIONS) == expected_actions,
        "action bank changed",
    )
    _require(
        tuple(_ACTION_BY_LABEL) == expected_actions,
        "typed actuator label map changed",
    )
    _require(len(banked.ACTIONS) == 10, "action-cell count changed")
    _require(banked._SINK_CELL == 10, "masked sink identity changed")
    _require(banked._OUTPUT_CELLS == 11, "output population width changed")
    _require(banked.CURRICULUM_EPOCHS == 3, "curriculum epoch count changed")
    _require(
        banked.TRAINING_ARGUMENT_GRIDS
        == ((-12, -5, 2, 9), (-11, -4, 3, 8)),
        "public training grids changed",
    )
    _require(training.motor_trials == 300, "motor schedule is not 3 x 10 x 10")
    _require(
        all(count == 30 for _, count in training.motor_coverage),
        "round-robin action coverage changed",
    )
    _require(
        (training.positive_advantages, training.neutral_advantages, training.negative_advantages)
        == (129, 0, 171),
        "centered-advantage schedule changed",
    )
    _require(training.local_updates == 300, "local-update count changed")
    _require(training.structural_recruitments == 383, "recruitment count changed")
    _require(training.frozen_before_gate_target, "training did not freeze pre-target")
    _require(banked._LEARNING_RATE_PA == 32.0, "learning rate changed")
    _require(banked._MAXIMUM_WEIGHT_PA == 1_600.0, "maximum weight changed")
    _require(banked._OUTPUT_BASE_CURRENT_PA == 600.0, "race base current changed")
    _require(banked._OUTPUT_RACE_STEPS == 600, "race step bound changed")
    _require(
        exact._PROJECTION_CURRENT_SCALE_PA == 20_000.0,
        "projection current scale changed",
    )
    _require(
        exact._PROJECTION_CURRENT_LIMIT_PA == 5_000.0,
        "projection current limit changed",
    )
    _require(exact._PROJECTION_STEPS == 30, "projection step count changed")
    _require(
        projection_sha == _EXPECTED_PROJECTION_SHA256,
        "exact-magnitude projection hash changed",
    )
    _require(
        training.frozen_checkpoint_sha256
        == frozen.checkpoint_sha256
        == _EXPECTED_FROZEN_SHA256,
        "frozen checkpoint hash changed",
    )
    _require(
        frozen.pacemaker_synapses_pa.shape == (11,)
        and np.count_nonzero(frozen.pacemaker_synapses_pa) == 0,
        "pacemaker is not identically zero",
    )
    return FrozenCommitments(
        seed=frozen.seed,
        feature_cells=exact._FEATURE_CELLS,
        readout_fan_in=banked._READOUT_FAN_IN,
        action_cells=len(banked.ACTIONS),
        output_cells_with_sink=banked._OUTPUT_CELLS,
        curriculum_epochs=banked.CURRICULUM_EPOCHS,
        motor_trials=training.motor_trials,
        learning_rate_pa=banked._LEARNING_RATE_PA,
        maximum_weight_pa=banked._MAXIMUM_WEIGHT_PA,
        output_base_current_pa=banked._OUTPUT_BASE_CURRENT_PA,
        output_race_steps=banked._OUTPUT_RACE_STEPS,
        projection_current_scale_pa=exact._PROJECTION_CURRENT_SCALE_PA,
        projection_current_limit_pa=exact._PROJECTION_CURRENT_LIMIT_PA,
        projection_steps=exact._PROJECTION_STEPS,
        projection_sha256=projection_sha,
        frozen_checkpoint_sha256=frozen.checkpoint_sha256,
        atomic_artifact_sha256=atomic_artifact_sha256,
        atomic_verdict=atomic_verdict,
        pacemaker_all_zero=True,
    )


def run_g_rma1() -> GRMA1Result:
    """Spend exactly one frozen three-action composition trajectory."""
    atomic_artifact_sha, atomic_verdict = _verify_atomic_readiness()
    frozen, training = exact.train_public_actor()
    commitments = _verify_frozen_commitments(
        frozen,
        training,
        atomic_artifact_sha,
        atomic_verdict,
    )

    # PRE-TARGET BARRIER: no fresh grid or composition target exists above.
    arguments = (-13, -6, 1, 10)
    identity = banked._identity_program()
    target = banked._apply_action(
        banked._apply_action(identity, _ACTION_BY_LABEL["WRAP_ABS"]),
        _ACTION_BY_LABEL["WRAP_ADD[3]"],
    )
    desired_outputs = banked._execute(target, arguments)
    _require(
        desired_outputs == (16, 9, 4, 13),
        "fresh target does not lower to the committed outputs",
    )

    program = identity
    steps: list[CompositionStep] = []
    selection_attempts = 0
    failure_stage: str | None = None
    error_message: str | None = None
    for step_number in range(1, 4):
        before_outputs = banked._execute(program, arguments)
        selection_attempts += 1
        try:
            selection = frozen.select(before_outputs, desired_outputs)
        except banked.NonUniqueActionRace as error:
            failure_stage = f"selection_{step_number}"
            error_message = str(error)
            break
        program = banked._apply_action(
            program, _ACTION_BY_LABEL[selection.action]
        )
        after_outputs = banked._execute(program, arguments)
        steps.append(
            CompositionStep(
                step=step_number,
                before_outputs=before_outputs,
                selected_action=selection.action,
                program_tokens=program.tokens,
                after_outputs=after_outputs,
                selection=selection,
            )
        )

    expected_actions = ("WRAP_ABS", "WRAP_ADD[3]", "DONE")
    expected_final_tokens = (
        "add",
        "absolute",
        "variable-0",
        "integer",
        "+",
        "3",
        "end",
    )
    selected_actions = tuple(step.selected_action for step in steps)
    final_outputs = banked._execute(program, arguments)
    final_tokens = program.tokens
    exact_action_sequence = selected_actions == expected_actions
    exact_final_outputs = final_outputs == desired_outputs == (16, 9, 4, 13)
    exact_final_tokens = final_tokens == expected_final_tokens
    unique_races = failure_stage is None and all(
        step.selection.simultaneous_spike_candidates == 1 for step in steps
    )
    frozen_checkpoint_immutable = all(
        step.selection.checkpoint_before_sha256
        == step.selection.checkpoint_after_sha256
        == commitments.frozen_checkpoint_sha256
        for step in steps
    )
    passed = (
        len(steps) == 3
        and exact_action_sequence
        and exact_final_outputs
        and exact_final_tokens
        and unique_races
        and frozen_checkpoint_immutable
    )
    return GRMA1Result(
        schema="public-typed-exact-magnitude-g-rma1-v1",
        commitments=commitments,
        arguments=arguments,
        desired_outputs=desired_outputs,
        expected_actions=expected_actions,
        expected_final_tokens=expected_final_tokens,
        steps=tuple(steps),
        selection_attempts=selection_attempts,
        applied_actions=len(steps),
        final_outputs=final_outputs,
        final_tokens=final_tokens,
        exact_action_sequence=exact_action_sequence,
        exact_final_outputs=exact_final_outputs,
        exact_final_tokens=exact_final_tokens,
        unique_races=unique_races,
        frozen_checkpoint_immutable=frozen_checkpoint_immutable,
        failure_stage=failure_stage,
        error=error_message,
        verdict="PASS_G_RMA1" if passed else "STOP_G_RMA1",
        retry_policy=(
            "One fixed seed-17 three-selection trajectory; no retry, seed, schedule, "
            "projection, reward, rate, fan-in, width, or action-bank sweep."
        ),
        claim_boundary=(
            "Fresh public scalar three-edit composition evidence only; not Simple-SWE, "
            "M0, held-out, C1, or broad coding capability evidence."
        ),
    )


def main() -> int:
    if _ARTIFACT_PATH.exists():
        stored = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {"status": "already-spent", "verdict": stored["verdict"]},
                sort_keys=True,
            )
        )
        return 0
    result = run_g_rma1()
    _ARTIFACT_PATH.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": _ARTIFACT_PATH.relative_to(REPO_ROOT).as_posix(),
                "verdict": result.verdict,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
