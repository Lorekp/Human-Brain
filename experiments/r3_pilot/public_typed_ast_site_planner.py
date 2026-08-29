"""Physical H=1 site-by-wrap planning over an incumbent typed AST.

The world model observes and executes only the incumbent program.  This module
crosses every observed edit site with the fixed nine generic wrap motors, adds
one global ``DONE`` candidate, predicts root outputs without constructing any
candidate AST, and lets fixed synapses drive a fresh candidate-cell race.

Only the selected factor pair is projected into separate site and action motor
populations.  The actuator is deliberately separate: it is the sole path that
may replace a subtree and observe the resulting real program.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for search_path in (REPO_ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.r3_pilot import public_typed_ast_world_model as world  # noqa: E402
from humanbrain import typed_program as tp  # noqa: E402
from humanbrain.typed_program_types import (  # noqa: E402
    ProgramParameter,
    ProgramType,
    TypedSignature,
)

total = world.total
UNKNOWN = world.UNKNOWN
SuccessorState = world.SuccessorState
CellRace = total.banked.CellRace

DONE = "DONE"
ACTION_LABELS = (DONE, *world.WRAP_ACTION_LABELS)
STOP_ALL_UNKNOWN = "STOP_ALL_UNKNOWN"

_VALUE_BASE_CURRENT_PA = 600.0
_KNOWN_SUPPORT_WEIGHT_PA = 120.0
_EXACT_MATCH_WEIGHT_PA = 140.0
_UNKNOWN_INHIBITION_MARGIN_PA = 200.0
_ALL_UNKNOWN_STOP_CURRENT_PA = 650.0
_FACTOR_MOTOR_CURRENT_PA = 5_000.0

VISIBLE_ARGUMENTS = (-8, -3, 0, 2, 7, 9)
_VISIBLE_ARTIFACT_PATH = Path(__file__).with_name(
    "public_typed_ast_site_visible_probe.json"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _path_label(path: tuple[int, ...]) -> str:
    return "root" if not path else ".".join(str(part) for part in path)


def _state_label(state: SuccessorState) -> str:
    return "UNKNOWN" if state is UNKNOWN else str(state)


@dataclass(frozen=True, slots=True)
class SiteAction:
    path: tuple[int, ...]
    action: str

    @property
    def label(self) -> str:
        return f"{_path_label(self.path)}::{self.action}"


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    candidate: SiteAction
    predicted_root_outputs: tuple[SuccessorState, ...]
    supported: bool
    unsupported_reason: str | None
    successor_races: tuple[CellRace, ...]


def _total_state(value: int) -> SuccessorState:
    if total.banked.VALUE_MIN <= value <= total.banked.VALUE_MAX:
        return value
    return UNKNOWN


def _candidate_bank(
    observation: world.CurrentProgramObservation,
) -> tuple[SiteAction, ...]:
    paths = tuple(site.path for site in observation.sites)
    if not paths or paths[0] != () or len(paths) != len(set(paths)):
        raise ValueError("AST observation needs unique preorder sites rooted at ()")
    return (
        *tuple(
            SiteAction(path, action)
            for path in paths
            for action in world.WRAP_ACTION_LABELS
        ),
        SiteAction((), DONE),
    )


def _predict_candidates(
    model: world.FrozenAstWorldModel,
    observation: world.CurrentProgramObservation,
) -> tuple[CandidatePrediction, ...]:
    predictions: list[CandidatePrediction] = []
    for candidate in _candidate_bank(observation):
        if candidate.action == DONE:
            root_outputs = observation.site(()).outputs
            pre_states = (
                tuple(UNKNOWN for _ in observation.arguments)
                if root_outputs is None
                else tuple(_total_state(value) for value in root_outputs)
            )
            outputs, races = model.successor.predict_many(pre_states, DONE)
            supported = all(state is not UNKNOWN for state in outputs)
            predictions.append(
                CandidatePrediction(
                    candidate=candidate,
                    predicted_root_outputs=outputs,
                    supported=supported,
                    unsupported_reason=(
                        None if supported else "incumbent root identity is unsupported"
                    ),
                    successor_races=races,
                )
            )
            continue

        prediction = model.predict(
            observation,
            candidate.path,
            candidate.action,
        )
        predictions.append(
            CandidatePrediction(
                candidate=candidate,
                predicted_root_outputs=prediction.predicted_root_outputs,
                supported=prediction.supported,
                unsupported_reason=prediction.unsupported_reason,
                successor_races=prediction.races,
            )
        )
    return tuple(predictions)


@dataclass(frozen=True, slots=True)
class FixedSiteActionValueProjection:
    """Materialized fixed synapses from state/match cells to candidate cells."""

    candidate_labels: tuple[str, ...]
    case_count: int
    weights_pa: np.ndarray
    bias_pa: np.ndarray
    stop_current_pa: float
    sha256: str

    @classmethod
    def create(
        cls,
        candidate_labels: Sequence[str],
        case_count: int,
    ) -> "FixedSiteActionValueProjection":
        labels = tuple(candidate_labels)
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("value projection needs unique candidate labels")
        if case_count <= 0:
            raise ValueError("value projection needs at least one case")

        # candidate x {predicted-state, exact-coincidence} x case x state-cell
        weights = np.zeros(
            (len(labels), 2, case_count, total.TOTAL_STATE_COUNT),
            dtype=np.float64,
        )
        weights[:, 0, :, : total.UNKNOWN_CELL] = _KNOWN_SUPPORT_WEIGHT_PA
        unknown_inhibition = (
            case_count
            * (_KNOWN_SUPPORT_WEIGHT_PA + _EXACT_MATCH_WEIGHT_PA)
            + _UNKNOWN_INHIBITION_MARGIN_PA
        )
        weights[:, 0, :, total.UNKNOWN_CELL] = -unknown_inhibition
        weights[:, 1, :, : total.UNKNOWN_CELL] = _EXACT_MATCH_WEIGHT_PA
        bias = np.full(len(labels), _VALUE_BASE_CURRENT_PA, dtype=np.float64)

        digest = sha256()
        digest.update(weights.tobytes())
        digest.update(bias.tobytes())
        digest.update(_canonical_json(labels))
        digest.update(str(_ALL_UNKNOWN_STOP_CURRENT_PA).encode("ascii"))
        frozen_weights = np.ascontiguousarray(weights)
        frozen_bias = np.ascontiguousarray(bias)
        frozen_weights.setflags(write=False)
        frozen_bias.setflags(write=False)
        return cls(
            candidate_labels=labels,
            case_count=case_count,
            weights_pa=frozen_weights,
            bias_pa=frozen_bias,
            stop_current_pa=_ALL_UNKNOWN_STOP_CURRENT_PA,
            sha256=digest.hexdigest(),
        )

    def currents(
        self,
        state_spikes: np.ndarray,
        coincidence_spikes: np.ndarray,
    ) -> np.ndarray:
        states = np.ascontiguousarray(state_spikes, dtype=np.float64)
        coincidences = np.ascontiguousarray(coincidence_spikes, dtype=np.float64)
        expected = (
            len(self.candidate_labels),
            self.case_count,
            total.TOTAL_STATE_COUNT,
        )
        if states.shape != expected or coincidences.shape != expected:
            raise ValueError("candidate features do not match fixed value synapses")
        features = np.ascontiguousarray(np.stack((states, coincidences), axis=1))
        candidate_currents = self.bias_pa + np.einsum(
            "ctiv,ctiv->c",
            self.weights_pa,
            features,
        )
        return np.ascontiguousarray(
            np.concatenate(
                (
                    candidate_currents,
                    np.asarray((self.stop_current_pa,), dtype=np.float64),
                )
            )
        )


def _prediction_features(
    predictions: Sequence[CandidatePrediction],
    desired_outputs: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, str]:
    candidates = tuple(predictions)
    desired = tuple(desired_outputs)
    if not candidates or not desired:
        raise ValueError("prediction features need candidates and desired outputs")
    states = np.zeros(
        (len(candidates), len(desired), total.TOTAL_STATE_COUNT),
        dtype=np.float64,
    )
    coincidences = np.zeros_like(states)
    feature_records: list[dict[str, object]] = []
    for candidate_cell, prediction in enumerate(candidates):
        predicted = prediction.predicted_root_outputs
        if len(predicted) != len(desired):
            raise ValueError("candidate prediction has the wrong case count")
        for case, state in enumerate(predicted):
            states[candidate_cell, case] = total.TotalValueCellEncoder.encode(state)
        coincidence, coincidence_sha = total.coincidence_features(predicted, desired)
        coincidences[candidate_cell] = coincidence
        feature_records.append(
            {
                "candidate": prediction.candidate.label,
                "predicted": tuple(_state_label(state) for state in predicted),
                "coincidence_sha256": coincidence_sha,
            }
        )
    digest = sha256(_canonical_json(feature_records)).hexdigest()
    return states, coincidences, digest


@dataclass(frozen=True, slots=True)
class FixedCandidateFactorProjection:
    """Fixed one-hot synapses from a complete candidate into its two motors."""

    site_paths: tuple[tuple[int, ...], ...]
    action_labels: tuple[str, ...]
    site_weights: np.ndarray
    action_weights: np.ndarray
    sha256: str

    @classmethod
    def create(
        cls,
        candidates: Sequence[SiteAction],
        site_paths: Sequence[tuple[int, ...]],
    ) -> "FixedCandidateFactorProjection":
        bank = tuple(candidates)
        paths = tuple(site_paths)
        if not bank or not paths or len(paths) != len(set(paths)):
            raise ValueError("factor projection needs candidates and unique sites")
        site_index = {path: index for index, path in enumerate(paths)}
        action_index = {label: index for index, label in enumerate(ACTION_LABELS)}
        site_weights = np.zeros((len(paths), len(bank)), dtype=np.float64)
        action_weights = np.zeros((len(ACTION_LABELS), len(bank)), dtype=np.float64)
        for candidate_cell, candidate in enumerate(bank):
            site_weights[site_index[candidate.path], candidate_cell] = 1.0
            action_weights[action_index[candidate.action], candidate_cell] = 1.0
        digest = sha256()
        digest.update(site_weights.tobytes())
        digest.update(action_weights.tobytes())
        digest.update(_canonical_json(tuple(candidate.label for candidate in bank)))
        frozen_sites = np.ascontiguousarray(site_weights)
        frozen_actions = np.ascontiguousarray(action_weights)
        frozen_sites.setflags(write=False)
        frozen_actions.setflags(write=False)
        return cls(
            site_paths=paths,
            action_labels=ACTION_LABELS,
            site_weights=frozen_sites,
            action_weights=frozen_actions,
            sha256=digest.hexdigest(),
        )

    def races(self, candidate_cell: int) -> tuple[CellRace, CellRace]:
        if not 0 <= candidate_cell < self.site_weights.shape[1]:
            raise ValueError("factor projection candidate is outside the bank")
        spikes = np.zeros(self.site_weights.shape[1], dtype=np.float64)
        spikes[candidate_cell] = 1.0
        site_currents = np.ascontiguousarray(
            _FACTOR_MOTOR_CURRENT_PA * (self.site_weights @ spikes)
        )
        action_currents = np.ascontiguousarray(
            _FACTOR_MOTOR_CURRENT_PA * (self.action_weights @ spikes)
        )
        site_race = total.banked._physical_race(
            site_currents,
            tuple(_path_label(path) for path in self.site_paths),
            bank="ast-site-motor",
        )
        action_race = total.banked._physical_race(
            action_currents,
            self.action_labels,
            bank="ast-action-motor",
        )
        return site_race, action_race


@dataclass(frozen=True, slots=True)
class SiteActionDecision:
    status: str
    selected_path: tuple[int, ...] | None
    selected_action: str | None
    predicted_root_outputs: tuple[SuccessorState, ...] | None
    candidate_race: CellRace
    site_motor_race: CellRace | None
    action_motor_race: CellRace | None
    candidates: tuple[CandidatePrediction, ...]
    observation_sha256: str
    program_sha256: str
    feature_sha256: str
    value_projection_sha256: str
    factor_projection_sha256: str | None
    candidate_digest: str
    world_checkpoint_before_sha256: str
    world_checkpoint_after_sha256: str


def plan_site_action_h1(
    model: world.FrozenAstWorldModel,
    observation: world.CurrentProgramObservation,
    desired_outputs: Sequence[int],
) -> SiteActionDecision:
    """Physically select one complete site/action candidate without execution."""
    desired = tuple(desired_outputs)
    if not desired or len(desired) != len(observation.arguments):
        raise ValueError("site planning needs paired desired outputs")
    if not all(
        type(value) is int
        and total.banked.VALUE_MIN <= value <= total.banked.VALUE_MAX
        for value in desired
    ):
        raise ValueError("desired outputs must fit the fixed raw-value population")

    checkpoint_before = model.live_checkpoint_sha256
    if checkpoint_before != model.checkpoint_sha256:
        raise RuntimeError("AST world-model checkpoint changed before site planning")
    predictions = _predict_candidates(model, observation)
    labels = tuple(prediction.candidate.label for prediction in predictions)
    state_spikes, coincidence_spikes, feature_sha = _prediction_features(
        predictions,
        desired,
    )
    projection = FixedSiteActionValueProjection.create(labels, len(desired))
    currents = projection.currents(state_spikes, coincidence_spikes)
    race = total.banked._physical_race(
        currents,
        (*labels, STOP_ALL_UNKNOWN),
        bank="ast-site-action-value",
    )
    checkpoint_after = model.live_checkpoint_sha256
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("frozen AST world model changed during site planning")

    candidate_digest = sha256(
        _canonical_json(
            tuple(
                {
                    "candidate": prediction.candidate.label,
                    "predicted": tuple(
                        _state_label(state)
                        for state in prediction.predicted_root_outputs
                    ),
                    "supported": prediction.supported,
                }
                for prediction in predictions
            )
        )
    ).hexdigest()
    if race.winner == STOP_ALL_UNKNOWN:
        return SiteActionDecision(
            status=STOP_ALL_UNKNOWN,
            selected_path=None,
            selected_action=None,
            predicted_root_outputs=None,
            candidate_race=race,
            site_motor_race=None,
            action_motor_race=None,
            candidates=predictions,
            observation_sha256=observation.sha256,
            program_sha256=observation.program_sha256,
            feature_sha256=feature_sha,
            value_projection_sha256=projection.sha256,
            factor_projection_sha256=None,
            candidate_digest=candidate_digest,
            world_checkpoint_before_sha256=checkpoint_before,
            world_checkpoint_after_sha256=checkpoint_after,
        )

    selected = predictions[race.winner_cell]
    factor_projection = FixedCandidateFactorProjection.create(
        tuple(prediction.candidate for prediction in predictions),
        tuple(site.path for site in observation.sites),
    )
    site_race, action_race = factor_projection.races(race.winner_cell)
    if site_race.winner != _path_label(selected.candidate.path):
        raise RuntimeError("candidate-to-site physical projection lost factor parity")
    if action_race.winner != selected.candidate.action:
        raise RuntimeError("candidate-to-action physical projection lost factor parity")
    return SiteActionDecision(
        status="SELECTED",
        selected_path=selected.candidate.path,
        selected_action=selected.candidate.action,
        predicted_root_outputs=selected.predicted_root_outputs,
        candidate_race=race,
        site_motor_race=site_race,
        action_motor_race=action_race,
        candidates=predictions,
        observation_sha256=observation.sha256,
        program_sha256=observation.program_sha256,
        feature_sha256=feature_sha,
        value_projection_sha256=projection.sha256,
        factor_projection_sha256=factor_projection.sha256,
        candidate_digest=candidate_digest,
        world_checkpoint_before_sha256=checkpoint_before,
        world_checkpoint_after_sha256=checkpoint_after,
    )


@dataclass(frozen=True, slots=True)
class SelectedActuation:
    path: tuple[int, ...]
    action: str
    program_sha256_before: str
    program_sha256_after: str
    tokens_after: tuple[str, ...]
    actual_root_outputs: tuple[int, ...]
    observation_sha256: str


def actuate_selected_site_action(
    program: tp.Program,
    decision: SiteActionDecision,
    arguments: Sequence[int],
) -> SelectedActuation:
    """Apply and execute only the already-selected physical factor pair."""
    if decision.status != "SELECTED":
        raise RuntimeError("an all-unknown STOP has no actuator command")
    if decision.selected_path is None or decision.selected_action is None:
        raise RuntimeError("selected decision is missing its physical factors")
    if program.sha256 != decision.program_sha256:
        raise RuntimeError("actuator program differs from the planned incumbent")
    edited = (
        program
        if decision.selected_action == DONE
        else world.apply_selected_wrap(
            program,
            decision.selected_path,
            decision.selected_action,
        )
    )
    actual = world.observe_current_program(edited, arguments)
    outputs = actual.site(()).outputs
    if outputs is None:
        raise RuntimeError("selected real edit did not produce scalar Int outputs")
    return SelectedActuation(
        path=decision.selected_path,
        action=decision.selected_action,
        program_sha256_before=program.sha256,
        program_sha256_after=edited.sha256,
        tokens_after=edited.tokens,
        actual_root_outputs=outputs,
        observation_sha256=actual.sha256,
    )


@dataclass(frozen=True, slots=True)
class VisibleOneEditTask:
    name: str
    incumbent_tokens: tuple[str, ...]
    desired_outputs: tuple[int, ...]
    expected_path: tuple[int, ...]
    expected_action: str
    expected_tokens: tuple[str, ...]


_LITERAL_1 = ("integer", "+", "1", "end")
_LITERAL_2 = ("integer", "+", "2", "end")

VISIBLE_TASKS = (
    VisibleOneEditTask(
        "root-absolute",
        ("variable-0",),
        (8, 3, 0, 2, 7, 9),
        (),
        "WRAP_ABS",
        ("absolute", "variable-0"),
    ),
    VisibleOneEditTask(
        "child-add-one-under-absolute",
        ("absolute", "variable-0"),
        (7, 2, 1, 3, 8, 10),
        (0,),
        "WRAP_ADD[1]",
        ("absolute", "add", "variable-0", *_LITERAL_1),
    ),
    VisibleOneEditTask(
        "child-absolute-under-add-two",
        ("add", "variable-0", *_LITERAL_2),
        (10, 5, 2, 4, 9, 11),
        (0,),
        "WRAP_ABS",
        ("add", "absolute", "variable-0", *_LITERAL_2),
    ),
    VisibleOneEditTask(
        "child-subtract-one-under-absolute",
        ("absolute", "variable-0"),
        (9, 4, 1, 1, 6, 8),
        (0,),
        "WRAP_SUB[1]",
        ("absolute", "subtract", "variable-0", *_LITERAL_1),
    ),
    VisibleOneEditTask(
        "root-absolute-around-add-one",
        ("add", "variable-0", *_LITERAL_1),
        (7, 2, 1, 3, 8, 10),
        (),
        "WRAP_ABS",
        ("absolute", "add", "variable-0", *_LITERAL_1),
    ),
    VisibleOneEditTask(
        "root-absolute-around-subtract-two",
        ("subtract", "variable-0", *_LITERAL_2),
        (10, 5, 2, 0, 5, 7),
        (),
        "WRAP_ABS",
        ("absolute", "subtract", "variable-0", *_LITERAL_2),
    ),
    VisibleOneEditTask(
        "deep-child-absolute",
        ("absolute", "add", "variable-0", *_LITERAL_1),
        (9, 4, 1, 3, 8, 10),
        (0, 0),
        "WRAP_ABS",
        ("absolute", "add", "absolute", "variable-0", *_LITERAL_1),
    ),
    VisibleOneEditTask(
        "global-done",
        ("variable-0",),
        (-8, -3, 0, 2, 7, 9),
        (),
        DONE,
        ("variable-0",),
    ),
)


def visible_scalar_signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("value", integer),),
        result=integer,
    )


def visible_incumbent(task: VisibleOneEditTask) -> tp.Program:
    return tp.Program(task.incumbent_tokens, visible_scalar_signature())


@dataclass(frozen=True, slots=True)
class VisibleTaskResult:
    name: str
    verdict: str
    passed: bool
    error: str | None
    incumbent_tokens: tuple[str, ...]
    expected_tokens: tuple[str, ...]
    desired_outputs: tuple[int, ...]
    expected_path: tuple[int, ...]
    expected_action: str
    selected_path: tuple[int, ...] | None
    selected_action: str | None
    predicted_root_outputs: tuple[SuccessorState, ...] | None
    actual_tokens: tuple[str, ...] | None
    actual_outputs: tuple[int, ...] | None
    candidate_count: int
    successor_race_count: int
    selection_race_count: int
    total_physical_races: int
    physical_unique: bool
    checkpoint_immutable: bool
    observation_sha256: str
    candidate_digest: str | None
    value_projection_sha256: str | None
    factor_projection_sha256: str | None


@dataclass(frozen=True, slots=True)
class VisibleMatrixResult:
    schema: str
    verdict: str
    passed_tasks: int
    total_tasks: int
    selected_real_executions: int
    candidate_count: int
    successor_race_count: int
    selection_race_count: int
    total_physical_races: int
    world_checkpoint_sha256: str
    target_free_observations: int
    tasks: tuple[VisibleTaskResult, ...]
    claim_boundary: str


def _validate_visible_target(task: VisibleOneEditTask) -> None:
    expected = tp.Program(task.expected_tokens, visible_scalar_signature())
    actual = world.observe_current_program(expected, VISIBLE_ARGUMENTS).site(()).outputs
    if actual != task.desired_outputs:
        raise RuntimeError(
            f"visible task {task.name!r} desired outputs do not match its typed target"
        )


def run_visible_task(
    model: world.FrozenAstWorldModel,
    task: VisibleOneEditTask,
) -> VisibleTaskResult:
    """Run one public task through one plan and at most one selected edit."""
    _validate_visible_target(task)
    incumbent = visible_incumbent(task)
    observation = world.observe_current_program(incumbent, VISIBLE_ARGUMENTS)
    checkpoint_before = model.live_checkpoint_sha256
    decision = plan_site_action_h1(
        model,
        observation,
        task.desired_outputs,
    )

    successor_races = sum(
        len(candidate.successor_races) for candidate in decision.candidates
    )
    selection_races = 1 + int(decision.site_motor_race is not None) + int(
        decision.action_motor_race is not None
    )
    all_races = (
        *(race for candidate in decision.candidates for race in candidate.successor_races),
        decision.candidate_race,
        *((decision.site_motor_race,) if decision.site_motor_race is not None else ()),
        *((decision.action_motor_race,) if decision.action_motor_race is not None else ()),
    )
    physical_unique = all(
        race.simultaneous_spike_candidates == 1 for race in all_races
    )
    if decision.status == STOP_ALL_UNKNOWN:
        return VisibleTaskResult(
            name=task.name,
            verdict="STOP",
            passed=False,
            error=STOP_ALL_UNKNOWN,
            incumbent_tokens=task.incumbent_tokens,
            expected_tokens=task.expected_tokens,
            desired_outputs=task.desired_outputs,
            expected_path=task.expected_path,
            expected_action=task.expected_action,
            selected_path=None,
            selected_action=None,
            predicted_root_outputs=None,
            actual_tokens=None,
            actual_outputs=None,
            candidate_count=len(decision.candidates),
            successor_race_count=successor_races,
            selection_race_count=selection_races,
            total_physical_races=successor_races + selection_races,
            physical_unique=physical_unique,
            checkpoint_immutable=(
                checkpoint_before
                == decision.world_checkpoint_before_sha256
                == decision.world_checkpoint_after_sha256
                == model.live_checkpoint_sha256
            ),
            observation_sha256=observation.sha256,
            candidate_digest=decision.candidate_digest,
            value_projection_sha256=decision.value_projection_sha256,
            factor_projection_sha256=None,
        )

    actuation = actuate_selected_site_action(
        incumbent,
        decision,
        VISIBLE_ARGUMENTS,
    )
    checkpoint_immutable = (
        checkpoint_before
        == decision.world_checkpoint_before_sha256
        == decision.world_checkpoint_after_sha256
        == model.live_checkpoint_sha256
    )
    passed = (
        decision.selected_path == task.expected_path
        and decision.selected_action == task.expected_action
        and actuation.tokens_after == task.expected_tokens
        and actuation.actual_root_outputs == task.desired_outputs
        and physical_unique
        and checkpoint_immutable
    )
    return VisibleTaskResult(
        name=task.name,
        verdict="PASS" if passed else "STOP",
        passed=passed,
        error=None if passed else "selected physical edit missed the public target",
        incumbent_tokens=task.incumbent_tokens,
        expected_tokens=task.expected_tokens,
        desired_outputs=task.desired_outputs,
        expected_path=task.expected_path,
        expected_action=task.expected_action,
        selected_path=decision.selected_path,
        selected_action=decision.selected_action,
        predicted_root_outputs=decision.predicted_root_outputs,
        actual_tokens=actuation.tokens_after,
        actual_outputs=actuation.actual_root_outputs,
        candidate_count=len(decision.candidates),
        successor_race_count=successor_races,
        selection_race_count=selection_races,
        total_physical_races=successor_races + selection_races,
        physical_unique=physical_unique,
        checkpoint_immutable=checkpoint_immutable,
        observation_sha256=observation.sha256,
        candidate_digest=decision.candidate_digest,
        value_projection_sha256=decision.value_projection_sha256,
        factor_projection_sha256=decision.factor_projection_sha256,
    )


def run_visible_matrix(
    tasks: Sequence[VisibleOneEditTask] = VISIBLE_TASKS,
) -> VisibleMatrixResult:
    """Run a public development cohort without writing or opening fresh data."""
    cohort = tuple(tasks)
    if not cohort:
        raise ValueError("visible matrix needs at least one task")
    model, training = world.train_public_ast_world_model()
    results = tuple(run_visible_task(model, task) for task in cohort)
    passed_tasks = sum(result.passed for result in results)
    total_tasks = len(results)
    passed = passed_tasks == total_tasks
    return VisibleMatrixResult(
        schema="public-typed-ast-site-visible-v1",
        verdict="PASS_AST_SITE_VISIBLE" if passed else "STOP_AST_SITE_VISIBLE",
        passed_tasks=passed_tasks,
        total_tasks=total_tasks,
        selected_real_executions=sum(
            result.selected_action is not None for result in results
        ),
        candidate_count=sum(result.candidate_count for result in results),
        successor_race_count=sum(result.successor_race_count for result in results),
        selection_race_count=sum(result.selection_race_count for result in results),
        total_physical_races=sum(result.total_physical_races for result in results),
        world_checkpoint_sha256=model.checkpoint_sha256,
        target_free_observations=training.target_free_observations,
        tasks=results,
        claim_boundary=(
            "Visible public one-edit Int AST development only; no H=2, Bool/List, "
            "fresh, sealed, M0, Simple-SWE, held-out, or C1 evidence."
        ),
    )


def write_visible_artifact(
    result: VisibleMatrixResult,
    path: Path = _VISIBLE_ARTIFACT_PATH,
) -> None:
    """Persist one guarded public-development result."""
    if path.exists():
        raise FileExistsError(f"visible AST-site artifact already exists: {path}")
    path.write_bytes(_canonical_json(asdict(result)) + b"\n")


def main() -> int:
    result = run_visible_matrix()
    write_visible_artifact(result)
    print(json.dumps(asdict(result), sort_keys=True, indent=2))
    return 0 if result.verdict == "PASS_AST_SITE_VISIBLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
