"""Genuine learned-cell H=2 planning for the public scalar typed-AST seam.

The first ply spans every incumbent edit site and every generic wrap action.
For each supported first-ply root prediction, the frozen total-successor cells
predict one root-level continuation action.  Fixed synapses score the resulting
two-ply values, and a physical sequence-cell race selects the first site/action
plus the planned continuation.  No hypothetical candidate AST is built or
executed.

Only the selected first edit crosses :func:`actuate_selected_first`.  The real
edited incumbent is then observed and planned again.  Consequently, a site
change after actuation is real closed-loop replanning; it is not a claim that
the hypothetical second ply can address arbitrary future AST sites.  ``DONE``
is terminal and has no non-DONE continuation.
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

from experiments.r3_pilot import public_typed_ast_site_planner as site  # noqa: E402
from experiments.r3_pilot import public_typed_ast_world_model as world  # noqa: E402
from humanbrain import typed_program as tp  # noqa: E402

total = world.total
UNKNOWN = world.UNKNOWN
SuccessorState = world.SuccessorState
CellRace = total.banked.CellRace

DONE = site.DONE
ACTION_LABELS = site.ACTION_LABELS
STOP_ALL_UNKNOWN = site.STOP_ALL_UNKNOWN

_EDIT_COST_CURRENT_PA = 40.0
_FACTOR_MOTOR_CURRENT_PA = 5_000.0
_MAX_DECISIONS = 3

PUBLIC_ARGUMENTS = site.VISIBLE_ARGUMENTS
_VISIBLE_ARTIFACT_PATH = Path(__file__).with_name(
    "public_typed_ast_horizon2_int_visible_probe.json"
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
class Horizon2Sequence:
    first: site.SiteAction
    continuation_action: str

    @property
    def label(self) -> str:
        return f"{self.first.label}=>root::{self.continuation_action}"

    @property
    def edit_count(self) -> int:
        return int(self.first.action != DONE) + int(self.continuation_action != DONE)


@dataclass(frozen=True, slots=True)
class Horizon2Prediction:
    sequence: Horizon2Sequence
    first_predicted_root_outputs: tuple[SuccessorState, ...]
    predicted_final_outputs: tuple[SuccessorState, ...]
    supported: bool
    unsupported_reason: str | None
    first_successor_races: tuple[CellRace, ...]
    continuation_races: tuple[CellRace, ...]


def _predict_sequences(
    model: world.FrozenAstWorldModel,
    observation: world.CurrentProgramObservation,
) -> tuple[Horizon2Prediction, ...]:
    """Predict all first-site/root-continuation pairs through frozen cells."""
    first_predictions = site._predict_candidates(model, observation)
    predictions: list[Horizon2Prediction] = []
    for first in first_predictions:
        continuations = (DONE,) if first.candidate.action == DONE else ACTION_LABELS
        for continuation_action in continuations:
            if first.candidate.action == DONE:
                final_outputs = first.predicted_root_outputs
                continuation_races: tuple[CellRace, ...] = ()
            else:
                final_outputs, continuation_races = model.successor.predict_many(
                    first.predicted_root_outputs,
                    continuation_action,
                )
            supported = first.supported and all(
                state is not UNKNOWN for state in final_outputs
            )
            reason = first.unsupported_reason
            if reason is None and not supported:
                reason = "root continuation left learned successor support"
            predictions.append(
                Horizon2Prediction(
                    sequence=Horizon2Sequence(
                        first=first.candidate,
                        continuation_action=continuation_action,
                    ),
                    first_predicted_root_outputs=first.predicted_root_outputs,
                    predicted_final_outputs=final_outputs,
                    supported=supported,
                    unsupported_reason=reason,
                    first_successor_races=first.successor_races,
                    continuation_races=continuation_races,
                )
            )
    return tuple(predictions)


def _sequence_features(
    predictions: Sequence[Horizon2Prediction],
    desired_outputs: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, str]:
    bank = tuple(predictions)
    desired = tuple(desired_outputs)
    if not bank or not desired:
        raise ValueError("H=2 prediction features need candidates and desired outputs")
    states = np.zeros(
        (len(bank), len(desired), total.TOTAL_STATE_COUNT),
        dtype=np.float64,
    )
    coincidences = np.zeros_like(states)
    records: list[dict[str, object]] = []
    for sequence_cell, prediction in enumerate(bank):
        predicted = prediction.predicted_final_outputs
        if len(predicted) != len(desired):
            raise ValueError("H=2 prediction has the wrong case count")
        for case, state in enumerate(predicted):
            states[sequence_cell, case] = total.TotalValueCellEncoder.encode(state)
        coincidence, coincidence_sha = total.coincidence_features(predicted, desired)
        coincidences[sequence_cell] = coincidence
        records.append(
            {
                "sequence": prediction.sequence.label,
                "predicted": tuple(_state_label(state) for state in predicted),
                "coincidence_sha256": coincidence_sha,
            }
        )
    return (
        states,
        coincidences,
        sha256(_canonical_json(records)).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class FixedHorizon2ValueProjection:
    """Fixed value synapses plus a target-free shortest-edit tonic prior."""

    base: site.FixedSiteActionValueProjection
    edit_prior_pa: np.ndarray
    sha256: str

    @classmethod
    def create(
        cls,
        sequences: Sequence[Horizon2Sequence],
        case_count: int,
    ) -> "FixedHorizon2ValueProjection":
        bank = tuple(sequences)
        if not bank:
            raise ValueError("H=2 value projection needs sequences")
        labels = tuple(sequence.label for sequence in bank)
        if len(labels) != len(set(labels)):
            raise ValueError("H=2 value projection needs unique sequence labels")
        base = site.FixedSiteActionValueProjection.create(labels, case_count)
        prior = np.asarray(
            tuple(-_EDIT_COST_CURRENT_PA * sequence.edit_count for sequence in bank),
            dtype=np.float64,
        )
        digest = sha256()
        digest.update(base.sha256.encode("ascii"))
        digest.update(prior.tobytes())
        digest.update(b"typed-ast-horizon2-int-value-v1")
        frozen_prior = np.ascontiguousarray(prior)
        frozen_prior.setflags(write=False)
        return cls(base=base, edit_prior_pa=frozen_prior, sha256=digest.hexdigest())

    def currents(
        self,
        state_spikes: np.ndarray,
        coincidence_spikes: np.ndarray,
    ) -> np.ndarray:
        currents = self.base.currents(state_spikes, coincidence_spikes)
        adjusted = np.ascontiguousarray(currents.copy())
        adjusted[:-1] += self.edit_prior_pa
        return adjusted


@dataclass(frozen=True, slots=True)
class FixedHorizon2FactorProjection:
    """One-hot sequence-cell projections into the three planned factors."""

    site_paths: tuple[tuple[int, ...], ...]
    action_labels: tuple[str, ...]
    first_site_weights: np.ndarray
    first_action_weights: np.ndarray
    continuation_weights: np.ndarray
    sha256: str

    @classmethod
    def create(
        cls,
        sequences: Sequence[Horizon2Sequence],
        site_paths: Sequence[tuple[int, ...]],
    ) -> "FixedHorizon2FactorProjection":
        bank = tuple(sequences)
        paths = tuple(site_paths)
        if not bank or not paths or len(paths) != len(set(paths)):
            raise ValueError("H=2 factor projection needs candidates and unique sites")
        path_index = {path: index for index, path in enumerate(paths)}
        action_index = {action: index for index, action in enumerate(ACTION_LABELS)}
        first_sites = np.zeros((len(paths), len(bank)), dtype=np.float64)
        first_actions = np.zeros((len(ACTION_LABELS), len(bank)), dtype=np.float64)
        continuations = np.zeros_like(first_actions)
        for sequence_cell, sequence in enumerate(bank):
            first_sites[path_index[sequence.first.path], sequence_cell] = 1.0
            first_actions[action_index[sequence.first.action], sequence_cell] = 1.0
            continuations[
                action_index[sequence.continuation_action], sequence_cell
            ] = 1.0
        digest = sha256()
        digest.update(first_sites.tobytes())
        digest.update(first_actions.tobytes())
        digest.update(continuations.tobytes())
        digest.update(_canonical_json(tuple(sequence.label for sequence in bank)))
        frozen_arrays = tuple(
            np.ascontiguousarray(array)
            for array in (first_sites, first_actions, continuations)
        )
        for array in frozen_arrays:
            array.setflags(write=False)
        return cls(
            site_paths=paths,
            action_labels=ACTION_LABELS,
            first_site_weights=frozen_arrays[0],
            first_action_weights=frozen_arrays[1],
            continuation_weights=frozen_arrays[2],
            sha256=digest.hexdigest(),
        )

    def races(self, sequence_cell: int) -> tuple[CellRace, CellRace, CellRace]:
        candidate_count = self.first_site_weights.shape[1]
        if not 0 <= sequence_cell < candidate_count:
            raise ValueError("H=2 factor projection candidate is outside the bank")
        spikes = np.zeros(candidate_count, dtype=np.float64)
        spikes[sequence_cell] = 1.0
        site_currents = np.ascontiguousarray(
            _FACTOR_MOTOR_CURRENT_PA * (self.first_site_weights @ spikes)
        )
        first_action_currents = np.ascontiguousarray(
            _FACTOR_MOTOR_CURRENT_PA * (self.first_action_weights @ spikes)
        )
        continuation_currents = np.ascontiguousarray(
            _FACTOR_MOTOR_CURRENT_PA * (self.continuation_weights @ spikes)
        )
        return (
            total.banked._physical_race(
                site_currents,
                tuple(_path_label(path) for path in self.site_paths),
                bank="ast-h2-first-site-motor",
            ),
            total.banked._physical_race(
                first_action_currents,
                ACTION_LABELS,
                bank="ast-h2-first-action-motor",
            ),
            total.banked._physical_race(
                continuation_currents,
                ACTION_LABELS,
                bank="ast-h2-continuation-motor",
            ),
        )


@dataclass(frozen=True, slots=True)
class Horizon2Decision:
    status: str
    selected_first_path: tuple[int, ...] | None
    selected_first_action: str | None
    selected_continuation_action: str | None
    predicted_final_outputs: tuple[SuccessorState, ...] | None
    sequence_race: CellRace
    first_site_motor_race: CellRace | None
    first_action_motor_race: CellRace | None
    continuation_motor_race: CellRace | None
    predictions: tuple[Horizon2Prediction, ...]
    observation_sha256: str
    program_sha256: str
    feature_sha256: str
    value_projection_sha256: str
    factor_projection_sha256: str | None
    sequence_digest: str
    world_checkpoint_before_sha256: str
    world_checkpoint_after_sha256: str


def plan_site_action_h2(
    model: world.FrozenAstWorldModel,
    observation: world.CurrentProgramObservation,
    desired_outputs: Sequence[int],
) -> Horizon2Decision:
    """Select the first edit from genuine learned-cell two-ply predictions."""
    desired = tuple(desired_outputs)
    if not desired or len(desired) != len(observation.arguments):
        raise ValueError("H=2 site planning needs paired desired outputs")
    if not all(
        type(value) is int
        and total.banked.VALUE_MIN <= value <= total.banked.VALUE_MAX
        for value in desired
    ):
        raise ValueError("desired outputs must fit the fixed raw-value population")

    checkpoint_before = model.live_checkpoint_sha256
    if checkpoint_before != model.checkpoint_sha256:
        raise RuntimeError("AST world-model checkpoint changed before H=2 planning")
    predictions = _predict_sequences(model, observation)
    sequences = tuple(prediction.sequence for prediction in predictions)
    states, coincidences, feature_sha = _sequence_features(predictions, desired)
    projection = FixedHorizon2ValueProjection.create(sequences, len(desired))
    currents = projection.currents(states, coincidences)
    labels = tuple(sequence.label for sequence in sequences)
    race = total.banked._physical_race(
        currents,
        (*labels, STOP_ALL_UNKNOWN),
        bank="ast-h2-sequence-value",
    )
    checkpoint_after = model.live_checkpoint_sha256
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("frozen AST world model changed during H=2 planning")
    sequence_digest = sha256(
        _canonical_json(
            tuple(
                {
                    "sequence": prediction.sequence.label,
                    "first": tuple(
                        _state_label(state)
                        for state in prediction.first_predicted_root_outputs
                    ),
                    "final": tuple(
                        _state_label(state)
                        for state in prediction.predicted_final_outputs
                    ),
                    "supported": prediction.supported,
                }
                for prediction in predictions
            )
        )
    ).hexdigest()

    if race.winner == STOP_ALL_UNKNOWN:
        return Horizon2Decision(
            status=STOP_ALL_UNKNOWN,
            selected_first_path=None,
            selected_first_action=None,
            selected_continuation_action=None,
            predicted_final_outputs=None,
            sequence_race=race,
            first_site_motor_race=None,
            first_action_motor_race=None,
            continuation_motor_race=None,
            predictions=predictions,
            observation_sha256=observation.sha256,
            program_sha256=observation.program_sha256,
            feature_sha256=feature_sha,
            value_projection_sha256=projection.sha256,
            factor_projection_sha256=None,
            sequence_digest=sequence_digest,
            world_checkpoint_before_sha256=checkpoint_before,
            world_checkpoint_after_sha256=checkpoint_after,
        )

    selected = predictions[race.winner_cell]
    factors = FixedHorizon2FactorProjection.create(
        sequences,
        tuple(observed_site.path for observed_site in observation.sites),
    )
    site_race, first_action_race, continuation_race = factors.races(race.winner_cell)
    if site_race.winner != _path_label(selected.sequence.first.path):
        raise RuntimeError("H=2 sequence-to-site projection lost factor parity")
    if first_action_race.winner != selected.sequence.first.action:
        raise RuntimeError("H=2 sequence-to-first-action projection lost factor parity")
    if continuation_race.winner != selected.sequence.continuation_action:
        raise RuntimeError("H=2 sequence-to-continuation projection lost factor parity")
    return Horizon2Decision(
        status="SELECTED",
        selected_first_path=selected.sequence.first.path,
        selected_first_action=selected.sequence.first.action,
        selected_continuation_action=selected.sequence.continuation_action,
        predicted_final_outputs=selected.predicted_final_outputs,
        sequence_race=race,
        first_site_motor_race=site_race,
        first_action_motor_race=first_action_race,
        continuation_motor_race=continuation_race,
        predictions=predictions,
        observation_sha256=observation.sha256,
        program_sha256=observation.program_sha256,
        feature_sha256=feature_sha,
        value_projection_sha256=projection.sha256,
        factor_projection_sha256=factors.sha256,
        sequence_digest=sequence_digest,
        world_checkpoint_before_sha256=checkpoint_before,
        world_checkpoint_after_sha256=checkpoint_after,
    )


@dataclass(frozen=True, slots=True)
class FirstActuation:
    path: tuple[int, ...]
    action: str
    program_before_sha256: str
    program_after: tp.Program
    actual_root_outputs: tuple[int, ...]
    observation_sha256: str


def actuate_selected_first(
    program: tp.Program,
    decision: Horizon2Decision,
    arguments: Sequence[int],
) -> FirstActuation:
    """Apply only the selected first edit and observe that real incumbent."""
    if decision.status != "SELECTED":
        raise RuntimeError("an all-unknown H=2 STOP has no actuator command")
    if decision.selected_first_path is None or decision.selected_first_action is None:
        raise RuntimeError("selected H=2 decision is missing first-ply factors")
    if decision.selected_first_action == DONE:
        raise RuntimeError("DONE is terminal and cannot cross the edit actuator")
    if program.sha256 != decision.program_sha256:
        raise RuntimeError("H=2 actuator program differs from the planned incumbent")
    edited = world.apply_selected_wrap(
        program,
        decision.selected_first_path,
        decision.selected_first_action,
    )
    observed = world.observe_current_program(edited, arguments)
    outputs = observed.site(()).outputs
    if outputs is None:
        raise RuntimeError("selected real H=2 edit did not produce scalar Int outputs")
    return FirstActuation(
        path=decision.selected_first_path,
        action=decision.selected_first_action,
        program_before_sha256=program.sha256,
        program_after=edited,
        actual_root_outputs=outputs,
        observation_sha256=observed.sha256,
    )


@dataclass(frozen=True, slots=True)
class PublicHorizon2Task:
    name: str
    incumbent_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]
    desired_outputs: tuple[int, ...]
    expected_decisions: tuple[tuple[tuple[int, ...], str], ...]


_LITERAL_1 = ("integer", "+", "1", "end")
_LITERAL_2 = ("integer", "+", "2", "end")

PUBLIC_TASKS = (
    PublicHorizon2Task(
        name="absolute-then-add",
        incumbent_tokens=("variable-0",),
        target_tokens=("add", "absolute", "variable-0", *_LITERAL_1),
        desired_outputs=(9, 4, 1, 3, 8, 10),
        expected_decisions=(
            ((), "WRAP_ABS"),
            ((), "WRAP_ADD[1]"),
            ((), DONE),
        ),
    ),
    PublicHorizon2Task(
        name="add-then-absolute",
        incumbent_tokens=("variable-0",),
        target_tokens=("absolute", "add", "variable-0", *_LITERAL_1),
        desired_outputs=(7, 2, 1, 3, 8, 10),
        expected_decisions=(
            ((), "WRAP_ADD[1]"),
            ((), "WRAP_ABS"),
            ((), DONE),
        ),
    ),
    PublicHorizon2Task(
        name="child-then-root-replan",
        incumbent_tokens=("absolute", "variable-0"),
        target_tokens=(
            "add",
            "absolute",
            "add",
            "variable-0",
            *_LITERAL_1,
            *_LITERAL_2,
        ),
        desired_outputs=(9, 4, 3, 5, 10, 12),
        expected_decisions=(
            ((0,), "WRAP_ADD[1]"),
            ((), "WRAP_ADD[2]"),
            ((), DONE),
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class PublicDecisionRecord:
    path: tuple[int, ...]
    action: str
    continuation_action: str
    predicted_final_outputs: tuple[SuccessorState, ...]
    actual_tokens_after: tuple[str, ...]
    actual_outputs_after: tuple[int, ...]
    sequence_count: int
    physical_unique: bool
    sequence_digest: str


@dataclass(frozen=True, slots=True)
class PublicTaskResult:
    name: str
    passed: bool
    verdict: str
    error: str | None
    expected_decisions: tuple[tuple[tuple[int, ...], str], ...]
    actual_decisions: tuple[tuple[tuple[int, ...], str], ...]
    final_tokens: tuple[str, ...]
    final_outputs: tuple[int, ...]
    decisions: tuple[PublicDecisionRecord, ...]
    checkpoint_immutable: bool


@dataclass(frozen=True, slots=True)
class PublicMatrixResult:
    schema: str
    verdict: str
    passed_tasks: int
    total_tasks: int
    world_checkpoint_sha256: str
    target_free_observations: int
    tasks: tuple[PublicTaskResult, ...]
    claim_boundary: str


def _validate_public_target(task: PublicHorizon2Task) -> None:
    target = tp.Program(task.target_tokens, site.visible_scalar_signature())
    outputs = world.observe_current_program(target, PUBLIC_ARGUMENTS).site(()).outputs
    if outputs != task.desired_outputs:
        raise RuntimeError(f"public H=2 task {task.name!r} has an invalid target")


def _decision_races(decision: Horizon2Decision) -> tuple[CellRace, ...]:
    return (
        *(
            race
            for prediction in decision.predictions
            for race in (
                *prediction.first_successor_races,
                *prediction.continuation_races,
            )
        ),
        decision.sequence_race,
        *((decision.first_site_motor_race,) if decision.first_site_motor_race else ()),
        *((decision.first_action_motor_race,) if decision.first_action_motor_race else ()),
        *((decision.continuation_motor_race,) if decision.continuation_motor_race else ()),
    )


def run_public_task(
    model: world.FrozenAstWorldModel,
    task: PublicHorizon2Task,
) -> PublicTaskResult:
    """Run two selected edits plus terminal DONE through act/observe/replan."""
    _validate_public_target(task)
    program = tp.Program(task.incumbent_tokens, site.visible_scalar_signature())
    records: list[PublicDecisionRecord] = []
    actual_decisions: list[tuple[tuple[int, ...], str]] = []
    checkpoint_before = model.live_checkpoint_sha256
    error: str | None = None

    for _ in range(_MAX_DECISIONS):
        observation = world.observe_current_program(program, PUBLIC_ARGUMENTS)
        decision = plan_site_action_h2(model, observation, task.desired_outputs)
        if decision.status == STOP_ALL_UNKNOWN:
            error = STOP_ALL_UNKNOWN
            break
        assert decision.selected_first_path is not None
        assert decision.selected_first_action is not None
        assert decision.selected_continuation_action is not None
        actual_decisions.append(
            (decision.selected_first_path, decision.selected_first_action)
        )
        races = _decision_races(decision)
        physical_unique = all(
            race.simultaneous_spike_candidates == 1 for race in races
        )
        if decision.selected_first_action == DONE:
            root_outputs = observation.site(()).outputs
            if root_outputs is None:
                error = "terminal incumbent lost scalar Int observation"
                break
            records.append(
                PublicDecisionRecord(
                    path=decision.selected_first_path,
                    action=DONE,
                    continuation_action=decision.selected_continuation_action,
                    predicted_final_outputs=decision.predicted_final_outputs or (),
                    actual_tokens_after=program.tokens,
                    actual_outputs_after=root_outputs,
                    sequence_count=len(decision.predictions),
                    physical_unique=physical_unique,
                    sequence_digest=decision.sequence_digest,
                )
            )
            break

        actuation = actuate_selected_first(program, decision, PUBLIC_ARGUMENTS)
        program = actuation.program_after
        records.append(
            PublicDecisionRecord(
                path=actuation.path,
                action=actuation.action,
                continuation_action=decision.selected_continuation_action,
                predicted_final_outputs=decision.predicted_final_outputs or (),
                actual_tokens_after=program.tokens,
                actual_outputs_after=actuation.actual_root_outputs,
                sequence_count=len(decision.predictions),
                physical_unique=physical_unique,
                sequence_digest=decision.sequence_digest,
            )
        )
    else:
        error = "public H=2 task exhausted the fixed decision bound"

    final_observation = world.observe_current_program(program, PUBLIC_ARGUMENTS)
    final_outputs = final_observation.site(()).outputs
    if final_outputs is None:
        raise RuntimeError("public H=2 runner lost scalar final outputs")
    checkpoint_immutable = checkpoint_before == model.live_checkpoint_sha256
    actual = tuple(actual_decisions)
    passed = (
        error is None
        and actual == task.expected_decisions
        and program.tokens == task.target_tokens
        and final_outputs == task.desired_outputs
        and all(record.physical_unique for record in records)
        and checkpoint_immutable
    )
    return PublicTaskResult(
        name=task.name,
        passed=passed,
        verdict="PASS" if passed else "STOP",
        error=None if passed else (error or "physical H=2 trajectory missed target"),
        expected_decisions=task.expected_decisions,
        actual_decisions=actual,
        final_tokens=program.tokens,
        final_outputs=final_outputs,
        decisions=tuple(records),
        checkpoint_immutable=checkpoint_immutable,
    )


def run_public_matrix(
    tasks: Sequence[PublicHorizon2Task] = PUBLIC_TASKS,
) -> PublicMatrixResult:
    """Run the small visible H=2 Int matrix; never writes an artifact."""
    cohort = tuple(tasks)
    if not cohort:
        raise ValueError("public H=2 matrix needs at least one task")
    model, training = world.train_public_ast_world_model()
    results = tuple(run_public_task(model, task) for task in cohort)
    passed_tasks = sum(result.passed for result in results)
    return PublicMatrixResult(
        schema="public-typed-ast-horizon2-int-v1",
        verdict=(
            "PASS_TYPED_AST_HORIZON2_INT"
            if passed_tasks == len(results)
            else "STOP_TYPED_AST_HORIZON2_INT"
        ),
        passed_tasks=passed_tasks,
        total_tasks=len(results),
        world_checkpoint_sha256=model.checkpoint_sha256,
        target_free_observations=training.target_free_observations,
        tasks=results,
        claim_boundary=(
            "Visible public Int exact-value H=2 development only. The genuine "
            "second ply is restricted to a root wrap over the first learned-cell "
            "root prediction; all incumbent sites are first-ply candidates, and "
            "site changes happen only after selected real actuation/re-observation. "
            "No arbitrary hypothetical second site, Bool/List, fresh, sealed, M0, "
            "Simple-SWE, held-out, or C1 evidence."
        ),
    )


def write_visible_artifact(
    result: PublicMatrixResult,
    path: Path = _VISIBLE_ARTIFACT_PATH,
) -> None:
    """Persist one guarded public H=2 development result."""
    if path.exists():
        raise FileExistsError(f"visible AST H=2 artifact already exists: {path}")
    path.write_bytes(_canonical_json(asdict(result)) + b"\n")


def main() -> int:
    result = run_public_matrix()
    write_visible_artifact(result)
    print(json.dumps(asdict(result), sort_keys=True, indent=2))
    return 0 if result.verdict == "PASS_TYPED_AST_HORIZON2_INT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
