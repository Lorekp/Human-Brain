"""B=4 physical first-ply filter for the public typed-AST H=2 planner.

This is an optimized development variant, not a replacement for the frozen
exhaustive evidence module.  Existing fixed value synapses score every
incumbent site/action on its learned intermediate root values.  A physical
resting-GLIF beam retains four complete first-ply candidates, and only those
candidates receive learned root-continuation predictions.  Terminal sequence
scoring and factor projection reuse the exhaustive H=2 mechanisms.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for search_path in (REPO_ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.r3_pilot import public_typed_ast_horizon2_int_planner as h2  # noqa: E402
from experiments.r3_pilot import public_typed_ast_site_planner as site  # noqa: E402
from experiments.r3_pilot import public_typed_ast_world_model as world  # noqa: E402
from humanbrain import typed_program as tp  # noqa: E402
from humanbrain.resting_glif_beam import (  # noqa: E402
    CompiledRestingGlifBeam,
    RestingGlifBeamResult,
    RestingGlifRaceFailure,
)

total = world.total
UNKNOWN = world.UNKNOWN
SuccessorState = world.SuccessorState
CellRace = total.banked.CellRace

DONE = h2.DONE
ACTION_LABELS = h2.ACTION_LABELS
STOP_ALL_UNKNOWN = h2.STOP_ALL_UNKNOWN
STOP_FIRST_PLY_BEAM_RACE = "STOP_FIRST_PLY_BEAM_RACE"
STOP_TERMINAL_SEQUENCE_RACE = "STOP_TERMINAL_SEQUENCE_RACE"

BEAM_WIDTH = 4
_FIRST_PLY_MIN_CURRENT_PA = -10_000.0
_FIRST_PLY_MAX_CURRENT_PA = 2_200.0
_FIRST_PLY_INHIBITION_CURRENT_PA = 13_000.0
_MAX_DECISIONS = 3

PUBLIC_ARGUMENTS = h2.PUBLIC_ARGUMENTS
PUBLIC_TASKS = h2.PUBLIC_TASKS


def _validate_desired(
    observation: world.CurrentProgramObservation,
    desired_outputs: Sequence[int],
) -> tuple[int, ...]:
    desired = tuple(desired_outputs)
    if len(desired) != len(PUBLIC_ARGUMENTS) or len(desired) != len(
        observation.arguments
    ):
        raise ValueError("beam H=2 planning needs the fixed six paired outputs")
    if not all(
        type(value) is int
        and total.banked.VALUE_MIN <= value <= total.banked.VALUE_MAX
        for value in desired
    ):
        raise ValueError("desired outputs must fit the fixed raw-value population")
    return desired


@dataclass(frozen=True, slots=True)
class FirstPlyBeamAudit:
    status: str
    error: str | None
    score_race: CellRace | None
    beam: RestingGlifBeamResult | None
    first_prediction_count: int
    retained_first_count: int
    feature_sha256: str
    value_projection_sha256: str


def _score_and_filter_first_ply(
    model: world.FrozenAstWorldModel,
    observation: world.CurrentProgramObservation,
    desired_outputs: Sequence[int],
) -> tuple[
    tuple[site.CandidatePrediction, ...],
    tuple[site.CandidatePrediction, ...],
    FirstPlyBeamAudit,
]:
    """Physically retain B first actions from intermediate typed values."""

    predictions = site._predict_candidates(model, observation)
    state_spikes, coincidence_spikes, feature_sha = site._prediction_features(
        predictions,
        desired_outputs,
    )
    labels = tuple(prediction.candidate.label for prediction in predictions)
    projection = site.FixedSiteActionValueProjection.create(
        labels,
        len(tuple(desired_outputs)),
    )
    scored_currents = projection.currents(state_spikes, coincidence_spikes)
    try:
        score_race = total.banked._physical_race(
            scored_currents,
            (*labels, STOP_ALL_UNKNOWN),
            bank="ast-h2-beam-first-value-stop",
        )
    except total.banked.CellRaceFailure as error:
        return predictions, (), FirstPlyBeamAudit(
            status=STOP_FIRST_PLY_BEAM_RACE,
            error=str(error),
            score_race=None,
            beam=None,
            first_prediction_count=len(predictions),
            retained_first_count=0,
            feature_sha256=feature_sha,
            value_projection_sha256=projection.sha256,
        )
    if score_race.winner == STOP_ALL_UNKNOWN:
        return predictions, (), FirstPlyBeamAudit(
            status=STOP_ALL_UNKNOWN,
            error=STOP_ALL_UNKNOWN,
            score_race=score_race,
            beam=None,
            first_prediction_count=len(predictions),
            retained_first_count=0,
            feature_sha256=feature_sha,
            value_projection_sha256=projection.sha256,
        )

    selector = CompiledRestingGlifBeam.compile(
        candidate_count=len(predictions),
        beam_width=BEAM_WIDTH,
        step_count=total.banked._RACE_STEPS,
        min_current_pa=_FIRST_PLY_MIN_CURRENT_PA,
        max_current_pa=_FIRST_PLY_MAX_CURRENT_PA,
        inhibition_current_pa=_FIRST_PLY_INHIBITION_CURRENT_PA,
    )
    try:
        beam = selector.select(scored_currents[:-1])
    except RestingGlifRaceFailure as error:
        return predictions, (), FirstPlyBeamAudit(
            status=STOP_FIRST_PLY_BEAM_RACE,
            error=str(error),
            score_race=score_race,
            beam=None,
            first_prediction_count=len(predictions),
            retained_first_count=0,
            feature_sha256=feature_sha,
            value_projection_sha256=projection.sha256,
        )
    retained = tuple(predictions[cell] for cell in beam.winner_cells)
    return predictions, retained, FirstPlyBeamAudit(
        status="RETAINED",
        error=None,
        score_race=score_race,
        beam=beam,
        first_prediction_count=len(predictions),
        retained_first_count=len(retained),
        feature_sha256=feature_sha,
        value_projection_sha256=projection.sha256,
    )


def _expand_retained_second_ply(
    model: world.FrozenAstWorldModel,
    retained: Sequence[site.CandidatePrediction],
) -> tuple[tuple[h2.Horizon2Prediction, ...], int]:
    predictions: list[h2.Horizon2Prediction] = []
    transition_expansions = 0
    for first in retained:
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
                transition_expansions += 1
            supported = first.supported and all(
                state is not UNKNOWN for state in final_outputs
            )
            reason = first.unsupported_reason
            if reason is None and not supported:
                reason = "retained root continuation left learned successor support"
            predictions.append(
                h2.Horizon2Prediction(
                    sequence=h2.Horizon2Sequence(
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
    return tuple(predictions), transition_expansions


def _sequence_digest(predictions: Sequence[h2.Horizon2Prediction]) -> str:
    return sha256(
        h2._canonical_json(
            tuple(
                {
                    "sequence": prediction.sequence.label,
                    "first": tuple(
                        h2._state_label(state)
                        for state in prediction.first_predicted_root_outputs
                    ),
                    "final": tuple(
                        h2._state_label(state)
                        for state in prediction.predicted_final_outputs
                    ),
                    "supported": prediction.supported,
                }
                for prediction in predictions
            )
        )
    ).hexdigest()


def _score_terminal_sequences(
    model: world.FrozenAstWorldModel,
    observation: world.CurrentProgramObservation,
    predictions: Sequence[h2.Horizon2Prediction],
    desired_outputs: Sequence[int],
    checkpoint_before: str,
) -> h2.Horizon2Decision:
    bank = tuple(predictions)
    sequences = tuple(prediction.sequence for prediction in bank)
    states, coincidences, feature_sha = h2._sequence_features(bank, desired_outputs)
    projection = h2.FixedHorizon2ValueProjection.create(
        sequences,
        len(tuple(desired_outputs)),
    )
    currents = projection.currents(states, coincidences)
    labels = tuple(sequence.label for sequence in sequences)
    race = total.banked._physical_race(
        currents,
        (*labels, STOP_ALL_UNKNOWN),
        bank="ast-h2-beam-terminal-sequence-value",
    )
    checkpoint_after = model.live_checkpoint_sha256
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("frozen AST world model changed during beam H=2 planning")
    digest = _sequence_digest(bank)
    if race.winner == STOP_ALL_UNKNOWN:
        return h2.Horizon2Decision(
            status=STOP_ALL_UNKNOWN,
            selected_first_path=None,
            selected_first_action=None,
            selected_continuation_action=None,
            predicted_final_outputs=None,
            sequence_race=race,
            first_site_motor_race=None,
            first_action_motor_race=None,
            continuation_motor_race=None,
            predictions=bank,
            observation_sha256=observation.sha256,
            program_sha256=observation.program_sha256,
            feature_sha256=feature_sha,
            value_projection_sha256=projection.sha256,
            factor_projection_sha256=None,
            sequence_digest=digest,
            world_checkpoint_before_sha256=checkpoint_before,
            world_checkpoint_after_sha256=checkpoint_after,
        )

    selected = bank[race.winner_cell]
    factors = h2.FixedHorizon2FactorProjection.create(
        sequences,
        tuple(observed_site.path for observed_site in observation.sites),
    )
    site_race, first_action_race, continuation_race = factors.races(race.winner_cell)
    if site_race.winner != h2._path_label(selected.sequence.first.path):
        raise RuntimeError("beam sequence-to-site projection lost factor parity")
    if first_action_race.winner != selected.sequence.first.action:
        raise RuntimeError("beam sequence-to-first-action projection lost parity")
    if continuation_race.winner != selected.sequence.continuation_action:
        raise RuntimeError("beam sequence-to-continuation projection lost parity")
    return h2.Horizon2Decision(
        status="SELECTED",
        selected_first_path=selected.sequence.first.path,
        selected_first_action=selected.sequence.first.action,
        selected_continuation_action=selected.sequence.continuation_action,
        predicted_final_outputs=selected.predicted_final_outputs,
        sequence_race=race,
        first_site_motor_race=site_race,
        first_action_motor_race=first_action_race,
        continuation_motor_race=continuation_race,
        predictions=bank,
        observation_sha256=observation.sha256,
        program_sha256=observation.program_sha256,
        feature_sha256=feature_sha,
        value_projection_sha256=projection.sha256,
        factor_projection_sha256=factors.sha256,
        sequence_digest=digest,
        world_checkpoint_before_sha256=checkpoint_before,
        world_checkpoint_after_sha256=checkpoint_after,
    )


@dataclass(frozen=True, slots=True)
class BeamHorizon2Decision:
    status: str
    error: str | None
    first_ply: FirstPlyBeamAudit
    terminal: h2.Horizon2Decision | None
    second_ply_transition_expansions: int
    terminal_sequence_count: int


def plan_site_action_h2_beam(
    model: world.FrozenAstWorldModel,
    observation: world.CurrentProgramObservation,
    desired_outputs: Sequence[int],
) -> BeamHorizon2Decision:
    """Filter first actions physically, then perform genuine H=2 scoring."""

    desired = _validate_desired(observation, desired_outputs)
    checkpoint_before = model.live_checkpoint_sha256
    if checkpoint_before != model.checkpoint_sha256:
        raise RuntimeError("AST world-model checkpoint changed before beam planning")
    _, retained, first_ply = _score_and_filter_first_ply(
        model,
        observation,
        desired,
    )
    if first_ply.status != "RETAINED":
        return BeamHorizon2Decision(
            status=first_ply.status,
            error=first_ply.error,
            first_ply=first_ply,
            terminal=None,
            second_ply_transition_expansions=0,
            terminal_sequence_count=0,
        )
    predictions, expansion_count = _expand_retained_second_ply(model, retained)
    try:
        terminal = _score_terminal_sequences(
            model,
            observation,
            predictions,
            desired,
            checkpoint_before,
        )
    except total.banked.CellRaceFailure as error:
        return BeamHorizon2Decision(
            status=STOP_TERMINAL_SEQUENCE_RACE,
            error=str(error),
            first_ply=first_ply,
            terminal=None,
            second_ply_transition_expansions=expansion_count,
            terminal_sequence_count=len(predictions),
        )
    return BeamHorizon2Decision(
        status=terminal.status,
        error=None if terminal.status == "SELECTED" else terminal.status,
        first_ply=first_ply,
        terminal=terminal,
        second_ply_transition_expansions=expansion_count,
        terminal_sequence_count=len(predictions),
    )


@dataclass(frozen=True, slots=True)
class BeamPlanRecord:
    status: str
    error: str | None
    first_prediction_count: int
    retained_first_count: int
    second_ply_transition_expansions: int
    terminal_sequence_count: int
    selected_path: tuple[int, ...] | None
    selected_action: str | None
    continuation_action: str | None


@dataclass(frozen=True, slots=True)
class BeamPublicTaskResult:
    name: str
    passed: bool
    verdict: str
    error: str | None
    actual_decisions: tuple[tuple[tuple[int, ...], str], ...]
    final_tokens: tuple[str, ...]
    final_outputs: tuple[int, ...]
    plans: tuple[BeamPlanRecord, ...]
    checkpoint_immutable: bool


@dataclass(frozen=True, slots=True)
class BeamPublicMatrixResult:
    schema: str
    verdict: str
    passed_tasks: int
    total_tasks: int
    first_predictions_scored: int
    retained_first_candidates: int
    second_ply_transition_expansions: int
    terminal_sequence_count: int
    wall_time_seconds: float
    world_checkpoint_sha256: str
    target_free_observations: int
    tasks: tuple[BeamPublicTaskResult, ...]
    claim_boundary: str


def run_public_task(
    model: world.FrozenAstWorldModel,
    task: h2.PublicHorizon2Task,
) -> BeamPublicTaskResult:
    h2._validate_public_target(task)
    program = tp.Program(task.incumbent_tokens, site.visible_scalar_signature())
    actual_decisions: list[tuple[tuple[int, ...], str]] = []
    plans: list[BeamPlanRecord] = []
    checkpoint_before = model.live_checkpoint_sha256
    error: str | None = None

    for _ in range(_MAX_DECISIONS):
        observation = world.observe_current_program(program, PUBLIC_ARGUMENTS)
        decision = plan_site_action_h2_beam(model, observation, task.desired_outputs)
        terminal = decision.terminal
        plans.append(
            BeamPlanRecord(
                status=decision.status,
                error=decision.error,
                first_prediction_count=decision.first_ply.first_prediction_count,
                retained_first_count=decision.first_ply.retained_first_count,
                second_ply_transition_expansions=(
                    decision.second_ply_transition_expansions
                ),
                terminal_sequence_count=decision.terminal_sequence_count,
                selected_path=(
                    None if terminal is None else terminal.selected_first_path
                ),
                selected_action=(
                    None if terminal is None else terminal.selected_first_action
                ),
                continuation_action=(
                    None if terminal is None else terminal.selected_continuation_action
                ),
            )
        )
        if decision.status != "SELECTED" or terminal is None:
            error = f"{decision.status}: {decision.error}"
            break
        assert terminal.selected_first_path is not None
        assert terminal.selected_first_action is not None
        actual_decisions.append(
            (terminal.selected_first_path, terminal.selected_first_action)
        )
        if terminal.selected_first_action == DONE:
            break
        actuation = h2.actuate_selected_first(program, terminal, PUBLIC_ARGUMENTS)
        program = actuation.program_after
    else:
        error = "beam H=2 task exhausted the fixed decision bound"

    final = world.observe_current_program(program, PUBLIC_ARGUMENTS).site(()).outputs
    if final is None:
        raise RuntimeError("beam H=2 runner lost scalar final outputs")
    actual = tuple(actual_decisions)
    checkpoint_immutable = checkpoint_before == model.live_checkpoint_sha256
    passed = (
        error is None
        and actual == task.expected_decisions
        and program.tokens == task.target_tokens
        and final == task.desired_outputs
        and checkpoint_immutable
    )
    return BeamPublicTaskResult(
        name=task.name,
        passed=passed,
        verdict="PASS" if passed else "STOP",
        error=None if passed else (error or "beam trajectory missed public target"),
        actual_decisions=actual,
        final_tokens=program.tokens,
        final_outputs=final,
        plans=tuple(plans),
        checkpoint_immutable=checkpoint_immutable,
    )


def run_public_matrix(
    tasks: Sequence[h2.PublicHorizon2Task] = PUBLIC_TASKS,
) -> BeamPublicMatrixResult:
    """Run the visible B=4 development gate without writing an artifact."""

    cohort = tuple(tasks)
    if not cohort:
        raise ValueError("beam public matrix needs at least one task")
    started = perf_counter()
    model, training = world.train_public_ast_world_model()
    results = tuple(run_public_task(model, task) for task in cohort)
    elapsed = perf_counter() - started
    passed_tasks = sum(task.passed for task in results)
    plans = tuple(plan for task in results for plan in task.plans)
    return BeamPublicMatrixResult(
        schema="public-typed-ast-horizon2-int-beam-v1",
        verdict=(
            "PASS_TYPED_AST_HORIZON2_INT_BEAM"
            if passed_tasks == len(results)
            else "STOP_TYPED_AST_HORIZON2_INT_BEAM"
        ),
        passed_tasks=passed_tasks,
        total_tasks=len(results),
        first_predictions_scored=sum(plan.first_prediction_count for plan in plans),
        retained_first_candidates=sum(plan.retained_first_count for plan in plans),
        second_ply_transition_expansions=sum(
            plan.second_ply_transition_expansions for plan in plans
        ),
        terminal_sequence_count=sum(plan.terminal_sequence_count for plan in plans),
        wall_time_seconds=elapsed,
        world_checkpoint_sha256=model.checkpoint_sha256,
        target_free_observations=training.target_free_observations,
        tasks=results,
        claim_boundary=(
            "Visible public Int B=4 optimization only: all incumbent first sites "
            "are physically scored on intermediate learned values, while only "
            "retained first plies receive root-wrap continuations. No frozen H=2 "
            "evidence replacement, arbitrary hypothetical second site, Bool/List, "
            "fresh, sealed, M0, Simple-SWE, held-out, or C1 evidence."
        ),
    )
