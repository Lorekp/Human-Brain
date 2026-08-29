"""Uncertainty-driven anytime expansion for the public typed-AST H=2 planner.

The fixed first-ply value currents are physically enumerated into complete
resting-GLIF spike tiers once, with STOP in the same population.  Learned
second-ply predictions are cached per first candidate.  Terminal sequence
scoring starts when at least B=4 candidates have fired and consumes one more
complete first-ply tier only when the physical terminal race is non-unique or
its unique winner lacks an exact-coincidence spike for every desired case.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for search_path in (REPO_ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.r3_pilot import public_typed_ast_horizon2_int_beam as strict  # noqa: E402
from experiments.r3_pilot import public_typed_ast_horizon2_int_planner as h2  # noqa: E402
from experiments.r3_pilot import public_typed_ast_horizon2_int_tier_beam as tier  # noqa: E402
from experiments.r3_pilot import public_typed_ast_site_planner as site  # noqa: E402
from experiments.r3_pilot import public_typed_ast_world_model as world  # noqa: E402
from humanbrain import typed_program as tp  # noqa: E402
from humanbrain.resting_glif_tier_beam import (  # noqa: E402
    CompiledRestingGlifTierBeam,
    RETAINED,
    RestingGlifRaceFailure,
    RestingGlifTierBeamResult,
    STOP_LEADING_TIER,
)

total = world.total
CellRace = total.banked.CellRace

DONE = h2.DONE
STOP_ALL_UNKNOWN = h2.STOP_ALL_UNKNOWN
STOP_FIRST_PLY_TIER_ENUMERATION = "STOP_FIRST_PLY_TIER_ENUMERATION"
STOP_ANYTIME_EXHAUSTED = "STOP_ANYTIME_EXHAUSTED"

# Frozen aliases: uncertainty controls expansion depth, never calibration.
BEAM_WIDTH = tier.BEAM_WIDTH
_FIRST_PLY_MIN_CURRENT_PA = tier._FIRST_PLY_MIN_CURRENT_PA
_FIRST_PLY_MAX_CURRENT_PA = tier._FIRST_PLY_MAX_CURRENT_PA
_FIRST_PLY_INHIBITION_CURRENT_PA = tier._FIRST_PLY_INHIBITION_CURRENT_PA
_MAX_DECISIONS = tier._MAX_DECISIONS

PUBLIC_ARGUMENTS = h2.PUBLIC_ARGUMENTS
PUBLIC_TASKS = h2.PUBLIC_TASKS


@dataclass(frozen=True, slots=True)
class FirstPlyTierEnumeration:
    status: str
    error: str | None
    predictions: tuple[site.CandidatePrediction, ...]
    tier_beam: RestingGlifTierBeamResult | None
    exhaustive_sequence_count: int
    feature_sha256: str
    value_projection_sha256: str


def _enumerate_first_ply_tiers(
    model: world.FrozenAstWorldModel,
    observation: world.CurrentProgramObservation,
    desired_outputs: Sequence[int],
) -> FirstPlyTierEnumeration:
    """Physically enumerate the fixed first-ply currents into tiers once."""

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
    exhaustive_sequence_count = sum(
        1 if prediction.candidate.action == DONE else len(h2.ACTION_LABELS)
        for prediction in predictions
    )
    selector = CompiledRestingGlifTierBeam.compile(
        candidate_count=len(scored_currents),
        beam_width=len(predictions),
        step_count=total.banked._RACE_STEPS,
        min_current_pa=_FIRST_PLY_MIN_CURRENT_PA,
        max_current_pa=_FIRST_PLY_MAX_CURRENT_PA,
        inhibition_current_pa=_FIRST_PLY_INHIBITION_CURRENT_PA,
    )
    try:
        tier_beam = selector.select(
            scored_currents,
            stop_cell=len(predictions),
        )
    except RestingGlifRaceFailure as error:
        return FirstPlyTierEnumeration(
            status=STOP_FIRST_PLY_TIER_ENUMERATION,
            error=str(error),
            predictions=predictions,
            tier_beam=None,
            exhaustive_sequence_count=exhaustive_sequence_count,
            feature_sha256=feature_sha,
            value_projection_sha256=projection.sha256,
        )
    if tier_beam.status == STOP_LEADING_TIER:
        return FirstPlyTierEnumeration(
            status=STOP_ALL_UNKNOWN,
            error=STOP_ALL_UNKNOWN,
            predictions=predictions,
            tier_beam=tier_beam,
            exhaustive_sequence_count=exhaustive_sequence_count,
            feature_sha256=feature_sha,
            value_projection_sha256=projection.sha256,
        )
    if tier_beam.status != RETAINED:
        raise RuntimeError("first-ply tier enumeration returned an unknown status")
    if len(tier_beam.retained_candidate_cells) != len(predictions):
        raise RuntimeError("first-ply tier enumeration did not retain every candidate")
    return FirstPlyTierEnumeration(
        status=RETAINED,
        error=None,
        predictions=predictions,
        tier_beam=tier_beam,
        exhaustive_sequence_count=exhaustive_sequence_count,
        feature_sha256=feature_sha,
        value_projection_sha256=projection.sha256,
    )


@dataclass(frozen=True, slots=True)
class TerminalAttemptAudit:
    tiers_consumed: int
    active_first_candidates: int
    sequence_count: int
    status: str
    error: str | None
    race: CellRace | None
    winner_sequence: str | None
    exact_case_spikes: tuple[int, ...]
    all_cases_exact: bool
    feature_sha256: str
    value_projection_sha256: str
    sequence_digest: str


def _cached_sequence_bank(
    cache: dict[int, tuple[h2.Horizon2Prediction, ...]],
    candidate_count: int,
) -> tuple[h2.Horizon2Prediction, ...]:
    """Materialize cached expansions in the frozen first-candidate order."""

    return tuple(
        prediction
        for candidate_cell in range(candidate_count)
        for prediction in cache.get(candidate_cell, ())
    )


def _expand_new_tier(
    model: world.FrozenAstWorldModel,
    first_predictions: Sequence[site.CandidatePrediction],
    tier_cells: Sequence[int],
    cache: dict[int, tuple[h2.Horizon2Prediction, ...]],
) -> int:
    """Expand each newly admitted physical candidate exactly once."""

    expansion_count = 0
    candidate_count = len(tuple(first_predictions))
    for candidate_cell in tier_cells:
        if candidate_cell >= candidate_count:
            continue
        if candidate_cell in cache:
            raise RuntimeError("a first candidate was expanded more than once")
        predictions, transitions = strict._expand_retained_second_ply(
            model,
            (first_predictions[candidate_cell],),
        )
        cache[candidate_cell] = predictions
        expansion_count += transitions
    return expansion_count


def _score_terminal_attempt(
    model: world.FrozenAstWorldModel,
    observation: world.CurrentProgramObservation,
    predictions: Sequence[h2.Horizon2Prediction],
    desired_outputs: Sequence[int],
    checkpoint_before: str,
    *,
    tiers_consumed: int,
    active_first_candidates: int,
) -> tuple[h2.Horizon2Decision | None, TerminalAttemptAudit]:
    """Return a decision only for a unique, physically all-exact winner."""

    bank = tuple(predictions)
    sequences = tuple(prediction.sequence for prediction in bank)
    states, coincidences, feature_sha = h2._sequence_features(
        bank,
        desired_outputs,
    )
    projection = h2.FixedHorizon2ValueProjection.create(
        sequences,
        len(tuple(desired_outputs)),
    )
    currents = projection.currents(states, coincidences)
    labels = tuple(sequence.label for sequence in sequences)
    digest = strict._sequence_digest(bank)
    try:
        race = total.banked._physical_race(
            currents,
            (*labels, STOP_ALL_UNKNOWN),
            bank="ast-h2-anytime-terminal-sequence-value",
        )
    except total.banked.CellRaceFailure as error:
        checkpoint_after = model.live_checkpoint_sha256
        if checkpoint_after != checkpoint_before:
            raise RuntimeError(
                "frozen AST world model changed during anytime terminal race"
            )
        return None, TerminalAttemptAudit(
            tiers_consumed=tiers_consumed,
            active_first_candidates=active_first_candidates,
            sequence_count=len(bank),
            status="NON_UNIQUE",
            error=str(error),
            race=None,
            winner_sequence=None,
            exact_case_spikes=(),
            all_cases_exact=False,
            feature_sha256=feature_sha,
            value_projection_sha256=projection.sha256,
            sequence_digest=digest,
        )

    checkpoint_after = model.live_checkpoint_sha256
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("frozen AST world model changed during anytime planning")
    if race.winner == STOP_ALL_UNKNOWN:
        return None, TerminalAttemptAudit(
            tiers_consumed=tiers_consumed,
            active_first_candidates=active_first_candidates,
            sequence_count=len(bank),
            status=STOP_ALL_UNKNOWN,
            error=STOP_ALL_UNKNOWN,
            race=race,
            winner_sequence=STOP_ALL_UNKNOWN,
            exact_case_spikes=(),
            all_cases_exact=False,
            feature_sha256=feature_sha,
            value_projection_sha256=projection.sha256,
            sequence_digest=digest,
        )

    exact_case_spikes = tuple(
        int(np.count_nonzero(coincidences[race.winner_cell, case]))
        for case in range(len(tuple(desired_outputs)))
    )
    if any(spike_count not in (0, 1) for spike_count in exact_case_spikes):
        raise RuntimeError("terminal coincidence audit was not one-cell bounded")
    all_cases_exact = all(spike_count == 1 for spike_count in exact_case_spikes)
    if not all_cases_exact:
        return None, TerminalAttemptAudit(
            tiers_consumed=tiers_consumed,
            active_first_candidates=active_first_candidates,
            sequence_count=len(bank),
            status="UNIQUE_INEXACT",
            error="unique terminal winner lacked all physical coincidence spikes",
            race=race,
            winner_sequence=race.winner,
            exact_case_spikes=exact_case_spikes,
            all_cases_exact=False,
            feature_sha256=feature_sha,
            value_projection_sha256=projection.sha256,
            sequence_digest=digest,
        )

    selected = bank[race.winner_cell]
    factors = h2.FixedHorizon2FactorProjection.create(
        sequences,
        tuple(observed_site.path for observed_site in observation.sites),
    )
    site_race, first_action_race, continuation_race = factors.races(
        race.winner_cell
    )
    if site_race.winner != h2._path_label(selected.sequence.first.path):
        raise RuntimeError("anytime sequence-to-site projection lost factor parity")
    if first_action_race.winner != selected.sequence.first.action:
        raise RuntimeError("anytime sequence-to-first-action projection lost parity")
    if continuation_race.winner != selected.sequence.continuation_action:
        raise RuntimeError("anytime sequence-to-continuation projection lost parity")
    decision = h2.Horizon2Decision(
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
    return decision, TerminalAttemptAudit(
        tiers_consumed=tiers_consumed,
        active_first_candidates=active_first_candidates,
        sequence_count=len(bank),
        status="UNIQUE_ALL_EXACT",
        error=None,
        race=race,
        winner_sequence=race.winner,
        exact_case_spikes=exact_case_spikes,
        all_cases_exact=True,
        feature_sha256=feature_sha,
        value_projection_sha256=projection.sha256,
        sequence_digest=digest,
    )


@dataclass(frozen=True, slots=True)
class AnytimeHorizon2Decision:
    status: str
    error: str | None
    first_ply: FirstPlyTierEnumeration
    terminal: h2.Horizon2Decision | None
    terminal_attempts: tuple[TerminalAttemptAudit, ...]
    expanded_first_candidates: int
    second_ply_transition_expansions: int
    unique_terminal_sequences: int
    terminal_sequence_evaluations: int


def plan_site_action_h2_anytime(
    model: world.FrozenAstWorldModel,
    observation: world.CurrentProgramObservation,
    desired_outputs: Sequence[int],
) -> AnytimeHorizon2Decision:
    """Deepen complete physical tiers until terminal certainty or exhaustion."""

    desired = strict._validate_desired(observation, desired_outputs)
    checkpoint_before = model.live_checkpoint_sha256
    if checkpoint_before != model.checkpoint_sha256:
        raise RuntimeError("AST world-model checkpoint changed before anytime planning")
    first_ply = _enumerate_first_ply_tiers(model, observation, desired)
    if first_ply.status != RETAINED or first_ply.tier_beam is None:
        return AnytimeHorizon2Decision(
            status=first_ply.status,
            error=first_ply.error,
            first_ply=first_ply,
            terminal=None,
            terminal_attempts=(),
            expanded_first_candidates=0,
            second_ply_transition_expansions=0,
            unique_terminal_sequences=0,
            terminal_sequence_evaluations=0,
        )

    cache: dict[int, tuple[h2.Horizon2Prediction, ...]] = {}
    attempts: list[TerminalAttemptAudit] = []
    transition_expansions = 0
    terminal_bank: tuple[h2.Horizon2Prediction, ...] = ()
    for tier_round in first_ply.tier_beam.rounds:
        transition_expansions += _expand_new_tier(
            model,
            first_ply.predictions,
            tier_round.tier_cells,
            cache,
        )
        if len(cache) < BEAM_WIDTH:
            continue
        terminal_bank = _cached_sequence_bank(cache, len(first_ply.predictions))
        terminal, attempt = _score_terminal_attempt(
            model,
            observation,
            terminal_bank,
            desired,
            checkpoint_before,
            tiers_consumed=tier_round.tier_index + 1,
            active_first_candidates=len(cache),
        )
        attempts.append(attempt)
        if terminal is not None:
            return AnytimeHorizon2Decision(
                status="SELECTED",
                error=None,
                first_ply=first_ply,
                terminal=terminal,
                terminal_attempts=tuple(attempts),
                expanded_first_candidates=len(cache),
                second_ply_transition_expansions=transition_expansions,
                unique_terminal_sequences=len(terminal_bank),
                terminal_sequence_evaluations=sum(
                    recorded.sequence_count for recorded in attempts
                ),
            )

    if len(cache) != len(first_ply.predictions):
        raise RuntimeError("anytime exhaustion did not consume every first candidate")
    if len(terminal_bank) != first_ply.exhaustive_sequence_count:
        raise RuntimeError("anytime exhaustion lost exhaustive sequence parity")
    last_error = "anytime terminal scoring exhausted all physical first-ply tiers"
    if attempts and attempts[-1].error is not None:
        last_error = attempts[-1].error
    return AnytimeHorizon2Decision(
        status=STOP_ANYTIME_EXHAUSTED,
        error=last_error,
        first_ply=first_ply,
        terminal=None,
        terminal_attempts=tuple(attempts),
        expanded_first_candidates=len(cache),
        second_ply_transition_expansions=transition_expansions,
        unique_terminal_sequences=len(terminal_bank),
        terminal_sequence_evaluations=sum(
            recorded.sequence_count for recorded in attempts
        ),
    )


@dataclass(frozen=True, slots=True)
class AnytimePlanRecord:
    status: str
    error: str | None
    first_prediction_count: int
    total_tier_count: int
    consumed_tier_count: int
    expanded_first_candidates: int
    second_ply_transition_expansions: int
    unique_terminal_sequences: int
    exhaustive_sequence_count: int
    terminal_sequence_evaluations: int
    terminal_attempt_count: int
    selected_all_cases_exact: bool
    selected_path: tuple[int, ...] | None
    selected_action: str | None
    continuation_action: str | None


@dataclass(frozen=True, slots=True)
class AnytimePublicTaskResult:
    name: str
    passed: bool
    verdict: str
    error: str | None
    actual_decisions: tuple[tuple[tuple[int, ...], str], ...]
    final_tokens: tuple[str, ...]
    final_outputs: tuple[int, ...]
    plans: tuple[AnytimePlanRecord, ...]
    checkpoint_immutable: bool


@dataclass(frozen=True, slots=True)
class AnytimePublicMatrixResult:
    schema: str
    verdict: str
    passed_tasks: int
    total_tasks: int
    first_predictions_scored: int
    first_ply_tiers_enumerated: int
    first_ply_tiers_consumed: int
    first_candidates_expanded: int
    second_ply_transition_expansions: int
    unique_terminal_sequences: int
    terminal_sequence_evaluations: int
    terminal_race_attempts: int
    wall_time_seconds: float
    world_checkpoint_sha256: str
    target_free_observations: int
    tasks: tuple[AnytimePublicTaskResult, ...]
    claim_boundary: str


def run_public_task(
    model: world.FrozenAstWorldModel,
    task: h2.PublicHorizon2Task,
) -> AnytimePublicTaskResult:
    h2._validate_public_target(task)
    program = tp.Program(task.incumbent_tokens, site.visible_scalar_signature())
    actual_decisions: list[tuple[tuple[int, ...], str]] = []
    plans: list[AnytimePlanRecord] = []
    checkpoint_before = model.live_checkpoint_sha256
    error: str | None = None

    for _ in range(_MAX_DECISIONS):
        observation = world.observe_current_program(program, PUBLIC_ARGUMENTS)
        decision = plan_site_action_h2_anytime(
            model,
            observation,
            task.desired_outputs,
        )
        terminal = decision.terminal
        tier_rounds = (
            ()
            if decision.first_ply.tier_beam is None
            else decision.first_ply.tier_beam.rounds
        )
        consumed_tiers = (
            0
            if not decision.terminal_attempts
            else decision.terminal_attempts[-1].tiers_consumed
        )
        selected_exact = bool(
            decision.terminal_attempts
            and decision.terminal_attempts[-1].all_cases_exact
        )
        plans.append(
            AnytimePlanRecord(
                status=decision.status,
                error=decision.error,
                first_prediction_count=len(decision.first_ply.predictions),
                total_tier_count=len(tier_rounds),
                consumed_tier_count=consumed_tiers,
                expanded_first_candidates=decision.expanded_first_candidates,
                second_ply_transition_expansions=(
                    decision.second_ply_transition_expansions
                ),
                unique_terminal_sequences=decision.unique_terminal_sequences,
                exhaustive_sequence_count=(
                    decision.first_ply.exhaustive_sequence_count
                ),
                terminal_sequence_evaluations=(
                    decision.terminal_sequence_evaluations
                ),
                terminal_attempt_count=len(decision.terminal_attempts),
                selected_all_cases_exact=selected_exact,
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
        error = "anytime H=2 task exhausted the fixed decision bound"

    final = world.observe_current_program(program, PUBLIC_ARGUMENTS).site(()).outputs
    if final is None:
        raise RuntimeError("anytime H=2 runner lost scalar final outputs")
    actual = tuple(actual_decisions)
    checkpoint_immutable = checkpoint_before == model.live_checkpoint_sha256
    passed = (
        error is None
        and actual == task.expected_decisions
        and program.tokens == task.target_tokens
        and final == task.desired_outputs
        and checkpoint_immutable
    )
    return AnytimePublicTaskResult(
        name=task.name,
        passed=passed,
        verdict="PASS" if passed else "STOP",
        error=None if passed else (error or "anytime trajectory missed public target"),
        actual_decisions=actual,
        final_tokens=program.tokens,
        final_outputs=final,
        plans=tuple(plans),
        checkpoint_immutable=checkpoint_immutable,
    )


def run_public_matrix(
    tasks: Sequence[h2.PublicHorizon2Task] = PUBLIC_TASKS,
) -> AnytimePublicMatrixResult:
    """Run the visible anytime development gate without writing an artifact."""

    cohort = tuple(tasks)
    if not cohort:
        raise ValueError("anytime public matrix needs at least one task")
    started = perf_counter()
    model, training = world.train_public_ast_world_model()
    results = tuple(run_public_task(model, task) for task in cohort)
    elapsed = perf_counter() - started
    passed_tasks = sum(task.passed for task in results)
    plans = tuple(plan for task in results for plan in task.plans)
    return AnytimePublicMatrixResult(
        schema="public-typed-ast-horizon2-int-anytime-v1",
        verdict=(
            "PASS_TYPED_AST_HORIZON2_INT_ANYTIME"
            if passed_tasks == len(results)
            else "STOP_TYPED_AST_HORIZON2_INT_ANYTIME"
        ),
        passed_tasks=passed_tasks,
        total_tasks=len(results),
        first_predictions_scored=sum(plan.first_prediction_count for plan in plans),
        first_ply_tiers_enumerated=sum(plan.total_tier_count for plan in plans),
        first_ply_tiers_consumed=sum(plan.consumed_tier_count for plan in plans),
        first_candidates_expanded=sum(
            plan.expanded_first_candidates for plan in plans
        ),
        second_ply_transition_expansions=sum(
            plan.second_ply_transition_expansions for plan in plans
        ),
        unique_terminal_sequences=sum(
            plan.unique_terminal_sequences for plan in plans
        ),
        terminal_sequence_evaluations=sum(
            plan.terminal_sequence_evaluations for plan in plans
        ),
        terminal_race_attempts=sum(plan.terminal_attempt_count for plan in plans),
        wall_time_seconds=elapsed,
        world_checkpoint_sha256=model.checkpoint_sha256,
        target_free_observations=training.target_free_observations,
        tasks=results,
        claim_boundary=(
            "Visible public Int uncertainty-driven optimization only: first-ply "
            "tiers and STOP are physical, learned second plies are cached, and "
            "only terminal race uniqueness plus physical all-case coincidence "
            "halts expansion. No exhaustive-evidence replacement, arbitrary "
            "hypothetical second site, Bool/List, fresh, sealed, M0, Simple-SWE, "
            "held-out, or C1 evidence."
        ),
    )


__all__ = [
    "AnytimeHorizon2Decision",
    "AnytimePlanRecord",
    "AnytimePublicMatrixResult",
    "AnytimePublicTaskResult",
    "BEAM_WIDTH",
    "FirstPlyTierEnumeration",
    "STOP_ANYTIME_EXHAUSTED",
    "STOP_FIRST_PLY_TIER_ENUMERATION",
    "TerminalAttemptAudit",
    "plan_site_action_h2_anytime",
    "run_public_matrix",
    "run_public_task",
]
