"""STOP-bounded uncertainty expansion for the public typed-AST H=2 planner.

This is a new visible-development mechanism after the spent anytime V1.  The
same frozen first-ply currents are enumerated into complete physical tiers only
through the first tier containing STOP.  Candidates below that admissibility
frontier are never awaited or expanded.  Above-frontier learned continuations
remain cached, and terminal scoring deepens only after a non-unique race or a
unique winner lacking all desired-case physical coincidence spikes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for search_path in (REPO_ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.r3_pilot import public_typed_ast_horizon2_int_anytime as v1  # noqa: E402
from experiments.r3_pilot import public_typed_ast_horizon2_int_planner as h2  # noqa: E402
from experiments.r3_pilot import public_typed_ast_site_planner as site  # noqa: E402
from experiments.r3_pilot import public_typed_ast_world_model as world  # noqa: E402
from humanbrain import typed_program as tp  # noqa: E402
from humanbrain.resting_glif_stop_bounded_tiers import (  # noqa: E402
    CompiledStopBoundedRestingGlifTiers,
    STOP_FRONTIER_REACHED,
    StopBoundedTierResult,
)

total = world.total

DONE = h2.DONE
STOP_ALL_UNKNOWN = h2.STOP_ALL_UNKNOWN
STOP_INSUFFICIENT_ADMISSIBLE = "STOP_INSUFFICIENT_ADMISSIBLE"
STOP_ADMISSIBLE_EXHAUSTED = "STOP_ADMISSIBLE_EXHAUSTED"

# Frozen aliases: V2 changes only the physical meaning of the STOP frontier.
BEAM_WIDTH = v1.BEAM_WIDTH
_FIRST_PLY_MIN_CURRENT_PA = v1._FIRST_PLY_MIN_CURRENT_PA
_FIRST_PLY_MAX_CURRENT_PA = v1._FIRST_PLY_MAX_CURRENT_PA
_FIRST_PLY_INHIBITION_CURRENT_PA = v1._FIRST_PLY_INHIBITION_CURRENT_PA
_MAX_DECISIONS = v1._MAX_DECISIONS

PUBLIC_ARGUMENTS = h2.PUBLIC_ARGUMENTS
PUBLIC_TASKS = h2.PUBLIC_TASKS


def _sequence_count(
    predictions: Sequence[site.CandidatePrediction],
    candidate_cells: Sequence[int],
) -> int:
    bank = tuple(predictions)
    return sum(
        1 if bank[cell].candidate.action == DONE else len(h2.ACTION_LABELS)
        for cell in candidate_cells
    )


@dataclass(frozen=True, slots=True)
class FirstPlyStopBoundedEnumeration:
    status: str
    error: str | None
    predictions: tuple[site.CandidatePrediction, ...]
    tiers: StopBoundedTierResult
    exhaustive_sequence_count: int
    admissible_sequence_count: int
    feature_sha256: str
    value_projection_sha256: str


def _enumerate_first_ply_through_stop(
    model: world.FrozenAstWorldModel,
    observation: world.CurrentProgramObservation,
    desired_outputs: Sequence[int],
) -> FirstPlyStopBoundedEnumeration:
    """Enumerate fixed first-ply currents only through physical STOP."""

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
    enumerator = CompiledStopBoundedRestingGlifTiers.compile(
        candidate_count=len(scored_currents),
        step_count=total.banked._RACE_STEPS,
        min_current_pa=_FIRST_PLY_MIN_CURRENT_PA,
        max_current_pa=_FIRST_PLY_MAX_CURRENT_PA,
        inhibition_current_pa=_FIRST_PLY_INHIBITION_CURRENT_PA,
    )
    tiers = enumerator.enumerate(
        scored_currents,
        stop_cell=len(predictions),
    )
    if tiers.status != STOP_FRONTIER_REACHED:
        raise RuntimeError("STOP-bounded enumeration returned an unknown status")
    exhaustive_cells = tuple(range(len(predictions)))
    exhaustive_sequence_count = _sequence_count(predictions, exhaustive_cells)
    admissible_sequence_count = _sequence_count(
        predictions,
        tiers.admissible_candidate_cells,
    )
    status = STOP_ALL_UNKNOWN if not tiers.admissible_candidate_cells else "ENUMERATED"
    return FirstPlyStopBoundedEnumeration(
        status=status,
        error=STOP_ALL_UNKNOWN if status == STOP_ALL_UNKNOWN else None,
        predictions=predictions,
        tiers=tiers,
        exhaustive_sequence_count=exhaustive_sequence_count,
        admissible_sequence_count=admissible_sequence_count,
        feature_sha256=feature_sha,
        value_projection_sha256=projection.sha256,
    )


@dataclass(frozen=True, slots=True)
class AnytimeV2Decision:
    status: str
    error: str | None
    first_ply: FirstPlyStopBoundedEnumeration
    terminal: h2.Horizon2Decision | None
    terminal_attempts: tuple[v1.TerminalAttemptAudit, ...]
    expanded_first_candidates: int
    second_ply_transition_expansions: int
    unique_terminal_sequences: int
    terminal_sequence_evaluations: int
    world_checkpoint_before_sha256: str
    world_checkpoint_after_sha256: str


def plan_site_action_h2_anytime_v2(
    model: world.FrozenAstWorldModel,
    observation: world.CurrentProgramObservation,
    desired_outputs: Sequence[int],
) -> AnytimeV2Decision:
    """Deepen admissible complete tiers until physical certainty or STOP."""

    desired = v1.strict._validate_desired(observation, desired_outputs)
    checkpoint_before = model.live_checkpoint_sha256
    if checkpoint_before != model.checkpoint_sha256:
        raise RuntimeError("AST world-model checkpoint changed before anytime V2")
    first_ply = _enumerate_first_ply_through_stop(model, observation, desired)
    if model.live_checkpoint_sha256 != checkpoint_before:
        raise RuntimeError("AST world-model checkpoint changed during tier enumeration")
    if first_ply.status == STOP_ALL_UNKNOWN:
        checkpoint_after = model.live_checkpoint_sha256
        return AnytimeV2Decision(
            status=STOP_ALL_UNKNOWN,
            error=STOP_ALL_UNKNOWN,
            first_ply=first_ply,
            terminal=None,
            terminal_attempts=(),
            expanded_first_candidates=0,
            second_ply_transition_expansions=0,
            unique_terminal_sequences=0,
            terminal_sequence_evaluations=0,
            world_checkpoint_before_sha256=checkpoint_before,
            world_checkpoint_after_sha256=checkpoint_after,
        )

    cache: dict[int, tuple[h2.Horizon2Prediction, ...]] = {}
    attempts: list[v1.TerminalAttemptAudit] = []
    transition_expansions = 0
    terminal_bank: tuple[h2.Horizon2Prediction, ...] = ()
    for tier_round in first_ply.tiers.rounds:
        before_count = len(cache)
        transition_expansions += v1._expand_new_tier(
            model,
            first_ply.predictions,
            tier_round.tier_cells,
            cache,
        )
        newly_admitted = len(cache) - before_count
        if len(cache) < BEAM_WIDTH or newly_admitted == 0:
            continue
        terminal_bank = v1._cached_sequence_bank(
            cache,
            len(first_ply.predictions),
        )
        terminal, attempt = v1._score_terminal_attempt(
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
            checkpoint_after = model.live_checkpoint_sha256
            if checkpoint_after != checkpoint_before:
                raise RuntimeError("AST world-model checkpoint changed during anytime V2 selection")
            return AnytimeV2Decision(
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
                world_checkpoint_before_sha256=checkpoint_before,
                world_checkpoint_after_sha256=checkpoint_after,
            )

    if model.live_checkpoint_sha256 != checkpoint_before:
        raise RuntimeError("AST world-model checkpoint changed during anytime V2")
    if len(cache) < BEAM_WIDTH:
        checkpoint_after = model.live_checkpoint_sha256
        return AnytimeV2Decision(
            status=STOP_INSUFFICIENT_ADMISSIBLE,
            error=(
                f"physical STOP frontier admitted {len(cache)} candidates below B="
                f"{BEAM_WIDTH}"
            ),
            first_ply=first_ply,
            terminal=None,
            terminal_attempts=tuple(attempts),
            expanded_first_candidates=len(cache),
            second_ply_transition_expansions=transition_expansions,
            unique_terminal_sequences=len(terminal_bank),
            terminal_sequence_evaluations=sum(
                recorded.sequence_count for recorded in attempts
            ),
            world_checkpoint_before_sha256=checkpoint_before,
            world_checkpoint_after_sha256=checkpoint_after,
        )
    if len(cache) != len(first_ply.tiers.admissible_candidate_cells):
        raise RuntimeError("anytime V2 did not consume every admissible candidate")
    if len(terminal_bank) != first_ply.admissible_sequence_count:
        raise RuntimeError("anytime V2 lost admissible sequence parity")
    last_error = "terminal uncertainty remained at the physical STOP frontier"
    if attempts and attempts[-1].error is not None:
        last_error = attempts[-1].error
    checkpoint_after = model.live_checkpoint_sha256
    return AnytimeV2Decision(
        status=STOP_ADMISSIBLE_EXHAUSTED,
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
        world_checkpoint_before_sha256=checkpoint_before,
        world_checkpoint_after_sha256=checkpoint_after,
    )


@dataclass(frozen=True, slots=True)
class AnytimeV2PlanRecord:
    status: str
    error: str | None
    first_prediction_count: int
    admissible_first_candidates: int
    excluded_first_candidates: int
    total_tier_count: int
    consumed_tier_count: int
    expanded_first_candidates: int
    second_ply_transition_expansions: int
    unique_terminal_sequences: int
    admissible_sequence_count: int
    exhaustive_sequence_count: int
    terminal_sequence_evaluations: int
    terminal_attempt_count: int
    selected_all_cases_exact: bool
    selected_path: tuple[int, ...] | None
    selected_action: str | None
    continuation_action: str | None


@dataclass(frozen=True, slots=True)
class AnytimeV2TaskResult:
    name: str
    passed: bool
    verdict: str
    error: str | None
    actual_decisions: tuple[tuple[tuple[int, ...], str], ...]
    final_tokens: tuple[str, ...]
    final_outputs: tuple[int, ...]
    plans: tuple[AnytimeV2PlanRecord, ...]
    checkpoint_immutable: bool


@dataclass(frozen=True, slots=True)
class AnytimeV2MatrixResult:
    schema: str
    verdict: str
    passed_tasks: int
    total_tasks: int
    first_predictions_scored: int
    admissible_first_candidates: int
    excluded_first_candidates: int
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
    tasks: tuple[AnytimeV2TaskResult, ...]
    claim_boundary: str


def run_public_task(
    model: world.FrozenAstWorldModel,
    task: h2.PublicHorizon2Task,
) -> AnytimeV2TaskResult:
    program = tp.Program(task.incumbent_tokens, site.visible_scalar_signature())
    actual_decisions: list[tuple[tuple[int, ...], str]] = []
    plans: list[AnytimeV2PlanRecord] = []
    checkpoint_before = model.live_checkpoint_sha256
    error: str | None = None

    for _ in range(_MAX_DECISIONS):
        observation = world.observe_current_program(program, PUBLIC_ARGUMENTS)
        decision = plan_site_action_h2_anytime_v2(
            model,
            observation,
            task.desired_outputs,
        )
        terminal = decision.terminal
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
            AnytimeV2PlanRecord(
                status=decision.status,
                error=decision.error,
                first_prediction_count=len(decision.first_ply.predictions),
                admissible_first_candidates=len(
                    decision.first_ply.tiers.admissible_candidate_cells
                ),
                excluded_first_candidates=len(
                    decision.first_ply.tiers.excluded_candidate_cells
                ),
                total_tier_count=len(decision.first_ply.tiers.rounds),
                consumed_tier_count=consumed_tiers,
                expanded_first_candidates=decision.expanded_first_candidates,
                second_ply_transition_expansions=(
                    decision.second_ply_transition_expansions
                ),
                unique_terminal_sequences=decision.unique_terminal_sequences,
                admissible_sequence_count=(
                    decision.first_ply.admissible_sequence_count
                ),
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
        error = "anytime V2 task exhausted the fixed decision bound"

    final = world.observe_current_program(program, PUBLIC_ARGUMENTS).site(()).outputs
    if final is None:
        raise RuntimeError("anytime V2 runner lost scalar final outputs")
    actual = tuple(actual_decisions)
    checkpoint_immutable = checkpoint_before == model.live_checkpoint_sha256
    h2._validate_public_target(task)
    passed = (
        error is None
        and actual == task.expected_decisions
        and program.tokens == task.target_tokens
        and final == task.desired_outputs
        and checkpoint_immutable
    )
    return AnytimeV2TaskResult(
        name=task.name,
        passed=passed,
        verdict="PASS" if passed else "STOP",
        error=None if passed else (error or "anytime V2 trajectory missed target"),
        actual_decisions=actual,
        final_tokens=program.tokens,
        final_outputs=final,
        plans=tuple(plans),
        checkpoint_immutable=checkpoint_immutable,
    )


def run_public_matrix(
    tasks: Sequence[h2.PublicHorizon2Task] = PUBLIC_TASKS,
) -> AnytimeV2MatrixResult:
    """Run the visible STOP-bounded V2 gate without writing an artifact."""

    cohort = tuple(tasks)
    if not cohort:
        raise ValueError("anytime V2 public matrix needs at least one task")
    started = perf_counter()
    model, training = world.train_public_ast_world_model()
    results = tuple(run_public_task(model, task) for task in cohort)
    elapsed = perf_counter() - started
    passed_tasks = sum(task.passed for task in results)
    plans = tuple(plan for task in results for plan in task.plans)
    return AnytimeV2MatrixResult(
        schema="public-typed-ast-horizon2-int-anytime-v2",
        verdict=(
            "PASS_TYPED_AST_HORIZON2_INT_ANYTIME_V2"
            if passed_tasks == len(results)
            else "STOP_TYPED_AST_HORIZON2_INT_ANYTIME_V2"
        ),
        passed_tasks=passed_tasks,
        total_tasks=len(results),
        first_predictions_scored=sum(plan.first_prediction_count for plan in plans),
        admissible_first_candidates=sum(
            plan.admissible_first_candidates for plan in plans
        ),
        excluded_first_candidates=sum(
            plan.excluded_first_candidates for plan in plans
        ),
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
            "Visible public Int STOP-bounded anytime optimization only: complete "
            "physical tiers through STOP define admissibility, learned second "
            "plies are cached, and only terminal uniqueness plus all-case physical "
            "coincidence halts expansion. No exhaustive-evidence replacement, "
            "arbitrary hypothetical second site, Bool/List, fresh, sealed, M0, "
            "Simple-SWE, held-out, or C1 evidence."
        ),
    )


__all__ = [
    "AnytimeV2Decision",
    "AnytimeV2MatrixResult",
    "AnytimeV2PlanRecord",
    "AnytimeV2TaskResult",
    "BEAM_WIDTH",
    "FirstPlyStopBoundedEnumeration",
    "STOP_ADMISSIBLE_EXHAUSTED",
    "STOP_INSUFFICIENT_ADMISSIBLE",
    "plan_site_action_h2_anytime_v2",
    "run_public_matrix",
    "run_public_task",
]
