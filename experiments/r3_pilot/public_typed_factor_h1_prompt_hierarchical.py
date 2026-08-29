"""Fresh one-shot hierarchical prompt/value H=1 public development probe.

The root operator successor predicts every compiler-valid root action without
constructing a candidate program.  A complete physical leading value tier,
including STOP, is retained before prompt evidence exists.  Prompt binding is
then projected only over that tier.  One unique tier-local physical winner may
pass through the six-factor motor, the structural successor, and the sole
compiler-owned actuator.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for search_path in (REPO_ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.r3_pilot import public_typed_factor_h1_root as base_h1  # noqa: E402
from experiments.r3_pilot import public_typed_factor_motor as factor_motor  # noqa: E402
from experiments.r3_pilot import (  # noqa: E402
    public_typed_factor_value_race as value_race,
)
from experiments.r3_pilot import (  # noqa: E402
    public_typed_prompt_factors as prompt_factors,
)
from experiments.r3_pilot import (  # noqa: E402
    public_typed_program_observation as root_observation,
)
from experiments.r3_pilot import (  # noqa: E402
    public_typed_structure_successor as structure,
)
from humanbrain import typed_program as tp  # noqa: E402
from humanbrain.resting_glif_race import (  # noqa: E402
    CompiledRestingGlifRace,
    RestingGlifRaceFailure,
    RestingGlifRaceResult,
)
from humanbrain.resting_glif_tier_beam import (  # noqa: E402
    CompiledRestingGlifTierBeam,
    RETAINED,
    RestingGlifTierBeamResult,
    STOP_LEADING_TIER,
)
from humanbrain.typed_program_types import (  # noqa: E402
    ProgramParameter,
    ProgramType,
    TypedSignature,
)


SCHEMA = "public-typed-factor-h1-prompt-hierarchical-fresh-v1"
PASS_VERDICT = "PASS_TYPED_FACTOR_H1_PROMPT_HIERARCHICAL_FRESH"
STOP_VERDICT = "STOP_TYPED_FACTOR_H1_PROMPT_HIERARCHICAL_FRESH"

STATUS_SELECTED = "SELECTED"
STATUS_STOP_VALUE = "STOP_VALUE_TIER"
STATUS_STOP_VALUE_RACE = "STOP_VALUE_TIER_RACE"
STATUS_STOP_PROMPT = "STOP_PROMPT"
STATUS_STOP_PROMPT_NONUNIQUE = "STOP_PROMPT_NONUNIQUE"
STATUS_STOP_STRUCTURE = "STOP_STRUCTURE_UNKNOWN"

PUBLIC_LITERAL_BANK = tuple(range(-4, 5))
PUBLIC_TWO_INT_CASES = ((7, 2), (-3, 4), (5, 5))
PUBLIC_BOOL_CASES = ((-3,), (0,), (2,), (5,))

# Fixed validation envelope, not a score calibration.  The current value bank
# is bounded much more tightly for these four- or three-case typed rows.
_VALUE_TIER_MIN_CURRENT_PA = -100_000.0
_VALUE_TIER_MAX_CURRENT_PA = 10_000.0
_VALUE_TIER_INHIBITION_PA = 120_000.0
_PROMPT_STOP_CURRENT_PA = 650.0
_PROMPT_ACTION_RACE = CompiledRestingGlifRace.compile(
    step_count=value_race._RACE_STEPS
)

_ARTIFACT_PATH = Path(__file__).with_suffix(".json")
_SPEND_MARKER_PATH = Path(__file__).with_suffix(".spend.json")


def _int_signature_two() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "solve",
        (ProgramParameter("a", integer), ProgramParameter("b", integer)),
        integer,
    )


def _bool_signature_one() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "solve",
        (ProgramParameter("x", integer),),
        ProgramType.Bool(),
    )


@dataclass(frozen=True, slots=True)
class PublicPromptH1Task:
    name: str
    prompt: str
    signature: TypedSignature
    cases: tuple[tuple[object, ...], ...]
    incumbent_tokens: tuple[str, ...]
    desired_outputs: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class PublicPromptH1Target:
    name: str
    expected_tokens: tuple[str, ...]


PUBLIC_TASKS = (
    PublicPromptH1Task(
        name="two-int-a-minus-b",
        prompt="Return a minus b.",
        signature=_int_signature_two(),
        cases=PUBLIC_TWO_INT_CASES,
        incumbent_tokens=("absolute", "variable-0"),
        desired_outputs=(5, -7, 0),
    ),
    PublicPromptH1Task(
        name="two-int-b-minus-a",
        prompt="Return b minus a.",
        signature=_int_signature_two(),
        cases=PUBLIC_TWO_INT_CASES,
        incumbent_tokens=("absolute", "variable-0"),
        desired_outputs=(-5, 7, 0),
    ),
    PublicPromptH1Task(
        name="bool-x-less-than-2",
        prompt="Return whether x is less than 2.",
        signature=_bool_signature_one(),
        cases=PUBLIC_BOOL_CASES,
        incumbent_tokens=("less-than", "variable-0", "variable-0"),
        desired_outputs=(True, True, False, False),
    ),
    PublicPromptH1Task(
        name="bool-2-less-than-x",
        prompt="Return whether 2 is less than x.",
        signature=_bool_signature_one(),
        cases=PUBLIC_BOOL_CASES,
        incumbent_tokens=("less-than", "variable-0", "variable-0"),
        desired_outputs=(False, False, False, True),
    ),
)

PUBLIC_TARGETS = (
    PublicPromptH1Target(
        "two-int-a-minus-b",
        ("subtract", "variable-1", "variable-0"),
    ),
    PublicPromptH1Target(
        "two-int-b-minus-a",
        ("subtract", "variable-0", "variable-1"),
    ),
    PublicPromptH1Target(
        "bool-x-less-than-2",
        ("less-than", "variable-0", "integer", "+", "2", "end"),
    ),
    PublicPromptH1Target(
        "bool-2-less-than-x",
        ("less-than", "integer", "+", "2", "end", "variable-0"),
    ),
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _action_record(action: tp.FactorAction) -> dict[str, object]:
    return {
        "path": action.path,
        "schema": action.schema,
        "operator": action.operator,
        "left_source": action.left_source,
        "right_source": action.right_source,
        "literal": action.literal,
    }


def _model_checkpoint_sha256(base_checkpoint_sha256: str) -> str:
    digest = sha256()
    digest.update(b"public-typed-factor-h1-prompt-hierarchical-v1\0")
    digest.update(base_checkpoint_sha256.encode("ascii"))
    digest.update(np.asarray(PUBLIC_LITERAL_BANK, dtype=np.int64).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class HierarchicalPromptH1Decision:
    status: str
    stop_reason: str | None
    incumbent_sha256: str
    candidate_bank: tp.FactorCandidateBank
    candidate_predictions: tuple[base_h1.CandidateValuePrediction, ...]
    current_observation: root_observation.CurrentProgramRootObservation
    current_structure: structure.StructureObservation
    value_projection_sha256: str
    value_feature_sha256: str
    value_currents_pa: tuple[float, ...]
    value_tier: RestingGlifTierBeamResult | None
    leading_candidate_cells: tuple[int, ...]
    leading_actions: tuple[tp.FactorAction, ...]
    prompt_projection_sha256: str | None
    prompt_decision: prompt_factors.PromptBindingProjectionDecision | None
    prompt_action_currents_pa: tuple[float, ...]
    prompt_action_race: RestingGlifRaceResult | None
    selected_candidate_cell: int | None
    selected_action: tp.FactorAction | None
    motor_decision: factor_motor.FactorMotorDecision | None
    structure_prediction: structure.StructurePrediction | None
    selected_program: tp.Program | None
    final_observation: root_observation.CurrentProgramRootObservation | None
    final_structure: structure.StructureObservation | None
    root_successor_action_count: int
    nonroot_unknown_count: int
    structure_prediction_calls: int
    actuator_calls: int
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenHierarchicalPromptH1:
    base_model: base_h1.FrozenTypedFactorH1Root
    checkpoint_sha256: str

    @property
    def live_checkpoint_sha256(self) -> str:
        return _model_checkpoint_sha256(self.base_model.live_checkpoint_sha256)

    def plan(
        self,
        incumbent: tp.Program,
        prompt: str,
        cases: Sequence[tuple[object, ...]],
        desired_outputs: Sequence[object],
    ) -> HierarchicalPromptH1Decision:
        """Select at most one real edit through value then prompt hierarchy."""

        if not isinstance(incumbent, tp.Program):
            raise TypeError("hierarchical H1 requires one incumbent Program")
        if type(prompt) is not str or not prompt:
            raise ValueError("hierarchical H1 requires a non-empty prompt")
        before = self.live_checkpoint_sha256
        if before != self.checkpoint_sha256:
            raise RuntimeError("hierarchical H1 checkpoint changed before planning")

        candidate_bank = tp.factor_action_candidates(
            incumbent,
            literals=PUBLIC_LITERAL_BANK,
        )
        current = root_observation.observe_current_program_root(incumbent, cases)
        if not current.all_supported:
            raise ValueError("incumbent root observation must be physically supported")
        current_structure = structure.observe_program_structure(incumbent)
        selected_rows = tuple(case.output_encoding for case in current.cases)
        scopes = base_h1._scope_rows(incumbent, current)
        supplied_desired = tuple(desired_outputs)
        if len(supplied_desired) != len(selected_rows):
            raise ValueError("desired outputs must align with the supplied cases")
        desired = tuple(
            base_h1._encode_value(value, incumbent.signature.result)
            for value in supplied_desired
        )
        if not all(item.supported for item in desired):
            raise ValueError("desired outputs must inhabit physical typed cells")

        unknown_root = base_h1._typed_unknown(
            base_h1._declared_output_type(incumbent.signature.result)
        )
        root_actions = tuple(
            action for action in candidate_bank.actions if action.path == ()
        )
        root_effect_rows = iter(
            self.base_model.operator_successor.predict_actions_many(
                selected_rows,
                root_actions,
                scope_rows=scopes,
            )
        )
        candidate_predictions: list[base_h1.CandidateValuePrediction] = []
        scored_rows: list[tuple[object, ...]] = []
        for action in candidate_bank.actions:
            if action.path == ():
                effects = next(root_effect_rows)
                scored = tuple(effect.output for effect in effects)
                root_supported = True
            else:
                effects = ()
                scored = (unknown_root,) * len(selected_rows)
                root_supported = False
            candidate_predictions.append(
                base_h1.CandidateValuePrediction(
                    action=action,
                    operator_predictions=effects,
                    scored_outputs=scored,
                    root_site_supported=root_supported,
                )
            )
            scored_rows.append(scored)

        frozen_predictions = tuple(candidate_predictions)
        projection = value_race.FixedTypedFactorValueProjection.build(
            candidate_bank.actions,
            desired,
        )
        features = projection.features(tuple(scored_rows))
        value_currents = projection.tier_currents(features)

        value_tier: RestingGlifTierBeamResult | None = None
        leading_cells: tuple[int, ...] = ()
        leading_actions: tuple[tp.FactorAction, ...] = ()
        prompt_projection_sha: str | None = None
        prompt_decision: prompt_factors.PromptBindingProjectionDecision | None = None
        prompt_action_currents: tuple[float, ...] = ()
        prompt_action_race: RestingGlifRaceResult | None = None
        selected_cell: int | None = None
        selected_action: tp.FactorAction | None = None
        motor_decision: factor_motor.FactorMotorDecision | None = None
        structure_prediction: structure.StructurePrediction | None = None
        selected_program: tp.Program | None = None
        final_observation: root_observation.CurrentProgramRootObservation | None = None
        final_structure: structure.StructureObservation | None = None
        structure_calls = 0
        actuator_calls = 0

        def finish(status: str, reason: str | None) -> HierarchicalPromptH1Decision:
            after = self.live_checkpoint_sha256
            if after != before or after != self.checkpoint_sha256:
                raise RuntimeError("hierarchical H1 checkpoint changed during planning")
            return HierarchicalPromptH1Decision(
                status=status,
                stop_reason=reason,
                incumbent_sha256=incumbent.sha256,
                candidate_bank=candidate_bank,
                candidate_predictions=frozen_predictions,
                current_observation=current,
                current_structure=current_structure,
                value_projection_sha256=projection.sha256,
                value_feature_sha256=features.sha256,
                value_currents_pa=tuple(float(value) for value in value_currents),
                value_tier=value_tier,
                leading_candidate_cells=leading_cells,
                leading_actions=leading_actions,
                prompt_projection_sha256=prompt_projection_sha,
                prompt_decision=prompt_decision,
                prompt_action_currents_pa=prompt_action_currents,
                prompt_action_race=prompt_action_race,
                selected_candidate_cell=selected_cell,
                selected_action=selected_action,
                motor_decision=motor_decision,
                structure_prediction=structure_prediction,
                selected_program=selected_program,
                final_observation=final_observation,
                final_structure=final_structure,
                root_successor_action_count=len(root_actions),
                nonroot_unknown_count=len(candidate_bank.actions) - len(root_actions),
                structure_prediction_calls=structure_calls,
                actuator_calls=actuator_calls,
                checkpoint_before_sha256=before,
                checkpoint_after_sha256=after,
            )

        selector = CompiledRestingGlifTierBeam.compile(
            candidate_count=len(value_currents),
            step_count=value_race._RACE_STEPS,
            min_current_pa=_VALUE_TIER_MIN_CURRENT_PA,
            max_current_pa=_VALUE_TIER_MAX_CURRENT_PA,
            inhibition_current_pa=_VALUE_TIER_INHIBITION_PA,
            beam_width=1,
        )
        try:
            value_tier = selector.select(
                value_currents,
                stop_cell=len(candidate_bank.actions),
            )
        except RestingGlifRaceFailure as error:
            return finish(STATUS_STOP_VALUE_RACE, str(error))
        if value_tier.status == STOP_LEADING_TIER:
            return finish(STATUS_STOP_VALUE, "physical STOP occupies the value tier")
        if value_tier.status != RETAINED or len(value_tier.rounds) != 1:
            raise RuntimeError(
                "value hierarchy did not return one complete leading tier"
            )

        leading_cells = value_tier.retained_candidate_cells
        leading_actions = tuple(
            candidate_bank.actions[cell] for cell in leading_cells
        )
        tier_prompt_projection = prompt_factors.FixedPromptBindingProjection.build(
            leading_actions
        )
        prompt_projection_sha = tier_prompt_projection.sha256
        prompt_decision = tier_prompt_projection.project(prompt, incumbent.signature)
        if prompt_decision.status != prompt_factors.PROJECTION_STATUS_PROJECTED:
            return finish(STATUS_STOP_PROMPT, prompt_decision.status)

        prompt_action_currents = (
            *prompt_decision.action_currents_pa,
            _PROMPT_STOP_CURRENT_PA,
        )
        try:
            prompt_action_race = _PROMPT_ACTION_RACE.race_many(
                np.asarray((prompt_action_currents,), dtype=np.float64)
            )[0]
        except RestingGlifRaceFailure as error:
            return finish(STATUS_STOP_PROMPT_NONUNIQUE, str(error))
        if prompt_action_race.winner_cell == len(leading_actions):
            return finish(STATUS_STOP_PROMPT, "tier-local prompt STOP won")

        tier_local_cell = prompt_action_race.winner_cell
        selected_cell = leading_cells[tier_local_cell]
        selected_action = candidate_bank.actions[selected_cell]
        motor_projection = factor_motor.FixedFactorMotorProjection.build(
            candidate_bank.actions
        )
        selected_spikes = np.zeros(len(candidate_bank.actions), dtype=np.float64)
        selected_spikes[selected_cell] = 1.0
        motor_decision = motor_projection.resolve(selected_spikes)
        if motor_decision.source_action != selected_action:
            raise RuntimeError("prompt winner and six-factor motor lost parity")

        structure_prediction = self.base_model.structure_successor.predict(
            current_structure,
            motor_decision.source_action,
        )
        structure_calls = 1
        if not structure_prediction.supported:
            return finish(
                STATUS_STOP_STRUCTURE,
                "selected action produced physical structural UNKNOWN",
            )

        selected_program = factor_motor.actuate_selected_factor_action(
            incumbent,
            motor_decision,
        )
        actuator_calls = 1
        final_observation = root_observation.observe_current_program_root(
            selected_program,
            cases,
        )
        final_structure = structure.observe_program_structure(selected_program)
        if structure_prediction.predicted.paths != final_structure.paths:
            raise RuntimeError("predicted structure differs from real reobservation")
        return finish(STATUS_SELECTED, None)


def build_public_hierarchical_prompt_h1() -> FrozenHierarchicalPromptH1:
    base_model = base_h1.build_public_typed_factor_h1_root()
    return FrozenHierarchicalPromptH1(
        base_model=base_model,
        checkpoint_sha256=_model_checkpoint_sha256(base_model.checkpoint_sha256),
    )


@dataclass(frozen=True, slots=True)
class PublicPromptH1TaskResult:
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
    leading_value_tier_current_pa: tuple[float, ...]
    value_stop_in_leading_tier: bool
    prompt_status: str | None
    prompt_pattern: dict[str, object] | None
    prompt_matching_tier_cells: tuple[int, ...]
    selected_candidate_cell: int | None
    selected_action: dict[str, object] | None
    final_tokens: tuple[str, ...]
    final_outputs: tuple[object, ...]
    desired_outputs: tuple[object, ...]
    structure_prediction_calls: int
    actuator_calls: int
    operator_races_unique: bool
    prompt_race_unique: bool
    motor_races_unique: bool
    structure_races_unique: bool
    checkpoint_immutable: bool


def _pattern_record(
    pattern: prompt_factors.PromptOperandPattern | None,
) -> dict[str, object] | None:
    if pattern is None:
        return None
    return {
        "operator": pattern.operator,
        "left_source": pattern.left_source,
        "right_source": pattern.right_source,
        "literal": pattern.literal,
    }


def run_public_task(
    model: FrozenHierarchicalPromptH1,
    task: PublicPromptH1Task,
) -> PublicPromptH1TaskResult:
    """Run inference first; consult the separate validation target afterward."""

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
    passed = (
        decision.status == STATUS_SELECTED
        and final_program.tokens == target.expected_tokens
        and final_outputs == task.desired_outputs
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
    prompt_observation = (
        None
        if decision.prompt_decision is None
        else decision.prompt_decision.observation
    )
    return PublicPromptH1TaskResult(
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
        leading_value_tier_current_pa=leading_currents,
        value_stop_in_leading_tier=(
            decision.value_tier is not None
            and decision.value_tier.stop_cell
            in decision.value_tier.rounds[0].tier_cells
        ),
        prompt_status=(
            None
            if decision.prompt_decision is None
            else decision.prompt_decision.status
        ),
        prompt_pattern=_pattern_record(
            None if prompt_observation is None else prompt_observation.pattern
        ),
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
        final_tokens=final_program.tokens,
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
class PublicPromptH1MatrixResult:
    schema: str
    verdict: str
    passed_tasks: int
    total_tasks: int
    wall_time_seconds: float
    literal_bank: tuple[int, ...]
    model_checkpoint_sha256: str
    tasks: tuple[PublicPromptH1TaskResult, ...]
    retry_policy: str
    claim_boundary: str


def run_public_matrix() -> PublicPromptH1MatrixResult:
    """Spend the four predeclared tasks once; callers must guard this boundary."""

    started = perf_counter()
    model = build_public_hierarchical_prompt_h1()
    results = tuple(run_public_task(model, task) for task in PUBLIC_TASKS)
    passed = sum(task.passed for task in results)
    return PublicPromptH1MatrixResult(
        schema=SCHEMA,
        verdict=PASS_VERDICT if passed == len(results) else STOP_VERDICT,
        passed_tasks=passed,
        total_tasks=len(results),
        wall_time_seconds=perf_counter() - started,
        literal_bank=PUBLIC_LITERAL_BANK,
        model_checkpoint_sha256=model.checkpoint_sha256,
        tasks=results,
        retry_policy=(
            "One four-task public-development attempt; no retry, prompt, grid, "
            "incumbent, literal-bank, tier, current, or task-order tuning."
        ),
        claim_boundary=(
            "Visible public H=1 Int/Bool factorization only: target-free root "
            "operator successors, complete physical leading value tiers, and "
            "generic prompt binding before one real actuation. Not List, H>1, "
            "Simple-SWE, held-out, protected, M0, C1, or coding capability."
        ),
    )


def _task_record(task: PublicPromptH1Task) -> dict[str, object]:
    return {
        "name": task.name,
        "prompt": task.prompt,
        "signature": task.signature.to_record(),
        "cases": task.cases,
        "incumbent_tokens": task.incumbent_tokens,
        "desired_outputs": task.desired_outputs,
    }


def _fresh_commitment() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "literal_bank": PUBLIC_LITERAL_BANK,
        "tasks": tuple(_task_record(task) for task in PUBLIC_TASKS),
        "validation_targets": tuple(asdict(target) for target in PUBLIC_TARGETS),
        "planner_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _acquire_spend_marker(commitment: dict[str, object]) -> bool:
    marker = {
        "schema": f"{SCHEMA}-spend",
        "commitment": commitment,
        "commitment_sha256": sha256(_canonical_json(commitment)).hexdigest(),
        "disposition": (
            "Retain after PASS, STOP, exception, interruption, or crash; the "
            "four-task public-development boundary must never rerun."
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
    if len(PUBLIC_TASKS) != len(PUBLIC_TARGETS) or len(PUBLIC_TASKS) != 4:
        raise RuntimeError("fresh hierarchical boundary must contain four tasks")
    names = tuple(task.name for task in PUBLIC_TASKS)
    if len(set(names)) != 4 or names != tuple(target.name for target in PUBLIC_TARGETS):
        raise RuntimeError("fresh task and validation-target identities differ")
    if PUBLIC_LITERAL_BANK != tuple(range(-4, 5)):
        raise RuntimeError("fresh hierarchical literal bank changed")


def main() -> int:
    if _ARTIFACT_PATH.exists():
        prior = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "artifact": str(_ARTIFACT_PATH),
                    "verdict": prior.get("verdict"),
                    "guard": "existing artifact; fresh matrix not rerun",
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
                    "guard": "fresh matrix already spent; no rerun",
                },
                sort_keys=True,
            )
        )
        return 2

    _validate_predeclared_boundary()
    commitment = _fresh_commitment()
    if not _acquire_spend_marker(commitment):
        raise RuntimeError("fresh hierarchical spend marker already exists")
    result = run_public_matrix()
    payload = asdict(result)
    payload["commitment_sha256"] = sha256(_canonical_json(commitment)).hexdigest()
    _ARTIFACT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(_ARTIFACT_PATH),
                "verdict": result.verdict,
                "passed_tasks": result.passed_tasks,
                "total_tasks": result.total_tasks,
                "commitment_sha256": payload["commitment_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if result.verdict == PASS_VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FrozenHierarchicalPromptH1",
    "HierarchicalPromptH1Decision",
    "PASS_VERDICT",
    "PUBLIC_BOOL_CASES",
    "PUBLIC_LITERAL_BANK",
    "PUBLIC_TARGETS",
    "PUBLIC_TASKS",
    "PUBLIC_TWO_INT_CASES",
    "PublicPromptH1MatrixResult",
    "PublicPromptH1Task",
    "PublicPromptH1TaskResult",
    "SCHEMA",
    "STATUS_SELECTED",
    "STOP_VERDICT",
    "build_public_hierarchical_prompt_h1",
    "run_public_matrix",
    "run_public_task",
]
