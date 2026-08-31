"""Reward-modulated cell actor for public scalar typed-program edits.

The actor's interface is deliberately small: ``learn`` receives raw current
and desired outputs, the action that was physically performed, and one signed
generic improvement scalar; ``select`` returns one fitted-cell-raced action.
Program parsing, goal-relation cell encoding, sparse local plasticity, the
masked rejection sink, checkpointing, and typed lowering stay behind that
seam.

The fixed public curriculum performs every motor action in deterministic
round-robin order.  It never asks the actor to infer an action from a host
policy table.  Public atomic replay failed, so the G-RMA1 composition target
stayed unopened and the obsolete gate runner is intentionally absent.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for search_path in (REPO_ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.r3_pilot.public_typed_sensorimotor_edit_controller import (  # noqa: E402
    _FEATURE_CELLS,
    _TransitionCellEncoder,
)
from humanbrain import typed_program as tp  # noqa: E402
from humanbrain.cells import AdaptiveGlifPopulation  # noqa: E402
from humanbrain.readout import SparseCellReadout  # noqa: E402
from humanbrain.typed_program_types import (  # noqa: E402
    ProgramParameter,
    ProgramType,
    TypedSignature,
)

SEED = 17
CURRICULUM_EPOCHS = 3
ATOMIC_REPLAY_ARGUMENTS = (-10, -3, 4, 7)
TRAINING_ARGUMENT_GRIDS = (
    (-12, -5, 2, 9),
    (-11, -4, 3, 8),
)

_OUTPUT_BASE_CURRENT_PA = 600.0
_OUTPUT_RACE_STEPS = 600
_LEARNING_RATE_PA = 32.0
_MAXIMUM_WEIGHT_PA = 1_600.0
_READOUT_FAN_IN = _FEATURE_CELLS // SparseCellReadout.CANDIDATE_FACTOR
_ARTIFACT_PATH = Path(__file__).with_name(
    "public_typed_reward_modulated_edit_actor_probe.json"
)


@dataclass(frozen=True, slots=True)
class EditAction:
    label: str
    kind: str
    amount: int = 0


ACTIONS = (
    EditAction("DONE", "done"),
    EditAction("WRAP_ABS", "absolute"),
    *tuple(EditAction(f"WRAP_ADD[{value}]", "add", value) for value in range(1, 5)),
    *tuple(
        EditAction(f"WRAP_SUB[{value}]", "subtract", value) for value in range(1, 5)
    ),
)
_SINK_CELL = len(ACTIONS)
_OUTPUT_CELLS = len(ACTIONS) + 1


class NonUniqueActionRace(RuntimeError):
    """The physical action population produced no unique first spiker."""


@dataclass(frozen=True, slots=True)
class ActionSelection:
    action: str
    winner_cell: int
    spike_step: int
    currents_pa: tuple[float, ...]
    masked_sink_current_pa: float
    simultaneous_spike_candidates: int
    relation_sha256: str
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str


@dataclass(frozen=True, slots=True)
class LearningEvent:
    performed_action: str
    signed_improvement: float
    modulation_gate: float
    target_cell: int | None
    opponent_cell: int | None
    local_update_applied: bool
    structural_recruited: bool
    relation_sha256: str
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str


@dataclass(frozen=True, slots=True)
class GoalAdvantageAudit:
    epoch: int
    goal: str
    argument_grid: int
    raw_rewards: tuple[tuple[str, float], ...]
    mean_raw_reward: float
    centered_advantages: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class CurriculumSummary:
    epochs: int
    public_goals: tuple[str, ...]
    argument_grids: tuple[tuple[int, ...], ...]
    motor_trials: int
    motor_coverage: tuple[tuple[str, int], ...]
    positive_advantages: int
    neutral_advantages: int
    negative_advantages: int
    local_updates: int
    structural_recruitments: int
    initial_checkpoint_sha256: str
    frozen_checkpoint_sha256: str
    frozen_before_gate_target: bool
    reward_audit: tuple[GoalAdvantageAudit, ...]


@dataclass(frozen=True, slots=True)
class AtomicReplayArm:
    goal: str
    desired_outputs: tuple[int, ...]
    selected_action: str | None
    passed: bool
    tied: bool
    error: str | None
    selection: ActionSelection | None


@dataclass(frozen=True, slots=True)
class AtomicReplayResult:
    schema: str
    arguments: tuple[int, ...]
    arms: tuple[AtomicReplayArm, ...]
    passed_goals: int
    total_goals: int
    verdict: str
    claim_boundary: str


@dataclass(frozen=True, slots=True)
class _FrozenArray:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes


def _freeze_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[_FrozenArray, ...]:
    return tuple(
        _FrozenArray(
            name,
            np.ascontiguousarray(arrays[name]).dtype.str,
            tuple(np.ascontiguousarray(arrays[name]).shape),
            np.ascontiguousarray(arrays[name]).tobytes(),
        )
        for name in sorted(arrays)
    )


def _thaw_arrays(arrays: Sequence[_FrozenArray]) -> dict[str, np.ndarray]:
    return {
        value.name: np.frombuffer(value.data, dtype=np.dtype(value.dtype))
        .reshape(value.shape)
        .copy()
        for value in arrays
    }


def _checkpoint_sha256(
    arrays: Sequence[_FrozenArray],
    counters: Mapping[str, int],
    output_heterogeneity_pa: np.ndarray,
    pacemaker_synapses_pa: np.ndarray,
) -> str:
    digest = sha256()
    for value in arrays:
        digest.update(value.name.encode("utf-8"))
        digest.update(value.dtype.encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.data)
    digest.update(
        json.dumps(dict(sorted(counters.items())), separators=(",", ":")).encode(
            "ascii"
        )
    )
    digest.update(np.ascontiguousarray(output_heterogeneity_pa).tobytes())
    digest.update(np.ascontiguousarray(pacemaker_synapses_pa).tobytes())
    return digest.hexdigest()


def _literal_tokens(value: int) -> tuple[str, ...]:
    return ("integer", "+", *tuple(str(value)), "end")


def _apply_action(program: tp.Program, action: EditAction) -> tp.Program:
    root = next(site for site in tp.program_edit_sites(program) if not site.path)
    if action.kind == "done":
        edited = program
    elif action.kind == "absolute":
        edited = tp.replace_subtree(program, (), ("absolute", *root.tokens))
    elif action.kind in {"add", "subtract"}:
        edited = tp.replace_subtree(
            program,
            (),
            (action.kind, *root.tokens, *_literal_tokens(action.amount)),
        )
    else:
        raise AssertionError("unknown typed edit action")
    tp.compile(edited)
    return edited


def _signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("value", integer),),
        result=integer,
    )


def _identity_program() -> tp.Program:
    program = tp.Program(("variable-0",), _signature())
    tp.compile(program)
    return program


def _execute(program: tp.Program, arguments: Sequence[int]) -> tuple[int, ...]:
    compiled = tp.compile(program)
    namespace: dict[str, object] = {}
    exec(
        compiled.source,
        {"__builtins__": {"abs": abs, "max": max, "min": min, "sum": sum}},
        namespace,
    )
    function = namespace[program.signature.entry_point]
    outputs: list[int] = []
    for value in arguments:
        result = function(value)  # type: ignore[operator]
        if type(result) is not int:
            raise TypeError("scalar actor observation must be an int")
        outputs.append(result)
    return tuple(outputs)


def signed_l1_improvement(
    current_outputs: Sequence[int],
    desired_outputs: Sequence[int],
    resulting_outputs: Sequence[int],
) -> float:
    """Return one action-agnostic, bounded progress signal.

    Reaching or holding the exact public goal receives the generic terminal
    reward ``+1``.  Otherwise the signal is the normalized L1 error reduction.
    """
    current = tuple(current_outputs)
    desired = tuple(desired_outputs)
    resulting = tuple(resulting_outputs)
    if not current or len(current) != len(desired) or len(current) != len(resulting):
        raise ValueError("L1 improvement needs three non-empty paired observations")
    if not all(type(value) is int for values in (current, desired, resulting) for value in values):
        raise TypeError("L1 observations must contain ints")
    before_error = sum(abs(value - target) for value, target in zip(current, desired))
    after_error = sum(abs(value - target) for value, target in zip(resulting, desired))
    if after_error == 0:
        return 1.0
    return float(np.clip((before_error - after_error) / max(before_error, 1), -1.0, 1.0))


def centered_advantages(raw_rewards: Sequence[float]) -> tuple[float, ...]:
    """Center one exhaustive motor batch without reading action identities."""
    rewards = np.asarray(tuple(raw_rewards), dtype=np.float64)
    if rewards.ndim != 1 or rewards.size != len(ACTIONS):
        raise ValueError("centered advantage needs one reward per fixed motor action")
    if not np.isfinite(rewards).all():
        raise ValueError("raw motor rewards must be finite")
    return tuple(float(value) for value in np.clip(rewards - rewards.mean(), -1.0, 1.0))


def _physical_action_race(currents_pa: np.ndarray) -> tuple[int, int, int]:
    if currents_pa.shape != (len(ACTIONS),):
        raise ValueError("action race current has the wrong shape")
    population = AdaptiveGlifPopulation(len(ACTIONS))
    for step in range(_OUTPUT_RACE_STEPS):
        fired = np.flatnonzero(population.step(currents_pa))
        if fired.size:
            if fired.size != 1:
                raise NonUniqueActionRace(
                    f"action race produced {fired.size} simultaneous first spikes"
                )
            return int(fired[0]), step, int(fired.size)
    raise NonUniqueActionRace("action race produced no spike within its fixed bound")


class RewardModulatedTypedEditActor:
    """Trainable sparse-synapse actor behind the ``learn``/``select`` seam."""

    def __init__(self, seed: int = SEED) -> None:
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self._eligibility = np.zeros(_FEATURE_CELLS, dtype=np.float64)
        self._output_heterogeneity_pa = np.ascontiguousarray(
            rng.normal(0.0, 4.0, _OUTPUT_CELLS), dtype=np.float64
        )
        self._pacemaker_synapses_pa = np.zeros(_OUTPUT_CELLS, dtype=np.float64)
        self._readout = SparseCellReadout.create(
            output_cells=_OUTPUT_CELLS,
            reservoir_cells=_FEATURE_CELLS,
            fan_in=_READOUT_FAN_IN,
            rng=rng,
            reservoir_eligibility=self._eligibility,
            source_block_size=_FEATURE_CELLS,
            source_block_decays=np.zeros(1, dtype=np.float64),
            output_heterogeneity_pa=self._output_heterogeneity_pa,
            pacemaker_synapses_pa=self._pacemaker_synapses_pa,
        )
        self._encoder = _TransitionCellEncoder()

    @property
    def checkpoint_sha256(self) -> str:
        return _checkpoint_sha256(
            _freeze_arrays(self._readout.checkpoint_arrays()),
            self._readout.checkpoint_counters(),
            self._output_heterogeneity_pa,
            self._pacemaker_synapses_pa,
        )

    def learn(
        self,
        current_outputs: Sequence[int],
        desired_outputs: Sequence[int],
        performed_action: str,
        signed_improvement: float,
    ) -> LearningEvent:
        """Apply one local reward-modulated motor event."""
        try:
            action_cell = next(
                index
                for index, action in enumerate(ACTIONS)
                if action.label == performed_action
            )
        except StopIteration as error:
            raise ValueError("performed action is outside the fixed motor bank") from error
        modulation = float(signed_improvement)
        if not np.isfinite(modulation):
            raise ValueError("signed improvement must be finite")
        modulation = float(np.clip(modulation, -1.0, 1.0))
        self._eligibility[:] = self._encoder.encode(current_outputs, desired_outputs)
        relation_sha = sha256(self._eligibility.tobytes()).hexdigest()
        before_sha = self.checkpoint_sha256

        # Motor efference copy establishes one local contact opportunity for
        # both the performed action and its hidden opponent before reward is
        # known.  target==winner records the post event/apposition but performs
        # no signed weight update.  This avoids the zero-contact dead start
        # without seeding any task/action association from the host.
        recruited = False
        for seed_cell in (action_cell, _SINK_CELL):
            self._readout.teach(
                target=seed_cell,
                winner=seed_cell,
                learning_rate_pa=_LEARNING_RATE_PA,
                maximum_weight_pa=_MAXIMUM_WEIGHT_PA,
                target_post_gate=1.0,
                winner_post_gate=1.0,
                activity_decay=0.99,
            )
            presentation = self._readout.advance_target(seed_cell)
            seeded, _ = self._readout.turnover(
                output=seed_cell,
                structural_event=presentation,
                random_seed=self.seed,
                minimum_age=2,
                turnover_probability=0.0,
                learning_rate_pa=_LEARNING_RATE_PA,
                maximum_weight_pa=_MAXIMUM_WEIGHT_PA,
                homeostatic_rate=0.0,
            )
            recruited = recruited or seeded

        target_cell: int | None = None
        opponent_cell: int | None = None
        applied = False
        if modulation != 0.0:
            if modulation > 0.0:
                target_cell, opponent_cell = action_cell, _SINK_CELL
            else:
                target_cell, opponent_cell = _SINK_CELL, action_cell
            gate = abs(modulation)
            applied = self._readout.teach(
                target=target_cell,
                winner=opponent_cell,
                learning_rate_pa=_LEARNING_RATE_PA,
                maximum_weight_pa=_MAXIMUM_WEIGHT_PA,
                target_post_gate=gate,
                winner_post_gate=gate,
                activity_decay=0.99,
            )
            # This actor has no context-free action policy.  SparseCellReadout's
            # pacemaker channel is therefore clamped off after every local
            # event; only feature-contact synapses survive into selection.
            self._pacemaker_synapses_pa.fill(0.0)
        after_sha = self.checkpoint_sha256
        return LearningEvent(
            performed_action=performed_action,
            signed_improvement=modulation,
            modulation_gate=abs(modulation),
            target_cell=target_cell,
            opponent_cell=opponent_cell,
            local_update_applied=applied,
            structural_recruited=recruited,
            relation_sha256=relation_sha,
            checkpoint_before_sha256=before_sha,
            checkpoint_after_sha256=after_sha,
        )

    def select(
        self,
        current_outputs: Sequence[int],
        desired_outputs: Sequence[int],
    ) -> ActionSelection:
        self._eligibility[:] = self._encoder.encode(current_outputs, desired_outputs)
        relation_sha = sha256(self._eligibility.tobytes()).hexdigest()
        before_sha = self.checkpoint_sha256
        all_currents = self._readout.current(_OUTPUT_BASE_CURRENT_PA)
        # The rejection sink participates in local opponent plasticity only.
        # It is anatomically absent from the physical motor race.
        action_currents = np.ascontiguousarray(all_currents[: len(ACTIONS)])
        winner, spike_step, simultaneous = _physical_action_race(action_currents)
        after_sha = self.checkpoint_sha256
        return ActionSelection(
            action=ACTIONS[winner].label,
            winner_cell=winner,
            spike_step=spike_step,
            currents_pa=tuple(float(value) for value in action_currents),
            masked_sink_current_pa=float(all_currents[_SINK_CELL]),
            simultaneous_spike_candidates=simultaneous,
            relation_sha256=relation_sha,
            checkpoint_before_sha256=before_sha,
            checkpoint_after_sha256=after_sha,
        )

    def freeze(self) -> FrozenRewardModulatedTypedEditActor:
        arrays = _freeze_arrays(self._readout.checkpoint_arrays())
        counters = self._readout.checkpoint_counters()
        return FrozenRewardModulatedTypedEditActor(
            seed=self.seed,
            arrays=arrays,
            counters=dict(counters),
            output_heterogeneity_pa=np.array(self._output_heterogeneity_pa, copy=True),
            pacemaker_synapses_pa=np.array(self._pacemaker_synapses_pa, copy=True),
            checkpoint_sha256=_checkpoint_sha256(
                arrays,
                counters,
                self._output_heterogeneity_pa,
                self._pacemaker_synapses_pa,
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenRewardModulatedTypedEditActor:
    """Immutable actor restored into fresh cells for every selection."""

    seed: int
    arrays: tuple[_FrozenArray, ...]
    counters: dict[str, int]
    output_heterogeneity_pa: np.ndarray
    pacemaker_synapses_pa: np.ndarray
    checkpoint_sha256: str

    def select(
        self,
        current_outputs: Sequence[int],
        desired_outputs: Sequence[int],
    ) -> ActionSelection:
        eligibility = _TransitionCellEncoder().encode(current_outputs, desired_outputs)
        relation_sha = sha256(eligibility.tobytes()).hexdigest()
        readout = SparseCellReadout.from_checkpoint(
            output_cells=_OUTPUT_CELLS,
            reservoir_cells=_FEATURE_CELLS,
            fan_in=_READOUT_FAN_IN,
            reservoir_eligibility=eligibility,
            source_block_size=_FEATURE_CELLS,
            source_block_decays=np.zeros(1, dtype=np.float64),
            output_heterogeneity_pa=np.array(self.output_heterogeneity_pa, copy=True),
            pacemaker_synapses_pa=np.array(self.pacemaker_synapses_pa, copy=True),
            arrays=_thaw_arrays(self.arrays),
            counters=dict(self.counters),
        )
        before_sha = _checkpoint_sha256(
            _freeze_arrays(readout.checkpoint_arrays()),
            readout.checkpoint_counters(),
            self.output_heterogeneity_pa,
            self.pacemaker_synapses_pa,
        )
        all_currents = readout.current(_OUTPUT_BASE_CURRENT_PA)
        action_currents = np.ascontiguousarray(all_currents[: len(ACTIONS)])
        winner, spike_step, simultaneous = _physical_action_race(action_currents)
        after_sha = _checkpoint_sha256(
            _freeze_arrays(readout.checkpoint_arrays()),
            readout.checkpoint_counters(),
            self.output_heterogeneity_pa,
            self.pacemaker_synapses_pa,
        )
        return ActionSelection(
            action=ACTIONS[winner].label,
            winner_cell=winner,
            spike_step=spike_step,
            currents_pa=tuple(float(value) for value in action_currents),
            masked_sink_current_pa=float(all_currents[_SINK_CELL]),
            simultaneous_spike_candidates=simultaneous,
            relation_sha256=relation_sha,
            checkpoint_before_sha256=before_sha,
            checkpoint_after_sha256=after_sha,
        )


def train_public_actor(
) -> tuple[FrozenRewardModulatedTypedEditActor, CurriculumSummary]:
    """Run the fixed public one-step curriculum, then freeze before G-RMA1."""
    actor = RewardModulatedTypedEditActor(SEED)
    initial_sha = actor.checkpoint_sha256
    identity = _identity_program()
    positive = neutral = negative = 0
    coverage = {action.label: 0 for action in ACTIONS}
    reward_audit: list[GoalAdvantageAudit] = []

    for epoch in range(CURRICULUM_EPOCHS):
        for goal_index, goal_action in enumerate(ACTIONS):
            arguments = TRAINING_ARGUMENT_GRIDS[
                (epoch + goal_index) % len(TRAINING_ARGUMENT_GRIDS)
            ]
            current = _execute(identity, arguments)
            desired = _execute(_apply_action(identity, goal_action), arguments)
            resulting_by_action = tuple(
                _execute(_apply_action(identity, performed_action), arguments)
                for performed_action in ACTIONS
            )
            raw_rewards = tuple(
                signed_l1_improvement(current, desired, resulting)
                for resulting in resulting_by_action
            )
            advantages = centered_advantages(raw_rewards)
            reward_audit.append(
                GoalAdvantageAudit(
                    epoch=epoch + 1,
                    goal=goal_action.label,
                    argument_grid=(epoch + goal_index)
                    % len(TRAINING_ARGUMENT_GRIDS),
                    raw_rewards=tuple(
                        (action.label, reward)
                        for action, reward in zip(ACTIONS, raw_rewards, strict=True)
                    ),
                    mean_raw_reward=float(np.mean(raw_rewards)),
                    centered_advantages=tuple(
                        (action.label, advantage)
                        for action, advantage in zip(
                            ACTIONS, advantages, strict=True
                        )
                    ),
                )
            )
            for performed_action, advantage in zip(ACTIONS, advantages, strict=True):
                actor.learn(
                    current,
                    desired,
                    performed_action.label,
                    advantage,
                )
                coverage[performed_action.label] += 1
                positive += int(advantage > 0.0)
                neutral += int(advantage == 0.0)
                negative += int(advantage < 0.0)

    frozen = actor.freeze()
    counters = actor._readout.checkpoint_counters()
    summary = CurriculumSummary(
        epochs=CURRICULUM_EPOCHS,
        public_goals=tuple(action.label for action in ACTIONS),
        argument_grids=TRAINING_ARGUMENT_GRIDS,
        motor_trials=CURRICULUM_EPOCHS * len(ACTIONS) * len(ACTIONS),
        motor_coverage=tuple((action.label, coverage[action.label]) for action in ACTIONS),
        positive_advantages=positive,
        neutral_advantages=neutral,
        negative_advantages=negative,
        local_updates=counters["applied_updates"],
        structural_recruitments=counters["structural_recruitments"],
        initial_checkpoint_sha256=initial_sha,
        frozen_checkpoint_sha256=frozen.checkpoint_sha256,
        frozen_before_gate_target=True,
        reward_audit=tuple(reward_audit),
    )
    return frozen, summary


def run_public_atomic_replay(
    frozen: FrozenRewardModulatedTypedEditActor,
) -> AtomicReplayResult:
    """Run each public one-step goal once on the fixed fresh replay grid."""
    identity = _identity_program()
    current = _execute(identity, ATOMIC_REPLAY_ARGUMENTS)
    arms: list[AtomicReplayArm] = []
    for goal_action in ACTIONS:
        desired = _execute(
            _apply_action(identity, goal_action), ATOMIC_REPLAY_ARGUMENTS
        )
        try:
            selection = frozen.select(current, desired)
        except NonUniqueActionRace as error:
            arms.append(
                AtomicReplayArm(
                    goal=goal_action.label,
                    desired_outputs=desired,
                    selected_action=None,
                    passed=False,
                    tied=True,
                    error=str(error),
                    selection=None,
                )
            )
            continue
        arms.append(
            AtomicReplayArm(
                goal=goal_action.label,
                desired_outputs=desired,
                selected_action=selection.action,
                passed=selection.action == goal_action.label,
                tied=False,
                error=None,
                selection=selection,
            )
        )
    passed_goals = sum(arm.passed for arm in arms)
    return AtomicReplayResult(
        schema="public-typed-reward-modulated-atomic-replay-v1",
        arguments=ATOMIC_REPLAY_ARGUMENTS,
        arms=tuple(arms),
        passed_goals=passed_goals,
        total_goals=len(ACTIONS),
        verdict=(
            "PASS_PUBLIC_ATOMIC_REPLAY"
            if passed_goals == len(ACTIONS)
            else "STOP_PUBLIC_ATOMIC_REPLAY"
        ),
        claim_boundary=(
            "Fresh public scalar one-step development replay only; not G-RMA1, "
            "composition, Simple-SWE, M0, held-out, C1, or capability evidence."
        ),
    )


def main() -> int:
    if not _ARTIFACT_PATH.exists():
        raise RuntimeError("spent public atomic replay artifact is missing")
    stored = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {"status": "already-spent", "verdict": stored["verdict"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
