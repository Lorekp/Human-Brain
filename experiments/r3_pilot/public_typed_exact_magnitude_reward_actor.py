"""Exact-magnitude cell projection for the banked public reward actor.

The preceding actor receives cumulative magnitude-threshold activity.  This
stage inserts one fixed opponent projection: within each 32-cell threshold
group, exact cell ``m`` receives ``+threshold_m - threshold_(m+1)`` (with no
successor for magnitude 32).  A fresh fitted-cell population turns those fixed
synaptic currents into 256 physical exact-magnitude features; the four banked
equality cells pass through the same physical population unchanged.

Everything after that projection is the banked masked-sink actor: seed 17,
anatomy, rates, ten actions, centered advantage, and the 3 x 10 x 10 public
schedule are unchanged.  This file defines no composition target.
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

from experiments.r3_pilot import (  # noqa: E402
    public_typed_reward_modulated_edit_actor as banked,
)
from experiments.r3_pilot.public_typed_sensorimotor_edit_controller import (  # noqa: E402
    _drive_spike_mask,
)
from humanbrain.cells import AdaptiveGlifPopulation  # noqa: E402
from humanbrain.readout import SparseCellReadout  # noqa: E402

_MAGNITUDE_GROUPS = 8
_MAGNITUDES_PER_GROUP = 32
_RELATION_CELLS = _MAGNITUDE_GROUPS * _MAGNITUDES_PER_GROUP
_EQUALITY_CELLS = 4
_FEATURE_CELLS = _RELATION_CELLS + _EQUALITY_CELLS
_PROJECTION_CURRENT_SCALE_PA = 20_000.0
_PROJECTION_CURRENT_LIMIT_PA = 5_000.0
_PROJECTION_STEPS = 30
_ARTIFACT_PATH = Path(__file__).with_name(
    "public_typed_exact_magnitude_reward_actor_probe.json"
)

if _FEATURE_CELLS != banked._FEATURE_CELLS:
    raise RuntimeError("exact projection must preserve the banked 260-cell width")


class ExactMagnitudeTransitionCellEncoder:
    """Fixed opponent projection over the banked cumulative transition cells."""

    def __init__(self) -> None:
        self._cumulative = banked._TransitionCellEncoder()
        self._exact_cells = AdaptiveGlifPopulation(_FEATURE_CELLS)
        self._weights = self._projection_weights()

    @staticmethod
    def _projection_weights() -> np.ndarray:
        weights = np.zeros((_FEATURE_CELLS, _FEATURE_CELLS), dtype=np.float64)
        for group in range(_MAGNITUDE_GROUPS):
            start = group * _MAGNITUDES_PER_GROUP
            for magnitude in range(_MAGNITUDES_PER_GROUP):
                row = start + magnitude
                weights[row, row] = 1.0
                if magnitude + 1 < _MAGNITUDES_PER_GROUP:
                    weights[row, row + 1] = -1.0
        for equality in range(_EQUALITY_CELLS):
            row = _RELATION_CELLS + equality
            weights[row, row] = 1.0
        return weights

    @property
    def projection_sha256(self) -> str:
        return sha256(self._weights.tobytes()).hexdigest()

    def project_cumulative(self, cumulative_features: Sequence[float]) -> np.ndarray:
        cumulative = np.ascontiguousarray(cumulative_features, dtype=np.float64)
        if cumulative.shape != (_FEATURE_CELLS,):
            raise ValueError("exact projection needs 260 cumulative cell features")
        if not np.isfinite(cumulative).all() or np.any(cumulative < 0.0):
            raise ValueError("cumulative cell features must be finite and nonnegative")
        opponent_net = np.maximum(self._weights @ cumulative, 0.0)
        currents_pa = np.ascontiguousarray(
            np.clip(
                opponent_net * _PROJECTION_CURRENT_SCALE_PA,
                0.0,
                _PROJECTION_CURRENT_LIMIT_PA,
            ),
            dtype=np.float64,
        )
        return np.ascontiguousarray(
            _drive_spike_mask(
                self._exact_cells,
                currents_pa,
                steps=_PROJECTION_STEPS,
            ).astype(np.float64)
        )

    def encode(
        self,
        current_outputs: Sequence[int],
        desired_outputs: Sequence[int],
    ) -> np.ndarray:
        return self.project_cumulative(
            self._cumulative.encode(current_outputs, desired_outputs)
        )


class ExactMagnitudeRewardModulatedTypedEditActor(
    banked.RewardModulatedTypedEditActor
):
    """Banked actor with only its raw relation encoder replaced."""

    def __init__(self, seed: int = banked.SEED) -> None:
        super().__init__(seed)
        self._encoder = ExactMagnitudeTransitionCellEncoder()

    def freeze(self) -> FrozenExactMagnitudeRewardActor:
        arrays = banked._freeze_arrays(self._readout.checkpoint_arrays())
        counters = self._readout.checkpoint_counters()
        return FrozenExactMagnitudeRewardActor(
            seed=self.seed,
            arrays=arrays,
            counters=dict(counters),
            output_heterogeneity_pa=np.array(self._output_heterogeneity_pa, copy=True),
            pacemaker_synapses_pa=np.array(self._pacemaker_synapses_pa, copy=True),
            checkpoint_sha256=banked._checkpoint_sha256(
                arrays,
                counters,
                self._output_heterogeneity_pa,
                self._pacemaker_synapses_pa,
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenExactMagnitudeRewardActor:
    seed: int
    arrays: tuple[banked._FrozenArray, ...]
    counters: dict[str, int]
    output_heterogeneity_pa: np.ndarray
    pacemaker_synapses_pa: np.ndarray
    checkpoint_sha256: str

    def select(
        self,
        current_outputs: Sequence[int],
        desired_outputs: Sequence[int],
    ) -> banked.ActionSelection:
        eligibility = ExactMagnitudeTransitionCellEncoder().encode(
            current_outputs, desired_outputs
        )
        relation_sha = sha256(eligibility.tobytes()).hexdigest()
        readout = SparseCellReadout.from_checkpoint(
            output_cells=banked._OUTPUT_CELLS,
            reservoir_cells=_FEATURE_CELLS,
            fan_in=banked._READOUT_FAN_IN,
            reservoir_eligibility=eligibility,
            source_block_size=_FEATURE_CELLS,
            source_block_decays=np.zeros(1, dtype=np.float64),
            output_heterogeneity_pa=np.array(self.output_heterogeneity_pa, copy=True),
            pacemaker_synapses_pa=np.array(self.pacemaker_synapses_pa, copy=True),
            arrays=banked._thaw_arrays(self.arrays),
            counters=dict(self.counters),
        )
        before_sha = banked._checkpoint_sha256(
            banked._freeze_arrays(readout.checkpoint_arrays()),
            readout.checkpoint_counters(),
            self.output_heterogeneity_pa,
            self.pacemaker_synapses_pa,
        )
        all_currents = readout.current(banked._OUTPUT_BASE_CURRENT_PA)
        action_currents = np.ascontiguousarray(
            all_currents[: len(banked.ACTIONS)]
        )
        winner, spike_step, simultaneous = banked._physical_action_race(
            action_currents
        )
        after_sha = banked._checkpoint_sha256(
            banked._freeze_arrays(readout.checkpoint_arrays()),
            readout.checkpoint_counters(),
            self.output_heterogeneity_pa,
            self.pacemaker_synapses_pa,
        )
        return banked.ActionSelection(
            action=banked.ACTIONS[winner].label,
            winner_cell=winner,
            spike_step=spike_step,
            currents_pa=tuple(float(value) for value in action_currents),
            masked_sink_current_pa=float(all_currents[banked._SINK_CELL]),
            simultaneous_spike_candidates=simultaneous,
            relation_sha256=relation_sha,
            checkpoint_before_sha256=before_sha,
            checkpoint_after_sha256=after_sha,
        )


def train_public_actor(
) -> tuple[FrozenExactMagnitudeRewardActor, banked.CurriculumSummary]:
    """Run the unchanged banked public schedule through exact features."""
    actor = ExactMagnitudeRewardModulatedTypedEditActor(banked.SEED)
    initial_sha = actor.checkpoint_sha256
    identity = banked._identity_program()
    positive = neutral = negative = 0
    coverage = {action.label: 0 for action in banked.ACTIONS}
    reward_audit: list[banked.GoalAdvantageAudit] = []

    for epoch in range(banked.CURRICULUM_EPOCHS):
        for goal_index, goal_action in enumerate(banked.ACTIONS):
            grid_index = (epoch + goal_index) % len(
                banked.TRAINING_ARGUMENT_GRIDS
            )
            arguments = banked.TRAINING_ARGUMENT_GRIDS[grid_index]
            current = banked._execute(identity, arguments)
            desired = banked._execute(
                banked._apply_action(identity, goal_action), arguments
            )
            resulting_by_action = tuple(
                banked._execute(
                    banked._apply_action(identity, performed_action), arguments
                )
                for performed_action in banked.ACTIONS
            )
            raw_rewards = tuple(
                banked.signed_l1_improvement(current, desired, resulting)
                for resulting in resulting_by_action
            )
            advantages = banked.centered_advantages(raw_rewards)
            reward_audit.append(
                banked.GoalAdvantageAudit(
                    epoch=epoch + 1,
                    goal=goal_action.label,
                    argument_grid=grid_index,
                    raw_rewards=tuple(
                        (action.label, reward)
                        for action, reward in zip(
                            banked.ACTIONS, raw_rewards, strict=True
                        )
                    ),
                    mean_raw_reward=float(np.mean(raw_rewards)),
                    centered_advantages=tuple(
                        (action.label, advantage)
                        for action, advantage in zip(
                            banked.ACTIONS, advantages, strict=True
                        )
                    ),
                )
            )
            for performed_action, advantage in zip(
                banked.ACTIONS, advantages, strict=True
            ):
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
    summary = banked.CurriculumSummary(
        epochs=banked.CURRICULUM_EPOCHS,
        public_goals=tuple(action.label for action in banked.ACTIONS),
        argument_grids=banked.TRAINING_ARGUMENT_GRIDS,
        motor_trials=(
            banked.CURRICULUM_EPOCHS
            * len(banked.ACTIONS)
            * len(banked.ACTIONS)
        ),
        motor_coverage=tuple(
            (action.label, coverage[action.label]) for action in banked.ACTIONS
        ),
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
    frozen: FrozenExactMagnitudeRewardActor,
) -> banked.AtomicReplayResult:
    """Run the same fresh ten-action atomic replay, without composition."""
    replay = banked.run_public_atomic_replay(frozen)
    return banked.AtomicReplayResult(
        schema="public-typed-exact-magnitude-atomic-replay-v1",
        arguments=replay.arguments,
        arms=replay.arms,
        passed_goals=replay.passed_goals,
        total_goals=replay.total_goals,
        verdict=(
            "PASS_EXACT_MAGNITUDE_ATOMIC_REPLAY"
            if replay.passed_goals == replay.total_goals
            else "STOP_EXACT_MAGNITUDE_ATOMIC_REPLAY"
        ),
        claim_boundary=(
            "Fresh public scalar exact-magnitude one-step development replay only; "
            "not composition, Simple-SWE, M0, held-out, C1, or capability evidence."
        ),
    )


@dataclass(frozen=True, slots=True)
class ExactMagnitudeStageResult:
    schema: str
    seed: int
    projection_sha256: str
    training: banked.CurriculumSummary
    replay: banked.AtomicReplayResult
    verdict: str
    retry_policy: str


def run_public_stage() -> ExactMagnitudeStageResult:
    frozen, training = train_public_actor()
    replay = run_public_atomic_replay(frozen)
    return ExactMagnitudeStageResult(
        schema="public-typed-exact-magnitude-reward-actor-v1",
        seed=banked.SEED,
        projection_sha256=ExactMagnitudeTransitionCellEncoder().projection_sha256,
        training=training,
        replay=replay,
        verdict=replay.verdict,
        retry_policy=(
            "One fixed seed-17 public atomic replay; no projection, width, reward, "
            "epoch, order, rate, fan-in, or action-bank sweep."
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
    result = run_public_stage()
    _ARTIFACT_PATH.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": _ARTIFACT_PATH.relative_to(REPO_ROOT).as_posix(),
                "passed_goals": result.replay.passed_goals,
                "total_goals": result.replay.total_goals,
                "verdict": result.verdict,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
