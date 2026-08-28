import json
from pathlib import Path

from experiments.r3_pilot.public_typed_reward_modulated_edit_actor import (
    ACTIONS,
    ATOMIC_REPLAY_ARGUMENTS,
    TRAINING_ARGUMENT_GRIDS,
    RewardModulatedTypedEditActor,
    centered_advantages,
    signed_l1_improvement,
)


ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "r3_pilot"
    / "public_typed_reward_modulated_edit_actor_probe.json"
)


def test_reward_actor_uses_masked_sink_and_freezes_selection_state() -> None:
    actor = RewardModulatedTypedEditActor()
    current = (-12, -5, 2, 9)
    desired = (-9, -2, 5, 12)

    positive = actor.learn(current, desired, "WRAP_ADD[3]", 1.0)
    negative = actor.learn(current, desired, "WRAP_SUB[3]", -1.0)

    assert positive.target_cell == 4
    assert positive.opponent_cell == len(ACTIONS)
    assert negative.target_cell == len(ACTIONS)
    assert negative.opponent_cell == 8
    assert positive.local_update_applied and negative.local_update_applied
    assert positive.structural_recruited and negative.structural_recruited
    frozen = actor.freeze()
    selection = frozen.select(current, desired)
    assert selection.action in {action.label for action in ACTIONS}
    assert selection.simultaneous_spike_candidates == 1
    assert selection.checkpoint_before_sha256 == frozen.checkpoint_sha256
    assert selection.checkpoint_after_sha256 == frozen.checkpoint_sha256


def test_public_reward_and_curriculum_are_action_agnostic() -> None:
    current = (-12, -5, 2, 9)
    desired = (-9, -2, 5, 12)
    assert signed_l1_improvement(current, desired, desired) == 1.0
    assert signed_l1_improvement(current, desired, current) == 0.0
    assert signed_l1_improvement(current, desired, (-15, -8, -1, 6)) == -1.0
    advantages = centered_advantages(
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    assert advantages[0] == 0.9
    assert advantages[1:] == (-0.1,) * 9
    assert set(ATOMIC_REPLAY_ARGUMENTS).isdisjoint(
        value for grid in TRAINING_ARGUMENT_GRIDS for value in grid
    )


def test_failed_public_atomic_replay_is_banked_without_opening_composition() -> None:
    result = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert result["curriculum"]["motor_trials"] == 300
    assert result["atomic_replay"]["passed_goals"] == 4
    assert result["atomic_replay"]["ties"] == 0
    assert result["composition_gate_opened"] is False
    assert result["verdict"] == "STOP_PUBLIC_ATOMIC_REPLAY"
