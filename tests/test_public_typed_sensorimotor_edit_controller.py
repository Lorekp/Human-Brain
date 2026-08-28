import json
from pathlib import Path

from experiments.r3_pilot.public_typed_sensorimotor_edit_controller import (
    MOTOR_EPISODES,
    SEED,
)


ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "r3_pilot"
    / "public_typed_sensorimotor_edit_controller_probe.json"
)


def test_spent_sensorimotor_inverse_edit_gate_is_banked_without_retry() -> None:
    result = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert result["seed"] == SEED == 17
    assert result["motor_episodes_planned"] == MOTOR_EPISODES == 100
    assert result["adjudicative_attempt"]["completed_learning_episodes_before_stop"] == 1
    assert result["adjudicative_attempt"]["failure_stage"] == "motor_episode_2_race"
    assert result["diagnostic_replay"]["simultaneous_spike_candidates"] == 4
    assert result["adjudicative_attempt"]["gate_arms_opened"] is False
    assert result["verdict"] == "STOP_NON_UNIQUE_MOTOR_RACE"
