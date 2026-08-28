import json
from pathlib import Path

import numpy as np

from experiments.r3_pilot.public_typed_exact_magnitude_reward_actor import (
    ExactMagnitudeTransitionCellEncoder,
)


ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "r3_pilot"
    / "public_typed_exact_magnitude_reward_actor_probe.json"
)


def test_opponent_projection_emits_only_exact_bins_and_preserves_equality() -> None:
    encoder = ExactMagnitudeTransitionCellEncoder()
    cumulative = np.zeros(260, dtype=np.float64)
    cumulative[0:3] = 1.0
    cumulative[32:37] = 0.25
    cumulative[224:256] = 1.0
    cumulative[256] = 1.0
    cumulative[258] = 0.5

    exact = encoder.project_cumulative(cumulative)

    assert set(np.flatnonzero(exact)) == {2, 36, 255, 256, 258}


def test_raw_uniform_shift_and_equality_take_distinct_exact_cell_paths() -> None:
    encoder = ExactMagnitudeTransitionCellEncoder()
    current = (-12, -5, 2, 9)

    plus_three = encoder.encode(current, (-9, -2, 5, 12))
    equality = encoder.encode(current, current)

    assert set(np.flatnonzero(plus_three)) == {2, 66, 194}
    assert set(np.flatnonzero(equality)) == {256, 257, 259}


def test_spent_exact_magnitude_atomic_replay_is_banked() -> None:
    result = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert result["seed"] == 17
    assert result["training"]["motor_trials"] == 300
    assert all(count == 30 for _, count in result["training"]["motor_coverage"])
    assert result["replay"]["passed_goals"] == 10
    assert result["replay"]["total_goals"] == 10
    assert all(
        arm["passed"]
        and not arm["tied"]
        and arm["goal"] == arm["selected_action"]
        and arm["selection"]["checkpoint_before_sha256"]
        == arm["selection"]["checkpoint_after_sha256"]
        == result["training"]["frozen_checkpoint_sha256"]
        for arm in result["replay"]["arms"]
    )
    assert result["verdict"] == "PASS_EXACT_MAGNITUDE_ATOMIC_REPLAY"
