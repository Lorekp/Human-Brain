import ast
import json
from pathlib import Path

from experiments.r3_pilot import public_typed_exact_magnitude_g_rma1 as gate


ARTIFACT = Path(gate.__file__).with_name(
    "public_typed_exact_magnitude_g_rma1_probe.json"
)


def test_gate_source_places_all_commitments_before_the_fresh_target() -> None:
    source = Path(gate.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_g_rma1"
    )
    runner_source = ast.get_source_segment(source, runner)
    assert runner_source is not None

    readiness_at = runner_source.index("_verify_atomic_readiness()")
    train_at = runner_source.index("exact.train_public_actor()")
    verify_at = runner_source.index("_verify_frozen_commitments")
    arguments_at = runner_source.index("arguments = (-13, -6, 1, 10)")
    target_at = runner_source.index("target = banked._apply_action")
    assert readiness_at < train_at < verify_at < arguments_at < target_at
    assert "for step_number in range(1, 4):" in runner_source
    assert runner_source.count("frozen.select(") == 1
    assert "run_fixed_gate" not in source
    assert "GATE_ARGUMENTS" not in source


def test_spent_artifact_guard_never_calls_the_gate(monkeypatch) -> None:
    class SpentArtifact:
        @staticmethod
        def exists() -> bool:
            return True

        @staticmethod
        def read_text(*, encoding: str) -> str:
            assert encoding == "utf-8"
            return json.dumps({"verdict": "BANKED"})

    artifact = SpentArtifact()
    monkeypatch.setattr(gate, "_ARTIFACT_PATH", artifact)

    def forbidden() -> None:
        raise AssertionError("spent artifact guard called G-RMA1")

    monkeypatch.setattr(gate, "run_g_rma1", forbidden)
    assert gate.main() == 0


def test_spent_g_rma1_artifact_banks_the_terminal_three_edit_stop() -> None:
    result = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    steps = result["steps"]
    frozen_sha256 = result["commitments"]["frozen_checkpoint_sha256"]

    assert result["schema"] == "public-typed-exact-magnitude-g-rma1-v1"
    assert result["verdict"] == "STOP_G_RMA1"
    assert result["selection_attempts"] == result["applied_actions"] == len(steps) == 3
    assert [step["selected_action"] for step in steps] == [
        "WRAP_ADD[3]",
        "WRAP_ABS",
        "WRAP_ABS",
    ]
    assert result["unique_races"] is True
    assert all(
        step["selection"]["simultaneous_spike_candidates"] == 1
        for step in steps
    )

    assert result["frozen_checkpoint_immutable"] is True
    assert all(
        step["selection"]["checkpoint_before_sha256"]
        == step["selection"]["checkpoint_after_sha256"]
        == frozen_sha256
        for step in steps
    )
    assert result["retry_policy"] == (
        "One fixed seed-17 three-selection trajectory; no retry, seed, schedule, "
        "projection, reward, rate, fan-in, width, or action-bank sweep."
    )

    assert result["expected_actions"] == ["WRAP_ABS", "WRAP_ADD[3]", "DONE"]
    assert result["desired_outputs"] == [16, 9, 4, 13]
    assert result["final_outputs"] == [10, 3, 4, 13]
    assert result["expected_final_tokens"] == [
        "add",
        "absolute",
        "variable-0",
        "integer",
        "+",
        "3",
        "end",
    ]
    assert result["final_tokens"] == [
        "absolute",
        "absolute",
        "add",
        "variable-0",
        "integer",
        "+",
        "3",
        "end",
    ]
    assert result["exact_action_sequence"] is False
    assert result["exact_final_outputs"] is False
    assert result["exact_final_tokens"] is False
