import ast
import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from experiments.r3_pilot import public_typed_ast_site_planner as planner


@dataclass(frozen=True)
class _Site:
    path: tuple[int, ...]
    outputs: tuple[int, ...] | None


class _Observation:
    def __init__(self, sites: tuple[_Site, ...], arguments: tuple[int, ...]) -> None:
        self.sites = sites
        self.arguments = arguments
        self.sha256 = "o" * 64
        self.program_sha256 = "p" * 64

    def site(self, path: tuple[int, ...]) -> _Site:
        return next(site for site in self.sites if site.path == path)


class _FakeSuccessor:
    def __init__(self, done: tuple[planner.SuccessorState, ...]) -> None:
        self.done = done

    def predict_many(self, states, action):
        assert action == planner.DONE
        return self.done, ()


class _FakeModel:
    checkpoint_sha256 = "w" * 64

    def __init__(
        self,
        predictions: dict[
            tuple[tuple[int, ...], str],
            tuple[planner.SuccessorState, ...],
        ],
        done: tuple[planner.SuccessorState, ...],
    ) -> None:
        self.predictions = predictions
        self.successor = _FakeSuccessor(done)

    @property
    def live_checkpoint_sha256(self) -> str:
        return self.checkpoint_sha256

    def predict(self, observation, path, action):
        outputs = self.predictions.get(
            (path, action),
            tuple(planner.UNKNOWN for _ in observation.arguments),
        )
        supported = all(state is not planner.UNKNOWN for state in outputs)
        return SimpleNamespace(
            predicted_root_outputs=outputs,
            supported=supported,
            unsupported_reason=None if supported else "unsupported fixture branch",
            races=(),
        )


def _fixture_observation() -> _Observation:
    return _Observation(
        (
            _Site((), (2, 3)),
            _Site((0,), (2, 3)),
            _Site((1,), None),
        ),
        (2, 3),
    )


def test_planner_does_not_construct_or_execute_candidates(monkeypatch) -> None:
    observation = _fixture_observation()
    model = _FakeModel(
        {((0,), "WRAP_ADD[1]"): (3, 4)},
        (planner.UNKNOWN, planner.UNKNOWN),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("candidate planning reached the selected actuator")

    monkeypatch.setattr(planner.world, "apply_selected_wrap", forbidden)
    monkeypatch.setattr(planner.world, "observe_current_program", forbidden)
    decision = planner.plan_site_action_h1(model, observation, (3, 4))

    assert decision.status == "SELECTED"
    source = Path(planner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    inference_names = {
        "_candidate_bank",
        "_predict_candidates",
        "_prediction_features",
        "plan_site_action_h1",
    }
    inference_source = "\n".join(
        ast.get_source_segment(source, node) or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in inference_names
    )
    for forbidden_name in (
        "replace_subtree",
        "apply_selected_wrap",
        "observe_current_program",
        "tp.compile",
        "exec(",
        "argmin",
        "argmax",
        "argsort",
    ):
        assert forbidden_name not in inference_source


def test_unknown_branches_remain_in_bank_and_physical_factors_match() -> None:
    observation = _fixture_observation()
    model = _FakeModel(
        {((0,), "WRAP_ADD[1]"): (3, 4)},
        (planner.UNKNOWN, planner.UNKNOWN),
    )

    decision = planner.plan_site_action_h1(model, observation, (3, 4))

    assert len(decision.candidates) == len(observation.sites) * 9 + 1
    assert any(
        prediction.candidate.path == (1,) and not prediction.supported
        for prediction in decision.candidates
    )
    assert decision.selected_path == (0,)
    assert decision.selected_action == "WRAP_ADD[1]"
    assert decision.candidate_race.winner == "0::WRAP_ADD[1]"
    assert decision.site_motor_race is not None
    assert decision.action_motor_race is not None
    assert decision.site_motor_race.winner == "0"
    assert decision.action_motor_race.winner == "WRAP_ADD[1]"
    assert decision.factor_projection_sha256 is not None
    assert decision.world_checkpoint_before_sha256 == (
        decision.world_checkpoint_after_sha256
    )


def test_all_unknown_has_one_physical_stop_and_no_motor_fallback() -> None:
    observation = _fixture_observation()
    unknown = (planner.UNKNOWN, planner.UNKNOWN)
    model = _FakeModel({}, unknown)

    decision = planner.plan_site_action_h1(model, observation, (3, 4))

    assert decision.status == planner.STOP_ALL_UNKNOWN
    assert decision.candidate_race.winner == planner.STOP_ALL_UNKNOWN
    assert decision.candidate_race.simultaneous_spike_candidates == 1
    assert decision.selected_path is None
    assert decision.selected_action is None
    assert decision.site_motor_race is None
    assert decision.action_motor_race is None


def test_visible_matrix_is_eight_valid_one_edit_tasks_across_site_depths() -> None:
    tasks = planner.VISIBLE_TASKS

    assert len(tasks) == 8
    assert all(len(task.desired_outputs) == len(planner.VISIBLE_ARGUMENTS) for task in tasks)
    assert {len(task.expected_path) for task in tasks} == {0, 1, 2}
    assert sum(task.expected_action == planner.DONE for task in tasks) == 1
    assert all(
        task.expected_action in planner.ACTION_LABELS
        for task in tasks
    )
    assert len({task.name for task in tasks}) == len(tasks)
    for task in tasks:
        incumbent = planner.visible_incumbent(task)
        expected = planner.tp.Program(
            task.expected_tokens,
            planner.visible_scalar_signature(),
        )
        planner.tp.compile(incumbent)
        planner.tp.compile(expected)

    source = inspect.getsource(planner.plan_site_action_h1)
    assert "VISIBLE_TASKS" not in source
    assert "expected_path" not in source
    assert "expected_action" not in source


def test_single_public_runner_arm_smoke_does_not_write_artifact() -> None:
    task = planner.VISIBLE_TASKS[0]

    result = planner.run_visible_matrix((task,))

    assert result.schema == "public-typed-ast-site-visible-v1"
    assert result.verdict == "PASS_AST_SITE_VISIBLE"
    assert result.passed_tasks == result.total_tasks == 1
    assert result.selected_real_executions == 1
    assert result.target_free_observations == 410
    arm = result.tasks[0]
    assert arm.passed
    assert arm.selected_path == task.expected_path == ()
    assert arm.selected_action == task.expected_action == "WRAP_ABS"
    assert arm.actual_tokens == task.expected_tokens
    assert arm.actual_outputs == task.desired_outputs
    assert arm.physical_unique
    assert arm.checkpoint_immutable
    assert arm.candidate_count == 10
    assert arm.successor_race_count == 60
    assert arm.selection_race_count == 3


def test_visible_pass_artifact_contract() -> None:
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "r3_pilot"
        / "public_typed_ast_site_visible_probe.json"
    )
    raw = artifact_path.read_bytes()
    artifact = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == (
        "6fd138c5da23122cbd0cdf4af286562f69eb1d20b8f9caa162f32b344e0cc69c"
    )
    assert artifact["schema"] == "public-typed-ast-site-visible-v1"
    assert artifact["verdict"] == "PASS_AST_SITE_VISIBLE"
    assert artifact["passed_tasks"] == artifact["total_tasks"] == 8
    assert artifact["candidate_count"] == 179
    assert artifact["successor_race_count"] == 1_776
    assert artifact["selection_race_count"] == 24
    assert artifact["total_physical_races"] == 1_800
    assert artifact["target_free_observations"] == 410
    assert artifact["selected_real_executions"] == 8

    expected_ledger = (
        ("root-absolute", (), "WRAP_ABS", 10, 60),
        ("child-add-one-under-absolute", (0,), "WRAP_ADD[1]", 19, 168),
        ("child-absolute-under-add-two", (0,), "WRAP_ABS", 28, 276),
        ("child-subtract-one-under-absolute", (0,), "WRAP_SUB[1]", 19, 168),
        ("root-absolute-around-add-one", (), "WRAP_ABS", 28, 276),
        ("root-absolute-around-subtract-two", (), "WRAP_ABS", 28, 276),
        ("deep-child-absolute", (0, 0), "WRAP_ABS", 37, 492),
        ("global-done", (), "DONE", 10, 60),
    )
    actual_ledger = tuple(
        (
            task["name"],
            tuple(task["selected_path"]),
            task["selected_action"],
            task["candidate_count"],
            task["successor_race_count"],
        )
        for task in artifact["tasks"]
    )
    assert actual_ledger == expected_ledger
    assert all(
        task["passed"]
        and task["physical_unique"]
        and task["checkpoint_immutable"]
        and task["selected_path"] == task["expected_path"]
        and task["selected_action"] == task["expected_action"]
        and task["actual_tokens"] == task["expected_tokens"]
        and task["actual_outputs"] == task["desired_outputs"]
        for task in artifact["tasks"]
    )
