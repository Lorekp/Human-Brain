import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from experiments.r3_pilot import public_typed_ast_horizon2_int_planner as planner


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


def _apply(values, action):
    if any(value is planner.UNKNOWN for value in values):
        return tuple(planner.UNKNOWN for _ in values)
    if action == planner.DONE:
        return tuple(values)
    if action == "WRAP_ABS":
        return tuple(abs(value) for value in values)
    kind, amount_text = action.removeprefix("WRAP_").split("[")
    amount = int(amount_text.removesuffix("]"))
    if kind == "ADD":
        return tuple(value + amount for value in values)
    if kind == "SUB":
        return tuple(value - amount for value in values)
    raise AssertionError(f"unknown fixture action {action}")


class _FakeSuccessor:
    def __init__(self, done: tuple[planner.SuccessorState, ...]) -> None:
        self.done = done

    def predict_many(self, states, action):
        values = tuple(states)
        if action == planner.DONE and all(
            value is planner.UNKNOWN for value in values
        ):
            return self.done, ()
        return _apply(values, action), ()


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
        supported = all(value is not planner.UNKNOWN for value in outputs)
        return SimpleNamespace(
            predicted_root_outputs=outputs,
            supported=supported,
            unsupported_reason=None if supported else "unsupported fixture branch",
            races=(),
        )


def _fixture_observation() -> _Observation:
    return _Observation(
        (
            _Site((), (-2, 3)),
            _Site((0,), (-2, 3)),
            _Site((1,), None),
        ),
        (-2, 3),
    )


def test_first_selection_uses_real_second_ply_without_candidate_execution(
    monkeypatch,
) -> None:
    observation = _fixture_observation()
    model = _FakeModel(
        {((0,), "WRAP_ABS"): (2, 3)},
        (planner.UNKNOWN, planner.UNKNOWN),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("H=2 inference crossed the selected actuator")

    monkeypatch.setattr(planner.world, "apply_selected_wrap", forbidden)
    monkeypatch.setattr(planner.world, "observe_current_program", forbidden)
    decision = planner.plan_site_action_h2(model, observation, (3, 4))

    assert decision.status == "SELECTED"
    assert decision.selected_first_path == (0,)
    assert decision.selected_first_action == "WRAP_ABS"
    assert decision.selected_continuation_action == "WRAP_ADD[1]"
    assert decision.predicted_final_outputs == (3, 4)

    source = Path(planner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    inference_names = {
        "_predict_sequences",
        "_sequence_features",
        "plan_site_action_h2",
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
        "tp.Program",
        "tp.compile",
        "exec(",
        "argmin",
        "argmax",
        "argsort",
        "sorted(",
    ):
        assert forbidden_name not in inference_source


def test_all_incumbent_sites_and_unknown_branches_stay_in_physical_bank() -> None:
    observation = _fixture_observation()
    model = _FakeModel(
        {((0,), "WRAP_ABS"): (2, 3)},
        (planner.UNKNOWN, planner.UNKNOWN),
    )

    decision = planner.plan_site_action_h2(model, observation, (3, 4))

    # 3 sites * 9 nonterminal first actions * 10 continuations + DONE=>DONE.
    assert len(decision.predictions) == 271
    assert any(
        prediction.sequence.first.path == (1,) and not prediction.supported
        for prediction in decision.predictions
    )
    assert decision.sequence_race.winner == (
        "0::WRAP_ABS=>root::WRAP_ADD[1]"
    )
    assert decision.first_site_motor_race is not None
    assert decision.first_action_motor_race is not None
    assert decision.continuation_motor_race is not None
    assert decision.first_site_motor_race.winner == "0"
    assert decision.first_action_motor_race.winner == "WRAP_ABS"
    assert decision.continuation_motor_race.winner == "WRAP_ADD[1]"
    assert decision.factor_projection_sha256 is not None


def test_done_is_terminal_and_has_no_deferred_edit() -> None:
    observation = _fixture_observation()
    model = _FakeModel({}, (-2, 3))

    decision = planner.plan_site_action_h2(model, observation, (-2, 3))

    assert decision.status == "SELECTED"
    assert decision.selected_first_path == ()
    assert decision.selected_first_action == planner.DONE
    assert decision.selected_continuation_action == planner.DONE
    done_sequences = tuple(
        prediction.sequence
        for prediction in decision.predictions
        if prediction.sequence.first.action == planner.DONE
    )
    assert len(done_sequences) == 1
    assert done_sequences[0].continuation_action == planner.DONE


def test_public_matrix_passes_both_orders_and_real_site_change_replan() -> None:
    result = planner.run_public_matrix()

    assert result.schema == "public-typed-ast-horizon2-int-v1"
    assert result.verdict == "PASS_TYPED_AST_HORIZON2_INT"
    assert result.passed_tasks == result.total_tasks == 3
    assert result.target_free_observations == 410
    assert all(task.passed and task.checkpoint_immutable for task in result.tasks)

    by_name = {task.name: task for task in result.tasks}
    assert by_name["absolute-then-add"].actual_decisions == (
        ((), "WRAP_ABS"),
        ((), "WRAP_ADD[1]"),
        ((), planner.DONE),
    )
    assert by_name["add-then-absolute"].actual_decisions == (
        ((), "WRAP_ADD[1]"),
        ((), "WRAP_ABS"),
        ((), planner.DONE),
    )
    site_change = by_name["child-then-root-replan"]
    assert site_change.actual_decisions == (
        ((0,), "WRAP_ADD[1]"),
        ((), "WRAP_ADD[2]"),
        ((), planner.DONE),
    )
    assert site_change.decisions[0].continuation_action == "WRAP_ADD[2]"
    assert site_change.decisions[1].path == ()
    assert all(
        task.final_tokens == public_task.target_tokens
        and task.final_outputs == public_task.desired_outputs
        and all(decision.physical_unique for decision in task.decisions)
        for task, public_task in zip(result.tasks, planner.PUBLIC_TASKS, strict=True)
    )


def test_visible_horizon2_pass_artifact_contract() -> None:
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "r3_pilot"
        / "public_typed_ast_horizon2_int_visible_probe.json"
    )
    raw = artifact_path.read_bytes()
    artifact = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == (
        "c861a859540af57081af1d8aa1c4a6270e5e79965de1a631dca74eb62a2aff58"
    )
    assert artifact["schema"] == "public-typed-ast-horizon2-int-v1"
    assert artifact["verdict"] == "PASS_TYPED_AST_HORIZON2_INT"
    assert artifact["passed_tasks"] == artifact["total_tasks"] == 3
    assert artifact["target_free_observations"] == 410

    expected_ledger = (
        (
            "absolute-then-add",
            (((), "WRAP_ABS"), ((), "WRAP_ADD[1]"), ((), "DONE")),
            (91, 181, 361),
        ),
        (
            "add-then-absolute",
            (((), "WRAP_ADD[1]"), ((), "WRAP_ABS"), ((), "DONE")),
            (91, 271, 361),
        ),
        (
            "child-then-root-replan",
            (((0,), "WRAP_ADD[1]"), ((), "WRAP_ADD[2]"), ((), "DONE")),
            (181, 361, 541),
        ),
    )
    actual_ledger = tuple(
        (
            task["name"],
            tuple(
                (tuple(path), action)
                for path, action in task["actual_decisions"]
            ),
            tuple(decision["sequence_count"] for decision in task["decisions"]),
        )
        for task in artifact["tasks"]
    )
    assert actual_ledger == expected_ledger
    assert sum(sum(counts) for _, _, counts in actual_ledger) == 2_439
    assert all(
        task["passed"]
        and task["checkpoint_immutable"]
        and task["actual_decisions"] == task["expected_decisions"]
        and all(decision["physical_unique"] for decision in task["decisions"])
        for task in artifact["tasks"]
    )
