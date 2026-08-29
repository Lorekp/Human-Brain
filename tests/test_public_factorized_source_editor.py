from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, cast

import pytest
from experiments.r3_pilot import public_factorized_source_editor as source_editor
from experiments.r3_pilot import public_typed_factor_horizon2_int as h2
from humanbrain import typed_program as tp
from humanbrain.simple_swe import make_task
from humanbrain.simple_swe.protocol import ExecutionLimits
from humanbrain.simple_swe.sandbox import ExecutionResult
from humanbrain.typed_program_types import (
    ProgramParameter,
    ProgramType,
    TypedSignature,
)

_CHECKPOINT = "7" * 64
_ADD_ONE = tp.FactorAction(
    (), "APPLY2", "add", "SELECTED", "LITERAL", 1
)
_DONE = tp.FactorAction((), "DONE", "NONE", "NONE", "NONE", None)


def _signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "solve",
        (ProgramParameter("x", integer),),
        integer,
    )


@dataclass(frozen=True, slots=True)
class _Observation:
    program_sha256: str
    cases: tuple[tuple[object, ...], ...]


@dataclass(frozen=True, slots=True)
class _Decision:
    status: str
    observation: _Observation
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str
    selected_sequence: h2.FactorH2Sequence | None
    stop_reason: str | None = None


class _ScriptedPlanner:
    checkpoint_sha256 = _CHECKPOINT

    def __init__(self, *, done_after_edits: int | None, mutate_on_actuation: bool = False):
        self._live_checkpoint = _CHECKPOINT
        self._done_after_edits = done_after_edits
        self._mutate_on_actuation = mutate_on_actuation
        self.edit_count = 0
        self.plan_observations: list[_Observation] = []
        self.plan_desired: list[tuple[object, ...]] = []

    @property
    def live_checkpoint_sha256(self) -> str:
        return self._live_checkpoint

    def observe(
        self,
        program: tp.Program,
        cases: tuple[tuple[object, ...], ...],
    ) -> _Observation:
        return _Observation(program_sha256=program.sha256, cases=tuple(cases))

    def plan(
        self,
        observation: _Observation,
        _prompt: str,
        desired: tuple[object, ...],
    ) -> _Decision:
        self.plan_observations.append(observation)
        self.plan_desired.append(tuple(desired))
        status = h2.STATUS_SELECTED
        if (
            self._done_after_edits is not None
            and self.edit_count >= self._done_after_edits
        ):
            status = h2.STATUS_DONE
        return _Decision(
            status=status,
            observation=observation,
            checkpoint_before_sha256=self._live_checkpoint,
            checkpoint_after_sha256=self._live_checkpoint,
            selected_sequence=(
                h2.FactorH2Sequence(_ADD_ONE, _DONE)
                if status == h2.STATUS_SELECTED
                else None
            ),
        )

    def actuate_and_reobserve(
        self,
        incumbent: tp.Program,
        decision: _Decision,
    ) -> h2.FactorH2Actuation:
        edited = tp.apply_factor_action(incumbent, _ADD_ONE)
        self.edit_count += 1
        if self._mutate_on_actuation:
            self._live_checkpoint = "8" * 64
        return h2.FactorH2Actuation(
            program=edited,
            observation=cast(Any, self.observe(edited, decision.observation.cases)),
            applied_action=_ADD_ONE,
            actuator_calls=1,
            structure_parity=True,
            action_site_parity=True,
            root_output_parity=True,
        )


def _run_done() -> tuple[
    source_editor.FactorizedSourceEditResult,
    _ScriptedPlanner,
]:
    planner = _ScriptedPlanner(done_after_edits=1)
    editor = source_editor.FactorizedSourceEditor(cast(Any, planner))
    result = editor.edit(
        tp.Program(("variable-0",), _signature()),
        "Increase x by one.",
        (
            source_editor.VisibleCase((-2,), -1),
            source_editor.VisibleCase((4,), 5),
        ),
        max_edits=2,
    )
    return result, planner


def test_source_editor_actuates_one_edit_reobserves_and_finishes_done() -> None:
    result, planner = _run_done()

    assert result.status == source_editor.STATUS_DONE
    assert result.stop_reason is None
    assert result.edit_count == 1
    assert result.applied_actions == (_ADD_ONE,)
    assert len(result.decisions) == 2
    assert result.decisions[-1].status == h2.STATUS_DONE
    assert result.source == "def solve(x):\n    return x + 1\n"
    assert planner.plan_desired == [(-1, 5), (-1, 5)]
    assert (
        planner.plan_observations[0].program_sha256
        != planner.plan_observations[1].program_sha256
    )
    assert result.checkpoint_audit.immutable
    assert all(step.immutable for step in result.checkpoint_audit.steps)

    edit_source = inspect.getsource(source_editor.FactorizedSourceEditor.edit)
    assert "execute_completion(" not in edit_source
    assert "apply_factor_action(" not in edit_source
    assert "candidate_actions" not in edit_source


def test_source_editor_replans_at_the_bound_without_actuating_past_it() -> None:
    planner = _ScriptedPlanner(done_after_edits=None)
    result = source_editor.FactorizedSourceEditor(cast(Any, planner)).edit(
        tp.Program(("variable-0",), _signature()),
        "Increase x repeatedly.",
        (source_editor.VisibleCase((0,), 3),),
        max_edits=1,
    )

    assert result.status == source_editor.STATUS_MAX_EDITS
    assert result.edit_count == 1
    assert result.applied_actions == (_ADD_ONE,)
    assert len(result.decisions) == 2
    assert planner.edit_count == 1
    assert result.source == "def solve(x):\n    return x + 1\n"
    assert result.checkpoint_audit.steps[-1].live_after_actuation_sha256 is None
    assert result.checkpoint_audit.immutable


def test_source_editor_fails_closed_if_actuation_mutates_the_checkpoint() -> None:
    planner = _ScriptedPlanner(done_after_edits=1, mutate_on_actuation=True)

    with pytest.raises(RuntimeError, match="checkpoint changed during actuation"):
        source_editor.FactorizedSourceEditor(cast(Any, planner)).edit(
            tp.Program(("variable-0",), _signature()),
            "Increase x by one.",
            (source_editor.VisibleCase((0,), 1),),
            max_edits=1,
        )


def test_completed_source_uses_public_task_bundle_and_sandbox_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, planner = _run_done()
    task = make_task(
        partition="train",
        family="t0_source_editor_canary",
        ordinal=0,
        signature=_signature(),
        description="Return x plus one.",
        oracle=lambda x: x + 1,
        regression_inputs=((0,),),
        property_inputs=(((9,), "public-canary"),),
    )
    limits = ExecutionLimits(
        wall_seconds=1.0,
        cpu_seconds=1,
        memory_mib=64,
        processes=1,
        output_bytes=1_024,
        max_completion_bytes=2_048,
        max_ast_nodes=128,
        python_hash_seed=17,
    )
    captured: dict[str, object] = {}

    def fake_execute(
        source: str,
        public: object,
        private: object,
        supplied_limits: object,
        **kwargs: object,
    ) -> ExecutionResult:
        captured.update(
            source=source,
            public=public,
            private=private,
            limits=supplied_limits,
            kwargs=kwargs,
        )
        return ExecutionResult("passed", True, 0.01, "ok", True)

    monkeypatch.setattr(source_editor, "execute_completion", fake_execute)
    execution = source_editor.execute_completed_source(
        result,
        task,
        limits,
        python_executable_sha256="1" * 64,
        worker_source=b"worker",
        worker_sha256="2" * 64,
    )

    assert execution.passed
    assert captured["source"] == result.source
    assert captured["public"] is task.public
    assert captured["private"] is task.private
    assert captured["limits"] is limits
    assert planner.plan_observations[0].cases == ((-2,), (4,))
    assert tuple(case.args for case in task.private.regression_cases) == ((0,),)
