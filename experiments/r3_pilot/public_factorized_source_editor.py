"""Bounded source-editing loop over the reusable factorized H=2 planner.

The adapter exposes only visible input/output cases to the planner.  It never
constructs, compiles, executes, or ranks hypothetical programs: the selected
first action is actuated through the planner, the real incumbent is observed
again, and planning restarts.  Only the terminal real ``Program`` is compiled.

This is pre-M0 mechanism code.  It does not load a Simple-SWE partition or
open hidden cases; a completed result can be handed to the existing sandbox
through :func:`execute_completed_source`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from experiments.r3_pilot import public_typed_factor_horizon2_int as h2
from humanbrain import typed_program as tp
from humanbrain.simple_swe.protocol import ExecutionLimits
from humanbrain.simple_swe.sandbox import ExecutionResult, execute_completion
from humanbrain.simple_swe.tasks import GeneratedTask


STATUS_DONE = "DONE"
STATUS_MAX_EDITS = "MAX_EDITS"
STATUS_PLANNER_STOP = "PLANNER_STOP"


@dataclass(frozen=True, slots=True)
class VisibleCase:
    """One case deliberately exposed to the editor's physical value model."""

    arguments: tuple[object, ...]
    expected: object

    def __post_init__(self) -> None:
        if not isinstance(self.arguments, tuple):
            raise TypeError("visible-case arguments must be a tuple")


@dataclass(frozen=True, slots=True)
class SourceEditorCheckpointStep:
    decision_index: int
    live_before_sha256: str
    decision_before_sha256: str
    decision_after_sha256: str
    live_after_plan_sha256: str
    live_after_actuation_sha256: str | None

    @property
    def immutable(self) -> bool:
        values = (
            self.live_before_sha256,
            self.decision_before_sha256,
            self.decision_after_sha256,
            self.live_after_plan_sha256,
        )
        if self.live_after_actuation_sha256 is not None:
            values = (*values, self.live_after_actuation_sha256)
        return len(set(values)) == 1


@dataclass(frozen=True, slots=True)
class SourceEditorCheckpointAudit:
    initial_sha256: str
    steps: tuple[SourceEditorCheckpointStep, ...]
    final_sha256: str

    @property
    def immutable(self) -> bool:
        return (
            self.final_sha256 == self.initial_sha256
            and all(
                step.immutable
                and step.live_before_sha256 == self.initial_sha256
                for step in self.steps
            )
        )


@dataclass(frozen=True, slots=True)
class FactorizedSourceEditResult:
    status: str
    stop_reason: str | None
    program: tp.Program
    compiled: tp.CompiledPython
    decisions: tuple[h2.FactorizedH2Decision, ...]
    actuations: tuple[h2.FactorH2Actuation, ...]
    checkpoint_audit: SourceEditorCheckpointAudit

    @property
    def source(self) -> str:
        return self.compiled.source

    @property
    def edit_count(self) -> int:
        return len(self.actuations)


@dataclass(frozen=True, slots=True)
class FactorizedSourceEditor:
    """Drive one frozen planner until ``DONE``, a stop, or ``max_edits``."""

    planner: h2.FrozenFactorizedHorizon2Int

    def edit(
        self,
        incumbent: tp.Program,
        prompt: str,
        visible_cases: Sequence[VisibleCase],
        *,
        max_edits: int,
    ) -> FactorizedSourceEditResult:
        if not isinstance(incumbent, tp.Program):
            raise TypeError("source editor requires a typed incumbent Program")
        if type(prompt) is not str or not prompt:
            raise ValueError("source editor requires a nonempty prompt")
        if isinstance(max_edits, bool) or not isinstance(max_edits, int):
            raise TypeError("max_edits must be an integer")
        if max_edits < 0:
            raise ValueError("max_edits must be nonnegative")
        cases = tuple(visible_cases)
        if not cases:
            raise ValueError("source editor requires at least one visible case")
        if not all(isinstance(case, VisibleCase) for case in cases):
            raise TypeError("visible_cases must contain VisibleCase values")

        arguments = tuple(case.arguments for case in cases)
        desired = tuple(case.expected for case in cases)
        initial_checkpoint = self.planner.live_checkpoint_sha256
        if initial_checkpoint != self.planner.checkpoint_sha256:
            raise RuntimeError("factorized source-editor checkpoint is not frozen")

        program = incumbent
        observation = self.planner.observe(program, arguments)
        decisions: list[h2.FactorizedH2Decision] = []
        actuations: list[h2.FactorH2Actuation] = []
        checkpoint_steps: list[SourceEditorCheckpointStep] = []
        result_status: str
        stop_reason: str | None

        while True:
            live_before = self.planner.live_checkpoint_sha256
            if live_before != initial_checkpoint:
                raise RuntimeError("factorized source-editor checkpoint changed before planning")
            decision = self.planner.plan(observation, prompt, desired)
            live_after_plan = self.planner.live_checkpoint_sha256
            if (
                decision.checkpoint_before_sha256 != initial_checkpoint
                or decision.checkpoint_after_sha256 != initial_checkpoint
                or live_after_plan != initial_checkpoint
            ):
                raise RuntimeError("factorized source-editor checkpoint changed during planning")
            if decision.observation.program_sha256 != program.sha256:
                raise RuntimeError("planner decision does not describe the live incumbent")
            decisions.append(decision)

            live_after_actuation: str | None = None
            if decision.status == h2.STATUS_DONE:
                result_status = STATUS_DONE
                stop_reason = None
            elif decision.status != h2.STATUS_SELECTED:
                result_status = STATUS_PLANNER_STOP
                stop_reason = decision.stop_reason or decision.status
            elif len(actuations) >= max_edits:
                result_status = STATUS_MAX_EDITS
                stop_reason = f"max_edits={max_edits} reached before another actuation"
            else:
                actuation = self.planner.actuate_and_reobserve(program, decision)
                live_after_actuation = self.planner.live_checkpoint_sha256
                if live_after_actuation != initial_checkpoint:
                    raise RuntimeError(
                        "factorized source-editor checkpoint changed during actuation"
                    )
                if actuation.actuator_calls != 1:
                    raise RuntimeError("one selected decision must actuate exactly one edit")
                if actuation.program.sha256 == program.sha256:
                    raise RuntimeError("selected factor action did not change the incumbent")
                if actuation.observation.program_sha256 != actuation.program.sha256:
                    raise RuntimeError("actuation reobservation does not describe its program")
                actuations.append(actuation)
                program = actuation.program
                observation = actuation.observation

            step = SourceEditorCheckpointStep(
                decision_index=len(decisions) - 1,
                live_before_sha256=live_before,
                decision_before_sha256=decision.checkpoint_before_sha256,
                decision_after_sha256=decision.checkpoint_after_sha256,
                live_after_plan_sha256=live_after_plan,
                live_after_actuation_sha256=live_after_actuation,
            )
            checkpoint_steps.append(step)
            if not step.immutable:
                raise RuntimeError("factorized source-editor checkpoint audit is mutable")
            if live_after_actuation is not None:
                continue
            break

        final_checkpoint = self.planner.live_checkpoint_sha256
        audit = SourceEditorCheckpointAudit(
            initial_sha256=initial_checkpoint,
            steps=tuple(checkpoint_steps),
            final_sha256=final_checkpoint,
        )
        if not audit.immutable:
            raise RuntimeError("factorized source-editor final checkpoint changed")
        return FactorizedSourceEditResult(
            status=result_status,
            stop_reason=stop_reason,
            program=program,
            compiled=tp.compile(program),
            decisions=tuple(decisions),
            actuations=tuple(actuations),
            checkpoint_audit=audit,
        )


def execute_completed_source(
    result: FactorizedSourceEditResult,
    task: GeneratedTask,
    limits: ExecutionLimits,
    *,
    python_executable: str | Path | None = None,
    python_executable_sha256: str,
    worker_source: bytes,
    worker_sha256: str,
) -> ExecutionResult:
    """Score only a terminal real program through the public sandbox seam."""

    if not isinstance(result, FactorizedSourceEditResult):
        raise TypeError("sandbox integration requires a source-editor result")
    if result.status != STATUS_DONE:
        raise ValueError("only a terminal DONE source-editor result may be scored")
    if not isinstance(task, GeneratedTask):
        raise TypeError("sandbox integration requires a GeneratedTask")
    if result.program.signature != task.public.signature:
        raise ValueError("edited program signature differs from the Simple-SWE task")
    return execute_completion(
        result.source,
        task.public,
        task.private,
        limits,
        python_executable=python_executable,
        python_executable_sha256=python_executable_sha256,
        worker_source=worker_source,
        worker_sha256=worker_sha256,
    )


__all__ = [
    "STATUS_DONE",
    "STATUS_MAX_EDITS",
    "STATUS_PLANNER_STOP",
    "FactorizedSourceEditResult",
    "FactorizedSourceEditor",
    "SourceEditorCheckpointAudit",
    "SourceEditorCheckpointStep",
    "VisibleCase",
    "execute_completed_source",
]
