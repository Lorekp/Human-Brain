"""Two-demonstration occurrence motor for the contextual development session.

The caller executes and verifies teacher transitions before calling ``learn``.
Training alone inspects their full-state difference. Inference tries every
distinct request lexeme against the learned boundary-relative binder; it does
not label entities or select query words using task semantics. Host actuation
copies replacement bytes into the single physically selected occurrence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from humanbrain import knowledge_conditioned_occurrence_substitution as _motor
from humanbrain.multiscale_opaque_event_tape import OpaqueWorkspaceFile
from humanbrain.unified_opaque_observation_tape import UnifiedOpaqueObservationTape
from humanbrain.variability_aware_segment_binder import (
    VariabilityAwareSegmentBinder,
    VariabilityBindingDecision,
    VariabilitySegmentDemonstration,
)


@dataclass(frozen=True, slots=True)
class OccurrenceMotorResult:
    files: tuple[OpaqueWorkspaceFile, ...]
    edited: bool
    stop_reason: str | None
    checkpoint_sha256: str
    request_seed_count: int
    supported_role_count: int
    physical_cell_steps: int
    evaluated_cells: int
    binder_decisions: tuple[VariabilityBindingDecision, ...]
    selection_race: _motor.KnowledgeSubstitutionRaceTrace


def _observe(request, files):
    observation = UnifiedOpaqueObservationTape.observe(request, (), files)
    seeds = tuple(
        sorted(
            {
                _motor._pointer_event(_motor._canonical_lexeme_pointer(token.identity_event))
                for token in observation.request_token_occurrences
            }
        )
    )
    return observation, _motor._workspace_view(observation), seeds


class ContextualOccurrenceMotor:
    """One fixed-layout motor prior, learned from two externally verified edits."""

    def __init__(self) -> None:
        self._pending: tuple[VariabilitySegmentDemonstration, ...] = ()
        self._evidence: tuple[str, ...] = ()
        self._binder: VariabilityAwareSegmentBinder | None = None

    @property
    def checkpoint_sha256(self) -> str:
        return _motor._digest(
            b"humanbrain.contextual-occurrence-motor.v1\0",
            {
                "evidence": self._evidence,
                "binder": None if self._binder is None else self._binder.checkpoint_sha256,
            },
        )

    @property
    def demonstration_count(self) -> int:
        return len(self._evidence)

    @property
    def retained_demonstration_event_bytes(self) -> int:
        return sum(
            sum(map(len, (*row.query_events, *row.source_events, row.demonstrated_role)))
            for row in self._pending
        )

    @property
    def training_summary(self):
        return None if self._binder is None else self._binder.training_summary

    def learn(
        self,
        request: bytes,
        before: tuple[OpaqueWorkspaceFile, ...],
        after: tuple[OpaqueWorkspaceFile, ...],
    ) -> None:
        """Accept a caller-verified PASS; malformed transitions never update state."""
        if len(self._evidence) >= 2:
            raise ValueError("motor has a two-demonstration training budget")
        observation, view, seeds = _observe(request, before)
        after = _motor._canonical_files(after, "after")
        if tuple(item.path for item in view.files) != tuple(item.path for item in after):
            raise ValueError("demonstration must preserve all workspace paths")
        changed = [
            (old, new)
            for old, new in zip(view.files, after, strict=True)
            if old.content != new.content
        ]
        if len(changed) != 1:
            raise ValueError("demonstration must change exactly one file")
        old, new = changed[0]
        targets = []
        for token in view.occurrences:
            if token.path != old.path:
                continue
            prefix, suffix = old.content[: token.byte_start], old.content[token.byte_end :]
            end = len(new.content) - len(suffix)
            if (
                end <= len(prefix)
                or not new.content.startswith(prefix)
                or not new.content.endswith(suffix)
            ):
                continue
            replacement = new.content[len(prefix) : end]
            if re.fullmatch(rb"[A-Za-z0-9_]+", replacement) is not None:
                targets.append(token)
        if len(targets) != 1:
            raise ValueError("demonstration must replace one unique complete lexical occurrence")
        row = VariabilitySegmentDemonstration(seeds, view.events, targets[0].pointer_event)
        pending = (*self._pending, row)
        binder = VariabilityAwareSegmentBinder.grow(pending) if len(pending) == 2 else None
        evidence = _motor._digest(
            b"humanbrain.contextual-occurrence-motor.lesson.v1\0",
            {
                "parent": self.checkpoint_sha256,
                "observation": _motor._observation_sha256(observation),
                "after": _motor._action_sha256(after),
            },
        )
        self._pending = () if binder is not None else pending
        self._binder = binder
        self._evidence = (*self._evidence, evidence)

    def apply(
        self, request: bytes, files: tuple[OpaqueWorkspaceFile, ...], replacement: bytes
    ) -> OccurrenceMotorResult:
        if not isinstance(replacement, bytes) or not replacement:
            raise ValueError("replacement must be non-empty bytes")
        checkpoint = self.checkpoint_sha256
        _, view, seeds = _observe(request, files)
        decisions = (
            tuple(self._binder.respond((seed,), view.events) for seed in seeds)
            if self._binder is not None and view.events
            else ()
        )
        supported = {row.role_identity for row in decisions if row.decision == "BIND"}
        choice = _motor._physical_choice(
            population="contextual-motor/occurrence-selection",
            scores=tuple(
                _motor._DRIVE_SCORE if token.pointer_event in supported else 0
                for token in view.occurrences
            ),
        )
        output = view.files
        reason = choice.trace.stop_reason
        if self._binder is None:
            reason = "motor:unlearned"
        elif not view.events:
            reason = "motor:empty_workspace"
        if choice.column is not None:
            target = view.occurrences[choice.column]
            modified = list(view.files)
            old = modified[target.file_ordinal]
            modified[target.file_ordinal] = OpaqueWorkspaceFile(
                old.path,
                old.content[: target.byte_start] + replacement + old.content[target.byte_end :],
            )
            output = tuple(modified)
        if self.checkpoint_sha256 != checkpoint:
            raise RuntimeError("motor inference changed its learned state")
        return OccurrenceMotorResult(
            output,
            choice.column is not None,
            reason,
            checkpoint,
            len(seeds),
            len(supported),
            sum(row.physical_cell_steps for row in decisions) + choice.trace.physical_cell_steps,
            sum(race.cell_count for row in decisions for race in row.races)
            + len(choice.trace.candidate_cell_ids)
            + 1
            + int(choice.trace.collision_stop_spike_step is not None),
            decisions,
            choice.trace,
        )
