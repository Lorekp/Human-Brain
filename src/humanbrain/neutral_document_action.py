"""Persistent neutral document acquisition and a learned literal-edit motor.

This development adapter reuses the frozen bridge's opaque workspace projection,
physical races, and authenticated transition receipts without changing its
behavior. Two executed demonstrations teach one boundary-relative occurrence
walk. Literal substitution is declared host actuation; it is not learned syntax
generation or autonomous recovery. One derived event stream is retained until
the second demonstration promotes the motor, then discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256

from humanbrain import knowledge_conditioned_occurrence_substitution as _motor
from humanbrain.multiscale_opaque_event_tape import OpaqueWorkspaceFile
from humanbrain.neutral_neuronal_document_learner import (
    FrozenNeutralNeuronalDocumentLearner,
    NeutralFrameAblation,
    NeutralFrameDecision,
)
from humanbrain.neutral_neuronal_document_learner import (
    start as start_documents,
)
from humanbrain.opaque_document_event_tape import OpaqueDocument
from humanbrain.unified_opaque_observation_tape import UnifiedOpaqueObservationTape
from humanbrain.variability_aware_segment_binder import (
    VariabilityAwareSegmentBinder,
    VariabilityBinderAblation,
    VariabilityBindingDecision,
    VariabilitySegmentDemonstration,
)


class DocumentActionAblation(StrEnum):
    NONE = "none"
    CORE_OFF = "core_off"
    KNOWLEDGE_OFF = "knowledge_off"
    ACTION_OFF = "action_off"
    PROPAGATION_OFF = "propagation_off"


@dataclass(frozen=True, slots=True)
class PhysicalActionDecision:
    status: str
    emitted: bytes | None
    after_workspace: tuple[OpaqueWorkspaceFile, ...] | None
    stop_reason: str | None
    producer_decision: NeutralFrameDecision
    binder_decision: VariabilityBindingDecision | None
    checkpoint_sha256: str
    observation_sha256: str
    physical_cell_steps: int
    motor_races: tuple[_motor.KnowledgeSubstitutionRaceTrace, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class _ActionEvidence:
    transition_sha256: str
    provenance_sha256: str
    observation_sha256: str
    path_sha256: str
    replacement_sha256: str


@dataclass(frozen=True, slots=True)
class NeutralDocumentActionLearner:
    document_learner: FrozenNeutralNeuronalDocumentLearner
    binder: VariabilityAwareSegmentBinder | None
    parent_checkpoint_sha256: str | None
    _pending: tuple[VariabilitySegmentDemonstration, ...] = ()
    _evidence: tuple[_ActionEvidence, ...] = ()

    @property
    def checkpoint_sha256(self) -> str:
        self.document_learner.authenticate()
        return _motor._digest(
            b"humanbrain.neutral-document-action.checkpoint.v1\0",
            {
                "documents": self.document_learner.checkpoint_sha256,
                "binder": None if self.binder is None else self.binder.checkpoint_sha256,
                "parent": self.parent_checkpoint_sha256,
                "evidence": [row.transition_sha256 for row in self._evidence],
                "pending": [
                    {
                        "query": [event.hex() for event in row.query_events],
                        "source": [event.hex() for event in row.source_events],
                        "role": row.demonstrated_role.hex(),
                    }
                    for row in self._pending
                ],
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

    def consolidate(
        self,
        documents: tuple[OpaqueDocument, ...],
        *,
        ablation: NeutralFrameAblation = NeutralFrameAblation.NONE,
    ) -> NeutralDocumentActionLearner:
        return replace(
            self,
            document_learner=self.document_learner.consolidate(documents, ablation=ablation),
            parent_checkpoint_sha256=self.checkpoint_sha256,
        )

    def respond(
        self,
        request: bytes,
        documents: tuple[OpaqueDocument, ...] = (),
        files: tuple[OpaqueWorkspaceFile, ...] = (),
        *,
        ablation: DocumentActionAblation = DocumentActionAblation.NONE,
    ) -> PhysicalActionDecision:
        if not isinstance(ablation, DocumentActionAblation):
            raise TypeError("ablation must be DocumentActionAblation")
        observation = UnifiedOpaqueObservationTape.observe(request, documents, files)
        checkpoint = self.checkpoint_sha256
        memory_ablation = NeutralFrameAblation.NONE
        if ablation is DocumentActionAblation.CORE_OFF:
            memory_ablation = NeutralFrameAblation.MOTOR_OFF
        elif ablation is DocumentActionAblation.KNOWLEDGE_OFF:
            memory_ablation = NeutralFrameAblation.FACT_RETRIEVAL_OFF
        producer = self.document_learner.respond(request, documents, ablation=memory_ablation)
        races: list[_motor.KnowledgeSubstitutionRaceTrace] = []
        binding: VariabilityBindingDecision | None = None

        def finish(status: str, after=None, reason=None) -> PhysicalActionDecision:
            if self.checkpoint_sha256 != checkpoint:
                raise RuntimeError("inference mutated the persistent learner")
            steps = (
                producer.inference_physical_cell_steps
                + sum(race.physical_cell_steps for race in races)
                + (0 if binding is None else binding.physical_cell_steps)
            )
            observation_root = _motor._observation_sha256(observation)
            record = {
                "checkpoint": checkpoint,
                "observation": observation_root,
                "producer": producer.sha256,
                "binder": None if binding is None else binding.sha256,
                "races": [race.sha256 for race in races],
                "status": status,
                "reason": reason,
                "after": None if after is None else _motor._action_sha256(after),
            }
            return PhysicalActionDecision(
                status,
                producer.emitted,
                after,
                reason,
                producer,
                binding,
                checkpoint,
                observation_root,
                steps,
                tuple(races),
                _motor._digest(b"humanbrain.neutral-document-action.decision.v1\0", record),
            )

        if producer.stopped:
            return finish("STOP", reason=producer.stop_reason)
        if not files:
            return finish("EMIT")
        if self.binder is None:
            return finish("STOP", reason="action:unlearned")
        anchors = tuple(sorted(set(producer.support_query_lexemes)))
        anchor = _motor._physical_choice(
            population="neutral-action/query_anchor",
            scores=tuple(_motor._DRIVE_SCORE for _ in anchors),
        )
        races.append(anchor.trace)
        if anchor.column is None:
            return finish("STOP", reason="action:ambiguous_anchor")
        view = _motor._workspace_view(observation)
        if not view.events:
            return finish("STOP", reason="action:no_workspace_occurrences")
        binding = self.binder.respond(
            (_motor._pointer_event(_motor._canonical_lexeme_pointer(anchors[anchor.column])),),
            view.events,
            ablation=(
                VariabilityBinderAblation.PROPAGATION_OFF
                if ablation is DocumentActionAblation.PROPAGATION_OFF
                else VariabilityBinderAblation.NONE
            ),
        )
        if binding.decision != "BIND":
            return finish("STOP", reason=f"action:{binding.stop_reason}")
        occurrence = _motor._physical_choice(
            population="neutral-action/occurrence",
            scores=tuple(
                _motor._DRIVE_SCORE if item.pointer_event == binding.role_identity else 0
                for item in view.occurrences
            ),
        )
        races.append(occurrence.trace)
        if occurrence.column is None:
            return finish("STOP", reason="action:ambiguous_occurrence")
        action = _motor._physical_choice(
            population="neutral-action/substitute",
            scores=(0 if ablation is DocumentActionAblation.ACTION_OFF else _motor._DRIVE_SCORE,),
        )
        races.append(action.trace)
        if action.column is None:
            return finish("STOP", reason="action:disabled")
        target = view.occurrences[occurrence.column]
        after = list(view.files)
        old = after[target.file_ordinal]
        after[target.file_ordinal] = OpaqueWorkspaceFile(
            old.path,
            old.content[: target.byte_start] + producer.emitted + old.content[target.byte_end :],
        )
        return finish("EDIT", tuple(after))

    def learn_action(
        self,
        success: _motor.KnowledgeSubstitutionSuccess,
    ) -> NeutralDocumentActionLearner:
        """Learn only after an authority has executed and accepted this transition."""
        if not isinstance(success, _motor.KnowledgeSubstitutionSuccess):
            raise TypeError("success must be KnowledgeSubstitutionSuccess")
        success.authenticate()
        if success.execution_receipt.producer_checkpoint_sha256 != self.checkpoint_sha256:
            raise ValueError("action receipt belongs to another parent checkpoint")
        if len(self._evidence) >= 2:
            raise ValueError("this development motor has a two-demonstration budget")
        observation = success.observation
        producer = self.document_learner.respond(
            observation.reconstruct_request(),
            observation.reconstruct_documents(),
        )
        anchors = tuple(sorted(set(producer.support_query_lexemes)))
        if producer.stopped or len(anchors) != 1:
            raise ValueError("demonstration lacks unique learned knowledge and anchor")
        before = observation.reconstruct_workspace()
        after = success.after_workspace
        if tuple(item.path for item in before) != tuple(item.path for item in after):
            raise ValueError("demonstration must preserve all workspace paths")
        changed = [
            (old, new) for old, new in zip(before, after, strict=True) if old.content != new.content
        ]
        if len(changed) != 1:
            raise ValueError("demonstration must change one file")
        old, new = changed[0]
        view = _motor._workspace_view(observation)
        targets = [
            item
            for item in view.occurrences
            if item.path == old.path
            and old.content[: item.byte_start] + producer.emitted + old.content[item.byte_end :]
            == new.content
        ]
        if len(targets) != 1:
            raise ValueError("demonstration must be one unique knowledge-grounded substitution")
        evidence = _ActionEvidence(
            success.sha256,
            sha256(success.provenance_id).hexdigest(),
            _motor._observation_sha256(observation),
            sha256(old.path.encode()).hexdigest(),
            sha256(producer.emitted).hexdigest(),
        )
        for prior in self._evidence:
            if any(
                getattr(prior, field) == getattr(evidence, field)
                for field in _ActionEvidence.__dataclass_fields__
            ):
                raise ValueError(
                    "demonstrations need independent observations, provenance, paths and values"
                )
        demonstration = VariabilitySegmentDemonstration(
            (_motor._pointer_event(_motor._canonical_lexeme_pointer(anchors[0])),),
            view.events,
            targets[0].pointer_event,
        )
        pending = (*self._pending, demonstration)
        binder = VariabilityAwareSegmentBinder.grow(pending) if len(pending) == 2 else None
        return replace(
            self,
            binder=binder,
            parent_checkpoint_sha256=self.checkpoint_sha256,
            _pending=() if binder is not None else pending,
            _evidence=(*self._evidence, evidence),
        )


def start() -> NeutralDocumentActionLearner:
    return NeutralDocumentActionLearner(start_documents(), None, None)
