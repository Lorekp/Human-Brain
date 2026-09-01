"""Typed consequence revision over the frozen continual token-edit schema bank.

The adapter leaves the spent V1 schema learner unchanged.  It projects one
source into a candidate-only view without running V1's race or materializing an
edit.  A domain-neutral continual action-outcome memory owns the sole final
physical action decision.  Only that selected schema is materialized.  An
authenticated ``TokenEditConsequenceV1`` is exact typed caller attestation; this
module does not establish trusted execution or public-cohort evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from experiments.r3_pilot import common_continual_action_outcome_memory_v1 as outcomes
from experiments.r3_pilot import common_continual_token_edit_schema_v1 as token_edits
from experiments.r3_pilot.common_continual_token_edit_schema_v1 import (
    _apply_schema as _apply_schema_v1,
)
from experiments.r3_pilot.common_continual_token_edit_schema_v1 import (
    _nearest_filler as _nearest_filler_v1,
)

SCHEMA = "COMMON_CONTINUAL_TOKEN_EDIT_CONSEQUENCE_POLICY_V1"
CLAIM_BOUNDARY = (
    "typed-caller-attested-consequence-revision-of-one-fixed-parameterized-"
    "token-edit-schema-bank-with-query-local-materialization-not-trusted-"
    "execution-public-cohort-evidence-schema-discovery-semantics-diagnosis-"
    "generic-repair-swe-or-intelligence"
)

CANDIDATES_READY = "CANDIDATES_READY"
READY = outcomes.READY
EXPLORATION_READY = outcomes.EXPLORATION_READY
STOP_CELL_OFF = outcomes.STOP_CELL_OFF
STOP_NO_CANDIDATE = outcomes.STOP_NO_CANDIDATE
STOP_PHYSICAL_RACE = outcomes.STOP_PHYSICAL_RACE
STOP_INSUFFICIENT_OUTCOME_EVIDENCE = outcomes.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
STOP_ARTIFACT_REPLAY = token_edits.STOP_ARTIFACT_REPLAY
STOP_NO_SCHEMA_MATCH = token_edits.STOP_NO_SCHEMA_MATCH
STOP_SCHEMA_OFF = token_edits.STOP_SCHEMA_OFF
STOP_SEEN_SOURCE = token_edits.STOP_SEEN_SOURCE

_READY_STATUSES = frozenset((READY, EXPLORATION_READY))
_CANDIDATE_STATUSES = frozenset(
    (
        CANDIDATES_READY,
        STOP_CELL_OFF,
        STOP_NO_SCHEMA_MATCH,
        STOP_SCHEMA_OFF,
        STOP_SEEN_SOURCE,
    )
)
_POLICY_STOP_STATUSES = frozenset(
    (
        STOP_ARTIFACT_REPLAY,
        STOP_CELL_OFF,
        STOP_INSUFFICIENT_OUTCOME_EVIDENCE,
        STOP_NO_CANDIDATE,
        STOP_NO_SCHEMA_MATCH,
        STOP_PHYSICAL_RACE,
        STOP_SCHEMA_OFF,
        STOP_SEEN_SOURCE,
    )
)
_CONSTRUCTION_TOKEN = object()
_CHECKPOINT_NAMESPACE = b"common-continual-token-edit-consequence-policy-checkpoint-v1"
_SELECTION_CONTEXT_NAMESPACE = (
    b"common-continual-token-edit-consequence-policy-selection-context-v1"
)


class ContinualTokenEditConsequencePolicyError(ValueError):
    """A policy checkpoint, projection, proposal, or consequence failed authentication."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class TokenEditSchemaActionV1:
    """Unambiguous mapping between one V1 schema index and one generic action root."""

    schema_index: int
    action_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema_index": self.schema_index,
            "action_sha256": self.action_sha256,
        }

    def authenticate(self) -> None:
        if (
            type(self.schema_index) is not int
            or not 0 <= self.schema_index < len(token_edits.PARAMETERIZED_SCHEMAS)
            or self.action_sha256 != token_edits.PARAMETERIZED_SCHEMAS[self.schema_index].sha256
        ):
            raise ContinualTokenEditConsequencePolicyError(
                "token-edit schema action mapping failed authentication"
            )


def _catalogue() -> tuple[TokenEditSchemaActionV1, ...]:
    return tuple(
        sorted(
            (
                TokenEditSchemaActionV1(
                    schema_index=schema.schema_index,
                    action_sha256=schema.sha256,
                )
                for schema in token_edits.PARAMETERIZED_SCHEMAS
            ),
            key=lambda item: item.action_sha256,
        )
    )


def _checkpoint_sha256(
    *,
    base_checkpoint_sha256: str,
    outcome_checkpoint_sha256: str,
) -> str:
    return _digest(
        _CHECKPOINT_NAMESPACE,
        {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "outcome_checkpoint_sha256": outcome_checkpoint_sha256,
            "catalogue": tuple(item.record() for item in _catalogue()),
        },
    )


class FrozenContinualTokenEditConsequencePolicyV1:
    """One immutable V1 schema bank plus one immutable generic outcome memory."""

    __slots__ = ("_base_checkpoint", "_checkpoint_sha256", "_outcome_memory")

    def __init__(
        self,
        *,
        base_checkpoint: token_edits.FrozenContinualTokenEditSchemaV1,
        outcome_memory: outcomes.FrozenContinualActionOutcomeMemoryV1,
        checkpoint_sha256: str,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("token-edit consequence policies are factory-owned")
        self._base_checkpoint = base_checkpoint
        self._outcome_memory = outcome_memory
        self._checkpoint_sha256 = checkpoint_sha256
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def base_checkpoint(self) -> token_edits.FrozenContinualTokenEditSchemaV1:
        self.authenticate()
        return self._base_checkpoint

    @property
    def outcome_memory(self) -> outcomes.FrozenContinualActionOutcomeMemoryV1:
        self.authenticate()
        return self._outcome_memory

    @property
    def catalogue(self) -> tuple[TokenEditSchemaActionV1, ...]:
        self.authenticate()
        return _catalogue()

    def authenticate(self) -> None:
        if type(self._base_checkpoint) is not token_edits.FrozenContinualTokenEditSchemaV1:
            raise TypeError("policy base checkpoint must be exact V1")
        if type(self._outcome_memory) is not outcomes.FrozenContinualActionOutcomeMemoryV1:
            raise TypeError("policy outcome memory must be exact")
        self._base_checkpoint.authenticate()
        self._outcome_memory.authenticate()
        catalogue = _catalogue()
        for schema in token_edits.PARAMETERIZED_SCHEMAS:
            schema.authenticate()
        for item in catalogue:
            item.authenticate()
        action_roots = tuple(item.action_sha256 for item in catalogue)
        if (
            len(catalogue) != len(token_edits.PARAMETERIZED_SCHEMAS)
            or len(set(action_roots)) != len(action_roots)
            or len(action_roots) > outcomes.MAX_ACTIONS
            or self._outcome_memory.action_sha256s != action_roots
            or self._checkpoint_sha256
            != _checkpoint_sha256(
                base_checkpoint_sha256=self._base_checkpoint.checkpoint_sha256,
                outcome_checkpoint_sha256=self._outcome_memory.checkpoint_sha256,
            )
        ):
            raise ContinualTokenEditConsequencePolicyError(
                "token-edit consequence policy authentication failed"
            )

    def propose(
        self,
        source: token_edits.TokenSourceFrameV1,
        *,
        cell_gate: bool = True,
        schema_gate: bool = True,
        outcome_gate: bool = True,
        exploration_gate: bool = False,
    ) -> TokenEditConsequencePolicyProposalV1:
        self.authenticate()
        if type(source) is not token_edits.TokenSourceFrameV1:
            raise TypeError("policy proposal source must be one exact token frame")
        if any(
            type(gate) is not bool
            for gate in (cell_gate, schema_gate, outcome_gate, exploration_gate)
        ):
            raise TypeError("policy proposal gates must be exact booleans")
        source.authenticate()
        candidate_view, activation, decision = _derive_policy_selection(
            self,
            source,
            cell_gate=cell_gate,
            schema_gate=schema_gate,
            outcome_gate=outcome_gate,
            exploration_gate=exploration_gate,
        )
        proposal = _materialize_policy_proposal(
            self,
            candidate_view,
            activation,
            decision,
        )
        proposal.authenticate()
        return proposal


def _freeze_policy(
    *,
    base_checkpoint: token_edits.FrozenContinualTokenEditSchemaV1,
    outcome_memory: outcomes.FrozenContinualActionOutcomeMemoryV1,
) -> FrozenContinualTokenEditConsequencePolicyV1:
    return FrozenContinualTokenEditConsequencePolicyV1(
        base_checkpoint=base_checkpoint,
        outcome_memory=outcome_memory,
        checkpoint_sha256=_checkpoint_sha256(
            base_checkpoint_sha256=base_checkpoint.checkpoint_sha256,
            outcome_checkpoint_sha256=outcome_memory.checkpoint_sha256,
        ),
        construction_token=_CONSTRUCTION_TOKEN,
    )


def start_continual_token_edit_consequence_policy_v1(
    base_checkpoint: token_edits.FrozenContinualTokenEditSchemaV1,
) -> FrozenContinualTokenEditConsequencePolicyV1:
    if type(base_checkpoint) is not token_edits.FrozenContinualTokenEditSchemaV1:
        raise TypeError("policy start requires one exact frozen V1 checkpoint")
    base_checkpoint.authenticate()
    memory = outcomes.start_continual_action_outcome_memory_v1(
        tuple(item.action_sha256 for item in _catalogue())
    )
    return _freeze_policy(base_checkpoint=base_checkpoint, outcome_memory=memory)


def _schema_binding(
    schema: token_edits.ParameterizedTokenEditSchemaV1,
    source: token_edits.TokenSourceFrameV1,
) -> tuple[int, int, int] | None:
    """Reproduce V1 structural binding without replacing or projecting a token."""

    bindings: set[tuple[int, int, int]] = set()
    for anchor, kind in enumerate(source.kinds):
        if kind != schema.anchor_kind:
            continue
        target = _nearest_filler_v1(source, anchor, schema.target_direction)
        if target is None:
            continue
        target_kind = source.kinds[target]
        target_token = source.tokens[target]
        mirror = _nearest_filler_v1(
            source,
            anchor,
            -schema.target_direction,
            required_kind=target_kind,
        )
        if mirror is None or source.tokens[mirror] != target_token:
            continue
        donor = _nearest_filler_v1(
            source,
            target,
            schema.donor_direction,
            required_kind=target_kind,
            distinct_from=target_token,
        )
        if donor is not None:
            bindings.add((anchor, target, donor))
    if len(bindings) != 1:
        return None
    return next(iter(bindings))


@dataclass(frozen=True, slots=True)
class TokenEditCandidateViewV1:
    """Candidate-only V1 projection with no winner, binding, or edited source."""

    base_checkpoint: token_edits.FrozenContinualTokenEditSchemaV1
    source: token_edits.TokenSourceFrameV1
    status: str
    applicable_schema_indices: tuple[int, ...]
    schema_support_counts: tuple[int, ...]
    schema_base_scores: tuple[int, ...]
    cell_gate: bool
    schema_gate: bool
    base_checkpoint_sha256: str
    source_frame_sha256: str
    source_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "status": self.status,
            "applicable_schema_indices": self.applicable_schema_indices,
            "schema_support_counts": self.schema_support_counts,
            "schema_base_scores": self.schema_base_scores,
            "cell_gate": self.cell_gate,
            "schema_gate": self.schema_gate,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "source_frame_sha256": self.source_frame_sha256,
            "source_sha256": self.source_sha256,
            "v1_race_run": False,
            "candidate_bindings_exposed": False,
            "edited_source_exposed": False,
        }

    def authenticate(self) -> None:
        if type(self.base_checkpoint) is not token_edits.FrozenContinualTokenEditSchemaV1:
            raise TypeError("candidate view checkpoint must be exact V1")
        if type(self.source) is not token_edits.TokenSourceFrameV1:
            raise TypeError("candidate view source must be one exact token frame")
        self.base_checkpoint.authenticate()
        self.source.authenticate()
        if (
            self.status not in _CANDIDATE_STATUSES
            or type(self.cell_gate) is not bool
            or type(self.schema_gate) is not bool
            or type(self.applicable_schema_indices) is not tuple
            or type(self.schema_support_counts) is not tuple
            or type(self.schema_base_scores) is not tuple
            or self.base_checkpoint_sha256 != self.base_checkpoint.checkpoint_sha256
            or self.source_frame_sha256 != self.source.sha256
            or self.source_sha256 != self.source.source_sha256
        ):
            raise ContinualTokenEditConsequencePolicyError(
                "token-edit candidate view structure failed"
            )
        canonical = _derive_candidate_view(
            self.base_checkpoint,
            self.source,
            cell_gate=self.cell_gate,
            schema_gate=self.schema_gate,
        )
        if self.record() != canonical.record():
            raise ContinualTokenEditConsequencePolicyError(
                "token-edit candidate view does not rederive"
            )


def _derive_candidate_view(
    base_checkpoint: token_edits.FrozenContinualTokenEditSchemaV1,
    source: token_edits.TokenSourceFrameV1,
    *,
    cell_gate: bool,
    schema_gate: bool,
) -> TokenEditCandidateViewV1:
    support = base_checkpoint.support_counts
    applicable: tuple[int, ...] = ()
    seen_source_roots = {
        value
        for row in base_checkpoint.provenance
        for value in (row.failed_source_sha256, row.successful_source_sha256)
    }
    if not cell_gate:
        status = STOP_CELL_OFF
    elif not schema_gate:
        status = STOP_SCHEMA_OFF
    elif source.source_sha256 in seen_source_roots:
        status = STOP_SEEN_SOURCE
    else:
        applicable = tuple(
            schema.schema_index
            for schema in token_edits.PARAMETERIZED_SCHEMAS
            if _schema_binding(schema, source) is not None
        )
        status = CANDIDATES_READY if applicable else STOP_NO_SCHEMA_MATCH
    scores = tuple(
        min(outcomes.MAX_OUTPUT_SCORE, support[index] * 2) if index in applicable else 0
        for index in range(len(token_edits.PARAMETERIZED_SCHEMAS))
    )
    return TokenEditCandidateViewV1(
        base_checkpoint=base_checkpoint,
        source=source,
        status=status,
        applicable_schema_indices=applicable,
        schema_support_counts=support,
        schema_base_scores=scores,
        cell_gate=cell_gate,
        schema_gate=schema_gate,
        base_checkpoint_sha256=base_checkpoint.checkpoint_sha256,
        source_frame_sha256=source.sha256,
        source_sha256=source.source_sha256,
    )


def _candidate_records(
    view: TokenEditCandidateViewV1,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "schema_index": index,
            "action_sha256": token_edits.PARAMETERIZED_SCHEMAS[index].sha256,
            "base_score": view.schema_base_scores[index],
        }
        for index in view.applicable_schema_indices
    )


def _selection_context_sha256(
    policy: FrozenContinualTokenEditConsequencePolicyV1,
    view: TokenEditCandidateViewV1,
) -> str:
    return _digest(
        _SELECTION_CONTEXT_NAMESPACE,
        {
            "policy_checkpoint_sha256": policy.checkpoint_sha256,
            "candidate_view": view.record(),
            "candidates": _candidate_records(view),
        },
    )


def _derive_policy_selection(
    policy: FrozenContinualTokenEditConsequencePolicyV1,
    source: token_edits.TokenSourceFrameV1,
    *,
    cell_gate: bool,
    schema_gate: bool,
    outcome_gate: bool,
    exploration_gate: bool,
) -> tuple[
    TokenEditCandidateViewV1,
    outcomes.ActionActivationV1,
    outcomes.ActionDecisionV1,
]:
    view = _derive_candidate_view(
        policy._base_checkpoint,
        source,
        cell_gate=cell_gate,
        schema_gate=schema_gate,
    )
    candidates = _candidate_records(view)
    activation = outcomes.seal_action_activation_v1(
        selection_context_sha256=_selection_context_sha256(policy, view),
        active_action_sha256s=tuple(str(candidate["action_sha256"]) for candidate in candidates),
        base_scores=tuple(int(candidate["base_score"]) for candidate in candidates),
    )
    decision = policy._outcome_memory.decide(
        activation,
        cell_gate=cell_gate,
        outcome_gate=outcome_gate,
        exploration_gate=exploration_gate,
    )
    return view, activation, decision


@dataclass(frozen=True, slots=True)
class TokenEditConsequencePolicyProposalV1:
    policy: FrozenContinualTokenEditConsequencePolicyV1
    candidate_view: TokenEditCandidateViewV1
    activation: outcomes.ActionActivationV1
    decision: outcomes.ActionDecisionV1
    status: str
    selected_schema_index: int | None
    selected_schema_sha256: str | None
    anchor_token_index: int | None
    target_token_index: int | None
    donor_token_index: int | None
    edited_source_frame_sha256: str | None
    edited_source_sha256: str | None
    edited_source: str | None
    policy_checkpoint_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "candidate_view": self.candidate_view.record(),
            "activation_sha256": self.activation.sha256,
            "decision_sha256": self.decision.sha256,
            "status": self.status,
            "selected_schema_index": self.selected_schema_index,
            "selected_schema_sha256": self.selected_schema_sha256,
            "anchor_token_index": self.anchor_token_index,
            "target_token_index": self.target_token_index,
            "donor_token_index": self.donor_token_index,
            "edited_source_frame_sha256": self.edited_source_frame_sha256,
            "edited_source_sha256": self.edited_source_sha256,
            "raw_edited_source_retained_by_policy_checkpoint": False,
        }

    def authenticate(self) -> None:
        if type(self.policy) is not FrozenContinualTokenEditConsequencePolicyV1:
            raise TypeError("proposal policy must be exact")
        if type(self.candidate_view) is not TokenEditCandidateViewV1:
            raise TypeError("proposal candidate view must be exact")
        if type(self.activation) is not outcomes.ActionActivationV1:
            raise TypeError("proposal activation must be exact")
        if type(self.decision) is not outcomes.ActionDecisionV1:
            raise TypeError("proposal decision must be exact")
        self.policy.authenticate()
        self.candidate_view.authenticate()
        self.activation.authenticate()
        self.decision.authenticate()
        expected_view, expected_activation, expected_decision = _derive_policy_selection(
            self.policy,
            self.candidate_view.source,
            cell_gate=self.candidate_view.cell_gate,
            schema_gate=self.candidate_view.schema_gate,
            outcome_gate=self.decision.outcome_gate,
            exploration_gate=self.decision.exploration_gate,
        )
        selected_values = (
            self.selected_schema_index,
            self.selected_schema_sha256,
            self.anchor_token_index,
            self.target_token_index,
            self.donor_token_index,
            self.edited_source_frame_sha256,
            self.edited_source_sha256,
            self.edited_source,
        )
        if (
            self.status not in _READY_STATUSES | _POLICY_STOP_STATUSES
            or self.candidate_view.base_checkpoint is not self.policy.base_checkpoint
            or self.candidate_view.record() != expected_view.record()
            or self.activation != expected_activation
            or self.decision != expected_decision
            or self.activation != self.decision.activation
            or self.decision.checkpoint_sha256 != self.policy.outcome_memory.checkpoint_sha256
            or self.decision.catalogue_sha256 != self.policy.outcome_memory.catalogue_sha256
            or self.policy_checkpoint_sha256 != self.policy.checkpoint_sha256
        ):
            raise ContinualTokenEditConsequencePolicyError(
                "token-edit consequence proposal structure failed"
            )
        if self.candidate_view.status != CANDIDATES_READY:
            if self.status != self.candidate_view.status or any(
                value is not None for value in selected_values
            ):
                raise ContinualTokenEditConsequencePolicyError(
                    "candidate STOP exposed a selected edit"
                )
            return
        if self.decision.status not in _READY_STATUSES:
            if self.status != self.decision.status or any(
                value is not None for value in selected_values
            ):
                raise ContinualTokenEditConsequencePolicyError(
                    "generic STOP exposed a selected edit"
                )
            return
        selected_root = self.decision.selected_action_sha256
        action_to_schema = {item.action_sha256: item.schema_index for item in self.policy.catalogue}
        if selected_root not in action_to_schema:
            raise ContinualTokenEditConsequencePolicyError(
                "generic decision selected an unknown schema root"
            )
        schema_index = action_to_schema[selected_root]
        binding = _schema_binding(
            token_edits.PARAMETERIZED_SCHEMAS[schema_index],
            self.candidate_view.source,
        )
        if binding is None or schema_index not in self.candidate_view.applicable_schema_indices:
            raise ContinualTokenEditConsequencePolicyError(
                "generic-selected schema has no authentic candidate binding"
            )
        if self.status == STOP_ARTIFACT_REPLAY:
            if any(value is not None for value in selected_values):
                raise ContinualTokenEditConsequencePolicyError(
                    "artifact replay STOP exposed an edit"
                )
            application = _apply_schema_v1(
                token_edits.PARAMETERIZED_SCHEMAS[schema_index],
                self.candidate_view.source,
            )
            seen = {
                value
                for row in self.policy.base_checkpoint.provenance
                for value in (row.failed_source_sha256, row.successful_source_sha256)
            }
            if application is None or application.edited_source.source_sha256 not in seen:
                raise ContinualTokenEditConsequencePolicyError(
                    "artifact replay STOP does not rederive"
                )
            return
        if self.status != self.decision.status or any(value is None for value in selected_values):
            raise ContinualTokenEditConsequencePolicyError("READY proposal is incomplete")
        self._authenticate_selected_edit(schema_index, selected_root, binding)

    def _authenticate_selected_edit(
        self,
        schema_index: int,
        selected_root: str,
        binding: tuple[int, int, int],
    ) -> None:
        anchor, target, donor = binding
        if type(self.edited_source) is not str:
            raise ContinualTokenEditConsequencePolicyError("READY edit source is not exact")
        edited = token_edits.project_token_source_v1(self.edited_source)
        expected_tokens = list(self.candidate_view.source.tokens)
        expected_tokens[target] = self.candidate_view.source.tokens[donor]
        if (
            self.selected_schema_index != schema_index
            or self.selected_schema_sha256 != selected_root
            or self.anchor_token_index != anchor
            or self.target_token_index != target
            or self.donor_token_index != donor
            or self.edited_source_frame_sha256 != edited.sha256
            or self.edited_source_sha256 != edited.source_sha256
            or edited.tokens != tuple(expected_tokens)
            or edited.kinds != self.candidate_view.source.kinds
            or edited.gaps != self.candidate_view.source.gaps
        ):
            raise ContinualTokenEditConsequencePolicyError(
                "generic-selected token edit failed exact authentication"
            )


def _materialize_policy_proposal(
    policy: FrozenContinualTokenEditConsequencePolicyV1,
    view: TokenEditCandidateViewV1,
    activation: outcomes.ActionActivationV1,
    decision: outcomes.ActionDecisionV1,
) -> TokenEditConsequencePolicyProposalV1:
    status = view.status if view.status != CANDIDATES_READY else decision.status
    selected_schema_index: int | None = None
    selected_schema_sha256: str | None = None
    anchor_token_index: int | None = None
    target_token_index: int | None = None
    donor_token_index: int | None = None
    edited_source_frame_sha256: str | None = None
    edited_source_sha256: str | None = None
    edited_source: str | None = None
    if view.status == CANDIDATES_READY and decision.status in _READY_STATUSES:
        action_to_schema = {item.action_sha256: item.schema_index for item in policy.catalogue}
        selected_root = decision.selected_action_sha256
        if selected_root not in action_to_schema:
            raise RuntimeError("generic policy selected an unknown V1 schema root")
        schema_index = action_to_schema[selected_root]
        application = _apply_schema_v1(
            token_edits.PARAMETERIZED_SCHEMAS[schema_index],
            view.source,
        )
        if application is None or schema_index not in view.applicable_schema_indices:
            raise RuntimeError("selected V1 schema did not materialize from its activation")
        seen_source_roots = {
            value
            for row in policy._base_checkpoint.provenance
            for value in (row.failed_source_sha256, row.successful_source_sha256)
        }
        if application.edited_source.source_sha256 in seen_source_roots:
            status = STOP_ARTIFACT_REPLAY
        else:
            selected_schema_index = schema_index
            selected_schema_sha256 = selected_root
            anchor_token_index = application.anchor_token_index
            target_token_index = application.target_token_index
            donor_token_index = application.donor_token_index
            edited_source_frame_sha256 = application.edited_source.sha256
            edited_source_sha256 = application.edited_source.source_sha256
            edited_source = application.edited_source.source
    return TokenEditConsequencePolicyProposalV1(
        policy=policy,
        candidate_view=view,
        activation=activation,
        decision=decision,
        status=status,
        selected_schema_index=selected_schema_index,
        selected_schema_sha256=selected_schema_sha256,
        anchor_token_index=anchor_token_index,
        target_token_index=target_token_index,
        donor_token_index=donor_token_index,
        edited_source_frame_sha256=edited_source_frame_sha256,
        edited_source_sha256=edited_source_sha256,
        edited_source=edited_source,
        policy_checkpoint_sha256=policy._checkpoint_sha256,
    )


@dataclass(frozen=True, slots=True)
class TokenEditConsequencePolicyOutcomeV1:
    proposal: TokenEditConsequencePolicyProposalV1
    consequence: token_edits.TokenEditConsequenceV1
    selected_outcome: outcomes.SelectedActionOutcomeV1
    passed_cell: int

    def record(self) -> dict[str, object]:
        return {
            "proposal": self.proposal.record(),
            "consequence_sha256": self.consequence.sha256,
            "selected_outcome_sha256": self.selected_outcome.sha256,
            "selected_schema_sha256": self.proposal.selected_schema_sha256,
            "proposed_source_sha256": self.proposal.edited_source_sha256,
            "execution_sha256": self.consequence.execution_sha256,
            "passed_cell": self.passed_cell,
        }

    def authenticate(self) -> None:
        if type(self.proposal) is not TokenEditConsequencePolicyProposalV1:
            raise TypeError("typed outcome proposal must be exact")
        if type(self.consequence) is not token_edits.TokenEditConsequenceV1:
            raise TypeError("typed outcome consequence must be exact")
        if type(self.selected_outcome) is not outcomes.SelectedActionOutcomeV1:
            raise TypeError("typed outcome generic receipt must be exact")
        self.proposal.authenticate()
        self.consequence.authenticate()
        self.selected_outcome.authenticate()
        canonical = _derive_policy_outcome(self.proposal, self.consequence)
        if self.record() != canonical.record():
            raise ContinualTokenEditConsequencePolicyError(
                "typed token-edit outcome does not rederive"
            )


def _record_custody_roots(record: dict[str, object]) -> set[str]:
    roots: set[str] = set()
    for key, value in record.items():
        if key == "sha256" or key.endswith("_sha256"):
            if type(value) is str:
                roots.add(value)
        elif key.endswith("_sha256s") and type(value) is tuple:
            roots.update(item for item in value if type(item) is str)
    return roots


def _reserved_consequence_roots(
    proposal: TokenEditConsequencePolicyProposalV1,
) -> set[str]:
    policy = proposal.policy
    base = policy.base_checkpoint
    outcome_memory = policy.outcome_memory
    roots = {
        policy.checkpoint_sha256,
        base.checkpoint_sha256,
        base.checkpoint_array_sha256,
        outcome_memory.checkpoint_sha256,
        outcome_memory.catalogue_sha256,
        proposal.candidate_view.source_frame_sha256,
        proposal.candidate_view.source_sha256,
        proposal.activation.sha256,
        proposal.activation.selection_context_sha256,
        proposal.decision.sha256,
        proposal.edited_source_frame_sha256,
        *(item.action_sha256 for item in policy.catalogue),
    }
    for row in base.provenance:
        roots.update(_record_custody_roots(row.record()))
    for row in outcome_memory.provenance:
        roots.update(_record_custody_roots(row.record()))
    return {root for root in roots if type(root) is str}


def _derive_policy_outcome(
    proposal: TokenEditConsequencePolicyProposalV1,
    consequence: token_edits.TokenEditConsequenceV1,
) -> TokenEditConsequencePolicyOutcomeV1:
    if proposal.status not in _READY_STATUSES or proposal.edited_source_sha256 is None:
        raise ContinualTokenEditConsequencePolicyError(
            "a STOP policy proposal has no executable consequence"
        )
    if consequence.source_sha256 != proposal.edited_source_sha256:
        raise ContinualTokenEditConsequencePolicyError(
            "token-edit consequence is not for the exact proposed source"
        )
    consequence_event_roots = (
        consequence.source_sha256,
        consequence.batch_sha256,
        consequence.test_manifest_sha256,
        consequence.execution_sha256,
        consequence.feedback_sha256,
        consequence.sha256,
    )
    if len(set(consequence_event_roots)) != len(consequence_event_roots):
        raise ContinualTokenEditConsequencePolicyError(
            "typed consequence reuses a custody root across event roles"
        )
    if _reserved_consequence_roots(proposal).intersection(consequence_event_roots):
        raise ContinualTokenEditConsequencePolicyError(
            "typed consequence aliases policy or historical custody"
        )
    passed_cell = int(
        consequence.compile_cell == 1
        and consequence.run_cell == 1
        and consequence.public_test_cell == 1
    )
    selected = outcomes.seal_selected_action_outcome_v1(
        decision=proposal.decision,
        proposed_source_sha256=proposal.edited_source_sha256,
        execution_sha256=consequence.execution_sha256,
        consequence_sha256=consequence.sha256,
        auxiliary_event_sha256s=tuple(
            sorted(
                (
                    consequence.batch_sha256,
                    consequence.test_manifest_sha256,
                    consequence.feedback_sha256,
                )
            )
        ),
        passed_cell=passed_cell,
    )
    return TokenEditConsequencePolicyOutcomeV1(
        proposal=proposal,
        consequence=consequence,
        selected_outcome=selected,
        passed_cell=passed_cell,
    )


def seal_token_edit_consequence_policy_outcome_v1(
    *,
    proposal: TokenEditConsequencePolicyProposalV1,
    consequence: token_edits.TokenEditConsequenceV1,
) -> TokenEditConsequencePolicyOutcomeV1:
    if type(proposal) is not TokenEditConsequencePolicyProposalV1:
        raise TypeError("outcome seal requires one exact policy proposal")
    if type(consequence) is not token_edits.TokenEditConsequenceV1:
        raise TypeError("outcome seal requires one exact token-edit consequence")
    proposal.authenticate()
    consequence.authenticate()
    result = _derive_policy_outcome(proposal, consequence)
    result.authenticate()
    return result


def advance_continual_token_edit_consequence_policy_v1(
    parent: FrozenContinualTokenEditConsequencePolicyV1,
    outcome: TokenEditConsequencePolicyOutcomeV1,
    *,
    plasticity_gate: bool = True,
    consequence_gate: bool = True,
) -> FrozenContinualTokenEditConsequencePolicyV1:
    if type(parent) is not FrozenContinualTokenEditConsequencePolicyV1:
        raise TypeError("policy advance requires one exact parent")
    if type(outcome) is not TokenEditConsequencePolicyOutcomeV1:
        raise TypeError("policy advance requires one exact typed outcome")
    if type(plasticity_gate) is not bool or type(consequence_gate) is not bool:
        raise TypeError("policy advance gates must be exact booleans")
    parent.authenticate()
    outcome.authenticate()
    proposal = outcome.proposal
    if (
        proposal.policy_checkpoint_sha256 != parent.checkpoint_sha256
        or proposal.candidate_view.base_checkpoint is not parent.base_checkpoint
        or proposal.decision.checkpoint_sha256 != parent.outcome_memory.checkpoint_sha256
    ):
        raise ContinualTokenEditConsequencePolicyError(
            "typed outcome proposal is not from this exact parent"
        )
    base_root = parent._base_checkpoint.checkpoint_sha256
    base_array_root = parent._base_checkpoint.checkpoint_array_sha256
    base_provenance = parent._base_checkpoint.provenance
    policy_root = parent._checkpoint_sha256
    outcome_root = parent._outcome_memory.checkpoint_sha256
    outcome_arrays = parent._outcome_memory.checkpoint_array_sha256s
    outcome_provenance = parent._outcome_memory.provenance
    successor_memory = outcomes.advance_continual_action_outcome_memory_v1(
        parent._outcome_memory,
        outcome.selected_outcome,
        plasticity_gate=plasticity_gate,
        consequence_gate=consequence_gate,
    )
    if successor_memory is parent._outcome_memory:
        return parent
    successor = _freeze_policy(
        base_checkpoint=parent._base_checkpoint,
        outcome_memory=successor_memory,
    )
    parent.authenticate()
    if (
        parent._checkpoint_sha256 != policy_root
        or parent._base_checkpoint.checkpoint_sha256 != base_root
        or parent._base_checkpoint.checkpoint_array_sha256 != base_array_root
        or parent._base_checkpoint.provenance != base_provenance
        or parent._outcome_memory.checkpoint_sha256 != outcome_root
        or parent._outcome_memory.checkpoint_array_sha256s != outcome_arrays
        or parent._outcome_memory.provenance != outcome_provenance
        or successor._base_checkpoint is not parent._base_checkpoint
    ):
        raise RuntimeError("token-edit consequence policy advance mutated its parent")
    return successor


__all__ = [
    "CANDIDATES_READY",
    "CLAIM_BOUNDARY",
    "EXPLORATION_READY",
    "READY",
    "SCHEMA",
    "STOP_ARTIFACT_REPLAY",
    "STOP_CELL_OFF",
    "STOP_INSUFFICIENT_OUTCOME_EVIDENCE",
    "STOP_NO_CANDIDATE",
    "STOP_NO_SCHEMA_MATCH",
    "STOP_PHYSICAL_RACE",
    "STOP_SCHEMA_OFF",
    "STOP_SEEN_SOURCE",
    "ContinualTokenEditConsequencePolicyError",
    "FrozenContinualTokenEditConsequencePolicyV1",
    "TokenEditCandidateViewV1",
    "TokenEditConsequencePolicyOutcomeV1",
    "TokenEditConsequencePolicyProposalV1",
    "TokenEditSchemaActionV1",
    "advance_continual_token_edit_consequence_policy_v1",
    "seal_token_edit_consequence_policy_outcome_v1",
    "start_continual_token_edit_consequence_policy_v1",
]
