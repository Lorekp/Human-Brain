"""Unordered-cue front end for the fixed two-axis factor memory.

Every opaque cue is projected onto both supplied action axes.  The projection
is uniform: callers cannot assign a cue to an axis.  Selected composite
PASS/FAIL outcomes then update cue x action-axis x component E/H bindings in
the established immutable factor memory, whose physical action race remains
the sole full-hypothesis selector.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from experiments.r3_pilot import common_continual_action_outcome_memory_v1 as outcomes
from experiments.r3_pilot import common_factorized_feedback_hypothesis_memory_v1 as factors

SCHEMA = "COMMON_UNORDERED_CUE_FACTOR_MEMORY_V1"
CLAIM_BOUNDARY = (
    "caller-attested-selected-composite-pass-fail-learning-of-opaque-unordered-"
    "cue-by-supplied-action-axis-by-supplied-component-bindings-through-the-"
    "established-physical-full-hypothesis-race-not-cue-action-axis-component-"
    "or-operation-discovery-semantics-localization-synthesis-debugging-transfer-"
    "swe-or-intelligence"
)

READY = outcomes.READY
EXPLORATION_READY = outcomes.EXPLORATION_READY
STOP_CELL_OFF = outcomes.STOP_CELL_OFF
STOP_NO_CANDIDATE = outcomes.STOP_NO_CANDIDATE
STOP_PHYSICAL_RACE = outcomes.STOP_PHYSICAL_RACE
STOP_INSUFFICIENT_OUTCOME_EVIDENCE = outcomes.STOP_INSUFFICIENT_OUTCOME_EVIDENCE

MAX_CUES_PER_OBSERVATION = factors.MAX_CUES_PER_AXIS
MAX_PROVENANCE_ROWS = factors.MAX_PROVENANCE_ROWS

_CONSTRUCTION_TOKEN = object()
_SHA256_HEX = frozenset("0123456789abcdef")

_OBSERVATION_NAMESPACE = b"common-unordered-cue-factor-observation-v1"
_BINDING_NAMESPACE = b"common-unordered-cue-factor-binding-v1"
_DECISION_NAMESPACE = b"common-unordered-cue-factor-decision-v1"
_OUTCOME_NAMESPACE = b"common-unordered-cue-factor-outcome-v1"
_CHECKPOINT_NAMESPACE = b"common-unordered-cue-factor-checkpoint-v1"


class UnorderedCueFactorMemoryError(ValueError):
    """An unordered observation, decision, outcome, or checkpoint is invalid."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _canonical_roots(
    values: object,
    *,
    label: str,
    nonempty: bool = False,
    maximum: int | None = None,
) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{label} must be one exact tuple")
    if any(not _valid_sha256(value) for value in values):
        raise UnorderedCueFactorMemoryError(f"{label} contains an invalid root")
    canonical = tuple(sorted(values))
    if canonical != values or len(set(values)) != len(values):
        raise UnorderedCueFactorMemoryError(f"{label} must be canonical and unique")
    if nonempty and not values:
        raise UnorderedCueFactorMemoryError(f"{label} must not be empty")
    if maximum is not None and len(values) > maximum:
        raise UnorderedCueFactorMemoryError(f"{label} exceeds capacity")
    return canonical


@dataclass(frozen=True, slots=True)
class UnorderedCueFactorObservationV1:
    """One rooted context containing only an unordered opaque cue set."""

    selection_context_sha256: str
    cue_sha256s: tuple[str, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "selection_context_sha256": self.selection_context_sha256,
            "cue_sha256s": self.cue_sha256s,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        cues = _canonical_roots(
            self.cue_sha256s,
            label="unordered cue roots",
            nonempty=True,
            maximum=MAX_CUES_PER_OBSERVATION,
        )
        if (
            cues != self.cue_sha256s
            or not _valid_sha256(self.selection_context_sha256)
            or self.selection_context_sha256 in cues
            or self.sha256 != _digest(_OBSERVATION_NAMESPACE, self.record(include_sha256=False))
        ):
            raise UnorderedCueFactorMemoryError(
                "unordered cue factor observation authentication failed"
            )


def seal_unordered_cue_factor_observation_v1(
    *,
    selection_context_sha256: str,
    cue_sha256s: tuple[str, ...],
) -> UnorderedCueFactorObservationV1:
    cues = tuple(sorted(cue_sha256s)) if type(cue_sha256s) is tuple else cue_sha256s
    payload = {
        "selection_context_sha256": selection_context_sha256,
        "cue_sha256s": cues,
    }
    result = UnorderedCueFactorObservationV1(
        **payload,
        sha256=_digest(_OBSERVATION_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def cue_axis_binding_sha256_v1(cue_sha256: str, action_axis: int) -> str:
    """Return one uniform presynaptic cue x postsynaptic action-axis address."""

    if not _valid_sha256(cue_sha256):
        raise UnorderedCueFactorMemoryError("cue-axis binding cue root is invalid")
    if type(action_axis) is not int or action_axis not in (0, 1):
        raise UnorderedCueFactorMemoryError("action axis must be exact zero or one")
    return _digest(
        _BINDING_NAMESPACE,
        {"cue_sha256": cue_sha256, "action_axis": action_axis},
    )


def _binding_observation(
    observation: UnorderedCueFactorObservationV1,
    active_hypothesis_sha256s: tuple[str, ...],
) -> factors.FactorizedFeedbackObservationV1:
    observation.authenticate()
    axes = tuple(
        tuple(
            sorted(
                cue_axis_binding_sha256_v1(cue_sha256, action_axis)
                for cue_sha256 in observation.cue_sha256s
            )
        )
        for action_axis in range(2)
    )
    return factors.seal_factorized_feedback_observation_v1(
        selection_context_sha256=observation.selection_context_sha256,
        axis_cue_sha256s=axes,  # type: ignore[arg-type]
        active_hypothesis_sha256s=active_hypothesis_sha256s,
    )


@dataclass(frozen=True, slots=True)
class UnorderedCueFactorDecisionV1:
    """Raw unordered custody plus the genuine inner physical factor decision."""

    checkpoint_sha256: str
    observation: UnorderedCueFactorObservationV1
    binding_observation: factors.FactorizedFeedbackObservationV1
    factorized_decision: factors.FactorizedFeedbackDecisionV1
    sha256: str

    @property
    def status(self) -> str:
        return self.factorized_decision.status

    @property
    def selected_hypothesis_sha256(self) -> str | None:
        return self.factorized_decision.selected_hypothesis_sha256

    @property
    def cell_gate(self) -> bool:
        return self.factorized_decision.cell_gate

    @property
    def outcome_gate(self) -> bool:
        return self.factorized_decision.outcome_gate

    @property
    def axis_gates(self) -> tuple[bool, bool]:
        return self.factorized_decision.axis_gates

    @property
    def exploration_hypothesis_sha256(self) -> str | None:
        return self.factorized_decision.exploration_hypothesis_sha256

    @property
    def hypothesis_scores(self) -> tuple[factors.FactorizedHypothesisScoreV1, ...]:
        return self.factorized_decision.hypothesis_scores

    @property
    def action_decision(self) -> outcomes.ActionDecisionV1:
        return self.factorized_decision.action_decision

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "checkpoint_sha256": self.checkpoint_sha256,
            "observation": self.observation.record(),
            "binding_observation": self.binding_observation.record(),
            "factorized_decision": self.factorized_decision.record(),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.observation) is not UnorderedCueFactorObservationV1:
            raise TypeError("unordered factor decision observation must be exact")
        if type(self.binding_observation) is not factors.FactorizedFeedbackObservationV1:
            raise TypeError("unordered factor binding observation must be exact")
        if type(self.factorized_decision) is not factors.FactorizedFeedbackDecisionV1:
            raise TypeError("unordered factor physical decision must be exact")
        self.observation.authenticate()
        self.binding_observation.authenticate()
        self.factorized_decision.authenticate()
        expected_binding = _binding_observation(
            self.observation,
            self.factorized_decision.observation.active_hypothesis_sha256s,
        )
        if (
            not _valid_sha256(self.checkpoint_sha256)
            or self.binding_observation != expected_binding
            or self.factorized_decision.observation != self.binding_observation
            or self.sha256 != _digest(_DECISION_NAMESPACE, self.record(include_sha256=False))
        ):
            raise UnorderedCueFactorMemoryError(
                "unordered cue factor decision authentication failed"
            )


def _seal_decision(
    *,
    checkpoint_sha256: str,
    observation: UnorderedCueFactorObservationV1,
    binding_observation: factors.FactorizedFeedbackObservationV1,
    factorized_decision: factors.FactorizedFeedbackDecisionV1,
) -> UnorderedCueFactorDecisionV1:
    payload = {
        "checkpoint_sha256": checkpoint_sha256,
        "observation": observation,
        "binding_observation": binding_observation,
        "factorized_decision": factorized_decision,
    }
    result = UnorderedCueFactorDecisionV1(
        **payload,
        sha256=_digest(
            _DECISION_NAMESPACE,
            {
                **payload,
                "observation": observation.record(),
                "binding_observation": binding_observation.record(),
                "factorized_decision": factorized_decision.record(),
            },
        ),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class UnorderedCueFactorOutcomeV1:
    """One outer raw-cue decision and its exact inner selected outcome."""

    decision: UnorderedCueFactorDecisionV1
    factorized_outcome: factors.FactorizedFeedbackOutcomeV1
    sha256: str

    @property
    def selected_hypothesis_sha256(self) -> str:
        return self.factorized_outcome.selected_hypothesis_sha256

    @property
    def passed_cell(self) -> int:
        return self.factorized_outcome.passed_cell

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "decision": self.decision.record(),
            "factorized_outcome": self.factorized_outcome.record(),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.decision) is not UnorderedCueFactorDecisionV1:
            raise TypeError("unordered factor outcome decision must be exact")
        if type(self.factorized_outcome) is not factors.FactorizedFeedbackOutcomeV1:
            raise TypeError("unordered factor inner outcome must be exact")
        self.decision.authenticate()
        self.factorized_outcome.authenticate()
        expected = factors.seal_factorized_feedback_outcome_v1(
            decision=self.decision.factorized_decision,
            proposal_sha256=self.factorized_outcome.proposal_sha256,
            execution_sha256=self.factorized_outcome.execution_sha256,
            consequence_sha256=self.factorized_outcome.consequence_sha256,
            feedback_sha256=self.factorized_outcome.feedback_sha256,
            passed_cell=self.factorized_outcome.passed_cell,
        )
        event_roots = _outcome_event_roots(self)
        if (
            self.factorized_outcome != expected
            or len(set(event_roots)) != len(event_roots)
            or set(self.decision.observation.cue_sha256s).intersection(event_roots)
            or self.sha256 != _digest(_OUTCOME_NAMESPACE, self.record(include_sha256=False))
        ):
            raise UnorderedCueFactorMemoryError(
                "unordered cue factor outcome authentication failed"
            )


def seal_unordered_cue_factor_outcome_v1(
    *,
    decision: UnorderedCueFactorDecisionV1,
    proposal_sha256: str,
    execution_sha256: str,
    consequence_sha256: str,
    feedback_sha256: str,
    passed_cell: int,
) -> UnorderedCueFactorOutcomeV1:
    if type(decision) is not UnorderedCueFactorDecisionV1:
        raise TypeError("unordered factor outcome requires one exact decision")
    decision.authenticate()
    inner = factors.seal_factorized_feedback_outcome_v1(
        decision=decision.factorized_decision,
        proposal_sha256=proposal_sha256,
        execution_sha256=execution_sha256,
        consequence_sha256=consequence_sha256,
        feedback_sha256=feedback_sha256,
        passed_cell=passed_cell,
    )
    payload = {"decision": decision, "factorized_outcome": inner}
    result = UnorderedCueFactorOutcomeV1(
        **payload,
        sha256=_digest(
            _OUTCOME_NAMESPACE,
            {
                "decision": decision.record(),
                "factorized_outcome": inner.record(),
            },
        ),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class CueAxisComponentEvidenceV1:
    """Canonical audit row for one learned three-factor E/H binding."""

    cue_sha256: str
    action_axis: int
    component_sha256: str
    excitatory_ticks: int
    inhibitory_ticks: int

    @property
    def net_ticks(self) -> int:
        return max(0, self.excitatory_ticks - self.inhibitory_ticks)

    def record(self) -> dict[str, object]:
        return {
            "cue_sha256": self.cue_sha256,
            "action_axis": self.action_axis,
            "component_sha256": self.component_sha256,
            "excitatory_ticks": self.excitatory_ticks,
            "inhibitory_ticks": self.inhibitory_ticks,
            "net_ticks": self.net_ticks,
        }

    def authenticate(self) -> None:
        if (
            not _valid_sha256(self.cue_sha256)
            or type(self.action_axis) is not int
            or self.action_axis not in (0, 1)
            or not _valid_sha256(self.component_sha256)
            or type(self.excitatory_ticks) is not int
            or not 0 <= self.excitatory_ticks <= factors.TICK_CAP
            or type(self.inhibitory_ticks) is not int
            or not 0 <= self.inhibitory_ticks <= factors.TICK_CAP
        ):
            raise UnorderedCueFactorMemoryError("cue-axis-component evidence is invalid")


def _outcome_event_roots(outcome: UnorderedCueFactorOutcomeV1) -> tuple[str, ...]:
    decision = outcome.decision
    inner = outcome.factorized_outcome
    physical = inner.decision.action_decision
    return (
        outcome.sha256,
        decision.sha256,
        decision.observation.sha256,
        inner.sha256,
        inner.decision.sha256,
        inner.decision.observation.sha256,
        inner.decision.observation.selection_context_sha256,
        physical.sha256,
        physical.activation.sha256,
        physical.activation.selection_context_sha256,
        inner.proposal_sha256,
        inner.execution_sha256,
        inner.consequence_sha256,
        inner.feedback_sha256,
    )


def _checkpoint_sha256(
    *,
    factorized_checkpoint_sha256: str,
    provenance: tuple[UnorderedCueFactorOutcomeV1, ...],
) -> str:
    return _digest(
        _CHECKPOINT_NAMESPACE,
        {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "factorized_checkpoint_sha256": factorized_checkpoint_sha256,
            "provenance": tuple(value.record() for value in provenance),
        },
    )


def _raw_cue_sha256s(
    provenance: tuple[UnorderedCueFactorOutcomeV1, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted({cue for outcome in provenance for cue in outcome.decision.observation.cue_sha256s})
    )


def _binding_evidence(
    memory: factors.FrozenFactorizedFeedbackHypothesisMemoryV1,
    provenance: tuple[UnorderedCueFactorOutcomeV1, ...],
) -> tuple[CueAxisComponentEvidenceV1, ...]:
    cues = _raw_cue_sha256s(provenance)
    axis_cues = memory.axis_cue_sha256s
    components = memory.axis_component_sha256s
    positive = memory.excitatory_ticks
    negative = memory.inhibitory_ticks
    rows: list[CueAxisComponentEvidenceV1] = []
    for cue in cues:
        for action_axis in range(2):
            binding_root = cue_axis_binding_sha256_v1(cue, action_axis)
            try:
                cue_row = axis_cues[action_axis].index(binding_root)
            except ValueError as error:
                raise UnorderedCueFactorMemoryError(
                    "factorized state omitted an admitted uniform cue-axis binding"
                ) from error
            for component_index, component in enumerate(components[action_axis]):
                row = CueAxisComponentEvidenceV1(
                    cue_sha256=cue,
                    action_axis=action_axis,
                    component_sha256=component,
                    excitatory_ticks=positive[action_axis][cue_row][component_index],
                    inhibitory_ticks=negative[action_axis][cue_row][component_index],
                )
                row.authenticate()
                rows.append(row)
    result = tuple(
        sorted(rows, key=lambda row: (row.cue_sha256, row.action_axis, row.component_sha256))
    )
    return result


class FrozenUnorderedCueFactorMemoryV1:
    """Immutable outer raw provenance around one authenticated factor memory."""

    __slots__ = ("_checkpoint_sha256", "_factorized_memory", "_provenance")

    def __init__(
        self,
        *,
        factorized_memory: factors.FrozenFactorizedFeedbackHypothesisMemoryV1,
        provenance: tuple[UnorderedCueFactorOutcomeV1, ...],
        checkpoint_sha256: str,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("unordered cue factor memories are factory-owned")
        self._factorized_memory = factorized_memory
        self._provenance = provenance
        self._checkpoint_sha256 = checkpoint_sha256
        self.authenticate()

    @property
    def hypotheses(self) -> tuple[factors.FactorizedHypothesisV1, ...]:
        self.authenticate()
        return self._factorized_memory.hypotheses

    @property
    def hypothesis_sha256s(self) -> tuple[str, ...]:
        self.authenticate()
        return self._factorized_memory.hypothesis_sha256s

    @property
    def axis_component_sha256s(self) -> tuple[tuple[str, str], tuple[str, str]]:
        self.authenticate()
        return self._factorized_memory.axis_component_sha256s

    @property
    def cue_sha256s(self) -> tuple[str, ...]:
        self.authenticate()
        return _raw_cue_sha256s(self._provenance)

    @property
    def binding_evidence(self) -> tuple[CueAxisComponentEvidenceV1, ...]:
        self.authenticate()
        return _binding_evidence(self._factorized_memory, self._provenance)

    @property
    def provenance(self) -> tuple[UnorderedCueFactorOutcomeV1, ...]:
        self.authenticate()
        return self._provenance

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def factorized_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._factorized_memory.checkpoint_sha256

    def record(self) -> dict[str, object]:
        self.authenticate()
        return {
            "schema": SCHEMA,
            "checkpoint_sha256": self._checkpoint_sha256,
            "factorized_checkpoint_sha256": self._factorized_memory.checkpoint_sha256,
            "hypotheses": tuple(value.record() for value in self._factorized_memory.hypotheses),
            "cue_sha256s": _raw_cue_sha256s(self._provenance),
            "binding_evidence": tuple(
                value.record()
                for value in _binding_evidence(
                    self._factorized_memory,
                    self._provenance,
                )
            ),
            "provenance": tuple(value.record() for value in self._provenance),
        }

    def authenticate(self) -> None:
        if type(self._factorized_memory) is not factors.FrozenFactorizedFeedbackHypothesisMemoryV1:
            raise TypeError("unordered cue factor memory must wrap one exact factor memory")
        self._factorized_memory.authenticate()
        if (
            type(self._provenance) is not tuple
            or len(self._provenance) > MAX_PROVENANCE_ROWS
            or not _valid_sha256(self._checkpoint_sha256)
        ):
            raise UnorderedCueFactorMemoryError("unordered cue factor checkpoint framing changed")
        if any(type(value) is not UnorderedCueFactorOutcomeV1 for value in self._provenance):
            raise TypeError("unordered cue factor provenance rows must be exact")
        if tuple(value.factorized_outcome for value in self._provenance) != (
            self._factorized_memory.provenance
        ):
            raise UnorderedCueFactorMemoryError(
                "outer raw provenance differs from factorized provenance"
            )

        raw_cues = _raw_cue_sha256s(self._provenance)
        binding_sets = tuple(
            {cue_axis_binding_sha256_v1(cue, action_axis) for cue in raw_cues}
            for action_axis in range(2)
        )
        actual_binding_sets = tuple(
            set(self._factorized_memory.axis_cue_sha256s[action_axis]) for action_axis in range(2)
        )
        if (
            binding_sets != actual_binding_sets
            or binding_sets[0].intersection(binding_sets[1])
            or set(raw_cues).intersection(binding_sets[0] | binding_sets[1])
        ):
            raise UnorderedCueFactorMemoryError(
                "factorized state is not the complete uniform cue-by-axis projection"
            )

        prefix: tuple[UnorderedCueFactorOutcomeV1, ...] = ()
        outer_prefix_roots: list[str] = []
        inner_prefix_roots: list[str] = []
        all_event_roots: set[str] = set()
        for outcome in self._provenance:
            outcome.authenticate()
            inner_parent = outcome.factorized_outcome.decision.checkpoint_sha256
            expected_parent = _checkpoint_sha256(
                factorized_checkpoint_sha256=inner_parent,
                provenance=prefix,
            )
            expected_binding = _binding_observation(
                outcome.decision.observation,
                self._factorized_memory.hypothesis_sha256s,
            )
            event_roots = _outcome_event_roots(outcome)
            if (
                outcome.decision.checkpoint_sha256 != expected_parent
                or outcome.decision.binding_observation != expected_binding
                or all_event_roots.intersection(event_roots)
            ):
                raise UnorderedCueFactorMemoryError(
                    "unordered cue factor provenance replay, splice, or parent mismatch"
                )
            outer_prefix_roots.append(expected_parent)
            inner_prefix_roots.append(inner_parent)
            all_event_roots.update(event_roots)
            prefix = (*prefix, outcome)

        expected_checkpoint = _checkpoint_sha256(
            factorized_checkpoint_sha256=self._factorized_memory.checkpoint_sha256,
            provenance=self._provenance,
        )
        bank_roots = {
            *(value.sha256 for value in self._factorized_memory.hypotheses),
            *(
                component
                for value in self._factorized_memory.hypotheses
                for component in value.component_sha256s
            ),
        }
        reserved_roots = {
            expected_checkpoint,
            self._factorized_memory.checkpoint_sha256,
            *outer_prefix_roots,
            *inner_prefix_roots,
            *bank_roots,
        }
        cue_role_roots = set(raw_cues) | binding_sets[0] | binding_sets[1]
        if (
            expected_checkpoint != self._checkpoint_sha256
            or len(set(outer_prefix_roots)) != len(outer_prefix_roots)
            or len(set(inner_prefix_roots)) != len(inner_prefix_roots)
            or all_event_roots.intersection(cue_role_roots)
            or all_event_roots.intersection(reserved_roots)
            or set(raw_cues).intersection(reserved_roots)
            or expected_checkpoint in bank_roots
        ):
            raise UnorderedCueFactorMemoryError(
                "unordered cue factor checkpoint authentication failed"
            )
        _binding_evidence(self._factorized_memory, self._provenance)

    def route(
        self,
        observation: UnorderedCueFactorObservationV1,
        *,
        cell_gate: bool = True,
        outcome_gate: bool = True,
        axis_gates: tuple[bool, bool] = (True, True),
        exploration_hypothesis_sha256: str | None = None,
    ) -> UnorderedCueFactorDecisionV1:
        self.authenticate()
        if type(observation) is not UnorderedCueFactorObservationV1:
            raise TypeError("route requires one exact unordered cue observation")
        observation.authenticate()
        reserved = {
            self._checkpoint_sha256,
            self._factorized_memory.checkpoint_sha256,
            *(value.sha256 for value in self._factorized_memory.hypotheses),
            *(
                component
                for value in self._factorized_memory.hypotheses
                for component in value.component_sha256s
            ),
        }
        raw_roots = {observation.selection_context_sha256, *observation.cue_sha256s}
        binding_roots = {
            cue_axis_binding_sha256_v1(cue, action_axis)
            for cue in observation.cue_sha256s
            for action_axis in range(2)
        }
        if raw_roots.intersection(reserved | binding_roots):
            raise UnorderedCueFactorMemoryError(
                "unordered observation aliases checkpoint, bank, or binding custody"
            )
        binding = _binding_observation(observation, self._factorized_memory.hypothesis_sha256s)
        inner = self._factorized_memory.route(
            binding,
            cell_gate=cell_gate,
            outcome_gate=outcome_gate,
            axis_gates=axis_gates,
            exploration_hypothesis_sha256=exploration_hypothesis_sha256,
        )
        result = _seal_decision(
            checkpoint_sha256=self._checkpoint_sha256,
            observation=observation,
            binding_observation=binding,
            factorized_decision=inner,
        )
        self.authenticate()
        return result


def _freeze_memory(
    *,
    factorized_memory: factors.FrozenFactorizedFeedbackHypothesisMemoryV1,
    provenance: tuple[UnorderedCueFactorOutcomeV1, ...],
) -> FrozenUnorderedCueFactorMemoryV1:
    checkpoint = _checkpoint_sha256(
        factorized_checkpoint_sha256=factorized_memory.checkpoint_sha256,
        provenance=provenance,
    )
    return FrozenUnorderedCueFactorMemoryV1(
        factorized_memory=factorized_memory,
        provenance=provenance,
        checkpoint_sha256=checkpoint,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def start_unordered_cue_factor_memory_v1(
    hypotheses: tuple[factors.FactorizedHypothesisV1, ...],
) -> FrozenUnorderedCueFactorMemoryV1:
    inner = factors.start_factorized_feedback_hypothesis_memory_v1(hypotheses)
    return _freeze_memory(factorized_memory=inner, provenance=())


def advance_unordered_cue_factor_memory_v1(
    parent: FrozenUnorderedCueFactorMemoryV1,
    outcome: UnorderedCueFactorOutcomeV1,
    *,
    plasticity_gate: bool = True,
    consequence_gate: bool = True,
) -> FrozenUnorderedCueFactorMemoryV1:
    if type(parent) is not FrozenUnorderedCueFactorMemoryV1:
        raise TypeError("advance requires one exact unordered cue factor memory")
    if type(outcome) is not UnorderedCueFactorOutcomeV1:
        raise TypeError("advance requires one exact unordered cue factor outcome")
    if type(plasticity_gate) is not bool or type(consequence_gate) is not bool:
        raise TypeError("advance gates must be exact booleans")
    parent.authenticate()
    outcome.authenticate()
    decision = outcome.decision
    expected = parent.route(
        decision.observation,
        cell_gate=decision.cell_gate,
        outcome_gate=decision.outcome_gate,
        axis_gates=decision.axis_gates,
        exploration_hypothesis_sha256=decision.exploration_hypothesis_sha256,
    )
    if expected != decision:
        raise UnorderedCueFactorMemoryError(
            "unordered cue factor outcome decision is not from this parent"
        )
    if not plasticity_gate or not consequence_gate:
        return parent
    if len(parent._provenance) >= MAX_PROVENANCE_ROWS:
        raise UnorderedCueFactorMemoryError("unordered cue factor provenance capacity exceeded")
    parent_snapshot = (
        parent._checkpoint_sha256,
        parent._factorized_memory.checkpoint_sha256,
        parent._provenance,
    )
    inner = factors.advance_factorized_feedback_hypothesis_memory_v1(
        parent._factorized_memory,
        outcome.factorized_outcome,
    )
    successor = _freeze_memory(
        factorized_memory=inner,
        provenance=(*parent._provenance, outcome),
    )
    parent.authenticate()
    if parent_snapshot != (
        parent._checkpoint_sha256,
        parent._factorized_memory.checkpoint_sha256,
        parent._provenance,
    ):
        raise RuntimeError("unordered cue factor advance mutated its parent")
    return successor
