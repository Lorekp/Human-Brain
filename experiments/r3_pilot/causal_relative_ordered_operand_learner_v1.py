"""Learn one relative ordered-operand decision from physical consequences.

The banked relative-literal checkpoint supplies one immutable repeated-payload
binding.  This successor observes only the temporal order of the two repeated
occurrences and the residual singleton: ``R,R,S`` or ``S,R,R``.  A two-action
cell race learns whether the repeated payload should occupy ordered operand
zero or one.  Only packed PASS/FAIL outcome cells reach the 2x2 plastic state.

The comparison/count/addition scaffold used for the intervention is explicit
self-posed scaffolding.  The learned state contains no payload, token position,
span assembly, prompt root, operator, program, or family identity.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import NamedTuple
from weakref import WeakSet

import numpy as np
from experiments.r3_pilot import causal_relative_literal_role_learner_v1 as literal_role
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v4 as breadth_v4
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.resting_glif_race import CompiledRestingGlifRace
from humanbrain.typed_program_types import ProgramType, TypedSignature

SCHEMA = "CAUSAL_RELATIVE_ORDERED_OPERAND_LEARNER_V1"
READY = "CAUSAL_RELATIVE_ORDERED_OPERAND_LEARNER_V1_READY"
STOP_DEPENDENCY = "STOP_CAUSAL_RELATIVE_ORDERED_OPERAND_LEARNER_V1_DEPENDENCY"
STOP_ORDER_GATE = "STOP_CAUSAL_RELATIVE_ORDERED_OPERAND_LEARNER_V1_ORDER_GATE"
STOP_TIE = "STOP_CAUSAL_RELATIVE_ORDERED_OPERAND_LEARNER_V1_TIE"
STOP_STRADDLED = "STOP_CAUSAL_RELATIVE_ORDERED_OPERAND_LEARNER_V1_STRADDLED"
STOP_INCOMPLETE = "STOP_CAUSAL_RELATIVE_ORDERED_OPERAND_LEARNER_V1_INCOMPLETE"
STOP_CONFLICT = "STOP_CAUSAL_RELATIVE_ORDERED_OPERAND_LEARNER_V1_CONFLICT"
STOP_ALIAS = "STOP_CAUSAL_RELATIVE_ORDERED_OPERAND_LEARNER_V1_ACTION_ALIAS"
STOP_ZERO = "STOP_CAUSAL_RELATIVE_ORDERED_OPERAND_LEARNER_V1_ZERO"
STOP_AMBIGUOUS = "STOP_CAUSAL_RELATIVE_ORDERED_OPERAND_LEARNER_V1_AMBIGUOUS"
STOP_UNSUPPORTED = "STOP_CAUSAL_RELATIVE_ORDERED_OPERAND_LEARNER_V1_UNSUPPORTED"
STOP_WORK_BOUND = "STOP_CAUSAL_RELATIVE_ORDERED_OPERAND_LEARNER_V1_WORK_BOUND"

CLAIM_BOUNDARY = (
    "self-posed consequence-trained transfer of one anonymous physical group-"
    "precedence relation into one generic ordered operand-zero/operand-one "
    "decision across unseen typed-Int payloads, span assemblies, and absolute "
    "positions; the two relations, two actions, comparison/count/add scaffold, "
    "typed Int decoder, and prompt token clock are preallocated, so parsing, "
    "operator discovery, arbitrary tree induction, and coding capability remain "
    "unproven"
)

REPEATED_BEFORE_SINGLETON = 0
SINGLETON_BEFORE_REPEATED = 1
ORDER_RELATION_CELL_COUNT = 2

REPEATED_TO_OPERAND_ZERO = 0
REPEATED_TO_OPERAND_ONE = 1
OPERAND_ACTION_CELL_COUNT = 2

EVENT_GATE_CELL_COUNT = 2
FIRST_GROUP_LATCH_CELL_COUNT = 2
STRADDLE_POISON_CELL_COUNT = 1
COMPLETE_CELL_COUNT = 1
SELECTION_CELL_COUNT = 1
TONIC_STOP_CELL_COUNT = 1
OUTCOME_CELL_COUNT = 2

EVENT_GATE_CELL_OFFSET = literal_role.TOTAL_CELL_COUNT
FIRST_GROUP_LATCH_CELL_OFFSET = EVENT_GATE_CELL_OFFSET + EVENT_GATE_CELL_COUNT
ORDER_RELATION_CELL_OFFSET = (
    FIRST_GROUP_LATCH_CELL_OFFSET + FIRST_GROUP_LATCH_CELL_COUNT
)
STRADDLE_POISON_CELL = ORDER_RELATION_CELL_OFFSET + ORDER_RELATION_CELL_COUNT
COMPLETE_CELL = STRADDLE_POISON_CELL + STRADDLE_POISON_CELL_COUNT
OPERAND_ACTION_CELL_OFFSET = COMPLETE_CELL + COMPLETE_CELL_COUNT
SELECTION_CELL = OPERAND_ACTION_CELL_OFFSET + OPERAND_ACTION_CELL_COUNT
TONIC_STOP_CELL = SELECTION_CELL + SELECTION_CELL_COUNT
PASS_OUTCOME_CELL = TONIC_STOP_CELL + TONIC_STOP_CELL_COUNT
FAIL_OUTCOME_CELL = PASS_OUTCOME_CELL + 1
NEW_CELL_COUNT = (
    EVENT_GATE_CELL_COUNT
    + FIRST_GROUP_LATCH_CELL_COUNT
    + ORDER_RELATION_CELL_COUNT
    + STRADDLE_POISON_CELL_COUNT
    + COMPLETE_CELL_COUNT
    + OPERAND_ACTION_CELL_COUNT
    + SELECTION_CELL_COUNT
    + TONIC_STOP_CELL_COUNT
    + OUTCOME_CELL_COUNT
)
TOTAL_CELL_COUNT = literal_role.TOTAL_CELL_COUNT + NEW_CELL_COUNT
if FAIL_OUTCOME_CELL != TOTAL_CELL_COUNT - 1:
    raise RuntimeError("ordered operand cell allocation changed")

TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = 4
TRAINABLE_STATE_BYTES = (
    2
    * ORDER_RELATION_CELL_COUNT
    * OPERAND_ACTION_CELL_COUNT
    * TICK_DTYPE.itemsize
)
MAX_LITERAL_OCCURRENCES = 3
MIN_CAUSAL_EPISODES = 2
MAX_CAUSAL_EPISODES = 4
MAX_PACKED_BUILDS = 1
MAX_PACKED_AUTHORIZATIONS = 8
MAX_PACKED_OUTCOME_RACES = 4
FIXED_MAX_COST = 8
ORDER_EVENT_CELL_STEPS = MAX_LITERAL_OCCURRENCES * (
    EVENT_GATE_CELL_COUNT
    + FIRST_GROUP_LATCH_CELL_COUNT
    + ORDER_RELATION_CELL_COUNT
    + STRADDLE_POISON_CELL_COUNT
    + COMPLETE_CELL_COUNT
)
ACTION_RACE_STEPS = 30
ACTION_RACE_CELL_STEPS = OPERAND_ACTION_CELL_COUNT * ACTION_RACE_STEPS
OUTPUT_CURRENT_PA = 10_000.0
_ACTION_RACE = CompiledRestingGlifRace.compile(step_count=ACTION_RACE_STEPS)

_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)
_LITERAL_INT = ("literal", "Int")
_ROLE_INT = ("role", "Int")
_ROLE_LIST_INT = ("role", "List[Int]")
_FIXED_TEMPLATE_LITERAL = -15
_LITERAL_KEYS = {"canonical_token_index", "typed_output_active_cells"}

_NAMESPACE = b"causal-relative-ordered-operand-learner-v1\0"
_CHECKPOINT_NAMESPACE = _NAMESPACE + b"checkpoint\0"
_BINDING_NAMESPACE = _NAMESPACE + b"binding\0"
_WORK_NAMESPACE = _NAMESPACE + b"work\0"
_CONSTRUCTION_TOKEN = object()
_BINDING_AUTHORITY = object()


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _array_frame(name: str, value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    return b"".join(
        (
            name.encode("ascii"),
            b"\0",
            array.dtype.str.encode("ascii"),
            b"\0",
            _canonical_json(array.shape),
            b"\0",
            array.tobytes(order="C"),
        )
    )


def _signature_ok(signature: TypedSignature) -> bool:
    return (
        type(signature) is TypedSignature
        and signature.result == _INT
        and len(signature.parameters) == 1
        and signature.parameters[0].type == _LIST_INT
    )


@dataclass(frozen=True, slots=True)
class CausalOrderedOperandEpisode:
    span_records: tuple[Mapping[str, object], ...]
    literal_records: tuple[Mapping[str, object], ...]
    signature_role_record: Mapping[str, object]
    desired: tuple[int, ...]


class _LiteralOccurrence(NamedTuple):
    token_index: int
    payload_cells: tuple[int, ...]


class _OrderObservation(NamedTuple):
    status: str
    reason: str | None
    relation: int | None
    repeated_payload_cells: tuple[int, ...]
    repeated_token_indices: tuple[int, ...]
    singleton_token_index: int | None
    literal_binding_sha256: str
    prompt_sha256: str


class PackedOrderedOperandAudit(NamedTuple):
    episode_ordinal: int
    relation: int
    pass_action: int
    authorized_action_count: int
    outcome_race_sha256: str
    pass_outcome_count: int


class OrderedOperandTrainingSummary(NamedTuple):
    episode_count: int
    literal_role_checkpoint_sha256: str
    packed_build_count: int
    packed_authorization_count: int
    packed_outcome_race_count: int
    order_event_count: int
    order_event_cell_steps: int
    action_race_cell_steps: int
    outcome_gate_on: bool
    excitatory_update_ticks: int
    inhibitory_update_ticks: int
    maximum_excitatory_tick: int
    maximum_inhibitory_tick: int
    trainable_state_bytes: int
    packed_work_sha256: str
    intervention_audits: tuple[PackedOrderedOperandAudit, ...]
    checkpoint_sha256: str


class OrderedOperandBindingWork(NamedTuple):
    literal_binding_work_sha256: str
    order_event_count: int
    order_event_cell_steps: int
    action_candidate_count: int
    action_race_cell_steps: int
    sha256: str


class _BindingReceipt(NamedTuple):
    authority: object
    status: str
    stop_reason: str | None
    checkpoint_sha256: str
    literal_role_checkpoint_sha256: str
    literal_binding_sha256: str
    prompt_sha256: str
    repeated_payload_cells: tuple[int, ...]
    repeated_token_indices: tuple[int, ...]
    singleton_token_index: int | None
    relation: int | None
    selected_action: int | None
    work: OrderedOperandBindingWork
    sha256: str


class RelativeOrderedOperandCertificate:
    """Weakly issued exact-object result for one target-free action recall."""

    __slots__ = ("__weakref__", "_receipt")

    def __init__(self, receipt: _BindingReceipt) -> None:
        if receipt.authority is not _BINDING_AUTHORITY:
            raise TypeError("ordered operand certificate requires private authority")
        self._receipt = receipt

    @property
    def status(self) -> str:
        return self._receipt.status

    @property
    def stop_reason(self) -> str | None:
        return self._receipt.stop_reason

    @property
    def sha256(self) -> str:
        return self._receipt.sha256

    @property
    def selected_action(self) -> int | None:
        return self._receipt.selected_action

    @property
    def work(self) -> OrderedOperandBindingWork:
        return self._receipt.work


class RelativeOrderedOperandBinding(NamedTuple):
    repeated_payload_cells: tuple[int, ...]
    repeated_token_indices: tuple[int, ...]
    singleton_token_index: int
    relation: int
    selected_action: int
    prompt_sha256: str
    literal_binding_sha256: str
    certificate_sha256: str


_ISSUED_BINDINGS: WeakSet[RelativeOrderedOperandCertificate] = WeakSet()


class OrderedOperandFitStop(RuntimeError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _literal_occurrences(
    literal_records: Sequence[Mapping[str, object]],
) -> tuple[_LiteralOccurrence, ...]:
    if isinstance(literal_records, (str, bytes)) or not isinstance(
        literal_records, Sequence
    ):
        raise TypeError("literal records must be a sequence")
    occurrences: list[_LiteralOccurrence] = []
    for row in literal_records:
        if not isinstance(row, Mapping) or set(row) != _LITERAL_KEYS:
            raise ValueError("typed literal record schema changed")
        token_index = row["canonical_token_index"]
        cells = row["typed_output_active_cells"]
        if type(token_index) is not int or isinstance(cells, (str, bytes)) or not isinstance(
            cells, Sequence
        ):
            raise ValueError("typed literal occurrence changed")
        payload = tuple(cells)
        if any(type(cell) is not int for cell in payload):
            raise ValueError("typed literal cells changed")
        occurrences.append(_LiteralOccurrence(token_index, payload))
    if len(occurrences) != MAX_LITERAL_OCCURRENCES:
        raise ValueError("ordered operand slice requires exactly three occurrences")
    return tuple(sorted(occurrences, key=lambda row: (row.token_index, row.payload_cells)))


def _observe_order(
    checkpoint: literal_role.FrozenCausalRelativeLiteralRoleV1,
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
    *,
    equality_gate: bool,
    literal_role_gate: bool,
    order_gate: bool,
) -> _OrderObservation:
    before = checkpoint.checkpoint_sha256
    literal_certificate = checkpoint.bind(
        span_records,
        literal_records,
        signature_role_record,
        equality_gate=equality_gate,
        cell_gate=literal_role_gate,
    )
    if checkpoint.checkpoint_sha256 != before:
        raise RuntimeError("banked relative-literal checkpoint mutated during inference")
    if literal_certificate.status != literal_role.READY:
        return _OrderObservation(
            STOP_DEPENDENCY,
            literal_certificate.stop_reason or literal_certificate.status,
            None,
            (),
            (),
            None,
            literal_certificate.sha256,
            "",
        )
    binding = literal_role.require_relative_literal_binding(literal_certificate)
    occurrences = _literal_occurrences(literal_records)
    positions = tuple(row.token_index for row in occurrences)
    if len(set(positions)) != len(positions):
        return _OrderObservation(
            STOP_TIE,
            "two physical literal occurrences share one token time",
            None,
            binding.payload_cells,
            binding.member_token_indices,
            None,
            literal_certificate.sha256,
            binding.prompt_sha256,
        )
    if not order_gate:
        return _OrderObservation(
            STOP_ORDER_GATE,
            "physical temporal-order cells were disabled",
            None,
            binding.payload_cells,
            binding.member_token_indices,
            None,
            literal_certificate.sha256,
            binding.prompt_sha256,
        )
    tags = tuple(row.payload_cells == binding.payload_cells for row in occurrences)
    singleton_rows = tuple(
        row
        for row, repeated in zip(occurrences, tags, strict=True)
        if not repeated
    )
    if tags == (True, True, False) and len(singleton_rows) == 1:
        relation = REPEATED_BEFORE_SINGLETON
    elif tags == (False, True, True) and len(singleton_rows) == 1:
        relation = SINGLETON_BEFORE_REPEATED
    elif tags == (True, False, True):
        return _OrderObservation(
            STOP_STRADDLED,
            "singleton occurrence lies between repeated occurrences",
            None,
            binding.payload_cells,
            binding.member_token_indices,
            singleton_rows[0].token_index,
            literal_certificate.sha256,
            binding.prompt_sha256,
        )
    else:
        return _OrderObservation(
            STOP_UNSUPPORTED,
            "literal-role geometry changed outside the fixed two-plus-one capacity",
            None,
            binding.payload_cells,
            binding.member_token_indices,
            None,
            literal_certificate.sha256,
            binding.prompt_sha256,
        )
    return _OrderObservation(
        READY,
        None,
        relation,
        binding.payload_cells,
        binding.member_token_indices,
        singleton_rows[0].token_index,
        literal_certificate.sha256,
        binding.prompt_sha256,
    )


def freeze_ordered_operand_scaffold(
    hole_value: int,
    selected_action: int,
) -> plan_slice.StructuralFactorCertificate:
    """Instantiate one of two explicit ordered comparison scaffolds."""

    if type(selected_action) is not int or selected_action not in {
        REPEATED_TO_OPERAND_ZERO,
        REPEATED_TO_OPERAND_ONE,
    }:
        raise ValueError("ordered operand action lies outside the fixed capacity")
    if type(hole_value) is not int or not (
        output_cells.INT_MIN <= hole_value <= output_cells.INT_MAX
    ):
        raise ValueError("literal-hole value lies outside the physical Int bank")
    if selected_action == REPEATED_TO_OPERAND_ZERO:
        threshold_position = 0
        threshold_occupancy = (0,)
        bound_position = 1
        bound_occupancy = (1, 2)
    else:
        threshold_position = 1
        threshold_occupancy = (1,)
        bound_position = 0
        bound_occupancy = (0, 2)
    occurrences = (
        plan_slice.StructuralOccurrence(
            None,
            None,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT, _ROLE_LIST_INT),
            (),
        ),
        plan_slice.StructuralOccurrence(
            0,
            0,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT),
            (0, 1, 2),
        ),
        plan_slice.StructuralOccurrence(
            1,
            0,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT),
            (0, 1, 2),
        ),
        plan_slice.StructuralOccurrence(
            2,
            threshold_position,
            (_LITERAL_INT,),
            threshold_occupancy,
        ),
        plan_slice.StructuralOccurrence(
            2,
            bound_position,
            (_LITERAL_INT, _ROLE_INT),
            bound_occupancy,
            (plan_slice.StructuralRelationConstraint(0, "ne", right_value=0),),
        ),
        plan_slice.StructuralOccurrence(4, 0, (_ROLE_INT,), (1,)),
        plan_slice.StructuralOccurrence(4, 1, (_LITERAL_INT,), (0,)),
        plan_slice.StructuralOccurrence(0, 1, (_ROLE_LIST_INT,), (3,)),
    )
    certificate = plan_slice.freeze_structural_factor_certificate(
        occurrences,
        (
            plan_slice.StructuralDomainConstraint(3, 0, (hole_value,)),
            plan_slice.StructuralDomainConstraint(6, 0, (_FIXED_TEMPLATE_LITERAL,)),
        ),
        (((3, 0),), ((6, 0),)),
    )
    if (
        selected_action == REPEATED_TO_OPERAND_ZERO
        and certificate.sha256
        != literal_role.freeze_literal_hole_scaffold(hole_value).sha256
    ):
        raise RuntimeError("banked operand-zero comparison scaffold changed")
    return certificate


def _require_distinct_action_outputs(
    outputs: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
) -> None:
    if len(outputs) != OPERAND_ACTION_CELL_COUNT or outputs[0] == outputs[1]:
        raise OrderedOperandFitStop(
            STOP_ALIAS,
            "the two preallocated operand actions have identical consequences",
        )


def _authorize_actions(
    frontier: breadth_v4.PackedGenericTypedTrainBreadthV4,
    payload_cells: tuple[int, ...],
) -> tuple[tuple[tuple[output_cells.TypedOutputEncoding, ...], ...], int]:
    hole_value = literal_role.decode_int_payload(payload_cells)
    outputs: list[tuple[output_cells.TypedOutputEncoding, ...]] = []
    authorizations = 0
    for action in range(OPERAND_ACTION_CELL_COUNT):
        certificate = freeze_ordered_operand_scaffold(hole_value, action)
        try:
            (
                status,
                count,
                receipt,
                _root_receipt,
                _build_work,
                _authorization_work,
            ) = plan_slice._bind_certificate(frontier, certificate)
        except packed.PackedRelationWorkLimitExceeded as error:
            raise OrderedOperandFitStop(
                STOP_WORK_BOUND,
                "ordered operand authorization exceeded packed relation work",
            ) from error
        if status != plan_slice.FRONTIER_READY or count != 1 or receipt is None:
            raise OrderedOperandFitStop(
                STOP_UNSUPPORTED,
                "one ordered operand action has no unique packed plan slice "
                f"(status={status}, authorized={count})",
            )
        binding = plan_slice._authenticate_plan_slice_receipt(frontier, receipt)
        outputs.append(tuple(binding.candidate.eclass.physical_outputs))
        authorizations += 1
    frozen = tuple(outputs)
    _require_distinct_action_outputs(frozen)
    return frozen, authorizations


def _race_outcomes(
    outputs: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
    desired: tuple[int, ...],
    *,
    episode_ordinal: int,
    relation: int,
) -> PackedOrderedOperandAudit:
    keys = tuple(f"episode-{episode_ordinal}-action-{action}" for action in range(2))
    encoded_desired = tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.INT,
        )
        for value in desired
    )
    race = semantic_race.select_exact_semantic_vector(keys, outputs, encoded_desired)
    if race.status != semantic_race.SELECTED or race.winner_key is None:
        raise OrderedOperandFitStop(
            STOP_AMBIGUOUS,
            "ordered operand interventions have zero or multiple PASS outcomes",
        )
    pass_action = keys.index(race.winner_key)
    return PackedOrderedOperandAudit(
        episode_ordinal,
        relation,
        pass_action,
        OPERAND_ACTION_CELL_COUNT,
        race.sha256,
        1,
    )


def _learn_policy(
    audits: Sequence[PackedOrderedOperandAudit],
    *,
    outcome_gate: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    labels: list[list[int]] = [[], []]
    for audit in audits:
        labels[audit.relation].append(audit.pass_action)
    if any(values and any(value != values[0] for value in values) for values in labels):
        raise OrderedOperandFitStop(
            STOP_CONFLICT,
            "one physical order relation has conflicting PASS actions",
        )
    if any(not values for values in labels) or {values[0] for values in labels} != {0, 1}:
        raise OrderedOperandFitStop(
            STOP_INCOMPLETE,
            "both order relations and both operand actions need causal support",
        )
    excitatory = np.zeros(
        (ORDER_RELATION_CELL_COUNT, OPERAND_ACTION_CELL_COUNT),
        dtype=TICK_DTYPE,
    )
    inhibitory = np.zeros_like(excitatory)
    excitatory_ticks = 0
    inhibitory_ticks = 0
    if outcome_gate:
        for audit in audits:
            for action in range(OPERAND_ACTION_CELL_COUNT):
                if action == audit.pass_action:
                    before = int(excitatory[audit.relation, action])
                    excitatory[audit.relation, action] = min(TICK_CAP, before + 1)
                    excitatory_ticks += int(excitatory[audit.relation, action]) - before
                else:
                    before = int(inhibitory[audit.relation, action])
                    inhibitory[audit.relation, action] = min(TICK_CAP, before + 1)
                    inhibitory_ticks += int(inhibitory[audit.relation, action]) - before
    return excitatory, inhibitory, excitatory_ticks, inhibitory_ticks


def _checkpoint_sha(
    literal_role_checkpoint_sha256: str,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> str:
    return sha256(
        _CHECKPOINT_NAMESPACE
        + literal_role_checkpoint_sha256.encode("ascii")
        + b"\0"
        + _array_frame("ordered-operand-excitatory", excitatory)
        + _array_frame("ordered-operand-inhibitory", inhibitory)
    ).hexdigest()


class FrozenCausalRelativeOrderedOperandV1:
    """Immutable 2x2 policy over relative order and operand position."""

    __slots__ = (
        "_checkpoint_sha256",
        "_excitatory",
        "_inhibitory",
        "_literal_role_checkpoint",
        "_summary",
    )

    def __init__(
        self,
        literal_role_checkpoint: literal_role.FrozenCausalRelativeLiteralRoleV1,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        summary: OrderedOperandTrainingSummary,
        *,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("ordered operand checkpoints are builder-owned")
        if type(literal_role_checkpoint) is not literal_role.FrozenCausalRelativeLiteralRoleV1:
            raise TypeError("ordered operand checkpoint requires the banked dependency type")
        literal_role_checkpoint.authenticate()
        expected_shape = (ORDER_RELATION_CELL_COUNT, OPERAND_ACTION_CELL_COUNT)
        if (
            excitatory.shape != expected_shape
            or inhibitory.shape != expected_shape
            or excitatory.dtype != TICK_DTYPE
            or inhibitory.dtype != TICK_DTYPE
            or np.any(excitatory > TICK_CAP)
            or np.any(inhibitory > TICK_CAP)
        ):
            raise ValueError("ordered operand checkpoint geometry changed")
        self._literal_role_checkpoint = literal_role_checkpoint
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._checkpoint_sha256 = _checkpoint_sha(
            literal_role_checkpoint.checkpoint_sha256,
            self._excitatory,
            self._inhibitory,
        )
        self._summary = summary
        if summary.checkpoint_sha256 != self._checkpoint_sha256:
            raise RuntimeError("ordered operand training summary changed")
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def literal_role_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._literal_role_checkpoint.checkpoint_sha256

    @property
    def training_summary(self) -> OrderedOperandTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def learned_state_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._excitatory.shape, self._inhibitory.shape

    def authenticate(self) -> None:
        self._literal_role_checkpoint.authenticate()
        if (
            self._excitatory.flags.writeable
            or self._inhibitory.flags.writeable
            or _checkpoint_sha(
                self._literal_role_checkpoint.checkpoint_sha256,
                self._excitatory,
                self._inhibitory,
            )
            != self._checkpoint_sha256
        ):
            raise RuntimeError("ordered operand checkpoint authentication failed")

    def bind(
        self,
        span_records: Sequence[Mapping[str, object]],
        literal_records: Sequence[Mapping[str, object]],
        signature_role_record: Mapping[str, object],
        *,
        equality_gate: bool = True,
        literal_role_gate: bool = True,
        order_gate: bool = True,
        cell_gate: bool = True,
    ) -> RelativeOrderedOperandCertificate:
        self.authenticate()
        if any(
            type(value) is not bool
            for value in (equality_gate, literal_role_gate, order_gate, cell_gate)
        ):
            raise TypeError("ordered operand gates must be exact booleans")
        observation = _observe_order(
            self._literal_role_checkpoint,
            span_records,
            literal_records,
            signature_role_record,
            equality_gate=equality_gate,
            literal_role_gate=literal_role_gate,
            order_gate=order_gate,
        )
        if observation.status != READY or observation.relation is None:
            return _issue_binding(self, observation, None, observation.status, observation.reason)
        relation = observation.relation
        scores = self._excitatory[relation].astype(np.int16) - self._inhibitory[
            relation
        ].astype(np.int16)
        currents = np.asarray(
            [max(0, int(score)) * OUTPUT_CURRENT_PA if cell_gate else 0.0 for score in scores],
            dtype=np.float64,
        )
        spikes = tuple(
            int(index)
            for index in np.flatnonzero(_ACTION_RACE.accumulated_spike_mask(currents))
        )
        if not spikes:
            return _issue_binding(
                self,
                observation,
                None,
                STOP_ZERO,
                "no learned operand action crossed the physical race",
            )
        if len(spikes) != 1:
            return _issue_binding(
                self,
                observation,
                None,
                STOP_AMBIGUOUS,
                "multiple learned operand actions crossed together",
            )
        return _issue_binding(self, observation, spikes[0], READY, None)


def _work_payload(value: OrderedOperandBindingWork) -> dict[str, object]:
    return {field: getattr(value, field) for field in value._fields if field != "sha256"}


def _receipt_payload(value: _BindingReceipt) -> dict[str, object]:
    return {
        field: getattr(value, field)
        for field in value._fields
        if field not in {"authority", "work", "sha256"}
    } | {"work_sha256": value.work.sha256}


def _issue_binding(
    checkpoint: FrozenCausalRelativeOrderedOperandV1,
    observation: _OrderObservation,
    selected_action: int | None,
    status: str,
    stop_reason: str | None,
) -> RelativeOrderedOperandCertificate:
    work_partial = OrderedOperandBindingWork(
        observation.literal_binding_sha256,
        MAX_LITERAL_OCCURRENCES,
        ORDER_EVENT_CELL_STEPS,
        OPERAND_ACTION_CELL_COUNT if observation.relation is not None else 0,
        ACTION_RACE_CELL_STEPS if observation.relation is not None else 0,
        "",
    )
    work = work_partial._replace(
        sha256=_digest(_WORK_NAMESPACE, _work_payload(work_partial))
    )
    partial = _BindingReceipt(
        _BINDING_AUTHORITY,
        status,
        stop_reason,
        checkpoint.checkpoint_sha256,
        checkpoint.literal_role_checkpoint_sha256,
        observation.literal_binding_sha256,
        observation.prompt_sha256,
        observation.repeated_payload_cells,
        observation.repeated_token_indices,
        observation.singleton_token_index,
        observation.relation,
        selected_action,
        work,
        "",
    )
    receipt = partial._replace(
        sha256=_digest(_BINDING_NAMESPACE, _receipt_payload(partial))
    )
    certificate = RelativeOrderedOperandCertificate(receipt)
    _ISSUED_BINDINGS.add(certificate)
    return certificate


def _authenticate_binding(certificate: RelativeOrderedOperandCertificate) -> _BindingReceipt:
    if (
        type(certificate) is not RelativeOrderedOperandCertificate
        or certificate not in _ISSUED_BINDINGS
        or certificate._receipt.authority is not _BINDING_AUTHORITY
    ):
        raise TypeError("ordered operand binding is not an issued exact object")
    receipt = certificate._receipt
    if (
        receipt.work.sha256 != _digest(_WORK_NAMESPACE, _work_payload(receipt.work))
        or receipt.sha256 != _digest(_BINDING_NAMESPACE, _receipt_payload(receipt))
    ):
        raise RuntimeError("ordered operand binding certificate changed")
    return receipt


def require_relative_ordered_operand_binding(
    certificate: RelativeOrderedOperandCertificate,
) -> RelativeOrderedOperandBinding:
    receipt = _authenticate_binding(certificate)
    if (
        receipt.status != READY
        or receipt.relation is None
        or receipt.selected_action is None
        or receipt.singleton_token_index is None
    ):
        raise RuntimeError(receipt.stop_reason or receipt.status)
    return RelativeOrderedOperandBinding(
        receipt.repeated_payload_cells,
        receipt.repeated_token_indices,
        receipt.singleton_token_index,
        receipt.relation,
        receipt.selected_action,
        receipt.prompt_sha256,
        receipt.literal_binding_sha256,
        receipt.sha256,
    )


def fit_causal_relative_ordered_operand_v1(
    literal_role_checkpoint: literal_role.FrozenCausalRelativeLiteralRoleV1,
    episodes: tuple[CausalOrderedOperandEpisode, ...],
    *,
    signature: TypedSignature,
    cases: tuple[tuple[object, ...], ...],
    literals: tuple[int, ...],
    max_cost: int,
    outcome_gate: bool = True,
) -> FrozenCausalRelativeOrderedOperandV1:
    """Fit the constant order-to-operand policy from packed PASS/FAIL cells."""

    if type(literal_role_checkpoint) is not literal_role.FrozenCausalRelativeLiteralRoleV1:
        raise TypeError("ordered operand fit requires the exact banked checkpoint type")
    literal_role_checkpoint.authenticate()
    dependency_sha = literal_role_checkpoint.checkpoint_sha256
    if type(episodes) is not tuple or any(
        type(episode) is not CausalOrderedOperandEpisode for episode in episodes
    ):
        raise TypeError("episodes must be an exact tuple of ordered operand episodes")
    if not MIN_CAUSAL_EPISODES <= len(episodes) <= MAX_CAUSAL_EPISODES:
        raise OrderedOperandFitStop(
            STOP_INCOMPLETE,
            "ordered operand learning requires two to four causal episodes",
        )
    if not _signature_ok(signature) or max_cost != FIXED_MAX_COST:
        raise ValueError("ordered operand scaffold requires List[Int] -> Int at cost 8")
    if type(outcome_gate) is not bool:
        raise TypeError("outcome_gate must be an exact boolean")
    frozen_cases = tuple(tuple(row) for row in cases)
    frozen_literals = tuple(literals)
    if (
        not frozen_cases
        or _FIXED_TEMPLATE_LITERAL not in frozen_literals
        or 0 not in frozen_literals
    ):
        raise ValueError("ordinary packed interventions require cases, -15, and zero")

    observations: list[_OrderObservation] = []
    for episode in episodes:
        if len(episode.desired) != len(frozen_cases) or any(
            type(value) is not int for value in episode.desired
        ):
            raise ValueError("episode desired outputs do not align with ordinary cases")
        observation = _observe_order(
            literal_role_checkpoint,
            episode.span_records,
            episode.literal_records,
            episode.signature_role_record,
            equality_gate=True,
            literal_role_gate=True,
            order_gate=True,
        )
        if observation.status != READY or observation.relation is None:
            raise OrderedOperandFitStop(
                observation.status,
                observation.reason or "ordered operand observation stopped",
            )
        observations.append(observation)
    if {item.relation for item in observations} != {0, 1}:
        raise OrderedOperandFitStop(
            STOP_INCOMPLETE,
            "both physical order relations need support before intervention",
        )
    if len({item.repeated_payload_cells for item in observations}) < 2:
        raise OrderedOperandFitStop(
            STOP_INCOMPLETE,
            "causal support must vary repeated payload identity",
        )

    frontier = breadth_v4.PackedGenericTypedTrainBreadthV4.build(
        signature,
        frozen_cases,
        frozen_literals,
        max_cost,
    )
    if frontier.status != breadth_v4.FRONTIER_READY:
        raise OrderedOperandFitStop(
            STOP_WORK_BOUND,
            "ordinary packed ordered-operand frontier is not ready",
        )

    output_cache: dict[
        tuple[int, ...], tuple[tuple[tuple[output_cells.TypedOutputEncoding, ...], ...], int]
    ] = {}
    audits: list[PackedOrderedOperandAudit] = []
    authorization_count = 0
    for ordinal, (episode, observation) in enumerate(zip(episodes, observations, strict=True)):
        cached = output_cache.get(observation.repeated_payload_cells)
        if cached is None:
            cached = _authorize_actions(frontier, observation.repeated_payload_cells)
            output_cache[observation.repeated_payload_cells] = cached
            authorization_count += cached[1]
        audit = _race_outcomes(
            cached[0],
            tuple(episode.desired),
            episode_ordinal=ordinal,
            relation=observation.relation,
        )
        audits.append(audit)
    if (
        authorization_count > MAX_PACKED_AUTHORIZATIONS
        or len(audits) > MAX_PACKED_OUTCOME_RACES
    ):
        raise OrderedOperandFitStop(
            STOP_WORK_BOUND,
            "ordered operand intervention count exceeded the fixed work cap",
        )
    excitatory, inhibitory, excitatory_ticks, inhibitory_ticks = _learn_policy(
        audits,
        outcome_gate=outcome_gate,
    )
    if literal_role_checkpoint.checkpoint_sha256 != dependency_sha:
        raise RuntimeError("banked relative-literal checkpoint changed during fit")
    checkpoint_sha = _checkpoint_sha(dependency_sha, excitatory, inhibitory)
    summary = OrderedOperandTrainingSummary(
        len(episodes),
        dependency_sha,
        MAX_PACKED_BUILDS,
        authorization_count,
        len(audits),
        len(episodes) * MAX_LITERAL_OCCURRENCES,
        len(episodes) * ORDER_EVENT_CELL_STEPS,
        0,
        outcome_gate,
        excitatory_ticks,
        inhibitory_ticks,
        int(excitatory.max()),
        int(inhibitory.max()),
        TRAINABLE_STATE_BYTES,
        frontier.work.sha256,
        tuple(audits),
        checkpoint_sha,
    )
    return FrozenCausalRelativeOrderedOperandV1(
        literal_role_checkpoint,
        excitatory,
        inhibitory,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def freeze_outcome_off_control(
    checkpoint: FrozenCausalRelativeOrderedOperandV1,
) -> FrozenCausalRelativeOrderedOperandV1:
    """Silence the only PASS/FAIL-to-policy gate on the audited stream."""

    if type(checkpoint) is not FrozenCausalRelativeOrderedOperandV1:
        raise TypeError("outcome-off control requires the exact checkpoint type")
    checkpoint.authenticate()
    zeros = np.zeros(
        (ORDER_RELATION_CELL_COUNT, OPERAND_ACTION_CELL_COUNT),
        dtype=TICK_DTYPE,
    )
    checkpoint_sha = _checkpoint_sha(
        checkpoint.literal_role_checkpoint_sha256,
        zeros,
        zeros,
    )
    summary = checkpoint.training_summary._replace(
        outcome_gate_on=False,
        excitatory_update_ticks=0,
        inhibitory_update_ticks=0,
        maximum_excitatory_tick=0,
        maximum_inhibitory_tick=0,
        checkpoint_sha256=checkpoint_sha,
    )
    return FrozenCausalRelativeOrderedOperandV1(
        checkpoint._literal_role_checkpoint,
        zeros,
        zeros,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


CONFIG_RECORD = MappingProxyType(
    {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "literal_role_config_sha256": literal_role.CONFIG_SHA256,
        "base_cell_count": literal_role.TOTAL_CELL_COUNT,
        "new_cell_count": NEW_CELL_COUNT,
        "total_cell_count": TOTAL_CELL_COUNT,
        "trainable_state_bytes": TRAINABLE_STATE_BYTES,
        "order_event_cell_steps": ORDER_EVENT_CELL_STEPS,
        "action_race_cell_steps": ACTION_RACE_CELL_STEPS,
        "maximum_causal_episodes": MAX_CAUSAL_EPISODES,
        "maximum_packed_builds": MAX_PACKED_BUILDS,
        "maximum_packed_authorizations": MAX_PACKED_AUTHORIZATIONS,
        "maximum_packed_outcome_races": MAX_PACKED_OUTCOME_RACES,
        "fixed_max_cost": FIXED_MAX_COST,
    }
)
CONFIG_SHA256 = _digest(_NAMESPACE + b"config\0", CONFIG_RECORD)


__all__ = [
    "ACTION_RACE_CELL_STEPS",
    "CLAIM_BOUNDARY",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "FAIL_OUTCOME_CELL",
    "FIXED_MAX_COST",
    "MAX_PACKED_AUTHORIZATIONS",
    "MAX_PACKED_BUILDS",
    "MAX_PACKED_OUTCOME_RACES",
    "NEW_CELL_COUNT",
    "OPERAND_ACTION_CELL_COUNT",
    "ORDER_EVENT_CELL_STEPS",
    "ORDER_RELATION_CELL_COUNT",
    "PASS_OUTCOME_CELL",
    "READY",
    "REPEATED_BEFORE_SINGLETON",
    "REPEATED_TO_OPERAND_ONE",
    "REPEATED_TO_OPERAND_ZERO",
    "SCHEMA",
    "SINGLETON_BEFORE_REPEATED",
    "STOP_ALIAS",
    "STOP_AMBIGUOUS",
    "STOP_CONFLICT",
    "STOP_DEPENDENCY",
    "STOP_INCOMPLETE",
    "STOP_ORDER_GATE",
    "STOP_STRADDLED",
    "STOP_TIE",
    "STOP_UNSUPPORTED",
    "STOP_WORK_BOUND",
    "STOP_ZERO",
    "TOTAL_CELL_COUNT",
    "TRAINABLE_STATE_BYTES",
    "CausalOrderedOperandEpisode",
    "FrozenCausalRelativeOrderedOperandV1",
    "OrderedOperandBindingWork",
    "OrderedOperandFitStop",
    "OrderedOperandTrainingSummary",
    "PackedOrderedOperandAudit",
    "RelativeOrderedOperandBinding",
    "RelativeOrderedOperandCertificate",
    "fit_causal_relative_ordered_operand_v1",
    "freeze_ordered_operand_scaffold",
    "freeze_outcome_off_control",
    "require_relative_ordered_operand_binding",
]
