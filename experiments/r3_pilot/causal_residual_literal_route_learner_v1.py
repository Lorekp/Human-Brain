"""Learn one residual-literal route from packed physical consequences.

The banked ordered-operand checkpoint authenticates the repeated literal, the
single residual occurrence, and comparator orientation.  This successor learns
whether that residual payload should replace the explicit ``-15`` additive
literal in the fixed self-posed scaffold.  Only packed PASS/FAIL outcome cells
reach the two-action uint8 plastic state; literal values remain transient.
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
from experiments.r3_pilot import causal_relative_ordered_operand_learner_v1 as ordered
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v4 as breadth_v4
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.resting_glif_race import CompiledRestingGlifRace
from humanbrain.typed_program_types import ProgramType, TypedSignature

SCHEMA = "CAUSAL_RESIDUAL_LITERAL_ROUTE_LEARNER_V1"
READY = "CAUSAL_RESIDUAL_LITERAL_ROUTE_LEARNER_V1_READY"
STOP_DEPENDENCY = "STOP_CAUSAL_RESIDUAL_LITERAL_ROUTE_LEARNER_V1_DEPENDENCY"
STOP_RESIDUAL_GATE = "STOP_CAUSAL_RESIDUAL_LITERAL_ROUTE_LEARNER_V1_RESIDUAL_GATE"
STOP_RESIDUAL_AUTH = "STOP_CAUSAL_RESIDUAL_LITERAL_ROUTE_LEARNER_V1_RESIDUAL_AUTH"
STOP_INCOMPLETE = "STOP_CAUSAL_RESIDUAL_LITERAL_ROUTE_LEARNER_V1_INCOMPLETE"
STOP_CONFLICT = "STOP_CAUSAL_RESIDUAL_LITERAL_ROUTE_LEARNER_V1_CONFLICT"
STOP_ALIAS = "STOP_CAUSAL_RESIDUAL_LITERAL_ROUTE_LEARNER_V1_ACTION_ALIAS"
STOP_ZERO = "STOP_CAUSAL_RESIDUAL_LITERAL_ROUTE_LEARNER_V1_ZERO"
STOP_AMBIGUOUS = "STOP_CAUSAL_RESIDUAL_LITERAL_ROUTE_LEARNER_V1_AMBIGUOUS"
STOP_UNSUPPORTED = "STOP_CAUSAL_RESIDUAL_LITERAL_ROUTE_LEARNER_V1_UNSUPPORTED"
STOP_WORK_BOUND = "STOP_CAUSAL_RESIDUAL_LITERAL_ROUTE_LEARNER_V1_WORK_BOUND"

CLAIM_BOUNDARY = (
    "self-posed consequence-trained transfer of one authenticated residual "
    "typed-Int occurrence into one preallocated additive literal hole across "
    "unseen repeated/residual payloads, assemblies, positions, and both banked "
    "operand orientations; the count/comparison/addition tree, destination "
    "domain (occurrence 6, slot 0), typed Int decoder, and token clock remain "
    "explicit scaffolding, so "
    "operator discovery, parsing, arbitrary composition, and coding capability "
    "remain unproven"
)

RESIDUAL_TO_LITERAL_HOLE = 0
KEEP_DEFAULT_LITERAL = 1
ROUTE_ACTION_CELL_COUNT = 2
ROUTE_PHASE_CELL_COUNT = 1

ROUTE_PHASE_CELL = ordered.TOTAL_CELL_COUNT
ROUTE_ACTION_CELL_OFFSET = ROUTE_PHASE_CELL + ROUTE_PHASE_CELL_COUNT
NEW_CELL_COUNT = ROUTE_PHASE_CELL_COUNT + ROUTE_ACTION_CELL_COUNT
TOTAL_CELL_COUNT = ordered.TOTAL_CELL_COUNT + NEW_CELL_COUNT
if ROUTE_ACTION_CELL_OFFSET + ROUTE_ACTION_CELL_COUNT != TOTAL_CELL_COUNT:
    raise RuntimeError("residual route cell allocation changed")

REUSED_SELECTION_CELL = ordered.SELECTION_CELL
REUSED_TONIC_STOP_CELL = ordered.TONIC_STOP_CELL
REUSED_PASS_OUTCOME_CELL = ordered.PASS_OUTCOME_CELL
REUSED_FAIL_OUTCOME_CELL = ordered.FAIL_OUTCOME_CELL

TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = 4
TRAINABLE_STATE_BYTES = 2 * ROUTE_ACTION_CELL_COUNT * TICK_DTYPE.itemsize
MIN_CAUSAL_EPISODES = 2
MAX_CAUSAL_EPISODES = 4
MAX_PACKED_BUILDS = 1
MAX_PACKED_AUTHORIZATIONS = 8
MAX_PACKED_OUTCOME_RACES = 4
FIXED_MAX_COST = 8
DEFAULT_LITERAL = -15
ADDITIVE_LITERAL_OCCURRENCE = 6
ADDITIVE_LITERAL_SLOT = 0
ROUTE_PHASE_CELL_STEPS = 1
ROUTE_ACTION_RACE_STEPS = 30
ROUTE_ACTION_RACE_CELL_STEPS = (
    ROUTE_ACTION_CELL_COUNT * ROUTE_ACTION_RACE_STEPS
)
UPSTREAM_LITERAL_COMPARISON_COUNT = 6
UPSTREAM_BINDING_CELL_STEPS = (
    UPSTREAM_LITERAL_COMPARISON_COUNT
    * literal_role.EQUALITY_CELL_STEPS_PER_COMPARISON
    + literal_role.RELATION_CELL_COUNT * literal_role.OUTPUT_STEPS
    + ordered.ORDER_EVENT_CELL_STEPS
    + ordered.ACTION_RACE_CELL_STEPS
)
STACKED_BINDING_CELL_STEPS = (
    UPSTREAM_BINDING_CELL_STEPS
    + ROUTE_PHASE_CELL_STEPS
    + ROUTE_ACTION_RACE_CELL_STEPS
)
OUTPUT_CURRENT_PA = 10_000.0
_ROUTE_ACTION_RACE = CompiledRestingGlifRace.compile(
    step_count=ROUTE_ACTION_RACE_STEPS
)

_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)
_LITERAL_KEYS = {"canonical_token_index", "typed_output_active_cells"}

_NAMESPACE = b"causal-residual-literal-route-learner-v1\0"
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
class CausalResidualLiteralEpisode:
    span_records: tuple[Mapping[str, object], ...]
    literal_records: tuple[Mapping[str, object], ...]
    signature_role_record: Mapping[str, object]
    desired: tuple[int, ...]


class _ResidualObservation(NamedTuple):
    status: str
    reason: str | None
    relation: int | None
    operand_action: int | None
    repeated_payload_cells: tuple[int, ...]
    repeated_token_indices: tuple[int, ...]
    residual_payload_cells: tuple[int, ...]
    residual_token_index: int | None
    ordered_binding_sha256: str
    prompt_sha256: str


class PackedResidualRouteAudit(NamedTuple):
    episode_ordinal: int
    relation: int
    operand_action: int
    pass_route: int
    authorized_route_count: int
    outcome_race_sha256: str
    pass_outcome_count: int


class ResidualLiteralTrainingSummary(NamedTuple):
    episode_count: int
    ordered_checkpoint_sha256: str
    packed_build_count: int
    packed_authorization_count: int
    packed_outcome_race_count: int
    upstream_binding_cell_steps: int
    route_phase_cell_steps: int
    route_action_race_cell_steps: int
    outcome_gate_on: bool
    excitatory_update_ticks: int
    inhibitory_update_ticks: int
    maximum_excitatory_tick: int
    maximum_inhibitory_tick: int
    trainable_state_bytes: int
    packed_work_sha256: str
    intervention_audits: tuple[PackedResidualRouteAudit, ...]
    checkpoint_sha256: str


class ResidualLiteralBindingWork(NamedTuple):
    ordered_binding_sha256: str
    upstream_binding_cell_steps: int
    route_phase_cell_steps: int
    route_candidate_count: int
    route_action_race_cell_steps: int
    stacked_binding_cell_steps: int
    sha256: str


class _BindingReceipt(NamedTuple):
    authority: object
    status: str
    stop_reason: str | None
    checkpoint_sha256: str
    ordered_checkpoint_sha256: str
    ordered_binding_sha256: str
    prompt_sha256: str
    repeated_payload_cells: tuple[int, ...]
    repeated_token_indices: tuple[int, ...]
    residual_payload_cells: tuple[int, ...]
    residual_token_index: int | None
    relation: int | None
    operand_action: int | None
    selected_route: int | None
    work: ResidualLiteralBindingWork
    sha256: str


class ResidualLiteralRouteCertificate:
    """Weakly issued exact-object result for one residual route recall."""

    __slots__ = ("__weakref__", "_receipt")

    def __init__(self, receipt: _BindingReceipt) -> None:
        if receipt.authority is not _BINDING_AUTHORITY:
            raise TypeError("residual route certificate requires private authority")
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
    def selected_route(self) -> int | None:
        return self._receipt.selected_route

    @property
    def work(self) -> ResidualLiteralBindingWork:
        return self._receipt.work


class ResidualLiteralRouteBinding(NamedTuple):
    repeated_payload_cells: tuple[int, ...]
    repeated_token_indices: tuple[int, ...]
    residual_payload_cells: tuple[int, ...]
    residual_token_index: int
    relation: int
    operand_action: int
    selected_route: int
    prompt_sha256: str
    ordered_binding_sha256: str
    certificate_sha256: str


_ISSUED_BINDINGS: WeakSet[ResidualLiteralRouteCertificate] = WeakSet()


class ResidualLiteralFitStop(RuntimeError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _extract_residual_payload(
    binding: ordered.RelativeOrderedOperandBinding,
    literal_records: Sequence[Mapping[str, object]],
) -> tuple[int, ...]:
    if isinstance(literal_records, (str, bytes)) or not isinstance(
        literal_records, Sequence
    ):
        raise ValueError("literal records must be a physical occurrence sequence")
    occurrences: list[tuple[int, tuple[int, ...]]] = []
    for row in literal_records:
        if not isinstance(row, Mapping) or set(row) != _LITERAL_KEYS:
            raise ValueError("typed literal record schema changed")
        token_index = row["canonical_token_index"]
        cells = row["typed_output_active_cells"]
        if (
            type(token_index) is not int
            or isinstance(cells, (str, bytes))
            or not isinstance(cells, Sequence)
        ):
            raise ValueError("typed literal occurrence changed")
        payload = tuple(cells)
        if any(type(cell) is not int for cell in payload):
            raise ValueError("typed literal payload changed")
        literal_role.decode_int_payload(payload)
        occurrences.append((token_index, payload))
    if len(occurrences) != ordered.MAX_LITERAL_OCCURRENCES:
        raise ValueError("residual route requires exactly three literal occurrences")
    chronological = tuple(sorted(occurrences, key=lambda row: (row[0], row[1])))
    residual_rows = tuple(
        row for row in chronological if row[0] == binding.singleton_token_index
    )
    repeated_rows = tuple(
        row for row in chronological if row[0] in binding.repeated_token_indices
    )
    if (
        len(residual_rows) != 1
        or len(repeated_rows) != 2
        or tuple(row[0] for row in repeated_rows) != binding.repeated_token_indices
        or any(row[1] != binding.repeated_payload_cells for row in repeated_rows)
        or residual_rows[0][1] == binding.repeated_payload_cells
    ):
        raise ValueError("ordered binding does not authenticate one exact residual")
    return residual_rows[0][1]


def _observe_residual(
    checkpoint: ordered.FrozenCausalRelativeOrderedOperandV1,
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
    *,
    equality_gate: bool,
    literal_role_gate: bool,
    order_gate: bool,
    upstream_cell_gate: bool,
    residual_gate: bool,
) -> _ResidualObservation:
    before = checkpoint.checkpoint_sha256
    certificate = checkpoint.bind(
        span_records,
        literal_records,
        signature_role_record,
        equality_gate=equality_gate,
        literal_role_gate=literal_role_gate,
        order_gate=order_gate,
        cell_gate=upstream_cell_gate,
    )
    if checkpoint.checkpoint_sha256 != before:
        raise RuntimeError("banked ordered checkpoint mutated during inference")
    if certificate.status != ordered.READY:
        return _ResidualObservation(
            STOP_DEPENDENCY,
            certificate.stop_reason or certificate.status,
            None,
            None,
            (),
            (),
            (),
            None,
            certificate.sha256,
            "",
        )
    binding = ordered.require_relative_ordered_operand_binding(certificate)
    if not residual_gate:
        return _ResidualObservation(
            STOP_RESIDUAL_GATE,
            "physical residual-source gate was disabled",
            binding.relation,
            binding.selected_action,
            binding.repeated_payload_cells,
            binding.repeated_token_indices,
            (),
            binding.singleton_token_index,
            certificate.sha256,
            binding.prompt_sha256,
        )
    try:
        residual_payload = _extract_residual_payload(binding, literal_records)
    except ValueError as error:
        return _ResidualObservation(
            STOP_RESIDUAL_AUTH,
            str(error),
            binding.relation,
            binding.selected_action,
            binding.repeated_payload_cells,
            binding.repeated_token_indices,
            (),
            binding.singleton_token_index,
            certificate.sha256,
            binding.prompt_sha256,
        )
    return _ResidualObservation(
        READY,
        None,
        binding.relation,
        binding.selected_action,
        binding.repeated_payload_cells,
        binding.repeated_token_indices,
        residual_payload,
        binding.singleton_token_index,
        certificate.sha256,
        binding.prompt_sha256,
    )


def freeze_residual_literal_route_scaffold(
    repeated_value: int,
    residual_value: int,
    operand_action: int,
    selected_route: int,
) -> plan_slice.StructuralFactorCertificate:
    """Vary only the explicit additive literal domain in the banked scaffold."""

    if type(selected_route) is not int or selected_route not in {
        RESIDUAL_TO_LITERAL_HOLE,
        KEEP_DEFAULT_LITERAL,
    }:
        raise ValueError("residual route action lies outside the fixed capacity")
    if type(residual_value) is not int or not (
        output_cells.INT_MIN <= residual_value <= output_cells.INT_MAX
    ):
        raise ValueError("residual value lies outside the physical Int bank")
    base = ordered.freeze_ordered_operand_scaffold(repeated_value, operand_action)
    expected_domains = (
        plan_slice.StructuralDomainConstraint(3, 0, (repeated_value,)),
        plan_slice.StructuralDomainConstraint(
            ADDITIVE_LITERAL_OCCURRENCE,
            ADDITIVE_LITERAL_SLOT,
            (DEFAULT_LITERAL,),
        ),
    )
    if base.domains != expected_domains:
        raise RuntimeError("banked ordered scaffold literal domains changed")
    routed_value = (
        residual_value
        if selected_route == RESIDUAL_TO_LITERAL_HOLE
        else DEFAULT_LITERAL
    )
    certificate = plan_slice.freeze_structural_factor_certificate(
        base.occurrences,
        (
            expected_domains[0],
            plan_slice.StructuralDomainConstraint(
                ADDITIVE_LITERAL_OCCURRENCE,
                ADDITIVE_LITERAL_SLOT,
                (routed_value,),
            ),
        ),
        base.literal_slot_copy_groups,
    )
    if selected_route == KEEP_DEFAULT_LITERAL and certificate.sha256 != base.sha256:
        raise RuntimeError("default-route scaffold changed")
    return certificate


def _require_distinct_route_outputs(
    outputs: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
) -> None:
    if len(outputs) != ROUTE_ACTION_CELL_COUNT or outputs[0] == outputs[1]:
        raise ResidualLiteralFitStop(
            STOP_ALIAS,
            "residual and default literal routes have identical consequences",
        )


def _authorize_routes(
    frontier: breadth_v4.PackedGenericTypedTrainBreadthV4,
    observation: _ResidualObservation,
) -> tuple[tuple[tuple[output_cells.TypedOutputEncoding, ...], ...], int]:
    if observation.operand_action is None:
        raise ResidualLiteralFitStop(STOP_DEPENDENCY, "operand action is unavailable")
    repeated_value = literal_role.decode_int_payload(
        observation.repeated_payload_cells
    )
    residual_value = literal_role.decode_int_payload(
        observation.residual_payload_cells
    )
    outputs: list[tuple[output_cells.TypedOutputEncoding, ...]] = []
    authorizations = 0
    for route in range(ROUTE_ACTION_CELL_COUNT):
        scaffold = freeze_residual_literal_route_scaffold(
            repeated_value,
            residual_value,
            observation.operand_action,
            route,
        )
        try:
            (
                status,
                count,
                receipt,
                _root_receipt,
                _build_work,
                _authorization_work,
            ) = plan_slice._bind_certificate(frontier, scaffold)
        except packed.PackedRelationWorkLimitExceeded as error:
            raise ResidualLiteralFitStop(
                STOP_WORK_BOUND,
                "residual route authorization exceeded packed relation work",
            ) from error
        if status != plan_slice.FRONTIER_READY or count != 1 or receipt is None:
            raise ResidualLiteralFitStop(
                STOP_UNSUPPORTED,
                "one residual route has no unique packed plan slice "
                f"(status={status}, authorized={count})",
            )
        binding = plan_slice._authenticate_plan_slice_receipt(frontier, receipt)
        outputs.append(tuple(binding.candidate.eclass.physical_outputs))
        authorizations += 1
    frozen = tuple(outputs)
    _require_distinct_route_outputs(frozen)
    return frozen, authorizations


def _race_outcomes(
    outputs: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
    desired: tuple[int, ...],
    *,
    episode_ordinal: int,
    relation: int,
    operand_action: int,
) -> PackedResidualRouteAudit:
    keys = tuple(
        f"episode-{episode_ordinal}-route-{route}"
        for route in range(ROUTE_ACTION_CELL_COUNT)
    )
    encoded_desired = tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.INT,
        )
        for value in desired
    )
    race = semantic_race.select_exact_semantic_vector(keys, outputs, encoded_desired)
    if race.status != semantic_race.SELECTED or race.winner_key is None:
        raise ResidualLiteralFitStop(
            STOP_AMBIGUOUS,
            "residual route interventions have zero or multiple PASS outcomes",
        )
    return PackedResidualRouteAudit(
        episode_ordinal,
        relation,
        operand_action,
        keys.index(race.winner_key),
        ROUTE_ACTION_CELL_COUNT,
        race.sha256,
        1,
    )


def _learn_policy(
    audits: Sequence[PackedResidualRouteAudit],
    *,
    outcome_gate: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if not audits:
        raise ResidualLiteralFitStop(STOP_INCOMPLETE, "no causal route audits")
    winners = tuple(audit.pass_route for audit in audits)
    if any(winner != winners[0] for winner in winners):
        raise ResidualLiteralFitStop(
            STOP_CONFLICT,
            "causal lessons disagree on the residual literal route",
        )
    excitatory = np.zeros((ROUTE_ACTION_CELL_COUNT,), dtype=TICK_DTYPE)
    inhibitory = np.zeros_like(excitatory)
    excitatory_ticks = 0
    inhibitory_ticks = 0
    if outcome_gate:
        for audit in audits:
            for route in range(ROUTE_ACTION_CELL_COUNT):
                target = excitatory if route == audit.pass_route else inhibitory
                before = int(target[route])
                target[route] = min(TICK_CAP, before + 1)
                delta = int(target[route]) - before
                if route == audit.pass_route:
                    excitatory_ticks += delta
                else:
                    inhibitory_ticks += delta
    return excitatory, inhibitory, excitatory_ticks, inhibitory_ticks


def _checkpoint_sha(
    ordered_checkpoint_sha256: str,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> str:
    return sha256(
        _CHECKPOINT_NAMESPACE
        + ordered_checkpoint_sha256.encode("ascii")
        + b"\0"
        + _array_frame("residual-route-excitatory", excitatory)
        + _array_frame("residual-route-inhibitory", inhibitory)
    ).hexdigest()


class FrozenCausalResidualLiteralRouteV1:
    """Immutable two-action route from the residual literal to one hole."""

    __slots__ = (
        "_checkpoint_sha256",
        "_excitatory",
        "_inhibitory",
        "_ordered_checkpoint",
        "_summary",
    )

    def __init__(
        self,
        ordered_checkpoint: ordered.FrozenCausalRelativeOrderedOperandV1,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        summary: ResidualLiteralTrainingSummary,
        *,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("residual route checkpoints are builder-owned")
        if type(ordered_checkpoint) is not ordered.FrozenCausalRelativeOrderedOperandV1:
            raise TypeError("residual route requires the banked ordered checkpoint")
        ordered_checkpoint.authenticate()
        expected_shape = (ROUTE_ACTION_CELL_COUNT,)
        if (
            excitatory.shape != expected_shape
            or inhibitory.shape != expected_shape
            or excitatory.dtype != TICK_DTYPE
            or inhibitory.dtype != TICK_DTYPE
            or np.any(excitatory > TICK_CAP)
            or np.any(inhibitory > TICK_CAP)
        ):
            raise ValueError("residual route checkpoint geometry changed")
        self._ordered_checkpoint = ordered_checkpoint
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._checkpoint_sha256 = _checkpoint_sha(
            ordered_checkpoint.checkpoint_sha256,
            self._excitatory,
            self._inhibitory,
        )
        self._summary = summary
        if summary.checkpoint_sha256 != self._checkpoint_sha256:
            raise RuntimeError("residual route training summary changed")
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def ordered_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._ordered_checkpoint.checkpoint_sha256

    @property
    def training_summary(self) -> ResidualLiteralTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def learned_state_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._excitatory.shape, self._inhibitory.shape

    def authenticate(self) -> None:
        self._ordered_checkpoint.authenticate()
        if (
            self._excitatory.flags.writeable
            or self._inhibitory.flags.writeable
            or _checkpoint_sha(
                self._ordered_checkpoint.checkpoint_sha256,
                self._excitatory,
                self._inhibitory,
            )
            != self._checkpoint_sha256
        ):
            raise RuntimeError("residual route checkpoint authentication failed")

    def bind(
        self,
        span_records: Sequence[Mapping[str, object]],
        literal_records: Sequence[Mapping[str, object]],
        signature_role_record: Mapping[str, object],
        *,
        equality_gate: bool = True,
        literal_role_gate: bool = True,
        order_gate: bool = True,
        upstream_cell_gate: bool = True,
        residual_gate: bool = True,
        cell_gate: bool = True,
    ) -> ResidualLiteralRouteCertificate:
        self.authenticate()
        gates = (
            equality_gate,
            literal_role_gate,
            order_gate,
            upstream_cell_gate,
            residual_gate,
            cell_gate,
        )
        if any(type(value) is not bool for value in gates):
            raise TypeError("residual route gates must be exact booleans")
        observation = _observe_residual(
            self._ordered_checkpoint,
            span_records,
            literal_records,
            signature_role_record,
            equality_gate=equality_gate,
            literal_role_gate=literal_role_gate,
            order_gate=order_gate,
            upstream_cell_gate=upstream_cell_gate,
            residual_gate=residual_gate,
        )
        if observation.status != READY:
            return _issue_binding(
                self,
                observation,
                None,
                observation.status,
                observation.reason,
            )
        scores = self._excitatory.astype(np.int16) - self._inhibitory.astype(
            np.int16
        )
        currents = np.asarray(
            [
                max(0, int(score)) * OUTPUT_CURRENT_PA if cell_gate else 0.0
                for score in scores
            ],
            dtype=np.float64,
        )
        spikes = tuple(
            int(index)
            for index in np.flatnonzero(
                _ROUTE_ACTION_RACE.accumulated_spike_mask(currents)
            )
        )
        if not spikes:
            return _issue_binding(
                self,
                observation,
                None,
                STOP_ZERO,
                "no learned residual route crossed the physical race",
            )
        if len(spikes) != 1:
            return _issue_binding(
                self,
                observation,
                None,
                STOP_AMBIGUOUS,
                "multiple residual routes crossed together",
            )
        return _issue_binding(self, observation, spikes[0], READY, None)


def _work_payload(value: ResidualLiteralBindingWork) -> dict[str, object]:
    return {field: getattr(value, field) for field in value._fields if field != "sha256"}


def _receipt_payload(value: _BindingReceipt) -> dict[str, object]:
    return {
        field: getattr(value, field)
        for field in value._fields
        if field not in {"authority", "work", "sha256"}
    } | {"work_sha256": value.work.sha256}


def _issue_binding(
    checkpoint: FrozenCausalResidualLiteralRouteV1,
    observation: _ResidualObservation,
    selected_route: int | None,
    status: str,
    stop_reason: str | None,
) -> ResidualLiteralRouteCertificate:
    ready_for_route = bool(observation.residual_payload_cells)
    work_partial = ResidualLiteralBindingWork(
        observation.ordered_binding_sha256,
        UPSTREAM_BINDING_CELL_STEPS,
        ROUTE_PHASE_CELL_STEPS if ready_for_route else 0,
        ROUTE_ACTION_CELL_COUNT if ready_for_route else 0,
        ROUTE_ACTION_RACE_CELL_STEPS if ready_for_route else 0,
        STACKED_BINDING_CELL_STEPS if ready_for_route else UPSTREAM_BINDING_CELL_STEPS,
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
        checkpoint.ordered_checkpoint_sha256,
        observation.ordered_binding_sha256,
        observation.prompt_sha256,
        observation.repeated_payload_cells,
        observation.repeated_token_indices,
        observation.residual_payload_cells,
        observation.residual_token_index,
        observation.relation,
        observation.operand_action,
        selected_route,
        work,
        "",
    )
    receipt = partial._replace(
        sha256=_digest(_BINDING_NAMESPACE, _receipt_payload(partial))
    )
    certificate = ResidualLiteralRouteCertificate(receipt)
    _ISSUED_BINDINGS.add(certificate)
    return certificate


def _authenticate_binding(certificate: ResidualLiteralRouteCertificate) -> _BindingReceipt:
    if (
        type(certificate) is not ResidualLiteralRouteCertificate
        or certificate not in _ISSUED_BINDINGS
        or certificate._receipt.authority is not _BINDING_AUTHORITY
    ):
        raise TypeError("residual route binding is not an issued exact object")
    receipt = certificate._receipt
    if (
        receipt.work.sha256 != _digest(_WORK_NAMESPACE, _work_payload(receipt.work))
        or receipt.sha256 != _digest(_BINDING_NAMESPACE, _receipt_payload(receipt))
    ):
        raise RuntimeError("residual route binding certificate changed")
    return receipt


def require_residual_literal_route_binding(
    certificate: ResidualLiteralRouteCertificate,
) -> ResidualLiteralRouteBinding:
    receipt = _authenticate_binding(certificate)
    if (
        receipt.status != READY
        or receipt.residual_token_index is None
        or receipt.relation is None
        or receipt.operand_action is None
        or receipt.selected_route is None
    ):
        raise RuntimeError(receipt.stop_reason or receipt.status)
    return ResidualLiteralRouteBinding(
        receipt.repeated_payload_cells,
        receipt.repeated_token_indices,
        receipt.residual_payload_cells,
        receipt.residual_token_index,
        receipt.relation,
        receipt.operand_action,
        receipt.selected_route,
        receipt.prompt_sha256,
        receipt.ordered_binding_sha256,
        receipt.sha256,
    )


def fit_causal_residual_literal_route_v1(
    ordered_checkpoint: ordered.FrozenCausalRelativeOrderedOperandV1,
    episodes: tuple[CausalResidualLiteralEpisode, ...],
    *,
    signature: TypedSignature,
    cases: tuple[tuple[object, ...], ...],
    literals: tuple[int, ...],
    max_cost: int,
    outcome_gate: bool = True,
) -> FrozenCausalResidualLiteralRouteV1:
    """Fit the constant residual-vs-default literal route from outcomes."""

    if type(ordered_checkpoint) is not ordered.FrozenCausalRelativeOrderedOperandV1:
        raise TypeError("residual route fit requires the banked checkpoint type")
    ordered_checkpoint.authenticate()
    dependency_sha = ordered_checkpoint.checkpoint_sha256
    if type(episodes) is not tuple or any(
        type(episode) is not CausalResidualLiteralEpisode for episode in episodes
    ):
        raise TypeError("episodes must be an exact tuple of residual route episodes")
    if not MIN_CAUSAL_EPISODES <= len(episodes) <= MAX_CAUSAL_EPISODES:
        raise ResidualLiteralFitStop(
            STOP_INCOMPLETE,
            "residual route learning requires two to four causal episodes",
        )
    if not _signature_ok(signature) or max_cost != FIXED_MAX_COST:
        raise ValueError("residual route scaffold requires List[Int] -> Int at cost 8")
    if type(outcome_gate) is not bool:
        raise TypeError("outcome_gate must be an exact boolean")
    frozen_cases = tuple(tuple(row) for row in cases)
    frozen_literals = tuple(literals)
    if (
        not frozen_cases
        or DEFAULT_LITERAL not in frozen_literals
        or 0 not in frozen_literals
    ):
        raise ValueError("causal route interventions require cases, -15, and zero")

    observations: list[_ResidualObservation] = []
    for episode in episodes:
        if len(episode.desired) != len(frozen_cases) or any(
            type(value) is not int for value in episode.desired
        ):
            raise ValueError("episode desired outputs do not align with cases")
        observation = _observe_residual(
            ordered_checkpoint,
            episode.span_records,
            episode.literal_records,
            episode.signature_role_record,
            equality_gate=True,
            literal_role_gate=True,
            order_gate=True,
            upstream_cell_gate=True,
            residual_gate=True,
        )
        if observation.status != READY:
            raise ResidualLiteralFitStop(
                observation.status,
                observation.reason or "residual route observation stopped",
            )
        observations.append(observation)
    if {item.relation for item in observations} != {
        ordered.REPEATED_BEFORE_SINGLETON,
        ordered.SINGLETON_BEFORE_REPEATED,
    }:
        raise ResidualLiteralFitStop(
            STOP_INCOMPLETE,
            "both banked operand orientations need causal support",
        )
    if len({item.repeated_payload_cells for item in observations}) < 2 or len(
        {item.residual_payload_cells for item in observations}
    ) < 2:
        raise ResidualLiteralFitStop(
            STOP_INCOMPLETE,
            "causal support must vary repeated and residual payload identities",
        )

    frontier = breadth_v4.PackedGenericTypedTrainBreadthV4.build(
        signature,
        frozen_cases,
        frozen_literals,
        max_cost,
    )
    if frontier.status != breadth_v4.FRONTIER_READY:
        raise ResidualLiteralFitStop(
            STOP_WORK_BOUND,
            "ordinary packed residual-route frontier is not ready",
        )

    audits: list[PackedResidualRouteAudit] = []
    authorization_count = 0
    for ordinal, (episode, observation) in enumerate(
        zip(episodes, observations, strict=True)
    ):
        outputs, authorizations = _authorize_routes(frontier, observation)
        authorization_count += authorizations
        if observation.relation is None or observation.operand_action is None:
            raise RuntimeError("ready residual observation lost its banked action")
        audits.append(
            _race_outcomes(
                outputs,
                tuple(episode.desired),
                episode_ordinal=ordinal,
                relation=observation.relation,
                operand_action=observation.operand_action,
            )
        )
    if (
        authorization_count > MAX_PACKED_AUTHORIZATIONS
        or len(audits) > MAX_PACKED_OUTCOME_RACES
    ):
        raise ResidualLiteralFitStop(
            STOP_WORK_BOUND,
            "residual route intervention count exceeded the fixed work cap",
        )
    excitatory, inhibitory, excitatory_ticks, inhibitory_ticks = _learn_policy(
        audits,
        outcome_gate=outcome_gate,
    )
    if ordered_checkpoint.checkpoint_sha256 != dependency_sha:
        raise RuntimeError("banked ordered checkpoint changed during route fit")
    checkpoint_sha = _checkpoint_sha(dependency_sha, excitatory, inhibitory)
    summary = ResidualLiteralTrainingSummary(
        len(episodes),
        dependency_sha,
        MAX_PACKED_BUILDS,
        authorization_count,
        len(audits),
        len(episodes) * UPSTREAM_BINDING_CELL_STEPS,
        len(episodes) * ROUTE_PHASE_CELL_STEPS,
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
    return FrozenCausalResidualLiteralRouteV1(
        ordered_checkpoint,
        excitatory,
        inhibitory,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def freeze_outcome_off_control(
    checkpoint: FrozenCausalResidualLiteralRouteV1,
) -> FrozenCausalResidualLiteralRouteV1:
    """Silence the only PASS/FAIL-to-route plasticity gate."""

    if type(checkpoint) is not FrozenCausalResidualLiteralRouteV1:
        raise TypeError("outcome-off control requires the exact checkpoint type")
    checkpoint.authenticate()
    zeros = np.zeros((ROUTE_ACTION_CELL_COUNT,), dtype=TICK_DTYPE)
    checkpoint_sha = _checkpoint_sha(
        checkpoint.ordered_checkpoint_sha256,
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
    return FrozenCausalResidualLiteralRouteV1(
        checkpoint._ordered_checkpoint,
        zeros,
        zeros,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


CONFIG_RECORD = MappingProxyType(
    {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "ordered_config_sha256": ordered.CONFIG_SHA256,
        "base_cell_count": ordered.TOTAL_CELL_COUNT,
        "new_cell_count": NEW_CELL_COUNT,
        "total_cell_count": TOTAL_CELL_COUNT,
        "reused_selection_cell": REUSED_SELECTION_CELL,
        "reused_tonic_stop_cell": REUSED_TONIC_STOP_CELL,
        "reused_pass_outcome_cell": REUSED_PASS_OUTCOME_CELL,
        "reused_fail_outcome_cell": REUSED_FAIL_OUTCOME_CELL,
        "trainable_state_bytes": TRAINABLE_STATE_BYTES,
        "upstream_binding_cell_steps": UPSTREAM_BINDING_CELL_STEPS,
        "stacked_binding_cell_steps": STACKED_BINDING_CELL_STEPS,
        "maximum_causal_episodes": MAX_CAUSAL_EPISODES,
        "maximum_packed_builds": MAX_PACKED_BUILDS,
        "maximum_packed_authorizations": MAX_PACKED_AUTHORIZATIONS,
        "maximum_packed_outcome_races": MAX_PACKED_OUTCOME_RACES,
        "fixed_max_cost": FIXED_MAX_COST,
        "default_literal_foil": DEFAULT_LITERAL,
        "additive_literal_occurrence": ADDITIVE_LITERAL_OCCURRENCE,
        "additive_literal_slot": ADDITIVE_LITERAL_SLOT,
    }
)
CONFIG_SHA256 = _digest(_NAMESPACE + b"config\0", CONFIG_RECORD)


__all__ = [
    "ADDITIVE_LITERAL_OCCURRENCE",
    "ADDITIVE_LITERAL_SLOT",
    "CLAIM_BOUNDARY",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "DEFAULT_LITERAL",
    "FIXED_MAX_COST",
    "KEEP_DEFAULT_LITERAL",
    "MAX_PACKED_AUTHORIZATIONS",
    "MAX_PACKED_BUILDS",
    "MAX_PACKED_OUTCOME_RACES",
    "NEW_CELL_COUNT",
    "READY",
    "RESIDUAL_TO_LITERAL_HOLE",
    "ROUTE_ACTION_CELL_COUNT",
    "ROUTE_ACTION_RACE_CELL_STEPS",
    "ROUTE_PHASE_CELL_STEPS",
    "SCHEMA",
    "STACKED_BINDING_CELL_STEPS",
    "STOP_ALIAS",
    "STOP_AMBIGUOUS",
    "STOP_CONFLICT",
    "STOP_DEPENDENCY",
    "STOP_INCOMPLETE",
    "STOP_RESIDUAL_AUTH",
    "STOP_RESIDUAL_GATE",
    "STOP_UNSUPPORTED",
    "STOP_WORK_BOUND",
    "STOP_ZERO",
    "TOTAL_CELL_COUNT",
    "TRAINABLE_STATE_BYTES",
    "UPSTREAM_BINDING_CELL_STEPS",
    "CausalResidualLiteralEpisode",
    "FrozenCausalResidualLiteralRouteV1",
    "PackedResidualRouteAudit",
    "ResidualLiteralBindingWork",
    "ResidualLiteralFitStop",
    "ResidualLiteralRouteBinding",
    "ResidualLiteralRouteCertificate",
    "ResidualLiteralTrainingSummary",
    "fit_causal_residual_literal_route_v1",
    "freeze_outcome_off_control",
    "freeze_residual_literal_route_scaffold",
    "require_residual_literal_route_binding",
]
