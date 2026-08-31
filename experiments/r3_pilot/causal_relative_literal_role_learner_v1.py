"""Causal learning of one relative typed-literal role.

This self-posed mechanism keeps every physical literal occurrence, discovers
equal-payload groups through a two-phase coincidence population, and learns
whether the anonymous UNIQUE or REPEATED relation predicts a successful packed
intervention.  Literal cell addresses are transient payload/equality evidence;
the frozen trainable state has exactly two relation weights and no value,
prompt, task, family, program, or input-root registry.

The fixed ``List[Int] -> Int`` structural template is explicit experimental
scaffolding.  This module does not claim to learn that template or raw prompt
semantics.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v4 as breadth_v4
from experiments.r3_pilot import typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice
from humanbrain.resting_glif_race import CompiledRestingGlifRace
from humanbrain.typed_program_types import ProgramType, TypedSignature

SCHEMA = "CAUSAL_RELATIVE_LITERAL_ROLE_LEARNER_V1"
READY = "CAUSAL_RELATIVE_LITERAL_ROLE_LEARNER_V1_READY"
STOP_EQUALITY_OFF = "STOP_CAUSAL_RELATIVE_LITERAL_ROLE_LEARNER_V1_EQUALITY_OFF"
STOP_ZERO = "STOP_CAUSAL_RELATIVE_LITERAL_ROLE_LEARNER_V1_ZERO"
STOP_AMBIGUOUS = "STOP_CAUSAL_RELATIVE_LITERAL_ROLE_LEARNER_V1_AMBIGUOUS"
STOP_UNSUPPORTED = "STOP_CAUSAL_RELATIVE_LITERAL_ROLE_LEARNER_V1_UNSUPPORTED"

CLAIM_BOUNDARY = (
    "self-posed consequence-trained target-free-at-recall transfer of one "
    "unseen typed-Int payload through a preallocated equality/repetition "
    "relation into one fixed List[Int]-to-Int literal hole only; the fixed "
    "template is scaffolding, and raw-span semantics, full occurrence-tree "
    "induction, public panels, protected evaluation, and coding capability "
    "remain unproven"
)

UNIQUE_RELATION = 0
REPEATED_RELATION = 1
RELATION_CELL_COUNT = 2

COMPARATOR_CELL_COUNT = output_cells.INT_OFFSET + output_cells.INT_KNOWN_CELL_COUNT
COMPARATOR_ACTIVE_PAYLOAD_CELLS = 2
PHASE_GATE_CELL_COUNT = 2
FULL_MATCH_CELL_COUNT = 1
SELECTION_CELL_COUNT = 1
TONIC_STOP_CELL_COUNT = 1
OUTCOME_CELL_COUNT = 2
COMPARATOR_CELL_OFFSET = output_cells.TOTAL_CELL_COUNT
PHASE_GATE_CELL_OFFSET = COMPARATOR_CELL_OFFSET + COMPARATOR_CELL_COUNT
FULL_MATCH_CELL_OFFSET = PHASE_GATE_CELL_OFFSET + PHASE_GATE_CELL_COUNT
RELATION_CELL_OFFSET = FULL_MATCH_CELL_OFFSET + FULL_MATCH_CELL_COUNT
SELECTION_CELL_OFFSET = RELATION_CELL_OFFSET + RELATION_CELL_COUNT
TONIC_STOP_CELL_OFFSET = SELECTION_CELL_OFFSET + SELECTION_CELL_COUNT
PASS_OUTCOME_CELL = TONIC_STOP_CELL_OFFSET + TONIC_STOP_CELL_COUNT
FAIL_OUTCOME_CELL = PASS_OUTCOME_CELL + 1
NEW_CELL_COUNT = (
    COMPARATOR_CELL_COUNT
    + PHASE_GATE_CELL_COUNT
    + FULL_MATCH_CELL_COUNT
    + RELATION_CELL_COUNT
    + SELECTION_CELL_COUNT
    + TONIC_STOP_CELL_COUNT
    + OUTCOME_CELL_COUNT
)
TOTAL_CELL_COUNT = output_cells.TOTAL_CELL_COUNT + NEW_CELL_COUNT
if FAIL_OUTCOME_CELL != TOTAL_CELL_COUNT - 1:
    raise RuntimeError("relative literal cell allocation changed")

TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = 4
TRAINABLE_STATE_BYTES = 2 * RELATION_CELL_COUNT * TICK_DTYPE.itemsize
FIXED_COMPARATOR_SYNAPSE_COUNT = 3 * COMPARATOR_CELL_COUNT
EQUALITY_CELL_STEPS_PER_COMPARISON = (
    2 * COMPARATOR_CELL_COUNT + PHASE_GATE_CELL_COUNT + FULL_MATCH_CELL_COUNT
)
MAX_LITERAL_OCCURRENCES = 8
MAX_LITERAL_GROUPS = 2
MIN_CAUSAL_EPISODES = 2
OUTPUT_CURRENT_PA = 5_000.0
OUTPUT_STEPS = 30
_OUTPUT_RACE = CompiledRestingGlifRace.compile(step_count=OUTPUT_STEPS)

PACKED_BUILD_COUNT_PER_FIT = 1
PUBLIC_TRAIN_INPUT_COUNT = 0
CANONICAL_PANEL_SPEND_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
HOST_STRUCTURAL_LESSON_COUNT = 0

_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)
_LITERAL_INT = ("literal", "Int")
_ROLE_INT = ("role", "Int")
_ROLE_LIST_INT = ("role", "List[Int]")
_FIXED_TEMPLATE_LITERAL = -15

_SPAN_KEYS = {
    "start_token",
    "stop_token",
    "first_spike_step",
    "assembly_cells",
    "spike_cells",
}
_LITERAL_KEYS = {"canonical_token_index", "typed_output_active_cells"}
_SIGNATURE_KEYS = {
    "parameters",
    "result",
    "role_binding",
    "entrypoint_spelling_present",
    "parameter_spelling_present",
}
_EXPECTED_SIGNATURE_ROLE = {
    "parameters": [
        {
            "role": "ARG0",
            "type": {
                "constructor": "List",
                "arguments": [{"constructor": "Int", "arguments": []}],
            },
        }
    ],
    "result": {"constructor": "Int", "arguments": []},
    "role_binding": "reverse-declaration-depth",
    "entrypoint_spelling_present": False,
    "parameter_spelling_present": False,
}

_NAMESPACE = b"causal-relative-literal-role-learner-v1\0"
_CHECKPOINT_NAMESPACE = _NAMESPACE + b"checkpoint\0"
_BINDING_NAMESPACE = _NAMESPACE + b"binding\0"
_PROMPT_NAMESPACE = _NAMESPACE + b"prompt\0"
_WORK_NAMESPACE = _NAMESPACE + b"work\0"
_CONSTRUCTION_TOKEN = object()


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


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an exact integer")
    return value


def _physical_cells(
    value: object,
    label: str,
    *,
    width: int,
    count: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a physical cell sequence")
    cells = tuple(_integer(cell, label) for cell in value)
    if (
        len(cells) != count
        or tuple(sorted(cells)) != cells
        or len(cells) != len(set(cells))
        or any(not 0 <= cell < width for cell in cells)
    ):
        raise ValueError(f"{label} changed")
    return cells


class RelativeLiteralFitStop(RuntimeError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass(frozen=True, slots=True)
class CausalLiteralEpisode:
    span_records: tuple[Mapping[str, object], ...]
    literal_records: tuple[Mapping[str, object], ...]
    signature_role_record: Mapping[str, object]
    desired: tuple[int, ...]


class _LiteralOccurrence(NamedTuple):
    token_index: int
    payload_cells: tuple[int, ...]


class _LiteralGroup(NamedTuple):
    member_indices: tuple[int, ...]
    member_token_indices: tuple[int, ...]
    payload_cells: tuple[int, ...]
    relation_cell: int


class PackedInterventionAudit(NamedTuple):
    episode_ordinal: int
    candidate_group_count: int
    authorized_candidate_count: int
    physical_outcome_race_sha256: str
    pass_relation_cell: int
    pass_outcome_count: int


class CausalRelativeLiteralTrainingSummary(NamedTuple):
    episode_count: int
    packed_build_count: int
    packed_authorization_count: int
    packed_outcome_race_count: int
    literal_occurrence_count: int
    equality_comparison_count: int
    equality_physical_cell_steps: int
    distinct_positive_payload_count: int
    distinct_positive_position_count: int
    distinct_prompt_extent_count: int
    outcome_gate_on: bool
    excitatory_update_ticks: int
    inhibitory_update_ticks: int
    maximum_excitatory_tick: int
    maximum_inhibitory_tick: int
    trainable_state_bytes: int
    packed_work_sha256: str
    intervention_audits: tuple[PackedInterventionAudit, ...]
    checkpoint_sha256: str


class LiteralBindingWorkAudit(NamedTuple):
    literal_occurrence_count: int
    equality_comparison_count: int
    equality_physical_cell_steps: int
    candidate_group_count: int
    relation_race_candidate_count: int
    physical_output_cell_steps: int
    sha256: str


class _BindingReceipt(NamedTuple):
    authority: object
    status: str
    stop_reason: str | None
    checkpoint_sha256: str
    prompt_sha256: str
    selected_payload_cells: tuple[int, ...]
    selected_member_token_indices: tuple[int, ...]
    selected_relation_cell: int | None
    work: LiteralBindingWorkAudit
    sha256: str


_BINDING_AUTHORITY = object()
_ISSUED_BINDINGS: dict[int, RelativeLiteralBindingCertificate] = {}


class RelativeLiteralBindingCertificate:
    """Opaque exact-object result for one target-free literal-role recall."""

    __slots__ = ("_receipt",)

    def __init__(self, receipt: _BindingReceipt) -> None:
        if receipt.authority is not _BINDING_AUTHORITY:
            raise TypeError("binding certificate requires private authority")
        self._receipt = receipt

    @property
    def status(self) -> str:
        return self._receipt.status

    @property
    def stop_reason(self) -> str | None:
        return self._receipt.stop_reason

    @property
    def checkpoint_sha256(self) -> str:
        return self._receipt.checkpoint_sha256

    @property
    def selected_payload_cells(self) -> tuple[int, ...]:
        return self._receipt.selected_payload_cells

    @property
    def selected_member_token_indices(self) -> tuple[int, ...]:
        return self._receipt.selected_member_token_indices

    @property
    def work(self) -> LiteralBindingWorkAudit:
        return self._receipt.work

    @property
    def sha256(self) -> str:
        return self._receipt.sha256


class RelativeLiteralBinding(NamedTuple):
    payload_cells: tuple[int, ...]
    member_token_indices: tuple[int, ...]
    relation_cell: int
    prompt_sha256: str
    certificate_sha256: str


def _normalize_prompt_records(
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
) -> tuple[tuple[dict[str, object], ...], tuple[_LiteralOccurrence, ...], int]:
    if isinstance(span_records, (str, bytes)) or not isinstance(span_records, Sequence):
        raise TypeError("span records must be a sequence")
    if isinstance(literal_records, (str, bytes)) or not isinstance(
        literal_records, Sequence
    ):
        raise TypeError("literal records must be a sequence")

    normalized_spans: list[dict[str, object]] = []
    prompt_extent = 0
    for row in span_records:
        if not isinstance(row, Mapping) or set(row) != _SPAN_KEYS:
            raise ValueError("physical span record schema changed")
        start = _integer(row["start_token"], "span start")
        stop = _integer(row["stop_token"], "span stop")
        first_spike = _integer(row["first_spike_step"], "span first spike")
        assembly = _physical_cells(
            row["assembly_cells"], "span assembly", width=512, count=12
        )
        spikes = _physical_cells(
            row["spike_cells"], "span spikes", width=512, count=12
        )
        if start < 0 or stop <= start or first_spike < 0 or assembly != spikes:
            raise ValueError("physical span event changed")
        prompt_extent = max(prompt_extent, stop)
        normalized_spans.append(
            {
                "start_token": start,
                "stop_token": stop,
                "first_spike_step": first_spike,
                "assembly_cells": assembly,
                "spike_cells": spikes,
            }
        )
    if not normalized_spans:
        raise ValueError("physical prompt has no span records")

    occurrences: list[_LiteralOccurrence] = []
    for row in literal_records:
        if not isinstance(row, Mapping) or set(row) != _LITERAL_KEYS:
            raise ValueError("typed literal record schema changed")
        token_index = _integer(row["canonical_token_index"], "literal token index")
        payload = _physical_cells(
            row["typed_output_active_cells"],
            "typed literal cells",
            width=COMPARATOR_CELL_COUNT,
            count=COMPARATOR_ACTIVE_PAYLOAD_CELLS,
        )
        if (
            token_index < 0
            or payload[0] != output_cells.TYPE_OFFSET
            or not output_cells.INT_OFFSET
            <= payload[1]
            < output_cells.INT_OFFSET + output_cells.INT_KNOWN_CELL_COUNT
        ):
            raise ValueError("typed literal is not one known physical Int")
        occurrences.append(_LiteralOccurrence(token_index, payload))
    if not 1 <= len(occurrences) <= MAX_LITERAL_OCCURRENCES:
        raise ValueError("literal occurrence count lies outside the fixed capacity")

    if (
        not isinstance(signature_role_record, Mapping)
        or set(signature_role_record) != _SIGNATURE_KEYS
        or _plain(signature_role_record) != _EXPECTED_SIGNATURE_ROLE
    ):
        raise ValueError("name-free List[Int]-to-Int signature changed")

    return (
        tuple(
            sorted(
                normalized_spans,
                key=lambda row: (
                    int(row["start_token"]),
                    int(row["stop_token"]),
                    tuple(row["assembly_cells"]),
                ),
            )
        ),
        tuple(sorted(occurrences, key=lambda row: (row.token_index, row.payload_cells))),
        prompt_extent,
    )


def _two_phase_equal(
    candidate: tuple[int, ...],
    reference: tuple[int, ...],
    *,
    equality_gate: bool,
) -> bool:
    """Run a two-phase persistent coincidence population.

    Each physical comparator cell integrates one candidate-phase and one
    reference-phase tick.  The full-match neuron fires only when both active
    Int cells cross together; the shared Int type cell alone is subthreshold.
    """

    if not equality_gate:
        return False
    phase_gate_spikes = np.zeros(PHASE_GATE_CELL_COUNT, dtype=np.bool_)
    membrane = np.zeros(COMPARATOR_CELL_COUNT, dtype=np.uint8)
    phase_gate_spikes[0] = True
    membrane[np.asarray(candidate, dtype=np.intp)] += 1
    phase_gate_spikes[1] = True
    membrane[np.asarray(reference, dtype=np.intp)] += 1
    coincidence_spikes = np.greater_equal(membrane, 2)
    full_match_potential = int(np.count_nonzero(coincidence_spikes))
    return bool(np.all(phase_gate_spikes)) and (
        full_match_potential >= COMPARATOR_ACTIVE_PAYLOAD_CELLS
    )


def _literal_groups(
    occurrences: tuple[_LiteralOccurrence, ...],
    *,
    equality_gate: bool,
) -> tuple[tuple[_LiteralGroup, ...], int]:
    count = len(occurrences)
    adjacency = np.eye(count, dtype=np.bool_)
    comparisons = 0
    for left in range(count):
        for right in range(count):
            if left == right:
                continue
            comparisons += 1
            adjacency[left, right] = _two_phase_equal(
                occurrences[left].payload_cells,
                occurrences[right].payload_cells,
                equality_gate=equality_gate,
            )

    visited = [False] * count
    groups: list[_LiteralGroup] = []
    for start in range(count):
        if visited[start]:
            continue
        members: list[int] = []
        frontier = [start]
        while frontier:
            index = frontier.pop()
            if visited[index]:
                continue
            visited[index] = True
            members.append(index)
            for candidate in range(count - 1, -1, -1):
                if adjacency[index, candidate] and not visited[candidate]:
                    frontier.append(candidate)
        ordered_members = tuple(sorted(members))
        payload = occurrences[ordered_members[0]].payload_cells
        relation = REPEATED_RELATION if len(ordered_members) > 1 else UNIQUE_RELATION
        groups.append(
            _LiteralGroup(
                ordered_members,
                tuple(occurrences[index].token_index for index in ordered_members),
                payload,
                RELATION_CELL_OFFSET + relation,
            )
        )
    return tuple(groups), comparisons


def decode_int_payload(cells: tuple[int, ...]) -> int:
    if (
        type(cells) is not tuple
        or len(cells) != COMPARATOR_ACTIVE_PAYLOAD_CELLS
        or cells[0] != output_cells.TYPE_OFFSET
        or not output_cells.INT_OFFSET
        <= cells[1]
        < output_cells.INT_OFFSET + output_cells.INT_KNOWN_CELL_COUNT
    ):
        raise ValueError("physical payload is not one known Int")
    return output_cells.INT_MIN + cells[1] - output_cells.INT_OFFSET


def freeze_literal_hole_scaffold(
    hole_value: int,
) -> plan_slice.StructuralFactorCertificate:
    """Instantiate the one explicit self-posed structural scaffold."""

    if type(hole_value) is not int or not (
        output_cells.INT_MIN <= hole_value <= output_cells.INT_MAX
    ):
        raise ValueError("literal-hole value lies outside the physical Int bank")
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
        plan_slice.StructuralOccurrence(2, 0, (_LITERAL_INT,), (0,)),
        plan_slice.StructuralOccurrence(
            2,
            1,
            (_LITERAL_INT, _ROLE_INT),
            (1, 2),
            (plan_slice.StructuralRelationConstraint(0, "ne", right_value=0),),
        ),
        plan_slice.StructuralOccurrence(4, 0, (_ROLE_INT,), (1,)),
        plan_slice.StructuralOccurrence(4, 1, (_LITERAL_INT,), (0,)),
        plan_slice.StructuralOccurrence(0, 1, (_ROLE_LIST_INT,), (3,)),
    )
    return plan_slice.freeze_structural_factor_certificate(
        occurrences,
        (
            plan_slice.StructuralDomainConstraint(3, 0, (hole_value,)),
            plan_slice.StructuralDomainConstraint(
                6, 0, (_FIXED_TEMPLATE_LITERAL,)
            ),
        ),
        (((3, 0),), ((6, 0),)),
    )


def _signature_ok(signature: TypedSignature) -> bool:
    return (
        type(signature) is TypedSignature
        and signature.result == _INT
        and len(signature.parameters) == 1
        and signature.parameters[0].type == _LIST_INT
    )


def _literal_roster(values: Iterable[int]) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if type(value) is not int:
            raise ValueError("packed literal roster contains a non-integer")
        if value not in result:
            result.append(value)
    if _FIXED_TEMPLATE_LITERAL not in result:
        raise ValueError("packed literal roster omits the fixed scaffold literal")
    return tuple(result)


def _packed_outcomes(
    frontier: breadth_v4.PackedGenericTypedTrainBreadthV4,
    groups: tuple[_LiteralGroup, ...],
    desired: tuple[int, ...],
    *,
    episode_ordinal: int,
) -> tuple[tuple[tuple[int, ...], ...], PackedInterventionAudit]:
    if len(groups) != MAX_LITERAL_GROUPS:
        raise RelativeLiteralFitStop(
            STOP_UNSUPPORTED,
            "the first causal slice requires exactly two anonymous literal groups",
        )
    candidate_outputs: list[tuple[output_cells.TypedOutputEncoding, ...]] = []
    candidate_keys: list[str] = []
    authorization_count = 0
    for ordinal, group in enumerate(groups):
        certificate = freeze_literal_hole_scaffold(decode_int_payload(group.payload_cells))
        try:
            (
                status,
                authorized_count,
                receipt,
                _root_receipt,
                _build_work,
                _authorization_work,
            ) = plan_slice._bind_certificate(frontier, certificate)
        except packed.PackedRelationWorkLimitExceeded as error:
            raise RelativeLiteralFitStop(
                STOP_UNSUPPORTED,
                "one bounded literal intervention exceeded packed relation work",
            ) from error
        if status != plan_slice.FRONTIER_READY or authorized_count != 1 or receipt is None:
            raise RelativeLiteralFitStop(
                STOP_UNSUPPORTED,
                "one bounded literal intervention has no unique packed plan slice "
                f"(status={status}, authorized={authorized_count})",
            )
        binding = plan_slice._authenticate_plan_slice_receipt(frontier, receipt)
        candidate_outputs.append(tuple(binding.candidate.eclass.physical_outputs))
        candidate_keys.append(f"episode-{episode_ordinal}-candidate-{ordinal}")
        authorization_count += 1

    encoded_desired = tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.INT,
        )
        for value in desired
    )
    race = semantic_race.select_exact_semantic_vector(
        tuple(candidate_keys), tuple(candidate_outputs), encoded_desired
    )
    if race.status != semantic_race.SELECTED or race.winner_key is None:
        raise RelativeLiteralFitStop(
            STOP_AMBIGUOUS,
            "bounded packed literal interventions have zero or multiple exact outcomes",
        )
    winner = candidate_keys.index(race.winner_key)
    outcomes = tuple(
        (PASS_OUTCOME_CELL,) if index == winner else (FAIL_OUTCOME_CELL,)
        for index in range(len(groups))
    )
    return outcomes, PackedInterventionAudit(
        episode_ordinal,
        len(groups),
        authorization_count,
        race.sha256,
        groups[winner].relation_cell,
        sum(PASS_OUTCOME_CELL in row for row in outcomes),
    )


def _checkpoint_sha(excitatory: np.ndarray, inhibitory: np.ndarray) -> str:
    return sha256(
        _CHECKPOINT_NAMESPACE
        + _array_frame("relation-excitatory", excitatory)
        + _array_frame("relation-inhibitory", inhibitory)
    ).hexdigest()


class FrozenCausalRelativeLiteralRoleV1:
    """Immutable two-relation checkpoint with no physical-value registry."""

    __slots__ = ("_checkpoint_sha256", "_excitatory", "_inhibitory", "_summary")

    def __init__(
        self,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        summary: CausalRelativeLiteralTrainingSummary,
        *,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("relative literal checkpoints are builder-owned")
        if (
            excitatory.shape != (RELATION_CELL_COUNT,)
            or inhibitory.shape != excitatory.shape
            or excitatory.dtype != TICK_DTYPE
            or inhibitory.dtype != TICK_DTYPE
            or np.any(excitatory > TICK_CAP)
            or np.any(inhibitory > TICK_CAP)
        ):
            raise ValueError("relative literal checkpoint geometry changed")
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._checkpoint_sha256 = _checkpoint_sha(
            self._excitatory, self._inhibitory
        )
        self._summary = summary
        if summary.checkpoint_sha256 != self._checkpoint_sha256:
            raise RuntimeError("relative literal training summary changed")
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> CausalRelativeLiteralTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def learned_state_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._excitatory.shape, self._inhibitory.shape

    def authenticate(self) -> None:
        if (
            self._excitatory.flags.writeable
            or self._inhibitory.flags.writeable
            or _checkpoint_sha(self._excitatory, self._inhibitory)
            != self._checkpoint_sha256
        ):
            raise RuntimeError("relative literal checkpoint authentication failed")

    def bind(
        self,
        span_records: Sequence[Mapping[str, object]],
        literal_records: Sequence[Mapping[str, object]],
        signature_role_record: Mapping[str, object],
        *,
        equality_gate: bool = True,
        cell_gate: bool = True,
    ) -> RelativeLiteralBindingCertificate:
        """Select one relation-bound payload without any desired evidence."""

        self.authenticate()
        if type(equality_gate) is not bool or type(cell_gate) is not bool:
            raise TypeError("equality_gate and cell_gate must be exact booleans")
        spans, occurrences, prompt_extent = _normalize_prompt_records(
            span_records, literal_records, signature_role_record
        )
        prompt_sha = _digest(
            _PROMPT_NAMESPACE,
            {
                "spans": spans,
                "literal_occurrences": occurrences,
                "signature_role": _plain(signature_role_record),
                "prompt_extent": prompt_extent,
            },
        )
        if not equality_gate:
            return _issue_binding(
                status=STOP_EQUALITY_OFF,
                stop_reason="physical equality population was disabled",
                checkpoint_sha256=self._checkpoint_sha256,
                prompt_sha256=prompt_sha,
                occurrences=occurrences,
                groups=(),
                comparisons=0,
            )
        groups, comparisons = _literal_groups(
            occurrences, equality_gate=equality_gate
        )
        scores = tuple(
            int(self._excitatory[group.relation_cell - RELATION_CELL_OFFSET])
            - int(self._inhibitory[group.relation_cell - RELATION_CELL_OFFSET])
            for group in groups
        )
        currents = np.asarray(
            [max(0, score) * OUTPUT_CURRENT_PA if cell_gate else 0.0 for score in scores],
            dtype=np.float64,
        )
        spikes = tuple(
            int(index)
            for index in np.flatnonzero(_OUTPUT_RACE.accumulated_spike_mask(currents))
        )
        if not spikes:
            status = STOP_ZERO
            reason = "no learned relative literal role crossed the physical race"
            selected = None
        elif len(spikes) != 1:
            status = STOP_AMBIGUOUS
            reason = "multiple relative literal groups crossed together"
            selected = None
        else:
            status = READY
            reason = None
            selected = groups[spikes[0]]
        return _issue_binding(
            status=status,
            stop_reason=reason,
            checkpoint_sha256=self._checkpoint_sha256,
            prompt_sha256=prompt_sha,
            occurrences=occurrences,
            groups=groups,
            comparisons=comparisons,
            selected=selected,
        )


def _work_payload(work: LiteralBindingWorkAudit) -> dict[str, object]:
    return {field: getattr(work, field) for field in work._fields if field != "sha256"}


def _binding_payload(receipt: _BindingReceipt) -> dict[str, object]:
    return {
        "status": receipt.status,
        "stop_reason": receipt.stop_reason,
        "checkpoint_sha256": receipt.checkpoint_sha256,
        "prompt_sha256": receipt.prompt_sha256,
        "selected_payload_cells": receipt.selected_payload_cells,
        "selected_member_token_indices": receipt.selected_member_token_indices,
        "selected_relation_cell": receipt.selected_relation_cell,
        "work_sha256": receipt.work.sha256,
    }


def _issue_binding(
    *,
    status: str,
    stop_reason: str | None,
    checkpoint_sha256: str,
    prompt_sha256: str,
    occurrences: tuple[_LiteralOccurrence, ...],
    groups: tuple[_LiteralGroup, ...],
    comparisons: int,
    selected: _LiteralGroup | None = None,
) -> RelativeLiteralBindingCertificate:
    work_partial = LiteralBindingWorkAudit(
        len(occurrences),
        comparisons,
        comparisons * EQUALITY_CELL_STEPS_PER_COMPARISON,
        len(groups),
        len(groups),
        len(groups) * OUTPUT_STEPS,
        "",
    )
    work = work_partial._replace(
        sha256=_digest(_WORK_NAMESPACE, _work_payload(work_partial))
    )
    partial = _BindingReceipt(
        _BINDING_AUTHORITY,
        status,
        stop_reason,
        checkpoint_sha256,
        prompt_sha256,
        () if selected is None else selected.payload_cells,
        () if selected is None else selected.member_token_indices,
        None if selected is None else selected.relation_cell,
        work,
        "",
    )
    receipt = partial._replace(
        sha256=_digest(_BINDING_NAMESPACE, _binding_payload(partial))
    )
    certificate = RelativeLiteralBindingCertificate(receipt)
    _ISSUED_BINDINGS[id(certificate)] = certificate
    return certificate


def _authenticate_binding(
    certificate: RelativeLiteralBindingCertificate,
) -> _BindingReceipt:
    if (
        type(certificate) is not RelativeLiteralBindingCertificate
        or _ISSUED_BINDINGS.get(id(certificate)) is not certificate
        or certificate._receipt.authority is not _BINDING_AUTHORITY
    ):
        raise TypeError("literal binding is not an issued exact object")
    receipt = certificate._receipt
    if (
        receipt.work.sha256 != _digest(_WORK_NAMESPACE, _work_payload(receipt.work))
        or receipt.sha256 != _digest(_BINDING_NAMESPACE, _binding_payload(receipt))
    ):
        raise RuntimeError("literal binding certificate changed")
    return receipt


def require_relative_literal_binding(
    certificate: RelativeLiteralBindingCertificate,
) -> RelativeLiteralBinding:
    receipt = _authenticate_binding(certificate)
    if receipt.status != READY or receipt.selected_relation_cell is None:
        raise RuntimeError(receipt.stop_reason or receipt.status)
    return RelativeLiteralBinding(
        receipt.selected_payload_cells,
        receipt.selected_member_token_indices,
        receipt.selected_relation_cell,
        receipt.prompt_sha256,
        receipt.sha256,
    )


def fit_causal_relative_literal_role_v1(
    episodes: tuple[CausalLiteralEpisode, ...],
    *,
    signature: TypedSignature,
    cases: tuple[tuple[object, ...], ...],
    literals: tuple[int, ...],
    max_cost: int,
    outcome_gate: bool = True,
) -> FrozenCausalRelativeLiteralRoleV1:
    """Fit two relation weights from bounded packed PASS/FAIL interventions."""

    if type(episodes) is not tuple or any(
        type(episode) is not CausalLiteralEpisode for episode in episodes
    ):
        raise TypeError("episodes must be an exact tuple of causal literal episodes")
    if len(episodes) < MIN_CAUSAL_EPISODES:
        raise RelativeLiteralFitStop(
            STOP_UNSUPPORTED, "relative literal learning needs two causal contexts"
        )
    if not _signature_ok(signature):
        raise ValueError("causal literal scaffold requires List[Int] -> Int")
    if type(outcome_gate) is not bool:
        raise TypeError("outcome_gate must be an exact boolean")
    roster = _literal_roster(literals)
    frozen_cases = tuple(tuple(row) for row in cases)
    if not frozen_cases:
        raise ValueError("packed interventions need ordinary cases")

    prepared: list[
        tuple[
            tuple[_LiteralOccurrence, ...],
            tuple[_LiteralGroup, ...],
            int,
            int,
            tuple[int, ...],
        ]
    ] = []
    positive_payloads: list[tuple[int, ...]] = []
    positive_positions: list[tuple[int, ...]] = []
    prompt_extents: list[int] = []
    total_occurrences = 0
    total_comparisons = 0

    for episode in episodes:
        _spans, occurrences, extent = _normalize_prompt_records(
            episode.span_records,
            episode.literal_records,
            episode.signature_role_record,
        )
        groups, comparisons = _literal_groups(occurrences, equality_gate=True)
        if (
            len(groups) != MAX_LITERAL_GROUPS
            or sorted(len(group.member_indices) for group in groups) != [1, 2]
        ):
            raise RelativeLiteralFitStop(
                STOP_UNSUPPORTED,
                "each causal context needs one repeated and one unique literal group",
            )
        if len(episode.desired) != len(frozen_cases) or any(
            type(value) is not int for value in episode.desired
        ):
            raise ValueError("episode desired outputs do not align with ordinary cases")
        for group in groups:
            value = decode_int_payload(group.payload_cells)
            if value not in roster:
                raise RelativeLiteralFitStop(
                    STOP_UNSUPPORTED,
                    "a causal literal action is absent from the frozen packed roster",
                )
        repeated = next(
            group
            for group in groups
            if group.relation_cell == RELATION_CELL_OFFSET + REPEATED_RELATION
        )
        positive_payloads.append(repeated.payload_cells)
        positive_positions.append(repeated.member_token_indices)
        prompt_extents.append(extent)
        prepared.append(
            (occurrences, groups, comparisons, extent, tuple(episode.desired))
        )
        total_occurrences += len(occurrences)
        total_comparisons += comparisons

    def distinct_count(rows: Sequence[object]) -> int:
        unique: list[object] = []
        for row in rows:
            if row not in unique:
                unique.append(row)
        return len(unique)

    if (
        distinct_count(positive_payloads) < MIN_CAUSAL_EPISODES
        or distinct_count(positive_positions) < MIN_CAUSAL_EPISODES
        or distinct_count(prompt_extents) < MIN_CAUSAL_EPISODES
    ):
        raise RelativeLiteralFitStop(
            STOP_UNSUPPORTED,
            "causal support does not vary payload identity, position, and prompt extent",
        )

    packed_frontier = breadth_v4.PackedGenericTypedTrainBreadthV4.build(
        signature,
        frozen_cases,
        roster,
        max_cost,
    )
    if packed_frontier.status != breadth_v4.FRONTIER_READY:
        raise RelativeLiteralFitStop(
            STOP_UNSUPPORTED, "ordinary packed intervention frontier is not ready"
        )

    excitatory = np.zeros(RELATION_CELL_COUNT, dtype=TICK_DTYPE)
    inhibitory = np.zeros_like(excitatory)
    audits: list[PackedInterventionAudit] = []
    relation_labels: list[list[bool]] = [[], []]
    packed_authorizations = 0
    excitatory_ticks = 0
    inhibitory_ticks = 0

    for ordinal, (_occurrences, groups, _comparisons, _extent, desired) in enumerate(
        prepared
    ):
        outcomes, audit = _packed_outcomes(
            packed_frontier, groups, desired, episode_ordinal=ordinal
        )
        audits.append(audit)
        packed_authorizations += audit.authorized_candidate_count
        if audit.pass_outcome_count != 1:
            raise RelativeLiteralFitStop(
                STOP_AMBIGUOUS,
                "one causal context did not publish exactly one PASS outcome cell",
            )
        if audit.pass_relation_cell != RELATION_CELL_OFFSET + REPEATED_RELATION:
            raise RelativeLiteralFitStop(
                STOP_UNSUPPORTED,
                "the fixed curriculum did not causally identify the repeated role",
            )
        for group, outcome_cells_for_group in zip(groups, outcomes, strict=True):
            relation = group.relation_cell - RELATION_CELL_OFFSET
            passed = outcome_cells_for_group == (PASS_OUTCOME_CELL,)
            if not passed and outcome_cells_for_group != (FAIL_OUTCOME_CELL,):
                raise RuntimeError("packed intervention emitted an unknown outcome cell")
            relation_labels[relation].append(passed)
            if not outcome_gate:
                continue
            if passed:
                before = int(excitatory[relation])
                excitatory[relation] = min(TICK_CAP, before + 1)
                excitatory_ticks += int(excitatory[relation]) - before
            else:
                before = int(inhibitory[relation])
                inhibitory[relation] = min(TICK_CAP, before + 1)
                inhibitory_ticks += int(inhibitory[relation]) - before

    if any(not labels or any(label != labels[0] for label in labels) for labels in relation_labels):
        raise RelativeLiteralFitStop(
            STOP_AMBIGUOUS,
            "one abstract relation has conflicting causal consequences",
        )

    checkpoint_sha = _checkpoint_sha(excitatory, inhibitory)
    summary = CausalRelativeLiteralTrainingSummary(
        len(episodes),
        PACKED_BUILD_COUNT_PER_FIT,
        packed_authorizations,
        len(audits),
        total_occurrences,
        total_comparisons,
        total_comparisons * EQUALITY_CELL_STEPS_PER_COMPARISON,
        distinct_count(positive_payloads),
        distinct_count(positive_positions),
        distinct_count(prompt_extents),
        outcome_gate,
        excitatory_ticks,
        inhibitory_ticks,
        int(excitatory.max()),
        int(inhibitory.max()),
        TRAINABLE_STATE_BYTES,
        packed_frontier.work.sha256,
        tuple(audits),
        checkpoint_sha,
    )
    return FrozenCausalRelativeLiteralRoleV1(
        excitatory,
        inhibitory,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def freeze_outcome_off_control(
    checkpoint: FrozenCausalRelativeLiteralRoleV1,
) -> FrozenCausalRelativeLiteralRoleV1:
    """Freeze the same audited interventions with outcome plasticity silenced.

    Intervention collection is independent of the two learned relation weights,
    so silencing the sole PASS/FAIL-to-weight gate is exactly the zero-update
    counterfactual for the already-audited physical episode stream.  This avoids
    rebuilding the identical expensive packed breadth solely for an ablation.
    """

    if type(checkpoint) is not FrozenCausalRelativeLiteralRoleV1:
        raise TypeError("outcome-off control requires the exact checkpoint type")
    checkpoint.authenticate()
    zero_excitatory = np.zeros(RELATION_CELL_COUNT, dtype=TICK_DTYPE)
    zero_inhibitory = np.zeros_like(zero_excitatory)
    zero_sha = _checkpoint_sha(zero_excitatory, zero_inhibitory)
    summary = checkpoint.training_summary._replace(
        outcome_gate_on=False,
        excitatory_update_ticks=0,
        inhibitory_update_ticks=0,
        maximum_excitatory_tick=0,
        maximum_inhibitory_tick=0,
        checkpoint_sha256=zero_sha,
    )
    return FrozenCausalRelativeLiteralRoleV1(
        zero_excitatory,
        zero_inhibitory,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


CONFIG_RECORD = MappingProxyType(
    {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "relation_cell_count": RELATION_CELL_COUNT,
        "comparator_cell_count": COMPARATOR_CELL_COUNT,
        "comparator_cell_offset": COMPARATOR_CELL_OFFSET,
        "new_cell_count": NEW_CELL_COUNT,
        "total_cell_count": TOTAL_CELL_COUNT,
        "fixed_comparator_synapse_count": FIXED_COMPARATOR_SYNAPSE_COUNT,
        "trainable_state_bytes": TRAINABLE_STATE_BYTES,
        "maximum_literal_occurrences": MAX_LITERAL_OCCURRENCES,
        "maximum_literal_groups": MAX_LITERAL_GROUPS,
        "packed_build_count_per_fit": PACKED_BUILD_COUNT_PER_FIT,
        "public_train_input_count": PUBLIC_TRAIN_INPUT_COUNT,
        "canonical_panel_spend_count": CANONICAL_PANEL_SPEND_COUNT,
        "dev_input_count": DEV_INPUT_COUNT,
        "m0_input_count": M0_INPUT_COUNT,
        "c1_input_count": C1_INPUT_COUNT,
        "path2_input_count": PATH2_INPUT_COUNT,
        "g0_input_count": G0_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "host_structural_lesson_count": HOST_STRUCTURAL_LESSON_COUNT,
    }
)
CONFIG_SHA256 = _digest(_NAMESPACE + b"config\0", CONFIG_RECORD)


__all__ = [
    "CANONICAL_PANEL_SPEND_COUNT",
    "CLAIM_BOUNDARY",
    "COMPARATOR_CELL_COUNT",
    "COMPARATOR_CELL_OFFSET",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "DEV_INPUT_COUNT",
    "FAIL_OUTCOME_CELL",
    "FIXED_COMPARATOR_SYNAPSE_COUNT",
    "G0_INPUT_COUNT",
    "HOST_STRUCTURAL_LESSON_COUNT",
    "M0_INPUT_COUNT",
    "MAX_LITERAL_OCCURRENCES",
    "NEW_CELL_COUNT",
    "PACKED_BUILD_COUNT_PER_FIT",
    "PASS_OUTCOME_CELL",
    "PATH2_INPUT_COUNT",
    "PROTECTED_INPUT_COUNT",
    "PUBLIC_TRAIN_INPUT_COUNT",
    "READY",
    "RELATION_CELL_COUNT",
    "REPEATED_RELATION",
    "SCHEMA",
    "STOP_AMBIGUOUS",
    "STOP_EQUALITY_OFF",
    "STOP_UNSUPPORTED",
    "STOP_ZERO",
    "TOTAL_CELL_COUNT",
    "TRAINABLE_STATE_BYTES",
    "UNIQUE_RELATION",
    "CausalLiteralEpisode",
    "CausalRelativeLiteralTrainingSummary",
    "FrozenCausalRelativeLiteralRoleV1",
    "LiteralBindingWorkAudit",
    "RelativeLiteralBinding",
    "RelativeLiteralBindingCertificate",
    "RelativeLiteralFitStop",
    "decode_int_payload",
    "fit_causal_relative_literal_role_v1",
    "freeze_literal_hole_scaffold",
    "freeze_outcome_off_control",
    "require_relative_literal_binding",
]
