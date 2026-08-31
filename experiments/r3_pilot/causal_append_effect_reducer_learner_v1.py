"""Learn one reducer choice from a physical append-effect law.

The banked residual-literal checkpoint supplies the fixed conditional-count
predicate.  Three ordinary probe cases expose a base list and two one-item
extensions.  Fixed effect hypotheses ask whether output change follows the
predicate spike or copies the appended payload.  Only the PASS/FAIL outcomes
of two anonymously attached reducer scaffolds reach the uint8 plastic state.

Predicate evaluation, integer addition, append alignment, and the two reducer
scaffolds remain explicit experimental machinery.  Payload values are
transient physical evidence and never index trainable state.
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
from experiments.r3_pilot import causal_residual_literal_route_learner_v1 as residual
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v4 as breadth_v4
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.resting_glif_race import CompiledRestingGlifRace
from humanbrain.typed_program_types import ProgramType, TypedSignature

SCHEMA = "CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1"
READY = "CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_READY"
STOP_DEPENDENCY = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_DEPENDENCY"
STOP_APPEND_SHAPE = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_APPEND_SHAPE"
STOP_EQUALITY_GATE = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_EQUALITY_GATE"
STOP_PREDICATE_GATE = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_PREDICATE_GATE"
STOP_EFFECT_GATE = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_EFFECT_GATE"
STOP_COMPLETENESS = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_COMPLETENESS"
STOP_NONDIAGNOSTIC = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_NONDIAGNOSTIC"
STOP_PREDICATE_STRADDLE = (
    "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_PREDICATE_STRADDLE"
)
STOP_EFFECT_ZERO = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_EFFECT_ZERO"
STOP_EFFECT_TIE = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_EFFECT_TIE"
STOP_INCOMPLETE = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_INCOMPLETE"
STOP_CONFLICT = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_CONFLICT"
STOP_ACTION_ALIAS = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_ACTION_ALIAS"
STOP_PREALLOCATION = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_PREALLOCATION"
STOP_ZERO = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_ZERO"
STOP_AMBIGUOUS = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_AMBIGUOUS"
STOP_UNSUPPORTED = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_UNSUPPORTED"
STOP_WORK_BOUND = "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_WORK_BOUND"
STOP_PROBE_FINAL_OVERLAP = (
    "STOP_CAUSAL_APPEND_EFFECT_REDUCER_LEARNER_V1_PROBE_FINAL_OVERLAP"
)

CLAIM_BOUNDARY = (
    "self-posed consequence-trained transfer of one physical append-response "
    "law into one of two anonymous preallocated reducer actions across unseen "
    "literal payloads, append payloads, prompt assemblies, and positions; "
    "append alignment, predicate/addition execution, conditional-count and "
    "raw-sum scaffolds, typed decoding, and the token clock remain explicit, "
    "so reducer discovery, parsing, arbitrary composition, and coding "
    "capability remain unproven"
)

PREDICATE_EFFECT_RELATION = 0
PAYLOAD_EFFECT_RELATION = 1
EFFECT_RELATION_CELL_COUNT = 2
REDUCER_ACTION_CELL_COUNT = 2
COMPLETENESS_CELL_COUNT = 1

EFFECT_RELATION_CELL_OFFSET = residual.TOTAL_CELL_COUNT
REDUCER_ACTION_CELL_OFFSET = EFFECT_RELATION_CELL_OFFSET + EFFECT_RELATION_CELL_COUNT
COMPLETENESS_CELL = REDUCER_ACTION_CELL_OFFSET + REDUCER_ACTION_CELL_COUNT
NEW_CELL_COUNT = (
    EFFECT_RELATION_CELL_COUNT + REDUCER_ACTION_CELL_COUNT + COMPLETENESS_CELL_COUNT
)
TOTAL_CELL_COUNT = residual.TOTAL_CELL_COUNT + NEW_CELL_COUNT
if COMPLETENESS_CELL + COMPLETENESS_CELL_COUNT != TOTAL_CELL_COUNT:
    raise RuntimeError("append-effect cell allocation changed")

REUSED_SELECTION_CELL = residual.REUSED_SELECTION_CELL
REUSED_TONIC_STOP_CELL = residual.REUSED_TONIC_STOP_CELL
REUSED_PASS_OUTCOME_CELL = residual.REUSED_PASS_OUTCOME_CELL
REUSED_FAIL_OUTCOME_CELL = residual.REUSED_FAIL_OUTCOME_CELL

TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = 4
TRAINABLE_STATE_BYTES = (
    2 * EFFECT_RELATION_CELL_COUNT * REDUCER_ACTION_CELL_COUNT * TICK_DTYPE.itemsize
)
MIN_CAUSAL_EPISODES = 2
MAX_CAUSAL_EPISODES = 4
MAX_PACKED_BUILDS = 1
MAX_PACKED_AUTHORIZATIONS = 2 * MAX_CAUSAL_EPISODES
MAX_EFFECT_RACES = MAX_CAUSAL_EPISODES
MAX_OUTCOME_RACES = MAX_CAUSAL_EPISODES
FIXED_MAX_COST = 8
PROBE_CASE_COUNT = 3
MAX_BASE_LENGTH = output_cells.LIST_CAPACITY - 1
COMPLETENESS_CELL_STEPS = 1
SEMANTIC_RACE_STEPS = 600
SEMANTIC_RACE_CELL_STEPS = (EFFECT_RELATION_CELL_COUNT + 1) * SEMANTIC_RACE_STEPS
ACTION_RACE_STEPS = 30
ACTION_RACE_CELL_STEPS = REDUCER_ACTION_CELL_COUNT * ACTION_RACE_STEPS
OUTPUT_CURRENT_PA = 10_000.0
_ACTION_RACE = CompiledRestingGlifRace.compile(step_count=ACTION_RACE_STEPS)

_COUNT_ACTION = 0
_SUM_ACTION = 1
_ROLE_LIST_INT = ("role", "List[Int]")
_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)

_NAMESPACE = b"causal-append-effect-reducer-learner-v1\0"
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
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _array_frame(name: str, value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    return (
        name.encode("ascii")
        + b"\0"
        + array.dtype.str.encode("ascii")
        + b"\0"
        + _canonical_json(array.shape)
        + b"\0"
        + memoryview(array).cast("B")
    )


def _signature_ok(signature: TypedSignature) -> bool:
    return (
        isinstance(signature, TypedSignature)
        and len(signature.parameters) == 1
        and signature.parameters[0].type == _LIST_INT
        and signature.result == _INT
    )


@dataclass(frozen=True, slots=True)
class CausalAppendEffectEpisode:
    span_records: Sequence[Mapping[str, object]]
    literal_records: Sequence[Mapping[str, object]]
    signature_role_record: Mapping[str, object]
    probe_outputs: tuple[output_cells.TypedOutputEncoding, ...]


class AppendEffectInterventionAudit(NamedTuple):
    episode_ordinal: int
    effect_relation: int
    pass_action: int
    effect_race_sha256: str
    outcome_race_sha256: str
    probe_sha256: str
    pass_outcome_count: int


class AppendEffectTrainingSummary(NamedTuple):
    episode_count: int
    dependency_checkpoint_sha256: str
    packed_build_count: int
    packed_authorization_count: int
    effect_race_count: int
    packed_outcome_race_count: int
    upstream_binding_cell_steps: int
    probe_comparison_count: int
    probe_comparison_cell_steps: int
    completeness_cell_steps: int
    reducer_action_race_cell_steps: int
    outcome_gate_on: bool
    excitatory_update_ticks: int
    inhibitory_update_ticks: int
    maximum_excitatory_tick: int
    maximum_inhibitory_tick: int
    trainable_state_bytes: int
    packed_work_sha256: str
    intervention_audits: tuple[AppendEffectInterventionAudit, ...]
    checkpoint_sha256: str


class AppendEffectBindingWork(NamedTuple):
    upstream_binding_cell_steps: int
    probe_comparison_count: int
    probe_comparison_cell_steps: int
    completeness_cell_steps: int
    effect_race_cell_steps: int
    reducer_action_race_cell_steps: int
    stacked_binding_cell_steps: int
    sha256: str


class _ProbeShape(NamedTuple):
    status: str
    reason: str | None
    cases: tuple[tuple[tuple[int, ...]], ...]
    inserted_values: tuple[int, int]
    comparison_count: int


class _AppendObservation(NamedTuple):
    status: str
    reason: str | None
    residual_certificate: residual.ResidualLiteralRouteCertificate
    residual_binding: residual.ResidualLiteralRouteBinding | None
    probe_cases: tuple[tuple[tuple[int, ...]], ...]
    probe_output_sha256s: tuple[str, ...]
    inserted_values: tuple[int, int]
    predicate_spikes: tuple[int, int]
    effect_relation: int | None
    probe_sha256: str
    effect_race_sha256: str
    comparison_count: int
    completeness_spikes: int


class _BindingReceipt(NamedTuple):
    authority: object
    checkpoint_sha256: str
    status: str
    stop_reason: str | None
    residual_certificate: residual.ResidualLiteralRouteCertificate
    repeated_payload_cells: tuple[int, ...]
    repeated_token_indices: tuple[int, ...]
    residual_payload_cells: tuple[int, ...]
    residual_token_index: int | None
    operand_action: int | None
    residual_route: int | None
    effect_relation: int | None
    selected_action: int | None
    probe_cases: tuple[tuple[tuple[int, ...]], ...]
    inserted_values: tuple[int, int]
    predicate_spikes: tuple[int, int]
    probe_sha256: str
    effect_race_sha256: str
    work: AppendEffectBindingWork
    sha256: str


class AppendEffectReducerCertificate:
    """Opaque exact-object certificate for one learned reducer binding."""

    __slots__ = ("__weakref__", "_receipt")

    def __init__(self, receipt: _BindingReceipt, *, construction_token: object) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("append-effect certificates are builder-owned")
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
    def work(self) -> AppendEffectBindingWork:
        return self._receipt.work


class AppendEffectReducerBinding(NamedTuple):
    repeated_payload_cells: tuple[int, ...]
    repeated_token_indices: tuple[int, ...]
    residual_payload_cells: tuple[int, ...]
    residual_token_index: int
    operand_action: int
    residual_route: int
    effect_relation: int
    selected_action: int
    probe_cases: tuple[tuple[tuple[int, ...]], ...]
    inserted_values: tuple[int, int]
    predicate_spikes: tuple[int, int]
    probe_sha256: str
    residual_binding_sha256: str
    certificate_sha256: str


class AppendEffectFitStop(RuntimeError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


_ISSUED_BINDINGS: WeakSet[AppendEffectReducerCertificate] = WeakSet()


def _probe_outputs(
    values: Sequence[output_cells.TypedOutputEncoding],
) -> tuple[output_cells.TypedOutputEncoding, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("probe outputs must be a physical encoding sequence")
    frozen = tuple(values)
    if len(frozen) != PROBE_CASE_COUNT or any(
        type(item) is not output_cells.TypedOutputEncoding for item in frozen
    ):
        raise ValueError("append-effect learning requires exactly three Int outputs")
    validated = tuple(output_cells.TypedOutputCellBank.validate(item) for item in frozen)
    decoded = tuple(output_cells.TypedOutputCellBank.decode(item) for item in validated)
    if any(type(item) is not int for item in decoded):
        raise ValueError("append-effect probe outputs must be known physical Ints")
    return validated


def _int_payload(value: int) -> tuple[int, ...]:
    return output_cells.TypedOutputCellBank.encode(
        value, declared_type=output_cells.OutputType.INT
    ).active_global_cell_indices()


def _probe_shape(
    probe_cases: object,
    *,
    equality_gate: bool,
) -> _ProbeShape:
    if type(probe_cases) is not tuple:
        raise TypeError("probe cases must be an exact tuple")
    if len(probe_cases) != PROBE_CASE_COUNT:
        return _ProbeShape(
            STOP_APPEND_SHAPE,
            "append-effect observation needs base plus two extensions",
            (),
            (),
            0,
        )
    normalized: list[tuple[tuple[int, ...]]] = []
    for row in probe_cases:
        if type(row) is not tuple or len(row) != 1 or type(row[0]) is not list:
            return _ProbeShape(
                STOP_APPEND_SHAPE,
                "each append-effect case must contain one physical Int list",
                (),
                (),
                0,
            )
        values = row[0]
        if any(
            type(value) is not int
            or not output_cells.INT_MIN <= value <= output_cells.INT_MAX
            for value in values
        ):
            return _ProbeShape(
                STOP_UNSUPPORTED,
                "probe list contains an unsupported physical Int",
                (),
                (),
                0,
            )
        normalized.append((tuple(values),))
    base = normalized[0][0]
    extensions = (normalized[1][0], normalized[2][0])
    if not 1 <= len(base) <= MAX_BASE_LENGTH or any(
        len(extension) != len(base) + 1 for extension in extensions
    ):
        return _ProbeShape(
            STOP_APPEND_SHAPE,
            "probe extensions must add exactly one terminal occurrence",
            (),
            (),
            0,
        )
    if not equality_gate:
        return _ProbeShape(
            STOP_EQUALITY_GATE,
            "physical append-prefix equality cells were disabled",
            (),
            (),
            0,
        )
    comparisons = 0
    for extension in extensions:
        for base_value, extension_value in zip(base, extension[:-1], strict=True):
            comparisons += 1
            if not literal_role._two_phase_equal(
                _int_payload(base_value),
                _int_payload(extension_value),
                equality_gate=equality_gate,
            ):
                return _ProbeShape(
                    STOP_APPEND_SHAPE,
                    "one extension changed the ordered base occurrences",
                    (),
                    (),
                    comparisons,
                )
    inserted = (extensions[0][-1], extensions[1][-1])
    if inserted[0] == inserted[1] or any(value in {0, 1} for value in inserted):
        return _ProbeShape(
            STOP_NONDIAGNOSTIC,
            "insertions must be distinct and neither zero nor one",
            (),
            inserted,
            comparisons,
        )
    return _ProbeShape(READY, None, tuple(normalized), inserted, comparisons)


def _predicate_spikes(
    binding: residual.ResidualLiteralRouteBinding,
    inserted: tuple[int, int],
) -> tuple[int, int]:
    repeated = literal_role.decode_int_payload(binding.repeated_payload_cells)
    residual_value = literal_role.decode_int_payload(binding.residual_payload_cells)
    additive = (
        residual_value
        if binding.selected_route == residual.RESIDUAL_TO_LITERAL_HOLE
        else residual.DEFAULT_LITERAL
    )
    if binding.operand_action == 0:
        return tuple(int(repeated < value + additive) for value in inserted)  # type: ignore[return-value]
    if binding.operand_action == 1:
        return tuple(int(value + additive < repeated) for value in inserted)  # type: ignore[return-value]
    raise RuntimeError("banked operand action left the fixed two-way capacity")


def _effect_race(
    outputs: tuple[output_cells.TypedOutputEncoding, ...],
    inserted: tuple[int, int],
    predicate: tuple[int, int],
) -> tuple[str, int | None, str]:
    decoded = tuple(output_cells.TypedOutputCellBank.decode(item) for item in outputs)
    if any(type(value) is not int for value in decoded):
        return STOP_UNSUPPORTED, None, ""
    base = int(decoded[0])
    candidate_values = (
        (base, base + predicate[0], base + predicate[1]),
        (base, base + inserted[0], base + inserted[1]),
    )
    if any(
        not output_cells.INT_MIN <= value <= output_cells.INT_MAX
        for row in candidate_values
        for value in row
    ):
        return STOP_UNSUPPORTED, None, ""
    candidates = tuple(
        tuple(
            output_cells.TypedOutputCellBank.encode(
                value, declared_type=output_cells.OutputType.INT
            )
            for value in row
        )
        for row in candidate_values
    )
    race = semantic_race.select_exact_semantic_vector(
        ("effect-cell-0", "effect-cell-1"), candidates, outputs
    )
    if race.status == semantic_race.STOP_NO_EXACT:
        return STOP_EFFECT_ZERO, None, race.sha256
    if race.status == semantic_race.STOP_MULTIPLE_EXACT:
        return STOP_EFFECT_TIE, None, race.sha256
    if race.status != semantic_race.SELECTED or race.winner_key is None:
        raise RuntimeError("effect relation race returned an unsupported state")
    return READY, (0 if race.winner_key.endswith("0") else 1), race.sha256


def _observe(
    checkpoint: residual.FrozenCausalResidualLiteralRouteV1,
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
    probe_cases: object,
    probe_outputs: Sequence[output_cells.TypedOutputEncoding],
    *,
    equality_gate: bool,
    literal_role_gate: bool,
    order_gate: bool,
    upstream_cell_gate: bool,
    residual_gate: bool,
    residual_cell_gate: bool,
    predicate_gate: bool,
    effect_gate: bool,
    completeness_gate: bool,
) -> _AppendObservation:
    before = checkpoint.checkpoint_sha256
    residual_certificate = checkpoint.bind(
        span_records,
        literal_records,
        signature_role_record,
        equality_gate=equality_gate,
        literal_role_gate=literal_role_gate,
        order_gate=order_gate,
        upstream_cell_gate=upstream_cell_gate,
        residual_gate=residual_gate,
        cell_gate=residual_cell_gate,
    )
    if checkpoint.checkpoint_sha256 != before:
        raise RuntimeError("banked residual checkpoint mutated during inference")
    if residual_certificate.status != residual.READY:
        return _AppendObservation(
            STOP_DEPENDENCY,
            residual_certificate.stop_reason or residual_certificate.status,
            residual_certificate,
            None,
            (),
            (),
            (),
            (),
            None,
            "",
            "",
            0,
            0,
        )
    upstream_binding = residual.require_residual_literal_route_binding(
        residual_certificate
    )
    shape = _probe_shape(probe_cases, equality_gate=equality_gate)
    if shape.status != READY:
        return _AppendObservation(
            shape.status,
            shape.reason,
            residual_certificate,
            upstream_binding,
            shape.cases,
            (),
            shape.inserted_values,
            (),
            None,
            "",
            "",
            shape.comparison_count,
            0,
        )
    outputs = _probe_outputs(probe_outputs)
    output_hashes = tuple(item.sha256 for item in outputs)
    if not predicate_gate:
        return _AppendObservation(
            STOP_PREDICATE_GATE,
            "fixed predicate-effect cells were disabled",
            residual_certificate,
            upstream_binding,
            shape.cases,
            output_hashes,
            shape.inserted_values,
            (),
            None,
            "",
            "",
            shape.comparison_count,
            0,
        )
    predicate = _predicate_spikes(upstream_binding, shape.inserted_values)
    if predicate not in {(0, 1), (1, 0)}:
        return _AppendObservation(
            STOP_PREDICATE_STRADDLE,
            "the two appended values did not straddle the fixed predicate",
            residual_certificate,
            upstream_binding,
            shape.cases,
            output_hashes,
            shape.inserted_values,
            predicate,
            None,
            "",
            "",
            shape.comparison_count,
            0,
        )
    if not completeness_gate:
        return _AppendObservation(
            STOP_COMPLETENESS,
            "append-effect completeness cell was disabled",
            residual_certificate,
            upstream_binding,
            shape.cases,
            output_hashes,
            shape.inserted_values,
            predicate,
            None,
            "",
            "",
            shape.comparison_count,
            0,
        )
    if not effect_gate:
        return _AppendObservation(
            STOP_EFFECT_GATE,
            "physical effect-relation cells were disabled",
            residual_certificate,
            upstream_binding,
            shape.cases,
            output_hashes,
            shape.inserted_values,
            predicate,
            None,
            "",
            "",
            shape.comparison_count,
            1,
        )
    status, relation, race_sha = _effect_race(
        outputs, shape.inserted_values, predicate
    )
    probe_payload = {
        "residual_binding_sha256": residual_certificate.sha256,
        "probe_cases": shape.cases,
        "probe_output_sha256s": output_hashes,
        "inserted_values": shape.inserted_values,
        "predicate_spikes": predicate,
    }
    probe_sha = _digest(_NAMESPACE + b"probe\0", probe_payload)
    return _AppendObservation(
        status,
        None if status == READY else "zero or multiple append-effect relations fired",
        residual_certificate,
        upstream_binding,
        shape.cases,
        output_hashes,
        shape.inserted_values,
        predicate,
        relation,
        probe_sha,
        race_sha,
        shape.comparison_count,
        1,
    )


def freeze_raw_sum_scaffold() -> plan_slice.StructuralFactorCertificate:
    """Freeze the explicit collection-to-scalar sum scaffold."""

    return plan_slice.freeze_structural_factor_certificate(
        (
            plan_slice.StructuralOccurrence(None, None, (_ROLE_LIST_INT,), ()),
            plan_slice.StructuralOccurrence(0, 0, (_ROLE_LIST_INT,), (0,)),
        )
    )


def _reducer_scaffold(
    binding: residual.ResidualLiteralRouteBinding,
    action: int,
) -> plan_slice.StructuralFactorCertificate:
    if action == _COUNT_ACTION:
        return residual.freeze_residual_literal_route_scaffold(
            literal_role.decode_int_payload(binding.repeated_payload_cells),
            literal_role.decode_int_payload(binding.residual_payload_cells),
            binding.operand_action,
            binding.selected_route,
        )
    if action == _SUM_ACTION:
        return freeze_raw_sum_scaffold()
    raise ValueError("reducer action lies outside the fixed capacity")


def _authorize_reducers(
    frontier: breadth_v4.PackedGenericTypedTrainBreadthV4,
    binding: residual.ResidualLiteralRouteBinding,
) -> tuple[tuple[tuple[output_cells.TypedOutputEncoding, ...], ...], int]:
    outputs: list[tuple[output_cells.TypedOutputEncoding, ...]] = []
    authorizations = 0
    for action in range(REDUCER_ACTION_CELL_COUNT):
        try:
            status, count, receipt, *_rest = plan_slice._bind_certificate(
                frontier, _reducer_scaffold(binding, action)
            )
        except packed.PackedRelationWorkLimitExceeded as error:
            raise AppendEffectFitStop(
                STOP_WORK_BOUND, "reducer authorization exceeded packed work"
            ) from error
        if status != plan_slice.FRONTIER_READY or count != 1 or receipt is None:
            raise AppendEffectFitStop(
                STOP_PREALLOCATION,
                "one private reducer scaffold did not authorize exactly once",
            )
        plan = plan_slice._authenticate_plan_slice_receipt(frontier, receipt)
        outputs.append(tuple(plan.candidate.eclass.physical_outputs))
        authorizations += 1
    frozen = tuple(outputs)
    _require_distinct_reducer_outputs(frozen)
    return frozen, authorizations


def _require_distinct_reducer_outputs(
    outputs: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
) -> None:
    if len(outputs) != REDUCER_ACTION_CELL_COUNT or outputs[0] == outputs[1]:
        raise AppendEffectFitStop(
            STOP_ACTION_ALIAS, "the two reducer actions have identical consequences"
        )


def _outcome_audit(
    outputs: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
    desired: tuple[output_cells.TypedOutputEncoding, ...],
    observation: _AppendObservation,
    *,
    episode_ordinal: int,
) -> AppendEffectInterventionAudit:
    keys = ("reducer-cell-0", "reducer-cell-1")
    race = semantic_race.select_exact_semantic_vector(keys, outputs, desired)
    if race.status != semantic_race.SELECTED or race.winner_key is None:
        raise AppendEffectFitStop(
            STOP_AMBIGUOUS,
            "reducer interventions produced zero or multiple PASS outcomes",
        )
    if observation.effect_relation is None:
        raise RuntimeError("ready append observation lost its effect relation")
    return AppendEffectInterventionAudit(
        episode_ordinal,
        observation.effect_relation,
        keys.index(race.winner_key),
        observation.effect_race_sha256,
        race.sha256,
        observation.probe_sha256,
        1,
    )


def _learn_policy(
    audits: Sequence[AppendEffectInterventionAudit],
    *,
    outcome_gate: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    labels: list[list[int]] = [[], []]
    for audit in audits:
        labels[audit.effect_relation].append(audit.pass_action)
    if any(values and any(value != values[0] for value in values) for values in labels):
        raise AppendEffectFitStop(
            STOP_CONFLICT, "one effect relation has conflicting PASS actions"
        )
    if any(not values for values in labels) or {values[0] for values in labels} != {0, 1}:
        raise AppendEffectFitStop(
            STOP_INCOMPLETE,
            "both effect relations and both reducer actions need causal support",
        )
    excitatory = np.zeros(
        (EFFECT_RELATION_CELL_COUNT, REDUCER_ACTION_CELL_COUNT), dtype=TICK_DTYPE
    )
    inhibitory = np.zeros_like(excitatory)
    e_ticks = 0
    h_ticks = 0
    if outcome_gate:
        for audit in audits:
            for action in range(REDUCER_ACTION_CELL_COUNT):
                target = excitatory if action == audit.pass_action else inhibitory
                before = int(target[audit.effect_relation, action])
                target[audit.effect_relation, action] = min(TICK_CAP, before + 1)
                delta = int(target[audit.effect_relation, action]) - before
                if action == audit.pass_action:
                    e_ticks += delta
                else:
                    h_ticks += delta
    return excitatory, inhibitory, e_ticks, h_ticks


def _checkpoint_sha(
    dependency_sha: str, excitatory: np.ndarray, inhibitory: np.ndarray
) -> str:
    return sha256(
        _CHECKPOINT_NAMESPACE
        + dependency_sha.encode("ascii")
        + b"\0"
        + _array_frame("append-effect-excitatory", excitatory)
        + _array_frame("append-effect-inhibitory", inhibitory)
    ).hexdigest()


class FrozenCausalAppendEffectReducerV1:
    """Immutable two-relation to two-action causal reducer policy."""

    __slots__ = (
        "_checkpoint_sha256",
        "_excitatory",
        "_inhibitory",
        "_residual_checkpoint",
        "_summary",
    )

    def __init__(
        self,
        residual_checkpoint: residual.FrozenCausalResidualLiteralRouteV1,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        summary: AppendEffectTrainingSummary,
        *,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("append-effect checkpoints are builder-owned")
        if type(residual_checkpoint) is not residual.FrozenCausalResidualLiteralRouteV1:
            raise TypeError("append-effect learning requires the banked checkpoint")
        residual_checkpoint.authenticate()
        shape = (EFFECT_RELATION_CELL_COUNT, REDUCER_ACTION_CELL_COUNT)
        if (
            excitatory.shape != shape
            or inhibitory.shape != shape
            or excitatory.dtype != TICK_DTYPE
            or inhibitory.dtype != TICK_DTYPE
            or np.any(excitatory > TICK_CAP)
            or np.any(inhibitory > TICK_CAP)
        ):
            raise ValueError("append-effect checkpoint geometry changed")
        self._residual_checkpoint = residual_checkpoint
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._checkpoint_sha256 = _checkpoint_sha(
            residual_checkpoint.checkpoint_sha256,
            self._excitatory,
            self._inhibitory,
        )
        self._summary = summary
        if summary.checkpoint_sha256 != self._checkpoint_sha256:
            raise RuntimeError("append-effect training summary changed")
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def residual_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._residual_checkpoint.checkpoint_sha256

    @property
    def training_summary(self) -> AppendEffectTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def learned_state_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._excitatory.shape, self._inhibitory.shape

    def authenticate(self) -> None:
        self._residual_checkpoint.authenticate()
        if (
            self._excitatory.flags.writeable
            or self._inhibitory.flags.writeable
            or _checkpoint_sha(
                self._residual_checkpoint.checkpoint_sha256,
                self._excitatory,
                self._inhibitory,
            )
            != self._checkpoint_sha256
        ):
            raise RuntimeError("append-effect checkpoint authentication failed")

    def bind(
        self,
        span_records: Sequence[Mapping[str, object]],
        literal_records: Sequence[Mapping[str, object]],
        signature_role_record: Mapping[str, object],
        probe_cases: tuple[tuple[list[int]], ...],
        probe_outputs: Sequence[output_cells.TypedOutputEncoding],
        *,
        equality_gate: bool = True,
        literal_role_gate: bool = True,
        order_gate: bool = True,
        upstream_cell_gate: bool = True,
        residual_gate: bool = True,
        residual_cell_gate: bool = True,
        predicate_gate: bool = True,
        effect_gate: bool = True,
        completeness_gate: bool = True,
        cell_gate: bool = True,
    ) -> AppendEffectReducerCertificate:
        self.authenticate()
        gates = (
            equality_gate,
            literal_role_gate,
            order_gate,
            upstream_cell_gate,
            residual_gate,
            residual_cell_gate,
            predicate_gate,
            effect_gate,
            completeness_gate,
            cell_gate,
        )
        if any(type(value) is not bool for value in gates):
            raise TypeError("append-effect gates must be exact booleans")
        observation = _observe(
            self._residual_checkpoint,
            span_records,
            literal_records,
            signature_role_record,
            probe_cases,
            probe_outputs,
            equality_gate=equality_gate,
            literal_role_gate=literal_role_gate,
            order_gate=order_gate,
            upstream_cell_gate=upstream_cell_gate,
            residual_gate=residual_gate,
            residual_cell_gate=residual_cell_gate,
            predicate_gate=predicate_gate,
            effect_gate=effect_gate,
            completeness_gate=completeness_gate,
        )
        if observation.status != READY or observation.effect_relation is None:
            return _issue_binding(self, observation, None, observation.status, observation.reason)
        scores = (
            self._excitatory[observation.effect_relation].astype(np.int16)
            - self._inhibitory[observation.effect_relation].astype(np.int16)
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
            for index in np.flatnonzero(_ACTION_RACE.accumulated_spike_mask(currents))
        )
        if not spikes:
            return _issue_binding(
                self,
                observation,
                None,
                STOP_ZERO,
                "no learned reducer action crossed the physical race",
            )
        if len(spikes) != 1:
            return _issue_binding(
                self,
                observation,
                None,
                STOP_AMBIGUOUS,
                "multiple learned reducer actions crossed together",
            )
        return _issue_binding(self, observation, spikes[0], READY, None)


def _work_payload(value: AppendEffectBindingWork) -> dict[str, object]:
    return {field: getattr(value, field) for field in value._fields if field != "sha256"}


def _receipt_payload(value: _BindingReceipt) -> dict[str, object]:
    return {
        field: getattr(value, field)
        for field in value._fields
        if field not in {"authority", "residual_certificate", "work", "sha256"}
    } | {
        "residual_certificate_sha256": value.residual_certificate.sha256,
        "work_sha256": value.work.sha256,
    }


def _issue_binding(
    checkpoint: FrozenCausalAppendEffectReducerV1,
    observation: _AppendObservation,
    selected_action: int | None,
    status: str,
    reason: str | None,
) -> AppendEffectReducerCertificate:
    binding = observation.residual_binding
    action_steps = ACTION_RACE_CELL_STEPS if observation.effect_relation is not None else 0
    partial_work = AppendEffectBindingWork(
        observation.residual_certificate.work.stacked_binding_cell_steps,
        observation.comparison_count,
        observation.comparison_count
        * literal_role.EQUALITY_CELL_STEPS_PER_COMPARISON,
        observation.completeness_spikes * COMPLETENESS_CELL_STEPS,
        SEMANTIC_RACE_CELL_STEPS if observation.effect_race_sha256 else 0,
        action_steps,
        0,
        "",
    )
    stacked = (
        partial_work.upstream_binding_cell_steps
        + partial_work.probe_comparison_cell_steps
        + partial_work.completeness_cell_steps
        + partial_work.effect_race_cell_steps
        + partial_work.reducer_action_race_cell_steps
    )
    partial_work = partial_work._replace(stacked_binding_cell_steps=stacked)
    work = partial_work._replace(
        sha256=_digest(_WORK_NAMESPACE, _work_payload(partial_work))
    )
    partial = _BindingReceipt(
        _BINDING_AUTHORITY,
        checkpoint.checkpoint_sha256,
        status,
        reason,
        observation.residual_certificate,
        () if binding is None else binding.repeated_payload_cells,
        () if binding is None else binding.repeated_token_indices,
        () if binding is None else binding.residual_payload_cells,
        None if binding is None else binding.residual_token_index,
        None if binding is None else binding.operand_action,
        None if binding is None else binding.selected_route,
        observation.effect_relation,
        selected_action,
        observation.probe_cases,
        observation.inserted_values,
        observation.predicate_spikes,
        observation.probe_sha256,
        observation.effect_race_sha256,
        work,
        "",
    )
    receipt = partial._replace(
        sha256=_digest(_BINDING_NAMESPACE, _receipt_payload(partial))
    )
    certificate = AppendEffectReducerCertificate(
        receipt, construction_token=_CONSTRUCTION_TOKEN
    )
    _ISSUED_BINDINGS.add(certificate)
    return certificate


def _authenticate_binding(certificate: AppendEffectReducerCertificate) -> _BindingReceipt:
    if type(certificate) is not AppendEffectReducerCertificate:
        raise TypeError("append-effect binding must be an exact certificate")
    if certificate not in _ISSUED_BINDINGS:
        raise RuntimeError("append-effect certificate was not issued here")
    receipt = certificate._receipt
    residual._authenticate_binding(receipt.residual_certificate)
    if (
        receipt.authority is not _BINDING_AUTHORITY
        or receipt.work.sha256 != _digest(_WORK_NAMESPACE, _work_payload(receipt.work))
        or receipt.sha256 != _digest(_BINDING_NAMESPACE, _receipt_payload(receipt))
    ):
        raise RuntimeError("append-effect certificate authentication failed")
    return receipt


def require_append_effect_reducer_binding(
    certificate: AppendEffectReducerCertificate,
) -> AppendEffectReducerBinding:
    receipt = _authenticate_binding(certificate)
    if (
        receipt.status != READY
        or receipt.residual_token_index is None
        or receipt.operand_action is None
        or receipt.residual_route is None
        or receipt.effect_relation is None
        or receipt.selected_action is None
    ):
        raise RuntimeError(receipt.stop_reason or receipt.status)
    return AppendEffectReducerBinding(
        receipt.repeated_payload_cells,
        receipt.repeated_token_indices,
        receipt.residual_payload_cells,
        receipt.residual_token_index,
        receipt.operand_action,
        receipt.residual_route,
        receipt.effect_relation,
        receipt.selected_action,
        receipt.probe_cases,
        receipt.inserted_values,
        receipt.predicate_spikes,
        receipt.probe_sha256,
        receipt.residual_certificate.sha256,
        receipt.sha256,
    )


def fit_causal_append_effect_reducer_v1(
    residual_checkpoint: residual.FrozenCausalResidualLiteralRouteV1,
    episodes: tuple[CausalAppendEffectEpisode, ...],
    *,
    signature: TypedSignature,
    probe_cases: tuple[tuple[list[int]], ...],
    literals: tuple[int, ...],
    max_cost: int,
    outcome_gate: bool = True,
) -> FrozenCausalAppendEffectReducerV1:
    """Fit the constant append-effect to reducer policy from consequences."""

    if type(residual_checkpoint) is not residual.FrozenCausalResidualLiteralRouteV1:
        raise TypeError("append-effect fit requires the banked checkpoint type")
    residual_checkpoint.authenticate()
    dependency_sha = residual_checkpoint.checkpoint_sha256
    if type(episodes) is not tuple or any(
        type(episode) is not CausalAppendEffectEpisode for episode in episodes
    ):
        raise TypeError("episodes must be an exact tuple of append-effect episodes")
    if not MIN_CAUSAL_EPISODES <= len(episodes) <= MAX_CAUSAL_EPISODES:
        raise AppendEffectFitStop(
            STOP_INCOMPLETE, "append-effect learning requires two to four episodes"
        )
    if not _signature_ok(signature) or max_cost != FIXED_MAX_COST:
        raise ValueError("append-effect scaffolds require List[Int] -> Int at cost 8")
    if type(literals) is not tuple or any(type(value) is not int for value in literals):
        raise TypeError("literals must be an exact Int tuple")
    if type(outcome_gate) is not bool:
        raise TypeError("outcome_gate must be an exact boolean")

    observations: list[_AppendObservation] = []
    desired_rows: list[tuple[output_cells.TypedOutputEncoding, ...]] = []
    for episode in episodes:
        desired = _probe_outputs(episode.probe_outputs)
        observation = _observe(
            residual_checkpoint,
            episode.span_records,
            episode.literal_records,
            episode.signature_role_record,
            probe_cases,
            desired,
            equality_gate=True,
            literal_role_gate=True,
            order_gate=True,
            upstream_cell_gate=True,
            residual_gate=True,
            residual_cell_gate=True,
            predicate_gate=True,
            effect_gate=True,
            completeness_gate=True,
        )
        if observation.status != READY:
            raise AppendEffectFitStop(
                observation.status,
                observation.reason or "append-effect observation stopped",
            )
        observations.append(observation)
        desired_rows.append(desired)

    frontier = breadth_v4.PackedGenericTypedTrainBreadthV4.build(
        signature, probe_cases, literals, max_cost
    )
    if frontier.status != breadth_v4.FRONTIER_READY:
        raise AppendEffectFitStop(
            STOP_WORK_BOUND, "ordinary packed append-effect frontier is not ready"
        )
    audits: list[AppendEffectInterventionAudit] = []
    authorizations = 0
    for ordinal, (observation, desired) in enumerate(
        zip(observations, desired_rows, strict=True)
    ):
        if observation.residual_binding is None:
            raise RuntimeError("ready observation lost its residual binding")
        reducer_outputs, count = _authorize_reducers(
            frontier, observation.residual_binding
        )
        authorizations += count
        audits.append(
            _outcome_audit(
                reducer_outputs, desired, observation, episode_ordinal=ordinal
            )
        )
    if (
        authorizations > MAX_PACKED_AUTHORIZATIONS
        or len(audits) > MAX_EFFECT_RACES
        or len(audits) > MAX_OUTCOME_RACES
    ):
        raise AppendEffectFitStop(
            STOP_WORK_BOUND, "append-effect intervention count exceeded its cap"
        )
    excitatory, inhibitory, e_ticks, h_ticks = _learn_policy(
        audits, outcome_gate=outcome_gate
    )
    if residual_checkpoint.checkpoint_sha256 != dependency_sha:
        raise RuntimeError("banked residual checkpoint changed during fit")
    checkpoint_sha = _checkpoint_sha(dependency_sha, excitatory, inhibitory)
    comparisons = sum(item.comparison_count for item in observations)
    summary = AppendEffectTrainingSummary(
        len(episodes),
        dependency_sha,
        MAX_PACKED_BUILDS,
        authorizations,
        len(audits),
        len(audits),
        sum(
            item.residual_certificate.work.stacked_binding_cell_steps
            for item in observations
        ),
        comparisons,
        comparisons * literal_role.EQUALITY_CELL_STEPS_PER_COMPARISON,
        sum(item.completeness_spikes for item in observations),
        0,
        outcome_gate,
        e_ticks,
        h_ticks,
        int(excitatory.max()),
        int(inhibitory.max()),
        TRAINABLE_STATE_BYTES,
        frontier.work.sha256,
        tuple(audits),
        checkpoint_sha,
    )
    return FrozenCausalAppendEffectReducerV1(
        residual_checkpoint,
        excitatory,
        inhibitory,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def freeze_outcome_off_control(
    checkpoint: FrozenCausalAppendEffectReducerV1,
) -> FrozenCausalAppendEffectReducerV1:
    """Silence the sole PASS/FAIL-to-reducer plasticity gate."""

    if type(checkpoint) is not FrozenCausalAppendEffectReducerV1:
        raise TypeError("outcome-off control requires the exact checkpoint type")
    checkpoint.authenticate()
    zeros = np.zeros(
        (EFFECT_RELATION_CELL_COUNT, REDUCER_ACTION_CELL_COUNT), dtype=TICK_DTYPE
    )
    checkpoint_sha = _checkpoint_sha(
        checkpoint.residual_checkpoint_sha256, zeros, zeros
    )
    summary = checkpoint.training_summary._replace(
        outcome_gate_on=False,
        excitatory_update_ticks=0,
        inhibitory_update_ticks=0,
        maximum_excitatory_tick=0,
        maximum_inhibitory_tick=0,
        checkpoint_sha256=checkpoint_sha,
    )
    return FrozenCausalAppendEffectReducerV1(
        checkpoint._residual_checkpoint,
        zeros,
        zeros,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


CONFIG_RECORD = MappingProxyType(
    {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "residual_config_sha256": residual.CONFIG_SHA256,
        "base_cell_count": residual.TOTAL_CELL_COUNT,
        "new_cell_count": NEW_CELL_COUNT,
        "total_cell_count": TOTAL_CELL_COUNT,
        "effect_relation_cell_count": EFFECT_RELATION_CELL_COUNT,
        "reducer_action_cell_count": REDUCER_ACTION_CELL_COUNT,
        "completeness_cell_count": COMPLETENESS_CELL_COUNT,
        "trainable_state_bytes": TRAINABLE_STATE_BYTES,
        "probe_case_count": PROBE_CASE_COUNT,
        "semantic_race_cell_steps": SEMANTIC_RACE_CELL_STEPS,
        "action_race_cell_steps": ACTION_RACE_CELL_STEPS,
        "public_train_input_count": 0,
        "canonical_panel_spend_count": 0,
        "dev_input_count": 0,
        "m0_input_count": 0,
        "c1_input_count": 0,
        "g0_input_count": 0,
        "protected_input_count": 0,
    }
)
CONFIG_SHA256 = _digest(_NAMESPACE + b"config\0", CONFIG_RECORD)

__all__ = [
    "ACTION_RACE_CELL_STEPS",
    "CLAIM_BOUNDARY",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "EFFECT_RELATION_CELL_COUNT",
    "NEW_CELL_COUNT",
    "PAYLOAD_EFFECT_RELATION",
    "PREDICATE_EFFECT_RELATION",
    "READY",
    "REDUCER_ACTION_CELL_COUNT",
    "SEMANTIC_RACE_CELL_STEPS",
    "STOP_ACTION_ALIAS",
    "STOP_AMBIGUOUS",
    "STOP_APPEND_SHAPE",
    "STOP_COMPLETENESS",
    "STOP_CONFLICT",
    "STOP_DEPENDENCY",
    "STOP_EFFECT_GATE",
    "STOP_EFFECT_TIE",
    "STOP_EFFECT_ZERO",
    "STOP_EQUALITY_GATE",
    "STOP_INCOMPLETE",
    "STOP_NONDIAGNOSTIC",
    "STOP_PREALLOCATION",
    "STOP_PREDICATE_GATE",
    "STOP_PREDICATE_STRADDLE",
    "STOP_PROBE_FINAL_OVERLAP",
    "STOP_UNSUPPORTED",
    "STOP_WORK_BOUND",
    "STOP_ZERO",
    "TOTAL_CELL_COUNT",
    "TRAINABLE_STATE_BYTES",
    "AppendEffectBindingWork",
    "AppendEffectFitStop",
    "AppendEffectReducerBinding",
    "AppendEffectReducerCertificate",
    "AppendEffectTrainingSummary",
    "CausalAppendEffectEpisode",
    "FrozenCausalAppendEffectReducerV1",
    "fit_causal_append_effect_reducer_v1",
    "freeze_outcome_off_control",
    "freeze_raw_sum_scaffold",
    "require_append_effect_reducer_binding",
]
