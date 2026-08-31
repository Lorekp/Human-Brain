"""Causal predicate-gated fold over one learned two-cell action race.

Three physical A/B/A-prime consequence lessons fit an immutable 2x2 uint8
E/H checkpoint.  A predicate Bool winner is treated only as an anonymous
physical feature cell.  For every bound item occurrence the checkpoint races
two anonymous action cells; only that physical winner dispatches either the
banked ADD intervention or the HOLD intervention.

The packed construction path is::

    less-than(item, literal)
      -> learned predicate-gated-item(predicate, same item)
      -> value-binder -> map(collection) -> sum

The predicate and actuator item slots are joined by one packed equality
relation, so their occurrence identity is physical rather than a host check.
Build sees no desired result, action label, operator choice, task family, or
structural certificate.  A private operator-name-free certificate authorizes
one proof, then a selected-only renderer lowers the gate to the existing typed
``if`` primitive with the additive identity as its fixed HOLD actuator.

This is a self-posed mechanism result, not parsing, arbitrary filter/fold
induction, SWE evidence, or a general coding capability claim.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from hashlib import sha256
from typing import NamedTuple, cast

import numpy as np

from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_frontier_full_typed_v1 as full_typed
from experiments.r3_pilot import typed_semantic_packed_frontier_v2 as packed_v2
from experiments.r3_pilot import typed_semantic_packed_frontier_v3 as packed_v3
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v3 as breadth_v3
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v4 as breadth_v4
from experiments.r3_pilot import typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice
from humanbrain import typed_program
from humanbrain.resting_glif_race import (
    CompiledRestingGlifRace,
    RestingGlifRaceFailure,
)
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

SCHEMA = "CAUSAL_PREDICATE_GATED_FOLD_V1"
READY = f"{SCHEMA}_READY"
STOP_CONSEQUENCE = f"STOP_{SCHEMA}_CONSEQUENCE"
STOP_LAW = f"STOP_{SCHEMA}_LAW"
STOP_FEATURE = f"STOP_{SCHEMA}_FEATURE"
STOP_ACTION_RACE = f"STOP_{SCHEMA}_ACTION_RACE"
STOP_FRONTIER = f"STOP_{SCHEMA}_FRONTIER"
STOP_PLAN_SLICE = f"STOP_{SCHEMA}_PLAN_SLICE"
STOP_WORK_BOUND = f"STOP_{SCHEMA}_WORK_BOUND"
STOP_WITNESS_NOT_SINGLETON = f"STOP_{SCHEMA}_WITNESS_NOT_SINGLETON"

CLAIM_BOUNDARY = (
    "one self-posed A/B/A-prime consequence-learned predicate-cell to "
    "ADD/HOLD action race, reused for every physically joined item occurrence "
    "inside one exact less-than/map/sum witness; the two intervention "
    "scaffolds, threshold literal, typed contexts, fixed zero actuator, "
    "structural certificate, compiler, and case framing remain explicit, so "
    "predicate/operator discovery, parsing, SWE, and coding capability remain "
    "unproved"
)

STATE_SHAPE = (2, 2)
TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = int(np.iinfo(TICK_DTYPE).max)
NEW_PERSISTENT_STATE_BYTES = 2 * int(np.prod(STATE_SHAPE)) * TICK_DTYPE.itemsize
REUSED_TYPED_BOOL_CELL_COUNT = 2
NEW_ACTION_CELL_COUNT = 2
FIXED_MAX_COST = 9
WITNESS_AST_COST = 9
ACTION_RACE_STEP_COUNT = 30
ACTION_RACE_CELL_STEPS = NEW_ACTION_CELL_COUNT * ACTION_RACE_STEP_COUNT
OUTPUT_CURRENT_PA = 10_000.0
DESIRED_READS_DURING_BUILD = 0
ACTION_LABEL_INPUTS_DURING_BUILD = 0

_INT = ProgramType.Int()
_BOOL = ProgramType.Bool()
_LIST_INT = ProgramType.List(_INT)
_ROLE_INT = ("role", "Int")
_ROLE_LIST_INT = ("role", "List[Int]")
_LITERAL_INT = ("literal", "Int")
_ADD_ORDINAL = 0
_HOLD_ORDINAL = 1
_ACTION_COUNT = 2
_LESS_THAN_ACTION = typed_program.FactorAction(
    (), "APPLY2", "less-than", "SELECTED", "SCOPE0", None
)
_ADD_ACTION = typed_program.FactorAction(
    (), "APPLY2", "add", "SELECTED", "SCOPE0", None
)
_ZERO_EVENT = output_cells.TypedOutputCellBank.encode(
    0, declared_type=output_cells.OutputType.INT
)
_ACTION_RACE = CompiledRestingGlifRace.compile(step_count=ACTION_RACE_STEP_COUNT)
_NAMESPACE = b"causal-predicate-gated-fold-v1\0"
_CHECKPOINT_AUTHORITY = object()
_ISSUED_CHECKPOINTS: dict[int, FrozenPredicateGateV1] = {}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(label: bytes, value: object) -> str:
    return sha256(_NAMESPACE + label + b"\0" + _canonical_json(value)).hexdigest()


def list_int_to_int_signature() -> TypedSignature:
    return TypedSignature(
        "aggregate",
        (ProgramParameter("numbers", _LIST_INT),),
        _INT,
    )


class PredicateGateLesson(NamedTuple):
    """One physical predicate event and its visible accumulator consequence."""

    predicate_event: output_cells.TypedOutputEncoding
    accumulator_before: output_cells.TypedOutputEncoding
    item_event: output_cells.TypedOutputEncoding
    observed_after: output_cells.TypedOutputEncoding


class PredicateGateFitStop(RuntimeError):
    def __init__(self, status: str, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class PredicateGatedFoldBuildStop(RuntimeError):
    def __init__(self, status: str, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class _GateInferenceStop(RuntimeError):
    def __init__(self, status: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _require_int_event(
    event: output_cells.TypedOutputEncoding,
    label: str,
) -> output_cells.TypedOutputEncoding:
    event = output_cells.TypedOutputCellBank.validate(event)
    if event.output_type is not output_cells.OutputType.INT or not event.supported:
        raise ValueError(f"{label} must be one supported physical Int event")
    return event


def _normalize_cell_ids(label: str, values: Sequence[int]) -> tuple[int, int]:
    result = tuple(values)
    if (
        len(result) != 2
        or any(type(value) is not int or value < 0 for value in result)
        or len(set(result)) != 2
    ):
        raise ValueError(f"{label} must contain two distinct nonnegative cell ids")
    return cast(tuple[int, int], result)


def observe_predicate_gate_lesson(
    *,
    accumulator_before: int,
    item: int,
    threshold: int,
    observed_after: int,
) -> PredicateGateLesson:
    """Project one lesson without decoding or branching on its predicate."""

    values = (accumulator_before, item, threshold, observed_after)
    if any(type(value) is not int for value in values):
        raise TypeError("lesson surfaces must be exact integers")
    before = output_cells.TypedOutputCellBank.encode(
        accumulator_before, declared_type=output_cells.OutputType.INT
    )
    item_event = output_cells.TypedOutputCellBank.encode(
        item, declared_type=output_cells.OutputType.INT
    )
    threshold_event = output_cells.TypedOutputCellBank.encode(
        threshold, declared_type=output_cells.OutputType.INT
    )
    after = output_cells.TypedOutputCellBank.encode(
        observed_after, declared_type=output_cells.OutputType.INT
    )
    operator = breadth_v4.operator_successor.train_public_typed_operator_successor()[0]
    before_checkpoint = operator.live_checkpoint_sha256
    prediction = operator.predict(
        _require_int_event(item_event, "item"),
        _LESS_THAN_ACTION,
        scopes=(_require_int_event(threshold_event, "threshold"),),
    )
    if operator.live_checkpoint_sha256 != before_checkpoint:
        raise RuntimeError("frozen typed operator changed while observing predicate")
    predicate = output_cells.TypedOutputCellBank.validate(prediction.output)
    if predicate.output_type is not output_cells.OutputType.BOOL or not predicate.supported:
        raise ValueError("lesson predicate must be one supported physical Bool event")
    return PredicateGateLesson(
        predicate,
        _require_int_event(before, "accumulator_before"),
        _require_int_event(item_event, "item"),
        _require_int_event(after, "observed_after"),
    )


def _feature_row(
    predicate: output_cells.TypedOutputEncoding,
    feature_cell_ids: tuple[int, int],
    *,
    cell_gate: bool,
) -> tuple[int, int]:
    if not cell_gate:
        raise _GateInferenceStop(STOP_FEATURE, "predicate feature cells were disabled")
    predicate = output_cells.TypedOutputCellBank.validate(predicate)
    if predicate.output_type is not output_cells.OutputType.BOOL or not predicate.supported:
        raise _GateInferenceStop(STOP_FEATURE, "predicate lost its physical Bool event")
    winner = predicate.factor_winner(output_cells.BOOL_FACTOR_SLOT)
    if type(winner) is not int or not 0 <= winner < len(feature_cell_ids):
        raise _GateInferenceStop(STOP_FEATURE, "predicate feature race had no exact cell")
    physical_cell_id = feature_cell_ids[winner]
    row = tuple(sorted(feature_cell_ids)).index(physical_cell_id)
    return row, physical_cell_id


def _action_column(
    action_ordinal: int,
    action_cell_ids: tuple[int, int],
) -> tuple[int, int]:
    if type(action_ordinal) is not int or not 0 <= action_ordinal < _ACTION_COUNT:
        raise ValueError("action ordinal lies outside the intervention roster")
    physical_cell_id = action_cell_ids[action_ordinal]
    column = tuple(sorted(action_cell_ids)).index(physical_cell_id)
    return column, physical_cell_id


def _consequence_action(
    operator: breadth_v4.operator_successor.FrozenTypedOperatorSuccessor,
    lesson: PredicateGateLesson,
    action_cell_ids: tuple[int, int],
    *,
    consequence_gate: bool,
) -> tuple[int, int, str]:
    if not consequence_gate:
        raise PredicateGateFitStop(
            STOP_CONSEQUENCE, "physical action consequences were disabled"
        )
    before = _require_int_event(lesson.accumulator_before, "accumulator_before")
    item = _require_int_event(lesson.item_event, "item")
    desired = _require_int_event(lesson.observed_after, "observed_after")
    add_prediction = operator.predict(before, _ADD_ACTION, scopes=(item,))
    add_output = _require_int_event(add_prediction.output, "ADD consequence")
    hold_output = before
    if add_output.sha256 == hold_output.sha256:
        raise PredicateGateFitStop(
            STOP_CONSEQUENCE, "ADD and HOLD interventions were physically aliased"
        )
    keys = tuple(f"physical-action-cell-{value}" for value in action_cell_ids)
    race = semantic_race.select_exact_semantic_vector(
        keys,
        ((add_output,), (hold_output,)),
        (desired,),
    )
    if race.status != semantic_race.SELECTED or race.winner_key is None:
        raise PredicateGateFitStop(
            STOP_CONSEQUENCE,
            "ADD/HOLD interventions did not produce one exact physical consequence",
        )
    action_ordinal = keys.index(race.winner_key)
    _column, action_cell_id = _action_column(action_ordinal, action_cell_ids)
    return action_ordinal, action_cell_id, race.sha256


def _checkpoint_payload(checkpoint: FrozenPredicateGateV1) -> dict[str, object]:
    return {
        "excitatory": checkpoint._excitatory_bytes.hex(),
        "inhibitory": checkpoint._inhibitory_bytes.hex(),
        "feature_cell_ids": checkpoint._feature_cell_ids,
        "action_cell_ids": checkpoint._action_cell_ids,
        "feature_trace": checkpoint._feature_trace,
        "action_trace": checkpoint._action_trace,
        "consequence_roots": checkpoint._consequence_roots,
        "operator_checkpoint_sha256": checkpoint._operator_checkpoint_sha256,
        "excitatory_ticks": checkpoint._excitatory_ticks,
        "inhibitory_ticks": checkpoint._inhibitory_ticks,
    }


class FrozenPredicateGateV1:
    """Identity-authenticated immutable two-feature/two-action E/H checkpoint."""

    __slots__ = (
        "_action_cell_ids",
        "_action_trace",
        "_authority",
        "_consequence_roots",
        "_excitatory_bytes",
        "_excitatory_ticks",
        "_feature_cell_ids",
        "_feature_trace",
        "_inhibitory_bytes",
        "_inhibitory_ticks",
        "_operator_checkpoint_sha256",
        "_sha256",
    )

    def __init__(self) -> None:
        raise TypeError("use fit_causal_predicate_gate")

    @property
    def excitatory(self) -> np.ndarray:
        self._authenticate()
        return np.frombuffer(self._excitatory_bytes, dtype=TICK_DTYPE).reshape(STATE_SHAPE)

    @property
    def inhibitory(self) -> np.ndarray:
        self._authenticate()
        return np.frombuffer(self._inhibitory_bytes, dtype=TICK_DTYPE).reshape(STATE_SHAPE)

    @property
    def feature_cell_ids(self) -> tuple[int, int]:
        self._authenticate()
        return self._feature_cell_ids

    @property
    def action_cell_ids(self) -> tuple[int, int]:
        self._authenticate()
        return self._action_cell_ids

    @property
    def feature_trace(self) -> tuple[int, int, int]:
        self._authenticate()
        return self._feature_trace

    @property
    def action_trace(self) -> tuple[int, int, int]:
        self._authenticate()
        return self._action_trace

    @property
    def sha256(self) -> str:
        self._authenticate()
        return self._sha256

    @property
    def persistent_state_bytes(self) -> int:
        return len(self._excitatory_bytes) + len(self._inhibitory_bytes)

    def _authenticate(self) -> None:
        if (
            self._authority is not _CHECKPOINT_AUTHORITY
            or _ISSUED_CHECKPOINTS.get(id(self)) is not self
            or len(self._excitatory_bytes) != int(np.prod(STATE_SHAPE))
            or len(self._inhibitory_bytes) != int(np.prod(STATE_SHAPE))
            or self._sha256 != _digest(b"checkpoint", _checkpoint_payload(self))
        ):
            raise RuntimeError("predicate gate checkpoint authentication failed")


def fit_causal_predicate_gate(
    lessons: Iterable[PredicateGateLesson],
    *,
    feature_cell_ids: Sequence[int] = (17, 31),
    action_cell_ids: Sequence[int] = (47, 23),
    feature_cell_gate: bool = True,
    consequence_gate: bool = True,
) -> FrozenPredicateGateV1:
    """Fit the minimal A/B/A-prime law solely from action consequences."""

    if type(feature_cell_gate) is not bool or type(consequence_gate) is not bool:
        raise TypeError("causal gates must be exact bools")
    frozen_lessons = tuple(lessons)
    if len(frozen_lessons) != 3 or any(
        type(lesson) is not PredicateGateLesson for lesson in frozen_lessons
    ):
        raise ValueError("predicate gate fit requires exactly three A/B/A-prime lessons")
    feature_ids = _normalize_cell_ids("feature_cell_ids", feature_cell_ids)
    action_ids = _normalize_cell_ids("action_cell_ids", action_cell_ids)
    excitatory = np.zeros(STATE_SHAPE, dtype=TICK_DTYPE)
    inhibitory = np.zeros(STATE_SHAPE, dtype=TICK_DTYPE)
    feature_trace: list[int] = []
    action_trace: list[int] = []
    consequence_roots: list[str] = []
    excitatory_ticks = 0
    inhibitory_ticks = 0
    operator = breadth_v4.operator_successor.train_public_typed_operator_successor()[0]
    operator_before = operator.live_checkpoint_sha256
    if operator_before != operator.checkpoint_sha256:
        raise RuntimeError("typed operator checkpoint changed before predicate fit")

    for lesson in frozen_lessons:
        try:
            row, feature_cell_id = _feature_row(
                lesson.predicate_event,
                feature_ids,
                cell_gate=feature_cell_gate,
            )
        except _GateInferenceStop as error:
            raise PredicateGateFitStop(error.status, error.detail) from error
        action_ordinal, action_cell_id, consequence_root = _consequence_action(
            operator,
            lesson,
            action_ids,
            consequence_gate=consequence_gate,
        )
        column, _physical_action = _action_column(action_ordinal, action_ids)
        for candidate in range(_ACTION_COUNT):
            state = excitatory if candidate == column else inhibitory
            before = int(state[row, candidate])
            state[row, candidate] = min(TICK_CAP, before + 1)
            delta = int(state[row, candidate]) - before
            if candidate == column:
                excitatory_ticks += delta
            else:
                inhibitory_ticks += delta
        feature_trace.append(feature_cell_id)
        action_trace.append(action_cell_id)
        consequence_roots.append(consequence_root)

    if operator.live_checkpoint_sha256 != operator_before:
        raise RuntimeError("frozen typed operator changed during predicate fit")
    if not (
        feature_trace[0] == feature_trace[2]
        and feature_trace[0] != feature_trace[1]
        and action_trace[0] == action_trace[2]
        and action_trace[0] != action_trace[1]
        and set(feature_trace) == set(feature_ids)
        and set(action_trace) == set(action_ids)
    ):
        raise PredicateGateFitStop(
            STOP_LAW, "lessons did not express one physical A/B/A-prime action law"
        )

    checkpoint = object.__new__(FrozenPredicateGateV1)
    checkpoint._authority = _CHECKPOINT_AUTHORITY
    checkpoint._excitatory_bytes = excitatory.tobytes(order="C")
    checkpoint._inhibitory_bytes = inhibitory.tobytes(order="C")
    checkpoint._feature_cell_ids = feature_ids
    checkpoint._action_cell_ids = action_ids
    checkpoint._feature_trace = cast(tuple[int, int, int], tuple(feature_trace))
    checkpoint._action_trace = cast(tuple[int, int, int], tuple(action_trace))
    checkpoint._consequence_roots = tuple(consequence_roots)
    checkpoint._operator_checkpoint_sha256 = operator_before
    checkpoint._excitatory_ticks = excitatory_ticks
    checkpoint._inhibitory_ticks = inhibitory_ticks
    checkpoint._sha256 = _digest(b"checkpoint", _checkpoint_payload(checkpoint))
    _ISSUED_CHECKPOINTS[id(checkpoint)] = checkpoint
    checkpoint._authenticate()
    return checkpoint


def _race_action(
    checkpoint: FrozenPredicateGateV1,
    row: int,
    *,
    action_cell_gate: bool,
) -> tuple[int, str]:
    checkpoint._authenticate()
    if type(action_cell_gate) is not bool:
        raise TypeError("action_cell_gate must be an exact bool")
    excitatory = checkpoint.excitatory
    inhibitory = checkpoint.inhibitory
    scores = tuple(
        int(excitatory[row, column]) - int(inhibitory[row, column])
        for column in range(_ACTION_COUNT)
    )
    currents = np.asarray(
        [max(score, 0) * OUTPUT_CURRENT_PA if action_cell_gate else 0.0 for score in scores],
        dtype=np.float64,
    )
    try:
        race = _ACTION_RACE.race_many(currents[None, :], require_unique=False)[0]
    except RestingGlifRaceFailure as error:
        raise _GateInferenceStop(
            STOP_ACTION_RACE, "learned action cells produced no physical spike"
        ) from error
    if race.simultaneous_spike_candidates != 1:
        raise _GateInferenceStop(
            STOP_ACTION_RACE, "learned action cells produced an ambiguous spike"
        )
    ordered_cell_ids = tuple(sorted(checkpoint.action_cell_ids))
    winner_cell_id = ordered_cell_ids[int(race.winner_cell)]
    action_ordinal = checkpoint.action_cell_ids.index(winner_cell_id)
    root = _digest(
        b"action-race",
        {
            "checkpoint_sha256": checkpoint.sha256,
            "row": row,
            "scores": scores,
            "action_cell_gate": action_cell_gate,
            "winner_cell_id": winner_cell_id,
            "spike_step": race.spike_step,
            "simultaneous": race.simultaneous_spike_candidates,
        },
    )
    return action_ordinal, root


class _EqualityFactorizedSemanticPartitionWorkset(
    breadth_v4._FactorizedSemanticPartitionWorkset
):
    """Retain one generic pushed equality relation in an otherwise stock V4 workset."""

    def _product_domains(
        self,
        plan: packed_v2._FamilyPlan,
        child_roots: tuple[str, ...],
        child_schemas: tuple[tuple[packed.TypedSlot, ...], ...],
        *,
        constraints: tuple[packed.SlotConstraint, ...] | None = None,
    ) -> tuple[tuple[int, ...], ...] | None:
        checks = plan.constraints if constraints is None else constraints
        if len(checks) == 1 and checks[0].operator == "eq":
            check = checks[0]
            if check.right_slot is None:
                raise RuntimeError("predicate occurrence equality lost its right slot")
            domains = super()._product_domains(
                plan,
                child_roots,
                child_schemas,
                constraints=(),
            )
            if domains is None:
                return None
            self.session.charge_workset("proof-plan-constraint-decision")
            left_index = plan.schema.index(check.left)
            right_index = plan.schema.index(check.right_slot)
            common = tuple(sorted(set(domains[left_index]) & set(domains[right_index])))
            if not common:
                return None
            constrained = list(domains)
            constrained[left_index] = common
            constrained[right_index] = common
            return tuple(constrained)
        return super()._product_domains(
            plan,
            child_roots,
            child_schemas,
            constraints=constraints,
        )


class _PredicateGateListIntBuilder(breadth_v4._ListIntBuilder):
    """The banked V4 builder plus one learned shared-occurrence gate family."""

    def __init__(
        self,
        signature: TypedSignature,
        supplied_case_arguments: Iterable[tuple[object, ...]],
        threshold: int,
        checkpoint: FrozenPredicateGateV1,
        *,
        predicate_cell_gate: bool,
        action_cell_gate: bool,
    ) -> None:
        checkpoint._authenticate()
        if type(threshold) is not int:
            raise TypeError("threshold must be an exact integer")
        if type(predicate_cell_gate) is not bool or type(action_cell_gate) is not bool:
            raise TypeError("inference gates must be exact bools")
        operator = breadth_v4.operator_successor.train_public_typed_operator_successor()[0]
        if operator.checkpoint_sha256 != checkpoint._operator_checkpoint_sha256:
            raise RuntimeError("predicate gate and typed operator checkpoints diverged")
        super().__init__(
            signature,
            supplied_case_arguments,
            (threshold,),
            FIXED_MAX_COST,
            operator,
        )
        self.workset = _EqualityFactorizedSemanticPartitionWorkset(self.session)
        self.authenticated_components = None
        self._checkpoint = checkpoint
        self._predicate_cell_gate = predicate_cell_gate
        self._action_cell_gate = action_cell_gate
        self._gate_race_roots: list[str] = []
        self._base_predicate: packed_v2.PackedPartitionHandle | None = None

    def _family(
        self,
        production: str,
        children: tuple[packed_v2.PackedPartitionHandle, ...],
        *,
        context_role: str,
        environment_type_labels: tuple[str, ...],
        result_type_label: str,
        root_candidate: bool,
        constraint_kind: str | None = None,
    ) -> packed_v2.PackedPartitionHandle:
        handle = super()._family(
            production,
            children,
            context_role=context_role,
            environment_type_labels=environment_type_labels,
            result_type_label=result_type_label,
            root_candidate=root_candidate,
            constraint_kind=constraint_kind,
        )
        if production == "less-than-item-literal":
            if self._base_predicate is not None:
                raise RuntimeError("base item/literal predicate was constructed twice")
            self._base_predicate = handle
        return handle

    def _shared_gate_family(
        self,
        predicate: packed_v2.PackedPartitionHandle,
        item: packed_v2.PackedPartitionHandle,
    ) -> packed_v2.PackedPartitionHandle:
        if (
            tuple((slot.kind, slot.value_type) for slot in predicate.schema)
            != (_LITERAL_INT, _ROLE_INT)
            or tuple((slot.kind, slot.value_type) for slot in item.schema) != (_ROLE_INT,)
        ):
            raise RuntimeError("predicate gate source schemas changed")
        predicate_role_slot = predicate.schema[1]
        if predicate_role_slot.name != item.schema[0].name:
            raise RuntimeError("predicate gate lost its shared item role")
        action_role_slot = packed.TypedSlot(
            predicate_role_slot.kind,
            predicate_role_slot.value_type,
            predicate_role_slot.name,
            predicate_role_slot.occurrence + 1,
        )
        schema = (predicate.schema[0], predicate_role_slot, action_role_slot)
        children = (predicate, item)
        plan = packed_v2._FamilyPlan(
            family_id=packed_v2._family_identity(
                "predicate-gated-item", tuple(child.family_id for child in children)
            ),
            production="predicate-gated-item",
            inside_cost=sum(
                self.workset._require_handle(child).plan.inside_cost for child in children
            )
            + 1,
            child_family_ids=tuple(child.family_id for child in children),
            child_target_slots=(predicate.schema, (action_role_slot,)),
            child_slot_indices=((0, 1), (2,)),
            schema=schema,
            constraints=(
                packed.SlotConstraint.slots(
                    predicate_role_slot, "eq", action_role_slot
                ),
            ),
        )
        self._register_typing(
            plan,
            context_role="bound",
            environment_type_labels=("Int", "List[Int]"),
            result_type_label="Int",
            root_candidate=False,
        )
        self._record("family-primary", plan.family_id)
        self._record("semantic-expand", plan.family_id)
        handle = self.workset.apply_family(plan, children, self._predict_vector)
        self.maximum_completed_ast_cost = max(
            self.maximum_completed_ast_cost, plan.inside_cost
        )
        return handle

    def _predict_gate(
        self,
        predicate: output_cells.TypedOutputEncoding,
        item: output_cells.TypedOutputEncoding,
    ) -> output_cells.TypedOutputEncoding:
        predicate = output_cells.TypedOutputCellBank.validate(predicate)
        item = _require_int_event(item, "gated item")
        key = (
            "predicate-gated-item",
            self._checkpoint.sha256,
            predicate.sha256,
            item.sha256,
            self._predicate_cell_gate,
            self._action_cell_gate,
        )
        self.session.charge_workset("prediction-cache-lookup")
        cached = self._prediction_cache.get(key)
        if cached is not None:
            self.prediction_cache_hits += 1
            return output_cells.TypedOutputCellBank.validate(cached)
        attempted_prediction = self.physical_predictions + 1
        if attempted_prediction > full_typed.MAX_PHYSICAL_PREDICTIONS:
            raise packed_v2._PhysicalWorkBoundExceeded(
                "physical-predictions", attempted_prediction
            )
        attempted_races = self.physical_races + 1
        if attempted_races > full_typed.MAX_PHYSICAL_RACES:
            raise packed_v2._PhysicalWorkBoundExceeded("physical-races", attempted_races)
        row, _feature_cell_id = _feature_row(
            predicate,
            self._checkpoint.feature_cell_ids,
            cell_gate=self._predicate_cell_gate,
        )
        action_ordinal, race_root = _race_action(
            self._checkpoint,
            row,
            action_cell_gate=self._action_cell_gate,
        )
        if action_ordinal == _ADD_ORDINAL:
            output = item
        elif action_ordinal == _HOLD_ORDINAL:
            output = _ZERO_EVENT
        else:  # pragma: no cover - authenticated two-cell race invariant.
            raise RuntimeError("physical gate action lies outside its actuator roster")
        self.physical_predictions = attempted_prediction
        self.physical_races = attempted_races
        self._record("prediction", full_typed._digest("prediction-key", key))
        self._gate_race_roots.append(race_root)
        output = output_cells.TypedOutputCellBank.validate(output)
        self.session.charge_workset("prediction-cache-insert")
        incumbent = self._prediction_cache.setdefault(key, output)
        if incumbent != output:
            raise RuntimeError("predicate gate prediction cache identity collision")
        return output

    def _predict_vector(
        self,
        production: str,
        children: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        if production == "predicate-gated-item":
            if (
                len(children) != 2
                or len(children[0]) != len(self.flattened_items)
                or len(children[1]) != len(children[0])
            ):
                raise RuntimeError("predicate gate bound context changed")
            return tuple(
                self._predict_gate(predicate, item)
                for predicate, item in zip(children[0], children[1], strict=True)
            )
        return super()._predict_vector(production, children)

    def _extend_before_publish(
        self,
        collection: packed_v2.PackedPartitionHandle,
        item: packed_v2.PackedPartitionHandle,
        literal: packed_v2.PackedPartitionHandle,
    ) -> None:
        del literal
        predicate = self._base_predicate
        if predicate is None:
            raise RuntimeError("base item/literal predicate was not retained")
        gated = self._shared_gate_family(predicate, item)
        binder = self._family(
            "value-binder",
            (gated,),
            context_role="binder",
            environment_type_labels=("List[Int]",),
            result_type_label="Function[Int,Int]",
            root_candidate=False,
        )
        mapped = self._family(
            "map",
            (binder, collection),
            context_role="root",
            environment_type_labels=("List[Int]",),
            result_type_label="List[Int]",
            root_candidate=False,
        )
        self._family(
            "sum",
            (mapped,),
            context_role="root",
            environment_type_labels=("List[Int]",),
            result_type_label="Int",
            root_candidate=True,
        )


def freeze_predicate_gated_fold_scaffold() -> plan_slice.StructuralFactorCertificate:
    """Freeze the operator-name-free shared-item gate/map/sum topology."""

    return plan_slice.freeze_structural_factor_certificate(
        (
            plan_slice.StructuralOccurrence(
                None,
                None,
                (_LITERAL_INT, _ROLE_INT, _ROLE_INT, _ROLE_LIST_INT),
                (),
            ),
            plan_slice.StructuralOccurrence(
                0,
                0,
                (_LITERAL_INT, _ROLE_INT, _ROLE_INT, _ROLE_LIST_INT),
                (0, 1, 2, 3),
            ),
            plan_slice.StructuralOccurrence(
                1, 0, (_LITERAL_INT, _ROLE_INT, _ROLE_INT), (0, 1, 2)
            ),
            plan_slice.StructuralOccurrence(
                2,
                0,
                (_LITERAL_INT, _ROLE_INT, _ROLE_INT),
                (0, 1, 2),
                (plan_slice.StructuralRelationConstraint(1, "eq", 2),),
            ),
            plan_slice.StructuralOccurrence(
                3, 0, (_LITERAL_INT, _ROLE_INT), (0, 1)
            ),
            plan_slice.StructuralOccurrence(4, 0, (_ROLE_INT,), (1,)),
            plan_slice.StructuralOccurrence(4, 1, (_LITERAL_INT,), (0,)),
            plan_slice.StructuralOccurrence(3, 1, (_ROLE_INT,), (2,)),
            plan_slice.StructuralOccurrence(1, 1, (_ROLE_LIST_INT,), (3,)),
        ),
        literal_slot_copy_groups=(((6, 0),),),
    )


def _build_frontier(
    checkpoint: FrozenPredicateGateV1,
    signature: TypedSignature,
    cases: Iterable[tuple[object, ...]],
    threshold: int,
    *,
    predicate_cell_gate: bool,
    action_cell_gate: bool,
) -> breadth_v4.PackedGenericTypedTrainBreadthV4:
    checkpoint._authenticate()
    if not isinstance(signature, TypedSignature):
        raise TypeError("predicate-gated fold requires a TypedSignature")
    if (
        signature.result != _INT
        or len(signature.parameters) != 1
        or signature.parameters[0].type != _LIST_INT
    ):
        raise ValueError("predicate-gated fold requires exactly List[Int] -> Int")
    if isinstance(cases, (str, bytes, bytearray)):
        raise TypeError("cases must be an iterable of typed rows")
    builder = _PredicateGateListIntBuilder(
        signature,
        cases,
        threshold,
        checkpoint,
        predicate_cell_gate=predicate_cell_gate,
        action_cell_gate=action_cell_gate,
    )
    operator = builder.operator
    operator.clear_effect_memo()
    before = operator.live_checkpoint_sha256
    state, workset = builder.build()
    if operator.live_checkpoint_sha256 != before:
        raise RuntimeError("typed operator checkpoint changed during gated fold build")
    route = breadth_v4._PackedListIntFrontier(state, workset, operator, builder)
    accepted_rows = cast(tuple[tuple[object, ...], ...], state.case_arguments)
    partial = breadth_v3.PackedGenericTypedState(
        status=breadth_v3._status_from_route(state.status),
        signature=signature,
        case_arguments=accepted_rows,
        literals=state.literals,
        requested_max_ast_cost=state.requested_max_ast_cost,
        maximum_completed_ast_cost=state.maximum_completed_ast_cost,
        operator_checkpoint_sha256=state.operator_checkpoint_sha256,
        root_eclass_ids=state.root_eclass_ids,
        route_frontier_sha256=route.sha256,
        route_build_work_sha256=state.build_work_sha256,
        sha256="",
    )
    outer_state = partial._replace(
        sha256=breadth_v3._digest("typed-frontier", breadth_v3._state_payload(partial))
    )
    frontier = object.__new__(breadth_v4.PackedGenericTypedTrainBreadthV4)
    frontier._route = route
    frontier._state = outer_state
    frontier._construction_receipt = breadth_v3._issue_construction_receipt(
        route=route,
        state=outer_state,
        signature=signature,
    )
    return frontier


class PredicateGatedFoldState(NamedTuple):
    status: str
    stop_reason: str | None
    checkpoint_sha256: str
    frontier_sha256: str
    structural_certificate_sha256: str
    authorized_slice_count: int
    plan_slice_receipt_sha256: str | None
    root_eclass_count: int
    desired_reads_during_build: int
    action_label_inputs_during_build: int
    reused_typed_bool_cell_count: int
    new_action_cell_count: int
    new_persistent_state_bytes: int
    build_work_sha256: str
    sha256: str


class PredicateGatedFoldMaterialization(NamedTuple):
    status: str
    stop_reason: str | None
    publication_sha256: str
    plan_slice_receipt_sha256: str | None
    exact_proof_count: int
    witness: breadth_v3.PackedGenericTypedWitness | None
    relation_completed_units: int
    physical_predictions: int
    prediction_cache_hits: int
    physical_races: int
    desired_reads: int
    sha256: str


def _state_payload(value: PredicateGatedFoldState) -> dict[str, object]:
    return {field: getattr(value, field) for field in value._fields if field != "sha256"}


def _materialization_payload(
    value: PredicateGatedFoldMaterialization,
) -> dict[str, object]:
    return {
        field: (
            value.witness.sha256
            if field == "witness" and value.witness is not None
            else getattr(value, field)
        )
        for field in value._fields
        if field != "sha256"
    }


def _render_predicate_gate_witness(
    route: breadth_v4._PackedListIntFrontier,
    witness: packed.ProofWitness,
    component: packed_v3._AuthenticatedComponents,
) -> breadth_v3._PackedListIntWitness:
    route._workset.session.charge_frontier("materialize-witness")

    def render(family_id: str, binding: tuple[int, ...]) -> tuple[str, ...]:
        route._workset.session.charge_frontier("materialize-family-lookup")
        family = component.family_index.get(family_id)
        if type(family) is not packed_v3.PackedFamily or family.family_id != family_id:
            raise RuntimeError("predicate gate proof family index changed")
        if len(binding) != len(family.schema):
            raise RuntimeError("predicate gate proof binding arity changed")
        if family.production in {"collection-role", "item-role"}:
            if binding != (0,):
                raise RuntimeError("predicate gate role binding changed")
            return ("variable-0",)
        if family.production == "literal":
            return typed_program._factor_literal_tokens(binding[0])
        children = tuple(
            render(child_id, tuple(binding[index] for index in indices))
            for child_id, indices in zip(
                family.child_family_ids,
                family.child_slot_indices,
                strict=True,
            )
        )
        if family.production == "less-than-item-literal":
            return ("less-than", *children[0], *children[1])
        if family.production == "predicate-gated-item":
            return (
                "if",
                *children[0],
                *children[1],
                *typed_program._factor_literal_tokens(0),
            )
        if family.production == "value-binder":
            return ("bind", *children[0])
        if family.production == "map":
            return ("map", *children[0], *children[1])
        if family.production == "sum":
            return ("sum", *children[0])
        raise RuntimeError("predicate gate proof production changed")

    tokens = render(witness.family_id, witness.binding)
    source = typed_program.compile(typed_program.Program(tokens, route._state.signature)).source
    payload = {
        "family_id": witness.family_id,
        "binding": witness.binding,
        "tokens": tokens,
        "source": source,
    }
    return breadth_v3._PackedListIntWitness(
        family_id=witness.family_id,
        binding=witness.binding,
        tokens=tokens,
        source=source,
        sha256=breadth_v3._digest("cross-type-witness", payload),
    )


class PackedPhysicalPredicateGatedFoldV1:
    """Build a desired-free exact plan, then materialize one opaque witness."""

    __slots__ = (
        "_checkpoint",
        "_frontier",
        "_materialization",
        "_plan_slice_receipt",
        "_state",
        "_structural_certificate",
    )

    def __init__(self) -> None:
        raise TypeError("use PackedPhysicalPredicateGatedFoldV1.build")

    @classmethod
    def build(
        cls,
        checkpoint: FrozenPredicateGateV1,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        threshold: int,
        *,
        predicate_cell_gate: bool = True,
        action_cell_gate: bool = True,
    ) -> PackedPhysicalPredicateGatedFoldV1:
        if type(checkpoint) is not FrozenPredicateGateV1:
            raise TypeError("exact frozen predicate gate checkpoint required")
        checkpoint._authenticate()
        try:
            frontier = _build_frontier(
                checkpoint,
                signature,
                cases,
                threshold,
                predicate_cell_gate=predicate_cell_gate,
                action_cell_gate=action_cell_gate,
            )
        except _GateInferenceStop as error:
            raise PredicateGatedFoldBuildStop(error.status, error.detail) from error
        certificate = freeze_predicate_gated_fold_scaffold()
        receipt: plan_slice.PlanSliceReceipt | None = None
        authorized = 0
        stop_reason: str | None = None
        if frontier.status != breadth_v4.FRONTIER_READY:
            status = STOP_FRONTIER
            stop_reason = frontier.status
        else:
            try:
                slice_status, authorized, receipt, *_roots = plan_slice._bind_certificate(
                    frontier, certificate
                )
            except packed.PackedRelationWorkLimitExceeded:
                status = STOP_WORK_BOUND
                stop_reason = "predicate-gated plan authorization exceeded packed work"
            else:
                if (
                    slice_status == plan_slice.FRONTIER_READY
                    and authorized == 1
                    and receipt is not None
                ):
                    status = READY
                else:
                    status = STOP_PLAN_SLICE
                    stop_reason = slice_status
        work = frontier.work
        partial = PredicateGatedFoldState(
            status,
            stop_reason,
            checkpoint.sha256,
            frontier.sha256,
            certificate.sha256,
            authorized,
            None if receipt is None else receipt.sha256,
            len(frontier.root_eclass_ids),
            DESIRED_READS_DURING_BUILD,
            ACTION_LABEL_INPUTS_DURING_BUILD,
            REUSED_TYPED_BOOL_CELL_COUNT,
            NEW_ACTION_CELL_COUNT,
            NEW_PERSISTENT_STATE_BYTES,
            work.sha256,
            "",
        )
        state = partial._replace(sha256=_digest(b"publication-state", _state_payload(partial)))
        instance = object.__new__(cls)
        instance._checkpoint = checkpoint
        instance._frontier = frontier
        instance._structural_certificate = certificate
        instance._plan_slice_receipt = receipt
        instance._state = state
        instance._materialization = None
        instance._authenticate()
        return instance

    @property
    def state(self) -> PredicateGatedFoldState:
        self._authenticate()
        return self._state

    @property
    def status(self) -> str:
        return self._state.status

    @property
    def work(self) -> full_typed.PackedFullTypedWorkAudit:
        return self._frontier.work

    @property
    def plan_slice_receipt(self) -> plan_slice.PlanSliceReceipt | None:
        self._authenticate()
        return self._plan_slice_receipt

    def _authenticate(self) -> None:
        self._checkpoint._authenticate()
        plan_slice._authenticate_certificate(self._structural_certificate)
        construction_receipt = self._frontier._construction_receipt
        if (
            not breadth_v3._authenticate_construction_receipt(construction_receipt)
            or construction_receipt.route is not self._frontier._route
            or construction_receipt.state is not self._frontier._state
            or construction_receipt.signature is not self._frontier._state.signature
        ):
            raise RuntimeError("predicate-gated frontier construction receipt changed")
        state = self._state
        receipt = self._plan_slice_receipt
        if (
            state.sha256 != _digest(b"publication-state", _state_payload(state))
            or state.checkpoint_sha256 != self._checkpoint.sha256
            or state.frontier_sha256 != self._frontier.sha256
            or state.structural_certificate_sha256 != self._structural_certificate.sha256
            or state.plan_slice_receipt_sha256 != (None if receipt is None else receipt.sha256)
            or (self._materialization is None and state.build_work_sha256 != self.work.sha256)
        ):
            raise RuntimeError("predicate-gated fold publication authentication failed")
        if receipt is not None:
            plan_slice._authenticate_plan_slice_receipt(self._frontier, receipt)
        result = self._materialization
        if result is not None and result.sha256 != _digest(
            b"materialization", _materialization_payload(result)
        ):
            raise RuntimeError("predicate-gated fold materialization changed")

    def materialize(self) -> PredicateGatedFoldMaterialization:
        self._authenticate()
        if self._materialization is not None:
            return self._materialization
        receipt = self._plan_slice_receipt
        if self._state.status != READY or receipt is None:
            result = self._freeze_materialization(
                status=self._state.status,
                stop_reason=self._state.stop_reason,
                exact_proof_count=0,
                witness=None,
            )
            self._materialization = result
            return result
        binding = plan_slice._authenticate_plan_slice_receipt(self._frontier, receipt)
        try:
            forest = plan_slice._materialize_authorized_forest(binding)
        except packed.PackedRelationWorkLimitExceeded:
            result = self._freeze_materialization(
                status=STOP_WORK_BOUND,
                stop_reason="predicate-gated singleton exceeded packed work",
                exact_proof_count=0,
                witness=None,
            )
            self._materialization = result
            return result
        proof_count = forest.exact_logical_recipe_count
        witnesses = forest.first_two_witnesses
        if proof_count != 1 or len(witnesses) != 1:
            result = self._freeze_materialization(
                status=STOP_WITNESS_NOT_SINGLETON,
                stop_reason="predicate-gated plan did not materialize one exact proof",
                exact_proof_count=proof_count,
                witness=None,
            )
            self._materialization = result
            return result
        route = cast(breadth_v4._PackedListIntFrontier, binding.route)
        route_witness = _render_predicate_gate_witness(
            route, witnesses[0], binding.component
        )
        witness = cast(
            breadth_v3.PackedGenericTypedWitness,
            self._frontier._compile_witness(route_witness),
        )
        result = self._freeze_materialization(
            status=READY,
            stop_reason=None,
            exact_proof_count=1,
            witness=witness,
        )
        self._materialization = result
        return result

    def _freeze_materialization(
        self,
        *,
        status: str,
        stop_reason: str | None,
        exact_proof_count: int,
        witness: breadth_v3.PackedGenericTypedWitness | None,
    ) -> PredicateGatedFoldMaterialization:
        work = self.work
        partial = PredicateGatedFoldMaterialization(
            status,
            stop_reason,
            self._state.sha256,
            self._state.plan_slice_receipt_sha256,
            exact_proof_count,
            witness,
            work.relation.completed_units,
            work.physical_predictions,
            work.prediction_cache_hits,
            work.physical_races,
            0,
            "",
        )
        return partial._replace(
            sha256=_digest(b"materialization", _materialization_payload(partial))
        )


def require_predicate_gated_fold_witness(
    publication: PackedPhysicalPredicateGatedFoldV1,
) -> breadth_v3.PackedGenericTypedWitness:
    if type(publication) is not PackedPhysicalPredicateGatedFoldV1:
        raise TypeError("exact predicate-gated fold publication required")
    result = publication.materialize()
    if result.status != READY or result.exact_proof_count != 1 or result.witness is None:
        raise ValueError(result.stop_reason or result.status)
    return result.witness


CONFIG_RECORD = {
    "schema": SCHEMA,
    "state_shape": STATE_SHAPE,
    "tick_dtype": TICK_DTYPE.name,
    "new_persistent_state_bytes": NEW_PERSISTENT_STATE_BYTES,
    "reused_typed_bool_cell_count": REUSED_TYPED_BOOL_CELL_COUNT,
    "new_action_cell_count": NEW_ACTION_CELL_COUNT,
    "fixed_max_cost": FIXED_MAX_COST,
    "witness_ast_cost": WITNESS_AST_COST,
    "action_race_step_count": ACTION_RACE_STEP_COUNT,
    "desired_reads_during_build": DESIRED_READS_DURING_BUILD,
    "action_label_inputs_during_build": ACTION_LABEL_INPUTS_DURING_BUILD,
    "claim_boundary": CLAIM_BOUNDARY,
}
CONFIG_SHA256 = _digest(b"config", CONFIG_RECORD)

__all__ = [
    "ACTION_LABEL_INPUTS_DURING_BUILD",
    "CLAIM_BOUNDARY",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "DESIRED_READS_DURING_BUILD",
    "NEW_ACTION_CELL_COUNT",
    "NEW_PERSISTENT_STATE_BYTES",
    "READY",
    "REUSED_TYPED_BOOL_CELL_COUNT",
    "STATE_SHAPE",
    "STOP_ACTION_RACE",
    "STOP_CONSEQUENCE",
    "STOP_FEATURE",
    "STOP_FRONTIER",
    "STOP_LAW",
    "STOP_PLAN_SLICE",
    "STOP_WITNESS_NOT_SINGLETON",
    "STOP_WORK_BOUND",
    "FrozenPredicateGateV1",
    "PackedPhysicalPredicateGatedFoldV1",
    "PredicateGateFitStop",
    "PredicateGateLesson",
    "PredicateGatedFoldBuildStop",
    "PredicateGatedFoldMaterialization",
    "PredicateGatedFoldState",
    "fit_causal_predicate_gate",
    "freeze_predicate_gated_fold_scaffold",
    "list_int_to_int_signature",
    "observe_predicate_gate_lesson",
    "require_predicate_gated_fold_witness",
]
