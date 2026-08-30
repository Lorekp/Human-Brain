"""Desired-independent, use-directed typed semantic version spaces.

This is the first V2 successor to the spent complete-tier V1 frontier.  It is
deliberately isolated from the V1 gate and its fixed work counters.  A caller
first builds a complete physical frontier without supplying desired outputs,
then asks the existing fitted-cell semantic-vector race to select from that
frozen frontier.

The bounded first layer covers the self-posed Int and List[Int] compositions
needed to exercise scalar pressure, predicate/count scheduling, and visible
version-space ambiguity.  Its compiler bookkeeping is proof-only: it may merge
programs that are equal by a frozen algebraic rule, but observed equality alone
never deletes an alternative recipe.

Operator-skeleton and literal/role-slot factorization is authenticated metadata
only in this slice. It does not compress admitted recipes, predictions, or
races. The only execution reuse is an exact memo keyed by the complete physical
operator input; a memo hit returns before ``_predict_effect`` is called.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import cast

from experiments.r3_pilot import public_typed_operator_successor as operator_successor
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_synthesizer as semantic_v1
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from humanbrain import typed_program as tp
from humanbrain import typed_semantic_synthesis as synthesis
from humanbrain.typed_program_types import ProgramType, TypedSignature


FRONTIER_READY = "FRONTIER_READY"
STOP_FRONTIER_WORK_BOUND = "STOP_SEMANTIC_FRONTIER_V2_WORK_BOUND"
SELECTED = "SELECTED"
STOP_NO_EXACT = "STOP_NO_EXACT_VISIBLE_SPEC"
STOP_AMBIGUOUS = "STOP_AMBIGUOUS_VISIBLE_SPEC"

MAX_ADMITTED_CANONICAL_RECIPES = 30_000
MAX_OPERATOR_PREDICTIONS = 30_000
MAX_PHYSICAL_RACES = 80_000
SATURATING_EXPANSION_LIMIT = 2**31 - 1

_INT = ProgramType.Int()
_BOOL = ProgramType.Bool()
_LIST_INT = ProgramType.List(_INT)

_USE_ROLE_ORDER = {
    "map_body": 0,
    "predicate_operand": 1,
    "predicate": 2,
    "binder": 3,
    "collection": 4,
    "root": 5,
}
_PREDICTION_SCHEMA_MAX_RACES = dict(semantic_v1._PREDICTION_SCHEMA_MAX_RACES)
_NORMALIZATION_VERSION = "typed-semantic-proof-normal-form-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(domain: bytes, value: object) -> str:
    digest = sha256()
    digest.update(domain)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _type_label(value: ProgramType) -> str:
    return synthesis._type_label(value)


@dataclass(frozen=True, slots=True)
class FrontierDemand:
    """One reachable typed use of a canonical recipe."""

    environment_type_labels: tuple[str, ...]
    result_type_label: str
    use_role: str
    inside_cost: int
    minimum_complete_root_cost: int

    def __post_init__(self) -> None:
        if self.use_role not in _USE_ROLE_ORDER:
            raise ValueError("unknown semantic-frontier use role")
        if self.inside_cost < 1:
            raise ValueError("inside cost must be positive")
        if self.minimum_complete_root_cost < self.inside_cost:
            raise ValueError("complete-root cost cannot precede inside cost")


@dataclass(frozen=True, slots=True)
class CanonicalRecipe:
    """One proof-distinct recipe factored into a shared operator skeleton."""

    canonical_recipe_id: str
    production: str
    child_semantic_cost_layer_ids: tuple[str, ...]
    structural_normal_form: tuple[object, ...]
    operator_skeleton_id: str
    slot_bindings: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class RecipeAdmission:
    ordinal: int
    canonical_recipe_id: str
    use_role: str
    inside_cost: int
    minimum_complete_root_cost: int


@dataclass(frozen=True, slots=True)
class RootExpansion:
    admission: RecipeAdmission
    program: tp.Program
    source: str


@dataclass(frozen=True, slots=True)
class ObservationLayer:
    inside_cost: int
    semantic_cost_layer_id: str
    canonical_recipe_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObservationClass:
    """One physical semantic class retaining proof-distinct OR expansions."""

    physical_semantic_key: str
    demand_environment_type_labels: tuple[str, ...]
    demand_result_type_label: str
    demand_use_role: str
    physical_observation_sha256s: tuple[str, ...]
    cost_layers: tuple[ObservationLayer, ...]
    canonical_representative_id: str
    saturating_expansion_count: int
    alias_digest: str
    first_two_witnesses: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrontierWorkAudit:
    status: str
    bound_name: str | None
    limit: int | None
    completed: int
    attempted: int
    minimum_complete_root_cost: int | None
    inside_cost: int | None
    admitted_canonical_recipes: int
    unique_canonical_recipes: int
    operator_skeletons: int
    operator_predictions: int
    physical_races: int
    unsupported_pruned_recipes: int
    normalization_raw_expansions: int
    normalization_merged_expansions: int
    maximum_admitted_complete_root_cost: int
    agenda_sha256: str
    operator_races_sha256: str
    frontier_sha256: str


@dataclass(frozen=True, slots=True)
class VersionSpaceSelection:
    status: str
    stop_reason: str | None
    selected_recipe_id: str | None
    program: tp.Program | None
    source: str | None
    exact_recipe_ids: tuple[str, ...]
    observation_class: ObservationClass | None
    race: semantic_race.TypedSemanticVectorRaceAudit
    frontier_sha256: str
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str


@dataclass(frozen=True, slots=True)
class _Expression:
    node: synthesis.CandidateNode
    normal_form: tuple[object, ...]

    @property
    def cost(self) -> int:
        return self.node.ast_cost


@dataclass(frozen=True, slots=True)
class _PendingRecipe:
    demand: FrontierDemand
    expression: _Expression
    child_refs: tuple[tuple[str, str], ...]
    witness: str


@dataclass(slots=True)
class _ObservationAccumulator:
    key: str
    demand: FrontierDemand
    observations: tuple[str, ...]
    recipe_ids_by_cost: dict[int, list[str]]
    witnesses: list[str]
    alias_hasher: object
    count: int = 0

    def add(self, inside_cost: int, recipe_id: str, witness: str) -> None:
        self.recipe_ids_by_cost.setdefault(inside_cost, []).append(recipe_id)
        self.count = min(SATURATING_EXPANSION_LIMIT, self.count + 1)
        cast(object, self.alias_hasher).update(recipe_id.encode("ascii"))
        cast(object, self.alias_hasher).update(b"\0")
        if len(self.witnesses) < 2:
            self.witnesses.append(witness)


class _FrontierDenied(RuntimeError):
    def __init__(
        self,
        *,
        bound_name: str,
        limit: int,
        completed: int,
        attempted: int,
        demand: FrontierDemand,
    ) -> None:
        super().__init__(bound_name)
        self.bound_name = bound_name
        self.limit = limit
        self.completed = completed
        self.attempted = attempted
        self.demand = demand


class _V2PhysicalBudget:
    """Prediction/race accounting compatible with the existing evaluator."""

    def __init__(self) -> None:
        self.completed_operator_predictions = 0
        self.completed_physical_races = 0
        self.current_demand: FrontierDemand | None = None
        self._reservation: str | None = None
        self._race_digest = sha256()
        self._race_digest.update(b"typed-semantic-version-space-v2-races\0")

    def set_demand(self, demand: FrontierDemand) -> None:
        self.current_demand = demand

    def reserve_prediction(self, schema: str) -> int:
        maximum = _PREDICTION_SCHEMA_MAX_RACES.get(schema)
        if maximum is None:
            raise RuntimeError(f"unknown physical prediction schema {schema!r}")
        if self.current_demand is None:
            raise RuntimeError("physical prediction has no frontier demand")
        if self._reservation is not None:
            raise RuntimeError("physical predictions cannot overlap reservations")
        attempted_predictions = self.completed_operator_predictions + 1
        if attempted_predictions > MAX_OPERATOR_PREDICTIONS:
            raise _FrontierDenied(
                bound_name="operator-predictions",
                limit=MAX_OPERATOR_PREDICTIONS,
                completed=self.completed_operator_predictions,
                attempted=attempted_predictions,
                demand=self.current_demand,
            )
        attempted_races = self.completed_physical_races + maximum
        if attempted_races > MAX_PHYSICAL_RACES:
            raise _FrontierDenied(
                bound_name="physical-races",
                limit=MAX_PHYSICAL_RACES,
                completed=self.completed_physical_races,
                attempted=attempted_races,
                demand=self.current_demand,
            )
        self._reservation = schema
        return maximum

    def complete_prediction(
        self,
        races: tuple[operator_successor.OperatorRace, ...],
        reserved_max_races: int,
    ) -> None:
        if len(races) > reserved_max_races:
            raise RuntimeError("physical prediction exceeded its reservation")
        schema = self._reservation
        self._reservation = None
        if schema is None or self.current_demand is None:
            raise RuntimeError("physical prediction completed without a reservation")
        self.completed_operator_predictions += 1
        self.completed_physical_races += len(races)
        for race in races:
            self._race_digest.update(_canonical_json(asdict(race)))

    @property
    def race_sha256(self) -> str:
        return self._race_digest.hexdigest()


class _V2CaseEvaluator(semantic_v1._CaseEvaluator):
    """Existing evaluator with a pre-execution exact physical-effect memo."""

    def __init__(
        self,
        *args: object,
        effect_cache: dict[
            str,
            tuple[
                output_cells.TypedOutputEncoding,
                tuple[operator_successor.OperatorRace, ...],
            ],
        ],
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._v2_effect_cache = effect_cache

    def _predict(
        self,
        selected: output_cells.TypedOutputEncoding,
        action: tp.FactorAction,
        scopes: tuple[output_cells.TypedOutputEncoding, ...] = (),
        *,
        schema: str,
    ) -> output_cells.TypedOutputEncoding:
        fact_sha256 = _sha256(
            b"typed-semantic-version-space-v2-operator-fact\0",
            {
                "schema": schema,
                "selected": selected.sha256,
                "action": asdict(action),
                "scopes": tuple(scope.sha256 for scope in scopes),
            },
        )
        cached = self._v2_effect_cache.get(fact_sha256)
        if cached is not None:
            return cached[0]
        budget = cast(_V2PhysicalBudget, self.budget)
        reserved_max_races = budget.reserve_prediction(schema)
        output, races = self.model._predict_effect(selected, action, scopes)
        budget.complete_prediction(races, reserved_max_races)
        self._v2_effect_cache[fact_sha256] = (output, races)
        self.predictions += 1
        self.races.extend(races)
        return output


def _form_key(form: tuple[object, ...]) -> bytes:
    return _canonical_json(form)


def _literal(form: tuple[object, ...]) -> int | None:
    if len(form) == 2 and form[0] == "literal" and type(form[1]) is int:
        return cast(int, form[1])
    return None


def _normalize_absolute(child: tuple[object, ...]) -> tuple[object, ...]:
    value = _literal(child)
    if value is not None:
        return ("literal", abs(value))
    if child and child[0] == "absolute":
        return child
    return ("absolute", child)


def _extract_literal_pair(
    form: tuple[object, ...],
) -> tuple[tuple[object, ...], int] | None:
    if len(form) != 2 or form[0] not in {"minimum", "maximum"}:
        return None
    operands = cast(tuple[tuple[object, ...], ...], form[1])
    if len(operands) != 2:
        return None
    left_value = _literal(operands[0])
    right_value = _literal(operands[1])
    if left_value is not None and right_value is None:
        return operands[1], left_value
    if right_value is not None and left_value is None:
        return operands[0], right_value
    return None


def _normalize_nary(
    production: str,
    operands: Iterable[tuple[object, ...]],
) -> tuple[object, ...]:
    flattened: list[tuple[object, ...]] = []
    for operand in operands:
        if operand and operand[0] == production:
            flattened.extend(cast(tuple[tuple[object, ...], ...], operand[1]))
        else:
            flattened.append(operand)

    if production == "add":
        constant = sum(value for item in flattened if (value := _literal(item)) is not None)
        flattened = [item for item in flattened if _literal(item) is None]
        if constant or not flattened:
            flattened.append(("literal", constant))
    else:
        constants = [value for item in flattened if (value := _literal(item)) is not None]
        flattened = [item for item in flattened if _literal(item) is None]
        if constants:
            folded = min(constants) if production == "minimum" else max(constants)
            flattened.append(("literal", folded))

    if production == "add":
        # Addition is associative and commutative, but not idempotent.
        # Preserve repeated nonliteral terms such as x+x.
        unique = sorted(flattened, key=_form_key)
    else:
        unique = sorted(set(flattened), key=_form_key)
    if production in {"minimum", "maximum"}:
        opposite = "maximum" if production == "minimum" else "minimum"
        direct = set(unique)
        unique = [
            item
            for item in unique
            if not (
                item
                and item[0] == opposite
                and any(
                    child in direct
                    for child in cast(tuple[tuple[object, ...], ...], item[1])
                )
            )
        ]
    if len(unique) == 1:
        return unique[0]
    result: tuple[object, ...] = (production, tuple(unique))

    # max(min(x, hi), lo) == min(max(x, lo), hi) only when the literal
    # ordering proves that lo is the lower clamp endpoint.
    if production == "maximum" and len(unique) == 2:
        outer_literal = None
        inner = None
        for item in unique:
            value = _literal(item)
            if value is not None:
                outer_literal = value
            elif item and item[0] == "minimum":
                inner = item
        if outer_literal is not None and inner is not None:
            extracted = _extract_literal_pair(inner)
            if extracted is not None:
                base, upper = extracted
                if outer_literal <= upper:
                    return _normalize_nary(
                        "minimum",
                        (
                            _normalize_nary(
                                "maximum",
                                (base, ("literal", outer_literal)),
                            ),
                            ("literal", upper),
                        ),
                    )
    return result


def _normalize_binary(
    production: str,
    left: tuple[object, ...],
    right: tuple[object, ...],
) -> tuple[object, ...]:
    left_value = _literal(left)
    right_value = _literal(right)
    if production == "subtract":
        if right_value == 0:
            return left
        if left == right:
            return ("literal", 0)
        if left_value is not None and right_value is not None:
            return ("literal", left_value - right_value)
        return ("subtract", left, right)
    if production in {"add", "minimum", "maximum"}:
        return _normalize_nary(production, (left, right))
    if production == "less-than":
        if left_value is not None and right_value is not None:
            return ("boolean", left_value < right_value)
        return ("less-than", left, right)
    raise RuntimeError(f"unsupported normalizer production {production!r}")


def _operator_skeleton(form: tuple[object, ...]) -> tuple[object, ...]:
    head = form[0]
    if head == "literal":
        return ("literal-slot", "Int")
    if head == "variable":
        return ("role-slot", "value")
    if head == "boolean":
        return ("boolean-slot",)
    if head in {"add", "minimum", "maximum"}:
        return (
            head,
            tuple(
                _operator_skeleton(child)
                for child in cast(tuple[tuple[object, ...], ...], form[1])
            ),
        )
    return (head, *tuple(
        _operator_skeleton(child) if isinstance(child, tuple) and child else child
        for child in form[1:]
    ))


def _slot_bindings(form: tuple[object, ...]) -> tuple[tuple[str, int], ...]:
    bindings: list[tuple[str, int]] = []

    def visit(item: object) -> None:
        if not isinstance(item, tuple) or not item:
            return
        if item[0] == "literal":
            bindings.append(("literal", cast(int, item[1])))
            return
        if item[0] == "variable":
            bindings.append(("role", cast(int, item[1])))
            return
        for child in item[1:]:
            if isinstance(child, tuple) and child and isinstance(child[0], tuple):
                for nested in child:
                    visit(nested)
            else:
                visit(child)

    visit(form)
    return tuple(bindings)


class _ScalarPool:
    """Incremental proof-normalized scalar grammar for one environment."""

    def __init__(
        self,
        builder: synthesis._NodeBuilder,
        environment: tuple[ProgramType, ...],
        literals: tuple[int, ...],
        counters: dict[str, int],
    ) -> None:
        self.builder = builder
        self.environment = environment
        self.literals = literals
        self.counters = counters
        self.by_cost: dict[int, tuple[_Expression, ...]] = {}
        self.seen: dict[tuple[object, ...], _Expression] = {}
        self.generated_cost = 0

    def _admit_form(
        self,
        *,
        cost: int,
        normal_form: tuple[object, ...],
        production: str,
        tokens: tuple[str, ...],
        operands: tuple[_Expression, ...] = (),
        variable_index: int | None = None,
        literal_value: int | None = None,
    ) -> _Expression | None:
        self.counters["raw"] += 1
        if normal_form in self.seen:
            self.counters["merged"] += 1
            return None
        node = self.builder.add(
            ast_cost=cost,
            environment=self.environment,
            result_type_label="Int",
            tokens=tokens,
            provenance=synthesis.CandidateProvenance(
                production,
                tuple(item.node.node_id for item in operands),
                variable_index=variable_index,
                literal_value=literal_value,
            ),
        )
        expression = _Expression(node, normal_form)
        self.seen[normal_form] = expression
        return expression

    def ensure(self, maximum_cost: int) -> None:
        for cost in range(self.generated_cost + 1, maximum_cost + 1):
            tier: list[_Expression] = []
            if cost == 1:
                for index, value_type in enumerate(self.environment[:2]):
                    if value_type != _INT:
                        continue
                    expression = self._admit_form(
                        cost=1,
                        normal_form=("variable", index),
                        production=f"variable-{index}",
                        tokens=(f"variable-{index}",),
                        variable_index=index,
                    )
                    if expression is not None:
                        tier.append(expression)
                for value in self.literals:
                    expression = self._admit_form(
                        cost=1,
                        normal_form=("literal", value),
                        production="integer",
                        tokens=tp._factor_literal_tokens(value),
                        literal_value=value,
                    )
                    if expression is not None:
                        tier.append(expression)
            else:
                for operand in self.by_cost.get(cost - 1, ()):
                    expression = self._admit_form(
                        cost=cost,
                        normal_form=_normalize_absolute(operand.normal_form),
                        production="absolute",
                        tokens=("absolute", *operand.node.tokens),
                        operands=(operand,),
                    )
                    if expression is not None:
                        tier.append(expression)
                for production in ("add", "subtract", "minimum", "maximum"):
                    for left_cost in range(1, cost - 1):
                        right_cost = cost - 1 - left_cost
                        for left in self.by_cost.get(left_cost, ()):
                            for right in self.by_cost.get(right_cost, ()):
                                if (
                                    production in {"add", "minimum", "maximum"}
                                    and _form_key(left.normal_form)
                                    > _form_key(right.normal_form)
                                ):
                                    continue
                                expression = self._admit_form(
                                    cost=cost,
                                    normal_form=_normalize_binary(
                                        production,
                                        left.normal_form,
                                        right.normal_form,
                                    ),
                                    production=production,
                                    tokens=(production, *left.node.tokens, *right.node.tokens),
                                    operands=(left, right),
                                )
                                if expression is not None:
                                    tier.append(expression)
            self.by_cost[cost] = tuple(
                sorted(tier, key=lambda item: self.builder.sort_key(item.node))
            )
            self.generated_cost = cost

    def through(self, maximum_cost: int) -> tuple[_Expression, ...]:
        self.ensure(maximum_cost)
        return tuple(
            item
            for cost in range(1, maximum_cost + 1)
            for item in self.by_cost.get(cost, ())
        )


class _UseDirectedGenerator:
    """Generate only recipes whose first complete root is in the active stage."""

    def __init__(
        self,
        signature: TypedSignature,
        literals: tuple[int, ...],
        max_ast_cost: int,
    ) -> None:
        if signature.result != _INT:
            raise ValueError("V2 first layer supports only Int results")
        self.signature = signature
        self.literals = literals
        self.max_ast_cost = max_ast_cost
        self.production_order = synthesis._compiler_production_order()
        self.builder = synthesis._NodeBuilder(self.production_order)
        self.counters = {"raw": 0, "merged": 0}
        self.root_environment = tuple(
            parameter.type for parameter in reversed(signature.parameters)
        )
        self.root_scalars = _ScalarPool(
            self.builder,
            self.root_environment,
            literals,
            self.counters,
        )
        self.list_parameters = tuple(
            index
            for index, value_type in enumerate(self.root_environment[:2])
            if value_type == _LIST_INT
        )
        self.body_environment = (_INT, *self.root_environment)
        self.body_scalars = _ScalarPool(
            self.builder,
            self.body_environment,
            literals,
            self.counters,
        )
        self.base_collections: tuple[_Expression, ...] = ()
        self.map_collections: list[_Expression] = []
        self.map_binders: dict[str, _Expression] = {}
        self.predicates_by_cost: dict[int, tuple[_Expression, ...]] = {}
        self.predicate_seen: dict[tuple[object, ...], _Expression] = {}
        self.predicate_binders: dict[str, _Expression] = {}
        self._initialize_collections()

    def _initialize_collections(self) -> None:
        collections: list[_Expression] = []
        for index in self.list_parameters:
            token = f"variable-{index}"
            node = self.builder.add(
                ast_cost=1,
                environment=self.root_environment,
                result_type_label="List[Int]",
                tokens=(token,),
                provenance=synthesis.CandidateProvenance(
                    token,
                    variable_index=index,
                ),
            )
            collections.append(_Expression(node, ("variable", index)))
        self.base_collections = tuple(collections)

    def _demand(
        self,
        expression: _Expression,
        role: str,
        minimum_root_cost: int,
    ) -> FrontierDemand:
        return FrontierDemand(
            expression.node.environment_type_labels,
            expression.node.result_type_label,
            role,
            expression.cost,
            minimum_root_cost,
        )

    def _pending(
        self,
        expression: _Expression,
        role: str,
        minimum_root_cost: int,
        child_roles: tuple[str, ...],
        witness: str | None = None,
    ) -> _PendingRecipe:
        operands = expression.node.provenance.operand_node_ids
        if len(operands) != len(child_roles):
            raise RuntimeError("recipe child roles lost compiler operand alignment")
        if witness is None and role == "root":
            witness = tp.compile(tp.Program(expression.node.tokens, self.signature)).source
        return _PendingRecipe(
            self._demand(expression, role, minimum_root_cost),
            expression,
            tuple(zip(operands, child_roles, strict=True)),
            witness or " ".join(expression.node.tokens),
        )

    def _ensure_predicates(self, predicate_cost: int) -> None:
        for cost in range(3, predicate_cost + 1):
            if cost in self.predicates_by_cost:
                continue
            self.body_scalars.ensure(cost - 2)
            tier: list[_Expression] = []
            for left_cost in range(1, cost - 1):
                right_cost = cost - 1 - left_cost
                for left in self.body_scalars.by_cost.get(left_cost, ()):
                    for right in self.body_scalars.by_cost.get(right_cost, ()):
                        normal_form = _normalize_binary(
                            "less-than",
                            left.normal_form,
                            right.normal_form,
                        )
                        self.counters["raw"] += 1
                        if normal_form in self.predicate_seen:
                            self.counters["merged"] += 1
                            continue
                        node = self.builder.add(
                            ast_cost=cost,
                            environment=self.body_environment,
                            result_type_label="Bool",
                            tokens=("less-than", *left.node.tokens, *right.node.tokens),
                            provenance=synthesis.CandidateProvenance(
                                "less-than",
                                (left.node.node_id, right.node.node_id),
                            ),
                        )
                        expression = _Expression(node, normal_form)
                        self.predicate_seen[normal_form] = expression
                        tier.append(expression)
            self.predicates_by_cost[cost] = tuple(
                sorted(tier, key=lambda item: self.builder.sort_key(item.node))
            )

    def _map_binder(self, body: _Expression) -> _Expression:
        existing = self.map_binders.get(body.node.node_id)
        if existing is not None:
            return existing
        node = self.builder.add(
            ast_cost=body.cost + 1,
            environment=self.root_environment,
            result_type_label="Function[Int,Int]",
            tokens=("bind", *body.node.tokens),
            provenance=synthesis.CandidateProvenance(
                "bind",
                (body.node.node_id,),
            ),
        )
        binder = _Expression(node, ("bind-int", body.normal_form))
        self.map_binders[body.node.node_id] = binder
        return binder

    def _predicate_binder(self, predicate: _Expression) -> _Expression:
        existing = self.predicate_binders.get(predicate.node.node_id)
        if existing is not None:
            return existing
        node = self.builder.add(
            ast_cost=predicate.cost + 1,
            environment=self.root_environment,
            result_type_label="Function[Int,Bool]",
            tokens=("bind", *predicate.node.tokens),
            provenance=synthesis.CandidateProvenance(
                "bind",
                (predicate.node.node_id,),
            ),
        )
        binder = _Expression(node, ("bind-bool", predicate.normal_form))
        self.predicate_binders[predicate.node.node_id] = binder
        return binder

    def stage(self, complete_root_cost: int) -> tuple[_PendingRecipe, ...]:
        """Materialize one guarded complete-root stage, never any later stage."""

        pending: list[_PendingRecipe] = []
        self.root_scalars.ensure(complete_root_cost)
        for expression in self.root_scalars.by_cost.get(complete_root_cost, ()):
            pending.append(
                self._pending(expression, "root", complete_root_cost, tuple(
                    "root" for _ in expression.node.provenance.operand_node_ids
                ))
            )

        if not self.base_collections:
            return tuple(pending)

        if complete_root_cost == 2:
            for collection in self.base_collections:
                pending.append(self._pending(collection, "collection", 2, ()))

        body_cost = complete_root_cost - 4
        if body_cost >= 1:
            self.body_scalars.ensure(body_cost)
            for body in self.body_scalars.by_cost.get(body_cost, ()):
                pending.append(
                    self._pending(
                        body,
                        "map_body",
                        complete_root_cost,
                        tuple("map_body" for _ in body.node.provenance.operand_node_ids),
                    )
                )
                binder = self._map_binder(body)
                pending.append(
                    self._pending(
                        binder,
                        "binder",
                        complete_root_cost,
                        ("map_body",),
                    )
                )
                for collection in self.base_collections:
                    map_node = self.builder.add(
                        ast_cost=2 + body.cost + collection.cost,
                        environment=self.root_environment,
                        result_type_label="List[Int]",
                        tokens=("map", *binder.node.tokens, *collection.node.tokens),
                        provenance=synthesis.CandidateProvenance(
                            "map",
                            (binder.node.node_id, collection.node.node_id),
                        ),
                    )
                    mapped = _Expression(
                        map_node,
                        ("map", body.normal_form, collection.normal_form),
                    )
                    self.map_collections.append(mapped)
                    pending.append(
                        self._pending(
                            mapped,
                            "collection",
                            complete_root_cost,
                            ("binder", "collection"),
                        )
                    )

        operand_cost = complete_root_cost - 5
        if operand_cost >= 1:
            self.body_scalars.ensure(operand_cost)
            for operand in self.body_scalars.by_cost.get(operand_cost, ()):
                pending.append(
                    self._pending(
                        operand,
                        "predicate_operand",
                        complete_root_cost,
                        tuple(
                            "predicate_operand"
                            for _ in operand.node.provenance.operand_node_ids
                        ),
                    )
                )

        predicate_cost = complete_root_cost - 3
        if predicate_cost >= 3:
            self._ensure_predicates(predicate_cost)
            for predicate in self.predicates_by_cost.get(predicate_cost, ()):
                pending.append(
                    self._pending(
                        predicate,
                        "predicate",
                        complete_root_cost,
                        ("predicate_operand", "predicate_operand"),
                    )
                )
                binder = self._predicate_binder(predicate)
                pending.append(
                    self._pending(
                        binder,
                        "binder",
                        complete_root_cost,
                        ("predicate",),
                    )
                )

        collections = (*self.base_collections, *self.map_collections)
        for collection in collections:
            if 1 + collection.cost == complete_root_cost:
                node = self.builder.add(
                    ast_cost=complete_root_cost,
                    environment=self.root_environment,
                    result_type_label="Int",
                    tokens=("sum", *collection.node.tokens),
                    provenance=synthesis.CandidateProvenance(
                        "sum",
                        (collection.node.node_id,),
                    ),
                )
                root = _Expression(node, ("sum", collection.normal_form))
                program = tp.Program(root.node.tokens, self.signature)
                pending.append(
                    self._pending(
                        root,
                        "root",
                        complete_root_cost,
                        ("collection",),
                        tp.compile(program).source,
                    )
                )

        for predicate in self.predicate_seen.values():
            binder = self._predicate_binder(predicate)
            for collection in collections:
                root_cost = 1 + binder.cost + collection.cost
                if root_cost != complete_root_cost:
                    continue
                node = self.builder.add(
                    ast_cost=root_cost,
                    environment=self.root_environment,
                    result_type_label="Int",
                    tokens=("count", *binder.node.tokens, *collection.node.tokens),
                    provenance=synthesis.CandidateProvenance(
                        "count",
                        (binder.node.node_id, collection.node.node_id),
                    ),
                )
                root = _Expression(
                    node,
                    ("count", predicate.normal_form, collection.normal_form),
                )
                program = tp.Program(root.node.tokens, self.signature)
                pending.append(
                    self._pending(
                        root,
                        "root",
                        complete_root_cost,
                        ("binder", "collection"),
                        tp.compile(program).source,
                    )
                )

        return tuple(pending)

    def dag_snapshot(self) -> synthesis.TypedCandidateDag:
        return synthesis.TypedCandidateDag(
            signature=self.signature,
            literal_bank=self.literals,
            max_ast_cost=self.max_ast_cost,
            nodes=tuple(
                sorted(self.builder.nodes.values(), key=self.builder.sort_key)
            ),
            tiers=tuple(
                synthesis.CandidateTier(cost, ())
                for cost in range(1, self.max_ast_cost + 1)
            ),
        )


def _context_shape(
    signature: TypedSignature,
    arguments: tuple[tuple[object, ...], ...],
    role: str,
) -> tuple[object, ...]:
    if role in {"root", "collection"}:
        return semantic_v1._root_context_shape(arguments)
    return semantic_v1._bound_context_shape(signature, arguments, _LIST_INT)


def _contexts(
    signature: TypedSignature,
    arguments: tuple[tuple[object, ...], ...],
    evaluators: tuple[semantic_v1._CaseEvaluator, ...],
    role: str,
) -> tuple[tuple[semantic_v1._CaseEvaluator, tuple[object, ...]], ...]:
    if role in {"root", "collection"}:
        return tuple((evaluator, ()) for evaluator in evaluators)
    return tuple(
        (evaluator, (item,))
        for evaluator, row in zip(evaluators, arguments, strict=True)
        for parameter, value in zip(signature.parameters, row, strict=True)
        if parameter.type == _LIST_INT
        for item in cast(list[int], value)
    )


def _recipe(
    pending: _PendingRecipe,
    child_layer_ids: tuple[str, ...],
) -> CanonicalRecipe:
    form = pending.expression.normal_form
    skeleton = _operator_skeleton(form)
    skeleton_id = _sha256(
        b"typed-semantic-version-space-v2-skeleton\0",
        {
            "inside_cost": pending.demand.inside_cost,
            "result_type": pending.demand.result_type_label,
            "skeleton": skeleton,
        },
    )
    payload = {
        "production": pending.expression.node.provenance.production,
        "children": child_layer_ids,
        "normal_form": form,
        "skeleton": skeleton_id,
        "normalization_version": _NORMALIZATION_VERSION,
    }
    return CanonicalRecipe(
        canonical_recipe_id=_sha256(
            b"typed-semantic-version-space-v2-recipe\0",
            payload,
        ),
        production=pending.expression.node.provenance.production,
        child_semantic_cost_layer_ids=child_layer_ids,
        structural_normal_form=form,
        operator_skeleton_id=skeleton_id,
        slot_bindings=_slot_bindings(form),
    )


def _observation_key(
    demand: FrontierDemand,
    context_shape: tuple[object, ...],
    observations: tuple[str, ...],
) -> str:
    return _sha256(
        b"typed-semantic-version-space-v2-observation\0",
        {
            "environment": demand.environment_type_labels,
            "result": demand.result_type_label,
            "role": demand.use_role,
            "context_shape": context_shape,
            "observations": observations,
        },
    )


def _semantic_layer_id(key: str, inside_cost: int) -> str:
    return _sha256(
        b"typed-semantic-version-space-v2-cost-layer\0",
        {"observation": key, "inside_cost": inside_cost},
    )


def _freeze_observation(accumulator: _ObservationAccumulator) -> ObservationClass:
    layers = tuple(
        ObservationLayer(
            inside_cost=cost,
            semantic_cost_layer_id=_semantic_layer_id(accumulator.key, cost),
            canonical_recipe_ids=tuple(recipe_ids),
        )
        for cost, recipe_ids in sorted(accumulator.recipe_ids_by_cost.items())
    )
    all_ids = tuple(
        recipe_id for layer in layers for recipe_id in layer.canonical_recipe_ids
    )
    return ObservationClass(
        physical_semantic_key=accumulator.key,
        demand_environment_type_labels=accumulator.demand.environment_type_labels,
        demand_result_type_label=accumulator.demand.result_type_label,
        demand_use_role=accumulator.demand.use_role,
        physical_observation_sha256s=accumulator.observations,
        cost_layers=layers,
        canonical_representative_id=all_ids[0],
        saturating_expansion_count=accumulator.count,
        alias_digest=cast(object, accumulator.alias_hasher).hexdigest(),
        first_two_witnesses=tuple(accumulator.witnesses),
    )


def _selection_manifest_digest(
    frontier_signature: TypedSignature,
    root_recipe_ids: tuple[str, ...],
    root_outputs: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
    root_programs: tuple[tp.Program, ...],
) -> str:
    """Authenticate the exact ordered physical bank exposed to selection."""

    if not (
        len(root_recipe_ids) == len(root_outputs) == len(root_programs)
    ):
        raise ValueError("selection manifest rows must have equal lengths")
    if len(set(root_recipe_ids)) != len(root_recipe_ids):
        raise ValueError("selection manifest recipe IDs must be unique")
    return _sha256(
        b"typed-semantic-version-space-v2-selection-manifest\0",
        {
            "frontier_signature": frontier_signature.to_record(),
            "rows": tuple(
                {
                    "ordinal": ordinal,
                    "recipe_id": recipe_id,
                    "physical_output_sha256s": tuple(
                        output.sha256 for output in outputs
                    ),
                    "program_sha256": program.sha256,
                    "program_tokens": program.tokens,
                    "program_signature": program.signature.to_record(),
                }
                for ordinal, (recipe_id, outputs, program) in enumerate(
                    zip(root_recipe_ids, root_outputs, root_programs, strict=True)
                )
            ),
        },
    )


class TypedSemanticVersionSpace:
    """Frozen desired-free frontier with one physical selection operation."""

    __slots__ = (
        "signature",
        "case_arguments",
        "literal_bank",
        "max_ast_cost",
        "operator",
        "status",
        "observation_classes",
        "canonical_recipes",
        "admissions",
        "root_expansions",
        "work",
        "checkpoint_sha256",
        "_selection_root_recipe_ids",
        "_selection_root_outputs",
        "_selection_root_programs",
        "_selection_manifest_sha256",
        "_root_class_by_recipe",
        "_sealed",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("TypedSemanticVersionSpace is sealed")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        signature: TypedSignature,
        case_arguments: tuple[tuple[object, ...], ...],
        literal_bank: tuple[int, ...],
        max_ast_cost: int,
        operator: operator_successor.FrozenTypedOperatorSuccessor,
        status: str,
        observation_classes: tuple[ObservationClass, ...],
        canonical_recipes: tuple[CanonicalRecipe, ...],
        admissions: tuple[RecipeAdmission, ...],
        root_expansions: tuple[RootExpansion, ...],
        root_recipe_ids: tuple[str, ...],
        root_outputs: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
        root_programs: tuple[tp.Program, ...],
        selection_manifest_sha256: str,
        work: FrontierWorkAudit,
        checkpoint_sha256: str,
    ) -> None:
        self.signature = signature
        self.case_arguments = case_arguments
        self.literal_bank = literal_bank
        self.max_ast_cost = max_ast_cost
        self.operator = operator
        self.status = status
        self.observation_classes = observation_classes
        self.canonical_recipes = canonical_recipes
        self.admissions = admissions
        self.root_expansions = root_expansions
        self._selection_root_recipe_ids = tuple(root_recipe_ids)
        self._selection_root_outputs = tuple(tuple(row) for row in root_outputs)
        self._selection_root_programs = tuple(root_programs)
        self._selection_manifest_sha256 = selection_manifest_sha256
        self.work = work
        self.checkpoint_sha256 = checkpoint_sha256
        if (
            _selection_manifest_digest(
                self.signature,
                self._selection_root_recipe_ids,
                self._selection_root_outputs,
                self._selection_root_programs,
            )
            != self._selection_manifest_sha256
        ):
            raise ValueError("selection manifest digest does not match root bank")
        self._root_class_by_recipe = MappingProxyType(
            {
                recipe_id: observation
                for observation in observation_classes
                if observation.demand_use_role == "root"
                for layer in observation.cost_layers
                for recipe_id in layer.canonical_recipe_ids
            }
        )
        object.__setattr__(self, "_sealed", True)

    @property
    def root_recipe_ids(self) -> tuple[str, ...]:
        return self._selection_root_recipe_ids

    @property
    def selection_manifest_sha256(self) -> str:
        return self._selection_manifest_sha256

    @classmethod
    def build(
        cls,
        *,
        signature: TypedSignature,
        case_arguments: Sequence[tuple[object, ...]],
        literals: Sequence[int],
        max_ast_cost: int,
        operator: operator_successor.FrozenTypedOperatorSuccessor | None = None,
    ) -> TypedSemanticVersionSpace:
        """Build and freeze a frontier without accepting desired outputs."""

        if not isinstance(signature, TypedSignature):
            raise TypeError("version-space synthesis requires a typed signature")
        arguments = tuple(tuple(row) for row in case_arguments)
        if not arguments:
            raise ValueError("version-space synthesis needs visible arguments")
        if any(len(row) != len(signature.parameters) for row in arguments):
            raise ValueError("visible arguments do not match the typed signature")
        literal_bank = tuple(sorted(set(literals)))
        if not literal_bank or not all(type(value) is int for value in literal_bank):
            raise ValueError("version-space synthesis needs exact integer literals")
        if type(max_ast_cost) is not int or not 1 <= max_ast_cost <= 9:
            raise ValueError("V2 max_ast_cost must be in [1,9]")
        if operator is None:
            operator = operator_successor.train_public_typed_operator_successor()[0]
        checkpoint_before = operator.live_checkpoint_sha256

        generator = _UseDirectedGenerator(signature, literal_bank, max_ast_cost)
        budget = _V2PhysicalBudget()
        agenda_hasher = sha256()
        agenda_hasher.update(b"typed-semantic-version-space-v2-agenda\0")
        accumulators: dict[str, _ObservationAccumulator] = {}
        layer_by_ref: dict[tuple[str, str], str] = {}
        observation_by_ref: dict[tuple[str, str], str] = {}
        unsupported_refs: set[tuple[str, str]] = set()
        recipes: dict[str, CanonicalRecipe] = {}
        skeletons: set[str] = set()
        admitted = 0
        unsupported_pruned = 0
        maximum_stage = 0
        root_ids: list[str] = []
        root_outputs: list[tuple[output_cells.TypedOutputEncoding, ...]] = []
        root_programs: list[tp.Program] = []
        admissions: list[RecipeAdmission] = []
        root_expansions: list[RootExpansion] = []
        physical_outputs_by_factor: dict[
            tuple[object, ...], tuple[output_cells.TypedOutputEncoding, ...]
        ] = {}
        physical_effect_cache: dict[
            str,
            tuple[
                output_cells.TypedOutputEncoding,
                tuple[operator_successor.OperatorRace, ...],
            ],
        ] = {}
        denied: _FrontierDenied | None = None
        evaluators: tuple[semantic_v1._CaseEvaluator, ...] | None = None

        for complete_root_cost in range(1, max_ast_cost + 1):
            stage = generator.stage(complete_root_cost)
            dag = generator.dag_snapshot()
            if evaluators is None:
                evaluators = tuple(
                    _V2CaseEvaluator(
                        operator,
                        dag,
                        row,
                        ascii_catalog_enabled=True,
                        budget=cast(semantic_v1._SemanticFrontierBudget, budget),
                        effect_cache=physical_effect_cache,
                    )
                    for row in arguments
                )
            else:
                for evaluator in evaluators:
                    evaluator.extend(dag)

            grouped: dict[tuple[int, int, int, int], list[_PendingRecipe]] = {}
            for pending in stage:
                key = (
                    pending.demand.minimum_complete_root_cost,
                    _USE_ROLE_ORDER[pending.demand.use_role],
                    pending.demand.inside_cost,
                    generator.production_order[
                        pending.expression.node.provenance.production
                    ],
                )
                grouped.setdefault(key, []).append(pending)

            for group_key in sorted(grouped):
                resolved: list[tuple[CanonicalRecipe, _PendingRecipe]] = []
                for pending in grouped[group_key]:
                    if any(ref in unsupported_refs for ref in pending.child_refs):
                        unsupported_pruned += 1
                        unsupported_refs.add(
                            (pending.expression.node.node_id, pending.demand.use_role)
                        )
                        continue
                    child_layers = tuple(
                        layer_by_ref[ref]
                        for ref in pending.child_refs
                        if ref in layer_by_ref
                    )
                    if len(child_layers) != len(pending.child_refs):
                        # A child with no supported physical observation cannot
                        # license a higher recipe.
                        unsupported_pruned += 1
                        unsupported_refs.add(
                            (pending.expression.node.node_id, pending.demand.use_role)
                        )
                        continue
                    resolved.append((_recipe(pending, child_layers), pending))
                resolved.sort(key=lambda item: item[0].canonical_recipe_id)

                for recipe, pending in resolved:
                    attempted = admitted + 1
                    if attempted > MAX_ADMITTED_CANONICAL_RECIPES:
                        denied = _FrontierDenied(
                            bound_name="admitted-canonical-recipes",
                            limit=MAX_ADMITTED_CANONICAL_RECIPES,
                            completed=admitted,
                            attempted=attempted,
                            demand=pending.demand,
                        )
                        break
                    admitted = attempted
                    maximum_stage = max(
                        maximum_stage,
                        pending.demand.minimum_complete_root_cost,
                    )
                    recipes.setdefault(recipe.canonical_recipe_id, recipe)
                    skeletons.add(recipe.operator_skeleton_id)
                    admission = RecipeAdmission(
                        ordinal=admitted,
                        canonical_recipe_id=recipe.canonical_recipe_id,
                        use_role=pending.demand.use_role,
                        inside_cost=pending.demand.inside_cost,
                        minimum_complete_root_cost=(
                            pending.demand.minimum_complete_root_cost
                        ),
                    )
                    admissions.append(admission)
                    agenda_hasher.update(
                        _canonical_json(
                            {
                                "minimum_complete_root_cost": (
                                    pending.demand.minimum_complete_root_cost
                                ),
                                "use_role_order": _USE_ROLE_ORDER[
                                    pending.demand.use_role
                                ],
                                "inside_cost": pending.demand.inside_cost,
                                "production_order": generator.production_order[
                                    recipe.production
                                ],
                                "canonical_recipe_id": recipe.canonical_recipe_id,
                            }
                        )
                    )

                    demand = pending.demand
                    budget.set_demand(demand)
                    if demand.use_role == "binder":
                        observations = recipe.child_semantic_cost_layer_ids
                        encoded_outputs: tuple[output_cells.TypedOutputEncoding, ...] = ()
                    else:
                        physical_factor_key: tuple[object, ...] = (
                            demand.environment_type_labels,
                            demand.result_type_label,
                            demand.use_role,
                            recipe.production,
                            (
                                tuple(observation_by_ref[ref] for ref in pending.child_refs)
                                if recipe.child_semantic_cost_layer_ids
                                else (recipe.structural_normal_form,)
                            ),
                        )
                        cached_outputs = physical_outputs_by_factor.get(
                            physical_factor_key
                        )
                        if cached_outputs is None:
                            contexts = _contexts(
                                signature,
                                arguments,
                                cast(
                                    tuple[semantic_v1._CaseEvaluator, ...],
                                    evaluators,
                                ),
                                demand.use_role,
                            )
                            if not contexts:
                                unsupported_refs.add(
                                    (pending.expression.node.node_id, demand.use_role)
                                )
                                continue
                            evaluated: list[output_cells.TypedOutputEncoding] = []
                            try:
                                for evaluator, bound in contexts:
                                    output = evaluator.evaluate(
                                        pending.expression.node.node_id,
                                        bound=bound,
                                    )
                                    if not isinstance(
                                        output, output_cells.TypedOutputEncoding
                                    ):
                                        raise RuntimeError(
                                            "compiler-internal value escaped a V2 recipe"
                                        )
                                    evaluated.append(output)
                            except _FrontierDenied as error:
                                denied = error
                                break
                            encoded_outputs = tuple(evaluated)
                            physical_outputs_by_factor[physical_factor_key] = (
                                encoded_outputs
                            )
                        else:
                            encoded_outputs = cached_outputs
                        if any(not output.supported for output in encoded_outputs):
                            unsupported_refs.add(
                                (pending.expression.node.node_id, demand.use_role)
                            )
                            continue
                        observations = tuple(
                            output.sha256 for output in encoded_outputs
                        )
                    context_shape = _context_shape(signature, arguments, demand.use_role)
                    observation_key = _observation_key(
                        demand,
                        context_shape,
                        tuple(observations),
                    )
                    accumulator = accumulators.get(observation_key)
                    if accumulator is None:
                        alias_hasher = sha256()
                        alias_hasher.update(
                            b"typed-semantic-version-space-v2-aliases\0"
                        )
                        accumulator = _ObservationAccumulator(
                            key=observation_key,
                            demand=demand,
                            observations=tuple(observations),
                            recipe_ids_by_cost={},
                            witnesses=[],
                            alias_hasher=alias_hasher,
                        )
                        accumulators[observation_key] = accumulator
                    accumulator.add(
                        demand.inside_cost,
                        recipe.canonical_recipe_id,
                        pending.witness,
                    )
                    layer_by_ref[(pending.expression.node.node_id, demand.use_role)] = (
                        _semantic_layer_id(observation_key, demand.inside_cost)
                    )
                    observation_by_ref[
                        (pending.expression.node.node_id, demand.use_role)
                    ] = observation_key
                    if demand.use_role == "root":
                        program = tp.Program(pending.expression.node.tokens, signature)
                        tp.compile(program)
                        root_ids.append(recipe.canonical_recipe_id)
                        root_outputs.append(encoded_outputs)
                        root_programs.append(program)
                        root_expansions.append(
                            RootExpansion(
                                admission=admission,
                                program=program,
                                source=tp.compile(program).source,
                            )
                        )
                if denied is not None:
                    break
            if denied is not None:
                break

        checkpoint_after = operator.live_checkpoint_sha256
        if checkpoint_after != checkpoint_before:
            raise RuntimeError("typed operator changed during V2 frontier construction")

        observations = tuple(
            sorted(
                (_freeze_observation(item) for item in accumulators.values()),
                key=lambda item: item.physical_semantic_key,
            )
        )
        recipe_tuple = tuple(
            sorted(recipes.values(), key=lambda item: item.canonical_recipe_id)
        )
        root_id_tuple = tuple(root_ids)
        root_output_tuple = tuple(root_outputs)
        root_program_tuple = tuple(root_programs)
        selection_manifest_sha = _selection_manifest_digest(
            signature,
            root_id_tuple,
            root_output_tuple,
            root_program_tuple,
        )
        frontier_payload = {
            "schema": "typed-semantic-version-space-v2",
            "signature": signature.to_record(),
            "literal_bank": literal_bank,
            "max_ast_cost": max_ast_cost,
            "status": FRONTIER_READY if denied is None else STOP_FRONTIER_WORK_BOUND,
            "agenda_sha256": agenda_hasher.hexdigest(),
            "operator_races_sha256": budget.race_sha256,
            "admitted": admitted,
            "predictions": budget.completed_operator_predictions,
            "races": budget.completed_physical_races,
            "recipes": tuple(item.canonical_recipe_id for item in recipe_tuple),
            "observations": tuple(
                (
                    item.physical_semantic_key,
                    item.alias_digest,
                    item.saturating_expansion_count,
                )
                for item in observations
            ),
            "selection_manifest_sha256": selection_manifest_sha,
            "checkpoint": checkpoint_before,
        }
        frontier_sha = _sha256(
            b"typed-semantic-version-space-v2-frontier\0",
            frontier_payload,
        )
        work = FrontierWorkAudit(
            status=FRONTIER_READY if denied is None else STOP_FRONTIER_WORK_BOUND,
            bound_name=None if denied is None else denied.bound_name,
            limit=None if denied is None else denied.limit,
            completed=admitted if denied is None else denied.completed,
            attempted=admitted if denied is None else denied.attempted,
            minimum_complete_root_cost=(
                None if denied is None else denied.demand.minimum_complete_root_cost
            ),
            inside_cost=None if denied is None else denied.demand.inside_cost,
            admitted_canonical_recipes=admitted,
            unique_canonical_recipes=len(recipes),
            operator_skeletons=len(skeletons),
            operator_predictions=budget.completed_operator_predictions,
            physical_races=budget.completed_physical_races,
            unsupported_pruned_recipes=unsupported_pruned,
            normalization_raw_expansions=generator.counters["raw"],
            normalization_merged_expansions=generator.counters["merged"],
            maximum_admitted_complete_root_cost=maximum_stage,
            agenda_sha256=agenda_hasher.hexdigest(),
            operator_races_sha256=budget.race_sha256,
            frontier_sha256=frontier_sha,
        )
        return cls(
            signature=signature,
            case_arguments=arguments,
            literal_bank=literal_bank,
            max_ast_cost=max_ast_cost,
            operator=operator,
            status=work.status,
            observation_classes=observations,
            canonical_recipes=recipe_tuple,
            admissions=tuple(admissions),
            root_expansions=tuple(root_expansions),
            root_recipe_ids=root_id_tuple,
            root_outputs=root_output_tuple,
            root_programs=root_program_tuple,
            selection_manifest_sha256=selection_manifest_sha,
            work=work,
            checkpoint_sha256=checkpoint_before,
        )

    def select(self, desired_outputs: Sequence[object]) -> VersionSpaceSelection:
        """Run the existing physical exact-vector race over the frozen roots."""

        if self.status != FRONTIER_READY:
            raise RuntimeError("an incomplete V2 frontier cannot expose a desired vector")
        live_manifest_sha = _selection_manifest_digest(
            self.signature,
            self._selection_root_recipe_ids,
            self._selection_root_outputs,
            self._selection_root_programs,
        )
        if live_manifest_sha != self._selection_manifest_sha256:
            raise RuntimeError("V2 selection manifest authentication failed")
        before = self.operator.live_checkpoint_sha256
        if before != self.checkpoint_sha256:
            raise RuntimeError("V2 operator checkpoint changed before selection")
        desired_raw = tuple(desired_outputs)
        if len(desired_raw) != len(self.case_arguments):
            raise ValueError("desired outputs must align with visible arguments")
        desired = tuple(
            semantic_v1._encode(value, self.signature.result)
            for value in desired_raw
        )
        race = semantic_race.select_exact_semantic_vector(
            self._selection_root_recipe_ids,
            self._selection_root_outputs,
            desired,
        )
        after = self.operator.live_checkpoint_sha256
        if after != before:
            raise RuntimeError("V2 operator checkpoint changed during selection")

        exact_ids = race.exact_winner_keys
        observation = (
            None if not exact_ids else self._root_class_by_recipe[exact_ids[0]]
        )
        if race.status == semantic_race.SELECTED:
            winner = cast(str, race.winner_key)
            winner_ordinal = self._selection_root_recipe_ids.index(winner)
            program = self._selection_root_programs[winner_ordinal]
            return VersionSpaceSelection(
                status=SELECTED,
                stop_reason=None,
                selected_recipe_id=winner,
                program=program,
                source=tp.compile(program).source,
                exact_recipe_ids=exact_ids,
                observation_class=observation,
                race=race,
                frontier_sha256=self.work.frontier_sha256,
                checkpoint_before_sha256=before,
                checkpoint_after_sha256=after,
            )
        if race.status == semantic_race.STOP_MULTIPLE_EXACT:
            status = STOP_AMBIGUOUS
            reason = "visible examples admit multiple proof-distinct recipes"
        elif race.status == semantic_race.STOP_NO_EXACT:
            status = STOP_NO_EXACT
            reason = "no physical root recipe matches every visible example"
        else:
            raise RuntimeError("semantic race returned an unknown V2 outcome")
        return VersionSpaceSelection(
            status=status,
            stop_reason=reason,
            selected_recipe_id=None,
            program=None,
            source=None,
            exact_recipe_ids=exact_ids,
            observation_class=observation,
            race=race,
            frontier_sha256=self.work.frontier_sha256,
            checkpoint_before_sha256=before,
            checkpoint_after_sha256=after,
        )


__all__ = [
    "CanonicalRecipe",
    "FRONTIER_READY",
    "FrontierDemand",
    "FrontierWorkAudit",
    "MAX_ADMITTED_CANONICAL_RECIPES",
    "MAX_OPERATOR_PREDICTIONS",
    "MAX_PHYSICAL_RACES",
    "ObservationClass",
    "ObservationLayer",
    "RecipeAdmission",
    "RootExpansion",
    "SELECTED",
    "STOP_AMBIGUOUS",
    "STOP_FRONTIER_WORK_BOUND",
    "STOP_NO_EXACT",
    "TypedSemanticVersionSpace",
    "VersionSpaceSelection",
]
