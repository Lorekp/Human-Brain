"""Bounded bottom-up enumeration for typed physical semantic synthesis.

The module deliberately stops before candidate ranking.  It constructs a
compiler-validated provenance DAG in increasing AST cost and groups complete
programs only when a caller supplies their physical output-cell observations.
No prompt, task, family, oracle, sandbox, or host-interpreter input crosses the
interface.

This first bounded fragment covers the productions needed by the current
public source-edit bridge: scalar integer offsets, ``List[Int]`` map offsets,
and ``List[Int]`` map absolute-value programs.  The bounds are part of the
interface rather than an accidental search cutoff.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

import humanbrain.typed_program as typed_program
from humanbrain.typed_program_types import ProgramType, TypedSignature


MAX_BRIDGE_AST_COST = 6
MAX_SCALAR_EXPRESSION_AST_COST = 3

_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)
_FUNCTION_INT_INT_LABEL = "Function[Int,Int]"
_FRAGMENT_PRODUCTIONS = (
    "add",
    "subtract",
    "absolute",
    "map",
    "bind",
    "variable-0",
    "variable-1",
    "integer",
)


def _type_label(value: ProgramType) -> str:
    if not value.arguments:
        return value.constructor
    arguments = ",".join(_type_label(item) for item in value.arguments)
    return f"{value.constructor}[{arguments}]"


@dataclass(frozen=True, slots=True)
class CandidateProvenance:
    """One compiler production and the DAG nodes supplied to its operands."""

    production: str
    operand_node_ids: tuple[str, ...] = ()
    variable_index: int | None = None
    literal_value: int | None = None

    def __post_init__(self) -> None:
        if self.production not in _FRAGMENT_PRODUCTIONS:
            raise ValueError("provenance production is outside the bounded grammar")
        if not isinstance(self.operand_node_ids, tuple) or not all(
            isinstance(node_id, str) and len(node_id) == 64
            for node_id in self.operand_node_ids
        ):
            raise TypeError("operand provenance must contain SHA-256 node identities")
        if (
            self.variable_index is not None
            and self.production != f"variable-{self.variable_index}"
        ):
            raise ValueError("variable provenance must name its compiler production")
        if self.literal_value is not None and self.production != "integer":
            raise ValueError("literal provenance requires the integer production")
        if self.variable_index is not None and self.literal_value is not None:
            raise ValueError("one provenance node cannot be both variable and literal")


@dataclass(frozen=True, slots=True)
class CandidateNode:
    """One typed expression node retained in the bottom-up provenance DAG."""

    node_id: str
    ast_cost: int
    environment_type_labels: tuple[str, ...]
    result_type_label: str
    tokens: tuple[str, ...]
    provenance: CandidateProvenance


@dataclass(frozen=True, slots=True)
class TypedCandidate:
    """One complete compiler-validated program rooted at a DAG node."""

    root_node_id: str
    ast_cost: int
    program: typed_program.Program


@dataclass(frozen=True, slots=True)
class CandidateTier:
    """All complete candidates at one exact compiler AST cost."""

    ast_cost: int
    candidates: tuple[TypedCandidate, ...]


@dataclass(frozen=True, slots=True)
class TypedCandidateDag:
    """A bounded typed expression DAG plus its complete-program cost tiers."""

    signature: TypedSignature
    literal_bank: tuple[int, ...]
    max_ast_cost: int
    nodes: tuple[CandidateNode, ...]
    tiers: tuple[CandidateTier, ...]


@dataclass(frozen=True, slots=True)
class TypedPhysicalSemanticVector:
    """Caller-supplied exact physical observations in visible-case order."""

    result_type: ProgramType
    observations: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.result_type, ProgramType):
            raise TypeError("semantic-vector result_type must be a ProgramType")
        if not isinstance(self.observations, tuple) or not self.observations:
            raise ValueError("semantic vector needs at least one visible observation")
        if not all(type(observation) is bytes for observation in self.observations):
            raise TypeError("physical semantic observations must be exact bytes")


@dataclass(frozen=True, slots=True)
class SemanticEquivalenceClass:
    """One typed observational class with its canonical first representative."""

    vector: TypedPhysicalSemanticVector
    representative_candidate_id: str
    member_candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObservationalDeduplication:
    """Deterministic observational classes without a task-level winner."""

    case_count: int
    observed_candidate_count: int
    representative_candidate_ids: tuple[str, ...]
    classes: tuple[SemanticEquivalenceClass, ...]


class _NodeBuilder:
    def __init__(self, production_order: Mapping[str, int]) -> None:
        self.production_order = production_order
        self.nodes: dict[str, CandidateNode] = {}

    def add(
        self,
        *,
        ast_cost: int,
        environment: tuple[ProgramType, ...],
        result_type_label: str,
        tokens: tuple[str, ...],
        provenance: CandidateProvenance,
    ) -> CandidateNode:
        record = {
            "ast_cost": ast_cost,
            "environment": [_type_label(value) for value in environment],
            "result_type": result_type_label,
            "tokens": list(tokens),
            "production": provenance.production,
            "operands": list(provenance.operand_node_ids),
            "variable_index": provenance.variable_index,
            "literal_value": provenance.literal_value,
        }
        node_id = sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        node = CandidateNode(
            node_id=node_id,
            ast_cost=ast_cost,
            environment_type_labels=tuple(
                _type_label(value) for value in environment
            ),
            result_type_label=result_type_label,
            tokens=tokens,
            provenance=provenance,
        )
        incumbent = self.nodes.setdefault(node_id, node)
        if incumbent != node:
            raise RuntimeError("candidate node identity collision")
        return incumbent

    def sort_key(self, node: CandidateNode) -> tuple[object, ...]:
        return (
            node.ast_cost,
            self.production_order[node.provenance.production],
            node.tokens,
            node.environment_type_labels,
            node.node_id,
        )


def _compiler_production_order() -> dict[str, int]:
    specs = tuple(typed_program._PRIMITIVE_SPECS)
    by_token = {spec.token: spec for spec in specs}
    expected = {
        "add": ("operation", 2),
        "subtract": ("operation", 2),
        "absolute": ("operation", 1),
        "map": ("operation", 2),
        "bind": ("binder", 0),
        "variable-0": ("variable", 0),
        "variable-1": ("variable", 0),
        "integer": ("literal", 0),
    }
    for token, (role, arity) in expected.items():
        spec = by_token.get(token)
        if spec is None or spec.role != role or len(spec.arguments) != arity:
            raise RuntimeError(
                "typed-program grammar no longer satisfies bounded production "
                f"{token!r}"
            )
    return {token: index for index, token in enumerate(spec.token for spec in specs)}


def _literal_bank(literals: Sequence[int]) -> tuple[int, ...]:
    if isinstance(literals, (str, bytes, bytearray)):
        raise TypeError("literal bank must be an integer sequence")
    try:
        supplied = tuple(literals)
    except TypeError as error:
        raise TypeError("literal bank must be an integer sequence") from error
    if not supplied or not all(type(value) is int for value in supplied):
        raise ValueError("literal bank must contain exact integers")
    bound = 10 ** typed_program._LITERAL_DIGITS - 1
    if any(abs(value) > bound for value in supplied):
        raise ValueError("literal exceeds the compiler's fixed digit bound")
    return tuple(sorted(set(supplied)))


def _variable_nodes(
    builder: _NodeBuilder,
    environment: tuple[ProgramType, ...],
    result_type: ProgramType,
) -> tuple[CandidateNode, ...]:
    nodes = []
    for index, value_type in enumerate(environment[:2]):
        if value_type == result_type:
            token = f"variable-{index}"
            nodes.append(
                builder.add(
                    ast_cost=1,
                    environment=environment,
                    result_type_label=_type_label(result_type),
                    tokens=(token,),
                    provenance=CandidateProvenance(
                        token,
                        variable_index=index,
                    ),
                )
            )
    return tuple(nodes)


def _scalar_nodes(
    builder: _NodeBuilder,
    environment: tuple[ProgramType, ...],
    literals: tuple[int, ...],
    max_ast_cost: int,
) -> tuple[CandidateNode, ...]:
    leaves = list(_variable_nodes(builder, environment, _INT))
    for value in literals:
        leaves.append(
            builder.add(
                ast_cost=1,
                environment=environment,
                result_type_label="Int",
                tokens=typed_program._factor_literal_tokens(value),
                provenance=CandidateProvenance(
                    "integer",
                    literal_value=value,
                ),
            )
        )
    leaves.sort(key=builder.sort_key)
    nodes = list(leaves)

    if max_ast_cost >= 2:
        for operand in leaves:
            nodes.append(
                builder.add(
                    ast_cost=2,
                    environment=environment,
                    result_type_label="Int",
                    tokens=("absolute", *operand.tokens),
                    provenance=CandidateProvenance(
                        "absolute",
                        (operand.node_id,),
                    ),
                )
            )

    if max_ast_cost >= 3:
        for production in ("add", "subtract"):
            for left in leaves:
                for right in leaves:
                    nodes.append(
                        builder.add(
                            ast_cost=3,
                            environment=environment,
                            result_type_label="Int",
                            tokens=(production, *left.tokens, *right.tokens),
                            provenance=CandidateProvenance(
                                production,
                                (left.node_id, right.node_id),
                            ),
                        )
                    )
    return tuple(sorted(nodes, key=builder.sort_key))


def _list_nodes(
    builder: _NodeBuilder,
    environment: tuple[ProgramType, ...],
    literals: tuple[int, ...],
    max_ast_cost: int,
) -> tuple[CandidateNode, ...]:
    collections = list(_variable_nodes(builder, environment, _LIST_INT))
    nodes = list(collections)
    if max_ast_cost < 4 or not collections:
        return tuple(sorted(nodes, key=builder.sort_key))

    body_environment = (_INT, *environment)
    body_nodes = _scalar_nodes(
        builder,
        body_environment,
        literals,
        min(MAX_SCALAR_EXPRESSION_AST_COST, max_ast_cost - 3),
    )
    for body in body_nodes:
        for collection in collections:
            map_cost = 2 + body.ast_cost + collection.ast_cost
            if map_cost > max_ast_cost:
                continue
            binder = builder.add(
                ast_cost=1 + body.ast_cost,
                environment=environment,
                result_type_label=_FUNCTION_INT_INT_LABEL,
                tokens=("bind", *body.tokens),
                provenance=CandidateProvenance("bind", (body.node_id,)),
            )
            nodes.append(
                builder.add(
                    ast_cost=map_cost,
                    environment=environment,
                    result_type_label="List[Int]",
                    tokens=("map", *binder.tokens, *collection.tokens),
                    provenance=CandidateProvenance(
                        "map",
                        (binder.node_id, collection.node_id),
                    ),
                )
            )
    return tuple(sorted(nodes, key=builder.sort_key))


def enumerate_typed_candidate_dag(
    signature: TypedSignature,
    *,
    literals: Sequence[int],
    max_ast_cost: int = MAX_BRIDGE_AST_COST,
) -> TypedCandidateDag:
    """Enumerate the bounded bridge fragment by exact increasing AST cost.

    Every returned complete candidate is validated and lowered by
    :mod:`humanbrain.typed_program`.  The function does not execute, score, or
    select a candidate.
    """

    if not isinstance(signature, TypedSignature):
        raise TypeError("typed synthesis requires a TypedSignature")
    if type(max_ast_cost) is not int or not 1 <= max_ast_cost <= MAX_BRIDGE_AST_COST:
        raise ValueError(
            f"max_ast_cost must be in [1,{MAX_BRIDGE_AST_COST}] for this fragment"
        )
    literal_bank = _literal_bank(literals)
    builder = _NodeBuilder(_compiler_production_order())
    environment = tuple(
        parameter.type for parameter in reversed(signature.parameters)
    )

    if signature.result == _INT:
        roots = _scalar_nodes(
            builder,
            environment,
            literal_bank,
            min(max_ast_cost, MAX_SCALAR_EXPRESSION_AST_COST),
        )
    elif signature.result == _LIST_INT:
        roots = _list_nodes(
            builder,
            environment,
            literal_bank,
            max_ast_cost,
        )
    else:
        raise ValueError(
            "bounded synthesis currently supports Int or List[Int] results"
        )

    candidates_by_cost: dict[int, list[TypedCandidate]] = {
        cost: [] for cost in range(1, max_ast_cost + 1)
    }
    for root in roots:
        if root.ast_cost > max_ast_cost:
            continue
        program = typed_program.Program(root.tokens, signature)
        typed_program.compile(program)
        candidates_by_cost[root.ast_cost].append(
            TypedCandidate(root.node_id, root.ast_cost, program)
        )

    tiers = tuple(
        CandidateTier(
            ast_cost=cost,
            candidates=tuple(candidates_by_cost[cost]),
        )
        for cost in range(1, max_ast_cost + 1)
    )
    return TypedCandidateDag(
        signature=signature,
        literal_bank=literal_bank,
        max_ast_cost=max_ast_cost,
        nodes=tuple(sorted(builder.nodes.values(), key=builder.sort_key)),
        tiers=tiers,
    )


def deduplicate_physical_semantics(
    dag: TypedCandidateDag,
    vectors_by_candidate_id: Mapping[str, TypedPhysicalSemanticVector],
) -> ObservationalDeduplication:
    """Group observed candidates by exact typed physical semantic vector.

    The caller decides how visible cases are evaluated and supplies immutable
    output-cell bytes.  The first candidate in deterministic cost order is the
    canonical representative of an equivalence class; it is not a task-level
    winner or ranking decision.  A subset may be supplied for incremental
    physical evaluation.
    """

    if not isinstance(dag, TypedCandidateDag):
        raise TypeError("observational deduplication requires a TypedCandidateDag")
    if not isinstance(vectors_by_candidate_id, Mapping) or not vectors_by_candidate_id:
        raise ValueError("at least one caller-supplied semantic vector is required")

    candidates = tuple(
        candidate for tier in dag.tiers for candidate in tier.candidates
    )
    candidate_ids = {candidate.root_node_id for candidate in candidates}
    unknown = set(vectors_by_candidate_id) - candidate_ids
    if unknown:
        raise ValueError("semantic vectors reference candidates outside the DAG")

    groups: dict[TypedPhysicalSemanticVector, list[str]] = {}
    case_count: int | None = None
    observed = 0
    for candidate in candidates:
        vector = vectors_by_candidate_id.get(candidate.root_node_id)
        if vector is None:
            continue
        if not isinstance(vector, TypedPhysicalSemanticVector):
            raise TypeError("semantic-vector mapping contains an invalid value")
        if vector.result_type != dag.signature.result:
            raise ValueError("semantic-vector type does not match the candidate result")
        if case_count is None:
            case_count = len(vector.observations)
        elif len(vector.observations) != case_count:
            raise ValueError(
                "all semantic vectors must use the same visible-case order"
            )
        groups.setdefault(vector, []).append(candidate.root_node_id)
        observed += 1

    if not observed or case_count is None:
        raise ValueError("no candidate in the DAG has a supplied semantic vector")
    classes = tuple(
        SemanticEquivalenceClass(
            vector=vector,
            representative_candidate_id=members[0],
            member_candidate_ids=tuple(members),
        )
        for vector, members in groups.items()
    )
    return ObservationalDeduplication(
        case_count=case_count,
        observed_candidate_count=observed,
        representative_candidate_ids=tuple(
            equivalence.representative_candidate_id for equivalence in classes
        ),
        classes=classes,
    )


__all__ = [
    "CandidateNode",
    "CandidateProvenance",
    "CandidateTier",
    "MAX_BRIDGE_AST_COST",
    "MAX_SCALAR_EXPRESSION_AST_COST",
    "ObservationalDeduplication",
    "SemanticEquivalenceClass",
    "TypedCandidate",
    "TypedCandidateDag",
    "TypedPhysicalSemanticVector",
    "deduplicate_physical_semantics",
    "enumerate_typed_candidate_dag",
]
