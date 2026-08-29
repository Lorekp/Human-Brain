"""Bottom-up typed synthesis driven only by physical execution semantics.

The compiler enumerates a bounded, task-blind program DAG.  A frozen fitted-cell
operator successor evaluates every compound node over the caller's visible
cases.  Exact physical output vectors are deduplicated by the reusable compiler
module, then one STOP-bounded fitted-cell tier race selects the first exact AST
cost tier.  Prompt text and task/family identities never enter this path.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np

from experiments.r3_pilot import public_typed_operator_successor as operator_successor
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from humanbrain import typed_program as tp
from humanbrain import typed_semantic_synthesis as synthesis
from humanbrain.typed_program_types import ProgramType, TypedSignature


STATUS_DONE = "DONE"
STATUS_STOP = "STOP_NO_EXACT_TYPED_SEMANTIC_PROGRAM"

_INT = ProgramType.Int()
_BOOL = ProgramType.Bool()
_CHAR = ProgramType.Char()
_STRING = ProgramType.String()
_LIST_INT = ProgramType.List(_INT)

_ABSOLUTE = tp.FactorAction((), "APPLY1", "absolute", "SELECTED", "NONE", None)
_BINARY_ACTIONS = {
    operator: tp.FactorAction(
        (), "APPLY2", operator, "SELECTED", "SCOPE0", None
    )
    for operator in ("add", "subtract", "minimum", "maximum", "less-than")
}
_MAP_COPY = tp.FactorAction(
    (), "MAP_BIND", "map", "BOUND0", "SELECTED", None
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _output_type(value: ProgramType) -> output_cells.OutputType:
    if value == _INT:
        return output_cells.OutputType.INT
    if value == _BOOL:
        return output_cells.OutputType.BOOL
    if value == _LIST_INT:
        return output_cells.OutputType.LIST_INT
    if value == _CHAR:
        return output_cells.OutputType.CHAR
    if value == _STRING:
        return output_cells.OutputType.STRING
    raise ValueError(
        "physical synthesis supports only Int, Bool, Char, String, and List[Int]"
    )


def _encode(
    value: object,
    value_type: ProgramType,
    *,
    require_supported: bool = True,
) -> output_cells.TypedOutputEncoding:
    if value_type == _LIST_INT and isinstance(value, tuple):
        value = list(value)
    encoded = output_cells.TypedOutputCellBank.encode(
        value,
        declared_type=_output_type(value_type),
    )
    if require_supported and not encoded.supported:
        raise ValueError("visible synthesis value exceeds the physical output bank")
    return encoded


def _unsupported(value_type: ProgramType) -> output_cells.TypedOutputEncoding:
    if value_type == _INT:
        value: object = output_cells.INT_MAX + 1
    elif value_type == _BOOL:
        value = None
    elif value_type == _LIST_INT:
        value = [output_cells.INT_MAX + 1]
    elif value_type == _CHAR:
        value = "\n"
    elif value_type == _STRING:
        value = "\n"
    else:
        raise ValueError(
            "physical synthesis supports only Int, Bool, Char, String, and List[Int]"
        )
    return output_cells.TypedOutputCellBank.encode(
        value,
        declared_type=_output_type(value_type),
    )


def _decode_int(value: output_cells.TypedOutputEncoding) -> int:
    if not value.supported or value.output_type is not output_cells.OutputType.INT:
        raise RuntimeError("physical operator did not produce a supported Int")
    cells = np.flatnonzero(value.int_cells)
    if len(cells) != 1 or int(cells[0]) == output_cells.INT_UNKNOWN_CELL:
        raise RuntimeError("physical Int output is not uniquely decodable")
    return output_cells.INT_MIN + int(cells[0])


def _decode_bool(value: output_cells.TypedOutputEncoding) -> bool:
    if not value.supported or value.output_type is not output_cells.OutputType.BOOL:
        raise RuntimeError("physical operator did not produce a supported Bool")
    cells = np.flatnonzero(value.bool_cells)
    if len(cells) != 1 or int(cells[0]) == output_cells.BOOL_UNKNOWN_CELL:
        raise RuntimeError("physical Bool output is not uniquely decodable")
    return int(cells[0]) == output_cells.BOOL_TRUE_CELL


def _decode_list(value: output_cells.TypedOutputEncoding) -> list[int]:
    if not value.supported or value.output_type is not output_cells.OutputType.LIST_INT:
        raise RuntimeError("physical operator did not produce a supported List[Int]")
    lengths = np.flatnonzero(value.list_length_cells)
    if len(lengths) != 1 or int(lengths[0]) > output_cells.LIST_CAPACITY:
        raise RuntimeError("physical List length is not uniquely decodable")
    length = int(lengths[0])
    decoded: list[int] = []
    for position in range(length):
        item_cells = np.flatnonzero(value.list_item_cells[position])
        if len(item_cells) != 1 or int(item_cells[0]) == output_cells.INT_UNKNOWN_CELL:
            raise RuntimeError("physical List item is not uniquely decodable")
        decoded.append(output_cells.INT_MIN + int(item_cells[0]))
    return decoded


def _decode_string(value: output_cells.TypedOutputEncoding) -> str:
    if not value.supported or value.output_type is not output_cells.OutputType.STRING:
        raise RuntimeError("physical operator did not produce a supported String")
    lengths = np.flatnonzero(value.string_length_cells)
    if len(lengths) != 1 or int(lengths[0]) > output_cells.STRING_CAPACITY:
        raise RuntimeError("physical String length is not uniquely decodable")
    length = int(lengths[0])
    characters: list[str] = []
    for position in range(length):
        item_cells = np.flatnonzero(value.string_item_cells[position])
        if len(item_cells) != 1 or int(item_cells[0]) == output_cells.CHAR_UNKNOWN_CELL:
            raise RuntimeError("physical String character is not uniquely decodable")
        characters.append(output_cells.PRINTABLE_ASCII[int(item_cells[0])])
    return "".join(characters)


def _string_value_supported(value: object) -> bool:
    return (
        type(value) is str
        and len(value) <= output_cells.STRING_CAPACITY
        and all(
            output_cells.PRINTABLE_ASCII_MIN
            <= ord(character)
            <= output_cells.PRINTABLE_ASCII_MAX
            for character in value
        )
    )


def _string_context_shape(
    signature: TypedSignature,
    arguments: tuple[tuple[object, ...], ...],
) -> tuple[tuple[tuple[int, bool], ...], ...]:
    """Authenticate visible String lengths/support separately from outputs."""

    return tuple(
        tuple(
            (
                len(value) if type(value) is str else -1,
                _string_value_supported(value),
            )
            for parameter, value in zip(signature.parameters, row, strict=True)
            if parameter.type == _STRING
        )
        for row in arguments
    )


@dataclass(frozen=True, slots=True)
class PhysicalCandidateEvaluation:
    candidate_id: str
    outputs: tuple[output_cells.TypedOutputEncoding, ...]
    vector: synthesis.TypedPhysicalSemanticVector
    operator_predictions: int
    operator_races: int
    operator_races_sha256: str


@dataclass(frozen=True, slots=True)
class SemanticSynthesisTierAttempt:
    ast_cost: int
    evaluated_candidates: int
    representative_candidate_ids: tuple[str, ...]
    race: semantic_race.TypedSemanticVectorRaceAudit | None


@dataclass(frozen=True, slots=True)
class TypedSemanticSynthesisDecision:
    status: str
    stop_reason: str | None
    candidate_count: int
    observed_semantic_classes: int
    tier_attempts: tuple[SemanticSynthesisTierAttempt, ...]
    selected_candidate_id: str | None
    program: tp.Program | None
    source: str | None
    operator_predictions: int
    operator_races: int
    operator_races_sha256: str
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str


class _CaseEvaluator:
    def __init__(
        self,
        model: operator_successor.FrozenTypedOperatorSuccessor,
        dag: synthesis.TypedCandidateDag,
        arguments: tuple[object, ...],
        *,
        ascii_catalog_enabled: bool,
    ) -> None:
        self.model = model
        self.dag = dag
        self.arguments = arguments
        self.ascii_catalog_enabled = ascii_catalog_enabled
        self.nodes = {node.node_id: node for node in dag.nodes}
        self.memo: dict[
            tuple[str, tuple[object, ...]],
            output_cells.TypedOutputEncoding | operator_successor._AsciiClassEncoding,
        ] = {}
        self.predictions = 0
        self.races: list[operator_successor.OperatorRace] = []

    def extend(self, dag: synthesis.TypedCandidateDag) -> None:
        """Add a newly licensed cost tier without discarding physical memo state."""

        if dag.signature != self.dag.signature:
            raise ValueError("an evaluator cannot cross typed signatures")
        for node in dag.nodes:
            incumbent = self.nodes.setdefault(node.node_id, node)
            if incumbent != node:
                raise RuntimeError("candidate identity changed across retained tiers")
        self.dag = dag

    def _environment_value(
        self,
        node: synthesis.CandidateNode,
        index: int,
        bound: tuple[object, ...],
    ) -> tuple[object, ProgramType]:
        values = (*bound, *reversed(self.arguments))
        if not 0 <= index < len(values):
            raise RuntimeError("candidate variable lies outside its typed environment")
        label = node.environment_type_labels[index]
        if label == "Int":
            value_type = _INT
        elif label == "Bool":
            value_type = _BOOL
        elif label == "List[Int]":
            value_type = _LIST_INT
        elif label == "Char":
            value_type = _CHAR
        elif label == "String":
            value_type = _STRING
        else:
            raise RuntimeError("candidate variable has an unsupported physical type")
        return values[index], value_type

    def _predict(
        self,
        selected: output_cells.TypedOutputEncoding,
        action: tp.FactorAction,
        scopes: tuple[output_cells.TypedOutputEncoding, ...] = (),
    ) -> output_cells.TypedOutputEncoding:
        # ``FrozenTypedSemanticSynthesizer.synthesize`` authenticates the full
        # operator checkpoint before and after the complete readonly DAG
        # evaluation.  Re-entering ``predict`` here would hash every frozen
        # operator array twice for every node, even though no mutation boundary
        # exists between nodes.  Keep the exact physical effect/race path while
        # sharing the enclosing transaction's authentication.
        output, races = self.model._predict_effect(selected, action, scopes)
        self.predictions += 1
        self.races.extend(races)
        return output

    def _ascii_class(
        self,
        label: str,
    ) -> operator_successor._AsciiClassEncoding:
        output, races = self.model._ascii_class(
            label,
            enabled=self.ascii_catalog_enabled,
        )
        self.predictions += 1
        self.races.extend(races)
        return output

    def _contains(
        self,
        ascii_class: operator_successor._AsciiClassEncoding,
        character: output_cells.TypedOutputEncoding,
    ) -> output_cells.TypedOutputEncoding:
        output, races = self.model._contains(ascii_class, character)
        self.predictions += 1
        self.races.extend(races)
        return output

    def _string_positions(
        self,
        value: output_cells.TypedOutputEncoding,
        positions: tuple[int, ...],
        *,
        bank: str,
    ) -> output_cells.TypedOutputEncoding:
        output, races = self.model._string_positions(value, positions, bank=bank)
        self.predictions += 1
        self.races.extend(races)
        return output

    def evaluate(
        self,
        node_id: str,
        *,
        bound: tuple[object, ...] = (),
    ) -> output_cells.TypedOutputEncoding | operator_successor._AsciiClassEncoding:
        key = (node_id, bound)
        cached = self.memo.get(key)
        if cached is not None:
            return cached
        node = self.nodes[node_id]
        provenance = node.provenance
        production = provenance.production

        if production.startswith("variable-"):
            if provenance.variable_index is None:
                raise RuntimeError("variable provenance lost its index")
            raw, value_type = self._environment_value(
                node, provenance.variable_index, bound
            )
            output = _encode(
                raw,
                value_type,
                require_supported=value_type not in {_CHAR, _STRING},
            )
        elif production == "integer":
            if provenance.literal_value is None:
                raise RuntimeError("integer provenance lost its literal")
            output = _encode(provenance.literal_value, _INT)
        elif production in tp.ASCII_CLASS_LABELS:
            output = self._ascii_class(production)
        elif production == "contains":
            ascii_class = self.evaluate(
                provenance.operand_node_ids[0],
                bound=bound,
            )
            character = self.evaluate(
                provenance.operand_node_ids[1],
                bound=bound,
            )
            if not isinstance(ascii_class, operator_successor._AsciiClassEncoding):
                raise RuntimeError("contains provenance lost its ASCII class")
            if not isinstance(character, output_cells.TypedOutputEncoding):
                raise RuntimeError("contains provenance lost its Char operand")
            output = self._contains(ascii_class, character)
        elif production == "absolute":
            operand = self.evaluate(provenance.operand_node_ids[0], bound=bound)
            if not isinstance(operand, output_cells.TypedOutputEncoding):
                raise RuntimeError("absolute provenance has an internal type operand")
            output = self._predict(operand, _ABSOLUTE)
        elif production in _BINARY_ACTIONS:
            left = self.evaluate(provenance.operand_node_ids[0], bound=bound)
            right = self.evaluate(provenance.operand_node_ids[1], bound=bound)
            if not isinstance(left, output_cells.TypedOutputEncoding) or not isinstance(
                right, output_cells.TypedOutputEncoding
            ):
                raise RuntimeError("binary provenance has an internal type operand")
            output = self._predict(
                left,
                _BINARY_ACTIONS[production],
                (right,),
            )
        elif production == "map":
            binder = self.nodes[provenance.operand_node_ids[0]]
            if binder.provenance.production != "bind":
                raise RuntimeError("map provenance lost its compiler binder")
            body_id = binder.provenance.operand_node_ids[0]
            collection = self.evaluate(
                provenance.operand_node_ids[1], bound=bound
            )
            if not isinstance(collection, output_cells.TypedOutputEncoding):
                raise RuntimeError("map provenance lost its List operand")
            copied = self._predict(collection, _MAP_COPY)
            if not copied.supported:
                output = _unsupported(_LIST_INT)
            else:
                items = _decode_list(copied)
                mapped_outputs = tuple(
                    self.evaluate(body_id, bound=(item, *bound)) for item in items
                )
                if any(
                    not isinstance(item, output_cells.TypedOutputEncoding)
                    or not item.supported
                    for item in mapped_outputs
                ):
                    output = _unsupported(_LIST_INT)
                else:
                    output = _encode(
                        [
                            _decode_int(item)
                            for item in mapped_outputs
                            if isinstance(item, output_cells.TypedOutputEncoding)
                        ],
                        _LIST_INT,
                    )
        elif production == "sum":
            collection = self.evaluate(
                provenance.operand_node_ids[0], bound=bound
            )
            if not isinstance(collection, output_cells.TypedOutputEncoding):
                raise RuntimeError("sum provenance lost its List operand")
            copied = self._predict(collection, _MAP_COPY)
            if not copied.supported:
                output = _unsupported(_INT)
            else:
                output = _encode(0, _INT)
                for item in _decode_list(copied):
                    output = self._predict(
                        output,
                        _BINARY_ACTIONS["add"],
                        (_encode(item, _INT),),
                    )
                    if not output.supported:
                        break
        elif production == "count":
            binder = self.nodes[provenance.operand_node_ids[0]]
            if binder.provenance.production != "bind":
                raise RuntimeError("count provenance lost its compiler binder")
            predicate_id = binder.provenance.operand_node_ids[0]
            collection = self.evaluate(
                provenance.operand_node_ids[1], bound=bound
            )
            if not isinstance(collection, output_cells.TypedOutputEncoding):
                raise RuntimeError("count provenance lost its List operand")
            copied = self._predict(collection, _MAP_COPY)
            if not copied.supported:
                output = _unsupported(_INT)
            else:
                output = _encode(0, _INT)
                for item in _decode_list(copied):
                    predicate = self.evaluate(
                        predicate_id,
                        bound=(item, *bound),
                    )
                    if (
                        not isinstance(predicate, output_cells.TypedOutputEncoding)
                        or not predicate.supported
                    ):
                        output = _unsupported(_INT)
                        break
                    output = self._predict(
                        output,
                        _BINARY_ACTIONS["add"],
                        (_encode(int(_decode_bool(predicate)), _INT),),
                    )
                    if not output.supported:
                        break
        elif production == "filter":
            binder = self.nodes[provenance.operand_node_ids[0]]
            if binder.provenance.production != "bind":
                raise RuntimeError("filter provenance lost its compiler binder")
            predicate_id = binder.provenance.operand_node_ids[0]
            collection = self.evaluate(
                provenance.operand_node_ids[1],
                bound=bound,
            )
            if not isinstance(collection, output_cells.TypedOutputEncoding):
                raise RuntimeError("filter provenance lost its String operand")
            if not collection.supported:
                output = _unsupported(_STRING)
            else:
                source = _decode_string(collection)
                copied = self._string_positions(
                    collection,
                    tuple(range(len(source))),
                    bank="operator-filter-source-copy",
                )
                if not copied.supported:
                    output = _unsupported(_STRING)
                else:
                    selected_positions: list[int] = []
                    predicate_unsupported = False
                    for position, character in enumerate(_decode_string(copied)):
                        predicate = self.evaluate(
                            predicate_id,
                            bound=(character, *bound),
                        )
                        if (
                            not isinstance(predicate, output_cells.TypedOutputEncoding)
                            or not predicate.supported
                        ):
                            predicate_unsupported = True
                            continue
                        if _decode_bool(predicate):
                            selected_positions.append(position)
                    if predicate_unsupported:
                        output = _unsupported(_STRING)
                    else:
                        output = self._string_positions(
                            copied,
                            tuple(selected_positions),
                            bank="operator-filter-output-copy",
                        )
        elif production == "reverse":
            operand = self.evaluate(provenance.operand_node_ids[0], bound=bound)
            if not isinstance(operand, output_cells.TypedOutputEncoding):
                raise RuntimeError("reverse provenance lost its String operand")
            if not operand.supported:
                output = _unsupported(_STRING)
            else:
                source = _decode_string(operand)
                output = self._string_positions(
                    operand,
                    tuple(reversed(range(len(source)))),
                    bank="operator-reverse-output-copy",
                )
        elif production == "bind":
            raise RuntimeError("a compiler binder cannot be evaluated outside an aggregate")
        else:
            raise RuntimeError(f"unsupported semantic production {production!r}")

        self.memo[key] = output
        return output

    def race_digest(self, start: int = 0) -> str:
        digest = sha256()
        digest.update(b"public-typed-semantic-operator-races-v1\0")
        for race in self.races[start:]:
            digest.update(_canonical_json(asdict(race)))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenTypedSemanticSynthesizer:
    operator: operator_successor.FrozenTypedOperatorSuccessor
    literal_bank: tuple[int, ...]
    max_ast_cost: int
    _ascii_catalog_enabled: bool
    checkpoint_sha256: str

    @property
    def live_checkpoint_sha256(self) -> str:
        digest = sha256()
        digest.update(b"public-typed-semantic-synthesizer-v2\0")
        digest.update(self.operator.live_checkpoint_sha256.encode("ascii"))
        digest.update(_canonical_json(self.literal_bank))
        digest.update(np.asarray((self.max_ast_cost,), dtype="<i8").tobytes())
        digest.update(bytes((int(self._ascii_catalog_enabled),)))
        digest.update(b"\0string-context-retention-v1")
        digest.update(b"\0string-structural-schedule-v1\0")
        digest.update(
            _canonical_json(
                (
                    (1, tuple(tp.ASCII_CLASS_LABELS), "Char"),
                    (2, "reverse-text"),
                    (3, "contains"),
                    (4, "bind-char-bool"),
                    (6, "filter"),
                    (7, "reverse-filter"),
                )
            )
        )
        return digest.hexdigest()

    def _synthesize_retained(
        self,
        signature: TypedSignature,
        arguments: tuple[tuple[object, ...], ...],
        desired: tuple[output_cells.TypedOutputEncoding, ...],
        before: str,
    ) -> TypedSemanticSynthesisDecision:
        """Grow exact-cost tiers only from prior physical representatives."""

        retained: set[str] = set()
        # ``composable`` authenticates the task-independent compiler schedule.
        # Ordinarily it follows physical semantic retention exactly.  The
        # catalog-off control still carries the same internal String inventory
        # after its terminals physically become UNKNOWN, while unsupported
        # nodes remain excluded from ``retained`` and from every root race.
        composable: set[str] = set()
        processed: set[str] = set()
        seen_semantics: dict[tuple[object, ...], str] = {}
        root_semantics: set[tuple[bytes, ...]] = set()
        vectors: dict[str, synthesis.TypedPhysicalSemanticVector] = {}
        evaluations: dict[str, PhysicalCandidateEvaluation] = {}
        candidates_by_id: dict[str, synthesis.TypedCandidate] = {}
        attempts: list[SemanticSynthesisTierAttempt] = []
        total_predictions = 0
        total_races = 0
        all_races_digest = sha256()
        all_races_digest.update(b"public-typed-semantic-all-races-string-v1\0")
        root_environment = tuple(
            synthesis._type_label(parameter.type)
            for parameter in reversed(signature.parameters)
        )
        body_environment = ("Int", *root_environment)
        string_body_environment = ("Char", *root_environment)
        string_context_shape = _string_context_shape(signature, arguments)
        evaluators: tuple[_CaseEvaluator, ...] | None = None

        for cost in range(1, self.max_ast_cost + 1):
            # Build against the frozen final bound on every pass.  This keeps
            # internal List/binder operands structurally present at their own
            # exact cost; ``retained`` alone decides which prior physical
            # representatives may compose into later nodes.
            dag = synthesis.enumerate_typed_candidate_dag(
                signature,
                literals=self.literal_bank,
                max_ast_cost=self.max_ast_cost,
                retained_node_ids=composable,
            )
            if evaluators is None:
                evaluators = tuple(
                    _CaseEvaluator(
                        self.operator,
                        dag,
                        row,
                        ascii_catalog_enabled=self._ascii_catalog_enabled,
                    )
                    for row in arguments
                )
            else:
                for evaluator in evaluators:
                    evaluator.extend(dag)

            tier = dag.tiers[cost - 1]
            tier_candidate_ids = {
                candidate.root_node_id for candidate in tier.candidates
            }
            candidates_by_id.update(
                (candidate.root_node_id, candidate)
                for candidate in tier.candidates
            )

            for node in (
                item
                for item in dag.nodes
                if item.ast_cost == cost and item.node_id not in processed
            ):
                processed.add(node.node_id)
                if node.provenance.production == "bind":
                    if (
                        node.result_type_label == "Function[Char,Bool]"
                        and len(node.provenance.operand_node_ids) == 1
                        and node.provenance.operand_node_ids[0] in composable
                    ):
                        composable.add(node.node_id)
                    if (
                        node.result_type_label
                        in {"Function[Int,Bool]", "Function[Char,Bool]"}
                        and len(node.provenance.operand_node_ids) == 1
                        and node.provenance.operand_node_ids[0] in retained
                    ):
                        retained.add(node.node_id)
                        composable.add(node.node_id)
                    continue
                string_structural = (
                    node.result_type_label in {"AsciiClass", "Char"}
                    or node.provenance.production in {"contains", "filter"}
                    or (
                        signature.result == _STRING
                        and node.result_type_label == "String"
                    )
                )
                if string_structural:
                    composable.add(node.node_id)
                if node.environment_type_labels == root_environment:
                    contexts = tuple((evaluator, ()) for evaluator in evaluators)
                elif (
                    node.environment_type_labels == body_environment
                    and node.result_type_label in {"Int", "Bool"}
                ):
                    contexts = tuple(
                        (evaluator, (item,))
                        for evaluator, row in zip(evaluators, arguments, strict=True)
                        for parameter, value in zip(
                            signature.parameters,
                            row,
                            strict=True,
                        )
                        if parameter.type == _LIST_INT
                        for item in value
                    )
                elif (
                    node.environment_type_labels == string_body_environment
                    and node.result_type_label in {"AsciiClass", "Char", "Bool"}
                ):
                    contexts = tuple(
                        (evaluator, (character,))
                        for evaluator, row in zip(evaluators, arguments, strict=True)
                        for parameter, value in zip(
                            signature.parameters,
                            row,
                            strict=True,
                        )
                        if parameter.type == _STRING and _string_value_supported(value)
                        for character in value
                    )
                else:
                    continue
                if not contexts:
                    continue

                outputs: list[
                    output_cells.TypedOutputEncoding
                    | operator_successor._AsciiClassEncoding
                ] = []
                node_predictions = 0
                node_races = 0
                node_digest = sha256()
                node_digest.update(b"public-typed-semantic-candidate-races-v1\0")
                for evaluator, bound in contexts:
                    predictions_before = evaluator.predictions
                    races_before = len(evaluator.races)
                    outputs.append(evaluator.evaluate(node.node_id, bound=bound))
                    node_predictions += evaluator.predictions - predictions_before
                    node_races += len(evaluator.races) - races_before
                    digest = evaluator.race_digest(races_before)
                    node_digest.update(digest.encode("ascii"))
                    all_races_digest.update(digest.encode("ascii"))
                output_tuple = tuple(outputs)
                total_predictions += node_predictions
                total_races += node_races
                if any(not output.supported for output in output_tuple):
                    continue
                observation_bytes = tuple(
                    output.sha256.encode("ascii") for output in output_tuple
                )
                semantic_key = (
                    node.environment_type_labels,
                    node.result_type_label,
                    string_context_shape,
                    observation_bytes,
                )
                if semantic_key not in seen_semantics:
                    seen_semantics[semantic_key] = node.node_id
                    retained.add(node.node_id)
                    composable.add(node.node_id)
                if node.node_id not in tier_candidate_ids:
                    continue
                if any(
                    not isinstance(output, output_cells.TypedOutputEncoding)
                    for output in output_tuple
                ):
                    raise RuntimeError("compiler-internal type escaped a root candidate")
                vector = synthesis.TypedPhysicalSemanticVector(
                    result_type=signature.result,
                    observations=observation_bytes,
                )
                vectors[node.node_id] = vector
                evaluations[node.node_id] = PhysicalCandidateEvaluation(
                    candidate_id=node.node_id,
                    outputs=tuple(
                        output
                        for output in output_tuple
                        if isinstance(output, output_cells.TypedOutputEncoding)
                    ),
                    vector=vector,
                    operator_predictions=node_predictions,
                    operator_races=node_races,
                    operator_races_sha256=node_digest.hexdigest(),
                )
                root_semantics.add(observation_bytes)

            representatives = tuple(
                candidate.root_node_id
                for candidate in tier.candidates
                if candidate.root_node_id in retained
            )
            if not representatives:
                attempts.append(
                    SemanticSynthesisTierAttempt(
                        cost,
                        len(tier.candidates),
                        (),
                        None,
                    )
                )
                continue
            race = semantic_race.select_exact_semantic_vector(
                representatives,
                tuple(evaluations[key].outputs for key in representatives),
                desired,
            )
            attempts.append(
                SemanticSynthesisTierAttempt(
                    cost,
                    len(tier.candidates),
                    representatives,
                    race,
                )
            )
            if race.status == semantic_race.SELECTED:
                selected = candidates_by_id[race.winner_key]
                compiled = tp.compile(selected.program)
                after = self.live_checkpoint_sha256
                if after != before:
                    raise RuntimeError("semantic synthesizer changed during inference")
                return TypedSemanticSynthesisDecision(
                    status=STATUS_DONE,
                    stop_reason=None,
                    candidate_count=len(vectors),
                    observed_semantic_classes=len(root_semantics),
                    tier_attempts=tuple(attempts),
                    selected_candidate_id=selected.root_node_id,
                    program=selected.program,
                    source=compiled.source,
                    operator_predictions=total_predictions,
                    operator_races=total_races,
                    operator_races_sha256=all_races_digest.hexdigest(),
                    checkpoint_before_sha256=before,
                    checkpoint_after_sha256=after,
                )
            if race.status == semantic_race.STOP_MULTIPLE_EXACT:
                break

        after = self.live_checkpoint_sha256
        if after != before:
            raise RuntimeError("semantic synthesizer changed during inference")
        return TypedSemanticSynthesisDecision(
            status=STATUS_STOP,
            stop_reason="no unique all-case physical semantic vector in bounded tiers",
            candidate_count=len(vectors),
            observed_semantic_classes=len(root_semantics),
            tier_attempts=tuple(attempts),
            selected_candidate_id=None,
            program=None,
            source=None,
            operator_predictions=total_predictions,
            operator_races=total_races,
            operator_races_sha256=all_races_digest.hexdigest(),
            checkpoint_before_sha256=before,
            checkpoint_after_sha256=after,
        )

    def synthesize(
        self,
        signature: TypedSignature,
        case_arguments: Sequence[tuple[object, ...]],
        desired_outputs: Sequence[object],
    ) -> TypedSemanticSynthesisDecision:
        before = self.live_checkpoint_sha256
        if before != self.checkpoint_sha256:
            raise RuntimeError("semantic synthesizer checkpoint changed before inference")
        arguments = tuple(tuple(row) for row in case_arguments)
        desired_raw = tuple(desired_outputs)
        if not arguments or len(arguments) != len(desired_raw):
            raise ValueError("visible arguments and desired outputs must align")
        desired = tuple(_encode(value, signature.result) for value in desired_raw)
        return self._synthesize_retained(signature, arguments, desired, before)


def build_public_typed_semantic_synthesizer(
    *,
    literals: Sequence[int],
    max_ast_cost: int = synthesis.MAX_BRIDGE_AST_COST,
    _ascii_catalog_enabled: bool = True,
) -> tuple[
    FrozenTypedSemanticSynthesizer,
    operator_successor.TypedOperatorTrainingSummary,
]:
    literal_bank = tuple(sorted(set(literals)))
    if not literal_bank or not all(type(value) is int for value in literal_bank):
        raise ValueError("semantic synthesizer needs an integer literal bank")
    if type(_ascii_catalog_enabled) is not bool:
        raise ValueError("ASCII catalog control must be an exact bool")
    operator, summary = operator_successor.train_public_typed_operator_successor()
    placeholder = FrozenTypedSemanticSynthesizer(
        operator=operator,
        literal_bank=literal_bank,
        max_ast_cost=max_ast_cost,
        _ascii_catalog_enabled=_ascii_catalog_enabled,
        checkpoint_sha256="",
    )
    model = FrozenTypedSemanticSynthesizer(
        operator=operator,
        literal_bank=literal_bank,
        max_ast_cost=max_ast_cost,
        _ascii_catalog_enabled=_ascii_catalog_enabled,
        checkpoint_sha256=placeholder.live_checkpoint_sha256,
    )
    return model, summary


__all__ = [
    "FrozenTypedSemanticSynthesizer",
    "PhysicalCandidateEvaluation",
    "STATUS_DONE",
    "STATUS_STOP",
    "SemanticSynthesisTierAttempt",
    "TypedSemanticSynthesisDecision",
    "build_public_typed_semantic_synthesizer",
]
