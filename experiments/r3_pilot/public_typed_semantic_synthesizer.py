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
_LIST_INT = ProgramType.List(_INT)

_ABSOLUTE = tp.FactorAction((), "APPLY1", "absolute", "SELECTED", "NONE", None)
_BINARY_ACTIONS = {
    operator: tp.FactorAction(
        (), "APPLY2", operator, "SELECTED", "SCOPE0", None
    )
    for operator in ("add", "subtract")
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
    if value == _LIST_INT:
        return output_cells.OutputType.LIST_INT
    raise ValueError("physical synthesis supports only Int and List[Int]")


def _encode(value: object, value_type: ProgramType) -> output_cells.TypedOutputEncoding:
    if value_type == _LIST_INT and isinstance(value, tuple):
        value = list(value)
    encoded = output_cells.TypedOutputCellBank.encode(
        value,
        declared_type=_output_type(value_type),
    )
    if not encoded.supported:
        raise ValueError("visible synthesis value exceeds the physical output bank")
    return encoded


def _decode_int(value: output_cells.TypedOutputEncoding) -> int:
    if not value.supported or value.output_type is not output_cells.OutputType.INT:
        raise RuntimeError("physical operator did not produce a supported Int")
    cells = np.flatnonzero(value.int_cells)
    if len(cells) != 1 or int(cells[0]) == output_cells.INT_UNKNOWN_CELL:
        raise RuntimeError("physical Int output is not uniquely decodable")
    return output_cells.INT_MIN + int(cells[0])


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
    ) -> None:
        self.model = model
        self.dag = dag
        self.arguments = arguments
        self.nodes = {node.node_id: node for node in dag.nodes}
        self.memo: dict[
            tuple[str, tuple[int, ...]], output_cells.TypedOutputEncoding
        ] = {}
        self.predictions = 0
        self.races: list[operator_successor.OperatorRace] = []

    def _environment_value(
        self,
        node: synthesis.CandidateNode,
        index: int,
        bound: tuple[int, ...],
    ) -> tuple[object, ProgramType]:
        values = (*bound, *reversed(self.arguments))
        if not 0 <= index < len(values):
            raise RuntimeError("candidate variable lies outside its typed environment")
        label = node.environment_type_labels[index]
        value_type = _INT if label == "Int" else _LIST_INT
        return values[index], value_type

    def _predict(
        self,
        selected: output_cells.TypedOutputEncoding,
        action: tp.FactorAction,
        scopes: tuple[output_cells.TypedOutputEncoding, ...] = (),
    ) -> output_cells.TypedOutputEncoding:
        prediction = self.model.predict(selected, action, scopes=scopes)
        self.predictions += 1
        self.races.extend(prediction.races)
        if not prediction.supported:
            raise RuntimeError("physical typed operator returned UNKNOWN")
        return prediction.output

    def evaluate(
        self,
        node_id: str,
        *,
        bound: tuple[int, ...] = (),
    ) -> output_cells.TypedOutputEncoding:
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
            output = _encode(raw, value_type)
        elif production == "integer":
            if provenance.literal_value is None:
                raise RuntimeError("integer provenance lost its literal")
            output = _encode(provenance.literal_value, _INT)
        elif production == "absolute":
            operand = self.evaluate(provenance.operand_node_ids[0], bound=bound)
            output = self._predict(operand, _ABSOLUTE)
        elif production in _BINARY_ACTIONS:
            left = self.evaluate(provenance.operand_node_ids[0], bound=bound)
            right = self.evaluate(provenance.operand_node_ids[1], bound=bound)
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
            copied = self._predict(collection, _MAP_COPY)
            items = _decode_list(copied)
            mapped = [
                _decode_int(self.evaluate(body_id, bound=(item, *bound)))
                for item in items
            ]
            output = _encode(mapped, _LIST_INT)
        elif production == "bind":
            raise RuntimeError("a compiler binder cannot be evaluated outside map")
        else:
            raise RuntimeError(f"unsupported semantic production {production!r}")

        self.memo[key] = output
        return output

    def race_digest(self) -> str:
        digest = sha256()
        digest.update(b"public-typed-semantic-operator-races-v1\0")
        for race in self.races:
            digest.update(_canonical_json(asdict(race)))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenTypedSemanticSynthesizer:
    operator: operator_successor.FrozenTypedOperatorSuccessor
    literal_bank: tuple[int, ...]
    max_ast_cost: int
    checkpoint_sha256: str

    @property
    def live_checkpoint_sha256(self) -> str:
        digest = sha256()
        digest.update(b"public-typed-semantic-synthesizer-v1\0")
        digest.update(self.operator.live_checkpoint_sha256.encode("ascii"))
        digest.update(_canonical_json(self.literal_bank))
        digest.update(np.asarray((self.max_ast_cost,), dtype="<i8").tobytes())
        return digest.hexdigest()

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
        dag = synthesis.enumerate_typed_candidate_dag(
            signature,
            literals=self.literal_bank,
            max_ast_cost=self.max_ast_cost,
        )
        candidates = tuple(
            candidate for tier in dag.tiers for candidate in tier.candidates
        )
        candidate_by_id = {
            candidate.root_node_id: candidate for candidate in candidates
        }
        vectors: dict[str, synthesis.TypedPhysicalSemanticVector] = {}
        evaluations: dict[str, PhysicalCandidateEvaluation] = {}
        attempts: list[SemanticSynthesisTierAttempt] = []
        total_predictions = 0
        total_races = 0
        all_races_digest = sha256()
        all_races_digest.update(b"public-typed-semantic-all-races-v1\0")
        observed_classes = 0

        for tier in dag.tiers:
            for candidate in tier.candidates:
                outputs: list[output_cells.TypedOutputEncoding] = []
                candidate_predictions = 0
                candidate_races = 0
                candidate_digest = sha256()
                candidate_digest.update(b"public-typed-semantic-candidate-races-v1\0")
                for row in arguments:
                    evaluator = _CaseEvaluator(self.operator, dag, row)
                    outputs.append(evaluator.evaluate(candidate.root_node_id))
                    candidate_predictions += evaluator.predictions
                    candidate_races += len(evaluator.races)
                    digest = evaluator.race_digest()
                    candidate_digest.update(digest.encode("ascii"))
                    all_races_digest.update(digest.encode("ascii"))
                output_tuple = tuple(outputs)
                vector = synthesis.TypedPhysicalSemanticVector(
                    result_type=signature.result,
                    observations=tuple(
                        output.sha256.encode("ascii") for output in output_tuple
                    ),
                )
                vectors[candidate.root_node_id] = vector
                evaluations[candidate.root_node_id] = PhysicalCandidateEvaluation(
                    candidate_id=candidate.root_node_id,
                    outputs=output_tuple,
                    vector=vector,
                    operator_predictions=candidate_predictions,
                    operator_races=candidate_races,
                    operator_races_sha256=candidate_digest.hexdigest(),
                )
                total_predictions += candidate_predictions
                total_races += candidate_races

            if not vectors:
                attempts.append(
                    SemanticSynthesisTierAttempt(tier.ast_cost, 0, (), None)
                )
                continue
            dedup = synthesis.deduplicate_physical_semantics(dag, vectors)
            observed_classes = len(dedup.classes)
            representative_set = set(dedup.representative_candidate_ids)
            representatives = tuple(
                candidate.root_node_id
                for candidate in tier.candidates
                if candidate.root_node_id in representative_set
            )
            if not representatives:
                attempts.append(
                    SemanticSynthesisTierAttempt(
                        tier.ast_cost,
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
                    tier.ast_cost,
                    len(tier.candidates),
                    representatives,
                    race,
                )
            )
            if race.status == semantic_race.SELECTED:
                selected = candidate_by_id[race.winner_key]
                compiled = tp.compile(selected.program)
                after = self.live_checkpoint_sha256
                if after != before:
                    raise RuntimeError("semantic synthesizer changed during inference")
                return TypedSemanticSynthesisDecision(
                    status=STATUS_DONE,
                    stop_reason=None,
                    candidate_count=len(vectors),
                    observed_semantic_classes=observed_classes,
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
            observed_semantic_classes=observed_classes,
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


def build_public_typed_semantic_synthesizer(
    *,
    literals: Sequence[int],
    max_ast_cost: int = synthesis.MAX_BRIDGE_AST_COST,
) -> tuple[
    FrozenTypedSemanticSynthesizer,
    operator_successor.TypedOperatorTrainingSummary,
]:
    literal_bank = tuple(sorted(set(literals)))
    if not literal_bank or not all(type(value) is int for value in literal_bank):
        raise ValueError("semantic synthesizer needs an integer literal bank")
    operator, summary = operator_successor.train_public_typed_operator_successor()
    placeholder = FrozenTypedSemanticSynthesizer(
        operator=operator,
        literal_bank=literal_bank,
        max_ast_cost=max_ast_cost,
        checkpoint_sha256="",
    )
    model = FrozenTypedSemanticSynthesizer(
        operator=operator,
        literal_bank=literal_bank,
        max_ast_cost=max_ast_cost,
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
