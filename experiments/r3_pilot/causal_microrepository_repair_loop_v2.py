"""Learn and apply one cross-type String microrepository repair.

The TRAIN adapter solves three witnesses from one desired-free cost-7 String
frontier.  Two simulated relation cells retain only role-normalized physical
sequence consequences: identity to decimal-character selection, and selected
subsequence to the learned one-hole outer context.  The generic d9e95d5
UNIQUE/NONE/MULTIPLE/MATCH/MISMATCH action arrays are retained by authenticated
object reference; its ABS-specific relation arrays are neither copied nor used.

Fresh repository cases are observed only after that checkpoint is frozen.
Rollout physically races one exterior typed-site/relation/donor affordance at a
time, executes only the selected AST, observes membership/order/cardinality and
input-output lineage cells, and commits through the reused action cells.  Fresh
desired outputs are read once after all terminal vectors are frozen.  Only one
unique semantic winner may render source, patch one file, and run one test.

Typed DSL enumeration, AST lowering, JSON decoding, filesystem/process I/O,
and rollback are explicit environment scaffolding.  This is evidence for one
self-posed String -> String microrepository family only.
"""

from __future__ import annotations

import ast
import builtins
import json
import subprocess
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

import numpy as np
from experiments.r3_pilot import causal_recurrent_fragment_relation_policy_v2 as recurrent
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v2 as breadth
from humanbrain import typed_program
from humanbrain.typed_program_types import ProgramType, TypedSignature

SCHEMA = "CAUSAL_MICROREPOSITORY_REPAIR_LOOP_V2"
READY = f"{SCHEMA}_READY"
PREPARED = f"{SCHEMA}_PREPARED"
COMMITTED = f"{SCHEMA}_COMMITTED"
STOP_TRAINING = f"STOP_{SCHEMA}_TRAINING"
STOP_CONTEXT = f"STOP_{SCHEMA}_CONTEXT"
STOP_LINEAGE = f"STOP_{SCHEMA}_LINEAGE"
STOP_POLICY = f"STOP_{SCHEMA}_POLICY"
STOP_EFFECT = f"STOP_{SCHEMA}_EFFECT"
STOP_ACTUATION = f"STOP_{SCHEMA}_ACTUATION"
STOP_NO_AFFORDANCE = f"STOP_{SCHEMA}_NO_AFFORDANCE"
STOP_AMBIGUOUS_AFFORDANCE = f"STOP_{SCHEMA}_AMBIGUOUS_AFFORDANCE"
STOP_POLICY_INVARIANT = f"STOP_{SCHEMA}_POLICY_INVARIANT"
STOP_SOURCE_DRIFT = f"STOP_{SCHEMA}_SOURCE_DRIFT"
STOP_ADAPTER_INPUT = f"STOP_{SCHEMA}_ADAPTER_INPUT"
STOP_PATCH_WRITE = f"STOP_{SCHEMA}_PATCH_WRITE"
STOP_TEST_FAILURE = f"STOP_{SCHEMA}_TEST_FAILURE"
STOP_CONSEQUENCE_POLICY = f"STOP_{SCHEMA}_CONSEQUENCE_POLICY"

RELATION_COUNT = 2
RELATION_SIDES = 2
PRECONDITION = 0
POSTCONDITION = 1
DESCRIPTOR_ROUTES = recurrent.DESCRIPTOR_ROUTES
DESCRIPTOR_RADIX = recurrent.DESCRIPTOR_RADIX
STRING_CAPACITY = 26
MAX_SELECTED_ACTUATIONS = 2
UNSELECTED_SPLICE_EXECUTIONS = 0
UNSELECTED_SOURCE_MATERIALIZATIONS = 0
UNSELECTED_PATCH_WRITES = 0
UNSELECTED_REPOSITORY_TESTS = 0

MEMBERSHIP_FULL = 1
MEMBERSHIP_SUBSET = 2
MEMBERSHIP_OTHER = 3
CARDINALITY_SAME = 1
CARDINALITY_SHRINK = 2
CARDINALITY_GROW = 3
CARDINALITY_MIXED = 4

CLAIM_BOUNDARY = (
    "one self-posed fresh String -> String identity-to-decimal-filter-to-reverse "
    "microrepository repair; typed DSL/site enumeration, AST lowering, lineage "
    "observation, manifests, filesystem/process I/O, and rollback remain host "
    "scaffolding; no broad coding or repository-repair claim"
)

_TRAIN_NAMESPACE = b"causal-string-relation-training-v2"
_RECEIPT_NAMESPACE = b"causal-string-relation-receipt-v2"
_CHECKPOINT_NAMESPACE = b"causal-string-relation-checkpoint-v2"
_FRONTIER_NAMESPACE = b"causal-string-transfer-frontier-v2"
_ATTESTATION_NAMESPACE = b"causal-string-root-attestation-v2"
_PREPARED_NAMESPACE = b"causal-string-repair-prepared-v2"
_RESULT_NAMESPACE = b"causal-string-repair-result-v2"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(value.tobytes(order="C")).hexdigest()


def _validate_string_signature(signature: TypedSignature) -> None:
    string = ProgramType.String()
    if (
        not isinstance(signature, TypedSignature)
        or signature.result != string
        or len(signature.parameters) != 1
        or signature.parameters[0].type != string
    ):
        raise ValueError("V2 requires one exact String -> String signature")


def _signature_record(signature: TypedSignature) -> object:
    return {
        "entry_point": signature.entry_point,
        "parameters": tuple(
            (item.name, repr(item.type)) for item in signature.parameters
        ),
        "result": repr(signature.result),
    }


def _encode_index(value: int) -> tuple[output_cells.TypedOutputEncoding, ...]:
    return (
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.INT,
        ),
    )


def _encode_strings(values: Sequence[str]) -> tuple[output_cells.TypedOutputEncoding, ...]:
    return tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.STRING,
        )
        for value in values
    )


def _encoding_vector_sha256(
    values: Sequence[output_cells.TypedOutputEncoding],
) -> str:
    return sha256(
        b"".join(item.factor_winners_le + item.physical_digest for item in values)
    ).hexdigest()


def _is_subsequence(needle: str, haystack: str) -> bool:
    cursor = iter(haystack)
    return all(any(candidate == value for candidate in cursor) for value in needle)


class PhysicalSequenceEffectV2(NamedTuple):
    membership: int
    increasing_possible: bool
    decreasing_possible: bool
    cardinality: int
    active_input_cells: int
    active_output_cells: int
    active_lineage_cells: int
    case_count: int
    trace_sha256: str


def _physical_sequence_effect(
    inputs: Sequence[str],
    outputs: Sequence[str],
) -> PhysicalSequenceEffectV2:
    frozen_inputs = tuple(inputs)
    frozen_outputs = tuple(outputs)
    if not frozen_inputs or len(frozen_inputs) != len(frozen_outputs):
        raise ValueError("physical sequence traces require aligned non-empty cases")
    if any(
        type(value) is not str or len(value) > STRING_CAPACITY
        for value in (*frozen_inputs, *frozen_outputs)
    ):
        raise ValueError("physical String traces require <=26 ASCII characters")
    if any(
        any(ord(character) > 127 for character in value)
        for value in (*frozen_inputs, *frozen_outputs)
    ):
        raise ValueError("physical String traces require ASCII values")

    full = []
    subset = []
    increasing = []
    decreasing = []
    lengths: list[int] = []
    physical = bytearray()
    active_input = 0
    active_output = 0
    active_lineage = 0
    for source, result in zip(frozen_inputs, frozen_outputs, strict=True):
        source_counts = Counter(source)
        result_counts = Counter(result)
        full.append(len(source) == len(result) and source_counts == result_counts)
        subset.append(
            all(result_counts[key] <= source_counts[key] for key in result_counts)
        )
        increasing.append(_is_subsequence(result, source))
        decreasing.append(_is_subsequence(result, source[::-1]))
        lengths.append((len(result) > len(source)) - (len(result) < len(source)))
        input_cells = bytes([1] * len(source) + [0] * (STRING_CAPACITY - len(source)))
        output_cells_active = bytes(
            [1] * len(result) + [0] * (STRING_CAPACITY - len(result))
        )
        coincidence = bytes(
            int(i < len(source) and j < len(result) and source[i] == result[j])
            for i in range(STRING_CAPACITY)
            for j in range(STRING_CAPACITY)
        )
        physical.extend(input_cells)
        physical.extend(output_cells_active)
        physical.extend(coincidence)
        active_input += sum(input_cells)
        active_output += sum(output_cells_active)
        active_lineage += sum(coincidence)

    membership = (
        MEMBERSHIP_FULL
        if all(full)
        else MEMBERSHIP_SUBSET
        if all(subset)
        else MEMBERSHIP_OTHER
    )
    cardinality = (
        CARDINALITY_SAME
        if all(item == 0 for item in lengths)
        else CARDINALITY_SHRINK
        if all(item < 0 for item in lengths)
        else CARDINALITY_GROW
        if all(item > 0 for item in lengths)
        else CARDINALITY_MIXED
    )
    return PhysicalSequenceEffectV2(
        membership=membership,
        increasing_possible=all(increasing),
        decreasing_possible=all(decreasing),
        cardinality=cardinality,
        active_input_cells=active_input,
        active_output_cells=active_output,
        active_lineage_cells=active_lineage,
        case_count=len(frozen_inputs),
        trace_sha256=sha256(bytes(physical)).hexdigest(),
    )


def _descriptor(
    effect: PhysicalSequenceEffectV2,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    targets = [0] * DESCRIPTOR_ROUTES
    mask = [False] * DESCRIPTOR_ROUTES
    for route in range(0, 8):
        targets[route] = effect.membership
        mask[route] = True
    for route in range(8, 16):
        targets[route] = effect.cardinality
        mask[route] = True
    if effect.increasing_possible:
        for route in range(16, 20):
            targets[route] = 1
            mask[route] = True
    if effect.decreasing_possible:
        for route in range(20, 24):
            targets[route] = 1
            mask[route] = True
    # These cells record that membership/order arose from an actual non-empty
    # input/output coincidence field, not an operator or path label.
    if effect.active_lineage_cells > 0:
        for route in range(24, 28):
            targets[route] = 1
            mask[route] = True
    # Exterior typed-site consequence: absence of an enclosing expression.
    for route in range(28, 32):
        targets[route] = 1
        mask[route] = True
    return tuple(targets), tuple(mask)


def _find_one_hole_context(
    base_tokens: tuple[str, ...],
    result_tokens: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    starts = tuple(
        start
        for start in range(len(result_tokens) - len(base_tokens) + 1)
        if result_tokens[start : start + len(base_tokens)] == base_tokens
    )
    if len(starts) != 1:
        raise ValueError("TRAIN result has no unique one-hole structural explanation")
    start = starts[0]
    prefix = result_tokens[:start]
    suffix = result_tokens[start + len(base_tokens) :]
    if not prefix and not suffix:
        raise ValueError("TRAIN relation did not learn a non-empty context")
    return prefix, suffix


def _evaluate_program_without_source(
    program: typed_program.Program,
    cases: tuple[tuple[str], ...],
) -> tuple[str, ...]:
    expression = typed_program._validate_program(program.tokens, program.signature)
    environment = tuple(
        parameter.name for parameter in reversed(program.signature.parameters)
    )
    module = ast.Module(
        body=[
            ast.FunctionDef(
                name=program.signature.entry_point,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[
                        ast.arg(arg=parameter.name)
                        for parameter in program.signature.parameters
                    ],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=[
                    ast.Return(
                        value=typed_program._lower_expression(
                            expression,
                            environment,
                            typed_program._LoweringContext(),
                        )
                    )
                ],
                decorator_list=[],
                returns=None,
                type_comment=None,
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(builtins.compile(module, "<selected-string-actuation>", "exec"), namespace)
    function = namespace[program.signature.entry_point]
    outputs = tuple(function(*case) for case in cases)  # type: ignore[operator]
    if any(type(value) is not str for value in outputs):
        raise RuntimeError("selected String actuation produced a non-String value")
    return outputs


class SolvedStringTrainingV2(NamedTuple):
    signature: TypedSignature
    cases: tuple[tuple[str], ...]
    frontier: breadth.PackedGenericTypedTrainBreadthV2
    build_work: object
    identity_program: typed_program.Program
    digit_selection: breadth.PackedGenericTypedSelection
    vowel_selection: breadth.PackedGenericTypedSelection
    reverse_vowel_selection: breadth.PackedGenericTypedSelection
    training_sha256: str


def solve_string_relation_training_v2(
    base_policy: recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2,
    signature: TypedSignature,
    cases: Iterable[tuple[str]],
) -> SolvedStringTrainingV2:
    """Build one desired-free frontier, then solve the three TRAIN witnesses."""
    base_policy.authenticate()
    _validate_string_signature(signature)
    frozen_cases = tuple(cases)
    if not frozen_cases or any(
        type(row) is not tuple
        or len(row) != 1
        or type(row[0]) is not str
        or len(row[0]) > STRING_CAPACITY
        for row in frozen_cases
    ):
        raise ValueError("TRAIN cases must be non-empty singleton String rows")
    frontier = breadth.PackedGenericTypedTrainBreadthV2.build(
        signature,
        frozen_cases,
        (),
        7,
    )
    build_work = frontier.work
    if (
        frontier.status != breadth.FRONTIER_READY
        or build_work.concrete_proof_witnesses != 0  # type: ignore[attr-defined]
        or build_work.materialized_witnesses != 0  # type: ignore[attr-defined]
    ):
        raise RuntimeError("String TRAIN frontier was not desired-free")
    values = tuple(row[0] for row in frozen_cases)
    digits = tuple("".join(char for char in value if char in "0123456789") for value in values)
    vowels = tuple("".join(char for char in value if char in "AEIOUaeiou") for value in values)
    reverse_vowels = tuple(value[::-1] for value in vowels)
    digit_selection = frontier.select(digits)
    vowel_selection = frontier.select(vowels)
    reverse_vowel_selection = frontier.select(reverse_vowels)
    selections = (digit_selection, vowel_selection, reverse_vowel_selection)
    if any(
        item.status != breadth.SELECTED
        or item.exact_proof_count != 1
        or len(item.witnesses) != 1
        for item in selections
    ):
        raise RuntimeError("String TRAIN lessons did not solve uniquely")
    identity = typed_program.Program(("variable-0",), signature)
    training_sha256 = _digest(
        _TRAIN_NAMESPACE,
        {
            "base_policy": base_policy.checkpoint_sha256,
            "signature": _signature_record(signature),
            "cases": sha256(_canonical_json(frozen_cases)).hexdigest(),
            "frontier": frontier.sha256,
            "selections": tuple(item.sha256 for item in selections),
            "identity": identity.sha256,
        },
    )
    return SolvedStringTrainingV2(
        signature=signature,
        cases=frozen_cases,
        frontier=frontier,
        build_work=build_work,
        identity_program=identity,
        digit_selection=digit_selection,
        vowel_selection=vowel_selection,
        reverse_vowel_selection=reverse_vowel_selection,
        training_sha256=training_sha256,
    )


class StringRelationReceiptV2(NamedTuple):
    cell_index: int
    role: str
    actuation_kind: str
    training_frontier_sha256: str
    base_selection_sha256: str
    result_selection_sha256: str
    base_program_sha256: str
    result_program_sha256: str
    precondition_trace_sha256: str
    postcondition_trace_sha256: str
    precondition_targets: tuple[int, ...]
    precondition_mask: tuple[bool, ...]
    postcondition_targets: tuple[int, ...]
    postcondition_mask: tuple[bool, ...]
    donor_tokens: tuple[str, ...]
    context_prefix_tokens: tuple[str, ...]
    context_suffix_tokens: tuple[str, ...]
    exterior_site_cell: int
    receipt_sha256: str


class StringRelationTrainingWorkV2(NamedTuple):
    frontier_relation_units: int
    frontier_predictions: int
    frontier_races: int
    train_desired_reads: int
    train_selected_witnesses: int
    selected_trace_program_executions: int
    selected_trace_case_evaluations: int
    unselected_trace_program_executions: int
    learned_relation_cells: int
    lineage_input_cells: int
    lineage_output_cells: int
    lineage_coincidence_cells: int
    relation_excitatory_ticks: int
    relation_inhibitory_ticks: int
    new_persistent_state_bytes: int
    referenced_policy_state_bytes: int
    transferred_abs_relation_bytes: int


class FrozenStringRelationPolicyStateV2(NamedTuple):
    status: str
    base_policy_checkpoint_sha256: str
    training_sha256: str
    training_frontier_sha256: str
    training_selection_sha256s: tuple[str, ...]
    training_root_program_sha256: str
    relation_receipt_sha256s: tuple[str, ...]
    relation_excitatory_sha256: str
    relation_inhibitory_sha256: str
    relation_masks_sha256: str
    relation_counts_sha256: str
    policy_excitatory_sha256: str
    policy_inhibitory_sha256: str
    policy_arrays_reused_by_reference: bool
    fresh_artifact_hashes: tuple[str, ...]
    work: StringRelationTrainingWorkV2
    checkpoint_sha256: str


def _receipt_sha256(receipt: StringRelationReceiptV2) -> str:
    payload = receipt._asdict()
    payload.pop("receipt_sha256")
    return _digest(_RECEIPT_NAMESPACE, payload)


def _make_receipt(
    *,
    cell_index: int,
    role: str,
    actuation_kind: str,
    training: SolvedStringTrainingV2,
    base_selection_sha256: str,
    result_selection_sha256: str,
    base_program: typed_program.Program,
    result_program: typed_program.Program,
    precondition: PhysicalSequenceEffectV2,
    postcondition: PhysicalSequenceEffectV2,
    donor_tokens: tuple[str, ...],
    context_prefix_tokens: tuple[str, ...] = (),
    context_suffix_tokens: tuple[str, ...] = (),
) -> StringRelationReceiptV2:
    pre_targets, pre_mask = _descriptor(precondition)
    post_targets, post_mask = _descriptor(postcondition)
    partial = StringRelationReceiptV2(
        cell_index=cell_index,
        role=role,
        actuation_kind=actuation_kind,
        training_frontier_sha256=training.frontier.sha256,
        base_selection_sha256=base_selection_sha256,
        result_selection_sha256=result_selection_sha256,
        base_program_sha256=base_program.sha256,
        result_program_sha256=result_program.sha256,
        precondition_trace_sha256=precondition.trace_sha256,
        postcondition_trace_sha256=postcondition.trace_sha256,
        precondition_targets=pre_targets,
        precondition_mask=pre_mask,
        postcondition_targets=post_targets,
        postcondition_mask=post_mask,
        donor_tokens=donor_tokens,
        context_prefix_tokens=context_prefix_tokens,
        context_suffix_tokens=context_suffix_tokens,
        exterior_site_cell=1,
        receipt_sha256="",
    )
    return partial._replace(receipt_sha256=_receipt_sha256(partial))


class FrozenStringRelationPolicyV2:
    """Two new sequence relation cells plus referenced d9 action cells."""

    __slots__ = (
        "_base_policy",
        "_checkpoint_payload",
        "_policy_excitatory",
        "_policy_inhibitory",
        "_receipts",
        "_relation_counts",
        "_relation_excitatory",
        "_relation_inhibitory",
        "_relation_masks",
        "_signature",
        "_state",
    )

    def __init__(self) -> None:
        raise TypeError("use fit_string_relation_policy_v2")

    @property
    def state(self) -> FrozenStringRelationPolicyStateV2:
        self.authenticate()
        return self._state

    @property
    def relation_receipts(self) -> tuple[StringRelationReceiptV2, ...]:
        self.authenticate()
        return self._receipts

    @property
    def checkpoint_sha256(self) -> str:
        return self._state.checkpoint_sha256

    @property
    def checkpoint_payload_bytes(self) -> bytes:
        self.authenticate()
        return self._checkpoint_payload

    def authenticate(self) -> None:
        self._base_policy.authenticate()
        if (
            self._policy_excitatory is not self._base_policy._policy_excitatory
            or self._policy_inhibitory is not self._base_policy._policy_inhibitory
            or self._relation_excitatory.shape
            != (RELATION_COUNT, RELATION_SIDES, DESCRIPTOR_ROUTES, DESCRIPTOR_RADIX)
            or self._relation_inhibitory.shape != self._relation_excitatory.shape
            or self._relation_masks.shape
            != (RELATION_COUNT, RELATION_SIDES, DESCRIPTOR_ROUTES)
            or self._relation_counts.shape != (RELATION_COUNT,)
            or any(
                array.flags.writeable
                for array in (
                    self._relation_excitatory,
                    self._relation_inhibitory,
                    self._relation_masks,
                    self._relation_counts,
                    self._policy_excitatory,
                    self._policy_inhibitory,
                )
            )
        ):
            raise RuntimeError("String relation cell geometry or policy reference changed")
        if (
            len(self._receipts) != RELATION_COUNT
            or len({item.receipt_sha256 for item in self._receipts}) != RELATION_COUNT
            or any(_receipt_sha256(item) != item.receipt_sha256 for item in self._receipts)
            or any(item.exterior_site_cell != 1 for item in self._receipts)
            or set(item.role for item in self._receipts)
            != {"selected-character-subsequence", "outer-one-hole-context"}
        ):
            raise RuntimeError("String relation receipts changed")
        state = self._state
        if (
            state.status != READY
            or state.base_policy_checkpoint_sha256
            != self._base_policy.checkpoint_sha256
            or state.relation_excitatory_sha256
            != _array_sha256(self._relation_excitatory)
            or state.relation_inhibitory_sha256
            != _array_sha256(self._relation_inhibitory)
            or state.relation_masks_sha256 != _array_sha256(self._relation_masks)
            or state.relation_counts_sha256 != _array_sha256(self._relation_counts)
            or state.policy_excitatory_sha256
            != _array_sha256(self._policy_excitatory)
            or state.policy_inhibitory_sha256
            != _array_sha256(self._policy_inhibitory)
            or not state.policy_arrays_reused_by_reference
            or state.fresh_artifact_hashes
            or _digest(_CHECKPOINT_NAMESPACE, json.loads(self._checkpoint_payload))
            != state.checkpoint_sha256
        ):
            raise RuntimeError("String relation checkpoint authentication failed")

    def build(
        self,
        signature: TypedSignature,
        cases: Iterable[tuple[str]],
        *,
        context_cell_gate: bool = True,
        lineage_cell_gate: bool = True,
        policy_cell_gate: bool = True,
        effect_cell_gate: bool = True,
        actuation_gate: bool = True,
    ) -> StringRelationFrontierV2:
        return StringRelationFrontierV2.build(
            self,
            signature,
            cases,
            context_cell_gate=context_cell_gate,
            lineage_cell_gate=lineage_cell_gate,
            policy_cell_gate=policy_cell_gate,
            effect_cell_gate=effect_cell_gate,
            actuation_gate=actuation_gate,
        )


def _learn_relation_side(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    masks: np.ndarray,
    cell: int,
    side: int,
    targets: Sequence[int],
    mask: Sequence[bool],
) -> tuple[int, int]:
    e_ticks = 0
    h_ticks = 0
    for route, enabled in enumerate(mask):
        if not enabled:
            continue
        masks[cell, side, route] = True
        winner = targets[route]
        for target in range(DESCRIPTOR_RADIX):
            state = excitatory if target == winner else inhibitory
            state[cell, side, route, target] = 1
            if target == winner:
                e_ticks += 1
            else:
                h_ticks += 1
    return e_ticks, h_ticks


def fit_string_relation_policy_v2(
    base_policy: recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2,
    training: SolvedStringTrainingV2,
    *,
    lesson_order: tuple[int, int] = (0, 1),
) -> FrozenStringRelationPolicyV2:
    """Learn two String sequence relations without copying d9 ABS relations."""
    base_policy.authenticate()
    if type(training) is not SolvedStringTrainingV2:
        raise TypeError("exact solved String training object required")
    if sorted(lesson_order) != [0, 1]:
        raise ValueError("lesson_order must contain each relation exactly once")
    selections = (
        training.digit_selection,
        training.vowel_selection,
        training.reverse_vowel_selection,
    )
    if any(item.status != breadth.SELECTED or len(item.witnesses) != 1 for item in selections):
        raise RuntimeError("solved String training lost a witness")
    digit_program = training.digit_selection.witnesses[0].program
    vowel_program = training.vowel_selection.witnesses[0].program
    reverse_vowel_program = training.reverse_vowel_selection.witnesses[0].program
    prefix, suffix = _find_one_hole_context(
        vowel_program.tokens,
        reverse_vowel_program.tokens,
    )
    inputs = tuple(row[0] for row in training.cases)
    # Consequences come from the authenticated selected programs themselves.
    # The fit path never reconstructs a digit, vowel, or reversal behavior from
    # a host predicate.  AST lowering/execution is explicit environment
    # scaffolding, and only these three selected witnesses plus the attested
    # identity root execute.
    identity_outputs = _evaluate_program_without_source(
        training.identity_program,
        training.cases,
    )
    digit_outputs = _evaluate_program_without_source(digit_program, training.cases)
    vowel_outputs = _evaluate_program_without_source(vowel_program, training.cases)
    reverse_outputs = _evaluate_program_without_source(
        reverse_vowel_program,
        training.cases,
    )
    identity_effect = _physical_sequence_effect(inputs, identity_outputs)
    digit_effect = _physical_sequence_effect(inputs, digit_outputs)
    vowel_effect = _physical_sequence_effect(inputs, vowel_outputs)
    reverse_effect = _physical_sequence_effect(inputs, reverse_outputs)
    if (
        (identity_effect.membership, identity_effect.cardinality)
        != (MEMBERSHIP_FULL, CARDINALITY_SAME)
        or not identity_effect.increasing_possible
        or (digit_effect.membership, digit_effect.cardinality)
        != (MEMBERSHIP_SUBSET, CARDINALITY_SHRINK)
        or not digit_effect.increasing_possible
        or (vowel_effect.membership, vowel_effect.cardinality)
        != (MEMBERSHIP_SUBSET, CARDINALITY_SHRINK)
        or not vowel_effect.increasing_possible
        or (reverse_effect.membership, reverse_effect.cardinality)
        != (MEMBERSHIP_SUBSET, CARDINALITY_SHRINK)
        or not reverse_effect.decreasing_possible
    ):
        raise RuntimeError("TRAIN cases do not causally identify both sequence relations")

    definitions = (
        {
            "role": "selected-character-subsequence",
            "kind": "replace-exterior",
            "base_selection": "authenticated-identity-root",
            "result_selection": training.digit_selection.sha256,
            "base": training.identity_program,
            "result": digit_program,
            "pre": identity_effect,
            "post": digit_effect,
            "donor": digit_program.tokens,
            "prefix": (),
            "suffix": (),
        },
        {
            "role": "outer-one-hole-context",
            "kind": "one-hole-context",
            "base_selection": training.vowel_selection.sha256,
            "result_selection": training.reverse_vowel_selection.sha256,
            "base": vowel_program,
            "result": reverse_vowel_program,
            "pre": vowel_effect,
            "post": reverse_effect,
            "donor": (*prefix, "<typed-hole>", *suffix),
            "prefix": prefix,
            "suffix": suffix,
        },
    )
    excitatory = np.zeros(
        (RELATION_COUNT, RELATION_SIDES, DESCRIPTOR_ROUTES, DESCRIPTOR_RADIX),
        dtype=recurrent.TICK_DTYPE,
    )
    inhibitory = np.zeros_like(excitatory)
    masks = np.zeros(
        (RELATION_COUNT, RELATION_SIDES, DESCRIPTOR_ROUTES),
        dtype=np.bool_,
    )
    counts = np.zeros((RELATION_COUNT,), dtype=recurrent.COUNT_DTYPE)
    receipts: list[StringRelationReceiptV2] = []
    e_ticks = 0
    h_ticks = 0
    for cell_index, definition_index in enumerate(lesson_order):
        definition = definitions[definition_index]
        receipt = _make_receipt(
            cell_index=cell_index,
            role=str(definition["role"]),
            actuation_kind=str(definition["kind"]),
            training=training,
            base_selection_sha256=str(definition["base_selection"]),
            result_selection_sha256=str(definition["result_selection"]),
            base_program=definition["base"],  # type: ignore[arg-type]
            result_program=definition["result"],  # type: ignore[arg-type]
            precondition=definition["pre"],  # type: ignore[arg-type]
            postcondition=definition["post"],  # type: ignore[arg-type]
            donor_tokens=definition["donor"],  # type: ignore[arg-type]
            context_prefix_tokens=definition["prefix"],  # type: ignore[arg-type]
            context_suffix_tokens=definition["suffix"],  # type: ignore[arg-type]
        )
        for side, targets, mask in (
            (PRECONDITION, receipt.precondition_targets, receipt.precondition_mask),
            (POSTCONDITION, receipt.postcondition_targets, receipt.postcondition_mask),
        ):
            learned_e, learned_h = _learn_relation_side(
                excitatory,
                inhibitory,
                masks,
                cell_index,
                side,
                targets,
                mask,
            )
            e_ticks += learned_e
            h_ticks += learned_h
        counts[cell_index] = 1
        receipts.append(receipt)
    for array in (excitatory, inhibitory, masks, counts):
        array.setflags(write=False)
    policy_e = base_policy._policy_excitatory
    policy_h = base_policy._policy_inhibitory
    training_work = training.build_work
    work = StringRelationTrainingWorkV2(
        frontier_relation_units=training_work.relation.completed_units,  # type: ignore[attr-defined]
        frontier_predictions=training_work.physical_predictions,  # type: ignore[attr-defined]
        frontier_races=training_work.physical_races,  # type: ignore[attr-defined]
        train_desired_reads=3,
        train_selected_witnesses=3,
        selected_trace_program_executions=4,
        selected_trace_case_evaluations=4 * len(training.cases),
        unselected_trace_program_executions=0,
        learned_relation_cells=RELATION_COUNT,
        lineage_input_cells=STRING_CAPACITY,
        lineage_output_cells=STRING_CAPACITY,
        lineage_coincidence_cells=STRING_CAPACITY * STRING_CAPACITY,
        relation_excitatory_ticks=e_ticks,
        relation_inhibitory_ticks=h_ticks,
        new_persistent_state_bytes=sum(
            array.nbytes for array in (excitatory, inhibitory, masks, counts)
        ),
        referenced_policy_state_bytes=policy_e.nbytes + policy_h.nbytes,
        transferred_abs_relation_bytes=0,
    )
    frozen_receipts = tuple(receipts)
    payload = {
        "schema": SCHEMA,
        "base_policy": base_policy.checkpoint_sha256,
        "training": training.training_sha256,
        "frontier": training.frontier.sha256,
        "selections": tuple(item.sha256 for item in selections),
        "signature": _signature_record(training.signature),
        "root_program": training.identity_program.sha256,
        "receipts": tuple(item.receipt_sha256 for item in frozen_receipts),
        "relation_excitatory": _array_sha256(excitatory),
        "relation_inhibitory": _array_sha256(inhibitory),
        "relation_masks": _array_sha256(masks),
        "relation_counts": _array_sha256(counts),
        "policy_excitatory": _array_sha256(policy_e),
        "policy_inhibitory": _array_sha256(policy_h),
        "fresh_artifact_hashes": (),
        "work": work._asdict(),
    }
    checkpoint_payload = _canonical_json(payload)
    checkpoint = _digest(_CHECKPOINT_NAMESPACE, payload)
    state = FrozenStringRelationPolicyStateV2(
        status=READY,
        base_policy_checkpoint_sha256=base_policy.checkpoint_sha256,
        training_sha256=training.training_sha256,
        training_frontier_sha256=training.frontier.sha256,
        training_selection_sha256s=tuple(item.sha256 for item in selections),
        training_root_program_sha256=training.identity_program.sha256,
        relation_receipt_sha256s=tuple(item.receipt_sha256 for item in frozen_receipts),
        relation_excitatory_sha256=_array_sha256(excitatory),
        relation_inhibitory_sha256=_array_sha256(inhibitory),
        relation_masks_sha256=_array_sha256(masks),
        relation_counts_sha256=_array_sha256(counts),
        policy_excitatory_sha256=_array_sha256(policy_e),
        policy_inhibitory_sha256=_array_sha256(policy_h),
        policy_arrays_reused_by_reference=True,
        fresh_artifact_hashes=(),
        work=work,
        checkpoint_sha256=checkpoint,
    )
    policy = object.__new__(FrozenStringRelationPolicyV2)
    policy._base_policy = base_policy
    policy._checkpoint_payload = checkpoint_payload
    policy._policy_excitatory = policy_e
    policy._policy_inhibitory = policy_h
    policy._receipts = frozen_receipts
    policy._relation_counts = counts
    policy._relation_excitatory = excitatory
    policy._relation_inhibitory = inhibitory
    policy._relation_masks = masks
    policy._signature = training.signature
    policy._state = state
    policy.authenticate()
    return policy


class StringRolloutWorkV2(NamedTuple):
    root_rollouts: int
    affordance_races: int
    policy_races: int
    effect_races: int
    selected_actuations: int
    selected_program_executions: int
    selected_case_evaluations: int
    committed_edits: int
    revisions: int
    lineage_traces: int
    unselected_splice_executions: int
    source_materializations_during_build: int
    desired_reads_during_build: int


class StringRelationFrontierStateV2(NamedTuple):
    status: str
    policy_checkpoint_sha256: str
    signature: TypedSignature
    case_count: int
    root_program_sha256: str
    terminal_keys: tuple[str, ...]
    terminal_program_sha256s: tuple[str, ...]
    terminal_semantic_sha256s: tuple[str, ...]
    applied_relation_receipt_sha256s: tuple[str, ...]
    work: StringRolloutWorkV2
    sha256: str


class StringRelationWitnessV2(NamedTuple):
    key: str
    program: typed_program.Program
    source: str
    physical_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    applied_relation_receipt_sha256s: tuple[str, ...]
    sha256: str


class StringRelationSelectionV2(NamedTuple):
    status: str
    frontier_sha256: str
    winner_key: str | None
    exact_proof_count: int
    witness: StringRelationWitnessV2 | None
    race: semantic_race.TypedSemanticVectorRaceAudit
    materialized_witnesses: int
    sha256: str


class _StringTerminal(NamedTuple):
    key: str
    program: typed_program.Program
    physical_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    applied_receipts: tuple[str, ...]


def _matches_relation_side(
    policy: FrozenStringRelationPolicyV2,
    cell: int,
    side: int,
    effect: PhysicalSequenceEffectV2,
) -> bool:
    targets, _ = _descriptor(effect)
    mask = policy._relation_masks[cell, side]
    if not bool(np.any(mask)):
        return False
    for route in np.flatnonzero(mask):
        target = targets[int(route)]
        if (
            int(policy._relation_excitatory[cell, side, route, target]) <= 0
            or int(policy._relation_inhibitory[cell, side, route, target]) != 0
        ):
            return False
    return True


def _exterior_site(program: typed_program.Program) -> typed_program.ProgramEditSite:
    sites = tuple(typed_program.program_edit_sites(program))
    exterior = tuple(site for site in sites if site.tokens == program.tokens)
    if len(exterior) != 1 or exterior[0].result_type_label != "String":
        raise RuntimeError("typed program lost its unique exterior String site cell")
    return exterior[0]


def _affordance_race(
    policy: FrozenStringRelationPolicyV2,
    program: typed_program.Program,
    effect: PhysicalSequenceEffectV2,
) -> tuple[
    str,
    StringRelationReceiptV2 | None,
    semantic_race.TypedSemanticVectorRaceAudit | None,
]:
    site = _exterior_site(program)
    matches = tuple(
        receipt
        for receipt in policy._receipts
        if receipt.exterior_site_cell == int(site.tokens == program.tokens)
        and _matches_relation_side(policy, receipt.cell_index, PRECONDITION, effect)
    )
    if not matches:
        return STOP_NO_AFFORDANCE, None, None
    if len(matches) != 1:
        return STOP_AMBIGUOUS_AFFORDANCE, None, None
    winner = matches[0]
    candidates = tuple(sorted(policy._receipts, key=lambda item: item.receipt_sha256))
    winner_index = candidates.index(winner)
    keys = tuple(
        _digest(
            b"string-affordance-key-v2",
            {
                "receipt": item.receipt_sha256,
                "site_cell": int(site.tokens == program.tokens),
                "donor": sha256(_canonical_json(item.donor_tokens)).hexdigest(),
            },
        )
        for item in candidates
    )
    race = semantic_race.select_exact_semantic_vector(
        keys,
        tuple(_encode_index(index) for index in range(len(candidates))),
        _encode_index(winner_index),
    )
    if race.status != semantic_race.SELECTED or race.winner_key != keys[winner_index]:
        raise RuntimeError("physical relation/site/donor race lost its learned cell")
    return READY, winner, race


def _effect_race(
    matched: bool,
) -> semantic_race.TypedSemanticVectorRaceAudit:
    keys = ("physical-effect:mismatch", "physical-effect:match")
    race = semantic_race.select_exact_semantic_vector(
        keys,
        (_encode_index(0), _encode_index(1)),
        _encode_index(int(matched)),
    )
    if race.status != semantic_race.SELECTED:
        raise RuntimeError("physical relation-effect race was not unique")
    return race


def _actuate_selected(
    program: typed_program.Program,
    receipt: StringRelationReceiptV2,
) -> typed_program.Program:
    _exterior_site(program)
    if receipt.actuation_kind == "replace-exterior":
        tokens = receipt.donor_tokens
    elif receipt.actuation_kind == "one-hole-context":
        tokens = (
            *receipt.context_prefix_tokens,
            *program.tokens,
            *receipt.context_suffix_tokens,
        )
    else:
        raise RuntimeError("learned String actuation kind changed")
    # Program construction validates the typed token expression but does not
    # render source.  Source rendering remains behind the final unique race.
    return typed_program.Program(tuple(tokens), program.signature)


def _rollout_work(
    *,
    root_rollouts: int = 0,
    affordance_races: int = 0,
    policy_races: int = 0,
    effect_races: int = 0,
    selected_actuations: int = 0,
    selected_program_executions: int = 0,
    selected_case_evaluations: int = 0,
    committed_edits: int = 0,
    revisions: int = 0,
    lineage_traces: int = 0,
) -> StringRolloutWorkV2:
    return StringRolloutWorkV2(
        root_rollouts=root_rollouts,
        affordance_races=affordance_races,
        policy_races=policy_races,
        effect_races=effect_races,
        selected_actuations=selected_actuations,
        selected_program_executions=selected_program_executions,
        selected_case_evaluations=selected_case_evaluations,
        committed_edits=committed_edits,
        revisions=revisions,
        lineage_traces=lineage_traces,
        unselected_splice_executions=UNSELECTED_SPLICE_EXECUTIONS,
        source_materializations_during_build=0,
        desired_reads_during_build=0,
    )


def _frontier_state(
    *,
    status: str,
    policy: FrozenStringRelationPolicyV2,
    signature: TypedSignature,
    case_count: int,
    root_program_sha256: str,
    terminals: tuple[_StringTerminal, ...],
    applied: tuple[str, ...],
    work: StringRolloutWorkV2,
) -> StringRelationFrontierStateV2:
    partial = StringRelationFrontierStateV2(
        status=status,
        policy_checkpoint_sha256=policy.checkpoint_sha256,
        signature=signature,
        case_count=case_count,
        root_program_sha256=root_program_sha256,
        terminal_keys=tuple(item.key for item in terminals),
        terminal_program_sha256s=tuple(item.program.sha256 for item in terminals),
        terminal_semantic_sha256s=tuple(
            _encoding_vector_sha256(item.physical_outputs) for item in terminals
        ),
        applied_relation_receipt_sha256s=applied,
        work=work,
        sha256="",
    )
    payload = partial._asdict()
    payload["signature"] = _signature_record(signature)
    payload["work"] = work._asdict()
    payload.pop("sha256")
    return partial._replace(sha256=_digest(_FRONTIER_NAMESPACE, payload))


class StringRelationFrontierV2:
    """Desired-free recurrent rollout over fresh visible String cases."""

    __slots__ = (
        "_materialization_count",
        "_policy",
        "_state",
        "_terminals",
    )

    def __init__(self) -> None:
        raise TypeError("use StringRelationFrontierV2.build")

    @property
    def state(self) -> StringRelationFrontierStateV2:
        self._policy.authenticate()
        return self._state

    @property
    def status(self) -> str:
        return self._state.status

    @property
    def source_materialization_count(self) -> int:
        return self._materialization_count

    @classmethod
    def build(
        cls,
        policy: FrozenStringRelationPolicyV2,
        signature: TypedSignature,
        cases: Iterable[tuple[str]],
        *,
        context_cell_gate: bool = True,
        lineage_cell_gate: bool = True,
        policy_cell_gate: bool = True,
        effect_cell_gate: bool = True,
        actuation_gate: bool = True,
    ) -> StringRelationFrontierV2:
        policy.authenticate()
        _validate_string_signature(signature)
        if signature != policy._signature:
            raise ValueError("fresh String signature must match the TRAIN signature")
        frozen_cases = tuple(cases)
        if not frozen_cases or any(
            type(row) is not tuple
            or len(row) != 1
            or type(row[0]) is not str
            or len(row[0]) > STRING_CAPACITY
            for row in frozen_cases
        ):
            raise ValueError("fresh cases must be singleton String rows")
        root = typed_program.Program(("variable-0",), signature)
        stop_status = next(
            (
                status
                for enabled, status in (
                    (context_cell_gate, STOP_CONTEXT),
                    (lineage_cell_gate, STOP_LINEAGE),
                    (policy_cell_gate, STOP_POLICY),
                    (effect_cell_gate, STOP_EFFECT),
                    (actuation_gate, STOP_ACTUATION),
                )
                if not enabled
            ),
            None,
        )
        if stop_status is not None:
            terminals: tuple[_StringTerminal, ...] = ()
            state = _frontier_state(
                status=stop_status,
                policy=policy,
                signature=signature,
                case_count=len(frozen_cases),
                root_program_sha256=root.sha256,
                terminals=terminals,
                applied=(),
                work=_rollout_work(),
            )
            instance = object.__new__(cls)
            instance._policy = policy
            instance._state = state
            instance._terminals = terminals
            instance._materialization_count = 0
            return instance

        inputs = tuple(row[0] for row in frozen_cases)
        root_outputs = inputs
        current_program = root
        current_outputs = root_outputs
        current_effect = _physical_sequence_effect(inputs, current_outputs)
        terminals_list = [
            _StringTerminal(
                key="terminal:root",
                program=root,
                physical_outputs=_encode_strings(root_outputs),
                applied_receipts=(),
            )
        ]
        applied: list[str] = []
        affordance_races = 0
        policy_races = 0
        effect_races = 0
        selected_actuations = 0
        selected_executions = 0
        case_evaluations = 0
        commits = 0
        revisions = 0
        lineage_traces = 1
        status = READY
        for _ in range(MAX_SELECTED_ACTUATIONS):
            affordance_races += 1
            relation_status, receipt, _ = _affordance_race(
                policy,
                current_program,
                current_effect,
            )
            if relation_status != READY or receipt is None:
                status = relation_status
                break
            action, _ = recurrent._policy_race(
                policy._policy_excitatory,
                policy._policy_inhibitory,
                recurrent.FEATURE_UNIQUE,
            )
            policy_races += 1
            if action != recurrent.ACTION_APPLY:
                status = STOP_POLICY
                break
            selected_program = _actuate_selected(current_program, receipt)
            selected_outputs = _evaluate_program_without_source(selected_program, frozen_cases)
            selected_actuations += 1
            selected_executions += 1
            case_evaluations += len(frozen_cases)
            observed_effect = _physical_sequence_effect(inputs, selected_outputs)
            lineage_traces += 1
            matched = _matches_relation_side(
                policy,
                receipt.cell_index,
                POSTCONDITION,
                observed_effect,
            )
            _effect_race(matched)
            effect_races += 1
            feature = (
                recurrent.FEATURE_EFFECT_MATCH
                if matched
                else recurrent.FEATURE_EFFECT_MISMATCH
            )
            effect_action, _ = recurrent._policy_race(
                policy._policy_excitatory,
                policy._policy_inhibitory,
                feature,
            )
            policy_races += 1
            if not matched or effect_action != recurrent.ACTION_COMMIT:
                revisions = int(effect_action == recurrent.ACTION_REVISE)
                status = STOP_EFFECT
                break
            current_program = selected_program
            current_outputs = selected_outputs
            current_effect = observed_effect
            applied.append(receipt.receipt_sha256)
            commits += 1
            terminals_list.append(
                _StringTerminal(
                    key=f"terminal:commit:{commits}",
                    program=current_program,
                    physical_outputs=_encode_strings(current_outputs),
                    applied_receipts=tuple(applied),
                )
            )
        if status == READY and (
            selected_actuations != MAX_SELECTED_ACTUATIONS
            or commits != MAX_SELECTED_ACTUATIONS
            or len(set(applied)) != RELATION_COUNT
        ):
            status = STOP_POLICY_INVARIANT
        terminals = tuple(terminals_list)
        work = _rollout_work(
            root_rollouts=1,
            affordance_races=affordance_races,
            policy_races=policy_races,
            effect_races=effect_races,
            selected_actuations=selected_actuations,
            selected_program_executions=selected_executions,
            selected_case_evaluations=case_evaluations,
            committed_edits=commits,
            revisions=revisions,
            lineage_traces=lineage_traces,
        )
        state = _frontier_state(
            status=status,
            policy=policy,
            signature=signature,
            case_count=len(frozen_cases),
            root_program_sha256=root.sha256,
            terminals=terminals,
            applied=tuple(applied),
            work=work,
        )
        instance = object.__new__(cls)
        instance._policy = policy
        instance._state = state
        instance._terminals = terminals
        instance._materialization_count = 0
        return instance

    def select(
        self,
        desired: Sequence[output_cells.TypedOutputEncoding],
    ) -> StringRelationSelectionV2:
        self._policy.authenticate()
        frozen_desired = tuple(desired)
        race = semantic_race.select_exact_semantic_vector(
            tuple(item.key for item in self._terminals),
            tuple(item.physical_outputs for item in self._terminals),
            frozen_desired,
        )
        exact_count = len(race.exact_winner_keys)
        witness = None
        if race.status == semantic_race.SELECTED and race.winner_key is not None:
            terminal = next(item for item in self._terminals if item.key == race.winner_key)
            compiled = typed_program.compile(terminal.program)
            self._materialization_count += 1
            witness_sha256 = _digest(
                b"causal-string-relation-witness-v2",
                {
                    "frontier": self._state.sha256,
                    "key": terminal.key,
                    "program": terminal.program.sha256,
                    "source": _bytes_sha256(compiled.source.encode("utf-8")),
                    "outputs": _encoding_vector_sha256(terminal.physical_outputs),
                    "receipts": terminal.applied_receipts,
                },
            )
            witness = StringRelationWitnessV2(
                key=terminal.key,
                program=terminal.program,
                source=compiled.source,
                physical_outputs=terminal.physical_outputs,
                applied_relation_receipt_sha256s=terminal.applied_receipts,
                sha256=witness_sha256,
            )
        selection_sha256 = _digest(
            b"causal-string-relation-selection-v2",
            {
                "frontier": self._state.sha256,
                "status": race.status,
                "winner": race.winner_key,
                "exact_count": exact_count,
                "witness": None if witness is None else witness.sha256,
                "race": race.sha256,
            },
        )
        return StringRelationSelectionV2(
            status=race.status,
            frontier_sha256=self._state.sha256,
            winner_key=race.winner_key,
            exact_proof_count=exact_count,
            witness=witness,
            race=race,
            materialized_witnesses=int(witness is not None),
            sha256=selection_sha256,
        )


class MicroRepositoryRepairSpecV2(NamedTuple):
    repo_root: Path
    source_relative_path: str
    case_manifest_relative_path: str
    expected_manifest_relative_path: str
    test_command: tuple[str, ...]
    root_program_sha256: str
    root_source_sha256: str
    test_timeout_seconds: float = 30.0


class RootSourceAttestationV2(NamedTuple):
    status: str
    policy_checkpoint_sha256: str
    root_program_sha256: str
    relation_receipt_sha256s: tuple[str, ...]
    expected_source_sha256: str
    observed_source_sha256: str
    sha256: str


class PreparedMicroRepositoryRepairStateV2(NamedTuple):
    status: str
    policy_checkpoint_sha256: str
    root_attestation_sha256: str
    case_count: int
    frontier_sha256: str | None
    relation_receipt_sha256s: tuple[str, ...]
    desired_reads_during_prepare: int
    source_materializations_during_prepare: int
    patch_writes_during_prepare: int
    repository_tests_during_prepare: int
    root_rollouts: int
    affordance_races: int
    policy_races: int
    effect_races: int
    selected_policy_actuations: int
    selected_program_executions: int
    committed_edits: int
    revisions: int
    unselected_splice_executions: int
    sha256: str


class MicroRepositoryRepairWorkV2(NamedTuple):
    source_reads: int
    case_manifest_reads: int
    desired_manifest_reads: int
    root_source_renderings: int
    terminal_selection_races: int
    source_materializations: int
    selected_policy_actuations: int
    selected_program_executions: int
    selected_patch_sources: int
    unselected_patch_sources: int
    patched_file_count: int
    patch_writes: int
    unselected_patch_writes: int
    repository_consequence_roots: int
    repository_tests: int
    unselected_repository_tests: int
    consequence_policy_races: int
    unselected_splice_executions: int
    commits: int
    revisions: int
    reverts: int


class MicroRepositoryRepairResultV2(NamedTuple):
    status: str
    prepared_sha256: str
    root_attestation_sha256: str
    selection_status: str | None
    winner_key: str | None
    winner_program_sha256: str | None
    exact_proof_count: int
    applied_relation_receipt_sha256s: tuple[str, ...]
    source_before_sha256: str
    source_after_sha256: str
    selection_race_sha256: str | None
    consequence_action_key: str | None
    consequence_race_sha256: str | None
    repository_test_exit_code: int | None
    repository_test_stdout_sha256: str | None
    repository_test_stderr_sha256: str | None
    work: MicroRepositoryRepairWorkV2
    sha256: str


def _child_path(root: Path, relative_path: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError("microrepository paths must be non-empty and relative")
    child = (root / relative_path).resolve()
    try:
        child.relative_to(root)
    except ValueError as exc:
        raise ValueError("microrepository path escaped its root") from exc
    return child


def _decode_string_value(value: object, declared_type: ProgramType) -> str:
    if declared_type != ProgramType.String() or type(value) is not str:
        raise ValueError("manifest value does not match its declared String type")
    if len(value) > STRING_CAPACITY or any(ord(character) > 127 for character in value):
        raise ValueError("manifest String exceeds the physical ASCII cell bank")
    return value


def _read_case_manifest(
    path: Path,
    signature: TypedSignature,
) -> tuple[tuple[str], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or tuple(payload) != ("cases",):
        raise ValueError("case manifest must contain only a cases field")
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("case manifest must contain at least one case")
    declared = signature.parameters[0].type
    return tuple((_decode_string_value(value, declared),) for value in raw_cases)


def _read_expected_manifest(
    path: Path,
    signature: TypedSignature,
    *,
    case_count: int,
) -> tuple[output_cells.TypedOutputEncoding, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or tuple(payload) != ("expected",):
        raise ValueError("expected manifest must contain only an expected field")
    raw_expected = payload["expected"]
    if not isinstance(raw_expected, list) or len(raw_expected) != case_count:
        raise ValueError("expected manifest must align with visible cases")
    values = tuple(
        _decode_string_value(value, signature.result) for value in raw_expected
    )
    return _encode_strings(values)


def _root_source_attestation(
    policy: FrozenStringRelationPolicyV2,
    spec: MicroRepositoryRepairSpecV2,
    observed_source: bytes,
) -> RootSourceAttestationV2:
    policy.authenticate()
    receipts = policy.relation_receipts
    valid = (
        len(receipts) == RELATION_COUNT
        and len({item.receipt_sha256 for item in receipts}) == RELATION_COUNT
        and any(item.actuation_kind == "one-hole-context" for item in receipts)
        and policy.state.work.transferred_abs_relation_bytes == 0
    )
    root_program = policy.state.training_root_program_sha256 if valid else ""
    status = (
        PREPARED
        if valid
        and root_program == spec.root_program_sha256
        and _bytes_sha256(observed_source) == spec.root_source_sha256
        else STOP_SOURCE_DRIFT
        if valid
        else STOP_POLICY_INVARIANT
    )
    receipt_sha256s = tuple(item.receipt_sha256 for item in receipts)
    partial = {
        "status": status,
        "policy": policy.checkpoint_sha256,
        "root_program": root_program,
        "receipts": receipt_sha256s,
        "expected_source": spec.root_source_sha256,
        "observed_source": _bytes_sha256(observed_source),
    }
    return RootSourceAttestationV2(
        status=status,
        policy_checkpoint_sha256=policy.checkpoint_sha256,
        root_program_sha256=root_program,
        relation_receipt_sha256s=receipt_sha256s,
        expected_source_sha256=spec.root_source_sha256,
        observed_source_sha256=_bytes_sha256(observed_source),
        sha256=_digest(_ATTESTATION_NAMESPACE, partial),
    )


def _prepared_state(
    *,
    status: str,
    policy: FrozenStringRelationPolicyV2,
    attestation: RootSourceAttestationV2,
    case_count: int,
    frontier: StringRelationFrontierV2 | None,
) -> PreparedMicroRepositoryRepairStateV2:
    frontier_state = None if frontier is None else frontier.state
    work = None if frontier_state is None else frontier_state.work
    partial = PreparedMicroRepositoryRepairStateV2(
        status=status,
        policy_checkpoint_sha256=policy.checkpoint_sha256,
        root_attestation_sha256=attestation.sha256,
        case_count=case_count,
        frontier_sha256=None if frontier_state is None else frontier_state.sha256,
        relation_receipt_sha256s=(
            () if frontier_state is None else frontier_state.applied_relation_receipt_sha256s
        ),
        desired_reads_during_prepare=0,
        source_materializations_during_prepare=0,
        patch_writes_during_prepare=0,
        repository_tests_during_prepare=0,
        root_rollouts=0 if work is None else work.root_rollouts,
        affordance_races=0 if work is None else work.affordance_races,
        policy_races=0 if work is None else work.policy_races,
        effect_races=0 if work is None else work.effect_races,
        selected_policy_actuations=0 if work is None else work.selected_actuations,
        selected_program_executions=(
            0 if work is None else work.selected_program_executions
        ),
        committed_edits=0 if work is None else work.committed_edits,
        revisions=0 if work is None else work.revisions,
        unselected_splice_executions=(
            0 if work is None else work.unselected_splice_executions
        ),
        sha256="",
    )
    payload = partial._asdict()
    payload.pop("sha256")
    return partial._replace(sha256=_digest(_PREPARED_NAMESPACE, payload))


def _repair_work(
    *,
    source_reads: int,
    case_manifest_reads: int,
    desired_manifest_reads: int,
    selected_policy_actuations: int,
    selected_program_executions: int,
    terminal_selection_races: int = 0,
    source_materializations: int = 0,
    selected_patch_sources: int = 0,
    patched_file_count: int = 0,
    patch_writes: int = 0,
    repository_consequence_roots: int = 0,
    repository_tests: int = 0,
    consequence_policy_races: int = 0,
    commits: int = 0,
    revisions: int = 0,
    reverts: int = 0,
) -> MicroRepositoryRepairWorkV2:
    return MicroRepositoryRepairWorkV2(
        source_reads=source_reads,
        case_manifest_reads=case_manifest_reads,
        desired_manifest_reads=desired_manifest_reads,
        root_source_renderings=0,
        terminal_selection_races=terminal_selection_races,
        source_materializations=source_materializations,
        selected_policy_actuations=selected_policy_actuations,
        selected_program_executions=selected_program_executions,
        selected_patch_sources=selected_patch_sources,
        unselected_patch_sources=UNSELECTED_SOURCE_MATERIALIZATIONS,
        patched_file_count=patched_file_count,
        patch_writes=patch_writes,
        unselected_patch_writes=UNSELECTED_PATCH_WRITES,
        repository_consequence_roots=repository_consequence_roots,
        repository_tests=repository_tests,
        unselected_repository_tests=UNSELECTED_REPOSITORY_TESTS,
        consequence_policy_races=consequence_policy_races,
        unselected_splice_executions=UNSELECTED_SPLICE_EXECUTIONS,
        commits=commits,
        revisions=revisions,
        reverts=reverts,
    )


@dataclass(slots=True)
class PreparedMicroRepositoryRepairV2:
    _policy: FrozenStringRelationPolicyV2
    _spec: MicroRepositoryRepairSpecV2
    _repo_root: Path
    _source_path: Path
    _expected_path: Path
    _original_source: bytes
    _attestation: RootSourceAttestationV2
    _frontier: StringRelationFrontierV2 | None
    state: PreparedMicroRepositoryRepairStateV2
    _source_reads: int
    _case_manifest_reads: int
    _consumed: bool = False

    def _result(
        self,
        *,
        status: str,
        work: MicroRepositoryRepairWorkV2,
        source_after_sha256: str,
        selection: StringRelationSelectionV2 | None = None,
        consequence_action_key: str | None = None,
        consequence_race_sha256: str | None = None,
        repository_test_exit_code: int | None = None,
        repository_test_stdout_sha256: str | None = None,
        repository_test_stderr_sha256: str | None = None,
    ) -> MicroRepositoryRepairResultV2:
        witness = None if selection is None else selection.witness
        payload = {
            "status": status,
            "prepared": self.state.sha256,
            "attestation": self._attestation.sha256,
            "selection_status": None if selection is None else selection.status,
            "winner": None if selection is None else selection.winner_key,
            "winner_program": None if witness is None else witness.program.sha256,
            "exact_proof_count": 0 if selection is None else selection.exact_proof_count,
            "relations": (
                () if witness is None else witness.applied_relation_receipt_sha256s
            ),
            "source_before": self._attestation.observed_source_sha256,
            "source_after": source_after_sha256,
            "selection_race": None if selection is None else selection.race.sha256,
            "consequence_action": consequence_action_key,
            "consequence_race": consequence_race_sha256,
            "test_exit": repository_test_exit_code,
            "test_stdout": repository_test_stdout_sha256,
            "test_stderr": repository_test_stderr_sha256,
            "work": work._asdict(),
        }
        return MicroRepositoryRepairResultV2(
            status=status,
            prepared_sha256=self.state.sha256,
            root_attestation_sha256=self._attestation.sha256,
            selection_status=None if selection is None else selection.status,
            winner_key=None if selection is None else selection.winner_key,
            winner_program_sha256=None if witness is None else witness.program.sha256,
            exact_proof_count=0 if selection is None else selection.exact_proof_count,
            applied_relation_receipt_sha256s=(
                () if witness is None else witness.applied_relation_receipt_sha256s
            ),
            source_before_sha256=self._attestation.observed_source_sha256,
            source_after_sha256=source_after_sha256,
            selection_race_sha256=(
                None if selection is None else selection.race.sha256
            ),
            consequence_action_key=consequence_action_key,
            consequence_race_sha256=consequence_race_sha256,
            repository_test_exit_code=repository_test_exit_code,
            repository_test_stdout_sha256=repository_test_stdout_sha256,
            repository_test_stderr_sha256=repository_test_stderr_sha256,
            work=work,
            sha256=_digest(_RESULT_NAMESPACE, payload),
        )

    def repair(self) -> MicroRepositoryRepairResultV2:
        """Read desired once, select/materialize/patch/test exactly once."""
        if self._consumed:
            raise RuntimeError("prepared microrepository repair is one-shot")
        self._consumed = True
        actuations = self.state.selected_policy_actuations
        executions = self.state.selected_program_executions
        if self.state.status != PREPARED or self._frontier is None:
            return self._result(
                status=self.state.status,
                source_after_sha256=self._attestation.observed_source_sha256,
                work=_repair_work(
                    source_reads=self._source_reads,
                    case_manifest_reads=self._case_manifest_reads,
                    desired_manifest_reads=0,
                    selected_policy_actuations=actuations,
                    selected_program_executions=executions,
                ),
            )
        try:
            current_source = self._source_path.read_bytes()
        except OSError:
            current_source = b""
        source_reads = self._source_reads + 1
        current_sha256 = _bytes_sha256(current_source)
        if current_source != self._original_source:
            return self._result(
                status=STOP_SOURCE_DRIFT,
                source_after_sha256=current_sha256,
                work=_repair_work(
                    source_reads=source_reads,
                    case_manifest_reads=self._case_manifest_reads,
                    desired_manifest_reads=0,
                    selected_policy_actuations=actuations,
                    selected_program_executions=executions,
                ),
            )
        try:
            desired = _read_expected_manifest(
                self._expected_path,
                self.state.frontier_sha256 and self._frontier.state.signature,
                case_count=self.state.case_count,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            return self._result(
                status=STOP_ADAPTER_INPUT,
                source_after_sha256=current_sha256,
                work=_repair_work(
                    source_reads=source_reads,
                    case_manifest_reads=self._case_manifest_reads,
                    desired_manifest_reads=1,
                    selected_policy_actuations=actuations,
                    selected_program_executions=executions,
                ),
            )
        selection = self._frontier.select(desired)
        source_materializations = self._frontier.source_materialization_count
        if selection.status != semantic_race.SELECTED or selection.witness is None:
            return self._result(
                status=selection.status,
                selection=selection,
                source_after_sha256=current_sha256,
                work=_repair_work(
                    source_reads=source_reads,
                    case_manifest_reads=self._case_manifest_reads,
                    desired_manifest_reads=1,
                    terminal_selection_races=1,
                    source_materializations=source_materializations,
                    selected_policy_actuations=actuations,
                    selected_program_executions=executions,
                ),
            )
        witness = selection.witness
        expected_receipts = {item.receipt_sha256 for item in self._policy.relation_receipts}
        observed_receipts = set(witness.applied_relation_receipt_sha256s)
        if (
            selection.exact_proof_count != 1
            or selection.materialized_witnesses != 1
            or source_materializations != 1
            or len(witness.applied_relation_receipt_sha256s) != RELATION_COUNT
            or observed_receipts != expected_receipts
            or actuations != MAX_SELECTED_ACTUATIONS
            or executions != MAX_SELECTED_ACTUATIONS
        ):
            return self._result(
                status=STOP_POLICY_INVARIANT,
                selection=selection,
                source_after_sha256=current_sha256,
                work=_repair_work(
                    source_reads=source_reads,
                    case_manifest_reads=self._case_manifest_reads,
                    desired_manifest_reads=1,
                    terminal_selection_races=1,
                    source_materializations=source_materializations,
                    selected_policy_actuations=actuations,
                    selected_program_executions=executions,
                ),
            )
        selected_source = witness.source.encode("utf-8")
        selected_source_sha256 = _bytes_sha256(selected_source)
        try:
            self._source_path.write_bytes(selected_source)
        except OSError:
            return self._result(
                status=STOP_PATCH_WRITE,
                selection=selection,
                source_after_sha256=current_sha256,
                work=_repair_work(
                    source_reads=source_reads,
                    case_manifest_reads=self._case_manifest_reads,
                    desired_manifest_reads=1,
                    terminal_selection_races=1,
                    source_materializations=source_materializations,
                    selected_policy_actuations=actuations,
                    selected_program_executions=executions,
                    selected_patch_sources=1,
                ),
            )
        try:
            completed = subprocess.run(
                self._spec.test_command,
                cwd=self._repo_root,
                capture_output=True,
                check=False,
                timeout=self._spec.test_timeout_seconds,
            )
            exit_code = int(completed.returncode)
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            exit_code = -1
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
        except OSError as exc:
            exit_code = -1
            stdout = b""
            stderr = str(exc).encode("utf-8", errors="replace")
        feature = (
            recurrent.FEATURE_EFFECT_MATCH
            if exit_code == 0
            else recurrent.FEATURE_EFFECT_MISMATCH
        )
        action, consequence_race = recurrent._policy_race(
            self._policy._policy_excitatory,
            self._policy._policy_inhibitory,
            feature,
        )
        action_key = recurrent.ACTION_KEYS[action]
        common = {
            "source_reads": source_reads,
            "case_manifest_reads": self._case_manifest_reads,
            "desired_manifest_reads": 1,
            "terminal_selection_races": 1,
            "source_materializations": source_materializations,
            "selected_policy_actuations": actuations,
            "selected_program_executions": executions,
            "selected_patch_sources": 1,
            "patched_file_count": 1,
            "patch_writes": 1,
            "repository_consequence_roots": 1,
            "repository_tests": 1,
            "consequence_policy_races": 1,
        }
        stdout_sha256 = _bytes_sha256(stdout)
        stderr_sha256 = _bytes_sha256(stderr)
        if exit_code == 0 and action == recurrent.ACTION_COMMIT:
            return self._result(
                status=COMMITTED,
                selection=selection,
                source_after_sha256=selected_source_sha256,
                consequence_action_key=action_key,
                consequence_race_sha256=consequence_race.sha256,
                repository_test_exit_code=exit_code,
                repository_test_stdout_sha256=stdout_sha256,
                repository_test_stderr_sha256=stderr_sha256,
                work=_repair_work(**common, commits=1),
            )
        self._source_path.write_bytes(self._original_source)
        status = (
            STOP_TEST_FAILURE
            if exit_code != 0 and action == recurrent.ACTION_REVISE
            else STOP_CONSEQUENCE_POLICY
        )
        return self._result(
            status=status,
            selection=selection,
            source_after_sha256=self._attestation.observed_source_sha256,
            consequence_action_key=action_key,
            consequence_race_sha256=consequence_race.sha256,
            repository_test_exit_code=exit_code,
            repository_test_stdout_sha256=stdout_sha256,
            repository_test_stderr_sha256=stderr_sha256,
            work=_repair_work(
                **common,
                revisions=int(action == recurrent.ACTION_REVISE),
                reverts=1,
            ),
        )


def prepare_microrepository_repair_v2(
    policy: FrozenStringRelationPolicyV2,
    signature: TypedSignature,
    spec: MicroRepositoryRepairSpecV2,
    *,
    context_cell_gate: bool = True,
    lineage_cell_gate: bool = True,
    policy_cell_gate: bool = True,
    effect_cell_gate: bool = True,
    actuation_gate: bool = True,
) -> PreparedMicroRepositoryRepairV2:
    """Authenticate a fresh root and freeze every terminal before desired read."""
    policy.authenticate()
    _validate_string_signature(signature)
    repo_root = spec.repo_root.resolve(strict=True)
    if not repo_root.is_dir() or not spec.test_command:
        raise ValueError("microrepository root and test command are required")
    source_path = _child_path(repo_root, spec.source_relative_path)
    case_path = _child_path(repo_root, spec.case_manifest_relative_path)
    expected_path = _child_path(repo_root, spec.expected_manifest_relative_path)
    try:
        original_source = source_path.read_bytes()
    except OSError:
        original_source = b""
    attestation = _root_source_attestation(policy, spec, original_source)
    if attestation.status != PREPARED:
        state = _prepared_state(
            status=attestation.status,
            policy=policy,
            attestation=attestation,
            case_count=0,
            frontier=None,
        )
        return PreparedMicroRepositoryRepairV2(
            policy,
            spec,
            repo_root,
            source_path,
            expected_path,
            original_source,
            attestation,
            None,
            state,
            1,
            0,
        )
    try:
        cases = _read_case_manifest(case_path, signature)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        state = _prepared_state(
            status=STOP_ADAPTER_INPUT,
            policy=policy,
            attestation=attestation,
            case_count=0,
            frontier=None,
        )
        return PreparedMicroRepositoryRepairV2(
            policy,
            spec,
            repo_root,
            source_path,
            expected_path,
            original_source,
            attestation,
            None,
            state,
            1,
            1,
        )
    frontier = policy.build(
        signature,
        cases,
        context_cell_gate=context_cell_gate,
        lineage_cell_gate=lineage_cell_gate,
        policy_cell_gate=policy_cell_gate,
        effect_cell_gate=effect_cell_gate,
        actuation_gate=actuation_gate,
    )
    status = frontier.status
    if status == READY:
        build = frontier.state
        work = build.work
        if (
            work.desired_reads_during_build != 0
            or work.source_materializations_during_build != 0
            or len(build.terminal_keys) != 3
            or build.terminal_program_sha256s[0] != attestation.root_program_sha256
            or work.selected_actuations != MAX_SELECTED_ACTUATIONS
            or work.selected_program_executions != MAX_SELECTED_ACTUATIONS
            or work.committed_edits != MAX_SELECTED_ACTUATIONS
            or work.revisions != 0
            or work.unselected_splice_executions != 0
            or len(build.applied_relation_receipt_sha256s) != RELATION_COUNT
            or set(build.applied_relation_receipt_sha256s)
            != {item.receipt_sha256 for item in policy.relation_receipts}
        ):
            status = STOP_POLICY_INVARIANT
        else:
            status = PREPARED
    state = _prepared_state(
        status=status,
        policy=policy,
        attestation=attestation,
        case_count=len(cases),
        frontier=frontier,
    )
    return PreparedMicroRepositoryRepairV2(
        policy,
        spec,
        repo_root,
        source_path,
        expected_path,
        original_source,
        attestation,
        frontier,
        state,
        1,
        1,
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "COMMITTED",
    "MAX_SELECTED_ACTUATIONS",
    "PREPARED",
    "READY",
    "RELATION_COUNT",
    "STOP_ACTUATION",
    "STOP_ADAPTER_INPUT",
    "STOP_AMBIGUOUS_AFFORDANCE",
    "STOP_CONSEQUENCE_POLICY",
    "STOP_CONTEXT",
    "STOP_EFFECT",
    "STOP_LINEAGE",
    "STOP_NO_AFFORDANCE",
    "STOP_PATCH_WRITE",
    "STOP_POLICY",
    "STOP_POLICY_INVARIANT",
    "STOP_SOURCE_DRIFT",
    "STOP_TEST_FAILURE",
    "FrozenStringRelationPolicyStateV2",
    "FrozenStringRelationPolicyV2",
    "MicroRepositoryRepairResultV2",
    "MicroRepositoryRepairSpecV2",
    "PreparedMicroRepositoryRepairStateV2",
    "PreparedMicroRepositoryRepairV2",
    "SolvedStringTrainingV2",
    "StringRelationFrontierStateV2",
    "StringRelationFrontierV2",
    "StringRelationReceiptV2",
    "StringRelationSelectionV2",
    "fit_string_relation_policy_v2",
    "prepare_microrepository_repair_v2",
    "solve_string_relation_training_v2",
]
