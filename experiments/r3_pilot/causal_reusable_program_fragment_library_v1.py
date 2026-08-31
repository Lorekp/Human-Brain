"""Learn and recombine typed program fragments from solved physical plans.

This experiment is the smallest bridge from a flat desired-free plan bank to a
learned library.  Two solved self-posed plans are first selected by the bank's
physical semantic race.  Only after those races may their typed subtree
occurrences enter a content-addressed cell memory.  Repeated exact occurrences
consolidate into the same learned prototype cell.

Fresh construction is desired-independent.  It reactivates every stored
prototype through the same exact physical race, enumerates all cross-program
one-subtree relations allowed by the existing typed edit metadata, and retains
only structurally valid recipes.  The recipe bank is evaluated on visible cases
without rendering Python source.  A desired vector is accepted only by
``select``; zero or tied exact tiers stop, while a unique winning recipe may be
materialized and compiled.

The typed edit-site extractor, type/scope compatibility, prefix-token splice,
visible-case evaluator, compiler, two solved lessons, and one-edit composition
depth remain explicit scaffolding.  This is self-posed compositional reuse, not
prompt parsing, arbitrary program induction, public evaluation, or broad coding
capability evidence.
"""

from __future__ import annotations

import ast
import builtins
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import NamedTuple

import numpy as np
from experiments.r3_pilot import causal_task_observation_candidate_plan_bank_v1 as plan_bank
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from humanbrain import typed_program
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "CAUSAL_REUSABLE_PROGRAM_FRAGMENT_LIBRARY_V1"
READY = f"{SCHEMA}_READY"
STOP_TRAINING = f"STOP_{SCHEMA}_TRAINING"
STOP_DESCRIPTOR_COLLISION = f"STOP_{SCHEMA}_DESCRIPTOR_COLLISION"
STOP_PROTOTYPE = f"STOP_{SCHEMA}_PROTOTYPE"
STOP_LIBRARY_DISABLED = f"STOP_{SCHEMA}_LIBRARY_DISABLED"
STOP_NO_RECIPE = f"STOP_{SCHEMA}_NO_RECIPE"

SELECTED = semantic_race.SELECTED
STOP_NO_EXACT = semantic_race.STOP_NO_EXACT
STOP_MULTIPLE_EXACT = semantic_race.STOP_MULTIPLE_EXACT

DESCRIPTOR_ROUTES = 32
DESCRIPTOR_RADIX = 16
TICK_DTYPE = np.dtype(np.uint8)
COUNT_DTYPE = np.dtype(np.uint16)
TICK_CAP = int(np.iinfo(TICK_DTYPE).max)

DESIRED_READS_DURING_BUILD = 0
SOURCE_MATERIALIZATIONS_DURING_BUILD = 0

CLAIM_BOUNDARY = (
    "two self-posed plans selected by a 131-way physical bank race, exact typed "
    "subtree occurrences consolidated into learned content-addressed prototype "
    "cells, and desired-independent cross-program one-edit recipes selected by "
    "one fresh exact semantic-vector race; typed edit metadata, token framing, "
    "visible-case execution, compiler, lesson count, and one-edit depth remain "
    "explicit"
)

_CHECKPOINT_NAMESPACE = b"causal-reusable-program-fragment-library-v1"
_BUILD_NAMESPACE = b"causal-reusable-program-fragment-build-v1"
_WITNESS_NAMESPACE = b"causal-reusable-program-fragment-witness-v1"
_CONSTRUCTION_TOKEN = object()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\x00" + _canonical_json(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    framed = np.ascontiguousarray(value)
    return sha256(memoryview(framed).cast("B")).hexdigest()


def _race_cell_steps(race: semantic_race.TypedSemanticVectorRaceAudit) -> int:
    return sum(
        len(round_.currents_before_pa) * (round_.race.spike_step + 1)
        for round_ in race.tier.rounds
    )


def _encode_descriptor(
    targets: Sequence[int],
) -> tuple[output_cells.TypedOutputEncoding, ...]:
    frozen = tuple(targets)
    if len(frozen) != DESCRIPTOR_ROUTES or any(
        type(value) is not int or not 0 <= value < DESCRIPTOR_RADIX
        for value in frozen
    ):
        raise ValueError("fragment descriptor targets changed")
    return tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.INT,
        )
        for value in frozen
    )


def _site_payload(site: typed_program.ProgramEditSite) -> dict[str, object]:
    return {
        "token_ids": typed_program._program_token_ids(site.tokens),
        "result_type": site.result_type_label,
        "scope": site.scope_binding_type_labels,
        "is_binder": site.is_binder,
    }


def _site_descriptor(
    site: typed_program.ProgramEditSite,
) -> tuple[str, tuple[int, ...]]:
    root = sha256(_canonical_json(_site_payload(site))).hexdigest()
    targets = tuple(int(value, 16) for value in root[:DESCRIPTOR_ROUTES])
    return root, targets


def _learn_row(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    cell_index: int,
    targets: Sequence[int],
) -> tuple[int, int]:
    e_ticks = 0
    h_ticks = 0
    for route, winner in enumerate(targets):
        for target in range(DESCRIPTOR_RADIX):
            state = excitatory if target == winner else inhibitory
            before = int(state[cell_index, route, target])
            state[cell_index, route, target] = min(TICK_CAP, before + 1)
            delta = int(state[cell_index, route, target]) - before
            if target == winner:
                e_ticks += delta
            else:
                h_ticks += delta
    return e_ticks, h_ticks


def _prototype_targets(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    cell_index: int,
) -> tuple[int, ...]:
    scores = (
        excitatory[cell_index].astype(np.int16)
        - inhibitory[cell_index].astype(np.int16)
    )
    targets: list[int] = []
    for route in range(DESCRIPTOR_ROUTES):
        row = scores[route]
        winner = int(np.argmax(row))
        if int(row[winner]) <= 0 or int(np.count_nonzero(row == row[winner])) != 1:
            raise RuntimeError("fragment prototype lost its unique physical target")
        targets.append(winner)
    return tuple(targets)


def _recall_prototype(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    targets: Sequence[int],
) -> semantic_race.TypedSemanticVectorRaceAudit | None:
    if excitatory.shape[0] == 0:
        return None
    keys = tuple(f"fragment-cell:{index}" for index in range(excitatory.shape[0]))
    vectors = tuple(
        _encode_descriptor(_prototype_targets(excitatory, inhibitory, index))
        for index in range(excitatory.shape[0])
    )
    return semantic_race.select_exact_semantic_vector(
        keys,
        vectors,
        _encode_descriptor(targets),
    )


class FragmentLibraryFitStop(RuntimeError):
    def __init__(self, status: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class FragmentOrigin(NamedTuple):
    program_index: int
    path: tuple[int, ...]


class FragmentPrototypeRecord(NamedTuple):
    cell_index: int
    descriptor_sha256: str
    descriptor_targets: tuple[int, ...]
    tokens: tuple[str, ...]
    result_type_label: str
    scope_binding_type_labels: tuple[str, str]
    is_binder: bool
    origins: tuple[FragmentOrigin, ...]


class LearnedProgramRecord(NamedTuple):
    program_index: int
    winner_key_sha256: str
    program_sha256: str
    program_tokens: tuple[str, ...]
    root_fragment_cell: int
    selection_race_sha256: str
    plan_slice_receipt_sha256: str


class FragmentLibraryTrainingWork(NamedTuple):
    solved_plan_races: int
    solved_plan_selector_cell_steps: int
    fragment_occurrences: int
    fragment_admission_races: int
    fragment_selector_cell_steps: int
    unique_fragment_cells: int
    consolidated_occurrences: int
    excitatory_ticks: int
    inhibitory_ticks: int
    persistent_cell_state_bytes: int
    host_fragment_payload_bytes: int


class FragmentLibraryState(NamedTuple):
    status: str
    training_program_count: int
    training_program_sha256s: tuple[str, ...]
    training_plan_receipt_sha256s: tuple[str, ...]
    fragment_cell_count: int
    fragment_occurrence_count: int
    recurring_fragment_cell_count: int
    excitatory_sha256: str
    inhibitory_sha256: str
    occurrence_counts_sha256: str
    work: FragmentLibraryTrainingWork
    checkpoint_sha256: str


class FragmentCompositionWork(NamedTuple):
    prototype_recall_races: int
    prototype_selector_cell_steps: int
    relation_attempts: int
    type_compatible_relations: int
    structurally_valid_relations: int
    duplicate_recipe_relations: int
    unsupported_recipe_relations: int
    unique_recipe_count: int
    candidate_case_executions: int
    physical_output_encodings: int


class FragmentCompositionState(NamedTuple):
    status: str
    checkpoint_sha256: str
    case_count: int
    recipe_count: int
    recipe_keys: tuple[str, ...]
    recipe_relation_sha256s: tuple[str, ...]
    desired_reads_during_build: int
    source_materializations_during_build: int
    work: FragmentCompositionWork
    sha256: str


class FragmentCompositionWitness(NamedTuple):
    program: typed_program.Program
    source: str
    relation_sha256: str
    sha256: str


class FragmentCompositionSelection(NamedTuple):
    status: str
    winner_key: str | None
    exact_winner_keys: tuple[str, ...]
    exact_proof_count: int
    witness: FragmentCompositionWitness | None
    materialized_witnesses: int
    race: semantic_race.TypedSemanticVectorRaceAudit | None


class _MutablePrototype:
    __slots__ = (
        "descriptor_sha256",
        "descriptor_targets",
        "is_binder",
        "origins",
        "result_type_label",
        "scope_binding_type_labels",
        "tokens",
    )

    def __init__(
        self,
        site: typed_program.ProgramEditSite,
        descriptor_sha256: str,
        descriptor_targets: tuple[int, ...],
        origin: FragmentOrigin,
    ) -> None:
        self.descriptor_sha256 = descriptor_sha256
        self.descriptor_targets = descriptor_targets
        self.tokens = site.tokens
        self.result_type_label = site.result_type_label
        self.scope_binding_type_labels = site.scope_binding_type_labels
        self.is_binder = site.is_binder
        self.origins = [origin]


@dataclass(frozen=True)
class _Recipe:
    key: str
    tokens: tuple[str, ...]
    program_sha256: str
    physical_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    relation_sha256: str


def _checkpoint_payload(
    program_records: tuple[LearnedProgramRecord, ...],
    prototype_records: tuple[FragmentPrototypeRecord, ...],
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    occurrence_counts: np.ndarray,
    work: FragmentLibraryTrainingWork,
) -> dict[str, object]:
    return {
        "programs": tuple(program_records),
        "prototypes": tuple(prototype_records),
        "excitatory": _array_sha256(excitatory),
        "inhibitory": _array_sha256(inhibitory),
        "occurrence_counts": _array_sha256(occurrence_counts),
        "work": work,
    }


class FrozenReusableProgramFragmentLibraryV1:
    """Frozen exact fragment prototypes and authenticated occurrence relations."""

    __slots__ = (
        "_excitatory",
        "_inhibitory",
        "_occurrence_counts",
        "_program_records",
        "_prototype_records",
        "_signature",
        "_state",
    )

    def __init__(
        self,
        program_records: tuple[LearnedProgramRecord, ...],
        prototype_records: tuple[FragmentPrototypeRecord, ...],
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        occurrence_counts: np.ndarray,
        signature: TypedSignature,
        state: FragmentLibraryState,
        *,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("fragment library requires private construction authority")
        self._program_records = program_records
        self._prototype_records = prototype_records
        self._excitatory = excitatory
        self._inhibitory = inhibitory
        self._occurrence_counts = occurrence_counts
        self._signature = signature
        self._state = state
        self.authenticate()

    @property
    def state(self) -> FragmentLibraryState:
        self.authenticate()
        return self._state

    @property
    def checkpoint_sha256(self) -> str:
        return self._state.checkpoint_sha256

    @property
    def learned_programs(self) -> tuple[LearnedProgramRecord, ...]:
        self.authenticate()
        return self._program_records

    @property
    def fragment_prototypes(self) -> tuple[FragmentPrototypeRecord, ...]:
        self.authenticate()
        return self._prototype_records

    def authenticate(self) -> None:
        state = self._state
        count = len(self._prototype_records)
        if (
            self._excitatory.shape
            != (count, DESCRIPTOR_ROUTES, DESCRIPTOR_RADIX)
            or self._inhibitory.shape != self._excitatory.shape
            or self._occurrence_counts.shape != (count,)
            or self._excitatory.dtype != TICK_DTYPE
            or self._inhibitory.dtype != TICK_DTYPE
            or self._occurrence_counts.dtype != COUNT_DTYPE
            or self._excitatory.flags.writeable
            or self._inhibitory.flags.writeable
            or self._occurrence_counts.flags.writeable
        ):
            raise RuntimeError("fragment library cell-state geometry changed")
        for index, record in enumerate(self._prototype_records):
            if record.cell_index != index:
                raise RuntimeError("fragment prototype cell order changed")
            site = typed_program.ProgramEditSite(
                path=(),
                tokens=record.tokens,
                result_type_label=record.result_type_label,
                head_production=record.tokens[0],
                scope_binding_type_labels=record.scope_binding_type_labels,
                is_binder=record.is_binder,
            )
            root, targets = _site_descriptor(site)
            if (
                root != record.descriptor_sha256
                or targets != record.descriptor_targets
                or targets
                != _prototype_targets(self._excitatory, self._inhibitory, index)
                or int(self._occurrence_counts[index]) != len(record.origins)
                or not record.origins
            ):
                raise RuntimeError("fragment prototype authentication failed")
        if len(self._program_records) != 2:
            raise RuntimeError("fragment library lesson count changed")
        for expected_index, program_record in enumerate(self._program_records):
            if (
                program_record.program_index != expected_index
                or typed_program.Program(
                    program_record.program_tokens,
                    self._signature,
                ).sha256
                != program_record.program_sha256
            ):
                raise RuntimeError("learned program authentication failed")
            root_record = self._prototype_records[program_record.root_fragment_cell]
            if FragmentOrigin(expected_index, ()) not in root_record.origins:
                raise RuntimeError("learned program lost its root prototype relation")
        payload = _checkpoint_payload(
            self._program_records,
            self._prototype_records,
            self._excitatory,
            self._inhibitory,
            self._occurrence_counts,
            state.work,
        )
        checkpoint = _digest(_CHECKPOINT_NAMESPACE, payload)
        if (
            state.status != READY
            or state.training_program_count != len(self._program_records)
            or state.training_program_sha256s
            != tuple(item.program_sha256 for item in self._program_records)
            or state.training_plan_receipt_sha256s
            != tuple(
                item.plan_slice_receipt_sha256 for item in self._program_records
            )
            or state.fragment_cell_count != count
            or state.fragment_occurrence_count
            != int(self._occurrence_counts.astype(np.uint64).sum())
            or state.recurring_fragment_cell_count
            != int(np.count_nonzero(self._occurrence_counts > 1))
            or state.excitatory_sha256 != _array_sha256(self._excitatory)
            or state.inhibitory_sha256 != _array_sha256(self._inhibitory)
            or state.occurrence_counts_sha256
            != _array_sha256(self._occurrence_counts)
            or state.checkpoint_sha256 != checkpoint
        ):
            raise RuntimeError("fragment library checkpoint changed")

    def build(
        self,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        *,
        prototype_cell_gate: bool = True,
    ) -> ReusableProgramFragmentFrontierV1:
        return ReusableProgramFragmentFrontierV1._build(
            self,
            signature,
            cases,
            prototype_cell_gate=prototype_cell_gate,
        )


def fit_reusable_program_fragment_library_v1(
    bank: plan_bank.CausalTaskObservationCandidatePlanBankV1,
    solved_desired_vectors: Sequence[
        Sequence[output_cells.TypedOutputEncoding]
    ],
) -> FrozenReusableProgramFragmentLibraryV1:
    """Select two solved plans and learn exact typed subtree prototypes."""

    if type(bank) is not plan_bank.CausalTaskObservationCandidatePlanBankV1:
        raise TypeError("exact candidate plan bank required")
    bank._authenticate()
    lessons = tuple(tuple(vector) for vector in solved_desired_vectors)
    if len(lessons) != 2:
        raise ValueError("fragment library V1 requires exactly two solved lessons")

    selections = tuple(bank.select(vector) for vector in lessons)
    if any(
        selection.status != SELECTED
        or selection.witness is None
        or selection.exact_proof_count != 1
        for selection in selections
    ):
        raise FragmentLibraryFitStop(
            STOP_TRAINING,
            "both solved lessons must yield one physically selected witness",
        )
    witnesses = tuple(selection.witness for selection in selections)
    if witnesses[0].program.signature != witnesses[1].program.signature:
        raise FragmentLibraryFitStop(
            STOP_TRAINING, "solved lessons changed typed signature"
        )
    if len({witness.program.sha256 for witness in witnesses}) != 2:
        raise FragmentLibraryFitStop(
            STOP_TRAINING, "solved lessons must select distinct programs"
        )

    excitatory = np.zeros(
        (0, DESCRIPTOR_ROUTES, DESCRIPTOR_RADIX), dtype=TICK_DTYPE
    )
    inhibitory = np.zeros_like(excitatory)
    occurrence_counts = np.zeros((0,), dtype=COUNT_DTYPE)
    prototypes: list[_MutablePrototype] = []
    occurrence_to_cell: dict[FragmentOrigin, int] = {}
    admission_races: list[semantic_race.TypedSemanticVectorRaceAudit] = []
    e_ticks = 0
    h_ticks = 0

    for program_index, witness in enumerate(witnesses):
        for site in typed_program.program_edit_sites(witness.program):
            origin = FragmentOrigin(program_index, site.path)
            root, targets = _site_descriptor(site)
            race = _recall_prototype(excitatory, inhibitory, targets)
            if race is not None:
                admission_races.append(race)
            if race is None or race.status == STOP_NO_EXACT:
                excitatory = np.concatenate(
                    (
                        excitatory,
                        np.zeros(
                            (1, DESCRIPTOR_ROUTES, DESCRIPTOR_RADIX),
                            dtype=TICK_DTYPE,
                        ),
                    ),
                    axis=0,
                )
                inhibitory = np.concatenate(
                    (
                        inhibitory,
                        np.zeros(
                            (1, DESCRIPTOR_ROUTES, DESCRIPTOR_RADIX),
                            dtype=TICK_DTYPE,
                        ),
                    ),
                    axis=0,
                )
                occurrence_counts = np.concatenate(
                    (occurrence_counts, np.zeros((1,), dtype=COUNT_DTYPE))
                )
                cell_index = len(prototypes)
                prototypes.append(_MutablePrototype(site, root, targets, origin))
            elif race.status == SELECTED and race.winner_key is not None:
                cell_index = int(race.winner_key.rsplit(":", 1)[1])
                prototype = prototypes[cell_index]
                if (
                    prototype.descriptor_sha256 != root
                    or prototype.descriptor_targets != targets
                    or prototype.tokens != site.tokens
                    or prototype.result_type_label != site.result_type_label
                    or prototype.scope_binding_type_labels
                    != site.scope_binding_type_labels
                    or prototype.is_binder != site.is_binder
                ):
                    raise FragmentLibraryFitStop(
                        STOP_DESCRIPTOR_COLLISION,
                        "physical fragment address collided across distinct occurrences",
                    )
                prototype.origins.append(origin)
            elif race.status == STOP_MULTIPLE_EXACT:
                raise FragmentLibraryFitStop(
                    STOP_PROTOTYPE,
                    "multiple learned cells matched one exact fragment occurrence",
                )
            else:
                raise FragmentLibraryFitStop(
                    STOP_PROTOTYPE,
                    "fragment admission race returned an unsupported state",
                )
            learned_e, learned_h = _learn_row(
                excitatory,
                inhibitory,
                cell_index,
                targets,
            )
            e_ticks += learned_e
            h_ticks += learned_h
            before = int(occurrence_counts[cell_index])
            if before == np.iinfo(COUNT_DTYPE).max:
                raise FragmentLibraryFitStop(
                    STOP_TRAINING, "fragment occurrence counter saturated"
                )
            occurrence_counts[cell_index] = before + 1
            occurrence_to_cell[origin] = cell_index

    prototype_records = tuple(
        FragmentPrototypeRecord(
            cell_index=index,
            descriptor_sha256=prototype.descriptor_sha256,
            descriptor_targets=prototype.descriptor_targets,
            tokens=prototype.tokens,
            result_type_label=prototype.result_type_label,
            scope_binding_type_labels=prototype.scope_binding_type_labels,
            is_binder=prototype.is_binder,
            origins=tuple(prototype.origins),
        )
        for index, prototype in enumerate(prototypes)
    )
    candidate_by_key = {candidate.key: candidate for candidate in bank._candidates}
    program_records: list[LearnedProgramRecord] = []
    for program_index, (selection, witness) in enumerate(
        zip(selections, witnesses, strict=True)
    ):
        if selection.winner_key is None:
            raise RuntimeError("selected lesson lost its winner key")
        candidate = candidate_by_key[selection.winner_key]
        program_records.append(
            LearnedProgramRecord(
                program_index=program_index,
                winner_key_sha256=sha256(
                    selection.winner_key.encode("utf-8")
                ).hexdigest(),
                program_sha256=witness.program.sha256,
                program_tokens=witness.program.tokens,
                root_fragment_cell=occurrence_to_cell[
                    FragmentOrigin(program_index, ())
                ],
                selection_race_sha256=selection.race.sha256,
                plan_slice_receipt_sha256=candidate.receipt_sha256,
            )
        )

    excitatory.setflags(write=False)
    inhibitory.setflags(write=False)
    occurrence_counts.setflags(write=False)
    fragment_occurrences = int(occurrence_counts.astype(np.uint64).sum())
    cell_bytes = excitatory.nbytes + inhibitory.nbytes + occurrence_counts.nbytes
    host_payload_bytes = sum(
        sum(len(token.encode("utf-8")) for token in record.tokens)
        for record in prototype_records
    )
    solved_selector_steps = sum(_race_cell_steps(item.race) for item in selections)
    prototype_selector_steps = sum(_race_cell_steps(item) for item in admission_races)
    work = FragmentLibraryTrainingWork(
        solved_plan_races=len(selections),
        solved_plan_selector_cell_steps=solved_selector_steps,
        fragment_occurrences=fragment_occurrences,
        fragment_admission_races=len(admission_races),
        fragment_selector_cell_steps=prototype_selector_steps,
        unique_fragment_cells=len(prototype_records),
        consolidated_occurrences=fragment_occurrences - len(prototype_records),
        excitatory_ticks=e_ticks,
        inhibitory_ticks=h_ticks,
        persistent_cell_state_bytes=cell_bytes,
        host_fragment_payload_bytes=host_payload_bytes,
    )
    frozen_program_records = tuple(program_records)
    checkpoint = _digest(
        _CHECKPOINT_NAMESPACE,
        _checkpoint_payload(
            frozen_program_records,
            prototype_records,
            excitatory,
            inhibitory,
            occurrence_counts,
            work,
        ),
    )
    state = FragmentLibraryState(
        status=READY,
        training_program_count=len(frozen_program_records),
        training_program_sha256s=tuple(
            item.program_sha256 for item in frozen_program_records
        ),
        training_plan_receipt_sha256s=tuple(
            item.plan_slice_receipt_sha256 for item in frozen_program_records
        ),
        fragment_cell_count=len(prototype_records),
        fragment_occurrence_count=fragment_occurrences,
        recurring_fragment_cell_count=int(np.count_nonzero(occurrence_counts > 1)),
        excitatory_sha256=_array_sha256(excitatory),
        inhibitory_sha256=_array_sha256(inhibitory),
        occurrence_counts_sha256=_array_sha256(occurrence_counts),
        work=work,
        checkpoint_sha256=checkpoint,
    )
    library = object.__new__(FrozenReusableProgramFragmentLibraryV1)
    library._program_records = frozen_program_records
    library._prototype_records = prototype_records
    library._excitatory = excitatory
    library._inhibitory = inhibitory
    library._occurrence_counts = occurrence_counts
    library._state = state
    library._signature = witnesses[0].program.signature
    library.authenticate()
    return library


def _splice_tokens(
    base: typed_program.Program,
    path: tuple[int, ...],
    replacement: tuple[str, ...],
) -> tuple[str, ...]:
    root = typed_program._validate_program(base.tokens, base.signature)
    selected = root
    for part in path:
        if part >= len(selected.children):
            raise ValueError("typed edit path changed")
        selected = selected.children[part]
    tokens = (
        *base.tokens[: selected.token_start],
        *replacement,
        *base.tokens[selected.token_stop :],
    )
    program = typed_program.Program(tokens, base.signature)
    typed_program._validate_program(program.tokens, program.signature)
    return program.tokens


def _execute_without_rendering(
    tokens: tuple[str, ...],
    signature: TypedSignature,
    cases: tuple[tuple[object, ...], ...],
) -> tuple[int, ...]:
    expression = typed_program._validate_program(tokens, signature)
    environment = tuple(
        parameter.name for parameter in reversed(signature.parameters)
    )
    module = ast.Module(
        body=[
            ast.FunctionDef(
                name=signature.entry_point,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=parameter.name) for parameter in signature.parameters],
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
    executable = builtins.compile(module, "<fragment-recipe>", "exec")
    exec(executable, {}, namespace)
    function = namespace[signature.entry_point]
    outputs: list[int] = []
    for case in cases:
        arguments = tuple(list(value) if type(value) is list else value for value in case)
        value = function(*arguments)  # type: ignore[operator]
        if type(value) is not int:
            raise TypeError("fragment recipe did not produce an exact integer")
        outputs.append(value)
    return tuple(outputs)


def _encode_outputs(
    values: Sequence[int],
) -> tuple[output_cells.TypedOutputEncoding, ...] | None:
    encoded = tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.INT,
        )
        for value in values
    )
    return encoded if all(item.supported for item in encoded) else None


def _build_state_sha256(state: FragmentCompositionState) -> str:
    return _digest(
        _BUILD_NAMESPACE,
        {
            "status": state.status,
            "checkpoint": state.checkpoint_sha256,
            "case_count": state.case_count,
            "recipe_count": state.recipe_count,
            "recipe_keys": state.recipe_keys,
            "relations": state.recipe_relation_sha256s,
            "desired_reads": state.desired_reads_during_build,
            "source_materializations": state.source_materializations_during_build,
            "work": state.work,
        },
    )


class ReusableProgramFragmentFrontierV1:
    """Desired-free fresh recipe bank backed by learned fragment cells."""

    __slots__ = (
        "_cases",
        "_library",
        "_materialized",
        "_recipes",
        "_signature",
        "_source_materialization_count",
        "_state",
    )

    def __init__(self) -> None:
        raise TypeError("use FrozenReusableProgramFragmentLibraryV1.build")

    @classmethod
    def _build(
        cls,
        library: FrozenReusableProgramFragmentLibraryV1,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        *,
        prototype_cell_gate: bool,
    ) -> ReusableProgramFragmentFrontierV1:
        library.authenticate()
        if type(prototype_cell_gate) is not bool:
            raise TypeError("prototype cell gate must be an exact boolean")
        if signature != library._signature:
            raise ValueError("fresh recipe signature changed from solved lessons")
        frozen_cases = tuple(cases)
        if not frozen_cases:
            raise ValueError("fresh fragment construction requires visible cases")
        if any(len(case) != len(signature.parameters) for case in frozen_cases):
            raise ValueError("fresh cases do not match typed signature")

        recalls: list[semantic_race.TypedSemanticVectorRaceAudit] = []
        recalled_cells: set[int] = set()
        if prototype_cell_gate:
            for record in library._prototype_records:
                race = _recall_prototype(
                    library._excitatory,
                    library._inhibitory,
                    record.descriptor_targets,
                )
                if (
                    race is None
                    or race.status != SELECTED
                    or race.winner_key != f"fragment-cell:{record.cell_index}"
                ):
                    raise FragmentLibraryFitStop(
                        STOP_PROTOTYPE,
                        "frozen fragment did not reactivate its authenticated cell",
                    )
                recalls.append(race)
                recalled_cells.add(record.cell_index)

        relation_attempts = 0
        compatible_relations = 0
        valid_relations = 0
        duplicate_relations = 0
        unsupported_relations = 0
        executions = 0
        encodings = 0
        recipes_by_program: dict[str, _Recipe] = {}
        training_hashes = {
            record.program_sha256 for record in library._program_records
        }

        if prototype_cell_gate:
            for base_record in library._program_records:
                if base_record.root_fragment_cell not in recalled_cells:
                    raise RuntimeError("base program root cell was not reactivated")
                base = typed_program.Program(base_record.program_tokens, signature)
                base_sites = typed_program.program_edit_sites(base)
                for site in base_sites:
                    if not site.path:
                        continue
                    for donor in library._prototype_records:
                        if donor.cell_index not in recalled_cells:
                            continue
                        if not any(
                            origin.program_index != base_record.program_index
                            for origin in donor.origins
                        ):
                            continue
                        relation_attempts += 1
                        if (
                            donor.tokens == site.tokens
                            or donor.result_type_label != site.result_type_label
                            or donor.scope_binding_type_labels
                            != site.scope_binding_type_labels
                            or donor.is_binder != site.is_binder
                        ):
                            continue
                        compatible_relations += 1
                        try:
                            tokens = _splice_tokens(base, site.path, donor.tokens)
                        except (
                            typed_program.ProgramFormatError,
                            TypeError,
                            ValueError,
                        ):
                            continue
                        program = typed_program.Program(tokens, signature)
                        if program.sha256 in training_hashes:
                            continue
                        valid_relations += 1
                        relation_sha = _digest(
                            b"fragment-relation",
                            {
                                "checkpoint": library.checkpoint_sha256,
                                "base_root_cell": base_record.root_fragment_cell,
                                "base_selection": base_record.selection_race_sha256,
                                "edit_path": site.path,
                                "donor_cell": donor.cell_index,
                                "donor_descriptor": donor.descriptor_sha256,
                                "program": program.sha256,
                            },
                        )
                        try:
                            values = _execute_without_rendering(
                                program.tokens,
                                signature,
                                frozen_cases,
                            )
                        except (ArithmeticError, RuntimeError, TypeError, ValueError):
                            unsupported_relations += 1
                            continue
                        executions += len(frozen_cases)
                        physical_outputs = _encode_outputs(values)
                        if physical_outputs is None:
                            unsupported_relations += 1
                            continue
                        encodings += len(physical_outputs)
                        key = "recipe:" + _digest(
                            b"fragment-recipe-key",
                            {
                                "program": program.sha256,
                                "relation": relation_sha,
                            },
                        )
                        recipe = _Recipe(
                            key=key,
                            tokens=program.tokens,
                            program_sha256=program.sha256,
                            physical_outputs=physical_outputs,
                            relation_sha256=relation_sha,
                        )
                        if program.sha256 in recipes_by_program:
                            duplicate_relations += 1
                            continue
                        recipes_by_program[program.sha256] = recipe

        recipes = tuple(
            sorted(recipes_by_program.values(), key=lambda item: item.key)
        )
        if not prototype_cell_gate:
            status = STOP_LIBRARY_DISABLED
        elif not recipes:
            status = STOP_NO_RECIPE
        else:
            status = READY
        work = FragmentCompositionWork(
            prototype_recall_races=len(recalls),
            prototype_selector_cell_steps=sum(_race_cell_steps(item) for item in recalls),
            relation_attempts=relation_attempts,
            type_compatible_relations=compatible_relations,
            structurally_valid_relations=valid_relations,
            duplicate_recipe_relations=duplicate_relations,
            unsupported_recipe_relations=unsupported_relations,
            unique_recipe_count=len(recipes),
            candidate_case_executions=executions,
            physical_output_encodings=encodings,
        )
        partial = FragmentCompositionState(
            status=status,
            checkpoint_sha256=library.checkpoint_sha256,
            case_count=len(frozen_cases),
            recipe_count=len(recipes),
            recipe_keys=tuple(item.key for item in recipes),
            recipe_relation_sha256s=tuple(
                item.relation_sha256 for item in recipes
            ),
            desired_reads_during_build=DESIRED_READS_DURING_BUILD,
            source_materializations_during_build=SOURCE_MATERIALIZATIONS_DURING_BUILD,
            work=work,
            sha256="",
        )
        state = partial._replace(sha256=_build_state_sha256(partial))
        instance = object.__new__(cls)
        instance._library = library
        instance._signature = signature
        instance._cases = frozen_cases
        instance._recipes = recipes
        instance._state = state
        instance._materialized = {}
        instance._source_materialization_count = 0
        instance._authenticate()
        return instance

    @property
    def state(self) -> FragmentCompositionState:
        self._authenticate()
        return self._state

    @property
    def status(self) -> str:
        return self._state.status

    @property
    def source_materialization_count(self) -> int:
        return self._source_materialization_count

    def _authenticate(self) -> None:
        self._library.authenticate()
        state = self._state
        if (
            self._signature != self._library._signature
            or state.checkpoint_sha256 != self._library.checkpoint_sha256
            or state.case_count != len(self._cases)
            or state.recipe_count != len(self._recipes)
            or state.recipe_keys != tuple(item.key for item in self._recipes)
            or state.recipe_relation_sha256s
            != tuple(item.relation_sha256 for item in self._recipes)
            or state.desired_reads_during_build != 0
            or state.source_materializations_during_build != 0
            or state.sha256 != _build_state_sha256(state)
        ):
            raise RuntimeError("fragment composition frontier changed")
        if len(set(state.recipe_keys)) != len(state.recipe_keys):
            raise RuntimeError("fragment recipe keys are not unique")
        training_hashes = {
            record.program_sha256 for record in self._library._program_records
        }
        for recipe in self._recipes:
            program = typed_program.Program(recipe.tokens, self._signature)
            if (
                program.sha256 != recipe.program_sha256
                or program.sha256 in training_hashes
            ):
                raise RuntimeError("fresh fragment recipe authentication failed")

    def select(
        self,
        desired_outputs: Sequence[output_cells.TypedOutputEncoding],
    ) -> FragmentCompositionSelection:
        self._authenticate()
        if self._state.status != READY:
            return FragmentCompositionSelection(
                status=self._state.status,
                winner_key=None,
                exact_winner_keys=(),
                exact_proof_count=0,
                witness=None,
                materialized_witnesses=0,
                race=None,
            )
        race = semantic_race.select_exact_semantic_vector(
            self._state.recipe_keys,
            tuple(item.physical_outputs for item in self._recipes),
            desired_outputs,
        )
        if race.status in {STOP_NO_EXACT, STOP_MULTIPLE_EXACT}:
            return FragmentCompositionSelection(
                status=race.status,
                winner_key=None,
                exact_winner_keys=race.exact_winner_keys,
                exact_proof_count=0,
                witness=None,
                materialized_witnesses=0,
                race=race,
            )
        if race.status != SELECTED or race.winner_key is None:
            raise RuntimeError("fragment semantic race returned unsupported state")
        recipe = next(item for item in self._recipes if item.key == race.winner_key)
        existing = self._materialized.get(recipe.key)
        materialized = 0
        if existing is None:
            program = typed_program.Program(recipe.tokens, self._signature)
            source = typed_program.compile(program).source
            witness_sha = _digest(
                _WITNESS_NAMESPACE,
                {
                    "frontier": self._state.sha256,
                    "relation": recipe.relation_sha256,
                    "program": program.sha256,
                    "source": source,
                },
            )
            existing = FragmentCompositionWitness(
                program=program,
                source=source,
                relation_sha256=recipe.relation_sha256,
                sha256=witness_sha,
            )
            self._materialized[recipe.key] = existing
            self._source_materialization_count += 1
            materialized = 1
        return FragmentCompositionSelection(
            status=SELECTED,
            winner_key=recipe.key,
            exact_winner_keys=race.exact_winner_keys,
            exact_proof_count=1,
            witness=existing,
            materialized_witnesses=materialized,
            race=race,
        )
