"""Desired-free bank of independently published aggregate plan candidates.

The bank composes three already-banked physical publication mechanisms over one
visible ``List[Int] -> Int`` case matrix:

* one raw SUM publication;
* one absolute(item) -> map -> SUM publication;
* one predicate-gated fold publication for every literal in ``[-64, 64]``.

All 131 plan slices are built and receipt-authenticated before a desired vector
is accepted.  Selection is one existing physical exact-semantic-vector race.
Only its unique winner may materialize a proof; zero and tied exact tiers stop
without materializing any candidate.

This is a self-posed visible-case plan-bank mechanism.  The candidate roster,
integer threshold interval, case framing, fitted predicate checkpoint, typed
compiler, and output encoding remain explicit.  It is not prompt parsing,
arbitrary program induction, public evaluation, or coding capability evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import NamedTuple, cast

from experiments.r3_pilot import causal_occurrence_absolute_map_fold_v1 as raw_sum
from experiments.r3_pilot import causal_predicate_gated_fold_v1 as predicate_fold
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_absolute_map_fold_v1 as absolute_fold
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v3 as breadth_v3
from experiments.r3_pilot import typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "CAUSAL_TASK_OBSERVATION_CANDIDATE_PLAN_BANK_V1"
READY = f"{SCHEMA}_READY"
STOP_PUBLICATION = f"STOP_{SCHEMA}_PUBLICATION"

SELECTED = semantic_race.SELECTED
STOP_NO_EXACT = semantic_race.STOP_NO_EXACT
STOP_MULTIPLE_EXACT = semantic_race.STOP_MULTIPLE_EXACT

RAW_SUM_KEY = "raw-sum"
ABSOLUTE_MAP_SUM_KEY = "absolute-map-sum"
PREDICATE_THRESHOLDS = tuple(range(-64, 65))
CANDIDATE_COUNT = 2 + len(PREDICATE_THRESHOLDS)
DESIRED_READS_DURING_BUILD = 0
PROOF_MATERIALIZATIONS_DURING_BUILD = 0
SEMANTIC_RACES_DURING_BUILD = 0

CLAIM_BOUNDARY = (
    "one self-posed desired-free bank of 131 independently receipt-authenticated "
    "raw-sum, absolute-map-sum, and item-less-than-literal gated-fold plan "
    "publications, followed by one physical exact-semantic-vector race per "
    "selection and selected-only singleton proof materialization; the fixed "
    "candidate roster, threshold interval, fitted predicate checkpoint, visible "
    "case framing, typed compiler, and output encoding remain explicit"
)

_RAW = "raw"
_ABSOLUTE = "absolute"
_PREDICATE = "predicate"


def predicate_candidate_key(threshold: int) -> str:
    if type(threshold) is not int or threshold not in PREDICATE_THRESHOLDS:
        raise ValueError("predicate threshold must be an exact integer in [-64, 64]")
    return f"predicate-item-less-than:{threshold}"


class CandidatePlanBankBuildStop(RuntimeError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.status = STOP_PUBLICATION
        self.detail = detail


class CandidatePlanBankState(NamedTuple):
    status: str
    candidate_count: int
    case_count: int
    candidate_keys: tuple[str, ...]
    plan_slice_receipt_sha256s: tuple[str, ...]
    desired_reads_during_build: int
    proof_materializations_during_build: int
    semantic_races_during_build: int


class CandidatePlanBankSelection(NamedTuple):
    status: str
    winner_key: str | None
    exact_winner_keys: tuple[str, ...]
    exact_proof_count: int
    witness: breadth_v3.PackedGenericTypedWitness | None
    materialized_witnesses: int
    race: semantic_race.TypedSemanticVectorRaceAudit


class _PublishedCandidate(NamedTuple):
    key: str
    kind: str
    threshold: int | None
    receipt_sha256: str
    physical_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    publication: object
    binding: object


def _freeze_candidate(
    *,
    key: str,
    kind: str,
    threshold: int | None,
    publication: object,
    binding: object | None = None,
) -> _PublishedCandidate:
    if kind == _RAW:
        typed_publication = cast(
            plan_slice.PackedGenericTypedPromptPlanSliceV1, publication
        )
        typed_publication._authenticate_state()
        if typed_publication.status != plan_slice.FRONTIER_READY:
            raise CandidatePlanBankBuildStop(f"{key} did not publish READY")
    elif kind == _ABSOLUTE:
        absolute_publication = cast(
            absolute_fold.PackedPhysicalAbsoluteMapFoldV1, publication
        )
        absolute_publication._authenticate()
        if absolute_publication.status != absolute_fold.READY:
            raise CandidatePlanBankBuildStop(f"{key} did not publish READY")
        typed_publication = absolute_publication
    elif kind == _PREDICATE:
        predicate_publication = cast(
            predicate_fold.PackedPhysicalPredicateGatedFoldV1, publication
        )
        predicate_publication._authenticate()
        if predicate_publication.status != predicate_fold.READY:
            raise CandidatePlanBankBuildStop(f"{key} did not publish READY")
        typed_publication = predicate_publication
    else:
        raise RuntimeError("candidate publication kind changed")

    receipt = typed_publication.plan_slice_receipt
    if receipt is None:
        raise CandidatePlanBankBuildStop(f"{key} lost its plan-slice receipt")
    authenticated = plan_slice._authenticate_plan_slice_receipt(
        typed_publication._frontier,
        receipt,
    )
    if binding is not None and authenticated is not binding:
        raise RuntimeError(f"{key} crossed its authenticated plan binding")
    work = typed_publication.work
    if work.concrete_proof_witnesses != 0 or work.materialized_witnesses != 0:
        raise CandidatePlanBankBuildStop(f"{key} materialized proof work during build")
    if (
        kind != _RAW
        and cast(
            absolute_fold.PackedPhysicalAbsoluteMapFoldV1
            | predicate_fold.PackedPhysicalPredicateGatedFoldV1,
            typed_publication,
        )._materialization
        is not None
    ):
        raise CandidatePlanBankBuildStop(f"{key} materialized a witness during build")
    physical_outputs = tuple(authenticated.candidate.eclass.physical_outputs)
    if not physical_outputs:
        raise CandidatePlanBankBuildStop(f"{key} published an empty semantic vector")
    return _PublishedCandidate(
        key=key,
        kind=kind,
        threshold=threshold,
        receipt_sha256=receipt.sha256,
        physical_outputs=physical_outputs,
        publication=typed_publication,
        binding=authenticated,
    )


class CausalTaskObservationCandidatePlanBankV1:
    """Immutable candidate roster with selected-only proof materialization."""

    def __init__(self) -> None:
        raise TypeError("use CausalTaskObservationCandidatePlanBankV1.build")

    @classmethod
    def build(
        cls,
        checkpoint: predicate_fold.FrozenPredicateGateV1,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
    ) -> CausalTaskObservationCandidatePlanBankV1:
        frozen_cases = tuple(cases)
        if not frozen_cases:
            raise ValueError("candidate plan bank requires visible cases")

        candidates: list[_PublishedCandidate] = []
        try:
            raw_publication, raw_binding = raw_sum._raw_sum_binding(
                signature,
                frozen_cases,
            )
        except raw_sum.OccurrenceAbsoluteMapFoldFitStop as error:
            raise CandidatePlanBankBuildStop(
                f"{RAW_SUM_KEY} stopped: {error.status}"
            ) from error
        candidates.append(
            _freeze_candidate(
                key=RAW_SUM_KEY,
                kind=_RAW,
                threshold=None,
                publication=raw_publication,
                binding=raw_binding,
            )
        )

        absolute_publication = absolute_fold.PackedPhysicalAbsoluteMapFoldV1.build(
            signature,
            frozen_cases,
        )
        candidates.append(
            _freeze_candidate(
                key=ABSOLUTE_MAP_SUM_KEY,
                kind=_ABSOLUTE,
                threshold=None,
                publication=absolute_publication,
            )
        )

        for threshold in PREDICATE_THRESHOLDS:
            key = predicate_candidate_key(threshold)
            try:
                publication = predicate_fold.PackedPhysicalPredicateGatedFoldV1.build(
                    checkpoint,
                    signature,
                    frozen_cases,
                    threshold,
                )
            except predicate_fold.PredicateGatedFoldBuildStop as error:
                raise CandidatePlanBankBuildStop(
                    f"{key} stopped: {error.status}"
                ) from error
            candidates.append(
                _freeze_candidate(
                    key=key,
                    kind=_PREDICATE,
                    threshold=threshold,
                    publication=publication,
                )
            )

        frozen_candidates = tuple(candidates)
        expected_keys = (
            RAW_SUM_KEY,
            ABSOLUTE_MAP_SUM_KEY,
            *(predicate_candidate_key(value) for value in PREDICATE_THRESHOLDS),
        )
        candidate_keys = tuple(item.key for item in frozen_candidates)
        if len(frozen_candidates) != CANDIDATE_COUNT or candidate_keys != expected_keys:
            raise RuntimeError("candidate plan-bank roster changed")
        if any(len(item.physical_outputs) != len(frozen_cases) for item in frozen_candidates):
            raise RuntimeError("candidate semantic vectors do not align with visible cases")

        state = CandidatePlanBankState(
            status=READY,
            candidate_count=len(frozen_candidates),
            case_count=len(frozen_cases),
            candidate_keys=candidate_keys,
            plan_slice_receipt_sha256s=tuple(
                item.receipt_sha256 for item in frozen_candidates
            ),
            desired_reads_during_build=DESIRED_READS_DURING_BUILD,
            proof_materializations_during_build=PROOF_MATERIALIZATIONS_DURING_BUILD,
            semantic_races_during_build=SEMANTIC_RACES_DURING_BUILD,
        )
        instance = object.__new__(cls)
        instance._candidates = frozen_candidates
        instance._state = state
        instance._witness_cache = {}
        instance._materialized_witness_count = 0
        instance._authenticate()
        return instance

    @property
    def state(self) -> CandidatePlanBankState:
        return self._state

    @property
    def materialized_witness_count(self) -> int:
        return self._materialized_witness_count

    def _authenticate(self) -> None:
        if self._state.status != READY:
            raise RuntimeError("candidate plan bank lost READY state")
        if self._state.candidate_count != CANDIDATE_COUNT:
            raise RuntimeError("candidate plan-bank count changed")
        if tuple(item.key for item in self._candidates) != self._state.candidate_keys:
            raise RuntimeError("candidate plan-bank keys changed")
        if (
            tuple(item.receipt_sha256 for item in self._candidates)
            != self._state.plan_slice_receipt_sha256s
        ):
            raise RuntimeError("candidate plan-bank receipts changed")
        for candidate in self._candidates:
            publication = candidate.publication
            if candidate.kind == _RAW:
                typed_publication = cast(
                    plan_slice.PackedGenericTypedPromptPlanSliceV1, publication
                )
                typed_publication._authenticate_state()
            elif candidate.kind == _ABSOLUTE:
                typed_publication = cast(
                    absolute_fold.PackedPhysicalAbsoluteMapFoldV1, publication
                )
                typed_publication._authenticate()
            elif candidate.kind == _PREDICATE:
                typed_publication = cast(
                    predicate_fold.PackedPhysicalPredicateGatedFoldV1, publication
                )
                typed_publication._authenticate()
            else:
                raise RuntimeError("candidate publication kind changed")
            receipt = typed_publication.plan_slice_receipt
            if receipt is None or receipt.sha256 != candidate.receipt_sha256:
                raise RuntimeError("candidate plan-slice receipt changed")
            binding = plan_slice._authenticate_plan_slice_receipt(
                typed_publication._frontier,
                receipt,
            )
            if binding is not candidate.binding:
                raise RuntimeError("candidate plan binding identity changed")
            if tuple(binding.candidate.eclass.physical_outputs) != candidate.physical_outputs:
                raise RuntimeError("candidate semantic vector changed")

    def _materialize_candidate(
        self,
        candidate: _PublishedCandidate,
    ) -> tuple[breadth_v3.PackedGenericTypedWitness, int]:
        cached = self._witness_cache.get(candidate.key)
        if cached is not None:
            return cached, 0

        if candidate.kind == _RAW:
            publication = cast(
                plan_slice.PackedGenericTypedPromptPlanSliceV1,
                candidate.publication,
            )
            binding = cast(plan_slice._IssuedPlanBinding, candidate.binding)
            forest = plan_slice._materialize_authorized_forest(binding)
            if forest.exact_logical_recipe_count != 1 or len(forest.first_two_witnesses) != 1:
                raise RuntimeError("selected raw-sum slice was not singleton")
            route_witness = binding.route._render_witness(
                forest.first_two_witnesses[0],
                binding.component,
            )
            witness = cast(
                breadth_v3.PackedGenericTypedWitness,
                publication._frontier._compile_witness(route_witness),
            )
            binding.route._concrete_proof_witnesses = 1
        elif candidate.kind == _ABSOLUTE:
            result = cast(
                absolute_fold.PackedPhysicalAbsoluteMapFoldV1,
                candidate.publication,
            ).materialize()
            if (
                result.status != absolute_fold.READY
                or result.exact_proof_count != 1
                or result.witness is None
            ):
                raise RuntimeError("selected absolute-map-sum slice was not singleton")
            witness = result.witness
        elif candidate.kind == _PREDICATE:
            result = cast(
                predicate_fold.PackedPhysicalPredicateGatedFoldV1,
                candidate.publication,
            ).materialize()
            if (
                result.status != predicate_fold.READY
                or result.exact_proof_count != 1
                or result.witness is None
            ):
                raise RuntimeError("selected predicate-gated slice was not singleton")
            witness = result.witness
        else:
            raise RuntimeError("candidate publication kind changed")

        self._witness_cache[candidate.key] = witness
        self._materialized_witness_count += 1
        return witness, 1

    def select(
        self,
        desired_outputs: Sequence[output_cells.TypedOutputEncoding],
    ) -> CandidatePlanBankSelection:
        self._authenticate()
        race = semantic_race.select_exact_semantic_vector(
            self._state.candidate_keys,
            tuple(item.physical_outputs for item in self._candidates),
            desired_outputs,
        )
        if race.status == STOP_NO_EXACT:
            return CandidatePlanBankSelection(
                status=STOP_NO_EXACT,
                winner_key=None,
                exact_winner_keys=race.exact_winner_keys,
                exact_proof_count=0,
                witness=None,
                materialized_witnesses=0,
                race=race,
            )
        if race.status == STOP_MULTIPLE_EXACT:
            return CandidatePlanBankSelection(
                status=STOP_MULTIPLE_EXACT,
                winner_key=None,
                exact_winner_keys=race.exact_winner_keys,
                exact_proof_count=0,
                witness=None,
                materialized_witnesses=0,
                race=race,
            )
        if race.status != SELECTED or race.winner_key is None:
            raise RuntimeError("candidate semantic race returned an unsupported state")
        candidate = next(
            item for item in self._candidates if item.key == race.winner_key
        )
        witness, materialized = self._materialize_candidate(candidate)
        return CandidatePlanBankSelection(
            status=SELECTED,
            winner_key=candidate.key,
            exact_winner_keys=race.exact_winner_keys,
            exact_proof_count=1,
            witness=witness,
            materialized_witnesses=materialized,
            race=race,
        )


__all__ = [
    "ABSOLUTE_MAP_SUM_KEY",
    "CANDIDATE_COUNT",
    "CLAIM_BOUNDARY",
    "DESIRED_READS_DURING_BUILD",
    "PREDICATE_THRESHOLDS",
    "PROOF_MATERIALIZATIONS_DURING_BUILD",
    "RAW_SUM_KEY",
    "READY",
    "SCHEMA",
    "SELECTED",
    "SEMANTIC_RACES_DURING_BUILD",
    "STOP_MULTIPLE_EXACT",
    "STOP_NO_EXACT",
    "STOP_PUBLICATION",
    "CandidatePlanBankBuildStop",
    "CandidatePlanBankSelection",
    "CandidatePlanBankState",
    "CausalTaskObservationCandidatePlanBankV1",
    "predicate_candidate_key",
]
