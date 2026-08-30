from __future__ import annotations

import inspect

import pytest
from experiments.r3_pilot import typed_semantic_packed_prompt_authorized_plan_slice_v1 as sliced
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

CASES = (
    ([],),
    ([-6, -2, 0, 8],),
    ([1, 2, 3],),
    ([-10, 10],),
    ([-1, 0, 1],),
)
CLAMP_LITERALS = (-2, 1, 2)
CLAMP_DESIRED = (0, -2, 5, 0, 0)

MAP_COUNT_CASES = (
    ([],),
    ([0, 1, 2, 3],),
    ([1, 2, 10],),
    ([0, 0, 5],),
    ([2, 2, 2],),
)
MAP_COUNT_LITERALS_19 = (
    -15,
    -14,
    -12,
    -10,
    -8,
    -6,
    -4,
    -2,
    0,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
)
MAP_COUNT_DESIRED = (0, 2, 2, 1, 3)

LITERAL_INT = ("literal", "Int")
ROLE_INT = ("role", "Int")
ROLE_LIST_INT = ("role", "List[Int]")


def _signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "aggregate",
        (ProgramParameter("numbers", ProgramType.List(integer)),),
        integer,
    )


def _map_count_certificate() -> sliced.StructuralFactorCertificate:
    occurrences = (
        sliced.StructuralOccurrence(
            None,
            None,
            (LITERAL_INT, LITERAL_INT, ROLE_INT, ROLE_LIST_INT),
            (),
        ),
        sliced.StructuralOccurrence(
            0,
            0,
            (LITERAL_INT, LITERAL_INT, ROLE_INT),
            (0, 1, 2),
        ),
        sliced.StructuralOccurrence(
            1,
            0,
            (LITERAL_INT, LITERAL_INT, ROLE_INT),
            (0, 1, 2),
        ),
        sliced.StructuralOccurrence(2, 0, (LITERAL_INT,), (0,)),
        sliced.StructuralOccurrence(
            2,
            1,
            (LITERAL_INT, ROLE_INT),
            (1, 2),
            (sliced.StructuralRelationConstraint(0, "ne", right_value=0),),
        ),
        sliced.StructuralOccurrence(4, 0, (ROLE_INT,), (1,)),
        sliced.StructuralOccurrence(4, 1, (LITERAL_INT,), (0,)),
        sliced.StructuralOccurrence(0, 1, (ROLE_LIST_INT,), (3,)),
    )
    return sliced.freeze_structural_factor_certificate(
        occurrences,
        (
            sliced.StructuralDomainConstraint(3, 0, (-14,)),
            sliced.StructuralDomainConstraint(6, 0, (-15,)),
        ),
        (((3, 0),), ((6, 0),)),
    )


def _clamp_sum_certificate() -> sliced.StructuralFactorCertificate:
    occurrences = (
        sliced.StructuralOccurrence(
            None,
            None,
            (LITERAL_INT, LITERAL_INT, ROLE_INT, ROLE_LIST_INT),
            (),
        ),
        sliced.StructuralOccurrence(
            0,
            0,
            (LITERAL_INT, LITERAL_INT, ROLE_INT, ROLE_LIST_INT),
            (0, 1, 2, 3),
        ),
        sliced.StructuralOccurrence(
            1,
            0,
            (LITERAL_INT, LITERAL_INT, ROLE_INT),
            (0, 1, 2),
        ),
        sliced.StructuralOccurrence(
            2,
            0,
            (LITERAL_INT, LITERAL_INT, ROLE_INT),
            (0, 1, 2),
            (sliced.StructuralRelationConstraint(0, "le", right_slot=1),),
        ),
        sliced.StructuralOccurrence(
            3,
            0,
            (LITERAL_INT, ROLE_INT),
            (0, 2),
        ),
        sliced.StructuralOccurrence(4, 0, (ROLE_INT,), (1,)),
        sliced.StructuralOccurrence(4, 1, (LITERAL_INT,), (0,)),
        sliced.StructuralOccurrence(3, 1, (LITERAL_INT,), (1,)),
        sliced.StructuralOccurrence(1, 1, (ROLE_LIST_INT,), (3,)),
    )
    return sliced.freeze_structural_factor_certificate(
        occurrences,
        (
            sliced.StructuralDomainConstraint(6, 0, (-2,)),
            sliced.StructuralDomainConstraint(7, 0, (2,)),
        ),
        (((6, 0),), ((7, 0),)),
    )


def _build(
    cases: tuple[tuple[list[int]], ...],
    literals: tuple[int, ...],
    certificate: sliced.StructuralFactorCertificate,
) -> sliced.PackedGenericTypedPromptPlanSliceV1:
    return sliced.PackedGenericTypedPromptPlanSliceV1.build(
        _signature(),
        cases,
        literals,
        9,
        certificate,
    )


def _events(
    frontier: sliced.PackedGenericTypedPromptPlanSliceV1,
) -> dict[str, int]:
    return dict(frontier.work.relation.event_counts)


def test_authorized_19_literal_map_count_slice_fits_unchanged_cap() -> None:
    certificate = _map_count_certificate()
    certificate_text = repr(certificate._asdict()).lower()
    assert all(
        forbidden not in certificate_text
        for forbidden in (
            "family_id",
            "task_id",
            "count",
            "map",
            "sum",
            "minimum",
            "maximum",
            "less-than",
        )
    )
    assert (
        "desired"
        not in inspect.signature(sliced.PackedGenericTypedPromptPlanSliceV1.build).parameters
    )

    frontier = _build(MAP_COUNT_CASES, MAP_COUNT_LITERALS_19, certificate)
    before = _events(frontier)
    receipt = frontier.plan_slice_receipt
    assert frontier.status == sliced.FRONTIER_READY
    assert frontier.sha256 == ("5a60a2ca6876d5a4c721417638d28554141f69a7cbdf2ec9fe2d3252567cd5ee")
    assert receipt is not None
    assert receipt.sha256 == ("50bf67642a3f38bd8bf51e9de81c7b8e728bbf715dd90f2c0ad3f2b72400a824")
    assert frontier.state.authorized_slice_count == 1
    assert frontier.work.relation.limit == sliced.MAX_RECIPE_RELATION_WORK == 30_000
    assert frontier.work.relation.completed_units == 25_652
    assert frontier.work.physical_predictions == 1_821
    assert frontier.work.prediction_cache_hits == 10_258
    assert frontier.work.physical_races == 4_381
    assert len(frontier.root_eclass_ids) == 45
    assert frontier.work.concrete_proof_witnesses == 0
    assert before.get("frontier:desired-value-read", 0) == 0
    assert before.get("operation:workset-materialize-authorized-plan-slice", 0) == 0
    receipt_text = repr(receipt._asdict()).lower()
    assert all(item not in receipt_text for item in ("source", "program", "witness"))

    result = frontier.select(MAP_COUNT_DESIRED)
    assert result.status == sliced.SELECTED
    assert result.sha256 == ("1f3cb7fc0b35c27049f23320355e6c30730e4d7821574236bc62577a9ba69ce7")
    assert result.selected_eclass_id == receipt.authenticated_eclass_id
    assert result.exact_proof_count == len(result.witnesses) == 1
    assert result.witnesses[0].source == (
        "def aggregate(numbers):\n    return sum((1 for item_0 in numbers if -14 < item_0 + -15))\n"
    )
    assert result.work.relation.completed_units == 26_073
    assert result.work.physical_races == 4_382
    assert _events(frontier)["operation:workset-materialize-authorized-plan-slice"] == 1


def test_authorized_clamp_slice_rejects_perturbed_desired_then_selects() -> None:
    frontier = _build(CASES, CLAMP_LITERALS, _clamp_sum_certificate())
    assert frontier.status == sliced.FRONTIER_READY

    perturbed = frontier.select((0, 0, 6, 0, 0))
    assert perturbed.status == sliced.STOP_AUTHORIZED_ECLASS_MISMATCH
    assert perturbed.exact_proof_count == 0
    assert perturbed.witnesses == ()
    assert perturbed.selected_proof_forest_sha256 is None
    assert (
        _events(frontier).get(
            "operation:workset-materialize-authorized-plan-slice",
            0,
        )
        == 0
    )

    selected = frontier.select(CLAMP_DESIRED)
    assert selected.status == sliced.SELECTED
    assert selected.exact_proof_count == 1
    assert selected.witnesses[0].source == (
        "def aggregate(numbers):\n    return sum([min(max(item_0, -2), 2) for item_0 in numbers])\n"
    )
    assert selected.work.relation.completed_units < 30_000


def test_structural_certificate_derangement_and_forgery_stop() -> None:
    certificate = _clamp_sum_certificate()
    forged = certificate._replace(sha256="0" * 64)
    with pytest.raises(RuntimeError, match="certificate authentication"):
        _build(CASES, CLAMP_LITERALS, forged)

    occurrences = list(certificate.occurrences)
    occurrences[1] = occurrences[1]._replace(parent_slot_occupancy=(0, 1, 3, 2))
    deranged = sliced.freeze_structural_factor_certificate(
        occurrences,
        certificate.domains,
        certificate.literal_slot_copy_groups,
    )
    stopped = _build(CASES, CLAMP_LITERALS, deranged)
    assert stopped.status == sliced.STOP_NO_AUTHORIZED_SLICE
    assert stopped.state.authorized_slice_count == 0
    assert stopped.plan_slice_receipt is None
    assert stopped.select(CLAMP_DESIRED).status == sliced.STOP_NO_AUTHORIZED_SLICE
