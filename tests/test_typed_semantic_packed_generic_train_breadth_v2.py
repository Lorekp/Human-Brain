from __future__ import annotations

import pytest
from experiments.r3_pilot import typed_semantic_packed_frontier_v3 as packed_v3
from experiments.r3_pilot import (
    typed_semantic_packed_generic_train_breadth_v2 as breadth,
)
from humanbrain.typed_program_types import (
    ProgramParameter,
    ProgramType,
    TypedSignature,
)

CASES = ((-9,), (-2,), (0,), (7,))
LITERALS = (-4, 0, 6)
LIST_CASES = (([],), ([-3, 0, 2],), ([5, -1],))
LIST_LITERALS = (-2, 3)


def _signature(
    *,
    entry_point: str = "solve",
    parameter_name: str = "value",
) -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        entry_point=entry_point,
        parameters=(ProgramParameter(parameter_name, integer),),
        result=integer,
    )


def _frontier() -> breadth.PackedGenericTypedTrainBreadthV2:
    return breadth.PackedGenericTypedTrainBreadthV2.build(
        _signature(),
        CASES,
        LITERALS,
        5,
    )


def _list_signature(
    *,
    entry_point: str = "transform",
    parameter_name: str = "values",
) -> TypedSignature:
    lists = ProgramType.List(ProgramType.Int())
    return TypedSignature(
        entry_point=entry_point,
        parameters=(ProgramParameter(parameter_name, lists),),
        result=lists,
    )


def _list_frontier() -> breadth.PackedGenericTypedTrainBreadthV2:
    return breadth.PackedGenericTypedTrainBreadthV2.build(
        _list_signature(),
        LIST_CASES,
        LIST_LITERALS,
        6,
    )


def test_layer1_emits_exact_typed_sources_for_three_generic_scalar_behaviors() -> None:
    offset = _frontier().select(tuple(value - 4 for (value,) in CASES))
    nonnegative = _frontier().select(tuple(max(value, 0) for (value,) in CASES))
    clamp = _frontier().select(tuple(min(max(value, -4), 6) for (value,) in CASES))

    assert offset.status == nonnegative.status == clamp.status == breadth.SELECTED
    assert offset.witnesses[0].source == ("def solve(value):\n    return value + -4\n")
    assert nonnegative.witnesses[0].source == ("def solve(value):\n    return max(value, 0)\n")
    assert clamp.witnesses[0].source == ("def solve(value):\n    return min(max(value, -4), 6)\n")
    assert offset.witnesses[0].program.signature == _signature()
    assert nonnegative.exact_proof_count == clamp.exact_proof_count == 1
    assert all(
        selection.selected_proof_forest_sha256 is not None
        for selection in (offset, nonnegative, clamp)
    )


def test_desired_permutation_cannot_change_the_desired_free_build_roots() -> None:
    treatment = _frontier()
    permuted = _frontier()
    build_events = dict(treatment.work.relation.event_counts)

    assert treatment.status == permuted.status == breadth.FRONTIER_READY
    assert treatment.sha256 == permuted.sha256
    assert treatment.root_eclass_ids == permuted.root_eclass_ids
    assert treatment.state.packed_frontier_sha256 == (permuted.state.packed_frontier_sha256)
    assert build_events.get("frontier:desired-value-read", 0) == 0
    assert build_events.get("operation:workset-materialize-selected-proof", 0) == 0
    assert treatment.work.concrete_proof_witnesses == 0

    exact = treatment.select(tuple(min(max(value, -4), 6) for (value,) in CASES))
    permuted_values = tuple(reversed(tuple(min(max(value, -4), 6) for (value,) in CASES)))
    absent = permuted.select(permuted_values)
    assert exact.status == breadth.SELECTED
    assert absent.status == breadth.STOP_NO_EXACT
    assert exact.frontier_sha256 == absent.frontier_sha256
    assert treatment.root_eclass_ids == permuted.root_eclass_ids


def test_layer1_keeps_the_v3_caps_and_has_no_direct_constructor_path() -> None:
    assert breadth.MAX_RECIPE_RELATION_WORK == packed_v3.MAX_RECIPE_RELATION_WORK == 30_000
    assert breadth.MAX_PHYSICAL_PREDICTIONS == packed_v3.MAX_PHYSICAL_PREDICTIONS == 30_000
    assert breadth.MAX_PHYSICAL_RACES == packed_v3.MAX_PHYSICAL_RACES == 80_000

    with pytest.raises(TypeError, match=r"use .*build"):
        breadth.PackedGenericTypedTrainBreadthV2()

    string = ProgramType.String()
    invalid = TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("text", string),),
        result=string,
    )
    with pytest.raises(ValueError, match="Int -> Int"):
        breadth.PackedGenericTypedTrainBreadthV2.build(
            invalid,
            (("x",),),
            (0,),
            3,
        )


def test_recomputed_signature_hijack_fails_before_desired_or_physical_work() -> None:
    frontier = _frontier()
    original_roots = frontier.root_eclass_ids
    work_before = frontier.work

    hijacked = _signature(entry_point="hijacked", parameter_name="other")
    state = frontier.state._replace(signature=hijacked, sha256="")
    state = state._replace(sha256=breadth._digest("typed-frontier", breadth._state_payload(state)))
    receipt = frontier._construction_receipt._replace(
        state=state,
        signature=hijacked,
        sha256="",
    )
    receipt = receipt._replace(
        sha256=breadth._digest(
            "construction-receipt",
            breadth._receipt_payload(receipt),
        )
    )
    frontier._state = state
    frontier._construction_receipt = receipt

    class UnreadDesired:
        reads = 0

        def __len__(self) -> int:
            self.reads += 1
            return len(CASES)

        def __getitem__(self, index: int) -> int:
            self.reads += 1
            return index

    desired = UnreadDesired()
    assert frontier.root_eclass_ids == original_roots
    assert frontier.sha256 == state.sha256
    assert receipt.sha256 == breadth._digest(
        "construction-receipt",
        breadth._receipt_payload(receipt),
    )

    with pytest.raises(RuntimeError, match="typed packed frontier authentication failed"):
        frontier.select(desired)  # type: ignore[arg-type]

    assert desired.reads == 0
    assert frontier.work == work_before
    assert frontier.work.concrete_proof_witnesses == 0


def test_layer2_selects_map_absolute_and_map_add_with_exact_typed_sources() -> None:
    absolute_frontier = _list_frontier()
    add_frontier = _list_frontier()
    absolute_build = absolute_frontier.work
    add_build = add_frontier.work

    assert absolute_frontier.status == add_frontier.status == breadth.FRONTIER_READY
    assert absolute_build.concrete_proof_witnesses == 0
    assert absolute_build.materialized_witnesses == 0
    assert add_build.concrete_proof_witnesses == 0
    absolute = absolute_frontier.select([[abs(value) for value in row[0]] for row in LIST_CASES])
    added = add_frontier.select([[value + 3 for value in row[0]] for row in LIST_CASES])

    assert absolute.status == added.status == breadth.SELECTED
    assert absolute.exact_proof_count == added.exact_proof_count == 1
    assert absolute.witnesses[0].source == (
        "def transform(values):\n    return [abs(item_0) for item_0 in values]\n"
    )
    assert added.witnesses[0].source == (
        "def transform(values):\n    return [item_0 + 3 for item_0 in values]\n"
    )
    assert absolute.witnesses[0].program.signature == _list_signature()
    assert absolute.selected_proof_forest_sha256 is not None
    assert added.selected_proof_forest_sha256 is not None


def test_layer2_desired_permutation_preserves_roots_and_empty_list_shape() -> None:
    treatment = _list_frontier()
    permuted = _list_frontier()
    build_events = dict(treatment.work.relation.event_counts)

    assert treatment.sha256 == permuted.sha256
    assert treatment.root_eclass_ids == permuted.root_eclass_ids
    assert treatment.state.packed_frontier_sha256 == (permuted.state.packed_frontier_sha256)
    assert treatment.state.case_arguments == (
        ((),),
        ((-3, 0, 2),),
        ((5, -1),),
    )
    assert build_events.get("frontier:desired-value-read", 0) == 0
    assert build_events.get("operation:workset-materialize-selected-proof", 0) == 0
    assert treatment.work.concrete_proof_witnesses == 0

    desired = [[value - 2 for value in row[0]] for row in LIST_CASES]
    exact = treatment.select(desired)
    absent = permuted.select(tuple(reversed(desired)))
    assert exact.status == breadth.SELECTED
    assert absent.status == breadth.STOP_NO_EXACT
    assert exact.frontier_sha256 == absent.frontier_sha256


def test_layer2_performs_zero_scalar_route_work_and_keeps_layer1_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scalar = _frontier()
    assert scalar.sha256 == ("3f25fa16d3b9811edf0bcd7f693dcc9d4745e846467c95b163a1664f55160af5")
    assert scalar.work.sha256 == (
        "a3d7a66f354c3b84788b46f15be54240c7a0538d808b72c3edd133226e82f9c3"
    )
    assert (
        scalar.work.relation.completed_units,
        scalar.work.physical_predictions,
        scalar.work.physical_races,
    ) == (1_391, 41, 128)

    def reject_scalar_build(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("List route touched the scalar frontier")

    monkeypatch.setattr(
        packed_v3.PackedScalarFrontier,
        "build",
        reject_scalar_build,
    )
    list_frontier = _list_frontier()
    assert list_frontier.status == breadth.FRONTIER_READY
    assert list_frontier.work.concrete_proof_witnesses == 0
