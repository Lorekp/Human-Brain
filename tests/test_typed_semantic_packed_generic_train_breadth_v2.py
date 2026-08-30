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
