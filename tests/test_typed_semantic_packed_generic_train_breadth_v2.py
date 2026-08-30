from __future__ import annotations

import pytest
from experiments.r3_pilot import public_typed_operator_successor as operator_successor
from experiments.r3_pilot import typed_semantic_packed_frontier_v3 as packed_v3
from experiments.r3_pilot import typed_semantic_packed_frontier_v4 as packed_v4
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
STRING_CASES = (
    ("",),
    ("A1b!eZ9",),
    ("lower UPPER 42?",),
    ("rhythms",),
)
VOWELS = "AEIOUaeiou"
LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"


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


def _string_signature(
    *,
    entry_point: str = "clean_text",
    parameter_name: str = "payload",
) -> TypedSignature:
    string = ProgramType.String()
    return TypedSignature(
        entry_point=entry_point,
        parameters=(ProgramParameter(parameter_name, string),),
        result=string,
    )


def _string_frontier() -> breadth.PackedGenericTypedTrainBreadthV2:
    return breadth.PackedGenericTypedTrainBreadthV2.build(
        _string_signature(),
        STRING_CASES,
        (),
        7,
    )


def _filtered(characters: str) -> tuple[str, ...]:
    return tuple(
        "".join(character for character in row[0] if character in characters)
        for row in STRING_CASES
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
    assert (
        breadth.MAX_RECIPE_RELATION_WORK
        == packed_v3.MAX_RECIPE_RELATION_WORK
        == packed_v4.MAX_RECIPE_RELATION_WORK
        == 30_000
    )
    assert breadth.MAX_PHYSICAL_PREDICTIONS == packed_v3.MAX_PHYSICAL_PREDICTIONS == 30_000
    assert breadth.MAX_PHYSICAL_RACES == packed_v3.MAX_PHYSICAL_RACES == 80_000

    with pytest.raises(TypeError, match=r"use .*build"):
        breadth.PackedGenericTypedTrainBreadthV2()

    boolean = ProgramType.Bool()
    invalid = TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("flag", boolean),),
        result=boolean,
    )
    with pytest.raises(ValueError, match="requires Int -> Int"):
        breadth.PackedGenericTypedTrainBreadthV2.build(
            invalid,
            ((True,),),
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


def test_layer3_selects_filter_reverse_and_composition_with_exact_sources() -> None:
    frontier = _string_frontier()
    build_work = frontier.work

    reverse = frontier.select(tuple(row[0][::-1] for row in STRING_CASES))
    vowels = frontier.select(_filtered(VOWELS))
    lowercase = frontier.select(_filtered(LOWERCASE))
    digits = frontier.select(_filtered(DIGITS))
    reverse_filtered = frontier.select(tuple(value[::-1] for value in _filtered(LOWERCASE)))

    assert frontier.status == breadth.FRONTIER_READY
    assert build_work.concrete_proof_witnesses == 0
    assert build_work.materialized_witnesses == 0
    assert all(
        selection.status == breadth.SELECTED
        and selection.exact_proof_count == 1
        and len(selection.witnesses) == 1
        and selection.selected_proof_forest_sha256 is not None
        for selection in (reverse, vowels, lowercase, digits, reverse_filtered)
    )
    assert reverse.witnesses[0].source == ("def clean_text(payload):\n    return payload[::-1]\n")
    assert vowels.witnesses[0].source == (
        "def clean_text(payload):\n"
        "    return ''.join((char_0 for char_0 in payload if char_0 in 'AEIOUaeiou'))\n"
    )
    assert lowercase.witnesses[0].source == (
        "def clean_text(payload):\n"
        "    return ''.join((char_0 for char_0 in payload if char_0 in "
        "'abcdefghijklmnopqrstuvwxyz'))\n"
    )
    assert digits.witnesses[0].source == (
        "def clean_text(payload):\n"
        "    return ''.join((char_0 for char_0 in payload if char_0 in "
        "'0123456789'))\n"
    )
    assert reverse_filtered.witnesses[0].source == (
        "def clean_text(payload):\n"
        "    return ''.join((char_0 for char_0 in payload if char_0 in "
        "'abcdefghijklmnopqrstuvwxyz'))[::-1]\n"
    )
    assert vowels.witnesses[0].program.signature == _string_signature()
    assert vowels.witnesses[0].program.tokens == (
        "filter",
        "bind",
        "contains",
        "english-vowels",
        "variable-0",
        "variable-0",
    )


def test_layer3_build_is_desired_free_and_permutation_invariant() -> None:
    treatment = _string_frontier()
    permuted = _string_frontier()
    build_events = dict(treatment.work.relation.event_counts)

    assert treatment.sha256 == permuted.sha256
    assert treatment.root_eclass_ids == permuted.root_eclass_ids
    assert treatment.state.case_arguments == STRING_CASES
    assert treatment.work.relation.completed_units == 1_151
    assert treatment.work.physical_predictions == 182
    assert treatment.work.physical_races == 399
    assert build_events.get("frontier:desired-value-read", 0) == 0
    assert build_events.get("operation:materialize-selected-lazy-proof-plan", 0) == 0
    assert treatment.work.concrete_proof_witnesses == 0

    desired = tuple(value[::-1] for value in _filtered(LOWERCASE))
    exact = treatment.select(desired)
    absent = permuted.select(tuple(reversed(desired)))
    assert exact.status == breadth.SELECTED
    assert absent.status == breadth.STOP_NO_EXACT
    assert exact.frontier_sha256 == absent.frontier_sha256


def test_layer3_performs_zero_scalar_or_list_route_work_and_preserves_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scalar = _frontier()
    lists = _list_frontier()
    assert scalar.sha256 == "3f25fa16d3b9811edf0bcd7f693dcc9d4745e846467c95b163a1664f55160af5"
    assert lists.sha256 == "5130db5783ad6599b628ff75febce1f51835535d1cc956a52a0f1c2c2b34d932"

    def reject_old_route(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("String route touched a scalar/List frontier")

    monkeypatch.setattr(packed_v3.PackedScalarFrontier, "build", reject_old_route)
    monkeypatch.setattr(breadth._PackedListMapFrontier, "build", reject_old_route)
    string_frontier = _string_frontier()
    assert string_frontier.status == breadth.FRONTIER_READY
    assert string_frontier.work.concrete_proof_witnesses == 0


def test_layer3_keeps_ambiguity_and_catalog_unknown_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ambiguous_cases = (("!!!",), ("---",), (" ",))
    ambiguous = breadth.PackedGenericTypedTrainBreadthV2.build(
        _string_signature(),
        ambiguous_cases,
        (),
        7,
    ).select(("", "", ""))
    assert ambiguous.status == breadth.STOP_AMBIGUOUS
    assert ambiguous.exact_proof_count == 6
    assert len(ambiguous.witnesses) == 2

    original = operator_successor.FrozenTypedOperatorSuccessor._ascii_class

    def catalog_off(
        operator: operator_successor.FrozenTypedOperatorSuccessor,
        label: str,
        *,
        enabled: bool,
    ) -> tuple[
        operator_successor._AsciiClassEncoding,
        tuple[operator_successor.OperatorRace, ...],
    ]:
        assert enabled
        return original(operator, label, enabled=False)

    monkeypatch.setattr(
        operator_successor.FrozenTypedOperatorSuccessor,
        "_ascii_class",
        catalog_off,
    )
    unknown = breadth.PackedGenericTypedTrainBreadthV2.build(
        _string_signature(),
        (("Ab1",), ("xY2",)),
        (),
        7,
    )
    assert unknown.status == breadth.FRONTIER_READY
    assert all(
        output.supported
        for eclass in unknown._packed.state.root_eclasses
        for output in eclass.physical_outputs
    )
    assert any(
        not output.supported
        for eclass in unknown._packed.state.eclasses
        for output in eclass.physical_outputs
    )
    no_unknown_winner = unknown.select(("A", "x"))
    assert no_unknown_winner.status == breadth.STOP_NO_EXACT
    assert no_unknown_winner.exact_proof_count == 0
    assert no_unknown_winner.work.concrete_proof_witnesses == 0


def test_layer3_exact_total_minus_one_stops_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _string_frontier()
    assert baseline.work.relation.completed_units == 1_151

    monkeypatch.setattr(breadth, "MAX_RECIPE_RELATION_WORK", 1_150)
    stopped = _string_frontier()
    assert stopped.status == breadth.STOP_WORK_BOUND
    assert stopped.root_eclass_ids == ()
    assert stopped.work.relation.completed_units == 1_150
    assert stopped.work.relation.attempted_units == 1_151
    assert stopped.work.bound_name == "recipe-relation-work"
    assert stopped.work.bound_attempted == 1_151
    assert stopped.work.concrete_proof_witnesses == 0
