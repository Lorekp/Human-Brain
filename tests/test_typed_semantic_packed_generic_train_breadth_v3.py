from __future__ import annotations

import inspect
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.r3_pilot import typed_semantic_packed_frontier_v4 as packed_v4
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v2 as breadth_v2
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v3 as breadth_v3
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

CASES = (
    ([],),
    ([-6, -2, 0, 8],),
    ([1, 2, 3],),
    ([-10, 10],),
    ([-1, 0, 1],),
)
LITERALS = (-2, 1, 2)
CLAMP_SUM_DESIRED = (0, -2, 5, 0, 0)
ADD_COUNT_DESIRED = (0, 2, 3, 1, 2)
AMBIGUOUS_CASES = (([],), ([0],), ([0, 0],), ([0, 0, 0],))

EXPECTED_V2_SOURCE_SHA256 = "4c6035992a773a18c848f57c362e0dea24f28b7e4c0dcb244553bb1ca98141de"
EXPECTED_V2_TEST_SHA256 = "02d54c1b1b2ba0871ea8c751d06a8b14298a506c5e3e39ea1b933db072473f9c"


def _signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "aggregate",
        (ProgramParameter("numbers", ProgramType.List(integer)),),
        integer,
    )


def _build(
    cases: tuple[tuple[list[int]], ...] = CASES,
) -> breadth_v3.PackedGenericTypedTrainBreadthV3:
    return breadth_v3.PackedGenericTypedTrainBreadthV3.build(
        _signature(),
        cases,
        LITERALS,
        9,
    )


def _events(
    frontier: breadth_v3.PackedGenericTypedTrainBreadthV3,
) -> dict[str, int]:
    return dict(frontier.work.relation.event_counts)


def test_cross_type_v4_build_is_desired_and_proof_free_and_selects_both_compositions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("List[Int] -> Int entered a frozen V2 route")

    monkeypatch.setattr(
        breadth_v2.PackedGenericTypedTrainBreadthV2,
        "build",
        forbidden,
    )
    assert (
        "desired"
        not in inspect.signature(breadth_v3.PackedGenericTypedTrainBreadthV3.build).parameters
    )

    clamp_frontier = _build()
    build_events = _events(clamp_frontier)
    assert clamp_frontier.status == breadth_v3.FRONTIER_READY
    assert type(clamp_frontier._route._workset) is packed_v4._SemanticPartitionWorkset
    assert clamp_frontier._route._workset.session.audit == clamp_frontier.work.relation
    assert clamp_frontier.work.relation.limit == 30_000
    assert clamp_frontier.work.relation.completed_units == 3_290
    assert clamp_frontier.work.relation.attempted_units == 3_290
    assert clamp_frontier.work.physical_predictions == 246
    assert clamp_frontier.work.prediction_cache_hits == 624
    assert clamp_frontier.work.physical_races == 660
    assert clamp_frontier.work.concrete_proof_witnesses == 0
    assert clamp_frontier.work.materialized_witnesses == 0
    assert len(clamp_frontier.root_eclass_ids) == 19
    assert clamp_frontier._route.state.frozen_workset is not None
    assert clamp_frontier._route.state.frozen_workset.plan_node_count == 95
    assert clamp_frontier._route.state.frozen_workset.plan_edge_count == 82
    assert clamp_frontier._route.state.frozen_workset.plan_recipe_count == 89
    assert build_events.get("frontier:desired-value-read", 0) == 0
    assert (
        build_events.get(
            "operation:workset-materialize-selected-lazy-proof-plan",
            0,
        )
        == 0
    )
    assert build_events.get("frontier:materialize-witness", 0) == 0

    roots_before = clamp_frontier.root_eclass_ids
    clamp = clamp_frontier.select(CLAMP_SUM_DESIRED)
    assert clamp_frontier.root_eclass_ids == roots_before
    assert clamp.status == breadth_v3.SELECTED
    assert clamp.exact_proof_count == 1
    assert len(clamp.witnesses) == 1
    assert clamp.witnesses[0].source == (
        "def aggregate(numbers):\n    return sum([min(max(item_0, -2), 2) for item_0 in numbers])\n"
    )
    assert clamp.work.relation.completed_units == 3_888
    assert clamp.work.physical_races == 661
    assert clamp.work.concrete_proof_witnesses == 1
    assert _events(clamp_frontier)["operation:workset-materialize-selected-lazy-proof-plan"] == 1

    add_frontier = _build()
    assert add_frontier.root_eclass_ids == roots_before
    assert add_frontier.state.route_build_work_sha256 == (
        clamp_frontier.state.route_build_work_sha256
    )
    shifted = add_frontier.select(ADD_COUNT_DESIRED)
    assert shifted.status == breadth_v3.SELECTED
    assert shifted.exact_proof_count == 1
    assert len(shifted.witnesses) == 1
    assert shifted.witnesses[0].source == (
        "def aggregate(numbers):\n    return sum((1 for item_0 in numbers if 1 < item_0 + 2))\n"
    )
    assert shifted.work.relation.completed_units == 3_784
    assert shifted.work.physical_races == 661
    assert shifted.work.concrete_proof_witnesses == 1


def test_cross_type_route_keeps_ambiguity_and_no_exact_explicit() -> None:
    ambiguous_frontier = _build(AMBIGUOUS_CASES)
    build_events = _events(ambiguous_frontier)
    ambiguous = ambiguous_frontier.select((0, 0, 0, 0))
    assert ambiguous.status == breadth_v3.STOP_AMBIGUOUS
    assert ambiguous.exact_proof_count == 18
    assert len(ambiguous.witnesses) == 2
    assert ambiguous.work.concrete_proof_witnesses == 2
    assert build_events.get("frontier:desired-value-read", 0) == 0

    no_exact_frontier = _build()
    no_exact = no_exact_frontier.select((0, 17, 17, 17, 17))
    assert no_exact.status == breadth_v3.STOP_NO_EXACT
    assert no_exact.exact_proof_count == 0
    assert no_exact.witnesses == ()
    assert no_exact.selected_proof_forest_sha256 is None
    assert no_exact.work.concrete_proof_witnesses == 0
    assert (
        _events(no_exact_frontier).get(
            "operation:workset-materialize-selected-lazy-proof-plan",
            0,
        )
        == 0
    )


def test_cross_type_cap_stops_atomically_and_v2_bytes_stay_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(breadth_v3, "MAX_RECIPE_RELATION_WORK", 3_289)
    stopped = _build()
    audit = stopped.work.relation
    assert stopped.status == breadth_v3.STOP_WORK_BOUND
    assert stopped.root_eclass_ids == ()
    assert audit.limit == 3_289
    assert audit.completed_units == 3_289
    assert audit.attempted_units == 3_290
    assert audit.denied_event == "frontier:frontier-seal"
    assert stopped.work.bound_name == "recipe-relation-work"
    assert stopped.work.bound_attempted == 3_290
    assert stopped.work.concrete_proof_witnesses == 0
    assert stopped.work.materialized_witnesses == 0
    stopped_selection = stopped.select(CLAMP_SUM_DESIRED)
    assert stopped_selection.status == breadth_v3.STOP_WORK_BOUND
    assert stopped_selection.exact_proof_count == 0
    assert stopped_selection.witnesses == ()

    repository = Path(__file__).resolve().parents[1]
    v2_source = Path(breadth_v2.__file__).resolve()
    v2_test = repository / "tests" / "test_typed_semantic_packed_generic_train_breadth_v2.py"
    assert sha256(v2_source.read_bytes()).hexdigest() == EXPECTED_V2_SOURCE_SHA256
    assert sha256(v2_test.read_bytes()).hexdigest() == EXPECTED_V2_TEST_SHA256
    source = Path(breadth_v3.__file__).read_text(encoding="utf-8")
    assert "public_typed_semantic_synthesizer" not in source
    assert "TypedSemanticVersionSpace" not in source
    assert "task_id" not in source
    assert "30_001" not in source
