from __future__ import annotations

import inspect
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v3 as breadth_v3
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v4 as breadth_v4
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
AMBIGUOUS_CASES = (([],), ([0],), ([0, 0],), ([0, 0, 0],))

MAP_COUNT_CASES = (
    ([],),
    ([0, 1, 2, 3],),
    ([1, 2, 10],),
    ([0, 0, 5],),
    ([2, 2, 2],),
)
MAP_COUNT_LITERALS = (-15, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12)
MAP_COUNT_DESIRED = (0, 2, 2, 1, 3)

EXPECTED_V3_SOURCE_SHA256 = "e8a4c3f846afc96e321a484b0b2c230bf5123851f390c2615bc9775655960ba6"
EXPECTED_V3_TEST_SHA256 = "f8d11131b9e0849e765d6e5fbb46491c98896197a7154bc8fb97d139e8318bf4"


def _signature(
    parameter_type: ProgramType | None = None,
    result_type: ProgramType | None = None,
) -> TypedSignature:
    integer = ProgramType.Int()
    parameter = ProgramType.List(integer) if parameter_type is None else parameter_type
    result = integer if result_type is None else result_type
    return TypedSignature(
        "aggregate",
        (ProgramParameter("numbers", parameter),),
        result,
    )


def _build(
    cases: tuple[tuple[list[int]], ...] = CASES,
    literals: tuple[int, ...] = LITERALS,
) -> breadth_v4.PackedGenericTypedTrainBreadthV4:
    return breadth_v4.PackedGenericTypedTrainBreadthV4.build(
        _signature(),
        cases,
        literals,
        9,
    )


def _events(
    frontier: breadth_v4.PackedGenericTypedTrainBreadthV4,
) -> dict[str, int]:
    return dict(frontier.work.relation.event_counts)


def test_factorized_selected_proof_selects_map_count_and_generic_clamp_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("cross-type V4 entered the frozen V3 public build")

    monkeypatch.setattr(
        breadth_v3.PackedGenericTypedTrainBreadthV3,
        "build",
        forbidden,
    )
    assert (
        "desired"
        not in inspect.signature(breadth_v4.PackedGenericTypedTrainBreadthV4.build).parameters
    )

    map_count_frontier = _build(MAP_COUNT_CASES, MAP_COUNT_LITERALS)
    build_events = _events(map_count_frontier)
    assert map_count_frontier.status == breadth_v4.FRONTIER_READY
    assert map_count_frontier.sha256 == (
        "3233564ce33c881734298d7b825bfaad4e4349ce2037989c29e64f162b183c89"
    )
    assert map_count_frontier.work.relation.completed_units == 16_062
    assert map_count_frontier.work.physical_predictions == 1_247
    assert map_count_frontier.work.prediction_cache_hits == 6_606
    assert map_count_frontier.work.physical_races == 3_057
    assert map_count_frontier.work.concrete_proof_witnesses == 0
    assert map_count_frontier.work.materialized_witnesses == 0
    assert len(map_count_frontier.root_eclass_ids) == 41
    assert build_events.get("frontier:desired-value-read", 0) == 0
    assert build_events.get("frontier:materialize-witness", 0) == 0
    assert (
        build_events.get(
            "operation:workset-materialize-selected-lazy-proof-plan",
            0,
        )
        == 0
    )

    map_count = map_count_frontier.select(MAP_COUNT_DESIRED)
    selected_events = _events(map_count_frontier)
    assert map_count.status == breadth_v4.SELECTED
    assert map_count.sha256 == ("4d70f0fdb200a96f3813d8201eaf3cc0e44f7a4fb171db6c4cf735de52ab3075")
    assert map_count.work.relation.completed_units == 16_436
    assert map_count.work.physical_races == 3_058
    assert map_count.exact_proof_count == len(map_count.witnesses) == 1
    assert map_count.witnesses[0].source == (
        "def aggregate(numbers):\n    return sum((1 for item_0 in numbers if -14 < item_0 + -15))\n"
    )
    assert selected_events["workset:selected-factor-position"] == 7
    assert selected_events["workset:selected-relation-receipt-registry-open"] == 1
    assert selected_events["workset:selected-relation-receipt-authenticate"] == 18

    clamp_frontier = _build()
    assert clamp_frontier.sha256 == (
        "2a3e5d737340ac97fb1c6f77474906f50b4bf48a1d0e241e3cc1b928af59fb9a"
    )
    assert clamp_frontier.work.relation.completed_units == 3_290
    clamp = clamp_frontier.select(CLAMP_SUM_DESIRED)
    assert clamp.status == breadth_v4.SELECTED
    assert clamp.work.relation.completed_units == 3_724
    assert clamp.exact_proof_count == 1
    assert clamp.witnesses[0].source == (
        "def aggregate(numbers):\n    return sum([min(max(item_0, -2), 2) for item_0 in numbers])\n"
    )
    assert _events(clamp_frontier)["workset:selected-factor-position"] > 0


def test_all_zero_ambiguity_and_impossible_target_remain_explicit() -> None:
    ambiguous_frontier = _build(AMBIGUOUS_CASES)
    ambiguous = ambiguous_frontier.select((0, 0, 0, 0))
    assert ambiguous.status == breadth_v4.STOP_AMBIGUOUS
    assert ambiguous.sha256 == ("422af814d8a032b58b679fe43538072c1911b2cfa5863f2150e05586172ccc0d")
    assert ambiguous.work.relation.completed_units == 4_071
    assert ambiguous.work.physical_races == 137
    assert ambiguous.exact_proof_count == 18
    assert len(ambiguous.witnesses) == 2

    no_exact_frontier = _build()
    no_exact = no_exact_frontier.select((0, 17, 17, 17, 17))
    assert no_exact.status == breadth_v4.STOP_NO_EXACT
    assert no_exact.sha256 == ("f01937b5e6bb04253675563ff8ea82992b65abbe14f55711ed931b8580d1b3b3")
    assert no_exact.work.relation.completed_units == 3_303
    assert no_exact.exact_proof_count == 0
    assert no_exact.witnesses == ()
    assert no_exact.selected_proof_forest_sha256 is None
    assert (
        _events(no_exact_frontier).get(
            "operation:workset-materialize-selected-lazy-proof-plan",
            0,
        )
        == 0
    )


def test_cap_is_atomic_and_v3_bytes_and_caps_stay_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(breadth_v4, "MAX_RECIPE_RELATION_WORK", 3_289)
    stopped = _build()
    audit = stopped.work.relation

    assert stopped.status == breadth_v4.STOP_WORK_BOUND
    assert stopped.root_eclass_ids == ()
    assert audit.limit == 3_289
    assert audit.completed_units == 3_289
    assert audit.attempted_units == 3_290
    assert audit.denied_event == "frontier:frontier-seal"
    assert stopped.work.bound_name == "recipe-relation-work"
    assert stopped.work.bound_attempted == 3_290
    assert stopped.work.concrete_proof_witnesses == 0
    assert stopped.work.materialized_witnesses == 0

    repository = Path(__file__).resolve().parents[1]
    v3_source = Path(breadth_v3.__file__).resolve()
    v3_test = repository / "tests/test_typed_semantic_packed_generic_train_breadth_v3.py"
    source = Path(breadth_v4.__file__).read_text(encoding="utf-8")
    assert sha256(v3_source.read_bytes()).hexdigest() == EXPECTED_V3_SOURCE_SHA256
    assert sha256(v3_test.read_bytes()).hexdigest() == EXPECTED_V3_TEST_SHA256
    assert breadth_v4.MAX_PHYSICAL_PREDICTIONS == breadth_v3.MAX_PHYSICAL_PREDICTIONS
    assert breadth_v4.MAX_PHYSICAL_RACES == breadth_v3.MAX_PHYSICAL_RACES
    assert "public_typed_semantic_synthesizer" not in source
    assert "canonical_train" not in source
    assert "task_id" not in source
    assert 'production == "count"' not in source
    assert 'family_id == "' not in source


def test_scalar_list_and_string_routes_delegate_without_cross_type_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    delegated: list[TypedSignature] = []

    def record(
        _cls: type[object],
        signature: TypedSignature,
        _cases: object,
        _literals: object,
        _max_cost: object,
    ) -> object:
        delegated.append(signature)
        return sentinel

    monkeypatch.setattr(
        breadth_v3.PackedGenericTypedTrainBreadthV3,
        "build",
        classmethod(record),
    )
    integer = ProgramType.Int()
    list_integer = ProgramType.List(integer)
    string = ProgramType.String()
    signatures = (
        _signature(integer, integer),
        _signature(list_integer, list_integer),
        _signature(string, string),
    )
    for signature in signatures:
        assert (
            breadth_v4.PackedGenericTypedTrainBreadthV4.build(
                signature,
                (),
                (),
                1,
            )
            is sentinel
        )
    assert tuple(delegated) == signatures
