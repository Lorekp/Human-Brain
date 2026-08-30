from __future__ import annotations

import pytest
from experiments.r3_pilot import typed_semantic_packed_frontier_v3 as v3
from experiments.r3_pilot import typed_semantic_packed_frontier_v4 as frontier_module

STRESS_ARGUMENTS = (-13, -7, -2, 3, 12)
STRESS_LITERALS = range(-9, 10)
STRESS_CLAMP = tuple(min(max(value, -4), 6) for value in STRESS_ARGUMENTS)
STRESS_LOW_CLAMP = tuple(min(max(value, -9), -8) for value in STRESS_ARGUMENTS)


def _events(frontier: frontier_module.PackedScalarFrontier) -> dict[str, int]:
    return dict(frontier.work.relation.event_counts)


def test_cost_five_preserves_v3_physics_and_selected_program() -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=STRESS_ARGUMENTS,
        literals=STRESS_LITERALS,
        max_ast_cost=5,
    )
    before = _events(frontier)

    assert frontier.status == frontier_module.FRONTIER_READY
    assert frontier.sha256 == ("7a2eec2133651d7fdff0a40509c191276e79cf583bb90cb44380a8a2f205e4ee")
    assert frontier.state.build_work_sha256 == (
        "1ae6f39ebcfd8c2b9b5aa789360dc3da37e6a943a72a81db3f5a82227b0cc90a"
    )
    assert frontier.work.relation.event_sha256 == (
        "ed0019ca5ce0dd37c303d3bd82dc89f62d76844cfeda446e0a3f088eeb69dc54"
    )
    assert frontier.work.relation.completed_units == 5_676
    assert frontier.work.physical_predictions == 422
    assert frontier.work.prediction_cache_hits == 723
    assert frontier.work.physical_races == 1_489
    assert len(frontier.families) == 7
    assert len(frontier.enodes) == 249
    assert len(frontier.eclasses) == 230
    assert frontier.state.frozen_workset is not None
    assert frontier.state.frozen_workset.plan_node_count == 235
    assert frontier.state.frozen_workset.plan_edge_count == 229
    assert frontier.state.frozen_workset.plan_recipe_count == 230
    assert frontier.state.frozen_workset.topology_node_count == 21
    assert frontier.work.relation.completed_units < 29_703
    assert frontier.work.concrete_proof_witnesses == 0
    assert frontier.work.materialized_witnesses == 0
    assert before.get("frontier:desired-value-read", 0) == 0
    assert before.get("operation:proof-forest-build", 0) == 0
    assert before.get("operation:family-slice", 0) == 0

    selection = frontier.select(STRESS_CLAMP)

    assert selection.status == frontier_module.SELECTED
    assert selection.sha256 == ("389a8384325e90287f4f675c14d224a41d0850bc8dc7a7948c9b68c3abd93050")
    assert selection.work.sha256 == (
        "ab17498e59aa0052eb9d6d02d869e7a16364d9f2d36022c2bdff789d35acf03e"
    )
    assert selection.work.relation.event_sha256 == (
        "f5250a88b12d5ab5668461acca4e4d48cdf0e87f8a40069088959ca955157d3a"
    )
    assert selection.work.relation.completed_units == 6_015
    assert selection.work.physical_predictions == 422
    assert selection.work.prediction_cache_hits == 723
    assert selection.work.physical_races == 1_490
    assert selection.exact_proof_count == 1
    assert tuple(item.source for item in selection.witnesses) == ("min(max(x, -4), 6)",)
    assert selection.selected_proof_forest_sha256 == (
        "b05a2a5334ae4e28fad50b8e3b914d120002128998820792c70a95bc6c490d13"
    )
    assert selection.race is not None
    assert selection.race.winner_key in frontier.state.root_eclass_ids
    assert v3.MAX_RECIPE_RELATION_WORK == frontier_module.MAX_RECIPE_RELATION_WORK


def test_cost_seven_builds_and_selects_under_the_unchanged_cap() -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=STRESS_ARGUMENTS,
        literals=STRESS_LITERALS,
        max_ast_cost=7,
    )

    assert frontier.status == frontier_module.FRONTIER_READY
    assert frontier.sha256 == ("94886ada2d54b2c3bdf4d5210e3a240112f1c35ee8d3125ca4f9be79802f6ea4")
    assert frontier.state.build_work_sha256 == (
        "778225d404c1de68c384371ae3357f741be5d5db5a6bfafdeb506b2b663fe385"
    )
    assert frontier.work.relation.event_sha256 == (
        "80af230861b52370ae10df1098b8ad049735bcd3a7517a9f389044932314033e"
    )
    assert frontier.work.relation.completed_units == 22_456
    assert frontier.state.maximum_completed_ast_cost == 7
    assert frontier.work.relation.completed_units < 30_000
    assert frontier.work.concrete_proof_witnesses == 0
    assert frontier.work.materialized_witnesses == 0
    assert frontier.work.physical_predictions == 726
    assert frontier.work.prediction_cache_hits == 18_469
    assert frontier.work.physical_races == 2_705
    assert len(frontier.families) == 8
    assert len(frontier.enodes) == 439
    assert len(frontier.eclasses) == 230
    assert frontier.state.frozen_workset is not None
    assert frontier.state.frozen_workset.plan_node_count == 426
    assert frontier.state.frozen_workset.plan_edge_count == 419
    assert frontier.state.frozen_workset.plan_recipe_count == 3_840
    assert frontier.state.frozen_workset.topology_node_count == 21
    assert _events(frontier).get("frontier:desired-value-read", 0) == 0
    assert _events(frontier).get("operation:proof-forest-build", 0) == 0

    selection = frontier.select(STRESS_LOW_CLAMP)

    assert selection.status == frontier_module.STOP_AMBIGUOUS
    assert selection.sha256 == ("65e508b947c066aa225794efead9c3097749f6fceff4bb3fdec0ec3a93a56c43")
    assert selection.work.sha256 == (
        "e1b1b47ee7f91e3ca4e4f6c6fc4963e5361a9dcfdb997db48355fe297a947c34"
    )
    assert selection.work.relation.event_sha256 == (
        "845f8ff5a83b04fd996d4c4b24ed089baca878d1c44da3676d8ee35e00b6a6e6"
    )
    assert selection.work.relation.completed_units == 23_075
    assert selection.work.relation.completed_units < 30_000
    assert selection.exact_proof_count == 2
    assert {item.source for item in selection.witnesses} == {
        "min(max(x, -9), -8)",
        "max(min(max(x, -9), -8), -9)",
    }
    assert selection.selected_proof_forest_sha256 == (
        "08df5e6e5e5d91a2dc1aa801b87aad9d4699f62c980cf3d04825583ba9ff3178"
    )


def test_cost_seven_exact_total_minus_one_stops_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(frontier_module, "MAX_RECIPE_RELATION_WORK", 22_455)
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=STRESS_ARGUMENTS,
        literals=STRESS_LITERALS,
        max_ast_cost=7,
    )

    assert frontier.status == frontier_module.STOP_WORK_BOUND
    assert frontier.work.relation.completed_units == 22_455
    assert frontier.work.relation.attempted_units == 22_456
    assert frontier.work.relation.denied_event == "frontier:frontier-seal"
    assert frontier.state.maximum_completed_ast_cost == 7
    assert frontier.state.frozen_workset is None
    assert frontier.families == frontier.enodes == frontier.eclasses == ()
    assert frontier.work.physical_predictions == 726
    assert frontier.work.prediction_cache_hits == 18_469
    assert frontier.work.physical_races == 2_705
    assert frontier.work.concrete_proof_witnesses == 0
    assert _events(frontier).get("frontier:desired-value-read", 0) == 0
    assert _events(frontier).get("operation:proof-forest-build", 0) == 0
