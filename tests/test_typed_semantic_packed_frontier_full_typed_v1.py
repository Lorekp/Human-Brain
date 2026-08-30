from __future__ import annotations

import inspect
from collections import Counter
from types import MappingProxyType

import pytest
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_frontier_full_typed_v1 as full
from experiments.r3_pilot import typed_semantic_packed_frontier_v2 as scalar_v2
from experiments.r3_pilot import typed_semantic_packed_frontier_v3 as scalar_v3
from experiments.r3_pilot import typed_semantic_version_space_v2 as typed_v2
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

CASES = (
    ([-2, 1, 4],),
    ([2, 3],),
    ([-1, 2, 5, 7],),
    ([1],),
)
LITERALS = (0, 2, 4)


def _signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "solve",
        (ProgramParameter("values", ProgramType.List(integer)),),
        integer,
    )


def _build(max_ast_cost: int = 6) -> full.PackedFullTypedFrontier:
    return full.PackedFullTypedFrontier.build(
        signature=_signature(),
        case_arguments=CASES,
        literals=LITERALS,
        max_ast_cost=max_ast_cost,
    )


def _events(frontier: full.PackedFullTypedFrontier) -> dict[str, int]:
    return dict(frontier.work.relation.event_counts)


def _delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in sorted(set(after) | set(before))
        if after.get(key, 0) != before.get(key, 0)
    }


def test_packed_primary_cross_type_build_and_item_less_than_count_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("full-typed V1 entered a public build/repack path")

    monkeypatch.setattr(scalar_v2.PackedScalarFrontier, "build", forbidden)
    monkeypatch.setattr(scalar_v3.PackedScalarFrontier, "build", forbidden)
    monkeypatch.setattr(typed_v2.TypedSemanticVersionSpace, "build", forbidden)
    for name in (
        "from_bindings",
        "rename",
        "cartesian_product",
        "select",
        "union",
        "difference",
        "project_distinct",
        "proof_forest",
        "family_slice",
    ):
        monkeypatch.setattr(packed.PackedRelationSession, name, forbidden)

    assert "desired" not in inspect.signature(full.PackedFullTypedFrontier.build).parameters
    frontier = _build()
    before = _events(frontier)
    state = frontier.state
    context = state.context_certificate

    assert frontier.status == full.FRONTIER_READY
    assert frontier.sha256 == ("29213c2fed27c383940efe369b675e0f64d4695f1084926fb47b1b5959e3e7a1")
    assert state.build_work_sha256 == (
        "4feb7387d26f02a8235a5b39727cf4ecf6356529ff9b38c6d60fb33a5ec22063"
    )
    assert frontier.work.relation.event_sha256 == (
        "bc0be2fc5473cd8621cbd567b1c5459f2ccfd68318293646bb05296b9332e1a4"
    )
    assert frontier.work.relation.completed_units == 1_839
    assert frontier.work.relation.attempted_units == 1_839
    assert sum(value for _key, value in frontier.work.relation.event_counts) == 1_839
    assert frontier.work.physical_predictions == 63
    assert frontier.work.prediction_cache_hits == 95
    assert frontier.work.physical_races == 136
    assert frontier.work.concrete_proof_witnesses == 0
    assert frontier.work.materialized_witnesses == 0
    assert len(frontier.families) == 10
    assert len(frontier.enodes) == 24
    assert len(frontier.eclasses) == 18
    assert len(frontier.root_eclasses) == 7
    assert context is not None
    assert context.sha256 == ("5f1ca53d2d32374d0b844bb72bf4e6f5843fb1ffc40d8e4fc9994a437c56c9d6")
    assert state.frozen_workset is not None
    assert state.frozen_workset.sha256 == (
        "cdf3ee7605a5acd1c0bbfb0b71a2f885f7706e234d34fec35068a84dd758d637"
    )
    assert state.component_capsule_sha256 == (
        "464897ab1e11759da14e5c52e5b0fb7c4b438fb9d3fd2b95cd63eaee18637233"
    )
    assert state.typed_capsule_sha256 == (
        "67ee5141ad4d78665bafa6f4b1aae99239a8ab9891e5c12ddf12eb6529ce3897"
    )

    productions = Counter(item.production for item in state.typed_families)
    assert productions == {
        "collection-role": 1,
        "item-role": 1,
        "literal": 1,
        "sum": 1,
        "less-than-item-literal": 1,
        "less-than-literal-item": 1,
        "predicate-binder": 2,
        "count": 2,
    }
    assert {item.production for item in state.typed_families if item.root_candidate} == {
        "sum",
        "count",
    }
    assert all(
        output.output_type is full.output_cells.OutputType.INT
        for eclass in frontier.root_eclasses
        for output in eclass.physical_outputs
    )
    less_family_ids = {
        item.family_id for item in state.typed_families if item.result_type_label == "Bool"
    }
    assert less_family_ids
    assert all(
        output.output_type is full.output_cells.OutputType.BOOL
        for enode in frontier.enodes
        if enode.family_id in less_family_ids
        for output in enode.physical_outputs
    )
    prediction_ids = tuple(
        identity for kind, identity in frontier._agenda_trace if kind == "prediction"
    )
    assert len(prediction_ids) == len(set(prediction_ids)) == 63
    assert before["operation:workset-seed-family"] == 3
    assert before["operation:workset-apply-family"] == 7
    assert before["operation:workset-authenticate-publication-topology"] == 1
    assert before["frontier:typed-family-index-insert"] == 10
    assert before["frontier:typed-root-index-insert"] == 7
    assert before.get("frontier:desired-value-read", 0) == 0
    assert before.get("operation:proof-forest-build", 0) == 0

    selection = frontier.select((2, 0, 1, 1))
    after = _events(frontier)
    assert selection.status == full.SELECTED
    assert selection.sha256 == ("8580ff6d3b47eaf2056584355ac8c583c7bc5dc4462d03d8c0a72385e60f1e74")
    assert selection.work.sha256 == (
        "564e0e2ce7adf72cb8dc2da1d0be2b76cd5b7a10f6196487bbcac0487ec089a2"
    )
    assert selection.work.relation.event_sha256 == (
        "317e592ee6f1c9e6edf70b89bee15972c8e77b87ea8d44de2a91bc36e3d3f709"
    )
    assert selection.work.relation.completed_units == 1_866
    assert selection.work.physical_races == 137
    assert selection.exact_proof_count == 1
    assert selection.selected_proof_forest_sha256 == (
        "56417f3b6f429abfd088f6f96927db0dd05671711411b6d37159ac0f31798eda"
    )
    assert tuple(item.source for item in selection.witnesses) == (
        "def solve(values):\n    return sum((1 for item_0 in values if item_0 < 2))\n",
    )
    assert _delta(after, before) == {
        "decision-edge-read": 3,
        "decision-node-read": 3,
        "frontier:authenticate-component-capsule": 1,
        "frontier:authenticate-typed-capsule": 1,
        "frontier:desired-value-read": 4,
        "frontier:materialize-family-lookup": 6,
        "frontier:materialize-witness": 1,
        "frontier:winner-eclass-index-lookup": 1,
        "operation:workset-authenticate-retained-publication-seal": 1,
        "operation:workset-materialize-selected-proof": 1,
        "workset:retained-publication-seal": 1,
        "workset:selected-group-lookup": 1,
        "workset:selected-proof-forest": 1,
        "workset:selected-proof-slice": 1,
        "workset:selected-root-materialization": 1,
    }


def test_sum_control_and_counter_orientation_select_from_root_only_types() -> None:
    control = _build(max_ast_cost=2)
    assert control.status == full.FRONTIER_READY
    assert control.sha256 == ("65015e3514f758428c7268700ac547b2a791780dbcd6aa7483279a39751bfa39")
    assert control.work.relation.completed_units == 326
    assert control.work.physical_predictions == 14
    assert control.work.physical_races == 38
    sum_selection = control.select((3, 5, 13, 1))
    assert sum_selection.status == full.SELECTED
    assert sum_selection.sha256 == (
        "badc3f5b6a27e7c1259e6487faf6c4c260b9a380a9c8e71267df3189fc616ffd"
    )
    assert sum_selection.exact_proof_count == 1
    assert sum_selection.selected_proof_forest_sha256 == (
        "582cd1bd05d9caa00cb4a6a202f3cb79d219a06c29955e80a3967401fe57d90f"
    )
    assert tuple(item.source for item in sum_selection.witnesses) == (
        "def solve(values):\n    return sum(values)\n",
    )

    counter = _build()
    counter_selection = counter.select((1, 1, 2, 0))
    assert counter_selection.status == full.SELECTED
    assert counter_selection.sha256 == (
        "47ecd3ba2d32b6be3a370eb75c92045d9ae6880bae062a3044e4b448c95f9d06"
    )
    assert counter_selection.exact_proof_count == 1
    assert counter_selection.selected_proof_forest_sha256 == (
        "d3292e5e2cbceadaa65cc16cac72c5a6a0ffbfdf703251f7a1f37dd7438a7b7d"
    )
    assert tuple(item.source for item in counter_selection.witnesses) == (
        "def solve(values):\n    return sum((1 for item_0 in values if 2 < item_0))\n",
    )


def test_typed_context_and_direct_root_maps_are_immutable_and_authenticated() -> None:
    frontier = _build(max_ast_cost=2)
    capsule = frontier._typed_capsule
    context = frontier.state.context_certificate
    assert capsule is not None
    assert context is not None
    assert capsule.family_records is frontier.state.typed_families
    assert capsule.root_eclasses is frontier.state.root_eclasses
    assert capsule.root_eclass_ids is frontier.state.root_eclass_ids
    with pytest.raises(TypeError):
        capsule.root_ordinal_index["substituted"] = 0
    with pytest.raises(AttributeError):
        object.__setattr__(context, "sha256", "0" * 64)

    frontier._typed_capsule = capsule._replace(root_ordinal_index=MappingProxyType({}))
    with pytest.raises(RuntimeError, match="typed context authentication"):
        frontier.select((3, 5, 13, 1))


def test_physical_count_winner_cannot_redirect_to_existing_sum_proof() -> None:
    frontier = _build()
    count_eclass_id = "61ebd3c0801837150b86fd7c31f827d2c6544eb988f9a374c3f6fb5e94b89104"
    count_eclass = next(
        eclass for eclass in frontier.root_eclasses if eclass.eclass_id == count_eclass_id
    )
    sum_eclass = next(
        eclass for eclass in frontier.root_eclasses if eclass.minimum_inside_cost == 2
    )
    assert count_eclass is not sum_eclass

    redirected = dict(frontier._typed_root_ordinal_index)
    redirected[count_eclass_id] = frontier.root_eclasses.index(sum_eclass)
    redirected_proxy = MappingProxyType(redirected)
    capsule = frontier._typed_capsule
    assert capsule is not None
    frontier._typed_root_ordinal_index = redirected_proxy
    frontier._typed_capsule = capsule._replace(root_ordinal_index=redirected_proxy)

    before = _events(frontier)
    with pytest.raises(RuntimeError, match=r"typed context authentication|winner"):
        frontier.select((2, 0, 1, 1))
    after = _events(frontier)
    assert after.get("operation:workset-materialize-selected-proof", 0) == before.get(
        "operation:workset-materialize-selected-proof", 0
    )
    assert after.get("frontier:materialize-witness", 0) == before.get(
        "frontier:materialize-witness", 0
    )
    assert frontier.work.concrete_proof_witnesses == 0
    assert frontier.work.materialized_witnesses == 0
