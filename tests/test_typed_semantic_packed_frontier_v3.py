from __future__ import annotations

import inspect
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import pytest
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_frontier_v2 as v2
from experiments.r3_pilot import typed_semantic_packed_frontier_v3 as frontier_module

STRESS_ARGUMENTS = (-13, -7, -2, 3, 12)
STRESS_LITERALS = range(-9, 10)
STRESS_CLAMP = tuple(min(max(value, -4), 6) for value in STRESS_ARGUMENTS)


def _events(frontier: frontier_module.PackedScalarFrontier) -> dict[str, int]:
    return dict(frontier.work.relation.event_counts)


def _delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in sorted(set(after) | set(before))
        if after.get(key, 0) != before.get(key, 0)
    }


def test_v3_replaces_public_v2_build_and_public_relation_repacking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("V3 called a completed V2/public repacking path")

    monkeypatch.setattr(v2.PackedScalarFrontier, "build", forbidden)
    for name in (
        "from_bindings",
        "rename",
        "cartesian_product",
        "select",
        "union",
        "difference",
        "project_distinct",
        "exact_count",
        "saturating_count",
        "first_bindings",
        "canonical_bytes",
        "family_slice",
        "proof_forest",
        "authenticate_forest",
        "forest_exact_count",
        "forest_saturating_count",
        "forest_first_witnesses",
        "forest_allocation",
        "forest_canonical_bytes",
    ):
        monkeypatch.setattr(packed.PackedRelationSession, name, forbidden)

    assert "desired" not in inspect.signature(frontier_module.PackedScalarFrontier.build).parameters
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, 2),
        literals=(-2, 0, 3),
        max_ast_cost=3,
    )
    before = _events(frontier)

    assert frontier.status == frontier_module.FRONTIER_READY
    assert frontier.work.concrete_proof_witnesses == 0
    assert frontier.work.materialized_witnesses == 0
    assert before.get("frontier:desired-value-read", 0) == 0
    assert before.get("operation:proof-forest-build", 0) == 0
    assert before.get("operation:family-slice", 0) == 0

    selection = frontier.select((-3, 2))

    assert selection.status == frontier_module.SELECTED
    assert selection.exact_proof_count == 1
    assert selection.witnesses[0].source == "x"
    assert _events(frontier).get("operation:proof-forest-build", 0) == 0
    assert _events(frontier).get("operation:family-slice", 0) == 0


def test_cost_five_construction_certificate_and_unique_selection_fit_unchanged_cap() -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=STRESS_ARGUMENTS,
        literals=STRESS_LITERALS,
        max_ast_cost=5,
    )
    frozen = frontier.state.frozen_workset
    before = _events(frontier)

    assert frontier.status == frontier_module.FRONTIER_READY
    assert frontier.sha256 == ("e0eecb0a980d14e77bb0e2fe0ec7d74d09408ceae8b6ebfe8a06f394653f7b81")
    assert frontier.state.build_work_sha256 == (
        "2f675749810e971359757f26f88f5d12745b1fafa94698fbc864da67169c58a7"
    )
    assert frontier.work.relation.event_sha256 == (
        "1d02c9fa19c25799a856a042caf442b343d67705c705245dedb3cc285e792dd8"
    )
    assert frontier.work.relation.completed_units == 29_703
    assert frontier.work.relation.attempted_units == 29_703
    assert frontier.work.relation.denied_event is None
    assert frontier.work.physical_predictions == 422
    assert frontier.work.prediction_cache_hits == 723
    assert frontier.work.physical_races == 1_489
    assert frontier.work.concrete_proof_witnesses == 0
    assert frontier.work.materialized_witnesses == 0
    assert len(frontier.families) == 7
    assert len(frontier.enodes) == 249
    assert len(frontier.eclasses) == 230
    assert frozen is not None
    assert len(frozen.partitions) == 7
    assert sum(len(item.groups) for item in frozen.partitions) == 249
    assert frozen.topology_node_count > 0
    assert len(frozen.metadata_sha256) == 64
    assert len(frozen.publication_certificate_sha256) == 64

    assert before["operation:workset-publish"] == 1
    assert before["operation:workset-authenticate-publication-topology"] == 1
    assert before.get("operation:workset-authenticate-publication", 0) == 0
    assert before["workset:publication-partition"] == 7
    assert before["workset:publication-group"] == 249
    assert before["workset:published-group-index-insert"] == 249
    assert before["frontier:family-index-insert"] == 7
    assert before["frontier:eclass-index-insert"] == 230
    assert before["workset:published-root-reference"] == 259
    assert before["workset:published-root-lookup"] == 274
    assert before.get("workset:publication-group-authentication", 0) == 0
    assert before.get("workset:published-child-ref-authentication", 0) == 0
    assert before.get("workset:retained-group-authentication", 0) == 0
    assert before.get("workset:retained-partition-authentication", 0) == 0
    assert before.get("frontier:desired-value-read", 0) == 0
    assert before.get("operation:proof-forest-build", 0) == 0

    selection = frontier.select(STRESS_CLAMP)
    after = _events(frontier)

    assert selection.status == frontier_module.SELECTED
    assert selection.sha256 == ("fb9601469c5cf34943f98d76c164dc4ffa5bdcae8a37a52652feb22c4b2fe1df")
    assert selection.work.sha256 == (
        "17c30c3e4e739df7b1166f834db74c1d2eed85ca192de6d48482875080372591"
    )
    assert selection.work.relation.event_sha256 == (
        "ba6edc261cb5658102d16db00817ffe4b0838be3142812ff4ee370aba7d1284e"
    )
    assert selection.work.relation.completed_units == 29_729
    assert selection.work.relation.attempted_units == 29_729
    assert selection.work.physical_predictions == 422
    assert selection.work.prediction_cache_hits == 723
    assert selection.work.physical_races == 1_490
    assert selection.exact_proof_count == 1
    assert tuple(item.source for item in selection.witnesses) == ("min(max(x, -4), 6)",)
    assert selection.selected_proof_forest_sha256 == (
        "b05a2a5334ae4e28fad50b8e3b914d120002128998820792c70a95bc6c490d13"
    )
    assert 30_603 - selection.work.relation.completed_units == 874
    assert 874 > 603
    assert _delta(after, before) == {
        "decision-edge-read": 3,
        "decision-node-read": 3,
        "frontier:authenticate-component-capsule": 1,
        "frontier:desired-value-read": 5,
        "frontier:materialize-family-lookup": 5,
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


def test_cost_five_supported_absent_reaches_physical_no_exact_without_proof() -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=STRESS_ARGUMENTS,
        literals=STRESS_LITERALS,
        max_ast_cost=5,
    )
    before = _events(frontier)
    selection = frontier.select((31, 31, 31, 31, 31))
    after = _events(frontier)

    assert selection.status == frontier_module.STOP_NO_EXACT
    assert selection.sha256 == ("11cde31e226967411bb490a186436ac88769333785a5344584f639ceb60bbadc")
    assert selection.work.sha256 == (
        "d2cc616c1df2e3803464349ed0ad6f6466fe472fc7d0d5d49f76b490c39fd37e"
    )
    assert selection.work.relation.event_sha256 == (
        "542535337a9b61d025cdb736e25cfd6198635e78f07a9a6391cbf60cea9e2142"
    )
    assert selection.work.relation.completed_units == 29_711
    assert selection.work.relation.attempted_units == 29_711
    assert selection.race is not None
    assert selection.race.sha256 == (
        "50b1dcc6dee4f9d6b067dfe0c2a9d8a8bc9a55ca7bfea2f8ab8faff581defbb1"
    )
    assert selection.exact_proof_count == 0
    assert selection.witnesses == ()
    assert selection.selected_proof_forest_sha256 is None
    assert selection.work.concrete_proof_witnesses == 0
    assert selection.work.materialized_witnesses == 0
    assert _delta(after, before) == {
        "frontier:authenticate-component-capsule": 1,
        "frontier:desired-value-read": 5,
        "operation:workset-authenticate-retained-publication-seal": 1,
        "workset:retained-publication-seal": 1,
    }
    assert after.get("workset:selected-root-materialization", 0) == 0
    assert after.get("operation:proof-forest-build", 0) == 0


def test_cost_five_cap_minus_one_is_atomic_before_public_frontier_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(frontier_module, "MAX_RECIPE_RELATION_WORK", 29_702)
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=STRESS_ARGUMENTS,
        literals=STRESS_LITERALS,
        max_ast_cost=5,
    )

    assert frontier.status == frontier_module.STOP_WORK_BOUND
    assert frontier.sha256 == ("c43b891c38d7af189df9437ea2952307f6e96dc908445d5f3e3d3e28812d72c7")
    assert frontier.work.sha256 == (
        "f2379f7a525b4e04f9e693c54b5f55ea969844e49d7e514549181653d1725349"
    )
    assert frontier.work.relation.event_sha256 == (
        "74f97b119592e3271e63e38ad81a6ba164308048162e36b2dc443829d321e512"
    )
    assert frontier.work.relation.completed_units == 29_702
    assert frontier.work.relation.attempted_units == 29_703
    assert frontier.work.relation.denied_event == "frontier:frontier-seal"
    assert frontier.state.maximum_completed_ast_cost == 5
    assert frontier.state.frozen_workset is None
    assert frontier.families == frontier.enodes == frontier.eclasses == ()
    assert frontier.work.physical_predictions == 422
    assert frontier.work.prediction_cache_hits == 723
    assert frontier.work.physical_races == 1_489
    assert frontier.work.concrete_proof_witnesses == 0
    assert _events(frontier).get("frontier:desired-value-read", 0) == 0
    assert _events(frontier).get("operation:proof-forest-build", 0) == 0


def test_cost_nine_remains_an_authenticated_atomic_stop_after_cost_five() -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=STRESS_ARGUMENTS,
        literals=STRESS_LITERALS,
        max_ast_cost=9,
    )

    assert frontier.status == frontier_module.STOP_WORK_BOUND
    assert frontier.sha256 == ("985403691cfb2a95cac9e9a7b266dea6b7ec445351aff54799c471570d52566c")
    assert frontier.work.sha256 == (
        "a00b6b943573d3a85602d38b113f5c0924892c8b1fe2a80da37baf126474690d"
    )
    assert frontier.work.relation.event_sha256 == (
        "8e4de868fc00be50d455bc8d6b61b5737acfd6d5c72f5c2de60b8f5f2efc18a3"
    )
    assert frontier.work.relation.completed_units == 30_000
    assert frontier.work.relation.attempted_units == 30_001
    assert frontier.work.relation.denied_event == "workset:schema-owner-lookup"
    assert frontier.work.bound_name == "recipe-relation-work"
    assert frontier.work.bound_attempted == 30_001
    assert frontier.state.maximum_completed_ast_cost == 5
    assert frontier.state.frozen_workset is None
    assert frontier.families == frontier.enodes == frontier.eclasses == ()
    assert frontier.work.physical_predictions == 422
    assert frontier.work.prediction_cache_hits == 723
    assert frontier.work.physical_races == 1_489
    assert frontier.work.concrete_proof_witnesses == 0

    selection = frontier.select((0, 0, 0, 0, 0))
    assert selection.status == frontier_module.STOP_WORK_BOUND
    assert selection.race is None
    assert selection.witnesses == ()
    assert selection.selected_proof_forest_sha256 is None
    assert selection.work.relation.completed_units == 30_000


def test_small_unique_normalization_multiplicity_and_ambiguity_keep_v2_semantics() -> None:
    arguments = (-9, -5, -2, 0, 3, 7, 11)
    unique = frontier_module.PackedScalarFrontier.build(
        case_arguments=arguments,
        literals=(-4, 0, 6),
        max_ast_cost=5,
    )
    unique_selection = unique.select(tuple(min(max(value, -4), 6) for value in arguments))
    assert unique.work.physical_predictions == 68
    assert unique.work.prediction_cache_hits == 23
    assert unique_selection.work.physical_races == 210
    assert unique_selection.status == frontier_module.SELECTED
    assert unique_selection.witnesses[0].source == "min(max(x, -4), 6)"
    assert unique_selection.selected_proof_forest_sha256 == (
        "b05a2a5334ae4e28fad50b8e3b914d120002128998820792c70a95bc6c490d13"
    )

    repeated = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-2, 1),
        literals=(0, 3),
        max_ast_cost=3,
    )
    role = next(item for item in repeated.families if item.production == "role")
    repeated_add = next(
        item
        for item in repeated.families
        if item.production == "add" and item.child_family_ids == (role.family_id, role.family_id)
    )
    role_slots = tuple(slot for slot in repeated_add.schema if slot.kind == "role")
    assert tuple(slot.occurrence for slot in role_slots) == (0, 1)
    frozen = repeated.state.frozen_workset
    assert frozen is not None
    normalized_add = next(item for item in frozen.partitions if item.normalization is not None)
    role_partition = next(item for item in frozen.partitions if item.family_id == role.family_id)
    assert normalized_add.normalization is not None
    assert (
        normalized_add.normalization.recovered_root_id
        == normalized_add.normalization.target_root_id
        == role_partition.root_id
    )
    identity = repeated.select((-2, 1))
    assert identity.status == frontier_module.SELECTED
    assert identity.exact_proof_count == 1
    assert identity.witnesses[0].source == "x"

    ambiguous = frontier_module.PackedScalarFrontier.build(
        case_arguments=(1, 2, 5),
        literals=(0, 4),
        max_ast_cost=2,
    )
    before = ambiguous.work
    ambiguity = ambiguous.select((1, 2, 5))
    assert before.concrete_proof_witnesses == before.materialized_witnesses == 0
    assert ambiguity.status == frontier_module.STOP_AMBIGUOUS
    assert ambiguity.exact_proof_count == 2
    assert {item.source for item in ambiguity.witnesses} == {"x", "abs(x)"}
    assert ambiguity.selected_proof_forest_sha256 == (
        "da7d0dbf30684bec8f32adaa8f21c47dbc8d3e7d7202838f1335c7757c378565"
    )


def test_publication_component_and_direct_indexes_are_strict_and_tamper_evident() -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, 2),
        literals=(-2, 0, 3),
        max_ast_cost=3,
    )
    frozen = frontier.state.frozen_workset
    capsule = frontier._authenticated_components
    assert frozen is not None
    assert capsule is frontier._workset._component_capsule
    assert capsule is not None
    assert len(capsule.family_index) == len(frontier.families)
    assert len(capsule.eclass_index) == len(frontier.eclasses)
    assert len(capsule.group_to_enode) == len(frontier.enodes)
    assert len(capsule.enode_to_eclass) == len(frontier.enodes)
    assert all(
        capsule.family_index[key] is item
        for key, item in zip(
            (item.family_id for item in frontier.families),
            frontier.families,
            strict=True,
        )
    )
    assert all(
        capsule.eclass_index[key] is item
        for key, item in zip(
            frontier.state.root_eclass_ids,
            frontier.eclasses,
            strict=True,
        )
    )
    with pytest.raises(AttributeError):
        object.__setattr__(frozen, "sha256", "0" * 64)
    with pytest.raises(AttributeError):
        object.__setattr__(frozen.partitions[0], "sha256", "0" * 64)
    with pytest.raises(TypeError):
        capsule.eclass_index["substituted"] = frontier.eclasses[0]

    frontier._authenticated_components = capsule._replace(sha256="0" * 64)
    with pytest.raises(RuntimeError, match="component authentication"):
        frontier.select((31, 31))

    retained = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, 2),
        literals=(-2, 0, 3),
        max_ast_cost=3,
    )
    authenticated = retained._workset._authenticated
    assert isinstance(authenticated, frontier_module._AuthenticatedFrozen)
    retained._workset._authenticated = authenticated._replace(retained_seal_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="authentication"):
        retained.select((31, 31))

    indexed = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, 2),
        literals=(-2, 0, 3),
        max_ast_cost=3,
    )
    indexed_capsule = indexed._authenticated_components
    assert indexed_capsule is not None
    indexed._authenticated_components = indexed_capsule._replace(eclass_index=MappingProxyType({}))
    with pytest.raises(RuntimeError, match="component authentication"):
        indexed.select((31, 31))


def test_retained_arena_blocks_forged_literal_witness_and_nested_edge_mutation() -> None:
    def literal_group(
        frontier: frontier_module.PackedScalarFrontier,
        value: int,
    ) -> tuple[
        frontier_module._AuthenticatedFrozen,
        frontier_module.PackedSemanticGroupRef,
        packed.DecisionNode,
    ]:
        frozen = frontier.state.frozen_workset
        authenticated = frontier._workset._authenticated
        assert frozen is not None
        assert type(authenticated) is frontier_module._AuthenticatedFrozen
        partition = next(item for item in frozen.partitions if item.production == "literal")
        group = next(
            item
            for item in partition.groups
            if item.root_id is not None
            and authenticated.arena.node(item.root_id).edges == ((value, packed._TERMINAL_ID),)
        )
        assert group.root_id is not None
        return authenticated, group, authenticated.arena.node(group.root_id)

    retained = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, 2),
        literals=(-2, 0, 3),
        max_ast_cost=3,
    )
    authenticated, group, node = literal_group(retained, -2)
    forged = packed.DecisionNode(
        node_id=node.node_id,
        slot=node.slot,
        edges=((999, packed._TERMINAL_ID),),
    )
    with pytest.raises(TypeError):
        authenticated.arena.nodes[group.root_id] = forged
    with pytest.raises(AttributeError):
        object.__setattr__(authenticated.arena, "nodes", MappingProxyType({}))
    with pytest.raises(AttributeError):
        object.__setattr__(authenticated, "arena", authenticated.arena)

    honest = retained.select((-2, -2))
    assert honest.status == frontier_module.SELECTED
    assert honest.exact_proof_count == 1
    assert tuple(item.source for item in honest.witnesses) == ("-2",)

    substituted = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, 2),
        literals=(-2, 0, 3),
        max_ast_cost=3,
    )
    substituted_auth, substituted_group, substituted_node = literal_group(substituted, -2)
    forged_nodes = dict(substituted_auth.arena.nodes)
    forged_nodes[substituted_group.root_id] = packed.DecisionNode(
        node_id=substituted_node.node_id,
        slot=substituted_node.slot,
        edges=((999, packed._TERMINAL_ID),),
    )
    forged_arena = frontier_module._StrictRetainedArena(
        MappingProxyType(forged_nodes),
    )
    substituted._workset._authenticated = substituted_auth._replace(arena=forged_arena)
    with pytest.raises(RuntimeError, match="authentication"):
        substituted.select((-2, -2))
    assert substituted.work.materialized_witnesses == 0

    nested = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, 2),
        literals=(-2, 0, 3),
        max_ast_cost=3,
    )
    _nested_auth, _nested_group, nested_node = literal_group(nested, -2)
    race_count = nested.work.physical_races
    object.__setattr__(nested_node, "edges", ((999, packed._TERMINAL_ID),))
    with pytest.raises(RuntimeError, match="content address"):
        nested.select((-2, -2))
    assert nested.work.physical_races == race_count + 1
    assert nested.work.concrete_proof_witnesses == 0
    assert nested.work.materialized_witnesses == 0


def test_post_race_work_overflow_exposes_no_race_forest_or_partial_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(1, 2, 5),
        literals=(0, 4),
        max_ast_cost=2,
    )
    original = frontier_module.semantic_race.select_exact_semantic_vector
    race_executed = False

    def tighten_after_race(*args: object, **kwargs: object) -> object:
        nonlocal race_executed
        result = original(*args, **kwargs)  # type: ignore[arg-type]
        race_executed = True
        session = frontier._workset.session
        session.limit = session.audit.completed_units + 1
        return result

    monkeypatch.setattr(
        frontier_module.semantic_race,
        "select_exact_semantic_vector",
        tighten_after_race,
    )
    decision = frontier.select((1, 2, 5))

    assert race_executed
    assert decision.status == frontier_module.STOP_WORK_BOUND
    assert decision.race is None
    assert decision.exact_proof_count == 0
    assert decision.witnesses == ()
    assert decision.selected_proof_forest_sha256 is None
    assert decision.work.concrete_proof_witnesses == 0
    assert decision.work.materialized_witnesses == 0
    assert decision.work.relation.attempted_units == (decision.work.relation.completed_units + 1)


def test_v2_source_remains_byte_exact() -> None:
    source = Path(v2.__file__).read_bytes()
    assert sha256(source).hexdigest() == (
        "07ca708ccd8e4d8c97c34e9bc459d32d509b93fdc1ed735d690ffbca15df7e9b"
    )
