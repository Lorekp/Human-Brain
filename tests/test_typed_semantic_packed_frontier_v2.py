from __future__ import annotations

import inspect
from collections.abc import Iterator
from dataclasses import replace

import pytest
from experiments.r3_pilot import public_typed_operator_successor as operator_successor
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_frontier_v2 as frontier_module


def _events(frontier: frontier_module.PackedScalarFrontier) -> dict[str, int]:
    return dict(frontier.work.relation.event_counts)


def test_session_native_workset_is_the_only_build_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("frontier used a public concrete/per-tuple relation path")

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
        "forest_exact_count",
        "forest_saturating_count",
        "forest_first_witnesses",
        "forest_allocation",
        "forest_canonical_bytes",
    ):
        monkeypatch.setattr(packed.PackedRelationSession, name, forbidden)

    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, 2),
        literals=(-2, 0, 3),
        max_ast_cost=3,
    )

    assert frontier.status == frontier_module.FRONTIER_READY
    assert frontier.state.maximum_completed_ast_cost == 3
    assert frontier.work.concrete_proof_witnesses == 0
    assert dict(frontier.work.relation.event_counts).get("operation:proof-forest-build", 0) == 0
    assert {family.production for family in frontier.families} == {
        "role",
        "literal",
        "absolute",
        "add",
        "maximum",
    }


@pytest.mark.parametrize("stream_name", ("arguments", "literals"))
def test_raw_duplicate_streams_stop_before_limit_plus_one_is_consumed(
    monkeypatch: pytest.MonkeyPatch,
    stream_name: str,
) -> None:
    monkeypatch.setattr(frontier_module, "MAX_RECIPE_RELATION_WORK", 100)
    consumed = 0

    def duplicate_stream() -> Iterator[int]:
        nonlocal consumed
        for _index in range(200_000):
            consumed += 1
            yield 1

    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(duplicate_stream() if stream_name == "arguments" else (1,)),
        literals=(duplicate_stream() if stream_name == "literals" else (1,)),
        max_ast_cost=1,
    )

    assert frontier.status == frontier_module.STOP_WORK_BOUND
    assert frontier.families == frontier.enodes == frontier.eclasses == ()
    assert frontier.work.relation.completed_units == 100
    assert frontier.work.relation.attempted_units == 101
    assert 0 < consumed <= 101
    if stream_name == "arguments":
        assert _events(frontier)["frontier:argument-value-read"] == 100
    else:
        assert frontier.work.relation.domain_value_reads == consumed
        assert frontier.work.relation.denied_event == "domain-value-read"


def test_physical_prediction_is_unique_per_supported_case_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = operator_successor.train_public_typed_operator_successor()[0]
    original = operator_successor.FrozenTypedOperatorSuccessor.predict
    observed: list[tuple[str, str, str | None]] = []

    def tracked_predict(
        self: operator_successor.FrozenTypedOperatorSuccessor,
        selected: object,
        action: object,
        *,
        scopes: object = (),
    ) -> object:
        scope_tuple = tuple(scopes)  # type: ignore[arg-type]
        observed.append(
            (
                action.operator,  # type: ignore[attr-defined]
                selected.sha256,  # type: ignore[attr-defined]
                None if not scope_tuple else scope_tuple[0].sha256,
            )
        )
        return original(self, selected, action, scopes=scope_tuple)  # type: ignore[arg-type]

    monkeypatch.setattr(
        operator_successor.FrozenTypedOperatorSuccessor,
        "predict",
        tracked_predict,
    )
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, -3, 2, 2),
        literals=(-1, 0, 1),
        max_ast_cost=3,
        operator=operator,
    )

    assert frontier.status == frontier_module.FRONTIER_READY
    assert len(observed) == len(set(observed)) == frontier.work.physical_predictions
    assert frontier.work.prediction_cache_hits > 0


def test_published_partition_authenticates_constraints_and_normalization() -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, 2),
        literals=(-2, 0, 3),
        max_ast_cost=3,
    )
    frozen = frontier.state.frozen_workset
    assert frozen is not None
    normalized = next(
        item
        for item in frozen.partitions
        if item.production == "add" and item.normalization_id is not None
    )
    assert normalized.normalization_id == "add-right-zero-v2"
    assert len(normalized.constraints) == 1
    assert normalized.constraints[0][1:] == ("ne", None, 0)
    certificate = normalized.normalization
    assert certificate is not None
    role = next(item for item in frozen.partitions if item.production == "role")
    assert certificate.target_family_id == role.family_id
    assert certificate.target_slots == role.schema
    assert certificate.retained_source_slots == role.schema
    literal_slot = next(slot for slot in normalized.schema if slot.kind == "literal")
    assert certificate.required_constants == ((literal_slot, 0),)
    assert certificate.recovered_root_id == certificate.target_root_id == role.root_id

    object.__setattr__(
        frozen,
        "partitions",
        tuple(
            replace(item, normalization_id="substituted") if item is normalized else item
            for item in frozen.partitions
        ),
    )
    with pytest.raises(RuntimeError, match="authentication"):
        frontier.select((-3, 2))


def test_selection_reuses_the_exact_publication_capsule_without_topology_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, 2),
        literals=(-2, 0, 3),
        max_ast_cost=3,
    )
    before = _events(frontier)
    assert before["operation:workset-authenticate-publication"] == 1

    def forbidden_authenticate(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("selection replayed publication authentication")

    monkeypatch.setattr(
        frontier_module._SemanticPartitionWorkset,
        "authenticate",
        forbidden_authenticate,
    )
    decision = frontier.select((31, 31))
    after = _events(frontier)

    assert decision.status == frontier_module.STOP_NO_EXACT
    assert after["operation:workset-authenticate-publication"] == 1
    assert after["operation:workset-authenticate-retained"] == (
        before.get("operation:workset-authenticate-retained", 0) + 1
    )
    for event in (
        "decision-node-read",
        "decision-edge-read",
        "workset:published-root-lookup",
    ):
        assert after.get(event, 0) == before.get(event, 0)


def test_session_nonce_is_private_and_public_builds_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nonces = iter(("11" * 16, "22" * 16))
    monkeypatch.setattr(frontier_module.secrets, "token_hex", lambda _size: next(nonces))
    first = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, 2), literals=(-2, 0, 3), max_ast_cost=3
    )
    second = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, 2), literals=(-2, 0, 3), max_ast_cost=3
    )

    assert first.sha256 == second.sha256
    assert first.state.frozen_workset == second.state.frozen_workset
    assert first.state.build_work_sha256 == second.state.build_work_sha256
    assert first.work.relation.event_sha256 == second.work.relation.event_sha256
    first_handle = next(iter(first._workset._records.values())).handle
    second_handle = next(iter(second._workset._records.values())).handle
    assert first_handle.session_nonce != second_handle.session_nonce
    assert first_handle.sha256 != second_handle.sha256


def test_nonzero_literal_domain_needs_no_normalization_projection() -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, 2), literals=(-2, 3), max_ast_cost=3
    )
    frozen = frontier.state.frozen_workset

    assert frontier.status == frontier_module.FRONTIER_READY
    assert frozen is not None
    assert all(item.normalization is None for item in frozen.partitions)
    assert _events(frontier).get("workset:normalization-source-merge", 0) == 0


def test_unique_clamp_materializes_one_proof_only_after_the_frozen_race() -> None:
    arguments = (-9, -5, -2, 0, 3, 7, 11)
    desired = tuple(min(max(value, -4), 6) for value in arguments)
    assert "desired" not in inspect.signature(frontier_module.PackedScalarFrontier.build).parameters
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=arguments,
        literals=(-4, 0, 6),
        max_ast_cost=5,
    )
    frozen_sha256 = frontier.sha256

    assert frontier.status == frontier_module.FRONTIER_READY
    assert frontier.sha256 == "cb9718712400bbaa753f90294fdc3740dee0e8b73dce95d70f93ef23b6cae3fe"
    assert frontier.state.build_work_sha256 == (
        "3208fd4fe24ced6efe10e942ec2bd7dff29f5daa90c726525c057ae3954ab7d8"
    )
    assert frontier.state.maximum_completed_ast_cost == 5
    assert frontier.work.relation.completed_units == 1_515
    assert frontier.work.physical_predictions == 68
    assert frontier.work.prediction_cache_hits == 23
    assert frontier.work.physical_races == 209
    assert frontier.work.concrete_proof_witnesses == 0
    assert frontier.work.materialized_witnesses == 0
    assert _events(frontier).get("operation:proof-forest-build", 0) == 0

    selection = frontier.select(desired)

    assert selection.status == frontier_module.SELECTED
    assert frontier.sha256 == selection.frontier_sha256 == frozen_sha256
    assert selection.exact_proof_count == 1
    assert len(selection.witnesses) == 1
    assert selection.witnesses[0].source == "min(max(x, -4), 6)"
    assert selection.selected_proof_forest_sha256 == (
        "b05a2a5334ae4e28fad50b8e3b914d120002128998820792c70a95bc6c490d13"
    )
    assert selection.work.concrete_proof_witnesses == 1
    assert selection.work.materialized_witnesses == 1
    assert _events(frontier)["operation:proof-forest-build"] == 1
    assert selection.checkpoint_before_sha256 == selection.checkpoint_after_sha256


def test_multiplicity_normalization_and_proof_ambiguity_remain_distinct() -> None:
    repeated = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-2, 1), literals=(0, 3), max_ast_cost=3
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
        case_arguments=(1, 2, 5), literals=(0, 4), max_ast_cost=2
    )
    before = ambiguous.work
    decision = ambiguous.select((1, 2, 5))

    assert ambiguous.sha256 == ("1b881d650c241271c48ed2853dffbc2552dcc6045c764097fdb071d19e261ace")
    assert before.relation.completed_units == 195
    assert before.physical_predictions == 3
    assert before.prediction_cache_hits == 0
    assert before.physical_races == 3
    assert before.concrete_proof_witnesses == before.materialized_witnesses == 0
    assert decision.status == frontier_module.STOP_AMBIGUOUS
    assert decision.exact_proof_count == 2
    assert len(decision.witnesses) == 2
    assert {item.source for item in decision.witnesses} == {"x", "abs(x)"}
    assert decision.selected_proof_forest_sha256 == (
        "da7d0dbf30684bec8f32adaa8f21c47dbc8d3e7d7202838f1335c7757c378565"
    )
    assert decision.work.concrete_proof_witnesses == 2
    assert decision.work.materialized_witnesses == 2
    assert _events(ambiguous)["operation:proof-forest-build"] == 1


def test_component_capsule_is_strictly_immutable_and_bijective() -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, 2), literals=(-2, 0, 3), max_ast_cost=3
    )
    capsule = frontier._authenticated_components
    assert capsule is frontier._workset._component_capsule
    assert capsule is not None
    assert len(capsule.group_to_enode) == len(frontier.enodes)
    assert len(capsule.enode_to_eclass) == len(frontier.enodes)
    assert {item[0] for item in capsule.group_to_enode} == {
        item.group_ref.group_id for item in frontier.enodes
    }
    assert {item[1] for item in capsule.group_to_enode} == {
        item.enode_id for item in frontier.enodes
    }
    assert {item[0] for item in capsule.enode_to_eclass} == {
        item.enode_id for item in frontier.enodes
    }
    assert {item[1] for item in capsule.enode_to_eclass} == {
        item.eclass_id for item in frontier.eclasses
    }
    with pytest.raises(AttributeError):
        object.__setattr__(
            frontier.eclasses[0],
            "minimum_inside_cost",
            frontier.eclasses[0].minimum_inside_cost + 99,
        )

    frontier._authenticated_components = capsule._replace(sha256="0" * 64)
    with pytest.raises(RuntimeError, match="component authentication"):
        frontier.select((31, 31))


def test_post_race_relation_overflow_exposes_no_partial_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(1, 2, 5), literals=(0, 4), max_ast_cost=2
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


def test_nineteen_literal_cost_five_is_exactly_ready_under_the_unchanged_cap() -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-13, -7, -2, 3, 12),
        literals=range(-9, 10),
        max_ast_cost=5,
    )
    frozen = frontier.state.frozen_workset

    assert frontier.status == frontier_module.FRONTIER_READY
    assert frontier.sha256 == ("2eea8c295685b1aa7ac98bdbdf42444f4fd9b344a5059f72bc98bce6324f20a8")
    assert frontier.work.sha256 == (
        "de95d63069272ede0565aa8b5578d06145b5fdaeed7bd40780877ee47b178b4b"
    )
    assert frontier.work.relation.event_sha256 == (
        "bed750213bf1ee8b3608a451b4db212092d6e9df9f57f26ef7f78a03e5528aef"
    )
    assert frontier.state.maximum_completed_ast_cost == 5
    assert len(frontier.families) == 7
    assert len(frontier.enodes) == 249
    assert len(frontier.eclasses) == 230
    assert frozen is not None
    clamp = next(item for item in frozen.partitions if item.production == "minimum")
    assert len(clamp.groups) == 190
    assert frontier.work.relation.completed_units == 29_951
    assert frontier.work.relation.attempted_units == 29_951
    assert frontier.work.relation.denied_event is None
    assert frontier.work.physical_predictions == 422
    assert frontier.work.prediction_cache_hits == 723
    assert frontier.work.physical_races == 1_489
    assert frontier.work.concrete_proof_witnesses == 0
    assert frontier.work.materialized_witnesses == 0
    counts = _events(frontier)
    assert counts["workset:semantic-tuple-read"] == 401
    assert counts["workset:supported-semantic-tuple"] == 229
    assert counts["operation:workset-authenticate-publication"] == 1
    assert counts.get("operation:proof-forest-build", 0) == 0
    assert not any(name.startswith("frontier:auth-") for name in counts)

    selection = frontier.select((31, 31, 31, 31, 31))

    assert selection.status == frontier_module.STOP_WORK_BOUND
    assert selection.sha256 == ("531979f6db6680c708a5530f0fe8439297445c2c06985a474399157e35807e75")
    assert selection.work.sha256 == (
        "368096a3cd82d1d21605e591c10d1241a9004c5288efb869ee40e17481738e01"
    )
    assert selection.work.relation.event_sha256 == (
        "082e5f6ef82b85e081cc39436de4f471836122affe8b1499b28481dd733992a9"
    )
    assert selection.work.relation.completed_units == 30_000
    assert selection.work.relation.attempted_units == 30_001
    assert selection.work.relation.denied_event == ("workset:retained-group-authentication")
    assert selection.work.bound_name == "recipe-relation-work"
    assert selection.work.bound_attempted == 30_001
    assert selection.race is None
    assert selection.exact_proof_count == 0
    assert selection.witnesses == ()
    assert selection.selected_proof_forest_sha256 is None
    assert selection.work.concrete_proof_witnesses == 0
    assert selection.work.materialized_witnesses == 0
    after = _events(frontier)
    assert after.get("frontier:desired-value-read", 0) == 0
    assert after.get("operation:proof-forest-build", 0) == 0


def test_cost_five_cap_minus_one_fails_before_seal_with_no_partial_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(frontier_module, "MAX_RECIPE_RELATION_WORK", 29_950)
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-13, -7, -2, 3, 12),
        literals=range(-9, 10),
        max_ast_cost=5,
    )

    assert frontier.status == frontier_module.STOP_WORK_BOUND
    assert frontier.sha256 == ("1bc80f4d7b05a897d90117a295fdb2d22f48a788c845fe87093fc76b4d7c135f")
    assert frontier.work.sha256 == (
        "1f6eaae9cd2306ff1d1863fd58dd21f463e230cf60a19a7331f83ed3e30a2768"
    )
    assert frontier.work.relation.event_sha256 == (
        "c4dab17101f239a2e818f493338d871875a9c46b279a78a133d23eaac846441b"
    )
    assert frontier.state.maximum_completed_ast_cost == 5
    assert frontier.work.relation.completed_units == 29_950
    assert frontier.work.relation.attempted_units == 29_951
    assert frontier.work.relation.denied_event == "frontier:frontier-seal"
    assert frontier.work.physical_predictions == 422
    assert frontier.work.prediction_cache_hits == 723
    assert frontier.work.physical_races == 1_489
    assert frontier.work.concrete_proof_witnesses == 0
    assert frontier.state.frozen_workset is None
    assert frontier.families == frontier.enodes == frontier.eclasses == ()
    assert sum(count for _event, count in frontier.work.relation.event_counts) == 29_950


def test_nineteen_literal_cost_nine_is_an_authenticated_atomic_stop() -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-13, -7, -2, 3, 12),
        literals=range(-9, 10),
        max_ast_cost=9,
    )

    assert frontier.status == frontier_module.STOP_WORK_BOUND
    assert frontier.sha256 == ("ad49a95c9b9355562e6f65a1cb4d26676892efe5fd3ced4746b924c1749bed76")
    assert frontier.work.sha256 == (
        "d983079204d5cc62e96903f2fc1a84bd3a5bb25aa976e7f85d0ae3d33ca96fe8"
    )
    assert frontier.work.relation.event_sha256 == (
        "8e4de868fc00be50d455bc8d6b61b5737acfd6d5c72f5c2de60b8f5f2efc18a3"
    )
    assert frontier.state.maximum_completed_ast_cost == 5
    assert frontier.work.relation.completed_units == 30_000
    assert frontier.work.relation.attempted_units == 30_001
    assert frontier.work.bound_name == "recipe-relation-work"
    assert frontier.work.bound_attempted == 30_001
    assert frontier.work.physical_predictions == 422
    assert frontier.work.prediction_cache_hits == 723
    assert frontier.work.physical_races == 1_489
    assert frontier.work.concrete_proof_witnesses == 0
    assert frontier.state.frozen_workset is None
    assert frontier.families == frontier.enodes == frontier.eclasses == ()
