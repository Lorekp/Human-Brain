from __future__ import annotations

import inspect
from collections.abc import Iterator

import pytest
from experiments.r3_pilot import public_typed_operator_successor as operator_successor
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_frontier_v1 as frontier_module


def _event_counts(
    frontier: frontier_module.PackedScalarFrontier,
) -> dict[str, int]:
    return dict(frontier.work.relation.event_counts)


def test_domains_are_primary_and_no_concrete_candidate_rows_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_from_bindings(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("frontier must not pack concrete candidates post hoc")

    monkeypatch.setattr(
        packed.PackedRelationSession,
        "from_bindings",
        forbidden_from_bindings,
    )
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-3, 2),
        literals=(-2, -2, 0, 3),
        max_ast_cost=3,
    )

    assert frontier.status == frontier_module.FRONTIER_READY
    assert frontier.state.literals == (-2, 0, 3)
    assert frontier.work.relation.binding_row_reads == 0
    # One role-domain value plus every supplied literal, including the duplicate.
    assert frontier.work.relation.domain_value_reads == 5
    assert not any(
        "CandidateNode" in type(value).__name__
        for collection in (frontier.families, frontier.enodes, frontier.eclasses)
        for value in collection
    )
    trace = frontier.work.agenda_trace
    for kind, family_id in trace:
        if kind == "semantic-expand":
            assert trace.index(("family-primary", family_id)) < trace.index((kind, family_id))


@pytest.mark.parametrize("stream_name", ("arguments", "literals"))
def test_raw_input_streams_stop_at_the_exact_cumulative_boundary(
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

    arguments: object = duplicate_stream() if stream_name == "arguments" else (1,)
    literals: object = duplicate_stream() if stream_name == "literals" else (1,)
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=arguments,  # type: ignore[arg-type]
        literals=literals,  # type: ignore[arg-type]
        max_ast_cost=1,
    )

    assert frontier.status == frontier_module.STOP_WORK_BOUND
    assert frontier.families == frontier.enodes == frontier.eclasses == ()
    assert frontier.work.relation.completed_units == 100
    assert frontier.work.relation.attempted_units == 101
    assert frontier.work.bound_name == "recipe-relation-work"
    assert frontier.work.bound_attempted == 101
    assert 0 < consumed <= 101
    counts = _event_counts(frontier)
    if stream_name == "arguments":
        assert counts["frontier:argument-value-read"] == 100
    else:
        # One successful role-domain read plus one denied literal fetch makes
        # the aggregate equal the number of literal values yielded.
        assert frontier.work.relation.domain_value_reads == consumed
        assert frontier.work.relation.denied_event == "domain-value-read"


def test_prediction_cache_calls_the_physical_operator_once_per_case_tuple(
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
        selected_sha = selected.sha256  # type: ignore[attr-defined]
        scope_tuple = tuple(scopes)  # type: ignore[arg-type]
        observed.append(
            (
                action.operator,  # type: ignore[attr-defined]
                selected_sha,
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
    assert len(observed) == len(set(observed))
    assert len(observed) == frontier.work.physical_predictions
    assert frontier.work.prediction_cache_hits > 0
    assert len(
        tuple(item for item in frontier.work.agenda_trace if item[0] == "prediction")
    ) == len(observed)


def test_unique_clamp_is_selected_only_after_freeze_and_materializes_one_proof() -> None:
    arguments = (-9, -5, -2, 0, 3, 7, 11)
    desired = tuple(min(max(value, -4), 6) for value in arguments)
    signature = inspect.signature(frontier_module.PackedScalarFrontier.build)
    assert "desired" not in signature.parameters
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=arguments,
        literals=(-4, 0, 6),
        max_ast_cost=5,
    )
    frozen_sha = frontier.sha256
    build_work = frontier.work

    assert frontier.status == frontier_module.FRONTIER_READY
    assert frontier.state.maximum_completed_ast_cost == 5
    assert build_work.concrete_proof_witnesses == 0
    assert build_work.materialized_witnesses == 0
    assert _event_counts(frontier).get("operation:proof-forest-build", 0) == 0
    assert all(
        all(type(item) is packed.PackedFamilySlice for item in eclass.proof_slices)
        for eclass in frontier.eclasses
    )
    assert build_work.physical_predictions < frontier_module.MAX_PHYSICAL_PREDICTIONS
    assert build_work.physical_races < frontier_module.MAX_PHYSICAL_RACES
    assert build_work.relation.completed_units < frontier_module.MAX_RECIPE_RELATION_WORK

    selection = frontier.select(desired)

    assert selection.status == frontier_module.SELECTED
    assert selection.frontier_sha256 == frozen_sha == frontier.sha256
    assert selection.exact_proof_count == 1
    assert len(selection.witnesses) == 1
    assert selection.selected_proof_forest_sha256 is not None
    assert selection.witnesses[0].source == "min(max(x, -4), 6)"
    assert selection.work.concrete_proof_witnesses == 1
    assert selection.work.materialized_witnesses == 1
    assert _event_counts(frontier)["operation:proof-forest-build"] == 1
    assert selection.checkpoint_before_sha256 == selection.checkpoint_after_sha256


def test_multiplicity_normalization_and_proof_ambiguity_remain_distinct() -> None:
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
    role_slots = tuple(slot for slot in repeated_add.relation.slots if slot.kind == "role")
    assert tuple(slot.occurrence for slot in role_slots) == (0, 1)
    assert _event_counts(repeated)["operation:normalize"] == 1
    # x + 0 was explicitly projected back to x, so it is not a third proof.
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
    decision = ambiguous.select((1, 2, 5))

    assert before.materialized_witnesses == 0
    assert before.concrete_proof_witnesses == 0
    assert dict(before.relation.event_counts).get("operation:proof-forest-build", 0) == 0
    assert decision.status == frontier_module.STOP_AMBIGUOUS
    assert decision.exact_proof_count == 2
    assert len(decision.witnesses) == 2
    assert {item.source for item in decision.witnesses} == {"x", "abs(x)"}
    assert decision.selected_proof_forest_sha256 is not None
    assert decision.work.concrete_proof_witnesses == 2
    assert decision.work.materialized_witnesses == 2
    assert _event_counts(ambiguous)["operation:proof-forest-build"] == 1


def test_recursive_live_tamper_fails_before_observation() -> None:
    topology = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-1, 2), literals=(-2, 3), max_ast_cost=1
    )
    literal_family = next(item for item in topology.families if item.production == "literal")
    node = literal_family.relation.nodes[0]
    object.__setattr__(node, "edges", ((99, node.edges[0][1]),))
    with pytest.raises(RuntimeError, match="authentication"):
        topology.select((-2, -2))

    physical = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-1, 2), literals=(-2, 3), max_ast_cost=1
    )
    encoded = physical.enodes[0].physical_outputs[0]
    object.__setattr__(encoded, "physical_digest", b"\x00" * 32)
    with pytest.raises((RuntimeError, ValueError)):
        physical.select((-2, -2))


def test_post_race_forest_overflow_returns_no_partial_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(1, 2, 5), literals=(0, 4), max_ast_cost=2
    )
    original_race = frontier_module.semantic_race.select_exact_semantic_vector
    race_executed = False

    def tighten_remaining_relation_budget(*args: object, **kwargs: object) -> object:
        nonlocal race_executed
        result = original_race(*args, **kwargs)  # type: ignore[arg-type]
        race_executed = True
        frontier._session.limit = frontier._session.audit.completed_units + 1
        return result

    monkeypatch.setattr(
        frontier_module.semantic_race,
        "select_exact_semantic_vector",
        tighten_remaining_relation_budget,
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


def test_audit_is_an_exact_sum_and_small_cap_returns_no_partial_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(frontier_module, "MAX_RECIPE_RELATION_WORK", 31)
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-2, 1), literals=(-1, 0, 2), max_ast_cost=3
    )

    relation = frontier.work.relation
    assert frontier.status == frontier_module.STOP_WORK_BOUND
    assert relation.completed_units == 31
    assert relation.attempted_units == 32
    assert relation.denied_event is not None
    assert sum(count for _event, count in relation.event_counts) == 31
    assert frontier.families == frontier.enodes == frontier.eclasses == ()
    assert frontier.select(iter(())).status == frontier_module.STOP_WORK_BOUND


def test_nineteen_literal_cost_nine_returns_an_authenticated_honest_boundary() -> None:
    frontier = frontier_module.PackedScalarFrontier.build(
        case_arguments=(-13, -7, -2, 3, 12),
        literals=range(-9, 10),
        max_ast_cost=9,
    )

    assert frontier.status == frontier_module.STOP_WORK_BOUND
    assert frontier.sha256 == ("7b9eec0de1e3e8f33ff000ac7946f01a172ec13ff8e17111f681ddfc68f41f00")
    assert frontier.work.sha256 == (
        "9aa96034838dc8c10fbd8c747e0037cac51297b6bcb20f2137c02363b0aed557"
    )
    assert frontier.state.maximum_completed_ast_cost == 3
    assert frontier.state.root_eclass_ids == ()
    assert frontier.work.bound_name == "recipe-relation-work"
    assert frontier.work.relation.completed_units == 30_000
    assert frontier.work.relation.attempted_units == 30_001
    assert frontier.work.relation.denied_event == "decision-node-read"
    assert frontier.work.relation.event_sha256 == (
        "2850cf276693ba25e6d225cf9a24f42fb5c8e8751f53626894d92d3319d7f44f"
    )
    assert frontier.work.physical_predictions == 321
    assert frontier.work.prediction_cache_hits == 304
    assert frontier.work.physical_races == 1_085
    assert frontier.work.concrete_proof_witnesses == 0
    assert frontier.families == frontier.enodes == frontier.eclasses == ()
