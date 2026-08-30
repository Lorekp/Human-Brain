from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from experiments.r3_pilot.typed_semantic_packed_bindings_v1 import (
    PackedRelationSession,
    PackedRelationWorkLimitExceeded,
    ProofNormalizationProjection,
    SlotConstraint,
    TypedSlot,
)


def _role(name: str, occurrence: int = 0) -> TypedSlot:
    return TypedSlot("role", "Int", name, occurrence)


def _literal(name: str, occurrence: int = 0) -> TypedSlot:
    return TypedSlot("literal", "Int", name, occurrence)


def _session(limit: int = 1_000_000) -> PackedRelationSession:
    return PackedRelationSession(work_limit=limit)


def _assert_audit_is_exact(session: PackedRelationSession) -> None:
    audit = session.audit
    assert audit.completed_units == (
        audit.operation_lookups
        + audit.domain_value_reads
        + audit.binding_row_reads
        + audit.decision_node_reads
        + audit.decision_edge_reads
        + audit.decision_node_proposals
        + audit.decision_edge_proposals
        + audit.proof_unions
    )
    assert audit.completed_units == sum(count for _event, count in audit.event_counts)
    assert audit.attempted_units == audit.completed_units
    assert audit.denied_event is None
    assert len(audit.event_sha256) == 64


def test_canonical_rows_cache_hits_and_observers_are_session_owned() -> None:
    argument = _role("argument")
    literal = _literal("offset")
    rows = ((3, 11), (1, 7), (3, 11), (2, 9))
    session = _session()

    first = session.from_bindings((argument, literal), rows)
    before = session.audit
    second = session.from_bindings(
        (literal, argument),
        ((right, left) for left, right in reversed(rows)),
    )
    after = session.audit

    assert second is first
    assert session.exact_count(first) == 3
    assert session.first_bindings(first, 3) == ((7, 1), (9, 2), (11, 3))
    assert session.authenticate_relation(first)
    assert session.canonical_bytes(first) == session.canonical_bytes(second)
    assert after.binding_row_reads - before.binding_row_reads == len(rows)
    assert after.cache_hits - before.cache_hits == 1
    assert after.completed_units - before.completed_units > len(rows)
    with pytest.raises(FrozenInstanceError):
        first.root_id = None  # type: ignore[misc]
    _assert_audit_is_exact(session)


class _ObservedRow:
    def __init__(self, consumed: list[int]) -> None:
        self._consumed = consumed

    def __iter__(self):
        self._consumed[0] += 1
        yield 7


def test_duplicate_inputs_are_charged_before_row_or_value_processing() -> None:
    slot = _literal("choice")
    yielded_rows = [0]
    consumed_rows = [0]

    def duplicate_rows():
        row = _ObservedRow(consumed_rows)
        for _index in range(200_000):
            yielded_rows[0] += 1
            yield row

    row_session = _session(limit=17)
    with pytest.raises(PackedRelationWorkLimitExceeded) as caught:
        row_session.from_bindings((slot,), duplicate_rows())

    assert caught.value.event == "binding-row-read"
    assert caught.value.completed == 17
    assert caught.value.attempted == 18
    assert yielded_rows[0] == 17
    assert consumed_rows[0] == 16
    assert row_session.audit.binding_row_reads == 16
    assert row_session.audit.operation_lookups == 1

    yielded_values = [0]

    def duplicate_values():
        for _index in range(200_000):
            yielded_values[0] += 1
            yield 3

    domain_session = _session(limit=13)
    with pytest.raises(PackedRelationWorkLimitExceeded) as caught:
        domain_session.domain(slot, duplicate_values())

    assert caught.value.event == "domain-value-read"
    assert yielded_values[0] == 13
    assert domain_session.audit.domain_value_reads == 12
    assert domain_session.audit.completed_units == 13
    assert domain_session.audit.attempted_units == 14


def test_cartesian_product_observers_and_exact_boundary_denial() -> None:
    build = _session()
    literal_domain = build.domain(_literal("literal"), reversed(range(-9, 10)))
    role_domain = build.domain(_role("role"), (2, 0, 1))
    product = build.cartesian_product(literal_domain, role_domain)

    assert build.exact_count(literal_domain) == 19
    assert build.exact_count(product) == 57
    assert build.saturating_count(product, 2) == 2
    assert build.first_bindings(product, 2) == ((-9, 0), (-9, 1))
    assert product.allocation.decision_nodes == 2
    assert product.allocation.decision_edges == 22

    measured = _session()
    assert measured.exact_count(product) == 57
    required = measured.audit.completed_units
    exact_boundary = _session(limit=required)
    assert exact_boundary.exact_count(product) == 57
    denied = _session(limit=required - 1)
    with pytest.raises(PackedRelationWorkLimitExceeded) as caught:
        denied.exact_count(product)
    assert caught.value.completed == required - 1
    assert caught.value.attempted == required
    assert denied.audit.denied_event in {
        "decision-node-read",
        "decision-edge-read",
    }
    _assert_audit_is_exact(build)


def test_join_constraints_projection_rename_union_and_difference() -> None:
    a = _role("a")
    b = _role("b")
    c = _role("c")
    renamed = _role("renamed")
    session = _session()
    left = session.from_bindings((a, b), ((1, 2), (2, 2), (3, 4)))
    right = session.from_bindings((b, c), ((2, 8), (2, 9), (5, 10)))

    joined = session.join(left, right)
    selected = session.select(
        joined,
        (
            SlotConstraint.value(c, "ge", 9),
            SlotConstraint.slots(a, "lt", c),
        ),
    )
    projected = session.project_distinct(selected, (a,))
    relabeled = session.rename(projected, (renamed,))

    assert session.exact_count(joined) == 4
    assert session.first_bindings(selected, 4) == ((1, 2, 9), (2, 2, 9))
    assert session.first_bindings(projected, 3) == ((1,), (2,))
    assert relabeled.slots == (renamed,)
    assert session.first_bindings(relabeled, 3) == ((1,), (2,))

    set_slot = _literal("set")
    low = session.domain(set_slot, (1, 2))
    high = session.domain(set_slot, (2, 3))
    combined = session.union(low, high)
    assert session.union(high, low) is combined
    assert session.first_bindings(combined, 4) == ((1,), (2,), (3,))
    assert session.first_bindings(session.difference(combined, high), 3) == ((1,),)
    assert session.audit.cache_hits >= 1
    _assert_audit_is_exact(session)


def test_large_union_cannot_finish_one_unit_beyond_the_limit() -> None:
    source = _session()
    slot = _literal("wide")
    left = source.domain(slot, range(0, 200, 2))
    right = source.domain(slot, range(1, 200, 2))

    union_only = _session()
    combined = union_only.union(left, right)
    assert source.exact_count(combined) == 200
    required = union_only.audit.completed_units

    exact_boundary = _session(limit=required)
    assert exact_boundary.union(left, right).sha256 == combined.sha256
    denied = _session(limit=required - 1)
    with pytest.raises(PackedRelationWorkLimitExceeded) as caught:
        denied.union(left, right)
    assert caught.value.completed == required - 1
    assert caught.value.attempted == required
    assert denied.audit.denied_event in {
        "decision-node-read",
        "decision-edge-read",
    }


def test_occurrence_multiplicity_and_explicit_add_zero_normalization() -> None:
    x0 = _role("operand", 0)
    x1 = _role("operand", 1)
    zero = _literal("zero")
    session = _session()

    x = session.from_bindings((x0,), ((7,),))
    repeated = session.cartesian_product(
        x,
        session.from_bindings((x1,), ((7,),)),
    )
    assert repeated.slots == tuple(sorted((x0, x1), key=lambda slot: slot.sort_key))
    assert session.exact_count(repeated) == 1
    assert len(repeated.slots) == 2

    add_zero = session.cartesian_product(
        x,
        session.from_bindings((zero,), ((0,),)),
    )
    source = session.family_slice("add-family", add_zero)
    target_slot = _role("value")
    projection = ProofNormalizationProjection(
        normalization_id="add-zero-v1",
        source_family_id="add-family",
        target_family_id="identity-family",
        retained_source_slots=(x0,),
        target_slots=(target_slot,),
        required_constants=((zero, 0),),
    )
    normalized = session.normalize(source, projection)

    assert normalized.family_id == "identity-family"
    assert normalized.relation.slots == (target_slot,)
    assert session.first_bindings(normalized.relation) == ((7,),)
    _assert_audit_is_exact(session)


def test_forest_merges_only_same_family_and_keeps_deterministic_witnesses() -> None:
    slot = _literal("choice")
    session = _session()
    f_left = session.family_slice("f", session.domain(slot, (1, 2)))
    f_right = session.family_slice("f", session.domain(slot, (2, 3)))
    g_same_binding = session.family_slice("g", session.domain(slot, (2,)))

    forest = session.proof_forest(
        (g_same_binding, f_right, f_left),
        saturation_limit=3,
    )

    assert tuple(item.family_id for item in forest.slices) == ("f", "g")
    assert session.first_bindings(forest.slices[0].relation, 4) == (
        (1,),
        (2,),
        (3,),
    )
    assert session.forest_exact_count(forest) == 4
    assert session.forest_saturating_count(forest) == 3
    assert tuple(
        (item.family_id, item.binding)
        for item in session.forest_first_witnesses(forest)
    ) == (("f", (1,)), ("f", (2,)))
    assert session.authenticate_forest(forest)
    assert session.audit.proof_unions == 1

    other = session.proof_forest(
        (session.family_slice("h", session.domain(slot, (9,))),),
        saturation_limit=3,
    )
    merged = session.union_forests(forest, other)
    assert tuple(item.family_id for item in merged.slices) == ("f", "g", "h")
    assert session.audit.proof_unions >= 2
    _assert_audit_is_exact(session)


def test_large_forest_build_and_observer_deny_at_exact_boundary() -> None:
    source = _session()
    left = source.domain(_literal("left"), range(19))
    right = source.domain(_role("right"), range(32))
    product = source.cartesian_product(left, right)
    family = source.family_slice("wide", product)

    measured_build = _session()
    forest = measured_build.proof_forest((family,), saturation_limit=2)
    build_units = measured_build.audit.completed_units
    denied_build = _session(limit=build_units - 1)
    with pytest.raises(PackedRelationWorkLimitExceeded) as caught:
        denied_build.proof_forest((family,), saturation_limit=2)
    assert caught.value.completed == build_units - 1
    assert caught.value.attempted == build_units

    assert source.exact_count(product) == 608
    assert source.saturating_count(product, 2) == 2
    assert measured_build.forest_exact_count(forest) == 608
    assert measured_build.forest_saturating_count(forest) == 2
    assert len(measured_build.forest_first_witnesses(forest)) == 2
    assert measured_build.forest_allocation(forest) == product.allocation

    measured_observer = _session()
    assert measured_observer.forest_exact_count(forest) == 608
    observer_units = measured_observer.audit.completed_units
    denied_observer = _session(limit=observer_units - 1)
    with pytest.raises(PackedRelationWorkLimitExceeded) as caught:
        denied_observer.forest_exact_count(forest)
    assert caught.value.completed == observer_units - 1
    assert caught.value.attempted == observer_units


def test_recursive_tamper_and_valid_cache_substitution_cannot_return_results() -> None:
    slot = _literal("choice")

    node_session = _session()
    node_relation = node_session.domain(slot, (1, 2))
    object.__setattr__(node_relation.nodes[0], "node_id", "0" * 64)
    with pytest.raises(RuntimeError, match="relation authentication failed"):
        node_session.exact_count(node_relation)

    edge_session = _session()
    edge_relation = edge_session.domain(slot, (1, 2))
    original_edges = edge_relation.nodes[0].edges
    object.__setattr__(
        edge_relation.nodes[0],
        "edges",
        ((99, original_edges[0][1]), *original_edges[1:]),
    )
    with pytest.raises(RuntimeError, match="relation authentication failed"):
        edge_session.first_bindings(edge_relation)

    slice_session = _session()
    family_slice = slice_session.family_slice(
        "family",
        slice_session.domain(slot, (3,)),
    )
    object.__setattr__(family_slice, "family_id", "tampered")
    with pytest.raises(RuntimeError, match="slice authentication failed"):
        slice_session.authenticate_slice(family_slice)

    forest_session = _session()
    clean_slice = forest_session.family_slice(
        "family",
        forest_session.domain(slot, (3,)),
    )
    forest = forest_session.proof_forest((clean_slice,))
    object.__setattr__(forest, "exact_logical_recipe_count", 99)
    with pytest.raises(RuntimeError, match="forest authentication failed"):
        forest_session.forest_exact_count(forest)

    cache_session = _session()
    cached = cache_session.domain(slot, (1, 2))
    replacement_session = _session()
    replacement = replacement_session.domain(slot, (7, 8))
    cache_entry = next(
        entry for entry in cache_session._cache.values() if entry.value is cached
    )
    object.__setattr__(cache_entry, "value", replacement)
    with pytest.raises(RuntimeError, match="cache result identity changed"):
        cache_session.domain(slot, (1, 2))


def test_audit_exact_sum_covers_every_work_event_category() -> None:
    slot = _literal("choice")
    session = _session()
    from_rows = session.from_bindings((slot,), ((1,), (1,), (2,)))
    from_domain = session.domain(slot, (2, 2, 3))
    left = session.family_slice("same", from_rows)
    right = session.family_slice("same", from_domain)
    forest = session.proof_forest((left, right), saturation_limit=2)
    assert session.forest_exact_count(forest) == 3

    audit = session.audit
    assert audit.operation_lookups > 0
    assert audit.domain_value_reads == 3
    assert audit.binding_row_reads == 3
    assert audit.decision_node_reads > 0
    assert audit.decision_edge_reads > 0
    assert audit.decision_node_proposals > 0
    assert audit.decision_edge_proposals > 0
    assert audit.proof_unions == 1
    _assert_audit_is_exact(session)
    with pytest.raises(FrozenInstanceError):
        audit.completed_units = 0  # type: ignore[misc]
