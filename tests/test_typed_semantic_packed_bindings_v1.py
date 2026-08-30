from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from experiments.r3_pilot.typed_semantic_packed_bindings_v1 import (
    PackedBindingRelation,
    PackedFamilySlice,
    PackedProofForest,
    ProofNormalizationProjection,
    ProofWitness,
    RelationAllocation,
    SlotConstraint,
    TypedSlot,
)


def _role(name: str, occurrence: int = 0) -> TypedSlot:
    return TypedSlot("role", "Int", name, occurrence)


def _literal(name: str, occurrence: int = 0) -> TypedSlot:
    return TypedSlot("literal", "Int", name, occurrence)


def test_relation_is_canonical_deterministic_immutable_and_authenticated() -> None:
    argument = _role("argument")
    literal = _literal("offset")
    rows = [[3, 11], [1, 7], [3, 11], [2, 9]]

    first = PackedBindingRelation.from_bindings((argument, literal), rows)
    second = PackedBindingRelation.from_bindings(
        (literal, argument),
        ((11, 3), (9, 2), (7, 1)),
    )
    before = first.canonical_bytes()
    rows[0][0] = 999

    assert first == second
    assert first.sha256 == second.sha256
    assert first.canonical_bytes() == before == second.canonical_bytes()
    assert first.first_bindings() == ((7, 1), (9, 2))
    assert first.authenticate() is True
    assert isinstance(before, bytes)
    with pytest.raises(FrozenInstanceError):
        first.root_id = None  # type: ignore[misc]
    with pytest.raises(ValueError, match="SHA-256"):
        replace(first, sha256="0" * 64)


def test_wide_domains_and_cartesian_products_charge_nodes_and_edges() -> None:
    literal = _literal("literal")
    role = _role("role")
    literal_domain = PackedBindingRelation.domain(literal, reversed(range(-9, 10)))
    role_domain = PackedBindingRelation.domain(role, range(19))

    assert literal_domain.exact_count == 19
    assert literal_domain.saturating_count(7) == 7
    assert literal_domain.allocation == RelationAllocation(1, 19)

    product = literal_domain.cartesian_product(role_domain)
    reverse_product = role_domain.cartesian_product(literal_domain)

    assert product == reverse_product
    assert product.exact_count == 19 * 19
    assert product.saturating_count(20) == 20
    # One shared suffix node per independent domain, but every domain edge is
    # charged.  A wide domain therefore cannot disappear behind one node.
    assert product.allocation == RelationAllocation(2, 38)
    assert product.first_bindings() == ((-9, 0), (-9, 1))


def test_natural_join_constraints_and_distinct_projection() -> None:
    a = _role("a")
    b = _role("b")
    c = _role("c")
    left = PackedBindingRelation.from_bindings(
        (a, b),
        ((1, 2), (2, 2), (3, 4)),
    )
    right = PackedBindingRelation.from_bindings(
        (b, c),
        ((2, 5), (2, 1), (4, 0)),
    )

    joined = left.join(right)
    assert joined.exact_count == 5
    assert joined.first_bindings(5) == (
        (1, 2, 1),
        (1, 2, 5),
        (2, 2, 1),
        (2, 2, 5),
        (3, 4, 0),
    )

    selected = joined.select(
        (
            SlotConstraint.slots(a, "lt", c),
            SlotConstraint.value(b, "eq", 2),
        )
    )
    assert selected.exact_count == 2
    assert selected.first_bindings() == ((1, 2, 5), (2, 2, 5))
    assert selected.project_distinct((b,)).first_bindings() == ((2,),)
    assert joined.project_distinct((b,)).first_bindings() == ((2,), (4,))


def test_union_and_difference_are_exact_and_order_independent() -> None:
    slot = _literal("choice")
    low = PackedBindingRelation.domain(slot, (1, 2))
    overlap = PackedBindingRelation.domain(slot, (2, 3))
    disjoint = PackedBindingRelation.domain(slot, (4, 5))

    merged = low.union(overlap).union(disjoint)
    reordered = disjoint.union(overlap.union(low))

    assert merged == reordered
    assert merged.exact_count == 5
    assert merged.first_bindings() == ((1,), (2,))
    assert merged.difference(overlap).first_bindings(5) == (
        (1,),
        (4,),
        (5,),
    )
    assert low.union(disjoint).exact_count == low.exact_count + disjoint.exact_count
    assert low.difference(low).exact_count == 0


def test_slot_occurrences_preserve_x_plus_x_and_add_zero_needs_normalization() -> None:
    x0 = _role("operand", 0)
    x1 = _role("operand", 1)
    zero = _literal("zero")

    x = PackedBindingRelation.from_bindings((x0,), ((7,),))
    x_plus_x = PackedBindingRelation.from_bindings((x0, x1), ((7, 7),))
    add_zero = PackedBindingRelation.from_bindings((x0, zero), ((7, 0),))

    assert x.sha256 != x_plus_x.sha256
    assert x.slots != x_plus_x.slots
    with pytest.raises(ValueError, match="schemas do not match"):
        x.union(add_zero)

    x_slice = PackedFamilySlice.build("x", x)
    add_zero_slice = PackedFamilySlice.build("add-zero", add_zero)
    unnormalized = PackedProofForest.build((x_slice, add_zero_slice))
    assert unnormalized.exact_logical_recipe_count == 2

    projection = ProofNormalizationProjection(
        normalization_id="add-zero-identity",
        source_family_id="add-zero",
        target_family_id="x",
        retained_source_slots=(x0,),
        target_slots=(x0,),
        required_constants=((zero, 0),),
    )
    normalized = add_zero_slice.normalize(projection)
    assert normalized.family_id == "x"
    assert normalized.relation == x

    merged = PackedProofForest.build((x_slice, normalized))
    assert merged.exact_logical_recipe_count == 1
    assert merged.first_two_witnesses == (ProofWitness("x", (7,)),)


def test_proof_forest_merges_duplicate_proofs_but_retains_distinct_families() -> None:
    slot = _literal("choice")
    f_left = PackedFamilySlice.build(
        "f", PackedBindingRelation.domain(slot, (1, 2))
    )
    f_right = PackedFamilySlice.build(
        "f", PackedBindingRelation.domain(slot, (2, 3))
    )
    g = PackedFamilySlice.build("g", PackedBindingRelation.domain(slot, (1, 2)))

    first = PackedProofForest.build((g, f_right, f_left), saturation_limit=4)
    second = PackedProofForest.build((f_left, g, f_right), saturation_limit=4)

    assert first == second
    assert tuple(item.family_id for item in first.slices) == ("f", "g")
    assert first.exact_logical_recipe_count == 5
    assert first.saturating_logical_recipe_count == 4
    assert first.first_two_witnesses == (
        ProofWitness("f", (1,)),
        ProofWitness("f", (2,)),
    )
    assert first.sha256 == second.sha256
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.authenticate() is True
    assert first.allocation.decision_edges == 5
    with pytest.raises(FrozenInstanceError):
        first.sha256 = "0" * 64  # type: ignore[misc]
    with pytest.raises(ValueError, match="exact logical count"):
        replace(first, exact_logical_recipe_count=0)


def test_large_forest_count_saturates_without_losing_exact_count_or_witnesses() -> None:
    left = PackedBindingRelation.domain(_literal("left"), range(19))
    right = PackedBindingRelation.domain(_role("right"), range(19))
    family = PackedFamilySlice.build("product", left.cartesian_product(right))

    forest = PackedProofForest.build((family,), saturation_limit=31)

    assert forest.exact_logical_recipe_count == 361
    assert forest.saturating_logical_recipe_count == 31
    assert forest.first_two_witnesses == (
        ProofWitness("product", (0, 0)),
        ProofWitness("product", (0, 1)),
    )
    assert forest.allocation == RelationAllocation(2, 38)
