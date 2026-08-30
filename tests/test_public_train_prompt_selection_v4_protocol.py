from __future__ import annotations

import ast
import dataclasses
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.r3_pilot import public_train_prompt_selection_v3_protocol as v3
from experiments.r3_pilot import public_train_prompt_selection_v4_protocol as v4

EXPECTED_PROTOCOL_SHA256 = (
    "97f8ea272aac11188a0c5f31355085fddcf69eed602b0140eabc2f90d4920918"
)


def test_v3_protocol_and_banked_learner_provenance_are_exact() -> None:
    assert v4.V3_PROTOCOL_COMMIT == "8248e3164fa6c1a52433787a2d6a6758c2d64fc2"
    assert v3.PROTOCOL.protocol_sha256 == v4.V3_PROTOCOL_SHA256
    assert sha256(Path(v3.__file__).read_bytes()).hexdigest() == (
        v4.V3_PROTOCOL_SOURCE_SHA256
    )
    assert sha256(
        Path("tests/test_public_train_prompt_selection_v3_protocol.py").read_bytes()
    ).hexdigest() == v4.V3_PROTOCOL_TEST_SHA256
    assert v4.DEPENDENCY_RECORD["v3_learner_result_sha256_audit_only"] == (
        v4.V3_LEARNER_RESULT_SHA256_AUDIT_ONLY
    )
    assert v4.DEPENDENCY_RECORD["v3_learner_result_is_dependency_authority"] is False
    assert v4.DEPENDENCY_RECORD["v3_learner_source_commit"] == (
        "582de79f7b75aceea8f4ed64871310a15c96830a"
    )
    assert sha256(Path(v4.V3_LEARNER_SOURCE_PATH).read_bytes()).hexdigest() == (
        v4.V3_LEARNER_SOURCE_SHA256
    )
    assert sha256(Path(v4.V3_LEARNER_TEST_PATH).read_bytes()).hexdigest() == (
        v4.V3_LEARNER_TEST_SHA256
    )
    assert v4.DEPENDENCY_RECORD["v3_learner_dependency_binding_count"] == 1
    assert (
        v4.DEPENDENCY_RECORD["v3_learner_source_and_test_bytes_authenticated"]
        is True
    )
    assert v4.DEPENDENCY_RECORD["v3_learner_imported"] is False
    assert v4.DEPENDENCY_RECORD["v3_learner_source_executed"] is False
    assert v4.LEARNER_GATE_RECORD["learner_execution_authorized"] is False
    assert v4.LEARNER_GATE_RECORD["atomic_stop"] == v4.STOP_LEARNER_EXECUTION
    assert (
        v4.LEARNER_GATE_RECORD["v3_learner_source_and_test_bytes_authenticated"]
        is True
    )

    source = Path(v4.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "experiments.r3_pilot.public_train_prompt_selection_v3" not in imported_modules


def test_seven_channel_vocabulary_and_fresh_self_posed_cohort_are_exact() -> None:
    assert tuple(row.channel for row in v4.RELAY_SCHEMAS) == (
        "root-production-family",
        "ordered-parent-child-edge",
        "argument-operand-orientation",
        "parameter-role-binding",
        "literal-slot-binding",
        "literal-value-copy",
        "terminal-root-eclass-agreement",
    )
    assert v4.REMOVED_CHANNEL not in {row.channel for row in v4.RELAY_SCHEMAS}
    assert len(v4.OUTPUT_ATOMS) == 103
    assert len(v4.OUTPUT_TRANSITIONS) == 24
    assert not any("|depth|" in row.atom_id for row in v4.OUTPUT_ATOMS)
    assert not any(row.channel == v4.REMOVED_CHANNEL for row in v4.OUTPUT_ATOMS)

    assert len(v4.LESSONS) == 110
    assert len(v4.COUNTERFACTUAL_PAIRS) == 57
    assert Counter(row.audit_channel for row in v4.LESSONS) == {
        "root-production-family": 20,
        "ordered-parent-child-edge": 34,
        "argument-operand-orientation": 26,
        "parameter-role-binding": 6,
        "literal-slot-binding": 10,
        "literal-value-copy": 10,
        "terminal-root-eclass-agreement": 4,
    }
    assert Counter(row.kind for row in v4.COUNTERFACTUAL_PAIRS) == {
        "event-order-reversal": 55,
        "role-filler-swap": 2,
    }
    audit = v4.PRELEARNING_AUDIT
    assert audit.status == v4.PASS_PROTOCOL
    assert audit.lesson_count == audit.unique_causal_key_count == 110
    assert audit.causal_key_target_collision_count == 0
    assert audit.minimum_atom_support == v4.MIN_ATOM_SUPPORT == 2
    assert audit.minimum_transition_support == v4.MIN_TRANSITION_SUPPORT == 2
    assert audit.unsupported_atom_count == audit.unsupported_transition_count == 0
    assert audit.counterfactual_failure_count == 0
    assert audit.v3_physical_key_reuse_count == 0

    old_retained = tuple(
        row for row in v3.PROTOCOL.lessons if row.audit_channel != v4.REMOVED_CHANNEL
    )
    assert len(old_retained) == len(v4.LESSONS)
    assert all(
        new.causal_input.record() != old.causal_input_record()
        for new, old in zip(v4.LESSONS, old_retained, strict=True)
    )
    assert all(new.sha256 != old.sha256 for new, old in zip(v4.LESSONS, old_retained, strict=True))


def test_occurrence_topology_preserves_multiplicity_and_recursive_add_poison() -> None:
    assert len(v4.RECURSIVE_ADD_SHORT) == 4
    assert len(v4.RECURSIVE_ADD_LONG) == 6
    assert (
        v4.RECURSIVE_ADD_SHORT_TYPE_SET_SHA256
        == v4.RECURSIVE_ADD_LONG_TYPE_SET_SHA256
    )
    assert v4.RECURSIVE_ADD_SHORT_TARGET_SHA256 != v4.RECURSIVE_ADD_LONG_TARGET_SHA256
    long_counts = Counter(row.edge_type() for row in v4.RECURSIVE_ADD_LONG)
    assert long_counts[("add", "add", 0)] == 2
    assert long_counts[("add", "literal", 1)] == 3

    contract = v4.TOPOLOGY_CONTRACT
    assert contract["absolute_depth_channel_count"] == 0
    assert contract["ephemeral_parent_child_occurrence_bindings"] == (
        "distributed-physical-assemblies"
    )
    assert contract["one_repeated_masked_wta_per_edge_occurrence_then_stop"] is True
    assert contract["multiplicity_preserved"] is True
    assert contract["host_union_forbidden"] is True
    assert contract["host_deduplication_forbidden"] is True
    assert contract["host_filtering_forbidden"] is True
    assert contract["host_reordering_forbidden"] is True
    assert contract["path_hash_neural_evidence_forbidden"] is True
    assert contract["absolute_depth_neural_evidence_forbidden"] is True


def test_fit_only_projection_is_24_of_24_and_all_forbidden_surfaces_stay_zero() -> None:
    audit = v4.FIT_PROJECTION_AUDIT
    assert audit.status == v4.PASS_PROTOCOL
    assert audit.fit_identity_count == audit.projectable_identity_count == 24
    assert audit.unsupported_identity_count == 0
    assert audit.original_target_class_count == audit.reduced_target_class_count == 6
    assert audit.target_class_merge_count == 0
    assert audit.target_class_split_count == 0
    assert audit.distinct_target_collision_count == 0
    assert audit.causal_key_collision_count == 0
    assert audit.retained_label_count == 357
    assert audit.removed_label_count == 44
    assert audit.edge_occurrence_count == 117
    assert audit.repeated_edge_identity_count == 0
    assert audit.class_sizes == (1, 2, 4, 5, 5, 7)
    assert audit.heldout_target_read_count == 0
    assert audit.heldout_replay_identity_count == 0
    assert len(v4.FIT_PROJECTION_RECEIPTS) == 24

    old_to_new: dict[str, set[str]] = defaultdict(set)
    new_to_old: dict[str, set[str]] = defaultdict(set)
    for receipt in v4.FIT_PROJECTION_RECEIPTS:
        old_to_new[receipt.original_target_set_sha256].add(receipt.reduced_target_sha256)
        new_to_old[receipt.reduced_target_sha256].add(receipt.original_target_set_sha256)
    assert len(old_to_new) == len(new_to_old) == 6
    assert all(len(values) == 1 for values in old_to_new.values())
    assert all(len(values) == 1 for values in new_to_old.values())

    surface = v4.SURFACE_BOUNDARY_RECORD
    assert surface["data_only"] is True
    assert surface["heldout_surface_opened"] is False
    assert all(surface[key] == 0 for key in surface if key.endswith("_count"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        v4.FIT_PROJECTION_RECEIPTS[0].retained_label_count = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        surface["learner_array_count"] = 1  # type: ignore[index]

    assert v4._build_protocol() == v4.PROTOCOL
    assert v4.PROTOCOL.protocol_sha256 == EXPECTED_PROTOCOL_SHA256
    assert v4.PROTOCOL.protocol_sha256 == v4._namespaced_sha256(
        v4._PROTOCOL_NAMESPACE,
        v4.PROTOCOL.root_record(include_protocol_sha256=False),
    )
