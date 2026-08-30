from __future__ import annotations

import dataclasses
import inspect
from collections import Counter, defaultdict
from hashlib import sha256

import pytest
from experiments.r3_pilot import public_train_prompt_selection_v3_protocol as v3

EXPECTED_ROOTS = {
    "dependency_sha256": "1bacdd2780e9efbb303ebb87e00e7fd08a36e93854a374c948be8ec5d91353a3",
    "grammar_sha256": "4d58322a966821a7b3e8535a004e6c34427a245fb8e2a7f34c8923d9144687d2",
    "vocabulary_sha256": "136c344e3ff4eec68983a9c23cc37423ce9ea408bf0001b51ec1bf39db23ced8",
    "curriculum_sha256": "1c65c147fb4933b45d82c616e0e66ba8431dd9513e4ea8d3bd9544973940d09d",
    "audit_sha256": "f484c623fef4b5dd86eabfca93d260d09040f2263ef87ce1435db24779edb53f",
    "controls_sha256": "567709d235c52fed4f6a01ac5da27fcb5abf9547d6311eae2c117431b4a0461f",
    "learner_gate_sha256": "5b2e02a82a86d226e79614a779f0b5487e3de54ab5ba775b350cdc4594b95df1",
    "surface_boundary_sha256": "26d8d6713271000e6f7b8f975bb02599a833c4a1c41b5ec2263371475db663e2",
    "protocol_sha256": "1eb2e3503790ebd721f69cc5d6f6dd8a32e0bb816d9b7f3dc8f1b4be9f15983f",
}


def test_spent_v2_and_capsule_are_authenticated_without_import_or_target_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert v3.V2_COMMIT == "b479688"
    assert v3.V2_SOURCE_SHA256 == (
        "b5a388c57b864c39bd9008d96a00ae03f2007ab2b28582a920aacdf8dd4608aa"
    )
    assert v3.V2_RESULT_SHA256 == (
        "72228a1177bfcaf162533f5e8d91a46ec73e0bc180d5b8ed869ff49162a74c2b"
    )
    assert v3.V2_FIT_EXACT_IDENTITY_COUNT == 16
    assert v3.V2_HELDOUT_EXACT_IDENTITY_COUNT == 0
    assert v3.CAPSULE_COMMIT == "5c34341"
    assert v3.CAPSULE_SHA256 == ("484521f34a98b53f4e21a1d279510c07165fc65f8de72b54845c06a831762caf")
    assert (
        v3.PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT,
        v3.CANONICAL_CAUSAL_IDENTITY_COUNT,
        v3.CANONICAL_FIT_IDENTITY_COUNT,
        v3.CANONICAL_HELDOUT_IDENTITY_COUNT,
    ) == (90, 42, 24, 18)
    for relative_path, expected in (
        (v3.V2_SOURCE_PATH, v3.V2_SOURCE_SHA256),
        (v3.CAPSULE_SOURCE_PATH, v3.CAPSULE_SOURCE_SHA256),
        (v3.COMPILER_GRAMMAR_SOURCE_PATH, v3.COMPILER_GRAMMAR_SOURCE_SHA256),
    ):
        assert sha256((v3._ROOT / relative_path).read_bytes()).hexdigest() == expected
    assert v3.DEPENDENCY_RECORD["canonical_annotation_module_imported"] is False
    assert v3.DEPENDENCY_RECORD["v2_learner_module_imported_or_executed"] is False
    assert v3.DEPENDENCY_RECORD["canonical_identity_payload_read_count"] == 0

    monkeypatch.setattr(v3, "V2_SOURCE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="frozen dependency changed"):
        v3._authenticate_dependencies()


def test_compiler_grammar_declares_eight_channel_atoms_before_canonical_binding() -> None:
    protocol = v3.PROTOCOL
    channels = tuple(schema.channel for schema in protocol.relay_schemas)
    assert channels == (
        "root-production-family",
        "ordered-parent-child-edge",
        "argument-operand-orientation",
        "nesting-relation",
        "parameter-role-binding",
        "literal-slot-binding",
        "literal-value-copy",
        "terminal-root-eclass-agreement",
    )
    assert len(protocol.grammar_productions) == 15
    assert len(protocol.output_atoms) == 132
    assert len(protocol.output_transitions) == 28
    assert len({row.atom_id for row in protocol.output_atoms}) == 132
    assert len({row.transition_id for row in protocol.output_transitions}) == 28
    assert {row.channel for row in protocol.output_atoms} == set(channels)
    assert {row.channel for row in protocol.output_transitions} == set(channels)
    assert set(v3.ROOT_PRODUCTION_FILLERS) == {
        "add",
        "maximum",
        "minimum",
        "map",
        "sum",
        "count",
        "filter",
        "reverse",
    }
    assert v3.GRAMMAR_RECORD["vocabulary_from_canonical_target_union"] is False
    assert v3.GRAMMAR_RECORD["canonical_target_label_read_count"] == 0
    source = inspect.getsource(v3)
    assert source.index("GRAMMAR_PRODUCTIONS =") < source.index("DEPENDENCY_RECORD =")
    assert "public_train_prompt_selection_v2 as" not in source
    assert "public_train_structural_annotation_capsule_v1 as" not in source


def test_fresh_role_filler_relation_lessons_are_supported_and_counterfactual() -> None:
    protocol = v3.PROTOCOL
    audit = protocol.prelearning_audit
    targets_by_key: dict[
        v3.PhysicalCausalAssemblyKey,
        set[tuple[tuple[str, ...], tuple[str, ...]]],
    ] = defaultdict(set)
    atom_support: Counter[str] = Counter()
    transition_support: Counter[str] = Counter()
    for lesson in protocol.lessons:
        assert lesson.split == v3.SELF_POSED_TRAIN
        assert tuple(lesson.causal_input_record()) == ("ordered_events",)
        assert len(lesson.causal_input.ordered_events) >= 2
        for event in lesson.causal_input.ordered_events:
            assert tuple(event.record()) == (
                "relation_cells",
                "role_cells",
                "filler_cells",
            )
            assert all(
                len(cells) == v3.ASSEMBLY_ACTIVE_CELL_COUNT
                for cells in (event.relation_cells, event.role_cells, event.filler_cells)
            )
        targets_by_key[lesson.causal_input].add(
            (lesson.target_atom_ids, lesson.target_transition_ids)
        )
        atom_support.update(lesson.target_atom_ids)
        transition_support.update(lesson.target_transition_ids)

    assert len(protocol.lessons) == len(targets_by_key) == 138
    assert all(len(targets) == 1 for targets in targets_by_key.values())
    assert min(atom_support.values()) == audit.minimum_atom_support == 2
    assert min(transition_support.values()) == audit.minimum_transition_support == 2
    assert set(atom_support) == {row.atom_id for row in protocol.output_atoms}
    assert set(transition_support) == {row.transition_id for row in protocol.output_transitions}
    assert (
        audit.unsupported_atom_count,
        audit.unsupported_transition_count,
        audit.causal_key_target_collision_count,
    ) == (0, 0, 0)
    assert audit.total_function and audit.full_support and audit.counterfactual_coverage

    order_pairs = tuple(
        row for row in protocol.counterfactual_pairs if row.kind == "event-order-reversal"
    )
    role_pairs = tuple(
        row for row in protocol.counterfactual_pairs if row.kind == "role-filler-swap"
    )
    assert len(order_pairs) == 69
    assert len(role_pairs) == 3
    for pair in order_pairs:
        left = protocol.lessons[pair.left_lesson_index]
        right = protocol.lessons[pair.right_lesson_index]
        assert left.causal_input.cell_multiset() == right.causal_input.cell_multiset()
        assert left.target_atom_ids == right.target_atom_ids
        assert left.target_transition_ids != right.target_transition_ids
    for pair in role_pairs:
        left = protocol.lessons[pair.left_lesson_index]
        right = protocol.lessons[pair.right_lesson_index]
        assert left.causal_input.cell_multiset() == right.causal_input.cell_multiset()
        assert left.target_atom_ids != right.target_atom_ids
        assert left.target_transition_ids == right.target_transition_ids


def test_total_function_falsifier_zero_work_boundary_and_exact_roots() -> None:
    protocol = v3.PROTOCOL
    first = protocol.lessons[0]
    poisoned = dataclasses.replace(
        first,
        lesson_index=len(protocol.lessons),
        target_atom_ids=(protocol.output_atoms[-1].atom_id,),
        sha256="0" * 64,
    )
    poison_audit = v3._audit_curriculum(
        (*protocol.lessons, poisoned), protocol.counterfactual_pairs
    )
    assert poison_audit.status == v3.STOP_TOTAL_FUNCTION
    assert poison_audit.causal_key_target_collision_count == 1
    assert poison_audit.total_function is False

    zero_fields = (
        "dev_input_count",
        "protected_input_count",
        "desired_output_read_count",
        "program_artifact_read_count",
        "executable_source_read_count",
        "ast_artifact_read_count",
        "property_row_read_count",
        "task_identifier_read_count",
        "family_identifier_read_count",
        "entrypoint_spelling_neural_evidence_count",
        "audit_hash_neural_evidence_count",
        "canonical_target_label_read_count",
        "canonical_annotation_import_count",
        "canonical_identity_payload_read_count",
        "learner_array_count",
        "learning_run_count",
        "electrical_execution_count",
        "candidate_count",
        "proof_count",
        "whole_target_class_head_count",
    )
    assert all(v3.SURFACE_BOUNDARY_RECORD[field] == 0 for field in zero_fields)
    assert v3.SURFACE_BOUNDARY_RECORD["data_only"] is True
    assert v3.SURFACE_BOUNDARY_RECORD["learner_or_execution_interface_present"] is False
    assert v3.CONTROL_RECORD["whole_target_label_head"] == "forbidden"
    assert v3.CONTROL_RECORD["canonical_heldout_target_access"] == "forbidden"
    assert v3.LEARNER_GATE_RECORD["licensed_successor_scope"] == (
        "PUBLIC_TRAIN_PROMPT_SELECTION_V3"
    )
    assert v3.LEARNER_GATE_RECORD["canonical_fit_training_identity_count"] == 24
    assert v3.LEARNER_GATE_RECORD["canonical_heldout_training_identity_count"] == 0
    assert v3.LEARNER_GATE_RECORD["canonical_heldout_replay_allowed"] is False
    assert v3.LEARNER_GATE_RECORD["whole_target_class_heads_allowed"] is False
    assert v3.LEARNER_GATE_RECORD["joint_train_integration_licensed"] is False
    assert protocol.status == v3.PASS_PROTOCOL
    assert protocol.root_record() == {
        **{
            key: protocol.root_record()[key]
            for key in protocol.root_record()
            if key not in EXPECTED_ROOTS
        },
        **EXPECTED_ROOTS,
    }
    assert {key: protocol.root_record()[key] for key in EXPECTED_ROOTS} == EXPECTED_ROOTS
    assert protocol.protocol_sha256 == v3._namespaced_sha256(
        v3._PROTOCOL_NAMESPACE,
        protocol.root_record(include_protocol_sha256=False),
    )
    assert protocol == v3._build_protocol()
    with pytest.raises(dataclasses.FrozenInstanceError):
        protocol.status = "STOP"  # type: ignore[misc]

    source = inspect.getsource(v3)
    assert "import numpy" not in source
    assert {
        "fit",
        "learn",
        "execute",
        "select",
        "rank_candidates",
        "build_candidates",
    }.isdisjoint(vars(v3))
