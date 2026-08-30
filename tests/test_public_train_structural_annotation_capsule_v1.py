from __future__ import annotations

import dataclasses
import inspect
from collections import Counter
from collections.abc import Mapping
from hashlib import sha256

import pytest
from experiments.r3_pilot import public_train_prompt_selection_freeze_v1_protocol as selection_v1
from experiments.r3_pilot import public_train_structural_annotation_capsule_v1 as v1

EXPECTED_ROOTS = {
    "dependency_sha256": "000a432ff9d6b81a40f8106db4f600cbc73630e33a03767b8afa2994cf0c8579",
    "teacher_projection_sha256": (
        "e9c5177a8e9ad9d455b212cbe84028320dd9b9c7ff76979c934438544963d7ac"
    ),
    "identifiability_audit_sha256": (
        "abf89bddf0de7ae3672fc01278fd3b62b2256829c59db4c61ec4ee68c2808a5b"
    ),
    "surface_boundary_sha256": ("8ce531f5e02d19c41700f884bc614d9c49c5bdb090bc58413bf00241be96ddbb"),
    "capsule_sha256": "484521f34a98b53f4e21a1d279510c07165fc65f8de72b54845c06a831762caf",
}


def _walk(value: object):
    yield value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _walk(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk(key)
            yield from _walk(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item)


def test_compiler_teacher_projection_is_pinned_auditable_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = v1._ROOT / v1.COMPILER_TEACHER_SOURCE_PATH
    assert sha256(source.read_bytes()).hexdigest() == v1.COMPILER_TEACHER_SOURCE_SHA256
    assert v1.COMPILER_TEACHER_SOURCE_COMMIT == ("0121fd279b3dd9dda59776a31b9fe7a12f774217")
    assert v1.COMPILER_TEACHER_PROJECTION_COUNT == 13
    assert selection_v1.PROTOCOL.protocol_sha256 == (v1.EXPECTED_SELECTION_PROTOCOL_SHA256)
    assert {
        "selection_protocol_scope": selection_v1.PROTOCOL.scope,
        "selection_protocol_status": selection_v1.PROTOCOL.status,
        "selection_protocol_sha256": selection_v1.PROTOCOL.protocol_sha256,
        "signature_bindings_sha256": selection_v1.PROTOCOL.signature_bindings_sha256,
        "canonical_inputs_sha256": selection_v1.PROTOCOL.canonical_inputs_sha256,
        "relay_target_vocabulary_sha256": (selection_v1.PROTOCOL.relay_target_vocabulary_sha256),
        "compiler_teacher_source_commit": v1.COMPILER_TEACHER_SOURCE_COMMIT,
        "compiler_teacher_source_path": v1.COMPILER_TEACHER_SOURCE_PATH,
        "compiler_teacher_source_sha256": v1.COMPILER_TEACHER_SOURCE_SHA256,
        "compiler_teacher_source_bytes_hashed_only": True,
        "compiler_teacher_module_imported": False,
        "task_generator_executed": False,
        "oracle_executed": False,
        "private_rows_materialized": False,
        "dependency_metadata_is_not_learner_evidence": True,
    } == v1.DEPENDENCY_RECORD

    monkeypatch.setattr(v1, "COMPILER_TEACHER_SOURCE_SHA256", "0" * 64)
    with pytest.raises(
        RuntimeError,
        match="compiler-owned public-TRAIN semantic teacher source changed",
    ):
        v1._authenticate_dependencies()


def test_exact_42_key_total_function_and_eight_channel_collision_audit() -> None:
    capsule = v1.CAPSULE
    audit = capsule.identifiability_audit
    expected_channels = tuple(channel.channel for channel in selection_v1.RELAY_TARGET_VOCABULARY)
    assert capsule.status == audit.status == v1.PASS_IDENTIFIABLE
    assert len(capsule.annotations) == audit.canonical_causal_identity_count == 42
    assert audit.prompt_appearance_count == 90
    assert audit.duplicate_prompt_appearance_count == 48
    assert audit.exact_target_set_count == 42
    assert len({row.causal_key_sha256 for row in capsule.annotations}) == 42
    assert len({row.target_set_sha256 for row in capsule.annotations}) == 13
    assert set(row.causal_key_sha256 for row in capsule.annotations) == set(
        row.causal_key_sha256 for row in selection_v1.PROTOCOL.canonical_inputs
    )
    assert v1.CANONICAL_FIT_IDENTITY_COUNT == 24
    assert v1.CANONICAL_HELDOUT_IDENTITY_COUNT == 18
    assert (
        audit.missing_causal_key_count,
        audit.extra_causal_key_count,
        audit.appearance_target_mismatch_count,
        audit.causal_key_target_collision_count,
    ) == (0, 0, 0, 0)
    assert audit.total_function_completed is True
    assert audit.collision_audit_completed is True
    assert tuple(channel for channel, _count in audit.relay_channel_counts) == (expected_channels)
    assert dict(audit.relay_channel_counts) == {
        "root-production-family": 42,
        "ordered-parent-child-edge": 214,
        "argument-operand-orientation": 176,
        "nesting-relation": 84,
        "parameter-role-binding": 42,
        "literal-slot-binding": 61,
        "literal-value-copy": 61,
        "terminal-root-eclass-agreement": 42,
    }
    assert {label.channel for row in capsule.annotations for label in row.relay_labels} == set(
        expected_channels
    )


def test_generic_relay_labels_preserve_root_orientation_and_copy_roles() -> None:
    roots: Counter[tuple[object, object]] = Counter()
    ascii_copy_roles: Counter[object] = Counter()
    count_rows = 0
    for annotation in v1.CAPSULE.annotations:
        root = next(
            label for label in annotation.relay_labels if label.channel == "root-production-family"
        )
        roots[(root.fields["root_production"], root.fields["root_family"])] += 1
        if root.fields["root_production"] == "count":
            count_rows += 1
            orientations = {
                (
                    label.fields["operator_slot"],
                    label.fields["operand_ordinal"],
                    label.fields["source_role"],
                )
                for label in annotation.relay_labels
                if label.channel == "argument-operand-orientation"
            }
            assert (
                "greater-than-as-less-than",
                0,
                "threshold-literal",
            ) in orientations
            assert (
                "greater-than-as-less-than",
                1,
                "nested-add",
            ) in orientations
        for label in annotation.relay_labels:
            if label.channel != "literal-value-copy":
                continue
            role = label.fields["typed_output_copy_role"]
            if str(role).startswith("emit-ascii-class-"):
                ascii_copy_roles[role] += 1

    assert roots == Counter(
        {
            ("minimum", "scalar-extremum"): 5,
            ("add", "scalar-additive"): 9,
            ("maximum", "scalar-extremum"): 1,
            ("map", "list-map"): 6,
            ("sum", "list-fold"): 7,
            ("count", "list-count"): 8,
            ("filter", "string-filter"): 3,
            ("reverse", "string-transform"): 3,
        }
    )
    assert count_rows == 8
    assert ascii_copy_roles == Counter(
        {
            "emit-ascii-class-ascii-decimal-digits": 2,
            "emit-ascii-class-ascii-lowercase": 2,
            "emit-ascii-class-english-vowels": 2,
        }
    )


def test_capsule_exports_only_data_is_immutable_and_has_exact_roots() -> None:
    capsule = v1.CAPSULE
    surface = dict(v1.SURFACE_BOUNDARY_RECORD)
    zero_fields = (
        "raw_prompt_text_export_count",
        "full_typed_signature_export_count",
        "role_normalized_prompt_input_sha256_export_count",
        "dev_input_count",
        "protected_input_count",
        "desired_output_read_count",
        "program_artifact_export_count",
        "executable_source_export_count",
        "ast_artifact_export_count",
        "property_row_read_count",
        "task_identifier_export_count",
        "family_identifier_export_count",
        "entrypoint_spelling_neural_evidence_count",
        "audit_hash_neural_evidence_count",
        "candidate_ranking_count",
        "learner_array_count",
        "learning_run_count",
        "electrical_execution_count",
        "candidate_count",
        "proof_count",
    )
    assert all(surface[field] == 0 for field in zero_fields)
    assert surface["learner_or_execution_interface_present"] is False
    assert tuple(field.name for field in dataclasses.fields(v1.StructuralAnnotation)) == (
        "causal_key_sha256",
        "relay_labels",
        "target_set_sha256",
        "sha256",
    )
    forbidden_learner_fields = {
        "prompt",
        "typed_signature",
        "desired",
        "program",
        "source",
        "ast",
        "property",
        "task_id",
        "family_id",
        "entry_point",
    }
    assert all(
        forbidden_learner_fields.isdisjoint(row.learner_record()) for row in capsule.annotations
    )
    assert not any(type(value).__module__.startswith("numpy") for value in _walk(capsule))
    public_functions = {
        name
        for name, value in inspect.getmembers(v1, inspect.isfunction)
        if not name.startswith("_") and value.__module__ == v1.__name__
    }
    assert public_functions == set()
    assert {
        "fit",
        "learn",
        "execute",
        "select",
        "rank_candidates",
        "build_candidates",
    }.isdisjoint(vars(v1))
    actual_roots = {
        field.name: getattr(capsule, field.name)
        for field in dataclasses.fields(capsule)
        if field.name.endswith("sha256")
    }
    assert actual_roots == EXPECTED_ROOTS
    assert capsule.dependency_sha256 == v1._namespaced_sha256(
        v1._DEPENDENCY_NAMESPACE,
        dict(v1.DEPENDENCY_RECORD),
    )
    assert capsule.teacher_projection_sha256 == v1._namespaced_sha256(
        v1._PROJECTION_NAMESPACE,
        tuple(row.learner_record() for row in capsule.annotations),
    )
    assert capsule.identifiability_audit_sha256 == v1._namespaced_sha256(
        v1._AUDIT_NAMESPACE,
        capsule.identifiability_audit.record(include_sha256=False),
    )
    assert capsule.surface_boundary_sha256 == v1._namespaced_sha256(
        v1._SURFACE_NAMESPACE,
        dict(v1.SURFACE_BOUNDARY_RECORD),
    )
    assert capsule.capsule_sha256 == v1._namespaced_sha256(
        v1._CAPSULE_NAMESPACE,
        capsule.root_record(include_capsule_sha256=False),
    )
    with pytest.raises(TypeError):
        v1.DEPENDENCY_RECORD["oracle_executed"] = True  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        capsule.status = "READY"  # type: ignore[misc]
