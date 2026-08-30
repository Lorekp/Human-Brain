from __future__ import annotations

import dataclasses
import inspect
from collections import defaultdict
from collections.abc import Mapping

import pytest
from experiments.r3_pilot import public_train_factor_workspace_v2_protocol as factor_v2
from experiments.r3_pilot import public_train_prompt_selection_freeze_v1_protocol as v1

EXPECTED_ROOTS = {
    "dependency_sha256": "57db75d04bba1739734bf122cbd1a860886c0fcc50ec8095b3b5b1932788d6ae",
    "signature_bindings_sha256": (
        "17136ed39594e0e4036c16324034582668058061d3db5752ace01bf46334b706"
    ),
    "canonical_inputs_sha256": ("c27732aca374fbe0c71e11449701f4773db05e91663db5bcbc3bf3c5d16ae4ec"),
    "factor_annotations_sha256": (
        "1cec652f746daf3c75e0a6c92eb284199e8786e017ba0229fbb1004e9418661a"
    ),
    "relay_target_vocabulary_sha256": (
        "9a38415544ab94c20d0be52ffdda1557eaf791cfb6521541ef2deea9b521ab5b"
    ),
    "annotation_schema_sha256": (
        "a6d0101d0ab9617cc29f17b246c222b2dec31acb6dff0550ec3b5951bb57bedb"
    ),
    "identifiability_audit_sha256": (
        "e9e1f6fd212bec4530d09b2cb890eb973db25304c4a086982a0f3265f9eb48e3"
    ),
    "canonical_split_sha256": ("ca2c82ffe4a45f257ea7d2d556ad40eae3de32642533eabe6e4f54398159dcf5"),
    "surface_boundary_sha256": ("ba1441ef61560b0c7380a5e207e2e65e58912fccc416d1ea73deffe0ca47626f"),
    "protocol_sha256": "dc6cb9afd1ce92e15cfeca3545f3b0c26ec041a71c040dae686f76acfcf3ff47",
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


def test_full_signatures_are_bound_but_only_role_events_enter_causal_keys() -> None:
    bindings = v1.PROTOCOL.signature_bindings
    assert len(bindings) == v1.PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT == 90
    assert len({binding.typed_signature_sha256 for binding in bindings}) == 90
    assert len({binding.signature_role_sha256 for binding in bindings}) == 4
    assert len({binding.causal_key.sha256 for binding in bindings}) == 42

    by_prompt_input: dict[str, list[v1.SignatureBoundPromptAppearance]] = defaultdict(list)
    for binding in bindings:
        by_prompt_input[binding.prompt_input_sha256].append(binding)
        assert set(binding.typed_signature_record) == {
            "entry_point",
            "parameters",
            "result",
        }
        assert set(binding.signature_role_record) == {
            "parameters",
            "result",
            "role_binding",
            "entrypoint_spelling_present",
            "parameter_spelling_present",
        }
        assert binding.signature_role_record["entrypoint_spelling_present"] is False
        assert binding.signature_role_record["parameter_spelling_present"] is False
        assert set(binding.causal_key.record()) == {
            "physical_span_events_sha256",
            "typed_literal_events_sha256",
            "signature_role_events_sha256",
            "causal_sources",
            "sha256",
        }
    assert len(by_prompt_input) == 42
    assert all(
        len({row.causal_key.sha256 for row in rows}) == 1 for rows in by_prompt_input.values()
    )
    assert all(
        len({row.signature_role_sha256 for row in rows}) == 1 for rows in by_prompt_input.values()
    )

    with pytest.raises(TypeError):
        bindings[0].typed_signature_record["entry_point"] = "forged"  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        bindings[0].prompt_input_sha256 = "0" * 64  # type: ignore[misc]


def test_exact_42_identity_24_18_inventory_and_factor_protocol_binding() -> None:
    protocol = v1.PROTOCOL
    canonical = protocol.canonical_inputs
    frozen_factor_rows = factor_v2.PROTOCOL.canonical_identities
    assert len(canonical) == v1.CANONICAL_SEMANTIC_IDENTITY_COUNT == 42
    assert sum(row.appearance_count for row in canonical) == 90
    assert len({row.causal_key_sha256 for row in canonical}) == 42
    assert tuple(row.prompt_input_sha256 for row in canonical) == tuple(
        row.prompt_input_sha256 for row in frozen_factor_rows
    )
    assert tuple(row.appearance_count for row in canonical) == tuple(
        row.appearance_count for row in frozen_factor_rows
    )
    assert sum(row.split == factor_v2.FIT for row in canonical) == 24
    assert sum(row.split == factor_v2.HELDOUT for row in canonical) == 18
    assert (
        tuple(row.prompt_input_sha256 for row in canonical if row.split == factor_v2.FIT)
        == factor_v2.PROTOCOL.canonical_fit_prompt_input_sha256s
    )
    assert tuple(
        row.prompt_input_sha256 for row in canonical if row.split == factor_v2.HELDOUT
    ) == tuple(
        row.prompt_input_sha256
        for row in frozen_factor_rows
        if row.prompt_input_sha256 in set(factor_v2.PROTOCOL.canonical_heldout_prompt_input_sha256s)
    )

    annotations = {row.prompt_input_sha256: row for row in protocol.factor_annotations}
    for frozen in frozen_factor_rows:
        assert (
            tuple(target.factor for target in annotations[frozen.prompt_input_sha256].targets)
            == frozen.target_factors
        )
    assert v1.DEPENDENCY_RECORD["factor_workspace_protocol_sha256"] == (
        factor_v2.PROTOCOL.protocol_sha256
    )
    assert factor_v2.PROTOCOL.protocol_sha256 == (v1.EXPECTED_FACTOR_WORKSPACE_PROTOCOL_SHA256)
    assert v1.DEPENDENCY_RECORD["factor_workspace_protocol_status"] == ("FROZEN_DATA_ONLY_NOT_RUN")
    assert v1.DEPENDENCY_RECORD["factor_workspace_learner_pass_assumed"] is False


def test_exact_structural_target_absence_freezes_the_prelearning_stop() -> None:
    protocol = v1.PROTOCOL
    channels = tuple(channel.channel for channel in protocol.relay_target_vocabulary)
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
    assert tuple(v1.ANNOTATION_SCHEMA_RECORD["unavailable_exact_relay_channels"]) == channels
    assert tuple(v1.ANNOTATION_SCHEMA_RECORD["available_frozen_semantic_annotation_fields"]) == (
        "start_token",
        "stop_token",
        "span_identity_sha256",
        "factor",
    )
    assert v1.ANNOTATION_SCHEMA_RECORD["invented_structural_labels_forbidden"] is True
    assert v1.ANNOTATION_SCHEMA_RECORD["program_artifact_recovery_forbidden"] is True

    audit = protocol.identifiability_audit
    assert protocol.status == audit.status == (v1.STOP_EXACT_STRUCTURAL_RELAY_TARGETS_UNAVAILABLE)
    assert audit.prompt_appearance_count == 90
    assert audit.canonical_causal_identity_count == 42
    assert audit.source_factor_annotation_set_count == 42
    assert audit.source_factor_annotation_alias_count == 0
    assert audit.required_exact_relay_target_set_count == 42
    assert audit.available_exact_relay_target_set_count == 0
    assert audit.missing_relay_channels == channels
    assert audit.causal_key_relay_target_collision_audit_completed is False
    assert "inventing labels" in audit.reason
    assert "forbidden program artifacts" in audit.reason


def test_protocol_is_exact_immutable_and_data_only() -> None:
    protocol = v1.PROTOCOL
    surface = dict(v1.SURFACE_BOUNDARY_RECORD)
    zero_fields = (
        "dev_input_count",
        "protected_input_count",
        "desired_output_read_count",
        "program_artifact_read_count",
        "property_row_read_count",
        "task_identifier_read_count",
        "family_identifier_read_count",
        "entrypoint_spelling_neural_evidence_count",
        "audit_hash_neural_evidence_count",
        "exact_relay_target_set_count",
        "learner_array_count",
        "learning_run_count",
        "electrical_execution_count",
        "candidate_count",
        "proof_count",
    )
    assert all(surface[field] == 0 for field in zero_fields)
    assert surface["raw_prompt_text_retained_count"] == 0
    assert surface["learner_or_selection_interface_present"] is False
    assert not any(type(value).__module__.startswith("numpy") for value in _walk(protocol))

    public_functions = {
        name
        for name, value in inspect.getmembers(v1, inspect.isfunction)
        if not name.startswith("_") and value.__module__ == v1.__name__
    }
    assert public_functions == set()
    assert {"fit", "learn", "execute", "select", "build_candidates"}.isdisjoint(vars(v1))

    actual_roots = {
        field.name: getattr(protocol, field.name)
        for field in dataclasses.fields(protocol)
        if field.name.endswith("sha256")
    }
    assert actual_roots == EXPECTED_ROOTS
    assert protocol.dependency_sha256 == v1._namespaced_sha256(
        v1._DEPENDENCY_NAMESPACE,
        dict(v1.DEPENDENCY_RECORD),
    )
    assert protocol.canonical_split_sha256 == v1._namespaced_sha256(
        v1._SPLIT_NAMESPACE,
        dict(v1.CANONICAL_SPLIT_RECORD),
    )
    assert protocol.surface_boundary_sha256 == v1._namespaced_sha256(
        v1._SURFACE_NAMESPACE,
        dict(v1.SURFACE_BOUNDARY_RECORD),
    )
    assert protocol.protocol_sha256 == v1._namespaced_sha256(
        v1._PROTOCOL_NAMESPACE,
        protocol.root_record(include_protocol_sha256=False),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        protocol.status = "READY"  # type: ignore[misc]
