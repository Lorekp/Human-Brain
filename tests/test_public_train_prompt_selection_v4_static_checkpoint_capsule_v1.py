from __future__ import annotations

import ast
import dataclasses
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.r3_pilot import (
    public_train_prompt_selection_v4_static_checkpoint_capsule_v1 as capsule,
)

FORBIDDEN_RUNTIME_MODULES = (
    "experiments.r3_pilot.public_train_prompt_selection_v4",
    "experiments.r3_pilot.public_train_prompt_selection_v4_protocol",
    "experiments.r3_pilot.public_train_prompt_selection_freeze_v1_protocol",
    "experiments.r3_pilot.public_train_structural_annotation_capsule_v1",
    "experiments.r3_pilot.public_train_prompt_factor_manifest_v1",
)


def test_static_artifact_authenticates_banked_source_result_and_checkpoint() -> None:
    assert sha256(capsule.BANKED_SOURCE_PATH.read_bytes()).hexdigest() == (
        capsule.BANKED_SOURCE_SHA256
    )
    assert sha256(capsule.BANKED_TEST_PATH.read_bytes()).hexdigest() == (capsule.BANKED_TEST_SHA256)
    assert sha256(capsule.ARTIFACT_PATH.read_bytes()).hexdigest() == (
        capsule.EXPECTED_ARTIFACT_SHA256
    )
    loaded = capsule.load_checkpoint_capsule()
    assert loaded.artifact_sha256 == capsule.EXPECTED_ARTIFACT_SHA256
    assert loaded.content_root == capsule.EXPECTED_ARTIFACT_CONTENT_ROOT
    assert loaded.checkpoint_sha256 == capsule.BANKED_CHECKPOINT_SHA256
    assert capsule.BANKED_COMMIT == "561a5819c9942a5be08ebabc94255ac463b3e496"
    assert capsule.BANKED_RESULT_SHA256 == (
        "e2249e0277a9ad49ea5eabe452843294ce7c658307e1ad467741a190b0e5fd21"
    )


def test_teacher_free_public_train_input_replays_only_banked_actual_winners() -> None:
    assert not any(name in sys.modules for name in FORBIDDEN_RUNTIME_MODULES)
    spans, literals, signature_role, expected_actual_winners = capsule._load_reproduction_fixture()
    fixture_text = capsule._canonical_json(
        {
            "spans": spans,
            "literals": literals,
            "signature_role": signature_role,
        }
    ).decode("ascii")
    assert all(
        token not in fixture_text
        for token in (
            "expected",
            "teacher",
            "desired",
            "program",
            "task_id",
            "family",
            "causal_key_sha256",
        )
    )

    certificate = capsule.infer_prompt(spans, literals, signature_role)

    assert not any(name in sys.modules for name in FORBIDDEN_RUNTIME_MODULES)
    assert certificate.status == capsule.STOP
    assert certificate.checkpoint_sha256 == capsule.BANKED_CHECKPOINT_SHA256
    assert len(certificate.occurrences) == 15
    assert certificate.terminal_winner is not None
    assert certificate.terminal_winner.decision == capsule.STOP
    assert certificate.actual_winner_sha256 == expected_actual_winners
    assert certificate.actual_winner_sha256 == (
        "cabba67ce5e79aa8079fda13f8a4998bae640b03a75bbebbe39bd3102dc93dd4"
    )
    assert certificate.sha256 == (
        "918e569f4df2a2d6b360a0f0007502577caefdc37aaaf13351b0f98dd209c213"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        certificate.status = capsule.CAP  # type: ignore[misc]


def test_runtime_imports_and_public_certificate_fields_are_target_free() -> None:
    runtime_tree = ast.parse(Path(capsule.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not set(FORBIDDEN_RUNTIME_MODULES).intersection(imported)

    exporter_path = Path(capsule.__file__).with_name(
        "export_public_train_prompt_selection_v4_static_checkpoint_capsule_v1.py"
    )
    exporter_tree = ast.parse(exporter_path.read_text(encoding="utf-8"))
    top_level_imports = {
        node.module
        for node in exporter_tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert FORBIDDEN_RUNTIME_MODULES[0] not in top_level_imports

    for public_type in (
        capsule.PromptPhysicalWinner,
        capsule.PromptOccurrenceBinding,
        capsule.PromptStructuralOccurrence,
        capsule.PromptStructuralCertificate,
    ):
        field_names = {field.name for field in dataclasses.fields(public_type)}
        assert not any(
            token in field_name
            for field_name in field_names
            for token in ("expected", "teacher", "target")
        )
