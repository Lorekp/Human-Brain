from __future__ import annotations

import ast
import dataclasses
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.g0 import spatial_candidate_adapter_v1 as adapter
from experiments.g0 import spatial_fixture_generator_v1 as generator


def test_quarantined_spatial_fixture_round_trip_and_frozen_cohorts_stop() -> None:
    assert generator.PREREGISTRATION_COMMIT == "42278a7"
    assert generator.PREREGISTRATION_PROTOCOL_SHA256 == (
        "955828b48591f198047d5ca4966885e253763d6ea353351585edb14390320468"
    )
    assert generator.PREREGISTRATION_ADAPTER_SHA256 == (
        "29a6dcec69d89c74c373e7f4bf8709e5e57d7cc06d3aa15f8ad3a2691db1fd99"
    )
    assert generator.PREREGISTRATION_SPATIAL_SHA256 == (
        "8705597cd610e00390a73bcd926fc8b3bcbfbd8be07b08646214972ad8402100"
    )
    assert len(generator.REQUIRED_BINDING_ARTIFACTS) == 9

    adapter_source = Path(adapter.__file__).read_text(encoding="utf-8")
    adapter_tree = ast.parse(adapter_source)
    imported_modules: set[str] = set()
    function_names: set[str] = set()
    for node in ast.walk(adapter_tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.add(node.name)
    assert imported_modules == {
        "__future__",
        "collections.abc",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "struct",
    }
    assert not any(
        name.startswith(("generate", "evaluate", "learn", "solve", "oracle"))
        for name in function_names
    )

    assert adapter.IMAGE_SHAPE == (7, 7, 3)
    assert adapter.IMAGE_BYTE_COUNT == 147
    assert adapter.CANDIDATE_FIELDS == (
        "image",
        "direction",
        "action_id",
        "reward",
        "terminated",
        "truncated",
    )
    assert tuple(range(7)) == adapter.RAW_ACTION_IDS
    assert adapter.SPATIAL_FORBIDDEN_FIELDS == (
        "mission",
        "full-grid",
        "absolute-agent-position",
        "visibility-mask",
        "shortest-path",
        "object-tracking-identity",
        "solver-state",
        "codebook-inverse",
        "layout-or-family-identity",
        "oracle-actions",
    )

    seed = generator.QUARANTINED_FIXTURE_SEEDS[0]
    first = generator.generate_quarantined_fixture(seed)
    second = generator.generate_quarantined_fixture(seed)
    assert first == second
    assert first.quarantined is True
    assert first.is_frozen_cohort is False
    assert first.fixture_execution_count == 1
    assert first.frozen_cohort_generation_count == 0
    assert first.step_count == 7
    assert first.raw_action_ids == tuple(range(7))
    assert first.candidate_transcript_sha256 == sha256(
        first.candidate_transcript
    ).hexdigest()
    assert first.fixture_sha256 == sha256(
        generator._FIXTURE_NAMESPACE
        + generator._canonical_json(first.root_record(include_fixture_sha256=False))
    ).hexdigest()

    decoded = adapter.decode_transcript(first.candidate_transcript)
    assert adapter.encode_transcript(decoded) == first.candidate_transcript
    assert tuple(step.action_id for step in decoded) == tuple(range(7))
    assert all(len(step.image) == 147 for step in decoded)
    assert all(step.direction in range(4) for step in decoded)
    assert tuple(step.reward for step in decoded) == (0.0,) * 6 + (1.0,)
    assert tuple(step.terminated for step in decoded) == (False,) * 6 + (True,)
    assert not any(step.truncated for step in decoded)

    clean_record = decoded[0].record()
    audit = adapter.audit_candidate_record(clean_record)
    assert audit.permitted_fields == adapter.CANDIDATE_FIELDS
    assert audit.checked_forbidden_fields == adapter.SPATIAL_FORBIDDEN_FIELDS
    assert audit.detected_forbidden_fields == ()
    assert audit.unknown_fields == ()
    assert audit.missing_fields == ()
    assert audit.input_field_count == 6
    assert first.negative_field_audit_sha256 == audit.sha256
    for forbidden_field in adapter.SPATIAL_FORBIDDEN_FIELDS:
        poisoned = dict(clean_record)
        poisoned[forbidden_field.replace("-", "_")] = object()
        with pytest.raises(
            adapter.CandidateSurfaceError,
            match=r"^STOP_G0_SPATIAL_CANDIDATE_SURFACE$",
        ):
            adapter.audit_candidate_record(poisoned)
    with pytest.raises(adapter.CandidateSurfaceError):
        adapter.audit_candidate_record({**clean_record, "time": 0})
    missing = dict(clean_record)
    missing.pop("direction")
    with pytest.raises(adapter.CandidateSurfaceError):
        adapter.audit_candidate_record(missing)

    for forbidden_field in adapter.SPATIAL_FORBIDDEN_FIELDS:
        assert forbidden_field.encode("ascii") not in first.candidate_transcript
    assert str(seed).encode("ascii") not in first.candidate_transcript
    assert first.fixture_name.encode("ascii") not in first.candidate_transcript

    assert dict(generator.SURFACE_BOUNDARY_RECORD) == {
        "common_architecture_binding_count": 0,
        "frozen_cohort_generation_count": 0,
        "learner_execution_count": 0,
        "evaluator_execution_count": 0,
        "outcome_count": 0,
        "dev_input_count": 0,
        "m0_input_count": 0,
        "protected_input_count": 0,
        "frozen_cohort_generation_authorized": False,
        "quarantined_fixture_generation_authorized": True,
        "nine_artifact_binding_required": True,
    }
    forged_binding = {name: "0" * 64 for name in generator.REQUIRED_BINDING_ARTIFACTS}
    for surface in generator.FROZEN_COHORT_SURFACES:
        with pytest.raises(
            generator.G0CommonArchitectureBindingRequired,
            match=r"^STOP_G0_COMMON_ARCHITECTURE_BINDING_REQUIRED$",
        ):
            generator.generate_frozen_spatial_cohort(
                surface=surface,
                architecture_binding=forged_binding,
            )
    with pytest.raises(generator.G0FixtureBoundaryError):
        generator.generate_quarantined_fixture(-1)

    with pytest.raises(dataclasses.FrozenInstanceError):
        first.quarantined = False  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        decoded[0].direction = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        generator.SURFACE_BOUNDARY_RECORD["outcome_count"] = 1  # type: ignore[index]
