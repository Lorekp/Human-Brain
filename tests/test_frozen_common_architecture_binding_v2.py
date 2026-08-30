from __future__ import annotations

import copy
import pickle
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.g0 import frozen_common_architecture_binding_v1 as binding_v1
from experiments.g0 import frozen_common_architecture_binding_v2 as binding
from experiments.g0 import frozen_cross_domain_g0_preregistration_v1 as prereg


def test_complete_nine_class_binding_issues_only_one_exact_opaque_authority() -> None:
    frozen = binding.authenticate_binding()

    assert tuple(row.requirement for row in frozen.classes) == (
        prereg.COMMON_ARCHITECTURE_FREEZE_REQUIREMENTS
    )
    assert tuple(row.ordinal for row in frozen.classes) == tuple(range(1, 10))
    assert all(row.bound for row in frozen.classes)
    assert frozen.bound_class_count == 9
    assert frozen.missing_class_count == 0
    assert frozen.complete_architecture_binding_count == 1
    assert frozen.generation_authorized is True
    assert frozen.generation_authorization_object_count == 1
    assert not any(value for _, value in frozen.domain_specific_primitive_ledger)
    assert not any(value for _, value in frozen.execution_counts)

    assert frozen.classes[0].artifact_sha256s[0] == (
        "architecture-source-commit",
        "20ee0eae68d698b5234b4cf1ebbb98e3cfd8c103",
    )
    assert frozen.classes[3].artifact_sha256s[0] == (
        "action-assignment",
        binding.ACTION_ASSIGNMENT_SHA256,
    )
    assert frozen.classes[4].artifact_sha256s[:2] == (
        ("simple-baselines", binding.BASELINE_SHA256),
        ("focused-ablations", binding.ABLATION_SHA256),
    )
    assert frozen.classes[5].artifact_sha256s[0] == (
        "seed-derivation",
        binding.SEED_DERIVATION_SHA256,
    )
    assert frozen.classes[7].artifact_sha256s[0] == (
        "evaluator",
        binding.EVALUATOR_SHA256,
    )
    assert frozen.classes[8].artifact_sha256s[:3] == (
        (
            "source-commit",
            "5892aa3e859b7e42c8ba26b4382d55141844d77c",
        ),
        (
            "source",
            "e2c86d3fa08927bbcba3540c67a9e77ae0cc8aa46ea4b50ef79bf672eae8a3e6",
        ),
        (
            "capsule",
            "ce8ac9aa502183eb00fa2ed89c2c31065450af1c92a6149942fcff9d9b32494f",
        ),
    )

    authority = binding.issue_generation_authorization()
    assert binding.issue_generation_authorization() is authority
    assert binding.require_generation_authorization(authority) is frozen
    assert "opaque exact" in repr(authority)
    with pytest.raises(
        binding.G0CommonArchitectureBindingV2Stop,
        match="STOP_G0_BINDING_V2_EXACT_AUTHORIZATION_OBJECT_REQUIRED",
    ):
        binding.require_generation_authorization(object())
    with pytest.raises(TypeError, match="not copyable"):
        copy.copy(authority)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(authority)

    terminal_v1 = binding_v1.authenticate_binding()
    assert terminal_v1.bound_class_count == 3
    assert terminal_v1.missing_class_count == 6
    assert terminal_v1.generation_authorized is False
    assert terminal_v1.sha256 == frozen.v1_terminal_binding_sha256

    root = Path(__file__).resolve().parents[1]
    for source in frozen.sources:
        assert sha256((root / source.path).read_bytes()).hexdigest() == (
            source.source_sha256
        )
    assert sha256(Path(binding.__file__).read_bytes()).hexdigest() == (
        binding.SOURCE_SHA256
    )
