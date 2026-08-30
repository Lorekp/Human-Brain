from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from experiments.g0 import frozen_common_architecture_binding_v1 as binding
from experiments.g0 import frozen_cross_domain_g0_preregistration_v1 as prereg


def test_partial_nine_class_binding_is_authenticated_and_fail_closed() -> None:
    frozen = binding.authenticate_binding()

    assert tuple(row.requirement for row in frozen.classes) == (
        prereg.COMMON_ARCHITECTURE_FREEZE_REQUIREMENTS
    )
    assert tuple(row.ordinal for row in frozen.classes) == tuple(range(1, 10))
    assert tuple(row.ordinal for row in frozen.classes if row.bound) == (2, 3, 7)
    assert frozen.bound_class_count == 3
    assert frozen.missing_class_count == 6
    assert frozen.complete_architecture_binding_count == 0
    assert frozen.generation_authorized is False
    assert frozen.stop_reason == binding.STOP_INCOMPLETE_BINDING
    assert not any(value for _, value in frozen.execution_counts)

    assert frozen.classes[0].status == (
        binding.STOP_MISSING_INTEGRATED_CHECKPOINT
    )
    assert frozen.classes[3].status == binding.STOP_MISSING_ACTION_ASSIGNMENT
    assert frozen.classes[4].status == binding.STOP_MISSING_BASELINE_SOURCES
    assert frozen.classes[5].status == binding.STOP_MISSING_ROLLOUT_SEEDS
    assert frozen.classes[7].status == binding.STOP_MISSING_EVALUATOR
    assert frozen.classes[8].status == binding.STOP_MISSING_DEPENDENCY_LOCK

    with pytest.raises(
        binding.G0CommonArchitectureBindingStop,
        match=binding.STOP_INCOMPLETE_BINDING,
    ):
        binding.require_generation_authorization()

    root = Path(__file__).resolve().parents[1]
    for source in frozen.sources:
        assert sha256((root / source.path).read_bytes()).hexdigest() == (
            source.source_sha256
        )

    assert frozen.sha256 == binding.BINDING.sha256
    assert frozen.claim_boundary == binding.CLAIM_BOUNDARY
