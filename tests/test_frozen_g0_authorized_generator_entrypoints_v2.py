from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.g0 import frozen_common_architecture_binding_v2 as binding
from experiments.g0 import frozen_g0_authorized_generator_entrypoints_v2 as entrypoints
from experiments.g0 import frozen_regulatory_generator_v1 as regulatory
from experiments.g0 import frozen_spatial_environment_generator_v1 as spatial


def test_exact_v2_authority_freezes_both_plans_without_calling_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden_call(*_args: object, **_kwargs: object) -> None:
        calls.append("generator-called")
        raise AssertionError("a generation plan must not call a generator")

    monkeypatch.setattr(regulatory, "_generate_system", forbidden_call)
    monkeypatch.setattr(regulatory, "generate_frozen_cohort", forbidden_call)
    monkeypatch.setattr(spatial, "_make_environment", forbidden_call)
    monkeypatch.setattr(spatial, "generate_frozen_spatial_cohort", forbidden_call)

    authority = binding.issue_generation_authorization()
    regulatory_plan = entrypoints.freeze_regulatory_generation_plan(authority)
    spatial_plan = entrypoints.freeze_spatial_generation_plan(authority)
    assert regulatory_plan is entrypoints.REGULATORY_PLAN
    assert spatial_plan is entrypoints.SPATIAL_PLAN
    assert (regulatory_plan.domain, spatial_plan.domain) == (
        "regulatory",
        "spatial",
    )
    assert regulatory_plan.binding_source_commit == (
        "d987edde74c6cfd704afaa03ec5d9248bef22c65"
    )
    assert regulatory_plan.binding_sha256 == spatial_plan.binding_sha256 == (
        "d496be2b5b8cadb980bb5ed2f8e423e01b965bb743f0d1f36a293274e06e5d01"
    )
    assert regulatory_plan.frozen_cohort_seeds == regulatory.FROZEN_COHORT_SEEDS
    assert spatial_plan.frozen_cohort_seeds == spatial.FROZEN_COHORT_SEEDS
    assert regulatory_plan.generator_invocation_count == 0
    assert spatial_plan.generator_invocation_count == 0
    assert not any(value for _, value in regulatory_plan.execution_counts)
    assert not any(value for _, value in spatial_plan.execution_counts)

    class EqualToEverything:
        def __eq__(self, _other: object) -> bool:
            return True

    forged_copy = object.__new__(type(authority))
    for forged in (object(), EqualToEverything(), forged_copy):
        with pytest.raises(
            entrypoints.G0AuthorizedGeneratorEntrypointStop,
            match="STOP_G0_AUTHORIZED_ENTRYPOINT_EXACT_BINDING_OBJECT_REQUIRED",
        ):
            entrypoints.freeze_regulatory_generation_plan(forged)
        with pytest.raises(
            entrypoints.G0AuthorizedGeneratorEntrypointStop,
            match="STOP_G0_AUTHORIZED_ENTRYPOINT_EXACT_BINDING_OBJECT_REQUIRED",
        ):
            entrypoints.freeze_spatial_generation_plan(forged)
    with pytest.raises(TypeError, match="not copyable"):
        copy.copy(authority)
    with pytest.raises(FrozenInstanceError):
        regulatory_plan.domain = "spatial"

    assert calls == []
    assert regulatory.GENERATION_AUTHORIZED is False
    assert regulatory.FROZEN_COHORT_EXECUTION_COUNT == 0
    assert spatial.FROZEN_COHORT_EXECUTION_COUNT == 0
    assert spatial.FROZEN_LAYOUT_GENERATION_COUNT == 0
    assert not any(value for _, value in entrypoints.ZERO_EXECUTION_COUNTS)

    root = Path(__file__).resolve().parents[1]
    assert sha256(Path(entrypoints.__file__).read_bytes()).hexdigest() == (
        entrypoints.SOURCE_SHA256
    )
    for identity in entrypoints.GENERATOR_IDENTITIES.values():
        assert sha256((root / identity["source_path"]).read_bytes()).hexdigest() == (
            identity["source_sha256"]
        )
