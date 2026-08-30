from __future__ import annotations

from dataclasses import replace

import pytest
from experiments.g0 import frozen_g0_dependency_environment_v1 as environment


def test_g0_dependency_lock_and_environment_identity_are_exact_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = environment.authenticate_dependency_environment()
    assert capsule.status == environment.STATUS
    assert capsule.requirement == "dependency-lock-and-environment-or-container-identity"
    assert capsule.active_project_extras == ("g0",)
    assert capsule.root_requirements == (
        "human-cell-brain==0.1.0",
        "minigrid==3.1.0",
    )
    assert len(capsule.locked_distributions) == 7
    assert {row.name for row in capsule.locked_distributions} == {
        "Farama-Notifications",
        "cloudpickle",
        "gymnasium",
        "minigrid",
        "numpy",
        "pygame-ce",
        "typing_extensions",
    }
    assert capsule.environment_identity.implementation == "CPython"
    assert all(value == 0 for _name, value in capsule.execution_counts)

    changed = replace(environment.ENVIRONMENT_IDENTITY, python_version="0.0.0")
    monkeypatch.setattr(environment, "_runtime_environment_identity", lambda: changed)
    with pytest.raises(
        environment.G0DependencyEnvironmentStop,
        match="STOP_G0_RUNTIME_ENVIRONMENT_CHANGED",
    ):
        environment.authenticate_dependency_environment()
