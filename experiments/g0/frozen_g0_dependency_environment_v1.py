"""Exact G0 runtime dependency lock and local environment identity.

This data-only capsule closes the ninth preregistered architecture-binding
class.  It follows the established ``pip freeze``/installed-distribution
pattern, but narrows the lock to the active ``base + g0`` runtime closure.
Optional SWE, visualizer, gate, and test dependencies are not part of a G0
adjudicative process and therefore cannot silently enter its environment.

Importing this module does not generate a cohort, execute a learner, evaluate
an outcome, or run a bootstrap.  ``authenticate_dependency_environment`` is a
fail-closed pre-generation check over the project manifest, exact installed
wheel RECORD identities, and the CPython/OS executable identity.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
import sysconfig
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

from experiments.g0 import frozen_cross_domain_g0_preregistration_v1 as prereg

SCHEMA = "FROZEN_G0_DEPENDENCY_ENVIRONMENT_V1"
STATUS = "BOUND_G0_DEPENDENCY_LOCK_AND_ENVIRONMENT_IDENTITY_V1"
REQUIREMENT = prereg.COMMON_ARCHITECTURE_FREEZE_REQUIREMENTS[8]
CLAIM_BOUNDARY = (
    "exact local base+g0 dependency and environment identity only; no common "
    "architecture checkpoint, generation, learner, evaluator, bootstrap, or outcome"
)

PROJECT_MANIFEST_SHA256 = "70fd2232f335929385f4b61e482aa714bc4baf8a706aaa825fa126427f23f77e"
ACTIVE_PROJECT_EXTRAS = ("g0",)
ROOT_REQUIREMENTS = ("human-cell-brain==0.1.0", "minigrid==3.1.0")

GENERATOR_EXECUTION_COUNT = 0
COHORT_GENERATION_COUNT = 0
LEARNER_EXECUTION_COUNT = 0
EVALUATOR_EXECUTION_COUNT = 0
BOOTSTRAP_EXECUTION_COUNT = 0
OUTCOME_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _digest(label: str, value: object) -> str:
    return sha256(label.encode("ascii") + b"\0" + _canonical_json(value)).hexdigest()


class LockedDistribution(NamedTuple):
    name: str
    version: str
    record_sha256: str


LOCKED_DISTRIBUTIONS = (
    LockedDistribution(
        "cloudpickle",
        "3.1.2",
        "bd3985f7752b9593d3621c8d95aedc411858fbecc516df904392432b9e1f1265",
    ),
    LockedDistribution(
        "Farama-Notifications",
        "0.0.6",
        "08925fa621c5abf7432aadd004c2380726ccbe9c5037a0804ab6bc03c478d175",
    ),
    LockedDistribution(
        "gymnasium",
        "1.3.0",
        "76ba26482e46157a41329577d6db3eb041fafc5b0adbcc297288d0ebc5874b8a",
    ),
    LockedDistribution(
        "minigrid",
        "3.1.0",
        "bda51b19eb9a310dfc2d045d7a940ee4ded7110b1a8cb1146e0a7e381fd049b9",
    ),
    LockedDistribution(
        "numpy",
        "2.4.6",
        "e2757e87e1a55a8df38e29a8e0a292967f5b7f96195ed4745fd1a151903fe56c",
    ),
    LockedDistribution(
        "pygame-ce",
        "2.5.8",
        "2d96d041485ccf1e898ff8f5ea6396f1083c3483b1add7f53e6b597da4b30a16",
    ),
    LockedDistribution(
        "typing_extensions",
        "4.16.0",
        "51ec3be62a52311731ad627591835d0f3fb982c47ffdd3e2b4b72850c3a92742",
    ),
)


@dataclass(frozen=True)
class RuntimeEnvironmentIdentity:
    python_version: str
    implementation: str
    cache_tag: str
    abi_flags: str
    platform: str
    machine: str
    system: str
    release: str
    system_version: str
    byteorder: str
    executable_sha256: str

    def record(self) -> dict[str, str]:
        return {
            "python_version": self.python_version,
            "implementation": self.implementation,
            "cache_tag": self.cache_tag,
            "abi_flags": self.abi_flags,
            "platform": self.platform,
            "machine": self.machine,
            "system": self.system,
            "release": self.release,
            "system_version": self.system_version,
            "byteorder": self.byteorder,
            "executable_sha256": self.executable_sha256,
        }


ENVIRONMENT_IDENTITY = RuntimeEnvironmentIdentity(
    python_version="3.11.15",
    implementation="CPython",
    cache_tag="cpython-311",
    abi_flags="",
    platform="win-amd64",
    machine="AMD64",
    system="Windows",
    release="10",
    system_version="10.0.26200",
    byteorder="little",
    executable_sha256="89e640c427d53939a746fd172f3f3071bfe45c2bdb171706655c8674a9d709ac",
)


@dataclass(frozen=True)
class FrozenG0DependencyEnvironment:
    schema: str
    status: str
    requirement: str
    claim_boundary: str
    project_manifest_sha256: str
    active_project_extras: tuple[str, ...]
    root_requirements: tuple[str, ...]
    locked_distributions: tuple[LockedDistribution, ...]
    environment_identity: RuntimeEnvironmentIdentity
    dependency_lock_sha256: str
    environment_sha256: str
    preregistration_sha256: str
    execution_counts: tuple[tuple[str, int], ...]
    sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "status": self.status,
            "requirement": self.requirement,
            "claim_boundary": self.claim_boundary,
            "project_manifest_sha256": self.project_manifest_sha256,
            "active_project_extras": self.active_project_extras,
            "root_requirements": self.root_requirements,
            "locked_distributions": tuple(self.locked_distributions),
            "environment_identity": self.environment_identity.record(),
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "environment_sha256": self.environment_sha256,
            "preregistration_sha256": self.preregistration_sha256,
            "execution_counts": self.execution_counts,
        }


class G0DependencyEnvironmentStop(RuntimeError):
    """Raised when the installed G0 environment differs from the frozen lock."""


def _project_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "pyproject.toml"


def _installed_record_sha256(name: str) -> tuple[str, str]:
    distribution = importlib.metadata.distribution(name)
    record = sorted(
        (str(path), str(path.hash), path.size)
        for path in (distribution.files or ())
        if path.hash is not None
    )
    return distribution.version, sha256(_canonical_json(record)).hexdigest()


def _runtime_environment_identity() -> RuntimeEnvironmentIdentity:
    executable = Path(sys.executable)
    return RuntimeEnvironmentIdentity(
        python_version=platform.python_version(),
        implementation=platform.python_implementation(),
        cache_tag=sys.implementation.cache_tag,
        abi_flags=getattr(sys, "abiflags", ""),
        platform=sysconfig.get_platform(),
        machine=platform.machine(),
        system=platform.system(),
        release=platform.release(),
        system_version=platform.version(),
        byteorder=sys.byteorder,
        executable_sha256=sha256(executable.read_bytes()).hexdigest(),
    )


def _build_capsule() -> FrozenG0DependencyEnvironment:
    dependency_record = {
        "project_manifest_sha256": PROJECT_MANIFEST_SHA256,
        "active_project_extras": ACTIVE_PROJECT_EXTRAS,
        "root_requirements": ROOT_REQUIREMENTS,
        "locked_distributions": tuple(LOCKED_DISTRIBUTIONS),
    }
    execution_counts = (
        ("generator", GENERATOR_EXECUTION_COUNT),
        ("cohort_generation", COHORT_GENERATION_COUNT),
        ("learner", LEARNER_EXECUTION_COUNT),
        ("evaluator", EVALUATOR_EXECUTION_COUNT),
        ("bootstrap", BOOTSTRAP_EXECUTION_COUNT),
        ("outcome", OUTCOME_COUNT),
        ("dev_input", DEV_INPUT_COUNT),
        ("m0_input", M0_INPUT_COUNT),
        ("protected_input", PROTECTED_INPUT_COUNT),
    )
    partial = FrozenG0DependencyEnvironment(
        schema=SCHEMA,
        status=STATUS,
        requirement=REQUIREMENT,
        claim_boundary=CLAIM_BOUNDARY,
        project_manifest_sha256=PROJECT_MANIFEST_SHA256,
        active_project_extras=ACTIVE_PROJECT_EXTRAS,
        root_requirements=ROOT_REQUIREMENTS,
        locked_distributions=LOCKED_DISTRIBUTIONS,
        environment_identity=ENVIRONMENT_IDENTITY,
        dependency_lock_sha256=_digest("g0-dependency-lock-v1", dependency_record),
        environment_sha256=_digest("g0-runtime-environment-v1", ENVIRONMENT_IDENTITY.record()),
        preregistration_sha256=prereg.PROTOCOL.protocol_sha256,
        execution_counts=execution_counts,
        sha256="",
    )
    return FrozenG0DependencyEnvironment(
        **{**partial.__dict__, "sha256": _digest("g0-dependency-environment-v1", partial.record())}
    )


CAPSULE = _build_capsule()


def authenticate_dependency_environment(
    capsule: FrozenG0DependencyEnvironment = CAPSULE,
) -> FrozenG0DependencyEnvironment:
    """Authenticate the exact local G0 runtime before cohort generation."""

    if capsule is not CAPSULE or capsule != _build_capsule():
        raise G0DependencyEnvironmentStop("STOP_G0_DEPENDENCY_CAPSULE_IDENTITY_CHANGED")
    if capsule.sha256 != _digest("g0-dependency-environment-v1", capsule.record()):
        raise G0DependencyEnvironmentStop("STOP_G0_DEPENDENCY_CAPSULE_ROOT_CHANGED")
    if sha256(_project_manifest_path().read_bytes()).hexdigest() != PROJECT_MANIFEST_SHA256:
        raise G0DependencyEnvironmentStop("STOP_G0_PROJECT_MANIFEST_CHANGED")
    for locked in LOCKED_DISTRIBUTIONS:
        try:
            installed_version, installed_record = _installed_record_sha256(locked.name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise G0DependencyEnvironmentStop(
                f"STOP_G0_DEPENDENCY_MISSING:{locked.name}"
            ) from exc
        if (installed_version, installed_record) != (
            locked.version,
            locked.record_sha256,
        ):
            raise G0DependencyEnvironmentStop(
                f"STOP_G0_DEPENDENCY_IDENTITY_CHANGED:{locked.name}"
            )
    if _runtime_environment_identity() != ENVIRONMENT_IDENTITY:
        raise G0DependencyEnvironmentStop("STOP_G0_RUNTIME_ENVIRONMENT_CHANGED")
    if any(value != 0 for _name, value in capsule.execution_counts):
        raise G0DependencyEnvironmentStop("STOP_G0_DEPENDENCY_CAPSULE_EXECUTED_G0")
    return capsule


__all__ = [
    "ACTIVE_PROJECT_EXTRAS",
    "CAPSULE",
    "CLAIM_BOUNDARY",
    "ENVIRONMENT_IDENTITY",
    "LOCKED_DISTRIBUTIONS",
    "PROJECT_MANIFEST_SHA256",
    "REQUIREMENT",
    "ROOT_REQUIREMENTS",
    "SCHEMA",
    "STATUS",
    "G0DependencyEnvironmentStop",
    "authenticate_dependency_environment",
]
