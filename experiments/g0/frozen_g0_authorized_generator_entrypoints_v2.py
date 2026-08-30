"""Exact-object G0 generation planning seam after the complete V2 binding.

The two successor entrypoints accept only the module-owned authorization from
``frozen_common_architecture_binding_v2``.  They authenticate the banked
binding and both unchanged V1 generator sources, then return immutable plans.
They never call either generator.  Existing V1 generation entrypoints remain
terminal and fail-closed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

from experiments.g0 import frozen_common_architecture_binding_v2 as binding
from experiments.g0 import frozen_regulatory_generator_v1 as regulatory
from experiments.g0 import frozen_spatial_environment_generator_v1 as spatial

SCOPE = "FROZEN_G0_AUTHORIZED_GENERATOR_ENTRYPOINTS_V2"
STATUS = "AUTHORIZED_NON_EXECUTING_GENERATION_PLANS_FROZEN"
CLAIM_BOUNDARY = "exact-object-generation-planning-only-no-g0-cohort-or-result"

BINDING_SOURCE_COMMIT = "d987edde74c6cfd704afaa03ec5d9248bef22c65"
EXPECTED_BINDING_SOURCE_SHA256 = (
    "e21184feb2d80e05105fd86b79fbd5e5622d4e96f1cb1fa567ef91f7ea61b9bf"
)
EXPECTED_BINDING_CONFIG_SHA256 = (
    "afdc11a1792bc207ffd3dba415996448b2e5bc58acce3470482131124516b116"
)
EXPECTED_BINDING_ROOT_SHA256 = (
    "d496be2b5b8cadb980bb5ed2f8e423e01b965bb743f0d1f36a293274e06e5d01"
)

REGULATORY_SOURCE_COMMIT = "04c6367538534071b9f76de4bd1132a29f648b9b"
EXPECTED_REGULATORY_SOURCE_SHA256 = (
    "1af319fe8bd27d885bbc1e80a7e14328b6d878f6c8df8ca3bd7debd947ee912d"
)
EXPECTED_REGULATORY_CONFIGURATION_SHA256 = (
    "e65bfcb7eccc32817d284df3c90243a8780fa755a20611d338590025792b4d05"
)

SPATIAL_SOURCE_COMMIT = "cfe4c5504f07bb6b14cf60057cfb6bd51460faa2"
EXPECTED_SPATIAL_SOURCE_SHA256 = (
    "63195a7eb35903e5b7e51a3d6ea5dc0a5154cf3dc4464a7ab9460f832322f0ef"
)
EXPECTED_SPATIAL_CONFIGURATION_SHA256 = (
    "3425f82374ff39ac23fbf72b7fee93d63926b2be84d07ae44dc1416ee8048b28"
)
EXPECTED_SPATIAL_DEPENDENCY_SHA256 = (
    "644bb2e9296b3f79618ae786eed55a98aa9cf1a700d8ba0528a599d6aa56414e"
)
EXPECTED_SPATIAL_GENERATOR_ROOT_SHA256 = (
    "7099bba537b1f08a04d0eb90ec5f8b3e8edcd3acde7ff07fc4d018348eb377e2"
)

GENERATOR_EXECUTION_COUNT = 0
COHORT_GENERATION_COUNT = 0
ADJUDICATIVE_SEED_EXECUTION_COUNT = 0
LEARNER_EXECUTION_COUNT = 0
EVALUATOR_EXECUTION_COUNT = 0
BOOTSTRAP_EXECUTION_COUNT = 0
OUTCOME_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATH = Path(__file__).resolve()
_CONFIG_NAMESPACE = b"frozen-g0-authorized-generator-entrypoints-v2-config\0"
_PLAN_NAMESPACE = b"frozen-g0-authorized-generator-entrypoints-v2-plan\0"
_ROOT_NAMESPACE = b"frozen-g0-authorized-generator-entrypoints-v2-root\0"


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _source_sha256(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _ROOT / candidate
    return sha256(candidate.read_bytes()).hexdigest()


class G0AuthorizedGeneratorEntrypointStop(RuntimeError):
    """Raised before any generator call when a frozen identity differs."""


@dataclass(frozen=True, slots=True)
class FrozenGenerationPlan:
    domain: str
    status: str
    claim_boundary: str
    successor_entrypoint: str
    terminal_v1_entrypoint: str
    binding_source_commit: str
    binding_sha256: str
    generator_source_commit: str
    generator_source_sha256: str
    generator_configuration_sha256: str
    generator_artifact_sha256s: tuple[tuple[str, str], ...]
    frozen_cohort_seeds: tuple[int, ...]
    generator_invocation_count: int
    execution_counts: tuple[tuple[str, int], ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "sha256"
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record
ZERO_EXECUTION_COUNTS = (
    ("generator", GENERATOR_EXECUTION_COUNT),
    ("cohort-generation", COHORT_GENERATION_COUNT),
    ("adjudicative-seed", ADJUDICATIVE_SEED_EXECUTION_COUNT),
    ("learner", LEARNER_EXECUTION_COUNT),
    ("evaluator", EVALUATOR_EXECUTION_COUNT),
    ("bootstrap", BOOTSTRAP_EXECUTION_COUNT),
    ("outcome", OUTCOME_COUNT),
    ("dev-input", DEV_INPUT_COUNT),
    ("m0-input", M0_INPUT_COUNT),
    ("protected-input", PROTECTED_INPUT_COUNT),
)

GENERATOR_IDENTITIES = MappingProxyType(
    {
        "regulatory": {
            "successor_entrypoint": "freeze_regulatory_generation_plan",
            "terminal_v1_entrypoint": "generate_frozen_cohort",
            "source_commit": REGULATORY_SOURCE_COMMIT,
            "source_path": "experiments/g0/frozen_regulatory_generator_v1.py",
            "source_sha256": EXPECTED_REGULATORY_SOURCE_SHA256,
            "configuration_sha256": EXPECTED_REGULATORY_CONFIGURATION_SHA256,
            "artifact_sha256s": (
                (
                    "source-specification",
                    EXPECTED_REGULATORY_CONFIGURATION_SHA256,
                ),
            ),
            "frozen_cohort_seeds": tuple(regulatory.FROZEN_COHORT_SEEDS),
        },
        "spatial": {
            "successor_entrypoint": "freeze_spatial_generation_plan",
            "terminal_v1_entrypoint": "generate_frozen_spatial_cohort",
            "source_commit": SPATIAL_SOURCE_COMMIT,
            "source_path": (
                "experiments/g0/frozen_spatial_environment_generator_v1.py"
            ),
            "source_sha256": EXPECTED_SPATIAL_SOURCE_SHA256,
            "configuration_sha256": EXPECTED_SPATIAL_CONFIGURATION_SHA256,
            "artifact_sha256s": (
                ("dependency", EXPECTED_SPATIAL_DEPENDENCY_SHA256),
                ("generator-root", EXPECTED_SPATIAL_GENERATOR_ROOT_SHA256),
            ),
            "frozen_cohort_seeds": tuple(spatial.FROZEN_COHORT_SEEDS),
        },
    }
)


def _build_plan(domain: str) -> FrozenGenerationPlan:
    identity = GENERATOR_IDENTITIES[domain]
    record = {
        "domain": domain,
        "status": STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "successor_entrypoint": identity["successor_entrypoint"],
        "terminal_v1_entrypoint": identity["terminal_v1_entrypoint"],
        "binding_source_commit": BINDING_SOURCE_COMMIT,
        "binding_sha256": EXPECTED_BINDING_ROOT_SHA256,
        "generator_source_commit": identity["source_commit"],
        "generator_source_sha256": identity["source_sha256"],
        "generator_configuration_sha256": identity["configuration_sha256"],
        "generator_artifact_sha256s": identity["artifact_sha256s"],
        "frozen_cohort_seeds": identity["frozen_cohort_seeds"],
        "generator_invocation_count": 0,
        "execution_counts": ZERO_EXECUTION_COUNTS,
    }
    return FrozenGenerationPlan(
        **record,
        sha256=_digest(_PLAN_NAMESPACE, record),
    )


REGULATORY_PLAN = _build_plan("regulatory")
SPATIAL_PLAN = _build_plan("spatial")

CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "binding": {
        "source_commit": BINDING_SOURCE_COMMIT,
        "source_sha256": EXPECTED_BINDING_SOURCE_SHA256,
        "config_sha256": EXPECTED_BINDING_CONFIG_SHA256,
        "binding_sha256": EXPECTED_BINDING_ROOT_SHA256,
        "authorization_policy": "module-owned-exact-object-identity",
    },
    "generator_identities": dict(GENERATOR_IDENTITIES),
    "plans": (REGULATORY_PLAN.record(), SPATIAL_PLAN.record()),
    "generator_calls_during_planning": 0,
    "execution_counts": ZERO_EXECUTION_COUNTS,
}
CONFIG_SHA256 = _digest(_CONFIG_NAMESPACE, CONFIG_RECORD)


def _authenticate_static_sources() -> None:
    source_expectations = {
        "experiments/g0/frozen_common_architecture_binding_v2.py": (
            EXPECTED_BINDING_SOURCE_SHA256
        ),
        "experiments/g0/frozen_regulatory_generator_v1.py": (
            EXPECTED_REGULATORY_SOURCE_SHA256
        ),
        "experiments/g0/frozen_spatial_environment_generator_v1.py": (
            EXPECTED_SPATIAL_SOURCE_SHA256
        ),
    }
    for path, expected in source_expectations.items():
        if _source_sha256(path) != expected:
            raise G0AuthorizedGeneratorEntrypointStop(
                f"STOP_G0_AUTHORIZED_ENTRYPOINT_SOURCE_MISMATCH:{path}"
            )
    if (
        binding.SOURCE_SHA256 != EXPECTED_BINDING_SOURCE_SHA256
        or binding.CONFIG_SHA256 != EXPECTED_BINDING_CONFIG_SHA256
        or binding.BINDING.sha256 != EXPECTED_BINDING_ROOT_SHA256
        or binding.BINDING.bound_class_count != 9
        or binding.BINDING.missing_class_count != 0
        or binding.BINDING.generation_authorized is not True
        or regulatory.SOURCE_SPECIFICATION_SHA256
        != EXPECTED_REGULATORY_CONFIGURATION_SHA256
        or regulatory.GENERATION_AUTHORIZED is not False
        or regulatory.ARCHITECTURE_BINDING_COUNT != 0
        or regulatory.FROZEN_COHORT_EXECUTION_COUNT != 0
        or regulatory.FROZEN_SYSTEM_GENERATION_COUNT != 0
        or regulatory.ADJUDICATIVE_SYSTEM_GENERATION_COUNT != 0
        or spatial.SOURCE_CONFIGURATION_SHA256
        != EXPECTED_SPATIAL_CONFIGURATION_SHA256
        or spatial.DEPENDENCY_SHA256 != EXPECTED_SPATIAL_DEPENDENCY_SHA256
        or spatial.GENERATOR_ROOT_SHA256
        != EXPECTED_SPATIAL_GENERATOR_ROOT_SHA256
        or spatial.ARCHITECTURE_BINDING_COUNT != 0
        or spatial.FROZEN_COHORT_EXECUTION_COUNT != 0
        or spatial.FROZEN_LAYOUT_GENERATION_COUNT != 0
        or spatial.LEARNER_EXECUTION_COUNT != 0
        or spatial.EVALUATOR_EXECUTION_COUNT != 0
        or spatial.OUTCOME_COUNT != 0
        or any(value for _, value in ZERO_EXECUTION_COUNTS)
    ):
        raise G0AuthorizedGeneratorEntrypointStop(
            "STOP_G0_AUTHORIZED_ENTRYPOINT_FROZEN_IDENTITY_MISMATCH"
        )


_authenticate_static_sources()


def _authorized_plan(authorization: object, plan: FrozenGenerationPlan) -> FrozenGenerationPlan:
    _authenticate_static_sources()
    try:
        authenticated = binding.require_generation_authorization(authorization)
    except binding.G0CommonArchitectureBindingV2Stop as exc:
        raise G0AuthorizedGeneratorEntrypointStop(
            "STOP_G0_AUTHORIZED_ENTRYPOINT_EXACT_BINDING_OBJECT_REQUIRED"
        ) from exc
    if authenticated is not binding.BINDING or authenticated.sha256 != plan.binding_sha256:
        raise G0AuthorizedGeneratorEntrypointStop(
            "STOP_G0_AUTHORIZED_ENTRYPOINT_BINDING_IDENTITY_MISMATCH"
        )
    return plan


def freeze_regulatory_generation_plan(
    authorization: object,
) -> FrozenGenerationPlan:
    """Authenticate the exact singleton and return the non-executing plan."""

    return _authorized_plan(authorization, REGULATORY_PLAN)


def freeze_spatial_generation_plan(
    authorization: object,
) -> FrozenGenerationPlan:
    """Authenticate the exact singleton and return the non-executing plan."""

    return _authorized_plan(authorization, SPATIAL_PLAN)


SOURCE_SHA256 = _source_sha256(_SOURCE_PATH)
ENTRYPOINT_ROOT_SHA256 = _digest(
    _ROOT_NAMESPACE,
    {
        "source_sha256": SOURCE_SHA256,
        "config_sha256": CONFIG_SHA256,
        "regulatory_plan_sha256": REGULATORY_PLAN.sha256,
        "spatial_plan_sha256": SPATIAL_PLAN.sha256,
    },
)

__all__ = [
    "BINDING_SOURCE_COMMIT",
    "CLAIM_BOUNDARY",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "ENTRYPOINT_ROOT_SHA256",
    "GENERATOR_IDENTITIES",
    "REGULATORY_PLAN",
    "SCOPE",
    "SOURCE_SHA256",
    "SPATIAL_PLAN",
    "STATUS",
    "ZERO_EXECUTION_COUNTS",
    "FrozenGenerationPlan",
    "G0AuthorizedGeneratorEntrypointStop",
    "freeze_regulatory_generation_plan",
    "freeze_spatial_generation_plan",
]
