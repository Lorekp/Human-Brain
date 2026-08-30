"""Non-executing nine-class binding ledger for the frozen G0 protocol.

This module authenticates the G0 sources and configurations that already exist.
It does not fill absent artifact classes with conventions.  A class is BOUND only
when the current repository contains its complete identity; otherwise its exact
missing seam is frozen as a STOP.  No generator, cohort, learner, evaluator,
bootstrap, protected coding input, or adjudicative seed is executed here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

from experiments.g0 import common_event_associative_memory_v1 as associative
from experiments.g0 import common_event_cells_v1 as event_cells
from experiments.g0 import common_factorized_transition_memory_v1 as transition
from experiments.g0 import frozen_cross_domain_g0_preregistration_v1 as prereg
from experiments.g0 import frozen_regulatory_adapter_v1 as regulatory_adapter
from experiments.g0 import frozen_regulatory_generator_v1 as regulatory_generator
from experiments.g0 import frozen_spatial_environment_generator_v1 as spatial_generator
from experiments.g0 import spatial_candidate_adapter_v1 as spatial_adapter

SCOPE = "FROZEN_G0_COMMON_ARCHITECTURE_BINDING_V1"
STATUS = "AUTHENTICATED_PARTIAL_BINDING_STOPPED_3_OF_9"
CLAIM_BOUNDARY = (
    "authenticated-source-and-configuration-ledger-only-no-generation-or-g0-evidence"
)

BOUND = "BOUND"
STOP_MISSING_INTEGRATED_CHECKPOINT = (
    "STOP_G0_BINDING_V1_MISSING_INTEGRATED_COMMON_ARCHITECTURE_CHECKPOINT_SHA256"
)
STOP_MISSING_ACTION_ASSIGNMENT = (
    "STOP_G0_BINDING_V1_MISSING_COMMON_OUTPUT_TO_REGULATORY_ACTION_ASSIGNMENT"
)
STOP_MISSING_BASELINE_SOURCES = (
    "STOP_G0_BINDING_V1_MISSING_EXACT_BASELINE_IMPLEMENTATION_SOURCES"
)
STOP_MISSING_ROLLOUT_SEEDS = (
    "STOP_G0_BINDING_V1_MISSING_ROLLOUT_AND_QUERY_SEED_DERIVATION"
)
STOP_MISSING_EVALUATOR = (
    "STOP_G0_BINDING_V1_MISSING_EVALUATOR_AND_BOOTSTRAP_SOURCE"
)
STOP_MISSING_DEPENDENCY_LOCK = (
    "STOP_G0_BINDING_V1_MISSING_PYTHON_DEPENDENCY_LOCK_AND_ENVIRONMENT_IDENTITY"
)
STOP_INCOMPLETE_BINDING = "STOP_G0_COMMON_ARCHITECTURE_BINDING_V1_MISSING_6_OF_9"

COMMON_ARCHITECTURE_SOURCE_COMMIT = (
    "973c3e5133e5162641493a28f42b4855abb55974"
)
PREREGISTRATION_SOURCE_COMMIT = "42278a75647951a7da7a51c442ac14bb54db7dab"
EXPECTED_PROTOCOL_SHA256 = (
    "955828b48591f198047d5ca4966885e253763d6ea353351585edb14390320468"
)

COMPLETE_ARCHITECTURE_BINDING_COUNT = 0
GENERATOR_EXECUTION_COUNT = 0
COHORT_GENERATION_COUNT = 0
LEARNER_EXECUTION_COUNT = 0
EVALUATOR_EXECUTION_COUNT = 0
BOOTSTRAP_EXECUTION_COUNT = 0
OUTCOME_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_NAMESPACE = b"frozen-g0-common-architecture-binding-source-v1\0"
_CLASS_NAMESPACE = b"frozen-g0-common-architecture-binding-class-v1\0"
_ROOT_NAMESPACE = b"frozen-g0-common-architecture-binding-root-v1\0"


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


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    name: str
    path: str
    origin_commit_sha: str
    source_sha256: str
    artifact_sha256s: tuple[tuple[str, str], ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "name": self.name,
            "path": self.path,
            "origin_commit_sha": self.origin_commit_sha,
            "source_sha256": self.source_sha256,
            "artifact_sha256s": self.artifact_sha256s,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class ArtifactClassBinding:
    ordinal: int
    requirement: str
    status: str
    evidence: tuple[str, ...]
    artifact_sha256s: tuple[tuple[str, str], ...]
    missing: tuple[str, ...]
    sha256: str

    @property
    def bound(self) -> bool:
        return self.status == BOUND

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "ordinal": self.ordinal,
            "requirement": self.requirement,
            "status": self.status,
            "evidence": self.evidence,
            "artifact_sha256s": self.artifact_sha256s,
            "missing": self.missing,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class FrozenCommonArchitectureBinding:
    scope: str
    status: str
    claim_boundary: str
    preregistration_sha256: str
    common_architecture_source_commit: str
    sources: tuple[SourceIdentity, ...]
    classes: tuple[ArtifactClassBinding, ...]
    bound_class_count: int
    missing_class_count: int
    complete_architecture_binding_count: int
    generation_authorized: bool
    stop_reason: str
    execution_counts: tuple[tuple[str, int], ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "scope": self.scope,
            "status": self.status,
            "claim_boundary": self.claim_boundary,
            "preregistration_sha256": self.preregistration_sha256,
            "common_architecture_source_commit": (
                self.common_architecture_source_commit
            ),
            "sources": tuple(row.record() for row in self.sources),
            "classes": tuple(row.record() for row in self.classes),
            "bound_class_count": self.bound_class_count,
            "missing_class_count": self.missing_class_count,
            "complete_architecture_binding_count": (
                self.complete_architecture_binding_count
            ),
            "generation_authorized": self.generation_authorized,
            "stop_reason": self.stop_reason,
            "execution_counts": self.execution_counts,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


class G0CommonArchitectureBindingStop(RuntimeError):
    """Raised before generation while any required artifact class is missing."""


_SOURCE_SPECS = (
    (
        "preregistration",
        "experiments/g0/frozen_cross_domain_g0_preregistration_v1.py",
        PREREGISTRATION_SOURCE_COMMIT,
        "e027e598aa926b108e90208e1586a93c0da35716e165a746bb2732d980774395",
        (
            ("protocol", EXPECTED_PROTOCOL_SHA256),
            ("adapter-contract", prereg.PROTOCOL.adapter_sha256),
            ("regulatory-protocol", prereg.PROTOCOL.regulatory_sha256),
            ("spatial-protocol", prereg.PROTOCOL.spatial_sha256),
            ("statistics", prereg.PROTOCOL.statistics_sha256),
            ("surface-boundary", prereg.PROTOCOL.surface_boundary_sha256),
        ),
    ),
    (
        "common-event-cells",
        "experiments/g0/common_event_cells_v1.py",
        "4843236073a57f614b6c70b1fd45c3ac6ccdedab",
        "f425c07c44942ff7cd9672a506f616d9e6f9640c52340b481a81ad7b1603eaeb",
        (("geometry", event_cells.GEOMETRY_SHA256),),
    ),
    (
        "common-event-associative-memory",
        "experiments/g0/common_event_associative_memory_v1.py",
        "9ad36b3def1d4d8e4b58d2ab50b1618474257017",
        "34185505cc1e07220f968f20ac7d4c0303ccd43d267e1b52842ef7012f3f2ef5",
        (("geometry", associative.GEOMETRY_SHA256),),
    ),
    (
        "common-factorized-transition-memory",
        "experiments/g0/common_factorized_transition_memory_v1.py",
        COMMON_ARCHITECTURE_SOURCE_COMMIT,
        "0f2c480a4ade72f96bddfc969b9e0103a75ba9829f91530b4b72cbb6cf21489b",
        (
            ("geometry", transition.GEOMETRY_SHA256),
            ("branch-current-table", transition.BRANCH_CURRENT_TABLE_SHA256),
        ),
    ),
    (
        "regulatory-generator",
        "experiments/g0/frozen_regulatory_generator_v1.py",
        "04c6367538534071b9f76de4bd1132a29f648b9b",
        "1af319fe8bd27d885bbc1e80a7e14328b6d878f6c8df8ca3bd7debd947ee912d",
        (("source-specification", regulatory_generator.SOURCE_SPECIFICATION_SHA256),),
    ),
    (
        "regulatory-adapter",
        "experiments/g0/frozen_regulatory_adapter_v1.py",
        "04c6367538534071b9f76de4bd1132a29f648b9b",
        "5cf0e0f52892e7b400940beeed015bb1908667a9408a6acb68484bc7cf0ed908",
        (("adapter", regulatory_adapter.ADAPTER_SHA256),),
    ),
    (
        "spatial-generator",
        "experiments/g0/frozen_spatial_environment_generator_v1.py",
        "cfe4c5504f07bb6b14cf60057cfb6bd51460faa2",
        "63195a7eb35903e5b7e51a3d6ea5dc0a5154cf3dc4464a7ab9460f832322f0ef",
        (
            ("source-configuration", spatial_generator.SOURCE_CONFIGURATION_SHA256),
            ("dependency", spatial_generator.DEPENDENCY_SHA256),
            ("generator-root", spatial_generator.GENERATOR_ROOT_SHA256),
        ),
    ),
    (
        "spatial-adapter",
        "experiments/g0/spatial_candidate_adapter_v1.py",
        "c4414efb65a2e4a22067a31e016cf453e80ecf21",
        "ac2c9307d45a48d6cdba72c5e4d0502c8e9a60af799783caa9c0fb41ad028fe1",
        (),
    ),
    (
        "spatial-fixture-preflight",
        "experiments/g0/spatial_fixture_generator_v1.py",
        "c4414efb65a2e4a22067a31e016cf453e80ecf21",
        "a28bc1c2fd46daccd88084e49403a1b639f9ef6d8f50856c8e68b20aec00f0b3",
        (),
    ),
    (
        "python-project-manifest",
        "pyproject.toml",
        "acdb095a24d5fecccfa7324d6bbe5506231ed308",
        "70fd2232f335929385f4b61e482aa714bc4baf8a706aaa825fa126427f23f77e",
        (),
    ),
)


def _source_identity(spec: tuple[object, ...]) -> SourceIdentity:
    name, path, origin_commit_sha, source_sha256, artifact_sha256s = spec
    record = {
        "name": name,
        "path": path,
        "origin_commit_sha": origin_commit_sha,
        "source_sha256": source_sha256,
        "artifact_sha256s": artifact_sha256s,
    }
    return SourceIdentity(
        name=str(name),
        path=str(path),
        origin_commit_sha=str(origin_commit_sha),
        source_sha256=str(source_sha256),
        artifact_sha256s=tuple(artifact_sha256s),  # type: ignore[arg-type]
        sha256=_digest(_SOURCE_NAMESPACE, record),
    )


SOURCES = tuple(_source_identity(spec) for spec in _SOURCE_SPECS)
SOURCE_BY_NAME = MappingProxyType({source.name: source for source in SOURCES})


def _class_binding(
    ordinal: int,
    requirement: str,
    status: str,
    evidence: tuple[str, ...],
    artifact_sha256s: tuple[tuple[str, str], ...],
    missing: tuple[str, ...] = (),
) -> ArtifactClassBinding:
    record = {
        "ordinal": ordinal,
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "artifact_sha256s": artifact_sha256s,
        "missing": missing,
    }
    return ArtifactClassBinding(
        ordinal=ordinal,
        requirement=requirement,
        status=status,
        evidence=evidence,
        artifact_sha256s=artifact_sha256s,
        missing=missing,
        sha256=_digest(_CLASS_NAMESPACE, record),
    )


def _build_classes() -> tuple[ArtifactClassBinding, ...]:
    requirements = prereg.COMMON_ARCHITECTURE_FREEZE_REQUIREMENTS
    return (
        _class_binding(
            1,
            requirements[0],
            STOP_MISSING_INTEGRATED_CHECKPOINT,
            (
                "common-event-cells",
                "common-event-associative-memory",
                "common-factorized-transition-memory",
            ),
            (
                ("architecture-source-commit", COMMON_ARCHITECTURE_SOURCE_COMMIT),
                ("event-cell-geometry", event_cells.GEOMETRY_SHA256),
                ("associative-memory-geometry", associative.GEOMETRY_SHA256),
                ("factorized-transition-geometry", transition.GEOMETRY_SHA256),
            ),
            ("one-integrated-common-architecture-checkpoint-sha256",),
        ),
        _class_binding(
            2,
            requirements[1],
            BOUND,
            (
                "common-event-cells",
                "common-event-associative-memory",
                "common-factorized-transition-memory",
            ),
            (
                ("event-cell-geometry", event_cells.GEOMETRY_SHA256),
                ("associative-memory-geometry", associative.GEOMETRY_SHA256),
                ("factorized-transition-geometry", transition.GEOMETRY_SHA256),
                ("branch-current-table", transition.BRANCH_CURRENT_TABLE_SHA256),
            ),
        ),
        _class_binding(
            3,
            requirements[2],
            BOUND,
            (
                "regulatory-generator",
                "spatial-generator",
                "spatial-fixture-preflight",
            ),
            (
                (
                    "regulatory-source-specification",
                    regulatory_generator.SOURCE_SPECIFICATION_SHA256,
                ),
                (
                    "spatial-source-configuration",
                    spatial_generator.SOURCE_CONFIGURATION_SHA256,
                ),
                ("spatial-dependency", spatial_generator.DEPENDENCY_SHA256),
                ("spatial-generator-root", spatial_generator.GENERATOR_ROOT_SHA256),
            ),
        ),
        _class_binding(
            4,
            requirements[3],
            STOP_MISSING_ACTION_ASSIGNMENT,
            (
                "regulatory-adapter",
                "spatial-adapter",
                "common-event-cells",
            ),
            (
                ("regulatory-adapter", regulatory_adapter.ADAPTER_SHA256),
                ("common-output-geometry", event_cells.GEOMETRY_SHA256),
            ),
            ("physical-output-byte-sequence-to-regulatory-composite-action",),
        ),
        _class_binding(
            5,
            requirements[4],
            STOP_MISSING_BASELINE_SOURCES,
            ("preregistration",),
            (
                ("regulatory-protocol", prereg.PROTOCOL.regulatory_sha256),
                ("spatial-protocol", prereg.PROTOCOL.spatial_sha256),
            ),
            (
                "five-regulatory-baseline-implementation-sources",
                "four-spatial-baseline-implementation-sources",
            ),
        ),
        _class_binding(
            6,
            requirements[5],
            STOP_MISSING_ROLLOUT_SEEDS,
            ("preregistration",),
            (("protocol", prereg.PROTOCOL.protocol_sha256),),
            ("deterministic-per-system-query-and-rollout-seed-derivation",),
        ),
        _class_binding(
            7,
            requirements[6],
            BOUND,
            ("preregistration",),
            (
                ("protocol", prereg.PROTOCOL.protocol_sha256),
                ("regulatory-protocol", prereg.PROTOCOL.regulatory_sha256),
                ("spatial-protocol", prereg.PROTOCOL.spatial_sha256),
                ("statistics", prereg.PROTOCOL.statistics_sha256),
            ),
        ),
        _class_binding(
            8,
            requirements[7],
            STOP_MISSING_EVALUATOR,
            ("preregistration",),
            (("statistics-parameters", prereg.PROTOCOL.statistics_sha256),),
            ("evaluator-source", "exact-50000-resample-bootstrap-algorithm-source"),
        ),
        _class_binding(
            9,
            requirements[8],
            STOP_MISSING_DEPENDENCY_LOCK,
            ("python-project-manifest", "spatial-generator"),
            (("spatial-dependency", spatial_generator.DEPENDENCY_SHA256),),
            ("transitive-python-dependency-lock", "environment-or-container-identity"),
        ),
    )


CLASSES = _build_classes()
CLASS_BY_REQUIREMENT = MappingProxyType({row.requirement: row for row in CLASSES})

EXECUTION_COUNTS = (
    ("generator", GENERATOR_EXECUTION_COUNT),
    ("cohort-generation", COHORT_GENERATION_COUNT),
    ("learner", LEARNER_EXECUTION_COUNT),
    ("evaluator", EVALUATOR_EXECUTION_COUNT),
    ("bootstrap", BOOTSTRAP_EXECUTION_COUNT),
    ("outcome", OUTCOME_COUNT),
    ("dev-input", DEV_INPUT_COUNT),
    ("m0-input", M0_INPUT_COUNT),
    ("protected-input", PROTECTED_INPUT_COUNT),
)


def _authenticate_existing_artifacts() -> None:
    if (
        prereg.PROTOCOL.protocol_sha256 != EXPECTED_PROTOCOL_SHA256
        or tuple(row.requirement for row in CLASSES)
        != prereg.COMMON_ARCHITECTURE_FREEZE_REQUIREMENTS
        or event_cells.STATUS != "COMMON_PROJECTION_ONLY_NO_LEARNER_OR_G0_RESULT"
        or associative.STATUS != "MECHANISM_SLICE_ONLY_NO_G0_COHORT_OR_RESULT"
        or transition.STATUS != "QUARANTINED_MECHANISM_ONLY_NO_G0_COHORT_OR_RESULT"
        or regulatory_generator.GENERATION_AUTHORIZED is not False
        or spatial_generator.SURFACE_BOUNDARY_RECORD[
            "frozen_cohort_generation_authorized"
        ]
        is not False
        or tuple(range(7)) != spatial_adapter.RAW_ACTION_IDS
        or any(value for _, value in EXECUTION_COUNTS)
        or any(
            (
                prereg.ARCHITECTURE_BINDING_COUNT,
                prereg.GENERATOR_EXECUTION_COUNT,
                prereg.COHORT_GENERATION_COUNT,
                prereg.LEARNER_EXECUTION_COUNT,
                prereg.EVALUATOR_EXECUTION_COUNT,
                prereg.OUTCOME_COUNT,
                regulatory_generator.ARCHITECTURE_BINDING_COUNT,
                regulatory_generator.FROZEN_COHORT_EXECUTION_COUNT,
                regulatory_generator.FROZEN_SYSTEM_GENERATION_COUNT,
                regulatory_generator.ADJUDICATIVE_SYSTEM_GENERATION_COUNT,
                spatial_generator.ARCHITECTURE_BINDING_COUNT,
                spatial_generator.FROZEN_COHORT_EXECUTION_COUNT,
                spatial_generator.FROZEN_LAYOUT_GENERATION_COUNT,
                spatial_generator.LEARNER_EXECUTION_COUNT,
                spatial_generator.EVALUATOR_EXECUTION_COUNT,
                spatial_generator.OUTCOME_COUNT,
            )
        )
    ):
        raise RuntimeError("STOP_G0_BINDING_V1_EXISTING_ARTIFACT_BOUNDARY_MISMATCH")
    for source in SOURCES:
        observed = sha256((_ROOT / source.path).read_bytes()).hexdigest()
        if observed != source.source_sha256:
            raise RuntimeError(
                f"STOP_G0_BINDING_V1_SOURCE_IDENTITY_MISMATCH:{source.name}"
            )


def _build_binding() -> FrozenCommonArchitectureBinding:
    _authenticate_existing_artifacts()
    bound_count = sum(row.bound for row in CLASSES)
    missing_count = len(CLASSES) - bound_count
    if bound_count != 3 or missing_count != 6:
        raise RuntimeError("STOP_G0_BINDING_V1_CLASS_LEDGER_CHANGED")
    record = {
        "scope": SCOPE,
        "status": STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "preregistration_sha256": prereg.PROTOCOL.protocol_sha256,
        "common_architecture_source_commit": COMMON_ARCHITECTURE_SOURCE_COMMIT,
        "sources": tuple(row.record() for row in SOURCES),
        "classes": tuple(row.record() for row in CLASSES),
        "bound_class_count": bound_count,
        "missing_class_count": missing_count,
        "complete_architecture_binding_count": COMPLETE_ARCHITECTURE_BINDING_COUNT,
        "generation_authorized": False,
        "stop_reason": STOP_INCOMPLETE_BINDING,
        "execution_counts": EXECUTION_COUNTS,
    }
    return FrozenCommonArchitectureBinding(
        scope=SCOPE,
        status=STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        preregistration_sha256=prereg.PROTOCOL.protocol_sha256,
        common_architecture_source_commit=COMMON_ARCHITECTURE_SOURCE_COMMIT,
        sources=SOURCES,
        classes=CLASSES,
        bound_class_count=bound_count,
        missing_class_count=missing_count,
        complete_architecture_binding_count=COMPLETE_ARCHITECTURE_BINDING_COUNT,
        generation_authorized=False,
        stop_reason=STOP_INCOMPLETE_BINDING,
        execution_counts=EXECUTION_COUNTS,
        sha256=_digest(_ROOT_NAMESPACE, record),
    )


BINDING = _build_binding()


def authenticate_binding() -> FrozenCommonArchitectureBinding:
    """Re-authenticate the frozen ledger without executing any G0 surface."""

    observed = _build_binding()
    if observed.sha256 != BINDING.sha256 or observed != BINDING:
        raise RuntimeError("STOP_G0_BINDING_V1_ROOT_IDENTITY_MISMATCH")
    return BINDING


def require_generation_authorization() -> None:
    """Fail before generator selection while the nine-class ledger is incomplete."""

    binding = authenticate_binding()
    if not binding.generation_authorized:
        raise G0CommonArchitectureBindingStop(binding.stop_reason)
    raise RuntimeError("unreachable generation authorization state")


__all__ = [
    "BINDING",
    "BOUND",
    "CLAIM_BOUNDARY",
    "CLASSES",
    "CLASS_BY_REQUIREMENT",
    "COMMON_ARCHITECTURE_SOURCE_COMMIT",
    "COMPLETE_ARCHITECTURE_BINDING_COUNT",
    "EXECUTION_COUNTS",
    "SCOPE",
    "SOURCES",
    "SOURCE_BY_NAME",
    "STATUS",
    "STOP_INCOMPLETE_BINDING",
    "ArtifactClassBinding",
    "FrozenCommonArchitectureBinding",
    "G0CommonArchitectureBindingStop",
    "SourceIdentity",
    "authenticate_binding",
    "require_generation_authorization",
]
