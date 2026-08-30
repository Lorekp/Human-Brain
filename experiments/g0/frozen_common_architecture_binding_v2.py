"""Complete non-executing nine-class binding for the frozen G0 protocol.

V1 remains an immutable terminal 3/9 ledger.  This successor binds its already
authenticated common configurations and generators to the separately banked
integrated checkpoint, execution/evaluation protocol, and exact environment
capsule.  It executes no generator, cohort, learner, evaluator, bootstrap,
adjudicative seed, protected input, or outcome.

The only generation authority exposed here is one module-owned exact object.
It is issued only after all nine classes and every frozen source identity have
been re-authenticated.  A boolean, string, copy, or caller-created object is not
an authorization.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

from experiments.g0 import frozen_common_architecture_binding_v1 as binding_v1
from experiments.g0 import frozen_cross_domain_g0_preregistration_v1 as prereg
from experiments.g0 import frozen_g0_dependency_environment_v1 as environment
from experiments.g0 import frozen_g0_execution_evaluation_protocol_v1 as execution
from experiments.g0 import (
    frozen_integrated_common_architecture_checkpoint_v1 as integrated,
)

SCOPE = "FROZEN_G0_COMMON_ARCHITECTURE_BINDING_V2"
STATUS = "AUTHENTICATED_COMPLETE_BINDING_9_OF_9"
CLAIM_BOUNDARY = (
    "complete-source-artifact-and-checkpoint-binding-only-no-g0-generation-or-result"
)
BOUND = "BOUND"

V1_BINDING_SOURCE_COMMIT = "c0b513a28162d3683a856e54e5e3a4640e362817"
INTEGRATED_ARCHITECTURE_SOURCE_COMMIT = (
    "20ee0eae68d698b5234b4cf1ebbb98e3cfd8c103"
)
EXECUTION_EVALUATION_SOURCE_COMMIT = (
    "5488574a375d215d3af90df74aa30c9d0d9ebd0d"
)
DEPENDENCY_ENVIRONMENT_SOURCE_COMMIT = (
    "5892aa3e859b7e42c8ba26b4382d55141844d77c"
)

EXPECTED_PREREGISTRATION_SHA256 = (
    "955828b48591f198047d5ca4966885e253763d6ea353351585edb14390320468"
)
EXPECTED_V1_BINDING_SOURCE_SHA256 = (
    "a5b9bf4ec1d139443f9e77f52b6f7659a45c60ffc999d3532a9b5faa3fc746f1"
)
EXPECTED_V1_BINDING_ROOT_SHA256 = (
    "957c3e3b81e5bedc826f330ff88c9d2ce2f83156b34e86300d38481fee7859f8"
)
EXPECTED_INTEGRATED_SOURCE_SHA256 = (
    "00afa1a4b44325b47408f6e758385652dcc1c63a79966ebd047d757267d9bdf7"
)
EXPECTED_INTEGRATED_CONFIG_SHA256 = (
    "66bb6a82842c4092b42db81325edc8a7a04570b6173ff9b876883de7f06e6f86"
)
EXPECTED_INTEGRATED_CURRICULUM_SHA256 = (
    "72e46776b7a4739ec5096af9d50a2db24a17de99e22f378bdd3c46f643bcfb1d"
)
EXPECTED_INTEGRATED_CHECKPOINT_SHA256 = (
    "4abaef5be4d962541b7a61ad9ae890e3811054d1885f95041c2aec65c097eafa"
)
EXPECTED_EXECUTION_SOURCE_SHA256 = (
    "afb5c1320cdaa4087576ef59213311ccf4b862375bb927eefdd3061563d565fc"
)
EXPECTED_EXECUTION_CONFIG_SHA256 = (
    "42fa98a076faaacbe98d9920f2540998b3e5af5f1f200814f5b19a645a47c4b3"
)
EXPECTED_EXECUTION_PROTOCOL_ROOT_SHA256 = (
    "72b196833da4ffaf40e8a3de56c92e5ee4eef701c460518e1e9b53239a581d24"
)
EXPECTED_DEPENDENCY_SOURCE_SHA256 = (
    "e2c86d3fa08927bbcba3540c67a9e77ae0cc8aa46ea4b50ef79bf672eae8a3e6"
)
EXPECTED_DEPENDENCY_CAPSULE_SHA256 = (
    "ce8ac9aa502183eb00fa2ed89c2c31065450af1c92a6149942fcff9d9b32494f"
)
EXPECTED_DEPENDENCY_LOCK_SHA256 = (
    "fc5594c7e9bbfe20c38ac448f2f9e449761397d499f588104b17b43372f950c4"
)
EXPECTED_ENVIRONMENT_SHA256 = (
    "0ec6e153c91404f764a77422207bfdf2e6c441c2059098de72445c0ee3b29af7"
)

COMPLETE_ARCHITECTURE_BINDING_COUNT = 1
GENERATION_AUTHORIZATION_OBJECT_COUNT = 1
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
_SOURCE_NAMESPACE = b"frozen-g0-common-architecture-binding-v2-source\0"
_CLASS_NAMESPACE = b"frozen-g0-common-architecture-binding-v2-class\0"
_CONFIG_NAMESPACE = b"frozen-g0-common-architecture-binding-v2-config\0"
_ROOT_NAMESPACE = b"frozen-g0-common-architecture-binding-v2-root\0"
_RECORD_NAMESPACE = b"frozen-g0-common-architecture-binding-v2-record\0"


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


def _record_root(label: str, value: object) -> str:
    return _digest(_RECORD_NAMESPACE, {"label": label, "value": value})


@dataclass(frozen=True, slots=True)
class FrozenSourceIdentity:
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
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class FrozenCommonArchitectureBindingV2:
    scope: str
    status: str
    claim_boundary: str
    source_sha256: str
    config_sha256: str
    preregistration_sha256: str
    v1_terminal_binding_sha256: str
    sources: tuple[FrozenSourceIdentity, ...]
    classes: tuple[ArtifactClassBinding, ...]
    bound_class_count: int
    missing_class_count: int
    complete_architecture_binding_count: int
    generation_authorized: bool
    generation_authorization_object_count: int
    domain_specific_primitive_ledger: tuple[tuple[str, int], ...]
    execution_counts: tuple[tuple[str, int], ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "scope": self.scope,
            "status": self.status,
            "claim_boundary": self.claim_boundary,
            "source_sha256": self.source_sha256,
            "config_sha256": self.config_sha256,
            "preregistration_sha256": self.preregistration_sha256,
            "v1_terminal_binding_sha256": self.v1_terminal_binding_sha256,
            "sources": tuple(row.record() for row in self.sources),
            "classes": tuple(row.record() for row in self.classes),
            "bound_class_count": self.bound_class_count,
            "missing_class_count": self.missing_class_count,
            "complete_architecture_binding_count": (
                self.complete_architecture_binding_count
            ),
            "generation_authorized": self.generation_authorized,
            "generation_authorization_object_count": (
                self.generation_authorization_object_count
            ),
            "domain_specific_primitive_ledger": (
                self.domain_specific_primitive_ledger
            ),
            "execution_counts": self.execution_counts,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


class G0CommonArchitectureBindingV2Stop(RuntimeError):
    """Raised before generation if any V2 identity or exact object differs."""


def _freeze_source(
    name: str,
    path: str,
    origin_commit_sha: str,
    source_sha256: str,
    artifact_sha256s: tuple[tuple[str, str], ...],
) -> FrozenSourceIdentity:
    record = {
        "name": name,
        "path": path,
        "origin_commit_sha": origin_commit_sha,
        "source_sha256": source_sha256,
        "artifact_sha256s": artifact_sha256s,
    }
    return FrozenSourceIdentity(
        **record,
        sha256=_digest(_SOURCE_NAMESPACE, record),
    )


_V1_SOURCES = tuple(
    _freeze_source(
        row.name,
        row.path,
        row.origin_commit_sha,
        row.source_sha256,
        (*row.artifact_sha256s, ("v1-source-identity", row.sha256)),
    )
    for row in binding_v1.SOURCES
)
SOURCES = (
    *_V1_SOURCES,
    _freeze_source(
        "terminal-common-architecture-binding-v1",
        "experiments/g0/frozen_common_architecture_binding_v1.py",
        V1_BINDING_SOURCE_COMMIT,
        EXPECTED_V1_BINDING_SOURCE_SHA256,
        (("terminal-binding-root", EXPECTED_V1_BINDING_ROOT_SHA256),),
    ),
    _freeze_source(
        "integrated-common-architecture-checkpoint-v1",
        "experiments/g0/frozen_integrated_common_architecture_checkpoint_v1.py",
        INTEGRATED_ARCHITECTURE_SOURCE_COMMIT,
        EXPECTED_INTEGRATED_SOURCE_SHA256,
        (
            ("config", EXPECTED_INTEGRATED_CONFIG_SHA256),
            ("curriculum", EXPECTED_INTEGRATED_CURRICULUM_SHA256),
            ("checkpoint", EXPECTED_INTEGRATED_CHECKPOINT_SHA256),
        ),
    ),
    _freeze_source(
        "execution-evaluation-protocol-v1",
        "experiments/g0/frozen_g0_execution_evaluation_protocol_v1.py",
        EXECUTION_EVALUATION_SOURCE_COMMIT,
        EXPECTED_EXECUTION_SOURCE_SHA256,
        (
            ("config", EXPECTED_EXECUTION_CONFIG_SHA256),
            ("protocol-root", EXPECTED_EXECUTION_PROTOCOL_ROOT_SHA256),
        ),
    ),
    _freeze_source(
        "dependency-environment-v1",
        "experiments/g0/frozen_g0_dependency_environment_v1.py",
        DEPENDENCY_ENVIRONMENT_SOURCE_COMMIT,
        EXPECTED_DEPENDENCY_SOURCE_SHA256,
        (
            ("capsule", EXPECTED_DEPENDENCY_CAPSULE_SHA256),
            ("dependency-lock", EXPECTED_DEPENDENCY_LOCK_SHA256),
            ("environment", EXPECTED_ENVIRONMENT_SHA256),
        ),
    ),
)
SOURCE_BY_NAME = MappingProxyType({row.name: row for row in SOURCES})

ACTION_ASSIGNMENT_SHA256 = _record_root(
    "action-assignment", execution.ACTION_ASSIGNMENT_RECORD
)
BASELINE_SHA256 = _record_root("simple-baselines", execution.BASELINE_RECORD)
ABLATION_SHA256 = _record_root("focused-ablations", execution.ABLATION_RECORD)
SEED_DERIVATION_SHA256 = _record_root(
    "seed-derivation", execution.SEED_DERIVATION_RECORD
)
EVALUATOR_SHA256 = _record_root("evaluator", execution.EVALUATOR_RECORD)

DOMAIN_SPECIFIC_PRIMITIVE_LEDGER = (
    ("common-event-cells", 0),
    ("common-event-associative-memory", 0),
    ("common-factorized-transition-memory", 0),
    (
        "integrated-common-architecture-checkpoint",
        integrated.DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA,
    ),
    (
        "raw-action-framing-seeding-baselines-ablations-and-evaluator",
        execution.DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA,
    ),
    ("regulatory-raw-serializer-adapter", 0),
    ("spatial-raw-serializer-adapter", 0),
    ("candidate-architecture-baseline-or-evaluator-primitive", 0),
)

EXECUTION_COUNTS = (
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


def _class_binding(
    ordinal: int,
    evidence: tuple[str, ...],
    artifact_sha256s: tuple[tuple[str, str], ...],
) -> ArtifactClassBinding:
    requirement = prereg.COMMON_ARCHITECTURE_FREEZE_REQUIREMENTS[ordinal - 1]
    record = {
        "ordinal": ordinal,
        "requirement": requirement,
        "status": BOUND,
        "evidence": evidence,
        "artifact_sha256s": artifact_sha256s,
    }
    return ArtifactClassBinding(
        **record,
        sha256=_digest(_CLASS_NAMESPACE, record),
    )


CLASSES = (
    _class_binding(
        1,
        ("integrated-common-architecture-checkpoint-v1",),
        (
            ("architecture-source-commit", INTEGRATED_ARCHITECTURE_SOURCE_COMMIT),
            ("source", EXPECTED_INTEGRATED_SOURCE_SHA256),
            ("config", EXPECTED_INTEGRATED_CONFIG_SHA256),
            ("curriculum", EXPECTED_INTEGRATED_CURRICULUM_SHA256),
            (
                "associative-checkpoint",
                integrated.EXPECTED_ASSOCIATIVE_CHECKPOINT_SHA256,
            ),
            (
                "factorized-transition-checkpoint",
                integrated.EXPECTED_TRANSITION_CHECKPOINT_SHA256,
            ),
            ("integrated-checkpoint", EXPECTED_INTEGRATED_CHECKPOINT_SHA256),
        ),
    ),
    _class_binding(
        2,
        (
            "terminal-common-architecture-binding-v1",
            "integrated-common-architecture-checkpoint-v1",
        ),
        (
            ("v1-class-2", binding_v1.CLASSES[1].sha256),
            ("integrated-config", EXPECTED_INTEGRATED_CONFIG_SHA256),
            ("event-cell-geometry", integrated.COMPONENT_ROOTS["common_projection_geometry"]),
            ("associative-geometry", integrated.COMPONENT_ROOTS["associative_geometry"]),
            (
                "factorized-transition-geometry",
                integrated.COMPONENT_ROOTS["factorized_transition_geometry"],
            ),
            (
                "factorized-branch-current-table",
                integrated.COMPONENT_ROOTS["factorized_branch_current_table"],
            ),
        ),
    ),
    _class_binding(
        3,
        (
            "regulatory-generator",
            "spatial-generator",
            "spatial-fixture-preflight",
        ),
        (
            ("v1-class-3", binding_v1.CLASSES[2].sha256),
            (
                "regulatory-generator-source",
                binding_v1.SOURCE_BY_NAME["regulatory-generator"].source_sha256,
            ),
            (
                "spatial-generator-source",
                binding_v1.SOURCE_BY_NAME["spatial-generator"].source_sha256,
            ),
            (
                "regulatory-source-specification",
                binding_v1.CLASSES[2].artifact_sha256s[0][1],
            ),
            (
                "spatial-generator-root",
                binding_v1.CLASSES[2].artifact_sha256s[-1][1],
            ),
        ),
    ),
    _class_binding(
        4,
        (
            "execution-evaluation-protocol-v1",
            "integrated-common-architecture-checkpoint-v1",
            "regulatory-adapter",
            "spatial-adapter",
        ),
        (
            ("action-assignment", ACTION_ASSIGNMENT_SHA256),
            ("execution-protocol", EXPECTED_EXECUTION_PROTOCOL_ROOT_SHA256),
            ("physical-output-checkpoint", EXPECTED_INTEGRATED_CHECKPOINT_SHA256),
            (
                "regulatory-adapter-source",
                binding_v1.SOURCE_BY_NAME["regulatory-adapter"].source_sha256,
            ),
            (
                "spatial-adapter-source",
                binding_v1.SOURCE_BY_NAME["spatial-adapter"].source_sha256,
            ),
        ),
    ),
    _class_binding(
        5,
        ("execution-evaluation-protocol-v1",),
        (
            ("simple-baselines", BASELINE_SHA256),
            ("focused-ablations", ABLATION_SHA256),
            ("execution-source", EXPECTED_EXECUTION_SOURCE_SHA256),
            ("execution-config", EXPECTED_EXECUTION_CONFIG_SHA256),
        ),
    ),
    _class_binding(
        6,
        ("execution-evaluation-protocol-v1", "preregistration"),
        (
            ("seed-derivation", SEED_DERIVATION_SHA256),
            ("execution-protocol", EXPECTED_EXECUTION_PROTOCOL_ROOT_SHA256),
            ("preregistration", EXPECTED_PREREGISTRATION_SHA256),
        ),
    ),
    _class_binding(
        7,
        ("preregistration", "terminal-common-architecture-binding-v1"),
        (
            ("v1-class-7", binding_v1.CLASSES[6].sha256),
            ("preregistration", EXPECTED_PREREGISTRATION_SHA256),
            ("statistics", prereg.PROTOCOL.statistics_sha256),
        ),
    ),
    _class_binding(
        8,
        ("execution-evaluation-protocol-v1", "preregistration"),
        (
            ("evaluator", EVALUATOR_SHA256),
            ("execution-source", EXPECTED_EXECUTION_SOURCE_SHA256),
            ("execution-protocol", EXPECTED_EXECUTION_PROTOCOL_ROOT_SHA256),
            ("statistics", prereg.PROTOCOL.statistics_sha256),
        ),
    ),
    _class_binding(
        9,
        ("dependency-environment-v1",),
        (
            ("source-commit", DEPENDENCY_ENVIRONMENT_SOURCE_COMMIT),
            ("source", EXPECTED_DEPENDENCY_SOURCE_SHA256),
            ("capsule", EXPECTED_DEPENDENCY_CAPSULE_SHA256),
            ("dependency-lock", EXPECTED_DEPENDENCY_LOCK_SHA256),
            ("environment", EXPECTED_ENVIRONMENT_SHA256),
            ("project-manifest", environment.PROJECT_MANIFEST_SHA256),
        ),
    ),
)
CLASS_BY_REQUIREMENT = MappingProxyType({row.requirement: row for row in CLASSES})

CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
    "v1_terminal_binding_sha256": EXPECTED_V1_BINDING_ROOT_SHA256,
    "banked_commits": (
        ("binding-v1", V1_BINDING_SOURCE_COMMIT),
        ("integrated-checkpoint", INTEGRATED_ARCHITECTURE_SOURCE_COMMIT),
        ("execution-evaluation", EXECUTION_EVALUATION_SOURCE_COMMIT),
        ("dependency-environment", DEPENDENCY_ENVIRONMENT_SOURCE_COMMIT),
    ),
    "sources": tuple(row.record() for row in SOURCES),
    "classes": tuple(row.record() for row in CLASSES),
    "domain_specific_primitive_ledger": DOMAIN_SPECIFIC_PRIMITIVE_LEDGER,
    "complete_architecture_binding_count": COMPLETE_ARCHITECTURE_BINDING_COUNT,
    "generation_authorization_object_count": GENERATION_AUTHORIZATION_OBJECT_COUNT,
    "execution_counts": EXECUTION_COUNTS,
}
CONFIG_SHA256 = _digest(_CONFIG_NAMESPACE, CONFIG_RECORD)
SOURCE_SHA256 = _source_sha256(_SOURCE_PATH)


def _authenticate_frozen_inputs() -> None:
    v1 = binding_v1.authenticate_binding()
    dependency = environment.authenticate_dependency_environment()
    source_expectations = {
        "experiments/g0/frozen_common_architecture_binding_v1.py": (
            EXPECTED_V1_BINDING_SOURCE_SHA256
        ),
        "experiments/g0/frozen_integrated_common_architecture_checkpoint_v1.py": (
            EXPECTED_INTEGRATED_SOURCE_SHA256
        ),
        "experiments/g0/frozen_g0_execution_evaluation_protocol_v1.py": (
            EXPECTED_EXECUTION_SOURCE_SHA256
        ),
        "experiments/g0/frozen_g0_dependency_environment_v1.py": (
            EXPECTED_DEPENDENCY_SOURCE_SHA256
        ),
    }
    for path, expected in source_expectations.items():
        if _source_sha256(path) != expected:
            raise G0CommonArchitectureBindingV2Stop(
                f"STOP_G0_BINDING_V2_SOURCE_IDENTITY_MISMATCH:{path}"
            )
    if (
        prereg.PROTOCOL.protocol_sha256 != EXPECTED_PREREGISTRATION_SHA256
        or v1.sha256 != EXPECTED_V1_BINDING_ROOT_SHA256
        or v1.status != binding_v1.STATUS
        or v1.bound_class_count != 3
        or v1.missing_class_count != 6
        or v1.generation_authorized is not False
        or integrated.SOURCE_SHA256 != EXPECTED_INTEGRATED_SOURCE_SHA256
        or integrated.CONFIG_SHA256 != EXPECTED_INTEGRATED_CONFIG_SHA256
        or integrated.CURRICULUM_SHA256 != EXPECTED_INTEGRATED_CURRICULUM_SHA256
        or integrated.EXPECTED_INTEGRATED_CHECKPOINT_SHA256
        != EXPECTED_INTEGRATED_CHECKPOINT_SHA256
        or integrated.EXPECTED_EXECUTION_PROTOCOL_ROOT_SHA256
        != EXPECTED_EXECUTION_PROTOCOL_ROOT_SHA256
        or execution.SOURCE_SHA256 != EXPECTED_EXECUTION_SOURCE_SHA256
        or execution.CONFIG_SHA256 != EXPECTED_EXECUTION_CONFIG_SHA256
        or execution.PROTOCOL_ROOT_SHA256
        != EXPECTED_EXECUTION_PROTOCOL_ROOT_SHA256
        or dependency.sha256 != EXPECTED_DEPENDENCY_CAPSULE_SHA256
        or dependency.dependency_lock_sha256 != EXPECTED_DEPENDENCY_LOCK_SHA256
        or dependency.environment_sha256 != EXPECTED_ENVIRONMENT_SHA256
        or tuple(row.requirement for row in CLASSES)
        != prereg.COMMON_ARCHITECTURE_FREEZE_REQUIREMENTS
        or tuple(row.ordinal for row in CLASSES) != tuple(range(1, 10))
        or not all(row.bound for row in CLASSES)
        or execution.MISSING_BASELINE_STOPS
        or execution.BASELINE_RECORD["candidate_dependency_count"] != 0
        or execution.BASELINE_RECORD["generator_dependency_count"] != 0
        or execution.BASELINE_RECORD["privileged_state_field_count"] != 0
        or any(value for _, value in DOMAIN_SPECIFIC_PRIMITIVE_LEDGER)
        or any(value for _, value in EXECUTION_COUNTS)
        or any(value for value in execution.ZERO_EXECUTION_RECORD.values())
        or any(value for value in integrated.ZERO_EXECUTION_RECORD.values())
        or any(value for _, value in dependency.execution_counts)
    ):
        raise G0CommonArchitectureBindingV2Stop(
            "STOP_G0_BINDING_V2_FROZEN_ARTIFACT_IDENTITY_MISMATCH"
        )


def _build_binding() -> FrozenCommonArchitectureBindingV2:
    _authenticate_frozen_inputs()
    root_record = {
        "scope": SCOPE,
        "status": STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_sha256": SOURCE_SHA256,
        "config_sha256": CONFIG_SHA256,
        "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
        "v1_terminal_binding_sha256": EXPECTED_V1_BINDING_ROOT_SHA256,
        "sources": tuple(row.record() for row in SOURCES),
        "classes": tuple(row.record() for row in CLASSES),
        "bound_class_count": 9,
        "missing_class_count": 0,
        "complete_architecture_binding_count": COMPLETE_ARCHITECTURE_BINDING_COUNT,
        "generation_authorized": True,
        "generation_authorization_object_count": GENERATION_AUTHORIZATION_OBJECT_COUNT,
        "domain_specific_primitive_ledger": DOMAIN_SPECIFIC_PRIMITIVE_LEDGER,
        "execution_counts": EXECUTION_COUNTS,
    }
    return FrozenCommonArchitectureBindingV2(
        scope=SCOPE,
        status=STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        source_sha256=SOURCE_SHA256,
        config_sha256=CONFIG_SHA256,
        preregistration_sha256=EXPECTED_PREREGISTRATION_SHA256,
        v1_terminal_binding_sha256=EXPECTED_V1_BINDING_ROOT_SHA256,
        sources=SOURCES,
        classes=CLASSES,
        bound_class_count=9,
        missing_class_count=0,
        complete_architecture_binding_count=COMPLETE_ARCHITECTURE_BINDING_COUNT,
        generation_authorized=True,
        generation_authorization_object_count=(
            GENERATION_AUTHORIZATION_OBJECT_COUNT
        ),
        domain_specific_primitive_ledger=DOMAIN_SPECIFIC_PRIMITIVE_LEDGER,
        execution_counts=EXECUTION_COUNTS,
        sha256=_digest(_ROOT_NAMESPACE, root_record),
    )


BINDING = _build_binding()

_AUTHORIZATION_FACTORY_TOKEN = object()


class _OpaqueExactGenerationAuthorization:
    __slots__ = ()

    def __new__(cls, token: object = None) -> _OpaqueExactGenerationAuthorization:
        if token is not _AUTHORIZATION_FACTORY_TOKEN:
            raise G0CommonArchitectureBindingV2Stop(
                "STOP_G0_BINDING_V2_AUTHORIZATION_CONSTRUCTION_FORBIDDEN"
            )
        return super().__new__(cls)

    def __repr__(self) -> str:
        return "<opaque exact G0 generation authorization>"

    def __copy__(self) -> object:
        raise TypeError("opaque exact G0 generation authorization is not copyable")

    def __deepcopy__(self, _memo: object) -> object:
        raise TypeError("opaque exact G0 generation authorization is not copyable")

    def __reduce__(self) -> object:
        raise TypeError("opaque exact G0 generation authorization is not serializable")

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("opaque exact G0 generation authorization is not serializable")


_GENERATION_AUTHORIZATION = (
    _OpaqueExactGenerationAuthorization(_AUTHORIZATION_FACTORY_TOKEN)
    if BINDING.bound_class_count == 9
    and BINDING.missing_class_count == 0
    and BINDING.generation_authorized
    else None
)


def authenticate_binding() -> FrozenCommonArchitectureBindingV2:
    """Re-authenticate all nine classes without executing a G0 surface."""

    observed = _build_binding()
    if observed != BINDING or observed.sha256 != BINDING.sha256:
        raise G0CommonArchitectureBindingV2Stop(
            "STOP_G0_BINDING_V2_ROOT_IDENTITY_MISMATCH"
        )
    return BINDING


def issue_generation_authorization() -> object:
    """Return the one opaque exact object only while the 9/9 binding matches."""

    binding = authenticate_binding()
    if (
        binding.bound_class_count != 9
        or binding.missing_class_count != 0
        or not binding.generation_authorized
        or _GENERATION_AUTHORIZATION is None
    ):
        raise G0CommonArchitectureBindingV2Stop(
            "STOP_G0_BINDING_V2_GENERATION_NOT_AUTHORIZED"
        )
    return _GENERATION_AUTHORIZATION


def require_generation_authorization(
    authorization: object,
) -> FrozenCommonArchitectureBindingV2:
    """Accept only this module's exact singleton after fresh 9/9 authentication."""

    binding = authenticate_binding()
    if authorization is not _GENERATION_AUTHORIZATION:
        raise G0CommonArchitectureBindingV2Stop(
            "STOP_G0_BINDING_V2_EXACT_AUTHORIZATION_OBJECT_REQUIRED"
        )
    return binding


__all__ = [
    "ABLATION_SHA256",
    "ACTION_ASSIGNMENT_SHA256",
    "BASELINE_SHA256",
    "BINDING",
    "BOUND",
    "CLAIM_BOUNDARY",
    "CLASSES",
    "CLASS_BY_REQUIREMENT",
    "COMPLETE_ARCHITECTURE_BINDING_COUNT",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "DEPENDENCY_ENVIRONMENT_SOURCE_COMMIT",
    "DOMAIN_SPECIFIC_PRIMITIVE_LEDGER",
    "EVALUATOR_SHA256",
    "EXECUTION_COUNTS",
    "EXECUTION_EVALUATION_SOURCE_COMMIT",
    "GENERATION_AUTHORIZATION_OBJECT_COUNT",
    "INTEGRATED_ARCHITECTURE_SOURCE_COMMIT",
    "SCOPE",
    "SEED_DERIVATION_SHA256",
    "SOURCES",
    "SOURCE_BY_NAME",
    "SOURCE_SHA256",
    "STATUS",
    "ArtifactClassBinding",
    "FrozenCommonArchitectureBindingV2",
    "FrozenSourceIdentity",
    "G0CommonArchitectureBindingV2Stop",
    "authenticate_binding",
    "issue_generation_authorization",
    "require_generation_authorization",
]
