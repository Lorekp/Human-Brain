"""Fail-closed G0 spatial generator preflight and quarantined fixtures.

Only byte-format fixtures are executable in this source.  Every frozen cohort
surface stops before inspecting a requested surface or a purported common-
architecture binding.  A future bound generator must be separately named.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

from experiments.g0 import frozen_cross_domain_g0_preregistration_v1 as prereg
from experiments.g0 import spatial_candidate_adapter_v1 as adapter

GENERATOR_SCOPE = "G0_SPATIAL_FIXTURE_GENERATOR_V1"
STOP_G0_COMMON_ARCHITECTURE_BINDING_REQUIRED = (
    "STOP_G0_COMMON_ARCHITECTURE_BINDING_REQUIRED"
)

PREREGISTRATION_COMMIT = "42278a7"
PREREGISTRATION_SOURCE_SHA256 = (
    "e027e598aa926b108e90208e1586a93c0da35716e165a746bb2732d980774395"
)
PREREGISTRATION_TEST_SHA256 = (
    "2f71368f0ad1aeca869a13ac5f153abf77d203fda208da265c330ceef7b367ac"
)
ESTABLISHED_PATTERN_REPORT_SHA256 = (
    "f1d22441f0f80333888b475bc95d0a33f1d693aad1ad54207e30a02d6811773d"
)
G0_DESIGN_REPORT_SHA256 = (
    "833ebeac019be738721c7b63858057309282dd611ef5c6620f91328d5fb1a6ea"
)
PREREGISTRATION_PROTOCOL_SHA256 = (
    "955828b48591f198047d5ca4966885e253763d6ea353351585edb14390320468"
)
PREREGISTRATION_ADAPTER_SHA256 = (
    "29a6dcec69d89c74c373e7f4bf8709e5e57d7cc06d3aa15f8ad3a2691db1fd99"
)
PREREGISTRATION_SPATIAL_SHA256 = (
    "8705597cd610e00390a73bcd926fc8b3bcbfbd8be07b08646214972ad8402100"
)

REQUIRED_BINDING_ARTIFACTS = prereg.COMMON_ARCHITECTURE_FREEZE_REQUIREMENTS
FROZEN_COHORT_SURFACES = ("smoke", "learning", "adjudicative", "rename")
QUARANTINED_FIXTURE_SEEDS = (910_001, 910_002)
FIXTURE_STEP_COUNT = len(adapter.RAW_ACTION_IDS)

MINIGRID_AVAILABLE = importlib.util.find_spec("minigrid") is not None
MINIGRID_DEPENDENCY_STATUS = (
    "AVAILABLE_NOT_IMPORTED" if MINIGRID_AVAILABLE else "ABSENT_NO_INSTALL_ATTEMPTED"
)

SURFACE_BOUNDARY_RECORD: Mapping[str, object] = MappingProxyType(
    {
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
)

_ROOT = Path(__file__).resolve().parents[2]
_DEPENDENCY_FILES = MappingProxyType(
    {
        "preregistration_source": (
            _ROOT / "experiments/g0/frozen_cross_domain_g0_preregistration_v1.py",
            PREREGISTRATION_SOURCE_SHA256,
        ),
        "preregistration_test": (
            _ROOT / "tests/test_frozen_cross_domain_g0_preregistration_v1.py",
            PREREGISTRATION_TEST_SHA256,
        ),
        "established_pattern_report": (
            _ROOT / "docs/research/g0-established-benchmark-patterns.md",
            ESTABLISHED_PATTERN_REPORT_SHA256,
        ),
        "g0_design_report": (
            _ROOT / "docs/research/frozen-architecture-cross-domain-generalization.md",
            G0_DESIGN_REPORT_SHA256,
        ),
    }
)
_FIXTURE_NAMESPACE = b"g0-spatial-quarantined-fixture-v1\0"


class G0CommonArchitectureBindingRequired(RuntimeError):
    """Frozen cohort generation has no authenticated authority yet."""


class G0FixtureBoundaryError(ValueError):
    """A request was outside the two declared non-cohort fixtures."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _authenticate_frozen_dependencies() -> None:
    if (
        prereg.PROTOCOL.protocol_sha256 != PREREGISTRATION_PROTOCOL_SHA256
        or prereg.PROTOCOL.adapter_sha256 != PREREGISTRATION_ADAPTER_SHA256
        or prereg.PROTOCOL.spatial_sha256 != PREREGISTRATION_SPATIAL_SHA256
        or tuple(REQUIRED_BINDING_ARTIFACTS)
        != tuple(prereg.COMMON_ARCHITECTURE_FREEZE_REQUIREMENTS)
        or len(REQUIRED_BINDING_ARTIFACTS) != 9
    ):
        raise RuntimeError("STOP_G0_PREREGISTRATION_BINDING_MISMATCH")
    for path, expected_sha256 in _DEPENDENCY_FILES.values():
        if sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise RuntimeError("STOP_G0_PREREGISTRATION_SOURCE_BINDING_MISMATCH")


_authenticate_frozen_dependencies()


def generate_frozen_spatial_cohort(
    *, surface: object, architecture_binding: object | None = None
) -> None:
    """Stop before inspecting either argument; no binding authority exists."""

    raise G0CommonArchitectureBindingRequired(
        STOP_G0_COMMON_ARCHITECTURE_BINDING_REQUIRED
    )


@dataclass(frozen=True, slots=True)
class QuarantinedFixture:
    fixture_name: str
    fixture_seed: int
    quarantined: bool
    is_frozen_cohort: bool
    fixture_execution_count: int
    frozen_cohort_generation_count: int
    candidate_transcript: bytes
    candidate_transcript_sha256: str
    negative_field_audit_sha256: str
    raw_action_ids: tuple[int, ...]
    step_count: int
    dependency_status: str
    fixture_sha256: str

    def root_record(self, *, include_fixture_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "scope": GENERATOR_SCOPE,
            "fixture_name": self.fixture_name,
            "fixture_seed": self.fixture_seed,
            "quarantined": self.quarantined,
            "is_frozen_cohort": self.is_frozen_cohort,
            "fixture_execution_count": self.fixture_execution_count,
            "frozen_cohort_generation_count": self.frozen_cohort_generation_count,
            "candidate_transcript_sha256": self.candidate_transcript_sha256,
            "negative_field_audit_sha256": self.negative_field_audit_sha256,
            "raw_action_ids": self.raw_action_ids,
            "step_count": self.step_count,
            "dependency_status": self.dependency_status,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preregistration_protocol_sha256": PREREGISTRATION_PROTOCOL_SHA256,
            "required_binding_artifact_count": len(REQUIRED_BINDING_ARTIFACTS),
            "surface_boundary": dict(SURFACE_BOUNDARY_RECORD),
        }
        if include_fixture_sha256:
            record["fixture_sha256"] = self.fixture_sha256
        return record


def _opaque_fixture_image(fixture_seed: int, step_index: int) -> bytes:
    """Return shape-only bytes; no object/color/state codebook is interpreted."""

    return bytes(
        (
            fixture_seed
            + step_index * 17
            + byte_index * 31
            + (byte_index // adapter.IMAGE_SHAPE[2]) * 7
        )
        & 0xFF
        for byte_index in range(adapter.IMAGE_BYTE_COUNT)
    )


def generate_quarantined_fixture(fixture_seed: int) -> QuarantinedFixture:
    """Generate one declared serialization fixture, never an environment cohort."""

    if type(fixture_seed) is not int or fixture_seed not in QUARANTINED_FIXTURE_SEEDS:
        raise G0FixtureBoundaryError("STOP_G0_QUARANTINED_FIXTURE_NOT_DECLARED")
    fixture_index = QUARANTINED_FIXTURE_SEEDS.index(fixture_seed)
    records = tuple(
        {
            "image": _opaque_fixture_image(fixture_seed, step_index),
            "direction": (fixture_seed + step_index) % len(adapter.DIRECTION_IDS),
            "action_id": action_id,
            "reward": 1.0 if step_index == FIXTURE_STEP_COUNT - 1 else 0.0,
            "terminated": step_index == FIXTURE_STEP_COUNT - 1,
            "truncated": False,
        }
        for step_index, action_id in enumerate(adapter.RAW_ACTION_IDS)
    )
    field_audit = adapter.audit_candidate_record(records[0])
    transcript = adapter.transcript_from_records(records)
    transcript_sha = adapter.transcript_sha256(transcript)
    root_without_sha = {
        "scope": GENERATOR_SCOPE,
        "fixture_name": f"quarantined-byte-round-trip-{fixture_index}",
        "fixture_seed": fixture_seed,
        "quarantined": True,
        "is_frozen_cohort": False,
        "fixture_execution_count": 1,
        "frozen_cohort_generation_count": 0,
        "candidate_transcript_sha256": transcript_sha,
        "negative_field_audit_sha256": field_audit.sha256,
        "raw_action_ids": adapter.RAW_ACTION_IDS,
        "step_count": FIXTURE_STEP_COUNT,
        "dependency_status": MINIGRID_DEPENDENCY_STATUS,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_protocol_sha256": PREREGISTRATION_PROTOCOL_SHA256,
        "required_binding_artifact_count": len(REQUIRED_BINDING_ARTIFACTS),
        "surface_boundary": dict(SURFACE_BOUNDARY_RECORD),
    }
    fixture_sha = sha256(
        _FIXTURE_NAMESPACE + _canonical_json(root_without_sha)
    ).hexdigest()
    return QuarantinedFixture(
        fixture_name=root_without_sha["fixture_name"],  # type: ignore[arg-type]
        fixture_seed=fixture_seed,
        quarantined=True,
        is_frozen_cohort=False,
        fixture_execution_count=1,
        frozen_cohort_generation_count=0,
        candidate_transcript=transcript,
        candidate_transcript_sha256=transcript_sha,
        negative_field_audit_sha256=field_audit.sha256,
        raw_action_ids=adapter.RAW_ACTION_IDS,
        step_count=FIXTURE_STEP_COUNT,
        dependency_status=MINIGRID_DEPENDENCY_STATUS,
        fixture_sha256=fixture_sha,
    )


__all__ = [
    "ESTABLISHED_PATTERN_REPORT_SHA256",
    "FIXTURE_STEP_COUNT",
    "FROZEN_COHORT_SURFACES",
    "G0_DESIGN_REPORT_SHA256",
    "GENERATOR_SCOPE",
    "MINIGRID_AVAILABLE",
    "MINIGRID_DEPENDENCY_STATUS",
    "PREREGISTRATION_ADAPTER_SHA256",
    "PREREGISTRATION_COMMIT",
    "PREREGISTRATION_PROTOCOL_SHA256",
    "PREREGISTRATION_SOURCE_SHA256",
    "PREREGISTRATION_SPATIAL_SHA256",
    "PREREGISTRATION_TEST_SHA256",
    "QUARANTINED_FIXTURE_SEEDS",
    "REQUIRED_BINDING_ARTIFACTS",
    "STOP_G0_COMMON_ARCHITECTURE_BINDING_REQUIRED",
    "SURFACE_BOUNDARY_RECORD",
    "G0CommonArchitectureBindingRequired",
    "G0FixtureBoundaryError",
    "QuarantinedFixture",
    "generate_frozen_spatial_cohort",
    "generate_quarantined_fixture",
]
