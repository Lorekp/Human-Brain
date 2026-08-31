"""Authenticated G0 cohort transaction preflight with an explicit capacity STOP.

The exact V2 plans and every upstream source/configuration root authenticate,
but the pinned spatial generator cannot satisfy its own preregistered
pairwise-distinct-layout condition. This module freezes that pre-spend STOP.
It never calls either generator and never publishes a partial artifact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path

from experiments.g0 import frozen_cross_domain_g0_preregistration_v1 as prereg
from experiments.g0 import frozen_g0_authorized_generator_entrypoints_v2 as entrypoints
from experiments.g0 import frozen_regulatory_generator_v1 as regulatory
from experiments.g0 import frozen_spatial_environment_generator_v1 as spatial

SCOPE = "FROZEN_G0_COHORT_MATERIALIZATION_TRANSACTION_V1"
STATUS = "FROZEN_PREFLIGHT_STOP_NO_COHORT_SPEND"
CLAIM_BOUNDARY = "authenticated-materialization-preflight-only-no-cohort-or-g0-result"

ENTRYPOINT_SOURCE_COMMIT = "29f8b843a3248f223f86a6b00111adea7f91c0c2"
EXPECTED_ENTRYPOINT_SOURCE_SHA256 = (
    "2cc495f10c47b29e75ba9a4154aed4a8fd8215af8c39ac3aa0ba828c093a11ac"
)
EXPECTED_ENTRYPOINT_CONFIG_SHA256 = (
    "ead19f30899616c332e350267f409558cedead24c767b49f8f4812b37ce026f4"
)
EXPECTED_REGULATORY_PLAN_SHA256 = "bd853d903b62913715eff38e171a14fa8d73d1126c96b7e884745bc5e00e5de3"
EXPECTED_SPATIAL_PLAN_SHA256 = "9c2c29edb9fc15fb3bdf9485d8b1b14b46ac6555de57517460b65550ea8aadae"
EXPECTED_ENTRYPOINT_ROOT_SHA256 = "5001c39610039b1822081b80ff2baf26fac007b4acd05e2e36d6570887ecf1d7"

MEMORY_S11_DEFINITION_ID = "learning-memory-s11-random"
MEMORY_S11_REQUIRED_LAYOUT_COUNT = 192
MEMORY_S11_HALLWAY_END_LOW = 4
MEMORY_S11_HALLWAY_END_HIGH_EXCLUSIVE = 9
MEMORY_S11_AGENT_POSITION_COUNT = sum(
    range(MEMORY_S11_HALLWAY_END_LOW, MEMORY_S11_HALLWAY_END_HIGH_EXCLUSIVE)
)
MEMORY_S11_START_OBJECT_CHOICES = 2
MEMORY_S11_TERMINAL_ORDER_CHOICES = 2
MEMORY_S11_MAXIMUM_DISTINCT_FULL_STATES = (
    MEMORY_S11_AGENT_POSITION_COUNT
    * MEMORY_S11_START_OBJECT_CHOICES
    * MEMORY_S11_TERMINAL_ORDER_CHOICES
)

ZERO_EXECUTION_COUNTS = (
    ("generator-root-seed", 0),
    ("regulatory-root-seed", 0),
    ("regulatory-system", 0),
    ("spatial-root-seed", 0),
    ("spatial-layout", 0),
    ("rename-root-seed", 0),
    ("adjudicative-root-seed", 0),
    ("cohort-artifact", 0),
    ("atomic-publication", 0),
    ("learner", 0),
    ("evaluator", 0),
    ("bootstrap", 0),
    ("outcome", 0),
    ("dev-input", 0),
    ("m0-input", 0),
    ("protected-input", 0),
)

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATH = Path(__file__).resolve()
_ENTRYPOINT_PATH = _ROOT / "experiments/g0/frozen_g0_authorized_generator_entrypoints_v2.py"
_CONFIG_NAMESPACE = b"frozen-g0-cohort-materialization-transaction-v1-config\0"
_STOP_NAMESPACE = b"frozen-g0-cohort-materialization-capacity-stop-v1\0"
_ROOT_NAMESPACE = b"frozen-g0-cohort-materialization-transaction-v1-root\0"


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
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


class G0CohortMaterializationStop(RuntimeError):
    """Raised before any generator call or artifact publication."""


@dataclass(frozen=True, slots=True)
class FrozenMaterializationStop:
    status: str
    blocker: str
    definition_id: str
    required_layout_count: int
    maximum_distinct_full_state_count: int
    derivation: tuple[tuple[str, object], ...]
    generator_seed_execution_count: int
    artifact_publication_count: int
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


_STOP_RECORD_WITHOUT_SHA = {
    "status": STATUS,
    "blocker": "spatial-memory-s11-distinct-layout-capacity-below-preregistered-count",
    "definition_id": MEMORY_S11_DEFINITION_ID,
    "required_layout_count": MEMORY_S11_REQUIRED_LAYOUT_COUNT,
    "maximum_distinct_full_state_count": MEMORY_S11_MAXIMUM_DISTINCT_FULL_STATES,
    "derivation": (
        (
            "hallway_end_values",
            tuple(
                range(
                    MEMORY_S11_HALLWAY_END_LOW,
                    MEMORY_S11_HALLWAY_END_HIGH_EXCLUSIVE,
                )
            ),
        ),
        ("agent_position_count_across_hallway_ends", MEMORY_S11_AGENT_POSITION_COUNT),
        ("start_object_choices", MEMORY_S11_START_OBJECT_CHOICES),
        ("terminal_object_order_choices", MEMORY_S11_TERMINAL_ORDER_CHOICES),
        ("agent_direction_choices", 1),
        ("object_color_choices", 1),
        ("maximum_product", MEMORY_S11_MAXIMUM_DISTINCT_FULL_STATES),
        ("pairwise_distinct_full_state_layout_hashes_required", True),
    ),
    "generator_seed_execution_count": 0,
    "artifact_publication_count": 0,
}
MATERIALIZATION_STOP = FrozenMaterializationStop(
    **_STOP_RECORD_WITHOUT_SHA,
    sha256=_digest(_STOP_NAMESPACE, _STOP_RECORD_WITHOUT_SHA),
)

CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "entrypoint_source_commit": ENTRYPOINT_SOURCE_COMMIT,
    "entrypoint_source_sha256": EXPECTED_ENTRYPOINT_SOURCE_SHA256,
    "entrypoint_config_sha256": EXPECTED_ENTRYPOINT_CONFIG_SHA256,
    "regulatory_plan_sha256": EXPECTED_REGULATORY_PLAN_SHA256,
    "spatial_plan_sha256": EXPECTED_SPATIAL_PLAN_SHA256,
    "entrypoint_root_sha256": EXPECTED_ENTRYPOINT_ROOT_SHA256,
    "materialization_stop": MATERIALIZATION_STOP.record(),
    "all_preconditions_before_generator_call": True,
    "ordinary_equal_same_type_copy_and_field_forged_plans_rejected": True,
    "partial_artifact_publication_allowed": False,
    "execution_counts": ZERO_EXECUTION_COUNTS,
}
CONFIG_SHA256 = _digest(_CONFIG_NAMESPACE, CONFIG_RECORD)


def materialization_execution_counts() -> tuple[tuple[str, int], ...]:
    return ZERO_EXECUTION_COUNTS


def _authenticate_exact_plans(regulatory_plan: object, spatial_plan: object) -> None:
    if regulatory_plan is not entrypoints.REGULATORY_PLAN:
        raise G0CohortMaterializationStop("STOP_G0_MATERIALIZATION_EXACT_REGULATORY_PLAN_REQUIRED")
    if spatial_plan is not entrypoints.SPATIAL_PLAN:
        raise G0CohortMaterializationStop("STOP_G0_MATERIALIZATION_EXACT_SPATIAL_PLAN_REQUIRED")


def _authenticate_static_preconditions() -> None:
    entrypoints._authenticate_static_sources()
    spatial._authenticate_dependencies()
    definition = spatial.DEFINITION_BY_ID[MEMORY_S11_DEFINITION_ID]
    if (
        sha256(_ENTRYPOINT_PATH.read_bytes()).hexdigest() != EXPECTED_ENTRYPOINT_SOURCE_SHA256
        or entrypoints.SOURCE_SHA256 != EXPECTED_ENTRYPOINT_SOURCE_SHA256
        or entrypoints.CONFIG_SHA256 != EXPECTED_ENTRYPOINT_CONFIG_SHA256
        or entrypoints.REGULATORY_PLAN.sha256 != EXPECTED_REGULATORY_PLAN_SHA256
        or entrypoints.SPATIAL_PLAN.sha256 != EXPECTED_SPATIAL_PLAN_SHA256
        or entrypoints.ENTRYPOINT_ROOT_SHA256 != EXPECTED_ENTRYPOINT_ROOT_SHA256
        or tuple(entrypoints.REGULATORY_PLAN.frozen_cohort_seeds)
        != tuple(regulatory.FROZEN_COHORT_SEEDS)
        or tuple(entrypoints.SPATIAL_PLAN.frozen_cohort_seeds) != tuple(spatial.FROZEN_COHORT_SEEDS)
        or entrypoints.REGULATORY_PLAN.generator_invocation_count != 0
        or entrypoints.SPATIAL_PLAN.generator_invocation_count != 0
        or entrypoints.REGULATORY_PLAN.execution_counts != entrypoints.ZERO_EXECUTION_COUNTS
        or entrypoints.SPATIAL_PLAN.execution_counts != entrypoints.ZERO_EXECUTION_COUNTS
        or prereg.GENERATOR_EXECUTION_COUNT != 0
        or prereg.COHORT_GENERATION_COUNT != 0
        or prereg.LEARNER_EXECUTION_COUNT != 0
        or prereg.EVALUATOR_EXECUTION_COUNT != 0
        or prereg.OUTCOME_COUNT != 0
        or regulatory.FROZEN_COHORT_EXECUTION_COUNT != 0
        or regulatory.FROZEN_SYSTEM_GENERATION_COUNT != 0
        or spatial.FROZEN_COHORT_EXECUTION_COUNT != 0
        or spatial.FROZEN_LAYOUT_GENERATION_COUNT != 0
        or definition.factory != "MemoryEnv"
        or definition.preregistered_layout_count != MEMORY_S11_REQUIRED_LAYOUT_COUNT
        or dict(definition.configuration) != {"size": 11, "random_length": True, "max_steps": 484}
        or prereg.SPATIAL_SPECIFICATION["pairwise_distinct_full_state_layout_hashes"] is not True
        or any(value for _name, value in ZERO_EXECUTION_COUNTS)
    ):
        raise G0CohortMaterializationStop("STOP_G0_MATERIALIZATION_FROZEN_PRECONDITION_MISMATCH")


def materialize_frozen_g0_cohorts(
    regulatory_plan: object,
    spatial_plan: object,
) -> None:
    """Authenticate the exact plans, then stop before the first frozen seed."""

    _authenticate_exact_plans(regulatory_plan, spatial_plan)
    _authenticate_static_preconditions()
    if MEMORY_S11_MAXIMUM_DISTINCT_FULL_STATES < MEMORY_S11_REQUIRED_LAYOUT_COUNT:
        raise G0CohortMaterializationStop(
            "STOP_G0_SPATIAL_MEMORY_S11_DISTINCT_LAYOUT_CAPACITY:"
            f"required={MEMORY_S11_REQUIRED_LAYOUT_COUNT}:"
            f"maximum={MEMORY_S11_MAXIMUM_DISTINCT_FULL_STATES}"
        )
    raise G0CohortMaterializationStop("STOP_G0_MATERIALIZATION_CAPACITY_PROOF_UNEXPECTEDLY_CHANGED")


SOURCE_SHA256 = sha256(_SOURCE_PATH.read_bytes()).hexdigest()
TRANSACTION_ROOT_SHA256 = _digest(
    _ROOT_NAMESPACE,
    {
        "source_sha256": SOURCE_SHA256,
        "config_sha256": CONFIG_SHA256,
        "entrypoint_source_commit": ENTRYPOINT_SOURCE_COMMIT,
        "entrypoint_root_sha256": EXPECTED_ENTRYPOINT_ROOT_SHA256,
        "regulatory_plan_sha256": EXPECTED_REGULATORY_PLAN_SHA256,
        "spatial_plan_sha256": EXPECTED_SPATIAL_PLAN_SHA256,
        "materialization_stop_sha256": MATERIALIZATION_STOP.sha256,
    },
)

__all__ = [
    "CLAIM_BOUNDARY",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "ENTRYPOINT_SOURCE_COMMIT",
    "MATERIALIZATION_STOP",
    "MEMORY_S11_MAXIMUM_DISTINCT_FULL_STATES",
    "MEMORY_S11_REQUIRED_LAYOUT_COUNT",
    "SCOPE",
    "SOURCE_SHA256",
    "STATUS",
    "TRANSACTION_ROOT_SHA256",
    "ZERO_EXECUTION_COUNTS",
    "FrozenMaterializationStop",
    "G0CohortMaterializationStop",
    "materialization_execution_counts",
    "materialize_frozen_g0_cohorts",
]
