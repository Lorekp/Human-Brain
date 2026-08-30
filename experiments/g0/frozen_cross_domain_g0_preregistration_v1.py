"""Data-only preregistration for the frozen-architecture G0 gate.

This module turns the primary-source benchmark-pattern report into one
machine-checkable protocol.  It contains no generator RNG, environment,
learner, evaluator, cohort, outcome, or protected coding input.  Cohort
generation remains forbidden until a separately frozen common-architecture
binding authenticates every required artifact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

PROTOCOL_SCOPE = "FROZEN_ARCH_CROSS_DOMAIN_G0_PREREGISTRATION_V1"
PROTOCOL_STATUS = "FROZEN_DESIGN_AWAITING_COMMON_ARCHITECTURE_BINDING"

LEARNER_SEEDS = (17, 29, 43, 71, 101)
BOOTSTRAP_RESAMPLES = 50_000
BOOTSTRAP_CONFIDENCE = 0.95

ARCHITECTURE_BINDING_COUNT = 0
GENERATOR_EXECUTION_COUNT = 0
COHORT_GENERATION_COUNT = 0
LEARNER_EXECUTION_COUNT = 0
EVALUATOR_EXECUTION_COUNT = 0
OUTCOME_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0

_PROTOCOL_NAMESPACE = b"frozen-cross-domain-g0-preregistration-v1\0"
_DOMAIN_NAMESPACE = b"frozen-cross-domain-g0-domain-v1\0"
_ADAPTER_NAMESPACE = b"frozen-cross-domain-g0-adapter-v1\0"
_STATISTICS_NAMESPACE = b"frozen-cross-domain-g0-statistics-v1\0"
_SURFACE_NAMESPACE = b"frozen-cross-domain-g0-surface-v1\0"


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


def _namespaced_sha256(namespace: bytes, value: object) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class CohortSplit:
    smoke_seed: int
    smoke_count: int
    learning_seed: int
    learning_count: int
    adjudicative_seed: int
    adjudicative_count: int
    rename_seed: int
    rename_scope: str

    def record(self) -> dict[str, object]:
        return {
            "smoke": {"seed": self.smoke_seed, "count": self.smoke_count},
            "learning": {"seed": self.learning_seed, "count": self.learning_count},
            "adjudicative": {
                "seed": self.adjudicative_seed,
                "count": self.adjudicative_count,
            },
            "rename": {"seed": self.rename_seed, "scope": self.rename_scope},
            "whole_system_or_layout_holdout": True,
            "canonical_identity_overlap_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class MetricThreshold:
    metric: str
    point_estimate_minimum: float
    lower_confidence_bound_minimum: float

    def record(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "point_estimate_minimum": self.point_estimate_minimum,
            "lower_confidence_bound_minimum": self.lower_confidence_bound_minimum,
        }


@dataclass(frozen=True, slots=True)
class FrozenDomainProtocol:
    domain: str
    split: CohortSplit
    generator_specification: object
    support_and_query_schedule: object
    thresholds: tuple[MetricThreshold, ...]
    baselines: tuple[str, ...]
    rename_requirements: object
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "domain": self.domain,
            "split": self.split.record(),
            "generator_specification": self.generator_specification,
            "support_and_query_schedule": self.support_and_query_schedule,
            "thresholds": tuple(row.record() for row in self.thresholds),
            "baselines": self.baselines,
            "rename_requirements": self.rename_requirements,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class FrozenG0Preregistration:
    scope: str
    status: str
    learner_seeds: tuple[int, ...]
    event_envelope: object
    adapter_contract: object
    regulatory: FrozenDomainProtocol
    spatial: FrozenDomainProtocol
    common_architecture_freeze_requirements: tuple[str, ...]
    focused_ablations: object
    statistics: object
    terminal_gate_requirements: tuple[str, ...]
    adapter_sha256: str
    regulatory_sha256: str
    spatial_sha256: str
    statistics_sha256: str
    surface_boundary_sha256: str
    protocol_sha256: str

    def root_record(self, *, include_protocol_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "scope": self.scope,
            "status": self.status,
            "learner_seeds": self.learner_seeds,
            "event_envelope": self.event_envelope,
            "adapter_sha256": self.adapter_sha256,
            "regulatory_sha256": self.regulatory_sha256,
            "spatial_sha256": self.spatial_sha256,
            "statistics_sha256": self.statistics_sha256,
            "surface_boundary_sha256": self.surface_boundary_sha256,
            "common_architecture_freeze_requirements": (
                self.common_architecture_freeze_requirements
            ),
            "focused_ablations": self.focused_ablations,
            "terminal_gate_requirements": self.terminal_gate_requirements,
            "architecture_binding_count": ARCHITECTURE_BINDING_COUNT,
            "cohort_generation_count": COHORT_GENERATION_COUNT,
            "outcome_count": OUTCOME_COUNT,
        }
        if include_protocol_sha256:
            record["protocol_sha256"] = self.protocol_sha256
        return record


EVENT_ENVELOPE = _freeze(
    {
        "observation": ("raw_byte_tensor", "shape", "time"),
        "action": ("discrete_token", "optional_raw_entity_or_value_operands"),
        "outcome": ("raw_scalar_or_next_observation",),
        "boundary": ("terminated", "truncated"),
        "query": ("raw_target_token", "answer_slot_token"),
        "shape_and_padding_are_serialization_only": True,
    }
)

ADAPTER_CONTRACT = _freeze(
    {
        "common_requirements": (
            "deterministic-parameter-free-serialization",
            "byte-exact-permitted-field-round-trip",
            "forbidden-field-negative-audit",
            "candidate-transcript-sha256",
            "gold-state-visible-only-to-separate-evaluator-after-output",
            "no-domain-generator-or-evaluator-dependency-reachable-from-candidate",
        ),
        "regulatory_allowed": (
            "current-binary-entity-states",
            "raw-intervention-entity-and-value",
            "time",
            "observed-consequences",
        ),
        "regulatory_forbidden": (
            "adjacency",
            "edge-signs",
            "update-equations",
            "ancestors-or-reachability",
            "topological-order",
            "causal-effects",
            "counterfactual-twins",
            "graph-or-motif-identity",
            "gold-answers",
        ),
        "spatial_allowed": (
            "partial-image-7x7x3",
            "direction",
            "raw-action-id",
            "reward",
            "terminated",
            "truncated",
        ),
        "spatial_forbidden": (
            "mission",
            "full-grid",
            "absolute-agent-position",
            "visibility-mask",
            "shortest-path",
            "object-tracking-identity",
            "solver-state",
            "codebook-inverse",
            "layout-or-family-identity",
            "oracle-actions",
        ),
        "domain_specific_computational_primitive_delta_required": 0,
    }
)

REGULATORY_SPLIT = CohortSplit(
    470000,
    64,
    470001,
    512,
    470002,
    128,
    470003,
    "learning+adjudicative",
)
REGULATORY_THRESHOLDS = (
    MetricThreshold("all-heldout-queries", 0.82, 0.78),
    MetricThreshold("next-state-prediction", 0.85, 0.80),
    MetricThreshold("single-intervention", 0.80, 0.75),
    MetricThreshold("paired-counterfactual", 0.75, 0.70),
    MetricThreshold("relation-composition", 0.75, 0.70),
    MetricThreshold("delayed-memory", 0.75, 0.70),
)
REGULATORY_SPECIFICATION = _freeze(
    {
        "entity_count": 8,
        "edge_count_range": (10, 16),
        "self_edges": False,
        "in_degree_range": (1, 3),
        "edge_weights": (-1, 1),
        "biases": (-1, 0, 1),
        "transition": "lagged-deterministic-signed-threshold",
        "cycles_allowed": True,
        "opaque_entity_token_bits": 32,
        "generator_rejections": (
            "each-edge-changes-child-under-at-least-one-of-256-current-states",
            "each-node-takes-both-binary-values",
            "each-query-has-one-binary-answer",
            "one-connected-underlying-component",
        ),
        "adjudicative_partition": {
            "fresh-structure": 64,
            "heldout-composition": 64,
        },
        "composition_motifs": (
            "signed-length-3-or-4-path-joined-to-fork",
            "nested-or-converging-diamond",
        ),
    }
)
REGULATORY_SCHEDULE = _freeze(
    {
        "support_episodes_per_system": 40,
        "unmanipulated_trajectories": 8,
        "single_clamp_trajectories": 32,
        "transitions_per_trajectory": 8,
        "query_strata_per_system": {
            "next-state-prediction": 12,
            "single-intervention": 12,
            "paired-counterfactual": 12,
            "relation-composition": 6,
            "delayed-memory": 6,
        },
        "queries_per_system": 48,
        "queries_per_learner_seed": 6_144,
        "delayed_memory_irrelevant_event_count": 16,
        "adjudicative_systems_isolated_from_each_other": True,
        "support_and_query_tapes_disjoint": True,
        "answer_bit_balanced_within_each_stratum": True,
    }
)

SPATIAL_SPLIT = CohortSplit(470100, 64, 470101, 512, 470102, 128, 470103, "learning+adjudicative")
SPATIAL_THRESHOLDS = (
    MetricThreshold("all-heldout-layout-success", 0.70, 0.65),
    MetricThreshold("fresh-memory-layouts", 0.75, 0.68),
    MetricThreshold("fresh-key-door-layouts", 0.70, 0.63),
    MetricThreshold("heldout-compositions", 0.60, 0.55),
    MetricThreshold("delayed-cue-subset", 0.70, 0.63),
    MetricThreshold("multi-step-intervention-subset", 0.60, 0.55),
)
SPATIAL_SPECIFICATION = _freeze(
    {
        "partial_observation_shape": (7, 7, 3),
        "direction_exposed": True,
        "mission_exposed": False,
        "discrete_action_count": 7,
        "reward": "sparse-terminal",
        "timeout": "4*side_length^2",
        "learning_layout_partition": {
            "memory-s11-random-style": 192,
            "door-key-8x8-style": 160,
            "key-corridor-s3r2-style": 160,
        },
        "adjudicative_layout_partition": {
            "memory-s13-random-style": 32,
            "door-key": 16,
            "key-corridor": 16,
            "cue-key-door-composition": 64,
        },
        "composition_constraint": (
            "initial-cue-selects-one-of-two-keys; matching-key-opens-locked-door; "
            "door-gates-terminal-object-or-goal"
        ),
        "cue_absent_before_irreversible_choice_min_actions": 16,
        "pairwise_distinct_full_state_layout_hashes": True,
        "independent_oracle_solvability_check_before_manifest_seal": True,
    }
)
SPATIAL_SCHEDULE = _freeze(
    {
        "learning_episodes_per_layout": 32,
        "learning_episodes_per_learner_seed": 16_384,
        "adjudicative_rollouts_per_layout": 4,
        "adjudicative_episodes_all_learner_seeds": 2_560,
        "fixed_seed_derived_learning_shuffle": True,
        "validation_selection_or_early_stopping": False,
        "inter_layout_or_inter_rollout_weight_leakage": False,
        "within_episode_recurrent_and_working_memory_active": True,
    }
)


def _domain(
    name: str,
    split: CohortSplit,
    specification: object,
    schedule: object,
    thresholds: tuple[MetricThreshold, ...],
    baselines: tuple[str, ...],
    rename_requirements: object,
) -> FrozenDomainProtocol:
    record = {
        "domain": name,
        "split": split.record(),
        "generator_specification": specification,
        "support_and_query_schedule": schedule,
        "thresholds": tuple(row.record() for row in thresholds),
        "baselines": baselines,
        "rename_requirements": rename_requirements,
    }
    return FrozenDomainProtocol(
        domain=name,
        split=split,
        generator_specification=specification,
        support_and_query_schedule=schedule,
        thresholds=thresholds,
        baselines=baselines,
        rename_requirements=_freeze(rename_requirements),
        sha256=_namespaced_sha256(_DOMAIN_NAMESPACE + name.encode("ascii") + b"\0", record),
    )


REGULATORY = _domain(
    "regulatory",
    REGULATORY_SPLIT,
    REGULATORY_SPECIFICATION,
    REGULATORY_SCHEDULE,
    REGULATORY_THRESHOLDS,
    (
        "balanced-coin",
        "target-state-persistence",
        "support-marginal-majority",
        "serialized-support-prefix-nearest-neighbor",
        "observational-correlation-without-intervention-semantics",
    ),
    {
        "mapped-query-agreement-minimum": 0.90,
        "absolute-accuracy-change-maximum": 0.03,
        "bijection_scope": "every-entity-token-in-learning-and-adjudicative-transcripts",
        "inverse_bijection_exposed_to_learner": False,
    },
)
SPATIAL = _domain(
    "spatial",
    SPATIAL_SPLIT,
    SPATIAL_SPECIFICATION,
    SPATIAL_SCHEDULE,
    SPATIAL_THRESHOLDS,
    (
        "uniform-random-action",
        "fixed-wall-follower",
        "nearest-observation-action-memorizer",
        "memoryless-feed-forward-policy-matched-transition-budget",
    ),
    {
        "absolute-success-change-maximum": 0.05,
        "paired-success-failure-agreement-minimum": 0.85,
        "bijection_scope": "every-object-and-color-token-in-learning-and-adjudicative-transcripts",
        "inverse_bijection_exposed_to_learner": False,
    },
)

COMMON_ARCHITECTURE_FREEZE_REQUIREMENTS = (
    "architecture-source-commit-and-checkpoint-sha256",
    "cell-plasticity-memory-competition-and-update-budget-configuration",
    "both-generator-sources-and-complete-configurations",
    "raw-data-only-serializer-and-action-output-token-assignment",
    "baseline-and-ablation-definitions",
    "learner-generator-rollout-and-rename-seeds",
    "split-query-metric-threshold-and-gate-logic",
    "evaluator-source-and-50000-resample-bootstrap-procedure",
    "dependency-lock-and-environment-or-container-identity",
)
FOCUSED_ABLATIONS = _freeze(
    {
        "learning-off": {"minimum_score_drop_each_domain": 0.15, "must_fail_absolute_gate": True},
        "cell-or-output-population-off": {"maximum_above_strongest_simple_baseline": 0.02},
        "working-and-persistent-memory-reset-each-observation": {
            "minimum_delayed_slice_drop_each_domain": 0.20
        },
        "fixed-representation-consequence-action-derangement": {
            "minimum_intervention_or_composition_drop_each_domain": 0.15
        },
        "post_outcome_ablation_expansion_forbidden": True,
    }
)
STATISTICS = _freeze(
    {
        "learner_seeds": LEARNER_SEEDS,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "bootstrap_unit": "whole-system-or-layout-and-learner-seed",
        "paired_identity_preserved_for_baseline-and-rename-gaps": True,
        "report": (
            "every-seed-and-system-or-layout-result",
            "mean",
            "interquartile-mean",
            "95-percent-interval",
            "performance-profile",
            "update-events",
            "simulated-cell-events",
            "environment-transitions",
            "wall-time",
        ),
        "strongest_simple_baseline_overall_gap_minimum": 0.10,
        "paired_gap_lower_confidence_bound_minimum": 0.05,
        "failed_stratum_rescue_by_aggregate_forbidden": True,
    }
)
TERMINAL_GATE_REQUIREMENTS = (
    "both-noncoding-domains-pass-every-absolute-confidence-baseline-rename-and-ablation-gate",
    "all-domain-and-rename-arms-authenticate-one-common-frozen-architecture",
    "domain-specific-computational-primitive-delta-is-zero",
    "adapter-audits-and-candidate-transcript-hashes-are-clean",
    "no-protected-coding-or-hidden-g0-outcome-influenced-design",
    "one-domain-cannot-average-over-the-other-domain-failure",
)
SURFACE_BOUNDARY_RECORD = _freeze(
    {
        "architecture_binding_count": ARCHITECTURE_BINDING_COUNT,
        "generator_execution_count": GENERATOR_EXECUTION_COUNT,
        "cohort_generation_count": COHORT_GENERATION_COUNT,
        "learner_execution_count": LEARNER_EXECUTION_COUNT,
        "evaluator_execution_count": EVALUATOR_EXECUTION_COUNT,
        "outcome_count": OUTCOME_COUNT,
        "dev_input_count": DEV_INPUT_COUNT,
        "m0_input_count": M0_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "generation_authorized": False,
        "common_architecture_binding_required_before_generation": True,
        "data_only": True,
    }
)


def _build_protocol() -> FrozenG0Preregistration:
    if (
        len(LEARNER_SEEDS) != 5
        or len(set(LEARNER_SEEDS)) != 5
        or REGULATORY_SPLIT.learning_count != 512
        or REGULATORY_SPLIT.adjudicative_count != 128
        or sum(REGULATORY_SCHEDULE["query_strata_per_system"].values()) != 48
        or REGULATORY_SPLIT.adjudicative_count * 48 != 6_144
        or SPATIAL_SPLIT.learning_count != 512
        or SPATIAL_SPLIT.adjudicative_count != 128
        or sum(SPATIAL_SPECIFICATION["learning_layout_partition"].values()) != 512
        or sum(SPATIAL_SPECIFICATION["adjudicative_layout_partition"].values()) != 128
        or 512 * 32 != 16_384
        or 128 * 4 * len(LEARNER_SEEDS) != 2_560
        or any(
            SURFACE_BOUNDARY_RECORD[key]
            for key in (
                "architecture_binding_count",
                "generator_execution_count",
                "cohort_generation_count",
                "learner_execution_count",
                "evaluator_execution_count",
                "outcome_count",
                "dev_input_count",
                "m0_input_count",
                "protected_input_count",
            )
        )
        or SURFACE_BOUNDARY_RECORD["generation_authorized"] is not False
        or SURFACE_BOUNDARY_RECORD["common_architecture_binding_required_before_generation"]
        is not True
        or SURFACE_BOUNDARY_RECORD["data_only"] is not True
    ):
        raise RuntimeError("G0 preregistration arithmetic or unspent boundary changed")

    adapter_sha = _namespaced_sha256(_ADAPTER_NAMESPACE, ADAPTER_CONTRACT)
    statistics_sha = _namespaced_sha256(_STATISTICS_NAMESPACE, STATISTICS)
    surface_sha = _namespaced_sha256(_SURFACE_NAMESPACE, SURFACE_BOUNDARY_RECORD)
    root_without_protocol = {
        "scope": PROTOCOL_SCOPE,
        "status": PROTOCOL_STATUS,
        "learner_seeds": LEARNER_SEEDS,
        "event_envelope": EVENT_ENVELOPE,
        "adapter_sha256": adapter_sha,
        "regulatory_sha256": REGULATORY.sha256,
        "spatial_sha256": SPATIAL.sha256,
        "statistics_sha256": statistics_sha,
        "surface_boundary_sha256": surface_sha,
        "common_architecture_freeze_requirements": COMMON_ARCHITECTURE_FREEZE_REQUIREMENTS,
        "focused_ablations": FOCUSED_ABLATIONS,
        "terminal_gate_requirements": TERMINAL_GATE_REQUIREMENTS,
        "architecture_binding_count": ARCHITECTURE_BINDING_COUNT,
        "cohort_generation_count": COHORT_GENERATION_COUNT,
        "outcome_count": OUTCOME_COUNT,
    }
    return FrozenG0Preregistration(
        scope=PROTOCOL_SCOPE,
        status=PROTOCOL_STATUS,
        learner_seeds=LEARNER_SEEDS,
        event_envelope=EVENT_ENVELOPE,
        adapter_contract=ADAPTER_CONTRACT,
        regulatory=REGULATORY,
        spatial=SPATIAL,
        common_architecture_freeze_requirements=COMMON_ARCHITECTURE_FREEZE_REQUIREMENTS,
        focused_ablations=FOCUSED_ABLATIONS,
        statistics=STATISTICS,
        terminal_gate_requirements=TERMINAL_GATE_REQUIREMENTS,
        adapter_sha256=adapter_sha,
        regulatory_sha256=REGULATORY.sha256,
        spatial_sha256=SPATIAL.sha256,
        statistics_sha256=statistics_sha,
        surface_boundary_sha256=surface_sha,
        protocol_sha256=_namespaced_sha256(_PROTOCOL_NAMESPACE, root_without_protocol),
    )


PROTOCOL = _build_protocol()


__all__ = [
    "ADAPTER_CONTRACT",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "COMMON_ARCHITECTURE_FREEZE_REQUIREMENTS",
    "EVENT_ENVELOPE",
    "FOCUSED_ABLATIONS",
    "LEARNER_SEEDS",
    "PROTOCOL",
    "PROTOCOL_SCOPE",
    "PROTOCOL_STATUS",
    "REGULATORY",
    "REGULATORY_SCHEDULE",
    "REGULATORY_SPECIFICATION",
    "REGULATORY_SPLIT",
    "REGULATORY_THRESHOLDS",
    "SPATIAL",
    "SPATIAL_SCHEDULE",
    "SPATIAL_SPECIFICATION",
    "SPATIAL_SPLIT",
    "SPATIAL_THRESHOLDS",
    "STATISTICS",
    "SURFACE_BOUNDARY_RECORD",
    "TERMINAL_GATE_REQUIREMENTS",
    "CohortSplit",
    "FrozenDomainProtocol",
    "FrozenG0Preregistration",
    "MetricThreshold",
]
