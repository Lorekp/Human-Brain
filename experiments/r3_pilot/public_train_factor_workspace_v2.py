"""Authenticated pre-allocation verdict for PUBLIC_TRAIN_FACTOR_WORKSPACE_V2.

The frozen V2 protocol is stopped before electrical execution or learner-state
allocation because two post-selector physical traces each carry contradictory
single-factor teachers.  Factor/profile/lesson hashes remain audit provenance;
they are deliberately absent from the causal branch-local input key.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256

from experiments.r3_pilot import public_train_factor_workspace_v2_protocol as spec

EXPECTED_PROTOCOL_SHA256 = "54aca3e9f9f7ae0bf1b1743510938522400305c64530db9b3f1467b241ee3171"
STOP_INPUT_ALIAS = "STOP_PUBLIC_TRAIN_FACTOR_WORKSPACE_V2_INPUT_ALIAS"
CAUSAL_INPUT_STAGE = "POST_SELECTOR_BRANCH_LOCAL_SELECTED_TRACE"

ELECTRICAL_CONTROL_ATTEMPT_COUNT = 0
LEARNER_ARRAY_ALLOCATION_COUNT = 0
LEARNING_RUN_COUNT = 0
COUNTERFACTUAL_FIT_LESSON_ATTEMPT_COUNT = 0
CANONICAL_FIT_IDENTITY_ATTEMPT_COUNT = 0
TEACHER_OFF_CONTROL_ATTEMPT_COUNT = 0
CELL_OFF_CONTROL_ATTEMPT_COUNT = 0
FACTOR_DERANGEMENT_CONTROL_ATTEMPT_COUNT = 0
COUNTERFACTUAL_REPLAY_ATTEMPT_COUNT = 0
CANONICAL_FIT_GATE_ATTEMPT_COUNT = 0
CANONICAL_HELDOUT_GATE_ATTEMPT_COUNT = 0
DEPLOYMENT_REBUILD_ATTEMPT_COUNT = 0

DEV_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
DESIRED_OUTPUT_READ_COUNT = 0
PROGRAM_ARTIFACT_READ_COUNT = 0
PROPERTY_ROW_READ_COUNT = 0
FAMILY_IDENTIFIER_READ_COUNT = 0

_FACTOR_INDEX = {factor: index for index, factor in enumerate(spec.FACTOR_REGISTRY)}
_RESULT_NAMESPACE = b"public-train-factor-workspace-v2-input-alias-result\0"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _namespaced_sha256(namespace: bytes, value: object) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _authenticate_protocol(
    protocol: spec.FrozenPublicTrainFactorWorkspaceV2Protocol,
) -> None:
    if not isinstance(protocol, spec.FrozenPublicTrainFactorWorkspaceV2Protocol):
        raise TypeError("preflight requires FrozenPublicTrainFactorWorkspaceV2Protocol")
    rebuilt_protocol = spec._build_protocol()[0]
    if (
        protocol != spec.PROTOCOL
        or rebuilt_protocol != protocol
        or protocol.protocol_sha256 != EXPECTED_PROTOCOL_SHA256
        or protocol.root_record() != spec.PROTOCOL.root_record()
        or protocol.scope != spec.PROTOCOL_SCOPE
        or protocol.status != spec.PROTOCOL_STATUS
    ):
        raise RuntimeError("public-TRAIN factor workspace V2 protocol authentication failed")
    if (
        len(protocol.counterfactual_groups) != spec.COUNTERFACTUAL_GROUP_COUNT
        or sum(len(group.lessons) for group in protocol.counterfactual_groups)
        != spec.COUNTERFACTUAL_LESSON_COUNT
        or len(protocol.canonical_fit_prompt_input_sha256s) != spec.CANONICAL_FIT_IDENTITY_COUNT
        or len(protocol.canonical_heldout_prompt_input_sha256s)
        != spec.CANONICAL_HELDOUT_IDENTITY_COUNT
        or any(
            (
                spec.DEV_INPUT_COUNT,
                spec.PROTECTED_INPUT_COUNT,
                spec.DESIRED_OUTPUT_COUNT,
                spec.PROGRAM_ARTIFACT_COUNT,
                spec.PROPERTY_ROW_COUNT,
                spec.FAMILY_IDENTIFIER_COUNT,
                spec.LEARNER_ARRAY_COUNT,
                spec.LEARNING_RUN_COUNT,
                spec.ELECTRICAL_EXECUTION_COUNT,
            )
        )
    ):
        raise RuntimeError("public-TRAIN factor workspace V2 protocol surface changed")


@dataclass(frozen=True, slots=True)
class BranchLocalPhysicalInputKey:
    """Everything causally visible to the factor race after span selection."""

    selected_assembly_cells: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.selected_assembly_cells) is not tuple
            or len(self.selected_assembly_cells) != spec.ASSEMBLY_ACTIVE_CELL_COUNT
            or len(set(self.selected_assembly_cells)) != spec.ASSEMBLY_ACTIVE_CELL_COUNT
            or tuple(sorted(self.selected_assembly_cells)) != self.selected_assembly_cells
            or any(
                type(cell) is not int or not 0 <= cell < spec.LEXICAL_CELL_COUNT
                for cell in self.selected_assembly_cells
            )
        ):
            raise ValueError("branch-local physical input key changed")

    def record(self) -> dict[str, object]:
        return {
            "causal_stage": CAUSAL_INPUT_STAGE,
            "selected_assembly_cells": self.selected_assembly_cells,
            "selected_active_cell_count": len(self.selected_assembly_cells),
        }


def branch_local_physical_input_key(
    lesson: spec.CounterfactualFactorLesson,
) -> BranchLocalPhysicalInputKey:
    """Erase every label/hash and retain only the post-selector physical trace."""

    if not isinstance(lesson, spec.CounterfactualFactorLesson):
        raise TypeError("physical-key projection requires a frozen V2 lesson")
    return BranchLocalPhysicalInputKey(lesson.target_assembly_cells)


@dataclass(frozen=True, slots=True)
class PhysicalInputAlias:
    physical_input: BranchLocalPhysicalInputKey
    audit_factor_labels: tuple[str, ...]
    audit_lesson_indices: tuple[int, ...]
    audit_lesson_sha256s: tuple[str, ...]
    audit_profile_sha256s: tuple[str, ...]
    audit_splits: tuple[str, ...]

    def record(self) -> dict[str, object]:
        return {
            "physical_input": self.physical_input.record(),
            "audit_only": {
                "factor_labels": self.audit_factor_labels,
                "lesson_indices": self.audit_lesson_indices,
                "lesson_sha256s": self.audit_lesson_sha256s,
                "profile_sha256s": self.audit_profile_sha256s,
                "splits": self.audit_splits,
            },
        }


@dataclass(frozen=True, slots=True)
class PublicTrainFactorWorkspaceV2Stop:
    status: str
    protocol_sha256: str
    protocol_authentication_count: int
    causal_input_stage: str
    counterfactual_lesson_scan_count: int
    ambiguous_physical_input_count: int
    ambiguous_lesson_count: int
    ambiguous_fit_lesson_count: int
    ambiguous_heldout_lesson_count: int
    aliases: tuple[PhysicalInputAlias, ...]
    electrical_control_attempt_count: int
    learner_array_allocation_count: int
    learning_run_count: int
    counterfactual_fit_lesson_attempt_count: int
    canonical_fit_identity_attempt_count: int
    teacher_off_control_attempt_count: int
    cell_off_control_attempt_count: int
    factor_derangement_control_attempt_count: int
    counterfactual_replay_attempt_count: int
    canonical_fit_gate_attempt_count: int
    canonical_heldout_gate_attempt_count: int
    deployment_rebuild_attempt_count: int
    dev_input_count: int
    protected_input_count: int
    desired_output_read_count: int
    program_artifact_read_count: int
    property_row_read_count: int
    family_identifier_read_count: int
    sha256: str

    @property
    def downstream_attempt_count(self) -> int:
        return sum(
            (
                self.electrical_control_attempt_count,
                self.learner_array_allocation_count,
                self.learning_run_count,
                self.counterfactual_fit_lesson_attempt_count,
                self.canonical_fit_identity_attempt_count,
                self.teacher_off_control_attempt_count,
                self.cell_off_control_attempt_count,
                self.factor_derangement_control_attempt_count,
                self.counterfactual_replay_attempt_count,
                self.canonical_fit_gate_attempt_count,
                self.canonical_heldout_gate_attempt_count,
                self.deployment_rebuild_attempt_count,
            )
        )

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "protocol_sha256": self.protocol_sha256,
            "protocol_authentication_count": self.protocol_authentication_count,
            "causal_input_stage": self.causal_input_stage,
            "counterfactual_lesson_scan_count": self.counterfactual_lesson_scan_count,
            "ambiguous_physical_input_count": self.ambiguous_physical_input_count,
            "ambiguous_lesson_count": self.ambiguous_lesson_count,
            "ambiguous_fit_lesson_count": self.ambiguous_fit_lesson_count,
            "ambiguous_heldout_lesson_count": self.ambiguous_heldout_lesson_count,
            "aliases": tuple(alias.record() for alias in self.aliases),
            "electrical_control_attempt_count": self.electrical_control_attempt_count,
            "learner_array_allocation_count": self.learner_array_allocation_count,
            "learning_run_count": self.learning_run_count,
            "counterfactual_fit_lesson_attempt_count": (
                self.counterfactual_fit_lesson_attempt_count
            ),
            "canonical_fit_identity_attempt_count": (self.canonical_fit_identity_attempt_count),
            "teacher_off_control_attempt_count": self.teacher_off_control_attempt_count,
            "cell_off_control_attempt_count": self.cell_off_control_attempt_count,
            "factor_derangement_control_attempt_count": (
                self.factor_derangement_control_attempt_count
            ),
            "counterfactual_replay_attempt_count": (self.counterfactual_replay_attempt_count),
            "canonical_fit_gate_attempt_count": self.canonical_fit_gate_attempt_count,
            "canonical_heldout_gate_attempt_count": (self.canonical_heldout_gate_attempt_count),
            "deployment_rebuild_attempt_count": self.deployment_rebuild_attempt_count,
            "downstream_attempt_count": self.downstream_attempt_count,
            "dev_input_count": self.dev_input_count,
            "protected_input_count": self.protected_input_count,
            "desired_output_read_count": self.desired_output_read_count,
            "program_artifact_read_count": self.program_artifact_read_count,
            "property_row_read_count": self.property_row_read_count,
            "family_identifier_read_count": self.family_identifier_read_count,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


def _scan_physical_input_aliases(
    protocol: spec.FrozenPublicTrainFactorWorkspaceV2Protocol,
) -> tuple[tuple[PhysicalInputAlias, ...], int, int]:
    profiles_by_sha256 = {profile.sha256: profile for profile in protocol.span_feature_profiles}
    lessons = tuple(lesson for group in protocol.counterfactual_groups for lesson in group.lessons)
    grouped: dict[
        BranchLocalPhysicalInputKey,
        list[spec.CounterfactualFactorLesson],
    ] = defaultdict(list)
    for lesson in lessons:
        profile = profiles_by_sha256.get(lesson.target_span_profile_sha256)
        if (
            profile is None
            or profile.factor != lesson.factor
            or profile.span_identity_sha256 != lesson.target_span_identity_sha256
            or profile.assembly_cells != lesson.target_assembly_cells
        ):
            raise RuntimeError("counterfactual lesson physical profile authentication failed")
        grouped[branch_local_physical_input_key(lesson)].append(lesson)

    aliases: list[PhysicalInputAlias] = []
    for physical_input, rows in grouped.items():
        factors = tuple(sorted({row.factor for row in rows}, key=_FACTOR_INDEX.__getitem__))
        if len(factors) < 2:
            continue
        ordered_rows = tuple(sorted(rows, key=lambda row: row.lesson_index))
        aliases.append(
            PhysicalInputAlias(
                physical_input=physical_input,
                audit_factor_labels=factors,
                audit_lesson_indices=tuple(row.lesson_index for row in ordered_rows),
                audit_lesson_sha256s=tuple(row.sha256 for row in ordered_rows),
                audit_profile_sha256s=tuple(row.target_span_profile_sha256 for row in ordered_rows),
                audit_splits=tuple(row.split for row in ordered_rows),
            )
        )
    frozen_aliases = tuple(
        sorted(aliases, key=lambda alias: alias.physical_input.selected_assembly_cells)
    )
    ambiguous_fit = sum(
        split == spec.FIT for alias in frozen_aliases for split in alias.audit_splits
    )
    ambiguous_heldout = sum(
        split == spec.HELDOUT for alias in frozen_aliases for split in alias.audit_splits
    )
    return frozen_aliases, ambiguous_fit, ambiguous_heldout


def run_public_train_factor_workspace_v2_preflight(
    protocol: spec.FrozenPublicTrainFactorWorkspaceV2Protocol = spec.PROTOCOL,
) -> PublicTrainFactorWorkspaceV2Stop:
    """Authenticate V2 and stop before controls/state on a causal input alias."""

    _authenticate_protocol(protocol)
    aliases, ambiguous_fit, ambiguous_heldout = _scan_physical_input_aliases(protocol)
    expected_lesson_index_sets = {
        (1, 2, 30, 39, 59),
        (22, 31, 51, 67, 68),
    }
    if (
        len(aliases) != 2
        or {alias.audit_lesson_indices for alias in aliases} != expected_lesson_index_sets
        or any(alias.audit_factor_labels != ("minimum", "maximum") for alias in aliases)
        or ambiguous_fit != 7
        or ambiguous_heldout != 3
    ):
        raise RuntimeError("frozen V2 causal input-alias verdict changed")
    _authenticate_protocol(protocol)

    root_without_sha = {
        "status": STOP_INPUT_ALIAS,
        "protocol_sha256": protocol.protocol_sha256,
        "protocol_authentication_count": 2,
        "causal_input_stage": CAUSAL_INPUT_STAGE,
        "counterfactual_lesson_scan_count": spec.COUNTERFACTUAL_LESSON_COUNT,
        "ambiguous_physical_input_count": len(aliases),
        "ambiguous_lesson_count": ambiguous_fit + ambiguous_heldout,
        "ambiguous_fit_lesson_count": ambiguous_fit,
        "ambiguous_heldout_lesson_count": ambiguous_heldout,
        "aliases": tuple(alias.record() for alias in aliases),
        "electrical_control_attempt_count": ELECTRICAL_CONTROL_ATTEMPT_COUNT,
        "learner_array_allocation_count": LEARNER_ARRAY_ALLOCATION_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "counterfactual_fit_lesson_attempt_count": (COUNTERFACTUAL_FIT_LESSON_ATTEMPT_COUNT),
        "canonical_fit_identity_attempt_count": CANONICAL_FIT_IDENTITY_ATTEMPT_COUNT,
        "teacher_off_control_attempt_count": TEACHER_OFF_CONTROL_ATTEMPT_COUNT,
        "cell_off_control_attempt_count": CELL_OFF_CONTROL_ATTEMPT_COUNT,
        "factor_derangement_control_attempt_count": (FACTOR_DERANGEMENT_CONTROL_ATTEMPT_COUNT),
        "counterfactual_replay_attempt_count": COUNTERFACTUAL_REPLAY_ATTEMPT_COUNT,
        "canonical_fit_gate_attempt_count": CANONICAL_FIT_GATE_ATTEMPT_COUNT,
        "canonical_heldout_gate_attempt_count": CANONICAL_HELDOUT_GATE_ATTEMPT_COUNT,
        "deployment_rebuild_attempt_count": DEPLOYMENT_REBUILD_ATTEMPT_COUNT,
        "downstream_attempt_count": 0,
        "dev_input_count": DEV_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "desired_output_read_count": DESIRED_OUTPUT_READ_COUNT,
        "program_artifact_read_count": PROGRAM_ARTIFACT_READ_COUNT,
        "property_row_read_count": PROPERTY_ROW_READ_COUNT,
        "family_identifier_read_count": FAMILY_IDENTIFIER_READ_COUNT,
    }
    return PublicTrainFactorWorkspaceV2Stop(
        status=STOP_INPUT_ALIAS,
        protocol_sha256=protocol.protocol_sha256,
        protocol_authentication_count=2,
        causal_input_stage=CAUSAL_INPUT_STAGE,
        counterfactual_lesson_scan_count=spec.COUNTERFACTUAL_LESSON_COUNT,
        ambiguous_physical_input_count=len(aliases),
        ambiguous_lesson_count=ambiguous_fit + ambiguous_heldout,
        ambiguous_fit_lesson_count=ambiguous_fit,
        ambiguous_heldout_lesson_count=ambiguous_heldout,
        aliases=aliases,
        electrical_control_attempt_count=ELECTRICAL_CONTROL_ATTEMPT_COUNT,
        learner_array_allocation_count=LEARNER_ARRAY_ALLOCATION_COUNT,
        learning_run_count=LEARNING_RUN_COUNT,
        counterfactual_fit_lesson_attempt_count=(COUNTERFACTUAL_FIT_LESSON_ATTEMPT_COUNT),
        canonical_fit_identity_attempt_count=CANONICAL_FIT_IDENTITY_ATTEMPT_COUNT,
        teacher_off_control_attempt_count=TEACHER_OFF_CONTROL_ATTEMPT_COUNT,
        cell_off_control_attempt_count=CELL_OFF_CONTROL_ATTEMPT_COUNT,
        factor_derangement_control_attempt_count=(FACTOR_DERANGEMENT_CONTROL_ATTEMPT_COUNT),
        counterfactual_replay_attempt_count=COUNTERFACTUAL_REPLAY_ATTEMPT_COUNT,
        canonical_fit_gate_attempt_count=CANONICAL_FIT_GATE_ATTEMPT_COUNT,
        canonical_heldout_gate_attempt_count=CANONICAL_HELDOUT_GATE_ATTEMPT_COUNT,
        deployment_rebuild_attempt_count=DEPLOYMENT_REBUILD_ATTEMPT_COUNT,
        dev_input_count=DEV_INPUT_COUNT,
        protected_input_count=PROTECTED_INPUT_COUNT,
        desired_output_read_count=DESIRED_OUTPUT_READ_COUNT,
        program_artifact_read_count=PROGRAM_ARTIFACT_READ_COUNT,
        property_row_read_count=PROPERTY_ROW_READ_COUNT,
        family_identifier_read_count=FAMILY_IDENTIFIER_READ_COUNT,
        sha256=_namespaced_sha256(_RESULT_NAMESPACE, root_without_sha),
    )


RESULT = run_public_train_factor_workspace_v2_preflight()


__all__ = [
    "CAUSAL_INPUT_STAGE",
    "EXPECTED_PROTOCOL_SHA256",
    "RESULT",
    "STOP_INPUT_ALIAS",
    "BranchLocalPhysicalInputKey",
    "PhysicalInputAlias",
    "PublicTrainFactorWorkspaceV2Stop",
    "branch_local_physical_input_key",
    "run_public_train_factor_workspace_v2_preflight",
]
