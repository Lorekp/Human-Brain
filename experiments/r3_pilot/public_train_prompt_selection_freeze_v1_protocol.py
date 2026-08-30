"""Data-only public-TRAIN prompt-selection freeze protocol V1.

The protocol authenticates the public prompt/signature capsule, binds every
role-normalized :class:`PromptSpanInput` identity to its complete
``TypedSignature`` record, and then projects a causal key containing only
physical span, typed-literal, and name-free signature-role events.

The banked semantic annotation surface contains span-attached factors only.  It
does not contain the exact structural relay targets required to select a packed
root/e-class.  This module therefore freezes the relay vocabulary and a precise
prelearning STOP.  It deliberately constructs no learner, execution, candidate,
proof, desired output, program artifact, property row, task/family identity, or
DEV/protected input.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

from experiments.r3_pilot import public_train_factor_workspace_v2_protocol as factor_v2
from experiments.r3_pilot import public_train_prompt_factor_manifest_v1 as manifest_v1
from experiments.r3_pilot import public_train_prompt_semantic_bridge as prompt_spans
from humanbrain.typed_program_types import TypedSignature

PROTOCOL_SCOPE = "PUBLIC_TRAIN_PROMPT_SELECTION_FREEZE_V1_PROTOCOL"
STOP_EXACT_STRUCTURAL_RELAY_TARGETS_UNAVAILABLE = (
    "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_FREEZE_V1_EXACT_STRUCTURAL_RELAY_TARGETS_UNAVAILABLE"
)

PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT = 90
CANONICAL_SEMANTIC_IDENTITY_COUNT = 42
CANONICAL_FIT_IDENTITY_COUNT = 24
CANONICAL_HELDOUT_IDENTITY_COUNT = 18

DEV_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
DESIRED_OUTPUT_READ_COUNT = 0
PROGRAM_ARTIFACT_READ_COUNT = 0
PROPERTY_ROW_READ_COUNT = 0
TASK_IDENTIFIER_READ_COUNT = 0
FAMILY_IDENTIFIER_READ_COUNT = 0
ENTRYPOINT_SPELLING_NEURAL_EVIDENCE_COUNT = 0
AUDIT_HASH_NEURAL_EVIDENCE_COUNT = 0
LEARNER_ARRAY_COUNT = 0
LEARNING_RUN_COUNT = 0
ELECTRICAL_EXECUTION_COUNT = 0
CANDIDATE_COUNT = 0
PROOF_COUNT = 0

EXPECTED_FACTOR_WORKSPACE_PROTOCOL_SHA256 = (
    "54aca3e9f9f7ae0bf1b1743510938522400305c64530db9b3f1467b241ee3171"
)

_FULL_SIGNATURE_NAMESPACE = b"public-train-prompt-selection-v1-full-signature\0"
_SIGNATURE_ROLE_NAMESPACE = b"public-train-prompt-selection-v1-signature-role\0"
_SPAN_EVENTS_NAMESPACE = b"public-train-prompt-selection-v1-span-events\0"
_LITERAL_EVENTS_NAMESPACE = b"public-train-prompt-selection-v1-literal-events\0"
_CAUSAL_KEY_NAMESPACE = b"public-train-prompt-selection-v1-causal-key\0"
_SIGNATURE_BINDING_NAMESPACE = b"public-train-prompt-selection-v1-signature-binding\0"
_FACTOR_ANNOTATION_NAMESPACE = b"public-train-prompt-selection-v1-factor-annotation\0"
_CANONICAL_INPUT_NAMESPACE = b"public-train-prompt-selection-v1-canonical-input\0"
_ORDERED_REGISTRY_NAMESPACE = b"public-train-prompt-selection-v1-registry\0"
_RELAY_VOCABULARY_NAMESPACE = b"public-train-prompt-selection-v1-relay-vocabulary\0"
_ANNOTATION_SCHEMA_NAMESPACE = b"public-train-prompt-selection-v1-annotation-schema\0"
_IDENTIFIABILITY_NAMESPACE = b"public-train-prompt-selection-v1-identifiability\0"
_DEPENDENCY_NAMESPACE = b"public-train-prompt-selection-v1-dependency\0"
_SPLIT_NAMESPACE = b"public-train-prompt-selection-v1-split\0"
_SURFACE_NAMESPACE = b"public-train-prompt-selection-v1-surface\0"
_PROTOCOL_NAMESPACE = b"public-train-prompt-selection-v1-protocol\0"


def _plain_record(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain_record(item) for item in value)
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain_record(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _namespaced_sha256(namespace: bytes, value: object) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _freeze_record(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_record(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_record(item) for item in value)
    return value


def _strict_record(value: object, keys: frozenset[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError(f"{label} must contain exactly {sorted(keys)}")
    return value


@dataclass(frozen=True, slots=True)
class PhysicalCausalInputKey:
    """Content address of only events that may drive a future learner."""

    physical_span_events_sha256: str
    typed_literal_events_sha256: str
    signature_role_events_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "physical_span_events_sha256": self.physical_span_events_sha256,
            "typed_literal_events_sha256": self.typed_literal_events_sha256,
            "signature_role_events_sha256": self.signature_role_events_sha256,
            "causal_sources": (
                "ordered-physical-span-events",
                "ordered-typed-literal-cell-events",
                "ordered-name-free-signature-role-events",
            ),
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class SignatureBoundPromptAppearance:
    """One full signature bound to a role-normalized prompt input identity."""

    prompt_input_sha256: str
    typed_signature_record: object
    typed_signature_sha256: str
    signature_role_record: object
    signature_role_sha256: str
    causal_key: PhysicalCausalInputKey
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "prompt_input_sha256": self.prompt_input_sha256,
            "typed_signature_record": self.typed_signature_record,
            "typed_signature_sha256": self.typed_signature_sha256,
            "signature_role_record": self.signature_role_record,
            "signature_role_sha256": self.signature_role_sha256,
            "causal_key": self.causal_key.record(),
            "full_signature_is_authentication_only": True,
            "entrypoint_and_parameter_spelling_are_not_causal": True,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class SpanFactorAnnotation:
    start_token: int
    stop_token: int
    span_identity_sha256: str
    factor: str

    def record(self) -> dict[str, object]:
        return {
            "start_token": self.start_token,
            "stop_token": self.stop_token,
            "span_identity_sha256": self.span_identity_sha256,
            "factor": self.factor,
        }


@dataclass(frozen=True, slots=True)
class FrozenFactorAnnotation:
    prompt_input_sha256: str
    targets: tuple[SpanFactorAnnotation, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "prompt_input_sha256": self.prompt_input_sha256,
            "targets": tuple(target.record() for target in self.targets),
            "annotation_semantics": "span-attached-factors-only",
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class CanonicalPromptSelectionInput:
    prompt_input_sha256: str
    appearance_count: int
    split: str
    causal_key_sha256: str
    signature_role_sha256: str
    factor_annotation_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "prompt_input_sha256": self.prompt_input_sha256,
            "appearance_count": self.appearance_count,
            "split": self.split,
            "causal_key_sha256": self.causal_key_sha256,
            "signature_role_sha256": self.signature_role_sha256,
            "factor_annotation_sha256": self.factor_annotation_sha256,
            "appearance_multiplicity_is_audit_only": True,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class RelayTargetChannel:
    channel: str
    required_fields: tuple[str, ...]
    purpose: str

    def record(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "required_fields": self.required_fields,
            "purpose": self.purpose,
        }


@dataclass(frozen=True, slots=True)
class PrelearningIdentifiabilityAudit:
    status: str
    prompt_appearance_count: int
    canonical_causal_identity_count: int
    source_factor_annotation_set_count: int
    source_factor_annotation_alias_count: int
    required_exact_relay_target_set_count: int
    available_exact_relay_target_set_count: int
    missing_relay_channels: tuple[str, ...]
    causal_key_relay_target_collision_audit_completed: bool
    reason: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "prompt_appearance_count": self.prompt_appearance_count,
            "canonical_causal_identity_count": self.canonical_causal_identity_count,
            "source_factor_annotation_set_count": self.source_factor_annotation_set_count,
            "source_factor_annotation_alias_count": self.source_factor_annotation_alias_count,
            "required_exact_relay_target_set_count": (self.required_exact_relay_target_set_count),
            "available_exact_relay_target_set_count": (self.available_exact_relay_target_set_count),
            "missing_relay_channels": self.missing_relay_channels,
            "causal_key_relay_target_collision_audit_completed": (
                self.causal_key_relay_target_collision_audit_completed
            ),
            "identical_causal_key_rule": (
                "every identical physical causal key must demand one exact identical "
                "relay target set"
            ),
            "reason": self.reason,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class FrozenPublicTrainPromptSelectionFreezeV1Protocol:
    scope: str
    status: str
    signature_bindings: tuple[SignatureBoundPromptAppearance, ...]
    canonical_inputs: tuple[CanonicalPromptSelectionInput, ...]
    factor_annotations: tuple[FrozenFactorAnnotation, ...]
    relay_target_vocabulary: tuple[RelayTargetChannel, ...]
    identifiability_audit: PrelearningIdentifiabilityAudit
    dependency_sha256: str
    signature_bindings_sha256: str
    canonical_inputs_sha256: str
    factor_annotations_sha256: str
    relay_target_vocabulary_sha256: str
    annotation_schema_sha256: str
    identifiability_audit_sha256: str
    canonical_split_sha256: str
    surface_boundary_sha256: str
    protocol_sha256: str

    def root_record(self, *, include_protocol_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "scope": self.scope,
            "status": self.status,
            "dependency_sha256": self.dependency_sha256,
            "signature_bindings_sha256": self.signature_bindings_sha256,
            "canonical_inputs_sha256": self.canonical_inputs_sha256,
            "factor_annotations_sha256": self.factor_annotations_sha256,
            "relay_target_vocabulary_sha256": self.relay_target_vocabulary_sha256,
            "annotation_schema_sha256": self.annotation_schema_sha256,
            "identifiability_audit_sha256": self.identifiability_audit_sha256,
            "canonical_split_sha256": self.canonical_split_sha256,
            "surface_boundary_sha256": self.surface_boundary_sha256,
            "public_train_prompt_appearance_count": len(self.signature_bindings),
            "canonical_semantic_identity_count": len(self.canonical_inputs),
            "canonical_fit_identity_count": sum(
                row.split == factor_v2.FIT for row in self.canonical_inputs
            ),
            "canonical_heldout_identity_count": sum(
                row.split == factor_v2.HELDOUT for row in self.canonical_inputs
            ),
            "exact_relay_target_set_count": 0,
            "learner_array_count": LEARNER_ARRAY_COUNT,
            "learning_run_count": LEARNING_RUN_COUNT,
            "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
            "candidate_count": CANDIDATE_COUNT,
            "proof_count": PROOF_COUNT,
            "learner_state_constructed": False,
            "learning_executed": False,
            "selection_executed": False,
        }
        if include_protocol_sha256:
            record["protocol_sha256"] = self.protocol_sha256
        return record


RELAY_TARGET_VOCABULARY = (
    RelayTargetChannel(
        "root-production-family",
        ("root_production", "root_family", "input_type", "output_type"),
        "select the typed packed root production and semantic family",
    ),
    RelayTargetChannel(
        "ordered-parent-child-edge",
        ("parent_slot", "child_ordinal", "child_slot"),
        "select each ordered packed production edge",
    ),
    RelayTargetChannel(
        "argument-operand-orientation",
        ("operator_slot", "operand_ordinal", "source_role"),
        "bind each non-commutative or role-sensitive operand",
    ),
    RelayTargetChannel(
        "nesting-relation",
        ("outer_slot", "inner_slot", "nesting_depth"),
        "bind compositional outer/inner structure",
    ),
    RelayTargetChannel(
        "parameter-role-binding",
        ("signature_parameter_role", "semantic_argument_slot"),
        "bind name-free typed parameters to semantic argument slots",
    ),
    RelayTargetChannel(
        "literal-slot-binding",
        ("literal_source_ordinal", "semantic_literal_slot"),
        "route each prompt literal occurrence to one semantic literal slot",
    ),
    RelayTargetChannel(
        "literal-value-copy",
        ("semantic_literal_slot", "typed_output_copy_role"),
        "copy the selected signed value through the physical typed-output bank",
    ),
    RelayTargetChannel(
        "terminal-root-eclass-agreement",
        ("root_handle_role", "eclass_handle_role", "exact_identity_required"),
        "require the prompt winner and authoritative packed root/e-class to agree",
    ),
)


ANNOTATION_SCHEMA_RECORD = _freeze_record(
    {
        "available_frozen_semantic_annotation_fields": (
            "start_token",
            "stop_token",
            "span_identity_sha256",
            "factor",
        ),
        "available_semantics": "span-attached-factor-membership",
        "unavailable_exact_relay_channels": tuple(
            channel.channel for channel in RELAY_TARGET_VOCABULARY
        ),
        "prompt_or_factor_to_structure_inference_forbidden": True,
        "program_artifact_recovery_forbidden": True,
        "invented_structural_labels_forbidden": True,
    }
)
ANNOTATION_SCHEMA_SHA256 = _namespaced_sha256(
    _ANNOTATION_SCHEMA_NAMESPACE,
    dict(ANNOTATION_SCHEMA_RECORD),
)


def _signature_role_record(signature: TypedSignature) -> dict[str, object]:
    parameter_count = len(signature.parameters)
    return {
        "parameters": tuple(
            {
                "role": f"ARG{parameter_count - index - 1}",
                "type": parameter.type.to_record(),
            }
            for index, parameter in enumerate(signature.parameters)
        ),
        "result": signature.result.to_record(),
        "role_binding": "reverse-declaration-depth",
        "entrypoint_spelling_present": False,
        "parameter_spelling_present": False,
    }


def _physical_span_event_record(
    prompt_input: prompt_spans.PromptSpanInput,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "start_token": span.start_token,
            "stop_token": span.stop_token,
            "assembly_cells": span.assembly_cells,
            "first_spike_step": span.first_spike_step,
            "spike_cells": span.spike_cells,
        }
        for span in prompt_input.spans
    )


def _typed_literal_event_record(
    prompt_input: prompt_spans.PromptSpanInput,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "canonical_token_index": literal.canonical_token_index,
            "typed_output_active_cells": literal.typed_output_active_cells,
        }
        for literal in prompt_input.signed_literals
    )


def _binding(
    prompt_input: prompt_spans.PromptSpanInput,
    signature: TypedSignature,
) -> SignatureBoundPromptAppearance:
    full_signature_record = signature.to_record()
    role_record = _signature_role_record(signature)
    full_signature_sha = _namespaced_sha256(
        _FULL_SIGNATURE_NAMESPACE,
        full_signature_record,
    )
    role_sha = _namespaced_sha256(_SIGNATURE_ROLE_NAMESPACE, role_record)
    span_events_sha = _namespaced_sha256(
        _SPAN_EVENTS_NAMESPACE,
        _physical_span_event_record(prompt_input),
    )
    literal_events_sha = _namespaced_sha256(
        _LITERAL_EVENTS_NAMESPACE,
        _typed_literal_event_record(prompt_input),
    )
    causal_without_sha = {
        "physical_span_events_sha256": span_events_sha,
        "typed_literal_events_sha256": literal_events_sha,
        "signature_role_events_sha256": role_sha,
        "causal_sources": (
            "ordered-physical-span-events",
            "ordered-typed-literal-cell-events",
            "ordered-name-free-signature-role-events",
        ),
    }
    causal_key = PhysicalCausalInputKey(
        physical_span_events_sha256=span_events_sha,
        typed_literal_events_sha256=literal_events_sha,
        signature_role_events_sha256=role_sha,
        sha256=_namespaced_sha256(_CAUSAL_KEY_NAMESPACE, causal_without_sha),
    )
    binding_without_sha = {
        "prompt_input_sha256": prompt_input.sha256,
        "typed_signature_record": full_signature_record,
        "typed_signature_sha256": full_signature_sha,
        "signature_role_record": role_record,
        "signature_role_sha256": role_sha,
        "causal_key": causal_key.record(),
        "full_signature_is_authentication_only": True,
        "entrypoint_and_parameter_spelling_are_not_causal": True,
    }
    return SignatureBoundPromptAppearance(
        prompt_input_sha256=prompt_input.sha256,
        typed_signature_record=_freeze_record(full_signature_record),
        typed_signature_sha256=full_signature_sha,
        signature_role_record=_freeze_record(role_record),
        signature_role_sha256=role_sha,
        causal_key=causal_key,
        sha256=_namespaced_sha256(_SIGNATURE_BINDING_NAMESPACE, binding_without_sha),
    )


def _load_observed_appearances() -> tuple[
    tuple[SignatureBoundPromptAppearance, ...],
    tuple[prompt_spans.PromptSpanInput, ...],
]:
    raw_bytes = manifest_v1.CAPSULE_PATH.read_bytes()
    if sha256(raw_bytes).hexdigest() != manifest_v1.CAPSULE_RAW_SHA256:
        raise RuntimeError("public-TRAIN prompt/signature capsule raw identity changed")
    try:
        capsule_value = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise RuntimeError("public-TRAIN prompt/signature capsule is not JSON") from error
    if sha256(_canonical_json(capsule_value)).hexdigest() != manifest_v1.CAPSULE_PAYLOAD_SHA256:
        raise RuntimeError("public-TRAIN prompt/signature capsule payload identity changed")
    capsule = _strict_record(
        capsule_value,
        frozenset({"schema", "scope", "records"}),
        "public-TRAIN prompt/signature capsule",
    )
    if (
        capsule["schema"] != manifest_v1.CAPSULE_SCHEMA
        or capsule["scope"] != manifest_v1.CAPSULE_SCOPE
    ):
        raise RuntimeError("public-TRAIN prompt/signature capsule schema changed")
    records = capsule["records"]
    if type(records) is not list or len(records) != PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT:
        raise RuntimeError("public-TRAIN prompt/signature appearance count changed")

    bindings: list[SignatureBoundPromptAppearance] = []
    observed: list[prompt_spans.PromptSpanInput] = []
    for index, value in enumerate(records):
        record = _strict_record(
            value,
            frozenset({"prompt", "signature"}),
            f"public-TRAIN prompt/signature records[{index}]",
        )
        prompt = record["prompt"]
        if type(prompt) is not str or not prompt:
            raise RuntimeError("public-TRAIN prompt text is invalid")
        signature = TypedSignature.from_record(record["signature"])
        prompt_input = prompt_spans.observe_public_train_prompt(prompt, signature)
        bindings.append(_binding(prompt_input, signature))
        observed.append(prompt_input)
    return tuple(bindings), tuple(observed)


def _factor_annotations(
    prompt_inputs: tuple[prompt_spans.PromptSpanInput, ...],
) -> tuple[FrozenFactorAnnotation, ...]:
    manifest = manifest_v1.parse_lesson_manifest(
        manifest_v1.build_lesson_manifest_record(prompt_inputs)
    )
    manifest_v1.authenticate_lesson_targets(manifest, prompt_inputs)
    if manifest.sha256 != manifest_v1.EXPECTED_LESSON_MANIFEST_SHA256:
        raise RuntimeError("public-TRAIN frozen semantic annotation root changed")

    grouped: dict[str, tuple[SpanFactorAnnotation, ...]] = {}
    for lesson in manifest.lessons:
        targets = tuple(
            SpanFactorAnnotation(
                start_token=target.start_token,
                stop_token=target.stop_token,
                span_identity_sha256=target.span_identity_sha256,
                factor=target.factor,
            )
            for target in lesson.targets
        )
        previous = grouped.setdefault(lesson.prompt_input_sha256, targets)
        if previous != targets:
            raise RuntimeError("identical prompt input has contradictory factor annotations")

    rows: list[FrozenFactorAnnotation] = []
    for frozen_identity in factor_v2.PROTOCOL.canonical_identities:
        targets = grouped.get(frozen_identity.prompt_input_sha256)
        if targets is None:
            raise RuntimeError("canonical prompt identity lacks frozen semantic annotation")
        record = {
            "prompt_input_sha256": frozen_identity.prompt_input_sha256,
            "targets": tuple(target.record() for target in targets),
            "annotation_semantics": "span-attached-factors-only",
        }
        rows.append(
            FrozenFactorAnnotation(
                prompt_input_sha256=frozen_identity.prompt_input_sha256,
                targets=targets,
                sha256=_namespaced_sha256(_FACTOR_ANNOTATION_NAMESPACE, record),
            )
        )
    return tuple(rows)


def _build_protocol() -> FrozenPublicTrainPromptSelectionFreezeV1Protocol:
    if factor_v2.PROTOCOL.protocol_sha256 != EXPECTED_FACTOR_WORKSPACE_PROTOCOL_SHA256:
        raise RuntimeError("banked public-TRAIN factor-workspace protocol root changed")

    bindings, prompt_inputs = _load_observed_appearances()
    annotations = _factor_annotations(prompt_inputs)
    annotation_by_identity = {row.prompt_input_sha256: row for row in annotations}
    bindings_by_identity: dict[str, list[SignatureBoundPromptAppearance]] = defaultdict(list)
    for binding in bindings:
        bindings_by_identity[binding.prompt_input_sha256].append(binding)

    fit_set = set(factor_v2.PROTOCOL.canonical_fit_prompt_input_sha256s)
    heldout_set = set(factor_v2.PROTOCOL.canonical_heldout_prompt_input_sha256s)
    canonical_inputs: list[CanonicalPromptSelectionInput] = []
    for frozen_identity in factor_v2.PROTOCOL.canonical_identities:
        identity = frozen_identity.prompt_input_sha256
        group = bindings_by_identity.get(identity, [])
        if len(group) != frozen_identity.appearance_count:
            raise RuntimeError("canonical prompt identity appearance multiplicity changed")
        causal_keys = {binding.causal_key.sha256 for binding in group}
        role_roots = {binding.signature_role_sha256 for binding in group}
        if len(causal_keys) != 1 or len(role_roots) != 1:
            raise RuntimeError("role-normalized prompt identity has divergent causal events")
        annotation = annotation_by_identity[identity]
        if tuple(target.factor for target in annotation.targets) != frozen_identity.target_factors:
            raise RuntimeError("factor-workspace and semantic annotation factors differ")
        split = factor_v2.FIT if identity in fit_set else factor_v2.HELDOUT
        if (identity in fit_set) == (identity in heldout_set):
            raise RuntimeError("canonical prompt identity split is not exact")
        record = {
            "prompt_input_sha256": identity,
            "appearance_count": len(group),
            "split": split,
            "causal_key_sha256": next(iter(causal_keys)),
            "signature_role_sha256": next(iter(role_roots)),
            "factor_annotation_sha256": annotation.sha256,
            "appearance_multiplicity_is_audit_only": True,
        }
        canonical_inputs.append(
            CanonicalPromptSelectionInput(
                prompt_input_sha256=identity,
                appearance_count=len(group),
                split=split,
                causal_key_sha256=record["causal_key_sha256"],
                signature_role_sha256=record["signature_role_sha256"],
                factor_annotation_sha256=annotation.sha256,
                sha256=_namespaced_sha256(_CANONICAL_INPUT_NAMESPACE, record),
            )
        )
    canonical = tuple(canonical_inputs)
    if (
        len(canonical) != CANONICAL_SEMANTIC_IDENTITY_COUNT
        or len({row.prompt_input_sha256 for row in canonical}) != len(canonical)
        or len({row.causal_key_sha256 for row in canonical}) != len(canonical)
        or sum(row.appearance_count for row in canonical) != PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT
        or sum(row.split == factor_v2.FIT for row in canonical) != CANONICAL_FIT_IDENTITY_COUNT
        or sum(row.split == factor_v2.HELDOUT for row in canonical)
        != CANONICAL_HELDOUT_IDENTITY_COUNT
    ):
        raise RuntimeError("canonical signature-bound causal inventory changed")

    factor_roots_by_causal_key: dict[str, set[str]] = defaultdict(set)
    for row in canonical:
        factor_roots_by_causal_key[row.causal_key_sha256].add(row.factor_annotation_sha256)
    source_alias_count = sum(len(roots) > 1 for roots in factor_roots_by_causal_key.values())
    if source_alias_count:
        raise RuntimeError("identical causal keys demand different frozen factor annotations")

    missing_channels = tuple(channel.channel for channel in RELAY_TARGET_VOCABULARY)
    audit_without_sha = {
        "status": STOP_EXACT_STRUCTURAL_RELAY_TARGETS_UNAVAILABLE,
        "prompt_appearance_count": len(bindings),
        "canonical_causal_identity_count": len(canonical),
        "source_factor_annotation_set_count": len(annotations),
        "source_factor_annotation_alias_count": source_alias_count,
        "required_exact_relay_target_set_count": len(canonical),
        "available_exact_relay_target_set_count": 0,
        "missing_relay_channels": missing_channels,
        "causal_key_relay_target_collision_audit_completed": False,
        "identical_causal_key_rule": (
            "every identical physical causal key must demand one exact identical relay target set"
        ),
        "reason": (
            "the frozen semantic annotation schema contains only span-attached factors; "
            "exact structural relay targets cannot be recovered without inventing labels "
            "or reading forbidden program artifacts"
        ),
    }
    audit = PrelearningIdentifiabilityAudit(
        status=STOP_EXACT_STRUCTURAL_RELAY_TARGETS_UNAVAILABLE,
        prompt_appearance_count=len(bindings),
        canonical_causal_identity_count=len(canonical),
        source_factor_annotation_set_count=len(annotations),
        source_factor_annotation_alias_count=source_alias_count,
        required_exact_relay_target_set_count=len(canonical),
        available_exact_relay_target_set_count=0,
        missing_relay_channels=missing_channels,
        causal_key_relay_target_collision_audit_completed=False,
        reason=audit_without_sha["reason"],
        sha256=_namespaced_sha256(_IDENTIFIABILITY_NAMESPACE, audit_without_sha),
    )

    bindings_sha = _namespaced_sha256(
        _ORDERED_REGISTRY_NAMESPACE + b"signature-bindings\0",
        tuple(binding.record() for binding in bindings),
    )
    canonical_sha = _namespaced_sha256(
        _ORDERED_REGISTRY_NAMESPACE + b"canonical-inputs\0",
        tuple(row.record() for row in canonical),
    )
    annotations_sha = _namespaced_sha256(
        _ORDERED_REGISTRY_NAMESPACE + b"factor-annotations\0",
        tuple(row.record() for row in annotations),
    )
    vocabulary_sha = _namespaced_sha256(
        _RELAY_VOCABULARY_NAMESPACE,
        tuple(channel.record() for channel in RELAY_TARGET_VOCABULARY),
    )
    dependency_record = {
        "prompt_capsule_raw_sha256": manifest_v1.CAPSULE_RAW_SHA256,
        "prompt_capsule_payload_sha256": manifest_v1.CAPSULE_PAYLOAD_SHA256,
        "prompt_input_projection_sha256": manifest_v1.INPUT_PROJECTION_SHA256,
        "frozen_semantic_annotation_sha256": (manifest_v1.EXPECTED_LESSON_MANIFEST_SHA256),
        "factor_registry_sha256": manifest_v1.FACTOR_REGISTRY_SHA256,
        "factor_workspace_protocol_scope": factor_v2.PROTOCOL.scope,
        "factor_workspace_protocol_status": factor_v2.PROTOCOL.status,
        "factor_workspace_protocol_sha256": factor_v2.PROTOCOL.protocol_sha256,
        "factor_workspace_canonical_inventory_sha256": (
            factor_v2.PROTOCOL.canonical_inventory_sha256
        ),
        "factor_workspace_canonical_split_sha256": (factor_v2.PROTOCOL.canonical_split_sha256),
        "factor_workspace_learner_pass_assumed": False,
        "dependency_access_scope": (
            "public-TRAIN prompt/signature, frozen semantic annotations, and "
            "data-only factor-workspace protocol"
        ),
    }
    dependency_sha = _namespaced_sha256(_DEPENDENCY_NAMESPACE, dependency_record)
    split_record = {
        "factor_workspace_protocol_sha256": factor_v2.PROTOCOL.protocol_sha256,
        "factor_workspace_canonical_split_sha256": factor_v2.PROTOCOL.canonical_split_sha256,
        "fit_prompt_input_sha256s": factor_v2.PROTOCOL.canonical_fit_prompt_input_sha256s,
        "heldout_prompt_input_sha256s": (factor_v2.PROTOCOL.canonical_heldout_prompt_input_sha256s),
        "identity_not_appearance_split": True,
        "split_is_audit_only_and_not_causal": True,
    }
    split_sha = _namespaced_sha256(_SPLIT_NAMESPACE, split_record)
    surface_record = {
        "public_train_prompt_appearance_count": len(bindings),
        "canonical_semantic_identity_count": len(canonical),
        "canonical_fit_identity_count": CANONICAL_FIT_IDENTITY_COUNT,
        "canonical_heldout_identity_count": CANONICAL_HELDOUT_IDENTITY_COUNT,
        "full_typed_signature_binding_count": len(bindings),
        "raw_prompt_text_retained_count": 0,
        "dev_input_count": DEV_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "desired_output_read_count": DESIRED_OUTPUT_READ_COUNT,
        "program_artifact_read_count": PROGRAM_ARTIFACT_READ_COUNT,
        "property_row_read_count": PROPERTY_ROW_READ_COUNT,
        "task_identifier_read_count": TASK_IDENTIFIER_READ_COUNT,
        "family_identifier_read_count": FAMILY_IDENTIFIER_READ_COUNT,
        "entrypoint_spelling_neural_evidence_count": (ENTRYPOINT_SPELLING_NEURAL_EVIDENCE_COUNT),
        "audit_hash_neural_evidence_count": AUDIT_HASH_NEURAL_EVIDENCE_COUNT,
        "exact_relay_target_set_count": 0,
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "candidate_count": CANDIDATE_COUNT,
        "proof_count": PROOF_COUNT,
        "learner_or_selection_interface_present": False,
    }
    surface_sha = _namespaced_sha256(_SURFACE_NAMESPACE, surface_record)
    root_without_protocol = {
        "scope": PROTOCOL_SCOPE,
        "status": STOP_EXACT_STRUCTURAL_RELAY_TARGETS_UNAVAILABLE,
        "dependency_sha256": dependency_sha,
        "signature_bindings_sha256": bindings_sha,
        "canonical_inputs_sha256": canonical_sha,
        "factor_annotations_sha256": annotations_sha,
        "relay_target_vocabulary_sha256": vocabulary_sha,
        "annotation_schema_sha256": ANNOTATION_SCHEMA_SHA256,
        "identifiability_audit_sha256": audit.sha256,
        "canonical_split_sha256": split_sha,
        "surface_boundary_sha256": surface_sha,
        "public_train_prompt_appearance_count": len(bindings),
        "canonical_semantic_identity_count": len(canonical),
        "canonical_fit_identity_count": CANONICAL_FIT_IDENTITY_COUNT,
        "canonical_heldout_identity_count": CANONICAL_HELDOUT_IDENTITY_COUNT,
        "exact_relay_target_set_count": 0,
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "candidate_count": CANDIDATE_COUNT,
        "proof_count": PROOF_COUNT,
        "learner_state_constructed": False,
        "learning_executed": False,
        "selection_executed": False,
    }
    return FrozenPublicTrainPromptSelectionFreezeV1Protocol(
        scope=PROTOCOL_SCOPE,
        status=STOP_EXACT_STRUCTURAL_RELAY_TARGETS_UNAVAILABLE,
        signature_bindings=bindings,
        canonical_inputs=canonical,
        factor_annotations=annotations,
        relay_target_vocabulary=RELAY_TARGET_VOCABULARY,
        identifiability_audit=audit,
        dependency_sha256=dependency_sha,
        signature_bindings_sha256=bindings_sha,
        canonical_inputs_sha256=canonical_sha,
        factor_annotations_sha256=annotations_sha,
        relay_target_vocabulary_sha256=vocabulary_sha,
        annotation_schema_sha256=ANNOTATION_SCHEMA_SHA256,
        identifiability_audit_sha256=audit.sha256,
        canonical_split_sha256=split_sha,
        surface_boundary_sha256=surface_sha,
        protocol_sha256=_namespaced_sha256(_PROTOCOL_NAMESPACE, root_without_protocol),
    )


PROTOCOL = _build_protocol()

DEPENDENCY_RECORD = _freeze_record(
    {
        "prompt_capsule_raw_sha256": manifest_v1.CAPSULE_RAW_SHA256,
        "prompt_capsule_payload_sha256": manifest_v1.CAPSULE_PAYLOAD_SHA256,
        "prompt_input_projection_sha256": manifest_v1.INPUT_PROJECTION_SHA256,
        "frozen_semantic_annotation_sha256": manifest_v1.EXPECTED_LESSON_MANIFEST_SHA256,
        "factor_registry_sha256": manifest_v1.FACTOR_REGISTRY_SHA256,
        "factor_workspace_protocol_scope": factor_v2.PROTOCOL.scope,
        "factor_workspace_protocol_status": factor_v2.PROTOCOL.status,
        "factor_workspace_protocol_sha256": factor_v2.PROTOCOL.protocol_sha256,
        "factor_workspace_canonical_inventory_sha256": (
            factor_v2.PROTOCOL.canonical_inventory_sha256
        ),
        "factor_workspace_canonical_split_sha256": factor_v2.PROTOCOL.canonical_split_sha256,
        "factor_workspace_learner_pass_assumed": False,
        "dependency_access_scope": (
            "public-TRAIN prompt/signature, frozen semantic annotations, and data-only "
            "factor-workspace protocol"
        ),
    }
)
CANONICAL_SPLIT_RECORD = _freeze_record(
    {
        "factor_workspace_protocol_sha256": factor_v2.PROTOCOL.protocol_sha256,
        "factor_workspace_canonical_split_sha256": factor_v2.PROTOCOL.canonical_split_sha256,
        "fit_prompt_input_sha256s": factor_v2.PROTOCOL.canonical_fit_prompt_input_sha256s,
        "heldout_prompt_input_sha256s": (factor_v2.PROTOCOL.canonical_heldout_prompt_input_sha256s),
        "identity_not_appearance_split": True,
        "split_is_audit_only_and_not_causal": True,
    }
)
SURFACE_BOUNDARY_RECORD = _freeze_record(
    {
        "public_train_prompt_appearance_count": len(PROTOCOL.signature_bindings),
        "canonical_semantic_identity_count": len(PROTOCOL.canonical_inputs),
        "canonical_fit_identity_count": CANONICAL_FIT_IDENTITY_COUNT,
        "canonical_heldout_identity_count": CANONICAL_HELDOUT_IDENTITY_COUNT,
        "full_typed_signature_binding_count": len(PROTOCOL.signature_bindings),
        "raw_prompt_text_retained_count": 0,
        "dev_input_count": DEV_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "desired_output_read_count": DESIRED_OUTPUT_READ_COUNT,
        "program_artifact_read_count": PROGRAM_ARTIFACT_READ_COUNT,
        "property_row_read_count": PROPERTY_ROW_READ_COUNT,
        "task_identifier_read_count": TASK_IDENTIFIER_READ_COUNT,
        "family_identifier_read_count": FAMILY_IDENTIFIER_READ_COUNT,
        "entrypoint_spelling_neural_evidence_count": (ENTRYPOINT_SPELLING_NEURAL_EVIDENCE_COUNT),
        "audit_hash_neural_evidence_count": AUDIT_HASH_NEURAL_EVIDENCE_COUNT,
        "exact_relay_target_set_count": 0,
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "candidate_count": CANDIDATE_COUNT,
        "proof_count": PROOF_COUNT,
        "learner_or_selection_interface_present": False,
    }
)


__all__ = [
    "ANNOTATION_SCHEMA_RECORD",
    "ANNOTATION_SCHEMA_SHA256",
    "CANONICAL_FIT_IDENTITY_COUNT",
    "CANONICAL_HELDOUT_IDENTITY_COUNT",
    "CANONICAL_SEMANTIC_IDENTITY_COUNT",
    "CANONICAL_SPLIT_RECORD",
    "DEPENDENCY_RECORD",
    "PROTOCOL",
    "PROTOCOL_SCOPE",
    "PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT",
    "RELAY_TARGET_VOCABULARY",
    "STOP_EXACT_STRUCTURAL_RELAY_TARGETS_UNAVAILABLE",
    "SURFACE_BOUNDARY_RECORD",
    "CanonicalPromptSelectionInput",
    "FrozenFactorAnnotation",
    "FrozenPublicTrainPromptSelectionFreezeV1Protocol",
    "PhysicalCausalInputKey",
    "PrelearningIdentifiabilityAudit",
    "RelayTargetChannel",
    "SignatureBoundPromptAppearance",
    "SpanFactorAnnotation",
]
