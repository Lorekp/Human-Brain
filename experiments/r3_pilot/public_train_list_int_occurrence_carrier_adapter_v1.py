"""Target-free public List[Int] occurrence-carrier adapter V1.

This adapter audits the only currently authorized causal inputs: the frozen
public prompt-input capsule, the static prompt V4 checkpoint, and physical
prompt-by-factor attention.  It deliberately stops before constructing a V5
event because those authorities do not publish a stable node-occurrence join.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256

from experiments.r3_pilot import (
    canonical_train_prompt_factor_attention_v1 as attention,
)
from experiments.r3_pilot import (
    public_train_list_int_to_int_prompt_input_capsule_v1 as input_capsule,
)
from experiments.r3_pilot import (
    public_train_prompt_selection_v4_static_checkpoint_capsule_v1 as prompt_checkpoint,
)

SCHEMA = "public-train-list-int-occurrence-carrier-adapter-v1"
STOP_NO_AUTHENTICATED_JOIN = (
    "STOP_PUBLIC_TRAIN_LIST_INT_OCCURRENCE_CARRIER_ADAPTER_V1_"
    "NO_AUTHENTICATED_OCCURRENCE_JOIN"
)
STOP_REASON = (
    "actual prompt winners expose only edge-local, non-reused endpoint assemblies "
    "while prompt-by-factor attention exposes aggregate membership; no authority "
    "publishes a stable node-occurrence cue reused across structural roles, an "
    "occurrence-local raw-span or typed-literal source, and generic parent "
    "factor-slot occupancy"
)


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def _digest(label: str, value: object) -> str:
    payload = json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    digest = sha256()
    digest.update(SCHEMA.encode("ascii"))
    digest.update(b"\0")
    digest.update(label.encode("ascii"))
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PromptInputJoinAudit:
    """Read-only audit for one authenticated physical input root."""

    input_root_sha256: str
    multiplicity: int
    prompt_status: str
    prompt_input_sha256: str
    prompt_actual_winner_sha256: str
    prompt_certificate_sha256: str
    prompt_occurrence_count: int
    prompt_relay_count: int
    prompt_selected_cell_count: int
    prompt_unique_selected_cell_count: int
    edge_binding_count: int
    edge_endpoint_count: int
    reused_edge_endpoint_count: int
    non_edge_binding_count: int
    attention_status: str
    attention_membership_spike_count: int
    attention_scan_attempt_count: int
    attention_ready_observation_count: int
    attention_receipt_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PublicListIntOccurrenceCarrierAdapterState:
    """Immutable STOP result with exact physical-evidence accounting."""

    status: str
    stop_reason: str
    input_artifact_sha256: str
    input_content_root_sha256: str
    prompt_checkpoint_sha256: str
    prompt_artifact_sha256: str
    unique_input_count: int
    appearance_count: int
    prompt_emitting_input_count: int
    prompt_empty_input_count: int
    prompt_occurrence_count: int
    prompt_relay_count: int
    prompt_selected_cell_count: int
    edge_binding_count: int
    edge_endpoint_count: int
    reused_edge_endpoint_count: int
    non_edge_binding_count: int
    attention_ready_input_count: int
    attention_zero_input_count: int
    attention_prompt_stop_input_count: int
    attention_factor_scan_attempt_count: int
    attention_membership_spike_count: int
    stable_node_occurrence_join_count: int
    occurrence_local_source_join_count: int
    parent_factor_slot_occupancy_join_count: int
    conversion_candidate_count: int
    physical_occurrence_event_build_count: int
    v5_inference_attempt_count: int
    caller_certificate_count: int
    source_ordinal_read_count: int
    runtime_relay_parse_count: int
    host_selected_structure_count: int
    semantic_label_input_count: int
    evaluation_input_count: int
    input_audits: tuple[PromptInputJoinAudit, ...]
    sha256: str


def _winner_rows(
    certificate: prompt_checkpoint.PromptStructuralCertificate,
) -> tuple[prompt_checkpoint.PromptPhysicalWinner, ...]:
    return tuple(
        winner
        for occurrence in certificate.occurrences
        for winner in (occurrence.channel_winner, *occurrence.branch_winners)
    )


def _audit_input(
    frozen_input: input_capsule.FrozenPromptInput,
) -> PromptInputJoinAudit:
    spans, literals, signature_role = frozen_input.neural_records()
    prompt = prompt_checkpoint.infer_prompt(spans, literals, signature_role)
    handoff = attention.infer_prompt_factor_attention(
        spans,
        literals,
        signature_role,
    )
    attention_state = handoff.state
    if (
        attention_state.prompt_checkpoint_sha256 != prompt.checkpoint_sha256
        or attention_state.prompt_input_sha256 != prompt.input_sha256
        or attention_state.prompt_actual_winner_sha256
        != prompt.actual_winner_sha256
        or attention_state.prompt_certificate_sha256 != prompt.sha256
    ):
        raise RuntimeError("prompt and attention authorities observed different winners")
    if (
        attention_state.status == attention.ATTENTION_READY
        and attention.require_prompt_factor_attention(handoff) is not attention_state
    ):
        raise RuntimeError("READY attention handoff did not authenticate exactly")

    winners = _winner_rows(prompt)
    selected_cells = tuple(
        winner.selected_cell_id
        for winner in winners
        if winner.selected_cell_id is not None
    )
    edge_bindings = tuple(
        occurrence.occurrence_binding
        for occurrence in prompt.occurrences
        if occurrence.occurrence_binding is not None
    )
    edge_endpoints = tuple(
        endpoint
        for binding in edge_bindings
        for endpoint in (binding.parent_cells, binding.child_cells)
    )
    endpoint_counts = Counter(edge_endpoints)
    reused_endpoints = sum(count - 1 for count in endpoint_counts.values() if count > 1)
    # The static capsule publishes bindings only through its physical edge
    # contract.  It publishes no corresponding source binding for any support
    # occurrence; do not recover that distinction from relay text.
    non_edge_bindings = 0
    if reused_endpoints:
        raise RuntimeError(
            "public prompt winner geometry changed; freeze a fresh adapter decision"
        )

    record = {
        "input_root_sha256": frozen_input.input_root,
        "multiplicity": frozen_input.multiplicity,
        "prompt_status": prompt.status,
        "prompt_input_sha256": prompt.input_sha256,
        "prompt_actual_winner_sha256": prompt.actual_winner_sha256,
        "prompt_certificate_sha256": prompt.sha256,
        "prompt_occurrence_count": len(prompt.occurrences),
        "prompt_relay_count": len(selected_cells),
        "prompt_selected_cell_count": len(selected_cells),
        "prompt_unique_selected_cell_count": len(set(selected_cells)),
        "edge_binding_count": len(edge_bindings),
        "edge_endpoint_count": len(edge_endpoints),
        "reused_edge_endpoint_count": reused_endpoints,
        "non_edge_binding_count": non_edge_bindings,
        "attention_status": attention_state.status,
        "attention_membership_spike_count": sum(
            attention_state.physical_coincidence_spikes
        ),
        "attention_scan_attempt_count": attention_state.scan.factor_inference_attempts,
        "attention_ready_observation_count": attention_state.scan.ready_observation_count,
        "attention_receipt_sha256": handoff.sha256,
    }
    return PromptInputJoinAudit(
        **record,
        sha256=_digest("prompt-input-join-audit", record),
    )


def run_public_list_int_occurrence_carrier_adapter_v1(
) -> PublicListIntOccurrenceCarrierAdapterState:
    """Audit all frozen unique inputs and STOP before any forged V5 event."""

    capsule = input_capsule.load_input_capsule()
    audits = tuple(
        sorted(
            (_audit_input(row) for row in capsule.iter_inputs()),
            key=lambda row: row.input_root_sha256,
        )
    )
    attention_statuses = Counter(row.attention_status for row in audits)
    admitted_attention_statuses = {
        attention.ATTENTION_READY,
        attention.STOP_ZERO,
        attention.STOP_PROMPT,
    }
    if set(attention_statuses) - admitted_attention_statuses:
        raise RuntimeError("public attention surface produced a new unadjudicated state")

    partial = PublicListIntOccurrenceCarrierAdapterState(
        status=STOP_NO_AUTHENTICATED_JOIN,
        stop_reason=STOP_REASON,
        input_artifact_sha256=capsule.artifact_sha256,
        input_content_root_sha256=capsule.content_root,
        prompt_checkpoint_sha256=prompt_checkpoint.BANKED_CHECKPOINT_SHA256,
        prompt_artifact_sha256=prompt_checkpoint.EXPECTED_ARTIFACT_SHA256,
        unique_input_count=len(audits),
        appearance_count=capsule.appearance_count,
        prompt_emitting_input_count=sum(row.prompt_occurrence_count > 0 for row in audits),
        prompt_empty_input_count=sum(row.prompt_occurrence_count == 0 for row in audits),
        prompt_occurrence_count=sum(row.prompt_occurrence_count for row in audits),
        prompt_relay_count=sum(row.prompt_relay_count for row in audits),
        prompt_selected_cell_count=sum(row.prompt_selected_cell_count for row in audits),
        edge_binding_count=sum(row.edge_binding_count for row in audits),
        edge_endpoint_count=sum(row.edge_endpoint_count for row in audits),
        reused_edge_endpoint_count=sum(row.reused_edge_endpoint_count for row in audits),
        non_edge_binding_count=sum(row.non_edge_binding_count for row in audits),
        attention_ready_input_count=attention_statuses[attention.ATTENTION_READY],
        attention_zero_input_count=attention_statuses[attention.STOP_ZERO],
        attention_prompt_stop_input_count=attention_statuses[attention.STOP_PROMPT],
        attention_factor_scan_attempt_count=sum(
            row.attention_scan_attempt_count for row in audits
        ),
        attention_membership_spike_count=sum(
            row.attention_membership_spike_count for row in audits
        ),
        stable_node_occurrence_join_count=0,
        occurrence_local_source_join_count=0,
        parent_factor_slot_occupancy_join_count=0,
        conversion_candidate_count=0,
        physical_occurrence_event_build_count=0,
        v5_inference_attempt_count=0,
        caller_certificate_count=0,
        source_ordinal_read_count=0,
        runtime_relay_parse_count=0,
        host_selected_structure_count=0,
        semantic_label_input_count=0,
        evaluation_input_count=0,
        input_audits=audits,
        sha256="",
    )
    payload = {
        field: getattr(partial, field)
        for field in partial.__dataclass_fields__
        if field != "sha256"
    }
    return PublicListIntOccurrenceCarrierAdapterState(
        **payload,
        sha256=_digest("adapter-stop-state", payload),
    )


__all__ = [
    "SCHEMA",
    "STOP_NO_AUTHENTICATED_JOIN",
    "STOP_REASON",
    "PromptInputJoinAudit",
    "PublicListIntOccurrenceCarrierAdapterState",
    "run_public_list_int_occurrence_carrier_adapter_v1",
]
